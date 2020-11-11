/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include "pch.h"

#include <mi/base/handle.h>
#include <mi/mdl/mdl_code_generators.h>

#include <mdl/compiler/compilercore/compilercore_cc_conf.h>
#include <mdl/compiler/compilercore/compilercore_streams.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_visitor.h>
#include <mdl/compiler/compilercore/compilercore_hash.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include <mdl/compiler/stdmodule/enums.h>

#include <mdl/codegenerators/generator_code/generator_code.h>

#include <cstring>

#include "generator_dag_generated_dag.h"
#include "generator_dag_walker.h"
#include "generator_dag_tools.h"
#include "generator_dag_code_printer.h"
#include "generator_dag_dep_graph.h"
#include "generator_dag_serializer.h"
#include "generator_dag_dumper.h"
#include "generator_dag_builder.h"

namespace mi {
namespace mdl {

typedef Store<bool> Flag_store;

namespace {

/// Translate function properties.
unsigned char get_function_properties(IDefinition const *f_def)
{
    unsigned char func_properties = 0;

    // get properties
    if (f_def->get_property(IDefinition::DP_ALLOW_INLINE))
        func_properties |= 1 << IGenerated_code_dag::FP_ALLOW_INLINE;
    if (f_def->get_property(IDefinition::DP_USES_TEXTURES))
        func_properties |= 1 << IGenerated_code_dag::FP_USES_TEXTURES;
    if (f_def->get_property(IDefinition::DP_CAN_THROW_BOUNDS))
        func_properties |= 1 << IGenerated_code_dag::FP_CAN_THROW_BOUNDS;
    if (f_def->get_property(IDefinition::DP_CAN_THROW_DIVZERO))
        func_properties |= 1 << IGenerated_code_dag::FP_CAN_THROW_DIVZERO;
    if (!f_def->get_property(IDefinition::DP_IS_VARYING))
        func_properties |= 1 << IGenerated_code_dag::FP_IS_UNIFORM;
    if (f_def->get_property(IDefinition::DP_IS_NATIVE))
        func_properties |= 1 << IGenerated_code_dag::FP_IS_NATIVE;

    return func_properties;
}

///
/// Helper class for parameter type binding.
///
class Type_binder {
    typedef ptr_hash_map<IType_array const, IType_array const *>::Type Bind_type_map;
    typedef ptr_hash_map<IType_array_size const, int>::Type            Bind_size_map;

public:
    /// Constructor.
    Type_binder(IAllocator *alloc, Type_factory &tf)
    : m_tf(tf)
    , m_type_bindings(0, Bind_type_map::hasher(), Bind_type_map::key_equal(), alloc)
    , m_size_bindings(0, Bind_size_map::hasher(), Bind_size_map::key_equal(), alloc)
    {
    }

    /// Bind the given abstract type to a concrete type.
    ///
    /// \param abs_type  an deferred sized array type
    /// \param type      an immediate sized array type that is bound to abs_type
    void bind_param_type(IType_array const *abs_type, IType_array const *type)
    {
        MDL_ASSERT(
            !abs_type->is_immediate_sized() && type->is_immediate_sized() && "Wrong type binding");
        IType_array_size const *abs_size = abs_type->get_deferred_size();
        int size                         = type->get_size();

        m_size_bindings[abs_size] = size;

        // bind the size NOT the element type
        IType const *e_type = abs_type->get_element_type();
        IType const *n_type = m_tf.create_array(e_type, type->get_size());
        m_type_bindings[abs_type] = cast<IType_array>(n_type);
    }

    /// Return the bound type for an array type.
    ///
    /// \param a_type  an array (parameter) type
    ///
    /// \return a_type or an immediate sized array type if a_type was bound
    IType_array const *get_bound_type(IType_array const *a_type)
    {
        Bind_type_map::const_iterator it = m_type_bindings.find(a_type);
        if (it != m_type_bindings.end())
            return it->second;

        // check if the size is bound
        if (!a_type->is_immediate_sized()) {
            IType_array_size const *abs_size = a_type->get_deferred_size();

            Bind_size_map::const_iterator sit = m_size_bindings.find(abs_size);
            if (sit != m_size_bindings.end()) {
                size_t size = sit->second;

                IType const       *e_type = a_type->get_element_type();
                IType_array const *r_type = cast<IType_array>(m_tf.create_array(e_type, size));

                m_type_bindings[a_type] = r_type;
                return r_type;
            }
        }

        return a_type;
    }

private:
    /// Type type factory for creating new array types.
    Type_factory &m_tf;

    /// Type bindings for overload resolution.
    Bind_type_map m_type_bindings;
    Bind_size_map m_size_bindings;
};

///
/// Helper class to dump an material (class) DAG as a dot file.
///
class Material_dumper : public DAG_dumper {
    typedef DAG_dumper Base;
public:
    /// Constructor.
    ///
    /// \param alloc      the allocator
    /// \param dag        the code dag
    /// \param mat_index  the material index of the material to dump
    /// \param out        an output stream, the dot file is written to
    explicit Material_dumper(
        IAllocator               *alloc,
        Generated_code_dag const &dag,
        int                      mat_index,
        IOutput_stream           *out);

    /// Dump the material expression DAG to the output stream.
    ///
    /// \param argc               number of arguments of the dag
    /// \param argv               the arguments
    void dump(
        size_t         argc,
        DAG_node const *argv[]);

    /// Dump the material instance expression DAG to the output stream.
    ///
    /// \param dag                the code dag
    /// \param mat_index          the material index of the material to dump
    void dump(Generated_code_dag::Material_instance &instance);

    /// Get the parameter name for the given index if any.
    ///
    /// \param index  the index of the parameter
    char const *get_parameter_name(int index) MDL_FINAL;

private:
    /// The DAG that is dumped.
    Generated_code_dag const &m_dag;

    /// The material index that is dumped;
    int const                m_mat_index;
};

// Constructor.
Material_dumper::Material_dumper(
    IAllocator               *alloc,
    Generated_code_dag const &dag,
    int                      mat_index,
    IOutput_stream           *out)
: Base(alloc, out)
, m_dag(dag)
, m_mat_index(mat_index)
{
}

// Dump the material expression DAG to the output stream.
void Material_dumper::dump(
    size_t         argc,
    DAG_node const *argv[])
{
    m_printer->print("digraph \"");
    char const *name = m_dag.get_material_name(m_mat_index);
    if (name == NULL || name[0] == '\0')
        name = "DAG";
    m_printer->print(name);
    m_printer->print("\" {\n");
    m_walker.walk_material(const_cast<Generated_code_dag *>(&m_dag), m_mat_index, this);

    for (size_t i = 0; i < argc; ++i) {
        DAG_node *arg = const_cast<DAG_node *>(argv[i]);

        m_walker.walk_node(arg, this);

        char const *arg_name = m_dag.get_material_parameter_name(m_mat_index, i);
        argument(arg_name, i, NULL);
        edge('a', i, arg, "value", NULL);
    }

    m_printer->print("}\n");
}

// Get the parameter name for the given index if any.
const char *Material_dumper::get_parameter_name(int index)
{
    return m_dag.get_material_parameter_name(m_mat_index, index);
}

///
/// Helper class to dump an material instance DAG as a dot file.
///
class Instance_dumper : public DAG_dumper {
    typedef DAG_dumper Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param inst   the instance
    /// \param out    an output stream, the dot file is written to
    explicit Instance_dumper(
        IAllocator                                  *alloc,
        Generated_code_dag::Material_instance const &inst,
        IOutput_stream                              *out);

    /// Dump the material instance expression DAG to the output stream.
    void dump();

    /// Get the parameter name for the given index if any.
    ///
    /// \param index  the index of the parameter
    char const *get_parameter_name(int index) MDL_FINAL;

private:
    /// The DAG that is dumped.
    Generated_code_dag::Material_instance const &m_inst;
};

// Constructor.
Instance_dumper::Instance_dumper(
    IAllocator                                  *alloc,
    Generated_code_dag::Material_instance const &inst,
    IOutput_stream                              *out)
: Base(alloc, out)
, m_inst(inst)
{
}

// Dump the material instance expression DAG to the output stream.
void Instance_dumper::dump()
{
    m_printer->print("digraph \"instance_DAG\" {\n");
    m_walker.walk_instance(const_cast<Generated_code_dag::Material_instance *>(&m_inst), this);
    m_printer->print("}\n");
}

// Get the parameter name for the given index if any.
const char *Instance_dumper::get_parameter_name(int index)
{
    return m_inst.get_parameter_name(index);
}

typedef ptr_hash_map<DAG_node const, size_t>::Type Phen_out_map;

/// A helper class computing the phen-out for every expression that is visited.
class Calc_phen_out : public IDAG_ir_visitor
{
public:
    /// Constructor.
    ///
    /// \param phen_outs  the phen-out map will be filled
    Calc_phen_out(Phen_out_map &phen_outs)
    : m_phen_outs(phen_outs)
    {
    }

    // Post-visit a Constant.
    void visit(DAG_constant *cnst) MDL_FINAL {}

    // Post-visit a variable (temporary).
    void visit(DAG_temporary *tmp) MDL_FINAL {
        MDL_ASSERT(!"There should be no temporaries at this point");
    }

    // Post-visit a call.
    void visit(DAG_call *call) MDL_FINAL
    {
        for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
            DAG_node const *arg = call->get_argument(i);

            Phen_out_map::iterator it = m_phen_outs.find(arg);
            if (it == m_phen_outs.end()) {
                m_phen_outs[arg] = 1;
            } else {
                it->second += 1;
            }
        }
    }

    // Post-visit a Parameter.
    void visit(DAG_parameter *param) MDL_FINAL {}

    // Post-visit a Temporary.
    void visit(int index, DAG_node *init) MDL_FINAL {}

private:

    /// The phen-out count;
    Phen_out_map &m_phen_outs;
};

/// Helper class: creates temporaries for node when phen-out > 1.
class Abstract_temporary_inserter : public IDAG_ir_visitor
{
    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Temporary_map;

public:
    typedef ptr_hash_map<DAG_node const, char const *>::Type Temporary_name_map;

    /// Constructor.
    ///
    /// \param alloc               the allocator for temporary memory
    /// \param expression_factory  the expression factory to create temporaries on
    /// \param phen_outs           the phen-out map for the visited expression DAG
    /// \param temp_name_map       the desired temporary names
    Abstract_temporary_inserter(
        IAllocator               *alloc,
        DAG_node_factory_impl    &expression_factory,
        Phen_out_map const       &phen_outs,
        Temporary_name_map const &temp_name_map)
    : m_node_factory(expression_factory)
    , m_phen_outs(phen_outs)
    , m_process_constants(false)
    , m_process_parameters(true)
    , m_temp_map(0, Temporary_map::hasher(), Temporary_map::key_equal(), alloc)
    , m_temp_name_map(temp_name_map)
    {
    }

    // Post-visit a Constant.
    void visit(DAG_constant *cnst) MDL_FINAL {}

    // Post-visit a variable (temporary).
    void visit(DAG_temporary *tmp) MDL_FINAL {
        MDL_ASSERT(!"There should be no temporaries at this point");
    }

    // Post-visit a call.
    void visit(DAG_call *call) MDL_FINAL
    {
        for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
            DAG_node const *arg = call->get_argument(i);

            Temporary_name_map::const_iterator it_name  = m_temp_name_map.find(arg);
            bool has_name = it_name != m_temp_name_map.end();
            char const *name = has_name ? it_name->second : "";

            switch (arg->get_kind()) {
            case DAG_node::EK_CONSTANT:
                if (!m_process_constants && !has_name)
                    continue;
                break;
            case DAG_node::EK_PARAMETER:
                if (!m_process_parameters && !has_name)
                    continue;
                break;
            default:
                break;
            }

            Phen_out_map::const_iterator it = m_phen_outs.find(arg);
            if (it == m_phen_outs.end()) {
                MDL_ASSERT(!"unknown expression occured");
            } else {
                if ((it->second > 1) || has_name) {
                    // multiple use or name found, replace
                    DAG_node const *temp = create_temporary(arg, name);

                    call->set_argument(i, temp);
                }
            }
        }
    }

    // Post-visit a Parameter.
    void visit(DAG_parameter *param) MDL_FINAL {}

    // Post-visit a Temporary.
    void visit(int index, DAG_node *init) MDL_FINAL {}

    // Create a temporary.
    DAG_node const *create_temporary(DAG_node const *node, char const *name)
    {
        Temporary_map::iterator it = m_temp_map.find(node);
        if (it != m_temp_map.end()) {
            return it->second;
        }
        int index = add_temporary(node, name);
        DAG_node const *temp = m_node_factory.create_temporary(node, index);

        m_temp_map[node] = temp;
        return temp;
    }

    /// Create and register a new temporary.
    ///
    /// \param node  the initializer for the temporary
    /// \param name  the name for the temporary
    virtual int add_temporary(DAG_node const *node, char const *name) = 0;

private:
    /// The expression factory to create temporaries on.
    DAG_node_factory_impl &m_node_factory;

    /// The phen-out count;
    Phen_out_map const &m_phen_outs;

    /// If true, constants will be placed into temporaries.
    bool m_process_constants;

    /// If true, parameters will be placed into temporaries.
    bool m_process_parameters;

    /// Map of created temporaries.
    Temporary_map m_temp_map;

    /// Map of desired temporary names.
    const Temporary_name_map &m_temp_name_map;
};

/// Helper class: visit a DAG and collect all parameter
class Parameter_collector {
    typedef ptr_hash_set<DAG_node const>::Type Visited_node_set;
public:

    /// Constructor.
    ///
    /// \param alloc          an allocator for temporary memory
    /// \param fold_ternary   if true, arguments whose data flows into the condition of ternary
    ///                       operators will be inlined
    /// \param params         array that will be filled with the found parameters,
    ///                       indexed by its index
    /// \param inline_params  array that will be filled with parameters that must be inlined,
    ///                       indexed by its index
    explicit Parameter_collector(
        IAllocator    *alloc,
        bool          fold_ternary,
        DAG_parameter *params[],
        DAG_parameter *inline_params[])
    : m_params(params)
    , m_inline_params(inline_params)
    , m_need_inline(false)
    , m_marker(0, Visited_node_set::hasher(), Visited_node_set::key_equal(), alloc)
    , m_arg_marker(0, Visited_node_set::hasher(), Visited_node_set::key_equal(), alloc)
    , m_fold_ternary(fold_ternary)
    {
    }

    /// Walk a DAG IR.
    ///
    /// \param root       the DAG root node that will be visited
    void collect(DAG_node const *root)
    {
        do_walk_node(root);
    }

    /// Check if parameter inlining is necessary.
    bool need_parameter_inline() const { return m_need_inline; }

private:
    /// Walk a DAG IR node.
    ///
    /// \param node  the DAG root node to traverse
    void do_walk_node(DAG_node const *node)
    {
        if (m_marker.find(node) != m_marker.end())
            return;
        m_marker.insert(node);

        switch (node->get_kind()) {
        case DAG_node::EK_CONSTANT:
            return;
        case DAG_node::EK_TEMPORARY:
            MDL_ASSERT(!"There should be no temporaries at this point");
            return;
        case DAG_node::EK_CALL:
            {
                DAG_call const *c = cast<DAG_call>(node);
                int start_idx = 0;

                if (m_fold_ternary &&
                    c->get_semantic() == operator_to_semantic(IExpression::OK_TERNARY)) {
                    // we found an unfolded ternary operator, bad
                    m_need_inline = true;
                    start_idx      = 1;

                    DAG_node const *cond = c->get_argument(0);

                    do_walk_node_arg(cond);
                }

                for (int i = start_idx, n = c->get_argument_count(); i < n; ++i) {
                    DAG_node const *arg = c->get_argument(i);

                    do_walk_node(arg);
                }
                return;
            }
        case DAG_node::EK_PARAMETER:
            {
                DAG_parameter const *p = cast<DAG_parameter>(node);
                m_params[p->get_index()] = const_cast<DAG_parameter *>(p);
                return;
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
    }

    /// Walk a DAG IR node in argument inline mode.
    ///
    /// \param node  the DAG root node to traverse
    void do_walk_node_arg(DAG_node const *node)
    {
        if (m_arg_marker.find(node) != m_arg_marker.end())
            return;
        m_arg_marker.insert(node);

        switch (node->get_kind()) {
        case DAG_node::EK_CONSTANT:
            return;
        case DAG_node::EK_TEMPORARY:
            MDL_ASSERT(!"There should be no temporaries at this point");
            return;
        case DAG_node::EK_CALL:
            {
                DAG_call const *c = cast<DAG_call>(node);

                for (int i = 0, n = c->get_argument_count(); i < n; ++i) {
                    DAG_node const *arg = c->get_argument(i);

                    do_walk_node_arg(arg);
                }
                return;
            }
        case DAG_node::EK_PARAMETER:
            {
                DAG_parameter const *p = cast<DAG_parameter>(node);
                m_inline_params[p->get_index()] = const_cast<DAG_parameter *>(p);
                return;
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
    }

private:
    /// The parameter array.
    DAG_parameter **m_params;

    /// The parameter array.
    DAG_parameter **m_inline_params;

    /// If true, found parameters must be inlined.
    bool m_need_inline;

    /// The marker set for visited expressions.
    Visited_node_set m_marker;

    /// The marker set for visited expressions in arg-search mode.
    Visited_node_set m_arg_marker;

    /// If true, inline arguments whose data flows into the condition of the ?: operator.
    bool m_fold_ternary;
};

/// Helper class: visit a DAG and collect all parameter
class Parameter_inliner {
    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Visited_node_map;
public:

    /// Constructor.
    ///
    /// \param alloc          an allocator for temporary memory
    /// \param factory        the expression factory of the DAG to modify
    /// \param inline_params  array that will be filled with parameters that must be inlined,
    ///                       indexed by its index
    /// \param param_values   the values of all parameters
    explicit Parameter_inliner(
        IAllocator                            *alloc,
        IGenerated_code_dag::DAG_node_factory &factory,
        DAG_parameter                         *inline_params[],
        IValue const                          *param_values[])
    : m_alloc(alloc)
    , m_factory(factory)
    , m_inline_params(inline_params)
    , m_param_values(param_values)
    , m_marker(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), alloc)
    {
    }

    /// Inline the given parameters.
    ///
    /// \param node       the DAG node that will be visited
    DAG_node const *inline_parameters(DAG_node const *node)
    {
        return do_walk_node(node);
    }

private:
    /// Walk a DAG.
    ///
    /// \param node  the DAG root node to traverse
    DAG_node const *do_walk_node(DAG_node const *node)
    {
        Visited_node_map::const_iterator it = m_marker.find(node);
        if (it != m_marker.end())
            return it->second;

        DAG_node const *res = node;

        switch (node->get_kind()) {
        case DAG_node::EK_CONSTANT:
             // do nothing, leave the constant as is, we are working on the same DAG
            break;
        case DAG_node::EK_TEMPORARY:
            MDL_ASSERT(!"There should be no temporaries at this point");
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *c = cast<DAG_call>(node);
                int n_args = c->get_argument_count();

                VLA<DAG_call::Call_argument> call_args(m_alloc, n_args);

                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = c->get_argument(i);

                    call_args[i].arg        = do_walk_node(arg);
                    call_args[i].param_name = c->get_parameter_name(i);
                }

                // optimization happens here ...
                res = m_factory.create_call(
                    c->get_name(),
                    c->get_semantic(),
                    call_args.data(),
                    n_args,
                    c->get_type());
            }
            break;
        case DAG_node::EK_PARAMETER:
            {
                DAG_parameter const *p = cast<DAG_parameter>(node);
                int idx = p->get_index();

                if (m_inline_params[idx] != NULL) {
                    IValue const *v = m_param_values[idx];
                    res = m_factory.create_constant(v);
                } else {
                    // let the parameter as is, so we can modify it in renumber!
                }
            }
            break;
        default:
            MDL_ASSERT(!"Unsupported DAG node kind");
            break;
        }
        m_marker[node] = res;
        return res;
    }


private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The expression factory to create new expressions.
    IGenerated_code_dag::DAG_node_factory &m_factory;

    /// The parameter array.
    DAG_parameter **m_inline_params;

    /// The Values of the parameters.
    IValue const **m_param_values;

    /// The marker set for visited expressions.
    Visited_node_map m_marker;
};

/// Helper class to handle enable_if dependencies.
class Condition_compute MDL_FINAL : public IDAG_ir_visitor {
public:
    typedef set<size_t>::Type Dependency_set;

    /// Process a parameter.
    ///
    /// \param param  the parameter
    /// \param index  the index of the parameter
    void process_parameter(Generated_code_dag::Parameter_info &param, size_t index)
    {
        if (DAG_node const *cond = param.get_enable_if_condition()) {
            m_dependencies.clear();

            m_walker.walk_node(const_cast<DAG_node *>(cond), this);

            for (Dependency_set::const_iterator
                    it(m_dependencies.begin()), end(m_dependencies.end());
                 it != end;
                 ++it)
            {
                size_t ctrl_index = *it;

                // the parameter with index ctrl_index controls the enable condition
                // of this parameter
                m_controls[ctrl_index].insert(index);
            }
        }
    }

    /// Get the controls for a given parameter index.
    Dependency_set const &get_controls(size_t index) const {
        return m_controls[index];
    }

private:
    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL {}

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL {}

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL {}

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL
    {
        m_dependencies.insert(param->get_index());
    }

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL {}

public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param n_params  number of parameters
    Condition_compute(IAllocator *alloc, size_t n_params)
    : m_walker(alloc, /*as_tree=*/false)
    , m_dependencies(Dependency_set::key_compare(), alloc)
    , m_controls(alloc)
    {
        m_controls.resize(n_params, Dependency_set(Dependency_set::key_compare(), alloc));
    }

private:
    /// The used IR walker.
    DAG_ir_walker m_walker;

    /// Temporary store the dependencies of one parameter.
    Dependency_set m_dependencies;

    /// The controls.
    vector<Dependency_set>::Type m_controls;
};

}  // anonymous


// Get a tag,for a resource constant that might be reachable from this DAG.
int Resource_tagger::get_resource_tag(
    IValue_resource const *res) const
{
    int tag = res->get_tag_value();
    if (tag != 0)
        return tag;

    Resource_tag_tuple::Kind kind = kind_from_value(res);

    // for now, linear search
    char const *url = res->get_string_value();
    for (size_t i = 0, n = m_resource_tag_map.size(); i < n; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        if (e.m_kind == kind && strcmp(e.m_url, url) == 0)
            return e.m_tag;
    }
    return 0;
}

// Constructor.
Generated_code_dag::Generated_code_dag(
    IAllocator      *alloc,
    MDL             *compiler,
    IModule const   *module,
    char const      *internal_space,
    Compile_options options,
    char const      *renderer_context_name)
: Base(alloc)
, m_arena(alloc)
, m_module_name(string(module != NULL ? module->get_name() : "", alloc))
, m_module_file_name(string(module != NULL ? module->get_filename() : "", alloc))
, m_mangler(alloc, compiler)
, m_printer(m_mangler.get_printer())
, m_sym_tab(m_arena)
, m_type_factory(m_arena, compiler, &m_sym_tab)
, m_value_factory(m_arena, m_type_factory)
, m_messages(alloc, module != NULL ? impl_cast<Module>(module)->get_msg_name() : "")
, m_module_imports(alloc)
, m_invisible_sym(m_sym_tab.create_user_type_symbol(""))
, m_builder(alloc)
, m_mdl(mi::base::make_handle_dup(compiler))
, m_node_factory(compiler, m_arena, m_value_factory, internal_space)
, m_module_annotations(alloc)
, m_functions(alloc)
, m_materials(alloc)
, m_annotations(alloc)
, m_user_types(alloc)
, m_user_constants(alloc)
, m_internal_space(internal_space, alloc)
, m_renderer_context_name(renderer_context_name, alloc)
, m_options(options)
, m_current_material_index(0)
, m_current_function_index(0)
, m_needs_anno(false)
, m_mark_generated((options & MARK_GENERATED_ENTITIES) != 0)
, m_resource_tag_map(alloc)
, m_resource_tagger(m_resource_tag_map)
{
    m_node_factory.enable_unsafe_math_opt((options & UNSAFE_MATH_OPTIMIZATIONS) != 0);
    m_node_factory.enable_expose_names_of_let_expressions((options & EXPOSE_NAMES_OF_LET_EXPRESSIONS) != 0);

    if (module != NULL) {
        int n = module->get_import_count();
        m_module_imports.reserve(n);

        for (int i = 0; i < n; ++i) {
            mi::base::Handle<IModule const> import(module->get_import(i));

            m_module_imports.push_back(string(import->get_name(), alloc));
        }
    }
}

// Get the material info for a given material index or NULL if the index is out of range.
Generated_code_dag::Material_info *Generated_code_dag::get_material_info(
    size_t material_index)
{
    if (material_index < m_materials.size()) {
        return &m_materials[material_index];
    }
    return NULL;
}

// Get the material info for a given material index or NULL if the index is out of range.
Generated_code_dag::Material_info const *Generated_code_dag::get_material_info(
    size_t material_index) const
{
    if (material_index < m_materials.size()) {
        return &m_materials[material_index];
    }
    return NULL;
}

// Get the parameter info for a given material and parameter index pair or NULL if
// one index is out of range.
Generated_code_dag::Parameter_info const *Generated_code_dag::get_mat_param_info(
    size_t material_index,
    size_t parameter_index) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        if (parameter_index < mat->get_parameter_count()) {
            return &mat->get_parameter(parameter_index);
        }
    }
    return NULL;
}

// Get the function info for a given function index or NULL if the index is out of range.
Generated_code_dag::Function_info const *Generated_code_dag::get_function_info(
    size_t function_index) const
{
    if (function_index < m_functions.size()) {
        return &m_functions[function_index];
    }
    return NULL;
}

// Get the parameter info for a given function and parameter index pair or NULL if
// one index is out of range.
Generated_code_dag::Parameter_info const *Generated_code_dag::get_func_param_info(
    size_t function_index,
    size_t parameter_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        if (parameter_index < func->get_parameter_count()) {
            return &func->get_parameter(parameter_index);
        }
    }
    return NULL;
}

// Get the annotation info for a given annotation index or NULL if the index is out of range.
Generated_code_dag::Annotation_info const *Generated_code_dag::get_annotation_info(
    size_t annotation_index) const
{
    if (annotation_index < m_annotations.size()) {
        return &m_annotations[annotation_index];
    }
    return NULL;
}

// Get the parameter info for a given annotation and parameter index pair or NULL if
// one index is out of range.
Generated_code_dag::Parameter_info const *Generated_code_dag::get_anno_param_info(
    size_t annotation_index,
    size_t parameter_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        if (parameter_index < anno->get_parameter_count()) {
            return &anno->get_parameter(parameter_index);
        }
    }
    return NULL;
}

// Get the user type info for a given type index or NULL if the index is out of range.
Generated_code_dag::User_type_info const *Generated_code_dag::get_type_info(
    size_t type_index) const
{
    if (type_index < m_user_types.size()) {
        return &m_user_types[type_index];
    }
    return NULL;
}

// Get the user constant info for a given constant index or NULL if the index is out of range.
Generated_code_dag::Constant_info const *Generated_code_dag::get_constant_info(
    size_t constant_index) const
{
    if (constant_index < m_user_constants.size()) {
        return &m_user_constants[constant_index];
    }
    return NULL;
}

// Add an import if not already there.
void Generated_code_dag::add_import(char const *mod_name)
{
    string new_mod = string(mod_name, get_allocator());

    for (size_t i = 0, n = m_module_imports.size(); i < n; ++i) {
        if (m_module_imports[i] == new_mod) {
            // already there
            return;
        }
    }
    m_module_imports.push_back(new_mod);
}

// Generate annotations for functions (only for the function itself).
void Generated_code_dag::gen_function_annotations(
    DAG_builder        &dag_builder,
    Function_info      &func,
    IDeclaration const *decl)
{
    IAnnotation_block const *annotations = NULL;

    // handle function annotations: Note that constructors are compiler generated but may
    // have a declaration: the struct elementary constructor point to the struct declaration,
    // the enum copy (and default) constructor to the enum declaration
    switch (decl->get_kind()) {
        case IDeclaration::DK_TYPE_STRUCT:
            annotations = cast<IDeclaration_type_struct>(decl)->get_annotations();
            break;
        case IDeclaration::DK_TYPE_ENUM:
            annotations = cast<IDeclaration_type_enum>(decl)->get_annotations();
            break;
        case IDeclaration::DK_FUNCTION:
            annotations = cast<IDeclaration_function>(decl)->get_annotations();
            break;
        default:
            break;
    }
    if (annotations != NULL) {
        int annotation_count = annotations->get_annotation_count();
        for (int k = 0; k < annotation_count; ++k)
            func.add_annotation(
                dag_builder.annotation_to_dag(annotations->get_annotation(k)));
    }
}

// Generate annotations for annotation declarations (only for the decl itself).
void Generated_code_dag::gen_anno_decl_annotations(
    DAG_builder                   &dag_builder,
    Annotation_info               &anno,
    IDeclaration_annotation const *decl)
{
    IAnnotation_block const *annotations = decl->get_annotations();
    if (annotations != NULL) {
        int annotation_count = annotations->get_annotation_count();
        for (int k = 0; k < annotation_count; ++k)
            anno.add_annotation(
                dag_builder.annotation_to_dag(annotations->get_annotation(k)));
    }
}

// Generate annotations for function return types.
void Generated_code_dag::gen_function_return_annotations(
    DAG_builder        &dag_builder,
    Function_info      &func,
    IDeclaration const *decl)
{
    // handle return value annotations: only real functions have them
    if (decl->get_kind() == IDeclaration::DK_FUNCTION) {
        IAnnotation_block const *annotations =
            cast<IDeclaration_function>(decl)->get_return_annotations();

        if (annotations != NULL) {
            int annotation_count = annotations->get_annotation_count();
            for (int k = 0; k < annotation_count; ++k)
                func.add_return_annotation(
                    dag_builder.annotation_to_dag(annotations->get_annotation(k)));
        }
    }
}

// Generate annotations for function parameters.
void Generated_code_dag::gen_function_parameter_annotations(
    DAG_builder        &dag_builder,
    Parameter_info     &param,
    IDefinition const  *f_def,
    IModule const      *owner_module,
    IDeclaration const *decl,
    int                k)
{
    switch (decl->get_kind()) {
    case IDeclaration::DK_TYPE_STRUCT:
        if (f_def->get_semantics() == IDefinition::DS_ELEM_CONSTRUCTOR) {
            // only the elemental constructor has annotations
            IDeclaration_type_struct const *s_decl = cast<IDeclaration_type_struct>(decl);

            if (IAnnotation_block const *annotations = s_decl->get_annotations(k)) {
                int annotation_count = annotations->get_annotation_count();
                for (int l = 0; l < annotation_count; ++l) {
                    IAnnotation const *anno = annotations->get_annotation(l);

                    if (IAnnotation_enable_if const *ei = as<IAnnotation_enable_if>(anno)) {
                        param.set_enable_if_condition(
                            dag_builder.exp_to_dag(ei->get_expression()));
                    }
                    param.add_annotation(dag_builder.annotation_to_dag(anno));
                }
            }
        }
        break;

    case IDeclaration::DK_TYPE_ENUM:
        // enums have only one copy constructor, no parameter annotations
        break;

    case IDeclaration::DK_FUNCTION:
        {
            IDeclaration_function const *fun_decl = cast<IDeclaration_function>(decl);

            if (fun_decl->is_preset()) {
                mi::base::Handle<IModule const> owner = mi::base::make_handle_dup(owner_module);
                fun_decl = skip_presets(fun_decl, owner);
            }

            IParameter const *parameter = fun_decl->get_parameter(k);

            if (IAnnotation_block const *annotations = parameter->get_annotations()) {
                int annotation_count = annotations->get_annotation_count();
                for (int l = 0; l < annotation_count; ++l) {
                    IAnnotation const *anno = annotations->get_annotation(l);

                    if (IAnnotation_enable_if const *ei = as<IAnnotation_enable_if>(anno)) {
                        param.set_enable_if_condition(
                            dag_builder.exp_to_dag(ei->get_expression()));
                    }
                    param.add_annotation(
                        dag_builder.annotation_to_dag(annotations->get_annotation(l)));
                }
            }
        }
        break;

    default:
        break;
    }
}

// Generate annotations for annotation (declaration) parameters.
void Generated_code_dag::gen_annotation_parameter_annotations(
    DAG_builder                   &dag_builder,
    Parameter_info                &param,
    IDefinition const             *f_def,
    IModule const                 *owner_module,
    IDeclaration_annotation const *decl,
    int                           k)
{
    IParameter const *parameter = decl->get_parameter(k);
    if (IAnnotation_block const *annotations = parameter->get_annotations()) {
        int annotation_count = annotations->get_annotation_count();
        for (int l = 0; l < annotation_count; ++l) {
            IAnnotation const *anno = annotations->get_annotation(l);

            param.add_annotation(dag_builder.annotation_to_dag(anno));
        }
    }
}

// Generate annotations for the module.
void Generated_code_dag::gen_module_annotations(
    DAG_builder               &dag_builder,
    IDeclaration_module const *decl)
{
    if (IAnnotation_block const *annotations = decl->get_annotations()) {
        int annotation_count = annotations->get_annotation_count();
        for (int k = 0; k < annotation_count; ++k) {
            m_module_annotations.push_back(
                dag_builder.annotation_to_dag(annotations->get_annotation(k)));
        }
        return;
    }
}

/// Get the single expression body of a function if its body can be expression in this form.
static IExpression const *get_single_expr_body(
    IDeclaration const *decl)
{
    IDeclaration_function const *func_decl = as<IDeclaration_function>(decl);
    if (func_decl == NULL) {
        return NULL;
    }
    IStatement const *body = func_decl->get_body();
    if (body == NULL) {
        return NULL;
    }

    if (IStatement_expression const *stmt_expr = as<IStatement_expression>(body)) {
        // new syntax: func() = expr
        return stmt_expr->get_expression();
    }
    if (IStatement_compound const *stmt_comp = as<IStatement_compound>(body)) {
        // real body
        if (stmt_comp->get_statement_count() > 0) {
            if (IStatement_return const *stmt_ret =
                as<IStatement_return>(stmt_comp->get_statement(0)))
            {
                // first statement is a return: get its expression
                return stmt_ret->get_expression();
            }
        }
    }
    return NULL;
}

// Compile functions.
void Generated_code_dag::compile_function(
    IModule const         *module,
    Dependence_node const *f_node)
{
    IType const *ret_type = f_node->get_return_type();

    // import the return type into our type factory
    ret_type = m_type_factory.import(ret_type);

    DAG_hash func_hash, *fh = NULL;

    IDefinition const *f_def = f_node->get_definition();

    if (f_def != NULL) {
        if (IModule::Function_hash const *h = module->get_function_hash(f_def)) {
            // we have a function hash value
            func_hash = DAG_hash(h->hash);
            fh        = &func_hash;
        }
    }

    DAG_builder dag_builder(get_allocator(), m_node_factory, m_mangler);

    Function_info func(
        get_allocator(),
        f_node->get_semantics(),
        ret_type,
        f_node->get_dag_name(),
        f_node->get_dag_simple_name(),
        f_node->get_dag_alias_name(),
        f_node->get_dag_preset_name(),
        fh);

    size_t parameter_count = f_node->get_parameter_count();
    for (size_t k = 0; k < parameter_count; ++k) {
        IType const *parameter_type;
        char const  *parameter_name;

        f_node->get_parameter(k, parameter_type, parameter_name);

        // import the parameter type into our type factory
        parameter_type = m_type_factory.import(parameter_type);
        string parameter_type_name = dag_builder.parameter_type_to_name(parameter_type);

        Parameter_info param(
            get_allocator(),
            parameter_type,
            parameter_name,
            parameter_type_name.c_str());

        func.add_parameter(param);
    }

    unsigned char func_properties = 1 << FP_IS_EXPORTED;

    if (f_def == NULL) {
        // DAG generated functions are not native but always uniform
        func_properties |= 1 << FP_IS_UNIFORM;

        // the rest of the processing is done inside the this module
        Module_scope scope(dag_builder, module);

        IDefinition::Semantics sema = f_node->get_semantics();

        bool mark_generated = false;

        if (is_DAG_semantics(sema) ||
            semantic_to_operator(sema) == IExpression::OK_TERNARY ||
            semantic_to_operator(sema) == IExpression::OK_ARRAY_INDEX)
        {
            // these are generated by the DAG-BE itself
            mark_generated = m_mark_generated;
        }

        // no annotations / default arguments for DAG generated entries
        if (mark_generated)
            mark_hidden(dag_builder, func);

        func.set_properties(func_properties);

        m_functions.push_back(func);

        build_function_temporaries(m_current_function_index);
        ++m_current_function_index;
        return;
    }

    func_properties |= get_function_properties(f_def);
    func.set_properties(func_properties);

    // annotations are attached to the prototype if one exists
    IDefinition const  *orig_f_def = module->get_original_definition(f_def);
    IDeclaration const *proto_decl = orig_f_def->get_prototype_declaration();
    IDeclaration const *func_decl  = orig_f_def->get_declaration();

    mi::base::Handle<IModule const> orig_module(module->get_owner_module(f_def));

    if (proto_decl == NULL)
        proto_decl = func_decl;

    // the rest of the processing is done inside the owner module
    Module_scope scope(dag_builder, orig_module.get());

    // annotations / default arguments must be retrieve from the prototype declaration
    if (proto_decl == NULL) {
        // can have defaults, handle them
        {
            // Do not CSE default parameters and annotations, don't build DAGs for them
            No_CSE_scope cse_off(m_node_factory);

            // create the default parameter initializers, but disable inlining
            // for them, so inspection shows the same structure as MDL declaration
            No_INLINE_scope no_inline(m_node_factory);

            // clear the temporary map, we will process default parameter initializers
            dag_builder.reset();

            for (size_t k = 0; k < parameter_count; ++k) {
                Parameter_info &param = func.get_parameter(k);

                param.set_default(
                    dag_builder.exp_to_dag(orig_f_def->get_default_param_initializer(k)));
            }
        }
    } else {
        gen_function_annotations(dag_builder, func, proto_decl);
        gen_function_return_annotations(dag_builder, func, proto_decl);

        // clear the temporary map, we will process default parameter initializers
        dag_builder.reset();

        for (size_t k = 0; k < parameter_count; ++k) {
            Parameter_info &param = func.get_parameter(k);

            // Do not CSE default parameters and annotations, don't build DAGs for them
            No_CSE_scope cse_off(m_node_factory);

            gen_function_parameter_annotations(
                dag_builder, param, f_def, orig_module.get(), proto_decl, k);

            // create the default parameter initializers, but disable inlining
            // for them, so inspection shows the same structure as MDL declaration
            {
                No_INLINE_scope no_inline(m_node_factory);

                if (proto_decl->get_kind() == IDeclaration::DK_FUNCTION) {
                    IDeclaration_function const *fun_decl = cast<IDeclaration_function>(proto_decl);

                    if (fun_decl->is_preset()) {
                        mi::base::Handle<IModule const> handle = orig_module;
                        fun_decl = skip_presets(fun_decl, handle);
                    }
                    IParameter const *parameter = fun_decl->get_parameter(k);
                    dag_builder.make_accessible(parameter);
                }

                param.set_default(
                    dag_builder.exp_to_dag(orig_f_def->get_default_param_initializer(k)));
            }
        }

        // compute control dependencies for enable_if conditions
        {
            Condition_compute compute(get_allocator(), parameter_count);

            for (size_t k = 0; k < parameter_count; ++k) {
                Parameter_info &param = func.get_parameter(k);

                compute.process_parameter(param, k);
            }

            for (size_t k = 0; k < parameter_count; ++k) {
                Parameter_info &param = func.get_parameter(k);

                typedef Condition_compute::Dependency_set DS;
                DS const &ctrls(compute.get_controls(k));

                for (DS::const_iterator it(ctrls.begin()), end(ctrls.end()); it != end; ++it)
                    param.add_user(*it);
            }
        }

        // convert the function body
        IExpression const *expr = get_single_expr_body(func_decl);

        func.set_body(expr != NULL ? dag_builder.exp_to_dag(expr) : NULL);

        collect_callees(func, f_node);
    }

    MDL_ASSERT(dag_builder.get_errors().size() == 0 && "Unexpected errors compiling function");

    m_functions.push_back(func);

    build_function_temporaries(m_current_function_index);
    ++m_current_function_index;
}

// Compile an annotation (declaration).
void Generated_code_dag::compile_annotation(
    IModule const         *module,
    Dependence_node const *f_node)
{
    Annotation_info anno(
        get_allocator(),
        f_node->get_semantics(),
        f_node->get_dag_name(),
        f_node->get_dag_simple_name(),
        f_node->get_dag_alias_name());

    DAG_builder dag_builder(get_allocator(), m_node_factory, m_mangler);

    size_t parameter_count = f_node->get_parameter_count();
    for (size_t k = 0; k < parameter_count; ++k) {
        IType const *parameter_type;
        char const  *parameter_name;

        f_node->get_parameter(k, parameter_type, parameter_name);

        // import the parameter type into our type factory
        parameter_type = m_type_factory.import(parameter_type);
        string parameter_type_name = dag_builder.parameter_type_to_name(parameter_type);

        Parameter_info param(
            get_allocator(),
            parameter_type,
            parameter_name,
            parameter_type_name.c_str());

        anno.add_parameter(param);
    }

    unsigned char anno_properties = 1 << AP_IS_EXPORTED;
    anno.set_properties(anno_properties);

    IDefinition const  *f_def = f_node->get_definition();
    IDefinition const  *orig_f_def = module->get_original_definition(f_def);

    IDeclaration_annotation const *decl =
        cast<IDeclaration_annotation>(orig_f_def->get_declaration());

    mi::base::Handle<IModule const> orig_module(module->get_owner_module(f_def));

    // the rest of the processing is done inside the owner module
    Module_scope scope(dag_builder, orig_module.get());

    gen_anno_decl_annotations(dag_builder, anno, decl);

    // clear the temporary map, we will process default parameter initializers
    dag_builder.reset();

    for (size_t k = 0; k < parameter_count; ++k) {
        Parameter_info &param = anno.get_parameter(k);

        // Do not CSE default parameters and annotations, don't build DAGs for them
        No_CSE_scope cse_off(m_node_factory);

        gen_annotation_parameter_annotations(
            dag_builder, param, f_def, orig_module.get(), decl, k);

        // create the default parameter initializers, but disable inlining
        // for them, so inspection shows the same structure as MDL declaration
        {
            No_INLINE_scope no_inline(m_node_factory);

            IParameter const *parameter = decl->get_parameter(k);
            dag_builder.make_accessible(parameter);

            param.set_default(
                dag_builder.exp_to_dag(orig_f_def->get_default_param_initializer(k)));
        }
    }

    MDL_ASSERT(dag_builder.get_errors().size() == 0 && "Unexpected errors compiling annotation");
    m_annotations.push_back(anno);
}

// Compile a local annotation (declaration).
void Generated_code_dag::compile_local_annotation(
    IModule const         *module,
    DAG_builder           &dag_builder,
    Dependence_node const *a_node)
{
    Annotation_info anno(
        get_allocator(),
        a_node->get_semantics(),
        a_node->get_dag_name(),
        a_node->get_dag_simple_name(),
        a_node->get_dag_alias_name());

    size_t parameter_count = a_node->get_parameter_count();
    for (size_t k = 0; k < parameter_count; ++k) {
        IType const *parameter_type;
        char const  *parameter_name;

        a_node->get_parameter(k, parameter_type, parameter_name);

        // import the parameter type into our type factory
        parameter_type = m_type_factory.import(parameter_type);
        string parameter_type_name = dag_builder.parameter_type_to_name(parameter_type);

        Parameter_info param(
            get_allocator(),
            parameter_type,
            parameter_name,
            parameter_type_name.c_str());

        anno.add_parameter(param);
    }

    unsigned char anno_properties = 0;

    IDefinition const *a_def = a_node->get_definition();

    // annotations are attached to the prototype if one exists
    IDefinition const  *orig_a_def = module->get_original_definition(a_def);
    IDeclaration_annotation const *a_decl =
        cast<IDeclaration_annotation>(orig_a_def->get_declaration());

    if (a_decl != NULL) {
        mi::base::Handle<IModule const> orig_module(module->get_owner_module(a_def));

        // annotations must be retrieve from the prototype declaration
        gen_anno_decl_annotations(dag_builder, anno, a_decl);
    }

    // clear the temporary map, we will process default parameter initializers
    dag_builder.reset();

    anno.set_properties(anno_properties);

    // Note: we do NOT create defaults for local annotations, even if they have ones

    m_annotations.push_back(anno);
}

// Compile a local function.
void Generated_code_dag::compile_local_function(
    IModule const         *module,
    DAG_builder           &dag_builder,
    Dependence_node const *f_node)
{
    IType const *ret_type = f_node->get_return_type();

    // import the return type into our type factory
    ret_type = m_type_factory.import(ret_type);

    DAG_hash func_hash, *fh = NULL;

    IDefinition const *f_def = f_node->get_definition();

    if (f_def != NULL) {
        if (IModule::Function_hash const *h = module->get_function_hash(f_def)) {
            // we have a function hash value
            func_hash = DAG_hash(h->hash);
            fh = &func_hash;
        }
    }

    Function_info func(
        get_allocator(),
        f_node->get_semantics(),
        ret_type,
        f_node->get_dag_name(),
        f_node->get_dag_simple_name(),
        f_node->get_dag_alias_name(),
        f_node->get_dag_preset_name(),
        fh);

    size_t parameter_count = f_node->get_parameter_count();
    for (size_t k = 0; k < parameter_count; ++k) {
        IType const *parameter_type;
        char const  *parameter_name;

        f_node->get_parameter(k, parameter_type, parameter_name);

        // import the parameter type into our type factory
        parameter_type = m_type_factory.import(parameter_type);
        string parameter_type_name = dag_builder.parameter_type_to_name(parameter_type);

        Parameter_info param(
            get_allocator(),
            parameter_type,
            parameter_name,
            parameter_type_name.c_str());

        func.add_parameter(param);
    }

    unsigned char func_properties = 0;

    if (f_def != NULL) {
        // get properties
        func_properties |= get_function_properties(f_def);

        // annotations are attached to the prototype if one exists
        IDefinition const  *orig_f_def = module->get_original_definition(f_def);
        IDeclaration const *proto_decl = orig_f_def->get_prototype_declaration();
        IDeclaration const *func_decl  = orig_f_def->get_declaration();

        mi::base::Handle<IModule const> orig_module(module->get_owner_module(f_def));

        if (proto_decl == NULL)
            proto_decl = func_decl;

        // annotations must be retrieve from the prototype declaration
        if (proto_decl != NULL) {
            gen_function_annotations(dag_builder, func, proto_decl);
            gen_function_return_annotations(dag_builder, func, proto_decl);

            // clear the temporary map
            dag_builder.reset();

            // convert the function body
            IExpression const *expr = get_single_expr_body(func_decl);

            func.set_body(expr != NULL ? dag_builder.exp_to_dag(expr) : NULL);
        }
    } else {
        // DAG generated functions are always uniform
        func_properties |= 1 << FP_IS_UNIFORM;
    }

    func.set_properties(func_properties);

    collect_callees(func, f_node);

    // Note: we do NOT create defaults for local functions, even if they have ones

    m_functions.push_back(func);

    build_function_temporaries(m_current_function_index);
    ++m_current_function_index;
}

namespace {

/// Helper struct for the let stack.
class Let_entry {
public:
    /// Constructor.
    Let_entry(IExpression_let const *let, IModule const *let_owner)
    : m_let(let), m_let_owner(let_owner)
    {
    }

    /// Get the let expression.
    IExpression_let const *get_let() const { return m_let; }

    /// Get the owner module of the let expression.
    IModule const *get_let_owner() const { return m_let_owner; }

private:
    /// The let expression.
    IExpression_let const *m_let;

    /// The owner module of the let expression.
    IModule const         *m_let_owner;
};


} // anonymous

// Compile a material.
void Generated_code_dag::compile_material(
    DAG_builder           &dag_builder,
    Dependence_node const *m_node)
{
    IDefinition const *material_def = m_node->get_definition();
    MDL_ASSERT(material_def != NULL);


    IModule const *module = dag_builder.tos_module();

    // We are starting a new DAG. Ensure CSE will not find old expressions.
    m_node_factory.identify_clear();

    Forbid_local_functions_scope forbid_scope(
        dag_builder, (m_options & FORBID_LOCAL_FUNC_CALLS) != 0);

    string mat_name(dag_builder.def_to_name(material_def, module));
    string mat_simple_name(dag_builder.def_to_name(material_def, (const char*)NULL));
    string orig_name(
        material_def->get_property(IDefinition::DP_IS_IMPORTED) ?
        dag_builder.def_to_name(material_def) : string("", get_allocator()));

    Material_info mat(get_allocator(), mat_name.c_str(), mat_simple_name.c_str(), orig_name.c_str());

    IType_function const *fun_type = as<IType_function>(material_def->get_type());
    int parameter_count = fun_type->get_parameter_count();

    for (int k = 0; k < parameter_count; ++k) {
        IType const   *parameter_type;
        ISymbol const *parameter_symbol;

        fun_type->get_parameter(k, parameter_type, parameter_symbol);

        // import the parameter type into our type factory
        parameter_type = m_type_factory.import(parameter_type);
        string parameter_type_name = dag_builder.parameter_type_to_name(parameter_type);

        Parameter_info param(
            get_allocator(),
            parameter_type,
            parameter_symbol->get_name(),
            parameter_type_name.c_str());

        mat.add_parameter(param);
    }

    if (material_def->get_semantics() == IDefinition::DS_ELEM_CONSTRUCTOR) {
        // The builtin material constructor.
        mat.set_body(build_material_dag(material_def));
    } else {
        // a real material or a clone
        IDefinition const *orig_mat_def = module->get_original_definition(material_def);
        mi::base::Handle<IModule const> orig_module(module->get_owner_module(material_def));


        IDeclaration_function const *mat_decl =
            cast<IDeclaration_function>(orig_mat_def->get_declaration());

        IDeclaration_function const *proto_decl =
            cast<IDeclaration_function>(orig_mat_def->get_prototype_declaration());
        if (proto_decl == NULL)
            proto_decl = mat_decl;

        // Handle annotations: These are attached at the prototype.
        // Note that the material might be imported, so ensure we have the right module scope.
        {
            Module_scope scope(dag_builder, orig_module.get());

            if (IAnnotation_block const *annotations = proto_decl->get_annotations()) {
                int annotation_count = annotations->get_annotation_count();
                for (int k = 0; k < annotation_count; ++k) {
                    IAnnotation const *anno = annotations->get_annotation(k);
                    mat.add_annotation(dag_builder.annotation_to_dag(anno));
                }
            }
        }

        // start temporaries
        dag_builder.reset();

        if (mat_decl->is_preset()) {
            // A preset: First handle the preset initializers: These are in the original module
            // (of the preset) at the original definition (of the preset).
            {
                Module_scope scope(dag_builder, orig_module.get());
                for (int k = 0; k < parameter_count; ++k) {
                    // Do not CSE default parameters, don't build DAGs for them
                    No_CSE_scope cse_off(m_node_factory);

                    // do not inline calls inside the default initializers
                    No_INLINE_scope no_inline(m_node_factory);

                    Parameter_info &param = mat.get_parameter(k);
                    param.set_default(
                        dag_builder.exp_to_dag(orig_mat_def->get_default_param_initializer(k)));
                }
            }

            // Now retrieve the original material definition of the preset.
            // Beware, might be a preset of a preset, so find out the original (non-preset)
            // material by iteration.

            IDefinition           const *orig_mat_def;
            IDeclaration_function const *orig_mat_proto;

            IDeclaration_function const     *orig_mat_decl = mat_decl;
            mi::base::Handle<IModule const> orig_mat_module(orig_module);

            typedef stack<Let_entry>::Type Let_stack;

            Let_stack let_stack(Let_stack::container_type(this->get_allocator()));

            do {
                IStatement_expression const *clone_stmnt =
                    cast<IStatement_expression>(orig_mat_decl->get_body());
                IExpression const *mat_inst = clone_stmnt->get_expression();
                IExpression_call const *call = NULL;
                if (IExpression_let const *let = as<IExpression_let>(mat_inst)) {
                    call = cast<IExpression_call>(let->get_expression());

                    let_stack.push(Let_entry(let, orig_mat_module.get()));
                } else {
                    call = cast<IExpression_call>(mat_inst);
                }
                IExpression_reference const *ref =
                    cast<IExpression_reference>(call->get_reference());

                orig_mat_def = ref->get_definition();

                mi::base::Handle<IModule const> next(
                    orig_mat_module->get_owner_module(orig_mat_def));

                orig_mat_def    = orig_mat_module->get_original_definition(orig_mat_def);
                orig_mat_module = next;

                // get the prototype of the clone material if any, else its definition
                orig_mat_decl =
                    cast<IDeclaration_function>(orig_mat_def->get_declaration());
                orig_mat_proto =
                    cast<IDeclaration_function>(orig_mat_def->get_prototype_declaration());
                if (orig_mat_proto == NULL)
                    orig_mat_proto = orig_mat_decl;
            } while (orig_mat_decl->is_preset());

            string preset_material(orig_mat_module->get_name(), get_allocator());
            preset_material += "::";
            preset_material += orig_mat_def->get_symbol()->get_name();

            mat.set_cloned_name(preset_material.c_str());


            // Retrieve the initializers of the preset. These are created in the module
            // of the preset.

            // First step: handle let declarations
            while (!let_stack.empty()) {
                // Do not CSE default parameters, don't build DAGs for them
                No_CSE_scope cse_off(m_node_factory);

                // do not inline calls inside the default initializers
                No_INLINE_scope no_inline(m_node_factory);

                Let_entry const &e = let_stack.top();

                Module_scope scope(dag_builder, e.get_let_owner());

                IExpression_let const *let = e.get_let();

                for (int i = 0, n = let->get_declaration_count(); i < n; ++i) {
                    IDeclaration_variable const *v_decl =
                        cast<IDeclaration_variable>(let->get_declaration(i));

                    dag_builder.var_decl_to_dag(v_decl);
                }

                let_stack.pop();
            }

            // the following operations are done at the module of the original material,
            // so enter it
            Module_scope scope(dag_builder, orig_mat_module.get());

            // Retrieve the parameter annotations from the original material, those are
            // "copied", because the preset itself does not have parameters.
            for (int k = 0; k < parameter_count; ++k) {
                IParameter const *orig_mat_param = orig_mat_proto->get_parameter(k);
                if (IAnnotation_block const *annotations = orig_mat_param->get_annotations()) {
                    Parameter_info &param = mat.get_parameter(k);

                    int annotation_count = annotations->get_annotation_count();
                    for (int l = 0; l < annotation_count; ++l) {
                        IAnnotation const *anno = annotations->get_annotation(l);
                        param.add_annotation(dag_builder.annotation_to_dag(anno));
                    }
                }
            }

            // finally create the "material expression" of the preset by
            // wiring the expression of the original material to the parameters
            // of the preset ...
            mat.set_body(dag_builder.preset_to_dag(orig_mat_def));
        } else {
            // a real material
            Module_scope scope(dag_builder, orig_module.get());

            for (int k = 0; k < parameter_count; ++k) {
                // Do not CSE default parameters and annotations, don't build DAGs for them
                No_CSE_scope   cse_off(m_node_factory);
                Parameter_info &param = mat.get_parameter(k);

                IParameter const *parameter = proto_decl->get_parameter(k);

                // retrieve the parameter annotation from the prototype declaration
                if (IAnnotation_block const *annotations = parameter->get_annotations()) {
                    int annotation_count = annotations->get_annotation_count();
                    for (int l = 0; l < annotation_count; ++l) {
                        IAnnotation const *anno = annotations->get_annotation(l);

                        if (IAnnotation_enable_if const *ei = as<IAnnotation_enable_if>(anno)) {
                            param.set_enable_if_condition(
                                dag_builder.exp_to_dag(ei->get_expression()));
                        }
                        param.add_annotation(dag_builder.annotation_to_dag(anno));
                    }
                }

                // retrieve the default initializers, but disable inlining
                // for them, so inspection shows the same structure as MDL declaration
                dag_builder.make_accessible(parameter);
                {
                    No_INLINE_scope no_inline(m_node_factory);

                    param.set_default(
                        dag_builder.exp_to_dag(orig_mat_def->get_default_param_initializer(k)));
                }
            }

            // compute control dependencies for enable_if conditions
            {
                Condition_compute compute(get_allocator(), parameter_count);

                for (size_t k = 0; k < parameter_count; ++k) {
                    Parameter_info &param = mat.get_parameter(k);

                    compute.process_parameter(param, k);
                }

                for (size_t k = 0; k < parameter_count; ++k) {
                    Parameter_info &param = mat.get_parameter(k);

                    typedef Condition_compute::Dependency_set DS;
                    DS const &ctrls(compute.get_controls(k));

                    for (DS::const_iterator it(ctrls.begin()), end(ctrls.end()); it != end; ++it)
                        param.add_user(*it);
                }
            }

            // convert the material body
            IStatement_expression const *expr_stmt =
                cast<IStatement_expression>(mat_decl->get_body());

            mat.set_body(dag_builder.exp_to_dag(expr_stmt->get_expression()));
        }
    }

    // handle errors
    DAG_builder::Ref_vector const &errors = dag_builder.get_errors();
    if (!errors.empty()) {
        for (size_t i = 0, n = errors.size(); i < n; ++i) {
            // This material uses an unexported function call which is forbidden in
            // the current context.
            IExpression_reference const *ref      = errors[i];
            IDefinition const           *call_def = ref->get_definition();
            string msg(get_allocator());

            msg += "Call to unexported function '";
            msg += call_def->get_symbol()->get_name();
            msg += "' is not allowed in this context (inside ";
            msg += m_renderer_context_name;
            msg += ")";
            error(
                FORBIDDEN_CALL_TO_UNEXPORTED_FUNCTION,
                ref->access_position(),
                msg.c_str());
        }
        // KILL the current material if errors were detected
    } else {
        m_materials.push_back(mat);

        // create temporaries based on CSE
        build_material_temporaries(m_current_material_index);

        ++m_current_material_index;
    }
}

// Compile a local material.
void Generated_code_dag::compile_local_material(
    DAG_builder       &dag_builder,
    IDefinition const *material_def)
{
    // ignore local materials for now. Currently they are always inlined, so there is no trouble
    // with them
}

/// Helper class to enumerate all local types of a module.
class Local_type_enumerator : public IDefinition_visitor {
public:
    /// Enumerate all local types.
    ///
    /// \param code_dag     the current code_dag
    /// \param mod          the current module
    /// \param dag_builder  the DAG builder to be used
    static void enumerate_local_types(
        Generated_code_dag &code_dag,
        Module const       *mod,
        DAG_builder        &dag_builder)
    {
        Local_type_enumerator enumerator(code_dag, dag_builder);

        Definition_table const &dt = mod->get_definition_table();
        dt.walk(&enumerator);
    }

private:
    /// Called for every visited definition.
    ///
    /// \param def  the definition
    void visit(Definition const *def) const MDL_FINAL
    {
        if (def->get_kind()!= Definition::DK_TYPE)
            return;

        if (def->has_flag(Definition::DEF_IS_EXPORTED))
            return;
        if (def->has_flag(Definition::DEF_IS_IMPORTED))
            return;

        IType const *type = def->get_type();

        if (!DAG_builder::is_user_type(type))
            return;

        m_code_dag.compile_type(m_dag_builder, def, /*is_exported=*/false);
    }

private:
    /// Constructor.
    ///
    /// \param code_dag     the current code_dag
    /// \param dag_builder  the DAG builder to be used
    Local_type_enumerator(
        Generated_code_dag &code_dag,
        DAG_builder        &dag_builder)
    : m_code_dag(code_dag)
    , m_dag_builder(dag_builder)
    {
    }

private:
    /// The current code dag.
    Generated_code_dag &m_code_dag;

    /// The DAG builder to be used.
    DAG_builder &m_dag_builder;
};

// Compile the module.
void Generated_code_dag::compile(IModule const *module)
{
    m_current_material_index = 0;

    m_node_factory.enable_cse(true);

    DAG_builder  dag_builder(get_allocator(), m_node_factory, m_mangler);
    Module_scope scope(dag_builder, module);


    // first step
    IAllocator *alloc = m_arena.get_allocator();

    if (IDeclaration_module const *mod_decl = module->get_module_declaration()) {
        gen_module_annotations(dag_builder, mod_decl);
    }

    // ... collect types and constants ... these are not really needed for the compilation itself
    for (int i = 0, n = module->get_exported_definition_count(); i < n; ++i) {
        IDefinition const *def = module->get_exported_definition(i);
        switch (def->get_kind()) {
        case IDefinition::DK_CONSTANT:
            compile_constant(dag_builder, def);
            break;
        case IDefinition::DK_TYPE:
            compile_type(dag_builder, def, /*is_exported*/true);
            break;
        default:
            break;
        }
    }

    // finally compile the module starting with the exported entities.
    bool include_locals = (m_options & INCLUDE_LOCAL_ENTITIES) != 0;

    if (include_locals) {
        // collect local types
        Local_type_enumerator::enumerate_local_types(
            *this, impl_cast<Module>(module), dag_builder);
    }

    // ... build the dependence graph first
    DAG_dependence_graph dep_graph(
        alloc, *this, dag_builder, m_invisible_sym, include_locals);

    // now create the topo-sort
    bool has_loops = false;
    typedef DAG_dependence_graph::Node_list Node_list;

    Node_list const &topo_list(dep_graph.get_module_entities(has_loops));
    MDL_ASSERT(!has_loops && "Dependency graph has loops");

    for (Node_list::const_iterator it(topo_list.begin()), end(topo_list.end()); it != end; ++it) {
        Dependence_node const *n = *it;

        if (n->is_export())
            compile_entity(dag_builder, n);
        else if (n->is_local())
            compile_local_entity(dag_builder, n);
    }

    // if the state must be imported, check if is was, else add it
    if (m_node_factory.needs_state_import()) {
        add_import("::state");
    }
    // if the state must be imported, check if is was, else add it
    if (m_node_factory.needs_nvidia_df_import()) {
        add_import("::nvidia::df");
    }
    // check if ::anno is needed as well
    if (m_needs_anno) {
        add_import("::anno");
    }

    // compilation has finished: clear the CSE table, so it will be safe to
    // update resource values with tags
    m_node_factory.identify_clear();
}

// Helper function, adds a "hidden" annotation to a generated function.
void Generated_code_dag::mark_hidden(
    DAG_builder   &dag_builder,
    Function_info &func)
{
    DAG_node const *hidden = dag_builder.create_hidden_annotation();
    if (hidden != NULL) {
        func.m_annotations.push_back(hidden);
        m_needs_anno = true;
    }
}

// Helper function, collect all direct callees of a given function.
void Generated_code_dag::collect_callees(
    Function_info         &func,
    Dependence_node const *node)
{
    Edge_list const &edges = node->get_callee_edges();

    for (Edge_iterator it(edges.begin()), end(edges.end()); it != end; ++it) {
        Dependence_node const *node = (*it)->get_dst();

        func.m_refs.push_back(string(node->get_dag_name(), get_allocator()));
    }
    if (char const *alias = node->get_dag_alias_name()) {
        // always add a reference to the original entity, we need it to handle this alias
        func.m_refs.push_back(string(alias, get_allocator()));
    }
}

// Build the dag for the builtin material constructor.
DAG_node const *Generated_code_dag::build_material_dag(
    IDefinition const *def)
{
    MDL_ASSERT(def->get_semantics() == IDefinition::DS_ELEM_CONSTRUCTOR);

    IType_function const *fun_type = cast<IType_function>(def->get_type());

    int argument_count = fun_type->get_parameter_count();
    VLA<DAG_call::Call_argument> arguments(get_allocator(), argument_count);

    for (int i = 0; i < argument_count; ++i) {
        IType const   *parameter_type;
        ISymbol const *parameter_symbol;

        fun_type->get_parameter(i, parameter_type, parameter_symbol);

        // import the parameter type
        parameter_type = m_type_factory.import(parameter_type->skip_type_alias());

        arguments[i].arg        = m_node_factory.create_parameter(parameter_type, i);
        arguments[i].param_name = parameter_symbol->get_name();
    }
    return m_node_factory.create_call(
        def->get_symbol()->get_name(),
        def->get_semantics(),
        arguments.data(),
        argument_count,
        fun_type->get_return_type());
}

// Compile a user defined type.
void Generated_code_dag::compile_type(
    DAG_builder       &dag_builder,
    IDefinition const *def,
    bool              is_exported)
{
    bool is_reexported = def->get_property(IDefinition::DP_IS_IMPORTED);

    IAnnotation_block const *annotations = NULL;

    IDeclaration const *decl = def->get_declaration();
    if (is_reexported) {
        // this is a re-exported type
        decl = NULL;
    }

    IType const *type = def->get_type();
    type = m_type_factory.import(type);

    ISymbol const *sym = def->get_symbol();

    char const *orig_name = NULL;

    // add original name
    if (is_reexported) {
        ISymbol const *sym = NULL;
        switch (type->get_kind()) {
        case IType::TK_STRUCT:
            sym = cast<IType_struct>(type)->get_symbol();
            break;
        case IType::TK_ENUM:
            sym = cast<IType_enum>(type)->get_symbol();
            break;
        case IType::TK_ALIAS:
            {
                IType_alias const *a_type = cast<IType_alias>(type);
                sym = a_type->get_symbol();
                MDL_ASSERT(sym != NULL && "re-exported alias has no name");
            }
            break;
        default:
            MDL_ASSERT(!"re-exported type is neither a enum nor a struct, nor an alias");
            break;
        }
        if (sym != NULL) {
            orig_name = sym->get_name();
        }
    }

    IAllocator *alloc = get_allocator();
    User_type_info user_type(alloc, is_exported, type, sym->get_name(), orig_name);

    if (decl == NULL) {
        // imported type, no annotations
        IType const *tp = def->get_type();
        switch (tp->get_kind()) {
        case IType::TK_ALIAS:
            // no annotations on a typedef
            break;
        case IType::TK_STRUCT:
            {
                IType_struct const *s_tp = cast<IType_struct>(tp);
                for (int i = 0, n = s_tp->get_field_count(); i < n; ++i) {
                    user_type.add_entity(User_type_info::Entity_info(alloc));
                }
            }
            break;
        case IType::TK_ENUM:
            {
                IType_enum const *e_tp = cast<IType_enum>(tp);
                for (int i = 0, n = e_tp->get_value_count(); i < n; ++i) {
                    user_type.add_entity(User_type_info::Entity_info(alloc));
                }
            }
            break;
        default:
            MDL_ASSERT(!"unknown type kind");
            break;
        }
    } else {
        switch (decl->get_kind()) {
        case IDeclaration::DK_TYPE_ALIAS:
            // no annotations on a typedef
            annotations = NULL;
            break;
        case IDeclaration::DK_TYPE_STRUCT:
            {
                IDeclaration_type_struct const *s_decl = cast<IDeclaration_type_struct>(decl);

                annotations = s_decl->get_annotations();

                for (int i = 0, n = s_decl->get_field_count(); i < n; ++i) {
                    User_type_info::Entity_info ent(alloc);

                    if (IAnnotation_block const *field_annos = s_decl->get_annotations(i)) {
                        int annotation_count = field_annos->get_annotation_count();
                        for (int k = 0; k < annotation_count; ++k) {
                           ent.add_annotation(
                                dag_builder.annotation_to_dag(field_annos->get_annotation(k)));
                        }
                    }

                    user_type.add_entity(ent);
                }
            }
            break;
        case IDeclaration::DK_TYPE_ENUM:
            {
                IDeclaration_type_enum const *e_decl = cast<IDeclaration_type_enum>(decl);
                annotations = e_decl->get_annotations();

                for (int i = 0, n = e_decl->get_value_count(); i < n; ++i) {
                    User_type_info::Entity_info ent(alloc);

                    if (IAnnotation_block const *value_annos = e_decl->get_annotations(i)) {
                        int annotation_count = value_annos->get_annotation_count();
                        for (int k = 0; k < annotation_count; ++k) {
                            ent.add_annotation(
                                dag_builder.annotation_to_dag(value_annos->get_annotation(k)));
                        }
                    }

                    user_type.add_entity(ent);
                }
            }
            break;
        default:
            MDL_ASSERT(!"unknown type kind");
            break;
        }
    }

    if (annotations != NULL) {
        int annotation_count = annotations->get_annotation_count();
        for (int k = 0; k < annotation_count; ++k) {
            user_type.add_annotation(
                dag_builder.annotation_to_dag(annotations->get_annotation(k)));
        }
    }

    m_user_types.push_back(user_type);
}

// Compile a constant.
void Generated_code_dag::compile_constant(
    DAG_builder       &dag_builder,
    IDefinition const *def)
{
    // Do not CSE constants, don't build DAGs for them
    No_CSE_scope cse_off(m_node_factory);

    IAllocator *alloc = get_allocator();

    ISymbol const *sym = def->get_symbol();
    IValue const  *v   = def->get_constant_value();

    v = m_value_factory.import(v);

    DAG_constant const *c = m_node_factory.create_constant(v);

    Constant_info con(alloc, c, sym->get_name());

    IAnnotation_block const *annotations = NULL;
    IDeclaration const      *decl        = def->get_declaration();

    if (decl != NULL) {
        IDeclaration_constant const *c_decl = cast<IDeclaration_constant>(decl);

        // one declaration can define several constants, find the right one
        int i = 0, n = c_decl->get_constant_count();
        for (; i < n; ++i) {
            ISimple_name const *sname = c_decl->get_constant_name(i);

            if (sname->get_definition() == def)
                break;
        }
        if (i < n) {
            annotations = c_decl->get_annotations(i);
        } else {
            MDL_ASSERT(!"constant was not found in associated declaration");
        }
    }

    if (annotations != NULL) {
        int annotation_count = annotations->get_annotation_count();
        for (int k = 0; k < annotation_count; ++k) {
            con.add_annotation(
                dag_builder.annotation_to_dag(annotations->get_annotation(k)));
        }
    }

    m_user_constants.push_back(con);
}

// Creates a new error message.
void Generated_code_dag::error(int code, Err_location const &loc, char const *msg)
{
    size_t fname_id = 0; // always current file
    m_messages.add_error_message(code, MESSAGE_CLASS, fname_id, loc.get_position(), msg);
}

// Check if the name for the given definition must get a signature suffix.
bool Generated_code_dag::need_signature_suffix(IDefinition const *def) const
{
    IDefinition::Kind kind = def->get_kind();
    if (kind == IDefinition::DK_ANNOTATION) {
        // annotations are not stored in the neuray DB
        return false;
    }
    if (IType_function const *fun_type = as<IType_function>(def->get_type())) {
        if (def->get_semantics() == IDefinition::DS_UNKNOWN) {
            // a user defined function
            IType const *ret_type = fun_type->get_return_type();

            // do not add a suffix for material creator function, these are not
            // overloaded
            return !is_material_type(ret_type);
        }
        return true;
    }
    return false;
}

// Returns true for MDL definition that should not be visible in the DAG backend.
bool Generated_code_dag::skip_definition(IDefinition const *def)
{
    return DAG_dependence_graph::skip_definition(def);
}

// Compile the given entity to the DAG representation.
void Generated_code_dag::compile_entity(
    DAG_builder           &dag_builder,
    Dependence_node const *node)
{
    IModule const *module = dag_builder.tos_module();

    IDefinition const *def      = node->get_definition();
    IType const       *ret_type = node->get_return_type();

    IDefinition::Kind kind = def != NULL ? def->get_kind() : IDefinition::DK_ERROR;

    if (kind == IDefinition::DK_ANNOTATION) {
        compile_annotation(module, node);
    } else if (kind == IDefinition::DK_FUNCTION && is_material_type(ret_type)) {
        // functions returning materials ARE materials
        compile_material(dag_builder, node);
    } else {
        compile_function(module, node);
    }
}

// Compile the given local entity to the DAG representation.
void Generated_code_dag::compile_local_entity(
    DAG_builder           &dag_builder,
    Dependence_node const *node)
{
    IDefinition const *def      = node->get_definition();
    IType const       *ret_type = node->get_return_type();

    IDefinition::Kind kind = def != NULL ? def->get_kind() : IDefinition::DK_ERROR;

    if (kind == IDefinition::DK_ANNOTATION) {
        IModule const *module = dag_builder.tos_module();

        compile_local_annotation(module, dag_builder, node);
    } else if (kind == IDefinition::DK_FUNCTION && is_material_type(ret_type)) {
        // functions returning materials ARE materials
        compile_local_material(dag_builder, def);
    } else {
        IModule const *module = dag_builder.tos_module();

        compile_local_function(module, dag_builder, node);
    }
}

// Build temporaries for a material by traversing the DAG and creating them
// for nodes with phen-out > 1.
void Generated_code_dag::build_material_temporaries(int mat_index)
{
    /// Helper class: creates temporaries for node when phen-out > 1.
    class Temporary_inserter : public Abstract_temporary_inserter
    {
    public:
        /// Constructor.
        ///
        /// \param dag                 the code DAG
        /// \param mat_index           the material index
        /// \param phen_outs           the phen-out map for the visited DAG IR
        /// \param temp_name_map       the desired temporary names
        Temporary_inserter(
            Generated_code_dag       &dag,
            int                      mat_index,
            Phen_out_map const       &phen_outs,
            Temporary_name_map const &temp_name_map)
        : Abstract_temporary_inserter(
            dag.get_allocator(),
            *dag.get_node_factory(),
            phen_outs,
            temp_name_map)
        , m_dag(dag)
        , m_mat_index(mat_index)
        {
        }

        /// Create and register a new temporary.
        ///
        /// \param node  the initializer for the temporary
        int add_temporary(DAG_node const *node, char const *name) MDL_FINAL
        {
            return m_dag.add_material_temporary(m_mat_index, node, name);
        }

    private:
        /// The DAG.
        Generated_code_dag &m_dag;
        /// The material index.
        int m_mat_index;
    };

    // we will modify the identify table, so clear it here, but safe the name map first
    DAG_node_factory_impl::Definition_temporary_name_map temp_name_map
        = m_node_factory.get_temp_name_map();
    m_node_factory.identify_clear();

    Phen_out_map phen_outs(0, Phen_out_map::hasher(), Phen_out_map::key_equal(), get_allocator());

    DAG_ir_walker walker(get_allocator(), /*as_tree=*/false);
    Calc_phen_out phen_counter(phen_outs);

    walker.walk_material(this, mat_index, &phen_counter);

    Temporary_inserter inserter(*this, mat_index, phen_outs, temp_name_map);

    walker.walk_material(this, mat_index, &inserter);
}

// Build temporaries for a material by traversing the DAG and creating them
// for nodes with phen-out > 1.
void Generated_code_dag::build_function_temporaries(int func_index)
{
    /// Helper class: creates temporaries for node when phen-out > 1.
    class Temporary_inserter : public Abstract_temporary_inserter
    {
    public:
        /// Constructor.
        ///
        /// \param dag                 the code DAG
        /// \param func_index          the function index
        /// \param phen_outs           the phen-out map for the visited DAG IR
        /// \param temp_name_map       the desired temporary names
        Temporary_inserter(
            Generated_code_dag &dag,
            int                func_index,
            Phen_out_map const &phen_outs,
            Temporary_name_map const &temp_name_map)
            : Abstract_temporary_inserter(
                dag.get_allocator(),
                *dag.get_node_factory(),
                phen_outs,
                temp_name_map)
            , m_dag(dag)
            , m_func_index(func_index)
        {
        }

        /// Create and register a new temporary.
        ///
        /// \param node  the initializer for the temporary
        int add_temporary(DAG_node const *node, char const *name) MDL_FINAL
        {
            return m_dag.add_function_temporary(m_func_index, node, name);
        }

    private:
        /// The DAG.
        Generated_code_dag &m_dag;
        /// The function index.
        int m_func_index;
    };

    if (get_function_body(func_index) == NULL) {
        // not all functions have a body, ignore those without
        return;
    }

    // we will modify the identify table, so clear it here, but safe the name map first
    DAG_node_factory_impl::Definition_temporary_name_map temp_name_map
        = m_node_factory.get_temp_name_map();
    m_node_factory.identify_clear();

    Phen_out_map phen_outs(0, Phen_out_map::hasher(), Phen_out_map::key_equal(), get_allocator());

    DAG_ir_walker walker(get_allocator(), /*as_tree=*/false);
    Calc_phen_out phen_counter(phen_outs);

    walker.walk_function(this, func_index, &phen_counter);

    Temporary_inserter inserter(*this, func_index, phen_outs, temp_name_map);

    walker.walk_function(this, func_index, &inserter);
}

// Get the kind of code generated.
IGenerated_code::Kind Generated_code_dag::get_kind() const
{
    return CK_DAG;
}

// Get the target language.
const char *Generated_code_dag::get_target_language() const
{
    return "dag";
}

// Get the module name from the module from which this code was generated.
const char *Generated_code_dag::get_module_name() const
{
    return m_module_name.c_str();
}

// Get the module file name from the module from which this code was generated.
const char *Generated_code_dag::get_module_file_name() const
{
    return m_module_file_name.c_str();
}

// Get the number of modules directly imported by the module
// from which this code was generated.
size_t Generated_code_dag::get_import_count() const
{
    return m_module_imports.size();
}

// Get the module at index imported from the module
// from which this code was generated.
char const *Generated_code_dag::get_import(
    size_t index) const
{
    if (index < m_module_imports.size()) {
        return m_module_imports[index].c_str();
    }
    return NULL;
}

// Get the number of functions in the generated code.
size_t Generated_code_dag::get_function_count() const
{
    return m_functions.size();
}

// Get the return type of the function at function_index.
IType const *Generated_code_dag::get_function_return_type(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_return_type();
    }
    return NULL;
}

// Get the semantics of the function at function_index.
IDefinition::Semantics Generated_code_dag::get_function_semantics(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_semantics();
    }
    return IDefinition::DS_UNKNOWN;
}

// Get the name of the function at function_index.
char const *Generated_code_dag::get_function_name(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_name();
    }
    return NULL;
}

// Get the original name of the function at function_index if the function name is an alias.
char const *Generated_code_dag::get_original_function_name(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_original_name();
    }
    return NULL;
}

// Get the simple name of the function at function_index.
char const *Generated_code_dag::get_simple_function_name(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_simple_name();
    }
    return NULL;
}

// Get the parameter count of the function at function_index.
size_t Generated_code_dag::get_function_parameter_count(size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_parameter_count();
    }
    return 0;
}

// Get the parameter type of the parameter at parameter_index
// of the function at function_index.
IType const *Generated_code_dag::get_function_parameter_type(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        return param->get_type();
    }
    return NULL;
}

/// Get the parameter type name of the parameter at parameter_index
/// of the function at function_index.
char const *Generated_code_dag::get_function_parameter_type_name(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        return param->get_type_name();
    }
    return NULL;
}

// Get the parameter name of the parameter at parameter_index
// of the function at function_index.
char const *Generated_code_dag::get_function_parameter_name(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        return param->get_name();
    }
    return NULL;
}

// Get the index of the parameter parameter_name.
size_t Generated_code_dag::get_function_parameter_index(
    size_t     function_index,
    char const *parameter_name) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        for (size_t i = 0, n = func->get_parameter_count(); i < n; ++i) {
            if (strcmp(parameter_name, func->get_parameter(i).get_name()) == 0)
                return i;
        }
    }
    return ~size_t(0);
}

// Get the enable_if condition for the given function parameter if one was specified.
DAG_node const *Generated_code_dag::get_function_parameter_enable_if_condition(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        return param->get_enable_if_condition();
    }
    return NULL;
}

// Get the number of parameters whose enable_if condition depends on this parameter.
size_t Generated_code_dag::get_function_parameter_enable_if_condition_users(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        Index_vector const &users = param->get_users();
        return users.size();
    }
    return 0;
}

// Get a parameter index whose enable_if condition depends on this parameter.
size_t Generated_code_dag::get_function_parameter_enable_if_condition_user(
    size_t function_index,
    size_t parameter_index,
    size_t user_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        Index_vector const &users = param->get_users();
        if (user_index < users.size())
            return users[user_index];
    }
    return ~size_t(0);
}

// Get the function hash value for the given function index if available.
DAG_hash const *Generated_code_dag::get_function_hash(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_hash();
    }
    return NULL;
}

// Check if the code contents are valid.
bool Generated_code_dag::is_valid() const
{
    return m_messages.get_error_message_count() == 0;
}

// Access messages.
Messages const &Generated_code_dag::access_messages() const
{
    return m_messages;
}

// Get the number of materials in the generated code.
size_t Generated_code_dag::get_material_count() const
{
    return m_materials.size();
}

// Get the name of the material at material_index.
char const *Generated_code_dag::get_material_name(size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index))
        return mat->get_name();
    return NULL;
}

// Get the simple name of the material at material_index.
char const *Generated_code_dag::get_simple_material_name(size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index))
        return mat->get_simple_name();
    return NULL;
}

// Get the original name of the material at material_index if the material name is an alias.
char const *Generated_code_dag::get_original_material_name(size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index))
        return mat->get_original_name();
    return NULL;
}

// Get the parameter count of the material at material_index.
size_t Generated_code_dag::get_material_parameter_count(size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index))
        return mat->get_parameter_count();
    return 0;
}

// Get the parameter type of the parameter at parameter_index
// of the material at material_index.
IType const *Generated_code_dag::get_material_parameter_type(
    size_t material_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index))
        return param->get_type();
    return NULL;
}

// Get the parameter name of the parameter at parameter_index
// of the material at material_index.
char const *Generated_code_dag::get_material_parameter_name(
    size_t material_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index))
        return param->get_name();
    return NULL;
}

// Get the index of the parameter parameter_name.
size_t Generated_code_dag::get_material_parameter_index(
    size_t     material_index,
    char const *parameter_name) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        for (size_t i = 0, n = mat->get_parameter_count(); i < n; ++i) {
            if (strcmp(parameter_name, mat->get_parameter(i).get_name()) == 0)
                return i;
        }
    }
    return ~size_t(0);
}

// Get the enable_if condition for the given material parameter if one was specified.
DAG_node const *Generated_code_dag::get_material_parameter_enable_if_condition(
    size_t material_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index))
        return param->get_enable_if_condition();
    return NULL;
}

// Get the number of parameters whose enable_if condition depends on this parameter.
size_t Generated_code_dag::get_material_parameter_enable_if_condition_users(
    size_t material_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index)) {
        Index_vector const &users = param->get_users();
        return users.size();
    }
    return 0;
}

// Get a parameter index whose enable_if condition depends on this parameter.
size_t Generated_code_dag::get_material_parameter_enable_if_condition_user(
    size_t material_index,
    size_t parameter_index,
    size_t user_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index)) {
        Index_vector const &users = param->get_users();
        if (user_index < users.size())
            return users[user_index];
    }
    return ~size_t(0);
}

// Get the node IR-node factory of this code DAG.
DAG_node_factory_impl *Generated_code_dag::get_node_factory()
{
    return &m_node_factory;
}

// Get the number of annotations of the function at function_index.
size_t Generated_code_dag::get_function_annotation_count(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_annotation_count();
    }
    return 0;
}

// Get the annotation at annotation_index of the function at function_index.
DAG_node const *Generated_code_dag::get_function_annotation(
    size_t function_index,
    size_t annotation_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        if (annotation_index < func->get_annotation_count())
            return func->get_annotation(annotation_index);
    }
    return NULL;
}

// Get the number of annotations of the function return type at function_index.
size_t Generated_code_dag::get_function_return_annotation_count(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_return_annotation_count();
    }
    return 0;
}

// Get the annotation at annotation_index of the function return type at function_index.
DAG_node const *Generated_code_dag::get_function_return_annotation(
    size_t function_index,
    size_t annotation_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        if (size_t(annotation_index) < func->get_return_annotation_count())
            return func->get_return_annotation(annotation_index);
    }
    return NULL;
}

// Get the default initializer of the parameter at parameter_index
// of the function at function_index.
DAG_node const *Generated_code_dag::get_function_parameter_default(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        return param->get_default();
    }
    return NULL;
}

// Get the number of annotations of the parameter at parameter_index
// of the function at function_index.
size_t Generated_code_dag::get_function_parameter_annotation_count(
    size_t function_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        return param->get_annotation_count();
    }
    return 0;
}

// Get the annotation at annotation_index of the parameter at parameter_index
// of the function at function_index.
DAG_node const *Generated_code_dag::get_function_parameter_annotation(
    size_t function_index,
    size_t parameter_index,
    size_t annotation_index) const
{
    if (Parameter_info const *param = get_func_param_info(function_index, parameter_index)) {
        if (size_t(annotation_index) < param->get_annotation_count())
            return param->get_annotation(annotation_index);
    }
    return NULL;
}

// Get the number of temporaries used by the function at function_index.
size_t Generated_code_dag::get_function_temporary_count(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_temporary_count();
    }
    return 0;
}

// Get the temporary at temporary_index used by the function at function_index.
DAG_node const *Generated_code_dag::get_function_temporary(
    size_t function_index,
    size_t temporary_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        if (temporary_index < func->get_temporary_count()) {
            return func->get_temporary(temporary_index);
        }
    }
    return NULL;
}

// Get the temporary name at temporary_index used by the function at function_index.
char const *Generated_code_dag::get_function_temporary_name(
    size_t function_index,
    size_t temporary_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        if (temporary_index < func->get_temporary_count()) {
            return func->get_temporary_name(temporary_index);
        }
    }
    return NULL;
}

// Get the body of the function at function_index.
DAG_node const *Generated_code_dag::get_function_body(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_body();
    }
    return NULL;
}

// Get the number of annotations of the material at material_index.
size_t Generated_code_dag::get_material_annotation_count(
    size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index))
        return mat->get_annotation_count();
    return 0;
}

// Get the annotation at annotation_index of the material at material_index.
DAG_node const *Generated_code_dag::get_material_annotation(
    size_t material_index,
    size_t annotation_index) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        if (annotation_index < mat->get_annotation_count()) {
            return mat->get_annotation(annotation_index);
        }
    }
    return NULL;
}

// Get the default initializer of the parameter at parameter_index
// of the material at material_index.
DAG_node const *Generated_code_dag::get_material_parameter_default(
    size_t material_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index))
        return param->get_default();
    return NULL;
}

// Get the number of annotations of the parameter at parameter_index
// of the material at material_index.
size_t Generated_code_dag::get_material_parameter_annotation_count(
    size_t material_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index))
        return param->get_annotation_count();
    return 0;
}

// Get the annotation at annotation_index of the parameter at parameter_index
// of the material at material_index.
DAG_node const *Generated_code_dag::get_material_parameter_annotation(
    size_t material_index,
    size_t parameter_index,
    size_t annotation_index) const
{
    if (Parameter_info const *param = get_mat_param_info(material_index, parameter_index)) {
        if (annotation_index < param->get_annotation_count()) {
            return param->get_annotation(annotation_index);
        }
    }
    return NULL;
}

// Get the number of temporaries used by the material at material_index.
size_t Generated_code_dag::get_material_temporary_count(
    size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        return mat->get_temporary_count();
    }
    return 0;
}

// Get the temporary at temporary_index used by the material at material_index.
DAG_node const *Generated_code_dag::get_material_temporary(
    size_t material_index,
    size_t temporary_index) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        if (temporary_index < mat->get_temporary_count()) {
            return mat->get_temporary(temporary_index);
        }
    }
    return NULL;
}

// Get the temporary name at temporary_index used by the material at material_index.
char const *Generated_code_dag::get_material_temporary_name(
    size_t material_index,
    size_t temporary_index) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        if (temporary_index < mat->get_temporary_count()) {
            return mat->get_temporary_name(temporary_index);
        }
    }
    return NULL;
}

// Get the value of the material at material_index.
DAG_node const *Generated_code_dag::get_material_value(
    size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index)) {
        return mat->get_body();
    }
    return NULL;
}

// Get the export flags of the material at material_index.
bool Generated_code_dag::get_material_exported(size_t material_index) const
{
    if (get_material_info(material_index) != NULL) {
        // currently, only exported materials are reported
        return true;
    }
    return false;
}

// Return the original material name of a cloned material or "" if the material
// is not a clone.
char const *Generated_code_dag::get_cloned_material_name(
    size_t material_index) const
{
    if (Material_info const *mat = get_material_info(material_index))
        return mat->get_cloned_name();
    return NULL;
}

// Constructor.
Generated_code_dag::Material_instance::Material_instance(
    IMDL        *mdl,
    IAllocator  *alloc,
    size_t      material_index,
    char const  *internal_space,
    bool        unsafe_math_optimizations)
: Base(alloc)
, m_builder(alloc)
, m_mdl(mi::base::make_handle_dup(impl_cast<MDL>(mdl)))
, m_arena(alloc)
, m_sym_tab(m_arena)
, m_type_factory(m_arena, mdl, &m_sym_tab)
, m_value_factory(m_arena, m_type_factory)
, m_node_factory(mdl, m_arena, m_value_factory, internal_space)
, m_messages(alloc, /*owner_fname=*/"")
, m_material_index(material_index)
, m_constructor(NULL)
, m_temporaries(alloc)
, m_default_param_values(alloc)
, m_param_names(alloc)
, m_hash()
, m_properties(0)
, m_referenced_scene_data(alloc)
, m_resource_tag_map(alloc)
, m_resource_tagger(m_resource_tag_map)
{
    m_node_factory.enable_unsafe_math_opt(unsafe_math_optimizations);

    memset(m_slot_hashes, 0, sizeof(m_slot_hashes));
}

// Acquires a const interface.
mi::base::IInterface const *Generated_code_dag::Material_instance::get_interface(
    mi::base::Uuid const &interface_id) const
{
    if (interface_id == IPrinter_interface::IID()) {
        return m_builder.create<Material_instance_printer>(m_builder.get_allocator());
    }
    return Base::get_interface(interface_id);
}

// Get the type factory of this instance.
IType_factory *Generated_code_dag::Material_instance::get_type_factory()
{
    return &m_type_factory;
}

// Get the value factory of this instance.
IValue_factory *Generated_code_dag::Material_instance::get_value_factory()
{
    return &m_value_factory;
}

// Create a constant.
DAG_constant const *Generated_code_dag::Material_instance::create_constant(
    IValue const *value)
{
    // Constants created here are used as arguments for material instances.
    // Do NOT do CSE here, because we don't want to combine equal constants into the same
    // argument IF they were created with different create_constant() calls.
    // Note: if would be slightly better to switch CSE only off in class compilation mode...
    bool old = m_node_factory.enable_cse(false);
    DAG_constant const *res = m_node_factory.create_constant(value);
    m_node_factory.enable_cse(old);
    return res;
}

// Create a temporary constant for the IR visitor.
DAG_constant const *Generated_code_dag::Material_instance::create_temp_constant(
    IValue const *value)
{
    // Temporary constants are used by the visitor to visit a value.
    // Always do CSE here, or the constants are accumulated.
    bool old = m_node_factory.enable_cse(true);
    DAG_constant const *res = m_node_factory.create_constant(value);
    m_node_factory.enable_cse(old);
    return res;
}

// Find the tag for a given resource.
int Generated_code_dag::Material_instance::find_resource_tag(
    IValue_resource const *res) const
{
    return m_resource_tagger.get_resource_tag(res);
}

// Adds a tag, version pair for a given resource.
void Generated_code_dag::Material_instance::add_resource_tag(
    IValue_resource const *res,
    int                   tag)
{
    size_t l = m_resource_tag_map.size();
    m_resource_tag_map.resize(l + 1);

    ISymbol const *shared = m_sym_tab.get_shared_symbol(res->get_string_value());
    m_resource_tag_map[l] = Resource_tag_tuple(
        kind_from_value(res), shared->get_name(), tag);
}

// Create a call.
DAG_node const *Generated_code_dag::Material_instance::create_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    int                           num_call_args,
    IType const                   *ret_type)
{
    // Note: we already assured that all constants are different, hence
    // any created call will be different ... No need to switch off CSE here ...
    return m_node_factory.create_call(
        name, sema, call_args, num_call_args, ret_type);
}

// Create a parameter reference.
DAG_parameter const *Generated_code_dag::Material_instance::create_parameter(
    IType const *type,
    int         index)
{
    MDL_ASSERT(!"should not be called");
    return NULL;
}

// Add a temporary.
int Generated_code_dag::Material_instance::add_temporary(DAG_node const *value) {
    int count = m_temporaries.size();
    m_temporaries.push_back(value);
    return count;
}

namespace {

/// Helper class to simplify NULL modifier handling.
class Null_resource_modifer : public IResource_modifier
{
public:
    /// Just return the resource unmodified
    ///
    /// \param res    the resource value to replace
    ///
    /// \return a new value of the same type or res if no modification is necessary
    IValue_resource const *modify(
        IValue_resource const *res,
        IModule         const *,
        IValue_factory        &) MDL_FINAL
    {
        return res;
    }
};

Null_resource_modifer null_modifier;

}  // anonymous

// Initialize a material instance.
Generated_code_dag::Error_code Generated_code_dag::Material_instance::initialize(
    ICall_name_resolver       *resolver,
    IResource_modifier        *resource_modifier,
    IGenerated_code_dag const *code_dag,
    size_t                    argc,
    DAG_node const            *argv[],
    bool                      use_temporaries,
    unsigned                  flags,
    ICall_evaluator           *evaluator,
    bool                      fold_meters_per_scene_unit,
    float                     mdl_meters_per_scene_unit,
    float                     wavelength_min,
    float                     wavelength_max,
    char const * const        fold_params[],
    size_t                    num_fold_params)
{
#if 0
    {
        char buffer[64];
        static unsigned idx = 0;

        snprintf(buffer, sizeof(buffer), "_%04u_ARGS", idx++);

        Generated_code_dag const *dag = impl_cast<Generated_code_dag>(code_dag);
        dag->dump_material_dag(m_material_index, buffer, argc, argv);
    }
#endif

    size_t parameter_count = code_dag->get_material_parameter_count(m_material_index);
    if (argc < parameter_count)
        return EC_TOO_FEW_ARGUMENTS;
    if (parameter_count < argc)
        return EC_TOO_MANY_ARGUMENTS;

    Type_binder type_binder(get_allocator(), m_type_factory);

    // check parameters
    for (int i = 0; i < argc; ++i) {
        if (DAG_node const *arg = argv[i]) {
            Error_code ec = check_argument(arg);
            if (ec != EC_NONE)
                return ec;

            // check if the argument type matches the material parameter type
            IType const *arg_type   = arg->get_type();
            IType const *p_type     = code_dag->get_material_parameter_type(m_material_index, i);
            IType::Modifiers p_mods = p_type->get_type_modifiers();

            // import the parameter type into the type factory of this material
            p_type = m_type_factory.import(p_type);

            arg_type = arg_type->skip_type_alias();
            p_type   = p_type->skip_type_alias();

            if (is<IType_array>(arg_type) && is<IType_array>(p_type)) {
                // both are arrays, it is allowed to pass a immediate size array to an
                // deferred size array
                IType_array const *a_arg_type = cast<IType_array>(arg_type);
                IType_array const *a_p_type   = cast<IType_array>(p_type);

                // check if it is already bound
                a_p_type = type_binder.get_bound_type(a_p_type);

                if (a_arg_type->is_immediate_sized()) {
                    if (a_p_type->is_immediate_sized()) {
                        // both are immediate sized, must be identical
                        arg_type = m_type_factory.get_equal(arg_type);
                        if (arg_type != p_type->skip_type_alias()) {
                            // types does not match
                            return EC_ARGUMENT_TYPE_MISMATCH;
                        }
                        // else fine
                        continue;
                    } else {
                        IType const *e_p_type   = a_p_type->get_element_type()->skip_type_alias();
                        IType const *e_arg_type = a_arg_type->get_element_type()->skip_type_alias();

                        e_arg_type = m_type_factory.get_equal(e_arg_type);

                        if (e_arg_type != e_p_type) {
                            // element types does not match
                            return EC_ARGUMENT_TYPE_MISMATCH;
                        }

                        // Otherwise the argument type will be bound to this parameter type.
                        // Note that it might not exists yet in our type factory, hence import it.
                        arg_type = m_type_factory.import(arg_type);

                        type_binder.bind_param_type(a_p_type, cast<IType_array>(arg_type));
                        continue;
                    }
                }
                // otherwise array types does not match
                return EC_ARGUMENT_TYPE_MISMATCH;
            } else {
                // non-array types
                arg_type = m_type_factory.get_equal(arg_type->skip_type_alias());
                if (arg_type != p_type->skip_type_alias()) {
                    // types does not match
                    return EC_ARGUMENT_TYPE_MISMATCH;
                }
            }
            if (p_mods & IType::MK_UNIFORM) {
                // parameter is uniform, check if the expression is
                // this is necessary because the API currently does not compute the
                // uniform property right when the argument expression is created
                IType const *real_arg_type = compute_type(*resolver, arg);

                if (is<IType_error>(real_arg_type)) {
                    // error inside the argument expression already handled
                }
                if (p_mods & IType::MK_UNIFORM) {
                    // material parameter must be uniform
                    IType::Modifiers a_mods = real_arg_type->get_type_modifiers();

                    if (a_mods & IType::MK_VARYING) {
                        // error on material parameter
                        Position_impl zero(0, 0, 0, 0);
                        string msg(get_allocator());

                        msg  = "uniform parameter '";
                        msg += code_dag->get_material_parameter_name(m_material_index, i);
                        msg += "' of material '";
                        msg += code_dag->get_material_name(m_material_index);
                        msg += "' got varying attachment";

                        warning(Generated_code_dag::VARYING_ON_UNIFORM, zero, msg.c_str());
                    }
                }
            }
        } else {
            // NULL arguments are not allowed
            return EC_INSTANTIATION_ERROR;
        }
    }

    if (resource_modifier == NULL)
        resource_modifier = &null_modifier;

    Generated_code_dag const *dag = impl_cast<Generated_code_dag>(code_dag);

    DAG_mangler dag_mangler(get_allocator(), m_mdl.get());
    DAG_builder dag_builder(get_allocator(), m_node_factory, dag_mangler);


    // set the resource modifier here, so inlining will modify resources
    dag_builder.set_resource_modifier(resource_modifier);

    Instantiate_helper creator(
        *resolver,
        *resource_modifier,
        dag,
        dag_builder,
        m_material_index,
        flags,
        evaluator,
        argc,
        argv,
        fold_meters_per_scene_unit,
        mdl_meters_per_scene_unit,
        wavelength_min,
        wavelength_max,
        fold_params,
        num_fold_params);

    DAG_call const *constructor = creator.compile();
    set_constructor(constructor);
    m_default_param_values = creator.get_default_parameter_values();
    m_param_names          = creator.get_parameter_names();

    // set properties
    Instantiate_helper::Properties props = creator.get_properties();

    set_property(IP_DEPENDS_ON_TRANSFORM,           0 != (props & IP_DEPENDS_ON_TRANSFORM));
    set_property(IP_DEPENDS_ON_OBJECT_ID,           0 != (props & IP_DEPENDS_ON_OBJECT_ID));
    set_property(IP_DEPENDS_ON_GLOBAL_DISTRIBUTION,
        0 != (props & IP_DEPENDS_ON_GLOBAL_DISTRIBUTION));
    set_property(IP_USES_TERNARY_OPERATOR,          0 != (props & IP_USES_TERNARY_OPERATOR));
    set_property(IP_USES_TERNARY_OPERATOR_ON_DF,    0 != (props & IP_USES_TERNARY_OPERATOR_ON_DF));
    set_property(IP_CLASS_COMPILED,                 0 != (flags & CLASS_COMPILATION));
    set_property(IP_DEPENDS_ON_UNIFORM_SCENE_DATA,
        0 != (props & IP_DEPENDS_ON_UNIFORM_SCENE_DATA));

    m_referenced_scene_data.insert(
        m_referenced_scene_data.end(),
        creator.get_referenced_scene_data().begin(),
        creator.get_referenced_scene_data().end());

    // make sure, the observable state is deterministic
    std::sort(m_referenced_scene_data.begin(), m_referenced_scene_data.end());

    Error_code res = EC_NONE;
    if ((flags & CLASS_COMPILATION) == 0) {
        if (!check_thin_walled_material())
            res = EC_WRONG_TRANSMISSION_ON_THIN_WALLED;
    }

    if (use_temporaries)
        build_temporaries();

    // add all resource entries from the code DAG
    for (size_t i = 0, n = code_dag->get_resource_tag_map_entries_count(); i < n; ++i) {
        Resource_tag_tuple const *t = code_dag->get_resource_tag_map_entry(i);
        m_resource_tag_map.push_back(*t);
    }

    calc_hashes();

#if 0
    {
        char buffer[64];
        static unsigned idx = 0;

        snprintf(buffer, sizeof(buffer), "_%04u", idx++);


        std::string material_name(code_dag->get_material_name(m_material_index));
        std::replace(material_name.begin(), material_name.end(), ':', '_');

        material_name += buffer;

        if (flags & CLASS_COMPILATION)
            material_name += "_class";
        else
            material_name += "_inst";

        dump_instance_dag(material_name.c_str());
    }
#endif

    return res;
}

// Return the material constructor.
DAG_call const *Generated_code_dag::Material_instance::get_constructor() const
{
    return m_constructor;
}

// Return the number of temporaries.
size_t Generated_code_dag::Material_instance::get_temporary_count() const
{
    return m_temporaries.size();
}

// Get the value of the temporary at index.
DAG_node const *Generated_code_dag::Material_instance::get_temporary_value(size_t index) const
{
    if (m_temporaries.size() <= index)
        return NULL;
    return m_temporaries[index];
}

// Return the number of parameters of this instance.
size_t Generated_code_dag::Material_instance::get_parameter_count() const
{
    return m_default_param_values.size();
}

/// Return the default value of a parameter of this instance.
IValue const *Generated_code_dag::Material_instance::get_parameter_default(size_t index) const
{
    if (m_default_param_values.size() <= index)
        return NULL;
    return m_default_param_values[index];
}

// Return the hash value of this material instance.
DAG_hash const *Generated_code_dag::Material_instance::get_hash() const
{
    return &m_hash;
}

// Return the hash value of one material slot of this material instance.
DAG_hash const *Generated_code_dag::Material_instance::get_slot_hash(Slot slot) const
{
    return &m_slot_hashes[slot];
}

// Return the canonical parameter name of the given parameter.
char const *Generated_code_dag::Material_instance::get_parameter_name(size_t index) const
{
    if (m_param_names.size() <= index)
        return NULL;
    return m_param_names[index].c_str();
}

// Returns true if this instance depends on object transforms.
bool Generated_code_dag::Material_instance::depends_on_transform() const
{
    return (get_properties() & IP_DEPENDS_ON_TRANSFORM) != 0;
}

// Returns true if this instance depends on the object id.
bool Generated_code_dag::Material_instance::depends_on_object_id() const
{
    return (get_properties() & IP_DEPENDS_ON_OBJECT_ID) != 0;
}

// Returns true if this instance depends on the global distribution (edf).
bool Generated_code_dag::Material_instance::depends_on_global_distribution() const
{
    return (get_properties() & IP_DEPENDS_ON_GLOBAL_DISTRIBUTION) != 0;
}

// Returns true if this instance depends on uniform scene data.
bool Generated_code_dag::Material_instance::depends_on_uniform_scene_data() const
{
    return (get_properties() & IP_DEPENDS_ON_UNIFORM_SCENE_DATA) != 0;
}

// Returns the number of scene data attributes referenced by this instance.
size_t Generated_code_dag::Material_instance::get_referenced_scene_data_count() const
{
    return m_referenced_scene_data.size();
}

// Return the name of a scene data attribute referenced by this instance.
char const *Generated_code_dag::Material_instance::get_referenced_scene_data_name(
    size_t index) const
{
    if (m_referenced_scene_data.size() <= index)
        return NULL;

    return m_referenced_scene_data[index].c_str();
}

class Instance_cloner {
public:
    typedef Generated_code_dag::Material_instance Material_instance;

    /// Constructor.
    ///
    /// \param alloc  an allocator
    Instance_cloner(
        IAllocator *alloc)
    : m_alloc(alloc)
    , m_marker_map(0, Visited_node_map::hasher(), Visited_node_map::key_equal(), alloc)
    , m_node_factory(NULL)
    , m_type_factory(NULL)
    , m_value_factory(NULL)
    {
    }

    /// Clone an instance.
    ///
    /// \param src              the instance to be cloned
    /// \param flags            flags for cloning
    /// \param unsafe_math_opt  enable unsafe math optimizations
    Generated_code_dag::Material_instance *clone(
        Material_instance const        *src,
        Material_instance::Clone_flags flags,
        bool                           unsafe_math_op);

private:
    /// Creates a (deep) copy of a node.
    ///
    /// \param node  the root node of the DAG to copy
    DAG_node const *copy_dag(
        DAG_node const *node);

private:
    /// The allocator.
    IAllocator *m_alloc;

    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Visited_node_map;

    /// The marker map for walking DAGs.
    Visited_node_map m_marker_map;

    /// The node factory of the destination instance.
    DAG_node_factory_impl *m_node_factory;

    /// The type factory of the destination instance.
    IType_factory         *m_type_factory;

    /// The value factory of the destination instance.
    IValue_factory        *m_value_factory;
};

// Creates a (deep) copy of a node.
DAG_node const *Instance_cloner::copy_dag(
    DAG_node const *node)
{
restart:
    Visited_node_map::const_iterator it = m_marker_map.find(node);
    if (it != m_marker_map.end()) {
        // already processed
        return it->second;
    }

    DAG_node const *original_node = node;

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c     = cast<DAG_constant>(node);
            IValue const       *value = c->get_value();
            node = m_node_factory->create_constant(m_value_factory->import(value));
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *p    = cast<DAG_parameter>(node);
            IType const         *type = p->get_type();
            int                 index = p->get_index();
            node = m_node_factory->create_parameter(m_type_factory->import(type), index);
        }
        break;
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *t = cast<DAG_temporary>(node);
            node = t->get_expr();
            goto restart;
        }
    case DAG_node::EK_CALL:
        {
            DAG_call const         *call    = cast<DAG_call>(node);
            int                    n_params = call->get_argument_count();
            IDefinition::Semantics sema     = call->get_semantic();
            char const             *name    = call->get_name();
            IType const            *type    = m_type_factory->import(call->get_type());

            VLA<DAG_call::Call_argument> args(m_alloc, n_params);

            for (int i = 0; i < n_params; ++i) {
                args[i].param_name = call->get_parameter_name(i);
                args[i].arg        = copy_dag(call->get_argument(i));
            }

            node = m_node_factory->create_call(
                name, sema, args.data(), args.size(), type);
        }
        break;
    }
    m_marker_map.insert(Visited_node_map::value_type(original_node, node));
    return node;
}

// Clone an instance.
Generated_code_dag::Material_instance *Instance_cloner::clone(
    Material_instance const        *src,
    Material_instance::Clone_flags flags,
    bool                           unsafe_math_opt)
{
    Allocator_builder builder(m_alloc);

    mi::base::Handle<IMDL> mdl(src->get_mdl());

    Generated_code_dag::Material_instance *curr =
        builder.create<Generated_code_dag::Material_instance>(
        mdl.get(),
        m_alloc,
        src->get_material_index(),
        src->get_node_factory().get_internal_space(),
        unsafe_math_opt);

    Store<DAG_node_factory_impl *> nf(m_node_factory,  curr->get_node_factory());
    Store<IType_factory *>         tf(m_type_factory,  curr->get_type_factory());
    Store<IValue_factory *>        vf(m_value_factory, curr->get_value_factory());

    Clear_scope<Visited_node_map>  mm(m_marker_map);

    Option_store<DAG_node_factory_impl, bool> optimization(
        *m_node_factory, &DAG_node_factory_impl::enable_opt,
        flags & Material_instance::CF_ENABLE_OPT);

    // Note: no temporaries after the copy
    DAG_node const *root = copy_dag(src->get_constructor());

    curr->set_constructor(cast<DAG_call>(root));

    curr->m_material_index       = src->m_material_index;
    curr->m_default_param_values = src->m_default_param_values;
    for (size_t i = 0, n = curr->m_default_param_values.size(); i < n; ++i) {
        curr->m_default_param_values[i] = m_value_factory->import(curr->m_default_param_values[i]);
    }
    curr->m_param_names = src->m_param_names;
    curr->m_properties  = src->m_properties;

    if (flags & Material_instance::CF_RECALC_HASH) {
        curr->calc_hashes();
    } else {
        curr->m_hash                 = src->m_hash;
        for (size_t i = 0; i < sizeof(curr->m_slot_hashes)/sizeof(curr->m_slot_hashes[0]); ++i) {
            curr->m_slot_hashes[i] = src->m_slot_hashes[i];
        }
    }

    return curr;
}

// Creates a clone of a this material instance.
Generated_code_dag::Material_instance *Generated_code_dag::Material_instance::clone(
    IAllocator  *alloc,
    Clone_flags flags,
    bool        unsafe_math_opt) const
{
    Instance_cloner cloner(alloc);

    return cloner.clone(this, flags, unsafe_math_opt);
}

// Dump the material expression DAG.
void Generated_code_dag::Material_instance::dump_instance_dag(char const *name) const
{
    string fname(name, get_allocator());
    fname += "_DAG.gv";

    for (size_t i = 0, n = fname.size(); i < n; ++i) {
        char c = fname[i];
        if (c == ':' || c == '/' || c == '\\')
            fname[i] = '_';
    }

    if (FILE *f = fopen(fname.c_str(), "w")) {
        Allocator_builder builder(get_allocator());

        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(get_allocator(), f, /*close_at_destroy=*/true));

        Instance_dumper dumper(get_allocator(), *this, out.get());

        dumper.dump();
    }
}


// Create a material instance.
IGenerated_code_dag::IMaterial_instance *Generated_code_dag::create_material_instance(
    size_t                          index,
    IGenerated_code_dag::Error_code *error_code) const
{
    size_t material_count = get_material_count();
    if (material_count <= index) {
        if (error_code)
            *error_code = EC_INVALID_INDEX;
        return NULL;
    }
    if (get_material_value(index) == NULL) {
        if (error_code)
            *error_code = EC_MATERIAL_HAS_ERROR;
        return NULL;
    }

    Material_instance *result = m_builder.create<Material_instance>(
        m_mdl.get(),
        m_builder.get_allocator(),
        index,
        m_internal_space.c_str(),
        (m_options & UNSAFE_MATH_OPTIMIZATIONS) != 0);
    if (error_code)
        *error_code = EC_NONE;
    return result;
}

// Get the number of exported types.
size_t Generated_code_dag::get_type_count() const
{
    return m_user_types.size();
}

// Get the name of the type at index.
char const *Generated_code_dag::get_type_name(
    size_t index) const
{
    if (User_type_info const *type = get_type_info(index)) {
        return type->get_name();
    }
    return NULL;
}

// Get the original name of the type at index  if the type name is an alias..
char const *Generated_code_dag::get_original_type_name(
    size_t index) const
{
    if (User_type_info const *type = get_type_info(index)) {
        return type->get_original_name();
    }
    return NULL;
}

// Get the user type at index.
IType const *Generated_code_dag::get_type(
    size_t index) const
{
    if (User_type_info const *type = get_type_info(index)) {
        return type->get_type();
    }
    return NULL;
}

// Returns true if the type at index is exported.
bool Generated_code_dag::is_type_exported(
    size_t index) const
{
    if (User_type_info const *type = get_type_info(index)) {
        return type->is_exported();
    }
    return false;
}

// Get the number of annotations of the type at index.
size_t Generated_code_dag::get_type_annotation_count(
    size_t index) const
{
    if (User_type_info const *type = get_type_info(index)) {
        return type->get_annotation_count();
    }
    return 0;
}

// Get the annotation at annotation_index of the type at type_index.
DAG_node const *Generated_code_dag::get_type_annotation(
    size_t type_index,
    size_t annotation_index) const
{
    if (User_type_info const *type = get_type_info(type_index)) {
        if (annotation_index < type->get_annotation_count()) {
            return type->get_annotation(annotation_index);
        }
    }
    return NULL;
}

// Get the number of type sub-entities (fields or enum constants).
size_t Generated_code_dag::get_type_sub_entity_count(
    size_t type_index) const
{
    if (User_type_info const *type = get_type_info(type_index)) {
        return type->get_entity_count();
    }
    return 0;
}

// Get the number of type sub-entities (fields or enum constants).
char const *Generated_code_dag::get_type_sub_entity_name(
    size_t type_index,
    size_t entity_index) const
{
    if (User_type_info const *type = get_type_info(type_index)) {
        IType const *u_tp = type->get_type();

        switch (u_tp->get_kind()) {
        case IType::TK_ALIAS:
            // no sub entities
            return NULL;
        case IType::TK_STRUCT:
            {
                // return number of fields
                IType_struct const *s_type = cast<IType_struct>(u_tp);

                if (entity_index < 0 || s_type->get_field_count() <= entity_index)
                    return NULL;

                IType const *f_type = NULL;
                ISymbol const *f_sym = NULL;
                s_type->get_field(entity_index, f_type, f_sym);

                return f_sym->get_name();
            }
        case IType::TK_ENUM:
            {
                // return number of enum values
                IType_enum const *e_type = cast<IType_enum>(u_tp);

                if (e_type->get_value_count() <= entity_index)
                    return NULL;

                ISymbol const *v_sym = NULL;
                int code = 0;
                e_type->get_value(entity_index, v_sym, code);

                return v_sym->get_name();
            }
        default:
            MDL_ASSERT(!"unknown type kind");
            return NULL;
        }
    }
    return NULL;
}

// Get the type of a type sub-entity (field or enum constant).
IType const *Generated_code_dag::get_type_sub_entity_type(
    size_t type_index,
    size_t entity_index) const
{
    if (User_type_info const *type = get_type_info(type_index)) {
        IType const *u_tp = type->get_type();

        switch (u_tp->get_kind()) {
        case IType::TK_ALIAS:
            // no sub entities
            return NULL;
        case IType::TK_STRUCT:
            {
                // return number of fields
                IType_struct const *s_type = cast<IType_struct>(u_tp);

                if (s_type->get_field_count() <= entity_index)
                    return NULL;

                IType const *f_type = NULL;
                ISymbol const *f_sym = NULL;
                s_type->get_field(entity_index, f_type, f_sym);

                return f_type;
            }
        case IType::TK_ENUM:
            return NULL;
        default:
            MDL_ASSERT(!"unknown type kind");
            return NULL;
        }
    }
    return NULL;
}

// Get the number of annotations of a type sub-entity (field or enum constant) at index.
size_t Generated_code_dag::get_type_sub_entity_annotation_count(
    size_t type_index,
    size_t entity_index) const
{
    if (User_type_info const *type = get_type_info(type_index)) {
        if (size_t(entity_index) < type->get_entity_count()) {
            return type->get_entity(entity_index).get_annotation_count();
        }
    }
    return 0;
}

// Get the annotation at annotation_index of the type sub-entity at (type_index, entity_index).
DAG_node const *Generated_code_dag::get_type_sub_entity_annotation(
    size_t type_index,
    size_t entity_index,
    size_t annotation_index) const
{
    if (User_type_info const *type = get_type_info(type_index)) {
        if (entity_index < type->get_entity_count()) {
            User_type_info::Entity_info const &ent = type->get_entity(entity_index);

            if (annotation_index < ent.get_annotation_count())
                return ent.get_annotation(annotation_index);
        }
    }
    return NULL;
}

// Get the number of exported constants.
size_t Generated_code_dag::get_constant_count() const
{
    return m_user_constants.size();
}

// Get the name of the constant at index.
char const *Generated_code_dag::get_constant_name(
    size_t index) const
{
    if (Constant_info const *con = get_constant_info(index)) {
        return con->get_name();
    }
    return NULL;
}

// Get the value of the constant at index.
DAG_constant const *Generated_code_dag::get_constant_value(
    size_t index) const
{
    if (Constant_info const *con = get_constant_info(index)) {
        return con->get_value();
    }
    return NULL;
}

// Get the number of annotations of the constant at index.
size_t Generated_code_dag::get_constant_annotation_count(
    size_t index) const
{
    if (Constant_info const *con = get_constant_info(index)) {
        return con->get_annotation_count();
    }
    return 0;
}

// Get the annotation at annotation_index of the constant at constant_index.
DAG_node const *Generated_code_dag::get_constant_annotation(
    size_t constant_index,
    size_t annotation_index) const
{
    if (Constant_info const *con = get_constant_info(constant_index)) {
        if (annotation_index < con->get_annotation_count()) {
            return con->get_annotation(annotation_index);
        }
    }
    return NULL;
}

// Build temporaries by traversing the DAG and creating them for nodes with phen-out > 1.
void Generated_code_dag::Material_instance::build_temporaries()
{
    /// Helper class: creates temporaries for node when phen-out > 1.
    class Temporary_inserter : public Abstract_temporary_inserter
    {
    public:
        /// Constructor.
        ///
        /// \param instance      the material instance
        /// \param phen_outs     the phen-out map for the visited expression DAG
        /// \param temp_name_map the desired temporary names
        Temporary_inserter(
            Material_instance &instance,
            Phen_out_map const &phen_outs,
            Temporary_name_map const &temp_name_map)
        : Abstract_temporary_inserter(
            instance.get_allocator(),
            *instance.get_node_factory(),
            phen_outs,
            temp_name_map)
        , m_instance(instance)
        {
        }

        /// Create and register a new temporary.
        ///
        /// \param node  the initializer for the temporary
        int add_temporary(DAG_node const *node, char const *name) MDL_FINAL
        {
            return m_instance.add_temporary(node);
        }

    private:
        /// The DAG.
        Material_instance &m_instance;
    };

    // we will modify the identify table, so clear it here
    MDL_ASSERT(m_node_factory.get_temp_name_map().empty());
    m_node_factory.identify_clear();

    Phen_out_map phen_outs(0, Phen_out_map::hasher(), Phen_out_map::key_equal(), get_allocator());

    DAG_ir_walker walker(get_allocator(), /*as_tree=*/false);
    Calc_phen_out phen_counter(phen_outs);

    walker.walk_instance(this, &phen_counter);

    // empty name map since we do not want to keep names of let expressions for material instances
    // (the map in the factory should be empty anyway, see assertion above)
    Abstract_temporary_inserter::Temporary_name_map temporary_names(get_allocator());
    Temporary_inserter inserter(*this, phen_outs, temporary_names);

    walker.walk_instance(this, &inserter);
}

// Calculate the hash values for this instance.
void Generated_code_dag::Material_instance::calc_hashes()
{
    MD5_hasher md5_hasher;
    Dag_hasher dag_hasher(md5_hasher);

    // Important: Walk as Tree here
    DAG_ir_walker walker(get_allocator(), /*as_tree=*/true);

    for (int i = 0; i <= MS_LAST; ++i) {
        walker.walk_instance_slot(this, Slot(i), &dag_hasher);

        md5_hasher.final(m_slot_hashes[i].data());
    }

    for (int i = 0; i <= MS_LAST; ++i) {
        md5_hasher.update(m_slot_hashes[i].data(), m_slot_hashes[i].size());
    }
    md5_hasher.final(m_hash.data());
}

// Check instance argument for restrictions.
Generated_code_dag::Error_code Generated_code_dag::Material_instance::check_argument(
    DAG_node const *arg) const
{
    switch (arg->get_kind()) {
    case DAG_node::EK_CONSTANT:
        break;
    case DAG_node::EK_TEMPORARY:
        MDL_ASSERT(!"Given argument is a temporary");
        return EC_INSTANTIATION_ERROR;
    case DAG_node::EK_CALL:
        {
            DAG_call const *c = cast<DAG_call>(arg);

            for (int i = 0, n = c->get_argument_count(); i < n; ++i) {
                DAG_node const *arg = c->get_argument(i);

                Error_code ec = check_argument(arg);
                if (ec != EC_NONE)
                    return ec;
            }
        }
        break;
    case DAG_node::EK_PARAMETER:
        MDL_ASSERT(!"Given argument is a parameter");
        return EC_INSTANTIATION_ERROR;
    }
    return EC_NONE;
}

/// Check if the given call is uniform.
///
/// \param call  the DAG call
/// \param def   the definition of the called function if any
///
/// \note that special DAG functions don't have a definition
static bool is_uniform_call(
    DAG_call const    *call,
    IDefinition const *def)
{
    // handle first those without a MDL definition
    IDefinition::Semantics sema = call->get_semantic();
    switch (sema) {
    case IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        // More complicated case: theoretically, the result might be uniform
        // even if the argument is varying. But we return the property of the operator
        // itself here, so it is always uniform
        return true;

    case IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
    case IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
    case IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
        // these are always uniform
        return true;

    default:
        MDL_ASSERT(!is_DAG_semantics(sema) && "DAG semantic not handled");
        if (semantic_is_operator(sema)) {
            // operators are (except the field select operator) always uniform
            return true;
        }
        break;
    }

    MDL_ASSERT(def != NULL && "non-DAG_semantics function should have a definition");

    // Note: don't use IS_UNIFORM here, it is not consistently set on the std library, because
    // it was not annotated there and the analysis did not enter it because of missing bodies
    return def != NULL && !def->get_property(mi::mdl::IDefinition::DP_IS_VARYING);
}

// Compute the type of a expression taking uniform rules into account.
IType const *Generated_code_dag::Material_instance::compute_type(
    ICall_name_resolver &resolver,
    DAG_node const      *arg)
{
    IType const *t = arg->get_type();
    t = m_type_factory.import(t);

    switch (arg->get_kind()) {
    case DAG_node::EK_CONSTANT:
        // constants are always uniform
        return m_type_factory.create_alias(t, NULL, IType::MK_UNIFORM);
    case DAG_node::EK_TEMPORARY:
        MDL_ASSERT(!"Given argument is a temporary");
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(arg);

            char const *signature = call->get_name();
            mi::base::Handle<mi::mdl::IModule const> mod(resolver.get_owner_module(signature));
            if (!mod.is_valid_interface()) {
                MDL_ASSERT(!"Cannot resolve call");
                return m_type_factory.create_error();
            }
            Module const *owner = impl_cast<mi::mdl::Module>(mod.get());

            IDefinition const *def = owner->find_signature(signature, /*only_exported=*/false);
            bool              uniform_call = is_uniform_call(call, def);

            IType_function const *f_type = NULL;
            if (def != NULL) {
                f_type = cast<IType_function>(def->get_type());

                MDL_ASSERT(f_type->get_parameter_count() == call->get_argument_count());
            }

            bool auto_is_uniform = true;
            for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
                DAG_node const *arg = call->get_argument(i);

                IType const *a_type = compute_type(resolver, arg);
                if (is<IType_error>(a_type)) {
                    // already error
                    return a_type;
                }

                if (f_type != NULL) {
                    ISymbol const  *p_sym  = NULL;
                    IType const    *p_type = NULL;

                    f_type->get_parameter(i, p_type, p_sym);

                    IType::Modifiers p_mods = p_type->get_type_modifiers();
                    IType::Modifiers a_mods = a_type->get_type_modifiers();

                    if (p_mods & IType::MK_UNIFORM) {
                        // argument must be uniform
                        if ((a_mods & IType::MK_UNIFORM) == 0) {
                            // bad, non-uniform argument to uniform parameter
                            MDL_ASSERT(a_mods & IType::MK_VARYING);

                            Position_impl zero(0, 0, 0, 0);
                            string msg(get_allocator());

                            msg  = "uniform parameter '";
                            msg += call->get_parameter_name(i);
                            msg += "' of '";
                            msg += signature;
                            msg += "' got varying attachment";

                            warning(Generated_code_dag::VARYING_ON_UNIFORM, zero, msg.c_str());
                        }
                    } else if ((p_mods & IType::MK_VARYING) == 0) {
                        // auto parameter
                        if (a_mods & IType::MK_VARYING) {
                            // varying argument to auto-parameter
                            auto_is_uniform = false;
                        } else {
                            MDL_ASSERT(a_mods & IType::MK_UNIFORM);
                        }
                    } else {
                        // varying parameter can take everything
                    }
                } else if (uniform_call) {
                    IType::Modifiers a_mods = a_type->get_type_modifiers();
                    if (a_mods & IType::MK_VARYING) {
                        // varying argument to auto-parameter
                        auto_is_uniform = false;
                    } else {
                        MDL_ASSERT(a_mods & IType::MK_UNIFORM);
                    }
                }
            }

            // Note: don't use IS_UNIFORM here, it is not consistently set on the std library,
            // because it was not annotated there and the analysis did not enter it because of
            // missing bodies
            IType::Modifier result_mod = IType::MK_UNIFORM;
            if (uniform_call) {
                // the result of uniform functions depend on auto parameters
                if (!auto_is_uniform) {
                    // not all auto parameters are uniform => varying result
                    result_mod = IType::MK_VARYING;
                }
            } else {
                // varying functions have varying result
                result_mod = IType::MK_VARYING;
            }
            return m_type_factory.create_alias(t->skip_type_alias(), NULL, result_mod);
        }
        break;
    case DAG_node::EK_PARAMETER:
        MDL_ASSERT(!"Given argument is a parameter");
        break;
    }
    return m_type_factory.create_error();
}

// Access messages.
Messages const &Generated_code_dag::Material_instance::access_messages() const
{
    return m_messages;
}

// Get the instance properties.
Generated_code_dag::Material_instance::Properties
Generated_code_dag::Material_instance::get_properties() const
{
    return m_properties;
}

// Get the internal space.
char const *Generated_code_dag::Material_instance::get_internal_space() const
{
    return m_node_factory.get_internal_space();
}

// Set a tag, version pair for a resource constant that might be reachable from this
// instance.
void Generated_code_dag::Material_instance::set_resource_tag(
    IValue_resource const *res,
    int                   tag)
{
    if (res->get_tag_value() != 0) {
        MDL_ASSERT(res->get_tag_value() == tag && "trying to overwrite a set tag value");
        return;
    }

    int old_tag = find_resource_tag(res);

    if (old_tag == 0) {
        add_resource_tag(res, tag);
    } else {
        MDL_ASSERT(old_tag == tag && "trying to overwrite a set tag value");
    }
}

// Get the number of resource map entries.
size_t Generated_code_dag::Material_instance::get_resource_tag_map_entries_count() const
{
    return m_resource_tag_map.size();
}

// Get the i'th resource tag tag map entry or NULL if the index is out of bounds;
Resource_tag_tuple const *Generated_code_dag::Material_instance::get_resource_tag_map_entry(
    size_t index) const
{
    if (index < m_resource_tag_map.size())
        return &m_resource_tag_map[index];
    return NULL;
}

// Get the resource tagger for this code DAG.
IResource_tagger *Generated_code_dag::Material_instance::get_resource_tagger() const
{
    return &m_resource_tagger;
}

// Creates a new error message.
void Generated_code_dag::Material_instance::error(
    int code, Err_location const &loc, char const *msg)
{
    size_t fname_id = 0; // always current file
    m_messages.add_error_message(code, MESSAGE_CLASS, fname_id, loc.get_position(), msg);
}

// Creates a new warning message.
void Generated_code_dag::Material_instance::warning(
    int code, Err_location const &loc, char const *msg)
{
    size_t fname_id = 0; // always current file
    m_messages.add_warning_message(code, MESSAGE_CLASS, fname_id, loc.get_position(), msg);
}

// ----------------------------- Instantiate_helper -----------------------------

// Constructor.
Generated_code_dag::Material_instance::Instantiate_helper::Instantiate_helper(
    ICall_name_resolver      &resolver,
    IResource_modifier       &resource_modifier,
    Generated_code_dag const *code_dag,
    DAG_builder              &dag_builder,
    int                      material_index,
    unsigned                 flags,
    ICall_evaluator          *evaluator,
    size_t                   argc,
    DAG_node const           *argv[],
    bool                     fold_meters_per_scene_unit,
    float                    mdl_meters_per_scene_unit,
    float                    wavelength_min,
    float                    wavelength_max,
    char const * const       fold_params[],
    size_t                   num_fold_params)
: m_resolver(resolver)
, m_resource_modifier(resource_modifier)
, m_code_dag(*code_dag)
, m_arena(dag_builder.get_allocator())
, m_dag_builder(dag_builder)
, m_node_factory(dag_builder.get_node_factory())
, m_value_factory(*m_node_factory.get_value_factory())
, m_type_factory(*m_value_factory.get_type_factory())
, m_old_evaluator(m_node_factory.get_call_evaluator())
, m_flags(flags)
, m_argc(argc)
, m_argv(argv)
, m_material_index(material_index)
, m_params(0)
, m_visit_map(0, Visit_map::hasher(), Visit_map::key_equal(), &m_arena)
, m_replacement_map(0, Replacement_map::hasher(), Replacement_map::key_equal(), &m_arena)
, m_resource_param_map(
    0, Resource_param_map::hasher(), Resource_param_map::key_equal(), &m_arena)
, m_default_param_values(get_allocator())
, m_param_names(get_allocator())
, m_curr_param_name(get_allocator())
, m_cache(
    0, Dep_analysis_cache::hasher(), Dep_analysis_cache::key_equal(), get_allocator())
, m_properties(0)
, m_referenced_scene_data(dag_builder.get_allocator())
, m_instantiate_args(flags & CLASS_COMPILATION)
, m_fold_params(get_allocator())
{
    // reset the CSE table, we will build new expressions
    m_node_factory.identify_clear();

    // set the call evaluator if any
    m_node_factory.set_call_evaluator(evaluator);

    // enable folding of unit conversion if requested
    if (fold_meters_per_scene_unit)
        m_node_factory.enable_unit_conv_fold(mdl_meters_per_scene_unit);

    // enable folding of state::wavelength_[min|max]
    m_node_factory.enable_wavelength_fold(wavelength_min, wavelength_max);

    // convert names of parameters to be folded
    for (size_t i = 0; i < num_fold_params; ++i) {
        m_fold_params.insert(string(fold_params[i], get_allocator()));
    }
}

// Destructor.
Generated_code_dag::Material_instance::Instantiate_helper::~Instantiate_helper()
{
    // reset the call evaluator if any
    m_node_factory.set_call_evaluator(m_old_evaluator);
}

DAG_node const *Generated_code_dag::Material_instance::Instantiate_helper::skip_temporaries(
    DAG_node const *expr)
{
    if (DAG_temporary const *temp = as<DAG_temporary>(expr)) {
         expr = temp->get_expr();
    }
    return expr;
}

DAG_node const *Generated_code_dag::Material_instance::Instantiate_helper::get_value(
    IValue const *value, Array_ref<char const *> const &path)
{
    for (size_t i = 0, n = path.size(); i < n; ++i) {
        IValue_struct const *s_value = cast<IValue_struct>(value);
        value = s_value->get_value(path[i]);
    }

    return m_node_factory.create_constant(value);
}

DAG_node const *Generated_code_dag::Material_instance::Instantiate_helper::get_value(
    DAG_node const *expr, Array_ref<char const *> const &path)
{
    for (size_t i = 0, n = path.size(); i < n; ++i) {
        expr = skip_temporaries(expr);

        while (DAG_parameter const *p = as<DAG_parameter>(expr)) {
            expr = m_argv[p->get_index()];
            expr = skip_temporaries(expr);
        }

        if (DAG_constant const *c = as<DAG_constant>(expr)) {
            IValue const *v = c->get_value();
            return get_value(v, path.slice(i));
        }

        if (DAG_call const *call = as<DAG_call>(expr)) {
            expr = call->get_argument(path[i]);
            if (expr == NULL)
                return NULL;
            continue;
        }

        MDL_ASSERT(!"wrong DAG node type");
        return NULL;
    }

    return expr;
}

// Fold geometry.cutout_opacity if in class-compilation mode, requested via flags,
// and evaluates to 0.0f or 1.0f.
void Generated_code_dag::Material_instance::Instantiate_helper::handle_cutout_opacity()
{
    if (!m_instantiate_args || (m_flags & NO_TRIVIAL_CUTOUT_OPACITY) == 0)
        return;

    DAG_node const *constructor = m_code_dag.get_material_value(m_material_index);

    static char const * const path[] = { "geometry", "cutout_opacity" };
    DAG_node const *cutout_opacity = get_value(constructor, path);

    // We might not find the path if there are calls in between that are similar to the copy
    // constructor, but without such a semantic.
    if (!cutout_opacity)
        return;

    // Instantiate cutout_opacity in instance compilation mode.
    DAG_node const *folded_cutout_opacity;
    Visit_map old_visit_map(m_visit_map);
    {
        Flag_store store(m_instantiate_args, false);
        folded_cutout_opacity = instantiate_dag(cutout_opacity);

        // No parameters should have been created by the call above in instance compilation mode.
        MDL_ASSERT(m_default_param_values.empty());
        MDL_ASSERT(m_param_names.empty());
    }

    // Nothing to do if the instantiation of cutout_opacity did not result in a constant.
    if (!is<DAG_constant>(folded_cutout_opacity)) {
        m_visit_map = std::move(old_visit_map);
        return;
    }

    DAG_constant const *c = cast<DAG_constant>(folded_cutout_opacity);

    float value = cast<IValue_float>(c->get_value())->get_value();

    // Nothing to do if the value of cutout_opacity is not 0.0f or 1.0f.
    if (value != 0.0f && value != 1.0f) {
        m_visit_map = std::move(old_visit_map);
        return;
    }

    // Remove all entries from m_visit_map that are not in old_visit_map, are not the node for
    // cutout_opacity itself, nor any of the parameters encountered during folding of that node.
    // Such parameters are kept in the map to avoid that they are folded here, but not for other
    // uses.
    Arena_ptr_hash_set<DAG_node const>::Type keep_set(&m_arena);
    for (auto const& o: old_visit_map)
        keep_set.insert(o.first);
    keep_set.insert(cutout_opacity);
    for (size_t i = 0; i < m_argc; ++i)
        keep_set.insert(m_argv[i]);
    for (Visit_map::iterator it = m_visit_map.begin(); it != m_visit_map.end(); ) {
        if (keep_set.count(it->first) == 0) {
            it = m_visit_map.erase(it);
        } else {
            ++it;
        }
    }
}

class Transparent_layers : public IDAG_ir_visitor {
public:
    Transparent_layers(
        Generated_code_dag::Material_instance::Instantiate_helper &instantiate_helper)
      : m_instantiate_helper(instantiate_helper) { }

    void visit(DAG_constant *cnst) { }
    void visit(DAG_temporary *tmp) { }
    void visit(DAG_call *call) { m_instantiate_helper.handle_transparent_layers(call); }
    void visit(DAG_parameter *param) { }
    void visit(int index, DAG_node *init) { }

private:
    Generated_code_dag::Material_instance::Instantiate_helper& m_instantiate_helper;
};

void Generated_code_dag::Material_instance::Instantiate_helper::handle_transparent_layers()
{
    if (!m_instantiate_args || (m_flags & NO_TRANSPARENT_LAYERS) == 0)
        return;

    Transparent_layers tl(*this);
    DAG_ir_walker walker(get_allocator(), /*as_tree*/ true);
    walker.walk_material(const_cast<Generated_code_dag*>(&m_code_dag), m_material_index, &tl);
}

void Generated_code_dag::Material_instance::Instantiate_helper::handle_transparent_layers(DAG_call const *call)
{
    // Extract properties from relevant layering functions.
    bool float_weight = true;
    int index_weight  = -1;
    int index_layer   = -1;
    int index_base    = -1;

    IDefinition::Semantics sema = call->get_semantic();
    switch (sema) {

        case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:        
            float_weight = false;
        case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
            index_weight = 0;
            index_layer  = 1;
            index_base   = 2;
            break;

        case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
            float_weight = false;
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
            index_weight = 1;
            index_layer  = 2;
            index_base   = 3;
            break;

        case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
            float_weight = false;
        case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
            index_weight = 3;
            index_layer  = 4;
            index_base   = 5;
            break;

        case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            float_weight = false;
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
            index_weight = 1;
            index_layer  = 2;
            index_base   = 3;
            break;

        // Nothing to do for other functions.
        default:
            return;
    }
    
    MDL_ASSERT(index_weight != -1 && index_layer != -1 && index_base != -1);

    // Nothing to do if the layer argument is not one of the qualified BSDFs.
    DAG_node const *arg_layer = call->get_argument(index_layer);
    Replacement_map::const_iterator it = m_replacement_map.find(arg_layer);
    if (it != m_replacement_map.end())
        arg_layer = it->second;
    if (!is_layer_qualified(arg_layer))
        return;

    // Instantiate the weight argument in instance compilation mode.
    DAG_node const *arg_weight = call->get_argument(index_weight);
    DAG_node const *folded_weight;
    Visit_map old_visit_map(m_visit_map);
    {
        Flag_store store(m_instantiate_args, false);
        folded_weight = instantiate_dag(arg_weight);

        // No parameters should have been created by the call above in instance compilation mode.
        MDL_ASSERT(m_default_param_values.empty());
        MDL_ASSERT(m_param_names.empty());
    }

    // Nothing to do if the instantiation of arg_weight did not result in a constant.
    if (!is<DAG_constant>(folded_weight)) {
        m_visit_map = std::move(old_visit_map);
        return;
    }

    DAG_constant const *c = cast<DAG_constant>(folded_weight);

    // Nothing to do if the value of arg_weight is not 0.0f or color(0.0f).
    if (float_weight) {
        IValue_float const *value = cast<IValue_float>(c->get_value());
        if (!value->is_zero()) {
            m_visit_map = std::move(old_visit_map);
            return;
        }
    } else {
        IValue_rgb_color const *value = cast<IValue_rgb_color>(c->get_value());
        if (!value->is_zero()) {
            m_visit_map = std::move(old_visit_map);
            return;
        }
    }

    // Replace call by arg_base when traversing the DAG later.
    DAG_node const *arg_base = call->get_argument(index_base);
    it = m_replacement_map.find(arg_base);
    if (it != m_replacement_map.end())
        arg_base = it->second;
    m_replacement_map[call] = arg_base;

    // Remove all entries from m_visit_map that are not in old_visit_map, nor any of the parameters
    // encountered during folding of the weight. Such parameters are kept in the map to avoid that
    // they are folded here, but not for other uses.
    Arena_ptr_hash_set<DAG_node const>::Type keep_set(&m_arena);
    for (auto const& o: old_visit_map)
        keep_set.insert(o.first);
    for (size_t i = 0; i < m_argc; ++i)
        keep_set.insert(m_argv[i]);
    for (Visit_map::iterator it = m_visit_map.begin(); it != m_visit_map.end(); ) {
        if (keep_set.count(it->first) == 0) {
            it = m_visit_map.erase(it);
        } else {
            ++it;
        }
    }
}

bool Generated_code_dag::Material_instance::Instantiate_helper::is_layer_qualified(DAG_node const *expr)
{
    switch (expr->get_kind())
    {
        case DAG_node::EK_CONSTANT:
            return false;

        case DAG_node::EK_TEMPORARY: {
            DAG_temporary const *temp = as<DAG_temporary>(expr);
            return is_layer_qualified(temp->get_expr());
        }

        case DAG_node::EK_PARAMETER: {
            DAG_parameter const *parameter = as<DAG_parameter>(expr);
            return is_layer_qualified(m_argv[parameter->get_index()]);
        }

        case DAG_node::EK_CALL: {

            DAG_call const *call = as<DAG_call>(expr);
            IDefinition::Semantics sema = call->get_semantic();

            // Ternary operators are qualified if both true and false expression are qualified.
            if (sema == operator_to_semantic(IExpression::OK_TERNARY))
                return is_layer_qualified(call->get_argument(1))
                    && is_layer_qualified(call->get_argument(2));

            // Extract scatter mode from relevant BSDFs.
            int index_scatter_mode = -1;
            switch (sema)
            {
                case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
                    return true; // No scatter mode paramter.
                case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
                    index_scatter_mode = 1;
                    break;
                case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
                    index_scatter_mode = 5;
                    break;
                case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
                    index_scatter_mode = 5;
                    break;
                case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
                    index_scatter_mode = 5;
                    break;
                case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
                    index_scatter_mode = 5;
                    break;
                case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
                    index_scatter_mode = 5;
                    break;
                default:
                    return false;
            }

            MDL_ASSERT(index_scatter_mode != -1);

            // Not qualified if scatter_mode is not a constant.
            DAG_node const *arg_scatter_mode = call->get_argument(index_scatter_mode);
            DAG_constant const *arg_scatter_mode_constant = cast<DAG_constant>(arg_scatter_mode);
            if (!arg_scatter_mode_constant)
                return false;

            // Not qualified if scatter_mode is not scatter_transmit or scatter_reflect_transmit.
            IValue const *arg_scatter_mode_value = arg_scatter_mode_constant->get_value();
            IValue_enum const *arg_scatter_mode_enum = cast<IValue_enum>(arg_scatter_mode_value);
            int value = arg_scatter_mode_enum->get_value();
            if (value != df::scatter_transmit && value != df::scatter_reflect_transmit)
                return false;        

            return true;
        }
    }

    MDL_ASSERT(!"Unsupported DAG node kind");
    return false;
}
    
// Compile the material.
DAG_call const *
Generated_code_dag::Material_instance::Instantiate_helper::compile()
{
    bool old_ignore_noinline = false;
    bool old_inline = false;

    // Ignore anno::noinline(), if requested
    if ((m_flags & mi::mdl::IGenerated_code_dag::IMaterial_instance::IGNORE_NOINLINE) != 0) {
        old_ignore_noinline = m_node_factory.enable_ignore_noinline(true);
    } else {
        // Otherwise, deactivate inlining in general: We want instantiation as fast as possible and
        // the material bodies were inlined during material (class) compilation.
        old_inline = m_node_factory.enable_inline(false);
    }

    // unfortunately we don't have the module of our material here, so retrieve it from the
    // name resolver
    mi::base::Handle<IModule const> mod(
        m_resolver.get_owner_module(m_code_dag.get_material_name(m_material_index)));
    Module_scope scope(m_dag_builder, mod.get());

    handle_cutout_opacity();
    handle_transparent_layers();

    DAG_node const *node = instantiate_dag(m_code_dag.get_material_value(m_material_index));

    m_visit_map.clear();
    m_resource_param_map.clear();

    if (m_params > 0) {
        // ensure that every parameter is used AFTER the optimization, if not, renumber
        node = renumber_parameter(node);
    }

    // Restore old node factory settings
    if ((m_flags & mi::mdl::IGenerated_code_dag::IMaterial_instance::IGNORE_NOINLINE) != 0) {
        m_node_factory.enable_ignore_noinline(old_ignore_noinline);
    } else {
        m_node_factory.enable_inline(old_inline);
    }

    return cast<DAG_call>(node);
}

// Inline parameters into a DAG expression.
DAG_node const *
Generated_code_dag::Material_instance::Instantiate_helper::inline_parameters(
    DAG_node const *node,
    DAG_parameter  *inline_params[])
{
    Parameter_inliner inliner(
        get_allocator(), m_node_factory, inline_params, &m_default_param_values[0]);

    return inliner.inline_parameters(node);
}

// Check that every parameter is still used after optimization and remove
// dead ones.
DAG_node const *
Generated_code_dag::Material_instance::Instantiate_helper::renumber_parameter(
    DAG_node const *node)
{
    size_t n_params = m_params;
    VLA<DAG_parameter *> params(get_allocator(), 2 * n_params);

    for (size_t i = 0; i < 2 * n_params; ++i)
        params[i] = NULL;

    // visit and collect
    DAG_parameter **live_params   = &params[0];
    DAG_parameter **inline_params = &params[n_params];

    // collect the life parameters and those that must be inlined because of
    // unresolved ternary operators
    bool arg_inline = (m_flags & NO_ARGUMENT_INLINE) == 0;
    Parameter_collector collector(get_allocator(), arg_inline, live_params, inline_params);

    collector.collect(const_cast<DAG_node *>(node));

    if (collector.need_parameter_inline()) {
        // parameters must be inlined due to unresolved ternary operator

        // kill those parameters that will be inlined from the live list
        for (size_t i = 0; i < n_params; ++i) {
            if (inline_params[i] != NULL) {
                // this parameter must be inlined
                live_params[i] = NULL;
            }
        }
        node = inline_parameters(node, inline_params);
    }

    // do the renumbering
    m_params = 0;
    for (size_t i = 0; i < n_params; ++i) {
        if (DAG_parameter *param = live_params[i]) {
            int param_idx = m_params++;

            set_parameter_index(param, param_idx);
            m_default_param_values[param_idx] = m_default_param_values[i];
            m_param_names[param_idx]          = m_param_names[i];
        }
    }
    m_default_param_values.resize(m_params);
    m_param_names.resize(m_params, string("", get_allocator()));

    return node;
}

namespace {

typedef Generated_code_dag::Material_instance::Dep_analysis_cache  Dep_analysis_cache;
typedef Generated_code_dag::Material_instance::Dependence_result   Dependence_result;

/// Helper class to analyze function ASTs.
class Ast_analysis : public Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param owner    the owner module of the analyzed function
    /// \param cache    the result cache to be used
    Ast_analysis(
        IAllocator         *alloc,
        IModule const      *owner,
        Dep_analysis_cache &cache)
    : m_alloc(alloc)
    , m_owner(owner)
    , m_cache(cache)
    , m_depends_on_transform(false)
    , m_depends_on_object_id(false)
    , m_edf_global_distribution(false)
    , m_depends_on_uniform_scene_data(false)
    , m_referenced_scene_data(alloc)
    {
    }

    /// Post visit of an call
    IExpression *post_visit(IExpression_call *call) MDL_FINAL
    {
        // assume the AST error free
        IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());

        if (ref->is_array_constructor()) {
            return call;
        }

        IDefinition const *def = ref->get_definition();

        switch (def->get_semantics()) {
        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_EDF:
            break;
        case IDefinition::DS_INTRINSIC_DF_SPOT_EDF:
            analyze_spot_edf(call);
            break;
        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
            analyze_measured_edf(call);
            break;
        case IDefinition::DS_UNKNOWN:
            analyze_unknown_call(call);
            break;
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
            m_depends_on_transform = true;
            break;
        case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
            m_depends_on_object_id = true;
            break;

        case IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
            {
                if (mi::mdl::IExpression_literal const *lit =
                        as<mi::mdl::IExpression_literal>(
                            call->get_argument(0)->get_argument_expr())) {
                    if (IValue_string const *name_str = as<IValue_string>(lit->get_value())) {
                        m_referenced_scene_data.insert(string(name_str->get_value(), m_alloc));
                    }
                }
                break;
            }

        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
        case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
            {
                m_depends_on_uniform_scene_data = true;
                if (mi::mdl::IExpression_literal const *lit =
                        as<mi::mdl::IExpression_literal>(
                            call->get_argument(0)->get_argument_expr())) {
                    if (IValue_string const *name_str = as<IValue_string>(lit->get_value())) {
                        m_referenced_scene_data.insert(string(name_str->get_value(), m_alloc));
                    }
                }
                break;
            }

        default:
            // all others have a known semantic and can be safely ignored.
            break;
        }
        return call;
    }

    /// Returns true if the analyzed entity depends on object transforms.
    bool depends_on_transform() const { return m_depends_on_transform; }

    /// Returns true if the analyzed entity depends on state::object_id().
    bool depends_on_object_id() const { return m_depends_on_object_id; }

    /// Returns true if the analyzed entity depends on any edf's global_distribution.
    bool depends_on_global_distribution() const { return m_edf_global_distribution; }

    /// Returns true if this instance depends on uniform scene data.
    bool depends_on_uniform_scene_data() const { return m_depends_on_uniform_scene_data; }

    /// Returns the set of scene data names referenced by the analyzed entity.
    Generated_code_dag::String_set const &referenced_scene_data() const {
        return m_referenced_scene_data;
    }

private:
    /// A constructor from parent.
    ///
    /// \param parent  the parent analysis
    /// \param owner   the owner of the entity to analyze
    Ast_analysis(IAllocator *alloc, Ast_analysis &parent, IModule const *owner)
    : m_alloc(alloc)
    , m_owner(owner)
    , m_cache(parent.m_cache)
    , m_depends_on_transform(false)
    , m_depends_on_object_id(false)
    , m_edf_global_distribution(false)
    , m_depends_on_uniform_scene_data(false)
    , m_referenced_scene_data(alloc)
    {
    }

    /// Analyze a call to a user defined function.
    void analyze_unknown_call(IExpression_call const *call)
    {
        IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());
        if (ref->is_array_constructor()) {
            return;
        }

        IDefinition const *def = ref->get_definition();

        analyze_function(def);
    }

    /// Analyze a user-defined function.
    ///
    /// \param def  the function's definition
    void analyze_function(IDefinition const *def)
    {
        Dep_analysis_cache::const_iterator it = m_cache.find(def);
        if (it != m_cache.end()) {
            Dependence_result const &res = it->second;

            m_depends_on_transform          |= res.m_depends_on_transform;
            m_depends_on_object_id          |= res.m_depends_on_object_id;
            m_edf_global_distribution       |= res.m_edf_global_distribution;
            m_depends_on_uniform_scene_data |= res.m_depends_on_uniform_scene_data;
            m_referenced_scene_data.insert(
                res.m_referenced_scene_data.begin(),
                res.m_referenced_scene_data.end());

        } else {
            mi::base::Handle<IModule const> owner(m_owner->get_owner_module(def));
            def = m_owner->get_original_definition(def);
            MDL_ASSERT(def != NULL && "Cannot lookup definition of called function");

            IDeclaration_function const *func = as<IDeclaration_function>(def->get_declaration());
            if (func == NULL) {
                // unexpected, cannot analyze further
                return;
            }

            Ast_analysis analysis(m_alloc, *this, owner.get());
            analysis.visit(func);

            // cache the result
            m_cache.emplace(def, Dependence_result(
                analysis.m_depends_on_transform,
                analysis.m_depends_on_object_id,
                analysis.m_edf_global_distribution,
                analysis.m_depends_on_uniform_scene_data,
                analysis.m_referenced_scene_data));

            m_depends_on_transform          |= analysis.m_depends_on_transform;
            m_depends_on_object_id          |= analysis.m_depends_on_object_id;
            m_edf_global_distribution       |= analysis.m_edf_global_distribution;
            m_depends_on_uniform_scene_data |= analysis.m_depends_on_uniform_scene_data;
            m_referenced_scene_data.insert(
                analysis.m_referenced_scene_data.begin(),
                analysis.m_referenced_scene_data.end());
        }
    }

    /// Analyze a call to df::spot_edf().
    void analyze_spot_edf(IExpression_call const *call)
    {
        bool global_distribution = true;
        switch (call->get_argument_count()) {
        default:
            MDL_ASSERT(!"Unsupported version of spot_edf()");
            break;
        case 4:
            {
                // pre MDL 1.1 version
                // export edf spot_edf (
                //     uniform float         exponent,
                //     uniform bool          global_distribution = true,
                //     uniform float3x3      global_frame = float3x3(1.0),
                //     uniform string        handle = "")
                IArgument const   *arg  = call->get_argument(1);
                IExpression const *expr = arg->get_argument_expr();

                if (IExpression_literal const *lit = as<IExpression_literal>(expr)) {
                    IValue_bool const *b = cast<IValue_bool>(lit->get_value());

                    global_distribution = b->get_value();
                }
            }
            break;
        case 5:
            {
                // MDL 1.1+ version
                //
                // export edf spot_edf (
                //     uniform float         exponent,
                //     uniform float         spread = math::PI,
                //     uniform bool          global_distribution = true,
                //     uniform float3x3      global_frame = float3x3(1.0),
                //     uniform string        handle = "")
                IArgument const   *arg  = call->get_argument(2);
                IExpression const *expr = arg->get_argument_expr();

                if (IExpression_literal const *lit = as<IExpression_literal>(expr)) {
                    IValue_bool const *b = cast<IValue_bool>(lit->get_value());

                    global_distribution = b->get_value();
                }
            }
            break;
        }
        if (global_distribution)
            m_edf_global_distribution = true;
    }

    /// Analyze a call to df::measured_edf().
    void analyze_measured_edf(IExpression_call const *call)
    {
        bool global_distribution = true;
        switch (call->get_argument_count()) {
        default:
            MDL_ASSERT(!"Unsupported version of spot_edf()");
            break;
        case 4:
            {
                // pre MDL 1.1 version
                //
                // export edf measured_edf (
                //     uniform light_profile profile,
                //     uniform bool          global_distribution = true,
                //     uniform float3x3      global_frame = float3x3(1.0),
                //     uniform string        handle = "")
                IArgument const   *arg  = call->get_argument(1);
                IExpression const *expr = arg->get_argument_expr();

                if (IExpression_literal const *lit = as<IExpression_literal>(expr)) {
                    IValue_bool const *b = cast<IValue_bool>(lit->get_value());

                    global_distribution = b->get_value();
                }
            }
            break;
        case 5:
        case 6:
            {
                // MDL 1.1 version
                //
                // export edf measured_edf(
                //     uniform light_profile profile,
                //     uniform float         multiplier          = 1.0,
                //     uniform bool          global_distribution = true,
                //     uniform float3x3      global_frame        = float3x3(1.0),
                //     uniform string        handle              = "")
                //
                // MDL 1.2+ version
                //
                // export edf measured_edf(
                //     uniform light_profile profile,
                //     uniform float         multiplier          = 1.0,
                //     uniform bool          global_distribution = true,
                //     uniform float3x3      global_frame        = float3x3(1.0),
                //     float3                tangent_u           = state::texture_tangent_u(0),
                //     uniform string        handle              = "")
                IArgument const   *arg  = call->get_argument(2);
                IExpression const *expr = arg->get_argument_expr();

                if (IExpression_literal const *lit = as<IExpression_literal>(expr)) {
                    IValue_bool const *b = cast<IValue_bool>(lit->get_value());

                    global_distribution = b->get_value();
                }
            }
            break;
        }
        if (global_distribution)
            m_edf_global_distribution = true;
    }

private:
    /// The allocator.
    IAllocator                        *m_alloc;

    /// The owner module of the analyzed function.
    IModule const                     *m_owner;

    /// The analysis result cache.
    Dep_analysis_cache                &m_cache;

    /// True if this instance depends on the object transforms.
    bool m_depends_on_transform;

    /// True if this instance depends of the object id.
    bool m_depends_on_object_id;

    /// True if this instance depends on global distribution (edf).
    bool m_edf_global_distribution;

    /// True if this instance depends on uniform scene data.
    bool m_depends_on_uniform_scene_data;

    /// Set of scene data names referenced by this instance.
    Generated_code_dag::String_set m_referenced_scene_data;
};

}  // anonymous

// Analyze a created call for dependencies.
void Generated_code_dag::Material_instance::Instantiate_helper::analyze_call(
    DAG_call const *call)
{
    switch (call->get_semantic()) {
    case IDefinition::DS_INTRINSIC_DF_DIFFUSE_EDF:
        break;
    case IDefinition::DS_INTRINSIC_DF_SPOT_EDF:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_EDF:
        set_property(IP_DEPENDS_ON_GLOBAL_DISTRIBUTION, true);
        break;
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        set_property(IP_DEPENDS_ON_TRANSFORM, true);
        break;
    case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
        set_property(IP_DEPENDS_ON_OBJECT_ID, true);
        break;

    case IDefinition::DS_INTRINSIC_SCENE_DATA_ISVALID:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
        {
            if (DAG_constant const *c = as<DAG_constant>(call->get_argument(0))) {
                if (IValue_string const *name_str = as<IValue_string>(c->get_value())) {
                    m_referenced_scene_data.insert(string(name_str->get_value(), get_allocator()));
                }
            }
            break;
        }

    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
        {
            set_property(IP_DEPENDS_ON_UNIFORM_SCENE_DATA, true);
            if (DAG_constant const *c = as<DAG_constant>(call->get_argument(0))) {
                if (IValue_string const *name_str = as<IValue_string>(c->get_value())) {
                    m_referenced_scene_data.insert(string(name_str->get_value(), get_allocator()));
                }
            }
            break;
        }

    case IDefinition::DS_UNKNOWN:
        {
            // user defined function

            char const *signature = call->get_name();
            mi::base::Handle<mi::mdl::IModule const> mod(m_resolver.get_owner_module(signature));
            if (!mod.is_valid_interface())
                break;
            Module const *owner = impl_cast<mi::mdl::Module>(mod.get());

            IDefinition const *def = owner->find_signature(signature, /*only_exported=*/false);
            if (def == NULL)
                break;

            analyze_function_ast(owner, def);
        }
        break;
    default:
        break;
    }
}

// Analyze a function AST for dependencies.
void Generated_code_dag::Material_instance::Instantiate_helper::analyze_function_ast(
    Module const *owner, IDefinition const *def)
{
    Dep_analysis_cache::const_iterator it = m_cache.find(def);
    if (it != m_cache.end()) {
        // already computed
        Dependence_result const &res = it->second;

        set_property(IP_DEPENDS_ON_TRANSFORM,           res.m_depends_on_transform);
        set_property(IP_DEPENDS_ON_OBJECT_ID,           res.m_depends_on_object_id);
        set_property(IP_DEPENDS_ON_GLOBAL_DISTRIBUTION, res.m_edf_global_distribution);
        set_property(IP_DEPENDS_ON_UNIFORM_SCENE_DATA,  res.m_depends_on_uniform_scene_data);
        m_referenced_scene_data.insert(
            res.m_referenced_scene_data.begin(),
            res.m_referenced_scene_data.end());
    } else {
        IDeclaration const *decl = def->get_declaration();
        if (decl == NULL)
            return;

        if (IDeclaration_function const *func = as<IDeclaration_function>(decl)) {
            Ast_analysis analysis(get_allocator(), owner, m_cache);

            analysis.visit(func);

            set_property(IP_DEPENDS_ON_TRANSFORM, analysis.depends_on_transform());
            set_property(IP_DEPENDS_ON_OBJECT_ID, analysis.depends_on_object_id());
            set_property(IP_DEPENDS_ON_GLOBAL_DISTRIBUTION,
                analysis.depends_on_global_distribution());
            set_property(IP_DEPENDS_ON_UNIFORM_SCENE_DATA,
                analysis.depends_on_uniform_scene_data());
            m_referenced_scene_data.insert(
                analysis.referenced_scene_data().begin(),
                analysis.referenced_scene_data().end());

            m_cache.emplace(def, Dependence_result(
                0 != (get_properties() & IP_DEPENDS_ON_TRANSFORM),
                0 != (get_properties() & IP_DEPENDS_ON_OBJECT_ID),
                0 != (get_properties() & IP_DEPENDS_ON_GLOBAL_DISTRIBUTION),
                0 != (get_properties() & IP_DEPENDS_ON_UNIFORM_SCENE_DATA),
                analysis.referenced_scene_data()));
        }
    }
}

// Check if we support instantiate_dag_arguments on this node.
bool
Generated_code_dag::Material_instance::Instantiate_helper::supported_arguments(
    DAG_node const *n)
{
    IType const *t = n->get_type();

    if (is<IType_df>(t)) {
        // do not create *df type parameters, nor promote the parameters of *df returning functions
        return false;
    }
    if (is_material_type_or_sub_type(t)) {
        if (is<DAG_constant>(n)) {
            // do NOT create material -types parameters
            return false;
        }
        if (DAG_call const *c = as<DAG_call>(n)) {
            if (c->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                // do not partially inline material or subtype constructors,
                return false;
            }
        }
    }
    return true;
}

// Instantiate a DAG expression.
DAG_node const *
Generated_code_dag::Material_instance::Instantiate_helper::instantiate_dag(
    DAG_node const *node)
{
    Replacement_map::const_iterator itr = m_replacement_map.find(node);
    if (itr != m_replacement_map.end())
        node = itr->second;

    Visit_map::const_iterator itv = m_visit_map.find(node);
    if (itv != m_visit_map.end())
        return itv->second;

    DAG_node const *res = NULL;

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c = cast<DAG_constant>(node);
            IValue const *v = c->get_value();
            v = m_value_factory.import(v);
            if (IValue_resource const *res = as<IValue_resource>(v)) {
                // maybe the resource value must be modified (aka, a TAG might be added
                v = m_resource_modifier.modify(res, m_dag_builder.tos_module(), m_value_factory);
            }
            res = m_node_factory.create_constant(v);
        }
        break;
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *tmp = cast<DAG_temporary>(node);
            int index = tmp->get_index();
            DAG_node const *init = m_code_dag.get_material_temporary(m_material_index, index);
            res = instantiate_dag(init);
        }
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(node);

            int n_args = call->get_argument_count();
            VLA<DAG_call::Call_argument> args(get_allocator(), n_args);

            res = NULL;

            bool args_processed = false;
            if (m_flags & NO_TERNARY_ON_DF) {
                if (call->get_semantic() == operator_to_semantic(IExpression::OK_TERNARY)) {
                    if (is<IType_df>(call->get_type()->skip_type_alias())) {
                        DAG_node const *cond = call->get_argument(0);

                        {
                            Flag_store store(m_instantiate_args, false);
                            cond = instantiate_dag(cond);
                        }

                        DAG_node const *t    = call->get_argument(1);
                        DAG_node const *f    = call->get_argument(2);

                        if (DAG_constant const *c = as<DAG_constant>(cond)) {
                            IValue_bool const *b_cond = cast<IValue_bool>(c->get_value());

                            if (b_cond->get_value()) {
                                res =  instantiate_dag(t);
                            } else {
                                res =  instantiate_dag(f);
                            }
                        } else {
                            args[0].arg        = cond;
                            args[0].param_name = call->get_parameter_name(0);
                            args[1].arg        = instantiate_dag(t);
                            args[1].param_name = call->get_parameter_name(1);
                            args[2].arg        = instantiate_dag(f);
                            args[2].param_name = call->get_parameter_name(2);
                        }

                        args_processed = true;
                    }
                }
            }

            string signature(call->get_name(), get_allocator());

            if (res == NULL) {
                if (!args_processed) {
                    for (int i = 0; i < n_args; ++i) {
                        args[i].arg = instantiate_dag(call->get_argument(i));
                        args[i].param_name = call->get_parameter_name(i);
                    }
                }

                if (m_node_factory.is_inline_allowed()) {
                    // basically this means we are inside an argument, see the parameter case
                    mi::base::Handle<IModule const> mod(
                        m_resolver.get_owner_module(signature.c_str()));

                    Module const *module = impl_cast<Module>(mod.get());
                    IDefinition const *def =
                        module->find_signature(signature.c_str(), /*only_exported=*/true);
                    if (def != NULL) {
                        if (def->get_property(IDefinition::DP_IS_IMPORTED)) {
                            // If this is an alias (imported/exported), then replace it by its
                            // original name. This does not help much, BUT the neuray material
                            // converter supports only the "original" names
                            // modify the signature to point to the original one
                            char const *old_mod_name = module->get_name();
                            size_t l = strlen(old_mod_name);
                            MDL_ASSERT(strncmp(signature.c_str(), old_mod_name, l) == 0);

                            char const *orig_module = module->get_owner_module_name(def);
                            signature = orig_module + signature.substr(l);
                        }
                    }

                    if (call->get_semantic() == IDefinition::DS_UNKNOWN) {
                        IDefinition const *def = NULL;
                        if (mod.is_valid_interface()) {
                            // beware, use old signature here, we retrieve the def again
                            def = module->find_signature(call->get_name(), /*only_exported=*/false);

                            if (def != NULL) {
                                // try to inline it
                                Module_scope module_scope(m_dag_builder, mod.get());

                                mi::base::Handle<IGenerated_code_dag const> owner_dag(
                                    m_resolver.get_owner_dag(signature.c_str()));

                                res = m_dag_builder.try_inline(
                                    owner_dag.get(), def, args.data(), n_args);

                                // must be analyzed when was inlined; do this here by analyzing the
                                // inlined function
                                if (res != NULL) {
                                    analyze_function_ast(module, def);
                                }
                            }
                        }
                    }
                }
            }

            if (res == NULL) {
                IType const *ret_type = call->get_type();
                ret_type = m_type_factory.import(ret_type);
                res = m_node_factory.create_call(
                    signature.c_str(), call->get_semantic(),
                    args.data(), args.size(), ret_type);

                if (DAG_call const *n_call = as<DAG_call>(res)) {
                    // still a call, check if it depends on the object
                    analyze_call(n_call);
                }
            }

            // check if its still a ternary operator and set flags accordingly
            if (DAG_call const *n_call = as<DAG_call>(res)) {
                if (n_call->get_semantic() == operator_to_semantic(IExpression::OK_TERNARY)) {
                    set_property(IP_USES_TERNARY_OPERATOR, true);
                    if (is<IType_df>(n_call->get_type()->skip_type_alias())) {
                        set_property(IP_USES_TERNARY_OPERATOR_ON_DF, true);
                    }
                }
            }
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            // Enable inlining inside the parameters.
            INLINE_scope inline_scope(m_node_factory);

            DAG_parameter const *para = cast<DAG_parameter>(node);
            int parameter_index = para->get_index();
            if (m_instantiate_args) {
                // inside class compilation: switch to argument compilation mode
                char const *p_name =
                    m_code_dag.get_material_parameter_name(m_material_index, parameter_index);

                Param_scope scope(*this, p_name, parameter_index);
                res = instantiate_dag_arguments(m_argv[parameter_index]);
            } else {
                // instance compilation, fold arguments completely
                res = instantiate_dag(m_argv[parameter_index]);
            }
        }
    }

    m_visit_map[node] = res;
    return res;
}

/// Check if the given type is the string type or contains the string type as a compound type.
///
/// \param tp  the type to check
static bool contains_string_type(IType const *tp)
{
restart:
    switch (tp->get_kind()) {
    case IType::TK_ALIAS:
        {
            IType_alias const *a_tp = cast<IType_alias>(tp);
            tp = a_tp->get_aliased_type();
            goto restart;
        }
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        return false;

    case IType::TK_STRING:
        return true;
    case IType::TK_ARRAY:
        {
            IType_array const *a_tp = cast<IType_array>(tp);
            tp = a_tp->get_element_type();
            goto restart;
        }
    case IType::TK_STRUCT:
        {
            IType_struct const *s_tp = cast<IType_struct>(tp);

            for (int i = 0, n = s_tp->get_field_count(); i < n; ++i) {
                IType const   *f_tp;
                ISymbol const *f_sym;

                s_tp->get_field(i, f_tp, f_sym);

                if (contains_string_type(f_tp))
                    return true;
            }
            return false;
        }
    case IType::TK_FUNCTION:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        // these types should not occur here
        break;
    }
    MDL_ASSERT(!"unexpected type kind");
    return true;
}

// Instantiate a DAG expression from an argument.
DAG_node const *
Generated_code_dag::Material_instance::Instantiate_helper::instantiate_dag_arguments(
    DAG_node const *node)
{
    Replacement_map::const_iterator itr = m_replacement_map.find(node);
    if (itr != m_replacement_map.end())
        node = itr->second;

    Visit_map::const_iterator itv = m_visit_map.find(node);
    if (itv != m_visit_map.end())
        return itv->second;

    DAG_node const *res = NULL;

    if (!supported_arguments(node))
        return instantiate_dag(node);

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *cnst = cast<DAG_constant>(node);
            IValue const       *v    = cnst->get_value();
            IType const        *t    = v->get_type();

            v = m_value_factory.import(v);
            IValue_resource const *resource = as<IValue_resource>(v);
            if (resource != NULL) {
                // maybe the resource value must be modified (aka, a TAG might be added
                v = resource = m_resource_modifier.modify(
                    resource, m_dag_builder.tos_module(), m_value_factory);
            }

            if ((m_flags & NO_STRING_PARAMS) != 0 && contains_string_type(t)) {
                // do not create parameters containing string values
                res = m_node_factory.create_constant(v);
            } else if ((m_flags & NO_BOOL_PARAMS) != 0 && is<IType_bool>(t)) {
                // do not create plain bool parameters
                res = m_node_factory.create_constant(v);
            } else if ((m_flags & NO_ENUM_PARAMS) != 0 && is<IType_enum>(t)) {
                // do not create plain enum parameters
                res = m_node_factory.create_constant(v);
            } else {
                // we reach a leave: create new argument(s) for it
                if ((m_flags & NO_RESOURCE_SHARING) == 0 && resource != NULL) {
                    // do not create different parameter for the same resource
                    Resource_param_map::const_iterator it = m_resource_param_map.find(resource);
                    if (it != m_resource_param_map.end()) {
                        res = it->second;
                    } else {
                        res = m_node_factory.create_parameter(t, m_params++);
                        m_default_param_values.push_back(v);
                        m_param_names.push_back(m_curr_param_name);

                        m_resource_param_map[resource] = res;
                    }
                } else if (m_fold_params.count(m_curr_param_name) > 0) {
                    // do not create parameter if explicitly disabled by name
                    res = m_node_factory.create_constant(v);
                } else {
                    // not a resource or sharing disabled, folding not explicitly requested by
                    // name: just create a new parameter
                    res = m_node_factory.create_parameter(t, m_params++);
                    m_default_param_values.push_back(v);
                    m_param_names.push_back(m_curr_param_name);
                }
            }
        }
        break;
    case DAG_node::EK_TEMPORARY:
        MDL_ASSERT(!"found a temporary inside a material argument");
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *call  = cast<DAG_call>(node);

            int n_args = call->get_argument_count();
            VLA<DAG_call::Call_argument> args(get_allocator(), n_args);
            for (int i = 0; i < n_args; ++i) {
                char const *param_name = call->get_parameter_name(i);

                Param_scope scope(*this, param_name, i);

                args[i].arg        = instantiate_dag_arguments(call->get_argument(i));
                args[i].param_name = param_name;
            }

            string signature(call->get_name(), get_allocator());
            res = NULL;

            if (m_node_factory.is_inline_allowed()) {
                // basically this means we are inside an argument, see the parameter case
                mi::base::Handle<IModule const> mod(m_resolver.get_owner_module(signature.c_str()));

                Module const *module = impl_cast<Module>(mod.get());
                IDefinition const *def =
                    module->find_signature(signature.c_str(), /*only_exported=*/true);
                if (def != NULL) {
                    if (def->get_property(IDefinition::DP_IS_IMPORTED)) {
                        // If this is an alias (imported/exported), then replace it by its
                        // original name. This does not help much, BUT the neuray material
                        // converter supports only the "original" names
                        // modify the signature to point to the original one
                        char const *old_mod_name = module->get_name();
                        size_t l = strlen(old_mod_name);
                        MDL_ASSERT(strncmp(signature.c_str(), old_mod_name, l) == 0);

                        char const *orig_module = module->get_owner_module_name(def);
                        signature = orig_module + signature.substr(l);
                    }
                }

                if (call->get_semantic() == IDefinition::DS_UNKNOWN) {
                    IDefinition const *def = NULL;
                    if (mod.is_valid_interface()) {
                        // beware, use old signature here, we retrieve the def again
                        def = module->find_signature(call->get_name(), /*only_exported=*/false);

                        if (def != NULL) {
                            // try to inline it
                            Module_scope module_scope(m_dag_builder, mod.get());

                            mi::base::Handle<IGenerated_code_dag const> owner_dag(
                                m_resolver.get_owner_dag(signature.c_str()));

                            res = m_dag_builder.try_inline(owner_dag.get(), def, args.data(), n_args);

                            // must be analyzed when was inlined; do this here by analyzing the
                            // inlined function
                            if (res != NULL) {
                                analyze_function_ast(module, def);
                            }
                        }
                    }
                }
            }

            if (res == NULL) {
                IType const *ret_type = call->get_type();
                ret_type = m_type_factory.import(ret_type);
                res = m_node_factory.create_call(
                    signature.c_str(), call->get_semantic(),
                    args.data(), args.size(), ret_type);

                if (DAG_call const *n_call = as<DAG_call>(res)) {
                    // still a call, check if it depends on the object
                    analyze_call(n_call);
                }
            }
        }
        break;
    case DAG_node::EK_PARAMETER:
        MDL_ASSERT(!"found a parameter reference inside a material argument");
        break;
    }

    m_visit_map[node] = res;
    return res;
}

// Access messages.
Messages &Generated_code_dag::access_messages()
{
    return m_messages;
}

bool has_dynamic_memory_consumption(Generated_code_dag::Parameter_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::Parameter_info const &param)
{
    return
        dynamic_memory_consumption(param.m_name) +
        dynamic_memory_consumption(param.m_type_name) +
        dynamic_memory_consumption(param.m_annotations);
}

bool has_dynamic_memory_consumption(Generated_code_dag::Material_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::Material_info const &mat)
{
    return
        dynamic_memory_consumption(mat.m_name) +
        dynamic_memory_consumption(mat.m_simple_name) +
        dynamic_memory_consumption(mat.m_original_name) +
        dynamic_memory_consumption(mat.m_cloned) +
        dynamic_memory_consumption(mat.m_parameters) +
        dynamic_memory_consumption(mat.m_annotations) +
        dynamic_memory_consumption(mat.m_temporaries) +
        dynamic_memory_consumption(mat.m_temporary_names);
}

bool has_dynamic_memory_consumption(Generated_code_dag::Function_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::Function_info const &func)
{
    return
        dynamic_memory_consumption(func.m_name) +
        dynamic_memory_consumption(func.m_simple_name) +
        dynamic_memory_consumption(func.m_original_name) +
        dynamic_memory_consumption(func.m_cloned) +
        dynamic_memory_consumption(func.m_parameters) +
        dynamic_memory_consumption(func.m_annotations) +
        dynamic_memory_consumption(func.m_temporaries) +
        dynamic_memory_consumption(func.m_temporary_names) +
        dynamic_memory_consumption(func.m_return_annos) +
        dynamic_memory_consumption(func.m_refs);
}

bool has_dynamic_memory_consumption(Generated_code_dag::Annotation_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::Annotation_info const &func)
{
    return
        dynamic_memory_consumption(func.m_name) +
        dynamic_memory_consumption(func.m_simple_name) +
        dynamic_memory_consumption(func.m_original_name) +
        dynamic_memory_consumption(func.m_parameters) +
        dynamic_memory_consumption(func.m_annotations);
}

bool has_dynamic_memory_consumption(Generated_code_dag::User_type_info::Entity_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::User_type_info::Entity_info const &ent)
{
    return dynamic_memory_consumption(ent.m_annotations);
}

bool has_dynamic_memory_consumption(Generated_code_dag::User_type_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::User_type_info const &type)
{
    return
        dynamic_memory_consumption(type.m_name) +
        dynamic_memory_consumption(type.m_original_name) +
        dynamic_memory_consumption(type.m_annotations) +
        dynamic_memory_consumption(type.m_entities);
}

bool has_dynamic_memory_consumption(Generated_code_dag::Constant_info const &)
{
    return true;
}

size_t dynamic_memory_consumption(Generated_code_dag::Constant_info const &con)
{
    return
        dynamic_memory_consumption(con.m_name) +
        dynamic_memory_consumption(con.m_annotations);
}

// Returns the amount of used memory by this code DAG.
size_t Generated_code_dag::get_memory_size() const
{
    size_t res = sizeof(*this);

    res += m_arena.get_chunks_size();
    res += dynamic_memory_consumption(m_messages);
    res += dynamic_memory_consumption(m_module_imports);

    res += dynamic_memory_consumption(m_module_annotations);
    res += dynamic_memory_consumption(m_functions);
    res += dynamic_memory_consumption(m_materials);
    res += dynamic_memory_consumption(m_user_types);
    res += dynamic_memory_consumption(m_user_constants);
    res += dynamic_memory_consumption(m_internal_space);

    return res;
}

// Get the export flags of the function at function_index.
bool Generated_code_dag::get_function_property(
    size_t            function_index,
    Function_property fp) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return (func->get_properties() & (1 << fp)) != 0;
    }
    return false;
}

// Get the number of entities referenced by a function.
size_t Generated_code_dag::get_function_references_count(size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_ref_count();
    }
    return 0;
}

// Get the signature of the i'th reference of a function
char const *Generated_code_dag::get_function_reference(
    size_t function_index,
    size_t callee_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        if (callee_index < func->get_ref_count()) {
            return func->get_ref(callee_index).c_str();
        }
    }
    return NULL;
}

// Return the original function name of a cloned function or "" if the function
// is not a clone.
char const *Generated_code_dag::get_cloned_function_name(
    size_t function_index) const
{
    if (Function_info const *func = get_function_info(function_index)) {
        return func->get_cloned_name();
    }
    return NULL;
}

// Get the number of annotations of the module.
size_t Generated_code_dag::get_module_annotation_count() const
{
    return m_module_annotations.size();
}

// Get the annotation at annotation_index of the module.
DAG_node const *Generated_code_dag::get_module_annotation(
    size_t annotation_index) const
{
    if (annotation_index < m_module_annotations.size()) {
        return m_module_annotations[annotation_index];
    }
    return NULL;
}

// Get the internal space.
char const *Generated_code_dag::get_internal_space() const
{
    return m_internal_space.c_str();
}

// Get the number of annotations in the generated code.
size_t Generated_code_dag::get_annotation_count() const
{
    return m_annotations.size();
}

// Get the semantics of the annotation at annotation_index.
IDefinition::Semantics Generated_code_dag::get_annotation_semantics(
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return anno->get_semantics();
    }
    return IDefinition::DS_UNKNOWN;
}

// Get the name of the annotation at annotation_index.
char const *Generated_code_dag::get_annotation_name(
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return anno->get_name();
    }
    return NULL;
}

// Get the simple name of the annotation at annotation_index.
char const *Generated_code_dag::get_simple_annotation_name(
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return anno->get_simple_name();
    }
    return NULL;
}

// Get the original name of the annotation at annotation_index if the annotation name is
// an alias, i.e. re-exported from a module.
char const *Generated_code_dag::get_original_annotation_name(
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return anno->get_original_name();
    }
    return NULL;
}

// Get the parameter count of the annotation at annotation_index.
size_t Generated_code_dag::get_annotation_parameter_count(
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return anno->get_parameter_count();
    }
    return 0;
}

// Get the parameter type of the parameter at parameter_index
// of the annotation at annotation_index.
IType const *Generated_code_dag::get_annotation_parameter_type(
    size_t annotation_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_anno_param_info(annotation_index, parameter_index)) {
        return param->get_type();
    }
    return NULL;
}

/// Get the parameter type name of the parameter at parameter_index
/// of the annotation at annotation_index.
char const *Generated_code_dag::get_annotation_parameter_type_name(
    size_t annotation_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_anno_param_info(annotation_index, parameter_index)) {
        return param->get_type_name();
    }
    return NULL;
}

// Get the parameter name of the parameter at parameter_index
// of the annotation at annotation_index.
char const *Generated_code_dag::get_annotation_parameter_name(
    size_t annotation_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_anno_param_info(annotation_index, parameter_index)) {
        return param->get_name();
    }
    return NULL;
}

// Get the index of the parameter parameter_name.
size_t Generated_code_dag::get_annotation_parameter_index(
    size_t     annotation_index,
    char const *parameter_name) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        for (size_t i = 0, n = anno->get_parameter_count(); i < n; ++i) {
            if (strcmp(parameter_name, anno->get_parameter(i).get_name()) == 0)
                return i;
        }
    }
    return ~size_t(0);
}

// Get the default initializer of the parameter at parameter_index
// of the annotation at annotation_index.
DAG_node const *Generated_code_dag::get_annotation_parameter_default(
    size_t annotation_index,
    size_t parameter_index) const
{
    if (Parameter_info const *param = get_anno_param_info(annotation_index, parameter_index)) {
        return param->get_default();
    }
    return NULL;
}

// Get the property flag of the annotation at annotation_index.
bool Generated_code_dag::get_annotation_property(
    size_t              annotation_index,
    Annotation_property ap) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return (anno->get_properties() & (1 << ap)) != 0;
    }
    return false;

}

// Get the number of annotations of the annotation at annotation_index.
size_t Generated_code_dag::get_annotation_annotation_count(
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(annotation_index)) {
        return anno->get_annotation_count();
    }
    return 0;
}

// Get the annotation at annotation_index of the annotation (declaration) at anno_decl_index.
DAG_node const *Generated_code_dag::get_annotation_annotation(
    size_t anno_decl_index,
    size_t annotation_index) const
{
    if (Annotation_info const *anno = get_annotation_info(anno_decl_index)) {
        if (annotation_index < anno->get_annotation_count())
            return anno->get_annotation(annotation_index);
    }
    return NULL;
}

// Get a tag,for a resource constant that might be reachable from this DAG.
int Generated_code_dag::get_resource_tag(
    IValue_resource const *res) const
{
    int tag = find_resource_tag(res);
    if (tag == 0) {
        tag = res->get_tag_value();
    }
    return tag;
}

// Set a tag, version pair for a resource constant that might be reachable from this DAG.
void Generated_code_dag::set_resource_tag(
    IValue_resource const *res,
    int                   tag)
{
    if (res->get_tag_value() != 0) {
        MDL_ASSERT(res->get_tag_value() == tag && "trying to overwrite a set tag value");
        return;
    }

    int old_tag = find_resource_tag(res);

    if (old_tag == 0) {
        add_resource_tag(res, tag);
    } else {
        MDL_ASSERT(old_tag == tag && "trying to overwrite a set tag value");
    }
}

// Get the number of resource map entries.
size_t Generated_code_dag::get_resource_tag_map_entries_count() const
{
    return m_resource_tag_map.size();
}

// Get the i'th resource tag tag map entry or NULL if the index is out of bounds;
Resource_tag_tuple const *Generated_code_dag::get_resource_tag_map_entry(size_t index) const
{
    if (index < m_resource_tag_map.size())
        return &m_resource_tag_map[index];
    return NULL;
}

// Get the resource tagger for this code DAG.
IResource_tagger *Generated_code_dag::get_resource_tagger() const
{
    return &m_resource_tagger;
}

// Find the tag for a given resource.
int Generated_code_dag::find_resource_tag(
    IValue_resource const *res) const
{
    return m_resource_tagger.get_resource_tag(res);
}

// Adds a tag, version pair for a given resource.
void Generated_code_dag::add_resource_tag(
    IValue_resource const *res,
    int                   tag)
{
    size_t l = m_resource_tag_map.size();
    m_resource_tag_map.resize(l + 1);

    ISymbol const *shared = m_sym_tab.get_shared_symbol(res->get_string_value());
    m_resource_tag_map[l] = Resource_tag_tuple(
        kind_from_value(res), shared->get_name(), tag);
}

// Serialize this code DAG.
void Generated_code_dag::serialize(
    ISerializer           *serializer,
    MDL_binary_serializer *bin_serializer) const
{
    DAG_serializer dag_serializer(get_allocator(), serializer, bin_serializer);

    // mark the start of the DAG
    dag_serializer.write_section_tag(Serializer::ST_DAG_START);
    DOUT(("Starting Serializing DAG\n")); INC_SCOPE();

    // write the module and file name
    dag_serializer.write_cstring(m_module_name.c_str());
    dag_serializer.write_cstring(m_module_file_name.c_str());

    // write the internal space
    dag_serializer.write_cstring(m_internal_space.c_str());

    // Serialize the Generated_code<> first
    // no need to serialize: m_printer
    m_sym_tab.serialize(dag_serializer);
    m_type_factory.serialize(dag_serializer);
    m_value_factory.serialize(dag_serializer);
    m_messages.serialize(dag_serializer);

    dag_serializer.serialize(m_module_imports);

    // m_module

    // m_module_stack
    // m_module_stack_tos

    // m_invisible_sym(m_sym_tab.create_user_type_symbol(""))

    // the compiler handle m_mdl will not be serialized

    // serialize the node factory m_node_factory by serializing all reachable DAGs
    serialize_dags(dag_serializer);

    dag_serializer.serialize(m_module_annotations);

    serialize_functions(dag_serializer);
    serialize_materials(dag_serializer);
    serialize_user_types(dag_serializer);
    serialize_constants(dag_serializer);

    // m_internal_space already written

    // These are only used during AST->DAG IR conversion:
    // m_tmp_value_map
    // m_current_material_index
    // m_accesible_parameters

    dag_serializer.write_unsigned(m_options);

    // serialize the resource table
    size_t n_entries = m_resource_tag_map.size();
    dag_serializer.write_unsigned(n_entries);

    for (size_t i = 0; i < n_entries; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        dag_serializer.write_byte(e.m_kind);
        dag_serializer.write_cstring(e.m_url);
        dag_serializer.write_db_tag(e.m_tag);
    }

    // mark the end of the DAG
    dag_serializer.write_section_tag(Serializer::ST_DAG_END);
    DEC_SCOPE(); DOUT(("DAG Serializing Finished\n\n"));
}

// Serialize all DAGs of this code DAG.
void Generated_code_dag::serialize_dags(DAG_serializer &dag_serializer) const
{
    // this is ugly, but the root set is scattered
    Dag_vector entity_roots(get_allocator());

    // functions
    for (size_t i = 0, n_functions = get_function_count(); i < n_functions; ++i) {
        Function_info const &func = m_functions[i];

        for (size_t j = 0, n_params = func.get_parameter_count(); j < n_params; ++j) {
            Parameter_info const &param = func.get_parameter(j);

            entity_roots.push_back(param.get_default());
            entity_roots.push_back(param.get_enable_if_condition());

            for (size_t k = 0, n_annos = param.get_annotation_count(); k < n_annos; ++k) {
                entity_roots.push_back(param.get_annotation(k));
            }
        }

        for (size_t j = 0, n_annos = func.get_annotation_count(); j < n_annos; ++j) {
            entity_roots.push_back(func.get_annotation(j));
        }
        for (size_t j = 0, n_annos = func.get_return_annotation_count(); j < n_annos; ++j) {
            entity_roots.push_back(func.get_return_annotation(j));
        }
        for (size_t j = 0, n_temps = func.get_temporary_count(); j < n_temps; ++j) {
            entity_roots.push_back(func.get_temporary(j));
        }
        if (DAG_node const *body = func.get_body()) {
            entity_roots.push_back(body);
        }
    }

    // materials
    for (size_t i = 0, n_functions = get_material_count(); i < n_functions; ++i) {
        Material_info const &mat = m_materials[i];

        entity_roots.push_back(mat.get_body());

        // not strictly necessary, temporaries are also reachable from the body
        for (size_t j = 0, n_tmps = mat.get_temporary_count(); j < n_tmps; ++j) {
            entity_roots.push_back(mat.get_temporary(j));
        }

        for (size_t j = 0, n_params = mat.get_parameter_count(); j < n_params; ++j) {
            Parameter_info const &param = mat.get_parameter(j);

            entity_roots.push_back(param.get_default());
            entity_roots.push_back(param.get_enable_if_condition());

            for (size_t k = 0, n_annos = param.get_annotation_count(); k < n_annos; ++k) {
                entity_roots.push_back(param.get_annotation(k));
            }
        }

        for (size_t j = 0, n_annos = mat.get_annotation_count(); j < n_annos; ++j) {
            entity_roots.push_back(mat.get_annotation(j));
        }
    }

    // types
    for (size_t i = 0, n_types = get_type_count(); i < n_types; ++i) {
        User_type_info const &type = m_user_types[i];

        for (size_t j = 0, n_annos = type.get_annotation_count(); j < n_annos; ++j) {
            entity_roots.push_back(type.get_annotation(j));
        }

        for (size_t j = 0, n_ents = type.get_entity_count(); j < n_ents; ++j) {
            User_type_info::Entity_info const &ent = type.get_entity(j);

            for (size_t k = 0, n_annos = ent.get_annotation_count(); k < n_annos; ++k) {
                entity_roots.push_back(ent.get_annotation(k));
            }
        }
    }

    // constants
    for (size_t i = 0, n_contants = get_constant_count(); i < n_contants; ++i) {
        Constant_info const &con = m_user_constants[i];

        entity_roots.push_back(con.get_value());

        for (size_t k = 0, n_annos = con.get_annotation_count(); k < n_annos; ++k) {
            entity_roots.push_back(con.get_annotation(k));
        }
    }

    Dag_vector const *roots[] = {
        &entity_roots,
        &m_module_annotations
    };

    dag_serializer.write_dags(roots, dimension_of(roots));
}

// Deserialize all DAGs of this code DAG.
void Generated_code_dag::deserialize_dags(DAG_deserializer &dag_deserializer)
{
    // disable CSE here: we do deserialization where the DAGs of different
    // materials are mixed. If CSE would not be disabled, the DAGs would mixes up.
    No_CSE_scope no_cse(m_node_factory);

    dag_deserializer.read_dags(m_node_factory);
}

// Serialize all Function_infos of this code DAG.
void Generated_code_dag::serialize_functions(DAG_serializer &dag_serializer) const
{
    size_t l = m_functions.size();

    dag_serializer.write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        Function_info const &func = m_functions[i];

        dag_serializer.write_encoded(func.m_semantics);
        dag_serializer.write_encoded(func.m_return_type);

        dag_serializer.write_encoded(func.m_name);
        dag_serializer.write_encoded(func.m_simple_name);
        dag_serializer.write_encoded(func.m_original_name);
        dag_serializer.write_encoded(func.m_cloned);

        if (func.m_has_hash) {
            dag_serializer.write_bool(true);

            for (size_t i = 0, n = func.m_hash.size(); i < n; ++i) {
                dag_serializer.write_byte(func.m_hash[i]);
            }
        } else {
            dag_serializer.write_bool(false);
        }

        serialize_parameters(func, dag_serializer);

        dag_serializer.serialize(func.m_annotations);
        dag_serializer.serialize(func.m_return_annos);
        dag_serializer.serialize(func.m_temporaries);
        dag_serializer.serialize(func.m_temporary_names);
        dag_serializer.serialize(func.m_refs);

        dag_serializer.write_unsigned(func.m_properties);

        if (func.m_body != NULL) {
            dag_serializer.write_bool(true);

            dag_serializer.write_encoded(func.m_body);
        } else {
            dag_serializer.write_bool(false);
        }
    }
}

// Deserialize all Material_infos of this code DAG.
void Generated_code_dag::deserialize_functions(DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    m_functions.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        Definition::Semantics sema        = dag_deserializer.read_encoded<Definition::Semantics>();
        IType const           *ret_type   = dag_deserializer.read_encoded<IType const *>();
        string                name        = dag_deserializer.read_encoded<string>();
        string                simple_name = dag_deserializer.read_encoded<string>();
        string                orig_name   = dag_deserializer.read_encoded<string>();
        string                cloned      = dag_deserializer.read_encoded<string>();
        bool                  has_hash    = dag_deserializer.read_bool();

        DAG_hash hash, *hp = NULL;

        if (has_hash) {
            for (size_t i = 0, n = hash.size(); i < n; ++i) {
                hash.data()[i] = dag_deserializer.read_byte();
            }
            hp = &hash;
        }

        Function_info func(
            get_allocator(),
            sema,
            ret_type,
            name.c_str(),
            simple_name.c_str(),
            orig_name.c_str(),
            cloned.c_str(),
            hp);

        deserialize_parameters(func, dag_deserializer);

        dag_deserializer.deserialize(func.m_annotations);
        dag_deserializer.deserialize(func.m_return_annos);
        dag_deserializer.deserialize(func.m_temporaries);
        dag_deserializer.deserialize(func.m_temporary_names);
        dag_deserializer.deserialize(func.m_refs);

        func.m_properties = dag_deserializer.read_unsigned();

        if (dag_deserializer.read_bool()) {
            func.m_body = dag_deserializer.read_encoded<DAG_node const *>();
        } else {
            func.m_body= NULL;
        }

        m_functions.push_back(func);
    }
}

// Serialize all Material_infos of this code DAG.
void Generated_code_dag::serialize_materials(DAG_serializer &dag_serializer) const
{
    size_t l = m_materials.size();

    dag_serializer.write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        Material_info const &mat = m_materials[i];

        dag_serializer.write_encoded(mat.m_name);
        dag_serializer.write_encoded(mat.m_simple_name);
        dag_serializer.write_encoded(mat.m_original_name);
        dag_serializer.write_encoded(mat.m_cloned);

        serialize_parameters(mat, dag_serializer);

        dag_serializer.serialize(mat.m_annotations);
        dag_serializer.serialize(mat.m_temporaries);
        dag_serializer.serialize(mat.m_temporary_names);

        dag_serializer.write_encoded(mat.m_body);
    }
}

// Deserialize all Material_infos of this code DAG.
void Generated_code_dag::deserialize_materials(DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    m_materials.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        string name        = dag_deserializer.read_encoded<string>();
        string simple_name = dag_deserializer.read_encoded<string>();
        string orig_name   = dag_deserializer.read_encoded<string>();
        Material_info mat(get_allocator(), name.c_str(), simple_name.c_str(), orig_name.c_str());

        mat.m_cloned = dag_deserializer.read_encoded<string>();

        deserialize_parameters(mat, dag_deserializer);

        dag_deserializer.deserialize(mat.m_annotations);
        dag_deserializer.deserialize(mat.m_temporaries);
        dag_deserializer.deserialize(mat.m_temporary_names);

        mat.m_body = dag_deserializer.read_encoded<DAG_node const *>();

        m_materials.push_back(mat);
    }
}

// Serialize all Annotation_infos of this code DAG.
void Generated_code_dag::serialize_annotations(DAG_serializer &dag_serializer) const
{
    size_t l = m_annotations.size();

    dag_serializer.write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        Annotation_info const &anno = m_annotations[i];

        dag_serializer.write_encoded(anno.m_semantics);
        dag_serializer.write_encoded(anno.m_name);
        dag_serializer.write_encoded(anno.m_simple_name);
        dag_serializer.write_encoded(anno.m_original_name);

        serialize_parameters(anno, dag_serializer);

        dag_serializer.serialize(anno.m_annotations);

        dag_serializer.write_unsigned(anno.m_properties);
    }
}

// Deserialize all Annotation_infos of this code DAG.
void Generated_code_dag::deserialize_annotations(DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    m_annotations.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        Definition::Semantics sema = dag_deserializer.read_encoded<Definition::Semantics>();
        string                name = dag_deserializer.read_encoded<string>();
        string                simple_name = dag_deserializer.read_encoded<string>();
        string                orig_name = dag_deserializer.read_encoded<string>();

        Annotation_info anno(
            get_allocator(), sema, name.c_str(), simple_name.c_str(), orig_name.c_str());

        deserialize_parameters(anno, dag_deserializer);

        dag_deserializer.deserialize(anno.m_annotations);

        anno.m_properties = dag_deserializer.read_unsigned();

        m_annotations.push_back(anno);
    }
}

// Serialize one parameter.
void Generated_code_dag::serialize_parameter(
    Parameter_info const &param,
    DAG_serializer       &dag_serializer) const
{

    dag_serializer.write_encoded(param.m_type);
    dag_serializer.write_encoded(param.m_name);
    dag_serializer.write_encoded(param.m_type_name);

    if (DAG_node const *def_arg = param.m_default) {
        dag_serializer.write_encoded(def_arg);
    } else {
        dag_serializer.write_encoded_tag(0);
    }

    dag_serializer.serialize(param.m_annotations);

    if (DAG_node const *cond = param.m_enable_if_cond) {
        dag_serializer.write_encoded(cond);
    } else {
        dag_serializer.write_encoded_tag(0);
    }

    dag_serializer.serialize(param.m_users);
}

// Deserialize one parameter.
Generated_code_dag::Parameter_info Generated_code_dag::deserialize_parameter(
    DAG_deserializer &dag_deserializer)
{
    IType const *type = dag_deserializer.read_encoded<IType const *>();

    string name      = dag_deserializer.read_encoded<string>();
    string type_name = dag_deserializer.read_encoded<string>();

    Parameter_info param(
        get_allocator(),
        type,
        name.c_str(),
        type_name.c_str());

    {
        Tag_t t = dag_deserializer.read_encoded_tag();
        if (t != Tag_t(0)) {
            param.m_default = dag_deserializer.get_ir_node(t);
        } else {
            param.m_default = NULL;
        }
    }

    dag_deserializer.deserialize(param.m_annotations);

    {
        Tag_t t = dag_deserializer.read_encoded_tag();
        if (t != Tag_t(0)) {
            param.m_enable_if_cond = dag_deserializer.get_ir_node(t);
        } else {
            param.m_enable_if_cond = NULL;
        }
    }

    dag_deserializer.deserialize(param.m_users);

    return param;
}

// Serialize all parameters of a function.
void Generated_code_dag::serialize_parameters(
    Function_info const &func,
    DAG_serializer      &dag_serializer) const
{
    size_t l = func.get_parameter_count();
    dag_serializer.write_encoded_tag(l);

    for (size_t i = 0; i < l; ++i) {
        serialize_parameter(func.get_parameter(i), dag_serializer);
    }
}

// Deserialize all parameters of a function.
void Generated_code_dag::deserialize_parameters(
    Function_info    &func,
    DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    func.m_parameters.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        func.add_parameter(deserialize_parameter(dag_deserializer));
    }
}

// Serialize all parameters of a material.
void Generated_code_dag::serialize_parameters(
    Material_info const &mat,
    DAG_serializer      &dag_serializer) const
{
    size_t l = mat.get_parameter_count();

    dag_serializer.write_encoded_tag(l);

    for (size_t i = 0; i < l; ++i) {
        serialize_parameter(mat.get_parameter(i), dag_serializer);
    }
}

// Deserialize all parameters of a material.
void Generated_code_dag::deserialize_parameters(
    Material_info    &mat,
    DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    mat.m_parameters.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        mat.add_parameter(deserialize_parameter(dag_deserializer));
    }
}

/// Serialize all parameters of an annotation.
void Generated_code_dag::serialize_parameters(
    Annotation_info const &anno,
    DAG_serializer        &dag_serializer) const
{
    size_t l = anno.get_parameter_count();

    dag_serializer.write_encoded_tag(l);

    for (size_t i = 0; i < l; ++i) {
        serialize_parameter(anno.get_parameter(i), dag_serializer);
    }
}

// Deserialize all parameters of a annotation.
void Generated_code_dag::deserialize_parameters(
    Annotation_info  &anno,
    DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    anno.m_parameters.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        anno.add_parameter(deserialize_parameter(dag_deserializer));
    }
}

// Serialize all User_type_infos of this code DAG.
void Generated_code_dag::serialize_user_types(DAG_serializer &dag_serializer) const
{
    size_t l = m_user_types.size();

    dag_serializer.write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        User_type_info const &type = m_user_types[i];

        dag_serializer.write_encoded(type.m_type);

        dag_serializer.write_encoded(type.m_name);
        dag_serializer.write_encoded(type.m_original_name);
        dag_serializer.write_bool(type.m_is_exported);

        dag_serializer.serialize(type.m_annotations);

        serialize_entities(type, dag_serializer);

    }
}

// Deserialize all User_type_infos of this code DAG.
void Generated_code_dag::deserialize_user_types(DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    m_user_types.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        IType const *type       = dag_deserializer.read_encoded<IType const *>();
        string      name        = dag_deserializer.read_encoded<string>();
        string      orig_name   = dag_deserializer.read_encoded<string>();
        bool        is_exported = dag_deserializer.read_bool();

        User_type_info func(
            get_allocator(), is_exported, type, name.c_str(), orig_name.c_str());

        dag_deserializer.deserialize(func.m_annotations);

        deserialize_entities(func, dag_deserializer);

        m_user_types.push_back(func);
    }
}

// Deserialize all entities of a user type.
void Generated_code_dag::deserialize_entities(
    User_type_info   &type,
    DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    type.m_entities.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        User_type_info::Entity_info ent(get_allocator());

        dag_deserializer.deserialize(ent.m_annotations);

        type.add_entity(ent);
    }
}

// Serialize all entities of a user type.
void Generated_code_dag::serialize_entities(
    User_type_info const &type,
    DAG_serializer       &dag_serializer) const
{
    size_t l = type.get_entity_count();

    dag_serializer.write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        User_type_info::Entity_info const &ent = type.get_entity(i);

        dag_serializer.serialize(ent.m_annotations);
    }
}

// Serialize all Constant_infos of this code DAG.
void Generated_code_dag::serialize_constants(DAG_serializer &dag_serializer) const
{
    size_t l = m_user_constants.size();

    dag_serializer.write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        Constant_info const &type = m_user_constants[i];

        dag_serializer.write_encoded(type.m_const);
        dag_serializer.write_encoded(type.m_name);
        dag_serializer.serialize(type.m_annotations);
    }
}

// Deserialize all Constant_infos of this code DAG.
void Generated_code_dag::deserialize_constants(DAG_deserializer &dag_deserializer)
{
    size_t l = dag_deserializer.read_encoded_tag();
    m_user_constants.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        DAG_node const *c   = dag_deserializer.read_encoded<DAG_node const *>();
        string         name = dag_deserializer.read_encoded<string>();

        Constant_info con(get_allocator(), cast<DAG_constant>(c), name.c_str());

        dag_deserializer.deserialize(con.m_annotations);

        m_user_constants.push_back(con);
    }
}

// Deserialize a code DAG.
Generated_code_dag const *Generated_code_dag::deserialize(
    IDeserializer           *deserializer,
    MDL_binary_deserializer *bin_deserializer,
    MDL                     *compiler)
{
    DAG_deserializer dag_deserializer(deserializer, bin_deserializer);

    // start tag is expected to be already read
    DOUT(("DAG START\n"));
    INC_SCOPE();

    // read the module name
    string module_name(dag_deserializer.read_cstring(), dag_deserializer.get_allocator());

    // read the module file name
    string module_file_name(dag_deserializer.read_cstring(), dag_deserializer.get_allocator());

    // read the internal space
    string internal_space(dag_deserializer.read_cstring(), dag_deserializer.get_allocator());

    mi::base::Handle<Generated_code_dag> code(
        dag_deserializer.create_code_dag(
            dag_deserializer.get_allocator(),
            compiler,
            (IModule const *)0,
            internal_space.c_str(),
            // FIXME: we do not serialize the context name here because we do not want to break
            // compatibility with the beta release.
            // However, this IS safe, because only compiled entities are serialized/deserialized
            // which are error free, so the context name is not needed.
            "renderer"));

    code->m_module_name      = module_name;
    code->m_module_file_name = module_file_name;

    // Deserialize the Generated_code<> first
    // no need to deserialize: m_printer
    code->m_sym_tab.deserialize(dag_deserializer);
    code->m_type_factory.deserialize(dag_deserializer);
    code->m_value_factory.deserialize(dag_deserializer);
    code->m_messages.deserialize(dag_deserializer);

    dag_deserializer.deserialize(code->m_module_imports);

    // m_module

    // m_module_stack
    // m_module_stack_tos

    // m_invisible_sym(m_sym_tab.create_user_type_symbol(""))

    // the compiler handle m_mdl will not be deserialized

    // deserialize the node factory m_node_factory by deserializing all reachable DAGs
    code->deserialize_dags(dag_deserializer);

    dag_deserializer.deserialize(code->m_module_annotations);

    code->deserialize_functions(dag_deserializer);
    code->deserialize_materials(dag_deserializer);
    code->deserialize_user_types(dag_deserializer);
    code->deserialize_constants(dag_deserializer);

    // m_internal_space already read

    // These are only used during AST->DAG IR conversion:
    // m_tmp_value_map
    // m_current_material_index
    // m_accesible_parameters

    code->m_options = dag_deserializer.read_unsigned();

    // serialize the resource table
    size_t n_entries = dag_deserializer.read_unsigned();

    code->m_resource_tag_map.clear();
    for (size_t i = 0; i < n_entries; ++i) {
        Resource_tag_tuple::Kind kind = Resource_tag_tuple::Kind(dag_deserializer.read_byte());
        string url(dag_deserializer.read_cstring(), dag_deserializer.get_allocator());
        unsigned tag     = dag_deserializer.read_db_tag();

        ISymbol const *shared = code->m_sym_tab.get_shared_symbol(url.c_str());
        code->m_resource_tag_map.push_back(Resource_tag_tuple(kind, shared->get_name(), tag));
    }

    DEC_SCOPE(); DOUT(("DAG END\n\n"));

    code->retain();
    return code.get();
}

// Add a material temporary.
int Generated_code_dag::add_material_temporary(
    int            mat_index,
    DAG_node const *node,
    char const     *name)
{
    Material_info &mat = m_materials[mat_index];
    size_t idx = mat.add_temporary(node, name);
    return int(idx);
}

// Add a function temporary.
int Generated_code_dag::add_function_temporary(
    int            func_index,
    DAG_node const *node,
    char const     *name)
{
    Function_info &func = m_functions[func_index];
    size_t idx = func.add_temporary(node, name);
    return int(idx);
}

// Dump the material expression DAG.
void Generated_code_dag::dump_material_dag(
    size_t         index,
    char const     *suffix,
    size_t         argc,
    DAG_node const *argv[]) const
{
    // dump the dependency graph
    string fname(get_allocator());
    fname += get_material_name(index);
    if (suffix)
        fname += suffix;
    fname += "_DAG.gv";

    for (size_t i = 0, n = fname.size(); i < n; ++i) {
        char c = fname[i];
        if (c == ':' || c == '/' || c == '\\')
            fname[i] = '_';
    }

    if (FILE *f = fopen(fname.c_str(), "w")) {
        Allocator_builder builder(get_allocator());

        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(get_allocator(), f, /*close_at_destroy=*/true));

        Material_dumper dumper(get_allocator(), *this, index, out.get());

        dumper.dump(argc, argv);
    }
}

// Acquires a const interface.
mi::base::IInterface const *Generated_code_dag::get_interface(
    mi::base::Uuid const &interface_id) const
{
    if (interface_id == IPrinter_interface::IID()) {
        return m_builder.create<DAG_code_printer>(m_builder.get_allocator());
    }
    return Base::get_interface(interface_id);
}

// Create a default enum.
IValue_enum const *Generated_code_dag::create_default_enum(
    IValue_factory   &value_factory,
    IType_enum const *type)
{
    ISymbol const *symbol;
    int code;

    // retrieve the first enum value
    type->get_value(0, symbol, code);
    return value_factory.create_enum(type, code);
}

// Create a default bsdf.
IValue_invalid_ref const *Generated_code_dag::create_default_bsdf(
    IValue_factory &value_factory)
{
    IType_bsdf const *type_bsdf = value_factory.get_type_factory()->create_bsdf();
    return value_factory.create_invalid_ref(type_bsdf);
}

// Create a default hair_bsdf.
IValue_invalid_ref const *Generated_code_dag::create_default_hair_bsdf(
    IValue_factory &value_factory)
{
    IType_hair_bsdf const *type_hair_bsdf = value_factory.get_type_factory()->create_hair_bsdf();
    return value_factory.create_invalid_ref(type_hair_bsdf);
}

// Create a default edf.
IValue_invalid_ref const *Generated_code_dag::create_default_edf(
    IValue_factory &value_factory)
{
    IType_edf const *type_edf = value_factory.get_type_factory()->create_edf();
    return value_factory.create_invalid_ref(type_edf);
}

// Create a default vdf.
IValue_invalid_ref const *Generated_code_dag::create_default_vdf(
    IValue_factory &value_factory)
{
    IType_vdf const *type_vdf = value_factory.get_type_factory()->create_vdf();
    return value_factory.create_invalid_ref(type_vdf);
}

/// Create a default vector.
IValue_vector const *Generated_code_dag::create_default_vector(
    IValue_factory     &value_factory,
    IType_vector const *type) const
{
    IType const *et = type->get_element_type();
    int count = type->get_size();
    Small_VLA<IValue const *, 4> values(get_allocator(), count);

    switch (et->get_kind()) {
    case IType::TK_BOOL:
        {
            IValue_bool const *false_value = value_factory.create_bool(false);
            for (int i = 0; i < count; ++i)
                values[i] = false_value;
            return value_factory.create_vector(type, values.data(), values.size());
        }
    case IType::TK_INT:
        {
            IValue_int const *zero_value = value_factory.create_int(0);
            for (int i = 0; i < count; ++i)
                values[i] = zero_value;
            return value_factory.create_vector(type, values.data(), values.size());
        }
    case IType::TK_FLOAT:
        {
            IValue_float const *zero_value = value_factory.create_float(0.0f);
            for (int i = 0; i < count; ++i)
                values[i] = zero_value;
            return value_factory.create_vector(type, values.data(), values.size());
        }
    case IType::TK_DOUBLE:
        {
            IValue const *zero_value = value_factory.create_double(0.0);
            for (int i = 0; i < count; ++i)
                values[i] = zero_value;
            return value_factory.create_vector(type, values.data(), values.size());
        }
    default:
        break;
    }
    return NULL;
}

// Create a default matrix.
IValue_matrix const *Generated_code_dag::create_default_matrix(
    IValue_factory     &value_factory,
    IType_matrix const *type) const
{
    IType_vector const *rt = type->get_element_type();
    int count = type->get_columns();
    Small_VLA<IValue const *, 4> values(get_allocator(), count);
    for (int i = 0; i < count; ++i)
        values[i] = create_default_vector(value_factory, rt);
    return value_factory.create_matrix(type, values.data(), values.size());
}

// Create a default color.
IValue_rgb_color const *Generated_code_dag::create_default_color(
    IValue_factory &value_factory)
{
    IValue_float const *zero_value = value_factory.create_float(0.0f);
    return value_factory.create_rgb_color(zero_value, zero_value, zero_value);
}

// Create a default texture.
IValue_invalid_ref const *Generated_code_dag::create_default_texture(
    IValue_factory      &value_factory,
    IType_texture const *type)
{
    return value_factory.create_invalid_ref(type);
}

} // mdl
} // mi

