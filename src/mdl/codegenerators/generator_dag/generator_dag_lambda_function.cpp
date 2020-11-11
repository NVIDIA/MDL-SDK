/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <base/system/stlext/i_stlext_no_unused_variable_warning.h>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_visitor.h"
#include "mdl/compiler/compilercore/compilercore_streams.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"
#include "mdl/compiler/compilercore/compilercore_hash.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "mdl/compiler/compilercore/compilercore_visitor.h"

#include "mdl/codegenerators/generator_code/generator_code.h"

#include "generator_dag_tools.h"
#include "generator_dag_serializer.h"
#include "generator_dag_walker.h"
#include "generator_dag_dumper.h"
#include "generator_dag_lambda_function.h"
#include "generator_dag_builder.h"

namespace mi {
namespace mdl {

namespace {

typedef Store<mi::base::Handle<IModule const> >  IModule_scope;


///
/// Helper class to dump an material expression DAG as a dot file.
///
class Lambda_dumper : public DAG_dumper {
    typedef DAG_dumper Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param out    an output stream, the dot file is written to
    Lambda_dumper(
        IAllocator     *alloc,
        IOutput_stream *out);

#if 0
    /// Dump the lambda expression DAG to the output stream.
    ///
    /// \param lambda  the lambda function
    void dump(Lambda_function const &lambda);
#endif

    /// Dump the lambda expression DAG to the output stream.
    ///
    /// \param lambda  the lambda function root expression
    void dump(DAG_node const *root);

    /// Get the parameter name for the given index if any.
    ///
    /// \param index  the index of the parameter
    char const *get_parameter_name(int index) MDL_FINAL;

private:
    /// Currently processed lambda function.
    Lambda_function const *m_lambda;
    char m_name_buf[40];
};

// Constructor.
Lambda_dumper::Lambda_dumper(
    IAllocator           *alloc,
    IOutput_stream       *out)
: Base(alloc, out)
, m_lambda(NULL)
{
}

#if 0
// Dump the lambda expression DAG to the output stream.
void Lambda_dumper::dump(Lambda_function const &lambda)
{
    m_lambda = &lambda;

    m_printer->print("digraph \"");
    char const *name = lambda.get_name();
    if (name == NULL || name[0] == '\0')
        name = "lambda";
    m_printer->print(name);
    m_printer->print("\" {\n");

    size_t lambda_id = get_unique_id();

    m_printer->print("  ");
    char lambda_root_name[32];
    snprintf(lambda_root_name, sizeof(lambda_root_name), "n%ld", (long)lambda_id);
    lambda_root_name[sizeof(lambda_root_name) - 1] = '\0';
    m_printer->print(lambda_root_name);
    m_printer->print(" [label=\"Lambda\"]");

    for (int i = 0, n = lambda.get_root_expr_count(); i < n; ++i) {
        DAG_node *root = const_cast<DAG_node *>(lambda.get_root_expr(i));

        if (root != NULL) {
            m_walker.walk_node(root, this);

            // add root edge
            m_printer->print("  ");
            m_printer->print(lambda_root_name);
            m_printer->print(" -> ");
            node_name(root);

            m_printer->print(" [label=\"");
            char label[32];
            snprintf(label, sizeof(label), "n%d", i);
            label[sizeof(label) - 1] = '\0';
            m_printer->print(label);
            m_printer->print("\"]");
        }
    }
    m_printer->print("}\n");
}
#endif

// Dump the lambda expression DAG to the output stream.
void Lambda_dumper::dump(DAG_node const *root)
{
    m_printer->print("digraph \"lambda\" {\n");

    m_walker.walk_node(const_cast<DAG_node *>(root), this);
    m_printer->print("}\n");
}

// Get the parameter name for the given index if any.
const char *Lambda_dumper::get_parameter_name(int index)
{
    if (m_lambda == NULL) {
        snprintf(m_name_buf, sizeof(m_name_buf), "param %d", index);
        return m_name_buf;
    }
    return m_lambda->get_parameter_name(index);
}

}  // anonymous

mi::base::Atom32 Lambda_function::g_next_serial;

// Constructor.
Lambda_function::Lambda_function(
    IAllocator               *alloc,
    MDL                      *compiler,
    Lambda_execution_context context)
: Base(alloc)
, m_mdl(mi::base::make_handle_dup(compiler))
, m_arena(alloc)
, m_sym_tab(m_arena)
, m_type_factory(m_arena, compiler, &m_sym_tab)
, m_value_factory(m_arena, m_type_factory)
, m_node_factory(compiler, m_arena, m_value_factory, internal_space(context))
, m_name(alloc)
, m_root_map(0, Root_map::hasher(), Root_map::key_equal(), alloc)
, m_roots(alloc)
, m_resource_attr_map(0, Resource_attr_map::hasher(), Resource_attr_map::key_equal(), alloc)
, m_has_resource_attributes(true)
, m_context(context)
, m_hash()
, m_body_expr(NULL)
, m_params(alloc)
, m_index_map(Index_map::key_compare(), alloc)
, m_serial_number(0u)
, m_uses_render_state(false)
, m_has_dead_code(false)
, m_is_modified(false)
, m_uses_lambda_results(false)
, m_serial_is_valid(false)
, m_hash_is_valid(false)
, m_deriv_infos_calculated(false)
, m_deriv_infos(alloc)
, m_resource_tag_map(alloc)
{
    // CSE is always enabled when creating a lambda function
    m_node_factory.enable_cse(true);

    // FIXME: should we allow unsafe math here?
    m_node_factory.enable_unsafe_math_opt(false);
}

// Get the internal space from the execution context.
char const *Lambda_function::internal_space(Lambda_execution_context context)
{
    char const *internal_space = "*";

    switch (context) {
    case LEC_ENVIRONMENT:
        // all spaces are equal inside MDL environment functions
        internal_space = "*";
        break;
    case LEC_CORE:
        // internal space is equal to world space inside the iray core
        internal_space = "coordinate_world";
        break;
    case LEC_DISPLACEMENT:
        // internal space is equal to object space inside displacement, but we do not support
        // displacement in world space yet, so map all
        internal_space = "*";
        break;
    }
    return internal_space;
}

// Create an empty lambda function with the same option as a give other.
Lambda_function *Lambda_function::clone_empty(Lambda_function const &other)
{
    IAllocator *alloc = other.get_allocator();

    Allocator_builder builder(alloc);

    return builder.create<Lambda_function>(
        alloc,
        other.m_mdl.get(),
        other.m_context);
}

// Get the type factory of this builder.
Type_factory *Lambda_function::get_type_factory()
{
    return &m_type_factory;
}

// Get the value factory of this builder.
Value_factory *Lambda_function::get_value_factory()
{
    return &m_value_factory;
}

// Create a constant.
DAG_constant const *Lambda_function::create_constant(
    IValue const *value)
{
    return m_node_factory.create_constant(value);
}

// Create a call.
DAG_node const *Lambda_function::create_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    int                           num_call_args,
    IType const                   *ret_type)
{
    if (is_varying_state_semantic(sema)) {
        // varying state functions require a render state
        m_uses_render_state = true;
    } else if (sema == IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA) {
        // expression lambda calls require an available lambda_results parameter
        m_uses_lambda_results = true;
    }
    return m_node_factory.create_call(name, sema, call_args, num_call_args, ret_type);
}

// Create a parameter reference.
DAG_parameter const *Lambda_function::create_parameter(
    IType const *type,
    int         index)
{
    Index_map::const_iterator it = m_index_map.find(index);
    if (it != m_index_map.end()) {
        // we have a remap
        index = it->second;
    }

    if (index >= m_params.size()) {
        MDL_ASSERT(!"parameter index out of range when constructing a lambda function");
        return NULL;
    }
    MDL_ASSERT(type->skip_type_alias() == get_parameter_type(index)->skip_type_alias());

    return m_node_factory.create_parameter(type, index);
}

// Get the body of this function.
DAG_node const *Lambda_function::get_body() const
{
    return m_body_expr;
}

// Set the body of this function.
void Lambda_function::set_body(DAG_node const *expr)
{
    m_body_expr = expr;
}

// Import (i.e. deep-copy) a DAG expression into this lambda function.
DAG_node const *Lambda_function::import_expr(DAG_node const *expr)
{
    for (;;) {
        switch (expr->get_kind()) {
        case DAG_node::EK_CONSTANT:
            {
                DAG_constant const *c = cast<DAG_constant>(expr);
                mi::mdl::IValue const *v = c->get_value();
                v = m_value_factory.import(v);
                return create_constant(v);
            }
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                continue;
            }
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                int n_args = call->get_argument_count();
                VLA<DAG_call::Call_argument> args(get_allocator(), n_args);

                for (int i = 0; i < n_args; ++i) {
                    DAG_call::Call_argument &arg = args[i];
                    arg.arg        = import_expr(call->get_argument(i));
                    arg.param_name = call->get_parameter_name(i);
                }

                IType const *ret_type = call->get_type();
                ret_type = m_type_factory.import(ret_type);

                return create_call(
                    call->get_name(), call->get_semantic(), args.data(), n_args, ret_type);
            }
        case DAG_node::EK_PARAMETER:
            {
                DAG_parameter const *p = cast<DAG_parameter>(expr);
                int index = p->get_index();
                IType const *type = p->get_type();
                type = m_type_factory.import(type);

                return create_parameter(type, index);
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
    }
}

// Return a free root index.
size_t Lambda_function::find_free_root_index()
{
    size_t n = m_roots.size();

    // search first free
    for (size_t idx = 0; idx < n; ++idx)
        if (m_roots[idx] == NULL)
            return idx;
    return n;
}

// Store a DAG (root) expression and returns an index for it.
size_t Lambda_function::store_root_expr(DAG_node const *expr)
{
    Root_map::const_iterator it(m_root_map.find(expr));
    if (it != m_root_map.end())
        return it->second;

    size_t idx = find_free_root_index();
    if (idx >= m_roots.size())
        m_roots.resize(idx + 1);

    MDL_ASSERT(m_roots[idx] == NULL);
    m_roots[idx] = expr;

    m_root_map[expr] = idx;
    m_is_modified    = true;

    // the serial number and the hash must be updated on next read because this lambda was modified
    m_serial_is_valid = false;
    m_hash_is_valid   = false;

    return idx;
}

// Remove a root expression.
bool Lambda_function::remove_root_expr(size_t idx)
{
    if (idx >= m_roots.size())
        return false;
    DAG_node const *root = m_roots[idx];
    if (root == NULL)
        return false;

    m_root_map.erase(root);
    m_roots[idx] = NULL;

    m_has_dead_code = true;
    return true;
}

// Run garbage collection AFTER a root expression was removed.
Lambda_function *Lambda_function::garbage_collection()
{
    if (!m_has_dead_code) {
        // expects a "new" one, so the reference must be increased
        retain();
        return this;
    }

    bool non_empty = false;
    size_t n = get_root_expr_count();

    for (size_t idx = 0; idx < n; ++idx) {
        DAG_node const *expr = get_root_expr(idx);

        if (expr != NULL) {
            non_empty = true;
            break;
        }
    }

    if (!non_empty) {
        return NULL;
    }

    Lambda_function *n_func = clone_empty(*this);

    n_func->m_roots.resize(n);

    for (size_t idx = 0; idx < n; ++idx) {
        DAG_node const *expr = get_root_expr(idx);

        if (expr != NULL)
            expr = n_func->import_expr(expr);

        MDL_ASSERT(n_func->m_roots[idx] == NULL);
        n_func->m_roots[idx] = expr;
        n_func->m_root_map[expr] = idx;
    }

    // copy the resource map, otherwise our new lambda function might have different
    // mapping from IValue_resources to resource indexes
    for (Resource_attr_map::const_iterator
         it(m_resource_attr_map.begin()), end(m_resource_attr_map.end());
         it != end;
         ++it)
    {
        n_func->m_resource_attr_map[it->first] = it->second;
    }

    return n_func;
}

// Get the remembered expression for a given index.
DAG_node const *Lambda_function::get_root_expr(size_t idx) const
{
    if (idx < m_roots.size())
        return m_roots[idx];
    return NULL;
}

// Get the number of root expressions.
size_t Lambda_function::get_root_expr_count() const
{
    return m_roots.size();
}

namespace {

typedef ptr_hash_set<IValue const>::Type Resource_set;
typedef vector<IValue const *>::Type Resource_list;
typedef ptr_hash_map<IValue const, ILambda_resource_enumerator::Texture_usage>::Type Texture_usage_map;
typedef ptr_hash_map<IDefinition const, ILambda_resource_enumerator::Texture_usage *>::Type
    Arg_usages_by_def;
typedef map<IDefinition::Semantics, ILambda_resource_enumerator::Texture_usage *>::Type
    Arg_usages_by_semantics;

class Resource_collector;

/// Helper class to collect all resources from a AST walk.
class Resource_AST_collector : private Module_visitor {
public:
    /// Constructor.
    Resource_AST_collector(
        IAllocator              *alloc,
        Memory_arena            &arena,
        Resource_collector      &res_collector,
        Resource_list           &textures,
        Resource_list           &light_profiles,
        Resource_list           &bsdf_measurements,
        Resource_set            &found_resources,
        Texture_usage_map       &tex_usage_map,
        Arg_usages_by_def       &arg_usages_by_def,
        Arg_usages_by_semantics &arg_usages_by_semantics)
    : m_arena(arena)
    , m_res_collector(res_collector)
    , m_textures(textures)
    , m_light_profiles(light_profiles)
    , m_bsdf_measurements(bsdf_measurements)
    , m_found_resources(found_resources)
    , m_tex_usage_map(tex_usage_map)
    , m_arg_usages_by_def(arg_usages_by_def)
    , m_arg_usages_by_semantics(arg_usages_by_semantics)
    , m_mod()
    , m_tex_usage(0)
    , m_param_usage_map(alloc)
    {
    }

    /// Calculate the argument usage of a function and collect used resources.
    ILambda_resource_enumerator::Texture_usage *process_function(
        mi::base::Handle<IModule const> owner,
        IDefinition const *def)
    {
        IDeclaration_function const *f = as<IDeclaration_function>(def->get_declaration());
        if (f == NULL)
            return NULL;

        // Allocate argument usage array
        size_t num_params = f->get_parameter_count();
        size_t alloc_size = std::max(size_t(1), num_params) *
            sizeof(ILambda_resource_enumerator::Texture_usage);
        ILambda_resource_enumerator::Texture_usage *arg_usages =
            reinterpret_cast<ILambda_resource_enumerator::Texture_usage *>(
                m_arena.allocate(alloc_size));
        memset(arg_usages, 0, alloc_size);

        // Map function parameters to argument usage slots
        for (size_t i = 0; i < num_params; ++i) {
            m_param_usage_map[f->get_parameter(i)->get_name()->get_definition()] = arg_usages + i;
        }

        IModule_scope scope(m_mod, owner);
        visit(f);

        return arg_usages;
    }

private:
    /// Post-visit a literal expression.
    ///
    /// \param cnst  the constant that is visited
    IExpression *post_visit(IExpression_literal *expr) MDL_FINAL
    {
        IValue const *v = expr->get_value();
        IType const  *t = v->get_type();

        // note: this also collects invalid references ...
        switch (t->get_kind()) {
        case IType::TK_TEXTURE:
            if (m_tex_usage)
                m_tex_usage_map[v] |= m_tex_usage;

            if (m_found_resources.insert(v).second) { // inserted for first time?
                m_textures.push_back(v);
            }
            break;
        case IType::TK_LIGHT_PROFILE:
            if (m_found_resources.insert(v).second) { // inserted for first time?
                m_light_profiles.push_back(v);
            }
            break;
        case IType::TK_BSDF_MEASUREMENT:
            if (m_found_resources.insert(v).second) { // inserted for first time?
                m_bsdf_measurements.push_back(v);
            }
            break;
        default:
            break;
        }
        return expr;
    }

    /// Pre-visit a call expression.
    ///
    /// \param call  the call that is visited
    ///
    /// \return true, if the children should be visited
    bool pre_visit(IExpression_call *call) MDL_FINAL;

    /// Post-visit a reference expression.
    IExpression *post_visit(IExpression_reference *ref) MDL_FINAL
    {
        IDefinition const *def = ref->get_definition();

        // Ignore references without definition (array constructor, for example string[] in anno)
        if (def == NULL)
            return ref;

        if (def->get_kind() == IDefinition::DK_PARAMETER) {
            if (is<IType_texture>(def->get_type()->skip_type_alias())) {
                ILambda_resource_enumerator::Texture_usage *usage = m_param_usage_map[def];
                MDL_ASSERT(usage);
                if (usage)
                    *usage |= m_tex_usage;
            }
        }
        return ref;
    }


private:
    Memory_arena            &m_arena;
    Resource_collector      &m_res_collector;
    Resource_list           &m_textures;
    Resource_list           &m_light_profiles;
    Resource_list           &m_bsdf_measurements;
    Resource_set            &m_found_resources;
    Texture_usage_map       &m_tex_usage_map;
    Arg_usages_by_def       &m_arg_usages_by_def;
    Arg_usages_by_semantics &m_arg_usages_by_semantics;

    mi::base::Handle<IModule const> m_mod;

    ILambda_resource_enumerator::Texture_usage m_tex_usage;

    typedef ptr_hash_map<IDefinition const, ILambda_resource_enumerator::Texture_usage *>::Type
        Parameter_usage_map;
    Parameter_usage_map m_param_usage_map;
};


struct Argument_usage_init_table
{
    IDefinition::Semantics sema;
    ILambda_resource_enumerator::Texture_usage arg_usage[8];  // for intrinsics support up to 8 args
};

static Argument_usage_init_table arg_usage_init_table[] = {
    #define USAGE_ON_FIRST_ARG(sema, usage) \
        { \
            IDefinition::sema, \
            { ILambda_resource_enumerator::usage, 0, 0, 0, 0, 0, 0, 0 } \
        }

    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_HEIGHT,           TU_HEIGHT),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_WIDTH,            TU_WIDTH),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_DEPTH,            TU_DEPTH),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_LOOKUP_FLOAT,     TU_LOOKUP_FLOAT),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_LOOKUP_FLOAT2,    TU_LOOKUP_FLOAT2),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_LOOKUP_FLOAT3,    TU_LOOKUP_FLOAT3),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_LOOKUP_FLOAT4,    TU_LOOKUP_FLOAT4),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_LOOKUP_COLOR,     TU_LOOKUP_COLOR),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_TEXEL_FLOAT,      TU_TEXEL_FLOAT),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_TEXEL_FLOAT2,     TU_TEXEL_FLOAT2),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_TEXEL_FLOAT3,     TU_TEXEL_FLOAT3),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_TEXEL_FLOAT4,     TU_TEXEL_FLOAT4),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_TEXEL_COLOR,      TU_TEXEL_COLOR),
    USAGE_ON_FIRST_ARG(DS_INTRINSIC_TEX_TEXTURE_ISVALID,  TU_TEXTURE_ISVALID),

    #undef USAGE_ON_FIRST_ARG
};


/// Helper class to collect all resources.
class Resource_collector {
public:
    /// Constructor.
    ///
    /// \param alloc              the allocator
    /// \param name_resolver      a call name resolver
    /// \param lambda_func        the lambda function
    /// \param textures           a list that will be filled with unique found textures
    /// \param light_profiles     a list that will be filled with unique found light profiles
    /// \param bsdf_measurements  a list that will be filled with unique found bsdf measurements
    Resource_collector(
        IAllocator                *alloc,
        ICall_name_resolver const &name_resolver,
        Lambda_function const     &lambda_func,
        Resource_list             &textures,
        Resource_list             &light_profiles,
        Resource_list             &bsdf_measurements);

    /// Collect resources and texture usage information for the given DAG node.
    void collect(DAG_node const *node);

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit_constant(DAG_constant const *cnst)
    {
        IValue const *v = cnst->get_value();
        IType const  *t = v->get_type();

        // note: this also collects invalid references, because we check for the type,
        // not for the value kind ...
        switch (t->get_kind()) {
        case IType::TK_TEXTURE:
            if (m_tex_usage)
                m_tex_usage_map[v] |= m_tex_usage;

            if (m_found_resources.insert(v).second)  // inserted for first time?
                m_textures.push_back(v);
            break;
        case IType::TK_LIGHT_PROFILE:
            if (m_found_resources.insert(v).second)  // inserted for first time?
                m_light_profiles.push_back(v);
            break;
        case IType::TK_BSDF_MEASUREMENT:
            if (m_found_resources.insert(v).second)  // inserted for first time?
                m_bsdf_measurements.push_back(v);
            break;
        default:
            break;
        }
    }

    /// Post-visit a Call.
    ///
    /// \param call  the call that is visited
    void visit_call(DAG_call const *call);

    /// Returns the potential texture usage of a value.
    ILambda_resource_enumerator::Texture_usage get_texture_usage(IValue const *value) const
    {
        Texture_usage_map::const_iterator it = m_tex_usage_map.find(value);
        if (it == m_tex_usage_map.end())
            return 0;
        return it->second;
    }

    /// Returns the argument usage for the given definition and calculates it, if necessary.
    ILambda_resource_enumerator::Texture_usage *get_or_calc_arg_usage(
        mi::base::Handle<IModule const> owner,
        IDefinition const *def);

    /// Ensure, we keep a reference to the module, to avoid IValues from disappearing.
    void keep_module_reference(mi::base::Handle<IModule const> module)
    {
        m_module_set.insert(module);
    }

private:
    /// The memory arena used to allocate usage information.
    Memory_arena               m_arena;

    ICall_name_resolver const &m_resolver;

    Lambda_function const     &m_lambda_func;

    Resource_list             &m_textures;
    Resource_list             &m_light_profiles;
    Resource_list             &m_bsdf_measurements;
    Resource_set              m_found_resources;

    typedef ptr_hash_map<IValue const, ILambda_resource_enumerator::Texture_usage>::Type
        Texture_usage_map;
    Texture_usage_map         m_tex_usage_map;

    Arg_usages_by_def         m_arg_usages_by_def;
    Arg_usages_by_semantics   m_arg_usages_by_sema;

    Resource_AST_collector    m_ast_collector;

    typedef ptr_hash_set<DAG_node const>::Type Visited_set;
    Visited_set               m_visited;

    ILambda_resource_enumerator::Texture_usage  m_tex_usage;

    typedef handle_hash_set<IModule const>::Type Module_set;
    Module_set                m_module_set;
};

// Pre-visit a call expression.
bool Resource_AST_collector::pre_visit(IExpression_call *call)
{
    if (IExpression_reference const *ref = as<IExpression_reference>(call->get_reference())) {
        if (IDefinition const *def = ref->get_definition()) {
            mi::base::Handle<IModule const> owner(m_mod->get_owner_module(def));
            m_res_collector.keep_module_reference(owner);

            def = m_mod->get_original_definition(def);

            // TODO: May lead to very deep recursion. Maybe better put into working queue
            //       together with state (m_tex_usage and the call)
            ILambda_resource_enumerator::Texture_usage *arg_usages =
                m_res_collector.get_or_calc_arg_usage(owner, def);

            ILambda_resource_enumerator::Texture_usage old_tex_usage = m_tex_usage;
            for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
                IArgument const *arg = call->get_argument(i);

                if (arg_usages) {
                    if (is<IType_texture>(
                            arg->get_argument_expr()->get_type()->skip_type_alias())) {
                        m_tex_usage = old_tex_usage | arg_usages[i];
                    } else {
                        m_tex_usage = arg_usages[i];
                    }
                }

                visit(arg);
            }
            m_tex_usage = old_tex_usage;
            return false;  // already visited
        }
    }
    return true;
}

// Constructor.
Resource_collector::Resource_collector(
    IAllocator                *alloc,
    ICall_name_resolver const &name_resolver,
    Lambda_function const     &lambda_func,
    Resource_list             &textures,
    Resource_list             &light_profiles,
    Resource_list             &bsdf_measurements)
: m_arena(alloc)
, m_resolver(name_resolver)
, m_lambda_func(lambda_func)
, m_textures(textures)
, m_light_profiles(light_profiles)
, m_bsdf_measurements(bsdf_measurements)
, m_found_resources(alloc)
, m_tex_usage_map(alloc)
, m_arg_usages_by_def(alloc)
, m_arg_usages_by_sema(alloc)
, m_ast_collector(alloc, m_arena, *this, textures, light_profiles, bsdf_measurements,
    m_found_resources, m_tex_usage_map, m_arg_usages_by_def, m_arg_usages_by_sema)
, m_visited(alloc)
, m_tex_usage(0)
, m_module_set(alloc)
{
    for (size_t i = 0, n = sizeof(arg_usage_init_table) / sizeof(*arg_usage_init_table);
            i < n; ++i) {
        m_arg_usages_by_sema[arg_usage_init_table[i].sema] = arg_usage_init_table[i].arg_usage;
    }
}

void Resource_collector::collect(DAG_node const *node)
{
    // note: we don't check, whether the node was already visited, because the same node
    //       may have to be visited with different texture usages.
    //       As this is a DAG, we won't cause an endless iteration, though

    switch (node->get_kind()) {
    case DAG_node::EK_TEMPORARY:
        {
            // should not happen, but we can handle it
            DAG_temporary const *t = cast<DAG_temporary>(node);
            node = t->get_expr();
            collect(node);
        }
        break;

    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c = cast<DAG_constant>(node);
            visit_constant(c);
        }
        break;

    case DAG_node::EK_PARAMETER:
        // nothing to do, yet
        break;

    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(node);
            visit_call(call);
        }
        break;
    }
}

// Post-visit a Call.
void Resource_collector::visit_call(DAG_call const *call)
{
    ILambda_resource_enumerator::Texture_usage *arg_usages = NULL;

    IDefinition::Semantics sema = call->get_semantic();
    if (sema == IDefinition::DS_UNKNOWN) {
        // visit function and other material bodies
        char const *signature = call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<mi::mdl::IModule const> mod(
            m_resolver.get_owner_module(signature));
        if (mod.is_valid_interface()) {
            keep_module_reference(mod);

            Module const *owner = impl_cast<mi::mdl::Module>(mod.get());
            IDefinition const *def = owner->find_signature(
                signature, /*only_exported=*/false);
            MDL_ASSERT(def != NULL && "signature has no definition");
            arg_usages = get_or_calc_arg_usage(mod, def);
        }
    } else {
        // handle known functions:
        // register multiscatter BSDF data textures
        IValue_texture::Bsdf_data_kind bsdf_data_kind = IValue_texture::BDK_NONE;
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
            bsdf_data_kind = IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
            bsdf_data_kind = IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
            bsdf_data_kind = IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
            bsdf_data_kind = IValue_texture::BDK_GGX_SMITH_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
            bsdf_data_kind = IValue_texture::BDK_BECKMANN_VC_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            bsdf_data_kind = IValue_texture::BDK_GGX_VC_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
            bsdf_data_kind = IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
            bsdf_data_kind = IValue_texture::BDK_SHEEN_MULTISCATTER;
            break;
        default:
            break;
        }

        if (bsdf_data_kind != IValue_texture::BDK_NONE) {
            // check whether multiscatter_tint is a constant zero color
            DAG_node const *multiscatter_tint = call->get_argument("multiscatter_tint");
            if (multiscatter_tint != NULL &&
                    multiscatter_tint->get_kind() == DAG_node::EK_CONSTANT) {
                IValue_rgb_color const *val = as<IValue_rgb_color>(
                        cast<DAG_constant>(multiscatter_tint)->get_value());
                if (val != NULL &&
                        val->get_value(0)->get_value() == 0.f &&
                        val->get_value(1)->get_value() == 0.f &&
                        val->get_value(2)->get_value() == 0.f) {
                    // no need to use multiscatter data textures for zero multiscatter tint
                    return;
                }
            }

            // ugly: we inject a bsdf_data texture value into the lambda function here
            Lambda_function *lambda = const_cast<Lambda_function *>(&m_lambda_func);
            IValue_factory *vf = lambda->get_value_factory();
            IValue const *v = vf->create_bsdf_data_texture(
                bsdf_data_kind,
                /*tag_value=*/ 0,
                /*tag_version=*/ 0);

            // libbsdf uses tex::lookup_float3_3d()
            m_tex_usage_map[v] |= ILambda_resource_enumerator::TU_LOOKUP_FLOAT3;
            if (m_found_resources.insert(v).second)  // inserted for first time?
                m_textures.push_back(v);
        }

        Arg_usages_by_semantics::const_iterator it = m_arg_usages_by_sema.find(sema);
        if (it != m_arg_usages_by_sema.end())
            arg_usages = it->second;
    }

    ILambda_resource_enumerator::Texture_usage old_tex_usage = m_tex_usage;
    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
        DAG_node const *arg = call->get_argument(i);

        if (arg_usages) {
            if (is<IType_texture>(arg->get_type()->skip_type_alias())) {
                m_tex_usage = old_tex_usage | arg_usages[i];
            } else {
                m_tex_usage = arg_usages[i];
            }
        }

        collect(arg);
    }
    m_tex_usage = old_tex_usage;
}

// Returns the argument usage for the given definition and calculates it, if necessary.
ILambda_resource_enumerator::Texture_usage *Resource_collector::get_or_calc_arg_usage(
    mi::base::Handle<IModule const> owner, IDefinition const *def)
{
    if (def == NULL)
        return NULL;

    IDefinition::Semantics sema = def->get_semantics();
    if (sema == IDefinition::DS_UNKNOWN) {
        Arg_usages_by_def::const_iterator it = m_arg_usages_by_def.find(def);
        if (it != m_arg_usages_by_def.end())
            return it->second;

        if (!is<IDeclaration_function>(def->get_declaration()))
            return NULL;

        ILambda_resource_enumerator::Texture_usage *arg_usages =
            m_ast_collector.process_function(owner, def);
        m_arg_usages_by_def[def] = arg_usages;
        return arg_usages;
    }

    Arg_usages_by_semantics::const_iterator it = m_arg_usages_by_sema.find(sema);
    if (it != m_arg_usages_by_sema.end())
        return it->second;

    return NULL;
}

}   // anonymous

/// Enumerate all used texture resources of this lambda function.
void Lambda_function::enumerate_resources(
    ICall_name_resolver const   &resolver,
    ILambda_resource_enumerator &enumerator,
    DAG_node const              *root) const
{
    DAG_ir_walker      walker(get_allocator(), /*as_tree=*/false);
    Resource_list      textures(get_allocator());
    Resource_list      light_profiles(get_allocator());
    Resource_list      bsdf_measurements(get_allocator());
    Resource_collector collector(
        get_allocator(), resolver, *this, textures, light_profiles, bsdf_measurements);

    if (root != NULL) {
        collector.collect(root);
    } else {
        // assume that a switch function is processed
        for (Root_vector::const_iterator it(m_roots.begin()), end(m_roots.end()); it != end; ++it) {
            // Note: due to material updates holes can occur in the root range
            if (DAG_node const *root = *it) {
                collector.collect(root);
            }
        }
    }

    for (Resource_list::const_iterator it(textures.begin()), end(textures.end());
         it != end;
         ++it)
    {
        IValue const *texture = *it;
        ILambda_resource_enumerator::Texture_usage tex_usage = collector.get_texture_usage(texture);

        enumerator.texture(texture, tex_usage);
    }
    for (Resource_list::const_iterator it(light_profiles.begin()), end(light_profiles.end());
         it != end;
          ++it)
    {
        IValue const *lp = *it;

        enumerator.light_profile(lp);
    }
    for (Resource_list::const_iterator it(bsdf_measurements.begin()), end(bsdf_measurements.end());
         it != end;
         ++it)
    {
        IValue const *lp = *it;

        enumerator.bsdf_measurement(lp);
    }
}

// Register a texture resource mapping.
void Lambda_function::map_tex_resource(
    IValue::Kind                   res_kind,
    char const                     *res_url,
    IValue_texture::gamma_mode     gamma,
    IValue_texture::Bsdf_data_kind bsdf_data_kind,
    IType_texture::Shape           shape,
    int                            res_tag,
    size_t                         idx,
    bool                           valid,
    int                            width,
    int                            height,
    int                            depth)
{
    Resource_attr_entry e;
    e.index        = idx;
    e.valid        = valid;
    e.u.tex.width  = width;
    e.u.tex.height = height;
    e.u.tex.depth  = depth;
    e.u.tex.shape  = shape;
    Resource_tag_tuple::Kind kind = Resource_tag_tuple::RK_BAD;
    switch (res_kind) {
    case IValue::VK_TEXTURE:
        switch (bsdf_data_kind) {
        case IValue_texture::BDK_NONE:
            // assume real texture here
            switch (gamma) {
            case IValue_texture::gamma_default:
                kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT;
                break;
            case IValue_texture::gamma_linear:
                kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR;
                break;
            case IValue_texture::gamma_srgb:
                kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB;
                break;
            default:
                MDL_ASSERT(!"unexpected gamma mode");
                kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT;
                break;
            }
            break;
        case IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER:
            kind = Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER;
            break;
        case IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER:
            kind = Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER;
            break;
        case IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER:
            kind = Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER;
            break;
        case IValue_texture::BDK_GGX_SMITH_MULTISCATTER:
            kind = Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER;
            break;
        case IValue_texture::BDK_BECKMANN_VC_MULTISCATTER:
            kind = Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER;
            break;
        case IValue_texture::BDK_GGX_VC_MULTISCATTER:
            kind = Resource_tag_tuple::RK_GGX_VC_MULTISCATTER;
            break;
        case IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER:
            kind = Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER;
            break;
        case IValue_texture::BDK_SHEEN_MULTISCATTER:
            kind = Resource_tag_tuple::RK_SHEEN_MULTISCATTER;
            break;
        default:
            MDL_ASSERT(!"unexpected bsdf data kind");
            kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT;
            break;
        }
        break;
    case IValue::VK_INVALID_REF:
        kind = Resource_tag_tuple::RK_INVALID_REF;
        break;
    default:
        MDL_ASSERT(!"unexpected value kind");
        break;
    }

    res_url = res_url != NULL ? Arena_strdup(m_arena, res_url) : NULL;
    Resource_tag_tuple key(kind, res_url, res_tag);

    m_resource_attr_map[key] = e;
}

// Register a light profile resource mapping.
void Lambda_function::map_lp_resource(
    IValue::Kind res_kind,
    char const   *res_url,
    int          res_tag,
    size_t       idx,
    bool         valid,
    float        power,
    float        maximum)
{
    Resource_attr_entry e;
    e.index        = idx;
    e.valid        = valid;
    e.u.lp.power   = power;
    e.u.lp.maximum = maximum;

    Resource_tag_tuple::Kind kind = Resource_tag_tuple::RK_BAD;
    switch (res_kind) {
    case IValue::VK_LIGHT_PROFILE:
        kind = Resource_tag_tuple::RK_LIGHT_PROFILE;
        break;
    case IValue::VK_INVALID_REF:
        kind = Resource_tag_tuple::RK_INVALID_REF;
        break;
    default:
        MDL_ASSERT(!"unexpected value kind");
        break;
    }

    res_url = res_url != NULL ? Arena_strdup(m_arena, res_url) : NULL;
    Resource_tag_tuple key(kind, res_url, res_tag);

    m_resource_attr_map[key] = e;
}

// Register a bsdf measurement resource mapping.
void Lambda_function::map_bm_resource(
    IValue::Kind res_kind,
    char const   *res_url,
    int          res_tag,
    size_t       idx,
    bool         valid)
{
    Resource_attr_entry e;
    e.index = idx;
    e.valid = valid;

    Resource_tag_tuple::Kind kind = Resource_tag_tuple::RK_BAD;
    switch (res_kind) {
    case IValue::VK_BSDF_MEASUREMENT:
        kind = Resource_tag_tuple::RK_BSDF_MEASUREMENT;
        break;
    case IValue::VK_INVALID_REF:
        kind = Resource_tag_tuple::RK_INVALID_REF;
        break;
    default:
        MDL_ASSERT(!"unexpected value kind");
        break;
    }

    res_url = res_url != NULL ? Arena_strdup(m_arena, res_url) : NULL;
    Resource_tag_tuple key(kind, res_url, res_tag);

    m_resource_attr_map[key] = e;
}

// Analyze a lambda function.
bool Lambda_function::analyze(
    size_t                    proj,
    ICall_name_resolver const *name_resolver,
    Analysis_result           &result) const
{
    if (proj >= m_roots.size())
        return false;

    result.tangent_spaces           = 0;
    result.texture_spaces           = 0;
    result.uses_state_normal        = 0;
    result.uses_state_rc_normal     = 0;
    result.uses_texresult_lookup    = 0;

    DAG_node const *root = m_roots[proj];

    return analyze(root, name_resolver, result);
}

namespace {

/// Helper class for optimizing a DAG.
class Optimize_helper
{
public:
    /// Constructor.
    ///
    /// \param alloc          The allocator.
    /// \param mdl            The MDL compiler.
    /// \param node_factory   The node factory to use for creating optimized nodes.
    /// \param name_resolver  The name resolver to use for inlining calls.
    Optimize_helper(
        IAllocator                *alloc,
        IMDL                      &mdl,
        DAG_node_factory_impl     &node_factory,
        ICall_name_resolver const &name_resolver)
        : m_alloc(alloc)
        , m_node_factory(node_factory)
        , m_name_resolver(name_resolver)
        , m_dag_mangler(alloc, &mdl)
        , m_dag_builder(alloc, node_factory, m_dag_mangler)
        , m_optimized_nodes(0, Node_map::hasher(), Node_map::key_equal(), alloc)
    {}

    /// Optimize the given DAG node.
    ///
    /// \param node  The DAG node to optimize. It must fit to the node_factory given in the
    ///              constructor.
    ///
    /// \returns an optimized DAG node or the given node, if no optimization was applied.
    DAG_node const *optimize(DAG_node const *node)
    {
        switch (node->get_kind()) {
            case DAG_node::EK_CONSTANT:
            case DAG_node::EK_TEMPORARY:
            case DAG_node::EK_PARAMETER:
                return node;

            case DAG_node::EK_CALL:
            {
                Node_map::const_iterator it = m_optimized_nodes.find(node);
                if (it != m_optimized_nodes.end())
                    return it->second;

                DAG_call const *call = cast<DAG_call>(node);

                bool changed = false;
                int n_args = call->get_argument_count();
                VLA<DAG_call::Call_argument> args(m_alloc, n_args);
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    args[i].arg = optimize(arg);
                    if (args[i].arg != arg)
                        changed = true;
                    args[i].param_name = call->get_parameter_name(i);
                }

                // try to inline user defined functions
                if (call->get_semantic() == IDefinition::DS_UNKNOWN) {
                    mi::base::Handle<IModule const> mod(
                        m_name_resolver.get_owner_module(call->get_name()));
                    if (mod.is_valid_interface()) {
                        Module const *module = impl_cast<Module>(mod.get());
                        IDefinition const *def = module->find_signature(
                            call->get_name(),
                            /*only_exported=*/ false);
                        if (def != NULL) {
                            Module_scope module_scope(m_dag_builder, mod.get());

                            mi::base::Handle<IGenerated_code_dag const> owner_dag(
                                m_name_resolver.get_owner_dag(call->get_name()));

                            DAG_node const *res = m_dag_builder.try_inline(
                                owner_dag.get(), def, args.data(), n_args);
                            if (res != NULL) {
                                // inlining was successful, try to optimize the result further
                                res = optimize(res);
                                m_optimized_nodes[node] = res;
                                return res;
                            }
                        }
                    }
                }

                if (!changed) {
                    m_optimized_nodes[node] = node;
                    return node;
                }

                // arguments have changed, so create new version of this call
                DAG_node const *res = m_node_factory.create_call(
                    call->get_name(), call->get_semantic(),
                    args.data(), args.size(), call->get_type());
                m_optimized_nodes[node] = res;
                return res;
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
        return NULL;
    }

private:
    /// The allocator.
    IAllocator                *m_alloc;

    /// The node factory.
    DAG_node_factory_impl     &m_node_factory;

    /// The name resolver to resolve function calls.
    ICall_name_resolver const &m_name_resolver;

    /// A DAG mangler required for the DAG builder.
    DAG_mangler               m_dag_mangler;

    /// A DAG builder for inlining functions.
    DAG_builder               m_dag_builder;

    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Node_map;

    /// A map containing already optimized nodes.
    Node_map m_optimized_nodes;
};
}  // anonymous

// Optimize the lambda function.
void Lambda_function::optimize(
    ICall_name_resolver const *name_resolver,
    ICall_evaluator           *call_evaluator)
{
    // Ignore no-inline annotations which are only necessary for the material converter
    Ignore_NO_INLINE_scope scope(m_node_factory);
    ICall_evaluator *old_call_evaluator = m_node_factory.get_call_evaluator();
    m_node_factory.set_call_evaluator(call_evaluator);

    Optimize_helper optimizer(
        get_allocator(),
        *m_mdl.get(),
        m_node_factory,
        *name_resolver);

    if (!m_roots.empty()) {
        for (size_t i = 0, n = m_roots.size(); i < n; ++i) {
            m_roots[i] = optimizer.optimize(m_roots[i]);
        }
    } else if (m_body_expr != NULL) {
        m_body_expr = optimizer.optimize(m_body_expr);
    }

    m_node_factory.set_call_evaluator(old_call_evaluator);
}

// Returns true if a switch function was "modified", by adding a new root expression.
bool Lambda_function::is_modified(bool reset)
{
    bool res = m_is_modified;
    if (reset)
        m_is_modified = false;
    return res;
}

// Returns true if a switch function was "modified" by removing a root expression.
bool Lambda_function::has_dead_code() const {
    return m_has_dead_code;
}

namespace {

/// Helper class to analyse the uniform state usage.
class Uniform_state_usage : public IDAG_ir_visitor
{
public:
    /// Constructor.
    explicit Uniform_state_usage(ICall_name_resolver const &name_resolver)
    : m_resolver(name_resolver)
    , m_uses_object_id(false)
    , m_uses_transform(false)
    {
    }

    /// Check if the analyzed expression depends on state::object_id().
    bool uses_object_id() const { return m_uses_object_id; }

    /// Check if the analyzed expession depends on state::tramnsform*().
    bool uses_transform() const { return m_uses_transform; }

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
    void visit(DAG_call *call) MDL_FINAL {
        if (is_DAG_semantics(call->get_semantic())) {
            // ignore DAG nodes, these have no definition
            return;
        }

        char const *signature = call->get_name();
        mi::base::Handle<mi::mdl::IModule const> mod(m_resolver.get_owner_module(signature));
        if (mod.is_valid_interface()) {
            Module const *owner = impl_cast<mi::mdl::Module>(mod.get());

            IDefinition const *def = owner->find_signature(signature, /*only_exported=*/false);
            if (def != NULL) {
                if (def->get_property(IDefinition::DP_USES_OBJECT_ID))
                    m_uses_object_id = true;
                if (def->get_property(IDefinition::DP_USES_TRANSFORM))
                    m_uses_transform = true;
            }
        }
    }

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL {}

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL {}

private:
    /// The call name resolver.
    ICall_name_resolver const &m_resolver;

    /// True if state::object_id() may be used.
    bool m_uses_object_id;

    /// True if state::tramsform*() may be used.
    bool m_uses_transform;
};

}  // anonymous

// Pass the uniform context for a given call node.
DAG_node const *Lambda_function::set_uniform_context(
    ICall_name_resolver const *name_resolver,
    DAG_node const            *expr,
    Float4_struct const       world_to_object[4],
    Float4_struct const       object_to_world[4],
    int                       object_id)
{
    DAG_ir_walker        walker(get_allocator(), /*as_tree=*/false);
    Uniform_state_usage  visitor(*name_resolver);

    walker.walk_node(const_cast<DAG_node *>(expr), &visitor);

    if (visitor.uses_object_id()) {
        IValue_int const *v        = m_value_factory.create_int(object_id);
        DAG_node const   *c        = create_constant(v);
        IType const      *res_type = expr->get_type();

        DAG_call::Call_argument args[2];

        args[0].param_name = "object_id";
        args[0].arg        = c;

        args[1].param_name = "expr";
        args[1].arg        = expr;

        expr = create_call(
            // use the magic name here
            "set:object_id",
            IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID,
            args,
            2,
            res_type);
    }

    if (visitor.uses_transform()) {
        IValue const *v_w2o[4];
        IValue const *v_o2w[4];

        // Note: We create the matrix row-major here to match the input from the iray/irt core.
        IType_float const  *float_type = m_type_factory.create_float();
        IType_vector const *f4_type    = m_type_factory.create_vector(float_type, 4);
        IType_matrix const *m_type     = m_type_factory.create_matrix(f4_type, 4);
        for (unsigned i = 0; i < 4; ++i) {
            Float4_struct const &w2o = world_to_object[i];

            IValue const *t_w2o[4] = {
                m_value_factory.create_float(w2o.x),
                m_value_factory.create_float(w2o.y),
                m_value_factory.create_float(w2o.z),
                m_value_factory.create_float(w2o.w)
            };

            v_w2o[i] = m_value_factory.create_vector(f4_type, t_w2o, 4);

            Float4_struct const &o2w = object_to_world[i];

            IValue const *t_o2w[4] = {
                m_value_factory.create_float(o2w.x),
                m_value_factory.create_float(o2w.y),
                m_value_factory.create_float(o2w.z),
                m_value_factory.create_float(o2w.w)
            };

            v_o2w[i] = m_value_factory.create_vector(f4_type, t_o2w, 4);
        }

        IType const         *res_type = expr->get_type();

        IValue_matrix const *m_w2o = m_value_factory.create_matrix(m_type, v_w2o, 4);
        DAG_node const      *c_w2o = create_constant(m_w2o);

        IValue_matrix const *m_o2w = m_value_factory.create_matrix(m_type, v_o2w, 4);
        DAG_node const      *c_o2w = create_constant(m_o2w);

        DAG_call::Call_argument args[3];

        args[0].param_name = "world_to_object";
        args[0].arg        = c_w2o;

        args[1].param_name = "object_to_world";
        args[1].arg        = c_o2w;

        args[2].param_name = "expr";
        args[2].arg        = expr;

        expr = create_call(
            // use the magic name here
            "set:transforms",
            IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS,
            args,
            3,
            res_type);
    }

    return expr;
}

// Get a "serial version" number of this lambda function.
unsigned Lambda_function::get_serial_number() const
{
    if (!m_serial_is_valid) {
        m_serial_number   = g_next_serial++;

        // avoid serial number 0, it is used as a sentinel
        if (m_serial_number == 0u)
            m_serial_number = g_next_serial++;

        m_serial_is_valid = true;
    }
    return m_serial_number;
}

// Set the name of this lambda function.
void Lambda_function::set_name(char const *name)
{
    m_name = name == NULL ? "lambda" : name;
}

// Get the name of the lambda function.
char const *Lambda_function::get_name() const
{
    return m_name.empty() ? "lambda" : m_name.c_str();
}

// Get the hash value of this lambda function.
DAG_hash const *Lambda_function::get_hash() const
{
    if (!m_hash_is_valid) {
        update_hash();
    }
    return &m_hash;
}

namespace {

class Ast_analysis : public Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param alloc    the allocator
    /// \param owner    the owner module of the analyzed function
    /// \param result   the analysis result
    Ast_analysis(
        IAllocator                        *alloc,
        IModule const                     *owner,
        ILambda_function::Analysis_result &result,
        Array_ref<IValue const *> const   &args)
    : m_owner(owner)
    , m_function_depth(0)
    , m_top_level_args(args)
    , m_result(result)
    , m_error(false)
    {
    }

    /// Return true if some error occurred.
    bool found_error() const {
        return m_error;
    }

    /// Pre visit a function declaration.
    bool pre_visit(IDeclaration_function *fkt_decl) MDL_FINAL
    {
        ++m_function_depth;

        // analyze further
        return true;
    }

    /// Post visit a function declaration.
    void post_visit(IDeclaration_function *fkt_decl) MDL_FINAL
    {
        --m_function_depth;
    }

    /// Post visit of an call
    IExpression *post_visit(IExpression_call *call) MDL_FINAL
    {
        if (m_error) {
            // stop here, error will not be better
            return call;
        }

        // assume the AST error free
        IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());
        if (ref->is_array_constructor()) {
            return call;
        }

        IDefinition const *def = ref->get_definition();

        switch (def->get_semantics()) {
        case IDefinition::DS_UNKNOWN:
            if (!analyze_unknown_call(call)) {
                // could not analyze
                m_error = true;
                m_result.tangent_spaces        = ~0u;
                m_result.texture_spaces        = ~0u;
                m_result.uses_state_normal     = 1;
                m_result.uses_state_rc_normal  = 1;
                m_result.uses_texresult_lookup = 1;
            }
            break;

        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
            MDL_ASSERT(call->get_argument_count() == 1);
            analyze_space(call->get_argument(0)->get_argument_expr(), m_result.tangent_spaces);
            break;

        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
            MDL_ASSERT(call->get_argument_count() == 1);
            analyze_space(call->get_argument(0)->get_argument_expr(), m_result.texture_spaces);
            break;

        case IDefinition::DS_INTRINSIC_STATE_NORMAL:
            m_result.uses_state_normal = 1;
            break;

        case IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
            m_result.uses_state_rc_normal = 1;
            break;

        case IDefinition::DS_INTRINSIC_JIT_LOOKUP:
            m_result.uses_texresult_lookup = 1;
            break;

        default:
            // all others have a known semantic and can be safely ignored.
            break;
        }
        return call;
    }

private:
    /// A constructor from parent.
    Ast_analysis(Ast_analysis &parent, IModule const *owner)
    : m_owner(owner)
    , m_function_depth(parent.m_function_depth)
    , m_top_level_args(parent.m_top_level_args)
    , m_result(parent.m_result)
    , m_error(false)
    {
    }

    /// Get the constant value of the index' argument if any.
    IValue const *get_argument_value(size_t index) const
    {
        if (m_function_depth != 1)
            return NULL;

        if (index >= m_top_level_args.size())
            return NULL;

        return m_top_level_args[index];
    }

    /// Analyze the coordinate space expression.
    void analyze_space(IExpression const *expr, unsigned &bitmap)
    {
        IValue_int_valued const *v = NULL;

        switch (expr->get_kind()) {
        case IExpression::EK_LITERAL:
            {
                IExpression_literal const *l = cast<IExpression_literal>(expr);
                v = cast<IValue_int_valued>(l->get_value());
            }
            break;
        case IExpression::EK_REFERENCE:
            {
                IExpression_reference const *ref = cast<IExpression_reference>(expr);
                IDefinition const           *def = ref->get_definition();

                if (def != NULL && def->get_kind() == IDefinition::DK_PARAMETER) {
                    // is a parameter, try inter-procedural
                    size_t       idx = def->get_parameter_index();
                    IValue const *cv = get_argument_value(idx);

                    if (cv != NULL) {
                        // the parameter has a constant value argument, good
                        v = cast<IValue_int_valued>(cv);
                    }
                }
            }
            break;
        default:
            // unsupported yet
            break;
        }
        if (v != NULL) {
            int space = v->get_value();

            if (0 <= space && space < 32) {
                bitmap |= 1U << space;
            } else {
                // out of bounds
                bitmap = ~0u;
            }
        } else {
            // could not determine the value of the argument
            bitmap = ~0u;
        }
    }

    /// Analyze a call to a user defined function.
    bool analyze_unknown_call(IExpression_call const *call)
    {
        IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());
        IDefinition const           *def = ref->get_definition();

        mi::base::Handle<IModule const> owner(m_owner->get_owner_module(def));
        def = m_owner->get_original_definition(def);

        IDeclaration_function const *func = as<IDeclaration_function>(def->get_declaration());
        if (func == NULL) {
            // unexpected
            return false;
        }

        Ast_analysis analysis(*this, owner.get());
        analysis.visit(func);

        m_error |= analysis.found_error();

        return true;
    }

private:
    /// The owner module of the analyzed function.
    IModule const                     *m_owner;

    /// Current analyzed function depth.
    size_t                            m_function_depth;

    /// Top level constant arguments.
    Array_ref<IValue const *> const   &m_top_level_args;

    /// The analysis result.
    ILambda_function::Analysis_result &m_result;

    /// Set to true once an error occurred.
    bool                              m_error;
};

/// Helper class to analyze the state usage.
class State_usage_analysis : public IDAG_ir_visitor {
public:
    /// Constructor.
    ///
    /// \param alloc          the allocator
    /// \param name_resolver  the call name resolver
    /// \param result         the analysis result
    State_usage_analysis(
        IAllocator                        *alloc,
        ICall_name_resolver const         &name_resolver,
        ILambda_function::Analysis_result &result)
    : m_alloc(alloc)
    , m_resolver(name_resolver)
    , m_result(result)
    {
    }

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL {
        // ignore
    }

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL {
        // ignore
    }

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL {
        IDefinition::Semantics sema = call->get_semantic();

        switch (sema) {
        case IDefinition::DS_UNKNOWN:
            if (!analyze_unknown_call(call)) {
                // could not analyze
                m_result.tangent_spaces = ~0;
            }
            break;

        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
            MDL_ASSERT(call->get_argument_count() == 1);
            analyze_space(call->get_argument(0), m_result.tangent_spaces);
            break;

        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
            MDL_ASSERT(call->get_argument_count() == 1);
            analyze_space(call->get_argument(0), m_result.texture_spaces);
            break;

        case IDefinition::DS_INTRINSIC_STATE_NORMAL:
            m_result.uses_state_normal = 1;
            break;

        case IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
            m_result.uses_state_rc_normal = 1;
            break;

        default:
            // all others have a known semantic and can be safely ignored.
            break;
        }
    }

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL {
        // ignore
    }

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL {
        // ignore
    }

private:
    /// Analyze a call to an unknown function.
    ///
    /// \param call  the DAG_call node to analyze
    ///
    /// \return true on success, false if analysis failed
    bool analyze_unknown_call(DAG_call const *call)
    {
        char const *signature = call->get_name();
        mi::base::Handle<mi::mdl::IModule const> mod(m_resolver.get_owner_module(signature));
        if (!mod.is_valid_interface())
            return false;
        mi::mdl::Module const *owner = impl_cast<mi::mdl::Module>(mod.get());

        IDefinition const *def = owner->find_signature(signature, /*only_exported=*/false);
        if (def == NULL)
            return false;

        IDeclaration const *decl = def->get_declaration();
        if (decl == NULL)
            return false;

        if (IDeclaration_function const *func = as<IDeclaration_function>(decl)) {
            size_t n_args = call->get_argument_count();

            // collect constant arguments for vary simple inter-procedural analysis
            VLA<IValue const *> const_args(m_alloc, n_args);
            for (size_t i = 0; i < n_args; ++i) {
                DAG_node const *arg = call->get_argument(i);

                if (DAG_constant const *c = as<DAG_constant>(arg)) {
                    const_args[i] = c->get_value();
                } else {
                    const_args[i] = NULL;
                }
            }

            return analyze_function_ast(owner, func, const_args);
        }

        // unsupported
        return false;
    }

    /// Analyze the AST of a function.
    ///
    /// \param owner  the owner module of the function to analyze
    /// \param func   the declaration of the function
    ///
    /// \return true on success, false if analysis failed
    bool analyze_function_ast(
        IModule const                   *owner,
        IDeclaration_function const     *func,
        Array_ref<IValue const *> const &const_args)
    {
        Ast_analysis analysis(m_alloc, owner, m_result, const_args);

        analysis.visit(func);

        return !analysis.found_error();
    }

    /// Analyze a space.
    ///
    /// \param node    the space
    /// \param bitmap  the result bitmap
    static void analyze_space(DAG_node const *node, unsigned &bitmap)
    {
        if (DAG_constant const *c = as<DAG_constant>(node)) {
            IValue_int const *iv = cast<IValue_int>(c->get_value());
            int space = iv->get_value();
            if (0 <= space && space < 32) {
                bitmap |= 1U << space;
            } else {
                // out of range, should not happen
                bitmap = ~0;
            }
            return;
        } else {
            // for now, unsupported
            bitmap = ~0;
        }
    }

private:
    /// The allocator.
    IAllocator                        *m_alloc;

    /// The call name resolver.
    ICall_name_resolver const         &m_resolver;

    /// The analysis result.
    ILambda_function::Analysis_result &m_result;
};

}  // anonymous

// Analyze an expression function.
bool Lambda_function::analyze(
    DAG_node const            *expr,
    ICall_name_resolver const *resolver,
    Analysis_result           &result) const
{
    DAG_ir_walker        walker(get_allocator(), /*as_tree=*/false);
    State_usage_analysis analysis(get_allocator(), *resolver, result);

    walker.walk_node(const_cast<DAG_node *>(expr), &analysis);

    return true;
}

namespace {

typedef std::pair<Resource_tag_tuple, size_t> Entry;

struct Entry_compare {
    bool operator()(Entry const &a, Entry const &b)
    {
        size_t a_index = a.second;
        size_t b_index = b.second;

        if (a_index != b_index)
            return a_index < b_index;

        Resource_tag_tuple const &a_t = a.first;
        Resource_tag_tuple const &b_t = b.first;

        if (a_t.m_kind != b_t.m_kind)
            return a_t.m_kind < b_t.m_kind;
        if (a_t.m_tag != b_t.m_tag)
            return a_t.m_tag < b_t.m_tag;
        if (a_t.m_url == NULL)
            return b_t.m_url != NULL;
        if (b_t.m_url == NULL)
            return false;
        return strcmp(a_t.m_url, b_t.m_url) < 0;
    }
};

}  // anonymous

// Update the hash value.
void Lambda_function::update_hash() const
{
    MD5_hasher md5_hasher;
    Dag_hasher dag_hasher(md5_hasher);

    // Important: walk as a tree here
    DAG_ir_walker walker(get_allocator(), /*as_tree=*/true);

    for (size_t i = 0, n = get_parameter_count(); i < n; ++i) {
        char const  *name = get_parameter_name(i);
        IType const *tp = get_parameter_type(i);

        dag_hasher.hash_parameter(name, tp);
    }

    if (!m_roots.empty()) {
        for (size_t i = 0, n = m_roots.size(); i < n; ++i) {
            DAG_node const *root = m_roots[i];
            if (root == NULL)
                md5_hasher.update(0);  // update hash to be able to differentiate different orders
            else
                walker.walk_node(const_cast<DAG_node *>(root), &dag_hasher);
        }
    } else {
        walker.walk_node(const_cast<DAG_node *>(m_body_expr), &dag_hasher);
    }

    // hash the resource attribute map, but sort it first.
    // Note that we hash only the resource tag tuple AND (implicitly) its index, not
    // the resource attributes, we assume same tags means same attributes here.
    if (m_has_resource_attributes) {
        vector<Entry>::Type resources(get_allocator());

        for (Resource_attr_map::const_iterator
            it(m_resource_attr_map.begin()), end(m_resource_attr_map.end());
            it != end;
            ++it)
        {
            Resource_tag_tuple const  &t = it->first;
            Resource_attr_entry const &e = it->second;

            resources.push_back(std::make_pair(t, e.index));
        }

        // sort the entries, to make it deterministic
        std::sort(resources.begin(), resources.end(), Entry_compare());

        for (size_t i = 0, n = resources.size(); i < n; ++i) {
            Entry const              &e     = resources[i];
            Resource_tag_tuple const &t     = e.first;
            size_t                   index  = e.second;

            md5_hasher.update(t.m_kind);
            md5_hasher.update(mi::Uint64(index));

            if (t.m_kind != Resource_tag_tuple::RK_BAD) {
                md5_hasher.update(t.m_tag);
                md5_hasher.update(t.m_url);
            }
        }
    }

    md5_hasher.final(m_hash.data());

    m_hash_is_valid = true;
}


// Get the return type of the lambda function.
mi::mdl::IType const *Lambda_function::get_return_type() const
{
    if (!m_roots.empty()) {
        // If this lambda has root nodes, it will return an union
        // passed via the first parameter and a bool if the expression
        // was successfully evaluated.
        return m_type_factory.create_bool();
    }
    return m_body_expr->get_type();
}

// Returns the number of parameters of this lambda function.
size_t Lambda_function::get_parameter_count() const
{
    return m_params.size();
}

// Return the type of the i'th parameter.
mi::mdl::IType const *Lambda_function::get_parameter_type(size_t i) const
{
    if (i < m_params.size())
        return m_params[i].m_type;
    return NULL;
}

// Return the name of the i'th parameter.
char const *Lambda_function::get_parameter_name(size_t i) const
{
    if (i < m_params.size())
        return m_params[i].m_name;
    return NULL;
}

// Add a new "captured" parameter.
size_t Lambda_function::add_parameter(
    mi::mdl::IType const *type,
    char const           *name)
{
    type = m_type_factory.import(type);
    name = Arena_strdup(m_arena, name);

    m_params.push_back(Parameter_info(type, name));
    return m_params.size() - 1;
}

// Map material parameter i to lambda parameter j
void Lambda_function::set_parameter_mapping(size_t i, size_t j)
{
    m_index_map[i] = j;
}

// Initialize the derivative information for this lambda function.
// This rewrites the body/sub-expressions with derivative types.
void Lambda_function::initialize_derivative_infos(ICall_name_resolver const *resolver)
{
    // optimize the expressions here, forcing inlining of code when possible.
    // We need to do this before calculating the derivative information, because the
    // inlining won't update the derivative information
    optimize(resolver, NULL);

    // collect information and rebuild DAG with derivative types
    m_deriv_infos.set_call_name_resolver(resolver);

    Deriv_DAG_builder deriv_builder(get_allocator(), *this, m_deriv_infos);

    if (!m_roots.empty()) {
        for (size_t i = 0, n = m_roots.size(); i < n; ++i) {
            m_roots[i] = import_expr(m_roots[i]);
            m_roots[i] = deriv_builder.rebuild(m_roots[i], /*want_derivatives=*/ false);
        }
    } else {
        m_body_expr = import_expr(m_body_expr);
        m_body_expr = deriv_builder.rebuild(m_body_expr, /*want_derivatives=*/ false);
    }
    m_deriv_infos.set_call_name_resolver(NULL);

    m_deriv_infos_calculated = true;
}

// Returns true, if the attributes in the resource attribute table are valid.
// If false, only the indices are valid.
bool Lambda_function::has_resource_attributes() const
{
    return m_has_resource_attributes;
}

// Sets whether the resource attribute table contains valid attributes.
void Lambda_function::set_has_resource_attributes(bool avail)
{
    m_has_resource_attributes = avail;
}

// Set a tag, version pair for a resource value that might be reachable from this function.
void Lambda_function::set_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                    *res_url,
    int                           tag)
{
    int old_tag = find_resource_tag(res_kind, res_url);

    if (old_tag == 0) {
        add_resource_tag(res_kind, res_url, tag);
    }
    else {
        MDL_ASSERT(old_tag == tag && "Changing tag of a resource");
    }
}

// Remap a resource value according to the resource map.
int Lambda_function::get_resource_tag(IValue_resource const *r) const
{
    int tag = find_resource_tag(kind_from_value(r), r->get_string_value());
    if (tag == 0) {
        tag = r->get_tag_value();
    }
    return tag;
}

// Get the number of entires in the resource map.
size_t Lambda_function::get_resource_entries_count() const
{
    return m_resource_tag_map.size();
}

// Get the i'th entry of the resource map.
Resource_tag_tuple const *Lambda_function::get_resource_entry(size_t index) const
{
    if (index < m_resource_tag_map.size())
        return &m_resource_tag_map[index];
    return NULL;
}

// Find the resource tag of a resource.
int Lambda_function::find_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url) const
{
    // linear search so far
    for (size_t i = 0, n = m_resource_tag_map.size(); i < n; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        // beware of NULL pointer
        if (e.m_kind == res_kind && (e.m_url == res_url || strcmp(e.m_url, res_url) == 0))
            return e.m_tag;
    }
    return 0;
}

// Add tag, version pair for a resource value that might be reachable from this function.
void Lambda_function::add_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url,
    int                            tag)
{
    res_url = res_url != NULL ? Arena_strdup(m_arena, res_url) : NULL;
    m_resource_tag_map.push_back(Resource_tag_tuple(res_kind, res_url, tag));
}

// Get the derivative information if they have been initialized.
Derivative_infos const *Lambda_function::get_derivative_infos() const
{
    if (!m_deriv_infos_calculated)
        return NULL;
    return &m_deriv_infos;
}

// Returns true if the given semantic belongs to a varying state function.
bool Lambda_function::is_varying_state_semantic(IDefinition::Semantics sema)
{
    if (is_state_semantics(sema)) {
        switch (sema) {
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        case IDefinition::DS_INTRINSIC_STATE_OBJECT_ID:
        case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MIN:
        case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MAX:
            // these have uniform results
            return false;
        default:
            break;
        }
        return true;
    }
    return false;
}

// Check if the given DAG expression may use varying state data.
bool Lambda_function::may_use_varying_state(
    ICall_name_resolver const *resolver,
    DAG_node const            *expr) const
{
    for (;;) {
        switch (expr->get_kind()) {
        case DAG_node::EK_CONSTANT:
            return false;
        case DAG_node::EK_TEMPORARY:
            {
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                continue;
            }
        case DAG_node::EK_CALL:
            {
                DAG_call const                  *call = cast<DAG_call>(expr);
                mi::mdl::IDefinition::Semantics sema  = call->get_semantic();

                if (is_varying_state_semantic(sema))
                    return true;

                // handle the DAG intrinsics here, they don't have a definition
                switch (sema) {
                case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
                case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
                case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
                    // those never access the state
                    break;
                default:
                    if (!semantic_is_operator(sema)) {
                        // lookup the definition and check it
                        char const *signature = call->get_name();
                        mi::base::Handle<mi::mdl::IModule const> mod(
                            resolver->get_owner_module(signature));
                        if (!mod.is_valid_interface()) {
                            MDL_ASSERT(!"module resolver found unsupported module");
                            return true;
                        }
                        mi::mdl::Module const      *module = impl_cast<mi::mdl::Module>(mod.get());
                        mi::mdl::IDefinition const *def    =
                            module->find_signature(signature, /*only_exported=*/false);
                        if (def != NULL &&
                            def->get_property(mi::mdl::IDefinition::DP_USES_VARYING_STATE))
                        {
                            return true;
                        }
                    }
                }

                // check arguments
                for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
                    DAG_node const *arg = call->get_argument(i);

                    if (may_use_varying_state(resolver, arg))
                        return true;
                }
                return false;
            }
        case DAG_node::EK_PARAMETER:
            return false;
        }
        MDL_ASSERT(!"unsupported DAG node kind");
    }
}

// Serialize this lambda function to the given serializer.
void Lambda_function::serialize(ISerializer *is) const
{
    IAllocator            *alloc = get_allocator();
    MDL_binary_serializer bin_serializer(alloc, m_mdl.get(), is);
    DAG_serializer        dag_serializer(alloc, is, &bin_serializer);

    bin_serializer.write_section_tag(Serializer::ST_LAMBDA_START);

    dag_serializer.write_unsigned(m_context);

    // will be automatically set on deserialization.
    // m_mdl, m_arena
    m_sym_tab.serialize(dag_serializer);
    m_type_factory.serialize(dag_serializer);
    m_value_factory.serialize(dag_serializer);

    // The jitted code singleton will be set on deserialization.
    // m_jitted_code;

    dag_serializer.write_cstring(m_name.c_str());

    // serialize the factory
    DAG_serializer::Dag_vector exprs(alloc);
    if (m_body_expr != NULL)
        exprs.push_back(m_body_expr);

    DAG_serializer::Dag_vector const *roots[] = {
        &m_roots,
        &exprs
    };

    dag_serializer.write_dags(roots, dimension_of(roots));

    // serialize m_roots
    size_t n_roots = m_roots.size();
    dag_serializer.write_encoded_tag(n_roots);
    for (size_t i = 0; i < n_roots; ++i) {
        DAG_node const *root = m_roots[i];
        if (root != NULL) {
            dag_serializer.write_bool(true);
            dag_serializer.write_encoded(root);
        } else {
            dag_serializer.write_bool(false);
        }
    }
    // serialize m_expr
    if (m_body_expr != NULL) {
        dag_serializer.write_bool(true);
        dag_serializer.write_encoded(m_body_expr);
    } else {
        dag_serializer.write_bool(false);
    }

    // serialize parameter map
    size_t n_params = m_params.size();
    dag_serializer.write_encoded_tag(n_params);
    for (size_t i = 0; i < n_params; ++i) {
        Parameter_info const &param = m_params[i];
        dag_serializer.write_encoded(param.m_type);
        dag_serializer.write_cstring(param.m_name);
    }

    // serialize index map
    dag_serializer.write_encoded_tag(m_index_map.size());
    for (Index_map::const_iterator it = m_index_map.begin(), end = m_index_map.end();
            it != end; ++it) {
        dag_serializer.write_encoded_tag(it->first);
        dag_serializer.write_encoded_tag(it->second);
    }

    // serialize the root map AFTER all expressions
    dag_serializer.write_encoded_tag(m_root_map.size());
    for (Root_map::const_iterator it(m_root_map.begin()), end(m_root_map.end()); it != end; ++it) {
        DAG_node const *node = it->first;
        size_t          idx  = it->second;

        dag_serializer.write_encoded(node);
        dag_serializer.write_encoded_tag(idx);
    }

    // serialize the resource-index-map after all expressions, so all values are known
    dag_serializer.write_encoded_tag(m_resource_attr_map.size());
    for (Resource_attr_map::const_iterator it(m_resource_attr_map.begin()),
         end(m_resource_attr_map.end());
         it != end;
         ++it)
    {
        Resource_tag_tuple const &k = it->first;
        dag_serializer.write_encoded(k.m_kind);
        dag_serializer.write_encoded(k.m_url);
        dag_serializer.write_db_tag(k.m_tag);

        Resource_attr_entry const &e = it->second;
        dag_serializer.write_encoded_tag(e.index);
        dag_serializer.write_bool(e.valid);

        switch (k.m_kind) {
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
        case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
        case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
            dag_serializer.write_int(e.u.tex.width);
            dag_serializer.write_int(e.u.tex.height);
            dag_serializer.write_int(e.u.tex.depth);
            dag_serializer.write_int(e.u.tex.shape);
            break;
        case Resource_tag_tuple::RK_LIGHT_PROFILE:
            dag_serializer.write_float(e.u.lp.power);
            dag_serializer.write_float(e.u.lp.maximum);
            break;
        case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
            break;
        default:
            MDL_ASSERT(!"unexpected resource kind");
        }
    }

    dag_serializer.write_bool(m_has_resource_attributes);
    dag_serializer.write_bool(m_uses_render_state);
    dag_serializer.write_bool(m_has_dead_code);
    dag_serializer.write_bool(m_uses_lambda_results);

    // The serial number is not serialized, but a new one is drawn:
    // Otherwise it is not possible to keep them in sync over the network ...
    // m_serial_number
    // m_serial_is_valid

    // hash values are not serialized but recomputed
    // m_hash_is_valid

    bin_serializer.write_section_tag(Serializer::ST_LAMBDA_END);
}

// Deserialize a lambda function from the given deserializer.
Lambda_function *Lambda_function::deserialize(
    IAllocator    *alloc,
    MDL           *mdl,
    IDeserializer *ds)
{
    MDL_binary_deserializer bin_deserializer(alloc, ds, mdl);
    DAG_deserializer        dag_deserializer(ds, &bin_deserializer);

    Tag_t tag;

    tag = bin_deserializer.read_section_tag();
    MDL_ASSERT(tag == Serializer::ST_LAMBDA_START);
    MI::STLEXT::no_unused_variable_warning_please(tag);

    // context first, needed to create the object
    Lambda_execution_context lec = Lambda_execution_context(dag_deserializer.read_unsigned());

    Allocator_builder builder(alloc);
    Lambda_function *res = builder.create<Lambda_function>(
        alloc,
        mdl,
        lec);

    // already set during creation:
    // m_mdl, m_arena

    res->m_sym_tab.deserialize(dag_deserializer);
    res->m_type_factory.deserialize(dag_deserializer);
    res->m_value_factory.deserialize(dag_deserializer);

    // Already set during creation.
    // m_jitted_code

    res->m_name = dag_deserializer.read_cstring();

    // deserialize the node factory m_node_factory by deserializing all reachable DAGs
    dag_deserializer.read_dags(res->m_node_factory);

    // deserialize m_roots
    size_t n_roots = dag_deserializer.read_encoded_tag();
    for (size_t i = 0; i < n_roots; ++i) {
        if (dag_deserializer.read_bool()) {
            DAG_node const *root = dag_deserializer.read_encoded<DAG_node const *>();
            res->m_roots.push_back(root);
        } else {
            DAG_node const *root = NULL;
            res->m_roots.push_back(root);
        }
    }
    // deserialize m_expr
    if (dag_deserializer.read_bool()) {
        res->m_body_expr = dag_deserializer.read_encoded<DAG_node const *>();
    }

    // deserialize parameter map
    size_t n_params = dag_deserializer.read_encoded_tag();
    for (size_t i = 0; i < n_params; ++i) {
        IType const *type = dag_deserializer.read_type(res->m_type_factory);
        char const  *name = dag_deserializer.read_cstring();

        res->m_params.push_back(Parameter_info(type, name));
    }

    // deserialize index map
    size_t n_index_mappings = dag_deserializer.read_encoded_tag();
    for (size_t i = 0; i < n_index_mappings; ++i) {
        size_t from_idx = dag_deserializer.read_encoded_tag();
        size_t to_idx   = dag_deserializer.read_encoded_tag();

        res->m_index_map[from_idx] = to_idx;
    }

    // deserialize the root map AFTER all expressions
    size_t len = dag_deserializer.read_encoded_tag();
    for (size_t i = 0; i < len; ++i) {
        DAG_node const *node = dag_deserializer.read_encoded<DAG_node const *>();
        size_t          idx  = dag_deserializer.read_encoded_tag();

        res->m_root_map[node] = idx;
    }

    // deserialize the resource-index-map AFTER all expressions
    size_t mlen = dag_deserializer.read_encoded_tag();
    for (size_t i = 0; i < mlen; ++i) {
        Resource_tag_tuple k;

        k.m_kind    = dag_deserializer.read_encoded<Resource_tag_tuple::Kind>();
        string url  = dag_deserializer.read_encoded<string>();
        k.m_url     = Arena_strdup(res->m_arena, url.c_str());
        k.m_tag     = dag_deserializer.read_db_tag();

        Resource_attr_entry e;
        e.index = dag_deserializer.read_encoded_tag();
        e.valid = dag_deserializer.read_bool();

        switch (k.m_kind) {
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
        case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
        case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
            e.u.tex.width  = dag_deserializer.read_int();
            e.u.tex.height = dag_deserializer.read_int();
            e.u.tex.depth  = dag_deserializer.read_int();
            e.u.tex.shape  = static_cast<IType_texture::Shape>(dag_deserializer.read_int());
            break;
        case Resource_tag_tuple::RK_LIGHT_PROFILE:
            e.u.lp.power   = dag_deserializer.read_float();
            e.u.lp.maximum = dag_deserializer.read_float();
            break;
        case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
            break;
        default:
            MDL_ASSERT(!"unexpected resource kind");
        }

        res->m_resource_attr_map[k] = e;
    }

    res->m_has_resource_attributes = dag_deserializer.read_bool();
    res->m_uses_render_state       = dag_deserializer.read_bool();
    res->m_has_dead_code           = dag_deserializer.read_bool();
    res->m_uses_lambda_results     = dag_deserializer.read_bool();

    // The serial number is not serialized, but a new one is drawn:
    // Otherwise it is not possible to keep them in sync over the network ...
    res->m_serial_number   = 0;
    res->m_serial_is_valid = false;

    // The hash is not serialized but recalculated
    res->m_hash_is_valid = false;

    tag = bin_deserializer.read_section_tag();
    MDL_ASSERT(tag == Serializer::ST_LAMBDA_END);
    MI::STLEXT::no_unused_variable_warning_please(tag);

    return res;
}

// Checks if the uniform state was set.
bool Lambda_function::is_uniform_state_set() const
{
    if (size_t n = get_root_expr_count()) {
        for (size_t i = 0; i < n; ++i) {
            if (DAG_node const *root = get_root_expr(i)) {
                if (DAG_call const *call = as<DAG_call>(root)) {
                    if (call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS ||
                        call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID)
                    {
                        // if it is set on one, it works for all
                        return true;
                    }
                }
            }
        }
    } else if (DAG_node const *root = get_body()) {
        if (DAG_call const *call = as<DAG_call>(root)) {
            return
                call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS ||
                call->get_semantic() == IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID;
        }
    }
    return false;
}

// Dump a lambda expression to a .gv file.
void Lambda_function::dump(DAG_node const *expr, char const *name) const
{
    string fname(name, get_allocator());
    fname += "_lambda.gv";

    Allocator_builder builder(get_allocator());

    if (FILE *f = fopen(fname.c_str(), "w")) {
        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(get_allocator(), f, /*close_at_destroy=*/true));

        Lambda_dumper dumper(get_allocator(), out.get());

        dumper.dump(expr);
    }
}

namespace {

/// Get a DAG node along the given path.
DAG_node const *get_dag_arg(
    DAG_node const *node,
    Array_ref<char const *> const &path,
    IDag_builder *dag_builder)
{
    if (node == NULL) return node;
    if (path.size() == 0) return node;

    switch(node->get_kind())
    {
    case DAG_node::EK_CONSTANT:
    {
        DAG_constant const *c = as<DAG_constant>(node);
        if (IValue_compound const *vc = as<IValue_compound>(c->get_value())) {
            if (IValue const *subval = vc->get_value(path[0])) {
                node = dag_builder->create_constant(subval);
                return get_dag_arg(node, path.slice(1), dag_builder);
            }
        }
        return NULL;
    }
    case DAG_node::EK_TEMPORARY:
    {
        DAG_temporary const *temp = as<DAG_temporary>(node);
        return get_dag_arg(temp->get_expr(), path, dag_builder);
    }
    case DAG_node::EK_CALL:
    {
        DAG_call const *call = static_cast<DAG_call const *>(node);
        node = call->get_argument(path[0]);
        return get_dag_arg(node, path.slice(1), dag_builder);
    }
    case DAG_node::EK_PARAMETER:
        return NULL;
    }
    MDL_ASSERT(!"unknown DAG node kind");
    return NULL;
}

/// Helper class for building distribution functions.
/// Creates expression lambda functions for all non-DF DAG nodes.
class Distribution_function_builder
{
public:
    enum Eval_state {
        ES_BEGIN_STATE,             ///< The node is evaluated before the end of the evaluation
                                    ///< of geometry.normal.
        ES_AFTER_GEOMETRY_NORMAL    ///< The node is evaluated after geometry.normal has been
                                    ///< evaluated.
    };

    enum Flag {
        FL_NONE                             = 0,        ///< No flags.
        FL_NEEDS_MATERIAL_IOR               = 1 << 0,   ///< Material needs material.ior.
        FL_NEEDS_MATERIAL_THIN_WALLED       = 1 << 1,   ///< Material needs material.thin_walled.
        FL_NEEDS_MATERIAL_VOLUME_ABSORPTION = 1 << 2,   ///< Material needs material.volume
                                                        ///< .absorption_coefficient.
        FL_CONTAINS_UNSUPPORTED_DF          = 1 << 3,   ///< Material contains unsupported DFs.
    };  // can be or'ed

    typedef unsigned Flags;

    /// Builds a distribution function.
    static IDistribution_function::Error_code build(
        IDistribution_function                     *idist_func,
        IAllocator                                 *alloc,
        IMDL                                       *compiler,
        ICall_name_resolver const                  *resolver,
        DAG_node const                             *mat_root_node,
        IDistribution_function::Requested_function *requested_functions,
        size_t                                      num_functions,
        bool                                        include_geometry_normal,
        bool                                        calc_derivative_infos,
        bool                                        allow_double_expr_lambdas)
    {
        if (mat_root_node == NULL || num_functions == 0 || requested_functions == NULL)
            return IDistribution_function::EC_INVALID_PARAMETERS;

        mi::base::Handle<Lambda_function> root_lambda(
            impl_cast<Lambda_function>(idist_func->get_root_lambda()));

        Distribution_function *dist_func = impl_cast<Distribution_function>(idist_func);

        if (calc_derivative_infos) {
            // calculate derivative information and rebuild DAG with derivative types
            Derivative_infos *deriv_infos = dist_func->get_writable_derivative_infos();
            deriv_infos->set_call_name_resolver(resolver);

            Deriv_DAG_builder deriv_builder(alloc, *root_lambda.get(), *deriv_infos);
            mat_root_node = deriv_builder.rebuild(mat_root_node, /*want_derivatives=*/ false);

            deriv_infos->set_call_name_resolver(NULL);
        }

        // translate all non-df nodes to call_lambda nodes
        Distribution_function_builder mat_builder(
            alloc,
            *dist_func,
            mat_root_node,
            compiler,
            resolver,
            calc_derivative_infos,
            allow_double_expr_lambdas);
        unsigned walk_id = 0;

        struct Request {
            Request(
                size_t reqfunc_index,
                DAG_node const *node,
                Distribution_function_builder::Eval_state eval_state)
            : reqfunc_index(reqfunc_index)
            , node(node)
            , eval_state(eval_state)
            {}

            size_t reqfunc_index;
            DAG_node const *node;
            Distribution_function_builder::Eval_state eval_state;
        };

        vector<Request>::Type requests(alloc);

        IDistribution_function::Error_code last_error = IDistribution_function::EC_NONE;

        for (size_t path_idx = 0; path_idx < num_functions; ++path_idx) {
            if (requested_functions[path_idx].path == NULL) {
                last_error = requested_functions[path_idx].error_code =
                    IDistribution_function::EC_INVALID_PATH;
                continue;
            }
            string path_copy(requested_functions[path_idx].path, alloc);

            // split path at '.'
            vector<char const *>::Type path_parts(alloc);
            size_t last_start = 0;
            for (size_t i = 0, n = path_copy.length(); i < n; ++i) {
                if (path_copy[i] == '.') {
                    path_copy[i] = 0;
                    path_parts.push_back(path_copy.c_str() + last_start);
                    last_start = i + 1;
                }
            }
            if (last_start < path_copy.length())
                path_parts.push_back(path_copy.c_str() + last_start);

            DAG_node const *node = get_dag_arg(mat_root_node, path_parts, root_lambda.get());
            if (node == NULL) {
                last_error = requested_functions[path_idx].error_code =
                    IDistribution_function::EC_INVALID_PATH;
                continue;
            }

            IType const *node_type = node->get_type()->skip_type_alias();
            switch (node_type->get_kind()) {
            case IType::TK_BSDF:
            case IType::TK_HAIR_BSDF:
            case IType::TK_EDF:
            case IType::TK_BOOL:
            case IType::TK_INT:
            case IType::TK_ENUM:
            case IType::TK_FLOAT:
            case IType::TK_DOUBLE:
            case IType::TK_STRING:
            case IType::TK_VECTOR:
            case IType::TK_MATRIX:
            case IType::TK_ARRAY:   // TODO: hmmm, does this work?
            case IType::TK_COLOR:
                break;

            case IType::TK_STRUCT:
                if (contains_df_type(node_type)) {
                    last_error = requested_functions[path_idx].error_code =
                        IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE;
                    continue;
                }
                break;

            case IType::TK_VDF:
                last_error = requested_functions[path_idx].error_code =
                    IDistribution_function::EC_UNSUPPORTED_DISTRIBUTION_TYPE;
                continue;

            case IType::TK_LIGHT_PROFILE:
            case IType::TK_TEXTURE:
            case IType::TK_BSDF_MEASUREMENT:
                last_error = requested_functions[path_idx].error_code =
                    IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE;
                continue;

            case IType::TK_ALIAS:
            case IType::TK_FUNCTION:
            case IType::TK_INCOMPLETE:
            case IType::TK_ERROR:
                MDL_ASSERT(!"unexpected expression type");
                last_error = requested_functions[path_idx].error_code =
                    IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE;
                continue;
            }

            // According to MDL Spec 1.6 13.3, the geometry fields are evaluated before all surface
            // fields and the normal evaluation happens last within the geometry fields.
            Distribution_function_builder::Eval_state eval_state;
            if (path_parts.size() > 0 && strcmp(path_parts[0], "geometry") == 0) {
                eval_state = Distribution_function_builder::ES_BEGIN_STATE;

                // remap "geometry.normal" to state::normal(), if it will be included
                if (include_geometry_normal &&
                    path_parts.size() > 1 &&
                    strcmp(path_parts[1], "normal") == 0)
                {
                    IType const *float3_type = mat_builder.m_type_factory.create_vector(
                        mat_builder.m_type_factory.create_float(), 3);

                    node = mat_builder.m_root_lambda->create_call(
                        "::state::normal()",
                        IDefinition::DS_INTRINSIC_STATE_NORMAL,
                        NULL,
                        0,
                        float3_type);
                }
            } else
                eval_state = Distribution_function_builder::ES_AFTER_GEOMETRY_NORMAL;

            mat_builder.collect_flags_and_used_nodes(node, eval_state, ++walk_id);

            requests.push_back(Request(path_idx, node, eval_state));
        }

        if (last_error != IDistribution_function::EC_NONE)
            return last_error;

        // handle "geometry.normal" if requested
        if (include_geometry_normal) {
            char const *normal_path[] = {"geometry", "normal"};
            DAG_node const *normal = get_dag_arg(mat_root_node, normal_path, root_lambda.get());

            bool handle_normal = false;

            // handle it, if geometry.normal is not state::normal()
            if (DAG_call const *normal_call = as<DAG_call>(normal)) {
                if (normal_call->get_semantic() != IDefinition::DS_INTRINSIC_STATE_NORMAL)
                    handle_normal = true;
            } else
                handle_normal = true;

            if (handle_normal) {
                mat_builder.register_special_lambda(
                    normal_path,
                    IDistribution_function::SK_MATERIAL_GEOMETRY_NORMAL, ++walk_id);
            }
        }

        // register all special lambdas required by the material
        Distribution_function_builder::Flags mat_flags = mat_builder.get_flags();
        if ((mat_flags & Distribution_function_builder::FL_CONTAINS_UNSUPPORTED_DF) != 0) {
            return IDistribution_function::EC_UNSUPPORTED_BSDF;
        }
        if ((mat_flags & Distribution_function_builder::FL_NEEDS_MATERIAL_IOR) != 0) {
            mat_builder.register_special_lambda(
                "ior", IDistribution_function::SK_MATERIAL_IOR, ++walk_id);
        }
        if ((mat_flags & Distribution_function_builder::FL_NEEDS_MATERIAL_THIN_WALLED) != 0) {
            mat_builder.register_special_lambda(
                "thin_walled", IDistribution_function::SK_MATERIAL_THIN_WALLED, ++walk_id);
        }
        if ((mat_flags & Distribution_function_builder::FL_NEEDS_MATERIAL_VOLUME_ABSORPTION) != 0) {
            char const *absorp_coeff_path[] = { "volume", "absorption_coefficient" };
            mat_builder.register_special_lambda(
                absorp_coeff_path,
                IDistribution_function::SK_MATERIAL_VOLUME_ABSORPTION, ++walk_id);
        }

        // first create all expression lambdas for special lambdas.
        // this makes sure that geometry normal and all its dependencies will be available
        mat_builder.create_special_lambdas();

        // create expression lambdas for multiply used expressions, after geometry.normal
        // has been calculated
        for (Request request : requests) {
            if (request.eval_state == Distribution_function_builder::ES_AFTER_GEOMETRY_NORMAL)
            {
                mat_builder.prepare_expr_lambda_calls(
                    request.node, request.eval_state);
            }
        }

        // construct the new DAG containing calls to expression lambdas
        for (Request request : requests) {
            const DAG_node *new_expr =
                mat_builder.transform_material_graph(request.node);
            mi::base::Handle<ILambda_function> expr_lambda(
                mat_builder.create_expr_lambda(
                    new_expr, request.eval_state));
            if (requested_functions[request.reqfunc_index].base_fname != NULL)
                expr_lambda->set_name(requested_functions[request.reqfunc_index].base_fname);

            size_t main_func_index = dist_func->add_main_function(expr_lambda.get());

            // collect DF handles used by this main function
            ++walk_id;
            mat_builder.m_handle_name_map.clear();
            mat_builder.collect_main_df_handles(new_expr, main_func_index, walk_id);
        }

#if 0
        mat_builder.dump_everything();
#endif

        return IDistribution_function::EC_NONE;
    }

    /// Constructor.
    Distribution_function_builder(
        IAllocator *alloc,
        Distribution_function &dist_func,
        DAG_node const *mat_root_node,
        IMDL *compiler,
        ICall_name_resolver const *resolver,
        bool calc_derivative_infos,
        bool allow_double_expr_lambdas)
    : m_alloc(alloc)
    , m_compiler(compiler, mi::base::DUP_INTERFACE)
    , m_dist_func(dist_func)
    , m_deriv_infos(calc_derivative_infos ? dist_func.get_writable_derivative_infos() : NULL)
    , m_mat_root_node(mat_root_node)
    , m_root_lambda(impl_cast<Lambda_function>(dist_func.get_root_lambda()))
    , m_type_factory(*m_root_lambda->get_type_factory())
    , m_resolver(resolver)
    , m_node_info_map(0, Node_info_map::hasher(), Node_info_map::key_equal(), alloc)
    , m_flags(FL_NONE)
    , m_allow_double_expr_lambdas(allow_double_expr_lambdas)
    , m_handle_name_map(0, Handle_name_map::hasher(), Handle_name_map::key_equal(), alloc)
    {
    }

    /// Returns the handle of an elemental DF or NULL if the call is not an elemental DF.
    char const *get_elemental_handle(DAG_call const *call)
    {
        if (!is_elemental_df_semantics(call->get_semantic()))
            return NULL;

        DAG_node const *handle_node = call->get_argument("handle");
        MDL_ASSERT(handle_node && is<DAG_constant>(handle_node) &&
            "Elemental DF must have a constant handle argument");
        if (handle_node == NULL || !is<DAG_constant>(handle_node))
            return NULL;

        DAG_constant const *handle_const = cast<DAG_constant>(handle_node);
        IValue const *handle_val = handle_const->get_value();
        IValue_string const *handle_str = as<IValue_string>(handle_val);
        MDL_ASSERT(handle_str != NULL && "DF handle must be string");
        if (handle_str == NULL)
            return NULL;

        return handle_str->get_value();
    }

    /// Collect the DF handles for a main function.
    void collect_main_df_handles(DAG_node const *expr, size_t main_func_index, unsigned walk_id)
    {
        Node_info &info = m_node_info_map[expr];
        if (info.already_visited(walk_id))
            return;
        info.mark_visited(walk_id);

        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                collect_main_df_handles(expr, main_func_index, walk_id);
                break;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                // stop at non-DFs
                if (!contains_df_type(call->get_type()))
                    break;

                if (char const *handle_name = get_elemental_handle(call)) {
                    Handle_name_map::const_iterator it = m_handle_name_map.find(handle_name);
                    if (it == m_handle_name_map.end()) {
                        // the handle is not known, yet -> register it for the main function
                        m_handle_name_map[handle_name] =
                            m_dist_func.add_main_func_df_handle(main_func_index, handle_name);
                    }
                }

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    collect_main_df_handles(arg, main_func_index, walk_id);
                }
                break;
            }
        }
    }

    /// Walk the material DAG to collect the flags and the used nodes (as in collect_used_nodes)
    /// and determine, whether the node is evaluation state dependent.
    /// If so, the function returns true.
    /// The eval_state is only relevant for the used nodes part, not for the flags.
    bool collect_flags_and_used_nodes(
        DAG_node const *expr, Eval_state eval_state, unsigned &walk_id)
    {
        Node_info &info = m_node_info_map[expr];

        // Stop when already visited in this walk
        if (info.already_visited(walk_id))
            return info.is_eval_state_dependent;

        // Stop if the node was already seen in at least two walks
        if (info.get_count(eval_state) > 0) {
            info.inc_count(eval_state, walk_id);
            return info.is_eval_state_dependent;
        }

        unsigned node_walk_id = walk_id;

        bool res = false;
        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                res = collect_flags_and_used_nodes(expr, eval_state, walk_id);
                break;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            // note: parameters cannot be evaluation state dependent. If state::normal()
            //    was used as a argument during material instantiation, the
            //    corresponding parameter would not be a parameter anymore.
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                IDefinition::Semantics sema = call->get_semantic();

                bool is_df_sema = is_df_semantics(sema);
                if (is_df_sema) {
                    if (needs_thin_walled(sema))
                        m_flags |= FL_NEEDS_MATERIAL_THIN_WALLED;

                    if (needs_ior(sema))
                        m_flags |= FL_NEEDS_MATERIAL_IOR;

                    if (char const *handle_name = get_elemental_handle(call)) {
                        Handle_name_map::const_iterator it = m_handle_name_map.find(handle_name);
                        if (it == m_handle_name_map.end()) {
                            // the handle is not known, yet -> register it
                            m_handle_name_map[handle_name] = m_dist_func.add_df_handle(handle_name);
                        }
                    }
                }

                res = is_eval_state_dependent_direct(call);

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);

                    // new walks start at all non-DFish arguments of DFs
                    if (is_df_sema && !contains_df_type(arg->get_type()))
                        ++walk_id;

                    res |= collect_flags_and_used_nodes(arg, eval_state, walk_id);
                }
            }
        }

        info.is_eval_state_dependent = res;

        // update counter now, that evaluation state dependence is known
        info.inc_count(eval_state, node_walk_id);

        return res;
    }

    /// Checks whether the type is a DF type or contains a DF type.
    static bool contains_df_type(IType const *type)
    {
        type = type->skip_type_alias();
        switch (type->get_kind()) {
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            return true;
        case IType::TK_ARRAY:
            return contains_df_type(as<IType_array>(type)->get_element_type());
        case IType::TK_STRUCT:
            {
                IType_compound const *comp_type = as<IType_compound>(type);
                for (int i = 0, n = comp_type->get_compound_size(); i < n; ++i) {
                    if (contains_df_type(comp_type->get_compound_type(i)))
                        return true;
                }
                return false;
            }
        default:
            return false;
        }
    }

    /// Checks whether an expression lambda may be created for the given DAG node.
    /// They shouldn't be created for matrices node and cannot be created for DF nodes,
    /// or any types containing DF types.
    bool may_create_expr_lambda(DAG_node const *expr)
    {
        IType const *type = expr->get_type();

        // don't allow expression lambdas returning matrices,
        // as matrices are float 16 vectors which need to be aligned to 64 byte
        // and they take up far too much memory space
        if (as<IType_matrix>(type) != NULL)
            return false;

        // no DF types or types containing DF types
        if (contains_df_type(type))
            return false;

        if (!m_allow_double_expr_lambdas && is<IType_double>(type))
            return false;

        return true;
    }

    /// Collect nodes which will be used for the distribution function.
    /// Counts the nodes to find nodes used by multiple paths.
    /// The provided node should have been generated by the builder.
    void collect_used_nodes(DAG_node const *expr, Eval_state eval_state, unsigned walk_id)
    {
        // Stop when already visited in this walk.
        // Otherwise stop, if the node was already seen in at least two walks
        Node_info &info = m_node_info_map[expr];
        if (info.already_visited(walk_id) || info.inc_count(eval_state, walk_id) > 1)
            return;

        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                collect_used_nodes(expr, eval_state, walk_id);
                return;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            return;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i)
                    collect_used_nodes(call->get_argument(i), eval_state, walk_id);
                return;
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
    }

    /// Prepare calls to expression lambda functions for nodes which are used
    /// multiple times. Do this in post-order to make sure, that all required
    /// expression lambda function calls used in sub-expressions already exist.
    ///
    /// \returns whether the given node is evaluation state dependent.
    void prepare_expr_lambda_calls(DAG_node const *expr, Eval_state eval_state)
    {
        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                prepare_expr_lambda_calls(expr, eval_state);
                return;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            return;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i)
                    prepare_expr_lambda_calls(call->get_argument(i), eval_state);

                Node_info &info = m_node_info_map[expr];

                // call result used multiple times and not processed, yet,
                // and an expression lambda may be created?
                if (info.get_count(eval_state) > 1 && info.get_node(eval_state) == NULL &&
                        may_create_expr_lambda(expr)) {
                    build_expr_lambda_call(expr, eval_state);
                }
                return;
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
        return;
    }

    /// Get the result node from the node cache.
    ///
    /// \param expr        The original node.
    /// \param eval_state  The evaluation state for which the node should be retrieved.
    DAG_node const *get_result_node(DAG_node const *expr, Eval_state eval_state)
    {
        return m_node_info_map[expr].get_node(eval_state);
    }

    /// Set the result node in the node cache.
    ///
    /// \param expr        The original node.
    /// \param res_node    The result node.
    /// \param eval_state  The evaluation state for which the node should be set.
    void set_result_node(DAG_node const *expr, DAG_node const *res_node, Eval_state eval_state)
    {
        m_node_info_map[expr].set_node(res_node, eval_state);
        if (m_deriv_infos && m_deriv_infos->should_calc_derivatives(expr))
            m_deriv_infos->mark_calc_derivatives(res_node);
    }

    /// Check if we have a ternary operator of BSDF's or EDF's.
    static bool is_ternary_on_df(
        IDefinition::Semantics sema,
        IType const            *ret_type)
    {
        if (sema != operator_to_semantic(IExpression::OK_TERNARY))
            return false;
        IType::Kind kind = ret_type->get_kind();
        return kind == IType::TK_BSDF || kind == IType::TK_HAIR_BSDF || kind == IType::TK_EDF;
    }

    /// Walk the material DAG, cloning the DF DAG nodes into the root lambda
    /// and creating expression lambdas from the non-DF DAG nodes.
    /// Also collects information about required material information.
    DAG_node const *transform_material_graph(DAG_node const *expr)
    {
        for (;;) {
            if (DAG_node const *cache_node = get_result_node(expr, ES_AFTER_GEOMETRY_NORMAL))
                return cache_node;

            switch (expr->get_kind()) {
            case DAG_node::EK_TEMPORARY:
                {
                    // should not happen, but we can handle it
                    DAG_temporary const *t = cast<DAG_temporary>(expr);
                    expr = t->get_expr();
                    continue;
                }
            case DAG_node::EK_CONSTANT:
                {
                    DAG_node const *res = m_root_lambda->import_expr(expr);
                    set_result_node(expr, res, ES_AFTER_GEOMETRY_NORMAL);
                    return res;
                }
            case DAG_node::EK_CALL:
            case DAG_node::EK_PARAMETER:
                {
                    DAG_node const *res;

                    IType const *ret_type = expr->get_type();
                    ret_type = m_type_factory.import(ret_type);

                    if (DAG_call const *call = as<DAG_call>(expr)) {
                        IType const *ret_type = call->get_type();

                        if (contains_df_type(ret_type)) {
                            // for df expressions continue copying into the root lambda,
                            // recursively handle all arguments and create a call

                            int n_args = call->get_argument_count();
                            Small_VLA<DAG_call::Call_argument, 8> args(m_alloc, n_args);

                            for (int i = 0; i < n_args; ++i) {
                                DAG_call::Call_argument &arg = args[i];
                                arg.arg        = transform_material_graph(call->get_argument(i));
                                arg.param_name = call->get_parameter_name(i);
                            }

                            res = m_root_lambda->create_call(
                                call->get_name(),
                                call->get_semantic(),
                                args.data(),
                                args.size(),
                                ret_type);

                            set_result_node(expr, res, ES_AFTER_GEOMETRY_NORMAL);
                            return res;
                        }
                    }

                    // build a new lambda function for the non-df expression, call it via
                    // a special intrinsic and put it into the node cache
                    res = build_expr_lambda_call(expr, ES_AFTER_GEOMETRY_NORMAL);
                    return res;
                }
            }
            MDL_ASSERT(!"Unsupported DAG node kind");
        }
    }

    /// Get the flags determined during walking the material.
    Flags get_flags() const { return m_flags; }

    /// Register a special expression lambda and collect information about the needed nodes.
    void register_special_lambda(
        Array_ref<char const *>              expr_path,
        IDistribution_function::Special_kind kind,
        unsigned                             walk_id)
    {
        DAG_node const *node = get_dag_arg(m_mat_root_node, expr_path, m_root_lambda.get());

        Eval_state eval_state = kind == IDistribution_function::SK_MATERIAL_GEOMETRY_NORMAL
            ? ES_BEGIN_STATE : ES_AFTER_GEOMETRY_NORMAL;
        collect_used_nodes(node, eval_state, walk_id);

        m_special_lambdas.push_back(Special_lambda_descr(kind, eval_state, node));
    }

    /// Create previously registered special expression lambdas.
    void create_special_lambdas()
    {
        for (size_t i = 0, n = m_special_lambdas.size(); i < n; ++i) {
            Special_lambda_descr &descr = m_special_lambdas[i];

            prepare_expr_lambda_calls(descr.node, descr.eval_state);

            mi::base::Handle<ILambda_function> lambda(
                create_expr_lambda(descr.node, descr.eval_state));

            m_dist_func.set_special_lambda_function(descr.kind, lambda.get());
        }
    }

    /// Dumps the whole DAG including the special lambdas to disk for debugging.
    void dump_everything()
    {
        size_t n_main_funcs = m_dist_func.get_main_function_count();
        size_t n_args = n_main_funcs + m_special_lambdas.size();
        Small_VLA<DAG_call::Call_argument, 8> args(m_alloc, n_args);
        for (size_t i = 0, n = m_dist_func.get_main_function_count(); i < n; ++i) {
            mi::base::Handle<ILambda_function> main_func(m_dist_func.get_main_function(i));
            args[i].param_name = "main_func";
            args[i].arg = main_func->get_body();
        }
        for (int i = 0, n = m_special_lambdas.size(); i < n; ++i) {
            args[i + n_main_funcs].param_name = "special_lambda";
            args[i + n_main_funcs].arg = m_special_lambdas[i].node;
        }
        DAG_node const *dump_root = m_root_lambda->create_call(
            "root", IDefinition::DS_HIDDEN_ANNOTATION, args.data(), n_args,
            m_root_lambda->get_body()->get_type());

        m_root_lambda->dump(dump_root, "mat-all-dumps");
    }

private:
    /// Determines whether the called function is depending on the evaluation state
    /// not considering the arguments.
    bool is_eval_state_dependent_direct(DAG_call const *call)
    {
        char const *signature = call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<IModule const> mod(m_resolver->get_owner_module(signature));
        if (!mod.is_valid_interface())
            return false;

        Module const *module = impl_cast<Module>(mod.get());

        IDefinition const *def = module->find_signature(signature, /*only_exported=*/false);
        if (def == NULL)
            return false;

        // skip presets
        def = skip_presets(def, mod);

        return def->get_property(IDefinition::DP_USES_NORMAL);
    }

    /// Build a new expression lambda function and call it via a special intrinsic.
    /// Also adds it to the node cache.
    DAG_node const *build_expr_lambda_call(DAG_node const *expr, Eval_state eval_state)
    {
        mi::base::Handle<ILambda_function> expr_lambda(
            create_expr_lambda(expr, eval_state));

        size_t lambda_id = m_dist_func.add_expr_lambda_function(expr_lambda.get());
        char lambda_name[21];
        snprintf(lambda_name, sizeof(lambda_name), "%u", unsigned(lambda_id));

        IType const *ret_type = expr->get_type();
        ret_type = m_type_factory.import(ret_type);

        DAG_node const *res = m_root_lambda->create_call(
            lambda_name,
            IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA,
            NULL,
            0,
            ret_type);

        set_result_node(expr, res, eval_state);
        return res;
    }

    /// Create a lambda function which can be used as an expression lambda function.
    /// Prepared expression lambda calls will be used while importing the given DAG.
    ///
    /// \param node        DAG node which will be imported as body.
    /// \param eval_state  The evaluation state used for importing the DAG node.
    mi::base::Handle<ILambda_function> create_expr_lambda(
        DAG_node const *node, Eval_state eval_state)
    {
        // TODO: What if there already exists a lambda function for this node in this state?

        mi::base::Handle<ILambda_function> lambda(
            m_compiler->create_lambda_function(
                ILambda_function::LEC_CORE));

        // add all material parameters from the main lambda to the lambda function
        for (size_t i = 0, n = m_root_lambda->get_parameter_count(); i < n; ++i) {

            size_t idx = lambda->add_parameter(
                m_root_lambda->get_parameter_type(i),
                m_root_lambda->get_parameter_name(i));

            /// map the i'th material parameter to this new parameter
            lambda->set_parameter_mapping(i, idx);
        }

        lambda->set_body(import_mat_expr(lambda.get(), node, eval_state));

        // for now, copy the resource table to every lambda
        mi::base::Handle<ILambda_function> root_lambda(m_dist_func.get_root_lambda());
        for (size_t i = 0, n = root_lambda->get_resource_entries_count(); i < n; ++i) {
            Resource_tag_tuple const *e = root_lambda->get_resource_entry(i);

            lambda->set_resource_tag(e->m_kind, e->m_url, e->m_tag);
        }
        return lambda;
    }

    /// Import an expression into a lambda function while using the node cache to
    /// take expression lambda calls into account.
    DAG_node const *import_mat_expr(
        ILambda_function *lambda, DAG_node const *expr, Eval_state eval_state)
    {
        if (DAG_node const *cache_node = get_result_node(expr, eval_state))
            return lambda->import_expr(cache_node);

        DAG_node const *res;
        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                res = import_mat_expr(lambda, expr, eval_state);
            }
            break;
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            res = lambda->import_expr(expr);
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                int n_args = call->get_argument_count();
                Small_VLA<DAG_call::Call_argument, 8> args(m_alloc, n_args);

                for (int i = 0; i < n_args; ++i) {
                    DAG_call::Call_argument &arg = args[i];
                    arg.arg        = import_mat_expr(lambda, call->get_argument(i), eval_state);
                    arg.param_name = call->get_parameter_name(i);
                }

                IType const *ret_type = call->get_type();
                ret_type = m_type_factory.import(ret_type);

                res = lambda->create_call(
                    call->get_name(), call->get_semantic(), args.data(), n_args, ret_type);
            }
            break;
        default:
            MDL_ASSERT(!"Unsupported DAG node kind");
            return NULL;
        }

        if (m_deriv_infos && m_deriv_infos->should_calc_derivatives(expr))
            m_deriv_infos->mark_calc_derivatives(res);

        return res;
    }

    /// Checks whether the given BSDF semantic needs access to the material.thin_walled field.
    bool needs_thin_walled(IDefinition::Semantics sema) {
        switch (sema)
        {
            case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
            case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
            case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
            case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
            case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
            case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
            case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
            case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
            case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
                return true;

            default:
                return false;
        }
    }

    /// Checks whether the given BSDF semantic needs access to the material.ior field.
    bool needs_ior(IDefinition::Semantics sema) {
        switch (sema)
        {
            case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
            case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
            case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
            case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
            case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
            case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
            case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
            case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
            case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
            case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
                return true;

            default:
                return false;
        }
    }

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The MDL compiler.
    mi::base::Handle<IMDL> m_compiler;

    /// The distribution function.
    Distribution_function &m_dist_func;

    /// The derivative infos, if calculation was requested.
    Derivative_infos *m_deriv_infos;

    /// The root DAG node of the material.
    DAG_node const *m_mat_root_node;

    /// The main lambda function of the distribution function, also used as owner for DAG nodes
    /// imported from special lambdas.
    mi::base::Handle<Lambda_function> m_root_lambda;

    /// The type factory of the root lambda.
    IType_factory &m_type_factory;

    /// The resolver for calls.
    ICall_name_resolver const *m_resolver;

    /// Helper struct collecting information about special lambda functions to be created.
    struct Special_lambda_descr {
        Special_lambda_descr(
            IDistribution_function::Special_kind kind,
            Eval_state eval_state,
            DAG_node const *node)
        : kind(kind)
        , eval_state(eval_state)
        , node(node)
        {}

        IDistribution_function::Special_kind kind;
        Eval_state eval_state;
        DAG_node const *node;
    };

    /// The list of special lambda descriptions.
    std::vector<Special_lambda_descr> m_special_lambdas;

    /// Helper struct collecting information about DAG nodes in different evaluation states.
    struct Node_info {
        // will be calculated in first phase (collect_flags)
        bool is_eval_state_dependent;

        // will be calculated in second phase (collect_used_nodes)
        unsigned begin_state_count;         // also used as any count, if not state dependent
        unsigned after_geometry_normal_count;

        // will be calculated in third phase (post visit in prepare_expr_lambda_calls)
        DAG_node const *begin_state_node;   // also used as any node, if not state dependent
        DAG_node const *after_geometry_normal_node;

        unsigned last_walk_id;              // used to differentiate between walks

        /// Constructor.
        Node_info()
            : is_eval_state_dependent(false)
            , begin_state_count(0)
            , after_geometry_normal_count(0)
            , begin_state_node(NULL)
            , after_geometry_normal_node(NULL)
            , last_walk_id(0)
        {}

        /// Return true, if this node has already been visited during the given walk.
        bool already_visited(unsigned walk_id) {
            return last_walk_id == walk_id;
        }

        /// Mark the node as visited in this walk.
        void mark_visited(unsigned walk_id) {
            last_walk_id = walk_id;
        }

        /// Increment the counter for the according evaluation state, if the node has not been
        /// visited in this walk, yet, mark as visited in this walk and return the new counter.
        unsigned inc_count(Eval_state eval_state, unsigned walk_id) {
            // only increment, if this node has not been seen in this walk, yet
            unsigned inc_val = last_walk_id == walk_id ? 0 : 1;
            last_walk_id = walk_id;

            // If not state dependent, begin_state_count is always used.
            if (!is_eval_state_dependent) {
                begin_state_count += inc_val;
                return begin_state_count;
            }

            switch (eval_state) {
            case ES_BEGIN_STATE:
                begin_state_count += inc_val;
                return begin_state_count;
            case ES_AFTER_GEOMETRY_NORMAL:
                after_geometry_normal_count += inc_val;
                return after_geometry_normal_count;
            }
            MDL_ASSERT("Unexpected evaluation state");
            return 0;
        }

        /// Get the counter for an evaluation state.
        unsigned get_count(Eval_state eval_state) const {
            // If not state dependent, begin_state_count is always used.
            if (!is_eval_state_dependent)
                return begin_state_count;

            switch (eval_state) {
            case ES_BEGIN_STATE: return begin_state_count;
            case ES_AFTER_GEOMETRY_NORMAL: return after_geometry_normal_count;
            }
            MDL_ASSERT("Unexpected evaluation state");
            return 0;
        }

        /// Set replacement node for an evaluation state.
        void set_node(DAG_node const *node, Eval_state eval_state) {
            // If not state dependent, begin_state_count is always used.
            if (!is_eval_state_dependent) {
                begin_state_node = node;
                return;
            }

            switch (eval_state) {
            case ES_BEGIN_STATE:
                begin_state_node = node;
                return;
            case ES_AFTER_GEOMETRY_NORMAL:
                after_geometry_normal_node = node;
                return;
            }
            MDL_ASSERT("Unexpected evaluation state");
            return;
        }

        /// Get replacement node for an evaluation state.
        DAG_node const *get_node(Eval_state eval_state) const {
            // If not state dependent, begin_state_count is always used.
            if (!is_eval_state_dependent) return begin_state_node;

            switch (eval_state) {
            case ES_BEGIN_STATE: return begin_state_node;
            case ES_AFTER_GEOMETRY_NORMAL: return after_geometry_normal_node;
            }
            MDL_ASSERT("Unexpected evaluation state");
            return NULL;
        }
    };

    typedef ptr_hash_map<DAG_node const, Node_info>::Type Node_info_map;

    /// Maps from DAG nodes created via the builder to an information structure.
    Node_info_map m_node_info_map;

    /// Collected flags.
    Flags m_flags;

    /// If true, expression lambdas may be created for double values.
    bool m_allow_double_expr_lambdas;

    typedef hash_map<
        char const *,
        size_t,
        cstring_hash,
        cstring_equal_to
    >::Type Handle_name_map;

    /// Maps from handle names to DF local handle IDs.
    Handle_name_map m_handle_name_map;
};


///
/// Helper class to dump a distribution function as a dot file.
///
class Distribution_function_dumper : public DAG_dumper {
    typedef DAG_dumper Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param out    an output stream, the dot file is written to
    Distribution_function_dumper(
        IAllocator     *alloc,
        IOutput_stream *out,
        Distribution_function const *dist_func);

    /// Dump the distribution function to the output stream.
    void dump();

    /// Get a name for a type.
    ///
    /// \param type  the type
    char const *get_type_name(IType const *type);

    /// Get the parameter name for the given index if any.
    ///
    /// \param index  the index of the parameter
    char const *get_parameter_name(int index) MDL_FINAL;

private:
    /// The allocator;
    IAllocator              *m_alloc;

    /// Currently processed distribution function.
    Distribution_function const *m_dist_func;
};

// Constructor.
Distribution_function_dumper::Distribution_function_dumper(
    IAllocator              *alloc,
    IOutput_stream          *out,
    Distribution_function const *dist_func)
    : Base(alloc, out)
    , m_alloc(alloc)
    , m_dist_func(dist_func)
{
}

// Get a name for a type.
char const *Distribution_function_dumper::get_type_name(IType const *type)
{
    type = type->skip_type_alias();

    switch (type->get_kind()) {
    case IType::TK_BOOL: return "bool";
    case IType::TK_INT: return "int";
    case IType::TK_ENUM: return "enum";
    case IType::TK_FLOAT:            return "float";
    case IType::TK_DOUBLE:           return "double";
    case IType::TK_STRING:           return "string";
    case IType::TK_COLOR:            return "color";
    case IType::TK_TEXTURE: return "texture";
    case IType::TK_LIGHT_PROFILE:    return "light_profile";
    case IType::TK_BSDF_MEASUREMENT: return "bsdf_measurement";
    case IType::TK_BSDF:             return "bsdf";
    case IType::TK_HAIR_BSDF:        return "hair_bsdf";
    case IType::TK_EDF:              return "edf";
    case IType::TK_VDF:              return "vdf";
    case IType::TK_VECTOR:
        {
            IType_vector const *tv = cast<IType_vector>(type);
            int size = tv->get_compound_size();
            IType::Kind elem_kind = tv->get_element_type()->get_kind();
            if (elem_kind == IType::TK_FLOAT) {
                if (size == 2) return "float2";
                if (size == 3) return "float3";
                if (size == 4) return "float4";
                return "floatN";
            }
            if (elem_kind == IType::TK_DOUBLE) {
                if (size == 2) return "double2";
                if (size == 3) return "double3";
                if (size == 4) return "double4";
                return "doubleN";
            }
            if (elem_kind == IType::TK_INT) {
                if (size == 2) return "int2";
                if (size == 3) return "int3";
                if (size == 4) return "int4";
                return "intN";
            }
            return "vector";
        }
    case IType::TK_MATRIX: return "matrix";
    case IType::TK_ARRAY: return "array";
    case IType::TK_STRUCT:
        {
            IType_struct const *ts = cast<IType_struct>(type);
            return ts->get_symbol()->get_name();
        }

    case IType::TK_ALIAS:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
    case IType::TK_FUNCTION:
        return "<unknown>";
    }
    return "<unexpected>";
}

// Dump the lambda expression DAG to the output stream.
void Distribution_function_dumper::dump()
{
    m_printer->print("digraph \"distribution_function\" {\n");

    for (size_t i = 0, n = m_dist_func->get_main_function_count(); i < n; ++i) {
        char main_name[30];
        snprintf(main_name, sizeof(main_name), "main_lambda_%u", (unsigned) i);

        mi::base::Handle<ILambda_function> main_lambda_handle(m_dist_func->get_main_function(i));
        Lambda_function const *main_lambda = impl_cast<Lambda_function>(main_lambda_handle.get());

        m_printer->print("  subgraph cluster_");
        m_printer->print(main_name);
        m_printer->print(" {\n"
            "    bgcolor = goldenrod1;\n"
            "    color = goldenrod;\n"
            "    node [style=filled];\n"
            "    label = \"DF\";\n");
        m_walker.walk_node(const_cast<DAG_node *>(main_lambda->get_body()), this);
        m_printer->print("  }\n");
    }

    for (size_t i = 0, n = m_dist_func->get_expr_lambda_count(); i < n; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(
            m_dist_func->get_expr_lambda(i));
        Lambda_function *lambda = impl_cast<Lambda_function>(expr_lambda.get());

        char expr_name[30];
        snprintf(expr_name, sizeof(expr_name), "expr_lambda_%u", (unsigned) i);

        m_printer->print("  subgraph cluster_");
        m_printer->print(expr_name);
        m_printer->print(" {\n"
            "    bgcolor = lightyellow;\n"
            "    color = lightyellow3;\n"
            "    node [style=filled];\n"
            "    label = \"");
        m_printer->print(expr_name);
        IType const *ret_type = lambda->get_return_type();
        if (ret_type != NULL) {
            m_printer->print(" : ");
            m_printer->print(get_type_name(ret_type));
        }
        m_printer->print("\";\n");
        m_walker.walk_node(const_cast<DAG_node *>(lambda->get_body()), this);
        m_printer->print("  }\n");
    }
    m_printer->print("}\n");
}

// Get the parameter name for the given index if any.
const char *Distribution_function_dumper::get_parameter_name(int index)
{
    mi::base::Handle<mi::mdl::ILambda_function> root_lambda(m_dist_func->get_root_lambda());
    return impl_cast<Lambda_function>(root_lambda.get())->get_parameter_name(index);
}

}  // anonymous

// Constructor.
Distribution_function::Distribution_function(
    IAllocator                         *alloc,
    MDL                                *compiler)
: Base(alloc)
, m_mdl(mi::base::make_handle_dup(compiler))
, m_root_lambda(compiler->create_lambda_function(Lambda_function::LEC_CORE))
, m_main_functions(alloc)
, m_expr_lambdas(alloc)
, m_deriv_infos_calculated(false)
, m_deriv_infos(alloc)
, m_df_handles(alloc)
, m_main_func_df_handles(alloc)
, m_arena(alloc)
, m_resource_tag_map(alloc)
{
    Lambda_function *root_lambda = impl_cast<Lambda_function>(m_root_lambda.get());

    // force always using render state
    root_lambda->set_uses_render_state(true);

    for (size_t i = 0, n = dimension_of(m_special_lambdas); i < n; ++i)
        m_special_lambdas[i] = ~0;
}

/// Initialize this distribution function object for the given material
/// with the given distribution function node. Any additionally required
/// expressions from the material will also be handled.
IDistribution_function::Error_code Distribution_function::initialize(
    DAG_node const            *material_constructor,
    Requested_function        *requested_functions,
    size_t                     num_functions,
    bool                       include_geometry_normal,
    bool                       calc_derivative_infos,
    bool                       allow_double_expr_lambdas,
    ICall_name_resolver const *name_resolver)
{
    m_deriv_infos_calculated = calc_derivative_infos;
    return Distribution_function_builder::build(
        this,
        get_allocator(),
        m_mdl.get(),
        name_resolver,
        material_constructor,
        requested_functions,
        num_functions,
        include_geometry_normal,
        calc_derivative_infos,
        allow_double_expr_lambdas);
}

// Get the root lambda function used to build nodes and manage parameters and resources.
ILambda_function *Distribution_function::get_root_lambda() const
{
    m_root_lambda.get()->retain();
    return m_root_lambda.get();
}

size_t Distribution_function::add_main_function(ILambda_function *lambda)
{
    m_main_functions.push_back(mi::base::make_handle_dup(lambda));
    m_main_func_df_handles.emplace_back(get_allocator());
    return m_main_functions.size() - 1;
}

ILambda_function *Distribution_function::get_main_function(size_t index) const
{
    if (index >= m_main_functions.size())
        return NULL;

    ILambda_function *lambda = m_main_functions[index].get();
    if (!lambda)
        return NULL;
    lambda->retain();
    return lambda;
}

size_t Distribution_function::get_main_function_count() const
{
    return m_main_functions.size();
}

// Add the given expression lambda function to the distribution function.
size_t Distribution_function::add_expr_lambda_function(ILambda_function *lambda)
{
    m_expr_lambdas.push_back(mi::base::make_handle_dup(lambda));
    return m_expr_lambdas.size() - 1;
}

// Get the expression lambda function for the given index.
ILambda_function *Distribution_function::get_expr_lambda(size_t index) const
{
    if (index >= m_expr_lambdas.size())
        return NULL;

    ILambda_function *lambda = m_expr_lambdas[index].get();
    if (!lambda)
        return NULL;
    lambda->retain();
    return lambda;
}

// Get the number of expression lambda functions.
size_t Distribution_function::get_expr_lambda_count() const
{
    return m_expr_lambdas.size();
}

// Set a special lambda function for getting certain material properties.
void Distribution_function::set_special_lambda_function(
    Special_kind kind,
    ILambda_function *lambda)
{
    MDL_ASSERT(kind < SK_NUM_KINDS);

    // add as expression lambda to use the same workflow for resource enumeration
    // and LLVM function construction. This will also increase the reference count.
    m_special_lambdas[kind] = add_expr_lambda_function(lambda);
}

// Get the expression lambda index for the given special lambda function kind.
size_t Distribution_function::get_special_lambda_function_index(Special_kind kind) const
{
    MDL_ASSERT(kind < SK_NUM_KINDS);
    return m_special_lambdas[kind];
}

/// Get the resource attribute map of this distribution function.
Resource_attr_map const &Distribution_function::get_resource_attribute_map() const {
    return impl_cast<Lambda_function>(m_root_lambda.get())->get_resource_attribute_map();
}

// Set a tag, version pair for a resource value that might be reachable from this function.
void Distribution_function::set_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url,
    int                            tag)
{
    int old_tag = find_resource_tag(res_kind, res_url);

    if (old_tag == 0) {
        add_resource_tag(res_kind, res_url, tag);
    } else {
        MDL_ASSERT(old_tag == tag && "Changing tag of a resource");
    }
}

// Find the resource tag of a resource.
int Distribution_function::find_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                    *res_url) const
{
    // linear search so far
    for (size_t i = 0, n = m_resource_tag_map.size(); i < n; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        // beware of NULL pointer
        if (e.m_kind == res_kind && (e.m_url == res_url || strcmp(e.m_url, res_url) == 0))
            return e.m_tag;
    }
    return 0;
}

// Add tag, version pair for a resource value that might be reachable from this function.
void Distribution_function::add_resource_tag(
    Resource_tag_tuple::Kind res_kind,
    char const               *res_url,
    int                      tag)
{
    res_url = res_url != NULL ? Arena_strdup(m_arena, res_url) : NULL;
    m_resource_tag_map.push_back(Resource_tag_tuple(res_kind, res_url, tag));
}

// Returns the number of distribution function handles referenced by this distribution function.
size_t Distribution_function::get_df_handle_count() const
{
    return m_df_handles.size();
}

// Returns a distribution function handle referenced by this distribution function.
char const *Distribution_function::get_df_handle(size_t index) const
{
    if (index >= m_df_handles.size())
        return NULL;
    return m_df_handles[index];
}

// Returns the number of distribution function handles referenced by this distribution function.
size_t Distribution_function::get_main_func_df_handle_count(size_t main_func_index) const
{
    if (main_func_index >= m_main_func_df_handles.size())
        return ~size_t(0);
    return m_main_func_df_handles[main_func_index].size();
}

// Returns a distribution function handle referenced by a given main function.
char const *Distribution_function::get_main_func_df_handle(
    size_t main_func_index, size_t index) const
{
    if (main_func_index >= m_main_func_df_handles.size() ||
        index >= m_main_func_df_handles[main_func_index].size())
    {
        return NULL;
    }
    return m_main_func_df_handles[main_func_index][index];
}

// Get the derivative information if they were requested during initialization.
Derivative_infos const *Distribution_function::get_derivative_infos() const
{
    if (!m_deriv_infos_calculated) return NULL;
    return &m_deriv_infos;
}

// Dump the distribution function to a .gv file with the given name.
void Distribution_function::dump(char const *name) const
{
    Allocator_builder builder(get_allocator());

    if (FILE *f = fopen(name, "w")) {
        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(get_allocator(), f, /*close_at_destroy=*/true));

        Distribution_function_dumper dumper(get_allocator(), out.get(), this);
        dumper.dump();
    }
}

} // mdl
} // mi
