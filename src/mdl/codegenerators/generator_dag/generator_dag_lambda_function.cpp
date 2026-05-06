/******************************************************************************
 * Copyright (c) 2013-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "generator_dag_ir_checker.h"

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
    /// \param lambda  the lambda function that owns the root expression
    /// \param root    the root expression
    void dump(
        Lambda_function const &lambda,
        DAG_node const        *root);

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
    IAllocator     *alloc,
    IOutput_stream *out)
: Base(alloc, out)
, m_lambda(NULL)
{
}

#if 0
// Dump the lambda expression DAG to the output stream.
void Lambda_dumper::dump(Lambda_function const &lambda)
{
    m_lambda = &lambda;

    set_dag_unit(&lambda.get_dag_unit());

    m_printer->print("digraph \"");
    char const *name = lambda.get_name();
    if (name == NULL || name[0] == '\0') {
        name = "lambda";
    }
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
void Lambda_dumper::dump(
    Lambda_function const &lambda,
    DAG_node const        *root)
{
    set_dag_unit(&lambda.get_dag_unit());

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
Lambda_function::Import_helper::Import_helper(
    DAG_unit &dest,
    DAG_unit const &src)
: m_node_cache(0, Node_cache::hasher(), Node_cache::key_equal(), dest.get_allocator())
, m_fname_tbl(0, Fname_tbl::hasher(), Fname_tbl::key_equal(), dest.get_allocator())
, m_translate(dest.get_allocator())
, m_has_dbg_info(false)
{
    if (!dest.has_dbg_info() || !src.has_dbg_info()) {
        // we do not need translations if either the destination does not except them OR
        // the source does not have them
        return;
    }

    unsigned n  = unsigned(dest.get_file_name_count());
    size_t   nn = src.get_file_name_count();

    if (n + nn == 0) {
        // debug info is enabled, but does not exists
        return;
    }

    // enable translation
    m_has_dbg_info = true;

    for (unsigned i = 0; i < n; ++i) {
        m_fname_tbl[dest.get_fname(i)] = i + 1;
    }

    m_translate.reserve(nn);

    for (size_t i = 0; i < nn; ++i) {
        char const *fname = src.get_fname(i);
        unsigned   id = 0;

        Fname_tbl::iterator it = m_fname_tbl.find(fname);
        if (it == m_fname_tbl.end()) {
            // new file found
            id = unsigned(dest.register_file_name(fname));
        } else {
            id = it->second;
        }
        m_translate.push_back(id);
    }
}

// Translate a src debug info into a destination debug info.
DAG_DbgInfo Lambda_function::Import_helper::import(DAG_DbgInfo src)
{
    if (m_has_dbg_info) {
        unsigned id = src.get_file_id();
        if (id == 0) {
            // either not available or special, does not need translation
            return src;
        }

        // translate: Note that the 0 is not stored in the translate table
        MDL_ASSERT(id <= m_translate.size() && "invalid debug info found");

        if (id <= m_translate.size()) {
            id = m_translate[id - 1];

            return DAG_DbgInfo(id, src.get_line(), src.get_column());
        }
    }
    return DAG_DbgInfo();
}

// Constructor.
Lambda_function::Lambda_function(
    IAllocator               *alloc,
    MDL                      *compiler,
    Lambda_execution_context context)
: Base(alloc)
, m_mdl(mi::base::make_handle_dup(compiler))
, m_dag_unit(compiler, /*enable_debug_info=*/true)
, m_node_factory(compiler, m_dag_unit, internal_space(context))
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
, m_uses_varying_state(false)
, m_has_dead_code(false)
, m_is_modified(false)
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

// Get the DAG_unit of this lambda function.
DAG_unit &Lambda_function::get_dag_unit()
{
    return m_dag_unit;
}

// Get the DAG_unit of this lambda function.
DAG_unit const &Lambda_function::get_dag_unit() const
{
    return m_dag_unit;
}

// Get the type factory of this builder.
Type_factory *Lambda_function::get_type_factory()
{
    return &m_dag_unit.get_type_factory();
}

// Get the value factory of this builder.
Value_factory *Lambda_function::get_value_factory()
{
    return &m_dag_unit.get_value_factory();
}

// Create a constant.
DAG_constant const *Lambda_function::create_constant(
    IValue const *value,
    DAG_DbgInfo  dbg_info)
{
    return m_node_factory.create_constant(value, dbg_info);
}

// Create a call.
DAG_node const *Lambda_function::create_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    int                           num_call_args,
    IType const                   *ret_type,
    DAG_DbgInfo                   dbg_info)
{
    if (is_varying_state_semantic(sema)) {
        m_uses_varying_state = true;
    }
    return m_node_factory.create_call(name, sema, call_args, num_call_args, ret_type, dbg_info);
}

// Create a parameter reference.
DAG_parameter const *Lambda_function::create_parameter(
    IType const *type,
    int         index,
    DAG_DbgInfo dbg_info)
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

    return m_node_factory.create_parameter(type, index, dbg_info);
}

// Enable common subexpression elimination.
bool Lambda_function::enable_cse(bool flag)
{
    return m_node_factory.enable_cse(flag);
}

// Enable optimization.
bool Lambda_function::enable_opt(bool flag)
{
    return m_node_factory.enable_opt(flag);
}

// Enable unsafe math optimizations.
bool Lambda_function::enable_unsafe_math_opt(bool flag)
{
    return m_node_factory.enable_unsafe_math_opt(flag);
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
DAG_node const *Lambda_function::import_expr(
    DAG_unit const &owner,
    DAG_node const *expr)
{
    MDL_ASSERT(owner.is_owner(expr) && "Wrong owner");

    Import_helper helper(m_dag_unit, owner);

    DAG_node const *imported_expr = do_import_expr(expr, helper);

    // now map node names
    DAG_unit::Node_name_map const &owner_name_map =
        owner.get_node_name_map();
    for (auto const &it : owner_name_map) {
        DAG_node const *owner_node = it.first;
        ISymbol const *node_name = it.second;

        auto nc_it = helper.find(owner_node);
        if (nc_it != helper.end()) {
            DAG_node const *imported_node = nc_it->second;
            m_dag_unit.set_node_name(imported_node, m_dag_unit.import_symbol(node_name));
        }
    }

    return imported_expr;
}

// Import (i.e. deep-copy) a DAG expression into this lambda function.
DAG_node const *Lambda_function::do_import_expr(
    DAG_node const *expr,
    Import_helper  &import_helper)
{
    for (;;) {
        Node_cache::iterator it = import_helper.find(expr);
        if (it != import_helper.end()) {
            return it->second;
        }

        switch (expr->get_kind()) {
        case DAG_node::EK_CONSTANT:
            {
                DAG_constant const *c = cast<DAG_constant>(expr);
                mi::mdl::IValue const *v = c->get_value();
                v = m_dag_unit.import(v);
                DAG_node const *res = create_constant(v, import_helper.import(c->get_dbg_info()));

                import_helper[expr] = res;
                return res;
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
                Small_VLA<DAG_call::Call_argument, 8> args(get_allocator(), n_args);

                for (int i = 0; i < n_args; ++i) {
                    DAG_call::Call_argument &arg = args[i];
                    arg.arg        = do_import_expr(call->get_argument(i), import_helper);
                    arg.param_name = call->get_parameter_name(i);
                }

                IType const *ret_type = call->get_type();
                ret_type = m_dag_unit.import(ret_type);

                DAG_node const *res = create_call(
                    call->get_name(),
                    call->get_semantic(),
                    args.data(),
                    args.size(),
                    ret_type,
                    import_helper.import(call->get_dbg_info()));

                import_helper[expr] = res;
                return res;
            }
        case DAG_node::EK_PARAMETER:
            {
                DAG_parameter const *p = cast<DAG_parameter>(expr);
                int index = p->get_index();
                IType const *type = p->get_type();
                type = m_dag_unit.import(type);

                DAG_node const *res = create_parameter(
                    type,
                    index,
                    import_helper.import(p->get_dbg_info()));

                import_helper[expr] = res;
                return res;
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
    for (size_t idx = 0; idx < n; ++idx) {
        if (m_roots[idx] == NULL) {
            return idx;
        }
    }
    return n;
}

// Store a DAG (root) expression and returns an index for it.
size_t Lambda_function::store_root_expr(DAG_node const *expr)
{
    Root_map::const_iterator it(m_root_map.find(expr));
    if (it != m_root_map.end()) {
        return it->second;
    }

    size_t idx = find_free_root_index();
    if (idx >= m_roots.size()) {
        m_roots.resize(idx + 1);
    }

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
    if (idx >= m_roots.size()) {
        return false;
    }
    DAG_node const *root = m_roots[idx];
    if (root == NULL) {
        return false;
    }

    m_root_map.erase(root);
    m_roots[idx] = NULL;

    m_has_dead_code = true;

    // Note: currently, we do NOT update the hash here, because we do not report it is "modified"

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

        if (expr != NULL) {
            expr = n_func->import_expr(m_dag_unit, expr);
        }

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
    if (idx < m_roots.size()) {
        return m_roots[idx];
    }
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
typedef ptr_hash_map<IValue const, ILambda_resource_enumerator::Texture_usage>::Type
    Texture_usage_map;
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
        if (f == NULL) {
            return NULL;
        }

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
            if (m_tex_usage) {
                m_tex_usage_map[v] |= m_tex_usage;
            }

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
        if (def == NULL) {
            return ref;
        }

        if (def->get_kind() == IDefinition::DK_PARAMETER) {
            if (is<IType_texture>(def->get_type()->skip_type_alias())) {
                ILambda_resource_enumerator::Texture_usage *usage = m_param_usage_map[def];
                MDL_ASSERT(usage != NULL);
                if (usage != NULL) {
                    *usage |= m_tex_usage;
                }
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

    /// Returns the argument usage for the given definition and calculates it, if necessary.
    ILambda_resource_enumerator::Texture_usage *get_or_calc_arg_usage(
        mi::base::Handle<IModule const> owner,
        IDefinition const               *def);

    /// Returns the potential texture usage of a value.
    ILambda_resource_enumerator::Texture_usage get_texture_usage(IValue const *value) const
    {
        Texture_usage_map::const_iterator it = m_tex_usage_map.find(value);
        if (it == m_tex_usage_map.end()) {
            return 0;
        }
        return it->second;
    }

    /// Ensure, we keep a reference to the module, to avoid IValues from disappearing.
    void keep_module_reference(mi::base::Handle<IModule const> module)
    {
        m_module_set.insert(module);
    }

private:
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
            if (m_tex_usage) {
                m_tex_usage_map[v] |= m_tex_usage;
            }

            if (m_found_resources.insert(v).second) {
                // inserted for first time?
                m_textures.push_back(v);
            }
            break;
        case IType::TK_LIGHT_PROFILE:
            if (m_found_resources.insert(v).second) {
                // inserted for first time?
                m_light_profiles.push_back(v);
            }
            break;
        case IType::TK_BSDF_MEASUREMENT:
            if (m_found_resources.insert(v).second) {
                // inserted for first time?
                m_bsdf_measurements.push_back(v);
            }
            break;
        default:
            break;
        }
    }

    /// Post-visit a Call.
    ///
    /// \param call  the call that is visited
    void visit_call(DAG_call const *call);

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
        i < n;
        ++i)
    {
        m_arg_usages_by_sema[arg_usage_init_table[i].sema] = arg_usage_init_table[i].arg_usage;
    }
}

void Resource_collector::collect(DAG_node const *node)
{
    // note: we don't check, whether the node was already visited, because the same node
    //       may have to be visited with different texture usages.
    //       As this is a DAG, we won't cause an endless iteration, though

    if (!m_visited.insert(node).second) {
        // already in the visited set
        return;
    }

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
        IValue_texture::Bsdf_data_kind bdk_multiscatter = IValue_texture::BDK_NONE;
        IValue_texture::Bsdf_data_kind bdk_general = IValue_texture::BDK_NONE;
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
            bdk_multiscatter = IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
            bdk_multiscatter = IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
            bdk_multiscatter = IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
            bdk_multiscatter = IValue_texture::BDK_GGX_SMITH_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
            bdk_multiscatter = IValue_texture::BDK_BECKMANN_VC_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            bdk_multiscatter = IValue_texture::BDK_GGX_VC_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
            bdk_multiscatter = IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
            bdk_multiscatter = IValue_texture::BDK_SHEEN_MULTISCATTER;
            break;
        case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:
            bdk_multiscatter = IValue_texture::BDK_MICROFLAKE_SHEEN_MULTISCATTER;
            bdk_general = IValue_texture::BDK_MICROFLAKE_SHEEN_GENERAL;
            break;
        default:
            break;
        }

        if (bdk_multiscatter != IValue_texture::BDK_NONE) {
            // check whether multiscatter_tint is a constant zero color
            DAG_node const *multiscatter_tint = call->get_argument("multiscatter_tint");
            if (multiscatter_tint != NULL &&
                    multiscatter_tint->get_kind() == DAG_node::EK_CONSTANT) {
                IValue_rgb_color const *val = as<IValue_rgb_color>(
                        cast<DAG_constant>(multiscatter_tint)->get_value());
                if (val != NULL && val->is_zero()) {
                    // no need to use multiscatter data textures for zero multiscatter tint
                    bdk_multiscatter = IValue_texture::BDK_NONE;
                }
            }
        }

        // ugly: we inject a bsdf_data textures value into the lambda function here
        if (bdk_general != IValue_texture::BDK_NONE) {
            Lambda_function *lambda = const_cast<Lambda_function *>(&m_lambda_func);
            IValue_factory *vf = lambda->get_value_factory();
            IValue const *v = vf->create_bsdf_data_texture(
                bdk_general,
                /*tag_value=*/ 0,
                /*tag_version=*/ 0);

            // libbsdf uses tex::lookup_float3_2d()
            m_tex_usage_map[v] |= ILambda_resource_enumerator::TU_LOOKUP_FLOAT3;
            if (m_found_resources.insert(v).second) {
                // inserted for first time?
                m_textures.push_back(v);
            }
        }
        if (bdk_multiscatter != IValue_texture::BDK_NONE) {
            Lambda_function *lambda = const_cast<Lambda_function *>(&m_lambda_func);
            IValue_factory *vf = lambda->get_value_factory();
            IValue const *v = vf->create_bsdf_data_texture(
                bdk_multiscatter,
                /*tag_value=*/ 0,
                /*tag_version=*/ 0);

            // libbsdf uses tex::lookup_float3_3d()
            m_tex_usage_map[v] |= ILambda_resource_enumerator::TU_LOOKUP_FLOAT3;
            if (m_found_resources.insert(v).second) {
                // inserted for first time?
                m_textures.push_back(v);
            }
        }

        Arg_usages_by_semantics::const_iterator it = m_arg_usages_by_sema.find(sema);
        if (it != m_arg_usages_by_sema.end()) {
            arg_usages = it->second;
        }
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
    mi::base::Handle<IModule const> owner,
    IDefinition const               *def)
{
    if (def == NULL) {
        return NULL;
    }

    IDefinition::Semantics sema = def->get_semantics();
    if (sema == IDefinition::DS_UNKNOWN) {
        Arg_usages_by_def::const_iterator it = m_arg_usages_by_def.find(def);
        if (it != m_arg_usages_by_def.end()) {
            return it->second;
        }

        if (!is<IDeclaration_function>(def->get_declaration())) {
            return NULL;
        }

        ILambda_resource_enumerator::Texture_usage *arg_usages =
            m_ast_collector.process_function(owner, def);
        m_arg_usages_by_def[def] = arg_usages;
        return arg_usages;
    }

    Arg_usages_by_semantics::const_iterator it = m_arg_usages_by_sema.find(sema);
    if (it != m_arg_usages_by_sema.end()) {
        return it->second;
    }
    return NULL;
}

}   // anonymous

/// Enumerate all used texture resources of this lambda function.
void Lambda_function::enumerate_resources(
    ICall_name_resolver const   &resolver,
    ILambda_resource_enumerator &enumerator,
    DAG_node const              *root) const
{
    DAG_ir_walker      walker(get_allocator());
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
    char const                     *res_sel,
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
        case IValue_texture::BDK_MICROFLAKE_SHEEN_GENERAL:
            kind = Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL;
            break;
        case IValue_texture::BDK_MICROFLAKE_SHEEN_MULTISCATTER:
            kind = Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER;
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

    res_url = res_url != NULL ? Arena_strdup(m_dag_unit.get_arena(), res_url) : NULL;
    res_sel = res_sel != NULL ? Arena_strdup(m_dag_unit.get_arena(), res_sel) : NULL;
    Resource_tag_tuple key(kind, res_url, res_sel, res_tag);

    m_resource_attr_map[key] = e;
    m_hash_is_valid = false;
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

    res_url = res_url != NULL ? Arena_strdup(m_dag_unit.get_arena(), res_url) : NULL;
    Resource_tag_tuple key(kind, res_url, /*selector=*/NULL, res_tag);

    m_resource_attr_map[key] = e;
    m_hash_is_valid = false;
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

    res_url = res_url != NULL ? Arena_strdup(m_dag_unit.get_arena(), res_url) : NULL;
    Resource_tag_tuple key(kind, res_url, /*selector=*/NULL, res_tag);

    m_resource_attr_map[key] = e;
    m_hash_is_valid = false;
}

// Analyze a lambda function.
bool Lambda_function::analyze(
    size_t                    proj,
    ICall_name_resolver const *name_resolver,
    Analysis_result           &result) const
{
    if (m_body_expr == NULL && proj >= m_roots.size()) {
        return false;
    }

    result.tangent_spaces           = 0;
    result.texture_spaces           = 0;
    result.uses_state_normal        = 0;
    result.uses_state_rc_normal     = 0;
    result.uses_texresult_lookup    = 0;
    result.uses_state_position      = 0;

    DAG_node const *root = !m_roots.empty() ? m_roots[proj] : m_body_expr;

    MDL_ASSERT(root);

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
        DAG_node_factory_impl     &node_factory,
        ICall_name_resolver const &name_resolver)
        : m_alloc(alloc)
        , m_node_factory(node_factory)
        , m_name_resolver(name_resolver)
        , m_dag_mangler(alloc)
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
                if (it != m_optimized_nodes.end()) {
                    return it->second;
                }

                DAG_call const *call = cast<DAG_call>(node);

                bool changed = false;
                int n_args = call->get_argument_count();
                VLA<DAG_call::Call_argument> args(m_alloc, n_args);
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    args[i].arg = optimize(arg);
                    if (args[i].arg != arg) {
                        changed = true;
                    }
                    args[i].param_name = call->get_parameter_name(i);
                }

                // try to inline user defined functions
                if (call->get_semantic() == IDefinition::DS_UNKNOWN) {
                    mi::base::Handle<Module const> mod(
                        impl_cast<Module>(m_name_resolver.get_owner_module(call->get_name())));
                    if (mod.is_valid_interface()) {
                        Module const *module = mod.get();
                        IDefinition const *def = module->find_signature(
                            call->get_name(),
                            /*only_exported=*/ false);
                        if (def != NULL) {
                            Module_scope module_scope(m_dag_builder, mod.get());

                            mi::base::Handle<IGenerated_code_dag const> owner_dag(
                                m_name_resolver.get_owner_dag(call->get_name()));

                            DAG_node const *res = m_dag_builder.try_inline(
                                call, owner_dag.get(), def, args.data(), n_args);
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
                    call->get_name(),
                    call->get_semantic(),
                    args.data(),
                    args.size(),
                    call->get_type(),
                    call->get_dbg_info());
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
        m_node_factory,
        *name_resolver);

    DAG_ir_checker checker(get_allocator(), name_resolver, /*allow_distiller_marker=*/false);

    checker.check_lambda(this);

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
    if (reset) {
        m_is_modified = false;
    }
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
                if (def->get_property(IDefinition::DP_USES_OBJECT_ID)) {
                    m_uses_object_id = true;
                }
                if (def->get_property(IDefinition::DP_USES_TRANSFORM)) {
                    m_uses_transform = true;
                }
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

    /// True if state::transform*() may be used.
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
    DAG_ir_walker        walker(get_allocator());
    Uniform_state_usage  visitor(*name_resolver);

    walker.walk_node(const_cast<DAG_node *>(expr), &visitor);

    if (visitor.uses_object_id()) {
        Value_factory    &vf       = m_dag_unit.get_value_factory();
        IValue_int const *v        = vf.create_int(object_id);
        DAG_node const   *c        = create_constant(v, DAG_DbgInfo::generated);
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
            res_type,
            DAG_DbgInfo::generated);
    }

    if (visitor.uses_transform()) {
        Value_factory &vf = m_dag_unit.get_value_factory();
        Type_factory  &tf = m_dag_unit.get_type_factory();

        IValue const *v_w2o[4];
        IValue const *v_o2w[4];

        // Note: We create the matrix row-major here to match the input from the iray/irt core.
        IType_float const  *float_type = tf.create_float();
        IType_vector const *f4_type    = tf.create_vector(float_type, 4);
        IType_matrix const *m_type     = tf.create_matrix(f4_type, 4);
        for (unsigned i = 0; i < 4; ++i) {
            Float4_struct const &w2o = world_to_object[i];

            IValue const *t_w2o[4] = {
                vf.create_float(w2o.x),
                vf.create_float(w2o.y),
                vf.create_float(w2o.z),
                vf.create_float(w2o.w)
            };

            v_w2o[i] = vf.create_vector(f4_type, t_w2o, 4);

            Float4_struct const &o2w = object_to_world[i];

            IValue const *t_o2w[4] = {
                vf.create_float(o2w.x),
                vf.create_float(o2w.y),
                vf.create_float(o2w.z),
                vf.create_float(o2w.w)
            };

            v_o2w[i] = vf.create_vector(f4_type, t_o2w, 4);
        }

        IType const         *res_type = expr->get_type();

        IValue_matrix const *m_w2o = vf.create_matrix(m_type, v_w2o, 4);
        DAG_node const      *c_w2o = create_constant(m_w2o, DAG_DbgInfo::generated);

        IValue_matrix const *m_o2w = vf.create_matrix(m_type, v_o2w, 4);
        DAG_node const      *c_o2w = create_constant(m_o2w, DAG_DbgInfo::generated);

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
            res_type,
            DAG_DbgInfo::generated);
    }

    return expr;
}

// Get a "serial version" number of this lambda function.
unsigned Lambda_function::get_serial_number() const
{
    if (!m_serial_is_valid) {
        m_serial_number   = g_next_serial++;

        // avoid serial number 0, it is used as a sentinel
        if (m_serial_number == 0u) {
            m_serial_number = g_next_serial++;
        }

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

/// Helper class to analyze the state usage on an AST.
class State_usage_ast_analysis : public Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param alloc    the allocator
    /// \param owner    the owner module of the analyzed function
    /// \param result   the analysis result
    /// \param args     top level constant arguments of the call (might be NULL)
    State_usage_ast_analysis(
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
                m_result.uses_state_position   = 1;
            }
            break;

        case IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE:
            // tangent space is constructed from state::normal(), state::texture_[u|v](), see
            // MDL spec
            m_result.uses_state_normal = 1;
            [[fallthrough]];
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
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

        case IDefinition::DS_INTRINSIC_STATE_POSITION:
            m_result.uses_state_position = 1;
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
    State_usage_ast_analysis(State_usage_ast_analysis &parent, IModule const *owner)
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
        if (m_function_depth != 1) {
            return NULL;
        }

        if (index >= m_top_level_args.size()) {
            return NULL;
        }

        return m_top_level_args[index];
    }

    /// Analyze the coordinate space expression.
    void analyze_space(
        IExpression const *expr,
        unsigned          &bitmap)
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

        State_usage_ast_analysis analysis(*this, owner.get());
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

/// Helper class to analyze the state usage on a DAG.
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

        case IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE:
            // tangent space is constructed from state::normal(), state::texture_[u|v](), see
            // MDL spec
            m_result.uses_state_normal = 1;
            [[fallthrough]];
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
        case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
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

        case IDefinition::DS_INTRINSIC_STATE_POSITION:
            m_result.uses_state_position = 1;
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
        if (!mod.is_valid_interface()) {
            return false;
        }
        mi::mdl::Module const *owner = impl_cast<mi::mdl::Module>(mod.get());

        IDefinition const *def = owner->find_signature(signature, /*only_exported=*/false);
        if (def == NULL) {
            return false;
        }

        IDeclaration const *decl = def->get_declaration();
        if (decl == NULL) {
            return false;
        }

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
    /// \param owner       the owner module of the function to analyze
    /// \param func        the declaration of the function
    /// \param const_args  constant arguments of the function call (NULL if non-const)
    ///
    /// \return true on success, false if analysis failed
    bool analyze_function_ast(
        IModule const                   *owner,
        IDeclaration_function const     *func,
        Array_ref<IValue const *> const &const_args)
    {
        State_usage_ast_analysis analysis(m_alloc, owner, m_result, const_args);

        analysis.visit(func);

        return !analysis.found_error();
    }

    /// Analyze a space.
    ///
    /// \param node    the space
    /// \param bitmap  the result bitmap
    static void analyze_space(
        DAG_node const *node,
        unsigned       &bitmap)
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

// Analyze a DAG expression.
bool Lambda_function::analyze(
    DAG_node const            *expr,
    ICall_name_resolver const *resolver,
    Analysis_result           &result) const
{
    DAG_ir_walker        walker(get_allocator());
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

        if (a_index != b_index) {
            return a_index < b_index;
        }

        Resource_tag_tuple const &a_t = a.first;
        Resource_tag_tuple const &b_t = b.first;

        if (a_t.m_kind != b_t.m_kind) {
            return a_t.m_kind < b_t.m_kind;
        }
        if (a_t.m_tag != b_t.m_tag) {
            return a_t.m_tag < b_t.m_tag;
        }
        if (a_t.m_url == NULL) {
            return b_t.m_url != NULL;
        }
        if (b_t.m_url == NULL) {
            return false;
        }
        return strcmp(a_t.m_url, b_t.m_url) < 0;
    }
};

}  // anonymous

// Update the hash value.
void Lambda_function::update_hash() const
{
    MD5_hasher md5_hasher;
    Dag_hasher dag_hasher(get_allocator(), md5_hasher);

    for (size_t i = 0, n = get_parameter_count(); i < n; ++i) {
        char const  *name = get_parameter_name(i);
        IType const *tp = get_parameter_type(i);

        dag_hasher.hash_parameter(name, tp);
    }

    if (!m_roots.empty()) {
        for (size_t i = 0, n = m_roots.size(); i < n; ++i) {
            DAG_node const *root = m_roots[i];
            if (root == NULL) {
                md5_hasher.update(0);  // update hash to be able to differentiate different orders
            } else {
                dag_hasher.hash_dag(root);
            }
        }
    } else {
        dag_hasher.hash_dag(m_body_expr);
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
        // Note that the cost cast is safe: there is only ONE immutable bool type
        return const_cast<Type_factory &>(m_dag_unit.get_type_factory()).create_bool();
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
    if (i < m_params.size()) {
        return m_params[i].m_type;
    }
    return NULL;
}

// Return the name of the i'th parameter.
char const *Lambda_function::get_parameter_name(size_t i) const
{
    if (i < m_params.size()) {
        return m_params[i].m_name;
    }
    return NULL;
}

// Add a new "captured" parameter.
size_t Lambda_function::add_parameter(
    mi::mdl::IType const *type,
    char const           *name)
{
    type = m_dag_unit.import(type);
    name = Arena_strdup(m_dag_unit.get_arena(), name);

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

    DAG_rebuilder deriv_builder(
        get_allocator(), *this, &m_deriv_infos, /*enable_spectral_conversions=*/ false);

    if (!m_roots.empty()) {
        for (size_t i = 0, n = m_roots.size(); i < n; ++i) {
            m_roots[i] = deriv_builder.rebuild(m_roots[i], /*want_derivatives=*/ false);
        }
    } else {
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

// Set a tag for a resource value that might be reachable from this lambda function.
void Lambda_function::set_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                    *res_url,
    char const                    *res_sel,
    int                           tag)
{
    int old_tag = find_resource_tag(res_kind, res_url, res_sel);

    if (old_tag == 0) {
        add_resource_tag(res_kind, res_url, res_sel, tag);
    } else {
        MDL_ASSERT(old_tag == tag && "Changing tag of a resource");
    }
}

// Remap a resource value according to the resource map.
int Lambda_function::get_resource_tag(IValue_resource const *r) const
{
    char const *selector = "";

    if (IValue_texture const *tex = as<IValue_texture>(r)) {
        selector = tex->get_selector();
    }

    int tag = find_resource_tag(kind_from_value(r), r->get_string_value(), selector);
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
    if (index < m_resource_tag_map.size()) {
        return &m_resource_tag_map[index];
    }
    return NULL;
}

// Find the resource tag of a resource.
int Lambda_function::find_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url,
    char const                     *res_sel) const
{
    // beware of NULL pointer
    if (res_url == nullptr) {
        res_url = "";
    }
    if (res_sel == nullptr) {
        res_sel = "";
    }

    // linear search so far
    for (size_t i = 0, n = m_resource_tag_map.size(); i < n; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        if (e.m_kind == res_kind &&
            (e.m_url      == res_url || strcmp(e.m_url,      res_url) == 0) &&
            (e.m_selector == res_sel || strcmp(e.m_selector, res_sel) == 0)) {
            return e.m_tag;
        }
    }
    return 0;
}

// Add a tag for a resource value that might be reachable from this function.
void Lambda_function::add_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url,
    char const                     *res_sel,
    int                            tag)
{
    res_url = res_url != NULL ? Arena_strdup(m_dag_unit.get_arena(), res_url) : NULL;
    m_resource_tag_map.push_back(Resource_tag_tuple(res_kind, res_url, res_sel, tag));
}

// Get the derivative information if they have been initialized.
Derivative_infos const *Lambda_function::get_derivative_infos() const
{
    if (!m_deriv_infos_calculated) {
        return NULL;
    }
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

                if (is_varying_state_semantic(sema)) {
                    return true;
                }

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

                    if (may_use_varying_state(resolver, arg)) {
                        return true;
                    }
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
    m_dag_unit.serialize_factories(dag_serializer);

    // The jitted code singleton will be set on deserialization.
    // m_jitted_code;

    dag_serializer.write_cstring(m_name.c_str());

    // serialize the factory
    DAG_serializer::Dag_vector exprs(alloc);
    if (m_body_expr != NULL) {
        exprs.push_back(m_body_expr);
    }

    DAG_serializer::Dag_vector const *roots[] = {
        &m_roots,
        &exprs
    };

    dag_serializer.write_dags(roots, dimension_of(roots));

    m_dag_unit.serialize_name_map(dag_serializer);

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
        it != end;
        ++it)
    {
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
        dag_serializer.write_encoded(k.m_selector);
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
        case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL:
        case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER:
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
    dag_serializer.write_bool(m_uses_varying_state);
    dag_serializer.write_bool(m_has_dead_code);

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
    MDL_binary_deserializer bin_deserializer(alloc, ds, *mdl);
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

    res->m_dag_unit.deserialize_factories(dag_deserializer);

    // Already set during creation.
    // m_jitted_code

    res->m_name = dag_deserializer.read_cstring();

    // deserialize the node factory m_node_factory by deserializing all reachable DAGs
    dag_deserializer.read_dags(res->m_node_factory);

    res->m_dag_unit.deserialize_name_map(dag_deserializer);

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

    Type_factory &tf = res->m_dag_unit.get_type_factory();

    // deserialize parameter map
    size_t n_params = dag_deserializer.read_encoded_tag();
    for (size_t i = 0; i < n_params; ++i) {
        IType const *type = dag_deserializer.read_type(tf);
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

        k.m_kind        = dag_deserializer.read_encoded<Resource_tag_tuple::Kind>();
        string url      = dag_deserializer.read_encoded<string>();
        k.m_url         = Arena_strdup(res->m_dag_unit.get_arena(), url.c_str());
        string selector = dag_deserializer.read_encoded<string>();
        k.m_selector    = Arena_strdup(res->m_dag_unit.get_arena(), selector.c_str());
        k.m_tag         = dag_deserializer.read_db_tag();

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
        case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL:
        case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER:
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
    res->m_uses_varying_state      = dag_deserializer.read_bool();
    res->m_has_dead_code           = dag_deserializer.read_bool();

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
                    IDefinition::Semantics sema = call->get_semantic();
                    if (sema == IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS ||
                        sema == IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID)
                    {
                        // if it is set on one, it works for all
                        return true;
                    }
                }
            }
        }
    } else if (DAG_node const *root = get_body()) {
        if (DAG_call const *call = as<DAG_call>(root)) {
            IDefinition::Semantics sema = call->get_semantic();
            return
                sema == IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS ||
                sema == IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID;
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

        dumper.dump(*this, expr);
    }
}

namespace {

/// A helper class to follow DAG path from a given root node.
class Dag_path_follower {
public:
    /// Constructor.
    ///
    /// \param alloc        the current allocator
    /// \param path         the access path
    /// \param dag_builder  the current DAG builder interface for building constants if necessary
    /// \param resolver     the entity resolver
    Dag_path_follower(
        IAllocator                    *alloc,
        Array_ref<char const *> const &path,
        IDag_builder                  *dag_builder,
        ICall_name_resolver const     *resolver = NULL)
    : m_alloc(alloc)
    , m_call_stack(Call_stack::container_type(alloc))
    , m_path(path)
    , m_dag_builder(dag_builder)
    , m_resolver(resolver)
    , m_curr_call(NULL)
    , m_depth(0)
    {
    }

    /// Get a DAG node along the given path looking through material calls.
    ///
    /// \param node         the root node
    DAG_node const *get_dag_arg(
        DAG_node const *node)
    {
        DAG_node const *res = follow_path(node);

        if (res != NULL && m_curr_call != NULL) {
            // we are returning an expression from inside a material call, to use it, it must be
            // cloned to remove parameters and temporaries
            res = clone(res);
        }
        return res;
    }

private:
    /// Get the value for a given material (name) from a code DAG.
    ///
    /// \param code           the code DAG
    /// \param material_name  the name of the requested material
    static DAG_node const *get_material_value(
        IGenerated_code_dag const *code,
        char const                *material_name)
    {
        for (size_t i = 0, n = code->get_material_count(); i < n; ++i) {
            char const *name = code->get_material_name(i);

            if (strcmp(name, material_name) == 0) {
                return code->get_material_value(i);
            }
        }
        return NULL;
    }

    /// Close debug info.
    DAG_DbgInfo clone_dbg_info(DAG_DbgInfo dbg_info)
    {
        // FIXME: NYI
        return DAG_DbgInfo();
    }

    /// Clone a DAG into the m_dag_builder owner, removing any parameters and temporaries.
    DAG_node const *clone(DAG_node const *node)
    {
        for (;;) {
            switch (node->get_kind()) {
            case DAG_node::EK_CONSTANT:
                {
                    DAG_constant const *c = cast<DAG_constant>(node);
                    IValue const *v = c->get_value();

                    IValue_factory *vf = m_dag_builder->get_value_factory();
                    v = vf->import(v);
                    return m_dag_builder->create_constant(v, clone_dbg_info(c->get_dbg_info()));
                }
            case DAG_node::EK_TEMPORARY:
                {
                    node = cast<DAG_temporary>(node)->get_expr();
                    continue;
                }
            case DAG_node::EK_CALL:
                {
                    DAG_call const *call  = cast<DAG_call>(node);
                    size_t         n_args = call->get_argument_count();

                    Small_VLA<DAG_call::Call_argument, 8> args(m_alloc, n_args);

                    // clone all arguments
                    for (size_t i = 0; i < n_args; ++i) {
                        DAG_node const *arg = call->get_argument(i);

                        args[i].arg        = clone(arg);
                        args[i].param_name = call->get_parameter_name(i);
                    }

                    // clone the call itself
                    IType const   *ret_type = call->get_type();
                    IType_factory *tf       = m_dag_builder->get_type_factory();

                    return m_dag_builder->create_call(
                        call->get_name(),
                        call->get_semantic(),
                        args.data(),
                        args.size(),
                        tf->import(ret_type),
                        clone_dbg_info(call->get_dbg_info()));
                }
            case DAG_node::EK_PARAMETER:
                {
                    DAG_parameter const *param = cast<DAG_parameter>(node);

                    if (m_call_stack.empty()) {
                        // inside the outermost material, just clone the parameter
                        IType const   *p_type = param->get_type();
                        IType_factory *tf     = m_dag_builder->get_type_factory();

                        return m_dag_builder->create_parameter(
                            tf->import(p_type),
                            param->get_index(),
                            clone_dbg_info(param->get_dbg_info()));
                    }

                    // we are inside a material call, but leave it now through its argument
                    DAG_call const *curr_call = m_curr_call;

                    MDL_ASSERT(!m_call_stack.empty());
                    m_call_stack.pop();

                    if (m_call_stack.empty()) {
                        m_curr_call = NULL;
                    } else {
                        m_curr_call = m_call_stack.top();
                    }

                    node = curr_call->get_argument(param->get_index());

                    node = clone(node);

                    // we are back in the current callee
                    m_call_stack.push(curr_call);
                    m_curr_call = curr_call;

                    return node;
                }
            }
            MDL_ASSERT(!"unknown DAG node kind");
            return NULL;
        }
    }

    /// Get a DAG node along the given path looking through material calls.
    ///
    /// \param node         the root node
    DAG_node const *follow_path(
        DAG_node const *node)
    {
        for (;;) {
            if (node == NULL || m_depth >= m_path.size()) {
                // either we miss, OR we found the node
                return node;
            }

            switch (node->get_kind()) {
            case DAG_node::EK_CONSTANT:
                {
                    DAG_constant const *c = cast<DAG_constant>(node);
                    if (IValue_compound const *vc = as<IValue_compound>(c->get_value())) {
                        // if we are inside a compound constant, retrieve the part of
                        // the compound regarding to the current path part and follow it
                        if (IValue const *subval = vc->get_value(m_path[m_depth])) {
                            node = m_dag_builder->create_constant(subval, c->get_dbg_info());
                            // no calls anymore
                            ++m_depth;
                            continue;
                        }
                    }
                    // we cannot follow the path further ...
                    return NULL;
                }
            case DAG_node::EK_TEMPORARY:
                {
                    // should not happen, but if, just ignore it
                    node = cast<DAG_temporary>(node)->get_expr();
                    continue;
                }
            case DAG_node::EK_CALL:
                {
                    DAG_call const *call = cast<DAG_call>(node);

                    if (m_resolver != NULL &&
                        call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR &&
                        is_material_type(call->get_type()))
                    {
                        // we enter a (real, i.e. not a constructor) material call
                        m_curr_call = call;
                        m_call_stack.push(call);

                        mi::base::Handle<IGenerated_code_dag const> callee_dag(
                            m_resolver->get_owner_dag(call->get_name()));

                        if (!callee_dag.is_valid_interface()) {
                            // bad, this should not happen
                            MDL_ASSERT(!"called material has no DAG");
                            return NULL;
                        }
                        // Enter the body of the called material (its value).
                        // note that we "leave" the root lambda here, but this is ok, as
                        // we will clone it, once we found the node
                        node = get_material_value(callee_dag.get(), call->get_name());
                    } else {
                        // not a material call, just follow the parameter of the given name
                        node = call->get_argument(m_path[m_depth++]);
                    }
                    continue;
                }
            case DAG_node::EK_PARAMETER:
                {
                    if (m_curr_call != NULL) {
                        // we are inside a material call, but leave it now through its argument
                        DAG_parameter const *param = cast<DAG_parameter>(node);
                        node = m_curr_call->get_argument(param->get_index());
                        m_call_stack.pop();

                        if (m_call_stack.empty()) {
                            m_curr_call = NULL;
                        } else {
                            m_curr_call = m_call_stack.top();
                        }
                        continue;
                    }
                    return NULL;
                }
            }
            MDL_ASSERT(!"unknown DAG node kind");
            return NULL;
        }
    }

private:
    typedef stack<DAG_call const *>::Type Call_stack;

    IAllocator                    *m_alloc;
    Call_stack                    m_call_stack;
    Array_ref<char const *> const &m_path;
    IDag_builder                  *m_dag_builder;
    ICall_name_resolver const     *m_resolver;
    DAG_call const                *m_curr_call;
    size_t                        m_depth;
};


/// RAII-like helper class to handle optimization flags.
template <typename T>
class Optimization_scope {
public:
    /// Constructor.
    ///
    /// \param entity  an entity that supports enable_opt(bool)
    /// \param flag    the flag for enable_opt(bool)
    Optimization_scope(T &entity, bool flag)
    : m_entity(entity)
    , m_flag(entity.enable_opt(flag))
    {
    }

    /// Destructor.
    ~Optimization_scope()
    {
        m_entity.enable_opt(m_flag);
    }

private:
    T &m_entity;
    bool m_flag;
};


/// Helper class for building distribution functions.
/// Creates expression lambda functions for all non-DF DAG nodes.
class Distribution_function_builder
{
    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Node_cache;
    typedef ptr_hash_set<DAG_node const>::Type                   Node_set;

public:
    enum Flag {
        FL_NONE                             = 0,        ///< No flags.
        FL_NEEDS_MATERIAL_IOR               = 1 << 0,   ///< Material needs material.ior.
        FL_NEEDS_MATERIAL_THIN_WALLED       = 1 << 1,   ///< Material needs material.thin_walled.
        FL_NEEDS_MATERIAL_VOLUME_ABSORPTION = 1 << 2,   ///< Material needs material.volume
                                                        ///< .absorption_coefficient.
        FL_CONTAINS_UNSUPPORTED_DF          = 1 << 3,   ///< Material contains unsupported DFs.
    };  // can be or'ed

    typedef unsigned Flags;

public:
    /// Builds a distribution function.
    ///
    /// \param dist_func            the distribution function that is build
    /// \param alloc                the memory allocator
    /// \param compiler             the MDL compiler
    /// \param resolver             the call name resolver
    /// \param mat_instance         the material instance we build the distribution function for
    /// \param requested_functions  the list of requested functions to be built from the material
    /// \param num_req_functions    the number of requested functions
    /// \param calc_derivative_infos
    ///                             if true, derivative infos will be computed
    /// \param enable_spectral_conversions
    ///                             if true, color parameters will be wrapped with spectral conversions
    static IDistribution_function::Error_code build(
        Distribution_function                      *dist_func,
        IAllocator                                 *alloc,
        IMDL                                       *compiler,
        ICall_name_resolver const                  *resolver,
        IMaterial_instance const                   *mat_instance,
        IDistribution_function::Requested_function *requested_functions,
        size_t                                      num_req_functions,
        bool                                        calc_derivative_infos,
        bool                                        enable_spectral_conversions)
    {
        if (mat_instance == NULL ||
            mat_instance->get_constructor() == NULL ||
            num_req_functions == 0 ||
            requested_functions == NULL)
        {
            return IDistribution_function::EC_INVALID_PARAMETERS;
        }

        mi::base::Handle<mi::mdl::Lambda_function> root_lambda(dist_func->get_root_lambda());

        // disable optimizations to ensure, that the expression paths will stay valid
        Optimization_scope opt_scope(*root_lambda.get(), false);

        mi::mdl::DAG_node const *root_node = root_lambda->import_expr(
            mat_instance->get_dag_unit(), mat_instance->get_constructor());

        if (calc_derivative_infos || enable_spectral_conversions) {
            // calculate derivative information and/or rebuild DAG with spectral conversions
            Derivative_infos *deriv_infos = NULL;
            if (calc_derivative_infos) {
                deriv_infos = dist_func->get_writable_derivative_infos();
                deriv_infos->set_call_name_resolver(resolver);
            }

            DAG_rebuilder dag_rebuilder(
                alloc, *root_lambda.get(), deriv_infos, enable_spectral_conversions);
            root_node = dag_rebuilder.rebuild(root_node, /*want_derivatives=*/ false);

            if (deriv_infos != NULL) {
                deriv_infos->set_call_name_resolver(NULL);
            }
        }

        // set the body to the material constructor (can be the derivative version)
        root_lambda->set_body(root_node);

        // translate all non-df nodes to call_lambda nodes
        Distribution_function_builder fct_builder(
            alloc,
            *dist_func,
            root_node,
            compiler,
            resolver,
            calc_derivative_infos);
        unsigned walk_id = 0;

        IDistribution_function::Error_code last_error = IDistribution_function::EC_NONE;

        // get the geometry.normal node
        // Note: we pass a resolver here, so we can "inline" target material model calls
        static char const * const normal_path[] = {"geometry", "normal"};
        DAG_node const *normal =
            Dag_path_follower(alloc, normal_path, root_lambda.get(), resolver)
            .get_dag_arg(root_node);

        // check whether geometry.normal is not state::normal()
        bool has_non_default_normal = false;
        if (normal != NULL) {
            if (DAG_call const *normal_call = as<DAG_call>(normal)) {
                if (normal_call->get_semantic() != IDefinition::DS_INTRINSIC_STATE_NORMAL) {
                    has_non_default_normal = true;
                }
            } else {
                has_non_default_normal = true;
            }
        }

        // process all requested functions
        for (size_t fkt_idx = 0; fkt_idx < num_req_functions; ++fkt_idx) {
            char const *path = requested_functions[fkt_idx].path;

            if (path == NULL) {
                last_error = requested_functions[fkt_idx].error_code =
                    IDistribution_function::EC_INVALID_PATH;
                continue;
            }
            string path_copy(path, alloc);

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
            if (last_start < path_copy.length()) {
                path_parts.push_back(path_copy.c_str() + last_start);
            }

            // get the node (of the root lambda) for the given path
            DAG_node const *node =
                Dag_path_follower(alloc, path_parts, root_lambda.get()).get_dag_arg(root_node);
            if (node == NULL) {
                last_error = requested_functions[fkt_idx].error_code =
                    IDistribution_function::EC_INVALID_PATH;
                continue;
            }

            // decide depending on the node's type what to do
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
            case IType::TK_SPECTRAL_SAMPLE:
            case IType::TK_SPECTRUM:
                break;

            case IType::TK_STRUCT:
                // so far we cannot generate partial DF expressions, either all or nothing
                if (contains_df_type(node_type)) {
                    last_error = requested_functions[fkt_idx].error_code =
                        IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE;
                    continue;
                }
                break;

            case IType::TK_VDF:
                // VDFs are not supported yet
                last_error = requested_functions[fkt_idx].error_code =
                    IDistribution_function::EC_UNSUPPORTED_DISTRIBUTION_TYPE;
                continue;

            case IType::TK_LIGHT_PROFILE:
            case IType::TK_TEXTURE:
            case IType::TK_BSDF_MEASUREMENT:
                // we cannot create resources, hence no functions returning them
                last_error = requested_functions[fkt_idx].error_code =
                    IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE;
                continue;

            case IType::TK_ALIAS:
            case IType::TK_FUNCTION:
            case IType::TK_PTR:
            case IType::TK_REF:
            case IType::TK_VOID:
            case IType::TK_AUTO:
            case IType::TK_ERROR:
                // should not happen
                MDL_ASSERT(!"unexpected expression type");
                last_error = requested_functions[fkt_idx].error_code =
                    IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE;
                continue;
            }

            // if we are here, we *do* support this path

            // According to MDL Spec 1.6 13.3, the geometry fields are evaluated before all surface
            // fields and the normal evaluation happens last within the geometry fields.
            Distribution_function::Eval_state eval_state;
            Distribution_function::Special_kind special_kind = Distribution_function::SK_INVALID;
            if (path_parts.size() == 2 && strcmp(path_parts[0], "geometry") == 0) {
                eval_state = Distribution_function::ES_BEGIN_STATE;

                // remap the "geometry.normal" path to a state::normal() call
                if (strcmp(path_parts[1], "normal") == 0)
                {
                    IType const *float3_type = fct_builder.m_type_factory.create_vector(
                        fct_builder.m_type_factory.create_float(), 3);

                    node = fct_builder.m_root_lambda->create_call(
                        "::state::normal()",
                        IDefinition::DS_INTRINSIC_STATE_NORMAL,
                        NULL,
                        0,
                        float3_type,
                        DAG_DbgInfo::builtin);

                    // this will actually be evaluated after state normal has been updated
                    eval_state = Distribution_function::ES_AFTER_GEOMETRY_NORMAL;
                } else if (strcmp(path_parts[1], "displacement") == 0) {
                    special_kind = Distribution_function::SK_MATERIAL_GEOMETRY_DISPLACEMENT;
                } else if (strcmp(path_parts[1], "cutout_opacity") == 0) {
                    special_kind = Distribution_function::SK_MATERIAL_GEOMETRY_CUTOUT_OPACITY;
                }
            } else {
                // not in geometry
                eval_state = Distribution_function::ES_AFTER_GEOMETRY_NORMAL;
            }
            bool is_eval_state_dependent =
                fct_builder.collect_flags_and_df_handles(node, ++walk_id);

            size_t req_node_index = dist_func->add_requested_node(
                node,
                path,
                requested_functions[fkt_idx].base_fname,
                eval_state);

            // only register the displacement / cutout_opacity special nodes,
            // if they depend on state::normal() and geometry.normal would change the normal.
            // Then, special care must be taken to evaluate those nodes before geometry.normal.
            if (special_kind != Distribution_function::SK_INVALID &&
                    is_eval_state_dependent &&
                    has_non_default_normal) {
                dist_func->set_special_node(special_kind, req_node_index);
            }
        }

        // bail out if at least one path generated an error
        if (last_error != IDistribution_function::EC_NONE) {
            return last_error;
        }

        // All requested nodes added until now have been explicitly requested by the application.
        dist_func->set_explicit_requested_node_count(dist_func->get_total_requested_node_count());

        // The DF handles used by all requests have been processed now, so m_handle_name_set
        // can be reused to collect per-request DF handles, now.
        for (size_t i = 0, n = dist_func->get_total_requested_node_count(); i < n; ++i) {
            // collect DF handles used by this requested node
            Distribution_function::Requested_node *req_node = dist_func->get_requested_node(i);
            fct_builder.m_handle_name_set.clear();
            fct_builder.collect_req_node_df_handles(req_node->node, req_node, ++walk_id);
        }

        // Now implicitly requested nodes may be added

        Distribution_function_builder::Flags mat_flags = fct_builder.get_flags();
        if ((mat_flags & Distribution_function_builder::FL_CONTAINS_UNSUPPORTED_DF) != 0) {
            return IDistribution_function::EC_UNSUPPORTED_BSDF;
        }

        // Register all special nodes required by the material

        if (has_non_default_normal) {
            fct_builder.register_special_node(
                normal, "geometry.normal", Distribution_function::SK_MATERIAL_GEOMETRY_NORMAL);
        }
        if ((mat_flags & Distribution_function_builder::FL_NEEDS_MATERIAL_IOR) != 0) {
            fct_builder.register_special_node(
                { "ior" }, "ior", Distribution_function::SK_MATERIAL_IOR);
        }
        if ((mat_flags & Distribution_function_builder::FL_NEEDS_MATERIAL_THIN_WALLED) != 0) {
            fct_builder.register_special_node(
                { "thin_walled" }, "thin_walled", Distribution_function::SK_MATERIAL_THIN_WALLED);
        }

        return IDistribution_function::EC_NONE;
    }

    /// Constructor.
    Distribution_function_builder(
        IAllocator                *alloc,
        Distribution_function     &dist_func,
        DAG_node const            *mat_root_node,
        IMDL                      *compiler,
        ICall_name_resolver const *resolver,
        bool                      calc_derivative_infos)
    : m_alloc(alloc)
    , m_compiler(compiler, mi::base::DUP_INTERFACE)
    , m_dist_func(dist_func)
    , m_deriv_infos(calc_derivative_infos ? dist_func.get_writable_derivative_infos() : NULL)
    , m_mat_root_node(mat_root_node)
    , m_root_lambda(dist_func.get_root_lambda())
    , m_type_factory(*m_root_lambda->get_type_factory())
    , m_resolver(resolver)
    , m_node_info_map(0, Node_info_map::hasher(), Node_info_map::key_equal(), alloc)
    , m_flags(FL_NONE)
    , m_handle_name_set(0, Handle_name_set::hasher(), Handle_name_set::key_equal(), alloc)
    {
    }

    /// Returns the handle of an elemental DF or NULL if the call is not an elemental DF.
    ///
    /// \param call  a DAG call node (should be a elemental DF call)
    char const *get_elemental_handle(DAG_call const *call)
    {
        if (!is_elemental_df_semantics(call->get_semantic())) {
            return NULL;
        }

        DAG_node const *handle_node = call->get_argument("handle");
        MDL_ASSERT(handle_node && is<DAG_constant>(handle_node) &&
            "Elemental DF must have a constant handle argument");
        if (handle_node == NULL || !is<DAG_constant>(handle_node)) {
            return NULL;
        }

        DAG_constant const *handle_const = cast<DAG_constant>(handle_node);
        IValue const *handle_val = handle_const->get_value();
        IValue_string const *handle_str = as<IValue_string>(handle_val);
        MDL_ASSERT(handle_str != NULL && "DF handle must be string");
        if (handle_str == NULL) {
            return NULL;
        }

        return handle_str->get_value();
    }

    /// Walk the expression to collect the flags and the used df handles
    /// and determine, whether the expression is evaluation state dependent.
    /// If so, the function returns true.
    bool collect_flags_and_df_handles(
        DAG_node const *expr,
        unsigned       &walk_id)
    {
        Node_info &info = m_node_info_map[expr];
        if (info.already_visited(walk_id)) {
            return info.is_eval_state_dependent;
        }
        info.mark_visited(walk_id);

        bool res = false;
        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                res = collect_flags_and_df_handles(expr, walk_id);
                break;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            // note: parameters cannot be evaluation state dependent. If state::normal()
            //    was used as a argument during material instantiation, the
            //    corresponding parameter would not be a parameter anymore (but inlined).
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                IDefinition::Semantics sema = call->get_semantic();

                bool is_df_sema = is_df_semantics(sema);
                if (is_df_sema) {
                    if (needs_thin_walled(sema)) {
                        m_flags |= FL_NEEDS_MATERIAL_THIN_WALLED;
                    }

                    if (needs_ior(sema)) {
                        m_flags |= FL_NEEDS_MATERIAL_IOR;
                    }

                    if (char const *handle_name = get_elemental_handle(call)) {
                        if (m_handle_name_set.count(handle_name) == 0) {
                            // the handle is not known yet -> register it
                            m_dist_func.add_df_handle(handle_name);
                            m_handle_name_set.insert(handle_name);
                        }
                    }
                }

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    res |= collect_flags_and_df_handles(arg, walk_id);
                }

                // only check this call, if we haven't found a state dependent call, yet
                if (!res) {
                    res = is_eval_state_dependent_direct(call);
                }

                break;
            }
        }

        info.is_eval_state_dependent = res;

        return res;
    }

    /// Collect the DF handles for a requested node function.
    void collect_req_node_df_handles(
        DAG_node const                        *expr,
        Distribution_function::Requested_node *req_node,
        unsigned                               walk_id)
    {
        Node_info &info = m_node_info_map[expr];
        if (info.already_visited(walk_id)) {
            return;
        }
        info.mark_visited(walk_id);

        switch (expr->get_kind()) {
        case DAG_node::EK_TEMPORARY:
            {
                // should not happen, but we can handle it
                DAG_temporary const *t = cast<DAG_temporary>(expr);
                expr = t->get_expr();
                collect_req_node_df_handles(expr, req_node, walk_id);
                break;
            }
        case DAG_node::EK_CONSTANT:
        case DAG_node::EK_PARAMETER:
            break;
        case DAG_node::EK_CALL:
            {
                DAG_call const *call = cast<DAG_call>(expr);
                // stop at non-DFs
                if (!contains_df_type(call->get_type())) {
                    break;
                }

                if (char const *handle_name = get_elemental_handle(call)) {
                    if (m_handle_name_set.count(handle_name) == 0) {
                        // the handle is not known, yet -> register it in the requested node
                        req_node->df_handles.push_back(handle_name);
                        m_handle_name_set.insert(handle_name);
                    }
                }

                int n_args = call->get_argument_count();
                for (int i = 0; i < n_args; ++i) {
                    DAG_node const *arg = call->get_argument(i);
                    collect_req_node_df_handles(arg, req_node, walk_id);
                }
                break;
            }
        }
    }

    /// Checks whether the type is a DF type or contains a DF type.
    ///
    /// \param type  the type to check
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
                    if (contains_df_type(comp_type->get_compound_type(i))) {
                        return true;
                    }
                }
                return false;
            }
        default:
            return false;
        }
    }

    /// Get the flags determined during walking the material.
    Flags get_flags() const { return m_flags; }

    /// Register a special node.
    ///
    /// \param node     the node representing the body of the lambda
    /// \param path     the path to the node in the material instance
    /// \param kind     the special kind of the node
    void register_special_node(
        DAG_node const                      *node,
        char const                          *path,
        Distribution_function::Special_kind  kind)
    {
        m_dist_func.add_special_node(kind, node, path);
    }

    /// Register a special node.
    ///
    /// \param expr_path  the path from the material root to the special node
    /// \param path       the original path string of expr_path
    /// \param kind       the special kind of the node
    /// \param walk_id    current walk id
    void register_special_node(
        Array_ref<char const *>              expr_path,
        char const                          *path,
        Distribution_function::Special_kind  kind)
    {
        DAG_node const *node =
            Dag_path_follower(m_alloc, expr_path, m_root_lambda.get())
                .get_dag_arg(m_mat_root_node);

        register_special_node(node, path, kind);
    }

private:
    /// Determines whether the called function is depending on the evaluation state
    /// not considering the arguments.
    ///
    /// \param call  the call to check
    bool is_eval_state_dependent_direct(DAG_call const *call)
    {
        char const *signature = call->get_name();
        if (signature[0] == '#') {
            // skip prefix for derivative variants
            ++signature;
        }
        mi::base::Handle<Module const> mod(
            impl_cast<Module>(m_resolver->get_owner_module(signature)));
        if (!mod.is_valid_interface()) {
            return false;
        }

        Module const *module = mod.get();

        IDefinition const *def = module->find_signature(signature, /*only_exported=*/false);
        if (def == NULL) {
            return false;
        }

        // skip presets
        def = skip_presets(def, mod);

        // as we divide only in "before state::normal()" and "after state::normal()", we
        // just check for this property here
        return def->get_property(IDefinition::DP_USES_NORMAL);
    }

    /// Checks whether the given BSDF semantic needs access to the material.thin_walled field.
    bool needs_thin_walled(IDefinition::Semantics sema) {
        switch (sema) {
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
        case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:
        case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
            return true;

        default:
            return false;
        }
    }

    /// Checks whether the given BSDF semantic needs access to the material.ior field.
    bool needs_ior(IDefinition::Semantics sema) {
        switch (sema) {
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

    /// Helper struct collecting information about DAG nodes.
    struct Node_info {
        /// True if the value of this node depends on the evaluation state.
        bool is_eval_state_dependent;

        /// Graph walk ID used as visited marker for different walks.
        unsigned last_walk_id;

        /// Default Constructor.
        Node_info()
        : is_eval_state_dependent(false)
        , last_walk_id(0)
        {
        }

        /// Return true, if this node has already been visited in this walk.
        ///
        /// \param walk_id  the ID of the current walk
        bool already_visited(unsigned walk_id) const {
            return last_walk_id == walk_id;
        }

        /// Mark the node as visited in this walk.
        ///
        /// \param walk_id  the ID of the current walk
        void mark_visited(unsigned walk_id) {
            last_walk_id = walk_id;
        }
    };

    typedef ptr_hash_map<DAG_node const, Node_info>::Type Node_info_map;

    /// Maps from DAG nodes created via the builder to an information structure.
    Node_info_map m_node_info_map;

    /// Collected flags.
    Flags m_flags;

    typedef hash_set<
        char const *,
        cstring_hash,
        cstring_equal_to
    >::Type Handle_name_set;

    /// Set of already seen handle names. Will be used for the handles over all requests,
    /// as well as request.
    Handle_name_set m_handle_name_set;
};

}  // anonymous


// Constructor.
Distribution_function_dumper::Distribution_function_dumper(
    IAllocator                  *alloc,
    IOutput_stream              *out,
    Distribution_function const *dist_func,
    Node_color_map              &node_color_map)
: Base(alloc, out)
, m_dist_func(dist_func)
, m_node_color_map(node_color_map)
{
    m_name_buf[0] = 0;

    mi::base::Handle<Lambda_function> root_lambda(m_dist_func->get_root_lambda());
    set_dag_unit(&root_lambda->get_dag_unit());
}

// Dump the lambda expression DAG to the output stream.
void Distribution_function_dumper::dump()
{
    m_printer->print("digraph \"distribution_function\" {\n"
        "  node [fillcolor=\"#E0E0A8\"]\n");

    mi::base::Handle<Lambda_function> root_lambda(m_dist_func->get_root_lambda());

    m_walker.walk_node(const_cast<DAG_node *>(root_lambda->get_body()), this);
    m_printer->print("}\n");
}

// Get the parameter name for the given index if any.
const char *Distribution_function_dumper::get_parameter_name(int index)
{
    mi::base::Handle<mi::mdl::Lambda_function> root_lambda(m_dist_func->get_root_lambda());
    return root_lambda.get()->get_parameter_name(index);
}

// Get a color for the given node or nullptr for default
char const *Distribution_function_dumper::get_node_color(DAG_node const *node)
{
    Node_color_map::const_iterator it = m_node_color_map.find(node);
    if (it == m_node_color_map.cend()) {
        return nullptr;
    } else {
        return it->second;
    }
}

// Get a prefix for the label for the given node or nullptr for no prefix.
char const *Distribution_function_dumper::get_node_label_prefix(DAG_node const *node)
{
    snprintf(m_name_buf, sizeof(m_name_buf), "#%u | ", unsigned(node->get_id()));
    return m_name_buf;
}


// Constructor.
Distribution_function::Distribution_function(
    IAllocator *alloc,
    MDL        *compiler)
: Base(alloc)
, m_mdl(mi::base::make_handle_dup(compiler))
, m_root_lambda(
    impl_cast<Lambda_function>(compiler->create_lambda_function(Lambda_function::LEC_CORE)))
, m_explicit_requested_node_count(0)
, m_requested_nodes(alloc)
, m_deriv_infos_calculated(false)
, m_deriv_infos(alloc)
, m_df_handles(alloc)
, m_arena(alloc)
, m_resource_tag_map(alloc)
{
    Lambda_function *root_lambda = m_root_lambda.get();

    // force always using varying state
    root_lambda->set_uses_varying_state(true);

    for (size_t i = 0, n = dimension_of(m_special_nodes); i < n; ++i) {
        m_special_nodes[i] = ~0;
    }
}

/// Initialize this distribution function object for the given material
/// with the given distribution function node. Any additionally required
/// expressions from the material will also be handled.
IDistribution_function::Error_code Distribution_function::initialize(
    IMaterial_instance const  *mat_instance,
    Requested_function        *requested_functions,
    size_t                     num_req_functions,
    bool                       calc_derivative_infos,
    bool                       enable_spectral_conversions,
    ICall_name_resolver const *name_resolver)
{
    m_deriv_infos_calculated = calc_derivative_infos;
    return Distribution_function_builder::build(
        this,
        get_allocator(),
        m_mdl.get(),
        name_resolver,
        mat_instance,
        requested_functions,
        num_req_functions,
        calc_derivative_infos,
        enable_spectral_conversions);
}

// Get the root lambda function used to build nodes and manage parameters and resources.
Lambda_function *Distribution_function::get_root_lambda() const
{
    m_root_lambda.get()->retain();
    return m_root_lambda.get();
}

// Add a requested node.
size_t Distribution_function::add_requested_node(
    DAG_node const *node,
    char const     *path,
    char const     *function_name,
    Eval_state      eval_state)
{
    path = path != nullptr ? Arena_strdup(m_arena, path) : nullptr;
    function_name = function_name != nullptr ? Arena_strdup(m_arena, function_name) : nullptr;
    m_requested_nodes.emplace_back(get_allocator(), node, path, function_name, eval_state);
    return m_requested_nodes.size() - 1;
}

// Get the requested node for the given index.
Distribution_function::Requested_node *Distribution_function::get_requested_node(
    size_t index)
{
    if (index >= m_requested_nodes.size()) {
        return nullptr;
    }

    return &m_requested_nodes[index];
}

// Get the requested node for the given index.
Distribution_function::Requested_node const *Distribution_function::get_requested_node(
    size_t index) const
{
    if (index >= m_requested_nodes.size()) {
        return nullptr;
    }

    return &m_requested_nodes[index];
}

// Set a special node for getting certain material properties.
void Distribution_function::add_special_node(
    Special_kind    kind,
    DAG_node const *node,
    char const     *path)
{
    if (kind <= SK_INVALID || kind >= SK_NUM_KINDS) {
        MDL_ASSERT(!"Invalid special kind");
        return;
    }

    m_special_nodes[kind] = add_requested_node(
        node,
        path,
        nullptr,
        kind == Special_kind::SK_MATERIAL_GEOMETRY_NORMAL
        ? Eval_state::ES_BEGIN_STATE : Eval_state::ES_AFTER_GEOMETRY_NORMAL);
}

// Set a special node for getting certain material properties.
void Distribution_function::set_special_node(
    Special_kind  kind,
    size_t        requested_node_index)
{
    if (kind <= SK_INVALID || kind >= SK_NUM_KINDS) {
        MDL_ASSERT(!"Invalid special kind");
        return;
    }

    m_special_nodes[kind] = requested_node_index;
}

// Get the requested node index for the given special node kind.
size_t Distribution_function::get_special_node_index(Special_kind kind) const
{
    if (kind <= SK_INVALID || kind >= SK_NUM_KINDS) {
        MDL_ASSERT(!"Invalid special kind");
        return ~0;
    }

    return m_special_nodes[kind];
}

/// Get the resource attribute map of this distribution function.
Resource_attr_map const &Distribution_function::get_resource_attribute_map() const {
    return m_root_lambda->get_resource_attribute_map();
}

// Set a tag, version pair for a resource value that might be reachable from this function.
void Distribution_function::set_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url,
    char const                     *res_sel,
    int                            tag)
{
    int old_tag = find_resource_tag(res_kind, res_url, res_sel);

    if (old_tag == 0) {
        add_resource_tag(res_kind, res_url, res_sel, tag);
    } else {
        MDL_ASSERT(old_tag == tag && "Changing tag of a resource");
    }
}

// Find the resource tag of a resource.
int Distribution_function::find_resource_tag(
    Resource_tag_tuple::Kind const res_kind,
    char const                     *res_url,
    char const                     *res_sel) const
{
    // beware of NULL pointer
    if (res_url == nullptr) {
        res_url = "";
    }
    if (res_sel == nullptr) {
        res_sel = "";
    }

    // linear search so far
    for (size_t i = 0, n = m_resource_tag_map.size(); i < n; ++i) {
        Resource_tag_tuple const &e = m_resource_tag_map[i];

        if (e.m_kind == res_kind &&
            (e.m_url      == res_url || strcmp(e.m_url,      res_url) == 0) &&
            (e.m_selector == res_sel || strcmp(e.m_selector, res_sel) == 0)) {
            return e.m_tag;
        }
    }
    return 0;
}

// Add tag, version pair for a resource value that might be reachable from this function.
void Distribution_function::add_resource_tag(
    Resource_tag_tuple::Kind res_kind,
    char const               *res_url,
    char const               *res_sel,
    int                      tag)
{
    res_url = res_url != NULL ? Arena_strdup(m_arena, res_url) : NULL;
    m_resource_tag_map.push_back(Resource_tag_tuple(res_kind, res_url, res_sel, tag));
}

// Returns the number of distribution function handles referenced by this distribution function.
size_t Distribution_function::get_df_handle_count() const
{
    return m_df_handles.size();
}

// Returns a distribution function handle referenced by this distribution function.
char const *Distribution_function::get_df_handle(size_t index) const
{
    if (index >= m_df_handles.size()) {
        return NULL;
    }
    return m_df_handles[index];
}

// Get the derivative information if they were requested during initialization.
Derivative_infos const *Distribution_function::get_derivative_infos() const
{
    if (!m_deriv_infos_calculated) {
        return NULL;
    }
    return &m_deriv_infos;
}

// Dump the distribution function to a .gv file with the given name.
void Distribution_function::dump(char const *name) const
{
    Allocator_builder builder(get_allocator());

    if (FILE *f = fopen(name, "w")) {
        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(get_allocator(), f, /*close_at_destroy=*/true));

        Distribution_function_dumper::Node_color_map node_color_map(get_allocator());
        for (size_t i = 0, n = get_total_requested_node_count(); i < n; ++i) {
            Distribution_function::Requested_node const *req = get_requested_node(i);
            node_color_map[req->node] = "\"#E29245\"";  // orange: requested node
        }

        for (int i = 0; i < Distribution_function::SK_NUM_KINDS; ++i) {
            size_t special_node_index = get_special_node_index(
                Distribution_function::Special_kind(i));
            if (special_node_index == ~0) {
                continue;
            }

            Distribution_function::Requested_node const *special_node =
                get_requested_node(special_node_index);

            // only mark special node, if they were not requested specifically
            if (node_color_map.find(special_node->node) == node_color_map.end()) {
                // green: special nodes, indirectly requested node
                node_color_map[special_node->node] = "\"#92E245\"";
            }
        }

        Distribution_function_dumper dumper(get_allocator(), out.get(), this, node_color_map);
        dumper.dump();
    }
}

} // mdl
} // mi
