/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <utility>

#include <mi/mdl/mdl_expressions.h>

#include <mdl/compiler/compilercore/compilercore_analysis.h>
#include <mdl/compiler/compilercore/compilercore_visitor.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include <mdl/codegenerators/generator_dag/generator_dag_lambda_function.h>

#include "generator_dag_derivatives.h"

namespace mi {
namespace mdl {


typedef Store<bool> Flag_scope;

/// Element of a linked list of IExpression objects.
struct Expr_list_node
{
    /// Constructor.
    Expr_list_node(IExpression const *expr, Expr_list_node *next)
    : expr(expr)
    , next(next)
    {
    }

    /// The expression.
    IExpression const *expr;

    /// The pointer to the next linked list element.
    Expr_list_node *next;
};


/// Returns, whether the given math function currently supports derivative arguments.
bool is_math_deriv_args_supported(IDefinition::Semantics math_sema)
{
    switch (math_sema) {
    case IDefinition::DS_INTRINSIC_MATH_EMISSION_COLOR:
        // derivative arguments not supported
        return false;
    case IDefinition::DS_INTRINSIC_MATH_STEP:
        // don't provide derivatives, as the derivatives don't depend on the arguments
        return false;
    default:
        // derivative arguments supported
        return true;
    }
}


/// Check whether the given type is based on a floating point type.
bool is_floating_point_based_type(IType const *type)
{
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_COLOR:
    case IType::TK_MATRIX:  // there are only float and double matrix types
        return true;

    case IType::TK_VECTOR:
        {
            IType_vector const *vec_type = as<IType_vector>(type);
            IType_atomic const *elem_type = vec_type->get_element_type();
            IType::Kind elem_kind = elem_type->get_kind();
            return elem_kind == IType::TK_FLOAT || elem_kind == IType::TK_DOUBLE;
        }

    case IType::TK_ARRAY:
        {
            IType_array const *array_type = as<IType_array>(type);
            IType const *elem_type = array_type->get_element_type();
            return is_floating_point_based_type(elem_type);
        }

    default:
        return false;
    }
}


/// Helper class analyzing derivative information for functions.
class Function_processor : public Module_visitor
{
    typedef ptr_hash_map<IDefinition const, Expr_list_node *>::Type Assignment_map;

public:
    /// Constructor.
    Function_processor(IAllocator *alloc, Derivative_infos &deriv_infos, Module const *module)
    : m_alloc(alloc)
    , m_arena(alloc)
    , m_builder(m_arena)
    , m_deriv_infos(deriv_infos)
    , m_module(module)
    , m_func_inst(NULL)
    , m_assignments(0, Assignment_map::hasher(), Assignment_map::key_equal(), alloc)
    , m_ret_statements(alloc)
    , m_visited_defs_set(0, Definition_set::hasher(), Definition_set::key_equal(), alloc)
    , m_var_worklist(alloc)
    , m_call_worklist(alloc)
    , m_want_derivs_vars(NULL)
    , m_want_derivs_params(alloc)
    , m_exprs_want_derivatives(NULL)
    , m_user_func_calls(alloc)
    , m_want_derivatives(false)
    {
    }

    /// Analyze the given function instance.
    Func_deriv_info *process(Function_instance const &func_inst)
    {
        m_func_inst = &func_inst;

        IDefinition const *def = m_func_inst->get_def();
        mi::mdl::IDeclaration_function const *func_decl =
            cast<mi::mdl::IDeclaration_function>(def->get_declaration());

        int num_params = func_decl->get_parameter_count();
        Func_deriv_info *info = m_deriv_infos.alloc_function_derivative_infos(num_params);
        info->returns_derivatives = m_func_inst->get_return_derivs();
        m_want_derivs_vars = &info->vars_want_derivatives;
        m_exprs_want_derivatives = &info->exprs_want_derivatives;

        IStatement const *stmt = func_decl->get_body();

        // collect all variable assignments and mark initial "want-derivatives" variables
        // and functions from known semantics and requested return derivative mask.
        // note: the visitor requires the const_cast
        visit(const_cast<IStatement *>(stmt));

        // now go through all assignments to want-derivatives-variables and process the
        // associated expressions
        m_want_derivatives = true;
        while (!m_var_worklist.empty() || !m_call_worklist.empty()) {
            // spread all variable information
            while (!m_var_worklist.empty()) {
                IDefinition const *cur_var = m_var_worklist.back();
                m_var_worklist.pop_back();
                Assignment_map::const_iterator it = m_assignments.find(cur_var);
                if (it == m_assignments.end()) continue;

                Expr_list_node *list_node = it->second;
                while (list_node) {
                    visit(list_node->expr);
                    list_node = list_node->next;
                }
            }

            // process one call, if available.
            // only processing one call should reduce the number of redundantly analyzed function
            // instances
            if (!m_call_worklist.empty()) {
                std::pair<IExpression_call const *, bool> workitem = m_call_worklist.back();
                m_call_worklist.pop_back();

                IExpression_call const *call = workitem.first;
                IExpression_reference const *ref =
                    cast<IExpression_reference>(call->get_reference());
                MDL_ASSERT(!ref->is_array_constructor() && "Array constructor added to worklist!");

                Module const *owner = m_module;
                IDefinition const *def = ref->get_definition();
                if (def->get_property(IDefinition::DP_IS_IMPORTED))
                    def = m_module->get_original_definition(def, owner);

                mi::base::Handle<IModule const> i_owner(owner, mi::base::DUP_INTERFACE);
                def = skip_presets(def, i_owner);

                // get derivative information for called function
                Function_instance::Array_instances array_insts(m_alloc);
                Function_instance func_inst(def, array_insts, workitem.second);
                Func_deriv_info *infos = m_deriv_infos.get_or_calc_function_derivative_infos(
                    impl_cast<Module>(i_owner.get()), func_inst);

                // any derivatives wanted at all?
                if (infos->args_want_derivatives.test_bit(0)) {
                    // visit all arguments for which the corresponding parameter wants derivatives
                    for (size_t i = 1, n = infos->args_want_derivatives.get_size(); i < n; ++i) {
                        if (infos->args_want_derivatives.test_bit(i)) {
                            visit(call->get_argument(i - 1));
                        }
                    }
                }
            }
        }

        // determine for which arguments the called must provide derivatives based
        // on the analysis information about the parameters
        for (int i = 0; i < num_params; ++i) {
            IParameter const *param = func_decl->get_parameter(i);
            IDefinition const *param_def = param->get_name()->get_definition();
            if (m_want_derivs_params.find(param_def) != m_want_derivs_params.end()) {
                info->args_want_derivatives.set_bit(0);
                info->args_want_derivatives.set_bit(i + 1);
            }
        }

        m_func_inst = NULL;

        return info;
    }

    /// Fallback pre_visit function for non implemented expression types.
    bool pre_visit(IExpression *expr) MDL_OVERRIDE
    {
        if (m_want_derivatives) {
            m_exprs_want_derivatives->insert(expr);
        }
        return true;
    }

    /// Post-visitor for a variable declaration.
    /// Collects variable initializers.
    void post_visit(IDeclaration_variable *var_decl) MDL_OVERRIDE
    {
        for (int i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
            IDefinition const *var_def = var_decl->get_variable_name(i)->get_definition();
            if (IExpression const *init = var_decl->get_variable_init(i))
                add_assign_expr(var_def, init);
        }
    }

    /// Visitor for a condition expression.
    /// Makes sure, no derivatives are requested for the condition.
    bool pre_visit(IExpression_conditional *expr) MDL_OVERRIDE
    {
        if (!pre_visit(static_cast<IExpression *>(expr)))
            return false;

        // No derivatives for the condition
        {
            Flag_scope flag_scope(m_want_derivatives, false);
            visit(expr->get_condition());
        }

        visit(expr->get_true());
        visit(expr->get_false());

        return false;
    }

    /// Pre-visitor for a binary expression with special cases.
    bool pre_visit(IExpression_binary *expr) MDL_OVERRIDE
    {
        if (!pre_visit(static_cast<IExpression *>(expr)))
            return false;

        // Special cases for some operators
        switch (expr->get_operator()) {
        case IExpression_binary::OK_ARRAY_INDEX: {
            visit(expr->get_left_argument());

            // No derivatives for array indices
            Flag_scope flag_scope(m_want_derivatives, false);
            visit(expr->get_right_argument());
            return false;
        }
        case IExpression_binary::OK_SELECT: {
            visit(expr->get_left_argument());

            // No derivatives for the selected member
            Flag_scope flag_scope(m_want_derivatives, false);
            visit(expr->get_right_argument());
            return false;
        }
        default:
            break;
        }

        return true;
    }

    /// Post-visitor for a binary expression.
    /// Collects assignment expressions.
    IExpression *post_visit(IExpression_binary *expr) MDL_OVERRIDE
    {
        // not an assignment operator?
        if (!is_binary_assign_operator(IExpression::Operator(expr->get_operator()))) {
            return expr;
        }

        IExpression const *l = expr->get_left_argument();
        if (IDefinition const *def = Analysis::get_lvalue_base(l)) {
            // add whole assignment expression, as the operator also needs to be handled
            add_assign_expr(def, expr);
        }
        return expr;
    }

    /// Post-visitor for a reference expression.
    /// Collects variables and parameters.
    IExpression *post_visit(IExpression_reference *ref) MDL_OVERRIDE
    {
        if (!m_want_derivatives) {
            return ref;
        }

        if (ref->is_array_constructor()) {
            return ref;  // nothing to do
        }

        IDefinition const *def = ref->get_definition();
        if (m_visited_defs_set.insert(def).second) {
            // variable not processed, yet, so mark as want-derivatives
            if (def->get_kind() == IDefinition::DK_VARIABLE) {
                m_want_derivs_vars->insert(def);
            } else if (def->get_kind() == IDefinition::DK_PARAMETER) {
                m_want_derivs_vars->insert(def);
                m_want_derivs_params.insert(def);
            } else {
                return ref;  // nothing to do
            }

            // add to work list for further processing
            m_var_worklist.push_back(def);
        }
        return ref;
    }

    /// Pre-visitor for call expressions.
    /// Handles some known function calls and adds unknown functions to a worklist.
    bool pre_visit(IExpression_call *call) MDL_OVERRIDE
    {
        if (m_want_derivatives) {
            m_exprs_want_derivatives->insert(call);
        }

        IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());
        if (ref->is_array_constructor()) {
            // want-derivatives also applies to all arguments, if set
            return true;
        } else {
            IDefinition const *def = ref->get_definition();
            IDefinition::Semantics sema = def->get_semantics();
            switch (sema) {
            case IDefinition::DS_UNKNOWN:
                {
                    // for user-defined functions, the arguments may not be processed, yet,
                    // because we don't know, whether the function expects derivatives as parameters
                    m_user_func_calls.push_back(call);

                    m_call_worklist.push_back(std::make_pair(call, m_want_derivatives));
                }
                return false;

            case IDefinition::DS_CONV_CONSTRUCTOR:
            case IDefinition::DS_CONV_OPERATOR:
                // only calculate argument derivatives for conversion from float-based
                // to float-based, otherwise they will be zero anyways
                if (is_floating_point_based_type(call->get_type()) &&
                        is_floating_point_based_type(
                            call->get_argument(0)->get_argument_expr()->get_type()))
                    return true;

                {
                    // no derivatives for the arguments
                    Flag_scope flag_scope(m_want_derivatives, false);
                    visit(call->get_argument(0));
                }

                // all arguments already visited
                return false;

            case IDefinition::DS_COPY_CONSTRUCTOR:
            case IDefinition::DS_ELEM_CONSTRUCTOR:
            case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
            case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
            case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
                // want-derivatives also applies to all arguments, if set
                return true;

            case IDefinition::DS_INTRINSIC_TEX_LOOKUP_COLOR:
            case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
            case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
            case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
            case IDefinition::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
                {
                    // we know how the lookup functions work, so we process them here

                    bool supports_derivatives = false;
                    IExpression const *tex = call->get_argument(0)->get_argument_expr();
                    if (IType_texture const *tex_type = as<IType_texture>(tex->get_type())) {
                        switch (tex_type->get_shape()) {
                        case IType_texture::TS_2D:
                            supports_derivatives = true;
                            break;
                        case IType_texture::TS_3D:
                        case IType_texture::TS_CUBE:
                        case IType_texture::TS_PTEX:
                        case IType_texture::TS_BSDF_DATA:
                            // not supported
                            break;
                        }
                    }

                    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
                        // We only want derivatives for the coord argument, if supported
                        Flag_scope flag_scope(m_want_derivatives, supports_derivatives && i == 1);
                        visit(call->get_argument(i));
                    }
                }

                // all arguments already visited
                return false;

            case IDefinition::DS_INTRINSIC_MATH_DX:
            case IDefinition::DS_INTRINSIC_MATH_DY:
                {
                    // Dx/Dy always wants derivatives from the only argument
                    Flag_scope flag_scope(m_want_derivatives, true);
                    visit(call->get_argument(0));
                }

                // all arguments already visited
                return false;

            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
            case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
                // want-derivatives also applies to all (derivable) arguments, if set
                return true;

            default:
                // for math functions want-derivatives also applies to all arguments, if set
                if (IDefinition::DS_INTRINSIC_MATH_FIRST <= sema &&
                        sema <= IDefinition::DS_INTRINSIC_MATH_LAST) {
                    if (is_math_deriv_args_supported(sema) &&
                        is_floating_point_based_type(call->get_type()))
                    {
                        return true;
                    }
                }

                {
                    // no derivatives for the arguments, by default
                    Flag_scope flag_scope(m_want_derivatives, false);
                    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
                        visit(call->get_argument(i));
                    }
                }

                // all arguments already visited
                return false;
            }
        }
    }

    /// Visitor for return statements.
    /// Return statements are sinks for derivatives, if the function instance should return
    /// derivatives.
    bool pre_visit(IStatement_return *stmt) MDL_OVERRIDE
    {
        if (m_func_inst->get_return_derivs()) {
            m_ret_statements.push_back(stmt);

            Flag_scope flag_scope(m_want_derivatives, true);
            visit(stmt->get_expression());
            // all arguments already visited
            return false;
        }
        return true;
    }

private:
    /// Add an assignment expression or a variable initialization to the list of assignments
    /// for the given variable definition.
    void add_assign_expr(IDefinition const *def, IExpression const *expr)
    {
        m_assignments[def] = m_builder.create<Expr_list_node>(expr, m_assignments[def]);
    }

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The arena allocator for the arena builder.
    Memory_arena m_arena;

    /// The arena builder used for storing Expr_list_node objects.
    Arena_builder m_builder;

    /// The Derivative_infos object which will receive the analysis information.
    Derivative_infos &m_deriv_infos;

    /// The module containing the body of the function to be processed.
    Module const *m_module;

    /// The current function instance being processed.
    Function_instance const *m_func_inst;

    /// Map from variable definitions to a linked list of assignment expressions or initializations
    /// assigning to the variable.
    Assignment_map m_assignments;

    /// List of return statements.
    vector<IStatement_return const *>::Type m_ret_statements;

    typedef ptr_hash_set<IDefinition const>::Type Definition_set;

    /// Set of visited referenced definitions.
    Definition_set m_visited_defs_set;

    /// Work list containing variables and parameters for which derivatives are wanted.
    vector<IDefinition const *>::Type m_var_worklist;

    /// Work list containing calls and whether return element derivatives are wanted.
    vector<std::pair<IExpression_call const *, bool> >::Type m_call_worklist;

    /// Set of variables for which we want derivative information.
    Func_deriv_info::Definition_set *m_want_derivs_vars;

    /// Set of parameters for which we want derivative information.
    Definition_set m_want_derivs_params;

    /// Set of expressions for which we want derivative information;
    Func_deriv_info::Expression_set *m_exprs_want_derivatives;

    /// List of calls to user-defined functions which need to be processed when more information
    /// is available.
    vector<IExpression_call const *>::Type m_user_func_calls;

    /// If true, an enclosing expression wants derivatives and all variables and functions in the
    /// sub-expressions should be marked as wanting derivatives, too.
    bool m_want_derivatives;
};


/// Determine for which arguments of a function derivative values should be provided.
///
/// \param def            the function definition
/// \param return_derivs  if true, the function should return an derivative value
/// \param arg_derivs     the bitset to be calculated, specifying for which arguments derivative
///                       values should be provided. The first bit indicates whether any derivative
///                       values should be provided at all.
template <typename V, typename A>
void set_known_function_argument_derivs(
    IDefinition const *def,
    bool return_derivs,
    Bitset_base<V, A> &arg_derivs)
{
    // check whether the function always requests derivatives for some arguments
    unsigned deriv_mask = def->get_parameter_derivable_mask();
    if (deriv_mask != 0) {
        mi::mdl::IType_function const *f_tp = cast<mi::mdl::IType_function>(def->get_type());

        arg_derivs.set_bit(0);  // mark as any derivatives requested
        // mark all parameters according to the bit mask
        for (size_t i = 0, n = f_tp->get_parameter_count(); i < n; ++i) {
            if ((deriv_mask & unsigned(1 << i)) != 0) {
                arg_derivs.set_bit(i + 1);
            }
        }
        return;
    }

    // nothing to do, if no derivatives should be returned
    if (!return_derivs)
        return;

    IDefinition::Semantics sema = def->get_semantics();

    switch (sema) {
    case IDefinition::DS_CONV_CONSTRUCTOR:
    case IDefinition::DS_CONV_OPERATOR:
        {
            mi::mdl::IType_function const *f_tp   = cast<mi::mdl::IType_function>(def->get_type());
            mi::mdl::IType const          *ret_tp = f_tp->get_return_type();
            mi::mdl::IType const          *arg_tp;
            mi::mdl::ISymbol const        *sym;
            f_tp->get_parameter(0, arg_tp, sym);

            // only calculate argument derivatives for conversion from float-based
            // to float-based, otherwise they will be zero anyways
            if (is_floating_point_based_type(ret_tp) && is_floating_point_based_type(arg_tp)) {
                arg_derivs.set_bits();
                return;
            }

            return;
        }

    case IDefinition::DS_COPY_CONSTRUCTOR:
    case IDefinition::DS_ELEM_CONSTRUCTOR:
    case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
    case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
    case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
        // when derivatives should be returned, all arguments should provide them
        arg_derivs.set_bits();
        return;

    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        // when derivatives should be returned, the point/vector/normal/scale argument
        // should provide them
        arg_derivs.set_bit(0);  // mark as any derivatives requested
        arg_derivs.set_bit(3);
        return;

    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4:
    case IDefinition::DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR:
        // when derivatives should be returned, the default value should provide them, too
        arg_derivs.set_bit(0);  // mark as any derivatives requested
        arg_derivs.set_bit(2);
        return;

    case IDefinition::DS_UNKNOWN:
        MDL_ASSERT(!"Should not be called with unknown semantics");
        return;

    default:
        break;
    }

    mi::mdl::IType_function const *f_tp   = cast<mi::mdl::IType_function>(def->get_type());
    mi::mdl::IType const          *ret_tp = f_tp->get_return_type();

    // for float or double based math functions, return_derivs also applies to all arguments
    if (IDefinition::DS_INTRINSIC_MATH_FIRST <= sema &&
            sema <= IDefinition::DS_INTRINSIC_MATH_LAST &&
            is_floating_point_based_type(ret_tp)) {

        if (!is_math_deriv_args_supported(sema))
            return;
        arg_derivs.set_bits();
        return;
    }

    // special handling of operators
    if (IDefinition::DS_OP_BASE <= sema && sema <= IDefinition::DS_OP_END) {
        // for float or double based operators, return_derivs also applies to arguments
        if (is_floating_point_based_type(ret_tp)) {
            switch (IExpression::Operator(sema - IDefinition::DS_OP_BASE)) {
            case IExpression::OK_SEQUENCE:
                // return_derivs only applies to the last argument
                arg_derivs.set_bit(0);  // mark as any derivatives requested
                arg_derivs.set_bit(f_tp->get_parameter_count());
                return;

            case IExpression::OK_SELECT:
            case IExpression::OK_ARRAY_INDEX:
                // return_derivs only applies to the compound value
                arg_derivs.set_bit(0);  // mark as any derivatives requested
                arg_derivs.set_bit(1);
                return;

            default:
                // apply it to all arguments
                arg_derivs.set_bits();
                return;
            }
        }
    }
}

// ------------------------------- Derivative_infos class -------------------------------

// Retrieve derivative infos for a function instance.
Func_deriv_info const *Derivative_infos::get_function_derivative_infos(
    Function_instance const &func_inst) const
{
    Deriv_func_inst_map::const_iterator it = m_deriv_func_inst_map.find(func_inst);
    if (it != m_deriv_func_inst_map.end()) {
        return it->second;
    }

    // Unknown functions for which derivatives are needed have to be processed already,
    // only intrinsics are supported by this const function.
    if (func_inst.get_def()->get_semantics() == IDefinition::DS_UNKNOWN) {
        return NULL;
    }

    // Handle known functions
    IDefinition const *def = func_inst.get_def();
    IDeclaration_function const *func_decl =
        cast<IDeclaration_function>(def->get_declaration());

    Func_deriv_info *info = alloc_function_derivative_infos(func_decl->get_parameter_count());
    set_known_function_argument_derivs(
        def,
        func_inst.get_return_derivs(),
        info->args_want_derivatives);
    m_deriv_func_inst_map.insert(std::make_pair(func_inst, info));
    return info;
}

// Retrieve derivative infos for a function instance or calculate it, if it is not available, yet.
Func_deriv_info *Derivative_infos::get_or_calc_function_derivative_infos(
    Module const *module,
    Function_instance const &func_inst)
{
    Deriv_func_inst_map::iterator it = m_deriv_func_inst_map.find(func_inst);
    if (it != m_deriv_func_inst_map.end()) {
        return it->second;
    }

    Func_deriv_info *info;

    // User-defined function? -> analyze it
    if (func_inst.get_def()->get_semantics() == IDefinition::DS_UNKNOWN) {
        Function_processor func_proc(m_alloc, *this, module);
        info = func_proc.process(func_inst);
    } else {
        // Handle known functions
        IDefinition const *def = func_inst.get_def();
        IDeclaration_function const *func_decl =
            cast<IDeclaration_function>(def->get_declaration());

        info = alloc_function_derivative_infos(func_decl->get_parameter_count());
        set_known_function_argument_derivs(
            def,
            func_inst.get_return_derivs(),
            info->args_want_derivatives);
    }

    m_deriv_func_inst_map.insert(std::make_pair(func_inst, info));
    return info;
}

// Return true if the given DAG node should calculate derivatives.
bool Derivative_infos::should_calc_derivatives(DAG_node const *node) const
{
    Deriv_info_dag_map::const_iterator it = m_deriv_info_dag_map.find(node);
    if (it == m_deriv_info_dag_map.end())
        return false;
    return it->second;
}

// Mark the given DAG node as a node for which derivatives should be calculated.
void Derivative_infos::mark_calc_derivatives(DAG_node const *node)
{
    m_deriv_info_dag_map[node] = true;
}

// Determine for which arguments of a call derivatives are needed.
Bitset Derivative_infos::call_wants_arg_derivatives(DAG_call const *call, bool want_derivatives)
{
    Bitset arg_derivs(m_alloc, size_t(call->get_argument_count() + 1));

    // Handle DAG intrinsics first
    switch (unsigned(call->get_semantic())) {
    case IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
        // want-derivatives only applies to the first argument, if set
        if (want_derivatives) {
            arg_derivs.set_bit(0);  // mark as any derivatives requested
            arg_derivs.set_bit(1);
        }
        return arg_derivs;
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        // want-derivatives applies to all arguments, if set
        if (want_derivatives)
            arg_derivs.set_bits();
        return arg_derivs;
    case IDefinition::Semantics(IDefinition::DS_OP_BASE + IExpression::OK_TERNARY):
        // want-derivatives only applies to second and third argument, if set
        if (want_derivatives) {
            arg_derivs.set_bit(0);  // mark as any derivatives requested
            arg_derivs.set_bit(2);
            arg_derivs.set_bit(3);
        }
        return arg_derivs;
    case IDefinition::Semantics(IDefinition::DS_OP_BASE + IExpression::OK_ARRAY_INDEX):
        // want-derivatives only applies to the first argument, if set
        if (want_derivatives) {
            arg_derivs.set_bit(0);  // mark as any derivatives requested
            arg_derivs.set_bit(1);
        }
        return arg_derivs;
    default:
        break;
    }

    char const *signature = call->get_name();
    mi::base::Handle<IModule const> mod(m_resolver->get_owner_module(signature));
    if (!mod) return arg_derivs;  // unknown -> no derivs requested

    Module const *module = impl_cast<Module>(mod.get());
    IDefinition const *def = module->find_signature(signature, /*only_exported=*/ false);
    if (def == NULL) return arg_derivs;  // unknown -> no derivs requested

    def = skip_presets(def, mod);

    if (def->get_kind() == IDefinition::DK_FUNCTION) {
        // handle function calls
        Function_instance::Array_instances arr_inst(m_alloc);
        Function_instance func_inst(def, arr_inst, want_derivatives);
        Func_deriv_info *infos = get_or_calc_function_derivative_infos(module, func_inst);
        arg_derivs.copy_data(infos->args_want_derivatives);
        return arg_derivs;
    } else {
        // handle calls to known constructors and operators
        set_known_function_argument_derivs(def, want_derivatives, arg_derivs);
        return arg_derivs;
    }
}

// Allocate a Func_deriv_info object owned by this Derivative_infos object.
Func_deriv_info *Derivative_infos::alloc_function_derivative_infos(size_t num_params) const
{
    return m_arena_builder.create<Func_deriv_info>(
        const_cast<Memory_arena *>(&m_arena), num_params);
}


// ------------------------------- Deriv_DAG_builder helper -------------------------------

// Constructor.
Deriv_DAG_builder::Deriv_DAG_builder(
    IAllocator *alloc,
    Lambda_function &lambda,
    Derivative_infos &deriv_infos)
: m_alloc(alloc)
, m_lambda(lambda)
, m_vf(*lambda.get_value_factory())
, m_tf(*lambda.get_type_factory())
, m_deriv_infos(deriv_infos)
, m_deriv_type_map(alloc)
, m_result_cache(alloc)
, m_name_id(0)
{
    Symbol_table *sym_table = m_tf.get_symbol_table();
    sym_val = sym_table->create_symbol("val");
    sym_dx = sym_table->create_symbol("dx");
    sym_dy = sym_table->create_symbol("dy");
}

// Gets or creates an MDL derivative type for a given type.
IType_struct const *Deriv_DAG_builder::get_deriv_type(IType const *type)
{
    Deriv_type_map::const_iterator it = m_deriv_type_map.find(type);
    if (it != m_deriv_type_map.end()) return it->second;

    string name("#deriv_", m_alloc);

    switch (type->get_kind()) {
    case IType::TK_STRUCT:
        name += cast<IType_struct>(type)->get_symbol()->get_name();
        break;
    case IType::TK_VECTOR:
        {
            char buf[32];
            IType_vector const *type_vector = cast<IType_vector>(type);
            snprintf(buf, sizeof(buf), "v%u_%d",
                unsigned(type_vector->get_element_type()->get_kind()),
                type_vector->get_compound_size());
            name += buf;
        }
        break;
    default:
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "%u_%u", unsigned(type->get_kind()), m_name_id++);
            name += buf;
        }
        break;
    }

    Symbol_table *sym_table = m_tf.get_symbol_table();
    ISymbol const *sym = sym_table->get_user_type_symbol(name.c_str());
    IType_struct *deriv_type = m_tf.create_struct(sym);
    deriv_type->add_field(type, sym_val);
    deriv_type->add_field(type, sym_dx);
    deriv_type->add_field(type, sym_dy);

    m_deriv_type_map[type] = deriv_type;
    return deriv_type;
}

// Creates a zero value for use a dx or dy component of a dual.
IValue const *Deriv_DAG_builder::create_dual_comp_zero(IType const *type)
{
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        return create_dual_comp_zero(type->skip_type_alias());

    case IType::TK_ENUM:
        return m_vf.create_enum(cast<IType_enum>(type), 0);

    case IType::TK_STRING:
        return m_vf.create_string("");

    case IType::TK_ARRAY:
    case IType::TK_STRUCT:
        {
            IType_compound const *tp = cast<IType_compound>(type);

            size_t count = tp->get_compound_size();
            VLA<IValue const *> values(m_alloc, count);
            for (size_t i = 0; i < count; ++i) {
                values[i] = create_dual_comp_zero(tp->get_compound_type(i));
            }
            return m_vf.create_compound(tp, values.data(), count);
        }

    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_COLOR:
        return m_vf.create_zero(type);

    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_FUNCTION:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        MDL_ASSERT(!"type not allowed as dual component");
        return NULL;
    }

    MDL_ASSERT(!"unsupported type");
    return NULL;
}

// Rebuild an expression with derivative information applied.
DAG_node const *Deriv_DAG_builder::rebuild(DAG_node const *expr, bool want_derivatives)
{
    DAG_node const *cache_key = reinterpret_cast<DAG_node const *>(
        reinterpret_cast<std::uintptr_t>(expr) | (want_derivatives ? 1 : 0));
    auto it = m_result_cache.find(cache_key);
    if (it != m_result_cache.end())
        return it->second;

    DAG_node const *res;
    switch (expr->get_kind()) {
    case DAG_node::EK_TEMPORARY:
        {
            // should not happen, but we can handle it
            DAG_temporary const *t = cast<DAG_temporary>(expr);
            expr = t->get_expr();
            res = rebuild(expr, want_derivatives);
        }
        break;

    case DAG_node::EK_CONSTANT:
        {
            if (!want_derivatives)
                return expr;

            // create (val, 0, 0) constant
            DAG_constant const *dag_const = cast<DAG_constant>(expr);
            IValue const *expr_val = m_vf.import(dag_const->get_value());
            IValue const *null_val = create_dual_comp_zero(expr_val->get_type());

            IType_struct const *deriv_type = get_deriv_type(expr->get_type());
            IValue const *vals[3] = { expr_val, null_val, null_val };
            IValue const *res_val = m_vf.create_struct(deriv_type, vals, 3);
            res = m_lambda.create_constant(res_val);
        }
        break;

    case DAG_node::EK_PARAMETER:
        {
            if (!want_derivatives)
                return expr;

            // a parameter can never provide derivatives, so wrap it in a make_deriv() call
            DAG_call::Call_argument wrap_args[1] = {
                DAG_call::Call_argument(expr, "value") };
            res = m_lambda.create_call(
                "make_deriv()",
                IDefinition::DS_INTRINSIC_DAG_MAKE_DERIV,
                wrap_args, 1,
                get_deriv_type(expr->get_type()));
        }
        break;

    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(expr);
            Bitset want_arg_derivs(
                m_deriv_infos.call_wants_arg_derivatives(call, want_derivatives));

            string call_name(m_alloc);
            if (want_derivatives)
                call_name += '#';
            call_name += call->get_name();

            int n_args = call->get_argument_count();
            Small_VLA<DAG_call::Call_argument, 8> args(m_alloc, size_t(n_args));
            for (int i = 0; i < n_args; ++i) {
                DAG_node const *arg = call->get_argument(i);

                bool calc_arg_derivs = want_arg_derivs.test_bit(i + 1);

                args[i].param_name = call->get_parameter_name(i);
                args[i].arg = rebuild(arg, calc_arg_derivs);
            }

            IDefinition::Semantics sema = call->get_semantic();
            res = m_lambda.create_call(
                call_name.c_str(),
                sema,
                args.data(), n_args,
                want_derivatives ? get_deriv_type(call->get_type()) : call->get_type());

            // function always returns derivatives, but we don't want them?
            if (!want_derivatives &&
                (
                    sema == IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE ||
                    sema == IDefinition::DS_INTRINSIC_STATE_POSITION
                ))
            {
                // wrap it in a get_deriv_val() call
                DAG_call::Call_argument wrap_args[1] = {
                    DAG_call::Call_argument(res, "deriv") };
                res = m_lambda.create_call(
                    "get_deriv_val()",
                    IDefinition::DS_INTRINSIC_DAG_GET_DERIV_VALUE,
                    wrap_args, 1,
                    call->get_type());
            }
        }
        break;
    default:
        MDL_ASSERT(!"Invalid DAG node kind");
        return NULL;
    }

    if (want_derivatives)
        m_deriv_infos.mark_calc_derivatives(res);

    m_result_cache[cache_key] = res;

    return res;
}

} // mdl
} // mi
