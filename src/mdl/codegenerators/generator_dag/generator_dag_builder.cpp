/******************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl/compiler/compilercore/compilercore_file_resolution.h"

#include "generator_dag_builder.h"
#include "generator_dag_tools.h"

#include <cstdio>

namespace mi {
namespace mdl {

namespace {

/// A resource modifier that uses the resource table of a code_dag.
class Restable_resource_modifer : public IResource_modifier
{
public:
    /// Modify a resource.
    ///
    /// \param res    the resource value to be modified
    ///
    /// \return a new value of the same type or res if no modification is necessary
    IValue_resource const *modify(
        IValue_resource const *res,
        IModule         const *owner,
        IValue_factory        &vf) MDL_FINAL
    {
        int old_tag = res->get_tag_value();
        if (old_tag == 0) {
            int res_tag = m_dag.find_resource_tag(res);
            if (res_tag != old_tag) {
                res = clone_resource(vf, res, res_tag);
            }
        }
        return m_chain.modify(res, owner, vf);
    }

    /// Constructor.
    Restable_resource_modifer(
        Generated_code_dag const &dag,
        IResource_modifier       &chain)
    : m_dag(dag)
    , m_chain(chain)
    {
    }

private:
    /// Clone a resource but insert a new tag.
    IValue_resource const *clone_resource(
        IValue_factory        &vf,
        IValue_resource const *r,
        int                   tag)
    {
        switch (r->get_kind()) {
        case IValue::VK_TEXTURE:
            {
                IValue_texture const *tex = cast<IValue_texture>(r);
                if (tex->get_type()->get_shape() == IType_texture::TS_BSDF_DATA) {
                    return vf.create_bsdf_data_texture(
                        tex->get_bsdf_data_kind(),
                        tag,
                        0);
                }
                return vf.create_texture(
                    tex->get_type(),
                    tex->get_string_value(),
                    tex->get_gamma_mode(),
                    tag,
                    0);
            }
        case IValue::VK_LIGHT_PROFILE:
            return vf.create_light_profile(
                cast<IType_light_profile>(r->get_type()),
                r->get_string_value(),
                tag,
                0);
        case IValue::VK_BSDF_MEASUREMENT:
            return vf.create_bsdf_measurement(
                cast<IType_bsdf_measurement>(r->get_type()),
                r->get_string_value(),
                tag,
                0);
        default:
            MDL_ASSERT(!"Unsupported resource kind");
            return r;
        }
    }

private:
    Generated_code_dag const &m_dag;
    IResource_modifier       &m_chain;
};

/// Check if the given expression is a "simple" select.
///
/// \param expr  the expression to check
static bool is_simple_select(IExpression const *expr)
{
    if (!is<IExpression_binary>(expr))
        return false;
    IExpression_binary const *b_expr = cast<IExpression_binary>(expr);
    if (b_expr->get_operator() != IExpression_binary::OK_SELECT)
        return false;

    IExpression const *left  = b_expr->get_left_argument();
    if (!is<IExpression_reference>(left))
        return false;

    MDL_ASSERT(is<IExpression_reference>(b_expr->get_right_argument()));

    IType const *l_tp = left->get_type()->skip_type_alias();

    // so far, we support structs and vectors only
    IType::Kind kind = l_tp->get_kind();
    return kind == IType::TK_STRUCT || kind == IType::TK_VECTOR;
}

/// Creates a one constant for the given type to be used in a pre/post inc/decrement operation.
///
/// \param type  the type, might be a int, float, double, or vector of those
static IValue const *create_pre_post_one(
    IType const    *type,
    IValue_factory *vf)
{
    switch (type->get_kind()) {
    case IType::TK_INT:
        return vf->create_int(1);
    case IType::TK_FLOAT:
        return vf->create_float(1.0f);
    case IType::TK_DOUBLE:
        return vf->create_double(1.0);
    default:
        // should not happen
        MDL_ASSERT(!"pre/post-increment type neither int nor float/double");
        return vf->create_bad();
    }
}

/// Helper class to capture a constant fold exception.
class Variable_lookup_handler : public IConst_fold_handler
{
public:
    /// Handle constant folding exceptions.
    void exception(
        Reason            r,
        IExpression const *expr,
        int               index = 0,
        int               length = 0) MDL_FINAL
    {
        m_error_state = true;
    }

    /// Handle variable lookup.
    IValue const *lookup(IDefinition const *var) MDL_FINAL
    {
        if (m_iter_var == var) return m_iter_value;

        DAG_builder::Definition_temporary_map::const_iterator it = m_tmp_value_map.find(var);
        if (it != m_tmp_value_map.end()) {
            DAG_node const *node = it->second;
            if (node->get_kind() == DAG_node::EK_CONSTANT)
                return m_value_factory->import(as<DAG_constant>(node)->get_value());
        }
        return m_value_factory->create_bad();
    }

    /// Check whether evaluate_intrinsic_function should be called for an unhandled
    /// intrinsic functions with the given semantic.
    bool is_evaluate_intrinsic_function_enabled(
        IDefinition::Semantics semantic) const MDL_FINAL
    {
        if (m_call_evaluator == NULL)
            return false;

        return m_call_evaluator->is_evaluate_intrinsic_function_enabled(semantic);
    }

    /// Handle intrinsic call evaluation.
    IValue const *evaluate_intrinsic_function(
        IDefinition::Semantics semantic,
        const IValue *const arguments[],
        size_t n_arguments) MDL_FINAL
    {
        if (m_call_evaluator == NULL)
            return m_value_factory->create_bad();

        return m_call_evaluator->evaluate_intrinsic_function(
            m_value_factory, semantic, arguments, n_arguments);
    }


    /// Constructor.
    explicit Variable_lookup_handler(
        IModule const *module,
        IValue_factory *value_factory,
        ICall_evaluator const *call_evaluator,
        DAG_builder::Definition_temporary_map &tmp_value_map)
    : m_module(module)
    , m_value_factory(value_factory)
    , m_call_evaluator(call_evaluator)
    , m_tmp_value_map(tmp_value_map)
    , m_error_state(false)
    , m_iter_var(NULL)
    , m_iter_value(NULL)
    {
    }

    /// Clear the captures error state.
    void clear_error_state() { m_error_state = false; }

    /// Returns true if an error occurred since the last clear_error_state() call.
    bool has_error() const { return m_error_state; }

    /// Get the iteration variable used for loop unrolling.
    IDefinition const *get_iteration_variable() const { return m_iter_var; }

    /// Get the iteration variable value used for loop unrolling.
    IValue const *get_iteration_value() const { return m_iter_value; }

    /// Set the iteration variable for loop unrolling.
    ///
    /// \param var    the definition of the iteration variable
    void set_iteration_variable(IDefinition const *var)
    {
        m_iter_var = var;
    }

    /// Set the value of the iteration variable for loop unrolling.
    ///
    /// \param value  the value of the iteration variable
    void set_iteration_variable_value(IValue const *value)
    {
        m_iter_value = value;
    }

    /// Try to update the iteration variable with the given expression.
    ///
    /// \param update_expr  the expression meant to update the iteration variable
    bool update_iteration_variable(IExpression const *update_expr);

private:
    /// The module owning the folding expressions.
    IModule const *m_module;

    /// The value factory to create new values.
    IValue_factory *m_value_factory;

    /// The call evaluator.
    ICall_evaluator const *m_call_evaluator;

    /// The map from definitions to temporary indices.
    DAG_builder::Definition_temporary_map &m_tmp_value_map;

    /// Set to true once an exception occurs.
    bool     m_error_state;

    IDefinition const *m_iter_var;
    IValue const *m_iter_value;
};

// Try to update the iteration variable with the given expression.
bool Variable_lookup_handler::update_iteration_variable(IExpression const *update_expr)
{
    IValue const *new_val = NULL;

    if (IExpression_binary const *bin_expr = as<IExpression_binary>(update_expr)) {
        // make sure lhs points to a reference to the iteration variable
        IExpression_reference const *lhs = as<IExpression_reference>(bin_expr->get_left_argument());
        if (lhs == NULL)
            return false;
        if (lhs->get_definition() != m_iter_var)
            return false;

        // const fold rhs
        IValue const *rhs = bin_expr->get_right_argument()->fold(m_module, m_value_factory, this);
        if (is<IValue_bad>(rhs))
            return false;

        switch (bin_expr->get_operator()) {
        case IExpression_binary::OK_SELECT:
        case IExpression_binary::OK_ARRAY_INDEX:
        case IExpression_binary::OK_MULTIPLY:
        case IExpression_binary::OK_DIVIDE:
        case IExpression_binary::OK_MODULO:
        case IExpression_binary::OK_PLUS:
        case IExpression_binary::OK_MINUS:
        case IExpression_binary::OK_SHIFT_LEFT:
        case IExpression_binary::OK_SHIFT_RIGHT:
        case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
        case IExpression_binary::OK_LESS:
        case IExpression_binary::OK_LESS_OR_EQUAL:
        case IExpression_binary::OK_GREATER_OR_EQUAL:
        case IExpression_binary::OK_GREATER:
        case IExpression_binary::OK_EQUAL:
        case IExpression_binary::OK_NOT_EQUAL:
        case IExpression_binary::OK_BITWISE_AND:
        case IExpression_binary::OK_BITWISE_XOR:
        case IExpression_binary::OK_BITWISE_OR:
        case IExpression_binary::OK_LOGICAL_AND:
        case IExpression_binary::OK_LOGICAL_OR:
        case IExpression_binary::OK_SEQUENCE:
            return false;

        case IExpression_binary::OK_ASSIGN:
            new_val = m_value_factory->import(rhs);
            break;

        case IExpression_binary::OK_MULTIPLY_ASSIGN:
            new_val = m_iter_value->multiply(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_DIVIDE_ASSIGN:
            new_val = m_iter_value->divide(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_MODULO_ASSIGN:
            new_val = m_iter_value->modulo(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_PLUS_ASSIGN:
            new_val = m_iter_value->add(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_MINUS_ASSIGN:
            new_val = m_iter_value->sub(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
            new_val = m_iter_value->shl(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
            new_val = m_iter_value->asr(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
            new_val = m_iter_value->lsr(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_BITWISE_AND_ASSIGN:
            new_val = m_iter_value->bitwise_and(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
            new_val = m_iter_value->bitwise_xor(m_value_factory, rhs);
            break;

        case IExpression_binary::OK_BITWISE_OR_ASSIGN:
            new_val = m_iter_value->bitwise_or(m_value_factory, rhs);
            break;
        }
    } else if (IExpression_unary const *un_expr = as<IExpression_unary>(update_expr)) {
        // make sure the expression points to a reference to the iteration variable
        IExpression_reference const *lhs = as<IExpression_reference>(un_expr->get_argument());
        if (lhs == NULL)
            return false;
        if (lhs->get_definition() != m_iter_var)
            return false;

        switch (un_expr->get_operator()) {
        case IExpression_unary::OK_BITWISE_COMPLEMENT:
        case IExpression_unary::OK_LOGICAL_NOT:
        case IExpression_unary::OK_POSITIVE:
        case IExpression_unary::OK_NEGATIVE:
        case IExpression_unary::OK_CAST:
            return false;

        case IExpression_unary::OK_PRE_INCREMENT:
        case IExpression_unary::OK_POST_INCREMENT:
            new_val = m_iter_value->add(
                m_value_factory, create_pre_post_one(m_iter_value->get_type(), m_value_factory));
            break;
        case IExpression_unary::OK_PRE_DECREMENT:
        case IExpression_unary::OK_POST_DECREMENT:
            new_val = m_iter_value->sub(
                m_value_factory, create_pre_post_one(m_iter_value->get_type(), m_value_factory));
            break;
        }
    } else {
        return false;
    }

    if (new_val == NULL || is<IValue_bad>(new_val))
        return false;

    m_iter_value = new_val;
    return true;
}


/// A helper class to check if inlining is possible.
class Inline_checker {
public:
    typedef vector<IType const *>::Type Type_vector;

    /// Constructor.
    ///
    /// \param module              the module used for folding values
    /// \param value_factory       the value factory used to create new values
    /// \param call_evaluator      the call evaluator for handling some intrinsics
    /// \param func_decl           a function represented by its declaration
    /// \param call                a call-site of this function
    /// \param forbid_local_calls  if set, calls to local functions are forbidden
    /// \param tmp_value_map       map of argument values
    Inline_checker(
        IModule const                          *module,
        IValue_factory                         *value_factory,
        ICall_evaluator const                  *call_evaluator,
        IDeclaration_function const            *func_decl,
        IExpression_call const                 *call,
        bool                                   forbid_local_calls,
        DAG_builder::Definition_temporary_map  &tmp_value_map)
    : m_module(module)
    , m_value_factory(value_factory)
    , m_func_decl(func_decl)
    , m_call(call)
    , m_arg_types(NULL)
    , m_forbid_local_calls(forbid_local_calls)
    , m_var_lookup_handler(module, value_factory, call_evaluator, tmp_value_map)
    , m_skip_flags(INL_NO_SKIP)
    {
    }

    /// Constructor.
    ///
    /// \param module              the module used for folding values
    /// \param value_factory       the value factory used to create new values
    /// \param call_evaluator      the call evaluator for handling some intrinsics
    /// \param func_decl           a function represented by its declaration
    /// \param arg_types           list of argument types of the call site
    /// \param forbid_local_calls  if set, calls to local functions are forbidden
    /// \param tmp_value_map       map of argument values
    Inline_checker(
        IModule const                          *module,
        IValue_factory                         *value_factory,
        ICall_evaluator const                  *call_evaluator,
        IDeclaration_function const            *func_decl,
        Type_vector const                      &arg_types,
        bool                                   forbid_local_calls,
        DAG_builder::Definition_temporary_map  &tmp_value_map)
    : m_module(module)
    , m_value_factory(value_factory)
    , m_func_decl(func_decl)
    , m_call(NULL)
    , m_arg_types(&arg_types)
    , m_forbid_local_calls(forbid_local_calls)
    , m_var_lookup_handler(module, value_factory, call_evaluator, tmp_value_map)
    , m_skip_flags(INL_NO_SKIP)
    {
    }

    /// Check if we support inlining the current function at the given call-site
    bool can_inline();

private:
    /// Check if we support inlining of the given expression.
    ///
    /// \param expr  the expression to check
    bool can_inline(IExpression const *expr);

    /// Check if we can inline the given statement.
    ///
    /// \param stmt  the statement to check
    bool can_inline(IStatement const *stmt);

    /// Check if the given expression is uniform.
    ///
    /// \param expr  the expression to check
    bool is_uniform(IExpression const *expr);

    /// Check if the given call argument is uniform.
    ///
    /// \param param  the definition of the parameter
    bool is_uniform_argument(IDefinition const *param);

    /// Check if a call has a side effect in MDL.
    ///
    /// \param call  the call to check
    bool has_side_effect(IExpression_call const *call);

    /// Check, if the given call is allowed AFTER inlining.
    ///
    /// \param call  the call
    bool allow_call(IExpression_call const *call);

    /// Check whether we support assigning to the given target during inlining.
    ///
    /// \param expr  the target expression
    bool is_allowed_assign_target(IExpression const *expr);

private:
    /// The module used for folding values.
    IModule const *m_module;

    /// The value factory to be used to create new values.
    IValue_factory *m_value_factory;

    /// The declaration of the function to inline.
    IDeclaration_function const *m_func_decl;

    /// The call site that is inlined or NULL.
    IExpression_call const *m_call;

    /// The call site argument types or NULL.
    Type_vector const      *m_arg_types;

    /// If set, calls to local functions are forbidden.
    bool m_forbid_local_calls;

    /// Handler for looking up variables during constant folding.
    Variable_lookup_handler m_var_lookup_handler;


    enum Inline_skip_flag
    {
        INL_NO_SKIP = 0,
        INL_SKIP_BREAK = 1,
        INL_SKIP_RETURN = 2
    };

    /// Specifies whether and how instructions should be skipped.
    Inline_skip_flag m_skip_flags;
};

} // anonymous

 // Check whether we support assigning to the given target during inlining.
bool Inline_checker::is_allowed_assign_target(IExpression const *expr)
{
    if (IExpression_reference const *ref = as<IExpression_reference>(expr)) {
        return ref->get_definition() != m_var_lookup_handler.get_iteration_variable();
    }
    return is_simple_select(expr);
}

// Check if we support inlining of the given expression.
bool Inline_checker::can_inline(IExpression const *expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *u_expr = cast<IExpression_unary>(expr);

            // currently, we cannot support ++, --
            switch (u_expr->get_operator()) {
            case IExpression_unary::OK_PRE_INCREMENT:
            case IExpression_unary::OK_PRE_DECREMENT:
            case IExpression_unary::OK_POST_INCREMENT:
            case IExpression_unary::OK_POST_DECREMENT:
                {
                    // currently, we support assign to locals only
                    IExpression const *arg = u_expr->get_argument();
                    return is_allowed_assign_target(arg);
                }
            default:
                return true;
            }
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *b_expr = cast<IExpression_binary>(expr);

            // currently, we cannot support &&, and || because these have lazy evaluation
            switch (b_expr->get_operator()) {
            case IExpression_binary::OK_LOGICAL_AND:
            case IExpression_binary::OK_LOGICAL_OR:
                return false;
            case IExpression_binary::OK_ASSIGN:
            case IExpression_binary::OK_MULTIPLY_ASSIGN:
            case IExpression_binary::OK_DIVIDE_ASSIGN:
            case IExpression_binary::OK_MODULO_ASSIGN:
            case IExpression_binary::OK_PLUS_ASSIGN:
            case IExpression_binary::OK_MINUS_ASSIGN:
            case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
            case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
            case IExpression_binary::OK_BITWISE_AND_ASSIGN:
            case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
            case IExpression_binary::OK_BITWISE_OR_ASSIGN:
                {
                    // currently, we support assign to locals only
                    IExpression const *left = b_expr->get_left_argument();
                    return is_allowed_assign_target(left);
                }
            default:
                return true;
            }
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *ternary = cast<IExpression_conditional>(expr);

            // Inlining ?: isn't always good, because it has lazy evaluation
            IExpression const *cond = ternary->get_condition();
            if (!is_uniform(cond) || !can_inline(cond))
                return false;

            if (!can_inline(ternary->get_true()))
                return false;
            if (!can_inline(ternary->get_false()))
                return false;
            return true;
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *call = cast<IExpression_call>(expr);
            return !has_side_effect(call) && allow_call(call);
        }
    default:
        return true;
    }
}

// Check if the given expression is uniform.
bool Inline_checker::is_uniform(IExpression const *expr)
{
    IType const *type = expr->get_type();
    IType::Modifiers mods = type->get_type_modifiers();

    if (mods & IType::MK_VARYING)
        return false;
    if ((mods & (IType::MK_UNIFORM | IType::MK_CONST)) != 0)
        return true;

    // else auto, track until arguments are found
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        // should not happen
        return false;
    case IExpression::EK_LITERAL:
        return true;
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *ref = cast<IExpression_reference>(expr);
            IDefinition const *def = ref->get_definition();
            MDL_ASSERT(def != NULL && "Should always have a definitin here");
            if (def->get_kind() == IDefinition::DK_PARAMETER) {
                // lookup the argument type
                return is_uniform_argument(def);
            }
            return false;
        }
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *unary = cast<IExpression_unary>(expr);
            return is_uniform(unary->get_argument());
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *binary = cast<IExpression_binary>(expr);
            switch (binary->get_operator()) {
            case IExpression_binary::OK_SELECT:
                // if we are here, the selected field is NOT uniform itself, so check
                // the lhs
                return is_uniform(binary->get_left_argument());
            case IExpression_binary::OK_ARRAY_INDEX:
            case IExpression_binary::OK_MULTIPLY:
            case IExpression_binary::OK_DIVIDE:
            case IExpression_binary::OK_MODULO:
            case IExpression_binary::OK_PLUS:
            case IExpression_binary::OK_MINUS:
            case IExpression_binary::OK_SHIFT_LEFT:
            case IExpression_binary::OK_SHIFT_RIGHT:
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
            case IExpression_binary::OK_LESS:
            case IExpression_binary::OK_LESS_OR_EQUAL:
            case IExpression_binary::OK_GREATER_OR_EQUAL:
            case IExpression_binary::OK_GREATER:
            case IExpression_binary::OK_EQUAL:
            case IExpression_binary::OK_NOT_EQUAL:
            case IExpression_binary::OK_BITWISE_AND:
            case IExpression_binary::OK_BITWISE_XOR:
            case IExpression_binary::OK_BITWISE_OR:
            case IExpression_binary::OK_LOGICAL_AND:
            case IExpression_binary::OK_LOGICAL_OR:
                // for those, both must be uniform
                return is_uniform(binary->get_left_argument()) &&
                    is_uniform(binary->get_right_argument());
            default:
                // assignments not yet supported
                return false;
            }
        }
    case IExpression::EK_CONDITIONAL:
    case IExpression::EK_CALL:
    case IExpression::EK_LET:
        // not yet supported
        return false;
    }
    return false;
}

// Check if the given call argument is uniform.
bool Inline_checker::is_uniform_argument(IDefinition const *param)
{
    MDL_ASSERT(param->get_kind() == IDefinition::DK_PARAMETER);
    int idx = param->get_parameter_index();

    if (m_call != NULL) {
        IArgument const   *arg  = m_call->get_argument(idx);
        IExpression const *expr = arg->get_argument_expr();
        IType const       *type = expr->get_type();

        return (type->get_type_modifiers() & (IType::MK_CONST | IType::MK_UNIFORM)) != 0;
    } else {
        IType const *type = (*m_arg_types)[idx];

        return (type->get_type_modifiers() & (IType::MK_CONST | IType::MK_UNIFORM)) != 0;
    }
}

// Check if a call has a side effect in MDL.
bool Inline_checker::has_side_effect(IExpression_call const *call)
{
    IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());

    if (ref->is_array_constructor())
        return false;

    IDefinition const *def = ref->get_definition();

    // in MDL only functions that uses calls to the ::debug module have a "side-effect"
    return def->get_property(IDefinition::DP_CONTAINS_DEBUG);
}

//  Check, if the given call is allowed AFTER inlining.
bool Inline_checker::allow_call(IExpression_call const *call)
{
    if (!m_forbid_local_calls)
        return true;

    IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());

    if (ref->is_array_constructor())
        return true;

    IDefinition const *def = ref->get_definition();

    // This IS very restrictive, because for local functions we could (recursively)
    // check if it is possible to inline them.
    // However, the "forbid_local_call" mode is not used currently, so no sophisticated code
    // is necessary.
    return !def->get_property(IDefinition::DP_IS_LOCAL_FUNCTION);
}

// Check if we can inline the given statement.
bool Inline_checker::can_inline(IStatement const *stmt)
{
    if (m_skip_flags != INL_NO_SKIP)
        return true;

    // check statements, for now only simple expressions
    switch (stmt->get_kind()) {
    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression const *e_stmt = cast<IStatement_expression>(stmt);
            IExpression           const *expr   = e_stmt->get_expression();

            return expr == NULL || can_inline(expr);
        }
    case IStatement::SK_DECLARATION:
        {
            IDeclaration const *decl = cast<IStatement_declaration>(stmt)->get_declaration();

            switch (decl->get_kind()) {
            case IDeclaration::DK_TYPE_ALIAS:
            case IDeclaration::DK_TYPE_ENUM:
            case IDeclaration::DK_TYPE_STRUCT:
                // local type declaration is fine
                return true;
            case IDeclaration::DK_VARIABLE:
                {
                    IDeclaration_variable const *var_decl = cast<IDeclaration_variable>(decl);

                    for (int i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
                        IExpression const *init = var_decl->get_variable_init(i);

                        if (init != NULL && !can_inline(init))
                            return false;
                    }
                    return true;
                }
            case IDeclaration::DK_CONSTANT:
                {
                    IDeclaration_constant const *const_decl = cast<IDeclaration_constant>(decl);

                    for (int i = 0, n = const_decl->get_constant_count(); i < n; ++i) {
                        IExpression const *init = const_decl->get_constant_exp(i);

                        if (init != NULL && !can_inline(init))
                            return false;
                    }
                    return true;
                }
            default:
                // all others are unsupported
                break;
            }
        }
        return false;
    case IStatement::SK_RETURN:
        {
            IExpression const *expr = cast<IStatement_return>(stmt)->get_expression();
            bool res = can_inline(expr);
            m_skip_flags = Inline_skip_flag(m_skip_flags | INL_SKIP_RETURN);
            return res;
        }
    case IStatement::SK_COMPOUND:
    case IStatement::SK_CASE:
        {
            IStatement_compound const *block = cast<IStatement_compound>(stmt);

            for (int i = 0, n = block->get_statement_count(); i < n; ++i) {
                IStatement const *stmt = block->get_statement(i);

                if (!can_inline(stmt))
                    return false;
                if (m_skip_flags != INL_NO_SKIP) break;
            }
            return true;
        }
    case IStatement::SK_IF:
        {
            IStatement_if const *if_stmt = cast<IStatement_if>(stmt);

            IValue const *cond_result = if_stmt->get_condition()->fold(
                m_module, m_value_factory, &m_var_lookup_handler);
            if (is<IValue_bad>(cond_result))
                return false;

            // true case
            if (cond_result->is_one())
                return can_inline(if_stmt->get_then_statement());
            else if (!cond_result->is_zero())
                return false;

            // false case
            if (if_stmt->get_else_statement() == NULL)
                return true;
            return can_inline(if_stmt->get_else_statement());
        }
    case IStatement::SK_SWITCH:
        {
            IStatement_switch const *switch_stmt = cast<IStatement_switch>(stmt);

            IValue const *cond_result = switch_stmt->get_condition()->fold(
                m_module, m_value_factory, &m_var_lookup_handler);
            if (is<IValue_bad>(cond_result))
                return false;
            IValue_int_valued const *cond_val = cast<IValue_int_valued>(cond_result);
            if (cond_val == NULL) return false;

            int cond_int = cond_val->get_value();

            // find a case for the folded condition value
            size_t num_cases = switch_stmt->get_case_count(), cur_case_idx = num_cases;
            for (size_t i = 0; i < num_cases; ++i) {
                IStatement_case const *case_stmt = cast<IStatement_case>(switch_stmt->get_case(i));

                IExpression const *label = case_stmt->get_label();
                if (label == NULL) {
                    // default case, remember as best match but continue searching
                    cur_case_idx = i;
                    continue;
                }

                IExpression_literal const *lit = cast<IExpression_literal>(label);
                IValue_int_valued const *v = cast<IValue_int_valued>(lit->get_value());
                if (v->get_value() == cond_int) {
                    cur_case_idx = i;
                    break;
                }
            }

            // process (next) matching case
            while (cur_case_idx < num_cases) {
                IStatement_case const *cur_case =
                    cast<IStatement_case>(switch_stmt->get_case(cur_case_idx));

                bool res = can_inline(cur_case);
                if (!res) return false;

                // case inlining results in any skipping?
                if (m_skip_flags != INL_NO_SKIP) {
                    // consume any breaks
                    m_skip_flags = Inline_skip_flag(m_skip_flags & ~INL_SKIP_BREAK);
                    return res;
                }

                // no skipping, so it's a fall-through, move to next case
                ++cur_case_idx;
            }

            // reached end of switch, but it's still inlineable
            return true;
        }
    case IStatement::SK_FOR:
        {
            // We inline for-loops with a limited constant number of iterations by unrolling.
            // For now we apply some restrictions to keep can_inline simple and avoid properly
            // handling variables in this step already.
            //  - the init statement must declare exactly one iteration variable
            //  - the initializer must evaluate to a constant
            //  - the iteration variable may only be modified in the update expression
            //  - the loop condition may only depend on the iteration variable and constant values,
            //    so in each unrolling step it must be foldable to a constant
            //  - no nested loops

            // don't allow nested loops
            if (m_var_lookup_handler.get_iteration_variable() != NULL)
                return false;

            IStatement_for const *for_stmt = cast<IStatement_for>(stmt);
            IStatement const *init_stmt = for_stmt->get_init();
            IStatement_declaration const *decl_stmt = as<IStatement_declaration>(init_stmt);
            if (decl_stmt == NULL)
                return false;

            // fold initializer of the only iteration variable to a constant
            IDeclaration_variable const *decl =
                cast<IDeclaration_variable>(decl_stmt->get_declaration());
            if (decl == NULL || decl->get_variable_count() != 1) return false;
            IValue const *init_val = decl->get_variable_init(0)->fold(
                m_module, m_value_factory, &m_var_lookup_handler);
            if (is<IValue_bad>(init_val))
                return false;

            // set iteration variable in lookup handler
            m_var_lookup_handler.set_iteration_variable(
                decl->get_variable_name(0)->get_definition());
            m_var_lookup_handler.set_iteration_variable_value(init_val);

            // allow up to 4 unroll steps
            IExpression const *cond = for_stmt->get_condition();
            const int max_steps = 4;
            bool res = true;
            int steps;
            for (steps = 0; steps <= max_steps; ++steps) {
                IValue const *cond_result = cond->fold(
                    m_module, m_value_factory, &m_var_lookup_handler);
                if (cond_result->is_zero()) {
                    break;
                } else if (!cond_result->is_one()) {
                    res = false;
                    break;
                }

                res = can_inline(for_stmt->get_body());
                if (!res)
                    break;
                if (m_skip_flags != INL_NO_SKIP) break;

                if (!m_var_lookup_handler.update_iteration_variable(for_stmt->get_update())) {
                    res = false;
                    break;
                }
            }
            // didn't exit loop in time?
            if (steps > max_steps)
                res = false;

            // consume any breaks
            m_skip_flags = Inline_skip_flag(m_skip_flags & ~INL_SKIP_BREAK);

            m_var_lookup_handler.set_iteration_variable(NULL);
            m_var_lookup_handler.set_iteration_variable_value(NULL);
            return res;
        }
    case IStatement::SK_BREAK:
        {
            m_skip_flags = Inline_skip_flag(m_skip_flags | INL_SKIP_BREAK);
            return true;
        }
    default:
        // others not yet supported
        return false;
    }
}

// Check if we support inlining the current function at the given call-site.
bool Inline_checker::can_inline()
{
    if (m_func_decl == NULL) {
        // no declaration: should normally not happen
        return false;
    }

    IStatement const *body = m_func_decl->get_body();
    if (body == NULL) return false;

    bool res = can_inline(body);
    MDL_ASSERT((m_skip_flags & INL_SKIP_BREAK) == 0 && "break was not properly handled");
    return res;
}

// ---------------------------------- DAG_builder ----------------------------------

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

// Constructor.
DAG_builder::DAG_builder(
    IAllocator            *alloc,
    DAG_node_factory_impl &node_factory,
    DAG_mangler           &mangler,
    File_resolver         &resolver)
: m_alloc(alloc)
, m_node_factory(node_factory)
, m_value_factory(*node_factory.get_value_factory())
, m_type_factory(*m_value_factory.get_type_factory())
, m_sym_tab(*m_type_factory.get_symbol_table())
, m_mangler(mangler)
, m_printer(mangler.get_printer())
, m_resolver(resolver)
, m_resource_modifier(&null_modifier)
, m_tmp_value_map(
    0, Definition_temporary_map::hasher(), Definition_temporary_map::key_equal(), alloc)
, m_module_stack(alloc)
, m_module_stack_tos(0)
, m_accesible_parameters(alloc)
, m_skip_flags(INL_NO_SKIP)
, m_inline_return_node(NULL)
, m_error_calls(alloc)
, m_forbid_local_calls(false)
, m_conditional_created(false)
, m_conditional_df_created(false)
{
}

// Set a resource modifier.
void DAG_builder::set_resource_modifier(IResource_modifier *modifier)
{
    if (modifier == NULL)
        modifier = &null_modifier;
    m_resource_modifier = modifier;
}

// Enable/disable local function calls.
bool DAG_builder::forbid_local_function_calls(bool flag)
{
    bool res = m_forbid_local_calls;

    m_forbid_local_calls = flag;
    return res;
}


// Check if the given type is a user defined type.
bool DAG_builder::is_user_type(IType const *type)
{
    if (IType_struct const *s_type = as<IType_struct>(type)) {
        if (s_type->get_predefined_id() == IType_struct::SID_USER) {
            return true;
        }
    } else if (IType_enum const *e_type = as<IType_enum>(type)) {
        IType_enum::Predefined_id id = e_type->get_predefined_id();
        if (id == IType_enum::EID_USER || id == IType_enum::EID_TEX_GAMMA_MODE) {
            // although tex::gamma_mode is predefined in the compiler (due to its use
            // in the texture constructor), it IS a user type: There is even MDL code
            // for it
            return true;
        }
    } else if (is<IType_array>(type))
        return true;
    return false;
}

// Convert a definition to a name.
string DAG_builder::def_to_name(
    IDefinition const *def, const char *module_name, bool with_signature_suffix) const
{
    return m_mangler.mangle(def, module_name, with_signature_suffix);
}

// Convert a definition to a name.
string DAG_builder::def_to_name(IDefinition const *def, IModule const *module, bool with_signature_suffix) const
{
    return m_mangler.mangle(def, module, with_signature_suffix);
}

// Convert a definition to a name.
string DAG_builder::def_to_name(IDefinition const *def) const
{
    return def_to_name(def, tos_module()->get_owner_module_name(def));
}

// Convert a type to a name.
string DAG_builder::type_to_name(IType const *type) const
{
    Name_printer &printer = m_mangler.get_printer();

    printer.print(type);
    return printer.get_line();
}

// Convert a parameter type to a name.
string DAG_builder::parameter_type_to_name(IType const *type) const
{
    return m_mangler.mangle_parameter_type(type);
}

// Clear temporary data to restart code generation.
void DAG_builder::reset()
{
    m_accesible_parameters.clear();
    m_tmp_value_map.clear();
    m_node_factory.clear_node_names();
}

// Make the given function/material parameter accessible.
void DAG_builder::make_accessible(mi::mdl::IDefinition const *p_def)
{
    MDL_ASSERT(p_def != NULL && p_def->get_kind() == IDefinition::DK_PARAMETER);

    m_accesible_parameters.push_back(p_def);
}

// Make the given function/material parameter accessible.
void DAG_builder::make_accessible(mi::mdl::IParameter const *param)
{
    IDefinition const *p_def = param->get_name()->get_definition();
    make_accessible(p_def);
}

// Push a module on the module stack.
void DAG_builder::push_module(IModule const *mod)
{
    if (m_module_stack_tos >= m_module_stack.size())
        m_module_stack.push_back(mi::base::make_handle_dup(mod));
    else
        m_module_stack[m_module_stack_tos] = mi::base::make_handle_dup(mod);
    ++m_module_stack_tos;
}

// Pop a module from the module stack.
void DAG_builder::pop_module()
{
    if (m_module_stack_tos > 0) {
        --m_module_stack_tos;
        m_module_stack[m_module_stack_tos].reset();
    }
}

// Return the top-of-stack module, not retained.
IModule const *DAG_builder::tos_module() const
{
    if (m_module_stack_tos > 0)
        return m_module_stack[m_module_stack_tos - 1].get();
    return NULL;
}

// Convert an unary MDL operator to a name.
const char *DAG_builder::unary_op_to_name(IExpression_unary::Operator op)
{
    switch(op) {
    case IExpression_unary::OK_BITWISE_COMPLEMENT:
        return "operator~";
    case IExpression_unary::OK_LOGICAL_NOT:
        return "operator!";
    case IExpression_unary::OK_POSITIVE:
        return "operator+";
    case IExpression_unary::OK_NEGATIVE:
        return "operator-";
    case IExpression_unary::OK_CAST:
        return "operator_cast";

    // Note: the following operators should not appear in the DAG
    // because there is no assignment
    case IExpression_unary::OK_PRE_INCREMENT:
    case IExpression_unary::OK_PRE_DECREMENT:
    case IExpression_unary::OK_POST_INCREMENT:
    case IExpression_unary::OK_POST_DECREMENT:
        MDL_ASSERT(!"increment/decrement operator inside DAG backend");
        break;
    }
    return "<unknown unary operator>";
}

// Return the name of a binary MDL operator.
const char *DAG_builder::binary_op_to_name(IExpression_binary::Operator op)
{
    switch(op) {
    case IExpression_binary::OK_SELECT:
        MDL_ASSERT(!"operator. should not be used");
        break;
    case IExpression_binary::OK_ARRAY_INDEX:
        return "operator[]";
    case IExpression_binary::OK_MULTIPLY:
        return "operator*";
    case IExpression_binary::OK_DIVIDE:
        return "operator/";
    case IExpression_binary::OK_MODULO:
        return "operator%";
    case IExpression_binary::OK_PLUS:
        return "operator+";
    case IExpression_binary::OK_MINUS:
        return "operator-";
    case IExpression_binary::OK_SHIFT_LEFT:
        return "operator<<";
    case IExpression_binary::OK_SHIFT_RIGHT:
        return "operator>>";
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
        return "operator>>>";
    case IExpression_binary::OK_LESS:
        return "operator<";
    case IExpression_binary::OK_LESS_OR_EQUAL:
        return "operator<=";
    case IExpression_binary::OK_GREATER_OR_EQUAL:
        return "operator>=";
    case IExpression_binary::OK_GREATER:
        return "operator>";
    case IExpression_binary::OK_EQUAL:
        return "operator==";
    case IExpression_binary::OK_NOT_EQUAL:
        return "operator!=";
    case IExpression_binary::OK_BITWISE_AND:
        return "operator&";
    case IExpression_binary::OK_BITWISE_XOR:
        return "operator^";
    case IExpression_binary::OK_BITWISE_OR:
        return "operator|";
    case IExpression_binary::OK_LOGICAL_AND:
        return "operator&&";
    case IExpression_binary::OK_LOGICAL_OR:
        return "operator||";

    // Note: the following operators should not appear in the DAG
    // because there is no assignment
    case IExpression_binary::OK_ASSIGN:
    case IExpression_binary::OK_MULTIPLY_ASSIGN:
    case IExpression_binary::OK_DIVIDE_ASSIGN:
    case IExpression_binary::OK_MODULO_ASSIGN:
    case IExpression_binary::OK_PLUS_ASSIGN:
    case IExpression_binary::OK_MINUS_ASSIGN:
    case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
    case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
    case IExpression_binary::OK_BITWISE_AND_ASSIGN:
    case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
    case IExpression_binary::OK_BITWISE_OR_ASSIGN:
        MDL_ASSERT(!"assign operator inside DAG backend");
        break;

        // Note: because we have no assignments, sequence are just to "hold"
        // some state referencing calls
    case IExpression_binary::OK_SEQUENCE:
        return "operator,";
    }
    return "<unknown binary operator>";
}

// Get the definition of the parameter of a function/material at index.
IDefinition const *DAG_builder::get_parameter_definition(
    IDeclaration_function const *decl,
    int                         index)
{
    MDL_ASSERT(!decl->is_preset());
    IParameter const   *parameter   = decl->get_parameter(index);
    ISimple_name const *simple_name = parameter->get_name();
    return simple_name->get_definition();
}

// Set array size for size-deferred arrays
void DAG_builder::set_parameter_array_size_var(
    IDeclaration_function const *decl,
    int                         index,
    DAG_node const *            arg_exp)
{
    IParameter const *param = decl->get_parameter(index);
    if (ISimple_name const *size_name = param->get_type_name()->get_size_name()) {
        IType_array const *arg_type = cast<IType_array>(arg_exp->get_type());
        IValue const *val = m_value_factory.create_int(arg_type->get_size());
        DAG_node const *array_size = m_node_factory.create_constant(val);
        m_tmp_value_map[size_name->get_definition()] = array_size;
    }
}

// Run local optimizations for a binary expression.
DAG_node const *DAG_builder::optimize_binary_operator(
    IExpression_binary::Operator op,
    DAG_node const               *l,
    DAG_node const               *r,
    IType const                  *type)
{
    DAG_node_factory_impl::normalize(op, l, r);

    IType const *lt = l->get_type();
    IType const *rt = r->get_type();

    string name(binary_op_to_name(op), get_allocator());
    name += '(';
    m_printer.print(lt->skip_type_alias());
    name += m_printer.get_line();
    name += ',';
    m_printer.print(rt->skip_type_alias());
    name += m_printer.get_line();
    name += ')';

    DAG_call::Call_argument call_args[2];
    call_args[0].arg        = l;
    call_args[0].param_name = "x";
    call_args[1].arg        = r;
    call_args[1].param_name = "y";
    return m_node_factory.create_call(
        name.c_str(),
        operator_to_semantic(op),
        call_args,
        2,
        type);
}

// Convert an MDL let temporary declaration to a DAG IR node.
DAG_node const *DAG_builder::var_decl_to_dag(
    IDeclaration_variable const *var_decl)
{
    for (int i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
        IExpression const  *init = var_decl->get_variable_init(i);
        ISimple_name const *name = var_decl->get_variable_name(i);
        IDefinition const  *vdef = name->get_definition();
        DAG_node const     *expr = NULL;

        if (init != NULL) {
            expr = exp_to_dag(init);
        } else {
            // missing default initializer, happens with "uninitialized" arrays
            IType const *var_type = var_decl->get_type_name()->get_type();

            expr = default_initializer(var_type);
        }

        m_tmp_value_map[vdef] = expr;
    }
    return NULL;
}

// Convert an MDL constant declaration to a DAG IR node.
DAG_node const *DAG_builder::const_decl_to_dag(
    IDeclaration_constant const *const_decl)
{
    for (int i = 0, n = const_decl->get_constant_count(); i < n; ++i) {
        IExpression const  *init = const_decl->get_constant_exp(i);
        ISimple_name const *name = const_decl->get_constant_name(i);
        IDefinition const  *cdef = name->get_definition();
        DAG_node const     *expr = NULL;

        if (init != NULL) {
            expr = exp_to_dag(init);
        } else {
            // missing default initializer, happens with "uninitialized" arrays
            IType const *const_type = const_decl->get_type_name()->get_type();

            expr = default_initializer(const_type);
        }

        m_tmp_value_map[cdef] = expr;
    }
    return NULL;
}

// Convert an MDL statement to a DAG IR node.
DAG_node const *DAG_builder::stmt_to_dag(
    IStatement const  *stmt)
{
    if (m_skip_flags != INL_NO_SKIP) return NULL;

    switch (stmt->get_kind()) {
    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression const *e_stmt = cast<IStatement_expression>(stmt);
            IExpression           const *expr   = e_stmt->get_expression();

            return exp_to_dag(expr);
        }
    case IStatement::SK_DECLARATION:
        {
            IDeclaration const *decl = cast<IStatement_declaration>(stmt)->get_declaration();

            switch (decl->get_kind()) {
            case IDeclaration::DK_CONSTANT:
                {
                    IDeclaration_constant const *const_decl = cast<IDeclaration_constant>(decl);

                    return const_decl_to_dag(const_decl);
                }
                break;
            case IDeclaration::DK_TYPE_ALIAS:
            case IDeclaration::DK_TYPE_ENUM:
            case IDeclaration::DK_TYPE_STRUCT:
                // local type declaration add nothing
                break;
            case IDeclaration::DK_VARIABLE:
                {
                    IDeclaration_variable const *var_decl = cast<IDeclaration_variable>(decl);

                    return var_decl_to_dag(var_decl);
                }
                break;
            default:
                // all others are unsupported
                MDL_ASSERT(!"unsupported declaration statement in inline");
                break;
            }
        }
        return NULL;
    case IStatement::SK_RETURN:
        {
            IExpression const *expr = cast<IStatement_return>(stmt)->get_expression();
            m_inline_return_node = exp_to_dag(expr);
            m_skip_flags = Inline_skip_flag(m_skip_flags | INL_SKIP_RETURN);
            return m_inline_return_node;
        }
    case IStatement::SK_COMPOUND:
    case IStatement::SK_CASE:
        {
            IStatement_compound const *block = cast<IStatement_compound>(stmt);

            DAG_node const *last = NULL;
            for (int i = 0, n = block->get_statement_count(); i < n; ++i) {
                IStatement const *stmt = block->get_statement(i);

                last = stmt_to_dag(stmt);
                if (m_skip_flags != INL_NO_SKIP) return NULL;
            }
            return last;
        }
    case IStatement::SK_IF:
        {
            IStatement_if const *if_stmt = cast<IStatement_if>(stmt);
            Variable_lookup_handler var_lookup_handler(
                tos_module(),
                m_node_factory.get_value_factory(),
                m_node_factory.get_call_evaluator(),
                m_tmp_value_map);
            IValue const *cond_result =
                if_stmt->get_condition()->fold(
                    tos_module(),
                    m_node_factory.get_value_factory(),
                    &var_lookup_handler);
            if (is<IValue_bad>(cond_result)) {
                MDL_ASSERT(!"try_inline lied about condition of if statement");
                return NULL;
            }

            // true case
            if (cond_result->is_one())
                return stmt_to_dag(if_stmt->get_then_statement());
            else if (!cond_result->is_zero()) {
                MDL_ASSERT(!"unexpected condition value");
                return NULL;
            }

            // false case
            if (if_stmt->get_else_statement() == NULL)
                return NULL;
            else
                return stmt_to_dag(if_stmt->get_else_statement());
        }
    case IStatement::SK_SWITCH:
        {
            IStatement_switch const *switch_stmt = cast<IStatement_switch>(stmt);
            Variable_lookup_handler var_lookup_handler(
                tos_module(),
                m_node_factory.get_value_factory(),
                m_node_factory.get_call_evaluator(),
                m_tmp_value_map);
            IValue const *cond_result =
                switch_stmt->get_condition()->fold(
                    tos_module(), m_node_factory.get_value_factory(), &var_lookup_handler);
            if (is<IValue_bad>(cond_result)) {
                MDL_ASSERT(!"try_inline lied about condition of switch statement");
                return NULL;
            }
            IValue_int_valued const *cond_val = cast<IValue_int_valued>(cond_result);
            MDL_ASSERT(cond_val != NULL && "condition not folded to int");
            int cond_int = cond_val->get_value();

            // find a case for the folded condition value
            size_t num_cases = switch_stmt->get_case_count(), cur_case_idx = num_cases;
            for (size_t i = 0; i < num_cases; ++i) {
                IStatement_case const *case_stmt = cast<IStatement_case>(switch_stmt->get_case(i));

                IExpression const *label = case_stmt->get_label();
                if (label == NULL) {
                    // default case, remember as best match but continue searching
                    cur_case_idx = i;
                    continue;
                }

                IExpression_literal const *lit = cast<IExpression_literal>(label);
                IValue_int_valued const *v = cast<IValue_int_valued>(lit->get_value());
                if (v->get_value() == cond_int) {
                    cur_case_idx = i;
                    break;
                }
            }

            // process (next) matching case
            while (cur_case_idx < num_cases) {
                IStatement_case const *cur_case =
                    cast<IStatement_case>(switch_stmt->get_case(cur_case_idx));

                DAG_node const *res = stmt_to_dag(cur_case);

                // case inlining results in any skipping?
                if (m_skip_flags != INL_NO_SKIP) {
                    // consume any breaks
                    m_skip_flags = Inline_skip_flag(m_skip_flags & ~INL_SKIP_BREAK);
                    return res;
                }

                // no skipping, so it's a fall-through, move to next case
                ++cur_case_idx;
            }

            // reached end of switch
            return NULL;
        }
    case IStatement::SK_FOR:
        {
            IStatement_for const *for_stmt = cast<IStatement_for>(stmt);
            IModule const *module = tos_module();
            Variable_lookup_handler var_lookup_handler(
                tos_module(),
                m_node_factory.get_value_factory(),
                m_node_factory.get_call_evaluator(),
                m_tmp_value_map);
            IExpression const *cond = for_stmt->get_condition();

            Inline_scope for_scope(*this, /*in_same_function=*/ true);
            stmt_to_dag(for_stmt->get_init());

            // allow up to 4 unroll steps
            const unsigned max_steps = 4;
            unsigned steps;
            for (steps = 0; steps <= max_steps; ++steps) {
                IValue const *cond_result = cond->fold(
                    module, m_node_factory.get_value_factory(),  &var_lookup_handler);
                if (cond_result->is_zero()) {
                    break;
                } else if (!cond_result->is_one()) {
                    MDL_ASSERT(!"condition could not be folded during for-loop unrolling");
                    break;
                }

                stmt_to_dag(for_stmt->get_body());
                if (m_skip_flags != INL_NO_SKIP)
                    break;

                exp_to_dag(for_stmt->get_update());
            }

            MDL_ASSERT(steps <= max_steps && "try_inline did not recognize number of steps");

            // consume any breaks
            m_skip_flags = Inline_skip_flag(m_skip_flags & ~INL_SKIP_BREAK);
            return NULL;
        }
    case IStatement::SK_BREAK:
        {
            m_skip_flags = Inline_skip_flag(m_skip_flags | INL_SKIP_BREAK);
            return NULL;
        }
    default:
        // others not yet supported
        MDL_ASSERT(!"unsupported statement in inline");
        return NULL;
    }
}

// Convert an MDL expression to a DAG expression.
DAG_node const *DAG_builder::exp_to_dag(
    IExpression const *expr)
{
    if (expr == NULL)
        return NULL;
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        return NULL;
    case IExpression::EK_LITERAL:
        {
            IExpression_literal const *lit = cast<IExpression_literal>(expr);
            return lit_to_dag(lit);
        }
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *ref = cast<IExpression_reference>(expr);
            return ref_to_dag(ref);
        }
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *unary = cast<IExpression_unary>(expr);
            return unary_to_dag(unary);
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *binary = cast<IExpression_binary>(expr);
            return binary_to_dag(binary);
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *cond = cast<IExpression_conditional>(expr);
            return cond_to_dag(cond);
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *call = cast<IExpression_call>(expr);
            return call_to_dag(call);
        }
    case IExpression::EK_LET:
        {
            IExpression_let const *let = cast<IExpression_let>(expr);
            return let_to_dag(let);
        }
    }
    // not reachable
    return NULL;
}

// Convert an MDL literal expression to a DAG constant.
DAG_constant const *DAG_builder::lit_to_dag(
    IExpression_literal const *lit)
{
    IValue const *value = lit->get_value();
    value = m_value_factory.import(value);
    if (IValue_resource const *res = as<IValue_resource>(value)) {
        res = process_resource_urls(res, lit->access_position());
        value = m_resource_modifier->modify(res, tos_module(), m_value_factory);
    }
    return m_node_factory.create_constant(value);
}

// Convert an MDL reference expression to a DAG expression.
DAG_node const *DAG_builder::ref_to_dag(
    IExpression_reference const *ref)
{
    const IDefinition *ref_def = ref->get_definition();
    switch (ref_def->get_kind()) {
    case IDefinition::DK_CONSTANT:
    case IDefinition::DK_ENUM_VALUE:
        {
            // enum values and constants are simply transformed into a DAG Constant
            IValue const *value = ref_def->get_constant_value();
            return m_node_factory.create_constant(m_value_factory.import(value));
        }

    case IDefinition::DK_VARIABLE:
        {
            Definition_temporary_map::const_iterator it = m_tmp_value_map.find(ref_def);
            if (it == m_tmp_value_map.end()) {
                // This is really an error: a reference to something we don't know.
                MDL_ASSERT(!"unknown temporary");
                break;
            } else {
                return it->second;
            }
        }
    case IDefinition::DK_PARAMETER:
        {
            Definition_temporary_map::const_iterator it = m_tmp_value_map.find(ref_def);
            if (it == m_tmp_value_map.end()) {
                // an unattached parameter
                int parameter_index = ref_def->get_parameter_index();
                MDL_ASSERT(parameter_index >= 0);

                // import the parameter type
                IType const *p_type = m_type_factory.import(
                    ref_def->get_type()->skip_type_alias());

                return m_node_factory.create_parameter(p_type, parameter_index);
            } else {
                return it->second;
            }
        }
    case IDefinition::DK_MEMBER:
        {
            // We should not come here ...
            const ISymbol *symbol = ref_def->get_symbol();
            const char *name = symbol->get_name();
            const IValue *str = m_value_factory.create_string(name);
            return m_node_factory.create_constant(str);
        }

    case IDefinition::DK_ARRAY_SIZE:
        {
            ISymbol const *symbol = ref_def->get_symbol();

            // The only source of array sizes here should be parameters (and temporaries?).
            // Find the parameter.
            IDefinition const *param_def = find_parameter_for_size(symbol);

            MDL_ASSERT(param_def != NULL && "could not find parameter for array size");
            if (param_def != NULL) {
                // get the parameter
                DAG_node const *node = NULL;

                Definition_temporary_map::const_iterator dit = m_tmp_value_map.find(param_def);
                if (dit == m_tmp_value_map.end()) {
                    // an unattached parameter
                    int parameter_index = param_def->get_parameter_index();
                    MDL_ASSERT(parameter_index >= 0);

                    // import the parameter type
                    IType const *p_type = m_type_factory.import(
                        param_def->get_type()->skip_type_alias());

                    node = m_node_factory.create_parameter(p_type, parameter_index);
                } else {
                    node = dit->second;

                    IType_array const *a_type = cast<IType_array>(node->get_type());
                    if (a_type->is_immediate_sized()) {
                        int size = a_type->get_size();

                        IValue const *val = m_value_factory.create_int(size);
                        return m_node_factory.create_constant(val);
                    }
                }

                // create a call
                DAG_call::Call_argument arg(node, "a");

                return m_node_factory.create_call(
                    "operator_len(<0>[])",
                    IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH,
                    &arg, 1,
                    m_type_factory.create_int());
            }
            break;
        }

    case IDefinition::DK_ERROR:
        // should not happen, we compile error free code only
        MDL_ASSERT(!"error definition found");
        break;
    case IDefinition::DK_ANNOTATION:
        // annotations could not be references in error free code
        MDL_ASSERT(!"annotation referenced in an expression");
        break;
    case IDefinition::DK_TYPE:
        // types could not be references in error free code
        MDL_ASSERT(!"type referenced in an expression");
        break;
    case IDefinition::DK_FUNCTION:
    case IDefinition::DK_CONSTRUCTOR:
    case IDefinition::DK_OPERATOR:
        // functions could not be references in error free code
        MDL_ASSERT(!"function referenced in an expression");
        break;
    case IDefinition::DK_NAMESPACE:
        MDL_ASSERT(!"namespace alias referenced in an expression");
        break;
    }
    return NULL;
}

// Returns the name of an unary operator.
string DAG_builder::get_unary_name(IExpression_unary const *unary, bool with_signature_suffix) const
{
    IExpression_unary::Operator op   = unary->get_operator();
    IExpression const           *arg = unary->get_argument();

    string name(unary_op_to_name(op), get_allocator());
    if (with_signature_suffix) {
        name += '(';
        m_printer.print(arg->get_type()->skip_type_alias());
        name += m_printer.get_line();
        name += ')';
    }

    return name;
}

// Convert an MDL unary expression to a DAG expression.
DAG_node const *DAG_builder::unary_to_dag(
    IExpression_unary const *unary)
{
    IExpression_unary::Operator op   = unary->get_operator();
    IExpression const           *arg = unary->get_argument();

    IExpression_binary::Operator bin_op;
    bool is_post_op = false;

    switch (op) {
    case IExpression_unary::OK_PRE_INCREMENT:
        bin_op = IExpression_binary::OK_PLUS;
        break;
    case IExpression_unary::OK_PRE_DECREMENT:
        bin_op = IExpression_binary::OK_MINUS;
        break;
    case IExpression_unary::OK_POST_INCREMENT:
        bin_op = IExpression_binary::OK_PLUS;
        is_post_op = true;
        break;
    case IExpression_unary::OK_POST_DECREMENT:
        bin_op = IExpression_binary::OK_MINUS;
        is_post_op = true;
        break;
    default:
        {
            DAG_node const *node = exp_to_dag(arg);
            DAG_constant const *c = as<DAG_constant>(node);
            if (c && m_node_factory.all_args_without_name(&node, 1)) {
                IValue const *v =
                    DAG_node_factory_impl::apply_unary_op(m_value_factory, op, c->get_value());
                if (v != NULL)
                    return m_node_factory.create_constant(v);
            }

            IType const *ret_type = m_type_factory.import(unary->get_type());
            string name(unary_op_to_name(op),get_allocator());
            name += '(';
            m_printer.print(arg->get_type()->skip_type_alias());
            name += m_printer.get_line();
            name += ')';

            DAG_call::Call_argument call_args[1];
            call_args[0].arg        = node;
            call_args[0].param_name = "x";

            return m_node_factory.create_call(
                name.c_str(),
                operator_to_semantic(op),
                call_args,
                1,
                ret_type);
        }
    }

    // if we got here, we have a post/pre increment/decrement
    IType const *type = arg->get_type()->skip_type_alias();

    if (IType_vector const *v_type = as<IType_vector>(type))
        type = v_type->get_element_type();

    IValue const   *v_one = create_pre_post_one(type, &m_value_factory);

    DAG_constant const *one = m_node_factory.create_constant(v_one);

    DAG_node const *pre_node = exp_to_dag(arg);
    DAG_node const *res = NULL;

    if (is<DAG_constant>(pre_node) && m_node_factory.all_args_without_name(&pre_node, 1)) {
        IValue const *v_l = cast<DAG_constant>(pre_node)->get_value();

        IValue const *v =
            DAG_node_factory_impl::apply_binary_op(m_value_factory, bin_op, v_l, v_one);
        if (v != NULL)
            res = m_node_factory.create_constant(v);
    }

    if (res == NULL) {
        res = optimize_binary_operator(
            bin_op,
            pre_node,
            one,
            m_type_factory.import(unary->get_type()));
    }

    if (is_simple_select(arg)) {
        IExpression_binary const    *select = cast<IExpression_binary>(arg);
        IExpression_reference const *ref    =
            cast<IExpression_reference>(select->get_left_argument());
        IExpression_reference const *member =
            cast<IExpression_reference>(select->get_right_argument());

        IDefinition const *vdef    = ref->get_definition();
        IType const       *c_tp    = ref->get_type()->skip_type_alias();

        // ensure we operate on own types
        c_tp = m_type_factory.import(c_tp);

        IDefinition const *mem_def = member->get_definition();

        DAG_node const *old_node = m_tmp_value_map[vdef];
        DAG_node const *new_node = old_node;

        switch (c_tp->get_kind()) {
        case IType::TK_STRUCT:
            {
                IType_struct const *s_type = cast<IType_struct>(c_tp);
                new_node =
                    create_struct_insert(s_type, mem_def->get_field_index(), old_node, res);
            }
            break;
        case IType::TK_VECTOR:
            {
                IType_vector const *v_type = cast<IType_vector>(c_tp);
                new_node =
                    create_vector_insert(v_type, mem_def->get_field_index(), old_node, res);
            }
            break;
        default:
            MDL_ASSERT(!"unsupported compound type");
            break;
        }

        // update the map with the new compound value
        m_tmp_value_map[vdef] = new_node;
    } else {
        // currently we do not support array assignments
        IExpression_reference const *ref  = cast<IExpression_reference>(arg);
        IDefinition const           *vdef = ref->get_definition();

        m_tmp_value_map[vdef] = res;
    }

    return is_post_op ? pre_node : res;
}

// Returns the name of a binary operator.
string DAG_builder::get_binary_name(IExpression_binary const *binary, bool with_signature_suffix) const
{
    IExpression_binary::Operator op = binary->get_operator();
    IExpression const *left  = binary->get_left_argument();
    IExpression const *right = binary->get_right_argument();

    switch (op) {
    case IExpression_binary::OK_SELECT:
        {
            // handle select operator
            IType const *left_type = left->get_type()->skip_type_alias();

            IExpression_reference const *ref     = cast<IExpression_reference>(right);
            IDefinition const           *ref_def = ref->get_definition();

            MDL_ASSERT(
                ref_def->get_kind() == IDefinition::DK_MEMBER &&
                "selector is not a member");

            ISymbol const *symbol = ref_def->get_symbol();
            // import it into our symbol table
            symbol = m_sym_tab.get_symbol(symbol->get_name());

            switch (left_type->get_kind()) {
            case IType::TK_STRUCT:
                {
                    IType_struct const *s_type = cast<IType_struct>(left_type);
                    char const         *s_name = s_type->get_symbol()->get_name();

                    // create the name (+ signature) of the getter here
                    string name(s_name, get_allocator());
                    name += '.';
                    name += symbol->get_name();
                    if (with_signature_suffix) {
                        name += '(';
                        name += s_name;
                        name += ')';
                    }

                    return name;
                }
            case IType::TK_VECTOR:
                {
                    IType_vector const *v_type = cast<IType_vector>(left_type);
                    string             v_name(type_to_name(v_type));

                    // create the name (+ signature) of the index function here
                    string name = v_name;
                    name += "@";
                    if (with_signature_suffix) {
                        name += "(";
                        name += v_name;
                        name += ",int)";
                    }

                    return name;
                }
            default:
                break;
            }
        }
        break;
    case IExpression_binary::OK_ARRAY_INDEX:
        // one for all
        return string("operator[](<0>[],int)", get_allocator());
    case IExpression_binary::OK_SEQUENCE:
        // ignore comma operator, left argument is dropped.
        return string(get_allocator());
    case IExpression_binary::OK_ASSIGN:
        // assignment operator
        return string(get_allocator());

    case IExpression_binary::OK_MULTIPLY_ASSIGN:
        op = IExpression_binary::OK_MULTIPLY;
        break;
    case IExpression_binary::OK_DIVIDE_ASSIGN:
        op = IExpression_binary::OK_DIVIDE;
        break;
    case IExpression_binary::OK_MODULO_ASSIGN:
        op = IExpression_binary::OK_MODULO;
        break;
    case IExpression_binary::OK_PLUS_ASSIGN:
        op = IExpression_binary::OK_PLUS;
        break;
    case IExpression_binary::OK_MINUS_ASSIGN:
        op = IExpression_binary::OK_MINUS;
        break;
    case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
        op = IExpression_binary::OK_SHIFT_LEFT;
        break;
    case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
        op = IExpression_binary::OK_SHIFT_RIGHT;
        break;
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
        op = IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT;
        break;
    case IExpression_binary::OK_BITWISE_AND_ASSIGN:
        op = IExpression_binary::OK_BITWISE_AND;
        break;
    case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
        op = IExpression_binary::OK_BITWISE_XOR;
        break;
    case IExpression_binary::OK_BITWISE_OR_ASSIGN:
        op = IExpression_binary::OK_BITWISE_OR;
        break;
    default:
        break;
    }

    IType const *lt = left->get_type();
    IType const *rt = right->get_type();

    string name(binary_op_to_name(op), get_allocator());

    if (with_signature_suffix) {
        name += '(';
        m_printer.print(lt->skip_type_alias());
        name += m_printer.get_line();
        name += ',';
        m_printer.print(rt->skip_type_alias());
        name += m_printer.get_line();
        name += ')';
    }

    return name;
}

// Creates a insert pseudo-instruction on a struct value.
DAG_node const *DAG_builder::create_struct_insert(
    IType_struct const *s_type,
    int                index,
    DAG_node const     *c_node,
    DAG_node const     *e_node)
{
    IValue_struct const *s_val = NULL;

    if (is<DAG_constant>(c_node)) {
        s_val = cast<IValue_struct>(cast<DAG_constant>(c_node)->get_value());
    }

    // create a S(old.a, ..., res, ..., old.z)
    int n_fields = s_type->get_field_count();

    bool all_args_const = true;

    VLA<DAG_call::Call_argument> call_args(get_allocator(), n_fields);

    // FIXME: This should be done somehow by the DAG mangler ...
    // create a constructor call
    string name = type_to_name(s_type);
    name += '(';

    for (int i = 0; i < n_fields; ++i) {
        DAG_node const *field = NULL;

        IType const   *field_tp;
        ISymbol const *field_sym;
        s_type->get_field(i, field_tp, field_sym);

        if (i != 0)
            name += ',';
        name += type_to_name(field_tp);

        if (i == index) {
            field = e_node;
        } else {
            // retrieve the field from the old value
            if (s_val != NULL) {
                IValue const *res = s_val->get_value(i);

                if (!is<IValue_bad>(res)) {
                    res = m_value_factory.import(res);
                    field = m_node_factory.create_constant(res);
                }
            }

            if (field == NULL) {
                // not a constant

                char const *s_name = s_type->get_symbol()->get_name();

                // create the name (+ signature) of the getter here
                string name(s_name, get_allocator());
                name += '.';
                name += field_sym->get_name();
                name += '(';
                name += s_name;
                name += ')';

                DAG_call::Call_argument call_args[1];

                call_args[0].arg        = c_node;
                call_args[0].param_name = "s";
                field = m_node_factory.create_call(
                    name.c_str(), IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS, call_args, 1,
                    field_tp);
            }
        }
        call_args[i].param_name = field_sym->get_name();
        call_args[i].arg        = field;
        all_args_const          = all_args_const && is<DAG_constant>(field);
    }

    name += ')';

    if (all_args_const && m_node_factory.all_args_without_name(call_args.data(), n_fields)) {
        // we know this is an elemental constructor, fold it
        Value_vector values(n_fields, NULL, get_allocator());
        for (int i = 0; i < n_fields; ++i)
            values[i] = cast<DAG_constant>(call_args[i].arg)->get_value();

        IValue const *v = m_node_factory.evaluate_constructor(
            m_value_factory, IDefinition::DS_ELEM_CONSTRUCTOR, s_type, values);
        if (v != NULL)
            return m_node_factory.create_constant(v);
    }

    // else create a constructor call
    // string name = elem_constructor_name(c_type);

    return m_node_factory.create_call(
        name.c_str(), IDefinition::DS_ELEM_CONSTRUCTOR, call_args.data(), n_fields, s_type);
}

// Creates a insert pseudo-instruction on a struct value.
DAG_node const *DAG_builder::create_vector_insert(
    IType_vector const *v_type,
    int                index,
    DAG_node const     *c_node,
    DAG_node const     *e_node)
{
    IValue_vector const *v_val = NULL;

    if (is<DAG_constant>(c_node)) {
        v_val = cast<IValue_vector>(cast<DAG_constant>(c_node)->get_value());
    }

    // create a V(old[0], ..., res, ..., old.[n-1])
    int n_fields = v_type->get_size();

    bool all_args_const = true;

    VLA<DAG_call::Call_argument> call_args(get_allocator(), n_fields);

    // create a constructor call
    string v_name(type_to_name(v_type));

    string name(v_name);
    name += '(';

    IType const *elem_tp = v_type->get_element_type();

    for (int i = 0; i < n_fields; ++i) {
        DAG_node const *field = NULL;

        // FIXME: remove implicit knowledge
        static char const *vector_components[] = { "x", "y", "z", "w" };
        char const        *field_name = vector_components[i];

        if (i != 0)
            name += ',';
        name += type_to_name(elem_tp);

        if (i == index) {
            field = e_node;
        } else {
            // retrieve the field from the old value
            if (v_val != NULL) {
                IValue const *res = v_val->get_value(i);

                if (!is<IValue_bad>(res)) {
                    res = m_value_factory.import(res);
                    field = m_node_factory.create_constant(res);
                }
            }

            if (field == NULL) {
                // not a constant

                DAG_call::Call_argument call_args[2];

                call_args[0].arg        = c_node;
                call_args[0].param_name = "a";

                call_args[1].arg        = m_node_factory.create_constant(
                    m_value_factory.create_int(i));
                call_args[1].param_name = "i";

                field = m_node_factory.create_call(
                    "operator[](<0>[],int)",
                    operator_to_semantic(IExpression::OK_ARRAY_INDEX),
                    call_args, 2,
                    elem_tp);
            }
        }
        call_args[i].param_name = field_name;
        call_args[i].arg        = field;
        all_args_const          = all_args_const && is<DAG_constant>(field);
    }

    name += ')';

    if (all_args_const && m_node_factory.all_args_without_name(call_args.data(), n_fields)) {
        // we know this is an elemental constructor, fold it
        Value_vector values(n_fields, NULL, get_allocator());
        for (int i = 0; i < n_fields; ++i)
            values[i] = cast<DAG_constant>(call_args[i].arg)->get_value();

        IValue const *v = m_node_factory.evaluate_constructor(
            m_value_factory, IDefinition::DS_ELEM_CONSTRUCTOR, v_type, values);
        if (v != NULL)
            return m_node_factory.create_constant(v);
    }

    // else create a constructor call
    // string name = elem_constructor_name(c_type);

    return m_node_factory.create_call(
        name.c_str(), IDefinition::DS_ELEM_CONSTRUCTOR, call_args.data(), n_fields, v_type);
}

// Convert an MDL binary expression to a DAG expression.
DAG_node const *DAG_builder::binary_to_dag(
    IExpression_binary const *binary)
{
    IExpression_binary::Operator op = binary->get_operator();
    IExpression const *left  = binary->get_left_argument();
    IExpression const *right = binary->get_right_argument();

    bool is_assign = false;

    switch (op) {
    case IExpression_binary::OK_SELECT:
        {
            // handle select operator
            IType const *left_type = left->get_type()->skip_type_alias();

            IExpression_reference const *ref     = cast<IExpression_reference>(right);
            IDefinition const           *ref_def = ref->get_definition();

            MDL_ASSERT(
                ref_def->get_kind() == IDefinition::DK_MEMBER &&
                "selector is not a member");

            ISymbol const *symbol = ref_def->get_symbol();
            // import it into our symbol table
            symbol = m_sym_tab.get_symbol(symbol->get_name());

            DAG_node const *compound = exp_to_dag(left);

            DAG_constant const *c = as<DAG_constant>(compound);
            if (c && m_node_factory.all_args_without_name(&compound, 1)) {
                // extracting a field
                IValue const *left = c->get_value();

                if (IValue_vector const *vv = as<IValue_vector>(left)) {
                    char const *name = symbol->get_name();
                    int index = -1;
                    switch (name[0]) {
                    case 'x': index = 0; break;
                    case 'y': index = 1; break;
                    case 'z': index = 2; break;
                    case 'w': index = 3; break;
                    default:
                        break;
                    }
                    MDL_ASSERT(0 <= index && index < vv->get_component_count());
                    IValue const *res = vv->get_value(index);
                    if (!is<IValue_bad>(res)) {
                        res = m_value_factory.import(res);
                        return m_node_factory.create_constant(res);
                    }
                } else if (IValue_struct const *sv = as<IValue_struct>(left)) {
                    IValue const *res = sv->get_field(symbol);
                    if (!is<IValue_bad>(res)) {
                        res = m_value_factory.import(res);
                        return m_node_factory.create_constant(res);
                    }
                }
            }

            switch (left_type->get_kind()) {
            case IType::TK_STRUCT:
                {
                    IType_struct const *s_type = cast<IType_struct>(left_type);
                    char const         *s_name = s_type->get_symbol()->get_name();

                    // create the name (+ signature) of the getter here
                    string name(s_name, get_allocator());
                    name += '.';
                    name += symbol->get_name();
                    name += '(';
                    name += s_name;
                    name += ')';

                    DAG_call::Call_argument call_args[1];

                    call_args[0].arg        = compound;
                    call_args[0].param_name = "s";
                    return m_node_factory.create_call(
                        name.c_str(), IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS, call_args, 1,
                        m_type_factory.import(binary->get_type()));
                }
            case IType::TK_VECTOR:
                {
                    char const *fname = symbol->get_name();

                    int index = -1;
                    switch (fname[0]) {
                    case 'x': index = 0; break;
                    case 'y': index = 1; break;
                    case 'z': index = 2; break;
                    case 'w': index = 3; break;
                    default:
                        break;
                    }
                    MDL_ASSERT(
                        fname[1] == '\0' &&
                        0 <= index &&
                        index < cast<IType_vector>(left_type)->get_size());

                    // create an index access
                    DAG_call::Call_argument call_args[2];

                    call_args[0].arg        = compound;
                    call_args[0].param_name = "a";

                    call_args[1].arg        = m_node_factory.create_constant(
                        m_value_factory.create_int(index));
                    call_args[1].param_name = "i";

                    return m_node_factory.create_call(
                        "operator[](<0>[],int)",
                        operator_to_semantic(IExpression::OK_ARRAY_INDEX),
                        call_args, 2,
                        m_type_factory.import(binary->get_type()));
                }
            default:
                break;
            }
        }
        break;
    case IExpression_binary::OK_ARRAY_INDEX:
        {
            string name(get_binary_name(binary));

            DAG_call::Call_argument call_args[2];
            call_args[0].arg        = exp_to_dag(left);
            call_args[0].param_name = "a";
            call_args[1].arg        = exp_to_dag(right);
            call_args[1].param_name = "i";
            return m_node_factory.create_call(
                name.c_str(),
                operator_to_semantic(IExpression::OK_ARRAY_INDEX),
                call_args,
                2,
                m_type_factory.import(binary->get_type()));
        }
    case IExpression_binary::OK_SEQUENCE:
        // ignore comma operator, dropping left argument
        return exp_to_dag(right);
    case IExpression_binary::OK_ASSIGN:
        is_assign = true;
        break;
    case IExpression_binary::OK_MULTIPLY_ASSIGN:
        op = IExpression_binary::OK_MULTIPLY;
        is_assign = true;
        break;
    case IExpression_binary::OK_DIVIDE_ASSIGN:
        op = IExpression_binary::OK_DIVIDE;
        is_assign = true;
        break;
    case IExpression_binary::OK_MODULO_ASSIGN:
        op = IExpression_binary::OK_MODULO;
        is_assign = true;
        break;
    case IExpression_binary::OK_PLUS_ASSIGN:
        op = IExpression_binary::OK_PLUS;
        is_assign = true;
        break;
    case IExpression_binary::OK_MINUS_ASSIGN:
        op = IExpression_binary::OK_MINUS;
        is_assign = true;
        break;
    case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
        op = IExpression_binary::OK_SHIFT_LEFT;
        is_assign = true;
        break;
    case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
        op = IExpression_binary::OK_SHIFT_RIGHT;
        is_assign = true;
        break;
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
        op = IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT;
        is_assign = true;
        break;
    case IExpression_binary::OK_BITWISE_AND_ASSIGN:
        op = IExpression_binary::OK_BITWISE_AND;
        is_assign = true;
        break;
    case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
        op = IExpression_binary::OK_BITWISE_XOR;
        is_assign = true;
        break;
    case IExpression_binary::OK_BITWISE_OR_ASSIGN:
        op = IExpression_binary::OK_BITWISE_OR;
        is_assign = true;
        break;
    default:
        break;
    }

    DAG_node const *r   = exp_to_dag(right);
    DAG_node const *res = NULL;

    if (op == IExpression_binary::OK_ASSIGN)
        res = r;
    else {
        DAG_node const *l = exp_to_dag(left);
        DAG_node const *args[2] = { l, r };

        if (is<DAG_constant>(l) && is<DAG_constant>(r) && m_node_factory.all_args_without_name(args, 2)) {
            IValue const *v_l = cast<DAG_constant>(l)->get_value();
            IValue const *v_r = cast<DAG_constant>(r)->get_value();

            IValue const *v =
                DAG_node_factory_impl::apply_binary_op(m_value_factory, op, v_l, v_r);
            if (v != NULL)
                res = m_node_factory.create_constant(v);
        }

        if (res == NULL) {
            res = optimize_binary_operator(
                op,
                l,
                r,
                m_type_factory.import(binary->get_type()));
        }
    }

    if (is_assign) {
        if (is_simple_select(left)) {
            IExpression_binary const    *select = cast<IExpression_binary>(left);
            IExpression_reference const *ref    =
                cast<IExpression_reference>(select->get_left_argument());
            IExpression_reference const *member =
                cast<IExpression_reference>(select->get_right_argument());

            IDefinition const *vdef    = ref->get_definition();
            IType const       *c_tp    = ref->get_type()->skip_type_alias();

            // ensure we operate on own types
            c_tp = m_type_factory.import(c_tp);

            IDefinition const *mem_def = member->get_definition();

            DAG_node const *old_node = m_tmp_value_map[vdef];
            DAG_node const *new_node = old_node;

            switch (c_tp->get_kind()) {
            case IType::TK_STRUCT:
                {
                    IType_struct const *s_type = cast<IType_struct>(c_tp);
                    new_node =
                        create_struct_insert(s_type, mem_def->get_field_index(), old_node, res);
                }
                break;
            case IType::TK_VECTOR:
                {
                    IType_vector const *v_type = cast<IType_vector>(c_tp);
                    new_node =
                        create_vector_insert(v_type, mem_def->get_field_index(), old_node, res);
                }
                break;
            default:
                MDL_ASSERT(!"unsupported compound type");
                break;
            }

            // update the map with the new compound value
            m_tmp_value_map[vdef] = new_node;
        } else {
            // currently we do not support array assignments
            IExpression_reference const *ref  = cast<IExpression_reference>(left);
            IDefinition const           *vdef = ref->get_definition();

            m_tmp_value_map[vdef] = res;
        }
    }
    return res;
}

// Convert an MDL conditional expression to a DAG expression.
DAG_node const *DAG_builder::cond_to_dag(
    IExpression_conditional const *cond)
{
    DAG_node const *sel = exp_to_dag(cond->get_condition());

    if (is<DAG_constant>(sel) && m_node_factory.all_args_without_name(&sel, 1)) {
        DAG_constant const *c = cast<DAG_constant>(sel);
        IValue_bool const  *v = cast<IValue_bool>(c->get_value());

        if (v->get_value())
            return exp_to_dag(cond->get_true());
        else
            return exp_to_dag(cond->get_false());
    }

    IExpression::Operator op = IExpression::OK_TERNARY;

    IType const *ret_type = m_type_factory.import(cond->get_type());

    DAG_call::Call_argument call_args[3];
    call_args[0].arg        = sel;
    call_args[0].param_name = "cond";
    call_args[1].arg        = exp_to_dag(cond->get_true());
    call_args[1].param_name = "true_exp";
    call_args[2].arg        = exp_to_dag(cond->get_false());
    call_args[2].param_name = "false_exp";

    IDefinition::Semantics sema = operator_to_semantic(op);
    DAG_node const *res = m_node_factory.create_call(
        get_ternary_operator_signature(),
        sema,
        call_args,
        dimension_of(call_args),
        ret_type);

    if (DAG_call const *call_res = as<DAG_call>(res)) {
        if (call_res->get_semantic() == sema) {
            // still a conditional operator
            m_conditional_created = true;

            if (is<IType_df>(ret_type->skip_type_alias())) {
                // even on a *df type
                m_conditional_df_created = true;
            }
        }
    }
    return res;
}

// Try to inline the given call.
DAG_node const *DAG_builder::try_inline(
    IExpression_call const *call)
{
    if (!m_node_factory.is_inline_allowed())
        return NULL;

    IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());
    IDefinition const      *call_def = ref->get_definition();

    IDefinition const *orig_call_def = tos_module()->get_original_definition(call_def);
    if (!orig_call_def->get_property(IDefinition::DP_ALLOW_INLINE)) {
        // inlining forbidden
        return NULL;
    }

    IType_function const *f_type = cast<IType_function>(orig_call_def->get_type());
    IType const *ret_type = f_type->get_return_type();

    IDeclaration_function const *func_decl =
        cast<IDeclaration_function>(orig_call_def->get_declaration());

    mi::base::Handle<const IModule> owner_mod(tos_module()->get_owner_module(call_def));

    func_decl = skip_presets(func_decl, owner_mod);

    bool is_material = false;
    if (is_material_type(ret_type) &&
        orig_call_def->get_semantics() != IDefinition::DS_ELEM_CONSTRUCTOR)
    {
        // always inline materials
        is_material = true;
    }

    // enter inlining
    Inline_scope inline_scope(*this);

    // if we are here, we can inline this call
    int n_args = call->get_argument_count();

    for (int i = 0; i < n_args; ++i) {
        IDefinition const *parameter_def = get_parameter_definition(func_decl, i);
        IArgument const   *arg           = call->get_argument(i);
        IExpression const *arg_exp       = arg->get_argument_expr();

        DAG_node const *exp = exp_to_dag(arg_exp);

        m_tmp_value_map[parameter_def] = exp;

        make_accessible(parameter_def);

        // set array size for size-deferred arrays
        set_parameter_array_size_var(func_decl, i, exp);
    }

    if (!is_material) {
        Inline_checker checker(
            tos_module(),
            m_node_factory.get_value_factory(),
            m_node_factory.get_call_evaluator(),
            func_decl,
            call,
            m_forbid_local_calls,
            m_tmp_value_map);
        if (!checker.can_inline())
            return NULL;
    }

    Module_scope mod_scope(*this, owner_mod.get());

    DAG_node const *res = stmt_to_dag(func_decl->get_body());
    MDL_ASSERT((m_skip_flags & INL_SKIP_BREAK) == 0 && "break was not properly handled");
    MDL_ASSERT(res != NULL || (m_skip_flags & INL_SKIP_RETURN) != 0);
    if ((m_skip_flags & INL_SKIP_RETURN) != 0) {
        res = m_inline_return_node;
        m_skip_flags = INL_NO_SKIP;
    }
    return res;
}

// Try to inline the given call.
DAG_node const *DAG_builder::try_inline(
    IGenerated_code_dag const     *owner_dag,
    IDefinition const             *def,
    DAG_call::Call_argument const *args,
    size_t                        n_args)
{
    if (!m_node_factory.is_inline_allowed())
        return NULL;

    if (!m_node_factory.is_noinline_ignored() &&
        !def->get_property(IDefinition::DP_ALLOW_INLINE))
    {
        // inlining forbidden
        return NULL;
    }

    if (def->get_kind() != IDefinition::DK_FUNCTION)
        return NULL;

    def = tos_module()->get_original_definition(def);
    IDeclaration_function const *func_decl = cast<IDeclaration_function>(def->get_declaration());
    if (func_decl == NULL) {
        // might happen for compiler generated
        return NULL;
    }

    mi::base::Handle<IModule const> owner_mod(tos_module()->get_owner_module(def));

    func_decl = skip_presets(func_decl, owner_mod);


    // enter inlining
    Inline_scope inline_scope(*this);

    // update the resource modifier if an owner dag was given
    Restable_resource_modifer modifier(
        *impl_cast<Generated_code_dag>(owner_dag), *m_resource_modifier);

    Store<IResource_modifier *> resource_modifier(
        m_resource_modifier, owner_dag == NULL ? m_resource_modifier : &modifier);

    for (size_t i = 0; i < n_args; ++i) {
        IDefinition const *parameter_def = get_parameter_definition(func_decl, i);
        DAG_node const *arg_exp          = args[i].arg;

        m_tmp_value_map[parameter_def] = arg_exp;

        make_accessible(parameter_def);

        // set array size for size-deferred arrays
        set_parameter_array_size_var(func_decl, i, arg_exp);
    }

    {
        vector<IType const *>::Type arg_types(n_args, (IType const *)NULL, get_allocator());
        for (size_t i = 0; i < n_args; ++i) {
            arg_types[i] = args[i].arg->get_type();
        }
        Inline_checker checker(
            tos_module(),
            m_node_factory.get_value_factory(),
            m_node_factory.get_call_evaluator(),
            func_decl,
            arg_types,
            m_forbid_local_calls,
            m_tmp_value_map);
        if (!checker.can_inline())
            return NULL;
    }

    Module_scope mod_scope(*this, owner_mod.get());

    DAG_node const *res = stmt_to_dag(func_decl->get_body());
    MDL_ASSERT((m_skip_flags & INL_SKIP_BREAK) == 0 && "break was not properly handled");
    MDL_ASSERT(res != NULL || (m_skip_flags & INL_SKIP_RETURN) != 0);
    if ( (m_skip_flags & INL_SKIP_RETURN) != 0 ) {
        res = m_inline_return_node;
        m_skip_flags = INL_NO_SKIP;
    }
    return res;
}

// Convert an MDL call expression to a DAG expression.
DAG_node const *DAG_builder::call_to_dag(
    IExpression_call const *call)
{
    // beware, the return type can be an uniform type, hence skip it
    IType const *ret_type = call->get_type()->skip_type_alias();
    ret_type = m_type_factory.import(ret_type);

    IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());

    int n_args = call->get_argument_count();

    if (ref->is_array_constructor()) {
        if (n_args == 1) {
            IArgument const   *arg      = call->get_argument(0);
            IExpression const *expr     = arg->get_argument_expr();
            IType const       *arg_type = expr->get_type()->skip_type_alias();
            if (is<IType_array>(arg_type)) {
                // skip array copy constructor
                MDL_ASSERT(arg_type == call->get_type()->skip_type_alias());
                return exp_to_dag(expr);
            }
        }
    } else {
        IDefinition const      *call_def = ref->get_definition();
        IDefinition::Semantics call_sema = call_def->get_semantics();

        if (call_sema == IDefinition::DS_UNKNOWN) {
            // try to inline user defined function or material
            DAG_node const *res = try_inline(call);
            if (res != NULL)
                return res;
        }

        // m_options & FORBID_LOCAL_FUNC_CALLS
        if (m_forbid_local_calls) {
            if (call_def->get_property(IDefinition::DP_IS_LOCAL_FUNCTION)) {
                error_local_call(ref);
            }
        }
    }

    bool all_args_const   = true;

    VLA<DAG_call::Call_argument> call_args(get_allocator(), n_args);

    for (int i = 0; i < n_args; ++i) {
        IArgument const   *arg     = call->get_argument(i);
        IExpression const *arg_exp = arg->get_argument_expr();
        DAG_node const    *node    = exp_to_dag(arg_exp);

        call_args[i].arg = node;
        all_args_const   = all_args_const && is<DAG_constant>(node);
    }

    if (ref->is_array_constructor()) {
        if (all_args_const && m_node_factory.all_args_without_name(call_args.data(), n_args)) {
            // create an array literal
            VLA<IValue const *> values(get_allocator(), n_args);
            for (int i = 0; i < n_args; ++i)
                values[i] = cast<DAG_constant>(call_args[i].arg)->get_value();

            IType_array const *a_type = cast<IType_array>(ret_type);
            IValue const *v = m_value_factory.create_array(a_type, values.data(), n_args);
            return m_node_factory.create_constant(v);
        }

        // create an array constructor call
        IType_array const *a_type = cast<IType_array>(ret_type);
        MDL_ASSERT(a_type->is_immediate_sized() && "Broken array constructor");

        VLA<char> names(get_allocator(), 32 * n_args);
        for (int i = 0; i < n_args; ++i) {
            char *name = &names[32 * i];
            snprintf(name, 32, "%d", i);

            call_args[i].param_name = name;
        }

        // use the magic array constructor name.
        return m_node_factory.create_call(
            get_array_constructor_signature(),
            IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
            call_args.data(), n_args, a_type);
    }
    // else not an array constructor ...

    IDefinition const      *call_def = ref->get_definition();
    IDefinition::Semantics call_sema = call_def->get_semantics();

    if (all_args_const && m_node_factory.all_args_without_name(call_args.data(), n_args)
        && call_sema != IDefinition::DS_UNKNOWN) {
        // known semantic, try to fold
        Value_vector values(n_args, NULL, get_allocator());
        for (int i = 0; i < n_args; ++i)
            values[i] = cast<DAG_constant>(call_args[i].arg)->get_value();

        IValue const *res = NULL;
        if (call_def->get_kind() == IDefinition::DK_CONSTRUCTOR) {
            res = m_node_factory.evaluate_constructor(
                m_value_factory, call_sema, ret_type, values);
        } else {
            size_t n_args = values.size();
            IValue const * const *args = n_args > 0 ? &values[0] : NULL;
            res = m_node_factory.evaluate_intrinsic_function(call_sema, args, n_args);
        }
        if (res != NULL)
            return m_node_factory.create_constant(res);
    }

    IType_function const *fun_type = cast<IType_function>(call_def->get_type());

    // collect parameter names
    for (int i = 0; i < n_args; ++i) {
        IArgument const *arg = call->get_argument(i);

        if (is<IArgument_positional>(arg)) {
            IType const *parameter_type;
            ISymbol const *parameter_symbol;

            fun_type->get_parameter(i, parameter_type, parameter_symbol);
            call_args[i].param_name = parameter_symbol->get_name();
        } else {
            IArgument_named const *na = cast<IArgument_named>(arg);

            // The core has already made sure that named parameters
            // appear in the correct position.
            ISimple_name const *simple_name = na->get_parameter_name();
            ISymbol const *symbol           = simple_name->get_symbol();

            call_args[i].param_name = symbol->get_name();
        }
    }

    char const * owner_name = tos_module()->get_owner_module_name(call_def);

    // and create a call
    string name = def_to_name(call_def, owner_name);

    return m_node_factory.create_call(
        name.c_str(), call_sema, call_args.data(), n_args, ret_type);
}

// Convert an MDL let expression to a DAG IR node.
DAG_node const *DAG_builder::let_to_dag(
    IExpression_let const *let)
{
    int decl_count = let->get_declaration_count();
    for (int i = 0; i < decl_count; ++i) {
        IDeclaration_variable const *vd = cast<IDeclaration_variable>(let->get_declaration(i));

        int var_count = vd->get_variable_count();
        for (int k = 0; k < var_count; ++k) {
            if (IExpression const *var_exp = vd->get_variable_init(k)) {
                ISimple_name const *var_name = vd->get_variable_name(k);
                IDefinition const  *var_def  = var_name->get_definition();
                ISymbol const      *var_symbol  = var_name->get_symbol();
                char const         *symbol_name = var_symbol->get_name();

                if (m_node_factory.is_exposing_names_of_let_expressions_enabled()) {
                    // We need to ensure that CSE does not identify the to-be-created named top-level
                    // node with a previously created unnamed node (the name is only added *after*
                    // creation). CSE for the top-level node is detected by comparing the node ID of
                    // the top-level node with the next node ID of the factory before calling
                    // exp_to_dag(). If necessary, we create a shallow clone of the top-level node
                    // with CSE disabled.
                    size_t next_id = m_node_factory.get_next_id();
                    DAG_node const *node = exp_to_dag(var_exp);
                    if (node->get_id() < next_id)
                        node = m_node_factory.shallow_copy(node);

                    m_tmp_value_map[var_def] = node;
                    m_node_factory.add_node_name(node, symbol_name);
                } else {
                    DAG_node const *node = exp_to_dag(var_exp);
                    m_tmp_value_map[var_def] = node;
                }
            }
        }
    }
    return exp_to_dag(let->get_expression());
}

// Convert an MDL preset call expression to a DAG IR node.
DAG_node const *DAG_builder::preset_to_dag(
    IDefinition const *orig_mat_def)
{
    IDeclaration_function const *func_decl =
        cast<IDeclaration_function>(orig_mat_def->get_declaration());

    // every parameter of the original material is connected to the corresponding parameter
    // of the preset material
    for (int i = 0, count = func_decl->get_parameter_count(); i < count; ++i) {
        IDefinition const *parameter_def = get_parameter_definition(func_decl, i);

        // import the parameter type
        IType const *p_type = m_type_factory.import(
            parameter_def->get_type()->skip_type_alias());

        DAG_node const *node = m_node_factory.create_parameter(p_type, i);
        m_tmp_value_map[parameter_def] = node;
    }

    IStatement_expression const *stmt = cast<IStatement_expression>(func_decl->get_body());
    IExpression const           *mat_expr = stmt->get_expression();
    return exp_to_dag(mat_expr);
}

// Convert an MDL annotation to a DAG IR node.
DAG_node const *DAG_builder::annotation_to_dag(
    IAnnotation const *annotation)
{
    IQualified_name const *qualified_name = annotation->get_name();
    IDefinition const     *annotation_def = qualified_name->get_definition();

    string name = def_to_name(annotation_def);
    int n_args  = annotation->get_argument_count();

    IType_function const *fun_type = cast<IType_function>(annotation_def->get_type());
    VLA<DAG_call::Call_argument> call_args(get_allocator(), n_args);

    for (int i = 0; i < n_args; ++i) {
        IArgument const   *arg     = annotation->get_argument(i);
        IExpression const *arg_exp = arg->get_argument_expr();

        if (is<IArgument_positional>(arg)) {
            IType const   *parameter_type;
            ISymbol const *parameter_symbol;

            fun_type->get_parameter(i, parameter_type, parameter_symbol);
            call_args[i].arg        = exp_to_dag(arg_exp);
            call_args[i].param_name = parameter_symbol->get_name();
        } else {
            IArgument_named const *na = cast<IArgument_named>(arg);

            // The core has already made sure that named parameters
            // appear in the correct position.
            ISimple_name const *simple_name = na->get_parameter_name();
            ISymbol const      *symbol      = simple_name->get_symbol();

            call_args[i].arg        = exp_to_dag(arg_exp);
            call_args[i].param_name = symbol->get_name();
        }
    }
    IDefinition::Semantics sema = annotation_def->get_semantics();
    return m_node_factory.create_call(
        name.c_str(), sema, call_args.data(), n_args, /*ret_type=*/NULL);
}

// Creates an anno::hidden() annotation.
DAG_node const *DAG_builder::create_hidden_annotation()
{
    if (tos_module()->is_builtins()) {
        // we cannot import ::anno into the builtin module
        return NULL;
    }

    const char *hidden_sig = "::anno::hidden()";
    return m_node_factory.create_call(
        hidden_sig, IDefinition::DS_HIDDEN_ANNOTATION, NULL, 0, /*ret_type=*/NULL);
}

// Find a parameter for a given array size symbol.
IDefinition const *DAG_builder::find_parameter_for_size(ISymbol const *sym) const
{
    for (size_t i = 0, n = m_accesible_parameters.size(); i < n; ++i) {
        IDefinition const *p_def  = m_accesible_parameters[i];
        IType_array const *a_type = as<IType_array>(p_def->get_type());

        if (a_type == NULL)
            continue;
        if (a_type->is_immediate_sized())
            continue;
        IType_array_size const *size = a_type->get_deferred_size();

        // Beware: we extracted the type from an definition that might originate from
        // another module, hence we cannot compare the symbols directly
        ISymbol const *size_sym = size->get_size_symbol();

        if (strcmp(size_sym->get_name(), sym->get_name()) == 0)
            return p_def;
    }
    return NULL;
}

// Report an error due to a call to a local function.
void DAG_builder::error_local_call(IExpression_reference const *ref)
{
    m_error_calls.push_back(ref);
}

// Creates a default initializer for the given type.
DAG_constant const *DAG_builder::default_initializer(IType const *type)
{
    return m_node_factory.create_constant(default_initializer_value(type));
}

// Creates a default initializer for the given type.
IValue const *DAG_builder::default_initializer_value(IType const *type)
{
    switch (type->get_kind()) {
    case IType::TK_ALIAS:
        return default_initializer_value(type->skip_type_alias());
    case IType::TK_BOOL:
        return m_value_factory.create_bool(false);
    case IType::TK_INT:
        return m_value_factory.create_int(0);
    case IType::TK_ENUM:
        return m_value_factory.create_enum(cast<IType_enum>(type), 0);
    case IType::TK_FLOAT:
        return m_value_factory.create_float(0.0);
    case IType::TK_DOUBLE:
        return m_value_factory.create_double(0.0);
    case IType::TK_STRING:
        return m_value_factory.create_string("");
    case IType::TK_VECTOR:
        {
            IType_vector const *v_tp = cast<IType_vector>(type);
            IType_atomic const *e_tp = v_tp->get_element_type();
            IValue const       *elem = default_initializer_value(e_tp);

            VLA<IValue const *> elems(get_allocator(), v_tp->get_size());

            for (int i = 0, n = v_tp->get_size(); i < n; ++i) {
                elems[i] = elem;
            }
            return m_value_factory.create_vector(v_tp, elems.data(), elems.size());
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_tp = cast<IType_matrix>(type);
            IType_vector const *e_tp = m_tp->get_element_type();
            IValue const       *elem = default_initializer_value(e_tp);

            VLA<IValue const *> elems(get_allocator(), m_tp->get_columns());

            for (int i = 0, n = m_tp->get_columns(); i < n; ++i) {
                elems[i] = elem;
            }
            return m_value_factory.create_matrix(m_tp, elems.data(), elems.size());
        }
    case IType::TK_ARRAY:
        {
            IType_array const *a_tp = cast<IType_array>(type);
            IType const       *e_tp = a_tp->get_element_type();
            IValue const      *elem = default_initializer_value(e_tp);

            VLA<IValue const *> elems(get_allocator(), a_tp->get_size());

            for (int i = 0, n = a_tp->get_size(); i < n; ++i) {
                elems[i] = elem;
            }
            return m_value_factory.create_array(a_tp, elems.data(), elems.size());
        }
    case IType::TK_COLOR:
        {
            IValue_float const *elem =
                cast<IValue_float>(default_initializer_value(m_type_factory.create_float()));

            return m_value_factory.create_rgb_color(elem, elem, elem);
        }
    case IType::TK_STRUCT:
        {
            IType_struct const *s_tp = cast<IType_struct>(type);

            VLA<IValue const *> elems(get_allocator(), s_tp->get_compound_size());

            for (int i = 0, n = s_tp->get_compound_size(); i < n; ++i) {
                IType const  *e_tp = s_tp->get_compound_type(i);
                IValue const *elem = default_initializer_value(e_tp);
                elems[i] = elem;
            }
            return m_value_factory.create_struct(s_tp, elems.data(), elems.size());
        }
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        return m_value_factory.create_invalid_ref(cast<IType_reference>(type));
    case IType::TK_FUNCTION:
    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        return m_value_factory.create_bad();
    }
    MDL_ASSERT(!"unsupported type kind");
    return m_value_factory.create_bad();
}

// Process a resource and update a relative resource URL if necessary.
IValue_resource const *DAG_builder::process_resource_urls(
    IValue_resource const *res,
    Position const        &pos)
{
    char const *url = res->get_string_value();
    if (url != NULL && url[0] != '\0') {

        // check for special marker for non-resolved  weak relative paths, keep it.
        for (size_t i = 0, n = strlen(url); i < n; ++i) {
            if (url[i] == ':' && i+1 < n && url[i+1] == ':')
                return res;
        }
        // non-empty URL, check if relative
        if (url[0] != '/') {
            // found a relative url. We might be in the context of a module PA::A, but compile
            // a material of PB::B. If PA != PB, then relative urls will not work and must be
            // updated.
            IModule const *owner = tos_module();

            return retarget_resource_url(
                res,
                pos,
                get_allocator(),
                m_type_factory,
                m_value_factory,
                owner,
                m_resolver);
        }
    }
    return res;
}


} // mdl
} // mi

