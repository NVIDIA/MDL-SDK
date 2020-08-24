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

#include "compilercore_cc_conf.h"

#include "compilercore_mdl.h"
#include "compilercore_allocator.h"
#include "compilercore_analysis.h"
#include "compilercore_call_graph.h"
#include "compilercore_dependency_graph.h"
#include "compilercore_errors.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"
#include "compilercore_mdl.h"

namespace mi {
namespace mdl {

typedef Store<bool> Flag_store;

/// Get the combined type modifiers of a type.
static IType::Modifiers get_type_modifiers(IType const *type)
{
    IType::Modifiers res = type->get_type_modifiers();

    if (IType_array const *a_tp = as<IType_array>(type)) {
        IType const *e_tp = a_tp->get_element_type();
        res |= get_type_modifiers(e_tp);
    }
    return res;
}

///
/// Helper RAII class
///
template<typename S>
class Stack_scope {
public:
    Stack_scope(S &st, Dependence_graph::Id_type id)
    : m_stack(st)
    {
        st.push(id);
    }

    ~Stack_scope() { m_stack.pop(); }

private:
    S &m_stack;
};

///
/// Helper analysis pass that will check the auto-types.
///
class AT_check : public Analysis
{
public:
    /// Constructor.
    ///
    /// \param compiler    the MDL compiler
    /// \param module      the current module
    /// \param ctx         the thread context
    /// \param dg          the type dependence graph
    /// \param inside_mat  true, if we currently check a function
    AT_check(
        MDL              *compiler,
        Module           &module,
        Thread_context   &ctx,
        Dependence_graph &dg,
        bool             inside_mat)
    : Analysis(compiler, module, ctx)
    , m_dg(dg)
    , m_inside_mat(inside_mat)
    {
    }

    /// Process the given function.
    ///
    /// \param decl  the declaration of the function
    void process(IDeclaration_function *decl) {
        visit(decl);
    }

private:
    /// Fix the type by applying type modifiers.
    ///
    /// \param type  the type to fix
    /// \param mod   the modifier to apply
    IType const *fix_type(IType const *type, IType::Modifiers mod);

    /// Implements the supremum of two type modifiers.
    ///
    /// \param a  left operand
    /// \param b  right operand
    ///
    /// \return the sup(a, b)
    static IType::Modifiers supremum(IType::Modifiers a, IType::Modifiers b);

    /// Implements the Infimum of two type modifiers.
    ///
    /// \param a  left operand
    /// \param b  right operand
    ///
    /// \return the sup(a, b)
    static IType::Modifiers infimum(IType::Modifiers a, IType::Modifiers b);

    /// Get the type modifier of an expression.
    ///
    /// \param expr  the expression
    static IType::Modifiers get_type_modifier(IExpression const *expr);

    IExpression *post_visit(IExpression_reference *ref) MDL_FINAL;
    IExpression *post_visit(IExpression_unary *un_expr) MDL_FINAL;
    IExpression *post_visit(IExpression_binary *bin_expr) MDL_FINAL;
    IExpression *post_visit(IExpression_conditional *cond_expr) MDL_FINAL;
    IExpression *post_visit(IExpression_call *call_expr) MDL_FINAL;

    void post_visit(IParameter *param) MDL_FINAL;

    bool pre_visit(IDeclaration_variable *var_decl) MDL_FINAL;

private:
    /// The dependence graph.
    Dependence_graph &m_dg;

    /// True, if we are inside a material.
    bool m_inside_mat;
};


// Fix the type by applying type modifiers.
IType const *AT_check::fix_type(IType const *type, IType::Modifiers mod)
{
    if (mod == IType::MK_VARYING)
        type = m_tc.decorate_type(type->skip_type_alias(), IType::MK_VARYING);
    else if (mod == IType::MK_UNIFORM || mod == IType::MK_CONST)
        type = m_tc.decorate_type(type->skip_type_alias(), IType::MK_UNIFORM);
    return type;
}

// Implements the supremum of two type modifiers.
IType::Modifiers AT_check::supremum(IType::Modifiers a, IType::Modifiers b)
{
    // the order is MK_CONST < MK_UNIFORM < MK_NONE (== AUTO) < MK_VARYING
    if (a == IType::MK_NONE) {
        return b == IType::MK_VARYING ? IType::MK_VARYING : IType::MK_NONE;
    }
    if (b == IType::MK_NONE) {
        return a == IType::MK_VARYING ? IType::MK_VARYING : IType::MK_NONE;
    }
    return a > b ? a : b;
}

// Implements the Infimum of two type modifiers.
IType::Modifiers AT_check::infimum(IType::Modifiers a, IType::Modifiers b)
{
    // the order is MK_CONST < MK_UNIFORM < MK_NONE (== AUTO) < MK_VARYING
    if (a == IType::MK_NONE) {
        return b == IType::MK_VARYING ? IType::MK_NONE : b;
    }
    if (b == IType::MK_NONE) {
        return a == IType::MK_VARYING ? IType::MK_NONE : a;
    }
    return a < b ? a : b;
}

// Get the type modifier of an expression.
IType::Modifiers AT_check::get_type_modifier(IExpression const *expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_LITERAL:
        // literals are always const
        return IType::MK_CONST;
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *ref = cast<IExpression_reference>(expr);
            IDefinition const *def = ref->get_definition();
            switch (def->get_kind()) {
            case IDefinition::DK_CONSTANT:
            case IDefinition::DK_ENUM_VALUE:
            case IDefinition::DK_ARRAY_SIZE:
                // constants, enum values, and array sizes are always const
                return IType::MK_CONST;
            default:
                break;
            }
        }
        break;
    default:
        break;
    }
    IType const *type = expr->get_type();
    if (is<IType_error>(type)) {
        // assume errors to be const, so no further error messages are generated
        return IType::MK_CONST;
    }
    return get_type_modifiers(type);
}

// end of a reference expression
IExpression *AT_check::post_visit(IExpression_reference *ref)
{
    if (!ref->is_array_constructor()) {
        // get the definition from the type name
        IDefinition const *def = ref->get_definition();

        Definition::Kind kind = def->get_kind();
        if (kind == Definition::DK_VARIABLE) {
            // the definition type was updated
            ref->set_type(def->get_type());
        }
    }
    return ref;
}

// end of unary expression
IExpression *AT_check::post_visit(IExpression_unary *un_expr)
{
    IType const *res_type = un_expr->get_type();
    if (is<IType_error>(res_type)) {
        // already error
        return un_expr;
    }

    // all unary expressions are uniform 
    IExpression const *arg = un_expr->get_argument();
    IType::Modifiers res_mod = get_type_modifier(arg);

    res_type = fix_type(res_type, res_mod);
    un_expr->set_type(res_type);
    return un_expr;
}

// end of a binary expression
IExpression *AT_check::post_visit(IExpression_binary *bin_expr)
{
    IType const *res_type = bin_expr->get_type();
    if (is<IType_error>(res_type)) {
        // already error
        return bin_expr;
    }

    IExpression_binary::Operator op   = bin_expr->get_operator();
    IExpression const            *lhs = bin_expr->get_left_argument();
    IExpression const            *rhs = bin_expr->get_right_argument();

    IType::Modifiers res_mod = IType::MK_NONE;

    switch (op) {
    case IExpression_binary::OK_SELECT:
        // the type of the select expression infimum of the lhs and the rhs type
        res_mod = infimum(get_type_modifier(lhs), get_type_modifier(rhs));
        break;
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
        // all those operators are uniform
        res_mod = supremum(get_type_modifier(lhs), get_type_modifier(rhs));
        break;
    case IExpression_binary::OK_SEQUENCE:
        // the type of a sequence expression is the type of the last one
        res_mod = get_type_modifier(rhs);
        break;
    }

    res_type = fix_type(res_type, res_mod);
    bin_expr->set_type(res_type);
    return bin_expr;
}

/// Check if the given type needs a uniform condition
///
/// \param type  the type to check
///
/// \return the type that requires that or NULL if no uniform condition is needed
static IType const *needs_uniform_condition(IType const *type)
{
    for (;;) {
        switch (type->get_kind()) {
        case IType::TK_ALIAS:
            type = cast<IType_alias>(type)->get_aliased_type();
            continue;

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_ENUM:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
        case IType::TK_COLOR:
        case IType::TK_FUNCTION:
        case IType::TK_TEXTURE:
        case IType::TK_INCOMPLETE:
        case IType::TK_ERROR:
            return NULL;

        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_VDF:
        case IType::TK_EDF:
            // resource types require a uniform condition
            return type;

        case IType::TK_STRUCT:
            {
                IType_struct const *s_type = cast<IType_struct>(type);
                if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL) {
                    // material need requires a uniform condition
                    return s_type;
                }
                // do a deep check
                for (int i = 0, n = s_type->get_field_count(); i < n; ++i) {
                    ISymbol const *f_sym;
                    IType const   *f_type;

                    s_type->get_field(i, f_type, f_sym);

                    IType const *bad_type = needs_uniform_condition(f_type);
                    if (bad_type != NULL)
                        return bad_type;
                }
                return NULL;
            }
            break;

        case IType::TK_ARRAY:
            {
                IType const *e_type = cast<IType_array>(type)->get_element_type();
                return needs_uniform_condition(e_type);
            }
        }
        MDL_ASSERT(!"unsupported type kind");
        return NULL;
    }
}


// end of a conditional expression
IExpression *AT_check::post_visit(IExpression_conditional *cond_expr)
{
    IType const *res_type = cond_expr->get_type();
    if (is<IType_error>(res_type)) {
        // already error
        return cond_expr;
    }

    IExpression const *cond = cond_expr->get_condition();

    IType::Modifiers res_mod = get_type_modifier(cond);

    if (m_inside_mat) {
        if ((res_mod & (IType::MK_CONST | IType::MK_UNIFORM)) == 0) {
            IType const *bad_type = needs_uniform_condition(res_type);
            if (bad_type != NULL) {
                // ternary operator on several types must have uniform condition
                error_mdl_11(
                    TERNARY_COND_NOT_UNIFORM,
                    cond_expr->access_position(),
                    Error_params(*this).add(res_type));
                if (bad_type->skip_type_alias() != res_type->skip_type_alias()) {
                    add_note(
                        TYPE_CONTAINS_FORBIDDEN_SUBTYPE,
                        cond_expr->access_position(),
                        Error_params(*this)
                            .add(res_type)
                            .add(bad_type));
                }
            }
        }
    }

    IExpression const *true_expr = cond_expr->get_true();

    res_mod = supremum(res_mod, get_type_modifier(true_expr));

    IExpression const *false_expr = cond_expr->get_false();

    res_mod = supremum(res_mod, get_type_modifier(false_expr));

    res_type = fix_type(res_type, res_mod);
    cond_expr->set_type(res_type);
    return cond_expr;
}

// end of a call
IExpression *AT_check::post_visit(IExpression_call *call_expr)
{
    IType const *res_type = call_expr->get_type();
    if (is<IType_error>(res_type)) {
        // already error
        return call_expr;
    }

    IType::Modifiers res_mod = get_type_modifiers(res_type);
    if (res_mod == IType::MK_NONE) {
        // auto typed, map to UNIFORM first and let supremum do the rest
        res_mod = IType::MK_UNIFORM;
    }

    IExpression_reference const *callee = cast<IExpression_reference>(call_expr->get_reference());
    Definition const            *f_def  = impl_cast<Definition>(callee->get_definition());

    if (callee->is_array_constructor()) {
        // array constructors are uniform calls with auto-typed parameters
        int n_args = call_expr->get_argument_count();

        for (int i = 0; i < n_args; ++i) {
            IArgument const   *arg      = call_expr->get_argument(i);
            IExpression const *arg_expr = arg->get_argument_expr();
            IType::Modifiers  a_mod     = get_type_modifier(arg_expr);

            // auto-type parameter
            res_mod = supremum(res_mod, a_mod);
        }
    } else if (f_def->get_kind() == Definition::DK_ERROR) {
        // already error
        return call_expr;
    } else if (f_def->has_flag(Definition::DEF_IS_VARYING)) {
        // varying function call
        res_mod = IType::MK_VARYING;

        IType_function const *f_type = cast<IType_function>(f_def->get_type());
        int n_params = f_type->get_parameter_count();

        // assume that all calls are flattened and reordered
        MDL_ASSERT(n_params == call_expr->get_argument_count() && "call not flattened");

        for (int i = 0; i < n_params; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;

            f_type->get_parameter(i, p_type, p_sym);

            IType::Modifiers p_mod = get_type_modifiers(p_type);
            if (p_mod & IType::MK_VARYING) {
                // this parameter can be varying, so it always matches
                continue;
            }

            IArgument const   *arg      = call_expr->get_argument(i);
            IExpression const *arg_expr = arg->get_argument_expr();
            IType::Modifiers  a_mod     = get_type_modifier(arg_expr);

            if (p_mod & IType::MK_UNIFORM) {
                // this argument must be uniform

                if (a_mod & IType::MK_VARYING) {
                    if (IArgument_named const *narg = as<IArgument_named>(arg)) {
                        ISimple_name const *sname = narg->get_parameter_name();
                        ISymbol const      *sym   = sname->get_symbol();

                        error_mdl_13(
                            CANNOT_CONVERT_ARG_TYPE,
                            arg->access_position(),
                            Error_params(*this)
                            .add(f_def->get_sym())
                            .add(sym)
                            .add(arg_expr->get_type())
                            .add(p_type));
                    } else {
                        error_mdl_13(
                            CANNOT_CONVERT_POS_ARG_TYPE,
                            arg->access_position(),
                            Error_params(*this)
                            .add(f_def->get_sym())
                            .add_numword(i + 1)  // count from 1
                            .add(arg_expr->get_type())
                            .add(p_type));
                    }
                } else if (a_mod == IType::MK_NONE) {
                    // an uniform parameter gets an parameter depend argument
                    if (IArgument_named const *narg = as<IArgument_named>(arg)) {
                        ISimple_name const *sname = narg->get_parameter_name();
                        ISymbol const      *sym   = sname->get_symbol();

                        error_mdl_13(
                            CANNOT_CONVERT_ARG_TYPE_PARAM_DEP,
                            arg->access_position(),
                            Error_params(*this)
                            .add(f_def->get_sym())
                            .add(sym)
                            .add(arg_expr->get_type())
                            .add(p_type));
                    } else {
                        error_mdl_13(
                            CANNOT_CONVERT_POS_ARG_TYPE_PARAM_DEP,
                            arg->access_position(),
                            Error_params(*this)
                            .add(f_def->get_sym())
                            .add_numword(i + 1)  // count from 1
                            .add(arg_expr->get_type())
                            .add(p_type));
                    }
                } else {
                    // fine, uniform expression on uniform parameter
                    MDL_ASSERT(a_mod & (IType::MK_UNIFORM|IType::MK_CONST));
                }
            } else {
                // auto-type parameter
            }
        }
    } else {
        // assume uniform otherwise: if can fail on recursive loops (error anyway) or
        // wrong stdlib, which will be check earlier ...
        IType_function const *f_type = cast<IType_function>(f_def->get_type());

        int n_params = f_type->get_parameter_count();

        // assume that all calls are flattened and reordered
        MDL_ASSERT(n_params == call_expr->get_argument_count() && "call not flattened");

        for (int i = 0; i < n_params; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;

            f_type->get_parameter(i, p_type, p_sym);

            IType::Modifiers p_mod = get_type_modifiers(p_type);
            if (p_mod & IType::MK_VARYING) {
                // this parameter can be varying, so it always matches
                continue;
            }

            IArgument const   *arg      = call_expr->get_argument(i);
            IExpression const *arg_expr = arg->get_argument_expr();
            IType::Modifiers  a_mod     = get_type_modifier(arg_expr);

            if (p_mod & IType::MK_UNIFORM) {
                // this argument must be uniform

                if (a_mod & IType::MK_VARYING) {
                    if (IArgument_named const *narg = as<IArgument_named>(arg)) {
                        ISimple_name const *sname = narg->get_parameter_name();
                        ISymbol const      *sym   = sname->get_symbol();

                        error_mdl_11(
                            CANNOT_CONVERT_ARG_TYPE,
                            arg->access_position(),
                            Error_params(*this)
                                .add(f_def->get_sym())
                                .add(sym)
                                .add(arg_expr->get_type())
                                .add(p_type));
                    } else {
                        error_mdl_11(
                            CANNOT_CONVERT_POS_ARG_TYPE,
                            arg->access_position(),
                            Error_params(*this)
                                .add(f_def->get_sym())
                                .add_numword(i + 1)  // count from 1
                                .add(arg_expr->get_type())
                                .add(p_type));
                    }
                } else if (a_mod == IType::MK_NONE) {
                    // an uniform parameter gets an parameter depend argument
                    if (IArgument_named const *narg = as<IArgument_named>(arg)) {
                        ISimple_name const *sname = narg->get_parameter_name();
                        ISymbol const      *sym   = sname->get_symbol();

                        error_mdl_11(
                            CANNOT_CONVERT_ARG_TYPE_PARAM_DEP,
                            arg->access_position(),
                            Error_params(*this)
                            .add(f_def->get_sym())
                            .add(sym)
                            .add(arg_expr->get_type())
                            .add(p_type));
                    } else {
                        error_mdl_11(
                            CANNOT_CONVERT_POS_ARG_TYPE_PARAM_DEP,
                            arg->access_position(),
                            Error_params(*this)
                            .add(f_def->get_sym())
                            .add_numword(i + 1)  // count from 1
                            .add(arg_expr->get_type())
                            .add(p_type));
                    }
                } else {
                    // fine, uniform expression on uniform parameter
                    MDL_ASSERT(a_mod & (IType::MK_UNIFORM|IType::MK_CONST));
                    res_mod = supremum(res_mod, a_mod);
                }
            } else {
                // auto-type parameter
                res_mod = supremum(res_mod, a_mod);
            }
        }

        IType const *f_ret_type = f_type->get_return_type();
        IType::Modifiers f_ret_mod = get_type_modifiers(f_ret_type);
        if (f_ret_mod != IType::MK_NONE) {
            // the return type is explicitly marked, use it
            res_mod = f_ret_mod;
        } else {
            if (n_params == 0) {
                // a uniform function without parameter always returns uniform
                res_mod = IType::MK_UNIFORM;
            }
        }
    }

    res_type = fix_type(res_type, res_mod);
    call_expr->set_type(res_type);
    return call_expr;
}

// end of an parameter
void AT_check::post_visit(IParameter *param)
{
    IType_name const *tname = param->get_type_name();

    IType const *type = tname->get_type();
    IType::Kind tkind = type->skip_type_alias()->get_kind();

    if (tkind == IType::TK_TEXTURE ||
        tkind == IType::TK_LIGHT_PROFILE ||
        tkind == IType::TK_BSDF_MEASUREMENT) {
        // check that resource parameters are uniform
        IType::Modifiers mod = get_type_modifiers(type);
        if (!(mod & IType::MK_UNIFORM)) {
            ISimple_name const *pname = param->get_name();
            error(
                RESOURCE_PARAMETER_NOT_UNIFORM,
                param->access_position(),
                Error_params(*this)
                    .add(pname->get_symbol())
                    .add(type->skip_type_alias()));
        }
    }
}

// start of a variable declaration
bool AT_check::pre_visit(IDeclaration_variable *var_decl)
{
    IType_name const *t_name = var_decl->get_type_name();
    visit(t_name);

    for (size_t i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
        ISimple_name     *v_name = const_cast<ISimple_name *>(var_decl->get_variable_name(i));
        Definition const *v_def  = impl_cast<Definition>(v_name->get_definition());

        if (IExpression const *init = var_decl->get_variable_init(i)) {
            visit(init);
        }

        if (IAnnotation_block const *anno = var_decl->get_annotations(i)) {
            visit(anno);
        }

        IType const *v_type = v_def->get_type();
        if (is<IType_error>(v_type))
            continue;

        Dependence_graph::Node *v_node = m_dg.get_node(v_def);
        MDL_ASSERT(v_node != NULL && "local has no node in dependence graph");

        IType::Modifiers            v_mode = get_type_modifiers(v_type);
        Dependence_graph::Auto_type at     = v_node->get_auto_type();

        switch (at) {
        case Dependence_graph::AT_TOP:
            /*FALLTHROUGH*/
        case Dependence_graph::AT_UNIFORM:
            if (v_mode == IType::MK_NONE) {
                v_type = fix_type(v_type, IType::MK_UNIFORM);
                const_cast<Definition *>(v_def)->set_type(v_type);
            }
            break;
        case Dependence_graph::AT_PARAM:
            if (v_mode & IType::MK_VARYING) {
                // ok, assignment of "param-depend" type to varying
            } else if (v_mode & IType::MK_UNIFORM) {
                error_mdl_11(
                    VARIABLE_DEPENDS_ON_AUTOTYPED_PARAMETER,
                    v_name->access_position(),
                    Error_params(*this)
                        .add(v_def->get_sym())
                        .add(v_type));
                // make it "param-depend" for the rest of the calculation
                v_type = v_type->skip_type_alias();
                const_cast<Definition *>(v_def)->set_type(v_type);
            }
            break;
        case Dependence_graph::AT_VARYING:
            if (v_mode & IType::MK_VARYING) {
                // ok, already varying
            } else if (v_mode & IType::MK_UNIFORM) {
                error_mdl_11(
                    VARIABLE_DEPENDS_ON_VARYING_VALUE,
                    v_name->access_position(),
                    Error_params(*this)
                    .add(v_def->get_sym())
                    .add(v_type));
                // make it varying for the rest of the calculation
                v_type = fix_type(v_type, IType::MK_VARYING);
                const_cast<Definition *>(v_def)->set_type(v_type);
            } else if (v_mode == IType::MK_NONE) {
                // make it varying for the rest of the calculation
                v_type = fix_type(v_type, IType::MK_VARYING);
                const_cast<Definition *>(v_def)->set_type(v_type);
            }
            break;
        }
    }

    // don't visit children anymore
    return false;
}

// ----------------------------------------------------------------------------------

// Constructor.
AT_analysis::AT_visitor::AT_visitor(AT_analysis &ana)
: m_ana(ana)
, m_curr_mod_id(ana.m_module.get_unique_id())
{
}

// Visit a node of the call graph.
void AT_analysis::AT_visitor::visit_cg_node(Call_node *node, ICallgraph_visitor::Order order)
{
    if (order == ICallgraph_visitor::POST_ORDER) {
        Definition const *def     = node->get_definition();
        Definition const *def_def = def->get_definite_definition();

        if (def_def != NULL)
            def = def_def;

        if (def->get_original_import_idx() == 0 && !def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
            if (def->get_owner_module_id() != m_curr_mod_id) {
                // FIXME: found an entity that will be auto-imported later. Ignore it,
                // it's from another module.
                return;
            }
            m_ana.process_function(def);
        }
    }
}

// Run the auto typing analysis on this module.
void AT_analysis::run(
    MDL              *compiler,
    Module           &module,
    Thread_context   &ctx,
    Call_graph const &cg)
{
    AT_analysis ana(compiler, module, ctx, cg);

    AT_analysis::AT_visitor visitor(ana);

    Call_graph_walker::walk(cg, &visitor);
}

AT_analysis::AT_analysis(
    MDL              *compiler,
    Module           &module,
    Thread_context   &ctx,
    Call_graph const &cg)
: Base(compiler, module, ctx)
, m_arena(module.get_allocator())
, m_builder(m_arena)
, m_cg(cg)
, m_dg(NULL)
, m_assignment_stack(module.get_allocator())
, m_control_stack(Dependency_stack::container_type(module.get_allocator()))
, m_context_stack(Dependency_stack::container_type(module.get_allocator()))
, m_loop_stack(Dependency_stack::container_type(module.get_allocator()))
, m_has_varying_call(false)
, m_inside_def_arg(false)
, m_curr_func_is_uniform(false)
, m_opt_dump_dg(
    compiler->get_compiler_bool_option(
        &ctx, MDL::option_dump_dependence_graph, /*def_value=*/false))
{
}

/// Follow uplinks and loop-exists to collect and add all control dependence nodes.
///
/// \param dg       the dependence graph
/// \param node_id  the ID of the source node (that is dependent of control flow)
/// \param boundary the ID of the control node that is the control boundary of node_id
/// \param ctrl_id  the ID of a control node that controls node_id
static void add_extra_control_dependence(
    Dependence_graph          *dg,
    size_t                    node_id,
    Dependence_graph::Id_type boundary,
    Dependence_graph::Id_type ctrl_id)
{
    // add all control dependence nodes until we reach the boundary
    while (boundary != ctrl_id) {
        dg->add_dependency(node_id, ctrl_id);

        Dependence_graph::Node const *ctrl_node = dg->get_node(ctrl_id);

        // check for loop exits
        for (Dependence_graph::Node::Edge_iterator
            it(ctrl_node->loop_exits_begin()), end(ctrl_node->loop_exits_end());
            it != end;
            ++it)
        {
            // Note: an loop exit must ALWAYS be part of a loop, so stop there,
            // else we got an endless loop
            add_extra_control_dependence(dg, node_id, ctrl_id, *it);
        }

        // check for uplinks
        ctrl_id = ctrl_node->get_control_uplink();
        if (ctrl_id == 0)
            break;
    }
}

// Add control dependence to the given node is one exists.
void AT_analysis::add_control_dependence(size_t node_id)
{
    if (m_control_stack.empty())
        return;

    Dependence_graph::Id_type ctrl_id = m_control_stack.top();

    Dependence_graph::Node const *n = m_dg->get_node(node_id);
    MDL_ASSERT(n->get_kind() != Dependence_graph::Node::NK_CONTROL);

    // handle uplinks and loop exists
    add_extra_control_dependence(m_dg, node_id, n->get_control_boundary(), ctrl_id);
}

// Set the control dependence uplink to the given node if one exists.
void AT_analysis::set_control_dependence_uplink(size_t node_id)
{
    if (m_control_stack.empty())
        return;

    Dependence_graph::Id_type ctrl_id = m_control_stack.top();

    Dependence_graph::Node *n = m_dg->get_node(node_id);

    n->set_control_uplink(ctrl_id);
}

/// Sets a function qualifier into the declaration (and its prototype if exists) AST.
///
/// \param decl  a function declaration
/// \param def   its definition
/// \param qual  the qualifier to set
static void set_function_qualifier(
    IDeclaration_function *decl,
    Definition const      *def,
    Qualifier             qual)
{
    decl->set_qualifier(qual);
    IDeclaration_function const *proto_type =
        cast<IDeclaration_function>(def->get_prototype_declaration());
    if (proto_type != NULL) {
        const_cast<IDeclaration_function *>(proto_type)->set_qualifier(qual);
    }
}

// Use the calculated auto-types to check the AST.
void AT_analysis::check_auto_types(IDeclaration_function *decl)
{
    Definition const *def = impl_cast<Definition>(decl->get_definition());
    if (is_error(def))
        return;

    IType_function const *f_type   = cast<IType_function>(def->get_type());
    IType const          *ret_type = f_type->get_return_type();
    bool                 in_mat    = ret_type->skip_type_alias() == m_tc.material_type;

    if (!in_mat) {
        // only check this for real functions, not for materials

        IType::Modifiers ret_mod = get_type_modifiers(ret_type);

        Dependence_graph::Node const *ret_val = m_dg->get_node(return_value_id);

        if (ret_mod & IType::MK_UNIFORM) {
            switch (ret_val->get_auto_type()) {
            case Dependence_graph::AT_TOP:
                // depends only on constants, good
                break;
            case Dependence_graph::AT_UNIFORM:
                // return value will be uniform, good
                break;
            case Dependence_graph::AT_PARAM:
                // return value depends on parameters, bad
                error(
                    UNIFORM_RESULT_DEPENDS_ON_AUTOTYPED_PARAMETER,
                    decl->get_return_type_name()->access_position(),
                    Error_params(*this).add_signature(def));
                break;
            case Dependence_graph::AT_VARYING:
                // return value is varying, bad
                error(
                    UNIFORM_RESULT_IS_VARYING,
                    decl->get_return_type_name()->access_position(),
                    Error_params(*this).add_signature(def));
                break;
            }
        }

        bool is_uniform = def->has_flag(Definition::DEF_IS_UNIFORM);
        bool is_varying = def->has_flag(Definition::DEF_IS_VARYING);

        // check the return value first
        if (!(is_uniform|is_varying)) {
            switch (ret_val->get_auto_type()) {
            case Dependence_graph::AT_TOP:
                // depends only on constants
                /*FALLTHROUGH*/
            case Dependence_graph::AT_UNIFORM:
                // return value will be uniform
                /*FALLTHROUGH*/
            case Dependence_graph::AT_PARAM:
                if (!m_has_varying_call) {
                    // the function is auto-typed AND the return value has "parameter-dependent"
                    // return type, hence it is uniform
                    const_cast<Definition *>(def)->set_flag(Definition::DEF_IS_UNIFORM);
                    set_function_qualifier(decl, def, FQ_UNIFORM);
                    break;
                }
                /*FALLTHROUGH*/
            case Dependence_graph::AT_VARYING:
                // function calls either a varying function OR the return value is varying
                const_cast<Definition *>(def)->set_flag(Definition::DEF_IS_VARYING);
                set_function_qualifier(decl, def, FQ_VARYING);
                break;
            }
        } else if (is_uniform) {
            // varying calls are already reported ...
            if (ret_val->get_auto_type() == Dependence_graph::AT_VARYING) {
                if (ret_mod & IType::MK_VARYING) {
                    // error already reported
                } else {
                    error_mdl_11(
                        RESULT_OF_UNIFORM_FUNCTION_IS_VARYING,
                        decl->get_return_type_name()->access_position(),
                        Error_params(*this).add_signature(def));
                }
            }
        } else if (is_varying) {
            if (!m_has_varying_call && ret_val->get_auto_type() < Dependence_graph::AT_VARYING) {
                // useless varying
                warning(
                    NONVARYING_RESULT_OF_VARYING_FUNCTION,
                    decl->get_return_type_name()->access_position(),
                    Error_params(*this).add_signature(def));
            }
        }
    }

    // finally check and fix the types inside the function
    AT_check at_checker(m_compiler, m_module, m_ctx, *m_dg, in_mat);

    at_checker.process(decl);
}

// Process a function given by its definition.
void AT_analysis::process_function(Definition const *def)
{
    if (IDeclaration const *decl = def->get_declaration()) {
        // found a function in the current module that is defined there, visit it

        Flag_store has_varying_call(m_has_varying_call, false);
        Flag_store is_uniform(m_curr_func_is_uniform, def->has_flag(Definition::DEF_IS_UNIFORM));

        visit(decl);
    }
}

// start of a function
bool AT_analysis::pre_visit(IDeclaration_function *decl)
{
    Definition const *fkt_def = impl_cast<Definition>(decl->get_definition());

    if (is_error(fkt_def)) {
        // something really bad here, even the type is broken. stop traversal.
        return false;
    }

    // create the dependence graph and enter it
    m_dg = m_builder.create<Dependence_graph>(&m_arena, fkt_def->get_sym()->get_name());

    Dependence_graph::Node *node = NULL;

    // create the return node, it has always the ID <return_value_id>
    IType_function const *fkt_type = cast<IType_function>(fkt_def->get_type());
    IType const          *ret_type = fkt_type->get_return_type();

    node = m_dg->create_aux_node(Dependence_graph::Node::NK_RETURN_VALUE, ret_type);
    MDL_ASSERT(node->get_id() == return_value_id);

    // create the one and only variadic call node
    node = m_dg->create_aux_node(Dependence_graph::Node::NK_VARYING_CALL, NULL);
    MDL_ASSERT(node->get_id() == varying_call_id);

    if (fkt_def->has_flag(Definition::DEF_IS_UNIFORM)) {
        IType::Modifiers mod = get_type_modifiers(ret_type);

        if (mod & IType::MK_VARYING) {
            error(
                UNIFORM_FUNCTION_DECLARED_WITH_VARYING_RESULT,
                decl->get_return_type_name()->access_position(),
                Error_params(*this).add_signature(fkt_def));
        }
    }

    // create all parameter nodes
    for (int i = 0, n = decl->get_parameter_count(); i < n; ++i) {
        IParameter const   *param = decl->get_parameter(i);
        ISimple_name const *sname = param->get_name();
        Definition const   *p_def = impl_cast<Definition>(sname->get_definition());

        node = m_dg->create_param_node(const_cast<Definition *>(p_def));
    }

    // visit children
    return true;
}

// end of a function
void AT_analysis::post_visit(IDeclaration_function *decl)
{
    // all stacks must be empty
    MDL_ASSERT(m_control_stack.empty());
    MDL_ASSERT(m_context_stack.empty());
    MDL_ASSERT(m_loop_stack.empty());

    // if the function is badly broken, we don't have a dependence graph
    if (m_dg != NULL) {
        // do not analyze presets, these are not call function bodies
        if (!decl->is_preset()) {
            if (m_opt_dump_dg) {
                dump_dg("_dg_before");
            }

            // build the auto types
            m_dg->calc_auto_types();

            if (m_opt_dump_dg) {
                dump_dg("_dg_after");
            }

            check_auto_types(decl);
        }

        // This is ugly: the dependence graph is on the Arena but contains STL objects on the
        // heap so call the destructor ...
        m_dg->~Dependence_graph();
        m_dg = NULL;
    }
}

// start of a variable declaration
bool AT_analysis::pre_visit(IDeclaration_variable *decl)
{
    IType_name const *tname = decl->get_type_name();
    visit(tname);

    Dependence_graph::Id_type control_boundary = 0;     
    if (!m_control_stack.empty())
        control_boundary = m_control_stack.top();

    for (size_t i = 0, n = decl->get_variable_count(); i < n; ++i) {
        ISimple_name const *vname   = decl->get_variable_name(i);
        Definition const   *var_def = impl_cast<Definition>(vname->get_definition());

        // create a node for this variable
        Dependence_graph::Node *node = m_dg->create_local_node(
            const_cast<Definition *>(var_def), control_boundary);

        if (IExpression const *init = decl->get_variable_init(i)) {
            // the local variable depends an its initializer
            Stack_scope<Assignment_stack> scope(m_assignment_stack, node->get_id());

            visit(init);
        }

        // ignore annotations completely
    }

    // don't visit children anymore
    return false;
}

// start of an return statement
bool AT_analysis::pre_visit(IStatement_return *stmt)
{
    if (IExpression const *expr = stmt->get_expression()) {
        m_assignment_stack.push(return_value_id);
        visit(expr);
        m_assignment_stack.pop();
    }
    // don't visit children anymore
    return false;
}

// start of an if statement
bool AT_analysis::pre_visit(IStatement_if *stmt)
{
    Dependence_graph::Node *node = m_dg->create_control_node(m_tc.bool_type, stmt);

    // this new node depends on above control flow
    set_control_dependence_uplink(node->get_id());

    // the control node itself depends on the expression inside the condition ...
    {
        Stack_scope<Assignment_stack> scope(m_assignment_stack, node->get_id());

        IExpression const *cond = stmt->get_condition();
        visit(cond);
    }

    // ... and is a control dependence for the statements
    {
        Stack_scope<Dependency_stack> scope(m_control_stack, node->get_id());

        IStatement const *t = stmt->get_then_statement();
        visit(t);

        if (IStatement const *e = stmt->get_else_statement())
            visit(e);
    }

    // do not visit children anymore
    return false;
}

// start of a while statement
bool AT_analysis::pre_visit(IStatement_while *stmt)
{
    Dependence_graph::Node *node = m_dg->create_control_node(m_tc.bool_type, stmt);

    Stack_scope<Dependency_stack> ctx(m_context_stack, node->get_id());
    Stack_scope<Dependency_stack> loop(m_loop_stack, node->get_id());

    // this new node depends on above control flow
    set_control_dependence_uplink(node->get_id());

    // the control node itself depends on the expression inside the condition ...
    {
        Stack_scope<Assignment_stack> scope(m_assignment_stack, node->get_id());

        IExpression const *cond = stmt->get_condition();
        visit(cond);
    }

    // ... and is a control dependence for the statements
    {
        Stack_scope<Dependency_stack> scope(m_control_stack, node->get_id());

        IStatement const *body = stmt->get_body();
        visit(body);
    }

    // do not visit children anymore
    return false;
}

// start of a do ... while statement
bool AT_analysis::pre_visit(IStatement_do_while *stmt)
{
    Dependence_graph::Node *node = m_dg->create_control_node(m_tc.bool_type, stmt);

    Stack_scope<Dependency_stack> ctx(m_context_stack, node->get_id());
    Stack_scope<Dependency_stack> loop(m_loop_stack, node->get_id());

    // this new node depends on above control flow
    set_control_dependence_uplink(node->get_id());

    // the control node itself is a control dependence for the statements
    {
        Stack_scope<Dependency_stack> scope(m_control_stack, node->get_id());

        IStatement const *body = stmt->get_body();
        visit(body);
    }

    // ... and depends on the expression inside the condition
    {
        Stack_scope<Assignment_stack> scope(m_assignment_stack, node->get_id());

        IExpression const *cond = stmt->get_condition();
        visit(cond);
    }

    // do not visit children anymore
    return false;
}

// start of a for statement
bool AT_analysis::pre_visit(IStatement_for *stmt)
{
    Dependence_graph::Node *node = m_dg->create_control_node(m_tc.bool_type, stmt);

    Stack_scope<Dependency_stack> ctx(m_context_stack, node->get_id());
    Stack_scope<Dependency_stack> loop(m_loop_stack, node->get_id());

    // this new node depends on above control flow
    set_control_dependence_uplink(node->get_id());

    if (IStatement const *init = stmt->get_init()) {
        // the init statement is executed in any case
        visit(init);
    }

    if (IExpression const *cond = stmt->get_condition()) {
        // the control node itself depends on the expression inside the condition ...
        Stack_scope<Assignment_stack> scope(m_assignment_stack, node->get_id());

        visit(cond);
    }

    // the update expression AND the body are control dependent
    {
        Stack_scope<Dependency_stack> scope(m_control_stack, node->get_id());

        if (IExpression const *upd = stmt->get_update())
            visit(upd);

        IStatement const *body = stmt->get_body();
        visit(body);
    }

    // do not visit children anymore
    return false;
}

// start of a switch statement
bool AT_analysis::pre_visit(IStatement_switch *stmt)
{
    IExpression const *cond = stmt->get_condition();

    Dependence_graph::Node *node = m_dg->create_control_node(cond->get_type(), stmt);

    Stack_scope<Dependency_stack> ctx(m_context_stack, node->get_id());

    // this new node depends on above control flow
    set_control_dependence_uplink(node->get_id());

    // the control node itself depends on the expression inside the condition ...
    {
        Stack_scope<Assignment_stack> scope(m_assignment_stack, node->get_id());

        visit(cond);
    }

    // ... and is a control dependence for the statements
    {
        Stack_scope<Dependency_stack> scope(m_control_stack, node->get_id());

        for (size_t i = 0, n = stmt->get_case_count(); i < n; ++i) {
            IStatement const *st = stmt->get_case(i);
            visit(st);
        }
    }

    // do not visit children anymore
    return false;
}

// end of a break statement
void AT_analysis::post_visit(IStatement_break *stmt)
{
    if (m_context_stack.empty()) {
        // a break outside any context, ignore it (error)
        return;
    }

    Dependence_graph::Id_type loop_id = m_context_stack.top();

    Dependence_graph::Node const *ctx_node = m_dg->get_node(loop_id);

    if (ctx_node->get_statement()->get_kind() == IStatement::SK_SWITCH) {
        // a break inside a switch, ignore
        return;
    }

    // We found a break inside a loop. If this break depends on a condition,
    // then all variables inside the loop depend on this condition too.
    Dependence_graph::Id_type ctrl_id = m_control_stack.top();

    if (ctrl_id == loop_id) {
        // An unconditional break. Ignore it. We *could* do it a little bit
        // better (for instance in detecting that a loop is not a loop anymore),
        // but it isn't worth doing so and makes is even more complicated to explain
        // how auto-typing works ...
        return;
    }

    m_dg->add_loop_exit(loop_id, ctrl_id);
}

// end of a continue statement
void AT_analysis::post_visit(IStatement_continue *stmt)
{
    if (m_loop_stack.empty()) {
        // a continue outside any loop, ignore it (error)
        return;
    }

    Dependence_graph::Id_type loop_id = m_loop_stack.top();

    // We found a continue inside a loop. If this continue depends on a condition,
    // then all variables inside the loop depend on this condition too.
    Dependence_graph::Id_type ctrl_id = m_control_stack.top();

    if (ctrl_id == loop_id) {
        // An unconditional continue. Ignore it. We *could* do it a little bit
        // better (for instance in detecting that a loop is endless),
        // but it isn't worth doing so and makes is even more complicated to explain
        // how auto-typing works ...
        return;
    }

    // the next might be irritating, but does what we want: Continues does not exit
    // the current loop, but "modify" the value of all later assignment in the loop.
    // At this point we don't know which these are, so ha handle it like an exit
    m_dg->add_loop_exit(loop_id, ctrl_id);
}

// start of a binary expression
bool AT_analysis::pre_visit(IExpression_binary *expr)
{
    IExpression_binary::Operator op = expr->get_operator();

    switch (op) {
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
        // an assignment, build dependencies
        break;
    case IExpression_binary::OK_SEQUENCE:
        // FIXME: only the last expression adds dependencies,
        // the other one is a top expression
        return true;
    default:
        // normal processing
        return true;
    }

    IExpression const *lhs = expr->get_left_argument();

    if (IDefinition const *lhs_def = get_lvalue_base(lhs)) {
        Dependence_graph::Node *lhs_node = m_dg->get_node(lhs_def);
        MDL_ASSERT(lhs_node != NULL && "LValue is not a local");

        Dependence_graph::Id_type lval_id = lhs_node->get_id();

        // this assignment depends on the control flow
        add_control_dependence(lval_id);

        // the lhs depends on the rhs
        {
            Stack_scope<Assignment_stack> scope(m_assignment_stack, lval_id);

            IExpression const *rhs = expr->get_right_argument();
            visit(rhs);
        }

        // we left one assignment. If this was not the top, add
        // a dependency to the upper one
        if (!m_assignment_stack.empty()) {
            Dependence_graph::Id_type src_id = m_assignment_stack.top();

            m_dg->add_dependency(src_id, lval_id);
        }
        // do not visit children anymore
        return false;
    }
    // normal processing
    return true;
}


// start of an unary expression
bool AT_analysis::pre_visit(IExpression_unary *expr)
{
    IExpression_unary::Operator op = expr->get_operator();

    switch (op) {
    case IExpression_unary::OK_PRE_INCREMENT:
    case IExpression_unary::OK_PRE_DECREMENT:
    case IExpression_unary::OK_POST_INCREMENT:
    case IExpression_unary::OK_POST_DECREMENT:
        break;
    default:
        // normal processing
        return true;
    }

    IExpression const *lval = expr->get_argument();
    visit(lval);

    if (IDefinition const *lval_def = get_lvalue_base(lval)) {
        Dependence_graph::Node *lval_node = m_dg->get_node(lval_def);
        MDL_ASSERT(lval_node != NULL && "LValue is not a local");

        Dependence_graph::Id_type lval_id = lval_node->get_id();

        // this inc/dec depends on the control flow
        add_control_dependence(lval_id);
    }

    // do not visit children anymore
    return false;
}

// start of a call expression
bool AT_analysis::pre_visit(IExpression_call *expr)
{
    IType const *type = expr->get_type();

    Definition const *def = NULL;
    bool ignore_param_deps = false;

    if (is<IType_error>(type)) {
        // already error, try to reduce further dependencies
        ignore_param_deps = true;
    } else {
        IType::Modifiers mod = get_type_modifiers(type);
        if (mod & IType::MK_VARYING) {
            // depends on a varying result
            ignore_param_deps = true;

            if (!m_assignment_stack.empty()) {
                Dependence_graph::Id_type lval_id = m_assignment_stack.top();

                m_dg->add_dependency(lval_id, varying_call_id);
            }
        } else if (mod & IType::MK_UNIFORM) {
            // uniform result, does not depend on parameters
            ignore_param_deps = true;
        }
    }

    IExpression_reference const *ref = as<IExpression_reference>(expr->get_reference());
    if (ref == NULL) {
        // error, ignore dependencies
        ignore_param_deps = true;
    } else {
        if (ref->is_array_constructor()) {
            // array constructors are uniform calls 
        } else {
            def = impl_cast<Definition>(ref->get_definition());

            if (is_error(def)) {
                // error, ignore dependencies
                ignore_param_deps = true;
            } else {
                if (def->has_flag(Definition::DEF_IS_VARYING)) {
                    // detected a varying call
                    if (m_inside_def_arg) {
                        // default arguments do not influence the function body
                    } else {
                        // we found a varying call inside the body
                        m_has_varying_call = true;

                        if (m_curr_func_is_uniform) {
                            error_mdl_11(
                                VARYING_CALL_FROM_UNIFORM_FUNC,
                                expr->access_position(),
                                Error_params(*this).add_signature(def));
                        }
                    }

                    // assume varying result
                    ignore_param_deps = true;

                    if (!m_assignment_stack.empty()) {
                        Dependence_graph::Id_type lval_id = m_assignment_stack.top();

                        m_dg->add_dependency(lval_id, varying_call_id);
                    }
                } else if (def->has_flag(Definition::DEF_IS_UNIFORM)) {
                    // depends on AUTO-typed parameters
                }
            }
        }
    }

    if (ignore_param_deps)
        def = NULL;

    IType_function const *fkt_type = def != NULL ? cast<IType_function>(def->get_type()) : NULL;

    for (size_t i = 0, n = expr->get_argument_count(); i < n; ++i) {
        IArgument const *arg = expr->get_argument(i);

        bool independent_param = ignore_param_deps;
        if (fkt_type != NULL) {
            IType const   *p_type;
            ISymbol const *p_sym;

            fkt_type->get_parameter(i, p_type, p_sym);

            IType::Modifiers mod = get_type_modifiers(type);
            if ((mod & (IType::MK_VARYING | IType::MK_UNIFORM)) == 0) {
                // auto-typed parameter
                independent_param = false;
            }
        }

        if (independent_param) {
            // do not propagate dependencies above
            size_t old_stop = m_assignment_stack.set_stop_depth();
            visit(arg);

            m_assignment_stack.set_stop_depth(old_stop);
        } else {
            // propagate dependencies
            visit(arg);
        }
    }

    // do not visit children anymore
    return false;
}


// end of a reference
IExpression *AT_analysis::post_visit(IExpression_reference *ref)
{
    if (m_assignment_stack.empty()) {
        // outside of an assignment
        return ref;
    }

    if (IDefinition const *def = ref->get_definition()) {
        if (Dependence_graph::Node *dst_node = m_dg->get_node(def)) {
            Dependence_graph::Id_type dst_id  = dst_node->get_id();
            Dependence_graph::Id_type lval_id = m_assignment_stack.top();

            m_dg->add_dependency(lval_id, dst_id);
        }
    }
    return ref;
}

// begin of a parameter
bool AT_analysis::pre_visit(IParameter *param)
{
    // any varying calls found here do NOT make the current function varying
    Flag_store inside_def_arg(m_inside_def_arg, true);

    IType_name const *tn = param->get_type_name();
    visit(tn);

    ISimple_name const *sname = param->get_name();
    visit(sname);

    /// Get the initializing expression.
    if (IExpression const *init = param->get_init_expr())
        visit(init);

    /// Get the annotation block.
    if (IAnnotation_block const *anno = param->get_annotations())
        visit(anno);

    // do not visit children
    return false;
}

// Dump the dependency graph.
void AT_analysis::dump_dg(char const *suffix)
{
    // dump the dependency graph
    string fname(get_allocator());
    fname += m_module.get_filename();
    fname += "_";
    fname += m_dg->get_name();
    fname += suffix;
    fname += ".gv";

    if (FILE *f = fopen(fname.c_str(), "w")) {
        Allocator_builder builder(get_allocator());

        mi::base::Handle<File_Output_stream> out(
            builder.create<File_Output_stream>(get_allocator(), f, /*close_at_destroy=*/true));

        m_dg->dump(out.get());
    }
}

} // mdl
} // mi

