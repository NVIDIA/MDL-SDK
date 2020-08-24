/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/base/iallocator.h>

#include "compilercore_cc_conf.h"
#include "compilercore_mdl.h"
#include "compilercore_modules.h"
#include "compilercore_factories.h"
#include "compilercore_positions.h"
#include "compilercore_memory_arena.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

/// A mixin for all base expression methods.
template <typename Interface>
class Expr_base : public Interface
{
    typedef Interface Base;
public:
    /// Get the kind of expression.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Get the type.
    /// The type of an expression is only available after
    /// the module containing it has been analyzed.
    IType const *get_type() const MDL_FINAL { return m_type; }

    /// Set the type.
    void set_type(IType const *type) MDL_OVERRIDE { m_type = type; }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

    /// Returns true if the expression was in additional parenthesis.
    bool in_parenthesis() const MDL_FINAL { return m_in_parenthesis; }

    /// Mark that this expression was in additional parenthesis.
    void mark_parenthesis() MDL_FINAL { m_in_parenthesis = true; }

    /// Fold this expression into a constant value if possible.
    ///
    /// \param module   The owner module of this expression.
    /// \param factory  The factory to be used to create new values if any.
    /// \param handler  The constant folding handler, may be NULL.
    ///
    /// \return IValue_bad if this expression could not be folded.
    IValue const *fold(
        IModule const       *module,
        IValue_factory      *factory,
        IConst_fold_handler *handler) const MDL_OVERRIDE
    {
        return factory->create_bad();
    }

protected:
    explicit Expr_base()
    : Base()
    , m_type(NULL)
    , m_in_parenthesis(false)
    , m_pos(0, 0, 0, 0)
    {
        // FIXME: set the error type here?
    }

private:
    // non copyable
    Expr_base(Expr_base const &) MDL_DELETED_FUNCTION;
    Expr_base &operator=(Expr_base const &) MDL_DELETED_FUNCTION;

private:
    /// The expression type.
    IType const *m_type;

    /// True if this expression is in extra parenthesis.
    bool m_in_parenthesis;

    /// The position of this expression.
    Position_impl m_pos;
};

/// An expression base mixin for expression with variadic number of arguments
template <typename Interface, typename ArgIf>
class Expr_base_variadic : public Expr_base<Interface>
{
    typedef Expr_base<Interface> Base;
public:

protected:
    explicit Expr_base_variadic(Memory_arena *arena)
    : Base()
    , m_args(arena)
    {
    }

    /// Return the number of variadic arguments.
    size_t argument_count() const { return m_args.size(); }

    /// Add a new argument.
    void add_argument(ArgIf arg) { m_args.push_back(arg); }

    /// Get the argument at given position.
    ArgIf argument_at(size_t pos) const { return m_args.at(pos); }

    /// Replace the argument at the given position.
    void replace_argument(size_t pos, ArgIf arg) { m_args.at(pos) = arg; }

    typename Arena_vector<ArgIf>::Type m_args;
};

/// Implementation of the invalid expression.
class Expression_invalid : public Expr_base<IExpression_invalid>
{
    typedef Expr_base<IExpression_invalid> Base;
public:
    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 0; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int) const MDL_FINAL { return NULL; }

    /// Constructor.
    explicit Expression_invalid()
    : Base()
    {
    }
};

/// The literal expression base mixin.
template<typename Interface>
class Expr_base_literal : public Expr_base<Interface>
{
    typedef Expr_base<Interface> Base;
public:
    /// Get the literal value.
    IValue const *get_value() const MDL_FINAL { return m_value; }

    /// Set the literal value.
    void set_value(IValue const *value) MDL_FINAL {
        m_value = value;
        Base::set_type(value->get_type());
    }

    /// Set the type.
    void set_type(IType const *type) MDL_FINAL {
        // Setting a type on a literal is forbidden
        MDL_ASSERT(type == Base::get_type());
    }

    /// Fold this literal expression into a constant value.
    IValue const *fold(
        IModule const       *,
        IValue_factory      *factory,
        IConst_fold_handler *) const MDL_FINAL
    {
        return factory->import(m_value);
    }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 0; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int) const MDL_FINAL { return NULL; }

    /// Constructor.
    ///
    /// \param value  the literal value
    explicit Expr_base_literal(IValue const *value)
    : Base()
    , m_value(value)
    {
        // automatically set the type
        Base::set_type(value->get_type());
    }

private:
    /// The value.
    IValue const *m_value;
};

/// Implementation of the literal expression.
class Expression_literal : public Expr_base_literal<IExpression_literal>
{
    typedef Expr_base_literal<IExpression_literal> Base;
public:
    /// Constructor.
    ///
    /// \param value  the literal value
    explicit Expression_literal(IValue const *value)
    : Base(value)
    {
    }
};

/// Implementation of the expression reference to a constant, variable, function, or type.
class Expression_reference : public Expr_base<IExpression_reference>
{
    typedef Expr_base<IExpression_reference> Base;
public:
    /// Get the definition of this reference.
    IDefinition const *get_definition() const MDL_FINAL { return m_def; }

    /// Set the definition of this reference.
    void set_definition(IDefinition const *def) MDL_FINAL {
        m_def = def;
        // the type of a reference if always the type of its definition
        // except for array constructors
        if (!is_array_constructor()) {
            if (def != NULL)
                set_type(def->get_type());
            else
                set_type(NULL);
        }
    }

    /// Get the name.
    IType_name const *get_name() const MDL_FINAL { return m_name; }

    /// Set the name.
    void set_name(IType_name const *name) MDL_FINAL { m_name = name; }

    /// Fold this reference expression into a constant value if possible.
    ///
    /// \param module   The owner module of this expression.
    /// \param factory  The factory to be used to create new values if any.
    /// \param handler  The const fold handler, may be NULL.
    IValue const *fold(
        IModule const       *module,
        IValue_factory      *factory,
        IConst_fold_handler *handler) const MDL_FINAL
    {
        if (m_def != NULL) {
            IDefinition::Kind kind = m_def->get_kind();
            if (kind == IDefinition::DK_CONSTANT || kind == IDefinition::DK_ENUM_VALUE)
                return factory->import(m_def->get_constant_value());
            if (handler != NULL) {
                IValue const *v = handler->lookup(m_def);
                if (!is<IValue_bad>(v))
                    return factory->import(v);
            }
        }
        return Base::fold(module, factory, handler);
    }

    /// Returns true if this references an array constructor.
    bool is_array_constructor() const MDL_FINAL { return m_is_array_constructor; }

    /// Set the is_array_constructor property.
    void set_array_constructor() MDL_FINAL { m_is_array_constructor = true; }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 0; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int) const MDL_FINAL { return NULL; }

    /// Constructor.
    ///
    /// \param name  the name of the referenced entity
    explicit Expression_reference(IType_name const *name)
    : Base()
    , m_name(name)
    , m_def(NULL)
    , m_is_array_constructor(false)
    {
    }

private:
    /// The name of this reference.
    IType_name const *m_name;

    /// The definition of this entity reference.
    IDefinition const *m_def;

    /// Set to true if the references an array constructor.
    bool m_is_array_constructor;
};

/// Implementation of the unary expression.
class Expression_unary : public Expr_base<IExpression_unary>
{
    typedef Expr_base<IExpression_unary> Base;
public:

    /// Get the operator.
    Operator get_operator() const MDL_FINAL { return m_op; }

    /// Set the operator.
    void set_operator(Operator op) MDL_FINAL { m_op = op; }

    /// Get the argument.
    IExpression const *get_argument() const MDL_FINAL { return m_expr; }

    /// Set the argument.
    void set_argument(IExpression const *expr) MDL_FINAL { m_expr = expr; }

    /// Get the typename (only for cast expressions).
    IType_name const *get_type_name() const MDL_FINAL { return m_type_name; }

    /// Set the typename (only for cast expressions).
    ///
    /// \param tn  the type name
    void set_type_name(IType_name const *tn) MDL_FINAL { m_type_name = tn; }

    /// Fold this unary expression into a constant value if possible.
    ///
    /// \param module   The owner module of this expression.
    /// \param factory  The factory to be used to create new values if any.
    /// \param handler  The const fold handler, may be NULL.
    IValue const *fold(
        IModule const       *module,
        IValue_factory      *factory,
        IConst_fold_handler *handler) const MDL_FINAL
    {
        IValue const *val = m_expr->fold(module, factory, handler);
        switch (m_op) {
        case OK_BITWISE_COMPLEMENT:
            return val->bitwise_not(factory);
        case OK_LOGICAL_NOT:
            return val->logical_not(factory);
        case OK_POSITIVE:
            return val;
        case OK_NEGATIVE:
            return val->minus(factory);

        case OK_PRE_INCREMENT:
        case OK_PRE_DECREMENT:
        case OK_POST_INCREMENT:
        case OK_POST_DECREMENT:
            // these cannot be folded
            break;

        case OK_CAST:
            {
                IType const *type = get_type();
                if (type != NULL) {
                    // type must be set
                    return val->convert(factory, type);
                }
                return factory->create_bad();
            }
        }
        return factory->create_bad();
    }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 1; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int i) const MDL_FINAL {
        return i == 0 ? m_expr : NULL;
    }

    /// Constructor.
    ///
    /// \param op    the operator
    /// \param expr  the single argument expression
    /// \param tn    the cast operator type name
    explicit Expression_unary(
        Operator          op,
        IExpression const *expr,
        IType_name const  *tn = NULL)
    : Base()
    , m_op(op)
    , m_expr(expr)
    , m_type_name(tn)
    {
    }

private:
    /// The operator of this expression.
    Operator m_op;

    /// the operand
    IExpression const *m_expr;

    /// the type name of a cast operator-
    IType_name const *m_type_name;
};

/// Implementation of the binary expression.
class Expression_binary : public Expr_base<IExpression_binary>
{
    typedef Expr_base<IExpression_binary> Base;
public:
    /// Get the operator.
    Operator get_operator() const MDL_FINAL { return m_op; }

    /// Set the operator.
    void set_operator(Operator op) MDL_FINAL { m_op = op; }

    /// Get the left argument.
    IExpression const *get_left_argument() const MDL_FINAL { return m_lhs; }

    /// Set the left argument.
    void set_left_argument(IExpression const *exp) MDL_FINAL { m_lhs = exp; }

    /// Get the right argument.
    IExpression const *get_right_argument() const MDL_FINAL { return m_rhs; }

    /// Set the right argument.
    void set_right_argument(IExpression const *exp) MDL_FINAL { m_rhs = exp; }

    /// Fold this binary expression into a constant value if possible.
    ///
    /// \param module   The owner module of this expression.
    /// \param factory  The factory to be used to create new values if any.
    /// \param handler  The const fold handler, may be NULL.
    IValue const *fold(
        IModule const       *module,
        IValue_factory      *factory,
        IConst_fold_handler *handler) const MDL_FINAL
    {
        IValue const *lhs = m_lhs->fold(module, factory, handler);
        if (is<IValue_bad>(lhs))
            return lhs;

        if (m_op == OK_SELECT) {
            if (IExpression_reference const *ref = as<IExpression_reference>(m_rhs)) {
                if (IDefinition const *def = ref->get_definition()) {
                    if (def->get_kind() == IDefinition::DK_MEMBER) {
                        int index = def->get_field_index();
                        return lhs->extract(factory, index);
                    }
                }
            }
        }

        // check for lazy evaluation ops
        if (m_op == OK_LOGICAL_AND && lhs->is_zero())
            return lhs;
        if (m_op == OK_LOGICAL_OR && lhs->is_one())
            return lhs;

        IValue const *rhs = m_rhs->fold(module, factory, handler);
        if (is<IValue_bad>(rhs))
            return rhs;

        switch (m_op) {
        case OK_SELECT:
            // handled above
            break;
        case OK_ARRAY_INDEX:
            if (is<IValue_compound>(lhs) && is<IValue_int>(rhs)) {
                IValue_compound const *a_value = cast<IValue_compound>(lhs);
                int                   index    = cast<IValue_int>(rhs)->get_value();

                if (index < 0 || index >= a_value->get_component_count()) {
                    if (handler)
                        handler->exception(
                            IConst_fold_handler::ER_INDEX_OUT_OF_BOUND,
                            m_rhs,
                            index,
                            a_value->get_component_count());
                    return factory->create_bad();
                } else {
                    return a_value->extract(factory, index);
                }
            }
            // cannot fold
            break;
        case OK_MULTIPLY:
            return lhs->multiply(factory, rhs);
        case OK_DIVIDE:
            if (is<IValue_int>(rhs) && handler != NULL) {
                int divisor = cast<IValue_int>(rhs)->get_value();

                if (divisor == 0) {
                    handler->exception(IConst_fold_handler::ER_INT_DIVISION_BY_ZERO, m_rhs);
                    return factory->create_bad();
                }
            }
            return lhs->divide(factory, rhs);
        case OK_MODULO:
            if (is<IValue_int>(rhs) && handler != NULL) {
                int divisor = cast<IValue_int>(rhs)->get_value();

                if (divisor == 0) {
                    handler->exception(IConst_fold_handler::ER_INT_DIVISION_BY_ZERO, m_rhs);
                    return factory->create_bad();
                }
            }
            return lhs->modulo(factory, rhs);
        case OK_PLUS:
            return lhs->add(factory, rhs);
        case OK_MINUS:
            return lhs->sub(factory, rhs);
        case OK_SHIFT_LEFT:
            return lhs->shl(factory, rhs);
        case OK_SHIFT_RIGHT:
            return lhs->asr(factory, rhs);
        case OK_UNSIGNED_SHIFT_RIGHT:
            return lhs->lsr(factory, rhs);
        case OK_LESS:
            {
                unsigned res = lhs->compare(rhs);
                return factory->create_bool(res == IValue::CR_LT);
            }
        case OK_LESS_OR_EQUAL:
            {
                unsigned res = lhs->compare(rhs);
                return factory->create_bool(
                    (res & IValue::CR_LE) != 0 &&
                    (res & IValue::CR_UO) == 0);
            }
        case OK_GREATER_OR_EQUAL:
            {
                unsigned res = lhs->compare(rhs);
                return factory->create_bool(
                    (res & IValue::CR_GE) != 0 &&
                    (res & IValue::CR_UO) == 0);
            }
        case OK_GREATER:
            {
                unsigned res = lhs->compare(rhs);
                return factory->create_bool(res == IValue::CR_GT);
            }
        case OK_EQUAL:
            {
                unsigned res = lhs->compare(rhs);
                return factory->create_bool(res == IValue::CR_EQ);
            }
        case OK_NOT_EQUAL:
            {
                unsigned res = lhs->compare(rhs);
                return factory->create_bool((res & IValue::CR_UEQ) == 0);
            }
        case OK_BITWISE_AND:
            return lhs->bitwise_and(factory, rhs);
        case OK_BITWISE_XOR:
            return lhs->bitwise_xor(factory, rhs);
        case OK_BITWISE_OR:
            return lhs->bitwise_or(factory, rhs);
        case OK_LOGICAL_AND:
            return lhs->logical_and(factory, rhs);
        case OK_LOGICAL_OR:
            return lhs->logical_or(factory, rhs);
        case OK_ASSIGN:
        case OK_MULTIPLY_ASSIGN:
        case OK_DIVIDE_ASSIGN:
        case OK_MODULO_ASSIGN:
        case OK_PLUS_ASSIGN:
        case OK_MINUS_ASSIGN:
        case OK_SHIFT_LEFT_ASSIGN:
        case OK_SHIFT_RIGHT_ASSIGN:
        case OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
        case OK_BITWISE_AND_ASSIGN:
        case OK_BITWISE_XOR_ASSIGN:
        case OK_BITWISE_OR_ASSIGN:
        case OK_SEQUENCE:
            // cannot fold
            break;
        }
        return factory->create_bad();
    }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 2; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int i) const MDL_FINAL
    {
        switch (i) {
        case 0:  return m_lhs;
        case 1:  return m_rhs;
        default: return NULL;
        }
    }

    /// Constructor.
    ///
    /// \param op   the operator
    /// \param lhs  the left-hand-side argument expression
    /// \param rhs  the right-hand-side argument expression
    explicit Expression_binary(Operator op, const IExpression *lhs, const IExpression *rhs)
    : Base()
    , m_op(op)
    , m_lhs(lhs)
    , m_rhs(rhs)
    {
    }

private:
    /// The operator of this expression.
    Operator m_op;

    /// The left hand side operand.
    IExpression const *m_lhs;

    /// The right hand side operand.
    IExpression const *m_rhs;
};

/// Implementation of the conditional (ternary) expression.
class Expression_conditional : public Expr_base<IExpression_conditional>
{
    typedef Expr_base<IExpression_conditional> Base;
public:

    /// Get the condition.
    IExpression const *get_condition() const MDL_FINAL { return m_cond_expr; }

    /// Set the condition.
    void set_condition(IExpression const *cond) MDL_FINAL{ m_cond_expr = cond; }

    /// Get the true expression.
    IExpression const *get_true() const MDL_FINAL { return m_true_expr; }

    /// Set the true expression.
    void set_true(IExpression const *expr) { m_true_expr = expr; }

    /// Get the false expression.
    IExpression const *get_false() const MDL_FINAL{ return m_false_expr; }

    /// Set the false expression.
    void set_false(IExpression const *expr) MDL_FINAL { m_false_expr = expr; }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 3; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int i) const MDL_FINAL
    {
        switch (i) {
        case 0:  return m_cond_expr;
        case 1:  return m_true_expr;
        case 2:  return m_false_expr;
        default: return NULL;
        }
    }

    /// Constructor.
    ///
    /// \param cond_expr   the conditional expression
    /// \param true_expr   the true expression
    /// \param false_expr  the false expression
    explicit Expression_conditional(
        IExpression const *cond_expr,
        IExpression const *true_expr,
        IExpression const *false_expr)
    : Base()
    , m_cond_expr(cond_expr)
    , m_true_expr(true_expr)
    , m_false_expr(false_expr)
    {
    }

private:
    /// The condition expression.
    IExpression const *m_cond_expr;

    /// The true expression.
    IExpression const *m_true_expr;

    /// The false expression.
    IExpression const *m_false_expr;
};

/// A mixin for all base argument methods.
template <typename Interface>
class Argument_base : public Interface
{
    typedef Interface Base;
public:
    /// Get the kind of the argument.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Get the argument expression.
    IExpression const *get_argument_expr() const MDL_FINAL { return m_expr; }

    /// Set the argument expression.
    void set_argument_expr(IExpression const *expr) MDL_FINAL { m_expr = expr; }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

protected:
    explicit Argument_base(IExpression const *expr)
    : Base()
    , m_expr(expr)
    , m_pos(0, 0, 0, 0)
    {
    }

private:
    /// The argument expression.
    IExpression const *m_expr;

    /// The position of this argument.
    Position_impl m_pos;
};

/// Implementation of the positional argument.
class Argument_positional : public Argument_base<IArgument_positional>
{
    typedef Argument_base<IArgument_positional> Base;
public:
    explicit Argument_positional(IExpression const *expr)
    : Base(expr)
    {
    }
};

/// Implementation of the named argument.
class Argument_named : public Argument_base<IArgument_named>
{
    typedef Argument_base<IArgument_named> Base;
public:
    /// Get the parameter name.
    ISimple_name const *get_parameter_name() const MDL_FINAL { return m_name; }

    /// Set the parameter name.
    void set_parameter_name(ISimple_name const *name) MDL_FINAL { m_name = name; }

    explicit Argument_named(ISimple_name const *name, IExpression const *expr)
    : Base(expr)
    , m_name(name)
    {
    }

private:
    /// The name of the argument.
    ISimple_name const *m_name;
};

/// Implementation of the constructor or function call.
class Expression_call : public Expr_base_variadic<IExpression_call, IArgument const *>
{
    typedef Expr_base_variadic<IExpression_call, IArgument const *> Base;
public:
    /// Get the function name.
    IExpression const *get_reference() const MDL_FINAL { return m_callee; }

    /// Set the function name.
    void set_reference(IExpression const *ref) MDL_FINAL { m_callee = ref; }

    /// Get the argument count.
    int get_argument_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the argument at index.
    IArgument const *get_argument(int index) const MDL_FINAL {
        return Base::argument_at(index);
    }

    /// Add an argument.
    void add_argument(IArgument const *arg) MDL_FINAL { Base::add_argument(arg); }

    /// Replace an argument.
    void replace_argument(int index, IArgument const *arg) MDL_FINAL {
        Base::replace_argument(index, arg);
    }

    /// Fold this call expression into a constant value if possible.
    ///
    /// \param module   The owner module of this expression.
    /// \param factory  The factory to be used to create new values if any.
    /// \param handler  The const fold handler, may be NULL.
    IValue const *fold(
        IModule const       *module,
        IValue_factory      *factory,
        IConst_fold_handler *handler) const MDL_FINAL
    {
        if (IExpression_reference const *ref = as<IExpression_reference>(m_callee)) {
            if (ref->is_array_constructor()) {
                // array constructor: fold all of its arguments
                IType_array const *a_type = as<IType_array>(get_type());
                if (a_type == NULL)
                    return factory->create_bad();

                int n = a_type->get_size();

                Module const *mod = impl_cast<Module>(module);
                IAllocator *alloc = mod->get_allocator();

                VLA<IValue const *> values(alloc, n);

                if (n == get_argument_count()) {
                    for (int i = 0; i < n; ++i) {
                        IArgument const   *arg  = get_argument(i);
                        IExpression const *expr = arg->get_argument_expr();
                        IValue const      *val  = expr->fold(module, factory, handler);
                        if (is<IValue_bad>(val))
                            return val;
                        values[i] = val;
                    }
                } else {
                    // array default constructor
                    IValue const *def_val =
                        mod->create_default_value(factory, a_type->get_element_type());
                    for (int i = 0; i < n; ++i) {
                        values[i] = def_val;
                    }
                }
                return factory->create_array(a_type, values.data(), n);
            }

            if (IDefinition const *def = ref->get_definition()) {
                IDefinition::Semantics sema = def->get_semantics();
                switch (sema) {
                case IDefinition::DS_UNKNOWN:
                    // cannot fold;
                    break;
                case IDefinition::DS_COPY_CONSTRUCTOR:
                    // a copy constructor copies its only argument
                    {
                        if (ref->is_array_constructor()) {
                            Module const *mod = impl_cast<Module>(module);
                            IAllocator *alloc = mod->get_allocator();

                            int n = get_argument_count();
                            VLA<IValue const *> values(alloc, n);

                            for (int i = 0; i < n; ++i) {
                                IArgument const   *arg  = get_argument(i);
                                IExpression const *expr = arg->get_argument_expr();
                                IValue const      *val  = expr->fold(module, factory, handler);

                                if (is<IValue_bad>(val))
                                    return val;
                                values[i] = val;
                            }
                            IType_array const *a_type = cast<IType_array>(get_type());
                            return factory->create_array(a_type, values.data(), n);
                        } else {
                            MDL_ASSERT(get_argument_count() == 1);
                            IExpression const *expr = get_argument(0)->get_argument_expr();
                            return expr->fold(module, factory, handler);
                        }
                    }
                case IDefinition::DS_CONV_CONSTRUCTOR:
                case IDefinition::DS_CONV_OPERATOR:
                    // a conversion constructor/operator converts its only argument
                    {
                        IType_function const *func_type = cast<IType_function>(def->get_type());
                        IType const          *dst_type  = func_type->get_return_type();

                        IExpression const *expr = get_argument(0)->get_argument_expr();
                        IValue const      *val  = expr->fold(module, factory, handler);
                        return val->convert(factory, dst_type);
                    }
                case IDefinition::DS_ELEM_CONSTRUCTOR:
                    // an element wise constructor build a value from all its argument values
                    {
                        Module const *mod = impl_cast<Module>(module);
                        IAllocator *alloc = mod->get_allocator();

                        IType_function const *func_type = cast<IType_function>(def->get_type());
                        IType const          *dst_type  = func_type->get_return_type();

                        if (IType_compound const *c_type = as<IType_compound>(dst_type)) {
                            int n_fields = c_type->get_compound_size();
                            int n_args   = get_argument_count();

                            MDL_ASSERT(n_fields == n_args || have_hidden_fields(c_type));

                            VLA<IValue const *> values(alloc, n_fields);

                            for (int i = 0; i < n_args; ++i) {
                                IArgument const   *arg  = get_argument(i);
                                IExpression const *expr = arg->get_argument_expr();
                                IValue const      *val  = expr->fold(module, factory, handler);
                                if (is<IValue_bad>(val))
                                    return val;
                                values[i] = val;
                            }

                            // fill hidden fields by their defaults
                            bool failed = false;
                            for (int i = n_args; i < n_fields; ++i) {
                                IType const *f_type = c_type->get_compound_type(i);

                                if (IType_enum const *e_type = as<IType_enum>(f_type)) {
                                    // for enum types, the default is always the first one
                                    values[i] =
                                        module->get_value_factory()->create_enum(e_type, 0);
                                } else {
                                    failed = true;
                                    break;
                                }
                            }

                            if (!failed)
                                return factory->create_compound(c_type, values.data(), n_fields);
                        }
                        // cannot fold
                        break;
                    }
                case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
                    // a color is constructed from a spectrum
#if 0
                    {
                        IValue const *wavelengths, *aplitudes;

                        IArgument const   *arg0  = get_argument(0);
                        IExpression const *expr0 = arg0->get_argument_expr();
                        wavelengths = expr0->fold(module, factory, handler);
                        if (is<IValue_bad>(wavelengths))
                            return wavelengths;

                        IArgument const   *arg1  = get_argument(1);
                        IExpression const *expr1 = arg1->get_argument_expr();
                        aplitudes = expr1->fold(module, factory, handler);
                        if (is<IValue_bad>(aplitudes))
                            return aplitudes;

                        return factory->create_spectrum_color(
                            cast<IValue_array>(wavelengths), cast<IValue_array>(aplitudes));
                    }
#else
                    // cannot fold
                    break;
#endif
                case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
                    // a matrix element wise constructor builds a matrix from element values
                    {

                        IType_function const *func_type = cast<IType_function>(def->get_type());
                        IType_matrix const   *m_type    =
                            cast<IType_matrix>(func_type->get_return_type());
                        IType_vector const   *v_type    = m_type->get_element_type();

                        IValue const *column_vals[4];
                        size_t n_cols = m_type->get_columns();
                        size_t n_rows = v_type->get_size();

                        int idx = 0;
                        for (size_t col = 0; col < n_cols; ++col) {
                            IValue const *row_vals[4];
                            for (size_t row = 0; row < n_rows; ++row, ++idx) {
                                IArgument const   *arg  = get_argument(idx);
                                IExpression const *expr = arg->get_argument_expr();
                                IValue const      *val  = expr->fold(module, factory, handler);
                                if (is<IValue_bad>(val))
                                    return val;
                                row_vals[row] = val;
                            }
                            column_vals[col] = factory->create_vector(v_type, row_vals, n_rows);
                        }
                        return factory->create_matrix(m_type, column_vals, n_cols);
                    }
                    break;
                case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
                    // a matrix diagonal constructor builds a matrix from zeros and
                    // its only argument
                    {
                        IType_function const *func_type = cast<IType_function>(def->get_type());
                        IType_matrix const   *m_type    =
                            cast<IType_matrix>(func_type->get_return_type());
                        IType_vector const   *v_type    = m_type->get_element_type();
                        IType_atomic const   *a_type    = v_type->get_element_type();

                        IExpression const *expr = get_argument(0)->get_argument_expr();
                        IValue const *val  = expr->fold(module, factory, handler);
                        IValue const *zero = factory->create_zero(a_type);

                        IValue const *column_vals[4];
                        size_t n_cols = m_type->get_columns();
                        size_t n_rows = v_type->get_size();

                        for (size_t col = 0; col < n_cols; ++col) {
                            IValue const *row_vals[4];
                            for (size_t row = 0; row < n_rows; ++row) {
                                row_vals[row] = col == row ? val : zero;
                            }
                            column_vals[col] = factory->create_vector(v_type, row_vals, n_rows);
                        }
                        return factory->create_matrix(m_type, column_vals, n_cols);
                    }
                case IDefinition::DS_INVALID_REF_CONSTRUCTOR:
                    // this constructor creates an invalid reference.
                    {
                        IType_function const *func_type = cast<IType_function>(def->get_type());
                        IType const          *dst_type  = func_type->get_return_type();

                        if (IType_reference const *r_type = as<IType_reference>(dst_type))
                            return factory->create_invalid_ref(r_type);
                        break;
                    }
                case IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR:
                    // this is the default constructor for a struct
                    {
                        Definition const *c_def = impl_cast<Definition>(def);

                        if (c_def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR)) {
                            IType_function const *func_type = cast<IType_function>(def->get_type());
                            IType const          *dst_type  = func_type->get_return_type();
                            Module const         *mod       = impl_cast<Module>(module);

                            return mod->create_default_value(factory, dst_type);
                        }
                        break;
                    }
                case IDefinition::DS_TEXTURE_CONSTRUCTOR:
                    // this constructor creates a texture
                    {
                        IType_function const *ftype    = cast<IType_function>(def->get_type());
                        IType_texture const  *tex_type =
                            cast<IType_texture>(ftype->get_return_type());

                        IExpression const *expr0 = get_argument(0)->get_argument_expr();
                        IValue const      *val0  = expr0->fold(module, factory, handler);

                        IExpression const *expr1 = get_argument(1)->get_argument_expr();
                        IValue const      *val1  = expr1->fold(module, factory, handler);

                        if (is<IValue_string>(val0) && is<IValue_enum>(val1)) {
                            IValue_string const *sval  = cast<IValue_string>(val0);
                            IValue_enum const   *gamma = cast<IValue_enum>(val1);
                            return factory->create_texture(
                                tex_type,
                                sval->get_value(),
                                IValue_texture::gamma_mode(gamma->get_value()),
                                /*tag_value=*/0,
                                /*tag_version=*/0);
                        }
                    }

                default:
                    {
                        bool fold_with_compiler =
                            def->get_property(IDefinition::DP_IS_CONST_EXPR) &&
                            is_math_semantics(sema);
                        if (fold_with_compiler || (handler != NULL &&
                            handler->is_evaluate_intrinsic_function_enabled(sema)))
                        {
                            // fold const_expr
                            Module const *mod = impl_cast<Module>(module);
                            IAllocator *alloc = mod->get_allocator();
                            int n = get_argument_count();
                            VLA<IValue const *> values(alloc, n);

                            for (int i = 0; i < n; ++i) {
                                IArgument const   *arg  = get_argument(i);
                                IExpression const *expr = arg->get_argument_expr();
                                IValue const      *val  = expr->fold(module, factory, handler);

                                if (is<IValue_bad>(val))
                                    return val;
                                values[i] = val;
                            }

                            if (fold_with_compiler)
                                return mod->m_compiler->evaluate_intrinsic_function(
                                    factory, sema, values.data(), n);
                            else
                                return handler->evaluate_intrinsic_function(
                                    sema, values.data(), n);

                        } else {
                            // unsupported
                        }
                        break;
                    }
                }
            }
        }
        return factory->create_bad();
    }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL {
        return int(1 + Base::argument_count());
    }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int i) const MDL_FINAL
    {
        if (i < 0 || size_t(i) > Base::argument_count())
            return NULL;
        if (i == 0)
            return m_callee;
        IArgument const *arg = Base::argument_at(i - 1);
        return arg->get_argument_expr();
    }

    /// Constructor.
    ///
    /// \param arena   the memory arena space are allocated on
    /// \param callee  the callee expression
    explicit Expression_call(Memory_arena *arena, IExpression const *callee)
    : Base(arena)
    , m_callee(callee)
    {
    }

    /// Check if the given compound type has hidden fields.
    static bool have_hidden_fields(IType_compound const *c_type) {
        if (IType_struct const *s_type = as<IType_struct>(c_type)) {
            // currently, only the material emission type has hidden fields
            return s_type->get_predefined_id() == IType_struct::SID_MATERIAL_EMISSION;
        }
        return false;
    }

private:
    /// The called entity.
    IExpression const *m_callee;
};

/// Implementation of the let expression.
class Expression_let : public Expr_base_variadic<IExpression_let, IDeclaration const *>
{
    typedef Expr_base_variadic<IExpression_let, IDeclaration const *> Base;
public:

    /// Get the expression in the let.
    IExpression const *get_expression() const MDL_FINAL { return m_expr; }

    /// Set the expression in the let.
    void set_expression(IExpression const *expr) MDL_FINAL { m_expr = expr; }

    /// Get the number of declarations in the let.
    int get_declaration_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the declaration at index.
    IDeclaration const *get_declaration(int index) const MDL_FINAL {
        return Base::argument_at(index);
    }

    /// Add a declaration.
    void add_declaration(IDeclaration const *decl) MDL_FINAL { Base::add_argument(decl); }

    /// Return the number of sub expressions of this expression.
    int get_sub_expression_count() const MDL_FINAL { return 1; }

    /// Return the i'th sub expression of this expression.
    IExpression const *get_sub_expression(int i) const MDL_FINAL {
        return i == 0 ? m_expr : NULL;
    }

    /// Constructor.
    ///
    /// \param arena   the memory arena space is allocated on
    /// \param expr    the let expression
    explicit Expression_let(Memory_arena *arena, IExpression const *expr)
    : Base(arena)
    , m_expr(expr)
    {
    }

private:
    /// The let expression.
    IExpression const *m_expr;
};

// -------------------------------------- expression factory --------------------------------------

// Constructs a new expression factory.
Expression_factory::Expression_factory(Memory_arena &arena)
: Base()
, m_builder(arena)
{
}

/// Set position on an expression.
static void set_position(
    IExpression *expr,
    int         start_line,
    int         start_column,
    int         end_line,
    int         end_column)
{
    Position &pos = expr->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
}

/// Set position on an argument.
static void set_position(
    IArgument *arg,
    int       start_line,
    int       start_column,
    int       end_line,
    int       end_column)
{
    Position &pos = arg->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
}

// Create a new invalid expression.
IExpression_invalid *Expression_factory::create_invalid(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IExpression_invalid *result = m_builder.create<Expression_invalid>();
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new literal expression.
IExpression_literal *Expression_factory::create_literal(
    IValue const *value,
    int          start_line,
    int          start_column,
    int          end_line,
    int          end_column)
{
    IExpression_literal *result = m_builder.create<Expression_literal>(value);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new reference expression.
IExpression_reference *Expression_factory::create_reference(
    IType_name const *name,
    int              start_line,
    int              start_column,
    int              end_line,
    int              end_column)
{
    IExpression_reference *result = m_builder.create<Expression_reference>(name);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new unary expression.
IExpression_unary *Expression_factory::create_unary(
    IExpression_unary::Operator const op,
    IExpression const                 *argument,
    int                               start_line,
    int                               start_column,
    int                               end_line,
    int                               end_column)
{
    IExpression_unary *result = m_builder.create<Expression_unary>(op, argument);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new binary expression.
IExpression_binary *Expression_factory::create_binary(
    IExpression_binary::Operator const op,
    IExpression const                  *left,
    IExpression const                  *right,
    int                                start_line,
    int                                start_column,
    int                                end_line,
    int                                end_column)
{
    IExpression_binary *result = m_builder.create<Expression_binary>(op, left, right);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new conditional expression.
IExpression_conditional *Expression_factory::create_conditional(
    IExpression const *condition,
    IExpression const *true_exp,
    IExpression const *false_exp,
    int               start_line,
    int               start_column,
    int               end_line,
    int               end_column)
{
    IExpression_conditional *result =
        m_builder.create<Expression_conditional>(condition, true_exp, false_exp);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new positional argument.
const IArgument_positional *Expression_factory::create_positional_argument(
    IExpression const *expr,
    int               start_line,
    int               start_column,
    int               end_line,
    int               end_column)
{
    IArgument_positional *result = m_builder.create<Argument_positional>(expr);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new named argument.
IArgument_named const *Expression_factory::create_named_argument(
    ISimple_name const *parameter_name,
    IExpression const  *expr,
    int                start_line,
    int                start_column,
    int                end_line,
    int                end_column)
{
    IArgument_named *result = m_builder.create<Argument_named>(parameter_name,expr);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new call expression.
IExpression_call *Expression_factory::create_call(
    IExpression const *name,
    int              start_line,
    int              start_column,
    int              end_line,
    int              end_column)
{
    IExpression_call *result = m_builder.create<Expression_call>(m_builder.get_arena(), name);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

// Create a new let expression.
IExpression_let *Expression_factory::create_let(
    IExpression const *expr,
    int               start_line,
    int               start_column,
    int               end_line,
    int               end_column)
{
    IExpression_let *result = m_builder.create<Expression_let>(m_builder.get_arena(), expr);
    set_position(result, start_line, start_column, end_line, end_column);
    return result;
}

}  // mdl
}  // mi

