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

#ifndef MDL_COMPILER_HLSL_EXPRS_H
#define MDL_COMPILER_HLSL_EXPRS_H 1

#include "mdl/compiler/compilercore/compilercore_array_ref.h"

#include "compiler_hlsl_cc_conf.h"
#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_locations.h"

#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

class Memory_arena;

namespace hlsl {

class Definition;
class Location;
class Symbol;
class Type;
class Type_name;
class Type_factory;
class Value;
class Value_factory;

/// TA hlsl expression.
class Expr : public Interface_owned
{
    typedef Interface_owned Base;
public:
    /// The possible kinds of expressions.
    enum Kind {
        EK_INVALID,                     ///< An invalid expression (syntax error).
        EK_LITERAL,                     ///< A literal expression.
        EK_REFERENCE,                   ///< A reference to a constant, variable, or function.
        EK_UNARY,                       ///< An unary expression.
        EK_BINARY,                      ///< A binary expression.
        EK_CONDITIONAL,                 ///< A conditional expression.
        EK_CALL,                        ///< A call expression.
        EK_COMPOUND,                    ///< A compound initializer expression.
    };

    /// The possible kinds of unary and binary operators.
    enum Operator {
        // unary
        OK_UNARY_FIRST,
        OK_BITWISE_COMPLEMENT = OK_UNARY_FIRST, ///< The bitwise complement operator.
        OK_LOGICAL_NOT,                         ///< The unary logical negation operator.
        OK_POSITIVE,                            ///< The unary arithmetic positive operator
                                                ///  (redundant).
        OK_NEGATIVE,                            ///< The unary arithmetic negation operator.
        OK_PRE_INCREMENT,                       ///< The pre-increment operator.
        OK_PRE_DECREMENT,                       ///< The pre-decrement operator.
        OK_POST_INCREMENT,                      ///< The post-increment operator.
        OK_POST_DECREMENT,                      ///< The post-decrement operator.
        OK_POINTER_DEREF,                       ///< The unary pointer dereference operator.
        OK_UNARY_LAST = OK_POINTER_DEREF,

        // binary
        OK_BINARY_FIRST,
        OK_SELECT = OK_BINARY_FIRST,            ///< The select operator.
        OK_ARROW,                               ///< The arrow operator.
        OK_ARRAY_SUBSCRIPT,                     ///< The array subscript operator.
        OK_MULTIPLY,                            ///< The multiplication operator.
        OK_DIVIDE,                              ///< The division operator.
        OK_MODULO,                              ///< The modulus operator.
        OK_PLUS,                                ///< The addition operator.
        OK_MINUS,                               ///< The subtraction operator.
        OK_SHIFT_LEFT,                          ///< The shift-left operator.
        OK_SHIFT_RIGHT,                         ///< The shift-right operator.
        OK_LESS,                                ///< The less operator.
        OK_LESS_OR_EQUAL,                       ///< The less-or-equal operator.
        OK_GREATER_OR_EQUAL,                    ///< The greater-or-equal operator.
        OK_GREATER,                             ///< The greater operator.
        OK_EQUAL,                               ///< The equal operator.
        OK_NOT_EQUAL,                           ///< The not-equal operator.
        OK_BITWISE_AND,                         ///< The bitwise and operator.
        OK_BITWISE_OR,                          ///< The bitwise or operator.
        OK_BITWISE_XOR,                         ///< The bitwise xor operator.
        OK_LOGICAL_AND,                         ///< The logical and operator.
        OK_LOGICAL_OR,                          ///< The logical or operator.
        OK_LOGICAL_XOR,                         ///< The logical xor operator.
        OK_ASSIGN,                              ///< The assign operator.
        OK_MULTIPLY_ASSIGN,                     ///< The multiplication-assign operator.
        OK_DIVIDE_ASSIGN,                       ///< The division-assign operator.
        OK_MODULO_ASSIGN,                       ///< The modulus-assign operator.
        OK_PLUS_ASSIGN,                         ///< The plus-assign operator.
        OK_MINUS_ASSIGN,                        ///< The minus-assign operator.
        OK_SHIFT_LEFT_ASSIGN,                   ///< The shift-left-assign operator.
        OK_SHIFT_RIGHT_ASSIGN,                  ///< The arithmetic shift-right-assign operator.
        OK_BITWISE_AND_ASSIGN,                  ///< The bitwise and-assign operator.
        OK_BITWISE_OR_ASSIGN,                   ///< The bitwise or-assign operator.
        OK_BITWISE_XOR_ASSIGN,                  ///< The bitwise xor-assign operator.
        OK_SEQUENCE,                            ///< The comma operator.
        OK_BINARY_LAST = OK_SEQUENCE,
        // typecast
        OK_TYPECAST,                            ///< The typecast operator.
        // ternary
        OK_TERNARY,                             ///< The ternary operator (conditional).
        // variadic
        OK_CALL,                                ///< The call operator.
        OK_LAST = OK_CALL
    };

    /// Get the kind of expression.
    virtual Kind get_kind() const = 0;

    /// Get the type of this expression.
    virtual Type *get_type() const;

    /// Set the type of this expression.
    virtual void set_type(Type *type);

    /// Get the Location.
    Location const &get_location() const { return m_loc; }

    /// Set the location.
    void set_location(Location const &loc) { m_loc = loc; }

    /// Fold this expression into a constant value if possible.
    ///
    /// \param factory     The value factory to create values.
    ///
    /// \return Value_bad if this expression could not be folded.
    virtual Value *fold(
        Value_factory &factory) const;

    /// Returns true if this expression was in parenthesis.
    bool in_parenthesis() const { return m_in_parenthesis; }

    /// Mark that this expression was in parenthesis.
    void mark_parenthesis() { m_in_parenthesis = true; }

protected:
    /// Constructor.
    ///
    /// \param loc   the location of this expression
    /// \param type  the type of this expression
    explicit Expr(
        Location const &loc,
        Type           *type);

private:
    // non copyable
    Expr(Expr const &) HLSL_DELETED_FUNCTION;
    Expr &operator=(Expr const &) HLSL_DELETED_FUNCTION;

protected:
    /// The type of the expression.
    Type *m_type;

    /// The location of this expression.
    Location m_loc;

    /// If set, this expression was in parenthesis.
    bool m_in_parenthesis;
};

/// An invalid expression (unused yet).
class Expr_invalid : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_INVALID;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param loc   the location of this expression
    /// \param type  the type of this expression
    explicit Expr_invalid(
        Location const &loc,
        Type           *type);
};

/// A literal.
class Expr_literal : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_LITERAL;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Fold this expression into a constant value if possible.
    ///
    /// \param factory     The value factory to create values.
    ///
    /// \return Value_bad if this expression could not be folded.
    Value *fold(
        Value_factory &factory) const HLSL_FINAL;

    /// Get the value of this literal.
    Value *get_value() const { return m_value; }

private:
    /// Constructor.
    ///
    /// \param loc    the location of this expression
    /// \param value  the value of this expression
    explicit Expr_literal(
        Location const &loc,
        Value          *value);

private:
    /// The value of this literal.
    Value *m_value;
};

/// A name reference.
class Expr_ref : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_REFERENCE;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Get the name of this expression.
    Type_name *get_name() { return m_name; }

    /// Get the name of this expression.
    Type_name const *get_name() const { return m_name; }

    /// Get the definition.
    Definition *get_definition() const { return m_def; }

    /// Set the definition of this reference.
    void set_definition(Definition *def) {
        m_def = def;
    }

private:
    /// Constructor.
    ///
    /// \param name  the (type) name of this reference expression
    explicit Expr_ref(
        Type_name *name);

private:
    /// The name of this expression.
    Type_name *m_name;

    /// The definition of this reference.
    Definition *m_def;
};

/// An unary expression.
class Expr_unary : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:

    /// The possible kinds of unary operators.
    enum Operator {
        OK_BITWISE_COMPLEMENT   = Expr::OK_BITWISE_COMPLEMENT,
        OK_LOGICAL_NOT          = Expr::OK_LOGICAL_NOT,
        OK_POSITIVE             = Expr::OK_POSITIVE,
        OK_NEGATIVE             = Expr::OK_NEGATIVE,
        OK_PRE_INCREMENT        = Expr::OK_PRE_INCREMENT,
        OK_PRE_DECREMENT        = Expr::OK_PRE_DECREMENT,
        OK_POST_INCREMENT       = Expr::OK_POST_INCREMENT,
        OK_POST_DECREMENT       = Expr::OK_POST_DECREMENT,
        OK_POINTER_DEREF        = Expr::OK_POINTER_DEREF
    };

    /// The kind of this subclass.
    static Kind const s_kind = EK_UNARY;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Get the type of this expression.
    Type *get_type() const HLSL_FINAL;

    /// Set the type of this expression.
    void set_type(Type *type) HLSL_FINAL;

    /// Fold this expression into a constant value if possible.
    ///
    /// \param factory     The value factory to create values.
    ///
    /// \return Value_bad if this expression could not be folded.
    Value *fold(
        Value_factory &factory) const HLSL_FINAL;

    /// Get the operator.
    Operator get_operator() const { return m_op; }

    /// Get the argument expression.
    Expr *get_argument() { return m_arg; }

    /// Get the argument expression.
    Expr const *get_argument() const { return m_arg; }

    /// Set the argument expression.
    ///
    /// \param expr  the new argument expression.
    void set_argument(Expr *expr) { m_arg = expr; }

private:
    /// Constructor.
    ///
    /// \param loc    the location of this expression
    /// \param op     the operation kind
    /// \param arg    the operation argument
    explicit Expr_unary(
        Location const &loc,
        Operator       op,
        Expr           *arg);

private:
    /// The operation argument.
    Expr *m_arg;

    /// The operation kind.
    Operator m_op;
};

/// A binary expression.
class Expr_binary : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:

    /// The possible kinds of binary operators.
    enum Operator {
        OK_SELECT                           = Expr::OK_SELECT,
        OK_ARROW                            = Expr::OK_ARROW,
        OK_ARRAY_SUBSCRIPT                  = Expr::OK_ARRAY_SUBSCRIPT,
        OK_MULTIPLY                         = Expr::OK_MULTIPLY,
        OK_DIVIDE                           = Expr::OK_DIVIDE,
        OK_MODULO                           = Expr::OK_MODULO,
        OK_PLUS                             = Expr::OK_PLUS,
        OK_MINUS                            = Expr::OK_MINUS,
        OK_SHIFT_LEFT                       = Expr::OK_SHIFT_LEFT,
        OK_SHIFT_RIGHT                      = Expr::OK_SHIFT_RIGHT,
        OK_LESS                             = Expr::OK_LESS,
        OK_LESS_OR_EQUAL                    = Expr::OK_LESS_OR_EQUAL,
        OK_GREATER_OR_EQUAL                 = Expr::OK_GREATER_OR_EQUAL,
        OK_GREATER                          = Expr::OK_GREATER,
        OK_EQUAL                            = Expr::OK_EQUAL,
        OK_NOT_EQUAL                        = Expr::OK_NOT_EQUAL,
        OK_BITWISE_AND                      = Expr::OK_BITWISE_AND,
        OK_BITWISE_OR                       = Expr::OK_BITWISE_OR,
        OK_BITWISE_XOR                      = Expr::OK_BITWISE_XOR,
        OK_LOGICAL_AND                      = Expr::OK_LOGICAL_AND,
        OK_LOGICAL_OR                       = Expr::OK_LOGICAL_OR,
        OK_LOGICAL_XOR                      = Expr::OK_LOGICAL_XOR,
        OK_ASSIGN                           = Expr::OK_ASSIGN,
        OK_MULTIPLY_ASSIGN                  = Expr::OK_MULTIPLY_ASSIGN,
        OK_DIVIDE_ASSIGN                    = Expr::OK_DIVIDE_ASSIGN,
        OK_MODULO_ASSIGN                    = Expr::OK_MODULO_ASSIGN,
        OK_PLUS_ASSIGN                      = Expr::OK_PLUS_ASSIGN,
        OK_MINUS_ASSIGN                     = Expr::OK_MINUS_ASSIGN,
        OK_SHIFT_LEFT_ASSIGN                = Expr::OK_SHIFT_LEFT_ASSIGN,
        OK_SHIFT_RIGHT_ASSIGN               = Expr::OK_SHIFT_RIGHT_ASSIGN,
        OK_BITWISE_AND_ASSIGN               = Expr::OK_BITWISE_AND_ASSIGN,
        OK_BITWISE_XOR_ASSIGN               = Expr::OK_BITWISE_XOR_ASSIGN,
        OK_BITWISE_OR_ASSIGN                = Expr::OK_BITWISE_OR_ASSIGN,
        OK_SEQUENCE                         = Expr::OK_SEQUENCE,
    };

    /// The kind of this subclass.
    static const Kind s_kind = EK_BINARY;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Fold this expression into a constant value if possible.
    ///
    /// \param factory     The value factory to create values.
    ///
    /// \return Value_bad if this expression could not be folded.
    Value *fold(
        Value_factory &factory) const HLSL_FINAL;

    /// Get the operator.
    Operator get_operator() const { return m_op; }

    /// Get the left argument.
    Expr *get_left_argument() { return m_lhs; }

    /// Get the left argument.
    Expr const *get_left_argument() const { return m_lhs; }

    /// Set the left argument.
    ///
    /// \param lhs_expr  the new left argument expression
    void set_left_argument(Expr *lhs_expr) { m_lhs = lhs_expr; }

    /// Get the right argument expression.
    Expr *get_right_argument() { return m_rhs; }

    /// Get the right argument expression.
    Expr const *get_right_argument() const { return m_rhs; }

    /// Set the right argument expression.
    ///
    /// \param rhs_expr  the new right argument expression
    void set_right_argument(Expr *rhs_expr) { m_rhs = rhs_expr; }

private:
    /// Constructor.
    explicit Expr_binary(
        Operator op,
        Expr     *lhs,
        Expr     *rhs);

private:
    /// The left hand side argument.
    Expr *m_lhs;
    
    /// The right hand side argument.
    Expr *m_rhs;

    /// The operation kind.
    Operator m_op;
};

/// A conditional expression.
class Expr_conditional : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:

    /// The kind of this subclass.
    static Kind const s_kind = EK_CONDITIONAL;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Fold this conditional expression into a constant value if possible.
    Value *fold(Value_factory &factory) const HLSL_FINAL;

    /// Get the condition.
    Expr *get_condition() { return m_cond; }

    /// Get the condition.
    Expr const *get_condition() const { return m_cond; }

    /// Set the condition.
    void set_condition(Expr *cond) { m_cond = cond; }

    /// Get the true expression.
    Expr *get_true() { return m_true_expr; }

    /// Get the true expression.
    Expr const *get_true() const { return m_true_expr; }

    /// Set the true expression.
    void set_true(Expr *expr) { m_true_expr = expr; }

    /// Get the false expression.
    Expr *get_false() { return m_false_expr; }

    /// Get the false expression.
    Expr const *get_false() const { return m_false_expr; }

    /// Set the false expression.
    void set_false(Expr *expr) { m_false_expr = expr; }

private:
    /// Constructor.
    explicit Expr_conditional(
        Expr *cond,
        Expr *true_expr,
        Expr *false_expr);

private:
    /// The condition.
    Expr *m_cond;

    /// The true expression.
    Expr *m_true_expr;

    /// The false expression.
    Expr *m_false_expr;
};

/// A constructor or function call.
class Expr_call : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_CALL;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Fold this call expression into a constant value if possible.
    Value *fold(Value_factory &factory) const HLSL_FINAL;

    /// Get the callee.
    Expr *get_callee() { return m_callee; }

    /// Get the callee.
    Expr const *get_callee() const { return m_callee; }

    /// Set the callee.
    void set_callee(Expr *callee) { m_callee = callee; }

    /// Get the argument count.
    size_t get_argument_count() const { return m_args.size(); }

    /// Get the argument at index.
    Expr *get_argument(size_t index);

    /// Get the argument at index.
    Expr const *get_argument(size_t index) const ;

    /// Set an argument at index.
    void set_argument(size_t index, Expr *arg);

    /// Check if this call expression is a typecast.
    bool is_typecast() const { return m_is_typecast; }

    /// Mark this call as a typecast.
    void set_typecast(bool flag = true);

private:
    /// Constructor.
    explicit Expr_call(
        Memory_arena            *arena,
        Expr                    *callee,
        Array_ref<Expr *> const &args);

private:
    /// The callee.
    Expr *m_callee;

    /// The call arguments;
    Arena_vector<Expr *>::Type m_args;

    /// True, if the call is a typecast.
    bool m_is_typecast;
};

/// A compound initializer expression
class Expr_compound : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_COMPOUND;

    /// Get the kind of expression.
    Kind get_kind() const HLSL_FINAL;

    /// Fold this call expression into a constant value if possible.
    Value *fold(Value_factory &factory) const HLSL_FINAL;

    /// Get the argument count.
    size_t get_element_count() const { return m_elems.size(); }

    /// Get the element at index.
    Expr *get_element(size_t index);

    /// Get the element at index.
    Expr const *get_element (size_t index) const;

    /// Add an element
    void add_element(Expr *elem);

    /// Set an element at index.
    void set_element(size_t index, Expr *elem);

private:
    /// Constructor.
    explicit Expr_compound(
        Memory_arena   *arena,
        Location const &loc);

private:
    /// The compound elements;
    Arena_vector<Expr *>::Type m_elems;
};

/// Cast to sub-expression or return NULL if types do not match.
template<typename T>
T *as(Expr *expr) {
    return (expr->get_kind() == T::s_kind) ? static_cast<T *>(expr) : NULL;
}

template<typename T>
T const *as(Expr const *expr) {
    return (expr->get_kind() == T::s_kind) ? static_cast<T const *>(expr) : NULL;
}

/// Check if an expression is of a certain type.
template<typename T>
bool is(Expr const *expr) {
    return expr->get_kind() == T::s_kind;
}

/// Cast an expression.
template<typename T>
T *cast(Expr *expr) {
    HLSL_ASSERT(expr == NULL || is<T>(expr));
    return static_cast<T *>(expr);
}

/// Create expressions.
class Expr_factory : public Interface_owned
{
    typedef Interface_owned Base;
    friend class Compilation_unit;
public:

    /// Create a new invalid expression.
    ///
    /// \param loc  the location
    /// \returns    the created expression
    Expr_invalid *create_invalid(Location const &loc);

    /// Create a new literal expression.
    ///
    /// \param loc    the location
    /// \param value  the value of the expression
    ///
    /// \returns    the created expression
    Expr_literal *create_literal(
        Location const &loc,
        Value          *value);

    /// Create a new reference expression.
    ///
    /// \param name  the (type) name of this reference expression
    ///
    /// \returns    the created expression
    Expr *create_reference(
        Type_name *name);

    /// Create a new unary expression.
    ///
    /// \param loc    the location
    /// \param op     the operator
    /// \param arg    the argument of the operator
    ///
    /// \returns    the created expression
    Expr *create_unary(
        Location const &loc,
        Expr_unary::Operator op,
        Expr                 *arg);

    /// Create a new binary expression.
    ///
    /// \param op     the operator
    /// \param left   the left argument of the operator
    /// \param right  the right argument of the operator
    ///
    /// \returns    the created expression
    Expr *create_binary(
        Expr_binary::Operator op,
        Expr                  *left,
        Expr                  *right);

    /// Create a new conditional expression.
    ///
    /// \param cond        the condition
    /// \param true_expr   the true expression
    /// \param false_expr  the false expression
    ///
    /// \returns    the created expression
    Expr *create_conditional(
        Expr           *cond,
        Expr           *true_expr,
        Expr           *false_expr);

    /// Create a new call expression.
    ///
    /// \param callee  the callee expression
    /// \param args    the call arguments
    ///
    /// \returns    the created expression
    Expr *create_call(
        Expr                    *callee,
        Array_ref<Expr *> const &args);

    /// Create a new typecast call expression.
    ///
    /// \param callee  the callee expression
    /// \param arg     the typecast argument
    ///
    /// \returns    the created expression
    Expr *create_typecast(
        Expr *callee,
        Expr *arg);

    /// Create a new compound initializer expression.
    ///
    /// \param loc         the location
    /// \param args        the compound arguments
    ///
    /// \returns    the created expression
    Expr *create_compound(
        Location const          &loc,
        Array_ref<Expr *> const &args);

    /// Get the type factory.
    Type_factory &get_type_factory() { return m_tf; }

private:
    /// Constructor.
    ///
    /// \param arena  the memory arena to allocate on.
    /// \param vf     the value factory to be used
    Expr_factory(
        Memory_arena &arena,
        Value_factory &vf);

private:
    /// The Arena bulder used;
    Arena_builder m_builder;

    /// The type factory used;
    Type_factory &m_tf;

    /// The value factory used;
    Value_factory &m_vf;
};

/// Checks if the given operator is an unary one.
static inline bool is_unary_operator(Expr::Operator op)
{
    return
        unsigned(op - Expr::OK_UNARY_FIRST) <=
        unsigned(Expr::OK_UNARY_LAST - Expr::OK_UNARY_FIRST);
}

/// Checks if the given operator is a binary one.
static inline bool is_binary_operator(Expr::Operator op)
{
    return
        unsigned(op - Expr::OK_BINARY_FIRST) <=
        unsigned(Expr::OK_BINARY_LAST - Expr::OK_BINARY_FIRST);
}

}  // hlsl
}  // mdl
}  // mi

#endif
