/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_EXPRS_H
#define MDLTLC_EXPRS_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_types.h>

#include <mdl/compiler/compilercore/compilercore_ast_list.h>

#include "mdltlc_pprint.h"
#include "mdltlc_locations.h"

class Symbol;
class Type;
class Type_factory;
class Value;
class Value_factory;
class Compilation_unit;

typedef mi::mdl::ptr_hash_set<Symbol const>::Type Var_set;

/// mdltl expression.
class Expr : public mi::mdl::Interface_owned
{
    typedef mi::mdl::Interface_owned Base;
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
        EK_TYPE_ANNOTATION,             ///< A type annotation expression.
        EK_ATTRIBUTE,                   ///< An attribute expression.
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
        OK_OPTION,                              ///< The OPTION operator.
        OK_NONODE,                              ///< The NONODE operator.
        OK_MATCH,                               ///< The MATCH operator.
        OK_IF_GUARD,                            ///< The IF rule guard.
        OK_MAYBE_GUARD,                         ///< The MAYBE rule guard.
        OK_UNARY_LAST = OK_MAYBE_GUARD,

        // binary
        OK_BINARY_FIRST,
        OK_SELECT = OK_BINARY_FIRST,            ///< The select operator.
        OK_ARRAY_SUBSCRIPT,                     ///< The array subscript operator.
        OK_MULTIPLY,                            ///< The multiplication operator.
        OK_DIVIDE,                              ///< The division operator.
        OK_MODULO,                              ///< The modulus operator.
        OK_PLUS,                                ///< The addition operator.
        OK_MINUS,                               ///< The subtraction operator.
        OK_SHIFT_LEFT,                          ///< The shift-left operator.
        OK_SHIFT_RIGHT,                         ///< The shift-right operator.
        OK_SHIFT_RIGHT_ARITH,                   ///< The shift-right-arithmetic operator.
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
        OK_ASSIGN,                              ///< The assign operator.
        OK_TILDE,                               ///< The assign operator.
        OK_BINARY_LAST = OK_ASSIGN,
        // ternary
        OK_TERNARY,                             ///< The ternary operator (conditional).
        // variadic
        OK_CALL,                                ///< The call operator.
        OK_LAST = OK_CALL
    };

    /// Get the kind of expression.
    virtual Kind get_kind() const = 0;

    // Get the priority of the expression (for pretty-printing).
    virtual int get_prio() const = 0;

    /// Get the type of this expression.
    virtual Type *get_type() const;

    /// Set the type of this expression.
    virtual void set_type(Type *type);

    /// Get the number of sub-expressions.
    virtual size_t get_sub_expression_count() const = 0;

    /// Get the i'th subexpression.
    virtual Expr *get_sub_expression(size_t i) = 0;

    /// Get the i'th subexpression.
    virtual Expr const *get_sub_expression(size_t i) const = 0;

    /// Get the Location.
    Location const &get_location() const { return m_loc; }

    /// Returns true if this expression was in parenthesis.
    bool in_parenthesis() const { return m_in_parenthesis; }

    /// Mark that this expression was in parenthesis.
    void mark_parenthesis() { m_in_parenthesis = true; }

    /// Pretty-print the expression using the given pretty-printer.
    virtual void pp(pp::Pretty_print &p) const = 0;
    void maybe_pp_type(pp::Pretty_print &p) const;

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
    Expr(Expr const &) = delete;
    Expr &operator=(Expr const &) = delete;

protected:
    /// The type of the expression.
    Type *m_type;

    /// The location of this expression.
    Location const m_loc;

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
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

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
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

    /// Get the value of this literal.
    Value *get_value() { return m_value; }

    /// Get the value of this literal.
    Value const *get_value() const { return m_value; }

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

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
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

    Symbol const* get_name();
    Symbol const* get_name() const;

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

private:
    /// Constructor.
    ///
    /// \param name  the (type) name of this reference expression
    explicit Expr_ref(
        Location const &loc,
        Type *type,
        Symbol *symbol);

private:
    /// The name of the reference.
    Symbol *m_symbol;
};

/// An type annotation expression.
class Expr_type_annotation : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:

    /// The kind of this subclass.
    static Kind const s_kind = EK_TYPE_ANNOTATION;

    /// Get the kind of expression.
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

    /// Get the argument expression.
    Expr *get_argument() { return m_arg; }

    /// Get the argument expression.
    Expr const *get_argument() const { return m_arg; }

    Symbol *get_type_name() { return m_type_name; }
    Symbol const *get_type_name() const { return m_type_name; }

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

private:
    /// Constructor.
    ///
    /// \param loc    the location of this expression
    /// \param op     the operation kind
    /// \param arg    the operation argument
    explicit Expr_type_annotation(
        Location const &loc,
        Type *type,
        Expr *arg,
        Symbol *type_name);

private:
    /// The operation argument.
    Expr *m_arg;

    /// The annotated type name.
    Symbol *m_type_name;
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
        OK_OPTION               = Expr::OK_OPTION,
        OK_NONODE               = Expr::OK_NONODE,
        OK_MATCH                = Expr::OK_MATCH,
        OK_IF_GUARD             = Expr::OK_IF_GUARD,
        OK_MAYBE_GUARD          = Expr::OK_MAYBE_GUARD,
    };

    /// The kind of this subclass.
    static Kind const s_kind = EK_UNARY;

    /// Get the kind of expression.
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

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

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

private:
    /// Constructor.
    ///
    /// \param loc    the location of this expression
    /// \param op     the operation kind
    /// \param arg    the operation argument
    explicit Expr_unary(
        Location const &loc,
        Type *type,
        Operator op,
        Expr *arg);

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
        OK_ARRAY_SUBSCRIPT                  = Expr::OK_ARRAY_SUBSCRIPT,
        OK_MULTIPLY                         = Expr::OK_MULTIPLY,
        OK_DIVIDE                           = Expr::OK_DIVIDE,
        OK_MODULO                           = Expr::OK_MODULO,
        OK_PLUS                             = Expr::OK_PLUS,
        OK_MINUS                            = Expr::OK_MINUS,
        OK_SHIFT_LEFT                       = Expr::OK_SHIFT_LEFT,
        OK_SHIFT_RIGHT                      = Expr::OK_SHIFT_RIGHT,
        OK_SHIFT_RIGHT_ARITH                = Expr::OK_SHIFT_RIGHT_ARITH,
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
        OK_ASSIGN                           = Expr::OK_ASSIGN,
        OK_TILDE                            = Expr::OK_TILDE
    };

    /// The kind of this subclass.
    static const Kind s_kind = EK_BINARY;

    /// Get the kind of expression.
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

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

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

private:
    /// Constructor.
    explicit Expr_binary(
        Location const &loc,
        Type *type,
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
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

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

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

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

/// Call argument.
class Argument : public mi::mdl::Ast_list_element<Argument>
{
    typedef mi::mdl::Ast_list_element<Argument> Base;

    friend class mi::mdl::Arena_builder;

public:

    Expr *get_expr();
    Expr const *get_expr() const;

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

protected:
    /// Constructor.
    explicit Argument(Expr *expr);

private:
    // non copyable
    Argument(Argument const &) = delete;
    Argument &operator=(Argument const &) = delete;

protected:
    Expr *m_expr;
};

typedef mi::mdl::Ast_list<Argument> Argument_list;

/// A constructor or function call.
class Expr_call : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_CALL;

    /// Get the kind of expression.
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

    /// Get the callee.
    Expr *get_callee() { return m_callee; }

    /// Get the callee.
    Expr const *get_callee() const { return m_callee; }

    /// Get the argument count.
    size_t get_argument_count() const { return m_argument_count; }

    /// Get the argument at index.
    Expr *get_argument(size_t index);

    /// Get the argument at index.
    Expr const *get_argument(size_t index) const;

    /// Get the argument at index.
    Argument_list const & get_argument_list() const;

    void add_argument(Argument *arg);

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

private:
    /// Constructor.
    explicit Expr_call(
        mi::mdl::Memory_arena     *arena,
        Type *type,
        Expr                      *callee);

private:
    /// The callee.
    Expr *m_callee;
    Argument_list m_arguments;
    unsigned m_argument_count;
};

/// An attribute expression.
class Expr_attribute : public Expr
{
    typedef Expr Base;
    friend class mi::mdl::Arena_builder;
public:

    struct Expr_attribute_entry {
        Symbol const * name;
        Type *type;
        Expr *expr;
        bool is_pattern;

    };

    typedef std::vector<Expr_attribute_entry> Expr_attribute_vector;

    /// The kind of this subclass.
    static Kind const s_kind = EK_ATTRIBUTE;

    /// Get the kind of expression.
    Kind get_kind() const;

    // Get the priority of the expression (for pretty-printing).
    int get_prio() const;

    /// Get the number of sub-expressions.
    size_t get_sub_expression_count() const;

    /// Get the i'th subexpression.
    Expr *get_sub_expression(size_t i);

    /// Get the i'th subexpression.
    Expr const *get_sub_expression(size_t i) const;

    /// Get the argument expression.
    Expr *get_argument() { return m_arg; }

    /// Get the argument expression.
    Expr const *get_argument() const { return m_arg; }

    /// Set the argument expression.
    ///
    /// \param expr  the new argument expression.
    void set_argument(Expr *expr) { m_arg = expr; }

    /// Access the attributes of this attribute node.
    Expr_attribute_vector &get_attributes() { return m_attributes; }
    Expr_attribute_vector const &get_attributes() const { return m_attributes; };

    /// Set the attributes of this attribute node.
    void set_attributes(Expr_attribute_vector attrs) { m_attributes = attrs; }

    /// Pretty-print the expression using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    /// Return the node name of this attribute node.
    char const *get_node_name() const;

private:
    /// Constructor.
    ///
    /// \param loc    the location of this expression
    /// \param op     the operation kind
    /// \param arg    the operation argument
    explicit Expr_attribute(
        Location const &loc,
        Type *type,
        Expr *arg,
        Expr_attribute_vector attributes,
        char const *node_name);

private:
    /// The expression to which the attribute list applies.
    Expr *m_arg;

    /// List of attributes.
    Expr_attribute_vector m_attributes;

    /// Node name of the attribute node, used in code generaetion.
    char const * m_node_name;
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
inline T *cast(Expr *expr) {
    MDL_ASSERT(expr == NULL || is<T>(expr));
    return static_cast<T *>(expr);
}

/// Cast an expression.
template<typename T>
inline T const *cast(Expr const *expr) {
    MDL_ASSERT(expr == NULL || is<T>(expr));
    return static_cast<T const *>(expr);
}

/// Create expressions.
class Expr_factory : public mi::mdl::Interface_owned
{
    typedef mi::mdl::Interface_owned Base;
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
    Expr_ref *create_reference(
        Location const &loc,
        Type *type,
        Symbol *symbol);

    /// Create a new type annotation expression.
    ///
    /// \param loc    the location
    /// \param type   the type
    /// \param arg    the argument of the operator
    /// \param type_name the name of the annotated type.
    ///
    /// \returns    the created expression
    Expr_type_annotation *create_type_annotation(
        Location const &loc,
        Type *type,
        Expr *arg,
        Symbol *type_name);

    /// Create a new unary expression.
    ///
    /// \param loc    the location
    /// \param type   the type
    /// \param op     the operator
    /// \param arg    the argument of the operator
    ///
    /// \returns    the created expression
    Expr_unary *create_unary(
        Location const &loc,
        Type *type,
        Expr_unary::Operator op,
        Expr *arg);

    /// Create a new binary expression.
    ///
    /// \param loc    the location
    /// \param type   the type
    /// \param op     the operator
    /// \param left   the left argument of the operator
    /// \param right  the right argument of the operator
    ///
    /// \returns    the created expression
    Expr_binary *create_binary(
        Location const &loc,
        Type *type,
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
    Expr_conditional *create_conditional(
        Expr           *cond,
        Expr           *true_expr,
        Expr           *false_expr);

    /// Create a new call argument.
    ///
    /// \param expr  the wrapped expression
    ///
    /// \returns    the created argument
    Argument *create_argument(
        Expr *expr);

    /// Create a new call expression without arguments.
    ///
    /// \param callee  the callee expression
    ///
    /// \returns    the created expression
    Expr_call *create_call(
        Type *type,
        Expr                    *callee);

    /// Create a new typecast call expression.
    ///
    /// \param callee  the callee expression
    /// \param arg     the typecast argument
    ///
    /// \returns    the created expression
    Expr_call *create_typecast(
        Expr *callee,
        Expr *arg);

    /// Create a new attribute expression.
    ///
    /// \param loc        the location
    /// \param type       the type
    /// \param arg        the argument of the operator
    /// \param attributes the vector of attributes
    ///
    /// \returns    the created expression
    Expr_attribute *create_attribute(
        Location const &loc,
        Type *type,
        Expr *arg,
        Expr_attribute::Expr_attribute_vector attributes,
        char const *node_name);

private:
    /// Constructor.
    ///
    /// \param arena  the memory arena to allocate on.
    /// \param vf     the value factory to be used
    Expr_factory(
        mi::mdl::Memory_arena &arena,
        Compilation_unit *compilation_unit,
        Value_factory &vf);

private:
    mi::mdl::Memory_arena &m_arena;

    /// The Arena bulder used;
    mi::mdl::Arena_builder m_builder;

    /// Pointer to the compilation unit that this expression factory
    /// belongs to.
    Compilation_unit *m_compilation_unit;

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

#endif
