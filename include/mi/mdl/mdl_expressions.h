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
/// \file mi/mdl/mdl_expressions.h
/// \brief Interfaces for MDL expressions in the AST
#ifndef MDL_EXPRESSIONS_H
#define MDL_EXPRESSIONS_H 1

#include <cstddef>

#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

class IValue;
class IModule;
class Position;
class IType;
class IDefinition;
class IDeclaration;
class IType_name;
class IQualified_name;
class ISimple_name;
class IExpression;
class IConst_fold_handler;
class IValue_factory;

/// The generic interface to expressions inside the MDL AST.
class IExpression : public Interface_owned
{
public:
    /// The possible kinds of MDL AST expressions.
    enum Kind {
        EK_INVALID,                     ///< An invalid expression (syntax error).
        EK_LITERAL,                     ///< A literal expression.
        EK_REFERENCE,                   ///< A reference to a constant, variable, or function.
        EK_UNARY,                       ///< An unary expression.
        EK_BINARY,                      ///< A binary expression.
        EK_CONDITIONAL,                 ///< A conditional expression.
        EK_CALL,                        ///< A call expression.
        EK_LET,                         ///< A let expression.
    };

    /// The possible kinds of MDL AST unary and binary operators.
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
        OK_CAST,                                ///< The cast operator.
        OK_UNARY_LAST = OK_CAST,

        // binary
        OK_BINARY_FIRST,
        OK_SELECT = OK_BINARY_FIRST,            ///< The select operator.
        OK_ARRAY_INDEX,                         ///< The array index operator.
        OK_MULTIPLY,                            ///< The multiplication operator.
        OK_DIVIDE,                              ///< The division operator.
        OK_MODULO,                              ///< The modulus operator.
        OK_PLUS,                                ///< The addition operator.
        OK_MINUS,                               ///< The subtraction operator.
        OK_SHIFT_LEFT,                          ///< The shift-left operator.
        OK_SHIFT_RIGHT,                         ///< The arithmetic shift-right operator.
        OK_UNSIGNED_SHIFT_RIGHT,                ///< The unsigned shift-right operator.
        OK_LESS,                                ///< The less operator.
        OK_LESS_OR_EQUAL,                       ///< The less-or-equal operator.
        OK_GREATER_OR_EQUAL,                    ///< The greater-or-equal operator.
        OK_GREATER,                             ///< The greater operator.
        OK_EQUAL,                               ///< The equal operator.
        OK_NOT_EQUAL,                           ///< The not-equal operator.
        OK_BITWISE_AND,                         ///< The bitwise and operator.
        OK_BITWISE_XOR,                         ///< The bitwise xor operator.
        OK_BITWISE_OR,                          ///< The bitwise or operator.
        OK_LOGICAL_AND,                         ///< The logical and operator.
        OK_LOGICAL_OR,                          ///< The logical or operator.

        // binary assignments
        OK_BINARY_ASSIGN_FIRST,
        OK_ASSIGN = OK_BINARY_ASSIGN_FIRST,     ///< The assign operator.
        OK_MULTIPLY_ASSIGN,                     ///< The multiplication-assign operator.
        OK_DIVIDE_ASSIGN,                       ///< The division-assign operator.
        OK_MODULO_ASSIGN,                       ///< The modulus-assign operator.
        OK_PLUS_ASSIGN,                         ///< The plus-assign operator.
        OK_MINUS_ASSIGN,                        ///< The minus-assign operator.
        OK_SHIFT_LEFT_ASSIGN,                   ///< The shift-left-assign operator.
        OK_SHIFT_RIGHT_ASSIGN,                  ///< The arithmetic shift-right-assign operator.
        OK_UNSIGNED_SHIFT_RIGHT_ASSIGN,         ///< The unsigned shift-right-assign operator.
        OK_BITWISE_OR_ASSIGN,                   ///< The bitwise or-assign operator.
        OK_BITWISE_XOR_ASSIGN,                  ///< The bitwise xor-assign operator.
        OK_BITWISE_AND_ASSIGN,                  ///< The bitwise and-assign operator.
        OK_BINARY_ASSIGN_LAST = OK_BITWISE_AND_ASSIGN,

        OK_SEQUENCE,                            ///< The comma operator.
        OK_BINARY_LAST = OK_SEQUENCE,

        // ternary
        OK_TERNARY,                             ///< The ternary operator (conditional).
        // variadic
        OK_CALL,                                ///< The call operator.
        OK_LAST = OK_CALL
    };

    /// Get the kind of expression.
    virtual Kind get_kind() const = 0;

    /// Get the type of this expression.
    ///
    /// The type of an expression is only available after
    /// the module containing it has been analyzed.
    virtual IType const *get_type() const = 0;

    /// Set the type of this expression.
    ///
    /// \param type  the type to set
    virtual void set_type(IType const *type) = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;

    /// Returns true if the expression was in additional parenthesis.
    virtual bool in_parenthesis() const = 0;

    /// Mark that this expression was in additional parenthesis.
    virtual void mark_parenthesis() = 0;

    /// Fold this expression into a constant value if possible.
    ///
    /// \param module   The owner module of this expression.
    /// \param factory  The factory to be used to create new values if any.
    /// \param handler  The const fold handler, may be NULL.
    ///
    /// \return IValue_bad if this expression could not be folded.
    virtual IValue const *fold(
        IModule const       *module,
        IValue_factory      *factory,
        IConst_fold_handler *handler) const = 0;

    /// Return the number of sub expressions of this expression.
    virtual int get_sub_expression_count() const = 0;

    /// Return the i'th sub expression of this expression.
    ///
    /// \param i  the index of the requested subexpression
    virtual IExpression const *get_sub_expression(int i) const = 0;
};

/// An invalid expression, created by the parser in case of a syntax error.
class IExpression_invalid : public IExpression
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_INVALID;
};

/// A literal inside the MDL AST.
class IExpression_literal : public IExpression
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_LITERAL;

    /// Get the value.
    virtual IValue const *get_value() const = 0;

    /// Set the value.
    virtual void set_value(IValue const *value) = 0;
};

/// A reference to a constant, variable, function, or type inside the MDL AST.
class IExpression_reference : public IExpression
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_REFERENCE;

    /// Get the definition of this reference.
    ///
    /// \return the definition or NULL if this is an array constructor
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition of this reference.
    ///
    /// \param def  the definition to set
    virtual void set_definition(IDefinition const *def) = 0;

    /// Get the name of the references entity.
    ///
    /// \note Even in the case that a non-type is referenced here,
    ///       the name is always encoded as a type name.
    virtual IType_name const *get_name() const = 0;

    /// Set the name of the referenced entity.
    ///
    /// \param name  the (type) name of the referenced entity
    virtual void set_name(IType_name const *name) = 0;

    /// Returns true if this references an array constructor.
    virtual bool is_array_constructor() const = 0;

    /// Set the is_array_constructor property.
    virtual void set_array_constructor() = 0;
};

/// An unary expression inside the MDL AST.
class IExpression_unary : public IExpression
{
public:
    /// The possible kinds of unary operators in the MDL AST.
    enum Operator {
        OK_BITWISE_COMPLEMENT   = IExpression::OK_BITWISE_COMPLEMENT,
        OK_LOGICAL_NOT          = IExpression::OK_LOGICAL_NOT,
        OK_POSITIVE             = IExpression::OK_POSITIVE,
        OK_NEGATIVE             = IExpression::OK_NEGATIVE,
        OK_PRE_INCREMENT        = IExpression::OK_PRE_INCREMENT,
        OK_PRE_DECREMENT        = IExpression::OK_PRE_DECREMENT,
        OK_POST_INCREMENT       = IExpression::OK_POST_INCREMENT,
        OK_POST_DECREMENT       = IExpression::OK_POST_DECREMENT,
        OK_CAST                 = IExpression::OK_CAST
    };

    /// The kind of this subclass.
    static Kind const s_kind = EK_UNARY;

    /// Get the operator.
    virtual Operator get_operator() const = 0;

    /// Set the operator.
    ///
    /// \param op  the operator
    virtual void set_operator(Operator op) = 0;

    /// Get the argument expression.
    virtual IExpression const *get_argument() const = 0;

    /// Set the argument expression.
    ///
    /// \param expr  the new argument expression.
    virtual void set_argument(IExpression const *expr) = 0;

    /// Get the typename (only for cast expressions).
    virtual IType_name const *get_type_name() const = 0;

    /// Set the typename (only for cast expressions).
    ///
    /// \param tn  the type name
    virtual void set_type_name(IType_name const *tn) = 0;
};

/// A binary expression inside the MDL AST.
class IExpression_binary : public IExpression
{
public:
    /// The possible kinds of binary operators inside the MDL AST.
    enum Operator {
        OK_SELECT                           = IExpression::OK_SELECT,
        OK_ARRAY_INDEX                      = IExpression::OK_ARRAY_INDEX,
        OK_MULTIPLY                         = IExpression::OK_MULTIPLY,
        OK_DIVIDE                           = IExpression::OK_DIVIDE,
        OK_MODULO                           = IExpression::OK_MODULO,
        OK_PLUS                             = IExpression::OK_PLUS,
        OK_MINUS                            = IExpression::OK_MINUS,
        OK_SHIFT_LEFT                       = IExpression::OK_SHIFT_LEFT,
        OK_SHIFT_RIGHT                      = IExpression::OK_SHIFT_RIGHT,
        OK_UNSIGNED_SHIFT_RIGHT             = IExpression::OK_UNSIGNED_SHIFT_RIGHT,
        OK_LESS                             = IExpression::OK_LESS,
        OK_LESS_OR_EQUAL                    = IExpression::OK_LESS_OR_EQUAL,
        OK_GREATER_OR_EQUAL                 = IExpression::OK_GREATER_OR_EQUAL,
        OK_GREATER                          = IExpression::OK_GREATER,
        OK_EQUAL                            = IExpression::OK_EQUAL,
        OK_NOT_EQUAL                        = IExpression::OK_NOT_EQUAL,
        OK_BITWISE_AND                      = IExpression::OK_BITWISE_AND,
        OK_BITWISE_XOR                      = IExpression::OK_BITWISE_XOR,
        OK_BITWISE_OR                       = IExpression::OK_BITWISE_OR,
        OK_LOGICAL_AND                      = IExpression::OK_LOGICAL_AND,
        OK_LOGICAL_OR                       = IExpression::OK_LOGICAL_OR,
        OK_ASSIGN                           = IExpression::OK_ASSIGN,
        OK_MULTIPLY_ASSIGN                  = IExpression::OK_MULTIPLY_ASSIGN,
        OK_DIVIDE_ASSIGN                    = IExpression::OK_DIVIDE_ASSIGN,
        OK_MODULO_ASSIGN                    = IExpression::OK_MODULO_ASSIGN,
        OK_PLUS_ASSIGN                      = IExpression::OK_PLUS_ASSIGN,
        OK_MINUS_ASSIGN                     = IExpression::OK_MINUS_ASSIGN,
        OK_SHIFT_LEFT_ASSIGN                = IExpression::OK_SHIFT_LEFT_ASSIGN,
        OK_SHIFT_RIGHT_ASSIGN               = IExpression::OK_SHIFT_RIGHT_ASSIGN,
        OK_UNSIGNED_SHIFT_RIGHT_ASSIGN      = IExpression::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN,
        OK_BITWISE_AND_ASSIGN               = IExpression::OK_BITWISE_AND_ASSIGN,
        OK_BITWISE_XOR_ASSIGN               = IExpression::OK_BITWISE_XOR_ASSIGN,
        OK_BITWISE_OR_ASSIGN                = IExpression::OK_BITWISE_OR_ASSIGN,
        OK_SEQUENCE                         = IExpression::OK_SEQUENCE,
    };

    /// The kind of this subclass.
    static Kind const s_kind = EK_BINARY;

    /// Get the operator.
    virtual Operator get_operator() const = 0;

    /// Set the operator.
    ///
    /// \param op  the operator
    virtual void set_operator(Operator op) = 0;

    /// Get the left argument of the binary expression.
    virtual IExpression const *get_left_argument() const = 0;

    /// Set the left argument of the binary expression.
    ///
    /// \param lhs_expr  the new left argument expression
    virtual void set_left_argument(IExpression const *lhs_expr) = 0;

    /// Get the right argument expression of the binary expression.
    virtual IExpression const *get_right_argument() const = 0;

    /// Set the right argument expression of the binary expression.
    ///
    /// \param rhs_expr  the new right argument expression
    virtual void set_right_argument(IExpression const *rhs_expr) = 0;
};

/// A conditional expression inside the MDL AST (aka ?: operator).
class IExpression_conditional : public IExpression
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_CONDITIONAL;

    /// Get the (boolean) condition argument expression.
    virtual IExpression const *get_condition() const = 0;

    /// Set the condition argument expression.
    virtual void set_condition(IExpression const *cond) = 0;

    /// Get the true argument expression.
    virtual IExpression const *get_true() const = 0;

    /// Set the true argument expression.
    ///
    /// \param expr  the new true argument expression
    virtual void set_true(IExpression const *expr) = 0;

    /// Get the false argument expression.
    virtual IExpression const *get_false() const = 0;

    /// Set the false argument expression.
    ///
    /// \param expr  the new false argument expression
    virtual void set_false(IExpression const *expr) = 0;
};

/// An argument of an IExpression_call inside the MDL AST.
///
/// An IArgument wraps a IExpression and adds the information if this was
/// a positional or a named argument of a IExpression_call.
///
/// This is the base class, use IArgument_positional and IArgument_named.
class IArgument : public Interface_owned
{
public:
    /// The possible kinds of arguments.
    ///
    /// \note Invalid arguments are expressed by an IExpression_invalid.
    enum Kind {
        AK_POSITIONAL,  ///< A positional argument.
        AK_NAMED        ///< A named argument.
    };

    /// Get the kind of the argument.
    virtual Kind get_kind() const = 0;

    /// Get the argument expression.
    virtual IExpression const *get_argument_expr() const = 0;

    /// Set the argument expression.
    ///
    /// \param expr  the argument expression
    virtual void set_argument_expr(IExpression const *expr) = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;
};

/// A positional argument of an IExpression_call.
class IArgument_positional : public IArgument
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = AK_POSITIONAL;
};

/// A named argument of an IExpression_call.
class IArgument_named : public IArgument
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = AK_NAMED;

    /// Get the assigned parameter name.
    virtual ISimple_name const *get_parameter_name() const = 0;

    /// Set the assigned parameter name.
    ///
    /// \param name  the name of the parameter that is assigned
    virtual void set_parameter_name(ISimple_name const *name) = 0;
};

/// Cast to subtype or return null if types do not match.
template<typename T>
T *as(IArgument *arg) {
    return (arg->get_kind() == T::s_kind) ? static_cast<T *>(arg) : 0;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IArgument const *arg) {
    return (arg->get_kind() == T::s_kind) ? static_cast<T const *>(arg) : 0;
}

/// Check if a value is of a certain type.
template<typename T>
bool is(IArgument const *arg) {
    return as<T>(arg) != NULL;
}

/// A constructor or function call inside the MDL AST.
class IExpression_call : public IExpression
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_CALL;

    /// Get the called function reference.
    ///
    /// \note In MDL this can be only a IExpression_invalid (in case of a syntax
    //        error) or an IExpression_reference.
    virtual IExpression const *get_reference() const = 0;

    /// Set the function reference.
    ///
    /// \param ref  the reference of the called function
    virtual void set_reference(IExpression const *ref) = 0;

    /// Get the argument count.
    virtual int get_argument_count() const = 0;

    /// Get the argument at index.
    ///
    /// \param index  the index of the requested argument
    ///
    /// \note After parsing, the arguments are in the order as they
    ///       occured in the source code. After the semantic analysis
    ///       arguments are reordered and defualt arguments are inserted, so
    ///       the i'th argument corresponds to the i'th function parameter.
    virtual IArgument const *get_argument(int index) const = 0;

    /// Add an argument (to the end of the argument list).
    ///
    /// \param arg  the argument to add
    virtual void add_argument(IArgument const *arg) = 0;

    /// Replace an argument.
    ///
    /// \param index  the index of the call argument to replace
    /// \param arg    the new argument expression
    virtual void replace_argument(
        int             index,
        IArgument const *arg) = 0;
};

/// A let expression inside the MDL AST.
class IExpression_let : public IExpression
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = EK_LET;

    /// Get the expression inside the let.
    virtual IExpression const *get_expression() const = 0;

    /// Set the expression inside the let.
    ///
    /// \param expr  the new expression to set
    virtual void set_expression(IExpression const *expr) = 0;

    /// Get the number of declarations in the let.
    virtual int get_declaration_count() const = 0;

    /// Get the declaration at index.
    ///
    /// \param index  the index of the requested declaration
    virtual IDeclaration const *get_declaration(int index) const = 0;

    /// Add a declaration (at the end of the declaration list).
    ///
    /// \param decl  the declaration to add
    virtual void add_declaration(IDeclaration const *decl) = 0;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(IExpression *expr) {
    return (expr->get_kind() == T::s_kind) ? static_cast<T *>(expr) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IExpression const *expr) {
    return (expr->get_kind() == T::s_kind) ? static_cast<T const *>(expr) : NULL;
}

/// Check if a value is of a certain type.
template<typename T>
bool is(IExpression const *expr) {
    return as<T>(expr) != NULL;
}

/// The interface factory for creating expressions.
///
/// An IExpression_factory interface can be obtained by calling
/// the method create_expression_factory() on the interface IModule.
class IExpression_factory : public Interface_owned
{
public:
    /// Create a new invalid expression.
    ///
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_invalid *create_invalid(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;

    /// Create a new literal expression.
    ///
    /// \param value            The value of the literal.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_literal *create_literal(
        IValue const *value = NULL,
        int          start_line = 0,
        int          start_column = 0,
        int          end_line = 0,
        int          end_column = 0) = 0;

    /// Create a new reference expression.
    ///
    /// \param ref              The referenced entity.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_reference *create_reference(
        IType_name const *ref = NULL,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) = 0;

    /// Create a new unary expression.
    //
    /// \param op               The operator.
    /// \param argument         The argument of the operator.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_unary *create_unary(
        IExpression_unary::Operator const op,
        IExpression const                 *argument = NULL,
        int                               start_line = 0,
        int                               start_column = 0,
        int                               end_line = 0,
        int                               end_column = 0) = 0;

    /// Create a new binary expression.
    ///
    /// \param op               The operator.
    /// \param left             The left argument of the operator.
    /// \param right            The right argument of the operator.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_binary *create_binary(
        IExpression_binary::Operator const op,
        IExpression const                  *left = NULL,
        IExpression const                  *right = NULL,
        int                                start_line = 0,
        int                                start_column = 0,
        int                                end_line = 0,
        int                                end_column = 0) = 0;

    /// Create a new conditional expression.
    ///
    /// \param cond_expr        The condition.
    /// \param true_expr        The true expression.
    /// \param false_expr       The false expression.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_conditional *create_conditional(
        IExpression const *cond_expr = NULL,
        IExpression const *true_expr = NULL,
        IExpression const *false_expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new positional argument.
    ///
    /// \param expr             The expression giving the value of the argument.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The positional argument.
    virtual IArgument_positional const *create_positional_argument(
        IExpression const *expr,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new named argument.
    ///
    /// \param parameter_name   The name of the parameter.
    /// \param expr             The expression giving the value of the argument.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The named argument.
    virtual IArgument_named const *create_named_argument(
        ISimple_name const *parameter_name,
        IExpression const  *expr,
        int                start_line = 0,
        int                start_column = 0,
        int                end_line = 0,
        int                end_column = 0) = 0;

    /// Create a new call expression.
    ///
    /// \param ref              The reference to the called entity.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_call *create_call(
        IExpression const *ref = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new let expression.
    ///
    /// \param expr             The expression in the let.
    /// \param start_line       The line on which the expression begins.
    /// \param start_column     The column on which the expression begins.
    /// \param end_line         The line on which the expression ends.
    /// \param end_column       The column on which the expression ends.
    /// \returns                The created expression.
    virtual IExpression_let *create_let(
        IExpression const *expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;
};

/// Checks if the given operator is an unary one.
///
/// \param op  an expression operator
static inline bool is_unary_operator(IExpression::Operator op)
{
    return
        unsigned(op - IExpression::OK_UNARY_FIRST) <=
        unsigned(IExpression::OK_UNARY_LAST - IExpression::OK_UNARY_FIRST);
}

/// Checks if the given operator is a binary one.
///
/// \param op  an expression operator
static inline bool is_binary_operator(IExpression::Operator op)
{
    return
        unsigned(op - IExpression::OK_BINARY_FIRST) <=
        unsigned(IExpression::OK_BINARY_LAST - IExpression::OK_BINARY_FIRST);
}

/// Checks if the given operator is a binary assign operator.
static inline bool is_binary_assign_operator(IExpression::Operator op)
{
    return
        unsigned(op - IExpression::OK_BINARY_ASSIGN_FIRST) <=
        unsigned(IExpression::OK_BINARY_ASSIGN_LAST - IExpression::OK_BINARY_ASSIGN_FIRST);
}

}  // mdl
}  // mi

#endif
