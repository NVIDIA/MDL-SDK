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
/// \file mi/mdl/mdl_statements.h
/// \brief Interfaces for MDL statements in the AST
#ifndef MDL_STATEMENTS_H
#define MDL_STATEMENTS_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_positions.h>
#include <mi/mdl/mdl_expressions.h>

namespace mi {
namespace mdl {

/// The base interface of statements inside the MDL AST.
class IStatement : public Interface_owned
{
public:
    /// The possible kinds of statements.
    enum Kind {
        SK_INVALID,             ///< An invalid statement (syntax error).
        SK_COMPOUND,            ///< A compound statement.
        SK_DECLARATION,         ///< A declaration statement.
        SK_EXPRESSION,          ///< An expression statement.
        SK_IF,                  ///< A conditional statement.
        SK_CASE,                ///< A case statement.
        SK_SWITCH,              ///< A switch statement.
        SK_WHILE,               ///< A while loop.
        SK_DO_WHILE,            ///< A do-while loop.
        SK_FOR,                 ///< A for loop.
        SK_BREAK,               ///< A break statement.
        SK_CONTINUE,            ///< A continue statement.
        SK_RETURN               ///< A return statement.
    };

    /// Get the kind of statement.
    virtual Kind get_kind() const = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;
};

/// An invalid statement (from a syntax errors).
class IStatement_invalid : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_INVALID;
};

/// A declaration statement inside the MDL AST.
class IStatement_declaration : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_DECLARATION;

    /// Get the declaration of this statement.
    virtual IDeclaration const *get_declaration() const = 0;

    /// Set the declaration of this statement.
    ///
    /// \param decl  the declaration to enter
    virtual void set_declaration(IDeclaration const *decl) = 0;
};

/// A compound statement inside the MDL AST.
class IStatement_compound : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_COMPOUND;

    /// Get the statement count inside this compound statement.
    virtual int get_statement_count() const = 0;

    /// Get the statement at index.
    ///
    /// \param index the index of the requested sub-statement
    virtual IStatement const *get_statement(int index) const = 0;

    /// Add a statement (at the end of the statement list).
    ///
    /// \param stmt  the sub-statement to add
    virtual void add_statement(IStatement const *stmt) = 0;

    /// Drop all statements starting with the given index.
    ///
    /// \param index  starting at this index, this all further sub-statements
    ///               are removed
    virtual void drop_statements_after(int index) = 0;

    /// Replace all the statements with this new block.
    ///
    /// \param stmts  new sub-statements
    /// \param len    length of stmts
    virtual void replace_statements(IStatement const * const stmts[], size_t len) = 0;
};

/// An expression statement inside the MDL AST.
class IStatement_expression : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_EXPRESSION;

    /// Get the expression.
    /// A NULL expression indicates an empty statement, aka ';'.
    virtual IExpression const *get_expression() const = 0;

    /// Set the expression.
    ///
    /// \param expr  teh expression to set, might be NULL
    virtual void set_expression(IExpression const *expr) = 0;
};

/// A conditional statement inside the MDL AST.
class IStatement_if : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_IF;

    /// Get the condition expression.
    virtual IExpression const *get_condition() const = 0;

    /// Set the condition expression.
    ///
    /// \param expr  the new condition expression
    virtual void set_condition(IExpression const *expr) = 0;

    /// Get the then statement.
    virtual IStatement const *get_then_statement() const = 0;

    /// Set the then statement.
    ///
    /// \param then_stmt  the new then statement
    virtual void set_then_statement(IStatement const *then_stmt) = 0;

    /// Get the else statement if any.
    virtual IStatement const *get_else_statement() const = 0;

    /// Set the else statement.
    ///
    /// \param else_stmt  the new else statement, might be NULL
    virtual void set_else_statement(IStatement const *else_stmt) = 0;

};

/// A switch case inside the MDL AST.
///
/// \note In constrast to C-like languages switch cases in MDL must always
///       be sub-statements of a switch statement.
class IStatement_case : public IStatement_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_CASE;

    /// Get the case label.
    ///
    /// \returns    The case label, or NULL if this is a default case.
    virtual IExpression const *get_label() const = 0;

    /// Set the case label.
    ///
    /// \param expr  the new case label, might be NULL for the default case
    virtual void set_label(IExpression const *expr) = 0;
};

/// A switch statement inside the MDL AST.
class IStatement_switch : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_SWITCH;

    /// Get the switch condition expression.
    virtual IExpression const *get_condition() const = 0;

    /// Set the switch condition expression.
    ///
    /// \param expr  the new switch condition expression
    virtual void set_condition(IExpression const *expr) = 0;

    /// Get the number of case statements inside this switch.
    virtual int get_case_count() const = 0;

    /// Get the case statement at index, either an IStatement_invalid or an IStatement_case.
    ///
    /// \param index  the index of the requested case staement
    virtual IStatement const *get_case(int index) const = 0;

    /// Add a case (at the end of the case statement list).
    ///
    /// \param switch_case  the case statement to add
    virtual void add_case(IStatement const *switch_case) = 0;
};

/// The base class for all loop statements inside the MDL AST.
class IStatement_loop : public IStatement
{
public:
    /// Get the loop condition expression if any.
    virtual IExpression const *get_condition() const = 0;

    /// Set the loop condition expression.
    ///
    /// \param expr  the new loop condition, might be NULL
    virtual void set_condition(IExpression const *expr) = 0;

    /// Get the loop body statement.
    virtual IStatement const *get_body() const = 0;

    /// Set the loop body statement.
    ///
    /// \param body  the new loop body statement
    virtual void set_body(IStatement const *body) = 0;
};

/// A while loop inside the MDL AST.
class IStatement_while : public IStatement_loop
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_WHILE;
};

/// A do-while loop inside the MDL AST.
class IStatement_do_while : public IStatement_loop
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_DO_WHILE;
};

/// A for loop inside the MDL AST.
class IStatement_for : public IStatement_loop
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_FOR;

    /// Get the initializer statement if any.
    ///
    /// This is either an expression or a declaration statement.
    /// If it is a declaration statement, it must be a variable declaration.
    virtual IStatement const *get_init() const = 0;

    /// Set the initializer statement.
    ///
    /// \param stmt  the initializer statement, might be NULL
    virtual void set_init(IStatement const *stmt) = 0;

    /// Get the update expression if any.
    virtual IExpression const *get_update() const = 0;

    /// Set the update expression.
    ///
    /// \param expr  the new update expression, might be NULL
    virtual void set_update(IExpression const *expr) = 0;
};

/// A break statement inside the MDL AST.
class IStatement_break : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_BREAK;
};

/// A continue statement inside the MDL AST.
class IStatement_continue : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_CONTINUE;
};

/// A return statement inside the MDL AST.
class IStatement_return : public IStatement
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_RETURN;

    /// Get the return expression if any.
    virtual IExpression const *get_expression() const = 0;

    /// Set the return expression.
    ///
    /// \param expr  the return expression, might be NULL
    virtual void set_expression(IExpression const *expr) = 0;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(IStatement *stmt) {
    return stmt != NULL && stmt->get_kind() == T::s_kind ? static_cast<T *>(stmt) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IStatement const *stmt) {
    return stmt != NULL && stmt->get_kind() == T::s_kind ? static_cast<T const *>(stmt) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline IStatement_loop *as<IStatement_loop>(IStatement *stmt) {
    switch (stmt->get_kind()) {
    case IStatement::SK_WHILE:
    case IStatement::SK_DO_WHILE:
    case IStatement::SK_FOR:
        return static_cast<IStatement_loop *>(stmt);
    default:
        return NULL;
    }
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline IStatement_loop const *as<IStatement_loop>(IStatement const *stmt) {
    switch (stmt->get_kind()) {
    case IStatement::SK_WHILE:
    case IStatement::SK_DO_WHILE:
    case IStatement::SK_FOR:
        return static_cast<IStatement_loop const *>(stmt);
    default:
        return NULL;
    }
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline IStatement_compound *as<IStatement_compound>(IStatement *stmt) {
    switch (stmt->get_kind()) {
    case IStatement::SK_COMPOUND:
    case IStatement::SK_CASE:
        return static_cast<IStatement_compound *>(stmt);
    default:
        return NULL;
    }
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline IStatement_compound const *as<IStatement_compound>(IStatement const *stmt) {
    switch (stmt->get_kind()) {
    case IStatement::SK_COMPOUND:
    case IStatement::SK_CASE:
        return static_cast<IStatement_compound const *>(stmt);
    default:
        return NULL;
    }
}

/// Check if a value is of a certain type.
template<typename T>
bool is(const IStatement *stmt) {
    return as<T>(stmt) != NULL;
}

/// Check if a value is of a certain type.
template<>
inline bool is<IStatement_loop>(IStatement const *stmt) {
    return as<IStatement_loop>(stmt) != NULL;
}

/// Check if a value is of a certain type.
template<>
inline bool is<IStatement_compound>(IStatement const *stmt) {
    return as<IStatement_compound>(stmt) != NULL;
}

/// The interface for creating statements.
///
/// An IStatement_factory interface can be obtained by calling
/// the method create_statement_factory() on the interface IModule.
class IStatement_factory : public Interface_owned
{
public:
    /// Create a new invalid statement.
    ///
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_invalid *create_invalid(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;

    /// Create a new compound statement.
    ///
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_compound *create_compound(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;

    /// Create a new declaration statement.
    ///
    /// \param decl             The declaration.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_declaration *create_declaration(
        IDeclaration const *decl = NULL,
        int                start_line = 0,
        int                start_column = 0,
        int                end_line = 0,
        int                end_column = 0) = 0;

    /// Create a new statement statement.
    ///
    /// \param expr             The expression.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_expression *create_expression(
        IExpression const *expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new conditional statement.
    ///
    /// \param cond             The condition.
    /// \param then_stmt        The statement executed if the condition is true.
    /// \param else_stmt        The statement executed if the condition is false.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_if *create_if(
        IExpression const *cond = NULL,
        IStatement const *then_stmt = NULL,
        IStatement const *else_stmt = NULL,
        int              start_line = 0,
        int              start_column = 0,
        int              end_line = 0,
        int              end_column = 0) = 0;

    /// Create a new switch case.
    ///
    /// \param label            The label expression, or null for default cases.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_case *create_switch_case(
        IExpression const *label = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new switch statement.
    ///
    /// \param cond             The switch condition.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_switch *create_switch(
        IExpression const *cond = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new while loop.
    ///
    /// \param cond             The loop condition.
    /// \param body             The loop body.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_while *create_while(
        IExpression const *cond = NULL,
        IStatement const  *body = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new do-while loop.
    ///
    /// \param cond             The loop condition.
    /// \param body             The loop body.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_do_while *create_do_while(
        IExpression const *cond = NULL,
        IStatement const  *body = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new for loop with an initializing expression.
    ///
    /// \param init             The init expression.
    /// \param cond             The loop condition.
    /// \param update           The update expression.
    /// \param body             The loop body.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_for *create_for(
        IStatement const  *init = NULL,
        IExpression const *cond = NULL,
        IExpression const *update = NULL,
        IStatement const  *body = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;

    /// Create a new break statement.
    ///
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_break *create_break(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;

    /// Create a new continue statement.
    ///
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_continue *create_continue(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;

    /// Create a new return statement.
    ///
    /// \param expr             The expression returned or null.
    /// \param start_line       The line on which the statement begins.
    /// \param start_column     The column on which the statement begins.
    /// \param end_line         The line on which the statement ends.
    /// \param end_column       The column on which the statement ends.
    /// \returns                The created statement.
    virtual IStatement_return *create_return(
        IExpression const *expr = NULL,
        int               start_line = 0,
        int               start_column = 0,
        int               end_line = 0,
        int               end_column = 0) = 0;
};

}  // mdl
}  // mi

#endif
