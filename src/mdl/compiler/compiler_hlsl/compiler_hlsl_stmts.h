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

#ifndef MDL_COMPILER_HLSL_STMTS_H
#define MDL_COMPILER_HLSL_STMTS_H 1

#include <mi/mdl/mdl_iowned.h>

#include <mdl/compiler/compilercore/compilercore_ast_list.h>

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_cc_conf.h"
#include "compiler_hlsl_locations.h"
#include "compiler_hlsl_values.h"

namespace mi {
namespace mdl {
namespace hlsl {

class Declaration;
class Expr;
class Stmt_list;
class Stmt_switch;

/// Base class of all HLSL attributes.
class HLSL_attribute {
public:
    /// A HLSL attribute on.
    enum Attribute_kind {
        ATTR_FASTOPT,             ///< do/for/while statement "fastopt" attribute
        ATTR_UNROLL,              ///< for/while statement "unroll(x)" attribute
        ATTR_LOOP,                ///< for/while statement "loop" attribute
        ATTR_ALLOW_UAV_CONDITION, ///< for/while statement "allow_uav_condition" attribute
        ATTR_BRANCH,              ///< if/switch statement "branch" attribute
        ATTR_FLATTEN,             ///< if/switch statement "flatten" attribute
        ATTR_FORECASE,            ///< switch statement "forcecase" attribute
        ATTR_CALL,                ///< switch statement "call" attribute
    };

public:
    /// Get the attribute kind;
    Attribute_kind get_kind() const { return m_kind; }

protected:
    /// Constructor.
    HLSL_attribute(Attribute_kind kind)
    : m_kind(kind)
    {
    }

protected:
    /// The kind.
    Attribute_kind const m_kind;
};

/// A HLSL unroll attribute.
class HLSL_attribute_unroll : public HLSL_attribute {
public:
    /// Get the unroll count if any.
    Value_uint_32 *get_unroll_count() const { return m_count; }

protected:
    /// Constructor.
    HLSL_attribute_unroll(Value_uint_32 *count)
    : HLSL_attribute(ATTR_UNROLL)
    , m_count(count)
    {
    }

protected:
    /// The unroll count if any.
    Value_uint_32 * const m_count;
};

/// A HLSL statement.
class Stmt : public Ast_list_element<Stmt>
{
    typedef Ast_list_element<Stmt> Base;
public:
    /// The possible kinds of statements.
    enum Kind {
        SK_INVALID,             ///< An invalid statement (syntax error).
        SK_COMPOUND,            ///< A compound statement.
        SK_DECLARATION,         ///< A declaration statement.
        SK_EXPRESSION,          ///< An expression statement.
        SK_IF,                  ///< A conditional statement.
        SK_CASE,                ///< A case label.
        SK_SWITCH,              ///< A switch statement.
        SK_WHILE,               ///< A while loop.
        SK_DO_WHILE,            ///< A do-while loop.
        SK_FOR,                 ///< A for loop.
        SK_BREAK,               ///< A break statement.
        SK_CONTINUE,            ///< A continue statement.
        SK_RETURN,              ///< A return statement.
        SK_DISCARD              ///< A discard statement.
    };

    /// Get the kind of statement.
    virtual Kind get_kind() const = 0;

    /// Get the Location.
    Location const &get_location() const { return m_loc; }

    /// Set the Location.
    void set_location(Location const &loc) { m_loc = loc; }

    /// Get the parent statement.
    Stmt_list *get_parent() { return m_parent; }

protected:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    explicit Stmt(
        Location const &loc);

protected:
    /// The location of this statement.
    Location m_loc;

    /// Points to the parent statement or NULL if this is the root statement.
    Stmt_list *m_parent;
};

/// An invalid statement (from a syntax error).
class Stmt_invalid : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_INVALID;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    explicit Stmt_invalid(
        Location const &loc);
};

/// A declaration statement.
class Stmt_decl : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_DECLARATION;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the declaration of this statement.
    Declaration *get_declaration() { return m_decl; }

    /// Get the declaration of this statement.
    Declaration const *get_declaration() const { return m_decl; }

    /// Set the declaration of this statement.
    void set_declaration(Declaration *decl) { m_decl = decl; }

private:
    /// Constructor.
    ///
    /// \param decl  the declaration of this statement
    explicit Stmt_decl(
        Declaration *decl);

private:
    /// The declaration of this statement.
    Declaration *m_decl;
};

/// A statement list.
class Stmt_list : public Stmt
{
    typedef Stmt Base;
public:
    typedef Ast_list<Stmt>::iterator       iterator;
    typedef Ast_list<Stmt>::const_iterator const_iterator;

    /// Insert a statement into this compound at the end.
    void add_stmt(Stmt *stmt);

    /// Returns true if this list is empty.
    bool empty() const { return m_stmts.empty(); }

    /// Returns the first statement of this list.
    iterator begin() { return m_stmts.begin(); }

    /// The end iterator.
    iterator end() { return m_stmts.end(); }

    /// Returns the first statement of this list.
    const_iterator begin() const { return m_stmts.begin(); }

    /// The end iterator.
    const_iterator end() const { return m_stmts.end(); }

    /// Access first element.
    Stmt *front() { return m_stmts.front(); }

    /// Access first element.
    Stmt const *front() const { return m_stmts.front(); }

    /// Access last element.
    Stmt *back() { return m_stmts.back(); }

    /// Access last element.
    Stmt const *back() const { return m_stmts.back(); }

    /// Delete last element.
    void pop_back() { m_stmts.pop_back(); }

    /// Get the number of elements in the list.
    size_t size() const { return m_stmts.size(); }

protected:
    /// Constructor.
    ///
    /// \param loc     the location of this statement
    explicit Stmt_list(
        Location const &loc);

protected:
    /// The statement list.
    Ast_list<Stmt> m_stmts;
};


/// A compound statement.
class Stmt_compound : public Stmt_list
{
    typedef Stmt_list Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_COMPOUND;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    Stmt_compound(
        Location const &loc);
};

/// An expression statement.
class Stmt_expr : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_EXPRESSION;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the expression.
    /// A NULL expression indicates an empty statement.
    Expr *get_expression() { return m_expr; }

    /// Get the expression.
    /// A NULL expression indicates an empty statement.
    Expr const *get_expression() const { return m_expr; }

    /// Set the expression.
    void set_expression(Expr *expr) { m_expr = expr; }

private:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    /// \param expr the expression
    Stmt_expr(
        Location const &loc,
        Expr           *expr);

private:
    /// The expression of this statement or NULL if this is an empty statement.
    Expr *m_expr;
};

/// A conditional statement.
class Stmt_if : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_IF;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the condition.
    Expr *get_condition() { return m_cond; }

    /// Get the condition.
    Expr const *get_condition() const { return m_cond; }

    /// Set the condition.
    void set_condition(Expr *expr) { m_cond = expr; }

    /// Get the then statement.
    Stmt *get_then_statement() { return m_then_stmt; }

    /// Get the then statement.
    Stmt const *get_then_statement() const { return m_then_stmt; }

    /// Set the then statement.
    void set_then_statement(Stmt *then_stmt) { m_then_stmt = then_stmt; }

    /// Get the else statement if any.
    Stmt *get_else_statement() { return m_else_stmt; }

    /// Get the else statement if any.
    Stmt const *get_else_statement() const { return m_else_stmt; }

    /// Set the else statement.
    void set_else_statement(Stmt *else_stmt) { m_else_stmt = else_stmt; }

    /// Get the attribute if any.
    HLSL_attribute const *get_attribute() const { return m_attribute; }

private:
    /// Constructor.
    ///
    /// \param loc        the location of this statement
    /// \param cond       the condition
    /// \param then_stmt  the then statement
    /// \param else_stmt  if non-NULL the else statement
    /// \param attribute  the attribute if any
    Stmt_if(
        Location const       &loc,
        Expr                 *cond,
        Stmt                 *then_stmt,
        Stmt                 *else_stmt,
        HLSL_attribute const *attribute = NULL);

private:
    /// The if condition.
    Expr *m_cond;

    /// The then statement;
    Stmt *m_then_stmt;

    /// The else statement or NULL if no else.
    Stmt *m_else_stmt;

    /// The attribute if any.
    HLSL_attribute const *m_attribute;
};

/// A case label.
class Stmt_case : public Stmt
{
    typedef Stmt Base;
    friend class Stmt_switch;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_CASE;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the case label.
    /// \returns    The case label, or NULL if this is a default case.
    Expr *get_label() { return m_label; }

    /// Get the case label.
    /// \returns    The case label, or NULL if this is a default case.
    Expr const *get_label() const { return m_label; }

    /// Set the case label.
    void set_label(Expr *expr) { m_label = expr; }

    /// Get the switch owner of this label.
    Stmt_switch *get_owner() { return m_owner; }

    /// set the switch owner of this label.
    void set_owner(Stmt_switch *owner) { m_owner = owner; }

    /// Get the next label of the same switch.
    Stmt_case *get_next_label() { return m_next_label; }

    /// Get the previous label of the same switch.
    Stmt_case *get_prev_label() { return m_prev_label; }

private:
    /// Constructor.
    ///
    /// \param loc     the location of this statement
    /// \param label   the label or NULL for a default label
    /// \param owner   the owner switch statement, can be different from parent
    Stmt_case(
        Location const &loc,
        Expr           *label,
        Stmt_switch    *owner);

private:
    /// The label expression.
    Expr *m_label;

    /// The owner switch statement, might be different from parent.
    Stmt_switch *m_owner;

    /// Points to the next label of the same owner.
    Stmt_case *m_next_label;

    /// Points to the previous label of the same owner.
    Stmt_case *m_prev_label;
};

/// A switch statement, from HLSL version 1.30;
class Stmt_switch : public Stmt_list
{
    typedef Stmt_list Base;
    friend class mi::mdl::Arena_builder;
public:
    typedef Stmt_list::iterator iterator;
    typedef Stmt_list::const_iterator const_iterator;

    /// The kind of this subclass.
    static Kind const s_kind = SK_SWITCH;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the switch condition.
    Expr *get_condition() { return m_cond; }

    /// Get the switch condition.
    Expr const *get_condition() const { return m_cond; }

    /// Set the condition.
    void set_condition(Expr *expr) { m_cond = expr; }

    /// Add a label.
    void add_label(Stmt_case *label);

    /// Get the first label of this switch.
    Stmt_case *get_first_label() { return m_first_label; }

    /// Get the last label of this switch.
    Stmt_case *get_last_label() { return m_last_label; }

    /// Get the default label of this switch.
    Stmt_case *get_default_label() { return m_default_label; }

    /// Get the attribute if any.
    HLSL_attribute const *get_attribute() const { return m_attribute; }

private:
    /// Constructor.
    ///
    /// \param loc        the location of this statement
    /// \param cond       the condition of this switch statement
    /// \param attribute  the attribute if any
    Stmt_switch(
        Location const       &loc,
        Expr                 *cond,
        HLSL_attribute const *attribute = NULL);

private:
    /// The switch condition.
    Expr *m_cond;

    /// Points to the first label of this switch.
    Stmt_case *m_first_label;

    /// Points to the last label of this switch.
    Stmt_case *m_last_label;

    /// Points to the default label of this switch if any.
    Stmt_case *m_default_label;

    /// The attribute if any.
    HLSL_attribute const *m_attribute;
};

/// A while loop.
class Stmt_while : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_WHILE;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the condition if any.
    Stmt *get_condition() { return m_cond; }

    /// Get the condition if any.
    Stmt const *get_condition() const { return m_cond; }

    /// Set the condition.
    void set_condition(Stmt *stmt) { m_cond = stmt; }

    /// Get the loop body.
    Stmt *get_body() { return m_body; }

    /// Get the loop body.
    Stmt const *get_body() const { return m_body; }

    /// Set the loop body.
    void set_body(Stmt *body) { m_body = body; }

    /// Get the attribute if any.
    HLSL_attribute const *get_attribute() const { return m_attribute; }

private:
    /// Constructor.
    ///
    /// \param loc        the location of this statement
    /// \param cond       the condition of this while statement
    /// \param body       the body of this while statement
    /// \param attribute  the attribute if any
    Stmt_while(
        Location const       &loc,
        Stmt                 *cond,
        Stmt                 *body,
        HLSL_attribute const *attribute = NULL);

private:
    /// The while condition.
    Stmt *m_cond;

    /// The body of this while loop.
    Stmt *m_body;

    /// The attribute if any.
    HLSL_attribute const *m_attribute;
};

/// A do-while loop.
class Stmt_do_while : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_DO_WHILE;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the loop body.
    Stmt *get_body() { return m_body; }

    /// Get the loop body.
    Stmt const *get_body() const { return m_body; }

    /// Set the loop body.
    void set_body(Stmt *body) { m_body = body; }

    /// Get the condition if any.
    Expr *get_condition() { return m_cond; }

    /// Get the condition if any.
    Expr const *get_condition() const { return m_cond; }

    /// Set the condition.
    void set_condition(Expr *expr) { m_cond = expr; }

    /// Get the attribute if any.
    HLSL_attribute const *get_attribute() const { return m_attribute; }

private:
    /// Constructor.
    ///
    /// \param loc        the location of this statement
    /// \param cond       the condition of this do-while statement
    /// \param body       the body of this do-while statement
    /// \param attribute  the attribute if any
    Stmt_do_while(
        Location const       &loc,
        Expr                 *cond,
        Stmt                 *body,
        HLSL_attribute const *attribute = NULL);

private:
    /// The do-while condition.
    Expr *m_cond;

    /// The body of this do-while loop.
    Stmt *m_body;

    /// The attribute if any.
    HLSL_attribute const *m_attribute;
};

/// A for loop.
class Stmt_for : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_FOR;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the initializer statement if any.
    /// This is either an expression or a declaration statement.
    /// If it is a declaration statement, it must be a variable declaration.
    Stmt *get_init() { return m_init_stmt; }

    /// Get the initializer statement if any.
    /// This is either an expression or a declaration statement.
    /// If it is a declaration statement, it must be a variable declaration.
    Stmt const *get_init() const { return m_init_stmt; }

    /// Set the initializer.
    void set_init(Stmt *stmt) { m_init_stmt = stmt; }

    /// Get the condition if any.
    Stmt *get_condition() { return m_cond; }

    /// Get the condition if any.
    Stmt const *get_condition() const { return m_cond; }

    /// Set the condition.
    void set_condition(Stmt *stmt) { m_cond = stmt; }

    /// Get the update expression if any.
    Expr *get_update() { return m_update_expr; }

    /// Get the update expression if any.
    Expr const *get_update() const { return m_update_expr; }

    /// Set the update expression.
    void set_update(Expr *expr) { m_update_expr = expr; }

    /// Get the loop body.
    Stmt *get_body() { return m_body; }

    /// Get the loop body.
    Stmt const *get_body() const { return m_body; }

    /// Set the loop body.
    void set_body(Stmt *body) { m_body = body; }

    /// Get the attribute if any.
    HLSL_attribute const *get_attribute() const { return m_attribute; }

private:
    /// Constructor.
    ///
    /// \param loc        the location of this statement
    /// \param init       the init statement or NULL
    /// \param cond       the condition of this for or NULL
    /// \param update     the update expression of the for or NULL
    /// \param body       the body of this do-while statement
    /// \param attribute  the attribute if any
    explicit Stmt_for(
        Location const      &loc,
        Stmt                *init,
        Stmt                *cond,
        Expr                *update,
        Stmt                *body,
        HLSL_attribute const *attribute = NULL);

private:
    /// The init statement if any.
    Stmt *m_init_stmt;

    /// The while condition.
    Stmt *m_cond;

    /// The update expression if any.
    Expr *m_update_expr;

    /// The body of this while loop.
    Stmt *m_body;

    /// The attribute if any.
    HLSL_attribute const *m_attribute;
};

/// A break statement.
class Stmt_break : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_BREAK;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    Stmt_break(
        Location const &loc);
};

/// A continue statement.
class Stmt_continue : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_CONTINUE;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    Stmt_continue(
        Location const &loc);
};

/// A discard statement.
class Stmt_discard : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_DISCARD;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param loc  the location of this statement
    Stmt_discard(
        Location const &loc);
};

/// A return statement.
class Stmt_return : public Stmt
{
    typedef Stmt Base;
    friend class mi::mdl::Arena_builder;
public:
    /// The kind of this subclass.
    static Kind const s_kind = SK_RETURN;

    /// Get the kind of statement.
    Kind get_kind() const HLSL_FINAL;

    /// Get the return expression if any.
    Expr *get_expression() { return m_expr; }

    /// Get the return expression if any.
    Expr const *get_expression() const { return m_expr; }

    /// Set the return expression.
    void set_expression(Expr *expr) { m_expr = expr; }

private:
    /// Constructor.
    ///
    /// \param loc   the location of this statement
    /// \param expr  if non-NULL, the return expression
    Stmt_return(
        Location const &loc,
        Expr           *expr);

private:
    /// The return expression if any.
    Expr *m_expr;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(Stmt *stmt) {
    return stmt->get_kind() == T::s_kind ? static_cast<T *>(stmt) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(Stmt const *stmt) {
    return stmt->get_kind() == T::s_kind ? static_cast<T const *>(stmt) : NULL;
}

/// Check if a statement is of a certain type.
template<typename T>
bool is(Stmt const *stmt) {
    return as<T const>(stmt) != NULL;
}

/// A static_cast with check in debug mode.
template <typename T>
inline T *cast(Stmt *arg) {
    HLSL_ASSERT(arg == NULL || is<T>(arg));
    return static_cast<T *>(arg);
}

/// The interface for creating statements.
class Stmt_factory : public Interface_owned
{
    typedef Interface_owned Base;
    friend class Compilation_unit;
public:
    /// Create a new invalid statement.
    ///
    /// \param loc  the location of this statement
    ///
    /// \returns    The created statement.
    Stmt_invalid *create_invalid(
        Location const &loc);

    /// Create a new declaration statement.
    ///
    /// \param decl  the declaration of this statement
    ///
    /// \returns     The created statement.
    Stmt_decl *create_declaration(
       Declaration *decl);

    /// Create a new compound statement.
    ///
    /// \param loc  the location of this statement
    ///
    /// \returns    The created statement.
    Stmt_compound *create_compound(
        Location const &loc);

    /// Create a new compound statement with initial children.
    ///
    /// \param loc       the location of this statement
    /// \param children  add this statements as children
    ///
    /// \returns    The created statement.
    Stmt_compound *create_compound(
        Location const    &loc,
        Array_ref<Stmt *> children);

    /// Create a new expression statement.
    ///
    /// \param loc   the location of this statement
    /// \param expr  the expression or NULL
    ///
    /// \returns     The created statement.
    Stmt_expr *create_expression(
        Location const &loc,
        Expr           *expr);

    /// Create a new conditional statement.
    ///
    /// \param loc        the location of this statement
    /// \param cond       the condition
    /// \param then_stmt  the then statement
    /// \param else_stmt  if non-NULL the else statement
    ///
    /// \returns          The created statement.
    Stmt_if *create_if(
        Location const &loc,
        Expr           *cond,
        Stmt           *then_stmt,
        Stmt           *else_stmt);

    /// Create a new case label.
    ///
    /// \param loc    the location of this statement
    /// \param label  the label or NULL for a default label
    /// \param owner  the owner switch statement, can be different from parent
    ///
    /// \returns      The created statement.
    Stmt_case *create_case_label(
        Location const &loc,
        Expr           *label,
        Stmt_switch    *owner);

    /// Create a new switch statement.
    ///
    /// \param loc   the location of this statement
    /// \param cond  the condition of this switch statement
    ///
    /// \returns     The created statement.
    Stmt_switch *create_switch(
        Location const &loc,
        Expr           *cond);

    /// Create a new while loop.
    ///
    /// \param loc   the location of this statement
    /// \param cond  the condition of this while statement
    /// \param body  the body of this while statement
    ///
    /// \returns     The created statement.
    Stmt_while *create_while(
        Location const &loc,
        Stmt           *cond,
        Stmt           *body);

    /// Create a new do-while loop.
    ///
    /// \param loc   the location of this statement
    /// \param cond  the condition of this do-while statement
    /// \param body  the body of this do-while statement
    ///
    /// \returns     The created statement.
    Stmt_do_while *create_do_while(
        Location const &loc,
        Expr           *cond,
        Stmt           *body);

    /// Create a new for loop with an initializing expression.
    ///
    /// \param loc     the location of this statement
    /// \param init    the init statement or NULL
    /// \param cond    the condition of this for or NULL
    /// \param update  the update expression of the for or NULL
    /// \param body    the body of this do-while statement
    ///
    /// \returns       The created statement.
    Stmt_for *create_for(
        Location const &loc,
        Stmt           *init,
        Stmt           *cond,
        Expr           *update,
        Stmt           *body);

    /// Create a new break statement.
    ///
    /// \param loc  the location of this statement
    ///
    /// \returns    The created statement.
    Stmt_break *create_break(
        Location const &loc);

    /// Create a new continue statement.
    ///
    /// \param loc  the location of this statement
    ///
    /// \returns    The created statement.
    Stmt_continue *create_continue(
        Location const &loc);

    /// Create a new discard statement.
    ///
    /// \param loc  the location of this statement
    ///
    /// \returns    The created statement.
    Stmt_discard *create_discard(
        Location const &loc);

    /// Create a new return statement.
    ///
    /// \param loc   the location of this statement
    /// \param expr  if non-NULL, the return expression
    ///
    /// \returns     The created statement.
    Stmt_return *create_return(
        Location const &loc,
        Expr           *expr);

private:
    /// Constructor.
    ///
    /// \param arena  the arena to allocate on
    Stmt_factory(
        Memory_arena &arena);

private:
    /// The builder for statement nodes.
    Arena_builder m_builder;
};

}  // hlsl
}  // mdl
}  // mi

#endif
