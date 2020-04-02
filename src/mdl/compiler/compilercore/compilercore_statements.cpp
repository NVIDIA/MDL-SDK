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

#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/base/iallocator.h>

#include <vector>

#include "compilercore_cc_conf.h"
#include "compilercore_factories.h"
#include "compilercore_positions.h"

namespace mi {
namespace mdl {

/// A mixin for all base statement methods.
template <typename Interface>
class Stmt_base : public Interface
{
    typedef Interface Base;
public:

    /// Get the kind of statement.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

protected:
    explicit Stmt_base()
    : Base()
    , m_pos(0, 0, 0, 0)
    {
    }

private:
    // non copyable
    Stmt_base(Stmt_base const &) MDL_DELETED_FUNCTION;
    Stmt_base &operator=(Stmt_base const &) MDL_DELETED_FUNCTION;

protected:
    /// The position of this statement.
    Position_impl m_pos;
};

/// A mixin for statements with variadic number of arguments.
template <typename Interface, typename ArgIf>
class Stmt_base_variadic : public Stmt_base<Interface>
{
    typedef Stmt_base<Interface> Base;
public:

protected:
    explicit Stmt_base_variadic(Memory_arena *arena)
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

    /// Drop all after given index.
    void drop_after(size_t index) {
        if (index < m_args.size()) {
            m_args.resize(index, ArgIf(0));
        }
    }

    /// Replace the arguments with this new block.
    void replace_arguments(ArgIf const args[], size_t len)
    {
        m_args.resize(len, ArgIf(0));
        for (size_t i = 0; i < len; ++i)
            m_args[i] = args[i];
    }

    std::vector<ArgIf, Memory_arena_allocator<ArgIf> > m_args;
};

/// A mixin for compound statements
template <typename Interface>
class Stmt_base_compound : public Stmt_base_variadic<Interface, IStatement const *>
{
    typedef Stmt_base_variadic<Interface, IStatement const *> Base;
public:

    /// Get the statement count.
    int get_statement_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the statement at index.
    ///
    /// \params index  the index of the requested sub-statement
    IStatement const *get_statement(int index) const MDL_FINAL {
        return Base::argument_at(index);
    }

    /// Add a statement.
    ///
    /// \param stmt  the sub-statement to add
    void add_statement(IStatement const *stmt) MDL_FINAL { Base::add_argument(stmt); }

    /// Drop all statements starting with the given index.
    ///
    /// \param index  starting at this index, this all further sub-statements
    ///               are removed
    void drop_statements_after(int index) MDL_FINAL {
        Base::drop_after(index);
    }

    /// Replace the statements with this new block.
    ///
    /// \param stmts  new sub-statements
    /// \param len    length of stmts
    void replace_statements(IStatement const * const stmts[], size_t len) MDL_FINAL {
        Base::replace_arguments(stmts, len);
    }

    explicit Stmt_base_compound(Memory_arena *arena)
    : Base(arena)
    {
    }
};

/// Implementation of the invalid statement.
class Statement_invalid : public Stmt_base<IStatement_invalid>
{
    typedef Stmt_base<IStatement_invalid> Base;
public:
    explicit Statement_invalid()
    : Base()
    {
    }
};

/// Implementation of the declaration statement.
class Statement_declaration : public Stmt_base<IStatement_declaration>
{
    typedef Stmt_base<IStatement_declaration> Base;
public:

    /// Get the declaration.
    const IDeclaration *get_declaration() const MDL_FINAL { return m_decl; }

    /// Set the declaration.
    void set_declaration(IDeclaration const *decl) MDL_FINAL { m_decl = decl; }

    explicit Statement_declaration(IDeclaration const *decl)
    : Base()
    , m_decl(decl)
    {
    }

private:
    /// The declaration.
    IDeclaration const *m_decl;
};

/// Implementation of the compound statement.
class Statement_compound : public Stmt_base_compound<IStatement_compound>
{
    typedef Stmt_base_compound<IStatement_compound> Base;
public:

    explicit Statement_compound(Memory_arena *arena)
    : Base(arena)
    {
    }
};

/// Implementation of the expression statement.
class Statement_expression : public Stmt_base<IStatement_expression>
{
    typedef Stmt_base<IStatement_expression> Base;
public:
    /// Get the expression.
    /// A null expression indicates an empty statement.
    IExpression const *get_expression() const MDL_FINAL { return m_expr; }

    /// Set the expression.
    void set_expression(IExpression const *expr) MDL_FINAL { m_expr = expr; }

    explicit Statement_expression(IExpression const *expr)
    : Base()
    , m_expr(expr)
    {
    }

private:
    /// The expression, might be NULL.
    IExpression const *m_expr;
};

/// Implementation of the conditional statement.
class Statement_if : public Stmt_base<IStatement_if>
{
    typedef Stmt_base<IStatement_if> Base;
public:
    /// Get the condition.
    IExpression const *get_condition() const MDL_FINAL { return m_cond_expr; }

    /// Set the condition.
    void set_condition(IExpression const *expr) MDL_FINAL { m_cond_expr = expr; }

    /// Get the then statement.
    IStatement const *get_then_statement() const MDL_FINAL { return m_then_stmt; }

    /// Set the then statement.
    void set_then_statement(IStatement const *then_stmnt) MDL_FINAL {
        m_then_stmt = then_stmnt;
    }

    /// Get the else statement.
    IStatement const *get_else_statement() const MDL_FINAL { return m_else_stmt; }

    /// Set the else statement.
    void set_else_statement(IStatement const *else_stmnt) MDL_FINAL {
        m_else_stmt = else_stmnt;
    }

    explicit Statement_if(
        IExpression const *cond_expr,
        IStatement const *then_stmt,
        IStatement const *else_stmt)
    : Base()
    , m_cond_expr(cond_expr)
    , m_then_stmt(then_stmt)
    , m_else_stmt(else_stmt)
    {
    }

private:
    /// The conditional expression.
    IExpression const *m_cond_expr;

    /// The then statement.
    IStatement const *m_then_stmt;

    /// The else statement, might be NULL.
    IStatement const *m_else_stmt;
};

/// Implementation of the switch case.
class Statement_case : public Stmt_base_compound<IStatement_case>
{
    typedef Stmt_base_compound<IStatement_case> Base;
public:

    /// Get the case label.
    /// \returns    The case label, or null if this is a default case.
    IExpression const *get_label() const MDL_FINAL { return m_label_expr; }

    /// Set the case label.
    void set_label(IExpression const *expr) MDL_FINAL { m_label_expr = expr; }

    explicit Statement_case(Memory_arena *arena, IExpression const *label)
    : Base(arena)
    , m_label_expr(label)
    {
    }

private:
    /// The case label expression.
    IExpression const *m_label_expr;
};

/// Implementation of the switch statement.
class Statement_switch : public Stmt_base_variadic<IStatement_switch, IStatement const *>
{
    typedef Stmt_base_variadic<IStatement_switch, IStatement const *> Base;
public:
    /// Get the condition.
    IExpression const *get_condition() const MDL_FINAL { return m_switch_expr; }

    /// Set the condition.
    void set_condition(IExpression const *expr) MDL_FINAL { m_switch_expr = expr; }

    /// Get the case count.
    int get_case_count() const MDL_FINAL { return Base::argument_count(); }

    /// Get the case at index, either an IStatement_invalid or an IStatement_case.
    IStatement const *get_case(int index) const MDL_FINAL {
        return Base::argument_at(index);
    }

    /// Add a case.
    void add_case(IStatement const *switch_case) MDL_FINAL {
        Base::add_argument(switch_case);
    }

    explicit Statement_switch(Memory_arena *arena, IExpression const *expr)
    : Base(arena)
    , m_switch_expr(expr)
    {
    }

private:
    /// The switch expression.
    IExpression const *m_switch_expr;
};

/// Implementation of the loop base mixin.
template <typename Interface>
class Stmt_loop : public Stmt_base<Interface>
{
    typedef Stmt_base<Interface> Base;
public:
    /// Get the condition.
    IExpression const *get_condition() const MDL_FINAL { return m_cond_expr; }

    /// Set the condition.
    void set_condition(IExpression const *expr) MDL_FINAL { m_cond_expr = expr; }

    /// Get the loop body.
    IStatement const *get_body() const MDL_FINAL { return m_body_stmt; }

    /// Set the loop body.
    void set_body(IStatement const *body) MDL_FINAL { m_body_stmt = body; }

protected:
    explicit Stmt_loop(IExpression const *cond, IStatement const *body)
    : Base()
    , m_cond_expr(cond)
    , m_body_stmt(body)
    {
    }

    /// The condition.
    IExpression const *m_cond_expr;
    /// The body.
    IStatement const *m_body_stmt;
};

/// Implementation of the while loop.
class Statement_while : public Stmt_loop<IStatement_while>
{
    typedef Stmt_loop<IStatement_while> Base;
public:
    explicit Statement_while(IExpression const *cond, IStatement const *body)
    : Base(cond, body)
    {
    }
};

/// Implementation of the do-while loop.
class Statement_do_while : public Stmt_loop<IStatement_do_while>
{
    typedef Stmt_loop<IStatement_do_while> Base;
public:
    explicit Statement_do_while(IExpression const *cond, IStatement const *body)
    : Base(cond, body)
    {
    }
};

/// Implementation of the for loop.
class Statement_for : public Stmt_loop<IStatement_for>
{
    typedef Stmt_loop<IStatement_for> Base;
public:
    /// Get the initializer statement.
    /// This is either an expression or a declaration statement.
    /// If it is a declaration statement, it must be a variable declaration.
    IStatement const *get_init() const MDL_FINAL { return m_init_stmt; }

    /// Set the initializer.
    void set_init(IStatement const *stmt) MDL_FINAL { m_init_stmt = stmt; }

    /// Get the update expression.
    IExpression const *get_update() const MDL_FINAL { return m_update_expr; }

    /// Set the update expression.
    void set_update(IExpression const *expr) MDL_FINAL { m_update_expr = expr; }

    explicit Statement_for(
        IStatement const  *init,
        IExpression const *cond,
        IExpression const *update,
        IStatement const  *body)
    : Base(cond, body)
    , m_init_stmt(init)
    , m_update_expr(update)
    {
    }

private:
    /// The init statement of this for loop, might be NULL.
    IStatement const *m_init_stmt;

    /// The update expression of this for loop, might be NULL.
    IExpression const *m_update_expr;
};

/// Implementation of the break statement.
class Statement_break : public Stmt_base<IStatement_break>
{
    typedef Stmt_base<IStatement_break> Base;
public:
    explicit Statement_break()
    : Base()
    {
    }
};

/// Implementation of the continue statement.
class Statement_continue : public Stmt_base<IStatement_continue>
{
    typedef Stmt_base<IStatement_continue> Base;
public:
    explicit Statement_continue()
    : Base()
    {
    }
};

/// Implementation of the return statement.
class Statement_return : public Stmt_base<IStatement_return>
{
    typedef Stmt_base<IStatement_return> Base;
public:

    /// Get the return expression.
    IExpression const *get_expression() const MDL_FINAL { return m_expr; }

    /// Set the return expression.
    void set_expression(IExpression const *expr) MDL_FINAL { m_expr = expr; }

    explicit Statement_return(IExpression const *expr)
    : Base()
    , m_expr(expr)
    {
    }

private:
    /// The returned expression or NULL.
    IExpression const *m_expr;
};

// -------------------------------------- statement factory --------------------------------------

Statement_factory::Statement_factory(Memory_arena &arena)
: Base()
, m_builder(arena)
{
}

/// Set position on a statement.
static void set_position(
    IStatement *stmnt,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    Position &pos = stmnt->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
}

/// Create a new invalid statement.
IStatement_invalid *Statement_factory::create_invalid(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_invalid *result = m_builder.create<Statement_invalid>();
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new compound statement.
IStatement_compound *Statement_factory::create_compound(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_compound *result = m_builder.create<Statement_compound>(m_builder.get_arena());
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new declaration statement.
IStatement_declaration *Statement_factory::create_declaration(
    IDeclaration const *decl,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_declaration *result = m_builder.create<Statement_declaration>(decl);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new expression statement.
IStatement_expression *Statement_factory::create_expression(
    IExpression const *expr,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_expression *result = m_builder.create<Statement_expression>(expr);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new conditional statement.
IStatement_if *Statement_factory::create_if(
    IExpression const *cond,
    IStatement const *then_stmnt,
    IStatement const *else_stmnt,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_if *result = m_builder.create<Statement_if>(cond, then_stmnt, else_stmnt);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new switch case.
IStatement_case *Statement_factory::create_switch_case(
    IExpression const *label,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_case *result = m_builder.create<Statement_case>(m_builder.get_arena(), label);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new switch statement.
IStatement_switch *Statement_factory::create_switch(
    IExpression const *cond,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_switch *result = m_builder.create<Statement_switch>(m_builder.get_arena(), cond);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new while loop.
IStatement_while *Statement_factory::create_while(
    IExpression const *cond,
    IStatement const *body,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_while *result = m_builder.create<Statement_while>(cond, body);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new do-while loop.
IStatement_do_while *Statement_factory::create_do_while(
    IExpression const *cond,
    IStatement const *body,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_do_while *result = m_builder.create<Statement_do_while>(cond, body);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new for loop with an initializing expression.
IStatement_for *Statement_factory::create_for(
    IStatement const *init,
    IExpression const *cond,
    IExpression const *update,
    IStatement const *body,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_for *result = m_builder.create<Statement_for>(init, cond, update, body);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new break statement.
IStatement_break *Statement_factory::create_break(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_break *result = m_builder.create<Statement_break>();
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new continue statement.
IStatement_continue *Statement_factory::create_continue(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_continue *result = m_builder.create<Statement_continue>();
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

/// Create a new return statement.
IStatement_return *Statement_factory::create_return(
    IExpression const *expr,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IStatement_return *result = m_builder.create<Statement_return>(expr);
    set_position(result,start_line,start_column,end_line,end_column);
    return result;
}

}  // mdl
}  // mi
