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

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_declarations.h"
#include "compiler_hlsl_locations.h"
#include "compiler_hlsl_stmts.h"

namespace mi {
namespace mdl {
namespace hlsl {

class Expr;

// Constructor.
Stmt::Stmt(
    Location const &loc)
: Base()
, m_loc(loc)
, m_parent(NULL)
{
}

// ---------------------------------- Stmt_invalid ----------------------------------

// Constructor.
Stmt_invalid::Stmt_invalid(
    Location const &loc)
: Base(loc)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_invalid::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_decl ----------------------------------

// Constructor.
Stmt_decl::Stmt_decl(
    Declaration *decl)
: Base(decl->get_location())
, m_decl(decl)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_decl::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_list ----------------------------------

// Constructor.
Stmt_list::Stmt_list(
    Location const &loc)
: Base(loc)
, m_stmts()
{
}

// Insert a statement into this compound at the end.
void Stmt_list::add_stmt(Stmt *stmt)
{
    m_stmts.push(stmt);
}

// ---------------------------------- Stmt_compound ----------------------------------

// Constructor.
Stmt_compound::Stmt_compound(
    Location const &loc)
: Base(loc)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_compound::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_expr ----------------------------------

// Constructor.
Stmt_expr::Stmt_expr(
    Location const &loc,
    Expr           *expr)
: Base(loc)
, m_expr(expr)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_expr::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_if ----------------------------------
// Constructor.
Stmt_if::Stmt_if(
    Location const       &loc,
    Expr                 *cond,
    Stmt                 *then_stmt,
    Stmt                 *else_stmt,
    HLSL_attribute const *attribute)
: Base(loc)
, m_cond(cond)
, m_then_stmt(then_stmt)
, m_else_stmt(else_stmt)
, m_attribute(attribute)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_if::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_case ----------------------------------

// Constructor.
Stmt_case::Stmt_case(
    Location const &loc,
    Expr           *label,
    Stmt_switch    *owner)
: Base(loc)
, m_label(label)
, m_owner(NULL)
, m_next_label(NULL)
, m_prev_label(NULL)
{
    if (owner != NULL)
        owner->add_label(this);
}

// Get the kind of statement.
Stmt::Kind Stmt_case::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_switch ----------------------------------

// Constructor.
Stmt_switch::Stmt_switch(
    Location const       &loc,
    Expr                 *cond,
    HLSL_attribute const *attribute)
: Base(loc)
, m_cond(cond)
, m_first_label(NULL)
, m_last_label(NULL)
, m_default_label(NULL)
, m_attribute(attribute)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_switch::get_kind() const
{
    return s_kind;
}

// Add a label.
void Stmt_switch::add_label(Stmt_case *label)
{
    HLSL_ASSERT(label != NULL && "case label must be non-NULL to add");
    HLSL_ASSERT(label->m_owner == NULL && "case label has alraedy an owner");

    if (label->get_label() == NULL) {
        // a default label
        HLSL_ASSERT(m_default_label == NULL && "switch has already a default label");

        m_default_label = label;
    }

    if (m_last_label != NULL) {
        m_last_label->m_next_label = label;
    }
    label->m_prev_label = m_last_label;
    label->m_next_label = NULL;
    label->m_owner      = this;

    m_last_label = label;

    if (m_first_label == NULL)
        m_first_label = label;
}

// ---------------------------------- Stmt_while ----------------------------------

// Constructor.
Stmt_while::Stmt_while(
    Location const       &loc,
    Stmt                 *cond,
    Stmt                 *body,
    HLSL_attribute const *attribute)
: Base(loc)
, m_cond(cond)
, m_body(body)
, m_attribute(attribute)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_while::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_do_while ----------------------------------

// Constructor.
Stmt_do_while::Stmt_do_while(
    Location const       &loc,
    Expr                 *cond,
    Stmt                 *body,
    HLSL_attribute const *attribute)
: Base(loc)
, m_cond(cond)
, m_body(body)
, m_attribute(attribute)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_do_while::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_for ----------------------------------

// Constructor.
Stmt_for::Stmt_for(
    Location const       &loc,
    Stmt                 *init,
    Stmt                 *cond,
    Expr                 *update,
    Stmt                 *body,
    HLSL_attribute const *attribute)
: Base(loc)
, m_init_stmt(init)
, m_cond(cond)
, m_update_expr(update)
, m_body(body)
, m_attribute(attribute)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_for::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_break ----------------------------------

// Constructor.
Stmt_break::Stmt_break(
    Location const &loc)
: Base(loc)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_break::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_continue ----------------------------------

// Constructor.
Stmt_continue::Stmt_continue(
    Location const &loc)
: Base(loc)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_continue::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_discard ----------------------------------

// Constructor.
Stmt_discard::Stmt_discard(
    Location const &loc)
: Base(loc)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_discard::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_return ----------------------------------

// Constructor.
Stmt_return::Stmt_return(
    Location const &loc,
    Expr           *expr)
: Base(loc)
, m_expr(expr)
{
}

// Get the kind of statement.
Stmt::Kind Stmt_return::get_kind() const
{
    return s_kind;
}

// ---------------------------------- Stmt_factory ----------------------------------

// Constructor.
Stmt_factory::Stmt_factory(
    Memory_arena &arena)
: Base()
, m_builder(arena)
{
}

// Create a new invalid statement.
Stmt_invalid *Stmt_factory::create_invalid(
    Location const &loc)
{
    return m_builder.create<Stmt_invalid>(loc);
}

// Stmt_factory::create a new declaration statement.
Stmt_decl *Stmt_factory::create_declaration(
    Declaration *decl)
{
    return m_builder.create<Stmt_decl>(decl);
}

// Stmt_factory::create a new compound statement.
Stmt_compound *Stmt_factory::create_compound(
    Location const &loc)
{
    return m_builder.create<Stmt_compound>(loc);
}

// Create a new compound statement with initial children.
Stmt_compound *Stmt_factory::create_compound(
    Location const    &loc,
    Array_ref<Stmt *> children)
{
    Stmt_compound *compound = create_compound(loc);
    for (size_t i = 0, n = children.size(); i < n; ++i) {
        compound->add_stmt(children[i]);
    }
    return compound;
}

// Stmt_factory::create a new expression statement.
Stmt_expr *Stmt_factory::create_expression(
    Location const &loc,
    Expr           *expr)
{
    return m_builder.create<Stmt_expr>(loc, expr);
}

// Stmt_factory::create a new conditional statement.
Stmt_if *Stmt_factory::create_if(
    Location const &loc,
    Expr           *cond,
    Stmt           *then_stmt,
    Stmt           *else_stmt)
{
    return m_builder.create<Stmt_if>(loc, cond, then_stmt, else_stmt);
}

// Stmt_factory::create a new case label.
Stmt_case *Stmt_factory::create_case_label(
    Location const &loc,
    Expr           *label,
    Stmt_switch    *owner)
{
    return m_builder.create<Stmt_case>(loc, label, owner);
}

// Stmt_factory::create a new switch statement.
Stmt_switch *Stmt_factory::create_switch(
    Location const &loc,
    Expr           *cond)
{
    return m_builder.create<Stmt_switch>(loc, cond);
}

// Stmt_factory::create a new while loop.
Stmt_while *Stmt_factory::create_while(
    Location const &loc,
    Stmt           *cond,
    Stmt           *body)
{
    return m_builder.create<Stmt_while>(loc, cond, body);
}

// Stmt_factory::create a new do-while loop.
Stmt_do_while *Stmt_factory::create_do_while(
    Location const &loc,
    Expr           *cond,
    Stmt           *body)
{
    return m_builder.create<Stmt_do_while>(loc, cond, body);
}

// Stmt_factory::create a new for loop with an initializing expression.
Stmt_for *Stmt_factory::create_for(
    Location const &loc,
    Stmt           *init,
    Stmt           *cond,
    Expr           *update,
    Stmt           *body)
{
    return m_builder.create<Stmt_for>(loc, init, cond, update, body);
}

// Stmt_factory::create a new break statement.
Stmt_break *Stmt_factory::create_break(
    Location const &loc)
{
    return m_builder.create<Stmt_break>(loc);
}

// Stmt_factory::create a new continue statement.
Stmt_continue *Stmt_factory::create_continue(
    Location const &loc)
{
    return m_builder.create<Stmt_continue>(loc);
}

// Stmt_factory::create a new discard statement.
Stmt_discard *Stmt_factory::create_discard(
    Location const &loc)
{
    return m_builder.create<Stmt_discard>(loc);
}

// Stmt_factory::create a new return statement.
Stmt_return *Stmt_factory::create_return(
    Location const &loc,
    Expr           *expr)
{
    return m_builder.create<Stmt_return>(loc, expr);
}

}  // hlsl
}  // mdl
}  // mi
