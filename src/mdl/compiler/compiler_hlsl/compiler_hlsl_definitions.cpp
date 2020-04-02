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

#include "compiler_hlsl_definitions.h"
#include "compiler_hlsl_types.h"
#include "compiler_hlsl_tools.h"

namespace mi {
namespace mdl {
namespace hlsl {


// ----------------------------- Definition -----------------------------

// Constructor.
Definition::Definition(
    Symbol         *sym,
    Type           *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base()
, m_sym(sym)
, m_type(type)
, m_decl(NULL)
, m_loc(loc)
, m_def_scope(parent_scope)
, m_next(NULL)
, m_same_prev(NULL)
, m_same_next(NULL)
, m_outer_def(outer)
, m_id(id)
{
}

// ----------------------------- Def_error -----------------------------

// Constructor.
Def_error::Def_error(
    Symbol         *sym,
    Type_error     *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
{
}

// Returns the kind of this definition.
Definition::Kind Def_error::get_kind() const
{
    return s_kind;
}

// Get the type of this definition.
Type_error *Def_error::get_type() const
{
    return cast<Type_error>(m_type);
}

// Get the declaration of the definition.
Declaration *Def_error::get_declaration() const
{
    return NULL;
}

// ----------------------------- Def_variable -----------------------------

// Constructor.
Def_variable::Def_variable(
    Symbol         *sym,
    Type           *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
{
}

// Returns the kind of this definition.
Definition::Kind Def_variable::get_kind() const
{
    return s_kind;
}

// Get the type of this definition.
Type *Def_variable::get_type() const
{
    return m_type;
}

// Get the declaration of the definition.
Declaration_variable *Def_variable::get_declaration() const
{
    return cast<Declaration_variable>(m_decl);
}

// ----------------------------- Def_param -----------------------------

// Constructor.
Def_param::Def_param(
    Symbol         *sym,
    Type           *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
{
}

// Returns the kind of this definition.
Definition::Kind Def_param::get_kind() const
{
    return s_kind;
}

// Get the type of this definition.
Type *Def_param::get_type() const
{
    return m_type;
}

// Get the declaration of the definition.
Declaration_param *Def_param::get_declaration() const
{
    return cast<Declaration_param>(m_decl);
}

// ----------------------------- Def_function -----------------------------

// Constructor.
Def_function::Def_function(
    Symbol         *sym,
    Type_function  *type,
    Semantics      semantics,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
, m_own_scope(NULL)
, m_semantics(semantics)
{
}

// Returns the kind of this definition.
Definition::Kind Def_function::get_kind() const
{
    return s_kind;
}

// Get the type of this definition.
Type_function *Def_function::get_type() const
{
    return cast<Type_function>(m_type);
}

// Get the declaration of the definition.
Declaration_function *Def_function::get_declaration() const
{
    return cast<Declaration_function>(m_decl);
}

// ----------------------------- Def_type -----------------------------

// Constructor.
Def_type::Def_type(
    Symbol         *sym,
    Type           *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
, m_own_scope(NULL)
{
}

// Returns the kind of this definition.
Definition::Kind Def_type::get_kind() const
{
    return s_kind;
}

// Get the declaration of the definition.
Declaration_struct *Def_type::get_declaration() const
{
    return cast<Declaration_struct>(m_decl);
}

// Change the type of this definition.
void Def_type::set_type(Type *type)
{
    HLSL_ASSERT(is<Type_error>(m_type));
    m_type = type;
}

// ----------------------------- Def_member -----------------------------

// Constructor.
Def_member::Def_member(
    Symbol         *sym,
    Type           *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
{
}

// Returns the kind of this definition.
Definition::Kind Def_member::get_kind() const
{
    return s_kind;
}

// Get the type of this definition.
Type *Def_member::get_type() const
{
    return m_type;
}

// Get the declaration of the definition.
Declaration_field *Def_member::get_declaration() const
{
    return cast<Declaration_field>(m_decl);
}

// ----------------------------- Def_operator -----------------------------

// Constructor.
Def_operator::Def_operator(
    Symbol         *sym,
    Type           *type,
    Location const *loc,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base(sym, type, loc, parent_scope, outer, id)
{
}

// Returns the kind of this definition.
Definition::Kind Def_operator::get_kind() const
{
    return s_kind;
}

// Get the type of this definition.
Type *Def_operator::get_type() const
{
    return m_type;
}

// Always returns NULL for operators.
Declaration *Def_operator::get_declaration() const
{
    return NULL;
}

// ----------------------------- Scope -----------------------------

// Creates a new environmental scope.
Scope::Scope(
    Scope            *parent,
    size_t           id,
    Definition const *owner_def,
    Type             *type)
: Base()
, m_definitions(NULL)
, m_owner_definition(owner_def)
, m_parent(parent)
, m_sub_scopes(NULL)
, m_last_sub_scope(NULL)
, m_next_subscope(NULL)
, m_prev_subscope(NULL)
, m_scope_type(type)
, m_id(id)
{
    // enter into sub scope list of the parent if any
    if (parent != NULL) {
        m_prev_subscope = parent->m_last_sub_scope;
        if (m_prev_subscope != NULL) {
            m_prev_subscope->m_next_subscope = this;
        } else {
            parent->m_sub_scopes = this;
        }
        parent->m_last_sub_scope = this;
    }
}

// Set the owner definition of this scope.
void Scope::set_owner_definition(Definition const *owner_def)
{
    m_owner_definition = owner_def;
}

// Re-enter the definitions for all defined symbols in the scope.
void Scope::enter_definitions(Definition_table *owner_table) const
{
    Definition *def, *next, *last;

    // restore entity definitions
    for (def = m_definitions; def != NULL; def = next) {
        next = def->m_next;

        // set the last declaration of the same type, so we can traverse the
        // chain towards begin
        for (last = def; last->m_same_next != NULL; last = last->m_same_next);

        owner_table->set_definition(last->m_sym, last);
    }
}

// Restore the definitions for all defined symbols in the scope.
void Scope::restore_definitions(Definition_table *owner_table)
{
    // restore entity definitions
    for (Definition *next, *def = m_definitions; def != NULL; def = next) {
        next = def->m_next;

        owner_table->restore_definition(def->m_sym, def->get_outer_def());
    }
}

// Find a definition inside this scope only.
Definition *Scope::find_definition_in_scope(Symbol *sym) const
{
    for (Definition *def = get_first_definition_in_scope();
        def != NULL;
        def = def->get_next_def_in_scope())
    {
        if (def->get_symbol() == sym) {
            // found: return the latest definition here, we always scan forward
            Definition *res;
            for (res = def; res->get_next_def() != NULL; res = res->get_next_def());
            return res;
        }
    }
    return NULL;
}

// Find the definition of the given ID in this scope only.
Definition *Scope::find_definition_in_scope(size_t id) const
{
    for (Definition *def = get_first_definition_in_scope();
        def != NULL;
        def = def->get_next_def_in_scope())
    {
        for (Definition *sdef = def; sdef != NULL; sdef = sdef->get_next_def()) {
            if (sdef->get_unique_id() == id) {
                // found: return the latest definition here, we always scan forward
                return sdef;
            }
        }
    }
    return NULL;
}

// Find a definition inside this scope or parent scopes.
Definition *Scope::find_def_in_scope_or_parent(Symbol *sym) const
{
    Scope const *scope = this;
    for (;;) {
        Definition *def = scope->find_definition_in_scope(sym);
        if (def)
            return def;
        if (scope->m_parent == NULL)
            return NULL;
        scope = scope->m_parent;
    }
}

// Visit a scope
void Scope::walk(IDefinition_visitor const *visitor) const
{
    for (Definition const *def = get_first_definition_in_scope();
        def != NULL;
        def = def->get_next_def_in_scope())
    {
        // inside a scope, the definitions are always linked "forward", i.e. the first
        // definition of a set is linked
        for (Definition const *odef = def; odef != NULL; odef = odef->get_next_def()) {
            visitor->visit(odef);
        }
    }

    for (Scope const *scope = m_sub_scopes; scope != NULL; scope = scope->m_next_subscope) {
        scope->walk(visitor);
    }
}

// Returns true if this is an empty scope that can be thrown away.
bool Scope::is_empty() const
{
    if (m_definitions != NULL || m_sub_scopes != NULL)
        return false;
    if (m_scope_type != NULL) {
        // type scopes are referenced and expected to exists
        return false;
    }
    if (m_owner_definition != NULL) {
        // has an owner ...
        return false;
    }
    return true;
}

// Remove this scope from its parent sub-scopes.
void Scope::remove_from_parent()
{
    HLSL_ASSERT(m_parent != NULL && "cannot remove from non-existing parent");

    if (m_prev_subscope == NULL) {
        // removing the first scope
        m_parent->m_sub_scopes = m_next_subscope;
    } else {
        m_prev_subscope->m_next_subscope = m_next_subscope;
    }
    if (m_next_subscope == NULL) {
        // removing the last scope
        m_parent->m_last_sub_scope = m_prev_subscope;
    } else {
        m_next_subscope->m_prev_subscope = m_prev_subscope;
    }
    // detach from parent
    m_parent        = NULL;
    m_next_subscope = NULL;
    m_prev_subscope = NULL;
}

// ----------------------------- Definition_table -----------------------------

// Constructor.
Definition_table::Definition_table(Compilation_unit &unit)
: m_unit(unit)
, m_curr_scope(NULL)
, m_predef_scope(NULL)
, m_global_scope(NULL)
, m_free_scopes(NULL)
, m_next_definition_id(0)
, m_arena(unit.get_allocator())
, m_builder(m_arena)
, m_type_scopes(0, Type_scope_map::hasher(), Type_scope_map::key_equal(), unit.get_allocator())
, m_definitions(unit.get_allocator())
{
    std::fill_n(
        &m_operator_definitions[0], dimension_of(m_operator_definitions), (Definition *)0);

    // create initial scopes
    m_predef_scope = enter_scope(NULL);
    m_global_scope = enter_scope(NULL);
    leave_scope();
    leave_scope();
}

// Enter a new scope owned by a definition.
Scope *Definition_table::enter_scope(Definition const *def)
{
    // create a new scope
    Scope *scope = create_scope(m_curr_scope, ++m_next_definition_id, def);
    m_curr_scope = scope;
    return scope;
}

// Enter a new scope created by a type declaration.
Scope *Definition_table::enter_scope(
    Type             *type,
    Definition const *type_def)
{
    // create a new scope
    Scope *scope = create_scope(m_curr_scope, ++m_next_definition_id, type_def, type);

    // associate it with the given type
    m_type_scopes[type] = scope;
    m_curr_scope = scope;
    return scope;
}

// Leave the current scope.
void Definition_table::leave_scope()
{
    Scope *scope = m_curr_scope;
    m_curr_scope = scope->get_parent();
    scope->restore_definitions(this);
}

// Reopen an already leaved scope.
void Definition_table::reopen_scope(Scope *scope)
{
    // can only open a child scope of the current one
    HLSL_ASSERT(scope->get_parent() == m_curr_scope);

    scope->enter_definitions(this);
    m_curr_scope = scope;
}

// Do a transition to the given scope. This scope will be the new current one.
void Definition_table::transition_to_scope(Scope *scope)
{
    // shortcut for often needed case: just reopen a scope
    if (scope && scope->get_parent() == m_curr_scope) {
        reopen_scope(scope);
        return;
    }

    IAllocator *alloc = m_unit.get_allocator();

    // otherwise a real transition: find common prefix
    vector<Scope *>::Type curr_stack(alloc);
    for (Scope *curr = m_curr_scope; curr != NULL; curr = curr->get_parent()) {
        curr_stack.push_back(curr);
    }

    vector<Scope *>::Type new_stack(alloc);
    for (Scope *curr = scope; curr != NULL; curr = curr->get_parent()) {
        new_stack.push_back(curr);
    }

    size_t curr_idx = curr_stack.size();
    size_t new_idx  = new_stack.size();

    if (curr_idx > 0 && new_idx > 0) {
        while (curr_stack[curr_idx - 1] == new_stack[new_idx - 1]) {
            --curr_idx;
            --new_idx;
            if (curr_idx == 0 || new_idx == 0) {
                break;
            }
        }
    }

    // remove until prefix is reached
    for (size_t i = 0; i < curr_idx; ++i)
        leave_scope();

    // reopen until top is reached
    for (size_t i = new_idx; i > 0; --i)
        reopen_scope(new_stack[i - 1]);
}

// Enter a new (entity) definition.
Definition *Definition_table::enter_definition(
    Definition::Kind          kind,
    Symbol                   *symbol,
    Type                     *type,
    Def_function::Semantics   semantics,
    Location const           *loc)
{
    Definition *new_def = NULL;
    HLSL_ASSERT(symbol != NULL);

    Definition *curr_def = get_definition(symbol);

    bool first_in_scope = curr_def == NULL || curr_def->get_def_scope() != m_curr_scope;

    Definition *outer = first_in_scope ? curr_def : NULL;

    switch (kind) {
    case Definition::DK_ERROR:
        new_def = m_builder.create<Def_error>(
            symbol, cast<Type_error>(type), loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    case mi::mdl::hlsl::Definition::DK_TYPE:
        new_def = m_builder.create<Def_type>(
            symbol, type, loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    case mi::mdl::hlsl::Definition::DK_FUNCTION:
        new_def = m_builder.create<Def_function>(
            symbol, cast<Type_function>(type), semantics,
            loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    case mi::mdl::hlsl::Definition::DK_VARIABLE:
        new_def = m_builder.create<Def_variable>(
            symbol, type, loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    case mi::mdl::hlsl::Definition::DK_MEMBER:
        new_def = m_builder.create<Def_member>(
            symbol, type, loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    case mi::mdl::hlsl::Definition::DK_PARAMETER:
        new_def = m_builder.create<Def_param>(
            symbol, type, loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    case mi::mdl::hlsl::Definition::DK_OPERATOR:
        new_def = m_builder.create<Def_operator>(
            symbol, type, loc, m_curr_scope, outer, ++m_next_definition_id);
        break;
    }

    if (first_in_scope) {
        // first definition inside this scope
        m_curr_scope->add_definition(new_def);
    } else {
        // there is already a definition for this symbol in the current scope, append it
        new_def->link_same_def(curr_def);
    }

    set_definition(symbol, new_def);
    return new_def;
}

// Enter a new function definition.
Def_function *Definition_table::enter_function_definition(
    Symbol                  *symbol,
    Type_function           *type,
    Def_function::Semantics  semantics,
    Location const          *loc)
{
    return cast<Def_function>(
        enter_definition(mi::mdl::hlsl::Definition::DK_FUNCTION, symbol, type, semantics, loc));
}

/// Enter a new variable definition.
Def_variable *Definition_table::enter_variable_definition(
    Symbol         *symbol,
    Type           *type,
    Location const *loc)
{
    return cast<Def_variable>(enter_definition(
        mi::mdl::hlsl::Definition::DK_VARIABLE, symbol, type, Def_function::DS_UNKNOWN, loc));
}

// Enter a new parameter definition.
Def_param *Definition_table::enter_parameter_definition(
    Symbol         *symbol,
    Type           *type,
    Location const *loc)
{
    return cast<Def_param>(enter_definition(
        mi::mdl::hlsl::Definition::DK_PARAMETER, symbol, type, Def_function::DS_UNKNOWN, loc));
}

// Enter a new member definition.
Def_member *Definition_table::enter_member_definition(
    Symbol         *symbol,
    Type           *type,
    Location const *loc)
{
    return cast<Def_member>(
        enter_definition(
            mi::mdl::hlsl::Definition::DK_MEMBER,
            symbol,
            type,
            Def_function::DS_UNKNOWN,
            loc));
}

// Enter a new operator definition.
Def_operator *Definition_table::enter_operator_definition(
    Expr::Operator kind,
    Symbol         *symbol,
    Type           *type)
{
    return cast<Def_operator>(
        enter_definition(
            mi::mdl::hlsl::Definition::DK_OPERATOR,
            symbol,
            type,
            Def_function::DS_UNKNOWN,
            /*loc=*/NULL));
}


// Enter an error definition for the given symbol.
Definition *Definition_table::enter_error(
    Symbol     *symbol,
    Type_error *err_type)
{
    MDL_ASSERT(get_definition(symbol) == NULL);

    // no definition inside this scope
    Definition *err_def = m_builder.create<Def_error>(
        symbol, err_type, (Location *)0, m_curr_scope, (Definition *)0, ++m_next_definition_id);

    m_curr_scope->add_definition(err_def);
    set_definition(symbol, err_def);
    return err_def;
}

// find a type scope
Scope *Definition_table::get_type_scope(
    Type *type) const
{
    Type_scope_map::const_iterator it = m_type_scopes.find(type);
    if (it != m_type_scopes.end())
        return it->second;
    return NULL;
}

// Return the current definition for the given symbol.
Definition *Definition_table::get_definition(
    Symbol *symbol) const
{
    size_t id   = symbol->get_id();
    size_t size = m_definitions.size();
    if (id < size) {
        return m_definitions[id];
    }
    return NULL;
}

// Set the current definition for a symbol in this definition table.
void Definition_table::set_definition(
    Symbol     *sym,
    Definition *def)
{
    size_t id   = sym->get_id();
    size_t size = m_definitions.size();

    if (id >= size) {
        size = ((id + 1) + 0x0F) & ~0x0F;
        m_definitions.resize(size, NULL);
    }
    if (def != NULL) {
        // because we have def-before-use in the global scope, we must
        // set the link here, it might NOT be static
        Definition *outer_def = m_definitions[id];
        if (outer_def != NULL && outer_def->get_def_scope() == def->get_def_scope()) {
            // The previous definition is in this scope, get its outer scope.
            // This happens because set_definition() is called by enter_definition().
            outer_def = outer_def->get_outer_def();
        }
        MDL_ASSERT(outer_def == NULL || outer_def->get_def_scope() != def->get_def_scope());
        def->set_outer_def(outer_def);
    }
    m_definitions[id] = def;
}

// Restore the current definition for a symbol in this definition table.
void Definition_table::restore_definition(
    Symbol     *sym,
    Definition *def)
{
    size_t id   = sym->get_id();
    size_t size = m_definitions.size();

    if (id >= size) {
        size = ((id + 1) + 0x0F) & ~0x0F;
        m_definitions.resize(size, NULL);
    }
    m_definitions[id] = def;
}

// Return the definition for an operator in this definition table.
Definition *Definition_table::get_operator_definition(
    Expr::Operator op) const
{
    return m_operator_definitions[op];
}

// Set the definition for an operator in this definition table.
void Definition_table::set_operator_definition(
    Expr::Operator op,
    Definition     *def)
{
    m_operator_definitions[op] = def;
}

// Walk over all definitions of this definition table.
void Definition_table::walk(IDefinition_visitor const *visitor) const
{
    m_global_scope->walk(visitor);
}

// Clear the definition table.
void Definition_table::clear()
{
    m_curr_scope         = NULL;
    m_next_definition_id = 0;
    m_arena.drop(NULL);
    m_type_scopes.clear();
    m_definitions.clear();

    std::fill_n(
        &m_operator_definitions[0], dimension_of(m_operator_definitions), (Definition *)0);

    // create initial scopes
    m_predef_scope = enter_scope(NULL);
    m_global_scope = enter_scope(NULL);
    leave_scope();
    leave_scope();
}

// Remove an empty scope from the scope tree.
void Definition_table::remove_empty_scope(Scope *scope)
{
    MDL_ASSERT(scope->is_empty() && "Removal of non-empty scopes is forbidden");

    scope->remove_from_parent();

    if (scope->get_unique_id() == m_next_definition_id) {
        // regain the id
        --m_next_definition_id;
    }

    scope->m_next_subscope = m_free_scopes;
    m_free_scopes = scope;
}

// Create a new scope.
Scope *Definition_table::create_scope(
    Scope            *parent,
    size_t           id,
    Definition const *owner_def,
    Type             *type)
{
    if (Scope *s = m_free_scopes) {
        m_free_scopes = s->m_next_subscope;

        new (s) Scope(parent, id, owner_def, type);
        return s;
    }
    return m_builder.create<Scope>(parent, id, owner_def, type);
}

}  // hlsl
}  // mdl
}  // mi

