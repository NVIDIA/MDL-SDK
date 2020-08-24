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

#include <algorithm>

#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_names.h>

#include <cstdio>

#include "compilercore_rawbitset.h"
#include "compilercore_printers.h"
#include "compilercore_symbols.h"
#include "compilercore_def_table.h"
#include "compilercore_modules.h"
#include "compilercore_allocator.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"
#include "compilercore_dynamic_memory.h"
#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

// ----------------------------- Implementation -----------------------------

// Returns the kind of this definition.
IDefinition::Kind Definition::get_kind() const
{
    return m_kind;
}

// Get the symbol of the definition.
ISymbol const *Definition::get_symbol() const
{
    return m_sym;
}

// Get the type of the definition.
IType const *Definition::get_type() const
{
    return m_type;
}

// Get the declaration of the definition.
IDeclaration const *Definition::get_declaration() const
{
    return m_decl;
}

// Get the default expression of a parameter of a function, constructor or annotation.
IExpression const *Definition::get_default_param_initializer(int param_idx) const
{
    // not allowed to call from an imported definition
    MDL_ASSERT(m_original_module_import_idx == 0 &&
           "get_default_param_initializer() called on imported entity");

    if ((m_kind == DK_FUNCTION || m_kind == DK_CONSTRUCTOR || m_kind == DK_ANNOTATION) &&
        m_parameter_inits != NULL)
    {
        if (0 <= param_idx && size_t(param_idx) < m_parameter_inits->count)
            return m_parameter_inits->exprs[param_idx];
    }
    return NULL;
}

// S// Set the default expression of a parameter of a function, constructor or annotation.
void Definition::set_default_param_initializer(size_t param_idx, IExpression const *expr)
{
    MDL_ASSERT(m_kind == DK_FUNCTION || m_kind == DK_CONSTRUCTOR || m_kind == DK_ANNOTATION);
    MDL_ASSERT(m_parameter_inits && "no parameter initializers allocated");
    MDL_ASSERT(param_idx < m_parameter_inits->count);
    m_parameter_inits->exprs[param_idx] = expr;
}

// Set the scope that this definition creates.
void Definition::set_own_scope(Scope *scope)
{
    m_own_scope = scope;
    scope->set_owner_definition(this);
}

// Copy the default initializers from one definition to another.
void Definition::copy_initializers(Module *module, Definition const *prev_def)
{
    MDL_ASSERT(m_kind == DK_FUNCTION || m_kind == DK_CONSTRUCTOR);
    MDL_ASSERT(
        m_owner_module_id == prev_def->m_owner_module_id &&
        "Initializers should only be copied inside the same module");

    if (prev_def->m_parameter_inits != NULL) {
        if (m_parameter_inits == NULL) {
            module->allocate_initializers(this, prev_def->m_parameter_inits->count);
        }
        for (size_t i = 0, n = prev_def->m_parameter_inits->count; i < n; ++i)
            m_parameter_inits->exprs[i] = prev_def->m_parameter_inits->exprs[i];
    } else {
        m_parameter_inits = NULL;
    }
}

// Return the value of an enum constant or a global constant.
IValue const *Definition::get_constant_value() const
{
    if (m_kind == DK_ENUM_VALUE || m_kind == DK_CONSTANT)
        return m_value;
    return NULL;
}

// Return the field index of a field member.
int Definition::get_field_index() const
{
    if (m_kind == DK_MEMBER)
        return m_u.field_index;
    return -1;
}

// Return the semantics of a function/constructor.
IDefinition::Semantics Definition::get_semantics() const
{
    if (m_kind == DK_FUNCTION    || m_kind == DK_OPERATOR ||
        m_kind == DK_CONSTRUCTOR || m_kind == DK_ANNOTATION)
        return m_u.sema_code;
    return DS_UNKNOWN;
}


// Return the parameter index of a parameter.
int Definition::get_parameter_index() const
{
    if (m_kind == DK_PARAMETER)
        return m_u.param_index;
    return -1;
}

// Return the namespace of a namespace alias.
ISymbol const *Definition::get_namespace() const
{
    if (m_kind == DK_NAMESPACE)
        return m_u.name_space;
    return NULL;
}

// Get the prototype declaration of the definition if any.
IDeclaration const *Definition::get_prototype_declaration() const
{
    return m_proto_decl;
}

// Get a boolean property of this definition.
bool Definition::get_property(Property prop) const
{
    switch (prop) {
    case DP_IS_OVERLOADED:
        if (m_kind == DK_FUNCTION || m_kind == DK_CONSTRUCTOR) {
            for (Definition const *o = get_prev_def(); o != NULL; o = o->get_prev_def()) {
                if (o->m_kind == m_kind)
                    return true;
            }
            for (Definition const *o = get_next_def(); o != NULL; o = o->get_next_def()) {
                if (o->m_kind == m_kind)
                    return true;
            }
        }
        break;
    case DP_IS_UNIFORM:
        return has_flag(DEF_IS_UNIFORM);
    case DP_NOT_WRITTEN:
        return !has_flag(DEF_IS_WRITTEN);
    case DP_IS_IMPORTED:
        // a definition is imported, if its original module is not its owner module
        return m_original_module_import_idx != 0;
    case DP_NEED_REFERENCE:
        // operators that require an lvalue must pass this by reference
        return has_flag(DEF_OP_LVALUE);
    case DP_ALLOW_INLINE:
        // a function without the noinline marker can be inlined
        return m_kind == DK_FUNCTION && !has_flag(DEF_NO_INLINE);
    case DP_IS_EXPORTED:
        return has_flag(Definition::DEF_IS_EXPORTED);
    case DP_IS_LOCAL_FUNCTION:
        // a function is local, if it is not exported, imported, or compiler known
        if (m_kind == DK_FUNCTION) {
            return
                get_semantics() == IDefinition::DS_UNKNOWN &&
                !has_flag(DEF_IS_EXPORTED) &&
                !has_flag(DEF_IS_IMPORTED);
        }
        return false;
    case DP_IS_WRITTEN:
        return has_flag(Definition::DEF_IS_WRITTEN);
    case DP_USES_STATE:
        return has_flag(Definition::DEF_USES_STATE);
    case DP_USES_TEXTURES:
        return has_flag(Definition::DEF_USES_TEXTURES);
    case DP_CAN_THROW_BOUNDS:
        return has_flag(Definition::DEF_CAN_THROW_BOUNDS);
    case DP_CAN_THROW_DIVZERO:
        return has_flag(Definition::DEF_CAN_THROW_DIVZERO);
    case DP_IS_VARYING:
        return has_flag(Definition::DEF_IS_VARYING);
    case DP_READ_TEX_ATTR:
        return has_flag(Definition::DEF_READ_TEXTURE_ATTR);
    case DP_READ_LP_ATTR:
        return has_flag(Definition::DEF_READ_LP_ATTR);
    case DP_USES_VARYING_STATE:
        return has_flag(Definition::DEF_USES_VARYING_STATE);
    case DP_CONTAINS_DEBUG:
        return has_flag(Definition::DEF_USES_DEBUG_CALLS);
    case DP_USES_OBJECT_ID:
        return has_flag(Definition::DEF_USES_OBJECT_ID);
    case DP_USES_TRANSFORM:
        return has_flag(Definition::DEF_USES_TRANSFORM);
    case DP_USES_NORMAL:
        return has_flag(Definition::DEF_USES_NORMAL);
    case DP_IS_NATIVE:
        return has_flag(Definition::DEF_IS_NATIVE);
    case DP_IS_CONST_EXPR:
        return has_flag(Definition::DEF_IS_CONST_EXPR);
    case DP_USES_DERIVATIVES:
        return has_flag(Definition::DEF_USES_DERIVATIVES);
    case DP_USES_SCENE_DATA:
        return has_flag(Definition::DEF_USES_SCENE_DATA);
    }
    return false;
}

// Return the position of this definition if any.
Position const *Definition::get_position() const
{
    return m_pos;
}

// Set the position of this definition if any.
void Definition::set_position(Position const *pos)
{
    m_pos = pos;
}

// Return the mask specifying which parameters of a function are derivable.
unsigned Definition::get_parameter_derivable_mask() const
{
    return m_parameter_deriv_mask;
}

// Change the type of the definition.
void Definition::set_type(IType const *type)
{
    MDL_ASSERT(m_kind == DK_VARIABLE || m_kind == DK_CONSTANT || is<IType_error>(type));
    m_type = type;
}

// Set the declaration of the definition.
void Definition::set_declaration(IDeclaration const *decl)
{
    m_decl = decl;
}

// Set the prototype declaration of the definition.
void Definition::set_prototype_declaration(IDeclaration const *decl)
{
    MDL_ASSERT(m_kind == DK_FUNCTION);
    m_proto_decl = decl;
}

// Set the value of an enum or global constant.
void Definition::set_constant_value(IValue const *value)
{
    MDL_ASSERT(m_kind == DK_ENUM_VALUE || m_kind == DK_CONSTANT);
    m_value = value;
}

// Set the field index of a member field.
void Definition::set_field_index(int index)
{
    MDL_ASSERT(m_kind == DK_MEMBER);
    m_u.field_index = index;
}

// Set the semantic of a function or constructor.
void Definition::set_semantic(Semantics sema)
{
    MDL_ASSERT(
        m_kind == DK_FUNCTION ||
        m_kind == DK_CONSTRUCTOR ||
        (m_kind == DK_OPERATOR && semantic_is_operator(sema)) ||
        (m_kind == DK_ANNOTATION && semantic_is_annotation(sema))
    );
    MDL_ASSERT(sema != DS_UNKNOWN);
    m_u.sema_code = sema;

    // handle some specials
    switch (sema) {
    case DS_INTRINSIC_TEX_WIDTH:
    case DS_INTRINSIC_TEX_HEIGHT:
    case DS_INTRINSIC_TEX_DEPTH:
        set_flag(DEF_READ_TEXTURE_ATTR);
        break;
    case DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
    case DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
        set_flag(DEF_READ_LP_ATTR);
        break;
    default:
        break;
    }
}

// Set the parameter index of a member field.
void Definition::set_parameter_index(int index)
{
    MDL_ASSERT(m_kind == DK_PARAMETER);
    m_u.param_index = index;
}

// Set the namespace of a namespace alias.
void Definition::set_namespace(
    ISymbol const *name_space)
{
    MDL_ASSERT(m_kind == DK_NAMESPACE);
    m_u.name_space = name_space;
}

// Constructor.
Definition::Definition(
    Kind           kind,
    size_t         module_id,
    ISymbol const  *sym,
    IType const    *type,
    Position const *pos,
    Scope          *parent_scope,
    Definition     *outer,
    size_t         id)
: Base()
, m_kind(kind)
, m_unique_id(id)
, m_owner_module_id(module_id)
, m_original_unique_id(id)
, m_original_module_import_idx(0)
, m_sym(sym)
, m_type(type)
, m_parameter_inits(NULL)
, m_def_scope(parent_scope)
, m_own_scope(NULL)
, m_pos(pos)
, m_decl(NULL)
, m_proto_decl(NULL)
, m_next(NULL)
, m_same_prev(NULL)
, m_same_next(NULL)
, m_outer_def(outer)
, m_definite_def(NULL)
, m_value(NULL)
, m_version_flags(0)
, m_flags()
, m_parameter_deriv_mask(0)
{
    m_u.code = 0;
}

// Creates a new definition from an imported one.
Definition::Definition(
    Definition const &other,
    ISymbol const    *imp_sym,
    IType const      *imp_type,
    Position const   *imp_pos,
    Scope            *parent_scope,
    Definition       *outer,
    size_t           module_id,
    size_t           id,
    size_t           owner_import_idx)
: Base()                
, m_kind(other.m_kind)
, m_unique_id(id)
, m_owner_module_id(module_id)
, m_original_unique_id(other.m_original_unique_id)
, m_original_module_import_idx(owner_import_idx)
, m_sym(imp_sym)
, m_type(imp_type)
, m_parameter_inits(NULL)
, m_def_scope(parent_scope)
, m_own_scope(NULL)
, m_pos(imp_pos)
, m_decl(NULL)
, m_proto_decl(NULL)
, m_next(NULL)
, m_same_prev(NULL)
, m_same_next(NULL)
, m_outer_def(outer)
, m_definite_def(NULL)
, m_value(NULL)
, m_version_flags(other.m_version_flags)
, m_flags(other.m_flags)
, m_parameter_deriv_mask(other.m_parameter_deriv_mask)
{
    m_u.code = 0;

    // do not import the "exported" flag
    m_flags.clear_bit(Definition::DEF_IS_EXPORTED);
    // but set the "imported" flag
    m_flags.set_bit(Definition::DEF_IS_IMPORTED);
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
Definition *Scope::find_definition_in_scope(ISymbol const *sym) const
{
    for (Definition *def = get_first_definition_in_scope();
        def != NULL;
        def = def->get_next_def_in_scope())
    {
        if (def->get_sym() == sym) {
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
Definition *Scope::find_def_in_scope_or_parent(ISymbol const *sym) const
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

// Find a named subscope.
Scope *Scope::find_named_subscope(ISymbol const *name) const
{
    for (Scope *scope = m_sub_scopes; scope != NULL; scope = scope->m_next_subscope) {
        if (scope->m_scope_type == NULL && scope->m_scope_name == name)
            return scope;
    }
    return NULL;
}

// Get the first named sub-scope.
Scope *Scope::get_first_named_subscope() const
{
    for (Scope *scope = m_sub_scopes; scope != NULL; scope = scope->m_next_subscope) {
        if (scope->m_scope_type == NULL && scope->m_scope_name != NULL)
            return scope;
    }
    return NULL;
}

// Get the next named sub-scope.
Scope *Scope::get_next_named_subscope() const
{
    for (Scope *scope = m_next_subscope; scope != NULL; scope = scope->m_next_subscope) {
        if (scope->m_scope_type == NULL && scope->m_scope_name != NULL)
            return scope;
    }
    return NULL;
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
    if (m_scope_type != NULL || m_scope_name != NULL) {
        // type or named scopes are referenced and expected to exists
        return false;
    }
    if (m_owner_definition != NULL) {
        // has an owner ...
        return false;
    }
    return true;
}

// Collect the definitions of all enum values of a given type.
void Scope::collect_enum_values(
    IType_enum const *e_type,
    Definition_list  &values)
{
    for (Definition const *def = m_definitions; def != NULL; def = def->m_next) {
        // no overload on enum values, so it is enough to check this
        if (def->get_kind() != IDefinition::DK_ENUM_VALUE)
            continue;
        if (def->get_type() == e_type)
            values.push_back(def);
    }
}

// Remove this scope from its parent sub-scopes.
void Scope::remove_from_parent()
{
    MDL_ASSERT(m_parent != NULL && "cannot remove from non-existing parent");

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


// Creates a new environmental scope.
Scope::Scope(
    Scope            *parent,
    size_t           id,
    Definition const *owner_def,
    IType const      *type,
    ISymbol const    *name)
: Base()
, m_definitions(NULL)
, m_owner_definition(owner_def)
, m_parent(parent)
, m_sub_scopes(NULL)
, m_last_sub_scope(NULL)
, m_next_subscope(NULL)
, m_prev_subscope(NULL)
, m_scope_type(type)
, m_scope_name(name)
, m_id(id)
, m_depth(parent != NULL ? parent->get_depth() + 1 : 0)
{
    // enter into subscope list of the parent if any
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

// Creates a new named scope.
Scope::Scope(
    Scope         *parent,
    ISymbol const *name,
    size_t        id)
: Base()
, m_definitions(NULL)
, m_owner_definition(NULL)
, m_parent(parent)
, m_sub_scopes(NULL)
, m_last_sub_scope(NULL)
, m_next_subscope(NULL)
, m_prev_subscope(NULL)
, m_scope_type(NULL)
, m_scope_name(name)
, m_id(id)
, m_depth(parent != NULL ? parent->get_depth() + 1 : 0)
{
    // enter into subscope list of the parent if any
    if (parent != NULL) {
        MDL_ASSERT(parent->find_named_subscope(name) == NULL && "Scope already exists");
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

/// Indentation printer.
static void indent(Printer *printer, size_t depth) {
    for (size_t i = 0; i < depth; ++i)
        printer->print("  ");
}

/// Prints a definition.
static void print_def(Printer *printer, Definition const *def)
{
    IType const   *type = def->get_type();
    ISymbol const *name = def->get_sym();

    if (def->has_flag(Definition::DEF_IS_EXPORTED))
        printer->print("export ");
    if (def->has_flag(Definition::DEF_IS_IMPORTED))
        printer->print("imported ");
    if (def->has_flag(Definition::DEF_IS_EXPLICIT))
        printer->print("explicit ");
    if (def->has_flag(Definition::DEF_IS_EXPLICIT_WARN))
        printer->print("explicit_warn ");
    if (def->has_flag(Definition::DEF_IS_UNIFORM))
        printer->print("uniform ");
    if (def->has_flag(Definition::DEF_IS_VARYING))
        printer->print("varying ");

    IDefinition::Kind kind = def->get_kind();
    switch (kind) {
    case IDefinition::DK_ERROR:
        printer->print("ERROR ");
        break;
    case IDefinition::DK_CONSTANT:
        printer->print("const ");
        break;
    case IDefinition::DK_ENUM_VALUE:
        printer->print("enum value ");
        break;
    case IDefinition::DK_ANNOTATION:
        printer->print("annotation ");
        break;
    case IDefinition::DK_TYPE:
        printer->print("type ");
        break;
    case IDefinition::DK_FUNCTION:
        printer->print("func ");
        if (!def->has_flag(Definition::DEF_IS_REACHABLE)) {
            printer->print("(unreachable) ");
        }
        break;
    case IDefinition::DK_VARIABLE:
        if (def->has_flag(Definition::DEF_IS_LET_TEMPORARY))
            printer->print("temporary ");
        else
            printer->print("var ");
        break;
    case IDefinition::DK_MEMBER:
        printer->print("member ");
        break;
    case IDefinition::DK_CONSTRUCTOR:
        if (def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR))
            printer->print("const ");
        printer->print("constructor ");
        break;
    case IDefinition::DK_PARAMETER:
        printer->print("param ");
        break;
    case IDefinition::DK_ARRAY_SIZE:
        printer->print("size ");
        break;
    case IDefinition::DK_OPERATOR:
        printer->print("operator ");
        break;
    case IDefinition::DK_NAMESPACE:
        printer->print("namespace ");
        break;
    }

    printer->print_type(type, name);
    if (!is<IType_function>(type)) {
        printer->print(" ");
        printer->print(name);
    }

    if (kind == IDefinition::DK_CONSTANT || kind == IDefinition::DK_ENUM_VALUE) {
        printer->print(" = ");
        printer->print(def->get_constant_value());
    }

    if (!def->has_flag(Definition::DEF_IS_USED)) {
        printer->print(" (unused)");
    }
    if (def->has_flag(Definition::DEF_IS_UNUSED)) {
        printer->print(" [[unused()]]");
    }
    if (def->has_flag(Definition::DEF_IS_WRITTEN)) {
        printer->print(" (written)");
    }
    if (def->has_flag(Definition::DEF_USES_DERIVATIVES)) {
        printer->print(" (derivatives)");
    }
}

// Serialize this scope.
void Scope::serialize(Definition_table const &dt, Module_serializer &serializer) const
{
    typedef ptr_hash_set<Scope const>::Type Visited_set;

    Visited_set owned_scopes(
        0, Visited_set::hasher(), Visited_set::key_equal(), serializer.get_allocator());

    serializer.write_section_tag(Serializer::ST_SCOPE_START);

    DOUT(("Scope {\n"));
    INC_SCOPE();

    // write the owner definition
    if (m_owner_definition != NULL) {
        serializer.write_bool(true);

        Tag_t owner_tag = serializer.get_definition_tag(m_owner_definition);
        serializer.write_encoded_tag(owner_tag);

        DOUT(("owner def %u\n", unsigned(owner_tag)));
    } else {
        serializer.write_bool(false);
    }

    // write the scope type
    if (m_scope_type != NULL) {
        serializer.write_bool(true);

        Tag_t type_tag = serializer.get_type_tag(m_scope_type);
        serializer.write_encoded_tag(type_tag);

        DOUT(("scope type %u\n", unsigned(type_tag)));
    } else {
        serializer.write_bool(false);
    }

    // write the scope name
    if (m_scope_name != NULL) {
        serializer.write_bool(true);

        Tag_t name_tag = serializer.get_symbol_tag(m_scope_name);
        serializer.write_encoded_tag(name_tag);

        DOUT(("scope name %u (%s)\n", unsigned(name_tag), m_scope_name->get_name()));
    } else {
        serializer.write_bool(false);
    }

    // serialize children
    typedef list<Definition const *>::Type Def_list;
    Def_list all_defs(serializer.get_allocator());

    for (Definition const *def = m_definitions; def != NULL; def = def->m_next) {
        all_defs.push_back(def);
    }

    // process definitions in reverse order, just as they occurred in the source
    for (Def_list::const_reverse_iterator it = all_defs.rbegin(), end = all_defs.rend();
        it != end;
        ++it)
    {
        Definition const *def_set = *it;

        for (Definition const *def = def_set; def != NULL; def = def->m_same_next) {
            // serialize this definition
            dt.serialize_def(def, serializer);

            // serialize the scope of the definition
            if (Scope const *own_scope = def->get_own_scope()) {
                own_scope->serialize(dt, serializer);
                owned_scopes.insert(own_scope);
            }
        }
    }

    // serialize all not yet serialized sub-scopes
    for (Scope const *sub_scope = m_sub_scopes;
        sub_scope != NULL;
        sub_scope = sub_scope->m_next_subscope)
    {
        if (owned_scopes.find(sub_scope) == owned_scopes.end())
            sub_scope->serialize(dt, serializer);
    }

    // ready, write marker
    serializer.write_section_tag(Serializer::ST_SCOPE_END);

    DEC_SCOPE();
    DOUT(("Scope }\n"));
}

// Deserialize this scope.
void Scope::deserialize(
    Definition_table    &dt,
    Module_deserializer &deserializer)
{
    Tag_t t;

    // assume here that Serializer::ST_SCOPE_START is already read

    DOUT(("Scope {\n"));
    INC_SCOPE();

    // read the owner definition
    if (deserializer.read_bool()) {
        Tag_t owner_tag = deserializer.read_encoded_tag();
        Definition *owner_def = deserializer.get_definition(owner_tag);

        owner_def->set_own_scope(this);

        DOUT(("owner def %u\n", unsigned(owner_tag)));
    }

    // read the scope type
    if (deserializer.read_bool()) {
        Tag_t type_tag = deserializer.read_encoded_tag();
        IType const *type = deserializer.get_type(type_tag);

        // associate it with the given type
        dt.associate_scope_type(deserializer, this, type);

        DOUT(("scope type %u\n", unsigned(type_tag)));
    }

    // read the scope name
    if (deserializer.read_bool()) {
        Tag_t name_tag = deserializer.read_encoded_tag();
        m_scope_name = deserializer.get_symbol(name_tag);

        DOUT(("scope name %u (%s)\n", unsigned(name_tag), m_scope_name->get_name()));
    }

    // deserialize children
    for (;;) {
        t = deserializer.read_section_tag();
        if (t == Serializer::ST_SCOPE_END)
            break;

        if (t == Serializer::ST_SCOPE_START) {
            // deserialize a sub scope
            Scope *s = dt.enter_scope(deserializer);
            s->deserialize(dt, deserializer);
            dt.leave_scope();
        } else {
            // deserialize a definition
            MDL_ASSERT(t == Serializer::ST_DEFINITION);

            dt.deserialize_def(deserializer);
        }
    }

    DEC_SCOPE();
    DOUT(("Scope }\n"));
}

// Some debug helper.
void Scope::dump(IAllocator *alloc, Printer *printer, size_t depth, bool is_owned) const
{
    typedef ptr_hash_set<Scope const>::Type Visited_set;

    Visited_set owned_scopes(0, Visited_set::hasher(), Visited_set::key_equal(), alloc);

    indent(printer, depth);
    printer->printf("Scope %u ", unsigned(m_id));

    if (m_owner_definition != NULL) {
        if (!is_owned) {
            // an orphan scope!
            print_def(printer, m_owner_definition);
            printer->print(" ORPHAN! ");
        }
    }
    if (m_scope_type != NULL) {
        if (!is_owned) {
            // an orphan type scope!
            printer->print(m_scope_type);
            printer->print(" ORPHAN! ");
        }
    } else if (m_scope_name != NULL) {
        // a named scope
        printer->print("\"");
        printer->print(m_scope_name);
        printer->print("\" ");
    }

    printer->print("{\n");
    ++depth;

    typedef list<Definition const *>::Type Def_list;
    Def_list all_defs(alloc);

    for (Definition const *def = m_definitions; def != NULL; def = def->m_next) {
        all_defs.push_back(def);
    }

    // print them in reverse order, just as they occurred in the source
    for (Def_list::const_reverse_iterator it = all_defs.rbegin(), end = all_defs.rend();
         it != end;
         ++it)
    {
        Definition const *def = *it;
        bool is_overloaded = def->m_same_next != NULL;

        if (is_overloaded) {
            indent(printer, depth);
            printer->print("Overload Set ");
            printer->print(def->get_sym());
            printer->print(" {\n");
            ++depth;
        }
        for (Definition const *last = def; last != NULL; last = last->m_same_next) {
            indent(printer, depth);

            print_def(printer, last);
            printer->print(";\n");
            if (Scope const *own_scope = last->get_own_scope()) {
                own_scope->dump(alloc, printer, depth, /*is_owned=*/true);
                owned_scopes.insert(own_scope);
            }
        }
        if (is_overloaded) {
            --depth;
            indent(printer, depth);
            printer->print("}  // OLS\n");
        }
    }

    for (Scope const *sub_scope = m_sub_scopes;
         sub_scope != NULL;
         sub_scope = sub_scope->m_next_subscope)
    {
        if (owned_scopes.find(sub_scope) == owned_scopes.end())
            sub_scope->dump(alloc, printer, depth, /*is_owned=*/false);
    }

    --depth;
    indent(printer, depth);
    printer->print("}");
    if (depth == 0)
        printer->print("  // Module\n");
    else {
        printer->print("\n");
    }
}

Definition_table::Definition_table(Module &owner)
: m_owner(owner)
, m_curr_scope(NULL)
, m_predef_scope(NULL)
, m_global_scope(NULL)
, m_free_scopes(NULL)
, m_next_definition_id(0)
, m_arena(owner.get_allocator())
, m_builder(m_arena)
, m_type_scopes(
    0, Type_scope_map::hasher(), Type_scope_map::key_equal(), owner.get_allocator())
, m_namespace_aliases(
    Namespace_aliases_map::key_compare(), owner.get_allocator())
, m_definitions(owner.get_allocator())
{
    std::fill_n(
        &m_operator_definitions[0], dimension_of(m_operator_definitions), (Definition *)0);

    // create initial scopes
    m_predef_scope = enter_scope(NULL);
    m_global_scope = enter_scope(NULL);
    leave_scope();
    leave_scope();
}

// Create a new scope.
Scope *Definition_table::create_scope(
    Scope            *parent,
    size_t           id,
    Definition const *owner_def,
    IType const      *type,
    ISymbol const    *name)
{
    if (Scope *s = m_free_scopes) {
        m_free_scopes = s->m_next_subscope;

        new (s) Scope(parent, id, owner_def, type, name);
        return s;
    }
    return m_builder.create<Scope>(parent, id, owner_def, type, name);
}

// Creates a new named scope.
Scope *Definition_table::create_scope(
    Scope         *parent,
    ISymbol const *name,
    size_t        id)
{
    if (Scope *s = m_free_scopes) {
        m_free_scopes = s->m_next_subscope;

        new (s) Scope(parent, name, id);
        return s;
    }
    return m_builder.create<Scope>(parent, name, id);
}

// Serialize a definition.
void Definition_table::serialize_def(
    Definition const  *def,
    Module_serializer &serializer) const
{
    serializer.write_section_tag(Serializer::ST_DEFINITION);

    // register this definition
    Tag_t def_tag = serializer.register_definition(def);
    serializer.write_encoded_tag(def_tag);

    DOUT(("Def tag %u {\n", unsigned(def_tag)));
    INC_SCOPE();

    // write the kind
    serializer.write_unsigned(def->m_kind);
    DOUT(("def kind %u\n", unsigned(def->m_kind)));

    // write the unique ID
    serializer.write_encoded_tag(def->m_unique_id);
    DOUT(("id %u\n", unsigned(def->m_unique_id)));

    // no need to write the owner module_id, is recalculated

    // write the original unique id
    serializer.write_encoded_tag(def->m_original_unique_id);
    DOUT(("orig id %u\n", unsigned(def->m_original_unique_id)));

    // write the m_original_module_import_id
    serializer.write_encoded_tag(def->m_original_module_import_idx);
    DOUT(("original mod id %u\n", unsigned(def->m_original_module_import_idx)));

    // write the symbol of this definition
    Tag_t sym_tag = serializer.get_symbol_tag(def->m_sym);
    serializer.write_encoded_tag(sym_tag);
    DOUT(("sym %u (%s)\n", unsigned(sym_tag), def->m_sym->get_name()));

    if (def->m_kind != Definition::DK_NAMESPACE) {
        // write the type of this definition
        Tag_t type_tag = serializer.get_type_tag(def->m_type);
        serializer.write_encoded_tag(type_tag);
        DOUT(("type %u\n", unsigned(type_tag)));
    }

    if (def->m_parameter_inits != NULL) {
        // write references to initializer expressions
        size_t count = def->m_parameter_inits->count;
        serializer.write_encoded_tag(count);
        DOUT(("#inits %u\n", unsigned(count)));
        INC_SCOPE();

        for (size_t i = 0; i < count; ++i) {
            IExpression const *init = def->m_parameter_inits->exprs[i];
            if (init != NULL) {
                Tag_t t = serializer.register_expression(init);
                serializer.write_encoded_tag(t);
                DOUT(("init expr %u\n", unsigned(t)));
            } else {
                serializer.write_encoded_tag(0);
                DOUT(("init expr 0\n"));
            }
        }
        DEC_SCOPE();
    } else {
        serializer.write_encoded_tag(0);
        DOUT(("#inits 0\n"));
    }

    // no need to serialize m_def_scope
    // m_own_scope is updated automatically

    // register the position of this definition if any
    if (def->m_pos != NULL) {
        serializer.write_bool(true);
    } else {
        serializer.write_bool(false);
    }

    // register the declaration of this definition if any
    if (def->m_decl != NULL) {
        serializer.write_bool(true);
        Tag_t decl_tag = serializer.register_declaration(def->m_decl);
        serializer.write_encoded_tag(decl_tag);
        DOUT(("decl %u\n", unsigned(decl_tag)));
    } else {
        serializer.write_bool(false);
    }

    // register the prototype declaration of this definition if any
    if (def->m_proto_decl != NULL) {
        serializer.write_bool(true);
        Tag_t decl_tag = serializer.register_declaration(def->m_proto_decl);
        serializer.write_encoded_tag(decl_tag);
        DOUT(("proto decl %u\n", unsigned(decl_tag)));
    } else {
        serializer.write_bool(false);
    }

    // these are set automatically:
    // m_next
    // m_same_prev
    // m_same_next
    // m_outer_def

    // register the definite definition if any
    if (def->m_definite_def != NULL) {
        serializer.write_bool(true);
        Tag_t def_tag = serializer.register_definition(def->m_definite_def);
        serializer.write_encoded_tag(def_tag);
        DOUT(("definite def %u\n", unsigned(def_tag)));
    } else {
        serializer.write_bool(false);
    }

    // serialize the value coupled with the definition if any
    if (def->m_value != NULL) {
        serializer.write_bool(true);
        Tag_t value_tag = serializer.get_value_tag(def->m_value);
        serializer.write_encoded_tag(value_tag);

        DOUT(("value %u\n", unsigned(value_tag)));
    } else {
        serializer.write_bool(false);
    }

    // serialize the semantic code
    serializer.write_unsigned(def->m_u.code);
    DOUT(("sema %u\n", def->m_u.code));

    // serialize flags
    unsigned char const *raw_data = def->m_flags.raw_data();
    for (size_t i = 0, n = (def->m_flags.get_size() + 7) / 8; i < n; ++i) {
        serializer.write_byte(raw_data[i]);
    }
    DOUT(("Flags:%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
        def->has_flag(Definition::DEF_IS_PREDEFINED)        ? " predef"   : "",
        def->has_flag(Definition::DEF_IS_DECL_ONLY)         ? " dclonly"  : "",
        def->has_flag(Definition::DEF_IS_INCOMPLETE)        ? " incmpl"   : "",
        def->has_flag(Definition::DEF_IS_EXPLICIT)          ? " explct"   : "",
        def->has_flag(Definition::DEF_IS_EXPLICIT_WARN)     ? " explct_w" : "",
        def->has_flag(Definition::DEF_IS_COMPILER_GEN)      ? " cc_gen"   : "",
        def->has_flag(Definition::DEF_IS_EXPORTED)          ? " exp"      : "",
        def->has_flag(Definition::DEF_IS_IMPORTED)          ? " imp"      : "",
        def->has_flag(Definition::DEF_OP_LVALUE)            ? " lval"     : "",
        def->has_flag(Definition::DEF_IGNORE_OVERLOAD)      ? " IGN"      : "",
        def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR) ? " cnst_con" : "",
        def->has_flag(Definition::DEF_IS_REACHABLE)         ? " reach"    : "",
        def->has_flag(Definition::DEF_IS_STDLIB)            ? " stdlib"   : "",
        def->has_flag(Definition::DEF_IS_USED)              ? " used"     : "",
        def->has_flag(Definition::DEF_IS_UNUSED)            ? " unused"   : "",
        def->has_flag(Definition::DEF_USED_INCOMPLETE)      ? " use_incpl": "",
        def->has_flag(Definition::DEF_IS_UNIFORM)           ? " uniform"  : "",
        def->has_flag(Definition::DEF_IS_VARYING)           ? " varying"  : "",
        def->has_flag(Definition::DEF_IS_WRITTEN)           ? " written"  : "",
        def->has_flag(Definition::DEF_IS_LET_TEMPORARY)     ? " let_tmp"  : "",
        def->has_flag(Definition::DEF_NO_INLINE)            ? " no_inl"   : ""
    ));

    // serialize version flags
    serializer.write_unsigned(def->m_version_flags);

    // serialize parameter derivative mask for functions
    if (def->m_kind == Definition::DK_FUNCTION) {
        serializer.write_unsigned(def->m_parameter_deriv_mask);
    }

    // serialize namespace
    if (def->m_kind== Definition::DK_NAMESPACE) {
        ISymbol const *name_space = def->get_namespace();
        Tag_t         sym_tag     = serializer.get_symbol_tag(name_space);
        serializer.write_encoded_tag(sym_tag);
        DOUT(("namespace %u (%s)\n", unsigned(sym_tag), name_space->get_name()));
    }

    DEC_SCOPE();
    DOUT(("Def }\n"));
}

// Deserialize a definition.
Definition *Definition_table::deserialize_def(Module_deserializer &deserializer)
{
    // assume the section tag is already read here

    // register this definition
    Tag_t def_tag = deserializer.read_encoded_tag();

    DOUT(("Def tag %u {\n", unsigned(def_tag)));
    INC_SCOPE();

    // read the kind
    IDefinition::Kind kind = IDefinition::Kind(deserializer.read_unsigned());
    DOUT(("def kind %u\n", unsigned(kind)));

    // read the unique ID
    size_t unique_id = deserializer.read_encoded_tag();
    DOUT(("id %u\n", unsigned(unique_id)));

    // no need to read the owner module id, this is recalculated

    // read the original unique id
    size_t original_unique_id = deserializer.read_encoded_tag();
    DOUT(("orig id %u\n", unsigned(original_unique_id)));

    // read the original module id
    size_t original_module_import_id = deserializer.read_encoded_tag();
    DOUT(("original mod id %u\n", unsigned(original_module_import_id)));

    // read the symbol of this definition
    Tag_t sym_tag = deserializer.read_encoded_tag();
    ISymbol const *sym = deserializer.get_symbol(sym_tag);
    DOUT(("sym %u (%s)\n", unsigned(sym_tag), sym->get_name()));

    IType const *type = NULL;
    if (kind != Definition::DK_NAMESPACE) {
        // read the type of this definition
        Tag_t type_tag = deserializer.read_encoded_tag();
        type = deserializer.get_type(type_tag);
        DOUT(("type %u\n", unsigned(type_tag)));
    }

    // no need to deserialize m_def_scope
    // m_own_scope is updated automatically

    Position *pos = NULL;

    Definition *new_def;
    if (kind == Definition::DK_NAMESPACE) {
        new_def = m_builder.create<Definition>(
            kind,
            m_owner.get_unique_id(),
            sym,
            type,
            pos,
            /*parant_scope=*/(Scope *)NULL,
            /*outer_def=*/(Definition *)NULL,
            unique_id);
    } else {
        Definition *curr_def = get_definition(sym);
        if (curr_def && curr_def->get_def_scope() == m_curr_scope) {
            // there is already a definition for this symbol, append it
            new_def = m_builder.create<Definition>(
                kind, m_owner.get_unique_id(), sym, type, pos,
                m_curr_scope, (Definition *)NULL, unique_id);
            new_def->link_same_def(curr_def);
        } else {
            // no definition inside this scope
            new_def = m_builder.create<Definition>(
                kind, m_owner.get_unique_id(), sym, type, pos,
                m_curr_scope, curr_def, unique_id);
            m_curr_scope->add_definition(new_def);
        }
        set_definition(sym, new_def);
    }

    new_def->m_original_unique_id = original_unique_id;
    new_def->m_original_module_import_idx = original_module_import_id;

    // read m_parameter_inits
    size_t count = deserializer.read_encoded_tag();
    DOUT(("#inits %u\n", unsigned(count)));

    if (count > 0) {
        INC_SCOPE();

        m_owner.allocate_initializers(new_def, count);
        for (size_t i = 0; i < count; ++i) {
            Tag_t t = deserializer.read_encoded_tag();

            if (t != Tag_t(0)) {
                deserializer.wait_for_expression(t, &new_def->m_parameter_inits->exprs[i]);
            } else {
                new_def->m_parameter_inits->exprs[i] = NULL;
            }
            DOUT(("init expr %u\n", unsigned(t)));
        }
        DEC_SCOPE();
    }

    // now register the definition
    deserializer.register_definition(def_tag, new_def);

    // register the position of this definition if any
    if (deserializer.read_bool()) {
        // there is one
        // FIXME:
    }

    // register the declaration of this definition if any
    if (deserializer.read_bool()) {
        Tag_t decl_tag = deserializer.read_encoded_tag();

        deserializer.wait_for_declaration(decl_tag, &new_def->m_decl);
        DOUT(("decl %u\n", unsigned(decl_tag)));
    }

    // register the prototype declaration of this definition if any
    if (deserializer.read_bool()) {
        Tag_t decl_tag = deserializer.read_encoded_tag();

        deserializer.wait_for_declaration(decl_tag, &new_def->m_proto_decl);
        DOUT(("proto decl %u\n", unsigned(decl_tag)));
    }

    // these are set automatically:
    // m_next
    // m_same_prev
    // m_same_next
    // m_outer_def

    // register the definite definition if any
    if (deserializer.read_bool()) {
        Tag_t def_tag = deserializer.read_encoded_tag();

        deserializer.wait_for_definition(def_tag, &new_def->m_definite_def);
        DOUT(("definite def %u\n", unsigned(def_tag)));
    }

    // deserialize the value coupled with the definition if any
    if (deserializer.read_bool()) {
        Tag_t value_tag = deserializer.read_encoded_tag();
        new_def->m_value  = deserializer.get_value(value_tag);

        DOUT(("value %u\n", unsigned(value_tag)));
    }

    // deserialize the semantic code
    new_def->m_u.code = deserializer.read_unsigned();
    DOUT(("sema %u\n", new_def->m_u.code));

    // deserialize flags
    unsigned char *raw_data = new_def->m_flags.raw_data();
    for (size_t i = 0, n = (new_def->m_flags.get_size() + 7) / 8; i < n; ++i) {
        raw_data[i] = deserializer.read_byte();
    }
    DOUT(("Flags:%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
        new_def->has_flag(Definition::DEF_IS_PREDEFINED)        ? " predef"   : "",
        new_def->has_flag(Definition::DEF_IS_DECL_ONLY)         ? " dclonly"  : "",
        new_def->has_flag(Definition::DEF_IS_INCOMPLETE)        ? " incmpl"   : "",
        new_def->has_flag(Definition::DEF_IS_EXPLICIT)          ? " explct"   : "",
        new_def->has_flag(Definition::DEF_IS_EXPLICIT_WARN)     ? " explct_w" : "",
        new_def->has_flag(Definition::DEF_IS_COMPILER_GEN)      ? " cc_gen"   : "",
        new_def->has_flag(Definition::DEF_IS_EXPORTED)          ? " exp"      : "",
        new_def->has_flag(Definition::DEF_IS_IMPORTED)          ? " imp"      : "",
        new_def->has_flag(Definition::DEF_OP_LVALUE)            ? " lval"     : "",
        new_def->has_flag(Definition::DEF_IGNORE_OVERLOAD)      ? " IGN"      : "",
        new_def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR) ? " cnst_con" : "",
        new_def->has_flag(Definition::DEF_IS_REACHABLE)         ? " reach"    : "",
        new_def->has_flag(Definition::DEF_IS_STDLIB)            ? " stdlib"   : "",
        new_def->has_flag(Definition::DEF_IS_USED)              ? " used"     : "",
        new_def->has_flag(Definition::DEF_IS_UNUSED)            ? " unused"   : "",
        new_def->has_flag(Definition::DEF_USED_INCOMPLETE)      ? " use_incpl": "",
        new_def->has_flag(Definition::DEF_IS_UNIFORM)           ? " uniform"  : "",
        new_def->has_flag(Definition::DEF_IS_VARYING)           ? " varying"  : "",
        new_def->has_flag(Definition::DEF_IS_WRITTEN)           ? " written"  : "",
        new_def->has_flag(Definition::DEF_IS_LET_TEMPORARY)     ? " let_tmp"  : "",
        new_def->has_flag(Definition::DEF_NO_INLINE)            ? " no_inl"   : ""
    ));

    // deserialize version flags
    new_def->m_version_flags = deserializer.read_unsigned();

    // deserialize parameter derivative mask for functions
    if (new_def->m_kind == Definition::DK_FUNCTION) {
        new_def->m_parameter_deriv_mask = deserializer.read_unsigned();
    }

    // deserialize namespace
    if (new_def->m_kind == Definition::DK_NAMESPACE) {
        size_t sym_tag = deserializer.read_encoded_tag();
        ISymbol const *name_space = deserializer.get_symbol(sym_tag);
        DOUT(("namespace %u (%s)\n", unsigned(sym_tag), name_space->get_name()));
        new_def->set_namespace(name_space);
    }

    DEC_SCOPE();
    DOUT(("Def }\n"));
    return new_def;
}

// Register all definition in the predefined scope.
void Definition_table::register_predefined_entities(
    Module_serializer &serializer,
    Scope const       *scope) const
{
    for (Definition const *def_set = scope->m_definitions;
         def_set != NULL;
         def_set = def_set->get_next_def_in_scope())
    {
        for (Definition const *def = def_set; def != NULL; def = def->get_next_def()) {
            serializer.register_definition(def);
            if (Scope *s = def->get_own_scope()) {
                register_predefined_entities(serializer, s);
            }
        }
    }
}

// Register all definition in the predefined scope
Tag_t Definition_table::register_predefined_entities(
    Module_deserializer &deserializer,
    Scope const         *scope,
    Tag_t               tag)
{
    for (Definition *def_set = scope->m_definitions;
        def_set != NULL;
        def_set = def_set->get_next_def_in_scope())
    {
        for (Definition *def = def_set; def != NULL; def = def->get_next_def()) {
            deserializer.register_definition(tag, def);
            ++tag;
            if (Scope *s = def->get_own_scope()) {
                tag = register_predefined_entities(deserializer, s, tag);
            }
        }
    }
    return tag;
}

// Associate a scope and a type in the deserialization.
void Definition_table::associate_scope_type(
    Module_deserializer &,
    Scope               *scope,
    IType const         *type)
{
    scope->m_scope_type = type;
    m_type_scopes[type] = scope;
}

// Enter a new scope empty scope for deserialization.
Scope *Definition_table::enter_scope(Module_deserializer &)
{
    // create a new scope
    Scope *scope = create_scope(
        m_curr_scope, 0, (Definition *)NULL, (IType *)NULL, (ISymbol *)NULL);
    m_curr_scope = scope;
    return scope;
}

// Enter a new scope.
Scope *Definition_table::enter_scope(Definition const *def)
{
    // create a new scope
    Scope *scope = create_scope(m_curr_scope, ++m_next_definition_id, def);
    m_curr_scope = scope;
    return scope;
}

// Enter a named scope (module or package import).
Scope *Definition_table::enter_named_scope(ISymbol const *name)
{
    MDL_ASSERT(m_curr_scope && m_curr_scope->find_named_subscope(name) == NULL &&
           "named scope already exists");

    // create a new scope
    Scope *scope = create_scope(m_curr_scope, name, ++m_next_definition_id);
    m_curr_scope = scope;
    return scope;
}

// Enter a new scope created by a type declaration.
Scope *Definition_table::enter_scope(
    IType const      *type,
    Definition const *type_def)
{
    // create a new scope
    Scope *scope = create_scope(
        m_curr_scope, ++m_next_definition_id, type_def, type, type_def->get_symbol());

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
    MDL_ASSERT(scope->get_parent() == m_curr_scope);

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

    IAllocator *alloc = m_owner.get_allocator();

    // otherwise a real transition: find common prefix
    vector<Scope *>::Type curr_stack(alloc);
    for (Scope *curr = m_curr_scope; curr != NULL; curr = curr->get_parent()) {
        curr_stack.push_back(curr);
    }

    vector<Scope *>::Type new_stack(alloc);
    for (Scope *curr = scope; curr != NULL; curr = curr->get_parent()) {
        new_stack.push_back(curr);
    }

    int curr_idx = int(curr_stack.size()) - 1;
    int new_idx  = int(new_stack.size()) - 1;

    if (curr_idx >= 0 && new_idx >= 0) {
        while (curr_stack[curr_idx] == new_stack[new_idx]) {
            --curr_idx;
            --new_idx;
            if (curr_idx < 0 || new_idx < 0) {
                break;
            }
        }
    }

    // remove until prefix is reached
    for (int i = 0; i <= curr_idx; ++i)
        leave_scope();

    // reopen until top is reached
    for (int i = new_idx; i >= 0; --i)
        reopen_scope(new_stack[i]);
}

// Enter a new (entity) definition.
Definition *Definition_table::enter_definition(
    Definition::Kind kind,
    ISymbol const    *symbol,
    IType const      *type,
    Position const   *pos)
{
    Definition *new_def;
    MDL_ASSERT(symbol != NULL);
    Definition *curr_def = get_definition(symbol);
    if (curr_def && curr_def->get_def_scope() == m_curr_scope) {
        // there is already a definition for this symbol, append it
        new_def = m_builder.create<Definition>(
            kind, m_owner.get_unique_id(), symbol, type, pos,
            m_curr_scope, (Definition *)0, ++m_next_definition_id);
        new_def->link_same_def(curr_def);
    } else {
        // no definition inside this scope
        new_def = m_builder.create<Definition>(
            kind, m_owner.get_unique_id(), symbol, type, pos,
            m_curr_scope, curr_def, ++m_next_definition_id);
        m_curr_scope->add_definition(new_def);
    }
    set_definition(symbol, new_def);
    return new_def;
}

// Enter a new operator definition.
Definition *Definition_table::enter_operator_definition(
    IExpression::Operator kind,
    ISymbol const         *symbol,
    IType const           *type)
{
    Definition *new_def;
    Definition *curr_def = get_operator_definition(kind);
    if (curr_def && curr_def->get_def_scope() == m_curr_scope) {
        // there is already a definition for this symbol, append it */
        new_def = m_builder.create<Definition>(
            Definition::DK_OPERATOR, m_owner.get_unique_id(), symbol, type, (Position *)0,
            m_curr_scope, (Definition *)0, ++m_next_definition_id);
        new_def->link_same_def(curr_def);
    } else {
        // no definition inside this scope
        new_def = m_builder.create<Definition>(
            Definition::DK_OPERATOR, m_owner.get_unique_id(), symbol, type, (Position *)0,
            m_curr_scope, curr_def, ++m_next_definition_id);
        m_curr_scope->add_definition(new_def);
    }
    // operators are currently always compiler generated and predefined
    new_def->set_flag(Definition::DEF_IS_COMPILER_GEN);
    new_def->set_flag(Definition::DEF_IS_PREDEFINED);
    new_def->set_semantic(operator_to_semantic(kind));

    set_operator_definition(kind, new_def);
    return new_def;
}

// Enter an error definition for the given symbol.
Definition *Definition_table::enter_error(ISymbol const *symbol, const IType_error *err_type)
{
    MDL_ASSERT(get_definition(symbol) == NULL);

    // no definition inside this scope
    Definition *err_def = m_builder.create<Definition>(
        Definition::DK_ERROR, m_owner.get_unique_id(), symbol, err_type, (Position *)0,
        m_curr_scope, (Definition *)0, ++m_next_definition_id);

    m_curr_scope->add_definition(err_def);
    set_definition(symbol, err_def);
    return err_def;
}

// find a type scope
Scope *Definition_table::get_type_scope(IType const *type) const
{
    Type_scope_map::const_iterator it = m_type_scopes.find(type);
    if (it != m_type_scopes.end())
        return it->second;
    return NULL;
}

// Return the current definition for the given symbol.
Definition *Definition_table::get_definition(ISymbol const *symbol) const
{
    size_t id   = symbol->get_id();
    size_t size = m_definitions.size();
    if (id < size) {
        return m_definitions[id];
    }
    return NULL;
}

// Set the current definition for a symbol in this definition table.
void Definition_table::set_definition(ISymbol const *sym, Definition *def)
{
    size_t id   = sym->get_id();
    size_t size = m_definitions.size();

    if (id >= size) {
        size = ((id + 1) + 0x0F) & ~0x0F;
        m_definitions.resize(size, NULL);
    }
    if (def) {
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
void Definition_table::restore_definition(ISymbol const *sym, Definition *def)
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
Definition *Definition_table::get_operator_definition(IExpression::Operator op) const
{
    return m_operator_definitions[op];
}

// Set the definition for an operator in this definition table.
void Definition_table::set_operator_definition(IExpression::Operator op, Definition *def)
{
    m_operator_definitions[op] = def;
}

// Enter an imported definition.
Definition *Definition_table::import_definition(
    Definition const *imported,
    size_t           owner_import_idx)
{
    Definition     *new_def;
    ISymbol const  *imp_sym  = m_owner.import_symbol(imported->get_sym());
    IType const    *imp_type = m_owner.import_type(imported->get_type());
    Position const *imp_pos  = m_owner.import_position(imported->get_position());

    Definition *curr_def = get_definition(imp_sym);
    if (curr_def && curr_def->get_def_scope() == m_curr_scope) {
        // there is already a definition for this symbol, append it */
        new_def = m_builder.create<Definition>(
            *imported, imp_sym, imp_type, imp_pos,
            m_curr_scope, (Definition *)0,
            m_owner.get_unique_id(), ++m_next_definition_id,
            owner_import_idx);
        new_def->link_same_def(curr_def);
    } else {
        // no definition inside this scope
        new_def = m_builder.create<Definition>(
            *imported, imp_sym, imp_type, imp_pos,
            m_curr_scope, curr_def,
            m_owner.get_unique_id(), ++m_next_definition_id,
            owner_import_idx);
        m_curr_scope->add_definition(new_def);
    }

    if (imported->m_value)
        new_def->m_value = m_owner.import_value(imported->m_value);
    new_def->m_u.code = imported->m_u.code;
    set_definition(imp_sym, new_def);

    // Note: neither the declaration NOR the prototype declaration nor the initializers
    // are imported.
    // They all point to the AST (or definitions) of the original module.
    // Use Module::get_original_definition() to retrieve this definition and use it.

    return new_def;
}

// Get a namespace alias.
Definition const *Definition_table::get_namespace_alias(
    ISymbol const *alias)
{
    Namespace_aliases_map::const_iterator it = m_namespace_aliases.find(alias);
    if (it != m_namespace_aliases.end())
        return it->second;
    return NULL;
}

// Enter a new namespace alias.
Definition *Definition_table::enter_namespace_alias(
    ISymbol const                      *alias,
    ISymbol const                      *ns,
    IDeclaration_namespace_alias const *decl)
{
    Definition *new_def = m_builder.create<Definition>(
        Definition::DK_NAMESPACE,
        m_owner.get_unique_id(),
        alias,
        /*type=*/(IType *)NULL,
        &decl->get_alias()->access_position(),
        /*parant_scope=*/(Scope *)NULL,
        /*outer_def=*/(Definition *)NULL,
        ++m_next_definition_id);

    new_def->set_namespace(ns);
    new_def->set_declaration(decl);

    bool res = m_namespace_aliases.insert(Namespace_aliases_map::value_type(alias, new_def)).second;
    MDL_ASSERT(res && "name clash for namespace alias");
    (void)res;
    return new_def;
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
    m_free_scopes        = NULL;
    m_next_definition_id = 0;
    m_arena.drop(NULL);
    m_type_scopes.clear();
    m_definitions.clear();
    m_namespace_aliases.clear();

    std::fill_n(
        &m_operator_definitions[0], dimension_of(m_operator_definitions), (Definition *)0);

    // create initial scopes
    m_predef_scope = enter_scope(NULL);
    m_global_scope = enter_scope(NULL);
    leave_scope();
    leave_scope();
}

// Returns the amount of used memory by this definition table.
size_t Definition_table::get_memory_size() const
{
    size_t res = sizeof(*this);

    res += m_arena.get_chunks_size();
    res += dynamic_memory_consumption(m_type_scopes);
    res += dynamic_memory_consumption(m_definitions);

    return res;
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

// Serialize this definition table.
void Definition_table::serialize(Module_serializer &serializer) const
{
    // register all definition in the predefined scope
    register_predefined_entities(serializer, m_predef_scope);

    serializer.write_section_tag(Serializer::ST_DEF_TABLE);

    Tag_t def_tbl_tag = serializer.register_definition_table(this);
    serializer.write_encoded_tag(def_tbl_tag);

    DOUT(("Def table %u {\n", unsigned(def_tbl_tag)));
    INC_SCOPE();

    // serialize namespace aliases
    serializer.write_encoded_tag(m_namespace_aliases.size());
    for (Namespace_aliases_map::const_iterator
            it(m_namespace_aliases.begin()), end(m_namespace_aliases.end());
        it != end;
        ++it)
    {
        ISymbol const    *alias = it->first;
        Definition const *def = it->second;

        Tag_t alias_tag = serializer.get_symbol_tag(alias);
        serializer.write_encoded_tag(alias_tag);
        DOUT(("alias %u (%s)\n", unsigned(alias_tag), alias->get_name()));

        serialize_def(def, serializer);
    }

    // serialize the global scope
    Scope const *s = m_global_scope;

    s->serialize(*this, serializer);

    DEC_SCOPE();
    DOUT(("Def table }\n"));
}

// Deserialize this definition table.
void Definition_table::deserialize(Module_deserializer &deserializer)
{
    // register all definition in the predefined scope
    register_predefined_entities(deserializer, m_predef_scope, Tag_t(1));

    Tag_t t;

    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_DEF_TABLE);

    Tag_t def_tbl_tag = deserializer.read_encoded_tag();
    deserializer.register_definition_table(def_tbl_tag, this);

    DOUT(("Def table %u {\n", unsigned(def_tbl_tag)));
    INC_SCOPE();

    // deserialize namespace aliases
    size_t n_namespaces = deserializer.read_encoded_tag();
    for (size_t i = 0; i < n_namespaces; ++i) {
        Tag_t alias_tag = deserializer.read_encoded_tag();
        ISymbol const *alias = deserializer.get_symbol(alias_tag);
        DOUT(("alias %u (%s)\n", unsigned(alias_tag), alias->get_name()));

        t = deserializer.read_section_tag();
        MDL_ASSERT(t == Serializer::ST_DEFINITION);

        Definition *def = deserialize_def(deserializer);
        m_namespace_aliases.insert(Namespace_aliases_map::value_type(alias, def));
    }

    // must start with a scope
    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_SCOPE_START);

    // deserialize the global scope
    Scope *s = m_global_scope;
    transition_to_scope(s);

    s->deserialize(*this, deserializer);
    DEC_SCOPE();
    DOUT(("Def table }\n"));
}

// Iterate over ALL visible definitions.
Definition const *Definition_table::get_visible_definition(size_t &index) const
{
    if (index < m_definitions.size()) {
        do {
            if (Definition const *def = m_definitions[index++]) {
                return def;
            }
        } while (index < m_definitions.size());
    }
    return NULL;
}

// Debug helper: Prints the definition table to the given printer.
void Definition_table::dump(Printer *printer, char const *name) const
{
    Scope const *s = m_global_scope;

    printer->print("Module '");
    printer->print(name);
    printer->print("':\n");
    s->dump(m_arena.get_allocator(), printer, 0, /*is_owned=*/false);
    printer->print("\n");
}

// Decode the since version.
unsigned mdl_since_version(unsigned flags)
{
    // lower 8bit are the since version
    return flags & 0xFF;
}

// Decode the removed version.
unsigned mdl_removed_version(unsigned flags)
{
    // upper 8bit are the until version
    unsigned rem = (flags >> 8) & 0xFF;
    if (rem == 0) {
        // never removed
        return 0xFFFFFFFF;
    }
    return rem;
}

// Check if a entity is available in the given MDL language level.
bool is_available_in_mdl(
    unsigned module_version,
    unsigned version_flags)
{
    unsigned since_ver = mdl_since_version(version_flags);
    bool res = since_ver <= module_version;

    unsigned removed_ver = mdl_removed_version(version_flags);
    if (removed_ver != 0 && module_version >= removed_ver) {
        // was removed
        res = false;
    }
    return res;
}

}  // mdl
}  // mi
