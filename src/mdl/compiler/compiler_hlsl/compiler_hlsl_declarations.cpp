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

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_declarations.h"
#include "compiler_hlsl_definitions.h"

namespace mi {
namespace mdl {
namespace hlsl {

// ------------------------------- Name -------------------------------


// Constructor.
Name::Name(
    Location const &loc,
    Symbol         *sym)
: Base()
, m_loc(loc)
, m_sym(sym)
, m_def(NULL)
{
}

// Get the type of this name.
Type *Name::get_type()
{
    return m_def != NULL ? m_def->get_type() : NULL;
}

// Set the definition for this name.
void Name::set_definition(Definition *def)
{
    HLSL_ASSERT(m_def == NULL);
    m_def = def;
}

// ------------------------------- Type_qualifier -------------------------------

// Default constructor, creates empty type qualifier.
Type_qualifier::Type_qualifier()
: Base()
, m_storage_qualifier(SQ_NONE)
, m_type_modifier(TM_NONE)
, m_parameter_qualifier(PQ_NONE)
, m_interpolation_qualifier(IQ_NONE)
, m_is_invariant(false)
, m_is_precise(false)
{
}

// Add a storage qualifier.
void Type_qualifier::set_storage_qualifier(Storage_qualifier qual)
{
    HLSL_ASSERT((m_storage_qualifier & qual) == 0);
    m_storage_qualifier |= qual;
}

// Add a type modifier.
void Type_qualifier::set_type_modifier(Type_modifier mod)
{
    HLSL_ASSERT((m_type_modifier & mod) == 0);
    m_type_modifier |= mod;
}

// Set the precision qualifier.
void Type_qualifier::set_parameter_qualifier(Parameter_qualifier qual)
{
    HLSL_ASSERT(m_parameter_qualifier == PQ_NONE);
    m_parameter_qualifier = qual;
}

// Set the interpolation qualifier.
void Type_qualifier::set_interpolation_qualifier(Interpolation_qualifier qual)
{
    HLSL_ASSERT(m_interpolation_qualifier == IQ_NONE);
    m_interpolation_qualifier = qual;
}

// Set the INVARIANT qualifier.
void Type_qualifier::set_invariant()
{
    HLSL_ASSERT(!m_is_invariant);
    m_is_invariant = true;
}

// Set the PRECISE qualifier.
void Type_qualifier::set_precise()
{
    HLSL_ASSERT(!m_is_precise);
    m_is_precise = true;
}

// ------------------------------- Array_specifier-------------------------------

// Constructor.
Array_specifier::Array_specifier(
    Location const &loc,
    Expr           *size)
: Base()
, m_loc(loc)
, m_size(size)
{
}

// ------------------------------- Type_name-------------------------------

// Constructor.
Type_name::Type_name(
    Location const &loc)
: Base()
, m_type_qualifier()
, m_name(NULL)
, m_struct_decl(NULL)
, m_type(NULL)
, m_loc(loc)
{
}

// Set the type represented by this name.
void Type_name::set_type(Type *type)
{
    HLSL_ASSERT(m_type == NULL);
    m_type = type;
}

// Set the name of this type name.
void Type_name::set_name(Name *name)
{
    HLSL_ASSERT(m_name == NULL);
    m_name = name;
}

// Set the struct declaration of this name.
void Type_name::set_struct_decl(Declaration *decl)
{
    m_struct_decl = decl;
}

// ------------------------------- Base_decl-------------------------------

// Constructor.
Declaration::Declaration(Location const &loc)
: Base()
, m_loc(loc)
, m_def(NULL)
{
}

// ------------------------------- Invalid_decl-------------------------------

// Constructor.
Declaration_invalid::Declaration_invalid(Location const &loc)
: Base(loc)
{
}

// Get the declaration kind.
Declaration::Kind Declaration_invalid::get_kind() const
{
    return s_kind;
}

// ------------------------------- Init_declarator-------------------------------

// Constructor.
Init_declarator::Init_declarator(
    Location const &loc)
: Base()
, m_loc(loc)
, m_name(NULL)
, m_array_specifiers()
, m_init(NULL)
{
}

// Set the identifier of this declarator.
void Init_declarator::set_name(Name *name)
{
    HLSL_ASSERT(name != NULL);
    m_name = name;
}

// Set the initializer of this declarator.
void Init_declarator::set_initializer(Expr *init)
{
    m_init = init;
}

// ------------------------------- Variable_declaration-------------------------------

// Constructor.
Declaration_variable::Declaration_variable(
    Type_name *name)
: Base(name->get_location())
, m_type_name(name)
, m_var_list()
{
}

// Get the declaration kind.
Declaration::Kind Declaration_variable::get_kind() const
{
    return s_kind;
}

// Add a variable declaration.
void Declaration_variable::add_init(Init_declarator *decl)
{
    m_var_list.push(decl);
}

// ------------------------------- Declaration_param-------------------------------

// Constructor.
Declaration_param::Declaration_param(
    Type_name *name)
: Base(name->get_location())
, m_type_name(name)
, m_name(NULL)
, m_default_argument(NULL)
, m_array_specifiers()
{
}

// Get the declaration kind.
Declaration::Kind Declaration_param::get_kind() const
{
    return s_kind;
}

// Set the name of the parameter.
void Declaration_param::set_name(Name *name)
{
    HLSL_ASSERT(m_name == NULL);
    m_name = name;
}

// Set the default argument of this parameter.
void Declaration_param::set_default_argument(Expr *expr)
{
    m_default_argument = expr;
}

// Add an array specifier.
void Declaration_param::add_array_specifier(Array_specifier *arr_spec)
{
    m_array_specifiers.push(arr_spec);
}

// ------------------------------- Declaration_function-------------------------------

// Constructor.
Declaration_function::Declaration_function(
    Type_name *type_name,
    Name      *name)
: Base(type_name->get_location())
, m_ret_type(type_name)
, m_name(name)
, m_params()
, m_body(NULL)
{
}

// Get the declaration kind.
Declaration::Kind Declaration_function::get_kind() const
{
    return s_kind;
}

// Add a function parameter.
void Declaration_function::add_param(Declaration *param)
{
    m_params.push(param);
}

// Set the function body.
void Declaration_function::set_body(Stmt *body)
{
    // can be overwritten by the optimizer
    m_body = body;
}

// ------------------------------- Field_declarator-------------------------------

// Constructor.
Field_declarator::Field_declarator(
    Location const &loc)
: Base()
, m_loc(loc)
, m_name(NULL)
, m_array_specifiers()
{
}

// Set the name of the field.
void Field_declarator::set_name(Name *name)
{
    HLSL_ASSERT(m_name == NULL);
    m_name = name;
}

// Add an array specifier.
void Field_declarator::add_array_specifier(Array_specifier *arr_spec)
{
    m_array_specifiers.push(arr_spec);
}

// ------------------------------- Declaration_field-------------------------------

// Constructor.
Declaration_field::Declaration_field(
    Type_name *type_name)
: Base(type_name->get_location())
, m_type_name(type_name)
, m_fields()
{
}

// Get the declaration kind.
Declaration::Kind Declaration_field::get_kind() const
{
    return s_kind;
}

/// Add a field declarator.
void Declaration_field::add_field(Field_declarator *declarator)
{
    m_fields.push(declarator);
}

// ------------------------------- Declaration_struct-------------------------------

// Constructor.
Declaration_struct::Declaration_struct(
    Location const &loc)
: Base(loc)
, m_name(NULL)
, m_fields()
{
}

// Get the declaration kind.
Declaration::Kind Declaration_struct::get_kind() const
{
    return s_kind;
}

// Set the struct name.
void Declaration_struct::set_name(Name *name)
{
    m_name = name;
}

// Add a struct field.
void Declaration_struct::add(Declaration *field)
{
    m_fields.push(field);
}

// ------------------------------- Declaration_interface-------------------------------

// Constructor.
Declaration_interface::Declaration_interface(
    Location const &loc,
    Name           *name)
: Base(loc)
, m_qual()
, m_name(name)
, m_fields()
, m_ident(NULL)
, m_array_specifiers()
{
}

// Get the declaration kind.
Declaration::Kind Declaration_interface::get_kind() const
{
    return s_kind;
}

// Set the interface block name.
void Declaration_interface::set_name(Name *name)
{
    m_name = name;
}

// Add a field declaration.
void Declaration_interface::add_field(Declaration *field)
{
    m_fields.push(field);
}

// Set the identifier.
void Declaration_interface::set_identifier(Name *ident)
{
    m_ident = ident;
}

// ------------------------------- Instance_name-------------------------------

// Constructor.
Instance_name::Instance_name(
    Name *name)
    : Base()
    , m_name(name)
{
}

// ------------------------------- Declaration_qualifier-------------------------------

// Constructor.
Declaration_qualified::Declaration_qualified(
    Location const &loc)
: Base(loc)
, m_qual()
, m_instances()
{
}

// Get the declaration kind.
Declaration::Kind Declaration_qualified::get_kind() const
{
    return s_kind;
}

// Add an instance.
void Declaration_qualified::add_instance(Instance_name *inst)
{
    m_instances.push(inst);
}

// ------------------------------- Decl_factory-------------------------------

// Constructor.
Decl_factory::Decl_factory(Memory_arena &arena)
: m_builder(arena)
{
}

// Creates a new name.
Name *Decl_factory::create_name(
    Location const &loc,
    Symbol         *sym)
{
    return m_builder.create<Name>(loc, sym);
}

// Creates a new array specifier.
Array_specifier *Decl_factory::create_array_specifier(
    Location const &loc,
    Expr           *size)
{
    return m_builder.create<Array_specifier>(loc, size);
}

// Creates a type name..
Type_name *Decl_factory::create_type_name(
    Location const &loc)
{
    return m_builder.create<Type_name>(loc);
}

// Create a new invalid declaration.
Declaration_invalid *Decl_factory::create_invalid(
    Location const &loc)
{
    return m_builder.create<Declaration_invalid>(loc);
}

// Creates a new init declarator.
Init_declarator *Decl_factory::create_init_declarator(
    Location const &loc)
{
    return m_builder.create<Init_declarator>(loc);
}

// Create a new variable declaration.
Declaration_variable *Decl_factory::create_variable(
    Type_name *name)
{
    HLSL_ASSERT(name != NULL);
    return m_builder.create<Declaration_variable>(name);
}

// Create a parameter declaration.
Declaration_param *Decl_factory::create_param(
    Type_name *name)
{
    HLSL_ASSERT(name != NULL);
    return m_builder.create<Declaration_param>(name);
}

// Creates a new function declaration.
Declaration_function *Decl_factory::create_function(
    Type_name *type_name,
    Name      *name)
{
    HLSL_ASSERT(type_name != NULL);
    HLSL_ASSERT(name != NULL);
    return m_builder.create<Declaration_function>(type_name, name);
}

// Creates a field declarator.
Field_declarator *Decl_factory::create_field(
    Location const &loc)
{
    return m_builder.create<Field_declarator>(loc);
}

// Creates a field declaration.
Declaration_field *Decl_factory::create_field_declaration(
    Type_name *type_name)
{
    HLSL_ASSERT(type_name != NULL);
    return m_builder.create<Declaration_field>(type_name);
}

// Creates a struct declaration.
Declaration_struct *Decl_factory::create_struct(
    Location const &loc)
{
    return m_builder.create<Declaration_struct>(loc);
}

// Creates an interface declaration.
Declaration_interface *Decl_factory::create_interface(
    Location const &loc,
    Name           *name)
{
    return m_builder.create<Declaration_interface>(loc, name);
}

// Creates an instance.
Instance_name *Decl_factory::create_instance(
    Name *name)
{
    HLSL_ASSERT(name != NULL);
    return m_builder.create<Instance_name>(name);
}

// Creates a qualified declaration.
Declaration_qualified *Decl_factory::create_qualified(
    Location const &loc)
{
     return m_builder.create<Declaration_qualified>(loc);
}

}  // hlsl
}  // mdl
}  // mi

