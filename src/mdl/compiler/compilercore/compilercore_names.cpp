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

#include "compilercore_cc_conf.h"
#include "compilercore_names.h"
#include "compilercore_positions.h"
#include "compilercore_memory_arena.h"
#include "compilercore_symbols.h"

namespace mi {
namespace mdl {

/// Implementation of a simple name.
class Simple_name : public ISimple_name
{
    typedef ISimple_name Base;
public:
    /// Get the symbol.
    ISymbol const *get_symbol() const MDL_FINAL;

    /// Access the position.
    Position &access_position() MDL_FINAL;

    /// Get the definition for this name.
    IDefinition const *get_definition() const MDL_FINAL;

    /// Set the definition for this name.
    void set_definition(IDefinition const *def) MDL_FINAL;

    /// Access the position.
    Position const &access_position() const MDL_FINAL;

    /// Constructor.
    ///
    /// \param sym           the Symbol of the name
    /// \param start_line    the start line of this name in the input
    /// \param start_column  the start column of this name in the input
    /// \param end_line      the end line of this name in the input
    /// \param end_column    the end column of this name in the input
    explicit Simple_name(
        ISymbol const *sym,
        int           start_line = 0,
        int           start_column = 0,
        int           end_line = 0,
        int           end_column = 0);

private:
    // non copyable
    Simple_name(Simple_name const &) MDL_DELETED_FUNCTION;
    Simple_name &operator=(Simple_name const &) MDL_DELETED_FUNCTION;

private:
    /// The symbol of this name.
    ISymbol const *m_sym;

    /// The definition of this name if any.
    IDefinition const *m_def;

    /// The position of this name.
    Position_impl m_pos;
};

/// Implementation of a qualified name.
class Qualified_name : public IQualified_name
{
    typedef IQualified_name Base;
public:

    /// Test if the qualified name is absolute.
    bool is_absolute() const MDL_FINAL;

    /// Set the absolute flag of the qualified name.
    void set_absolute(bool flag = true) MDL_FINAL;

    /// Get the component count.
    int get_component_count() const MDL_FINAL;

    /// Get the component at index.
    ISimple_name const *get_component(int index) const MDL_FINAL;

    /// Add a component.
    void add_component(ISimple_name const *name) MDL_FINAL;

    /// Removes all componens.
    void clear_components() MDL_FINAL;

    /// Get the definition for this name.
    IDefinition const *get_definition() const MDL_FINAL;

    /// Set the definition for this name.
    void set_definition(IDefinition const *def) MDL_FINAL;

    /// Access the position.
    Position &access_position() MDL_FINAL;

    /// Access the position.
    Position const &access_position() const MDL_FINAL;

    /// Constructor.
    ///
    /// \param arena         the memory arena to allocate on
    /// \param start_line    the start line of this name in the input
    /// \param start_column  the start column of this name in the input
    /// \param end_line      the end line of this name in the input
    /// \param end_column    the end column of this name in the input
    explicit Qualified_name(
        Memory_arena *arena,
        int           start_line = 0,
        int           start_column = 0,
        int           end_line = 0,
        int           end_column = 0);

private:
    // non copyable
    Qualified_name(Qualified_name const &) MDL_DELETED_FUNCTION;
    Qualified_name &operator=(Qualified_name const &) MDL_DELETED_FUNCTION;

private:
    /// The definition of this name if any.
    IDefinition const *m_def;

    /// A flag to indicate if this name is absolute.
    bool m_is_absolute;

    /// The symbol of this name.
    Arena_vector<ISimple_name const *>::Type m_components;

    /// The position of this name.
    Position_impl m_pos;
};

/// Implementation of a type name.
class Type_name : public IType_name
{
    typedef IType_name Base;
public:

    /// Test if the qualified name is absolute.
    bool is_absolute() const MDL_FINAL;

    /// Set the absolute flag of the qualified name.
    void set_absolute() MDL_FINAL;

    /// Get the qualifier.
    Qualifier get_qualifier() const MDL_FINAL { return m_qualifier; }

    /// Set the qualifier.
    void set_qualifier(Qualifier qualifier) MDL_FINAL { m_qualifier = qualifier; }

    /// Get the qualified name.
    IQualified_name *get_qualified_name() const MDL_FINAL;

    /// Set the qualified name.
    void set_qualified_name(IQualified_name *name) MDL_FINAL;

    /// Check if this is an array.
    bool is_array() const MDL_FINAL;

    /// Check if this is a concrete array.
    bool is_concrete_array() const MDL_FINAL;

    /// Get the array size argument.
    IExpression const *get_array_size() const MDL_FINAL;

    /// Get the name of the abstract array length name.
    ISimple_name const *get_size_name() const MDL_FINAL;

    /// Set a concrete array size.
    void set_array_size(IExpression const *size) MDL_FINAL;

    /// Set a size name.
    void set_size_name(ISimple_name const *name) MDL_FINAL;

    /// Access the position.
    Position &access_position() MDL_FINAL;

    /// Access the position.
    Position const &access_position() const MDL_FINAL;

    /// Get the type represented by this type name.
    IType const *get_type() const MDL_FINAL;

    /// Set the type represented by this name.
    void set_type(IType const *type) MDL_FINAL;

    /// Check if this is an incomplete array.
    bool is_incomplete_array() const MDL_FINAL;

    /// Mark this as an incomplete array.
    void set_incomplete_array() MDL_FINAL;

    /// Constructor.
    ///
    /// \param arena           the memory arena to allocate on
    /// \param qualified_name  the qualified name of this type name
    /// \param start_line      the start line of this name in the input
    /// \param start_column    the start column of this name in the input
    /// \param end_line        the end line of this name in the input
    /// \param end_column      the end column of this name in the input
    explicit Type_name(
        Memory_arena    *arena,
        IQualified_name *qualified_name,
        int             start_line = 0,
        int             start_column = 0,
        int             end_line = 0,
        int             end_column = 0);

private:
    // non copyable
    Type_name(Type_name const &) MDL_DELETED_FUNCTION;
    Type_name &operator=(Type_name const &) MDL_DELETED_FUNCTION;

private:
    /// The frequency qualifier.
    Qualifier m_qualifier;

    /// The qualified_name.
    IQualified_name *m_qualified_name;

    /// The flag to indicate if this is an array.
    bool m_is_array;

    /// The flag to indicate if this is a concrete array.
    bool m_is_concrete_array;

    /// The concrete array size.
    IExpression const *m_array_size;

    /// The size name.
    ISimple_name const *m_size_name;

    /// The described type.
    IType const *m_type;

    /// The position of this name.
    Position_impl m_pos;
};

// Get the name.
ISymbol const *Simple_name::get_symbol() const {
    return m_sym;
}

// Get the definition for this name.
IDefinition const *Simple_name::get_definition() const
{
    return m_def;
}

// Set the definition for this name.
void Simple_name::set_definition(IDefinition const *def)
{
    m_def = def;
}

// Access the position.
Position &Simple_name::access_position() {
    return m_pos;
}

// Access the position.
Position const &Simple_name::access_position() const {
    return m_pos;
}

// Constructor.
Simple_name::Simple_name(
    ISymbol const *sym,
    int           start_line,
    int           start_column,
    int           end_line,
    int           end_column)
: Base()
, m_sym(sym)
, m_def(NULL)
, m_pos(start_line, start_column, end_line, end_column)
{
}

// Test if the qualified name is absolute.
bool Qualified_name::is_absolute() const
{
    return m_is_absolute;
}

// Set the absolute flag of the qualified name.
void Qualified_name::set_absolute(bool flag)
{
    m_is_absolute = flag;
}

// Get the component count.
int Qualified_name::get_component_count() const
{
    return m_components.size();
}

// Get the component at index.
ISimple_name const *Qualified_name::get_component(int index) const
{
    return m_components.at(index);
}

// Add a component.
void Qualified_name::add_component(ISimple_name const *name)
{
    Position const &pos = name->access_position();
    if (m_pos.get_start_line() == 0) {
        m_pos.set_start_line(pos.get_start_line());
        m_pos.set_start_column(pos.get_start_column());
    }
    m_pos.set_end_line(pos.get_end_line());
    m_pos.set_end_column(pos.get_end_column());

    m_components.push_back(name);
}

// Removes all components.
void Qualified_name::clear_components()
{
    m_components.clear();
}

// Get the definition for this name.
IDefinition const *Qualified_name::get_definition() const
{
    return m_def;
}

// Set the definition for this name.
void Qualified_name::set_definition(IDefinition const *def)
{
    m_def = def;
}

// Access the position.
Position &Qualified_name::access_position()
{
    return m_pos;
}

// Access the position.
Position const &Qualified_name::access_position() const
{
    return m_pos;
}

// Constructor.
Qualified_name::Qualified_name(
    Memory_arena *arena,
    int          start_line,
    int          start_column,
    int          end_line,
    int          end_column)
: Base()
, m_def(NULL)
, m_is_absolute(false)
, m_components(arena)
, m_pos(start_line, start_column, end_line, end_column)
{
}

// Constructor.
Type_name::Type_name(
    Memory_arena    *arena,
    IQualified_name *qualified_name,
    int             start_line,
    int             start_column,
    int             end_line,
    int             end_column)
: Base()
, m_qualifier(FQ_NONE)
, m_qualified_name(qualified_name)
, m_is_array(false)
, m_is_concrete_array(false)
, m_array_size(NULL)
, m_size_name(NULL)
, m_type(NULL)
, m_pos(start_line, start_column, end_line, end_column)
{
}

// Test if the qualified name is absolute.
bool Type_name::is_absolute() const
{
    return get_qualified_name()->is_absolute();
}

// Set the absolute flag of the qualified name.
void Type_name::set_absolute()
{
    get_qualified_name()->set_absolute();
}

// Get the qualified name.
IQualified_name *Type_name::get_qualified_name() const
{
    return m_qualified_name;
}

// Set the qualified name.
void Type_name::set_qualified_name(IQualified_name *name)
{
    m_qualified_name = name;
}

// Check if this is an array.
bool Type_name::is_array() const
{
    return m_is_array;
}

// Check if this is a concrete array.
bool Type_name::is_concrete_array() const
{
    return m_is_concrete_array;
}

// Get the array size argument.
IExpression const *Type_name::get_array_size() const
{
    return m_array_size;
}

// Get the name of the abstract array length symbol.
ISimple_name const *Type_name::get_size_name() const
{
    return m_size_name;
}

// Set a concrete array size.
void Type_name::set_array_size(IExpression const *size)
{
    m_array_size = size;
    m_is_array = true;
    m_is_concrete_array = true;
}

// Set a size name.
void Type_name::set_size_name(ISimple_name const *name)
{
    m_size_name = name;
    m_is_array = true;
    m_is_concrete_array = false;
}

// Get the type represented by this type name.
IType const *Type_name::get_type() const
{
    return m_type;
}

// Set the type represented by this name.
void Type_name::set_type(IType const *type)
{
    m_type = type;
}

// Check if this is an incomplete array.
bool Type_name::is_incomplete_array() const
{
    return m_is_concrete_array && m_array_size == NULL;
}

// Mark this as an incomplete array.
void Type_name::set_incomplete_array()
{
    m_is_array          = true;
    m_is_concrete_array = true;
    m_array_size        = NULL;
    m_size_name         = NULL;
}

// Access the position.
Position &Type_name::access_position() {
    return m_pos;
}

// Access the position.
Position const &Type_name::access_position() const {
    return m_pos;
}

// Constructor.
Name_factory::Name_factory(
    Symbol_table &sym_tab,
    Memory_arena &arena)
: m_sym_tab(sym_tab)
, m_builder(arena)
{
}

// Create a new Symbol.
ISymbol const *Name_factory::create_symbol(char const *name)
{
    return m_sym_tab.create_symbol(name);
}

// Creates a simple name.
ISimple_name *Name_factory::create_simple_name(
    ISymbol const *sym,
    int           start_line,
    int           start_column,
    int           end_line,
    int           end_column)
{
    return m_builder.create<Simple_name>(sym, start_line, start_column, end_line, end_column);
}

// Creates a new (empty) qualified name.
IQualified_name *Name_factory::create_qualified_name(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    return m_builder.create<Qualified_name>(m_builder.get_arena());
}

// Creates a new type name.
IType_name *Name_factory::create_type_name(
    IQualified_name *qualified_name,
    int             start_line,
    int             start_column,
    int             end_line,
    int             end_column)
{
    return m_builder.create<Type_name>(
        m_builder.get_arena(), qualified_name, start_line, start_column, end_line, end_column);
}

}  // mdl
}  // mi
