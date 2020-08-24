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
/// \file mi/mdl/mdl_names.h
/// \brief Interfaces for referenced names inside the MDL AST
#ifndef MDL_NAMES_H
#define MDL_NAMES_H 1

#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

class IExpression;
class ISymbol;
class IType;
class IDefinition;
class Position;

/// A simple name (i.e. does not contain '::') inside the MDL AST.
class ISimple_name : public Interface_owned
{
public:
    /// Get the symbol of this name.
    virtual ISymbol const *get_symbol() const = 0;

    /// Access the position of this name.
    virtual Position &access_position() = 0;

    /// Get the definition of this name if any was set.
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition for this name.
    virtual void set_definition(IDefinition const *def) = 0;

    /// Access the position of this name.
    virtual Position const &access_position() const = 0;
};

/// A qualified name inside the MDL AST.
class IQualified_name : public Interface_owned
{
public:
    /// Test if the qualified name is absolute, i.e. starts with '::'.
    virtual bool is_absolute() const = 0;

    /// Set the absolute flag of the qualified name.
    virtual void set_absolute(bool flag = true) = 0;

    /// Get the component count.
    virtual int get_component_count() const = 0;

    /// Get the component at index.
    virtual ISimple_name const *get_component(int index) const = 0;

    /// Add a component to this qualified name (at the end).
    ///
    /// param sname  a simple name to add
    virtual void add_component(ISimple_name const *sname) = 0;

    /// Removes all components from this qualified name.
    virtual void clear_components() = 0;

    /// Get the definition for this name if any was set.
    virtual IDefinition const *get_definition() const = 0;

    /// Set the definition for this name.
    virtual void set_definition(IDefinition const *def) = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;
};

/// Frequency qualifiers inside the MDL AST.
enum Qualifier {
    FQ_NONE,                ///< No explicit qualifier.
    FQ_VARYING,             ///< varying.
    FQ_UNIFORM              ///< uniform.
};

/// A type name inside the MDL AST.
class IType_name : public Interface_owned
{
public:
    /// Test if the qualified name is absolute.
    virtual bool is_absolute() const = 0;

    /// Set the absolute flag of the qualified name.
    virtual void set_absolute() = 0;

    /// Get the frequency qualifier.
    virtual Qualifier get_qualifier() const = 0;

    /// Set the frequency qualifier.
    ///
    /// param qualifier  the qualifier to set
    virtual void set_qualifier(Qualifier qualifier) = 0;

    /// Get the qualified name.
    /// If the type name is an anonymous tuple return type, this returns NULL.
    virtual IQualified_name *get_qualified_name() const = 0;

    /// Set the qualified name.
    virtual void set_qualified_name(IQualified_name *name) = 0;

    /// Check if this is an array.
    virtual bool is_array() const = 0;

    /// Check if this is a concrete array.
    virtual bool is_concrete_array() const = 0;

    /// Get the array size argument.
    /// If the type name is not an array or not a concrete array, this returns NULL.
    virtual IExpression const *get_array_size() const = 0;

    /// Get the name of the abstract array length name.
    /// If the type name is not an array or not an abstract array, this returns NULL.
    virtual ISimple_name const *get_size_name() const = 0;

    /// Set a concrete array size.
    ///
    /// \param size  a constant expression whose value is the array size
    virtual void set_array_size(IExpression const *size) = 0;

    /// Set a size name.
    ///
    /// \param sname  the deffered size array name
    virtual void set_size_name(ISimple_name const *sname) = 0;

    /// Access the position.
    virtual Position &access_position() = 0;

    /// Access the position.
    virtual Position const &access_position() const = 0;

    /// Get the type represented by this type name (if it references a type).
    ///
    /// \return if this type name references a name of a type, the type
    ///         NULL otherwise
    virtual IType const *get_type() const = 0;

    /// Set the type represented by this name.
    ///
    /// \param type  the MDL type to set
    virtual void set_type(IType const *type) = 0;

    /// Mark this as an incomplete array, i.e. has no array size.
    virtual void set_incomplete_array() = 0;

    /// Check if this is an incomplete array.
    virtual bool is_incomplete_array() const = 0;
};

/// A factory for creating MDL AST names.
class IName_factory : public Interface_owned
{
public:
    /// Create a new Symbol and enters it into the symbol table.
    virtual ISymbol const *create_symbol(char const *name) = 0;

    /// Creates a simple name.
    ///
    /// \param sym           the Symbol of the name
    /// \param start_line    the start line of this name in the input
    /// \param start_column  the start column of this name in the input
    /// \param end_line      the end line of this name in the input
    /// \param end_column    the end column of this name in the input
    virtual ISimple_name *create_simple_name(
        ISymbol const *sym,
        int           start_line   = 0,
        int           start_column = 0,
        int           end_line     = 0,
        int           end_column  = 0) = 0;

    /// Creates a new (empty) qualified name.
    ///
    /// \param start_line    the start line of this name in the input
    /// \param start_column  the start column of this name in the input
    /// \param end_line      the end line of this name in the input
    /// \param end_column    the end column of this name in the input
    virtual IQualified_name *create_qualified_name(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;

    /// Creates a new type name.
    ///
    /// \param qualified_name  the qualified name of the type name
    /// \param start_line      the start line of this name in the input
    /// \param start_column    the start column of this name in the input
    /// \param end_line        the end line of this name in the input
    /// \param end_column      the end column of this name in the input
    virtual IType_name *create_type_name(
        IQualified_name *qualified_name,
        int             start_line = 0,
        int             start_column = 0,
        int             end_line = 0,
        int             end_column = 0) = 0;
};

}  // mdl
}  // mi

#endif
