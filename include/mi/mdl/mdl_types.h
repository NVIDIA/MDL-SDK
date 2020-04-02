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
/// \file mi/mdl/mdl_types.h
/// \brief Declaration of MDL types used inside the MDL AST and DAG IR
#ifndef MDL_TYPES_H
#define MDL_TYPES_H 1

#include <cstddef>

#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

class IDefinition;
class ISymbol;
class ISymbol_table;

/// The base interface of an MDL type.
class IType : public Interface_owned
{
public:
    /// The possible kinds of types.
    enum Kind {
        TK_ALIAS,                   ///< An alias for another type, aka typedef.
        TK_BOOL,                    ///< The boolean type.
        TK_INT,                     ///< The integer type.
        TK_ENUM,                    ///< An enum type.
        TK_FLOAT,                   ///< The float type.
        TK_DOUBLE,                  ///< The double type.
        TK_STRING,                  ///< The string type.
        TK_LIGHT_PROFILE,           ///< The light profile type.
        TK_BSDF,                    ///< The bsdf type.
        TK_HAIR_BSDF,               ///< The hair_bsdf type.
        TK_EDF,                     ///< The edf type.
        TK_VDF,                     ///< The vdf type.
        TK_VECTOR,                  ///< A vector type.
        TK_MATRIX,                  ///< A matrix type.
        TK_ARRAY,                   ///< An array type.
        TK_COLOR,                   ///< The color type.
        TK_FUNCTION,                ///< A function type.
        TK_STRUCT,                  ///< A struct type.
        TK_TEXTURE,                 ///< A texture type.
        TK_BSDF_MEASUREMENT,        ///< The bsdf measurement type.
        TK_INCOMPLETE,              ///< The incomplete type, used internally by the compiler.
        TK_ERROR = 0xFFFFFFFF       ///< The error type.
    };

    /// The possible kinds of type modifiers.
    enum Modifier {
        MK_NONE        = 0,         ///< auto-typed
        MK_CONST       = (1 << 1),  ///< a constant type
        MK_UNIFORM     = (1 << 2),  ///< a uniform type
        MK_VARYING     = (1 << 4)   ///< a varying type
    };

    /// A bitset of type modifier.
    typedef unsigned Modifiers;

    /// Get the kind of type.
    virtual Kind get_kind() const = 0;

    /// Get the type modifiers of a type
    virtual Modifiers get_type_modifiers() const = 0;

    /// If this type is an alias type, skip all aliases and return the base type,
    /// else the type itself.
    virtual IType *skip_type_alias() = 0;

    /// If this type is an alias type, skip all aliases and return the base type,
    /// else the type itself.
    virtual IType const *skip_type_alias() const = 0;
};

/// An MDL alias type (aka typedef).
///
/// Note that types with modifiers are represented solely using alias types,
/// so a uniform T is an alias of the type T (without a name).
class IType_alias : public IType
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ALIAS;

    /// Get the alias name of the type, can be NULL.
    virtual ISymbol const *get_symbol() const = 0;

    /// Get the type aliased by this type.
    virtual IType const *get_aliased_type() const = 0;

    /// Get the modifier set of this type.
    virtual IType::Modifiers get_type_modifiers() const = 0;
};

/// The MDL error type.
///
/// The error type represents a type error in the syntax tree.
/// No valid MDL module contains error types.
class IType_error : public IType
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ERROR;
};

/// The incomplete type temporary represents a in the syntax tree that is not yet deduced.
class IType_incomplete : public IType
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INCOMPLETE;
};

/// The base interface of all atomic types.
class IType_atomic : public IType
{
};

/// The bool type.
class IType_bool : public IType_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BOOL;
};

/// The int type
class IType_int : public IType_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_INT;
};

/// An enum type.
class IType_enum : public IType_atomic
{
public:
    /// Identifiers of enum types.
    enum Predefined_id {
        EID_USER           = -1,  ///< This is a user defined enum type.
        EID_TEX_GAMMA_MODE = 0,   ///< This is the ::%tex::gamma_mode enum type.
        EID_INTENSITY_MODE = 1,   ///< This is the MDL 1.1 intensity_mode enum type.
        EID_LAST           = EID_INTENSITY_MODE
    };

    /// The kind of this subclass.
    static Kind const s_kind = TK_ENUM;

    /// Get the name of this enum type.
    virtual ISymbol const *get_symbol() const = 0;

    /// Add an enum value.
    ///
    /// \param name  The name of the value.
    /// \param code  The code of the value.
    ///
    /// \return The index of this new value.
    virtual int add_value(ISymbol const *name, int code) = 0;

    /// Get the number of values.
    virtual int get_value_count() const = 0;

    /// Get a value at given index.
    ///
    /// \param[in]  index    The index of the value.
    /// \param[out] name     The name of the value.
    /// \param[out] code     The code of the value.
    ///
    /// \return  true of success, false on index error.
    virtual bool get_value(
        int           index,
        ISymbol const *&name,
        int           &code) const = 0;

    /// Lookup a value in O(N).
    ///
    /// \param[in]  name     The name of the value.
    /// \param[out] code     The code of the value.
    ///
    /// \return true if the name was found, false otherwise.
    virtual bool lookup(
        ISymbol const *name,
        int           &code) const = 0;

    /// If this enum is a predefined one, return its ID, else EID_USER.
    virtual Predefined_id get_predefined_id() const = 0;
};

/// The float type.
class IType_float : public IType_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_FLOAT;
};

/// The double type.
class IType_double : public IType_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_DOUBLE;
};

/// The string type.
class IType_string : public IType_atomic
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_STRING;
};

/// A reference types.
///
/// Reference types are similar to pointer types in C.
class IType_reference : public IType
{
};

/// The type of distribution functions.
class IType_df : public IType_reference
{
};

/// The bsdf type.
class IType_bsdf : public IType_df
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BSDF;
};

/// The hair_bsdf type.
class IType_hair_bsdf : public IType_df
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_HAIR_BSDF;
};

/// The edf type.
class IType_edf : public IType_df
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_EDF;
};

/// The vdf type.
class IType_vdf : public IType_df
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VDF;
};

/// A compound type.
class IType_compound : public IType
{
public:
    /// Get the compound sub type at index.
    ///
    /// \param index  the index of the compound sub type
    virtual IType const *get_compound_type(int index) const = 0;

    /// Get the number of compound elements.
    virtual int get_compound_size() const = 0;
};

/// A vector type.
class IType_vector : public IType_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_VECTOR;

    /// Get the type of the vector elements.
    virtual IType_atomic const *get_element_type() const = 0;

    /// Get the number of vector elements.
    virtual int get_size() const = 0;
};

/// A matrix type.
class IType_matrix : public IType_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_MATRIX;

    /// Get the type of the matrix elements.
    virtual IType_vector const *get_element_type() const = 0;

    /// Get the number of matrix columns.
    virtual int get_columns() const = 0;
};

/// An abstract array length.
class IType_array_size : public Interface_owned
{
public:
    /// Get the name of the abstract array length symbol.
    virtual ISymbol const *get_size_symbol() const = 0;

    /// Get the absolute name of the this array length.
    virtual ISymbol const *get_name() const = 0;
};

/// An array type.
class IType_array : public IType_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_ARRAY;

    /// Get the type of the array elements.
    virtual IType const *get_element_type() const = 0;

    /// Test if the array is immediate sized.
    virtual bool is_immediate_sized() const = 0;

    /// Get the size of the (immediate sized) array.
    virtual int get_size() const = 0;

    /// Get the deferred array size.
    virtual IType_array_size const *get_deferred_size() const = 0;
};

/// The color type.
class IType_color : public IType_compound
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_COLOR;
};

/// A function type.
class IType_function : public IType
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_FUNCTION;

    /// Get the return type of the function.
    virtual IType const *get_return_type() const = 0;

    /// Get the number of parameters of the function.
    virtual int get_parameter_count() const = 0;

    /// Get a parameter of the function.
    ///
    /// \param[in]  index    The index of the parameter in the parameter list.
    /// \param[out] type     The type of the parameter.
    /// \param[out] name     The name of the parameter.
    virtual void get_parameter(
        int           index,
        IType const   *&type,
        ISymbol const *&name) const = 0;
};

/// A struct type.
class IType_struct : public IType_compound
{
public:
    /// Identifiers of enum types.
    enum Predefined_id {
        SID_USER = -1,
        SID_MATERIAL_EMISSION = 0,
        SID_MATERIAL_SURFACE,
        SID_MATERIAL_VOLUME,
        SID_MATERIAL_GEOMETRY,
        SID_MATERIAL,
        SID_LAST = SID_MATERIAL
    };

    /// The kind of this subclass.
    static Kind const s_kind = TK_STRUCT;

    /// Get the name of the struct type.
    virtual ISymbol const *get_symbol() const = 0;

    /// Add a field to the struct type (at the end of the field list).
    ///
    /// \param type  The type of the field.
    /// \param name  The name of the field.
    virtual void add_field(
        IType const   *type,
        ISymbol const *name) = 0;

    /// Get the number of fields.
    virtual int get_field_count() const = 0;

    /// Get a field.
    ///
    /// \param[in]  index    The index of the field.
    /// \param[out] type     The type of the field.
    /// \param[out] name     The name of the field.
    virtual void get_field(
        int           index,
        IType const   *&type,
        ISymbol const *&name) const = 0;

    /// Return the index of a field in O(N) if it is present and -1 otherwise.
    ///
    /// \param name     The name of the field.
    virtual int find_field(ISymbol const *name) const = 0;

    /// Return the index of a field in O(N) if it is present and -1 otherwise.
    ///
    /// \param name     The name of the field.
    virtual int find_field(char const *name) const = 0;

    /// Return the index of a method in O(N) if it is present and -1 otherwise.
    ///
    /// \param name     The name of the method.
    virtual int find_method(ISymbol const *name) const = 0;

    /// Add a method.
    ///
    /// \param type  The type of the method.
    /// \param name  The name of the method.
    virtual void add_method(
        IType_function const *type,
        ISymbol const        *name) = 0;

    /// Get the number of methods.
    virtual int get_method_count() const = 0;

    /// Get a method.
    ///
    /// \param[in]  index    The index of the method.
    /// \param[out] type     The type of the field.
    /// \param[out] name     The name of the field.
    virtual void get_method(
        int                  index,
        IType_function const *&type,
        ISymbol const        *&name) const = 0;

    /// If this struct is a predefined one, return its ID, else SID_USER.
    virtual Predefined_id get_predefined_id() const = 0;
};

/// A string valued resource type.
class IType_resource : public IType_reference
{
};

/// The light profile type.
class IType_light_profile : public IType_resource
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_LIGHT_PROFILE;
};

/// A texture type.
class IType_texture : public IType_resource
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_TEXTURE;

    /// The possible texture shapes.
    enum Shape {
        TS_2D,         ///< A 2D texture.
        TS_3D,         ///< A 3D texture.
        TS_CUBE,       ///< A cube texture.
        TS_PTEX,       ///< A PTEX texture.
        TS_BSDF_DATA,  ///< A 3D texture representing a BSDF data table.
    };

    /// Get the texture shape.
    virtual Shape get_shape() const = 0;

    /// Get the coordinate type.
    virtual IType const *get_coord_type() const = 0;
};

/// The bsdf_measurement type.
class IType_bsdf_measurement : public IType_resource
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BSDF_MEASUREMENT;
};

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T *as(IType *type) {
    type = type->skip_type_alias();
    return (type->get_kind() == T::s_kind) ? static_cast<T *>(type) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline IType_alias *as<>(IType *type) {
    return (type->get_kind() == IType_alias::s_kind) ? static_cast<IType_alias *>(type) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<typename T>
T const *as(IType const *type) {
    type = type->skip_type_alias();
    return (type->get_kind() == T::s_kind) ? static_cast<T const *>(type) : NULL;
}

/// Cast to subtype or return NULL if types do not match.
template<>
inline IType_alias const *as<>(IType const *type) {
    return (type->get_kind() == IType_alias::s_kind) ?
        static_cast<IType_alias const *>(type) : NULL;
}

/// Cast to IType_atomic or return NULL if types do not match.
template<>
inline IType_atomic *as<IType_atomic>(IType *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
        return static_cast<IType_atomic *>(type);
    default:
        return NULL;
    }
}

/// Cast to IType_atomic or return NULL if types do not match.
template<>
inline IType_atomic const *as<IType_atomic>(IType const *type) { //-V659 PVS
    return const_cast<IType_atomic const *>(as<IType_atomic>(const_cast<IType *>(type)));
}

/// Cast to IType_reference or return NULL if types do not match.
template<>
inline IType_reference *as<IType_reference>(IType *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_VDF:
    case IType::TK_EDF:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        return static_cast<IType_reference *>(type);
    default:
        return NULL;
    }
}

/// Cast to IType_df or return NULL if types do not match.
template<>
inline IType_reference const *as<IType_reference>(IType const *type) { //-V659 PVS
    return const_cast<IType_reference const *>(as<IType_reference>(const_cast<IType *>(type)));
}

/// Cast to IType_df or return NULL if types do not match.
template<>
inline IType_df *as<IType_df>(IType *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_VDF:
    case IType::TK_EDF:
        return static_cast<IType_df *>(type);
    default:
        return NULL;
    }
}

/// Cast to IType_df or return NULL if types do not match.
template<>
inline IType_df const *as<IType_df>(IType const *type) { //-V659 PVS
    return const_cast<IType_df const *>(as<IType_df>(const_cast<IType *>(type)));
}

/// Cast to IType_compound or return NULL if types do not match.
template<>
inline IType_compound *as<IType_compound>(IType *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_ARRAY:
    case IType::TK_COLOR:
    case IType::TK_STRUCT:
        return static_cast<IType_compound *>(type);
    default:
        return NULL;
    }
}

/// Cast to IType_compound or return NULL if types do not match.
template<>
inline IType_compound const *as<IType_compound>(IType const *type) { //-V659 PVS
    return const_cast<IType_compound const *>(as<IType_compound>(const_cast<IType *>(type)));
}

/// Cast to IType_resource or return NULL if types do not match.
template<>
inline IType_resource *as<IType_resource>(IType *type) {
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_TEXTURE:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF_MEASUREMENT:
        return static_cast<IType_resource *>(type);
    default:
        return NULL;
    }
}

/// Cast to IType_resource or return NULL if types do not match.
template<>
inline IType_resource const *as<IType_resource>(IType const *type) { //-V659 PVS
    return const_cast<IType_resource const *>(as<IType_resource>(const_cast<IType *>(type)));
}

/// Check if a type is of a certain type.
template<typename T>
bool is(IType const *type) {
    return type->get_kind() == T::s_kind;
}

/// Check if a type is of type IType_atomic.
template<>
inline bool is<IType_atomic>(IType const *type) {
    switch (type->get_kind()) {
    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_ENUM:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type IType_reference.
template<>
inline bool is<IType_reference>(IType const *type) {
    switch (type->get_kind()) {
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_VDF:
    case IType::TK_EDF:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type IType_df.
template<>
inline bool is<IType_df>(IType const *type) {
    switch (type->get_kind()) {
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_VDF:
    case IType::TK_EDF:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type IType_compound.
template<>
inline bool is<IType_compound>(IType const *type) {
    switch (type->get_kind()) {
    case IType::TK_VECTOR:
    case IType::TK_MATRIX:
    case IType::TK_ARRAY:
    case IType::TK_COLOR:
    case IType::TK_STRUCT:
        return true;
    default:
        return false;
    }
}

/// Check if a type is of type IType_resource.
template<>
inline bool is<IType_resource>(IType const *type) {
    switch (type->get_kind()) {
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_TEXTURE:
    case IType::TK_BSDF_MEASUREMENT:
        return true;
    default:
        return false;
    }
}

/// The interface for creating types.
///
/// An IType_factory interface can be obtained by calling
/// the method get_type_factory() on the interfaces IModule and IValue_factory.
///
/// \note Real user defined types in MDL are only enum types, struct types, alias
///       types and array type.
///       All other types are singletons and exists only once in the MDL Core.
///       Also it is an error to create the same type twice for one module.
///       Hence, two types can be checked for equality by a pointer-compare.
///       However, sometimes one has to check for equality but ignore any aliases,
///       use t1->skip_type_alias() == t2->skip_type_alias() then.
class IType_factory : public Interface_owned
{
public:
    /// Create a new type alias instance.
    ///
    /// \param type       The aliased type.
    /// \param name       The alias name, may be NULL.
    /// \param modifiers  The type modifiers.
    virtual IType const *create_alias(
        IType const      *type,
        ISymbol const    *name,
        IType::Modifiers modifiers) = 0;

    /// Create a new type error instance.
    virtual IType_error const *create_error() = 0;

    /// Create a new type incomplete instance.
    virtual IType_incomplete const *create_incomplete() = 0;

    /// Create a new type bool instance.
    virtual IType_bool const *create_bool() = 0;

    /// Create a new type int instance.
    virtual IType_int const *create_int() = 0;

    /// Create a new type enum instance.
    ///
    /// \param name The name of the enum.
    virtual IType_enum *create_enum(ISymbol const *name) = 0;

    /// Lookup an enum type.
    ///
    /// \param name The name of the enum.
    ///
    /// \returns the type enum instance or NULL if it does not exist.
    virtual IType_enum const *lookup_enum(char const *name) const = 0;

    /// Create a new type float instance.
    virtual IType_float const *create_float() = 0;

    /// Create a new type double instance.
    virtual IType_double const *create_double() = 0;

    /// Create a new type string instance.
    virtual IType_string const *create_string() = 0;

    /// Create a new type bsdf instance.
    virtual IType_bsdf const *create_bsdf() = 0;

    /// Create a new type hair_bsdf instance.
    virtual IType_hair_bsdf const *create_hair_bsdf() = 0;

    /// Create a new type edf instance.
    virtual IType_edf const *create_edf() = 0;

    /// Create a new type vdf instance.
    virtual IType_vdf const *create_vdf() = 0;

    /// Create a new type light profile instance.
    virtual IType_light_profile const *create_light_profile() = 0;

    /// Create a new type vector instance.
    ///
    /// \param element_type The type of the vector elements.
    /// \param size         The size of the vector.
    virtual IType_vector const *create_vector(
        IType_atomic const *element_type,
        int                size) = 0;

    /// Create a new type matrix instance.
    ///
    /// \param element_type The type of the matrix elements.
    /// \param columns      The number of columns.
    virtual IType_matrix const *create_matrix(
        IType_vector const *element_type,
        int                columns) = 0;

    /// Create a new type abstract array instance.
    ///
    /// \param element_type The element type of the array.
    /// \param sym          The array size symbol.
    /// \param abs_name     The absolute name of the array size.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    virtual IType const *create_array(
        IType const   *element_type,
        ISymbol const *sym,
        ISymbol const *abs_name) = 0;

    /// Create a new type sized array instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The size of the array.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    virtual IType const *create_array(
        IType const *element_type,
        size_t      size) = 0;

    /// Create a new type color instance.
    virtual const IType_color *create_color() = 0;

    /// A simple value helper class, a pair of an parameter type and name.
    struct Function_parameter {
        IType const   *p_type;        ///< The type of the parameter.
        ISymbol const *p_sym;         ///< The name of the parameter.
    };

    /// Create a new type function type instance.
    ///
    /// \param return_type   The return type of the function.
    /// \param parameters    The parameters of the function.
    /// \param n_parameters  The number of parameters.
    virtual IType_function const *create_function(
        IType const                      *return_type,
        Function_parameter const * const parameters,
        size_t                           n_parameters) = 0;

    /// Create a new type struct instance.
    ///
    /// \param name The name of the struct.
    virtual IType_struct *create_struct(ISymbol const *name) = 0;

    /// Lookup a struct type.
    ///
    /// \param name The name of the struct.
    ///
    /// \returns the type struct instance or NULL if it does not exist.
    virtual IType_struct const *lookup_struct(char const *name) const = 0;

    /// Create a new type texture instance.
    ///
    /// \param shape    The shape of the texture.
    virtual IType_texture const *create_texture(
        IType_texture::Shape shape) = 0;

    /// Create a new type bsdf_measurement instance.
    virtual IType_bsdf_measurement const *create_bsdf_measurement() = 0;

    /// Import a type from another type factory.
    ///
    /// \param type  the type to import
    virtual IType const *import(IType const *type) = 0;

    /// Return a predefined struct.
    ///
    /// \param part  the ID of the predefined struct
    virtual IType_struct *get_predefined_struct(IType_struct::Predefined_id part) = 0;

    /// Return a predefined enum.
    ///
    /// \param part  the ID of the predefined enum
    virtual IType_enum *get_predefined_enum(IType_enum::Predefined_id part) = 0;

    /// Return the symbol table of this type factory.
    virtual ISymbol_table *get_symbol_table() = 0;
};

}  // mdl
}  // mi

#endif
