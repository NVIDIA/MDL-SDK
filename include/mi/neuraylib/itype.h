/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief      Types of the MDL type system

#ifndef MI_NEURAYLIB_ITYPE_H
#define MI_NEURAYLIB_ITYPE_H

#include <mi/base/interface_declare.h>

namespace mi {

class IString;

namespace neuraylib {

/** \defgroup mi_neuray_mdl_types MDL type system
    \ingroup mi_neuray

    The MDL type system mainly consists of four sets of interfaces:
    - Types are represented by #mi::neuraylib::IType and are constructed via
      #mi::neuraylib::IType_factory,
    - Values are represented by #mi::neuraylib::IValue and are constructed via
      #mi::neuraylib::IValue_factory,
    - Expressions are represented by #mi::neuraylib::IExpression and are constructed via
      #mi::neuraylib::IExpression_factory, and
    - Annotations are represented by #mi::neuraylib::IAnnotation and are constructed via
      #mi::neuraylib::IExpression_factory.

    The three factories mentioned above can be obtained from #mi::neuraylib::IMdl_factory.

    In addition, the free functions #mi::neuraylib::get_value() and #mi::neuraylib::set_value() are
    useful to read and write instances of #mi::neuraylib::IValue.

    See \ref mi_neuray_mdl_elements for MDL elements that make use of this type system.
*/

/** \addtogroup mi_neuray_mdl_types
@{
*/

class IAnnotation_block;

/// The interface to MDL types.
///
/// Types can be created using the type factory #mi::neuraylib::IType_factory.
class IType : public
    mi::base::Interface_declare<0x242af675,0xeaa2,0x48b7,0x81,0x63,0xba,0x06,0xa5,0xfb,0x68,0xf0>
{
public:
    /// The possible kinds of types.
    enum Kind {
        /// An alias for another type, aka typedef. See #mi::neuraylib::IType_alias.
        TK_ALIAS,
        /// The \c boolean type. See #mi::neuraylib::IType_bool.
        TK_BOOL,
        /// The \c integer type. See #mi::neuraylib::IType_int.
        TK_INT,
        /// An \c enum type. See #mi::neuraylib::IType_enum.
        TK_ENUM,
        /// The \c float type. See #mi::neuraylib::IType_float.
        TK_FLOAT,
        /// The \c double type. See #mi::neuraylib::IType_double.
        TK_DOUBLE,
        ///  The \c string type. See #mi::neuraylib::IType_string.
        TK_STRING,
        /// A vector type. See #mi::neuraylib::IType_vector.
        TK_VECTOR,
        /// A matrix type. See #mi::neuraylib::IType_matrix.
        TK_MATRIX,
        /// The color type. See #mi::neuraylib::IType_color.
        TK_COLOR,
        /// An array type. See #mi::neuraylib::IType_array.
        TK_ARRAY,
        /// A struct type. See #mi::neuraylib::IType_struct.
        TK_STRUCT,
        /// A texture type. See #mi::neuraylib::IType_texture.
        TK_TEXTURE,
        /// The \c light_profile type. See #mi::neuraylib::IType_light_profile.
        TK_LIGHT_PROFILE,
        /// The \c bsdf_measurement type. See #mi::neuraylib::IType_bsdf_measurement.
        TK_BSDF_MEASUREMENT,
        /// The \c bsdf type. See #mi::neuraylib::IType_bsdf.
        TK_BSDF,
        /// The \c hair_bsdf type. See #mi::neuraylib::IType_hair_bsdf.
        TK_HAIR_BSDF,
        /// The \c edf type. See #mi::neuraylib::IType_edf.
        TK_EDF,
        /// The \c vdf type. See #mi::neuraylib::IType_vdf.
        TK_VDF,
        //  Undocumented, for alignment only.
        TK_FORCE_32_BIT = 0xffffffffU
    };

    /// The possible kinds of type modifiers.
    enum Modifier {
        MK_NONE        = 0,  ///< No type modifier (mutable, auto-typed).
        MK_UNIFORM     = 2,  ///< A uniform type.
        MK_VARYING     = 4,  ///< A varying type.
        MK_FORCE_32_BIT      //   Undocumented, for alignment only.
    };

    /// Returns the kind of type.
    virtual Kind get_kind() const = 0;

    /// Returns all type modifiers of a type.
    ///
    /// Returns 0 if \c this is not an alias. Otherwise, the method follows the chain of aliases by
    /// calling #mi::neuraylib::IType_alias::get_aliased_type() as long as #get_kind() returns
    /// #TK_ALIAS. The method returns the union of
    /// #mi::neuraylib::IType_alias::get_type_modifiers() calls on \c this and all intermediate
    /// aliases.
    virtual Uint32 get_all_type_modifiers() const = 0;

    /// Returns the base type.
    ///
    /// Returns \c this if \c this is not an alias. Otherwise, the method follows the chain of
    /// aliases by calling #mi::neuraylib::IType_alias::get_aliased_type() as long as #get_kind()
    /// returns #TK_ALIAS. The method returns the first non-alias type.
    virtual const IType* skip_all_type_aliases() const = 0;
};

mi_static_assert( sizeof( IType::Kind) == sizeof( Uint32));
mi_static_assert( sizeof( IType::Modifier) == sizeof( Uint32));

/// The type of kind alias.
///
/// Note that types with modifiers are represented solely using alias types, so a \c uniform \c T is
/// an alias of the type \c T (without a name).
class IType_alias : public
    mi::base::Interface_declare<0x69d8c70a,0xdfda,0x4e8e,0xaa,0x09,0x12,0x1f,0xa9,0x78,0xc6,0x6a,
                                neuraylib::IType>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_ALIAS;

    /// Returns the type aliased by this type.
    virtual const IType* get_aliased_type() const = 0;

    /// Returns the modifiers of this type.
    virtual Uint32 get_type_modifiers() const = 0;

    /// Returns the qualified name of the type, or \c NULL if no such name exists.
    virtual const char* get_symbol() const = 0;
};

/// An atomic type.
class IType_atomic : public
    mi::base::Interface_declare<0x9d5f9116,0x3896,0x45c8,0xb4,0x5a,0x8b,0x03,0x84,0x49,0x0a,0x77,
                                neuraylib::IType>
{
};

/// The type of kind bool.
class IType_bool : public
    mi::base::Interface_declare<0x831d8a38,0x26d3,0x4fd2,0xa7,0xf7,0x15,0xc2,0xa5,0x20,0x76,0x6c,
                                neuraylib::IType_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_BOOL;
};

/// The type of kind int.
class IType_int : public
    mi::base::Interface_declare<0xbbad021c,0xbfe5,0x45de,0xaf,0x66,0xfd,0xe8,0x45,0xbe,0x48,0x49,
                                neuraylib::IType_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_INT;
};

/// A type of kind enum.
class IType_enum : public
    mi::base::Interface_declare<0x0e5b167c,0x9c3e,0x48bf,0xb5,0xfd,0x37,0x96,0xaa,0x47,0xaf,0xd1,
                                neuraylib::IType_atomic>
{
public:
    /// IDs to distinguish predefined enum types.
    enum Predefined_id {
        EID_USER           = -1,             ///< A user-defined enum type.
        EID_TEX_GAMMA_MODE =  0,             ///< The \c "::tex::gamma_mode" enum type.
        EID_INTENSITY_MODE =  1,             ///< The \c "::intensity_mode" enum type.
        EID_FORCE_32_BIT   =  0x7fffffff     //   Undocumented, for alignment only.
    };

    /// The kind of this subclass.
    static const Kind s_kind = TK_ENUM;

    /// Returns the qualified name of this enum type.
    virtual const char* get_symbol() const = 0;

    /// Returns the number of values.
    virtual Size get_size() const = 0;

    /// Returns the name of a value.
    ///
    /// \param index         The index of the value.
    /// \return              The unqualified name of the value, or \c NULL if \p index is invalid.
    virtual const char* get_value_name( Size index) const = 0;

    /// Returns the code of a value.
    ///
    /// \param index         The index of the value.
    /// \param[out] errors
    ///                      -  0: Success.
    ///                      - -1: \p index is invalid.
    /// \return              The code of the value, or 0 in case of errors.
    virtual Sint32 get_value_code( Size index, Sint32* errors = 0) const = 0;

    /// Returns the index of a value in linear time.
    ///
    /// \param name          The unqualified name of the value.
    /// \return              The index of the value, or -1 if there is no such value.
    virtual Size find_value( const char* name) const = 0;

    /// Returns the index of a value in linear time.
    ///
    /// \param code          The code of the value.
    /// \return              The index of the value, or -1 if there is no such value.
    virtual Size find_value( Sint32 code) const = 0;

    /// If this enum is a predefined one, return its ID, else EID_USER.
    virtual Predefined_id get_predefined_id() const = 0;

    /// Returns the annotations of the enum type.
    ///
    /// \return              The annotations of the enum type, or \c NULL if there are no
    ///                      annotations for the enum type.
    virtual const IAnnotation_block* get_annotations() const = 0;

    /// Returns the annotations of a value.
    ///
    /// \param index         The index of the value.
    /// \return              The annotation of that value, or \c NULL if \p index is out of bounds,
    ///                      or there are no annotations for that value.
    virtual const IAnnotation_block* get_value_annotations( Size index) const = 0;
};

mi_static_assert( sizeof( IType_enum::Predefined_id) == sizeof( Uint32));

/// The type of kind float.
class IType_float : public
    mi::base::Interface_declare<0x613711b3,0x41f2,0x44a9,0xbb,0x78,0x43,0xe2,0x41,0x64,0xb3,0xda,
                                neuraylib::IType_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_FLOAT;
};

/// The type of kind double.
class IType_double : public
    mi::base::Interface_declare<0xc381508b,0x7945,0x4c70,0x8a,0x20,0x57,0xd5,0x2b,0x36,0x35,0x40,
                                neuraylib::IType_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_DOUBLE;
};

/// The type of kind string.
class IType_string : public
    mi::base::Interface_declare<0x4b4629bc,0xa2ce,0x4008,0xba,0x76,0xf6,0x4d,0x60,0x76,0x0a,0x85,
                                neuraylib::IType_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_STRING;
};

/// A compound type.
class IType_compound : public
    mi::base::Interface_declare<0xc9ca497f,0xc38b,0x411f,0xa8,0x16,0xa7,0xd8,0x23,0x28,0xa5,0x40,
                                neuraylib::IType>
{
public:
    /// Returns the component type at \p index.
    virtual const IType* get_component_type( Size index) const = 0;

    /// Returns the number of components.
    virtual Size get_size() const = 0;
};

/// The type of kind vector.
///
/// The dimension of the vector is given by the size of the underlying compound, see
/// #mi::neuraylib::IType_compound::get_size(). The dimension of a vector is either 2, 3, or 4.
class IType_vector : public
    mi::base::Interface_declare<0x412a8a91,0x9062,0x46fd,0xaa,0xcf,0x46,0xbd,0xb3,0xde,0x5b,0x9c,
                                neuraylib::IType_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_VECTOR;

    /// Returns the type of the vector elements.
    ///
    /// The element type of vectors is either #mi::neuraylib::IType_bool, #mi::neuraylib::IType_int,
    /// #mi::neuraylib::IType_float, or #mi::neuraylib::IType_double. If the vector is a column
    /// vector of a matrix, then the element type is either #mi::neuraylib::IType_float or
    /// #mi::neuraylib::IType_double.
    virtual const IType_atomic* get_element_type() const = 0;
};

/// The type of kind matrix.
///
/// The matrix is represented as a compound of column vectors. The number of matrix columns is given
/// by the size of the underlying compound, see #mi::neuraylib::IType_compound::get_size(). The
/// number of matrix rows is given by the dimension of a column vector. Both dimensions are either
/// 2, 3, or 4.
///
/// \note MDL matrix types are named \c TypeColxRow where \c Type is one of \c float or \c double,
///       \c Col is the number of columns and \c Row is the number of rows (see also section 6.9 in
///       [\ref MDLLS]). This convention is different from the convention used by
///       #mi::math::Matrix.
class IType_matrix : public
    mi::base::Interface_declare<0x6b76570e,0x51b2,0x4e9b,0x9f,0xe7,0xda,0x03,0x1c,0x37,0xbc,0x75,
                                neuraylib::IType_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_MATRIX;

    /// Returns the type of the matrix elements, i.e., the type of a column vector.
    virtual const IType_vector* get_element_type() const = 0;
};

/// The type of kind color.
///
/// The color is represented as a compound of 3 elements of type #mi::neuraylib::IType_float.
class IType_color : public
    mi::base::Interface_declare<0xedb16770,0xdf70,0x4def,0x83,0xa5,0xc4,0x4f,0xcd,0x09,0x47,0x0f,
                                neuraylib::IType_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_COLOR;
};

/// The type of kind array.
class IType_array : public
    mi::base::Interface_declare<0x21ab6abe,0x0e26,0x40da,0xa1,0x98,0x42,0xc0,0x89,0x71,0x5d,0x2a,
                                neuraylib::IType_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_ARRAY;

    /// Returns the type of the array elements.
    virtual const IType* get_element_type() const = 0;

    /// Indicates whether the array is immediate-sized or deferred-sized.
    virtual bool is_immediate_sized() const = 0;

    /// Returns the size of the array in case of immediate-sized arrays, and -1 otherwise.
    virtual Size get_size() const = 0;

    /// Returns the abstract size of the array in case of deferred-sized arrays, and \c NULL
    /// otherwise.
    ///
    /// Note that the empty string is a valid return value for deferred-sized arrays.
    virtual const char* get_deferred_size() const = 0;
};

/// The type of kind struct.
class IType_struct : public
    mi::base::Interface_declare<0x19566cb2,0x0b5d,0x41ca,0xa0,0x31,0x96,0xe2,0x9a,0xd4,0xc3,0x1a,
                                neuraylib::IType_compound>
{
public:
    /// IDs to distinguish predefined struct types.
    enum Predefined_id {
        SID_USER               = -1,             ///< A user-defined struct type.
        SID_MATERIAL_EMISSION  =  0,             ///< The \c "::material_emission" struct type.
        SID_MATERIAL_SURFACE   =  1,             ///< The \c "::material_surface" struct type.
        SID_MATERIAL_VOLUME    =  2,             ///< The \c "::material_volume" struct type.
        SID_MATERIAL_GEOMETRY  =  3,             ///< The \c "::material_geometry" struct type.
        SID_MATERIAL           =  4,             ///< The \c "::material" struct type.
        SID_FORCE_32_BIT       =  0x7fffffff     //   Undocumented, for alignment only.
    };

    /// The kind of this subclass.
    static const Kind s_kind = TK_STRUCT;

    /// Returns the qualified name of the struct type.
    virtual const char* get_symbol() const = 0;

    /// Returns a field type.
    ///
    /// \param index    The index of the field.
    /// \return         The type of the field.
    virtual const IType* get_field_type( Size index) const = 0;

    /// Returns a field name.
    ///
    /// \param index    The index of the field.
    /// \return         The unqualified name of the field.
    virtual const char* get_field_name( Size index) const = 0;

    /// Returns the index of a field in linear time.
    ///
    /// \param name     The unqualified name of the field.
    /// \return         The index of the field, or -1 if there is no such field.
    virtual Size find_field( const char* name) const = 0;

    /// If this struct is a predefined one, return its ID, else SID_USER.
    virtual Predefined_id get_predefined_id() const = 0;

    /// Returns the annotations of the struct type.
    ///
    /// \return              The annotations of the struct type, or \c NULL if there are no
    ///                      annotations for the struct type.
    virtual const IAnnotation_block* get_annotations() const = 0;

    /// Returns the annotations of a field.
    ///
    /// \param index         The index of the field.
    /// \return              The annotation of that field, or \c NULL if \p index is out of bounds,
    ///                      or there are no annotations for that field.
    virtual const IAnnotation_block* get_field_annotations( Size index) const = 0;
};

mi_static_assert( sizeof( IType_struct::Predefined_id) == sizeof( Uint32));

/// The reference types.
class IType_reference : public
    mi::base::Interface_declare<0x3e12cdec,0xdaba,0x460c,0x9e,0x8a,0x21,0x4c,0x43,0x9a,0x1a,0x90,
                                neuraylib::IType>
{
};

/// A string valued resource type.
class IType_resource : public
    mi::base::Interface_declare<0x142f5bea,0x139e,0x42e4,0xb1,0x1c,0xb3,0x4d,0xd8,0xe3,0xd9,0x8d,
                                neuraylib::IType_reference>
{
};

/// The type of kind texture.
class IType_texture : public
    mi::base::Interface_declare<0x2f11253f,0xb8ac,0x4b7d,0x8d,0xd6,0x43,0x66,0xf5,0x97,0xd0,0x93,
                                neuraylib::IType_resource>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_TEXTURE;

    /// The possible texture shapes.
    enum Shape {
        TS_2D           = 0,            ///< Two-dimensional texture.
        TS_3D           = 1,            ///< Three-dimensional texture.
        TS_CUBE         = 2,            ///< Cube map texture.
        TS_PTEX         = 3,            ///< PTEX texture.
        TS_BSDF_DATA    = 4,            ///< Three-dimensional texture representing
                                        ///  a BSDF data table.
        TS_FORCE_32_BIT = 0xffffffffU   //   Undocumented, for alignment only.
    };

    /// Returns the texture type.
    virtual Shape get_shape() const = 0;
};

mi_static_assert( sizeof( IType_texture::Shape) == sizeof( Uint32));

/// The type of kind light_profile.
class IType_light_profile : public
    mi::base::Interface_declare<0x11b80cd8,0x14aa,0x4dfa,0x8b,0xf6,0x0e,0x56,0x0f,0x10,0x9c,0x37,
                                neuraylib::IType_resource>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_LIGHT_PROFILE;
};

/// The type of kind bsdf_measurement.
class IType_bsdf_measurement : public
    mi::base::Interface_declare<0xf061d204,0xc649,0x4a6b,0xb6,0x2d,0x67,0xe6,0x47,0x53,0xa9,0xda,
                                neuraylib::IType_resource>
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = TK_BSDF_MEASUREMENT;
};

/// The type of distribution functions.
class IType_df : public
    mi::base::Interface_declare<0xf4bcba08,0x7777,0x4662,0x8e,0x29,0x67,0xe1,0x52,0xac,0x05,0x3e,
                                neuraylib::IType_reference>
{
};

/// The type of kind bsdf.
class IType_bsdf : public
    mi::base::Interface_declare<0x6542a02c,0xe1d2,0x485d,0x9a,0x51,0x7b,0xed,0xff,0x7f,0x24,0x7b,
                                neuraylib::IType_df>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_BSDF;
};

/// The type of kind bsdf.
class IType_hair_bsdf : public
    mi::base::Interface_declare<0x8eac6c90,0x2b8f,0x4650,0x8b,0x93,0x88,0xe0,0x42,0xff,0x19,0x9c,
    neuraylib::IType_df>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_HAIR_BSDF;
};

/// The type of kind edf.
class IType_edf : public
    mi::base::Interface_declare<0x3e3ce697,0xa2a7,0x43ef,0xa2,0xec,0x52,0x5a,0x4c,0x27,0x8f,0xeb,
                                neuraylib::IType_df>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_EDF;
};

/// The type of kind vdf.
class IType_vdf : public
    mi::base::Interface_declare<0x44782b21,0x9e60,0x40b2,0xba,0xae,0x41,0x74,0xc9,0x98,0xe1,0x86,
                                neuraylib::IType_df>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = TK_VDF;
};

/// An ordered collection of types identified by name or index.
///
/// Type lists can be created with #mi::neuraylib::IType_factory::create_type_list().
class IType_list : public
    mi::base::Interface_declare<0x68a97390,0x22ea,0x4f03,0xa5,0xb5,0x5c,0x18,0x32,0x38,0x28,0x91>
{
public:
    /// Returns the number of elements.
    virtual Size get_size() const = 0;

    /// Returns the index for the given name, or -1 if there is no such type.
    virtual Size get_index( const char* name) const = 0;

    /// Returns the name for the given index, or \c NULL if there is no such type.
    virtual const char* get_name( Size index) const = 0;

    /// Returns the type for \p index, or \c NULL if there is no such type.
    virtual const IType* get_type( Size index) const = 0;

    /// Returns the type for \p index, or \c NULL if there is no such type.
    template <class T>
    const T* get_type( Size index) const
    {
        const IType* ptr_type = get_type( index);
        if( !ptr_type)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
        ptr_type->release();
        return ptr_T;
    }

    /// Returns the type for \p name, or \c NULL if there is no such type.
    virtual const IType* get_type( const char* name) const = 0;

    /// Returns the type for \p name, or \c NULL if there is no such type.
    template <class T>
    const T* get_type( const char* name) const
    {
        const IType* ptr_type = get_type( name);
        if( !ptr_type)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
        ptr_type->release();
        return ptr_T;
    }

    /// Sets a type at a given index.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameters (\c NULL pointer).
    ///           - -2: \p index is out of bounds.
    virtual Sint32 set_type( Size index, const IType* type) = 0;

    /// Sets a type identified by name.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameters (\c NULL pointer).
    ///           - -2: There is no type mapped to \p name in the list.
    virtual Sint32 set_type( const char* name, const IType* type) = 0;

    /// Adds a type at the end of the list.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameters (\c NULL pointer).
    ///           - -2: There is already a type mapped to \p name in the list.
    virtual Sint32 add_type( const char* name, const IType* type) = 0;
};

/// The interface for creating types.
///
/// A type factory can be obtained from #mi::neuraylib::IMdl_factory::create_type_factory().
class IType_factory : public
    mi::base::Interface_declare<0x353803c0,0x74a6,0x48ac,0xab,0xa1,0xe4,0x25,0x42,0x1d,0xa1,0xbc>
{
public:
    /// Creates a new instance of the type alias.
    virtual const IType_alias* create_alias(
        const IType* type, Uint32 modifiers, const char* symbol) const = 0;

    /// Creates a new instance of the type boolean.
    virtual const IType_bool* create_bool() const = 0;

    /// Creates a new instance of the type int.
    virtual const IType_int* create_int() const = 0;

    /// Returns a registered enum type, or \c NULL if \p symbol is invalid or unknown.
    virtual const IType_enum* create_enum( const char* symbol) const = 0;

    /// Creates a new instance of the float type.
    virtual const IType_float* create_float() const = 0;

    /// Creates a new instance of the double type.
    virtual const IType_double* create_double() const = 0;

    /// Creates a new instance of the string type.
    virtual const IType_string* create_string() const = 0;

    /// Creates a new instance of a vector type.
    ///
    /// \param element_type   The element type needs to be either #mi::neuraylib::IType_bool,
    ///                       #mi::neuraylib::IType_int, #mi::neuraylib::IType_float, or
    ///                       #mi::neuraylib::IType_double.
    /// \param size           The number of elements, either 2, 3, or 4.
    /// \return               The corresponding vector type, or \c NULL in case of errors.
    virtual const IType_vector* create_vector(
        const IType_atomic* element_type, Size size) const = 0;

    /// Creates a new instance of a matrix type.
    ///
    /// \param column_type    The column type needs to be a vector of either
    ///                       #mi::neuraylib::IType_float or #mi::neuraylib::IType_double.
    /// \param columns        The number of columns, either 2, 3, or 4.
    /// \return               The corresponding matrix type, or \c NULL in case of errors.
    virtual const IType_matrix* create_matrix(
        const IType_vector* column_type, Size columns) const = 0;

    /// Creates a new instance of the type color.
    virtual const IType_color* create_color() const = 0;

    /// Creates a new instance of an immediate-sized array type.
    virtual const IType_array* create_immediate_sized_array(
        const IType* element_type, Size size) const = 0;

    /// Creates a new instance of a deferred-sized array type.
    virtual const IType_array* create_deferred_sized_array(
        const IType* element_type, const char* size) const = 0;

    /// Returns a registered struct type, or \c NULL if \p symbol is invalid or unknown.
    virtual const IType_struct* create_struct( const char* symbol) const = 0;

    /// Creates a new instance of the type texture.
    virtual const IType_texture* create_texture( IType_texture::Shape shape) const = 0;

    /// Creates a new instance of the type light_profile.
    virtual const IType_light_profile* create_light_profile() const = 0;

    /// Creates a new instance of the type bsdf_measurement.
    virtual const IType_bsdf_measurement* create_bsdf_measurement() const = 0;

    /// Creates a new instance of the type bsdf.
    virtual const IType_bsdf* create_bsdf() const = 0;

    /// Creates a new instance of the type hair_bsdf.
    virtual const IType_hair_bsdf* create_hair_bsdf() const = 0;

    /// Creates a new instance of the type edf.
    virtual const IType_edf* create_edf() const = 0;

    /// Creates a new instance of the type vdf.
    virtual const IType_vdf* create_vdf() const = 0;

    /// Creates a new type map.
    virtual IType_list* create_type_list() const = 0;

    /// Returns a registered enum type, or \c NULL if \p id is unknown.
    virtual const IType_enum* get_predefined_enum( IType_enum::Predefined_id id) const = 0;

    /// Returns a registered struct type, or \c NULL if \p id is unknown.
    virtual const IType_struct* get_predefined_struct( IType_struct::Predefined_id id) const = 0;

    /// Compares two instances of #mi::neuraylib::IType.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IType is defined as follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Otherwise, the kind of the types are compared. If they are different, the result is
    ///   determined by \c operator< on the #mi::neuraylib::IType::Kind values.
    /// - Finally, specific types are compared as follows:
    ///   - #mi::neuraylib::IType_enum and #mi::neuraylib::IType_struct: The result is determined by
    ///     \c strcmp() on the corresponding symbol names.
    ///   - #mi::neuraylib::IType_vector, #mi::neuraylib::IType_matrix, #mi::neuraylib::IType_array:
    ///     If the element types are different, they determine the result of the comparison. If the
    ///     element types are identical the number of compound elements determines the result.
    ///   - #mi::neuraylib::IType_alias: If the modifiers are different, they determine the result
    ///     of the comparison. If the modifiers are identical, the aliased types determine the
    ///     result.
    ///   - #mi::neuraylib::IType_texture: The result is determined by a comparison of the
    ///     corresponding shapes.
    ///   - All other pairs of (the same kind of) types are considered equal.
    ///
    /// \param lhs   The left-hand side operand for the comparison.
    /// \param rhs   The right-hand side operand for the comparison.
    /// \return      -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare( const IType* lhs, const IType* rhs) const = 0;

    /// Compares two instances of #mi::neuraylib::IType_list.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IType_list is defined as follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Next, the list sizes are compared using \c operator<().
    /// - Next, the lists are traversed by increasing index and the names are compared using
    ///   \c strcmp().
    /// - Finally, the list elements are enumerated by increasing index and the types are compared.
    ///
    /// \param lhs   The left-hand side operand for the comparison.
    /// \param rhs   The right-hand side operand for the comparison.
    /// \return      -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare( const IType_list* lhs, const IType_list* rhs) const = 0;

    /// Checks, if two instances of #mi::neuraylib::IType are compatible, meaning that \p src
    /// can be casted to \p dst.
    /// 
    /// \p src is compatible with and therefore can be casted to \p dst, if
    /// - \p src and \p dst are of identical type (see #mi::neuraylib::IType_factory::compare()).
    /// - \p src and \p dst are of type #mi::neuraylib::IType_struct, have the same number of
    ///   fields and all fields are compatible.
    /// - \p src and \p dst are of type #mi::neuraylib::IType_enum and both enumeration types have
    ///   the same set of numerical enumeration values. The name of the enumeration values, their
    ///   order, or whether multiple enumeration value names share the same numerical value
    ///   do not matter.
    /// - \p src and \p dst are of type #mi::neuraylib::IType_array, both arrays have the same size
    ///   and their element types are compatible.
    /// 
    /// \param src The source type.
    /// \param dst the target type to which src is intended to be compatible.
    /// \return
    ///           -  0 if \p src can be casted to \p dst, but \p src and \p dst are not of identical
    ///                type.
    ///           -  1 if \p src and \p dst are of identical type.
    ///           - -1 if \p src cannot be casted to \p dst.
    virtual Sint32 is_compatible(const IType* src, const IType* dst) const = 0;

    /// Returns a textual representation of a type.
    ///
    /// The representation of the type might contain line breaks, for example for structures and
    /// enums. Subsequent lines have a suitable indentation. The assumed indentation level of the
    /// first line is specified by \p depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump( const IType* type, Size depth = 0) const = 0;

    /// Returns a textual representation of a type list.
    ///
    /// The representation of the type list will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump( const IType_list* list, Size depth = 0) const = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ITYPE_H
