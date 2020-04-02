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
/// \brief      Values of the MDL type system

#ifndef MI_NEURAYLIB_IVALUE_H
#define MI_NEURAYLIB_IVALUE_H

#include <mi/base/handle.h>
#include <mi/math/color.h>
#include <mi/math/matrix.h>
#include <mi/math/spectrum.h>
#include <mi/math/vector.h>
#include <mi/neuraylib/itype.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_types
@{
*/

/// The interface to MDL values.
///
/// Values can be created using the value factory #mi::neuraylib::IValue_factory.
class IValue : public
    mi::base::Interface_declare<0xbf837f4a,0x9034,0x4f32,0xaf,0x5c,0x75,0xb3,0x67,0x64,0x53,0x23>
{
public:
    /// The possible kinds of values.
    enum Kind {
        /// A boolean value. See #mi::neuraylib::IValue_bool.
        VK_BOOL,
        /// An integer value. See #mi::neuraylib::IValue_int.
        VK_INT,
        /// An enum value. See #mi::neuraylib::IValue_enum.
        VK_ENUM,
        /// A float value. See #mi::neuraylib::IValue_float.
        VK_FLOAT,
        /// A double value. See #mi::neuraylib::IValue_double.
        VK_DOUBLE,
        /// A string value. See #mi::neuraylib::IValue_string.
        VK_STRING,
        /// A vector value. See #mi::neuraylib::IValue_vector.
        VK_VECTOR,
        /// A matrix value. See #mi::neuraylib::IValue_matrix.
        VK_MATRIX,
        /// A color value. See #mi::neuraylib::IValue_color.
        VK_COLOR,
        /// An array value. See #mi::neuraylib::IValue_array.
        VK_ARRAY,
        /// A struct value. See #mi::neuraylib::IValue_struct.
        VK_STRUCT,
        /// An invalid distribution function value. See #mi::neuraylib::IValue_invalid_df.
        VK_INVALID_DF,
        /// A texture value. See #mi::neuraylib::IValue_texture.
        VK_TEXTURE,
        /// A light_profile value. See #mi::neuraylib::IValue_light_profile.
        VK_LIGHT_PROFILE,
        /// A bsdf_measurement value. See #mi::neuraylib::IValue_bsdf_measurement.
        VK_BSDF_MEASUREMENT,
        //  Undocumented, for alignment only.
        VK_FORCE_32_BIT = 0xffffffffU
    };

    /// Returns the kind of the value.
    virtual Kind get_kind() const = 0;

    /// Returns the type of this value.
    virtual const IType* get_type() const = 0;

    /// Returns the type of this value.
    template <class T>
    const T* get_type() const
    {
        const IType* ptr_type = get_type();
        if( !ptr_type)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
        ptr_type->release();
        return ptr_T;
    }
};

mi_static_assert( sizeof( IValue::Kind) == sizeof( Uint32));

/// An atomic value.
class IValue_atomic : public
    mi::base::Interface_declare<0xf2413c80,0x8e71,0x4974,0xaa,0xf2,0x60,0xd5,0xe2,0x94,0x9d,0x3e,
                                neuraylib::IValue>
{
public:
    /// Returns the type of this value.
    const IType_atomic* get_type() const = 0;
};

/// A value of type boolean.
class IValue_bool : public
    mi::base::Interface_declare<0xaf253a14,0x1f04,0x4b67,0xba,0x70,0x7b,0x01,0x05,0xfb,0xc8,0xf5,
                                neuraylib::IValue_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_BOOL;

    /// Returns the type of this value.
    const IType_bool* get_type() const = 0;

    /// Returns the value.
    virtual bool get_value() const = 0;

    /// Sets the value.
    virtual void set_value( bool value) = 0;
};

/// A value of type integer.
class IValue_int : public
    mi::base::Interface_declare<0x91e6f145,0x280d,0x4d68,0x95,0x57,0xe1,0xd0,0x9c,0xd2,0x5c,0x74,
                                neuraylib::IValue_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_INT;

    /// Returns the type of this value.
    const IType_int* get_type() const = 0;

    /// Returns the value.
    virtual Sint32 get_value() const = 0;

    /// Sets the value.
    virtual void set_value( Sint32 value) = 0;
};

/// A value of type enum.
class IValue_enum : public
    mi::base::Interface_declare<0xdc876204,0x8a97,0x40e9,0xb9,0xb6,0xca,0xdc,0xdd,0x60,0x1f,0xbf,
                                neuraylib::IValue_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_ENUM;

    /// Returns the type of this value.
    virtual const IType_enum* get_type() const = 0;

    /// Returns the (integer) value of this enum value.
    virtual Sint32 get_value() const = 0;

    /// Returns the index of this enum value.
    virtual Size get_index() const = 0;

    /// Sets the enum value by integer in linear time.
    ///
    /// If there are multiple indices with the same value the one with the smallest index is chosen.
    ///
    /// \return   0 in case of success, -1 if \p value is not valid for this enum type
    virtual Sint32 set_value( Sint32 value) = 0;

    /// Sets the enum value by index.
    ///
    /// \return   0 in case of success, -1 if \p index is not valid for this enum type
    virtual Sint32 set_index( Size index) = 0;

    /// Returns the string representation of this enum value.
    virtual const char* get_name() const = 0;

    /// Sets the enum value by string representation in linear time.
    ///
    /// \return   0 in case of success, -1 if \p name is not valid for this enum type
    virtual Sint32 set_name( const char* name) = 0;
};

/// A value of type float.
class IValue_float : public
    mi::base::Interface_declare<0x21f07151,0x74b5,0x4296,0x90,0x29,0xc7,0xde,0x49,0x38,0x2a,0xbc,
                                neuraylib::IValue_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_FLOAT;

    /// Returns the type of this value.
    const IType_float* get_type() const = 0;

    /// Returns the value.
    virtual Float32 get_value() const = 0;

    /// Sets the value.
    virtual void set_value( Float32 value) = 0;
};

/// A value of type double.
class IValue_double : public
    mi::base::Interface_declare<0xbdc84417,0x3e83,0x4bab,0x90,0xb1,0x9f,0x57,0xed,0x7b,0x15,0x03,
                                neuraylib::IValue_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_DOUBLE;

    /// Returns the type of this value.
    const IType_double* get_type() const = 0;

    /// Returns the value.
    virtual Float64 get_value() const = 0;

    /// Sets the value.
    virtual void set_value( Float64 value) = 0;
};

/// A value of type string.
class IValue_string : public
    mi::base::Interface_declare<0x64b28506,0x8675,0x4724,0xa1,0x0d,0xc6,0xf2,0x35,0x46,0x26,0x39,
                                neuraylib::IValue_atomic>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_STRING;

    /// Returns the type of this value.
    const IType_string* get_type() const = 0;

    /// Returns the value.
    virtual const char* get_value() const = 0;

    /// Sets the value.
    virtual void set_value( const char* value) = 0;
};

/// A value of type string which can be used to obtain the original, non-localized value of a localized string.
class IValue_string_localized : public
    mi::base::Interface_declare<0x1fe80d3d, 0xe79e, 0x4bdb, 0xb6, 0x30, 0xe3, 0x36, 0x31, 0xa4, 0x1e, 0x39,
    neuraylib::IValue_string>
{
public:
    /// Returns the original value of a localized string.
    /// While IValue_string::value() returns the translated string.
    virtual const char* get_original_value() const = 0;
};

/// A compound value.
class IValue_compound : public
    mi::base::Interface_declare<0xdabc8fe3,0x5c70,0x4ef0,0xa2,0xf7,0x34,0x30,0xb5,0x67,0xdc,0x75,
                                neuraylib::IValue>
{
public:
    /// Returns the type of this value.
    virtual const IType_compound* get_type() const = 0;

    /// Returns the number of components in this compound value.
    virtual Size get_size() const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual const IValue* get_value( Size index) const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    template <class T>
    const T* get_value( Size index) const
    {
        const IValue* ptr_value = get_value( index);
        if( !ptr_value)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual IValue* get_value( Size index) = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    template <class T>
    T* get_value( Size index)
    {
        IValue* ptr_value = get_value( index);
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Sets the value at \p index.
    ///
    /// \param index   The index of the field.
    /// \param value   The new value of the field.
    /// \return
    ///                -  0: Success.
    ///                - -1: Invalid parameter (\p NULL pointer).
    ///                - -2: \p index is out of bounds.
    ///                - -3: Incorrect type of \p value.
    virtual Sint32 set_value( Size index, IValue* value) = 0;
};

/// A value of type vector.
///
/// The \c get_value() methods from #mi::neuraylib::IValue_compound are duplicated here due to
/// their covariant return type. See the parent interface for methods to set values.
class IValue_vector : public
    mi::base::Interface_declare<0xf5d09fc3,0xd783,0x4571,0x8d,0x59,0x41,0xb1,0xff,0xd3,0x91,0x49,
                                neuraylib::IValue_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_VECTOR;

    /// Returns the type of this value.
    virtual const IType_vector* get_type() const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual const IValue_atomic* get_value( Size index) const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    template <class T>
    const T* get_value( Size index) const
    {
        const IValue_atomic* ptr_value = get_value( index);
        if( !ptr_value)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual IValue_atomic* get_value( Size index) = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    template <class T>
    T* get_value( Size index)
    {
        IValue_atomic* ptr_value = get_value( index);
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }
};

/// A value of type matrix.
///
/// The \c %get_value() methods from #mi::neuraylib::IValue_compound are duplicated here due to
/// their covariant return type. See the parent interface for methods to set values.
class IValue_matrix : public
    mi::base::Interface_declare<0x9ee95da6,0x2cd6,0x4168,0x89,0xea,0x92,0x10,0x57,0xda,0xe6,0xdc,
                                neuraylib::IValue_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_MATRIX;

    /// Returns the type of this value.
    virtual const IType_matrix* get_type() const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual const IValue_vector* get_value( Size index) const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual IValue_vector* get_value( Size index) = 0;
};

/// A value of type color.
class IValue_color : public
    mi::base::Interface_declare<0x3bb9bf46,0x1cbb,0x4460,0xbe,0x27,0x10,0xf5,0x71,0x61,0x96,0xa2,
                                neuraylib::IValue_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_COLOR;

    /// Returns the type of this value.
    virtual const IType_color* get_type() const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual const IValue_float* get_value( Size index) const = 0;

    /// Returns the value at \p index, or \c NULL if \p index is out of bounds.
    virtual IValue_float* get_value( Size index) = 0;

    /// Sets the value at \p index.
    ///
    /// \return
    ///                -  0: Success.
    ///                - -1: Invalid parameter (\c NULL pointer).
    ///                - -2: \p index is out of bounds.
    virtual Sint32 set_value( Size index, IValue_float* value) = 0;

    using IValue_compound::set_value;
};

/// A value of type array.
class IValue_array : public
    mi::base::Interface_declare<0xa17c5f57,0xa647,0x41c4,0x86,0x2f,0x4c,0x0d,0xe1,0x30,0x08,0xfc,
                                neuraylib::IValue_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_ARRAY;

    /// Returns the type of this value.
    virtual const IType_array* get_type() const = 0;

    /// Sets the size for dynamic arrays.
    ///
    /// \param size   The desired array size.
    /// \return
    ///               -  0: Success.
    ///               - -1: The array is a static array.
    virtual Sint32 set_size( Size size) = 0;
};

/// A value of type struct.
class IValue_struct : public
    mi::base::Interface_declare<0xcbe089ce,0x4aea,0x474d,0x94,0x5f,0x52,0x13,0xef,0x01,0xce,0x81,
                                neuraylib::IValue_compound>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_STRUCT;

    /// Returns the type of this value.
    virtual const IType_struct* get_type() const = 0;

    /// Returns a field by name in linear time.
    ///
    /// \param name    The name of the field.
    /// \return        The value of the field, or \c NULL if there is no such field.
    virtual const IValue* get_field( const char* name) const = 0;

    /// Returns a field by name in linear time.
    ///
    /// \param name    The name of the field.
    /// \return        The value of the field, or \c NULL if there is no such field.
    template <class T>
    const T* get_field( const char* name) const
    {
        const IValue* ptr_value = get_field( name);
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Returns a field by name in linear time.
    ///
    /// \param name    The name of the field.
    /// \return        The value of the field, or \c NULL if there is no such field.
    virtual IValue* get_field( const char* name) = 0;

    /// Returns a field by name in linear time.
    ///
    /// \param name    The name of the field.
    /// \return        The value of the field, or \c NULL if there is no such field.
    template <class T>
    T* get_field( const char* name)
    {
        IValue* ptr_value = get_field( name);
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Sets a field by name in linear time.
    ///
    /// \param name    The name of the field.
    /// \param value   The new value of the field.
    /// \return
    ///                -  0: Success.
    ///                - -1: Invalid parameter (\c NULL pointer).
    ///                - -2: There is no such field of the given name.
    ///                - -3: Incorrect type of \p value.
    virtual Sint32 set_field( const char* name, IValue* value) = 0;
};

/// Base class for resource values.
class IValue_resource : public
    mi::base::Interface_declare<0x479bb10c,0xd444,0x426c,0x83,0xab,0x26,0xdf,0xf6,0x1d,0x6f,0xd7,
                                neuraylib::IValue>
{
public:
    /// Returns the type of this value.
    virtual const IType_resource* get_type() const = 0;

    /// Returns the name of the DB element representing this resource.
    ///
    /// \return        The name of the DB element, or \c NULL if no valid resource is set.
    virtual const char* get_value() const = 0;

    /// Sets the name of the DB element representing this resource.
    ///
    /// Pointing this instance to a different DB element resets the MDL file path returned by
    /// #get_file_path().
    ///
    /// \param value   The name of the resource, or \c NULL to release the current resource.
    /// \return
    ///                -  0: Success.
    ///                - -1: There is no DB element with that name.
    ///                - -2: The DB element has not the correct type for this resource.
    virtual Sint32 set_value( const char* value) = 0;

    /// Returns the absolute MDL file path of the resource, or \c NULL if not known.
    ///
    /// \note The value returned here is not a property of this object, but a property of the
    ///       referenced resource.
    virtual const char* get_file_path() const = 0;
};

/// A texture value.
class IValue_texture : public
    mi::base::Interface_declare<0xf2a03651,0x8883,0x4ba4,0xb9,0xa9,0xe6,0x87,0x34,0x3a,0xb3,0xb8,
                                neuraylib::IValue_resource>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_TEXTURE;

    /// Returns the type of this value.
    virtual const IType_texture* get_type() const = 0;

    /// Returns the gamma value of this texture.
    ///
    /// \note: A gamma value of 0 corresponds to the default gamma value for the given texture
    ///        kind.
    virtual Float32 get_gamma() const = 0;
};

/// A light profile value.
class IValue_light_profile : public
    mi::base::Interface_declare<0xd7c9ffbd,0xb5e4,0x4bf4,0x90,0xd0,0xe9,0x75,0x4d,0x6d,0x49,0x07,
                                neuraylib::IValue_resource>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_LIGHT_PROFILE;

    /// Returns the type of this value.
    virtual const IType_light_profile* get_type() const = 0;
};

/// A BSDF measurement value.
class IValue_bsdf_measurement : public
    mi::base::Interface_declare<0x31a55244,0x415c,0x4b4d,0xa7,0x86,0x2f,0x21,0x9c,0xb8,0xb9,0xff,
                                neuraylib::IValue_resource>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_BSDF_MEASUREMENT;

    /// Returns the type of this value.
    virtual const IType_bsdf_measurement* get_type() const = 0;
};

/// An invalid distribution function value.
class IValue_invalid_df : public
    mi::base::Interface_declare<0x1588b6fa,0xa143,0x4bac,0xa0,0x32,0x06,0xbd,0x9e,0x7f,0xb6,0xe5,
                                neuraylib::IValue>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = VK_INVALID_DF;

    /// Returns the type of this value.
    const IType_reference* get_type() const = 0;
};

/// An ordered collection of values identified by name or index.
///
/// Value lists can be created with #mi::neuraylib::IValue_factory::create_value_list().
class IValue_list : public
    mi::base::Interface_declare<0x8027a5e5,0x4e25,0x410c,0xbb,0xce,0x84,0xb4,0x88,0x8b,0x03,0x46>
{
public:
    /// Returns the number of elements.
    virtual Size get_size() const = 0;

    /// Returns the index for the given name, or -1 if there is no such value.
    virtual Size get_index( const char* name) const = 0;

    /// Returns the name for the given index, or \c NULL if there is no such value.
    virtual const char* get_name( Size index) const = 0;

    /// Returns the value for \p index, or \c NULL if there is no such value.
    virtual const IValue* get_value( Size index) const = 0;

    /// Returns the value for \p index, or \c NULL if there is no such value.
    template <class T>
    const T* get_value( Size index) const
    {
        const IValue* ptr_value = get_value( index);
        if( !ptr_value)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Returns the value for \p name, or \c NULL if there is no such value.
    virtual const IValue* get_value( const char* name) const = 0;

    /// Returns the value for \p name, or \c NULL if there is no such value.
    template <class T>
    const T* get_value( const char* name) const
    {
        const IValue* ptr_value = get_value( name);
        if( !ptr_value)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Sets a value at a given index.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: \p index is out of bounds.
    virtual Sint32 set_value( Size index, const IValue* value) = 0;

    /// Sets a value identified by name.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: There is no value mapped to \p name in the list.
    virtual Sint32 set_value( const char* name, const IValue* value) = 0;

    /// Adds a value at the end of the list.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: There is already a value mapped to \p name in the list.
    virtual Sint32 add_value( const char* name, const IValue* value) = 0;
};

/// The interface for creating values.
///
/// A value factory can be obtained from #mi::neuraylib::IMdl_factory::create_value_factory().
class IValue_factory : public
    mi::base::Interface_declare<0x82595c0d,0x3687,0x4b45,0xa3,0x38,0x42,0x20,0x02,0xea,0x3f,0x9b>
{
public:

    /// Returns the type factory associated with this value factory.
    virtual IType_factory* get_type_factory() const = 0;

    /// Creates a new value of type boolean.
    virtual IValue_bool* create_bool( bool value = false) const = 0;

    /// Creates a new value of type integer.
    virtual IValue_int* create_int( Sint32 value = 0) const = 0;

    /// Creates a new value of type enum, or returns \c NULL in case of errors.
    virtual IValue_enum* create_enum( const IType_enum* type, Size index = 0) const = 0;

    /// Creates a new value of type float.
    virtual IValue_float* create_float( Float32 value = 0.0f) const = 0;

    /// Creates a new value of type double.
    virtual IValue_double* create_double( Float64 value = 0.0) const = 0;

    /// Creates a new value of type string.
    ///
    /// \param value   The value \c NULL is handled like the empty string.
    virtual IValue_string* create_string( const char* value = "") const = 0;

    /// Creates a new value of type vector, or returns \c NULL in case of errors.
    virtual IValue_vector* create_vector( const IType_vector* type) const = 0;

    /// Creates a new value of type matrix, or returns \c NULL in case of errors.
    virtual IValue_matrix* create_matrix( const IType_matrix* type) const = 0;

    /// Creates a new value of type color.
    virtual IValue_color* create_color(
        Float32 red = 0.0f,
        Float32 green = 0.0f,
        Float32 blue = 0.0f) const = 0;

    /// Creates a new value of type array, or returns \c NULL in case of errors.
    virtual IValue_array* create_array( const IType_array* type) const = 0;

    /// Creates a new value of type struct, or returns \c NULL in case of errors.
    virtual IValue_struct* create_struct( const IType_struct* type) const = 0;

    /// Creates a new texture value, or returns \c NULL in case of errors.
    virtual IValue_texture* create_texture( const IType_texture* type, const char* value) const = 0;

    /// Creates a new light profile value, or returns \c NULL in case of errors.
    virtual IValue_light_profile* create_light_profile( const char* value) const = 0;

    /// Creates a new BSDF measurement value, or returns \c NULL in case of errors.
    virtual IValue_bsdf_measurement* create_bsdf_measurement( const char* value) const = 0;

    /// Creates a new invalid distribution function value.
    virtual IValue_invalid_df* create_invalid_df( const IType_reference* type) const = 0;

    /// Creates a default-constructed value of the given type.
    virtual IValue* create( const IType* type) const = 0;

    /// Creates a default-constructed value of the given type.
    template <class T>
    T* create( const IType* type) const
    {
        IValue* ptr_value = create( type);
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Creates a new value list.
    virtual IValue_list* create_value_list() const = 0;

    /// Clones the given value.
    ///
    /// Note that referenced DB elements, e.g., resources, are not copied, but shared.
    virtual IValue* clone( const IValue* value) const = 0;

    /// Clones the given value.
    ///
    /// Note that referenced DB elements, e.g., resources, are not copied, but shared.
    template <class T>
    T* clone( const T* value) const
    {
        IValue* ptr_value = clone( static_cast<const IValue*>( value));
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Clones the given value list.
    ///
    /// Note that referenced DB elements, e.g., resources, are not copied, but shared.
    virtual IValue_list* clone( const IValue_list* value_list) const = 0;

    /// Compares two instances of #mi::neuraylib::IValue.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IValue is defined as follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Otherwise, the types of \p lhs and \p rhs are compared. If they are different, the result
    ///   is determined by that comparison.
    /// - Next, the kind of the values are compared. If they are different, the result is determined
    ///   by \c operator< on the #mi::neuraylib::IValue::Kind values.
    /// - Finally, the values are compared as follows:
    ///   - For atomic types, their values are compared using \c operator< or \c strcmp(), with the
    ///     exception of enums, for which the indices rather than the values are compared.
    ///   - For compounds, the compound size is compared using \c operator< (the compound size might
    ///     be different for dynamic arrays). If both compounds are of equal size, the compounds
    ///     elements are compared in lexicographic order.
    ///   - For resources, the values are compared using \c strcmp().
    ///
    /// \param lhs          The left-hand side operand for the comparison.
    /// \param rhs          The right-hand side operand for the comparison.
    /// \return             -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare( const IValue* lhs, const IValue* rhs) const = 0;

    /// Compares two instances of #mi::neuraylib::IValue_list.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IValue_list is defined as follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Next, the list sizes are compared using \c operator<().
    /// - Next, the lists are traversed by increasing index and the names are compared
    ///   using \c strcmp().
    /// - Finally, the list elements are enumerated by increasing index and the values are compared.
    ///
    /// \param lhs          The left-hand side operand for the comparison.
    /// \param rhs          The right-hand side operand for the comparison.
    /// \return             -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare( const IValue_list* lhs, const IValue_list* rhs) const = 0;

    /// Returns a textual representation of a value.
    ///
    /// The textual representation is of the form "type name = value" if \p name is not \c NULL, and
    /// of the form "value" if \p name is \c NULL. The representation of the value might contain
    /// line breaks, for example for structures, enums, and arrays. Subsequent lines have a suitable
    /// indentation. The assumed indentation level of the first line is specified by \p depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump( const IValue* value, const char* name, Size depth = 0) const = 0;

    /// Returns a textual representation of a value list.
    ///
    /// The representation of the value list will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump(
        const IValue_list* list, const char* name, Size depth = 0) const = 0;
};

/// Simplifies setting the value of #mi::neuraylib::IValue from the corresponding classes from the
/// %base and %math API.
///
/// \param value           The instance of #mi::neuraylib::IValue to modify.
/// \param v               The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p value does not match the static type of
///                              \p v.
///                        - -2: The value of v is not valid.
///
/// This general template handles #mi::neuraylib::IValue_int and #mi::neuraylib::IValue_enum and
/// expects #mi::Uint32 as second argument. Since it is a template it will handle other types as
/// second argument if they are accepted in place of parameters of type #mi::Uint32, e.g.,
/// #mi::Sint32.
template<class T>
mi::Sint32 set_value( mi::neuraylib::IValue* value, const T& v)
{
    mi::base::Handle<mi::neuraylib::IValue_int> value_int(
        value->get_interface<mi::neuraylib::IValue_int>());
    if( value_int) {
        value_int->set_value( v);
        return 0;
    }

    mi::base::Handle<mi::neuraylib::IValue_enum> value_enum(
        value->get_interface<mi::neuraylib::IValue_enum>());
    if( value_enum) {
        if( value_enum->set_value( v) != 0)
            return -2;
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_bool.
///
/// It expects a \c bool as second argument. See #mi::neuraylib::set_value() for details.
inline mi::Sint32 set_value( mi::neuraylib::IValue* value, const bool& v)
{
    mi::base::Handle<mi::neuraylib::IValue_bool> value_bool(
        value->get_interface<mi::neuraylib::IValue_bool>());
    if( value_bool) {
        value_bool->set_value( v);
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_float and #mi::neuraylib::IValue_double.
///
/// It expects an #mi::Float32 as second argument. See #mi::neuraylib::set_value() for details.
inline mi::Sint32 set_value( mi::neuraylib::IValue* value, const mi::Float32& v)
{
    mi::base::Handle<mi::neuraylib::IValue_float> value_float(
        value->get_interface<mi::neuraylib::IValue_float>());
    if( value_float) {
        value_float->set_value( v);
        return 0;
    }

    mi::base::Handle<mi::neuraylib::IValue_double> value_double(
        value->get_interface<mi::neuraylib::IValue_double>());
    if( value_double) {
        value_double->set_value( v);
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_float and #mi::neuraylib::IValue_double.
///
/// It expects an #mi::Float64 as second argument. See #mi::neuraylib::set_value() for details.
inline mi::Sint32 set_value( mi::neuraylib::IValue* value, const mi::Float64& v)
{
    mi::base::Handle<mi::neuraylib::IValue_float> value_float(
        value->get_interface<mi::neuraylib::IValue_float>());
    if( value_float) {
        value_float->set_value( static_cast<mi::Float32>( v));
        return 0;
    }

    mi::base::Handle<mi::neuraylib::IValue_double> value_double(
        value->get_interface<mi::neuraylib::IValue_double>());
    if( value_double) {
        value_double->set_value( v);
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_enum, #mi::neuraylib::IValue_string and
/// #mi::neuraylib::IValue_resource.
///
/// It expects a \c const \c char* as second argument. See #mi::neuraylib::set_value() for details.
inline mi::Sint32 set_value( mi::neuraylib::IValue* value, const char* v)
{
    mi::base::Handle<mi::neuraylib::IValue_enum> value_enum(
        value->get_interface<mi::neuraylib::IValue_enum>());
    if( value_enum) {
        if( value_enum->set_name( v) != 0)
            return -2;
        return 0;
    }

    mi::base::Handle<mi::neuraylib::IValue_string> value_string(
        value->get_interface<mi::neuraylib::IValue_string>());
    if( value_string) {
        value_string->set_value( v);
        return 0;
    }

    mi::base::Handle<mi::neuraylib::IValue_resource> value_resource(
        value->get_interface<mi::neuraylib::IValue_resource>());
    if( value_resource) {
        if( value_resource->set_value( v) != 0)
            return -2;
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_vector.
///
/// It expects an #mi::math::Vector of matching dimension and element type as second argument. See
/// #mi::neuraylib::set_value() for details.
template<class T, Size DIM>
mi::Sint32 set_value( mi::neuraylib::IValue* value, const mi::math::Vector<T,DIM>& v)
{
    mi::base::Handle<mi::neuraylib::IValue_vector> value_vector(
        value->get_interface<mi::neuraylib::IValue_vector>());
    if( value_vector) {
        if( value_vector->get_size() != DIM)
            return -1;
        for( Size  i = 0; i < DIM; ++i) {
            mi::base::Handle<mi::neuraylib::IValue> component( value_vector->get_value( i));
            mi::Sint32 result = set_value( component.get(), v[i]);
            if( result != 0)
                return result;
        }
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_matrix.
///
/// It expects an #mi::math::Matrix of matching dimensions and element type as second argument. See
/// #mi::neuraylib::set_value() for details.
///
/// \note The conversion between #mi::neuraylib::IValue_matrix and #mi::math::Matrix is supposed to
///       preserve the memory layout. Since #mi::neuraylib::IValue_matrix uses a column-major layout
///       and #mi::math::Matrix uses a row-major layout, the conversion process effectively
///       transposes the matrix.
template<class T, Size ROW, Size COL>
mi::Sint32 set_value( mi::neuraylib::IValue* value, const mi::math::Matrix<T,ROW,COL>& v)
{
    mi::base::Handle<mi::neuraylib::IValue_matrix> value_matrix(
        value->get_interface<mi::neuraylib::IValue_matrix>());
    if( value_matrix) {
        if( value_matrix->get_size() != ROW)
            return -1;
        for( Size  i = 0; i < ROW; ++i) {
            mi::base::Handle<mi::neuraylib::IValue_vector> column( value_matrix->get_value( i));
            if( column->get_size() != COL)
                return -1;
            for( Size j = 0; j < COL; ++j) {
                mi::base::Handle<mi::neuraylib::IValue> element( column->get_value( j));
                mi::Sint32 result = set_value( element.get(), v[i][j]);
                if( result != 0)
                    return result;
            }
        }
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_color.
///
/// It expects an #mi::math::Color as second argument. See #mi::neuraylib::set_value() for details.
inline mi::Sint32 set_value( mi::neuraylib::IValue* value, const mi::math::Color& v)
{
    mi::base::Handle<mi::neuraylib::IValue_color> value_color(
        value->get_interface<mi::neuraylib::IValue_color>());
    if( value_color) {
        mi::base::Handle<mi::neuraylib::IValue_float> red  ( value_color->get_value( 0));
        red  ->set_value( v.r);
        mi::base::Handle<mi::neuraylib::IValue_float> green( value_color->get_value( 1));
        green->set_value( v.g);
        mi::base::Handle<mi::neuraylib::IValue_float> blue ( value_color->get_value( 2));
        blue ->set_value( v.b);
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_color.
///
/// It expects an #mi::math::Spectrum as second argument. See #mi::neuraylib::set_value() for
/// details.
inline mi::Sint32 set_value( mi::neuraylib::IValue* value, const mi::math::Spectrum& v)
{
    mi::base::Handle<mi::neuraylib::IValue_color> value_color(
        value->get_interface<mi::neuraylib::IValue_color>());
    if( value_color) {
        mi::base::Handle<mi::neuraylib::IValue_float> red  ( value_color->get_value( 0));
        red  ->set_value( v[0]);
        mi::base::Handle<mi::neuraylib::IValue_float> green( value_color->get_value( 1));
        green->set_value( v[1]);
        mi::base::Handle<mi::neuraylib::IValue_float> blue ( value_color->get_value( 2));
        blue ->set_value( v[2]);
        return 0;
    }

    return -1;
}

/// This variant handles elements of compounds identified via an additional index.
///
/// \param value           The instance of #mi::neuraylib::IValue to modify.
/// \param index           The index of the affected compound element.
/// \param v               The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p value does not match the static type of
///                              \p v.
///                        - -3: The index is not valid.
template<class T>
mi::Sint32 set_value( mi::neuraylib::IValue* value, mi::Size index, const T& v)
{
    mi::base::Handle<mi::neuraylib::IValue_compound> value_compound(
        value->get_interface<mi::neuraylib::IValue_compound>());
    if( value_compound) {
        mi::base::Handle<mi::neuraylib::IValue> component( value_compound->get_value( index));
        if( !component)
            return -3;
        return set_value( component.get(), v);
    }

    return -1;
}

/// This variant handles fields of structs identified via an additional field name.
///
/// \param value           The instance of #mi::neuraylib::IValue to modify.
/// \param name            The name of the affected struct field.
/// \param v               The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p value does not match the static type of
///                              \p v.
///                        - -3: The field name is not valid.
template<class T>
mi::Sint32 set_value( mi::neuraylib::IValue* value, const char* name, const T& v)
{
    mi::base::Handle<mi::neuraylib::IValue_struct> value_struct(
        value->get_interface<mi::neuraylib::IValue_struct>());
    if( value_struct) {
        mi::base::Handle<mi::neuraylib::IValue> field( value_struct->get_field( name));
        if( !field)
            return -3;
        return set_value( field.get(), v);
    }

    return -1;
}

// Simplifies reading the value of #mi::neuraylib::IValue into the corresponding classes from the
/// %base and %math API.
///
/// \param value           The instance of #mi::neuraylib::IValue to read.
/// \param[out] v          The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p value does not match the static type of
///                              \p v.
///
/// This general template handles #mi::neuraylib::IValue_int and #mi::neuraylib::IValue_enum and
/// expects #mi::Uint32 as second argument. Since it is a template it will handle other types as
/// second argument if they are accepted in place of parameters of type #mi::Uint32, e.g.,
/// #mi::Sint32.
template<class T>
mi::Sint32 get_value( const mi::neuraylib::IValue* value, T& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_int> value_int(
        value->get_interface<mi::neuraylib::IValue_int>());
    if( value_int) {
        v = value_int->get_value();
        return 0;
    }

    mi::base::Handle<const mi::neuraylib::IValue_enum> value_enum(
        value->get_interface<mi::neuraylib::IValue_enum>());
    if( value_enum) {
        v = value_enum->get_value();
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_bool.
///
/// It expects a \c bool as second argument. See #mi::neuraylib::get_value() for details.
inline mi::Sint32 get_value( const mi::neuraylib::IValue* value, bool& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_bool> value_bool(
        value->get_interface<mi::neuraylib::IValue_bool>());
    if( value_bool) {
        v = value_bool->get_value();
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_float and #mi::neuraylib::IValue_double.
///
/// It expects an #mi::Float32 as second argument. See #mi::neuraylib::get_value() for details.
inline mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::Float32& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_float> value_float(
        value->get_interface<mi::neuraylib::IValue_float>());
    if( value_float) {
        v = value_float->get_value();
        return 0;
    }

    mi::base::Handle<const mi::neuraylib::IValue_double> value_double(
        value->get_interface<mi::neuraylib::IValue_double>());
    if( value_double) {
        v = static_cast<mi::Float32>( value_double->get_value());
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_float and #mi::neuraylib::IValue_double.
///
/// It expects an #mi::Float64 as second argument. See #mi::neuraylib::get_value() for details.
inline mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::Float64& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_float> value_float(
        value->get_interface<mi::neuraylib::IValue_float>());
    if( value_float) {
        v = value_float->get_value();
        return 0;
    }

    mi::base::Handle<const mi::neuraylib::IValue_double> value_double(
        value->get_interface<mi::neuraylib::IValue_double>());
    if( value_double) {
        v = value_double->get_value();
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_enum, #mi::neuraylib::IValue_string and
/// #mi::neuraylib::IValue_resource.
///
/// It expects a \c const \c char* as second argument. See #mi::neuraylib::get_value() for details.
inline mi::Sint32 get_value( const mi::neuraylib::IValue* value, const char*& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_enum> value_enum(
        value->get_interface<mi::neuraylib::IValue_enum>());
    if( value_enum) {
        v = value_enum->get_name();
        return 0;
    }

    mi::base::Handle<const mi::neuraylib::IValue_string> value_string(
        value->get_interface<mi::neuraylib::IValue_string>());
    if( value_string) {
        v = value_string->get_value();
        return 0;
    }

    mi::base::Handle<const mi::neuraylib::IValue_resource> value_resource(
        value->get_interface<mi::neuraylib::IValue_resource>());
    if( value_resource) {
        v = value_resource->get_value();
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_vector.
///
/// It expects an #mi::math::Vector of matching dimension and element type as second argument. See
/// #mi::neuraylib::get_value() for details.
template <class T, Size DIM>
mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::math::Vector<T,DIM>& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_vector> value_vector(
        value->get_interface<mi::neuraylib::IValue_vector>());
    if( value_vector) {
        if( value_vector->get_size() != DIM)
            return -1;
        for( Size i = 0; i < DIM; ++i) {
            mi::base::Handle<const mi::neuraylib::IValue> component( value_vector->get_value( i));
            mi::Sint32 result = get_value( component.get(), v[i]);
            if( result != 0)
                return result;
        }
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_matrix.
///
/// It expects an #mi::math::Matrix of matching dimensions and element type as second argument. See
/// #mi::neuraylib::get_value() for details.
///
/// \note The conversion between #mi::neuraylib::IValue_matrix and #mi::math::Matrix is supposed to
///       preserve the memory layout. Since #mi::neuraylib::IValue_matrix uses a column-major layout
///       and #mi::math::Matrix uses a row-major layout, the conversion process effectively
///       transposes the matrix.
template <class T, Size ROW, Size COL>
mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::math::Matrix<T,ROW,COL>& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_matrix> value_matrix(
        value->get_interface<mi::neuraylib::IValue_matrix>());
    if( value_matrix) {
        if( value_matrix->get_size() != ROW)
            return -1;
        for( Size i = 0; i < ROW; ++i) {
            mi::base::Handle<const mi::neuraylib::IValue_vector> column(
                value_matrix->get_value( i));
            if( column->get_size() != COL)
                return -1;
            for( Size j = 0; j < COL; ++j) {
                mi::base::Handle<const mi::neuraylib::IValue> element( column->get_value( j));
                mi::Sint32 result = get_value( element.get(), v[i][j]);
                if( result != 0)
                    return result;
            }
        }
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_color.
///
/// It expects an #mi::math::Color as second argument. See #mi::neuraylib::get_value() for details.
inline mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::math::Color& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_color> value_color(
        value->get_interface<mi::neuraylib::IValue_color>());
    if( value_color) {
        mi::base::Handle<const mi::neuraylib::IValue_float> red  ( value_color->get_value( 0));
        v.r = red  ->get_value();
        mi::base::Handle<const mi::neuraylib::IValue_float> green( value_color->get_value( 1));
        v.g = green->get_value();
        mi::base::Handle<const mi::neuraylib::IValue_float> blue ( value_color->get_value( 2));
        v.b = blue ->get_value();
        v.a = 1.0f;
        return 0;
    }

    return -1;
}

/// This specialization handles #mi::neuraylib::IValue_color.
///
/// It expects an #mi::math::Spectrum as second argument. See #mi::neuraylib::get_value() for
/// details.
inline mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::math::Spectrum& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_color> value_color(
        value->get_interface<mi::neuraylib::IValue_color>());
    if( value_color) {
        mi::base::Handle<const mi::neuraylib::IValue_float> red  ( value_color->get_value( 0));
        v[0] = red  ->get_value();
        mi::base::Handle<const mi::neuraylib::IValue_float> green( value_color->get_value( 1));
        v[1] = green->get_value();
        mi::base::Handle<const mi::neuraylib::IValue_float> blue ( value_color->get_value( 2));
        v[2] = blue ->get_value();
        return 0;
    }

    return -1;
}

/// This variant handles elements of compounds identified via an additional index.
///
/// \param value           The instance of #mi::neuraylib::IValue to read.
/// \param index           The index of the affected compound element.
/// \param v               The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p value does not match the static type of
///                              \p v.
///                        - -3: The index is not valid.
template<class T>
mi::Sint32 get_value( const mi::neuraylib::IValue* value, mi::Size index, T& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_compound> value_compound(
        value->get_interface<mi::neuraylib::IValue_compound>());
    if( value_compound) {
        mi::base::Handle<const mi::neuraylib::IValue> component( value_compound->get_value( index));
        if( !component)
            return -3;
        return get_value( component.get(), v);
    }

    return -1;
}

/// This variant handles fields of structs identified via an additional field name.
///
/// \param value           The instance of #mi::neuraylib::IValue to read.
/// \param name            The name of the affected struct field.
/// \param v               The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p value does not match the static type of
///                              \p v.
///                        - -3: The field name is not valid.
template<class T>
mi::Sint32 get_value( const mi::neuraylib::IValue* value, const char* name, T& v)
{
    mi::base::Handle<const mi::neuraylib::IValue_struct> value_struct(
        value->get_interface<mi::neuraylib::IValue_struct>());
    if( value_struct) {
        mi::base::Handle<const mi::neuraylib::IValue> field( value_struct->get_field( name));
        if( !field)
            return -3;
        return get_value( field.get(), v);
    }

    return -1;
}

/*@}*/ // end group mi_neuray_mdl_types

}  // namespace neuraylib

}  // namespace mi

#endif // MI_NEURAYLIB_IVALUE_H
