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
///
///             For documentation, see the counterparts in the public API
///             in <mi/neuraylib/ivalue.h>.

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_VALUE_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_VALUE_H

#include <base/data/db/i_db_tag.h>

#include "i_mdl_elements_type.h"

namespace MI {

namespace DB { class Transaction; }
namespace SERIAL { class Deserializer; class Serializer; }

namespace MDL {

class IValue : public
    mi::base::Interface_declare<0x5e629c36,0xe7e7,0x4bbc,0x87,0x86,0xbc,0x24,0x54,0xc6,0xf8,0x40>
{
public:
    enum Kind {
        VK_BOOL,
        VK_INT,
        VK_ENUM,
        VK_FLOAT,
        VK_DOUBLE,
        VK_STRING,
        VK_VECTOR,
        VK_MATRIX,
        VK_COLOR,
        VK_ARRAY,
        VK_STRUCT,
        VK_INVALID_DF,
        VK_TEXTURE,
        VK_LIGHT_PROFILE,
        VK_BSDF_MEASUREMENT,
        VK_FORCE_32_BIT = 0xffffffffU
    };

    virtual Kind get_kind() const = 0;

    virtual const IType* get_type() const = 0;

    template <class T>
    const T* get_type() const
    {
        mi::base::Handle<const IType> ptr_type( get_type());
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual mi::Size get_memory_consumption() const = 0;
};

mi_static_assert( sizeof( IValue::Kind) == sizeof( mi::Uint32));

class IValue_atomic : public
    mi::base::Interface_declare<0x3630b70f,0xf2d3,0x4fbf,0x9c,0xaa,0x3d,0xed,0xd6,0x0d,0xf7,0xa5,
                                IValue>
{
public:
    const IType_atomic* get_type() const = 0;
};

class IValue_bool : public
    mi::base::Interface_declare<0x3ef3f4a5,0x26c2,0x49c5,0xa8,0xfc,0x94,0xb2,0x8f,0x9f,0x64,0xf2,
                                IValue_atomic>
{
public:
    static const Kind s_kind = VK_BOOL;

    const IType_bool* get_type() const = 0;

    virtual bool get_value() const = 0;

    virtual void set_value( bool value) = 0;
};

class IValue_int : public
    mi::base::Interface_declare<0x22e48f0b,0xa951,0x4559,0xbf,0x30,0xa4,0xa9,0x38,0x9e,0xd5,0xce,
                                IValue_atomic>
{
public:
    static const Kind s_kind = VK_INT;

    const IType_int* get_type() const = 0;

    virtual mi::Sint32 get_value() const = 0;

    virtual void set_value( mi::Sint32 value) = 0;
};

class IValue_enum : public
    mi::base::Interface_declare<0xa2aaa8f4,0x6ecf,0x4d58,0xa6,0xc8,0x1d,0x9e,0xe4,0x06,0xe6,0xfa,
                                IValue_atomic>
{
public:
    static const Kind s_kind = VK_ENUM;

    virtual const IType_enum* get_type() const = 0;

    virtual mi::Sint32 get_value() const = 0;

    virtual const char* get_name() const = 0;

    virtual mi::Size get_index() const = 0;

    virtual mi::Sint32 set_value( mi::Sint32 value) = 0;

    virtual mi::Sint32 set_index( mi::Size index) = 0;

    virtual mi::Sint32 set_name( const char* name) = 0;
};

class IValue_float : public
    mi::base::Interface_declare<0xcd2ba0b1,0xe6ab,0x4620,0x9f,0xa6,0xd1,0xb2,0x9e,0x9a,0x59,0x72,
                                IValue_atomic>
{
public:
    static const Kind s_kind = VK_FLOAT;

    const IType_float* get_type() const = 0;

    virtual mi::Float32 get_value() const = 0;

    virtual void set_value( mi::Float32 value) = 0;
};

class IValue_double : public
    mi::base::Interface_declare<0xa53dc381,0x1391,0x4d3f,0xbc,0x1e,0x8d,0x89,0x92,0x14,0x64,0x8e,
                                IValue_atomic>
{
public:
    static const Kind s_kind = VK_DOUBLE;

    const IType_double* get_type() const = 0;

    virtual mi::Float64 get_value() const = 0;

    virtual void set_value( mi::Float64 value) = 0;
};

class IValue_string : public
    mi::base::Interface_declare<0x0e28cde1,0x9bc7,0x4f4b,0xbc,0xae,0xdd,0xc6,0x96,0x9b,0xb6,0x6c,
                                IValue_atomic>
{
public:
    static const Kind s_kind = VK_STRING;

    const IType_string* get_type() const = 0;

    virtual const char* get_value() const = 0;

    virtual void set_value( const char* value) = 0;
};

class IValue_string_localized : public
    mi::base::Interface_declare<0x9f699d83, 0xe6be, 0x41f9, 0xbe, 0x76, 0xef, 0x95, 0x55, 0x1e, 0xbe, 0xdb,
    IValue_string>
{
public:
    virtual const char* get_original_value() const = 0;

    virtual void set_original_value( const char* value) = 0;
};

class IValue_compound : public
    mi::base::Interface_declare<0xce8262f5,0x37c2,0x4472,0xb9,0xde,0x22,0x9e,0x42,0x6b,0x48,0x8f,
                                IValue>
{
public:
    virtual const IType_compound* get_type() const = 0;

    virtual mi::Size get_size() const = 0;

    virtual const IValue* get_value( mi::Size index) const = 0;

    template <class T>
    const T* get_value( mi::Size index) const
    {
        mi::base::Handle<const IValue> ptr_value( get_value( index));
        if( !ptr_value)
            return nullptr;
        return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual IValue* get_value( mi::Size index) = 0;

    template <class T>
    T* get_value( mi::Size index)
    {
        mi::base::Handle<IValue> ptr_value( get_value( index));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual mi::Sint32 set_value( mi::Size index, IValue* value) = 0;
};

class IValue_vector : public
    mi::base::Interface_declare<0xa17adbd0,0x03e1,0x4d99,0xb8,0xcb,0x94,0x20,0xa9,0x42,0x76,0x4b,
                                IValue_compound>
{
public:
    static const Kind s_kind = VK_VECTOR;

    virtual const IType_vector* get_type() const = 0;

    virtual const IValue_atomic* get_value( mi::Size index) const = 0;

    virtual IValue_atomic* get_value( mi::Size index) = 0;

    template <class T>
    const T* get_value( mi::Size index) const
    {
        mi::base::Handle<const IValue_atomic> ptr_value( get_value( index));
        if( !ptr_value)
            return nullptr;
        return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
    }

    template <class T>
    T* get_value( mi::Size index)
    {
        mi::base::Handle<IValue_atomic> ptr_value( get_value( index));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }
};

class IValue_matrix : public
    mi::base::Interface_declare<0x02cea143,0xc8fd,0x4422,0x98,0x9e,0x12,0xb6,0x8f,0xcf,0xd8,0x21,
                                IValue_compound>
{
public:
    static const Kind s_kind = VK_MATRIX;

    virtual const IType_matrix* get_type() const = 0;

    virtual const IValue_vector* get_value( mi::Size index) const = 0;

    virtual IValue_vector* get_value( mi::Size index) = 0;
};

class IValue_color : public
    mi::base::Interface_declare<0xd6f9034c,0x3144,0x4c01,0xb3,0x61,0x49,0x9a,0x3a,0x74,0xc0,0xe1,
                                IValue_compound>
{
public:
    static const Kind s_kind = VK_COLOR;

    virtual const IType_color* get_type() const = 0;

    virtual const IValue_float* get_value( mi::Size index) const = 0;

    virtual IValue_float* get_value( mi::Size index) = 0;

    virtual mi::Sint32 set_value( mi::Size index, IValue_float* value) = 0;

    using IValue_compound::set_value;
};

class IValue_array : public
    mi::base::Interface_declare<0x6fd32516,0x4a03,0x4fdb,0x9f,0x21,0x2d,0xe2,0x17,0x71,0x0e,0x95,
                                IValue_compound>
{
public:
    static const Kind s_kind = VK_ARRAY;

    virtual const IType_array* get_type() const = 0;

    virtual mi::Sint32 set_size( mi::Size size) = 0;
};

class IValue_struct : public
    mi::base::Interface_declare<0x05db0790,0x9f91,0x4eaf,0xb6,0x5f,0x23,0xa7,0xfd,0x30,0x6a,0x2f,
                                IValue_compound>
{
public:
    static const Kind s_kind = VK_STRUCT;

    virtual const IType_struct* get_type() const = 0;

    virtual const IValue* get_field( const char* name) const = 0;

    template <class T>
    const T* get_field( const char* name) const
    {
        mi::base::Handle<const IValue> ptr_value( get_field( name));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual IValue* get_field( const char* name) = 0;

    template <class T>
    T* get_field( const char* name)
    {
        mi::base::Handle<IValue> ptr_value( get_field( name));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual mi::Sint32 set_field( const char* name, IValue* value) = 0;
};

class IValue_invalid_df : public
    mi::base::Interface_declare<0x1615fd17,0xee05,0x4218,0x93,0xfe,0x24,0xc0,0x20,0x1e,0x3f,0xf8,
                                  IValue>
{
public:
    static const Kind s_kind = VK_INVALID_DF;

    const IType_reference* get_type() const = 0;
};

class IValue_resource : public
    mi::base::Interface_declare<0x21171930,0x519c,0x4a80,0xb6,0x76,0xb0,0xa0,0x8a,0xfe,0x20,0x9d,
                                IValue>
{
public:
    virtual const IType_resource* get_type() const = 0;

    virtual DB::Tag get_value() const = 0;

    virtual void set_value( DB::Tag value) = 0;

    virtual const char* get_unresolved_mdl_url() const = 0;

    virtual void set_unresolved_mdl_url(const char *url) = 0;

    virtual const char *get_owner_module() const = 0;

    virtual void set_owner_module(const char *module) = 0;

    virtual const char* get_file_path( DB::Transaction* transaction) const = 0;
};

class IValue_texture : public
    mi::base::Interface_declare<0xff46b645,0x0d00,0x4e38,0x82,0x1d,0xa8,0xf6,0x63,0x41,0xf0,0x15,
                                IValue_resource>
{
public:
    static const Kind s_kind = VK_TEXTURE;

    virtual const IType_texture* get_type() const = 0;

    virtual Float32 get_gamma() const = 0;

    virtual void set_gamma(mi::Float32 gamma) = 0;

};

class IValue_light_profile : public
    mi::base::Interface_declare<0x3b49aaee,0xef75,0x44fe,0xaa,0x57,0xb6,0x29,0x92,0x61,0x61,0x8a,
                                IValue_resource>
{
public:
    static const Kind s_kind = VK_LIGHT_PROFILE;

    virtual const IType_light_profile* get_type() const = 0;
};

class IValue_bsdf_measurement : public
    mi::base::Interface_declare<0x854d4dd2,0xd08f,0x4d18,0x91,0xcc,0x37,0x94,0x50,0x1a,0x67,0xb2,
                                IValue_resource>
{
public:
    static const Kind s_kind = VK_BSDF_MEASUREMENT;

    virtual const IType_bsdf_measurement* get_type() const = 0;
};

class IValue_list : public
    mi::base::Interface_declare<0x001251cb,0x41b3,0x4e34,0xb2,0xa5,0x78,0x80,0x12,0x97,0x78,0x42>
{
public:
    virtual mi::Size get_size() const = 0;

    virtual mi::Size get_index( const char* name) const = 0;

    virtual const char* get_name( mi::Size index) const = 0;

    virtual const IValue* get_value( mi::Size index) const = 0;

    template <class T>
    const T* get_value( mi::Size index) const
    {
        mi::base::Handle<const IValue> ptr_value( get_value( index));
        if( !ptr_value)
            return nullptr;
        return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual const IValue* get_value( const char* name) const = 0;

    template <class T>
    const T* get_value( const char* name) const
    {
        mi::base::Handle<const IValue> ptr_value( get_value( name));
        if( !ptr_value)
            return nullptr;
        return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual mi::Sint32 set_value( mi::Size index, const IValue* value) = 0;

    virtual mi::Sint32 set_value( const char* name, const IValue* value) = 0;

    virtual mi::Sint32 add_value( const char* name, const IValue* value) = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

class IValue_factory : public
    mi::base::Interface_declare<0x99053c52,0xdda5,0x4edb,0x84,0xbd,0xd3,0x7a,0xda,0xd5,0xa3,0xb1>
{
public:

    virtual IType_factory* get_type_factory() const = 0;

    virtual IValue_bool* create_bool( bool value = false) const = 0;

    virtual IValue_int* create_int( mi::Sint32 value = 0) const = 0;

    virtual IValue_enum* create_enum( const IType_enum* type, mi::Size index = 0) const = 0;

    virtual IValue_float* create_float( mi::Float32 value = 0.0f) const = 0;

    virtual IValue_double* create_double( mi::Float64 value = 0.0) const = 0;

    virtual IValue_string* create_string( const char* value = "") const = 0;

    virtual IValue_string_localized* create_string_localized( const char* value = "", const char* original_value = "") const = 0;

    virtual IValue_vector* create_vector( const IType_vector* type) const = 0;

    virtual IValue_matrix* create_matrix( const IType_matrix* type) const = 0;

    virtual IValue_color* create_color(
        mi::Float32 red = 0.0f,
        mi::Float32 green = 0.0f,
        mi::Float32 blue = 0.0f) const = 0;

    virtual IValue_array* create_array( const IType_array* type) const = 0;

    virtual IValue_struct* create_struct( const IType_struct* type) const = 0;

    virtual IValue_texture* create_texture(
        const IType_texture* type,
        DB::Tag value) const = 0;

    virtual IValue_texture* create_texture(
        const IType_texture* type,
        DB::Tag value,
        const char *unresolved_mdl_url,
        const char *owner_module,
        mi::Float32 gamma) const = 0;

    virtual IValue_light_profile* create_light_profile( DB::Tag value) const = 0;

    virtual IValue_light_profile* create_light_profile(
        DB::Tag value,
        const char *unresolved_mdl_url,
        const char *owner_module) const = 0;

    virtual IValue_bsdf_measurement* create_bsdf_measurement( DB::Tag value) const = 0;

    virtual IValue_bsdf_measurement* create_bsdf_measurement(
        DB::Tag value,
        const char *unresolved_mdl_url,
        const char *owner_module) const = 0;

    virtual IValue_invalid_df* create_invalid_df( const IType_reference* type) const = 0;

    virtual IValue* create( const IType* type) const = 0;

    template <class T>
    T* create( const IType* type) const
    {
        mi::base::Handle<IValue> ptr_value( create( type));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual IValue_list* create_value_list() const = 0;

    virtual IValue* clone( const IValue* value) const = 0;

    template <class T>
    T* clone( const T* value) const
    {
        mi::base::Handle<IValue> ptr_value( clone( static_cast<const IValue*>( value)));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual IValue_list* clone( const IValue_list* list) const = 0;

    virtual mi::Sint32 compare( const IValue* lhs, const IValue* rhs) const = 0;

    virtual mi::Sint32 compare( const IValue_list* lhs, const IValue_list* rhs) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IValue* value,
        const char* name,
        mi::Size depth = 0) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IValue_list* list,
        const char* name,
        mi::Size depth = 0) const = 0;

    /// Serializes a value to \p serializer.
    virtual void serialize( SERIAL::Serializer* serializer, const IValue* value) const = 0;

    /// Deserializes a value from \p deserializer.
    virtual IValue* deserialize( SERIAL::Deserializer* deserializer) const = 0;

    /// Deserializes a value from \p deserializer.
    template <class T>
    T* deserialize( SERIAL::Deserializer* deserializer) const
    {
        mi::base::Handle<IValue> ptr_value( deserialize( deserializer));
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    /// Serializes a value list to \p serializer.
    virtual void serialize_list(
        SERIAL::Serializer* serializer, const IValue_list* list) const = 0;

    /// Deserializes a value list from \p deserializer.
    virtual IValue_list* deserialize_list( SERIAL::Deserializer* deserializer) const = 0;
};

/// Returns the global value factory.
IValue_factory* get_value_factory();

// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption( const IValue*) { return true; }
inline size_t dynamic_memory_consumption( const IValue* v)
{ return v->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption( const IValue_list*) { return true; }
inline size_t dynamic_memory_consumption( const IValue_list* l)
{ return l->get_memory_consumption(); }

}  // namespace MDL

}  // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_VALUE_H
