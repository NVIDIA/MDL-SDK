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
/// \brief      Header for the IValue hierarchy and IValue_factory implementation.

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_VALUE_IMPL_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_VALUE_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include "i_mdl_elements_value.h"

#include <map>
#include <string>
#include <vector>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace MDL {

class Type_factory;

template <class V, class T>
class Value_base : public mi::base::Interface_implement<V>
{
public:
    typedef V Value;
    typedef T Type;
    typedef Value_base<V, T> Base;

    Value_base( const T* type) : m_type( type, mi::base::DUP_INTERFACE) { }

    IValue::Kind get_kind() const { return V::s_kind; };

    const T* get_type() const { m_type->retain(); return m_type.get(); }

    using V::get_type;

protected:
    const mi::base::Handle<const T> m_type;
};


class Value_bool : public Value_base<IValue_bool, IType_bool>
{
public:
    Value_bool( const Type* type, bool value) : Base( type), m_value( value) { }

    bool get_value() const { return m_value; }

    void set_value( bool value) { m_value = value; }

    mi::Size get_memory_consumption() const;

private:
    bool m_value;
};


class Value_int : public Value_base<IValue_int, IType_int>
{
public:
    Value_int( const Type* type, mi::Sint32 value) : Base( type), m_value( value) { }

    mi::Sint32 get_value() const { return m_value; }

    void set_value( mi::Sint32 value) { m_value = value; }

    mi::Size get_memory_consumption() const;

private:
    mi::Sint32 m_value;
};


class Value_enum : public Value_base<IValue_enum, IType_enum>
{
public:
    Value_enum( const Type* type, mi::Size index)
      : Base( type), m_index( index) { ASSERT( M_SCENE, index < m_type->get_size()); }

    mi::Sint32 get_value() const { return m_type->get_value_code( m_index); }

    const char* get_name() const { return m_type->get_value_name( m_index); }

    mi::Size get_index() const { return m_index; }

    mi::Sint32 set_value( mi::Sint32 value);

    mi::Sint32 set_name( const char* name);

    mi::Sint32 set_index( mi::Size index);

    mi::Size get_memory_consumption() const;

private:
    mi::Size m_index;
};


class Value_float : public Value_base<IValue_float, IType_float>
{
public:
    Value_float( const Type* type, mi::Float32 value) : Base( type), m_value( value) { }

    mi::Float32 get_value() const { return m_value; }

    void set_value( mi::Float32 value) { m_value = value; }

    mi::Size get_memory_consumption() const;

private:
    mi::Float32 m_value;
};


class Value_double : public Value_base<IValue_double, IType_double>
{
public:
    Value_double( const Type* type, mi::Float64 value) : Base( type), m_value( value) { }

    mi::Float64 get_value() const { return m_value; }

    void set_value( mi::Float64 value) { m_value = value; }

    mi::Size get_memory_consumption() const;

private:
    mi::Float64 m_value;
};


class Value_string : public Value_base<IValue_string, IType_string>
{
public:
    Value_string( const Type* type, const char* value) : Base( type), m_value( value) { }

    const char* get_value() const { return m_value.c_str(); }

    void set_value( const char* value) { m_value = value ? value : ""; }

    mi::Size get_memory_consumption() const;

private:
    std::string m_value;
};


class Value_string_localized : public Value_base<IValue_string_localized, IType_string>
{
public:
    Value_string_localized( const Type* type, const char* value, const char* original_value) : Base( type), m_value( value), m_original_value( original_value) { }

    const char* get_value() const { return m_value.c_str(); }

    void set_value( const char* value) { m_value = value ? value : ""; }

    const char* get_original_value() const { return m_original_value.c_str(); }

    void set_original_value( const char* value) { m_original_value = value ? value : ""; }

    mi::Size get_memory_consumption() const;

private:
    std::string m_value;
    std::string m_original_value;
};


class Value_vector : public Value_base<IValue_vector, IType_vector>
{
public:
    Value_vector( const Type* type, const IValue_factory* value_factory);

    mi::Size get_size() const { return m_type->get_size(); }

    const IValue_atomic* get_value( mi::Size index) const;

    IValue_atomic* get_value( mi::Size index);

    using IValue_compound::get_value;

    mi::Sint32 set_value( mi::Size index, IValue* value);

    mi::Size get_memory_consumption() const;

private:
    mi::base::Handle<IValue_atomic> m_values[4];
};


class Value_matrix : public Value_base<IValue_matrix, IType_matrix>
{
public:
    Value_matrix( const Type* type, const IValue_factory* value_factory);

    mi::Size get_size() const { return m_type->get_size(); }

    const IValue_vector* get_value( mi::Size index) const;

    IValue_vector* get_value( mi::Size index);

    using IValue_compound::get_value;

    mi::Sint32 set_value( mi::Size index, IValue* value);

    mi::Size get_memory_consumption() const;

private:
    mi::base::Handle<IValue_vector> m_values[4];
};


class Value_color : public Value_base<IValue_color, IType_color>
{
public:
    Value_color(
        const Type* type, const IValue_factory* value_factory,
        mi::Float32 red, mi::Float32 green, mi::Float32 blue);

    mi::Size get_size() const { return m_type->get_size(); }

    const IValue_float* get_value( mi::Size index) const;

    IValue_float* get_value( mi::Size index);

    using IValue_compound::get_value;

    mi::Sint32 set_value( mi::Size index, IValue* value);

    mi::Sint32 set_value( mi::Size index, IValue_float* value);

    mi::Size get_memory_consumption() const;

private:
    mi::base::Handle<IValue_float> m_values[3];
};


class Value_array : public Value_base<IValue_array, IType_array>
{
public:
    Value_array( const Type* type, const IValue_factory* value_factory);

    mi::Size get_size() const { return m_values.size(); }

    const IValue* get_value( mi::Size index) const;

    IValue* get_value( mi::Size index);

    using IValue_compound::get_value;

    mi::Sint32 set_value( mi::Size index, IValue* value);

    mi::Sint32 set_size( mi::Size size);

    mi::Size get_memory_consumption() const;

private:
    std::vector<mi::base::Handle<IValue> > m_values;
    mi::base::Handle<const IValue_factory> m_value_factory;
};


class Value_struct : public Value_base<IValue_struct, IType_struct>
{
public:
    Value_struct( const Type* type, const IValue_factory* value_factory);

    mi::Size get_size() const { return m_values.size(); }

    const IValue* get_value( mi::Size index) const;

    IValue* get_value( mi::Size index);

    using IValue_compound::get_value;

    mi::Sint32 set_value( mi::Size index, IValue* value);

    const IValue* get_field( const char* name) const;

    IValue* get_field( const char* name);

    using IValue_struct::get_field;

    mi::Sint32 set_field( const char* name, IValue* value);

    mi::Size get_memory_consumption() const;

private:
    std::vector<mi::base::Handle<IValue> > m_values;
};


class Value_texture : public Value_base<IValue_texture, IType_texture>
{
public:

    Value_texture(const Type* type, DB::Tag value) : Base(type), m_value(value), m_gamma(0.0f) { }

    Value_texture(
        const Type* type,
        DB::Tag value,
        const char* unresolved_mdl_url,
        const char *owner_module,
        mi::Float32 gamma)
        : Base( type)
        , m_value( value)
        , m_unresolved_mdl_url(unresolved_mdl_url ? unresolved_mdl_url : "")
        , m_owner_module(owner_module ? owner_module : "")
        , m_gamma(gamma)
    { }

    DB::Tag get_value() const { return m_value; }

    void set_value( DB::Tag value) { m_value = value; }

    const char* get_file_path( DB::Transaction* transaction) const;

    const char* get_unresolved_mdl_url() const { return m_unresolved_mdl_url.c_str(); }

    void set_unresolved_mdl_url(const char* url) { if (url) m_unresolved_mdl_url = url; }

    const char* get_owner_module() const { return m_owner_module.c_str(); }

    void set_owner_module(const char* module) { if (module) m_owner_module = module; }

    void set_gamma(Float32 gamma) { m_gamma = gamma; }

    mi::Float32 get_gamma() const { return m_gamma; }

    mi::Size get_memory_consumption() const;

private:
    DB::Tag m_value;
    mutable std::string m_cached_file_path;
    std::string m_unresolved_mdl_url;
    std::string m_owner_module;
    mi::Float32 m_gamma;
};


class Value_light_profile : public Value_base<IValue_light_profile, IType_light_profile>
{
public:
    Value_light_profile( const Type* type, DB::Tag value) : Base( type), m_value( value) { }

    Value_light_profile(
        const Type* type,
        DB::Tag value,
        const char* unresolved_mdl_url,
        const char *owner_module)
        : Base(type)
        , m_value(value)
        , m_unresolved_mdl_url(unresolved_mdl_url ? unresolved_mdl_url : "")
        , m_owner_module(owner_module ? owner_module : "") { }

    DB::Tag get_value() const { return m_value; }

    void set_value( DB::Tag value) { m_value = value; }

    const char* get_file_path( DB::Transaction* transaction) const;

    const char* get_unresolved_mdl_url() const { return m_unresolved_mdl_url.c_str(); }

    void set_unresolved_mdl_url(const char* url) { if (url) m_unresolved_mdl_url = url; }

    const char* get_owner_module() const { return m_owner_module.c_str(); }

    void set_owner_module(const char* module) { if (module) m_owner_module = module; }

    mi::Size get_memory_consumption() const;

private:
    DB::Tag m_value;
    mutable std::string m_cached_file_path;
    std::string m_unresolved_mdl_url;
    std::string m_owner_module;
};


class Value_bsdf_measurement
  : public Value_base<IValue_bsdf_measurement, IType_bsdf_measurement>
{
public:
    Value_bsdf_measurement( const Type* type, DB::Tag value) : Base( type), m_value( value) { }

    Value_bsdf_measurement(
        const Type* type,
        DB::Tag value,
        const char* unresolved_mdl_url,
        const char *owner_module)
        : Base(type)
        , m_value(value)
        , m_unresolved_mdl_url(unresolved_mdl_url ? unresolved_mdl_url : "")
        , m_owner_module(owner_module ? owner_module : "") { }

    DB::Tag get_value() const { return m_value; }

    void set_value( DB::Tag value) { m_value = value; }

    const char* get_file_path( DB::Transaction* transaction) const;

    const char* get_unresolved_mdl_url() const { return m_unresolved_mdl_url.c_str(); }

    void set_unresolved_mdl_url(const char* url) { if (url) m_unresolved_mdl_url = url; }

    const char* get_owner_module() const { return m_owner_module.c_str(); }

    void set_owner_module(const char* module) { if (module) m_owner_module = module; }

    mi::Size get_memory_consumption() const;

private:
    DB::Tag m_value;
    mutable std::string m_cached_file_path;
    std::string m_unresolved_mdl_url;
    std::string m_owner_module;
};


class Value_invalid_df : public Value_base<IValue_invalid_df, IType_reference>
{
public:
    Value_invalid_df( const Type* type) : Base( type) { }

    mi::Size get_memory_consumption() const;
};


class Value_list : public mi::base::Interface_implement<IValue_list>
{
public:
    // public API methods

    mi::Size get_size() const;

    mi::Size get_index( const char* name) const;

    const char* get_name( mi::Size index) const;

    const IValue* get_value( mi::Size index) const;

    const IValue* get_value( const char* name) const;

    mi::Sint32 set_value( mi::Size index, const IValue* value);

    mi::Sint32 set_value( const char* name, const IValue* value);

    mi::Sint32 add_value( const char* name, const IValue* value);

    mi::Size get_memory_consumption() const;

    friend class Value_factory; // for serialization/deserialization

private:

    typedef std::map<std::string, mi::Size> Name_index_map;
    Name_index_map m_name_index;

    typedef std::vector<std::string> Index_name_vector;
    Index_name_vector m_index_name;

    typedef std::vector<mi::base::Handle<const IValue> > Values_vector;
    Values_vector m_values;
};


class Value_factory : public mi::base::Interface_implement<IValue_factory>
{
public:
    Value_factory( IType_factory* type_factory);

    // public API methods

    IType_factory* get_type_factory() const;

    IValue_bool* create_bool( bool value) const;

    IValue_int* create_int( mi::Sint32 value) const;

    IValue_enum* create_enum(
        const IType_enum* type, mi::Size index) const;

    IValue_float* create_float( mi::Float32 value) const;

    IValue_double* create_double( mi::Float64 value) const;

    IValue_string* create_string( const char* value) const;

    IValue_string_localized* create_string_localized( const char* value, const char* original_value) const;

    IValue_vector* create_vector( const IType_vector* type) const;

    IValue_matrix* create_matrix( const IType_matrix* type) const;

    IValue_color* create_color(
        mi::Float32 red, mi::Float32 green, mi::Float32 blue) const;

    IValue_array* create_array( const IType_array* type) const;

    IValue_struct* create_struct( const IType_struct* type) const;

    IValue_texture* create_texture( const IType_texture* type, DB::Tag value) const;

    IValue_texture* create_texture(
        const IType_texture* type,
        DB::Tag value,
        const char *unresolved_mdl_url,
        const char *owner_module,
        mi::Float32 gamma) const;

    IValue_light_profile* create_light_profile( DB::Tag value) const;

    IValue_light_profile* create_light_profile(
        DB::Tag value,
        const char *unresolved_mdl_url,
        const char *owner_module) const;

    IValue_bsdf_measurement* create_bsdf_measurement( DB::Tag value) const;

    IValue_bsdf_measurement* create_bsdf_measurement(
        DB::Tag value,
        const char *unresolved_mdl_url,
        const char *owner_module) const;

    IValue_invalid_df* create_invalid_df( const IType_reference* type) const;

    IValue* create( const IType* type) const;

    using IValue_factory::create;

    IValue_list* create_value_list() const;

    IValue* clone( const IValue* value) const;

    using IValue_factory::clone;

    IValue_list* clone( const IValue_list* list) const;

    mi::Sint32 compare( const IValue* lhs, const IValue* rhs) const
    { return compare_static( lhs, rhs); }

    mi::Sint32 compare( const IValue_list* lhs, const IValue_list* rhs) const
    { return compare_static( lhs, rhs); }

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IValue* value,
        const char* name,
        mi::Size depth = 0) const;

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IValue_list* list,
        const char* name,
        mi::Size depth = 0) const;

    void serialize( SERIAL::Serializer* serializer, const IValue* value) const;

    IValue* deserialize( SERIAL::Deserializer* deserializer) const;

    using IValue_factory::deserialize;

    void serialize_list( SERIAL::Serializer* serializer, const IValue_list* list) const;

    IValue_list* deserialize_list( SERIAL::Deserializer* deserializer) const;

    // internal methods

    static mi::Sint32 compare_static( const IValue* lhs, const IValue* rhs);

    static mi::Sint32 compare_static( const IValue_list* lhs, const IValue_list* rhs);

    static void dump_static(
        DB::Transaction* transaction,
        const IValue* value,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);


    static void dump_static(
        DB::Transaction* transaction,
        const IValue_list* list,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);

private:
    mi::base::Handle<IType_factory> m_type_factory;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_VALUE_IMPL_H
