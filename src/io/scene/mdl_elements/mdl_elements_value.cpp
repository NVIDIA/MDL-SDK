/***************************************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Source for the IValue hierarchy and IValue_factory implementation.

#include "pch.h"

#include "mdl_elements_expression.h"
#include "mdl_elements_utilities.h"
#include "mdl_elements_value.h"
#include "mdl_elements_type.h"

#include <cstring>
#include <sstream>

#include <mi/neuraylib/istring.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/mem/i_mem_consumption.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>


namespace MI {

namespace MDL {

mi::Size Value_bool::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Size Value_int::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Sint32 Value_enum::set_value( mi::Sint32 value)
{
    mi::Size index = m_type->find_value( value);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    m_index = index;
    return 0;
}

mi::Sint32 Value_enum::set_name( const char* name)
{
    mi::Size index = m_type->find_value( name);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    m_index = index;
    return 0;
}

mi::Sint32 Value_enum::set_index( mi::Size index)
{
    if( index >= m_type->get_size())
        return -1;

    m_index = index;
    return 0;
}

mi::Size Value_enum::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Size Value_float::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Size Value_double::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Size Value_string::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type)
        + dynamic_memory_consumption( m_value);
}

mi::Size Value_string_localized::get_memory_consumption() const
{
    return sizeof(*this)
        + dynamic_memory_consumption( m_type)
        + dynamic_memory_consumption( m_value)
        + dynamic_memory_consumption( m_original_value);
}

Value_vector::Value_vector( const Type* type, const IValue_factory* value_factory)
  : Base( type)
{
    mi::base::Handle<const IType_atomic> element_type(
        type->get_element_type<IType_atomic>());
    ASSERT( M_SCENE, element_type);
    for( mi::Size i = 0; i < get_size(); ++i) {
        m_values[i] = value_factory->create<IValue_atomic>( element_type.get());
        ASSERT( M_SCENE, m_values[i]);
    }
}

const IValue_atomic* Value_vector::get_value( mi::Size index) const
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

IValue_atomic* Value_vector::get_value( mi::Size index)
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

mi::Sint32 Value_vector::set_value( mi::Size index, IValue* value)
{
    if( !value)
        return -1;
    if( index >= get_size())
        return -2;
    mi::base::Handle<const IType> actual_type( value->get_type());
    mi::base::Handle<const IType> expected_type( m_type->get_element_type());
    if( actual_type->get_kind() != expected_type->get_kind())
        return -3;
    m_values[index] = value->get_interface<IValue_atomic>();
    ASSERT( M_SCENE, m_values[index]);
    return 0;
}

mi::Size Value_vector::get_memory_consumption() const
{
    mi::Size size = sizeof( *this)
        + dynamic_memory_consumption( m_type);
    for( mi::Size i = 0; i < get_size(); ++i)
        size += dynamic_memory_consumption( m_values[i]);
    return size;
}

Value_matrix::Value_matrix( const Type* type, const IValue_factory* value_factory)
  : Base( type)
{
    mi::base::Handle<const IType_vector> element_type(
        type->get_element_type<IType_vector>());
    ASSERT( M_SCENE, element_type);
    for( mi::Size i = 0; i < get_size(); ++i) {
        m_values[i] = value_factory->create<IValue_vector>( element_type.get());
        ASSERT( M_SCENE, m_values[i]);
    }
}

const IValue_vector* Value_matrix::get_value( mi::Size index) const
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

IValue_vector* Value_matrix::get_value( mi::Size index)
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

mi::Sint32 Value_matrix::set_value( mi::Size index, IValue* value)
{
    if( !value)
        return -1;
    if( index >= get_size())
        return -2;
    mi::base::Handle<const IType> actual_type( value->get_type());
    mi::base::Handle<const IType> expected_type( m_type->get_element_type());
    if( actual_type->get_kind() != expected_type->get_kind())
        return -3;
    m_values[index] = value->get_interface<IValue_vector>();
    ASSERT( M_SCENE, m_values[index]);
    return 0;
}

mi::Size Value_matrix::get_memory_consumption() const
{
    mi::Size size = sizeof( *this)
        + dynamic_memory_consumption( m_type);
    for( mi::Size i = 0; i < get_size(); ++i)
        size += dynamic_memory_consumption( m_values[i]);
    return size;
}

Value_color::Value_color(
    const Type* type, const IValue_factory* value_factory,
    mi::Float32 red, mi::Float32 green, mi::Float32 blue)
  : Base( type)
{
    mi::base::Handle<const IType> component_type( type->get_component_type( 0));
    for( mi::Size i = 0; i < get_size(); ++i)
        m_values[i] = value_factory->create<IValue_float>( component_type.get());
    m_values[0]->set_value( red);
    m_values[1]->set_value( green);
    m_values[2]->set_value( blue);
}

const IValue_float* Value_color::get_value( mi::Size index) const
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

IValue_float* Value_color::get_value( mi::Size index)
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

mi::Sint32 Value_color::set_value( mi::Size index, IValue* value)
{
    if( !value)
        return -1;
    if( index >= get_size())
        return -2;
    mi::base::Handle<IValue_float> float_value(
        value->get_interface<IValue_float>());
    if( !float_value)
        return -3;
    m_values[index] = float_value;
    return 0;
}

mi::Sint32 Value_color::set_value( mi::Size index, IValue_float* value)
{
    if( !value)
        return -1;
    if( index >= get_size())
        return -2;
    m_values[index] = make_handle_dup( value);
    return 0;
}

mi::Size Value_color::get_memory_consumption() const
{
    mi::Size size = sizeof( *this)
        + dynamic_memory_consumption( m_type);
    for( mi::Size i = 0; i < get_size(); ++i)
        size += dynamic_memory_consumption( m_values[i]);
    return size;
}

Value_array::Value_array( const Type* type, const IValue_factory* value_factory)
  : Base( type), m_value_factory( value_factory, mi::base::DUP_INTERFACE)
{
    if( !type->is_immediate_sized())
        return;

    mi::Size size = type->get_size();
    m_values.resize( size);

    mi::base::Handle<const IType> element_type( type->get_element_type());
    for( mi::Size i = 0; i < size; ++i)
        m_values[i] = value_factory->create( element_type.get());
}

const IValue* Value_array::get_value( mi::Size index) const
{
    if( index >= m_values.size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

IValue* Value_array::get_value( mi::Size index)
{
    if( index >= m_values.size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

mi::Sint32 Value_array::set_value( mi::Size index, IValue* value)
{
    if( !value)
        return -1;
    if( index >= m_values.size())
        return -2;
    mi::base::Handle<const IType> actual_type( value->get_type());
    mi::base::Handle<const IType> expected_type( m_type->get_element_type());
    mi::base::Handle<const IType> expected_type_stripped( expected_type->skip_all_type_aliases());
    if( Type_factory::compare_static( actual_type.get(), expected_type_stripped.get()) != 0)
        return -3;
    m_values[index] = make_handle_dup( value);
    return 0;
}

mi::Sint32 Value_array::set_size( mi::Size size)
{
    if( m_type->is_immediate_sized())
        return -1;

    mi::Size old_size = m_values.size();
    m_values.resize( size);
    if( size <= old_size)
        return 0;

    mi::base::Handle<const IType> element_type( m_type->get_element_type());
    for( mi::Size i = old_size; i < size; ++i)
        m_values[i] = m_value_factory->create( element_type.get());
    return 0;
}

mi::Size Value_array::get_memory_consumption() const
{
    mi::Size size = sizeof( *this)
        + dynamic_memory_consumption( m_type);
    for( mi::Size i = 0; i < get_size(); ++i)
        size += dynamic_memory_consumption( m_values[i]);
    return size;
}

Value_struct::Value_struct( const Type* type, const IValue_factory* value_factory)
  : Base( type)
{
    mi::Size size = type->get_size();
    m_values.resize( size);

    for( mi::Size i = 0; i < size; ++i) {
        mi::base::Handle<const IType> component_type( type->get_component_type( i));
        m_values[i] = value_factory->create( component_type.get());
    }
}

const IValue* Value_struct::get_value( mi::Size index) const
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

IValue* Value_struct::get_value( mi::Size index)
{
    if( index >= get_size())
        return nullptr;

    m_values[index]->retain();
    return m_values[index].get();
}

mi::Sint32 Value_struct::set_value( mi::Size index, IValue* value)
{
    if( !value)
        return -1;
    if( index >= get_size())
        return -2;
    mi::base::Handle<const IType> actual_type( value->get_type());
    mi::base::Handle<const IType> expected_type( m_type->get_component_type( index));
    mi::base::Handle<const IType> expected_type_stripped( expected_type->skip_all_type_aliases());
    if( Type_factory::compare_static( actual_type.get(), expected_type_stripped.get()) != 0)
        return -3;
    m_values[index] = make_handle_dup( value);
    return 0;
}

const IValue* Value_struct::get_field( const char* name) const
{
    mi::Size index = m_type->find_field( name);
    return index != static_cast<mi::Size>( -1) ? get_value( index) : nullptr;
}

IValue* Value_struct::get_field( const char* name)
{
    mi::Size index = m_type->find_field( name);
    return index != static_cast<mi::Size>( -1) ? get_value( index) : nullptr;
}

mi::Sint32 Value_struct::set_field( const char* name, IValue* value)
{
    mi::Size index = m_type->find_field( name);
    return index != static_cast<mi::Size>( -1) ? set_value( index, value) : -2;
}

mi::Size Value_struct::get_memory_consumption() const
{
    mi::Size size = sizeof( *this)
        + dynamic_memory_consumption( m_type);
    for( mi::Size i = 0; i < get_size(); ++i)
        size += dynamic_memory_consumption( m_values[i]);
    return size;
}

std::string Value_texture::get_file_path( DB::Transaction* transaction) const
{
    if( !m_value && !m_unresolved_file_path.empty())
        return m_unresolved_file_path.c_str();

    if( !m_value || transaction->get_class_id( m_value) != TEXTURE::ID_TEXTURE)
        return {};
    DB::Access<TEXTURE::Texture> texture( m_value, transaction);
    DB::Tag image_tag = texture->get_image();
    if( image_tag && transaction->get_class_id(image_tag) == DBIMAGE::ID_IMAGE) {
        DB::Access<DBIMAGE::Image> image( image_tag, transaction);
        return image->get_mdl_file_path();
    }
    return {};
}

mi::Size Value_texture::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

std::string Value_light_profile::get_file_path( DB::Transaction* transaction) const
{
    if( !m_value && !m_unresolved_file_path.empty())
        return m_unresolved_file_path.c_str();

    if( !m_value || transaction->get_class_id( m_value) != LIGHTPROFILE::ID_LIGHTPROFILE)
        return {};
    DB::Access<LIGHTPROFILE::Lightprofile> light_profile( m_value, transaction);
    return light_profile->get_mdl_file_path();
}

mi::Size Value_light_profile::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

std::string Value_bsdf_measurement::get_file_path( DB::Transaction* transaction) const
{
    if( !m_value && !m_unresolved_file_path.empty())
        return m_unresolved_file_path.c_str();

    if( !m_value || transaction->get_class_id( m_value) != BSDFM::ID_BSDF_MEASUREMENT)
        return {};
    DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( m_value, transaction);
    return bsdf_measurement->get_mdl_file_path();
}

mi::Size Value_bsdf_measurement::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Size Value_invalid_df::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

Value_list::Value_list( mi::Size initial_capacity)
{
    m_index_name.reserve( initial_capacity);
    m_values.reserve( initial_capacity);
}

mi::Size Value_list::get_size() const
{
    return m_values.size();
}

mi::Size Value_list::get_index( const char* name) const
{
    if( !name)
        return static_cast<mi::Size>( -1);

    // For typical list sizes a linear search is much faster than maintaining a map from names to
    // indices.
    for( mi::Size i = 0; i < m_index_name.size(); ++i)
        if( m_index_name[i] == name)
            return i;

    return static_cast<mi::Size>( -1);
}

const char* Value_list::get_name( mi::Size index) const
{
    if( index >= m_index_name.size())
        return nullptr;
    return m_index_name[index].c_str();
}

const IValue* Value_list::get_value( mi::Size index) const
{
    if( index >= m_values.size())
        return nullptr;
    m_values[index]->retain();
    return m_values[index].get();
}

const IValue* Value_list::get_value( const char* name) const
{
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return nullptr;
    return get_value( index);
}

mi::Sint32 Value_list::set_value( mi::Size index, const IValue* value)
{
    if( !value)
        return -1;
    if( index >= m_values.size())
        return -2;
    m_values[index] = make_handle_dup( value);
    return 0;
}

mi::Sint32 Value_list::set_value( const char* name, const IValue* value)
{
    if( !value)
        return -1;
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -2;
    m_values[index] = make_handle_dup( value);
    return 0;
}

mi::Sint32 Value_list::add_value( const char* name, const IValue* value)
{
    if( !name || !value)
        return -1;
    mi::Size index = get_index( name);
    if( index != static_cast<mi::Size>( -1))
        return -2;
    m_values.push_back( make_handle_dup( value));
    m_index_name.push_back( name);
    return 0;
}

void Value_list::add_value_unchecked( const char* name, const IValue* value)
{
    m_values.push_back( make_handle_dup( value));
    m_index_name.push_back( name);
}

mi::Size Value_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_index_name)
        + dynamic_memory_consumption( m_values);
}

Value_factory::Value_factory( IType_factory* type_factory)
  : m_type_factory( static_cast<Type_factory*>( type_factory), mi::base::DUP_INTERFACE)
{
}

IType_factory* Value_factory::get_type_factory() const
{
    m_type_factory->retain();
    return m_type_factory.get();
}

IValue_bool* Value_factory::create_bool( bool value) const
{
    mi::base::Handle<const IType_bool> type( m_type_factory->create_bool());
    return new Value_bool( type.get(), value);
}

IValue_int* Value_factory::create_int( mi::Sint32 value) const
{
    mi::base::Handle<const IType_int> type( m_type_factory->create_int());
    return new Value_int( type.get(), value);
}

IValue_enum* Value_factory::create_enum(
    const IType_enum* type, mi::Size index) const
{
    return type ? new Value_enum( type, index) : nullptr;
}

IValue_float* Value_factory::create_float( mi::Float32 value) const
{
    mi::base::Handle<const IType_float> type( m_type_factory->create_float());
    return new Value_float( type.get(), value);
}

IValue_double* Value_factory::create_double( mi::Float64 value) const
{
    mi::base::Handle<const IType_double> type( m_type_factory->create_double());
    return new Value_double( type.get(), value);
}

IValue_string* Value_factory::create_string( const char* value) const
{
    mi::base::Handle<const IType_string> type( m_type_factory->create_string());
    return new Value_string( type.get(), value);
}

IValue_string_localized* Value_factory::create_string_localized(
    const char* value, const char* original_value) const
{
    mi::base::Handle<const IType_string> type( m_type_factory->create_string());
    return new Value_string_localized( type.get(), value, original_value);
}

IValue_vector* Value_factory::create_vector( const IType_vector* type) const
{
    return type ? new Value_vector( type, this) : nullptr;
}

IValue_matrix* Value_factory::create_matrix( const IType_matrix* type) const
{
    return type ? new Value_matrix( type, this) : nullptr;
}

IValue_color* Value_factory::create_color(
    mi::Float32 red, mi::Float32 green, mi::Float32 blue) const
{
    mi::base::Handle<const IType_color> type( m_type_factory->create_color());
    return new Value_color( type.get(), this, red, green, blue);
}

IValue_array* Value_factory::create_array( const IType_array* type) const
{
    return type ? new Value_array( type, this) : nullptr;
}

IValue_struct* Value_factory::create_struct( const IType_struct* type) const
{
    return type ? new Value_struct( type, this) : nullptr;
}

IValue_texture* Value_factory::create_texture( const IType_texture* type, DB::Tag value) const
{
    return type ? new Value_texture( type, value) : nullptr;
}

IValue_texture* Value_factory::create_texture(
    const IType_texture* type,
    DB::Tag value,
    const char* unresolved_file_path,
    const char* owner_module,
    mi::Float32 gamma,
    const char* selector) const
{
    if( !type)
        return nullptr;

    auto* tex = new Value_texture(
        type, value, unresolved_file_path, owner_module, gamma, selector);
    return tex;
}

IValue_light_profile* Value_factory::create_light_profile( DB::Tag value) const
{
    mi::base::Handle<const IType_light_profile> type( m_type_factory->create_light_profile());
    return new Value_light_profile( type.get(), value);
}

IValue_light_profile* Value_factory::create_light_profile(
    DB::Tag value,
    const char* unresolved_file_path,
    const char* owner_module) const
{
    mi::base::Handle<const IType_light_profile> type( m_type_factory->create_light_profile());
    return new Value_light_profile( type.get(), value, unresolved_file_path, owner_module);
}

IValue_bsdf_measurement* Value_factory::create_bsdf_measurement( DB::Tag value) const
{
    mi::base::Handle<const IType_bsdf_measurement> type( m_type_factory->create_bsdf_measurement());
    return new Value_bsdf_measurement( type.get(), value);
}

IValue_bsdf_measurement* Value_factory::create_bsdf_measurement(
    DB::Tag value,
    const char* unresolved_file_path,
    const char* owner_module) const
{
    mi::base::Handle<const IType_bsdf_measurement> type( m_type_factory->create_bsdf_measurement());
    return new Value_bsdf_measurement( type.get(), value, unresolved_file_path, owner_module);
}

IValue_invalid_df* Value_factory::create_invalid_df( const IType_reference* type) const
{
    if( !type)
        return nullptr;
    mi::base::Handle<const IType_resource> type_resource( type->get_interface<IType_resource>());
    if( type_resource)
        return nullptr;
    return new Value_invalid_df( type);
}

IValue* Value_factory::create( const IType* type) const
{
    if( !type)
        return nullptr;

    // Explicit cast to make the default arguments defined in the interface available.
    const IValue_factory* value_factory = this;

    IType::Kind kind = type->get_kind();

    switch( kind) {

        case IType::TK_BOOL:
            return value_factory->create_bool();
        case IType::TK_INT:
            return value_factory->create_int();
        case IType::TK_ENUM: {
            mi::base::Handle<const IType_enum> type_enum(
                type->get_interface<IType_enum>());
            return value_factory->create_enum( type_enum.get());
        }
        case IType::TK_FLOAT:
            return value_factory->create_float();
        case IType::TK_DOUBLE:
            return value_factory->create_double();
        case IType::TK_STRING:
            return value_factory->create_string();
        case IType::TK_VECTOR: {
            mi::base::Handle<const IType_vector> type_vector(
                type->get_interface<IType_vector>());
            return value_factory->create_vector( type_vector.get());
        }
        case IType::TK_MATRIX: {
            mi::base::Handle<const IType_matrix> type_matrix(
                type->get_interface<IType_matrix>());
            return value_factory->create_matrix( type_matrix.get());
        }
        case IType::TK_COLOR:
            return value_factory->create_color();
        case IType::TK_ARRAY: {
            mi::base::Handle<const IType_array> type_array(
                type->get_interface<IType_array>());
            return value_factory->create_array( type_array.get());
        }
        case IType::TK_STRUCT: {
            mi::base::Handle<const IType_struct> type_struct(
                type->get_interface<IType_struct>());
            return value_factory->create_struct( type_struct.get());
        }
        case IType::TK_TEXTURE: {
            mi::base::Handle<const IType_texture> type_texture(
                type->get_interface<IType_texture>());
            return value_factory->create_texture( type_texture.get(), DB::Tag());
        }
        case IType::TK_LIGHT_PROFILE:
            return value_factory->create_light_profile( DB::Tag());
        case IType::TK_BSDF_MEASUREMENT:
            return value_factory->create_bsdf_measurement( DB::Tag());
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF: {
            mi::base::Handle<const IType_reference> type_reference(
                type->get_interface<IType_reference>());
            return value_factory->create_invalid_df( type_reference.get());
        }
        case IType::TK_ALIAS: {
            mi::base::Handle<const IType> type_base(
                type->skip_all_type_aliases());
            return value_factory->create( type_base.get());
        }
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

namespace {

/// Sets \p average to the average of \p min and \p max.
///
/// Assume that all three arguments have the same type. Supported types are
/// TK_INT/TK_FLOAT/TK_DOUBLE.
void get_average( const IValue* min, const IValue* max, IValue* average)
{
    mi::base::Handle<const IType> type( average->get_type());

    switch( type->get_kind()) {

        case IType::TK_INT: {
            mi::base::Handle<const IValue_int> min_int( min->get_interface<IValue_int>());
            mi::base::Handle<const IValue_int> max_int( max->get_interface<IValue_int>());
            mi::base::Handle<IValue_int> average_int( average->get_interface<IValue_int>());
            mi::Sint32 min_v = min_int->get_value();
            mi::Sint32 max_v = max_int->get_value();
            average_int->set_value( (min_v + max_v) / 2);
            break;
        }

        case IType::TK_FLOAT: {
            mi::base::Handle<const IValue_float> min_float( min->get_interface<IValue_float>());
            mi::base::Handle<const IValue_float> max_float( max->get_interface<IValue_float>());
            mi::base::Handle<IValue_float> average_float( average->get_interface<IValue_float>());
            mi::Float32 min_v = min_float->get_value();
            mi::Float32 max_v = max_float->get_value();
            average_float->set_value( (min_v + max_v) / 2.0f);
            break;
        }

        case IType::TK_DOUBLE: {
            mi::base::Handle<const IValue_double> min_double( min->get_interface<IValue_double>());
            mi::base::Handle<const IValue_double> max_double( max->get_interface<IValue_double>());
            mi::base::Handle<IValue_double> average_double(
                average->get_interface<IValue_double>());
            mi::Float64 min_v = min_double->get_value();
            mi::Float64 max_v = max_double->get_value();
            average_double->set_value( (min_v + max_v) / 2.0);
            break;
        }

        case IType::TK_ALIAS:
        case IType::TK_BOOL:
        case IType::TK_ENUM:
        case IType::TK_STRING:
        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
        case IType::TK_COLOR:
        case IType::TK_ARRAY:
        case IType::TK_STRUCT:
        case IType::TK_TEXTURE:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            ASSERT( M_SCENE, false);
            break;
    }
}

} // namespace

IValue* Value_factory::create( const IAnnotation* annotation) const
{
    if( !annotation)
        return nullptr;

    const char* anno_name = annotation->get_name();
    if(    (strncmp( anno_name, "::anno::soft_range(", 19) != 0)
        && (strncmp( anno_name, "::anno::hard_range(", 19) != 0))
        return nullptr;

     mi::base::Handle<const IExpression_list> arguments( annotation->get_arguments());
     mi::base::Handle<const IExpression_constant> min(
         arguments->get_expression<IExpression_constant>( "min"));
     mi::base::Handle<const IExpression_constant> max(
         arguments->get_expression<IExpression_constant>( "max"));
     ASSERT( M_SCENE, min && max);
     if( !min || !max)
         return nullptr;

     mi::base::Handle<const IValue> min_value( min->get_value());
     mi::base::Handle<const IValue> max_value( max->get_value());

     mi::base::Handle<const IType> min_type( min_value->get_type());
     mi::base::Handle<const IType> max_type( max_value->get_type());
     IType::Kind min_kind = min_type->get_kind();
     [[maybe_unused]] IType::Kind max_kind = max_type->get_kind();
     ASSERT( M_SCENE, min_kind == max_kind);

     switch( min_kind) {

         case IType::TK_INT: {
            IValue_int* result = create_int( 0);
            get_average( min_value.get(), max_value.get(), result);
            return result;
        }

        case IType::TK_FLOAT: {
            IValue_float* result = create_float( 0.0f);
            get_average( min_value.get(), max_value.get(), result);
            return result;
        }

        case IType::TK_DOUBLE: {
            IValue_double* result = create_double( 0.0);
            get_average( min_value.get(), max_value.get(), result);
            return result;
        }

        case IType::TK_COLOR: {

            mi::base::Handle<const IValue_color> min_color(
                min_value->get_interface<IValue_color>());
            mi::base::Handle<const IValue_color> max_color(
                max_value->get_interface<IValue_color>());
            IValue_color* result = create_color( 0.0f, 0.0f, 0.0f);

            for( mi::Size i = 0; i < 3; ++i) {
                mi::base::Handle<const IValue_float> min_value( min_color->get_value( i));
                mi::base::Handle<const IValue_float> max_value( max_color->get_value( i));
                mi::base::Handle<      IValue_float> result_value( result->get_value( i));
                get_average( min_value.get(), max_value.get(), result_value.get());
            }

            return result;
        }

       case IType::TK_VECTOR: {
            mi::base::Handle<const IType_vector> min_vector_type(
                min_type->get_interface<IType_vector>());
            mi::base::Handle<const IType_vector> max_vector_type(
                max_type->get_interface<IType_vector>());
           mi::base::Handle<const IType_atomic> min_element_type(
               min_vector_type->get_element_type());
           mi::base::Handle<const IType_atomic> max_element_type(
               max_vector_type->get_element_type());
           [[maybe_unused]] IType::Kind min_element_kind = min_element_type->get_kind();
           [[maybe_unused]] IType::Kind max_element_kind = max_element_type->get_kind();
           ASSERT( M_SCENE, min_element_kind == max_element_kind);

           mi::base::Handle<const IValue_vector> min_vector(
               min_value->get_interface<IValue_vector>());
           mi::base::Handle<const IValue_vector> max_vector(
               max_value->get_interface<IValue_vector>());
           IValue_vector* result = create_vector( min_vector_type.get());

           mi::Size n = min_vector_type->get_size();
           for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue_atomic> min_element( min_vector->get_value( i));
                mi::base::Handle<const IValue_atomic> max_element( max_vector->get_value( i));
                mi::base::Handle<      IValue_atomic> res_element( result->get_value( i));
                get_average( min_element.get(), max_element.get(), res_element.get());
            }

            return result;
        }

        case IType::TK_ALIAS:
        case IType::TK_BOOL:
        case IType::TK_ENUM:
        case IType::TK_STRING:
        case IType::TK_MATRIX:
        case IType::TK_ARRAY:
        case IType::TK_STRUCT:
        case IType::TK_TEXTURE:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            break;
     }

     ASSERT( M_SCENE, !"unsupported overload");
     return nullptr;
}

IValue* Value_factory::create( const IType* type, const IAnnotation_block* annotation_block) const
{
    if( !annotation_block)
        return create( type);

    mi::base::Handle<const IAnnotation> range_annotation;

    for( mi::Size i = 0, n = annotation_block->get_size(); i < n; ++i) {

        mi::base::Handle<const IAnnotation> annotation(
            annotation_block->get_annotation( i));
        const char* anno_name = annotation->get_name();
        if(    (strncmp( anno_name, "::anno::soft_range(", 19) != 0)
            && (strncmp( anno_name, "::anno::hard_range(", 19) != 0))
            continue;

        // ::anno::soft_range has higher priority than ::anno::hard_range
        if(    (strncmp( anno_name, "::anno::hard_range(", 19) == 0)
            && range_annotation)
            continue;

        range_annotation = annotation;
    }

    return range_annotation ? create( range_annotation.get()) : create( type);
}

IValue_list* Value_factory::create_value_list( mi::Size initial_capacity) const
{
    return new Value_list( initial_capacity);
}

IValue* Value_factory::clone( const IValue* value) const
{
    if( !value)
        return nullptr;

    IValue::Kind kind = value->get_kind();

    switch( kind) {

        case IValue::VK_BOOL: {
            mi::base::Handle<const IValue_bool> value_bool(
                value->get_interface<IValue_bool>());
            return create_bool( value_bool->get_value());
        }
        case IValue::VK_INT: {
            mi::base::Handle<const IValue_int> value_int(
                value->get_interface<IValue_int>());
            return create_int( value_int->get_value());
        }
        case IValue::VK_ENUM: {
            mi::base::Handle<const IType_enum> type_enum(
                value->get_type<IType_enum>());
            mi::base::Handle<const IValue_enum> value_enum(
                value->get_interface<IValue_enum>());
            return create_enum( type_enum.get(), value_enum->get_index());
        }
        case IValue::VK_FLOAT: {
            mi::base::Handle<const IValue_float> value_float(
                value->get_interface<IValue_float>());
            return create_float( value_float->get_value());
        }
        case IValue::VK_DOUBLE: {
            mi::base::Handle<const IValue_double> value_double(
                value->get_interface<IValue_double>());
            return create_double( value_double->get_value());
        }
        case IValue::VK_STRING: {
            mi::base::Handle<const IValue_string_localized> value_string_localized(
                value->get_interface<IValue_string_localized>());
            if( value_string_localized)
                return create_string_localized(
                    value_string_localized->get_value(),
                    value_string_localized->get_original_value());
            mi::base::Handle<const IValue_string> value_string(
                value->get_interface<IValue_string>());
            return create_string( value_string->get_value());
        }
        case IValue::VK_VECTOR:
        case IValue::VK_MATRIX:
        case IValue::VK_COLOR:
        case IValue::VK_ARRAY:
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IType_compound> type_compound(
                value->get_type<IType_compound>());
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            auto* result = create<IValue_compound>( type_compound.get());
            mi::Size n = value_compound->get_size();
            if( kind == IValue::VK_ARRAY) {
                mi::base::Handle<IValue_array> result_array(
                    result->get_interface<IValue_array>());
                result_array->set_size( n);
            }
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element( value_compound->get_value( i));
                mi::base::Handle<IValue> element_clone( clone( element.get()));
                result->set_value( i, element_clone.get());
            }
            return result;
        }
        case IValue::VK_TEXTURE: {
            mi::base::Handle<const IType_texture> type_texture(
                value->get_type<IType_texture>());
            mi::base::Handle<const IValue_texture> value_texture(
                value->get_interface<IValue_texture>());
            return create_texture(
                type_texture.get(),
                value_texture->get_value(),
                value_texture->get_unresolved_file_path(),
                value_texture->get_owner_module(),
                value_texture->get_gamma(),
                value_texture->get_selector());
        }
        case IValue::VK_LIGHT_PROFILE: {
            mi::base::Handle<const IValue_light_profile> value_light_profile(
                value->get_interface<IValue_light_profile>());
            return create_light_profile(
                value_light_profile->get_value(),
                value_light_profile->get_unresolved_file_path(),
                value_light_profile->get_owner_module());
        }
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_bsdf_measurement> value_bsdf_measurement(
                value->get_interface<IValue_bsdf_measurement>());
            return create_bsdf_measurement(
                value_bsdf_measurement->get_value(),
                value_bsdf_measurement->get_unresolved_file_path(),
                value_bsdf_measurement->get_owner_module());
        }
        case IValue::VK_INVALID_DF: {
            mi::base::Handle<const IType_reference> type_reference(
                value->get_type<IType_reference>());
            return create_invalid_df( type_reference.get());
        }
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

IValue_list* Value_factory::clone( const IValue_list* list) const
{
    if( !list)
        return nullptr;

    mi::Size n = list->get_size();
    IValue_list* result = create_value_list( n);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IValue> value( list->get_value( i));
        mi::base::Handle<IValue> clone_value( clone( value.get()));
        const char* name = list->get_name( i);
        result->add_value( name, clone_value.get());
    }
    return result;
}

namespace {

std::string get_prefix( mi::Size depth)
{
    std::string prefix;
    for( mi::Size i = 0; i < depth; i++)
        prefix += "    ";
    return prefix;
}

} // namespace

const mi::IString* Value_factory::dump(
    DB::Transaction* transaction,
    const IValue* value,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    dump_static( transaction, value, name, depth, s);
    return create_istring( s.str().c_str());
}

const mi::IString* Value_factory::dump(
    DB::Transaction* transaction,
    const IValue_list* list,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    dump_static( transaction, list, name, depth, s);
    return create_istring( s.str().c_str());
}

mi::Sint32 Value_factory::compare_static( const IValue* lhs, const IValue* rhs, mi::Float64 epsilon)
{
    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::base::Handle<const IType> lhs_type( lhs->get_type()); //-V522 PVS
    mi::base::Handle<const IType> rhs_type( rhs->get_type()); //-V522 PVS
    mi::Sint32 type_cmp = Type_factory::compare_static( lhs_type.get(), rhs_type.get());
    if( type_cmp != 0)
        return type_cmp;

    IValue::Kind kind = lhs->get_kind();

    switch( kind) {

        case IValue::VK_BOOL: {
            mi::base::Handle<const IValue_bool> lhs_bool(
                lhs->get_interface<IValue_bool>());
            mi::base::Handle<const IValue_bool> rhs_bool(
                rhs->get_interface<IValue_bool>());
            bool lhs_value = lhs_bool->get_value();
            bool rhs_value = rhs_bool->get_value();
            if( lhs_value < rhs_value) return -1;
            if( lhs_value > rhs_value) return +1;
            return 0;
        }
        case IValue::VK_INT: {
            mi::base::Handle<const IValue_int> lhs_int(
                lhs->get_interface<IValue_int>());
            mi::base::Handle<const IValue_int> rhs_int(
                rhs->get_interface<IValue_int>());
            mi::Sint32 lhs_value = lhs_int->get_value();
            mi::Sint32 rhs_value = rhs_int->get_value();
            if( lhs_value < rhs_value) return -1;
            if( lhs_value > rhs_value) return +1;
            return 0;
        }
        case IValue::VK_ENUM: {
            mi::base::Handle<const IValue_enum> lhs_enum(
                lhs->get_interface<IValue_enum>());
            mi::base::Handle<const IValue_enum> rhs_enum(
                rhs->get_interface<IValue_enum>());
            mi::Size lhs_index = lhs_enum->get_index();
            mi::Size rhs_index = rhs_enum->get_index();
            if( lhs_index < rhs_index) return -1;
            if( lhs_index > rhs_index) return +1;
            return 0;
        }
        case IValue::VK_FLOAT: {
            mi::base::Handle<const IValue_float> lhs_float(
                lhs->get_interface<IValue_float>());
            mi::base::Handle<const IValue_float> rhs_float(
                rhs->get_interface<IValue_float>());
            mi::Float32 lhs_value = lhs_float->get_value();
            mi::Float32 rhs_value = rhs_float->get_value();
            if( epsilon > 0 && fabsf( lhs_value-rhs_value) <= epsilon)
                return 0;
            if( lhs_value < rhs_value) return -1;
            if( lhs_value > rhs_value) return +1;
            return 0;
        }
        case IValue::VK_DOUBLE: {
            mi::base::Handle<const IValue_double> lhs_double(
                lhs->get_interface<IValue_double>());
            mi::base::Handle<const IValue_double> rhs_double(
                rhs->get_interface<IValue_double>());
            mi::Float64 lhs_value = lhs_double->get_value();
            mi::Float64 rhs_value = rhs_double->get_value();
            if( epsilon > 0 && fabs( lhs_value-rhs_value) <= epsilon)
                return 0;
            if( lhs_value < rhs_value) return -1;
            if( lhs_value > rhs_value) return +1;
            return 0;
        }
        case IValue::VK_STRING: {
            mi::base::Handle<const IValue_string_localized> lhs_string_localized(
                lhs->get_interface<IValue_string_localized>());
            if( lhs_string_localized) {
                mi::base::Handle<const IValue_string_localized> rhs_string_localized(
                    rhs->get_interface<IValue_string_localized>());
                const char* lhs_value = lhs_string_localized->get_value();
                const char* rhs_value = rhs_string_localized->get_value();
                mi::Sint32 result = strcmp( lhs_value, rhs_value);
                if (result < 0) return -1;
                if (result > 0) return +1;
                const char* lhs_original_value = lhs_string_localized->get_original_value();
                const char* rhs_original_value = rhs_string_localized->get_original_value();
                result = strcmp( lhs_original_value, rhs_original_value);
                if( result < 0) return -1;
                if( result > 0) return +1;
                return 0;
            }
            mi::base::Handle<const IValue_string> lhs_string(
                lhs->get_interface<IValue_string>());
            mi::base::Handle<const IValue_string> rhs_string(
                rhs->get_interface<IValue_string>());
            const char* lhs_value = lhs_string->get_value();
            const char* rhs_value = rhs_string->get_value();
            mi::Sint32 result = strcmp( lhs_value, rhs_value);
            if( result < 0) return -1;
            if( result > 0) return +1;
            return 0;
        }
        case IValue::VK_VECTOR:
        case IValue::VK_MATRIX:
        case IValue::VK_COLOR:
        case IValue::VK_ARRAY:
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IValue_compound> lhs_compound(
                lhs->get_interface<IValue_compound>());
            mi::base::Handle<const IValue_compound> rhs_compound(
                rhs->get_interface<IValue_compound>());
            mi::Size lhs_size = lhs_compound->get_size();
            mi::Size rhs_size = rhs_compound->get_size();
            if( lhs_size < rhs_size) return -1; // for deferred-sized arrays
            if( lhs_size > rhs_size) return +1; // for deferred-sized arrays
            for( mi::Size i = 0; i < lhs_size; ++i) {
                mi::base::Handle<const IValue> lhs_element(
                    lhs_compound->get_value( i));
                mi::base::Handle<const IValue> rhs_element(
                    rhs_compound->get_value( i));
                mi::Sint32 result = compare_static( lhs_element.get(), rhs_element.get(), epsilon);
                if( result != 0)
                    return result;
            }
            return 0;
        }
        case IValue::VK_TEXTURE:
        case IValue::VK_LIGHT_PROFILE:
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_resource> lhs_resource(
                lhs->get_interface<IValue_resource>());
            mi::base::Handle<const IValue_resource> rhs_resource(
                rhs->get_interface<IValue_resource>());
            DB::Tag lhs_value = lhs_resource->get_value();
            DB::Tag rhs_value = rhs_resource->get_value();
            if( lhs_value < rhs_value) return -1;
            if( lhs_value > rhs_value) return +1;
            return 0;
        }
        case IValue::VK_INVALID_DF:
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

mi::Sint32 Value_factory::compare_static(
   const IValue_list* lhs, const IValue_list* rhs, mi::Float64 epsilon)
{
    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::Size lhs_n = lhs->get_size(); //-V522 PVS
    mi::Size rhs_n = rhs->get_size(); //-V522 PVS
    if( lhs_n < rhs_n) return -1;
    if( lhs_n > rhs_n) return +1;

    for( mi::Size i = 0; i < lhs_n; ++i) {
        const char* lhs_name = lhs->get_name( i);
        const char* rhs_name = rhs->get_name( i);
        mi::Sint32 result = strcmp( lhs_name, rhs_name);
        if( result < 0) return -1;
        if( result > 0) return +1;
        mi::base::Handle<const IValue> lhs_value( lhs->get_value( i));
        mi::base::Handle<const IValue> rhs_value( rhs->get_value( i));
        result = compare_static( lhs_value.get(), rhs_value.get(), epsilon);
        if( result != 0)
            return result;
    }

    return 0;
}

void Value_factory::dump_static(
    DB::Transaction* transaction,
    const IValue* value,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !value)
        return;

    IValue::Kind kind = value->get_kind();

    mi::base::Handle<const IType> type( value->get_type());

    if( name) {
        std::string type_name = Type_factory::get_dump_type_name(
            type.get(), /*include_aliased_type*/ true);
        s << type_name << ' ' << name << " = ";
    }

    switch( kind) {

        case IValue::VK_BOOL: {
            mi::base::Handle<const IValue_bool> value_bool(
                value->get_interface<IValue_bool>());
            s << (value_bool->get_value() ? "true" : "false");
            return;
        }
        case IValue::VK_INT: {
            mi::base::Handle<const IValue_int> value_int(
                value->get_interface<IValue_int>());
            s << value_int->get_value();
            return;
        }
        case IValue::VK_ENUM: {
            mi::base::Handle<const IValue_enum> value_enum(
                value->get_interface<IValue_enum>());
            mi::base::Handle<const IType_enum> type_enum( value_enum->get_type());
            mi::Size index = value_enum->get_index();
            s << type_enum->get_value_name( index)
              << '(' << type_enum->get_value_code( index) << ')';
            return;
        }
        case IValue::VK_FLOAT: {
            mi::base::Handle<const IValue_float> value_float(
                value->get_interface<IValue_float>());
            s << value_float->get_value();
            return;
        }
        case IValue::VK_DOUBLE: {
            mi::base::Handle<const IValue_double> value_double(
                value->get_interface<IValue_double>());
            s << value_double->get_value();
            return;
        }
        case IValue::VK_STRING: {
            mi::base::Handle<const IValue_string_localized> value_string_localized(
                value->get_interface<IValue_string_localized>());
            if( value_string_localized) {
                s << '\"' << value_string_localized->get_value() << '\"';
                s << ", ";
                s << '\"' << value_string_localized->get_original_value() << '\"';
                return;
            }
            mi::base::Handle<const IValue_string> value_string(
                value->get_interface<IValue_string>());
            s << '\"' << value_string->get_value() << '\"';
            return;
        }
        case IValue::VK_VECTOR:
        case IValue::VK_COLOR: {
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            s << '(';
            mi::Size n = value_compound->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element(
                    value_compound->get_value( i));
                dump_static(  transaction, element.get(), nullptr, 0, s);
                if( i < n-1)
                    s << ", ";
            }
            s << ')';
            return;
        }
        case IValue::VK_MATRIX:
        {
            mi::base::Handle<const IValue_matrix> value_matrix(
                value->get_interface<IValue_matrix>());
            s << '(';
            mi::Size columns = value_matrix->get_size();
            for( mi::Size i = 0; i < columns; ++i) {
                mi::base::Handle<const IValue_vector> column(
                    value_matrix->get_value( i));
                mi::Size rows = column->get_size();
                for( mi::Size j = 0; j < rows; ++j) {
                    mi::base::Handle<const IValue> element( column->get_value( j));
                    dump_static( transaction, element.get(), nullptr, 0, s);
                    if( i < columns-1 || j < rows-1)
                        s << ", ";
                }
            }
            s << ')';
            return;
        }
        case IValue::VK_ARRAY: {
            mi::base::Handle<const IValue_array> value_array(
                value->get_interface<IValue_array>());
            mi::Size n = value_array->get_size();
            s << (n > 0 ? "[\n" : "[ ");
            const std::string& prefix = get_prefix( depth);
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element(
                    value_array->get_value( i));
                std::ostringstream element_name;
                element_name << i;
                s << prefix << "    ";
                dump_static( transaction, element.get(), element_name.str().c_str(), depth+1, s);
                s << ";\n";
            }
            s << (n > 0 ? prefix : "") << "]";
            return;
        }
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IValue_struct> value_struct(
                value->get_interface<IValue_struct>());
            mi::base::Handle<const IType_struct> type_struct(
                type->get_interface<IType_struct>());
            mi::Size n = value_struct->get_size();
            s << "{\n";
            const std::string& prefix = get_prefix( depth);
            for( mi::Size i = 0; i < n; ++i) {
                const char* field_name = type_struct->get_field_name( i);
                mi::base::Handle<const IValue> element(
                    value_struct->get_value( i));
                s << prefix << "    ";
                dump_static( transaction, element.get(), field_name, depth+1, s);
                s << ";\n";
            }
            s << prefix << '}';
            return;
        }
        case IValue::VK_TEXTURE:
        case IValue::VK_LIGHT_PROFILE:
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_resource> value_resource(
                value->get_interface<IValue_resource>());
            DB::Tag tag = value_resource->get_value();
            if( !tag) {
                s << "(unset";
                const std::string& owner_module = value_resource->get_owner_module();
                s << ", owner module \"" << owner_module << "\"";
                const std::string& unresolved_mdl_file_path
                    = value_resource->get_unresolved_file_path();
                s << ", unresolved MDL file path \"" << unresolved_mdl_file_path << "\")";
                return;
            }
            if( transaction)
                s << '\"' << transaction->tag_to_name( tag) << '\"';
            else
                s << "tag " << tag.get_uint();
            return;
        }
        case IValue::VK_INVALID_DF:
            s << "(invalid reference)";
            return;
    }

    ASSERT( M_SCENE, false);
}

void Value_factory::dump_static(
    DB::Transaction* transaction,
    const IValue_list* list,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !list)
        return;

    s << "value_list ";
    if( name)
        s << name << " = ";

    mi::Size n = list->get_size();
    s << (n > 0 ? "[\n" : "[ ");

    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IValue> value( list->get_value( i));
        s << prefix << "    ";
        std::ostringstream name;
        name << i << ": " << list->get_name( i);
        dump_static( transaction, value.get(), name.str().c_str(), depth+1, s);
        s << ";\n";
    }

    s << (n > 0 ? prefix : "") << "]";
}

void Value_factory::serialize( SERIAL::Serializer* serializer, const IValue* value) const
{
    IValue::Kind kind = value->get_kind();
    mi::Uint32 kind_as_uint32 = kind;
    SERIAL::write( serializer, kind_as_uint32);

    switch( kind) {

        case IValue::VK_BOOL: {
            mi::base::Handle<const IValue_bool> value_bool(
                value->get_interface<IValue_bool>());
            SERIAL::write( serializer, value_bool->get_value());
            return;
        }
        case IValue::VK_INT: {
            mi::base::Handle<const IValue_int> value_int(
                value->get_interface<IValue_int>());
            SERIAL::write( serializer, value_int->get_value());
            return;
        }
        case IValue::VK_ENUM: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_enum> value_enum(
                value->get_interface<IValue_enum>());
            SERIAL::write( serializer, value_enum->get_index());
            return;
        }
        case IValue::VK_FLOAT: {
            mi::base::Handle<const IValue_float> value_float(
                value->get_interface<IValue_float>());
            SERIAL::write( serializer, value_float->get_value());
            return;
        }
        case IValue::VK_DOUBLE: {
            mi::base::Handle<const IValue_double> value_double(
                value->get_interface<IValue_double>());
            SERIAL::write( serializer, value_double->get_value());
            return;
        }
        case IValue::VK_STRING: {
            mi::base::Handle<const IValue_string_localized> value_string_localized(
                value->get_interface<IValue_string_localized>());
            if ( value_string_localized) {
                SERIAL::write( serializer, true);// this is a localized string
                SERIAL::write( serializer, value_string_localized->get_value());
                SERIAL::write( serializer, value_string_localized->get_original_value());
                return;
            }
            mi::base::Handle<const IValue_string> value_string(
                value->get_interface<IValue_string>());
            SERIAL::write( serializer, false);// this is not a localized string
            SERIAL::write( serializer, value_string->get_value());
            return;
        }
        case IValue::VK_VECTOR: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            mi::Size n = value_compound->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element( value_compound->get_value( i));
                serialize( serializer, element.get());
            }
            return;
        }
        case IValue::VK_MATRIX: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            mi::Size n = value_compound->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element( value_compound->get_value( i));
                serialize( serializer, element.get());
            }
            return;
        }
        case IValue::VK_COLOR: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            mi::Size n = value_compound->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element( value_compound->get_value( i));
                serialize( serializer, element.get());
            }
            return;
        }
        case IValue::VK_ARRAY: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            mi::Size n = value_compound->get_size();
            SERIAL::write( serializer, n);
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element( value_compound->get_value( i));
                serialize( serializer, element.get());
            }
            return;
        }
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            mi::Size n = value_compound->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element( value_compound->get_value( i));
                serialize( serializer, element.get());
            }
            return;
        }
        case IValue::VK_TEXTURE: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            mi::base::Handle<const IValue_texture> value_texture(
                value->get_interface<IValue_texture>());
            SERIAL::write( serializer, value_texture->get_value());
            serializer->write( value_texture->get_unresolved_file_path());
            serializer->write( value_texture->get_owner_module());
            serializer->write( value_texture->get_gamma());
            const char* selector = value_texture->get_selector();
            SERIAL::write( serializer, selector ? std::string( selector) : std::string());
            return;
        }
        case IValue::VK_LIGHT_PROFILE: {
            mi::base::Handle<const IValue_light_profile> value_light_profile(
                value->get_interface<IValue_light_profile>());
            SERIAL::write( serializer, value_light_profile->get_value());
            serializer->write( value_light_profile->get_unresolved_file_path());
            serializer->write( value_light_profile->get_owner_module());
            return;
        }
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_bsdf_measurement> value_bsdf_measurement(
                value->get_interface<IValue_bsdf_measurement>());
            SERIAL::write( serializer, value_bsdf_measurement->get_value());
            serializer->write( value_bsdf_measurement->get_unresolved_file_path());
            serializer->write( value_bsdf_measurement->get_owner_module());
            return;
        }
        case IValue::VK_INVALID_DF: {
            mi::base::Handle<const IType> type( value->get_type());
            m_type_factory->serialize( serializer, type.get());
            return;
        }
    }

    ASSERT( M_SCENE, false);
}

IValue* Value_factory::deserialize( SERIAL::Deserializer* deserializer) const
{
    mi::Uint32 kind_as_uint32;
    SERIAL::read( deserializer, &kind_as_uint32);
    auto kind = static_cast<IValue::Kind>( kind_as_uint32);

    switch( kind) {

        case IValue::VK_BOOL: {
            bool value;
            SERIAL::read( deserializer, &value);
            return create_bool( value);
        }
        case IValue::VK_INT: {
            mi::Sint32 value;
            SERIAL::read( deserializer, &value);
            return create_int( value);
        }
        case IValue::VK_ENUM: {
            mi::base::Handle<const IType_enum> type(
                m_type_factory->deserialize<IType_enum>( deserializer));
            mi::Size index;
            SERIAL::read( deserializer, &index);
            return create_enum( type.get(), index);
        }
        case IValue::VK_FLOAT: {
            mi::Float32 value;
            SERIAL::read( deserializer, &value);
            return create_float( value);
        }
        case IValue::VK_DOUBLE: {
            mi::Float64 value;
            SERIAL::read( deserializer, &value);
            return create_double( value);
        }
        case IValue::VK_STRING: {
            bool localized;
            SERIAL::read( deserializer, &localized);
            char* value;
            deserializer->read( &value);
            if( localized) {
                char* value_original;
                SERIAL::read( deserializer, &value_original);
                IValue_string* result = create_string_localized( value, value_original);
                deserializer->release( value);
                deserializer->release( value_original);
                return result;
            }
            IValue_string* result = create_string( value);
            deserializer->release( value);
            return result;
        }
        case IValue::VK_VECTOR: {
            mi::base::Handle<const IType_vector> type(
                m_type_factory->deserialize<IType_vector>( deserializer));
            IValue_vector* result = create_vector( type.get());
            mi::Size n = type->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<IValue> element( deserialize( deserializer));
                result->set_value( i, element.get()); //-V522 PVS
            }
            return result;
        }
        case IValue::VK_MATRIX: {
            mi::base::Handle<const IType_matrix> type(
                m_type_factory->deserialize<IType_matrix>( deserializer));
            IValue_matrix* result = create_matrix( type.get());
            mi::Size n = type->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<IValue> element( deserialize( deserializer));
                result->set_value( i, element.get()); //-V522 PVS
            }
            return result;
        }
        case IValue::VK_COLOR: {
            mi::base::Handle<const IType_color> type(
                m_type_factory->deserialize<IType_color>( deserializer));
            IValue_color* result = create_color( 0, 0, 0);
            mi::Size n = type->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<IValue_float> element( deserialize<IValue_float>( deserializer));
                result->set_value( i, element.get()); //-V522 PVS
            }
            return result;
        }
        case IValue::VK_ARRAY: {
            mi::base::Handle<const IType_array> type(
                m_type_factory->deserialize<IType_array>( deserializer));
            IValue_array* result = create_array( type.get());
            mi::Size n;
            SERIAL::read( deserializer, &n);
            result->set_size( n); //-V522 PVS
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<IValue> element( deserialize( deserializer));
                result->set_value( i, element.get()); //-V522 PVS
            }
            return result;
        }
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IType_struct> type(
                m_type_factory->deserialize<IType_struct>( deserializer));
            IValue_struct* result = create_struct( type.get());
            mi::Size n = type->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<IValue> element( deserialize( deserializer));
                result->set_value( i, element.get()); //-V522 PVS
            }
            return result;
        }
        case IValue::VK_TEXTURE: {
            mi::base::Handle<const IType_texture> type(
                m_type_factory->deserialize<IType_texture>( deserializer));
            DB::Tag value;
            SERIAL::read( deserializer, &value);
            std::string unresolved_file_path;
            SERIAL::read( deserializer, &unresolved_file_path);
            std::string owner_module;
            SERIAL::read( deserializer, &owner_module);
            mi::Float32 gamma;
            SERIAL::read( deserializer, &gamma);
            std::string selector;
            SERIAL::read( deserializer, &selector);
            const char* selector_cstr = !selector.empty() ? selector.c_str() : nullptr;
            IValue* result = create_texture( type.get(),
                value, unresolved_file_path.c_str(), owner_module.c_str(), gamma, selector_cstr);
            return result;
        }
        case IValue::VK_LIGHT_PROFILE: {
            DB::Tag value;
            SERIAL::read( deserializer, &value);
            std::string unresolved_file_path;
            SERIAL::read( deserializer, &unresolved_file_path);
            std::string owner_module;
            SERIAL::read( deserializer, &owner_module);
            IValue* result = create_light_profile(
                value, unresolved_file_path.c_str(), owner_module.c_str());
            return result;
        }
        case IValue::VK_BSDF_MEASUREMENT: {
            DB::Tag value;
            SERIAL::read( deserializer, &value);
            std::string unresolved_file_path;
            SERIAL::read( deserializer, &unresolved_file_path);
            std::string owner_module;
            SERIAL::read( deserializer, &owner_module);
            IValue* result = create_bsdf_measurement(
                value, unresolved_file_path.c_str(), owner_module.c_str());
            return result;
        }
        case IValue::VK_INVALID_DF: {
            mi::base::Handle<const IType_reference> type(
                m_type_factory->deserialize<IType_reference>( deserializer));
            return create_invalid_df( type.get());
        }
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

void Value_factory::serialize_list(
    SERIAL::Serializer* serializer, const IValue_list* list) const
{
    const auto* list_impl = static_cast<const Value_list*>( list);

    write( serializer, list_impl->m_index_name);

    mi::Size size = list_impl->m_values.size();
    SERIAL::write( serializer, size);
    for( mi::Size i = 0; i < size; ++i)
        serialize( serializer, list_impl->m_values[i].get());
}

IValue_list* Value_factory::deserialize_list( SERIAL::Deserializer* deserializer) const
{
    auto* list_impl = new Value_list( /*initial_capacity*/ 0);

    read( deserializer, &list_impl->m_index_name);

    mi::Size size;
    SERIAL::read( deserializer, &size);
    list_impl->m_values.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_values[i] = deserialize( deserializer);

    return list_impl;
}

} // namespace MDL

} // namespace MI

