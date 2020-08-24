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

/** \file
 ** \brief Source for the IValue implementation.
 **/

#include "pch.h"

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>

#include "neuray_transaction_impl.h"
#include "neuray_value_impl.h"

namespace MI {

namespace NEURAY {

MDL::IValue* get_internal_value( mi::neuraylib::IValue* value)
{
    if( !value)
        return nullptr;
    mi::base::Handle<IValue_wrapper> value_wrapper( value->get_interface<IValue_wrapper>());
    if( !value_wrapper)
        return nullptr;
    return value_wrapper->get_internal_value();
}

const MDL::IValue* get_internal_value( const mi::neuraylib::IValue* value)
{
    if( !value)
        return nullptr;
    mi::base::Handle<const IValue_wrapper> value_wrapper( value->get_interface<IValue_wrapper>());
    if( !value_wrapper)
        return nullptr;
    return value_wrapper->get_internal_value();
}

const MDL::IValue_list* get_internal_value_list( const mi::neuraylib::IValue_list* value_list)
{
    if( !value_list)
        return nullptr;
    mi::base::Handle<const IValue_list_wrapper> value_list_wrapper(
        value_list->get_interface<IValue_list_wrapper>());
    if( !value_list_wrapper)
        return nullptr;
    return value_list_wrapper->get_internal_value_list();
}

template <class E, class I, class T>
Value_base<E, I, T>::~Value_base() { }

template <class E, class I, class T>
const T* Value_base<E, I, T>::get_type() const
{
    mi::base::Handle<const MDL::IType> result_int( m_value->get_type());
    mi::base::Handle<Type_factory> tf( static_cast<Type_factory*>( m_vf->get_type_factory()));
    return tf->create<T>( result_int.get(), m_owner.get());
}

const mi::neuraylib::IValue_atomic* Value_vector::get_value( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue_atomic> result_int( m_value->get_value( index));
    return m_vf->create<mi::neuraylib::IValue_atomic>( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue_atomic* Value_vector::get_value( mi::Size index)
{
    mi::base::Handle<MDL::IValue_atomic> result_int( m_value->get_value( index));
    return m_vf->create<mi::neuraylib::IValue_atomic>( result_int.get(), m_owner.get());
}

mi::Sint32 Value_vector::set_value( mi::Size index, mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value->set_value( index, value_int.get());
}

const mi::neuraylib::IValue_vector* Value_matrix::get_value( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue_vector> result_int( m_value->get_value( index));
    return m_vf->create<mi::neuraylib::IValue_vector>( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue_vector* Value_matrix::get_value( mi::Size index)
{
    mi::base::Handle<MDL::IValue_vector> result_int( m_value->get_value( index));
    return m_vf->create<mi::neuraylib::IValue_vector>( result_int.get(), m_owner.get());
}

mi::Sint32 Value_matrix::set_value( mi::Size index, mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value->set_value( index, value_int.get());
}

const mi::neuraylib::IValue_float* Value_color::get_value( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue_float> result_int( m_value->get_value( index));
    return m_vf->create<mi::neuraylib::IValue_float>( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue_float* Value_color::get_value( mi::Size index)
{
    mi::base::Handle<MDL::IValue_float> result_int( m_value->get_value( index));
    return m_vf->create<mi::neuraylib::IValue_float>( result_int.get(), m_owner.get());
}

mi::Sint32 Value_color::set_value( mi::Size index, mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    mi::base::Handle<MDL::IValue_float> value_int_float(
        value_int->get_interface<MDL::IValue_float>());
    return m_value->set_value( index, value_int_float.get());
}

mi::Sint32 Value_color::set_value( mi::Size index, mi::neuraylib::IValue_float* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    mi::base::Handle<MDL::IValue_float> value_int_float(
        value_int->get_interface<MDL::IValue_float>());
    return m_value->set_value( index, value_int_float.get());
}

const mi::neuraylib::IValue* Value_array::get_value( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue> result_int( m_value->get_value( index));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue* Value_array::get_value( mi::Size index)
{
    mi::base::Handle<MDL::IValue> result_int( m_value->get_value( index));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::Sint32 Value_array::set_value( mi::Size index, mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value->set_value( index, value_int.get());
}

const mi::neuraylib::IValue* Value_struct::get_value( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue> result_int( m_value->get_value( index));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue* Value_struct::get_value( mi::Size index)
{
    mi::base::Handle<MDL::IValue> result_int( m_value->get_value( index));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::Sint32 Value_struct::set_value( mi::Size index, mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value->set_value( index, value_int.get());
}

const mi::neuraylib::IValue* Value_struct::get_field( const char* name) const
{
    mi::base::Handle<const MDL::IValue> result_int( m_value->get_field( name));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue* Value_struct::get_field( const char* name)
{
    mi::base::Handle<MDL::IValue> result_int( m_value->get_field( name));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::Sint32 Value_struct::set_field( const char* name, mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value->set_field( name, value_int.get());
}

const char* Value_texture::get_value() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = m_value->get_value();
    return db_transaction->tag_to_name( tag);
}

mi::Sint32 Value_texture::set_value( const char* value)
{
    if( !value) {
        m_value->set_value( DB::Tag());
        return 0;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( value);
    if( !tag)
        return -1;
    if( db_transaction->get_class_id( tag) != TEXTURE::ID_TEXTURE)
        return -2;

    m_value->set_value( tag);
    return 0;
}

const char* Value_texture::get_file_path() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    return m_value->get_file_path( db_transaction);
}

mi::Float32 Value_texture::get_gamma() const
{
    DB::Tag texture_tag = m_value->get_value();
    if (texture_tag.is_valid()) {
        Transaction_impl* transaction_impl = static_cast<Transaction_impl*>(m_transaction.get());
        DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
        DB::Access<TEXTURE::Texture> tex(texture_tag, db_transaction);
        return tex->get_gamma();
    }
    return m_value->get_gamma();
}

const char* Value_light_profile::get_value() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = m_value->get_value();
    return db_transaction->tag_to_name( tag);
}

mi::Sint32 Value_light_profile::set_value( const char* value)
{
    if( !value) {
        m_value->set_value( DB::Tag());
        return 0;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( value);
    if( !tag)
        return -1;
    if( db_transaction->get_class_id( tag) != LIGHTPROFILE::ID_LIGHTPROFILE)
        return -2;

    m_value->set_value( tag);
    return 0;
}

const char* Value_light_profile::get_file_path() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    return m_value->get_file_path( db_transaction);
}

const char* Value_bsdf_measurement::get_value() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = m_value->get_value();
    return db_transaction->tag_to_name( tag);
}

mi::Sint32 Value_bsdf_measurement::set_value( const char* value)
{
    if( !value) {
        m_value->set_value( DB::Tag());
        return 0;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( value);
    if( !tag)
        return -1;
    if( db_transaction->get_class_id( tag) != BSDFM::ID_BSDF_MEASUREMENT)
        return -2;

    m_value->set_value( tag);
    return 0;
}

const char* Value_bsdf_measurement::get_file_path() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    return m_value->get_file_path( db_transaction);
}

const mi::neuraylib::IValue* Value_list::get_value( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue> result_int( m_value_list->get_value( index));
    return m_vf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IValue* Value_list::get_value( const char* name) const
{
    mi::base::Handle<const MDL::IValue> result_int( m_value_list->get_value( name));
    return m_vf->create( result_int.get(), m_owner.get());
}

mi::Sint32 Value_list::set_value( mi::Size index, const mi::neuraylib::IValue* value)
{
    mi::base::Handle<const MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value_list->set_value( index, value_int.get());
}

mi::Sint32 Value_list::set_value( const char* name, const mi::neuraylib::IValue* value)
{
    mi::base::Handle<const MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value_list->set_value( name, value_int.get());
}

mi::Sint32 Value_list::add_value( const char* name, const mi::neuraylib::IValue* value)
{
    mi::base::Handle<const MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_value_list->add_value( name, value_int.get());
}

MDL::IValue_list* Value_list::get_internal_value_list()
{
    m_value_list->retain();
    return m_value_list.get();
}

const MDL::IValue_list* Value_list::get_internal_value_list() const
{
    m_value_list->retain();
    return m_value_list.get();
}

mi::neuraylib::IValue_bool* Value_factory::create_bool( bool value) const
{
    mi::base::Handle<MDL::IValue_bool> result_int( m_vf->create_bool( value));
    return new Value_bool( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_int* Value_factory::create_int( mi::Sint32 value) const
{
    mi::base::Handle<MDL::IValue_int> result_int( m_vf->create_int( value));
    return new Value_int( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_enum* Value_factory::create_enum(
    const mi::neuraylib::IType_enum* type, mi::Size index) const
{
    if( !type)
        return nullptr;
    mi::base::Handle<const MDL::IType_enum> type_int(
        get_internal_type<MDL::IType_enum>( type));
    mi::base::Handle<MDL::IValue_enum> result_int( m_vf->create_enum( type_int.get(), index));
    return new Value_enum( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_float* Value_factory::create_float( mi::Float32 value) const
{
    mi::base::Handle<MDL::IValue_float> result_int( m_vf->create_float( value));
    return new Value_float( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_double* Value_factory::create_double( mi::Float64 value) const
{
    mi::base::Handle<MDL::IValue_double> result_int( m_vf->create_double( value));
    return new Value_double( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_string* Value_factory::create_string( const char* value) const
{
    mi::base::Handle<MDL::IValue_string> result_int( m_vf->create_string( value));
    return new Value_string( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_string_localized* Value_factory::create_string_localized( const char* value, const char* original) const
{
    mi::base::Handle<MDL::IValue_string_localized> result_int( m_vf->create_string_localized( value, original));
    return new Value_string_localized( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_vector* Value_factory::create_vector(
    const mi::neuraylib::IType_vector* type) const
{
    if( !type)
        return nullptr;
    mi::base::Handle<const MDL::IType_vector> type_int(
        get_internal_type<MDL::IType_vector>( type));
    mi::base::Handle<MDL::IValue_vector> result_int( m_vf->create_vector( type_int.get()));
    return new Value_vector( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_matrix* Value_factory::create_matrix(
    const mi::neuraylib::IType_matrix* type) const
{
    if( !type)
        return nullptr;
    mi::base::Handle<const MDL::IType_matrix> type_int(
        get_internal_type<MDL::IType_matrix>( type));
    mi::base::Handle<MDL::IValue_matrix> result_int( m_vf->create_matrix( type_int.get()));
    return new Value_matrix( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_color* Value_factory::create_color(
    mi::Float32 red, mi::Float32 green, mi::Float32 blue) const
{
    mi::base::Handle<MDL::IValue_color> result_int( m_vf->create_color( red, green, blue));
    return new Value_color( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_array* Value_factory::create_array(
    const mi::neuraylib::IType_array* type) const
{
    if( !type)
        return nullptr;

    mi::base::Handle<const MDL::IType_array> type_int(
        get_internal_type<MDL::IType_array>( type));
    mi::base::Handle<MDL::IValue_array> result_int( m_vf->create_array( type_int.get()));
    return new Value_array( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_struct* Value_factory::create_struct(
    const mi::neuraylib::IType_struct* type) const
{
    if( !type)
        return nullptr;

    mi::base::Handle<const MDL::IType_struct> type_int(
        get_internal_type<MDL::IType_struct>( type));
    mi::base::Handle<MDL::IValue_struct> result_int( m_vf->create_struct( type_int.get()));
    return new Value_struct( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_texture* Value_factory::create_texture(
    const mi::neuraylib::IType_texture* type,
    const char* value) const
{
    if( !type)
        return nullptr;

    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = value ? db_transaction->name_to_tag( value) : DB::Tag();
    if( value && !tag)
        return nullptr;

    mi::base::Handle<const MDL::IType_texture> type_int(
        get_internal_type<MDL::IType_texture>( type));
    mi::base::Handle<MDL::IValue_texture> result_int(
        m_vf->create_texture( type_int.get(), tag));
    return new Value_texture( this, m_transaction.get(), result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_light_profile* Value_factory::create_light_profile(
    const char* value) const
{
    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = value ? db_transaction->name_to_tag( value) : DB::Tag();
    if( value && !tag)
        return nullptr;

    mi::base::Handle<MDL::IValue_light_profile> result_int(
        m_vf->create_light_profile( tag));
    return new Value_light_profile( this, m_transaction.get(), result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_bsdf_measurement* Value_factory::create_bsdf_measurement(
    const char* value) const
{
    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = value ? db_transaction->name_to_tag( value) : DB::Tag();
    if( value && !tag)
        return nullptr;

    mi::base::Handle<MDL::IValue_bsdf_measurement> result_int(
        m_vf->create_bsdf_measurement( tag));
    return new Value_bsdf_measurement( this, m_transaction.get(), result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_invalid_df* Value_factory::create_invalid_df(
    const mi::neuraylib::IType_reference* type) const
{
    if( !type)
        return nullptr;

    mi::base::Handle<const MDL::IType_reference> type_int(
        get_internal_type<MDL::IType_reference>( type));
    mi::base::Handle<MDL::IValue_invalid_df> result_int(
        m_vf->create_invalid_df( type_int.get()));
    if( !result_int)
        return nullptr;

    return new Value_invalid_df( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue* Value_factory::create(
    const mi::neuraylib::IType* type) const
{
    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    mi::base::Handle<MDL::IValue> result_int( m_vf->create( type_int.get()));
    return create( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_list* Value_factory::create_value_list() const
{
    mi::base::Handle<MDL::IValue_list> result_int( m_vf->create_value_list());
    return create_value_list( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue* Value_factory::clone( const mi::neuraylib::IValue* value) const
{
    mi::base::Handle<const MDL::IValue> value_int( get_internal_value( value));
    mi::base::Handle<MDL::IValue> result_int( m_vf->clone( value_int.get()));
    return create( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_list* Value_factory::clone(
    const mi::neuraylib::IValue_list* value_list) const
{
    mi::base::Handle<const MDL::IValue_list> value_list_int( get_internal_value_list( value_list));
    mi::base::Handle<MDL::IValue_list> result_int( m_vf->clone( value_list_int.get()));
    return create_value_list( result_int.get(), /*owner*/ nullptr);
}

mi::Sint32 Value_factory::compare(
    const mi::neuraylib::IValue* lhs, const mi::neuraylib::IValue* rhs) const
{
    mi::base::Handle<const MDL::IValue> lhs_int( get_internal_value( lhs));
    mi::base::Handle<const MDL::IValue> rhs_int( get_internal_value( rhs));
    return m_vf->compare( lhs_int.get(), rhs_int.get());
}

mi::Sint32 Value_factory::compare(
    const mi::neuraylib::IValue_list* lhs, const mi::neuraylib::IValue_list* rhs) const
{
    mi::base::Handle<const MDL::IValue_list> lhs_int( get_internal_value_list( lhs));
    mi::base::Handle<const MDL::IValue_list> rhs_int( get_internal_value_list( rhs));
    return m_vf->compare( lhs_int.get(), rhs_int.get());
}

const mi::IString* Value_factory::dump(
    const mi::neuraylib::IValue* value, const char* name, mi::Size depth) const
{
    if( !value)
        return nullptr;

    mi::base::Handle<const MDL::IValue> value_int( get_internal_value( value));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_vf->dump( db_transaction, value_int.get(), name, depth);
}

const mi::IString* Value_factory::dump(
    const mi::neuraylib::IValue_list* list, const char* name, mi::Size depth) const
{
    if( !list)
        return nullptr;

    mi::base::Handle<const MDL::IValue_list> list_int( get_internal_value_list( list));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_vf->dump( db_transaction, list_int.get(), name, depth);
}

mi::neuraylib::IValue* Value_factory::create(
    MDL::IValue* value, const mi::base::IInterface* owner) const
{
    if( !value)
        return nullptr;

    MDL::IValue::Kind kind = value->get_kind();

    switch( kind) {
        case MDL::IValue::VK_BOOL: {
            mi::base::Handle<MDL::IValue_bool> v( value->get_interface<MDL::IValue_bool>());
            return new Value_bool( this, v.get(), owner);
        }
        case MDL::IValue::VK_INT: {
            mi::base::Handle<MDL::IValue_int> v( value->get_interface<MDL::IValue_int>());
            return new Value_int( this, v.get(), owner);
        }
        case MDL::IValue::VK_ENUM: {
            mi::base::Handle<MDL::IValue_enum> v( value->get_interface<MDL::IValue_enum>());
            return new Value_enum( this, v.get(), owner);
        }
        case MDL::IValue::VK_FLOAT: {
            mi::base::Handle<MDL::IValue_float> v( value->get_interface<MDL::IValue_float>());
            return new Value_float( this, v.get(), owner);
        }
        case MDL::IValue::VK_DOUBLE: {
            mi::base::Handle<MDL::IValue_double> v( value->get_interface<MDL::IValue_double>());
            return new Value_double( this, v.get(), owner);
        }
        case MDL::IValue::VK_STRING: {
            mi::base::Handle<MDL::IValue_string_localized> v_localized( value->get_interface<MDL::IValue_string_localized>());
            if( v_localized) {
                return new Value_string_localized( this, v_localized.get(), owner);
            }
            mi::base::Handle<MDL::IValue_string> v( value->get_interface<MDL::IValue_string>());
            return new Value_string( this, v.get(), owner);
        }
        case MDL::IValue::VK_VECTOR: {
            mi::base::Handle<MDL::IValue_vector> v( value->get_interface<MDL::IValue_vector>());
            return new Value_vector( this, v.get(), owner);
        }
        case MDL::IValue::VK_MATRIX: {
            mi::base::Handle<MDL::IValue_matrix> v( value->get_interface<MDL::IValue_matrix>());
            return new Value_matrix( this, v.get(), owner);
        }
        case MDL::IValue::VK_COLOR: {
            mi::base::Handle<MDL::IValue_color> v( value->get_interface<MDL::IValue_color>());
            return new Value_color( this, v.get(), owner);
        }
        case MDL::IValue::VK_ARRAY: {
            mi::base::Handle<MDL::IValue_array> v( value->get_interface<MDL::IValue_array>());
            return new Value_array( this, v.get(), owner);
        }
        case MDL::IValue::VK_STRUCT: {
            mi::base::Handle<MDL::IValue_struct> v( value->get_interface<MDL::IValue_struct>());
            return new Value_struct( this, v.get(), owner);
        }
        case MDL::IValue::VK_INVALID_DF: {
            mi::base::Handle<MDL::IValue_invalid_df> v(
                value->get_interface<MDL::IValue_invalid_df>());
            return new Value_invalid_df( this, v.get(), owner);
        }
        case MDL::IValue::VK_TEXTURE: {
            mi::base::Handle<MDL::IValue_texture> v(
                value->get_interface<MDL::IValue_texture>());
            return new Value_texture( this, m_transaction.get(), v.get(), owner);
        }
        case MDL::IValue::VK_LIGHT_PROFILE: {
            mi::base::Handle<MDL::IValue_light_profile> v(
                value->get_interface<MDL::IValue_light_profile>());
            return new Value_light_profile( this, m_transaction.get(), v.get(), owner);
        }
        case MDL::IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<MDL::IValue_bsdf_measurement> v(
                value->get_interface<MDL::IValue_bsdf_measurement>());
            return new Value_bsdf_measurement( this, m_transaction.get(), v.get(), owner);
        }
        case MDL::IValue::VK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    };

   ASSERT( M_SCENE, false);
   return nullptr;
}

const mi::neuraylib::IValue* Value_factory::create(
    const MDL::IValue* value, const mi::base::IInterface* owner) const
{
    return create( const_cast<MDL::IValue*>( value), owner);
}

mi::neuraylib::IValue_list* Value_factory::create_value_list(
    MDL::IValue_list* value_list, const mi::base::IInterface* owner) const
{
    if( !value_list)
        return nullptr;
    return new Value_list( this, value_list, owner);
}

const mi::neuraylib::IValue_list* Value_factory::create_value_list(
    const MDL::IValue_list* value_list, const mi::base::IInterface* owner) const
{
    return create_value_list( const_cast<MDL::IValue_list*>( value_list), owner);
}

DB::Transaction* Value_factory::get_db_transaction() const
{
    if( !m_transaction)
        return nullptr;

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    return transaction_impl->get_db_transaction();
}

} // namespace NEURAY

} // namespace MI
