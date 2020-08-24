/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IStructure implementation.
 **/

#include "pch.h"

#include "i_neuray_attribute_context.h"
#include "neuray_attribute_set_impl_helper.h"
#include "neuray_class_factory.h"
#include "neuray_structure_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_utilities.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure_decl.h>

#include <sstream>
#include <boost/core/ignore_unused.hpp>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

extern mi::neuraylib::IFactory* s_factory;

mi::base::IInterface* Structure_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;
    mi::base::Handle<const mi::IStructure_decl> istructure_decl(
        argv[0]->get_interface<mi::IStructure_decl>());
    if( !istructure_decl.is_valid_interface())
        return nullptr;
    mi::base::Handle<const mi::IString> itype_name( argv[1]->get_interface<mi::IString>());
    if( !itype_name.is_valid_interface())
        return nullptr;
    const char* type_name = itype_name->get_c_str();

    Structure_impl* structure = new Structure_impl( transaction, istructure_decl.get(), type_name);
    if( !structure->successfully_constructed()) {
        structure->release();
        return nullptr;
    } else
        return structure;
}

Structure_impl::Structure_impl(
    mi::neuraylib::ITransaction* transaction,
    const mi::IStructure_decl* structure_decl,
    const char* type_name)
{
    // transaction might be NULL
    m_transaction = make_handle_dup( static_cast<Transaction_impl*>( transaction));

    m_structure_decl
        = mi::base::Handle<const mi::IStructure_decl>( structure_decl, mi::base::DUP_INTERFACE);
    m_type_name = type_name;

    m_length = structure_decl->get_length();
    m_index_to_key.resize( m_length);
    m_member.resize( m_length);

    m_successfully_constructed = true;

    for( mi::Size i = 0; i < m_length; ++i) {
        std::string member_name = structure_decl->get_member_name( i);
        std::string member_type_name = structure_decl->get_member_type_name( i);
        if( member_type_name == "Interface")
            member_type_name = "Void";

        m_key_to_index[member_name] = i;
        m_index_to_key[i] = member_name;
        mi::base::IInterface* element = s_class_factory->create_type_instance(
            m_transaction.get(), member_type_name.c_str());
        if( !element) {
            m_successfully_constructed = false;
            return;
        }
        m_member[i] = element;
    }
}

Structure_impl::~Structure_impl()
{
    std::vector<mi::base::IInterface*>::iterator it = m_member.begin();
    std::vector<mi::base::IInterface*>::iterator end = m_member.end();
    for ( ; it != end; ++it)
        if( *it)
            (*it)->release();
}

const char* Structure_impl::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Size Structure_impl::get_length() const
{
    return m_length;
}

const char* Structure_impl::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
    return m_cached_key.c_str();
}

bool Structure_impl::has_key( const char* key) const
{
    mi::Size index;
    return key_to_index( key, index);
}

const mi::base::IInterface* Structure_impl::get_value( const char* key) const
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_value( index);
}

mi::base::IInterface* Structure_impl::get_value( const char* key)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_value( index);
}

const mi::base::IInterface* Structure_impl::get_value( mi::Size index) const
{
    if( index >= m_length)
        return nullptr;

    m_member[index]->retain();
    return m_member[index];
}

mi::base::IInterface* Structure_impl::get_value( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    m_member[index]->retain();
    return m_member[index];
}

mi::Sint32 Structure_impl::set_value( const char* key, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    mi::Size index;
    if( !key_to_index( key, index))
        return -2;
    if( !has_correct_value_type( index, value))
        return -3;

    m_member[index]->release();
    m_member[index] = value;
    m_member[index]->retain();
    return 0;
}

mi::Sint32 Structure_impl::set_value( mi::Size index, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    if( index >= m_length)
        return -2;
    if( !has_correct_value_type( index, value))
        return -3;

    m_member[index]->release();
    m_member[index] = value;
    m_member[index]->retain();
    return 0;
}

const mi::IStructure_decl* Structure_impl::get_structure_decl() const
{
    m_structure_decl->retain();
    return m_structure_decl.get();
}

mi::neuraylib::ITransaction* Structure_impl::get_transaction() const
{
    if( m_transaction.is_valid_interface())
        m_transaction->retain();
    return m_transaction.get();
}

bool Structure_impl::has_correct_value_type(
    mi::Size index, const mi::base::IInterface* value) const
{
    if( !value)
        return false;

    std::string value_type_name = m_structure_decl->get_member_type_name( index);
    if( value_type_name == "Interface")
        return true;

    mi::base::Handle<const mi::IData> data( value->get_interface<mi::IData>());
    if( !data.is_valid_interface())
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return value_type_name == type_name;
}

bool Structure_impl::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_length)
        return false;

    key = m_index_to_key[index];
    return true;
}

bool Structure_impl::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    std::map<std::string, mi::Size>::const_iterator it = m_key_to_index.find( key);
    if( it == m_key_to_index.end())
        return false;

    index = it->second;
    return true;
}


mi::base::IInterface* Structure_impl_proxy::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 3)
        return nullptr;
    mi::base::Handle<const mi::IStructure_decl> istructure_decl(
        argv[0]->get_interface<mi::IStructure_decl>());
    if( !istructure_decl.is_valid_interface())
        return nullptr;
    mi::base::Handle<const mi::IString> itype_name( argv[1]->get_interface<mi::IString>());
    if( !itype_name.is_valid_interface())
        return nullptr;
    const char* type_name = itype_name->get_c_str();
    mi::base::Handle<const mi::IString> iattribute_name( argv[2]->get_interface<mi::IString>());
    if( !iattribute_name.is_valid_interface())
        return nullptr;
    const char* attribute_name = iattribute_name->get_c_str();
    return (new Structure_impl_proxy(
        transaction, istructure_decl.get(), type_name, attribute_name))->cast_to_major();
}

Structure_impl_proxy::Structure_impl_proxy(
    mi::neuraylib::ITransaction* transaction,
    const mi::IStructure_decl* structure_decl,
    const char* type_name,
    const char* attribute_name)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( static_cast<Transaction_impl*>( transaction));

    m_structure_decl
        = mi::base::Handle<const mi::IStructure_decl>( structure_decl, mi::base::DUP_INTERFACE);
    m_type_name = type_name;
    m_attribute_name = attribute_name;

    m_length = structure_decl->get_length();
    m_index_to_key.resize( m_length);

    for( mi::Size i = 0; i < m_length; ++i) {
        std::string member_name = structure_decl->get_member_name( i);
        m_key_to_index[member_name] = i;
        m_index_to_key[i] = member_name;
    }
}

const char* Structure_impl_proxy::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Size Structure_impl_proxy::get_length() const
{
    return m_length;
}

const char* Structure_impl_proxy::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
    return m_cached_key.c_str();
}

bool Structure_impl_proxy::has_key( const char* key) const
{
    mi::Size index;
    return key_to_index( key, index);
}

const mi::base::IInterface* Structure_impl_proxy::get_value( const char* key) const
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_value( index);
}

mi::base::IInterface* Structure_impl_proxy::get_value( const char* key)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_value( index);
}

const mi::base::IInterface* Structure_impl_proxy::get_value( mi::Size index) const
{
    if( index >= m_length)
        return nullptr;

    std::ostringstream s;
    s << m_attribute_name << "." << m_structure_decl->get_member_name( index);

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    return Attribute_set_impl_helper::get_attribute( attribute_context.get(), s.str());
}

mi::base::IInterface* Structure_impl_proxy::get_value( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    std::ostringstream s;
    s << m_attribute_name << "." << m_structure_decl->get_member_name( index);

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    return Attribute_set_impl_helper::get_attribute( attribute_context.get(), s.str());
}

mi::Sint32 Structure_impl_proxy::set_value( const char* key, mi::base::IInterface* value)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return -2;

    return set_value( index, value);
}

mi::Sint32 Structure_impl_proxy::set_value( mi::Size index, mi::base::IInterface* value)
{
    if( index >= m_length)
        return -1;

    if( !value)
        return -2;
    mi::base::Handle<const mi::IData> new_data( value->get_interface<mi::IData>());
    if( !new_data.is_valid_interface())
        return -2;
    std::string new_data_type_name = new_data->get_type_name();
    if( std::string( m_structure_decl->get_member_type_name( index)) != new_data_type_name)
        return -2;

    mi::base::Handle<mi::base::IInterface> old_element( get_value( index));
    mi::base::Handle<mi::IData> old_data( old_element->get_interface<mi::IData>());
    ASSERT( M_NEURAY_API, old_data.is_valid_interface());
    mi::Uint32 result = s_factory->assign_from_to( new_data.get(), old_data.get());
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);

    return 0;
}

const mi::IStructure_decl* Structure_impl_proxy::get_structure_decl() const
{
    m_structure_decl->retain();
    return m_structure_decl.get();
}

void Structure_impl_proxy::set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner)
{
    m_owner = mi::base::Handle<const mi::base::IInterface>( owner, mi::base::DUP_INTERFACE);
}

void Structure_impl_proxy::release_referenced_memory()
{
    for( mi::Size i = 0; i < m_length; ++i) {
        mi::base::Handle<mi::base::IInterface> element( get_value( i));
        mi::base::Handle<IProxy> proxy( element->get_interface<IProxy>());
        proxy->release_referenced_memory();
    }
}

mi::neuraylib::ITransaction* Structure_impl_proxy::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

bool Structure_impl_proxy::has_correct_value_type(
    mi::Size index, const mi::base::IInterface* value) const
{
    if( !value)
        return false;

    std::string value_type_name = m_structure_decl->get_member_type_name( index);

    mi::base::Handle<const mi::IData> data( value->get_interface<mi::IData>());
    if( !data.is_valid_interface())
        return false;
    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return value_type_name == type_name;
}

bool Structure_impl_proxy::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_length)
        return false;

    key = m_index_to_key[index];
    return true;
}

bool Structure_impl_proxy::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    std::map<std::string, mi::Size>::const_iterator it = m_key_to_index.find( key);
    if( it == m_key_to_index.end())
        return false;

    index = it->second;
    return true;
}


} // namespace NEURAY

} // namespace MI

