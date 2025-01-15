/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IStructure and IStructure_decl implementations.
 **/

#include "pch.h"

#include "idata_structure_impl.h"

#include <boost/core/ignore_unused.hpp>

#include <mi/neuraylib/istring.h>

#include <base/system/main/i_assert.h>

#include "i_idata_factory.h"

namespace MI {

namespace IDATA {

mi::base::IInterface* Structure_decl_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Structure_decl_impl();
}

mi::Sint32 Structure_decl_impl::add_member( const char* type_name, const char* name)
{
    if( !type_name || !name)
        return -1;

    for( auto& n : m_names)
        if( n == name)
            return -2;

    m_type_names.emplace_back( type_name);
    m_names.emplace_back( name);
    return 0;
}

mi::Sint32 Structure_decl_impl::remove_member( const char* name)
{
    if( !name)
        return -1;

    for( mi::Size i = 0; i < m_names.size(); ++i)
        if( m_names[i] == name) {
            m_type_names.erase( m_type_names.begin() + i);
            m_names.erase( m_names.begin() + i);
            return 0;
        }

    return -2;
}

const char* Structure_decl_impl::get_member_type_name( mi::Size index) const
{
    if( index >= m_type_names.size())
        return nullptr;

    return m_type_names[index].c_str();
}

const char* Structure_decl_impl::get_member_type_name( const char* name) const
{
    if( !name)
        return nullptr;

    for( mi::Size i = 0; i < m_names.size(); ++i)
        if( m_names[i] == name)
            return m_type_names[i].c_str();

    return nullptr;
}

const char* Structure_decl_impl::get_member_name( mi::Size index) const
{
    if( index >= m_names.size())
        return nullptr;

    return m_names[index].c_str();
}

const char* Structure_decl_impl::get_structure_type_name() const
{
    if( m_structure_type_name.empty())
        return nullptr;

    return m_structure_type_name.c_str();
}

void Structure_decl_impl::set_structure_type_name( const char* structure_type_name)
{
    MI_ASSERT( structure_type_name);

    m_structure_type_name = structure_type_name;
}

mi::base::IInterface* Structure_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;

    mi::base::Handle structure_decl( argv[0]->get_interface<mi::IStructure_decl>());
    if( !structure_decl)
        return nullptr;

    mi::base::Handle type_name( argv[1]->get_interface<mi::IString>());
    if( !type_name)
        return nullptr;

    const char* type_name_cstr = type_name->get_c_str();
    mi::base::Handle structure( new Structure_impl(
        factory, transaction, structure_decl.get(), type_name_cstr));
    return structure->successfully_constructed() ? structure.extract() : nullptr;
}

Structure_impl::Structure_impl(
    const Factory* factory,
    DB::Transaction* transaction,
    const mi::IStructure_decl* structure_decl,
    const char* type_name)
  : m_transaction( transaction),
    m_structure_decl( make_handle_dup( structure_decl)),
    m_type_name( type_name),
    m_length( structure_decl->get_length())
{
    m_index_to_key.resize( m_length);
    m_member.resize( m_length);

    for( mi::Size i = 0; i < m_length; ++i) {

        std::string member_name = structure_decl->get_member_name( i);
        std::string member_type_name = structure_decl->get_member_type_name( i);
        if( member_type_name == "Interface")
            member_type_name = "Void";

        m_key_to_index[member_name] = i;
        m_index_to_key[i] = member_name;
        m_member[i] = factory->create( transaction, member_type_name.c_str());
        if( !m_member[i]) {
            m_successfully_constructed = false;
            return;
        }
    }

    m_successfully_constructed = true;
}

Structure_impl::~Structure_impl() = default;

const char* Structure_impl::get_key( mi::Size index) const
{
    if( !index_to_key( index, m_cached_key))
        return nullptr;
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
    return m_member[index].get();
}

mi::base::IInterface* Structure_impl::get_value( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    m_member[index]->retain();
    return m_member[index].get();
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

    m_member[index] = make_handle_dup( value);
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

    m_member[index] = make_handle_dup( value);
    return 0;
}

const mi::IStructure_decl* Structure_impl::get_structure_decl() const
{
    m_structure_decl->retain();
    return m_structure_decl.get();
}

bool Structure_impl::has_correct_value_type(
    mi::Size index, const mi::base::IInterface* value) const
{
    if( !value)
        return false;

    std::string value_type_name = m_structure_decl->get_member_type_name( index);
    if( value_type_name == "Interface")
        return true;

    mi::base::Handle data( value->get_interface<mi::IData>());
    if( !data)
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
    auto it = m_key_to_index.find( key);
    if( it == m_key_to_index.end())
        return false;

    index = it->second;
    return true;
}

mi::base::IInterface* Structure_impl_proxy::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;

    if( argc != 3)
        return nullptr;

    mi::base::Handle structure_decl( argv[0]->get_interface<mi::IStructure_decl>());
    if( !structure_decl)
        return nullptr;

    mi::base::Handle type_name( argv[1]->get_interface<mi::IString>());
    if( !type_name)
        return nullptr;

    const char* type_name_cstr = type_name->get_c_str();
    mi::base::Handle attribute_name( argv[2]->get_interface<mi::IString>());
    if( !attribute_name)
        return nullptr;

    const char* attribute_name_cstr = attribute_name->get_c_str();
    return (new Structure_impl_proxy(
            factory, transaction, structure_decl.get(), type_name_cstr, attribute_name_cstr))
        ->cast_to_major();
}

Structure_impl_proxy::Structure_impl_proxy(
    const Factory* factory,
    DB::Transaction* transaction,
    const mi::IStructure_decl* structure_decl,
    const char* type_name,
    const char* attribute_name)
  : m_factory( factory),
    m_transaction( transaction),
    m_structure_decl( make_handle_dup( structure_decl)),
    m_type_name( type_name),
    m_attribute_name( attribute_name),
    m_length( structure_decl->get_length())
{
    MI_ASSERT( transaction);

    m_index_to_key.resize( m_length);

    for( mi::Size i = 0; i < m_length; ++i) {
        std::string member_name = structure_decl->get_member_name( i);
        m_key_to_index[member_name] = i;
        m_index_to_key[i] = member_name;
    }
}

const char* Structure_impl_proxy::get_key( mi::Size index) const
{
    if( !index_to_key( index, m_cached_key))
        return nullptr;
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

    std::string s( m_attribute_name);
    s += '.';
    s += m_structure_decl->get_member_name( index);

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    return attribute_context->get_attribute( s.c_str());
}

mi::base::IInterface* Structure_impl_proxy::get_value( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    std::string s( m_attribute_name);
    s += '.';
    s += m_structure_decl->get_member_name( index);

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    return attribute_context->get_attribute( s.c_str());
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

    mi::base::Handle new_data( value->get_interface<mi::IData>());
    if( !new_data)
        return -2;

    std::string new_data_type_name = new_data->get_type_name();
    if( std::string( m_structure_decl->get_member_type_name( index)) != new_data_type_name)
        return -2;

    mi::base::Handle old_data( get_value<mi::IData>( index));
    MI_ASSERT( old_data);
    mi::Uint32 result = m_factory->assign_from_to( new_data.get(), old_data.get(), /*options*/ 0);
    MI_ASSERT( result == 0);
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
    m_owner = make_handle_dup( owner);
}

void Structure_impl_proxy::release_referenced_memory()
{
    for( mi::Size i = 0; i < m_length; ++i) {
        mi::base::Handle proxy( get_value<IProxy>( i));
        proxy->release_referenced_memory();
    }
}

bool Structure_impl_proxy::has_correct_value_type(
    mi::Size index, const mi::base::IInterface* value) const
{
    if( !value)
        return false;

    std::string value_type_name = m_structure_decl->get_member_type_name( index);

    mi::base::Handle data( value->get_interface<mi::IData>());
    if( !data)
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

    auto it = m_key_to_index.find( key);
    if( it == m_key_to_index.end())
        return false;

    index = it->second;
    return true;
}

} // namespace IDATA

} // namespace MI
