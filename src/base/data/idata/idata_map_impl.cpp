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
 ** \brief Source for the IMap implementations.
 **/

#include "pch.h"

#include "idata_map_impl.h"

#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/istring.h>

#include <boost/core/ignore_unused.hpp>
#include <base/system/main/i_assert.h>

#include "i_idata_factory.h"

namespace MI {

namespace IDATA {

mi::base::IInterface* Map_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;

    mi::base::Handle string( argv[0]->get_interface<mi::IString>());
    if( !string)
        return nullptr;

    const char* value_type_name = string->get_c_str();
    mi::base::Handle map( new Map_impl( factory, transaction, value_type_name));
    return map->successfully_constructed() ? map.extract() : nullptr;
}

Map_impl::Map_impl(
    const Factory* factory, DB::Transaction* transaction, const char* value_type_name)
  : m_transaction( transaction),
    m_value_type_name( value_type_name),
    m_type_name( "Map<" + m_value_type_name + '>')
{
    MI_ASSERT( value_type_name);

    std::string mangled_value_type_name
        = (m_value_type_name == "Interface") ? "Void" : m_value_type_name;
    mi::base::Handle<mi::base::IInterface> element( factory->create(
        transaction, mangled_value_type_name.c_str()));
    m_successfully_constructed = !!element;
}

Map_impl::~Map_impl()
{
    clear();
}

const char* Map_impl::get_key( mi::Size index) const
{
    if( !index_to_key( index, m_cached_key))
        return nullptr;
    return m_cached_key.c_str();
}

bool Map_impl::has_key( const char* key) const
{
    if( !key)
        return false;

    auto it = m_map.find( key);
    return it != m_map.end();
}

const mi::base::IInterface* Map_impl::get_value( const char* key) const
{
    if( !key)
        return nullptr;

    auto it = m_map.find( key);
    if( it == m_map.end())
        return nullptr;

    it->second->retain();
    return it->second.get();
}

mi::base::IInterface* Map_impl::get_value( const char* key)
{
    if( !key)
        return nullptr;

    auto it = m_map.find( key);
    if( it == m_map.end())
        return nullptr;

    it->second->retain();
    return it->second.get();
}

const mi::base::IInterface* Map_impl::get_value( mi::Size index) const
{
   std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    return get_value( key.c_str());
}

mi::base::IInterface* Map_impl::get_value( mi::Size index)
{
   std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    return get_value( key.c_str());
}

mi::Sint32 Map_impl::set_value( const char* key, mi::base::IInterface* value)
{
    if( !key || !value)
        return -1;

    auto it = m_map.find( key);
    if( it == m_map.end())
        return -2;

    if( !has_correct_value_type( value))
        return -3;

    it->second = make_handle_dup( value);
    return 0;
}

mi::Sint32 Map_impl::set_value( mi::Size index, mi::base::IInterface* value)
{
    std::string key;
    if( !index_to_key( index, key))
        return -2;

    return set_value( key.c_str(), value);
}

void Map_impl::clear()
{
    m_cache_valid = false;
    m_map.clear();
}

mi::Sint32 Map_impl::insert( const char* key, mi::base::IInterface* value)
{
    if( !key || !value)
        return -1;

    auto it = m_map.find( key);
    if( it != m_map.end())
        return -2;

    if( !has_correct_value_type( value))
        return -3;

    m_cache_valid = false;

    m_map[key] = make_handle_dup( value);
    return 0;
}

mi::Sint32 Map_impl::erase( const char* key)
{
    if( !key)
        return -1;

    auto it = m_map.find( key);
    if( it == m_map.end())
        return -2;

    m_cache_valid = false;

    m_map.erase( it);
    return 0;
}

bool Map_impl::has_correct_value_type( const mi::base::IInterface* value) const
{
    if( !value)
        return false;

    if( m_value_type_name == "Interface")
        return true;

    mi::base::Handle data( value->get_interface<mi::IData>());
    if( !data)
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_value_type_name == type_name;
}

bool Map_impl::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_map.size())
        return false;

    Map_type::const_iterator it;
    mi::Size i;
    if( m_cache_valid && index >= m_cached_index) {
        it = m_cached_iterator;
        i  = m_cached_index;
    } else {
        it = m_map.begin();
        i  = 0;
    }
    MI_ASSERT( i <= index);
    MI_ASSERT( it != m_map.end());

    for( ; i < index; ++i, ++it) {
        MI_ASSERT( it != m_map.end());
    }

    m_cache_valid     = true;
    m_cached_iterator = it;
    m_cached_index    = index;

    key = it->first;
    return true;
}

} // namespace IDATA

} // namespace MI
