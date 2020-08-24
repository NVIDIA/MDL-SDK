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
 ** \brief Source for the Map_impl implementation.
 **/

#include "pch.h"

#include "neuray_map_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>

#include "neuray_class_factory.h"
#include "neuray_transaction_impl.h"

#include <boost/core/ignore_unused.hpp>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Map_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* value_type_name = istring->get_c_str();
    Map_impl* map = new Map_impl( transaction, value_type_name);
    if( !map->successfully_constructed()) {
        map->release();
        return nullptr;
    } else
        return map;
}

Map_impl::Map_impl( mi::neuraylib::ITransaction* transaction, const char* value_type_name)
{
    // transaction might be NULL
    m_transaction = make_handle_dup( transaction);

    ASSERT( M_NEURAY_API, value_type_name);
    m_value_type_name = value_type_name;

    m_type_name = "Map<" + m_value_type_name + ">";

    std::string mangled_value_type_name
        = (m_value_type_name == "Interface") ? "Void" : m_value_type_name;
    mi::base::Handle<mi::base::IInterface> element( s_class_factory->create_type_instance(
        static_cast<Transaction_impl*>( transaction), mangled_value_type_name.c_str(), 0, nullptr));
    m_successfully_constructed = element.is_valid_interface();

    m_cache_valid = false;
    m_cached_index = 0; // avoid cppcheck warning
}

Map_impl::~Map_impl()
{
    clear();
}

const char* Map_impl::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Size Map_impl::get_length() const
{
    return m_map.size();
}

const char* Map_impl::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
    return m_cached_key.c_str();
}

bool Map_impl::has_key( const char* key) const
{
    if( !key)
        return false;

    m_map_type::const_iterator it = m_map.find( key);
    return it != m_map.end();
}

const mi::base::IInterface* Map_impl::get_value( const char* key) const
{
    if( !key)
        return nullptr;

    m_map_type::const_iterator it = m_map.find( key);
    if( it == m_map.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

mi::base::IInterface* Map_impl::get_value( const char* key)
{
    if( !key)
        return nullptr;

    m_map_type::iterator it = m_map.find( key);
    if( it == m_map.end())
        return nullptr;

    it->second->retain();
    return it->second;
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

    m_map_type::iterator it = m_map.find( key);
    if( it == m_map.end())
        return -2;

    if( !has_correct_value_type( value))
        return -3;

    it->second->release();
    it->second = value;
    it->second->retain();
    return 0;
}

mi::Sint32 Map_impl::set_value( mi::Size index, mi::base::IInterface* value)
{
    std::string key;
    if( !index_to_key( index, key))
        return -2;

    return set_value( key.c_str(), value);
}

bool Map_impl::empty() const
{
    return m_map.empty();
}

void Map_impl::clear()
{
    m_cache_valid = false;

    m_map_type::iterator it  = m_map.begin();
    m_map_type::iterator end = m_map.end();
    for ( ; it != end; ++it)
        it->second->release();

    m_map.clear();
}

mi::Sint32 Map_impl::insert( const char* key, mi::base::IInterface* value)
{
    if( !key || !value)
        return -1;

    m_map_type::iterator it = m_map.find( key);
    if( it != m_map.end())
        return -2;

    if( !has_correct_value_type( value))
        return -3;

    m_cache_valid = false;

    value->retain();
    m_map[key] = value;
    return 0;
}

mi::Sint32 Map_impl::erase( const char* key)
{
    if( !key)
        return -1;

    m_map_type::iterator it = m_map.find( key);
    if( it == m_map.end())
        return -2;

    m_cache_valid = false;

    it->second->release();
    m_map.erase( it);
    return 0;
}

mi::neuraylib::ITransaction* Map_impl::get_transaction() const
{
    if( m_transaction.is_valid_interface())
        m_transaction->retain();
    return m_transaction.get();
}

bool Map_impl::has_correct_value_type( const mi::base::IInterface* value) const
{
    if( !value)
        return false;

    if( m_value_type_name == "Interface")
        return true;

    mi::base::Handle<const mi::IData> data( value->get_interface<mi::IData>());
    if( !data.is_valid_interface())
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

    m_map_type::const_iterator it;
    mi::Size i;
    if( m_cache_valid && index >= m_cached_index) {
        it = m_cached_iterator;
        i  = m_cached_index;
    } else {
        it = m_map.begin();
        i  = 0;
    }
    ASSERT( M_NEURAY_API, i <= index);
    ASSERT( M_NEURAY_API, it != m_map.end());

    for( ; i < index; ++i, ++it)
        ;

    m_cache_valid     = true;
    m_cached_iterator = it;
    m_cached_index    = index;

    key = it->first;
    return true;
}

} // namespace NEURAY

} // namespace MI
