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
 ** \brief Source for the Array_impl_proxy implementation.
 **/

#include "pch.h"

#include "i_neuray_attribute_context.h"
#include "neuray_array_impl_proxy.h"
#include "neuray_attribute_set_impl_helper.h"
#include "neuray_type_utilities.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>

#include <sstream>
#include <base/system/stlext/i_stlext_likely.h>
#include <boost/core/ignore_unused.hpp>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/data/attr/attr.h>

namespace MI {

namespace NEURAY {

extern mi::neuraylib::IFactory* s_factory;

mi::base::IInterface* Array_impl_proxy::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 3)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* element_type_name = istring->get_c_str();
    mi::base::Handle<const mi::ISize> ivalue( argv[1]->get_interface<mi::ISize>());
    if( !ivalue.is_valid_interface())
        return nullptr;
    mi::Size length = ivalue->get_value<mi::Size>();
    istring = argv[2]->get_interface<mi::IString>();
    if( !istring.is_valid_interface())
        return nullptr;
    const char* attribute_name = istring->get_c_str();
    return (new Array_impl_proxy(
        transaction, element_type_name, length, attribute_name))->cast_to_major();
}

Array_impl_proxy::Array_impl_proxy(
    mi::neuraylib::ITransaction* transaction,
    const char* element_type_name,
    mi::Size length,
    const char* attribute_name)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( transaction);

    m_element_type_name = element_type_name;
    set_length_internal( length);
    m_attribute_name = attribute_name;

    std::ostringstream s;
    s << length;
    m_type_name  = element_type_name;
    m_type_name += "[";
    m_type_name += s.str();
    m_type_name += "]";
}

const char* Array_impl_proxy::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Size Array_impl_proxy::get_length() const
{
    return m_length;
}

const char* Array_impl_proxy::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
    return m_cached_key.c_str();
}

bool Array_impl_proxy::has_key( const char* key) const
{
    mi::Size index;
    return key_to_index( key, index);
}

const mi::base::IInterface* Array_impl_proxy::get_value( const char* key) const
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_element( index);
}

mi::base::IInterface* Array_impl_proxy::get_value( const char* key)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_element( index);
}

const mi::base::IInterface* Array_impl_proxy::get_value( mi::Size index) const
{
    return get_element( index);
}

mi::base::IInterface* Array_impl_proxy::get_value( mi::Size index)
{
    return get_element( index);
}

mi::Sint32 Array_impl_proxy::set_value( const char* key, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    mi::Size index;
    if( !key_to_index( key, index))
        return -2;

    return (set_element( index, value) == 0) ? 0 : -3;
}

mi::Sint32 Array_impl_proxy::set_value( mi::Size index, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    if( index >= m_length)
        return -2;

    return (set_element( index, value) == 0) ? 0 : -3;
}

const mi::base::IInterface* Array_impl_proxy::get_element( mi::Size index) const
{
    if( index >= m_length)
        return nullptr;

    std::ostringstream s;
    s << m_attribute_name << "[" << index << "]";

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    return Attribute_set_impl_helper::get_attribute( attribute_context.get(), s.str());
}

mi::base::IInterface* Array_impl_proxy::get_element( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    std::ostringstream s;
    s << m_attribute_name << "[" << index << "]";

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    return Attribute_set_impl_helper::get_attribute( attribute_context.get(), s.str());
}

mi::Sint32 Array_impl_proxy::set_element(
    mi::Size index, mi::base::IInterface* element)
{
    if( index >= m_length)
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    mi::base::Handle<const mi::IData> new_data( element->get_interface<mi::IData>());
    ASSERT( M_NEURAY_API, new_data.is_valid_interface());

    mi::base::Handle<mi::base::IInterface> old_element( get_element( index));
    mi::base::Handle<mi::IData> old_data( old_element->get_interface<mi::IData>());
    ASSERT( M_NEURAY_API, old_data.is_valid_interface());
    mi::Uint32 result = s_factory->assign_from_to( new_data.get(), old_data.get());
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);

    return 0;
}

bool Array_impl_proxy::empty() const
{
    return m_length == 0;
}

void Array_impl_proxy::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_owner = make_handle_dup( owner);
}

void Array_impl_proxy::release_referenced_memory()
{
    for( mi::Size i = 0; i < m_length; ++i) {
        mi::base::Handle<mi::base::IInterface> element( get_value( i));
        mi::base::Handle<IProxy> proxy( element->get_interface<IProxy>());
        proxy->release_referenced_memory();
    }
}

mi::neuraylib::ITransaction* Array_impl_proxy::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

void Array_impl_proxy::set_length_internal( mi::Size length)
{
    m_length = length;
}

bool Array_impl_proxy::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_likely.get_status())
        return false;
    index = *index_likely.get_ptr(); //-V522 PVS
    return index < m_length;
}

bool Array_impl_proxy::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_length)
        return false;
    std::ostringstream s;
    s << index;
    key = s.str();
    return true;
}

bool Array_impl_proxy::has_correct_element_type( const mi::base::IInterface* element) const
{
    ASSERT( M_NEURAY_API, element);

    mi::base::Handle<const mi::IData> data( element->get_interface<mi::IData>());
    if( !data.is_valid_interface())
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_element_type_name == type_name;
}

mi::base::IInterface* Dynamic_array_impl_proxy::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 2)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* element_type_name = istring->get_c_str();
    istring = argv[1]->get_interface<mi::IString>();
    if( !istring.is_valid_interface())
        return nullptr;
    const char* attribute_name = istring->get_c_str();
    return (new Dynamic_array_impl_proxy(
        transaction, element_type_name, attribute_name))->cast_to_major();
}

Dynamic_array_impl_proxy::Dynamic_array_impl_proxy(
    mi::neuraylib::ITransaction* transaction,
    const char* element_type_name,
    const char* attribute_name)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( transaction);

    m_element_type_name = element_type_name;
    m_attribute_name = attribute_name;

    m_type_name  = element_type_name;
    m_type_name += "[]";

    m_pointer = nullptr;
}

Dynamic_array_impl_proxy::~Dynamic_array_impl_proxy()
{
    // check invariant
    ASSERT( M_NEURAY_API, static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);
}

const char* Dynamic_array_impl_proxy::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Size Dynamic_array_impl_proxy::get_length() const
{
    return m_length;
}

const char* Dynamic_array_impl_proxy::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
    return m_cached_key.c_str();
}

bool Dynamic_array_impl_proxy::has_key( const char* key) const
{
    mi::Size index;
    return key_to_index( key, index);
}

const mi::base::IInterface* Dynamic_array_impl_proxy::get_value( const char* key) const
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_element( index);
}

mi::base::IInterface* Dynamic_array_impl_proxy::get_value( const char* key)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_element( index);
}

const mi::base::IInterface* Dynamic_array_impl_proxy::get_value( mi::Size index) const
{
    return get_element( index);
}

mi::base::IInterface* Dynamic_array_impl_proxy::get_value( mi::Size index)
{
    return get_element( index);
}

mi::Sint32 Dynamic_array_impl_proxy::set_value( const char* key, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    mi::Size index;
    if( !key_to_index( key, index))
        return -2;

    return (set_element( index, value) == 0) ? 0 : -3;
}

mi::Sint32 Dynamic_array_impl_proxy::set_value( mi::Size index, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    if( index >= m_length)
        return -2;

    return (set_element( index, value) == 0) ? 0 : -3;
}

const mi::base::IInterface* Dynamic_array_impl_proxy::get_element( mi::Size index) const
{
    if( index >= m_length)
        return nullptr;

    std::ostringstream s;
    s << m_attribute_name << "[" << index << "]";

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    return Attribute_set_impl_helper::get_attribute( attribute_context.get(), s.str());
}

mi::base::IInterface* Dynamic_array_impl_proxy::get_element( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    std::ostringstream s;
    s << m_attribute_name << "[" << index << "]";

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    return Attribute_set_impl_helper::get_attribute( attribute_context.get(), s.str());
}

mi::Sint32 Dynamic_array_impl_proxy::set_element(
    mi::Size index, mi::base::IInterface* element)
{
    if( index >= m_length)
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    mi::base::Handle<const mi::IData> new_data( element->get_interface<mi::IData>());
    ASSERT( M_NEURAY_API, new_data.is_valid_interface());

    mi::base::Handle<mi::base::IInterface> old_element( get_element( index));
    mi::base::Handle<mi::IData> old_data( old_element->get_interface<mi::IData>());
    ASSERT( M_NEURAY_API, old_data.is_valid_interface());
    mi::Uint32 result = s_factory->assign_from_to( new_data.get(), old_data.get());
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);

    return 0;
}

bool Dynamic_array_impl_proxy::empty() const
{
    return m_length == 0;
}

void Dynamic_array_impl_proxy::set_length( mi::Size size)
{
    set_length_internal( size);
}

void Dynamic_array_impl_proxy::clear()
{
    set_length( 0);
}

mi::Sint32 Dynamic_array_impl_proxy::insert( mi::Size index, mi::base::IInterface* element)
{
    if( index > m_length) // note special case ">" instead of ">=" here
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    set_length( m_length+1);

    if( index < m_length) {
        // Shift entries [index, m_length-1) one slot to the back.
        // Note that m_length now reflects the new length already,
        // hence "-1" in the comment above and in the memmove call below.
        char* base = static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
        char* source = base +  index    * m_size_of_element;
        char* target = base + (index+1) * m_size_of_element;
        memmove( target, source, ((m_length-1) - index) * m_size_of_element);
        memset( source, 0, m_size_of_element);
    }

    set_element( index, element);
    return 0;
}

mi::Sint32 Dynamic_array_impl_proxy::erase( mi::Size index)
{
    if( index >= m_length)
        return -1;

    // Shift entries [index+1, m_length) one slot to the front and
    // move the element at slot index to slot m_length-1.
    char* base   = static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
    char* source = base + (index+1)    * m_size_of_element;
    char* target = base + index        * m_size_of_element;
    char* last   = base + (m_length-1) * m_size_of_element;
    char* buffer = new char[m_size_of_element];
    memcpy( buffer, target, m_size_of_element);
    memmove( target, source, (m_length - (index+1)) * m_size_of_element);
    memcpy( last, buffer, m_size_of_element);
    delete[] buffer;

    set_length( m_length-1);
    return 0;
}

mi::Sint32 Dynamic_array_impl_proxy::push_back( mi::base::IInterface* element)
{
    if( !element || !has_correct_element_type( element))
        return -2;

    set_length( m_length+1);
    set_element( m_length-1, element);
    return 0;
}

mi::Sint32 Dynamic_array_impl_proxy::pop_back()
{
    if( empty())
        return -3;

    set_length( m_length-1);
    return 0;
}

const mi::base::IInterface* Dynamic_array_impl_proxy::back() const
{
    if( empty())
        return nullptr;

    return get_element( m_length-1);
}

mi::base::IInterface* Dynamic_array_impl_proxy::back()
{
    if( empty())
        return nullptr;

    return get_element( m_length-1);
}

const mi::base::IInterface* Dynamic_array_impl_proxy::front() const
{
    if( empty())
        return nullptr;

    return get_element( 0);
}

mi::base::IInterface* Dynamic_array_impl_proxy::front()
{
    if( empty())
        return nullptr;

    return get_element( 0);
}

void Dynamic_array_impl_proxy::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = pointer;
    m_owner = make_handle_dup( owner);

    m_length = static_cast<ATTR::Dynamic_array*>( pointer)->m_count;

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    const ATTR::Type* type = attribute_context->get_type( m_attribute_name.c_str());
    m_size_of_element = type->sizeof_elem();

    // check invariant
    ASSERT( M_NEURAY_API, static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);
}

void Dynamic_array_impl_proxy::release_referenced_memory()
{
    for( mi::Size i = 0; i < m_length; ++i) {
        mi::base::Handle<mi::base::IInterface> element( get_value( i));
        mi::base::Handle<IProxy> proxy( element->get_interface<IProxy>());
        proxy->release_referenced_memory();
    }

    delete[] static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
    static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value = nullptr;
}

mi::neuraylib::ITransaction* Dynamic_array_impl_proxy::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

void Dynamic_array_impl_proxy::set_length_internal( mi::Size new_length)
{
    // check invariant
    ASSERT( M_NEURAY_API, static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);

    // get old values
    mi::Size old_length = m_length;
    char* old_data = static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
    if( new_length == old_length)
        return;

    // release memory referenced by surplus elements
    for( mi::Size i = new_length; i < old_length; ++i) {
        mi::base::Handle<mi::base::IInterface> element( get_value( i));
        mi::base::Handle<IProxy> proxy( element->get_interface<IProxy>());
        proxy->release_referenced_memory();
    }

    // allocate memory for new data and clear it
    char* new_data = new_length > 0 ? new char[new_length * m_size_of_element] : nullptr;
    if( new_length > 0)
        memset( new_data, 0, new_length * m_size_of_element); //-V575 PVS

    // copy old data over (as much as possible)
    if( new_length > 0 && old_length > 0)
        memcpy( new_data, old_data, mi::base::min( old_length, new_length) * m_size_of_element);

    // free memory for old data
    delete[] old_data;

    // store new length and pointer to new data
    m_length = new_length;
    static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count = static_cast<int>(new_length);
    static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value = new_data;

    // check invariant
    ASSERT( M_NEURAY_API, static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);
}

bool Dynamic_array_impl_proxy::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_likely.get_status())
        return false;
    index = *index_likely.get_ptr(); //-V522 PVS
    return index < m_length;
}

bool Dynamic_array_impl_proxy::key_to_index_unbounded( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_likely.get_status())
        return false;
    index = *index_likely.get_ptr(); //-V522 PVS
    return true;
}

bool Dynamic_array_impl_proxy::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_length)
        return false;
    std::ostringstream s;
    s << index;
    key = s.str();
    return true;
}

bool Dynamic_array_impl_proxy::has_correct_element_type( const mi::base::IInterface* element) const
{
    ASSERT( M_NEURAY_API, element);

    mi::base::Handle<const mi::IData> data( element->get_interface<mi::IData>());
    if( !data.is_valid_interface())
        return false;
    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_element_type_name == type_name;
}

} // namespace NEURAY

} // namespace MI
