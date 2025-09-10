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
 ** \brief Source for the IArray and IDynamic_array implementations.
 **/

#include "pch.h"

#include "idata_array_impl.h"

#include <cstring>

#include <base/data/attr/i_attr_types.h>
#include <base/system/main/i_assert.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>

#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>

#include "i_idata_factory.h"

namespace MI {

namespace IDATA {

template <typename T>
Array_impl_base<T>::Array_impl_base(
    const Factory* factory, DB::Transaction* transaction, const char* element_type_name)
  : m_factory( factory),
    m_transaction( transaction),
    m_element_type_name( element_type_name)
{
    MI_ASSERT( element_type_name);
}

template <typename T>
const char* Array_impl_base<T>::get_key( mi::Size index) const
{
    if( !index_to_key( index, m_cached_key))
        return nullptr;

    return m_cached_key.c_str();
}

template <typename T>
bool Array_impl_base<T>::has_key( const char* key) const
{
    mi::Size index;
    return key_to_index( key, index);
}

template <typename T>
const mi::base::IInterface* Array_impl_base<T>::get_value( const char* key) const
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_element( index);
}

template <typename T>
mi::base::IInterface* Array_impl_base<T>::get_value( const char* key)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_element( index);
}

template <typename T>
const mi::base::IInterface* Array_impl_base<T>::get_value( mi::Size index) const
{
    return get_element( index);
}

template <typename T>
mi::base::IInterface* Array_impl_base<T>::get_value( mi::Size index)
{
    return get_element( index);
}

template <typename T>
mi::Sint32 Array_impl_base<T>::set_value( const char* key, mi::base::IInterface* value)
{
    if( !value)
        return -1;

    mi::Size index;
    if( !key_to_index( key, index))
        return -2;

    return (set_element( index, value) == 0) ? 0 : -3;
}

template <typename T>
mi::Sint32 Array_impl_base<T>::set_value( mi::Size index, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    if( index >= m_array.size())
        return -2;

    return (set_element( index, value) == 0) ? 0 : -3;
}

template <typename T>
const mi::base::IInterface* Array_impl_base<T>::get_element( mi::Size index) const
{
    if( index >= m_array.size())
        return nullptr;

    m_array[index]->retain();
    return m_array[index].get();
}

template <typename T>
mi::base::IInterface* Array_impl_base<T>::get_element( mi::Size index)
{
    if( index >= m_array.size())
        return nullptr;

    m_array[index]->retain();
    return m_array[index].get();
}

template <typename T>
mi::Sint32 Array_impl_base<T>::set_element( mi::Size index, mi::base::IInterface* element)
{
    if( index >= m_array.size())
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    m_array[index] = make_handle_dup( element);
    return 0;
}

template <typename T>
bool Array_impl_base<T>::set_length_internal( mi::Size length)
{
    mi::Size old_length = m_array.size();
    m_array.resize( length, {});

    std::string element_type_name
        = (m_element_type_name == "Interface") ? "Void" : m_element_type_name;
    for( mi::Size i = old_length; i < length; ++i) {
        mi::base::IInterface* element = m_factory->create(
            m_transaction, element_type_name.c_str());
        if( !element) {
            // might happen for no longer registered type names
            m_array.resize( i>0 ? i-1 : 0);
            return false;
        }
        m_array[i] = element;
    }

    return true;
}

template <typename T>
bool Array_impl_base<T>::has_correct_element_type( const mi::base::IInterface* element) const
{
    MI_ASSERT( element);

    if( m_element_type_name == "Interface")
        return true;

    mi::base::Handle data( element->get_interface<mi::IData>());
    if( !data)
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_element_type_name == type_name;
}

template <typename T>
bool Array_impl_base<T>::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::optional<mi::Size> index_optional = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_optional.has_value())
        return false;

    index = index_optional.value();
    return index < m_array.size();
}

template <typename T>
bool Array_impl_base<T>::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_array.size())
        return false;

    key = std::to_string( index);
    return true;
}

mi::base::IInterface* Array_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;

    mi::base::Handle string( argv[0]->get_interface<mi::IString>());
    if( !string)
        return nullptr;

    const char* element_type_name = string->get_c_str();
    mi::base::Handle value( argv[1]->get_interface<mi::ISize>());
    if( !value)
        return nullptr;

    auto length = value->get_value<mi::Size>();
    mi::base::Handle array( new Array_impl( factory, transaction, element_type_name, length));
    return array->successfully_constructed() ? array.extract() : nullptr;
}

Array_impl::Array_impl(
    const Factory* factory,
    DB::Transaction* transaction,
    const char* element_type_name,
    mi::Size length)
  : Array_impl_base<mi::base::Interface_implement<mi::IArray>>(
        factory, transaction, element_type_name)
{
    m_successfully_constructed = set_length_internal( length);

    m_type_name = element_type_name;
    m_type_name += '[';
    m_type_name += std::to_string( length);
    m_type_name += ']';
}

mi::base::IInterface* Dynamic_array_impl::create_instance(
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

    const char* element_type_name = string->get_c_str();
    mi::base::Handle dynamic_array(
        new Dynamic_array_impl( factory, transaction, element_type_name));
    return dynamic_array->successfully_constructed() ? dynamic_array.extract() : nullptr;
}

Dynamic_array_impl::Dynamic_array_impl(
    const Factory* factory,
    DB::Transaction* transaction,
    const char* element_type_name)
  : Array_impl_base<mi::base::Interface_implement<mi::IDynamic_array>>(
        factory, transaction, element_type_name)
{
    // check if we have the transaction if needed
    m_successfully_constructed = set_length_internal( 1);
    set_length_internal( 0);

    m_type_name = element_type_name;
    m_type_name += "[]";
}

mi::Sint32 Dynamic_array_impl::insert( mi::Size index, mi::base::IInterface* element)
{
    if( index > m_array.size()) // note special case ">" instead of ">=" here
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    m_array.insert( m_array.begin()+index, make_handle_dup( element));
    return 0;
}

mi::Sint32 Dynamic_array_impl::erase( mi::Size index)
{
    if( index >= m_array.size())
        return -1;

    m_array.erase( m_array.begin()+index);
    return 0;
}

mi::Sint32 Dynamic_array_impl::push_back( mi::base::IInterface* element)
{
    if( !element || !has_correct_element_type( element))
        return -2;

    m_array.push_back( make_handle_dup( element));
    return 0;
}

mi::Sint32 Dynamic_array_impl::pop_back()
{
    if( empty())
        return -3;

    m_array.pop_back();
    return 0;
}

const mi::base::IInterface* Dynamic_array_impl::back() const
{
    if( empty())
        return nullptr;

    const mi::base::IInterface* element = m_array.back().get();
    element->retain();
    return element;
}

mi::base::IInterface* Dynamic_array_impl::back()
{
    if( empty())
        return nullptr;

    mi::base::IInterface* element = m_array.back().get();
    element->retain();
    return element;
}

const mi::base::IInterface* Dynamic_array_impl::front() const
{
    if( empty())
        return nullptr;

    const mi::base::IInterface* element = m_array.front().get();
    element->retain();
    return element;
}

mi::base::IInterface* Dynamic_array_impl::front()
{
    if( empty())
        return nullptr;

    mi::base::IInterface* element = m_array.front().get();
    element->retain();
    return element;
}

bool Dynamic_array_impl::key_to_index_unbounded( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::optional<mi::Size> index_optional = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_optional.has_value())
        return false;

    index = index_optional.value();
    return true;
}

mi::base::IInterface* Array_impl_proxy::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;

    if( argc != 3)
        return nullptr;

    mi::base::Handle string( argv[0]->get_interface<mi::IString>());
    if( !string)
        return nullptr;

    const char* element_type_name = string->get_c_str();
    mi::base::Handle value( argv[1]->get_interface<mi::ISize>());
    if( !value)
        return nullptr;

    auto length = value->get_value<mi::Size>();
    string = argv[2]->get_interface<mi::IString>();
    if( !string)
        return nullptr;

    const char* attribute_name = string->get_c_str();
    return (new Array_impl_proxy(
        factory, transaction, element_type_name, length, attribute_name))->cast_to_major();
}

Array_impl_proxy::Array_impl_proxy(
    const Factory* factory,
    DB::Transaction* transaction,
    const char* element_type_name,
    mi::Size length,
    const char* attribute_name)
  : m_factory( factory),
    m_transaction( transaction),
    m_element_type_name( element_type_name),
    m_type_name( m_element_type_name + "[" + std::to_string( length) + "]"),
    m_attribute_name( attribute_name)
{
    MI_ASSERT( transaction);

    set_length_internal( length);
}

const char* Array_impl_proxy::get_key( mi::Size index) const
{
    if( !index_to_key( index, m_cached_key))
        return nullptr;

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

    std::string s( m_attribute_name);
    s += '[';
    s += std::to_string( index);
    s += ']';

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    return attribute_context->get_attribute( s.c_str());
}

mi::base::IInterface* Array_impl_proxy::get_element( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    std::string s( m_attribute_name);
    s += '[';
    s += std::to_string( index);
    s += ']';

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    return attribute_context->get_attribute( s.c_str());
}

mi::Sint32 Array_impl_proxy::set_element(
    mi::Size index, mi::base::IInterface* element)
{
    if( index >= m_length)
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    mi::base::Handle new_data( element->get_interface<mi::IData>());
    MI_ASSERT( new_data);

    mi::base::Handle old_data( get_element<mi::IData>( index));
    MI_ASSERT( old_data);
    [[maybe_unused]] mi::Uint32 result
       = m_factory->assign_from_to( new_data.get(), old_data.get(), /*options*/ 0);
    MI_ASSERT( result == 0);

    return 0;
}

void Array_impl_proxy::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_owner = make_handle_dup( owner);
}

void Array_impl_proxy::release_referenced_memory()
{
    for( mi::Size i = 0; i < m_length; ++i) {
        mi::base::Handle proxy( get_value<IProxy>( i));
        proxy->release_referenced_memory();
    }
}

bool Array_impl_proxy::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::optional<mi::Size> index_optional = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_optional.has_value())
        return false;

    index = index_optional.value();
    return index < m_length;
}

bool Array_impl_proxy::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_length)
        return false;

    key = std::to_string( index);
    return true;
}

bool Array_impl_proxy::has_correct_element_type( const mi::base::IInterface* element) const
{
    MI_ASSERT( element);

    mi::base::Handle data( element->get_interface<mi::IData>());
    if( !data)
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_element_type_name == type_name;
}

mi::base::IInterface* Dynamic_array_impl_proxy::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;

    if( argc != 2)
        return nullptr;

    mi::base::Handle string( argv[0]->get_interface<mi::IString>());
    if( !string)
        return nullptr;

    const char* element_type_name = string->get_c_str();
    string = argv[1]->get_interface<mi::IString>();
    if( !string)
        return nullptr;

    const char* attribute_name = string->get_c_str();
    return (new Dynamic_array_impl_proxy(
        factory, transaction, element_type_name, attribute_name))->cast_to_major();
}

Dynamic_array_impl_proxy::Dynamic_array_impl_proxy(
    const Factory* factory,
    DB::Transaction* transaction,
    const char* element_type_name,
    const char* attribute_name)
  : m_factory( factory),
    m_transaction( transaction),
    m_element_type_name( element_type_name),
    m_type_name( m_element_type_name + "[]"),
    m_attribute_name( attribute_name)
{
    MI_ASSERT( transaction);
}

Dynamic_array_impl_proxy::~Dynamic_array_impl_proxy()
{
    // check invariant
    MI_ASSERT( static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);
}

const char* Dynamic_array_impl_proxy::get_key( mi::Size index) const
{
    if( !index_to_key( index, m_cached_key))
        return nullptr;

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

    std::string s( m_attribute_name);
    s += '[';
    s += std::to_string( index);
    s += ']';

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    return attribute_context->get_attribute( s.c_str());
}

mi::base::IInterface* Dynamic_array_impl_proxy::get_element( mi::Size index)
{
    if( index >= m_length)
        return nullptr;

    std::string s( m_attribute_name);
    s += '[';
    s += std::to_string( index);
    s += ']';

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    return attribute_context->get_attribute( s.c_str());
}

mi::Sint32 Dynamic_array_impl_proxy::set_element(
    mi::Size index, mi::base::IInterface* element)
{
    if( index >= m_length)
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    mi::base::Handle new_data( element->get_interface<mi::IData>());
    MI_ASSERT( new_data);

    mi::base::Handle old_data( get_element<mi::IData>( index));
    MI_ASSERT( old_data);
    [[maybe_unused]] mi::Uint32 result
        = m_factory->assign_from_to( new_data.get(), old_data.get(), /*options*/ 0);
    MI_ASSERT( result == 0);

    return 0;
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
    char* const base   = static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
    char* const source = base + (index+1)    * m_size_of_element;
    char* const target = base + index        * m_size_of_element;
    char* const last   = base + (m_length-1) * m_size_of_element;
    {
        const std::vector<char> buffer( target, target + m_size_of_element);
        memmove( target, source, (m_length - (index+1)) * m_size_of_element);
        memcpy( last, buffer.data(), m_size_of_element);
    }

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

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    m_size_of_element = attribute_context->get_sizeof_elem( m_attribute_name.c_str());
    MI_ASSERT( m_size_of_element > 0);

    // check invariant
    MI_ASSERT( static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);
}

void Dynamic_array_impl_proxy::release_referenced_memory()
{
    for( mi::Size i = 0; i < m_length; ++i) {
        mi::base::Handle proxy( get_value<IProxy>( i));
        proxy->release_referenced_memory();
    }

    delete[] static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
    static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value = nullptr;
}

void Dynamic_array_impl_proxy::set_length_internal( mi::Size new_length)
{
    // check invariant
    MI_ASSERT( static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);

    // get old values
    mi::Size old_length = m_length;
    char* old_data = static_cast<ATTR::Dynamic_array*>( m_pointer)->m_value;
    if( new_length == old_length)
        return;

    // release memory referenced by surplus elements
    for( mi::Size i = new_length; i < old_length; ++i) {
        mi::base::Handle proxy( get_value<IProxy>( i));
        proxy->release_referenced_memory();
    }

    // allocate memory for new data and clear it
    char* new_data = new_length > 0 ? new char[new_length * m_size_of_element] : nullptr;
    if( new_length > 0)
        memset( new_data, 0, new_length * m_size_of_element);

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
    MI_ASSERT( static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length);
}

bool Dynamic_array_impl_proxy::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::optional<mi::Size> index_optional = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_optional.has_value())
        return false;

    index = index_optional.value();
    return index < m_length;
}

bool Dynamic_array_impl_proxy::key_to_index_unbounded( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::optional<mi::Size> index_optional = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_optional.has_value())
        return false;

    index = index_optional.value();
    return true;
}

bool Dynamic_array_impl_proxy::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_length)
        return false;

    key = std::to_string( index);
    return true;
}

bool Dynamic_array_impl_proxy::has_correct_element_type( const mi::base::IInterface* element) const
{
    MI_ASSERT( element);

    mi::base::Handle data( element->get_interface<mi::IData>());
    if( !data)
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_element_type_name == type_name;
}

} // namespace IDATA

} // namespace MI
