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
 ** \brief Source for the IArray and IDynamic_array implementation.
 **/

#include "pch.h"

#include "neuray_array_impl.h"
#include "neuray_class_factory.h"
#include "neuray_transaction_impl.h"

#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>

#include <sstream>
#include <base/system/stlext/i_stlext_likely.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>

namespace MI {

namespace NEURAY {

template <typename T>
Array_impl_base<T>::Array_impl_base(
    mi::neuraylib::ITransaction* transaction, const char* element_type_name)
{
    // transaction might be NULL
    m_transaction = make_handle_dup( transaction);

    ASSERT( M_NEURAY_API, element_type_name);
    m_element_type_name = element_type_name;
}

template <typename T>
Array_impl_base<T>::~Array_impl_base()
{
    std::vector<mi::base::IInterface*>::iterator it = m_array.begin();
    std::vector<mi::base::IInterface*>::iterator end = m_array.end();
    for ( ; it != end; ++it)
        if( *it)
            (*it)->release();
}

template <typename T>
const char* Array_impl_base<T>::get_type_name() const
{
    return m_type_name.c_str();
}

template <typename T>
mi::Size Array_impl_base<T>::get_length() const
{
    return m_array.size();
}

template <typename T>
const char* Array_impl_base<T>::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
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
    return m_array[index];
}

template <typename T>
mi::base::IInterface* Array_impl_base<T>::get_element( mi::Size index)
{
    if( index >= m_array.size())
        return nullptr;

    m_array[index]->retain();
    return m_array[index];
}

template <typename T>
mi::Sint32 Array_impl_base<T>::set_element( mi::Size index, mi::base::IInterface* element)
{
    if( index >= m_array.size())
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    m_array[index]->release();
    m_array[index] = element;
    m_array[index]->retain();
    return 0;
}

template <typename T>
bool Array_impl_base<T>::empty() const
{
    return m_array.empty();
}

template <typename T>
mi::neuraylib::ITransaction* Array_impl_base<T>::get_transaction() const
{
    if( m_transaction.is_valid_interface())
        m_transaction->retain();
    return m_transaction.get();
}

template <typename T>
bool Array_impl_base<T>::set_length_internal( mi::Size length)
{
    mi::Size old_length = m_array.size();
    for( mi::Size i = length; i < old_length; ++i)
        m_array[i]->release();

    m_array.resize( length, nullptr);

    std::string element_type_name
        = (m_element_type_name == "Interface") ? "Void" : m_element_type_name;
    for( mi::Size i = old_length; i < length; ++i) {
        mi::base::IInterface* element = s_class_factory->create_type_instance(
            static_cast<Transaction_impl*>( m_transaction.get()), element_type_name.c_str());
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
    ASSERT( M_NEURAY_API, element);

    if( m_element_type_name == "Interface")
        return true;

    mi::base::Handle<const mi::IData> data( element->get_interface<mi::IData>());
    if( !data.is_valid_interface())
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
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_likely.get_status())
        return false;
    index = *index_likely.get_ptr(); //-V522 PVS
    return index < m_array.size();
}

template <typename T>
bool Array_impl_base<T>::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= m_array.size())
        return false;
    std::ostringstream s;
    s << index;
    key = s.str();
    return true;
}

// This method requires explicit instantiation since it is not used in this translation unit.
template mi::neuraylib::ITransaction*
Array_impl_base<mi::base::Interface_implement<mi::IArray> >::get_transaction() const;

mi::base::IInterface* Array_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* element_type_name = istring->get_c_str();
    mi::base::Handle<const mi::ISize> ivalue( argv[1]->get_interface<mi::ISize>());
    if( !ivalue.is_valid_interface())
        return nullptr;
    mi::Size length = ivalue->get_value<mi::Size>();

    Array_impl* array = new Array_impl( transaction, element_type_name, length);
    if( !array->successfully_constructed()) {
        array->release();
        return nullptr;
    } else
        return array;
}

Array_impl::Array_impl(
    mi::neuraylib::ITransaction* transaction,
    const char* element_type_name,
    mi::Size length)
  : Array_impl_base<mi::base::Interface_implement<mi::IArray> >( transaction, element_type_name)
{
    m_successfully_constructed = set_length_internal( length);

    std::ostringstream s;
    s << length;
    m_type_name = element_type_name;
    m_type_name += "[";
    m_type_name += s.str();
    m_type_name += "]";
}

// This method requires explicit instantiation since it is not used in this translation unit.
template mi::neuraylib::ITransaction*
Array_impl_base<mi::base::Interface_implement<mi::IDynamic_array> >::get_transaction() const;

mi::base::IInterface* Dynamic_array_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* element_type_name = istring->get_c_str();

    Dynamic_array_impl* dynamic_array = new Dynamic_array_impl( transaction, element_type_name);
    if( !dynamic_array->successfully_constructed()) {
        dynamic_array->release();
        return nullptr;
    } else
        return dynamic_array;
}

Dynamic_array_impl::Dynamic_array_impl(
    mi::neuraylib::ITransaction* transaction,
    const char* element_type_name)
  : Array_impl_base<mi::base::Interface_implement<mi::IDynamic_array> >(
        transaction, element_type_name)
{
    // check if we have the transaction if needed
    m_successfully_constructed = set_length_internal( 1);
    set_length_internal( 0);

    m_type_name = element_type_name;
    m_type_name += "[]";
}

void Dynamic_array_impl::set_length( mi::Size length)
{
    set_length_internal( length);
}

void Dynamic_array_impl::clear()
{
    set_length( 0);
}

mi::Sint32 Dynamic_array_impl::insert( mi::Size index, mi::base::IInterface* element)
{
    if( index > m_array.size()) // note special case ">" instead of ">=" here
        return -1;

    if( !element || !has_correct_element_type( element))
        return -2;

    element->retain();
    m_array.insert( m_array.begin()+index, element);
    return 0;
}

mi::Sint32 Dynamic_array_impl::erase( mi::Size index)
{
    if( index >= m_array.size())
        return -1;

    mi::base::IInterface* element = m_array[index];
    element->release();
    m_array.erase( m_array.begin()+index);
    return 0;
}

mi::Sint32 Dynamic_array_impl::push_back( mi::base::IInterface* element)
{
    if( !element || !has_correct_element_type( element))
        return -2;

    element->retain();
    m_array.push_back( element);
    return 0;
}

mi::Sint32 Dynamic_array_impl::pop_back()
{
    if( empty())
        return -3;

    mi::base::IInterface* element = m_array.back();
    element->release();
    m_array.pop_back();
    return 0;
}

const mi::base::IInterface* Dynamic_array_impl::back() const
{
    if( empty())
        return nullptr;

    const mi::base::IInterface* element = m_array.back();
    element->retain();
    return element;
}

mi::base::IInterface* Dynamic_array_impl::back()
{
    if( empty())
        return nullptr;

    mi::base::IInterface* element = m_array.back();
    element->retain();
    return element;
}

const mi::base::IInterface* Dynamic_array_impl::front() const
{
    if( empty())
        return nullptr;

    const mi::base::IInterface* element = m_array.front();
    element->retain();
    return element;
}

mi::base::IInterface* Dynamic_array_impl::front()
{
    if( empty())
        return nullptr;

    mi::base::IInterface* element = m_array.front();
    element->retain();
    return element;
}

bool Dynamic_array_impl::key_to_index_unbounded( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( key);
    if( !index_likely.get_status())
        return false;
    index = *index_likely.get_ptr(); //-V522 PVS
    return true;
}

} // namespace NEURAY

} // namespace MI
