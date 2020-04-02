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
 ** \brief Header for the Array_impl_proxy implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_ARRAY_IMPL_PROXY_H
#define API_API_NEURAY_NEURAY_ARRAY_IMPL_PROXY_H

#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idynamic_array.h>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>

#include "i_neuray_proxy.h"

#include <string>
#include <vector>
#include <boost/core/noncopyable.hpp>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace DB { class Tag; }

namespace NEURAY {

/// This class contains an alternative implementation of the IArray interface.
///
/// It is used for attribute arrays. The implementation differs from the standard IArray
/// implementation Array_impl in the following ways:
///
/// - Untyped arrays are not supported in the proxy implementation.
/// - The implementation does not store the interface pointers, but operates on the actual values
///   in the attribute set.
/// - The implementation does not own the storage for the actual values, it is a proxy
///   implementation.
/// - Because it does not store interface pointers, set_element() extracts the actual value from
///   the interface pointer and stores it.
class Array_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IArray>, IProxy>,
    public boost::noncopyable
{
public:

    /// The factory expects exactly one argument of type IString, one argument of type ISize, and
    /// one argument of type IString. The arguments are passed to the constructor in this order.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Array_impl_proxy(
        mi::neuraylib::ITransaction* transaction,
        const char* element_type_name,
        mi::Size length,
        const char* attribute_name);

    // public API methods (IData)

    const char* get_type_name() const;

    // public API methods (IData_collection)

    mi::Size get_length() const;

    const char* get_key( mi::Size index) const;

    bool has_key( const char* key) const;

    const mi::base::IInterface* get_value( const char* key) const;

    mi::base::IInterface* get_value( const char* key);

    const mi::base::IInterface* get_value( mi::Size index) const;

    mi::base::IInterface* get_value( mi::Size index);

    mi::Sint32 set_value( const char* key, mi::base::IInterface* value);

    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value);

    // public API methods (IArray)

    const mi::base::IInterface* get_element( mi::Size index) const;

    mi::base::IInterface* get_element( mi::Size index);

    mi::Sint32 set_element( mi::Size index, mi::base::IInterface* element);

    bool empty() const;

    // internal methods (IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner);

    void release_referenced_memory();

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

protected:

    /// Sets the length of the array.
    ///
    /// Stores the length of the array in m_length.
    void set_length_internal( mi::Size length);

    /// Converts a given key to the corresponding index.
    ///
    /// If \p key is a valid key, return \c true and the corresponding index in \c index.
    /// Otherwise return \c false.
    bool key_to_index( const char* key, mi::Size& index) const;

    /// Converts a given index to the corresponding key.
    ///
    /// If \p index is a valid index, return \c true and the key in \c key.
    /// Otherwise return \c false.
    bool index_to_key( mi::Size index, std::string& key) const;

    /// Indicates whether the element has the correct element type.
    ///
    /// #mi::IData::get_type_name() is compared against m_element_type_name.
    bool has_correct_element_type( const mi::base::IInterface* element) const;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;

    /// The length of the array.
    mi::Size m_length;

    /// The name that identifies the attribute (or the corresponding part of the attribute).
    std::string m_attribute_name;

    /// The type name of the array itself.
    std::string m_type_name;

    /// The type name of array elements.
    std::string m_element_type_name;

    /// The transaction used for array element accesses.
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;
};

/// This class contains an alternative implementation of the IDynamic_array interface.
///
/// It is used for attribute arrays. The implementation differs from the standard IDynamic_array
/// implementation Dynamic_array_impl in the following ways:
///
/// - Untyped arrays are not supported in the proxy implementation.
/// - The implementation does not store the interface pointers, but operates on the actual values
///   in the attribute set.
/// - The implementation does not own the storage for the actual values, it is a proxy
///   implementation.
/// - Because it does not store interface pointers, set_element() extracts the actual value from
///   the interface pointer and stores it.
class Dynamic_array_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IDynamic_array>, IProxy>,
    public boost::noncopyable
{
public:

    /// The factory expects exactly two arguments of type IString. The arguments are passed to the
    /// constructor in this order.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Dynamic_array_impl_proxy(
        mi::neuraylib::ITransaction* transaction,
        const char* element_type_name,
        const char* attribute_name);

    /// Destructor
    ~Dynamic_array_impl_proxy();

    // public API methods (IData)

    const char* get_type_name() const;

    // public API methods (IData_collection)

    mi::Size get_length() const;

    const char* get_key( mi::Size index) const;

    bool has_key( const char* key) const;

    const mi::base::IInterface* get_value( const char* key) const;

    mi::base::IInterface* get_value( const char* key);

    const mi::base::IInterface* get_value( mi::Size index) const;

    mi::base::IInterface* get_value( mi::Size index);

    mi::Sint32 set_value( const char* key, mi::base::IInterface* value);

    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value);

    // public API methods (IArray)

    const mi::base::IInterface* get_element( mi::Size index) const;

    mi::base::IInterface* get_element( mi::Size index);

    mi::Sint32 set_element( mi::Size index, mi::base::IInterface* element);

    bool empty() const;

    // public API methods (IDynamic_array)

    void set_length( mi::Size size);

    void clear();

    mi::Sint32 insert( mi::Size index, mi::base::IInterface* element);

    mi::Sint32 erase( mi::Size index);

    mi::Sint32 push_back( mi::base::IInterface* element);

    mi::Sint32 pop_back();

    const mi::base::IInterface* back() const;

    mi::base::IInterface* back();

    const mi::base::IInterface* front() const;

    mi::base::IInterface* front();

    // internal methods (IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner);

    void release_referenced_memory();

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

protected:

    /// Sets the length of the array.
    ///
    /// Stores the length of the array in m_length.
    void set_length_internal( mi::Size length);

    /// Converts a given key to the corresponding index.
    ///
    /// If \p key is a valid key, return \c true and the corresponding index in \c index.
    /// Otherwise return \c false.
    bool key_to_index( const char* key, mi::Size& index) const;

    /// Converts a given key to the corresponding index.
    ///
    /// If \p key is a valid key, return \c true and the corresponding index in \c index.
    /// Otherwise return \c false.
    ///
    /// Note that indices exceeding #get_length()-1 are considered valid (in contrast to
    /// the method #key_to_index() which is used most of the time). This method is used
    /// by #assign_from() to determine whether to resize the array.
    bool key_to_index_unbounded( const char* key, mi::Size& index) const;

    /// Converts a given index to the corresponding key.
    ///
    /// If \p index is a valid index, return \c true and the key in \c key.
    /// Otherwise return \c false.
    bool index_to_key( mi::Size index, std::string& key) const;

    /// Indicates whether the element has the correct element type.
    ///
    /// #mi::IData::get_type_name() is compared against m_element_type_name.
    bool has_correct_element_type( const mi::base::IInterface* element) const;

    /// Pointer to the storage
    ///
    /// Invariant: static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length
    ///            (after set_pointer_and_owner() has been called).
    void* m_pointer;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;

    /// The length of the array.
    ///
    /// Invariant: static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length
    ///            (after set_pointer_and_owner() has been called).
    mi::Size m_length;

    /// The name that identifies the attribute (or the corresponding part of the attribute).
    std::string m_attribute_name;

    /// The type name of the array itself.
    std::string m_type_name;

    /// The type name of array elements.
    std::string m_element_type_name;

    /// The size of one element in bytes.
    ///
    /// Needed for all resize operations, in particular set_length_internal(), insert(), erase().
    mi::Size m_size_of_element;

    /// The transaction used for array element accesses.
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ARRAY_IMPL_PROXY_H
