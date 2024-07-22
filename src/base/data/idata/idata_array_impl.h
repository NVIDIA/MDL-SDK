/***************************************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IArray and IDynamic_array implementations.
 **/

#ifndef BASE_DATA_IDATA_IDATA_ARRAY_IMPL_H
#define BASE_DATA_IDATA_IDATA_ARRAY_IMPL_H

#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idynamic_array.h>

#include <string>
#include <vector>

#include <boost/core/noncopyable.hpp>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>

#include "idata_interfaces.h"

namespace MI {

namespace DB { class Transaction; }

namespace IDATA {

class Factory;

template<class T>
class Array_impl_base
  : public T,
    public boost::noncopyable
{
public:
    Array_impl_base(
        const Factory* factory, DB::Transaction* transaction, const char* element_type_name);

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); }

    // public API methods (IData_collection)

    mi::Size get_length() const final { return m_array.size(); }

    const char* get_key( mi::Size index) const final;

    bool has_key( const char* key) const final;

    const mi::base::IInterface* get_value( const char* key) const final;

    mi::base::IInterface* get_value( const char* key) final;

    const mi::base::IInterface* get_value( mi::Size index) const final;

    mi::base::IInterface* get_value( mi::Size index) final;

    mi::Sint32 set_value( const char* key, mi::base::IInterface* value) final;

    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value) final;

    // public API methods (IArray)

    const mi::base::IInterface* get_element( mi::Size index) const final;

    mi::base::IInterface* get_element( mi::Size index) final;

    mi::Sint32 set_element( mi::Size index, mi::base::IInterface* element) final;

    bool empty() const final { return m_array.empty(); }

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

protected:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Other types (like "Ref") require a transaction. Hence,
    /// #create_instance() checks whether the constructor was successful.
    bool successfully_constructed() { return m_successfully_constructed; }

    /// Sets the length of the array.
    ///
    /// The interface pointers of excess array slots are released. Additional array slots are filled
    /// with default constructed instances.
    ///
    /// \return   Success or failure. The call might fail if it has to create instances of
    ///           structures, but the corresponding type name is no longer valid.
    bool set_length_internal( mi::Size length);

    /// Indicates whether the element has the correct element type.
    ///
    /// For untyped arrays, always returns \c true. For typed arrays, #mi::IData::get_type_name() is
    /// compared against m_element_type_name.
    bool has_correct_element_type( const mi::base::IInterface* element) const;

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

    /// The class factory.
    const Factory* m_factory = nullptr;

    /// The transaction used for #Factory::assign_from_to().
    DB::Transaction* m_transaction = nullptr;

    /// The array of interface pointers.
    ///
    /// Do not call resize() directly on this vector. Use set_length_internal() instead.
    std::vector<mi::base::Handle<mi::base::IInterface>> m_array;

    /// The type name of the array itself.
    std::string m_type_name;

    /// The type name of array elements, or "Interface" for untyped arrays.
    std::string m_element_type_name;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed = false;
};

class Array_impl : public Array_impl_base<mi::base::Interface_implement<mi::IArray>>
{
public:

    /// The factory expects exactly one argument of type IString and one arguments of type ISize
    /// in \p argv.
   static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Array_impl(
        const Factory* factory,
        DB::Transaction* transaction,
        const char* element_type_name,
        mi::Size length);

    // public API methods

    // (implemented in Array_impl_base)

    // internal methods
};

class Dynamic_array_impl
  : public Array_impl_base<mi::base::Interface_implement<mi::IDynamic_array>>
{
public:

    /// The factory expects exactly one argument of type IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Dynamic_array_impl(
        const Factory* factory,
        DB::Transaction* transaction,
        const char* element_type_name);

    // public API methods

    // (remaining ones implemented in Array_impl_base)

    void set_length( mi::Size length) final { set_length_internal( length); }

    void clear() final { set_length( 0); }

    mi::Sint32 insert( mi::Size index, mi::base::IInterface* element) final;

    mi::Sint32 erase( mi::Size index) final;

    mi::Sint32 push_back( mi::base::IInterface* element) final;

    mi::Sint32 pop_back() final;

    const mi::base::IInterface* back() const final;

    mi::base::IInterface* back() final;

    const mi::base::IInterface* front() const final;

    mi::base::IInterface* front() final;

    // internal methods

private:

    /// Converts a given key to the corresponding index.
    ///
    /// If \p key is a valid key, return \c true and the corresponding index in \c index.
    /// Otherwise return \c false.
    ///
    /// Note that indices exceeding #get_length()-1 are considered valid (in contrast to
    /// the method #key_to_index() which is used most of the time). This method is used
    /// by #Class_factory::assign_from_to() to determine whether to resize the array.
    bool key_to_index_unbounded( const char* key, mi::Size& index) const;
};

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
    /// one argument of type IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Array_impl_proxy(
        const Factory* factory,
        DB::Transaction* transaction,
        const char* element_type_name,
        mi::Size length,
        const char* attribute_name);

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); }

    // public API methods (IData_collection)

    mi::Size get_length() const final { return m_length; }

    const char* get_key( mi::Size index) const final;

    bool has_key( const char* key) const final;

    const mi::base::IInterface* get_value( const char* key) const final;

    mi::base::IInterface* get_value( const char* key) final;

    const mi::base::IInterface* get_value( mi::Size index) const final;

    mi::base::IInterface* get_value( mi::Size index) final;

    mi::Sint32 set_value( const char* key, mi::base::IInterface* value) final;

    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value) final;

    using mi::IData_collection::get_value;

    // public API methods (IArray)

    const mi::base::IInterface* get_element( mi::Size index) const final;

    mi::base::IInterface* get_element( mi::Size index) final;

    mi::Sint32 set_element( mi::Size index, mi::base::IInterface* element) final;

    bool empty() const final { return m_length == 0; }

    using mi::IArray::get_element;

    // internal methods (IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) final;

    void release_referenced_memory() final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

protected:

    /// Sets the length of the array.
    ///
    /// Stores the length of the array in m_length.
    void set_length_internal( mi::Size length) { m_length = length; }

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

    /// The class factory.
    const Factory* m_factory = nullptr;

    /// The transaction used for array element accesses.
    DB::Transaction* m_transaction = nullptr;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;

    /// The length of the array.
    mi::Size m_length = 0;

    /// The type name of array elements.
    std::string m_element_type_name;

    /// The type name of the array itself.
    std::string m_type_name;

    /// The name that identifies the attribute (or the corresponding part of the attribute).
    std::string m_attribute_name;

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

    /// The factory expects exactly two arguments of type IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Dynamic_array_impl_proxy(
        const Factory* factory,
        DB::Transaction* transaction,
        const char* element_type_name,
        const char* attribute_name);

    /// Destructor
    ~Dynamic_array_impl_proxy();

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); }

    // public API methods (IData_collection)

    mi::Size get_length() const final { return m_length; }

    const char* get_key( mi::Size index) const final;

    bool has_key( const char* key) const final;

    const mi::base::IInterface* get_value( const char* key) const final;

    mi::base::IInterface* get_value( const char* key) final;

    const mi::base::IInterface* get_value( mi::Size index) const final;

    mi::base::IInterface* get_value( mi::Size index) final;

    mi::Sint32 set_value( const char* key, mi::base::IInterface* value) final;

    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value) final;

    using mi::IData_collection::get_value;

    // public API methods (IArray)

    const mi::base::IInterface* get_element( mi::Size index) const final;

    mi::base::IInterface* get_element( mi::Size index) final;

    mi::Sint32 set_element( mi::Size index, mi::base::IInterface* element) final;

    bool empty() const final { return m_length == 0; }

    using mi::IArray::get_element;

    // public API methods (IDynamic_array)

    void set_length( mi::Size size) final { set_length_internal( size); }

    void clear() final { set_length( 0); }

    mi::Sint32 insert( mi::Size index, mi::base::IInterface* element) final;

    mi::Sint32 erase( mi::Size index) final;

    mi::Sint32 push_back( mi::base::IInterface* element) final;

    mi::Sint32 pop_back() final;

    const mi::base::IInterface* back() const final;

    mi::base::IInterface* back() final;

    const mi::base::IInterface* front() const final;

    mi::base::IInterface* front() final;

    // internal methods (IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) final;

    void release_referenced_memory() final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

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
    /// by #Class_factory::assign_from_to() to determine whether to resize the array.
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

    /// The class factory.
    const Factory* m_factory = nullptr;

    /// The transaction used for array element accesses.
    DB::Transaction* m_transaction;

    /// Pointer to the storage
    ///
    /// Invariant: static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length
    ///            (after set_pointer_and_owner() has been called).
    void* m_pointer = nullptr;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;

    /// The length of the array.
    ///
    /// Invariant: static_cast<ATTR::Dynamic_array*>( m_pointer)->m_count == m_length
    ///            (after set_pointer_and_owner() has been called).
    mi::Size m_length = 0;

    /// The type name of array elements.
    std::string m_element_type_name;

    /// The type name of the array itself.
    std::string m_type_name;

    /// The name that identifies the attribute (or the corresponding part of the attribute).
    std::string m_attribute_name;

    /// The size of one element in bytes.
    ///
    /// Needed for all resize operations, in particular set_length_internal(), insert(), erase().
    mi::Size m_size_of_element = 0;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;
};

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_ARRAY_IMPL_H
