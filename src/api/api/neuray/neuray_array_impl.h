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
 ** \brief Header for the IArray and IDynamic_array implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_ARRAY_IMPL_H
#define API_API_NEURAY_NEURAY_ARRAY_IMPL_H

#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idynamic_array.h>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include <string>
#include <vector>
#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

template<class T>
class Array_impl_base
  : public T,
    public boost::noncopyable
{
public:
    Array_impl_base( mi::neuraylib::ITransaction* transaction, const char* element_type_name);

    ~Array_impl_base();

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

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

protected:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Other types (like "Ref") require a transaction. Hence,
    /// create_api_class() checks whether the constructor was successful.
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

    /// The array of interface pointers.
    ///
    /// Do not call resize() directly on this vector. Use set_length_internal() instead.
    std::vector<mi::base::IInterface*> m_array;

    /// The type name of the array itself.
    std::string m_type_name;

    /// The type name of array elements, or "Interface" for untyped arrays.
    std::string m_element_type_name;

    /// The transaction that might be needed to construct the default array elements (might be
    /// \c NULL).
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed;
};

class Array_impl : public Array_impl_base<mi::base::Interface_implement<mi::IArray> >
{
public:

    /// The factory expects exactly one argument of type IString and one arguments of type ISize.
    /// The arguments are passed to the constructor in this order.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Array_impl(
        mi::neuraylib::ITransaction* transaction,
        const char* element_type_name,
        mi::Size length);

    // public API methods

    // (implemented in Array_impl_base)

    // internal methods

};

class Dynamic_array_impl
  : public Array_impl_base<mi::base::Interface_implement<mi::IDynamic_array> >
{
public:

    /// The factory expects exactly one argument of type IString. The argument is passed to the
    /// constructor.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Dynamic_array_impl( mi::neuraylib::ITransaction* transaction, const char* element_type_name);

    // public API methods

    // (remaining ones implemented in Array_impl_base)

    void set_length( mi::Size length);

    void clear();

    mi::Sint32 insert( mi::Size index, mi::base::IInterface* element);

    mi::Sint32 erase( mi::Size index);

    mi::Sint32 push_back( mi::base::IInterface* element);

    mi::Sint32 pop_back();

    const mi::base::IInterface* back() const;

    mi::base::IInterface* back();

    const mi::base::IInterface* front() const;

    mi::base::IInterface* front();

    // internal methods

private:

    /// Converts a given key to the corresponding index.
    ///
    /// If \p key is a valid key, return \c true and the corresponding index in \c index.
    /// Otherwise return \c false.
    ///
    /// Note that indices exceeding #get_length()-1 are considered valid (in contrast to
    /// the method #key_to_index() which is used most of the time). This method is used
    /// by #assign_from() to determine whether to resize the array.
    bool key_to_index_unbounded( const char* key, mi::Size& index) const;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ARRAY_IMPL_H
