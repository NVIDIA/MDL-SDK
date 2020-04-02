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
 ** \brief Header for the Map_impl implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_MAP_IMPL_H
#define API_API_NEURAY_NEURAY_MAP_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/imap.h>

#include <string>
#include <map>
#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

class Map_impl
  : public mi::base::Interface_implement<mi::IMap>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IString. The arguments is passed to the
    /// constructor.
    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Map_impl( mi::neuraylib::ITransaction* transaction, const char* value_type_name);

    /// Destructor
    ~Map_impl();

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

    // public API methods (IMap)

    bool empty() const;

    void clear();

    mi::Sint32 insert( const char* key, mi::base::IInterface* value);

    mi::Sint32 erase( const char* key);

    // internal methods

    /// Returns the embedded transaction.
    mi::neuraylib::ITransaction* get_transaction() const;

private:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Other types (like "Ref") require a transaction. Hence,
    /// create_api_class() checks whether the constructor was successful.
    bool successfully_constructed() { return m_successfully_constructed; }

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped maps, always returns \c true. For typed maps, #mi::IData::get_type_name() is
    /// compared against m_value_type_name.
    bool has_correct_value_type( const mi::base::IInterface* value) const;

    /// Converts a given index to the corresponding key.
    ///
    /// If \p index is a valid index, return \c true and the key in \c key.
    /// Otherwise return \c false.
    ///
    /// \note The implementation is sub-optimal (takes linear time). It should be possible to make
    ///       it constant time for subsequent accesses to indices in increasing order by caching
    ///       the last iterator (assuming there are no changes in between).
    bool index_to_key( mi::Size index, std::string& key) const;

    /// The type of the map of interface pointers below.
    typedef std::map<std::string, mi::base::IInterface*> m_map_type;

    /// The map of interface pointers.
    ///
    /// Do not call resize() directly on this vector. Use set_length_internal() instead.
    m_map_type m_map;

    /// The type name of the map itself.
    std::string m_type_name;

    /// The type name of map values, or "Interface" for untyped maps.
    std::string m_value_type_name;

    /// The transaction that might be needed to construct the map members.
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;

    /// Indicates whether the next two members #m_cached_iterator and #m_cached_index are valid.
    ///
    /// Used to speed-up index-based lookups in non-decreasing order. The cached values are updated
    /// on each successful lookup. They are invalidated by each operation that modifies the
    /// structure of the map (#insert(), #erase() and #clear), but not by operations that modify
    /// the values of the map.
    mutable bool m_cache_valid;

    /// The index used in the last successful lookup.
    mutable mi::Size m_cached_index;

    /// The iterator for #m_cached_index.
    mutable m_map_type::const_iterator m_cached_iterator;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MAP_IMPL_H
