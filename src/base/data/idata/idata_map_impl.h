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
 ** \brief Header for the IMap implementations.
 **/

#ifndef BASE_DATA_IDATA_IDATA_MAP_IMPL_H
#define BASE_DATA_IDATA_IDATA_MAP_IMPL_H

#include <mi/neuraylib/imap.h>

#include <string>
#include <map>

#include <boost/core/noncopyable.hpp>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

namespace MI {

namespace DB { class Transaction; }

namespace IDATA {

class Factory;

class Map_impl
  : public mi::base::Interface_implement<mi::IMap>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Map_impl(
        const Factory* factory, DB::Transaction* transaction, const char* value_type_name);

    /// Destructor
    ~Map_impl();

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); }

    // public API methods (IData_collection)

    mi::Size get_length() const final { return m_map.size(); }

    const char* get_key( mi::Size index) const final;

    bool has_key( const char* key) const final;

    const mi::base::IInterface* get_value( const char* key) const final;

    mi::base::IInterface* get_value( const char* key) final;

    const mi::base::IInterface* get_value( mi::Size index) const final;

    mi::base::IInterface* get_value( mi::Size index) final;

    mi::Sint32 set_value( const char* key, mi::base::IInterface* value) final;

    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value) final;

    // public API methods (IMap)

    bool empty() const final { return m_map.empty(); }

    void clear() final;

    mi::Sint32 insert( const char* key, mi::base::IInterface* value) final;

    mi::Sint32 erase( const char* key) final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

private:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Other types (like "Ref") require a transaction. Hence,
    /// #create_instance() checks whether the constructor was successful.
    bool successfully_constructed() const { return m_successfully_constructed; }

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped maps, always returns \c true. For typed maps, #mi::IData::get_type_name() is
    /// compared against m_value_type_name.
    bool has_correct_value_type( const mi::base::IInterface* value) const;

    /// Converts a given index to the corresponding key.
    ///
    /// If \p index is a valid index, return \c true and the key in \c key.
    /// Otherwise return \c false.
    bool index_to_key( mi::Size index, std::string& key) const;

    /// The transaction used for #Factory::assign_from_to().
    DB::Transaction* m_transaction = nullptr;

    /// The type of the map of interface pointers below.
    using Map_type = std::map<std::string, mi::base::Handle<mi::base::IInterface>>;

    /// The map of interface pointers.
    ///
    /// Do not call resize() directly on this vector. Use set_length_internal() instead.
    Map_type m_map;

    /// The type name of map values, or "Interface" for untyped maps.
    std::string m_value_type_name;

    /// The type name of the map itself.
    std::string m_type_name;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;

    /// Indicates whether the next two members #m_cached_iterator and #m_cached_index are valid.
    ///
    /// Used to speed-up index-based lookups in non-decreasing order. The cached values are updated
    /// on each successful lookup. They are invalidated by each operation that modifies the
    /// structure of the map (#insert(), #erase() and #clear), but not by operations that modify
    /// the values of the map.
    mutable bool m_cache_valid = false;

    /// The index used in the last successful lookup.
    mutable mi::Size m_cached_index = 0;

    /// The iterator for #m_cached_index.
    mutable Map_type::const_iterator m_cached_iterator;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed = false;
};

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_MAP_IMPL_H
