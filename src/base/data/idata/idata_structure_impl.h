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
 ** \brief Header for the IStructure and IStructure_decl implementations.
 **/

#ifndef BASE_DATA_IDATA_IDATA_STRUCTURE_DECL_IMPL_H
#define BASE_DATA_IDATA_IDATA_STRUCTURE_DECL_IMPL_H

#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/istructure_decl.h>

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

#include <base/lib/unordered_dense/unordered_dense.h>

#include "idata_interfaces.h"

namespace MI {

namespace DB { class Transaction; }

namespace IDATA {

class Factory;

class Structure_decl_impl
  : public mi::base::Interface_implement<mi::IStructure_decl>,
    public boost::noncopyable
{
public:
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    mi::Sint32 add_member( const char* type_name, const char* name) final;

    mi::Sint32 remove_member( const char* name) final;

    mi::Size get_length() const final  { return m_names.size(); }

    const char* get_member_type_name( mi::Size index) const final;

    const char* get_member_type_name( const char* name) const final;

    const char* get_member_name( mi::Size index) const final;

    const char* get_structure_type_name() const final;

    // internal methods

    void set_structure_type_name( const char* structure_type_name);

private:

    /// The name under which this structure declaration was registered.
    ///
    /// Non-empty only for registered structure declarations.
    std::string m_structure_type_name;

    /// Stores the type names of the struct members
    std::vector<std::string> m_type_names;

    /// Stores the names of the struct members
    std::vector<std::string> m_names;
};

/// Default implementation of IStructure
///
/// The default implementation Structure_impl of IStructure owns the memory used to store the actual
/// value. See the proxy implementation Structure_impl_proxy for a variant that does not own the
/// memory.
class Structure_impl
  : public mi::base::Interface_implement<mi::IStructure>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IStructure_decl and one argument of type
    /// IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Structure_impl(
        const Factory* factory,
        DB::Transaction* transaction,
        const mi::IStructure_decl* structure_decl,
        const char* type_name);

    /// Destructor
    ~Structure_impl();

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

    // public API methods (IStructure)

    const mi::IStructure_decl* get_structure_decl() const final;

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
    bool has_correct_value_type( mi::Size index, const mi::base::IInterface* value) const;

    /// Converts a given key to the corresponding index.
    bool key_to_index( const char* key, mi::Size& index) const;

    /// Converts a given index to the corresponding key.
    bool index_to_key( mi::Size index, std::string& key) const;

    /// The transaction used for #Factory::assign_from_to().
    DB::Transaction* m_transaction = nullptr;

    /// The corresponding structure declaration.
    mi::base::Handle<const mi::IStructure_decl> m_structure_decl;

    /// The type name of the structure itself.
    std::string m_type_name;

    /// The length of the structure.
    mi::Size m_length = 0;

    /// The vector of interface pointers.
    std::vector<mi::base::Handle<mi::base::IInterface>> m_member;

    /// Maps keys to indices.
    ankerl::unordered_dense::map<std::string, mi::Size> m_key_to_index;

    /// Maps indices to keys.
    std::vector<std::string> m_index_to_key;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed = false;
};

// Proxy implementation of IStructure
///
/// The proxy implementation Structure_impl_proxy of IStructure does not own the memory used to
/// store the actual value. See the default implementation Structure_impl for a variant that does
/// own the memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
class Structure_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IStructure>, IProxy>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IStructure_decl and two arguments of type
    /// IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Structure_impl_proxy(
        const Factory* factory,
        DB::Transaction* transaction,
        const mi::IStructure_decl* structure_decl,
        const char* type_name,
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

    using IData_collection::get_value;

    // public API methods (IStructure)

    const mi::IStructure_decl* get_structure_decl() const final;

    // internal methods (of IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) final;

    void release_referenced_memory() final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

private:

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped maps, always returns \c true. For typed maps, #mi::IData::get_type_name() is
    /// compared against m_value_type_name.
    bool has_correct_value_type( mi::Size index, const mi::base::IInterface* value) const;

    /// Converts a given key to the corresponding index.
    bool key_to_index( const char* key, mi::Size& index) const;

    /// Converts a given index to the corresponding key.
    bool index_to_key( mi::Size index, std::string& key) const;

    /// The class factory.
    const Factory* m_factory = nullptr;

    /// The transaction used for #Factory::assign_from_to().
    DB::Transaction* m_transaction = nullptr;

    /// The corresponding structure declaration.
    mi::base::Handle<const mi::IStructure_decl> m_structure_decl;

    /// The type name of the structure itself.
    std::string m_type_name;

    /// The name that identifies the attribute (or the corresponding part of the attribute).
    std::string m_attribute_name;

    /// The length of the structure.
    mi::Size m_length = 0;

    /// Maps keys to indices.
    ankerl::unordered_dense::map<std::string, mi::Size> m_key_to_index;

    /// Maps indices to keys.
    std::vector<std::string> m_index_to_key;

    /// Caches the last return value of get_key().
    mutable std::string m_cached_key;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;
};

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_STRUCTURE_IMPL_H
