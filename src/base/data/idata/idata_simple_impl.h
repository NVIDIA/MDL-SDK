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
 ** \brief Header for the IData_simple implementations.
 **/

#ifndef BASE_DATA_IDATA_IDATA_ENUM_DECL_IMPL_H
#define BASE_DATA_IDATA_IDATA_ENUM_DECL_IMPL_H

#include <mi/neuraylib/idata.h> // IVoid
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/ipointer.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/iuuid.h>

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

#include <base/data/db/i_db_tag.h>

#include "idata_interfaces.h"

namespace MI {

namespace DB { class Transaction; }

namespace IDATA {

class Factory;

class Enum_decl_impl
  : public mi::base::Interface_implement<mi::IEnum_decl>,
    public boost::noncopyable
{
public:
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    mi::Sint32 add_enumerator( const char* name, mi::Sint32 value) final;

    mi::Sint32 remove_enumerator( const char* name) final;

    mi::Size get_length() const final;

    const char* get_name( mi::Size index) const final;

    mi::Sint32 get_value( mi::Size index) const final;

    mi::Size get_index( const char* name) const final;

    mi::Size get_index( mi::Sint32 value) const final;

    const char* get_enum_type_name() const final;

    // internal methods

    void set_enum_type_name( const char* enum_type_name);

private:

    /// The name under which this enum declaration was registered.
    ///
    /// Non-empty only for registered enum declarations.
    std::string m_enum_type_name;

    /// Stores the names of the enumerators
    std::vector<std::string> m_names;

    /// Stores the values of the enumerators
    std::vector<mi::Sint32> m_values;
};

/// Default implementation of IEnum
///
/// The default implementation Enum_impl of IEnum owns the memory used to store the actual
/// value. See the proxy implementation Enum_impl_proxy for a variant that does not own the
/// memory.
class Enum_impl
  : public mi::base::Interface_implement<mi::IEnum>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IEnum_decl and one argument of type
    /// IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Enum_impl( const mi::IEnum_decl* enum_decl, const char* type_name);

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); }

    // public API methods (IEnum)

    void get_value( mi::Sint32& value) const final;

    const char* get_value_by_name() const final;

    mi::Sint32 set_value( mi::Sint32 value) final;

    mi::Sint32 set_value_by_name( const char* name) final;

    const mi::IEnum_decl* get_enum_decl() const final;

    // internal methods

private:
    /// The corresponding enum declaration.
    mi::base::Handle<const mi::IEnum_decl> m_enum_decl;

    /// The type name of the enum itself.
    std::string m_type_name;

    /// Storage
    ///
    /// Note that the storage holds the index of the enumerator in the declaration, not the value
    /// of the enumerator.
    mi::Uint32 m_storage = 0;
};

/// Proxy implementation of IEnum
///
/// The proxy implementation Enum_impl_proxy of IEnum does not own the memory used to
/// store the actual value. See the default implementation Enum_impl for a variant that does
/// own the memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
class Enum_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IEnum>, IProxy>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IEnum_decl and two arguments of type
    /// IString in \p argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Enum_impl_proxy( const mi::IEnum_decl* enum_decl, const char* type_name);

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); }

    // public API methods (IEnum)

    void get_value( mi::Sint32& value) const final;

    const char* get_value_by_name() const final;

    mi::Sint32 set_value( mi::Sint32 value) final;

    mi::Sint32 set_value_by_name( const char* name) final;

    const mi::IEnum_decl* get_enum_decl() const final;

    // internal methods (of IProxy)

     void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) final;

    void release_referenced_memory();

private:
    /// The corresponding enum declaration.
    mi::base::Handle<const mi::IEnum_decl> m_enum_decl;

    /// The type name of the enum itself.
    std::string m_type_name;

    /// Pointer to the storage.
    ///
    /// Note that the pointer points to the index of the enumerator in the declaration, not to
    /// the value of the enumerator.
    mi::Uint32* m_pointer = nullptr;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;
};

/// Default implementation of interfaces derived from INumber
///
/// The default implementation Number_impl of interfaces derived from INumber owns the memory used
/// to store the actual value. See the proxy implementation Number_impl_proxy for a variant that
/// does not own the memory.
///
/// Note that only a fixed set of types is permitted for the template parameters I and T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template<typename I, typename T>
class Number_impl
  : public mi::base::Interface_implement<I>,
    public boost::noncopyable
{
public:
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    const char* get_type_name() const final;

    void get_value( bool& value) const final;

    void set_value( bool value) final;

    void get_value( mi::Uint8& value) const final;

    void set_value( mi::Uint8 value) final;

    void get_value( mi::Uint16& value) const final;

    void set_value( mi::Uint16 value) final;

    void get_value( mi::Uint32& value) const final;

    void set_value( mi::Uint32 value) final;

    void get_value( mi::Uint64& value) const final;

    void set_value( mi::Uint64 value) final;

    void get_value( mi::Sint8& value) const final;

    void set_value( mi::Sint8 value) final;

    void get_value( mi::Sint16& value) const final;

    void set_value( mi::Sint16 value) final;

    void get_value( mi::Sint32& value) const final;

    void set_value( mi::Sint32 value) final;

    void get_value( mi::Sint64& value) const final;

    void set_value( mi::Sint64 value) final;

    void get_value( mi::Float32& value) const final;

    void set_value( mi::Float32 value) final;

    void get_value( mi::Float64& value) const final;

    void set_value( mi::Float64 value) final;

    // internal methods

private:
    /// Storage
    T m_storage{ 0};
};

/// Proxy implementation of interfaces derived from INumber
///
/// The proxy implementation Number_impl_proxy of interfaces derived from INumber does not own the
/// memory used to store the actual value. See the default implementation Number_impl for a variant
/// that does own the memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
///
/// Note that only a fixed set of types is permitted for the template parameters I and T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template<typename I, typename T>
class Number_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<I>, IProxy>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    const char* get_type_name() const final;

    void get_value( bool& value) const final;

    void set_value( bool value) final;

    void get_value( mi::Uint8& value) const final;

    void set_value( mi::Uint8 value) final;

    void get_value( mi::Uint16& value) const final;

    void set_value( mi::Uint16 value) final;

    void get_value( mi::Uint32& value) const final;

    void set_value( mi::Uint32 value) final;

    void get_value( mi::Uint64& value) const final;

    void set_value( mi::Uint64 value) final;

    void get_value( mi::Sint8& value) const final;

    void set_value( mi::Sint8 value) final;

    void get_value( mi::Sint16& value) const final;

    void set_value( mi::Sint16 value) final;

    void get_value( mi::Sint32& value) const final;

    void set_value( mi::Sint32 value) final;

    void get_value( mi::Sint64& value) const final;

    void set_value( mi::Sint64 value) final;

    void get_value( mi::Float32& value) const final;

    void set_value( mi::Float32 value) final;

    void get_value( mi::Float64& value) const final;

    void set_value( mi::Float64 value) final;

    // internal methods (of IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner);

    void release_referenced_memory();

private:
    /// Pointer to the storage
    T* m_pointer = nullptr;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;
};

/// Helper class to map from the actual type to the type name.
///
/// Used by Number_impl, Number_impl_proxy, and Compound_impl.
///
/// For the first two use cases the template parameter could also be the corresponding interface I.
/// However, in Compound_impl the interface I of the element type T is not available, only the
/// type T itself. However, if T is used, then we cannot use the Type_traits for mi::Size and
/// mi::Difference, because they are either mi::Uint32/mi::Sint32 or mi::Uint64/mi::Sint64,
/// but have different type names.
///
/// Note that only a fixed set of types is permitted for the template parameter T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template<typename T>
class Type_traits
{
public:
    static const char* get_type_name();
};

class Pointer_impl
  : public mi::base::Interface_implement<mi::IPointer>,
    public boost::noncopyable
{
public:
    /// The factory expects exactly one argument of type IString \p in argv.
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    Pointer_impl(
        const Factory* factory,
        DB::Transaction* transaction,
        const char* value_type_name);

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); };

    // public API methods (IPointer)

    mi::Sint32 set_pointer( mi::base::IInterface* pointer) final;

    mi::base::IInterface* get_pointer() const final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

private:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Hence, #create_instance() checks whether the constructor was
    /// successful.
    bool successfully_constructed() const { return m_successfully_constructed; }

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped pointers, always returns \c true. For typed pointers,
    /// #mi::IData::get_type_name() is compared against m_value_type_name.
    bool has_correct_value_type( const mi::base::IInterface* value) const;

    /// The transaction used for #Factory::assign_from_to().
    DB::Transaction* m_transaction = nullptr;

    /// The type name of the pointed value, or "Interface" for untyped pointers.
    std::string m_value_type_name;

    /// The type name of the pointer itself.
    std::string m_type_name;

    /// The actual pointer.
    mi::base::Handle<mi::base::IInterface> m_pointer;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed = false;
};

class Const_pointer_impl
  : public mi::base::Interface_implement<mi::IConst_pointer>,
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
    Const_pointer_impl(
       const Factory* factory,
        DB::Transaction* transaction,
       const char* value_type_name);

    // public API methods (IData)

    const char* get_type_name() const final { return m_type_name.c_str(); };

    // public API methods (IConst_pointer)

    mi::Sint32 set_pointer( const mi::base::IInterface* pointer) final;

    const mi::base::IInterface* get_pointer() const final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

private:

    /// Indicates whether the constructor successfully constructed the instance.
    ///
    /// Note that a structure type name can become invalid because it was unregistered between check
    /// and the actual construction. Hence, #create_instance() checks whether the constructor was
    /// successful.
    bool successfully_constructed() const { return m_successfully_constructed; }

    /// Indicates whether the value has the correct value type.
    ///
    /// For untyped pointers, always returns \c true. For typed pointers,
    /// #mi::IData::get_type_name() is compared against m_value_type_name.
    bool has_correct_value_type( const mi::base::IInterface* value) const;

    /// The transaction used for #Factory::assign_from_to().
    DB::Transaction* m_transaction = nullptr;

    /// The type name of the pointed value, or "Interface" for untyped pointers.
    std::string m_value_type_name;

    /// The type name of the pointer itself.
    std::string m_type_name;

   /// The actual pointer.
    mi::base::Handle<const mi::base::IInterface> m_pointer;

    /// Indicates whether the constructor successfully constructed the instance.
    /// \see #successfully_constructed()
    bool m_successfully_constructed = false;
};

/// Default implementation of IRef
///
/// The default implementation Ref_impl of IRef owns the memory used to store the actual
/// value. See the proxy implementation Ref_impl_proxy for a variant that does not own the
/// memory.
class Ref_impl
  : public mi::base::Interface_implement<mi::IRef>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    Ref_impl( const Factory* factory, DB::Transaction* transaction);

    // public API methods

    const char* get_type_name() const final { return "Ref"; };

    mi::Sint32 set_reference( const IInterface* db_element) final;

    mi::Sint32 set_reference( const char* name) final;

    const IInterface* get_reference() const final;

    IInterface* get_reference() final;

    const char* get_reference_name() const final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

private:
    /// The class factory.
    const Factory* m_factory = nullptr;

    /// The transaction used to convert names to tags, and to access the referenced element.
    DB::Transaction* m_transaction = nullptr;

    /// Storage
    DB::Tag m_storage;
};

/// Proxy implementation of IRef
///
/// The proxy implementation Ref_impl_proxy of IRef does not own the memory used to store the
/// actual value. See the default implementation Ref_impl for a variant that does own the
/// memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
class Ref_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IRef>, IProxy>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    Ref_impl_proxy( const Factory* factory, DB::Transaction* transaction);

    // public API methods

    const char* get_type_name() const final { return "Ref"; };

    mi::Sint32 set_reference( const IInterface* db_element) final;

    mi::Sint32 set_reference( const char* name) final;

    const IInterface* get_reference() const final;

    IInterface* get_reference() final;

    const char* get_reference_name() const final;

    // internal methods (IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) final;

    void release_referenced_memory() final;

    // internal methods

    /// Returns the embedded transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

private:
    /// The class factory.
    const Factory* m_factory = nullptr;

    /// The transaction used to convert names to tags, and to access the referenced element.
    DB::Transaction* m_transaction = nullptr;

    /// Pointer to the storage
    DB::Tag* m_pointer = nullptr;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;
};

/// Default implementation of IString
///
/// The default implementation String_impl of IString owns the memory used to store the actual
/// value. See the proxy implementation String_impl_proxy for a variant that does not own the
/// memory.
class String_impl
  : public mi::base::Interface_implement<mi::IString>,
    public boost::noncopyable
{
public:
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    /// Constructor
    ///
    /// The string is initialized to the empty string.
    String_impl( const char* str = nullptr) { set_c_str( str); }

    // public API methods

    const char* get_type_name() const final { return "String"; }

    const char* get_c_str() const final { return m_storage.c_str(); }

    void set_c_str( const char* str) final { m_storage = str ? str : ""; }

    // internal methods

private:
    /// Storage
    std::string m_storage;
};

/// Proxy implementation of IString
///
/// The proxy implementation String_impl_proxy of IString does not own the memory used to store the
/// actual value. See the default implementation String_impl for a variant that does own the
/// memory.
///
/// Users are not supposed to construct instances of this class directly. They might get
/// an instance of this class though, e.g., when accessing attributes.
class String_impl_proxy
  : public mi::base::Interface_merger<mi::base::Interface_implement<mi::IString>, IProxy>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    const char* get_type_name() const final { return "String"; };

    const char* get_c_str() const final;

    void set_c_str( const char* str) final;

    // internal methods (of IProxy)

    void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) final;

    void release_referenced_memory() final;

private:
    /// Pointer to the storage
    const char** m_pointer = nullptr;

    /// Owner of the storage
    ///
    /// The class uses reference counting on the owner to ensure that the pointer to the storage
    /// is valid.
    mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Uuid_impl
  : public mi::base::Interface_implement<mi::IUuid>,
    public boost::noncopyable
{
public:
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods (IData)

    const char* get_type_name() const final { return "Uuid"; };

    // public API methods (IUuid)

    void set_uuid( mi::base::Uuid uuid) final { m_uuid = uuid; }

    mi::base::Uuid get_uuid() const final { return m_uuid; }

private:

    /// The stored UUID.
    mi::base::Uuid m_uuid{ 0, 0, 0, 0 };
};

/// Default implementation of IVoid
class Void_impl
  : public mi::base::Interface_implement<mi::IVoid>,
    public boost::noncopyable
{
public:
    static mi::base::IInterface* create_instance(
        const Factory* factory,
        DB::Transaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    const char* get_type_name() const final { return "Void"; };

    // internal methods
};

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_VOID_IMPL_H
