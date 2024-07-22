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
 ** \brief Header for the IProxy and IAttribute_context interfaces.
 **/

#ifndef BASE_DATA_IDATA_IDATA_INTERFACES_H
#define BASE_DATA_IDATA_IDATA_INTERFACES_H

#include <utility>

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <base/data/db/i_db_tag.h>

namespace mi { class IData; }

namespace MI {

namespace ATTR { class Type; }
namespace DB { class Transaction; }
namespace NEURAY { class IDb_element; }

namespace IDATA {

/// The interface for proxies.
///
/// Most interfaces derived from IData have two implementations: a default implementation (which
/// owns the corresponding memory), and a proxy implementation (which does not own the corresponding
/// memory). The interfaces derived from ICompound have just one implementation which handles
/// both use cases simultaneously.
///
/// The proxy implementations are derived from this interface which offers a single method to set
/// the pointer to the memory to be used and the owner of that memory. The proxy implementations
/// are typically used to provide access to attributes (the owner is of type IAttribute_context
/// then). Another use case is to provide interface pointers for compound elements (the owner is
/// of type ICompound then).
class IProxy : public
    mi::base::Interface_declare<0x82240810,0xa358,0x4175,0x82,0xa7,0xc3,0x45,0xff,0x3a,0x3d,0x42,
                                mi::base::IInterface>
{
public:
    /// Sets the pointer to the memory and its owner.
    ///
    /// \param pointer       The pointer to the memory to be used by the proxy.
    /// \param owner         The owner of the memory. Reference counting on \p owner is used to
    ///                      ensure \p pointer is valid.
    virtual void set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner) = 0;

    /// Releases referenced memory.
    ///
    /// Releases memory that is referenced and owned by the attribute's data, but not part of
    /// the data itself. Note that memory is typically *not* released in the destructor because
    /// IProxy does not own the memory. However, if dynamic arrays are shrunk we need to release
    /// the memory references by elements that are destroyed.
    ///
    /// For IString, the method releases the memory of the string. For other instances of
    /// IData_simple the method does nothing. For IData_collection it recursively invokes itself
    /// on the collection's elements. For IDynamic_array, in addition, it releases the memory
    /// referenced in the Dynamic_array struct.
    ///
    /// Optimization: for ICompound, no recursion happens because there is no need to do so (neither
    /// strings nor collections can be elements of compounds).
    virtual void release_referenced_memory() = 0;
};

/// This small helper class retains an ATTR::Attribute and the corresponding DB element.
///
/// It provides the necessary context for accesses to array elements or structure members.
///
/// Retaining the ATTR::Attribute is important if a user removes the attribute while he/she still
/// holds a reference for accessing/editing the attribute. Not sure whether it is necessary to
/// retain the DB element as well.
class IAttribute_context : public
    mi::base::Interface_declare<0x923a0667,0xeea8,0x4e64,0x9c,0x9d,0x9e,0x89,0xb4,0xa7,0xf8,0x89,
                                mi::base::IInterface>
{
public:
    /// Returns the DB element that is retained by the attribute context.
    virtual const NEURAY::IDb_element* get_db_element() const = 0;

    /// Returns the ATTR type for the (part of the) attribute identified by \p attribute_name.
    ///
    /// Note that for array elements the returned type is not correct. ATTR::Type::lookup() returns
    /// a type tree where the top-level element has the array size of the array itself (and
    /// not 1 as one would expect for a non-nested array). This is due to the fact that
    /// ATTR::Type::lookup() returns a pointer to a subtree of the type tree of the attribute
    /// itself.
    ///
    /// \return    The type, or \c NULL in case of failure (e.g., \p attribute_name does
    ///           identify a part of the attribute).
    virtual const ATTR::Type* get_type( const char* attribute_name) const = 0;

    /// Compute the memory address for the (part of the) attribute identified by \p attribute_name.
    ///
    /// \return   The memory address, or \c NULL in case of failure (e.g., \p attribute_name does
    ///           identify a part of the attribute).
    virtual void* get_address( const char* attribute_name) const = 0;

    /// Retrieves the (part of the) attribute identified by \p attribute_name.
    virtual mi::IData* get_attribute( const char* attribute_name) const = 0;

    /// Returns the size of one array element for the (part of the) attribute identified by
    /// \p attribute_name.
    ///
    /// See #ATTR::Type::sizeof_elem().
    virtual mi::Size get_sizeof_elem( const char* attribute_name) const = 0;

    /// Indicates whether the owning DB element can reference the given tag.
    virtual bool can_reference_tag( DB::Tag tag) const = 0;
};

/// This callback interface provides the support for the implementation of #mi::IRef.
class ITag_handler : public
    mi::base::Interface_declare<0xe1c09717,0xca2c,0x44a9,0x91,0xf7,0x2f,0x6d,0xec,0xd4,0xe2,0x58,
                                mi::base::IInterface>
{
public:
    /// Looks up the name of a tag (within the context of this transaction).
    ///
    /// Wrapper for #DB::Transaction::tag_to_name().
    virtual const char* tag_to_name( DB::Transaction* transaction, DB::Tag tag) = 0;

    /// Looks up the name of a tag (within the context of this transaction).
    ///
    /// Wrapper for #DB::Transaction::name_to_tag().
    virtual DB::Tag name_to_tag( DB::Transaction* transaction, const char* name) = 0;

    /// Performs a database lookup for a given tag (access).
    ///
    /// Similar to #mi::neuraylib::ITransaction::access(), but for a tag.
    virtual const mi::base::IInterface* access_tag( DB::Transaction* transaction, DB::Tag tag) = 0;

    /// Performs a database lookup for a given tag (edit).
    ///
    /// Similar to #mi::neuraylib::ITransaction::edit(), but for a tag.
    virtual mi::base::IInterface* edit_tag( DB::Transaction* transaction, DB::Tag tag) = 0;

    /// Extracts a tag from a dababase element.
    ///
    /// \param db_element    The database element to extract the tag from.
    /// \return
    ///                      -  0: Success.
    ///                      - -2: \p db_element does not represent a DB element.
    ///                      - -3: The DB element has not yet been stored in the database, and
    ///                            therefore, has no tag.
    virtual std::pair<DB::Tag,mi::Sint32> get_tag( const mi::base::IInterface* db_element) = 0;
};

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_INTERFACES_H
