/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the ITransaction implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_TRANSACTION_IMPL_H
#define API_API_NEURAY_NEURAY_TRANSACTION_IMPL_H

#include <mi/neuraylib/itransaction.h>

#include <regex>
#include <set>
#include <string>
#include <mi/base/lock.h>
#include <mi/base/interface_implement.h>
#include <boost/core/noncopyable.hpp>
#include <base/data/db/i_db_tag.h>
#include <base/data/serial/i_serial_classid.h>

#include "i_neuray_transaction.h"

namespace mi { class IDynamic_array; }

namespace MI {

namespace DB { class Element_base; class Transaction; }

namespace NEURAY {

class Class_factory;
class Db_element_impl_base;
class Expression_factory;
class IDb_element;
class Value_factory;
class Type_factory;

class Transaction_impl
  : public mi::base::Interface_implement<NEURAY::ITransaction>,
    public boost::noncopyable
{
public:

    /// Constructs a Transaction_impl
    ///
    /// \param db_transaction            The internal DB transaction wrapped by this instance
    /// \param class_factory             The class factory
    /// \param commit_or_abort_warning   By default, the destructor emits a warning if the
    ///                                  transaction was not committed or aborted. This warning can
    ///                                  suppressed which is needed if the DB transaction is only
    ///                                  wrapped temporarily and remains open after the destruction
    ///                                  this instance.
    Transaction_impl(
        DB::Transaction* db_transaction,
        const Class_factory* class_factory,
        bool commit_or_abort_warning = true);

    /// Destructs a Transaction_impl
    ~Transaction_impl();

    // public API methods

    mi::Sint32 commit();

    void abort();

    bool is_open() const;

    mi::base::IInterface* create(
        const char* type_name,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    using mi::neuraylib::ITransaction::create;

    mi::Sint32 store( mi::base::IInterface* db_element, const char* name, mi::Uint8 privacy);

    const mi::base::IInterface* access( const char* name);

    using mi::neuraylib::ITransaction::access;

    mi::base::IInterface* edit( const char* name);

    using mi::neuraylib::ITransaction::edit;

    mi::Sint32 copy( const char* source, const char* target, mi::Uint8 privacy);

    mi::Sint32 remove( const char* name, bool only_localized);

    const char* name_of( const mi::base::IInterface* db_element) const;

    const char* get_time_stamp() const;

    const char* get_time_stamp( const char* element) const;

    bool has_changed_since_time_stamp( const char* element, const char* time_stamp) const;

    const char* get_id() const;

    mi::neuraylib::IScope* get_scope() const;

    mi::IArray* list_elements(
        const char* root_element, const char* name_pattern, const mi::IArray* type_names) const;

    mi::Sint32 get_privacy_level( const char* name) const;

    // methods of IInterface

    /// Override default implementation to handle mi::neuraylib::IDice_transaction
    /// (if MI_PRODUCT_DICE is defined).
    const mi::base::IInterface* get_interface( const mi::base::Uuid & interface_id) const;

    /// Override default implementation to handle mi::neuraylib::IDice_transaction
    /// (if MI_PRODUCT_DICE is defined).
    mi::base::IInterface* get_interface( const mi::base::Uuid & interface_id);

    // methods of NEURAY::ITransaction

    // Returns the internal transaction
    DB::Transaction* get_db_transaction() const;

    // internal methods

    /// Returns the element with tag \p tag in the database.
    ///
    /// \param tag    The tag of the element to return.
    /// \return       The element in the database or \c NULL if no such element exists or
    ///               the transaction is already closed.
    mi::base::IInterface* edit( DB::Tag tag);

    template<class T>
    T* edit( DB::Tag tag)
    {
        mi::base::IInterface* ptr_iinterface = edit( tag);
        if( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Returns the element with tag \p tag in the database.
    ///
    /// \param name   The tag of the element to return.
    /// \return       The element in the database or \c NULL if no such element exists or
    ///               the transaction is already closed.
    const mi::base::IInterface* access( const DB::Tag tag);

    template<class T>
    const T* access( DB::Tag tag)
    {
        const mi::base::IInterface* ptr_iinterface = access( tag);
        if( !ptr_iinterface)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// See #mi::neuraylib::IDice_transaction::get_time_stamp().
    const char* get_time_stamp( DB::Tag tag) const;

    /// See #mi::neuraylib::IDice_transaction::has_changed_since_time_stamp().
    bool has_changed_since_time_stamp( DB::Tag tag, const char* time_stamp) const;

    /// See #mi::neuraylib::IDice_transaction::get_privacy_level().
    mi::Sint32 get_privacy_level( DB::Tag tag) const;

    /// Returns the class factory.
    ///
    /// \note This method does \em not increase the reference count of the return value.
    const Class_factory* get_class_factory() const;

    /// Returns the tag to be used for storing a DB element of that name.
    ///
    /// Calls DB::Transaction::name_to_tag() followed by get_tag_for_store(DB::Tag).
    DB::Tag get_tag_for_store( const char* name);

    /// Returns the tag to be used for storing a DB element with that tag.
    ///
    /// Returns the passed tag if it is not the null tag and (has not been been marked for removal
    /// or is still referenced). Otherwise, it reserves a new tag and returns it.
    ///
    /// We must not reuse an existing tag if DB::Transaction::remove() was called for that tag and
    /// and its reference count is 0: The DB requires that a tag that has been removed, i.e.,
    /// requested for removal, must not be reused (at least those tags with reference count 0 might
    /// disappear at any time -- even within a transaction).
    DB::Tag get_tag_for_store( DB::Tag tag);

    /// Record the construction of (an API class for) a DB element.
    ///
    /// Funtionality duplicated from the Db_element_tracker, but in the context of the transaction.
    void add_element( const Db_element_impl_base* db_element);

    /// Record the destruction of (an API class for) a DB element.
    ///
    /// Funtionality duplicated from the Db_element_tracker, but in the context of the transaction.
    void remove_element( const Db_element_impl_base* db_element);

    /// Returns the MDL type factory for this transaction.
    Type_factory* get_type_factory();

    /// Returns the MDL value factory for this transaction.
    Value_factory* get_value_factory();

    /// Returns the MDL expression factory for this transaction.
    Expression_factory* get_expression_factory();

private:

    /// Checks that m_elements is empty and emits suitable error messages otherwise.
    void check_no_referenced_elements( const char* committed_or_aborted);

    /// Recursive functions used to implement list_elements().
    ///
    /// The method performs a DFS post-order graph traversal starting at \p tag. All scene elements
    /// whose name matches an optional regular expression and an optional set of class IDs are
    /// reported. The post-order traversal ensures ensures that the elements are in the correct
    /// order needed e.g. for exporters.
    ///
    /// \param tag          The graph traversal starts here.
    /// \param name_regex   Only elements with matching name are reported (unless \c NULL).
    /// \param class_ids    Only elements with matching class ID are reported (unless \c NULL).
    /// \param[out] result  The found elements.
    /// \param tags_seen    Used to skip already handled graph nodes.
    void list_elements_internal(
        DB::Tag tag,
        const std::wregex* name_regex,
        const std::set<SERIAL::Class_id>* class_ids,
        mi::IDynamic_array* result,
        std::set<DB::Tag>& tags_seen) const;

    /// The DB transaction used by this instance.
    DB::Transaction* m_db_transaction;

    /// The class factory.
    const Class_factory* m_class_factory;

    /// Flag that triggers the commit-or-abort warning in the destructor.
    ///
    /// Note that this flag is explicitly set to \c false in abort() or commit() (and can be
    /// initialized to \c false via the constructor) instead of using is_open() in the destructor.
    /// The reason is that after commit() or abort() was called, is_open() might still return
    /// \c true for a short amount of time (at least if networking is enabled).
    bool m_commit_or_abort_warning;

    /// Caches the result of the last get_time_stamp() call.
    mutable std::string m_timestamp;

    /// ID of the transaction (as number)
    mi::Uint32 m_id_as_uint;

    /// ID of the transaction (as string).
    mutable std::string m_id_as_string;

    /// Lock for #m_elements.
    mi::base::Lock m_elements_lock;

    typedef std::set<const Db_element_impl_base*> Elements;

    /// Contains the DB elements currently in use by the API for this transaction.
    ///
    /// Needs #m_elements_lock. Transactions are not multi-threading-safe anyway, but even the
    /// release of a DB element manipulates the map, which is not necessarily perceived as "using
    /// the transaction". So we are a bit more carefule here for this container.
    Elements m_elements;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_TRANSACTION_IMPL_H

