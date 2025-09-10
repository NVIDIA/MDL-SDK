/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_TRANSACTION_H
#define BASE_DATA_DB_I_DB_TRANSACTION_H

#include <memory>
#include <utility>
#include <vector>

#include <boost/core/noncopyable.hpp>

#include "i_db_journal_type.h"
#include "i_db_scope.h"
#include "i_db_tag.h"

#include <mi/base/types.h>
#include <base/data/serial/i_serial_classid.h>

namespace MI {

namespace SCHED { class Job_base; }

namespace DB {

class Element_base;
class Fragmented_job;
class IExecution_listener;
class Info;
class Scope;

/// A transaction provides a consistent view on the database.
///
/// This view on the database is isolated from changes by other (parallel) transactions. Eventually,
/// each transaction must be either committed or aborted, i.e., all changes become either atomically
/// visible to transactions started afterwards, or not at all.
///
/// Transactions are associated with a scope of the database and can be created with
/// #Scope::start_transaction().
///
/// Transactions are not thread-safe. If you use a particular transaction from multiple threads,
/// then you have to serialize all transaction uses. TODO This rule is frequently violated by
/// fragmented jobs. It is unclear to what extend this is safely possible.
///
/// See #mi::neuraylib::ITransaction for semantics of concurrent transactions, and for concurrent
/// accesses to the very same database element within one particular transaction.
///
/// \section tags_and_names Tags and names
///
/// Tags are mandatory and the primary key to identify DB element/jobs, names are optional and a
/// secondary key. Most methods and reference counting operates on tags, only few methods also work
/// on names. Conversion methods name_to_tag() and tag_to_name() exist on DB::Transaction.
///
/// Usually there is a 1:1 relation between the used (named) tags and names, however that is not
/// required. Processing similar content in independent scopes often leads to m:1 relations. Also,
/// when storing a DB element with a name that is eligible for GC one often allocates a new tag,
/// leading to an m:1 relation. Even 1:n or m:n relations are legal from a DB point of view.
/// However, the effects can be quite confusing for users, and its recommended to avoid them.
///
/// \section levels Store level and privacy level
///
/// When storing database elements one can specify two parameters, the store level and the privacy
/// level. These two parameters control in which scope the element is stored, from which scopes it
/// is visible for later accesses, and what happens if an element is edited. ("Edit level" might be
/// a better name for privacy level). Note that the store level is always less than or equal to the
/// privacy level (at least conceptually, actual arguments might violate that rule and are clamped
/// accordingly).
///
/// The store level indicates in which scope of the scope stack an element is stored. The scope
/// is selected as follows:
/// - If the store level is larger than the privacy level, then it is set to the privacy level.
/// - Pick the most local scope from the scope stack of the current transaction whose level is less
///   than or equal to the requested store level (scope levels do not need to be consecutive).
///
/// Such an element is visible for transactions associated with the selected scope or its child
/// scopes. It is not visible for transactions associated with any parent or other scopes.
///
/// The privacy level indicates in which scope of the scope stack the copy created for editing an
/// element is stored. The scope is selected as follows:
/// - Pick the most local scope from the scope stack of the current transaction whose level is less
///   than or equal to the requested privacy level (scope levels do not need to be consecutive).
/// In other words, when editing a database element, it is automatically localized to its privacy
/// level (and avoids one additional copy of the database element).
///
/// \note The API does not distinguish between store level and privacy level. It only exposes the
///       privacy level and uses the default store level (currently 255, which is internally
///       effectively clamped to the privacy level). The API also rejects privacy levels larger
///       than the level of the current scope, instead of silently adjusting them -- but note the
///       special meaning of the constant mi::neuraylib::ITransaction::LOCAL_SCOPE = 255.
class Transaction : private boost::noncopyable
{
public:
    /// \name Reference counting
    //@{

    /// Pins the transaction, incrementing its reference count.
    virtual void pin() = 0;

    /// Unpins the transaction, decrementing its reference count.
    virtual void unpin() = 0;

    //@}
    /// \name Information about the transaction itself
    //@{

    /// Returns the ID of this transaction.
    virtual Transaction_id get_id() const = 0;

    /// Returns the scope of this transaction. RCS:NEU
    virtual Scope* get_scope() = 0;

    /// Returns the sequence number for the next update within this transaction.
    ///
    /// Note that this method does \em not increment the sequence number.
    virtual mi::Uint32 get_next_sequence_number() const = 0;

    //@}
    /// \name Committing and aborting the transaction
    //@{

    /// Commits the transaction.
    ///
    /// Committing a transaction is one of two ways of closing an open transaction, such that all
    /// changes done from within that transaction become visible for all transactions started
    /// later. Although the user may no longer use the transaction after committing it, the
    /// transaction might continue living in the database for an unspecified amount of time.
    ///
    /// \note The transaction must no longer be used in any way after a call to this method
    ///       \em without explicitly pinning it beforehand.
    ///
    /// \return   \c true in case of success, \c false otherwise, e.g., if the transaction was not
    ///           open, or a host contributing to the transaction failed before the transaction was
    ///           committed.
    virtual bool commit() = 0;

    /// Aborts the transaction.
    ///
    /// Aborting a transaction is one of two ways of closing an open transaction, such that all
    /// changes done from within that transaction are thrown away and no other transaction will
    /// ever see them. Although the user may no longer use the transaction after aborting it, the
    /// transaction might continue living in the database for an unspecified amount of time.
    ///
    /// \note The transaction must no longer be used in any way after a call to this method
    ///       \em without explicitly pinning it beforehand.
    virtual void abort() = 0;

    /// Indicates whether a transaction is still open.
    ///
    /// \param closing_is_open   Controls whether a transaction that is in the process of being
    ///                          closed, but not fully closed, as open.
    virtual bool is_open( bool closing_is_open = true) const = 0;

    /// Blocks the transaction from committing or aborting.
    ///
    /// This call increments a counter. As long as the counter is larger than 0, an attempt to
    /// call #commit() or #abort() will block until the counter drops to 0.
    ///
    /// \return   \c true if the counter was increased (i.e., the transaction was open), or
    ///           \c false the transaction was already closing, committed, or aborted.
    virtual bool block_commit_or_abort() = 0;

    /// Unblocks the transaction from committing or aborting.
    ///
    /// This call decrements a counter. As long as the counter is larger than 0, an attempt to
    /// call #commit() or #abort() will block until the counter drops to 0.
    ///
    /// \return   \c true if the counter was decreased (i.e., the transaction was open or closing
    ///           and the counter was larger than 0), or \c false otherwise.
    virtual bool unblock_commit_or_abort() = 0;

    //@}
    /// \name Accessing/editing database elements
    //@{

    /// Returns the info for the requested tag, intended for access (const) operations.
    ///
    /// The method will most likely not return for invalid tags (NULL tags or otherwise invalid),
    /// but emit a fatal log message (which is not supposed to return) and/or abort the process.
    ///
    /// Low-level interface function, using DB::Access instead is strongly recommended.
    ///
    /// \param tag     The tag to look up.
    /// \return        The info for that tag. RCS:ICE
    virtual Info* access_element( Tag tag) = 0;

    /// Returns the info for the requested name, intended for access (const) operations.
    ///
    /// \param name    The name to look up.
    /// \return        The info for that name, or \c nullptr if there is no info for that name.
    ///                RCS:ICE
    virtual Info* access_element( const char* name) = 0;

    /// Returns the info for the requested tag, intended for edit (non-const) operations.
    ///
    /// Needs a call to #finish_edit() when done editing.
    ///
    /// The method will most likely not return for invalid tags (NULL tags or otherwise invalid),
    /// but emit a fatal log message (which is not supposed to return) and/or abort the process.
    ///
    /// Low-level interface function, using DB::Edit instead is strongly recommended.
    ///
    /// \note It is not possible to edit database elements representing job results.
    ///
    /// \param tag     The tag to look up.
    /// \return        The info for that tag. RCS:ICE
    virtual Info* edit_element( Tag tag) = 0;

    /// Returns the info for the requested name, intended for edit (non-const) operations.
    ///
    /// Needs a call to #finish_edit() when done editing.
    ///
    /// \note It is not possible to edit database elements representing job results.
    ///
    /// \param name    The name to look up.
    /// \return        The info for that name, or \c nullptr if there is no info for that name.
    ///                RCS:ICE
    virtual Info* edit_element( const char* name) = 0;

    /// Finishes a previously started edit operation started with #edit_element().
    ///
    /// This comprises e.g. updating the list of database elements referenced by this element,
    /// distributing changes over the network, and recording the journal flags.
    ///
    /// Low-level interface function, using DB::Edit instead is strongly recommended.
    ///
    /// \param info           The info for the edit operation. RCS:ICR
    /// \param journal_type   Type of changes to be recorded for this edit operation.
    virtual void finish_edit( Info* info, Journal_type journal_type) = 0;

    //@}
    /// \name Storing database elements
    ///
    /// The #store() and #store_for_reference_counting() methods below come in two variants: tag as
    /// return value and tag as first parameter. The first variants reserve the tag itself
    /// internally and return the reserved tag. But sometimes it is more practical to reserve the
    /// tag upfront explicitly (because it is stored by the user in some other data structure), and
    /// to specify that tag later when storing. This is the purpose of the second variants of the
    /// #store() and #store_for_reference_counting() methods below. The second variants also allow
    /// to specify an initial value for the journal flags, whereas the first variants use
    /// JOURNAL_NONE.
    ///
    //@{

    /// Reserves and returns a free tag.
    virtual Tag reserve_tag() = 0;

    /// Stores a DB element in the database (tag as return value).
    ///
    /// The journal flags are set to JOURNAL_NONE.
    ///
    /// \param element                  The DB element to store. RCS:TRO
    /// \param name                     Optional name for the DB element. The empty string is not
    ///                                 valid.
    /// \param privacy_level            Privacy level of the DB element.
    /// \param store_level              Store level of the DB element.
    /// \return                         The assigned tag.
    virtual Tag store(
        Element_base* element,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB element in the database (tag as parameter).
    ///
    /// \param tag                      The tag to be used for the DB element. Retrieved from
    ///                                 #reserve_tag(), or from a previous store operation or
    ///                                 #name_to_tag() (to overwrite the current version of the DB
    ///                                 element).
    /// \param element                  The DB element to store. RCS:TRO
    /// \param name                     Optional name for the DB element. The empty string is not
    ///                                 valid.
    /// \param privacy_level            Privacy level of the DB element.
    /// \param journal_type             The journal flags of the DB element.
    /// \param store_level              Store level of the DB element.
    /// \return                         The assigned tag.
    virtual void store(
        Tag tag,
        Element_base* element,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_ALL,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB element in the database (tag as return value).
    ///
    /// Same as #store(Element_base*,...) above, but with an additional call to #remove().
    virtual Tag store_for_reference_counting(
        Element_base* element,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB element in the database (tag as parameter).
    ///
    /// Same as #store(Tag,Element_base*,...) above, but with an additional call to #remove().
    virtual void store_for_reference_counting(
        Tag tag,
        Element_base* element,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_ALL,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB element or Job in the database.
    ///
    /// This functions the same as the overloads but yields a typed tag.
    template <typename T,
              typename = std::enable_if_t<   std::is_convertible_v<T*,Element_base*>
                                          || std::is_convertible_v<T*,SCHED::Job_base*>>>
    Typed_tag<T> store(
            T* element_or_job,
            Privacy_level privacy_level = 0,
            Privacy_level store_level = 255,
            const char* name = nullptr)
    {
        return Typed_tag<T>{store(element_or_job, name, privacy_level, store_level)};
    }

    /// Stores a DB element or Job in the database.
    ///
    /// This functions the same as the overloads but yields a typed tag.
    template <typename T,
              typename = std::enable_if_t<   std::is_convertible_v<T*,Element_base*>
                                          || std::is_convertible_v<T*,SCHED::Job_base*>>>
    Typed_tag<T> store_for_reference_counting(
            T* element_or_job,
            Privacy_level privacy_level = 0,
            Privacy_level store_level = 255,
            const char* name = nullptr)
    {
        return Typed_tag<T>{store_for_reference_counting(
                element_or_job, name, privacy_level, store_level)};
    }

    //@}
    /// \name Storing database jobs
    //@{

    /// Stores a DB job in the database (tag as return value).
    ///
    /// The journal flags are set to JOURNAL_NONE.
    ///
    /// \param job                      The DB job to store. RCS:TRO
    /// \param name                     Optional name for the DB job. The empty string is not
    ///                                 valid.
    /// \param privacy_level            Privacy level of the DB job.
    /// \param store_level              Store level of the DB job.
    /// \return                         The assigned tag.
    virtual Tag store(
        SCHED::Job_base* job,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB job in the database (tag as parameter).
    ///
    /// \param tag                      The tag to be used for the DB job. Retrieved from
    ///                                 #reserve_tag(), or from a previous store operation or
    ///                                 #name_to_tag() (to overwrite the current version of the DB
    ///                                 job).
    /// \param job                      The DB job to store. RCS:TRO
    /// \param name                     Optional name for the DB job. The empty string is not
    ///                                 valid.
    /// \param privacy_level            Privacy level of the DB job.
    /// \param journal_type             The journal flags of the DB element.
    /// \param store_level              Store level of the DB job.
    /// \return                         The assigned tag.
    virtual void store(
        Tag tag,
        SCHED::Job_base* job,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_NONE,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB job in the database (tag as return value).
    ///
    /// Same as #store(Job_base*,...) above, but with an additional call to #remove().
    virtual Tag store_for_reference_counting(
        SCHED::Job_base* job,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Stores a DB job in the database (tag as parameter).
    ///
    /// Same as #store(Tag,Job_base*,...) above, but with an additional call to #remove().
    virtual void store_for_reference_counting(
        Tag tag,
        SCHED::Job_base* job,
        const char* name = nullptr,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_NONE,
        Privacy_level store_level = 255) = 0;

    //@}
    /// \name Localization and removal
    //@{

    /// Localizes a tag to the given privacy level.
    ///
    /// \param tag                      The tag to be localized.
    /// \param privacy_level            Localize to this privacy level.
    /// \param journal_type             Type of changes to be recorded for this localize operation.
    virtual void localize(
        Tag tag, Privacy_level privacy_level, Journal_type journal_type = JOURNAL_NONE) = 0;

    /// Marks a tag for removal from the database.
    ///
    /// \par Global removals
    ///
    /// The purpose of global removals is to mark all versions of a tag for garbage collection.
    /// Such a marker has no effect while the tag is still referenced (in any scope) by other
    /// database elements or while the transaction where the removal request was made is still open.
    /// When these conditions do no longer apply, the tag becomes eligible for garbage collection
    /// and must no longer be used in any way. There is no guarantee when the garbage collection
    /// will actually remove the tag.
    ///
    /// This implies that a #remove() call might actually remove an element that was stored later
    /// under the same tag. This can potentially lead to invalid tag accesses. Those cases can be
    /// avoided by using #Database::garbage_collection() after a transaction was committed and
    /// before starting the next one to force garbage collection of all possible elements.
    ///
    /// \par Local removals
    ///
    /// The purpose of local removals is to undo the effects of an earlier localization via
    /// #DB::Transaction::localize(). A local removal request requires that a tag version exists in
    /// the scope of the transaction, and at least one more tag version exists in one of the parent
    /// scopes. The effect of a local removal request is to immediately hide the tag version the
    /// scope of the transaction (the \em local copy), and to make the next tag version in one of
    /// the parent scopes accessible from the very same transaction. The hidden local copy will be
    /// lazily removed by the garbage collection of the DB. There is no guarantee when this will
    /// happen.
    ///
    /// \param tag                      The tag to be marked for removal.
    /// \param remove_local_copy        \c false for global removals (the default) or \c true for
    ///                                 local removals. The flag is ignored in favor of global
    ///                                 removals if the transaction belongs to the global scope.
    /// \return                         \c true in case of success (including subsequent global
    ///                                 removals on tags already marked for global removal), or \c
    ///                                 false otherwise (local removal and the tag version is
    ///                                 missing in the the scope of the transaction or missing in
    ///                                 the parent scopes).
    virtual bool remove( Tag tag, bool remove_local_copy = false) = 0;

    //@}
    /// \name Translation between names and tags
    //@{

    /// Looks up the name of a tag (within the context of this transaction).
    ///
    /// \note There is no guarantee that #tag_to_name() followed by #name_to_tag() produces the
    ///       original tag.
    ///
    /// \param tag   The tag to look up.
    /// \return      The corresponding name, or \c nullptr if the tag has no associated name.
    virtual const char* tag_to_name( Tag tag) = 0;

    /// Provides context information for #name_to_tag().
    enum Name_to_tag_context {
        /// The call happens in the context of a store operation (including targets of copy
        /// operations).
        STORE_CONTEXT,
        /// The call happens in the context of a lookup operation.
        LOOKUP_CONTEXT,
        /// The call happens in an unknown context.
        UNKNOWN_CONTEXT
    };

    /// Looks up the tag for a name (within the context of this transaction, safe version).
    ///
    /// This method differs from #name_to_tag() in the following way:
    /// - (1) No difference if no tag is found for the given name.
    /// - (2) No difference if the found tag is not flagged for removal.
    /// - (3) If flagged for removal and (the reference count is equal to 0 and \p context
    ///       is \c STORE_CONTEXT), pretend that no tag was found.
    /// - (4) If flagged for removal and (the reference count is larger than 0 or \p context
    ///       is not \c STORE_CONTEXT), keep the found info pinned until the end of the transaction.
    ///
    /// Case (3): The tag can be garbage collected at any time after this method returned. Instead
    /// of handing out a tag that might lead to an invalid tag access, pretend that the lookup
    /// failed, with the intention that the caller requests a new tag for store or copy operations.
    /// A new tag also avoids that the removal flag carries over to a new version of that database
    /// element.
    ///
    /// Case (4): A garbage collection of all referencing elements would decrease the reference
    /// count of this tag to 0, and then it can be garbage collected, too. However, we do not know
    /// whether the referencing elements will be garbage collected at all, and unconditionally
    /// letting the look up fail is logically wrong. Instead we pin the info until the end of the
    /// transaction. This gives callers a chance to store a new version of that database element,
    /// and to reference it again (or keep it referenced). The removal flag will carry over to a
    /// potentially new version of that database element.
    ///
    /// \note There is no guarantee that name_to_tag() followed by tag_to_name() produces
    ///       the original name.
    ///
    /// \see #name_to_tag()
    ///
    /// \param name      The name to look up.
    /// \param context   The context of the operation controls whether case (3) above is a feasible
    ///                  outcome. Note that \c LOOKUP_CONTEXT and \c UNKNOWN_CONTEXT are currently
    ///                  treated the same way.
    /// \return          The corresponding tag, or the invalid tag if the name was not found.
    virtual Tag name_to_tag( const char* name, Name_to_tag_context context = UNKNOWN_CONTEXT) = 0;

    /// Looks up the name of a tag (within the context of this transaction, unsafe version).
    ///
    /// \note There is no guarantee that #tag_to_name_unsafe () followed by #name_to_tag() produces
    ///       the original tag.
    ///
    /// \param tag   The tag to look up.
    /// \return      The corresponding name, or \c nullptr if the tag has no associated name.
    virtual Tag name_to_tag_unsafe( const char* name) = 0;

    //@}
    /// \name Information about a specific tag
    //@{

    /// Indicates whether a tag corresponds to a DB job.
    ///
    /// \param tag   The tag to look up.
    /// \return      \c true, if the tag corresponds to a DB job, and \c false otherwise, i.e.,
    ///              corresponds to a DB element.
    virtual bool get_tag_is_job( Tag tag) = 0;

    /// Returns the class ID of a tag.
    ///
    /// The method can fail, e.g., if the element is not in the cache or if the tag corresponds to
    /// a job. In such cases the database will not fetch the element or execute the job to
    /// determine the class ID. When the class ID is really needed (or rather, a check whether the
    /// element has a given class ID), then a caller should use
    /// DB::Access<DB::Element>::get_base_ptr()->is_type_of() instead.
    ///
    /// \param tag   The tag to look up.
    /// \return      The class ID, or SERIAL::class_id_unknown in case of failure.
    virtual SERIAL::Class_id get_class_id( Tag tag) = 0;

    /// Return the privacy level of a tag.
    ///
    /// \param tag   The tag to look up.
    /// \return      The privacy level, or 0 in case of failure.
    virtual Privacy_level get_tag_privacy_level( Tag tag) = 0;

    /// Return the store level of a tag.
    ///
    /// \param tag   The tag to look up.
    /// \return      The store level, or 0 in case of failure
    virtual Privacy_level get_tag_store_level( Tag tag) = 0;

    /// Returns the unique ID of a tag.
    ///
    /// \param tag   The tag to look up.
    /// \return      The corresponding tag version, or default-constructed in case of failure.
    virtual Tag_version get_tag_version( Tag tag) = 0;

    /// Returns the reference count for a tag.
    ///
    /// \param tag   The tag to look up.
    /// \return      The reference count, or 0 in case of failure
    virtual mi::Uint32 get_tag_reference_count( Tag tag) = 0;

    /// Indicates whether another tag can be referenced from a given scope level.
    ///
    /// This is the case if the referenced tag has at least one version in a scope with privacy
    /// level between 0 and the referencing level. Otherwise, one can trigger an invalid tag access
    /// because the referenced tag can not be seen from the scope at the referencing level (and
    /// maybe some of its children).
    ///
    /// \param referencing_level   The tag which is meant to reference the other tag will be stored
    ///                            in this scope.
    /// \param referenced_tag      The tag which is meant to be referenced from the referencing
    ///                            level.
    /// \return                    \c true if the reference is valid, \c false otherwise.
    virtual bool can_reference_tag( Privacy_level referencing_level, Tag referenced_tag) = 0;

    /// Indicates whether a given tag can reference another tag.
    ///
    /// This is the case if the privacy level of the referencing tag is not smaller than the
    /// privacy level of the referenced tag. Otherwise, one can trigger an invalid tag access
    /// because the referenced tag can not be seen from all scopes that provide the referencing
    /// tag.
    ///
    /// \param referencing_tag     The tag which is meant to reference the other tag.
    /// \param referenced_tag      The tag which is meant to be referenced by the other tag.
    /// \return                    \c true if the reference is valid, \c false otherwise.
    virtual bool can_reference_tag( Tag referencing_tag, Tag referenced_tag) = 0;

    /// Indicates whether #remove() has already been called on a tag.
    ///
    /// \param tag                 The tag to look up.
    /// \return                    \c true if #remove() has already been called, \c false
    ///                            otherwise.
    virtual bool get_tag_is_removed( Tag tag) = 0;

    //@}
    /// \name Journal
    //@{

    /// Returns a list of database changes since some point in time.
    ///
    /// The start of the query range is given by the pair of \p last_transaction_id and
    /// \p last_transaction_change_version (which usually matches the last call of this function,
    /// hence the parameter names). The end of the query range is implicitly defined by the
    /// current sequence number (see #get_next_sequence_number()) of the current transaction.
    ///
    /// \note This variant includes changes from the current transaction, in contrast to
    ///       #DB::IScope::get_journal().
    ///
    /// \param last_transaction_id               The transaction ID of start of the query range.
    /// \param last_transaction_change_version   The sequence number of the start of the query
    ///                                          range.
    /// \param journal_type                      A filter for the journal type. Only changes with
    ///                                          a non-zero intersection with \p journal_type will
    ///                                          be reported. Use #JOURNAL_ALL to not filter based
    ///                                          on the journal type.
    /// \param lookup_parents                    Indicates whether parent scopes should be
    ///                                          considered, too.
    /// \return                                  A vector of tag/journal type pairs describing the
    ///                                          changes in the relevant range matching the filter.
    ///                                          Returns \c nullptr if the queried range is too
    ///                                          large for the journal capacity. In this case the
    ///                                          caller should assume that all relevant database
    ///                                          elements have changed. \n
    ///                                          The same tag might occur multiple times, with
    ///                                          identical or different journal types.
    virtual std::unique_ptr<Journal_query_result> get_journal(
        Transaction_id last_transaction_id,
        mi::Uint32 last_transaction_change_version,
        Journal_type journal_type,
        bool lookup_parents) = 0;

    //@}
    /// \name Fragmented jobs
    //@{

    /// Executes a fragmented job, splitting it in a given number of fragments (synchronous).
    ///
    /// This method will not return before all fragments have been executed. The fragments may be
    /// executed in any number of threads and on any number of hosts.
    ///
    /// \param job                      The fragmented job to be executed. RCS:NEU
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero unless the scheduling mode
    ///                                 of the job is ONCE_PER_HOST.
    ///                                 \note If the job has a scheduling mode which defines the
    ///                                 number of fragments implicitly (e.g. ONCE_PER_HOST) and
    ///                                 count is not 0 then count indicates the maximum number of
    ///                                 fragments to be scheduled. It will in that case not exceed
    ///                                 the number of hosts.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c nullptr, or \c count
    ///                                       is zero but the scheduling is not ONCE_PER_HOST).
    ///                                 - -3: Invalid job priority (negative value).
    ///                                 - -4: The transaction is no longer open.
    virtual mi::Sint32 execute_fragmented(
        Fragmented_job* job, size_t count) = 0;

    /// Execute a job splitting, it in a given number of fragments (asynchronous).
    ///
    /// This method will return immediately, typically before all fragments have been executed. The
    /// fragments may be executed in any number of threads and on any number of hosts.
    ///
    /// \param job                      The fragmented job to be executed. RCS:NEU
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero.
    /// \param listener                 Provides a callback to be called when the job is done.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c nullptr or \c count
    ///                                       is zero).
    ///                                 - -2: Invalid scheduling mode (asynchronous execution is
    ///                                       restricted to local jobs).
    ///                                 - -3: Invalid job priority (negative value).
    ///                                 - -4: The transaction is no longer open.
    virtual mi::Sint32 execute_fragmented_async(
        Fragmented_job* job, size_t count, IExecution_listener* listener) = 0;

    /// Cancels all fragmented jobs running in this transaction as soon as possible.
    virtual void cancel_fragmented_jobs() = 0;

    /// Indicates whether fragmented jobs were canceled via #cancel_fragmented_jobs().
    virtual bool get_fragmented_jobs_cancelled() = 0;

    //@}
    /// \name Database jobs
    //@{

    /// Invalidates the result of a DB job.
    ///
    /// This can be used when the results are no longer valid because some other tag's data has
    /// been changed which directly or indirectly influences the job's results. The visibility of
    /// the invalidation follows the usual rules.
    ///
    /// Similarly, results of DB jobs invalidated previously in the current transaction cannot be
    /// invalidated once again.
    ///
    /// \param tag    The job tag whose results should be invalidated. Results of DB jobs created/
    ///               previously invalidated in the current transaction are never invalidated
    ///               (again).
    virtual void invalidate_job_results( Tag tag) = 0;

    //@}
    /// \name Misc methods
    //@{

    /// Advises the database that a tag will be needed soon.
    ///
    /// \param tag   The tag to advise.
    virtual void advise( Tag tag) = 0;

    /// Constructs an empty element of the given type.
    ///
    /// This is used in case there are lookup failures, to avoid getting stuck or crashing the
    /// process.
    ///
    /// \param class_id   Construct an empty element of the given class ID.
    /// \return           The new element. RCS:TRO
    virtual Element_base* construct_empty_element( SERIAL::Class_id class_id) = 0;

    /// Returns the wrapped transaction (or \c this if there is no wrapper).
    ///
    /// There are a number of transactions wrappers which can be used to modify the behavior of a
    /// transaction. However, in some places one needs access to the real transaction.
    ///
    /// \return   RCS:NEU
    virtual Transaction* get_real_transaction() = 0;
};

/// Used by the Boost intrusive pointer to Transaction.
inline void intrusive_ptr_add_ref( Transaction* transaction)
{
    transaction->pin();
}

/// Used by the Boost intrusive pointer to Transaction.
inline void intrusive_ptr_release( Transaction* transaction)
{
    transaction->unpin();
}

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_TRANSACTION_H
