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

/** \file i_db_transaction.h
 ** \brief This file contains the pure virtual base class for the transaction class.
 **/

#ifndef BASE_DATA_DB_I_DB_TRANSACTION_H
#define BASE_DATA_DB_I_DB_TRANSACTION_H

#include <vector>

#include "i_db_tag.h"
#include "i_db_journal_type.h"
#include "i_db_scope.h"

#include <base/data/serial/i_serial_classid.h>
#include <base/system/main/types.h>
#include <base/system/stlext/i_stlext_concepts.h>

namespace MI
{

namespace SCHED { class Job; }
namespace NET { typedef Uint32 Host_id; }
namespace DB
{

// The maximum redundancy level we support. This is reserved for storing the owners in an Info
static const Uint MAX_REDUNDANCY_LEVEL = 4;

class Element_base;
class Fragmented_job;
class Info;
class Scope;
class IExecution_listener;

/// A transaction lives within a scope and provides a consistent view on the database for the
/// lifetime of the transaction.
///
/// \section Storage level and privacy level
///
/// When storing database elements one can specify two parameters, the storage level and the privacy
/// level. These two parameters control in which scope the element is stored, from which scopes it
/// is visible for later accesses, and what happens if an element is edited. Note that the storage
/// level is always less than or equal to the privacy level (at least conceptually, actual arguments
/// might violate that rule and are clamped accordingly).
///
/// The storage level indicates in which scope of the scope stack an element is stored. The scope
/// is selected as follows:
/// - If the storage level is larger than the privacy level, then it is set to the privacy level.
/// - Pick the most local scope from the scope stack of the current transaction whose level is less
///   than or equal to the requested storage level (scope levels do not need to be consecutive).
///
/// Such an element is visible for transactions associated with the selected scope or its child
/// scopes. It is not visible for transactions associated with any parent or other scopes.
///
/// The privacy level indicates in which scope of the scope stack the copy created for editing an
/// element is stored. The scope is selected as follows:
/// - Pick the most local scope from the scope stack of the current transaction whose level is less
///   than or equal to the requested privacy level (scope levels do not need to be consecutive).
/// In other words, when editing a database element, it is automatically localized to its privacy
/// level.
///
/// \note The API does not distinguish between storage level and privacy level. It only exposes the
///       privacy level and uses the default storage level (currently 255, which is internally
///       effectively clamped to the privacy level). The API also rejects invalid privacy levels
///       instead of silently adjusting them, but note the special meaning of the constant
///       mi::neuraylib::ITransaction::LOCAL_SCOPE = 255.
class Transaction : private STLEXT::Non_copyable
{
  public:
    /// pin the transaction incrementing its reference count
    virtual void pin() = 0;

    /// Unpin the transaction decrementing its reference count. Delete the transaction once the
    /// reference count reaches zero.
    virtual void unpin() = 0;

    /// Block the transaction from committing or aborting. This will increment a counter. As long
    /// as the counter is >0 a commit or abort will wait. 
    /// \return True, if the transaction could be blocked, or false, otherwise. If it returns
    ///         false that means that the transaction was already committed or aborted
    virtual bool block_commit() = 0;

    /// Unblock the transaction which was previously blocked. If the counter reaches 0, a waiting
    /// commit or abort will proceed.
    /// \return True, if the transaction could be unblocked, or false, otherwise.
    virtual bool unblock_commit() = 0;

    /// Commit the transaction. All changes done from within this transaction will now become
    /// visible for all transactions created later. Although the application may no longer use the
    /// transaction after committing it, it might continue living in the database for an unspecified
    /// amount of time. The return value tells, if the commitment has succeed (true) or not. The
    /// latter can be the case, if a host contributing to the transaction failed, before the
    /// transaction was committed.
    virtual bool commit() = 0;

    /// Abort the transaction. All changes will be thrown away. No other transaction will ever see
    /// the changes done from within this transaction. Although the application may no longer use
    /// the transaction after aborting it, it might continue living in the database for an
    /// unspecified amount of time.
    virtual void abort() = 0;

    /// Is the transaction still open? This can be used by jobs to query, if they shall abort their
    /// operation.
    virtual bool is_open() = 0;

    /// Reserve and return a free tag id. This tag id must be used to store a database element or
    /// job in a later call to store. It can be used e.g. with groups of database elements where
    /// circular references exist.
    virtual Tag reserve_tag() = 0;

    /// Insert a new element into the database. the return value is the tag which will now identify
    /// the element.
    ///
    /// \param element                  The element to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param store_level              Level of the scope the tag is stored in
    /// \return                         The assigned tag
    virtual Tag store(
        Element_base* element,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Insert a new element into the database reusing a tag.
    ///
    /// \param tag                      The tag to recreate
    /// \param element                  The element to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param journal_type             Type for journal entries
    /// \param store_level              Level of the scope the tag is stored in
    virtual void store(
        Tag tag,
        Element_base* element,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_ALL,
        Privacy_level store_level = 255) = 0;

    /// Insert a new job into the database. The return value is the tag which will now identify the
    /// job.
    ///
    /// \param job                      The job to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param store_level              Level of the scope the tag is stored in
    /// \return                         The assigned tag
    virtual Tag store(
        SCHED::Job* job,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Insert a new job into the database reusing a tag.
    ///
    /// \param tag                      The tag to recreate
    /// \param job                      The job to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param journal_type             Type for journal entries
    /// \param store_level              Level of the scope the tag is stored in
    virtual void store(
        Tag tag,
        SCHED::Job* job,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_NONE,
        Privacy_level store_level = 255) = 0;

    /// Insert a new element into the database. The return value is the tag which will now identify
    /// the element. The tag will be removed immediately, automatically. So to prevent it from being
    /// deleted completely, one or more other database element need to list it in the list of
    /// references returned from get_references.
    ///
    /// \param element                  The element to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param store_level              Level of the scope the tag is stored in
    /// \return                         The assigned tag
    virtual Tag store_for_reference_counting(
        Element_base* element,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Insert a new element into the database reusing a tag.
    /// The tag will be removed immediately, automatically. So to prevent it from being deleted
    /// completely, one or more other database element need to list it in the list of references
    /// returned from get_references.
    ///
    /// \param tag                      The tag to recreate
    /// \param element                  The element to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param journal_type             Type for journal entries
    /// \param store_level              Level of the scope the tag is stored in
    virtual void store_for_reference_counting(
        Tag tag,
        Element_base* element,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_ALL,
        Privacy_level store_level = 255) = 0;

    /// Insert a new job into the database. The return value is the tag which will now identify the
    /// job.
    /// The tag will be removed immediately, automatically. So to prevent it from being deleted
    /// completely, one or more other database element need to list it in the list of references
    /// returned from get_references.
    ///
    /// \param job                      The job to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param store_level              Level of the scope the tag is stored in
    /// \return                         The assigned tag
    virtual Tag store_for_reference_counting(
        SCHED::Job* job,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Privacy_level store_level = 255) = 0;

    /// Insert a new job into the database reusing a tag.
    /// The tag will be removed immediately, automatically. So to prevent it from being deleted
    /// completely, one or more other database element need to list it in the list of references
    /// returned from get_references.
    ///
    /// \param tag                      The tag to recreate
    /// \param job                      The job to insert
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param journal_type             Type for journal entries
    /// \param store_level              Level of the scope the tag is stored in
    virtual void store_for_reference_counting(
        Tag tag,
        SCHED::Job* job,
        const char* name = NULL,
        Privacy_level privacy_level = 0,
        Journal_type journal_type = JOURNAL_NONE,
        Privacy_level store_level = 255) = 0;

    /// Invalidate the results of a certain job for the current and all later transactions. this
    /// should be used when an application decides that the results are no longer valid because some
    /// other tag's data has been changed which directly or indirectly influences the job's results.
    ///
    /// \param tag                      The tag to invalidate
    virtual void invalidate_job_results(
        Tag tag) = 0;

    /// Remove a tag from the database. Return true, if succeeded, otherwise false.
    ///
    /// \param tag                      The tag to remove
    /// \param remove_local_copy        If this is true, only the local copy in the scope of this
    ///                                 transaction is checked for the existence of the tag. Only
    ///                                 this copy is removed. This can be used to undo effects of
    ///                                 a localize call.
    virtual bool remove(
        Tag tag,
        bool remove_local_copy = false) = 0;

    /// Wrapper function for typed tags. Simply forwards to the untyped remove() method.
    ///
    /// \param tag                      The tag to remove
    /// \param remove_local_copy        If this is true, only the local copy in the scope of this
    ///                                 transaction is checked for the existence of the tag. Only
    ///                                 this copy is removed. This can be used to undo effects of
    ///                                 a localize call.
    template <class T> bool remove(
        const Typed_tag<T> &    tag,
        bool                    remove_local_copy = false)
    {
        return remove(tag.get_untyped(), remove_local_copy);
    }

    /// Advise the database that a certain tag will be needed soon.
    ///
    /// \param tag                      The tag to advise
    virtual void advise(
        Tag tag) = 0;

    /// Localize a tag to the given scope level
    ///
    /// \param tag                      Localize this tag
    /// \param privacy_level            Localize to this level
    /// \param journal_type             Type for journal entries
    virtual void localize(
        Tag tag,
        Privacy_level privacy_level,
        Journal_type journal_type = JOURNAL_NONE) = 0;

    /// Lookup the name of a tag within the context of this transaction.
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The name or NULL if the tag has no associated name.
    virtual const char* tag_to_name(
        Tag tag) = 0;

    /// Lookup name of typed tag, forwards to raw tag name lookup.
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The name or NULL if the tag has no associated name.
    template <typename T> const char* tag_to_name(
        const Typed_tag<T>& tag)
    {
        return tag_to_name(tag.get_untyped());
    }

    /// Lookup the tag for a name within the context of this transaction.
    ///
    /// \param name                     The name to lookup.
    /// \return                         The found tag or the 0 tag if the name was not found
    virtual Tag name_to_tag(
        const char* name) = 0;

    /// Get the class id of a tag. If the returned class id is class_id_unknown, then it means that
    /// the value could not be determined and must be ignored! This will happen when the element is
    /// not in the cache or if it is a job. In such cases the database will not fetch the element or
    /// execute the job to determine the class id. When the class id is really needed, then a caller
    /// must do an Access on the element.
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The class id.
    virtual SERIAL::Class_id get_class_id(
        Tag tag) = 0;

    /// Get the unique id of a certain tag version. The result of a database lookup on a certain tag
    /// depends on the asking transaction and may return different versions for different
    /// transactions. For caching data derived from the tag, such a unique id uniquely identifies
    /// the actual version of the tag obtained from a certain transaction. This means, that it may
    /// be used to identify this version. The database guarantees, that any legal change to the data
    /// (done through an edit) will also change this id.
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The found tag version.
    virtual Tag_version get_tag_version(
        Tag tag) = 0;

    /// Get the sequence number for the next update done on this host. This can be used as a
    /// timestamp in the transaction.
    virtual Uint32 get_update_sequence_number() = 0;

    /// Get reference count for a certain tag
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The reference count.
    virtual Uint32 get_tag_reference_count(
        Tag tag) = 0;

    /// Checks whether another tag can be referenced from a given scope level.
    ///
    /// This is the case if the referenced tag has at least one version in a scope with level
    /// between 0 and the referencing level. Otherwise, one can trigger invalid tag accesses because
    /// the referenced tag can not be seen from the scope at the referencing level (and maybe some
    /// of its children).
    ///
    /// \param referencing_level        The tag which is meant to reference the other tag will be
    ///                                 stored in this scope.
    /// \param referenced_tag           The tag which is meant to be referenced from the referencing
    ///                                 level.
    /// \return                         \c true if the reference is valid.
    virtual bool can_reference_tag(   
        Privacy_level referencing_level,
        Tag referenced_tag) = 0;

    /// Checks whether a given tag can reference another tag.
    ///
    /// This is the case if the scope level of the referencing tag is not smaller than the scope
    /// level of the referenced tag. Otherwise, one can trigger invalid tag accesses because the
    /// referenced tag can not be seen from all scopes that provide the referencing tag.
    ///
    /// \param referencing_tag          The tag which is meant to reference the other tag.
    /// \param referenced_tag           The tag which is meant to be referenced by the other tag.
    /// \return                         \c true if the reference is valid.
    virtual bool can_reference_tag(
        Tag referencing_tag,
        Tag referenced_tag) = 0;

    /// Check, if Transaction::remove has already been called on the tag
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         True, if the tag was removed, false, otherwise.
    virtual bool get_tag_is_removed(
        Tag tag) = 0;

    /// Check, if a certain tag points to a job
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         True, if the tag is a job, false, otherwise.
    virtual bool get_tag_is_job(
        Tag tag) = 0;

    /// Return the privacy level of a tag
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The privacy level
    virtual Privacy_level get_tag_privacy_level(
        Tag tag) = 0;

    /// Return the storage level of a tag. The storage level is the privacy level of the scope
    /// the tag belongs to
    ///
    /// \param tag                      The tag to lookup.
    /// \return                         The storage level
    virtual Privacy_level get_tag_storage_level(
        Tag tag) = 0;

    /// Get the id of this transaction
    virtual Transaction_id get_id() const = 0;

    /// Get the list of changes relevant for this transaction since the given transaction id. This
    /// is meant to be used e.g. for preprocessing. The transaction id is the one in which the last
    /// preprocessing was done. The list of tags returned will contain all tags touched since then,
    /// if the change to the tag is actually visible to this transaction. The bitmask content is
    /// application defined. Each change can be associated with some value in the bitmask. Only
    /// those journal which have a value which is in the bitmask are returned. The functions journal
    /// entries set the journal type by default to 0xffffffff. This will result in unknown changes
    /// appearing in every journal.
    ///
    /// \param last_transaction_id      The id of the last transaction when the journal was checked.
    /// \param last_transaction_change_version The version counter the transaction had at the last
    ///                                 time it was checked.
    /// \param journal_type             A filter for the journal type.
    /// \param lookup_parents           Shall we lookup parents (aka parent scopes) too?
    /// \return                         A vector of tag/journal type pairs which needs to be
    ///                                 released by the caller.
    ///                                 NOTE: The returned value can be NULL. This means that the
    ///                                 journal needed to be pruned because it exceeded the storage
    ///                                 capacity of the database and would have been incomplete.
    ///                                 In this case the caller has to assume that everything may
    ///                                 have changed.
    virtual std::vector<std::pair<Tag, Journal_type> >* get_journal(
        Transaction_id last_transaction_id,
        Uint32 last_transaction_change_version,
        Journal_type journal_type,
        bool lookup_parents) = 0;

    /// Execute a job splitting it in a given number of fragments. This will not return before all
    /// fragments have been executed. The fragments may be executed in any number of threads.
    /// NOTE: Currently this is restricted to local host only jobs! This restriction might be
    ///       lifted in the future.
    ///
    /// \param job                      The fragmented job to be executed.
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero unless the scheduling mode
    ///                                 of the job is ONCE_PER_HOST.
    ///                                 NOTE: If the job has a scheduling mode which defines the
    ///                                 number of fragments implicitly (e.g. ONCE_PER_HOST) and
    ///                                 count is not 0 then count indicates the maximum number of
    ///                                 fragments to be scheduled. It will in that case not exceed
    ///                                 the number of hosts.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c NULL, or \c count is
    ///                                       zero but the scheduling is not ONCE_PER_HOST).
    ///                                 - -2: Invalid scheduling mode (transaction-less or
    ///                                       asynchronous execution is retricted to local jobs).
    ///                                 - -3: Invalid job priority (negative value).
    virtual Sint32 execute_fragmented(
        Fragmented_job* job,
        size_t count) = 0;

    /// Execute a job splitting it in a given number of fragments. This will return immediately,
    /// typically before all fragments have been executed. The fragments may be executed in any
    /// number of threads.
    /// NOTE: Currently this is restricted to local host only jobs! This restriction might be
    ///       lifted in the future.
    ///
    /// \param job                      The fragmented job to be executed.
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero.
    /// \param listener                 Provides a callback to be called when the job is done.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c NULL or \c count is
    ///                                       zero).
    ///                                 - -2: Invalid scheduling mode (transaction-less or
    ///                                       asynchronous execution is retricted to local jobs).
    ///                                 - -3: Invalid job priority (negative value).
    virtual Sint32 execute_fragmented_async(
        Fragmented_job* job,
        size_t count,
        IExecution_listener* listener) = 0;

    /// Cancel all fragmented jobs running in this transaction as soon as possible
    ///
    /// \note Only the first call will cancel the currently running (or submitted?) fragmented jobs.
    ///       Subsequent calls will be ignored (this is rather a bug in the implementation than a
    ///       feature). Currently this is not really a problem since this method is not called
    ///       explicitly (and therefore never multiple times per transaction), but only implicitly
    ///       one from commit() or abort().
    virtual void cancel_fragmented_jobs() = 0;

    /// Check if for this transaction fragmented jobs were canceled
    virtual bool get_fragmented_jobs_cancelled() = 0;

    /// Allocate the storage for a new element. The element itself is not yet stored in the
    /// database. If it is accessed, the assembly manager is asked to supply the element, instead.
    /// NOTE: This is only meant to be used in special cases, such as loading binary serialized
    /// files. Do not use in normal cases!
    ///
    /// \param name                     Optional name for tag
    /// \param privacy_level            Privacy level of element
    /// \param references               The referenced tags (may be NULL)
    virtual Tag store_deferred(
        const char* name,
        Privacy_level privacy_level,
        Tag_set* references)
    { return Tag(); }

    /// Return the parent scope of this transaction
    virtual Scope* get_scope() = 0;

  //
  // The functions below may only be used by SCHED!!!!
  //

    /// This is to be used by the sched module only. it will wait until the given job is locally
    /// available and return it. this may not be called from within the system thread because it is
    /// a blocking call.
    ///
    /// \param                          The tag of the job to be looked up.
    /// \return                         The job if it was found.
    virtual Info* get_job(
        Tag tag) = 0;

    /// This is to be used by the sched module only. it will store the given element into the
    /// database. It should be used in the remote case only, because it requires a tag lookup which
    /// is expensive.
    ///
    /// \param tag                      The tag the element belongs to.
    /// \param element                  The new element.
    virtual void store_job_result(
        Tag tag,
        Element_base* element) = 0;
    /// This is to be used by the sched module only. It will store the given element into the
    /// database. It should be used in the remote case only, because it requires a tag lookup which
    /// is expensive.
    ///
    /// \param tag                      The tag the element belongs to.
    /// \param host_id                  The destination host.
    virtual void send_element_to_host(
        Tag tag,
        NET::Host_id host_id) = 0;
    /// Get the list of updates received by the local node
    ///
    /// \return                         The list of updates.
    virtual class Update_list* get_received_updates() = 0;

    /// Wait until the transaction is ready to be worked on. this means we must have seen the
    /// creation of the transaction and we must have seen all updates requested in the given update
    /// list. the list may be NULL, which means, that no further updates are needed. this may not be
    /// called from within the system thread, because it will block
    ///
    /// \param needed_updates           The updates needed for this transaction to become ready.
    virtual void wait(
        class Update_list* needed_updates) = 0;

    /// Used for the intrusive pointer to Transaction
    ///
    /// \param transaction              The transaction to pin
    friend inline void increment_ref_count(
        Transaction* transaction)
    {
        transaction->pin();
    }

    /// Used for the intrusive pointer to Transaction
    ///
    /// \param transaction              The transaction to unpin
    friend inline void decrement_ref_count_and_release(
        Transaction* transaction)
    {
        transaction->unpin();
    }

    /// The destructor
    virtual ~Transaction() { }

    /// Do everything needed when editing an element.
    ///
    /// \param tag                      The tag which is being edited.
    /// \return                         The info object for the tag
    virtual Info* edit_element(
        Tag tag) = 0;

    /// Finish an edit and do what is necessary to commit and communicate the changes done during
    /// the edit. Note that this method (potentially asynchronously) unpins \p info.
    ///
    /// \param info                     The info for the edited tag.
    /// \param journal_type             Final resulting journal type for all changes on the object.
    virtual void finish_edit(
        Info* info,
        Journal_type journal_type) = 0;

    /// Make sure that we have the element for the given tag, pin the info for the element and
    /// return the info.
    /// \param tag                      The tag to lookup.
    /// \param do_wait                  If true wait for the element, if not return with NULL if the
    ///                                 element is not available locally.
    virtual Info* get_element(
        Tag tag,
        bool do_wait = true) = 0;

    /// Construct an empty element of the given type. This is used in case there are lookup
    /// failures, to avoid getting stuck or crashing the process.
    ///
    /// \param class_id                 Construct an empty element of the given class id
    /// \return                         The new element.
    virtual Element_base* construct_empty_element(
        SERIAL::Class_id class_id) = 0;

    /// There are a number of wrapper transactions which are given to job executors etc. In places,
    /// one might want to know the real transaction behind this. This is protected, because it is
    /// not meant to be used by everyone. Currently it is needed by the Access_cache class.
    virtual Transaction* get_real_transaction() = 0;

    friend class Access_base;
    friend class Caching_transaction;
    friend class Transaction_job_context;
    friend class Fragment_scheduler;
    template <class A> friend class Access_cache;
};


} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_TRANSACTION_H

