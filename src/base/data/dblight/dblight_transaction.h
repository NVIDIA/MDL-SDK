/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_TRANSACTION_H
#define BASE_DATA_DBLIGHT_DBLIGHT_TRANSACTION_H

#include <atomic>

#include <boost/core/noncopyable.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>

#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_tag.h>
#include <base/hal/thread/i_thread_condition.h>

#include "dblight_fragmented_job.h"
#include "dblight_util.h"

namespace MI {

namespace DBLIGHT {

class Database_impl;
class Info_impl;
class Scope_impl;
class Transaction_manager;

namespace bi = boost::intrusive;

/// A transaction journal entry comprises a tag, its version (from the sequence number), scope ID,
/// and the journal type.
class Transaction_journal_entry
{
public:
    /// Constructor.
    Transaction_journal_entry() = default;

    /// Default constructor.
    Transaction_journal_entry(
        DB::Tag tag, mi::Uint32 version, DB::Scope_id scope_id, DB::Journal_type journal_type)
      : m_tag( tag), m_version( version), m_scope_id( scope_id), m_journal_type( journal_type) { }

    // The "const" constraint is violated by the assignment operator used by std::sort.

    /*const*/ DB::Tag m_tag;
    /*const*/ mi::Uint32 m_version = 0;
    /*const*/ DB::Scope_id m_scope_id = 0;
    /*const*/ DB::Journal_type m_journal_type = DB::JOURNAL_NONE;
};

/// A transaction of the database.
///
/// Transactions are created with pin count 1. The pin count is decremented again in
/// #Transaction_manager::end_transaction().
class Transaction_impl : public DB::Transaction
{
public:
    /// The various states for transactions.
    ///
    ///                                     /-> COMMITTED
    /// Transition diagram: OPEN -> CLOSING
    ///                                     \-> ABORTED
    enum State {
        OPEN,        ///< Open transaction (after creation).
        CLOSING,     ///< During commit() or abort().
        COMMITTED,   ///< After commit().
        ABORTED      ///< After abort().
    };

    /// Constructor.
    ///
    /// \param database              Instance of the database this transaction belongs to.
    /// \param transaction_manager   Manager that created this transaction.
    /// \param scope                 Scope this transaction belongs to. RCS:NEU
    /// \param id                    ID of this transaction.
    Transaction_impl(
        Database_impl* database,
        Transaction_manager* transaction_manager,
        Scope_impl* scope,
        DB::Transaction_id id);

    /// Destructor.
    virtual ~Transaction_impl();

    // methods of DB::Transaction

    void pin() override { ++m_pin_count; }

    void unpin() override { if( --m_pin_count == 0) delete this; }

    DB::Transaction_id get_id() const override { return m_id; }

    DB::Scope* get_scope() override;

    /// Note that this method does \em not increment the sequence number.
    ///
    /// Use #allocate_sequence_number() to allocate a valid sequence number for updates and to
    /// increment the internal counter for the next allocation.
    mi::Uint32 get_next_sequence_number() const override { return m_next_sequence_number; }

    bool commit() override;

    void abort() override;

    bool is_open( bool closing_is_open) const override;

    bool block_commit_or_abort() override;

    bool unblock_commit_or_abort() override;

    DB::Info* access_element( DB::Tag tag) override;

    DB::Info* access_element( const char* name) override;

    DB::Info* edit_element( DB::Tag tag) override;

    DB::Info* edit_element( const char* name) override;

    void finish_edit( DB::Info* info, DB::Journal_type journal_type) override;

    DB::Tag reserve_tag() override;

    DB::Tag store(
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level) override;

    void store(
        DB::Tag tag,
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level) override;

    DB::Tag store_for_reference_counting(
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level) override;

    void store_for_reference_counting(
        DB::Tag tag,
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level) override;


    DB::Tag store(
        SCHED::Job_base* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level) override;

    void store(
        DB::Tag tag,
        SCHED::Job_base* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level) override;

    DB::Tag store_for_reference_counting(
        SCHED::Job_base* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level) override;

    void store_for_reference_counting(
        DB::Tag tag,
        SCHED::Job_base* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level) override;

    void localize(
        DB::Tag tag, DB::Privacy_level privacy_level, DB::Journal_type journal_type) override;

    bool remove( DB::Tag tag, bool remove_local_copy) override;

    const char* tag_to_name( DB::Tag tag) override;

    DB::Tag name_to_tag( const char* name, Name_to_tag_context context) override;

    DB::Tag name_to_tag_unsafe( const char* name) override;

    bool get_tag_is_job( DB::Tag tag) override;

    SERIAL::Class_id get_class_id( DB::Tag tag) override;

    DB::Privacy_level get_tag_privacy_level( DB::Tag tag) override;

    DB::Privacy_level get_tag_store_level( DB::Tag tag) override;

    mi::Uint32 get_tag_reference_count( DB::Tag tag) override;

    DB::Tag_version get_tag_version( DB::Tag tag) override;

    bool can_reference_tag( DB::Privacy_level referencing_level, DB::Tag referenced_tag) override;

    bool can_reference_tag( DB::Tag referencing_tag, DB::Tag referenced_tag) override;

    bool get_tag_is_removed( DB::Tag tag) override;

    std::unique_ptr<DB::Journal_query_result> get_journal(
         DB::Transaction_id last_transaction_id,
         mi::Uint32 last_transaction_change_version,
         DB::Journal_type journal_type,
         bool lookup_parents) override;

    mi::Sint32 execute_fragmented( DB::Fragmented_job* job, size_t count) override;

    mi::Sint32 execute_fragmented_async(
        DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener) override;

    void cancel_fragmented_jobs() override;

    bool get_fragmented_jobs_cancelled() override { return m_fragmented_jobs_cancelled; }

    void invalidate_job_results( DB::Tag tag) override;

    /// The asynchronous execution of the advised job blocks commit()/abort() until execution
    /// finished (no cancellation). But the transaction is then in CLOSING state, i.e., no longer
    /// OPEN.
    ///
    /// Must not be used from suspended worker threads (requirement of the thread pool for async
    /// execution).
    void advise( DB::Tag tag) override;

    DB::Element_base* construct_empty_element( SERIAL::Class_id class_id) override;

    Transaction* get_real_transaction() override { return this; }

    // internal methods

    /// Returns the current pin count.
    mi::Uint32 get_pin_count() const { return m_pin_count; }

    /// Sets the state.
    void set_state( State state) { m_state = state; }

    /// Returns the current state.
    State get_state() const { return m_state; }

    /// Sets the visibility_id.
    void set_visibility_id( DB::Transaction_id visibility_id) { m_visibility_id = visibility_id; }

    /// Returns the current visibility_id.
    DB::Transaction_id get_visibility_id() const { return m_visibility_id; }

    /// Allocates a valid sequence number for updates and increments the internal counter for the
    /// next allocation.
    ///
    /// Use #get_next_sequence_number() to query the number \em without incrementing it.
    mi::Uint32 allocate_sequence_number() { return m_next_sequence_number++; }

    /// The type of the transaction journal.
    using Transaction_journal = std::vector<Transaction_journal_entry>;

    /// Returns the complete transaction journal (for sorting and clearing during commit).
    Transaction_journal& get_journal() { return m_journal; }

    /// Returns the complete transaction journal (for dumping).
    const Transaction_journal& get_journal() const { return m_journal; }

    /// Indicates whether changes from this transaction are visible for transaction \p id.
    ///
    /// \note This method considers only the creation/commit sequence and states. It completely
    ///       ignores the corresponding scopes.
    bool is_visible_for( DB::Transaction_id id) const;

    /// Implements the four store() overloads for DB elements.
    void store_element_internal(
        DB::Tag tag,
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level,
        bool store_for_rc);

    /// Implements the four store() overloads for DB jobs.
    void store_job_internal(
        DB::Tag tag,
        SCHED::Job_base* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level,
        bool store_for_rc);

    /// Implements the four store() overloads for DB jobs.
    void store_job_internal_locked(
        DB::Tag tag,
        SCHED::Job_base* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level,
        bool store_for_rc,
        bool temporary);

    /// Same as #access_element() except that not statistics are updated. Used by #get_class_id().
    DB::Info* access_element_no_stats( DB::Tag tag);

    /// Implements #access_element() (by tag) and #access_element_no_stats().
    ///
    /// \param tag                    The tag to look up.
    /// \param is_exclusive           Indicates whether the shared or exclusive lock was obtained.
    /// \return                       A pair of an info and a flag. If \p is_exclusive is \c false,
    ///                               but the method needs the exclusive lock to succeed, then it
    ///                               indicates that by returning \p {nullptr,true}. In all other
    ///                               cases the first component is the info for that tag (or
    ///                               \c nullptr) and the second component is not relevant. RCS:ICE
    std::pair<DB::Info*,bool> access_element_locked( DB::Tag tag, bool is_exclusive);

    /// Implements #access_element() (by name).
    ///
    /// \param name                   The name to look up.
    /// \param is_exclusive           Indicates whether the shared or exclusive lock was obtained.
    /// \return                       A pair of an info and a flag. If \p is_exclusive is \c false,
    ///                               but the method needs the exclusive lock to succeed, then it
    ///                               indicates that by returning \p {nullptr,true}. In all other
    ///                               cases the first component is the info for that name (or
    ///                               \c nullptr) and the second component is not relevant. RCS:ICE
    std::pair<DB::Info*,bool> access_element_locked( const char* name, bool is_exclusive);

    /// Implements both overloads of access_element_locked().
    ///
    /// \param info                   The info that is to be accessed. RCS:ICR
    /// \param is_exclusive           Indicates whether the shared or exclusive lock was obtained.
    /// \return                       A pair of an info and a flag. If \p is_exclusive is \c false,
    ///                               but the method needs the exclusive lock to succeed, then it
    ///                               indicates that by returning \p {nullptr,true}. In all other
    ///                               cases the first component is the info that was passed (or
    ///                               \c nullptr) and the second component is not relevant. RCS:ICE
    std::pair<DB::Info*,bool> access_element_shared( Info_impl* info, bool is_exclusive);

    /// Implements both overloads of edit_element().
    ///
    /// \param info                   The info that is to be edited. RCS:ICR
    /// \return                       A copy of the info to be edited (or \c nullptr in case of
    ///                               failures, i.e., jobs, which cannot be edited). RCS:ICE
    DB::Info* edit_element_shared( Info_impl* info);

    /// Indicates whether the job result is valid for this transaction.
    bool is_job_result_valid( DB::Info* info);

    /// Splits the job.
    void split_job( Info_impl* info);

    /// Same as can_reference_tag(), but with m_database->get_lock() locked.
    bool can_reference_tag_locked(
        DB::Privacy_level referencing_level, DB::Tag referenced_tag);

    /// Checks that references do not point to tags that can not be found in the given privacy level
    /// and emits an error message otherwise.
    ///
    /// Note that such a reference could be considered legal if the reference is never used when
    /// that element is retrieved from a transaction in its store level, only from transactions in
    /// the store level of the referenced element (or its child scopes). This is very error-prone
    /// and we do not want to support such corner cases.
    ///
    /// \param referencing_level    The element which references \p references is to be stored in a
    ////                            scope with this privacy level.
    /// \param references           The tags referenced by that element.
    /// \param tag                  Tag of the referencing element (only used for error messages).
    /// \param name                 Name of the referencing element (only used for error messages).
    ///                             Can be \c nullptr.
    /// \param store                Flag that distinguishes between store and edit operations (only
    ///                             used for error messages).
    void check_privacy_levels(
        DB::Privacy_level referencing_level,
        const DB::Tag_set& references,
        DB::Tag tag,
        const char* name,
        bool store);

    /// Checks that references do not create a reference cycle.
    ///
    /// Note that this check is performed from the context of this transaction. Running it when
    /// storing elements or finishing edits will find many simple mistakes, but not all mistakes
    /// that can be made in more advanced scenarios. For example:
    /// - Parallel transactions: Transaction 1 creates elements A and B and is committed. Parallel
    ///   transactions 2 and 3 edit A and B, respectively, to add a reference to the respective
    ///   other element. There is no reference cycle visible in both transactions, only in
    ///   transactions started after transactions 2 and 3 have been committed.
    /// - Multiple scopes: Create elements A and B in the global scope. Localize A to a child scope
    ///   and create a reference to B. Edit B from a transaction in the global scope and create a
    ///   reference to A. There is no reference cycle visible in this or later transactions from the
    ///   global scope, only in transactions from the child scope.
    ///
    /// \param references           The tags referenced by the root element.
    /// \param root                 Tag of the root element to be stored/edited.
    /// \param name                 Name of the root element (only used for error messages). Can be
    ///                             \c nullptr.
    /// \param store                Flag that distinguishes between store and edit operations (only
    ///                             used for error messages).
    void check_reference_cycles(
        const DB::Tag_set& references,
        DB::Tag root,
        const char* name,
        bool store);

    /// Helper method to check that references do not create a reference cycle.
    ///
    /// Used by #check_cycles().
    ///
    /// \param tag                  The tag to process.
    /// \param root                 Tag of the root element to be stored/edited (only used for
    ///                             error messages).
    /// \param name                 Name of the root element (only used for error messages). Can be
    ///                             \c nullptr.
    /// \param store                Flag that distinguishes between store and edit operations (only
    ///                             used for error messages).
    /// \param processing           Tags currently on the stack for DFS traversal.
    /// \param done                 Tags of subgraph that has already been processed.
    void check_reference_cycles_internal(
        DB::Tag tag,
        DB::Tag root,
        const char* name,
        bool store,
        DB::Tag_set& processing,
        DB::Tag_set& done);

    /// Returns the number of infos pinned by this transaction.
    size_t get_pinned_infos_size() const;

    /// Unpins all infos pinned by this transaction.
    void unpin_pinned_infos();

    /// Implements #block_commit_or_abort().
    bool block_commit_or_abort_locked();

    /// Implements #unblock_commit_or_abort().
    bool unblock_commit_or_abort_locked();

    /// Waits for #m_block_counter to reach zero.
    ///
    /// Returns immediately if #m_block_counter is zero. Otherwise releases the database lock,
    /// waits for #m_block_condition, and re-acquires the database lock.
    ///
    /// \param commit        \c true to commit, \c false to abort (for log messages only).
    void wait_for_unblocked_locked( bool commit);

    /// Waits for #m_fragmented_jobs_counter to reach zero.
    ///
    /// Returns immediately if m_fragmented_jobs_counter is zero. Otherwise releases the database
    /// lock, waits for #m_fragmented_jobs_condition, and re-acquires the database lock.
    ///
    /// \param commit        \c true to commit, \c false to abort (for log messages only).
    void wait_for_fragmented_jobs_locked( bool commit);

    /// Cancels all fragmented jobs currently running.
    ///
    /// Notifies the running fragmented jobs via DB::Fragmented_job::cancel(). Does \em not block
    /// until the fragmented jobs have actually finished their execution.
    void cancel_fragmented_jobs_locked();

    /// Notifies this transaction that a particular fragmented job has finished its execution.
    void fragmented_job_finished( DBLIGHT::Fragmented_job* job);

private:
    /// Instance of the database this transaction belongs to.
    Database_impl* const m_database;
    /// The manager that created this transaction.
    Transaction_manager* const m_transaction_manager;
    /// Scope this transaction belongs to.
    Scope_impl* const m_scope;

    /// ID of this transaction.
    const DB::Transaction_id m_id;
    /// Reference count of the transaction.
    std::atomic_uint32_t m_pin_count = 1;
    /// State of the transaction.
    State m_state = OPEN;
    /// Visibility of changes from this transaction (only valid in COMMITTED state).
    DB::Transaction_id m_visibility_id;
    /// Sequence number for the next update within this transaction.
    std::atomic_uint32_t m_next_sequence_number = 0;
    /// Complete journal of updates for this transaction.
    std::vector<Transaction_journal_entry> m_journal;

    /// The lock for #m_pinned_infos.
    mutable THREAD::Lock m_pinned_infos_lock;
    /// Tracks infos from #name_to_tag. Needs #m_pinned_infos_lock.
    std::vector<DB::Info*> m_pinned_infos;

    /// The counter to track outstanding commit/abort blocking requests.
    std::atomic_uint32_t m_block_counter = 0;
    /// The condition to wait for no outstanding commit/abort blocking requests during commit/abort.
    THREAD::Condition m_block_condition;

    using Fragmented_jobs_hook = bi::member_hook<
        Fragmented_job, bi::list_member_hook<>, &Fragmented_job::m_fragmented_jobs_hook>;

    using Fragmented_jobs_list = bi::list<Fragmented_job, Fragmented_jobs_hook>;

    /// All running fragmented jobs.
    Fragmented_jobs_list m_fragmented_jobs;
    /// The counter to track running fragmented jobs (equal to size of #m_fragmented_jobs).
    std::atomic_uint32_t m_fragmented_jobs_counter = 0;
    /// The condition to wait for no running fragmented jobs during commit/abort.
    THREAD::Condition m_fragmented_jobs_condition;
    /// Indicates whether fragmented jobs have been notified to cancel.
    bool m_fragmented_jobs_cancelled = false;

public:
    /// Hook for Transaction_manager::m_all_transactions.
    bi::set_member_hook<> m_all_transactions_hook;
    /// Hook for Transaction_manager::m_open_transactions.
    bi::set_member_hook<> m_open_transactions_hook;
};

/// Comparison operator for Transaction_impl.
///
/// Sorts by comparison of the transaction ID.
inline bool operator<( const Transaction_impl& lhs, const Transaction_impl& rhs)
{ return lhs.get_id() < rhs.get_id(); }

/// Output operator for Transaction_impl::State.
std::ostream& operator<<( std::ostream& s, const Transaction_impl::State& state);

/// Manager for transactions.
class Transaction_manager : private boost::noncopyable
{
public:
    /// Constructor.
    ///
    /// \param database   Instance of the database this manager belongs to.
    Transaction_manager( Database_impl* database) : m_database( database) { }

    /// Destructor.
    ///
    /// Checks
    /// - that there are no open transactions anymore and
    /// - that there no alive transactions at all anymore.
    ~Transaction_manager();

    /// Starts a new transaction in the given scope.
    ///
    /// \param scope   Scope this transaction belongs to. RCS:NEU
    /// \return        The new transaction. RCS:NEU
    Transaction_impl* start_transaction( Scope_impl* scope);

    /// Ends a transaction.
    ///
    /// Sets the state to CLOSING, removes the transaction from the set of open transactions,
    /// sets the visibility ID, sets the state to COMMITTED or ABORTED depending on \p commit,
    /// decrements the pin count, and invokes the garbage collection.
    ///
    /// \param transaction   The transaction to end. RCS:NEU
    /// \param commit        \c true to commit, \c false to abort.
    /// \return              \c true in case of success, \c false otherwise (only possible if the
    ///                      initial transaction state was not OPEN).
    bool end_transaction( Transaction_impl* transaction, bool commit);

    /// Removes a transaction from the set of all transactions.
    ///
    /// Used by the destructor of Transaction_impl to remove itself from the set.
    ///
    /// \param transaction   The transaction to remove. RCS:NEU
    void remove_from_all_transactions( Transaction_impl* transaction);

    /// Returns the lowest ID of all open transactions (or the ID of the next transaction if there
    /// are no open transactions).
    DB::Transaction_id get_lowest_open_transaction_id() const;

    /// Dumps the state of the transaction manager to the stream.
    void dump( std::ostream& s, bool verbose, bool mask_pointer_values);

private:
    /// Instance of the database this manager belongs to.
    Database_impl* const m_database;

    /// Lock for m_all_transactions.
    THREAD::Lock m_all_transactions_lock;

    using All_transactions_hook = bi::member_hook<
        Transaction_impl, bi::set_member_hook<>, &Transaction_impl::m_all_transactions_hook>;

    using Open_transactions_hook = bi::member_hook<
        Transaction_impl, bi::set_member_hook<>, &Transaction_impl::m_open_transactions_hook>;

    /// Set of all still existing transactions.
    ///
    /// Needs m_all_transactions_lock. Used only by the dump() method.
    bi::set<Transaction_impl, All_transactions_hook> m_all_transactions;

    /// Set of all open transactions.
    bi::set<Transaction_impl, Open_transactions_hook> m_open_transactions;

    /// ID of the next transaction to be created.
    DB::Transaction_id m_next_transaction_id;
};

// Used by the Boost intrusive pointer to Transaction_impl.
inline void intrusive_ptr_add_ref( Transaction_impl* transaction)
{
    transaction->pin();
}

/// Used by the Boost intrusive pointer to Transaction_impl.
inline void intrusive_ptr_release( Transaction_impl* transaction)
{
    transaction->unpin();
}

/// Intrusive pointer for Transaction_impl.
using Transaction_impl_ptr = boost::intrusive_ptr<Transaction_impl>;

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_TRANSACTION_H

