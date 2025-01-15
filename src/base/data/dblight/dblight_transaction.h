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
#include <boost/intrusive/set.hpp>

#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_tag.h>

#include "dblight_util.h"

namespace MI {

namespace DBLIGHT {

class Database_impl;
class Scope_impl;
class Transaction_manager;

namespace bi = boost::intrusive;

/// A transaction of the database.
///
/// Transactions are created with pin count 1. The pin count is decremented again in
/// #Transaction_manager::end_transaction().
///
/// "NI" means DBLIGHT does not implement/support that method of the interface.
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

    /*NI*/ bool block_commit_or_abort() override;

    /*NI*/ bool unblock_commit_or_abort() override;

    DB::Info* access_element( DB::Tag tag) override;

    DB::Info* edit_element( DB::Tag tag) override;

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

    /*NI*/ DB::Tag store(
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level) override;

    /*NI*/ void store(
        DB::Tag tag,
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level) override;

    /*NI*/ DB::Tag store_for_reference_counting(
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level) override;

    /*NI*/ void store_for_reference_counting(
        DB::Tag tag,
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level) override;

    void localize(
        DB::Tag tag, DB::Privacy_level privacy_level, DB::Journal_type journal_type) override;

    bool remove( DB::Tag tag, bool remove_local_copy) override;

    const char* tag_to_name( DB::Tag tag) override;

    DB::Tag name_to_tag( const char* name) override;

    bool get_tag_is_job( DB::Tag tag) override { return false; }

    SERIAL::Class_id get_class_id( DB::Tag tag) override;

    DB::Privacy_level get_tag_privacy_level( DB::Tag tag) override;

    DB::Privacy_level get_tag_store_level( DB::Tag tag) override;

    mi::Uint32 get_tag_reference_count( DB::Tag tag) override;

    DB::Tag_version get_tag_version( DB::Tag tag) override;

    bool can_reference_tag( DB::Privacy_level referencing_level, DB::Tag referenced_tag) override;

    bool can_reference_tag( DB::Tag referencing_tag, DB::Tag referenced_tag) override;

    bool get_tag_is_removed( DB::Tag tag) override;

    /*NI*/ std::unique_ptr<DB::Journal_query_result> get_journal(
         DB::Transaction_id last_transaction_id,
         mi::Uint32 last_transaction_change_version,
         DB::Journal_type journal_type,
         bool lookup_parents) override;

    /// Only the scheduling mode LOCAL is supported.
    mi::Sint32 execute_fragmented( DB::Fragmented_job* job, size_t count) override;

    /// Only the scheduling mode LOCAL is supported.
    mi::Sint32 execute_fragmented_async(
        DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener) override;

    /*NI*/ void cancel_fragmented_jobs() override;

    bool get_fragmented_jobs_cancelled() override { return false; }

    /*NI*/ void invalidate_job_results( DB::Tag tag) override;

    /*NI*/ void advise( DB::Tag tag) override;

    /*NI*/ DB::Element_base* construct_empty_element( SERIAL::Class_id class_id) override;

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
    ///                             Can be \c NULL.
    /// \param store                Flag that distinguishes between store and edit operations (only
    ///                             used for error messages)
    void check_privacy_levels(
        DB::Privacy_level referencing_level,
        const DB::Tag_set& references,
        DB::Tag tag,
        const char* name,
        bool store);

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
    void end_transaction( Transaction_impl* transaction, bool commit);

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
    void dump( std::ostream& s, bool mask_pointer_values);

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
