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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_DATABASE_H
#define BASE_DATA_DBLIGHT_DBLIGHT_DATABASE_H

#include <base/data/db/i_db_database.h>

#include <atomic>
#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mi/base/condition.h>
#include <mi/base/handle.h>

#include "dblight_util.h"

namespace MI {

namespace SERIAL { class Deserialization_manager; }
namespace THREAD_POOL { class Thread_pool; }

namespace DBLIGHT {

class Info_manager;
class Scope_impl;
class Scope_manager;
class Transaction_manager;

/// A request for a DB lock.
///
/// Used if a different thread attempts to acquire a lock already held by another thread.
class Db_lock_request
{
public:
    /// Constructor.
    Db_lock_request( const std::thread::id& id) : m_thread_id( id) { }

    std::thread::id m_thread_id;       ///< Thread requesting the lock.
    mi::base::Condition m_condition;   ///< Condition variable that the requesting thread blocks on.
};

/// A DB lock.
///
/// See #Database_impl::lock(), #Database_impl::unlock(), and #Database_impl::check_locked().
class Db_lock
{
public:
    /// Constructor.
    ///
    /// A newly acquired lock is locked once, and by the calling thread.
    Db_lock() : m_counter( 1), m_thread_id( std::this_thread::get_id()) { }

    mi::Uint32 m_counter;                    ///< Number of recursive locks.
    std::thread::id m_thread_id;             ///< Locking thread.
    std::list<Db_lock_request*> m_requests;  ///< Outstanding lock requests.
};

/// The database class manages the whole database.
///
/// Limits:
/// - Tags: at most 2^32-1 tags. If exceeded, wraps around to the invalid tag, and subsequently
///   allocates tags possibly still in use as new tags.
/// - Transaction IDs: wrap around at 2^32-1 is supported. At any time there must be at least a
///   range of 2^31 unused transaction IDs for proper ordering. Elements created more that 2^31
///   transactions ago will become invisible due to ordering problems.
/// - Versions within a transaction: at most 2^32 versions. If exceeded, wraps around and mixes up
///   the ordering.
class Database_impl : public DB::Database
{
public:
    /// Constructor
    ///
    /// \param thread_pool               The thread pool to use, or \c nullptr to use an independent
    ///                                  thread pool instance.
    /// \param deserialization_manager   The deserialization manager to use, or \c nullptr to use
    ///                                  an independent deserialization manager.
    /// \param enable_journal            Indicates whether the enable the journal. Maintaining the
    ///                                  journal requires memory and time.
    Database_impl(
        THREAD_POOL::Thread_pool* thread_pool,
        SERIAL::Deserialization_manager* deserialization_manager,
        bool enable_journal);

    /// Destructor, clears the database.
    virtual ~Database_impl();

    // methods of DB::Database

    DB::Scope* get_global_scope() override;
    DB::Scope* lookup_scope( DB::Scope_id id) override;
    DB::Scope* lookup_scope( const std::string& name) override;
    bool remove_scope( DB::Scope_id id) override;

    void prepare_close() override { }
    void close() override { delete this; }

    void garbage_collection( int priority) override;

    /// Note that the flexibility of lock IDs comes with a non-negligible overhead compared to
    /// a global mutex, in particular in case of lock contention.
    void lock( mi::Uint32 lock_id) override;
    bool unlock( mi::Uint32 lock_id) override;
    void check_is_locked( mi::Uint32 lock_id) override;

    /// Note that the configured limits are simply ignored.
    mi::Sint32 set_memory_limits( size_t low_water, size_t high_water) override;
    void get_memory_limits( size_t& low_water, size_t& high_water) const override;

    /// Note that status listeners are never invoked since the status of the DBLIGHT database
    /// never changes (always DB_OK).
    void register_status_listener( DB::IStatus_listener* listener) override;

    /// Note that status listeners are never invoked since the status of the DBLIGHT database
    /// never changes (always DB_OK).
    void unregister_status_listener( DB::IStatus_listener* listener) override;

    void register_transaction_listener( DB::ITransaction_listener* listener) override;
    void unregister_transaction_listener( DB::ITransaction_listener* listener) override;
    void register_scope_listener( DB::IScope_listener* listener) override;
    void unregister_scope_listener( DB::IScope_listener* listener) override;

    mi::Sint32 execute_fragmented( DB::Fragmented_job* job, size_t count) override;
    mi::Sint32 execute_fragmented_async(
        DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener) override;

    void suspend_current_job() override;
    void resume_current_job() override;
    void yield() override;

    // internal methods

    /// Returns the central database lock.
    THREAD::Shared_lock& get_lock() { return m_lock; }

    /// Returns the info manager.
    Info_manager* get_info_manager() { return m_info_manager.get(); }

    /// Returns the scope manager.
    Scope_manager* get_scope_manager() { return m_scope_manager.get(); }

    /// Returns the transaction manager.
    Transaction_manager* get_transaction_manager() { return m_transaction_manager.get(); }

    /// Returns the thread pool.
    THREAD_POOL::Thread_pool* get_thread_pool() { return m_thread_pool; }

    /// Returns the deserialization manager used for serialization checks.
    ///
    /// Note that the database itself does \em not require to register all classes of possible
    /// database elements upfront with the deserialization manager. This is only necessary if
    /// Transaction_impl::construct_empty_element() is called or if the debug options for
    /// serializer checks are enabled.
    SERIAL::Deserialization_manager* get_deserialization_manager();

    /// Indicates whether serialization should be tested in Transaction::store().
    bool get_check_serialization_store() const { return m_check_serialization_store; }

    /// Indicates whether serialization should be tested in Transaction::finish_edit().
    bool get_check_serialization_edit() const { return m_check_serialization_edit; }

    /// Indicates whether privacy levels of tag references should be checked in Transaction::store()
    /// and Transaction::finish_edit().
    bool get_check_privacy_levels() const { return m_check_privacy_levels; }

    /// Indicates whether reference cycles should be tested in Transaction::store().
    bool get_check_reference_cycles_store() const { return m_check_reference_cycles_store; }

    /// Indicates whether reference cycles should be tested in Transaction::finish_edit().
    bool get_check_reference_cycles_edit() const { return m_check_reference_cycles_edit; }

    /// Indicates whether the unsafe implementation of Transaction::name_to_tag() is enabled.
    bool get_unsafe_name_to_tag() const { return m_unsafe_name_to_tag; }

    /// Indicates whether the journal is enabled.
    bool get_journal_enabled() const { return m_journal_enabled; }

    /// Indicates whether the maximum journal size.
    size_t get_journal_max_size() const { return m_journal_max_size; }

    /// Notifies all status listeners.
    void notify_status_listeners( DB::Db_status argument);

    /// Type of a member function of DB::ITransaction_listener.
    using Transaction_listener_method
        = std::function< void( DB::ITransaction_listener&, DB::Transaction*) >;

    /// Notifies all transactions listeners.
    void notify_transaction_listeners(
        Transaction_listener_method method, DB::Transaction* argument);

    /// Type of a member function of DB::IScope_listener.
    using Scope_listener_method
        = std::function< void( DB::IScope_listener&, DB::Scope*) >;

    /// Notifies all scope listeners.
    void notify_scope_listeners(
        Scope_listener_method method, DB::Scope* argument);

    /// Used by transactions to allocate new tags.
    DB::Tag allocate_tag();

    /// Dumps the state of the database to the stream.
    void dump( std::ostream& s, bool verbose = false, bool mask_pointer_values = false);

private:
    /// The central database lock.
    THREAD::Shared_lock m_lock;

    /// The info manager.
    std::unique_ptr<Info_manager> m_info_manager;

    /// The scope manager.
    std::unique_ptr<Scope_manager> m_scope_manager;

    /// The transaction manager.
    std::unique_ptr<Transaction_manager> m_transaction_manager;

    /// The thread pool.
    THREAD_POOL::Thread_pool* m_thread_pool = nullptr;

    /// Indicates whether the thread pool is independent (or shared).
    bool m_independent_thread_pool = true;

    /// The next tag to allocate.
    std::atomic_uint32_t m_next_tag;

    /// The deserialization manager.
    SERIAL::Deserialization_manager* m_deserialization_manager = nullptr;

    /// Indicates whether the deserialization manager is independent (or shared).
    bool m_independent_deserialization_manager = true;

    /// The status listeners.
    std::vector<mi::base::Handle<DB::IStatus_listener> > m_status_listeners;

    /// The scope listeners.
    std::vector<mi::base::Handle<DB::IScope_listener> > m_scope_listeners;

    /// The transaction listeners.
    std::vector<mi::base::Handle<DB::ITransaction_listener> > m_transaction_listeners;

    /// All currently acquired DB locks.
    std::map<mi::Uint32, Db_lock> m_db_locks;

    /// Indicates whether serialization should be tested in Transaction::store().
    bool m_check_serialization_store = false;

    /// Indicates whether serialization should be tested in Transaction::finish_edit().
    bool m_check_serialization_edit = false;

    /// Indicates whether privacy levels of tag references should be checked in Transaction::store()
    /// and Transaction::finish_edit().
    ///
    /// Ideally this check should always be enabled. Due to unclear performance implications it is
    /// enabled by default only for debug builds.
#ifdef NDEBUG
    bool m_check_privacy_levels = false;
#else
    bool m_check_privacy_levels = true;
#endif

    /// Indicates whether reference cycles should be tested in Transaction::store().
    bool m_check_reference_cycles_store = false;

    /// Indicates whether reference cycles should be tested in Transaction::finish_edit).
    bool m_check_reference_cycles_edit = false;

    /// Indicates whether the unsafe implementation of Transaction::name_to_tag() is enabled.
    bool m_unsafe_name_to_tag = false;

    /// Indicates whether the journal is enabled.
    const bool m_journal_enabled;

    /// The maximum journal size.
    size_t m_journal_max_size = 10'000'000;

    /// The low water mark for the memory limits (ignored).
    size_t m_low_water = 0;

    /// The high water mark for the memory limits (ignored).
    size_t m_high_water = 0;
};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_DATABASE_H
