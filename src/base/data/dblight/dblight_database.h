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
#include <map>
#include <memory>
#include <string>

#include "dblight_util.h"

namespace MI {

namespace SERIAL { class Deserialization_manager; }
namespace THREAD_POOL { class Thread_pool; }

namespace DBLIGHT {

class Info_manager;
class Scope_impl;
class Scope_manager;
class Transaction_manager;

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
///
/// "NI" means DBLIGHT does not implement/support that method of the interface.
class Database_impl : public DB::Database
{
public:
    /// Constructor
    Database_impl();

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

    /*NI*/ void lock( mi::Uint32 lock_id) override;
    /*NI*/ bool unlock( mi::Uint32 lock_id) override;
    /*NI*/ void check_is_locked( mi::Uint32 lock_id) override;

    /*NI*/ mi::Sint32 set_memory_limits( size_t low_water, size_t high_water) override;
    /*NI*/ void get_memory_limits( size_t& low_water, size_t& high_water) const override;

    /*NI*/ void register_status_listener( DB::Status_listener* listener) override;
    /*NI*/ void unregister_status_listener( DB::Status_listener* listener) override;
    /*NI*/ void register_transaction_listener( DB::ITransaction_listener* listener) override;
    /*NI*/ void unregister_transaction_listener( DB::ITransaction_listener* listener) override;
    /*NI*/ void register_scope_listener( DB::IScope_listener* listener) override;
    /*NI*/ void unregister_scope_listener( DB::IScope_listener* listener) override;

    /// Only the scheduling mode LOCAL is supported.
    mi::Sint32 execute_fragmented( DB::Fragmented_job* job, size_t count) override;

    /// Only the scheduling mode LOCAL is supported.
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

    /// Used by transactions to allocate new tags.
    DB::Tag allocate_tag() { return DB::Tag( m_next_tag++); }

    /// Implementation of DB::Database::execute_fragmented() and
    /// DB::Transaction::execute_fragmented().
    ///
    /// \param transaction   The transaction to be passed around. Might be \c NULL. RCS:NEU
    /// \param job           The fragmented job to be executed. RCS:NEU
    /// \param count         See #DB::Database::execute_fragmented() for details.
    /// \return              See #DB::Database::execute_fragmented() for details.
    mi::Sint32 execute_fragmented(
        DB::Transaction* transaction, DB::Fragmented_job* job, size_t count);

    /// Implementation of DB::Database::execute_fragmented_async() and
    /// DB::Transaction::execute_fragmented_async().
    ///
    /// \param transaction   The transaction to be passed around. Might be \c NULL.
    /// \param job           The fragmented job to be executed. RCS:NEU
    /// \param count         See #DB::Database::execute_fragmented() for details.
    /// \param listener      See #DB::Database::execute_fragmented() for details. RCS:NEU
    /// \return              See #DB::Database::execute_fragmented() for details.
    mi::Sint32 execute_fragmented_async(
        DB::Transaction* transaction,
        DB::Fragmented_job* job,
        size_t count,
        DB::IExecution_listener* listener);

    /// Returns the deserialization manager used for serialization checks.
    ///
    /// Note that the database itself does \em not require to register all classes of possible
    /// database elements upfront with the deserialization manager. This is only necessary if the
    /// debug options for serializer checks are enabled.
    SERIAL::Deserialization_manager* get_deserialization_manager();

    /// Indicates whether serialization should be tested in Transaction::store().
    bool get_check_serialization_store() const { return m_check_serialization_store; }

    /// Indicates whether serialization should be tested in Transaction::finish_edit().
    bool get_check_serialization_edit() const { return m_check_serialization_edit; }

    /// Indicates whether privacy levels of tag references should be checked in Transaction::store()
    /// and Transaction::finish_edit().
    bool get_check_privacy_levels() const { return m_check_privacy_levels; }

    /// Dumps the state of the database to the stream.
    void dump( std::ostream& s, bool mask_pointer_values = false);

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
    std::unique_ptr<THREAD_POOL::Thread_pool> m_thread_pool;

    /// The next tag to allocate.
    std::atomic_uint32_t m_next_tag;

    /// The deserialization manager.
    SERIAL::Deserialization_manager* m_deserialization_manager;

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
};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_DATABASE_H
