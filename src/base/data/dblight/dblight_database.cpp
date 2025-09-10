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

#include "pch.h"

#include "dblight_database.h"

#include "dblight_info.h"
#include "dblight_fragmented_job.h"
#include "dblight_scope.h"
#include "dblight_transaction.h"
#include "dblight_util.h"

// After inclusion of dblight_util.h which might define the macro.
#ifdef DBLIGHT_ENABLE_STATISTICS
#include <iostream>
#endif // DBLIGHT_ENABLE_STATISTICS

#include <base/data/db/i_db_element.h>
#include <base/data/serial/serial.h>
#include <base/data/thread_pool/i_thread_pool_thread_pool.h>
#include <base/hal/host/i_host.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/registry/i_config_registry.h>
#include <base/system/main/access_module.h>
#include <base/system/main/i_assert.h>

namespace MI {

namespace DBLIGHT {

Database_impl::Database_impl(
    THREAD_POOL::Thread_pool* thread_pool,
    SERIAL::Deserialization_manager* deserialization_manager,
    bool enable_journal)
  : m_info_manager( new Info_manager( this)),
    m_scope_manager( new Scope_manager( this)),
    m_transaction_manager( new Transaction_manager( this)),
    m_next_tag( 1),
    m_journal_enabled( enable_journal)
{
    if( thread_pool) {
        m_thread_pool = thread_pool;
        m_independent_thread_pool = false;
    } else {
        SYSTEM::Access_module<HOST::Host_module> host_module( false);
        int number_of_cpus = host_module->get_number_of_cpus();
        auto cpu_load_limit = static_cast<float>( number_of_cpus);
        m_thread_pool = new THREAD_POOL::Thread_pool(
            cpu_load_limit, /*gpu_load_limit*/ 0.0f, /*nr_of_worker_threads*/ 0);
        m_independent_thread_pool = true;
    }

    if( deserialization_manager) {
        m_deserialization_manager = deserialization_manager;
        m_independent_deserialization_manager = false;
    } else {
        m_deserialization_manager = SERIAL::Deserialization_manager::create();
        m_independent_deserialization_manager = true;
    }

    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();

    CONFIG::update_value( registry, "check_serializer_store", m_check_serialization_store);
    if( m_check_serialization_store)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of serialization for database elements and jobs during store enabled.");

    CONFIG::update_value( registry, "check_serializer_edit", m_check_serialization_edit);
    if( m_check_serialization_edit)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of serialization for database elements after edits enabled.");

    CONFIG::update_value( registry, "check_privacy_levels", m_check_privacy_levels);
#ifdef NDEBUG
    if( m_check_privacy_levels)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of privacy levels of references during store and after edits enabled.");
#endif

    CONFIG::update_value( registry, "check_reference_cycles_store", m_check_reference_cycles_store);
    if( m_check_reference_cycles_store)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of reference cycles for database elements during store enabled.");

    CONFIG::update_value( registry, "check_reference_cycles_edit", m_check_reference_cycles_edit);
    if( m_check_reference_cycles_edit)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of reference cycles for database elements after edits enabled.");

    CONFIG::update_value( registry, "unsafe_name_to_tag", m_unsafe_name_to_tag);
    if( m_unsafe_name_to_tag)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Unsafe implementation of DB::Transaction::name_to_tag() enabled.");

    CONFIG::update_value( registry, "dblight_journal_max_size", m_journal_max_size);

#if defined( DBLIGHT_NO_SHARED_LOCK)
    LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
        "Using THREAD::Lock class for main database lock.");
#elif defined( DBLIGHT_NO_BLOCK_SHARED)
    LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
        "Using THREAD::Shared_lock exclusively with THREAD::Block for main database lock.");
#endif
}

Database_impl::~Database_impl()
{
    // Dumping the database content here might provide some insights into assertions during
    // destruction.
    // dump( std::cerr, /*verbose*/ true, /*mask_pointer_values*/ true);

    // Removing all scopes should leave no transactions or infos behind.
    m_scope_manager.reset();
    m_transaction_manager.reset();
    m_info_manager.reset();

   if( m_independent_deserialization_manager)
       SERIAL::Deserialization_manager::release( m_deserialization_manager);
   if( m_independent_thread_pool)
       delete m_thread_pool;

#ifdef DBLIGHT_ENABLE_STATISTICS
    dump_statistics( std::cerr, m_next_tag.load());
#endif // DBLIGHT_ENABLE_STATISTICS
}

DB::Scope* Database_impl::get_global_scope()
{
    return lookup_scope( 0);
}

DB::Scope* Database_impl::lookup_scope( DB::Scope_id id)
{
    THREAD::Block block( &m_lock);
    return m_scope_manager->lookup_scope( id);
}

DB::Scope* Database_impl::lookup_scope( const std::string& name)
{
    THREAD::Block block( &m_lock);
    return m_scope_manager->lookup_scope( name);
}

bool Database_impl::remove_scope( DB::Scope_id id)
{
    return m_scope_manager->remove_scope( id);
}

void Database_impl::garbage_collection( int priority)
{
    THREAD::Block block( &m_lock);
    m_info_manager->garbage_collection( m_transaction_manager->get_lowest_open_transaction_id());
}

void Database_impl::lock( mi::Uint32 lock_id)
{
    THREAD::Block block( &m_lock);

    auto it = m_db_locks.find( lock_id);

    // If the lock ID is unknown, create the lock and lock it.
    if( it == m_db_locks.end()) {
        m_db_locks.emplace( std::make_pair( lock_id, Db_lock()));
        return;
    }

    Db_lock& lock = it->second;
    std::thread::id own_thread_id = std::this_thread::get_id();

    // If the lock is already locked by this thread, just increase the lock count.
    if( lock.m_thread_id == own_thread_id) {
        ++lock.m_counter;
        return;
    }

    // Otherwise, enqueue a lock request.
    MI_ASSERT( lock.m_counter > 0);
    Db_lock_request request( own_thread_id);
    lock.m_requests.push_back( &request);

    // Release global lock and block on the request.
    block.release();
    request.m_condition.wait();
}

bool Database_impl::unlock( mi::Uint32 lock_id)
{
    THREAD::Block block( &m_lock);

    auto it = m_db_locks.find( lock_id);

    if( it == m_db_locks.end()) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE,
            "Attempt to unlock unknown DB lock with ID %u.", lock_id);
        return false;
    }

    Db_lock& lock = it->second;
    std::thread::id own_thread_id = std::this_thread::get_id();

    if( lock.m_thread_id != own_thread_id) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE,
            "Attempt to unlock DB lock with ID %u from another thread.", lock_id);
        return false;
    }

    // Decrease the lock count. Return if the lock is still locked recursively by this thread.
    --lock.m_counter;
    if( lock.m_counter > 0)
        return true;

    // This thread just unlocked the lock (including recursively). If there are no outstanding
    // request, remove the lock itself.
    if( lock.m_requests.empty()) {
        m_db_locks.erase( it);
        return true;
    }

    // Otherwise, wake up oldest request and remove it from the queue.
    Db_lock_request* request = lock.m_requests.front();
    lock.m_requests.pop_front();
    lock.m_thread_id = request->m_thread_id;
    lock.m_counter = 1;
    request->m_condition.signal();

    return true;
}

void Database_impl::check_is_locked( mi::Uint32 lock_id)
{
    auto it = m_db_locks.find( lock_id);

    if( it == m_db_locks.end()) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "DB lock with ID %u is not locked.", lock_id);
        MI_ASSERT( !"DB lock not locked");
        return;
    }

    Db_lock& lock = it->second;
    std::thread::id own_thread_id = std::this_thread::get_id();

    if( lock.m_thread_id != own_thread_id) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "DB lock with ID %u is locked from another thread.",
            lock_id);
        MI_ASSERT( !"DB lock locked from other thread");
    }
}

mi::Sint32 Database_impl::set_memory_limits( size_t low_water, size_t high_water)
{
    if( (low_water >= high_water) && (high_water > 0))
        return -1;

    m_low_water  = low_water;
    m_high_water = high_water;
    return 0;
}

void Database_impl::get_memory_limits( size_t& low_water, size_t& high_water) const
{
    low_water  = m_low_water;
    high_water = m_high_water;
}

void Database_impl::register_status_listener( DB::IStatus_listener* listener)
{
    if( !listener)
        return;

    THREAD::Block block( &m_lock);
    m_status_listeners.push_back( make_handle_dup( listener));
}

void Database_impl::unregister_status_listener( DB::IStatus_listener* listener)
{
    THREAD::Block block( &m_lock);
    auto it = std::find( m_status_listeners.begin(), m_status_listeners.end(), listener);
    if( it != m_status_listeners.end())
        m_status_listeners.erase( it);
}

void Database_impl::register_transaction_listener( DB::ITransaction_listener* listener)
{
    if( !listener)
        return;

    THREAD::Block block( &m_lock);
    m_transaction_listeners.push_back( make_handle_dup( listener));
}

void Database_impl::unregister_transaction_listener( DB::ITransaction_listener* listener)
{
    THREAD::Block block( &m_lock);
    auto it = std::find( m_transaction_listeners.begin(), m_transaction_listeners.end(), listener);
    if( it != m_transaction_listeners.end())
        m_transaction_listeners.erase( it);
}

void Database_impl::register_scope_listener( DB::IScope_listener* listener)
{
    if( !listener)
        return;

    THREAD::Block block( &m_lock);
    m_scope_listeners.push_back( make_handle_dup( listener));
}

void Database_impl::unregister_scope_listener( DB::IScope_listener* listener)
{
    THREAD::Block block( &m_lock);
    auto it = std::find( m_scope_listeners.begin(), m_scope_listeners.end(), listener);
    if( it != m_scope_listeners.end())
        m_scope_listeners.erase( it);
}

mi::Sint32 Database_impl::execute_fragmented( DB::Fragmented_job* job, size_t count)
{
    if( !job || count == 0)
        return -1;
    if( job->get_scheduling_mode() != DB::Fragmented_job::LOCAL)
        return -2;
    if( job->get_priority() < 0)
        return -3;

    mi::base::Handle<DBLIGHT::Fragmented_job> wrapped_job(
        new DBLIGHT::Fragmented_job( /*transaction*/ nullptr, count, job, /*listener*/ nullptr));
    m_thread_pool->submit_job_and_wait( wrapped_job.get());
    return 0;
}

mi::Sint32 Database_impl::execute_fragmented_async(
    DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener)
{
    if( !job || count == 0)
        return -1;
    if( job->get_scheduling_mode() != DB::Fragmented_job::LOCAL)
        return -2;
    if( job->get_priority() < 0)
        return -3;

    mi::base::Handle<DBLIGHT::Fragmented_job> wrapped_job(
        new DBLIGHT::Fragmented_job( /*transaction*/ nullptr, count, job, listener));
    m_thread_pool->submit_job( wrapped_job.get());
    return 0;
}

void Database_impl::suspend_current_job()
{
    m_thread_pool->suspend_current_job();
}

void Database_impl::resume_current_job()
{
    m_thread_pool->resume_current_job();
}

void Database_impl::yield()
{
    m_thread_pool->yield();
}

SERIAL::Deserialization_manager* Database_impl::get_deserialization_manager()
{
    return m_deserialization_manager;
}

void Database_impl::notify_status_listeners( DB::Db_status argument)
{
    MI_ASSERT( !"Unexpected status change");

    m_lock.check_is_owned();
    for( const auto& listener: m_status_listeners)
        listener->status_changed( argument);
}

void Database_impl::notify_transaction_listeners(
    Transaction_listener_method method, DB::Transaction* argument)
{
    m_lock.check_is_owned();
    for( const auto& listener: m_transaction_listeners)
        method( *listener.get(), argument);
}

void Database_impl::notify_scope_listeners(
    Scope_listener_method method, DB::Scope* argument)
{
    m_lock.check_is_owned();
    for( const auto& listener: m_scope_listeners)
        method( *listener.get(), argument);
}

DB::Tag Database_impl::allocate_tag()
{
    mi::Uint32 result = m_next_tag++;
    if( result == 0)
        result = m_next_tag++;
    return DB::Tag( result);
}

void Database_impl::dump( std::ostream& s, bool verbose, bool mask_pointer_values)
{
    THREAD::Block_shared block( &m_lock);

    m_scope_manager->dump( s, verbose, mask_pointer_values);
    m_transaction_manager->dump( s, verbose, mask_pointer_values);
    m_info_manager->dump( s, verbose, mask_pointer_values);
}

DB::Database* factory(
    THREAD_POOL::Thread_pool* thread_pool,
    SERIAL::Deserialization_manager* deserialization_manager,
    bool enable_journal)
{
    return new Database_impl( thread_pool, deserialization_manager, enable_journal);
}

} // namespace DBLIGHT

} // namespace MI
