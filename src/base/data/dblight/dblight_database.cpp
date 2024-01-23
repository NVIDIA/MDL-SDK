/***************************************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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
#include <base/hal/thread/i_thread_block.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/registry/i_config_registry.h>
#include <base/system/main/access_module.h>
#include <base/system/main/i_assert.h>

namespace MI {

namespace DBLIGHT {

Database_impl::Database_impl()
  : m_info_manager( new Info_manager( this)),
    m_transaction_manager( new Transaction_manager( this)),
    m_global_scope( new Scope_impl( this)),
    m_next_tag( 1)
{
    SYSTEM::Access_module<HOST::Host_module> host_module( false);
    int number_of_cpus = host_module->get_number_of_cpus();
    float cpu_load_limit = static_cast<float>( number_of_cpus);
    m_thread_pool = std::make_unique<THREAD_POOL::Thread_pool>(
        cpu_load_limit, /*gpu_load_limit*/ 0.0f, /*nr_of_worker_threads*/ 0);

    m_deserialization_manager = SERIAL::Deserialization_manager::create();

    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();
    CONFIG::update_value( registry, "check_serializer_store", m_check_serialization_store);
    if( m_check_serialization_store)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of serialization for database elements during store enabled.");
    CONFIG::update_value( registry, "check_serializer_edit", m_check_serialization_edit);
    if( m_check_serialization_edit)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Testing of serialization for database elements after edits enabled.");
}

Database_impl::~Database_impl()
{
    m_global_scope->unpin();

    // Clearing the transaction manager first will trigger assertions if there are infos left which
    // (incorrectly) still reference transactions.
    m_transaction_manager.reset();
    m_info_manager.reset();

   SERIAL::Deserialization_manager::release( m_deserialization_manager);

#ifdef DBLIGHT_ENABLE_STATISTICS
    dump_statistics( std::cerr, m_next_tag.load());
#endif // DBLIGHT_ENABLE_STATISTICS
}

DB::Scope* Database_impl::get_global_scope()
{
    return m_global_scope;
}

DB::Scope* Database_impl::lookup_scope( DB::Scope_id id)
{
    return id == 0 ? m_global_scope : nullptr;
}

DB::Scope* Database_impl::lookup_scope( const std::string& name)
{
    return name.empty() ? m_global_scope : nullptr;
}

void Database_impl::garbage_collection( int priority)
{
    THREAD::Block block( &m_lock);
    m_info_manager->garbage_collection( m_transaction_manager->get_lowest_open_transaction_id());
}

#define NOT_IMPLEMENTED { MI_ASSERT( !"Not implemented"); }

void Database_impl::lock( mi::Uint32 lock_id) NOT_IMPLEMENTED

bool Database_impl::unlock( mi::Uint32 lock_id) { MI_ASSERT( !"Not implemented"); return false; }

void Database_impl::check_is_locked( mi::Uint32 lock_id) NOT_IMPLEMENTED

mi::Sint32 Database_impl::set_memory_limits( size_t low_water, size_t high_water)
{
    MI_ASSERT( !"Not implemented");
    return -1;
}

void Database_impl::get_memory_limits( size_t& low_water, size_t& high_water) const
{
    MI_ASSERT( !"Not implemented");
    low_water = 0;
    high_water = 0;
}

void Database_impl::register_status_listener( DB::Status_listener* listener) NOT_IMPLEMENTED

void Database_impl::unregister_status_listener( DB::Status_listener* listener) NOT_IMPLEMENTED

void Database_impl::register_transaction_listener( DB::ITransaction_listener* listener)
NOT_IMPLEMENTED

void Database_impl::unregister_transaction_listener( DB::ITransaction_listener* listener)
NOT_IMPLEMENTED

void Database_impl::register_scope_listener( DB::IScope_listener* listener) NOT_IMPLEMENTED

void Database_impl::unregister_scope_listener( DB::IScope_listener* listener) NOT_IMPLEMENTED

mi::Sint32 Database_impl::execute_fragmented( DB::Fragmented_job* job, size_t count)
{
    return execute_fragmented( /*transaction*/ nullptr, job, count);
}

mi::Sint32 Database_impl::execute_fragmented_async(
    DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener)
{
    return execute_fragmented_async( /*transaction*/ nullptr, job, count, listener);
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

mi::Sint32 Database_impl::execute_fragmented(
    DB::Transaction* transaction, DB::Fragmented_job* job, size_t count)
{
    if( !job || count == 0)
        return -1;
    if( !transaction && (job->get_scheduling_mode() != DB::Fragmented_job::LOCAL))
        return -2;

    mi::base::Handle<DBLIGHT::Fragmented_job> wrapped_job(
        new DBLIGHT::Fragmented_job( transaction, count, job, /*listener*/ nullptr));
    m_thread_pool->submit_job_and_wait( wrapped_job.get());
    return 0;
}

mi::Sint32 Database_impl::execute_fragmented_async(
    DB::Transaction* transaction,
    DB::Fragmented_job* job,
    size_t count,
    DB::IExecution_listener* listener)
{
    if( !job || count == 0)
        return -1;
    if( job->get_scheduling_mode() != DB::Fragmented_job::LOCAL)
        return -2;

    mi::base::Handle<DBLIGHT::Fragmented_job> wrapped_job(
        new DBLIGHT::Fragmented_job( transaction, count, job, listener));
    m_thread_pool->submit_job( wrapped_job.get());
    return 0;
}

SERIAL::Deserialization_manager* Database_impl::get_deserialization_manager()
{
    return m_deserialization_manager;
}

void Database_impl::dump( std::ostream& s, bool mask_pointer_values)
{
    THREAD::Block_shared block( &m_lock);

    m_transaction_manager->dump( s, mask_pointer_values);
    m_info_manager->dump( s, mask_pointer_values);
}

DB::Database* factory()
{
    return new Database_impl;
}

#undef NOT_IMPLEMENTED

} // namespace DBLIGHT

} // namespace MI
