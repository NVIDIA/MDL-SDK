/***************************************************************************************************
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation for Worker_thread.
 **/

#include "pch.h"

#include "thread_pool_worker_thread.h"

#include <mi/base/config.h>

#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>

#include "i_thread_pool_ijob.h"
#include "i_thread_pool_thread_pool.h"

namespace MI {

namespace THREAD_POOL {

Worker_thread::Worker_thread( Thread_pool* thread_pool, mi::Uint32 cpu_id)
  : m_thread_pool( thread_pool),
    m_state( THREAD_STARTING),
    m_shutdown( false),
    m_thread_id( 0),
    m_cpu_id( cpu_id),
    m_thread_affinity_enabled( false)
{
    m_thread_pool->increase_thread_state_counter( m_state);
}

Worker_thread::~Worker_thread()
{
    ASSERT( M_THREAD_POOL, m_state == THREAD_SHUTDOWN);
    m_thread_pool->decrease_thread_state_counter( m_state);

}

void Worker_thread::start()
{
    ASSERT( M_THREAD_POOL, m_state == THREAD_STARTING);
    THREAD::Thread::start();
    m_start_condition.wait();
    ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
}

void Worker_thread::shutdown()
{
    ASSERT( M_THREAD_POOL,    m_state == THREAD_SLEEPING || m_state == THREAD_IDLE
                           || m_state == THREAD_RUNNING  || m_state == THREAD_SUSPENDED);
    m_shutdown = true;
    m_condition.signal();
    THREAD::Thread::join();
}

void Worker_thread::wake_up()
{
    ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
    m_condition.signal();
}

void Worker_thread::set_thread_affinity_enabled( bool value)
{
    m_thread_affinity_enabled = value;
}

void Worker_thread::set_state( Thread_state state)
{
    m_thread_pool->decrease_thread_state_counter( m_state);
    m_state = state;
    m_thread_pool->increase_thread_state_counter( m_state);
}

void Worker_thread::run()
{
    m_thread_id = static_cast<mi::Uint64>( THREAD::Thread_id().get_uint());

    ASSERT( M_THREAD_POOL, m_state == THREAD_STARTING);
    set_state( THREAD_SLEEPING);
    m_start_condition.signal();

    m_condition.wait();
    while( !m_shutdown) {
        ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
        set_state( THREAD_IDLE);
        process_jobs();
        ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
        m_condition.wait();
    }

    ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
    set_state( THREAD_SHUTDOWN);

    m_thread_id = 0;
}

void Worker_thread::process_jobs()
{
    bool job_done;
    do {
        ASSERT( M_THREAD_POOL, m_state == THREAD_IDLE);
        job_done = process_job();
        ASSERT( M_THREAD_POOL, m_state == THREAD_IDLE || m_state == THREAD_SLEEPING);
    } while( job_done);
    ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
}

bool Worker_thread::process_job()
{
    // LOG::mod_log->vdebug( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Entering process_job()");
    IJob* job = m_thread_pool->get_next_job( this);
    if( !job) {
        // LOG::mod_log->vdebug( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        //     "Leaving process_job() (got no job)");
        ASSERT( M_THREAD_POOL, m_state == THREAD_SLEEPING);
        return false;
    }

    [[maybe_unused]] bool result = m_thread_affinity_enabled ? pin_cpu( m_cpu_id) : unpin_cpu();
#ifndef MI_PLATFORM_MACOSX
    // Setting the thread affinity is not supported on MacOS X.
    ASSERT( M_THREAD_POOL, result);
#endif

    ASSERT( M_THREAD_POOL, m_state == THREAD_IDLE);
    set_state( THREAD_RUNNING);
    // LOG::mod_log->vdebug( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Executing job %p ... ", job);
    job->execute( this);
    // LOG::mod_log->vdebug( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Executing job %p done.", job);
    ASSERT( M_THREAD_POOL, m_state == THREAD_RUNNING);
    set_state( THREAD_IDLE);

    m_thread_pool->job_execution_finished( this, job);
    job->release();
    // LOG::mod_log->vdebug( M_THREAD_POOL, LOG::Mod_log::C_MISC,
    //     "Leaving process_job() (executed a job)");
    return true;
}

} // namespace THREAD_POOL

} // namespace MI
