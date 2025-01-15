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
 ** \brief Header for Worker_thread.
 **/

#ifndef BASE_DATA_THREAD_POOL_THREAD_POOL_WORKER_THREAD_H
#define BASE_DATA_THREAD_POOL_THREAD_POOL_WORKER_THREAD_H

#include <mi/base/condition.h>
#include <mi/neuraylib/iserializer.h> // IJob_execution_context
#include <base/system/main/i_module_id.h>
#include <base/hal/thread/i_thread_thread.h>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace THREAD_POOL {

class Thread_pool;

/// The various states for worker threads.
///
/// The transition diagram is as follows:
///
/// STARTING
///     |
///     V
/// SLEEPING -> IDLE -> RUNNING -> SUSPENDED
///     |    <-      <-         <-
///     V
/// SHUTDOWN
///
/// See also the note for Worker_thread::m_state and THREAD_SUSPENDED.
enum Thread_state {
    THREAD_STARTING,
    THREAD_SLEEPING,
    THREAD_IDLE,
    THREAD_RUNNING,
    THREAD_SUSPENDED,
    THREAD_SHUTDOWN,
    N_THREAD_STATES
};

/// A worker thread used by the thread pool.
///
/// Worker threads are created on-demand by the thread pool.
class Worker_thread : public THREAD::Thread, public mi::neuraylib::IJob_execution_context
{
public:
    /// Constructor.
    ///
    /// Sets the thread state to THREAD_STARTING.
    Worker_thread( Thread_pool* thread_pool, mi::Uint32 cpu_id);

    /// Destructor.
    ///
    /// Expects the thread state to be THREAD_SHUTDOWN.
    ~Worker_thread();

    /// Starts the thread.
    ///
    /// Blocks until the run() method is actually executed and sets the state from THREAD_STARTING
    /// to THREAD_SLEEPING.
    void start();

    /// Shuts the thread down.
    ///
    /// Blocks until the run() methods ends and set the state to THREAD_SHUTDOWN.
    void shutdown();

    /// Wakes the thread up from sleeping state.
    ///
    /// Signals the condition variable used by run() between process_jobs() calls.
    void wake_up();

    /// Returns the thread ID or 0 if the thread is not running.
    ///
    /// Abstract method of IJob_execution_context.
    mi::Uint64 get_thread_id() const { return m_thread_id; }

    /// Sets the thread affinity.
    void set_thread_affinity_enabled( bool value);

    /// Sets the thread state.
    ///
    /// Takes care of decrementing the counter for the old state and incrementing the counter for
    /// the new state.
    void set_state( Thread_state state);

    /// Returns the threat state.
    ///
    /// Note that suspended threads still return THREAD_RUNNING instead of THREAD_SUSPENDED here
    /// because the pointer to the thread is not available during suspend/resume. This is just a
    /// cosmetic annoyance but has no bad effects. The thread state counters in the thread pool
    /// are updated correctly, though.
    Thread_state get_state() const { return m_state; }

private:
    /// The main method of the thread.
    ///
    /// Calls process_jobs() in a loop until shutdown is initiated. Waits for the condition variable
    /// after each invocation.
    void run();

    /// Processes jobs (if possible).
    ///
    /// Calls process_job() in a loop as long as this method is successful (or shutdown was
    /// initiated).
    void process_jobs();

    /// Processes a job (if possible).
    ///
    /// Attempts to get a job that can be executed from the work queue. If successful, executes it
    /// and returns \c true. Otherwise (no job, or does not fit load limits) returns \c false.
    bool process_job();

    /// The thread pool this worker thread belongs to.
    Thread_pool* m_thread_pool;

    /// The state of the worker thread.
    ///
    /// Note that suspended threads still use THREAD_RUNNING instead of THREAD_SUSPENDED here
    /// because the pointer to the thread is not available during suspend/resume. This is just a
    /// cosmetic annoyance but has no bad effects. The thread state counters in the thread pool
    /// are updated correctly, though.
    Thread_state m_state;

    /// If set, the thread will go into state THREAD_SHUTDOWN instead of THREAD_IDLE when woken
    /// up from state THREAD_SLEEPING.
    bool m_shutdown;

    /// Used to wake up the thread when in state THREAD_SLEEPING.
    mi::base::Condition m_condition;

    /// Used to synchronize thread startup between start() and run().
    mi::base::Condition m_start_condition;

    /// The thread ID.
    mi::Uint64 m_thread_id;

    /// The CPU ID (used if thread affinity is enabled).
    mi::Uint32 m_cpu_id;

    /// Indicates whether thread affinity is enabled.
    bool m_thread_affinity_enabled;
};

} // namespace THREAD_POOL

} // namespace MI

#endif // BASE_DATA_THREAD_POOL_THREAD_POOL_WORKER_THREAD_H
