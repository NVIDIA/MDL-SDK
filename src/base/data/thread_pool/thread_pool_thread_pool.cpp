/***************************************************************************************************
 * Copyright (c) 2013-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation for Thread_pool.
 **/

#include "pch.h"

#include "i_thread_pool_thread_pool.h"
#include "thread_pool_jobs.h"

#include <cfloat>
#include <utility>
#include <boost/core/ignore_unused.hpp>
#include <base/hal/time/i_time.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>


// #define MI_THREAD_POOL_VERBOSE

namespace MI {

namespace THREAD_POOL {

/// Permit jobs that do not create any CPU load at all (for jobs that do not do much work, RPC
/// calls), etc. The number of threads is \em not bound in terms of the number of CPUs.
mi::Float32 Thread_pool::s_min_cpu_load = 0.0f;

/// A positive number might create problems if the system has no GPUs.
mi::Float32 Thread_pool::s_min_gpu_load = 0.0f;

Thread_pool::Thread_pool(
    mi::Float32 cpu_load_limit,
    mi::Float32 gpu_load_limit,
    mi::Size nr_of_worker_threads)
  : m_cpu_load_limit( cpu_load_limit),
    m_gpu_load_limit( gpu_load_limit),
    m_cpu_load_license_limit( FLT_MAX),
    m_gpu_load_license_limit( FLT_MAX),
    m_current_cpu_load( 0.0f),
    m_current_gpu_load( 0.0f),
    m_thread_affinity( false),
    m_next_cpu_id( 0),
    m_shutdown( false)
{
    for( mi::Size i = 0; i < N_THREAD_STATES; ++i)
        m_thread_state_counter[i] = 0;

    for( mi::Size i = 0; i < nr_of_worker_threads; ++i)
        create_worker_thread();

    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_STARTING]  == 0);
    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_SLEEPING]  == nr_of_worker_threads);
}

Thread_pool::~Thread_pool()
{
    mi::base::Lock::Block block( &m_lock);
    m_shutdown = true;

    // Wait until job queue is empty.
    while( !m_job_queue.empty()) {
        block.release();
        TIME::sleep( 0.01);
        block.set( &m_lock);
    }

    // Wait for all worker threads to terminate.
    //
    // Do not hold m_lock because idle threads might call one last time into get_next_job() and
    // wait for m_lock. Accessing m_all_threads without holding m_lock should be fine since at
    // this time no new threads should be created anymore and m_all_threads should remain constant.
    //
    // If this approach is not sufficient one could use a more elaborate scheme were worker threads
    // are told to shutdown (without blocking) and wait in a loop similar as above for confirmations
    // from all threads that they left their main loop.
    block.release();
    for( mi::Size i = 0; i < m_all_threads.size(); ++i)
        m_all_threads[i]->shutdown();
    block.set( &m_lock);

    ASSERT( M_THREAD_POOL, m_job_queue.empty());
    ASSERT( M_THREAD_POOL, m_running_jobs.empty());
    ASSERT( M_THREAD_POOL, m_suspended_jobs.empty());

    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_STARTING]  == 0);
    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_SLEEPING]  == 0);
    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_IDLE]      == 0);
    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_RUNNING]   == 0);
    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_SUSPENDED] == 0);
    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_SHUTDOWN]  == m_all_threads.size());

    for( mi::Size i = 0; i < m_all_threads.size(); ++i)
        delete m_all_threads[i];
    m_all_threads.clear();
    m_sleeping_threads.clear();

    ASSERT( M_THREAD_POOL, m_thread_state_counter[THREAD_SHUTDOWN]  == 0);
}

void Thread_pool::install_admin_http_server_page( HTTP::Server* server, const char* uri)
{
    ASSERT( M_THREAD_POOL, !"Not implemented");
}

bool Thread_pool::set_cpu_load_limit( mi::Float32 limit)
{
    if( limit < 1.0f)
        return false;

    mi::base::Lock::Block block( &m_lock);
    m_cpu_load_limit = limit > m_cpu_load_license_limit ? m_cpu_load_license_limit : limit;
    return true;
}

bool Thread_pool::set_gpu_load_limit( mi::Float32 limit)
{
    if( limit < 1.0f)
        return false;

    mi::base::Lock::Block block( &m_lock);
    m_gpu_load_limit = limit > m_gpu_load_license_limit ? m_gpu_load_license_limit : limit;
    return true;
}

bool Thread_pool::set_cpu_load_license_limit( mi::Float32 license_limit)
{
    if( license_limit < 1.0f)
        return false;

    mi::base::Lock::Block block( &m_lock);
    m_cpu_load_license_limit = license_limit;
    if( m_cpu_load_limit > license_limit)
        m_cpu_load_limit = license_limit;
    return true;
}

bool Thread_pool::set_gpu_load_license_limit( mi::Float32 license_limit)
{
    if( license_limit < 1.0f)
        return false;

    mi::base::Lock::Block block( &m_lock);
    m_gpu_load_license_limit = license_limit;
    if( m_gpu_load_limit > license_limit)
        m_gpu_load_limit = license_limit;
    return true;
}

bool Thread_pool::set_thread_affinity_enabled( bool value)
{
    // This test is supposed to tell whether the actual call asynchronously executed by the worker
    // threads will succeed.
    int cpu = THREAD::Thread::get_cpu();
    if (cpu < 0)
        return false;

    mi::base::Lock::Block block( &m_lock);

    if (value == m_thread_affinity)
        return true;

    m_thread_affinity = value;

    for( mi::Size i = 0; i < m_all_threads.size(); ++i)
        m_all_threads[i]->set_thread_affinity_enabled( value);
    return true;
}

bool Thread_pool::get_thread_affinity_enabled() const
{
    mi::base::Lock::Block block( &m_lock);
    return m_thread_affinity;
}

void Thread_pool::submit_job( IJob* job)
{
#ifdef ENABLE_ASSERT
    // Check that jobs are not submitted from suspended worker threads (to prevent misuse of the
    // suspend/resume feature).
    mi::Float32 cpu_load;
    mi::Float32 gpu_load;
    mi::Sint8 priority;
    mi::Uint64 thread_id;
    bool suspended_worker_thread
        = get_current_job_data( cpu_load, gpu_load, priority, thread_id, true);
    ASSERT( M_THREAD_POOL, !suspended_worker_thread);
#endif // ENABLE_ASSERT

    submit_job_internal( job, /*log_asynchronous*/ true);
}

void Thread_pool::submit_job_and_wait( IJob* job)
{
#ifdef ENABLE_ASSERT
    // Check that jobs are not submitted from suspended worker threads (to prevent misuse of the
    // suspend/resume feature).
    mi::Float32 cpu_load;
    mi::Float32 gpu_load;
    mi::Sint8 priority;
    mi::Uint64 thread_id;
    bool suspended_worker_thread
        = get_current_job_data( cpu_load, gpu_load, priority, thread_id, true);
    ASSERT( M_THREAD_POOL, !suspended_worker_thread);
#endif // ENABLE_ASSERT

    // Submit new job ...
    mi::base::Handle<Condition_job> wrapped_job( new Condition_job( job));
    submit_job_internal( wrapped_job.get(), /*log_asynchronous*/ false);

    // ... before the current job (if any) is suspended, such that child jobs have higher priority
    // than the jobs currently in the queue.
    suspend_current_job();
    wrapped_job->wait();
    resume_current_job();
}

bool Thread_pool::remove_job( IJob* job)
{
    mi::base::Lock::Block block( &m_lock);

    Job_queue::iterator it_map     = m_job_queue.begin();
    Job_queue::iterator it_map_end = m_job_queue.end();
    while( it_map != it_map_end) {

        Job_list::iterator it_list     = it_map->second.begin();
        Job_list::iterator it_list_end = it_map->second.end();
        while( it_list != it_list_end) {

            if( it_list->get() == job) {
                it_map->second.erase( it_list);
                if( it_map->second.empty())
                    m_job_queue.erase( it_map);
                return true;
            }

            ++it_list;
        }
        ++it_map;
    }

    return false;
}

void Thread_pool::suspend_current_job()
{
    suspend_current_job_internal( /*only_for_higher_priority*/ false);
}

void Thread_pool::resume_current_job()
{
    resume_current_job_internal();
}

void Thread_pool::yield()
{
    bool success = suspend_current_job_internal( /*only_for_higher_priority*/ true);
    if( success)
        resume_current_job_internal();
}

IJob* Thread_pool::get_next_job( Worker_thread* thread)
{
    mi::base::Lock::Block block( &m_lock);

    ASSERT( M_THREAD_POOL, thread->get_state() == THREAD_IDLE);

    Job_queue::iterator it_map      = m_job_queue.begin();
    Job_queue::iterator it_map_end  = m_job_queue.end();
    Job_list::iterator  it_list;
    Job_list::iterator  it_list_end;
#ifdef MI_THREAD_POOL_VERBOSE
    mi::Size k = 0;
#endif // MI_THREAD_POOL_VERBOSE

    IJob* job = 0;
    mi::Float32 requested_cpu_load = 0.f;
    mi::Float32 requested_gpu_load = 0.f;

    // find first job whose resource request fits the load limits
    while( it_map != it_map_end) {

        it_list     = it_map->second.begin();
        it_list_end = it_map->second.end();
        while( it_list != it_list_end) {

            job = it_list->get();
            requested_cpu_load = job->get_cpu_load();
            requested_gpu_load = job->get_gpu_load();
            adjust_load( requested_cpu_load, requested_gpu_load);
            if( job_fits_load_limits( requested_cpu_load, requested_gpu_load))
                break;
#ifdef MI_THREAD_POOL_VERBOSE
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "Delaying job %p, queue index %" FMT_SIZE_T ", "
                "CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d\n", job, (size_t) k,
                requested_cpu_load, m_current_cpu_load, m_cpu_load_limit,
                requested_gpu_load, m_current_gpu_load, m_gpu_load_limit,
                (int) job->get_priority());
#endif // MI_THREAD_POOL_VERBOSE
            ++it_list;
#ifdef MI_THREAD_POOL_VERBOSE
            ++k;
#endif // MI_THREAD_POOL_VERBOSE
            job = 0;
        }
        if( job)
            break;
        ++it_map;
    }

    if( it_map == it_map_end) {
        ASSERT( M_THREAD_POOL, !job);
        thread->set_state( THREAD_SLEEPING);
        std::pair<Sleeping_threads::iterator,bool> result = m_sleeping_threads.insert( thread);
        ASSERT( M_THREAD_POOL, result.second);
        boost::ignore_unused( result);
        return 0;
    }

#ifdef MI_THREAD_POOL_VERBOSE
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Executing job %p, queue index %" FMT_SIZE_T ", "
        "CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d\n", job, (size_t) k,
        requested_cpu_load, m_current_cpu_load, m_cpu_load_limit,
        requested_gpu_load, m_current_gpu_load, m_gpu_load_limit,
        (int) job->get_priority());
#endif // MI_THREAD_POOL_VERBOSE

    // retain first matching job for the return value below (before potentially removing it from the
    // queue)
    job->retain();

    // notify job about upcoming execute() call (might affect is_remaining_work_splittable() below)
    job->pre_execute( thread);

    // remove job from queue if the job does no want more parallel calls
    bool dequeue_job = !job->is_remaining_work_splittable();
    if( dequeue_job) {
        it_map->second.erase( it_list);
        if( it_map->second.empty())
            m_job_queue.erase( it_map);
    }

    // adjust current load
    m_current_cpu_load += requested_cpu_load;
    m_current_gpu_load += requested_gpu_load;

    // wake up another worker thread for jobs that want more parallel calls (after removing this
    // thread from the set of sleeping threads)
    if( !dequeue_job)
       wake_up_worker_thread();

    // map thread to the job
    mi::Uint64 thread_id = thread->get_thread_id();
    ASSERT( M_THREAD_POOL, m_running_jobs.find( thread_id) == m_running_jobs.end());
    m_running_jobs[thread_id] = job;

    return job;
}

void Thread_pool::job_execution_finished( Worker_thread* thread, IJob* job)
{
    mi::base::Lock::Block block( &m_lock);

    ASSERT( M_THREAD_POOL, thread->get_state() == THREAD_IDLE);

    mi::Float32 requested_cpu_load = job->get_cpu_load();
    mi::Float32 requested_gpu_load = job->get_gpu_load();
#ifdef MI_THREAD_POOL_VERBOSE
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Finished job %p, queue index n/a, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, "
        "priority %d\n", job,
        requested_cpu_load, m_current_cpu_load, m_cpu_load_limit,
        requested_gpu_load, m_current_gpu_load, m_gpu_load_limit,
        (int) job->get_priority());
#endif // MI_THREAD_POOL_VERBOSE

    // adjust current load except for resume jobs
    if( job->get_iid() != Resume_job::IID()) {
        adjust_load( requested_cpu_load, requested_gpu_load);
        m_current_cpu_load -= requested_cpu_load;
        m_current_gpu_load -= requested_gpu_load;
    }

    // unmap job from thread
    mi::Uint64 thread_id = thread->get_thread_id();
    Job_map::iterator it = m_running_jobs.find( thread_id);
    ASSERT( M_THREAD_POOL, it != m_running_jobs.end());
    m_running_jobs.erase( it); //-V783 PVS
}

void Thread_pool::dump_load() const
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Current CPU load: %.1f/%.1f, GPU load: %.1f/%.1f",
        m_current_cpu_load, m_cpu_load_limit, m_current_gpu_load, m_gpu_load_limit);
}

void Thread_pool::dump_thread_state_counters() const
{
    mi::Uint32 starting  = m_thread_state_counter[THREAD_STARTING];
    mi::Uint32 idle      = m_thread_state_counter[THREAD_IDLE];
    mi::Uint32 sleeping  = m_thread_state_counter[THREAD_SLEEPING];
    mi::Uint32 running   = m_thread_state_counter[THREAD_RUNNING];
    mi::Uint32 suspended = m_thread_state_counter[THREAD_SUSPENDED];
    mi::Uint32 shutdown  = m_thread_state_counter[THREAD_SHUTDOWN];
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Worker thread states: "
        "%u starting, %u sleeping, %u idle, %u running, %u suspended, %u shutdown, %u total",
        starting,  sleeping,  idle,  running,  suspended,  shutdown,
        starting + sleeping + idle + running + suspended + shutdown);
}

void Thread_pool::submit_job_internal( IJob* job, bool log_asynchronous)
{
    mi::base::Lock::Block block( &m_lock);

    // Submitting new jobs while another thread invokes the destructor is an error.
    ASSERT( M_THREAD_POOL, !m_shutdown);
    if( m_shutdown)
        return;

    // Put top-level jobs at the end of the job queue, put child jobs and resume jobs at the
    // beginning of the queue.
    mi::Uint64 thread_id = THREAD::Thread_id().get_uint();
    bool resume_job = m_running_jobs.find( thread_id) != m_running_jobs.end();
    bool child_job  = m_suspended_jobs.find( thread_id) != m_suspended_jobs.end();

    mi::Sint8 priority = job->get_priority();

    if( child_job || resume_job)
        m_job_queue[priority].push_front( make_handle_dup( job));
    else
        m_job_queue[priority].push_back( make_handle_dup( job));

    // Check whether the job could be executed immediately.
    mi::Float32 requested_cpu_load = job->get_cpu_load();
    mi::Float32 requested_gpu_load = job->get_gpu_load();
    adjust_load( requested_cpu_load, requested_gpu_load);
    if( !job_fits_load_limits( requested_cpu_load, requested_gpu_load)) {
#ifdef MI_THREAD_POOL_VERBOSE
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "Submitted job %p, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d, "
            "%s%sjob, execution delayed\n", job,
            requested_cpu_load, m_current_cpu_load, m_cpu_load_limit,
            requested_gpu_load, m_current_gpu_load, m_gpu_load_limit,
            (int) priority,
            resume_job ? "resume " : (child_job ? "child " : "top-level "),
            resume_job ? "" : (log_asynchronous ? "asynchronous " : "synchronous "));
#endif // MI_THREAD_POOL_VERBOSE
        return;
    }

    // Wake up some worker thread to process some job (not necessarily the one just submitted).
#ifdef MI_THREAD_POOL_VERBOSE
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitted job %p, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d, "
        "%s%sjob, waking up thread\n", job,
        requested_cpu_load, m_current_cpu_load, m_cpu_load_limit,
        requested_gpu_load, m_current_gpu_load, m_gpu_load_limit,
        (int) priority,
        resume_job ? "resume " : (child_job ? "child " : "top-level "),
        resume_job ? "" : (log_asynchronous ? "asynchronous " : "synchronous "));
#endif // MI_THREAD_POOL_VERBOSE
    wake_up_worker_thread();
}

bool Thread_pool::suspend_current_job_internal( bool only_for_higher_priority)
{
    mi::base::Lock::Block block( &m_lock);

    // check whether we actually suspend if the flag is set, part 1
    if( only_for_higher_priority && m_job_queue.empty())
        return false;

    // get job and thread properties (and check whether this a running worker thread at all)
    mi::Float32 cpu_load;
    mi::Float32 gpu_load;
    mi::Sint8 priority;
    mi::Uint64 thread_id;
    bool running_worker_thread
        = get_current_job_data_locked( cpu_load, gpu_load, priority, thread_id, false);
    if( !running_worker_thread) {
#ifdef ENABLE_ASSERT
        // detect nested suspend calls
        bool suspended_worker_thread
            = get_current_job_data_locked( cpu_load, gpu_load, priority, thread_id, true);
        ASSERT( M_THREAD_POOL, !suspended_worker_thread);
#endif // ENABLE_ASSERT
        return false;
    }

    // check whether we actually suspend if the flag is set, part 2
    if( only_for_higher_priority) {
        Job_queue::iterator it = m_job_queue.begin();
        if( it->first >= priority)
            return false;
    }

    --m_thread_state_counter[THREAD_RUNNING];
    ++m_thread_state_counter[THREAD_SUSPENDED];

    // adjust current load
    m_current_cpu_load -= cpu_load;
    m_current_gpu_load -= gpu_load;

    // move job from map of running threads to map of suspended threads
    Job_map::iterator it = m_running_jobs.find( thread_id);
    ASSERT( M_THREAD_POOL, it != m_running_jobs.end());
    ASSERT( M_THREAD_POOL, m_suspended_jobs.find( thread_id) == m_suspended_jobs.end());
    m_suspended_jobs[thread_id] = it->second; //-V783 PVS
    m_running_jobs.erase( it);

    // wake up some worker thread if there are jobs in the queue
    if( !m_job_queue.empty())
        wake_up_worker_thread();

    return true;
}

void Thread_pool::resume_current_job_internal()
{
    // get job and thread properties (and check whether this a suspended worker thread at all)
    mi::Float32 cpu_load;
    mi::Float32 gpu_load;
    mi::Sint8 priority;
    mi::Uint64 thread_id;
    bool suspended_worker_thread
        = get_current_job_data( cpu_load, gpu_load, priority, thread_id, true);
    if( !suspended_worker_thread) {
#ifdef ENABLE_ASSERT
        // detect nested resume calls
        bool running_worker_thread
            = get_current_job_data( cpu_load, gpu_load, priority, thread_id, false);
        ASSERT( M_THREAD_POOL, !running_worker_thread);
#endif // ENABLE_ASSERT
        return;
    }

    // wait for the required resources
    mi::base::Handle<Resume_job> resume_job( new Resume_job( cpu_load, gpu_load, priority));
    submit_job_internal( resume_job.get(), /*log_asynchronous*/ false);
    resume_job->wait();

    // No need to adjust the current load: when resume_job is finished the current load is \em not
    // decreased (the whole purpose of resume_job is to grab the required resources for the job to
    // be resumed via the regular job queue).

    --m_thread_state_counter[THREAD_SUSPENDED];
    ++m_thread_state_counter[THREAD_RUNNING];

    // move job from map of suspended threads to map of running threads
    mi::base::Lock::Block block( &m_lock);
    Job_map::iterator it = m_suspended_jobs.find( thread_id);
    ASSERT( M_THREAD_POOL, it != m_suspended_jobs.end());
    ASSERT( M_THREAD_POOL, m_running_jobs.find( thread_id) == m_running_jobs.end());
    m_running_jobs[thread_id] = it->second; //-V783 PVS
    m_suspended_jobs.erase( it);
}

void Thread_pool::create_worker_thread()
{
    // Attempts to create a worker thread while another thread invokes the destructor is an error.
    // (The destructor uses m_all_threads without lock.)
    ASSERT( M_THREAD_POOL, !m_shutdown);
    if( m_shutdown)
        return;

    // The caller is supposed to hold m_lock.
    Worker_thread* thread = new Worker_thread( this, m_next_cpu_id);
    m_next_cpu_id = (m_next_cpu_id+1) % THREAD::Thread::get_nr_of_cpus();
    thread->set_thread_affinity_enabled( m_thread_affinity);
    thread->start();
    m_all_threads.push_back( thread);
    ASSERT( M_THREAD_POOL, thread->get_state() == THREAD_SLEEPING);
    m_sleeping_threads.insert( thread);
}

void Thread_pool::wake_up_worker_thread()
{
    // The caller is supposed to hold m_lock.
    bool m_sleeping_threads_was_empty = m_sleeping_threads.empty();
    if( m_sleeping_threads_was_empty)
        create_worker_thread();
    Worker_thread* thread = * m_sleeping_threads.begin();
    ASSERT( M_THREAD_POOL, thread->get_state() == THREAD_SLEEPING);
    thread->wake_up();

    // remove thread from the set of sleeping threads
    size_t result = m_sleeping_threads.erase( thread);
    ASSERT( M_THREAD_POOL, result == 1);
    boost::ignore_unused( result);
}

bool Thread_pool::job_fits_load_limits( mi::Float32 cpu_load, mi::Float32 gpu_load) const
{
    // The caller is supposed to hold m_lock.

    // Clip requested resources against limits to avoid delaying forever jobs with unsatisfiable
    // requirements. Such jobs will only be executed if the current load is 0.0, ignoring the limit.
    if( cpu_load > m_cpu_load_limit) cpu_load = m_cpu_load_limit;
    if( gpu_load > m_gpu_load_limit) gpu_load = m_gpu_load_limit;

    return m_current_cpu_load + cpu_load <= m_cpu_load_limit * 1.001
        && m_current_gpu_load + gpu_load <= m_gpu_load_limit * 1.001;
}

bool Thread_pool::get_current_job_data(
    mi::Float32& cpu_load,
    mi::Float32& gpu_load,
    mi::Sint8& priority,
    mi::Uint64& thread_id,
    bool suspended) const
{
    mi::base::Lock::Block block( &m_lock);
    return get_current_job_data_locked( cpu_load, gpu_load, priority, thread_id, suspended);
}

bool Thread_pool::get_current_job_data_locked(
    mi::Float32& cpu_load,
    mi::Float32& gpu_load,
    mi::Sint8& priority,
    mi::Uint64& thread_id,
    bool suspended) const
{
    // The caller is supposed to hold m_lock.
    thread_id = THREAD::Thread_id().get_uint();
    Job_map::const_iterator it;
    if( suspended) {
        it = m_suspended_jobs.find( thread_id);
        if( it == m_suspended_jobs.end()) {
            cpu_load = 0.0;
            gpu_load = 0.0;
            priority = 0;
            return false;
        }
    } else {
        it = m_running_jobs.find( thread_id);
        if( it == m_running_jobs.end()) {
            cpu_load = 0.0;
            gpu_load = 0.0;
            priority = 0;
            return false;
        }
    }

    cpu_load = it->second->get_cpu_load();
    gpu_load = it->second->get_gpu_load();
    adjust_load( cpu_load, gpu_load);
    priority = it->second->get_priority();
    return true;
}

bool Thread_pool::generate_admin_http_server_page( HTTP::Connection* connection)
{
    ASSERT( M_THREAD_POOL, !"Not implemented");
    return false;
}

void Thread_pool::adjust_load( mi::Float32& cpu_load, mi::Float32& gpu_load)
{
    cpu_load = cpu_load == 0.0f ? 0.0f : std::max( cpu_load, s_min_cpu_load);
    gpu_load = gpu_load == 0.0f ? 0.0f : std::max( gpu_load, s_min_gpu_load);
    if( cpu_load == 0.0f && gpu_load == 0.0f)
        cpu_load = s_min_cpu_load;
}

} // namespace THREAD_POOL

} // namespace MI

