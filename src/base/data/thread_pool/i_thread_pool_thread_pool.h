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
 ** \brief Header for Thread_pool.
 **/

#ifndef BASE_DATA_THREAD_POOL_I_THREAD_POOL_THREAD_POOL_H
#define BASE_DATA_THREAD_POOL_I_THREAD_POOL_THREAD_POOL_H

#include <atomic>
#include <list>
#include <set>
#include <vector>
#include <map>

#include <mi/base/handle.h>
#include <mi/base/lock.h>

#include <boost/core/noncopyable.hpp>

#include "thread_pool_worker_thread.h" // for Thread_state

namespace MI {

namespace HTTP { class Connection; class Server; }

namespace THREAD_POOL {

class IJob;

/// The thread pool can be used to execute work synchronously or asynchronously in other threads.
///
/// Work is represented by instances of the IJob class. The thread pool manages a queue of jobs to
/// be executed in a FIFO order. It also manages a set of worker threads which continuously pick
/// jobs from the job queue and execute them, or sleep if there is currently no job to be executed.
///
/// The maximum amount of parallelism is controlled via load limits for the thread pool. Each job
/// announces the load it causes per worker thread (typically 1.0). The thread pool executes a job
/// only if the addition of its load does not exceed the configured load limit. Loads and load
/// limits exist separately for CPU and GPU, even though the GPU load and GPU load limit is hardly
/// used. This mechanism allows indirectly to control the number of worker threads.
///
/// The thread pool supports jobs that accept multiple worker threads (so-called fragmented jobs).
/// Worker threads will be assigned to the first job in the queue as long as this job indicates that
/// it accepts more worker threads via #IJob::is_remaining_work_splittable() (and the load limit is
/// not yet reached). If a job does not accept more worker threads, it is removed from the queue and
/// the next job is considered. In other words, the thread pool attempts to minimize the runtime of
/// jobs by assigning as many worker threads as possible, instead of running as many jobs as
/// possible in parallel.
///
/// The thread pool also supports nested jobs, i.e., a parent job submitting new jobs, so-called
/// child jobs. This requires some care if the parent job waits for the completion of the child
/// jobs.
///
/// Last but not least the thread pool supports job priorities which can be used to influence the
/// position of the job in the queue.
class Thread_pool : public boost::noncopyable
{
public:
    /// Constructor
    ///
    /// Creates the given number of worker threads and starts them.
    ///
    /// \param cpu_load_limit         The initial limit for the CPU load.
    /// \param gpu_load_limit         The initial limit for the GPU load.
    /// \param nr_of_worker_threads   The number of worker threads to create upfront. Note that the
    ///                               thread pool might start additional worker threads as needed.
    Thread_pool(
        mi::Float32 cpu_load_limit = 1.0,
        mi::Float32 gpu_load_limit = 1.0,
        mi::Size nr_of_worker_threads = 1);

    /// Destructor
    ///
    /// Shuts down the worker threads and destroys them.
    ~Thread_pool();

    /// Install the admin HTTP server page on \p server.
    void install_admin_http_server_page( HTTP::Server* server, const char* uri);

    /// \name Load limits and thread affinity
    //@{

    /// Sets the CPU load limit.
    ///
    /// If it is reduced it might take some time until the current CPU load obeys the limit.
    /// The value is clamped against the CPU load license limit.
    ///
    /// Returns \c true in case of success (iff \c limit is greater to or equal to 1.0).
    bool set_cpu_load_limit( mi::Float32 limit);

    /// Sets the GPU load limit.
    ///
    /// If it is reduced it might take some time until the current GPU load obeys the limit.
    /// The value is clamped against the GPU load license limit.
    ///
    /// Returns \c true in case of success (iff \c limit is greater to or equal to 1.0).
    bool set_gpu_load_limit( mi::Float32 limit);

    /// Returns the CPU load limit.
    mi::Float32 get_cpu_load_limit()
    {
        mi::base::Lock::Block block( &m_lock);
        return m_cpu_load_limit;
    }

    /// Returns the GPU load limit.
    mi::Float32 get_gpu_load_limit()
    {
        mi::base::Lock::Block block( &m_lock);
        return m_gpu_load_limit;
    }

    /// Sets an upper limit for the CPU load limit based on license restrictions.
    ///
    /// The default is unlimited. Adjusts the current CPU load limit if needed.
    bool set_cpu_load_license_limit( mi::Float32 license_limit);

    /// Sets an upper limit for the GPU load limit based on license restrictions.
    ///
    /// The default is unlimited. Adjusts the current GPU load limit if needed.
    bool set_gpu_load_license_limit( mi::Float32 license_limit);

    /// Returns the CPU load license limit.
    mi::Float32 get_cpu_load_license_limit()
    {
        mi::base::Lock::Block block( &m_lock);
        return m_cpu_load_license_limit;
    }

    /// Returns the GPU load license limit.
    mi::Float32 get_gpu_load_license_limit()
    {
        mi::base::Lock::Block block( &m_lock);
        return m_gpu_load_license_limit;
    }

    /// Enables or disables the thread affinity setting.
    ///
    /// \param value   The new thread affinity setting.
    /// \return        \c true in case of success, \c false otherwise.
    bool set_thread_affinity_enabled( bool value);

    /// Returns the current thread affinity setting.
    bool get_thread_affinity_enabled() const;

    //@}
    /// \name Jobs
    //@{

    /// Submits a job for asynchronous execution.
    ///
    /// The job is executed asynchronously and the method returns immediately. Jobs may not be
    /// submitted from suspended worker threads.
    ///
    /// If the submitted job is a child job, i.e., the job is submitted from a job currently being
    /// executed (the parent job), and the parent job later waits for the completion of the child
    /// job, special care needs to taken to avoid dead locks, see #suspend_current_job() and
    /// #resume_current_job() for details. If the parent job does nothing else except waiting for
    /// completion of the child job, consider using #submit_job_and_wait() which is less
    /// error-prone.
    ///
    /// \see #submit_job_and_wait() for synchronous execution
    void submit_job( IJob* job);

    /// Submits a job for synchronous execution.
    ///
    /// The job is executed synchronously, i.e., the method blocks until the job has been executed.
    /// Jobs may not be submitted from suspended worker threads.
    ///
    /// \see #submit() for asynchronous execution
    void submit_job_and_wait( IJob* job);

    /// Removes a job from the job queue and indicates success/failure.
    ///
    /// \note Removing an already submitted job should not be done imprudently. This method is
    ///       expensive in the sense that it needs to search the entire job queue to find the job.
    ///       In addition, nothing is known about the execution status (see return value), and
    ///       currently running jobs are not interrupted.
    ///
    /// \return Indicates whether the job was found in the queue (and therefore, removed from the
    ///         queue). The return value \true does \em not indicate that the job has not already
    ///         been started. The return value \false does \em not indicate that the job has
    ///         already been finished.
    bool remove_job( IJob* job);

    /// Notifies the thread pool that this worker thread suspends job execution (because it is going
    /// to wait for some event).
    ///
    /// The thread pool does not do anything except that it decreases the current load values
    /// accordingly. The method returns immediately if not being called from a worker thread.
    ///
    /// Usage of this method is mandatory if a child job is executed asynchronously from within
    /// a parent job, and the parent job waits for the child job to complete. Usage of this method
    /// is strongly recommended (but not mandatory) if the parent jobs waits for some event
    /// unrelated to child jobs. Usage of this method is not necessary for synchronous execution
    /// of child jobs.
    ///
    /// Failure to use this method when mandatory might lead to dead locks. Failure to use this
    /// method when recommended might lead to reduced performance.
    ///
    /// Example:
    /// \code
    /// mi::base::Condition condition;
    /// ...
    /// thread_pool->suspend_current_job();
    /// condition.wait();
    /// thread_pool->resume_current_job();
    /// \endcode
    ///
    /// This method needs to be used in conjunction with #resume_current_job().
    void suspend_current_job();

    /// Notifies the thread pool that this worker thread resumes job execution (because it waited
    /// for some event that now happened).
    ///
    /// The thread pool does not do anything except that it increases the current load values
    /// accordingly. This method blocks if the current load values and limits do not permit instant
    /// resuming of the job. The method returns immediately if not being called from a worker
    /// thread.
    ///
    /// \see #suspend_current_job() for further details
    void resume_current_job();

    /// Notifies the thread pool that this worker thread is willing to give up its resources for a
    /// while in favor of other jobs.
    ///
    /// Yielding is similar to calling #suspend_current_job() followed by #resume_current_job(), but
    /// it takes job priorities into account and is more efficient if there is no job of higher
    /// priority in the job queue.
    void yield();

    //@}
    /// \name Methods to be used by worker threads only
    //@{

    /// Returns the next job to be executed by \p thread.
    ///
    /// Returns \c NULL if there is no job to be executed, or no job whose load requirements fit the
    /// gap between load limits and current load values.
    ///
    /// Otherwise, the job is returned and removed from the job queue, the current load values are
    /// increased according to the job's requirements, and the worker thread is removed from the set
    /// of sleeping worker threads.
    IJob* get_next_job( Worker_thread* thread);

    /// Notifies the thread pool that execution of a job has finished.
    ///
    /// The current load values are decreased according to the job's requirements, and the worker
    /// thread is added again to the set of sleeping worker threads.
    void job_execution_finished( Worker_thread* thread, IJob* job);

    /// Increases the thread state counter for \p state.
    void increase_thread_state_counter( Thread_state state) { ++m_thread_state_counter[state]; }

    /// Decreases the thread state counter for \p state.
    void decrease_thread_state_counter( Thread_state state) { --m_thread_state_counter[state]; }

    //@}
    /// \name Methods to be used by the admin HTTP server/LOG output only
    //@{

    /// Returns the number of worker threads.
    ///
    /// \note This value is for informational purposes only. It is not meaningful without holding
    ///       the corresponding lock. Do not base any scheduling decisions on this value.
    mi::Uint32 get_number_of_worker_threads() const
    {
        mi::base::Lock::Block block( &m_lock);
        return m_all_threads.size();
    }

    /// Returns a particular thread state counter.
    ///
    /// \note These values are for informational purposes only. Modifications are atomic, but not
    ///       locked. Temporarily the values do \em not add up to the correct value. Do not base
    ///       any scheduling decisions on these values.
    mi::Uint32 get_thread_state_counter( Thread_state state) const
    {
        return m_thread_state_counter[state];
    }

    /// Returns the current CPU load.
    mi::Float32 get_current_cpu_load() const
    {
        mi::base::Lock::Block block( &m_lock);
        return m_current_cpu_load;
    }

    /// Returns the current GPU load.
    mi::Float32 get_current_gpu_load() const
    {
        mi::base::Lock::Block block( &m_lock);
        return m_current_gpu_load;
    }

    /// Dumps the current CPU/GPU load/limits to the log (category MISC, severity INFO).
    void dump_load() const;

    /// Dumps the thread state counters to the log (category MISC, severity INFO).
    void dump_thread_state_counters() const;

    //@}

private:
    /// Submits a job, i.e., puts it into the job queue.
    ///
    /// The job is executed asynchronously and the method returns immediately. Used by #submit_job()
    /// and #submit_job_and_wait() to do the actual work.
    ///
    /// \param job                The job to be executed.
    /// \param log_asynchronous   Indicates whether the jobs is executed synchronously or
    ///                           asynchronously. This value is passed for log output only!
    void submit_job_internal( IJob* job, bool log_asynchronous);

    /// Notifies the thread pool that this worker thread suspends job execution (because it is going
    /// to wait for some event).
    ///
    /// See #suspend_current_job() for details. The additional flag and return value is used by
    /// #yield().
    ///
    /// \param only_for_higher_priority
    ///                     Indicates whether execution should be suspended only if there is job of
    ///                     higher priority in the queue. #suspend_current_job() calls this method
    ///                     passing \c false; #yield() calls this method passing \c true.
    /// \return             Indicates whether the job execution was actually suspended (might return
    ///                     \c false if \p only_for_higher_priority was set and there was no such
    ///                     job, or if job execution for this thread was already suspended).
    bool suspend_current_job_internal( bool only_for_higher_priority);

    /// Notifies the thread pool that this worker thread resumes job execution (because it waited
    /// for some event that now happened).
    ///
    /// See #resume_current_job() for details.
    void resume_current_job_internal();

    /// Creates a new worker thread and adds it to m_all_threads and m_sleeping_threads.
    ///
    /// The callers needs to hold m_lock.
    void create_worker_thread();

    /// Wakes up a sleeping worker thread.
    ///
    /// If there are no sleeping worker threads, a new thread will be created.
    ///
    /// The callers needs to hold m_lock.
    void wake_up_worker_thread();

    /// Indicates whether a job with given CPU/GPU loads can be executed given the current CPU/GPU
    /// load and the CPU/GPU load limits.
    ///
    /// The callers needs to hold m_lock.
    bool job_fits_load_limits( mi::Float32 cpu_load, mi::Float32 gpu_load) const;

    /// Returns some job data for the job assigned to a running or suspended worker thread.
    ///
    /// Acquires m_lock and calls #get_current_job_data_locked().
    bool get_current_job_data(
        mi::Float32& cpu_load,
        mi::Float32& gpu_load,
        mi::Sint8& priority,
        mi::Uint64& thread_id,
        bool suspended) const;

    /// Returns some job data for the job assigned to a running or suspended worker thread.
    ///
    /// The callers needs to hold m_lock.
    ///
    /// \param[out] cpu_load   The CPU load of the job, or 0.0 if the calling thread is not a
    ///                        running/suspended worker thread.
    /// \param[out] gpu_load   The GPU load of the job, or 0.0 if the calling thread is not a
    ///                        running/suspended worker thread.
    /// \param[out] priority   The priority of the job.
    /// \param[out] thread_id  The thread ID.
    /// \param suspended       Indicates whether thread is supposed to be running or suspended.
    /// \return                \true if the calling thread is a running/suspended worker thread,
    ///                        and \false otherwise
    bool get_current_job_data_locked(
        mi::Float32& cpu_load,
        mi::Float32& gpu_load,
        mi::Sint8& priority,
        mi::Uint64& thread_id,
        bool suspended) const;

    /// Generates the admin HTTP server page for the thread pool.
    bool generate_admin_http_server_page( HTTP::Connection* connection);

    /// Clips the values against #s_min_cpu_load and #s_min_gpu_load.
    ///
    /// This is done to protect against malicious job implementations that return negative values
    /// which would cause a huge number of threads to be spawned.
    static void adjust_load( mi::Float32& cpu_load, mi::Float32& gpu_load);

    /// The configured CPU load limit.
    mi::Float32 m_cpu_load_limit;
    /// The configured GPU load limit.
    mi::Float32 m_gpu_load_limit;
    /// The configured CPU load license limit.
    mi::Float32 m_cpu_load_license_limit;
    /// The configured GPU load license_limit.
    mi::Float32 m_gpu_load_license_limit;
    /// The current CPU load. Protected by m_lock.
    mi::Float32 m_current_cpu_load;
    /// The current GPU load. Protected by m_lock.
    mi::Float32 m_current_gpu_load;
    /// The smallest CPU load a job can cause.
    static mi::Float32 s_min_cpu_load;
    /// The smallest GPU load a job can cause.
    static mi::Float32 s_min_gpu_load;
    /// Indicates whether thread affinity is enabled (cached here for new threads).
    bool m_thread_affinity;

    /// The type of the vector of all worker threads.
    typedef std::vector<Worker_thread*> All_threads;
    /// The vector of all worker threads. Protected by m_lock.
    All_threads m_all_threads;

    /// The type of the set of sleeping worker threads.
    typedef std::set<Worker_thread*> Sleeping_threads;
    /// The set of sleeping worker threads. Protected by m_lock.
    Sleeping_threads m_sleeping_threads;

    /// The type of the job list. One job list is used for each priority.
    typedef std::list<mi::base::Handle<IJob> > Job_list;
    /// The type of the job queue. Maps priorities to job lists.
    typedef std::map<mi::Sint32, Job_list> Job_queue;
    /// The job queue. Protected by m_lock.
    ///
    /// Actually, this is a map of lists instead of a queue such that we can efficiently insert and
    /// remove jobs at the required position (insertion at the front and back per priority, removal
    /// at any position with given iterator).
    ///
    /// Invariant: there is no empty list as map element (otherwise we cannot efficiently check
    /// whether the job queue is empty).
    Job_queue m_job_queue;

    /// The type of the maps below.
    typedef std::map<mi::Uint64, IJob*> Job_map;
    /// The map that holds for each running worker thread the job that it is executing.
    /// Protected by m_lock.
    Job_map m_running_jobs;
    /// The map that holds for each suspended worker thread the job that it is executing.
    /// Protected by m_lock.
    Job_map m_suspended_jobs;

    /// The lock that protects the current loads, the vector of all worker threads, the set of
    /// sleeping worker threads, the job queue, and the maps for running and suspended jobs.
    mutable mi::base::Lock m_lock;

    /// The thread state counters.
    ///
    /// These values are updated by the worker threads themselves, not by the thread pool.
    ///
    /// \note These values are for informational purposes only. Modifications are atomic, but not
    ///       locked. Temporarily the values do \em not add up to the correct value. Do not base
    ///       any scheduling decisions on these values.
    std::atomic_uint32_t m_thread_state_counter[N_THREAD_STATES];

    /// The CPU ID that is passed to next created worker thread.
    mi::Uint32 m_next_cpu_id;

    /// Used by the destructor to block submitting of new jobs.
    bool m_shutdown;
};

} // namespace THREAD_POOL

} // namespace MI

#endif // BASE_DATA_THREAD_POOL_I_THREAD_POOL_THREAD_POOL_H
