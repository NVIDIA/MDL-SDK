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
 ** \brief IJob and Fragmented_job
 **/

#ifndef BASE_DATA_THREAD_POOL_I_THREAD_POOL_IJOB_H
#define BASE_DATA_THREAD_POOL_I_THREAD_POOL_IJOB_H

#include <atomic>

#include <mi/base/interface_declare.h>
#include <mi/base/interface_implement.h>
#include <boost/core/noncopyable.hpp>
#include <base/lib/log/i_log_assert.h>

namespace mi { namespace neuraylib { class IJob_execution_context; } }

namespace MI {

namespace DB { class Transaction; }

namespace THREAD_POOL {

/// The abstract interface for jobs processed by the thread pool.
///
/// Database jobs (SCHED::Job) and fragmented jobs (DB::Fragmented_jobs) are directly or indirectly
/// derived from this interface.
class IJob : public
    mi::base::Interface_declare<0x287beee3,0xc3f8,0x4c18,0x8b,0x85,0x40,0xa5,0xed,0x53,0x87,0x88>
{
public:
    /// Returns the CPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual mi::Float32 get_cpu_load() const = 0;

    /// Returns the GPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual mi::Float32 get_gpu_load() const = 0;

    /// Returns the priority of the job.
    ///
    /// The smaller the value the higher the priority of the job to be executed.
    ///
    /// \note Negative values are reserved for internal purposes (for the thread pool and the
    ///       fragment scheduler).
    virtual mi::Sint8 get_priority() const = 0;

    /// Notifies the job about an upcoming #execute() call from the same thread.
    ///
    /// The call to #pre_execute() is still done under the main lock of the thread pool. The state
    /// of the worker thread is still THREAD_IDLE. This method should not do any work except trivial
    /// and fast book-keeping.
    virtual void pre_execute( const mi::neuraylib::IJob_execution_context* context) = 0;

    /// Executes the job.
    virtual void execute( const mi::neuraylib::IJob_execution_context* context) = 0;

    /// Indicates whether the remaining work of the job can be split into multiple fragments.
    ///
    /// \note This method must either always returns \c false, or the value must change from \c true
    ///       to \c false (and not back) exactly once during the lifetime of a given instance.
    ///
    /// \note If this method ever returns \c true, then the #execute() method must handle parallel
    ///       invocations from different threads, even if (at that time) the work cannot be split
    ///       anymore into multiple fragments or if there is no more work to do at all.
    ///
    /// \note Note that returning \c true here is only a hint, i.e., the #execute() method must not
    ///       rely on further multiple invocations. The #execute() method must not return until all
    ///       work is done (unless it is guaranteed that the remaining concurrently ongoing calls to
    ///       #execute() will do all the work).
    virtual bool is_remaining_work_splittable() = 0;
};

/// This is the base class for fragmented jobs of the thread pool.
///
/// The base class implements #pre_execute(), #execute() and #is_remaining_work_splittable() from
/// the #IJob interface in terms of an additional virtual method #execute_fragment(). Derived
/// classes need to implement this new virtual method, in addition to #get_cpu_load(),
/// #get_gpu_load(), and #get_priority() from the base class.
///
/// Using this base class is not mandatory for fragmented jobs, but strongly recommended.
class Fragmented_job
  : public mi::base::Interface_implement<THREAD_POOL::IJob>, public boost::noncopyable
{
public:
    /// Constructor.
    ///
    /// \param transaction    The transaction that will be passed to #execute_fragment() or returned
    ///                       in #get_transaction().
    /// \param count          The number of fragments.
    Fragmented_job( DB::Transaction* transaction, size_t count)
      : m_transaction( transaction),
        m_count( count),
        m_next_fragment( 0),
        m_outstanding_fragments( count),
        m_threads( 0) { }

    /// Sets the transaction.
    ///
    /// Should \em not be called anymore after the first call to #execute() (this class does not
    /// care but derived classes might care).
    void set_transaction( DB::Transaction* transaction)
    {
        ASSERT( M_THREAD_POOL, m_next_fragment == 0);
        m_transaction = transaction;
    }

    /// Returns the transaction.
    DB::Transaction* get_transaction() const { return m_transaction; }

    /// Sets the fragment count (including outstanding fragments).
    ///
    /// Must \em not be called anymore after the first call to #execute().
    void set_count( size_t count)
    {
        ASSERT( M_THREAD_POOL, m_next_fragment == 0);
        m_count = m_outstanding_fragments = count;
    }

    /// Returns the fragment count.
    size_t get_count() const { return m_count; }

    /// Executes one fragment of the fragmented job.
    ///
    /// \param transaction   The transaction in which the fragment is executed.
    /// \param index         The index of the fragment.
    /// \param count         The total number of fragments.
    /// \param context       The execution context.
    virtual void execute_fragment(
        DB::Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context) = 0;

    /// Bounds the maximum number of worker threads for this job.
    ///
    /// Can be used to disable parallelization, e.g., for debugging.
    ///
    /// Unbounded in the default implementation. Can be overridden if desired.
    virtual size_t get_thread_limit() const { return 0; }

    /// Notification that all fragments have been executed.
    ///
    /// Empty in default implementation. Can be overridden if desired.
    virtual void job_finished() { }

    //@{ Default implementation of some methods of THREAD_POOL::IJob

    /// Do not change/override this implementation unless you really know what you are doing.
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) final
    {
        ++m_threads;
    }

    /// Implemented in terms of #execute_fragment().
    ///
    /// This method invokes #execute_fragment() in a loop with the stored transaction and fragment
    /// count, and passes the execution context along. Each loop iteration performs one call using
    /// a unique fragment index. The method is thread-safe, parallel calls will use distinct
    /// fragment indices.
    ///
    /// Do not change/override this implementation unless you really know what you are doing.
    void execute( const mi::neuraylib::IJob_execution_context* context) final
    {
        // The assertion should go to the constructor and set_count(), but DBNR's fragment scheduler
        // actually creates instances of Remove_active_job with count == 0, although it never
        // submits them.
        ASSERT( M_THREAD_POOL, m_count > 0);

        size_t index = m_next_fragment++;
        bool executed_last_fragment = false;
        while( index < m_count) {
            execute_fragment( m_transaction, index, m_count, context);
            executed_last_fragment = --m_outstanding_fragments == 0;
            index = m_next_fragment++;
        }
        if( executed_last_fragment)
            job_finished();
    }

    /// Do not change/override this implementation unless you really know what you are doing.
    bool is_remaining_work_splittable() final
    {
        size_t thread_limit = get_thread_limit();
        return (m_next_fragment+1 < m_count) && (thread_limit == 0 || m_threads < thread_limit);
    }

    /// Returns 0. Can be overridden if desired.
    mi::Sint8 get_priority() const override { return 0; }

    //@}

private:
    /// The transaction used by this fragmented job.
    DB::Transaction* m_transaction;
    /// The number of fragments of this fragmented job.
    size_t m_count;
    /// The next fragment that will be executed.
    std::atomic_uint32_t m_next_fragment;
    /// The number of fragments not yet completed.
    std::atomic_uint32_t m_outstanding_fragments;
    /// The number of threads in #execute().
    std::atomic_uint32_t m_threads;
};

} // namespace THREAD_POOL

} // namespace MI

#endif // BASE_DATA_THREAD_POOL_I_THREAD_POOL_IJOB_H
