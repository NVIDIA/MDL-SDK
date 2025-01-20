/***************************************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_FRAGMENTED_JOB_H
#define BASE_DATA_DBLIGHT_DBLIGHT_FRAGMENTED_JOB_H

#include <base/data/db/i_db_fragmented_job.h>
#include <base/data/thread_pool/i_thread_pool_ijob.h>

#include <boost/intrusive/list.hpp>

namespace MI {

namespace DB { class Fragmented_job; class IExecution_listener; class Transaction; }

namespace DBLIGHT {

namespace bi = boost::intrusive;

/// Adapts DB::Fragmented_job to THREAD_POOL::Fragmented_job.
class Fragmented_job : public THREAD_POOL::Fragmented_job
{
public:
    /// Constructor.
    ///
    /// Note that \p transaction and \p listener might be \c nullptr.
    Fragmented_job(
        DB::Transaction* transaction,
        size_t count,
        DB::Fragmented_job* job,
        DB::IExecution_listener* listener)
      : THREAD_POOL::Fragmented_job( transaction, count),
        m_transaction( transaction),
        m_fragmented_job( job),
        m_listener( listener),
        m_cpu_load( job->get_cpu_load()),
        m_gpu_load( job->get_gpu_load()),
        m_priority( job->get_priority()),
        m_thread_limit( job->get_thread_limit())
    {
    }

    // methods of THREAD_POOL::Fragmented_job

    mi::Float32 get_cpu_load() const override { return m_cpu_load; }

    mi::Float32 get_gpu_load() const override { return m_gpu_load; }

    mi::Sint8 get_priority() const override { return m_priority; }

    size_t get_thread_limit() const override { return m_thread_limit; }

    void execute_fragment(
        DB::Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context) override
    {
        m_fragmented_job->execute_fragment( transaction, index, count, context);
    }

    void job_finished() override;

    /// Expose cancel() of the wrapped job.
    void cancel() { m_fragmented_job->cancel(); }

    /// Hook for Transaction_impl::m_fragmented_jobs.
    bi::list_member_hook<> m_fragmented_jobs_hook;

private:
    DB::Transaction* m_transaction;
    DB::Fragmented_job* m_fragmented_job;
    DB::IExecution_listener* m_listener;
    float m_cpu_load;
    float m_gpu_load;
    mi::Sint8 m_priority;
    size_t m_thread_limit;
};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_FRAGMENTED_JOB_H
