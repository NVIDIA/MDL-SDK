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
 ** \brief Header and implementation for Condition_job and Resume_job.
 **/

#ifndef BASE_DATA_THREAD_POOL_THREAD_POOL_JOBS_H
#define BASE_DATA_THREAD_POOL_THREAD_POOL_JOBS_H

#include <mi/base/condition.h>
#include <mi/base/interface_declare.h>
#include <mi/base/interface_implement.h>
#include <mi/base/handle.h>

#include "i_thread_pool_ijob.h"

namespace MI {

namespace THREAD_POOL {

/// Wrapper for IJob that allows to wait for #execute() to finish.
///
/// The additional method #wait() waits for the signal sent at the end of #execute(). If the
/// wrapped jobs supports parallel calls to #execute() the signal is sent by the thread that
/// leaves #execute() last.
class Condition_job : public mi::base::Interface_implement<IJob>
{
public:
    Condition_job( IJob* job) : m_job( make_handle_dup( job)), m_parallel_invocations( 0) { }
    mi::Float32 get_cpu_load() const { return m_job->get_cpu_load(); }
    mi::Float32 get_gpu_load() const { return m_job->get_gpu_load(); }
    mi::Sint8 get_priority() const { return m_job->get_priority(); }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context)
    { m_job->pre_execute( context); }
    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        ++m_parallel_invocations;
        m_job->execute( context);
        if( --m_parallel_invocations == 0)
            m_condition.signal();
    }
    bool is_remaining_work_splittable() { return m_job->is_remaining_work_splittable(); }
    void wait() { m_condition.wait(); }

private:
    mi::base::Handle<IJob> m_job;
    mi::base::Condition m_condition;
    std::atomic_uint32_t m_parallel_invocations;
};

/// Interface for Resume_job. Exists to be able to recognize such jobs if only IJob is given.
class IResume_job : public
    mi::base::Interface_declare<0xafdb52bc,0xdd76,0x4a84,0xa4,0x7a,0xa3,0x76,0x64,0x21,0xbc,0x0a,
                                IJob>
{
};

/// A resume job is used to allocate the resources needed to resume a currently suspended job via
/// the regular job queue. This is achieved by a special case inside the thread pool specific to
/// this class.
class Resume_job : public mi::base::Interface_implement<IResume_job>
{
public:
    Resume_job( mi::Float32 cpu_load, mi::Float32 gpu_load, mi::Sint8 priority)
      : m_cpu_load( cpu_load), m_gpu_load( gpu_load), m_priority( priority) { }
    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return m_priority; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }
    void execute( const mi::neuraylib::IJob_execution_context* context) { m_condition.signal(); }
    bool is_remaining_work_splittable() { return false; }
    void wait() { m_condition.wait(); }

private:
    mi::base::Condition m_condition;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Sint8 m_priority;
};

} // namespace THREAD_POOL

} // namespace MI

#endif // BASE_DATA_THREAD_POOL_THREAD_POOL_JOBS_H
