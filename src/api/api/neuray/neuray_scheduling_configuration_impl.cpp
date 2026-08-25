/***************************************************************************************************
 * Copyright (c) 2008-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
 ** \brief Implementation of IScheduling_configuration
 **
 ** Implements the IScheduling_configuration interface
 **/

#include "pch.h"

#include "neuray_scheduling_configuration_impl.h"

#include <base/data/thread_pool/i_thread_pool_thread_pool.h>

namespace MI {

namespace NEURAY {

Scheduling_configuration_impl::Scheduling_configuration_impl(
    mi::neuraylib::INeuray::Status& status)
  : m_status( status)
{
}

Scheduling_configuration_impl::~Scheduling_configuration_impl() = default;

mi::Sint32 Scheduling_configuration_impl::set_cpu_load_limit( mi::Float32 limit)
{
    if( limit < 1.0f)
        return -1;

    m_cpu_load_limit = limit;
    if( m_status == mi::neuraylib::INeuray::STARTED)
        return m_thread_pool->set_cpu_load_limit( limit) ? 0 : -1;
    else
        return 0;
}

mi::Float32 Scheduling_configuration_impl::get_cpu_load_limit() const
{
    if( m_status == mi::neuraylib::INeuray::STARTED)
        return m_thread_pool->get_cpu_load_limit();
    else
        return m_cpu_load_limit <= 0.0f ? 1.0f : m_cpu_load_limit;
}

mi::Sint32 Scheduling_configuration_impl::set_gpu_load_limit( mi::Float32 limit)
{
    if( limit < 1.0f)
        return -1;

    m_gpu_load_limit = limit;
    if( m_status == mi::neuraylib::INeuray::STARTED)
        return m_thread_pool->set_gpu_load_limit( limit) ? 0 : -1;
    else
        return 0;
}

mi::Float32 Scheduling_configuration_impl::get_gpu_load_limit() const
{
    if( m_status == mi::neuraylib::INeuray::STARTED)
        return m_thread_pool->get_gpu_load_limit();
    else
        return m_gpu_load_limit <= 0.0f ? 1.0f : m_gpu_load_limit;
}

mi::Sint32 Scheduling_configuration_impl::set_thread_affinity_enabled( bool value)
{
    if( m_status != mi::neuraylib::INeuray::STARTED)
       return -2;

    return m_thread_pool->set_thread_affinity_enabled( value) ? 0 : -1;
}

bool Scheduling_configuration_impl::get_thread_affinity_enabled() const
{
    if( m_status != mi::neuraylib::INeuray::STARTED)
       return false;

    return m_thread_pool->get_thread_affinity_enabled();
}

mi::Sint32 Scheduling_configuration_impl::set_accept_delegations( bool value)
{
    return -1;
}

bool Scheduling_configuration_impl::get_accept_delegations() const
{
    return false;
}

mi::Sint32 Scheduling_configuration_impl::set_work_delegation_enabled( bool value)
{
    return -1;
}

bool Scheduling_configuration_impl::get_work_delegation_enabled() const
{
    return false;
}

mi::Sint32 Scheduling_configuration_impl::set_gpu_work_delegation_enabled( bool value)
{
    return -1;
}

bool Scheduling_configuration_impl::get_gpu_work_delegation_enabled() const
{
    return false;
}

mi::Sint32 Scheduling_configuration_impl::start(
    THREAD_POOL::Thread_pool* thread_pool, SCHED::IScheduler* scheduler)
{
    m_thread_pool = thread_pool;
    if( m_cpu_load_limit != 0.0f) m_thread_pool->set_cpu_load_limit( m_cpu_load_limit);
    if( m_gpu_load_limit != 0.0f) m_thread_pool->set_gpu_load_limit( m_gpu_load_limit);


    return 0;
}

mi::Sint32 Scheduling_configuration_impl::shutdown()
{
    m_thread_pool = nullptr;
    m_cpu_load_limit = 0.0f;
    m_gpu_load_limit = 0.0f;

    return 0;
}

} // namespace NEURAY

} // namespace MI

