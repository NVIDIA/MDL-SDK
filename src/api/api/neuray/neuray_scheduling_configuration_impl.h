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

#ifndef API_API_NEURAY_NEURAY_SCHEDULING_CONFIGURATION_IMPL_H
#define API_API_NEURAY_NEURAY_SCHEDULING_CONFIGURATION_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/ischeduling_configuration.h>

#include <boost/core/noncopyable.hpp>

namespace MI {

namespace SCHED { class IScheduler; }
namespace THREAD_POOL { class Thread_pool; }


namespace NEURAY {

class Scheduling_configuration_impl
  : public mi::base::Interface_implement<mi::neuraylib::IScheduling_configuration>,
    public boost::noncopyable
{
public:
    /// Constructor of Scheduling_configuration_impl
    ///
    /// \param status           The status of the interface this API component belongs to.
    Scheduling_configuration_impl(mi::neuraylib::INeuray::Status& status);

    /// Destructor of Scheduling_configuration_impl
    ~Scheduling_configuration_impl();

    // public API methods

    mi::Sint32 set_cpu_load_limit( mi::Float32 limit);

    mi::Float32 get_cpu_load_limit() const;

    mi::Sint32 set_gpu_load_limit( mi::Float32 limit);

    mi::Float32 get_gpu_load_limit() const;

    mi::Sint32 set_thread_affinity_enabled( bool value);

    bool get_thread_affinity_enabled() const;

    mi::Sint32 set_accept_delegations( bool value);

    bool get_accept_delegations() const;

    mi::Sint32 set_work_delegation_enabled( bool value);

    bool get_work_delegation_enabled() const;

    mi::Sint32 set_gpu_work_delegation_enabled( bool value);

    bool get_gpu_work_delegation_enabled() const;

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \param thread_pool   The thread pool used by this API component.
    /// \param scheduler     The scheduler to be used by this API component. Might be \c nullptr.
    /// \return              0, in case of success, -1 in case of failure.
    mi::Sint32 start( THREAD_POOL::Thread_pool* thread_pool, SCHED::IScheduler* scheduler);

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:
    /// The status of the interface this API component belongs to (INeuray/ICluster).
    mi::neuraylib::INeuray::Status& m_status;

    /// The thread pool to be used by this API component.
    THREAD_POOL::Thread_pool* m_thread_pool = nullptr;
    /// Configured CPU Load limit.
    mi::Float32 m_cpu_load_limit = 0.0f;
    /// Configured GPU Load limit.
    mi::Float32 m_gpu_load_limit = 0.0f;

};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_SCHEDULING_CONFIGURATION_IMPL_H

