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
/// \file
/// \brief API component for scheduling related settings.

#ifndef MI_NEURAYLIB_ISCHEDULING_CONFIGURATION_H
#define MI_NEURAYLIB_ISCHEDULING_CONFIGURATION_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_configuration
@{
*/

/// This interface is used to query and change the scheduling configuration.
class IScheduling_configuration : public
    mi::base::Interface_declare<0x4f1fe336,0x111a,0x44c3,0xb2,0x95,0xa3,0x30,0xf0,0xb6,0xc2,0x05>
{
public:
    /// Sets the CPU load limit.
    ///
    /// The CPU load is the sum of the CPU loads caused by all currently active threads. A single
    /// thread fully using a single CPU core is said to cause a CPU load of 1.0. The CPU load limit
    /// defaults to the number of CPU cores.
    ///
    /// \note If the limit is reduced, it might take some time until the current CPU load obeys the
    ///       limit.
    ///
    /// \see #get_cpu_load_limit()
    ///
    /// \param limit   The new CPU load limit. \ifnot MDL_SDK_API The value might be clamped
    ///                against some upper bound imposed by license restrictions.\endif
    /// \return
    ///                -  0: Success
    ///                - -1: Invalid new limit (less than 1.0).
    virtual Sint32 set_cpu_load_limit( Float32 limit) = 0;

    /// Returns the CPU load limit.
    ///
    /// \see #set_cpu_load_limit()
    virtual Float32 get_cpu_load_limit() const = 0;

    /// Sets the GPU load limit.
    ///
    /// The GPU load is the sum of the GPU loads caused by all currently active threads. A single
    /// thread fully using a single GPU is said to cause a GPU load of 1.0. The GPU load limit
    /// defaults to the number of GPUs.
    ///
    /// \note If the limit is reduced, it might take some time until the current GPU load obeys the
    ///       limit.
    ///
    /// \see #get_gpu_load_limit()
    ///
    /// \param limit   The new GPU load limit. \ifnot MDL_SDK_API The value might be clamped
    ///                against some upper bound imposed by license restrictions.\endif
    /// \return
    ///                -  0: Success
    ///                - -1: Invalid new limit (less than 1.0).
    virtual Sint32 set_gpu_load_limit( Float32 limit) = 0;

    /// Returns the GPU load limit.
    ///
    /// \see #set_gpu_load_limit()
    virtual Float32 get_gpu_load_limit() const = 0;

    /// Sets the CPU affinity for threads.
    ///
    /// If thread affinity is enabled for a thread then this thread is bound to the CPU it is
    /// currently running on. If thread affinity is disabled (the default) the operating system is
    /// free to migrate the thread between CPUs as it sees fit (which might suffer from cache
    /// misses).
    ///
    /// This thread affinity setting only affects internal threads that are used to execute jobs.
    /// Application threads are not affected by this setting.
    ///
    /// Note that changing this value does not affect a job that is currently being executed, it
    /// only affects subsequently started jobs (or fragments thereof).
    ///
    /// This method can only be used while \neurayProductName is running.
    ///
    /// \param value
    /// \return
    ///                -  0: Success.
    ///                - -1: Setting the thread affinity is not supported on this operating system
    ///                      (Linux with glibc older than 2.6 and MacOS X).
    ///                - -2: The method cannot be called at this point of time.
    virtual Sint32 set_thread_affinity_enabled( bool value) = 0;

    /// Returns the CPU affinity for threads.
    ///
    /// \see #set_thread_affinity_enabled()
    virtual bool get_thread_affinity_enabled() const = 0;

    /// \if MDL_SDK_API
    /// Not supported.
    /// \else
    /// Sets if the host accepts delegations from other hosts.
    ///
    /// This can be called at any time by any thread. Enforcing the limitation may take a short
    /// amount of time, though, because current delegations will still be handled.
    ///
    /// \see #get_accept_delegations()
    ///
    /// \param value                   \c true if delegations are accepted, \c false otherwise.
    /// \return                        0, in case of success, -1 in case of failure.
    /// \endif
    virtual Sint32 set_accept_delegations( bool value) = 0;

    /// \if MDL_SDK_API
    /// Not supported.
    /// \else
    /// Returns if delegations of work are currently accepted from other hosts.
    ///
    /// \see #set_accept_delegations()
    ///
    /// \return                        The currently configured value.
    /// \endif
    virtual bool get_accept_delegations() const = 0;

    /// \if MDL_SDK_API
    /// Not supported.
    /// \else
    /// Sets if the host will currently delegate work to other hosts.
    ///
    /// This can be called at any time by any thread. Enforcing the limitation may take a short
    /// amount of time, though, because current delegations will still be handled.
    ///
    /// \see #get_work_delegation_enabled()
    ///
    /// \param value                   \c true if work is delegated, \c false otherwise.
    /// \return                        0, in case of success, -1 in case of failure.
    /// \endif
    virtual Sint32 set_work_delegation_enabled( bool value) = 0;

    /// \if MDL_SDK_API
    /// Not supported.
    /// \else
    /// Returns if work is currently delegated to other hosts.
    ///
    /// \see #set_work_delegation_enabled()
    ///
    /// \return                        The currently configured value.
    /// \endif
    virtual bool get_work_delegation_enabled() const = 0;

    /// \if MDL_SDK_API
    /// Not supported.
    /// \else
    /// Sets if the host will currently delegate GPU work to other hosts.
    ///
    /// This can be called at any time by any thread. Enforcing the limitation may take a short
    /// amount of time, though, because current delegations will still be handled.
    ///
    /// Note that if the system does not delegate work at all this is also applied to GPU work.
    /// Thus, even if this is set to \c true, no GPU work will be delegated if work delegation as a
    /// whole is disabled.
    ///
    /// \see #get_gpu_work_delegation_enabled(),
    ///      #set_work_delegation_enabled(), #get_work_delegation_enabled()
    ///
    /// \param value                   \c true if work is delegated, \c false otherwise.
    /// \return                        0, in case of success, -1 in case of failure.
    /// \endif
    virtual Sint32 set_gpu_work_delegation_enabled( bool value) = 0;

    /// \if MDL_SDK_API
    /// Not supported.
    /// \else
    /// Returns if GPU work is currently delegated to other hosts.
    ///
    /// Note that this reflects the configured value but is not affected by disabling work
    /// delegation completely.
    ///
    /// \see #set_gpu_work_delegation_enabled(),
    ///      #set_work_delegation_enabled(), #get_work_delegation_enabled()
    ///
    /// \return                        The currently configured value.
    /// \endif
    virtual bool get_gpu_work_delegation_enabled() const = 0;
};

/**@}*/ // end group mi_neuray_configuration

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ISCHEDULING_CONFIGURATION_H
