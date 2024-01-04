/******************************************************************************
 * Copyright (c) 2008-2023, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/
/// \file
/// Abstract host class

#ifndef BASE_HAL_HOST_I_HOST_H
#define BASE_HAL_HOST_I_HOST_H

#include <base/system/main/i_module.h>
#include <base/lib/mem/i_mem_allocatable.h>

namespace MI {

namespace SYSTEM { class Module_registration_entry; }

namespace HOST {

/// Interface for the host module
class Host_module : public SYSTEM::IModule, public MEM::Allocatable
{
  public:
    /// Returns the number of CPUs (or 1 in case of failure)
    virtual int get_number_of_cpus() = 0;

    /// Returns the number of siblings per CPU.
    ///
    /// A sibling is a virtual CPU in a hyperthreaded CPU.
    virtual int get_sibs_per_cpu() = 0;

    /// Returns the size of the physical memory in MiB (or 0 in case of failure).
    virtual int get_memory_size() = 0;

    /// Returns the cpuid register info for mode 1 (e.g. processor info and feature bits).
    virtual void get_cpuid(int cpuid_info[4]) = 0;

    /// Checks whether this machine supports AVX
    virtual bool has_avx() const = 0;

    /// Required functionality for implementing a SYSTEM::IModule.
    //@{

    /// Retrieve the name.
    /// \return the module's name
    static const char* get_name() { return "HOST"; }

    /// Allow link time detection.
    /// \return the static module's pointer
    static SYSTEM::Module_registration_entry* get_instance();
    //@}
};

} // namespace HOST

} // namespace MI

#endif
