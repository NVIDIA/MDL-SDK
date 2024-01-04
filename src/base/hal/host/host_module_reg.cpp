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
/// Host module impl.

#include "pch.h"

#include "i_host.h"

#include <cstdlib>
#include <base/system/main/module_registration.h>

namespace MI {

namespace HOST {



// The next three method are implemented in host.cpp

/// Returns the number of logical cpus (or 1 in case of failure).
int number_of_cpus();
/// Returns the number of CPUs per socket (siblings).
int sibs_per_cpu();
/// Returns the size of the physical memory in MiB (or 0 in case of failure).
int memory_size();
/// checks for AVX support based on cpuid info
bool has_avx_support(const int[4]);

#ifdef MI_ARCH_X86
/// Encapsulates cpuid for all compilers
void cpuid(int op, int* eax, int* ebx, int* ecx, int* edx);
#endif

// The implementation of the abstract interface Host_module
class Host_module_impl : public Host_module
{
public:
    bool init();

    void exit() { };

    int get_number_of_cpus() { return m_ncpus; }

    int get_sibs_per_cpu() { return m_nsibs; }

    int get_memory_size() { return m_memory_size; } // in MiB

    void get_cpuid(int cpuid_info[4])
    {
        cpuid_info[0] = m_cpuid_info[0];
        cpuid_info[1] = m_cpuid_info[1];
        cpuid_info[2] = m_cpuid_info[2];
        cpuid_info[3] = m_cpuid_info[3];
    }

    bool has_avx() const { return m_has_avx; }

private:
    int m_ncpus;
    int m_nsibs;
    int m_memory_size;
    bool m_has_avx = false;

    int m_cpuid_info[4];
};

static SYSTEM::Module_registration<Host_module_impl> s_module(M_HOST, "HOST");

bool Host_module_impl::init()
{
    m_ncpus = number_of_cpus();
    m_nsibs = sibs_per_cpu();
    m_memory_size = memory_size();
#ifdef MI_ARCH_X86
    cpuid(1,m_cpuid_info,m_cpuid_info+1,m_cpuid_info+2,m_cpuid_info+3);
#else
    m_cpuid_info[0] = 0;
    m_cpuid_info[1] = 0;
    m_cpuid_info[2] = 0;
    m_cpuid_info[3] = 0;
#endif

    m_has_avx = has_avx_support(m_cpuid_info);

    return true;
}

Module_registration_entry* Host_module::get_instance()
{
    return s_module.init_module(s_module.get_name());
}

} // namespace HOST

} // namespace MI
