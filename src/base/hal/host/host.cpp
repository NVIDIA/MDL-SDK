/******************************************************************************
 * Copyright (c) 2006-2024, NVIDIA CORPORATION. All rights reserved.
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
/// Helper for Host module implementation
///
/// Determines host specific parameter, like number of CPUs and number
/// of sibling for a CPU socket.

#include "pch.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <mi/base/config.h>

//#include <thread>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

#ifdef MI_PLATFORM_MACOSX
#include <sys/param.h>
#include <sys/sysctl.h>
#endif

namespace MI {

namespace HOST {

// ********** LINUX and MACOSX ************************************************

#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX)

#ifdef MI_ARCH_X86
#ifdef MI_ARCH_64BIT

// assembler function to readout the cpuid information, 64-bit x86 Linux & Mac
void cpuid(int op, int* eax, int* ebx, int* ecx, int* edx)
{
    unsigned int ax = op, bx = 0, cx = 0, dx = 0;

    __asm__ __volatile__ (
                 "pushq %%rbx\n\t"
                 "cpuid\n\t"
                 "movl %%ebx, %%edi\n\t"
                 "popq %%rbx\n\t"
                : "=a" (ax),
                  "=D" (bx),
                  "=c" (cx),
                  "=d" (dx)
                : "0"  (op) );

    *eax = ax;
    *ebx = bx;
    *ecx = cx;
    *edx = dx;
}

#else // MI_ARCH_64BIT

// assembler function to readout the cpuid information, 32-bit x86 Linux & Mac
void cpuid(int op, int* eax, int* ebx, int* ecx, int* edx)
{
    unsigned int ax = op, bx = 0, cx = 0, dx = 0;

    __asm__ __volatile__ (
                 "pushl %%ebx\n\t"
                 "cpuid\n\t"
                 "movl %%ebx, %%edi\n\t"
                 "popl %%ebx\n\t"
                : "=a" (ax),
                  "=D" (bx),
                  "=c" (cx),
                  "=d" (dx)
                : "0"  (op) );

    *eax = ax;
    *ebx = bx;
    *ecx = cx;
    *edx = dx;
}
#endif // MI_ARCH_X86
#endif // MI_ARCH_64BIT
#endif // MI_PLATFORM_LINUX || MI_PLATFORM_MACOSX

// ********** WIN_NT **********************************************************

#ifdef MI_PLATFORM_WINDOWS
#ifdef MI_ARCH_X86

//#ifdef MI_ARCH_64BIT

//#define cpuid mi_host_cpu_id

// assembler function to readout the cpuid information, 64-bit x86 Windows
//extern "C" void mi_host_cpu_id(int op, int* eax, int* ebx, int* ecx, int* edx);

void cpuid(int op, int* eax, int* ebx, int* ecx, int* edx)
{
    int cpuInfo[4];
    __cpuid(cpuInfo, op);
    *eax = cpuInfo[0];
    *ebx = cpuInfo[1];
    *ecx = cpuInfo[2];
    *edx = cpuInfo[3];
}

/*#else // MI_ARCH_64BIT

// assembler function to readout the cpuid information, 32-bit x86 Windows
static void cpuid(int op, int* eax, int* ebx, int* ecx, int* edx)
{
    unsigned int Regax, Regbx, Regcx, Regdx;

     __asm
     {
         mov eax, op
         cpuid
         mov Regax, eax
         mov Regbx, ebx
         mov Regcx, ecx
         mov Regdx, edx
     }

    *eax = Regax;
    *ebx = Regbx;
    *ecx = Regcx;
    *edx = Regdx;
}

#endif*/ // MI_ARCH_64BIT

int number_of_cpus()
{
    //return std::max(1u, std::thread::hardware_concurrency()); // limits to 64 on windows though (due to processor groups)

    const DWORD noc = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return (noc == 0) ? 1 : noc;
}

int memory_size()
{
    MEMORYSTATUSEX stat;
    stat.dwLength = sizeof(stat);
    GlobalMemoryStatusEx(&stat);
    return static_cast<int>(stat.ullTotalPhys / (1024 * 1024));
}

#endif
#else // not MI_PLATFORM_WINDOWS
#ifdef MI_ARCH_X86
unsigned long long _xgetbv(unsigned int index)
{
        unsigned int eax, edx;
        __asm__ __volatile__(
                "xgetbv;"
                : "=a" (eax), "=d"(edx)
                : "c" (index)
        );
        return ((unsigned long long)edx << 32) | eax;
}
#endif
#endif // MI_PLATFORM_WINDOWS

// ********** LINUX ***********************************************************

#ifdef MI_PLATFORM_LINUX

int number_of_cpus()
{
    //return std::max(1u, std::thread::hardware_concurrency());

    FILE* fd = fopen("/proc/cpuinfo", "r");
    if (!fd)
        return 1;

    char str[1024];
    int count = 0;
    while (fgets(str, sizeof(str), fd))
        if (strncmp(str, "processor", 9) == 0)
            count++;

    fclose(fd);
    return count;
}

int memory_size()
{
    FILE* fd = fopen("/proc/meminfo", "r");
    if (!fd)
        return 0;

    char str[1024];
    while (fgets(str, sizeof(str), fd))
        if (strncmp(str, "MemTotal", 8) == 0) {
            char* ptr = str;
            ptr += strcspn(ptr, " \t");
            ptr += strspn(ptr, " \t");
            int size = atoi(ptr);
            fclose(fd);
            return size / 1024;
        }

    fclose(fd);
    return 0;
}

#endif // MI_PLATFORM_LINUX

// ********** MACOSX **********************************************************

#ifdef MI_PLATFORM_MACOSX

int number_of_cpus()
{
    //return std::max(1u, std::thread::hardware_concurrency());

    int name[] = {CTL_HW, HW_NCPU};
    int value;
    size_t length = sizeof(value);
    int result = sysctl(name, sizeof(name)/sizeof(name[0]), &value, &length, 0, 0);
    return result == 0 ? value : 1;
}

int memory_size()
{
    int name[] = { CTL_HW, HW_MEMSIZE };
    int64_t value = 0;
    size_t length = sizeof(value);
    int result = sysctl(name, sizeof(name)/sizeof(name[0]), &value, &length, 0, 0);
    return result == 0 ? value / (1024 * 1024) : 0;
}

#endif // MI_PLATFORM_MACOSX

// ********** ALL *************************************************************

// returns the number of CPUs per socket (siblings)
// set the affinity to a particular cpu and read their cpuid
// detect if this CPU is a logical or physical CPU
// if a logical CPU was detected, the HT bit is set
int sibs_per_cpu()
{
#ifdef MI_ARCH_X86
    int eax = 0, ebx = 0, ecx = 0, edx = 0;
    cpuid(1, &eax, &ebx, &ecx, &edx);

    const int HT_BIT = 0x10000000;
    if (!(edx & HT_BIT))
        return 1;

    int sibs = (ebx & 0xff0000) >> 16;
    return sibs;
#else
    return 1; //TODO: Implement for other architectures
#endif // MI_ARCH_X86
}


#ifndef _XCR_XFEATURE_ENABLED_MASK
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

bool has_avx_support(const int cpu_info[4])
{
#ifdef MI_ARCH_X86
    const bool uses_XSAVE_XRSTORE = !!(cpu_info[2] & (1 << 27));
    const bool has_avx_support = !!(cpu_info[2] & (1 << 28));
    if (uses_XSAVE_XRSTORE && has_avx_support) {
        const unsigned long long xcr_feature_mask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        return !!(xcr_feature_mask & 0x6);
    }
#endif
    return false;
}

} // namespace HOST

} // namespace MI
