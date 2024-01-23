/******************************************************************************
 * Copyright (c) 2004-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Implementation of init and exit functions for Mod_thread 
///        and those function that are not inline'd.
///
/// Most of the functions in this module are inlined. a few
/// exceptions are included in this file. there's also a need
/// to have a Mod_thread class to initialize and free thread
/// specific memory. some initializations for thread specific
/// data (tsd) are done here.

#include "pch.h"

#include "i_thread_attr.h"
#include "i_thread_thread.h"

#include <vector>
#include <atomic>

#include <base/lib/log/i_log_logger.h>

#ifndef WIN_NT
#ifdef LINUX
#include <dlfcn.h>
#endif
#else
#include <windows.h>
#include <process.h>
#endif
#if defined(MI_PLATFORM_MACOSX)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace MI {

namespace THREAD {

static Thread_attr thread_attr(true);

#ifdef LINUX
typedef int (*GLIBC_GETCPU_FUNCTION)(void);
static GLIBC_GETCPU_FUNCTION g_sched_getcpu = 0;
static bool g_libc_load_attempted = false;
#endif

// Anonymous namespace for stuff only used internally.
namespace {
// Atomic counter which is only incremented in this file
std::atomic_uint32_t s_thread_id_counter(1);
int s_nr_of_cpus = 0;
// The set of currently unused thread IDs (smaller than s_thread_id_counter).
// They can be re-used for new threads to avoids large thread IDs.
std::vector<int> s_unused_thread_ids;
// Lock for the vector above.
Lock s_unused_thread_ids_lock;
#ifdef WIN_NT
std::atomic_uint32_t s_affinity_group_last(0);
int s_affinity_group_n(-1);
#endif
}

// key to access thread specific data
#ifndef WIN_NT
pthread_key_t Thread_attr::m_key;
#else
DWORD Thread_attr::m_key;
#endif

// Destruct a thread.
Thread::~Thread()
{
#ifdef WIN_NT
    CloseHandle(m_handle);
#else
    pthread_attr_destroy(&m_attributes);
#endif
}


// used to register application threads with neuray. The reason for
// this is that neuray stores information in thread local storage that it
// then wants to extract, e.g. when logging information, and if the
// structures in TLS are null it crashes.
void Thread::register_thread()
{
    Thread_attr thread_attr;
    Thr_data *thread_data = new Thr_data;
#ifdef WIN_NT
    TlsSetValue(thread_attr.m_key, (LPVOID)thread_data);
#else
#ifdef ENABLE_ASSERT
    int result;
    result =
#endif
    pthread_setspecific(thread_attr.m_key, (void*)thread_data);
    ASSERT(M_THREAD, result == 0);
#endif
    set_thread_id(thread_attr);
}


// The unregister_thread function currently only frees that storage,
// something that is done for neuray threads when they terminate their
// mainline in do_run(), but obviosuly this can not be done for application
// threads.
void Thread::unregister_thread()
{
    Thread_attr thread_attr;
    reclaim_thread_id(thread_attr);
    Thr_data *thread_data = thread_attr.access_data();
    delete thread_data;
}

int Thread::get_nr_of_cpus()
{
    if (s_nr_of_cpus == 0)
    {
#if 0
        s_nr_of_cpus = std::max(1u, std::thread::hardware_concurrency()); // limits to 64 on windows though (due to processor groups)
#else
#ifdef WIN_NT
        s_nr_of_cpus = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#else
#ifdef LINUX
        s_nr_of_cpus = sysconf( _SC_NPROCESSORS_ONLN );
#else
        int num_logical_cores;
        size_t len = sizeof(num_logical_cores);
        sysctlbyname("hw.logicalcpu", &num_logical_cores, &len, NULL, 0);
        s_nr_of_cpus = num_logical_cores;
#endif
#endif
#endif
    }

    return s_nr_of_cpus;
}

int Thread::get_cpu()
{
#ifdef WIN_NT
    PROCESSOR_NUMBER pn;
    GetCurrentProcessorNumberEx(&pn);
    unsigned int idx = pn.Number;
    for (WORD i = 0; i < pn.Group; ++i)
        idx += GetActiveProcessorCount(i);
    return idx;
#else
#ifndef LINUX
    // We currently do not have support for Mac OS
    return -1;
#else
    if (g_sched_getcpu == 0 && !g_libc_load_attempted)
    {
        void* handle = dlopen("libc.so.6", RTLD_LAZY|RTLD_NOLOAD);
        if (handle == NULL)
        {
            LOG::mod_log->error(M_THREAD, LOG::Mod_log::C_MISC, 2,
                    "Fail to load libc: %s", dlerror());
            g_libc_load_attempted = true;
            return -1;
        }
        ASSERT(M_THREAD, handle);

        g_sched_getcpu = (GLIBC_GETCPU_FUNCTION) dlsym(handle, "sched_getcpu");
        if (!g_sched_getcpu)
        {
            LOG::mod_log->error(M_THREAD, LOG::Mod_log::C_MISC, 2,
                    "Cannot find sched_getcpu in glibc (needs glibc version >= 2.6)");
            g_libc_load_attempted = true;
            return -1;
        }
        g_libc_load_attempted = true;
    }

    if (g_sched_getcpu == 0)
        return -1;
    else
        return (*g_sched_getcpu)();
#endif
#endif
}

bool Thread::set_cpu_affinity(const boost::dynamic_bitset<>& cpu_bit_mask)
{
    if (cpu_bit_mask == m_current_affinity)
        return true;
    m_current_affinity = cpu_bit_mask;

#ifdef WIN_NT
    if (cpu_bit_mask.empty()) // use default mask: pick a processor group (round robin) and use all CPUs in there
    {
        bool result;
        if (s_affinity_group_n > 1)
        {
            GROUP_AFFINITY ga = {};

            USHORT lg = ++s_affinity_group_last;
            ga.Group = lg % s_affinity_group_n; // does a round robin assignment of threads to processor groups

            DWORD apc = GetActiveProcessorCount(ga.Group);
            ASSERT(M_THREAD, apc <= MAXIMUM_PROC_PER_GROUP);
            ASSERT(M_THREAD, MAXIMUM_PROC_PER_GROUP == 64);
            ASSERT(M_THREAD, apc > 0);
            ga.Mask = (apc < 64) ? ((1ULL << apc) - 1ULL) : ~0ULL;

            result = (SetThreadGroupAffinity(m_handle, &ga, nullptr) != 0);

            for(int i = 0; i < s_affinity_group_n; ++i)
                if (i != ga.Group)
                {
                    GROUP_AFFINITY ga2 = {};
                    ga2.Group = i;
                    ga2.Mask = 0;
                    result &= (SetThreadGroupAffinity(m_handle, &ga2, nullptr) != 0);
                }
        }
        else
        {
            DWORD apc = get_nr_of_cpus();
            DWORD_PTR cpu_set = (apc < 64) ? ((1ULL << apc) - 1ULL) : ~0ULL;
            result = SetThreadAffinityMask(m_handle, cpu_set);
        }

        return result;
    }
    else
    if (cpu_bit_mask.none()) // use current CPU only
    {
        bool result;
        GROUP_AFFINITY ga = {};
        PROCESSOR_NUMBER pn;
        GetCurrentProcessorNumberEx(&pn);
        ga.Group = pn.Group;
        ga.Mask = ((DWORD_PTR)1) << pn.Number;

        result = (SetThreadGroupAffinity(m_handle, &ga, nullptr) != 0);

        for(int i = 0; i < s_affinity_group_n; ++i)
            if (i != ga.Group)
            {
                GROUP_AFFINITY ga2 = {};
                ga2.Group = i;
                ga2.Mask = 0;
                result &= (SetThreadGroupAffinity(m_handle, &ga2, nullptr) != 0);
            }

        return result;
    }
    else // use CPUs specified via cpu_bit_mask
    {
        DWORD_PTR cpu_set = 0;
        WORD pg = 0;
        DWORD apc = GetActiveProcessorCount(pg);
        unsigned int cpu_set_idx = 0;
        bool result = true;
        for (unsigned int i = 0; i < cpu_bit_mask.size(); i++, cpu_set_idx++)
        {
            if (apc == i) // do we hit the next processor group? -> set mask for previous one
            {
                GROUP_AFFINITY ga = {};
                ga.Group = pg;
                ga.Mask = cpu_set;
                result &= (SetThreadGroupAffinity(m_handle,&ga,NULL) != 0);
                cpu_set = 0;
                cpu_set_idx = 0;
                pg++;
                if (pg >= s_affinity_group_n)
                    return result;
                apc += GetActiveProcessorCount(pg);
            }
            if (cpu_bit_mask[i])
                cpu_set |= ((DWORD_PTR)1) << cpu_set_idx;
        }

        GROUP_AFFINITY ga = {};
        ga.Group = pg;
        ga.Mask = cpu_set;
        result &= (SetThreadGroupAffinity(m_handle, &ga, NULL) != 0);

        return result;
    }

    
#else
#if !defined(LINUX)
    return false;
#else
    cpu_set_t cpu_set;
    int current_cpu;

    CPU_ZERO(&cpu_set);

    if (cpu_bit_mask.empty())
    {
        int nr_of_cpus = Thread::get_nr_of_cpus();
        for (unsigned int i = 0; i < nr_of_cpus; i++)
            CPU_SET(i, &cpu_set);
    }
    else
    if (cpu_bit_mask.none())
    {
        current_cpu = Thread::get_cpu();
        CPU_SET(current_cpu, &cpu_set);
    }
    else
    {
        for (unsigned int i = 0; i < cpu_bit_mask.size(); i++)
            if (cpu_bit_mask[i])
                CPU_SET(i, &cpu_set);
    }

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set))
        return false;
    return true;
#endif
#endif
}

bool Thread::pin_cpu()
{
    int nr_of_cpus = Thread::get_nr_of_cpus();
    if (nr_of_cpus <= 0)
        return false;

    boost::dynamic_bitset<> cpu_mask(nr_of_cpus);
    return set_cpu_affinity(cpu_mask); // all zero? -> use current CPU
}

bool Thread::pin_cpu(int cpu)
{
    int nr_of_cpus = Thread::get_nr_of_cpus();
    if (nr_of_cpus <= 0)
        return false;

    if (cpu >= nr_of_cpus)
        return false;

    boost::dynamic_bitset<> cpu_mask(nr_of_cpus);
    cpu_mask[cpu] = true; // all zero except for requested CPU
    return set_cpu_affinity(cpu_mask);
}

bool Thread::unpin_cpu()
{
    int nr_of_cpus = Thread::get_nr_of_cpus();
    if (nr_of_cpus <= 0)
        return false;

    boost::dynamic_bitset<> cpu_mask;
    return set_cpu_affinity(cpu_mask); // empty? -> use default mask
}

void Thread::set_thread_id(Thread_attr& thread_attr)
{
    Lock::Block block(&s_unused_thread_ids_lock);
    int thread_id;
    if (s_unused_thread_ids.empty())
        thread_id = s_thread_id_counter++;
    else {
        thread_id = s_unused_thread_ids.back();
        s_unused_thread_ids.pop_back();
    }
    thread_attr.set_id(thread_id);
}
    
void Thread::reclaim_thread_id(Thread_attr& thread_attr)
{
    Lock::Block block(&s_unused_thread_ids_lock);
    s_unused_thread_ids.push_back(thread_attr.get_id());
}

#ifdef WIN_NT
unsigned __stdcall
#else
void *
#endif
Thread::do_run(
    void *thread)			// thread class instance
{
    // for clarification: in neuray, all threads except for the system thread
    // allocate and delete their thread specific memory here. for the system
    // thread it is done in the module init/exit functions because it cannot be
    // derived from the Thread class. foreign threads (not derived from Thread,
    // e.g. oem threads) allocate/free thread specific memory in the register/
    // unregister functions which are mandatory before executing neuray code.
    Thread_attr thr_attr;
    auto thr_data = std::make_unique<Thr_data>();// allocate thread specific memory

#ifdef WIN_NT
    TlsSetValue(thr_attr.m_key, (LPVOID)thr_data.get());
#else
#ifdef ENABLE_ASSERT
    int result;
    result =
#endif
    pthread_setspecific(thr_attr.m_key, thr_data.get());
    ASSERT(M_THREAD, result == 0);
#endif

    set_thread_id(thread_attr);
    ((Thread *)thread)->run();		// invoke thread mainline
    reclaim_thread_id(thread_attr);

#ifdef WIN_NT
    _endthreadex(0);
#endif

    return 0;
}

#ifndef WIN_NT

class Thread_attr;

// Construct a thread.
Thread::Thread(bool create_detached)
    : m_tid{}, m_was_started(false), m_create_detached(create_detached)
{
    m_policy = SCHED_OTHER;
    m_params.sched_priority = 0;
    pthread_attr_init(&m_attributes);
    if (m_create_detached)
	pthread_attr_setdetachstate(&m_attributes, PTHREAD_CREATE_DETACHED);
}

// Start the thread, invoking the run method.
bool Thread::start()
{
    {
        THREAD::Lock::Block locker(&m_lock);
        if (m_was_started)
            return false;
        m_was_started = true;
    }
    pthread_create(&m_tid,&m_attributes,do_run,this);
    return true;
}

// Wait for the thread to terminate.
void Thread::join()
{
    THREAD::Lock::Block locker(&m_lock);
    if (m_was_started && !m_create_detached)
	pthread_join(m_tid, NULL);
}

// Set thread scheduling priority
void Thread::set_priority(
	const int prio)			// set thread priority to this
{
    m_params.sched_priority = prio;
#ifdef _POSIX_THREAD_PRIORITY_SCHEDULING
    pthread_setschedparam(m_tid,m_policy,&m_params);
#endif
}

// Get thread scheduling priority
int Thread::get_priority()
{
#ifdef _POSIX_THREAD_PRIORITY_SCHEDULING
    pthread_getschedparam(m_tid,&m_policy,&m_params);
#endif
    return m_params.sched_priority;
}

bool Thread::is_current_thread() const
{
    return (m_was_started && (m_tid == pthread_self()));
}

#else // WIN_NT

// Construct a thread.
Thread::Thread(bool create_detached)
    : m_was_started(false), m_create_detached(create_detached)
{
    m_handle = (HANDLE)_beginthreadex(NULL, 0, &Thread::do_run, this, CREATE_SUSPENDED, &m_tid);

    // set thread affinity group
    if(s_affinity_group_n == -1)
        s_affinity_group_n = GetActiveProcessorGroupCount();
    if (s_affinity_group_n > 1) {
        GROUP_AFFINITY ga = {};

        USHORT lg = ++s_affinity_group_last;
        ga.Group = lg % s_affinity_group_n; // does a round robin assignment of threads to processor groups

        DWORD apc = GetActiveProcessorCount(ga.Group);
        ASSERT(M_THREAD, apc <= MAXIMUM_PROC_PER_GROUP);
        ASSERT(M_THREAD, MAXIMUM_PROC_PER_GROUP == 64);
        ASSERT(M_THREAD, apc > 0);
        ga.Mask = (apc < 64) ? ((1ULL << apc) - 1ULL) : ~0ULL;

        if (!SetThreadGroupAffinity(m_handle, &ga, nullptr)) {
            // warn, but no need to return - this is just a potential performance issue
            //!! mi_nwarning(334, "cannot set thread group for thread %d: %s", m_tid, mi_strerror());
        }
    }
}

// Start the thread, invoking the run method.
bool Thread::start()
{
    {
	THREAD::Lock::Block locker(&m_lock);
	if (m_was_started)
	    return false;
	m_was_started = true;
    }
    const DWORD result = ResumeThread(m_handle);
    return (result != -1);
}

// Wait for the thread to terminate.
void Thread::join()
{
    THREAD::Lock::Block locker(&m_lock);
    if (m_was_started)
	// The thread handle is signaled when the thread terminates.
	WaitForSingleObject(m_handle, INFINITE);
}

// Set thread scheduling priority
void Thread::set_priority(
    const int prio)			// set thread priority to this
{
    SetThreadPriority(m_handle, prio);
}

// Get thread scheduling priority
int Thread::get_priority()
{
    return GetThreadPriority(m_handle);
}

bool Thread::is_current_thread() const
{
    return (m_was_started && (GetCurrentThreadId() == m_tid));
}

#endif

}

}
