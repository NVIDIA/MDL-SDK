/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief Interface to threads.
///
/// A Thread implements a separate thread of execution.
/// When the thread is started, it invokes its run method.
/// A thread terminates either when it reaches the end of
/// the run method.
/// Invoking the join method on a thread will block
/// until the thread terminates.
///
/// This is an example of how a Thread is defined and invoked:
/// \code
///
///    int counter = 0;
///
///    class My_thread : public Thread {
///    protected:
///        void run(void) {
///            for(int i = 0; i < 10; i++)
///                ++counter;
///        }
///    };
///
///    void example_thread(void) {
///        My_thread my_thread;
///
///        my_thread.start();
///        my_thread.join();
///        printf("counter value is %d\n",counter);
///    }
///
/// \endcode

#ifndef BASE_HAL_THREAD_I_THREAD_THREAD_H
#define BASE_HAL_THREAD_I_THREAD_THREAD_H


#ifndef WIN_NT
#include <pthread.h>
#else
#include <mi/base/miwindows.h>
#endif

#include "i_thread_lock.h"
#include <base/system/main/types.h>
#include <base/system/stlext/i_stlext_concepts.h>

#include <boost/dynamic_bitset.hpp>

namespace MI {
namespace THREAD {

class Thread_attr;

///
/// This represents a platform independent thread id. Most of the code does not make assumptions
/// about the internal representation of the thread id, only that operators < and == are possible.
/// The exception is that it is possible to get a numerical Uint32 for the thread id. This should
/// be sparsely used.
///
class Thread_id
{
public:
    /// Constructor.
    ///
    /// The ID will be initialized with the thread id of the calling thread.
    Thread_id()
#ifndef WIN_NT
    : m_thread_id(pthread_self())
#else
    : m_thread_id(GetCurrentThreadId())
#endif
    {
    }

    /// Comparison operator
    bool operator== (const Thread_id & other) const
    {
#ifndef WIN_NT
        return pthread_equal( m_thread_id, other.m_thread_id ) != 0;
#else
        return m_thread_id == other.m_thread_id;
#endif
    }

    /// Comparison operator
    bool operator!= (const Thread_id & other) const
    {
#ifndef WIN_NT
        return pthread_equal( m_thread_id, other.m_thread_id ) == 0;
#else
        return m_thread_id != other.m_thread_id;
#endif
    }
    
    /// Comparison operator
    bool operator< (const Thread_id & other) const
    {
        return m_thread_id < other.m_thread_id;
    }
    
    /// Return the numerical representation of the thread id. Should be used sparsely because this
    /// is not guaranteed to be supported on all platforms.
    /// \return A 32-bit thread id which is unique amongst all threads
    Uint64 get_uint()
    {
    	return (Uint64)m_thread_id;
    }

private:
#ifndef WIN_NT
    pthread_t m_thread_id;
#else
    DWORD m_thread_id;
#endif
};

///
/// The thread class which implements a separate thread of invocation.
///
class Thread : public STLEXT::Non_copyable {

public:

    /// Construct a thread.
    Thread(bool create_detached = false);

    /// Destruct a thread.
    virtual ~Thread();

    /// Start the thread, invoking the run method. It is never legal to start a thread more
    /// than once. In particular, a thread may not be restarted once it has completed execution.
    /// Any attempt to call start() multiple times will return false.
    /// \return true, when function succeeds, or false else
    bool start();

    /// Wait for the thread to terminate.
    void join();

    /// Set thread scheduling priority
    /// \param prio new prio
    void set_priority(const int prio);

    /// Get thread scheduling priority
    /// \return the priority
    int get_priority();

    /// Returns true iff the calling thread is the same as the thread managed by 'this'
    bool is_current_thread() const;

    /// Register an application thread with neuray,
    /// initializing the thread local storage appropriately in the process.
    static void register_thread();

    /// Unregister an application thread with neuray,
    /// freeing the thread local storage in the process.
    static void unregister_thread();

    /// Pin the thread to the current CPU
    /// \return success or failure
    bool pin_cpu();

    /// Pin the thread to the given CPU
    /// \return success or failure
    bool pin_cpu( int cpu);

    /// Unpin the thread from any CPU
    /// \return success or failure
    bool unpin_cpu();

    /// Get the CPU on which the calling thread is running
    /// \return the CPU on which the calling thread is running (-1 means error)
    static int get_cpu();

    /// Get the number of available CPU(s)
    /// \return the number of available CPU(s) (-1 means error)
    static int get_nr_of_cpus();

protected:

    /// The run method.
    virtual void run() = 0;

private:

    /// Set the CPU affinity mask for a thread so that the thread will be
    /// scheduled to run only on these CPUs. Special cases: If all bits are
    /// zero, the thread is pinned to its current CPU. If the bitset is empty
    /// the initial default mask is used again.
    /// \return true, when function succeeds, or false else
    bool set_cpu_affinity(const boost::dynamic_bitset<>& cpu_bit_mask);

    /// Obtains a new thread ID and sets it as the ID of this thread.
    static void set_thread_id(Thread_attr& thread_attr);
    
    /// Makes the thread ID available for later re-use.
    static void reclaim_thread_id(Thread_attr& thread_attr);
    
#ifndef WIN_NT
    /// The thread id.
    pthread_t m_tid;
    /// Thread scheduling policy.
    int m_policy;

    /// Thread scheduling parameters.
    struct sched_param m_params;

    /// Thread attributes.
    pthread_attr_t m_attributes;
    /// Utility function to invoke the run method.
    /// \param thread the actual thread to start
    static void *do_run(void *thread);
#else
    HANDLE m_handle;			///< the thread handle
    unsigned int m_tid;                 ///< the thread ID
    /// Utility function to invoke the run method.
    /// \param thread the actual thread to start
    static unsigned __stdcall do_run(void *thread);
#endif
    boost::dynamic_bitset<> m_current_affinity; ///< current/cached CPU affinity mask, otherwise the job system constantly sets it to the same exact value
    THREAD::Lock m_lock;		///< protect the flag \c m_was_started
    bool m_was_started;			///< was the thread already started?
    bool m_create_detached;		///< need/can not be collected with join()
};

}
}

#endif
