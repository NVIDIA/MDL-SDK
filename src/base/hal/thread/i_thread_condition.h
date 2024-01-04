/***************************************************************************************************
 * Copyright (c) 2004-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Interface to conditions.
///
/// A Condition implements a signaling mechanism between threads. The signaling thread sets the
/// condition by invoking the signal method. The receiving thread invokes the #wait() method, which
/// will block it till the condition becomes true. Once the condition becomes true, the receiving
/// thread proceeds and the condition is reset. If multiple threads wait for the condition, only one
/// thread will pass and reset the condition.
///
/// It does not matter if signal or wait is invoked first. Note that if there are several calls to
/// #signal() without a call to #wait() in between (and no outstanding #wait() call), all calls to
/// #signal() except the first are ignored, i.e., calls to #signal() do not increment some counter,
/// but just set a flag.
///
/// This mechanism allows to signal to a thread that a condition has been attained, without that
/// thread needing to poll the condition.
///
/// This is an example of the use of Conditions:
///
/// \code
///
///    int counter = 0;
///
///    Condition counter_write_condition;
///    Condition counter_read_condition;
///
///    class Condition_writer : public Thread {
///    protected:
///        void run(void) {
///            for(int i = 0; i < 10; i++) {
///                counter_read_condition.wait();
///                printf("wrote counter value %d\n",++counter);
///                counter_write_condition.signal();
///            }
///        }
///    };
///
///    class Condition_reader : public Thread {
///    protected:
///        void run(void) {
///            for(int i = 0; i < 10; i++) {
///                counter_write_condition.wait();
///                printf("read counter value %d\n",counter);
///                counter_read_condition.signal();
///            }
///        }
///    };
///
///    void example_condition(void) {
///        Condition_writer thread_1;
///        Condition_reader thread_2;
///
///        thread_1.start();
///        thread_2.start();
///        counter_read_condition.signal();
///        thread_1.join();
///        thread_2.join();
///    }
///
///    \endcode
///

#ifndef BASE_HAL_THREAD_I_THREAD_CONDITION_H
#define BASE_HAL_THREAD_I_THREAD_CONDITION_H

#ifndef WIN_NT
#include <sys/time.h> // for gettimeofday
#include <pthread.h>
#include <cerrno>
#include <cmath>
#else
#include <mi/base/miwindows.h>
#endif
#include <base/lib/mem/i_mem_allocatable.h>

namespace MI {
namespace THREAD {

/// The Condition class that implements a signaling mechanism between threads.
class Condition : public MEM::Allocatable
{

public:

    /// Constructor.
    ///
    /// \p auto_reset is set to true for every signal to wake exactly one waiting thread,
    /// false for every signal to remain open (thus unblocking all waiting threads) until
    /// explicitly reset.
    Condition(bool auto_reset = true, bool signaled = false);

    /// Destructor.
    ~Condition();

    /// Wait for condition.
    void wait();

    /// Wait for condition or timeout in seconds.
    ///
    /// \p timed_out is set to true iff the function returned
    /// as a result of a timeout.
    void timed_wait(double timeout_seconds, bool& timed_out);

    /// Signal condition.
    void signal();

    /// Re-set the signal state of the condition.
    void reset();

private:
    // non-copyable
    Condition(const Condition&);
    Condition& operator=(const Condition&);


#ifndef WIN_NT
    /// The pthread mutex associated to this Condition.
    pthread_mutex_t m_mutex;

    /// The pthread condvar implementing this condition.
    pthread_cond_t m_condvar;

    /// Flag to indicate that this condition has been signaled.
    bool m_signaled;

    /// Flags the condition as being auto-reset,
    // i.e. a single thread is awaken by any call to signal().
    bool m_auto_reset;
#else
    HANDLE m_handle;
#endif

};


#ifndef WIN_NT

// Constructor.
inline Condition::Condition(const bool auto_reset, const bool signaled)
: m_signaled(signaled)
, m_auto_reset(auto_reset)
{
    pthread_mutex_init(&m_mutex,NULL);
    pthread_cond_init(&m_condvar,NULL);
}

// Destructor.
inline Condition::~Condition()
{
    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy(&m_condvar);
}

// Wait for condition.
inline void Condition::wait()
{
    pthread_mutex_lock(&m_mutex);
    while(!m_signaled)
        pthread_cond_wait(&m_condvar,&m_mutex);
    m_signaled = !m_auto_reset;
    pthread_mutex_unlock(&m_mutex);
}

inline void Condition::timed_wait(double timeout_seconds, bool& timed_out)
{
    struct timespec timeout_time;
//#ifndef MACOSX
#if 0 // avoid the need to link "rt" on Linux
    clock_gettime(CLOCK_REALTIME, &timeout_time);
    timeout_time.tv_sec += static_cast<long>(floor(timeout_seconds));
    timeout_time.tv_nsec += static_cast<long>((timeout_seconds - floor(timeout_seconds)) * 1e9);
#else
    // Macos doesn't support clock_gettime()
    struct timeval tval;
    gettimeofday(&tval, NULL);
    timeout_time.tv_sec = tval.tv_sec + static_cast<long>(floor(timeout_seconds));
    timeout_time.tv_nsec = tval.tv_usec * 1000
	+ static_cast<long>((timeout_seconds - floor(timeout_seconds)) * 1e9);
#endif
    if(timeout_time.tv_nsec > 1000000000) // overflow?
    {
        timeout_time.tv_sec += 1;
        timeout_time.tv_nsec -= 1000000000;
    }

    timed_out = false;
    pthread_mutex_lock(&m_mutex);
    while(!m_signaled)
    {
        int result = pthread_cond_timedwait(&m_condvar,&m_mutex, &timeout_time);
        if(result == ETIMEDOUT) {
            timed_out = true;
        }
        if (result != EINTR) {
            break;
        }
    }
    // Only change state if signaled. Above loop may also exit
    // due to time out.
    if (m_signaled && m_auto_reset)
        m_signaled = false;
    pthread_mutex_unlock(&m_mutex);
}

// Signal condition.
inline void Condition::signal()
{
    pthread_mutex_lock(&m_mutex);
    m_signaled = true;
    if(m_auto_reset)
        pthread_cond_signal(&m_condvar);    // wake at least 1 thread
    else
        pthread_cond_broadcast(&m_condvar); // wake all threads
    pthread_mutex_unlock(&m_mutex);
}

inline void Condition::reset()
{
    pthread_mutex_lock(&m_mutex);
    m_signaled = false;
    pthread_mutex_unlock(&m_mutex);
}

#else // WIN_NT

// Constructor.
inline Condition::Condition(const bool auto_reset, const bool signaled)
{
    m_handle = CreateEvent(NULL, !auto_reset, signaled, NULL);
}

// Destructor.
inline Condition::~Condition()
{
    CloseHandle(m_handle);
}

// Wait for condition.
inline void Condition::wait()
{
    WaitForSingleObject(m_handle, INFINITE);
}

inline void Condition::timed_wait(const double timeout_seconds, bool& timed_out)
{
    const DWORD milliseconds = static_cast<DWORD>(timeout_seconds * 1000.0);
    DWORD result = WaitForSingleObject(m_handle, milliseconds);
    timed_out = (result == WAIT_TIMEOUT);
}

// Signal condition.
inline void Condition::signal()
{
    SetEvent(m_handle);
}

inline void Condition::reset()
{
    ResetEvent(m_handle);
}

#endif

}
}

#endif
