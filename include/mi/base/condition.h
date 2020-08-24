/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file       mi/base/condition.h
/// \brief      Multithreading condition.
///
/// See \ref mi_base_threads.

#ifndef MI_BASE_CONDITION_H
#define MI_BASE_CONDITION_H

#include <mi/base/config.h>
#include <mi/base/types.h>

#ifndef MI_PLATFORM_WINDOWS
#include <cerrno>
#include <pthread.h>
#include <sys/time.h>
#else
#include <mi/base/miwindows.h>
#endif

namespace mi {

namespace base {

/** \addtogroup mi_base_threads
@{
*/

/// Conditions allow threads to signal an event and to wait for such a signal, respectively.
class Condition
{
public:
    /// Constructor
    Condition()
    {
#ifndef MI_PLATFORM_WINDOWS
        m_signaled = false;
        pthread_mutex_init( &m_mutex, NULL);
        pthread_cond_init( &m_condvar, NULL);
#else
        m_handle = CreateEvent( NULL, false, false, NULL);
#endif
    }

    /// Destructor
    ~Condition()
    {
#ifndef MI_PLATFORM_WINDOWS
        pthread_mutex_destroy( &m_mutex);
        pthread_cond_destroy( &m_condvar);
#else
        CloseHandle( m_handle);
#endif
    }

    /// Waits for the condition to be signaled.
    ///
    /// If the condition is already signaled at this time the call will return immediately.
    void wait()
    {
#ifndef MI_PLATFORM_WINDOWS
        pthread_mutex_lock( &m_mutex);
        while( !m_signaled) //-V776 PVS
            pthread_cond_wait( &m_condvar, &m_mutex);
        m_signaled = false;
        pthread_mutex_unlock( &m_mutex);
#else
        WaitForSingleObject( m_handle, INFINITE);
#endif
    }

    /// Waits for the condition to be signaled until a given timeout.
    ///
    /// If the condition is already signaled at this time the call will return immediately.
    ///
    /// \param timeout    Maximum time period (in seconds) to wait for the condition to be signaled.
    /// \return           \c true if the timeout was hit, and \c false if the condition was
    ///                   signaled.
    bool timed_wait( Float64 timeout) {
#ifndef MI_PLATFORM_WINDOWS
        struct timeval now;
        gettimeofday( &now, NULL);
        struct timespec timeout_abs;
        timeout_abs.tv_sec = now.tv_sec + static_cast<long>( floor( timeout));
        timeout_abs.tv_nsec
            = 1000 * now.tv_usec + static_cast<long>( 1E9 * ( timeout - floor( timeout)));
        if( timeout_abs.tv_nsec > 1000000000) {
            timeout_abs.tv_sec  += 1;
            timeout_abs.tv_nsec -= 1000000000;
        }

        bool timed_out = false;
        pthread_mutex_lock( &m_mutex);
        while( !m_signaled)
        {
            int result = pthread_cond_timedwait( &m_condvar, &m_mutex, &timeout_abs);
            timed_out = result == ETIMEDOUT;
            if( result != EINTR)
                break;
        }
        m_signaled = false;
        pthread_mutex_unlock( &m_mutex);
        return timed_out;
#else
        DWORD timeout_ms = static_cast<DWORD>( 1000 * timeout);
        DWORD result = WaitForSingleObject( m_handle, timeout_ms);
        return result == WAIT_TIMEOUT;   
#endif    
    }

    /// Signals the condition.
    ///
    /// This will wake up one thread waiting for the condition. It does not matter if the call to
    /// #signal() or #wait() comes first.
    ///
    /// \note If there are two or more calls to #signal() without a call to #wait() in between (and
    /// no outstanding #wait() call), all calls to #signal() except the first one are ignored, i.e.,
    /// calls to #signal() do not increment some counter, but just set a flag.
    void signal()
    {
#ifndef MI_PLATFORM_WINDOWS
        pthread_mutex_lock( &m_mutex);
        m_signaled = true;
        pthread_cond_signal( &m_condvar);
        pthread_mutex_unlock( &m_mutex);
#else
        SetEvent( m_handle);
#endif
    }

    /// Resets the condition.
    ///
    /// This will undo the effect of a #signal() call if there was no outstanding #wait() call.
    void reset()
    {
#ifndef MI_PLATFORM_WINDOWS
        pthread_mutex_lock( &m_mutex);
        m_signaled = false;
        pthread_mutex_unlock( &m_mutex);
#else
        ResetEvent( m_handle);
#endif
    }

private:
#ifndef MI_PLATFORM_WINDOWS
    /// The mutex to be used to protect the m_signaled variable.
    pthread_mutex_t m_mutex;
    /// The condition used to let the thread sleep.
    pthread_cond_t m_condvar;
    /// A variable storing the signaled state.
    bool m_signaled;
#else
    /// The event handle
    HANDLE m_handle;
#endif
};

/*@}*/ // end group mi_base_threads

} // namespace base

} // namespace mi

#endif // MI_BASE_CONDITION_H
