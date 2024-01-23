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
/// \brief

#ifndef BASE_HAL_THREAD_I_THREAD_LOCK_H
#define BASE_HAL_THREAD_I_THREAD_LOCK_H

#include <cstdlib>

#include <mi/base/config.h>
#include <base/system/main/i_assert.h>
#include <base/lib/mem/i_mem_allocatable.h>

#include "i_thread_block.h"

#ifndef MI_PLATFORM_WINDOWS
#include <cerrno>
#include <pthread.h>
#else
#include <mi/base/miwindows.h>
#endif

namespace MI {

namespace THREAD {

/// Non-recursive lock class.
///
/// The lock implements a critical region that only one thread can enter at a time. The lock is
/// non-recursive, i.e., a thread that holds the lock can not lock it again. Any attempt to do so
/// will abort the process.
///
/// Other pre- and post-conditions are checked via #MI_ASSERT.
///
/// \see #MI::THREAD::Lock::Block, #mi::base::Lock, #mi::base::Lock::Block
///
/// Differences between #mi::base::Lock and this class:
/// - Namespaces
/// - This class is derived from MEM::Allocatable.
/// - This class uses MI_ASSERT, the public API uses mi_base_assert.
/// - This class uses a separate templated Block class that is shared with other locks and pulled in
///   via typedef, the public API uses a nested class.
/// - In this class lock(), try_lock(), unlock() are public (probably for history reasons), the
///   public API enforces the use of Lock::Block.
/// - This class has the check_is_owned() method.
class Lock : public MEM::Allocatable
{
public:
    /// Constructor.
    Lock();

    Lock( Lock const &) = delete;
    Lock& operator=( Lock const &) = delete;

    /// Destructor.
    ~Lock();

    /// Utility class to acquire a lock that is released by the destructor.
    using Block = THREAD::Block<Lock>;

    /// %Locks the lock.
    void lock();

    /// Tries to lock the lock.
    bool try_lock();

    /// Unlocks the lock.
    void unlock();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - On Linux/MacOS, the method checks that the lock is held by this thread (if assertions are
    ///   enabled). The implementation is flagged by helgrind/drd as error.
    /// - On Windows, the method checks that the lock is held by \em some thread, not necessarily
    ///   by this thread (if assertions are enabled).
    ///
    /// Places that use this check might consider using a recursive lock.
    void check_is_owned();

private:
#ifndef MI_PLATFORM_WINDOWS
    // The mutex implementing the lock.
    pthread_mutex_t m_mutex;
#else
    // The srwlock implementing the lock.
    SRWLOCK m_srwlock;
    // The flag used to ensure that the lock is non-recursive.
    bool m_locked;
#endif
};

/// Recursive lock class.
///
/// The lock implements a critical region that only one thread can enter at a time. The lock is
/// recursive, i.e., a thread that holds the lock can lock it again.
///
/// Pre- and post-conditions are checked via #MI_ASSERT.
///
/// \see #MI::THREAD::Recursive_lock::Block, #mi::base::Recursive_lock,
///      #mi::base::Recursive_lock::Block
///
/// Differences between #mi::base::Recursive_lock and this class:
/// - Namespaces
/// - This class is derived from MEM::Allocatable.
/// - This class uses MI_ASSERT, the public API uses mi_base_assert.
/// - This class uses a separate templated Block class that is shared with other locks and pulled in
///   via typedef, the public API uses a nested class.
/// - In this class lock(), try_lock(), unlock() are public (probably for history reasons), the
///   public API enforces the use of Recursive_lock::Block.
class Recursive_lock : public MEM::Allocatable
{
public:
    /// Constructor.
    Recursive_lock();

    /// Destructor.
    ~Recursive_lock();

    Recursive_lock( Recursive_lock const &) = delete;
    Recursive_lock& operator=( Recursive_lock const &) = delete;

    /// Utility class to acquire a lock that is released by the destructor.
    using Block = THREAD::Block<Recursive_lock>;

    /// Locks the lock.
    void lock();

    /// Tries to lock the lock.
    bool try_lock();

    /// Unlocks the lock.
    void unlock();

private:
#ifndef MI_PLATFORM_WINDOWS
    // The mutex implementing the lock.
    pthread_mutex_t m_mutex;
#else
    // The critical section implementing the lock.
    CRITICAL_SECTION m_critical_section;
#endif
};

inline Lock::Lock()
#ifdef MI_PLATFORM_WINDOWS
  : m_srwlock( SRWLOCK_INIT),
    m_locked( false)
#endif
{
#ifndef MI_PLATFORM_WINDOWS
    pthread_mutexattr_t mutex_attributes;
    pthread_mutexattr_init( &mutex_attributes);
    pthread_mutexattr_settype( &mutex_attributes, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init( &m_mutex, &mutex_attributes);
#endif
}

inline Lock::~Lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_destroy( &m_mutex);
    MI_ASSERT( result == 0);
    (void) result;
#else
    MI_ASSERT( !m_locked);
#endif
}

inline void Lock::lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_lock( &m_mutex);
    if( result == EDEADLK) {
        MI_ASSERT( !"Dead lock");
        abort();
    }
#else
    AcquireSRWLockExclusive( &m_srwlock);
    if( m_locked) {
        MI_ASSERT( !"Dead lock");
        abort();
    }
    m_locked = true;
#endif
}

inline bool Lock::try_lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_trylock( &m_mutex);
    // Old glibc versions incorrectly return EDEADLK instead of EBUSY
    // (https://sourceware.org/bugzilla/show_bug.cgi?id=4392).
    MI_ASSERT( result == 0 || result == EBUSY || result == EDEADLK);
    return result == 0;
#else
    BOOL result = TryAcquireSRWLockExclusive( &m_srwlock);
    if( result == FALSE)
        return false;
    if( m_locked) {
        ReleaseSRWLockExclusive( &m_srwlock);
        return false;
    }
    m_locked = true;
    return true;
#endif
}

inline void Lock::unlock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_unlock( &m_mutex);
    MI_ASSERT( result == 0);
    (void) result;
#else
    MI_ASSERT( m_locked);
    m_locked = false;
    ReleaseSRWLockExclusive( &m_srwlock);
#endif
}

inline void Lock::check_is_owned()
{
#ifndef MI_PLATFORM_WINDOWS
    MI_ASSERT( pthread_mutex_lock( &m_mutex) == EDEADLK);
#else
    MI_ASSERT( m_locked);
#endif
}

inline Recursive_lock::Recursive_lock()
{
#ifndef MI_PLATFORM_WINDOWS
    pthread_mutexattr_t mutex_attributes;
    pthread_mutexattr_init( &mutex_attributes);
    pthread_mutexattr_settype( &mutex_attributes, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init( &m_mutex, &mutex_attributes);
#else
    InitializeCriticalSection( &m_critical_section);
#endif
}

inline Recursive_lock::~Recursive_lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_destroy( &m_mutex);
    MI_ASSERT( result == 0);
    (void) result;
#else
    DeleteCriticalSection( &m_critical_section);
#endif
}

inline void Recursive_lock::lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_lock( &m_mutex);
    MI_ASSERT( result == 0);
    (void) result;
#else
    EnterCriticalSection( &m_critical_section);
#endif
}

inline bool Recursive_lock::try_lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_trylock( &m_mutex);
    MI_ASSERT( result == 0 || result == EBUSY);
    return result == 0;
#else
    BOOL result = TryEnterCriticalSection( &m_critical_section);
    return result != 0;
#endif
}

inline void Recursive_lock::unlock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_mutex_unlock( &m_mutex);
    MI_ASSERT( result == 0);
    (void) result;
#else
    LeaveCriticalSection( &m_critical_section);
#endif
}

}  // namespace THREAD

}  // namespace MI

#endif // BASE_HAL_THREAD_I_THREAD_LOCK_H
