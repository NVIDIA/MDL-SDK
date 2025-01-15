/***************************************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_HAL_THREAD_I_THREAD_RW_LOCK_H
#define BASE_HAL_THREAD_I_THREAD_RW_LOCK_H

#include <cstdlib>
#include <atomic>

#include <mi/base/config.h>
#include <base/system/main/i_assert.h>

#include "i_thread_block.h"

#ifndef MI_PLATFORM_WINDOWS
#include <cerrno>
#include <pthread.h>
#else
#include <mi/base/miwindows.h>
#endif

namespace MI {

namespace THREAD {

class Block_shared;

/// Non-recursive reader/writer lock class.
class Shared_lock
{
public:
    /// Constructor.
    Shared_lock() = default;

    Shared_lock( const Shared_lock&) = delete;
    Shared_lock& operator=( const Shared_lock&) = delete;

    /// Destructor.
    ~Shared_lock();

    using Block_shared = THREAD::Block_shared;
    using Block_exclusive = THREAD::Block<Shared_lock>;

    /// %Locks the lock in shared mode.
    void lock_shared();

    /// Tries to lock the lock in shared mode.
    bool try_lock_shared();

    /// Unlocks the lock in shared mode.
    void unlock_shared();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - Otherwise, the method checks that the lock is held in shared mode by \em some thread, not
    ///   necessarily by this thread.
    void check_is_owned_shared();

    /// %Locks the lock in exclusive mode.
    void lock();

    /// Tries to lock the lock in exclusive mode.
    bool try_lock();

    /// Unlocks the lock in exclusive mode.
    void unlock();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - Otherwise, the method checks that the lock is held in exclusive mode by \em some thread,
    ///   not necessarily by this thread.
    void check_is_owned();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - Otherwise, the method checks that the lock is held in shared \em or exclusive mode by
    ///   \em some thread, not necessarily by this thread.
    void check_is_owned_shared_or_exclusive();

private:
#ifndef MI_PLATFORM_WINDOWS
    /// The pthread rwlock implementing the lock.
    ///
    /// Prefer writers to avoid writer starvation.
#ifdef MI_PLATFORM_MACOSX
    pthread_rwlock_t m_rwlock = PTHREAD_RWLOCK_INITIALIZER;
#else
    pthread_rwlock_t m_rwlock = PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP;
#endif
#else
    /// The srwlock implementing the lock.
    SRWLOCK m_srwlock = SRWLOCK_INIT;
#endif
    /// Number of threads holding the lock in shared mode. Used by the sanity check.
    std::atomic_uint32_t m_locked_shared = 0;
    /// Indicates whether lock is held is exclusive mode. Used by the sanity check.
    bool m_locked_exclusive = false;
};

/// Utility class to acquire a RW lock in \em shared mode.
///
/// \see THREAD::Shared_lock
class Block_shared
{
public:
    /// Constructor.
    ///
    /// Acquires the lock.
    explicit Block_shared( Shared_lock& lock);

    /// Constructor.
    ///
    /// \param lock   If not \c NULL, this lock is acquired. If \c NULL, #set() can be used to
    ///               explicitly acquire a lock later.
    explicit Block_shared( Shared_lock* lock = nullptr);

    Block_shared( const Block_shared&) = delete;
    Block_shared& operator=( const Block_shared&) = delete;

    /// Destructor.
    ///
    /// Releases the RW lock (if it is acquired).
    ~Block_shared();

    /// Acquires a lock.
    ///
    /// Releases the current lock (if it is set) and acquires the given lock. Useful to acquire
    /// a different lock, or to acquire a lock if no lock was acquired in the constructor.
    ///
    /// This method does nothing if the passed lock is already acquired by this class.
    ///
    /// \param lock   The new lock to acquire.
    void set( Shared_lock* lock);

    /// Releases the lock.
    ///
    /// Useful to release the lock before the destructor is called.
    void release();

    /// Tries to acquire a lock.
    ///
    /// Releases the current lock (if it is set) and tries to acquire the given lock. Useful to
    /// acquire a different lock without blocking, or to acquire a lock without blocking if no
    /// lock was acquired in the constructor.
    ///
    /// This method does nothing if the passed lock is already acquired by this class.
    ///
    /// \param lock   The new lock to acquire.
    /// \return       \c true if the lock was acquired, \c false otherwise.
    bool try_set( Shared_lock* lock);

    /// Returns the lock currently owned by this block.
    Shared_lock* get_lock() const;

private:
    // The lock associated with this helper class.
    Shared_lock* m_lock;
};

inline Shared_lock::~Shared_lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_rwlock_destroy( &m_rwlock);
    MI_ASSERT( result == 0);
    (void) result;
#endif
    MI_ASSERT( m_locked_shared == 0);
    MI_ASSERT( !m_locked_exclusive);
}

inline void Shared_lock::lock_shared()
{
#ifndef MI_PLATFORM_WINDOWS
    pthread_rwlock_rdlock( &m_rwlock);
#else
    AcquireSRWLockShared( &m_srwlock);
#endif
    ++m_locked_shared;
}

inline bool Shared_lock::try_lock_shared()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_rwlock_tryrdlock( &m_rwlock);
    if( result != 0)
        return false;
    ++m_locked_shared;
    return true;
#else
    BOOL result = TryAcquireSRWLockShared( &m_srwlock);
    if( result == FALSE)
        return false;
    ++m_locked_shared;
    return true;
#endif
}

inline void Shared_lock::unlock_shared()
{
    --m_locked_shared;
#ifndef MI_PLATFORM_WINDOWS
    pthread_rwlock_unlock( &m_rwlock);
#else
    ReleaseSRWLockShared( &m_srwlock);
#endif
}

inline void Shared_lock::check_is_owned_shared()
{
    MI_ASSERT( m_locked_shared > 0);
}

inline void Shared_lock::lock()
{
#ifndef MI_PLATFORM_WINDOWS
    pthread_rwlock_wrlock( &m_rwlock);
#else
    AcquireSRWLockExclusive( &m_srwlock);
#endif
    m_locked_exclusive = true;
}

inline bool Shared_lock::try_lock()
{
#ifndef MI_PLATFORM_WINDOWS
    int result = pthread_rwlock_trywrlock( &m_rwlock);
    if( result != 0)
        return false;
    m_locked_exclusive = true;
    return true;
#else
    BOOL result = TryAcquireSRWLockExclusive( &m_srwlock);
    if( result == FALSE)
        return false;
    m_locked_exclusive = true;
    return true;
#endif
}

inline void Shared_lock::unlock()
{
    MI_ASSERT( m_locked_exclusive);
    m_locked_exclusive = false;
#ifndef MI_PLATFORM_WINDOWS
    pthread_rwlock_unlock( &m_rwlock);
#else
    ReleaseSRWLockExclusive( &m_srwlock);
#endif
}

inline void Shared_lock::check_is_owned()
{
    MI_ASSERT( m_locked_exclusive);
}

inline void Shared_lock::check_is_owned_shared_or_exclusive()
{
    MI_ASSERT( (m_locked_shared > 0) || m_locked_exclusive);
}

inline Block_shared::Block_shared( Shared_lock& lock)
  : m_lock( &lock)
{
    m_lock->lock_shared();
}

inline Block_shared::Block_shared( Shared_lock* lock)
  : m_lock( lock)
{
    if( m_lock)
        m_lock->lock_shared();
}

inline Block_shared::~Block_shared()
{
    if( m_lock)
        m_lock->unlock_shared();
}

inline void Block_shared::set( Shared_lock* lock)
{
    if( m_lock == lock)
        return;

    if( m_lock)
        m_lock->unlock_shared();
    m_lock = lock;
    if( m_lock)
        m_lock->lock_shared();
}

inline void Block_shared::release()
{
    if( m_lock) {
        m_lock->unlock_shared();
        m_lock = nullptr;
    }
}

inline bool Block_shared::try_set( Shared_lock* lock)
{
    if( m_lock == lock)
        return true;

    if( m_lock)
        m_lock->unlock_shared();
    if( lock && lock->try_lock_shared()) {
        m_lock = lock;
        return true;
    } else
        return false;
}

inline Shared_lock* Block_shared::get_lock() const
{
    return m_lock;
}

}  // namespace THREAD

}  // namespace MI

#endif // BASE_HAL_THREAD_I_THREAD_RW_LOCK_H
