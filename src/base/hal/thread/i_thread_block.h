/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Block guard for locks.
///

#ifndef BASE_HAL_THREAD_I_THREAD_BLOCK_H
#define BASE_HAL_THREAD_I_THREAD_BLOCK_H

#ifndef NULL
#define NULL (0)
#endif

namespace MI {

namespace THREAD {

/// This class uses the guard pattern to acquire a lock within the scope of its lifetime.
/// The class is templated to work on either Lock or Spin_lock.
template<typename T>
class Block
{
public:

    explicit Block(T& lock);

    // Can be constructed with NULL to delay setting the lock.
    explicit Block(T* lock = NULL);

    ~Block();

    // Replace the lock held by this block (first unlocks the current lock).
    // Calling with NULL is the same as calling release().
    //
    // This method does nothing if the passed lock is already acquired by this class.
    void set(T* lock);

    // Release the lock prematurely
    void release();

    // Tries to acquire the lock, returns whether success. If success, unlocking happens in
    // destructor as usual.
    //
    // This method does nothing if the passed lock is already acquired by this class.
    bool try_set(T* lock);

    // Returns the lock currently owned by this block.
    T* get_lock() const;

private:

    // Disallow copy
    Block(const Block&);
    Block& operator=(const Block&);

    T* m_lock;
};

template<typename T>
inline Block<T>::Block(T& lock)
: m_lock(&lock)
{
    m_lock->lock();
}

template<typename T>
inline Block<T>::Block(T* lock)
: m_lock(lock)
{
    if(m_lock != NULL)
        m_lock->lock();
}

template<typename T>
inline Block<T>::~Block()
{
    if(m_lock != NULL)
        m_lock->unlock();
}

template<typename T>
inline void Block<T>::set(T* lock)
{
    if(m_lock == lock)
        return;

    // Unlock current lock
    if(m_lock != NULL)
        m_lock->unlock();

    // Acquire new lock
    m_lock = lock;
    if(m_lock != NULL)
        m_lock->lock();
}

template<typename T>
inline void Block<T>::release()
{
    if(m_lock != NULL)
    {
        m_lock->unlock();
        m_lock = NULL;
    }
}

template<typename T>
inline bool Block<T>::try_set(T* lock)
{
    if(m_lock == lock)
        return true;

    release();
    if(lock && lock->try_lock())
    {
        m_lock = lock;
        return true;
    }
    else
    {
        return false;
    }
}

template<typename T>
inline T* Block<T>::get_lock() const
{
    return m_lock;
}

}
}

#endif
