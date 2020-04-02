/******************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Internal helper functions for \c Atomic_counter.

#ifndef BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_INLINE_H
#define BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_INLINE_H

#ifndef BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_H
#  error "do not include this file directly -- include i_stlext_atomic_counter.h"
#endif

#include <base/system/main/platform.h>          // define MI_FORCE_INLINE

///////////////////////////////////////////////////////////////////////////////
// Implementation for x86, x86-64, GNU C Compiler, Intel C Compiler
///////////////////////////////////////////////////////////////////////////////

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) && ! defined(__INTEL_COMPILER))

namespace MI { namespace STLEXT { namespace IMPL {

typedef Uint32 volatile Native_atomic_counter;

MI_FORCE_INLINE void create_counter(Native_atomic_counter & counter, Uint32 i)
{
    counter = i;
}

MI_FORCE_INLINE void destroy_counter(Native_atomic_counter & counter)
{
}

MI_FORCE_INLINE Uint32 get_counter(Native_atomic_counter const & counter)
{
    return counter;
}

MI_FORCE_INLINE Uint32 atomic_add(Native_atomic_counter & counter, Uint32 i)
{
    Uint32 retval;
    asm volatile (
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        "addl %2,%0\n"
        : "=&r" (retval), "+m" (counter)
        : "r" (i)
        : "cc"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_sub(Native_atomic_counter & counter, Uint32 i)
{
    Uint32 retval;
    asm volatile (
        "neg %2\n"
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        "addl %2,%0\n"
        : "=&r" (retval), "+m" (counter)
        : "r" (i)
        : "cc", "%2"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_inc(Native_atomic_counter & counter)
{
    Uint32 retval;
    asm volatile (
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $1,%0\n"
        : "=&r" (retval), "+m" (counter)
        :
        : "cc"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_dec(Native_atomic_counter & counter)
{
    Uint32 retval;
    asm volatile (
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $-1,%0\n"
        : "=&r" (retval), "+m" (counter)
        :
        : "cc"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_inc(Native_atomic_counter & counter)
{
    Uint32 retval;
    asm volatile (
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r" (retval), "+m" (counter)
        :
        : "cc"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_dec(Native_atomic_counter & counter)
{
    Uint32 retval;
    asm volatile (
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r" (retval), "+m" (counter)
        :
        : "cc"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_add(Native_atomic_counter& counter, Uint32 i)
{
    Uint32 retval;
    asm volatile (
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r" (retval), "+m" (counter)
        : "r" (i)
        : "cc"
        );
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_sub(Native_atomic_counter& counter, Uint32 i)
{
    Uint32 retval;
    asm volatile (
        "neg %2\n"
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r" (retval), "+m" (counter)
        : "r" (i)
        : "cc", "%2"
        );
    return retval;
}

}}} // MI::STLEXT::IMPL


///////////////////////////////////////////////////////////////////////////////
// Implementation for x86, x86-64, Microsoft Visual C++
///////////////////////////////////////////////////////////////////////////////

#elif (defined(X86) || defined(_M_IX86)) && defined(_MSC_VER)

#include <intrin.h>

#pragma intrinsic (_InterlockedExchangeAdd)
#pragma intrinsic (_InterlockedCompareExchange)

namespace MI { namespace STLEXT { namespace IMPL {

typedef long volatile Native_atomic_counter;

MI_FORCE_INLINE void create_counter(Native_atomic_counter & counter, Uint32 i)
{
    counter = i;
}

MI_FORCE_INLINE void destroy_counter(Native_atomic_counter & counter)
{
}

MI_FORCE_INLINE Uint32 get_counter(Native_atomic_counter const & counter)
{
    return static_cast<Uint32>(counter);
}

MI_FORCE_INLINE Uint32 atomic_add(Native_atomic_counter & counter, Uint32 i)
{
    return _InterlockedExchangeAdd(&counter, i) + i;
}

MI_FORCE_INLINE Uint32 atomic_sub(Native_atomic_counter & counter, Uint32 i)
{
    return _InterlockedExchangeAdd(&counter, -static_cast<Sint32>(i)) - i;
}

MI_FORCE_INLINE Uint32 atomic_inc(Native_atomic_counter & counter)
{
    return _InterlockedExchangeAdd(&counter, 1L) + 1L;
}

MI_FORCE_INLINE Uint32 atomic_dec(Native_atomic_counter & counter)
{
    return _InterlockedExchangeAdd(&counter, -1L) - 1L;
}

MI_FORCE_INLINE Uint32 atomic_post_inc(Native_atomic_counter & counter)
{
    return _InterlockedExchangeAdd(&counter, 1L);
}

MI_FORCE_INLINE Uint32 atomic_post_dec(Native_atomic_counter & counter)
{
    return _InterlockedExchangeAdd(&counter, -1L);
}

MI_FORCE_INLINE Uint32 atomic_post_add(Native_atomic_counter& counter, Uint32 i)
{
    return _InterlockedExchangeAdd(&counter, i);
}

MI_FORCE_INLINE Uint32 atomic_post_sub(Native_atomic_counter& counter, Uint32 i)
{
    return _InterlockedExchangeAdd(&counter, -static_cast<int>(i));
}

}}} // MI::STLEXT::IMPL

///////////////////////////////////////////////////////////////////////////////
// Generic Implementation for the GNU C++ Compiler
///////////////////////////////////////////////////////////////////////////////

#elif defined(__GNUG__)

#if !defined(__ia64__) && !defined(__APPLE__) && !defined(__PPC64__) && !defined(__PPC64__) && !defined(__aarch64__)
#  warning "No native atomic counting implementation available."
#  warning "Using GCC library code as a fallback."
#endif

#include <ext/atomicity.h>             // _Atomic_word and __exchange_and_add()

namespace MI { namespace STLEXT { namespace IMPL {

#if (__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
using __gnu_cxx::__exchange_and_add;
#endif

typedef _Atomic_word volatile Native_atomic_counter;

MI_FORCE_INLINE void create_counter(Native_atomic_counter & counter, Uint32 i)
{
    counter = i;
}

MI_FORCE_INLINE void destroy_counter(Native_atomic_counter & counter)
{
}

MI_FORCE_INLINE Uint32 get_counter(Native_atomic_counter const & counter)
{
    return counter;
}

MI_FORCE_INLINE Uint32 atomic_add(Native_atomic_counter & counter, Uint32 i)
{
    return __exchange_and_add(&counter, static_cast<int>(i)) + i;
}

MI_FORCE_INLINE Uint32 atomic_sub(Native_atomic_counter & counter, Uint32 i)
{
    return __exchange_and_add(&counter, -static_cast<int>(i)) - i;
}

MI_FORCE_INLINE Uint32 atomic_inc(Native_atomic_counter & counter)
{
    return atomic_add(counter, 1u);
}

MI_FORCE_INLINE Uint32 atomic_dec(Native_atomic_counter & counter)
{
    return atomic_sub(counter, 1u);
}

MI_FORCE_INLINE Uint32 atomic_post_inc(Native_atomic_counter & counter)
{
    return __exchange_and_add(&counter, 1);
}

MI_FORCE_INLINE Uint32 atomic_post_dec(Native_atomic_counter & counter)
{
    return __exchange_and_add(&counter, -1);
}

MI_FORCE_INLINE Uint32 atomic_post_add(Native_atomic_counter& counter, Uint32 i)
{
    return __exchange_and_add(&counter, static_cast<int>(i));
}

MI_FORCE_INLINE Uint32 atomic_post_sub(Native_atomic_counter& counter, Uint32 i)
{
    return __exchange_and_add(&counter, -static_cast<int>(i));
}

}}} // MI::STLEXT::IMPL

///////////////////////////////////////////////////////////////////////////////
// Generic implementation using pthreads mutex-locking
///////////////////////////////////////////////////////////////////////////////

#elif defined(_REENTRANT) || defined(_THREAD_SAFE) || defined(_PTHREADS)

#if !defined(IRIX)
#  warning "No lock-free atomic counting implementation available."
#  warning "Using pthreads as a fallback."
#endif

#include <utility>          // std::pair<>
#include <pthread.h>                            // POSIX threads

namespace MI { namespace STLEXT { namespace IMPL {

typedef std::pair<pthread_mutex_t, Uint32 volatile> Native_atomic_counter;

MI_FORCE_INLINE void create_counter(Native_atomic_counter & counter, Uint32 i)
{
    pthread_mutex_init(&counter.first, 0);
    counter.second = i;
}

MI_FORCE_INLINE void destroy_counter(Native_atomic_counter & counter)
{
    pthread_mutex_destroy(&counter.first);
}

MI_FORCE_INLINE Uint32 get_counter(Native_atomic_counter const & counter)
{
    return counter.second;
}

MI_FORCE_INLINE Uint32 atomic_add(Native_atomic_counter & counter, Uint32 i)
{
    pthread_mutex_lock(&counter.first);
    Uint32 const retval = (counter.second += i);
    pthread_mutex_unlock(&counter.first);
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_sub(Native_atomic_counter & counter, Uint32 i)
{
    pthread_mutex_lock(&counter.first);
    Uint32 const retval = (counter.second -= i);
    pthread_mutex_unlock(&counter.first);
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_inc(Native_atomic_counter & counter)
{
    return atomic_add(counter, 1u);
}

MI_FORCE_INLINE Uint32 atomic_dec(Native_atomic_counter & counter)
{
    return atomic_sub(counter, 1u);
}

MI_FORCE_INLINE Uint32 atomic_post_inc(Native_atomic_counter & counter)
{
    pthread_mutex_lock(&counter.first);
    Uint32 const retval = counter.second++;
    pthread_mutex_unlock(&counter.first);
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_dec(Native_atomic_counter & counter)
{
    pthread_mutex_lock(&counter.first);
    Uint32 const retval = counter.second--;
    pthread_mutex_unlock(&counter.first);
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_add(Native_atomic_counter& counter, Uint32 i)
{
    pthread_mutex_lock(&counter.first);
    Uint32 const retval = counter.second;
    counter.second += i;
    pthread_mutex_unlock(&counter.first);
    return retval;
}

MI_FORCE_INLINE Uint32 atomic_post_sub(Native_atomic_counter& counter, Uint32 i)
{
    pthread_mutex_lock(&counter.first);
    Uint32 const retval = counter.second;
    counter.second -= i;
    pthread_mutex_unlock(&counter.first);
    return retval;
}

}}} // MI::STLEXT::IMPL

///////////////////////////////////////////////////////////////////////////////
// Report an error if no implementation was selected
///////////////////////////////////////////////////////////////////////////////

#else
#  error "No atomic counter implementation available for this platform!"
#endif

#endif // BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_INLINE_H
