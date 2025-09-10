/***************************************************************************************************
 * Copyright (c) 2006-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \file  mi/base/atom.h
/// \brief 32-bit unsigned counter with atomic arithmetic, increments, and decrements.
///
/// See \ref mi_base_threads.

#ifndef MI_BASE_ATOM_H
#define MI_BASE_ATOM_H

#include <mi/base/config.h>
#include <mi/base/types.h>

// Select implementation to use.
#if !defined( MI_BASE_ATOM32_USE_ATOMIC) && !defined( MI_BASE_ATOM32_USE_ASSEMBLY)
#if defined( MI_ARCH_X86) && (defined( MI_COMPILER_GCC) || defined( MI_COMPILER_ICC))
#  define MI_BASE_ATOM32_USE_ASSEMBLY
#else
#  define MI_BASE_ATOM32_USE_ATOMIC
#endif
#endif

// Sanity check.
#if ! (defined( MI_BASE_ATOM32_USE_ATOMIC) ^ defined( MI_BASE_ATOM32_USE_ASSEMBLY))
#error Exactly one of MI_BASE_ATOM32_USE_ATOMIC and MI_BASE_ATOM32_USE_ASSEMBLY should be defined.
#endif

#if defined( MI_BASE_ATOM32_USE_ATOMIC)
#include <atomic>
#endif

namespace mi {

namespace base {

/** \addtogroup mi_base_threads
@{
*/

/// A 32-bit unsigned counter with atomic arithmetic, increments, and decrements.
class Atom32
{
public:
    /// The default constructor initializes the counter to zero.
    Atom32() : m_value( 0) { }

    /// This constructor initializes the counter to \p value.
    Atom32( const Uint32 value) : m_value( value) { }

#ifndef MI_BASE_ATOM32_USE_ASSEMBLY
    /// The copy constructor assigns the value of \p other to the counter.
    Atom32( const Atom32& other);

    /// Assigns the value of \p rhs to the counter.
    Atom32& operator=( const Atom32& rhs);
#endif

    /// Assigns \p rhs to the counter.
    Uint32 operator=( const Uint32 rhs) { m_value = rhs; return rhs; }

    /// Adds \p rhs to the counter.
    Uint32 operator+=( const Uint32 rhs);

    /// Subtracts \p rhs from the counter.
    Uint32 operator-=( const Uint32 rhs);

    /// Increments the counter by one (pre-increment).
    Uint32 operator++();

    /// Increments the counter by one (post-increment).
    Uint32 operator++( int);

    /// Decrements the counter by one (pre-decrement).
    Uint32 operator--();

    /// Decrements the counter by one (post-decrement).
    Uint32 operator--( int);

    /// Conversion operator to #mi::Uint32.
    operator Uint32() const { return m_value; }

    /// Assigns \p rhs to the counter and returns the old value of counter.
    Uint32 swap( const Uint32 rhs);

private:
#ifdef MI_BASE_ATOM32_USE_ASSEMBLY
    // The counter.
    volatile Uint32 m_value;
#else
    // The counter.
    std::atomic_uint32_t m_value;
#endif
};

#ifndef MI_FOR_DOXYGEN_ONLY

#ifdef MI_BASE_ATOM32_USE_ASSEMBLY

inline Uint32 Atom32::operator+=( const Uint32 rhs)
{
    Uint32 retval;
    asm volatile(
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        "addl %2,%0\n"
        : "=&r"( retval), "+m"( m_value)
        : "r"( rhs)
        : "cc"
        );
    return retval;
}

inline Uint32 Atom32::operator-=( const Uint32 rhs)
{
    Uint32 retval;
    asm volatile(
        "neg %2\n"
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        "addl %2,%0\n"
        : "=&r"( retval), "+m"( m_value)
        : "r"( rhs)
        : "cc", "%2"
        );
    return retval;
}

inline Uint32 Atom32::operator++()
{
    Uint32 retval;
    asm volatile(
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $1,%0\n"
        : "=&r"( retval), "+m"( m_value)
        :
        : "cc"
        );
    return retval;
}

inline Uint32 Atom32::operator++( int)
{
    Uint32 retval;
    asm volatile(
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r"( retval), "+m"( m_value)
        :
        : "cc"
        );
    return retval;
}

inline Uint32 Atom32::operator--()
{
    Uint32 retval;
    asm volatile(
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $-1,%0\n"
        : "=&r"( retval), "+m"( m_value)
        :
        : "cc"
        );
    return retval;
}

inline Uint32 Atom32::operator--( int)
{
    Uint32 retval;
    asm volatile(
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r"( retval), "+m"( m_value)
        :
        : "cc"
        );
    return retval;
}

inline Uint32 Atom32::swap( const Uint32 rhs)
{
    Uint32 retval;
    asm volatile(
    "0:\n"
        "movl %1,%0\n"
        "lock; cmpxchg %2,%1\n"
        "jnz 0b\n"
        : "=&a"( retval), "+m"( m_value)
        : "r"( rhs)
        : "cc"
        );
    return retval;
}

#else

inline Atom32::Atom32( const Atom32& other) : m_value( other.m_value.load()) { }

inline Atom32& Atom32::operator=( const Atom32& rhs)
{
    m_value = rhs.m_value.load();
    return *this;
}

inline Uint32 Atom32::operator+=( const Uint32 rhs)
{
    m_value += rhs;
    return m_value;
}

inline Uint32 Atom32::operator-=( const Uint32 rhs)
{
    m_value -= rhs;
    return m_value;
}

inline Uint32 Atom32::operator++()
{
    return ++m_value;
}

inline Uint32 Atom32::operator++( int)
{
    return m_value++;
}

inline Uint32 Atom32::operator--()
{
    return --m_value;
}

inline Uint32 Atom32::operator--( int)
{
    return m_value--;
}

inline Uint32 Atom32::swap( const Uint32 rhs)
{
    return m_value.exchange( rhs);
}

#endif // MI_BASE_ATOM32_USE_ASSEMBLY

#endif // !MI_FOR_DOXYGEN_ONLY

/*@}*/ // end group mi_base_threads

} // namespace base

} // namespace mi

#endif // MI_BASE_ATOM_H
