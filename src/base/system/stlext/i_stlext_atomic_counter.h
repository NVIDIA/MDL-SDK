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
/// \brief A portable interface to atomic counting operations.

#ifndef BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_H
#define BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_H

#include <base/system/main/types.h>             // Uint32 type
#include "i_stlext_concepts.h"                  // Non_copyable
#include "stlext_atomic_counter_inline.h"       // system-dependent implementation

namespace MI { namespace STLEXT {

/**
 * \brief Portable interface to atomic counting operations.
 *
 * The Atomic_counter class can be used like any other primitive integer type
 * except for two details: (a) it is non-copyable and (b) all operations occur
 * atomically. In other words, it is safe for two threads to use the same
 * atomic counter concurrently.
 * In addition, it provides the atomic \c post_add and \c post_sub functions.
 * These functions work like post increment and decrement, but accept
 * arbitrary values.
 */

class Atomic_counter : private Non_copyable
{
public:
    explicit Atomic_counter(Uint32 val = 0u)    { IMPL::create_counter(m_counter, val);    }
    ~Atomic_counter()                           { IMPL::destroy_counter(m_counter);        }
    operator Uint32 () const                    { return IMPL::get_counter(m_counter);     }

    Uint32 operator++ ()                        { return IMPL::atomic_inc(m_counter);      }
    Uint32 operator++ (int)                     { return IMPL::atomic_post_inc(m_counter); }

    Uint32 operator-- ()                        { return IMPL::atomic_dec(m_counter);      }
    Uint32 operator-- (int)                     { return IMPL::atomic_post_dec(m_counter); }

    Uint32 operator+= (Uint32 rhs)              { return IMPL::atomic_add(m_counter, rhs); }
    Uint32 operator-= (Uint32 rhs)              { return IMPL::atomic_sub(m_counter, rhs); }

    Uint32 post_add(Uint32 val)                 { return IMPL::atomic_post_add(m_counter, val); }
    Uint32 post_sub(Uint32 val)                 { return IMPL::atomic_post_sub(m_counter, val); }

private:
    IMPL::Native_atomic_counter m_counter;
};

}} // MI::STLEXT

#endif // BASE_SYSTEM_STLEXT_ATOMIC_COUNTER_H
