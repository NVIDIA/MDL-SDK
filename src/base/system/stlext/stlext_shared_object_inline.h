/******************************************************************************
 * Copyright (c) 2006-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Internal helper class for \c std::shared_ptr.

#ifndef BASE_SYSTEM_STLEXT_SHARED_OBJECT_H
#define BASE_SYSTEM_STLEXT_SHARED_OBJECT_H

#include "i_stlext_atomic_counter.h"
#include "i_stlext_concepts.h"
#include "i_stlext_no_unused_variable_warning.h"

namespace MI { namespace STLEXT { namespace IMPL {

/** \internal
 *
 * A Shared_object<T> represents a reference-counted T pointer with a custom
 * deleter attached to it (through a virtual destructor). The fact that the
 * deleter is embedded into the instance allows Shared_object to be casted to
 * other derived types.
 */

template <class T>
struct Shared_object_base : private Abstract_interface
{
    T * const           m_ptr;          // is never NULL
    Atomic_counter      m_ref_count;

    Shared_object_base(T & obj) : m_ptr(&obj), m_ref_count(1)
    {
    }

    size_t ref_count() { return m_ref_count; }

    /// Store a down-casted copy of \c m_ptr at \c dst_ptr.
    template <class U>
    void assign_to(Shared_object_base<U> * & dst_ptr)
    {
        T * const t_ptr( 0 );
        U * const u_ptr( t_ptr ); // Casting from 'T*' to 'U*' must be legal.
        no_unused_variable_warning_please(u_ptr);
        dst_ptr = reinterpret_cast<Shared_object_base<U>*>(this);
    }
};

template <class T, class D>
class Shared_object : public Shared_object_base<T>
{
public:
    Shared_object(T & obj, D const & deleter = D())
    : Shared_object_base<T>(obj), m_deleter(deleter)
    {
    }

    ~Shared_object()
    {
        m_deleter(this->m_ptr);
    }

private:
    D  m_deleter;
};

}}} // MI::STLEXT::IMPL

#endif // BASE_SYSTEM_STLEXT_SHARED_OBJECT_H
