/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_ARRAY_REF_H
#define MDL_COMPILERCORE_ARRAY_REF_H 1

#include "compilercore_memory_arena.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

/// Helper class to handle array passing safely.
template<typename T>
class Array_ref {
public:
    typedef const T *iterator;
    typedef const T *const_iterator;

    /// Construct an empty array reference.
    /*implicit*/ Array_ref()
    : m_arr(NULL)
    , m_n(0)
    {
    }

    /// Construct from a pointer and a length.
    ///
    /// \param arr  the array
    /// \param n    length of the array
    Array_ref(T const *arr, size_t n)
    : m_arr(arr)
    , m_n(n)
    {
    }

    /// Construct from a start and an end pointer.
    ///
    /// \param begin  the start pointer
    /// \param end    the end pointer
    Array_ref(T const *begin, T const *end)
    : m_arr(begin)
    , m_n(end - begin)
    {
    }

    /// Construct from one element.
    ///
    /// \param elem  the element
    /*implicit*/ Array_ref(T const &elem)
    : m_arr(&elem)
    , m_n(1)
    {
    }

    /// Construct a descriptor from a C array.
    template <size_t N>
    /*implicit*/ Array_ref(T const (&arr)[N])
    : m_arr(arr), m_n(N)
    {
    }

    /// Construct a descriptor from a VLA.
    /*implicit*/ Array_ref(VLA<T> const &vla)
    : m_arr(vla.data()), m_n(vla.size())
    {
    }

    /// Construct a descriptor from a Small_VLA.
    template <size_t N>
    /*implicit*/ Array_ref(Small_VLA<T, N> const &vla)
    : m_arr(vla.data()), m_n(vla.size())
    {
    }

    /// Construct from an vector.
    template <typename A>
    /*implecit*/ Array_ref(std::vector<T, A> const &v)
    : m_arr(v.empty() ? NULL : &v[0]), m_n(v.size())
    {
    }

    /// Get the begin iterator.
    iterator begin() const { return m_arr; }

    /// Get the end iterator.
    iterator end() const { return m_arr + m_n; }

    /// Get the array size.
    size_t size() const { return m_n; }

    /// Index operator.
    T const &operator[](size_t i) const
    {
        MDL_ASSERT(i < m_n && "index out of bounds");
        return m_arr[i];
    }

    /// Chop off the first N elements of the array.
    Array_ref<T> slice(size_t n) const
    {
        MDL_ASSERT(n <= size() && "cannot chop more than whole array");
        return Array_ref<T>(&m_arr[n], m_n - n);
    }

    /// slice(n, m) - Chop off the first N elements of the array, and keep M
    /// elements in the array.
    Array_ref<T> slice(size_t N, size_t M) const
    {
        MDL_ASSERT(N + M <= size() && "Invalid size");
        return Array_ref<T>(data() + N, M);
    }

    /// Get the data.
    T const *data() const { return m_arr; }

private:
    T const *m_arr;
    size_t  m_n;
};

}  // mdl
}  // mi

#endif
