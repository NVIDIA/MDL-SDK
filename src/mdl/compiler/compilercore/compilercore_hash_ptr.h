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

#ifndef MDL_COMPILERCORE_HASH_PTR_H
#define MDL_COMPILERCORE_HASH_PTR_H 1

namespace mi {
namespace mdl {

/// A hash functor for pointers.
template <typename T>
class Hash_ptr {
public:
    size_t operator()(T const *p) const {
        size_t t = p - (T const *)0;
        return ((t) / (sizeof(size_t) * 2)) ^ (t >> 16);
    }
};

/// A hash functor for void pointers.
template <>
class Hash_ptr<void> {
public:
    size_t operator()(void const *p) const {
        size_t t = (char const *)p - (char const *)0;
        return (t)  ^ (t >> 4);
    }
};

/// A hash functor for void const pointers.
template <>
class Hash_ptr<void const> {
public:
    size_t operator()(void const *p) const {
        size_t t = (char const *)p - (char const *)0;
        return (t)  ^ (t >> 4);
    }
};

/// An Equal functor for pointers.
template <typename T>
struct Equal_ptr {
    inline unsigned operator()(const T *a, const T *b) const {
        return a == b;
    }
};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_HASH_PTR_H
