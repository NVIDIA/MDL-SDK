/******************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/**
    \file
    \brief  The definition of the Allocatable class and allocators. 
            Global new and delete are redefined.
*/

#ifndef BASE_LIB_MEM_ALLOCATABLE_H
#define BASE_LIB_MEM_ALLOCATABLE_H

#include <cstddef>
#include <new>


#ifdef __cplusplus
#ifndef DEFINED_MI_MEM_ALLOCATABLE
#define DEFINED_MI_MEM_ALLOCATABLE


namespace MI {
namespace MEM {

/// Allocator class implementation based on new/delete[char].
class Allocator_impl_new
{
#ifndef MI_MEM_TRACKER
public:
    static void* amalloc(size_t sz)
    { return ::operator new[] (sz); }

    static void* amalloc(const char*, int, size_t sz)
    { return ::operator new[] (sz); }

    static void afree(void* p)
    {::operator delete[] (p); }

    static void afree(const char*, int, void *p)
    {::operator delete[] (p); }
#endif // MI_MEM_TRACKER
};

/// The base class for all allocatable objects.
///
/// This class allows the introduction of system-wide memory handling
/// strategies (at least for the inheriting objects) without overloading global
/// \c new and \c delete.
/// Users should derive objects which are allocated with \c new from this class.

template < class Impl >
class Tallocatable : public Impl
{
#ifndef MI_MEM_TRACKER
public:
    /// Class-specific dynamic-storage handling functions.
    /// Note that all those members come in pairs, where each allocation function requires
    /// a corresponding deallocation function. For sematic issues, look at 3.7.3.1 and 3.7.3.2,
    /// for the mechanisms behind it 5.3.4, 5.3.5 and 12.5. Please note that the matching placement
    /// deallocation functions required for one case only - when the corresponding allocation
    /// function throws an exception 5.3.4(17). Not providing such a matching deallocation
    /// function results in C4291 "no matching operator delete found; memory will not be freed
    /// if initialization throws an exception".

    void* operator new ( size_t	sz )    { return Impl::amalloc(sz); }
    void* operator new[] ( size_t sz )  { return Impl::amalloc(sz); }

    void operator delete ( void* p )    { Impl::afree(p); }
    void operator delete[] ( void* p )  { Impl::afree(p); }


    void* operator new ( size_t	sz, const char* file, int line )
    { return Impl::amalloc(file, line, sz); }

    void* operator new[] ( size_t sz, const char* file, int line )
    { return Impl::amalloc(file, line, sz); }

    /// The placement allocation function.
    void* operator new (size_t sz, void* a)  { return a; }

    /// The placement deallocation function.
    void operator delete ( void* a, void* ) {}
    void operator delete[] ( void* a, void* ) {}

    void operator delete ( void* p, const char* file, int line)
    { Impl::afree(file, line, p); }

    void operator delete[] (void* p, const char* file, int line)
    { Impl::afree(file, line, p); }

#endif // MI_MEM_TRACKER
};


/// The Allocatable base class.
typedef Tallocatable< Allocator_impl_new >  Allocatable;


#endif  // DEFINED_MI_MEM_ALLOCATABLE
#endif  // __cplusplus



/// Templates for allocation and deallocation of array with given allocator.


template<typename T>
inline T* new_array(size_t size) { return (T*) (new T[size]); } //-V572 PVS

template<typename T>
inline void delete_array(T* ptr) { delete[] (ptr); }


/// A simple wrapper around a class attaching the allocatable interface.
/// This is a handy shortcut for deriving existing classes (like STL container)
/// from the Allocatable interface.
/// Currently, constructors with up to 5 parameters are forwarded.

template<typename T>
class Make_allocatable
  : public T
  , public MI::MEM::Allocatable
{
  public:
    /// "forward" default constructor
    Make_allocatable() : T() { }

    /// speculatively forward constructors with 1 argument
    template <typename A1>
    Make_allocatable(A1 a1) : T(a1) { }

    /// speculatively forward contructors with 2 arguments
    template <typename A1, typename A2>
    Make_allocatable(A1 a1, A2 a2) : T(a1, a2) { }

    /// speculatively forward contructors with 3 arguments
    template <typename A1, typename A2, typename A3>
    Make_allocatable(A1 a1, A2 a2, A3 a3) : T(a1, a2, a3) { }

    /// speculatively forward contructors with 4 arguments
    template <typename A1, typename A2, typename A3, typename A4>
    Make_allocatable(A1 a1, A2 a2, A3 a3, A4 a4) : T(a1, a2, a3, a4) { }

    /// speculatively forward contructors with 5 arguments
    template <typename A1, typename A2, typename A3, typename A4, typename A5>
    Make_allocatable(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) : T(a1, a2, a3, a4, a5) { }

    /// speculatively forward contructors with 6 arguments
    template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
    Make_allocatable(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) : T(a1, a2, a3, a4, a5, a6) { }

    // constructors with more arguments could be added on demand


    /// const referece to the base class view
    const T& base() const { return *static_cast<const T*>(this); }

    /// referece to the base class view
    T& base() { return *static_cast<T *>(this); }
      
};

}}

#endif	// BASE_LIB_MEM_ALLOCATABLE_H
