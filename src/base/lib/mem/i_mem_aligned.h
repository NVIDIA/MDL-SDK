/******************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_LIB_MEM_ALIGNED_H
#define BASE_LIB_MEM_ALIGNED_H

/// Portable alignment macro.
/// Also see alloc_aligned / free_aligned for dynamic memory allocations.
#ifndef MI_ALIGN16                                                                         
#ifdef WIN_NT                                                                               
#define MI_ALIGN16 __declspec(align(16))                                                   
#else                                                                                       
#define MI_ALIGN16 __attribute__ ((aligned(16)))                                           
#endif                                                                                      
#endif


#include <base/system/stlext/i_stlext_shared_array.h>

namespace MI {
namespace MEM {

/// Allocate an uninitialized block of memory with the given alignment (must be a multiple of
/// sizeof(size_t)). The caller needs to release this block of memory using \c free_aligned.
void* alloc_aligned(const size_t size, const size_t alignment);


/// Free a block of memory that was allocated using \c alloc_aligned.
void free_aligned(void* ptr);


/// Deleter that calls \free_aligned.
/// Note that no destructor is called!
template <typename T>
struct Aligned_POD_deleter
{
    void operator()(T* const ptr) const
    {
        free_aligned(static_cast<void*>(ptr));
    }
};


/// Create a POD array with a smart pointer that calls
/// \c free_aligned for deletion. Note that the memory
/// is not initialized and no constructors / destructors
/// are called!
template <typename T>
inline STLEXT::Shared_array<T> alloc_aligned_POD(
    const size_t num_elements,
    const size_t alignment)
{
    return STLEXT::Shared_array<T>(
        static_cast<T*>(alloc_aligned(sizeof(T) * num_elements, alignment)),
        Aligned_POD_deleter<T>());
}

} // namespace MEM
} // namespace MI

#include <memory>
#include <new>

namespace MI {
namespace MEM {

template<class T, size_t alignment>
struct aligned_allocator : std::allocator<T>
{
#if defined(_MSC_VER)
#if (_MSC_VER >= 1900)
    aligned_allocator() noexcept {}
    template <typename U>
    aligned_allocator(const aligned_allocator<U, alignment>&) noexcept {}
#endif
#endif
    template<class U>
    struct rebind { typedef aligned_allocator<U, alignment> other; };

    typedef std::allocator<T> base;

    typedef typename base::pointer pointer;
    typedef typename base::size_type size_type;

    pointer allocate(size_type n)
    {
	if(pointer p = (pointer)alloc_aligned(n * sizeof(T), alignment))
	    return p;
	throw std::bad_alloc();
    }

    pointer allocate(size_type n, void const*)
    {
	return this->allocate(n);
    }

    void deallocate(pointer p, size_type)
    {
	if( p != NULL )
	    free_aligned(p);
    }
};

} // namespace MEM
} // namespace MI

#endif // BASE_LIB_MEM_ALIGNED_H
