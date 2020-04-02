/***************************************************************************************************
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
 **************************************************************************************************/
/// \file mi/base/default_allocator.h
/// \brief Default allocator implementation based on global new and delete.
///
/// See \ref mi_base_iinterface.

#ifndef MI_BASE_DEFAULT_ALLOCATOR_H
#define MI_BASE_DEFAULT_ALLOCATOR_H

#include <mi/base/types.h>
#include <mi/base/iallocator.h>
#include <mi/base/interface_implement.h>
#include <new>

namespace mi
{

namespace base
{

/** \addtogroup mi_base_iallocator
@{
*/

/** A default allocator implementation based on global new and delete.

    This implementation realizes the singleton pattern. An instance of the
    default allocator can be obtained through the static inline method 
    #mi::base::Default_allocator::get_instance().

       \par Include File:
       <tt> \#include <mi/base/default_allocator.h></tt>

*/
class Default_allocator : public Interface_implement_singleton<IAllocator>
{
    Default_allocator() {}
    Default_allocator( const Default_allocator&) {}
public:

    /** Allocates a memory block of the given size.

        Implements #mi::base::IAllocator::malloc through a global non-throwing
        \c operator \c new call.

        \param size   The requested size of memory in bytes. It may be zero.
        \return       The allocated memory block.
    */
    virtual void* malloc(Size size) {
        // Use non-throwing new call, which may return NULL instead
        return ::new(std::nothrow) char[size];
    }

    /** Releases the given memory block. 

        Implements #mi::base::IAllocator::free through a global 
        \c operator \c delete call.

        \param  memory   A memory block previously allocated by a call to #malloc().
                         If \c memory is \c NULL, no operation is performed.
    */
    virtual void free(void* memory) {
        ::delete[] reinterpret_cast<char*>(memory);
    }

    /// Returns the single instance of the default allocator.
    static IAllocator* get_instance() {
        // We claim that this is multithreading safe because the
        // Default_allocator has an empty default constructor.
        // Whatever number of threads gets into the constructor, there
        // should be no way to screw up the initialization in each
        // thread. The optimizer might even be able to eliminate all
        // code here.
        static Default_allocator allocator;
        return &allocator;
    }
};

/*@}*/ // end group mi_base_iallocator

} // namespace base
} // namespace mi

#endif // MI_BASE_DEFAULT_ALLOCATOR_H
