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
/// \file mi/base/iallocator.h
/// \brief %Allocator interface class to dynamically allocate and deallocate memory.
///
/// See \ref mi_base_iinterface.

#ifndef MI_BASE_IALLOCATOR_H
#define MI_BASE_IALLOCATOR_H

#include <mi/base/types.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

namespace mi
{

namespace base
{

/** \defgroup mi_base_iallocator Memory Management
    \ingroup mi_base

    The memory management provides an interface for allocators including a default implementation.
    
    You can request dynamic memory from an allocator implementation through
    an #mi::base::IAllocator interface. A default implementation based on 
    global new and delete operators is #mi::base::Default_allocator. 

    You can adapt an allocator to become a standard STL allocator using
    #mi::base::Std_allocator.
*/

/** \addtogroup mi_base_iallocator
@{
*/

/** The %IAllocator interface class supports allocating and releasing memory dynamically.

    Different APIs allow to be configured with your own implementation of an
    allocator to override their internal implementation.

    \par Include File:
        <tt> \#include <mi/base/iallocator.h></tt>
*/
class IAllocator : public
    Interface_declare<0xa1836db8,0x6f63,0x4079,0x82,0x82,0xb3,0x5d,0x17,0x36,0x96,0xef>
{
public:
    /** Allocates a memory block of the given size.

        The memory must be aligned to an address which can accommodate any type of 
        object on the current platform.

        An allocation of zero bytes returns a valid non-null pointer which must
        be freed in the end. However, dereferencing this pointer gives undefined 
        behavior.

        This function can be called at any time from any thread, including concurrent 
        calls from several threads at the same time. 

        If the requested memory is not available this function returns \c NULL. 
        Some products give the stronger guarantee that allocation can never fail.
        (They can do this, for example, by flushing parts or in its extreme, 
        stop executing.) See the API documentation of the specific products
        for specific allocator interfaces or allocator documentation.

        \param size   The requested size of memory in bytes. It may be zero.
        \return       The allocated memory block.
    */
    virtual void* malloc(Size size) = 0;

    /** Releases the given memory block. 

        This function can be called at any time from any thread, including concurrent 
        calls from several threads at the same time.
       
        \param  memory   A memory block previously allocated by a call to #malloc().
                         If \c memory is \c NULL, no operation is performed.
    */
    virtual void free(void* memory) = 0;
};


/*@}*/ // end group mi_base_iallocator

} // namespace base
} // namespace mi

#endif // MI_BASE_IALLOCATOR_H
