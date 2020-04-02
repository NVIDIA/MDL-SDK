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

#ifndef MDL_COMPILERCORE_MALLOC_ALLOCATOR_H
#define MDL_COMPILERCORE_MALLOC_ALLOCATOR_H 1

#include <mi/base/lock.h>
#include <mi/base/interface_implement.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// A very simple allocator interface using malloc/free.
class MallocAllocator : public mi::base::Interface_implement<mi::mdl::IAllocator>
{
    typedef mi::base::Interface_implement<mi::mdl::IAllocator> Base;
public:
    void *malloc(mi::Size size) MDL_FINAL;

    void free(void *memory) MDL_FINAL;

    /// Decrements the reference count.
    Uint32 release() const MDL_FINAL;

    /// Create a net MallocAllocator.
    static MallocAllocator *create_instance();

private:
    /// Constructor.
    MallocAllocator();

    /// Destructor.
    ~MallocAllocator() MDL_FINAL;
};

/// An allocator based on new/delete.
class NewAllocator : public mi::base::Interface_implement<mi::base::IAllocator>
{
public:
    void *malloc(mi::Size size) MDL_FINAL;

    void free(void *memory) MDL_FINAL;

    /// Create a new NewAllocator.
    static NewAllocator *create_instance();

private:
    /// Constructor.
    NewAllocator();

    /// Destructor.
    ~NewAllocator() MDL_FINAL;
};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_MALLOC_ALLOCATOR_H
