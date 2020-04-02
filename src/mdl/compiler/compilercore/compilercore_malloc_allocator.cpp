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

#include "pch.h"

#include <cstdio>
#include <cstdlib>

#include <mi/base/interface_implement.h>

#include "compilercore_malloc_allocator.h"

namespace mi {
namespace mdl {

void *MallocAllocator::malloc(mi::Size size)
{
    void *res = ::malloc(size);
    if (res == NULL && size != 0) {
        fprintf(stderr, "*** Memory exhausted.\n");
        abort();
    }
    return res;
}

void MallocAllocator::free(void *memory)
{
    ::free(memory);
}

// Decrements the reference count.
Uint32 MallocAllocator::release() const
{
    Uint32 cnt = Base::release();
    if (cnt == 1) {
        // we have reached our self-reference, kick this object.
        this->~MallocAllocator();
        // don't do this normally
        ::free((void *)this);
    }
    return cnt;
}

// Create a new MallocAllocator.
MallocAllocator *MallocAllocator::create_instance() {
    MallocAllocator *p = (MallocAllocator *)::malloc(sizeof(MallocAllocator));

    if (p != NULL)
        new (p) MallocAllocator;
    return p;
}

MallocAllocator::MallocAllocator()
: Base(/*initial=*/2)
{
}

MallocAllocator::~MallocAllocator()
{
}

void *NewAllocator::malloc(mi::Size size)
{
    return new char[size];
}

void NewAllocator::free(void *memory)
{
    delete[] (char*) memory;
}


// Create a new NewAllocator.
NewAllocator *NewAllocator::create_instance() {
    return new NewAllocator;
}

NewAllocator::NewAllocator()
{
}

NewAllocator::~NewAllocator()
{
}

}  // mdl
}  // mi
