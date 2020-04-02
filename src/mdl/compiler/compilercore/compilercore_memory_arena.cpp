/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <mi/base/iallocator.h>
#include <mi/base/handle.h>
#include <mi/mdl/mdl_iowned.h>

#include "compilercore_memory_arena.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

// The base destructor of Interface_owned interfaces.
Interface_owned::~Interface_owned()
{
}


template<typename T>
static inline size_t align_of() {
    struct X {
        char a;
        T    align;
    };
    return size_t((char *)&(((struct X*)0)->align) - (char *)0);
}

// Alignment helper
struct Struct_align
{
    union {
        void *p;
        int i;
        long l;
        long long ll;
        double d;
        long double ld;
    } u;
};

static inline size_t align(size_t size, size_t a)
{
    MDL_ASSERT((a & (a-1)) == 0);
    return (size + a - 1) & ~(a - 1);
}

template<typename T>
static inline T *align(T *p, size_t a)
{
    size_t s = align(p - (T*)0, a);

    return (T *)0 + s;
}

static inline size_t start_offset(unsigned char *p, size_t a)
{
    size_t adr = p - (unsigned char *)0;
    return (0 - adr) & (a-1);
}

Memory_arena::Memory_arena(IAllocator *alloc, size_t chunk_size)
: m_alloc(alloc, mi::base::DUP_INTERFACE)
, m_chunk_size(chunk_size)
, m_chunks(NULL)
, m_next(NULL)
, m_curr_size(0)
{
    MDL_ASSERT(alloc && chunk_size > 16);
}
/// Destructs the memory arena and frees ALL memory.
Memory_arena::~Memory_arena()
{
    for (Header *p = m_chunks, *q; p != NULL; p = q) {
        q = p->next;

        m_alloc->free((void *)p);
    }
    m_chunks = NULL;
}

/// Allocates size bytes from the memory area.
void *Memory_arena::allocate(size_t o_size, size_t o_align)
{
    size_t a = o_align == 0 ? align_of<Struct_align>() : o_align;

    size_t ofs = start_offset(m_next, a);
    size_t size = o_size + ofs;

    if (size > m_curr_size) {
        // allocate a new chunk
        size_t load_ofs = align(((Header *)0)->load - (Byte *)0, a);

        size_t chunk_size = m_chunk_size;
        if (o_size + (a-1) > chunk_size - load_ofs)
            chunk_size = o_size + (a-1) + load_ofs;

        Header *h = (Header *)m_alloc->malloc(chunk_size);
        // printf("Allocated chunk %p\n", h);
        if (h == NULL)
            return NULL;
        h->next       = m_chunks;
        h->chunk_size = chunk_size;
        m_chunks      = h;

        m_next = align(h->load, a);
        ofs    = 0;
        size   = o_size;

        size_t lost = m_next - (Byte *)h;
        m_curr_size = chunk_size - lost;
    }

    void *res = m_next + ofs;
    m_next += size;

    MDL_ASSERT(m_curr_size >= size);
    m_curr_size -= size;

    MDL_ASSERT((((Byte *)res - (Byte *)0) & (a - 1)) == 0);
    return res;
}

// Drop the given object AND all later allocated objects from the arena.
void Memory_arena::drop(void *obj)
{
    if (obj == NULL) {
        // drop the whole
        for (Header *p = m_chunks; p != NULL; p = m_chunks) {
            m_chunks = p->next;
            m_alloc->free(p);
        }
        m_next      = NULL;
        m_curr_size = 0;
        return;
    }

    // check current chunk first
    if (m_chunks->load <= obj && obj <= m_next) {
        m_next = (Byte *)obj;
        m_curr_size = (char *)m_chunks + m_chunks->chunk_size - (char *)obj;
    } else {
        MDL_ASSERT(!(m_chunks->load <= obj && obj < m_chunks->load + m_chunk_size));

        // try to find the old chunk
        Header *stop = NULL;
        do {
            stop = m_chunks->next;
        } while (stop != NULL &&
                (stop->load > obj || ((char *)stop + stop->chunk_size) <= obj));

        // drop chunks until old one is reached
        if (stop != NULL) {
            Header *p = m_chunks;
            do {
                m_chunks = p->next;
                m_alloc->free(p);
            } while (m_chunks != stop);

            // now we could drop it in the current chunk
            m_next = (Byte *)obj;
            m_curr_size = (char *)m_chunks + m_chunks->chunk_size - (char *)obj;
        } else {
            MDL_ASSERT(!"dropped object from wrong Memory Arena");
        }
    }
}

// Check if an object lies in this memory arena.
bool Memory_arena::contains(void const *obj) const
{
    if (m_chunks->load <= obj && obj < m_next) {
        return true;
    } else {
        Header *stop = m_chunks;
        do {
            stop = stop->next;
        } while (stop != NULL &&
            (stop->load > obj || ((char *)stop + stop->chunk_size) <= obj));

        return stop != NULL;
    }
}

// Return the size of the allocated memory arena chunks.
size_t Memory_arena::get_chunks_size() const
{
    size_t size = 0;

    for (Header const *h = m_chunks; h != NULL; h = h->next) {
        size += h->chunk_size;
    }
    return size;
}

// Swap this memory arena content with another.
void Memory_arena::swap(Memory_arena &other)
{
    std::swap(m_alloc,      other.m_alloc);
    std::swap(m_chunk_size, other.m_chunk_size);
    std::swap(m_chunks,     other.m_chunks);
    std::swap(m_next,       other.m_next);
    std::swap(m_curr_size,  other.m_curr_size);
}

// Put a C-string into the memory arena.
char *Arena_strdup(Memory_arena &arena, char const *s)
{
    if (s == NULL)
        return NULL;
    size_t len = strlen(s) + 1;

    char *res = reinterpret_cast<char *>(arena.allocate(len, 1));
    if (res) {
        memcpy(res, s, len);
    }
    return res;
}

// Put a blob into the memory arena.
void *Arena_memdup(Memory_arena &arena, void const *mem, size_t size)
{
    if (mem == NULL)
        return NULL;

    char *res = reinterpret_cast<char *>(arena.allocate(size));
    if (res) {
        memcpy(res, mem, size);
    }
    return res;
}

}  // mdl
}  // mi
