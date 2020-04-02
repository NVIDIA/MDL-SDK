/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_CODE_CACHE_H
#define MDL_COMPILERCORE_CODE_CACHE_H 1

#include <cstring>

#include <mi/base/lock.h>
#include <mi/mdl/mdl_code_generators.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// The code cache helper class.
class Code_cache : public Allocator_interface_implement<mi::mdl::ICode_cache>
{
    typedef Allocator_interface_implement<mi::mdl::ICode_cache> Base;

    class Key {
        friend class Code_cache;
    public:
        /// Constructor.
        Key(unsigned char const key[16])
        {
            // copy the key
            memcpy(m_key, key, sizeof(m_key));
        }

        bool operator <(Key const &other) const
        {
            return memcmp(m_key, other.m_key, sizeof(m_key)) < 0;
        }

        bool operator ==(Key const &other) const
        {
            return memcmp(m_key, other.m_key, sizeof(m_key)) == 0;
        }

    private:
        unsigned char m_key[16];
    };

    class Cache_entry : public mi::mdl::ICode_cache::Entry {
        typedef mi::mdl::ICode_cache::Entry Base;
        friend class Code_cache;
        friend class Cache_entry_less;
    public:
        /// Constructor.
        Cache_entry(
            IAllocator          *alloc,
            Base const          &entry,
            unsigned char const key[16]);

        /// Destructor.
        ~Cache_entry();

    private:
        Key m_key;

        IAllocator  *m_alloc;
        Cache_entry *m_prev;
        Cache_entry *m_next;
    };

    class Cache_entry_less {
    public:
        bool operator() (Cache_entry const *a, Cache_entry const *b)
        {
            return a->m_key < b->m_key;
        }
    };

public:
    // Lookup a data blob.
    Entry const *lookup(unsigned char const key[16]) const MDL_FINAL;

    // Enter a data blob.
    bool enter(unsigned char const key[16], Entry const &entry) MDL_FINAL;

private:
    /// Create a new entry and put it in front.
    /// Assumes that current size has already been updated.
    Cache_entry *new_entry(mi::mdl::ICode_cache::Entry const &entry, unsigned char const key[16]);

    /// Remove an entry from the list.
    void remove_from_list(Cache_entry &entry) const;

    /// Move an entry to front.
    void to_front(Cache_entry &entry) const;

    /// Compare an entry with a key.
    static int cmp(Cache_entry const &entry, unsigned char const key[16])
    {
        return memcmp(entry.m_key.m_key, key, sizeof(entry.m_key));
    }

    // Drop entries from the end until size is reached.
    void strip_size();

public:
    /// Constructor.
    Code_cache(
        IAllocator *alloc,
        size_t     max_size);

    /// Destructor.
    virtual ~Code_cache();

private:
    mutable mi::base::Lock m_cache_lock;

    mutable Cache_entry *m_head;
    mutable Cache_entry *m_tail;

    typedef map<Key, Cache_entry *>::Type Search_map;

    /// The map of all cache entry to speed up searches.
    mutable Search_map m_search_map;

    /// Maximum size of this cache object.
    size_t m_max_size;

    /// Current size.
    size_t m_curr_size;
};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_CODE_CACHE_H
