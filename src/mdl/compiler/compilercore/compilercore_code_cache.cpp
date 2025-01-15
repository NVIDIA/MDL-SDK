/******************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_code_cache.h"

#include <cstdint>

namespace mi {
namespace mdl {

// Constructor.
Code_cache::Cache_entry::Cache_entry(
    IAllocator          *alloc,
    Base const          &entry,
    unsigned char const key[16])
: Base(entry)
, m_key(key)
, m_alloc(alloc)
, m_prev(NULL)
, m_next(NULL)
{
    Allocator_builder builder(alloc);

    // copy all data
    size_t size = entry.get_cache_data_size();

    char *blob = builder.alloc<char>(size);
    char **mapped = (char **)blob;
    char *data_area = blob + entry.mapped_string_size * sizeof(char *);
    Func_info *func_info_area = (Func_info *)(data_area + entry.mapped_string_data_size);
    char *func_info_data_area = (char *)(func_info_area + entry.func_info_size);

    blob = func_info_data_area + entry.func_info_string_data_size;

    char *seg = blob + entry.code_size;
    char *layout = seg + entry.const_seg_size;

    if (entry.code_size > 0)
        memcpy(blob, entry.code, entry.code_size);

    if (entry.const_seg_size > 0)
        memcpy(seg, entry.const_seg, entry.const_seg_size);

    if (entry.arg_layout_size > 0)
        memcpy(layout, entry.arg_layout, entry.arg_layout_size);

    if (entry.mapped_string_data_size > 0) {
        char *p = data_area;
        for (size_t i = 0; i < entry.mapped_string_size; ++i) {
            size_t l = strlen(entry.mapped_strings[i]);

            memcpy(p, entry.mapped_strings[i], l + 1);
            mapped[i] = p;
            p += l + 1;
        }
    }

    if (entry.func_info_size > 0) {
        Func_info *cur_info = func_info_area;
        char *p = func_info_data_area;
        for (size_t i = 0; i < entry.func_info_size; ++i, ++cur_info) {
            size_t len = strlen(entry.func_infos[i].name);
            memcpy(p, entry.func_infos[i].name, len + 1);
            cur_info->name = p;
            p += len + 1;

            cur_info->dist_kind = entry.func_infos[i].dist_kind;
            cur_info->func_kind = entry.func_infos[i].func_kind;

            for (int j = 0; j < int(mi::mdl::IGenerated_code_executable::PL_NUM_LANGUAGES); ++j) {
                len = strlen(entry.func_infos[i].prototypes[j]);
                memcpy(p, entry.func_infos[i].prototypes[j], len + 1);
                cur_info->prototypes[j] = p;
                p += len + 1;
            }

            cur_info->arg_block_index = entry.func_infos[i].arg_block_index;

            cur_info->num_df_handles = entry.func_infos[i].num_df_handles;
            if (cur_info->num_df_handles == 0) {
                cur_info->df_handles = NULL;
            } else {
                // align
                p = (char *)((uintptr_t(p) + sizeof(char *) - 1) & ~(sizeof(char *) - 1));

                cur_info->df_handles = (char const **) p;
                p += cur_info->num_df_handles * sizeof(char *);
                for (size_t j = 0; j < cur_info->num_df_handles; ++j) {
                    len = strlen(entry.func_infos[i].df_handles[j]);
                    memcpy(p, entry.func_infos[i].df_handles[j], len + 1);
                    cur_info->df_handles[j] = p;
                    p += len + 1;
                }
            }
        }
    }

    code = blob;
    const_seg = seg;
    arg_layout = layout;
    mapped_strings = mapped;
    func_infos = func_info_area;
}

// Destructor.
Code_cache::Cache_entry::~Cache_entry()
{
    Allocator_builder builder(m_alloc);

    char const *blob = (char const *)mapped_strings;

    builder.destroy(blob);
}


// Lookup a data blob.
Code_cache::Entry const *Code_cache::lookup(unsigned char const key[16]) const
{
    mi::base::Lock::Block block(&m_cache_lock);

    Search_map::const_iterator it = m_search_map.find(Key(key));
    if (it != m_search_map.end()) {
        // found
        Cache_entry *p = it->second;
        to_front(*p);
        return p;
    }
    return NULL;
}

// Enter a data blob.
bool Code_cache::enter(unsigned char const key[16], Entry const &entry)
{
    mi::base::Lock::Block block(&m_cache_lock);

    // don't try to enter it if it doesn't fit into the cache at all
    if (entry.get_cache_data_size() > m_max_size)
        return false;

    m_curr_size += entry.get_cache_data_size();
    strip_size();

    Cache_entry *res = new_entry(entry, key);

    m_search_map.insert(Search_map::value_type(res->m_key, res));
    return true;
}

// Create a new entry and put it in front.
// Assumes that current size has already been updated.
Code_cache::Cache_entry *Code_cache::new_entry(
    mi::mdl::ICode_cache::Entry const &entry, unsigned char const key[16])
{
    Allocator_builder builder(get_allocator());

    Cache_entry *res = builder.create<Cache_entry>(get_allocator(), entry, key);

    to_front(*res);

    return res;
}

// Remove an entry from the list.
void Code_cache::remove_from_list(Cache_entry &entry) const
{
    if (m_head == &entry)
        m_head = entry.m_next;
    if (m_tail == &entry)
        m_tail = entry.m_prev;

    if (entry.m_next != NULL)
        entry.m_next->m_prev = entry.m_prev;
    if (entry.m_prev != NULL)
        entry.m_prev->m_next = entry.m_next;

    entry.m_prev = entry.m_next = NULL;
}

// Move an entry to front.
void Code_cache::to_front(Cache_entry &entry) const
{
    remove_from_list(entry);

    entry.m_next = m_head;

    if (m_head != NULL)
        m_head->m_prev = &entry;

    m_head = &entry;

    if (m_tail == NULL)
        m_tail = &entry;
}

// Drop entries from the end until size is reached.
void Code_cache::strip_size()
{
    Allocator_builder builder(get_allocator());

    Cache_entry *next = NULL;
    for (Cache_entry *p = m_tail; p != NULL; p = next) {
        if (m_curr_size < m_max_size)
            break;

        next = p->m_prev;

        m_curr_size -= p->get_cache_data_size();
        m_search_map.erase(p->m_key);
        remove_from_list(*p);
        builder.destroy(p);
    }
}

// Constructor.
Code_cache::Code_cache(
    IAllocator *alloc,
    size_t     max_size)
: Base(alloc)
, m_cache_lock()
, m_head(NULL)
, m_tail(NULL)
, m_search_map(Search_map::key_compare(), alloc)
, m_max_size(max_size)
, m_curr_size(0)
{
}

// Destructor.
Code_cache::~Code_cache()
{
    Allocator_builder builder(get_allocator());

    m_search_map.clear();
    for (Cache_entry *n = NULL, *p = m_head; p != NULL; p = n) {
        n = p->m_next;
        builder.destroy(p);
    }
}

}  // mdl
}  // mi
