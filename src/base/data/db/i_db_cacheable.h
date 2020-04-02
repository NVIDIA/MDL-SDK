/***************************************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file

#ifndef BASE_DATA_DB_I_CACHE_H
#define BASE_DATA_DB_I_CACHE_H

#include <mi/base/atom.h>
#include <base/system/main/types.h>
#include <base/system/stlext/i_stlext_concepts.h>
#include <base/lib/cont/i_cont_dlist.h>

namespace MI {

namespace DBNR { class Cache; }

namespace DB {

/// The base class of objects managed by the #Cache class.
class Cacheable : public STLEXT::Non_copyable
{
public:
    /// Constructor.
    ///
    /// The initial reference count is 1.
    ///
    /// \param cache   The #Cache instance this cacheable belongs to.
    Cacheable( DBNR::Cache* cache);

    /// Destructor.
    virtual ~Cacheable();

    /// Returns the current reference count.
    virtual Uint32 get_ref_count() const { return static_cast<Uint32>( m_reference_count); }

    /// Offloads the cacheable and returns the difference in memory usages (in bytes, typically
    /// negative).
    ///
    /// This method is invoked via #Cache_module::offload() for cacheables that are selected
    /// based on an LRU strategy among all flushable cacheables.
    virtual ptrdiff_t offload() = 0;

    /// The #Cache class may access the private members, but no-one else.
    friend class DBNR::Cache;

private:
    /// The reference count of the cacheable.
    ///
    /// \em Not protected by #m_cache->m_lock.
    mi::base::Atom32 m_reference_count;

    /// The #Cache instance this cacheable belongs to.
    DBNR::Cache* m_cache;

    /// The link for #Cache_module_impl::m_list.
    ///
    /// Protected by #m_cache->m_lock.
    CONT::Dlist_link<Cacheable> m_list_link;

    /// Indicates whether the cacheable is in the list of flushable cacheables.
    ///
    /// A cacheable is appended to the list if the list reference count is decremented from 1 to 0.
    /// It is removed from the list if the list reference count is incremented from 0 to 1 or if it
    /// is flushed during #offload().
    ///
    /// Protected by #m_cache->m_lock.
    bool m_in_list;

    /// The reference count of the cacheable used for list addition/removal decisions.
    ///
    /// This value might be temporarily different from #m_reference_count.
    ///
    /// Protected by #m_cache->m_lock.
    mi::Uint32 m_list_reference_count;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_CACHE_H
