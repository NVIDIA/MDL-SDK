/**************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 *************************************************/

/* \file
 * \brief Thread-safe caching of strings.
 */

#ifndef BASE_DATA_IDATA_IDATA_STRING_CACHE_H
#define BASE_DATA_IDATA_IDATA_STRING_CACHE_H

#include <list>
#include <mutex>
#include <vector>

#include <boost/core/noncopyable.hpp>

namespace MI {

namespace IDATA {

/// Thread-safe cache for strings.
///
/// The cache returns a C-style char pointer for each added string. The pointer is guaranteed to be
/// valid for the lifetime of the cache, indepent of the lifetime of the instance of the string
/// that was originally added.
///
/// The cache does not detect duplicate strings.
class String_cache : public boost::noncopyable
{
public:

    /// Adds the string \p s to the cache (also supports \c nullptr).
    const char* add( const char* s);

    /// Adds the string \p s to the cache.
    const char* add( const std::string& s);

private:
    std::mutex m_lock;
    std::list<std::string> m_cache;
};

inline const char* String_cache::add( const char* s)
{
    if( !s)
        return nullptr;

    std::lock_guard lock( m_lock);
    return m_cache.emplace_back( s).c_str();
}

inline const char* String_cache::add( const std::string& s)
{
    std::lock_guard lock( m_lock);
    return m_cache.emplace_back( s).c_str();
}

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_STRING_CACHE_H
