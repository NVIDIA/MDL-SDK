/***************************************************************************************************
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_UTIL_H
#define BASE_DATA_DBLIGHT_DBLIGHT_UTIL_H

#include <chrono>
#include <iosfwd>

#include <boost/core/noncopyable.hpp>

#include <mi/base/types.h>

/// Enable this macro to collect some statistics.
///
/// The statistics are dumped when the database is destroyed.
// #define DBLIGHT_ENABLE_STATISTICS

namespace MI {

namespace DBLIGHT {

void dump_statistics( std::ostream& s, mi::Uint32 next_tag);

struct Statistics_data
{
    size_t m_count = 0;
    double m_time  = 0.0;
};

class Statistics_helper : private boost::noncopyable
{
#ifdef DBLIGHT_ENABLE_STATISTICS
public:
    Statistics_helper( Statistics_data& data);
    ~Statistics_helper();

private:
    Statistics_data& m_data;
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
#else // DBLIGHT_ENABLE_STATISTICS
public:
    Statistics_helper( Statistics_data& /*data*/) { }
#endif // DBLIGHT_ENABLE_STATISTICS
};

extern Statistics_data g_name_to_tag;
extern Statistics_data g_tag_to_name;
extern Statistics_data g_access;
extern Statistics_data g_edit;
extern Statistics_data g_finish_edit;
extern Statistics_data g_store;
extern Statistics_data g_remove;
extern Statistics_data g_commit;
extern Statistics_data g_abort;
extern Statistics_data g_get_tag_reference_count;
extern Statistics_data g_get_tag_is_removed;
extern Statistics_data g_lookup_info_by_tag;
extern Statistics_data g_lookup_info_by_name;

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_UTIL_H
