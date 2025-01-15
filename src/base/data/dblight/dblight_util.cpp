/***************************************************************************************************
 * Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "dblight_util.h"

#include <ostream>

namespace MI {

namespace DBLIGHT {

THREAD::Lock g_stats_lock;
Statistics_data g_commit;
Statistics_data g_abort;
Statistics_data g_access;
Statistics_data g_edit;
Statistics_data g_finish_edit;
Statistics_data g_store;
Statistics_data g_localize;
Statistics_data g_remove;
Statistics_data g_name_to_tag;
Statistics_data g_tag_to_name;
Statistics_data g_get_class_id;
Statistics_data g_get_tag_privacy_level;
Statistics_data g_get_tag_store_level;
Statistics_data g_get_tag_reference_count;
Statistics_data g_get_tag_version;
Statistics_data g_can_reference_tag;
Statistics_data g_get_tag_is_removed;
Statistics_data g_lookup_info_by_tag;
Statistics_data g_lookup_info_by_name;
Statistics_data g_garbage_collection;

#define dump( x, y) \
    snprintf( buffer, sizeof( buffer), "%-44s %7zu, %6.3lf ms, %8.3lf μs\n", \
        x, y.m_count, 1000.0*y.m_time, (1000000.0*y.m_time)/(y.m_count>0?y.m_count:1)); \
    s << buffer;

void dump_statistics( std::ostream& s, mi::Uint32 next_tag)
{
#ifdef DBLIGHT_ENABLE_STATISTICS
    // Do not include g_lookup_info_by_tag, g_lookup_info_by_name, and g_garbage_collection which
    // are already included in other calls.
    double sum = g_commit.m_time
               + g_abort.m_time
               + g_access.m_time
               + g_edit.m_time
               + g_finish_edit.m_time
               + g_store.m_time
               + g_localize.m_time
               + g_remove.m_time
               + g_name_to_tag.m_time
               + g_tag_to_name.m_time
               + g_get_class_id.m_time
               + g_get_tag_privacy_level.m_time
               + g_get_tag_store_level.m_time
               + g_get_tag_reference_count.m_time
               + g_get_tag_version.m_time
               + g_can_reference_tag.m_time
               + g_get_tag_is_removed.m_time;

    char buffer[256];
    dump( "Transaction_impl::commit():", g_commit);
    dump( "Transaction_impl::abort():", g_abort);
    dump( "Transaction_impl::access_element():", g_access);
    dump( "Transaction_impl::edit_element():", g_edit);
    dump( "Transaction_impl::finish_edit():", g_finish_edit);
    dump( "Transaction_impl::store():", g_store);
    dump( "Transaction_impl::localize():", g_localize);
    dump( "Transaction_impl::remove():", g_remove);
    dump( "Transaction_impl::name_to_tag():", g_name_to_tag);
    dump( "Transaction_impl::tag_to_name():", g_tag_to_name);
    dump( "Transaction_impl::get_class_id():", g_get_class_id);
    dump( "Transaction_impl::get_tag_privacy_level():", g_get_tag_privacy_level);
    dump( "Transaction_impl::get_tag_store_level():", g_get_tag_store_level);
    dump( "Transaction_impl::get_tag_reference_count():", g_get_tag_reference_count);
    dump( "Transaction_impl::get_tag_version():", g_get_tag_version);
    dump( "Transaction_impl::can_reference_tag():", g_can_reference_tag);
    dump( "Transaction_impl::get_tag_is_removed():", g_get_tag_is_removed);
    s << std::endl;
    dump( "Info_manager::lookup_info_by_tag():", g_lookup_info_by_tag);
    dump( "Info_manager::lookup_info_by_name():", g_lookup_info_by_name);
    dump( "Info_manager::garbage_collection():", g_garbage_collection);
    s << std::endl;

    s << "sum: " << 1000.0 * sum << "ms" << std::endl;
    s << "next tag: " << next_tag << std::endl;
#endif // DBLIGHT_ENABLE_STATISTICS
}

#ifdef DBLIGHT_ENABLE_STATISTICS
Statistics_helper::Statistics_helper( Statistics_data& data)
  : m_data( data),
    m_start_time( std::chrono::system_clock::now())
{
}

Statistics_helper::~Statistics_helper()
{
    auto stop_time = std::chrono::system_clock::now();
    double duration = std::chrono::duration<double>( stop_time - m_start_time).count();
    // std::atomic<double> needs C++20, use a lock until then.
    THREAD::Block block( g_stats_lock);
    ++m_data.m_count;
    m_data.m_time += duration;
}
#endif // DBLIGHT_ENABLE_STATISTICS

} // namespace DBLIGHT

} // namespace MI
