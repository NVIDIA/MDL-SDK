/***************************************************************************************************
 * Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/registry/i_config_registry.h>
#include <base/system/main/access_module.h>

namespace MI {

namespace DBLIGHT {

Flexible_lock::Flexible_lock( Lock_implementation impl)
  : m_lock_implementation( impl)
{
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();

    std::string lock_implementation;
    if( registry.get_value( "dblight_lock_implementation", lock_implementation)) {
        if( lock_implementation == "shared_lock")
            m_lock_implementation = SHARED_LOCK;
        else if( lock_implementation == "shared_lock_but_used_exclusively")
            m_lock_implementation = SHARED_LOCK_BUT_USED_EXCLUSIVELY;
        else if( lock_implementation == "exclusive_lock")
            m_lock_implementation = EXCLUSIVE_LOCK;
        else
            LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
                R"(Invalid value "%s" for debug option "dblight_lock_implementation".)",
                lock_implementation.c_str());
    }
    if( m_lock_implementation != SHARED_LOCK)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "Lock implementation set to %s.", lock_implementation.c_str());
}

std::string Flexible_lock::get_lock_impl_str() const
{
    switch( m_lock_implementation) {
        case SHARED_LOCK:                      return "shared_lock";
        case SHARED_LOCK_BUT_USED_EXCLUSIVELY: return "shared_lock_but_used_exclusively";
        case EXCLUSIVE_LOCK:                   return "exclusive_lock";
    }
    return {};
}

void Flexible_lock::lock_shared()
{
    if( m_lock_implementation == SHARED_LOCK)
        m_shared_lock.lock_shared();
    else if( m_lock_implementation == SHARED_LOCK_BUT_USED_EXCLUSIVELY)
        m_shared_lock.lock();
    else
        m_lock.lock();
}

bool Flexible_lock::try_lock_shared()
{
    if( m_lock_implementation == SHARED_LOCK)
        return m_shared_lock.try_lock_shared();
    else if( m_lock_implementation == SHARED_LOCK_BUT_USED_EXCLUSIVELY)
        return m_shared_lock.try_lock();
    else
        return m_lock.try_lock();
}

void Flexible_lock::unlock_shared()
{
    if( m_lock_implementation == SHARED_LOCK)
        m_shared_lock.unlock_shared();
    else if( m_lock_implementation == SHARED_LOCK_BUT_USED_EXCLUSIVELY)
        m_shared_lock.unlock();
    else
        m_lock.unlock();
}

void Flexible_lock::check_is_owned_shared()
{
    if( m_lock_implementation == SHARED_LOCK)
        m_shared_lock.check_is_owned_shared();
    else if( m_lock_implementation == SHARED_LOCK_BUT_USED_EXCLUSIVELY)
        m_shared_lock.check_is_owned();
    else
        m_lock.check_is_owned();
}

void Flexible_lock::lock()
{
    if( m_lock_implementation != EXCLUSIVE_LOCK)
        m_shared_lock.lock();
    else
        m_lock.lock();
}

bool Flexible_lock::try_lock()
{
    if( m_lock_implementation != EXCLUSIVE_LOCK)
        return m_shared_lock.try_lock();
    else
        return m_lock.try_lock();
}

void Flexible_lock::unlock()
{
    if( m_lock_implementation != EXCLUSIVE_LOCK)
        m_shared_lock.unlock();
    else
        m_lock.unlock();
}

void Flexible_lock::check_is_owned()
{
    if( m_lock_implementation != EXCLUSIVE_LOCK)
        m_shared_lock.check_is_owned();
    else
        m_lock.check_is_owned();
}

void Flexible_lock::check_is_owned_shared_or_exclusive()
{
    if( m_lock_implementation != EXCLUSIVE_LOCK)
        m_shared_lock.check_is_owned_shared_or_exclusive();
    else
        m_lock.check_is_owned();
}

const char* to_yes_no( bool value)
{
    return value ? "yes" : "no";
}

template <class T>
void dump_html_settings(
    std::ostream& s,
    const char* name,
    const T& value,
    const char* alignment,
    const char* suffix = "")
{
    s << "<tr>\n";
    s << "<td>" << name << "</td>\n";
    s << "<td align=" << alignment << ">" << value << suffix << "</td>\n";
    s << "</tr>\n";
}

void dump_html_bool_settings( std::ostream& s, const char* name, bool value)
{
    dump_html_settings( s, name, to_yes_no( value), "right", "");
}

void dump_html_double_setting( std::ostream& s, const char* name, double value, const char* suffix)
{
    dump_html_settings( s, name, value, "right", suffix);
}

void dump_html_size_t_setting( std::ostream& s, const char* name, size_t value)
{
    dump_html_settings( s, name, value, "right", "");
}

void dump_html_string_setting( std::ostream& s, const char* name, const std::string& value)
{
    dump_html_settings( s, name, value, "center", "");
}

THREAD::Lock g_stats_lock;

Statistics_data g_commit;
Statistics_data g_abort;
Statistics_data g_access_by_name;
Statistics_data g_access_by_tag;
Statistics_data g_edit_by_name;
Statistics_data g_edit_by_tag;
Statistics_data g_finish_edit;
Statistics_data g_store;
Statistics_data g_localize;
Statistics_data g_remove;
Statistics_data g_name_to_tag;
Statistics_data g_tag_to_name;
Statistics_data g_get_class_id;
Statistics_data g_get_tag_is_job;
Statistics_data g_get_tag_is_removed;
Statistics_data g_get_tag_privacy_level;
Statistics_data g_get_tag_store_level;
Statistics_data g_get_tag_reference_count;
Statistics_data g_get_tag_version;
Statistics_data g_invalidate_job_results;
Statistics_data g_advise;
Statistics_data g_can_reference_tag;
Statistics_data g_block_commit_or_abort;
Statistics_data g_unblock_commit_or_abort;
Statistics_data g_transaction_get_journal;

Statistics_data g_scope_get_journal;
Statistics_data g_scope_destructor;

Statistics_data g_lookup_info_by_name;
Statistics_data g_lookup_info_by_tag;
Statistics_data g_garbage_collection;

Statistics_data g_element_job_destructors;

#define dump( x, y) \
    snprintf( buffer, sizeof( buffer), "%-44s %7zu, %8.3lf ms, %10.3lf μs\n", \
        x, y.m_count, 1000.0*y.m_time, (1'000'000.0*y.m_time)/(y.m_count>0?y.m_count:1)); \
    s << buffer;

void dump_statistics( std::ostream& s, mi::Uint32 next_tag)
{
#ifdef DBLIGHT_ENABLE_STATISTICS
    // Do not include g_lookup_info_by_tag, g_lookup_info_by_name, g_element_job_destructors, and
    // g_garbage_collection which are already included in other calls (the last one only partially).
    double sum = g_commit.m_time
               + g_abort.m_time
               + g_access_by_name.m_time
               + g_access_by_tag.m_time
               + g_edit_by_name.m_time
               + g_edit_by_tag.m_time
               + g_finish_edit.m_time
               + g_store.m_time
               + g_localize.m_time
               + g_remove.m_time
               + g_name_to_tag.m_time
               + g_tag_to_name.m_time
               + g_get_class_id.m_time
               + g_get_tag_is_job.m_time
               + g_get_tag_is_removed.m_time
               + g_get_tag_privacy_level.m_time
               + g_get_tag_store_level.m_time
               + g_get_tag_reference_count.m_time
               + g_get_tag_version.m_time
               + g_invalidate_job_results.m_time
               + g_advise.m_time
               + g_can_reference_tag.m_time
               + g_block_commit_or_abort.m_time
               + g_unblock_commit_or_abort.m_time
               + g_transaction_get_journal.m_time
               + g_scope_get_journal.m_time
               + g_scope_destructor.m_time;

    char buffer[256];
    dump( "Transaction_impl::commit():", g_commit);
    dump( "Transaction_impl::abort():", g_abort);
    dump( "Transaction_impl::access_element() by name:", g_access_by_name);
    dump( "Transaction_impl::access_element() by tag:", g_access_by_tag);
    dump( "Transaction_impl::edit_element() by name:", g_edit_by_name);
    dump( "Transaction_impl::edit_element() by tag:", g_edit_by_tag);
    dump( "Transaction_impl::finish_edit():", g_finish_edit);
    dump( "Transaction_impl::store():", g_store);
    dump( "Transaction_impl::localize():", g_localize);
    dump( "Transaction_impl::remove():", g_remove);
    dump( "Transaction_impl::name_to_tag():", g_name_to_tag);
    dump( "Transaction_impl::tag_to_name():", g_tag_to_name);
    dump( "Transaction_impl::get_class_id():", g_get_class_id);
    dump( "Transaction_impl::get_tag_is_job():", g_get_tag_is_job);
    dump( "Transaction_impl::get_tag_is_removed():", g_get_tag_is_removed);
    dump( "Transaction_impl::get_tag_privacy_level():", g_get_tag_privacy_level);
    dump( "Transaction_impl::get_tag_store_level():", g_get_tag_store_level);
    dump( "Transaction_impl::get_tag_reference_count():", g_get_tag_reference_count);
    dump( "Transaction_impl::get_tag_version():", g_get_tag_version);
    dump( "Transaction_impl::invalidate_job_results():", g_invalidate_job_results);
    dump( "Transaction_impl::advise():", g_advise);
    dump( "Transaction_impl::can_reference_tag():", g_can_reference_tag);
    dump( "Transaction_impl::block_commit_or_abort():", g_block_commit_or_abort);
    dump( "Transaction_impl::unblock_commit_or_abort():", g_unblock_commit_or_abort);
    dump( "Transaction_impl::get_journal():", g_transaction_get_journal);
    s << std::endl;
    dump( "Scope_impl::get_journal():", g_scope_get_journal);
    dump( "Scope_impl destructor:", g_scope_destructor);
    s << std::endl;
    dump( "Info_manager::lookup_info_by_name():", g_lookup_info_by_name);
    dump( "Info_manager::lookup_info_by_tag():", g_lookup_info_by_tag);
    dump( "Info_manager::garbage_collection():", g_garbage_collection);
    s << std::endl;
    dump( "Element and job destructors:", g_element_job_destructors);
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
