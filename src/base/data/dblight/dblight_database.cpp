/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Implementation of the lightweight database.
 **
 ** This is an implementation of the database. It does only very little things.
 **/

#include "pch.h"

#include "dblight_database.h"
#include "dblight_scope.h"
#include "dblight_transaction.h"

#include <base/system/main/i_assert.h>
#include <base/data/db/i_db_element.h>
#include <base/data/db/i_db_info.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_database.h>

namespace MI {

namespace DBLIGHT {

Database_impl::Database_impl()
  : m_global_scope(new Scope_impl(this))
{
}

Database_impl::~Database_impl()
{
    for (Tag_map::iterator it = m_tags.begin(); it != m_tags.end(); ++it) {
        DB::Info* info = it->second;
        MI_ASSERT(info->get_pin_count() == 1);
        info->unpin();
    }

    m_global_scope->unpin();
}

void Database_impl::prepare_close() { }

void Database_impl::close()
{
    delete this;
}

void Database_impl::garbage_collection()
{
    Uint32 counter = m_global_scope->increment_transaction_count();
    if (counter > 1) {
        m_global_scope->decrement_transaction_count();
        return;
    }

    garbage_collection_internal();
    m_global_scope->decrement_transaction_count();
}

DB::Scope* Database_impl::get_global_scope() { return m_global_scope; }

DB::Scope* Database_impl::lookup_scope(DB::Scope_id id)
{
    return id == 0 ? m_global_scope : 0;
}

DB::Scope* Database_impl::lookup_scope(const std::string& name)
{
    return 0;
}

bool Database_impl::remove(DB::Scope_id id) { return false; }

Sint32 Database_impl::set_memory_limits(size_t low_water, size_t high_water)
{
    MI_ASSERT(false);
    return -1;
}

void Database_impl::get_memory_limits(size_t& low_water, size_t& high_water) const
{
    MI_ASSERT(false);
    low_water = 0; //-V779 PVS
    high_water = 0;
}

Sint32 Database_impl::set_disk_swapping(const char* path)
{
    MI_ASSERT(false);
    return -1;
}

const char* Database_impl::get_disk_swapping() const
{
    MI_ASSERT(false);
    return 0;
}

void Database_impl::lock(DB::Tag tag) { MI_ASSERT(false); }
bool Database_impl::unlock(DB::Tag tag) { MI_ASSERT(false); return false; }
void Database_impl::check_is_locked(DB::Tag tag) { MI_ASSERT(false); }

bool Database_impl::wait_for_notify(DB::Tag tag, TIME::Time timeout)
{
    MI_ASSERT(false);
    return false;
}

void Database_impl::notify(DB::Tag tag) { MI_ASSERT(false); }

std::string Database_impl::get_next_name(const std::string& pattern)
{
    MI_ASSERT(false);
    return "";
}

DB::Database_statistics Database_impl::get_statistics()
{
    MI_ASSERT(false);
    return DB::Database_statistics();
}

DB::Db_status Database_impl::get_database_status() { return DB::DB_OK; }

void Database_impl::register_status_listener(DB::IStatus_listener* listener) { MI_ASSERT(false); }
void Database_impl::unregister_status_listener(DB::IStatus_listener* listener) { MI_ASSERT(false); }
void Database_impl::register_transaction_listener(DB::ITransaction_listener* listener)
    { MI_ASSERT(false); }
void Database_impl::unregister_transaction_listener(DB::ITransaction_listener* listener)
    { MI_ASSERT(false); }
void Database_impl::register_scope_listener(DB::IScope_listener* listener) { MI_ASSERT(false); }
void Database_impl::unregister_scope_listener(DB::IScope_listener* listener) { MI_ASSERT(false); }
void Database_impl::set_ready_event(EVENT::Event0_base* event) { MI_ASSERT(false); }

DB::Transaction* Database_impl::get_transaction(DB::Transaction_id id)
{
    MI_ASSERT(false);
    return 0;
}

void Database_impl::cancel_all_fragmented_jobs() { MI_ASSERT(false); }

Sint32 Database_impl::execute_fragmented(DB::Fragmented_job* job, size_t count)
{
    MI_ASSERT(false);
    return -1;
}

Sint32 Database_impl::execute_fragmented_async(
    DB::Fragmented_job* job, size_t count,  DB::IExecution_listener* listener)
{
    MI_ASSERT(false);
    return -1;
}

void Database_impl::suspend_current_job() { MI_ASSERT(false); }
void Database_impl::resume_current_job() { MI_ASSERT(false); }
void Database_impl::yield() { MI_ASSERT(false); }

void Database_impl::increment_reference_count(DB::Tag tag)
{
    Uint32 value = ++m_reference_counts[tag];
    if (value == 1)
        m_reference_count_zero.erase(tag);
}

void Database_impl::decrement_reference_count(DB::Tag tag)
{
    Uint32 value = --m_reference_counts[tag];
    if (value == 0)
        m_reference_count_zero.insert(tag);
}

void Database_impl::increment_reference_counts(const DB::Tag_set& tag_set)
{
    DB::Tag_set::const_iterator it     = tag_set.begin();
    DB::Tag_set::const_iterator it_end = tag_set.end();

    for ( ; it != it_end; ++it)
        increment_reference_count(*it);
}

void Database_impl::decrement_reference_counts(const DB::Tag_set& tag_set)
{
    DB::Tag_set::const_iterator it     = tag_set.begin();
    DB::Tag_set::const_iterator it_end = tag_set.end();

    for ( ; it != it_end; ++it)
        decrement_reference_count(*it);
}

Uint32 Database_impl::get_tag_reference_count(DB::Tag tag)
{
    mi::base::Lock::Block block(&m_lock);
    return m_reference_counts[tag];
}

void Database_impl::garbage_collection_internal()
{
    mi::base::Lock::Block block(&m_lock);

    while (true) {

        DB::Tag_set candidates = m_reference_count_zero;
        if (candidates.empty())
            return;

        DB::Tag_set::const_iterator it     = candidates.begin();
        DB::Tag_set::const_iterator it_end = candidates.end();
        for ( ;  it != it_end; ++it) {

            DB::Tag tag = *it;

            Tag_map::iterator it_info = m_tags.find(tag);
            it_info->second->unpin(); //-V783 PVS
            m_tags.erase(it_info);

            Reverse_named_tag_map::iterator it_name = m_reverse_named_tags.find(tag);
            if( it_name != m_reverse_named_tags.end()) {
                m_named_tags.erase(it_name->second);
                m_reverse_named_tags.erase(it_name);
            }

            m_tags_flagged_for_removal.erase(tag);
            m_reference_counts.erase(tag);
            m_reference_count_zero.erase(tag);
        }
    }
}


DB::Database* factory()
{
    return new Database_impl;
}

} // namespace DBLIGHT

namespace DBNR { class Transaction_impl : public DB::Transaction { }; }

namespace DB { Database_statistics::Database_statistics() { MI_ASSERT(false); } } //-V730 PVS

} // namespace MI

