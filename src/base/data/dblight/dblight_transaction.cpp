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
 ** \brief This file implements a very simple transaction
 **/

#include "pch.h"

#include "dblight_transaction.h"

#include "dblight_database.h"
#include "dblight_scope.h"

#include <base/system/main/i_assert.h>
#include <base/data/db/i_db_info.h>
#include <base/data/db/i_db_element.h>

namespace MI {

namespace DBLIGHT {

Transaction_impl::Transaction_impl(
    Database_impl* database, Scope_impl* scope, DB::Transaction_id id)
  : m_database(database)
  , m_scope(scope)
  , m_id(id)
  , m_refcount(1)
  , m_next_sequence_number(0)
  , m_is_open(true)
{
}

Transaction_impl::~Transaction_impl() { }

void Transaction_impl::pin()
{
    ++m_refcount;
}

void Transaction_impl::unpin()
{
    if (--m_refcount == 0)
        delete this;
}

bool Transaction_impl::block_commit()
{
    MI_ASSERT(false);
    return false;
}

bool Transaction_impl::unblock_commit()
{
    MI_ASSERT(false);
    return false;
}

bool Transaction_impl::commit()
{
    if (!m_is_open)
        return false;

    m_database->garbage_collection_internal();
    m_scope->decrement_transaction_count();
    m_is_open = false;
    return true;
}

void Transaction_impl::abort() { MI_ASSERT(false); }

bool Transaction_impl::is_open() { return m_is_open; }

DB::Tag Transaction_impl::reserve_tag() { return m_database->allocate_tag(); }

DB::Tag Transaction_impl::store(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    if (!m_is_open)
        return DB::Tag();

    DB::Tag tag = m_database->allocate_tag();
    element->prepare_store(this, tag);

    Uint32 version = m_next_sequence_number++;
    DB::Info* info = new DB::Info(m_database, tag, this, DB::Scope_id(0), version, element);

    mi::base::Lock::Block block(&m_database->m_lock);

    info->store_references();
    m_database->get_tag_map()[tag] = info;
    m_database->increment_reference_count(tag);

    if (name) {
        m_database->get_named_tag_map()[name] = tag;
        m_database->get_reverse_named_tag_map()[tag] = name;
    }

    return tag;
}

void Transaction_impl::store(
    DB::Tag tag,
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    if (!m_is_open)
        return;

    element->prepare_store(this, tag);

    Uint32 version = m_next_sequence_number++;
    DB::Info* info = new DB::Info(m_database, tag, this, DB::Scope_id(0), version, element);

    mi::base::Lock::Block block(&m_database->m_lock);

    info->store_references();

    Tag_map::iterator it = m_database->get_tag_map().find(tag);
    if (it != m_database->get_tag_map().end()) {
         it->second->unpin();
         it->second = info;
         // leave self-reference as is
    } else {
        m_database->get_tag_map()[tag] = info;
        m_database->increment_reference_count(tag);
    }

    if (name) {
         m_database->get_named_tag_map()[name] = tag;
         m_database->get_reverse_named_tag_map()[tag] = name;
    }
}

DB::Tag Transaction_impl::store(
    SCHED::Job* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    MI_ASSERT(false);
    return DB::Tag();
}

void Transaction_impl::store(
    DB::Tag tag,
    SCHED::Job* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    MI_ASSERT(false);
}

DB::Tag Transaction_impl::store_for_reference_counting(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    DB::Tag tag = store(element, name, privacy_level, store_level);
    if (tag)
        remove(tag,false);
    return tag;
}

void Transaction_impl::store_for_reference_counting(
    DB::Tag tag,
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    store(tag, element, name, privacy_level, journal_type, store_level);
    remove(tag,false);
}

DB::Tag Transaction_impl::store_for_reference_counting(
    SCHED::Job* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    MI_ASSERT(false);
    return DB::Tag();
}

void Transaction_impl::store_for_reference_counting(
    DB::Tag tag, SCHED::Job* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    MI_ASSERT(false);
}

void Transaction_impl::invalidate_job_results(DB::Tag tag)
{
    MI_ASSERT(false);
}

bool Transaction_impl::remove(DB::Tag tag, bool remove_local_copy)
{
    if (!m_is_open)
        return false;

    mi::base::Lock::Block block(&m_database->m_lock);
    std::pair<Flagged_for_removal_set::iterator,bool> result
        = m_database->get_flagged_for_removal_set().insert(tag);
    if (result.second)
        m_database->decrement_reference_count(tag);

    return true;
}

void Transaction_impl::advise(DB::Tag tag)
{
    MI_ASSERT(false);
}

void Transaction_impl::localize(
    DB::Tag tag, DB::Privacy_level privacy_level, DB::Journal_type journal_type)
{
    MI_ASSERT(false);
}

const char* Transaction_impl::tag_to_name(DB::Tag tag)
{
    if (!m_is_open)
        return 0;

    mi::base::Lock::Block block(&m_database->m_lock);
    Reverse_named_tag_map::const_iterator it = m_database->get_reverse_named_tag_map().find(tag);
    if (it == m_database->get_reverse_named_tag_map().end())
        return 0;
    return it->second.c_str(); // TODO unsafe
}

DB::Tag Transaction_impl::name_to_tag(const char* name)
{
    if (!m_is_open || !name)
        return DB::Tag();

    mi::base::Lock::Block block(&m_database->m_lock);
    Named_tag_map::const_iterator it = m_database->get_named_tag_map().find(name);
    if (it == m_database->get_named_tag_map().end())
         return DB::Tag();
    return it->second;
}

SERIAL::Class_id Transaction_impl::get_class_id(DB::Tag tag)
{
    if (!m_is_open)
        return SERIAL::Class_id();

    DB::Info* info = Transaction_impl::get_element(tag, true);
    SERIAL::Class_id class_id = info->get_element()->get_class_id();
    info->unpin();
    return class_id;
}

DB::Tag_version Transaction_impl::get_tag_version(DB::Tag tag)
{
    DB::Info* info = Transaction_impl::get_element(tag, true);
    if (!info)
        return DB::Tag_version();

    DB::Tag_version result;
    result.m_tag = tag;
    result.m_transaction_id = info->get_transaction_id();
    result.m_version = info->get_version();
    info->unpin();
    return result;
}

Uint32 Transaction_impl::get_update_sequence_number()
{
    return m_next_sequence_number;
}

Uint32 Transaction_impl::get_tag_reference_count(DB::Tag tag)
{
    return m_database->get_tag_reference_count(tag);
}

bool Transaction_impl::can_reference_tag(
    DB::Privacy_level referencing_level, DB::Tag referenced_tag)
{
    return true;
}

bool Transaction_impl::can_reference_tag(
    DB::Tag referencing_tag, DB::Tag referenced_tag)
{
    return true;
}

bool Transaction_impl::get_tag_is_removed(DB::Tag tag)
{
    if (!m_is_open)
        return false;

    mi::base::Lock::Block block(&m_database->m_lock);
    const Flagged_for_removal_set& set = m_database->get_flagged_for_removal_set();
    return set.find(tag) != set.end();
}

bool Transaction_impl::get_tag_is_job(DB::Tag tag) { return false; }

DB::Privacy_level Transaction_impl::get_tag_privacy_level(DB::Tag tag) { return 0; }

DB::Privacy_level Transaction_impl::get_tag_storage_level(DB::Tag tag) { return 0; }

DB::Transaction_id Transaction_impl::get_id() const { return m_id; }

std::vector<std::pair<DB::Tag, DB::Journal_type> >* Transaction_impl::get_journal(
    DB::Transaction_id last_transaction_id,
    Uint32 last_transaction_change_version,
    DB::Journal_type journal_type,
    bool lookup_parent_scopes)
{
    MI_ASSERT(false);
    return 0;
}

Sint32 Transaction_impl::execute_fragmented(DB::Fragmented_job* job, size_t count)
{
    MI_ASSERT(false);
    return -1;
}

Sint32 Transaction_impl::execute_fragmented_async(
    DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener)
{
    MI_ASSERT(false);
    return -1;
}

void Transaction_impl::cancel_fragmented_jobs() { MI_ASSERT(false); }

bool Transaction_impl::get_fragmented_jobs_cancelled() { return false; }

DB::Scope* Transaction_impl::get_scope() { return m_scope; }

DB::Info* Transaction_impl::get_job(DB::Tag tag) { MI_ASSERT(false); return 0; }

void Transaction_impl::store_job_result(DB::Tag tag, DB::Element_base* element) { MI_ASSERT(false);}

void Transaction_impl::send_element_to_host(DB::Tag tag, NET::Host_id host_id) { MI_ASSERT(false); }

DB::Update_list* Transaction_impl::get_received_updates() { MI_ASSERT(false); return 0; }

void Transaction_impl::wait(DB::Update_list* needed_updates) { MI_ASSERT(false); }

DB::Info* Transaction_impl::edit_element(DB::Tag tag)
{
    if (!m_is_open)
        return 0;

    mi::base::Lock::Block block(&m_database->m_lock);

    Tag_map::const_iterator it = m_database->get_tag_map().find(tag);
    if (it == m_database->get_tag_map().end())
         return 0;

    DB::Info* old_info = it->second;
    DB::Element_base* new_element = old_info->get_element()->copy();
    Uint32 version = m_next_sequence_number++;
    DB::Info* new_info = new DB::Info(m_database, tag, this, DB::Scope_id(0), version, new_element);
    new_info->store_references();

    old_info->unpin();
    m_database->get_tag_map()[tag] = new_info;

    new_info->pin();
    return new_info;
}

void Transaction_impl::finish_edit(DB::Info* info, DB::Journal_type journal_type)
{
    info->get_element()->prepare_store(this, info->get_tag());

    mi::base::Lock::Block block(&m_database->m_lock);
    info->store_references();
    info->unpin();
}

DB::Info* Transaction_impl::get_element(DB::Tag tag, bool do_wait)
{
    if (!m_is_open)
        return 0;

    mi::base::Lock::Block block(&m_database->m_lock);

    Tag_map::const_iterator it = m_database->get_tag_map().find(tag);
    if (it == m_database->get_tag_map().end())
        return 0;

    DB::Info* info = it->second;
    info->pin();
    return info;
}

DB::Element_base* Transaction_impl::construct_empty_element(SERIAL::Class_id class_id)
{
    MI_ASSERT(false);
    return 0;
}

DB::Transaction* Transaction_impl::get_real_transaction() { return this; }

} // namespace DBLIGHT

} // namespace MI

