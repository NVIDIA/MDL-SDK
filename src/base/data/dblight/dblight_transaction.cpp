/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "dblight_transaction.h"

#include <sstream>
#include <vector>

#include "dblight_database.h"
#include "dblight_fragmented_job.h"
#include "dblight_info.h"
#include "dblight_scope.h"

#include <base/data/db/i_db_element.h>
#include <base/data/serial/i_serial_buffer_serializer.h>
#include <base/data/serial/serial.h>
#include <base/data/thread_pool/i_thread_pool_thread_pool.h>
#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_assert.h>

namespace MI {

namespace DBLIGHT {

Transaction_impl::Transaction_impl(
    Database_impl* database,
    Transaction_manager* transaction_manager,
    Scope_impl* scope,
    DB::Transaction_id id)
  : m_database( database),
    m_transaction_manager( transaction_manager),
    m_scope( scope),
    m_id( id),
    m_visibility_id( ~0U)
{
    m_scope->pin();
}

Transaction_impl::~Transaction_impl()
{
    MI_ASSERT( m_state == COMMITTED || m_state == ABORTED);
    m_transaction_manager->remove_from_all_transactions( this);

    m_scope->unpin();
}

DB::Scope* Transaction_impl::get_scope()
{
    return m_scope;
}

bool Transaction_impl::commit()
{
    Statistics_helper helper( g_commit);

    return m_transaction_manager->end_transaction( this, /*commit*/ true);
}

void Transaction_impl::abort()
{
    Statistics_helper helper( g_abort);

    m_transaction_manager->end_transaction( this, /*commit*/ false);
}

bool Transaction_impl::is_open( bool closing_is_open) const
{
    if( m_state == OPEN)
        return true;
    if( (m_state == CLOSING) && closing_is_open)
        return true;
    return false;
}

bool Transaction_impl::block_commit_or_abort()
{
    Statistics_helper helper( g_block_commit_or_abort);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN)
        return false;

    ++m_block_counter;
    return true;
}

bool Transaction_impl::unblock_commit_or_abort()
{
    Statistics_helper helper( g_unblock_commit_or_abort);

    THREAD::Block_shared block( &m_database->get_lock());

    if( (m_state != OPEN) && (m_state != CLOSING))
        return false;

    if( m_block_counter == 0) {
        MI_ASSERT( !"Unbalanced commit/abort blocking");
        return false;
    }

    mi::Uint32 result = --m_block_counter;
    if( (result == 0) && (m_state == CLOSING))
        m_block_condition.signal();
    return true;
}

DB::Info* Transaction_impl::access_element( DB::Tag tag)
{
    Statistics_helper helper( g_access);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return nullptr;
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info) {
        LOG::mod_log->fatal(
            M_DB, LOG::Mod_log::C_DATABASE, "Access of invalid tag " FMT_TAG, tag.get_uint());
        return nullptr;
    }

    return info;
}

DB::Info* Transaction_impl::edit_element( DB::Tag tag)
{
    Statistics_helper helper( g_edit);

    THREAD::Block block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return nullptr;
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info) {
        LOG::mod_log->fatal(
            M_DB, LOG::Mod_log::C_DATABASE, "Edit of invalid tag " FMT_TAG, tag.get_uint());
        return nullptr;
    }

    DB::Element_base* element = info->get_element()->copy();
    MI_ASSERT( element);

    mi::Uint32 version = allocate_sequence_number();
    DB::Privacy_level privacy_level = info->get_privacy_level();
    const DB::Tag_set& references = info->get_references();

    // Find scope for store level.
    DB::Scope* scope = m_scope;
    while( scope->get_level() > privacy_level)
        scope = scope->get_parent();
    ASSERT( M_DB, scope->get_level() <= privacy_level);

    auto* scope_impl = static_cast<Scope_impl*>( scope);
    Info_impl* new_info = m_database->get_info_manager()->start_edit(
        element, scope_impl, this, version, tag, privacy_level, info->get_name(), references);

    info->unpin();
    return new_info;
}

void Transaction_impl::finish_edit( DB::Info* info, DB::Journal_type journal_type)
{
    Statistics_helper helper( g_finish_edit);

    // Invoke callback to prepare store.
    info->get_element()->prepare_store( this, info->get_tag());

    THREAD::Block block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return;
    }

    // Check serialization.
    if( m_database->get_check_serialization_edit()) {
            SERIAL::Buffer_serializer serializer;
            serializer.serialize( info->get_element());
            SERIAL::Buffer_deserializer deserializer( m_database->get_deserialization_manager());
            static_cast<Info_impl*>( info)->set_element( static_cast<DB::Element_base*>(
                deserializer.deserialize( serializer.get_buffer(), serializer.get_buffer_size())));
    }

    m_database->get_info_manager()->finish_edit( static_cast<Info_impl*>( info), this);

    if( m_database->get_journal_enabled()) {
        DB::Journal_type journal_mask = info->get_element()->get_journal_flags();
        journal_type.restrict_journal( journal_mask);
        if( journal_type != DB::JOURNAL_NONE)
            m_journal.emplace_back(
                info->get_tag(), info->get_version(), info->get_scope_id(), journal_type);
    }
}

DB::Tag Transaction_impl::reserve_tag()
{
    return m_database->allocate_tag();
}

DB::Tag Transaction_impl::store(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    DB::Tag tag = m_database->allocate_tag();
    store_element_internal(
        tag, element, name, privacy_level, DB::JOURNAL_NONE, store_level, /*store_for_rc*/ false);
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
    store_element_internal(
        tag, element, name, privacy_level, journal_type, store_level, /*store_for_rc*/ false);
}

DB::Tag Transaction_impl::store_for_reference_counting(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    DB::Tag tag = m_database->allocate_tag();
    store_element_internal(
        tag, element, name, privacy_level, DB::JOURNAL_NONE, store_level, /*store_for_rc*/ true);
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
    store_element_internal(
        tag, element, name, privacy_level, journal_type, store_level, /*store_for_rc*/ true);
}

void Transaction_impl::store_element_internal(
    DB::Tag tag,
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level,
    bool store_for_rc)
{
    Statistics_helper helper( g_store);

    if( !tag) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Invalid tag used with "
            "Transaction::%s().", store_for_rc ? "store_for_reference_counting" : "store");
        delete element;
        return;
    }

    if( name && !name[0]) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Invalid empty name used with "
            "Transaction::%s().", store_for_rc ? "store_for_reference_counting" : "store");
        delete element;
        return;
    }

    if( !element) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Invalid element used with "
            "Transaction::%s().", store_for_rc ? "store_for_reference_counting" : "store");
        return;
    }

    // Invoke callback to prepare store.
    element->prepare_store( this, tag);

    THREAD::Block block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        delete element;
        return;
    }

    // Check serialization.
    std::string name_str;
    if( m_database->get_check_serialization_store()) {
        // Copy name to avoid that the pointer becomes invalid in case it points into the element
        // itself.
        if( name)
            name_str = name;
        SERIAL::Buffer_serializer serializer;
        serializer.serialize( element);
        SERIAL::Buffer_deserializer deserializer( m_database->get_deserialization_manager());
        delete element;
        element = static_cast<DB::Element_base*>(
            deserializer.deserialize( serializer.get_buffer(), serializer.get_buffer_size()));
        // Ensure that the name is valid again in case it pointed into the element itself.
        if( name)
            name = name_str.c_str();
    }

    // Clamp store level.
    if( store_level > privacy_level)
        store_level = privacy_level;
    MI_ASSERT( store_level <= privacy_level);

    // Find scope for store level.
    DB::Scope* scope = m_scope;
    while( scope->get_level() > store_level)
        scope = scope->get_parent();
    ASSERT( M_DB, scope->get_level() <= store_level);

    DB::Tag_set references;
    element->get_references( &references);

    // Check privacy levels.
    if( m_database->get_check_privacy_levels()) {
        DB::Privacy_level referencing_level = scope->get_level();
        check_privacy_levels( referencing_level, references, tag, name, /*store*/ true);
    }

    // Check reference cycles.
    if( m_database->get_check_reference_cycles_store()) {
        check_reference_cycles( references, tag, name, /*store*/ true);
    }

    auto* scope_impl = static_cast<Scope_impl*>( scope);
    mi::Uint32 version1 = allocate_sequence_number();
    // Modifies references.
    m_database->get_info_manager()->store(
        element, scope_impl, this, version1, tag, privacy_level, name, references);

    if( m_database->get_journal_enabled()) {
        DB::Journal_type journal_mask = element->get_journal_flags();
        journal_type.restrict_journal( journal_mask);
        if( journal_type != DB::JOURNAL_NONE)
            m_journal.emplace_back( tag, version1, scope_impl->get_id(), journal_type);
    }

    if( store_for_rc) {
        mi::Uint32 version2 = allocate_sequence_number();
        m_database->get_info_manager()->remove(
            m_scope, this, version2, tag, /*remove_local_copy*/ false);
    }
}

DB::Tag Transaction_impl::store(
    SCHED::Job_base* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    MI_ASSERT( !"Not implemented");
    return {};
}

void Transaction_impl::store(
    DB::Tag tag,
    SCHED::Job_base* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    MI_ASSERT( !"Not implemented");
}

DB::Tag Transaction_impl::store_for_reference_counting(
    SCHED::Job_base* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    MI_ASSERT( !"Not implemented");
    return {};
}

void Transaction_impl::store_for_reference_counting(
    DB::Tag tag, SCHED::Job_base* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    MI_ASSERT( !"Not implemented");
}

void Transaction_impl::localize(
    DB::Tag tag, DB::Privacy_level privacy_level, DB::Journal_type journal_type)
{
    Statistics_helper helper( g_localize);

    DB::Info* info = access_element( tag);
    DB::Element_base* element = info->get_element();
    DB::Element_base* copy = element->copy();
    store( tag, copy, info->get_name(), privacy_level, journal_type, privacy_level);
    info->unpin();
}

bool Transaction_impl::remove( DB::Tag tag, bool remove_local_copy)
{
    Statistics_helper helper( g_remove);

    THREAD::Block block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return false;
    }

    mi::Uint32 version = allocate_sequence_number();
    return m_database->get_info_manager()->remove( m_scope, this, version, tag, remove_local_copy);
}

const char* Transaction_impl::tag_to_name( DB::Tag tag)
{
    Statistics_helper helper( g_tag_to_name);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return nullptr;
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info)
        return nullptr;

    const char* result = info->get_name();
    info->unpin();
    return result;
}

DB::Tag Transaction_impl::name_to_tag( const char* name)
{
    Statistics_helper helper( g_name_to_tag);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return {};
    }

    if( !name)
        return {};

    Info_impl* info = m_database->get_info_manager()->lookup_info( name, m_scope, m_id);
    if( !info)
        return {};

    DB::Tag result = info->get_tag();
    info->unpin();
    return result;
}

SERIAL::Class_id Transaction_impl::get_class_id( DB::Tag tag)
{
    Statistics_helper helper( g_get_class_id);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return 0;
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info)
        return SERIAL::class_id_unknown;

    SERIAL::Class_id class_id = info->get_element()->get_class_id();
    info->unpin();
    return class_id;
}

DB::Privacy_level Transaction_impl::get_tag_privacy_level( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_privacy_level);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return 0;
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info)
        return 0;

    DB::Privacy_level privacy_level = info->get_privacy_level();
    info->unpin();
    return privacy_level;
}

DB::Privacy_level Transaction_impl::get_tag_store_level( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_store_level);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return 0;
    }

    DB::Privacy_level store_level;
    Info_impl* info = m_database->get_info_manager()->lookup_info(
        tag, m_scope, m_id, &store_level);
    if( !info)
        return 0;

    info->unpin();
    return store_level;
}

mi::Uint32 Transaction_impl::get_tag_reference_count( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_reference_count);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return 0;
    }

    return m_database->get_info_manager()->get_tag_reference_count( tag);
}

DB::Tag_version Transaction_impl::get_tag_version( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_version);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return {};
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info)
        return {};

    DB::Tag_version result( tag, info->get_transaction_id(), info->get_version());
    info->unpin();
    return result;
}

bool Transaction_impl::can_reference_tag(
    DB::Privacy_level referencing_level, DB::Tag referenced_tag)
{
    Statistics_helper helper( g_can_reference_tag);

    THREAD::Block_shared block( &m_database->get_lock());

    return can_reference_tag_locked( referencing_level, referenced_tag);
}

bool Transaction_impl::can_reference_tag_locked(
    DB::Privacy_level referencing_level, DB::Tag referenced_tag)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    DB::Scope* scope = m_scope;
    while( scope->get_level() > referencing_level)
        scope = scope->get_parent();

    Info_impl* referenced_info = m_database->get_info_manager()->lookup_info(
        referenced_tag, scope, m_id);
    if( !referenced_info)
        return false;

    referenced_info->unpin();
    return true;
}

bool Transaction_impl::can_reference_tag(
    DB::Tag referencing_tag, DB::Tag referenced_tag)
{
    Statistics_helper helper( g_can_reference_tag);

    THREAD::Block_shared block( &m_database->get_lock());

    DB::Privacy_level referencing_level;
    Info_impl* referencing_info = m_database->get_info_manager()->lookup_info(
        referencing_tag, m_scope, m_id, &referencing_level);
    if( !referencing_info)
        return false;

    referencing_info->unpin();
    return can_reference_tag_locked( referencing_level, referenced_tag);
}

bool Transaction_impl::get_tag_is_removed( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_is_removed);

    THREAD::Block_shared block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return false;
    }

    return m_database->get_info_manager()->get_tag_is_removed( tag);
}

std::unique_ptr<DB::Journal_query_result> Transaction_impl::get_journal(
    DB::Transaction_id last_transaction_id,
    mi::Uint32 last_transaction_change_version,
    DB::Journal_type journal_type,
    bool lookup_parents)
{
    Statistics_helper helper( g_transaction_get_journal);

    if( !m_database->get_journal_enabled()) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Journal query with disabled journal.");
        return {};
    }

    THREAD::Block_shared block( &m_database->get_lock());
    auto result = std::make_unique<DB::Journal_query_result>();

    // Consider current transaction.
    DB::Scope_id scope_id = m_scope->get_id();
    for( size_t i = 0, n = m_journal.size(); i < n; ++i) {
        const Transaction_journal_entry& entry = m_journal[i];
        // If the query starts with the current transaction, skip entries which happened before
        // \p last_transaction_change_version.
        if(    m_id == last_transaction_id
            && entry.m_version < last_transaction_change_version)
            continue;
        // Skip entries from other scopes, i.e., parent scopes if \p lookup_parents is \c false.
        if( !lookup_parents && (entry.m_scope_id != scope_id))
            continue;
        // Skip entries that do not match the journal type filter.
        if( (entry.m_journal_type.get_type() & journal_type.get_type()) == 0)
            continue;
        result->emplace_back( entry.m_tag, entry.m_journal_type);
    }

    // Consider changes from other transactions.
    bool success = m_scope->get_journal(
        last_transaction_id,
        last_transaction_change_version,
        m_id,
        journal_type,
        lookup_parents,
        *result.get());
    if( !success)
        return nullptr;

    return result;
}

mi::Sint32 Transaction_impl::execute_fragmented( DB::Fragmented_job* job, size_t count)
{
    if( !job || count == 0)
        return -1;
    if( job->get_priority() < 0)
        return -3;

    DB::Fragmented_job::Scheduling_mode mode = job->get_scheduling_mode();
    if( mode == DB::Fragmented_job::ONCE_PER_HOST) {
        count = 1;
    } else if( mode == DB::Fragmented_job::USER_DEFINED) {
        // The information is ignored, but the job might expect and depend on the callback.
        std::vector<mi::Uint32> slots( count);
        job->assign_fragments_to_hosts( slots.data(), count);
    }

    THREAD::Block block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return -4;
    }

    mi::base::Handle<DBLIGHT::Fragmented_job> wrapped_job(
        new DBLIGHT::Fragmented_job( this, count, job, /*listener*/ nullptr));
    m_fragmented_jobs.push_back( *wrapped_job.get());
    ++m_fragmented_jobs_counter;

    block.release();

    m_database->get_thread_pool()->submit_job_and_wait( wrapped_job.get());
    return 0;
}

mi::Sint32 Transaction_impl::execute_fragmented_async(
    DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener)
{
    if( !job || count == 0)
        return -1;
    if( job->get_scheduling_mode() != DB::Fragmented_job::LOCAL)
        return -2;
    if( job->get_priority() < 0)
        return -3;

    THREAD::Block block( &m_database->get_lock());

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return -4;
    }

    mi::base::Handle<DBLIGHT::Fragmented_job> wrapped_job(
        new DBLIGHT::Fragmented_job( this, count, job, listener));
    m_fragmented_jobs.push_back( *wrapped_job.get());
    ++m_fragmented_jobs_counter;

    block.release();

    m_database->get_thread_pool()->submit_job( wrapped_job.get());
    return 0;
}

void Transaction_impl::cancel_fragmented_jobs()
{
    THREAD::Block block( &m_database->get_lock());

    cancel_fragmented_jobs_locked();
}

void Transaction_impl::invalidate_job_results( DB::Tag tag)
{
    MI_ASSERT( !"Not implemented");
}

void Transaction_impl::advise( DB::Tag tag)
{
    MI_ASSERT( !"Not implemented");
}

DB::Element_base* Transaction_impl::construct_empty_element( SERIAL::Class_id class_id)
{
    MI_ASSERT( !"Not implemented");
    return nullptr;
}

bool Transaction_impl::is_visible_for( DB::Transaction_id id) const
{
    if( id == m_id)
        return true;

    return (m_state == COMMITTED) && (m_visibility_id <= id);
}

namespace {

/// Quotes name for logging purposes (or returns "without name" for \p nullptr).
std::string get_log_name( const char* name)
{
    if( !name)
        return "without name";

    std::string result;
    result += '\"';
    result += name;
    result += '\"';
    return result;
}

/// Enumerates all tags in the tag set as a human-readable string.
std::string get_log_cycle( const DB::Tag_set& tag_set)
{
    MI_ASSERT( !tag_set.empty());

    std::ostringstream result;
    size_t n = tag_set.size();
    if( n == 1) {
        result << "tag " << (*tag_set.begin())();
    } else if( n == 2) {
        result << "tags " << (*tag_set.begin())() << " and " << (*++tag_set.begin())();
    } else {
        result << "tags ";
        size_t i = 0;
        for( const auto& tag: tag_set) {
            result << tag();
            if( i+2 < n)
                result << ", ";
            else if( i+1 < n)
                result << ", and ";
            ++i;
        }
        MI_ASSERT( i == n);
    }
    return result.str();
}

} // namespace

void Transaction_impl::check_privacy_levels(
    DB::Privacy_level referencing_level,
    const DB::Tag_set& references,
    DB::Tag tag,
    const char* name,
    bool store)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    for( DB::Tag referenced_tag: references) {
        if( !can_reference_tag_locked( referencing_level, referenced_tag)) {
            std::string log_name = get_log_name( name);
            LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
                "%s database element %s (tag %u) contains an invalid reference to a database "
                "element with tag %u. The referenced database element exists only in a more "
                "private scope (or not at all).",
                (store ? "Stored" : "Edited"), log_name.c_str(), tag(), referenced_tag());
        }
    }
}

void Transaction_impl::check_reference_cycles(
    const DB::Tag_set& references,
    DB::Tag root,
    const char* name,
    bool store)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    DB::Tag_set processing;
    DB::Tag_set done;

    processing.insert( root);

    for( const auto& tag: references) {
        if( done.find( tag) != done.end())
            continue;
        check_reference_cycles_internal( tag, root, name, store, processing, done);
    }

    // For completeness of the algorithm, but not relevant anymore.
    // processing.erase( root);
    // done.insert( root);
}

void Transaction_impl::check_reference_cycles_internal(
    DB::Tag tag,
    DB::Tag root,
    const char* name,
    bool store,
    DB::Tag_set& processing,
    DB::Tag_set& done)
{
    if( processing.find( tag) != processing.end()) {
        std::string log_name = get_log_name( name);
        std::string log_cycle = get_log_cycle( processing);
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "%s database element %s (tag %u) creates (or references) a reference cycle in the "
            "database involving %s.",
            (store ? "Stored" : "Edited"), log_name.c_str(), root(), log_cycle.c_str());
        return;
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info) {
        std::string log_name = get_log_name( name);
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "%s database element %s (tag %u) creates a (possibly indirect) invalid reference "
            "to a database element with tag %u. The referenced database element exists only "
            "in a more private scope (or not at all).",
            (store ? "Stored" : "Edited"), log_name.c_str(), root(), tag());
        return;
    }

    const DB::Tag_set& references = info->get_references();

    processing.insert( tag);

    for( const auto& tag: references) {
        if( done.find( tag) != done.end())
            continue;
        check_reference_cycles_internal( tag, root, name, store, processing, done);
    }

    processing.erase( tag);
    done.insert( tag);

    info->unpin();
}

void Transaction_impl::wait_for_unblocked_locked( bool commit)
{
    THREAD::Shared_lock& lock = m_database->get_lock();
    lock.check_is_owned();

    MI_ASSERT( m_state == CLOSING);

    if( m_block_counter == 0)
        return;

    LOG::mod_log->debug(
        M_DB, LOG::Mod_log::C_DATABASE, "%s transaction " FMT_TAG " is blocked.",
        commit ? "Committing" : "Aborting", m_id());

    lock.unlock();
    m_block_condition.wait();
    lock.lock();

    LOG::mod_log->debug(
        M_DB, LOG::Mod_log::C_DATABASE, "%s transaction " FMT_TAG " continues.",
        commit ? "Committing" : "Aborting", m_id());
}

void Transaction_impl::wait_for_fragmented_jobs_locked( bool commit)
{
    THREAD::Shared_lock& lock = m_database->get_lock();
    lock.check_is_owned();

    MI_ASSERT( m_state == CLOSING);

    if( m_fragmented_jobs_counter == 0)
        return;

    LOG::mod_log->debug(
        M_DB, LOG::Mod_log::C_DATABASE, "%s transaction " FMT_TAG " is waiting for fragmented "
        "jobs still being executed.", commit ? "Committing" : "Aborting", m_id());

    lock.unlock();
    m_fragmented_jobs_condition.wait();
    lock.lock();

    LOG::mod_log->debug(
        M_DB, LOG::Mod_log::C_DATABASE, "%s transaction " FMT_TAG " continues.",
        commit ? "Committing" : "Aborting", m_id());
}

void Transaction_impl::cancel_fragmented_jobs_locked()
{
    THREAD::Shared_lock& lock = m_database->get_lock();
    lock.check_is_owned();

    MI_ASSERT( m_state == OPEN || m_state == CLOSING);

    for( auto& job: m_fragmented_jobs)
        job.cancel();

    m_fragmented_jobs_cancelled = true;
}

void Transaction_impl::fragmented_job_finished( DBLIGHT::Fragmented_job* job)
{
    THREAD::Block block( &m_database->get_lock());

    auto it = Fragmented_jobs_list::s_iterator_to( *job);
    m_fragmented_jobs.erase( it);

    mi::Uint32 result = --m_fragmented_jobs_counter;
    if( (result == 0) && (m_state == CLOSING))
        m_fragmented_jobs_condition.signal();
}

std::ostream& operator<<( std::ostream& s, const Transaction_impl::State& state)
{
    switch( state) {
        case Transaction_impl::OPEN:      s << "OPEN";      break;
        case Transaction_impl::CLOSING:   s << "CLOSING";   break;
        case Transaction_impl::COMMITTED: s << "COMMITTED"; break;
        case Transaction_impl::ABORTED:   s << "ABORTED";   break;
    }
    return s;
}

Transaction_manager::~Transaction_manager()
{
    MI_ASSERT( m_open_transactions.empty());

    // Removal of all scopes (and their infos) should not leave any transactions behind.
    MI_ASSERT( m_all_transactions.empty());
}

Transaction_impl* Transaction_manager::start_transaction( Scope_impl* scope)
{
    THREAD::Block block( &m_database->get_lock());

    auto* transaction = new Transaction_impl(
        m_database, this, scope, m_next_transaction_id++);
    {
        THREAD::Block block( m_all_transactions_lock);
        m_all_transactions.insert( *transaction);
    }
    m_open_transactions.insert( *transaction);

    m_database->notify_transaction_listeners(
        &DB::ITransaction_listener::transaction_created, transaction);

    return transaction;
}

bool Transaction_manager::end_transaction( Transaction_impl* transaction, bool commit)
{
    THREAD::Block block( &m_database->get_lock());

    if( transaction->get_state() != Transaction_impl::OPEN)
        return false;

    transaction->set_state( Transaction_impl::CLOSING);

    m_database->notify_transaction_listeners(
        commit
            ? &DB::ITransaction_listener::transaction_pre_commit
            : &DB::ITransaction_listener::transaction_pre_abort,
        transaction);

    transaction->cancel_fragmented_jobs_locked();

    // This call might temporarily unlock the database lock.
    transaction->wait_for_unblocked_locked( commit);
    m_database->get_lock().check_is_owned();

    // This call might temporarily unlock the database lock.
    transaction->wait_for_fragmented_jobs_locked( commit);
    m_database->get_lock().check_is_owned();

    auto it = m_open_transactions.find( *transaction);
    MI_ASSERT( it != m_open_transactions.end());
    m_open_transactions.erase( it);

    transaction->set_visibility_id( m_next_transaction_id);
    transaction->set_state( commit ? Transaction_impl::COMMITTED : Transaction_impl::ABORTED);

    if( m_database->get_journal_enabled()) {

        Transaction_impl::Transaction_journal& journal = transaction->get_journal();

        if( commit) {
            // Sort journal by scope ID to reduce number of iterations (including scope lookups) in
            // the following loop.
            auto cmp = [](
                const Transaction_journal_entry& lhs, const Transaction_journal_entry& rhs)
                { return lhs.m_scope_id < rhs.m_scope_id; };
            sort( journal.begin(), journal.end(), cmp);

            // Pass each chunk with a common scope ID to the corresponding scope.
            Scope_manager* scope_manager = m_database->get_scope_manager();
            for( size_t i = 0, n = journal.size(); i < n; ) {
                DB::Scope* scope = scope_manager->lookup_scope( journal[i].m_scope_id);
                Scope_impl* scope_impl = static_cast<Scope_impl*>( scope);
                DB::Transaction_id id = transaction->get_id();
                DB::Transaction_id visibility_id = transaction->get_visibility_id();
                size_t l = scope_impl->update_journal( id, visibility_id, journal.data()+i, n-i);
                MI_ASSERT( (l > 0) && (l <= n-i));
                i += l;
            }
        }

        journal.clear();
    }

    m_database->notify_transaction_listeners(
        commit
            ? &DB::ITransaction_listener::transaction_committed
            : &DB::ITransaction_listener::transaction_aborted,
        transaction);

    transaction->unpin();

    Statistics_helper helper( g_garbage_collection);
    m_database->get_info_manager()->garbage_collection( get_lowest_open_transaction_id());

    return true;
}

void Transaction_manager::remove_from_all_transactions( Transaction_impl* transaction)
{
    THREAD::Block block( m_all_transactions_lock);

    auto it = m_all_transactions.find( *transaction);
    MI_ASSERT( it != m_all_transactions.end());
    m_all_transactions.erase( it);
}

DB::Transaction_id Transaction_manager::get_lowest_open_transaction_id() const
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    auto it = m_open_transactions.begin();
    return it == m_open_transactions.end() ? m_next_transaction_id : it->get_id();
}

void Transaction_manager::dump( std::ostream& s, bool mask_pointer_values)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();
    THREAD::Block block( m_all_transactions_lock);

    s << "Count of all transactions: " << m_all_transactions.size() << std::endl;
    s << "Count of open transactions: " << m_open_transactions.size() << std::endl;

    for( const auto& t: m_all_transactions) {

        s << "ID " << t.get_id()();
        if( !mask_pointer_values) s << " at " << &t;
        s << ": pin count = " << t.get_pin_count()
          << ", state = " << t.get_state()
          << ", next sequence number = " << t.get_next_sequence_number()
          << ", visibility ID = " << t.get_visibility_id()()
          << std::endl;

        if( m_database->get_journal_enabled()) {
            const Transaction_impl::Transaction_journal& journal = t.get_journal();
            size_t n = journal.size();
            s << "    Journal size: " << n << std::endl;
            for( size_t i = 0; i < n; ++i) {
                 const Transaction_journal_entry& entry = journal[i];
                 s << "    Item " << i
                   << ": tag = " << entry.m_tag()
                   << ", version = " << entry.m_version
                   << ", scope ID = " << entry.m_scope_id
                   << ", journal type = " << entry.m_journal_type.get_type()
                   << std::endl;
            }
         }
    }

    if( !m_all_transactions.empty())
        s << std::endl;
}

void Fragmented_job::job_finished()
{
    if( m_transaction) {
        auto* transaction_impl = static_cast<DBLIGHT::Transaction_impl*>( m_transaction);
        transaction_impl->fragmented_job_finished( this);
    }

    if( m_listener)
        m_listener->job_finished();
}

} // namespace DBLIGHT

} // namespace MI

