/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "dblight_database.h"
#include "dblight_info.h"
#include "dblight_scope.h"
#include "dblight_util.h"

#include <base/data/db/i_db_element.h>
#include <base/data/serial/i_serial_buffer_serializer.h>
#include <base/data/serial/serial.h>
#include <base/hal/thread/i_thread_block.h>
#include <base/hal/thread/i_thread_rw_lock.h>
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
}

void Transaction_impl::unpin()
{
    if( --m_pin_count > 0)
        return;

    MI_ASSERT( m_state == COMMITTED || m_state == ABORTED);
    m_transaction_manager->remove_from_all_transactions( this);
    delete this;
}

DB::Scope* Transaction_impl::get_scope()
{
    return m_scope;
}

bool Transaction_impl::commit()
{
    Statistics_helper helper( g_commit);

    if( m_state != OPEN)
        return false;

    m_transaction_manager->end_transaction( this, /*commit*/ true);
    return true;
}

void Transaction_impl::abort()
{
    Statistics_helper helper( g_abort);

    if( m_state != OPEN)
        return;

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
    MI_ASSERT( !"Not implemented");
    return false;
}

bool Transaction_impl::unblock_commit_or_abort()
{
    MI_ASSERT( !"Not implemented");
    return false;
}

DB::Info* Transaction_impl::access_element( DB::Tag tag)
{
    Statistics_helper helper( g_access);

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
    const DB::Tag_set& references = info->get_references();
    Info_impl* new_info = m_database->get_info_manager()->start_edit(
        element, DB::Scope_id( 0), this, version, tag, info->get_name(), references);

    info->unpin();
    return new_info;
}

void Transaction_impl::finish_edit( DB::Info* info, DB::Journal_type journal_type)
{
    Statistics_helper helper( g_finish_edit);

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return;
    }

    if( m_database->get_check_serialization_edit()) {
        SERIAL::Buffer_serializer serializer;
        serializer.serialize( info->get_element());
        SERIAL::Buffer_deserializer deserializer( m_database->get_deserialization_manager());
        static_cast<Info_impl*>( info)->set_element( static_cast<DB::Element_base*>(
            deserializer.deserialize( serializer.get_buffer(), serializer.get_buffer_size())));
    }

    info->get_element()->prepare_store( this, info->get_tag());
    m_database->get_info_manager()->finish_edit( static_cast<Info_impl*>( info));
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
    store( tag, element, name, privacy_level, DB::JOURNAL_NONE, store_level);
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
    Statistics_helper helper( g_store);

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        delete element;
        return;
    }

    if( !tag) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Invalid tag used with "
            " Transaction::store() or Transaction::store_for_reference_counting().");
        delete element;
        return;
    }

    if( name && !name[0]) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Invalid empty name used with "
            " Transaction::store() or Transaction::store_for_reference_counting().");
        delete element;
        return;
    }

    if( !element) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Invalid element used with "
            " Transaction::store() or Transaction::store_for_reference_counting().");
        return;
    }

    MI_ASSERT( privacy_level == 0 || privacy_level == 255);
    MI_ASSERT( journal_type == DB::JOURNAL_NONE || journal_type == DB::JOURNAL_ALL);
    MI_ASSERT( store_level == 0 || store_level == 255);

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

    mi::Uint32 version = allocate_sequence_number();
    element->prepare_store( this, tag);
    m_database->get_info_manager()->store( element, DB::Scope_id( 0), this, version, tag, name);
}

DB::Tag Transaction_impl::store_for_reference_counting(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    DB::Tag tag = store( element, name, privacy_level, store_level);
    if( tag)
        remove( tag, false);
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
    store( tag, element, name, privacy_level, journal_type, store_level);
    remove( tag, false);
}

DB::Tag Transaction_impl::store(
    SCHED::Job* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    MI_ASSERT( !"Not implemented");
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
    MI_ASSERT( !"Not implemented");
}

DB::Tag Transaction_impl::store_for_reference_counting(
    SCHED::Job* job,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    MI_ASSERT( !"Not implemented");
    return DB::Tag();
}

void Transaction_impl::store_for_reference_counting(
    DB::Tag tag, SCHED::Job* job,
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
    MI_ASSERT( !"Not implemented");
}

bool Transaction_impl::remove( DB::Tag tag, bool remove_local_copy)
{
    Statistics_helper helper( g_remove);

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return false;
    }

    mi::Uint32 version = allocate_sequence_number();
    return m_database->get_info_manager()->remove( DB::Scope_id( 0), this, version, tag);
}

const char* Transaction_impl::tag_to_name( DB::Tag tag)
{
    Statistics_helper helper( g_tag_to_name);

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

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return DB::Tag();
    }

    if( !name)
        return DB::Tag();

    Info_impl* info = m_database->get_info_manager()->lookup_info( name, m_scope, m_id);
    if( !info)
        return DB::Tag();

    DB::Tag result = info->get_tag();
    info->unpin();
    return result;
}

SERIAL::Class_id Transaction_impl::get_class_id( DB::Tag tag)
{
    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return SERIAL::Class_id();
    }

    DB::Info* info = access_element( tag);
    SERIAL::Class_id class_id = info->get_element()->get_class_id();
    info->unpin();
    return class_id;
}

mi::Uint32 Transaction_impl::get_tag_reference_count( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_reference_count);

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return 0;
    }

    return m_database->get_info_manager()->get_tag_reference_count( tag);
}

DB::Tag_version Transaction_impl::get_tag_version( DB::Tag tag)
{
    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return DB::Tag_version();
    }

    Info_impl* info = m_database->get_info_manager()->lookup_info( tag, m_scope, m_id);
    if( !info)
        return DB::Tag_version();

    DB::Tag_version result( tag, info->get_transaction_id(), info->get_version());
    info->unpin();
    return result;
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

bool Transaction_impl::get_tag_is_removed( DB::Tag tag)
{
    Statistics_helper helper( g_get_tag_is_removed);

    if( m_state != OPEN) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Use of non-open transaction.");
        return 0;
    }

    return m_database->get_info_manager()->get_tag_is_removed( tag);
}

std::unique_ptr<DB::Journal_query_result> Transaction_impl::get_journal(
    DB::Transaction_id last_transaction_id,
    mi::Uint32 last_transaction_change_version,
    DB::Journal_type journal_type,
    bool lookup_parents)
{
    MI_ASSERT( !"Not implemented");
    return {};
}

mi::Sint32 Transaction_impl::execute_fragmented( DB::Fragmented_job* job, size_t count)
{
    return m_database->execute_fragmented( this, job, count);
}

mi::Sint32 Transaction_impl::execute_fragmented_async(
    DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener)
{
    return m_database->execute_fragmented_async( this, job, count, listener);
}

void Transaction_impl::cancel_fragmented_jobs()
{
    MI_ASSERT( !"Not implemented");
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
    MI_ASSERT( m_all_transactions.empty());
}

Transaction_impl* Transaction_manager::start_transaction( Scope_impl* scope)
{
    THREAD::Block block( &m_database->get_lock());

    Transaction_impl* transaction = new Transaction_impl(
        m_database, this, scope, m_next_transaction_id++);
    m_all_transactions.insert( *transaction);
    m_open_transactions.insert( *transaction);

    return transaction;
}

void Transaction_manager::end_transaction( Transaction_impl* transaction, bool commit)
{
    THREAD::Block block( &m_database->get_lock());

    transaction->set_state( Transaction_impl::CLOSING);

    auto it = m_open_transactions.find( *transaction);
    MI_ASSERT( it != m_open_transactions.end());
    m_open_transactions.erase( it);

    transaction->set_visibility_id( m_next_transaction_id);
    transaction->set_state( commit ? Transaction_impl::COMMITTED : Transaction_impl::ABORTED);
    transaction->unpin();

    m_database->get_info_manager()->garbage_collection( get_lowest_open_transaction_id());
}

void Transaction_manager::remove_from_all_transactions( Transaction_impl* transaction)
{
    THREAD::Block block( m_all_transaction_lock);

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
    THREAD::Block block( m_all_transaction_lock);

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
    }
    if( !m_all_transactions.empty())
        s << std::endl;
}

} // namespace DBLIGHT

} // namespace MI
