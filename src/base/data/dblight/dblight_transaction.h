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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_TRANSACTION_H
#define BASE_DATA_DBLIGHT_DBLIGHT_TRANSACTION_H

#include <base/data/db/i_db_transaction.h>

#include <mi/base/atom.h>
#include <base/data/db/i_db_tag.h>

namespace MI {

namespace DBLIGHT {

class Database_impl;
class Scope_impl;

class Transaction_impl : public DB::Transaction
{
public:
    Transaction_impl(Database_impl* database, Scope_impl* scope, DB::Transaction_id id);

    ~Transaction_impl();

    void pin();

    void unpin();

    bool block_commit();

    bool unblock_commit();

    bool commit();

    void abort();

    bool is_open();

    DB::Tag reserve_tag();

    DB::Tag store(
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level);

    void store(
        DB::Tag tag,
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level);

    DB::Tag store(
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level);

    void store(
        DB::Tag tag,
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level);

    DB::Tag store_for_reference_counting(
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level);

    void store_for_reference_counting(
        DB::Tag tag,
        DB::Element_base* element,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level);

    DB::Tag store_for_reference_counting(
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Privacy_level store_level);

    void store_for_reference_counting(
        DB::Tag tag,
        SCHED::Job* job,
        const char* name,
        DB::Privacy_level privacy_level,
        DB::Journal_type journal_type,
        DB::Privacy_level store_level);

    void invalidate_job_results(DB::Tag tag);

    bool remove(DB::Tag tag, bool remove_local_copy);

    void advise(DB::Tag tag);

    void localize(
        DB::Tag tag, DB::Privacy_level privacy_level, DB::Journal_type journal_type);

    const char* tag_to_name(DB::Tag tag);

    DB::Tag name_to_tag(const char* name);

    SERIAL::Class_id get_class_id(DB::Tag tag);

    DB::Tag_version get_tag_version(DB::Tag tag);

    Uint32 get_update_sequence_number();

    Uint32 get_tag_reference_count(DB::Tag tag);

    bool can_reference_tag(DB::Privacy_level referencing_level, DB::Tag referenced_tag);

    bool can_reference_tag(DB::Tag referencing_tag, DB::Tag referenced_tag);

    bool get_tag_is_removed(DB::Tag tag);

    bool get_tag_is_job(DB::Tag tag);

    DB::Privacy_level get_tag_privacy_level(DB::Tag tag);

    DB::Privacy_level get_tag_storage_level(DB::Tag tag);

    DB::Transaction_id get_id() const;

    std::vector<std::pair<DB::Tag, DB::Journal_type> >* get_journal(
         DB::Transaction_id last_transaction_id,
         Uint32 last_transaction_change_version,
         DB::Journal_type journal_type,
         bool lookup_parents);

    Sint32 execute_fragmented(DB::Fragmented_job* job, size_t count);

    Sint32 execute_fragmented_async(
        DB::Fragmented_job* job, size_t count, DB::IExecution_listener* listener);

    void cancel_fragmented_jobs();

    bool get_fragmented_jobs_cancelled();

    DB::Scope* get_scope();

    /// Pins the return value (but currently always NULL).
    DB::Info* get_job(DB::Tag tag);

    void store_job_result(DB::Tag tag, DB::Element_base* element);

    void send_element_to_host(DB::Tag tag, NET::Host_id host_id);

    DB::Update_list* get_received_updates();

    void wait(DB::Update_list* needed_updates);

    /// Pins the return value.
    DB::Info* edit_element(DB::Tag tag);

    void finish_edit(DB::Info* info, DB::Journal_type journal_type);

    /// Pins the return value.
    DB::Info* get_element(DB::Tag tag, bool do_wait);

    DB::Element_base* construct_empty_element(SERIAL::Class_id class_id);

    Transaction* get_real_transaction();

private:
    Database_impl* m_database;
    Scope_impl* m_scope;
    DB::Transaction_id m_id;
    mi::base::Atom32 m_refcount;
    mi::base::Atom32 m_next_sequence_number;
    bool m_is_open;
};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_TRANSACTION_H
