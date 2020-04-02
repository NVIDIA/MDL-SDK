/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file i_db_transaction_wrapper.h
 ** \brief This transaction simply passes all function calls through to the wrapped transaction.
 **/

#ifndef BASE_DATA_DB_DB_TRANSACTION_WRAPPER_H
#define BASE_DATA_DB_DB_TRANSACTION_WRAPPER_H

#include "i_db_transaction.h"
#include "i_db_tag.h"
#include "i_db_element.h"

namespace MI
{

namespace DB
{

/// The transaction wrapper wraps another transaction and simply passes through all calls to the
/// wrapped transaction.
///
/// By itself this does not make much sense but it is meant to be used to derive other wrapping
/// transactions from. The derived transaction only has to overwrite those functions which actually
/// have differing implementations.
class Transaction_wrapper : public Transaction
{
  public:
    /// Constructor
    ///
    /// \param transaction      The wrapped transaction
    Transaction_wrapper(Transaction* transaction) { m_transaction = transaction; }

    bool is_open() { return m_transaction->is_open(); }

    Tag reserve_tag() { return m_transaction->reserve_tag(); }

    Tag store(Element_base* element, const char* name, Privacy_level privacy_level, 
        Privacy_level store_level = 255)
    {
        return m_transaction->store(element, name, privacy_level, store_level);
    }

    void store(Tag tag, Element_base* element, const char* name, Privacy_level privacy_level, 
        Journal_type journal_type, Privacy_level store_level = 255)
    {
        m_transaction->store(tag, element, name, privacy_level, journal_type, store_level);
    }

    Tag store(SCHED::Job* job, const char* name, Privacy_level privacy_level, 
        Privacy_level store_level = 255)
    {
        return m_transaction->store(job, name, privacy_level, store_level);
    }

    void store(Tag tag, SCHED::Job* job, const char* name, Privacy_level privacy_level, 
        Journal_type journal_type, Privacy_level store_level = 255)
    {
        m_transaction->store(tag, job, name, privacy_level, journal_type, store_level);
    }

    Tag store_for_reference_counting(Element_base* element, const char* name, 
        Privacy_level privacy_level, Privacy_level store_level = 255)
    {
        return m_transaction->store_for_reference_counting(element, name, privacy_level, 
            store_level);
    }

    void store_for_reference_counting(Tag tag, Element_base* element, const char* name,
        Privacy_level privacy_level, Journal_type journal_type, Privacy_level store_level = 255)
    {
        m_transaction->store_for_reference_counting(tag, element, name, privacy_level, 
            journal_type, store_level);
    }

    Tag store_for_reference_counting(SCHED::Job* job, const char* name, 
        Privacy_level privacy_level, Privacy_level store_level = 255)
    {
        return m_transaction->store_for_reference_counting(job, name, privacy_level, store_level);
    }

    void store_for_reference_counting(Tag tag, SCHED::Job* job, const char* name, 
        Privacy_level privacy_level, Journal_type journal_type, Privacy_level store_level = 255)
    {
        store_for_reference_counting(tag, job, name, privacy_level, journal_type, store_level);
    }

    Tag store_deferred(const char* name, Privacy_level privacy_level, Tag_set* references)
    {
        return m_transaction->store_deferred(name, privacy_level, references);
    }

    void advise(Tag tag) { m_transaction->advise(tag); }

    void localize(Tag tag, Privacy_level privacy_level, Journal_type journal_type)
    {
        m_transaction->localize(tag, privacy_level, journal_type);
    }

    const char* tag_to_name(Tag tag) { return m_transaction->tag_to_name(tag); }

    Tag name_to_tag(const char* name) { return m_transaction->name_to_tag(name); }

    SERIAL::Class_id get_class_id(Tag tag) { return m_transaction->get_class_id(tag); }

    Tag_version get_tag_version(Tag tag) { return m_transaction->get_tag_version(tag); }

    Uint32 get_update_sequence_number() { return m_transaction->get_update_sequence_number(); }

    Uint32 get_tag_reference_count(Tag tag) { return m_transaction->get_tag_reference_count(tag); }

    bool can_reference_tag(Privacy_level referencing_level, Tag referenced_tag)
    {
        return m_transaction->can_reference_tag(referencing_level, referenced_tag);
    }

    bool can_reference_tag(Tag referencing_tag, Tag referenced_tag)
    {
        return m_transaction->can_reference_tag(referencing_tag, referenced_tag);
    }

    bool get_tag_is_removed(Tag tag) { return m_transaction->get_tag_is_removed(tag); }

    bool get_tag_is_job(Tag tag) { return m_transaction->get_tag_is_job(tag); }

    Privacy_level get_tag_privacy_level(Tag tag)
    {
        return m_transaction->get_tag_privacy_level(tag);
    }

    Privacy_level get_tag_storage_level(Tag tag)
    {
        return m_transaction->get_tag_storage_level(tag);
    }

    Transaction_id get_id() const { return m_transaction->get_id(); }

    void invalidate_job_results(Tag tag) { m_transaction->invalidate_job_results(tag); }

    bool remove(Tag tag, bool remove_local_copy)
    {
        return m_transaction->remove(tag, remove_local_copy);
    }

    Info* get_job(Tag tag) { return m_transaction->get_job(tag); }

    void store_job_result(Tag tag, Element_base* element)
    {
        m_transaction->store_job_result(tag, element);
    }

    void send_element_to_host(Tag tag, NET::Host_id host_id)
    {
        m_transaction->send_element_to_host(tag, host_id);
    }

    Update_list* get_received_updates() { return m_transaction->get_received_updates(); }

    void wait(Update_list* needed_updates) { m_transaction->wait(needed_updates); }

    void pin() { m_transaction->pin(); }

    void unpin() { m_transaction->unpin(); }

    bool block_commit() { return m_transaction->block_commit(); }

    bool unblock_commit() { return m_transaction->unblock_commit(); }

    bool commit() { return m_transaction->commit(); }

    void abort() { m_transaction->abort(); }

    std::vector<std::pair<Tag, Journal_type> >* get_journal(
        Transaction_id last_transaction_id,
        Uint32 last_transaction_change_version,
        Journal_type journal_type,
        bool lookup_parents)
    {
        return m_transaction->get_journal(last_transaction_id, last_transaction_change_version,
            journal_type, lookup_parents);
    }

    Sint32 execute_fragmented(Fragmented_job* job, size_t count)
    {
        return m_transaction->execute_fragmented(job, count);
    }

    Sint32 execute_fragmented_async(
        Fragmented_job* job,
        size_t count,
        IExecution_listener* listener)
    {
        return m_transaction->execute_fragmented_async(job, count, listener);
    }

    void cancel_fragmented_jobs()
    {
        m_transaction->cancel_fragmented_jobs();
    }

    bool get_fragmented_jobs_cancelled()
    {
        return m_transaction->get_fragmented_jobs_cancelled();
    }

    Transaction* get_real_transaction() { return m_transaction->get_real_transaction(); }

    Info* edit_element(Tag tag) { return m_transaction->edit_element(tag); }

    void finish_edit(Info* info, Journal_type journal_type)
    {
        return m_transaction->finish_edit(info, journal_type);
    }

    Info* get_element(Tag tag, bool do_wait) { return m_transaction->get_element(tag, do_wait); }

    Element_base* construct_empty_element(SERIAL::Class_id class_id)
    {
        return m_transaction->construct_empty_element(class_id);
    }

    Scope* get_scope() { return m_transaction->get_scope(); }

  protected:
    /// The wrapped transaction
    Transaction* m_transaction;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_DB_CACHING_TRANSACTION_H
