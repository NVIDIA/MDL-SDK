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
 ** \brief This file implements a very simple info
 ** Most of the functionality is actually not needed but only stubs are implemented.
 **/

#include "pch.h"

#include <base/data/db/i_db_element.h>
#include <base/data/db/i_db_info.h>
#include <base/data/db/i_db_transaction.h>

#include "dblight_database.h"

namespace MI {

namespace DBNR { class Transaction_impl : public DB::Transaction { }; }

namespace DB {

int Info_base::compare(Info_base* first, Info_base* second)
{
    if (first->m_scope_id > second->m_scope_id) return -1;
    if (first->m_scope_id < second->m_scope_id) return 1;
    if (first->m_transaction_id > second->m_transaction_id) return -1;
    if (first->m_transaction_id < second->m_transaction_id) return 1;
    if (first->m_version > second->m_version) return -1;
    if (first->m_version < second->m_version) return 1;
    return 0;
}

Cacheable::Cacheable(DBNR::Cache* cache) { } //-V730 PVS

Cacheable::~Cacheable() { } //-V730 PVS

Info::Info( //-V730 PVS
    DBNR::Info_container* container,
    Tag tag,
    DBNR::Transaction_impl* transaction,
    Scope_id scope_id,
    Uint32 version,
    Element_base* element)
  : DB::Info_base(DB::Scope_id(), DB::Transaction_id(), 0),
    DB::Cacheable(NULL)
{
    // Not to be used in DBLIGHT.
    MI_ASSERT(false);
}

Info::Info( //-V730 PVS
    DBNR::Info_container* container,
    Tag tag,
    DBNR::Transaction_impl* transaction,
    Scope_id scope_id,
    Uint32 version,
    SCHED::Job* job)
  : DB::Info_base(DB::Scope_id(), DB::Transaction_id(), 0),
    DB::Cacheable(NULL)
{
    // Not to be used in DBLIGHT.
    MI_ASSERT(false);
}

Info::Info(
    DBLIGHT::Database_impl* database,
    DB::Tag tag,
    DB::Transaction* transaction,
    DB::Scope_id scope_id,
    Uint32 version,
    DB::Element_base* element)
  : DB::Info_base(scope_id, transaction->get_id(), version),
    DB::Cacheable(NULL),
    // unused
    m_container(NULL),               
    // used
    m_database(database),
    m_tag(tag),
    m_element(element),
    m_element_messages(NULL),
    m_job(NULL),
    m_job_messages(NULL),
    // unused    
    m_element_size(0),                           
    m_element_messages_size(0),                  
    m_job_size(0),                               
    m_job_messages_size(0),                      
    m_privacy_level(0),                   
    m_is_temporary(false),                             
    m_is_creation(false),                              
    m_is_deletion(false),                              
    m_is_job(false),                                   
    m_is_scope_deleted(false),                         
    m_offload_to_disk(false),                          
    // used
    m_pin_count_dblight(1),
    // unused
    m_named_tag_list(NULL),          
    m_creator_transaction(NULL),
    m_references_added(false)                        
{
    for(size_t i = 0; i < MAX_REDUNDANCY_LEVEL; ++i)
        m_owners[i] = 0;
}

Info::~Info()
{
    set_element(NULL);
    MI_ASSERT(m_element_messages == NULL);
    MI_ASSERT(m_job == NULL);
    MI_ASSERT(m_job_messages == NULL);

    m_database->decrement_reference_counts(m_references);
}

void Info::pin()
{
    ++m_pin_count_dblight;
}

void Info::unpin()
{
    if (--m_pin_count_dblight == 0)
        delete this;
}

Uint Info::get_pin_count() const
{
    return m_pin_count_dblight;
}

const char* Info::get_name() const { MI_ASSERT(false); return 0; }
Uint Info::get_nr_of_owners() const { return 1; }
bool Info::is_owner(NET::Host_id host_id) const { return true; }
bool Info::is_owned_by_us() const { return true; }
void Info::compact_owners() { }
NET::Host_id Info::get_first_owner() const { return 0; }
bool Info::remove_owner(NET::Host_id host_id) { MI_ASSERT(false); return true; }
bool Info::add_owner(NET::Host_id host_id) { MI_ASSERT(false); return 0; }
ptrdiff_t Info::offload() { MI_ASSERT(false); return 0; }

// Needs m_database->m_lock.
void Info::store_references()
{
    m_database->decrement_reference_counts(m_references);
    m_references.clear();

    if (m_element)
        m_element->get_references(&m_references);
    m_database->increment_reference_counts(m_references);
}

ptrdiff_t Info::set_element(Element_base* element)
{
    delete m_element;
    m_element = element;
    return 0;
}

ptrdiff_t Info::set_element_messages(DBNET::Message_list* element_messages)
{
    MI_ASSERT(false);
    return 0;
}

ptrdiff_t Info::set_job(SCHED::Job* job)
{
    MI_ASSERT(false);
    return 0;
}

ptrdiff_t Info::set_job_messages(DBNET::Message_list* job_messages)
{
    MI_ASSERT(false);
    return 0;
}

} // namespace DB

} // namespace MI
