/***************************************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/lib/log/i_log_logger.h>

namespace MI {

namespace DB {

Access_base::Access_base( const Access_base& other)
  : m_element( nullptr),
    m_transaction( nullptr),
    m_info( nullptr),
    m_is_edit( false)
{
    set_access( other);
}

Access_base& Access_base::operator=( const Access_base& other)
{
    MI_ASSERT( !m_is_edit);
    set_access( other);
    return *this;
}

Access_base::~Access_base()
{
    cleanup();
    if( m_transaction)
        m_transaction->unpin();
}

Tag_version Access_base::get_tag_version() const
{
    if( !m_info)
        return Tag_version();

    return Tag_version( m_tag, m_info->get_transaction_id(), m_info->get_version());
}

const SCHED::Job* Access_base::get_job() const
{
    return m_info && m_info->get_is_job() ? m_info->get_job() : nullptr;
}

Element_base* Access_base::set_access( Tag tag, Transaction* transaction, SERIAL::Class_id id)
{
    cleanup();

    m_tag = tag;

    if( transaction) {
        if( m_transaction)
            m_transaction->unpin();
        m_transaction = transaction;
        m_transaction->pin();
    }

    if( !tag) {
        m_element = nullptr;
        m_info = nullptr;
        return nullptr;
    }

    m_info = m_transaction->access_element( tag);
    if( m_info) {
        m_element = m_info->get_element();
        return m_element;
    }

    // Unclear whether this location can be reached.
    MI_ASSERT( !"Unexpected creation of dummy element");
    LOG::mod_log->debug( M_DB, LOG::Mod_log::C_DATABASE, "Access will return an empty element.");
    m_element = m_transaction->construct_empty_element( id);
    return m_element;
}

Element_base* Access_base::set_edit(
    Tag tag,
    Transaction* transaction,
    SERIAL::Class_id id,
    Journal_type journal_type)
{
    cleanup();

    m_tag = tag;
    m_journal_type = journal_type;
    m_is_edit = true;

    if( transaction) {
        if( m_transaction)
            m_transaction->unpin();
        m_transaction = transaction;
        m_transaction->pin();
    }

    if( !tag) {
        m_element = nullptr;
        m_info = nullptr;
        return nullptr;
    }

    m_info = m_transaction->edit_element( tag);
    if( m_info) {
        m_element = m_info->get_element();
        return m_element;
    }

    // Unclear whether this location can be reached.
    MI_ASSERT( !"Unexpected creation of dummy element");
    LOG::mod_log->debug( M_DB, LOG::Mod_log::C_DATABASE, "Edit will return an empty element.");
    m_element = m_transaction->construct_empty_element( id);
    return m_element;
}

void Access_base::clear_transaction()
{
    MI_ASSERT( !m_is_edit);
    if( m_transaction) {
        m_transaction->unpin();
        m_transaction = nullptr;
    }
}

void Access_base::cleanup()
{
    if( m_is_edit) {

        if( m_info) {
            m_transaction->finish_edit( m_info, m_journal_type);
            m_info = nullptr;
            m_element = nullptr;
        } else {
            // Delete the dummy element we might have created.
            delete m_element;
            m_element = nullptr;
        }

        m_is_edit = false;

    } else {

        if( m_info) {
            m_info->unpin();
            m_info = nullptr;
            m_element = nullptr;
        } else {
            // Delete the dummy element we might have created.
            delete m_element;
            m_element = nullptr;
        }
    }

    m_journal_type = JOURNAL_NONE;
}

Element_base* Access_base::set_access( const Access_base& other)
{
    cleanup();

    m_tag = other.m_tag;

    if( m_transaction)
        m_transaction->unpin();
    m_transaction = other.m_transaction;
    if( m_transaction)
        m_transaction->pin();

    m_info = other.m_info;
    m_journal_type = other.m_journal_type;

    if( !m_info) {
        m_element = nullptr;
        return nullptr;
    }

    m_info->pin();
    m_element = m_info->get_element();
    return m_element;
}

} // namespace DB

} // namespace MI
