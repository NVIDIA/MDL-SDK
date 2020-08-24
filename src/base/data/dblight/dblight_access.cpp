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
 ** \brief Remaining definitions of methods of DB::Access_base.
 **/

#include "pch.h"

#include <base/system/main/i_assert.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_info.h>

namespace MI {

namespace DB {

Access_base::Access_base()
  : m_pointer(nullptr),
    m_transaction(nullptr),
    m_info(nullptr),
    m_is_edit(false)
{
}

Access_base::Access_base(const Access_base& access)
  : m_pointer(nullptr),
    m_transaction(nullptr),
    m_info(nullptr),
    m_is_edit(false)
{
    set_access(access);
}

Access_base& Access_base::operator=(const Access_base& access)
{
    set_access(access);
    MI_ASSERT(!m_is_edit);
    return *this;
}

Access_base::~Access_base()
{
    cleanup();
    if (m_transaction)
        m_transaction->unpin();
}

Element_base* Access_base::set_access(
    Tag tag,
    Transaction* transaction,
    SERIAL::Class_id id,
    bool wait)
{
    cleanup();

    m_tag = tag;
    if (transaction) {
        if (m_transaction)
            m_transaction->unpin();
        m_transaction = transaction;
        m_transaction->pin();
    }

    if (tag.is_invalid()) {
        // does not point to anything, anymore
        m_pointer = nullptr;
        m_info = nullptr;
        return nullptr;
    }

    // lookup the tag in the context of the current transaction
    m_info = m_transaction->get_element(tag, wait);
    if (m_info) {
        m_pointer = m_info->get_element();
        return m_pointer;
    }

#if 1
    MI_ASSERT(false);
#else
    LOG::mod_log->debug(M_DB, LOG::Mod_log::C_DATABASE,
        "Access will return empty element (transaction no longer open or "
        "fragmented jobs have been cancelled).");
#endif
    m_pointer = m_transaction->construct_empty_element(id); //-V779 PVS
    return m_pointer;
}

Element_base* Access_base::set_access(const Access_base& source)
{
    cleanup();

    m_tag = source.m_tag;
    if (m_transaction)
        m_transaction->unpin();
    m_transaction = source.m_transaction;
    if (m_transaction)
        m_transaction->pin();
    m_info = source.m_info;
    m_journal_type = source.m_journal_type;

    if (!m_info) {
        m_pointer = nullptr;
        return nullptr;
    }

    m_info->pin();
    m_pointer = m_info->get_element();
    return m_pointer;
}

Element_base* Access_base::set_edit(
    Tag tag,
    Transaction* transaction,
    SERIAL::Class_id id,
    Journal_type journal_type)
{
    cleanup();

    m_is_edit = true;
    m_tag = tag;
    m_journal_type = journal_type;
    if (transaction) {
        if (m_transaction)
            m_transaction->unpin();
        m_transaction = transaction;
        m_transaction->pin();
    }

    if (tag.is_invalid()) {
        // does not point to anything, anymore
        m_pointer = nullptr;
        m_info = nullptr;
        return nullptr;
    }

    m_info = m_transaction->edit_element(tag);
    if (m_info) {
        m_pointer =  m_info->get_element();
        return m_pointer;
    }

    m_pointer = m_transaction->construct_empty_element(id);
    return m_pointer;
}

const SCHED::Job* Access_base::get_job() const
{
    MI_ASSERT(false);
    if (m_tag.is_invalid() || !m_transaction || !m_info->get_is_job()) //-V779 PVS
        return nullptr;
    return m_info->get_job();
}

Tag_version Access_base::get_tag_version() const
{
    if (!m_info)
        return Tag_version();

    Tag_version result;
    result.m_tag = m_tag;
    result.m_transaction_id = m_info->get_transaction_id();
    result.m_version = m_info->get_version();
    return result;
}

void Access_base::clear_transaction()
{
    if (m_transaction) {
        m_transaction->unpin();
        m_transaction = 0;
    }
}

void Access_base::cleanup()
{
    if (m_is_edit) {
        if (m_info) {
            // cleanup after an edit was finished. This includes updating references,
            // sending data over the network etc.
            m_transaction->finish_edit(m_info, m_journal_type);
            m_info = nullptr;
        }
        m_is_edit = false;
    } else {
        // unpin old info, if any
        if (m_info) {
            m_info->unpin();
        } else {
            // This is not valid. We might have returned a dummy element, though,
            // which we have to delete, now.
            delete m_pointer;
        }
    }
}

} // namespace DB

} // namespace MI
