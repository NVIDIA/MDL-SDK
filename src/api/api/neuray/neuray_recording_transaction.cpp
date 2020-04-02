/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the Recording_transaction implementation.
 **/

#include "pch.h"

#include "neuray_recording_transaction.h"

namespace MI {

namespace NEURAY {

Recording_transaction::Recording_transaction( DB::Transaction* db_transaction)
  : Transaction_wrapper( db_transaction)
{
}

const std::vector<DB::Tag>& Recording_transaction::get_stored_tags() const
{
    return m_tags;
}

DB::Info* Recording_transaction::edit_element( DB::Tag tag)
{
    m_tags.push_back( tag);
    return m_transaction->edit_element( tag);
}

DB::Tag Recording_transaction::store(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    DB::Tag tag = m_transaction->store( element, name, privacy_level, store_level);
    m_tags.push_back( tag);
    return tag;
}

void Recording_transaction::store(
    DB::Tag tag,
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    m_transaction->store( tag, element, name, privacy_level, journal_type, store_level);
    m_tags.push_back( tag);
}

DB::Tag Recording_transaction::store_for_reference_counting(
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Privacy_level store_level)
{
    DB::Tag tag = m_transaction->store_for_reference_counting(
        element, name, privacy_level, store_level);
    m_tags.push_back( tag);
    return tag;
}

void Recording_transaction::store_for_reference_counting(
    DB::Tag tag,
    DB::Element_base* element,
    const char* name,
    DB::Privacy_level privacy_level,
    DB::Journal_type journal_type,
    DB::Privacy_level store_level)
{
    m_transaction->store_for_reference_counting(
        tag, element, name, privacy_level, journal_type, store_level);
    m_tags.push_back( tag);
}

bool Recording_transaction::remove(
    DB::Tag tag,
    bool remove_local_copy)
{
    bool result = m_transaction->remove( tag, remove_local_copy);
    if( !result || remove_local_copy)
        return result;

    for( std::vector<DB::Tag>::iterator it = m_tags.begin(); it != m_tags.end(); ) {
        if( tag == *it)
          it = m_tags.erase( it);
        else
          ++it;
    }

    return result;
}

} // namespace NEURAY

} // namespace MI
