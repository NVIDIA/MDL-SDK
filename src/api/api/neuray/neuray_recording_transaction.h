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
 ** \brief Header for the Recording_transaction implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_RECORDING_TRANSACTION_H
#define API_API_NEURAY_NEURAY_RECORDING_TRANSACTION_H

#include <base/data/db/i_db_transaction_wrapper.h>

#include <vector>

namespace MI {

namespace NEURAY {

/// Wraps a DB::Transaction to record the tags of all stored elements.
///
/// All store() methods are intercepted to record the tag of the stored element. The vector
/// of all tags stored so far can be obtained via #get_stored_tags().
class Recording_transaction : public DB::Transaction_wrapper
{
public:

    // public API methods

    // (none)

    // internal methods

    /// Constructor
    ///
    /// \param db_transaction   The wrapped transaction
    Recording_transaction( DB::Transaction* db_transaction);

    /// Obtains the stored tags
    const std::vector<DB::Tag>& get_stored_tags() const;

    // re-implemented methods from DB::Transaction_wrapper

    DB::Info* edit_element( DB::Tag tag);

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

    bool remove(
        DB::Tag tag,
        bool remove_local_copy = false);

private:
    /// Stores the recorded tags
    std::vector<DB::Tag> m_tags;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_RECORDING_TRANSACTION_H
