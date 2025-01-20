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

#ifndef BASE_DATA_DBLIGHT_I_DBLIGHT_H
#define BASE_DATA_DBLIGHT_I_DBLIGHT_H

namespace MI {

namespace DB { class Database; }
namespace SERIAL { class Deserialization_manager; }
namespace THREAD_POOL { class Thread_pool; }

namespace DBLIGHT {

// External interface.

/// Creates an instance of the database.
///
/// The instances are independent, except for the statistics which are shared by all instances
/// (but statistics are disabled by default, see DBLIGHT_ENABLE_STATISTICS).
///
/// \param thread_pool               The thread pool to use, or \c nullptr to use an independent
///                                  thread pool instance.
/// \param deserialization_manager   The deserialization manager to use, or \c nullptr to use an
///                                  independent deserialization manager.
/// \param enable_journal            Indicates whether to enable the journal. Maintaining the
///                                  journal requires memory and time.
DB::Database* factory(
    THREAD_POOL::Thread_pool* thread_pool,
    SERIAL::Deserialization_manager* deserialization_manager,
    bool enable_journal);

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_I_DBLIGHT_H
