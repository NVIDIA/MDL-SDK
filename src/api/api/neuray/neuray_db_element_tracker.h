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
 ** \brief Header for the Db_element_tracker.
 **/

#ifndef API_API_NEURAY_NEURAY_DB_ELEMENT_TRACKER_H
#define API_API_NEURAY_NEURAY_DB_ELEMENT_TRACKER_H

#include "i_neuray_db_element.h"

#include <mi/base/lock.h>
#include <string>


namespace MI { namespace HTTP { class Connection; } }

namespace MI {

namespace NEURAY {

class Db_element_impl_base;

/// The tracker can be used to monitor DB elements in use by the API. The constructor and
/// destructor of Db_element_impl record these events with the tracker. The tracker installs a
/// callback for the admin HTTP server to dump all DB elements currently in use by the API.
class Db_element_tracker {

public:
    /// Constructor.
    Db_element_tracker();

    /// Destructor.
    ~Db_element_tracker();

    /// Record the construction of (an API class for) a DB element.
    ///
    /// The first invocation of this method also installs the callback handler for the admin HTTP
    /// server.
    void add_element( const Db_element_impl_base* db_element);

    /// Record the destruction of (an API class for) a DB element.
    void remove_element( const Db_element_impl_base* db_element);

    /// Returns a string representation of the element state.
    static std::string state_to_string( Db_element_state state);

    /// Returns a string representation of the journal flags.
    static std::string flags_to_string( DB::Journal_type flags);

private:


    /// Indicates whether the callback has already been installed.
    bool m_initialized;

    /// Contains the DB elements currently in use by the API.
    std::set<const Db_element_impl_base*> m_elements;

    /// Lock for the set above.
    mi::base::Lock m_lock;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_DB_ELEMENT_TRACKER_H

