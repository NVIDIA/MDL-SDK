/***************************************************************************************************
 * Copyright (c) 2006-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_INFO_H
#define BASE_DATA_DB_I_DB_INFO_H

#include <boost/core/noncopyable.hpp>

#include "i_db_scope.h"
#include "i_db_tag.h"
#include "i_db_transaction_id.h"

namespace MI {

namespace SCHED { class Job; }

namespace DB {

class Element_base;

/// Metadata that the database stores with each element or job.
class Info : private boost::noncopyable
{
public:
    /// Pins the info, i.e., increments its reference count.
    virtual void pin() = 0;

    /// Unpins the info, i.e., decrements its reference count.
    virtual void unpin() = 0;

    /// Indicates whether this info holds a job or an element.
    virtual bool get_is_job() const = 0;

    /// Returns the element held by this info.
    ///
    /// Returns \c NULL if the info holds a job that has not yet been executed, or if the element
    /// is still serialized.
    virtual Element_base* get_element() const = 0;

    /// Returns the job held by this info.
    ///
    /// Returns \c NULL if the info does not hold a job, or if the job is still serialized.
    virtual SCHED::Job* get_job() const = 0;

    /// Returns the tag.
    virtual Tag get_tag() const = 0;

    /// Returns the ID of the corresponding scope.
    virtual Scope_id get_scope_id() const = 0;

    /// Returns the ID of the creating transaction.
    virtual Transaction_id get_transaction_id() const = 0;

    /// Returns the version of the tag in the creating transaction.
    virtual mi::Uint32 get_version() const = 0;

    /// Returns the privacy level.
    virtual Privacy_level get_privacy_level() const = 0;

    /// Returns the name associated with the tag (or \c NULL).
    virtual const char* get_name() const = 0;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_INFO_H
