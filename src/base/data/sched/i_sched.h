/**************************************************************************************************
 * Copyright (c) 2004-2025, NVIDIA CORPORATION. All rights reserved.
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
 *************************************************************************************************/

#ifndef BASE_DATA_SCHED_I_SCHED_H
#define BASE_DATA_SCHED_I_SCHED_H

#include <string>

#include <mi/base/types.h>

#include <base/data/db/i_db_tag.h>
#include <base/data/serial/i_serial_serializable.h>

namespace MI {

namespace DB { class Element_base; class Transaction; }

namespace SCHED {

/// Base class for database jobs.
///
/// Database jobs are pieces of work (jobs) that can be stored in the database and are executed by
/// the job scheduler on access. Users can derive their job classes from this base class, but it is
/// recommended to derive from the mixin class below.
///
/// Note that there is no dependency tracking between job result and job inputs (typically database
/// elements). Is is up to the user to invalidate stale job results via
/// #DB::Transaction::invalid_job_results().
///
/// Database jobs are not reference-counted. Storing them in the database passes ownership from
/// the creator to the database.
class Job_base : public SERIAL::Serializable
{
public:
    /// The class ID for this base class.
    static const SERIAL::Class_id id = 0;

    /// Indicates whether this object is an instance of the given class.
    ///
    /// \param arg_id   The class given by its class ID.
    /// \return         \c true if this object is directly or indirectly derived from the given
    ///                 class.
    virtual bool is_type_of( SERIAL::Class_id arg_id) const { return arg_id == 0; }

    /// Returns a deep copy of the job.
    ///
    /// \return   The new copy of the element. RCS:TRO
    virtual Job_base* copy() const = 0;

    /// Return a human readable version of the class ID.
    virtual std::string get_class_name() const = 0;

    /// Returns the size of this job.
    virtual size_t get_size() const = 0;

    /// Tells the scheduler about tags needed for execution.
    ///
    /// This information can be used for improved performance in the cluster.
    ///
    /// \param[out] tag_array   The vector of tags.
    /// \param count            Size of \c tag_array.
    /// \return                 The number of valid tags in \c tag_array.
    virtual mi::Uint32 pre_exec( DB::Tag* tag_array, mi::Uint32 count) { return 0; }

    /// This method executes the jobs and returns a DB element as result.
    virtual DB::Element_base* execute( DB::Transaction* transaction) const = 0;

    /// Indicates whether the job result is shared between different transactions.
    virtual bool get_is_shared() const { return false; }

    /// Indicates whether the jobs creates other database elements and/or jobs.
    ///
    /// These elements and jobs need special attention w.r.t. invalidation, garbage collection, and
    /// synchronization in the cluster.
    ///
    /// Repeated executions of the job must be consistent in the number/sequence/purpose of
    /// created database elements and/or jobs.
    virtual bool get_is_parent() const { return false; }

    /// Indicates whether the job is to be executed locally on each node that needs its result
    /// instead of making the result of the first execution available in the cluster.
    virtual bool get_is_local() const { return false; }

    /// Returns the CPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual float get_cpu_load() const { return 1.0f; }

    /// Returns the GPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual float get_gpu_load() const { return 0.0f; }
};

/// Helper function that defines some methods for derived classes.
///
/// Examples:
///
//// - If the class shall be directly derived from Job:
///     class My_job : public Job<My_job, 1000> { ... }
///
/// - If the class shall be derived from some other class which is derived from Job:
///     class My_derived_job : public Job<My_derived_job, 1001, My_job> { ... }
template <class T, SERIAL::Class_id ID = 0, class P = Job_base>
class Job : public P
{
public:
    /// Class ID of this class
    static const SERIAL::Class_id id = ID;

    /// Factory function.
    static SERIAL::Serializable* factory() { return new T; }

    /// Default constructor.
    Job() : P() { }

    /// Copy constructor.
    Job( const Job& other) : P( other) { }

    /// Returns the class ID of this class.
    SERIAL::Class_id get_class_id() const { return id; }

    /// Returns a deep copy of the job.
    ///
    /// \return   The new copy of the element. RCS:TRO
    Job_base* copy() const { return new T( * reinterpret_cast<const T*>( this)); }

    /// Returns the size of this job.
    size_t get_size() const { return sizeof( *this); }

    /// Checks whether the class ID of this instance matches the specified class ID
    bool is_type_of( SERIAL::Class_id arg_id) const
    {
        return arg_id == ID ? true : P::is_type_of( arg_id);
    }
};

} // namespace SCHED

} // namespace MI

#endif  // BASE_DATA_SCHED_I_SCHED_H
