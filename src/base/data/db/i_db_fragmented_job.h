/***************************************************************************************************
 * Copyright (c) 2008-2023, NVIDIA CORPORATION. All rights reserved.
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

/** \file i_db_fragmented_job.h
 ** \brief This declares the database Fragmented_job class.
 **
 ** This file contains the pure virtual base class for the Fragmented_job class which is the base
 ** class for all fragmented jobs.
 **/

#ifndef BASE_DATA_DB_I_DB_FRAGMENTED_JOB_H
#define BASE_DATA_DB_I_DB_FRAGMENTED_JOB_H

#include <base/data/serial/i_serial_serializable.h>
#include <base/data/serial/i_serial_classid.h>
#include <cstddef>
#include <base/system/main/i_module_id.h>
#include <base/lib/log/i_log_assert.h>

namespace mi {
namespace neuraylib {
class IJob_execution_context;
class IRDMA_buffer;
class IRDMA_context;
}
}

namespace MI {

namespace SERIAL { class Serializer; class Deserializer; }

// for backward compatibility
namespace SCHED { typedef mi::neuraylib::IJob_execution_context Job_execution_context; }

namespace DB {

class Transaction;

/// This is a job which can be executed immediately in fragments. It is not stored in the database
/// to avoid overhead. It needs to be serializable because it will be transmitted to other hosts.
/// The derived class can store any kind of parameter in the job. The execute_fragment needs to
/// figure out from the number of fragments and the current index which fragment it works on.
/// The execute_fragment function is the only one which needs to be implemented. In this case, the
/// job can be used locally, only and will not be delegated to other hosts.
class Fragmented_job : public SERIAL::Serializable
{
  public:
    /// Define the scheduling mode of the job
    /// The following modes are available:
    /// LOCAL: All fragments will be done on the local host.
    /// CLUSTER: The fragments will be spread across all hosts in the cluster.
    ///          If a fragment fails to execute on a given host (e.g. due to the host leaving
    ///          the cluster), it is re-assigned to a different host, so that fragments are
    ///          guaranteed to be executed.
    /// ONCE_PER_HOST: At most one fragment will be done per host. If less fragments are scheduled
    ///                then only some hosts will get fragments. This mode is not possible for GPUs.
    ///                NOTE: Only hosts which are in the same sub cluster are eligible.
    /// USER_DEFINED: The job implements an explicit assignment of fragments to host.
    ///               Fragments assigned to hosts which are unknown will be skipped. If a host
    ///               fails during fragment execution, its workload will not be re-run on another
    ///               machine.
    enum Scheduling_mode { LOCAL, CLUSTER, ONCE_PER_HOST, USER_DEFINED };

    /// virtual destructor
    virtual ~Fragmented_job() { }

    /// Get the scheduling mode
    ///
    /// \return LOCAL, CLUSTER, ONCE_PER_HOST, USER_DEFINED
    virtual Scheduling_mode get_scheduling_mode() const { return LOCAL; }

    /// Returns the CPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual float get_cpu_load() const { return 1.0f; }

    /// Returns the GPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual float get_gpu_load() const { return 0.0f; }

    /// Returns the priority of the job.
    ///
    /// The smaller the value the higher the priority of the job to be executed.
    ///
    /// \note Negative values are reserved for internal purposes (for the thread pool and the
    ///       fragment scheduler).
    virtual mi::Sint8 get_priority() const { return 0; }

    /// Returns the maximum number of threads that should be used to execute the fragments of this
    /// job.
    ///
    /// With certain job patterns a large number of threads might be used to execute as many
    /// fragments as possible in parallel. This property can be used to limit the numer of threads,
    /// potentially at the cost of performance. The special value 0 means \em no limit on the number
    /// of threads.
    virtual size_t get_thread_limit() const { return 0; }

    /// Indicates whether chunks can be delivered out of order
    ///
    /// \return  \c true if chunks can be delivered out of order
    ///          \c false if chunks must be delivered in order
    virtual bool get_allow_non_sequential_chunks() const { return false; }

    /// Execute one fragment of the job. This is executed on the calling host only. It will be
    /// executed on the original instance of the class and _not_ on any copy. So it can access all
    /// members of the job, even if they are not serialized.
    ///
    /// \param transaction              Execute in this transaction
    /// \param index                    Which fragment of the the job is this?
    /// \param count                    In how many fragments is this job split?
    virtual void execute_fragment(
        Transaction*    transaction,
        size_t          index,
        size_t          count)
    {
        ASSERT(M_SCH, !"Fragmented_job::execute_fragment is deprecated and should not be called.");
    }

    /// Execute one fragment of the job. This is executed on the calling host only. It will be
    /// executed on the original instance of the class and _not_ on any copy. So it can access all
    /// members of the job, even if they are not serialized.
    ///
    /// \param transaction                      Execute in this transaction
    /// \param index                            Which fragment of the the job is this?
    /// \param count                            In how many fragments is this job split?
    /// \param context                          Job execution context provides thread id, etc.
    virtual void execute_fragment(
        Transaction*                          transaction,
        size_t                                index,
        size_t                                count,
        const mi::neuraylib::IJob_execution_context* context) = 0;

    /// Cancel execution of all fragments of the fragmented job. The default implementation does
    /// nothing and thus on jobs not overriding the function the call will have no effect. Jobs
    /// which do override this can for example set some internal flag and terminate an ongoing loop
    /// if it is set.
    virtual void cancel() { }

    // The functions below are optional and need to be implemented only, if is_local_only returns
    // false!

    /// Execute one fragment of the job. This may be executed on a different host. So it can access
    /// only the serialized members of the job. It needs to return a Serializable, which then will
    /// be sent to the calling host and given to the receive_remote_result function.
    /// NOTE: This function _may_ be called on the calling host as well, if the job can be
    /// delegated. This might be used in a debug build to find problems with remote execution even
    /// in the local case.
    ///
    /// \param serializer               Write the results to this serializer
    /// \param transaction              Execute in this transaction
    /// \param index                    Which fragment of the the job is this?
    /// \param count                    In how many fragments is this job split?
    /// \param ctx                      Job execution context provides thread id, etc.
    virtual void execute_fragment_remote(
        SERIAL::Serializer*                 serializer,
        Transaction*                        transaction,
        size_t                              index,
        size_t                              count,
        const mi::neuraylib::IJob_execution_context* ctx) { }

    /// This function will get a Serializable as generated by the execute_fragment_remote function.
    /// It is executed on the calling host only. It is executed on the same object as the
    /// execute_fragment function. So again, it can access all the members of the original instance.
    ///
    /// \param deserializer             Get the received result from here
    /// \param transaction              Execute in this transaction
    /// \param index                    Which fragment of the the job is this?
    /// \param count                    In how many fragments is this job split?
    virtual void receive_remote_result(
        SERIAL::Deserializer*   deserializer,
        Transaction*            transaction,
        size_t                  index,
        size_t                  count) { }

    /// This will return the class id of the given object. This is needed by the serializer when it
    /// wants to write the class id in the stream. It is public because the smart pointers need to
    /// access it.
    ///
    /// \return                                 The class id
    virtual SERIAL::Class_id get_class_id() const { return 0; }

    /// This will serialize the object to the given serializer including all sub elements pointed to
    /// but serialized together with this object. The function must return a pointer behind itself
    /// (e.g. this + 1) to be able to serialize arrays.
    ///
    /// \param serializer                       Serialize to this serializer
    /// \return                                 Pointer behind the object (usually this + 1)
    virtual const SERIAL::Serializable *serialize(
        SERIAL::Serializer *serializer) const { return NULL; }

    /// This will deserialize the object from the given deserializer including all sub elements
    /// pointed to but serialized together with this object. The function must return a pointer
    /// behind itself (e.g. this + 1) to be able to serialize arrays.
    ///
    /// \param deserializer                     Deserialize from this deserializer
    /// \return                                 Pointer behind the object (usually this + 1)
    virtual SERIAL::Serializable *deserialize(
        SERIAL::Deserializer* deserializer) { return NULL; }    // deserialize from here

    /// Override the normal scheduling of fragments to host by providing an array of host ids
    /// corresponding to the index for the fragment to assign to that host. For example:
    /// 1 2 2 2 3 would mean fragment 0 is assigned to host 1, fragment 1 to host 2 etc.
    ///
    /// \param slots        Pointer to an array of host ids in the order of which
    ///                     fragment should be assigned to which host.
    /// \param nr_slots     The number of host ids in the array that the pointer \p slots
    ///                     points to.
    ///
    virtual void assign_fragments_to_hosts(
        Uint32 *slots,
        size_t nr_slots) { }

    /// Optional function to dump the contents of the element for debugging
    /// purposes to stdout.
    virtual void dump() const { }

    /// Optional function. Used to allocate an RDMA_buffer which will be used to store the result
    /// of a fragment execution on a remote host. This will only be called for fragments which are
    /// scheduled on remote hosts. The RDMA_buffer must have been obtained from a RDMA_context.
    ///
    /// The RDMA_buffer must at least be able to store the maximum expected result of the fragment 
    /// execution. If the function returns NULL, then RDMA shall not be used for this fragment.
    ///
    /// If this function requests RDMA then the execute_fragment_remote_rdma / 
    /// receive_remote_result_rdma pair will be used instead of the usual functionality.
    /// 
    /// Note that, if RDMA is not available on the system, the behaviour will be the same as if
    /// RDMA would be used, only the performance might be lower.
    ///
    /// \param rdma_context Context to be used to acquire RDMA buffers for the other host
    /// \param index The index for which the RDMA_buffer is requested
    /// \return The RDMA_buffer or NULL, if RDMA should not be used
    virtual mi::neuraylib::IRDMA_buffer* get_rdma_result_buffer(
        mi::neuraylib::IRDMA_context* rdma_context, size_t index) 
    { return NULL; }

    /// Execute one fragment of the job. This will be executed on a different host. So it can access
    /// only the serialized members of the job. This will only be called, if the 
    /// get_rdma_result_buffer function returned a buffer. The function can return an RDMA_buffer 
    /// previously allocated from an RDMA_context (e.g. the one provided in the 
    /// Job_execution_context) or NULL. In the latter case no data will be sent back.
    ///
    /// The same RDMA_buffer can be returned from multiple fragments and from different jobs, in 
    /// that case it will be sent back as the result of all the fragments using it.
    /// TODO: It is not completely clear if an RDMA_buffer can be shared.
    ///
    /// Note that the returned RDMA_buffer has to stay unchanged as long as the reception has not
    /// been confirmed from the receiving side. At this time the system will return the buffer
    /// to the RDMA_context, so that it can be used for further transmissions.
    ///
    /// \param transaction Execute in this transaction
    /// \param index Which fragment of the the job is this?
    /// \param count In how many fragments is this job split?
    /// \param rdma_context Context to be used to acquire RDMA buffers for the other host
    /// \param job_execution_context Job execution context provides thread id, etc.
    /// \return An RDMA_buffer or NULL
    virtual mi::neuraylib::IRDMA_buffer* execute_fragment_remote_rdma(Transaction* transaction, 
        size_t index, size_t count, mi::neuraylib::IRDMA_context* rdma_context,
        const mi::neuraylib::IJob_execution_context* job_execution_context) 
    { return NULL; }

    /// This function will get the RDMA_buffer generated by the get_rdma_result_buffer function.
    /// It is executed on the calling host only. It is executed on the same object as the
    /// execute_fragment function. So again, it can access all the members of the original instance.
    ///
    /// The function can process the RDMA_buffer immediately but it also can retain it and use it
    /// later, if necessary.
    ///
    /// \param buffer The RDMA_buffer holding the results
    /// \param transaction Execute in this transaction
    /// \param index Which fragment of the the job is this?
    /// \param count In how many fragments is this job split?
    virtual void receive_remote_result_rdma(mi::neuraylib::IRDMA_buffer* buffer, 
        Transaction* transaction, size_t index, size_t count)  { }
};

/// The listener for an asnychronous fragmented job execution. Can be used to asynchronously execute
/// a fragmented job and react when it is finished.
class IExecution_listener
{
public:
    /// Call when the execution was finished.
    virtual void job_finished() = 0;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_FRAGMENTED_JOB_H
