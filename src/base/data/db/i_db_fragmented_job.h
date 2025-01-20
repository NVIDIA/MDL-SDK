/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_FRAGMENTED_JOB_H
#define BASE_DATA_DB_I_DB_FRAGMENTED_JOB_H

#include <mi/base/types.h>

#include <base/system/main/i_assert.h>
#include <base/data/serial/i_serial_serializable.h>

namespace mi { namespace neuraylib {
class IJob_execution_context;
class IRDMA_buffer;
class IRDMA_context;
} }

namespace MI {

namespace SERIAL { class Serializer; class Deserializer; }

namespace DB {

class Transaction;

/// Base class for all fragmented jobs.
///
/// Each fragmented job must be derived from the this interface.
///
/// Fragmented jobs enable distributing compute intensive work to the CPUs and/or GPUs available in
/// a cluster environment.
///
/// Compared to a database jobs, fragmented jobs are lightweight means to achieve distributed,
/// parallel computing in a cluster. Fragmented jobs differ from database jobs as follows:
/// - A fragmented job is not stored in the database but instantiated on demand to be executed
///   immediately.
/// - Each instance of a fragmented job splits into a pre-defined number of fragments each of which
///   executes independently possibly on different CPUs and/or GPUs in the cluster.
/// - A fragmented job does not return a result but the execution function may operate on the
///   members of the fragmented job class instance to store results. Note that it is necessary to
///   synchronize accesses to those members from different fragments because they may be executed
///   in parallel.
///
/// In general, a fragmented job should be used if the result of the parallel algorithm is used only
/// once by the host initiating the execution. For instance, a fragmented job for rendering may
/// initiate the rendering of a single frame.
class Fragmented_job : public SERIAL::Serializable
{
public:
    /// \name General methods
    //@{

    /// Constants for possible scheduling modes.
    enum Scheduling_mode {
        /// All fragments will be done on the local host. In consequence, dummy implementations for
        /// #serialize() and #deserialize() as well as for execute_fragment_remote() and
        /// #receive_remote_result() suffice.
        LOCAL = 0,
        /// The fragments will be spread across all hosts in the cluster. If a fragment fails to
        /// execute on a given host (for example, due to a host leaving the cluster), it is
        /// re-assigned to a different host such that fragments are guaranteed to be executed.
        ///
        /// \note Although DiCE tries to distribute fragments in a fair manner to the cluster nodes,
        ///       there is no guarantee that this will be the case.
        CLUSTER = 1,
        /// At most one fragment will be done per remote host. If the specified number of fragments
        /// is larger than zero and less than the number of remote hosts then only some hosts will
        /// get a fragment. To ensure that exactly one fragment will be done per remote host the
        /// number of fragments should be set to the special value 0. This mode is not possible for
        /// jobs which need a GPU.
        ///
        /// \note Only hosts which are in the same sub cluster are eligible for executing a
        ///       fragment of the job.
        ONCE_PER_HOST = 2,
        /// The job implements an explicit assignment of fragments to hosts and must implement the
        /// #assign_fragments_to_hosts() function to fill out all slots. Fragments assigned to hosts
        /// which are unknown will be skipped. If a host fails during fragment execution, its
        /// workload will \em not be re-assigned to a different host
        USER_DEFINED = 3,
    };

    /// Destructor.
    virtual ~Fragmented_job() { }

    /// Returns the scheduling mode of the job.
    virtual Scheduling_mode get_scheduling_mode() const { return LOCAL; }

    /// Returns the CPU load per fragment of the fragmented job.
    ///
    /// Typically 1.0 for CPU jobs and 0.0 for GPU jobs. A value larger than 1.0 might be used for
    /// jobs that concurrently use multiple threads per fragment, e.g., if OpenMP or MPI is used. A
    /// value between 0.0 and 1.0 might be used for jobs that do not much work themselves, but are
    /// rather used as synchronization primitive.
    ///
    /// \note This value must \em never change for a given instance of the fragmented job.
    virtual float get_cpu_load() const { return 1.0f; }

    /// Returns the GPU load per fragment of the fragmented job.
    ///
    /// Typically 0.0 for CPU jobs and 1.0 for GPU jobs. A value larger than 1.0 might be used for
    /// jobs that concurrently use multiple GPUs per fragment. A value between 0.0 and 1.0 might be
    /// used for jobs that do not much work themselves, but are rather used as synchronization
    /// primitive.
    ///
    /// \note This value must \em never change for a given instance of the fragmented job.
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
    /// fragments as possible in parallel. This property can be used to limit the number of threads,
    /// potentially at the cost of performance. The special value 0 means \em no limit on the number
    /// of threads.
    virtual size_t get_thread_limit() const { return 0; }

    /// Indicates whether chunks can be delivered out of order
    virtual bool get_allow_non_sequential_chunks() const { return false; }

    /// Executes one of many fragments of the fragmented job on the local host.
    ///
    /// Executes one fragment of the many fragments spawned by the fragmented job. This method is
    /// used for fragments executed on the calling host only and, thus, operates on the original
    /// instance of the fragmented job class and not on a copy.
    ///
    /// Note that other fragments operating on the same instance might be executed in parallel.
    /// Therefore all accesses to instance data need to be properly serialized.
    ///
    /// \see #execute_fragment_remote() and #receive_remote_result() are the counterpart of this
    ///      method for remote execution
    /// \see #execute_fragment_remote_rdma() and #receive_remote_result_rdma() are the counterparts
    ///      of this method for remote execution using RDMA
    ///
    /// \param transaction  The transaction in which the fragmented job is executed. The transaction
    ///                     can be used to access database elements and database jobs required for
    ///                     execution but should not be used to edit or create tags. Might be
    ///                     \c nullptr if the fragmented job was started without transaction.
    /// \param index        The index of the fragment to be executed. The value is in the range from
    ///                     0 to \p count-1.
    /// \param count        The number of fragments into which the fragmented job is split.
    /// \param context      The context in which the fragmented job is executed.
    virtual void execute_fragment(
        Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context) = 0;

    /// Cancels the execution of not yet completed jobs.
    ///
    /// This method is called if the job has been submitted for execution, but not all its fragments
    /// have been completed when the transaction is closed or aborted. If the job result is no
    /// longer needed in such a case, this notification can be used to terminate currently running
    /// fragments and/or skip the execution of not yet started fragments.
    virtual void cancel() { }

    /// Dumps the contents of the fragmented job for debugging purposes.
    virtual void dump() const { }

    //@}
    /// \name Methods for remote execution
    //@{

    /// Returns the class ID of the fragmented job.
    virtual SERIAL::Class_id get_class_id() const { return 0; }

    /// See mi::neuraylib::IFragmented_job::serialize();
    virtual const SERIAL::Serializable* serialize(
        SERIAL::Serializer* serializer) const
    { MI_ASSERT( !"Required for non-LOCAL scheduling mode"); return nullptr; }

    /// See mi::neuraylib::IFragmented_job::deserialize();
    virtual SERIAL::Serializable* deserialize(
        SERIAL::Deserializer* deserializer)
    { MI_ASSERT( !"Required for non-LOCAL scheduling mode"); return nullptr; }

    /// See mi::neuraylib::IFragmented_job::assign_fragments_to_hosts().
    virtual void assign_fragments_to_hosts(
        mi::Uint32* slots,
        size_t nr_slots)
    { MI_ASSERT( !"Required if get_scheduling_mode() returns USER_DEFINED"); }

    /// See mi::neuraylib::IFragmented_job::execute_fragment_remote().
    virtual void execute_fragment_remote(
        SERIAL::Serializer* serializer,
        Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context)
    { MI_ASSERT( !"Required for non-LOCAL scheduling mode"); }

    /// See mi::neuraylib::IFragmented_job::receive_remote_result().
    virtual void receive_remote_result(
        SERIAL::Deserializer* deserializer,
        Transaction* transaction,
        size_t index,
        size_t count)
    { MI_ASSERT( !"Required for non-LOCAL scheduling mode"); }

    /// See mi::neuraylib::IFragmented_job::get_rdma_result_buffer().
    virtual mi::neuraylib::IRDMA_buffer* get_rdma_result_buffer(
        mi::neuraylib::IRDMA_context* rdma_context, size_t index)
    { return nullptr; }

    /// See mi::neuraylib::IFragmented_job::execute_fragment_remote_rdma().
    virtual mi::neuraylib::IRDMA_buffer* execute_fragment_remote_rdma(
        Transaction* transaction,
        size_t index,
        size_t count,
        mi::neuraylib::IRDMA_context* rdma_context,
        const mi::neuraylib::IJob_execution_context* job_execution_context)
    { MI_ASSERT( !"Required if get_rdma_result_buffer() returns a valid pointer"); return nullptr; }

    /// See mi::neuraylib::IFragmented_job::receive_remote_result_rdma().
    virtual void receive_remote_result_rdma( mi::neuraylib::IRDMA_buffer* buffer,
        Transaction* transaction, size_t index, size_t count)
    { MI_ASSERT( !"Required if get_rdma_result_buffer() returns a valid pointer"); }

    //@}
};

/// Abstract interface for asynchronous execution of fragmented jobs.
class IExecution_listener
{
public:
    /// Invoked when the fragmented job has been executed.
    virtual void job_finished() = 0;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_FRAGMENTED_JOB_H
