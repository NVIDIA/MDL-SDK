/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief The Info class for the database stores everything the database needs to know about
///        one version of one data element or job.
///
/// IMPORTANT NOTE: This file is only meant to be included by db itself and the sched module. It
/// contains functionality which is not meant to be exposed to other modules.


#ifndef BASE_DATA_DB_I_DB_INFO_H
#define BASE_DATA_DB_I_DB_INFO_H

#include "i_db_cacheable.h"
#include "i_db_transaction.h"

#include <base/lib/cont/i_cont_set.h>
#include <set>
#include <base/system/stlext/i_stlext_intrusive_ptr.h>

namespace MI {

namespace DBNET { class Message_list; }
namespace SCHED { class Job; }
namespace DBNR
{
    class Info_container;
    class Transaction_impl;
    class Named_tag_list;
    typedef STLEXT::Intrusive_ptr<Transaction_impl> Transaction_impl_ptr;
}
namespace DBLIGHT { class Database_impl; }

namespace DB {

/// Base class of Info.
///
/// This is used when storing an info in an info list. Using this class speeds up searching, because
/// when using an Info_base as a pattern for tree search, we avoid constructing all parts of an
/// Info, especially the lock.
class Info_base
{
public:
    /// Default constructor.
    Info_base() : m_scope_id(0), m_transaction_id(0), m_version(0) { }

    /// Constructor.
    Info_base(Scope_id scope_id, Transaction_id transaction_id, Uint32 version)
      : m_scope_id(scope_id), m_transaction_id(transaction_id), m_version(version) { }

    /// Sets the  ID of the scope this Info_base belongs to.
    void set_scope_id(Scope_id scope_id) { m_scope_id = scope_id; }

    /// Returns the ID of the scope this Info_base belongs to.
    Scope_id get_scope_id() { return m_scope_id; }

    /// Sets the ID of the creating transaction.
    void set_transaction_id(Transaction_id transaction_id) { m_transaction_id = transaction_id; }

    /// Return the ID of the creating transaction.
    Transaction_id get_transaction_id() { return m_transaction_id; }

    /// Set the version of the tag in the transaction.
    void set_version(Uint32 version) { m_version = version; }

    /// Returns the version of the tag in the transaction.
    Uint32 get_version() { return m_version; }

    /// Compares two instances by lexicographic comparison of scope ID, transaction ID, and version.
    ///
    /// Returns -1 if \p first is larger (!) than \p second, +1 if \p first is smaller than
    /// \p second, and 0 otherwise.
    static int compare(Info_base* first, Info_base* second);

    CONT::Set_link<Info_base> m_link;           ///< Link for the Info_list
    CONT::Set_link<Info_base> m_name_link;      ///< Link for the Named_tag_list
    CONT::Dlist_link<Info_base> m_remove_link;  ///< Link for removal list in Scope

protected:
    Scope_id m_scope_id;                        ///< ID of the scope this Info_base belongs to
    Transaction_id m_transaction_id;            ///< ID of the creating transaction
    Uint32 m_version;                           ///< Version of the tag in the creating transaction
};

/// The info holds all information about one version of a database tag.
///
/// See documentation of the fields for limitations w.r.t. DBNR and DBLIGHT.
class Info : public Info_base, public Cacheable
{
public:
    /// \name Constructors, destructors, and reference counting
    //@{

    /// Constructor for an element (as used by DBNR)
    Info(
        DBNR::Info_container* container,        ///< The info container this Info belongs to
        Tag tag,                                ///< The tag this Info belongs to
        DBNR::Transaction_impl* transaction,    ///< Creating transaction
        Scope_id scope_id,                      ///< ID of scope this Info belongs to
        Uint32 version,                         ///< Version of the tag in the creating transaction
        Element_base* element = NULL);          ///< The element to be stored

    /// Constructor for a job (as used by DBNR)
    Info(
        DBNR::Info_container* container,        ///< The info container this Info belongs to
        Tag tag,                                ///< The tag this Info belongs to
        DBNR::Transaction_impl* transaction,    ///< Creating transaction
        Scope_id scope_id,                      ///< ID of scope this Info belongs to
        Uint32 version,                         ///< Version of the tag in the creating transaction
        SCHED::Job* job);                       ///< The job to be stored

    /// Constructor for an element (as used by DBLIGHT)
    Info(
        DBLIGHT::Database_impl* database,       ///< The database this Info belongs to
        Tag tag,                                ///< The tag this Info belongs to
        DB::Transaction* transaction,           ///< Creating transaction
        Scope_id scope_id,                      ///< ID of scope this Info belongs to
        Uint32 version,                         ///< Version of the tag in the creating transaction
        Element_base* element = NULL);          ///< The element to be stored

    /// Destructor
    ~Info();

    /// Pins the info, i.e., increments its reference count.
    void pin();

    /// Unpins the info, i.e., decrements its reference count.
    void unpin();

    // Returns the current pin count.
    Uint get_pin_count() const;

    //@}
    /// \name Container, tag, name, and flags
    //@{

    /// Returns the info container that owns this Info.
    DBNR::Info_container* get_container() const { return m_container; }

    /// Returns the tag.
    Tag get_tag() const { return m_tag; }

    /// Returns the name associated with the tag (or \c NULL).
    const char* get_name() const;

    /// Sets the privacy level.
    void set_privacy_level( Privacy_level privacy_level) { m_privacy_level = privacy_level; }

    /// Returns the privacy level.
    Privacy_level get_privacy_level() const { return m_privacy_level; }

    /// Sets the flag for temporary tags.
    void set_is_temporary( bool is_temporary) { m_is_temporary = is_temporary; }

    /// Indicates whether tag is a temporary tag.
    bool get_is_temporary() const { return m_is_temporary; }

    /// Sets the flag for tag creation.
    void set_is_creation( bool is_creation) { m_is_creation = is_creation; }

    /// Indicates whether this is the creation of a tag.
    bool get_is_creation() const { return m_is_creation; }

    /// Sets the flag for tag deletion.
    void set_is_deletion( bool is_deletion) { m_is_deletion = is_deletion; }

    /// Indicates whether this is the deletion of a tag.
    bool get_is_deletion() const { return m_is_deletion; }

    /// Sets the flag that distinguishes jobs and elements.
    void set_is_job( bool is_job) { m_is_job = is_job; }

    /// Indicates whether this is a job or a element.
    bool get_is_job() const { return m_is_job; }

    /// Sets the flag that distinguishes jobs and elements.
    void set_is_scope_deleted(bool is_scope_deleted) { m_is_scope_deleted = is_scope_deleted; }

    /// Indicates whether this is a job or a element.
    bool get_is_scope_deleted() const { return m_is_scope_deleted; }

    /// Sets the offload-to-disk flag.
    void set_offload_to_disk( bool offload_to_disk)
    { m_offload_to_disk = offload_to_disk; }

    /// Returns the offload-to-disk flag.
    bool get_offload_to_disk() const { return m_offload_to_disk; }

    //@}
    /// \name Element and element messages
    ///
    /// \note Both setters and getters require the caller to hold m_lock. Pinning the info is
    ///       sufficient to guarantee that the element (but not the message) remains valid. Due to
    ///       lock ordering, m_lock requires the lock of the info container.
    //@{

    /// Sets m_element and keeps track of its memory usage.
    ///
    /// Pass \c NULL to clear m_element. The previous element (if any) is deleted.
    ///
    /// \return The difference in memory usage (as reported by Element_base::get_size()).
    ptrdiff_t set_element(Element_base* element);

    /// Returns the element or \c NULL if still serialized or for a job.
    Element_base* get_element() const { return m_element; }

    /// Sets m_element_messages and keeps track of its memory usage.
    ///
    /// Pass \c NULL to clear m_element_messages. The previous element messages (if any) is deleted.
    ///
    /// \return The difference in memory usage (as reported by DBNET::Message_list::get_size()).
    ptrdiff_t set_element_messages(DBNET::Message_list* element_messages);

    /// Returns the list of messages for an element or \c NULL if already deserialized of for a job.
    DBNET::Message_list* get_element_messages() const { return m_element_messages; }

    //@}
    /// \name Job and job messages
    ///
    /// \note Both setters and getters require the caller to hold m_lock. Pinning the info is
    ///       sufficient to guarantee that the job (but not the message) remains valid. Due to
    ///       lock ordering, m_lock requires the lock of the info container.
    //@{

    /// Sets m_job and keeps track of its memory usage.
    ///
    /// Pass \c NULL to clear m_job. The previous job (if any) is deleted.
    ///
    /// \return The difference in memory usage (as reported by SCHED::Job::get_size()).
    ptrdiff_t set_job(SCHED::Job* job);

    /// Returns the job or \c NULL if still serialized or for an element.
    SCHED::Job* get_job() const { return m_job; }

    /// Sets m_job_messages and keeps track of its memory usage.
    ///
    /// Pass \c NULL to clear m_job_messages. The previous job messages (if any) is deleted.
    ///
    /// \return The difference in memory usage (as reported by DBNET::Message_list::get_size()).
    ptrdiff_t set_job_messages(DBNET::Message_list* job_messages);

    /// Returns the list of messages for a job or \c NULL if already deserialized or for an element.
    DBNET::Message_list* get_job_messages() const { return m_job_messages; }

    //@}
    /// \name Query memory usage
    //@{

    /// Returns the cached element size.
    size_t get_element_size() const { return m_element_size;  }

    /// Returns the cached element message size.
    size_t get_element_messages_size() const { return m_element_messages_size; }

    /// Returns the cached job size.
    size_t get_job_size() const { return m_job_size; }

    /// Returns the cached job message size.
    size_t get_job_messages_size() const { return m_job_messages_size; }

    //@}
    /// \name Owners
    //@{

    /// Returns the number of owners.
    Uint get_nr_of_owners() const;

    /// Returns the i-th owner.
    NET::Host_id get_owner(int i) const { return m_owners[i]; }

    /// Checks whether \p host_id is among the owners.
    bool is_owner(NET::Host_id host_id) const;

    /// Checks if we are an owner.
    ///
    /// \note This method always returns \c true if the cluster consists of only one host.
    ///       Otherwise, it calls #is_owner() with the own host ID.
    bool is_owned_by_us() const;

    /// Returns the first owner, or 0 if there are none.
    NET::Host_id get_first_owner() const;

    /// Adds \p host_id to the owners.
    ///
    /// \return \c true in case success, i.e., \p host_id was not among the owners, and \c false
    ///         otherwise.
    bool add_owner(NET::Host_id host_id);

    /// Removes \p host_id from the owners.
    ///
    /// \return \c true in case success, i.e., \p host_id was among the owners, and \c false
    ///         otherwise.
    bool remove_owner(NET::Host_id host_id);

private:
    /// Compacts the array of owners so that the existing owners are at the start of the array.
    void compact_owners();
public:

    //@}
    /// \name Miscellaneous
    //@{

    /// Updates the memory usage.
    ///
    /// Decrements the memory usage the old (cached) element size, and increments it by the new
    /// (current) element size. Used by #Transaction_impl::finish_edit().
    void update_memory_usage();

    /// Offloads data to disk (for owners) or throws it away (for non-owners).
    ///
    /// Returns the delta in memory usage achieved by offloading (should be negative or zero).
    /// Should only be called if #get_offload_to_disk() returns \c true.
    ///
    /// This variant assumes that the info container lock is held, but not the lock of the info
    /// itself.
    ptrdiff_t offload();

    /// Offloads data to disk (for owners) or throws it away (for non-owners).
    ///
    /// Returns the delta in memory usage achieved by offloading (should be negative or zero).
    /// Should only be called if #get_offload_to_disk() returns \c true.
    ///
    /// This variant assumes that all required locks are held (in particular the lock of the info
    /// container and the lock of the info itself).
    ptrdiff_t offload_locked();

    /// Stores the references of m_element in m_references and increments the reference counts of
    /// the references.
    void store_references();

    //@}

private:

    /// \note The offload-to-disk flag is cached in the info because it is not available if the
    ///       element or job exists only in serialized form.
    ///
    /// \note The size of elements, jobs, and messages are cached to maintain consistency (even
    ///       if the total sum is temporarily slightly off):
    ///       - Edits of elements change the size (updated when the edit is finished).
    ///       - Deserialization of element messages and job messages steals the iovectors from the
    ///         messages without telling us (updated when the message is set again, typically
    ///         shortly afterwards set to NULL to delete the now empty message itself).
    ///       - Not needed for jobs, just for consistency.
    ///
    /// \note All accesses to #m_element, #m_element_messages, #m_job, and #m_job_messages need to
    ///       hold #m_lock. Pinning the info is sufficient to guarantee that #m_element and #m_job
    ///       remain valid (this does not hold for #m_element_messages and #m_job_messages). Note
    ///       the lock order: first the lock of the info container, then the lock of the cache, and
    ///       finally the lock of the info.
    ///
    /// \note This class does is not properly split into interface and implementation. DBNR uses all
    ///       of the fields below with the exception of those marked as "DBLIGHT only". DBLIGHT uses
    ///       only m_tag, m_element, m_references, and the those marked as "DBLIGHT only".

    DBNR::Info_container* m_container;                ///< Info container this Info belongs to
    DBLIGHT::Database_impl* m_database;               ///< DB this Info belongs to (DBLIGHT only)
    Tag m_tag;                                        ///< The tag this Info belongs to
    Element_base* m_element;                          ///< The deserialized element
    DBNET::Message_list* m_element_messages;          ///< The serialized element
    SCHED::Job* m_job;                                ///< The deserialized job
    DBNET::Message_list* m_job_messages;              ///< The serialized job
    size_t m_element_size;                            ///< Cached element size
    size_t m_element_messages_size;                   ///< Cached element messages size
    size_t m_job_size;                                ///< Cached job size
    size_t m_job_messages_size;                       ///< Cached job messages size
    Privacy_level m_privacy_level;                    ///< Privacy level
    bool m_is_temporary;                              ///< Flag for temporary tags
    bool m_is_creation;                               ///< Flag for tag creation
    bool m_is_deletion;                               ///< Flag for tag deletion
    bool m_is_job;                                    ///< Flag for jobs
    bool m_is_scope_deleted;                          ///< Is the scope already gone?
    bool m_offload_to_disk;                           ///< Flag for offloading data to disk
    mi::base::Atom32 m_pin_count_dblight;             ///< Pin count (DBLIGHT only)

public: // setter/getter methods still missing
    DBNR::Named_tag_list* m_named_tag_list;           ///< Named tag list used for get_name()
    DBNR::Transaction_impl_ptr m_creator_transaction; ///< Creating transaction
    NET::Host_id m_owners[MAX_REDUNDANCY_LEVEL];      ///< All known owners
    Tag_set m_references;                             ///< References held by element
    bool m_references_added;                          ///< References already added to database?
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_INFO_H

