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
 ** \brief Implementation of the lightweight database.
 **
 ** This is an implementation of the database. It does only very few things.
 **/

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_IMPL_H
#define BASE_DATA_DBLIGHT_DBLIGHT_IMPL_H

#include <base/data/db/i_db_database.h>

#include <string>
#include <map>
#include <mi/base/atom.h>
#include <mi/base/lock.h>

namespace MI {

namespace DB { class Info; }


namespace DBLIGHT {

class Scope_impl;

/// Map of tags to infos
typedef std::map<DB::Tag, DB::Info*> Tag_map;

/// Map of names (strings) to tags
typedef std::map<std::string, DB::Tag> Named_tag_map;

/// Map of tags to names (strings)
typedef std::map<DB::Tag, std::string> Reverse_named_tag_map;

/// Set of tags flagged for removal
typedef std::set<DB::Tag> Flagged_for_removal_set;

/// Map of tags to reference count
typedef std::map<DB::Tag, Uint32> Reference_count_map;

/// Set of tags with reference count zero
typedef std::set<DB::Tag> Reference_count_zero_set;

/// The database class manages the whole database.
class Database_impl : public DB::Database
{
public:
    /// Constructor
    Database_impl();

    /// Destructor, empties the database
    ~Database_impl();

    // Implementation of the virtual database interface
    void prepare_close();
    void close();
    void garbage_collection();
    DB::Scope* get_global_scope();
    DB::Scope* lookup_scope(DB::Scope_id id);
    DB::Scope* lookup_scope(const std::string& name);
    bool remove(DB::Scope_id id);
    Sint32 set_memory_limits(size_t low_water, size_t high_water);
    void get_memory_limits(size_t& low_water, size_t& high_water) const;
    Sint32 set_disk_swapping(const char* path);
    const char* get_disk_swapping() const;
    void lock(DB::Tag tag);
    bool unlock(DB::Tag tag);
    void check_is_locked(DB::Tag tag);
    bool wait_for_notify(DB::Tag tag, TIME::Time timeout);
    void notify(DB::Tag tag);
    std::string get_next_name(const std::string& pattern);
    DB::Database_statistics get_statistics();
    DB::Db_status get_database_status();
    void register_status_listener(DB::IStatus_listener* listener);
    void unregister_status_listener(DB::IStatus_listener* listener);
    void register_transaction_listener(DB::ITransaction_listener* listener);
    void unregister_transaction_listener(DB::ITransaction_listener* listener);
    void register_scope_listener(DB::IScope_listener* listener);
    void unregister_scope_listener(DB::IScope_listener* listener);
    void lowest_open_transaction_id_changed(DB::Transaction_id transaction_id);

    void set_ready_event(EVENT::Event0_base* event);
    DB::Transaction* get_transaction(DB::Transaction_id id);
    void cancel_all_fragmented_jobs();

    Sint32 execute_fragmented(DB::Fragmented_job* job, size_t count);
    Sint32 execute_fragmented_async(
        DB::Fragmented_job* job, size_t count,  DB::IExecution_listener* listener);
    void suspend_current_job();
    void resume_current_job();
    void yield();

    /// Used by the transaction to allocate new tags
    DB::Tag allocate_tag() { return DB::Tag(++m_next_tag); }

    /// Used by the scope to allocate new transaction ids
    DB::Transaction_id allocate_transaction_id()
    { return DB::Transaction_id(++m_next_transaction_id); }

    /// Used by the info/transaction to increment the reference count of the tag.
    /// Needs #m_lock.
    void increment_reference_count(DB::Tag tag);

    /// Used by the info/transaction to decrement the reference counts of the tag.
    /// Needs #m_lock.
    void decrement_reference_count(DB::Tag tag);

    /// Used by the info to increment the reference counts of the referenced elements.
    /// Needs #m_lock.
    void increment_reference_counts(const DB::Tag_set& tag_set);

    /// Used by the info to decrement the reference counts of the referenced elements.
    /// Needs #m_lock.
    void decrement_reference_counts(const DB::Tag_set& tag_set);

    /// Returns the reference count of the tag.
    Uint32 get_tag_reference_count(DB::Tag tag);

    /// Used by the transaction during commit(). The caller must ensure that there is no open
    /// transaction.
    void garbage_collection_internal();

    /// Used by the transaction to access the tag map. Needs #m_lock.
    Tag_map& get_tag_map() { return m_tags; }
    /// Used by the transaction to access the named tag map. Needs #m_lock.
    Named_tag_map& get_named_tag_map() { return m_named_tags; }
    /// Used by the transaction to access the reverse tag map. Needs #m_lock.
    Reverse_named_tag_map& get_reverse_named_tag_map() { return m_reverse_named_tags; }
    /// Used by the transaction to track removal requests. Needs #m_lock.
    Flagged_for_removal_set& get_flagged_for_removal_set() { return m_tags_flagged_for_removal; }


private:
    /// This is used for allocating tags
    mi::base::Atom32 m_next_tag;
    /// This is used for allocating transaction ids
    mi::base::Atom32 m_next_transaction_id;

public:
    /// The lock for the six containers below.
    mi::base::Lock m_lock;

private:
    /// Holds the DB::Info for each tag. Needs #m_lock.
    Tag_map m_tags;
    /// This is used for converting names in the corresponding tags. Needs #m_lock.
    Named_tag_map m_named_tags;
    /// This is used for converting tags into names. Needs #m_lock.
    Reverse_named_tag_map m_reverse_named_tags;
    /// This holds the tags flagged for removal. Needs #m_lock.
    Flagged_for_removal_set m_tags_flagged_for_removal;
    /// Holds the reference count for each tag. Needs #m_lock.
    Reference_count_map m_reference_counts;
    /// Holds the tags with reference count zero. Needs #m_lock.
    Reference_count_zero_set m_reference_count_zero;

    /// The global scope is currently the only scope
    Scope_impl* m_global_scope;

};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_IMPL_H

