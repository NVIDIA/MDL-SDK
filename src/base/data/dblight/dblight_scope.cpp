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

#include "pch.h"

#include "dblight_scope.h"

#include <sstream>
#include <utility>

#include <boost/core/ignore_unused.hpp>

#include "dblight_database.h"
#include "dblight_transaction.h"

#include <base/lib/log/i_log_logger.h>

namespace MI {

namespace DBLIGHT {

Scope_impl::Scope_impl(
    Database_impl* database,
    Scope_manager* scope_manager,
    DB::Scope_id id,
    std::string name,
    Scope_impl* parent,
    DB::Privacy_level level)
  : m_database( database),
    m_scope_manager( scope_manager),
    m_id( id),
    m_name( std::move( name)),
    m_parent( parent),
    m_level( level)
{
    if( m_parent)
        m_parent->pin();
}

Scope_impl::~Scope_impl()
{
    THREAD::Block block( &m_database->get_lock());

    Info_manager* info_manager = m_database->get_info_manager();

    for( Info_impl& info: m_infos) {
        info.clear_scope();
        info_manager->consider_tag_for_gc( info.get_tag());
    }
    m_infos.clear();

    m_scope_manager->remove_scope_internal( this);

    Transaction_manager* transaction_manager = m_database->get_transaction_manager();
    info_manager->garbage_collection( transaction_manager->get_lowest_open_transaction_id());

    block.release();
    if( m_parent)
        m_parent->unpin();
}

DB::Scope* Scope_impl::create_child(
    DB::Privacy_level level, bool /*is_temporary*/, const std::string& name)
{
    return m_scope_manager->create_scope( name, this, level);
}

DB::Transaction* Scope_impl::start_transaction()
{
    return m_database->get_transaction_manager()->start_transaction( this);
}

std::unique_ptr<DB::Journal_query_result> Scope_impl::get_journal(
    DB::Transaction_id last_transaction_id,
    mi::Uint32 last_transaction_change_version,
    DB::Transaction_id current_transaction_id,
    DB::Journal_type journal_type,
    bool lookup_parents)
{
    Statistics_helper helper( g_scope_get_journal);

    if( !m_database->get_journal_enabled()) {
        LOG::mod_log->error(
            M_DB, LOG::Mod_log::C_DATABASE, "Journal query with disabled journal.");
        return {};
    }

    THREAD::Block_shared block( &m_database->get_lock());
    auto result = std::make_unique<DB::Journal_query_result>();

    bool success = get_journal(
        last_transaction_id,
        last_transaction_change_version,
        current_transaction_id,
        journal_type,
        lookup_parents,
        *result.get());
    if( !success)
        return nullptr;

    return result;
}

void Scope_impl::insert_info( Info_impl* info)
{
    m_database->get_lock().check_is_owned();

    MI_ASSERT( info->get_scope_id() == m_id);

    m_infos.push_back( *info);
}

void Scope_impl::erase_info( Info_impl* info)
{
    m_database->get_lock().check_is_owned();

    MI_ASSERT( info->get_scope_id() == m_id);

    auto it = Infos_list::s_iterator_to( *info);
    m_infos.erase( it);
}

size_t Scope_impl::update_journal(
    DB::Transaction_id transaction_id,
    DB::Transaction_id visibility_id,
    const Transaction_journal_entry* journal,
    size_t count)
{
    m_database->get_lock().check_is_owned();
    MI_ASSERT( m_database->get_journal_enabled());

    MI_ASSERT( count > 0);
    MI_ASSERT( journal[0].m_scope_id == m_id);

    // Find length of initial array segment with matching scope IDs.
    size_t i = 0;
    for( ; (i < count) && (journal[i].m_scope_id == m_id); ++i)
        ;
    count = i;

    // Prune entire journal if the newly added entries exceed the maximum size.
    size_t max_size = m_database->get_journal_max_size();
    if( count > max_size) {
        m_journal_last_pruned_visibility = visibility_id;
        m_journal.clear();
        return count;
    }

    // Partially prune journal if required.
    size_t new_size = m_journal.size() + count;
    if( new_size > max_size) {
        size_t prune_count = new_size - max_size;
        auto first = m_journal.begin();
        auto last  = first;
        advance( last, prune_count-1);
        m_journal_last_pruned_visibility = last->first;
        ++last;
        m_journal.erase( first, last);
    }

    // Add all journal entries (from the initial array segment with matching scope IDs).
    for( i = 0; i < count; ++i) {
        const Transaction_journal_entry& entry = journal[i];
        MI_ASSERT( entry.m_journal_type != DB::JOURNAL_NONE);
        Scope_journal_entry new_entry(
            entry.m_tag, entry.m_version, transaction_id, entry.m_journal_type);
        m_journal.emplace( visibility_id, new_entry);
    }

    return count;
}

bool Scope_impl::get_journal(
    DB::Transaction_id last_transaction_id,
    mi::Uint32 last_transaction_change_version,
    DB::Transaction_id current_transaction_id,
    DB::Journal_type journal_type,
    bool include_parent_scopes,
    DB::Journal_query_result& result)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();
    MI_ASSERT( m_database->get_journal_enabled());

    // Fail if the query range includes pruned parts of the journal.
    if( last_transaction_id <= m_journal_last_pruned_visibility)
        return false;

    // Loop over the journal with visibility IDs from \p last_transaction_id+1 to
    // \p current_transaction_id.
    //
    // Note that the visibility ID of changes from \p last_transaction is at least
    // \p last_transaction_id+1.
    auto it     = m_journal.upper_bound( last_transaction_id);
    auto it_end = m_journal.end();
    for( ; (it != it_end) && (it->first <= current_transaction_id); ++it) {

        const Scope_journal_entry& entry = it->second;
        // Skip entries from \p last_transaction_id which happened before
        // \p last_transaction_change_version.
        if(    entry.m_transaction_id == last_transaction_id
            && entry.m_version < last_transaction_change_version)
            continue;
        // Skip entries that do not match the journal type filter.
        if( (entry.m_journal_type.get_type() & journal_type.get_type()) == 0)
            continue;
        result.emplace_back( entry.m_tag, entry.m_journal_type);
    }

    if( !include_parent_scopes)
        return true;
    if( !m_parent)
        return true;

    return m_parent->get_journal(
        last_transaction_id,
        last_transaction_change_version,
        current_transaction_id,
        journal_type,
        include_parent_scopes,
        result);
}

Scope_manager::Scope_manager( Database_impl* database)
  : m_database( database)
{
    // Create global scope.
    create_scope( /*name*/ {}, /*parent*/ nullptr, /*level*/ 0);
}

Scope_manager::~Scope_manager()
{
    // Note that the loop modifies the container contents. Traverse in reverse order such that
    // child scopes are released before the parent scope. Note that the unpin() operation might
    // cause more than one scope to remove itself from the containers if such a scope has been
    // marked for removal but is still pinned by its child scopes.
    while( !m_scopes_by_id.empty()) {
        Scope_impl* last = & *m_scopes_by_id.rbegin();
        // An assertion failure indicates
        // - a leaked DB element/transaction/scope,
        // - a reference cycle between elements, or
        // - a DB element pinned while the last transaction was committed (edit, or access after
        //   edit), which prevented the GC from clearing the creator transaction.
        // Calling dump() from Database_impl::~Database_impl() might provide some insights.
        MI_ASSERT( last->get_pin_count() == 1);
        last->unpin();
    }
}

DB::Scope* Scope_manager::lookup_scope( DB::Scope_id id)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    auto it = m_scopes_by_id.find( id);
    if( it == m_scopes_by_id.end())
        return nullptr;

    return & *it;
}

DB::Scope* Scope_manager::lookup_scope( const std::string& name)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    auto it = m_scopes_by_name.find( name);
    if( it == m_scopes_by_name.end())
        return nullptr;

    return & *it;
}

DB::Scope* Scope_manager::create_scope(
    const std::string& name, DB::Scope* parent, DB::Privacy_level level)
{
    THREAD::Block block( &m_database->get_lock());

    // Check if named scope exists already and return it if parent and level match.
    auto it = m_scopes_by_name.find( name);
    if( it != m_scopes_by_name.end()) {
        if( parent != it->get_parent())
            return nullptr;
        if( level != it->get_level())
            return nullptr;
        return & *it;
    }

    if( parent) {
        // Enforce increasing privacy level.
        if( level <= parent->get_level())
            return nullptr;
    } else {
        // Enforce global scope invariants.
        MI_ASSERT( level == 0);
        MI_ASSERT( m_scopes_by_id.empty());
        MI_ASSERT( m_scopes_by_name.empty());
    }

    auto* parent_impl = static_cast<Scope_impl*>( parent);
    DB::Scope_id id = m_next_scope_id++;
    auto* scope = new Scope_impl( m_database, this, id, name, parent_impl, level);
    m_scopes_by_id.insert( *scope);
    if( !name.empty())
        m_scopes_by_name.insert( *scope);

    if( parent)
        m_database->notify_scope_listeners( &DB::IScope_listener::scope_created, scope);

    return scope;
}

bool Scope_manager::remove_scope( DB::Scope_id id)
{
    THREAD::Block block( &m_database->get_lock());

    if( id == 0)
        return false;

    auto it = m_scopes_by_id.find( id);
    if( it == m_scopes_by_id.end())
        return false;

    // Prevent double removals.
    if( it->get_is_removed())
        return true;

    it->set_is_removed();

    if( it->get_parent())
        m_database->notify_scope_listeners( &DB::IScope_listener::scope_removed, & *it);

    block.release();
    it->unpin();
    return true;
}

void Scope_manager::remove_scope_internal( Scope_impl* scope)
{
    m_database->get_lock().check_is_owned();

    auto it_id = Scopes_by_id_map::s_iterator_to( *scope);
    m_scopes_by_id.erase( it_id);

    if( !scope->get_name().empty()) {
        auto it_name = Scopes_by_name_map::s_iterator_to( *scope);
        m_scopes_by_name.erase( it_name);
    }
}

void Scope_manager::dump( std::ostream& s, bool mask_pointer_values)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "Count of all scopes: " << m_scopes_by_id.size() << std::endl;

    for( const auto& scope: m_scopes_by_id) {

        const std::string& name = scope.get_name();
        std::string name_str = !name.empty() ? ("\"" + name + "\"") : "(null)";
        DB::Scope* parent = scope.get_parent();

        s << "ID " << scope.get_id();
        if( !mask_pointer_values) s << " at " << &scope;
        s << ": name = " << name_str
          << ", pin count = " << scope.get_pin_count()
          << ", level = " << static_cast<mi::Uint32>( scope.get_level());
        if( parent)
            s << ", parent ID = " << parent->get_id();
        else
            s << ", parent ID = (null)";
        s << ", removed = " << scope.get_is_removed() << std::endl;

        if( m_database->get_journal_enabled()) {
            s << "    Journal last pruned visibility: "
              << scope.get_journal_last_pruned_visibility()()
              << std::endl;
            const Scope_impl::Scope_journal& journal = scope.get_journal();
            size_t n = journal.size();
            s << "    Journal size: " << n << std::endl;
            size_t i = 0;
            for( const auto& entry: journal) {
                 s << "    Item " << i++
                   << ": visibility ID = " << entry.first()
                   << ", tag = " << entry.second.m_tag()
                   << ", version = " << entry.second.m_version
                   << ", transaction ID = " << entry.second.m_transaction_id()
                   << ", journal type = " << entry.second.m_journal_type.get_type()
                   << std::endl;
            }
         }
    }

    if( !m_scopes_by_id.empty())
        s << std::endl;
}

} // namespace DBLIGHT

} // namespace MI
