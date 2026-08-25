/***************************************************************************************************
 * Copyright (c) 2012-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
    DB::Privacy_level level,
    DB::Transaction_id next_transaction_id)
  : m_database( database),
    m_scope_manager( scope_manager),
    m_id( id),
    m_name( std::move( name)),
    m_parent( parent),
    m_level( level),
    m_journal_last_pruned_visibility( next_transaction_id-1),
    m_lowest_open_transaction_id( next_transaction_id)
{
    if( m_parent)
        m_parent->pin();
}

Scope_impl::~Scope_impl()
{
    Statistics_helper helper( g_scope_destructor);

    THREAD::Block block( &m_database->get_lock());

    Info_manager* info_manager = m_database->get_info_manager();

    for( Info_impl& info: m_infos) {
        info.clear_scope();
        info_manager->consider_tag_for_gc( info.get_tag());
    }
    m_infos.clear();

    m_scope_manager->remove_scope_internal( this);

    info_manager->garbage_collection(
        /*force*/ false, /*update_lowest_open_transaction_ids*/ false);

    block.release();
    if( m_parent)
        m_parent->unpin();
}

DB::Database* Scope_impl::get_database() const
{
    return m_database;
}

DB::Scope* Scope_impl::create_child(
    DB::Privacy_level level, bool /*is_temporary*/, const std::string& name)
{
    DB::Transaction_id next_transaction_id
        = m_database->get_transaction_manager()->get_next_transaction_id();
    return m_scope_manager->create_scope( name, this, level, next_transaction_id);
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

void Scope_impl::add_open_transaction( Transaction_impl* transaction)
{
    m_database->get_lock().check_is_owned();

    MI_ASSERT( transaction->get_state() == Transaction_impl::OPEN);

    m_open_transactions.insert( *transaction);
}

void Scope_impl::remove_open_transaction( Transaction_impl* transaction)
{
    m_database->get_lock().check_is_owned();

    MI_ASSERT( transaction->get_state() == Transaction_impl::CLOSING);

    auto it = m_open_transactions.find( *transaction);
    MI_ASSERT( it != m_open_transactions.end());
    m_open_transactions.erase( it);
}

void Scope_impl::update_lowest_open_transaction_id( DB::Transaction_id next_id)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    if( m_open_transactions.empty())
        m_lowest_open_transaction_id = next_id;
    else
        m_lowest_open_transaction_id = m_open_transactions.begin()->get_id();
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

    // Loop over the journal with visibilities from \p last_transaction_id+1 to
    // \p current_transaction_id.
    //
    // Note that the visibility of changes from \p last_transaction is at least
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
    DB::Transaction_id next_transaction_id( 0);
    create_scope( /*name*/ {}, /*parent*/ nullptr, /*level*/ 0, next_transaction_id);
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
        // - a reference cycle between elements (rare), or
        // - a DB element pinned while the last transaction was committed (edit, or access after
        //   edit), which prevented the GC from clearing the creator transaction (rare).
        //
        // Calling dump() from Database_impl::~Database_impl() might provide some insights.
        // However, the output is huge and identifying the root cause from that output can be
        // difficult. Usually, it is faster to bisect the changes to a known good state. Typical
        // mistakes are:
        // - Function calls returning a reference-counted interface without capturing it in a
        //   handle (or similar handling).
        // - A special case of the above is chaining such calls incorrectly as in
        //   foo->get_bar()->get_baz().
        //
        // Also, watch out for error messages from the API layer about DB elements still referenced
        // when a transaction is committed/aborted. And from the Python binding for imbalanced
        // reference counts.
        //
        // Is the assertion the consequence of an earlier error, in particular, abnormal shutdown?
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
    const std::string& name,
    DB::Scope* parent,
    DB::Privacy_level level,
    DB::Transaction_id next_transaction_id)
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
    auto* scope
        = new Scope_impl( m_database, this, id, name, parent_impl, level, next_transaction_id);
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

void Scope_manager::update_lowest_open_transaction_ids()
{
    // For simplicity, this implementation is linear in the number of scopes, even though only the
    // scopes on the path from the scope of an ending transaction to the global scope need an
    // update. However, such an implementation requires more book-keeping and does not pay off as
    // the number of scopes is typically rather small.

    m_database->get_lock().check_is_owned();

    DB::Transaction_id next_id = m_database->get_transaction_manager()->get_next_transaction_id();

    // Compute lowest open transaction ID per scope, order does not matter.
    for( auto it = m_scopes_by_id.rbegin(); it != m_scopes_by_id.rend(); ++it)
        it->update_lowest_open_transaction_id( next_id);

    // Propagate lowest open transaction ID upwards in the scope tree, order does matter.
    for( auto current = m_scopes_by_id.rbegin(); true; ++current) {
        Scope_impl* parent = static_cast<Scope_impl*>( current->get_parent());
        if( !parent)
            break;
        DB::Transaction_id current_id = current->get_lowest_open_transaction_id();
        DB::Transaction_id parent_id  = parent->get_lowest_open_transaction_id();
        if( current_id < parent_id)
            parent->set_lowest_open_transaction_id( current_id);
    }
}

void Scope_manager::dump( std::ostream& s, bool verbose, bool mask_pointer_values)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "Count of all scopes: " << m_scopes_by_id.size() << std::endl;

    for( const auto& scope: m_scopes_by_id) {

        const std::string& name = scope.get_name();
        std::string name_str = !name.empty() ? ("\"" + name + "\"") : "(null)";
        DB::Scope* parent = scope.get_parent();

        s << "Index " << scope.get_id();
        if( !mask_pointer_values) s << " at " << &scope;
        s << ": name = " << name_str
          << ", pin count = " << scope.get_pin_count()
          << ", level = " << static_cast<mi::Uint32>( scope.get_level());
        if( parent)
            s << ", parent ID = " << parent->get_id();
        else
            s << ", parent ID = (null)";
        s << ", removed = " << scope.get_is_removed();
        s << ", lowest open transaction ID = " << scope.get_lowest_open_transaction_id().get_uint();
        s << std::endl;

        if( m_database->get_journal_enabled()) {
            s << "    Journal last pruned visibility: "
              << scope.get_journal_last_pruned_visibility()()
              << std::endl;
            const Scope_impl::Scope_journal& journal = scope.get_journal();
            size_t n = journal.size();
            s << "    Journal size: " << n << std::endl;
            if( verbose) {
                size_t i = 0;
                for( const auto& entry: journal) {
                    s << "    Item " << i++
                    << ": visibility = " << entry.first()
                    << ", tag = " << entry.second.m_tag()
                    << ", version = " << entry.second.m_version
                    << ", transaction ID = " << entry.second.m_transaction_id()
                    << ", journal type = " << entry.second.m_journal_type.get_type()
                    << std::endl;
                }
            }
         }
    }

    s << std::endl;
}

void Scope_manager::dump_html( std::ostream& s, const Html_context& context)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>ID</th>\n";
    s << "<th>Name</th>\n";
    s << "<th>Pin count</th>\n";
    s << "<th>Level</th>\n";
    s << "<th>Parent ID</th>\n";
    s << "<th>Removed</th>\n";
    s << "<th>Low. open TX ID</th>\n";
    s << "<th># Infos</th>\n";
    s << "</tr>\n";

    for( const auto& scope: m_scopes_by_id) {

        std::string name = scope.get_name();
        if( name.empty())
            name = "-";
        DB::Scope* parent = scope.get_parent();
        std::string parent_id = parent ? std::to_string( parent->get_id()) : "-";

        s << "<tr>\n";
        s << "<td align=right>"  << scope.get_id() << "</td>\n";
        s << "<td>"              << context.m_html_encoder( name) << "</td>\n";
        s << "<td align=right>"  << scope.get_pin_count() << "</td>\n";
        s << "<td align=right>"  << static_cast<mi::Uint32>( scope.get_level()) << "</td>\n";
        s << "<td align=right>"  << parent_id << "</td>\n";
        s << "<td align=center>" << to_yes_no( scope.get_is_removed()) << "</td>\n";
        s << "<td align=right>"  << scope.get_lowest_open_transaction_id()() << "</td>\n";
        s << "<td align=right>"  << scope.get_infos().size() << "</td>\n";
        s << "</tr>\n";
    }

    s << "</table>\n";
}

} // namespace DBLIGHT

} // namespace MI
