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

#include "dblight_info.h"

#include <iomanip>
#include <numeric>
#include <set>

#include <base/data/db/i_db_element.h>
#include <base/data/sched/i_sched.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/registry/i_config_registry.h>
#include <base/system/main/access_module.h>
#include <base/system/main/i_assert.h>

#include "dblight_database.h"
#include "dblight_scope.h"
#include "dblight_transaction.h"
#include "dblight_util.h"

namespace MI {

namespace DBLIGHT {

bool operator==( const Info_base& lhs, const Info_base& rhs)
{
    if( lhs.m_scope_id != rhs.m_scope_id)
        return false;
    if( lhs.m_transaction_id != rhs.m_transaction_id)
        return false;
    if( lhs.m_version != rhs.m_version)
        return false;
    return true;
}

bool operator!=( const Info_base& lhs, const Info_base& rhs)
{
    return ! (lhs == rhs);
}

bool operator<( const Info_base& lhs, const Info_base& rhs)
{
    if( lhs.m_scope_id < rhs.m_scope_id)
        return true;
    if( lhs.m_scope_id > rhs.m_scope_id)
        return false;
    if( lhs.m_transaction_id < rhs.m_transaction_id)
        return true;
    if( lhs.m_transaction_id > rhs.m_transaction_id)
        return false;
    if( lhs.m_version < rhs.m_version)
        return true;
    return false;
}

bool operator<=( const Info_base& lhs, const Info_base& rhs)
{
    return ! (rhs < lhs);
}

bool operator>( const Info_base& lhs, const Info_base& rhs)
{
    return rhs < lhs;
}

bool operator>=( const Info_base& lhs, const Info_base& rhs)
{
    return rhs <= lhs;
}

Info_impl::Info_impl(
    DB::Element_base* element,
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag,
    DB::Privacy_level privacy_level,
    DB::Tag_set& references)
  : Info_base( scope->get_id(), transaction->get_id(), version),
    m_element( element),
    m_tag( tag),
    m_scope( scope),
    m_references( std::move( references)),
    m_state_and_visibility( transaction->get_state_and_visibility()),
    m_privacy_level( privacy_level)
{
    MI_ASSERT( m_references.find( m_tag) == m_references.end());
    MI_ASSERT( m_references.find( DB::Tag()) == m_references.end());
}

Info_impl::Info_impl(
    SCHED::Job_base* job,
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag,
    DB::Privacy_level privacy_level,
    bool temporary)
  : Info_base( scope->get_id(), transaction->get_id(), version),
    m_job( job),
    m_tag( tag),
    m_scope( scope),
    m_state_and_visibility( transaction->get_state_and_visibility()),
    m_privacy_level( privacy_level),
    m_temporary( temporary)
{
}

Info_impl::Info_impl(
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag)
  : Info_base( scope->get_id(), transaction->get_id(), version),
    m_tag( tag),
    m_scope( scope),
    m_state_and_visibility( transaction->get_state_and_visibility()),
    m_privacy_level( scope->get_level()),
    m_removal( true)
{
}

Info_impl::~Info_impl()
{
    Statistics_helper helper( g_element_job_destructors);

    delete m_element;
    delete m_job;
}

void Info_impl::pin()
{
    MI_ASSERT( m_element || m_job);
    ++m_pin_count;
}

void Info_impl::unpin()
{
    // No assertion for m_element or m_job to allow unpinning a just constructed instance for
    // removal operations with pin count 1. (We could initialize m_pin_count to 0 in that
    // particular constructor, but that would be inconsistent and counter-intuitive for callers.)
    --m_pin_count;
}

const char* Info_impl::get_name() const
{
    return m_infos_per_name ? m_infos_per_name->get_name().c_str() : nullptr;
}

void Info_impl::update_references()
{
    m_references.clear();
    m_element->get_references( &m_references);

    MI_ASSERT( m_references.find( m_tag) == m_references.end());
    MI_ASSERT( m_references.find( DB::Tag()) == m_references.end());
}

void Info_impl::set_infos_per_name( Infos_per_name* infos_per_name)
{
    MI_ASSERT( !!infos_per_name ^ !!m_infos_per_name);

    m_infos_per_name = infos_per_name;
}

void Info_impl::set_element_from_serialization_check( DB::Element_base* element)
{
    MI_ASSERT( m_element);
    MI_ASSERT( element);

    delete m_element;
    m_element = element;
}

void Info_impl::set_element_from_job_execution( DB::Element_base* element, DB::Tag_set& references)
{
    MI_ASSERT( !m_element);
    MI_ASSERT( element);

    m_element = element;
    m_references = std::move( references);
}

namespace {

/// Indicates whether the transaction has been committed.
bool is_committed( const State_and_visibility_ptr& state_and_visibility)
{
    return state_and_visibility && (state_and_visibility->m_state == Transaction_impl::COMMITTED);
}

/// Indicates whether the transaction has been aborted.
bool is_aborted( const State_and_visibility_ptr& state_and_visibility)
{
    return state_and_visibility && (state_and_visibility->m_state == Transaction_impl::ABORTED);
}

/// Indicates whether changes from a particular transaction are visible for transaction \p id.
///
/// \param creator_id             Transaction ID of the creator transaction.
/// \param state_and_visibility   State and visibility of the creator transaction.
/// \param id                     Are the changes visible in this transaction?
///
/// \note This method considers only the creation/commit sequence and states. It completely
///       ignores the corresponding scopes.
bool is_visible_for(
    DB::Transaction_id creator_id,
    const State_and_visibility_ptr& state_and_visibility,
    DB::Transaction_id id)
{
    MI_ASSERT( state_and_visibility);

    if( id == creator_id)
        return true;

    return (state_and_visibility->m_state == Transaction_impl::COMMITTED)
        && (state_and_visibility->m_id <= id);
}

/// Indicates whether two transactions definitely have the same visibility.
///
/// Assumes that globally visible transactions have already been cleared.
///
/// Note that the computation based on the visibility is an approximation and the method errs
/// on the safe side for GC purposes. It returns \c true if both transaction definitely have the
/// same visibility. It might return \c false if the visibility is indeed identical, but this
/// cannot be determined due to the approximation scheme.
bool same_visibility(
    const State_and_visibility_ptr& lhs, const State_and_visibility_ptr& rhs)
{
    // Transactions which are globally visible have same visibility.
    if( !lhs && !rhs)
        return true;

    // Transactions with different global visibility have different visibility.
    if( !!lhs ^ !!rhs)
        return false;

    // Check for identical/equal visibility information.
    MI_ASSERT( !!lhs && !!rhs);
    return *lhs == *rhs;
}

/// Indicates whether the info is a removal info.
///
/// Template to support iterators from Infos_per_tag_set and Infos_per_name_set.
template <class T>
bool is_removal( T info)
{
    return info->get_is_removal();
}

/// Indicates whether the info is a global removal info.
///
/// Template to support iterators from Infos_per_tag_set and Infos_per_name_set.
template <class T>
bool is_global_removal( T info)
{
    return info->get_is_removal() && info->get_scope_id() == 0;
}

/// Indicates whether the info is a local removal info.
///
/// Template to support iterators from Infos_per_tag_set and Infos_per_name_set.
template <class T>
bool is_local_removal( T info)
{
    return info->get_is_removal() && info->get_scope_id() != 0;
}

} // namespace

/// Non-trivial code shared between Infos_per_name and Infos_per_tag.
namespace IMPL {

/// Comparison functor for (Info_impl,Info_base) pairs.
class Info_comp
{
public:
    bool operator()( const Info_impl& lhs, const Info_base& rhs) const { return lhs < rhs; }
    bool operator()( const Info_base& lhs, const Info_impl& rhs) const { return rhs > lhs; }
};

/// Returns an iterator to the last info that is less than or equal to the given key.
///
/// Returns infos.end() if there is no such iterator.
template <class T>
typename T::iterator find_less_or_equal(
    T& infos, DB::Scope_id scope_id, DB::Transaction_id transaction_id, mi::Uint32 version)
{
    // Find first info that is larger than the requested one.
    Info_base pattern( scope_id, transaction_id, version);
    auto it = infos.upper_bound( pattern, Info_comp());
    MI_ASSERT( (it == infos.end()) || (*it > pattern));

    // Decrement iterator to get to the last info that is lesser than or equal to the requested one.
    if( it == infos.begin())
        return infos.end();
    --it;
    MI_ASSERT( *it <= pattern);
    return it;
}

/// Looks up an info.
///
/// \param infos              The intrusive set to use for the look up.
/// \param scope              The scope where to start the look up.
/// \param transaction_id     The transaction ID looking up the info.
/// \param[out] level_found   The privacy level of the scope that contains the returned info,
///                           or unspecified in case of failure.
/// \return                   The looked up info, or \c nullptr in case of failure.
template <class T>
Info_impl* lookup_info(
    T& infos, DB::Scope* scope, DB::Transaction_id transaction_id, DB::Privacy_level* level_found)
{
    MI_ASSERT( scope);

    while( scope) {

        DB::Scope_id scope_id = scope->get_id();

        auto it = find_less_or_equal( infos, scope_id, transaction_id, ~0U);
        if( it == infos.end())
            return nullptr;

        while( it->get_scope_id() == scope_id) {

            // Skip global removal infos which can not be looked up.
            if( is_global_removal( it)) {
                if( it == infos.begin())
                    break;
                --it;
                continue;
            }

            // Check whether the info is visible for all open (and future) transactions.
            State_and_visibility_ptr state_and_visibility = it->get_state_and_visibility();
            if( !state_and_visibility) {
                // Visible local removal infos hide any previous infos in that scope.
                if( is_local_removal( it))
                     break;
                it->pin();
                if( level_found)
                    *level_found = scope->get_level();
                return & *it;
            }

            // Check whether the info is from an aborted transaction. With a synchronous garbage
            // collection these infos can only survive the GC collection if the GC period is
            // larger than 1 or the user incorrectly pins them while aborting the transaction.
            if( state_and_visibility->m_state == Transaction_impl::ABORTED) {
                if( it == infos.begin())
                    break;
                --it;
                continue;
            }

            // Check whether the info is visible for the given transaction ID.
            DB::Transaction_id creator_id = it->get_transaction_id();
            if( is_visible_for( creator_id, state_and_visibility, transaction_id)) {
                // Visible local removal infos hide any previous infos in that scope.
                if( is_local_removal( it))
                     break;
                it->pin();
                if( level_found)
                    *level_found = scope->get_level();
                return & *it;
            }

            if( it == infos.begin())
                break;
            --it;
        }

        scope = scope->get_parent();
    }

    return nullptr;
}

/// Looks up a global removal info.
///
/// \param infos              The intrusive set to use for the look up.
/// \param transaction_id     The transaction ID looking up the info.
/// \return                   Returns \c true if a global removal info was found,
///                           \c false otherwise.
template <class T>
bool lookup_global_removal_info( T& infos, DB::Transaction_id transaction_id)
{
    // Global removal infos are always in the global scope.
    DB::Scope_id scope_id = 0;

    auto it = find_less_or_equal( infos, scope_id, transaction_id, ~0U);
    if( it == infos.end())
        return false;

    while( it->get_scope_id() == scope_id) {

        // Check whether the info is visible for all open (and future) transactions.
        State_and_visibility_ptr state_and_visibility = it->get_state_and_visibility();
        if( !state_and_visibility) {
            if( is_global_removal( it))
                return true;
            if( it == infos.begin())
                break;
            --it;
            continue;
        }

        // Check whether the info is from an aborted transaction. With a synchronous garbage
        // collection these infos can only survive the GC collection if the GC period is
        // larger than 1 or the user incorrectly pins them while aborting the transaction.
        if( state_and_visibility->m_state == Transaction_impl::ABORTED) {
            if( it == infos.begin())
                break;
            --it;
            continue;
        }

        // Check whether the info is visible for the given transaction ID.
        DB::Transaction_id creator_id = it->get_transaction_id();
        if( is_visible_for( creator_id, state_and_visibility, transaction_id))
            if( is_global_removal( it))
                return true;

        if( it == infos.begin())
            break;
        --it;
    }

    return false;
}

} // namespace IMPL

void Infos_per_name::insert_info( Info_impl* info)
{
    [[maybe_unused]] auto result = m_infos.insert( *info);
    MI_ASSERT( result.second);

    info->set_infos_per_name( this);
}

Infos_per_name::Infos_per_name_set::iterator Infos_per_name::erase_info( Info_impl* info)
{
    info->set_infos_per_name( nullptr);
    auto it = Infos_per_name_set::s_iterator_to( *info);
    return m_infos.erase( it);
}

Info_impl* Infos_per_name::lookup_info(
    DB::Scope* scope, DB::Transaction_id transaction_id, DB::Privacy_level* level_found)
{
    return IMPL::lookup_info( m_infos, scope, transaction_id, level_found);
}

void Infos_per_tag::insert_info( Info_impl* info)
{
    MI_ASSERT( info->get_tag() == m_tag);

    [[maybe_unused]] auto result = m_infos.insert( *info);
    MI_ASSERT( result.second);
}

Infos_per_tag::Infos_per_tag_set::iterator Infos_per_tag::erase_info( Info_impl* info)
{
    MI_ASSERT( info->get_tag() == m_tag);

    auto it = Infos_per_tag_set::s_iterator_to( *info);
    return m_infos.erase( it);
}

Info_impl* Infos_per_tag::lookup_info(
    DB::Scope* scope, DB::Transaction_id transaction_id, DB::Privacy_level* level_found)
{
    return IMPL::lookup_info( m_infos, scope, transaction_id, level_found);
}

bool Infos_per_tag::lookup_global_removal_info( DB::Transaction_id transaction_id)
{
    return IMPL::lookup_global_removal_info( m_infos, transaction_id);
}

void Infos_per_tag::set_removed()
{
    MI_ASSERT( !m_is_removed);
    m_is_removed = true;
}

Minor_page::Minor_page()
{
    for( auto& infos_per_tag : m_infos_per_tags)
        infos_per_tag = nullptr;
}

Infos_per_tag* Minor_page::find( size_t index) const
{
    MI_ASSERT( index < L);
    return m_infos_per_tags[index];
}

void Minor_page::insert( size_t index, Infos_per_tag* element)
{
    MI_ASSERT( index < L);
    auto& ptr = m_infos_per_tags[index];
    MI_ASSERT( !ptr);
    ptr = element;
    ++m_local_size;
}

void Minor_page::erase( size_t index)
{
    MI_ASSERT( index < L);
    auto& ptr = m_infos_per_tags[index];
    MI_ASSERT( ptr);
    ptr = nullptr;
    --m_local_size;
}

void Minor_page::apply( std::function<void( Infos_per_tag*)> f) const
{
    for( auto infos_per_tag : m_infos_per_tags)
        if( infos_per_tag)
            f( infos_per_tag);
}

void Minor_page::get_tags( std::vector<DB::Tag>& tags) const
{
    for( auto infos_per_tag : m_infos_per_tags)
        if( infos_per_tag)
            tags.push_back( infos_per_tag->get_tag());
}

Major_page::Major_page()
{
    for( auto& minor_page : m_minor_pages)
        minor_page = nullptr;
}

Major_page::~Major_page()
{
    for( auto& minor_page : m_minor_pages)
        delete minor_page;
}

Infos_per_tag* Major_page::find( size_t index) const
{
    MI_ASSERT( index < L);
    auto ptr = m_minor_pages[index >> S];
    if( !ptr)
        return nullptr;

    return ptr->find( index & M);
}

void Major_page::insert( size_t index, Infos_per_tag* element)
{
    MI_ASSERT( index < L);
    auto& ptr = m_minor_pages[index >> S];
    if( !ptr) {
        ptr = new Minor_page;
        ++m_local_size;
    }

    ptr->insert( index & M, element);
}

void Major_page::erase( size_t index)
{
    MI_ASSERT( index < L);
    auto& ptr = m_minor_pages[index >> S];
    MI_ASSERT( ptr);
    ptr->erase( index & M);
    if( ptr->get_local_size() == 0) {
        delete ptr;
        ptr = nullptr;
        --m_local_size;
    }
}

void Major_page::apply( std::function<void( Infos_per_tag*)> f) const
{
    for( auto ptr : m_minor_pages) {
        if( ptr)
            ptr->apply( f);
    }
}

void Major_page::get_tags( std::vector<DB::Tag>& tags) const
{
   for( auto ptr : m_minor_pages) {
        if( ptr)
            ptr->get_tags( tags);
    }
}

Tag_tree::Tag_tree()
{
    for( auto& major_page : m_major_pages)
        major_page = nullptr;
}

Tag_tree::~Tag_tree()
{
    for( auto & major_page : m_major_pages)
        delete major_page;
}

Infos_per_tag* Tag_tree::find( DB::Tag tag) const
{
    size_t index = tag();

    MI_ASSERT( index < L);
    auto ptr = m_major_pages[index >> S];
    if( !ptr)
        return nullptr;

    return ptr->find( index & M);
}

void Tag_tree::insert( DB::Tag tag, Infos_per_tag* element)
{
    size_t index = tag();

    MI_ASSERT( index < L);
    auto& ptr = m_major_pages[index >> S];
    if( !ptr) {
        ptr = new Major_page;
        ++m_local_size;
    }

    ptr->insert( index & M, element);
    ++m_total_size;
}

void Tag_tree::erase( DB::Tag tag)
{
    size_t index = tag();

    MI_ASSERT( index < L);
    auto& ptr = m_major_pages[index >> S];
    MI_ASSERT( ptr);
    ptr->erase( index & M);
    if( ptr->get_local_size() == 0) {
        delete ptr;
        ptr = nullptr;
        --m_local_size;
    }

    --m_total_size;
}

void Tag_tree::apply( std::function<void( Infos_per_tag*)> f) const
{
   for( auto ptr : m_major_pages) {
        if( ptr)
            ptr->apply( f);
    }
}

void Tag_tree::get_tags( std::vector<DB::Tag>& tags) const
{
   for( auto ptr : m_major_pages) {
        if( ptr)
            ptr->get_tags( tags);
    }
}

Info_manager::Info_manager( Database_impl* database)
  : m_database( database)
{
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();

    std::string gc_method;
    if( registry.get_value( "dblight_gc_method", gc_method)) {
        if( gc_method == "full_sweeps_only")
            m_gc_method = GC_FULL_SWEEPS_ONLY;
        else if( gc_method == "full_sweep_then_pin_count_zero")
            m_gc_method = GC_FULL_SWEEP_THEN_PIN_COUNT_ZERO;
        else if( gc_method == "general_candidates_then_pin_count_zero")
            m_gc_method = GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO;
        else
            LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
                R"(Invalid value "%s" for debug option "dblight_gc_method".)", gc_method.c_str());
    }
    if( m_gc_method != GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "GC method set to %s.", gc_method.c_str());

    CONFIG::update_value( registry, "dblight_gc_period", m_gc_period);
    if( m_gc_period != 1)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "GC period set to %zu.", m_gc_period);

    CONFIG::update_value( registry, "dblight_gc_interval", m_gc_interval);
    if( m_gc_interval != 1.0)
        LOG::mod_log->info( M_DB, LOG::Mod_log::C_DATABASE,
            "GC interval set to %lf.", m_gc_interval);
}

Info_manager::~Info_manager()
{
    THREAD::Block block( &m_database->get_lock());

    garbage_collection( /*force*/ true, /*update_lowest_open_transaction_ids*/ false);

    // Check that there are no GC candidates left, otherwise the GC might have missed something.
    MI_ASSERT( m_gc_candidates_general.empty());
    MI_ASSERT( m_gc_candidates_pin_count_zero.empty());

    // Removal of all scopes should not leave any infos behind.
    MI_ASSERT( m_infos_by_tag.empty());
    MI_ASSERT( m_infos_by_name.empty());
}

void Info_manager::store(
    DB::Element_base* element,
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag,
    DB::Privacy_level privacy_level,
    const char* name,
    DB::Tag_set& references)
{
    m_database->get_lock().check_is_owned();

    // Retrieve (or create) set of infos for \p tag.
    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
    if( !infos_per_tag) {
        infos_per_tag = new Infos_per_tag( tag);
        m_infos_by_tag.insert( tag, infos_per_tag);
    }

    // Retrieve (or create) set of infos for \p name (if not \c nullptr).
    Infos_per_name* infos_per_name = nullptr;
    if( name) {
        auto it_by_name = m_infos_by_name.find( name);
        if( it_by_name == m_infos_by_name.end()) {
            infos_per_name = new Infos_per_name( name);
            m_infos_by_name[name] = infos_per_name;
        } else {
            infos_per_name = it_by_name->second;
        }
    }

    // Record DB element references of this info.
    increment_pin_counts( references);

    // Create info (destroys references).
    auto info = make_ptr_no_add_ref( new Info_impl(
        element, scope, transaction, version, tag, privacy_level, references));

    // Insert info into the sets of infos for that tag/name/scope.
    infos_per_tag->insert_info( info.get());
    if( infos_per_name)
        infos_per_name->insert_info( info.get());
    scope->insert_info( info.get());

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);
}

void Info_manager::store(
    SCHED::Job_base* job,
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag,
    DB::Privacy_level privacy_level,
    const char* name,
    bool temporary)
{
    m_database->get_lock().check_is_owned();

    // Retrieve (or create) set of infos for \p tag.
    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
    if( !infos_per_tag) {
        infos_per_tag = new Infos_per_tag( tag);
        m_infos_by_tag.insert( tag, infos_per_tag);
    }

    // Retrieve (or create) set of infos for \p name (if not \c nullptr).
    Infos_per_name* infos_per_name = nullptr;
    if( name) {
        auto it_by_name = m_infos_by_name.find( name);
        if( it_by_name == m_infos_by_name.end()) {
            infos_per_name = new Infos_per_name( name);
            m_infos_by_name[name] = infos_per_name;
        } else {
            infos_per_name = it_by_name->second;
        }
    }

    // Create info.
    auto info = make_ptr_no_add_ref(
        new Info_impl( job, scope, transaction, version, tag, privacy_level, temporary));

    // Insert info into the sets of infos for that tag/name/scope.
    infos_per_tag->insert_info( info.get());
    if( infos_per_name)
        infos_per_name->insert_info( info.get());
    scope->insert_info( info.get());

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);
}

void Info_manager::store(
    DB::Element_base* element,
    Info_impl* info,
    DB::Tag_set& references)
{
    // Record DB element references of this info.
    increment_pin_counts( references);

    info->set_element_from_job_execution( element, references);
}

Info_impl* Info_manager::lookup_info(
    DB::Tag tag,
    DB::Scope* scope,
    DB::Transaction_id transaction_id,
    DB::Privacy_level* level_found)
{
    Statistics_helper helper( g_lookup_info_by_tag);

    m_database->get_lock().check_is_owned_shared_or_exclusive();

    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
    if( !infos_per_tag)
        return nullptr;

    return infos_per_tag->lookup_info( scope, transaction_id, level_found);
}

Info_impl* Info_manager::lookup_info(
    const char* name,
    DB::Scope* scope,
    DB::Transaction_id transaction_id,
    DB::Privacy_level* level_found)
{
    Statistics_helper helper( g_lookup_info_by_name);

    m_database->get_lock().check_is_owned_shared_or_exclusive();

    if( !name)
        return nullptr;

    auto it = m_infos_by_name.find( name);
    if( it == m_infos_by_name.end())
        return nullptr;

    return it->second->lookup_info( scope, transaction_id, level_found);
}

Info_impl* Info_manager::start_edit(
    DB::Element_base* element,
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag,
    DB::Privacy_level privacy_level,
    Infos_per_name* infos_per_name)
{
    m_database->get_lock().check_is_owned();

    // Retrieve set of infos for \p tag.
    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);

    // Create info (modifies empty_references).
    DB::Tag_set empty_references;
    auto* info = new Info_impl(
        element, scope, transaction, version, tag, privacy_level, empty_references);

    // Insert info into the sets of infos for that tag/name/scope.
    infos_per_tag->insert_info( info);
    if( infos_per_name)
        infos_per_name->insert_info( info);
    scope->insert_info( info);

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);

    return info;
}

void Info_manager::finish_edit( Info_impl* info, Transaction_impl* transaction)
{
    m_database->get_lock().check_is_owned();

    info->update_references();

    const DB::Tag_set& new_references = info->get_references();

    // Check privacy levels.
    if( m_database->get_check_privacy_levels()) {
        DB::Privacy_level referencing_level = info->get_privacy_level();
        DB::Tag tag = info->get_tag();
        const char* name = info->get_name();
        transaction->check_privacy_levels(
            referencing_level, new_references, tag, name, /*store*/ false);
    }

    // Check reference cycles.
    if( m_database->get_check_reference_cycles_edit()) {
        DB::Tag tag = info->get_tag();
        const char* name = info->get_name();
        transaction->check_reference_cycles( new_references, tag, name, /*store*/ false);
    }

    increment_pin_counts( new_references);

    info->unpin();
}

bool Info_manager::remove(
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag,
    bool remove_local_copy)
{
    m_database->get_lock().check_is_owned();

    // Ignore removal_local_copy flag for the global scope.
    bool is_global_removal = !remove_local_copy || scope->get_id() == 0;

    // Retrieve set of infos for \p tag.
    Infos_per_tag* ipt = m_infos_by_tag.find( tag);
    if( !ipt)
        return false;

    Infos_per_name* ipn = nullptr;
    if( is_global_removal) {

        // Make sure that global removals are recorded as such.
        if( scope->get_id() != 0)
            scope = static_cast<Scope_impl*>( m_database->get_scope_manager()->lookup_scope( 0));

    } else {

        auto info = make_ptr_no_add_ref<Info_impl>(
            ipt->lookup_info( scope, transaction->get_id()));
        if( !info)
            return false;

        // Reject local removals without a version in the current scope.
        if( info->get_scope_id() != scope->get_id())
            return false;

        ipn = info->get_infos_per_name();
        info.reset();

        // Reject local removals without another version in a more global scope (otherwise we can
        // end up with invalid tag references).
        auto* parent_scope = static_cast<Scope_impl*>( scope->get_parent());
        auto parent_info = make_ptr_no_add_ref<Info_impl>(
            ipt->lookup_info( parent_scope, transaction->get_id()));
        if( !parent_info)
            return false;
    }

    // Create removal info.
    auto info = make_ptr_no_add_ref( new Info_impl( scope, transaction, version, tag));

    // Insert info into the sets of infos for that tag/scope (name only for local removals and if
    // present).
    ipt->insert_info( info.get());
    scope->insert_info( info.get());
    if( ipn)
        ipn->insert_info( info.get());

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);

    return true;
}

void Info_manager::consider_tag_for_gc( DB::Tag tag)
{
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);
}

void Info_manager::garbage_collection( bool force, bool update_lowest_open_transaction_ids)
{
    Statistics_helper helper( g_garbage_collection);

    m_database->get_lock().check_is_owned();

    if( !do_run_garbage_collection( force))
        return;

    if( update_lowest_open_transaction_ids)
        m_database->get_scope_manager()->update_lowest_open_transaction_ids();

    if( m_gc_method == GC_FULL_SWEEPS_ONLY) {

        while( true) {

            std::vector<DB::Tag> tags;
            tags.reserve( m_infos_by_tag.size());
            m_infos_by_tag.get_tags( tags);

            bool progress = false;
            for( const auto& tag: tags) {
                bool progress_tag = false;
                cleanup_tag_general( tag, progress_tag);
                progress |= progress_tag;
            }

            if( !progress)
                break;
        }

    } else if( m_gc_method == GC_FULL_SWEEP_THEN_PIN_COUNT_ZERO) {

        std::vector<DB::Tag> tags1;
        tags1.reserve( m_infos_by_tag.size());
        m_infos_by_tag.get_tags( tags1);

        for( const auto& tag: tags1) {
            bool progress_tag = false;
            cleanup_tag_general( tag, progress_tag);
        }

        bool progress = true;

        while( true) {

            if( !progress || m_gc_candidates_pin_count_zero.empty())
                break;

            DB::Tag_set tags2 = std::move( m_gc_candidates_pin_count_zero);
            m_gc_candidates_pin_count_zero.clear();
            progress = false;

            for( const auto& tag: tags2) {
                Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
                MI_ASSERT( infos_per_tag);
                bool progress_tag = false;
                cleanup_tag_with_pin_count_zero( infos_per_tag, progress_tag);
                progress |= progress_tag;
            }
        }

    } else if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO) {

        size_t old_size = m_gc_candidates_general.size();
        if( old_size > m_gc_candidates_general_max_size)
            m_gc_candidates_general_max_size = old_size;

        std::vector<DB::Tag> tags1;
        tags1.reserve( old_size);
        tags1.insert(
            tags1.begin(), m_gc_candidates_general.begin(), m_gc_candidates_general.end());

        for( const auto& tag: tags1) {
            bool progress_tag = false;
            cleanup_tag_general( tag, progress_tag);
        }

        bool progress = true;

        while( true) {

            if( !progress || m_gc_candidates_pin_count_zero.empty())
                break;

            DB::Tag_set tags2 = std::move( m_gc_candidates_pin_count_zero);
            m_gc_candidates_pin_count_zero.clear();
            progress = false;

            for( const auto& tag: tags2) {
                Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
                MI_ASSERT( infos_per_tag);
                bool progress_tag = false;
                cleanup_tag_with_pin_count_zero( infos_per_tag, progress_tag);
                progress |= progress_tag;
            }
        }

        // Rehash the container if it got way smaller to improve traversing it for tags1 above in
        // the next GC run.
        size_t new_size = m_gc_candidates_general.size();
        if( new_size <= m_gc_candidates_general_max_size/2) {
            m_gc_candidates_general.rehash( new_size);
            m_gc_candidates_general_max_size = new_size;
        }

    } else {
        MI_ASSERT( !"Unexpected GC method");
    }
}

mi::Uint32 Info_manager::get_tag_reference_count( DB::Tag tag)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    // Retrieve set of infos for \p tag.
    Infos_per_tag* ipt = m_infos_by_tag.find( tag);
    if( !ipt)
        return 0;

    return ipt->get_pin_count();
}

bool Info_manager::get_tag_is_removed(
    DB::Tag tag, DB::Scope* scope, DB::Transaction_id transaction_id)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

   // Retrieve set of infos for \p tag.
    Infos_per_tag* ipt = m_infos_by_tag.find( tag);
    if( !ipt)
        return false;

    if( ipt->get_is_removed())
        return true;

    if( ipt->lookup_global_removal_info( transaction_id))
        return true;

    return false;
}

namespace {

std::ostream& operator<<( std::ostream& s, const DB::Tag_set& tag_set)
{
    std::set<DB::Tag> ordered_tag_set;
    for( const auto& item: tag_set)
        ordered_tag_set.insert( item);

    bool first = true;
    s << '{';

    for( const auto& tag: ordered_tag_set) {
        if( !first)
            s << ',';
        s << ' ' << tag();
        first = false;
    }

    s << " }";
    return s;
}

void dump( std::ostream& s, bool mask_pointer_values, const Infos_per_name* ipn, size_t j1)
{
    const auto& ipn_set = ipn->get_infos();

    s << "Index " << j1
      << ": name = \"" << ipn->get_name() << "\""
      << ", count = " << ipn_set.get_size()
      << std::endl;

    size_t j2 = 0;
    for( const auto& i: ipn_set) {

        s << "    Index " << j1 << "/" << j2++;
        if( !mask_pointer_values)
            s << " at " << &i;
        s << ": ";
        s << "scope ID = " << i.get_scope_id() << ", ";
        // Omit the scope pointer. With a correct synchronous GC the cleared state should never be
        // visible from the outside.
        s << "creator transaction ID = " << i.get_transaction_id()();
        s << ", version = " << i.get_version();
        const auto& state_and_visibility = i.get_state_and_visibility();
        s << ", creator transaction state = ";
        if( !state_and_visibility)
            s << "COMMITTED";
        else
            s << state_and_visibility->m_state;
        s << ", visibility ID = ";
        if( !state_and_visibility)
            s << "(globally visible)";
        else if (state_and_visibility->m_state == Transaction_impl::COMMITTED)
            s << state_and_visibility->m_id.get_uint();
        else
            s << "(current transaction only)";
        s << ", pin count = " << i.get_pin_count();
        s << ", tag = " << i.get_tag()();
        s << ", privacy level = " << static_cast<mi::Uint32>( i.get_privacy_level());
        s << ", removal = " << i.get_is_removal();

        if( i.get_is_job()) {
            // Extra information only for jobs to keep common case short.
            s << ", job = ";
            SCHED::Job_base* job = i.get_job();
            if( mask_pointer_values)
                s << "(set)";
            else
                s << job;
            s << ", shared = " << job->get_is_shared();
            s << ", parent = " << job->get_is_parent();
            s << ", temporary = " << i.get_is_temporary();
            s << ", element = ";
            DB::Element_base* element = i.get_element();
            if( !element)
                s << "(cleared)";
            else if( mask_pointer_values)
                s << "(set)";
            else
                s << element;
        } else if( !mask_pointer_values) {
            // Element pointer only if not masked.
            s << ", element = ";
            DB::Element_base* element = i.get_element();
            s << element;
        }

        s << ", references = " << i.get_references();
        s << std::endl;
    }
}

void dump( std::ostream& s, bool mask_pointer_values, const Infos_per_tag* ipt, size_t j1)
{
    const auto& ipt_set = ipt->get_infos();

    s << "Index " << j1++
      << ": tag = " << ipt->get_tag()()
      << ", count = " << ipt_set.get_size()
      << ", pin count = " << ipt->get_pin_count()
      << ", removed = " << ipt->get_is_removed()
      << std::endl;

    size_t j2 = 0;
    for( const auto& i: ipt->get_infos()) {

        const char* name = i.get_name();
        std::string name_str = name ? (std::string( "\"") + name + "\"") : "(null)";

        s << "    Index " << j1 << "/" << j2++;
        if( !mask_pointer_values)
            s << " at " << &i;
        s << ": ";
        s << "scope ID = " << i.get_scope_id() << ", ";
        // Omit the scope pointer. With a correct synchronous GC the cleared state should never be
        // visible from the outside.
        s << "creator transaction ID = " << i.get_transaction_id()();
        s << ", version = " << i.get_version();
        const auto& state_and_visibility = i.get_state_and_visibility();
        s << ", creator transaction state = ";
        if( !state_and_visibility)
            s << "COMMITTED";
        else
            s << state_and_visibility->m_state;
        s << ", visibility ID = ";
        if( !state_and_visibility)
            s << "(globally visible)";
        else if (state_and_visibility->m_state == Transaction_impl::COMMITTED)
            s << state_and_visibility->m_id.get_uint();
        else
            s << "(current transaction only)";
        s << ", pin count = " << i.get_pin_count();
        s << ", tag = " << i.get_tag()();
        s << ", privacy level = " << static_cast<mi::Uint32>( i.get_privacy_level());
        s << ", removal = " << i.get_is_removal();

        if( i.get_is_job()) {
            // Extra information only for jobs to keep common case short.
            s << ", job = ";
            SCHED::Job_base* job = i.get_job();
            if( mask_pointer_values)
                s << "(set)";
            else
                s << job;
            s << ", shared = " << job->get_is_shared();
            s << ", parent = " << job->get_is_parent();
            s << ", temporary = " << i.get_is_temporary();
            s << ", element = ";
            DB::Element_base* element = i.get_element();
            if( !element)
                s << "(cleared)";
            else if( mask_pointer_values)
                s << "(set)";
            else
                s << element;
        } else if( !mask_pointer_values) {
            // Element pointer only if not masked.
            s << ", element = ";
            DB::Element_base* element = i.get_element();
            if( !element)
                s << "(cleared)";
            else
                s << element;
        }

        s << ", name = " << name_str;
        s << ", references = " << i.get_references();
        s << std::endl;
    }
}

std::string decode_class_id( SERIAL::Class_id class_id)
{
    if( class_id == SERIAL::class_id_unknown)
        return "(unknown)";

    std::string result = "\"????\"";
    for( int i = 0; i < 4; ++i) {
        result[4-i] = static_cast<char>( class_id % 256);
        class_id /= 256;
    }
    return result;
}

} // namespace

void Info_manager::dump( std::ostream& s, bool verbose, bool mask_pointer_values)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "Count of infos by distinct names: " << m_infos_by_name.size() << std::endl;

    if( verbose) {
        size_t j1 = 0;
        // Dump by order of names, not by order of hashes.
        std::set<std::string> names;
        for( const auto& ipn: m_infos_by_name)
            names.insert( ipn.first);
        for( const auto& name: names)
            DBLIGHT::dump( s, mask_pointer_values, m_infos_by_name[name], j1++);
        s << std::endl;
    }

    s << "Count of infos by distinct tags: " << m_infos_by_tag.size() << std::endl;

    if( verbose) {
        size_t j1 = 0;
        auto dump_as_lambda = [&s, mask_pointer_values, &j1]( Infos_per_tag* ipt)
        { DBLIGHT::dump( s, mask_pointer_values, ipt, j1++); };
        m_infos_by_tag.apply( dump_as_lambda);
        s << std::endl;
    }

    Statistics stats;
    size_t sum_count = 0;
    size_t sum_sizes = 0;
    get_statistics( stats, sum_count, sum_sizes);

    s << "Count of all infos: " << sum_count << std::endl;
    s << "Count of infos by class IDs:" << std::endl;

    std::ios old_state( nullptr);
    old_state.copyfmt( s);

    for( const auto& entry: stats) {
        s << std::setbase( 10) << std::noshowbase << std::setfill( ' ') << std::setw( 5);
        s << entry.second.m_count << "   ";
        s << std::setbase( 16) << std::showbase << std::setfill( '0') << std::setw( 10);
        s << entry.first << " ";
        s << decode_class_id( entry.first) << std::endl;
    }
    s.copyfmt( old_state);

    s << std::endl;
}

bool Info_manager::do_run_garbage_collection( bool force)
{
    if( force) {
        m_gc_counter = 0;
        if( m_gc_period > 1)
            m_gc_last_timestamp = std::chrono::system_clock::now();
        return true;
    }

    if( ++m_gc_counter >= m_gc_period) {
        m_gc_counter = 0;
        if( m_gc_period > 1)
            m_gc_last_timestamp = std::chrono::system_clock::now();
        return true;
    }

    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    double interval = std::chrono::duration<double>( now - m_gc_last_timestamp).count();
    if( interval >  m_gc_interval) {
        m_gc_counter = 0;
        m_gc_last_timestamp = now;
        return true;
    }

    return false;
}

void Info_manager::dump_html_tags( std::ostream& s, const Html_context& context)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "<div>Number of different tags: " << m_infos_by_tag.size() << "</div>\n";
    s << "<p></p>\n";

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Tag</th>\n";
    s << "<th># Infos</th>\n";
    s << "<th>Ref. count</th>\n";
    s << "<th>Removed</th>\n";
    s << "</tr>\n";

    auto dump_as_lambda = [&s, &context]( Infos_per_tag* ipt)
    {
        s << "<tr>\n";
        s << "<td align=right><a href=\"" << context.m_tag_url_prefix << ipt->get_tag()() << "\">";
        s << ipt->get_tag()();
        s << "</a></td>\n";
        s << "<td align=right>"  << ipt->get_infos().get_size() << "</td>\n";
        s << "<td align=right>"  << ipt->get_pin_count() << "</td>\n";
        s << "<td align=center>" << to_yes_no( ipt->get_is_removed()) << "</td>\n";
        s << "</tr>\n";
    };

    m_infos_by_tag.apply( dump_as_lambda);

    s << "</table>\n";
}

void Info_manager::dump_html_names( std::ostream& s, const Html_context& context)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "<div>Number of different names: " << m_infos_by_name.size() << "</div>\n";
    s << "<p></p>\n";

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Name</th>\n";
    s << "<th># Infos</th>\n";
    s << "</tr>\n";

    std::set<std::string> names;
    for( const auto& ipn: m_infos_by_name)
        names.insert( ipn.first);

    for( const auto& name: names) {
        std::string name_html = context.m_html_encoder( name);
        std::string name_url  = context.m_name_url_prefix + context.m_url_encoder( name);
        s << "<tr>\n";
        s << "<td><a href=\"" << name_url << "\">" << name_html << "</a></td>\n";
        s << "<td align=right>" << m_infos_by_name[name]->get_infos().get_size() << "</td>\n";
        s << "</tr>\n";
    };

    s << "</table>\n";
}

void Info_manager::dump_html_tag( std::ostream& s, const Html_context& context, DB::Tag tag)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
    if( !infos_per_tag) {
        s << "<p>No such tag</p>\n";
        return;
    }

    const auto& infos = infos_per_tag->get_infos();

    s << "<div>Number of versions: " << infos.size() << "</div>\n";
    s << "<div>Pin count: " << infos_per_tag->get_pin_count() << "</div>\n";
    s << "<div>Removed: " << to_yes_no( infos_per_tag->get_is_removed()) << "</div>\n";

    s << "<h2>Versions</h2>\n";
    bool add_tag = false;
    dump_html_info_header( s, context, add_tag);
    for( const auto& info: infos)
        dump_html_info( s, context, info, add_tag);
    s << "</table>\n";

    DB::Tag_set reverse_refs = get_reverse_references( tag);

    s << "<h2>Reverse references</h2>\n";
    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Count</th>\n";
    s << "<th>Reverse references</th>\n";
    s << "</tr>\n";
    s << "<tr>\n";
    s << "<td>" << reverse_refs.size() << "</td>\n";
    s << "<td>\n";
    dump_html_tag_set( s, context, reverse_refs);
    s << "</td>\n";
    s << "</tr>\n";
    s << "</table>\n";
}

void Info_manager::dump_html_name(
    std::ostream& s, const Html_context& context, const std::string& name)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    Infos_by_name::iterator it = m_infos_by_name.find( name);
    if( it == m_infos_by_name.end()) {
        s << "<p>No such name</p>\n";
        return;
    }

    const auto& infos = it->second->get_infos();

    s << "<div>Number of versions: " << infos.size() << "</div>\n";

    s << "<h2>Versions</h2>\n";
    bool add_tag = true;
    dump_html_info_header( s, context, add_tag);
    for( const auto& info: infos)
        dump_html_info( s, context, info, add_tag);
    s << "</table>\n";
}

void Info_manager::dump_html_garbage_collection( std::ostream& s, const Html_context& context)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Setting</th>\n";
    s << "<th>Value</th>\n";
    s << "</tr>\n";

    dump_html_string_setting( s, "GC method", context.m_html_encoder( get_gc_method_str()));
    dump_html_size_t_setting( s, "GC period", m_gc_period);
    dump_html_size_t_setting( s, "GC counter", m_gc_counter);

    std::ios old_state( nullptr);
    old_state.copyfmt( s);
    s << std::fixed << std::setprecision( 1);
    dump_html_double_setting( s, "GC interval", m_gc_interval, " s");
    s.copyfmt( old_state);

    s << "</table>\n";
    s << "<p></p>\n";

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Count</th>\n";
    s << "<th>Candidates general</th>\n";
    s << "</tr>\n";
    s << "<tr>\n";
    s << "<td>" << m_gc_candidates_general.size() << "</td>\n";
    s << "<td>";
    dump_html_tag_set( s, context, m_gc_candidates_general);
    s << "</td>\n";
    s << "</tr>\n";
    s << "</table>\n";
    s << "<p></p>\n";

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Count</th>\n";
    s << "<th>Candidates pin count zero</th>\n";
    s << "</tr>\n";
    s << "<tr>\n";
    s << "<td>" << m_gc_candidates_pin_count_zero.size() << "</td>\n";
    s << "<td>";
    dump_html_tag_set( s, context, m_gc_candidates_pin_count_zero);
    s << "</td>\n";
    s << "</tr>\n";
    s << "</table>\n";
}

void Info_manager::dump_html_statistics( std::ostream& s, const Html_context& context)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    Statistics stats;
    size_t sum_count = 0;
    size_t sum_sizes = 0;
    get_statistics( stats, sum_count, sum_sizes);

    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>Count</th>\n";
    s << "<th>Avg Size</th>\n";
    s << "<th>Total Size</th>\n";
    s << "<th>Class ID</th>\n";
    s << "</tr>\n";

    std::ios old_state( nullptr);
    old_state.copyfmt( s);

    for( const auto& entry: stats) {
        size_t avg_size   = (entry.second.m_size/entry.second.m_count + 512) / 1024;
        size_t total_size = (entry.second.m_size + 512) / 1024;
        s << "<tr>\n";
        s << std::setbase( 10) << std::noshowbase ;
        s << "<td align=right>" << entry.second.m_count << "</td>\n";
        s << std::setbase( 10) << std::noshowbase;
        s << "<td align=right>" << avg_size << " kB</td>\n";
        s << std::setbase( 10) << std::noshowbase;
        s << "<td align=right>" << total_size << " kB</td>\n";
        s << "<td>";
        s << std::setbase( 16) << std::showbase << std::setfill( '0') << std::setw( 10);
        s << entry.first;
        s << std::setw( 0);
        s << " " << context.m_html_encoder( decode_class_id( entry.first)) << "</td>\n";
        s << "</tr>\n";
    }

    s << "<tr>\n";
    s << std::setbase( 10) << std::noshowbase ;
    s << "<td align=right>" << sum_count << "</td>\n";
    s << std::setbase( 10) << std::noshowbase;
    if( sum_count == 0) sum_count = 1;
    s << "<td align=right>" << (sum_sizes/sum_count + 512) / 1024 << " kB</td>\n";
    s << std::setbase( 10) << std::noshowbase;
    s << "<td align=right>" << (sum_sizes + 512) / 1024 << " kB</td>\n";
    s << std::setbase( 16) << std::showbase;
    s << "<td>Total</td>\n";
    s << "</tr>\n";

    s << "</table>\n";

    s.copyfmt( old_state);
}

void Info_manager::cleanup_tag_general( DB::Tag tag, bool& progress)
{
    // Note that while we are holding the lock the pin counts of Info_impl's can decrease, but not
    // increase (at least not from zero to non-zero in a legal way).

    m_database->get_lock().check_is_owned();

    progress = false;
    bool temporarily_skipped = false;

    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
    MI_ASSERT( infos_per_tag);
    Infos_per_tag::Infos_per_tag_set& infos = infos_per_tag->get_infos();

    // Consider single infos.
    auto current = infos.begin();
    while( current != infos.end()) {

        // Skip infos with non-zero pin count.
        if( current->get_pin_count() > 0) {
            temporarily_skipped = true;
            ++current;
            continue;
        }

        // Erase infos from aborted transactions.
        const State_and_visibility_ptr& state_and_visibility = current->get_state_and_visibility();
        if( is_aborted( state_and_visibility)) {
            current = cleanup_info( infos_per_tag, current);
            progress = true;
            continue;
        }

        // Erase temporary infos from committed transactions.
        bool committed = is_committed( state_and_visibility);
        if( committed && current->get_is_temporary()) {
            current = cleanup_info( infos_per_tag, current);
            progress = true;
            continue;
        }

        // Erase infos from removed scopes.
        Scope_impl* scope = current->get_scope();
        if( !scope) {
            current = cleanup_info( infos_per_tag, current);
            progress = true;
            continue;
        }

        // Figure out whether the info is globally visible, i.e., for all currently open or future
        // transactions in the subtree of scopes rooted at the scope of this info.
        //
        // Note that the check for the committed state implies that it is not the lowest open
        // transaction and is required to rule out the ID equality check in is_visible_for().
        DB::Transaction_id creator_id = current->get_transaction_id();
        DB::Transaction_id lowest_open_id = scope->get_lowest_open_transaction_id();
        bool globally_visible = !state_and_visibility
            || (committed && is_visible_for( creator_id, state_and_visibility, lowest_open_id));

        // Clear creator transaction for infos that are globally visible. This is just book-keeping
        // and does not count as GC progress w.r.t. the infos.
        if( state_and_visibility) {
            if( globally_visible) {
                current->release_state_and_visiblity();
                MI_ASSERT( !state_and_visibility);
            } else {
                temporarily_skipped = true;
            }
        }

        // Process globally visible removal infos.
        if( is_global_removal( current)) {
            if( globally_visible) {
                if( !infos_per_tag->get_is_removed()) {
                    infos_per_tag->set_removed();
                    mi::Uint32 pin_count = infos_per_tag->unpin();
                    if( (pin_count == 0) && (m_gc_method != GC_FULL_SWEEPS_ONLY))
                        m_gc_candidates_pin_count_zero.insert( tag);
                }
                current = cleanup_info( infos_per_tag, current);
                progress = true;
                continue;
             } else {
                temporarily_skipped = true;
             }
        }

        ++current;
    }

    // Remove sets that are marked for removal and are not referenced.
    if( infos_per_tag->get_pin_count() == 0) {
        bool progress_pin_count_zero = false;
        cleanup_tag_with_pin_count_zero( infos_per_tag, progress_pin_count_zero);
        if( progress_pin_count_zero) {
            progress = true;
            return;
        }
    }

    // Remove empty sets (from aborted transactions with no other info version).
    if( infos.empty()) {
        m_infos_by_tag.erase( tag);
        delete infos_per_tag;
        if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
            m_gc_candidates_general.erase( tag);
        progress = true;
        return;
    }

    // Consider pairs of subsequent infos.
    while( true) {

        size_t n = infos.get_size();

        // Iterate over the entire set ...
        current = infos.begin();
        while( current != infos.end()) {

            auto next = current;
            ++next;

            if( next == infos.end())
                break;

            bool same_scope = current->get_scope_id() == next->get_scope_id();

            // Remove current info in favor of next one if both are from the same scope and have
            // the same visibility, unless one of them is a global removal (which are processed
            // individually further up), the current info is a local removal (processed further
            // down in a different iteration of the loop), or the next info is temporary.
            if(    same_scope
                && !is_removal( current)
                && !is_global_removal( next)
                && !next->get_is_temporary()) {

                const auto& current_sav = current->get_state_and_visibility();
                const auto& next_sav    = next->get_state_and_visibility();

                if(    same_visibility( current_sav, next_sav)
                    && (current->get_pin_count() == 0)) {
                    current = cleanup_info( infos_per_tag, current);
                    progress = true;
                    continue;
                } else {
                    temporarily_skipped = true;
                    ++current;
                    continue;
                }
            }

            // Remove the next info if it is a local removal and it is the first in its scope,
            // i.e., it is in a different scope than the current info.
            if( is_local_removal( next) && !same_scope) {
                if( next->get_pin_count() == 0) {
                    /*next =*/ cleanup_info( infos_per_tag, next);
                    progress = true;
                    continue;
                } else {
                    temporarily_skipped = true;
                    ++current;
                    continue;
                }
            }

            ++current;
        }

        // ... until no (further) progress is possible.
        if( infos.get_size() == n)
            break;
    }

    if( (m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO) && !temporarily_skipped)
        m_gc_candidates_general.erase( tag);

    MI_ASSERT( !infos.empty());
}

void Info_manager::cleanup_tag_with_pin_count_zero( Infos_per_tag* infos_per_tag, bool& progress)
{
    m_database->get_lock().check_is_owned();

    MI_ASSERT( infos_per_tag);
    MI_ASSERT( infos_per_tag->get_pin_count() == 0);

    DB::Tag tag = infos_per_tag->get_tag();
    Infos_per_tag::Infos_per_tag_set& infos = infos_per_tag->get_infos();

    auto add = []( mi::Uint32 sum, const Info_impl& i) { return sum + i.get_pin_count(); };
    mi::Uint32 sum_info_pin_counts = std::accumulate( infos.begin(), infos.end(), 0, add);
    if( sum_info_pin_counts == 0) {

        auto current = infos.begin();
        while( current != infos.end())
            current = cleanup_info( infos_per_tag, current);
        m_infos_by_tag.erase( tag);
        delete infos_per_tag;
        if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
            m_gc_candidates_general.erase( tag);
        if( m_gc_method != GC_FULL_SWEEPS_ONLY)
            m_gc_candidates_pin_count_zero.erase( tag);
        progress = true;

    } else {

        if( m_gc_method != GC_FULL_SWEEPS_ONLY)
            m_gc_candidates_pin_count_zero.insert( tag);
        progress = false;

    }
}

Infos_per_tag::Infos_per_tag_set::iterator Info_manager::cleanup_info(
    Infos_per_tag* infos_per_tag, Infos_per_tag::Infos_per_tag_set::iterator it)
{
    m_database->get_lock().check_is_owned();

    Info_impl* info = & *it;
    MI_ASSERT( info->get_pin_count() == 0);

    const char* name = info->get_name();
    if( name) {
        Infos_per_name* infos_per_name = info->get_infos_per_name();
        infos_per_name->erase_info( info);
        if( infos_per_name->get_infos().empty()) {
            m_infos_by_name.erase( name);
            delete infos_per_name;
        }
    }

    auto next = infos_per_tag->erase_info( info);

    Scope_impl* scope = info->get_scope();
    if( scope)
        scope->erase_info( info);

    const DB::Tag_set& old_references = info->get_references();
    decrement_pin_counts( old_references, /*from_gc*/ true);
    delete info;

    return next;
}

void Info_manager::increment_pin_counts( const DB::Tag_set& tag_set)
{
    m_database->get_lock().check_is_owned();

    for( const DB::Tag& tag: tag_set) {
        Infos_per_tag* ipt = m_infos_by_tag.find( tag);
        MI_ASSERT( ipt);
        mi::Uint32 pin_count = ipt->pin();
        if( (pin_count == 1) && (m_gc_method != GC_FULL_SWEEPS_ONLY))
            m_gc_candidates_pin_count_zero.erase( tag);
    }
}

void Info_manager::decrement_pin_counts( const DB::Tag_set& tag_set, bool from_gc)
{
    m_database->get_lock().check_is_owned();

    for( const DB::Tag& tag: tag_set) {
        Infos_per_tag* ipt = m_infos_by_tag.find( tag);
        // With aborted transactions it can happen that the referenced element was already removed
        // in the current garbage collection run.
        MI_ASSERT( ipt || from_gc);
        (void) from_gc;
        if( !ipt)
            continue;
        mi::Uint32 pin_count = ipt->unpin();
        if( (pin_count == 0) && (m_gc_method != GC_FULL_SWEEPS_ONLY))
            m_gc_candidates_pin_count_zero.insert( tag);
    }
}

void Info_manager::get_statistics( Statistics& stats, size_t& sum_count, size_t& sum_sizes) const
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    stats.clear();

    auto get_stats = [&stats]( Infos_per_tag* ipt)
    {
        for( const Info_impl& info: ipt->get_infos()) {
            DB::Element_base* element = info.get_element();
            if( element) {
                SERIAL::Class_id class_id = element->get_class_id();
                Count_and_size& cs = stats[class_id];
                ++cs.m_count;
                cs.m_size += element->get_size();
            }
            SCHED::Job_base* job = info.get_job();
            if( job) {
                SERIAL::Class_id class_id = job->get_class_id();
                Count_and_size& cs = stats[class_id];
                ++cs.m_count;
                cs.m_size += job->get_size();
            }
        }
    };
    m_infos_by_tag.apply( get_stats);

    sum_count = 0;
    for( const auto& entry: stats)
        sum_count += entry.second.m_count;

    sum_sizes = 0;
    for( const auto& entry: stats)
        sum_sizes += entry.second.m_size;
}

void Info_manager::dump_html_info_header(
    std::ostream& s, const Html_context& context, bool add_tag)
{
    s << "<table border cellspacing=0 cellpadding=5>\n";
    s << "<tr>\n";
    s << "<th>" << (add_tag ? "Tag" : "Name") << "</th>\n";
    s << "<th>Class ID</th>\n";
    s << "<th>Scope ID</th>\n";
    s << "<th>Creator TX</th>\n";
    s << "<th>Version</th>\n";
    s << "<th>Creator TX State</th>\n";
    s << "<th>Visibility ID</th>\n";
    s << "<th>Pin count</th>\n";
    s << "<th>Removed</th>\n";
    s << "<th># Ref.</th>\n";
    s << "<th>References</th>\n";
    s << "</tr>\n";
}


void Info_manager::dump_html_info(
    std::ostream& s, const Html_context& context, const Info_impl& info, bool add_tag)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    DB::Tag tag = info.get_tag();

    const char* name      = info.get_name();
    std::string name_html = name ? context.m_html_encoder( name) : "-";
    std::string name_url  = name ? context.m_url_encoder( name)  : "";

    SERIAL::Class_id class_id = SERIAL::class_id_unknown;
    if( info.get_element())
        class_id = info.get_element()->get_class_id();
    else if( info.get_job())
        class_id = info.get_job()->get_class_id();

    const State_and_visibility_ptr& sv = info.get_state_and_visibility();
    std::ostringstream state;
    if( !sv)
        state << "COMMITTED";
    else
        state << sv->m_state;
    std::string visibility;
    if( !sv)
        visibility = "(globally visible)";
    else if( sv->m_state == Transaction_impl::OPEN)
        visibility = "(transaction only)";
    else
        visibility = std::to_string( sv->m_id());

    const DB::Tag_set& references = info.get_references();

    s << "<tr>\n";
    if( add_tag) {
        s << "<td align=right>";
        s << "<a href=\"" << context.m_tag_url_prefix << tag() << "\">" << tag() << "</a>";
        s << "</td>\n";
    } else if( name) {
        s << "<td>";
        s << "<a href=\"" << context.m_name_url_prefix << name_url << "\">" << name_html << "</a>";
        s << "</td>\n";
    } else
        s << "<td>" << name_html << "</td>\n";

    if( class_id != SERIAL::class_id_unknown) {
        s << "<td>";
        std::ios old_state( nullptr);
        old_state.copyfmt( s);
        s << std::setbase( 16) << std::showbase << std::setfill( '0') << std::setw( 10);
        s << class_id;
        s.copyfmt( old_state);
        s  << " " << context.m_html_encoder( decode_class_id( class_id)) << "</td>\n";
    } else {
        s << "<td>-</td>\n";
    }

    s << "<td align=right>"  << info.get_scope_id() << "</td>\n";
    s << "<td align=right>"  << info.get_transaction_id()() << "</td>\n";
    s << "<td align=right>"  << info.get_version() << "</td>\n";
    s << "<td align=center>" << state.str() << "</td>\n";
    s << "<td align=right>"  << visibility << "</td>\n";
    s << "<td align=right>"  << info.get_pin_count() << "</td>\n";
    s << "<td align=center>" << to_yes_no( info.get_is_removal()) << "</td>\n";
    s << "<td align=right>"  << references.size() << "</td>\n";

    s << "<td>\n";
    dump_html_tag_set( s, context, references);
    s << "</td>\n";

    s << "</tr>\n";
}

void Info_manager::dump_html_tag_set(
    std::ostream& s, const Html_context& context, const DB::Tag_set& tag_set)
{
    if( tag_set.empty()) {
        s << "-";
        return;
    }

    std::set<DB::Tag> ordered_tag_set;
    for( const auto& tag: tag_set)
        ordered_tag_set.insert( tag);

    for( const DB::Tag& tag: ordered_tag_set)
        s << "<a href=\"" << context.m_tag_url_prefix << tag() << "\">" << tag() << "</a>\n";
}

DB::Tag_set Info_manager::get_reverse_references( DB::Tag tag) const
{
    m_database->get_lock().check_is_owned();

    DB::Tag_set result;

    auto get_reverse_refs = [&tag, &result]( Infos_per_tag* ipt)
    {
        for( const Info_impl& info: ipt->get_infos()) {
            DB::Element_base* element = info.get_element();
            if( !element)
                continue;
            const DB::Tag_set& references = info.get_references();
            if( references.find( tag) != references.end()) {
                result.insert( info.get_tag());
                break;
            }
        }
    };
    m_infos_by_tag.apply( get_reverse_refs);

    return result;
}

std::string Info_manager::get_gc_method_str() const
{
    switch( m_gc_method) {
        case Info_manager::GC_FULL_SWEEPS_ONLY:
            return "full_sweeps_only";
        case Info_manager::GC_FULL_SWEEP_THEN_PIN_COUNT_ZERO:
            return "full_sweep_then_pin_count_zero";
        case Info_manager::GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO:
            return "general_candidates_then_pin_count_zero";
    }
    return {};
}

} // namespace DBLIGHT

} // namespace MI
