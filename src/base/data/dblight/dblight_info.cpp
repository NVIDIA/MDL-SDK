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

#include "dblight_info.h"

#include <numeric>
#include <sstream>

#include <boost/core/ignore_unused.hpp>

#include <base/data/db/i_db_element.h>
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
    const char* name,
    DB::Tag_set& references)
  : Info_base( scope->get_id(), transaction->get_id(), version),
    m_element( element),
    m_tag( tag),
    m_name( name),
    m_scope( scope),
    m_references( std::move( references)),
    m_transaction( transaction),
    m_privacy_level( privacy_level)
{
    MI_ASSERT( m_references.find( m_tag) == m_references.end());
    MI_ASSERT( m_references.find( DB::Tag()) == m_references.end());
}

Info_impl::Info_impl(
    Scope_impl* scope,
    Transaction_impl* transaction,
    mi::Uint32 version,
    DB::Tag tag)
  : Info_base( scope->get_id(), transaction->get_id(), version),
    m_tag( tag),
    m_scope( scope),
    m_transaction( transaction),
    m_privacy_level( scope->get_level())
{
}

Info_impl::~Info_impl()
{
    delete m_element;
}

void Info_impl::pin()
{
    MI_ASSERT( m_element);
    ++m_pin_count;
}

void Info_impl::unpin()
{
    // No assertion for m_element to allow unpinning a just constructed instance for removal
    // operations with pin count 1. (We could initialize m_pin_count to 0 in that particular
    // constructor, but that would be inconsistent and counter-intuitive for callers.)
    --m_pin_count;
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
    MI_ASSERT( m_name);
    MI_ASSERT( !!infos_per_name ^ !!m_infos_per_name);

    m_infos_per_name = infos_per_name;
}

void Info_impl::set_element( DB::Element_base* element)
{
    MI_ASSERT( m_element);
    MI_ASSERT( element);

    delete m_element;
    m_element = element;
}

namespace {

/// Indicates whether the transaction has been committed.
bool is_committed( const Transaction_impl_ptr& transaction)
{
    return transaction && (transaction->get_state() == Transaction_impl::COMMITTED);
}

/// Indicates whether the transaction has been aborted.
bool is_aborted( const Transaction_impl_ptr& transaction)
{
    return transaction && (transaction->get_state() == Transaction_impl::ABORTED);
}

/// Indicates whether two transactions definitely have the same visibility.
///
/// Assumes that globally visible transactions have already been cleared.
///
/// Note that the computation based on the visibility ID is an approximation and the method errs
/// on the safe side for GC purposes. It returns \c true if both transaction definitely have the
/// same visibility. It might return \c false if the visibility is indeed identical, but this
/// cannot be determined due to the approximation scheme.
bool same_visibility( const Transaction_impl_ptr& lhs, const Transaction_impl_ptr& rhs)
{
    // Indistinguishable transactions (either identical IDs, or both globally visible).
    if( lhs == rhs)
        return true;
    // Exactly one transaction globally visible.
    if( !!lhs ^ !!rhs)
        return false;
    // Compare visibility IDs (Could be improved by replacing each visibility ID by the lowest open
    // transaction larger than or equal to that visibility ID.)
    return lhs->get_visibility_id() == rhs->get_visibility_id();
}

/// Indicates whether the info is a removal info.
template <class T>
bool is_removal( T info)
{
    return info->get_is_removal();
}

/// Indicates whether the info is a global removal info.
template <class T>
bool is_global_removal( T info)
{
    return info->get_is_removal() && info->get_scope_id() == 0;
}

/// Indicates whether the info is a local removal info.
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
            const Transaction_impl_ptr& creator_transaction = it->get_transaction();
            if( !creator_transaction) {
                // Visible local removal infos hide any previous infos in that scope.
                if( is_local_removal( it))
                     break;
                it->pin();
                if( level_found)
                    *level_found = scope->get_level();
                return & *it;
            }

            // Check whether the info is from an aborted transaction (those can never be found).
            // With a synchronous garbage collection those infos should actually never survive the
            // lock unless the user incorrectly pins them while aborting the transaction.
            if( creator_transaction->get_state() == Transaction_impl::ABORTED) {
                MI_ASSERT( !"Found info from aborted transaction");
                if( it == infos.begin())
                    break;
                --it;
                continue;
            }

            // Check whether the info is visible for the given transaction ID.
            if( creator_transaction->is_visible_for( transaction_id)) {
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

} // namespace IMPL

void Infos_per_name::insert_info( Info_impl* info)
{
    MI_ASSERT( info->get_name() == m_name.c_str());

    auto result = m_infos.insert( *info);
    MI_ASSERT( result.second);
    boost::ignore_unused( result);

    info->set_infos_per_name( this);
}

Infos_per_name::Infos_per_name_set::iterator Infos_per_name::erase_info( Info_impl* info)
{
    MI_ASSERT( info->get_name() == m_name.c_str());

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

    auto result = m_infos.insert( *info);
    MI_ASSERT( result.second);
    boost::ignore_unused( result);
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
}

Info_manager::~Info_manager()
{
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

    // Retrieve (or create) set of infos for \p name (if not \c NULL).
    Infos_per_name* infos_per_name = nullptr;
    if( name) {
        auto it_by_name = m_infos_by_name.find( name);
        if( it_by_name == m_infos_by_name.end()) {
            infos_per_name = new Infos_per_name( name);
            m_infos_by_name[name] = infos_per_name;
        } else {
            infos_per_name = it_by_name->second;
        }
        // Re-map name to a pointer that is guaranteed to exist as long as we need it.
        name = infos_per_name->get_name().c_str();
    }

    // Record DB element references of this info.
    increment_pin_counts( references);

    // Create info (destroys references).
    auto* info = new Info_impl(
        element, scope, transaction, version, tag, privacy_level, name, references);

    // Insert info into the sets of infos for that tag/name/scope.
    infos_per_tag->insert_info( info);
    if( infos_per_name)
        infos_per_name->insert_info( info);
    scope->insert_info( info);

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);

    info->unpin();
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

    MI_ASSERT( name);

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
    const char* name,
    const DB::Tag_set& references)
{
    m_database->get_lock().check_is_owned();

    // Retrieve set of infos for \p tag.
    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);

    // Retrieve set of infos for \p name (if not \c NULL).
    Infos_per_name* infos_per_name = nullptr;
    if( name) {
        infos_per_name = m_infos_by_name.find( name)->second;
        // No need to re-map name (it points already to the re-mapped destination).
        MI_ASSERT( name == infos_per_name->get_name().c_str());
    }

    // Create info (modifies copy_references).
    DB::Tag_set copy_references( references);
    auto* info = new Info_impl(
        element, scope, transaction, version, tag, privacy_level, name, copy_references);

    // Insert info into the sets of infos for that tag/name/scope.
    infos_per_tag->insert_info( info);
    if( infos_per_name)
        infos_per_name->insert_info( info);
    scope->insert_info( info);

    // Record DB element references of this info.
    increment_pin_counts( references);

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);

    return info;
}

void Info_manager::finish_edit( Info_impl* info, Transaction_impl* transaction)
{
    m_database->get_lock().check_is_owned();

    const DB::Tag_set& old_references = info->get_references();
    decrement_pin_counts( old_references, /*from_gc*/ false);

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

    if( is_global_removal) {

        // Prevent double global removals (still counts as successful request).
        if( ipt->get_is_removed())
            return true;

        // Make sure that global removals are recorded as such.
        if( scope->get_id() != 0)
            scope = static_cast<Scope_impl*>( m_database->get_scope_manager()->lookup_scope( 0));

    } else {

        Info_impl* info = ipt->lookup_info( scope, transaction->get_id());
        if( !info)
            return false;

        // Reject local removals without a version in the current scope.
        if( info->get_scope_id() != scope->get_id()) {
            info->unpin();
            return false;
        }

        info->unpin();

        // Reject local removals without another version in a more global scope (otherwise we can
        // end up with invalid tag references).
        auto* parent_scope = static_cast<Scope_impl*>( scope->get_parent());
        Info_impl* parent_info = ipt->lookup_info( parent_scope, transaction->get_id());
        if( !parent_info)
            return false;

        parent_info->unpin();
    }

    // Create removal info.
    auto* info = new Info_impl( scope, transaction, version, tag);

    // Insert info into the sets of infos for that tag/scope (not name).
    ipt->insert_info( info);
    scope->insert_info( info);

    // Consider tag as a candidate for garbage collection.
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);

    // Prevent double global removals.
    if( is_global_removal)
        ipt->set_removed();

    info->unpin();
    return true;
}

void Info_manager::consider_tag_for_gc( DB::Tag tag)
{
    if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        m_gc_candidates_general.insert( tag);
}

void Info_manager::garbage_collection( DB::Transaction_id lowest_open)
{
    m_database->get_lock().check_is_owned();

    if( m_gc_method == GC_FULL_SWEEPS_ONLY) {

        while( true) {

            std::vector<DB::Tag> tags;
            tags.reserve( m_infos_by_tag.size());
            m_infos_by_tag.get_tags( tags);

            bool progress = false;
            for( const auto& tag: tags) {
                bool progress_tag = false;
                cleanup_tag_general( tag, lowest_open, progress_tag);
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
            cleanup_tag_general( tag, lowest_open, progress_tag);
        }

        bool progress = true;

        while( true) {

            if( !progress || m_gc_candidates_pin_count_zero.empty())
                break;

            DB::Tag_set tags2 = std::move( m_gc_candidates_pin_count_zero);
            progress = false;

            for( const auto& tag: tags2) {
                bool progress_tag = false;
                cleanup_tag_with_pin_count_zero( tag, progress_tag);
                progress |= progress_tag;
            }
        }

    } else if( m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO) {

        DB::Tag_set tags = m_gc_candidates_general;
        for( const auto& tag: tags) {
            bool progress_tag = false;
            cleanup_tag_general( tag, lowest_open, progress_tag);
        }

        bool progress = true;

        while( true) {

            if( !progress || m_gc_candidates_pin_count_zero.empty())
                break;

            tags = std::move( m_gc_candidates_pin_count_zero);
            progress = false;

            for( const auto& tag: tags) {
                bool progress_tag = false;
                cleanup_tag_with_pin_count_zero( tag, progress_tag);
                progress |= progress_tag;
            }
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

bool Info_manager::get_tag_is_removed( DB::Tag tag)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

   // Retrieve set of infos for \p tag.
    Infos_per_tag* ipt = m_infos_by_tag.find( tag);
    if( !ipt)
        return false;

    return ipt->get_is_removed();
}

namespace {

std::ostream& operator<<( std::ostream& s, const DB::Tag_set& tag_set)
{
    bool first = true;
    s << "{";

    for( const auto& tag: tag_set) {
        if( !first)
            s << ",";
        s << " " << tag();
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

        s << "    Index " << j2++;
        if( !mask_pointer_values)
            s << " at " << &i;
        s << ": ";
        s << "scope ID = " << i.get_scope_id() << ", ";
        // Omit the scope pointer. With a correct synchronous GC the cleared state should never be
        // visible from the outside.
        s << "transaction ID = " << i.get_transaction_id()();
        if( !mask_pointer_values)
            s << " (" << i.get_transaction().get() << ")";
        else
            s << (i.get_transaction().get() ? " (set)" : " (cleared)");
        s << ", version = " << i.get_version();
        s << ", pin count = " << i.get_pin_count();
        s << ", tag = " << i.get_tag()();
        s << ", privacy level = " << static_cast<mi::Uint32>( i.get_privacy_level());
        s << ", removal = " << i.get_is_removal();
        if( !mask_pointer_values)
            s << ", element = " << i.get_element();
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

        s << "    Index " << j2++;
        if( !mask_pointer_values)
            s << " at " << &i;
        s << ": ";
        s << "scope ID = " << i.get_scope_id() << ", ";
        // Omit the scope pointer. With a correct synchronous GC the cleared state should never be
        // visible from the outside.
        s << "transaction ID = " << i.get_transaction_id()();
        if( !mask_pointer_values)
            s << " (" << i.get_transaction().get() << ")";
        else
            s << (i.get_transaction().get() ? " (set)" : " (cleared)");
        s << ", version = " << i.get_version();
        s << ", pin count = " << i.get_pin_count();
        s << ", tag = " << i.get_tag()();
        s << ", privacy level = " << static_cast<mi::Uint32>( i.get_privacy_level());
        s << ", removal = " << i.get_is_removal();
        if( !mask_pointer_values)
            s << ", element = " << i.get_element();
        s << ", name = " << name_str;
        s << ", references = " << i.get_references();
        s << std::endl;
    }
}

} // namespace

void Info_manager::dump( std::ostream& s, bool mask_pointer_values)
{
    m_database->get_lock().check_is_owned_shared_or_exclusive();

    s << "Count of infos by distinct names: " << m_infos_by_name.size() << std::endl;

    size_t j1 = 0;
    // Dump by order of names, not by order of hashes.
    std::set<std::string> names;
    for( const auto& ipn: m_infos_by_name)
        names.insert( ipn.first);
    for( const auto& name: names)
        DBLIGHT::dump( s, mask_pointer_values, m_infos_by_name[name], j1++);
    if( !m_infos_by_name.empty())
        s << std::endl;

    s << "Count of infos by distinct tags: " << m_infos_by_tag.size() << std::endl;

    j1 = 0;
    auto dump_as_lambda = [&s, mask_pointer_values, &j1]( Infos_per_tag* ipt)
    { DBLIGHT::dump( s, mask_pointer_values, ipt, j1++); };
    m_infos_by_tag.apply( dump_as_lambda);
    if( !m_infos_by_tag.empty())
        s << std::endl;

    s << std::endl;
}

void Info_manager::cleanup_tag_general( DB::Tag tag, DB::Transaction_id lowest_open, bool& progress)
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
        const Transaction_impl_ptr& transaction = current->get_transaction();
        if( is_aborted( transaction)) {
            current = cleanup_info( infos_per_tag, current);
            progress = true;
            continue;
        }

        // Clear creator transaction for infos that are globally visible. This is required to
        // eventually release the transaction, but does not count as GC progress w.r.t. the infos.
        //
        // Note that the check for the committed state implies that it is not the lowest open
        // transaction and is required to rule out the ID equality check in is_visible_for().
        //
        // This can be improved as follows to clear transactions faster in case of unrelated scopes:
        // We do not really need to know whether that info is globally visible, just whether it is
        // visible in all open transactions in the entire subtree of scopes rooted at the scope of
        // this info. This could be checked by tracking the lowest open transaction ID per scope
        // and computing the lowest open transaction ID for the scope subtree rooted at the scope
        // of this info.
        bool globally_visible = !transaction
            || (is_committed( transaction) && transaction->is_visible_for( lowest_open));
        if( transaction) {
            if( globally_visible) {
                current->clear_transaction();
                MI_ASSERT( !transaction);
            } else {
                temporarily_skipped = true;
            }
        }

        // Erase infos from removed scopes.
        if( !current->get_scope()) {
            current = cleanup_info( infos_per_tag, current);
            progress = true;
            continue;
        }

        // Process globally visible removal infos.
        if( is_global_removal( current)) {
            if( globally_visible) {
                mi::Uint32 pin_count = infos_per_tag->unpin();
                if( (pin_count == 0) && (m_gc_method != GC_FULL_SWEEPS_ONLY))
                    m_gc_candidates_pin_count_zero.insert( tag);
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
        cleanup_tag_with_pin_count_zero( tag, progress_pin_count_zero);
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
            // invidividually further up) or the current info is a local removal (processed further
            // down in a different iteration of the loop).
            const Transaction_impl_ptr& current_transaction = current->get_transaction();
            const Transaction_impl_ptr& next_transaction = next->get_transaction();
            if( same_scope && !is_removal( current) && !is_global_removal( next)) {
                if(    same_visibility( current_transaction, next_transaction)
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

    if(    (m_gc_method == GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO)
        && (infos.get_size() == 1)
        && !temporarily_skipped)
        m_gc_candidates_general.erase( tag);

    MI_ASSERT( !infos.empty());
}

void Info_manager::cleanup_tag_with_pin_count_zero( DB::Tag tag, bool& progress)
{
    m_database->get_lock().check_is_owned();

    Infos_per_tag* infos_per_tag = m_infos_by_tag.find( tag);
    MI_ASSERT( infos_per_tag);
    MI_ASSERT( infos_per_tag->get_pin_count() == 0);
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

} // namespace DBLIGHT

} // namespace MI
