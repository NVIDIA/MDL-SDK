/***************************************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_SCOPE_H
#define BASE_DATA_DBLIGHT_DBLIGHT_SCOPE_H

#include <base/data/db/i_db_scope.h>

#include <atomic>

#include <boost/core/noncopyable.hpp>
#include <boost/intrusive/set.hpp>

#include "dblight_info.h"
#include "dblight_util.h"

namespace MI {

namespace DBLIGHT {

class Database_impl;
class Info_impl;
class Scope_manager;

namespace bi = boost::intrusive;

/// A scope of the database.
class Scope_impl : public DB::Scope
{
public:
    /// Constructor.
    ///
    /// \param database        Instance of the database this scope belongs to.
    /// \param scope_manager   Manager that creates this scope.
    /// \param id              ID of this scope (0 for the global scope).
    /// \param name            Name of this scope (or the empty string).
    /// \param parent          Parent scope (or \c NULL for the global scope).
    /// \param level           Privacy level of the scope (0 for the global scope).
    Scope_impl(
        Database_impl* database,
        Scope_manager* scope_manager,
        DB::Scope_id id,
        std::string name,
        Scope_impl* parent,
        DB::Privacy_level level);

    /// Destructor.
    virtual ~Scope_impl();

    // methods of DB::Scope

    void pin() override { ++m_pin_count; }

    void unpin() override { if( --m_pin_count == 0) delete this; }

    DB::Scope_id get_id() const override { return m_id; }

    const std::string& get_name() const override { return m_name; }

    DB::Scope* get_parent() const override { return m_parent; }

    DB::Privacy_level get_level() const override { return m_level; }

    /// The \p is_temporary parameter is meaningless in this implementation.
    DB::Scope* create_child(
        DB::Privacy_level level, bool /*is_temporary*/, const std::string& name) override;

    DB::Transaction* start_transaction() override;

    // internal methods

    /// Returns the current pin count.
    mi::Uint32 get_pin_count() const { return m_pin_count; }

    /// Indicates whether the scope has been marked for removal.
    bool get_is_removed() const { return m_is_removed; }

    /// Marks the scope for removal.
    void set_is_removed() { m_is_removed = true; }

    /// Inserts the info into the info set of this scope.
    ///
    /// \pre info->get_id() equals get_id()
    /// \pre The set does not contain the info already.
    ///
    /// \param info   The info to insert. RCS:NEU
    void insert_info( Info_impl* info);

    /// Removes the info from the info set of this scope.
    ///
    /// \pre info->get_id() equals get_id()
    ///
    /// \param info   The info to remove. RCS:NEU
    void erase_info( Info_impl* info);

private:
    /// The database instance this scope belongs to.
    Database_impl* const m_database;
    /// The manager that created this scope.
    Scope_manager* const m_scope_manager;

    /// The ID of the scope.
    const DB::Scope_id m_id;
    /// The name of the scope.
    const std::string m_name;
    /// The parent scope (or \c NULL for the global scope).
    Scope_impl* const m_parent;
    /// The privacy level of the scope.
    const DB::Privacy_level m_level;
    /// Reference count of the scope.
    std::atomic_uint32_t m_pin_count = 1;
    /// Indicates whether this scope was already marked for removal.
    bool m_is_removed = false;

    using Infos_hook = bi::member_hook<
        Info_impl, bi::list_member_hook<>, &Info_impl::m_scope_hook>;

    using Infos_list = bi::list<Info_impl, Infos_hook>;

    /// List of all infos in this scope.
    Infos_list m_infos;

public:
    // Key for Scope_manager::m_scopes_by_id.
    struct Id_is_key {
        using type = DB::Scope_id;
        const type& operator()( const Scope_impl& s) const { return s.m_id; }
    };

    // Key for Scope_manager::m_scopes_by_name.
    struct Name_is_key {
        using type = std::string;
        const type& operator()( const Scope_impl& s) const { return s.m_name; }
    };

    /// Hook for Scope_manager::m_scopes_by_id.
    bi::set_member_hook<> m_scopes_by_id_hook;
    /// Hook for Scope_manager::m_scopes_by_name.
    bi::set_member_hook<> m_scopes_by_name_hook;
};

/// Manager for scopes.
class Scope_manager : private boost::noncopyable
{
public:
    /// Constructor.
    ///
    /// \param database   Instance of the database this manager belongs to.
    Scope_manager( Database_impl* database);

    /// Destructor.
    ///
    /// Unpins all scopes in reverse creation order.
    ~Scope_manager();

    /// Looks up and returns a scope with a given ID.
    ///
    /// \param id     The ID of the scope as returned by #DB::Scope::get_id(). The global scope has
    ///               ID 0.
    /// \return       The found scope or \c NULL if no such scope exists. RCS:NEU
    DB::Scope* lookup_scope( DB::Scope_id id);

    /// Looks up a named scope.
    ///
    /// \param name   The name of the scope as returned by #DB::Scope::get_name(). The global scope
    ///               has the empty string as name.
    /// \return       The found scope or \c NULL if no such scope exists. RCS:NEU
    DB::Scope* lookup_scope( const std::string& name);

    /// Creates a new scope (or retrieves an already existing named scope).
    ///
    /// \param name     The name of the scope. The empty string creates an unnamed scope.
    /// \param parent   The parent scope (or \c NULL for the global scope).
    /// \param level    Privacy level for the new scope. Must be higher than the privacy level of
    ///                 the parent scope.
    /// \return         The created child scope, or \c NULL in case of failure:
    ///                 - Missing parent scope.
    ///                 - Privacy level not higher than that of the parent scope.
    ///                 - A scope with that name exists already, but with different parent scope
    ///                 and/or privacy level.
    ///                 RCS:NEU
    DB::Scope* create_scope( const std::string& name, DB::Scope* parent, DB::Privacy_level level);

    /// Removes a scope from the database.
    ///
    /// \param id     The ID of the scope to remove
    /// \return       \c true in case of success, \c false otherwise (invalid scope ID, or already
    ///               marked for removal).
    bool remove_scope( DB::Scope_id);

    /// Removes a scope from the sets of scopes.
    ///
    /// Used by the destructor of Scope_impl to remove itself from these sets.
    ///
    /// \param scope   The scope to remove. RCS:NEU
    void remove_scope_internal( Scope_impl* scope);

    /// Dumps the state of the scope manager to the stream.
    void dump( std::ostream& s, bool mask_pointer_values);

private:
    /// Instance of the database this manager belongs to.
    Database_impl* const m_database;

    using Scopes_by_id_hook = bi::member_hook<
        Scope_impl, bi::set_member_hook<>, &Scope_impl::m_scopes_by_id_hook>;

    using Scopes_by_id_map = bi::set<
        Scope_impl, Scopes_by_id_hook, bi::key_of_value<Scope_impl::Id_is_key>>;

    /// Set of all scopes sorted by ID.
    Scopes_by_id_map m_scopes_by_id;

    using Scopes_by_name_hook = bi::member_hook<
        Scope_impl, bi::set_member_hook<>, &Scope_impl::m_scopes_by_name_hook>;

    using Scopes_by_name_map = bi::set<
        Scope_impl, Scopes_by_name_hook, bi::key_of_value<Scope_impl::Name_is_key>>;

    /// Set of all named scopes sorted by name.
    Scopes_by_name_map m_scopes_by_name;

    /// ID of the next scope to be created.
    DB::Scope_id m_next_scope_id = 0;
};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_SCOPE_H
