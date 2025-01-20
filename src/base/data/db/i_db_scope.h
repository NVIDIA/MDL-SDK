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

#ifndef BASE_DATA_DB_I_DB_SCOPE_H
#define BASE_DATA_DB_I_DB_SCOPE_H

#include <memory>
#include <string>

#include <boost/core/noncopyable.hpp>

#include "i_db_journal_type.h"

#include <mi/base/types.h>

namespace MI {

namespace DB {

class Transaction;

/// The privacy level defines to which level changes to an element go.
using Privacy_level = mi::Uint8;

/// Each scope is identified by a cluster-unique scope ID.
using Scope_id = mi::Uint32;

/// A scope is the context which determines the visibility of database elements.
///
/// Scopes are organized in a tree-like fashion and have a so-called \em privacy \em level. The root
/// of the tree is called \em global \em scope and has privacy level 0. On each path from the global
/// scope to one of the leafs of the tree the privacy levels are strictly increasing. Scopes are
/// identified with a cluster-unique ID which can be used to access a scope on any host in the
/// cluster.
///
/// A database element stored in a given scope is only visible in this scope and all child scopes.
/// For example, a database element stored in the global scope is visible in all scopes. This
/// visibility concept for database elements is similar to visibility of stack variables in
/// programming languages.
///
/// Any database element can exist in multiple versions (at most one version per scope). In this
/// case the scope at hand does not just determine the visibility itself but also determines which
/// version is visible. The version from the current scope has highest priority, next is the version
/// from the parent scope, etc., until the global scope is reached. Again, this is similar to
/// shadowing of variables with the same name in programming languages.
class Scope : private boost::noncopyable
{
public:
    /// Pins the scope, incrementing its reference count.
    virtual void pin() = 0;

    /// Unpins the scope, decrementing its reference count.
    ///
    /// When the user holds no more references, the database may decide to destroy the scope.
    /// Although the user may no longer use the scope it may actually live longer than the last
    /// reference from the user, see #IDatabase::remove_scope() for details.
    ///
    /// Note that the user should commit or abort all transactions from this scope before releasing
    /// the last reference to it.
    virtual void unpin() = 0;

    /// Returns the ID of this scope.
    virtual Scope_id get_id() const = 0;

    /// Returns the name of this scope (or the empty string for unnamed scopes).
    virtual const std::string& get_name() const = 0;

    /// Returns the direct parent of this scope (or \c nullptr for the global scope).
    ///
    /// \return   The parent scope. RCS:NEU
    virtual Scope* get_parent() const = 0;

    /// Returns the privacy level of this scope.
    virtual Privacy_level get_level() const = 0;

    /// Creates a new scope (or retrieves an already existing named scope).
    ///
    /// This may involve network operations and thus may take a while. The call will not return
    /// before the scope is created.
    ///
    /// The created scope is either temporary or not temporary. A temporary scope will be removed
    /// when the host which created the scope is removed from the cluster. A non-temporary scope
    /// will not be automatically removed in such a situation. Note that all child scopes of a
    /// temporary scope also need to be temporary.
    ///
    /// \param level          Privacy level for the new scope. Must be higher than the privacy level
    ///                       of the parent scope.
    /// \param is_temporary   Flag for temporary scopes.
    /// \param name           The name of the scope. The empty string creates an unnamed scope.
    /// \return               The created child scope, or \c nullptr in case of failure:
    ///                       - Privacy level not higher than that of the parent scope.
    ///                       - A scope with that name exists already, but with different parent
    ///                         scope and/or privacy level.
    ///                       RCS:NEU
    virtual Scope* create_child(
        Privacy_level level, bool is_temporary = false, const std::string& name = "") = 0;

    /// Creates a new transaction associated with this scope.
    ///
    /// This may involve network operations and thus may take a while. The call will not return
    /// before the transaction is created.
    ///
    /// \return   The created transaction. RCS:NEU
    virtual Transaction* start_transaction() = 0;

    /// Returns a list of database changes since some point in time.
    ///
    /// The start of the query range is given by the pair of \p last_transaction_id and
    /// \p last_transaction_change_version (which usually matches the last call of this function,
    /// hence the parameter names). The end of the query range is implicitly defined by what is
    /// visible to \p current_transaction_id.
    ///
    /// \note This variant does not include changes from a currently open transaction, in contrast
    ///       to #DB::ITransaction::get_journal().
    ///
    /// \note It is recommended to use #DB::ITransaction::get_journal() instead of this method.
    ///
    /// \param last_transaction_id               The transaction ID of start of the query range.
    /// \param last_transaction_change_version   The sequence number of the start of the query
    ///                                          range.
    /// \param current_transaction_id            Only changes visible to this transaction are
    ///                                          reported (excluding that transaction itself).
    /// \param journal_type                      A filter for the journal type. Only changes with
    ///                                          a non-zero intersection with \p journal_type will
    ///                                          be reported. Use #JOURNAL_ALL to not filter based
    ///                                          on the journal type.
    /// \param lookup_parents                    Indicates whether parent scopes should be
    ///                                          considered, too.
    /// \return                                  A vector of tag/journal type pairs describing the
    ///                                          changes in the relevant range matching the filter.
    ///                                          Returns \c nullptr if the queried range is too
    ///                                          large for the journal capacity. In this case the
    ///                                          caller should assume that all relevant database
    ///                                          elements have changed. \n
    ///                                          The same tag might occur multiple times, with
    ///                                          identical or different journal types.
    virtual std::unique_ptr<Journal_query_result> get_journal(
        Transaction_id last_transaction_id,
        mi::Uint32 last_transaction_change_version,
        DB::Transaction_id current_transaction_id,
        Journal_type journal_type,
        bool lookup_parents) = 0;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_SCOPE_H
