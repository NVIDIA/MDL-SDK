/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file i_db_scope.h
 ** \brief This declares the database scope class.
 **
 ** This file contains the pure virtual base class for the database scope class.
 **/

#ifndef BASE_DATA_DB_I_DB_SCOPE_H
#define BASE_DATA_DB_I_DB_SCOPE_H

#include <base/system/main/types.h>
#include <string>

namespace MI
{

namespace DB
{

class Transaction;

/// The privacy level defines to which level changes to an element go.
typedef Uint8 Privacy_level;

/// Each scope (see below) is identified by a globally unique scope id
typedef Uint32 Scope_id;

/// A scope limits visibility of changes to transactions which live within the same scope. Scopes
/// may be nested and each inner scope sees all changes from all its ancestor scopes (but not vice
/// versa). Scopes are identified with a network unique scope id. Scopes must be accessible from
/// within the whole network. The information about a scope which is needed is which child scopes it
/// has and what is its parent scope. Thus the creation and destruction of scopes must be
/// distributed to the whole network. Each time a new scope is created, the scope id and the id of
/// the parent scope is sent to all hosts. Each host builds up a tree of scopes which it will use to
/// resolve tags.
class Scope
{
  public:
    /// Pin the scope incrementing its reference count
    virtual void pin() = 0;

    /// Unpin the scope decrementing its reference count. When the application holds no more
    /// reference, the database may decide to destroy the scope. Although the application may no
    /// longer use the scope it may actually live longer. This might be the case if there are still
    /// transactions taking place in the scope. It might not be possible to abort them at once. Note
    /// that the application should abort or commit all transactions from this scope before
    /// releasing the last reference to it.
    virtual void unpin() = 0;

    /// Create a new scope which is a child of this scope. This may involve network operations and
    /// thus may take a while. The call will not return before the scope is created.
    ///
    /// The created scope is either temporary or not temporary. A temporary scope will be removed
    /// when the host which created the scope is removed from the cluster. A scope which is not
    /// temporary will not be removed when the host which created the scope is removed from
    /// the cluster.
    ///
    /// The final argument to this method is a bool that indicates if the scope created as a result
    /// of calling this method is temporary or not temporary. Note that a scope that is not
    /// temporary can not be created as the child of a scope that is temporary.
    ///
    /// \param level                    Privacy level for the new scope.
    /// \param is_temporary             A bool indicating if the new scope is temporary
    /// \param name                     The name of the scope. If empty the scope is unnamed.
    /// \return                         The created child scope. Can be NULL, if creation failed.
    virtual Scope *create_child(
        Privacy_level level,
        bool is_temporary = false,
        const std::string& name = "")= 0;

    /// Get the id of a scope. This needs to be in the interface because for some applications the
    /// id of a scope needs to be stored in database elements.
    ///
    /// \return                         The scope id of this scope.
    virtual Scope_id get_id() = 0;

    /// Get the name of a scope. An unnamed scope returns the empty string.
    ///
    /// \return                         The name of the scope.
    virtual const std::string& get_name() const = 0;

    /// Get the direct parent of this scope
    ///
    /// \return                         The parent of this scope.
    virtual Scope* get_parent() = 0;

    /// Get the privacy level of the scope
    ///
    /// \return                         The privacy level of this scope.
    virtual Privacy_level get_level() = 0;

    /// Start a transaction. This is part of the scope interface because each transaction belongs to
    /// a scope. The database will ensure that the scope is known to all hosts which will take part
    /// in transactions created within this scope. This may involve network operations and thus may
    /// take a while. The call will not return before the transaction is created.
    ///
    /// \return                         The created transaction
    virtual Transaction *start_transaction() = 0;

  protected:
    /// The destructor is private because only the database may delete scopes when they are no
    /// longer needed.
    virtual ~Scope() { }
};

} // namespace DB

} // namespace MI

#endif
