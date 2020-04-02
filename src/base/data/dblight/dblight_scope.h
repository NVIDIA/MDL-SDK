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
 ** \brief This declares the database scope class.
 **
 ** This file contains the pure base class for the database scope class.
 **/

#ifndef BASE_DATA_DBLIGHT_SCOPE_H
#define BASE_DATA_DBLIGHT_SCOPE_H

#include <mi/base/atom.h>
#include <base/data/db/i_db_scope.h>

namespace MI {

namespace DBLIGHT {

class Database_impl;

class Scope_impl : public DB::Scope
{
public:
    Scope_impl(Database_impl* database);
    ~Scope_impl();
    void pin();
    void unpin();
    DB::Scope* create_child(
        DB::Privacy_level level, bool is_temporary, const std::string& name);
    DB::Scope_id get_id();
    const std::string& get_name() const;
    DB::Scope* get_parent();
    DB::Privacy_level get_level();
    DB::Transaction* start_transaction();

    /// Increments #m_transaction_count.
    Uint32 increment_transaction_count() { return ++m_transaction_count; }
    /// Decrements #m_transaction_count.
    void decrement_transaction_count() { --m_transaction_count; }

private:
    Database_impl* m_database;
    std::string m_name;
    mi::base::Atom32 m_refcount;

    // The current (or last) transaction.
    DB::Transaction* m_transaction;

    // Number of transactions started and not yet committed in this scope. Used to limit the number
    // of such transaction1 to 1.
    mi::base::Atom32 m_transaction_count;
};

} // namespace DB

} // namespace MI

#endif
