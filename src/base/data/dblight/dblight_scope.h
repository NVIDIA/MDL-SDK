/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

namespace MI {

namespace DBLIGHT {

class Database_impl;

/// A scope of the database.
///
/// Currently, there is only scope per database, the global scope.
///
/// "NI" means DBLIGHT does not implement/support that method of the interface.
class Scope_impl : public DB::Scope
{
public:
    /// Constructor.
    ///
    /// \param database Instance of the database this scope belongs to.
    Scope_impl( Database_impl* database) : m_database( database) { }

    /// Destructor.
    virtual ~Scope_impl() { }

    // methods of DB::Scope

    void pin() override { ++m_pin_count; }

    /// Invokes the destructor if the pin count drops to zero.
    void unpin() override { if( --m_pin_count == 0) delete this; }

    DB::Scope_id get_id() override { return 0; }

    const std::string& get_name() const override { return m_name; }

    DB::Scope* get_parent() override { return nullptr; }

    DB::Privacy_level get_level() override { return 0; }

    /*NI*/ DB::Scope* create_child(
        DB::Privacy_level level, bool is_temporary, const std::string& name) override
    { return nullptr; }

    DB::Transaction* start_transaction() override;

private:
    /// The database instance this scope belongs to.
    Database_impl* const m_database;

    /// The name of the scope.
    const std::string m_name;
    /// Reference count of the scope.
    std::atomic_uint32_t m_pin_count = 1;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_SCOPE_H
