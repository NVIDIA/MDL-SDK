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

/** \file
 ** \brief Implementation of IScope
 **
 ** Implementation of IScope
 **/

#include "pch.h"

#include "neuray_scope_impl.h"

#include "neuray_transaction_impl.h"

#include <base/lib/log/i_log_logger.h>

#include <sstream>

namespace MI {

namespace NEURAY {

Scope_impl::Scope_impl( DB::Scope* scope, const Class_factory* class_factory)
  : m_scope( scope),
    m_class_factory( class_factory)
{
    m_scope->pin();

    m_name = m_scope->get_name();

    DB::Scope_id id = scope->get_id();
    Uint32 id_as_uint = static_cast<Uint32>( id);
    std::ostringstream stream;
    stream << id_as_uint;
    m_id = stream.str();

    // The string cannot be empty at this point. If it still is, then this indicates a misbehaving
    // libstdc++ runtime. This was seen on Linux with devsl builds (non-static libstdc++ runtime)
    // when Python/Java bindings are used. The symbol binding/static initialization seems to be
    // affected and therefore the behavior of ostream<< (locale facet for num_put is NULL).
    // Workaround: Use a static build.
    if (m_id.empty())
    {
        ASSERT( M_NEURAY_API, false);
        LOG::mod_log->fatal( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "String stream returned empty string when converting scope id %u. "
            "Please use static build of "
            "libneuray"
            ".",
            id_as_uint);
    }

    ASSERT( M_NEURAY_API, id == 0);
}

Scope_impl::~Scope_impl()
{
    m_scope->unpin();
    m_class_factory = nullptr;
}

mi::neuraylib::ITransaction* Scope_impl::create_transaction()
{
    DB::Transaction* db_transaction = m_scope->start_transaction();
    if( !db_transaction)
        return nullptr;

    return new Transaction_impl( db_transaction, m_class_factory);
}

const char* Scope_impl::get_id() const
{
    return m_id.c_str();
}

mi::Uint8 Scope_impl::get_privacy_level() const
{
    return m_scope->get_level();
}

DB::Scope* Scope_impl::get_scope() const
{
    return m_scope;
}

const char* Scope_impl::get_name() const
{
    return m_name.empty() ? nullptr : m_name.c_str();
}

mi::neuraylib::IScope* Scope_impl::get_parent() const
{
    DB::Scope* parent_db_scope = m_scope->get_parent();
    if( !parent_db_scope)
        return nullptr;

    return new Scope_impl( parent_db_scope, m_class_factory);
}

} // namespace NEURAY

} // namespace MI

