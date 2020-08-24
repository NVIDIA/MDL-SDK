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
 ** \brief Implementation of IDatabase
 **
 ** Implements the IDatabase interface
 **/

#include "pch.h"

#include "neuray_database_impl.h"

#include "neuray_class_factory.h"
#include "neuray_scope_impl.h"

#include <sstream>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/data/db/i_db_database.h>

namespace MI {

namespace NEURAY {

Database_impl::Database_impl( mi::neuraylib::INeuray::Status& status)
  : m_status( status),
    m_database( nullptr)
{
}

Database_impl::~Database_impl()
{
}

mi::neuraylib::IScope* Database_impl::get_global_scope() const
{
    if( m_status != mi::neuraylib::INeuray::STARTED)
        return nullptr;

    DB::Scope* scope = m_database->get_global_scope();
    if( !scope)
        return nullptr;

    return new Scope_impl( scope, s_class_factory);
}

mi::neuraylib::IScope* Database_impl::create_scope(
    mi::neuraylib::IScope* parent,
    mi::Uint8 privacy_level,
    bool temp)
{
    if( m_status != mi::neuraylib::INeuray::STARTED)
        return nullptr;

    if( !parent) {
        // a recursive call here simplifies resource management, alternative: use handles
        parent = get_global_scope();
        if( !parent)
            return nullptr;
        mi::neuraylib::IScope* child = create_scope( parent, privacy_level, temp);
        parent->release();
        return child;
    }

    if( privacy_level == 0)
        privacy_level = parent->get_privacy_level() + 1;
    if( privacy_level <= parent->get_privacy_level())
        return nullptr;

    Scope_impl* parent_scope_impl = static_cast<Scope_impl *>( parent);
    DB::Scope* parent_db_scope = parent_scope_impl->get_scope();
    DB::Scope* child_db_scope = parent_db_scope->create_child( privacy_level, temp, "");
    if( !child_db_scope)
        return nullptr;

    return new Scope_impl( child_db_scope, s_class_factory);
}

mi::neuraylib::IScope* Database_impl::create_or_get_named_scope(
    const char* name,
    mi::neuraylib::IScope* parent,
    mi::Uint8 privacy_level)
{
    if( !name)
        return nullptr;
    if( m_status != mi::neuraylib::INeuray::STARTED)
        return nullptr;

    if( !parent){
        // a recursive call here simplifies resource management, alternative: use handles
        parent = get_global_scope();
        if( !parent)
            return nullptr;
        mi::neuraylib::IScope* child = create_or_get_named_scope( name, parent, privacy_level);
        parent->release();
        return child;
    }

    if( privacy_level == 0)
        privacy_level = parent->get_privacy_level() + 1;
    if( privacy_level <= parent->get_privacy_level())
        return nullptr;

    Scope_impl* parent_scope_impl = static_cast<Scope_impl *>( parent);
    DB::Scope* parent_db_scope = parent_scope_impl->get_scope();
    DB::Scope* child_db_scope = parent_db_scope->create_child( privacy_level, false, name);
    if( !child_db_scope)
        return nullptr;

    return new Scope_impl( child_db_scope, s_class_factory);
}

mi::neuraylib::IScope* Database_impl::get_scope( const char* id_string) const
{
    if( !id_string)
        return nullptr;
    if( m_status != mi::neuraylib::INeuray::STARTED)
        return nullptr;

    STLEXT::Likely<mi::Uint32> id_likely = STRING::lexicographic_cast_s<mi::Uint32>( id_string);
    if( !id_likely.get_status())
        return nullptr;
    mi::Uint32 id = *id_likely.get_ptr(); //-V522 PVS
    DB::Scope_id scope_id( id);

    DB::Scope* scope = m_database->lookup_scope( scope_id);
    if(! scope)
        return nullptr;

    return new Scope_impl( scope, s_class_factory);
}

mi::neuraylib::IScope* Database_impl::get_named_scope( const char* name) const
{
    if( !name)
        return nullptr;
    if( m_status != mi::neuraylib::INeuray::STARTED)
        return nullptr;

    DB::Scope* scope = m_database->lookup_scope( name);
    if(! scope)
        return nullptr;

    return new Scope_impl( scope, s_class_factory);
}

mi::Sint32 Database_impl::remove_scope( const char* id_string) const
{
    if( !id_string)
        return -1;
    if( m_status != mi::neuraylib::INeuray::STARTED)
        return -1;

    STLEXT::Likely<mi::Uint32> id_likely = STRING::lexicographic_cast_s<mi::Uint32>( id_string);
    if( !id_likely.get_status())
        return -1;
    mi::Uint32 id = *id_likely.get_ptr(); //-V522 PVS
    if( id == 0)
        return -1;

    DB::Scope_id scope_id( id);
    return m_database->remove( scope_id) ? 0 : -1;
}

void Database_impl::lock( mi::Uint32 lock_id)
{
    m_database->lock( DB::Tag( lock_id));
}

mi::Sint32 Database_impl::unlock( mi::Uint32 lock_id)
{
    return m_database->unlock( DB::Tag( lock_id)) ? 0 : -1;
}

void Database_impl::garbage_collection()
{
    m_database->garbage_collection();
}

mi::Sint32 Database_impl::start( DB::Database* database)
{
    if( !database)
        return -1;

    m_database = database;
    return 0;
}

mi::Sint32 Database_impl::shutdown()
{
    m_database = nullptr;
    return 0;
}

} // namespace NEURAY

} // namespace MI
