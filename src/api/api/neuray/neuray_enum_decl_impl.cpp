/***************************************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IEnum_decl implementation.
 **/

#include "pch.h"

#include "neuray_enum_decl_impl.h"

#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Enum_decl_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Enum_decl_impl();
}

Enum_decl_impl::Enum_decl_impl()
{
}

Enum_decl_impl::~Enum_decl_impl()
{
}

mi::Sint32 Enum_decl_impl::add_enumerator( const char* name, mi::Sint32 value)
{
    if( !name)
        return -1;

    for( mi::Size i = 0; i < m_name.size(); ++i)
        if( m_name[i] == name)
            return -2;

    m_name.push_back( name);
    m_value.push_back( value);
    return 0;
}

mi::Sint32 Enum_decl_impl::remove_enumerator( const char* name)
{
    if( !name)
        return -1;

    for( mi::Size i = 0; i < m_name.size(); ++i)
        if( m_name[i] == name) {
            m_name.erase( m_name.begin() + i);
            m_value.erase( m_value.begin() + i);
            return 0;
        }

    return -2;
}

mi::Size Enum_decl_impl::get_length() const
{
    return m_name.size();
}

const char* Enum_decl_impl::get_name( mi::Size index) const
{
    if( index >= m_name.size())
        return nullptr;

    return m_name[index].c_str();
}

mi::Sint32 Enum_decl_impl::get_value( mi::Size index) const
{
    if( index >= m_value.size())
        return m_value[0];

    return m_value[index];
}

mi::Size Enum_decl_impl::get_index( const char* name) const
{
    if( !name)
    	return 0;

    for( mi::Size i = 0; i < m_name.size(); ++i)
        if( m_name[i] == name)
            return i;

    return static_cast<mi::Size>( -1);
}

mi::Size Enum_decl_impl::get_index( mi::Sint32 value) const
{
    for( mi::Size i = 0; i < m_value.size(); ++i)
        if( m_value[i] == value)
            return i;

    return static_cast<mi::Size>( -1);
}

const char* Enum_decl_impl::get_enum_type_name() const
{
    if( m_enum_type_name.empty())
        return nullptr;

    return m_enum_type_name.c_str();
}

void Enum_decl_impl::set_enum_type_name( const char* enum_type_name)
{
    ASSERT( M_NEURAY_API, enum_type_name);

    m_enum_type_name = enum_type_name;
}

} // namespace NEURAY

} // namespace MI
