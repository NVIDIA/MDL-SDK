/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IStructure_decl implementation.
 **/

#include "pch.h"

#include "neuray_structure_decl_impl.h"

#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Structure_decl_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Structure_decl_impl();
}

Structure_decl_impl::Structure_decl_impl()
{
}

Structure_decl_impl::~Structure_decl_impl()
{
}

mi::Sint32 Structure_decl_impl::add_member( const char* type_name, const char* name)
{
    if( !type_name || !name)
        return -1;

    for( mi::Size i = 0; i < m_name.size(); ++i)
        if( m_name[i] == name)
            return -2;

    m_type_name.push_back( type_name);
    m_name.push_back( name);
    return 0;
}

mi::Sint32 Structure_decl_impl::remove_member( const char* name)
{
    if( !name)
        return -1;

    for( mi::Size i = 0; i < m_name.size(); ++i)
        if( m_name[i] == name) {
            m_type_name.erase( m_type_name.begin() + i);
            m_name.erase( m_name.begin() + i);
            return 0;
        }

    return -2;
}

mi::Size Structure_decl_impl::get_length() const
{
    return m_name.size();
}

const char* Structure_decl_impl::get_member_type_name( mi::Size index) const
{
    if( index >= m_type_name.size())
        return nullptr;

    return m_type_name[index].c_str();
}

const char* Structure_decl_impl::get_member_type_name( const char* name) const
{
    if( !name)
    	return nullptr;

    for( mi::Size i = 0; i < m_name.size(); ++i)
        if( m_name[i] == name)
            return m_type_name[i].c_str();

    return nullptr;
}

const char* Structure_decl_impl::get_member_name( mi::Size index) const
{
    if( index >= m_name.size())
        return nullptr;

    return m_name[index].c_str();
}

const char* Structure_decl_impl::get_structure_type_name() const
{
    if( m_structure_type_name.empty())
        return nullptr;

    return m_structure_type_name.c_str();
}

void Structure_decl_impl::set_structure_type_name( const char* structure_type_name)
{
    ASSERT( M_NEURAY_API, structure_type_name);

    m_structure_type_name = structure_type_name;
}

} // namespace NEURAY

} // namespace MI
