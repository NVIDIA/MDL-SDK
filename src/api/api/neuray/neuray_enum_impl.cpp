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
 ** \brief Source for the IEnum implementation.
 **/

#include "pch.h"

#include "neuray_enum_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/istring.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Enum_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;
    mi::base::Handle<const mi::IEnum_decl> ienum_decl(
        argv[0]->get_interface<mi::IEnum_decl>());
    if( !ienum_decl.is_valid_interface())
        return nullptr;
    mi::base::Handle<const mi::IString> itype_name( argv[1]->get_interface<mi::IString>());
    if( !itype_name.is_valid_interface())
        return nullptr;
    const char* type_name = itype_name->get_c_str();

    return new Enum_impl( ienum_decl.get(), type_name);
}

Enum_impl::Enum_impl( const mi::IEnum_decl* enum_decl, const char* type_name)
  : m_storage( 0)
{
    m_enum_decl = mi::base::Handle<const mi::IEnum_decl>( enum_decl, mi::base::DUP_INTERFACE);
    m_type_name = type_name;
}

Enum_impl::~Enum_impl()
{
    m_enum_decl = nullptr;
}

const char* Enum_impl::get_type_name() const
{
    return m_type_name.c_str();
}

void Enum_impl::get_value( mi::Sint32& value) const
{
    value = m_enum_decl->get_value( m_storage);
}

const char* Enum_impl::get_value_by_name() const
{
    return m_enum_decl->get_name( m_storage);
}

mi::Sint32 Enum_impl::set_value( mi::Sint32 value)
{
    mi::Size index = m_enum_decl->get_index( value);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    m_storage = static_cast<mi::Uint32>( index);
    return 0;
}

mi::Sint32 Enum_impl::set_value_by_name( const char* name)
{
    mi::Size index = m_enum_decl->get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    m_storage = static_cast<mi::Uint32>( index);
    return 0;
}

const mi::IEnum_decl* Enum_impl::get_enum_decl() const
{
    m_enum_decl->retain();
    return m_enum_decl.get();
}


mi::base::IInterface* Enum_impl_proxy::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;
    mi::base::Handle<const mi::IEnum_decl> ienum_decl(
        argv[0]->get_interface<mi::IEnum_decl>());
    if( !ienum_decl.is_valid_interface())
        return nullptr;
    mi::base::Handle<const mi::IString> itype_name( argv[1]->get_interface<mi::IString>());
    if( !itype_name.is_valid_interface())
        return nullptr;
    const char* type_name = itype_name->get_c_str();
    return (new Enum_impl_proxy( ienum_decl.get(), type_name))->cast_to_major();
}

Enum_impl_proxy::Enum_impl_proxy( const mi::IEnum_decl* enum_decl, const char* type_name)
{
    m_enum_decl = mi::base::Handle<const mi::IEnum_decl>( enum_decl, mi::base::DUP_INTERFACE);
    m_type_name = type_name;
    m_pointer = nullptr;
}

Enum_impl_proxy::~Enum_impl_proxy()
{
    m_enum_decl = nullptr;
}

const char* Enum_impl_proxy::get_type_name() const
{
    return m_type_name.c_str();
}

void Enum_impl_proxy::get_value( mi::Sint32& value) const
{
    value = m_enum_decl->get_value( *m_pointer);
}

const char* Enum_impl_proxy::get_value_by_name() const
{
    return m_enum_decl->get_name( *m_pointer);
}

mi::Sint32 Enum_impl_proxy::set_value( mi::Sint32 value)
{
    mi::Size index = m_enum_decl->get_index( value);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    *m_pointer = static_cast<mi::Uint32>( index);
    return 0;
}

mi::Sint32 Enum_impl_proxy::set_value_by_name( const char* name)
{
    mi::Size index = m_enum_decl->get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    *m_pointer = static_cast<mi::Uint32>( index);
    return 0;
}

const mi::IEnum_decl* Enum_impl_proxy::get_enum_decl() const
{
    m_enum_decl->retain();
    return m_enum_decl.get();
}

void Enum_impl_proxy::set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<mi::Uint32*>( pointer);
    m_owner = mi::base::Handle<const mi::base::IInterface>( owner, mi::base::DUP_INTERFACE);
}

void Enum_impl_proxy::release_referenced_memory()
{
    // nothing to do
}


} // namespace NEURAY

} // namespace MI

