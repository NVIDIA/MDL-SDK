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
 ** \brief Source for the IString implementation.
 **/

#include "pch.h"

#include "neuray_string_impl.h"

#include <cstring>

namespace MI {

namespace NEURAY {

mi::base::IInterface* String_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new String_impl();
}

String_impl::String_impl(const char* const str)
  : m_storage( nullptr)
{
    set_c_str( str);
}

String_impl::~String_impl()
{
    delete[] m_storage;
}

const char* String_impl::get_type_name() const
{
    return "String";
}
    
const char* String_impl::get_c_str() const
{
    return m_storage;
}

void String_impl::set_c_str( const char* str)
{
    // Defer deallocation of m_storage, str might be identical to m_storage.
    if( !str)
        str = "";
    mi::Size len = strlen( str);
    char* copy = new char[len+1];
    strncpy( copy, str, len+1);
    delete[] m_storage;
    m_storage = copy;
}

mi::base::IInterface* String_impl_proxy::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new String_impl_proxy())->cast_to_major();
}

String_impl_proxy::String_impl_proxy()
{
    m_pointer = nullptr;
}

const char* String_impl_proxy::get_type_name() const
{
    return "String";
}

const char* String_impl_proxy::get_c_str() const
{
    // Note that the constructor of the proxy implementation does not call set_c_str( "").
    // Hence, we have to check for NULL pointers here.
    if( *m_pointer == nullptr)
        return "";
    return *m_pointer;
}

void String_impl_proxy::set_c_str( const char* str)
{
    // Defer deallocation of *m_pointer, str might be identical to *m_pointer.
    if( !str)
        str = "";
    mi::Size len = strlen( str);
    char* copy = new char[len+1];
    strncpy( copy, str, len+1);
    delete[] *m_pointer;
    *m_pointer = copy;
}

void String_impl_proxy::set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<const char**>( pointer);
    m_owner = make_handle_dup( owner);
}

void String_impl_proxy::release_referenced_memory()
{
    delete[] *m_pointer;
    *m_pointer = nullptr;
}

} // namespace NEURAY

} // namespace MI
