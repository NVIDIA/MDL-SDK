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
 ** \brief Source for the Pointer_impl and Const_pointer_impl implementation.
 **/

#include "pch.h"

#include "neuray_class_factory.h"
#include "neuray_pointer_impl.h"
#include "neuray_transaction_impl.h"

#include <mi/neuraylib/istring.h>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Pointer_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* value_type_name = istring->get_c_str();

    Pointer_impl* pointer = new Pointer_impl( transaction, value_type_name);
    if( !pointer->successfully_constructed()) {
        pointer->release();
        return nullptr;
    } else
        return pointer;
}

Pointer_impl::Pointer_impl( mi::neuraylib::ITransaction* transaction, const char* value_type_name)
{
    // transaction might be NULL
    m_transaction = make_handle_dup( transaction);

    ASSERT( M_NEURAY_API, value_type_name);
    m_value_type_name = value_type_name;

    m_type_name = "Pointer<" + m_value_type_name + ">";

    std::string mangled_value_type_name
        = (m_value_type_name == "Interface") ? "Void" : m_value_type_name;
    mi::base::Handle<mi::base::IInterface> element( s_class_factory->create_type_instance(
        static_cast<Transaction_impl*>( transaction), mangled_value_type_name.c_str(), 0, nullptr));
    m_successfully_constructed = element.is_valid_interface();
}

const char* Pointer_impl::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Sint32 Pointer_impl::set_pointer( mi::base::IInterface* pointer)
{
    if( !has_correct_value_type( pointer))
        return -1;

    m_pointer = pointer;
    if( m_pointer)
        m_pointer->retain();
    return 0;
}

mi::base::IInterface* Pointer_impl::get_pointer() const
{
    mi::base::IInterface* pointer = m_pointer.get();
    if( pointer)
        pointer->retain();
    return pointer;
}

mi::neuraylib::ITransaction* Pointer_impl::get_transaction() const
{
    if( m_transaction.is_valid_interface())
        m_transaction->retain();
    return m_transaction.get();
}

bool Pointer_impl::has_correct_value_type( const mi::base::IInterface* value) const
{
    if( !value)
        return true;

    if( m_value_type_name == "Interface")
        return true;

    mi::base::Handle<const mi::IData> data( value->get_interface<mi::IData>());
    if( !data.is_valid_interface())
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_value_type_name == type_name;
}

mi::base::IInterface* Const_pointer_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;
    mi::base::Handle<const mi::IString> istring( argv[0]->get_interface<mi::IString>());
    if( !istring.is_valid_interface())
        return nullptr;
    const char* value_type_name = istring->get_c_str();

    Const_pointer_impl* const_pointer
        = new Const_pointer_impl( transaction, value_type_name);
    if( !const_pointer->successfully_constructed()) {
        const_pointer->release();
        return nullptr;
    } else
        return const_pointer;
}

Const_pointer_impl::Const_pointer_impl(
    mi::neuraylib::ITransaction* transaction, const char* value_type_name)
{
    // transaction might be NULL
    m_transaction = make_handle_dup( transaction);

    ASSERT( M_NEURAY_API, value_type_name);
    m_value_type_name = value_type_name;

    m_type_name = "Const_pointer<" + m_value_type_name + ">";

    std::string mangled_value_type_name
        = (m_value_type_name == "Interface") ? "Void" : m_value_type_name;
    mi::base::Handle<mi::base::IInterface> element( s_class_factory->create_type_instance(
        static_cast<Transaction_impl*>( transaction), mangled_value_type_name.c_str(), 0, nullptr));
    m_successfully_constructed = element.is_valid_interface();
}

const char* Const_pointer_impl::get_type_name() const
{
    return m_type_name.c_str();
}

mi::Sint32 Const_pointer_impl::set_pointer( const mi::base::IInterface* pointer)
{
    if( !has_correct_value_type( pointer))
        return -1;

    m_pointer = pointer;
    if( m_pointer)
        m_pointer->retain();
    return 0;
}

const mi::base::IInterface* Const_pointer_impl::get_pointer() const
{
    const mi::base::IInterface* pointer = m_pointer.get();
    if( pointer)
        pointer->retain();
    return pointer;
}

mi::neuraylib::ITransaction* Const_pointer_impl::get_transaction() const
{
    if( m_transaction.is_valid_interface())
        m_transaction->retain();
    return m_transaction.get();
}

bool Const_pointer_impl::has_correct_value_type( const mi::base::IInterface* value) const
{
    if( !value)
        return true;

    if( m_value_type_name == "Interface")
        return true;

    mi::base::Handle<const mi::IData> data( value->get_interface<mi::IData>());
    if( !data.is_valid_interface())
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_value_type_name == type_name;
}

} // namespace NEURAY

} // namespace MI
