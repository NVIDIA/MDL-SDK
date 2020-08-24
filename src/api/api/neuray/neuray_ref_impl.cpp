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
 ** \brief Source for the IRef implementation.
 **/

#include "pch.h"

#include "neuray_ref_impl.h"

#include "i_neuray_attribute_context.h"
#include "i_neuray_db_element.h"
#include "neuray_transaction_impl.h"

#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_access.h>

#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Ref_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new Ref_impl( transaction);
}

Ref_impl::Ref_impl( mi::neuraylib::ITransaction* transaction)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( static_cast<Transaction_impl*>( transaction));
}

const char* Ref_impl::get_type_name() const
{
    return "Ref";
}

mi::Sint32 Ref_impl::set_reference( const IInterface* interface)
{
    if( !interface) {
        m_storage = DB::Tag();
        return 0;
    }

    mi::base::Handle<const IDb_element> db_element( interface->get_interface<IDb_element>());
    if( !db_element.is_valid_interface())
        return -2;

    DB::Tag tag = db_element->get_tag();
    if( !tag.is_valid())
        return -3;

    m_storage = tag;
    return 0;
}

mi::Sint32 Ref_impl::set_reference( const char* name)
{
    if( !name) {
        m_storage = DB::Tag();
        return 0;
    }

    DB::Tag tag = m_transaction->get_db_transaction()->name_to_tag( name);
    if( !tag.is_valid())
        return -2;

    m_storage = tag;
    return 0;
}

const mi::base::IInterface* Ref_impl::get_reference() const
{
    if( !m_storage.is_valid())
        return nullptr;

    return m_transaction->access( m_storage);
}

mi::base::IInterface* Ref_impl::get_reference()
{
    if( !m_storage.is_valid())
        return nullptr;

    return m_transaction->edit( m_storage);
}

const char* Ref_impl::get_reference_name() const
{
    if( !m_storage.is_valid())
        return nullptr;

    const char* name = m_transaction->get_db_transaction()->tag_to_name( m_storage);
    return name;
}

mi::neuraylib::ITransaction* Ref_impl::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

mi::base::IInterface* Ref_impl_proxy::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Ref_impl_proxy( transaction))->cast_to_major();
}

Ref_impl_proxy::Ref_impl_proxy( mi::neuraylib::ITransaction* transaction)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( static_cast<Transaction_impl*>( transaction));
    m_pointer = nullptr;
}

const char* Ref_impl_proxy::get_type_name() const
{
    return "Ref";
}

mi::Sint32 Ref_impl_proxy::set_reference( const IInterface* interface)
{
    if( !interface) {
        *m_pointer = DB::Tag();
        return 0;
    }

    mi::base::Handle<const IDb_element> db_element( interface->get_interface<IDb_element>());
    if( !db_element.is_valid_interface())
        return -2;

    DB::Tag tag = db_element->get_tag();
    if( !tag.is_valid())
        return -3;

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    mi::base::Handle<const IDb_element> owner_db_element(
        attribute_context ? attribute_context->get_db_element() : nullptr);
    if( !owner_db_element->can_reference_tag( tag))
        return -4;

    *m_pointer = tag;
    return 0;
}

mi::Sint32 Ref_impl_proxy::set_reference( const char* name)
{
    if( !name) {
        *m_pointer = DB::Tag();
        return 0;
    }

    DB::Tag tag = m_transaction->get_db_transaction()->name_to_tag( name);
    if( !tag.is_valid())
        return -2;

    mi::base::Handle<const IAttribute_context> attribute_context(
        m_owner->get_interface<IAttribute_context>());
    mi::base::Handle<const IDb_element> owner_db_element(
        attribute_context ? attribute_context->get_db_element() : nullptr);
    if( !owner_db_element->can_reference_tag( tag))
        return -4;

    *m_pointer = tag;
    return 0;
}

const mi::base::IInterface* Ref_impl_proxy::get_reference() const
{
    if( !m_pointer->is_valid())
        return nullptr;

    return m_transaction->access( *m_pointer);
}

mi::base::IInterface* Ref_impl_proxy::get_reference()
{
    if( !m_pointer->is_valid())
        return nullptr;

    return m_transaction->edit( *m_pointer);
}

const char* Ref_impl_proxy::get_reference_name() const
{
    if( !m_pointer->is_valid())
        return nullptr;

    const char* name = m_transaction->get_db_transaction()->tag_to_name( *m_pointer);
    return name;
}

void Ref_impl_proxy::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<DB::Tag*>( pointer);
    m_owner = make_handle_dup( owner);
}

void Ref_impl_proxy::release_referenced_memory()
{
    // nothing to do
}

mi::neuraylib::ITransaction* Ref_impl_proxy::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

} // namespace NEURAY

} // namespace MI

