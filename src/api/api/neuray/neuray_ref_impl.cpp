/***************************************************************************************************
 * Copyright (c) 2010-2018, NVIDIA CORPORATION. All rights reserved.
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

template <Ref_type T>
mi::base::IInterface* Ref_impl<T>::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return 0;
    if( argc != 0)
        return 0;
    return new Ref_impl( transaction);
}

template <Ref_type T>
Ref_impl<T>::Ref_impl( mi::neuraylib::ITransaction* transaction)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( static_cast<Transaction_impl*>( transaction));
}

template <>
const char* Ref_impl<REF_UNTYPED>::get_type_name() const
{
    return "Ref";
}


template <>
const char* Ref_impl<REF_TEXTURE>::get_type_name() const
{
    return "Ref<Texture>";
}

template <>
const char* Ref_impl<REF_LIGHTPROFILE>::get_type_name() const
{
    return "Ref<Lightprofile>";
}

template <>
const char* Ref_impl<REF_BSDF_MEASUREMENT>::get_type_name() const
{
    return "Ref<Bsdf_measurement>";
}


template <Ref_type T>
mi::Sint32 Ref_impl<T>::set_reference( const IInterface* interface)
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

    if( !is_valid_reference_type( tag))
        return -5;

    m_storage = tag;
    return 0;
}

template <Ref_type T>
mi::Sint32 Ref_impl<T>::set_reference( const char* name)
{
    if( !name) {
        m_storage = DB::Tag();
        return 0;
    }

    DB::Tag tag = m_transaction->get_db_transaction()->name_to_tag( name);
    if( !tag.is_valid())
        return -2;

    if( !is_valid_reference_type( tag))
        return -5;

    m_storage = tag;
    return 0;
}

template <Ref_type T>
const mi::base::IInterface* Ref_impl<T>::get_reference() const
{
    if( !m_storage.is_valid())
        return 0;

    return m_transaction->access( m_storage);
}

template <Ref_type T>
mi::base::IInterface* Ref_impl<T>::get_reference()
{
    if( !m_storage.is_valid())
        return 0;

    return m_transaction->edit( m_storage);
}

template <Ref_type T>
const char* Ref_impl<T>::get_reference_name() const
{
    if( !m_storage.is_valid())
        return 0;

    const char* name = m_transaction->get_db_transaction()->tag_to_name( m_storage);
    return name;
}

template <Ref_type T>
mi::neuraylib::ITransaction* Ref_impl<T>::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

template <>
bool Ref_impl<REF_UNTYPED>::is_valid_reference_type( DB::Tag tag) const
{
    return true;
}


template <>
bool Ref_impl<REF_TEXTURE>::is_valid_reference_type( DB::Tag tag) const
{
    return m_transaction->get_db_transaction()->get_class_id( tag) == TEXTURE::ID_TEXTURE;
}

template <>
bool Ref_impl<REF_LIGHTPROFILE>::is_valid_reference_type( DB::Tag tag) const
{
    return m_transaction->get_db_transaction()->get_class_id( tag) == LIGHTPROFILE::ID_LIGHTPROFILE;
}

template <>
bool Ref_impl<REF_BSDF_MEASUREMENT>::is_valid_reference_type( DB::Tag tag) const
{
    return m_transaction->get_db_transaction()->get_class_id( tag) == BSDFM::ID_BSDF_MEASUREMENT;
}


template <Ref_type T>
mi::base::IInterface* Ref_impl_proxy<T>::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return 0;
    if( argc != 0)
        return 0;
    return (new Ref_impl_proxy( transaction))->cast_to_major();
}

template <Ref_type T>
Ref_impl_proxy<T>::Ref_impl_proxy( mi::neuraylib::ITransaction* transaction)
{
    ASSERT( M_NEURAY_API, transaction);
    m_transaction = make_handle_dup( static_cast<Transaction_impl*>( transaction));
    m_pointer = 0;
}

template <>
const char* Ref_impl_proxy<REF_UNTYPED>::get_type_name() const
{
    return "Ref";
}


template <>
const char* Ref_impl_proxy<REF_TEXTURE>::get_type_name() const
{
    return "Ref<Texture>";
}

template <>
const char* Ref_impl_proxy<REF_LIGHTPROFILE>::get_type_name() const
{
    return "Ref<Lightprofile>";
}

template <>
const char* Ref_impl_proxy<REF_BSDF_MEASUREMENT>::get_type_name() const
{
    return "Ref<Bsdf_measurement>";
}


template <Ref_type T>
mi::Sint32 Ref_impl_proxy<T>::set_reference( const IInterface* interface)
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
        attribute_context ? attribute_context->get_db_element() : 0);
    if( !owner_db_element->can_reference_tag( tag))
        return -4;

    if( !is_valid_reference_type( tag))
        return -5;

    *m_pointer = tag;
    return 0;
}

template <Ref_type T>
mi::Sint32 Ref_impl_proxy<T>::set_reference( const char* name)
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
        attribute_context ? attribute_context->get_db_element() : 0);
    if( !owner_db_element->can_reference_tag( tag))
        return -4;

    if( !is_valid_reference_type( tag))
        return -5;

    *m_pointer = tag;
    return 0;
}

template <Ref_type T>
const mi::base::IInterface* Ref_impl_proxy<T>::get_reference() const
{
    if( !m_pointer->is_valid())
        return 0;

    return m_transaction->access( *m_pointer);
}

template <Ref_type T>
mi::base::IInterface* Ref_impl_proxy<T>::get_reference()
{
    if( !m_pointer->is_valid())
        return 0;

    return m_transaction->edit( *m_pointer);
}

template <Ref_type T>
const char* Ref_impl_proxy<T>::get_reference_name() const
{
    if( !m_pointer->is_valid())
        return 0;

    const char* name = m_transaction->get_db_transaction()->tag_to_name( *m_pointer);
    return name;
}

template <Ref_type T>
void Ref_impl_proxy<T>::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<DB::Tag*>( pointer);
    m_owner = make_handle_dup( owner);
}

template <Ref_type T>
void Ref_impl_proxy<T>::release_referenced_memory()
{
    // nothing to do
}

template <Ref_type T>
mi::neuraylib::ITransaction* Ref_impl_proxy<T>::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

template <>
bool Ref_impl_proxy<REF_UNTYPED>::is_valid_reference_type( DB::Tag tag) const
{
    return true;
}


template <>
bool Ref_impl_proxy<REF_TEXTURE>::is_valid_reference_type( DB::Tag tag) const
{
    return m_transaction->get_db_transaction()->get_class_id( tag) == TEXTURE::ID_TEXTURE;
}

template <>
bool Ref_impl_proxy<REF_LIGHTPROFILE>::is_valid_reference_type( DB::Tag tag) const
{
    return m_transaction->get_db_transaction()->get_class_id( tag) == LIGHTPROFILE::ID_LIGHTPROFILE;
}

template <>
bool Ref_impl_proxy<REF_BSDF_MEASUREMENT>::is_valid_reference_type( DB::Tag tag) const
{
    return m_transaction->get_db_transaction()->get_class_id( tag) == BSDFM::ID_BSDF_MEASUREMENT;
}


// explicit template instantiation for Ref_impl<T>
template class Ref_impl<REF_UNTYPED>;
template class Ref_impl<REF_TEXTURE>;
template class Ref_impl<REF_LIGHTPROFILE>;
template class Ref_impl<REF_BSDF_MEASUREMENT>;

// explicit template instantiation for Ref_impl_proxy<T>
template class Ref_impl_proxy<REF_UNTYPED>;
template class Ref_impl_proxy<REF_TEXTURE>;
template class Ref_impl_proxy<REF_LIGHTPROFILE>;
template class Ref_impl_proxy<REF_BSDF_MEASUREMENT>;

} // namespace NEURAY

} // namespace MI

