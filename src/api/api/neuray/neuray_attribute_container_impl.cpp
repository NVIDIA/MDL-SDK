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
 ** \brief Source for the IAttribute_container implementation.
 **/

#include "pch.h"

#include "neuray_attribute_container_impl.h"

#include <base/data/serial/i_serializer.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Attribute_container_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new Attribute_container;
}

mi::base::IInterface* Attribute_container_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Attribute_container_impl())->cast_to_major();
}

mi::IData* Attribute_container_impl::create_attribute( const char* name, const char* type_name)
{
    ATTR::Attribute_set* attribute_set = get_attribute_set();
    return Attribute_set_impl_helper::create_attribute(
        attribute_set, this, name, type_name, /*skip_type_check*/ true);
}

mi::neuraylib::Element_type Attribute_container_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_ATTRIBUTE_CONTAINER;
}

void Attribute_container_impl::set_attribute_set(
    ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner)
{
    Parent_type::set_attribute_set( attribute_set, owner);
}

void Attribute_container_impl::set_attribute_set(
    const ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner) const
{
    Parent_type::set_attribute_set( attribute_set, owner);
}

size_t Attribute_container::get_size() const
{
    return sizeof(*this)
        + SCENE::Scene_element<Attribute_container, ID_ATTRIBUTE_CONTAINER>::get_size()
            - sizeof(SCENE::Scene_element<Attribute_container, ID_ATTRIBUTE_CONTAINER>);
}

const SERIAL::Serializable* Attribute_container::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);
    return this + 1;
}

SERIAL::Serializable* Attribute_container::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);
    return this + 1;
}

void Attribute_container::get_scene_element_references( DB::Tag_set* result) const
{
}

std::string Attribute_container::get_class_name() const
{
    return "Attribute_container";
}

} // namespace NEURAY

} // namespace MI
