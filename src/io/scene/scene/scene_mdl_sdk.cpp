/***************************************************************************************************
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Stubs for the MDL SDK library.

#include "pch.h"

#include <io/scene/scene/i_scene_mdl_sdk.h>
#include <io/scene/scene/i_scene_scene_element_base.h>

#include <base/data/dblight/dblight_database.h>
#include <base/data/serial/serial.h>
#include <base/system/main/module_registration.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_annotation_definition_proxy.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/texture/i_texture.h>

namespace MI {

namespace SCENE {

static SYSTEM::Module_registration<Scene_module> s_module( M_SCENE, "SCENE");

SYSTEM::Module_registration_entry* Scene_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

std::string get_class_name( SERIAL::Class_id id)
{
    return "";
}

std::string Scene_element_base::get_class_name() const
{
    return SCENE::get_class_name( this->get_class_id());
}

void Scene_element_base::swap( Scene_element_base& other)
{
    std::swap( m_attributes, other.m_attributes);
}

size_t Scene_element_base::get_size() const
{
    return sizeof( *this)
        + DB::Element_base::get_size() - sizeof( DB::Element_base)
        + m_attributes.get_size() - sizeof( ATTR::Attribute_set);
}

const SERIAL::Serializable* Scene_element_base::serialize( SERIAL::Serializer* serializer) const
{
    m_attributes.serialize( serializer);
    return this + 1;
}

SERIAL::Serializable* Scene_element_base::deserialize( SERIAL::Deserializer* deserializer)
{
    m_attributes.deserialize( deserializer);
    return this + 1;
}

void Scene_element_base::get_references( DB::Tag_set* result) const
{
    get_attributes()->get_references( result);
    get_scene_element_references( result);
}

void Scene_module::register_db_elements( DB::Database* db)
{
    auto* db_impl = static_cast<DBLIGHT::Database_impl*>( db);
    SERIAL::Deserialization_manager* manager = db_impl->get_deserialization_manager();

#define REGISTER_CLASS(c) manager->register_class( c::id, c::factory)

    REGISTER_CLASS( BSDFM::Bsdf_measurement);
    REGISTER_CLASS( BSDFM::Bsdf_measurement_impl);
    REGISTER_CLASS( DBIMAGE::Image);
    REGISTER_CLASS( DBIMAGE::Image_impl);
    REGISTER_CLASS( LIGHTPROFILE::Lightprofile);
    REGISTER_CLASS( LIGHTPROFILE::Lightprofile_impl);
    REGISTER_CLASS( MDL::Mdl_annotation_definition_proxy);
    REGISTER_CLASS( MDL::Mdl_compiled_material);
    REGISTER_CLASS( MDL::Mdl_function_call);
    REGISTER_CLASS( MDL::Mdl_function_definition);
    REGISTER_CLASS( MDL::Mdl_module);
    REGISTER_CLASS( TEXTURE::Texture);

#undef REGISTER_CLASS

}

} // namespace SCENE

} // namespace MI
