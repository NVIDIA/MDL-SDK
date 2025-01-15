/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the class registration.
 **/

#include "pch.h"

#include "neuray_class_registration.h"

#include "neuray_class_factory.h"

#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/iextension_api.h>
#include <mi/neuraylib/istructure_decl.h>

#include <boost/core/ignore_unused.hpp>

#include <base/system/main/access_module.h>
#include <base/data/serial/serial.h>



#include <base/data/dblight/dblight_database.h>

// for the factory methods
#include "neuray_attribute_container_impl.h"
#include "neuray_bsdf_measurement_impl.h"
#include "neuray_compiled_material_impl.h"
#include "neuray_export_result_ext_impl.h"
#include "neuray_function_call_impl.h"
#include "neuray_function_definition_impl.h"
#include "neuray_image_impl.h"
#include "neuray_lightprofile_impl.h"
#include "neuray_module_impl.h"
#include "neuray_texture_impl.h"

// for the class IDs
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/texture/i_texture.h>



namespace MI {

namespace NEURAY {


void Class_registration::register_classes_part2( Class_factory* factory, DB::Database* db)
{
// The result is not checked since DiCE and the MDL SDK support INeuray::start() after
// INeuray::shutdown() and there is no unregistration implemented. That is, for subsequent calls
// of this method, all register_class() calls fail.
#define REG factory->register_class


    // register API classes with DB counterparts
    REG( "Attribute_container", ID_ATTRIBUTE_CONTAINER,
        Attribute_container_impl::create_api_class, Attribute_container_impl::create_db_element);
    REG( "Bsdf_measurement", BSDFM::ID_BSDF_MEASUREMENT,
        Bsdf_measurement_impl::create_api_class, Bsdf_measurement_impl::create_db_element);
    REG( "__Compiled_material", MDL::ID_MDL_COMPILED_MATERIAL,
        Compiled_material_impl::create_api_class,
        Compiled_material_impl::create_db_element);
    REG( "__Function_call", MDL::ID_MDL_FUNCTION_CALL,
        Function_call_impl::create_api_class,
        Function_call_impl::create_db_element);
    REG( "__Function_definition", MDL::ID_MDL_FUNCTION_DEFINITION,
        Function_definition_impl::create_api_class,
        Function_definition_impl::create_db_element);
    REG( "Image", DBIMAGE::ID_IMAGE,
        Image_impl::create_api_class, Image_impl::create_db_element);
    REG( "Lightprofile", LIGHTPROFILE::ID_LIGHTPROFILE,
        Lightprofile_impl::create_api_class, Lightprofile_impl::create_db_element);
    REG( "__Module", MDL::ID_MDL_MODULE,
        Module_impl::create_api_class,
        Module_impl::create_db_element);
    REG( "Texture", TEXTURE::ID_TEXTURE,
        Texture_impl::create_api_class, Texture_impl::create_db_element);

    // register API classes without DB counterparts
    REG( "Export_result_ext", Export_result_ext_impl::create_api_class);

    // register DB classes
    auto* db_impl = static_cast<DBLIGHT::Database_impl*>( db);
    SERIAL::Deserialization_manager* manager = db_impl->get_deserialization_manager();
    manager->register_class( Attribute_container::id, Attribute_container::factory);


#undef REG
}

void Class_registration::register_structure_declarations( Class_factory* factory)
{
    mi::Sint32 result = 0;
    boost::ignore_unused( result);

    mi::base::Handle<mi::IStructure_decl> decl;

#define REG(s, d) \
    result = factory->register_structure_decl( s, d); \
    ASSERT( M_NEURAY_API, result == 0)


    decl = factory->create_type_instance<mi::IStructure_decl>(
        nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String", "key");
    decl->add_member( "String", "value");
    REG( "Manifest_field", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>(
        nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String", "material_name");
    decl->add_member( "String", "prototype_name");
    decl->add_member( "Interface", "parameters");
    decl->add_member( "Interface", "annotations");
    REG( "Material_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>(
        nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "Sint32", "u");
    decl->add_member( "Sint32", "v");
    decl->add_member( "Size", "frame");
    decl->add_member( "Interface", "canvas");
    REG( "Uvtile", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>(
        nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "Sint32", "u");
    decl->add_member( "Sint32", "v");
    decl->add_member( "Size", "frame");
    decl->add_member( "Interface", "reader");
    REG( "Uvtile_reader", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>(
        nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String", "prototype_name");
    decl->add_member( "Interface", "defaults");
    decl->add_member( "Interface", "annotations");
    decl->add_member( "String", "thumbnail_path");
    decl->add_member( "Interface", "user_files");
    REG( "Mdle_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>(
        nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String", "source_path");
    decl->add_member( "String", "target_path");
    REG( "Mdle_user_file", decl.get());

#undef REG
}

void Class_registration::unregister_structure_declarations( Class_factory* factory)
{
    mi::Sint32 result = 0;
    boost::ignore_unused( result);

#define UNREG(s) \
    if( make_handle( factory->get_structure_decl( s))) { \
        result = factory->unregister_structure_decl( s); \
        ASSERT( M_NEURAY_API, result == 0); \
    }


    UNREG( "Manifest_field");
    UNREG( "Material_data");
    UNREG( "Uvtile");
    UNREG( "Uvtile_reader");
    UNREG( "Mdle_data");
    UNREG( "Mdle_user_file");

#undef UNREG
}

bool Class_registration::is_predefined_structure_declaration( const char* name)
{

    if( strcmp( name, "Manifest_field") == 0)
        return true;
    if( strcmp( name, "Material_data") == 0)
        return true;
    if( strcmp( name, "Uvtile") == 0)
        return true;
    if( strcmp( name, "Uvtile_reader") == 0)
        return true;
    if( strcmp( name, "Mdle_data") == 0)
        return true;
    if( strcmp( name, "Mdle_user_file") == 0)
        return true;

    return false;
}

void Class_registration::register_importers( mi::neuraylib::IExtension_api* extension_api)
{
}

void Class_registration::unregister_importers( mi::neuraylib::IExtension_api* extension_api)
{
}

void Class_registration::register_exporters( mi::neuraylib::IExtension_api* extension_api)
{
}

void Class_registration::unregister_exporters( mi::neuraylib::IExtension_api* extension_api)
{
}

} // namespace NEURAY

} // namespace MI

