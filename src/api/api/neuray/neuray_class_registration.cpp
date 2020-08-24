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
 ** \brief Source for the class registration.
 **/

#include "pch.h"

#include "neuray_class_registration.h"

#include "neuray_class_factory.h"

#include <mi/neuraylib/iextension_api.h>

#include <boost/core/ignore_unused.hpp>

#include <base/system/main/access_module.h>
#include <base/data/serial/serial.h>

// for the factory methods
#include "neuray_array_impl.h"
#include "neuray_compound_impl.h"
#include "neuray_enum_decl_impl.h"
#include "neuray_enum_impl.h"
#include "neuray_map_impl.h"
#include "neuray_pointer_impl.h"
#include "neuray_ref_impl.h"
#include "neuray_string_impl.h"
#include "neuray_structure_decl_impl.h"
#include "neuray_structure_impl.h"
#include "neuray_uuid_impl.h"
#include "neuray_number_impl.h"
#include "neuray_void_impl.h"



// for the factory methods
#include "neuray_array_impl_proxy.h"
#include "neuray_attribute_container_impl.h"
#include "neuray_bsdf_measurement_impl.h"
#include "neuray_compiled_material_impl.h"
#include "neuray_export_result_ext_impl.h"
#include "neuray_function_call_impl.h"
#include "neuray_function_definition_impl.h"
#include "neuray_image_impl.h"
#include "neuray_lightprofile_impl.h"
#include "neuray_material_definition_impl.h"
#include "neuray_material_instance_impl.h"
#include "neuray_module_impl.h"
#include "neuray_texture_impl.h"

// for the class IDs
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/texture/i_texture.h>



namespace MI {

namespace NEURAY {


void Class_registration::register_classes_part1( Class_factory* factory)
{
    mi::Sint32 result = 0;
    boost::ignore_unused( result);

#define REG2( a, b) \
    result = factory->register_class( a, b); \
    ASSERT( M_NEURAY_API, result == 0)

#define REG3( a, b, c) \
    result = factory->register_class( a, b, c); \
    ASSERT( M_NEURAY_API, result == 0)

    // IData_simple
    REG3( "Boolean",         Number_impl<mi::IBoolean,    bool>::create_api_class);
    REG3( "Sint8",           Number_impl<mi::ISint8,      mi::Sint8>::create_api_class);
    REG3( "Sint16",          Number_impl<mi::ISint16,     mi::Sint16>::create_api_class);
    REG3( "Sint32",          Number_impl<mi::ISint32,     mi::Sint32>::create_api_class);
    REG3( "Sint64",          Number_impl<mi::ISint64,     mi::Sint64>::create_api_class);
    REG3( "Uint8",           Number_impl<mi::IUint8,      mi::Uint8>::create_api_class);
    REG3( "Uint16",          Number_impl<mi::IUint16,     mi::Uint16>::create_api_class);
    REG3( "Uint32",          Number_impl<mi::IUint32,     mi::Uint32>::create_api_class);
    REG3( "Uint64",          Number_impl<mi::IUint64,     mi::Uint64>::create_api_class);
    REG3( "Float32",         Number_impl<mi::IFloat32,    mi::Float32>::create_api_class);
    REG3( "Float64",         Number_impl<mi::IFloat64,    mi::Float64>::create_api_class);
    REG3( "Size",            Number_impl<mi::ISize,       mi::Size>::create_api_class);
    REG3( "Difference",      Number_impl<mi::IDifference, mi::Difference>::create_api_class);
    REG2( "String",          String_impl::create_api_class);
    REG2( "Ref",             Ref_impl::create_api_class);
    REG2( "Uuid",            Uuid_impl::create_api_class);
    REG2( "Void",            Void_impl::create_api_class);
    REG2( "__Pointer",       Pointer_impl::create_api_class);
    REG2( "__Const_pointer", Const_pointer_impl::create_api_class);
    REG2( "__Enum",          Enum_impl::create_api_class);
    REG2( "Enum_decl",       Enum_decl_impl::create_api_class);

    // vector variants of ICompound interface
    REG2( "Boolean<2>", Boolean_2_impl::create_api_class);
    REG2( "Boolean<3>", Boolean_3_impl::create_api_class);
    REG2( "Boolean<4>", Boolean_4_impl::create_api_class);
    REG2( "Sint32<2>",  Sint32_2_impl::create_api_class );
    REG2( "Sint32<3>",  Sint32_3_impl::create_api_class );
    REG2( "Sint32<4>",  Sint32_4_impl::create_api_class );
    REG2( "Uint32<2>",  Uint32_2_impl::create_api_class );
    REG2( "Uint32<3>",  Uint32_3_impl::create_api_class );
    REG2( "Uint32<4>",  Uint32_4_impl::create_api_class );
    REG2( "Float32<2>", Float32_2_impl::create_api_class);
    REG2( "Float32<3>", Float32_3_impl::create_api_class);
    REG2( "Float32<4>", Float32_4_impl::create_api_class);
    REG2( "Float64<2>", Float64_2_impl::create_api_class);
    REG2( "Float64<3>", Float64_3_impl::create_api_class);
    REG2( "Float64<4>", Float64_4_impl::create_api_class);

    // matrix variants of ICompound interface
    REG2( "Boolean<2,2>", Boolean_2_2_impl::create_api_class);
    REG2( "Boolean<2,3>", Boolean_2_3_impl::create_api_class);
    REG2( "Boolean<2,4>", Boolean_2_4_impl::create_api_class);
    REG2( "Boolean<3,2>", Boolean_3_2_impl::create_api_class);
    REG2( "Boolean<3,3>", Boolean_3_3_impl::create_api_class);
    REG2( "Boolean<3,4>", Boolean_3_4_impl::create_api_class);
    REG2( "Boolean<4,2>", Boolean_4_2_impl::create_api_class);
    REG2( "Boolean<4,3>", Boolean_4_3_impl::create_api_class);
    REG2( "Boolean<4,4>", Boolean_4_4_impl::create_api_class);

    REG2( "Sint32<2,2>",  Sint32_2_2_impl::create_api_class );
    REG2( "Sint32<2,3>",  Sint32_2_3_impl::create_api_class );
    REG2( "Sint32<2,4>",  Sint32_2_4_impl::create_api_class );
    REG2( "Sint32<3,2>",  Sint32_3_2_impl::create_api_class );
    REG2( "Sint32<3,3>",  Sint32_3_3_impl::create_api_class );
    REG2( "Sint32<3,4>",  Sint32_3_4_impl::create_api_class );
    REG2( "Sint32<4,2>",  Sint32_4_2_impl::create_api_class );
    REG2( "Sint32<4,3>",  Sint32_4_3_impl::create_api_class );
    REG2( "Sint32<4,4>",  Sint32_4_4_impl::create_api_class );

    REG2( "Uint32<2,2>",  Uint32_2_2_impl::create_api_class );
    REG2( "Uint32<2,3>",  Uint32_2_3_impl::create_api_class );
    REG2( "Uint32<2,4>",  Uint32_2_4_impl::create_api_class );
    REG2( "Uint32<3,2>",  Uint32_3_2_impl::create_api_class );
    REG2( "Uint32<3,3>",  Uint32_3_3_impl::create_api_class );
    REG2( "Uint32<3,4>",  Uint32_3_4_impl::create_api_class );
    REG2( "Uint32<4,2>",  Uint32_4_2_impl::create_api_class );
    REG2( "Uint32<4,3>",  Uint32_4_3_impl::create_api_class );
    REG2( "Uint32<4,4>",  Uint32_4_4_impl::create_api_class );

    REG2( "Float32<2,2>", Float32_2_2_impl::create_api_class);
    REG2( "Float32<2,3>", Float32_2_3_impl::create_api_class);
    REG2( "Float32<2,4>", Float32_2_4_impl::create_api_class);
    REG2( "Float32<3,2>", Float32_3_2_impl::create_api_class);
    REG2( "Float32<3,3>", Float32_3_3_impl::create_api_class);
    REG2( "Float32<3,4>", Float32_3_4_impl::create_api_class);
    REG2( "Float32<4,2>", Float32_4_2_impl::create_api_class);
    REG2( "Float32<4,3>", Float32_4_3_impl::create_api_class);
    REG2( "Float32<4,4>", Float32_4_4_impl::create_api_class);

    REG2( "Float64<2,2>", Float64_2_2_impl::create_api_class);
    REG2( "Float64<2,3>", Float64_2_3_impl::create_api_class);
    REG2( "Float64<2,4>", Float64_2_4_impl::create_api_class);
    REG2( "Float64<3,2>", Float64_3_2_impl::create_api_class);
    REG2( "Float64<3,3>", Float64_3_3_impl::create_api_class);
    REG2( "Float64<3,4>", Float64_3_4_impl::create_api_class);
    REG2( "Float64<4,2>", Float64_4_2_impl::create_api_class);
    REG2( "Float64<4,3>", Float64_4_3_impl::create_api_class);
    REG2( "Float64<4,4>", Float64_4_4_impl::create_api_class);

    // other variants of the ICompound interface
    REG2( "Color", Color_impl::create_api_class);
    REG2( "Color3", Color3_impl::create_api_class);
    REG2( "Spectrum", Spectrum_impl::create_api_class);
    REG2( "Bbox3", Bbox3_impl::create_api_class);

    // IData_collection
    REG2( "__Array", Array_impl::create_api_class);
    REG2( "__Dynamic_array", Dynamic_array_impl::create_api_class);
    REG2( "__Map", Map_impl::create_api_class);
    REG2( "__Structure", Structure_impl::create_api_class);
    REG2( "Structure_decl", Structure_decl_impl::create_api_class);

    // proxies (for attributes and elements of compounds)
    REG3( "__Boolean_proxy", Number_impl_proxy<mi::IBoolean, bool>::create_api_class);
    REG3( "__Sint32_proxy",  Number_impl_proxy<mi::ISint32,  mi::Sint32>::create_api_class);
    REG3( "__Uint32_proxy",  Number_impl_proxy<mi::IUint32,  mi::Uint32>::create_api_class);
    REG3( "__Float32_proxy", Number_impl_proxy<mi::IFloat32, mi::Float32>::create_api_class);
    REG3( "__Float64_proxy", Number_impl_proxy<mi::IFloat64, mi::Float64>::create_api_class);

    // remaining proxies (for attributes)
    REG3( "__Sint8_proxy",  Number_impl_proxy<mi::ISint8,   mi::Sint8>::create_api_class);
    REG3( "__Sint16_proxy", Number_impl_proxy<mi::ISint16,  mi::Sint16>::create_api_class);
    REG3( "__Sint64_proxy", Number_impl_proxy<mi::ISint64,  mi::Sint64>::create_api_class);
    REG3( "__Uint8_proxy",  Number_impl_proxy<mi::IUint8,   mi::Uint8>::create_api_class);
    REG3( "__Uint16_proxy", Number_impl_proxy<mi::IUint16,  mi::Uint16>::create_api_class);
    REG3( "__Uint64_proxy", Number_impl_proxy<mi::IUint64,  mi::Uint64>::create_api_class);
    REG2( "__String_proxy", String_impl_proxy::create_api_class);
    REG2( "__Ref_proxy", Ref_impl_proxy::create_api_class);
    REG2( "__Enum_proxy", Enum_impl_proxy::create_api_class);
    REG2( "__Array_proxy", Array_impl_proxy::create_api_class);
    REG2( "__Dynamic_array_proxy", Dynamic_array_impl_proxy::create_api_class);
    REG2( "__Structure_proxy", Structure_impl_proxy::create_api_class);

#undef REG2
#undef REG3
}

void Class_registration::register_classes_part2( Class_factory* factory)
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
    REG( "__Material_definition", MDL::ID_MDL_MATERIAL_DEFINITION,
        Material_definition_impl::create_api_class,
        Material_definition_impl::create_db_element);
    REG( "__Material_instance", MDL::ID_MDL_MATERIAL_INSTANCE,
        Material_instance_impl::create_api_class,
        Material_instance_impl::create_db_element);
    REG( "__Module", MDL::ID_MDL_MODULE,
        Module_impl::create_api_class,
        Module_impl::create_db_element);
    REG( "Texture", TEXTURE::ID_TEXTURE,
        Texture_impl::create_api_class, Texture_impl::create_db_element);

    // register API classes without DB counterparts
    REG( "Export_result_ext", Export_result_ext_impl::create_api_class);


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


    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String",  "key");
    decl->add_member( "String",  "value");
    REG( "Manifest_field", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String",  "material_name");
    decl->add_member( "String",  "prototype_name");
    decl->add_member( "Interface",  "parameters");
    decl->add_member( "Interface",  "annotations");
    REG( "Material_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>(nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String", "definition_name");
    decl->add_member( "String", "prototype_name");
    decl->add_member( "Interface", "parameters");
    decl->add_member( "Interface", "annotations");
    decl->add_member( "Interface", "return_annotations");
    REG( "Create_materials_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String",  "path");
    decl->add_member( "String",  "name");
    decl->add_member( "Boolean",  "enforce_uniform");
    decl->add_member( "Interface",  "annotations");
    REG( "Parameter_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String",  "preset_name");
    decl->add_member( "String",  "prototype_name");
    decl->add_member( "Interface",  "defaults");
    decl->add_member( "Interface",  "annotations");
    REG( "Preset_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String",  "variant_name");
    decl->add_member( "String",  "prototype_name");
    decl->add_member( "Interface",  "defaults");
    decl->add_member( "Interface",  "annotations");
    REG( "Variant_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "Sint32",  "u");
    decl->add_member( "Sint32",  "v");
    decl->add_member( "Interface",  "canvas");
    REG( "Uvtile", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "Sint32",  "u");
    decl->add_member( "Sint32",  "v");
    decl->add_member( "Interface",  "reader");
    REG( "Uvtile_reader", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
    decl->add_member( "String", "prototype_name");
    decl->add_member( "Interface", "defaults");
    decl->add_member( "Interface", "annotations");
    decl->add_member( "String", "thumbnail_path");
    decl->add_member( "Interface", "user_files");
    REG( "Mdle_data", decl.get());

    decl = factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr);
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
    if (mi::base::make_handle(factory->get_structure_decl(s)).is_valid_interface()) { \
        result = factory->unregister_structure_decl(s); \
        ASSERT(M_NEURAY_API, result == 0); \
    }


    UNREG( "Manifest_field");
    UNREG( "Material_data");
    UNREG( "Create_materials_data");
    UNREG( "Parameter_data");
    UNREG( "Preset_data");
    UNREG( "Variant_data");
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
    if (strcmp( name, "Create_materials_data") == 0)
        return true;
    if( strcmp( name, "Parameter_data") == 0)
        return true;
    if( strcmp( name, "Preset_data") == 0)
        return true;
    if( strcmp( name, "Variant_data") == 0)
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

