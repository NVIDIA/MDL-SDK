/******************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#include "mdl_material_description.h"
#include "mdl_material.h"
#include "mdl_sdk.h"
#include "base_application.h"

#include "example_shared.h"
#include "utils.h"

#include <mi/mdl_sdk.h>
#include <mi/neuraylib/definition_wrapper.h>

#include <cassert>

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{
    std::atomic<uint64_t> s_material_desc_id_counter(0);
    std::atomic<uint64_t> s_texture_lookup_call_counter(0);
} // anonymous

// ------------------------------------------------------------------------------------------------

const std::string Mdl_material_description::Invalid_material_identifier = "::dxr::not_available";

// ------------------------------------------------------------------------------------------------

Mdl_material_description::Mdl_material_description(
    const std::string& unique_material_identifier)
    : m_scene(nullptr)
    , m_scene_material()
    , m_is_loaded(false)
    , m_is_fallback(false)
    , m_supports_reloading(false)
    , m_loader(nullptr)
    , m_parameter_list(nullptr)
    , m_module_db_names(1, "")
    , m_material_definition_db_name("")
    , m_qualified_module_name("")
    , m_material_name("")
    , m_name_in_scene("")
    , m_source_code("")
    , m_flags(IMaterial::Flags::None)
    , m_unique_id(s_material_desc_id_counter.fetch_add(1))
{
    m_scene_material.name = unique_material_identifier;
}

// ------------------------------------------------------------------------------------------------

Mdl_material_description::Mdl_material_description()
    : Mdl_material_description(Invalid_material_identifier)
{
}

// ------------------------------------------------------------------------------------------------

Mdl_material_description::Mdl_material_description(
    const IScene_loader::Scene* scene,
    const IScene_loader::Material& material)
    : Mdl_material_description()
{
    m_scene = scene;
    m_scene_material = material;
}

// ------------------------------------------------------------------------------------------------

Mdl_material_description::~Mdl_material_description()
{
    m_parameter_list = nullptr;
}

// ------------------------------------------------------------------------------------------------

const IScene_loader::Scene* Mdl_material_description::get_scene() const
{
    return m_scene;
}

// ------------------------------------------------------------------------------------------------

const IScene_loader::Material& Mdl_material_description::get_scene_material() const
{
    return m_scene_material;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // loaded before
    if (m_is_loaded)
        return true;

    // special case: the invalid material is requested directly
    if (m_scene_material.name == Invalid_material_identifier)
    {
        if (!load_material_definition_fallback(sdk, scene_directory, context))
        {
            log_error("[FATAL] Invalid (fall-back) material can not be loaded.", SRC);
            return false;
        }

        m_is_loaded = true;
    }

    // depending on the type of material override (simple name convention)
    // it can make sense to use a gltf material in case of loading failure
    bool skip_glft_material_as_first_fall_back = false;

    // handle mdl (by convention when the name starts with '::')
    if (!m_is_loaded && mi::examples::strings::starts_with(m_scene_material.name, "::"))
    {
        log_info("Material name matches the convention for overriding by an MDL material: " +
            m_scene_material.name);

        m_is_loaded = load_material_definition_mdl(sdk, scene_directory, context);
        skip_glft_material_as_first_fall_back = true;
    }

    // handle mdle
    if (!m_is_loaded && (
        mi::examples::strings::ends_with(m_scene_material.name, ".mdle") ||
        mi::examples::strings::ends_with(m_scene_material.name, ".mdle::main")))
    {
        log_info("Material name matches the convention for overriding by an MDLE material: " +
            m_scene_material.name);

        m_is_loaded = load_material_definition_mdle(sdk, scene_directory, context);
        skip_glft_material_as_first_fall_back = true;
    }

    // handle loaders
    if (!m_is_loaded)
    {
        sdk.get_library()->visit_material_description_loaders(
            [&](const IMdl_material_description_loader* loader)
            {
                // load if there is a match
                if (loader->match_gltf_name(m_scene_material.name))
                {
                    log_info("Material name matches the convention for overriding: " +
                        m_scene_material.name);

                    m_loader = loader; // keep the loader for reloading
                    m_is_loaded = load_material_definition_loader(sdk, scene_directory, context, loader);

                    // do not use the gltf support as fall-back (pink instead)
                    skip_glft_material_as_first_fall_back = true;
                    return false; // stop visits, even in case of failure
                }
                return true; // continue visits
            });
    }

    // fall-back to gltf support materials
    if (!skip_glft_material_as_first_fall_back && !m_is_loaded)
    {
        m_is_loaded = load_material_definition_gltf_support(sdk, scene_directory, context);
        log_info("Material name does not match any convention, "
            "so the GLTF support material is used: " + m_scene_material.name);
    }

    // fall-back
    if (!m_is_loaded)
        m_is_loaded = load_material_definition_fallback(sdk, scene_directory, context);

    if (m_is_loaded)
    {
        // get the loaded module
        mi::base::Handle<const mi::neuraylib::IModule> module(
            sdk.get_transaction().access<mi::neuraylib::IModule>(m_module_db_names[0].c_str()));

        // compute definition name including the parameter list
        m_material_definition_db_name = mi::examples::mdl::add_missing_material_signature(
            module.get(), m_module_db_names[0] + "::" + m_material_name);

        // reflect changed imports
        sdk.get_library()->update_module_dependencies(m_module_db_names[0]);

        return true;
    }

    log_error("[FATAL] Material definition can't be loaded and fall-back solutions failed for: "
        + m_scene_material.name, SRC);
    return false;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::is_loaded() const
{
    return m_is_loaded;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::get_scene_name() const
{
    return m_is_loaded ? m_name_in_scene.c_str() : nullptr;
}

// ------------------------------------------------------------------------------------------------

IMaterial::Flags Mdl_material_description::get_flags() const
{
    return m_is_loaded ? m_flags : IMaterial::Flags::None;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::is_fallback() const
{
    return m_is_loaded ? m_is_fallback : false;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::supports_reloading() const
{
    return m_is_loaded ? m_supports_reloading : false;
}

// ------------------------------------------------------------------------------------------------

const std::vector<std::string>& Mdl_material_description::get_module_db_names() const
{
    return m_module_db_names;
}

// ------------------------------------------------------------------------------------------------

std::string Mdl_material_description::build_material_graph(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    const std::string& unique_db_prefix,
    mi::neuraylib::IMdl_execution_context* context)
{
    // check if available and process materials specified using the NV_materials_mdl extension
    size_t root_call = m_scene_material.ext_NV_materials_mdl.functionCall;
    if (root_call != static_cast<int32_t>(-1) &&
        m_scene && m_scene->ext_NV_materials_mdl.functionCalls.size() > root_call)
    {
        std::string graph_root_db_name = build_material_graph_NV_materials_mdl(
            sdk, scene_directory, unique_db_prefix, context);
        if (!graph_root_db_name.empty())
            return graph_root_db_name;
    }

    // fall-back to naming conventions, glTF PBR materials or the invalid material as last resort

    // load the definitions
    load_material_definition(sdk, scene_directory, context);
    if (!is_loaded())
        return ""; // error case, no fall-back

    // get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        sdk.get_transaction().access<mi::neuraylib::IFunction_definition>(
            m_material_definition_db_name.c_str()));

    // create a material instance with the provided parameters and store it in the database
    mi::Sint32 ret = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        material_definition->create_function_call(m_parameter_list.get(), &ret));
    if (ret != 0 || !material_instance)
    {
        log_error("Instantiating material '" + m_name_in_scene + "' failed", SRC);
        return ""; // error case, no fall-back
    }

    // store the instance in the DB
    sdk.get_transaction().store(material_instance.get(), unique_db_prefix.c_str());
    return unique_db_prefix;
}
// ------------------------------------------------------------------------------------------------

namespace // anonymous
{
    // repeatedly used functions
    const mi::neuraylib::IModule* access_module(
        Mdl_sdk& sdk,
        const char* qualified_module_name,
        std::string& db_name,
        mi::neuraylib::IMdl_execution_context* context)
    {
        // assuming here we do not store encoded names in the glTF files
        mi::base::Handle<const mi::IString> encoded_qualified_module_name(
            sdk.get_factory().encode_module_name(qualified_module_name));
        qualified_module_name = encoded_qualified_module_name->get_c_str();

        // expected database name of the module to load
        mi::base::Handle<const mi::IString> module_db_name(
            sdk.get_factory().get_db_module_name(qualified_module_name));

        // check if the module and thereby the material definition is already loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            sdk.get_transaction().access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

        // if not, load it
        if (!module)
        {
            // Load the module that contains the material.
            // This functions supports multi-threading. It blocks when the requested module
            // or a dependency is loaded by a different thread.
            sdk.get_impexp_api().load_module(
                sdk.get_transaction().get(), qualified_module_name, context);

            // loading failed
            if (!sdk.log_messages(
                "Loading the module failed: " + std::string(qualified_module_name), context))
            {
                return nullptr;
            }
            // access the module for overload resolution
            module = sdk.get_transaction().access<mi::neuraylib::IModule>(
                module_db_name->get_c_str());
        }
        db_name = module_db_name->get_c_str();
        module->retain();
        return module.get();
    };

    struct function_call_data
    {
        std::string db_name;
        std::unique_ptr<mi::neuraylib::Argument_editor> ae;
    };

    struct type_data
    {
        mi::base::Handle<const mi::neuraylib::IType> type;
    };

    struct texture_data
    {
        std::string db_name_image;
        std::string db_name_texture_srgb;
        std::string db_name_texture_linear;
    };


    std::string convert_module_uri_to_package(const std::string& uri)
    {
        // remove .mdl
        std::string result = (uri.rfind(".mdl") != std::string::npos)
            ? uri.substr(0, uri.size() - 4) : uri;

        // remove mdl://
        if (result.find("mdl://") != std::string::npos)
            result = result.substr(6);

        // remove preceding . and ..
        if (result[0] == '.')
        {
            if (result[1] == '.')
                result = result.substr(2);
            else
                result = result.substr(1);
        }

        // prepend slash (/) if missing
        if (result[0] != '/')
            result = '/' + result;

        // replace all slashes (/) with a double colon (::)
        for (size_t pos = 0; pos < result.size(); pos++)
        {
            if (result[pos] != '/')
                continue;

            // find end of multi slash
            size_t end = pos + 1;
            while (end < result.size() && result[end] == '/')
                end++;

            result.replace(pos, end - pos, "::");
        }

        return result;
    }

    const mi::neuraylib::IType* create_type_from_gltf(
        Mdl_sdk& sdk,
        const IScene_loader::Scene* scene,
        const fx::gltf::NV_MaterialsMDL::Type& gltf_type,
        std::unordered_map<int32_t, type_data>& types,
        mi::neuraylib::IType_factory* tf,
        mi::neuraylib::IMdl_execution_context* context
    )
    {
        mi::base::Handle<const mi::neuraylib::IType> type;

        size_t vector_element_count = 0;
        size_t matrix_column_count = 0;
        mi::base::Handle<const mi::neuraylib::IType_atomic> vector_element_type;

        // handle builtin types as defined in the MDL specification Appendix A
        if (gltf_type.kind == fx::gltf::NV_MaterialsMDL::Type::Kind::BuiltinType)
        {
            // booleans, numeric types, strings, ...
            if (gltf_type.builtinType == "bool")
            {
                type = tf->create_bool();
            }
            else if (gltf_type.builtinType == "bool2")
            {
                vector_element_type = tf->create_bool();
                vector_element_count = 2;
            }
            else if (gltf_type.builtinType == "bool3")
            {
                vector_element_type = tf->create_bool();
                vector_element_count = 3;
            }
            else if (gltf_type.builtinType == "bool4")
            {
                vector_element_type = tf->create_bool();
                vector_element_count = 4;
            }
            else if (gltf_type.builtinType == "color")
            {
                type = tf->create_color();
            }
            else if (gltf_type.builtinType == "double")
            {
                type = tf->create_double();
            }
            else if (gltf_type.builtinType == "double2")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 2;
            }
            else if (gltf_type.builtinType == "double3")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 3;
            }
            else if (gltf_type.builtinType == "double4")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 4;
            }
            else if (gltf_type.builtinType == "double2x2")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 2;
                matrix_column_count = 2;
            }
            else if (gltf_type.builtinType == "double2x3")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 3;
                matrix_column_count = 2;
            }
            else if (gltf_type.builtinType == "double2x4")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 4;
                matrix_column_count = 2;
            }
            else if (gltf_type.builtinType == "double3x2")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 2;
                matrix_column_count = 3;
            }
            else if (gltf_type.builtinType == "double3x3")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 3;
                matrix_column_count = 3;
            }
            else if (gltf_type.builtinType == "double3x4")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 4;
                matrix_column_count = 3;
            }
            else if (gltf_type.builtinType == "double4x2")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 2;
                matrix_column_count = 4;
            }
            else if (gltf_type.builtinType == "double4x3")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 3;
                matrix_column_count = 4;
            }
            else if (gltf_type.builtinType == "double4x4")
            {
                vector_element_type = tf->create_double();
                vector_element_count = 4;
                matrix_column_count = 4;
            }
            else if (gltf_type.builtinType == "float")
            {
                type = tf->create_float();
            }
            else if (gltf_type.builtinType == "float2")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 2;
            }
            else if (gltf_type.builtinType == "float3")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 3;
            }
            else if (gltf_type.builtinType == "float4")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 4;
            }
            else if (gltf_type.builtinType == "float2x2")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 2;
                matrix_column_count = 2;
            }
            else if (gltf_type.builtinType == "float2x3")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 3;
                matrix_column_count = 2;
            }
            else if (gltf_type.builtinType == "float2x4")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 4;
                matrix_column_count = 2;
            }
            else if (gltf_type.builtinType == "float3x2")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 2;
                matrix_column_count = 3;
            }
            else if (gltf_type.builtinType == "float3x3")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 3;
                matrix_column_count = 3;
            }
            else if (gltf_type.builtinType == "float3x4")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 4;
                matrix_column_count = 3;
            }
            else if (gltf_type.builtinType == "float4x2")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 2;
                matrix_column_count = 4;
            }
            else if (gltf_type.builtinType == "float4x3")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 3;
                matrix_column_count = 4;
            }
            else if (gltf_type.builtinType == "float4x4")
            {
                vector_element_type = tf->create_float();
                vector_element_count = 4;
                matrix_column_count = 4;
            }
            else if (gltf_type.builtinType == "int")
            {
                type = tf->create_int();
            }
            else if (gltf_type.builtinType == "int2")
            {
                vector_element_type = tf->create_int();
                vector_element_count = 2;
            }
            else if (gltf_type.builtinType == "int3")
            {
                vector_element_type = tf->create_int();
                vector_element_count = 3;
            }
            else if (gltf_type.builtinType == "int4")
            {
                vector_element_type = tf->create_int();
                vector_element_count = 4;
            }
            else if (gltf_type.builtinType == "string")
            {
                type = tf->create_string();
            }
            // resources
            else if (gltf_type.builtinType == "texture_2d")
            {
                type = tf->create_texture(mi::neuraylib::IType_texture::TS_2D);
            }
            else if (gltf_type.builtinType == "light_profile")
            {
                type = tf->create_light_profile();
            }
            else if (gltf_type.builtinType == "bsdf_measurement")
            {
                type = tf->create_bsdf_measurement();
            }
            // builtinType structures
            else if (
                gltf_type.builtinType == "material" ||
                gltf_type.builtinType == "material_emission" ||
                gltf_type.builtinType == "material_geometry" ||
                gltf_type.builtinType == "material_surface" ||
                gltf_type.builtinType == "material_volume")
            {
                type = tf->create_struct(("::" + gltf_type.builtinType).c_str());
            }
            // builtinType enums
            else if (gltf_type.builtinType == "intensity_mode")
            {
                type = tf->create_enum(("::" + gltf_type.builtinType).c_str());
            }
            else
            {
                log_error("NV_materials_ext: builtin type '" + gltf_type.builtinType +
                    "' unknown or not handled. Do you intend to use a userType?", SRC);
            }

            // handle vectors
            if (matrix_column_count == 0 && vector_element_count > 0)
            {
                type = tf->create_vector(vector_element_type.get(), vector_element_count);
            }

            // handle matrices
            if (matrix_column_count > 0 && vector_element_count > 0)
            {
                mi::base::Handle<const mi::neuraylib::IType_vector> col_type(
                    tf->create_vector(vector_element_type.get(), vector_element_count));

                type = tf->create_matrix(col_type.get(), matrix_column_count);
            }
        }
        else
        {
            // type already processed?
            auto it = types.find(gltf_type.userType);
            if (it == types.end())
            {
                const auto& gltf_user_type =
                    scene->ext_NV_materials_mdl.userTypes[gltf_type.userType];

                // load/get the module that contains the type definition
                const std::string& type_module_name =
                    gltf_user_type.module == static_cast<int32_t>(-1)
                    ? "::<builtins>"
                    : convert_module_uri_to_package(scene->ext_NV_materials_mdl.modules[gltf_user_type.module].uri);

                std::string type_module_db_name;
                mi::base::Handle<const mi::neuraylib::IModule> type_module(
                    access_module(sdk, type_module_name.c_str(), type_module_db_name, context));
                if (!type_module)
                    return nullptr;

                // get the user type by name
                mi::base::Handle<const mi::neuraylib::IType_list> module_type_list(
                    type_module->get_types());
                const std::string user_type_name =
                    type_module_name + "::" + gltf_user_type.typeName;
                type = module_type_list->get_type(user_type_name.c_str());
                if (!type)
                {
                    log_error(
                        "User gltf_type '" + gltf_user_type.typeName +
                        "' not found in found in module '" + type_module_name.c_str() + "'.");
                    return nullptr;
                }

                it = types.insert({ gltf_type.userType, type_data{} }).first;
                it->second.type = mi::base::make_handle_dup(type.get());
            }
            else
            {
                type = it->second.type;
            }
        }

        type->retain();
        return type.get();
    }

    mi::neuraylib::IValue* create_value_from_gltf(
        const mi::neuraylib::IType* mdl_type,
        const fx::gltf::NV_MaterialsMDL::Argument& gltf_arg,
        std::unordered_map<int32_t, type_data>& types,
        mi::neuraylib::IType_factory* tf,
        mi::neuraylib::IValue_factory* vf,
        size_t array_offset
    )
    {
        mi::base::Handle<mi::neuraylib::IValue> value;

        switch (mdl_type->get_kind())
        {
        case mi::neuraylib::IType::TK_BOOL: {
            value = vf->create_bool(gltf_arg.value.data[array_offset].boolean);
            break;
        }
        case mi::neuraylib::IType::TK_COLOR: {
            value = vf->create_color(
                gltf_arg.value.data[array_offset * 3 + 0].decimal,
                gltf_arg.value.data[array_offset * 3 + 1].decimal,
                gltf_arg.value.data[array_offset * 3 + 2].decimal);
            break;
        }
        case mi::neuraylib::IType::TK_DOUBLE: {
            value = vf->create_double(gltf_arg.value.data[array_offset].decimal);
            break;
        }
        case mi::neuraylib::IType::TK_FLOAT: {
            value = vf->create_float(gltf_arg.value.data[array_offset].decimal);
            break;
        }
        case mi::neuraylib::IType::TK_INT: {
            value = vf->create_int(gltf_arg.value.data[array_offset].integer);
            break;
        }
        case mi::neuraylib::IType::TK_STRING: {
            value = vf->create_string(gltf_arg.value.dataStrings[array_offset].c_str());
            break;
        }
        case mi::neuraylib::IType::TK_TEXTURE: {
            // create the invalid texture here for the overload resolution only
            // the actual texture will be created in the texture_2d constructor calls
            mi::base::Handle<const mi::neuraylib::IType_texture> tex_type(
                tf->create_texture(mi::neuraylib::IType_texture::TS_2D));
            value = vf->create_texture(tex_type.get(), nullptr);
            break;
        }
        case mi::neuraylib::IType::TK_LIGHT_PROFILE: {
            // same here, invalid (empty) light profile
            value = vf->create_light_profile(nullptr);
            break;
        }
        case mi::neuraylib::IType::TK_BSDF_MEASUREMENT: {
            // same here, invalid (empty) measured BSDF
            value = vf->create_bsdf_measurement(nullptr);
            break;
        }
        case mi::neuraylib::IType::TK_VECTOR: {
            mi::base::Handle<const mi::neuraylib::IType_vector> vec_type(
                mdl_type->get_interface<mi::neuraylib::IType_vector>());
            mi::base::Handle<const mi::neuraylib::IType_atomic> elment_type(
                vec_type->get_element_type());
            mi::Size element_count = vec_type->get_size();

            value = vf->create_vector(vec_type.get());
            mi::base::Handle<mi::neuraylib::IValue_vector> vector_value(
                value->get_interface<mi::neuraylib::IValue_vector>());
            for (mi::Size i = 0; i < element_count; ++i)
            {
                mi::base::Handle<mi::neuraylib::IValue_atomic> scalar;
                switch (elment_type->get_kind())
                {
                case mi::neuraylib::IType::TK_BOOL:
                    scalar = vf->create_bool(
                        gltf_arg.value.data[array_offset * element_count + i].boolean);
                    break;
                case mi::neuraylib::IType::TK_DOUBLE:
                    scalar = vf->create_double(
                        gltf_arg.value.data[array_offset * element_count + i].decimal);
                    break;
                case mi::neuraylib::IType::TK_FLOAT:
                    scalar = vf->create_float(
                        gltf_arg.value.data[array_offset * element_count + i].decimal);
                    break;
                case mi::neuraylib::IType::TK_INT:
                    scalar = vf->create_int(
                        gltf_arg.value.data[array_offset * element_count + i].integer);
                    break;
                }
                vector_value->set_value(i, scalar.get());
            }
            break;
        }
        case mi::neuraylib::IType::TK_MATRIX: {

            mi::base::Handle<const mi::neuraylib::IType_matrix> mat_type(
                mdl_type->get_interface<mi::neuraylib::IType_matrix>());
            mi::base::Handle<const mi::neuraylib::IType_vector> vec_type(
                mat_type->get_element_type());
            mi::base::Handle<const mi::neuraylib::IType_atomic> elment_type(
                vec_type->get_element_type());
            mi::Size row_count = vec_type->get_size();
            mi::Size col_count = mat_type->get_size();
            size_t mat_size = row_count * col_count;

            value = vf->create_matrix(mat_type.get());
            mi::base::Handle<mi::neuraylib::IValue_matrix> mat_value(
                value->get_interface<mi::neuraylib::IValue_matrix>());

            for (mi::Size c = 0; c < col_count; ++c)
            {
                mi::base::Handle<mi::neuraylib::IValue_vector> col_value(
                    vf->create_vector(vec_type.get()));

                for (mi::Size r = 0; r < row_count; ++r)
                {
                    size_t col_major_index = c * row_count + r;
                    mi::base::Handle<mi::neuraylib::IValue_atomic> scalar;
                    switch (elment_type->get_kind())
                    {
                    case mi::neuraylib::IType::TK_DOUBLE:
                        scalar = vf->create_double(
                            gltf_arg.value.data[array_offset * mat_size + col_major_index].decimal);
                        break;
                    case mi::neuraylib::IType::TK_FLOAT:
                        scalar = vf->create_float(
                            gltf_arg.value.data[array_offset * mat_size + col_major_index].decimal);
                        break;
                    }
                    col_value->set_value(r, scalar.get());
                }
                mat_value->set_value(c, col_value.get());
            }
            break;
        }
        case mi::neuraylib::IType::TK_STRUCT: {
            // create a struct with default values here
            // the actual fields will be created in the constructor calls
            value = vf->create(mdl_type);
            break;
        }
        case mi::neuraylib::IType::TK_ENUM: {
            value = vf->create(mdl_type);
            mi::base::Handle<mi::neuraylib::IValue_enum> value_enum(
                value->get_interface<mi::neuraylib::IValue_enum>());
            if (value_enum->set_name(gltf_arg.value.dataStrings[array_offset].c_str()) != 0)
            {
                log_error(
                    "Enum value '" + gltf_arg.value.dataStrings[array_offset] +
                    "' not found for the specified gltf_type.");
                return nullptr;
            }
            break;
        }
        default:
            assert(false);
            break;
        }

        value->retain();
        return value.get();
    }
}

std::string Mdl_material_description::build_material_graph_NV_materials_mdl(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    const std::string& unique_db_prefix,
    mi::neuraylib::IMdl_execution_context* context)
{
    // traverse the graph for this material, create the nodes, collect the modules

    // collect modules used by this material, for reloading
    std::unordered_set<std::string> modules_used;

    std::unordered_map<int32_t, function_call_data> nodes;
    std::unordered_map<int32_t, type_data> types;
    std::unordered_map<int32_t, texture_data> textures;

    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        sdk.get_factory().create_expression_factory(sdk.get_transaction().get()));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        sdk.get_factory().create_value_factory(sdk.get_transaction().get()));
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        sdk.get_factory().create_type_factory(sdk.get_transaction().get()));

    std::stack<int32_t> to_process;
    to_process.push(m_scene_material.ext_NV_materials_mdl.functionCall);
    while (!to_process.empty())
    {
        int32_t current_function_call_index = to_process.top();
        to_process.pop();

        // node processed already, can appear at multiple locations of the graph
        if (nodes.find(current_function_call_index) != nodes.end())
            continue;

        // the function call to process
        const auto& gltf_function_call =
            m_scene->ext_NV_materials_mdl.functionCalls[current_function_call_index];

        // load/get the module that contains the function call definition
        // NOTE: The MDL glTF extension uses URIs for modules, which the SDK doesn't support yet.
        //       For now convert URI to the old package style names.
        std::string module_name = gltf_function_call.module == static_cast<int32_t>(-1)
            ? "::<builtins>"
            : convert_module_uri_to_package(m_scene->ext_NV_materials_mdl.modules[gltf_function_call.module].uri);

        std::string module_db_name;
        mi::base::Handle<const mi::neuraylib::IModule> module(
            access_module(sdk, module_name.c_str(), module_db_name, context));
        if (!module)
            return "";

        // overload resolution
        // therefore, create an argument list with known parameters
        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(ef->create_expression_list());
        for (const auto& gltf_arg : gltf_function_call.arguments)
        {
            fx::gltf::NV_MaterialsMDL::Type type;
            bool is_call = false;
            if (gltf_arg.kind == fx::gltf::NV_MaterialsMDL::Argument::Kind::Value)
            {
                type = gltf_arg.type;
            }
            else
            {
                type = m_scene->ext_NV_materials_mdl.functionCalls[gltf_arg.functionCall].type;
                is_call = true;
                to_process.push(gltf_arg.functionCall);
            }

            // convert the gltf argument to an neuray type...
            mi::base::Handle<const mi::neuraylib::IType> mdl_type(
                create_type_from_gltf(sdk, m_scene, type, types, tf.get(), context));

            // ... and convert the values
            mi::base::Handle<mi::neuraylib::IValue> value;
            if (type.arraySize == static_cast<int32_t>(-1))
            {
                // non-array values
                if (is_call)
                    value = vf->create(mdl_type.get());
                else
                    value = create_value_from_gltf(
                        mdl_type.get(), gltf_arg, types, tf.get(), vf.get(), 0);
            }
            else
            {
                // array values
                mi::base::Handle<const mi::neuraylib::IType_array> array_type(
                    tf->create_immediate_sized_array(mdl_type.get(), type.arraySize));
                value = vf->create_array(array_type.get());

                if (!is_call)
                {
                    mi::base::Handle<mi::neuraylib::IValue_array> array_value(
                        value->get_interface<mi::neuraylib::IValue_array>());
                    for (size_t i = 0; i < type.arraySize; ++i)
                    {
                        mi::base::Handle<mi::neuraylib::IValue> element(
                            create_value_from_gltf(
                                mdl_type.get(), gltf_arg, types, tf.get(), vf.get(), i));
                        array_value->set_value(i, element.get());
                    }
                }
            }

            // add an expression with the value of the builtin or user type the argument list
            if (value)
            {
                mi::base::Handle<mi::neuraylib::IExpression> expr(ef->create_constant(value.get()));
                arguments->add_expression(gltf_arg.name.c_str(), expr.get());
            }
        }

        // compute the database name of the material or function
        // .. without the argument list
        std::string function_definition_name;
        if (gltf_function_call.module != static_cast<int32_t>(-1))
            function_definition_name = module_name;
        function_definition_name += "::" + gltf_function_call.functionName;

        // special handling for resources in glTF
        if (function_definition_name == "::texture_2d")
        {
            // as arguments we get the texture name (in form of an image array index)
            // we can also get a value for gamma and a selector
            // the API handles resources a differently and for that we replace the constructor
            // call here.

            std::string texture_db_name;
            mi::base::Handle<const mi::neuraylib::IImage> image;
            mi::base::Handle<mi::neuraylib::IValue_texture> tex_value;

            bool load_image_from_uri = true; // hard coded for now
            if (load_image_from_uri)
            {
                // read back the arguments parsed

                // the index of the image in the glTF image array
                mi::base::Handle<const mi::neuraylib::IExpression_constant> name_expr(
                    arguments->get_expression<mi::neuraylib::IExpression_constant>("name"));
                mi::base::Handle<const mi::neuraylib::IValue_int> name_value(
                    name_expr->get_value<mi::neuraylib::IValue_int>());

                // images are loaded to the db while parsing the glTF file
                const std::string& image_db_name =
                    m_scene->resources[name_value->get_value()].resource_db_name;

                // gamma is optional
                float gamma = 0.0f;
                mi::base::Handle<const mi::neuraylib::IExpression_constant> gamma_expr(
                    arguments->get_expression<mi::neuraylib::IExpression_constant>("gamma"));
                if (gamma_expr)
                {
                    mi::base::Handle<const mi::neuraylib::IValue_enum> gamma_value(
                        gamma_expr->get_value<mi::neuraylib::IValue_enum>());
                    switch (gamma_value->get_index())
                    {
                    case 1:
                        gamma = 1.0f;
                        break;
                    case 2:
                        gamma = 2.2f;
                        break;
                    default:
                        gamma = 0.0f;
                        break;
                    }
                }

                // compute new database name
                std::string gamma_str = std::to_string(int(gamma * 100));
                texture_db_name = image_db_name + "_texture2d_" + gamma_str;

                // if the texture does not exist yet, create it
                mi::base::Handle<const mi::neuraylib::ITexture> texture(
                    sdk.get_transaction().access<mi::neuraylib::ITexture>(texture_db_name.c_str()));
                if (!texture)
                {
                    mi::base::Handle<mi::neuraylib::ITexture> new_texture(
                        sdk.get_transaction().create<mi::neuraylib::ITexture>("Texture"));
                    new_texture->set_image(image_db_name.c_str());
                    new_texture->set_gamma(gamma);
                    sdk.get_transaction().store(new_texture.get(), texture_db_name.c_str());
                }

                // create a value
                mi::base::Handle<const mi::neuraylib::IType_texture> tex_type(
                    tf->create_texture(mi::neuraylib::IType_texture::TS_2D));
                tex_value = vf->create_texture(tex_type.get(), texture_db_name.c_str());
            }

            // replace the entire constructor call by the copy constructor
            // that takes the texture_2d object we just created
            mi::base::Handle<mi::neuraylib::IExpression> tex_expr(
                ef->create_constant(tex_value.get()));
            arguments = ef->create_expression_list();
            arguments->add_expression("value", tex_expr.get());
        }

        // special handling for template functions, since overload resolution
        // doesn't work for them without the template parameter list
        if (function_definition_name == "::T[]")
        {
            function_definition_name = "::T[](...)";
        }
        else if (function_definition_name == "::operator[]")
        {
            function_definition_name = "::operator[](%3C0%3E[],int)";
        }
        else if (function_definition_name == "::operator_len")
        {
            function_definition_name = "::operator_len(%3C0%3E[])";
        }
        else if (function_definition_name == "::operator?")
        {
            function_definition_name = "::operator%3F(bool,%3C0%3E,%3C0%3E)";
        }
        else if (function_definition_name == "::operator_cast")
        {
            function_definition_name = "::operator_cast(%3C0%3E)";

            // the cast operator takes a second special expression which is used to
            // determine the return type of the call
            fx::gltf::NV_MaterialsMDL::Type return_type = gltf_function_call.type;
            mi::base::Handle<const mi::neuraylib::IType> mdl_type(
                create_type_from_gltf(sdk, m_scene, return_type, types, tf.get(), context));
            mi::base::Handle<mi::neuraylib::IValue> value(
                vf->create(mdl_type.get()));

            mi::base::Handle<mi::neuraylib::IExpression> cast_return_expr(
                ef->create_constant(value.get()));
            arguments->add_expression("cast_return", cast_return_expr.get());
        }
        else
        {
            // handles non-template cases that require encoding, like the shift operators
            mi::base::Handle<const mi::IString> encoded_function_definition_name(
                sdk.get_factory().encode_function_definition_name(function_definition_name.c_str(), nullptr));
            function_definition_name = encoded_function_definition_name->get_c_str();
            // drop the `()` at the end which are added by `encode_function_definition_name`
            // since we want to do overload resolution in the next step
            // note: the actual parameter list is always empty because we did not pass any parameters
            // `encode_function_definition_name` works on strings only and does not check if such a function exits.
            function_definition_name = function_definition_name.substr(0, function_definition_name.size() - 2);
        }

        mi::base::Handle<const mi::IString> definition_db_name(
            sdk.get_factory().get_db_definition_name(function_definition_name.c_str()));

        mi::base::Handle<const mi::IArray> overloads(
            module->get_function_overloads(
                definition_db_name->get_c_str(), arguments.get()));

        if (overloads->get_length() == 0)
        {
            log_error(
                "No matching overload for function '" + gltf_function_call.functionName +
                "' found in module '" + module_name + "' for material: " + m_scene_material.name +
                "\nUsing a fall-back material.");
            return "";
        }

        if (overloads->get_length() > 1)
        {
            log_warning(
                "No unique overload for function '" + gltf_function_call.functionName +
                "' found in module '" + module_name + "' for material: " + m_scene_material.name +
                "\nUsing the first one.");
        }
        // now with argument list
        definition_db_name = overloads->get_element<mi::IString>(0);

        // find the MDL definition
        mi::neuraylib::Definition_wrapper dw(
            sdk.get_transaction().get(), definition_db_name->get_c_str(), &sdk.get_factory());

        if (!dw.is_valid())
        {
            log_error(
                "Module was loaded but the material/function name could not be found: " +
                m_scene_material.name + "\nUsing a fall-back material.");
            return "";
        }
        // create a call with default arguments, we will edit them later
        mi::Sint32 ret = 0;
        mi::base::Handle<mi::neuraylib::IScene_element> scene_element(
            dw.create_instance(arguments.get(), &ret));
        if (!scene_element || ret != 0)
        {
            log_error(
                "Function call could not be created: " + m_scene_material.name +
                "\nUsing a fall-back material.");
            return "";
        }

        // store to DB
        std::string call_db_name =
            unique_db_prefix + "_" + std::to_string(current_function_call_index);
        if (!gltf_function_call.name.empty()) // attach the name if available
            call_db_name += "_" + gltf_function_call.name;

        if (sdk.get_transaction().store(scene_element.get(), call_db_name.c_str()) != 0)
        {
            log_error(
                "Function call could not be stored in DB: " + m_scene_material.name +
                "\nUsing a fall-back material.");
            return "";
        }
        scene_element = nullptr;

        // create a node
        nodes.insert({ current_function_call_index, function_call_data{} });
        auto& node = nodes[current_function_call_index];
        node.db_name = call_db_name;
        node.ae = std::make_unique<mi::neuraylib::Argument_editor>(
            sdk.get_transaction().get(), call_db_name.c_str(), &sdk.get_factory(), true);
        if (!node.ae->is_valid())
        {
            log_error(
                "Function call could be edited in DB: " + m_scene_material.name +
                "\nUsing a fall-back material.");
            return "";
        }

        // add the module to the used modules list if not yet in it
        if(modules_used.find(module_db_name.c_str()) == modules_used.end())
            modules_used.insert(module_db_name.c_str());
    }

    // construct the actual graph by making connections
    for (auto& node : nodes)
    {
        // source infos
        const auto& gltf_function_call =
            m_scene->ext_NV_materials_mdl.functionCalls[node.first];

        // target node
        mi::neuraylib::Argument_editor* ae = node.second.ae.get();

        // covert arguments
        for (const auto& arg : gltf_function_call.arguments)
        {
            // only edit the calls, we already set the constants
            if (arg.kind == fx::gltf::NV_MaterialsMDL::Argument::Kind::FunctionCall)
            {
                if (ae->set_call(arg.name.c_str(), nodes[arg.functionCall].db_name.c_str()) != 0)
                {
                    log_error(
                        "Setting parameter `" + arg.name + "` failed: "+ m_scene_material.name);
                    return "";
                }
            }
        }

        // destructing the argument editor stores the changes
        // we synchronize this because other DB stores could happen on different threads
        sdk.get_transaction().execute<void>([&node](mi::neuraylib::ITransaction*) {
            node.second.ae.reset();
        });
    }

    m_name_in_scene = "[mdl] " + m_scene_material.name + " (" + std::to_string(m_unique_id) + ")";
    m_module_db_names.insert(m_module_db_names.end(), modules_used.begin(), modules_used.end());
    m_is_loaded = true;
    m_is_fallback = false;
    m_flags = IMaterial::Flags::None;
    m_supports_reloading = true;

    // db name of the root node
    return nodes[m_scene_material.ext_NV_materials_mdl.functionCall].db_name;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::regenerate_source_code(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // not supported or no module that was loaded from source
    if (!m_is_loaded || !m_supports_reloading || m_source_code.empty())
        return nullptr;

    // materials with generated code
    std::string generated_code = "";

    // handle loader
    if (m_loader)
        generated_code = m_loader->generate_mdl_source_code(m_scene_material.name, scene_directory);

    // code was generated successfully
    if (!generated_code.empty())
    {
        m_source_code = generated_code;
        return m_source_code.c_str();
    }

    // code generation failed, or no generator for this type of material
    load_material_definition_fallback(sdk, scene_directory, context);
    return nullptr;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_mdle(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // drop the optional module name
    if (mi::examples::strings::ends_with(m_scene_material.name, ".mdle::main"))
        m_scene_material.name = m_scene_material.name.substr(0, m_scene_material.name.length() - 6);

    // resolve relative paths within the scene directory
    if (mi::examples::io::is_absolute_path(m_scene_material.name))
        m_qualified_module_name = m_scene_material.name;
    else
        m_qualified_module_name = scene_directory + "/" + m_scene_material.name;

    // check if the file exists
    if (!mi::examples::io::file_exists(m_qualified_module_name))
    {
        log_warning(
            "The referenced MDLE file does not exist: " + m_scene_material.name +
            "\nUsing a fall-back material instead.");
        return false;
    }

    m_material_name = "main";

    // name only for display
    std::string name = m_qualified_module_name.substr(
        m_qualified_module_name.find_last_of('/') + 1);
    name = name.substr(0, name.length() - 4);
    m_name_in_scene = "[mdle] " + name + " (" + std::to_string(m_unique_id) + ")";

    // load the module
    if (!load_mdl_module(sdk, scene_directory, context))
        return false;

    // get the loaded module
    mi::base::Handle<const mi::neuraylib::IModule> module(
        sdk.get_transaction().access<mi::neuraylib::IModule>(m_module_db_names[0].c_str()));

    // check if the mdle contains a material
    std::string material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), m_module_db_names[0] + "::" + m_material_name);

    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        sdk.get_transaction().access<const mi::neuraylib::IFunction_definition>(
            material_db_name.c_str()));
    if (!material_definition)
    {
        log_warning(
            "The referenced MDLE file does not contain a material: " +
            m_qualified_module_name + "\nUsing a fall-back material instead.");
        return false;
    }

    // mdle have defaults for all parameters
    // TODO: they can be overridden, possibly by data from the loaded scene
    m_parameter_list = nullptr;
    m_supports_reloading = true;
    m_is_fallback = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_mdl(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // split module and material name
    // [::<package>]::<module>::<material>
    if (!mi::examples::mdl::parse_cmd_argument_material_name(
        m_scene_material.name, m_qualified_module_name, m_material_name, false))
    {
        log_warning(
            "Material name is not a fully qualified material name: " + m_scene_material.name +
            "\nUsing a fall-back material instead.");
        return false;
    }

    // name only for display
    m_name_in_scene = "[mdl] " + m_material_name + " (" + std::to_string(m_unique_id) + ")";

    // load the module
    if (!load_mdl_module(sdk, scene_directory, context))
        return false;

    // get the loaded module
    mi::base::Handle<const mi::neuraylib::IModule> module(
        sdk.get_transaction().access<mi::neuraylib::IModule>(m_module_db_names[0].c_str()));

    // database name of the material
    std::string material_definition_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), m_module_db_names[0] + "::" + m_material_name);

    // check if the module contains the requested material
    mi::base::Handle<const mi::neuraylib::IFunction_definition> definition(
        sdk.get_transaction().access<const mi::neuraylib::IFunction_definition>(
            material_definition_db_name.c_str()));
    if (!definition)
    {
        log_error(
            "Module was loaded but the material name could not be found: " + m_scene_material.name +
            "\nUsing a fall-back material instead.");
        m_module_db_names[0] = "";
        return false;
    }

    // create a parameter list for all parameters (even without default)
    // for that the Definition_wrapper can also be used when creating an instance
    sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
    {
        mi::base::Handle<const  mi::neuraylib::IType_list> parameter_types(
            definition->get_parameter_types());
        mi::base::Handle<const  mi::neuraylib::IExpression_list> defaults(
            definition->get_defaults());

        mi::base::Handle<mi::neuraylib::IValue_factory> vf(
            sdk.get_factory().create_value_factory(t));
        mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
            sdk.get_factory().create_expression_factory(t));

        m_parameter_list = ef->create_expression_list();

        mi::Size count = definition->get_parameter_count();
        for (mi::Size i = 0; i < count; ++i) {
            const char* name = definition->get_parameter_name(i);
            mi::base::Handle<const mi::neuraylib::IExpression> default_(
                defaults->get_expression(name));
            if (!default_) {
                mi::base::Handle<const mi::neuraylib::IType> type(parameter_types->get_type(i));
                mi::base::Handle<mi::neuraylib::IValue> value(vf->create(type.get()));
                mi::base::Handle<mi::neuraylib::IExpression> expr(ef->create_constant(value.get()));
                m_parameter_list->add_expression(name, expr.get());
            }
        }
    });

    m_supports_reloading = true;
    m_is_fallback = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_gltf_support(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    m_qualified_module_name = "::nvidia::sdk_examples::gltf_support";

    // .. but it will be disabled for opaque material instances
    if (m_scene_material.alpha_mode == IScene_loader::Material::Alpha_mode::Opaque)
        m_flags = mi::examples::enums::set_flag(m_flags, IMaterial::Flags::Opaque);

    if (m_scene_material.single_sided == true)
        m_flags = mi::examples::enums::set_flag(m_flags, IMaterial::Flags::SingleSided);

    // model dependent support material name
    switch (m_scene_material.pbr_model)
    {
        case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
            m_material_name = "gltf_material_khr_specular_glossiness";
            break;

        case IScene_loader::Material::Pbr_model::Metallic_roughness:
        default:
            m_material_name = "gltf_material";
            break;
    }

    // name only for display
    m_name_in_scene = "[gltf] " + m_scene_material.name + " (" + std::to_string(m_unique_id) + ")";

    // load the module
    if (!load_mdl_module(sdk, scene_directory, context))
        return false;

    // setup the parameter list based on the gltf scene data
    parameterize_gltf_support_material(sdk, scene_directory, context);

    m_supports_reloading = true;
    m_is_fallback = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_mdl_module(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // expected database name of the module to load
    mi::base::Handle<const mi::IString> module_db_name(
        sdk.get_factory().get_db_module_name(m_qualified_module_name.c_str()));

    // check if the module and thereby the material definition is already loaded
    mi::base::Handle<const mi::neuraylib::IModule> module(
        sdk.get_transaction().access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

    // if not, load it
    if (!module)
    {
        // Load the module that contains the material.
        // This functions supports multi-threading. It blocks when the requested module
        // or a dependency is loaded by a different thread.
        sdk.get_impexp_api().load_module(
            sdk.get_transaction().get(), m_qualified_module_name.c_str(), context);

        // loading failed
        if (!sdk.log_messages(
            "Loading the module failed: " + m_qualified_module_name, context))
                return false;
    }

    m_module_db_names.resize(1);
    m_module_db_names[0] = module_db_name->get_c_str();
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_fallback(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // default module
    m_qualified_module_name = "::dxr";
    m_material_name = "not_available";

    static const char* default_module_src =
        "mdl 1.2;\n"
        "import df::*;\n"
        "export material not_available() = material(\n"
        "    surface: material_surface(\n"
        "        df::diffuse_reflection_bsdf(\n"
        "            tint: color(0.8, 0.0, 0.8)\n"
        "        )\n"
        "    )"
        ");";

    // expected database name of the module to load
    mi::base::Handle<const mi::IString> module_db_name(
        sdk.get_factory().get_db_module_name(m_qualified_module_name.c_str()));

    return sdk.get_transaction().execute<bool>([&](auto t)
    {
        // check if the module and thereby the material definition is already loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            t->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

        // module is not loaded already
        if (!module)
        {
            // load it
            sdk.get_impexp_api().load_module_from_string(
                t, m_qualified_module_name.c_str(), default_module_src, context);

            // loading failed?
            if (!sdk.log_messages(
                "Loading the module failed: " + m_qualified_module_name, context, SRC))
                    return false;
        }

        // module is loaded now
        m_module_db_names.resize(1);
        m_module_db_names[0] = module_db_name->get_c_str();
        m_name_in_scene = "[mdl] invalid material (" + std::to_string(m_unique_id) + ")";

        m_supports_reloading = false;
        m_is_fallback = true;
        m_parameter_list = nullptr;
        m_flags = IMaterial::Flags::None;
        return true;
    });
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_description::parameterize_gltf_support_material(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // this creates elements in the database, therefore it has to be locked when using threads
    sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
    {
        // create material parameters
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            sdk.get_neuray().get_api_component<mi::neuraylib::IMdl_factory>());
        mi::base::Handle<mi::neuraylib::IValue_factory> vf(
            mdl_factory->create_value_factory(t));
        mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
            mdl_factory->create_expression_factory(t));
        mi::base::Handle<mi::neuraylib::IType_factory> tf(
            mdl_factory->create_type_factory(t));

        mi::base::Handle<const mi::IString> gltf_support_module_db_name(
            mdl_factory->get_db_module_name("::nvidia::sdk_examples::gltf_support"));
        mi::base::Handle<const mi::neuraylib::IModule> gltf_support_module(
            t->access<mi::neuraylib::IModule>(gltf_support_module_db_name->get_c_str()));

        // create a new parameter list
        m_parameter_list = ef->create_expression_list();

        // helper to add a texture if it is available
        auto add_texture_resource = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            const mdl_d3d12::IScene_loader::Material::Texture_info& texture_info,
            float gamma)
        {
            std::string gamma_str = std::to_string(int(gamma * 100));
            std::string image_name;
            std::string texture_name;
            if (!texture_info.resource_db_name.empty())
            {
                image_name = texture_info.resource_db_name;
                texture_name = texture_info.resource_db_name + "_texture2d_" + gamma_str;
                mi::base::Handle<const mi::neuraylib::IImage> image(
                    t->access<mi::neuraylib::IImage>(image_name.c_str()));
                if (!image)
                {
                    log_warning("Resource expected to loaded to neuray db already: " +
                        texture_info.resource_db_name, SRC);
                    return;
                }
            }
            else if (!texture_info.resource_uri.empty())
            {
                image_name = "mdl::" + texture_info.resource_uri + "_image";
                texture_name = "mdl::" + texture_info.resource_uri + "_texture2d_" + gamma_str;
                mi::base::Handle<const mi::neuraylib::IImage> image(
                    t->access<mi::neuraylib::IImage>(image_name.c_str()));
                if (!image)
                {
                    mi::base::Handle<mi::neuraylib::IImage> new_image(
                        t->create<mi::neuraylib::IImage>("Image"));
                    std::string file_path = scene_directory + "/" + texture_info.resource_uri;
                    new_image->reset_file(file_path.c_str());
                    t->store(new_image.get(), image_name.c_str());
                }
            }
            else
                return;

            // if the texture does not exist yet, create it
            mi::base::Handle<const mi::neuraylib::ITexture> texture(
                t->access<mi::neuraylib::ITexture>(texture_name.c_str()));
            if (!texture)
            {
                mi::base::Handle<mi::neuraylib::ITexture> new_texture(
                    t->create<mi::neuraylib::ITexture>("Texture"));
                new_texture->set_image(image_name.c_str());
                new_texture->set_gamma(gamma);
                t->store(new_texture.get(), texture_name.c_str());
            }

            // Mark the texture for removing right away.
            // Note, this will not delete the data immediately. Instead it will be deleted
            // with the next transaction::commit(). Until then, we will have copied to resources
            // to the GPU.
            t->remove(texture_name.c_str());
            t->remove(image_name.c_str());

            mi::base::Handle<const mi::neuraylib::IType_texture> type(
                tf->create_texture(mi::neuraylib::IType_texture::TS_2D));
            mi::base::Handle<mi::neuraylib::IValue_texture> value(
                vf->create_texture(type.get(), texture_name.c_str()));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // helper to add a color parameter
        auto add_color = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            float r, float g, float b)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_color(r, g, b));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // helper to add a bool
        auto add_bool = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            bool v)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_bool(v));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // helper to add a float
        auto add_float = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            float x)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_float(x));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // helper to add a int
        auto add_int = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            int32_t x)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_int(x));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // helper to add a float2
        auto add_float2 = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            float x, float y)
        {
            mi::base::Handle<const mi::neuraylib::IType_float> float_type(tf->create_float());
            mi::base::Handle<const mi::neuraylib::IType_vector> vec_type(
                tf->create_vector(float_type.get(), 2));

            mi::base::Handle<mi::neuraylib::IValue_vector> value(vf->create_vector(vec_type.get()));
            mi::base::Handle<mi::neuraylib::IValue_float> value_x(vf->create_float(x));
            mi::base::Handle<mi::neuraylib::IValue_float> value_y(vf->create_float(y));
            value->set_value(0, value_x.get());
            value->set_value(1, value_y.get());
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // helper to add a enum
        auto add_enum = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            const std::string& enum_type_name,
            mi::Sint32 enum_value)
        {
            mi::base::Handle<const mi::neuraylib::IType_enum> type(
                tf->create_enum(enum_type_name.c_str()));

            mi::base::Handle<mi::neuraylib::IValue_enum> value(vf->create_enum(type.get()));
            value->set_value(enum_value);
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF parameter '" + expression_name + "'.", SRC);
        };

        // get definitions for texture lookup functions
        std::string db_name_without_param_list =
            gltf_support_module_db_name->get_c_str() + std::string("::gltf_texture_lookup");
        mi::base::Handle<const mi::IArray> overloads(
            gltf_support_module->get_function_overloads(db_name_without_param_list.c_str()));
        assert(overloads->get_length() == 1 && "Unexpected overload for 'gltf_texture_lookup'");
        mi::base::Handle<const mi::IString> gltf_texture_lookup_db_name(
            overloads->get_element<const mi::IString>(0));

        db_name_without_param_list =
            gltf_support_module_db_name->get_c_str() + std::string("::gltf_normal_texture_lookup");
        overloads = gltf_support_module->get_function_overloads(db_name_without_param_list.c_str());
        assert(overloads->get_length() == 1 &&
            "Unexpected overload for 'gltf_normal_texture_lookup'");
        mi::base::Handle<const mi::IString> gltf_normal_texture_lookup_db_name(
            overloads->get_element<const mi::IString>(0));

        // helper to add a texture lookup function
        auto add_texture = [&](
            mi::neuraylib::IExpression_list* expr_list,
            const std::string& expression_name,
            const mdl_d3d12::IScene_loader::Material::Texture_info& texture_info,
            float gamma,
            bool normal_texture = false,
            float normal_factor = 1.0f,
            int tex_tangent_index = 0)
        {
            if (texture_info.resource_uri.empty() && texture_info.resource_db_name.empty())
                return;

            mi::base::Handle<mi::neuraylib::IExpression_list> call_parameter_list(
                ef->create_expression_list());

            add_texture_resource(
                call_parameter_list.get(), "texture", texture_info, gamma);
            add_int(call_parameter_list.get(), "tex_coord_index", texture_info.texCoord);
            add_float2(
                call_parameter_list.get(), "offset", texture_info.offset.x, texture_info.offset.y);
            add_float(call_parameter_list.get(), "rotation", texture_info.rotation);
            add_float2(
                call_parameter_list.get(), "scale", texture_info.scale.x, texture_info.scale.y);

            add_enum(call_parameter_list.get(), "wrap_s",
                "::nvidia::sdk_examples::gltf_support::gltf_wrapping_mode",
                static_cast<mi::Sint32>(texture_info.wrap_s));
            add_enum(call_parameter_list.get(), "wrap_t",
                "::nvidia::sdk_examples::gltf_support::gltf_wrapping_mode",
                static_cast<mi::Sint32>(texture_info.wrap_s));

            if (normal_texture)
            {
                add_float(call_parameter_list.get(), "normal_scale_factor", normal_factor);
                add_int(call_parameter_list.get(), "tex_tangent_index", tex_tangent_index);
            }

            mi::neuraylib::Definition_wrapper dw(
                t, normal_texture ? gltf_normal_texture_lookup_db_name->get_c_str()
                                  : gltf_texture_lookup_db_name->get_c_str(),
                mdl_factory.get());

            mi::Sint32 res;
            mi::base::Handle<mi::neuraylib::IFunction_call> lookup_call(
                dw.create_instance<mi::neuraylib::IFunction_call>(call_parameter_list.get(), &res));

            if (res != 0)
            {
                log_error("Failed to create glTF texture lookup call", SRC);
                return;
            }

            std::string call_db_name = "mdl::gltf_support::texture_lookup_call_" +
                std::to_string(s_texture_lookup_call_counter++);
            t->store(lookup_call.get(), call_db_name.c_str());

            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_call(call_db_name.c_str()));

            if (expr_list->add_expression(expression_name.c_str(), expr.get()) != 0)
                log_error("Failed to set glTF texture lookup call '" + expression_name + "'.", SRC);
        };

        // helper to add transmission
        auto add_transmission = [&](
            const mdl_d3d12::IScene_loader::Material::Model_data_materials_transmission& trans)
        {
            add_float(m_parameter_list.get(), "transmission_factor", trans.transmission_factor);
            add_texture(m_parameter_list.get(), "transmission_texture",
                trans.transmission_texture, 1.0f);
        };

        // helper to add clear-coat
        auto add_clearcoat = [&](
            const mdl_d3d12::IScene_loader::Material::Model_data_materials_clearcoat& clearcoat)
        {
            add_float(m_parameter_list.get(), "clearcoat_factor", clearcoat.clearcoat_factor);
            add_texture(m_parameter_list.get(), "clearcoat_texture",
                clearcoat.clearcoat_texture, 1.0f);
            add_float(m_parameter_list.get(), "clearcoat_roughness_factor",
                clearcoat.clearcoat_roughness_factor);
            add_texture(m_parameter_list.get(), "clearcoat_roughness_texture",
                clearcoat.clearcoat_roughness_texture, 1.0f);
            add_texture(m_parameter_list.get(), "clearcoat_normal_texture",
                clearcoat.clearcoat_normal_texture, 1.0f, true, 1.0f);
        };

        // helper to add sheen
        auto add_sheen = [&](
            const mdl_d3d12::IScene_loader::Material::Model_data_materials_sheen& sheen)
        {
            add_color(m_parameter_list.get(), "sheen_color_factor",
                sheen.sheen_color_factor.x, sheen.sheen_color_factor.y, sheen.sheen_color_factor.z);
            add_texture(m_parameter_list.get(), "sheen_color_texture",
                sheen.sheen_color_texture, 2.2f);
            add_float(m_parameter_list.get(), "sheen_roughness_factor",
                sheen.sheen_roughness_factor);
            add_texture(m_parameter_list.get(), "sheen_roughness_texture",
                sheen.sheen_roughness_texture, 2.2f); // alpha channel is not affected by gamma
                                                      // chance to reuse the sheen color texture
        };

        // helper to add specular
        auto add_specular = [&](
            const mdl_d3d12::IScene_loader::Material::Model_data_materials_specular& specular)
        {
            add_float(m_parameter_list.get(), "specular_factor",
                specular.specular_factor);
            add_texture(m_parameter_list.get(), "specular_texture",
                specular.specular_texture, 2.2f);   // alpha channel is not affected by gamma
                                                    // chance to reuse the sheen color texture
            add_color(m_parameter_list.get(), "specular_color_factor",
                specular.specular_color_factor.x,
                specular.specular_color_factor.y,
                specular.specular_color_factor.z);
            add_texture(m_parameter_list.get(), "specular_color_texture",
                specular.specular_color_texture, 2.2f);
        };

        // helper to add volume
        auto add_volume = [&](
            const mdl_d3d12::IScene_loader::Material::Model_data_materials_volume& volume)
        {
            add_bool(m_parameter_list.get(), "thin_walled", volume.thin_walled);
            add_float(m_parameter_list.get(), "attenuation_distance", volume.attenuation_distance);
            add_color(m_parameter_list.get(), "attenuation_color",
                volume.attenuation_color.x,
                volume.attenuation_color.y,
                volume.attenuation_color.z);
        };

        // add the actual parameters to the parameter list
        add_texture(m_parameter_list.get(), "normal_texture",
            m_scene_material.normal_texture, 1.0f, true, m_scene_material.normal_scale_factor);

        add_texture(m_parameter_list.get(), "occlusion_texture",
            m_scene_material.occlusion_texture, 1.0f);
        add_float(m_parameter_list.get(), "occlusion_strength", m_scene_material.occlusion_strength);

        add_texture(m_parameter_list.get(), "emissive_texture",
            m_scene_material.emissive_texture, 2.2f);
        add_color(m_parameter_list.get(), "emissive_factor", m_scene_material.emissive_factor.x,
            m_scene_material.emissive_factor.y, m_scene_material.emissive_factor.z);

        add_enum(m_parameter_list.get(), "alpha_mode",
            "::nvidia::sdk_examples::gltf_support::gltf_alpha_mode",
            static_cast<mi::Sint32>(m_scene_material.alpha_mode));
        add_float(m_parameter_list.get(), "alpha_cutoff", m_scene_material.alpha_cutoff);

        // general extensions
        add_float(m_parameter_list.get(), "emissive_strength",
            m_scene_material.emissive_strength.emissive_strength);

        // model dependent parameters
        switch (m_scene_material.pbr_model)
            {
            case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
            {
                add_texture(m_parameter_list.get(), "diffuse_texture",
                            m_scene_material.khr_specular_glossiness.diffuse_texture, 2.2f);

                add_color(m_parameter_list.get(), "diffuse_factor",
                            m_scene_material.khr_specular_glossiness.diffuse_factor.x,
                            m_scene_material.khr_specular_glossiness.diffuse_factor.y,
                            m_scene_material.khr_specular_glossiness.diffuse_factor.z);

                add_float(m_parameter_list.get(), "base_alpha",
                            m_scene_material.khr_specular_glossiness.diffuse_factor.w);

                add_texture(m_parameter_list.get(), "specular_glossiness_texture",
                            m_scene_material.khr_specular_glossiness.specular_glossiness_texture, 2.2f);

                add_color(m_parameter_list.get(), "specular_factor",
                            m_scene_material.khr_specular_glossiness.specular_factor.x,
                            m_scene_material.khr_specular_glossiness.specular_factor.y,
                            m_scene_material.khr_specular_glossiness.specular_factor.z);
                add_float(m_parameter_list.get(), "glossiness_factor",
                            m_scene_material.khr_specular_glossiness.glossiness_factor);
                return;
            }

            case IScene_loader::Material::Pbr_model::Metallic_roughness:
            default:
            {
                add_texture(m_parameter_list.get(), "base_color_texture",
                            m_scene_material.metallic_roughness.base_color_texture, 2.2f);

                add_color(m_parameter_list.get(), "base_color_factor",
                            m_scene_material.metallic_roughness.base_color_factor.x,
                            m_scene_material.metallic_roughness.base_color_factor.y,
                            m_scene_material.metallic_roughness.base_color_factor.z);

                add_float(m_parameter_list.get(), "base_alpha",
                            m_scene_material.metallic_roughness.base_color_factor.w);

                add_texture(m_parameter_list.get(), "metallic_roughness_texture",
                            m_scene_material.metallic_roughness.metallic_roughness_texture, 1.0f);

                add_float(m_parameter_list.get(), "metallic_factor",
                            m_scene_material.metallic_roughness.metallic_factor);

                add_float(m_parameter_list.get(), "roughness_factor",
                            m_scene_material.metallic_roughness.roughness_factor);

                add_transmission(m_scene_material.metallic_roughness.transmission);
                add_clearcoat(m_scene_material.metallic_roughness.clearcoat);
                add_sheen(m_scene_material.metallic_roughness.sheen);
                add_specular(m_scene_material.metallic_roughness.specular);
                add_float(m_parameter_list.get(), "ior", m_scene_material.metallic_roughness.ior.ior);
                add_volume(m_scene_material.metallic_roughness.volume);
                return;
            }
        }
    });
}

// -------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_loader(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context,
    const IMdl_material_description_loader* loader)
{
    // naming convention using the gltf material name attribute
    std::string gltf_name = m_scene_material.name;

    // generate the source code
    m_source_code = loader->generate_mdl_source_code(gltf_name, scene_directory);
    if (m_source_code.empty())
        return false;

    // compute a full qualified module name
    std::string module_name = gltf_name;
    module_name = mi::examples::strings::replace(module_name, "\\", "::");
    module_name = mi::examples::strings::replace(module_name, "/", "::");
    module_name = mi::examples::strings::replace(module_name, ":", "%3A");
    module_name = mi::examples::strings::replace(module_name, ".", "%2E");
    module_name = mi::examples::strings::replace(module_name, "=", "%3D");
    mi::base::Handle<const mi::IString> encoded_module_name(
        sdk.get_factory().encode_module_name(module_name.c_str()));
    module_name = encoded_module_name->get_c_str();

    if (module_name.length() < 2 || module_name[0] != ':' || module_name[1] == ':')
        module_name = "::" + module_name;
    m_qualified_module_name = module_name;

    // load the actual module, sequentially for now
    sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
        {
            auto& mdl_impexp = sdk.get_impexp_api();
            mdl_impexp.load_module_from_string(
                t, m_qualified_module_name.c_str(),
                m_source_code.c_str(),
                context);
        });
    if (!sdk.log_messages(
        "Loading generated material (from materialX) failed: " + m_scene_material.name,
        context, SRC))
            return false;

    // expected database name of the module to load
    mi::base::Handle<const mi::IString> module_db_name(
        sdk.get_factory().get_db_module_name(m_qualified_module_name.c_str()));
    m_module_db_names.resize(1);
    m_module_db_names[0] = module_db_name->get_c_str();

    // get the loaded module
    mi::base::Handle<const mi::neuraylib::IModule> module(
        sdk.get_transaction().access<const mi::neuraylib::IModule>(
            module_db_name->get_c_str()));

    if (module->get_material_count() == 0)
    {
        log_error("Generated MDL exports no material: " + m_scene_material.name, SRC);
        return false;
    }

    // get the first material in the module
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        sdk.get_transaction().access<const mi::neuraylib::IFunction_definition>(module->get_material(0)));
    m_material_name = material_definition->get_mdl_simple_name();

    // set the name in the scene graph and on the GUI
    m_name_in_scene = loader->get_scene_name_prefix() + " " + m_material_name +
        " (" + std::to_string(m_unique_id) + ")";

    m_supports_reloading = loader->supports_reload();
    m_parameter_list = nullptr;         // defaults will be used
    m_flags = IMaterial::Flags::None;   // no hard coded assumptions
    return true;
}

}}} // mi::examples::mdl_d3d12
