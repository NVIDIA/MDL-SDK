/******************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/distilling/example_distilling.cpp
//
// Introduces the distillation of mdl materials to a fixed target model
// and showcases how to bake material paths to a texture

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <thread>

#include "example_shared.h"
#include "example_distilling_shared.h"

#include "utils/profiling.h"
using namespace mi::examples::profiling;

// Small struct used to store the result of a texture baking process
// of a material sub expression
struct Material_parameter
{
    // Small struct used to store the UV tiles
    // This approach assumes that there are no u/v transformations, otherwise it ends up baking the wrong u/v pairs
    struct UVTile
    {
        UVTile(mi::Sint32 u, mi::Sint32 v)
            :min_uv(u, v)
        {}
        std::pair<mi::Sint32, mi::Sint32> min_uv;
        bool operator < (const UVTile & o) const
        {
            return min_uv < o.min_uv;
        }
    };
    using Uvtiles = std::map<UVTile, mi::base::Handle<mi::neuraylib::ICanvas>>;
    using Frame_number = mi::Size;
    using Canvases = std::map <Frame_number, Uvtiles>;
    Canvases canvases;

    typedef void (Remap_func(mi::base::IInterface*));

    mi::base::Handle<mi::IData>                 value;
    mi::base::Handle<mi::neuraylib::ICanvas>    texture;

    std::string                                 value_type;
    std::string                                 bake_path;

    Remap_func*                                 remap_func;

    Material_parameter() : remap_func(nullptr)
    {

    }

    Material_parameter(
        const std::string& value_type,
        Remap_func* func = nullptr)
        : value_type(value_type)
        , remap_func(func)
    {

    }
};

typedef std::map<std::string, Material_parameter> Material;

// Creates an instance of the given material.
mi::neuraylib::IFunction_call* create_material_instance(
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_execution_context* context,
    const std::string& module_qualified_name,
    const std::string& material_simple_name)
{
    // Load the module.
    mdl_impexp_api->load_module(transaction, module_qualified_name.c_str(), context);
    if (!print_messages(context))
        exit_failure("Loading module '%s' failed.", module_qualified_name.c_str());

    // Get the database name for the module we loaded
    mi::base::Handle<const mi::IString> module_db_name(
        mdl_factory->get_db_module_name(module_qualified_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
        exit_failure("Failed to access the loaded module.");

    // Attach the material name
    std::string material_db_name
        = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
    material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), material_db_name);
    if (material_db_name.empty())
        exit_failure("Failed to find the material %s in the module %s.",
            material_simple_name.c_str(), module_qualified_name.c_str());

    // Get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Accessing definition '%s' failed.", material_db_name.c_str());

    // Create a material instance from the material definition with the default arguments.
    mi::Sint32 result;
    mi::neuraylib::IFunction_call* material_instance =
        material_definition->create_function_call(0, &result);
    if (result != 0)
        exit_failure("Instantiating '%s' failed.", material_db_name.c_str());

    return material_instance;
}

// Compiles the given material instance in the given compilation modes and stores
// it in the DB.
mi::neuraylib::ICompiled_material* compile_material_instance(
    mi::neuraylib::IMdl_factory *mdl_factory,
    mi::neuraylib::ITransaction *transaction,
    const mi::neuraylib::IFunction_call* material_instance,
    mi::neuraylib::IMdl_execution_context* context,
    bool class_compilation)
{
    Timing timing("Compiling");
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    // convert to target type SID_MATERIAL
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory(transaction));
    mi::base::Handle<const mi::neuraylib::IType> standard_material_type(
        tf->get_predefined_struct(mi::neuraylib::IType_struct::SID_MATERIAL));
    context->set_option("target_type", standard_material_type.get());

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::neuraylib::ICompiled_material* compiled_material =
        material_instance2->create_compiled_material(flags, context);
    check_success(print_messages(context));

    return compiled_material;
}

// Distills the given compiled material to the requested target model,
// and returns it
const mi::neuraylib::ICompiled_material* create_distilled_material(
    mi::neuraylib::IMdl_distiller_api* distiller_api,
    const mi::neuraylib::ICompiled_material* compiled_material,
    const char* target_model)
{
    mi::Sint32 result = 0;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_material(
        distiller_api->distill_material(compiled_material, target_model, nullptr, &result));
    check_success(result == 0);

    distilled_material->retain();
    return distilled_material.get();
}

// remap normal
void remap_normal(mi::base::IInterface* icanvas)
{
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        icanvas->get_interface<mi::neuraylib::ICanvas>());
    if(!canvas)
        return;
    // Convert normal values from the interval [-1.0,1.0] to [0.0, 1.0]
    mi::base::Handle<mi::neuraylib::ITile> tile (canvas->get_tile());
    if (!tile)
        return;
    mi::Float32* data = static_cast<mi::Float32*>(tile->get_data());

    const mi::Uint32 n = canvas->get_resolution_x() * canvas->get_resolution_y() * 3;
    for(mi::Uint32 i=0; i<n; ++i)
    {
        data[i] = (data[i] + 1.f) * 0.5f;
    }
}

// simple roughness to glossiness conversion
void rough_to_gloss(mi::base::IInterface* ii)
{
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        ii->get_interface<mi::neuraylib::ICanvas>());
    if(canvas)
    {
        mi::base::Handle<mi::neuraylib::ITile> tile (canvas->get_tile());
        mi::Float32* data = static_cast<mi::Float32*>(tile->get_data());

        const mi::Uint32 n = canvas->get_resolution_x() * canvas->get_resolution_y();
        for(mi::Uint32 i=0; i<n; ++i)
        {
            data[i] = 1.0f - data[i];
        }
        return;
    }
    mi::base::Handle<mi::IFloat32> value(
        ii->get_interface<mi::IFloat32>());
    if(value)
    {
        mi::Float32 f;
        mi::get_value(value.get(), f);
        value->set_value(1.0f - f);
    }
}

// Setup material parameters according to target model and
// collect relevant bake paths
void setup_target_material(
    const std::string& target_model,
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    Material& out_material)
{
    Timing timing("Setup");
    // Access surface.scattering function
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> parent_call(
        lookup_call("surface.scattering", cm));
    // ... and get its semantic
    mi::neuraylib::IFunction_definition::Semantics semantic(
        get_call_semantic(transaction, parent_call.get()));

    if(target_model == "diffuse")
    {
        // The target model is supposed to be a diffuse reflection bsdf
        check_success(semantic ==
            mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF);

        // Setup diffuse material parameters
        out_material["color"] = Material_parameter("Rgb_fp");
        out_material["roughness"] = Material_parameter("Float32");
        out_material["normal"] = Material_parameter("Float32<3>", remap_normal);

        // Specify bake paths
        out_material["color"].bake_path = "surface.scattering.tint";
        out_material["roughness"].bake_path = "surface.scattering.roughness";
        out_material["normal"].bake_path = "geometry.normal";

    }
    else if (target_model == "ue4" || target_model == "transmissive_pbr")
    {
        // Setup some UE4 material parameters
        out_material["base_color"] = Material_parameter("Rgb_fp");
        out_material["metallic"]  = Material_parameter("Float32");
        out_material["specular"]  = Material_parameter("Float32");
        out_material["roughness"]  = Material_parameter("Float32");
        out_material["normal"] = Material_parameter("Float32<3>", remap_normal);

        out_material["clearcoat_weight"] = Material_parameter("Float32");
        out_material["clearcoat_roughness"] = Material_parameter("Float32");
        out_material["clearcoat_normal"] = Material_parameter("Float32<3>", remap_normal);

        out_material["opacity"] = Material_parameter("Float32");

        std::string path_prefix = "surface.scattering.";

        bool is_transmissive_pbr = false;
        if (target_model == "transmissive_pbr")
        {
            is_transmissive_pbr = true;

            // insert parameters that only apply to transmissive_pbr
            out_material["anisotropy"] = Material_parameter("Float32");
            out_material["anisotropy_rotation"] = Material_parameter("Float32");
            out_material["transparency"] = Material_parameter("Float32");
            out_material["transmission_color"] = Material_parameter("Rgb_fp");

            // uniform
            out_material["attenuation_color"] = Material_parameter("Rgb_fp");
            out_material["attenuation_distance"] = Material_parameter("Float32");
            out_material["subsurface_color"] = Material_parameter("Rgb_fp");
            out_material["volume_ior"] = Material_parameter("Rgb_fp");

            // collect volume properties, they are guaranteed to exist
            out_material["attenuation_color"].bake_path = "volume.absorption_coefficient.s.v.attenuation";
            out_material["subsurface_color"].bake_path = "volume.absorption_coefficient.s.v.subsurface";
            out_material["attenuation_distance"].bake_path = "volume.scattering_coefficient.s.v.distance";
            out_material["volume_ior"].bake_path = "ior";
        }

        // Check for a clearcoat layer, first. If present, it is the outermost layer
        if(semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
        {
            // Setup clearcoat bake paths
            out_material["clearcoat_weight"].bake_path = path_prefix + "weight";
            out_material["clearcoat_roughness"].bake_path = path_prefix + "layer.roughness_u";
            out_material["clearcoat_normal"].bake_path = path_prefix + "normal";

            // Get clear-coat base layer
            parent_call = lookup_call("base", cm, parent_call.get());
            // Get clear-coat base layer semantic
            semantic = get_call_semantic(transaction, parent_call.get());
            // Extend path prefix
            path_prefix += "base.";
        }

        // Check for a weighted layer. Sole purpose of this layer is the transportation of
        // the under-clearcoat-normal. It contains an empty base and a layer with the
        // actual material body
        if(semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_WEIGHTED_LAYER)
        {
            // Collect under-clearcoat normal
            out_material["normal"].bake_path = path_prefix + "normal";

            // Chain further
            parent_call = lookup_call("layer", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "layer.";
        }
        // Check for a normalized mix. This mix combines the metallic and dielectric parts
        // of the material
        if(semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX)
        {
            // The top-mix component is supposed to be a glossy bsdf
            // Collect metallic weight
            out_material["metallic"].bake_path = path_prefix + "components.value1.weight";

            // And other metallic parameters
            if (is_transmissive_pbr) {
                out_material["roughness"].bake_path = path_prefix + "components.value1.component.roughness_u.s.r.roughness";
                out_material["anisotropy"].bake_path = path_prefix + "components.value1.component.roughness_u.s.r.anisotropy";
                out_material["anisotropy_rotation"].bake_path = path_prefix + "components.value1.component.roughness_u.s.r.rotation";
            }
            else
                out_material["roughness"].bake_path = path_prefix + "components.value1.component.roughness_u";
            // Base_color can be taken from any of the leaf-bsdfs. It is supposed to
            // be the same.
            out_material["base_color"].bake_path = path_prefix + "components.value1.component.tint";

            // Chain further
            parent_call = lookup_call(
                "components.value0.component", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "components.value0.component.";
        }
        if(semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
        {
            // Collect specular parameters
            out_material["specular"].bake_path = path_prefix + "weight";
            if (is_transmissive_pbr)
            {
                out_material["roughness"].bake_path = path_prefix + "layer.roughness_u.s.r.roughness";
                out_material["anisotropy"].bake_path = path_prefix + "layer.roughness_u.s.r.anisotropy";
                out_material["anisotropy_rotation"].bake_path = path_prefix + "layer.roughness_u.s.r.rotation";
            }
            else
            {
                out_material["roughness"].bake_path = path_prefix + "layer.roughness_u";
            }

            // Chain further
            parent_call = lookup_call("base", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "base.";
        }
        if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX)
        {
            check_success(is_transmissive_pbr);

            out_material["transparency"].bake_path = path_prefix + "components.value1.weight";
            out_material["transmission_color"].bake_path = path_prefix + "components.value1.component.tint";
            // Chain further
            parent_call = lookup_call("components.value0.component", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "components.value0.component.";
        }
        if(semantic ==
            mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF)
        {
            if(out_material["metallic"].bake_path.empty())
                out_material["metallic"].value = create_value(transaction, "Float32", 1.0f);
            if(out_material["roughness"].bake_path.empty())
                out_material["roughness"].bake_path = path_prefix + "roughness_u";
            if(out_material["base_color"].bake_path.empty())
                out_material["base_color"].bake_path = path_prefix + "tint";
        }
        else if(semantic ==
            mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF)
        {
            if(out_material["base_color"].bake_path.empty())
                out_material["base_color"].bake_path = path_prefix + "tint";
        }

        // Check for cutout-opacity
        mi::base::Handle<const mi::neuraylib::IExpression> cutout(
            cm->lookup_sub_expression("geometry.cutout_opacity"));
        if(cutout.is_valid_interface())
            out_material["opacity"].bake_path = "geometry.cutout_opacity";
    }
    else if (target_model == "specular_glossy")
    {
        // Setup parameters for the specular - glossy material model
        out_material["base_color"] = Material_parameter("Rgb_fp");
        out_material["f0"] = Material_parameter("Rgb_fp");
        out_material["f0_color"] = Material_parameter("Rgb_fp");
        out_material["f0_refl"] = Material_parameter("Float32");
        out_material["f0_weight"] = Material_parameter("Float32");
        out_material["glossiness"]  = Material_parameter("Float32", rough_to_gloss);
        out_material["opacity"]  = Material_parameter("Float32");
        out_material["normal_map"] = Material_parameter("Float32<3>", remap_normal);

        // Specular-glossy distillation can result in a diffuse bsdf, a glossy bsdf
        // or a curve-weighted combination of both. Explicitly check the cases
        // and save the corresponding bake paths.
        switch(semantic)
        {
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
            out_material["base_color"].bake_path = "surface.scattering.tint";
            out_material["f0_weight"].value =  create_value(transaction, "Float32", 0.0f);
            out_material["f0_color"].value = create_value(transaction, "Color", mi::Color(0.0f));
            break;
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            out_material["f0_color"].bake_path = "surface.scattering.tint";
            out_material["f0_refl"].value = create_value(transaction, "Float32", 1.0f);
            out_material["f0_weight"].value = create_value(transaction, "Float32", 1.0f);
            out_material["glossiness"].bake_path =
                "surface.scattering.roughness_u"; // needs inversion
            break;
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
            out_material["base_color"].bake_path ="surface.scattering.base.tint";

            out_material["f0_color"].bake_path = "surface.scattering.layer.tint";
            out_material["f0_refl"].bake_path = "surface.scattering.normal_reflectivity";
            out_material["f0_weight"].bake_path = "surface.scattering.weight";

            out_material["glossiness"].bake_path =
                "surface.scattering.layer.roughness_u"; // needs inversion

            break;
        default:
            // unknown function, nothing to bake
            break;
        }
        out_material["normal_map"].bake_path = "geometry.normal";
        out_material["opacity"].bake_path = "geometry.cutout_opacity";
    }
}

// If \p value is a texture, add all its u/v pairs and frame number to \p param.
mi::Size build_canvases(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IValue* value,
    Material_parameter& param)
{
    if( value->get_kind() !=  mi::neuraylib::IValue::VK_TEXTURE)
        return 0;

    mi::base::Handle<const mi::neuraylib::IValue_texture> value_texture(
        value->get_interface<mi::neuraylib::IValue_texture>());
    const char* texture_name = value_texture->get_value();
    if( !texture_name)
        return 0;

    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>( texture_name));
    if( !texture)
        return 0;

    const char* image_name = texture->get_image();
    if( !image_name)
        return 0;

    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>( image_name));
    if( !image)
        return 0;

    bool is_animated = image->is_animated();
    bool is_uvtile = image->is_uvtile();
    if (!is_animated && !is_uvtile)
    {
        // Exclude non uvtile and non animated texture from the traversal
        // These are handled later as Material_parameter::[texture, value]
        return 0;
    }

    mi::Size count = 0;
    mi::Size length = image->get_length();
    for( mi::Size i = 0; i < length; ++i) {

        mi::Size frame_number = image->get_frame_number( i);
        mi::Size frame_length = image->get_frame_length( i);
        count += frame_length;

        for( mi::Size j = 0; j < frame_length; ++j) {
            mi::Sint32 u = 0;
            mi::Sint32 v = 0;
            mi::Sint32 rtn = image->get_uvtile_uv( i, j, u, v);
            assert(0 == rtn);
            if (0 == rtn)
            {
                param.canvases[frame_number][Material_parameter::UVTile(u, v)] = NULL;
            }
            else
            {
                std::cerr << "ERROR: uvtile_id is out of range." << std::endl;
            }
        }
    }

    return count;
}

// Adds u/v pairs and frame numbers of all found textures to \p param.
//
// Note that this simple traversal does not keep track of referenced temporaries and parameters
// and traverses them once for each reference.
mi::Size build_canvases(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IExpression* expression,
    const mi::neuraylib::ICompiled_material* cm,
    Material_parameter& param,
    std::vector<bool>& visited_temps)
{
    switch( expression->get_kind()) {
        case mi::neuraylib::IExpression::EK_CONSTANT: {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> constant(
                expression->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<const mi::neuraylib::IValue> value( constant->get_value());
            return build_canvases( transaction, value.get(), param);
        }
        case mi::neuraylib::IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> direct_call(
                expression->get_interface<mi::neuraylib::IExpression_direct_call>());
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                direct_call->get_arguments());
            mi::Size count = 0;
            for( mi::Size i = 0; i < args->get_size(); ++i) {
                mi::base::Handle<const mi::neuraylib::IExpression> arg( args->get_expression( i));
                count += build_canvases( transaction, arg.get(), cm, param, visited_temps);
            }
            return count;
        }
        case mi::neuraylib::IExpression::EK_TEMPORARY: {
            mi::base::Handle<const mi::neuraylib::IExpression_temporary> temporary_ref(
                expression->get_interface<mi::neuraylib::IExpression_temporary>());
            mi::Size index = temporary_ref->get_index();
            // visit every temporary expression only once
            if( visited_temps[index])
                return 0;
            visited_temps[index] = true;
            mi::base::Handle<const mi::neuraylib::IExpression> temporary(
                cm->get_temporary( index));
            return build_canvases( transaction, temporary.get(), cm, param, visited_temps);
        }
        case mi::neuraylib::IExpression::EK_PARAMETER: {
            mi::base::Handle<const mi::neuraylib::IExpression_parameter> parameter_ref(
                expression->get_interface<mi::neuraylib::IExpression_parameter>());
            mi::Size index = parameter_ref->get_index();
            mi::base::Handle<const mi::neuraylib::IValue> parameter(
                cm->get_argument( index));
            return build_canvases( transaction, parameter.get(), param);
        }
        case mi::neuraylib::IExpression::EK_CALL:
            break;
    }

    assert( false);
    return 0;
}

mi::Size build_canvases(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    Material& out_material)
{
    Timing timing("Build canvases");
    mi::Size count = 0;

    std::vector<bool> visited_temps(cm->get_temporary_count(), false);

    for (Material::iterator it = out_material.begin(); it != out_material.end(); ++it)
    {
        Material_parameter& param = it->second;

        // Do not attempt to bake empty paths
        if (param.bake_path.empty())
            continue;

        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            cm->lookup_sub_expression(param.bake_path.c_str()));

        count += build_canvases(transaction, expr.get(), cm, param, visited_temps);
    }

    return count;
}

mi::IData* create_constant(mi::neuraylib::ITransaction* transaction, const std::string & value_type)
{
    if (value_type == "Rgb_fp")
    {
        mi::base::Handle<mi::IColor> v(
            transaction->create<mi::IColor>());
        return v->get_interface<mi::IData>();
    }
    else if (value_type == "Float32<3>")
    {
        mi::base::Handle<mi::IFloat32_3> v(
            transaction->create<mi::IFloat32_3>());
        return v->get_interface<mi::IData>();
    }
    else if (value_type == "Float32")
    {
        mi::base::Handle<mi::IFloat32> v(
            transaction->create<mi::IFloat32>());
        return v->get_interface<mi::IData>();
    }

    std::cout << "Ignoring unsupported value type '" << value_type
        << "'" << std::endl;
    return NULL;
}

// Constructs a material for the target model, extracts the bake paths relevant for this
// model from the compiled material and bakes those paths into textures or constant values
void bake_target_material_inputs(
    mi::neuraylib::Baker_resource baker_resource,
    mi::Uint32 baking_samples,
    mi::Uint32 baking_resolution,
    mi::Float32 min_u,
    mi::Float32 max_u,
    mi::Float32 min_v,
    mi::Float32 max_v,
    bool constant_detection,
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    mi::neuraylib::IMdl_distiller_api* distiller_api,
    mi::neuraylib::IImage_api* image_api,
    Material& out_material)
{
    Timing timing("Baking");
    for(Material::iterator it_mat = out_material.begin();
        it_mat != out_material.end(); ++it_mat)
    {
        Material_parameter& param = it_mat->second;

        // Do not attempt to bake empty paths
        if(param.bake_path.empty())
            continue;

        // Create baker for current path
        mi::base::Handle<const mi::neuraylib::IBaker> baker(distiller_api->create_baker(
            cm, param.bake_path.c_str(), baker_resource));
        check_success(baker.is_valid_interface());

        if(baker->is_uniform())
        {
            // Create constant
            mi::base::Handle<mi::IData> value(create_constant(transaction, param.value_type));
            if (!value)
            {
                continue;
            }

            // Bake constant value
            mi::Sint32 result = baker->bake_constant(value.get());
            check_success(result == 0);

            if(param.remap_func)
                param.remap_func(value.get());

            param.value = value;
        }
        else
        {
            if (!param.canvases.empty())
            {
                Material_parameter::Canvases::iterator it;
                for (it = param.canvases.begin(); it != param.canvases.end(); it++)
                {
                    Material_parameter::Frame_number frame_number(it->first);

                    Material_parameter::Uvtiles::iterator it2;
                    for (it2 = it->second.begin(); it2 != it->second.end(); it2++)
                    {
                        Material_parameter::UVTile uv(it2->first);

                        // Create a canvas
                        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                            image_api->create_canvas(param.value_type.c_str(), baking_resolution, baking_resolution));

                        // Bake texture
                        mi::Float32 min_u(mi::Float32(uv.min_uv.first));
                        mi::Float32 max_u(min_u + 1);
                        mi::Float32 min_v(mi::Float32(uv.min_uv.second));
                        mi::Float32 max_v(min_v + 1);
                        mi::Float32 animation_time = mi::Float32(frame_number);
                        mi::Sint32 result = baker->bake_texture(canvas.get(), min_u, max_u, min_v, max_v, animation_time, baking_samples);
                        check_success(result == 0);

                        if (param.remap_func)
                            param.remap_func(canvas.get());

                        it2->second = canvas;
                    }
                }
            }
            else
            {
                // Create a canvas
                mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                    image_api->create_canvas(param.value_type.c_str(), baking_resolution, baking_resolution));

                bool is_constant = false;
                mi::base::Handle<mi::IData> value;
                mi::Sint32 result = -1;
                if (constant_detection)
                {
                    // Create constant
                    value = create_constant(transaction, param.value_type);
                    result = baker->bake_texture_with_constant_detection(
                        canvas.get(),
                        value.get(),
                        is_constant,
                        min_u, max_u, min_v, max_v, 0.0f/*animation_time*/, baking_samples);
                }
                else
                {
                    result = baker->bake_texture(
                        canvas.get(),
                        min_u, max_u, min_v, max_v, 0.0f/*animation_time*/, baking_samples);
                }
                check_success(result == 0);
                check_success(is_constant ? value : true);

                if (is_constant && value)
                {
                    if (param.remap_func)
                        param.remap_func(value.get());

                    param.value = value;
                    canvas = NULL;
                }
                else
                {
                    if (param.remap_func)
                        param.remap_func(canvas.get());

                    param.texture = canvas;
                    value = NULL;
                }
            }
        }
    }
}

template <typename T, typename U>
void init_value(mi::neuraylib::ICanvas* canvas, mi::IData* value, T*& out_array, U& out_value)
{
    if(canvas)
    {
        mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
        out_array =static_cast<T*>(tile->get_data());
    }
    else if(value)
    {
        mi::get_value(value, out_value);
    }
}

void calculate_f0(mi::neuraylib::ITransaction* trans, Material& material)
{
    // if refl_weight value exists and is zero, set f0 to zero, too
    if(material["f0_weight"].value)
    {
        float v;
        mi::get_value(material["f0_weight"].value.get(), v);

        if(v==0.0f)
        {
            material["f0"].value = create_value(trans, "Color", mi::Color(0.0f));
            material["f0"].texture = 0;
            return;
        }
    }

    mi::Uint32 rx = material["f0"].texture->get_resolution_x();
    mi::Uint32 ry = material["f0"].texture->get_resolution_y();

    mi::Color   f0_color_value(0.0f);
    mi::Float32 f0_weight_value = 0.0f;
    mi::Float32 f0_refl_value = 0.0f;

    mi::Float32_3* f0 = nullptr;
    mi::Float32_3* f0_color = nullptr;
    mi::Float32* f0_weight = nullptr;
    mi::Float32* f0_refl = nullptr;

    init_value(material["f0"].texture.get(), nullptr,
        f0, /* dummy */ f0_color_value);

    init_value(material["f0_color"].texture.get(), material["f0_color"].value.get(),
        f0_color, f0_color_value);
    init_value(material["f0_weight"].texture.get(), material["f0_weight"].value.get(),
        f0_weight, f0_weight_value);
    init_value(material["f0_refl"].texture.get(), material["f0_refl"].value.get(),
        f0_refl, f0_refl_value);

    const mi::Uint32 n =rx * ry;
    for(mi::Uint32 i=0; i<n; ++i)
    {
        const mi::Float32 t = (f0_weight ? f0_weight[i] : f0_weight_value) *
            (f0_refl ? f0_refl[i] : f0_refl_value);

        f0[i][0] = (f0_color ? f0_color[i][0] : f0_color_value[0]) * t;
        f0[i][1] = (f0_color ? f0_color[i][1] : f0_color_value[1]) * t;
        f0[i][2] = (f0_color ? f0_color[i][2] : f0_color_value[2]) * t;
    }
}

// Helper class to export canvases either sequentially or in parallel threads
class Canvas_exporter
{
    bool m_in_parallel = true;
    std::map<std::string/*filename*/, mi::base::Handle<const mi::neuraylib::ICanvas>> m_canvases;

public:
    Canvas_exporter(bool parallel)
        : m_in_parallel(parallel)
    {
    }
    void add_canvas(const std::string& filename, const mi::neuraylib::ICanvas* canvas)
    {
        m_canvases[filename] = mi::base::make_handle_dup(canvas);
    }

    void do_export(mi::neuraylib::IMdl_impexp_api * mdl_impexp_api)
    {
        auto export_canvas = [mdl_impexp_api](const char* filename, const mi::neuraylib::ICanvas* canvas)
        {
            check_success(mdl_impexp_api->export_canvas(filename, canvas) == 0);
        };
        std::vector<std::thread> threads;

        for (auto & canvas_file : m_canvases)
        {
            const char * filename(canvas_file.first.c_str());
            const mi::neuraylib::ICanvas* canvas(canvas_file.second.get());
            if (m_in_parallel)
            {
                threads.emplace_back(
                    std::thread(export_canvas, filename, canvas)
                );
            }
            else
            {
                export_canvas(filename, canvas);
            }
        }
        for (auto & t : threads)
        {
            t.join();
        }
    }
};

// Print some information about baked material parameters to the console and
// save the baked textures to disk
void process_target_material(
    const std::string& target_model,
    const std::string& material_name,
    const Material& material,
    bool save_baked_textures,
    bool parallel,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    Timing timing("Saving");
    std::cout << "--------------------------------------------------------------------------------"
        << std::endl;
    std::cout << "Material model: " << target_model << std::endl;
    std::cout << "--------------------------------------------------------------------------------"
        << std::endl;

    Canvas_exporter canvas_exporter(parallel);
    for(Material::const_iterator it_mat = material.begin();
        it_mat != material.end(); ++it_mat)
    {
        const std::string& param_name = it_mat->first;
        const Material_parameter& param = it_mat->second;

        std::cout << "Parameter: '" << param_name << "': ";
        if(param.bake_path.empty())
        {
            std::cout << " no matching bake path found in target material."
                << std::endl;

            if(param.value)
                std::cout << "--> value set to ";
            if(param.texture)
                std::cout << "--> calculated ";
        }
        else
            std::cout << "path '"<< param.bake_path << "' baked to ";

        if (!param.canvases.empty())
        {
            bool multiple_frames = param.canvases.size() > 1;
            bool multiple_textures_first_frame = multiple_frames || param.canvases.begin()->second.size() > 1;
            std::cout << "texture" << (multiple_textures_first_frame ? "s:" : ":") << std::endl << std::endl;
            Material_parameter::Canvases::const_iterator it_cvs;
            for (it_cvs = param.canvases.begin(); it_cvs != param.canvases.end(); it_cvs++)
            {
                bool multiple_uvtiles = it_cvs->second.size() > 1;

                Material_parameter::Frame_number frame_number = it_cvs->first;

                Material_parameter::Uvtiles::const_iterator it2;
                for (it2 = it_cvs->second.begin(); it2 != it_cvs->second.end(); it2++)
                {
                    Material_parameter::UVTile uv = it2->first;
                    if (!it2->second) // texture
                        continue;

                    if (save_baked_textures)
                    {
                        // write texture to disc
                        // these filenames match the UVTILE0 convention (independent of the convention used for the input)
                        // 0-based uv-tileset, expands to "_u"u"_v"v
                        std::stringstream file_name;
                        file_name << material_name << "-" << param_name;
                        if(multiple_frames)
                            file_name << "_frame" << frame_number;
                        if (multiple_uvtiles)
                            file_name << "_u" << it2->first.min_uv.first << "_v" << it2->first.min_uv.second;
                        file_name << ".png";

                        canvas_exporter.add_canvas(file_name.str(), it2->second.get());
                        std::cout << file_name.str() << std::endl;
                    }
                    else
                    {
                        std::cout << "<Not saved>" << std::endl;
                    }
                }
            }
        }
        else if (param.texture)
        {
            std::cout << "texture:" << std::endl << std::endl;
            if (save_baked_textures)
            {
                // write texture to disc
                std::stringstream file_name;
                file_name << material_name << "-" << param_name << ".png";
                canvas_exporter.add_canvas(file_name.str(), param.texture.get());
                std::cout << file_name.str() << std::endl;
            }
            else
            {
                std::cout << "<Not saved>" << std::endl;
            }
        }
        else if(param.value)
        {
            std::cout << "constant ";
            if(param.value_type == "Rgb_fp")
            {
                mi::base::Handle<mi::IColor> color(
                    param.value->get_interface<mi::IColor>());
                mi::Color c;
                color->get_value(c);
                std::cout << "color ("
                    << c.r << ", " << c.g << ", " << c.b << ")."<< std::endl << std::endl;
            }
            else if(param.value_type == "Float32")
            {
                mi::base::Handle<mi::IFloat32> value(
                    param.value->get_interface<mi::IFloat32>());
                mi::Float32 v;
                value->get_value(v);
                std::cout << "float "  << v << "." << std::endl;
            }
            else if(param.value_type == "Float32<3>")
            {
                mi::base::Handle<mi::IFloat32_3> value(
                    param.value->get_interface<mi::IFloat32_3>());
                mi::Float32_3 v;
                value->get_value(v);
                std::cout << "vector ("
                    << v.x << ", " << v.y << ", " << v.z << ")."<< std::endl << std::endl;
            }
        }
        std::cout
            << "--------------------------------------------------------------------------------"
            << std::endl;
    }

    // Export canvases
    canvas_exporter.do_export(mdl_impexp_api);
}

// Prints program usage
static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1> ...]\n"
        << "-h                      print this text\n"
        << "--target                distilling target:diffuse|ue4|transmissive_pbr|\n"
        << "                        specular_glossy (default: ue4)\n"
        << "--baker_resource        baking device: gpu|cpu|gpu_with_cpu_fallback (default: cpu)\n"
        << "--samples               baking samples (default: 4)\n"
        << "--resolution            baking resolution (default: 1024)\n"
        << "--uv_range              baking UV range: min_u max_u min_v max_v (default: 0.0f 1.0f 0.0f 1.0f)\n"
        << "--material_file <file>  file containing fully qualified names of materials to distill\n"
        << "--do_not_save_textures  if set, avoid saving baked textures to file\n"
        << "--no_constant_detection if set, do not perform constant detection optimization when baking textures\n"
        << "--module <module_name>  distill all materials from the module, can occur multiple times\n"
        << "--no_parallel           do not save texture files in parallel threads\n"
        << "--mdl_path <path>       mdl search path, can occur multiple times.\n"
        << "--plugin <filename>     add additional distiller plugin, can be used more than once.\n";

    exit(EXIT_FAILURE);
}

void load_materials_from_file(const std::string & material_file, std::vector<std::string> & material_names)
{
    std::fstream file;
    file.open(material_file, std::fstream::in);
    if (!file)
    {
        std::cout << "Invalid file: " + material_file;
        return;
    }
    std::string fn;
    while (getline(file, fn))
    {
        material_names.emplace_back(fn);
    }
    file.close();
}

void load_materials_from_modules(
    mi::neuraylib::IMdl_factory * mdl_factory
    , mi::neuraylib::ITransaction * transaction
    , mi::neuraylib::IMdl_impexp_api * mdl_impexp_api
    , const std::vector<std::string> & module_names
    , std::vector<std::string> & material_names)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    for (auto module_name : module_names)
    {
        // Sanity check
        if (module_name.find("::") != 0)
        {
            module_name = std::string("::") + module_name;
        }
        mi::Sint32 rtn(mdl_impexp_api->load_module(transaction, module_name.c_str(), context.get()));
        check_success(rtn == 0 || rtn == 1);
        mi::base::Handle<const mi::IString> db_module_name(
            mdl_factory->get_db_module_name(module_name.c_str()));
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>(db_module_name->get_c_str()));
        check_success(module.is_valid_interface());
        mi::Size material_count = module->get_material_count();
        for (mi::Size i = 0; i < material_count; i++)
        {
            std::string mname(module->get_material(i));
            mi::base::Handle<const mi::neuraylib::IFunction_definition> material(
                transaction->access<mi::neuraylib::IFunction_definition>(mname.c_str()));
            material_names.push_back(material->get_mdl_name());
        }
    }
}

int MAIN_UTF8(int argc, char* argv[])
{
    std::string                     target_model = "ue4";
    mi::neuraylib::Baker_resource   baker_resource = mi::neuraylib::BAKE_ON_CPU;
    mi::Uint32                      baking_samples = 4;
    mi::Uint32                      baking_resolution = 1024;
    bool                            uv_range_set(false);
    mi::Float32                     min_u = 0;
    mi::Float32                     max_u = 1;
    mi::Float32                     min_v = 0;
    mi::Float32                     max_v = 1;
    bool                            parallel = true;
    std::vector<std::string>        material_names;
    std::vector<std::string>        module_names;
    std::string material_file;
    std::vector<std::string>        additional_plugins;
    bool save_baked_textures(true);
    // By default optimize the process of baking textures by detecting constant colors
    bool constant_detection(true);

    mi::examples::mdl::Configure_options configure_options;

    // Collect command line arguments, if any
    for (int i = 1; i < argc; ++i) {
        const char *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--mdl_path") == 0) {
                if (i < argc - 1)
                    configure_options.additional_mdl_paths.push_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else  if (strcmp(opt, "--plugin") == 0) {
                if (i < argc - 1)
                    additional_plugins.push_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--target") == 0) {
                if (i < argc - 1)
                    target_model = argv[++i];
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--baker_resource") == 0) {
                if (i < argc - 1) {
                    std::string res = argv[++i];
                    if (res == "gpu")
                        baker_resource = mi::neuraylib::BAKE_ON_GPU;
                    else if (res == "gpu_with_cpu_fallback")
                        baker_resource = mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK;
                    else if (res != "cpu")
                        usage(argv[0]);
                }
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--samples") == 0) {
                if (i < argc - 1)
                {
                    int val(atoi(argv[++i]));
                    if (val > 0)
                        baking_samples = val;
                    else
                        std::cout << "Invalid number of samples ignored\n";
                }
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--resolution") == 0) {
                if (i < argc - 1)
                {
                    int val(atoi(argv[++i]));
                    if (val > 0)
                        baking_resolution = val;
                    else
                        std::cout << "Invalid resolution ignored\n";
                }
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--uv_range") == 0)
            {
                mi::Float32 uv_range[4];
                for (int idx = 0; idx < 4; idx++)
                {
                    if (i < argc - 1)
                    {
                        mi::Float32 val;
                        int ok = sscanf(argv[++i], "%f", &val);
                        if (ok != 1)
                        {
                            std::cout << "Invalid UV range\n";
                            usage(argv[0]);
                        }
                        uv_range[idx] = val;
                    }
                    else
                    {
                        std::cout << "Invalid UV range\n";
                        usage(argv[0]);
                    }
                }
                min_u = uv_range[0];
                max_u = uv_range[1];
                min_v = uv_range[2];
                max_v = uv_range[3];
                uv_range_set = true;
            }
            else if (strcmp(opt, "--material_file") == 0) {
                if (i < argc - 1)
                    material_file = argv[++i];
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--do_not_save_textures") == 0) {
                save_baked_textures = false;
            }
            else if (strcmp(opt, "--no_constant_detection") == 0) {
                constant_detection = false;
            }
            else if (strcmp(opt, "--no_parallel") == 0) {
                parallel = false;
            }
            else if (strcmp(opt, "--module") == 0) {
                if (i < argc - 1)
                    module_names.emplace_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else
                usage(argv[0]);
        }
        else
            material_names.push_back(opt);
    }
    if (!material_file.empty())
        load_materials_from_file(material_file, material_names);

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Load the distilling plugin
    if (mi::examples::mdl::load_plugin(neuray.get(), "mdl_distiller" MI_BASE_DLL_FILE_EXT) != 0)
        exit_failure("Failed to load the mdl_distiller plugin.");

    // Load any additional distilling plugins given on the command line.
    for (auto& plugin : additional_plugins) {
        std::string plugin_filename(plugin);
        plugin += MI_BASE_DLL_FILE_EXT;
        if (mi::examples::mdl::load_plugin(neuray.get(), plugin.c_str()) != 0)
            exit_failure("Failed to load the %s plugin.", plugin.c_str());
    }

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        // Get MDL factory
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        if (!module_names.empty())
        {
            load_materials_from_modules(mdl_factory.get(), transaction.get(), mdl_impexp_api.get(), module_names, material_names);
        }
        if (material_names.empty())
        {
            material_names.push_back(
                "::nvidia::sdk_examples::tutorials_distilling::example_distilling1");
        }
        for (const auto& m : material_names)
        {
            // split module and material name
            std::string module_qualified_name, material_simple_name;
            if (!mi::examples::mdl::parse_cmd_argument_material_name(
                m, module_qualified_name, material_simple_name, true))
                exit_failure();

            // Create an execution context
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            // Load mdl module and create a material instance
            mi::base::Handle<mi::neuraylib::IFunction_call> instance(
                create_material_instance(
                    mdl_factory.get(),
                    transaction.get(),
                    mdl_impexp_api.get(),
                    context.get(),
                    module_qualified_name,
                    material_simple_name));

            // Compile the material instance
            mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
                compile_material_instance(
                    mdl_factory.get(),
                    transaction.get(),
                    instance.get(),
                    context.get(),
                    false));

            // Acquire distilling API used for material distilling and baking
            mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distilling_api(
                neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

            // Distill compiled material to diffuse/glossy material model
            mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_material(
                create_distilled_material(
                    distilling_api.get(),
                    compiled_material.get(),
                    target_model.c_str()));

            // Acquire image API needed to create a canvas for baking
            mi::base::Handle<mi::neuraylib::IImage_api> image_api(
                neuray->get_api_component<mi::neuraylib::IImage_api>());

            Material out_material;
            // Setup result material parameters relevant for target_model
            // and collect bake paths
            setup_target_material(
                target_model,
                transaction.get(),
                distilled_material.get(),
                out_material);

            // Search for UV textures, number of frames, ...
            // number of frames is relevant to animated textures.
            mi::Size canvas_count = build_canvases(
                transaction.get(),
                distilled_material.get(),
                out_material);

            if (canvas_count > 0 && uv_range_set)
            {
                std::cerr << "WARNING: UV range will be ignored for UV tile textured parameters\n";
            }

            // Bake material inputs
            bake_target_material_inputs(
                baker_resource,
                baking_samples,
                baking_resolution,
                min_u,
                max_u,
                min_v,
                max_v,
                constant_detection,
                transaction.get(),
                distilled_material.get(),
                distilling_api.get(),
                image_api.get(),
                out_material);

            if (target_model == "specular_glossy")
            {
                // the specular glossy models f0 parameter cannot
                // be directly taken from the distilling result but
                // needs to be calculated

                // Create f0 canvas
                out_material["f0"].texture =
                    image_api->create_canvas("Rgb_fp", baking_resolution, baking_resolution);

                // fill it
                calculate_f0(transaction.get(), out_material);
            }
            // Process resulting material, in this case we simply
            // print some information about the baked parameters
            // and save the textures to disk, if any
            process_target_material(
                target_model,
                material_simple_name,
                out_material,
                save_baked_textures,
                parallel,
                mdl_impexp_api.get());
        }
        transaction->commit();
    }

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
