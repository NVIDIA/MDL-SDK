/******************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/distilling_unity/example_distilling_unity.cpp
//
// Introduces the distillation of mdl materials to a fixed target model
// and showcases how to bake material paths to a texture

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <thread>
#include <set>

#include "example_shared.h"
#include "example_distilling_shared.h"
#include "example_cuda_shared.h"
#include "example_distilling_unity.h"

// Lookup tables for baking oversampling
float RADINV2[] = { 0, 0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f, 0.5625f, 0.3125f, 0.8125f, 0.1875f, 0.6875f, 0.4375f };
float RADINV3[] = { 0, 0.333333f, 0.666667f, 0.111111f, 0.444444f, 0.777778f, 0.222222f, 0.555556f, 0.888889f, 0.037037f, 0.37037f, 0.703704f, 0.148148f, 0.481481f, 0.814815f };

class Logger;
mi::base::Handle<Logger> the_logger;
class Logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    Logger(int level)
    {
        set_verbosity_level(level);
    }

    void message(
        mi::base::Message_severity level,
        const char* /*module_category*/,
        const mi::base::Message_details& /*details*/,
        const char* message)
    {
        if (level > m_level)
        {
            return;
        }
        const char* severity = 0;
        switch (level) {
        case mi::base::MESSAGE_SEVERITY_FATAL:        severity = "fatal: "; break;
        case mi::base::MESSAGE_SEVERITY_ERROR:        severity = "error: "; break;
        case mi::base::MESSAGE_SEVERITY_WARNING:      severity = "warn:  "; break;
        case mi::base::MESSAGE_SEVERITY_INFO:         severity = "info:  "; break;
        case mi::base::MESSAGE_SEVERITY_VERBOSE:      severity = "verbose:  "; break;
        case mi::base::MESSAGE_SEVERITY_DEBUG:        severity = "debug:  "; break;
        case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT: return;
        }

        fprintf(stderr, "%s", severity);
        fprintf(stderr, "%s", message);
        putc('\n', stderr);

#ifdef MI_PLATFORM_WINDOWS
        fflush(stderr);
#endif
    }

    void set_verbosity_level(int level)
    {
        if (level >= 0 && level <= 6)
        {
            m_level = level;
        }
    }
public:
    void log(mi::base::Message_severity level, const std::string & str)
    {
        message(level, "Distilling"/*module_category*/, mi::base::Message_details(), str.c_str());
    }

private:
    // Logging level up to which messages are reported
    // level = 0 will disable all logging
    // level >= 1 logs fatal messages
    // level >= 2 logs error in addition
    // level >= 3 logs warning in addition
    // level >= 4 logs info in addition
    // level >= 5 logs verbose in addition
    // level = 6 logs debug in addition
    int  m_level = 0;
};

// Custom Timing output with a logger
#include <chrono>
#include <string>
struct LoggerTiming
{
    explicit LoggerTiming(std::string operation)
        : m_operation(operation)
    {
        m_start = std::chrono::steady_clock::now();
    }

    ~LoggerTiming()
    {
        auto stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = stop - m_start;
        std::stringstream strStream;
        strStream << "Finished '" << m_operation << "' after " << elapsed_seconds.count()
            << " seconds.";
        the_logger->log(mi::base::MESSAGE_SEVERITY_INFO, strStream.str());
    }

private:
    std::string m_operation;
    std::chrono::steady_clock::time_point m_start;
};

// Small struct used to store the result of a texture baking process
// of a material sub expression
struct Material_parameter
{
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

class Material : public std::map<std::string, Material_parameter>
{
public:
    // Base Map
    // - RGB: Stores the base color
    // - Alpha : Stores the opacity
    mi::neuraylib::ICanvas * get_base_color_map(mi::neuraylib::IImage_api* image_api, mi::Uint32 resolution)
    {
        if (!m_base_color_map.is_valid_interface())
        {
            m_base_color_map = mi::base::Handle<mi::neuraylib::ICanvas>(image_api->create_canvas("Color", resolution, resolution));
        }
        m_base_color_map->retain();
        return m_base_color_map.get();
    }

    bool has_base_color_map() const
    {
        return m_base_color_map.is_valid_interface();
    }

    // Mask map
    // A Texture that packs different Material maps into each of its RGBA channels
    // - Red : Stores the metallic map
    // - Green : Stores the ambient occlusion map
    // - Blue : Stores the detail Mask Map
    // - Alpha : Stores the smoothness map
    mi::neuraylib::ICanvas * get_mask_map(mi::neuraylib::IImage_api* image_api, mi::Uint32 resolution)
    {
        if(! m_mask_map.is_valid_interface())
        {
            m_mask_map = mi::base::Handle<mi::neuraylib::ICanvas>(image_api->create_canvas("Color", resolution, resolution));
        }
        m_mask_map->retain();
        return m_mask_map.get();
    }

    bool has_mask_map() const
    {
        return m_mask_map.is_valid_interface();
    }

    Material_parameter * find_parameter(const std::string & name)
    {
        std::map<std::string, Material_parameter>::iterator it(find(name));
        if (it != end())
        {
            return &(it->second);
        }
        return NULL;
    }

    bool is_base_map_parm(const std::string& param_name)
    {
        return (param_name == "base_color" || param_name == "transparency" || param_name == "opacity");
    }

    bool is_mask_map_parm(const std::string& param_name)
    {
        return (param_name == "metallic" || param_name == "roughness");
    }

    mi::neuraylib::ICanvas * get_texture_for_parameter(const std::string & param_name, mi::neuraylib::IImage_api* image_api, mi::Uint32 resolution)
    {
        if (is_base_map_parm(param_name))
        {
            return get_base_color_map(image_api, resolution);
        }
        else if (is_mask_map_parm(param_name))
        {
            return get_mask_map(image_api, resolution);
        }
        else
        {
            Material_parameter * p(find_parameter(param_name));
            if (p)
            {
                return image_api->create_canvas(p->value_type.c_str(), resolution, resolution);
            }
        }
        return NULL;
    }

private:
    mi::base::Handle<mi::neuraylib::ICanvas> m_base_color_map;
    mi::base::Handle<mi::neuraylib::ICanvas> m_mask_map;
};

// Log the messages of the given context.
// Returns true, if the context does not contain any error messages, false otherwise.
bool print_messages_local(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0; i < context->get_messages_count(); ++i) {

        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        the_logger->log(message->get_severity(), message->get_string());
    }
    return context->get_error_messages_count() == 0;
}

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
        material_instance->get_interface<const mi::neuraylib::IMaterial_instance>());
    mi::neuraylib::ICompiled_material* compiled_material =
        material_instance2->create_compiled_material(flags, context);
    check_success(print_messages_local(context));

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
    mi::Float32* data = static_cast<mi::Float32*>(tile->get_data());

    const mi::Uint32 n = canvas->get_resolution_x() * canvas->get_resolution_y() * 3;
    for(mi::Uint32 i=0; i<n; ++i)
    {
        data[i] = (data[i] + 1.f) * 0.5f;
    }
}

// Setup material parameters and collect relevant bake paths
void setup_target_material(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    Material& out_material)
{
    // Access surface.scattering function
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> parent_call(
        lookup_call("surface.scattering", cm));
    // ... and get its semantic
    mi::neuraylib::IFunction_definition::Semantics semantic(
        get_call_semantic(transaction, parent_call.get()));

    // Setup some material parameters
    out_material["base_color"] = Material_parameter("Color");
    out_material["metallic"] = Material_parameter("Float32");
    out_material["specular"] = Material_parameter("Float32");
    out_material["roughness"] = Material_parameter("Float32");
    out_material["normal"] = Material_parameter("Float32<3>", remap_normal);

    out_material["clearcoat_weight"] = Material_parameter("Float32");
    out_material["clearcoat_roughness"] = Material_parameter("Float32");
    out_material["clearcoat_normal"] = Material_parameter("Float32<3>", remap_normal);

    out_material["opacity"] = Material_parameter("Float32");

    std::string path_prefix = "surface.scattering.";

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

    // Check for a clearcoat layer, first. If present, it is the outermost layer
    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
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
    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_WEIGHTED_LAYER)
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
    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX)
    {
        // The top-mix component is supposed to be a glossy bsdf
        // Collect metallic weight
        out_material["metallic"].bake_path = path_prefix + "components.value1.weight";

        // And other metallic parameters
        out_material["roughness"].bake_path = path_prefix + "components.value1.component.roughness_u.s.r.roughness";
        out_material["anisotropy"].bake_path = path_prefix + "components.value1.component.roughness_u.s.r.anisotropy";
        out_material["anisotropy_rotation"].bake_path = path_prefix + "components.value1.component.roughness_u.s.r.rotation";

        // Base_color can be taken from any of the leaf-bsdfs. It is supposed to
        // be the same.
        out_material["base_color"].bake_path = path_prefix + "components.value1.component.tint";

        // Chain further
        parent_call = lookup_call(
            "components.value0.component", cm, parent_call.get());
        semantic = get_call_semantic(transaction, parent_call.get());
        path_prefix += "components.value0.component.";
    }
    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
    {
        // Collect specular parameters
        out_material["specular"].bake_path = path_prefix + "weight";
        out_material["roughness"].bake_path = path_prefix + "layer.roughness_u.s.r.roughness";
        out_material["anisotropy"].bake_path = path_prefix + "layer.roughness_u.s.r.anisotropy";
        out_material["anisotropy_rotation"].bake_path = path_prefix + "layer.roughness_u.s.r.rotation";

        // Chain further
        parent_call = lookup_call("base", cm, parent_call.get());
        semantic = get_call_semantic(transaction, parent_call.get());
        path_prefix += "base.";
    }
    if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX)
    {
        out_material["transparency"].bake_path = path_prefix + "components.value1.weight";
        out_material["transmission_color"].bake_path = path_prefix + "components.value1.component.tint";
        // Chain further
        parent_call = lookup_call("components.value0.component", cm, parent_call.get());
        semantic = get_call_semantic(transaction, parent_call.get());
        path_prefix += "components.value0.component.";
    }
    if (semantic ==
        mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF)
    {
        if (out_material["metallic"].bake_path.empty())
            out_material["metallic"].value = create_value(transaction, "Float32", 1.0f);
        if (out_material["roughness"].bake_path.empty())
            out_material["roughness"].bake_path = path_prefix + "roughness_u";
        if (out_material["base_color"].bake_path.empty())
            out_material["base_color"].bake_path = path_prefix + "tint";
    }
    else if (semantic ==
        mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF)
    {
        if (out_material["base_color"].bake_path.empty())
            out_material["base_color"].bake_path = path_prefix + "tint";
    }

    // Check for cutout-opacity
    mi::base::Handle<const mi::neuraylib::IExpression> cutout(
        cm->lookup_sub_expression("geometry.cutout_opacity"));
    if (cutout.is_valid_interface())
        out_material["opacity"].bake_path = "geometry.cutout_opacity";
}

// Function_descriptions
// Helper class which holds a vector of Target_function_description
// and allows to query a specific function description for a given expression (aka description) or for a given parameter name
class Function_descriptions : public std::vector<mi::neuraylib::Target_function_description>
{
public:
    void add_function(const std::string & parameter_name, const std::string & description)
    {
        if (!description.empty())
        {
            m_parms[parameter_name] = description;
            emplace_back(mi::neuraylib::Target_function_description(description.c_str()));
            // TODO Set a name for each fct?
        }
    }

    bool get_function_description(const std::string & description, mi::neuraylib::Target_function_description & function_description) const
    {
        for (auto & fd : *this)
        {
            if (fd.path == description)
            {
                function_description = fd;
                return true;
            }
        }
        return false;
    }

    bool get_function_description_from_parameter(const std::string & parameter_name, mi::neuraylib::Target_function_description & function_description) const
    {
        Parameter_map::const_iterator it(m_parms.find(parameter_name));
        if (it != m_parms.end())
        {
            return get_function_description(it->second, function_description);
        }
        return false;
    }

private:
    typedef std::map<std::string/*parameter_name*/, std::string /*description*/> Parameter_map;
    Parameter_map m_parms;
};

class Baker
{
public:
    Baker(
        mi::neuraylib::Baker_resource baker_resource
        , mi::neuraylib::ITransaction* transaction
        , const mi::neuraylib::ICompiled_material* material
        , Material& out_material
        , mi::neuraylib::IMdl_execution_context * context
        , mi::neuraylib::IMdl_backend_api* mdl_backend_api
    )
        : m_out_material(out_material)
        , m_transaction(mi::base::make_handle_dup(transaction))
        , m_context(mi::base::make_handle_dup(context))
        , m_mdl_backend_api(mi::base::make_handle_dup(mdl_backend_api))
    {
        // Determine how to bake: CPU || GPU || GPU with fallback
        init_baker_resource(baker_resource);

        // Initialize function descriptions
        init_function_descriptions();

        // Build, compile, code to bake params
        build_baker_programs(material);
    }

    bool bake(
        mi::neuraylib::IImage_api* image_api
        , mi::Uint32 baking_samples
        , mi::Uint32 baking_resolution
        , bool parallel
    ) const
    {
        // Prepare for baking
        Parm_list plist;
        pre_bake(plist);

        if (m_bake_gpu)
        {
            if (bake_cuda_ptx(image_api, baking_samples, baking_resolution, plist))
            {
                return true;
            }
            // Will bake on CPU below if GPU fails and if BAKE_ON_GPU_WITH_CPU_FALLBACK was chosen
        }
        if (m_bake_cpu)
        {
            if (bake_native(image_api, baking_samples, baking_resolution, plist))
            {
                return true;
            }
        }
        return false;
    }

private:
    Function_descriptions m_descs;
    Material & m_out_material;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> m_mdl_backend_api;
    bool m_bake_gpu = false;
    bool m_bake_cpu = false;
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_native_code;
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_gpu_code;
    // The CUDA device ID.
    int m_cuda_device = 0;

private:
    // Helper to sort the parameters to bake.
    // This is particularly useful for GPU baking.
    struct Parms
    {
        Parms(const std::string & name, Expression_type expr_type)
            : m_name(name), m_expr_type(expr_type)
        {}
        std::string m_name;
        Expression_type m_expr_type = EXT_UNKNOWN;
    };

    class Parm_list : public std::vector<Parms>
    {
        std::set<std::string> m_added;
    public:
        void add_parm(const std::string & name, Expression_type expr_type = EXT_UNKNOWN)
        {
            if (m_added.find(name) == m_added.end())
            {
                m_added.insert(name);
                emplace_back(Parms(name, expr_type));
            }
        }
        void add_parms(const Material & m)
        {
            for (const auto & p : m)
            {
                add_parm(p.first);
            }
        }
    };

    // Build ordered list of material parameters
    void pre_bake(Parm_list & plist) const
    {
        // WARNING: Order of the following maps is important
        plist.add_parm("roughness", EXT_ROUGHNESS);
        plist.add_parm("metallic", EXT_METALLIC);
        plist.add_parm("base_color", EXT_BASE_COLOR);
        plist.add_parm("transparency", EXT_TRANSPARENCY);
        plist.add_parm("opacity", EXT_OPACITY);
        // Add all other parameters
        plist.add_parms(m_out_material);
    }

private:
    bool bake_native(
        mi::neuraylib::IImage_api* image_api
        , const mi::Uint32& baking_samples
        , const mi::Uint32& baking_resolution
        , const Parm_list& plist
    ) const
    {
        auto bake_native_function = [](
            size_t index
            , mi::Size num_rows_per_frag
            , mi::Size num_frags_with_extra_row
            , mi::Uint32 tex_width
            , mi::neuraylib::ICanvas* texture
            , mi::Uint32 num_samples
            , mi::Float32 du
            , mi::Float32 dv
            , const mi::neuraylib::ITarget_code* code
            , const mi::neuraylib::Target_function_description& function_description
            , const mi::Uint32& components_per_pixel
            , Expression_type expr_type
            , float metallic
            )
        {
            mi::Uint32 start_row = mi::Uint32(
                index * num_rows_per_frag +
                ((index < num_frags_with_extra_row) ? index : num_frags_with_extra_row));

            mi::Uint32 end_row = mi::Uint32(
                start_row + num_rows_per_frag - 1 +
                ((index < num_frags_with_extra_row) ? 1 : 0));

            mi::Uint32 start_col = 0;
            mi::Uint32 end_col = tex_width - 1;

            mi::neuraylib::Shading_state_material state;

            state.normal = mi::Float32_3(0, 0, 1);
            state.geom_normal = mi::Float32_3(0, 0, 1);
            //state.position        = mi::Float32_3(0, 0, 0);
            state.animation_time = 0;
            state.ro_data_segment = 0;

            const int max_tex_spaces = 4;
            mi::Float32_3 tex_coords[max_tex_spaces];
            mi::Float32_3 tangent_u[max_tex_spaces];
            mi::Float32_3 tangent_v[max_tex_spaces];

            state.text_coords = tex_coords;
            state.tangent_u = tangent_u;
            state.tangent_v = tangent_v;

            for (int ts = 0; ts < max_tex_spaces; ++ts)
            {
                tangent_u[ts] = mi::Float32_3(1, 0, 0);
                tangent_v[ts] = mi::Float32_3(0, 1, 0);
            }

            // text result are currently unused
            state.text_results = 0;

            // we have no uniform state here
            mi::Float32_4_4 world_to_obj(1.0f);
            mi::Float32_4_4 obj_to_world(1.0f);

            state.world_to_object = &world_to_obj[0];
            state.object_to_world = &obj_to_world[0];
            state.object_id = 0;

            mi::base::Handle<mi::neuraylib::ITile> tile(texture->get_tile());
            mi::Float32_4 pixel_data;
            mi::base::Atom32 failure;

            float* data = static_cast<float*>(tile->get_data());
            for (mi::Uint32 i = start_row; i <= end_row; i++)
            {
                for (mi::Uint32 j = start_col; j <= end_col; j++)
                {
                    mi::Float32_4 pixel(0.0f, 0.0f, 0.0f, 1.0f);
                    pixel_data = pixel;

                    check_success(num_samples <= 16);
                    for (mi::Uint32 k = 0; k < num_samples; k++)
                    {
                        const mi::Float32 y = (i + mi::math::frac(RADINV3[k] + 0.5f)) * dv;
                        const mi::Float32 x = (j + mi::math::frac(RADINV2[k] + 0.5f)) * du;

                        state.position = mi::Float32_3(x, y, 0);
                        for (int ts = 0; ts < max_tex_spaces; ++ts)
                        {
                            tex_coords[ts] = mi::Float32_3(x, y, 0);
                        }

                        if (code->execute(
                            function_description.function_index, state, nullptr, nullptr, (mi::Float32*)&pixel.x) != 0)
                        {
                            failure.swap(1);
                            return;
                        }
                        pixel_data += pixel;
                    }
                    pixel_data /= num_samples;

                    if (expr_type == EXT_BASE_COLOR)
                    {
                        // Gamma correction
                        const float gammainv(1 / 2.2f);
                        pixel_data.x = powf(pixel_data.x, gammainv);
                        pixel_data.y = powf(pixel_data.y, gammainv);
                        pixel_data.z = powf(pixel_data.z, gammainv);
                        pixel_data.w = 1.0f;
                        tile->set_pixel(j, i, (mi::Float32*)&pixel_data.x);
                    }
                    else if (expr_type == EXT_TRANSPARENCY)
                    {
                        mi::Float32* const pixel = data + (j + i * static_cast<mi::Size>(tex_width)) * components_per_pixel;
                        pixel[3] *= 1 - pixel_data.x;
                    }
                    else if (expr_type == EXT_OPACITY)
                    {
                        mi::Float32* const pixel = data + (j + i * static_cast<mi::Size>(tex_width)) * components_per_pixel;
                        pixel[3] *= pixel_data.x;
                    }
                    else if (expr_type == EXT_METALLIC)
                    {
                        mi::Float32* const pixel = data + (j + i * static_cast<mi::Size>(tex_width)) * components_per_pixel;
                        // Red : Stores the metallic map.
                        pixel[0] = pixel_data.x;
                        // Green : Stores the ambient occlusion map.
                        pixel[1] = 1.0f;
                    }
                    else if (expr_type == EXT_ROUGHNESS)
                    {
                        mi::Float32* const pixel = data + (j + i * static_cast<mi::Size>(tex_width)) * components_per_pixel;
                        // Alpha: Stores the smoothness map.
                        // smoothness = 1 - sqrt(roughness)
                        pixel[3] = 1 - sqrtf(pixel_data.x);
                        // Green : Stores the ambient occlusion map.
                        pixel[1] = 1.0f;
                        // If metallic has no bake_path, need to set its value here
                        if (metallic > 0)
                        {
                            pixel[0] = metallic;
                        }
                    }
                    else
                    {
                        tile->set_pixel(j, i, (mi::Float32*)&pixel_data.x);
                    }
                }
            }
        };

        if (!m_native_code)
        {
            return false;
        }

        float metallic(-1); // no need to set metallic unless bake_path is not set
        {
            Material_parameter* param = m_out_material.find_parameter("metallic");
            if (param && param->bake_path.empty())
            {
                if (param->value_type == "Float32" && param->value)
                {
                    mi::base::Handle<mi::IFloat32> value(param->value->get_interface<mi::IFloat32>());
                    value->get_value(metallic);
                }
            }
        }

        // Determine whether base map is uniform
        std::set<std::string> base_map_parm = {"base_color", "transparency" , "opacity"};
        bool is_base_map_uniform(parms_are_uniform(base_map_parm, m_native_code.get()));

        // Determine whether Mask Map is uniform
        std::set<std::string> mask_map_parm = {"metallic", "roughness"};
        bool is_mask_map_uniform(parms_are_uniform(mask_map_parm, m_native_code.get()));

        for (auto& p : plist)
        {
            Material_parameter* param = m_out_material.find_parameter(p.m_name);
            if (!param)
                continue;

            mi::neuraylib::Target_function_description function_description;
            if (!m_descs.get_function_description(param->bake_path, function_description))
            {
                // Did not find function description
                continue;
            }

            mi::Uint32 samples(baking_samples);
            mi::Uint32 resolution(baking_resolution);
            // Determine if need to bake texture or constant
            bool uniform(is_uniform(m_native_code.get(), function_description.function_index));
            if (m_out_material.is_base_map_parm(p.m_name))
            {
                uniform = is_base_map_uniform;
            }
            else if (m_out_material.is_mask_map_parm(p.m_name))
            {
                uniform = is_mask_map_uniform;
            }
            if (uniform)
            {
                // 1 pixel canvas
                samples = 1;
                resolution = 1;
            }

            // Bake texture for expression
            mi::base::Handle<mi::neuraylib::ICanvas> texture(m_out_material.get_texture_for_parameter(p.m_name, image_api, resolution));
            if (!texture)
                continue;

            Expression_type expr_type(p.m_expr_type);

            const mi::Uint32 tex_width(texture->get_resolution_x());
            const mi::Uint32 tex_height(texture->get_resolution_y());
            const mi::Float32 du((mi::Float32)(1.0 / tex_width));
            const mi::Float32 dv((mi::Float32)(1.0 / tex_height));
            const mi::Size num_fragments(tex_height);
            const mi::Size num_rows_per_frag(tex_height / (int)num_fragments);
            const mi::Size num_frags_with_extra_row(tex_height - num_rows_per_frag * (int)num_fragments);
            const mi::Uint32 components_per_pixel(get_components_per_pixel(mi::base::make_handle(texture->get_tile())->get_type()));

            std::vector<std::thread> threads;
            for (size_t index = 0; index < num_fragments; index++)
            {
                threads.emplace_back(std::thread(
                    bake_native_function
                    , index
                    , num_rows_per_frag
                    , num_frags_with_extra_row
                    , tex_width
                    , texture.get()
                    , samples
                    , du
                    , dv
                    , m_native_code.get()
                    , function_description
                    , components_per_pixel
                    , expr_type
                    , metallic
                ));
            }

            for (auto& t : threads)
            {
                t.join();
            }

            if (param->remap_func)
                param->remap_func(texture.get());

            param->texture = texture;
        }

        return true;
    }

    mi::Uint32 get_components_per_pixel(const std::string & pixel_type) const
    {
        if (pixel_type == "Rgb_fp")
        {
            return 3;
        }
        else if (pixel_type == "Float32")
        {
            return 1;
        }
        else if (pixel_type == "Float32<3>")
        {
            return 3;
        }
        else if (pixel_type == "Color")
        {
            return 4;
        }
        return 0;
    }

    bool bake_cuda_ptx(
        mi::neuraylib::IImage_api* image_api
        , const mi::Uint32 & baking_samples
        , const mi::Uint32 & baking_resolution
        , const Parm_list & plist
    ) const
    {
        auto bake_cuda_ptx_function = [](
            mi::neuraylib::ICanvas* texture
            , const mi::Uint32 & baking_samples
            , const mi::Uint32 & components_per_pixel
            , CUdeviceptr & device_outbuf
            , CUdeviceptr & device_tc_data_list
            , CUdeviceptr & device_arg_block_list
            , CUfunction & cuda_function
            , const mi::neuraylib::Target_function_description & function_description
            , Expression_type expr_type
            , float metallic
            )
        {
            int res_x = texture->get_resolution_x();
            int res_y = texture->get_resolution_y();
            int num_samples = baking_samples;
            size_t function_index(function_description.function_index);
            unsigned int cpp(components_per_pixel);

            // Launch kernel for the whole image
            dim3 threads_per_block(16, 16);
            dim3 num_blocks((res_x + 15) / 16, (res_y + 15) / 16);
            void *kernel_params[] = {
                &device_outbuf,
                &device_tc_data_list,
                &device_arg_block_list,
                &res_x,
                &res_y,
                &num_samples,
                &function_index,
                &cpp,
                &expr_type,
                &metallic
            };

            check_cuda_success(cuLaunchKernel(
                cuda_function,
                num_blocks.x, num_blocks.y, num_blocks.z,
                threads_per_block.x, threads_per_block.y, threads_per_block.z,
                0, nullptr, kernel_params, nullptr));

            mi::base::Handle<mi::neuraylib::ITile> tile(texture->get_tile());
            float *data = static_cast<float *>(tile->get_data());
            check_cuda_success(cuMemcpyDtoH(
                data, device_outbuf, res_x * res_y * sizeof(float) * components_per_pixel));
        };

        check_success(m_transaction.is_valid_interface());
        if (!m_gpu_code)
        {
            return false;
        }

        CUcontext cuda_context = init_cuda(m_cuda_device);
        {
            // Build the full CUDA kernel with all the generated code
            CUfunction  cuda_function;
            char const *ptx_name = "example_distilling_unity.ptx";

            std::vector <mi::base::Handle<const mi::neuraylib::ITarget_code>> target_codes;
            target_codes.emplace_back(m_gpu_code);

            CUmodule    cuda_module = build_linked_kernel(
                target_codes,
                (mi::examples::io::get_executable_folder() + "/" + ptx_name).c_str(),
                "evaluate_mat_expr",
                &cuda_function);

            // Prepare the needed data of all target codes for the GPU
            std::vector<size_t> arg_block_indices;
            for (auto & d : m_descs)
            {
                arg_block_indices.emplace_back(d.argument_block_index);
            }
            Material_gpu_context material_gpu_context(false/*options.enable_derivatives*/);
            for (size_t i = 0, num_target_codes = target_codes.size(); i < num_target_codes; ++i)
            {
                if (!material_gpu_context.prepare_target_code_data(
                    m_transaction.get(), image_api, target_codes[i].get(), arg_block_indices))
                {
                    return false;
                }
            }
            CUdeviceptr device_tc_data_list = material_gpu_context.get_device_target_code_data_list();
            CUdeviceptr device_arg_block_list =
                material_gpu_context.get_device_target_argument_block_list();

            // Allocate GPU output buffer
            int res_x = baking_resolution;
            int res_y = baking_resolution;

            // Create an output buffer large enough to contain data for all the possible expressions
            CUdeviceptr device_outbuf;
            check_cuda_success(cuMemAlloc(&device_outbuf, res_x * res_y * sizeof(float4)));

            float metallic(-1); // no need to set metallic unless bake_path is not set
            {
                Material_parameter * param = m_out_material.find_parameter("metallic");
                if (param && param->bake_path.empty())
                {
                    if (param->value_type == "Float32" && param->value)
                    {
                        mi::base::Handle<mi::IFloat32> value(param->value->get_interface<mi::IFloat32>());
                        value->get_value(metallic);
                    }
                }
            }

            // Determine whether base map is uniform
            std::set<std::string> base_map_parm = {"base_color", "transparency" , "opacity"};
            bool is_base_map_uniform(parms_are_uniform(base_map_parm, m_gpu_code.get()));

            // Determine whether Mask Map is uniform
            std::set<std::string> mask_map_parm = {"metallic", "roughness"};
            bool is_mask_map_uniform(parms_are_uniform(mask_map_parm, m_gpu_code.get()));

            for(auto & p : plist)
            {
                Material_parameter * param = m_out_material.find_parameter(p.m_name);
                if (!param)
                    continue;

                mi::neuraylib::Target_function_description function_description;
                if (!m_descs.get_function_description(param->bake_path, function_description))
                {
                    // Did not find function description
                    continue;
                }

                mi::Uint32 samples(baking_samples);
                mi::Uint32 resolution(baking_resolution);
                // Determine if need to bake texture or constant
                bool uniform(is_uniform(m_gpu_code.get(), function_description.function_index));
                if (m_out_material.is_base_map_parm(p.m_name))
                {
                    uniform = is_base_map_uniform;
                }
                else if (m_out_material.is_mask_map_parm(p.m_name))
                {
                    uniform = is_mask_map_uniform;
                }
                if (uniform)
                {
                    // 1 pixel canvas
                    samples = 1;
                    resolution = 1;
                }

                // Bake texture for expression
                mi::base::Handle<mi::neuraylib::ICanvas> texture(m_out_material.get_texture_for_parameter(p.m_name, image_api, resolution));
                if (!texture)
                    continue;

                Expression_type expr_type(p.m_expr_type);

                const mi::Uint32 components_per_pixel(get_components_per_pixel(mi::base::make_handle(texture->get_tile())->get_type()));
                check_success(components_per_pixel > 0 && components_per_pixel <= 4);

                bake_cuda_ptx_function(
                    texture.get()
                    , baking_samples
                    , components_per_pixel
                    , device_outbuf
                    , device_tc_data_list
                    , device_arg_block_list
                    , cuda_function
                    , function_description
                    , expr_type
                    , metallic
                );

                if (param->remap_func)
                    param->remap_func(texture.get());

                param->texture = texture;
            }

            // Cleanup resources not handled by Material_gpu_context
            check_cuda_success(cuMemFree(device_outbuf));
            check_cuda_success(cuModuleUnload(cuda_module));
        }

        uninit_cuda(cuda_context);

        return true;
    }

    void init_baker_resource(mi::neuraylib::Baker_resource baker_resource)
    {
        switch (baker_resource)
        {
        case mi::neuraylib::BAKE_ON_GPU:
            m_bake_gpu = true;
            break;
        case mi::neuraylib::BAKE_ON_CPU:
            m_bake_cpu = true;
            break;
        case mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK:
            m_bake_gpu = true;
            m_bake_cpu = true;
            break;
        default:
            break;
        }
    }

    void init_function_descriptions()
    {
        // Build function description list
        for (auto & m : m_out_material)
        {
            Material_parameter& param = m.second;
            m_descs.add_function(m.first, param.bake_path);
        }
    }

    const mi::neuraylib::ITarget_code* build_baker_programs_for_backend_kind(
        const mi::neuraylib::ICompiled_material* material
        , mi::neuraylib::IMdl_backend * backend
    )
    {
        check_success(m_transaction.is_valid_interface());
        check_success(m_context.is_valid_interface());
        //////////////////////////////////////////////////////////////////////////////////////////
        // Link unit
        //
        // Create a link unit
        mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(backend->create_link_unit(m_transaction.get(), m_context.get()));
        check_success(link_unit.is_valid_interface());

        // Add all expressions to the link unit
        mi::Sint32 result = link_unit->add_material(material, m_descs.data(), m_descs.size(), m_context.get());
        check_success(0 == result);

        //////////////////////////////////////////////////////////////////////////////////////////
        // Use backend to translate link unit to target code
        //
        // Translate link unit
        return backend->translate_link_unit(link_unit.get(), m_context.get());
    }

    void build_baker_programs(const mi::neuraylib::ICompiled_material* material)
    {
        check_success(m_mdl_backend_api.is_valid_interface());
        if (m_bake_cpu)
        {
            //////////////////////////////////////////////////////////////////////////////////////////
            // Get a backend
            //
            mi::base::Handle<mi::neuraylib::IMdl_backend> backend(m_mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE));
            check_success(backend.is_valid_interface());

            m_native_code = build_baker_programs_for_backend_kind(material, backend.get());
            check_success(m_native_code.is_valid_interface());
        }
        if (m_bake_gpu)
        {
            mi::base::Handle<mi::neuraylib::IMdl_backend> backend(m_mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
            check_success(backend.is_valid_interface());

            // Set backend options
            check_success(backend->set_option("num_texture_spaces", "1") == 0);
            check_success(backend->set_option("tex_lookup_call_mode", "direct_call") == 0);
            check_success(backend->set_option("enable_ro_segment", "off") == 0);
            check_success(backend->set_option("fast_math", "off") == 0);

            m_gpu_code = build_baker_programs_for_backend_kind(material, backend.get());
            check_success(m_gpu_code.is_valid_interface());

            //std::cout << "Dumping CUDA PTX code:\n\n" << m_gpu_code->get_code() << std::endl;
        }
    }

    bool is_uniform(const mi::neuraylib::ITarget_code * code, const mi::Size & function_index) const
    {
        mi::neuraylib::ITarget_code::State_usage render_state_usage =
            code->get_callable_function_render_state_usage(function_index);

        // everything but state::texture_coordinate() and state::position() is constant for baking
        return ((render_state_usage &
            (mi::neuraylib::ITarget_code::SU_TEXTURE_COORDINATE |
                mi::neuraylib::ITarget_code::SU_POSITION)) == 0);
    }

    bool parms_are_uniform(const std::set<std::string> parms, const mi::neuraylib::ITarget_code* code) const
    {
        for (auto& pname : parms)
        {
            Material_parameter* param = m_out_material.find_parameter(pname);
            if (!param)
                continue;

            mi::neuraylib::Target_function_description function_description;
            if (!m_descs.get_function_description(param->bake_path, function_description))
            {
                // Did not find function description
                continue;
            }
            if (!is_uniform(code, function_description.function_index))
            {
                // Return as soon as one of the parms is not uniform
                return false;
            }
        }
        return true;
    }
};

void bake_target_material_inputs(
    mi::neuraylib::Baker_resource baker_resource
    , mi::Uint32 baking_samples
    , mi::Uint32 baking_resolution
    , mi::neuraylib::ITransaction* transaction
    , const mi::neuraylib::ICompiled_material* material
    , mi::neuraylib::IImage_api* image_api
    , Material& out_material
    , mi::neuraylib::IMdl_execution_context * context
    , mi::neuraylib::IMdl_backend_api* mdl_backend_api
    , bool parallel
)
{
    Baker baker(baker_resource, transaction, material, out_material, context, mdl_backend_api);
    check_success(baker.bake(image_api, baking_samples, baking_resolution, parallel));
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
    mi::neuraylib::IMdl_impexp_api* impexp_api)
{
    the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, std::string("Material model: " + target_model));

    Canvas_exporter canvas_exporter(parallel);
    for(Material::const_iterator it = material.begin();
        it != material.end(); ++it)
    {
        const std::string& param_name = it->first;
        const Material_parameter& param = it->second;

        bool mask_map(false);
        bool base_map(false);
        // Do not save redundant textures
        if (param_name == "transparency" && material.has_base_color_map())
        {
            // We skip transparency if a Base Map exists since the Base Map contains transparency value
            the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, "Skip transparency stored in Base Map");
            continue;
        }
        if (param_name == "opacity" && material.has_base_color_map())
        {
            // We skip opacity if a Base Map exists since the Base Map contains opacity value combined with transparency
            the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, "Skip opacity stored in Base Map");
            continue;
        }
        if (param_name == "roughness" && material.has_mask_map())
        {
            // We skip roughness if a Mask Map exists since the Mask Map contains roughness value
            the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, "Skip roughness stored in Mask Map");
            continue;
        }
        if (param_name == "metallic" && material.has_mask_map())
        {
            // Here is the Mask Map
            mask_map = true;
        }
        else if (param_name == "base_color" && material.has_base_color_map())
        {
            // Here is the Base Map
            base_map = true;
        }

        std::stringstream log_message;
        if(param.texture)
        {
            bool skip_uniform(false);
            if (param.texture->get_resolution_x() == 1 && param.texture->get_resolution_y() == 1)
            {
                skip_uniform = true;
                if (mask_map)
                {
                    log_message << "Mask Map (metallic, ao, detail, smoothness) constant ";
                    skip_uniform = false; // Do save the Mask Map
                }
                else if (base_map)
                {
                    log_message << "Base Map (base_color, opacity) constant ";
                    skip_uniform = false; // Do save the Base Map
                }
                else
                {
                    log_message << param_name << " baked to constant ";
                }
                mi::Float32_4 pixel_data;
                make_handle(param.texture->get_tile())->get_pixel(0, 0, (mi::Float32*)&pixel_data.x);
                std::string type(param.texture->get_type());
                if (type == "Rgb_fp")
                {
                    log_message << "color: ("
                        << pixel_data[0] << ", " << pixel_data[1] << ", " << pixel_data[2] << ")";
                }
                else if (type == "Float32")
                {
                    log_message << "float: " << pixel_data[0];
                }
                else if (type == "Float32<3>")
                {
                    log_message << "vector: ("
                        << pixel_data[0] << ", " << pixel_data[1] << ", " << pixel_data[2] << ")";
                }
                else if (type == "Color")
                {
                    log_message << "color: ("
                        << pixel_data[0] << ", " << pixel_data[1] << ", " << pixel_data[2] << ", " << pixel_data[3] << ")";
                    if (base_map)
                    {
                        if (pixel_data[0] >= 0 && pixel_data[0] <= 1 &&
                            pixel_data[1] >= 0 && pixel_data[1] <= 1 &&
                            pixel_data[2] >= 0 && pixel_data[2] <= 1)
                        {
                            log_message << " (hexadecimal color: "
                                << std::setfill('0') << std::setw(2) << std::hex << int(pixel_data[0] * 255)
                                << std::setfill('0') << std::setw(2) << std::hex << int(pixel_data[1] * 255)
                                << std::setfill('0') << std::setw(2) << std::hex << int(pixel_data[2] * 255) << ")";
                        }
                    }
                }
                else
                {
                    the_logger->log(mi::base::MESSAGE_SEVERITY_WARNING, std::string("Unsupported data type"));
                }
                if (!log_message.str().empty())
                {
                    the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, log_message.str());
                }
                log_message = std::stringstream();
            }
            if (save_baked_textures && !skip_uniform)
            {
                // write texture to disc
                std::stringstream file_name;
                if (mask_map)
                {
                    file_name << material_name << "-" << "mask_map" << ".png";
                }
                else if (base_map)
                {
                    file_name << material_name << "-" << "base_map" << ".png";
                }
                else
                {
                    file_name << material_name << "-" << param_name << ".png";
                }

                log_message << param_name << " baked to texture : " << file_name.str();

                canvas_exporter.add_canvas(file_name.str(), param.texture.get());
            }
        }
        if (!log_message.str().empty())
        {
            the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, log_message.str());
        }
    }

    // Export canvases
    canvas_exporter.do_export(impexp_api);
}

// Prints program usage
static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1> ...]\n"
        << "-h                      print this text\n"
        << "--verbosity             verbosity level (0: no messages, 1: fatal, 2: error, 3: warning, 4: info, 5: verbose, 6: debug)\n"
        << "--baker_resource        baking device: gpu|cpu|gpu_with_cpu_fallback (default: cpu)\n"
        << "--samples               baking samples (default: 4, max: 16)\n"
        << "--resolution            baking resolution (default: 1024)\n"
        << "--material_file <file>  file containing fully qualified names of materials to distill\n"
        << "--do_not_save_textures  if set, avoid saving baked textures to file\n"
        << "--module <module_name>  distill all materials from the module, can occur multiple times\n"
        << "--no_parallel           do not save texture files in parallel threads\n"
        << "--mdl_path <path>       mdl search path, can occur multiple times.\n";

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
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(mdl_factory->create_execution_context());
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
            material_names.push_back(std::string(material->get_mdl_module_name()) + "::" + std::string(material->get_mdl_simple_name()));
        }
    }
}

int MAIN_UTF8(int argc, char* argv[])
{
    std::string                     target_model = "transmissive_pbr";
    mi::neuraylib::Baker_resource   baker_resource = mi::neuraylib::BAKE_ON_CPU;
    mi::Uint32                      baking_samples = 4;
    mi::Uint32                      baking_resolution = 1024;
    bool                            parallel = true;
    std::vector<std::string>        material_names;
    std::vector<std::string>        module_names;
    std::string material_file;
    bool save_baked_textures(true);
    int verbosity_level(mi::base::MESSAGE_SEVERITY_VERBOSE);
    the_logger = new Logger(verbosity_level);

    mi::examples::mdl::Configure_options configure_options;
    configure_options.add_admin_space_search_paths = false;
    configure_options.add_user_space_search_paths = false;
    configure_options.add_example_search_path = false;
    configure_options.logger = the_logger.get();

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
                    if (val > 0 && val <= 16)
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
            else if (strcmp(opt, "--verbosity") == 0) {
                if (i < argc - 1)
                {
                    int val(atoi(argv[++i]));
                    if (val >= 0 && val <= 6)
                        verbosity_level = val;
                    else
                        std::cout << "Invalid verbosity ignored\n";
                }
                else
                    usage(argv[0]);
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

    // Update verbosity level after processing command line arguments
    the_logger->set_verbosity_level(verbosity_level);

    if (configure_options.additional_mdl_paths.empty())
        configure_options.add_example_search_path = true;
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

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        LoggerTiming timing("Load, distill and bake all materials");

        // Get MDL Import/Export API
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        // Get MDL backend API
        mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
            neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

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
        size_t n_materials_done(0);
        size_t n_materials_todo(material_names.size());
        for (const auto& m : material_names)
        {
            LoggerTiming timing("Distill and bake");

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

            // Acquire distilling api used for material distilling and baking
            mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distilling_api(
                neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

            // Distill compiled material to diffuse/glossy material model
            mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_material(
                create_distilled_material(
                    distilling_api.get(),
                    compiled_material.get(),
                    target_model.c_str()));

            // Acquire image api needed to create a canvas for baking
            mi::base::Handle<mi::neuraylib::IImage_api> image_api(
                neuray->get_api_component<mi::neuraylib::IImage_api>());

            Material out_material;
            // Setup result material parameters and collect bake paths
            setup_target_material(
                transaction.get(),
                distilled_material.get(),
                out_material);

            // Bake material inputs
            bake_target_material_inputs(
                baker_resource,
                baking_samples,
                baking_resolution,
                transaction.get(),
                distilled_material.get(),
                image_api.get(),
                out_material,
                context.get(),
                mdl_backend_api.get(),
                parallel);

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

            the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, std::string("Distilled material: ") + material_simple_name);

            n_materials_done++;

            std::stringstream strStream;
            strStream << "Progress: " << (float(n_materials_done) / n_materials_todo) * 100 << " % (" << n_materials_done << "/" << n_materials_todo << ")";
            the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, strStream.str());
        }

        the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, std::string("Transaction commit"));
        transaction->commit();
    }

    // Shut down the MDL SDK
    the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, std::string("Shutting down the SDK"));
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, std::string("Unloading the SDK"));
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    the_logger->log(mi::base::MESSAGE_SEVERITY_VERBOSE, std::string("Exiting"));
    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
