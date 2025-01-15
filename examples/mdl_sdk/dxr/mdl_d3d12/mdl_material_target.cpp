/******************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_material_target.h"

#include <iostream>
#include <fstream>

#include "base_application.h"
#include "descriptor_heap.h"
#include "mdl_material.h"
#include "mdl_sdk.h"
#include "texture.h"

#include <example_shared.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

std::string compute_shader_cache_filename(
    const Base_options& options,
    const std::string& material_hash,
    bool create_parent_folders = false)
{
    std::string compiler = "dxc";
    #ifdef MDL_ENABLE_SLANG
        if(options.use_slang)
            compiler = "slang";
    #endif

    std::string folder = mi::examples::io::get_executable_folder() +"/shader_cache/" + compiler;
    if (create_parent_folders)
        mi::examples::io::mkdir(folder);
    return folder + "/" + material_hash + ".bin";
}

// ------------------------------------------------------------------------------------------------

Mdl_material_target::Resource_callback::Resource_callback(
    Mdl_sdk* sdk, Mdl_material_target* target, Mdl_material* material)
    : m_sdk(sdk)
    , m_target(target)
    , m_material(material)
{
}

// ------------------------------------------------------------------------------------------------

mi::Uint32 Mdl_material_target::Resource_callback::get_resource_index(
    mi::neuraylib::IValue_resource const *resource)
{
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
        m_target->get_target_code());

    // resource available in the target code?
    // this is the case for resources that are in the material body and for
    // resources contained in the parameters of the first appearance of a material
    mi::Uint32 index = m_sdk->get_transaction().execute<mi::Uint32>(
        [&](mi::neuraylib::ITransaction* t)
    {
        return target_code->get_known_resource_index(t, resource);
    });

    // resource is part of the target code, so we use it
    if (index > 0)
    {
        // we loaded only the body resources so far so we only accept those as is
        switch (resource->get_kind())
        {
        case mi::neuraylib::IValue::VK_TEXTURE:
            if (target_code->get_texture_is_body_resource(index))
                return index;
            break;

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            if (target_code->get_light_profile_is_body_resource(index))
                return index;
            break;

        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            if (target_code->get_bsdf_measurement_is_body_resource(index))
                return index;
            break;
        default:
            break;
        }
    }

    // invalid (or empty) resource
    const char* name = resource->get_value();
    if (!name)
    {
        return 0;
    }

    // All resources that are loaded for later appearances of a material, i.e. when a
    // material is reused (probably with different parameters), have to handled separately.
    // If the target was not yet generated (usually the case when a shared target code is used),
    // additional resources can be added to the list of resources of the target.
    // Otherwise, resources are added to the material (when separate link units are used).

    Mdl_resource_kind kind;
    Texture_dimension dimension = Texture_dimension::Undefined;
    switch (resource->get_kind())
    {
        case mi::neuraylib::IValue::VK_TEXTURE:
        {
            mi::base::Handle<const mi::neuraylib::IType> type(
                resource->get_type());

            mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(
                type->get_interface<const mi::neuraylib::IType_texture>());

            switch (texture_type->get_shape())
            {
                case mi::neuraylib::IType_texture::TS_2D:
                    kind = Mdl_resource_kind::Texture;
                    dimension = Texture_dimension::Texture_2D;
                    break;

                case mi::neuraylib::IType_texture::TS_3D:
                    kind = Mdl_resource_kind::Texture;
                    dimension = Texture_dimension::Texture_3D;
                    break;

                default:
                    log_error("Invalid texture shape for: " + std::string(name), SRC);
                    return 0;
            }
            break;
        }

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            kind = Mdl_resource_kind::Light_profile;
            break;

        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            kind = Mdl_resource_kind::Bsdf_measurement;
            break;

        default:
            log_error("Invalid resource kind for: " + std::string(name), SRC);
            return 0;
    }

    // store textures at the material
    size_t mat_resource_index = m_material->register_resource(kind, dimension, name);

    // log these manually defined indices
    log_info(
        "target code: " + m_target->get_compiled_material_hash() +
        " - texture id: " + std::to_string(mat_resource_index) +
        " (material id: " + std::to_string(m_material->get_id()) + ")" +
        " - resource: " + std::string(name) + " (reused material)");

    return mat_resource_index;
}

// ------------------------------------------------------------------------------------------------

mi::Uint32 Mdl_material_target::Resource_callback::get_string_index(
    mi::neuraylib::IValue_string const *s)
{
    // of the string was known to the compiler the mapped id MUST match
    // the one of the target code
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
        m_target->get_target_code());
    mi::Size n = target_code->get_string_constant_count();
    for (mi::Size i = 0; i < n; ++i)
        if (strcmp(target_code->get_string_constant(i), s->get_value()) == 0)
            return static_cast<mi::Uint32>(i);

    // invalid (or empty) string
    const char* name = s->get_value();
    if (!name || name[0] == '\0')
    {
        return 0;
    }

    // additional new string mappings:
    // store string constant at the material
    size_t mat_string_index = m_material->map_string_constant(name);
    assert(mat_string_index >= n); // the new IDs must not collide with the ones of the target code
    return mat_string_index;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Mdl_material_target::Mdl_material_target(
    Base_application* app,
    Mdl_sdk* sdk,
    const std::string& compiled_material_hash)
    : m_app(app)
    , m_sdk(sdk)
    , m_compiled_material_hash(compiled_material_hash)
    , m_target_code(nullptr)
    , m_generation_required(true)
    , m_hlsl_source_code("")
    , m_compilation_required(true)
    , m_dxil_compiled_libraries()
    , m_read_only_data_segment(nullptr)
    , m_target_resources()
    , m_target_string_constants()
{
    // add the empty resources
    for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
    {
        Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
        m_target_resources[kind_i] = std::vector<Mdl_resource_assignment>();
    }

    // will be used in the shaders and when setting up the rt pipeline
    m_radiance_closest_hit_name = "MdlRadianceClosestHitProgram_" + get_shader_name_suffix();
    m_radiance_any_hit_name = "MdlRadianceAnyHitProgram_" + get_shader_name_suffix();
    m_shadow_any_hit_name = "MdlShadowAnyHitProgram_" + get_shader_name_suffix();
}

// ------------------------------------------------------------------------------------------------

Mdl_material_target::~Mdl_material_target()
{
    m_target_code = nullptr;
    m_dxil_compiled_libraries.clear();

    if (m_read_only_data_segment) delete m_read_only_data_segment;

    // free heap block
    if(m_first_resource_heap_handle.is_valid())
        m_app->get_resource_descriptor_heap()->free_views(m_first_resource_heap_handle);
}

// ------------------------------------------------------------------------------------------------

mi::neuraylib::ITarget_resource_callback* Mdl_material_target::create_resource_callback(
    Mdl_material* material)
{
    return new Resource_callback(m_sdk, this, material);
}

// ------------------------------------------------------------------------------------------------

const mi::neuraylib::ITarget_code* Mdl_material_target::get_target_code() const
{
    if (!m_target_code)
        return nullptr;

    m_target_code->retain();
    return m_target_code.get();
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_target::add_material_to_link_unit(
    Mdl_material_target_interface& interface_data,
    Mdl_material* material,
    mi::neuraylib::ILink_unit* link_unit,
    mi::neuraylib::IMdl_execution_context* context)
{
    // get the compiled material and add the material to the link unit
    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        m_sdk->get_transaction().access<const mi::neuraylib::ICompiled_material>(
            material->get_material_compiled_db_name().c_str()));

    // Selecting expressions to generate code for is one of the most important parts
    // You can either choose to generate all supported functions and just use them which is fine.
    // To reduce the effort in the code generation and compilation steps later it makes sense
    // to reduce the code size as good as possible. Therefore, we inspect the compiled
    // material before selecting functions for code generation.

    // third, either the bsdf or the edf need to be non-default (black)

    // helper to check if a expression exists in the material type
    auto exists = [&compiled_material](const char* expression_path)
    {
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            compiled_material->lookup_sub_expression(expression_path));
        return expr.is_valid_interface();
    };

    // helper function to check if a distribution function is invalid
    // if true, the distribution function needs no eval because it has no contribution
    auto is_invalid_df = [&compiled_material](const char* expression_path)
    {
        // fetch the constant expression
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            compiled_material->lookup_sub_expression(expression_path));
        if (expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
            return false;

        // get the constant value
        mi::base::Handle<const mi::neuraylib::IExpression_constant> expr_constant(
            expr->get_interface<mi::neuraylib::IExpression_constant>());
        mi::base::Handle<const mi::neuraylib::IValue> value(
            expr_constant->get_value());

        return value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF;
    };

    // helper function to check if a color function needs to be evaluted
    // returns true if the expression value is constant black, otherwise true
    auto is_constant_black_color = [&compiled_material](const char* expression_path)
    {
        // fetch the constant expression
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            compiled_material->lookup_sub_expression(expression_path));
        if (expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
            return false;

        // get the constant value
        mi::base::Handle<const mi::neuraylib::IExpression_constant> expr_constant(
            expr->get_interface<mi::neuraylib::IExpression_constant>());
        mi::base::Handle<const mi::neuraylib::IValue_color> value(
            expr_constant->get_value<mi::neuraylib::IValue_color>());
        if (!value)
            return false;

        // check for black
        for (mi::Size i = 0; i < value->get_size(); ++i)
        {
            mi::base::Handle<mi::neuraylib::IValue_float const> element(value->get_value(i));
            if (element->get_value() != 0.0f)
                return false;
        }
        return true;
    };

    // helper function to check if a bool function needs to be evaluted
    // returns true if the expression value is constant false, otherwise true
    auto is_constant_false = [&compiled_material](const char* expression_path)
    {
        // fetch the constant expression
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            compiled_material->lookup_sub_expression(expression_path));
        if (expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
            return false;

        // get the constant value
        mi::base::Handle<const mi::neuraylib::IExpression_constant> expr_constant(
            expr->get_interface<mi::neuraylib::IExpression_constant>());
        mi::base::Handle<const mi::neuraylib::IValue_bool> value(
            expr_constant->get_value<mi::neuraylib::IValue_bool>());
        if (!value)
            return false;

        // check for "false"
        return value->get_value() == false;
    };

    // select expressions to generate HLSL code for
    std::vector<mi::neuraylib::Target_function_description> selected_functions;

    // always add init
    {
        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "init", "mdl_init"));
        interface_data.add_code_feature(Material_code_feature::HAS_INIT);
    }

    // add surface emission if available
    if (exists("surface.emission.emission") && 
        exists("surface.emission.intensity") && 
        !is_invalid_df("surface.emission.emission") &&
        !is_constant_black_color("surface.emission.intensity"))
    {
        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "surface.emission.emission", "mdl_surface_emission"));
        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "surface.emission.intensity", "mdl_surface_emission_intensity"));
        interface_data.add_code_feature(Material_code_feature::SURFACE_EMISSION);
    }

    // add absorption
    if (exists("volume.absorption_coefficient") && 
        !is_constant_black_color("volume.absorption_coefficient"))
    {
        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "volume.absorption_coefficient", "mdl_volume_absorption_coefficient"));
        interface_data.add_code_feature(Material_code_feature::VOLUME_ABSORPTION);
    }

    // add surface scattering if available
    if (exists("surface.scattering") && !is_invalid_df("surface.scattering"))
    {
        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "surface.scattering", "mdl_surface_scattering"));
        interface_data.add_code_feature(Material_code_feature::SURFACE_SCATTERING);
    }

    // thin walled and potentially with a different backface
    if (exists("thin_walled") && !is_constant_false("thin_walled"))
    {
        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "thin_walled", "mdl_thin_walled"));
        interface_data.add_code_feature(Material_code_feature::CAN_BE_THIN_WALLED);

        if (exists("backface.scattering"))
        {
            // back faces could be different for thin walled materials
            // we only need to generate new code
            // 1. if surface and backface are different
            bool need_backface_bsdf =
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING) !=
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING);
            bool need_backface_edf =
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION) || 
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY) !=
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY) ||
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_MODE) !=
                compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_MODE);

            // 2. either the bsdf or the edf need to be non-default (black)
            bool none_default_backface =
                !is_invalid_df("backface.scattering") || !is_invalid_df("backface.emission.emission");
            need_backface_bsdf &= none_default_backface;
            need_backface_edf &= none_default_backface;

            if (need_backface_bsdf || need_backface_edf)
            {
                // generate code for both backface functions here, even if they are black
                // because it could be requested to have black backsides
                selected_functions.push_back(mi::neuraylib::Target_function_description(
                    "backface.scattering", "mdl_backface_scattering"));
                interface_data.add_code_feature(Material_code_feature::BACKFACE_SCATTERING);

                selected_functions.push_back(mi::neuraylib::Target_function_description(
                    "backface.emission.emission", "mdl_backface_emission"));
                selected_functions.push_back(mi::neuraylib::Target_function_description(
                    "backface.emission.intensity", "mdl_backface_emission_intensity"));
                interface_data.add_code_feature(Material_code_feature::BACKFACE_EMISSION);
            }
        }
    }

    // with MDL 1.9, custom material types are supported
    // generate code for a used defined arbitrary output variable (AOV)
    // this is just an illustration of how to generate and use code for AOV. Real-world
    // applications would feed the computed values into a simulation for example.
    const std::vector<std::string>& aov_expression_paths =
        m_app->get_dynamic_options()->get_available_aovs();
    if (!aov_expression_paths.empty())
    {
        mi::base::Handle<mi::neuraylib::IType_factory> tf(
            m_sdk->get_factory().create_type_factory(m_sdk->get_transaction().get()));

        // the renderer requests AOVs
        // if they are present in the material, we generate code
        // otherwise, we return black in the switch later
        interface_data.add_code_feature(Material_code_feature::HAS_AOVS);
        interface_data.aovs.reserve(aov_expression_paths.size());

        for (size_t i = 0; i < aov_expression_paths.size(); ++i)
        {
            const std::string& expression_path = aov_expression_paths[i];
            mi::base::Handle<const mi::neuraylib::IExpression> aov_expr(
                compiled_material->lookup_sub_expression(expression_path.c_str()));
            if (!aov_expr)
            {
                log_error("AOV '" + expression_path + "' not available in material: " +
                    material->get_name(), SRC);
            }
            else
            {
                mi::base::Handle<const mi::neuraylib::IType> ret_type(aov_expr->get_type());
                ret_type = ret_type->skip_all_type_aliases();
                if (ret_type->get_kind() == mi::neuraylib::IType::TK_ARRAY ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_BSDF ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_BSDF_MEASUREMENT ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_EDF ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_ENUM ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_HAIR_BSDF ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_LIGHT_PROFILE ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_MATRIX ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_STRUCT ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_TEXTURE ||
                    ret_type->get_kind() == mi::neuraylib::IType::TK_VDF)
                {
                    m_sdk->log_messages(
                        "AOV visualization for the type of '" + expression_path + "' not support.",
                        context, SRC);
                }
                else
                {
                    // collect information about the selected outputs
                    // this is mainly for generating the code that visualizes the AOVs
                    size_t aov_index = interface_data.aovs.size();
                    interface_data.aovs.push_back({});
                    Mdl_aov_info& info = interface_data.aovs.back();

                    info.runtime_index = uint32_t(i);
                    info.expression_path = expression_path;
                    info.hlsl_name = "mdl_aov_" + expression_path;
                    std::replace(info.hlsl_name.begin(), info.hlsl_name.end(), '.', '_');
                    mi::base::Handle<const mi::IString> ret_type_name(
                        tf->get_mdl_type_name(ret_type.get()));
                    info.mdl_type_name = ret_type_name->get_c_str();

                    // add function to link unit
                    selected_functions.push_back(mi::neuraylib::Target_function_description(
                        info.expression_path.c_str(), info.hlsl_name.c_str()));
                }
            }
        }
    }

    // add the main material to the link unit for code generation
    if (selected_functions.size() == 1)
    {
        log_warning("Material has supported renderer feature and will be black.");
        interface_data.material_code_paths = 0; // disable
    }
    else
    {
        link_unit->add_material(
            compiled_material.get(),
            selected_functions.data(), selected_functions.size(),
            context);

        if (!m_sdk->log_messages("Failed to select functions for code generation.", context, SRC))
            return false;
    }

    // check if the material is trivial opaque, if not we need to generate code for cutout opacity
    float opacity = -1.0f;
    if (exists("geometry.cutout_opacity") &&
        (!compiled_material->get_cutout_opacity(&opacity) || opacity < 1.0f))
    {
        // compile cutout_opacity also as standalone version to be used in the anyhit programs,
        // to avoid costly precalculation of expressions only used by other expressions
        interface_data.add_code_feature(Material_code_feature::CUTOUT_OPACITY);

        mi::neuraylib::Target_function_description standalone_opacity(
            "geometry.cutout_opacity", "mdl_standalone_geometry_cutout_opacity");

        link_unit->add_material(
            compiled_material.get(),
            &standalone_opacity, 1,
            context);

        if (!m_sdk->log_messages("Failed to add cutout_opacity for code generation.", context, SRC))
            return false;
    }

    // get the resulting target code information
    // constant for the entire material, for one material per link unit 0
    interface_data.argument_layout_index = selected_functions[0].argument_block_index;
    return true;
}

// ------------------------------------------------------------------------------------------------

/// Keep a pointer (no ownership) to the material for notifying the material when the
/// target code generation is finished.
void Mdl_material_target::register_material(Mdl_material* material)
{
    Mdl_material_target* current_target = material->get_target_code();

    // mark changed because registered material is called only for new or changed materials
    m_generation_required = true;
    m_compilation_required = true;

    // register with this target code
    std::unique_lock<std::mutex> lock(m_materials_mtx);
    m_materials[material->get_id()] = material;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_target::unregister_material(Mdl_material* material)
{
    if (material->get_target_code() != this)
    {
        log_error("Tried to remove a material from the wrong target: " + material->get_name(), SRC);
        return false;
    }

    std::unique_lock<std::mutex> lock(m_materials_mtx);
    auto found = m_materials.find(material->get_id());
    if (found != m_materials.end())
    {
        material->reset_target_interface();
        m_materials.erase(found);
        m_generation_required = true;
        m_compilation_required = true;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------

size_t Mdl_material_target::get_material_resource_count(
    Mdl_resource_kind kind) const
{
    const auto& found = m_material_resource_count.find(kind);
    return found->second;
}

// ------------------------------------------------------------------------------------------------

uint32_t Mdl_material_target::map_string_constant(const std::string& string_value)
{
    // the empty string is also the invalid string
    if (string_value == "")
        return 0;

    // if the constant is already mapped, use it
    for (auto& c : m_target_string_constants)
        if (c.value == string_value)
            return c.runtime_string_id;

    // map the new constant. keep this mapping dense in order to easy the data layout on the GPU
    uint32_t runtime_id =
        m_target_string_constants.empty() ? 1 : m_target_string_constants.back().runtime_string_id + 1;
    Mdl_string_constant entry;
    entry.runtime_string_id = runtime_id;
    entry.value = string_value;

    m_target_string_constants.push_back(entry);
    return runtime_id;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_target::visit_materials(std::function<bool(Mdl_material*)> action)
{
    std::lock_guard<std::mutex> lock(m_materials_mtx);
    for (auto it = m_materials.begin(); it != m_materials.end(); it++)
        if (!action(it->second))
            return false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_target::generate()
{
    if (!m_generation_required)
    {
        log_info("Target code does not need generation. Hash: " + m_compiled_material_hash, SRC);
        return true;
    }

    // since this method can be called from multiple threads simultaneously
    // a new context for is created
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

    // use shader caching if enabled
    bool loaded_from_shader_cache = false;
    Mdl_material_target_interface interface_data;
    if (m_sdk->get_options().enable_shader_cache)
    {
        auto p = m_app->get_profiling().measure("loading from shader cache");

        const std::string filename = compute_shader_cache_filename(
            *m_app->get_options(), m_compiled_material_hash);

        // read the cache if it exists
        if (mi::examples::io::file_exists(filename))
        {
            std::ifstream file(filename,
                std::ios::in | std::ios::binary | std::ios::ate);

            char* cache_buffer = nullptr;
            while (file.is_open())
            {
                size_t cache_buffer_size = file.tellg();
                file.seekg(0, std::ios::beg);
                cache_buffer = new char[cache_buffer_size];
                file.read(cache_buffer, cache_buffer_size);
                file.close();

                // get the target code
                if (cache_buffer_size < sizeof(size_t))
                    break;
                size_t target_code_size = *(reinterpret_cast<size_t*>(cache_buffer));
                size_t offset = sizeof(size_t);
                if (cache_buffer_size < (offset + target_code_size))
                    break;

                // use the back-end to restore the serialized target code
                // this is expected to fail when build or protocol versions do not match
                m_target_code = m_sdk->get_backend().deserialize_target_code(
                    m_sdk->get_transaction().get(),
                    reinterpret_cast<mi::Uint8*>(cache_buffer + offset),
                    target_code_size, context.get());

                if (!m_sdk->log_messages("Deserializing Shader Cache failed.", context.get()) ||
                    !m_target_code)
                        break;

                offset += target_code_size;

                // get the material interface data
                if (cache_buffer_size < sizeof(Mdl_material_target_interface))
                    break;
                interface_data = *(reinterpret_cast<Mdl_material_target_interface*>(
                    cache_buffer + offset));
                offset += sizeof(Mdl_material_target_interface);

                // also get the compiles shader
                if (cache_buffer_size < (offset + sizeof(size_t)))
                    break;

                // number of dxil libraries
                size_t num_libraries = *(reinterpret_cast<size_t*>(cache_buffer + offset));
                offset += sizeof(size_t);
                m_dxil_compiled_libraries.clear();
                m_dxil_compiled_libraries.reserve(num_libraries);

                for (size_t l = 0; l < num_libraries; ++l)
                {
                    // first, the symbols, starting with the number of symbols ...
                    size_t num_exp_symbols = *(reinterpret_cast<size_t*>(cache_buffer + offset));
                    offset += sizeof(size_t);

                    std::vector<std::string> exports;
                    for (size_t s = 0; s < num_exp_symbols; ++s)
                    {
                        // ... and then for each, string length and string data
                        size_t symbol_size = *(reinterpret_cast<size_t*>(cache_buffer + offset));
                        offset += sizeof(size_t);

                        std::string symbol_name(cache_buffer + offset, cache_buffer + offset + symbol_size);
                        offset += symbol_size;
                        exports.push_back(symbol_name);
                    }

                    // the dxil blob
                    size_t dxil_blob_size = *(reinterpret_cast<size_t*>(cache_buffer + offset));
                    offset += sizeof(size_t);
                    auto dxil_compiled_library = new DxcBlobFromMemory(cache_buffer + offset, dxil_blob_size);
                    m_dxil_compiled_libraries.push_back(Shader_library(dxil_compiled_library, exports));
                    offset += dxil_blob_size;
                }

                loaded_from_shader_cache = true;
                break;
            }
            if (cache_buffer)
                delete[] cache_buffer;
        }
    }

    // use the back-end to generate HLSL code
    mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(nullptr);
    if (!loaded_from_shader_cache)
    {
        // in order to change the scene scale setting at runtime we need to preserve the conversions
        // in the generated code and expose the factor in the MDL material state of the shader.
        context->set_option("fold_meters_per_scene_unit", false);

        link_unit = m_sdk->get_backend().create_link_unit(
            m_sdk->get_transaction().get(), context.get());
        if (!m_sdk->log_messages("MDL creating a link unit failed.", context.get(), SRC))
            return false;
    }

    // empty resource list (in case of reload) and rest the counter
    for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
    {
        Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
        m_target_resources[kind_i].clear();
    }

    // add materials to link unit
    std::string process_hash;
    for (auto& pair : m_materials)
    {
        // add materials with the same hash only once
        const std::string& hash = pair.second->get_material_compiled_hash();
        if (process_hash.empty())
        {
            process_hash = hash;

            // add this material to the link unit
            if (!loaded_from_shader_cache && !add_material_to_link_unit(
                interface_data, pair.second, link_unit.get(), context.get()))
            {
                log_error("Adding to link unit failed: " + pair.second->get_name(), SRC);
                return false;
            }
        }
        else
        {
            if (process_hash != hash)
            {
                log_error("Material added to the wrong target: " + pair.second->get_name(), SRC);
                return false;
            }
        }

        // pass target information to the material
        pair.second->set_target_interface(this, interface_data);
    }

    // generate HLSL code
    if (!loaded_from_shader_cache && interface_data.material_code_paths != 0)
    {
        auto p = m_app->get_profiling().measure("generating HLSL (translate link unit)");
        m_target_code = m_sdk->get_backend().translate_link_unit(link_unit.get(), context.get());
        if (!m_sdk->log_messages("MDL target code generation failed.", context.get(), SRC))
            return false;
    }

    // create a command list for uploading data to the GPU
    Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    D3DCommandList* command_list = command_queue->get_command_list();

    // add all body textures, the ones that are required independent of the parameter set
    for (size_t i = 1, n = m_target_code ? m_target_code->get_texture_count() : 0; i < n; ++i)
    {
        if (!m_target_code->get_texture_is_body_resource(i))
            continue;

        Mdl_resource_assignment assignment(Mdl_resource_kind::Texture);
        assignment.resource_name = m_target_code->get_texture(i);
        assignment.runtime_resource_id = i;

        switch (m_target_code->get_texture_shape(i))
        {
            case mi::neuraylib::ITarget_code::Texture_shape_2d:
                assignment.dimension = Texture_dimension::Texture_2D;
                break;

            case mi::neuraylib::ITarget_code::Texture_shape_3d:
            case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data:
                assignment.dimension = Texture_dimension::Texture_3D;
                break;

            default:
                log_error("Only 2D and 3D textures are supported by this example.", SRC);
                return false;
        }

        m_target_resources[Mdl_resource_kind::Texture].emplace_back(assignment);
    }

    // add all body light profiles
    for (size_t i = 1, n = m_target_code ? m_target_code->get_light_profile_count() : 0; i < n; ++i)
    {
        if (!m_target_code->get_light_profile_is_body_resource(i))
            continue;

        Mdl_resource_assignment assignment(Mdl_resource_kind::Light_profile);
        assignment.resource_name = m_target_code->get_light_profile(i);
        assignment.runtime_resource_id = i;
        m_target_resources[Mdl_resource_kind::Light_profile].emplace_back(assignment);
    }

    // add all body bsdf measurements
    for (size_t i = 1, n = m_target_code ? m_target_code->get_bsdf_measurement_count() : 0; i < n; ++i)
    {
        if (!m_target_code->get_bsdf_measurement_is_body_resource(i))
            continue;

        Mdl_resource_assignment assignment(Mdl_resource_kind::Bsdf_measurement);
        assignment.resource_name = m_target_code->get_bsdf_measurement(i);
        assignment.runtime_resource_id = i;
        m_target_resources[Mdl_resource_kind::Bsdf_measurement].emplace_back(assignment);
    }

    // add all string constants known to the link unit
    m_target_string_constants.clear();
    for (size_t i = 1, n = m_target_code ? m_target_code->get_string_constant_count() : 0; i < n; ++i)
    {
        Mdl_string_constant constant;
        constant.runtime_string_id = i;
        constant.value = m_target_code->get_string_constant(i);
        m_target_string_constants.push_back(constant);
    }

    // add TEXCOORD_0 to demonstrate renderer driven scene data elements
    // NOTE, if this is added manually, MDL code will not create any runtime function call
    // that with the 'scene_data_id'. Instead, only the render can call this outside of the
    // generated code.
    map_string_constant("TEXCOORD_0");

    #if defined(MDL_ENABLE_MATERIALX)
        // add camera position for MaterialX NPR nodes
        // NOTE, using this in a path tracer is questionable
        map_string_constant("CAMERA_POSITION");
    #endif

    // create per material resources, parameter bindings, ...
    // ------------------------------------------------------------

    // ... in parallel, if not forced otherwise
    std::vector<std::thread> tasks;
    std::atomic_bool success = true;
    for (auto mat : m_materials)
    {
        // sequentially
        if (m_app->get_options()->force_single_threading)
        {
            if (!mat.second->on_target_generated(command_list))
                success.store(false);
        }
        // asynchronously
        else
        {
            tasks.emplace_back(std::thread([&, mat]()
            {
                // no not fill command lists from different threads
                D3DCommandList* local_command_list = command_queue->get_command_list();
                if (!mat.second->on_target_generated(local_command_list))
                    success.store(false);
                command_queue->execute_command_list(local_command_list);
            }));
        }
    }

    // wait for all loading tasks
    for (auto &t : tasks)
        t.join();

    // any errors?
    if (!success.load())
    {
        log_error("On generate code callback return with failure.", SRC);
        return false;
    }

    // at this point, we know the number of resources in instances of the materials.
    // Since the root signature for all instances of the "same" material (probably different
    // parameter sets when using MDL class compilation) has to be identical, we go for the
    // maximum amount of occurring resources.
    for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
        m_material_resource_count[static_cast<Mdl_resource_kind>(i)] = 0;

    visit_materials([&](const Mdl_material* mat)
    {
        for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
        {
            Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
            size_t current = mat->get_resources(kind_i).size();
            m_material_resource_count[kind_i] =
                std::max(m_material_resource_count[kind_i], current);
        }
        return true;
    });

    // in order to load resources in parallel a continuous block of resource handles
    // for this target_code is allocated
    Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();
    size_t handle_count = 1; // read-only segment

    // if we already have a block on the resource heap (previous generation)
    // we try to reuse is if it fits
    if (m_first_resource_heap_handle.is_valid())
    {
        if (resource_heap.get_block_size(m_first_resource_heap_handle) < handle_count)
            resource_heap.free_views(m_first_resource_heap_handle); // free block
    }

    // reserve a new block of the required size and check if that was successful
    if (!m_first_resource_heap_handle.is_valid())
    {
        m_first_resource_heap_handle = resource_heap.reserve_views(handle_count);
        if (!m_first_resource_heap_handle.is_valid())
            return false;
    }

    // create per target resources
    // --------------------------------------

    // read-only data, all jit back-ends, including HLSL produce zero or one segments
    if (m_read_only_data_segment)
    {
        // free old buffer
        delete m_read_only_data_segment;
        m_read_only_data_segment = nullptr;
    }
    if (m_target_code && m_target_code->get_ro_data_segment_count() > 0)
    {
        // create a new buffer
        size_t ro_data_seg_index = 0; // assuming one material per target code only
        const char* name = m_target_code->get_ro_data_segment_name(ro_data_seg_index);
        m_read_only_data_segment = new Buffer(
            m_app, m_target_code->get_ro_data_segment_size(ro_data_seg_index),
            "MDL_ReadOnly_" + std::string(name));

        m_read_only_data_segment->set_data(
            m_target_code->get_ro_data_segment_data(ro_data_seg_index),
            m_target_code->get_ro_data_segment_size(ro_data_seg_index));
    }
    if (m_read_only_data_segment == nullptr)
    {
        // create a dummy buffer to always have the same alignment in the heap
        m_read_only_data_segment = new Buffer(m_app, 4, "MDL_ReadOnly_nullptr");
        uint32_t zero(0);
        m_read_only_data_segment->set_data(&zero, 1);
    }

    // create resource view on the heap (at the first position of the target codes block)
    if (!resource_heap.create_shader_resource_view(
        m_read_only_data_segment, true, m_first_resource_heap_handle))
        return false;

    // copy data to the GPU
    if (m_read_only_data_segment && !m_read_only_data_segment->upload(command_list))
        return false;


    // prepare descriptor table for all per target resources
    // -------------------------------------------------------------------

    // note that the offset in the heap starts with zero
    // for each target we set 'target_heap_region_start' in the local root signature

    m_resource_descriptor_table.clear();

    // bind read-only data segment to shader
    m_resource_descriptor_table.register_srv(0, 2, 0);

    // generate some dxr specific shader code to hook things up
    // -------------------------------------------------------------------

    // ... this is not required when the shader is loaded from the cache
    if (loaded_from_shader_cache)
    {
        command_queue->execute_command_list(command_list);
        m_generation_required = false;
        m_compilation_required = false;
        return true;
    }

    // generate the actual shader code with the help of some snippets
    m_hlsl_source_code.clear();
    m_hlsl_compilation_defines["TARGET_CODE_ID"] = get_shader_name_suffix();
    m_hlsl_compilation_defines["DISABLE_READABILITY"] = "1"; // disable hlsl defines for editing

    // to allow the render code to be optimized for material features defines can be used
    // if not, the availality of a feature and the necessity of runnign the corresponding code path
    // is evaluated dynamically.

    bool optimize_renderer_for_available_material_code_path = true;
    if (optimize_renderer_for_available_material_code_path)
    {
        // depending on the functions selected for code generation
        // 
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::HAS_INIT)
            ? "#define MDL_HAS_INIT 1\n"
            : "#define MDL_HAS_INIT 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::SURFACE_SCATTERING)
            ? "#define MDL_HAS_SURFACE_SCATTERING 1\n"
            : "#define MDL_HAS_SURFACE_SCATTERING 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::SURFACE_EMISSION)
            ? "#define MDL_HAS_SURFACE_EMISSION 1\n"
            : "#define MDL_HAS_SURFACE_EMISSION 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::BACKFACE_SCATTERING)
            ? "#define MDL_HAS_BACKFACE_SCATTERING 1\n"
            : "#define MDL_HAS_BACKFACE_SCATTERING 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::BACKFACE_EMISSION)
            ? "#define MDL_HAS_BACKFACE_EMISSION 1\n"
            : "#define MDL_HAS_BACKFACE_EMISSION 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::VOLUME_ABSORPTION)
            ? "#define MDL_HAS_VOLUME_ABSORPTION 1\n"
            : "#define MDL_HAS_VOLUME_ABSORPTION 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::CAN_BE_THIN_WALLED)
            ? "#define MDL_CAN_BE_THIN_WALLED 1\n"
            : "#define MDL_CAN_BE_THIN_WALLED 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::CUTOUT_OPACITY)
            ? "#define MDL_HAS_CUTOUT_OPACITY 1\n"
            : "#define MDL_HAS_CUTOUT_OPACITY 0\n";
        m_hlsl_source_code +=
            interface_data.has_code_feature(Material_code_feature::HAS_AOVS)
            ? "#define MDL_HAS_AOVS 1\n"
            : "#define MDL_HAS_AOVS 0\n";
    }

    m_hlsl_source_code += "\n";

    m_hlsl_source_code += "#define MDL_NUM_TEXTURE_RESULTS " +
        std::to_string(m_app->get_options()->texture_results_cache_size) + "\n";

    m_hlsl_source_code += "\n";
    if (m_app->get_options()->automatic_derivatives) m_hlsl_source_code += "#define USE_DERIVS\n";
    if (m_app->get_options()->enable_auxiliary) m_hlsl_source_code += "#define ENABLE_AUXILIARY\n";

    // To illustrate the usage of the slot handles we just take maximum handle count.
    // With LPE support (see example df_cuda) it would make sense to limit the loop size per df.
    // The shader code will need to loop over the handles to collect the contribution of the
    // individual distribution function groupes defined by the handle parameters.
    {
        size_t max_slot_count = 0;
        for (size_t i = 0; i < (m_target_code ? m_target_code->get_callable_function_count() : 0); ++i)
        {
            max_slot_count =
                std::max(max_slot_count, m_target_code->get_callable_function_df_handle_count(i));
        }

        m_hlsl_source_code +=
            "// Note, real-world renderers would not support all slot modes.\n"
            "// A slot mode other than `none`, is only required for light path expressions (LPEs).\n"
            "#define MDL_DF_HANDLE_SLOT_COUNT ";
        m_hlsl_source_code += std::to_string(max_slot_count) + "\n";

        switch (m_app->get_options()->slot_mode)
        {
            case Base_options::SM_NONE:
                m_hlsl_source_code += "#define MDL_DF_HANDLE_SLOT_MODE -1\n";
                break;
            case Base_options::SM_FIXED_1:
                m_hlsl_source_code += "#define MDL_DF_HANDLE_SLOT_MODE 1\n";
                break;
            case Base_options::SM_FIXED_2:
                m_hlsl_source_code += "#define MDL_DF_HANDLE_SLOT_MODE 2\n";
                break;
            case Base_options::SM_FIXED_4:
                m_hlsl_source_code += "#define MDL_DF_HANDLE_SLOT_MODE 4\n";
                break;
            case Base_options::SM_FIXED_8:
                m_hlsl_source_code += "#define MDL_DF_HANDLE_SLOT_MODE 8\n";
                break;
        }
    }

    // since scene data access is more expensive than direct vertex data access and since
    // texture coordinates are extremely common, MDL typically fetches those from the state.
    // for demonstration purposes, this renderer uses the scene data instead which makes
    // texture coordinates optional
    m_hlsl_source_code += "\n";
    m_hlsl_source_code += "#define SCENE_DATA_ID_TEXCOORD_0 " +
        std::to_string(map_string_constant("TEXCOORD_0")) + "\n"; // registered before

    #if defined(MDL_ENABLE_MATERIALX)
        // add viewdirection for MaterialX NPR nodes
        // NOTE, using this in a path tracer is questionable
        m_hlsl_source_code += "#define SCENE_DATA_ID_CAMERA_POSITION " +
            std::to_string(map_string_constant("CAMERA_POSITION")) + "\n"; // registered before
    #endif

    m_hlsl_source_code += "\n";
    m_hlsl_source_code += "#include \"content/common.hlsl\"\n";
    m_hlsl_source_code += "#include \"content/mdl_renderer_runtime.hlsl\"\n\n";
    m_hlsl_source_code += m_target_code ? m_target_code->get_code() : "/* NO CODE GENERATED */\n";

    // add default implementations for all optional functions if not generated
    if (!interface_data.has_code_feature(Material_code_feature::HAS_INIT))
        m_hlsl_source_code += "void mdl_init(const Shading_state_material) {}\n";
    if (!interface_data.has_code_feature(Material_code_feature::SURFACE_SCATTERING))
        m_hlsl_source_code +=
        "void mdl_surface_scattering_evaluate(inout Bsdf_evaluate_data, const Shading_state_material) {}\n"
        "void mdl_surface_scattering_sample(inout Bsdf_sample_data, const Shading_state_material) {}\n"
        "void mdl_surface_scattering_auxiliary(inout Bsdf_auxiliary_data, const Shading_state_material) {}\n";
    if (!interface_data.has_code_feature(Material_code_feature::SURFACE_EMISSION))
        m_hlsl_source_code +=
        "void mdl_surface_emission_evaluate(inout Edf_evaluate_data, const Shading_state_material) {}\n"
        "void mdl_surface_emission_sample(inout Edf_sample_data, const Shading_state_material) {}\n"
        "void mdl_surface_emission_auxiliary(inout Edf_auxiliary_data, const Shading_state_material) {}\n"
        "float3 mdl_surface_emission_intensity(const Shading_state_material) { return float3(0, 0, 0); }\n";
    if (!interface_data.has_code_feature(Material_code_feature::BACKFACE_SCATTERING))
        m_hlsl_source_code +=
        "void mdl_backface_scattering_evaluate(inout Bsdf_evaluate_data, const Shading_state_material) {}\n"
        "void mdl_backface_scattering_sample(inout Bsdf_sample_data, const Shading_state_material) {}\n"
        "void mdl_backface_scattering_auxiliary(inout Bsdf_auxiliary_data, const Shading_state_material) {}\n";
    if (!interface_data.has_code_feature(Material_code_feature::BACKFACE_EMISSION))
        m_hlsl_source_code +=
        "void mdl_backface_emission_evaluate(inout Edf_evaluate_data, const Shading_state_material) {}\n"
        "void mdl_backface_emission_sample(inout Edf_sample_data, const Shading_state_material) {}\n"
        "void mdl_backface_emission_auxiliary(inout Edf_auxiliary_data, const Shading_state_material) {}\n"
        "float3 mdl_backface_emission_intensity(const Shading_state_material) { return float3(0, 0, 0); }\n";
    if (!interface_data.has_code_feature(Material_code_feature::VOLUME_ABSORPTION))
        m_hlsl_source_code +=
        "float3 mdl_volume_absorption_coefficient(const Shading_state_material) { return float3(0, 0, 0); }\n";
    if (!interface_data.has_code_feature(Material_code_feature::CUTOUT_OPACITY))
        m_hlsl_source_code +=
        "float mdl_standalone_geometry_cutout_opacity(const Shading_state_material) { return 1.0; }\n";
    if (!interface_data.has_code_feature(Material_code_feature::CAN_BE_THIN_WALLED))
        m_hlsl_source_code += "bool mdl_thin_walled(const Shading_state_material) { return false; }\n";

    // add functions to illustrate the usage of AOVs
    // A custom material type is probably very well known to the renderer or other components that use the code
    // Here, we add more generic functions to display AOVs depending on the type rather than feeding the results
    // into a simulation or post processing pipeline for example.
    {
        std::string aov_switch =
            "float3 mdl_aov(int index, const Shading_state_material state)\n"
            "{\n"
            "   switch(index)\n"
            "   {\n";

        // loop over the available AOVs to display
        if (interface_data.has_code_feature(Material_code_feature::HAS_AOVS))
        {
            for (size_t i = 0; i < interface_data.aovs.size(); ++i)
            {
                aov_switch += mi::examples::strings::format(
                    "       case %d:\n", int(interface_data.aovs[i].runtime_index));

                // handle the type conversion to output a visible color (just for illustration)
                const std::string& type = interface_data.aovs[i].mdl_type_name == "color"
                    ? "float3" 
                    : interface_data.aovs[i].mdl_type_name;

                // select components to create a color
                std::string postfix = ".xxx";
                if (type.back() == '4' or type.back() == '3')
                    postfix = ".xyz";
                else if (type.back() == '2')
                    postfix = ".xy, 0.0f";

                aov_switch += mi::examples::strings::format(
                    "           return float3(%s(state)%s);\n",
                    interface_data.aovs[i].hlsl_name.c_str(), postfix.c_str());
            }
        }

        aov_switch +=
            "       default:\n"
            "           return float3(0.0f, 0.0f, 0.0f);\n"
            "   }\n"
            "}\n\n";
        m_hlsl_source_code += aov_switch;
    }

    // this last snipped contains the actual hit shader and the renderer logic
    // ideally, this is the only part that is handwritten
    m_hlsl_source_code += "\n\n#include \"content/mdl_hit_programs.hlsl\"\n\n";

    command_queue->execute_command_list(command_list);

    m_generation_required = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

const mi::neuraylib::ITarget_code* Mdl_material_target::get_generated_target() const
{
    if (!m_target_code)
        return nullptr;

    m_target_code->retain();
    return m_target_code.get();
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_target::compile()
{
    if (!m_compilation_required)
    {
        log_info("Target code does not need compilation. Hash: " + m_compiled_material_hash, SRC);
        return true;
    }

    // generate has be called first
    if (m_generation_required)
    {
        log_error("Compiling HLSL target code not possible before generation. Hash: " +
            m_compiled_material_hash, SRC);
        return false;
    }

    // compile to DXIL
    {
        auto p = m_app->get_profiling().measure("compiling HLSL to DXIL");
        // use the material name of the first material
        std::string pseudo_file_name = "link_unit_code";
        if (m_materials.size() > 0)
        {
            for (const auto& it : m_materials)
            {
                if (!it.second)
                    continue;
                std::string mat_name = it.second->get_material_desciption().get_scene_name();
                std::replace(mat_name.begin(), mat_name.end(), ' ', '_');
                mat_name.erase(std::remove_if(mat_name.begin(), mat_name.end(),
                    [](auto const& c) { return !std::isalnum(c) && c != '_'; }), mat_name.end());
                if (!mat_name.empty())
                    pseudo_file_name = mat_name;
                break;
            }
        }

        // run the shader compiler to produce one or multiple dxil libraries
        Shader_compiler compiler(m_app);
        m_dxil_compiled_libraries = compiler.compile_shader_library_from_string(
            m_app->get_options(),
            get_hlsl_source_code(), pseudo_file_name + "_" + get_shader_name_suffix(),
            &m_hlsl_compilation_defines,
            { m_radiance_closest_hit_name, m_radiance_any_hit_name, m_shadow_any_hit_name });
    }

    bool success = !m_dxil_compiled_libraries.empty();
    m_compilation_required = !success;

    // write the shader cache if enabled
    while (success && m_sdk->get_options().enable_shader_cache)
    {
        auto p = m_app->get_profiling().measure("writing to shader cache");

        // create context to get results from the serialization
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

        // discard the instance specific data, the argument blocks are generated from scratch
        context->set_option("serialize_class_instance_data", false);

        // start the actual serialization
        mi::base::Handle<const mi::neuraylib::IBuffer> tci_buffer(
            m_target_code->serialize(context.get()));

        if (!m_sdk->log_messages("MDL target code serialization failed.", context.get(), SRC))
            return false;

        // open cache file
        const std::string filename = compute_shader_cache_filename(
            *m_app->get_options(), m_compiled_material_hash, true);

        // create the parent folder if required
        const std::string folder = mi::examples::io::dirname(filename);
        if (!mi::examples::io::mkdir(folder, true))
        {
            log_warning("Failed to create shader cache folder: " + folder);
            break;
        }

        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (!file.is_open())
        {
            log_warning("Failed to write shader cache: " + filename);
            break;
        }

        // write target code information
        const size_t tci_buffer_size = tci_buffer->get_data_size();
        const mi::Uint8* tci_buffer_data = tci_buffer->get_data();
        file.write(reinterpret_cast<const char*>(&tci_buffer_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(tci_buffer_data), tci_buffer_size);

        // write the interface information
        const Mdl_material_target_interface& interface_data =
            m_materials.begin()->second->get_target_interface();
        file.write(reinterpret_cast<const char*>(
            &interface_data), sizeof(Mdl_material_target_interface));

        // write dxil libraries
        // with support for slang the we can have multiple libraries per material, i.e., one
        // per entry point. To get a mapping between entry point and library we also need the
        // exported symbol name which makes the de/serialization a bit more elaborate.

        size_t num_libraries = m_dxil_compiled_libraries.size();
        file.write(reinterpret_cast<const char*>(&num_libraries), sizeof(size_t));

        for (size_t l = 0; l < num_libraries; ++l)
        {
            // first, the symbols, starting with the number of symbols ...
            const std::vector<std::string>& exports = m_dxil_compiled_libraries[l].get_exports();
            size_t num_exp_symbols = exports.size();
            file.write(reinterpret_cast<const char*>(&num_exp_symbols), sizeof(size_t));
            for (size_t s = 0; s < num_exp_symbols; ++s)
            {
                // ... and then for each, string length and string data
                size_t symbol_size = exports[s].size();
                file.write(reinterpret_cast<const char*>(&symbol_size), sizeof(size_t));
                file.write(exports[s].data(), symbol_size);
            }

            // the dxil blob
            const size_t dxil_blob_buffer_size = m_dxil_compiled_libraries[l].get_dxil_library()->GetBufferSize();
            const LPVOID dxil_blob_buffer = m_dxil_compiled_libraries[l].get_dxil_library()->GetBufferPointer();
            file.write(reinterpret_cast<const char*>(&dxil_blob_buffer_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(dxil_blob_buffer), dxil_blob_buffer_size);
        }

        file.close();
        break;
    }

    return success;
}

}}} // mi::examples::mdl_d3d12
