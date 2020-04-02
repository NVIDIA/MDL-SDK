/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "example_shared.h"


namespace mdl_d3d12
{
    Mdl_material_target::Resource_callback::Resource_callback(
        Mdl_sdk* sdk, Mdl_material_target* target, Mdl_material* material)
        : m_sdk(sdk)
        , m_target(target)
        , m_material(material)
    {
    }

    // --------------------------------------------------------------------------------------------

    mi::Uint32 Mdl_material_target::Resource_callback::get_resource_index(
        mi::neuraylib::IValue_resource const *resource)
    {
        // resource available in the target code.
        // this is the case for resources that are in the material body and for
        // resources in contained in the parameters of the first appearance of a material
        mi::Uint32 index = m_sdk->get_transaction().execute<mi::Uint32>(
            [&](mi::neuraylib::ITransaction* t)
        {
            mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
                m_target->get_target_code());
            return target_code->get_known_resource_index(t, resource);
        });
        if (index > 0)
            return index;

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

            // currently not supported by this example
            // case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            //     kind = Mdl_resource_kind::Light_profile;
            //     break;
            // 
            // case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            //     kind = Mdl_resource_kind::Bsdf_measurement;
            //     break;

            default:
                log_error("Invalid resource kind for: " + std::string(name), SRC);
                return 0;
        }

        // store textures at the material
        size_t mat_resource_index = m_material->register_resource(kind, dimension, name);

        // log these manually defined indices
        log_info(
            "target code id: " + std::to_string(m_target->get_id()) +
            " - texture id: " + std::to_string(mat_resource_index) +
            " (material id: " + std::to_string(m_material->get_id()) + ")" +
            " - resource: " + std::string(name) + " (reused material)");

        return mat_resource_index;
    }

    mi::Uint32 Mdl_material_target::Resource_callback::get_string_index(
        mi::neuraylib::IValue_string const *s)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
            m_target->get_target_code());

        for (mi::Size i = 0, n = target_code->get_string_constant_count(); i < n; ++i)
            if (strcmp(target_code->get_string_constant(i), s->get_value()) == 0)
                return static_cast<mi::Uint32>(i);

        log_error("TODO: String constant not found.", SRC);
        return 0;
    }

    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    namespace { std::atomic<size_t> sa_target_code_id(0); }
    Mdl_material_target::Mdl_material_target(Base_application* app, Mdl_sdk* sdk)
        : m_app(app)
        , m_sdk(sdk)
        , m_id(sa_target_code_id.fetch_add(1))
        , m_target_code(nullptr)
        , m_hlsl_source_code("")
        , m_dxil_compiled_library(nullptr)
        , m_read_only_data_segment(nullptr)
        , m_target_resources()
        , m_generation_required(true)
        , m_compilation_required(true)
    {
        // add the empty resources
        for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
        {
            Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
            m_target_resources[kind_i] = std::vector<Mdl_resource_assignment>();
        }
    }

    // --------------------------------------------------------------------------------------------

    Mdl_material_target::~Mdl_material_target()
    {
        m_target_code = nullptr;
        m_dxil_compiled_library = nullptr;

        if (m_read_only_data_segment) delete m_read_only_data_segment;
    }

    // --------------------------------------------------------------------------------------------

    mi::neuraylib::ITarget_resource_callback* Mdl_material_target::create_resource_callback(
        Mdl_material* material)
    {
        return new Resource_callback(m_sdk, this, material);
    }

    // --------------------------------------------------------------------------------------------

    const mi::neuraylib::ITarget_code* Mdl_material_target::get_target_code() const
    {
        if (!m_target_code)
            return nullptr;

        m_target_code->retain();
        return m_target_code.get();
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_target::add_material(
        Mdl_material_target_interface& interface_data,
        Mdl_material* material,
        mi::neuraylib::ILink_unit* link_unit,
        mi::neuraylib::IMdl_execution_context* context)
    {
        // since all expression will be in the same link unit, the need to be identified
        std::string scattering_name = "mdl_df_scattering_" + std::to_string(material->get_id());
        std::string opacity_name = "mdl_opacity_" + std::to_string(material->get_id());

        std::string emission_name = "mdl_df_emission_" + std::to_string(material->get_id());
        std::string emission_intensity_name =
            "mdl_emission_intensity_" + std::to_string(material->get_id());

        std::string thin_walled_name = "mdl_thin_walled_" + std::to_string(material->get_id());
        std::string hair_name = "mdl_hair_" + std::to_string(material->get_id());

        // select expressions to generate HLSL code for
        std::vector<mi::neuraylib::Target_function_description> selected_functions;

        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "surface.scattering", scattering_name.c_str()));

        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "geometry.cutout_opacity", opacity_name.c_str()));

        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "surface.emission.emission", emission_name.c_str()));

        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "surface.emission.intensity", emission_intensity_name.c_str()));

        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "thin_walled", thin_walled_name.c_str()));

        selected_functions.push_back(mi::neuraylib::Target_function_description(
            "hair", hair_name.c_str()));

        // get the compiled material and add the material to the link unit
        mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
            m_sdk->get_transaction().access<const mi::neuraylib::ICompiled_material>(
                material->get_material_compiled_db_name().c_str()));

        link_unit->add_material(
            compiled_material.get(),
            selected_functions.data(), selected_functions.size(),
            context);

        if (!m_sdk->log_messages(context))
            return false;

        // get the resulting target code information

        // function index for "surface.scattering"
        interface_data.indices.scattering_function_index =
            selected_functions[0].function_index == static_cast<mi::Size>(-1)
            ? -1
            : static_cast<int32_t>(selected_functions[0].function_index);

        // function index for "geometry.cutout_opacity"
        interface_data.indices.opacity_function_index =
            selected_functions[1].function_index == static_cast<mi::Size>(-1)
            ? -1
            : static_cast<int32_t>(selected_functions[1].function_index);

        // function index for "surface.emission.emission"
        interface_data.indices.emission_function_index =
            selected_functions[2].function_index == static_cast<mi::Size>(-1)
            ? -1
            : static_cast<int32_t>(selected_functions[2].function_index);

        // function index for "surface.emission.intensity"
        interface_data.indices.emission_intensity_function_index =
            selected_functions[3].function_index == static_cast<mi::Size>(-1)
            ? -1
            : static_cast<int32_t>(selected_functions[3].function_index);

        // function index for "thin_walled"
        interface_data.indices.thin_walled_function_index =
            selected_functions[4].function_index == static_cast<mi::Size>(-1)
            ? -1
            : static_cast<int32_t>(selected_functions[4].function_index);

        // function index for "hair"
        interface_data.indices.hair_function_index =
            selected_functions[5].function_index == static_cast<mi::Size>(-1)
            ? -1
            : static_cast<int32_t>(selected_functions[5].function_index);

        // if the material contains a hair BSDF, we use that instead of the surface BSDF
        // TODO remove after testing
        if (contains_hair_bsdf(compiled_material.get()))
            interface_data.indices.scattering_function_index = 
                interface_data.indices.hair_function_index;

        // also constant for the entire material
        interface_data.argument_layout_index = selected_functions[0].argument_block_index;
        return true;
    }

    // --------------------------------------------------------------------------------------------

    /// Keep a pointer (no ownership) to the material for notifying the material when the 
    /// target code generation is finished.
    void Mdl_material_target::register_material(Mdl_material* material)
    {
        Mdl_material_target* current_target = material->get_target_code();

        // mark changed because registered material is called only for new or changed materials
        m_generation_required = true;
        m_compilation_required = true;

        // already registered with this target code
        if (current_target == this)
            return;

        // unregister from current target
        else if (current_target != nullptr)
        {
            std::unique_lock<std::mutex> lock(current_target->m_materials_mtx);
            auto found = current_target->m_materials.find(material->get_id());
            if (found != current_target->m_materials.end())
            {
                current_target->m_materials.erase(found);
                current_target->m_generation_required = true;
                current_target->m_compilation_required = true;
            }
        }

        // register with this target code
        std::unique_lock<std::mutex> lock(m_materials_mtx);
        m_materials[material->get_id()] = material;
    }

    // --------------------------------------------------------------------------------------------

    size_t Mdl_material_target::get_material_resource_count(
        Mdl_resource_kind kind) const
    {
        const auto& found = m_material_resource_count.find(kind);
        return found->second;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_target::visit_materials(std::function<bool(Mdl_material*)> action)
    {
        std::lock_guard<std::mutex> lock(m_materials_mtx);
        for (auto it = m_materials.begin(); it != m_materials.end(); it++)
            if (!action(it->second))
                return false;
        return true;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_target::generate()
    {
        if (!m_generation_required)
        {
            log_info("Target code does not need generation. ID: " + std::to_string(m_id), SRC);
            return true;
        }

        // create a command list for uploading data to the GPU
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        // since this method can be called from multiple threads simultaneously
        // a new context for is created
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

        mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(
            m_sdk->get_backend().create_link_unit(m_sdk->get_transaction().get(), context.get()));

        if (!m_sdk->log_messages(context.get()))
        {
            log_error("MDL creating a link unit failed.", SRC);
            return false;
        }

        // empty resource list (in case of reload) and rest the counter
        for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
        {
            Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
            m_target_resources[kind_i].clear();
        }

        // add materials to link unit
        std::unordered_map<std::string, Mdl_material_target_interface> processed_hashes;
        for (auto& pair : m_materials)
        {
            // add materials with the same hash only once
            const std::string& hash = pair.second->get_material_compiled_hash();
            auto found = processed_hashes.find(hash);
            if (found == processed_hashes.end())
            {
                // add this material to the link unit
                Mdl_material_target_interface interface_data;
                if (!add_material(interface_data, pair.second, link_unit.get(), context.get()))
                {
                    log_error("Adding to link unit failed: " + pair.second->get_name(), SRC);
                    return false;
                }
                processed_hashes.emplace(hash, interface_data);
            }

            // pass target information to the material
            pair.second->set_target_interface(this, processed_hashes[hash]);
        }

        // generate HLSL code
        m_target_code = m_sdk->get_backend().translate_link_unit(link_unit.get(), context.get());
        if (!m_sdk->log_messages(context.get()))
        {
            log_error("MDL target code generation failed.", SRC);
            return false;
        }

        // add all textures known to the link unit
        for (size_t t = 1, n = m_target_code->get_texture_count(); t < n; ++t)
        {
            Mdl_resource_assignment set(Mdl_resource_kind::Texture);
            set.resource_name = m_target_code->get_texture(t);
            set.runtime_resource_id = t;

            switch (m_target_code->get_texture_shape(t))
            {
                case mi::neuraylib::ITarget_code::Texture_shape_2d:
                    set.dimension = Texture_dimension::Texture_2D;
                    break;

                case mi::neuraylib::ITarget_code::Texture_shape_3d:
                case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data:
                    set.dimension = Texture_dimension::Texture_3D;
                    break;

                default:
                    log_error("Only 2D and 3D textures are supported by this example.", SRC);
                    return false;
            }

            m_target_resources[Mdl_resource_kind::Texture].emplace_back(set);
        }

        // create per material resources, parameter bindings, ...
        // ------------------------------------------------------------

        // ... in parallel, if not forced otherwise
        std::vector<std::thread> tasks;
        std::atomic_bool success = true;
        for (auto mat : m_materials)
        {
            // sequentially
            if (m_app->get_options()->force_single_theading)
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
            {
                // TODO free block
                m_first_resource_heap_handle = Descriptor_heap_handle(); // reset
            }
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

        // read-only data
        size_t ro_data_seg_index = 0; // ?
        if (m_target_code->get_ro_data_segment_count() > ro_data_seg_index)
        {
            const char* name = m_target_code->get_ro_data_segment_name(ro_data_seg_index);
            auto read_only_data_segment = new Buffer(
                m_app, m_target_code->get_ro_data_segment_size(ro_data_seg_index),
                "MDL_ReadOnly_" + std::string(name));

            read_only_data_segment->set_data(
                m_target_code->get_ro_data_segment_data(ro_data_seg_index),
                m_target_code->get_ro_data_segment_size(ro_data_seg_index));

            if (!m_read_only_data_segment) delete m_read_only_data_segment;
            m_read_only_data_segment = read_only_data_segment;
        }

        if(m_read_only_data_segment == nullptr)
        {
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

        // generate the actual shader code with the help of some snippets
        m_hlsl_source_code.clear();

        // per target data
        m_hlsl_source_code += "#define MDL_TARGET_REGISTER_SPACE space2\n";
        m_hlsl_source_code += "#define MDL_TARGET_RO_DATA_SEGMENT_SLOT t0\n";
        m_hlsl_source_code += "\n";

        // per material data
        m_hlsl_source_code += "#define MDL_MATERIAL_REGISTER_SPACE space3\n"; // there are more
        m_hlsl_source_code += "#define MDL_MATERIAL_ARGUMENT_BLOCK_SLOT t1\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_RESOURCE_INFO_SLOT t2\n";
        m_hlsl_source_code += "\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_TEXTURE_2D_REGISTER_SPACE space4\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_TEXTURE_3D_REGISTER_SPACE space5\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_TEXTURE_SLOT_BEGIN t0\n";
        m_hlsl_source_code += "\n";

        // global data
        m_hlsl_source_code += "#define MDL_TEXTURE_SAMPLER_SLOT s0\n";
        m_hlsl_source_code += "#define MDL_LATLONGMAP_SAMPLER_SLOT s1\n";
        m_hlsl_source_code += "#define MDL_NUM_TEXTURE_RESULTS " +
            std::to_string(m_app->get_options()->texture_results_cache_size) + "\n";

        m_hlsl_source_code += "\n";
        if (m_app->get_options()->automatic_derivatives) m_hlsl_source_code += "#define USE_DERIVS\n";
        if (m_app->get_options()->enable_auxiliary) m_hlsl_source_code += "#define ENABLE_AUXILIARY\n";
        m_hlsl_source_code += "#define MDL_DF_HANDLE_SLOT_MODE -1\n";

        // since scene data access is more expensive than direct vertex data access and since 
        // texture coordinates are extremely common, MDL typically fetches those from the state.
        // for demonstration purposes, this renderer uses the scene data instead which makes 
        // texture coordinates optional
        m_hlsl_source_code += "\n";
        m_hlsl_source_code += "#define SCENE_DATA_ID_TEXCOORD_0 " +
            std::to_string(std::max(m_target_code->get_string_constant_count(), mi::Size(1))) + "\n";

        m_hlsl_source_code += "\n";
        m_hlsl_source_code += "#include \"content/common.hlsl\"\n";
        m_hlsl_source_code += "#include \"content/mdl_target_code_types.hlsl\"\n";
        m_hlsl_source_code += "#include \"content/mdl_renderer_runtime.hlsl\"\n\n";
        m_hlsl_source_code += m_target_code->get_code();

        // assuming multiple materials that have been compiled to this target/link unit
        // it has to be possible to select the individual functions based on the hit object

        std::string init_switch_function[2] = {
        {
            "void mdl_bsdf_init(in uint function_index, inout Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        },
        {
            "void mdl_edf_init(in uint function_index, inout Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        }};

        std::string sample_switch_function[2] = {
            "void mdl_bsdf_sample(in uint function_index, inout Bsdf_sample_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_sample(in uint function_index, inout Edf_sample_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string evaluate_switch_function[2] = {
            "void mdl_bsdf_evaluate(in uint function_index, inout Bsdf_evaluate_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_evaluate(in uint function_index, inout Edf_evaluate_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string pdf_switch_function[2] = {
            "void mdl_bsdf_pdf(in uint function_index, inout Bsdf_pdf_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_pdf(in uint function_index, inout Edf_pdf_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string auxiliary_switch_function[2] = {
            "void mdl_bsdf_auxiliary(in uint function_index, inout Bsdf_auxiliary_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_auxiliary(in uint function_index, inout Edf_auxiliary_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string opacity_switch_function =
            "float mdl_geometry_cutout_opacity(in uint function_index, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n";

        std::string emission_intensity_switch_function =
            "float3 mdl_emission_intensity(in uint function_index, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n";

        std::string thin_walled_switch_function =
            "bool mdl_thin_walled(in uint function_index, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n";

        for (size_t f = 0, n = m_target_code->get_callable_function_count(); f < n; ++f)
        {
            mi::neuraylib::ITarget_code::Function_kind func_kind =
                m_target_code->get_callable_function_kind(f);

            mi::neuraylib::ITarget_code::Distribution_kind dist_kind =
                m_target_code->get_callable_function_distribution_kind(f);

            std::string name = m_target_code->get_callable_function(f);


            if (dist_kind == mi::neuraylib::ITarget_code::DK_NONE)
            {
                if (str_starts_with(name, "mdl_opacity_"))
                {
                    opacity_switch_function += "       case " + std::to_string(f) + ": " +
                        "return " + name + "(state);\n";
                }
                else if (str_starts_with(name, "mdl_emission_intensity_"))
                {
                    emission_intensity_switch_function += "       case " + std::to_string(f) + ": " +
                        "return " + name + "(state);\n";
                }
                else if (str_starts_with(name, "mdl_thin_walled_"))
                {
                    thin_walled_switch_function += "       case " + std::to_string(f) + ": " +
                        "return " + name + "(state);\n";
                }
            }
            else if (dist_kind == mi::neuraylib::ITarget_code::DK_BSDF ||
                     dist_kind == mi::neuraylib::ITarget_code::DK_HAIR_BSDF ||
                     dist_kind == mi::neuraylib::ITarget_code::DK_EDF)
            {
                // store BSDFs and Hair BSDFs at index 0 and EDFs at index 1
                size_t index = dist_kind == mi::neuraylib::ITarget_code::DK_EDF ? 1 : 0;

                switch (func_kind)
                {
                case mi::neuraylib::ITarget_code::FK_DF_INIT:
                    init_switch_function[index] +=
                        "       case " + std::to_string(f) + ": " +
                        name + "(state); return;\n";
                    break;

                case mi::neuraylib::ITarget_code::FK_DF_SAMPLE:
                    sample_switch_function[index] +=
                        "       case " + std::to_string(f - 1) + ": " +
                        name + "(sret_ptr, state); return;\n";
                    break;

                case mi::neuraylib::ITarget_code::FK_DF_EVALUATE:
                    evaluate_switch_function[index] +=
                        "       case " + std::to_string(f - 2) + ": " +
                        name + "(sret_ptr, state); return;\n";
                    break;

                case mi::neuraylib::ITarget_code::FK_DF_PDF:
                    pdf_switch_function[index] +=
                        "       case " + std::to_string(f - 3) + ": " +
                        name + "(sret_ptr, state); return;\n";
                    break;

                case mi::neuraylib::ITarget_code::FK_DF_AUXILIARY:
                    auxiliary_switch_function[index] +=
                        "       case " + std::to_string(f - 4) + ": " +
                        name + "(sret_ptr, state); return;\n";
                    break;
                }
            }

        }

        for (size_t i = 0; i < 2; ++i)
        {
            init_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += init_switch_function[i];

            sample_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += sample_switch_function[i];

            evaluate_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += evaluate_switch_function[i];

            pdf_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += pdf_switch_function[i];

            auxiliary_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += auxiliary_switch_function[i];
        }

        opacity_switch_function +=
            "       default: break;\n"
            "   }\n"
            "   return 1.0f;\n"
            "}\n\n";
        m_hlsl_source_code += opacity_switch_function;

        emission_intensity_switch_function +=
            "       default: break;\n"
            "   }\n"
            "   return float3(1.0f, 1.0f, 1.0f);\n"
            "}\n\n";
        m_hlsl_source_code += emission_intensity_switch_function;

        thin_walled_switch_function +=
            "       default: break;\n"
            "   }\n"
            "   return false;\n"
            "}\n\n";
        m_hlsl_source_code += thin_walled_switch_function;


        // this last snipped contains the actual hit shader and the renderer logic
        // ideally, this is the only part that is handwritten
        m_hlsl_source_code += "\n\n#include \"content/mdl_hit_programs.hlsl\"\n\n";

        // write to file for debugging purpose
        std::ofstream file_stream;
        file_stream.open(get_executable_folder() + "/link_unit_code_" + std::to_string(m_id) + ".hlsl");
        if (file_stream)
        {
            file_stream << m_hlsl_source_code.c_str();
            file_stream.close();
        }

        command_queue->execute_command_list(command_list);

        m_target_code = nullptr;

        m_generation_required = false;
        return true;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_target::compile()
    {
        if (!m_compilation_required)
        {
            log_info("Target code does not need compilation. ID: " + std::to_string(m_id), SRC);
            return true;
        }

        // generate has be called first
        if (m_generation_required)
        {
            log_error("Compiling HLSL target code not possible before generation. Target ID: "
                      + std::to_string(m_id), SRC);
            return false;
        }

        // compile to DXIL
        {
            std::map<std::string, std::string> defines;
            defines["TARGET_CODE_ID"] = std::to_string(m_id);

            Shader_compiler compiler;
            m_dxil_compiled_library = compiler.compile_shader_library_from_string(
                get_hlsl_source_code(), "link_unit_code_" + std::to_string(m_id),
                &defines);
        }

        bool success = m_dxil_compiled_library != nullptr;
        m_compilation_required = !success;
        return success;
    }
}

