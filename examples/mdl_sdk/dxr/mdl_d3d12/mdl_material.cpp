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

#include "mdl_material.h"
#include "base_application.h"
#include "command_queue.h"
#include "descriptor_heap.h"
#include "mdl_material_description.h"
#include "mdl_material_library.h"
#include "mdl_sdk.h"
#include "texture.h"
#include "shader.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{
    std::atomic<uint32_t> s_material_id_counter(0);
}

Mdl_material::Mdl_material(Base_application* app)
    : m_app(app)
    , m_sdk(&app->get_mdl_sdk())
    , m_material_id(s_material_id_counter.fetch_add(1))
    , m_name("material_" + std::to_string(m_material_id))
    , m_description()
    , m_flags(IMaterial::Flags::None)
    , m_compiled_hash("")
    , m_resource_hash("")
    , m_target(nullptr)
    , m_scene_data_name_map()
    , m_constants(m_app, m_name + "_Constants")
    , m_argument_block_buffer(nullptr) // size is not known at this point
    , m_argument_block_data()
    , m_argument_layout_index(static_cast<mi::Size>(-1))
    , m_first_resource_heap_handle()
    , m_resources()
    , m_resource_infos(nullptr)
{
    // add the empty resources
    for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
    {
        Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
        m_resources[kind_i] = std::vector<Mdl_resource_assignment>();
    }

    m_constants.data.material_id = m_material_id;
}

// ------------------------------------------------------------------------------------------------

Mdl_material::~Mdl_material()
{
    if (m_argument_block_buffer) delete m_argument_block_buffer;
    if (m_resource_infos) delete m_resource_infos;

    m_target->unregister_material(this);
    m_target = nullptr;

    // free heap block
    m_app->get_resource_descriptor_heap()->free_views(m_first_resource_heap_handle);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material::compile_material(const Mdl_material_description& description)
{
    // since this method can be called from multiple threads simultaneously
    // a new context for is created
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        m_sdk->create_context());

    // copy description
    m_description = description;

    // load the definitions
    std::string scene_directory = mi::examples::io::dirname(m_app->get_scene_path());
    m_description.load_material_definition(*m_sdk, scene_directory, context.get());
    if (!m_description.is_loaded())
        return false; // error case, no fall-back

    // get the scene for this material
    set_name(m_description.get_scene_name());

    // database names for the instance and the compiled material
    std::string base = "mdl_dxr::material_" + std::to_string(m_material_id) + "_";
    m_instance_db_name = base + "instance";
    m_compiled_db_name = base + "compiled";

    // get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        m_sdk->get_transaction().access<mi::neuraylib::IMaterial_definition>(
            m_description.get_material_defintion_db_name()));

    // create a material instance with the provided parameters and store it in the database
    {
        auto p = m_app->get_profiling().measure("creating MDL material instance");
        mi::Sint32 ret = 0;
        mi::base::Handle<const mi::neuraylib::IExpression_list> parameter_list(
            m_description.get_parameters());


        mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
            material_definition->create_material_instance(parameter_list.get(), &ret));
        if (ret != 0 || !material_instance)
        {
            log_error("Instantiating material '" + get_name() + "' failed", SRC);
            return false;
        }
        m_sdk->get_transaction().store(material_instance.get(), m_instance_db_name.c_str());
    }

    // the instance is setup and in the DB, now it is the same as recompiling
    return recompile_material(context.get());
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material::recompile_material(mi::neuraylib::IMdl_execution_context* context)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
        m_sdk->get_transaction().access<const mi::neuraylib::IMaterial_instance>(
            m_instance_db_name.c_str())); // get back after storing

    // is this instance still valid
    if (!material_instance->is_valid(context))
    {
        mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance_edit(
            m_sdk->get_transaction().edit<mi::neuraylib::IMaterial_instance>(
                m_instance_db_name.c_str())); // get back after storing

        // try to repair
        m_sdk->log_messages(
            "Repairing the material instance: " + get_name(), context);

        material_instance_edit->repair(
            mi::neuraylib::MDL_REPAIR_INVALID_ARGUMENTS |
            mi::neuraylib::MDL_REMOVE_INVALID_ARGUMENTS, context);

        // if repair fails, the expression graph can be fixed manually or even automatically
        // by recursively matching parameter types and names and adding default values where
        // this matching fails

        // the reason is reported to the log that reload failed and use fall-backs
        if (!material_instance_edit->is_valid(context))
        {
            m_sdk->log_messages(
                "Repairing the material instance failed: " + get_name(), context);

            // try to recreate the instance from scratch
            if (compile_material(get_material_desciption()))
            {
                log_info("Recreated the material instance from scratch: " + get_name());
                return false;
            }

            // fall-back, the invalid material
            IScene_loader::Material copy = get_material_desciption().get_material_parameters();
            copy.name = Mdl_material_description::Invalid_material_identifier;
            Mdl_material_description invalid(copy); // invalid but keep material parameters
            if (compile_material(invalid))
            {
                log_warning("Repairing and recreating the material instance failed. "
                            "Using invalid material as fall-back: " + get_name());
                return false;
            }

            log_error("[FATAL] Repairing and all fall-backs failed: " + get_name());
            return false;
        }

        // repairing successful
        material_instance = material_instance_edit;
    }

    // compile the instance and store it in the database
    {
        auto p = m_app->get_profiling().measure("creating MDL compiled material");

        Mdl_sdk::Options& mdl_options = m_sdk->get_options();

        mi::Uint32 flags = mdl_options.use_class_compilation
            ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
            : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

        // performance optimizations available only in class compilation mode
        // (all parameters are folded in instance mode)
        context->set_option("fold_all_bool_parameters", mdl_options.fold_all_bool_parameters);
        context->set_option("fold_all_enum_parameters", mdl_options.fold_all_enum_parameters);

        mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
            material_instance->create_compiled_material(flags, context));
        if (!m_sdk->log_messages("Compiling material failed: " + get_name(), context, SRC) ||
            !compiled_material)
                return false;
        m_sdk->get_transaction().store(compiled_material.get(), m_compiled_db_name.c_str());
    }

    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        m_sdk->get_transaction().access<const mi::neuraylib::ICompiled_material>(
        m_compiled_db_name.c_str())); // get back after storing

    // generate the hash compiled material hash
    // this is used to check if a new target is required or if an existing one can be reused.
    const std::string old_hash = m_compiled_hash;
    const mi::base::Uuid hash = compiled_material->get_hash();
    m_compiled_hash  = std::to_string(hash.m_id1);
    m_compiled_hash += std::to_string(hash.m_id2);
    m_compiled_hash += std::to_string(hash.m_id3);
    m_compiled_hash += std::to_string(hash.m_id4);

    if (old_hash.empty())
        log_verbose("Hash: " + get_name() + ": " + m_compiled_hash);
    else
    {
        log_verbose("Old Hash: " + get_name() + ": " + old_hash);
        log_verbose("New Hash: " + get_name() + ": " + m_compiled_hash);
    }

    // some minor optimization
    // use inspection to override the opacity option
    m_flags = mi::examples::enums::remove_flag(m_description.get_flags(), IMaterial::Flags::Opaque);
    float opacity = -1.0f;
    if (compiled_material->get_cutout_opacity(&opacity) && opacity >= 1.0f)
        m_flags = mi::examples::enums::set_flag(m_description.get_flags(), IMaterial::Flags::Opaque);

    return true;
}

// ------------------------------------------------------------------------------------------------

void Mdl_material::set_target_interface(
    Mdl_material_target* target,
    const Mdl_material_target_interface& target_data)
{
    m_target = target;
    m_constants.data.function_indices = target_data.indices;
    m_argument_layout_index = target_data.argument_layout_index;
    m_material_target_interface = target_data;
}

// ------------------------------------------------------------------------------------------------

const Descriptor_table Mdl_material::get_descriptor_table()
{
    // same as the static case + additional per material resources
    Descriptor_table table;

    // bind material constants
    table.register_cbv(0, 3, 0);

    // bind material argument block
    table.register_srv(1, 3, 1);

    // bind per material resources info
    table.register_srv(2, 3, 2);

    // bind textures
    size_t tex_count =
        std::max(size_t(1), m_target->get_material_resource_count(Mdl_resource_kind::Texture));
    table.register_srv(0, 4, 3, tex_count);
    table.register_srv(0, 5, 3, tex_count);
    return table;
}

// ------------------------------------------------------------------------------------------------

std::vector<D3D12_STATIC_SAMPLER_DESC> Mdl_material::get_sampler_descriptions()
{
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;

    // for standard textures
    D3D12_STATIC_SAMPLER_DESC  desc = {};
    desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.MipLODBias = 0.0f;
    desc.MaxAnisotropy = 16;
    desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    desc.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    desc.MinLOD = 0;
    desc.MaxLOD = 0;  // no mip-maps, otherwise use D3D11_FLOAT32_MAX;
    desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    desc.ShaderRegister = 0;                        // bind sampler to shader register(s0)
    desc.RegisterSpace = 0;
    samplers.push_back(desc);

    // for latitude-longitude environment maps
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.ShaderRegister = 1;                        // bind sampler to shader register(s1)
    samplers.push_back(desc);

    return samplers;
}

// ------------------------------------------------------------------------------------------------

size_t Mdl_material::get_target_code_id() const
{
    return m_target->get_id();
}

// ------------------------------------------------------------------------------------------------

const std::unordered_map<std::string, uint32_t>& Mdl_material::get_scene_data_name_map() const
{
    return m_scene_data_name_map;
}

// ------------------------------------------------------------------------------------------------

const mi::neuraylib::ITarget_value_layout* Mdl_material::get_argument_layout() const
{
    if (m_argument_layout_index == static_cast<mi::Size>(-1))
        return nullptr;

    // get the target code
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(m_target->get_target_code());
    if (!target_code)
        return nullptr;

    // get the layout
    mi::base::Handle<const mi::neuraylib::ITarget_value_layout> arg_layout(
        target_code->get_argument_block_layout(m_argument_layout_index));

    if (!arg_layout)
        return nullptr;

    arg_layout->retain();
    return arg_layout.get();
}

// ------------------------------------------------------------------------------------------------

uint8_t* Mdl_material::get_argument_data()
{
    if (m_argument_block_data.empty())
        return nullptr;

    return m_argument_block_data.data();
}

// ------------------------------------------------------------------------------------------------

void Mdl_material::update_material_parameters()
{
    m_argument_block_buffer->set_data(m_argument_block_data);

    // assuming material parameters do not change on a per frame basis ...
    Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    D3DCommandList* command_list = command_queue->get_command_list();
    m_argument_block_buffer->upload(command_list);
    command_queue->execute_command_list(command_list);

    m_constants.data.material_flags = static_cast<uint32_t>(m_flags);
}

// ------------------------------------------------------------------------------------------------

// Called by the target to register per material resources.
size_t Mdl_material::register_resource(
    Mdl_resource_kind kind,
    Texture_dimension dimension,
    const std::string& resource_name)
{
    std::vector<Mdl_resource_assignment>& vec = m_resources[kind];
    size_t runtime_id = vec.empty() ? 1 : vec.back().runtime_resource_id + 1;

    Mdl_resource_assignment set(kind);
    set.resource_name = resource_name;
    set.dimension = dimension;
    set.runtime_resource_id = runtime_id;
    vec.emplace_back(set);
    return runtime_id;
}

// ------------------------------------------------------------------------------------------------

const std::vector<Mdl_resource_assignment>& Mdl_material::get_resources(
    Mdl_resource_kind kind) const
{
    const auto& found = m_resources.find(kind);
    return found->second;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material::on_target_generated(D3DCommandList* command_list)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        m_sdk->get_transaction().access<const mi::neuraylib::IMaterial_definition>(
            m_description.get_material_defintion_db_name()));

    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        m_sdk->get_transaction().access<const mi::neuraylib::ICompiled_material>(
        m_compiled_db_name.c_str()));

    // resources are still valid for this material
    if (m_compiled_hash == m_resource_hash)
        return true;

    auto p = m_app->get_profiling().measure("loading material instance data");

    const mi::base::Uuid hash = compiled_material->get_hash();
    m_resource_hash = std::to_string(hash.m_id1);
    m_resource_hash += std::to_string(hash.m_id2);
    m_resource_hash += std::to_string(hash.m_id3);
    m_resource_hash += std::to_string(hash.m_id4);

    // copy resource list from target an re-fill it
    for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
    {
        Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
        const std::vector<Mdl_resource_assignment>& to_copy = m_target->get_resources(kind_i);
        m_resources[kind_i].clear();
        m_resources[kind_i].insert(m_resources[kind_i].end(), to_copy.begin(), to_copy.end());
    }

    // get target code information for this specific material
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(m_target->get_target_code());
    uint32_t zero(0);

    // if there is already an argument block (material has parameter and class compilation)
    if (m_argument_layout_index != static_cast<mi::Size>(-1) &&
        m_argument_layout_index < target_code->get_argument_layout_count())
    {
        // argument block for class compilation parameter data
        mi::base::Handle<const mi::neuraylib::ITarget_argument_block> arg_block;
        {
            auto p = m_app->get_profiling().measure(
                "loading material instance data: argument block");

            // get the layout
            mi::base::Handle<const mi::neuraylib::ITarget_value_layout> arg_layout(
                target_code->get_argument_block_layout(m_argument_layout_index));

            // for the first instances of the materials, the argument block already exists
            // for further blocks new ones have to be created. To avoid special treatment,
            // an new block is created for every material
            mi::base::Handle<mi::neuraylib::ITarget_resource_callback> callback(
                m_target->create_resource_callback(this));

            arg_block = target_code->create_argument_block(
                m_argument_layout_index,
                compiled_material.get(),
                callback.get());

            if (!arg_block)
            {
                log_error("Failed to create material argument block: " + get_name(), SRC);
                return false;
            }}

        // create a buffer to provide those parameters to the shader
        size_t buffer_size = round_to_power_of_two(arg_block->get_size(), 4);
        m_argument_block_data = std::vector<uint8_t>(buffer_size, 0);
        memcpy(m_argument_block_data.data(), arg_block->get_data(), arg_block->get_size());

        auto argument_block_buffer = new Buffer(
            m_app, buffer_size, get_name() + "_ArgumentBlock");

        argument_block_buffer->set_data(m_argument_block_data);

        if (m_argument_block_buffer != nullptr) delete m_argument_block_buffer;
        m_argument_block_buffer = argument_block_buffer;
    }

    // if there is no data, a dummy buffer that contains a null pointer is created
    if (!m_argument_block_buffer)
    {
        m_argument_block_buffer = new Buffer(m_app, 4, get_name() + "_ArgumentBlock_nullptr");
        m_argument_block_buffer->set_data(&zero, 1);
    }


    // load per material textures
    // load the texture in parallel, if not forced otherwise
    // skip the invalid texture that is always present

    Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    std::vector<std::thread> tasks;

    // textures
    for (size_t t = 0; t < m_resources[Mdl_resource_kind::Texture].size(); ++t)
    {
        Mdl_resource_assignment* assignment = &m_resources[Mdl_resource_kind::Texture][t];
        if (assignment->data != nullptr)
            continue; // data already loaded, e.g. by the target

        // load sequentially
        if (m_app->get_options()->force_single_threading)
        {
            assignment->data = m_sdk->get_library()->access_texture_resource(
                assignment->resource_name, assignment->dimension, command_list);
        }
        // load asynchronously
        else
        {
            tasks.emplace_back(std::thread([&, assignment]()
            {
                // do not fill command lists from different threads
                D3DCommandList* local_command_list = command_queue->get_command_list();

                assignment->data = m_sdk->get_library()->access_texture_resource(
                    assignment->resource_name, assignment->dimension, local_command_list);

                command_queue->execute_command_list(local_command_list);
            }));
        }
    }

    // wait for all loading tasks
    for (auto &t : tasks)
        t.join();

    // check the actual texture2D resource count (UDIMs)
    size_t tex_count = 0;
    for (auto& set : m_resources[Mdl_resource_kind::Texture])
        tex_count += set.data ? set.data->get_tile_count() : 0;
        // check for null textures in UDIM tiles and add a default pink one

    // reserve/create resource views
    Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();

    size_t handle_count = 1;    // for constant buffer
    handle_count++;             // for argument block
    handle_count++;             // for resource infos
    handle_count += tex_count;  // textures 2D and 3D
    // light profiles, ...

    // if we already have a block on the resource heap (previous generation)
    // we try to reuse is if it fits
    if (m_first_resource_heap_handle.is_valid())
    {
        if (resource_heap.get_block_size(m_first_resource_heap_handle) < handle_count)
            resource_heap.free_views(m_first_resource_heap_handle); // free heap block
    }

    // reserve a new block of the required size and check if that was successful
    if (!m_first_resource_heap_handle.is_valid())
    {
        m_first_resource_heap_handle = resource_heap.reserve_views(handle_count);
        if (!m_first_resource_heap_handle.is_valid())
            return false;
    }

    // create the actual constant buffer view
    if (!resource_heap.create_constant_buffer_view(&m_constants, m_first_resource_heap_handle))
        return false;

    // create a resource view for the argument block
    auto argument_block_data_srv = m_first_resource_heap_handle.create_offset(1);
    if (!argument_block_data_srv.is_valid())
        return false;

    if (!resource_heap.create_shader_resource_view(
        m_argument_block_buffer, true, argument_block_data_srv))
            return false;

    // create the resource info buffer and view to handle resource mapping
    std::vector<Mdl_resource_info> info_data(
        m_resources[Mdl_resource_kind::Texture].size(), {0, 0, 0});

    if (m_resource_infos &&
        m_resource_infos->get_element_count() < std::max(info_data.size(), size_t(1)))
    {
        delete m_resource_infos;
        m_resource_infos = nullptr;
    }
    if (!m_resource_infos)
    {
        m_resource_infos = new Structured_buffer<Mdl_resource_info>(
            m_app, std::max(info_data.size(), size_t(1)), get_name() + "_ResourceInfos");
    }

    auto resource_info_srv = m_first_resource_heap_handle.create_offset(2);
    if (!resource_info_srv.is_valid())
        return false;

    if (!resource_heap.create_shader_resource_view(m_resource_infos, resource_info_srv))
        return false;

    // create shader resource views for resources for textures
    size_t descriptor_heap_offset = 3;
    size_t gpu_resource_array_offset = 0;
    for (size_t t = 0; t < m_resources[Mdl_resource_kind::Texture].size(); ++t)
    {
        Mdl_resource_assignment& assignment = m_resources[Mdl_resource_kind::Texture][t];

        // dense uv-tile-map, for large and sparse maps, this approach is too simple
        std::vector<Resource*> tile_map(
            assignment.data ? assignment.data->get_tile_count() : 0, nullptr);

        if (assignment.data && !assignment.data->is_udim_tiled)
            tile_map[0] = assignment.data->entries[0].resource;
        else if (assignment.data)
        {
            for (size_t e = 0, n = assignment.data->entries.size(); e < n; ++e)
            {
                size_t index = assignment.data->compute_linear_udim_index(e);
                tile_map[index] = assignment.data->entries[e].resource;
            }
        }

        // create resource views for each tile
        for (auto& tile : tile_map)
        {
            Descriptor_heap_handle heap_handle =
                m_first_resource_heap_handle.create_offset(descriptor_heap_offset++);
            if (!heap_handle.is_valid())
                return false;

            // create 2D or 3D view
            if (!resource_heap.create_shader_resource_view(
                static_cast<Texture*>(tile),
                assignment.dimension,
                heap_handle))
                    return false;
        }

        // mapping from runtime texture id to index into the array of views
        Mdl_resource_info& info = info_data[assignment.runtime_resource_id - 1];
        info.gpu_resource_array_start = gpu_resource_array_offset;
        info.gpu_resource_array_size = tile_map.size();
        info.gpu_resource_udim_u_min = assignment.data ? assignment.data->udim_u_min : 0;
        info.gpu_resource_udim_v_min = assignment.data ? assignment.data->udim_v_min : 0;
        info.gpu_resource_udim_width = (assignment.data && assignment.data->is_udim_tiled)
            ? assignment.data->get_udim_width() : 0;
        gpu_resource_array_offset += tile_map.size();
    }

    // copy the infos into the buffer
    m_resource_infos->set_data(info_data.data(), info_data.size());

    // get all names in the compiled material
    std::unordered_set<std::string> present_names;
    for (mi::Size i = 0, n = compiled_material->get_referenced_scene_data_count(); i < n; ++i)
        present_names.insert(compiled_material->get_referenced_scene_data_name(i));

    // check which of them is still available after code generation
    // and update the scene data name map
    m_scene_data_name_map.clear();
    for (mi::Size i = 1, n = target_code->get_string_constant_count(); i < n; ++i)
    {
        const char* name = target_code->get_string_constant(i);
        if (present_names.find(name) != present_names.end())
            m_scene_data_name_map[name] = static_cast<uint32_t>(i);
    }

    // add TEXCOORD_0 to demonstrate renderer driven scene data elements
    // NOTE, if this is added manually, MDL code will not create any runtime function call
    // that with the 'scene_data_id'. Instead, only the render can call this outside of the
    // generated code.
    if (m_scene_data_name_map.find("TEXCOORD_0") == m_scene_data_name_map.end())
        m_scene_data_name_map["TEXCOORD_0"] =
            std::max(static_cast<uint32_t>(target_code->get_string_constant_count()), 1u);

    // optimization potential
    m_constants.data.material_flags = static_cast<uint32_t>(m_flags);

    m_constants.upload();
    if (!m_argument_block_buffer->upload(command_list)) return false;
    if (!m_resource_infos->upload(command_list)) return false;

    return true;
}

}}} // mi::examples::mdl_d3d12
