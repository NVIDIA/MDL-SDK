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

#include "mdl_material.h"
#include "base_application.h"
#include "command_queue.h"
#include "descriptor_heap.h"
#include "mdl_material_description.h"
#include "mdl_material_library.h"
#include "mdl_sdk.h"
#include "texture.h"
#include "light_profile.h"
#include "bsdf_measurement.h"
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
    , m_compiled_hash("")
    , m_resource_hash("")
    , m_target(nullptr)
    , m_constants(m_app, m_name + "_Constants")
    , m_argument_block_buffer(nullptr) // size is not known at this point
    , m_argument_block_data()
    , m_argument_layout_index(static_cast<mi::Size>(-1))
    , m_first_resource_heap_handle()
    , m_resources()
    , m_texture_infos(nullptr)
    , m_light_profile_infos(nullptr)
    , m_mbsdf_infos(nullptr)
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
    // unregister
    m_sdk->get_library()->destroy_material(this);

    // delete local resources
    if (m_argument_block_buffer) delete m_argument_block_buffer;
    if (m_texture_infos) delete m_texture_infos;
    if (m_light_profile_infos) delete m_light_profile_infos;
    if (m_mbsdf_infos) delete m_mbsdf_infos;

    // free heap block
    m_app->get_resource_descriptor_heap()->free_views(m_first_resource_heap_handle);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material::compile_material(const Mdl_material_description& description)
{
    // copy description
    m_description = description;

    // since this method can be called from multiple threads simultaneously
    // a new context for is created
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        m_sdk->create_context());

    // database names for the instance and the compiled material
    {
        auto p = m_app->get_profiling().measure("creating MDL material instance");
        m_instance_db_name = m_description.build_material_graph(
            *m_sdk,
            mi::examples::io::dirname(m_app->get_scene_path()),
            "mdl_dxr::material_graph_" + std::to_string(m_material_id),
            context.get());
    }
    if (m_instance_db_name.empty())
        return false;

    m_compiled_db_name = "mdl_dxr::compiled_material_" + std::to_string(m_material_id);

    // get the scene for this material
    set_name(m_description.get_scene_name());

    // the instance is setup and in the DB, now it is the same as recompiling
    return recompile_material(context.get());
}

// ------------------------------------------------------------------------------------------------

namespace // anonymous
{
    // Utility function to dump the hash, arguments, temporaries, and fields of a compiled material.
    std::string dump_compiled_material(mdl_d3d12::Mdl_sdk* sdk, const mi::neuraylib::ICompiled_material* cm)
    {
        std::stringstream s;
        mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
            sdk->get_factory().create_expression_factory(sdk->get_transaction().get()));
        mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
            expression_factory->get_value_factory());

        mi::base::Uuid hash = cm->get_hash();
        char buffer[36];
        snprintf(buffer, sizeof(buffer),
            "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
        s << "    hash overall = " << buffer << std::endl;

        for (mi::Uint32 i = mi::neuraylib::SLOT_FIRST; i <= mi::neuraylib::SLOT_LAST; ++i) {
            hash = cm->get_slot_hash(mi::neuraylib::Material_slot(i));
            snprintf(buffer, sizeof(buffer),
                "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
            s << "    hash slot " << std::setw(2) << i << " = " << buffer << std::endl;
        }

        mi::Size parameter_count = cm->get_parameter_count();
        for (mi::Size i = 0; i < parameter_count; ++i) {
            mi::base::Handle<const mi::neuraylib::IValue> argument(cm->get_argument(i));
            std::stringstream name;
            name << i;
            mi::base::Handle<const mi::IString> result(
                value_factory->dump(argument.get(), name.str().c_str(), 1));
            s << "    argument " << result->get_c_str() << std::endl;
        }

        mi::Size temporary_count = cm->get_temporary_count();
        for (mi::Size i = 0; i < temporary_count; ++i) {
            mi::base::Handle<const mi::neuraylib::IExpression> temporary(cm->get_temporary(i));
            std::stringstream name;
            name << i;
            mi::base::Handle<const mi::IString> result(
                expression_factory->dump(temporary.get(), name.str().c_str(), 1));
            s << "    temporary " << result->get_c_str() << std::endl;
        }

        mi::base::Handle<const mi::neuraylib::IExpression> body(cm->get_body());
        mi::base::Handle<const mi::IString> result(expression_factory->dump(body.get(), 0, 1));
        s << "    body " << result->get_c_str() << std::endl;

        s << std::endl;
        return s.str();
    }
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material::recompile_material(mi::neuraylib::IMdl_execution_context* context)
{
    mi::base::Handle<const mi::neuraylib::IFunction_call> material_instance(
        m_sdk->get_transaction().access<const mi::neuraylib::IFunction_call>(
            m_instance_db_name.c_str())); // get back after storing

    // is this instance still valid
    if (!material_instance->is_valid(context))
    {
        mi::base::Handle<mi::neuraylib::IFunction_call> material_instance_edit(
            m_sdk->get_transaction().edit<mi::neuraylib::IFunction_call>(
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
            }
            else
            {

                // fall-back, the invalid material
                const IScene_loader::Scene* scene = get_material_desciption().get_scene();
                IScene_loader::Material copy = get_material_desciption().get_scene_material();
                copy.name = Mdl_material_description::Invalid_material_identifier;

                // invalid but keep material parameters
                Mdl_material_description invalid(scene, copy); 
                if (compile_material(invalid))
                {
                    log_warning("Repairing and recreating the material instance failed. "
                        "Using invalid material as fall-back: " + get_name());
                }
                else
                {
                    log_error("[FATAL] Repairing and all fall-backs failed: " + get_name());
                    return false;
                }
            }

            // get the recreated or fallback instance
            material_instance = 
                m_sdk->get_transaction().access<const mi::neuraylib::IFunction_call>(
                    m_instance_db_name.c_str());
        }
        else
        {
            // repairing successful
            material_instance = material_instance_edit;
        }
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
        context->set_option("ignore_noinline", true);

        // in order to change the scene scale setting at runtime we need to preserve the conversions
        // in the generated code and expose the factor in the MDL material state of the shader.
        context->set_option("fold_meters_per_scene_unit", false);

        // We can compiled a material to the selected target type.
        // This will influence the expressions paths available depending on that target type.
        // Without the option the type is not casted.
        mi::base::Handle<mi::neuraylib::IType_factory> tf(
            m_sdk->get_factory().create_type_factory(m_sdk->get_transaction().get()));
        mi::base::Handle<const mi::neuraylib::IType_struct> material_target_type(
            tf->create_struct(m_app->get_options()->material_type.c_str()));
        context->set_option("target_type", material_target_type.get());

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
            material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
        mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
            material_instance2->create_compiled_material(flags, context));
        if (!m_sdk->log_messages("Compiling material failed: " + get_name(), context, SRC) ||
            !compiled_material)
                return false;

        if (m_app->get_options()->gpu_debug)
        {
            // for debugging it might also be valuable to see the compiled material graph
            // that is as basis for HLSL code generation
            std::string dir = mi::examples::io::get_working_directory() + "/compiled_materials";
            mi::examples::io::mkdir(dir);

            const mi::base::Uuid compiled_hash = compiled_material->get_hash();
            std::string compiled_hash_s = mi::examples::strings::format(
                "%08x_%08x_%08x_%08x", compiled_hash.m_id1, compiled_hash.m_id2,
                compiled_hash.m_id3, compiled_hash.m_id4);

            std::string filename = dir + "/" + get_name() + "_" + compiled_hash_s + ".log";

            FILE* file = _wfopen(mi::examples::strings::str_to_wstr(filename).c_str(), L"w");
            if (file)
            {
                std::string dump = dump_compiled_material(m_sdk, compiled_material.get());
                fwrite(dump.c_str(), dump.size(), 1, file);
                fclose(file);
            }
        }

        if (mdl_options.distilling_support_enabled && mdl_options.distill_target != "none")
        {
            mi::Sint32 res = 0;
            mi::base::Handle<mi::neuraylib::ICompiled_material> distilled_material(
                m_app->get_mdl_sdk().get_distiller().distill_material(
                    compiled_material.get(), mdl_options.distill_target.c_str(), nullptr, &res));
            if (res == -2)
            {
                log_error("Distilling target not registered: '" + mdl_options.distill_target);
            }
            else if (distilled_material)
            {
                log_info("Distilled material to '" + mdl_options.distill_target + "': " + get_name());

                const mi::base::Uuid compiled_hash = compiled_material->get_hash();
                const mi::base::Uuid distilled_hash = distilled_material->get_hash();

                std::string compiled_hash_s = mi::examples::strings::format(
                    "%08x_%08x_%08x_%08x", compiled_hash.m_id1, compiled_hash.m_id2,
                    compiled_hash.m_id3, compiled_hash.m_id4);

                std::string distilled_hash_s = mi::examples::strings::format(
                    "%08x_%08x_%08x_%08x", distilled_hash.m_id1, distilled_hash.m_id2,
                    distilled_hash.m_id3, distilled_hash.m_id4);

                log_verbose("Compiled Hash: " + get_name() + ": " + compiled_hash_s);
                log_verbose("Distilled Hash: " + get_name() + ": " + distilled_hash_s);

                if (m_app->get_options()->distill_debug)
                {
                    // store the distiller output in a sub-folder
                    std::string dir = mi::examples::io::get_working_directory() + "/_distilling/";
                    mi::examples::io::mkdir(dir);

                    // dump both, the original and the distilled
                    auto action = [&](std::string target, std::string hash, auto mat)
                        {
                            std::string filename = dir + get_name() +
                                "_" + target + "_" + hash + ".log";
                            FILE* file =
                                _wfopen(mi::examples::strings::str_to_wstr(filename).c_str(), L"w");
                            if (file)
                            {
                                std::string dump = dump_compiled_material(m_sdk, mat.get());
                                fwrite(dump.c_str(), dump.size(), 1, file);
                                fclose(file);
                            }
                        };
                    action("original", compiled_hash_s, compiled_material);
                    action(mdl_options.distill_target, distilled_hash_s, distilled_material);
                }

                compiled_material = distilled_material;
            }
            else
                log_error("Distilling material failed, continue with original: " + get_name());
        }

        m_sdk->get_transaction().store(compiled_material.get(), m_compiled_db_name.c_str());
    }

    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        m_sdk->get_transaction().access<const mi::neuraylib::ICompiled_material>(
        m_compiled_db_name.c_str())); // get back after storing

    // generate the hash compiled material hash
    // this is used to check if a new target is required or if an existing one can be reused.
    const std::string old_hash = m_compiled_hash;
    const mi::base::Uuid hash = compiled_material->get_hash();
    m_compiled_hash = mi::examples::strings::format(
        "%08x_%08x_%08x_%08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);

    if (old_hash.empty())
        log_verbose("Hash: " + get_name() + ": " + m_compiled_hash);
    else
    {
        log_verbose("Old Hash: " + get_name() + ": " + old_hash);
        log_verbose("New Hash: " + get_name() + ": " + m_compiled_hash);
    }

    return true;
}

// ------------------------------------------------------------------------------------------------

void Mdl_material::set_target_interface(
    Mdl_material_target* target,
    const Mdl_material_target_interface& target_data)
{
    m_target = target;
    m_argument_layout_index = target_data.argument_layout_index;
    m_material_target_interface = target_data;
}

// ------------------------------------------------------------------------------------------------

void Mdl_material::reset_target_interface()
{
    m_target = nullptr;
}

// ------------------------------------------------------------------------------------------------

std::vector<D3D12_STATIC_SAMPLER_DESC> Mdl_material::get_sampler_descriptions()
{
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;

    // for standard textures
    D3D12_STATIC_SAMPLER_DESC  desc = {};
    desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
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

    // for light profiles
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    desc.ShaderRegister = 1;                        // bind sampler to shader register(s1)
    samplers.push_back(desc);

    // for bsdf measurements
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.ShaderRegister = 2;                        // bind sampler to shader register(s2)
    samplers.push_back(desc);

    // for latitude-longitude environment maps
    desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.ShaderRegister = 3;                        // bind sampler to shader register(s3)
    samplers.push_back(desc);

    return samplers;
}

// ------------------------------------------------------------------------------------------------

uint32_t Mdl_material::register_scene_data_name(const std::string& name)
{
    // scene data names are actually just string constants
    return map_string_constant(name);
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

    m_constants.data.flags = static_cast<uint32_t>(m_description.get_flags());
}

// ------------------------------------------------------------------------------------------------

// Called by the target to register per material resources.
size_t Mdl_material::register_resource(
    Mdl_resource_kind kind,
    Texture_dimension dimension,
    const std::string& resource_name)
{
    std::vector<Mdl_resource_assignment>& vec = m_resources[kind];

    // check if that resource is know already
    for (auto& a : vec)
        if (a.resource_name == resource_name)
            return a.runtime_resource_id;

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

uint32_t Mdl_material::map_string_constant(const std::string& string_value)
{
    // the empty string is also the invalid string
    if (string_value == "")
        return 0;

    // if the constant is already mapped, use it
    for (auto& c : m_string_constants)
    if(c.value == string_value)
        return c.runtime_string_id;

    // map the new constant. keep this mapping dense in order to easy the data layout on the GPU
    uint32_t runtime_id =
        m_string_constants.empty() ? 1 : m_string_constants.back().runtime_string_id + 1;
    Mdl_string_constant entry;
    entry.runtime_string_id = runtime_id;
    entry.value = string_value;
    m_string_constants.push_back(entry);

    return runtime_id;
}

// ------------------------------------------------------------------------------------------------

const std::vector<Mdl_string_constant>& Mdl_material::get_string_constants() const
{
    return m_string_constants;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material::on_target_generated(D3DCommandList* command_list)
{
    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        m_sdk->get_transaction().access<const mi::neuraylib::ICompiled_material>(
        m_compiled_db_name.c_str()));

    // resources are still valid for this material
    if (m_compiled_hash == m_resource_hash)
        return true;

    auto p = m_app->get_profiling().measure("loading material instance data");

    const mi::base::Uuid hash = compiled_material->get_hash();
    m_resource_hash = mi::examples::strings::format(
        "%08x_%08x_%08x_%08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);


    // copy resource list from target an re-fill it
    for (size_t i = 0, n = static_cast<size_t>(Mdl_resource_kind::_Count); i < n; ++i)
    {
        Mdl_resource_kind kind_i = static_cast<Mdl_resource_kind>(i);
        const std::vector<Mdl_resource_assignment>& to_copy = m_target->get_resources(kind_i);
        m_resources[kind_i].clear();
        m_resources[kind_i].insert(m_resources[kind_i].end(), to_copy.begin(), to_copy.end());
    }

    // copy the string constant list
    m_string_constants.clear();
    auto to_copy = m_target->get_string_constants();
    m_string_constants.insert(m_string_constants.end(), to_copy.begin(), to_copy.end());

    // get target code information for this specific material
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(m_target->get_target_code());
    uint32_t zero(0);

    // free old buffer
    if (m_argument_block_buffer)
    {
        delete m_argument_block_buffer;
        m_argument_block_buffer = nullptr;
    }

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
            }
        }

        // create a buffer to provide those parameters to the shader
        size_t buffer_size = round_to_power_of_two(arg_block->get_size(), 4);
        m_argument_block_data = std::vector<uint8_t>(buffer_size, 0);
        memcpy(m_argument_block_data.data(), arg_block->get_data(), arg_block->get_size());

        m_argument_block_buffer = new Buffer(
            m_app, buffer_size, get_name() + "_ArgumentBlock");
        m_argument_block_buffer->set_data(m_argument_block_data);
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

    for (Mdl_resource_kind resource_kind : {
            Mdl_resource_kind::Texture,
            Mdl_resource_kind::Light_profile,
            Mdl_resource_kind::Bsdf_measurement
        })
    {
        for (size_t t = 0; t < m_resources[resource_kind].size(); ++t)
        {
            Mdl_resource_assignment* assignment = &m_resources[resource_kind][t];
            if (assignment->has_data())
                continue; // data already loaded, e.g. by the target

            // load sequentially
            if (m_app->get_options()->force_single_threading)
            {
                switch (resource_kind)
                {
                case Mdl_resource_kind::Texture:
                    assignment->texture_data = m_sdk->get_library()->access_texture_resource(
                        assignment->resource_name, assignment->dimension, command_list);
                    break;

                case Mdl_resource_kind::Light_profile:
                    assignment->light_profile_data = m_sdk->get_library()->access_light_profile_resource(
                        assignment->resource_name, command_list);
                    break;

                case Mdl_resource_kind::Bsdf_measurement:
                    assignment->mbsdf_data = m_sdk->get_library()->access_bsdf_measurement_resource(
                        assignment->resource_name, command_list);
                    break;
                }
            }
            // load asynchronously
            else
            {
                tasks.emplace_back(std::thread([&, assignment, resource_kind]()
                {
                    // do not fill command lists from different threads
                    D3DCommandList* local_command_list = command_queue->get_command_list();

                    switch (resource_kind)
                    {
                    case Mdl_resource_kind::Texture:
                        assignment->texture_data = m_sdk->get_library()->access_texture_resource(
                            assignment->resource_name, assignment->dimension, local_command_list);
                        break;

                    case Mdl_resource_kind::Light_profile:
                        assignment->light_profile_data = m_sdk->get_library()->access_light_profile_resource(
                            assignment->resource_name, local_command_list);
                        break;

                    case Mdl_resource_kind::Bsdf_measurement:
                        assignment->mbsdf_data = m_sdk->get_library()->access_bsdf_measurement_resource(
                            assignment->resource_name, local_command_list);
                        break;
                    }

                    command_queue->execute_command_list(local_command_list);
                }));
            }
        }
    }

    // wait for all loading tasks
    for (auto &t : tasks)
        t.join();

    // check the actual texture2D resource count (uv-tiles)
    size_t tex_count = 0;
    for (auto& set : m_resources[Mdl_resource_kind::Texture])
        tex_count += set.has_data() ? set.texture_data->get_uvtile_count() : 0;
        // check for null textures in uv-tiles

    size_t light_profile_count = m_resources[Mdl_resource_kind::Light_profile].size();
    size_t mbsdf_count = m_resources[Mdl_resource_kind::Bsdf_measurement].size();

    // reserve/create resource views
    Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();

    size_t handle_count = 1;                 // for constant buffer
    handle_count++;                          // for argument block
    handle_count++;                          // for texture infos
    handle_count++;                          // for light profile infos
    handle_count++;                          // for mbsdf infos
    handle_count += tex_count;               // textures 2D and 3D
    handle_count += light_profile_count * 2; // light profiles (sample buffer, data texture)
    handle_count += mbsdf_count * 6;         // MBSDFs (sample buffer, albedo buffer, data texture)

    // if we already have a block on the resource heap (previous generation)
    // we try to reuse is if it fits
    if (m_first_resource_heap_handle.is_valid())
    {
        if (resource_heap.get_block_size(m_first_resource_heap_handle) != handle_count)
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
    if (!resource_heap.create_shader_resource_view(
        m_argument_block_buffer, true, m_first_resource_heap_handle.create_offset(1)))
            return false;

    // create the texture info buffer and view to handle resource mapping
    std::vector<Mdl_texture_info> texture_info_data(
        m_resources[Mdl_resource_kind::Texture].size(), {0, 0, 0});

    if (m_texture_infos)
    {
        // resize buffer by recreating it
        delete m_texture_infos;
        m_texture_infos = nullptr;
    }
    if (!m_texture_infos)
    {
        m_texture_infos = new Structured_buffer<Mdl_texture_info>(
            m_app, std::max(texture_info_data.size(), size_t(1)), get_name() + "_TextureInfos");
    }
    if (!resource_heap.create_shader_resource_view(m_texture_infos, m_first_resource_heap_handle.create_offset(2)))
        return false;

    // create the light profile info buffer and view to handle resource mapping
    std::vector<Mdl_light_profile_info> light_profile_info_data(
        m_resources[Mdl_resource_kind::Light_profile].size());

    if (m_light_profile_infos)
    {
        // resize buffer by recreating it
        delete m_light_profile_infos;
        m_light_profile_infos = nullptr;
    }
    if (!m_light_profile_infos)
    {
        m_light_profile_infos = new Structured_buffer<Mdl_light_profile_info>(
            m_app, std::max(light_profile_info_data.size(), size_t(1)), get_name() + "_LightProfileInfos");
    }
    if (!resource_heap.create_shader_resource_view(m_light_profile_infos, m_first_resource_heap_handle.create_offset(3)))
        return false;

    // create the mbsdf info buffer and view to handle resource mapping
    std::vector<Mdl_mbsdf_info> mbsdf_info_data(
        m_resources[Mdl_resource_kind::Bsdf_measurement].size());
    if (m_mbsdf_infos)
    {
        // resize buffer by recreating it
        delete m_mbsdf_infos;
        m_mbsdf_infos = nullptr;
    }
    if (!m_mbsdf_infos)
    {
        m_mbsdf_infos = new Structured_buffer<Mdl_mbsdf_info>(
            m_app, std::max(mbsdf_info_data.size(), size_t(1)), get_name() + "_MbsdfInfos");
    }
    if (!resource_heap.create_shader_resource_view(m_mbsdf_infos, m_first_resource_heap_handle.create_offset(4)))
        return false;

    // create shader resource views for textures
    size_t descriptor_heap_offset = 5;
    for (size_t t = 0; t < m_resources[Mdl_resource_kind::Texture].size(); ++t)
    {
        Mdl_resource_assignment& assignment = m_resources[Mdl_resource_kind::Texture][t];

        // dense uv-tile-map, for large and sparse maps, this approach is too simple
        std::vector<Resource*> tile_map(
            assignment.texture_data ? assignment.texture_data->get_uvtile_count() : 0, nullptr);

        if (assignment.texture_data)
        {
            for (size_t e = 0, n = assignment.texture_data->entries.size(); e < n; ++e)
            {
                size_t index = assignment.texture_data->compute_linear_uvtile_index(e);
                tile_map[index] = assignment.texture_data->entries[e].resource;
            }
        }

        // mapping from runtime texture id to index into the array of views
        Mdl_texture_info& info = texture_info_data[assignment.runtime_resource_id - 1];
        info.gpu_resource_array_start = 
            m_first_resource_heap_handle.create_offset(descriptor_heap_offset).get_heap_index();
        info.gpu_resource_array_size = tile_map.size();
        info.gpu_resource_frame_first = assignment.has_data() ? assignment.texture_data->frame_first : 0;
        info.gpu_resource_uvtile_u_min = assignment.has_data() ? assignment.texture_data->uvtile_u_min : 0;
        info.gpu_resource_uvtile_v_min = assignment.has_data() ? assignment.texture_data->uvtile_v_min : 0;
        info.gpu_resource_uvtile_width = assignment.has_data() ? assignment.texture_data->get_uvtile_width() : 1;
        info.gpu_resource_uvtile_height =
            assignment.has_data() ? assignment.texture_data->get_uvtile_height() : 1;

        // create resource views for each tile
        for (auto& tile : tile_map)
        {
            // create 2D or 3D view
            if (!resource_heap.create_shader_resource_view(
                static_cast<Texture*>(tile),
                assignment.dimension,
                m_first_resource_heap_handle.create_offset(descriptor_heap_offset++)))
                    return false;
        }
    }

    // create texture and buffer shader resource views and gpu infos for light profiles
    for (size_t i = 0; i < m_resources[Mdl_resource_kind::Light_profile].size(); ++i)
    {
        Mdl_resource_assignment& assignment = m_resources[Mdl_resource_kind::Light_profile][i];

        Descriptor_heap_handle eval_tex_heap_handle =
            m_first_resource_heap_handle.create_offset(descriptor_heap_offset++);
        if (!eval_tex_heap_handle.is_valid())
            return false;

        if (!resource_heap.create_shader_resource_view(
            assignment.light_profile_data->get_evaluation_data(),
            Texture_dimension::Texture_2D,
            eval_tex_heap_handle))
                return false;

        Descriptor_heap_handle sample_buffer_heap_handle =
            m_first_resource_heap_handle.create_offset(descriptor_heap_offset++);
        if (!sample_buffer_heap_handle.is_valid())
            return false;

        if (!resource_heap.create_shader_resource_view(
            assignment.light_profile_data->get_sample_data(),
            sample_buffer_heap_handle))
                return false;

        Mdl_light_profile_info& light_profile_info = light_profile_info_data[i];
        light_profile_info.eval_data_index = eval_tex_heap_handle.get_heap_index();
        light_profile_info.sample_data_index = sample_buffer_heap_handle.get_heap_index();

        light_profile_info.angular_resolution = assignment.light_profile_data->get_angular_resolution();
        light_profile_info.inv_angular_resolution = {
            1.0f / float(light_profile_info.angular_resolution.x),
            1.0f / float(light_profile_info.angular_resolution.y)
        };
        light_profile_info.theta_phi_start = assignment.light_profile_data->get_theta_phi_start();
        light_profile_info.theta_phi_delta = assignment.light_profile_data->get_theta_phi_delta();
        light_profile_info.theta_phi_inv_delta = {
            1.0f / float(light_profile_info.theta_phi_delta.x),
            1.0f / float(light_profile_info.theta_phi_delta.y)
        };
        light_profile_info.candela_multiplier = assignment.light_profile_data->get_candela_multiplier();
        light_profile_info.total_power = assignment.light_profile_data->get_total_power();
    }
    
    // create texture and buffer shader resource views and gpu infos for MBSDFs
    for (size_t i = 0; i < m_resources[Mdl_resource_kind::Bsdf_measurement].size(); ++i)
    {
        Mdl_resource_assignment& assignment = m_resources[Mdl_resource_kind::Bsdf_measurement][i];

        Mdl_mbsdf_info& mbsdf_info = mbsdf_info_data[i];
        for (mi::neuraylib::Mbsdf_part part : { mi::neuraylib::MBSDF_DATA_REFLECTION, mi::neuraylib::MBSDF_DATA_TRANSMISSION })
        {
            mbsdf_info.has_data[part] = assignment.mbsdf_data->has_part(part) ? 1 : 0;
            if (mbsdf_info.has_data[part] == 0)
                continue;

            const Bsdf_measurement::Part& mbsdf_part = assignment.mbsdf_data->get_part(part);

            Descriptor_heap_handle eval_tex_heap_handle =
                m_first_resource_heap_handle.create_offset(descriptor_heap_offset++);
            if (!eval_tex_heap_handle.is_valid())
                return false;

            if (!resource_heap.create_shader_resource_view(
                mbsdf_part.evaluation_data,
                Texture_dimension::Texture_3D,
                eval_tex_heap_handle))
                    return false;

            Descriptor_heap_handle sample_buffer_heap_handle =
                m_first_resource_heap_handle.create_offset(descriptor_heap_offset++);
            if (!sample_buffer_heap_handle.is_valid())
                return false;

            if (!resource_heap.create_shader_resource_view(
                mbsdf_part.sample_data,
                sample_buffer_heap_handle))
                    return false;

            Descriptor_heap_handle albedo_buffer_heap_handle =
                m_first_resource_heap_handle.create_offset(descriptor_heap_offset++);
            if (!albedo_buffer_heap_handle.is_valid())
                return false;

            if (!resource_heap.create_shader_resource_view(
                mbsdf_part.albedo_data,
                albedo_buffer_heap_handle))
                    return false;

            mbsdf_info.eval_data_index[part] = eval_tex_heap_handle.get_heap_index();
            mbsdf_info.sample_data_index[part] = sample_buffer_heap_handle.get_heap_index();
            mbsdf_info.albedo_data_index[part] = albedo_buffer_heap_handle.get_heap_index();
            mbsdf_info.max_albedo[part] = mbsdf_part.max_albedo;
            mbsdf_info.angular_resolution_theta[part] = mbsdf_part.angular_resolution_theta;
            mbsdf_info.angular_resolution_phi[part] = mbsdf_part.angular_resolution_phi;
            mbsdf_info.num_channels[part] = mbsdf_part.num_channels;
        }
    }

    // copy the infos into the buffer
    m_texture_infos->set_data(texture_info_data.data(), texture_info_data.size());
    m_light_profile_infos->set_data(light_profile_info_data.data(), light_profile_info_data.size());
    m_mbsdf_infos->set_data(mbsdf_info_data.data(), mbsdf_info_data.size());

    // material code features
    m_constants.data.features = m_material_target_interface.material_code_paths;

    // optimization potential
    m_constants.data.flags = static_cast<uint32_t>(m_description.get_flags());

    m_constants.upload();
    if (!m_argument_block_buffer->upload(command_list)) return false;
    if (!m_texture_infos->upload(command_list)) return false;
    if (!m_light_profile_infos->upload(command_list)) return false;
    if (!m_mbsdf_infos->upload(command_list)) return false;

    return true;
}

}}} // mi::examples::mdl_d3d12
