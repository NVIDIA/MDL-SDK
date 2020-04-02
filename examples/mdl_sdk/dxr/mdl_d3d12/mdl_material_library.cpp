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

#include "mdl_material_library.h"

#include "base_application.h"
#include "mdl_material.h"
#include "mdl_material_description.h"
#include "mdl_material_target.h"
#include "mdl_sdk.h"
#include "texture.h"

namespace mdl_d3d12
{
    
    Mdl_material_library::Target_entry::Target_entry(Mdl_material_target* target)
        : m_target(target)
        , m_mutex()
    {
    }

    // --------------------------------------------------------------------------------------------

    Mdl_material_library::Target_entry::~Target_entry()
    {
        delete m_target;
    }

    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    Mdl_material_library::Mdl_material_library(
        Base_application* app, Mdl_sdk* sdk)
        : m_app(app)
        , m_sdk(sdk)
        , m_targets()
        , m_target_map()
        , m_targets_mtx()
        , m_resources()
        , m_resources_mtx()
    {
    }

    // --------------------------------------------------------------------------------------------

    Mdl_material_library::~Mdl_material_library()
    {
        m_target_map.clear();
        for (auto& entry : m_targets)
            delete entry.second;

        for (auto& resource : m_resources)
            for (auto& entry : resource.second.entries)
                delete entry.resource;
    }

    // --------------------------------------------------------------------------------------------

    Mdl_material_library::Target_entry* Mdl_material_library::get_target_for_material_creation(
        const std::string& key)
    {
        std::lock_guard<std::mutex> lock(m_targets_mtx);
        Mdl_material_library::Target_entry* entry;

        // reuse the existing target if the material is already added to a link unit
        auto found = m_target_map.find(key);
        if (found == m_target_map.end())
        {
            entry = new Target_entry(new Mdl_material_target(m_app, m_sdk));

            // store a key based on the compiled material hash to identify already handled ones
            m_target_map.emplace(key, entry);
            m_targets[entry->m_target->get_id()] = entry->m_target;
        }
        else
            entry = found->second;

        return entry;
    }

    // --------------------------------------------------------------------------------------------

    Mdl_material* Mdl_material_library::create(const Mdl_material_description& material_desc)
    {
        Mdl_material* mat = new Mdl_material(m_app, material_desc.get_scene_name());

        // since this method can be called from multiple threads simultaneously
        // a new context for is created
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

        // check if the module and thereby the material definition is already loaded
        // if not load, load the module
        std::string module_db_name = load_module(
            material_desc.get_qualified_module_name(), false, context.get());
        if (!m_sdk->log_messages(context.get()) || module_db_name.empty())
        {
            delete mat;
            return nullptr;
        }

        // get the loaded material from the database
        std::string material_db_name = module_db_name + "::" + material_desc.get_material_name();

        // create a material instance and compile it (mdl compiler, not HLSL jet)
        mi::base::Handle<const mi::neuraylib::IExpression_list> parameters(
            material_desc.get_parameters());
        if (!mat->compile_material(material_db_name, parameters.get(), context.get()))
        {
            delete mat;
            return nullptr;
        }

        // check if the compiled material is already present and reuse it in that case
        // otherwise create a new target
        auto target_entry = get_target_for_material_creation(mat->get_material_compiled_hash());

        // assign the material to the target
        target_entry->m_target->register_material(mat);
        return mat;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_library::reload_material(Mdl_material* material, bool& targets_changed)
    {
        // iterate over the dependencies and store in reverse order for reloading
        // the 'deepest' imports first
        std::stack<std::string> load_order;
        material->visit_module_dependencies([&](const std::string& module_db_name)
        {
            load_order.push(module_db_name);
            return true; // continue visiting
        }); 

        // reload all the modules this material depends on
        while (!load_order.empty())
        {
            if(!reload_module(load_order.top(), targets_changed))
               return false;
            load_order.pop();
        }
        return true;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_library::reload_module(
        const std::string& module_db_name, 
        bool& targets_changed)
    {
        targets_changed = false;

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

        mi::base::Handle<mi::neuraylib::IModule> module(
            m_sdk->get_transaction().edit<mi::neuraylib::IModule>(module_db_name.c_str()));

        // these modules should not change during runtime
        if (module->is_standard_module() || module_db_name == "mdl::base")
            return true;

        if (!module)
        {
            log_error("Reloading failed: " + module_db_name + " is not known to the DB.", SRC);
            return false;
        }

        // do the actual reload of the module
        module->reload(false, context.get());
        if (!m_sdk->log_messages(context.get()))
            return false;
 
        // update material instances and compiled materials
        std::vector<Mdl_material*> changed_materials;
        std::set<std::string> old_hashes;
        visit_materials([&](Mdl_material* material) // could be done in parallel if compiling
        {                                           // becomes a bottleneck
            // ignore materials that do not depend on the reloaded module
            if (!material->depends(module_db_name))
                return true;

            // recompile the material and check if the hash changed
            const std::string current_hash = material->get_material_compiled_hash();

            if (!material->recompile_material(context.get()))
            {
                log_error("Reloading to recompile: " + material->get_name(), SRC);
                m_sdk->log_messages(context.get());
                return false;
            }

            if (current_hash != material->get_material_compiled_hash())
            {
                changed_materials.push_back(material);
                old_hashes.insert(current_hash);
                targets_changed = true;
            }

            return true; // continue visiting
        });

        // if no material changed, we can stop here
        if (changed_materials.empty())
            return true;

        // update material to target assignment
        for (auto& mat : changed_materials)
        {
            // check if the compiled material is already present and reuse it in that case
            // otherwise create a new target
            auto target_entry = get_target_for_material_creation(mat->get_material_compiled_hash());
            target_entry->m_target->register_material(mat); // no lock, single threaded here
        }

        // purge unused targets
        {
            std::lock_guard<std::mutex> lock(m_targets_mtx);
            for (auto& hash : old_hashes)
            {
                Mdl_material_target* target = m_target_map[hash]->m_target;
                if (target->get_material_count() == 0)
                {
                    m_targets.erase(target->get_id());
                    std::vector<std::string> keys_to_delete;
                    for (auto& pair : m_target_map)
                        if (pair.second->m_target == target)
                            keys_to_delete.push_back(pair.first);
               
                    for (auto& key : keys_to_delete)
                        m_target_map.erase(key);
                    
                    delete target;
                }
            }
        }

        return true;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_library::generate_and_compile_targets()
    {
        // generate and compile in separate loops just to be able to measure individual timings
        std::atomic<bool> success = true;
        std::vector<std::thread> tasks;
        bool force_single_theading = m_app->get_options()->force_single_theading;

        {
            Timing t("generating target code");
            visit_target_codes([&](Mdl_material_target* target)
            {
                if (!target->is_generation_required())
                    return true; // skip target and continue visits

                // sequentially
                if (force_single_theading)
                {
                    if (!target->generate())
                        success.store(false);
                }
                // asynchronously
                else
                {
                    tasks.emplace_back(std::thread([&, target]()
                    {
                        if (!target->generate())
                            success.store(false);
                    }));
                }
                return true; // continue visits
            });

            for (auto &t : tasks)
                t.join();

            if (!success.load())
                return false;
        }


        tasks.clear();
        {
            Timing t("compiling target code");
            visit_target_codes([&](Mdl_material_target* target)
            {
                if (!target->is_compilation_required())
                    return true; // skip target and continue visits

                // sequentially
                if (force_single_theading)
                {
                    if (!target->compile())
                        success.store(false);
                }
                // asynchronously
                else
                {
                    tasks.emplace_back(std::thread([&, target]()
                    {
                        if (!target->compile())
                            success.store(false);
                    }));
                }
                return true; // continue visits
            });

            for (auto &t : tasks)
                t.join();
        }
        // false, when at least one task failed
        return success.load();
    }

    // --------------------------------------------------------------------------------------------

    std::string Mdl_material_library::load_module(
        const std::string& qualified_module_name,
        bool reload,
        mi::neuraylib::IMdl_execution_context* context)
    {
        // check if the module and thereby the material definition is already loaded
        const char* module_db_name = m_sdk->get_transaction().execute<const char*>(
            [&](mi::neuraylib::ITransaction* t)
        {
            return m_sdk->get_compiler().get_module_db_name(
                t, qualified_module_name.c_str(), context);
        });

        // parameter is no a valid qualified module name 
        if (!m_sdk->log_messages(context))
            return "";

        // if not, load it
        if (!module_db_name)
        {
            // Load the module that contains the material.
            // This functions supports multi-threading. It blocks when the requested module
            // or a dependency is loaded by a different thread.           
            m_sdk->get_compiler().load_module(
                m_sdk->get_transaction().get(), qualified_module_name.c_str(), context);

            // loading failed
            if (!m_sdk->log_messages(context))
                return "";

            // get the database name of the loaded module
            module_db_name = m_sdk->get_transaction().execute<const char*>(
                [&](mi::neuraylib::ITransaction* t)
            {
                return m_sdk->get_compiler().get_module_db_name(
                    t, qualified_module_name.c_str(), context);
            });

            // make sure there is no other error, should not happen
            if (!m_sdk->log_messages(context) || !module_db_name)
                return "";
        }

        return module_db_name;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_library::visit_target_codes(std::function<bool(Mdl_material_target*)> action)
    {
        std::lock_guard<std::mutex> lock(m_targets_mtx);
        for (auto it = m_targets.begin(); it != m_targets.end(); it++)
            if (!action(it->second))
                return false;
        return true;
    }

    // --------------------------------------------------------------------------------------------

    bool Mdl_material_library::visit_materials(std::function<bool(Mdl_material*)> action)
    {
        return visit_target_codes([&](Mdl_material_target* tc) {
            return tc->visit_materials(action);
        });
    }

    // --------------------------------------------------------------------------------------------

    Mdl_resource_set* Mdl_material_library::access_texture_resource(
        std::string db_name, 
        Texture_dimension dimension, 
        D3DCommandList* command_list)
    {
        // check if the texture already exists
        {
            std::lock_guard<std::mutex> lock(m_resources_mtx);
            auto found = m_resources.find(db_name);
            if (found != m_resources.end())
                return &found->second;
        }

        // if not, collect all infos required to create the texture
        mi::base::Handle<const mi::neuraylib::ITexture> texture(
            m_sdk->get_transaction().access<mi::neuraylib::ITexture>(db_name.c_str()));

        mi::base::Handle<const mi::neuraylib::IImage> image(
            m_sdk->get_transaction().access<mi::neuraylib::IImage>(texture->get_image()));

        // collect basic information about the number of tiles
        mi::Size num_tiles = image->get_uvtile_length();
        mi::Sint32 u_min = std::numeric_limits<mi::Sint32>::max();
        mi::Sint32 u_max = std::numeric_limits<mi::Sint32>::min();
        mi::Sint32 v_min = u_min;
        mi::Sint32 v_max = u_max;

        // create empty textures for all tiles
        {
            std::lock_guard<std::mutex> lock(m_resources_mtx);
            auto found = m_resources.find(db_name);
            if (found != m_resources.end())
                return &found->second; // return the other

            // add the created texture to the library
            m_resources[db_name] = Mdl_resource_set();
            Mdl_resource_set& set = m_resources[db_name];
            set.is_udim_tiled = image->is_uvtile();

            // create the resources
            set.entries = std::vector<Mdl_resource_set::Entry>(
                num_tiles, Mdl_resource_set::Entry());
            for (mi::Size tile_id = 0; tile_id < num_tiles; ++tile_id)
            {
                mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
                    image->get_canvas(0, mi::Uint32(tile_id)));

                // create the d3d texture
                switch (dimension)
                {
                    case Texture_dimension::Texture_2D:
                        set.entries[tile_id].resource = Texture::create_texture_2d(
                            m_app, GPU_access::shader_resource,
                            canvas->get_resolution_x(),
                            canvas->get_resolution_y(),
                            DXGI_FORMAT_R32G32B32A32_FLOAT, // TODO
                            set.is_udim_tiled 
                                ? (db_name + "_tile_" + std::to_string(tile_id)) : db_name);
                        break;

                    case Texture_dimension::Texture_3D:
                        set.entries[tile_id].resource = Texture::create_texture_3d(
                            m_app, GPU_access::shader_resource,
                            canvas->get_resolution_x(),
                            canvas->get_resolution_y(),
                            canvas->get_layers_size(),
                            DXGI_FORMAT_R32G32B32A32_FLOAT, // TODO
                            set.is_udim_tiled 
                                ? (db_name + "_tile_" + std::to_string(tile_id)) : db_name);
                        break;

                    default:
                        log_error("Unhandled texture dimension: " + db_name, SRC);
                        continue;
                }


                mi::Sint32 u, v;
                image->get_uvtile_uv(mi::Uint32(tile_id), u, v);
                set.entries[tile_id].udim_u = u;
                set.entries[tile_id].udim_v = v;
                u_min = std::min(u_min, u);
                u_max = std::max(u_max, u);
                v_min = std::min(v_min, v);
                v_max = std::max(v_max, v);
            }

            // resource set data
            set.udim_u_min = u_min;
            set.udim_u_max = u_max;
            set.udim_v_min = v_min;
            set.udim_v_max = v_max;

            // release the lock so other threads can reference the texture even when the
            // actual data is not loaded yet. So when loading in parallel, wait if the data
            // is required on the GPU
        }

        // load the actual texture data and copy it to GPU
        bool success = true;
        Mdl_resource_set& set = m_resources[db_name];
        uint8_t* buffer = nullptr;
        size_t buffer_size = 0;
        for (mi::Size tile_id = 0; tile_id < num_tiles; ++tile_id)
        {
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
                image->get_canvas(0, mi::Uint32(tile_id)));

            // For simplicity, the texture access functions are only implemented for float4 and
            // gamma is pre-applied here (all images are converted to linear space).
            // Convert to linear color space if necessary
            float gamma = texture->get_effective_gamma();
            char const* image_type = image->get_type();
            if (gamma != 1.0f)
            {
                // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
                mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
                    m_sdk->get_image_api().convert(canvas.get(), "Color"));
                gamma_canvas->set_gamma(gamma);
                m_sdk->get_image_api().adjust_gamma(gamma_canvas.get(), 1.0f);
                canvas = gamma_canvas;
            }
            else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
            {
                // Convert to expected format
                canvas = m_sdk->get_image_api().convert(canvas.get(), "Color");
            }

            Texture* texture_resource = static_cast<Texture*>(set.entries[tile_id].resource);

            size_t tex_width = texture_resource->get_width();
            size_t tex_height = texture_resource->get_height();
            size_t tex_layers = texture_resource->get_depth();
            size_t data_row_pitch = texture_resource->get_pixel_stride() * tex_width;
            size_t gpu_row_pitch = texture_resource->get_gpu_row_pitch();

            // use a temporary buffer for adding padding 
            size_t new_size = gpu_row_pitch * tex_height * tex_layers;
            if (buffer_size < new_size)
            {
                if (buffer != nullptr) 
                    delete[] buffer;

                buffer_size = new_size;
                buffer = new uint8_t[buffer_size];

                if (gpu_row_pitch != data_row_pitch) // alignment required
                    memset(buffer, 0, buffer_size);  // pad with zeros
            }

            // get and copy data to GPU
            for (size_t l = 0; l < tex_layers; ++l)
            {
                mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0, l));
                const uint8_t* tex_data = static_cast<const uint8_t*>(tile->get_data());

                // copy line by line if required
                uint8_t* buffer_layer = buffer + gpu_row_pitch * tex_height * l;
                if (gpu_row_pitch != data_row_pitch)
                {
                    for (size_t r = 0; r < tex_height; ++r)
                        memcpy((void*) (buffer_layer + r * gpu_row_pitch),
                        (void*) (tex_data + r * data_row_pitch),
                               data_row_pitch);
                }
                else // copy directly
                {
                    memcpy((void*) buffer_layer, (void*) tex_data, data_row_pitch * tex_height);
                }
            }

            // copy data to the GPU
            if (texture_resource->upload(command_list, (const uint8_t*) buffer, gpu_row_pitch))
            {
                // .. since the compute pipeline is used for ray tracing
                texture_resource->transition_to(
                    command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            else
                success = false;

            // free data
            if (!success)
            {
                std::lock_guard<std::mutex> lock(m_resources_mtx);
                for (auto& entry : set.entries)
                    delete entry.resource;

                m_resources.erase(db_name);
                delete[] buffer;
                return nullptr;
            }
        }

        delete[] buffer;
        return &m_resources[db_name];
    }
}
