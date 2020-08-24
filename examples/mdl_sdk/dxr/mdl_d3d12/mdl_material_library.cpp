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

#include "example_shared.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Mdl_material_library::Target_entry::Target_entry(Mdl_material_target* target)
    : m_target(target)
    , m_mutex()
{
}

// ------------------------------------------------------------------------------------------------

Mdl_material_library::Target_entry::~Target_entry()
{
    delete m_target;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------------

Mdl_material_library::~Mdl_material_library()
{
    m_target_map.clear();
    for (auto& entry : m_targets)
        delete entry.second;

    for (auto& resource : m_resources)
        for (auto& entry : resource.second.entries)
            delete entry.resource;
}

// ------------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------------

Mdl_material* Mdl_material_library::create_material()
{
    Mdl_material* mat = new Mdl_material(m_app);
    return mat;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::set_description(
    Mdl_material* material,
    const Mdl_material_description& material_desc)
{
    // create and compile a material instance from the given description
    if (!material->compile_material(material_desc))
    {
        set_invalid_material(material);
        return false;
    }

    // check if the compiled material is already present and reuse it in that case
    // otherwise create a new target
    auto target_entry =
        get_target_for_material_creation(material->get_material_compiled_hash());

    // assign the material to the target
    auto old_target = material->get_target_code();
    target_entry->m_target->register_material(material);
    if (old_target && old_target->get_material_count() == 0)
    {
        std::lock_guard<std::mutex> lock(m_targets_mtx);
        if (old_target->get_material_count() == 0)
        {
            m_targets.erase(old_target->get_id());
            std::vector<std::string> keys_to_delete;
            for (auto& pair : m_target_map)
                if (pair.second->m_target == old_target)
                    keys_to_delete.push_back(pair.first);

            for (auto& key : keys_to_delete)
                m_target_map.erase(key);

            delete old_target;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_library::set_invalid_material(Mdl_material* material)
{
    Mdl_material_description invalid("::dxr::not_available");
    set_description(material, invalid);
}

// ------------------------------------------------------------------------------------------------

namespace
{

// helper to check if a value is in an container
template <typename TValue, typename Alloc, template <typename, typename> class TContainer>
inline bool contains(TContainer<TValue, Alloc>& vector, const TValue& value)
{
    return std::find(vector.begin(), vector.end(), value) != vector.end();
}

// ------------------------------------------------------------------------------------------------

// helper to push back values into a compatible container only if the value is not already
// in that container
template <typename TValue, typename Alloc, template <typename, typename> class TContainer>
inline bool push_back_uniquely(TContainer<TValue, Alloc>& vector, const TValue& value)
{
    if (!contains(vector, value))
    {
        vector.push_back(value);
        return true;
    }
    return false;
}

} // anonymous

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::recompile_material(Mdl_material* material, bool& targets_changed)
{
    targets_changed = false;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

    // recompile the material and check if the hash changed
    const std::string current_hash = material->get_material_compiled_hash();
    if (!material->recompile_material(context.get()))
        return false;

    // the hash did not change
    if (current_hash == material->get_material_compiled_hash())
        return true;

    // get the current (old) target
    Mdl_material_target* old_target = material->get_target_code();
    targets_changed = true;

    // check if the compiled material is already present and reuse it in that case
    // otherwise create a new target
    auto entry = get_target_for_material_creation(material->get_material_compiled_hash());
    entry->m_target->register_material(material); // no lock, single threaded here

    // free the old target if not used anymore
    if (old_target->get_material_count() == 0)
    {
        std::lock_guard<std::mutex> lock(m_targets_mtx);
        m_targets.erase(old_target->get_id());
        m_target_map.erase(current_hash);
        delete old_target;
    }

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::recompile_materials(bool& targets_changed)
{
    std::atomic<bool> success = true;

    targets_changed = false;
    std::atomic<bool> changed = targets_changed;

    std::vector<std::thread> tasks;
    bool force_single_threading = m_app->get_options()->force_single_threading;

    visit_materials([&](Mdl_material* material)
        {
            // sequentially
            if (force_single_threading)
            {
                bool tc_changed;
                if (!recompile_material(material, tc_changed))
                    success.store(false);
                changed.store(tc_changed);
            }
            // asynchronously
            else
            {
                tasks.emplace_back(std::thread([&, material]()
                    {
                        bool tc_changed;
                        if (!recompile_material(material, tc_changed))
                            success.store(false);
                        changed.store(tc_changed);
                    }));
            }
            return true; // continue visits
        });

    for (auto& t : tasks)
        t.join();

    targets_changed = changed.load();
    return success.load();
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::reload_material(
    Mdl_material* material,
    bool& targets_changed)
{
    if (!material->get_material_desciption().supports_reloading())
    {
        log_warning("Reloading is not supported for material: " + material->get_name());
        targets_changed = false;
        return true;
    }

    // description, containing module, material, and parameter information for this material
    auto desc = material->get_material_desciption();

    // update the source code if the module was generated
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());
    std::string scene_directory = mi::examples::io::dirname(m_app->get_scene_path());
    const char* code = desc.regenerate_source_code(*m_sdk, scene_directory, context.get());

    return reload_module(desc.get_module_db_name(), code, targets_changed);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::reload_module(
    const std::string& module_db_name,
    bool& targets_changed)
{
    return reload_module(module_db_name, "", targets_changed);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::reload_module(
    const std::string& module_db_name,
    const char* module_source_code,
    bool& targets_changed)
{
    targets_changed = false;

    // collect all modules to reload
    std::vector<std::string> dependencies;
    std::deque<std::string> to_process;

    // add the module of the selected material to be reloaded
    to_process.push_back(module_db_name);
    {
        std::lock_guard<std::mutex> lock(m_module_dependencies_mtx);

        // It is not enough to reload just the one module, imported modules could have changed
        // too. Each module that actually was reloaded could be imported by other modules as
        // well. These modules also have to be reloaded in order to get a consistent state.
        std::function<void(const std::string&)> add_modules_to_load =
            [&](const std::string& db_name)
        {
            auto found = m_module_dependencies.find(db_name);

            // Iterate over the imports first to make sure they are updated first
            for (auto i_name : found->second.imports)
                add_modules_to_load(i_name);

            // then add the current module
            push_back_uniquely(dependencies, db_name);

            // If this materials module is imported by another material, it has to be updated, too.
            // Add all modules that import the current one to be processed afterwards.

            for (auto i_name : found->second.is_imported_by)
                if(!contains(dependencies, i_name) && !contains(to_process, i_name))
                    to_process.push_back(i_name);
        };

        // process all modules until there are no further dependencies
        while (!to_process.empty())
        {
            add_modules_to_load(to_process.front());
            to_process.pop_front();
        }
    }

    // reload the collected the module
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());
    for (auto& module_name : dependencies)
    {
        mi::base::Handle<mi::neuraylib::IModule> module(
            m_sdk->get_transaction().edit<mi::neuraylib::IModule>(module_name.c_str()));

        // invalid database name
        if (!module)
        {
            log_error("Reloading failed: " + module_name + " is not known to the DB.", SRC);
            return false;
        }

        // do the actual reload of the module
        if (module_source_code && module_name == module_db_name)
        {
            log_info("Reloading module from string: " + module_name);
            module->reload_from_string(module_source_code, false, context.get());
        }
        else
        {
            log_info("Reloading module: " + module_name);
            module->reload(false, context.get());
        }

        if (!m_sdk->log_messages("Reloading failed: " + module_name, context.get(), SRC))
            return false;

        // reflect changed imports
        update_module_dependencies(module_name);
    }

    // update material instances and compiled materials
    std::vector<Mdl_material*> changed_materials;
    std::set<std::string> old_hashes;
    visit_materials([&](Mdl_material* material) // could be done in parallel if compiling
    {                                           // becomes a bottleneck

        // recompile the material and check if the hash changed
        const std::string current_hash = material->get_material_compiled_hash();
        if (!material->recompile_material(context.get()))
            return true; // continue visiting

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
            auto entry = m_target_map[hash];
            if (!entry)
            {
                assert(false && "Target code map contains invalid entries");
                m_target_map.erase(hash);
                continue;
            }

            Mdl_material_target* target = entry->m_target;
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

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::generate_and_compile_targets()
{
    // generate and compile in separate loops just to be able to measure individual timings
    std::atomic<bool> success = true;
    std::vector<std::thread> tasks;
    bool force_single_threading = m_app->get_options()->force_single_threading;

    {
        Timing t("generating target code");
        std::lock_guard<std::mutex> lock(m_targets_mtx);

        std::set<size_t> target_ids_to_delete;
        std::set<std::string> target_map_entries_to_delete;

        for (auto it = m_targets.begin(); it != m_targets.end(); it++)
        {
            Mdl_material_target* target = it->second;

            // mark targets without registered materials for deletion
            if (target->get_material_count() == 0)
            {
                target_ids_to_delete.insert(target->get_id());
                for (const auto& pair : m_target_map)
                    if (pair.second->m_target->get_id() == target->get_id())
                        target_map_entries_to_delete.insert(pair.first);

                continue;
            }

            // skip targets without changes
            if (!target->is_generation_required())
                continue;

            // sequentially
            if (force_single_threading)
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
        }

        for (auto &t : tasks)
            t.join();

        // free the old target if not used anymore
        for (const auto& hash : target_map_entries_to_delete)
            m_target_map.erase(hash);
        for (const auto& id : target_ids_to_delete)
        {
            delete m_targets[id];
            m_targets.erase(id);
        }

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
            if (force_single_threading)
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

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::visit_target_codes(std::function<bool(Mdl_material_target*)> action)
{
    std::lock_guard<std::mutex> lock(m_targets_mtx);
    for (auto it = m_targets.begin(); it != m_targets.end(); it++)
        if (!action(it->second))
            return false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::visit_materials(std::function<bool(Mdl_material*)> action)
{
    return visit_target_codes([&](Mdl_material_target* tc) {
        return tc->visit_materials(action);
    });
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_library::register_mdl_material_description_loader(
    const IMdl_material_description_loader* loader)
{
    m_loaders.push_back(loader);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::visit_material_description_loaders(
    std::function<bool(const IMdl_material_description_loader*)> action)
{
    for (const auto& it : m_loaders)
        if (!action(it))
            return false;
    return true;
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_library::update_module_dependencies(const std::string& module_db_name)
{
    std::lock_guard<std::mutex> lock(m_module_dependencies_mtx);

    std::function<void(const mi::neuraylib::IModule*, const std::string&)> update_recursively =
        [&](const mi::neuraylib::IModule* current_module, const std::string& current_db_name)
    {
        // find/create the entry for this module
        Mdl_module_dependency* dep_info = &m_module_dependencies[current_db_name];

        // TODO check if the module changed since that last update
        // without this check updates are done multiple times when loading scenes
        // with many materials that use the same imports
        // if (dep_info.version == current_module.version)
        //     return;

        std::vector<std::string> updated_dependencies;
        for (mi::Size i = 0, n = current_module->get_import_count(); i < n; ++i)
        {
            std::string imported_db_name = current_module->get_import(i);
            mi::base::Handle<const mi::neuraylib::IModule> imported_module(
                m_sdk->get_transaction().access<mi::neuraylib::IModule>(
                    imported_db_name.c_str()));

            // these modules should not change during runtime
            if (imported_module->is_standard_module() ||
                imported_db_name == "mdl::base" ||
                imported_db_name == "mdl::<builtins>")
                continue;

            updated_dependencies.push_back(imported_db_name);

            // register usages
            m_module_dependencies[imported_db_name].is_imported_by.insert(current_db_name);

            // go on recursively
            update_recursively(imported_module.get(), imported_db_name);
        }

        // handle removed imports
        for (const auto& i_name : dep_info->imports)
            if (std::find(updated_dependencies.begin(), updated_dependencies.end(),
                i_name) == updated_dependencies.end())
                m_module_dependencies[i_name].is_imported_by.erase(module_db_name);

        // replace import list
        dep_info->imports = updated_dependencies;
    };

    mi::base::Handle<const mi::neuraylib::IModule> mod(
        m_sdk->get_transaction().access<mi::neuraylib::IModule>(module_db_name.c_str()));

    update_recursively(mod.get(), module_db_name);
    return;
}

// ------------------------------------------------------------------------------------------------

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

    // if not, collect all information required to create the texture
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

    // no tiles no texture
    if (num_tiles == 0)
        return nullptr;

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

                // TODO
                // currently all 3D textures we have are multi-scatter lookup tables
                // which are float32 textures. so this is hard-coded for now
                case Texture_dimension::Texture_3D:
                    set.entries[tile_id].resource = Texture::create_texture_3d(
                        m_app, GPU_access::shader_resource,
                        canvas->get_resolution_x(),
                        canvas->get_resolution_y(),
                        canvas->get_layers_size(),
                        DXGI_FORMAT_R32_FLOAT,
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
        if (dimension == Texture_dimension::Texture_2D && gamma != 1.0f)
        {
            // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
            mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
                m_sdk->get_image_api().convert(canvas.get(), "Color"));
            gamma_canvas->set_gamma(gamma);
            m_sdk->get_image_api().adjust_gamma(gamma_canvas.get(), 1.0f);
            canvas = gamma_canvas;
        }
        else if (dimension == Texture_dimension::Texture_2D &&
            strcmp(image_type, "Color") != 0 &&
            strcmp(image_type, "Float32<4>") != 0)
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

}}} // mi::examples::mdl_d3d12
