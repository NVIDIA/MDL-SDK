/******************************************************************************
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "light_profile.h"
#include "bsdf_measurement.h"

#include "example_shared.h"

namespace mi { namespace examples { namespace mdl_d3d12 {

Mdl_material_library::Mdl_material_library(
    Base_application* app, Mdl_sdk* sdk)
    : m_app(app)
    , m_sdk(sdk)
    , m_targets()
    , m_targets_mtx()
    , m_textures()
    , m_textures_mtx()
    , m_mbsdfs()
    , m_mbsdfs_mtx()
{
}

// ------------------------------------------------------------------------------------------------

Mdl_material_library::~Mdl_material_library()
{
    m_targets.clear();

    for (auto& resource : m_textures)
        for (auto& entry : resource.second.entries)
            delete entry.resource;

    for (auto& light_profile : m_light_profiles)
        delete light_profile.second;

    for (auto& mbsdf : m_mbsdfs)
        delete mbsdf.second;

    m_loaders.clear();
}

// ------------------------------------------------------------------------------------------------

Mdl_material_target* Mdl_material_library::get_target_for_material_creation(
    const std::string& key)
{
    std::lock_guard<std::mutex> lock(m_targets_mtx);

    // reuse the existing target if the material is already added to a link unit
    auto mapIt = m_targets.find(key);
    if (mapIt == m_targets.end())
    {
        // store a key based on the compiled material hash to identify already handled ones
        mapIt = m_targets.insert(
            { key, std::make_unique<Mdl_material_target>(m_app, m_sdk, key) }).first;
    }
    return mapIt->second.get();
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_library::register_material(Mdl_material* material)
{
    // check if the compiled material is already present and reuse it in that case
    // otherwise create a new target
    auto new_target = get_target_for_material_creation(material->get_material_compiled_hash());
    auto old_target = material->get_target_code();

    // nothing changed
    if (old_target == new_target)
        return;

    // an old target is set that needs to be unregistered
    if (old_target)
        unregister_material(material);

    new_target->register_material(material);
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_library::unregister_material(Mdl_material* material)
{
    Mdl_material_target* target = material->get_target_code();
    if (!target)
        return;

    target->unregister_material(material);

    // since we can have more than one material registered with this target code,
    // we delete the target code itself only if the last material was unregistered.
    std::lock_guard<std::mutex> lock(m_targets_mtx);
    if (target->get_material_count() == 0)
    {
        const std::string& hash = target->get_compiled_material_hash();
        assert(m_targets.find(hash) != m_targets.end() && "untracked target found");
        m_targets.erase(hash);
    }
}

// ------------------------------------------------------------------------------------------------

Mdl_material* Mdl_material_library::create_material()
{
    Mdl_material* mat = new Mdl_material(m_app);
    return mat;
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_library::destroy_material(Mdl_material* material)
{
    unregister_material(material);
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

    // assign the material to the target
    // this unregisters the material from the old target as well
    register_material(material);
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
    register_material(material);

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

    const std::vector<std::string>& modules = desc.get_module_db_names();
    bool success = true;
    for(const auto& module_db_name : modules)
        success &= reload_module(module_db_name, code);

    // repair materials
    success &= repair_materials_after_reload(targets_changed);
    return success;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::reload_module(const std::string& module_db_name)
{
    return reload_module(module_db_name, "");
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::reload_module(
    const std::string& module_db_name,
    const char* module_source_code)
{
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
                if (!contains(dependencies, i_name) && !contains(to_process, i_name))
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
    return true;
}

bool Mdl_material_library::repair_materials_after_reload(bool& targets_changed)
{
    // update material instances and compiled materials
    targets_changed = false;
    std::vector<Mdl_material*> changed_materials;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());
    visit_materials([&](Mdl_material* material) // could be done in parallel if compiling
    {                                           // becomes a bottleneck

        // recompile the material and check if the hash changed
        const std::string current_hash = material->get_material_compiled_hash();
        if (!material->recompile_material(context.get()))
            return true; // continue visiting

        if (current_hash != material->get_material_compiled_hash())
        {
            changed_materials.push_back(material);
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
        register_material(mat); // no lock, single threaded here
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

        // since targets could contain multiple materials we need to find the set of unique ones
        std::unordered_set<Mdl_material_target*> unique_targets;
        for (const auto& it : m_targets)
        {
            // don't expect targets here that have no material registered
            if (it.second->get_material_count() == 0)
            {
                assert(false && "unused target found");
                continue;
            }
            unique_targets.insert(it.second.get());
        }

        for (auto target : unique_targets)
        {
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

        if (!success.load())
            return false;
    }

    tasks.clear();
    {
        Timing t("compiling target code");
        visit_target_codes([&](Mdl_material_target* target)
        {
            if (target->get_material_count() == 0)
            {
                assert(false && "scheduled target without materials for compilation");
                return true;
            }

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
    std::unordered_set<Mdl_material_target*> unique_targets;
    for (const auto& it : m_targets)
    {
        // make sure to visit the target only once in case it contains multiple materials
        if (unique_targets.insert(it.second.get()).second)
        {
            if (!action(it.second.get()))
                return false;
        }
    }
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
    std::unique_ptr<IMdl_material_description_loader> loader)
{
    m_loaders.push_back(std::move(loader));
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_library::visit_material_description_loaders(
    std::function<bool(const IMdl_material_description_loader*)> action)
{
    for (const auto& it : m_loaders)
        if (!action(it.get()))
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

Mdl_texture_set* Mdl_material_library::access_texture_resource(
    std::string db_name,
    Texture_dimension dimension,
    D3DCommandList* command_list)
{
    // check if the texture already exists
    {
        std::lock_guard<std::mutex> lock(m_textures_mtx);
        auto found = m_textures.find(db_name);
        if (found != m_textures.end())
            return &found->second;
    }

    // if not, collect all information required to create the texture
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        m_sdk->get_transaction().access<mi::neuraylib::ITexture>(db_name.c_str()));

    mi::base::Handle<const mi::neuraylib::IImage> image(
        m_sdk->get_transaction().access<mi::neuraylib::IImage>(texture->get_image()));

    // collect basic information about the number of tiles
    mi::Size num_tiles = 0;
    mi::Sint32 u_min = std::numeric_limits<mi::Sint32>::max();
    mi::Sint32 u_max = std::numeric_limits<mi::Sint32>::min();
    mi::Sint32 v_min = u_min;
    mi::Sint32 v_max = u_max;

    mi::Size num_frames = image->get_length();
    for (mi::Size f = 0; f < num_frames; ++f)
    {
        num_tiles += image->get_frame_length(f);
    }

    // create empty textures for all tiles
    {
        std::lock_guard<std::mutex> lock(m_textures_mtx);
        auto found = m_textures.find(db_name);
        if (found != m_textures.end())
            return &found->second; // return the other

        // add the created texture to the library
        m_textures[db_name] = Mdl_texture_set();
        Mdl_texture_set& set = m_textures[db_name];
        bool has_tiles_or_frames = image->is_uvtile() || image->is_animated();

        // create the resources
        set.entries = std::vector<Mdl_texture_set::Entry>(
            num_tiles, Mdl_texture_set::Entry());

        mi::Size global_tile_id = 0;
        for (mi::Size f = 0; f < num_frames; ++f)
        {
            for (mi::Size tile_id = 0; tile_id < image->get_frame_length(f); ++tile_id)
            {
                mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
                    image->get_canvas(f, tile_id, 0));

                // create the d3d texture
                switch (dimension)
                {
                case Texture_dimension::Texture_2D:
                    set.entries[global_tile_id].resource = Texture::create_texture_2d(
                        m_app, GPU_access::shader_resource,
                        canvas->get_resolution_x(),
                        canvas->get_resolution_y(),
                        DXGI_FORMAT_R32G32B32A32_FLOAT, // TODO
                        has_tiles_or_frames
                        ? (db_name + "_frame_" + std::to_string(f) +
                            "_tile_" + std::to_string(tile_id)) : db_name);
                    break;

                // TODO
                // currently all 3D textures we have are multi-scatter lookup tables
                // which are float32 textures. so this is hard-coded for now
                case Texture_dimension::Texture_3D:
                    set.entries[global_tile_id].resource = Texture::create_texture_3d(
                        m_app, GPU_access::shader_resource,
                        canvas->get_resolution_x(),
                        canvas->get_resolution_y(),
                        canvas->get_layers_size(),
                        DXGI_FORMAT_R32_FLOAT,
                        has_tiles_or_frames
                        ? (db_name + "_frame_" + std::to_string(f) +
                            "_tile_" + std::to_string(tile_id)) : db_name);
                    break;

                default:
                    log_error("Unhandled texture dimension: " + db_name, SRC);
                    continue;
                }


                mi::Sint32 u, v;
                image->get_uvtile_uv(f, tile_id, u, v);
                set.entries[global_tile_id].frame = int32_t(image->get_frame_number(f));
                set.entries[global_tile_id].uvtile_u = u;
                set.entries[global_tile_id].uvtile_v = v;
                u_min = std::min(u_min, u);
                u_max = std::max(u_max, u);
                v_min = std::min(v_min, v);
                v_max = std::max(v_max, v);
                global_tile_id++;
            }
        }

        // resource set data
        set.frame_first = int32_t(image->get_frame_number(0));
        set.frame_last = int32_t(image->get_frame_number(num_frames - 1));
        set.uvtile_u_min = u_min;
        set.uvtile_u_max = u_max;
        set.uvtile_v_min = v_min;
        set.uvtile_v_max = v_max;

        // release the lock so other threads can reference the texture even when the
        // actual data is not loaded yet. So when loading in parallel, wait if the data
        // is required on the GPU
    }

    // load the actual texture data and copy it to GPU
    bool success = true;
    Mdl_texture_set& set = m_textures[db_name];
    uint8_t* buffer = nullptr;
    size_t buffer_size = 0;
    mi::Size global_tile_id = 0;
    for (mi::Size f = 0; f < num_frames; ++f)
    {
        for (mi::Size tile_id = 0; tile_id < image->get_frame_length(f); ++tile_id)
        {
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
                image->get_canvas(f, tile_id, 0));

            // For simplicity, the texture access functions are only implemented for float4 and
            // gamma is pre-applied here (all images are converted to linear space).
            // Convert to linear color space if necessary
            float gamma = texture->get_effective_gamma(f, tile_id);
            char const* image_type = image->get_type(f, tile_id);
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

            Texture* texture_resource = static_cast<Texture*>(set.entries[global_tile_id].resource);

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
                mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(l));
                const uint8_t* tex_data = static_cast<const uint8_t*>(tile->get_data());
                // Note, the data is alligned bottom up. So the first pixel row we get is from the bottom.
                // Direct3D expects the data the other way around.

                // copy line by line if required
                uint8_t* buffer_layer = buffer + gpu_row_pitch * tex_height * l;
                if (gpu_row_pitch != data_row_pitch)
                {
                    for (size_t r = 0; r < tex_height; ++r)
                        memcpy((void*)(buffer_layer + r * gpu_row_pitch),
                            (void*)(tex_data + r * data_row_pitch),
                            data_row_pitch);
                }
                else // copy directly
                {
                    memcpy((void*)buffer_layer, (void*)tex_data, data_row_pitch * tex_height);
                }
            }

            // copy data to the GPU
            if (texture_resource->upload(command_list, (const uint8_t*)buffer, gpu_row_pitch))
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
                std::lock_guard<std::mutex> lock(m_textures_mtx);
                for (auto& entry : set.entries)
                    delete entry.resource;

                m_textures.erase(db_name);
                delete[] buffer;
                return nullptr;
            }
            global_tile_id++;
        }
    }

    delete[] buffer;
    return &m_textures[db_name];
}

Light_profile* Mdl_material_library::access_light_profile_resource(
    std::string db_name,
    D3DCommandList* command_list)
{
    {
        std::lock_guard<std::mutex> lock(m_light_profiles_mtx);

        // check if the light profile already exists
        auto found = m_light_profiles.find(db_name);
        if (found != m_light_profiles.end())
            return found->second;

        // if not, create the resource
        mi::base::Handle<const mi::neuraylib::ILightprofile> light_profile(
            m_sdk->get_transaction().access<mi::neuraylib::ILightprofile>(db_name.c_str()));

        // add the created light profile to the library
        m_light_profiles[db_name] = new Light_profile(m_app, light_profile.get(), db_name);

        // release the lock so other threads can reference the mbsdf even when the
        // actual data is not loaded yet. So when loading in parallel, wait if the data
        // is required on the GPU
    }

    return m_light_profiles[db_name];
}

Bsdf_measurement* Mdl_material_library::access_bsdf_measurement_resource(
    std::string db_name,
    D3DCommandList* command_list)
{
    {
        std::lock_guard<std::mutex> lock(m_mbsdfs_mtx);

        // check if the bsdf measurement already exists
        auto found = m_mbsdfs.find(db_name);
        if (found != m_mbsdfs.end())
            return found->second;

        // if not, create the resource
        mi::base::Handle<const mi::neuraylib::IBsdf_measurement> bsdf_measurement(
            m_sdk->get_transaction().access<mi::neuraylib::IBsdf_measurement>(db_name.c_str()));

        // add the created mbsdf to the library
        m_mbsdfs[db_name] = new Bsdf_measurement(m_app, bsdf_measurement.get(), db_name);

        // release the lock so other threads can reference the mbsdf even when the
        // actual data is not loaded yet. So when loading in parallel, wait if the data
        // is required on the GPU
    }

    return m_mbsdfs[db_name];
}

}}} // mi::examples::mdl_d3d12
