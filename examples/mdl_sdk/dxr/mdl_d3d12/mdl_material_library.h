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

 // examples/mdl_sdk/dxr/mdl_d3d12/mdl_material_library.h

#ifndef MDL_D3D12_MDL_MATERIAL_LIBRARY_H
#define MDL_D3D12_MDL_MATERIAL_LIBRARY_H

#include "common.h"
#include "command_queue.h"
#include "mdl_material_target.h"

namespace mi { namespace neuraylib
{
    class IExpression_list;
    class IMdl_execution_context;
}}

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Mdl_material;
    class Mdl_material_description;
    class IMdl_material_description_loader;
    class Mdl_sdk;
    class Light_profile;
    class Bsdf_measurement;
    struct Mdl_texture_set;
    enum class Texture_dimension;

    // --------------------------------------------------------------------------------------------

    /// keeps all materials that are loaded by the application
    class Mdl_material_library
    {
        // --------------------------------------------------------------------

        /// Import relationships between loaded modules.
        struct Mdl_module_dependency
        {
            explicit Mdl_module_dependency()
                : imports()
                , is_imported_by()
            {}

            /// List of modules that this module depends on.
            /// Contains all modules that are imported directly or indirectly.
            /// Note, this list is only filled when the module is loaded directly, i.e.,
            /// calling load_module for it. Otherwise it's empty as the info is not needed.
            std::vector<std::string> imports;

            /// List of all modules that depend on this module directly or indirectly.
            /// All modules that have to reloaded when this module is changed.
            std::unordered_set<std::string> is_imported_by;
        };

        // --------------------------------------------------------------------

    public:
        /// Constructor.
        explicit Mdl_material_library(
            Base_application* app,
            Mdl_sdk* sdk);

        /// Destructor.
        virtual ~Mdl_material_library();

        /// Creates a new material to be used in the scene.
        /// The actual type and parameterization is set using a description.
        Mdl_material* create_material();

        /// Called by the Mdl_material destructor to unregister the material.
        void destroy_material(Mdl_material* material);

        /// Set the type and parameterization for this material as defined
        /// in the Scene to load or the material to assign.
        /// If this fails, an invalid material, that still can be rendered will
        /// be assigned instead. In that case the method returns false.
        bool set_description(
            Mdl_material* material,
            const Mdl_material_description& material_desc);

        /// Set a pink error material to signalize problems without crashing.
        void set_invalid_material(Mdl_material* material);

        /// Reload all modules required by a certain material.
        bool reload_material(Mdl_material* material, bool& targets_changed);

        /// Reload an (already loaded) module and all importing modules in order
        /// get back to a consistent state.
        bool reload_module(const std::string& module_db_name, bool& targets_changed);

        /// Reload an (already loaded) module and all importing modules in order
        /// get back to a consistent state. If \c module_source_code is not empty,
        /// the module is reloaded from string.
        bool reload_module(
            const std::string& module_db_name,
            const char* module_source_code,
            bool& targets_changed);

        /// recompile a material after parameter changes in instance compilation mode
        /// or after structural changes in class compilation mode.
        bool recompile_material(Mdl_material* material, bool& targets_changed);

        /// recompile all material. E.g., after global compilation settings changed.
        bool recompile_materials(bool& targets_changed);

        /// Creates or updates the generated HLSL code and the compiled DXIL libraries.
        bool generate_and_compile_targets();

        /// iterate over all target codes inside a lock.
        ///
        /// \param action   action to run while visiting a target code.
        ///                 if the action returns false, the iteration is aborted.
        ///
        /// \returns        false if the iteration was aborted, true otherwise.
        bool visit_target_codes(std::function<bool(Mdl_material_target*)> action);

        /// iterate over all materials inside a lock.
        ///
        /// \param action   action to run while visiting a material.
        ///                 if the action returns false, the iteration is aborted.
        ///
        /// \returns        false if the iteration was aborted, true otherwise.
        bool visit_materials(std::function<bool(Mdl_material*)> action);

        /// Updates the dependencies between modules after loading or reloading a module.
        void update_module_dependencies(const std::string& module_db_name);

        /// get access to the texture data by the texture database name and create a resource.
        /// if the resource is loaded already, no loading is required
        Mdl_texture_set* access_texture_resource(
            std::string db_name,
            Texture_dimension dimension,
            D3DCommandList* command_list);

        /// get access to the light profile data by the database name and create a resource.
        /// if the resource is loaded already, no loading is required
        Light_profile* access_light_profile_resource(
            std::string db_name,
            D3DCommandList* command_list);

        /// get access to the bsdf measurement data by the database name and create a resource.
        /// if the resource is loaded already, no loading is required
        Bsdf_measurement* access_bsdf_measurement_resource(
            std::string db_name,
            D3DCommandList* command_list);

        /// Add a custom MDL code generated to handle materials by naming convention in gltf.
        void register_mdl_material_description_loader(
            std::unique_ptr<IMdl_material_description_loader> loader);

        /// iterate over all registered loaders
        ///
        /// \param action   action to run while visiting a material.
        ///                 if the action returns false, the iteration is aborted.
        ///
        /// \returns        false if the iteration was aborted, true otherwise.
        bool visit_material_description_loaders(
            std::function<bool(const IMdl_material_description_loader*)> action);

    private:
        /// get an already existing target or a created new one.
        Mdl_material_target* get_target_for_material_creation(
            const std::string& key);

        /// registers a material at the target code that matches the material hash.
        void register_material(Mdl_material* material);

        /// unregister a material from its current target code.
        void unregister_material(Mdl_material* material);

        Base_application* m_app;
        Mdl_sdk* m_sdk;

        // map that stores targets based on the compiled material hash,
        // which all materials registered with this target have in common.
        std::map<std::string, std::unique_ptr<Mdl_material_target>> m_targets;
        std::mutex m_targets_mtx;

        // all texture resources used by MDL materials
        // TODO: use ref counting when considering dynamic loading and unloading of materials
        std::map<std::string, Mdl_texture_set> m_textures;
        std::mutex m_textures_mtx;

        // all light profile resources used by MDL materials
        std::map<std::string, Light_profile*> m_light_profiles;
        std::mutex m_light_profiles_mtx;

        // all MBSDF resources used by MDL materials
        std::map<std::string, Bsdf_measurement*> m_mbsdfs;
        std::mutex m_mbsdfs_mtx;

        /// dependencies between loaded modules
        std::unordered_map<std::string, Mdl_module_dependency> m_module_dependencies;
        std::mutex m_module_dependencies_mtx;

        /// registered loaders to process generated MDL materials
        std::vector<std::unique_ptr<IMdl_material_description_loader>> m_loaders;
    };

}}} // mi::examples::mdl_d3d12
#endif
