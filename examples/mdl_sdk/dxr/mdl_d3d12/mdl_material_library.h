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

 // examples/mdl_sdk/dxr/mdl_d3d12/mdl_material_library.h

#ifndef MDL_D3D12_MDL_MATERIAL_LIBRARY_H
#define MDL_D3D12_MDL_MATERIAL_LIBRARY_H

#include "common.h"
#include "command_queue.h"
#include "mdl_material_target.h"

namespace mi
{
    namespace neuraylib
    {
        class IExpression_list;
        class IMdl_execution_context;
    }
}

namespace mdl_d3d12
{
    class Base_application;
    class Mdl_material;
    class Mdl_material_description;
    class Mdl_sdk;
    struct Mdl_resource_set;
    enum class Texture_dimension;

    /// keeps all materials that are loaded by the application
    class Mdl_material_library
    {
        struct Target_entry
        {
            // Constructor.
            explicit Target_entry(Mdl_material_target* target);

            // Destructor.
            ~Target_entry();

            Mdl_material_target* m_target;
            std::mutex m_mutex;
        };

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

    public:
        /// Constructor.
        explicit Mdl_material_library(
            Base_application* app, 
            Mdl_sdk* sdk);
        
        /// Destructor.
        virtual ~Mdl_material_library();

        /// creates a new material or reuses an existing one.
        Mdl_material* create(const Mdl_material_description& material_desc);

        /// reload all modules required by a certain material.
        bool reload_material(Mdl_material* material, bool& targets_changed);

        /// recompile all materials that got invalid after a reload.
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

        /// Updates the depencies between modules after loading or reloading a module.
        void update_module_dependencies(const std::string& module_db_name);

        /// get access to the texture data by the texture database name and create a resource.
        /// if there resource is loaded already, no loading is required
        Mdl_resource_set* access_texture_resource(
            std::string db_name, 
            Texture_dimension dimension,
            D3DCommandList* command_list);

    private:
        /// Triggers the loading of a module that is not yet loaded.
        ///
        /// \param qualified_module_name    Qualified name of the module to load.
        /// \param reload                   Triggers a reload of the module is already loaded.
        /// \param context                  Used for error reporting.
        ///
        /// \returns                        The modules DB name or empty string in case of error.
        std::string load_module(
            const std::string& qualified_module_name,
            bool reload,
            mi::neuraylib::IMdl_execution_context* context);

        /// get an already existing target or a created new one.
        Mdl_material_library::Target_entry* get_target_for_material_creation(
            const std::string& key);

        Base_application* m_app;
        Mdl_sdk* m_sdk;

        // map that stores targets based on the compiled material hash
        std::map<size_t, Mdl_material_target*> m_targets;
        std::map<std::string, Target_entry*> m_target_map;
        std::mutex m_targets_mtx;

        // all texture resources used by MDL materials
        // TODO: use ref counting when considering dynamic loading and unloading of materials
        std::map<std::string, Mdl_resource_set> m_resources;
        std::mutex m_resources_mtx;

        /// depencies between loaded modules
        std::unordered_map<std::string, Mdl_module_dependency> m_module_dependencies;
        std::mutex m_module_dependencies_mtx;
    };
}

#endif
