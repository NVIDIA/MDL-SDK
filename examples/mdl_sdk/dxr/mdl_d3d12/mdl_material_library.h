/******************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
    class Mdl_material_shared;
    class Mdl_sdk;
    

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

    public:
        // Constructor.
        explicit Mdl_material_library(
            Base_application* app, 
            Mdl_sdk* sdk, 
            bool share_target_code);
        
        // Destructor.
        virtual ~Mdl_material_library();

        // creates a new material or reuses an existing one
        Mdl_material* create(const Mdl_material_description& material_desc);

        bool reload_material(Mdl_material* material, bool& targets_changed);


        bool reload_module(const std::string& module_db_name, bool& targets_changed);

        /// Creates or updates the generated HLSL code and the compiled DXIL libraries.
        bool generate_and_compile_targets();

        /// gets the global link unit to be use for all materials that are loaded
        /// only available if this library was created with `share_target_code` set.
        Mdl_material_target* get_shared_target_code();

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

        /// get access to the texture data by the texture database name and create a resource.
        /// if there resource is loaded already, no loading is required
        Texture* access_texture_resource(
            std::string db_name, 
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

        /// depending 'share_target_code' flag, the shared target is returned or a new created one.
        Mdl_material_library::Target_entry* get_target_for_material_creation(
            const std::string& key);

        Base_application* m_app;
        Mdl_sdk* m_sdk;

        bool m_share_target_code;

        // map that stores targets based on the compiled material hash
        std::map<size_t, Mdl_material_target*> m_targets;
        std::map<std::string, Target_entry*> m_target_map;
        std::mutex m_targets_mtx;

        // all texture resources used by MDL materials
        // TODO: use ref counting when considering dynamic loading and unloading of materials
        std::map<std::string, Texture*> m_textures;
        std::mutex m_textures_mtx;
    };
}

#endif
