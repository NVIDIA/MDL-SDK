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

 // examples/mdl_sdk/dxr/mdl_d3d12/mdl_material_description.h

#ifndef MDL_D3D12_MDL_MATERIAL_DESCRIPTION_H
#define MDL_D3D12_MDL_MATERIAL_DESCRIPTION_H

#include "common.h"
#include "scene.h"
#include <mi/base/handle.h>

namespace mi { namespace neuraylib
{
    class IExpression_list;
    class IMdl_execution_context;
}}

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Base_options;
    class Mdl_sdk;

    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    /// Interface to load different types of materials based on naming convention in glTF.
    class IMdl_material_description_loader
    {
    public:
        /// check if this loaded is responsible for loading the material
        /// based on out naming convention.
        /// \returns true if the material can be loaded.
        virtual bool match_gltf_name(const std::string& gltf_name) const = 0;

        /// Generate source code based an the provided \c gltf_name.
        virtual std::string generate_mdl_source_code(
            const std::string& gltf_name,
            const std::string& scene_directory) const = 0;

        /// Get a small token used as prefix in the GUI and the scene graph. e.g. "[MDLE]"
        virtual std::string get_scene_name_prefix() const = 0;

        /// True if the module can be reloaded. If the code is static, false should be returned.
        virtual bool supports_reload() const = 0;

        /// If the loader matches certain file types, this will return the number of types or 0.
        virtual size_t get_file_type_count() const = 0;

        /// If the loader matches certain file types, this will return the extension of the i`th
        /// type or multiple extensions separated by ';'
        virtual std::string get_file_type_extension(size_t index) const = 0;

        /// If the loader matches certain file types, this will return the description of the i`th
        /// type.
        virtual std::string get_file_type_description(size_t index) const = 0;
    };


    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    /// Description of a mdl material instance to create.
    /// This handles the loading of MDL modules for all supported material formats too.
    /// After calling load_material_definition, all information to create an MDL material
    /// instance are available.
    class Mdl_material_description final
    {
    public:
        static const std::string Invalid_material_identifier;

        /// Constructor.
        /// Create a material description for the invalid (fall-back) material.
        explicit Mdl_material_description();

        /// Constructor.
        /// Create a material description as described in a scene file.
        explicit Mdl_material_description(const IScene_loader::Material& parameters);

        /// Constructor.
        /// Create a material description by a unique name like the fully qualified MDL material
        /// name, or an MDLE file path.
        explicit Mdl_material_description(const std::string& unique_material_identifier);

        /// Destructor.
        ~Mdl_material_description();

        /// Get the material parameters read from GLTF.
        const IScene_loader::Material& get_material_parameters() const;

        /// Prepares the material for being instantiated.
        /// Depending on the type and source of the material, this can be a more time consuming
        /// Operation. Afterwards the MDL Material definition for this material is available in the
        /// Database, or in case of errors and invalid material that still can be rendered will
        /// be returned when the material library calls the getter functions to setup a material
        /// instance. In that case, the method will return false.
        bool load_material_definition(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        /// Checks if 'load_material_definition' was called and if the resulting data is available.
        bool is_loaded() const;

        /// Get the mdl parameter list for this material instance. This can contain complex graphs.
        /// When this function is called by the material library, this materials parenting module
        /// is already loaded, so it is valid to use exported types and types from imported modules.
        /// All other types present in the database are also valid.
        const mi::neuraylib::IExpression_list* get_parameters() const;

        /// Get the (qualified) module name.
        /// This is expected to be of the form: [::<package>]::<module>
        /// Note: available after calling load_material_definition.
        const char* get_qualified_module_name() const;

        /// Get the (simple) material name.
        /// This is only <material>-part of a qualified material name
        /// of the form [::<package>]::<module>::<material>
        /// Note: available after calling load_material_definition.
        const char* get_material_name() const;

        /// Name of the material in the scene, which is only for display on the UI or in logs.
        /// Note: available after calling load_material_definition.
        const char* get_scene_name() const;

        // get material flags e.g. for optimization
        /// Note: available after calling load_material_definition.
        IMaterial::Flags get_flags() const;

        /// Indicates if this the fall-back material. True when 'load_material_definition' failed.
        /// Note: available after calling load_material_definition.
        bool is_fallback() const;

        /// Indicates if the material definition can be reloaded when changed on disk.
        /// Note: available after calling load_material_definition.
        bool supports_reloading() const;

        /// Get the database name of the module that contains the material definition.
        const char* get_module_db_name() const;

        /// Get the database name of the material definition.
        const char* get_material_defintion_db_name() const;

        /// Generate and get the source code of materials that are loaded from string.
        /// MDLs that are loaded from a search path (including the GLTF support materials)
        /// return NULL. Modules that don't support reloading also return NULL.
        const char* regenerate_source_code(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

    private:
        bool load_material_definition_mdl(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        bool load_material_definition_mdle(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        bool load_material_definition_gltf_support(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        bool load_material_definition_fallback(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        bool load_material_definition_loader(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context,
            const IMdl_material_description_loader* loader);

        // shared method to load the module corresponding to the definition.
        bool load_mdl_module(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        // create mdl parameters from the material description data.
        void parameterize_gltf_support_material(
            Mdl_sdk& sdk,
            const std::string& scene_directory,
            mi::neuraylib::IMdl_execution_context* context);

        IScene_loader::Material m_parameters;
        bool m_is_loaded;
        bool m_is_fallback;
        bool m_supports_reloading;
        const IMdl_material_description_loader* m_loader;

        mi::base::Handle<mi::neuraylib::IExpression_list> m_parameter_list;
        std::string m_module_db_name;
        std::string m_material_defintion_db_name;

        std::string m_qualified_module_name;
        std::string m_material_name;
        std::string m_name_in_scene;
        std::string m_source_code;
        IMaterial::Flags m_flags;
        uint64_t m_unique_id;
    };

}}} // mi::examples::mdl_d3d12
#endif
