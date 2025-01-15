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

 // examples/mdl_sdk/dxr/mdl_d3d12/mdl_material_target.h

#ifndef MDL_D3D12_MDL_MATERIAL_TARGET_H
#define MDL_D3D12_MDL_MATERIAL_TARGET_H

#include "common.h"
#include "scene.h"
#include "shader.h"
#include <mi/mdl_sdk.h>

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Mdl_sdk;
    class Mdl_material;
    class Mdl_material_target;
    enum class Mdl_resource_kind;
    struct Mdl_resource_assignment;
    struct Mdl_string_constant;

    // --------------------------------------------------------------------------------------------

    enum class Material_code_feature
    {
        HAS_INIT            = 1 << 0,
        SURFACE_SCATTERING  = 1 << 1,
        SURFACE_EMISSION    = 1 << 2,
        BACKFACE_SCATTERING = 1 << 3,
        BACKFACE_EMISSION   = 1 << 4,
        VOLUME_ABSORPTION   = 1 << 5,
        CUTOUT_OPACITY      = 1 << 6,
        CAN_BE_THIN_WALLED  = 1 << 7,
        HAS_AOVS            = 1 << 8
    };

    // --------------------------------------------------------------------------------------------

    struct Mdl_aov_info
    {
        uint32_t runtime_index;
        std::string expression_path;
        std::string hlsl_name;
        std::string mdl_type_name;
    };

    // --------------------------------------------------------------------------------------------

    /// Information about a target that is required by a material.
    struct Mdl_material_target_interface
    {
        mi::Size argument_layout_index = 0;
        uint32_t material_code_paths = 0;

        void add_code_feature(Material_code_feature to_add) 
        { 
            material_code_paths |= static_cast<uint32_t>(to_add);
        }

        void remove_code_feature(Material_code_feature to_remove) {
            material_code_paths &= ~static_cast<uint32_t>(to_remove);
        }

        bool has_code_feature(Material_code_feature to_check) {
            return (material_code_paths & static_cast<uint32_t>(to_check)) != 0;
        }

        std::vector< Mdl_aov_info> aovs;
    };

    // --------------------------------------------------------------------------------------------

    // The target that is responsible for shader code generation.
    // for simplicity, we use one target code for a unique material
    // (same compiled material but potentially different parameter values)
    class Mdl_material_target
    {
        /// Callback that notifies the application about new resources when generating an
        /// argument block for an existing target code.
        class Resource_callback
            : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
        {
        public:
            // Constructor.
            explicit Resource_callback(
                Mdl_sdk* sdk,
                Mdl_material_target* target,
                Mdl_material* material);

            // Destructor.
            virtual ~Resource_callback() = default;

            /// Returns a resource index for the given resource value usable by the target code
            /// resource handler for the corresponding resource type.
            ///
            /// \param resource  the resource value
            ///
            /// \returns a resource index or 0 if no resource index can be returned
            mi::Uint32 get_resource_index(mi::neuraylib::IValue_resource const *resource) override;

            /// Returns a string identifier for the given string value usable by the target code.
            ///
            /// The value 0 is always the "not known string".
            ///
            /// \param s  the string value
            mi::Uint32 get_string_index(mi::neuraylib::IValue_string const *s) override;

        private:
            Mdl_sdk* m_sdk;
            Mdl_material_target* m_target;
            Mdl_material* m_material;
        };

        // --------------------------------------------------------------------

    public:

        /// Constructor.
        explicit Mdl_material_target(
            Base_application* app,
            Mdl_sdk* sdk,
            const std::string& compiled_material_hash);

        /// Destructor.
        ~Mdl_material_target();

        /// Create a resource callback that is required when creating new argument blocks for
        /// a material registered to this target.
        mi::neuraylib::ITarget_resource_callback* create_resource_callback(Mdl_material* material);

        /// Get the hash of the material(s) this target generates code for.
        const std::string& get_compiled_material_hash() { return m_compiled_material_hash; }

        /// Get the generated target.
        /// Note, this is managed and has to put into a handle.
        const mi::neuraylib::ITarget_code* get_target_code() const;

        /// check if generation is required after changing materials.
        bool is_generation_required() const { return m_generation_required; }

        /// generate HLSL code for all functions in the link unit.
        bool generate();

        /// get the mdl target code object, which is available after calling generate.
        const mi::neuraylib::ITarget_code* get_generated_target() const;

        /// get the result of the generation step.
        const std::string& get_hlsl_source_code() const { return m_hlsl_source_code; }

        /// check if compilation is required after changing materials.
        bool is_compilation_required() const { return m_compilation_required; }

        /// compiles the generated HLSL code into a DXIL library.
        bool compile();

        /// get the shader entry point name for the closet hit shader
        const std::string get_entrypoint_radiance_closest_hit_name() const
        {
            return m_radiance_closest_hit_name;
        }

        /// get the shader entry point name for the any hit shader
        const std::string get_entrypoint_radiance_any_hit_name() const
        {
            return m_radiance_any_hit_name;
        }

        /// get the shader entry point name for the any hit shader
        const std::string get_entrypoint_shadow_any_hit_name() const
        {
            return m_shadow_any_hit_name;
        }

        /// get the result of the compiling step.
        const std::vector<Shader_library>& get_dxil_compiled_libraries() const
        {
            return m_dxil_compiled_libraries;
        }

        /// Get the start index of the resources belonging to material target.
        /// The index can be used directly in the shader code.
        uint32_t get_resource_heap_index() const
        {
            return static_cast<uint32_t>(m_first_resource_heap_handle.get_heap_index());
        }

        /// get the number of materials that are registered at this target
        size_t get_material_count() const { return m_materials.size(); }

        /// Assign a material or reassign a changed material to this target.
        /// Afterwards, the target code has to be regenerated and recompiled.
        void register_material(Mdl_material* material);

        /// Removes a material from this target. This happens when disposing the material.
        /// Afterwards, the target code has to be regenerated and recompiled.
        bool unregister_material(Mdl_material* material);

        /// in case the material is reused with a different set of parameters, there have to
        /// be texture slots for all possible textures in the generated code
        /// called by the material when creating the descriptor table
        size_t get_material_resource_count(Mdl_resource_kind kind) const;

        /// get the per target resources
        const std::vector<Mdl_resource_assignment>& get_resources(Mdl_resource_kind kind) const
        {
            return m_target_resources.find(kind)->second; // always present
        }

        /// register a new string constant for the target code or return an already mapped value.
        /// this can be used to force the availability of a certain scene data element
        /// even though it's not used in the mdl generated code.
        uint32_t map_string_constant(const std::string& string_value);

        /// get the target string constants
        const std::vector<Mdl_string_constant>& get_string_constants()
        {
            return m_target_string_constants;
        }

        /// iterate over all materials inside a lock.
        ///
        /// \param action   action to run while visiting a material.
        ///                 if the action returns false, the iteration is aborted.
        ///
        /// \returns        false if the iteration was aborted, true otherwise.
        bool visit_materials(std::function<bool(Mdl_material*)> action);

        /// Get the shader suffix for this target to create unique hit groups names.
        /// is not enabled.
        const std::string get_shader_name_suffix() const {
            return m_compiled_material_hash;
        }

    private:
        /// adds a material to this target.
        bool add_material_to_link_unit(
            Mdl_material_target_interface& interface_data,
            Mdl_material* material,
            mi::neuraylib::ILink_unit* link_unit,
            mi::neuraylib::IMdl_execution_context* context);

        Base_application* m_app;
        Mdl_sdk* m_sdk;

        // The compiled material hash of the material(s) we generate code for
        const std::string m_compiled_material_hash;

        mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;

        bool m_generation_required;
        std::string m_hlsl_source_code;
        bool m_compilation_required;
        std::map<std::string, std::string> m_hlsl_compilation_defines;

        std::string m_radiance_any_hit_name;
        std::string m_radiance_closest_hit_name;
        std::string m_shadow_any_hit_name;
        std::vector<Shader_library> m_dxil_compiled_libraries;

        Descriptor_heap_handle m_first_resource_heap_handle;
        Descriptor_table m_resource_descriptor_table;

        Buffer* m_read_only_data_segment;

        /// resources present in the target code already
        /// mainly the ones in the body but also the ones in the parameter list of the instance
        /// that is used during compilation of the instance to a compiled material
        std::map<Mdl_resource_kind, std::vector<Mdl_resource_assignment>> m_target_resources;

        /// string constants present in the target code already
        std::vector<Mdl_string_constant> m_target_string_constants;

        /// when resources are managed by the material we need to account for different numbers
        /// of resources for different instances that use the same code with a different parameter
        /// set (e.g. one material might have a diffuse map only where another has an additional
        /// normal map). Since the descriptor tables of materials that use the same code have to be
        /// equal, descriptors for all possible resources have to be available.
        std::map<Mdl_resource_kind, size_t> m_material_resource_count;

        // all materials that use this target code (owned by the scene)
        std::map<size_t, Mdl_material*> m_materials;
        std::mutex m_materials_mtx;
    };

}}} // mi::examples::mdl_d3d12
#endif
