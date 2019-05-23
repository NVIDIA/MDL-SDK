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

// examples/mdl_sdk/dxr/mdl_d3d12/mdl_material.h

#ifndef MDL_D3D12_MDL_MATERIAL_H
#define MDL_D3D12_MDL_MATERIAL_H

#include "common.h"
#include "base_application.h"
#include "buffer.h"
#include "scene.h"
#include "shader.h"
#include <mi/mdl_sdk.h>

namespace mdl_d3d12
{
    class Base_application;
    class Descriptor_heap;
    class Material;
    class Mdl_target;
    class Mdl_material;
    class Mdl_material_info;
    class Mdl_material_library;
    class Mdl_material_shared;

    class Mdl_sdk
    {
    public:
        explicit Mdl_sdk(Base_application* app);
        virtual ~Mdl_sdk();

        bool is_running() const { return m_hlsl_backend.is_valid_interface(); }

        /// logs errors, warnings, infos, ... and returns true in case the was NO error
        bool log_messages(const mi::neuraylib::IMdl_execution_context& context);

        mi::neuraylib::INeuray& get_neuray() { return *m_neuray; }
        mi::neuraylib::IDatabase& get_database() { return *m_database; }
        mi::neuraylib::IMdl_compiler& get_compiler() { return *m_mdl_compiler; }
        mi::neuraylib::IImage_api& get_image_api() { return *m_image_api; }
        mi::neuraylib::IMdl_backend& get_backend() { return *m_hlsl_backend; }
        mi::neuraylib::IMdl_execution_context& get_context() { return *m_context; }

        size_t get_num_texture_results() const { return 1; }

        /// contains the global link unit to be use for all materials that are loaded
        Mdl_target* get_global_target() { return m_global_target; }

        /// keeps all materials that are loaded by the application
        Mdl_material_library* get_library() { return m_library; }

        // enable or disable MDL class compilation mode
        bool use_class_compilation;

    private:
        Base_application* m_app;

        mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
        mi::base::Handle<mi::neuraylib::IDatabase> m_database;
        mi::base::Handle<mi::neuraylib::IMdl_compiler> m_mdl_compiler;
        mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;
        mi::base::Handle<mi::neuraylib::IMdl_backend> m_hlsl_backend;
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;

        Mdl_target* m_global_target;
        Mdl_material_library* m_library;
    };

    // --------------------------------------------------------------------------------------------

    /// keeps all materials that are loaded by the application
    class Mdl_material_library
    {
        friend Mdl_sdk::Mdl_sdk(Base_application*);
        explicit Mdl_material_library(Base_application* app, Mdl_sdk* sdk);

    public:
        // creates a new material or reuses an existing one
        Mdl_material* create(
            const IScene_loader::Material& material_desc,
            const mi::neuraylib::IExpression_list* parameters);

        virtual ~Mdl_material_library();

    private:
        Base_application* m_app;
        Mdl_sdk* m_sdk;

        // map stores the materials with a unique body and temporaries, to be reused materials
        // that differ only in their parameters
        std::map<mi::base::Uuid, Mdl_material_shared*> m_map_shared;
    };

    // --------------------------------------------------------------------------------------------

    class Mdl_target
    {
        class Resource_callback
            : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
        {
        public:
            explicit Resource_callback(Mdl_target* target);
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
            Mdl_target* m_target;
        };

    public:
        Mdl_target(Base_application* app, Mdl_sdk* sdk);
        ~Mdl_target();

        mi::neuraylib::ITransaction& get_transaction() { return *m_transaction; }
        mi::neuraylib::ILink_unit& get_link_unit() { return *m_link_unit; }
        mi::neuraylib::ITarget_resource_callback& get_resource_callback() { 
            return *m_resource_callback; 
        }

        /// Get the generated target.
        /// Note, this is managed and has to put into a handle.
        const mi::neuraylib::ITarget_code* get_target_code() const;

        void add_on_generated(std::function<bool(Mdl_target*, D3DCommandList*)> callback);
       
        bool generate();

        const std::string& get_hlsl_source_code() const { 
            return m_hlsl_source_code; 
        }

        /// all per target resources can be access in this region of the descriptor heap
        D3D12_GPU_DESCRIPTOR_HANDLE get_descriptor_heap_region() const { 
            return m_first_descriptor_heap_gpu_handle; 
        }

        /// get a descriptor table that describes the resource layout in the target heap region
        const Descriptor_table& get_descriptor_table() const { 
            return m_resource_descriptor_table; 
        }


        friend mi::Uint32 Resource_callback::get_resource_index(
            mi::neuraylib::IValue_resource const*);
        std::vector<std::string>& get_resource_names() { return m_resource_names; }

    private:
        Base_application* m_app;
        Mdl_sdk* m_sdk;

        mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
        mi::base::Handle<mi::neuraylib::ILink_unit> m_link_unit;
        mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;
        mi::base::Handle<Resource_callback> m_resource_callback;
        
        std::vector<std::function<bool(Mdl_target*, D3DCommandList*)>> m_on_generated_callbacks;
        std::string m_hlsl_source_code;

        D3D12_GPU_DESCRIPTOR_HANDLE m_first_descriptor_heap_gpu_handle;
        Descriptor_table m_resource_descriptor_table;

        Buffer* m_read_only_data_segment;
        std::vector<std::string> m_resource_names;
        std::vector<Texture*> m_textures;
    };

    // --------------------------------------------------------------------------------------------
    
    // parts of the material that can be reused in case the same material definition is used
    // multiple times but with different arguments
    class Mdl_material_shared
    {
        friend class Mdl_material_library;
        friend class Mdl_material;

        struct Constants
        {
            int32_t scattering_function_index;
            int32_t opacity_function_index;
            int32_t emission_function_index;
            int32_t emission_intensity_function_index;
            int32_t thin_walled_function_index;
        };

        explicit Mdl_material_shared();
        virtual ~Mdl_material_shared() = default;

        size_t m_material_shared_id;
        Constants m_constants;

        mi::Size m_argument_layout_index;
    };
    
    // --------------------------------------------------------------------------------------------

    class Mdl_material : public IMaterial
    {
        friend class Mdl_material_library;

    public:

        struct Constants
        {
            Mdl_material_shared::Constants shared;
            int32_t material_id;
            uint32_t material_flags;
        };

        explicit Mdl_material(Base_application* app, const IScene_loader::Material& material_desc);
        virtual ~Mdl_material();

        virtual Flags get_flags() const override { return m_flags;  }
        virtual void set_flags(Flags flag_mask) override { m_flags = flag_mask; }

        /// get a descriptor table that describes the resource layout in the materials heap region
        static const Descriptor_table& get_descriptor_table();

        /// samplers used for mdl texture lookups
        static std::vector<D3D12_STATIC_SAMPLER_DESC> get_sampler_descriptions();

        const std::string& get_name() const override { return m_name; }

        /// all per target resources can be access in this region of the descriptor heap
        D3D12_GPU_DESCRIPTOR_HANDLE get_target_descriptor_heap_region() const override {
            return m_target->get_descriptor_heap_region();
        }

        /// get the GPU handle of to the first resource of this material in the descriptor heap
        D3D12_GPU_DESCRIPTOR_HANDLE get_material_descriptor_heap_region() const override {
            return m_material_descriptor_heap_region;
        }

        /// get the interface to get and set material arguments
        Mdl_material_info* get_info() { return m_info; }

        void update_material_parameters();
       
    private:
        bool on_target_code_generated(
            Mdl_target* target,
            const mi::neuraylib::IMaterial_definition* material_definition,
            const mi::neuraylib::ICompiled_material* compiled_material,
            bool first_instance,
            D3DCommandList* command_list);

        static Descriptor_table s_resource_descriptor_table;

        Base_application* m_app;
        Mdl_material_shared* m_shared;
        const std::string m_name;
        uint32_t m_material_id;
        IMaterial::Flags m_flags;

        Mdl_target* m_target;
        Constant_buffer<Constants> m_constants;
        Descriptor_heap_handle m_constants_cbv;

        Buffer* m_argument_block_data;
        Descriptor_heap_handle m_argument_block_data_srv;

        Mdl_material_info* m_info;

        D3D12_GPU_DESCRIPTOR_HANDLE m_material_descriptor_heap_region;
    };
}

#endif
