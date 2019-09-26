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
    class Mdl_transaction;
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
        bool log_messages(const mi::neuraylib::IMdl_execution_context* context);

        mi::neuraylib::INeuray& get_neuray() { return *m_neuray; }
        mi::neuraylib::IDatabase& get_database() { return *m_database; }
        mi::neuraylib::IMdl_compiler& get_compiler() { return *m_mdl_compiler; }
        mi::neuraylib::IImage_api& get_image_api() { return *m_image_api; }
        mi::neuraylib::IMdl_backend& get_backend() { return *m_hlsl_backend; }

        // Creates a new execution context. At least one per thread is required.
        // This means you can share the context for multiple calls from the same thread.
        // However, sharing is not required. Creating a context for each call is valid too but
        // slightly more expensive.
        // Use a neuray handle to hold the pointer returned by this function.
        mi::neuraylib::IMdl_execution_context* create_context();

        size_t get_num_texture_results() const { return 16; }

        /// access point to the database 
        Mdl_transaction& get_transaction() { return *m_transaction; }

        /// keeps all materials that are loaded by the application
        Mdl_material_library* get_library() { return m_library; }

        /// enable or disable MDL class compilation mode
        bool use_class_compilation;

    private:
        Base_application* m_app;

        mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
        mi::base::Handle<mi::neuraylib::IDatabase> m_database;
        mi::base::Handle<mi::neuraylib::IMdl_compiler> m_mdl_compiler;
        mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;
        mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdl_factory;
        mi::base::Handle<mi::neuraylib::IMdl_backend> m_hlsl_backend;
        
        Mdl_transaction* m_transaction;
        Mdl_material_library* m_library;
    };

    // --------------------------------------------------------------------------------------------

    /// keeps all materials that are loaded by the application
    class Mdl_material_library
    {
        friend Mdl_sdk::Mdl_sdk(Base_application*);
        explicit Mdl_material_library(Base_application* app, Mdl_sdk* sdk, bool share_target_code);

    public:
        // creates a new material or reuses an existing one
        Mdl_material* create(
            const IScene_loader::Material& material_desc,
            const mi::neuraylib::IExpression_list* parameters);

        virtual ~Mdl_material_library();

        /// gets the global link unit to be use for all materials that are loaded
        /// only available if this library was created with `share_target_code` set.
        Mdl_target* get_shared_target_code();

        /// gets a target code (and link unit) when library was created without `share_target_code`.
        Mdl_target* get_target_code(size_t target_code_id);

        /// iterate over all target codes inside a lock.
        ///
        /// \param action   action to run while visiting a target code.
        ///                 if the action returns false, the iteration is aborted.
        ///
        /// \returns        false if the iteration was aborted, true otherwise.
        bool visit_target_codes(std::function<bool(Mdl_target*)> action);

        /// iterate over all materials inside a lock.
        ///
        /// \param action   action to run while visiting a material.
        ///                 if the action returns false, the iteration is aborted.
        ///
        /// \returns        false if the iteration was aborted, true otherwise.
        bool visit_materials(std::function<bool(Mdl_material*)> action);

        /// get access to the texture data by the texture database name and create a resource.
        /// if there resource is loaded already, no loading is required
        Texture* access_texture_resource(std::string db_name, D3DCommandList* command_list);

    private:

        /// depending 'share_target_code' flag, the shared target is returned or a new created one.
        Mdl_target* get_target_code_for_material_creation();

        Base_application* m_app;
        Mdl_sdk* m_sdk;

        bool m_share_target_code;

        // all existing targets, either one shared target or one per (unique) material
        std::map<size_t, Mdl_target*> m_target_codes;
        std::mutex m_target_codes_mtx;

        // map stores the materials with a unique body and temporaries, to be reused materials
        // that differ only in their parameters
        //std::map<mi::base::Uuid, Mdl_material_shared*> m_map_shared;
        std::map<std::string, Mdl_material_shared*> m_map_shared;
        std::mutex m_map_shared_mtx;

        // all texture resources used by MDL materials
        // TODO: use ref counting when considering dynamic loading and unloading of materials
        std::map<std::string, Texture*> m_textures;
        std::mutex m_textures_mtx;

        // all existing materials (owned by the scene), added here for easier DXR pipeline setup  
        std::vector<Mdl_material*> m_materials;
        std::mutex m_materials_mtx;
    };

    // --------------------------------------------------------------------------------------------

    class Mdl_transaction
    {
        // make sure there is only one transaction
        friend class Mdl_sdk;

        explicit Mdl_transaction(Mdl_sdk* sdk);
    public:
        virtual ~Mdl_transaction();

        // runs an operation on the database.
        // concurrent calls are executed in sequence using a lock.
        template<typename TRes>
        TRes execute(std::function<TRes(mi::neuraylib::ITransaction* t)> action)
        {
            std::lock_guard<std::mutex> lock(m_transaction_mtx);
            return action(m_transaction.get());
        }

        template<>
        void execute<void>(std::function<void(mi::neuraylib::ITransaction* t)> action)
        {
            std::lock_guard<std::mutex> lock(m_transaction_mtx);
            action(m_transaction.get());
        }

        // locked database access function
        template<typename TIInterface>
        const TIInterface* access(const char* db_name)
        {
            return execute<const TIInterface*>(
                [&](mi::neuraylib::ITransaction* t) {
                    return t->access<TIInterface>(db_name);
                });
        }

        // locked database create function
        template<typename TIInterface>
        TIInterface* create(const char* type_name)
        {
            return execute<TIInterface*>(
                [&](mi::neuraylib::ITransaction* t) {
                    return t->create<TIInterface>(type_name);
                });
        }

        // locked database commit function.
        // For that, all handles to neuray objects have to be released.
        // Initializes for further actions afterwards.
        void commit();

    private:
        mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
        std::mutex m_transaction_mtx;
        Mdl_sdk* m_sdk;
    };

    // --------------------------------------------------------------------------------------------

    class Mdl_target
    {
    public:
        class Resource_callback
            : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
        {
        public:
            explicit Resource_callback(Mdl_sdk* sdk, Mdl_target* target, Mdl_material* material);
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
            Mdl_target* m_target;
            Mdl_material* m_material;
        };


        Mdl_target(Base_application* app, Mdl_sdk* sdk);
        ~Mdl_target();


        mi::neuraylib::ILink_unit& get_link_unit() { return *m_link_unit; }
        Mdl_target::Resource_callback get_resource_callback(Mdl_material* material) {
            return std::move(Resource_callback(m_sdk, this, material));
        }

        /// unique id of this target code.
        size_t get_id() const { return m_id; }

        /// Get the generated target.
        /// Note, this is managed and has to put into a handle.
        const mi::neuraylib::ITarget_code* get_target_code() const;


        /// generate HLSL code for all functions in the link unit.
        bool generate();

        /// get the result of the generation step.
        const std::string& get_hlsl_source_code() const { return m_hlsl_source_code; }

        /// compiles the generated HLSL code into a DXIL library.
        bool compile();

        /// get the result of the compiling step.
        const IDxcBlob* get_dxil_compiled_library() const { return m_dxil_compiled_library; }

        /// all per target resources can be access in this region of the descriptor heap
        D3D12_GPU_DESCRIPTOR_HANDLE get_descriptor_heap_region() const { 
            return m_first_descriptor_heap_gpu_handle; 
        }

        /// get a descriptor table that describes the resource layout in the target heap region
        const Descriptor_table& get_descriptor_table() const { 
            return m_resource_descriptor_table; 
        }

        /// in case the material is reused with a different set of parameters, there have to
        /// be texture slots for all possible textures in the generated code
        size_t update_min_texture_count(size_t count);

        friend mi::Uint32 Resource_callback::get_resource_index(
            mi::neuraylib::IValue_resource const*);
        std::vector<std::string>& get_resource_names() { return m_resource_names; }

    private:

        friend Mdl_material* Mdl_material_library::create(
            const IScene_loader::Material&,
            const mi::neuraylib::IExpression_list*);

        // callback registered in 'Mdl_material_library::create'
        void add_on_generated(std::function<bool(D3DCommandList*)> callback);
       
        Base_application* m_app;
        Mdl_sdk* m_sdk;
        const size_t m_id;

        mi::base::Handle<mi::neuraylib::ILink_unit> m_link_unit;
        mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;
        mi::base::Handle<Resource_callback> m_resource_callback;
        
        std::vector<std::function<bool(D3DCommandList*)>> m_on_generated_callbacks;
        std::string m_hlsl_source_code;
        IDxcBlob* m_dxil_compiled_library;

        D3D12_GPU_DESCRIPTOR_HANDLE m_first_descriptor_heap_gpu_handle;
        Descriptor_table m_resource_descriptor_table;

        Buffer* m_read_only_data_segment;
        size_t m_min_texture_count;
        std::vector<std::string> m_resource_names;

        std::atomic_bool m_finalized;
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
        std::string m_material_db_name; // to check for hash collisions (reduces changes at least..)
        Constants m_constants;

        mi::Size m_argument_layout_index;

        Mdl_target* m_target;
        size_t m_max_texture_count;
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

        uint32_t get_id() const { return m_material_id; }
        const std::string& get_name() const override { return m_name; }

        virtual Flags get_flags() const override { return m_flags;  }
        virtual void set_flags(Flags flag_mask) override { m_flags = flag_mask; }

        /// get a descriptor table that describes the resource layout in the materials heap region.
        /// Note, static table can be used only when shared target is used.
        static const Descriptor_table& get_static_descriptor_table();

        /// get a descriptor table that describes the resource layout in the materials heap region.
        /// Note, non-static table has to be used when a target is created for each material.
        const Descriptor_table get_descriptor_table();

        /// samplers used for mdl texture lookups
        static std::vector<D3D12_STATIC_SAMPLER_DESC> get_sampler_descriptions();

        /// get the id of the target code that contains this material. 
        /// can be used with the material library for instance.
        size_t get_target_code_id() const { return m_shared->m_target->get_id(); }

        /// get the target code that contains this material. 
        const Mdl_target* get_target_code() const { return m_shared->m_target; }

        /// get the GPU handle of to the first resource of this material in the descriptor heap
        D3D12_GPU_DESCRIPTOR_HANDLE get_descriptor_heap_region() const override {
            return m_descriptor_heap_region;
        }

        /// get the interface to get and set material arguments
        Mdl_material_info* get_info() { return m_info; }
        /// after material parameters have been changed, they have to copied to the GPU
        void update_material_parameters();

    private:
        bool on_target_code_generated(
            const mi::neuraylib::IMaterial_definition* material_definition,
            const mi::neuraylib::ICompiled_material* compiled_material,
            bool first_instance,
            D3DCommandList* command_list);

        Base_application* m_app;
        Mdl_material_shared* m_shared;
        const std::string m_name;
        uint32_t m_material_id;
        IMaterial::Flags m_flags;

        /// constant buffer for function indices, material id and flags
        Constant_buffer<Constants> m_constants;

        /// buffer that contains the material parameters, changing them during runtime is possible
        Buffer* m_argument_block_data;
        /// helper structure that allows to access the 'm_argument_block_data' fields
        Mdl_material_info* m_info;

        /// all material resources have views on the descriptor heap (all in one continuous block)
        Descriptor_heap_handle m_first_descriptor;              // first descriptor on the heap
        D3D12_GPU_DESCRIPTOR_HANDLE m_descriptor_heap_region;   // GPU address of continuous block 

        // TODO, refactor that
        friend mi::Uint32 Mdl_target::Resource_callback::get_resource_index(
            mi::neuraylib::IValue_resource const*);
        std::vector<std::string>& get_resource_names() { return m_resource_names; }
        std::vector<std::string> m_resource_names;
    };
}

#endif
