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
#include "mdl_material_library.h"
#include "mdl_material_target.h"
#include "buffer.h"
#include "scene.h"
#include "shader.h"
#include <mi/mdl_sdk.h>


namespace mdl_d3d12
{
    class Base_application;
    class Descriptor_heap;
    class Material;
    class Mdl_transaction;
    class Mdl_material_info;
    class Mdl_material_description;
    class Mdl_material_target;
    class Mdl_sdk;
    enum class Mdl_resource_kind;

    class Mdl_material : public IMaterial
    {
        //friend class Mdl_material_library;

    public:
        struct Constants
        {
            Mdl_material_function_indices function_indices;
            int32_t material_id;
            uint32_t material_flags;
        };

        explicit Mdl_material(Base_application* app, const std::string& name);
        virtual ~Mdl_material();

        uint32_t get_id() const { return m_material_id; }
        const std::string& get_name() const override { return m_name; }
        Flags get_flags() const override { return m_flags;  }

        /// use a material definition and a set of parameters to create an MDL compiled material
        /// that can be added to an Mdl_material_target.
        bool compile_material(
            const std::string& material_defintion_db_name,
            const mi::neuraylib::IExpression_list* parameter_list,
            mi::neuraylib::IMdl_execution_context* context);

        /// create an MDL compiled material from the existing instance by reusing the existing
        /// definition and parameter list. This can be used after reloading modules.
        bool recompile_material(mi::neuraylib::IMdl_execution_context* context);

        const std::string& get_material_definition_db_name() const { return m_defintion_db_name; }
        const std::string& get_material_instance_db_name() const { return m_instance_db_name; }
        const std::string& get_material_compiled_db_name() const { return m_compiled_db_name; }
        const std::string& get_material_compiled_hash() const { return m_compiled_hash; }

        /// iterate over all qualified module names this material depends on.
        bool visit_module_dependencies(std::function<bool(const std::string&)> action);

        /// check if this material depends on a certain module.
        bool depends(const std::string& module_db_name) const;

        /// set the per target code constants in the materials constant buffer.
        void set_target_interface(
            Mdl_material_target* target,
            const Mdl_material_target_interface& target_data);

        /// called by the Mdl_material_target after the target code was generated in order to
        /// allocate and update per material buffers for constants and textures.
        bool on_target_generated(D3DCommandList* command_list);

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
        size_t get_target_code_id() const override;

        /// get the target code that contains this material. 
        Mdl_material_target* get_target_code() const { return m_target; }

        /// get the GPU handle of to the first resource of this target in the descriptor heap.
        D3D12_GPU_DESCRIPTOR_HANDLE get_target_descriptor_heap_region() const override {
            return m_target->get_descriptor_heap_region();
        }

        /// get the GPU handle of to the first resource of this material in the descriptor heap.
        D3D12_GPU_DESCRIPTOR_HANDLE get_material_descriptor_heap_region() const override {
            return m_first_resource_heap_handle.get_gpu_handle();
        }

        /// get the interface to get and set material arguments.
        Mdl_material_info* get_info() { return m_info; }
        /// after material parameters have been changed, they have to copied to the GPU.
        void update_material_parameters();

        // Called by the target to register per material resources.
        size_t register_resource(
            Mdl_resource_kind kind, 
            const std::string& resource_name);

    private:
        Base_application* m_app;
        Mdl_sdk* m_sdk;
        uint32_t m_material_id;
        const std::string m_name;
        IMaterial::Flags m_flags;

        /// database name of the material definition for this material.
        std::string m_defintion_db_name;

        /// database name of the material instance of this material.
        std::string m_instance_db_name;

        /// database name of the compiled material of this material.
        std::string m_compiled_db_name;

        /// combination of material db name and the compiled material hash to detect
        /// materials that only differ in their parameters values.
        std::string m_compiled_hash;

        /// set of qualified module names that this material depends on.
        std::deque<std::string> m_module_dependencies;

        /// contains the actual shader code for material.
        Mdl_material_target* m_target;

        /// constant buffer for function indices, material id and flags.
        Constant_buffer<Constants> m_constants;

        /// buffer that contains the material parameters, changing them during runtime is possible.
        Buffer* m_argument_block_data;

        /// index target code that describes the layout of the argument block data.
        mi::Size m_argument_layout_index;

        /// helper structure that allows to access the 'm_argument_block_data' fields.
        Mdl_material_info* m_info;

        /// all material resources have views on the descriptor heap (all in one continuous block)
        Descriptor_heap_handle m_first_resource_heap_handle;    // first descriptor on the heap

        std::map<Mdl_resource_kind, std::vector<std::string>> m_resource_names;

    };
}

#endif
