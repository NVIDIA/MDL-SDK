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

// examples/mdl_sdk/dxr/mdl_d3d12/raytracing_pipeline.h

#ifndef MDL_D3D12_RAYTRACING_PIPELINE_H
#define MDL_D3D12_RAYTRACING_PIPELINE_H

#include "common.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Root_signature;
    class Shader_binding_tables;
    class Buffer;
    template<typename TVertex>
    class Vertex_buffer;
    class Index_buffer;

    // --------------------------------------------------------------------------------------------

    class Raytracing_pipeline
    {
        friend class Shader_binding_tables;

        // --------------------------------------------------------------------

    private:
        /// helper to store added libraries
        struct Library
        {
        public:
            explicit Library(
                const IDxcBlob* dxil_library,
                bool owns_dxil_library,
                const std::vector<std::string>& exported_symbols);

            const IDxcBlob* m_dxil_library;
            bool m_owns_dxil_library;
            std::vector<std::wstring> m_exported_symbols;
            std::vector<D3D12_EXPORT_DESC> m_exports;
            D3D12_DXIL_LIBRARY_DESC m_desc;
        };

        // --------------------------------------------------------------------

        /// helper to store added hit groups
        struct Hitgroup
        {
        public:
            explicit Hitgroup(
                std::string name,
                std::string closest_hit_symbol,
                std::string any_hit_symbol,
                std::string intersection_symbol);

            std::wstring m_name;
            std::wstring m_closest_hit_symbol;
            std::wstring m_any_hit_symbol;
            std::wstring m_intersection_symbol;
            D3D12_HIT_GROUP_DESC m_desc = {};
        };

        // --------------------------------------------------------------------

        /// helper to store the association between programs and their root signature
        struct Root_signature_association
        {
            explicit Root_signature_association(
                Root_signature* signature,
                bool owns_signature,
                const std::vector<std::string>& symbols);

            Root_signature* m_root_signature;
            ID3D12RootSignature* m_signature;
            bool m_owns_root_signature;
            std::vector<std::wstring> m_symbols;
            std::vector<LPCWSTR> m_symbol_pointers;
            D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION m_desc;
        };

        // --------------------------------------------------------------------

    public:
        explicit Raytracing_pipeline(Base_application* app, std::string debug_name);
        virtual ~Raytracing_pipeline();

        /// Add a DXIL library to the pipeline.
        ///
        /// \param dxil_library         A library compiled using a \c Shader_compiler.
        /// \param take_ownership       If true, the pipeline will own the library and delete it
        ///                             when destructed.
        /// \param exported_symbols     Exact names of functions defined in the library sources.
        ///                             Unused ones can be omitted.
        /// \return                     True in case of success.
        bool add_library(
            const IDxcBlob* dxil_library,
            bool take_ownership,
            const std::vector<std::string>& exported_symbols);

        /// Add a hit group to the pipeline.
        ///
        /// \param name                 Name of the hit group to add.
        ///                             Used also in the shader binding table.
        /// \param closest_hit_symbol   Program to be executed for the closest hit (has to be set).
        /// \param any_hit_symbol       Program to be executed for any hit
        ///                             (can be empty for default behavior).
        /// \param intersection_symbol  Intersection program (can be empty for triangle meshes).
        /// \return                     True in case of success.
        bool add_hitgroup(
            std::string name,
            std::string closest_hit_symbol,
            std::string any_hit_symbol,
            std::string intersection_symbol);

        /// Associates a symbol or a hit group with a shader root signature.
        ///
        /// \param signature            The signature to associate
        /// \param owns_signature       If true, the pipeline will own the signature and delete it
        ///                             when destructed.
        /// \param symbols              Symbol or hit group names to associate with the signature.
        /// \return                     True in case of success.
        bool add_signature_association(
            Root_signature* signature,
            bool owns_signature,
            const std::vector<std::string>& symbols);

        /// Complete the setup of the object in order to be used for rendering.
        ///
        /// Afterwards, no changes to the object are allowed anymore.
        bool finalize();

        /// Set the payload size which defines the maximum size of the data carried by the rays.
        ///
        /// Keep this value as low as possible.
        void set_max_payload_size(size_t size_in_byte) {
            m_max_payload_size_in_byte = size_in_byte;
        }

        /// Size of the data that is provided for each hit.
        ///
        /// Usually, this contains barycentric coordinates. 2 float values, as all 3 sum up to 1.0f.
        void set_max_attribute_size(size_t size_in_byte) {
            m_max_attribute_size_in_byte = size_in_byte;
        }

        /// The ray tracing process can shoot rays from existing hit points
        /// and this sets the number of allowed indirections.
        void set_max_recursion_depth(size_t depth) { m_max_recursion_depth = depth; }

        /// Ray tracing pipeline state properties,
        /// retaining the shader identifiers to use in the Shader Binding Table
        ID3D12StateObjectProperties* get_state_properties() {
            return m_pipeline_state_properties.Get();
        }

        /// Get the state that has to bound using the command list before casting rays.
        ID3D12StateObject* get_state() { return m_pipeline_state.Get(); }

        Root_signature* get_global_root_signature() { return m_global_root_signature; }

    private:
        Base_application* m_app;
        std::string m_debug_name;
        bool m_is_finalized;
        size_t m_max_payload_size_in_byte;
        size_t m_max_attribute_size_in_byte;
        size_t m_max_recursion_depth;

        std::vector<Library> m_libraries;
        std::vector<Hitgroup> m_hitgroups;
        std::vector<Root_signature_association> m_signature_associations;
        std::unordered_map<std::string, Root_signature*> m_signature_map;

        std::unordered_set<std::wstring> m_all_exported_symbols;
        std::unordered_set<std::wstring> m_all_associated_symbols;

        Root_signature* m_dummy_local_root_signature;
        Root_signature* m_global_root_signature;

        ComPtr<ID3D12StateObject> m_pipeline_state;
        ComPtr<ID3D12StateObjectProperties> m_pipeline_state_properties;
    };

    // --------------------------------------------------------------------------------------------

    class Raytracing_acceleration_structure : public Resource
    {
        // --------------------------------------------------------------------

        struct Bottom_level
        {
            explicit Bottom_level(std::string debug_name_suffix);
            virtual ~Bottom_level();

            const std::string m_debug_name_suffix;
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_geometry_descriptions;
            ComPtr<ID3D12Resource> m_blas_resource;
            ComPtr<ID3D12Resource> m_scratch_resource;
        };

        // --------------------------------------------------------------------

    public:
        struct BLAS_handle
        {
            friend class Raytracing_acceleration_structure;

            explicit BLAS_handle();
            virtual ~BLAS_handle() = default;
            bool is_valid() const { return m_acceleration_structure != nullptr; }

        private:
            explicit BLAS_handle(
                Raytracing_acceleration_structure* acceleration_structure,
                size_t index);

            Raytracing_acceleration_structure* m_acceleration_structure;
            size_t m_index;
        };

        // --------------------------------------------------------------------

        struct Geometry_handle
        {
            friend class Raytracing_acceleration_structure;

            explicit Geometry_handle();
            virtual ~Geometry_handle() = default;
            bool is_valid() const { return m_acceleration_structure != nullptr; }
        private:
            explicit Geometry_handle(
                Raytracing_acceleration_structure* acceleration_structure,
                size_t blas_index,
                size_t geometry_index);

            Raytracing_acceleration_structure* m_acceleration_structure;
            size_t m_blas_index;
            size_t m_geometry_index;
        };

        // --------------------------------------------------------------------

        struct Instance_handle
        {
            friend class Raytracing_acceleration_structure;

            explicit Instance_handle();
            virtual ~Instance_handle() = default;
            bool is_valid() const { return m_acceleration_structure != nullptr; }

        private:
            explicit Instance_handle(
                Raytracing_acceleration_structure* acceleration_structure,
                size_t blas_index,
                size_t instance_index,
                size_t instance_id);

            Raytracing_acceleration_structure* m_acceleration_structure;
            size_t m_blas_index;
            size_t m_instance_index;
        public:
            size_t instance_id;
        };

        // --------------------------------------------------------------------

        /// Constructor.
        explicit Raytracing_acceleration_structure(
            Base_application* app,
            size_t ray_type_count,
            std::string debug_name);

        /// Destructor.
        virtual ~Raytracing_acceleration_structure();

        std::string get_debug_name() const override { return m_debug_name; }

        /// Create a new bottom level acceleration structure,
        /// to which multiple geometries can be added.
        ///
        /// \param debug_name_suffix    Debug name that is append to the debug name of
        ///                             the acceleration structure.
        /// \return                     A handle to add geometries and to create instances,
        ///                             or an invalid handle in case of failure.
        const BLAS_handle add_bottom_level_structure(const std::string& debug_name_suffix);

        /// Add geometry to a bottom level structure.
        /// This can be mesh parts with different materials.
        ///
        /// \param vertex_buffer                // vertex buffer of the mesh (contains all parts)
        /// \param vertex_buffer_byte_offset    // base address of the first vertex in the buffer
        /// \param vertex_count                 // number of vertices to add
        /// \param vertex_stride_in_byte        // size of a vertex in bytes
        /// \param vertex_position_byte_offset  // offset to the position semantic within a vertex
        /// \param index_buffer                 // index data for the entire mesh
        /// \param index_offset                 // offset in the index_buffer for the part to add
        /// \param index_count                  // number of indices to add (triangle_count x 3)
        const Geometry_handle add_geometry(
            const BLAS_handle& blas,
            Buffer* vertex_buffer,
            size_t vertex_buffer_byte_offset,
            size_t vertex_count,
            size_t vertex_stride_in_byte,
            size_t vertex_position_byte_offset,
            Index_buffer* index_buffer,
            size_t index_offset,
            size_t index_count);

        /// Create an instance of a BLAS with all its geometries.
        const Instance_handle add_instance(
            const BLAS_handle& blas,
            const DirectX::XMMATRIX& transform,
            UINT instance_mask = 0xFF,
            UINT flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE,
            size_t instance_id = 0);

        bool set_instance_transform(
            const Instance_handle& instance_handle,
            const DirectX::XMMATRIX& transform);

        /// Constructs the acceleration data structure.
        bool build(D3DCommandList* command_list);

        /// After executing the command list that created to the acceleration structure,
        /// temporary data can be deleted. This will free all temporary buffers that are
        /// not required for potential dynamic updates.
        void release_static_scratch_buffers();

        /// Computes the record index for one specific ray type - instance - geometry combination.
        /// This allows to specify a material for each instance on geometry level.
        /// Note, This requires the TraceRay() calls to set
        /// MultiplierForGeometryContributionToHitGroupIndex to ray_type_count.
        ///
        /// The record index is computes as follows
        ///     geometry_offset = geometry_index_in_BLAS * ray_type_count
        ///     instance_offset_i = sum over k from 0 to i-1 : geometry_count_in_BLAS_i
        ///     hit_record_offset = instance_offset_i + geometry_offset + ray_type
        size_t compute_hit_record_index(
            size_t ray_type,
            const Instance_handle& instance_handle,
            const Geometry_handle& geometry_handle);

        /// Get the maximum number of hit records
        /// based on the number of ray types, the number of instances and their individual number
        /// of geometries (see also 'compute_hit_record_index')
        size_t get_hit_record_count() const;

        size_t get_ray_type_count() const { return m_ray_type_count;  }

        ID3D12Resource* get_resource() { return m_top_level_structure.Get(); }
        bool get_shader_resource_view_description(D3D12_SHADER_RESOURCE_VIEW_DESC& desc) const;

    private:
        Base_application* m_app;
        std::string m_debug_name;

        size_t m_ray_type_count;
        std::vector<Bottom_level> m_bottom_level_structures;
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> m_instances;
        std::vector<size_t> m_instance_blas_indices;
        std::vector<size_t> m_instance_contribution_to_hit_record_index;

        // has to match MultiplierForGeometryContributionToHitGroupIndex in TraceRay()-calls
        size_t m_geometry_contribution_multiplier_to_hit_record_index;
        ComPtr<ID3D12Resource> m_instance_buffer;
        ComPtr<ID3D12Resource> m_top_level_structure;
        ComPtr<ID3D12Resource> m_scratch_resource;

        bool build_bottom_level_structure(D3DCommandList* command_list, size_t blas_index);
        bool build_top_level_structure(D3DCommandList* command_list);

        bool allocate_resource(
            ID3D12Resource** resource,
            UINT64 size_in_byte,
            D3D12_RESOURCE_STATES initial_state,
            const std::string& debug_name_suffix);
    };

}}} // mi::examples::mdl_d3d12
#endif
