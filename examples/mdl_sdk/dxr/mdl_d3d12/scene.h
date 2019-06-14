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

// examples/mdl_sdk/dxr/mdl_d3d12/scene.h

#ifndef MDL_D3D12_SCENE_H
#define MDL_D3D12_SCENE_H

#include "common.h"
#include "raytracing_pipeline.h"

namespace mdl_d3d12
{
    class Base_application;
    class Buffer;
    class Constant_buffer_base;
    class IMaterial;
    struct Update_args;

    template<typename T> class Constant_buffer;

    struct Vertex
    {
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 normal;
        DirectX::XMFLOAT2 texcoord0;
        DirectX::XMFLOAT4 tangent0;
    };

    struct Transform
    {
        explicit Transform();

        bool is_identity() const;
        static bool try_from_matrix(const DirectX::XMMATRIX& matrix, Transform& out_transform);

        DirectX::XMFLOAT3 translation;
        DirectX::XMVECTOR rotation;
        DirectX::XMFLOAT3 scale;

        DirectX::XMMATRIX get_matrix();

        static const Transform identity;
    };

    // --------------------------------------------------------------------------------------------

    class IScene_loader
    {
    public:
        class Node
        {
        public:
            // Flags to be used as filter
            enum class Kind
            {
                Empty = 0,
                Camera = 1,
                Mesh = 2,
                // Next = 4,
            };

            Kind kind;
            std::string name;
            size_t index;
            Transform local;
            std::vector<Node> children;
        };

        class Primitive
        {
        public:
            size_t vertex_offset;
            size_t vertex_count;
            size_t index_offset;
            size_t index_count;
            size_t material;
        };

        class Mesh
        {
        public:
            std::string name;
            std::vector<Primitive> primitives;
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
        };

        class Camera
        {
        public:
            std::string name;
            float vertical_fov;
            float aspect_ratio;
            float near_plane_distance;
            float far_plane_distance;
        };


        class Material
        {
        public:

            enum class Alpha_mode
            {
                Opaque = 0, // alpha is ignored, 1.0 is used instead
                Mask,       // opaque if alpha (base_color.w) is >= alpha_cutoff, 0.0 otherwise 
                Blend       // blending based on alpha (base_color.w)
            };

            enum class Pbr_model
            {
                Metallic_roughness = 0,
                Khr_specular_glossiness
            };

            struct Pbr_model_data_metallic_roughness
            {
                std::string base_color_texture;
                DirectX::XMFLOAT4 base_color_factor;
                std::string metallic_roughness_texture;
                float metallic_factor;
                float roughness_factor;
            };

            struct Pbr_model_data_khr_specular_glossiness
            {
                std::string diffuse_texture;
                DirectX::XMFLOAT4 diffuse_factor;

                std::string specular_glossiness_texture;
                DirectX::XMFLOAT3 specular_factor;
                float glossiness_factor;
            };

            std::string name;

            std::string normal_texture;
            float normal_scale_factor;

            std::string occlusion_texture;
            float occlusion_strength;

            std::string emissive_texture;
            DirectX::XMFLOAT3 emissive_factor;

            Alpha_mode alpha_mode;
            float alpha_cutoff;
            bool single_sided;

            // depending on the material model different sub classes 
            Pbr_model pbr_model;
            Pbr_model_data_metallic_roughness metallic_roughness;
            Pbr_model_data_khr_specular_glossiness khr_specular_glossiness;

        };


        class Scene
        {
        public:
            std::vector<Mesh> meshes;
            std::vector<Camera> cameras;
            std::vector<Material> materials;
            Node root;
        };

        virtual bool load(const std::string& file_name) = 0;
        virtual const Scene& get_scene() const = 0;
    };

    // --------------------------------------------------------------------------------------------

    class Mesh
    {
        friend class Scene;

    public:
        struct Geometry
        {
            friend class Mesh;
            friend class Scene;
            explicit Geometry();

            const Raytracing_acceleration_structure::Geometry_handle& get_geometry() const { 
                return m_geometry; 
            }

            const IMaterial* get_material() const { return m_material; }
            uint32_t get_index_offset() const { return m_index_offset; }

        private:
            IMaterial* m_material;
            Raytracing_acceleration_structure::Geometry_handle m_geometry;
            uint32_t m_index_offset;
        };

        explicit Mesh(
            Base_application* app, 
            Raytracing_acceleration_structure* acceleration_structure, 
            const IScene_loader::Mesh& mesh_desc);
        virtual ~Mesh();

        const Raytracing_acceleration_structure::Instance_handle create_instance();
        bool upload_buffers(D3DCommandList* command_list);

        const std::vector<Geometry>& get_geometries() const { return m_geometries; }
        const Vertex_buffer<Vertex>* get_vertex_buffer() const { return m_vertex_buffer; }
        const Index_buffer* get_index_buffer() const { return m_index_buffer; }
        
    private:
        Base_application* m_app;
        
        std::string m_name;
        Vertex_buffer<Vertex>* m_vertex_buffer;
        Index_buffer* m_index_buffer;

        Raytracing_acceleration_structure* m_acceleration_structur;
        Raytracing_acceleration_structure::BLAS_handle m_blas;

        std::vector<Geometry> m_geometries;
    };

    // --------------------------------------------------------------------------------------------

    class Camera
    {
        friend class Scene;
        friend class Scene_node;
    public:

        struct Constants
        {
            DirectX::XMMATRIX view;
            DirectX::XMMATRIX perspective;
            DirectX::XMMATRIX view_inv;
            DirectX::XMMATRIX perspective_inv;
        };

        explicit Camera(
            Base_application* app,
            const IScene_loader::Camera& camera_desc);

        virtual ~Camera();

        const std::string& get_name() const { return m_name; }
        const Constant_buffer<Camera::Constants>* get_constants() const { return m_constants; }

        float get_field_of_view () const { return m_field_of_view; }
        void set_field_of_view(float vertical_fov) {
            m_field_of_view = vertical_fov, m_projection_changed = true;
        }

        float get_aspect_ratio() const { return m_aspect_ratio; }
        void set_aspect_ratio(float aspect) {
            m_aspect_ratio = aspect, m_projection_changed = true; 
        }

    private:
        void update(const DirectX::XMMATRIX& global_transform, bool transform_changed);

        Base_application* m_app;
        std::string m_name;

        float m_field_of_view;
        float m_aspect_ratio;
        float m_near_plane_distance;
        float m_far_plane_distance;
        bool m_projection_changed;
        
        Constant_buffer<Camera::Constants>* m_constants;
    };

    // --------------------------------------------------------------------------------------------

    class Scene_node
    {
        friend class Scene;
        
    public:
        typedef IScene_loader::Node::Kind Kind;

        explicit Scene_node(
            Base_application* app, 
            Scene* scene, 
            Kind kind, 
            const std::string& name);
        virtual ~Scene_node();

        Kind get_kind() const { return m_kind; }
        const std::string& get_name() const { return m_name; }

        const Mesh* get_mesh() const { return m_kind == Kind::Mesh ? m_mesh : nullptr; }
        const Raytracing_acceleration_structure::Instance_handle& get_mesh_instance() const { 
            return m_mesh_instance; 
        }

        Camera* get_camera() { return m_kind == Kind::Camera ? m_camera : nullptr; }
        const Camera* get_camera() const { return m_kind == Kind::Camera ? m_camera : nullptr; }

        // get the local transformation of this node relative to its parent.
        Transform& get_local_transformation() { return m_local_transformation; }
        bool transformed_on_last_update() const { return m_transformed_on_last_update; }

        // set the entire transform at once
        void set_local_transformation(const Transform& transform) { 
            m_local_transformation = transform; 
        }

        // get the world transformation of this node (read-only).
        const DirectX::XMMATRIX& get_global_transformation() { return m_global_transformation; }

        void add_child(Scene_node* to_add);

        void update(const Update_args& args);

    private:
        Base_application* m_app;

        Kind m_kind;
        std::string m_name;
        Scene* m_scene;
        Scene_node* m_parent;
        Transform m_local_transformation;
        DirectX::XMMATRIX m_global_transformation;
        bool m_transformed_on_last_update;
        std::vector<Scene_node*> m_children;
            
        Mesh* m_mesh;
        Raytracing_acceleration_structure::Instance_handle m_mesh_instance;

        Camera* m_camera;
    };

    // --------------------------------------------------------------------------------------------

    class IMaterial
    {
    public:
        enum class Flags
        {
            None            = 0,
            Opaque          = 1 << 0, // allows to skip opacity evaluation
            SingleSided     = 1 << 1  // geometry is only visible from the front side
        };

        virtual ~IMaterial() = default;
        virtual const std::string& get_name() const = 0;

        /// all per target resources can be access in this region of the descriptor heap
        virtual D3D12_GPU_DESCRIPTOR_HANDLE get_target_descriptor_heap_region() const = 0;

        /// get the GPU handle of to the first resource of this material in the descriptor heap
        virtual D3D12_GPU_DESCRIPTOR_HANDLE get_material_descriptor_heap_region() const = 0;

        // get material flags e.g. for optimization
        virtual Flags get_flags() const = 0;

        // set material flags e.g. for optimization
        virtual void set_flags(Flags flag_mask) = 0;
    };

    // --------------------------------------------------------------------------------------------

    class Scene
    {
    public:
        explicit Scene(
            Base_application* app, 
            const std::string& debug_name, 
            size_t ray_type_count);
        virtual ~Scene();

        bool build_scene(const IScene_loader::Scene& scene);

        Raytracing_acceleration_structure* get_acceleration_structure() const { 
            return m_acceleration_structure; 
        }

        /// iterate recursively over all nodes that match the Scene_node::Kind mask.
        ///
        /// \param mask     selected kind of nodes to visit
        /// \param action   action to run while visiting a selected node.
        ///                 if the action returns false, the traversal is aborted.
        ///
        /// \returns        false if the traversal was aborted, true otherwise.
        bool traverse(
            Scene_node::Kind mask,
            std::function<bool(const Scene_node* node)> action) const;

        /// iterate recursively over all nodes that match the Scene_node::Kind mask.
        ///
        /// \param mask     selected kind of nodes to visit
        /// \param action   action to run while visiting a selected node.
        ///                 if the action returns false, the traversal is aborted.
        ///
        /// \returns        false if the traversal was aborted, true otherwise.
        bool traverse(
            Scene_node::Kind mask,
            std::function<bool(Scene_node* node)> action);

        /// create a new camera and add it to the scene.
        Scene_node* create(
            const IScene_loader::Camera& camera_description,
            const Transform& local_transform = Transform::identity,
            Scene_node* parent = nullptr);

        void update(const Update_args& args) { m_root.update(args); }

        size_t get_material_count() const { return m_materials.size(); }
        IMaterial* get_material(size_t index);

    private:
        Base_application* m_app;
        const std::string& m_debug_name;

        std::vector<Camera*> m_cameras;
        std::vector<Mesh*> m_meshes;
        std::vector<IMaterial*> m_materials;

        Scene_node m_root;
        Raytracing_acceleration_structure* m_acceleration_structure;
    };
}

#endif
