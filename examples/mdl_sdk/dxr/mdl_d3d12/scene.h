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

// examples/mdl_sdk/dxr/mdl_d3d12/scene.h

#ifndef MDL_D3D12_SCENE_H
#define MDL_D3D12_SCENE_H

#include "common.h"
#include "raytracing_pipeline.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Buffer;
    class Constant_buffer_base;
    class IMaterial;
    class Scene;
    struct Update_args;

    template<typename T> class Constant_buffer;
    template<typename T> class Dynamic_constant_buffer;
    template<typename T> class Structured_buffer;

    // --------------------------------------------------------------------------------------------

    struct Vertex
    {
        DirectX::XMFLOAT3 position;
        DirectX::XMFLOAT3 normal;
        DirectX::XMFLOAT2 texcoord0;
        DirectX::XMFLOAT4 tangent0;
    };

    // --------------------------------------------------------------------------------------------

    inline static DirectX::XMVECTOR vector(const DirectX::XMFLOAT3& vec3, float w)
    {

        DirectX::XMVECTOR res = DirectX::XMLoadFloat3(&vec3);
        res.m128_f32[3] = w;
        return std::move(res);
    }

    // --------------------------------------------------------------------------------------------

    struct Transform
    {
        explicit Transform();

        bool is_identity() const;
        static bool try_from_matrix(const DirectX::XMMATRIX& matrix, Transform& out_transform);
        static Transform look_at(
            const DirectX::XMFLOAT3& camera_pos,
            const DirectX::XMFLOAT3& focus,
            const DirectX::XMFLOAT3& up);

        DirectX::XMFLOAT3 translation;
        DirectX::XMVECTOR rotation;
        DirectX::XMFLOAT3 scale;

        DirectX::XMMATRIX get_matrix() const;

        static const Transform Identity;
    };

    // --------------------------------------------------------------------------------------------

    struct Bounding_box
    {
        static const Bounding_box Invalid;
        static const Bounding_box Zero;

        explicit Bounding_box(const DirectX::XMFLOAT3& min, const DirectX::XMFLOAT3& max);
        explicit Bounding_box();

        void merge(const Bounding_box& second);
        static Bounding_box merge(const Bounding_box& value1, const Bounding_box& value2);

        void extend(const DirectX::XMFLOAT3& point);
        static Bounding_box extend(const Bounding_box& box, const DirectX::XMFLOAT3& point);

        static Bounding_box transform(const Bounding_box& box, const Transform& transformation);
        static Bounding_box transform(const Bounding_box& box, const DirectX::XMMATRIX& matrix);

        void get_corners(std::vector<DirectX::XMVECTOR>& outCorners) const;

        DirectX::XMFLOAT3 center() const;
        DirectX::XMFLOAT3 size() const;

        bool is_zero() const;
        bool is_valid() const;

        DirectX::XMFLOAT3 min;
        DirectX::XMFLOAT3 max;
    };

    inline bool operator==(const Bounding_box& lhs, const Bounding_box& rhs)
    {
        if (lhs.min.x != rhs.min.x) return false;
        if (lhs.min.y != rhs.min.y) return false;
        if (lhs.min.z != rhs.min.z) return false;
        if (lhs.max.x != rhs.max.x) return false;
        if (lhs.max.y != rhs.max.y) return false;
        if (lhs.max.z != rhs.max.z) return false;
        return true;
    }

    inline bool operator!=(const Bounding_box& lhs, const Bounding_box& rhs)
    {
        return !(lhs == rhs);
    }

    // --------------------------------------------------------------------------------------------

    class Scene_data
    {
    public:

        // --------------------------------------------------------------------

        /// Scope a scene data element belongs to
        enum class Kind
        {
            None = 0,       // used as invalid or not present
            Vertex = 1,     // data is expected to be found in the vertex buffers
            Instance = 2    // data is to be found in the per mesh instance buffer
        };

        // --------------------------------------------------------------------

        /// Currently supported types of scene data
        enum class Value_kind
        {
            Int,
            Int2,
            Int3,
            Int4,
            Float,
            Vector2,
            Vector3,
            Vector4,
            Color
        };

        // --------------------------------------------------------------------

        /// Basic element type of the scene data
        enum class Element_type
        {
            Float = 0,
            Int = 1,
            Color = 2
        };

        // --------------------------------------------------------------------

        /// interpolation of the data over the primitive
        enum class Interpolation_mode
        {
            None = 0,       // default for instance / object data
            Linear = 1,     // default for floating point vertex data
            Nearest = 2,    // default for integer vertex data
        };

        // --------------------------------------------------------------------

        /// Describes the layout of a certain scene data element in the scene data buffer.
        /// There are several use cases to consider:
        ///
        /// - Vertex data in an interleaved or non-interleaved vertex buffer
        ///   The address of the data to fetch computes as:
        ///   base_address + vertex_id * byte_stride + byte_offset
        ///
        /// - Global scene data (not implemented yet)
        ///
        /// - Per object data (not implemented yet)
        struct Info
        {
            explicit Info();

            /// Scope a scene data element belongs to (4 bits)
            Kind get_kind() const;
            void set_kind(Kind value);

            /// Basic element type of the scene data (4 bits)
            Element_type get_element_type() const;
            void set_element_type(Element_type value);

            /// Interpolation of the data over the primitive (4 bits)
            Interpolation_mode get_interpolation_mode() const;
            void set_interpolation_mode(Interpolation_mode value);

            /// Indicates whether there the scene data is uniform. (1 bit)
            bool get_uniform() const;
            void set_uniform(bool value);

            /// Offset between two elements. For interleaved vertex buffers, this is the vertex size
            /// in byte. For non-interleaved buffers, this is the element size in byte. (16 bit)
            uint16_t get_byte_stride() const;
            void set_byte_stride(uint16_t value);

            /// The offset to the data element within an interleaved vertex buffer, or the absolute
            /// offset to the base (e.g. of the geometry data) in non-interleaved buffers
            uint32_t get_byte_offset() const;
            void set_byte_offset(uint32_t value);

        private:
            uint32_t m_packed_data;
            uint32_t m_byte_offset;
        };

        // --------------------------------------------------------------------

        struct Value
        {
            Value_kind kind;
            std::string name;
            union
            {
                int32_t data_int[4];
                float data_float[4];
            };
        };
    };

    // --------------------------------------------------------------------------------------------

    class IScene_loader
    {
    public:

        // --------------------------------------------------------------------

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

            std::vector<Scene_data::Value> scene_data;
        };

        // --------------------------------------------------------------------

        /// Date element stored per vertex, e.g.: position, normal, ... , vertex color
        struct Vertex_element
        {
            /// Name of the semantic (name of the prim-var)
            std::string semantic;

            /// Data type
            Scene_data::Value_kind kind;

            /// the i`th prim-var at this vertex (data is stored interleaved)
            uint8_t layout_index;

            /// the offset to the data within the vertex (data is stored interleaved)
            uint32_t byte_offset;

            /// size of the date in bytes (multiple of 4 bytes)
            uint32_t element_size;

            /// interpolation of the data over the primitive
            Scene_data::Interpolation_mode interpolation_mode;
        };

        // --------------------------------------------------------------------

        class Primitive
        {
        public:
            size_t vertex_buffer_byte_offset;
            size_t vertex_count;
            size_t index_offset;
            size_t index_count;
            size_t material;

            std::vector<Vertex_element> vertex_element_layout;
        };

        // --------------------------------------------------------------------

        class Mesh
        {
        public:
            std::string name;
            std::vector<Primitive> primitives;
            std::vector<uint8_t> vertex_data;
            std::vector<uint32_t> indices;
        };

        // --------------------------------------------------------------------

        class Camera
        {
        public:
            std::string name;
            float vertical_fov;
            float aspect_ratio;
            float near_plane_distance;
            float far_plane_distance;
        };

        // --------------------------------------------------------------------

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

            // corresponds to glTF Extension: KHR_materials_clearcoat
            struct Model_data_materials_clearcoat
            {
                float clearcoat_factor = 0.0f;
                std::string clearcoat_texture = "";
                float clearcoat_roughness_factor = 0.0f;
                std::string clearcoat_roughness_texture = "";
                std::string clearcoat_normal_texture = "";
            };

            struct Pbr_model_data_metallic_roughness
            {
                std::string base_color_texture = "";
                DirectX::XMFLOAT4 base_color_factor = { 1.0f, 1.0f, 1.0f, 1.0f };
                std::string metallic_roughness_texture = "";
                float metallic_factor = 0.0f;
                float roughness_factor = 0.1f;

                Model_data_materials_clearcoat clearcoat;
            };

            // corresponds to glTF Extension: KHR_materials_pbrSpecularGlossiness
            struct Pbr_model_data_khr_specular_glossiness
            {
                std::string diffuse_texture = "";
                DirectX::XMFLOAT4 diffuse_factor = { 1.0f, 1.0f, 1.0f, 1.0f };

                std::string specular_glossiness_texture = "";
                DirectX::XMFLOAT3 specular_factor = { 1.0f, 1.0f, 1.0f };
                float glossiness_factor = 0.9f;
            };

            std::string name = "<name>";

            std::string normal_texture = "";
            float normal_scale_factor = 1.0f;

            std::string occlusion_texture = "";
            float occlusion_strength = 0.0f;

            std::string emissive_texture = "";
            DirectX::XMFLOAT3 emissive_factor = { 0.0f, 0.0f, 0.0f };

            Alpha_mode alpha_mode = Alpha_mode::Opaque;
            float alpha_cutoff = 0.5f;
            bool single_sided = false;

            // depending on the material model different sub classes
            Pbr_model pbr_model;
            Pbr_model_data_metallic_roughness metallic_roughness;
            Pbr_model_data_khr_specular_glossiness khr_specular_glossiness;

        };

        // --------------------------------------------------------------------

        class Scene
        {
        public:
            std::vector<Mesh> meshes;
            std::vector<Camera> cameras;
            std::vector<Material> materials;
            Node root;
        };

        // --------------------------------------------------------------------

        class Scene_options
        {
        public:
            explicit Scene_options();
            virtual ~Scene_options() = default;

            float units_per_meter; // todo
            bool handle_z_axis_up;
        };

        // --------------------------------------------------------------------

        virtual bool load(const std::string& file_name, const Scene_options& options) = 0;
        virtual const Scene& get_scene() const = 0;
    };

    // --------------------------------------------------------------------------------------------

    class Mesh
    {
        friend class Scene;

    public:
        struct Instance;

        // --------------------------------------------------------------------

        struct Geometry
        {
            friend class Mesh;
            friend struct Instance;

            explicit Geometry(
                Base_application* app,
                const Mesh& parent_mesh,
                const IScene_loader::Primitive& primitive,
                size_t index_in_mesh);

            ~Geometry();

            const Raytracing_acceleration_structure::Geometry_handle& get_geometry_handle() const {
                return m_geometry_handle;
            }

            size_t get_index_in_mesh() const { return m_index_in_mesh; }
            size_t get_vertex_buffer_byte_offset() const { return m_vertex_buffer_byte_offset; }
            size_t get_vertex_stride() const;
            size_t get_vertex_count() const { return m_vertex_count; }
            size_t get_index_offset() const { return m_index_offset; }
            size_t get_index_count() const { return m_index_count; }

            /// index of the first info in the scene data info buffer
            size_t get_scene_data_info_buffer_offset() const { return m_scene_data_info_offset; }

            const std::vector<IScene_loader::Vertex_element>& get_vertex_layout() const  {
                return m_vertex_layout;
            }

        private:
            bool update_scene_data_infos(
                const std::unordered_map<std::string, uint32_t>& scene_data_name_map,
                Scene_data::Info* scene_data_buffer,
                uint32_t geometry_scene_data_info_offset,
                uint32_t geometry_scene_data_info_count);


            Base_application* m_app;
            std::string m_name;
            size_t m_index_in_mesh;
            Raytracing_acceleration_structure::Geometry_handle m_geometry_handle;

            size_t m_vertex_buffer_byte_offset;
            size_t m_vertex_count;
            size_t m_index_offset;
            size_t m_index_count;
            size_t m_scene_data_info_offset;

            std::vector<IScene_loader::Vertex_element> m_vertex_layout;
        };

        // --------------------------------------------------------------------

        struct Instance
        {
            friend class Mesh;
            explicit Instance(
                Base_application* app,
                const IScene_loader::Node& node_desc);
            ~Instance();

            /// update scene data of the instance along with per vertex data of the geometry
            /// of this instance, it also involves the scene data name that appear in the
            /// material assigned to the geometry, so afterwards, material assignments
            /// are not possible or require another update of the scene data infos.
            bool update_scene_data_infos(D3DCommandList* command_list);

            const Mesh* get_mesh() const { return m_mesh; }
            Mesh* get_mesh() { return m_mesh; }

            const IMaterial* get_material(const Mesh::Geometry* geometry) const {
                return m_materials[geometry->m_index_in_mesh];
            }

            void set_material(const Mesh::Geometry* geometry, IMaterial* material) {
                m_materials[geometry->m_index_in_mesh] = material;
            }

            const Raytracing_acceleration_structure::Instance_handle& get_instance_handle() const {
                return m_instance_handle;
            }

            const Structured_buffer<Scene_data::Info>* get_scene_data_info_buffer() const
            {
                return m_scene_data_infos;
            }

            /// The per object/instance buffer data, can be NULL if there is no such data.
            const Structured_buffer<uint32_t>* get_scene_data_buffer() const
            {
                return m_scene_data_buffer;
            }

        private:
            Base_application* m_app;
            Mesh* m_mesh;
            Raytracing_acceleration_structure::Instance_handle m_instance_handle;
            std::vector<IMaterial*> m_materials; // materials for each geometry

            // contains scene data infos on a per object base and vertex (of the meshes geometry)
            Structured_buffer<Scene_data::Info>* m_scene_data_infos;

            // contains scene data (CPU side, pushed to GPU in update_scene_data_infos(...))
            std::vector<Scene_data::Value> m_scene_data;

            // contains scene data (GPU side)
            Structured_buffer<uint32_t>* m_scene_data_buffer;

        };

        // --------------------------------------------------------------------

        explicit Mesh(
            Base_application* app,
            Raytracing_acceleration_structure* acceleration_structure,
            const IScene_loader::Mesh& mesh_desc);
        virtual ~Mesh();

        Instance* create_instance(const IScene_loader::Node& node_desc);
        bool upload_buffers(D3DCommandList* command_list);

        const std::string& get_name() const { return m_name; }

        const std::vector<Geometry>& get_geometries() const { return m_geometries; }

        bool visit_geometries(std::function<bool(const Geometry*)> action) const;
        bool visit_geometries(std::function<bool(Geometry*)> action);

        const Vertex_buffer<uint8_t>* get_vertex_buffer() const { return m_vertex_buffer; }
        const Index_buffer* get_index_buffer() const { return m_index_buffer; }

        const Bounding_box& get_local_bounding_box() const { return m_local_aabb; };

    private:
        Base_application* m_app;

        std::string m_name;
        Vertex_buffer<uint8_t>* m_vertex_buffer;
        Index_buffer* m_index_buffer;

        Raytracing_acceleration_structure* m_acceleration_structur;
        Raytracing_acceleration_structure::BLAS_handle m_blas;

        std::vector<Geometry> m_geometries;
        Bounding_box m_local_aabb;
    };

    // --------------------------------------------------------------------------------------------

    class Camera
    {
        friend class Scene;
        friend class Scene_node;
    public:

        // --------------------------------------------------------------------

        struct Constants
        {
            DirectX::XMMATRIX view;
            DirectX::XMMATRIX perspective;
            DirectX::XMMATRIX view_inv;
            DirectX::XMMATRIX perspective_inv;
        };

        // --------------------------------------------------------------------

        explicit Camera(
            Base_application* app,
            const IScene_loader::Camera& camera_desc);

        virtual ~Camera();

        const std::string& get_name() const { return m_name; }
        const Dynamic_constant_buffer<Camera::Constants>* get_constants() const { return m_constants; }

        // get the vertical field of view in radians
        float get_field_of_view () const { return m_field_of_view; }

        // set the vertical field of view in radians
        void set_field_of_view(float vertical_fov) {
            m_field_of_view = vertical_fov;
            m_projection_changed = true;
        }

        // get focal length in mm, computed from field of view, assuming 36x24mm
        float get_focal_length() const
        {
            float l = 12.0f / std::tanf(m_field_of_view * 0.5f);
            return l;
        }

        // set focal length in mm, re-computes field of view, assuming 36x24mm
        void set_focal_length(float focal_length)
        {
            m_field_of_view = std::atan(12.0f / focal_length) * 2.0f;
            m_projection_changed = true;
        }

        float get_aspect_ratio() const { return m_aspect_ratio; }
        void set_aspect_ratio(float aspect) {
            m_aspect_ratio = aspect;
            m_projection_changed = true;
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

        Dynamic_constant_buffer<Camera::Constants>* m_constants;
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

        const Mesh::Instance* get_mesh_instance() const {
            return m_kind == Kind::Mesh ? m_mesh_instance : nullptr;
        }

        Mesh::Instance* get_mesh_instance() {
            return m_kind == Kind::Mesh ? m_mesh_instance : nullptr;
        }

        Camera* get_camera() { return m_kind == Kind::Camera ? m_camera : nullptr; }
        const Camera* get_camera() const { return m_kind == Kind::Camera ? m_camera : nullptr; }

        // get the local transformation of this node relative to its parent.
        Transform& get_local_transformation() { return m_local_transformation; }

        // set the entire transform at once
        void set_local_transformation(const Transform& transform) {
            m_local_transformation = transform;
        }

        const Bounding_box& get_global_bounding_box() const { return m_global_aabb; }
        const Bounding_box& get_local_bounding_box() const;

        // get the world transformation of this node (read-only).
        const DirectX::XMMATRIX& get_global_transformation() const { return m_global_transformation; }

        void add_child(Scene_node* to_add);

        bool update(const Update_args& args);

        const Scene_node* get_parent() const { return m_parent; }

    private:
        void update_bounding_volumes();

        Base_application* m_app;

        Kind m_kind;
        std::string m_name;
        Scene* m_scene;
        Scene_node* m_parent;
        Transform m_local_transformation;
        DirectX::XMMATRIX m_global_transformation;

        Bounding_box m_global_aabb;

        std::vector<Scene_node*> m_children;

        Mesh::Instance* m_mesh_instance;

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

        /// set the display name of the material, which will show up in a the GUI for instance.
        virtual void set_name(const std::string& value) = 0;

        /// get the display name of the material, which will show up in a the GUI for instance.
        virtual const std::string& get_name() const = 0;

        /// get the id of the target code that contains this material.
        /// can be used with the material library for instance.
        virtual size_t get_target_code_id() const = 0;

        /// get the scene data names mapped to IDs that will be requested in the shader.
        virtual const std::unordered_map<std::string, uint32_t>& get_scene_data_name_map() const = 0;

        /// get the GPU handle of to the first resource of the target in the descriptor heap
        virtual D3D12_GPU_DESCRIPTOR_HANDLE get_target_descriptor_heap_region() const = 0;

        /// get the GPU handle of to the first resource of this material in the descriptor heap
        virtual D3D12_GPU_DESCRIPTOR_HANDLE get_material_descriptor_heap_region() const = 0;
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
        bool visit(
            Scene_node::Kind mask,
            std::function<bool(const Scene_node* node)> action) const;

        /// iterate recursively over all nodes that match the Scene_node::Kind mask.
        ///
        /// \param mask     selected kind of nodes to visit
        /// \param action   action to run while visiting a selected node.
        ///                 if the action returns false, the traversal is aborted.
        ///
        /// \returns        false if the traversal was aborted, true otherwise.
        bool visit(
            Scene_node::Kind mask,
            std::function<bool(Scene_node* node)> action);

        /// create a new camera and add it to the scene.
        Scene_node* create(
            const IScene_loader::Camera& camera_description,
            const Transform& local_transform = Transform::Identity,
            Scene_node* parent = nullptr);


        Scene_node* get_root() { return &m_root; }

        bool update(const Update_args& args) { return m_root.update(args); }

    private:
        Base_application* m_app;
        const std::string& m_debug_name;

        std::vector<Camera*> m_cameras;
        std::vector<Mesh*> m_meshes;
        std::vector<IMaterial*> m_materials;

        Scene_node m_root;
        Raytracing_acceleration_structure* m_acceleration_structure;
    };

}}} // mi::examples::mdl_d3d12
#endif
