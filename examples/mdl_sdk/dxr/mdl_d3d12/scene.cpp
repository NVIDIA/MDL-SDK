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

#include "scene.h"
#include "base_application.h"
#include "buffer.h"
#include "command_queue.h"
#include "mdl_material.h"


namespace mdl_d3d12
{
    Transform::Transform()
        : translation({0.0f, 0.0f, 0.0f})
        , rotation(DirectX::XMQuaternionIdentity())
        , scale({1.0f, 1.0f, 1.0f})
    {
    }

    bool Transform::is_identity() const
    {
        if (translation.x != 0.0f || translation.y != 0.0f || translation.z != 0.0f) return false;
        if (scale.x != 1.0f || scale.y != 1.0f || scale.z != 1.0f) return false;
        if (!DirectX::XMQuaternionIsIdentity(rotation)) return false;
        return true;
    }

    bool Transform::try_from_matrix(const DirectX::XMMATRIX& matrix, Transform& out_transform)
    {
        DirectX::XMVECTOR s, r, t;
        bool success = DirectX::XMMatrixDecompose(&s, &r, &t, matrix);
        out_transform.translation.x = t.m128_f32[0];
        out_transform.translation.y = t.m128_f32[1];
        out_transform.translation.z = t.m128_f32[2];
        out_transform.rotation = DirectX::XMQuaternionNormalize(r);
        out_transform.scale.x = s.m128_f32[0];
        out_transform.scale.y = s.m128_f32[1];
        out_transform.scale.z = s.m128_f32[2];
        return success;
    }

    DirectX::XMMATRIX Transform::get_matrix()
    {
        DirectX::XMMATRIX res = DirectX::XMMatrixMultiply(DirectX::XMMatrixMultiply(
            DirectX::XMMatrixScaling(scale.x, scale.y, scale.z),
            DirectX::XMMatrixRotationQuaternion(rotation)),
            DirectX::XMMatrixTranslation(translation.x, translation.y, translation.z));
        return std::move(res);
    }

    const Transform Transform::identity = Transform();

    // --------------------------------------------------------------------------------------------

    Mesh::Geometry::Geometry()
        : m_material(nullptr)
        , m_geometry()
    {
    }

    Mesh::Mesh(
        Base_application* app, 
        Raytracing_acceleration_structure* acceleration_structure, 
        const IScene_loader::Mesh& mesh_desc)

        : m_app(app)
        , m_name(mesh_desc.name)
        , m_vertex_buffer(new Vertex_buffer<Vertex>(
            app, mesh_desc.vertices.size(), mesh_desc.name + "_VertexBuffer"))
        , m_index_buffer(new Index_buffer(
            app, mesh_desc.indices.size(), mesh_desc.name + "_IndexBuffer"))
        , m_acceleration_structur(acceleration_structure)
        , m_blas()
        , m_geometries(mesh_desc.primitives.size())
    {
        m_vertex_buffer->set_data(mesh_desc.vertices.data());
        m_index_buffer->set_data(mesh_desc.indices.data());
        m_blas = m_acceleration_structur->add_bottom_level_structure(mesh_desc.name + "_BLAS");

        for (size_t i = 0, n = mesh_desc.primitives.size(); i < n; ++i)
        {
            auto& p = mesh_desc.primitives[i];
            m_geometries[i].m_geometry = m_acceleration_structur->add_geometry(
                m_blas,
                m_vertex_buffer, p.vertex_offset, p.vertex_count, 0,
                m_index_buffer, p.index_offset, p.index_count);
            m_geometries[i].m_index_offset = static_cast<uint32_t>(p.index_offset);
        }
    }

    Mesh::~Mesh()
    {
        delete m_vertex_buffer;
        delete m_index_buffer;
    }

    const Raytracing_acceleration_structure::Instance_handle Mesh::create_instance()
    {
        return m_acceleration_structur->add_instance(
            m_blas,
            DirectX::XMMatrixIdentity());
    }

    bool Mesh::upload_buffers(D3DCommandList* command_list)
    {
        if (!m_vertex_buffer->upload(command_list)) return false;
        if (!m_index_buffer->upload(command_list)) return false;
        return true;
    }

    // --------------------------------------------------------------------------------------------

    Camera::Camera(Base_application* app, const IScene_loader::Camera& camera_desc)
        : m_app(app)
        , m_name(camera_desc.name)
        , m_field_of_view(camera_desc.vertical_fov)
        , m_aspect_ratio(camera_desc.aspect_ratio)
        , m_near_plane_distance(camera_desc.near_plane_distance)
        , m_far_plane_distance(camera_desc.far_plane_distance)
        , m_projection_changed(true)
    {
        m_constants = new Constant_buffer<Camera::Constants>(m_app, m_name + "_Constants");
    }

    Camera::~Camera()
    {
        delete m_constants;
    }

    void Camera::update(const DirectX::XMMATRIX& global_transform, bool transform_changed)
    {
        if (!m_projection_changed && !transform_changed)
            return;

        if (transform_changed)
        {
            m_constants->data.view = DirectX::XMMatrixInverse(nullptr, global_transform);
            m_constants->data.view_inv = global_transform;
        }

        if (m_projection_changed)
        {
            m_constants->data.perspective = DirectX::XMMatrixPerspectiveFovRH(
                m_field_of_view, m_aspect_ratio, m_near_plane_distance, m_far_plane_distance);

            m_constants->data.perspective_inv =
                DirectX::XMMatrixInverse(nullptr, m_constants->data.perspective);

            m_projection_changed = false;
        }

        m_constants->upload();
    }

    // --------------------------------------------------------------------------------------------

    Scene_node::Scene_node(Base_application* app, Scene* scene, Kind kind, const std::string& name)
        : m_app(app)
        , m_kind(kind)
        , m_name(name)
        , m_scene(scene)
        , m_parent(nullptr)
        , m_local_transformation()
        , m_global_transformation(DirectX::XMMatrixIdentity())
        , m_transformed_on_last_update(false)
        , m_mesh(nullptr)
        , m_mesh_instance()
        , m_camera(nullptr)
    {
    }

    Scene_node::~Scene_node()
    {
        for (size_t c = 0, n = m_children.size(); c < n; ++c)
            delete m_children[c];
    }

    void Scene_node::add_child(Scene_node* to_add)
    {
        m_children.push_back(to_add);
        to_add->m_parent = this;
    }

    void Scene_node::update(const Update_args& args)
    {
        DirectX::XMMATRIX previous = m_global_transformation;

        if (m_parent)
            m_global_transformation = DirectX::XMMatrixMultiply(
                m_local_transformation.get_matrix(), m_parent->m_global_transformation);
        else
            m_global_transformation = m_local_transformation.get_matrix();

        m_transformed_on_last_update = false;
        float epsilon = 0.000001f;
        for(size_t c = 0; c < 16; ++c)
            if (std::fabsf(((float*)(&previous))[c] - 
                           ((float*)(&m_global_transformation))[c]) > epsilon)
            {
                m_transformed_on_last_update = true;
                break;
            }


        switch (m_kind)
        {
            case Kind::Mesh:
            {
                if(m_transformed_on_last_update)
                    m_scene->get_acceleration_structure()->set_instance_transform(
                        m_mesh_instance, m_global_transformation);
                break;
            }

            case Kind::Camera:
            {
                m_camera->update(m_global_transformation, m_transformed_on_last_update);
                break;
            }

            default:
                break;
        }


        for (auto& c : m_children)
            c->update(args);
    }

    // --------------------------------------------------------------------------------------------

    Scene::Scene(Base_application* app, const std::string& debug_name, size_t ray_type_count)
        : m_app(app)
        , m_debug_name(debug_name)
        , m_acceleration_structure(
            new Raytracing_acceleration_structure(app, ray_type_count, "AccelerationStructure"))
        , m_root(app, this, Scene_node::Kind::Empty, "Root")
    {
    }

    Scene::~Scene()
    {
        delete m_acceleration_structure;

        for (auto& m : m_meshes)
            delete m;

        for (auto& c : m_cameras)
            delete c;

        for (auto& m : m_materials)
            delete m;
    }

    bool Scene::build_scene(const IScene_loader::Scene& scene)
    {
        std::unordered_map<size_t, Mesh*> handled_meshes;
        std::unordered_map<size_t, IMaterial*> handled_materials;

        // process the scene graph
        std::function<void(Scene_node&, const IScene_loader::Node&)> traverse = 
            [&](Scene_node& parent, const IScene_loader::Node& src_child)
            {
                Scene_node* child = new Scene_node(m_app, this, src_child.kind, src_child.name);

                switch (src_child.kind)
                {
                    case IScene_loader::Node::Kind::Mesh:
                    {
                        // mesh already handles?
                        auto it = handled_meshes.find(src_child.index);
                        if (it != handled_meshes.end()) {
                            child->m_mesh = it->second;
                            child->m_mesh_instance = child->m_mesh->create_instance();
                            break;
                        }

                        // process materials first
                        auto& primitives = scene.meshes[src_child.index].primitives;
                        for (const auto& p : primitives)
                        {
                            // mesh already handles?
                            auto it = handled_materials.find(p.material);
                            if (it != handled_materials.end())
                                continue;

                            // create material
                            IMaterial* mat = m_app->get_mdl_sdk().get_library()->create(
                                scene.materials[p.material], nullptr);

                            if (!mat) {
                                log_error("Failed to create material: " + 
                                          scene.materials[p.material].name, SRC);
                            }

                            handled_materials[p.material] = mat;
                        }

                        // create mesh
                        child->m_mesh = new Mesh(
                            m_app, m_acceleration_structure, scene.meshes[src_child.index]);
                        child->m_mesh_instance = child->m_mesh->create_instance();
                        handled_meshes[src_child.index] = child->m_mesh;

                        // assign materials to the primitives
                        for (size_t p = 0, n = primitives.size(); p < n; ++p)
                            child->m_mesh->m_geometries[p].m_material = 
                                handled_materials[primitives[p].material];

                        break;
                    }

                    case IScene_loader::Node::Kind::Camera:
                    {
                        // create individual cameras
                        child->m_camera = new Camera(m_app, scene.cameras[src_child.index]);
                        m_cameras.push_back(child->m_camera);
                        break;
                    }

                    default:
                        break;
                }

                // transforms
                child->set_local_transformation(src_child.local);

                // add to parent
                parent.add_child(std::move(child));

                // go down recursively
                for (const auto& c : src_child.children)
                    traverse(*child, c);
            };
        traverse(m_root, scene.root);
        m_root.set_local_transformation(scene.root.local);

        // add handled meshes, materials, ... to the scene
        for (auto& m : handled_materials)
            m_materials.push_back(m.second);

        for (auto& m : handled_meshes)
            m_meshes.push_back(m.second);

        // upload data
        // ----------------------------------------------------------------------------------------
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        for (auto& m : m_meshes)
            if (!m->upload_buffers(command_list)) return false;

        command_queue->execute_command_list(command_list);
        command_queue->flush();

        // build acceleration data structure
        // ----------------------------------------------------------------------------------------
        update(Update_args());

        command_list = command_queue->get_command_list();

        if (!m_acceleration_structure->build(command_list)) return false;

        command_queue->execute_command_list(command_list);
        command_queue->flush();

        // free unused temp data - required for construction
        m_acceleration_structure->release_static_scratch_buffers();

        return true;
    }

    Scene_node* Scene::create(
        const IScene_loader::Camera& camera_description,
        const Transform& local_transform,
        Scene_node* parent)
    {
        // create a scene node
        Scene_node* node = new Scene_node(
            m_app, this, Scene_node::Kind::Camera, camera_description.name);
        node->set_local_transformation(local_transform);

        // create the cameras
        node->m_camera = new Camera(m_app, camera_description);
        m_cameras.push_back(node->m_camera);

        // add to selected parent or the root
        if (parent)
            parent->m_children.emplace_back(node);
        else
            m_root.m_children.emplace_back(node);

        return node;
    }


    bool Scene::traverse(
        Scene_node::Kind mask, std::function<bool(const Scene_node* node)> action) const
    {
        std::function<bool(const Scene_node&)> traverse = [&](const Scene_node& node)
        {
            if (has_flag(node.m_kind, mask))
                if(!action(&node))
                    return false;

            for (const auto& c : node.m_children)
                if (!traverse(*c))
                    return false;

            return true;
        };
        return traverse(m_root);
    }

    bool Scene::traverse(
        Scene_node::Kind mask, std::function<bool(Scene_node* node)> action)
    {
        std::function<bool(Scene_node&)> traverse = [&](Scene_node& node)
        {
            if (has_flag(node.m_kind, mask))
                if (!action(&node))
                    return false;

            for (const auto& c : node.m_children)
                if (!traverse(*c))
                    return false;

            return true;
        };
        return traverse(m_root);
    }


    IMaterial* Scene::get_material(size_t index)
    {
        if(index >= m_materials.size())
            return nullptr;
        return m_materials[index];
    }
}
