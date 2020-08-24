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

#include "scene.h"
#include "base_application.h"
#include "buffer.h"
#include "command_queue.h"
#include "mdl_material.h"
#include "mdl_material_description.h"
#include "mdl_material_library.h"
#include "mdl_sdk.h"


namespace mi { namespace examples { namespace mdl_d3d12
{

const Transform Transform::Identity = Transform();

// ------------------------------------------------------------------------------------------------

Transform::Transform()
    : translation({0.0f, 0.0f, 0.0f})
    , rotation(DirectX::XMQuaternionIdentity())
    , scale({1.0f, 1.0f, 1.0f})
{
}

// ------------------------------------------------------------------------------------------------

bool Transform::is_identity() const
{
    if (translation.x != 0.0f || translation.y != 0.0f || translation.z != 0.0f) return false;
    if (scale.x != 1.0f || scale.y != 1.0f || scale.z != 1.0f) return false;
    if (!DirectX::XMQuaternionIsIdentity(rotation)) return false;
    return true;
}

// ------------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------------

Transform Transform::look_at(
    const DirectX::XMFLOAT3& camera_pos,
    const DirectX::XMFLOAT3& focus,
    const DirectX::XMFLOAT3& up)
{
    DirectX::XMMATRIX lookat =
        XMMatrixLookAtRH(vector(camera_pos, 1.0f), vector(focus, 1.0f), vector(up, 1.0f));

    DirectX::XMVECTOR det;
    lookat = DirectX::XMMatrixInverse(&det, lookat);

    Transform cam_trafo;
    Transform::try_from_matrix(lookat, cam_trafo);
    return cam_trafo;
}

// ------------------------------------------------------------------------------------------------

DirectX::XMMATRIX Transform::get_matrix() const
{
    DirectX::XMMATRIX res = DirectX::XMMatrixMultiply(DirectX::XMMatrixMultiply(
        DirectX::XMMatrixScaling(scale.x, scale.y, scale.z),
        DirectX::XMMatrixRotationQuaternion(rotation)),
        DirectX::XMMatrixTranslation(translation.x, translation.y, translation.z));
    return std::move(res);
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

const Bounding_box Bounding_box::Zero = Bounding_box();

const Bounding_box Bounding_box::Invalid = Bounding_box(
    DirectX::XMFLOAT3{std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()},
    DirectX::XMFLOAT3{std::numeric_limits<float>::min(),
                        std::numeric_limits<float>::min(),
                        std::numeric_limits<float>::min()});

// ------------------------------------------------------------------------------------------------

Bounding_box::Bounding_box(const DirectX::XMFLOAT3& min, const DirectX::XMFLOAT3& max)
    : min(min)
    , max(max)
{
}

// ------------------------------------------------------------------------------------------------

Bounding_box::Bounding_box()
    : min({0.0f, 0.0f, 0.0f})
    , max({0.0f, 0.0f, 0.0f})
{
}

// ------------------------------------------------------------------------------------------------

Bounding_box Bounding_box::merge(const Bounding_box& first, const Bounding_box& second)
{
    Bounding_box f = first;
    f.merge(second);
    return f;
}

// ------------------------------------------------------------------------------------------------

void Bounding_box::merge(const Bounding_box& second)
{
    // ignore NaNs
    min.x = std::min(min.x, second.min.x);
    min.y = std::min(min.y, second.min.y);
    min.z = std::min(min.z, second.min.z);
    max.x = std::max(max.x, second.max.x);
    max.y = std::max(max.y, second.max.y);
    max.z = std::max(max.z, second.max.z);
}

// ------------------------------------------------------------------------------------------------

Bounding_box Bounding_box::extend(const Bounding_box& box, const DirectX::XMFLOAT3& point)
{
    Bounding_box b = box;
    b.extend(point);
    return b;
}

// ------------------------------------------------------------------------------------------------

void Bounding_box::extend(const DirectX::XMFLOAT3& point)
{
    min.x = std::min(min.x, point.x);
    min.y = std::min(min.y, point.y);
    min.z = std::min(min.z, point.z);
    max.x = std::max(max.x, point.x);
    max.y = std::max(max.y, point.y);
    max.z = std::max(max.z, point.z);
}

// ------------------------------------------------------------------------------------------------

void Bounding_box::get_corners(std::vector<DirectX::XMVECTOR>& out_corners) const
{
    out_corners.resize(8);
    out_corners[0] = DirectX::XMVECTOR{min.x, max.y, max.z, 1.0f};
    out_corners[1] = DirectX::XMVECTOR{max.x, max.y, max.z, 1.0f};
    out_corners[2] = DirectX::XMVECTOR{max.x, min.y, max.z, 1.0f};
    out_corners[3] = DirectX::XMVECTOR{min.x, min.y, max.z, 1.0f};
    out_corners[4] = DirectX::XMVECTOR{min.x, max.y, min.z, 1.0f};
    out_corners[5] = DirectX::XMVECTOR{max.x, max.y, min.z, 1.0f};
    out_corners[6] = DirectX::XMVECTOR{max.x, min.y, min.z, 1.0f};
    out_corners[7] = DirectX::XMVECTOR{min.x, min.y, min.z, 1.0f};
}

// ------------------------------------------------------------------------------------------------

Bounding_box Bounding_box::transform(const Bounding_box& box, const Transform& transformation)
{
    DirectX::XMMATRIX mat = transformation.get_matrix();
    return transform(box, mat);
}

// ------------------------------------------------------------------------------------------------

Bounding_box Bounding_box::transform(
    const Bounding_box& box, const DirectX::XMMATRIX& matrix)
{
    if (!box.is_valid())
        return box;

    Bounding_box new_box = Bounding_box::Invalid;
    DirectX::XMVECTOR box_min = DirectX::XMLoadFloat3(&new_box.min);
    DirectX::XMVECTOR box_max = DirectX::XMLoadFloat3(&new_box.max);

    std::vector<DirectX::XMVECTOR> corners(8);
    box.get_corners(corners);

    for (DirectX::XMVECTOR& c : corners)
    {
        c = DirectX::XMVector3TransformCoord(c, matrix);
        box_min = DirectX::XMVectorMin(box_min, c);
        box_max = DirectX::XMVectorMax(box_max, c);
    }

    DirectX::XMStoreFloat3(&new_box.min, box_min);
    DirectX::XMStoreFloat3(&new_box.max, box_max);
    return new_box;
}

// ------------------------------------------------------------------------------------------------

DirectX::XMFLOAT3 Bounding_box::center() const
{
    return {
        (min.x + max.x) * 0.5f,
        (min.y + max.y) * 0.5f,
        (min.z + max.z) * 0.5f
    };
};

// ------------------------------------------------------------------------------------------------

DirectX::XMFLOAT3 Bounding_box::size() const
{
    return {
        (max.x - min.x),
        (max.y - min.y),
        (max.z - min.z)
    };
};

// ------------------------------------------------------------------------------------------------

bool Bounding_box::is_zero() const
{
    return *this == Bounding_box::Zero;
}

// ------------------------------------------------------------------------------------------------

bool Bounding_box::is_valid() const
{
    return *this != Bounding_box::Invalid;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

namespace
{

// c++11
static const uint32_t VERTEX_INFO_KIND =    0xF0000000u;
static const uint32_t VERTEX_INFO_ELEMENT = 0x0F800000u;
static const uint32_t VERTEX_INFO_INTERP = 0x00F00000u;

static const uint32_t VERTEX_INFO_UNIFORM = 0x00010000u;
// 3 bits left
static const uint32_t VERTEX_INFO_STRIDE = 0x0000FFFFu;

// c++14
/*
static const uint32_t VERTEX_INFO_KIND =    0b 1111 0000  0000 0000  0000 0000  0000 0000 u;
static const uint32_t VERTEX_INFO_ELEMENT = 0b 0000 1111  0000 0000  0000 0000  0000 0000 u;
static const uint32_t VERTEX_INFO_INTERP =  0b 0000 0000  1111 0000  0000 0000  0000 0000 u;
static const uint32_t VERTEX_INFO_UNIFORM = 0b 0100 0000  0000 0001  0000 0000  0000 0000 u;
/// 3 bits left
static const uint32_t VERTEX_INFO_STRIDE =  0b 0000 0000  0000 0000  1111 1111  1111 1111 u;
*/

} // anonymous

Scene_data::Info::Info()
    : m_packed_data(0)
    , m_byte_offset(0)
{
}

// ------------------------------------------------------------------------------------------------

Scene_data::Kind Scene_data::Info::get_kind() const
{
    uint32_t kind = (m_packed_data & VERTEX_INFO_KIND) >> 28;
    return static_cast<Scene_data::Kind>(kind);
}

// ------------------------------------------------------------------------------------------------

void Scene_data::Info::set_kind(Scene_data::Kind value)
{
    m_packed_data = m_packed_data & ~VERTEX_INFO_KIND;
    m_packed_data += static_cast<uint32_t>(value) << 28;
}

// ------------------------------------------------------------------------------------------------

Scene_data::Element_type Scene_data::Info::get_element_type() const
{
    uint32_t type = (m_packed_data & VERTEX_INFO_ELEMENT) >> 24;
    return static_cast<Scene_data::Element_type>(type);
}

// ------------------------------------------------------------------------------------------------

void Scene_data::Info::set_element_type(Scene_data::Element_type value)
{
    m_packed_data = m_packed_data & ~VERTEX_INFO_ELEMENT;
    m_packed_data += static_cast<uint32_t>(value) << 24;
}

// ------------------------------------------------------------------------------------------------

Scene_data::Interpolation_mode Scene_data::Info::get_interpolation_mode() const
{
    uint32_t mode = (m_packed_data & VERTEX_INFO_INTERP) >> 20;
    return static_cast<Scene_data::Interpolation_mode>(mode);
}

// ------------------------------------------------------------------------------------------------

void Scene_data::Info::set_interpolation_mode(Scene_data::Interpolation_mode value)
{
    m_packed_data = m_packed_data & ~VERTEX_INFO_INTERP;
    m_packed_data += static_cast<uint32_t>(value) << 20;
}

// ------------------------------------------------------------------------------------------------

bool Scene_data::Info::get_uniform() const
{
    return (m_packed_data & VERTEX_INFO_UNIFORM) > 0;
}

// ------------------------------------------------------------------------------------------------

void Scene_data::Info::set_uniform(bool value)
{
    if (value)
        m_packed_data = m_packed_data | VERTEX_INFO_UNIFORM;
    else
        m_packed_data = m_packed_data & ~VERTEX_INFO_UNIFORM;
}

// ------------------------------------------------------------------------------------------------

uint16_t Scene_data::Info::get_byte_stride() const
{
    return uint16_t(m_packed_data & VERTEX_INFO_STRIDE);
}

// ------------------------------------------------------------------------------------------------

void Scene_data::Info::set_byte_stride(uint16_t value)
{
    m_packed_data = m_packed_data & ~VERTEX_INFO_STRIDE;
    m_packed_data += value;
}

// ------------------------------------------------------------------------------------------------

uint32_t Scene_data::Info::get_byte_offset() const
{
    return m_byte_offset;
}

// ------------------------------------------------------------------------------------------------

void Scene_data::Info::set_byte_offset(uint32_t value)
{
    m_byte_offset = value;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Mesh::Geometry::Geometry(
    Base_application* app,
    const Mesh& parent_mesh,
    const IScene_loader::Primitive& primitive,
    size_t index_in_mesh)
    : m_app(app)
    , m_name(parent_mesh.get_name() + "_" + std::to_string(index_in_mesh))
    , m_index_in_mesh(index_in_mesh)
    , m_geometry_handle()
    , m_vertex_buffer_byte_offset(static_cast<uint32_t>(primitive.vertex_buffer_byte_offset))
    , m_vertex_count(primitive.vertex_count)
    , m_index_offset(static_cast<uint32_t>(primitive.index_offset))
    , m_index_count(primitive.index_count)
    , m_scene_data_info_offset(0)
    , m_vertex_layout(primitive.vertex_element_layout.size())
{
    for (size_t s = 0; s < primitive.vertex_element_layout.size(); s++)
        m_vertex_layout[s] = primitive.vertex_element_layout[s];
}

// ------------------------------------------------------------------------------------------------

Mesh::Geometry::~Geometry()
{
}

// ------------------------------------------------------------------------------------------------

bool Mesh::Geometry::update_scene_data_infos(
    const std::unordered_map<std::string, uint32_t>& scene_data_name_map,
    Scene_data::Info* scene_data_buffer,
    uint32_t geometry_scene_data_info_offset,
    uint32_t geometry_scene_data_info_count)
{
    // gather information required for the scene data
    for (auto& e : m_vertex_layout)
    {
        auto found = scene_data_name_map.find(e.semantic);
        if (found == scene_data_name_map.end())
            continue;

        if (found->second >= geometry_scene_data_info_count)
        {
            log_warning("Implementation issue in DXR. Per vertex scene data ignored due to "
                        "insufficient space in the info-buffer: " + m_name +
                        " Semantic: " + e.semantic);
            continue;
        }

        // if the data is available on this geometry, fill the info
        Scene_data::Info& info =
            scene_data_buffer[geometry_scene_data_info_offset + found->second];
        info.set_kind(Scene_data::Kind::Vertex);
        info.set_byte_offset(e.byte_offset);
        info.set_byte_stride(static_cast<uint16_t>(get_vertex_stride()));
        info.set_interpolation_mode(e.interpolation_mode);
        info.set_uniform(false);

        switch (e.kind)
        {
            case Scene_data::Value_kind::Int:
            case Scene_data::Value_kind::Int2:
            case Scene_data::Value_kind::Int3:
            case Scene_data::Value_kind::Int4:
                info.set_element_type(Scene_data::Element_type::Int);
                break;
            case Scene_data::Value_kind::Float:
            case Scene_data::Value_kind::Vector2:
            case Scene_data::Value_kind::Vector3:
            case Scene_data::Value_kind::Vector4:
                info.set_element_type(Scene_data::Element_type::Float);
                break;
            case Scene_data::Value_kind::Color:
                info.set_element_type(Scene_data::Element_type::Color);
                break;
        }
    }
    // keep this offset, it's required by the renderer
    m_scene_data_info_offset = geometry_scene_data_info_offset;
    return true;
}

// ------------------------------------------------------------------------------------------------

size_t Mesh::Geometry::get_vertex_stride() const
{
    return m_vertex_layout.back().byte_offset + m_vertex_layout.back().element_size;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Mesh::Mesh(
    Base_application* app,
    Raytracing_acceleration_structure* acceleration_structure,
    const IScene_loader::Mesh& mesh_desc)

    : m_app(app)
    , m_name(mesh_desc.name)
    , m_vertex_buffer(new Vertex_buffer<uint8_t>(
        app, mesh_desc.vertex_data.size(), mesh_desc.name + "_VertexBuffer"))
    , m_index_buffer(new Index_buffer(
        app, mesh_desc.indices.size(), mesh_desc.name + "_IndexBuffer"))
    , m_acceleration_structur(acceleration_structure)
    , m_blas()
    , m_geometries()
{
    m_vertex_buffer->set_data(mesh_desc.vertex_data);
    m_index_buffer->set_data(mesh_desc.indices);
    m_blas = m_acceleration_structur->add_bottom_level_structure(mesh_desc.name + "_BLAS");

    m_local_aabb = Bounding_box::Invalid;

    std::vector<Scene_data::Info> primvar_vertex_infos;

    for (size_t i = 0, n = mesh_desc.primitives.size(); i < n; ++i)
    {
        m_geometries.push_back(Mesh::Geometry(app, *this, mesh_desc.primitives[i], i));
        Mesh::Geometry& part = m_geometries.back();

        // add geometry to acceleration structure
        part.m_geometry_handle = m_acceleration_structur->add_geometry(
            m_blas,
            m_vertex_buffer,
            part.get_vertex_buffer_byte_offset(),
            part.get_vertex_count(),
            part.get_vertex_stride(),
            0 /* vertex byte offset */,
            m_index_buffer,
            part.get_index_offset(),
            part.get_index_count());


        // compute local bounding box
        const uint8_t* vertex_buffer_part = mesh_desc.vertex_data.data() +
            part.get_vertex_buffer_byte_offset();
        for (size_t v = 0; v < part.get_vertex_count(); ++v)
        {
            auto vec = reinterpret_cast<const DirectX::XMFLOAT3*>(
                vertex_buffer_part + v * part.get_vertex_stride());
            m_local_aabb.extend(*vec);
        }
    }
}

// ------------------------------------------------------------------------------------------------

Mesh::~Mesh()
{
    delete m_vertex_buffer;
    delete m_index_buffer;
}

// ------------------------------------------------------------------------------------------------

Mesh::Instance::Instance(
    Base_application* app,
    const IScene_loader::Node& node_desc)
    : m_app(app)
    , m_mesh(nullptr)
    , m_instance_handle()
    , m_materials()
    , m_scene_data_infos(nullptr)
    , m_scene_data(node_desc.scene_data)
    , m_scene_data_buffer(nullptr)
{
}

// ------------------------------------------------------------------------------------------------

Mesh::Instance::~Instance()
{
    if (m_scene_data_infos)
        delete m_scene_data_infos;

    if (m_scene_data_buffer)
        delete m_scene_data_buffer;
}

// ------------------------------------------------------------------------------------------------

Mesh::Instance* Mesh::create_instance(const IScene_loader::Node& node_desc)
{
    auto handle = m_acceleration_structur->add_instance(
        m_blas,
        DirectX::XMMatrixIdentity());

    if (!handle.is_valid()) return nullptr;

    Mesh::Instance* instance = new Mesh::Instance(m_app, node_desc);
    instance->m_mesh = this;
    instance->m_instance_handle = handle;
    instance->m_materials.resize(m_geometries.size(), nullptr);
    return instance;
}

// ------------------------------------------------------------------------------------------------

namespace
{

template<typename T>
uint32_t asuint(T value)
{
    union
    {
        uint32_t u;
        T t;
    } data;
    data.t = value;
    return data.u;
}

}

// ------------------------------------------------------------------------------------------------

bool Mesh::Instance::update_scene_data_infos(D3DCommandList* command_list)
{
    std::vector<Scene_data::Info> scene_data_infos;
    std::vector<uint32_t> scene_data;

    // iterate over all mesh parts
    if (!m_mesh->visit_geometries([&](Mesh::Geometry* part)
    {
        // get material for this part of this instance
        const IMaterial* material = get_material(part);

        // map scene data between material and geometry
        const std::unordered_map<std::string, uint32_t>& scene_data_name_map =
            material->get_scene_data_name_map();

        // to keep it simple and assuming there are not that many used scene data names as well
        // as small, ideally densely packed, scene data IDs, a dense map is used to store the
        // mapping infos about the data layout on the GPU
        // so find the maximum ID and use it as size
        uint32_t info_count = 0;
        for (auto& pair : scene_data_name_map)
            info_count = std::max(info_count, pair.second);
        info_count++; // IDs are used as index

        uint32_t info_offset = static_cast<uint32_t>(scene_data_infos.size());
        scene_data_infos.insert(
            scene_data_infos.end(), info_count, Scene_data::Info());

        // update the info block reserved for this geometry
        // this will only affect scene data when the name matches an vertex element semantic
        if (!part->update_scene_data_infos(
            scene_data_name_map, scene_data_infos.data(), info_offset, info_count))
            return false;

        // update the per object/instance scene data info
        for (auto present_data : m_scene_data)
        {
            // check if this data is requested by the material
            auto found = scene_data_name_map.find(present_data.name);
            if (found == scene_data_name_map.end())
                continue;

            // this should only modify infos that currently have kind invalid
            // otherwise the vertex semantic would be overridden
            Scene_data::Info& info = scene_data_infos[info_offset + found->second];
            if (info.get_kind() != Scene_data::Kind::None)
                continue;

            info.set_kind(Scene_data::Kind::Instance);
            info.set_byte_offset(scene_data.size() * sizeof(uint32_t));
            info.set_interpolation_mode(Scene_data::Interpolation_mode::None);
            info.set_uniform(true);

            size_t element_count = 0;
            switch (present_data.kind)
            {
                case Scene_data::Value_kind::Int:
                    element_count = 1;
                    info.set_element_type(Scene_data::Element_type::Int);
                    break;
                case Scene_data::Value_kind::Int2:
                    element_count = 2;
                    info.set_element_type(Scene_data::Element_type::Int);
                    break;
                case Scene_data::Value_kind::Int3:
                    element_count = 3;
                    info.set_element_type(Scene_data::Element_type::Int);
                    break;
                case Scene_data::Value_kind::Int4:
                    element_count = 4;
                    info.set_element_type(Scene_data::Element_type::Int);
                    break;
                case Scene_data::Value_kind::Float:
                    element_count = 1;
                    info.set_element_type(Scene_data::Element_type::Float);
                    break;
                case Scene_data::Value_kind::Vector2:
                    element_count = 2;
                    info.set_element_type(Scene_data::Element_type::Float);
                    break;
                case Scene_data::Value_kind::Vector3:
                    element_count = 3;
                    info.set_element_type(Scene_data::Element_type::Float);
                    break;
                case Scene_data::Value_kind::Vector4:
                    element_count = 4;
                    info.set_element_type(Scene_data::Element_type::Float);
                    break;
                case Scene_data::Value_kind::Color:
                    element_count = 3;
                    info.set_element_type(Scene_data::Element_type::Color);
                    break;
            }
            info.set_byte_stride(element_count * 4);

            // copy the data into the instance_scene_data (and later to the GPU buffer)
            for (size_t i = 0; i < element_count; ++i)
                scene_data.push_back(asuint(present_data.data_int[i]));
        }
        return true;
    })) return false;

    // resize if to small or create if not yet done
    if (m_scene_data_infos && m_scene_data_infos->get_element_count() < scene_data_infos.size())
    {
        delete m_scene_data_infos;
        m_scene_data_infos = nullptr;
    }
    if (!m_scene_data_infos)
    {
        m_scene_data_infos = new Structured_buffer<Scene_data::Info>(
            m_app, scene_data_infos.size(),
            m_mesh->get_name() + "[Instance]_SceneDataInfos");
    }

    // same for the data
    if (m_scene_data_buffer && m_scene_data_buffer->get_element_count() < scene_data.size())
    {
        delete m_scene_data_buffer;
        m_scene_data_buffer = nullptr;
    }
    if (!m_scene_data_buffer && !scene_data.empty())
    {
        m_scene_data_buffer = new Structured_buffer<uint32_t>(
            m_app, scene_data.size(),
            m_mesh->get_name() + "[Instance]_InstanceSceneData");
    }

    // upload only if there is data
    if (!scene_data.empty())
    {
        m_scene_data_buffer->set_data(scene_data);
        if (!m_scene_data_buffer->upload(command_list))
            return false;
    }

    // push info data to the GPU
    m_scene_data_infos->set_data(scene_data_infos);
    return m_scene_data_infos->upload(command_list);
}

// ------------------------------------------------------------------------------------------------

bool Mesh::upload_buffers(D3DCommandList* command_list)
{
    if (!m_vertex_buffer->upload(command_list)) return false;
    if (!m_index_buffer->upload(command_list)) return false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mesh::visit_geometries(std::function<bool(Geometry*)> action)
{
    bool success = true;
    for (auto& g : m_geometries)
    {
        success &= action(&g);
        if (!success)
            break;
    }
    return success;
}

// ------------------------------------------------------------------------------------------------

bool Mesh::visit_geometries(std::function<bool(const Geometry*)> action) const
{
    bool success = true;
    for (auto& g : m_geometries)
    {
        success &= action(&g);
        if (!success)
            break;
    }
    return success;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Camera::Camera(Base_application* app, const IScene_loader::Camera& camera_desc)
    : m_app(app)
    , m_name(camera_desc.name)
    , m_field_of_view(camera_desc.vertical_fov)
    , m_aspect_ratio(camera_desc.aspect_ratio)
    , m_near_plane_distance(camera_desc.near_plane_distance)
    , m_far_plane_distance(camera_desc.far_plane_distance)
    , m_projection_changed(true)
{
    m_constants =
        new Dynamic_constant_buffer<Camera::Constants>(m_app, m_name + "_Constants", 2);
}

// ------------------------------------------------------------------------------------------------

Camera::~Camera()
{
    delete m_constants;
}

// ------------------------------------------------------------------------------------------------

void Camera::update(const DirectX::XMMATRIX& global_transform, bool transform_changed)
{
    if (!m_projection_changed && !transform_changed)
        return;

    if (transform_changed)
    {
        Camera::Constants& data = m_constants->data();
        data.view = DirectX::XMMatrixInverse(nullptr, global_transform);
        data.view_inv = global_transform;
    }

    if (m_projection_changed)
    {
        Camera::Constants& data = m_constants->data();
        data.perspective = DirectX::XMMatrixPerspectiveFovRH(
            m_field_of_view, m_aspect_ratio, m_near_plane_distance, m_far_plane_distance);

        data.perspective_inv =
            DirectX::XMMatrixInverse(nullptr, data.perspective);

        m_projection_changed = false;
    }
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Scene_node::Scene_node(Base_application* app, Scene* scene, Kind kind, const std::string& name)
    : m_app(app)
    , m_kind(kind)
    , m_name(name)
    , m_scene(scene)
    , m_parent(nullptr)
    , m_local_transformation()
    , m_global_transformation(DirectX::XMMatrixIdentity())
    , m_children()
    , m_global_aabb(Bounding_box::Invalid)
    , m_mesh_instance(nullptr)
    , m_camera(nullptr)
{
}

// ------------------------------------------------------------------------------------------------

Scene_node::~Scene_node()
{
    for (size_t c = 0, n = m_children.size(); c < n; ++c)
        delete m_children[c];

    if (m_mesh_instance)
        delete m_mesh_instance;
}

// ------------------------------------------------------------------------------------------------

void Scene_node::add_child(Scene_node* to_add)
{
    m_children.push_back(to_add);
    to_add->m_parent = this;
}

// ------------------------------------------------------------------------------------------------

bool Scene_node::update(const Update_args& args)
{
    bool changed = false;

    // keep current data to check for changes
    DirectX::XMMATRIX previous_trafo = m_global_transformation;
    Bounding_box previous_aabb = m_global_aabb;

    // update own transformation
    if (m_parent)
        m_global_transformation = DirectX::XMMatrixMultiply(
            m_local_transformation.get_matrix(), m_parent->m_global_transformation);
    else
        m_global_transformation = m_local_transformation.get_matrix();

    // update children
    for (auto& c : m_children)
        changed &= c->update(args);

    // update bounding boxes
    // at this point the child boxes are already updated
    update_bounding_volumes();

    // detect changes
    float epsilon = 0.000001f;
    for (size_t c = 0; c < 16; ++c)
        if (std::fabsf(((float*)(&previous_trafo))[c] -
            ((float*)(&m_global_transformation))[c]) > epsilon)
        {
            changed = true;
            break;
        }
    for (size_t c = 0; c < 6; ++c)
        if (std::fabsf(((float*)(&previous_aabb))[c] -
            ((float*)(&m_global_aabb))[c]) > epsilon)
        {
            changed = true;
            break;
        }

    // update node specific properties
    switch (m_kind)
    {
        case Kind::Mesh:
        {
            if(changed)
                m_scene->get_acceleration_structure()->set_instance_transform(
                    m_mesh_instance->get_instance_handle(), m_global_transformation);
            break;
        }

        case Kind::Camera:
        {
            m_camera->update(m_global_transformation, changed);
            break;
        }

        default:
            break;
    }

    return changed;
}

// ------------------------------------------------------------------------------------------------

const Bounding_box& Scene_node::get_local_bounding_box() const
{
    switch (m_kind)
    {
        case Kind::Mesh:
            return m_mesh_instance->get_mesh()->get_local_bounding_box();

        case Kind::Camera:
        default:
            return Bounding_box::Invalid;
    }
}

// ------------------------------------------------------------------------------------------------

void Scene_node::update_bounding_volumes()
{
    // initialize with invalid .. some kind of negative bounding box
    Bounding_box new_global_aabb = Bounding_box::Invalid;

    // merge the ones of all children
    // at this point the child boxes are already updated
    for (auto& child : m_children)
        new_global_aabb.merge(child->m_global_aabb);

    // if this node has no own dimensions we have our result
    // if not we need to account for the own local extend
    const Bounding_box& own_local_aabb = get_local_bounding_box();
    if (own_local_aabb.is_valid())
    {
        Bounding_box own_global_aabb_contrib =
            Bounding_box::transform(own_local_aabb, m_global_transformation);
        new_global_aabb.merge(own_global_aabb_contrib);
    }

    m_global_aabb = new_global_aabb;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

IScene_loader::Scene_options::Scene_options()
    : units_per_meter(1.0f)
    , handle_z_axis_up(false)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Scene::Scene(Base_application* app, const std::string& debug_name, size_t ray_type_count)
    : m_app(app)
    , m_debug_name(debug_name)
    , m_acceleration_structure(
        new Raytracing_acceleration_structure(app, ray_type_count, "AccelerationStructure"))
    , m_root(app, this, Scene_node::Kind::Empty, "Root")
{
}

// ------------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------------

bool Scene::build_scene(const IScene_loader::Scene& scene)
{
    std::unordered_map<size_t, Mesh*> handled_meshes;

    // Loading of materials is done in two steps to be able to parallelize the creation
    // while keeping the order deterministically.
    // Loaded scene materials that share the same material, will also share the same
    // IMaterial instance.
    std::unordered_map<size_t, IMaterial*> handled_materials;
    std::unordered_map<const Mesh::Geometry*, size_t> map_geometry_materials;

    // process the scene graph
    std::function<void(Scene_node&, const IScene_loader::Node&)> visit_loaded_nodes =
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
                        child->m_mesh_instance = it->second->create_instance(src_child);
                        break;
                    }

                    // collect all materials needed
                    auto& primitives = scene.meshes[src_child.index].primitives;
                    for (const auto& p : primitives)
                    {
                        // mesh already handles?
                        auto it = handled_materials.find(p.material);
                        if (it != handled_materials.end())
                            continue;

                        // reserve a place in the map
                        handled_materials[p.material] = nullptr;
                    }

                    // create mesh
                    Mesh* mesh = new Mesh(
                        m_app, m_acceleration_structure, scene.meshes[src_child.index]);
                    child->m_mesh_instance = mesh->create_instance(src_child);
                    handled_meshes[src_child.index] = mesh;

                    // keep track of which geometry uses which material
                    // the material instances themselves are not yet created,
                    // so we store an index only
                    for (size_t p = 0, n = primitives.size(); p < n; ++p)
                        map_geometry_materials[&(mesh->m_geometries[p])] = primitives[p].material;

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
            parent.add_child(child);

            // go down recursively
            for (const auto& c : src_child.children)
                visit_loaded_nodes(*child, c);
        };
    visit_loaded_nodes(m_root, scene.root);
    m_root.set_local_transformation(scene.root.local);

    // create the materials
    // ... in parallel, if not forced otherwise
    std::vector<std::thread> tasks;
    std::atomic_bool success = true;
    std::mutex mtx;
    auto material_library = m_app->get_mdl_sdk().get_library();
    for (auto it = handled_materials.begin(); it != handled_materials.end(); ++it)
    {
        // sequentially
        if (m_app->get_options()->force_single_threading)
        {
            Mdl_material* mdl_material(material_library->create_material());
            if (!material_library->set_description(mdl_material,
                it->first < scene.materials.size()
                    ? Mdl_material_description(scene.materials[it->first])
                    : Mdl_material_description()))
            {
                success.store(false);
                log_error("Failed to create material: " +
                    scene.materials[it->first].name, SRC);
            }
            it->second = mdl_material;
            m_materials.push_back(it->second);
        }
        // asynchronously
        else
        {
            tasks.emplace_back(std::thread([&, it]()
            {
                Mdl_material* mdl_material(material_library->create_material());
            if (!material_library->set_description(mdl_material,
                it->first < scene.materials.size()
                    ? Mdl_material_description(scene.materials[it->first])
                    : Mdl_material_description()))
                {
                    success.store(false);
                    log_error("Failed to create material: " +
                        scene.materials[it->first].name, SRC);
                }
                it->second = mdl_material;

                std::unique_lock<std::mutex> lock(mtx);
                m_materials.push_back(it->second);
            }));
        }
    }

    // wait for all loading tasks
    for (auto& t : tasks)
        t.join();

    // any errors?
    if (!success.load())
        return false;

    // set material pointers
    visit(IScene_loader::Node::Kind::Mesh, [&](Scene_node* node)
    {
        // assign materials to the primitives
        Mesh::Instance* instance = node->get_mesh_instance();
        const Mesh* mesh = instance->get_mesh();

        mesh->visit_geometries([&](const Mesh::Geometry* geometry)
        {
            IMaterial* mat = handled_materials[map_geometry_materials[geometry]];
            instance->set_material(geometry, mat);
            return true;
        });
        return true;
    });

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

    // update transformations and bounding box structure
    update(Update_args{});
    return true;
}

// ------------------------------------------------------------------------------------------------

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
        parent->add_child(node);
    else
        m_root.add_child(node);

    node->update(Update_args{});
    return node;
}

// ------------------------------------------------------------------------------------------------

bool Scene::visit(
    Scene_node::Kind mask, std::function<bool(const Scene_node* node)> action) const
{
    std::function<bool(const Scene_node&)> visit = [&](const Scene_node& node)
    {
        if (mi::examples::enums::has_flag(node.m_kind, mask))
            if(!action(&node))
                return false;

        for (const auto& c : node.m_children)
            if (!visit(*c))
                return false;

        return true;
    };
    return visit(m_root);
}

// ------------------------------------------------------------------------------------------------

bool Scene::visit(
    Scene_node::Kind mask, std::function<bool(Scene_node* node)> action)
{
    std::function<bool(Scene_node&)> visit = [&](Scene_node& node)
    {
        if (mi::examples::enums::has_flag(node.m_kind, mask))
            if (!action(&node))
                return false;

        for (const auto& c : node.m_children)
            if (!visit(*c))
                return false;

        return true;
    };
    return visit(m_root);
}

}}} // mi::examples::mdl_d3d12
