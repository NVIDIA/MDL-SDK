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

#include "gltf.h"
#include <fx/gltf.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{
size_t get_vertex_count(const fx::gltf::Document& doc, const fx::gltf::Primitive& primitive)
{
    auto att = primitive.attributes.find("POSITION");
    auto acc = doc.accessors[att->second];
    return acc.count;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
const T read(const uint8_t* p_data)
{
    return *reinterpret_cast<const T*>(p_data);
}

// ------------------------------------------------------------------------------------------------

const uint8_t* get_index_data(
    const fx::gltf::Document& doc,
    const fx::gltf::Primitive& primitive,
    size_t& out_index_count,
    size_t& out_stride)
{
    if (primitive.indices < 0 || primitive.indices >= doc.accessors.size())
        return nullptr;

    const auto& acc = doc.accessors[primitive.indices];
    out_index_count = acc.count;

    if (acc.componentType == fx::gltf::Accessor::ComponentType::UnsignedInt)
    {
        const auto& bv = doc.bufferViews[acc.bufferView];
        const auto& buf = doc.buffers[bv.buffer];
        out_stride = bv.byteStride == 0 ? sizeof(uint32_t) : bv.byteStride;
        return buf.data.data() + bv.byteOffset + acc.byteOffset;
    }
    else if(acc.componentType == fx::gltf::Accessor::ComponentType::UnsignedShort)
    {
        const auto& bv = doc.bufferViews[acc.bufferView];
        const auto& buf = doc.buffers[bv.buffer];
        out_stride = bv.byteStride == 0 ? sizeof(uint16_t) : bv.byteStride;
        return buf.data.data() + bv.byteOffset + acc.byteOffset;
    }
    else if (acc.componentType == fx::gltf::Accessor::ComponentType::UnsignedByte)
    {
        const auto& bv = doc.bufferViews[acc.bufferView];
        const auto& buf = doc.buffers[bv.buffer];
        out_stride = bv.byteStride == 0 ? sizeof(uint8_t) : bv.byteStride;
        return buf.data.data() + bv.byteOffset + acc.byteOffset;
    }

    assert(false || "Unsupported index format");
    return nullptr;
}

// ------------------------------------------------------------------------------------------------

void apply_transform(
    Transform& target,
    const fx::gltf::Node& source,
    const IScene_loader::Scene_options& options)
{
    // check the quaternion to see if translation, rotation, scale are used
    if (source.rotation[0] != 0 ||
        source.rotation[1] != 0 ||
        source.rotation[2] != 0 ||
        source.rotation[3] != 0)
    {

        target.translation =
            {source.translation[0], source.translation[1], source.translation[2]};
        target.rotation =
            {source.rotation[0], source.rotation[1], source.rotation[2], source.rotation[3]};
        target.scale =
            {source.scale[0], source.scale[1], source.scale[2]};
    }

    // also try the matrix form
    if (target.is_identity())
    {
        DirectX::XMMATRIX m(source.matrix.data());

        if (options.handle_z_axis_up)
        {
            DirectX::XMMATRIX flip = DirectX::XMMatrixIdentity();
            flip.r[1].m128_f32[1] = 0.0f;
            flip.r[1].m128_f32[2] = -1.0f;
            flip.r[2].m128_f32[1] = 1.0f;
            flip.r[2].m128_f32[2] = 0.0f;
            m = DirectX::XMMatrixMultiply(flip, m);
        }
        Transform from_matrix;
        if (Transform::try_from_matrix(m, from_matrix) || !from_matrix.is_identity())
            target = from_matrix; // use identity
    }

    float scale = 1.0f / options.units_per_meter;
    target.translation = {
        target.translation.x * scale,
        target.translation.y * scale,
        target.translation.z * scale
    };
}

// ------------------------------------------------------------------------------------------------

size_t get_vertex_stride(IScene_loader::Primitive& part)
{
    return part.vertex_element_layout.size() == 0 ? 0 :
        part.vertex_element_layout.back().byte_offset +
        part.vertex_element_layout.back().element_size;
}

// ------------------------------------------------------------------------------------------------

void add_vertex_element(
    IScene_loader::Primitive& part,
    const std::string& semantic,
    mdl_d3d12::Scene_data::Value_kind kind)
{

    IScene_loader::Vertex_element element;
    element.byte_offset = get_vertex_stride(part);

    switch (kind)
    {
        case mdl_d3d12::Scene_data::Value_kind::Float:
            element.element_size = 1 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Linear;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Vector2:
            element.element_size = 2 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Linear;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Vector3:
        case mdl_d3d12::Scene_data::Value_kind::Color:
            element.element_size = 3 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Linear;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Vector4:
            element.element_size = 4 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Linear;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Int:
            element.element_size = 1 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Nearest;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Int2:
            element.element_size = 2 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Nearest;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Int3:
            element.element_size = 3 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Nearest;
            break;
        case mdl_d3d12::Scene_data::Value_kind::Int4:
            element.element_size = 4 * 4;
            element.interpolation_mode = Scene_data::Interpolation_mode::Nearest;
            break;
        default:
            log_error("Vertex element kind not handled.", SRC);
            return;
    }

    element.semantic = semantic;
    element.kind = kind;
    element.layout_index = part.vertex_element_layout.size();
    part.vertex_element_layout.push_back(element);
}

// ------------------------------------------------------------------------------------------------

void guess_tangent(const DirectX::XMFLOAT3& normal, DirectX::XMFLOAT3& out_tangent)
{
    const float yz = -normal.y * normal.z;
    out_tangent = (std::fabsf(normal.z) > 0.99999f)
        ? DirectX::XMFLOAT3(-normal.x*normal.y, 1.0f - normal.y*normal.y, yz)
        : DirectX::XMFLOAT3(-normal.x*normal.z, yz, 1.0f - normal.z*normal.z);

    out_tangent = normalize(out_tangent);
}

// ------------------------------------------------------------------------------------------------

void compute_normals(
    IScene_loader::Primitive part,
    uint8_t* vertex_buffer_part,
    uint32_t* indices,
    size_t index_count)
{
    size_t vertex_stride = get_vertex_stride(part);
    DirectX::XMFLOAT3 zero = {0.0f, 0.0f, 0.0f};
    std::vector<DirectX::XMFLOAT3> normals(part.vertex_count, zero);

    for (size_t i = 0; i < index_count; i += 3)
    {
        uint32_t ai = indices[i + 0];
        uint32_t bi = indices[i + 1];
        uint32_t ci = indices[i + 2];

        auto pos_a = *reinterpret_cast<DirectX::XMFLOAT3*>(
            vertex_buffer_part + ai * vertex_stride);

        auto pos_b = *reinterpret_cast<DirectX::XMFLOAT3*>(
            vertex_buffer_part + bi * vertex_stride);

        auto pos_c = *reinterpret_cast<DirectX::XMFLOAT3*>(
            vertex_buffer_part + ci * vertex_stride);

        DirectX::XMFLOAT3 ab = pos_b - pos_a;
        DirectX::XMFLOAT3 ac = pos_c - pos_a;
        DirectX::XMFLOAT3 bc = pos_c - pos_b;
        DirectX::XMFLOAT3 weighted_face_normal = cross(ab, ac);

        // area weight
        /*
        float area_weight = length(weighted_face_normal);
        weighted_face_normal *= 1.0f / area_weight;
        normals[ai] += weighted_face_normal;
        normals[bi] += weighted_face_normal;
        normals[ci] += weighted_face_normal;
        */

        // angle weight

        ab *= 1.0f / length(ab);
        ac *= 1.0f / length(ac);
        bc *= 1.0f / length(bc);
        weighted_face_normal *= 0.5f / length(weighted_face_normal);

        float angle_weight_a = acos(dot(ab, ac));
        float angle_weight_b = acos(dot(-ab, bc));
        float angle_weight_c = acos(dot(-ac, -bc));
        normals[ai] += (weighted_face_normal * angle_weight_a);
        normals[bi] += (weighted_face_normal * angle_weight_b);
        normals[ci] += (weighted_face_normal * angle_weight_c);
    }

    assert(part.vertex_element_layout[1].semantic == "NORMAL");
    size_t normal_offset = part.vertex_element_layout[1].byte_offset;
    for (size_t i = 0; i < part.vertex_count; ++i)
    {
        auto& v_n = *reinterpret_cast<DirectX::XMFLOAT3*>(
            vertex_buffer_part + i * vertex_stride + normal_offset);

        // data is present and probably okay
        if (length2(v_n) > 0.9f)
            continue;

        v_n = length2(normals[i]) > 0.001f ? normalize(normals[i]) : zero;
    }
}

// ------------------------------------------------------------------------------------------------

void compute_tangent_frame(
    IScene_loader::Primitive part,
    uint8_t* vertex_buffer_part,
    uint32_t* indices,
    size_t index_count)
{
    // TODO fix this, its not working correctly

    size_t vertex_stride = get_vertex_stride(part);

    DirectX::XMFLOAT3 zero = {0.0f, 0.0f, 0.0f};
    std::vector<DirectX::XMFLOAT3> tan0(part.vertex_count, zero);
    std::vector<DirectX::XMFLOAT3> tan1(part.vertex_count, zero);


    int texcoord_layout_index = -1;
    for (size_t s = 0, sn = part.vertex_element_layout.size(); s < sn; ++s)
        if (part.vertex_element_layout[s].semantic == "TEXCOORD_0")
        {
            texcoord_layout_index = s;
            break;
        }

    if (texcoord_layout_index >= 0)
    {
        size_t tex_coord_offset = part.vertex_element_layout[texcoord_layout_index].byte_offset;
        for (size_t i = 0; i < index_count; i += 3)
        {
            uint32_t ai = indices[i + 0];
            uint32_t bi = indices[i + 1];
            uint32_t ci = indices[i + 2];

            auto pos_a = *reinterpret_cast<DirectX::XMFLOAT3*>(
                vertex_buffer_part + ai * vertex_stride);

            auto pos_b = *reinterpret_cast<DirectX::XMFLOAT3*>(
                vertex_buffer_part + bi * vertex_stride);

            auto pos_c = *reinterpret_cast<DirectX::XMFLOAT3*>(
                vertex_buffer_part + ci * vertex_stride);

            auto texcoord_a = *reinterpret_cast<DirectX::XMFLOAT2*>(
                vertex_buffer_part + ai * vertex_stride + tex_coord_offset);

            auto texcoord_b = *reinterpret_cast<DirectX::XMFLOAT2*>(
                vertex_buffer_part + bi * vertex_stride + tex_coord_offset);

            auto texcoord_c = *reinterpret_cast<DirectX::XMFLOAT2*>(
                vertex_buffer_part + ci * vertex_stride + tex_coord_offset);

            float x1 = pos_b.x - pos_a.x;
            float x2 = pos_c.x - pos_a.x;
            float y1 = pos_b.y - pos_a.y;
            float y2 = pos_c.y - pos_a.y;
            float z1 = pos_b.z - pos_a.z;
            float z2 = pos_c.z - pos_a.z;

            float s1 = texcoord_b.x - texcoord_a.x;
            float s2 = texcoord_c.x - texcoord_a.x;
            float t1 = (1.0f - texcoord_b.y) - (1.0f - texcoord_a.y);
            float t2 = (1.0f - texcoord_c.y) - (1.0f - texcoord_a.y);

            if ((s1 * s1 + t1 * t1) < 0.000001f && (s2 * s2 + t2 * t2) < 0.000001f)
                continue;

            if ((s1 * t2 - s2 * t1) < 0.000001f)
                continue;

            float r = 1.0f / (s1 * t2 - s2 * t1);
            DirectX::XMFLOAT3 sdir =
            {(t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r};
            DirectX::XMFLOAT3 tdir =
            {(s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r};

            tan0[ai] += sdir;
            tan0[bi] += sdir;
            tan0[ci] += sdir;

            tan1[ai] += tdir;
            tan1[bi] += tdir;
            tan1[ci] += tdir;
        }
    }

    assert(part.vertex_element_layout[1].semantic == "NORMAL");
    assert(part.vertex_element_layout[2].semantic == "TANGENT");
    size_t normal_offset = part.vertex_element_layout[1].byte_offset;
    size_t tangent_offset = part.vertex_element_layout[2].byte_offset;


    for (long a = 0; a < part.vertex_count; a++)
    {
        auto& v_t3 = *reinterpret_cast<DirectX::XMFLOAT3*>(
            vertex_buffer_part + a * vertex_stride + tangent_offset);
        auto& v_t4 = *reinterpret_cast<DirectX::XMFLOAT4*>(
            vertex_buffer_part + a * vertex_stride + tangent_offset);

        // data is present and probably okay
        if (length2(v_t3) > 0.9f && abs(v_t4.w) > 0.9f)
            continue;

        auto& v_n = *reinterpret_cast<DirectX::XMFLOAT3*>(
            vertex_buffer_part + a * vertex_stride + normal_offset);

        const DirectX::XMFLOAT3& t1 = tan0[a];
        const DirectX::XMFLOAT3& t2 = tan1[a];
        DirectX::XMFLOAT3 t;
        float sign = 1.0f;

        if (texcoord_layout_index < 0 || length2(t1) < 0.001f || length2(t2) < 0.001f)
        {
            // arbitrary tangent that is orthogonal to the normal
            guess_tangent(v_n, t);
        }
        else
        {
            // Gram-Schmidt
            t = t1 - (v_n * dot(v_n, t1));
            if (length2(t) > 0.0001f)
            {
                // get sign
                sign = (dot(cross(v_n, t1), t2) < 0.0f) ? -1.0f : 1.0f;
                t = normalize(t);
            }
            else
                guess_tangent(v_n, t);
        }
        v_t3 = t;
        v_t4.w = sign;
    }
}

// ------------------------------------------------------------------------------------------------

std::string get_texture_uri(
    const fx::gltf::Document& doc, const fx::gltf::Material::Texture& tex)
{
    int32_t tex_index = tex.index;
    if (tex_index < 0) return "";
    int32_t src_index = doc.textures[tex_index].source;
    if (src_index < 0) return "";
    return doc.images[src_index].uri;
}

// ------------------------------------------------------------------------------------------------

// non-standardized extra fields on scene nodes allow per mesh-instance scene data
std::vector<Scene_data::Value> read_scene_data(const nlohmann::json& json)
{
    std::vector<Scene_data::Value> result;
    if (json.empty() || !json.contains("extras"))
        return result;

    auto extras = json["extras"];
    if (!extras.is_object() || !extras.contains("sceneData"))
        return result;

    auto scene_data = extras["sceneData"];
    if (!scene_data.is_array())
        return result;

    for (auto it = scene_data.begin(); it != scene_data.end(); ++it)
    {
        if (it.value().is_object() &&
            it.value().contains("name") &&
            it.value().contains("value"))
        {
            auto name_obj = it.value()["name"];
            if (name_obj.type() != nlohmann::json::value_t::string)
                continue;

            Scene_data::Value value;
            value.name = name_obj.get<std::string>();

            auto value_obj = it.value()["value"];
            auto value_type = value_obj.type();

            switch (value_obj.type())
            {
            case nlohmann::json::value_t::boolean:
                value.data_int[0] = value_obj.get<bool>() ? 1 : 0;
                value.kind = Scene_data::Value_kind::Int;
                break;

            case nlohmann::json::value_t::number_float:
                value.data_float[0] = value_obj.get<float>();
                value.kind = Scene_data::Value_kind::Float;
                break;

            case nlohmann::json::value_t::number_integer:
                value.data_int[0] = value_obj.get<int32_t>();
                value.kind = Scene_data::Value_kind::Int;
                break;

            case nlohmann::json::value_t::number_unsigned:
                value.data_int[0] = static_cast<int32_t>(value_obj.get<uint32_t>());
                value.kind = Scene_data::Value_kind::Int;
                break;

            case nlohmann::json::value_t::array:
                size_t array_size = value_obj.size();
                if (array_size < 1 || array_size > 4)
                    continue;

                auto element_type = value_obj.begin().value().type();
                if (element_type != nlohmann::json::value_t::number_float &&
                    element_type != nlohmann::json::value_t::number_integer &&
                    element_type != nlohmann::json::value_t::number_unsigned)
                    continue;

                switch (element_type)
                {
                case nlohmann::json::value_t::number_float:
                {
                    int i = 0;
                    for (auto e = value_obj.begin(); e != value_obj.end(); ++e, ++i)
                        value.data_float[i] = e.value().get<float>();

                    switch (array_size)
                    {
                    case 1: value.kind = Scene_data::Value_kind::Float; break;
                    case 2: value.kind = Scene_data::Value_kind::Vector2; break;
                    case 3: value.kind = Scene_data::Value_kind::Vector3; break;
                    case 4: value.kind = Scene_data::Value_kind::Vector4; break;
                    }
                    break;
                }

                case nlohmann::json::value_t::number_integer:
                {
                    int i = 0;
                    for (auto e = value_obj.begin(); e != value_obj.end(); ++e, ++i)
                        value.data_int[i] = e.value().get<int32_t>();

                    switch (array_size)
                    {
                    case 1: value.kind = Scene_data::Value_kind::Int; break;
                    case 2: value.kind = Scene_data::Value_kind::Int2; break;
                    case 3: value.kind = Scene_data::Value_kind::Int3; break;
                    case 4: value.kind = Scene_data::Value_kind::Int4; break;
                    }
                    break;
                }

                case nlohmann::json::value_t::number_unsigned:
                {
                    int i = 0;
                    for (auto e = value_obj.begin(); e != value_obj.end(); ++e, ++i)
                        value.data_int[i] = static_cast<int32_t>(value_obj.get<uint32_t>());

                    switch (array_size)
                    {
                    case 1: value.kind = Scene_data::Value_kind::Int; break;
                    case 2: value.kind = Scene_data::Value_kind::Int2; break;
                    case 3: value.kind = Scene_data::Value_kind::Int3; break;
                    case 4: value.kind = Scene_data::Value_kind::Int4; break;
                    }
                    break;
                }
                }
                break;
            }
            result.push_back(value);
        }
    }
    return result;
}

// ------------------------------------------------------------------------------------------------

void add_clear_coat(
    const fx::gltf::Document& doc,
    IScene_loader::Material::Model_data_materials_clearcoat& material_clearcoat,
    const fx::gltf::Material::KHR_MaterialsClearcoat& gltf_clearcoat)
{
    if (gltf_clearcoat.empty())
        return;

    material_clearcoat.clearcoat_factor = gltf_clearcoat.clearcoatFactor;
    material_clearcoat.clearcoat_texture =
        get_texture_uri(doc, gltf_clearcoat.clearcoatTexture);
    material_clearcoat.clearcoat_roughness_factor = gltf_clearcoat.clearcoatRoughnessFactor;
    material_clearcoat.clearcoat_roughness_texture =
        get_texture_uri(doc, gltf_clearcoat.clearcoatRoughnessTexture);
    material_clearcoat.clearcoat_normal_texture =
        get_texture_uri(doc, gltf_clearcoat.clearcoatNormalTexture);
}

} // anonymous

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

bool Loader_gltf::load(const std::string& file_name, const Scene_options& options)
{
    m_scene = {};
    m_scene.root.kind = Node::Kind::Empty;
    m_scene.root.index = static_cast<size_t>(-1);

    // extend quotas to 2 GB
    fx::gltf::ReadQuotas quotas;
    quotas.MaxFileSize *= 64;
    quotas.MaxBufferByteLength *= 64;

    fx::gltf::Document doc;
    try
    {
        if (mi::examples::strings::ends_with(file_name, ".glb"))
            doc = fx::gltf::LoadFromBinary(file_name, quotas);
        else
            doc = fx::gltf::LoadFromText(file_name, quotas);
    }
    catch (std::exception &ex)
    {
        log_error(ex, SRC);
        return false;
    }

    // process all meshes
    for (const auto& m : doc.meshes)
    {
        Mesh mesh;
        for (const auto& p : m.primitives)
        {
            if (p.mode != fx::gltf::Primitive::Mode::Triangles)
                continue;

            struct input_data
            {
                const uint8_t* base_element;
                size_t element_offset;
            };

            Primitive part;
            part.material = p.material;
            part.vertex_buffer_byte_offset = mesh.vertex_data.size();
            part.vertex_count = get_vertex_count(doc, p);
            part.index_offset = mesh.indices.size();

            // construct the vertex layout for this primitive
            // ----------------------------------------

            // always have position, normal and tangent first as these are mandatory for
            // the renderer
            add_vertex_element(part, "POSITION", Scene_data::Value_kind::Vector3);
            add_vertex_element(part, "NORMAL", Scene_data::Value_kind::Vector3);
            add_vertex_element(part, "TANGENT", Scene_data::Value_kind::Vector4);
            std::vector<input_data> input(3, input_data{nullptr, 0});

            // iterate over the available other semantics
            for (const auto& att : p.attributes)
            {
                const fx::gltf::Accessor& acc = doc.accessors[att.second];
                const fx::gltf::BufferView& bv = doc.bufferViews[acc.bufferView];
                const fx::gltf::Buffer& b = doc.buffers[bv.buffer];

                size_t layout_index;
                if      (att.first == "POSITION")   layout_index = 0;
                else if (att.first == "NORMAL")     layout_index = 1;
                else if (att.first == "TANGENT")    layout_index = 2;
                else
                {
                    switch (acc.type)
                    {
                        case fx::gltf::Accessor::Type::Scalar:
                            add_vertex_element(part, att.first, Scene_data::Value_kind::Float);
                            break;
                        case fx::gltf::Accessor::Type::Vec2:
                            add_vertex_element(part, att.first, Scene_data::Value_kind::Vector2);
                            break;
                        case fx::gltf::Accessor::Type::Vec3:
                            add_vertex_element(part, att.first, Scene_data::Value_kind::Vector3);
                            break;
                        case fx::gltf::Accessor::Type::Vec4:
                            add_vertex_element(part, att.first, Scene_data::Value_kind::Vector4);
                            break;
                        case fx::gltf::Accessor::Type::Mat2:
                        case fx::gltf::Accessor::Type::Mat3:
                        case fx::gltf::Accessor::Type::Mat4:
                        case fx::gltf::Accessor::Type::None:
                        default:
                            log_error("GLTF accessor type not handled.", SRC);
                            continue;
                    }

                    layout_index = input.size();
                    input.push_back(input_data{nullptr, 0});
                }

                // input data alignment
                input[layout_index].base_element =
                    b.data.data() + bv.byteOffset + acc.byteOffset;

                input[layout_index].element_offset = (bv.byteStride == 0)
                    ? part.vertex_element_layout[layout_index].element_size
                    : bv.byteStride;
            }

            // allocate a buffer for this part, vertex count * vertex size
            size_t vertex_stride = get_vertex_stride(part);
            std::vector<uint8_t> vertex_buffer_part(part.vertex_count * vertex_stride, 0);

            // iterate over the semantics
            bool found_normals = false;
            bool fix_normals = false;
            bool found_tangents = false;
            bool fix_tangents = false;
            for (size_t s = 0, sn = part.vertex_element_layout.size(); s < sn; ++s)
            {
                bool is_position = part.vertex_element_layout[s].semantic == "POSITION";
                bool is_normal = part.vertex_element_layout[s].semantic == "NORMAL";
                bool is_tangent = part.vertex_element_layout[s].semantic == "TANGENT";

                // skip if the value is missing, can happen for e.g. for tangents
                if (input[s].base_element == nullptr)
                    continue;

                // iterate over the vertices
                for (size_t i = 0, in = part.vertex_count; i < in; ++i)
                {
                    uint8_t* dest_ptr = vertex_buffer_part.data() +
                        part.vertex_element_layout[s].byte_offset +
                        vertex_stride * i;

                    memcpy(
                        dest_ptr,
                        input[s].base_element + input[s].element_offset * i,
                        part.vertex_element_layout[s].element_size);

                    // special handling of mandatory data
                    if ((is_position || is_normal || is_tangent) && options.handle_z_axis_up)
                    {
                        auto vec = reinterpret_cast<DirectX::XMFLOAT3*>(dest_ptr);
                        *vec = {vec->x, -vec->z, vec->y};
                    }
                    if (is_position && options.units_per_meter != 1.0f)
                    {
                        float scale = 1.0f / options.units_per_meter;
                        auto vec = reinterpret_cast<DirectX::XMFLOAT3*>(dest_ptr);
                        vec->x *= scale;
                        vec->y *= scale;
                        vec->z *= scale;
                    }
                    if (is_normal)
                    {
                        found_normals = true;
                        auto vec = reinterpret_cast<DirectX::XMFLOAT3*>(dest_ptr);
                        if (length2(*vec) > 0.01)
                            *vec = normalize(*vec);
                        else
                            fix_normals = true;
                    }
                    if (is_tangent)
                    {
                        found_tangents = true;
                        auto vec4 = reinterpret_cast<DirectX::XMFLOAT4*>(dest_ptr);
                        auto vec3 = reinterpret_cast<DirectX::XMFLOAT3*>(dest_ptr);
                        if (length2(*vec3) > 0.01 && fabsf(vec4->w) > 0.5f)
                            *vec3 = normalize(*vec3);
                        else
                            fix_tangents = true;

                    }
                }
            }


            // get or generate index data
            size_t index_count, stride_index;
            auto p_first_index = get_index_data(doc, p, index_count, stride_index);

            if (p_first_index)
            {
                part.index_count = index_count;
                if (stride_index == sizeof(uint32_t))
                {
                    for (size_t i = 0; i < part.index_count; ++i)
                        mesh.indices.push_back(
                            static_cast<uint32_t>(
                                read<uint32_t>(p_first_index + i * stride_index)));
                }
                else if (stride_index == sizeof(uint16_t))
                {
                    for (size_t i = 0; i < part.index_count; ++i)
                        mesh.indices.push_back(
                            static_cast<uint32_t>(
                                read<uint16_t>(p_first_index + i * stride_index)));
                }
                else if (stride_index == sizeof(uint8_t))
                {
                    for (size_t i = 0; i < part.index_count; ++i)
                        mesh.indices.push_back(
                            static_cast<uint32_t>(
                                read<uint8_t>(p_first_index + i * stride_index)));
                }
                else
                {
                    log_error("Index format not supported.", SRC);
                    return false;
                }
            }
            else
            {
                part.index_count = part.vertex_count;
                for (size_t i = 0; i < part.index_count; ++i)
                    mesh.indices.push_back(static_cast<uint32_t>(i));
            }

            // generate normals if not present (very simple)
            if (!found_normals || fix_normals)
            {
                compute_normals(
                    part, vertex_buffer_part.data(),
                    mesh.indices.data() + part.index_offset, part.index_count);
            }

            // generate tangents if not present (simple)
            if (!found_tangents || fix_tangents)
            {
                compute_tangent_frame(
                    part, vertex_buffer_part.data(),
                    mesh.indices.data() + part.index_offset, part.index_count);
            }

            // copy vertex buffer to mesh
            mesh.vertex_data.insert(
                mesh.vertex_data.end(),
                vertex_buffer_part.begin(),
                vertex_buffer_part.end());

            mesh.primitives.push_back(std::move(part));
        }

            m_scene.meshes.push_back(std::move(mesh));
    }

    // process all cameras
    for (const auto& c : doc.cameras)
    {
        if (c.type == fx::gltf::Camera::Type::Perspective)
        {
            Camera cam;
            cam.name = c.name;
            cam.aspect_ratio =
                c.perspective.aspectRatio > 0.0f ? c.perspective.aspectRatio : (16.0f / 9.0f);

            cam.vertical_fov = c.perspective.yfov;
            cam.near_plane_distance = c.perspective.znear > 0.0f ? c.perspective.znear : 0.01f;
            cam.far_plane_distance = c.perspective.zfar > 0.0f ? c.perspective.zfar : 1000.0f;
            m_scene.cameras.push_back(std::move(cam));
        }
    }

    // process all materials
    for (const auto& m : doc.materials)
    {
        Material mat;
        mat.name = std::string(m.name);

        // KHR specular glossiness ?
        if (!m.pbrSpecularGlossiness.empty())
        {
            mat.pbr_model = Material::Pbr_model::Khr_specular_glossiness;

            mat.khr_specular_glossiness.diffuse_texture =
                get_texture_uri(doc, m.pbrSpecularGlossiness.diffuseTexture);
            mat.khr_specular_glossiness.diffuse_factor = {
                m.pbrSpecularGlossiness.diffuseFactor[0],
                m.pbrSpecularGlossiness.diffuseFactor[1],
                m.pbrSpecularGlossiness.diffuseFactor[2],
                m.pbrSpecularGlossiness.diffuseFactor[3] };

            mat.khr_specular_glossiness.specular_glossiness_texture =
                get_texture_uri(doc, m.pbrSpecularGlossiness.specularGlossinessTexture);
            mat.khr_specular_glossiness.specular_factor = {
                m.pbrSpecularGlossiness.specularFactor[0],
                m.pbrSpecularGlossiness.specularFactor[1],
                m.pbrSpecularGlossiness.specularFactor[2]};
            mat.khr_specular_glossiness.glossiness_factor =
                m.pbrSpecularGlossiness.glossinessFactor;

            // no clear coat as defined by the extension spec
        }
        // metallic roughness (Default)
        else
        {
            mat.pbr_model = Material::Pbr_model::Metallic_roughness;

            mat.metallic_roughness.base_color_texture =
                get_texture_uri(doc, m.pbrMetallicRoughness.baseColorTexture);
            mat.metallic_roughness.base_color_factor = {
                m.pbrMetallicRoughness.baseColorFactor[0],
                m.pbrMetallicRoughness.baseColorFactor[1],
                m.pbrMetallicRoughness.baseColorFactor[2],
                m.pbrMetallicRoughness.baseColorFactor[3] };

            mat.metallic_roughness.metallic_roughness_texture =
                get_texture_uri(doc, m.pbrMetallicRoughness.metallicRoughnessTexture);
            mat.metallic_roughness.metallic_factor = m.pbrMetallicRoughness.metallicFactor;
            mat.metallic_roughness.roughness_factor = m.pbrMetallicRoughness.roughnessFactor;

            add_clear_coat(doc, mat.metallic_roughness.clearcoat, m.materialsClearcoat);
        }

        mat.normal_texture = get_texture_uri(doc, m.normalTexture);
        mat.normal_scale_factor = m.normalTexture.scale;

        mat.occlusion_texture = get_texture_uri(doc, m.occlusionTexture);
        mat.occlusion_strength = m.occlusionTexture.strength;

        mat.emissive_texture = get_texture_uri(doc, m.emissiveTexture);
        mat.emissive_factor = {
            m.emissiveFactor[0],
            m.emissiveFactor[1],
            m.emissiveFactor[2]};

        switch(m.alphaMode)
        {
            case fx::gltf::Material::AlphaMode::Blend:
                mat.alpha_mode = Material::Alpha_mode::Blend;
                break;

            case fx::gltf::Material::AlphaMode::Mask:
                mat.alpha_mode = Material::Alpha_mode::Mask;
                break;

            default:
                mat.alpha_mode = Material::Alpha_mode::Opaque;
                break;
        }
        mat.alpha_cutoff = m.alphaCutoff;

        mat.single_sided = !m.doubleSided;

        m_scene.materials.push_back(std::move(mat));
    }

    // process the scene graph
    std::function<void(Node&, const fx::gltf::Node&)> visit =
        [&](Node& parent, const fx::gltf::Node& src_child)
        {
            Node node;
            node.name = src_child.name;
            node.kind = Node::Kind::Empty;
            node.index = static_cast<size_t>(-1);
            apply_transform(node.local, src_child, options);

            if (src_child.mesh >= 0 || src_child.mesh < doc.meshes.size())
            {
                node.index = src_child.mesh;
                bool empty = m_scene.meshes[node.index].primitives.size() == 0;

                node.kind = empty ? Node::Kind::Empty : Node::Kind::Mesh;
                if (m_scene.meshes[node.index].name.empty())
                    m_scene.meshes[node.index].name = node.name +
                        (empty ? "_Node" : "_Mesh");
            }

            if (src_child.camera >= 0 || src_child.camera < doc.cameras.size())
            {
                node.kind = Node::Kind::Camera;
                node.index = src_child.camera;
                if (m_scene.cameras[node.index].name.empty())
                    m_scene.cameras[node.index].name = node.name + "_Camera";
            }

            // read (non-standardized) scene data from the extra fields
            node.scene_data = read_scene_data(src_child.extensionsAndExtras);

            // go down recursively
            for (const auto& c : src_child.children)
                visit(node, doc.nodes[c]);

            parent.children.push_back(std::move(node));
        };

    auto s = doc.scenes[doc.scene]; // default scene
    for (const auto& n : s.nodes)
        visit(m_scene.root, doc.nodes[n]);

    return true;
}

// ------------------------------------------------------------------------------------------------

void Loader_gltf::replace_all_materials(const std::string & mdl_name)
{
    m_scene.materials.clear();

    m_scene.materials.emplace_back(IScene_loader::Material());
    m_scene.materials.back().name = mdl_name;

    for (auto& m : m_scene.meshes)
        for (auto& p : m.primitives)
            p.material = 0;
}

}}} // mi::examples::mdl_d3d12
