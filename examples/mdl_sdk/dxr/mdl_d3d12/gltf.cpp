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

#include "gltf.h"
#include <fx/gltf.h>


namespace mdl_d3d12
{

namespace
{
    size_t get_vertex_count(const fx::gltf::Document& doc, const fx::gltf::Primitive& primitive)
    {
        auto att = primitive.attributes.find("POSITION");
        auto acc = doc.accessors[att->second];
        return acc.count;
    }

    template<typename T>
    const T read(const uint8_t* p_data)
    {
        return *reinterpret_cast<const T*>(p_data);
    }

    template<typename T>
    const uint8_t* get_vertex_data(
        const fx::gltf::Document& doc, 
        const fx::gltf::Primitive& primitive, 
        const std::string& semantic, 
        size_t& out_stride)
    {
        const auto& att = primitive.attributes.find(semantic);
        if (att == primitive.attributes.end())
            return nullptr;

        const auto& acc = doc.accessors[att->second];
        const auto& bv = doc.bufferViews[acc.bufferView];
        const auto& buf = doc.buffers[bv.buffer];
        out_stride = bv.byteStride == 0 ? sizeof(T) : bv.byteStride;
        return buf.data.data() + bv.byteOffset + acc.byteOffset;
    }

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

    void apply_transform(Transform& target, const fx::gltf::Node& source)
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
            Transform from_matrix;
            if (Transform::try_from_matrix(m, from_matrix) || !from_matrix.is_identity())
                target = from_matrix; // use identity
        }
    }


    DirectX::XMFLOAT3 normalize(const DirectX::XMFLOAT3& v)
    {
        float inv_length = 1.0f / std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return {v.x * inv_length, v.y * inv_length, v.z * inv_length};
    }

    DirectX::XMFLOAT3 cross(const DirectX::XMFLOAT3& v1, const DirectX::XMFLOAT3& v2)
    {
        DirectX::XMFLOAT3 res;
        res.x = (v1.y * v2.z) - (v1.z * v2.y);
        res.y = (v1.z * v2.x) - (v1.x * v2.z);
        res.z = (v1.x * v2.y) - (v1.y * v2.x);
        return res;
    }

    void guess_tangent(const DirectX::XMFLOAT3& normal, DirectX::XMFLOAT3& out_tangent)
    {
        const float yz = -normal.y * normal.z;
        out_tangent = (std::fabsf(normal.z) > 0.99999f)
            ? DirectX::XMFLOAT3(-normal.x*normal.y, 1.0f - normal.y*normal.y, yz)
            : DirectX::XMFLOAT3(-normal.x*normal.z, yz, 1.0f - normal.z*normal.z);

        out_tangent = normalize(out_tangent);
    }

    void compute_tangent_frame(
        Vertex* vertices, 
        size_t vertex_offset, 
        size_t vertex_count, 
        uint32_t* indices, 
        size_t index_count)
    {
        // TODO fix this, its not working correctly

        DirectX::XMVECTOR zero = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<DirectX::XMVECTOR> tan0(vertex_count, zero);
        std::vector<DirectX::XMVECTOR> tan1(vertex_count, zero);

        for (size_t i = 0; i < index_count; i += 3)
        {
            uint32_t ai = indices[i + 0];
            uint32_t bi = indices[i + 1];
            uint32_t ci = indices[i + 2];

            Vertex& a = vertices[ai];
            Vertex& b = vertices[bi];
            Vertex& c = vertices[ci];

            ai -= static_cast<uint32_t>(vertex_offset);
            bi -= static_cast<uint32_t>(vertex_offset);
            ci -= static_cast<uint32_t>(vertex_offset);

            float x1 = b.position.x - a.position.x;
            float x2 = c.position.x - a.position.x;
            float y1 = b.position.y - a.position.y;
            float y2 = c.position.y - a.position.y;
            float z1 = b.position.z - a.position.z;
            float z2 = c.position.z - a.position.z;

            float s1 = b.texcoord0.x - a.texcoord0.x;
            float s2 = c.texcoord0.x - a.texcoord0.x;
            float t1 = b.texcoord0.y - a.texcoord0.y;
            float t2 = c.texcoord0.y - a.texcoord0.y;

            if ((s1 * s1 + t1 * t1) < 0.000001f && (s2 * s2 + t2 * t2) < 0.000001f)
                continue;

            float r = 1.0F / (s1 * t2 - s2 * t1);
            DirectX::XMVECTOR sdir = 
                {(t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r, 0.0f };
            DirectX::XMVECTOR tdir = 
                {(s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r, 0.0f };

            tan0[ai] = DirectX::XMVectorAdd(tan0[ai], sdir);
            tan0[bi] = DirectX::XMVectorAdd(tan0[bi], sdir);
            tan0[ci] = DirectX::XMVectorAdd(tan0[ci], sdir);

            tan1[ai] = DirectX::XMVectorAdd(tan1[ai], tdir);
            tan1[bi] = DirectX::XMVectorAdd(tan1[bi], tdir);
            tan1[ci] = DirectX::XMVectorAdd(tan1[ci], tdir);
        }

        for (long a = 0; a < vertex_count; a++)
        {
            Vertex& v = vertices[vertex_offset + a];

            const DirectX::XMVECTOR n = DirectX::XMLoadFloat3(&v.normal);
            const DirectX::XMVECTOR& t1 = tan0[a];
            const DirectX::XMVECTOR& t2 = tan1[a];

            if (DirectX::XMVector3LengthSq(t1).m128_f32[0] < 0.01f)
            {
                DirectX::XMFLOAT3 t;
                guess_tangent(v.normal, t);
                t = normalize(t);
                v.tangent0.x = t.x;
                v.tangent0.y = t.y;
                v.tangent0.z = t.z;
                v.tangent0.w = 1.0f;
                continue;
            }

            DirectX::XMVECTOR tangent = DirectX::XMVectorSubtract(
                t1, DirectX::XMVectorMultiply(n, DirectX::XMVector3Dot(n, t1)));
            tangent = DirectX::XMVector3Normalize(tangent);
            DirectX::XMStoreFloat4(&v.tangent0, tangent);
            v.tangent0.w = (DirectX::XMVector3Dot(
                DirectX::XMVector3Cross(n, t1), t2).m128_f32[0] < 0.0F) ? -1.0F : 1.0F;
        }
    }

    std::string get_texture_uri(
        const fx::gltf::Document& doc, const fx::gltf::Material::Texture& tex)
    {
        int32_t tex_index = tex.index;
        if (tex_index < 0) return "";
        int32_t src_index = doc.textures[tex_index].source;
        if (src_index < 0) return "";
        return doc.images[src_index].uri;
    }

} // anonymous

    bool Loader_gltf::load(const std::string& file_name)
    {
        m_scene = {};
        m_scene.root.kind = Node::Kind::Empty;
        m_scene.root.index = static_cast<size_t>(-1);

        // extend quotas to 1 GB
        fx::gltf::ReadQuotas quotas;
        quotas.MaxFileSize *= 32;
        quotas.MaxBufferByteLength *= 32;


        fx::gltf::Document doc;
        try
        {
            if (str_ends_with(file_name, ".glb"))
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

                Primitive part;
                part.material = p.material;
                part.vertex_offset = mesh.vertices.size();
                part.vertex_count = get_vertex_count(doc, p);
                part.index_offset = mesh.indices.size();

                size_t index_count, stride_index;
                auto p_first_index = get_index_data(doc, p, index_count, stride_index);

                size_t stride_pos = 0;
                const uint8_t* p_first_pos = 
                    get_vertex_data<DirectX::XMFLOAT3>(doc, p, "POSITION", stride_pos);

                size_t stride_normal = 0;
                const uint8_t* p_first_normal = 
                    get_vertex_data<DirectX::XMFLOAT3>(doc, p, "NORMAL", stride_normal);

                size_t stride_texcoord = 0;
                const uint8_t* p_first_texcoord = 
                    get_vertex_data<DirectX::XMFLOAT2>(doc, p, "TEXCOORD_0", stride_texcoord);

                size_t stride_tangent = 0;
                const uint8_t* p_first_tangent =
                    get_vertex_data<DirectX::XMFLOAT2>(doc, p, "TANGENT", stride_tangent);

                bool warned_because_of_tangents = false;
                for (size_t i = 0; i < part.vertex_count; ++i)
                {
                    Vertex v;
                    v.position = read<DirectX::XMFLOAT3>(p_first_pos + i * stride_pos);

                    v.normal = p_first_normal 
                        ? normalize(read<DirectX::XMFLOAT3>(p_first_normal + i * stride_normal)) 
                        : DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);

                    v.texcoord0 = p_first_texcoord 
                        ? read<DirectX::XMFLOAT2>(p_first_texcoord + i * stride_texcoord) 
                        : DirectX::XMFLOAT2(0.0f, 0.0f);
                    v.texcoord0.y = 1.0f - v.texcoord0.y;

                    v.tangent0 = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (p_first_tangent)
                    {
                        v.tangent0 = read<DirectX::XMFLOAT4>(p_first_tangent + i * stride_tangent);
                        float l = v.tangent0.x * v.tangent0.x + 
                                  v.tangent0.y * v.tangent0.y +
                                  v.tangent0.z * v.tangent0.z;

                        if (l < 0.01 || fabsf(v.tangent0.w) < 0.5f) 
                        {
                            if (!warned_because_of_tangents) {
                                log_warning("inconsistent tangents found in mesh: " + m.name, SRC);
                                warned_because_of_tangents = true;
                            }
                        }
                        else
                        {
                            auto tangent = normalize({v.tangent0.x, v.tangent0.y, v.tangent0.z});
                            v.tangent0.x = tangent.x;
                            v.tangent0.y = tangent.y;
                            v.tangent0.z = tangent.z;
                        }
                    }

                    mesh.vertices.push_back(std::move(v));
                }

                if (p_first_index)
                {
                    part.index_count = index_count;
                    if (stride_index == sizeof(uint32_t))
                    {
                        for (size_t i = 0; i < part.index_count; ++i)
                            mesh.indices.push_back(
                                static_cast<uint32_t>(part.vertex_offset +
                                    read<uint32_t>(p_first_index + i * stride_index)));
                    }
                    else if (stride_index == sizeof(uint16_t))
                    {
                        for (size_t i = 0; i < part.index_count; ++i)
                            mesh.indices.push_back(
                                static_cast<uint32_t>(part.vertex_offset +
                                    read<uint16_t>(p_first_index + i * stride_index)));
                    }
                    else if (stride_index == sizeof(uint8_t))
                    {
                        for (size_t i = 0; i < part.index_count; ++i)
                            mesh.indices.push_back(
                                static_cast<uint32_t>(part.vertex_offset +
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
                        mesh.indices.push_back(static_cast<uint32_t>(part.vertex_offset + i));
                }

                if (!p_first_tangent)
                {
                    compute_tangent_frame(
                        mesh.vertices.data(), part.vertex_offset, part.vertex_count,
                        mesh.indices.data() + part.index_offset, part.index_count);
                }

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
        std::function<void(Node&, const fx::gltf::Node&)> traverse = 
            [&](Node& parent, const fx::gltf::Node& src_child)
            {
                Node node;
                node.name = src_child.name;
                node.kind = Node::Kind::Empty;
                node.index = static_cast<size_t>(-1);
                apply_transform(node.local, src_child);

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
                    if(m_scene.cameras[node.index].name.empty())
                        m_scene.cameras[node.index].name = node.name + "_Camera";
                }

                // go down recursively
                for (const auto& c : src_child.children)
                    traverse(node, doc.nodes[c]);

                parent.children.push_back(std::move(node));
            };

        auto s = doc.scenes[doc.scene]; // default scene
        for (const auto& n : s.nodes)
            traverse(m_scene.root, doc.nodes[n]);

        return true;
    }
    void Loader_gltf::replace_all_materials(const std::string & mdl_name)
    {
        m_scene.materials.clear();
        
        m_scene.materials.emplace_back(IScene_loader::Material());
        m_scene.materials.back().name = mdl_name;

        for (auto& m : m_scene.meshes)
            for (auto& p : m.primitives)
                p.material = 0;
    }
}
