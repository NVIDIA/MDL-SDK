/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#define NORMAL_GENERATION_COMPUTE
#include "content/displacement_common.hlsl"

static const uint FACE_NORMAL_STRIDE = 16;

bool is_valid_length2(float length2)
{
    return length2 > 1e-16f && !isnan(length2) && !isinf(length2);
}

float3 safe_normalize(float3 value, float3 fallback)
{
    const float length2 = dot(value, value);
    return is_valid_length2(length2) ? value * rsqrt(length2) : fallback;
}

float compute_corner_angle(float3 edge_a, float3 edge_b)
{
    const float length2_a = dot(edge_a, edge_a);
    const float length2_b = dot(edge_b, edge_b);
    if (!is_valid_length2(length2_a) || !is_valid_length2(length2_b))
        return 0.0f;

    const float3 edge_a_norm = edge_a * rsqrt(length2_a);
    const float3 edge_b_norm = edge_b * rsqrt(length2_b);
    return acos(clamp(dot(edge_a_norm, edge_b_norm), -1.0f, 1.0f));
}

[numthreads(128, 1, 1)]
void ComputeFaceNormals(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint triangle_id = dispatch_thread_id.x;
    const uint triangle_count = geometry_index_count / 3;
    if (triangle_id >= triangle_count)
        return;

    RWByteAddressBuffer vertex_buffer = get_geometry_vertex_buffer_uav();
    RWByteAddressBuffer face_normal_buffer = get_geometry_face_normal_buffer_uav();
    StructuredBuffer<uint> indices = get_geometry_indices();

    const uint i = triangle_id * 3;
    const uint i0 = indices[geometry_index_offset + i + 0];
    const uint i1 = indices[geometry_index_offset + i + 1];
    const uint i2 = indices[geometry_index_offset + i + 2];

    const float3 p0 = load_geometry_float3_uav(vertex_buffer, i0, VERT_BYTEOFFSET_POSITION);
    const float3 p1 = load_geometry_float3_uav(vertex_buffer, i1, VERT_BYTEOFFSET_POSITION);
    const float3 p2 = load_geometry_float3_uav(vertex_buffer, i2, VERT_BYTEOFFSET_POSITION);
    const float3 face_normal = safe_normalize(cross(p1 - p0, p2 - p0), float3(0.0f, 0.0f, 0.0f));

    const uint angle0 = f32tof16(compute_corner_angle(p1 - p0, p2 - p0));
    const uint angle1 = f32tof16(compute_corner_angle(p0 - p1, p2 - p1));
    const uint packed_angles = (angle0 & 0xffffu) | ((angle1 & 0xffffu) << 16);

    face_normal_buffer.Store4(triangle_id * FACE_NORMAL_STRIDE,
        uint4(asuint(face_normal), packed_angles));
}

[numthreads(128, 1, 1)]
void AccumulateVertexNormals(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint vertex = dispatch_thread_id.x;
    if (vertex >= geometry_vertex_count)
        return;

    RWByteAddressBuffer vertex_buffer = get_geometry_vertex_buffer_uav();
    RWByteAddressBuffer face_normal_buffer = get_geometry_face_normal_buffer_uav();
    StructuredBuffer<uint> adjacency_offsets = get_geometry_adjacency_offsets();
    StructuredBuffer<uint> adjacency_triangles = get_geometry_adjacency_triangles();

    float3 accumulated_normal = float3(0.0f, 0.0f, 0.0f);
    const uint adjacency_begin = adjacency_offsets[geometry_normal_adjacency_offset + vertex];
    const uint adjacency_end = adjacency_offsets[geometry_normal_adjacency_offset + vertex + 1];
    for (uint a = adjacency_begin; a < adjacency_end; ++a)
    {
        const uint packed_adjacency = adjacency_triangles[a];
        const uint triangle_index = packed_adjacency >> 2;
        const uint corner = packed_adjacency & 3u;

        uint4 packed_face = face_normal_buffer.Load4(triangle_index * FACE_NORMAL_STRIDE);
        const float3 face_normal = asfloat(packed_face.xyz);
        const float angle0 = f16tof32(packed_face.w & 0xffffu);
        const float angle1 = f16tof32(packed_face.w >> 16);
        const float angle2 = max(0.0f, M_PI - angle0 - angle1);

        const float angle_weight =
            corner == 0u ? angle0 :
            corner == 1u ? angle1 :
                           angle2;

        accumulated_normal += face_normal * angle_weight;
    }

    float3 normal;
    const float length2 = dot(accumulated_normal, accumulated_normal);
    if (is_valid_length2(length2))
    {
        normal = accumulated_normal * rsqrt(length2);
    }
    else
    {
        const float3 original_normal =
            load_geometry_float3_uav(vertex_buffer, vertex, VERT_BYTEOFFSET_NORMAL);
        normal = normalize(original_normal);
    }
    store_geometry_float3(vertex_buffer, vertex, VERT_BYTEOFFSET_NORMAL, normal);

    float4 tangent = load_geometry_float4_uav(vertex_buffer, vertex, VERT_BYTEOFFSET_TANGENT);
    tangent.xyz = normalize(tangent.xyz - dot(tangent.xyz, normal) * normal);
    store_geometry_float4(vertex_buffer, vertex, VERT_BYTEOFFSET_TANGENT, tangent);
}
