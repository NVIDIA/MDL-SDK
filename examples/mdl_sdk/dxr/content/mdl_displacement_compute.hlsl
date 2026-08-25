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

#include "content/displacement_common.hlsl"

uint get_ro_data_segment_heap_index() { return geometry_material_target_heap_index; }
uint get_argument_block_heap_index() { return geometry_material_instance_heap_index + 1; }
uint get_texture_infos_heap_index() { return geometry_material_instance_heap_index + 2; }
uint get_light_profile_heap_index() { return geometry_material_instance_heap_index + 3; }
uint get_mbsdf_infos_heap_index() { return geometry_material_instance_heap_index + 4; }
uint get_vertex_buffer_heap_index() { return geometry_mesh_resource_heap_index + 2; }
uint get_index_buffer_heap_index() { return geometry_mesh_resource_heap_index + 1; }
uint get_scene_data_info_heap_index() { return geometry_instance_resource_heap_index; }
uint get_scene_data_buffer_heap_index() { return geometry_instance_resource_heap_index + 1; }

Shading_state_material make_vertex_state(uint vertex)
{
    ByteAddressBuffer original_vertices = get_geometry_original_vertex_buffer();

    const float3 local_position = load_geometry_float3(
        original_vertices, vertex, VERT_BYTEOFFSET_POSITION);
    const float3 local_normal = normalize(
        load_geometry_float3(original_vertices, vertex, VERT_BYTEOFFSET_NORMAL));
    const float4 tangent = load_geometry_float4(
        original_vertices, vertex, VERT_BYTEOFFSET_TANGENT);

    float3 local_tangent = normalize(tangent.xyz);
    local_tangent = normalize(local_tangent - dot(local_tangent, local_normal) * local_normal);
    float3 local_tangent_v = cross(local_normal, local_tangent) * tangent.w;

    Shading_state_material state = (Shading_state_material)0;
    // MDL displacement lambdas use object space as their internal space. In particular,
    // state position, normals, tangent frames, and the resulting displacement vector all
    // have to remain in the mesh's local coordinate system.
    state.normal = local_normal;
    state.geom_normal = local_normal;
    #if defined(USE_DERIVS)
        state.position.val = local_position;
    #else
        state.position = local_position;
    #endif
    state.animation_time = scene_constants.enable_animiation ? scene_constants.total_time : 0.0f;
    state.tangent_u[0] = local_tangent;
    state.tangent_v[0] = local_tangent_v;
    state.world_to_object = geometry_world_to_object;
    state.object_to_world = geometry_object_to_world;
    state.meters_per_scene_unit = scene_constants.meters_per_scene_unit;

    state.renderer_state.scene_data_info_offset = geometry_scene_data_info_offset;
    state.renderer_state.scene_data_geometry_byte_offset = geometry_vertex_buffer_byte_offset;
    // Vertex displacement evaluates scene data at one vertex; repeated indices and
    // barycentric (1, 0, 0) make interpolated lookups return that vertex's value.
    state.renderer_state.hit_vertex_indices = uint3(vertex, vertex, vertex);
    state.renderer_state.barycentric = float3(1.0f, 0.0f, 0.0f);

    float2 texcoord0 = scene_data_lookup_float2(
        state, SCENE_DATA_ID_TEXCOORD_0, float2(0.0f, 0.0f), false);
    texcoord0 = texcoord0 * scene_constants.uv_scale + scene_constants.uv_offset;
    if (scene_constants.uv_repeat != 0)
        texcoord0 = texcoord0 - floor(texcoord0);
    if (scene_constants.uv_clamp != 0)
        texcoord0 = saturate(texcoord0);

    #if defined(USE_DERIVS)
        state.text_coords[0].val = float3(texcoord0, 0.0f);
    #else
        state.text_coords[0] = float3(texcoord0, 0.0f);
    #endif

    return state;
}

[numthreads(128, 1, 1)]
void MdlDisplaceVertices(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint vertex = dispatch_thread_id.x;
    if (vertex >= geometry_vertex_count)
        return;

    Shading_state_material state = make_vertex_state(vertex);
    ByteAddressBuffer original_vertices = get_geometry_original_vertex_buffer();
    RWByteAddressBuffer displaced_vertices = get_geometry_vertex_buffer_uav();

    const float3 position = load_geometry_float3(
        original_vertices, vertex, VERT_BYTEOFFSET_POSITION);
    const float3 displacement_local = mdl_standalone_geometry_displacement(state);
    const float3 displaced_position = position + displacement_local;

    store_geometry_float3(
        displaced_vertices, vertex, VERT_BYTEOFFSET_POSITION, displaced_position);
}
