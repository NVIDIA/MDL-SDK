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

#if !defined(MDL_DISPLACEMENT_COMMON_HLSLI)
#define MDL_DISPLACEMENT_COMMON_HLSLI

#include "content/common.hlsl"

#if defined(NORMAL_GENERATION_COMPUTE)
cbuffer Normal_generation_constants : register(b10, space0)
{
    uint geometry_mesh_resource_heap_index;
    uint geometry_vertex_buffer_byte_offset;
    uint geometry_vertex_stride;
    uint geometry_index_offset;

    uint geometry_vertex_count;
    uint geometry_index_count;
    uint geometry_normal_adjacency_offset;
    uint geometry_face_normal_buffer_uav_heap_index;
}
#else
cbuffer Displacement_constants : register(b10, space0)
{
    uint geometry_mesh_resource_heap_index;
    uint geometry_instance_resource_heap_index;
    uint geometry_material_target_heap_index;
    uint geometry_material_instance_heap_index;

    uint geometry_vertex_buffer_byte_offset;
    uint geometry_vertex_stride;
    uint geometry_vertex_count;
    uint geometry_scene_data_info_offset;

    row_major float4x4 geometry_object_to_world;
    row_major float4x4 geometry_world_to_object;
}
#endif

ByteAddressBuffer get_geometry_vertex_buffer()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index)];
    #else
        return Global_SRVs_ByteAddressBuffer[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index)];
    #endif
}

ByteAddressBuffer get_geometry_original_vertex_buffer()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 2)];
    #else
        return Global_SRVs_ByteAddressBuffer[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 2)];
    #endif
}

RWByteAddressBuffer get_geometry_vertex_buffer_uav()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 3)];
    #else
        return Global_UAVs_ByteAddressBuffer[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 3)];
    #endif
}

StructuredBuffer<uint> get_geometry_indices()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 1)];
    #else
        return Global_SRVs_StructuredBuffer_uint[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 1)];
    #endif
}

#if defined(NORMAL_GENERATION_COMPUTE)
RWByteAddressBuffer get_geometry_face_normal_buffer_uav()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_face_normal_buffer_uav_heap_index)];
    #else
        return Global_UAVs_ByteAddressBuffer[
            NonUniformResourceIndex(geometry_face_normal_buffer_uav_heap_index)];
    #endif
}

StructuredBuffer<uint> get_geometry_adjacency_offsets()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 4)];
    #else
        return Global_SRVs_StructuredBuffer_uint[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 4)];
    #endif
}

StructuredBuffer<uint> get_geometry_adjacency_triangles()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 5)];
    #else
        return Global_SRVs_StructuredBuffer_uint[
            NonUniformResourceIndex(geometry_mesh_resource_heap_index + 5)];
    #endif
}
#endif

uint get_geometry_vertex_address(uint vertex, uint byte_offset)
{
    return geometry_vertex_buffer_byte_offset +
        vertex * geometry_vertex_stride + byte_offset;
}

float3 load_geometry_float3(ByteAddressBuffer buffer, uint vertex, uint byte_offset)
{
    return asfloat(buffer.Load3(get_geometry_vertex_address(vertex, byte_offset)));
}

float4 load_geometry_float4(ByteAddressBuffer buffer, uint vertex, uint byte_offset)
{
    return asfloat(buffer.Load4(get_geometry_vertex_address(vertex, byte_offset)));
}

float3 load_geometry_float3_uav(RWByteAddressBuffer buffer, uint vertex, uint byte_offset)
{
    return asfloat(buffer.Load3(get_geometry_vertex_address(vertex, byte_offset)));
}

float4 load_geometry_float4_uav(RWByteAddressBuffer buffer, uint vertex, uint byte_offset)
{
    return asfloat(buffer.Load4(get_geometry_vertex_address(vertex, byte_offset)));
}

void store_geometry_float3(RWByteAddressBuffer buffer, uint vertex, uint byte_offset, float3 value)
{
    buffer.Store3(get_geometry_vertex_address(vertex, byte_offset), asuint(value));
}

void store_geometry_float4(RWByteAddressBuffer buffer, uint vertex, uint byte_offset, float4 value)
{
    buffer.Store4(get_geometry_vertex_address(vertex, byte_offset), asuint(value));
}

#endif
