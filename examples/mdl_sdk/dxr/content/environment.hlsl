/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_DXR_EXAMPLE_ENVIRONMENT_HLSL
#define MDL_DXR_EXAMPLE_ENVIRONMENT_HLSL

#include "content/common.hlsl"

// evaluate the environment for a given direction
float3 environment_evaluate(
    float3 normalized_dir,
    out float pdf)
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        Texture2D<float4> environment_texture = ResourceDescriptorHeap[environment_heap_index];
    #else
        Texture2D<float4> environment_texture = Global_SRVs_Texture2D_float4[environment_heap_index];
    #endif

    // assuming lat long
    float u = atan2(normalized_dir.z, normalized_dir.x) * 0.5f * M_ONE_OVER_PI + 0.5f;
    u -= scene_constants.environment_rotation;
    if (u < 0.0f)
        u += 1.0f;
    const float v = acos(-normalized_dir.y) * M_ONE_OVER_PI;

    // get radiance and calculate pdf
    float3 t = environment_texture.SampleLevel(
        Global_SamplerState_LatLong, float2(u, v), /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0).xyz;
    pdf = max(t.x, max(t.y, t.z)) * scene_constants.environment_inv_integral;
    return t * scene_constants.environment_intensity_factor;
}

// importance sample the enviroment based on intensity
float3 environment_sample(
    inout uint seed,
    out float3 to_light,
    out float pdf)
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        Texture2D<float4> environment_texture = ResourceDescriptorHeap[environment_heap_index];
        StructuredBuffer<Env_Sample> environment_sample_buffer = ResourceDescriptorHeap[environment_heap_index + 1];
    #else
        Texture2D<float4> environment_texture = Global_SRVs_Texture2D_float4[environment_heap_index];
        StructuredBuffer<Env_Sample> environment_sample_buffer = Global_SRVs_Env_Sample[environment_heap_index + 1];
    #endif

    float3 xi;
    xi.x = rnd(seed);
    xi.y = rnd(seed);
    xi.z = rnd(seed);

    uint width;
    uint height;
    environment_texture.GetDimensions(width, height);

    const uint size = width * height;
    const uint idx = min(uint(xi.x * float(size)), size - 1);

    uint env_idx;
    if (xi.y < environment_sample_buffer[idx].q)
    {
        env_idx = idx;
        xi.y /= environment_sample_buffer[idx].q;
    }
    else
    {
        env_idx = environment_sample_buffer[idx].alias;
        xi.y = (xi.y - environment_sample_buffer[idx].q) / (1.0f - environment_sample_buffer[idx].q);
    }

    const uint py = env_idx / width;
    const uint px = env_idx % width;

    // uniformly sample spherical area of pixel
    const float u = float(px + xi.y) / float(width);
    float u_rot = u + scene_constants.environment_rotation;
    if (u_rot > 1.0f)
        u_rot -= 1.0f;

    const float phi = u_rot * (2.0f * M_PI) - M_PI;
    float sin_phi;
    float cos_phi;
    sincos(phi, sin_phi, cos_phi);

    const float step_theta = M_PI / float(height);
    const float theta0 = float(py) * step_theta;
    const float cos_theta = cos(theta0) * (1.0f - xi.z) + cos(theta0 + step_theta) * xi.z;
    const float theta = acos(cos_theta);
    const float sin_theta = sin(theta);
    to_light = float3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

    // lookup filtered value and calculate pdf
    const float v = theta * M_ONE_OVER_PI;
    float3 t = environment_texture.SampleLevel(
        Global_SamplerState_LatLong, float2(u, v), /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0).xyz;
    pdf = max(t.x, max(t.y, t.z)) * scene_constants.environment_inv_integral;
    return t * scene_constants.environment_intensity_factor;
}

#endif