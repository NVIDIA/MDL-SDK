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
#include "common.hlsl"

// ------------------------------------------------------------------------------------------------
// defined in the global root signature
// ------------------------------------------------------------------------------------------------

cbuffer CameraParams : register(b0)
{
  float4x4 view;
  float4x4 projection;
  float4x4 viewI;
  float4x4 projectionI;
}

// Ray tracing output texture, accessed as a UAV
RWTexture2D<float4> OutputBuffer : register(u0,space0); // 32bit floating point precision
RWTexture2D<float4> FrameBuffer  : register(u1,space0); // 8bit

// for some post processing effects or for AI denoising, auxiliary outputs are required.
// from the MDL material perspective albedo (approximation) and normals can be generated.
#if defined(ENABLE_AUXILIARY)
    // in order to limit the payload size, this data is written directly from the hit programs
    RWTexture2D<float4> AlbedoBuffer : register(u2,space0);
    RWTexture2D<float4> NormalBuffer : register(u3,space0);
#endif

// Ray tracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

// ------------------------------------------------------------------------------------------------
// loop over path segments
// ------------------------------------------------------------------------------------------------

float3 trace_path(inout RayDesc ray, inout uint seed)
{
    // Initialize the ray payload
    RadianceHitInfo payload;
    payload.contribution = float3(0.0f, 0.0f, 0.0f);
    payload.weight = float3(1.0f, 1.0f, 1.0f);
    payload.seed = seed;
    payload.last_bsdf_pdf = DIRAC;
    payload.flags = FLAG_FIRST_PATH_SEGMENT;

    [loop]
    for (uint i = 0; i < max_ray_depth; ++i)
    {
        TraceRay(
            SceneBVH,               // AccelerationStructure
            RAY_FLAG_NONE,          // RayFlags
            0xFF /* allow all */,   // InstanceInclusionMask
            RAY_TYPE_RADIANCE,      // RayContributionToHitGroupIndex
            RAY_TYPE_COUNT,         // MultiplierForGeometryContributionToHitGroupIndex
            RAY_TYPE_RADIANCE,      // MissShaderIndex
            ray,
            payload);

        // check if we are done
        if (has_flag(payload.flags, FLAG_DONE))
            break;

        // setup ray for the next segment
        ray.Origin = payload.ray_origin_next;
        ray.Direction = payload.ray_direction_next;
        remove_flag(payload.flags, FLAG_FIRST_PATH_SEGMENT);
    }

    // pick up the probably altered seed
    seed = payload.seed;

    // clamp fireflies
    float3 contribution = payload.contribution;
    if (firefly_clamp_threshold > 0.0)
    {
        float lum = dot(contribution, float3(0.212671f, 0.715160f, 0.072169f));
        if (lum > firefly_clamp_threshold)
            contribution *= firefly_clamp_threshold / lum;
    }

    // check for errors and return
    return encode_errors(contribution);
}

// ------------------------------------------------------------------------------------------------
// main entry point
// ------------------------------------------------------------------------------------------------

[shader("raygeneration")]
void RayGenProgram()
{
    // Get the location within the dispatched 2D grid of work items
    // (often maps to pixels, so this could represent a pixel coordinate).
    uint3 launch_index = DispatchRaysIndex();
    uint3 launch_dim = DispatchRaysDimensions();
    float3 camera_position = mul(viewI, float4(0, 0, 0, 1)).xyz;

    // Generate a ray from a perspective camera
    RayDesc ray;
    ray.TMin = 0.0f;
    ray.TMax = 10000.0f;

    #if defined(ENABLE_AUXILIARY)
        // in order to limit the payload size, this data is written directly from the hit programs
        // for a progressive refinement of the buffer content we store the current value locally
        // this has other costs: register usage + additional reads/writes to global memory (here)
        float4 tmp_albedo = float4(0, 0, 0, 1);
        float4 tmp_normal = float4(0, 0, 0, 1);
        if (progressive_iteration == 0)
        {
            AlbedoBuffer[launch_index.xy] = tmp_albedo; // could be replaced by clear call from CPU
            NormalBuffer[launch_index.xy] = tmp_normal; // could be replaced by clear call from CPU
        }
        else
        {
            tmp_albedo = AlbedoBuffer[launch_index.xy];
            tmp_normal = NormalBuffer[launch_index.xy];
        }
    #endif

    // when vsync is active, it is possible to compute multiple iterations per frame
    [loop]
    for (uint it_frame = 0; it_frame < iterations_per_frame; ++it_frame)
    {
        uint it = progressive_iteration + it_frame;

        // random number seed
        unsigned int seed = tea(
            16, /*magic (see OptiX path tracing example)*/
            launch_dim.x * launch_index.y + launch_index.x,
            it);

        // pick (uniform) random position in pixel
        float2 in_pixel_pos = rnd2(seed);
        float2 d = (((launch_index.xy + in_pixel_pos) / float2(launch_dim.xy)) * 2.0f - 1.0f);
        float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));

        ray.Origin = camera_position;
        ray.Direction = normalize(mul(viewI, float4(target.xyz, 0)).xyz);

        // start tracing one path
        float3 result = trace_path(ray, seed);

        // write results to the output buffers and average equally over all iterations
        float it_weight = 1.0f / float(it + 1);
        OutputBuffer[launch_index.xy] = lerp(OutputBuffer[launch_index.xy], float4(result, 1.0f), it_weight);

        #if defined(ENABLE_AUXILIARY)
            // note, while the 'OutputBuffer' contains converging image, the auxiliary buffers contain
            // only the last values and 'tmp_*' stores the converged data
            tmp_albedo = lerp(tmp_albedo, AlbedoBuffer[launch_index.xy], it_weight);
            tmp_normal = lerp(tmp_normal, NormalBuffer[launch_index.xy], it_weight);
        #endif
    }

    // linear HDR data
    float3 color = OutputBuffer[launch_index.xy].xyz;

    // apply exposure
    color *= pow(2.0f, exposure_compensation);

    // Tone-mapping
    color.x *= (1.0f + color.x * burn_out) / (1.0f + color.x);
    color.y *= (1.0f + color.y * burn_out) / (1.0f + color.y);
    color.z *= (1.0f + color.z * burn_out) / (1.0f + color.z);

    #if defined(ENABLE_AUXILIARY)

    // safe normalize
    bool valid_normal = false;
    if (dot(tmp_normal.xyz, tmp_normal.xyz) > 0.01f)
    {
        valid_normal = true;
        tmp_normal.xyz = normalize(tmp_normal.xyz);
    }

    switch (display_buffer_index)
    {
        case 1: /* albedo */
        {
            color = tmp_albedo.xyz;
            break;
        }

        case 2: /* normal */
        {
            color = valid_normal ? (tmp_normal.xyz * 0.5f + 0.5f) : 0.0f;
            break;
        }

        default:
            break;
    }

    #endif

    // apply gamma corrections for display
    FrameBuffer[launch_index.xy] = float4(pow(color, output_gamma_correction), 1.0f);

    // write auxiliary buffer
    #if defined(ENABLE_AUXILIARY)
        AlbedoBuffer[launch_index.xy] = tmp_albedo;
        NormalBuffer[launch_index.xy] = tmp_normal;
    #endif
}
