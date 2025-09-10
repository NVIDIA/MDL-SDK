/******************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "content/common.hlsl"

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
    payload.flags = FLAG_FIRST_PATH_SEGMENT | FLAG_CAMERA_RAY;

    uint sss_steps_left = scene_constants.max_sss_depth;

    [loop]
    for (uint i = 0; i < scene_constants.max_ray_depth; ++i)
    {
        // last path segment skips next event estimation
        if (i == scene_constants.max_ray_depth - 1)
            add_flag(payload.flags, FLAG_LAST_PATH_SEGMENT);

        // don't count volume scattering steps using the regular ray depth
        if (has_flag(payload.flags, FLAG_SSS) && sss_steps_left > 0)
        {
            sss_steps_left--;
            i--;
        }

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

    // replace the background with a constant color when visible to the camera
    if (scene_constants.background_color_enabled != 0 && has_flag(payload.flags, FLAG_CAMERA_RAY))
    {
        return scene_constants.background_color;
    }

    // clamp fireflies
    float3 contribution = payload.contribution;
    if (scene_constants.firefly_clamp_threshold > 0.0)
    {
        float lum = dot(contribution, float3(0.212671f, 0.715160f, 0.072169f));
        if (lum > scene_constants.firefly_clamp_threshold)
            contribution *= scene_constants.firefly_clamp_threshold / lum;
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
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        // With Shader Model 6.6 we can access the heap directly
        RWTexture2D<float4> OutputBuffer = ResourceDescriptorHeap[0]; // 32bit floating point precision
        RWTexture2D<float4> FrameBuffer = ResourceDescriptorHeap[1]; // 8bit
        #if defined(ENABLE_AUXILIARY)
            RWTexture2D<float4> AlbedoDiffuseBuffer = ResourceDescriptorHeap[2];
            RWTexture2D<float4> AlbedoGlossyBuffer = ResourceDescriptorHeap[3];
            RWTexture2D<float4> NormalBuffer = ResourceDescriptorHeap[4];
            RWTexture2D<float4> RoughnessBuffer = ResourceDescriptorHeap[5];
        #endif
    #else // FEATURE_DYNAMIC_RESOURCES
        // Before SM 6.6 we use global descriptor table instead, the arrays are defined in common.hlsl
        RWTexture2D<float4> OutputBuffer = Global_UAVs_Texture2D_float4[0]; // 32bit floating point precision
        RWTexture2D<float4> FrameBuffer = Global_UAVs_Texture2D_float4[1]; // 8bit
        #if defined(ENABLE_AUXILIARY)
            RWTexture2D<float4> AlbedoDiffuseBuffer = Global_UAVs_Texture2D_float4[2];
            RWTexture2D<float4> AlbedoGlossyBuffer = Global_UAVs_Texture2D_float4[3];
            RWTexture2D<float4> NormalBuffer = Global_UAVs_Texture2D_float4[4];
            RWTexture2D<float4> RoughnessBuffer = Global_UAVs_Texture2D_float4[5];
        #endif
    #endif // FEATURE_DYNAMIC_RESOURCES

    // Get the location within the dispatched 2D grid of work items
    // (often maps to pixels, so this could represent a pixel coordinate).
    uint3 launch_index = DispatchRaysIndex();
    uint3 launch_dim = DispatchRaysDimensions();

    // Generate a ray from a perspective camera
    RayDesc ray;
    ray.TMin = 0.0f;
    ray.TMax = scene_constants.far_plane_distance;

    #if defined(ENABLE_AUXILIARY)
        // in order to limit the payload size, this data is written directly from the hit programs
        // for a progressive refinement of the buffer content we store the current value locally
        // this has other costs: register usage + additional reads/writes to global memory (here)
        float4 tmp_albedo_diffuse = float4(0, 0, 0, 1);
        float4 tmp_albedo_glossy = float4(0, 0, 0, 1);
        float4 tmp_normal = float4(0, 0, 0, 1);
        float4 tmp_roughness = float4(0, 0, 0, 1);
        if (scene_constants.progressive_iteration == 0)
        { 
            // could be replaced by clear calls from CPU
            AlbedoDiffuseBuffer[launch_index.xy] = tmp_albedo_diffuse;
            AlbedoGlossyBuffer[launch_index.xy] = tmp_albedo_glossy;
            NormalBuffer[launch_index.xy] = tmp_normal;
            RoughnessBuffer[launch_index.xy] = tmp_roughness;
        }
        else
        {
            tmp_albedo_diffuse = AlbedoDiffuseBuffer[launch_index.xy];
            tmp_albedo_glossy = AlbedoGlossyBuffer[launch_index.xy];
            tmp_normal = NormalBuffer[launch_index.xy];
            tmp_roughness = RoughnessBuffer[launch_index.xy];
        }
    #endif

    // compute one progressive iteration
    {
        // limit the progression count to 32 if we are in animation mode
        uint it = scene_constants.enable_animiation 
            ? min(scene_constants.progressive_iteration, 32)
            : scene_constants.progressive_iteration;

        // random number seed
        uint seed = tea(
            16, /*magic (see OptiX path tracing example)*/
            launch_dim.x * launch_index.y + launch_index.x,
            it + (scene_constants.enable_animiation ? asint(scene_constants.total_time) : 0));

        // pick (uniform) random position in pixel
        float2 in_pixel_pos = rnd2(seed);
        float2 d = (((launch_index.xy + in_pixel_pos) / float2(launch_dim.xy)) * 2.0f - 1.0f);
        float4 target = mul(camera.projectionI, float4(d.x, -d.y, 1, 1));
        ray.Origin = mul(camera.viewI, float4(0, 0, 0, 1)).xyz;
        ray.Direction = normalize(mul(camera.viewI, float4(target.xyz, 0)).xyz);

        // start tracing one path
        float3 result = trace_path(ray, seed);

        // write results to the output buffers and average equally over all iterations
        float it_weight = 1.0f / float(it + 1);
        OutputBuffer[launch_index.xy] = lerp(OutputBuffer[launch_index.xy], float4(result, 1.0f), it_weight);

        #if defined(ENABLE_AUXILIARY)
            // note, while the 'OutputBuffer' contains converging image, the auxiliary buffers contain
            // only the last values and 'tmp_*' stores the converged data
            tmp_albedo_diffuse = lerp(tmp_albedo_diffuse, AlbedoDiffuseBuffer[launch_index.xy], it_weight);
            tmp_albedo_glossy = lerp(tmp_albedo_glossy, AlbedoGlossyBuffer[launch_index.xy], it_weight);
            tmp_normal = lerp(tmp_normal, NormalBuffer[launch_index.xy], it_weight);
            tmp_roughness = lerp(tmp_roughness, RoughnessBuffer[launch_index.xy], it_weight);
        #endif
    }

    // linear HDR data
    float3 color = OutputBuffer[launch_index.xy].xyz;

    // apply exposure
    color *= pow(2.0f, scene_constants.exposure_compensation);

    // Tone-mapping
    color.x *= (1.0f + color.x * scene_constants.burn_out) / (1.0f + color.x);
    color.y *= (1.0f + color.y * scene_constants.burn_out) / (1.0f + color.y);
    color.z *= (1.0f + color.z * scene_constants.burn_out) / (1.0f + color.z);

    #if defined(ENABLE_AUXILIARY)

    // safe normalize
    bool valid_normal = false;
    if (dot(tmp_normal.xyz, tmp_normal.xyz) > 0.01f)
    {
        valid_normal = true;
        tmp_normal.xyz = normalize(tmp_normal.xyz);
    }

    switch (scene_constants.display_buffer_index)
    {
        case 1: /* albedo */
        {
            color = tmp_albedo_diffuse.xyz + tmp_albedo_glossy.xyz;
            break;
        }
        case 2: /* albedo diffuse */
        {
            color = tmp_albedo_diffuse.xyz;
            break;
        }
        case 3: /* albedo glossy */
        {
            color = tmp_albedo_glossy.xyz;
            break;
        }

        case 4: /* normal */
        {
            color = valid_normal ? (tmp_normal.xyz * 0.5f + 0.5f) : 0.0f;
            break;
        }

        case 5: /* roughness */
        {
            color = tmp_roughness.xyz;
            break;
        }

        case 0: /* beauty */
        case 6: /* aov */
        default:
            break;
    }

    #endif

    // apply gamma corrections for display
    FrameBuffer[launch_index.xy] =
        float4(pow(color, scene_constants.output_gamma_correction), 1.0f);

    // write auxiliary buffer
    #if defined(ENABLE_AUXILIARY)
        AlbedoDiffuseBuffer[launch_index.xy] = tmp_albedo_diffuse;
        AlbedoGlossyBuffer[launch_index.xy] = tmp_albedo_glossy;
        NormalBuffer[launch_index.xy] = tmp_normal;
        RoughnessBuffer[launch_index.xy] = tmp_roughness;
    #endif
}
