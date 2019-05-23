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
RWTexture2D<float4> OutputBuffer : register(u0); // 32bit floating point precision
RWTexture2D<float4> FrameBuffer : register(u1);  // 8bit 

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
    payload.last_pdf = -1.0f;
    payload.flags = FLAG_NONE;

    [fastopt]
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
    }

    // pick up the probably altered seed
    seed = payload.seed; 

    // apply firefly clamp
    float3 contribution = payload.contribution;
    contribution = isinf(contribution) || isnan(contribution) ? 0.0f : contribution;
    contribution = max(0.0f, contribution);

    // clamp fireflies
    if (firefly_clamp_threshold > 0.0)
    {
        float lum = dot(contribution, float3(0.212671f, 0.715160f, 0.072169f));
        if (lum > firefly_clamp_threshold)
            contribution *= firefly_clamp_threshold / lum;
    }

    return contribution;
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

    // when vsync is active, it is possible to compute multiple iterations per frame
    [fastopt]
    for (uint it = 0; it < iterations_per_frame; ++it)
    {
        uint total_iteration_number = progressive_iteration + it;

        // random number seed
        unsigned int seed = tea(
            16, /*magic (see OptiX path tracing example)*/
            launch_dim.x * launch_index.y + launch_index.x,
            total_iteration_number);


        // pick (uniform) random position in pixel
        float2 in_pixel_pos = rnd2(seed);
        float2 d = (((launch_index.xy + in_pixel_pos) / float2(launch_dim.xy)) * 2.0f - 1.0f);
        float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));

        ray.Origin = camera_position;
        ray.Direction = normalize(mul(viewI, float4(target.xyz, 0)).xyz);

        // start tracing one path
        float3 result = trace_path(ray, seed);

        // write results to the output buffer
        if (total_iteration_number == 0)
            OutputBuffer[launch_index.xy] = float4(result, 1.0f);
        else
            OutputBuffer[launch_index.xy] = lerp(
                OutputBuffer[launch_index.xy], 
                float4(result, 1.0f), 
                1.0f / float(total_iteration_number + 1)); // average over all iterations
    }

    // linear HDR data
    float3 color = OutputBuffer[launch_index.xy].xyz;

    // apply exposure
    color *= pow(2.0f, exposure_compensation); 

    // Tone-mapping
    color.x *= (1.0f + color.x * burn_out) / (1.0f + color.x);
    color.y *= (1.0f + color.y * burn_out) / (1.0f + color.y);
    color.z *= (1.0f + color.z * burn_out) / (1.0f + color.z);

    // apply gamma corrections for display
    FrameBuffer[launch_index.xy] = float4(pow(color, output_gamma_correction), 1.0f);
}
