#include "common.hlsl"

// ------------------------------------------------------------------------------------------------
// defined in the global root signature
// ------------------------------------------------------------------------------------------------

// Environment map and sample data for importance sampling
Texture2D<float4> environment_texture : register(t0,space1);
StructuredBuffer<Environment_sample_data> environment_sample_buffer : register(t1,space1);

// ------------------------------------------------------------------------------------------------
// miss program for RayType::Radiance
// ------------------------------------------------------------------------------------------------

[shader("miss")]
void RadianceMissProgram(inout RadianceHitInfo payload : SV_RayPayload)
{
    float pdf;
    float3 radiance = environment_evaluate( // (see common.hlsl)
        environment_texture,                // assuming lat long map
        environment_sample_buffer,          // importance sampling data of the environment map
        WorldRayDirection(),                // assuming WorldRayDirection() to be normalized
        pdf);

    // to incorporate the point light selection probability
    if (point_light_enabled == 1)
        pdf *= 0.5f;

    if (payload.last_pdf > 0.0f)
    {
        float mis_weight = payload.last_pdf / (payload.last_pdf + pdf);
        radiance *= mis_weight;
    }

    payload.contribution += payload.weight * radiance;
    add_flag(payload.flags, FLAG_DONE);
}

// ------------------------------------------------------------------------------------------------
// miss program for RayType::Shadow
// ------------------------------------------------------------------------------------------------

[shader("miss")]
void ShadowMissProgram(inout ShadowHitInfo payload : SV_RayPayload)
{
    payload.isHit = false;
}
