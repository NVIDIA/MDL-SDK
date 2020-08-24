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

// Environment map and sample data for importance sampling
Texture2D<float4> environment_texture : register(t0,space1);
StructuredBuffer<Environment_sample_data> environment_sample_buffer : register(t1,space1);

// ------------------------------------------------------------------------------------------------
// miss program for RayType::Radiance
// ------------------------------------------------------------------------------------------------

[shader("miss")]
void RadianceMissProgram(inout RadianceHitInfo payload : SV_RayPayload)
{
    float light_pdf;
    float3 radiance = environment_evaluate( // (see common.hlsl)
        environment_texture,                // assuming lat long map
        environment_sample_buffer,          // importance sampling data of the environment map
        WorldRayDirection(),                // assuming WorldRayDirection() to be normalized
        light_pdf);

    // to incorporate the point light selection probability
    if (point_light_enabled == 1)
        light_pdf *= 0.5f;

    // MIS weight for non-specular BSDF events
    const float mis_weight = (payload.last_bsdf_pdf == DIRAC)
        ? 1.0f
        : payload.last_bsdf_pdf / (payload.last_bsdf_pdf + light_pdf);

    payload.contribution += payload.weight * radiance * mis_weight;
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
