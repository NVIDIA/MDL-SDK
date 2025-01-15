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

#if !defined(RENDERER_TYPES_HLSLI)
#define RENDERER_TYPES_HLSLI
// This file is included at the end of "common.hlsl" to make it globally avaiable after
// all user defined types are declared.


//-------------------------------------------------------------------------------------------------
// Global Resource Tables
//-------------------------------------------------------------------------------------------------

// With Shader Model 6.6 we can access the heap directly using `ResourceDescriptorHeap[<index>]`
// Before SM 6.6 we use global descriptor table instead:
#if (FEATURE_DYNAMIC_RESOURCES == 0)

    // UAVs can be overlapping but we need only one at this point.
    RWTexture2D<float4> Global_UAVs_Texture2D_float4[]                  : register(u0, space100);

    // All SRVs are overlapping to handle the different types.
    // In order for this to work, different spaces are used.
    Texture2D<float4> Global_SRVs_Texture2D_float4[]                    : register(t0, space100);
    Texture3D<float4> Global_SRVs_Texture3D_float4[]                    : register(t0, space101);
    RaytracingAccelerationStructure Global_SRVs_BVH[]                   : register(t0, space102);
    ByteAddressBuffer Global_SRVs_ByteAddressBuffer[]                   : register(t0, space103);
    StructuredBuffer<float> Global_SRVs_StructuredBuffer_float[]        : register(t0, space104);
    StructuredBuffer<uint> Global_SRVs_StructuredBuffer_uint[]          : register(t0, space105);
    StructuredBuffer<Env_Sample> Global_SRVs_Env_Sample[]               : register(t0, space106);
    StructuredBuffer<SceneDataInfo> Global_SRVs_SceneDataInfo[]         : register(t0, space107);
    StructuredBuffer<Mdl_texture_info> Global_SRVs_MDL_tex_info[]       : register(t0, space108);
    StructuredBuffer<Mdl_light_profile_info> Global_SRVs_MDL_lp_info[]  : register(t0, space109);
    StructuredBuffer<Mdl_mbsdf_info> Global_SRVs_MDL_mbsdf_info[]       : register(t0, space110);

    // CBVs can be overlapping but we need only one at this point.
    ConstantBuffer<Material_constants> Global_CBVs_Material_constants[] : register(b0, space100);

#endif

//-------------------------------------------------------------------------------------------------
// Global Resources and Samplers
//-------------------------------------------------------------------------------------------------

// For globally required resources we pass the heap indices in the global root signature directly
cbuffer Global_Constants_b0_space0 : register(b0, space0) { CameraParams camera; }
cbuffer Global_Constants_b1_space0 : register(b1, space0) { SceneConstants scene_constants; }
cbuffer Global_Constants_b2_space0 : register(b2, space0) { uint environment_heap_index; }

// Ray tracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0, space0);

// Sampler States
SamplerState Global_SamplerState_MDL_tex    : register(s0);
SamplerState Global_SamplerState_MDL_lp     : register(s1);
SamplerState Global_SamplerState_MDL_mbsdf  : register(s2);
SamplerState Global_SamplerState_LatLong    : register(s3);

#endif // RENDERER_TYPES_HLSLI
