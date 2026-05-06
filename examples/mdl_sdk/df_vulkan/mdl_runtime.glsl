/******************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/df_vulkan/mdl_runtime.glsl

// Expected defines:
//   MDL_SET_MATERIAL_TEXTURES_2D         : The set index for the array of material 2D textures
//   MDL_SET_MATERIAL_TEXTURES_3D         : The set index for the array of material 3D textures
//   MDL_SET_MATERIAL_ARGUMENT_BLOCK      : The set index for the material argument block buffer
//   MDL_SET_MATERIAL_RO_DATA_SEGMENT     : The set index for the material read-only data segment buffer
//   MDL_BINDING_MATERIAL_TEXTURES_2D     : The binding index for the array of material 2D textures
//   MDL_BINDING_MATERIAL_TEXTURES_3D     : The binding index for the array of material 3D textures
//   MDL_BINDING_MATERIAL_ARGUMENT_BLOCK  : The binding index for the material argument block buffer
//   MDL_BINDING_MATERIAL_RO_DATA_SEGMENT : The binding index for the material read-only data segment buffer
//   NUM_TEX_RESULTS                      : The size of the texture results cache (only defined if size > 0)
//   USE_RO_DATA_SEGMENT                  : Defined if the read-only data segment is enabled

#ifndef MDL_RUNTIME_GLSL
#define MDL_RUNTIME_GLSL

#extension GL_EXT_nonuniform_qualifier : require

#include "mdl_target_code_types.glsl"

// The arrays of material textures used in the texturing functions
layout(set = MDL_SET_MATERIAL_TEXTURES_2D, binding = MDL_BINDING_MATERIAL_TEXTURES_2D)
uniform sampler2D uMaterialTextures2D[];

layout(set = MDL_SET_MATERIAL_TEXTURES_3D, binding = MDL_BINDING_MATERIAL_TEXTURES_3D)
uniform sampler3D uMaterialTextures3D[];

// The material argument block used for dynamic parameters in class compilation mode
layout(std430, set = MDL_SET_MATERIAL_ARGUMENT_BLOCK, binding = MDL_BINDING_MATERIAL_ARGUMENT_BLOCK)
readonly restrict buffer ArgumentBlockBuffer
{
    uint uMaterialArgumentBlock[];
};

#ifdef USE_RO_DATA_SEGMENT
// The read-only data segment
layout(std430, set = MDL_SET_MATERIAL_RO_DATA_SEGMENT, binding = MDL_BINDING_MATERIAL_RO_DATA_SEGMENT)
readonly restrict buffer RODataSegmentBuffer
{
    uint uMaterialRODataSegment[];
};
#endif // USE_RO_DATA_SEGMENT


// ------------------------------------------------------------------------------------------------
// Argument block access for dynamic parameters in class compilation mode
// ------------------------------------------------------------------------------------------------

float mdl_read_argblock_as_float(int offs)
{
    return uintBitsToFloat(uMaterialArgumentBlock[offs >> 2]);
}

double mdl_read_argblock_as_double(int offs)
{
    return packDouble2x32(
        uvec2(uMaterialArgumentBlock[offs >> 2], uMaterialArgumentBlock[(offs >> 2) + 1]));
}

int mdl_read_argblock_as_int(int offs)
{
    return int(uMaterialArgumentBlock[offs >> 2]);
}

uint mdl_read_argblock_as_uint(int offs)
{
    return uMaterialArgumentBlock[offs >> 2];
}

bool mdl_read_argblock_as_bool(int offs)
{
    uint val = uMaterialArgumentBlock[offs >> 2];
    return (val & (0xff << (8 * (offs & 3)))) != 0;
}


// ------------------------------------------------------------------------------------------------
// Read-only data access via read functions
// ------------------------------------------------------------------------------------------------

#ifdef USE_RO_DATA_SEGMENT

float mdl_read_rodata_as_float(int offs)
{
    return uintBitsToFloat(uMaterialRODataSegment[offs >> 2]);
}

double mdl_read_rodata_as_double(int offs)
{
    return packDouble2x32(
        uvec2(uMaterialRODataSegment[offs >> 2], uMaterialRODataSegment[(offs >> 2) + 1]));
}

int mdl_read_rodata_as_int(int offs)
{
    return int(uMaterialRODataSegment[offs >> 2]);
}

uint mdl_read_rodata_as_uint(int offs)
{
    return uMaterialRODataSegment[offs >> 2];
}

bool mdl_read_rodata_as_bool(int offs)
{
    uint val = uMaterialRODataSegment[offs >> 2];
    return (val & (0xff << (8 * (offs & 3)))) != 0;
}

#endif // USE_RO_DATA_SEGMENT


//-----------------------------------------------------------------------------
// Texture helper functions
//-----------------------------------------------------------------------------

// corresponds to ::tex::texture_isvalid(uniform texture_2d tex)
// corresponds to ::tex::texture_isvalid(uniform texture_3d tex)
// corresponds to ::tex::texture_isvalid(uniform texture_cube tex) // not supported by this example
// corresponds to ::tex::texture_isvalid(uniform texture_ptex tex) // not supported by this example
bool tex_texture_isvalid(int tex)
{
    // assuming that there is no indexing out of bounds of the texture arrays
    return tex != 0; // invalid texture
}

// helper function to realize wrap and crop.
// Out of bounds case for TEX_WRAP_CLIP must already be handled.
float apply_wrap_and_crop(float coord, int wrap, vec2 crop, int res)
{
    if (wrap != TEX_WRAP_REPEAT || crop.x != 0.0 || crop.y != 1.0)
    {
        if (wrap == TEX_WRAP_REPEAT)
        {
            coord -= floor(coord);
        }
        else
        {
            if (wrap == TEX_WRAP_MIRRORED_REPEAT)
            {
                float floored_val = floor(coord);
                if ((int(floored_val) & 1) != 0)
                    coord = 1 - (coord - floored_val);
                else
                    coord -= floored_val;
            }
            float inv_hdim = 0.5 / float(res);
            coord = clamp(coord, inv_hdim, 1.0 - inv_hdim);
        }
        coord = coord * (crop.y - crop.x) + crop.x;
    }
    return coord;
}

// Modify texture coordinates to get better texture filtering,
// see http://www.iquilezles.org/www/articles/texture/texture.htm
vec2 apply_smootherstep_filter(vec2 uv, ivec2 size)
{
    vec2 res = uv * vec2(size) + 0.5;
    vec2 i = floor(res);
    vec2 f = res - i;
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return ((i + f) - 0.5) / vec2(size);
}


//-----------------------------------------------------------------------------
// Texture function implementations, 2D
//-----------------------------------------------------------------------------

// corresponds to ::tex::width(uniform texture_2d tex, int2 uv_tile, float frame)
int tex_width_2d(int tex, ivec2 uv_tile, float frame)
{
    if (tex == 0) return 0; // invalid texture
    return textureSize(uMaterialTextures2D[nonuniformEXT(tex - 1)], 0).x;
}

// corresponds to ::tex::height(uniform texture_2d tex, int2 uv_tile, float frame)
int tex_height_2d(int tex, ivec2 uv_tile, float frame)
{
    if (tex == 0) return 0; // invalid texture
    return textureSize(uMaterialTextures2D[nonuniformEXT(tex - 1)], 0).y;
}

// corresponds to ::tex::lookup_float4(uniform texture_2d tex, float2 coord, ...)
vec4 tex_lookup_float4_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    if (tex == 0) return vec4(0.0); // invalid texture

    if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x >= 1.0))
        return vec4(0.0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y >= 1.0))
        return vec4(0.0);

    ivec2 tex_size = textureSize(uMaterialTextures2D[nonuniformEXT(tex - 1)], 0);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, tex_size.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, tex_size.y);
    coord = apply_smootherstep_filter(coord, tex_size);

    return texture(uMaterialTextures2D[nonuniformEXT(tex - 1)], coord);
}

vec3 tex_lookup_float3_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

vec3 tex_lookup_color_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

vec2 tex_lookup_float2_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_float_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    return tex_lookup_float4_2d(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}

// corresponds to ::tex::texel_float4(uniform texture_2d tex, int2 coord, int2 uv_tile, float frame)
vec4 tex_texel_float4_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    if (tex == 0) return vec4(0.0); // invalid texture

    ivec2 res = textureSize(uMaterialTextures2D[nonuniformEXT(tex - 1)], 0);
    if (coord.x < 0 || coord.y < 0 || coord.x >= res.x || coord.y >= res.y)
        return vec4(0.0); // out of bounds

    return texelFetch(uMaterialTextures2D[nonuniformEXT(tex - 1)], coord, 0);
}

vec3 tex_texel_float3_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xyz;
}

vec3 tex_texel_color_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xyz;
}

vec2 tex_texel_float2_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).xy;
}

float tex_texel_float_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)
{
    return tex_texel_float4_2d(tex, coord, uv_tile, frame).x;
}


//-----------------------------------------------------------------------------
// Texture function implementations, 3D
//-----------------------------------------------------------------------------

// corresponds to ::tex::width(uniform texture_3d tex, float frame)
int tex_width_3d(int tex, float frame)
{
    if (tex == 0) return 0; // invalid texture
    return textureSize(uMaterialTextures3D[nonuniformEXT(tex - 1)], 0).x;
}

// corresponds to ::tex::height(uniform texture_3d tex, float frame)
int tex_height_3d(int tex, float frame)
{
    if (tex == 0) return 0; // invalid texture
    return textureSize(uMaterialTextures3D[nonuniformEXT(tex - 1)], 0).y;
}

// corresponds to ::tex::depth(uniform texture_3d tex, float frame)
int tex_depth_3d(int tex, float frame)
{
    if (tex == 0) return 0; // invalid texture
    return textureSize(uMaterialTextures3D[nonuniformEXT(tex - 1)], 0).z;
}

// corresponds to ::tex::lookup_float4(uniform texture_3d tex, float3 coord, ...)
vec4 tex_lookup_float4_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    if (tex == 0) return vec4(0.0); // invalid texture

    if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x >= 1.0))
        return vec4(0.0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y >= 1.0))
        return vec4(0.0);
    if (wrap_w == TEX_WRAP_CLIP && (coord.z < 0.0 || coord.z >= 1.0))
        return vec4(0.0);

    ivec3 tex_size = textureSize(uMaterialTextures3D[nonuniformEXT(tex - 1)], 0);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, tex_size.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, tex_size.y);
    coord.z = apply_wrap_and_crop(coord.z, wrap_w, crop_w, tex_size.z);

    return texture(uMaterialTextures3D[nonuniformEXT(tex - 1)], coord);
}

vec3 tex_lookup_float3_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

vec3 tex_lookup_color_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

vec2 tex_lookup_float2_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xy;
}

float tex_lookup_float_3d(int tex, vec3 coord, int wrap_u, int wrap_v, int wrap_w, vec2 crop_u, vec2 crop_v, vec2 crop_w, float frame)
{
    return tex_lookup_float4_3d(tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).x;
}

// corresponds to ::tex::texel_float4(uniform texture_3d tex, int3 coord, float frame)
vec4 tex_texel_float4_3d(int tex, ivec3 coord, float frame)
{
    if (tex == 0) return vec4(0.0); // invalid texture

    ivec3 res = textureSize(uMaterialTextures3D[nonuniformEXT(tex - 1)], 0);
    if (coord.x < 0 || coord.y < 0 || coord.z < 0 || coord.x >= res.x || coord.y >= res.y || coord.z >= res.z)
        return vec4(0.0); // out of bounds

    return texelFetch(uMaterialTextures3D[nonuniformEXT(tex - 1)], coord, 0);
}

vec3 tex_texel_float3_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xyz;
}

vec3 tex_texel_color_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xyz;
}

vec2 tex_texel_float2_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).xy;
}

float tex_texel_float_3d(int tex, ivec3 coord, float frame)
{
    return tex_texel_float4_3d(tex, coord, frame).x;
}


// ------------------------------------------------------------------------------------------------
// Texture function implementations, Cube (not supported by this example)
// ------------------------------------------------------------------------------------------------

// corresponds to ::tex::width(uniform texture_cube tex)
int tex_width_cube(int tex)
{
    return 0;
}

// corresponds to ::tex::height(uniform texture_cube tex)
int tex_height_cube(int tex)
{
    return 0;
}

// corresponds to ::tex::lookup_float4(uniform texture_cube tex, float3 coord)
vec4 tex_lookup_float4_cube(int tex, vec3 coord)
{
    return vec4(0.0);
}

vec3 tex_lookup_float3_cube(int tex, vec3 coord)
{
    return tex_lookup_float4_cube(tex, coord).xyz;
}

vec3 tex_lookup_color_cube(int tex, vec3 coord)
{
    return tex_lookup_float4_cube(tex, coord).xyz;
}

vec2 tex_lookup_float2_cube(int tex, vec3 coord)
{
    return tex_lookup_float4_cube(tex, coord).xy;
}

float tex_lookup_float_cube(int tex, vec3 coord)
{
    return tex_lookup_float4_cube(tex, coord).x;
}

// corresponds to ::tex::texel_float4(uniform texture_cube tex, int3 coord)
vec4 tex_texel_float4_cube(int tex, ivec3 coord)
{
    return vec4(0.0);
}

vec3 tex_texel_float3_cube(int tex, ivec3 coord)
{
    return tex_texel_float4_cube(tex, coord).xyz;
}

vec3 tex_texel_color_cube(int tex, ivec3 coord)
{
    return tex_texel_float4_cube(tex, coord).xyz;
}

vec2 tex_texel_float2_cube(int tex, ivec3 coord)
{
    return tex_texel_float4_cube(tex, coord).xy;
}

float tex_texel_float_cube(int tex, ivec3 coord)
{
    return tex_texel_float4_cube(tex, coord).x;
}


//-----------------------------------------------------------------------------
// Texture function implementations, PTEX (not supported by this example)
//-----------------------------------------------------------------------------

vec4 tex_lookup_float4_ptex(int tex, int channel)
{
    return vec4(0.0);
}

vec3 tex_lookup_float3_ptex(int tex, int channel)
{
    return tex_lookup_float4_ptex(tex, channel).xyz;
}

vec3 tex_lookup_color_ptex(int tex, int channel)
{
    return tex_lookup_float3_ptex(tex, channel);
}

vec2 tex_lookup_float2_ptex(int tex, int channel)
{
    return tex_lookup_float4_ptex(tex, channel).xy;
}

float tex_lookup_float_ptex(int tex, int channel)
{
    return tex_lookup_float4_ptex(tex, channel).x;
}


// ------------------------------------------------------------------------------------------------
// Light Profiles function implementations (not supported by this example)
// ------------------------------------------------------------------------------------------------

bool df_light_profile_isvalid(int lp_idx)
{
    return false;
}

float df_light_profile_power(int lp_idx)
{
    return 0.0;
}

float df_light_profile_maximum(int lp_idx)
{
    return 0.0;
}

float df_light_profile_evaluate(int lp_idx, vec2 theta_phi)
{
    return 0.0;
}

vec3 df_light_profile_sample(int lp_idx, vec3 xi)
{
    return vec3(0.0);
}

float df_light_profile_pdf(int lp_idx, vec2 theta_phi)
{
    return 0.0;
}


// ------------------------------------------------------------------------------------------------
// Measured BSDFs function implementations (not supported by this example)
// ------------------------------------------------------------------------------------------------

bool df_bsdf_measurement_isvalid(int bm_idx)
{
    return false;
}

ivec3 df_bsdf_measurement_resolution(int bm_idx, int part)
{
    return ivec3(0);
}

vec3 df_bsdf_measurement_evaluate(int bm_idx, vec2 theta_phi_in, vec2 theta_phi_out, int part)
{
    return vec3(0.0);
}

vec3 df_bsdf_measurement_sample(int bm_idx, vec2 theta_phi_out, vec3 xi, int part)
{
    return vec3(0.0);
}

float df_bsdf_measurement_pdf(int bm_idx, vec2 theta_phi_in, vec2 theta_phi_out, int part)
{
    return 0.0;
}

vec4 df_bsdf_measurement_albedos(int bm_idx, vec2 theta_phi)
{
    return vec4(0.0);
}


// ------------------------------------------------------------------------------------------------
// Scene Data API function implementations (not supported by this example)
// ------------------------------------------------------------------------------------------------

bool scene_data_isvalid(State state, int scene_data_id)
{
    return false;
}

vec4 scene_data_lookup_float4(State state, int scene_data_id, vec4 default_value, bool uniform_lookup)
{
    return default_value;
}

vec3 scene_data_lookup_float3(State state, int scene_data_id, vec3 default_value, bool uniform_lookup)
{
    return default_value;
}

vec3 scene_data_lookup_color(State state, int scene_data_id, vec3 default_value, bool uniform_lookup)
{
    return default_value;
}

vec2 scene_data_lookup_float2(State state, int scene_data_id, vec2 default_value, bool uniform_lookup)
{
    return default_value;
}

float scene_data_lookup_float(State state, int scene_data_id, float default_value, bool uniform_lookup)
{
    return default_value;
}

ivec4 scene_data_lookup_int4(State state, int scene_data_id, ivec4 default_value, bool uniform_lookup)
{
    return default_value;
}

ivec3 scene_data_lookup_int3(State state, int scene_data_id, ivec3 default_value, bool uniform_lookup)
{
    return default_value;
}

ivec2 scene_data_lookup_int2(State state, int scene_data_id, ivec2 default_value, bool uniform_lookup)
{
    return default_value;
}

int scene_data_lookup_int(State state, int scene_data_id, int default_value, bool uniform_lookup)
{
    return default_value;
}

mat4 scene_data_lookup_float4x4(State state, int scene_data_id, mat4 default_value, bool uniform_lookup)
{
    return default_value;
}


// ------------------------------------------------------------------------------------------------
// Spectral rendering runtime functions
// Called by MDL-generated GLSL code when libbsdf_enable_spectral is active.
// ------------------------------------------------------------------------------------------------

#ifdef MDL_SPECTRAL_RENDERING

// D65 standard illuminant spectral power distribution, 360-830 nm in 5 nm steps (95 entries).
// Scaled so that the luminance integral (Y channel) equals 1.
const float s_cie_d65[95] = float[](
    6.462114e-06, 6.839740e-06, 7.217367e-06, 7.070938e-06,
    6.924510e-06, 7.248223e-06, 7.571951e-06, 9.519149e-06,
    1.146636e-05, 1.207124e-05, 1.267613e-05, 1.281093e-05,
    1.294573e-05, 1.247813e-05, 1.201053e-05, 1.327021e-05,
    1.452989e-05, 1.537108e-05, 1.621241e-05, 1.626811e-05,
    1.632381e-05, 1.611929e-05, 1.591492e-05, 1.598850e-05,
    1.606207e-05, 1.556936e-05, 1.507664e-05, 1.511419e-05,
    1.515188e-05, 1.504436e-05, 1.493684e-05, 1.472817e-05,
    1.451950e-05, 1.472027e-05, 1.492118e-05, 1.469367e-05,
    1.446616e-05, 1.444122e-05, 1.441642e-05, 1.413611e-05,
    1.385581e-05, 1.360185e-05, 1.334788e-05, 1.331004e-05,
    1.327220e-05, 1.278016e-05, 1.228811e-05, 1.237960e-05,
    1.247109e-05, 1.244288e-05, 1.241468e-05, 1.228302e-05,
    1.215136e-05, 1.184583e-05, 1.154031e-05, 1.156876e-05,
    1.159720e-05, 1.134278e-05, 1.108836e-05, 1.110137e-05,
    1.111438e-05, 1.125732e-05, 1.140026e-05, 1.112358e-05,
    1.084691e-05, 1.025367e-05, 9.660451e-06, 9.791236e-06,
    9.922021e-06, 1.011183e-05, 1.030166e-05, 9.418694e-06,
    8.535733e-06, 9.109474e-06, 9.683216e-06, 1.004356e-05,
    1.040391e-05, 9.607591e-06, 8.811283e-06, 7.621443e-06,
    6.431617e-06, 7.844023e-06, 9.256429e-06, 9.019315e-06,
    8.782200e-06, 8.846020e-06, 8.909840e-06, 8.573684e-06,
    8.237542e-06, 7.718434e-06, 7.199340e-06, 7.579100e-06,
    7.958860e-06, 8.157816e-06, 8.356785e-06
);

float lookup_d65(float lambda)
{
    float f = (lambda - 360.0) / (830.0 - 360.0);
    if (f < 0.0 || f > 1.0)
        return 0.0;
    f *= float(95 - 1);
    int b0 = min(int(f), 95 - 1);
    int b1 = (b0 < (95 - 1)) ? (b0 + 1) : b0;
    float w1 = f - float(b0);
    return s_cie_d65[b0] * (1.0 - w1) + s_cie_d65[b1] * w1;
}

// RGB-to-spectral conversion.
// Uses Jendersie - "Fast Spectral Upsampling of Volume Attenuation Coefficients".
Spectral_sample rgb_to_spectral(vec3 rgb, Spectral_sample lambdas, bool is_emission)
{
    Spectral_sample s;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        float lambda = lambdas.values[i];
        s.values[i] = (lambda < 485.0) ? rgb.b : ((lambda < 595.9) ? rgb.g : rgb.r);

        // for emission, apply spectral illuminant
        if (is_emission)
            s.values[i] *= lookup_d65(lambda);
    }
    return s;
}

// Piecewise-linear IOR spectrum: point samples at 435 nm (blue), 546 nm (green), 700 nm (red).
Spectral_sample mdl_rgb_to_spectral_ior(inout State state, vec3 rgb)
{
    Spectral_sample s;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        float lambda = state.spectral_wavelengths.values[i];
        if (lambda > 546.0) {
            float t = min((lambda - 546.0) * (1.0 / (700.0 - 546.0)), 1.0);
            s.values[i] = t * rgb.r + (1.0 - t) * rgb.g;
        } else {
            float t = max((lambda - 435.0) * (1.0 / (546.0 - 435.0)), 0.0);
            s.values[i] = t * rgb.g + (1.0 - t) * rgb.b;
        }
    }
    return s;
}

// RGB-to-spectral conversion for reflectance.
Spectral_sample mdl_rgb_to_spectral_reflectance(inout State state, vec3 rgb)
{
    return rgb_to_spectral(rgb, state.spectral_wavelengths, false);
}

// RGB-to-spectral conversion weighted by the D65 illuminant (for emission/luminance).
Spectral_sample mdl_rgb_to_spectral_luminance(inout State state, vec3 rgb)
{
    return rgb_to_spectral(rgb, state.spectral_wavelengths, true);
}

// Volume attenuation coefficients use the same reflectance mapping as non-emission.
Spectral_sample mdl_rgb_to_spectral_volume_coefficient(inout State state, vec3 rgb)
{
    return mdl_rgb_to_spectral_reflectance(state, rgb);
}

// Return the active wavelengths stored in the shading state.
Spectral_sample mdl_get_wavelengths(inout State state)
{
    return state.spectral_wavelengths;
}

#endif // MDL_SPECTRAL_RENDERING

#endif // MDL_RUNTIME_GLSL
