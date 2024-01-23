/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
//   NUM_MATERIAL_TEXTURES_2D          : The number of material 2D textures
//   NUM_MATERIAL_TEXTURES_3D          : The number of material 3D textures
//   SET_MATERIAL_TEXTURES_INDICES     : The set index for the array of material texture array indices
//   SET_MATERIAL_TEXTURES_2D          : The set index for the array of material 2D textures
//   SET_MATERIAL_TEXTURES_3D          : The set index for the array of material 3D textures
//   BINDING_MATERIAL_TEXTURES_INDICES : The binding index for the array of material texture array indices
//   BINDING_MATERIAL_TEXTURES_2D      : The binding index for the array of material 2D textures
//   BINDING_MATERIAL_TEXTURES_3D      : The binding index for the array of material 3D textures

#ifndef MDL_RUNTIME_GLSL
#define MDL_RUNTIME_GLSL

// The array indices of the material textures. This is needed because in MDL the 2D and 3D textures
// are not differentiated and thus, the 2D and 3D sampler arrays should overlap.
// e.g. 2D textures: | A | _ | C | D | _ | _ |
//      3D textures: | _ | B | _ | _ | E | F |
// However, in Vulkan 1.0 without extensions this is not possible since all bindings of a descriptor
// set must be bound to a valid resource. Therefore, we need to tightly pack the texture arrays
// defined below and add the indirection from MDL texture index to the real index. Alternatively,
// the VK_EXT_robustness2 extension allows "null descriptors" which lets the texture arrays be
// sparse. The best solution, however, would be to use "Descriptor Indexing" which is core in
// Vulkan 1.2 or can be used through various extensions in older versions. This would allow
// variable sized texture arrays that can have undefined entries. We show the solution that is
// most compatible with older hardware, but recommend using "Descriptor Indexing" if possible.
#if (NUM_MATERIAL_TEXTURES_2D + NUM_MATERIAL_TEXTURES_3D > 0)
layout(std140, set = SET_MATERIAL_TEXTURES_INDICES, binding = BINDING_MATERIAL_TEXTURES_INDICES)
uniform MaterialTextureIndiciesBuffer
{
    uint uMaterialTextureIndices[NUM_MATERIAL_TEXTURES_2D + NUM_MATERIAL_TEXTURES_3D];
};
#endif // (NUM_MATERIAL_TEXTURES_2D + NUM_MATERIAL_TEXTURES_3D > 0)

// The arrays of material textures used in the texturing functions
#if (NUM_MATERIAL_TEXTURES_2D > 0)
layout(set = SET_MATERIAL_TEXTURES_2D, binding = BINDING_MATERIAL_TEXTURES_2D)
uniform sampler2D uMaterialTextures2D[NUM_MATERIAL_TEXTURES_2D];
#endif // (NUM_MATERIAL_TEXTURES_2D > 0)

#if (NUM_MATERIAL_TEXTURES_3D > 0)
layout(set = SET_MATERIAL_TEXTURES_3D, binding = BINDING_MATERIAL_TEXTURES_3D)
uniform sampler3D uMaterialTextures3D[NUM_MATERIAL_TEXTURES_3D];
#endif // (NUM_MATERIAL_TEXTURES_3D > 0)


//-----------------------------------------------------------------------------
// MDL data types and constants
//-----------------------------------------------------------------------------
#define Tex_wrap_mode            int
#define TEX_WRAP_CLAMP           0
#define TEX_WRAP_REPEAT          1
#define TEX_WRAP_MIRRORED_REPEAT 2
#define TEX_WRAP_CLIP            3

#define Bsdf_event_type          int
#define BSDF_EVENT_ABSORB        0
#define BSDF_EVENT_DIFFUSE       1
#define BSDF_EVENT_GLOSSY       (1 << 1)
#define BSDF_EVENT_SPECULAR     (1 << 2)
#define BSDF_EVENT_REFLECTION   (1 << 3)
#define BSDF_EVENT_TRANSMISSION (1 << 4)

#define BSDF_EVENT_DIFFUSE_REFLECTION    (BSDF_EVENT_DIFFUSE  | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_DIFFUSE_TRANSMISSION  (BSDF_EVENT_DIFFUSE  | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_GLOSSY_REFLECTION     (BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_GLOSSY_TRANSMISSION   (BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_SPECULAR_REFLECTION   (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_SPECULAR_TRANSMISSION (BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION)

#define BSDF_EVENT_FORCE_32_BIT 0xffffffffU

#define Edf_event_type         int
#define EDF_EVENT_NONE         0
#define EDF_EVENT_EMISSION     1
#define EDF_EVENT_FORCE_32_BIT 0xffffffffU

#define BSDF_USE_MATERIAL_IOR (-1.0)

struct State
{
    vec3 normal;
    vec3 geom_normal;
    vec3 position;
    float animation_time;
    vec3 text_coords[1];
    vec3 tangent_u[1];
    vec3 tangent_v[1];
    int ro_data_segment_offset;
    mat4 world_to_object;
    mat4 object_to_world;
    int object_id;
    float meters_per_scene_unit;
    int arg_block_offset;
};

struct Bsdf_sample_data
{
    /*Input*/ vec3 ior1;           // IOR current med
    /*Input*/ vec3 ior2;           // IOR other side
    /*Input*/ vec3 k1;             // outgoing direction
    /*Output*/ vec3 k2;            // incoming direction
    /*Input*/ vec4 xi;             // pseudo-random sample numbers in range [0, 1)
    /*Output*/ float pdf;          // pdf (non-projected hemisphere)
    /*Output*/ vec3 bsdf_over_pdf; // bsdf * dot(normal, k2) / pdf
    /*Output*/ int event_type;     // the type of event for the generated sample
    /*Output*/ int handle;         // handle of the sampled elemental BSDF (lobe)
};

struct Bsdf_evaluate_data
{
    /*Input*/ vec3 ior1;          // IOR current medium
    /*Input*/ vec3 ior2;          // IOR other side
    /*Input*/ vec3 k1;            // outgoing direction
    /*Input*/ vec3 k2;            // incoming direction
    /*Output*/ vec3 bsdf_diffuse; // bsdf_diffuse * dot(normal, k2)
    /*Output*/ vec3 bsdf_glossy;  // bsdf_glossy * dot(normal, k2)
    /*Output*/ float pdf;         // pdf (non-projected hemisphere)
};

struct Bsdf_auxiliary_data
{
    /*Input*/ vec3 ior1;    // IOR current medium
    /*Input*/ vec3 ior2;    // IOR other side
    /*Input*/ vec3 k1;      // outgoing direction
    /*Output*/ vec3 albedo_diffuse;
    /*Output*/ vec3 albedo_glossy;
    /*Output*/ vec3 normal;
    /*Output*/ vec3 roughness;
};

struct Bsdf_pdf_data
{
    /*Input*/ vec3 ior1;  // IOR current medium
    /*Input*/ vec3 ior2;  // IOR other side
    /*Input*/ vec3 k1;    // outgoing direction
    /*Input*/ vec3 k2;    // incoming direction
    /*Output*/ float pdf; // pdf (non-projected hemisphere)
};

struct Edf_sample_data
{
    /*Input*/ vec4 xi;            // pseudo-random sample numbers in range [0, 1)
    /*Output*/ vec3 k1;           // outgoing direction
    /*Output*/ float pdf;         // pdf (non-projected hemisphere)
    /*Output*/ vec3 edf_over_pdf; // edf * dot(normal,k1) / pdf
    /*Output*/ int event_type;    // the type of event for the generated sample
    /*Output*/ int handle;        // handle of the sampled elemental EDF (lobe)
};

struct Edf_evaluate_data
{
    /*Input*/ vec3 k1;    // outgoing direction
    /*Output*/ float cos; // dot(normal, k1)
    /*Output*/ vec3 edf;  // edf
    /*Output*/ float pdf; // pdf (non-projected hemisphere)
};

struct Edf_auxiliary_data
{
    /*Input*/ vec3 k1; // outgoing direction
};

struct Edf_pdf_data
{
    /*Input*/ vec3 k1;    // outgoing direction
    /*Output*/ float pdf; // pdf (non-projected hemisphere)
};


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

#if (NUM_MATERIAL_TEXTURES_2D > 0)
// corresponds to ::tex::width(uniform texture_2d tex, int2 uv_tile, float frame)
int tex_width_2d(int tex, ivec2 uv_tile, float frame)
{
    if (tex == 0) return 0; // invalid texture
    uint texture_index = uMaterialTextureIndices[tex - 1];
    return textureSize(uMaterialTextures2D[texture_index], 0).x;
}

// corresponds to ::tex::height(uniform texture_2d tex, int2 uv_tile, float frame)
int tex_height_2d(int tex, ivec2 uv_tile, float frame)
{
    if (tex == 0) return 0; // invalid texture
    uint texture_index = uMaterialTextureIndices[tex - 1];
    return textureSize(uMaterialTextures2D[texture_index], 0).y;
}

// corresponds to ::tex::lookup_float4(uniform texture_2d tex, float2 coord, ...)
vec4 tex_lookup_float4_2d(int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    if (tex == 0) return vec4(0.0); // invalid texture

    if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x >= 1.0))
        return vec4(0.0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y >= 1.0))
        return vec4(0.0);

    uint texture_index = uMaterialTextureIndices[tex - 1];
    ivec2 tex_size = textureSize(uMaterialTextures2D[texture_index], 0);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, tex_size.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, tex_size.y);
    coord = apply_smootherstep_filter(coord, tex_size);

    return texture(uMaterialTextures2D[texture_index], coord);
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

    uint texture_index = uMaterialTextureIndices[tex - 1];
    ivec2 res = textureSize(uMaterialTextures2D[texture_index], 0);
    if (coord.x < 0 || coord.y < 0 || coord.x >= res.x || coord.y >= res.y)
        return vec4(0.0); // out of bounds

    return texelFetch(uMaterialTextures2D[texture_index], coord, 0);
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
#endif // (NUM_MATERIAL_TEXTURES_2D > 0)


//-----------------------------------------------------------------------------
// Texture function implementations, 3D
//-----------------------------------------------------------------------------

#if (NUM_MATERIAL_TEXTURES_3D > 0)
// corresponds to ::tex::width(uniform texture_3d tex, float frame)
int tex_width_3d(int tex, float frame)
{
    if (tex == 0) return 0; // invalid texture
    uint texture_index = uMaterialTextureIndices[tex - 1];
    return textureSize(uMaterialTextures3D[texture_index], 0).x;
}

// corresponds to ::tex::height(uniform texture_3d tex, float frame)
int tex_height_3d(int tex, float frame)
{
    if (tex == 0) return 0; // invalid texture
    uint texture_index = uMaterialTextureIndices[tex - 1];
    return textureSize(uMaterialTextures3D[texture_index], 0).y;
}

// corresponds to ::tex::depth(uniform texture_3d tex, float frame)
int tex_depth_3d(int tex, float frame)
{
    if (tex == 0) return 0; // invalid texture
    uint texture_index = uMaterialTextureIndices[tex - 1];
    return textureSize(uMaterialTextures3D[texture_index], 0).z;
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

    uint texture_index = uMaterialTextureIndices[tex - 1];
    ivec3 tex_size = textureSize(uMaterialTextures3D[texture_index], 0);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, tex_size.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, tex_size.y);
    coord.z = apply_wrap_and_crop(coord.z, wrap_w, crop_w, tex_size.z);

    return texture(uMaterialTextures3D[texture_index], coord);
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

    uint texture_index = uMaterialTextureIndices[tex - 1];
    ivec3 res = textureSize(uMaterialTextures3D[texture_index], 0);
    if (coord.x < 0 || coord.y < 0 || coord.z < 0 || coord.x >= res.x || coord.y >= res.y || coord.z >= res.z)
        return vec4(0.0); // out of bounds

    return texelFetch(uMaterialTextures3D[texture_index], coord, 0);
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
#endif // (NUM_MATERIAL_TEXTURES_3D > 0)


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

#endif // MDL_RUNTIME_GLSL
