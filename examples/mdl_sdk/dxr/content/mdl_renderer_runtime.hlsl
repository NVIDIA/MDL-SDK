/******************************************************************************
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#if !defined(MDL_RENDERER_RUNTIME_HLSLI)
#define MDL_RENDERER_RUNTIME_HLSLI

// compiler constants defined from outside:
// - MDL_TARGET_REGISTER_SPACE
// - MDL_TARGET_RO_DATA_SEGMENT_SLOT
//
// - MDL_MATERIAL_REGISTER_SPACE
// - MDL_MATERIAL_ARGUMENT_BLOCK_SLOT
// - MDL_MATERIAL_TEXTURE_INFO_SLOT
// - MDL_MATERIAL_MBSDF_INFO_SLOT

// - MDL_MATERIAL_TEXTURE_2D_REGISTER_SPACE
// - MDL_MATERIAL_TEXTURE_3D_REGISTER_SPACE
// - MDL_MATERIAL_TEXTURE_SLOT_BEGIN
// 
// - MDL_MATERIAL_BUFFER_REGISTER_SPACE
// - MDL_MATERIAL_BUFFER_SLOT_BEGIN
//
// - MDL_TEXTURE_SAMPLER_SLOT
// - MDL_LIGHT_PROFILE_SAMPLER_SLOT
// - MDL_MBSDF_SAMPLER_SLOT


/// Information passed to GPU for mapping id requested in the runtime functions to texture
/// views of the corresponding type.
struct Mdl_texture_info
{
    // index into the tex2d, tex3d, ... buffers, depending on the type requested
    uint gpu_resource_array_start;

    // number resources (e.g. uv-tiles) that belong to this resource
    uint gpu_resource_array_size;

    // frame number of the first texture/uv-tile
    int gpu_resource_frame_first;

    // coordinate of the left bottom most uv-tile (also bottom left corner)
    int2 gpu_resource_uvtile_min;

    // in case of uv-tiled textures, required to calculate a linear index (u + v * width
    uint gpu_resource_uvtile_width;
    uint gpu_resource_uvtile_height;

    // get the last frame of an animated texture
    int get_last_frame()
    {
        return gpu_resource_array_size / (gpu_resource_uvtile_width * gpu_resource_uvtile_height)
            + gpu_resource_frame_first - 1;
    }

    // return the resource view index for a given uv-tile id. (see compute_uvtile_and_update_uv(...))
    // returning of -1 indicates out of bounds, 0 refers to the invalid resource.
    int compute_uvtile_id(float frame, int2 uv_tile)
    {
        if (gpu_resource_array_size == 1) // means no uv-tiles
            return int(gpu_resource_array_start);

        // simplest handling possible
        int frame_number = floor(frame) - gpu_resource_frame_first;

        uv_tile -= gpu_resource_uvtile_min;
        const int offset = uv_tile.x +
                           uv_tile.y * int(gpu_resource_uvtile_width) +
                           frame_number * int(gpu_resource_uvtile_width) * int(gpu_resource_uvtile_height);
        if (frame_number < 0 || uv_tile.x < 0 || uv_tile.y < 0 ||
            uv_tile.x >= int(gpu_resource_uvtile_width) ||
            uv_tile.y >= int(gpu_resource_uvtile_height) ||
            offset >= gpu_resource_array_size)
            return -1; // out of bounds

        return int(gpu_resource_array_start) + offset;
    }

    // for uv-tiles, uv coordinate implicitly specifies which resource to use
    // the index of the resource is returned while the uv mapped into the uv-tile
    // if uv-tiles are not used, the data is just passed through
    // returning of -1 indicates out of bounds, 0 refers to the invalid resource.
    int compute_uvtile_and_update_uv(float frame, inout float2 uv)
    {
        if(gpu_resource_array_size == 1) // means no uv-tiles
            return int(gpu_resource_array_start);

        // uv-coordinate in the tile
        const int2 uv_tile = int2(floor(uv)); // floor
        uv = frac(uv);

        // compute a linear index
        return compute_uvtile_id(frame, uv_tile);
    }

    // for texel fetches the uv tile is given explicitly
    int compute_uvtile_and_update_uv(float frame, int2 uv_tile)
    {
        if (gpu_resource_array_size == 1) // means no uv-tiles
            return int(gpu_resource_array_start);

        // compute a linear index
        return compute_uvtile_id(frame, uv_tile);
    }
};

/// Information passed to the GPU for each light profile resource
struct Mdl_light_profile_info
{
    // angular resolution of the grid and its inverse
    uint2 angular_resolution;
    float2 inv_angular_resolution;

    // starting angles of the grid
    float2 theta_phi_start;

    // angular step size and its inverse
    float2 theta_phi_delta;
    float2 theta_phi_inv_delta;

    // factor to rescale the normalized data
    // also represents the maximum candela value of the data
    float candela_multiplier;

    // power (radiant flux)
    float total_power;

    // index into the textures_2d array
    // -  texture contains normalized data sampled on grid
    uint eval_data_index;

    // index into the buffers
    // - CDFs for sampling a light profile
    uint sample_data_index;
};

/// Information passed to the GPU for each BSDF measurement resource
struct Mdl_mbsdf_info
{
    // if the MBSDF has data for reflection (0) and transmission (1)
    uint2 has_data;

    // index into the texture_3d array for both parts
    // - texture contains the measurement values for evaluation
    uint2 eval_data_index;

    // indices into the buffers array for both parts
    // - sample_data buffer contains CDFs for sampling
    // - albedo_data buffer contains max albedos for each theta (isotropic)
    uint2 sample_data_index;
    uint2 albedo_data_index;

    // maximum albedo values for both parts, used for limiting the multiplier
    float2 max_albedo;

    // discrete angular resolution for both parts
    uint2 angular_resolution_theta;
    uint2 angular_resolution_phi;

    // number of color channels (1 for scalar, 3 for rgb) for both parts
    uint2 num_channels;
};

// per target data
ByteAddressBuffer mdl_ro_data_segment : register(MDL_TARGET_RO_DATA_SEGMENT_SLOT, MDL_TARGET_REGISTER_SPACE);

// per material data
// - argument block contains dynamic parameter data exposed in class compilation mode
ByteAddressBuffer mdl_argument_block : register(MDL_MATERIAL_ARGUMENT_BLOCK_SLOT, MDL_MATERIAL_REGISTER_SPACE);
// - resource infos map resource IDs, generated by the SDK, to actual buffer views
StructuredBuffer<Mdl_texture_info> mdl_texture_infos : register(MDL_MATERIAL_TEXTURE_INFO_SLOT, MDL_MATERIAL_REGISTER_SPACE);
// - light profile infos
StructuredBuffer<Mdl_light_profile_info> mdl_light_profile_infos : register(MDL_MATERIAL_LIGHT_PROFILE_INFO_SLOT, MDL_MATERIAL_REGISTER_SPACE);
// - bsdf measurement infos
StructuredBuffer<Mdl_mbsdf_info> mdl_mbsdf_infos : register(MDL_MATERIAL_MBSDF_INFO_SLOT, MDL_MATERIAL_REGISTER_SPACE);
// - texture views, unbound and overlapping for 2D and 3D resources
Texture2D mdl_textures_2d[] : register(MDL_MATERIAL_TEXTURE_SLOT_BEGIN, MDL_MATERIAL_TEXTURE_2D_REGISTER_SPACE);
Texture3D mdl_textures_3d[] : register(MDL_MATERIAL_TEXTURE_SLOT_BEGIN, MDL_MATERIAL_TEXTURE_3D_REGISTER_SPACE);
// - buffer views, unbound. contains sampling and albedo buffers for MBSDFs and sampling buffers for light profiles
StructuredBuffer<float> mdl_buffers[] : register(MDL_MATERIAL_BUFFER_SLOT_BEGIN, MDL_MATERIAL_BUFFER_REGISTER_SPACE);

// mesh data, includes the per mesh scene data
ByteAddressBuffer vertices : register(t1, space0);

// instance data
// - scene data buffer for object/instance data
ByteAddressBuffer scene_data : register(t3, space0);
// - mapping between scene_data_id and scene data buffer layout
StructuredBuffer<SceneDataInfo> scene_data_infos: register(t4, space0);

// global samplers
SamplerState mdl_sampler_tex : register(MDL_TEXTURE_SAMPLER_SLOT);
SamplerState mdl_sampler_light_profile : register(MDL_LIGHT_PROFILE_SAMPLER_SLOT);
SamplerState mdl_sampler_mbsdf : register(MDL_MBSDF_SAMPLER_SLOT);

// If USE_RES_DATA is defined, add a Res_data parameter to all resource handler functions.
// This example doesn't use it, so we only put a dummy field in Res_data.
#if USE_RES_DATA

struct Res_data
{
    uint dummy;
};

#define RES_DATA_PARAM_DECL     Res_data res_data,
#define RES_DATA_PARAM          res_data,

#else

#define RES_DATA_PARAM_DECL
#define RES_DATA_PARAM

#endif

// ------------------------------------------------------------------------------------------------
// Argument block access for dynamic parameters in class compilation mode
// ------------------------------------------------------------------------------------------------

float mdl_read_argblock_as_float(int offs)
{
    return asfloat(mdl_argument_block.Load(offs));
}

double mdl_read_argblock_as_double(int offs)
{
    return asdouble(mdl_argument_block.Load(offs), mdl_argument_block.Load(offs + 4));
}

int mdl_read_argblock_as_int(int offs)
{
    return asint(mdl_argument_block.Load(offs));
}

uint mdl_read_argblock_as_uint(int offs)
{
    return mdl_argument_block.Load(offs);
}

bool mdl_read_argblock_as_bool(int offs)
{
    uint val = mdl_argument_block.Load(offs & ~3);
    return (val & (0xffU << (8 * (offs & 3)))) != 0;
}

float mdl_read_rodata_as_float(int offs)
{
    return asfloat(mdl_ro_data_segment.Load(offs));
}

double mdl_read_rodata_as_double(int offs)
{
    return asdouble(mdl_ro_data_segment.Load(offs), mdl_ro_data_segment.Load(offs + 4));
}

int mdl_read_rodata_as_int(int offs)
{
    return asint(mdl_ro_data_segment.Load(offs));
}

int mdl_read_rodata_as_uint(int offs)
{
    return mdl_ro_data_segment.Load(offs);
}

bool mdl_read_rodata_as_bool(int offs)
{
    uint val = mdl_ro_data_segment.Load(offs & ~3);
    return (val & (0xffU << (8 * (offs & 3)))) != 0;
}

// ------------------------------------------------------------------------------------------------
// Texturing functions, check if valid
// ------------------------------------------------------------------------------------------------

// corresponds to ::tex::texture_isvalid(uniform texture_2d tex)
// corresponds to ::tex::texture_isvalid(uniform texture_3d tex)
// corresponds to ::tex::texture_isvalid(uniform texture_cube tex) // not supported by this example
// corresponds to ::tex::texture_isvalid(uniform texture_ptex tex) // not supported by this example
bool tex_texture_isvalid(RES_DATA_PARAM_DECL int tex)
{
    // assuming that there is no indexing out of bounds of the resource_infos and the view arrays
    return tex != 0; // invalid texture
}

// helper function to realize wrap and crop.
// Out of bounds case for TEX_WRAP_CLIP must already be handled.
float apply_wrap_and_crop(
    float coord,
    int wrap,
    float2 crop,
    int res)
{
    if (wrap != TEX_WRAP_REPEAT || any(crop != float2(0, 1)))
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
            float inv_hdim = 0.5f / float(res);
            coord = clamp(coord, inv_hdim, 1.f - inv_hdim);
        }
        coord = coord * (crop.y - crop.x) + crop.x;
    }
    return coord;
}

// Modify texture coordinates to get better texture filtering,
// see http://www.iquilezles.org/www/articles/texture/texture.htm
float2 apply_smootherstep_filter(float2 uv, uint2 size)
{
    float2 res;
    res = uv * size + 0.5f;

    float u_i = floor(res.x), v_i = floor(res.y);
    float u_f = res.x - u_i;
    float v_f = res.y - v_i;
    u_f = u_f * u_f * u_f * (u_f * (u_f * 6.f - 15.f) + 10.f);
    v_f = v_f * v_f * v_f * (v_f * (v_f * 6.f - 15.f) + 10.f);
    res.x = u_i + u_f;
    res.y = v_i + v_f;

    res = (res - 0.5f) / size;
    return res;
}

// ------------------------------------------------------------------------------------------------
// Texturing functions, 2D
// ------------------------------------------------------------------------------------------------

int2 tex_res_2d(RES_DATA_PARAM_DECL int tex, int2 uv_tile, float frame)
{
    if (tex == 0) return uint2(0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    int array_index = info.compute_uvtile_id(frame, uv_tile);
    if (array_index < 0) return uint2(0, 0); // out of bounds or no uv-tile

    uint2 res;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y);
    return int2(res);
}

// corresponds to ::tex::width(uniform texture_2d tex, int2 uv_tile, float frame)
int tex_width_2d(RES_DATA_PARAM_DECL int tex, int2 uv_tile, float frame)
{
    return tex_res_2d(RES_DATA_PARAM tex, uv_tile, frame).x;
}

// corresponds to ::tex::height(uniform texture_2d tex, int2 uv_tile)
int tex_height_2d(RES_DATA_PARAM_DECL int tex, int2 uv_tile, float frame)
{
    return tex_res_2d(RES_DATA_PARAM tex, uv_tile, frame).y;
}

// corresponds to ::tex::first__frame(uniform texture_2d)
int tex_first_frame_2d(RES_DATA_PARAM_DECL int tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds
    return info.gpu_resource_frame_first;
}

// corresponds to ::tex::last_frame(uniform texture_2d)
int tex_last_frame_2d(RES_DATA_PARAM_DECL int tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds
    return info.get_last_frame();
}

// corresponds to ::tex::lookup_float4(uniform texture_2d tex, float2 coord, ...)
float4 tex_lookup_float4_2d(
    RES_DATA_PARAM_DECL
    int tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    // handle uv-tiles and/or get texture array index
    int array_index = info.compute_uvtile_and_update_uv(frame, coord);
    if (array_index < 0) return float4(0, 0, 0, 0); // out of bounds or no uv-tile

    if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x >= 1.0))
        return float4(0, 0, 0, 0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y >= 1.0))
        return float4(0, 0, 0, 0);

    uint2 res;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, res.x);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, res.y);

    coord = apply_smootherstep_filter(coord, res);

    // Note, since we don't have ddx and ddy in the compute pipeline, TextureObject::Sample() is not
    // available, we use SampleLevel instead and go for the most detailed level. Therefore, we don't
    // need mipmaps. Manual mip level computation is possible though.
    return mdl_textures_2d[NonUniformResourceIndex(array_index)].SampleLevel(
        mdl_sampler_tex, coord, /*lod=*/ 0.0f, /*offset=*/ int2(0, 0));
}

float3 tex_lookup_float3_2d(RES_DATA_PARAM_DECL int tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float3 tex_lookup_color_2d(RES_DATA_PARAM_DECL int tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float2 tex_lookup_float2_2d(RES_DATA_PARAM_DECL int tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_float_2d(RES_DATA_PARAM_DECL int tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}

// corresponds to ::tex::lookup_float4(uniform texture_2d tex, float2 coord, ...) when derivatives are enabled
float4 tex_lookup_deriv_float4_2d(
    RES_DATA_PARAM_DECL
    int tex,
    Derived_float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    // handle uv-tiles and/or get texture array index
    int array_index = info.compute_uvtile_and_update_uv(frame, coord.val);
    if (array_index < 0) return float4(0, 0, 0, 0); // out of bounds or no uv-tile

    if (wrap_u == TEX_WRAP_CLIP && (coord.val.x < 0.0 || coord.val.x >= 1.0))
        return float4(0, 0, 0, 0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.val.y < 0.0 || coord.val.y >= 1.0))
        return float4(0, 0, 0, 0);

    uint2 res;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y);
    coord.val.x = apply_wrap_and_crop(coord.val.x, wrap_u, crop_u, res.x);
    coord.val.y = apply_wrap_and_crop(coord.val.y, wrap_v, crop_v, res.y);

    coord.val = apply_smootherstep_filter(coord.val, res);

    // Note, since we don't have ddx and ddy in the compute pipeline, TextureObject::Sample() is not
    // available, we use SampleLevel instead and go for the most detailed level. Therefore, we don't
    // need mipmaps. Manual mip level computation is possible though.
    return mdl_textures_2d[NonUniformResourceIndex(array_index)].SampleGrad(
        mdl_sampler_tex, coord.val, coord.dx, coord.dy, /*offset=*/ int2(0, 0));
}

float3 tex_lookup_deriv_float3_2d(RES_DATA_PARAM_DECL int tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float3 tex_lookup_deriv_color_2d(RES_DATA_PARAM_DECL int tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float2 tex_lookup_deriv_float2_2d(RES_DATA_PARAM_DECL int tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_deriv_float_2d(RES_DATA_PARAM_DECL int tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}


// corresponds to ::tex::texel_float4(uniform texture_2d tex, float2 coord, int2 uv_tile)
float4 tex_texel_float4_2d(
    RES_DATA_PARAM_DECL
    int tex,
    int2 coord,
    int2 uv_tile,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    // handle uv-tiles and/or get texture array index
    int array_index = info.compute_uvtile_and_update_uv(frame, uv_tile);
    if (array_index < 0) return float4(0, 0, 0, 0); // out of bounds or no uv-tile

    uint2 res;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y);
    if (0 > coord.x || res.x <= coord.x || 0 > coord.y || res.y <= coord.y)
        return float4(0, 0, 0, 0); // out of bounds

    return mdl_textures_2d[NonUniformResourceIndex(array_index)].Load(int3(coord, /*mipmaplevel=*/ 0));
}

float3 tex_texel_float3_2d(RES_DATA_PARAM_DECL int tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile, frame).xyz;
}

float3 tex_texel_color_2d(RES_DATA_PARAM_DECL int tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float3_2d(RES_DATA_PARAM tex, coord, uv_tile, frame);
}

float2 tex_texel_float2_2d(RES_DATA_PARAM_DECL int tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile, frame).xy;
}

float tex_texel_float_2d(RES_DATA_PARAM_DECL int tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile, frame).x;
}


// ------------------------------------------------------------------------------------------------
// Texturing functions, 3D
// ------------------------------------------------------------------------------------------------

int3 tex_res_3d(RES_DATA_PARAM_DECL int tex, float frame)
{
    if (tex == 0) return uint3(0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    // no uv-tiles for 3D textures (shortcut the index calculation)
    int array_index = info.gpu_resource_array_start;

    uint3 res;
    mdl_textures_3d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y, res.z);
    return int3(res);
}

// corresponds to ::tex::first__frame(uniform texture_3d)
int tex_first_frame_3d(RES_DATA_PARAM_DECL int tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds
    return info.gpu_resource_frame_first;
}

// corresponds to ::tex::last_frame(uniform texture_3d)
int tex_last_frame_3d(RES_DATA_PARAM_DECL int tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds
    return info.get_last_frame();
}

// corresponds to ::tex::width(uniform texture_3d tex, int2 uv_tile)
int tex_width_3d(RES_DATA_PARAM_DECL int tex, float frame) { return tex_res_3d(RES_DATA_PARAM tex, frame).x; }

// corresponds to ::tex::height(uniform texture_3d tex, int2 uv_tile)
int tex_height_3d(RES_DATA_PARAM_DECL int tex, float frame) { return tex_res_3d(RES_DATA_PARAM tex, frame).y; }

// corresponds to ::tex::depth(uniform texture_3d tex, int2 uv_tile)
int tex_depth_3d(RES_DATA_PARAM_DECL int tex, float frame) { return tex_res_3d(RES_DATA_PARAM tex, frame).z; }

// corresponds to ::tex::lookup_float4(uniform texture_3d tex, float2 coord, ...)
float4 tex_lookup_float4_3d(
    RES_DATA_PARAM_DECL
    int tex,
    float3 coord,
    int wrap_u,
    int wrap_v,
    int wrap_w,
    float2 crop_u,
    float2 crop_v,
    float2 crop_w,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x >= 1.0))
        return float4(0, 0, 0, 0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y >= 1.0))
        return float4(0, 0, 0, 0);
    if (wrap_w == TEX_WRAP_CLIP && (coord.z < 0.0 || coord.z >= 1.0))
        return float4(0, 0, 0, 0);

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    // no uv-tiles for 3D textures (shortcut the index calculation)
    int array_index = info.gpu_resource_array_start;

    uint width, height, depth;
    mdl_textures_3d[NonUniformResourceIndex(array_index)].GetDimensions(width, height, depth);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, width);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, height);
    coord.z = apply_wrap_and_crop(coord.z, wrap_w, crop_w, depth);

    // Note, since we don't have ddx and ddy in the compute pipeline, TextureObject::Sample() is not
    // available, we use SampleLevel instead and go for the most detailed level. Therefore, we don't
    // need mipmaps. Manual mip level computation is possible though.
    return mdl_textures_3d[NonUniformResourceIndex(array_index)].SampleLevel(
        mdl_sampler_tex, coord, /*lod=*/ 0.0f, /*offset=*/ int3(0, 0, 0));
}

float3 tex_lookup_float3_3d(RES_DATA_PARAM_DECL int tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

float3 tex_lookup_color_3d(RES_DATA_PARAM_DECL int tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

float2 tex_lookup_float2_3d(RES_DATA_PARAM_DECL int tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xy;
}

float tex_lookup_float_3d(RES_DATA_PARAM_DECL int tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).x;
}

// corresponds to ::tex::texel_float4(uniform texture_3d tex, float3 coord)
float4 tex_texel_float4_3d(
    RES_DATA_PARAM_DECL
    int tex,
    int3 coord,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_texture_info info = mdl_texture_infos[tex - 1]; // assuming this is in bounds

    // no uv-tiles for 3D textures (shortcut the index calculation)
    int array_index = info.gpu_resource_array_start;

    uint3 res;
    mdl_textures_3d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y, res.z);
    if (0 > coord.x || res.x <= coord.x || 0 > coord.y || res.y <= coord.y || 0 > coord.z || res.z <= coord.z)
        return float4(0, 0, 0, 0); // out of bounds

    return mdl_textures_3d[NonUniformResourceIndex(array_index)].Load(int4(coord, /*mipmaplevel=*/ 0));
}

float3 tex_texel_float3_3d(RES_DATA_PARAM_DECL int tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord, frame).xyz;
}

float3 tex_texel_color_3d(RES_DATA_PARAM_DECL int tex, int3 coord, float frame)
{
    return tex_texel_float3_3d(RES_DATA_PARAM tex, coord, frame);
}

float2 tex_texel_float2_3d(RES_DATA_PARAM_DECL int tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord, frame).xy;
}

float tex_texel_float_3d(RES_DATA_PARAM_DECL int tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord, frame).x;
}


// ------------------------------------------------------------------------------------------------
// Texturing functions, Cube (not supported by this example)
// ------------------------------------------------------------------------------------------------

int tex_width_cube(RES_DATA_PARAM_DECL int tex) { return 0; }
int tex_height_cube(RES_DATA_PARAM_DECL int tex) { return 0; }

float4 tex_lookup_float4_cube(RES_DATA_PARAM_DECL int tex, float3 coord)
{
    return float4(0, 0, 0, 0);
}

float3 tex_lookup_float3_cube(RES_DATA_PARAM_DECL int tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float3 tex_lookup_color_cube(RES_DATA_PARAM_DECL int tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float2 tex_lookup_float2_cube(RES_DATA_PARAM_DECL int tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xy;
}

float tex_lookup_float_cube(RES_DATA_PARAM_DECL int tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).x;
}

float4 tex_texel_float4_cube(RES_DATA_PARAM_DECL int tex, int3 coord)
{
    return float4(0, 0, 0, 0);
}

float3 tex_texel_float3_cube(RES_DATA_PARAM_DECL int tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float3 tex_texel_color_cube(RES_DATA_PARAM_DECL int tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float2 tex_texel_float2_cube(RES_DATA_PARAM_DECL int tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xy;
}

float tex_texel_float_cube(RES_DATA_PARAM_DECL int tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).x;
}

// ------------------------------------------------------------------------------------------------
// Texturing functions, PTEX (not supported by this example)
// ------------------------------------------------------------------------------------------------


float4 tex_lookup_float4_ptex(RES_DATA_PARAM_DECL int tex, int channel)
{
    return float4(0, 0, 0, 0);
}

float3 tex_lookup_float3_ptex(RES_DATA_PARAM_DECL int tex, int channel)
{
    return tex_lookup_float4_ptex(RES_DATA_PARAM tex, channel).xyz;
}

float3 tex_lookup_color_ptex(RES_DATA_PARAM_DECL int tex, int channel)
{
    return tex_lookup_float3_ptex(RES_DATA_PARAM tex, channel);
}

float2 tex_lookup_float2_ptex(RES_DATA_PARAM_DECL int tex, int channel)
{
    return tex_lookup_float4_ptex(RES_DATA_PARAM tex, channel).xy;
}

float tex_lookup_float_ptex(RES_DATA_PARAM_DECL int tex, int channel)
{
    return tex_lookup_float4_ptex(RES_DATA_PARAM tex, channel).x;
}

// ------------------------------------------------------------------------------------------------
// Light Profiles (not supported by this example)
// ------------------------------------------------------------------------------------------------

// binary search through CDF
uint sample_cdf(StructuredBuffer<float> cdf, uint cdf_offset, uint cdf_size, float xi)
{
    uint li = 0;
    uint ri = cdf_size - 1;
    uint m = (li + ri) / 2;
    while (ri > li)
    {
        if (xi < cdf[cdf_offset + m])
            ri = m;
        else
            li = m + 1;
        m = (li + ri) / 2;
    }
    return m;
}

bool df_light_profile_isvalid(RES_DATA_PARAM_DECL int lp_idx)
{
    // assuming that there is no indexing out of bounds of the light_profile_infos and the view arrays
    return lp_idx != 0; // 0 is the invalid light profile
}

float df_light_profile_power(RES_DATA_PARAM_DECL int lp_idx)
{
    if (lp_idx == 0) return 0; // invalid light profile

    const Mdl_light_profile_info lp = mdl_light_profile_infos[lp_idx - 1]; // assuming this is in bounds
    return lp.total_power;
}

float df_light_profile_maximum(RES_DATA_PARAM_DECL int lp_idx)
{
    if (lp_idx == 0) return 0; // invalid light profile

    const Mdl_light_profile_info lp = mdl_light_profile_infos[lp_idx - 1]; // assuming this is in bounds
    return lp.candela_multiplier;
}

float df_light_profile_evaluate(
    RES_DATA_PARAM_DECL
    int   lp_idx,
    float2 theta_phi)
{
    if (lp_idx == 0) return 0; // invalid light profile

    const Mdl_light_profile_info lp = mdl_light_profile_infos[lp_idx - 1]; // assuming this is in bounds

    // map theta to 0..1 range
    float u = (theta_phi[0] - lp.theta_phi_start[0]) *
        lp.theta_phi_inv_delta.x * lp.inv_angular_resolution.x;

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi[1] > 0.0f) ? theta_phi[1] : (2.0 * M_PI + theta_phi[1]);

    // floor wraps phi range into 0..2pi
    phi = phi - lp.theta_phi_start.y -
        floor((phi - lp.theta_phi_start.y) * (0.5 * M_ONE_OVER_PI)) * (2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handled by the (black) border
    // since it implies lp.theta_phi_start.y > 0 (and we really have "no data" below that)
    float v = phi * lp.theta_phi_inv_delta.y * lp.inv_angular_resolution.y;

    // half pixel offset for linear filtering
    u += 0.5f * lp.inv_angular_resolution.x;
    v += 0.5f * lp.inv_angular_resolution.y;

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return 0.0f;

    float value = mdl_textures_2d[NonUniformResourceIndex(lp.eval_data_index)].SampleLevel(
        mdl_sampler_light_profile, float2(u, v), /*lod=*/ 0.0f, /*offset=*/ int2(0, 0)).x;
    return value * lp.candela_multiplier;
}

float3 df_light_profile_sample(
    RES_DATA_PARAM_DECL
    int   lp_idx,
    float3 xi)
{
    float3 result = float3(
        -1.0, // negative theta (x value) means no emission
        -1.0,
         0.0);
    if (lp_idx == 0) return result; // invalid light profile

    const Mdl_light_profile_info lp = mdl_light_profile_infos[lp_idx - 1]; // assuming this is in bounds
    StructuredBuffer<float> sample_data = mdl_buffers[NonUniformResourceIndex(lp.sample_data_index)];
    const uint2 res = lp.angular_resolution;

    // sample theta_out
    //-------------------------------------------
    uint idx_theta = sample_cdf(sample_data, res.x - 1, 0, xi[0]); // binary search

    float prob_theta = sample_data[idx_theta];
    if (idx_theta > 0)
    {
        const float tmp = sample_data[idx_theta - 1];
        prob_theta -= tmp;
        xi[0] -= tmp;
    }
    xi[0] /= prob_theta;  // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    const uint phi_data_offset = (res.x - 1) +              // CDF theta block
                                 (idx_theta * (res.y - 1)); // selected CDF for phi
    const uint idx_phi = sample_cdf(sample_data, res.y - 1, phi_data_offset, xi[1]); // binary search

    float prob_phi = sample_data[phi_data_offset + idx_phi];
    if (idx_phi > 0)
    {
        const float tmp = sample_data[phi_data_offset + idx_phi - 1];
        prob_phi -= tmp;
        xi[1] -= tmp;
    }
    xi[1] /= prob_phi;  // rescale for re-usage

    // compute theta and phi
    //-------------------------------------------
    // sample uniformly within the patch (grid cell)
    const float2 start = lp.theta_phi_start;
    const float2 delta = lp.theta_phi_delta;

    const float cos_theta_0 = cos(start.x + float(idx_theta) * delta.x);
    const float cos_theta_1 = cos(start.x + float(idx_theta + 1u) * delta.x);

    //               n = \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    //                 = 1 / (\cos{\theta_0} - \cos{\theta_1})
    //
    //             \xi = n * \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    // => \cos{\theta} = (1 - \xi) \cos{\theta_0} + \xi \cos{\theta_1}

    const float cos_theta = lerp(cos_theta_0, cos_theta_1, xi[1]);
    result[0] = acos(cos_theta);
    result[1] = start.y + (float(idx_phi) + xi[0]) * delta.y;

    // align phi
    if (result[1] > 2.0 * M_PI) result[1] -= 2.0 * M_PI;             // wrap
    if (result[1] > 1.0 * M_PI) result[1] = -2.0 * M_PI + result[1]; // to [-pi, pi]

    // compute pdf
    //-------------------------------------------
    result[2] = prob_theta * prob_phi / (delta.y * (cos_theta_0 - cos_theta_1));

    return result;
}

float df_light_profile_pdf(
    RES_DATA_PARAM_DECL
    int   lp_idx,
    float2 theta_phi)
{
    if (lp_idx == 0) return 0; // invalid light profile

    const Mdl_light_profile_info lp = mdl_light_profile_infos[lp_idx - 1]; // assuming this is in bounds
    StructuredBuffer<float> sample_data = mdl_buffers[NonUniformResourceIndex(lp.sample_data_index)];
    const uint2 res = lp.angular_resolution;

    // map theta to 0..1 range
    const float theta = theta_phi[0] - lp.theta_phi_start.x;
    const int idx_theta = int(theta * lp.theta_phi_inv_delta.x);

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi[1] > 0.0) ? theta_phi[1] : (2.0 * M_PI + theta_phi[1]);

    // floorf wraps phi range into 0..2pi
    phi = phi - lp.theta_phi_start.y -
        floor((phi - lp.theta_phi_start.y) * (0.5 * M_ONE_OVER_PI)) * (2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border
    // since it implies lp.theta_phi_start.y > 0 (and we really have "no data" below that)
    const int idx_phi = int(phi * lp.theta_phi_inv_delta.y);

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (idx_theta < 0 || idx_theta > res.x - 2 || idx_phi < 0 || idx_phi > res.x - 2)
        return 0;

    // get probability for theta
    //-------------------------------------------
    float prob_theta = sample_data[idx_theta];
    if (idx_theta > 0)
        prob_theta -= sample_data[idx_theta - 1];

    // get probability for phi
    //-------------------------------------------
    const uint phi_data_offset = (res.x - 1)                // CDF theta block
                               + (idx_theta * (res.y - 1)); // selected CDF for phi

    float prob_phi = sample_data[phi_data_offset + idx_phi];
    if (idx_phi > 0)
        prob_phi -= sample_data[phi_data_offset + idx_phi - 1];

    // compute probability to select a position in the sphere patch
    const float2 start = lp.theta_phi_start;
    const float2 delta = lp.theta_phi_delta;

    const float cos_theta_0 = cos(start.x + float(idx_theta) * delta.x);
    const float cos_theta_1 = cos(start.x + float(idx_theta + 1u) * delta.x);

    return prob_theta * prob_phi / (delta.y * (cos_theta_0 - cos_theta_1));
}

// ------------------------------------------------------------------------------------------------
// Measured BSDFs
// ------------------------------------------------------------------------------------------------

float3 bsdf_compute_uvw(float2 theta_phi_in, float2 theta_phi_out)
{
    // assuming each phi is between -pi and pi
    float u = theta_phi_out[1] - theta_phi_in[1];
    if (u < 0.0) u += 2.0 * M_PI;
    if (u > M_PI) u = 2.0 * M_PI - u;
    u *= M_ONE_OVER_PI;

    const float v = theta_phi_out[0] * (2.0 * M_ONE_OVER_PI);
    const float w = theta_phi_in[0] * (2.0 * M_ONE_OVER_PI);

    return float3(u, v, w);
}

bool df_bsdf_measurement_isvalid(RES_DATA_PARAM_DECL int bm_idx)
{
    // assuming that there is no indexing out of bounds of the mbsdf_infos and the view arrays
    return bm_idx != 0; // 0 is the invalid bsdf measurement
}

int3 df_bsdf_measurement_resolution(RES_DATA_PARAM_DECL int bm_idx, int part)
{
    if (bm_idx == 0) return int3(0, 0, 0); // invalid bsdf measurement
    
    const Mdl_mbsdf_info bm = mdl_mbsdf_infos[bm_idx - 1]; // assuming this is in bounds
    if (bm.has_data[part] == 0)
        return int3(0, 0, 0);

    return int3(
        bm.angular_resolution_theta[part],
        bm.angular_resolution_phi[part],
        bm.num_channels[part]);
}

float3 df_bsdf_measurement_evaluate(
    RES_DATA_PARAM_DECL
    int   bm_idx,
    float2 theta_phi_in,
    float2 theta_phi_out,
    int    part)
{
    if (bm_idx == 0) return float3(0, 0, 0); // invalid bsdf measurement

    const Mdl_mbsdf_info bm = mdl_mbsdf_infos[bm_idx - 1]; // assuming this is in bounds
    if (bm.has_data[part] == 0)
        return float3(0, 0, 0);

    Texture3D eval_data = mdl_textures_3d[NonUniformResourceIndex(bm.eval_data_index[part])];
    const float3 uvw = bsdf_compute_uvw(theta_phi_in, theta_phi_out);
    const float4 sample = eval_data.SampleLevel(
        mdl_sampler_mbsdf, uvw, /*lod=*/ 0.0f, /*offset=*/ int3(0, 0, 0));

    return (bm.num_channels[part] == 3) ? sample.xyz : sample.x;
}

// output: theta, phi, pdf
float3 df_bsdf_measurement_sample(
    RES_DATA_PARAM_DECL
    int   bm_idx,
    float2 theta_phi_out,
    float3 xi,
    int    part)
{
    float3 result = float3(
        -1.0, // negative theta (x value) means absorption
        -1.0,
         0.0);
    if (bm_idx == 0) return result; // invalid bsdf measurement

    const Mdl_mbsdf_info bm = mdl_mbsdf_infos[bm_idx - 1]; // assuming this is in bounds
    if (bm.has_data[part] == 0)
        return result;

    // CDF data
    StructuredBuffer<float> sample_data = mdl_buffers[NonUniformResourceIndex(bm.sample_data_index[part])];
    const uint res_x = bm.angular_resolution_theta[part];
    const uint res_y = bm.angular_resolution_phi[part];

    // compute the theta_in index (flipping input and output, BSDFs are symmetric)
    uint idx_theta_in = uint(theta_phi_out[0] * 2.0 * M_ONE_OVER_PI * float(res_x));
    idx_theta_in = min(idx_theta_in, res_x - 1);

    // sample theta_out
    //-------------------------------------------
    float xi0 = xi[0];
    const uint theta_data_offset = idx_theta_in * res_x;
    const uint idx_theta_out = sample_cdf(sample_data, theta_data_offset, res_x, xi0); // binary search

    float prob_theta = sample_data[theta_data_offset + idx_theta_out];
    if (idx_theta_out > 0)
    {
        const float tmp = sample_data[theta_data_offset + idx_theta_out - 1];
        prob_theta -= tmp;
        xi0 -= tmp;
    }
    xi0 /= prob_theta; // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi[1];
    const uint phi_data_offset = (res_x * res_x)                                 // CDF theta block
                               + (idx_theta_in * res_x + idx_theta_out) * res_y; // selected CDF phi

    // select which half-circle to choose with probability 0.5
    const bool flip = (xi1 > 0.5);
    if (flip) xi1 = 1.0 - xi1;
    xi1 *= 2.0;

    const uint idx_phi_out = sample_cdf(sample_data, phi_data_offset, res_y, xi1); // binary search
    float prob_phi = sample_data[phi_data_offset + idx_phi_out];
    if (idx_phi_out > 0)
    {
        const float tmp = sample_data[phi_data_offset + idx_phi_out - 1];
        prob_phi -= tmp;
        xi1 -= tmp;
    }
    xi1 /= prob_phi; // rescale for re-usage

    // compute theta and phi out
    //-------------------------------------------
    const float s_theta = (0.5 * M_PI) * (1.0 / float(res_x));
    const float s_phi   = (1.0 * M_PI) * (1.0 / float(res_y));

    const float cos_theta_0 = cos(float(idx_theta_out)      * s_theta);
    const float cos_theta_1 = cos(float(idx_theta_out + 1u) * s_theta);

    const float cos_theta = lerp(cos_theta_0, cos_theta_1, xi1);
    result[0] = acos(cos_theta);
    result[1] = (float(idx_phi_out) + xi0) * s_phi;

    if (flip)
        result[1] = 2.0 * M_PI - result[1]; // phi \in [0, 2pi]

    // align phi
    result[1] += (theta_phi_out[1] > 0) ? theta_phi_out[1] : (2.0 * M_PI + theta_phi_out[1]);
    if (result[1] > 2.0 * M_PI) result[1] -= 2.0 * M_PI;
    if (result[1] > 1.0 * M_PI) result[1] = -2.0 * M_PI + result[1]; // to [-pi, pi]

    // compute pdf
    //-------------------------------------------
    result[2] = prob_theta * prob_phi * 0.5 / (s_phi * (cos_theta_0 - cos_theta_1));

    return result;
}

float df_bsdf_measurement_pdf(
    RES_DATA_PARAM_DECL
    int   bm_idx,
    float2 theta_phi_in,
    float2 theta_phi_out,
    int    part)
{
    if (bm_idx == 0) return 0.0; // invalid measured bsdf

    const Mdl_mbsdf_info bm = mdl_mbsdf_infos[bm_idx - 1]; // assuming this is in bounds
    if (bm.has_data[part] == 0)
        return 0.0;

    // CDF data and resolution
    StructuredBuffer<float> sample_data = mdl_buffers[NonUniformResourceIndex(bm.sample_data_index[part])];
    const uint res_x = bm.angular_resolution_theta[part];
    const uint res_y = bm.angular_resolution_phi[part];

    // compute indices in the CDF data
    const float3 uvw = bsdf_compute_uvw(theta_phi_in, theta_phi_out); // phi_delta, theta_out, theta_in
    uint idx_theta_in  = uint(theta_phi_in[0]  * M_ONE_OVER_PI * 2.0 * float(res_x));
    uint idx_theta_out = uint(theta_phi_out[0] * M_ONE_OVER_PI * 2.0 * float(res_x));
    uint idx_phi_out   = uint(uvw.x * float(res_y));
    idx_theta_in = min(idx_theta_in, res_x - 1);
    idx_theta_out = min(idx_theta_out, res_x - 1);
    idx_phi_out = min(idx_phi_out, res_y - 1);

    // get probability to select theta_out
    const uint theta_data_offset = idx_theta_in * res_x;
    float prob_theta = sample_data[theta_data_offset + idx_theta_out];
    if (idx_theta_out > 0)
        prob_theta -= sample_data[theta_data_offset + idx_theta_out - 1];

    // get probability to select phi_out
    const uint phi_data_offset = (res_x * res_x)                                 // CDF theta block
                               + (idx_theta_in * res_x + idx_theta_out) * res_y; // selected CDF phi
    float prob_phi = sample_data[phi_data_offset + idx_phi_out];
    if (idx_phi_out > 0)
        prob_phi -= sample_data[phi_data_offset + idx_phi_out - 1];

    // compute probability to select a position in the sphere patch
    const float s_theta = (0.5 * M_PI) * (1.0 / float(res_x));
    const float s_phi   = (1.0 * M_PI) * (1.0 / float(res_y));

    const float cos_theta_0 = cos(float(idx_theta_out)      * s_theta);
    const float cos_theta_1 = cos(float(idx_theta_out + 1u) * s_theta);

    return prob_theta * prob_phi * 0.5 / (s_phi * (cos_theta_0 - cos_theta_1));
}

// output: max (in case of color) albedo for the selected direction (x) and global (y)
float2 df_bsdf_measurement_albedo(RES_DATA_PARAM_DECL int bm_idx, float2 theta_phi, int part)
{
    const Mdl_mbsdf_info bm = mdl_mbsdf_infos[bm_idx - 1]; // assuming this is in bounds

    // check for the part
    if (bm.has_data[part] == 0)
        return float2(0, 0);

    StructuredBuffer<float> albedo_data = mdl_buffers[NonUniformResourceIndex(bm.albedo_data_index[part])];

    const uint res_x = bm.angular_resolution_theta[part];
    uint idx_theta = uint(theta_phi[0] * 2.0 * M_ONE_OVER_PI * float(res_x));
    idx_theta = min(idx_theta, res_x - 1u);

    return float2(albedo_data[idx_theta], bm.max_albedo[part]);
}

float4 df_bsdf_measurement_albedos(RES_DATA_PARAM_DECL int bm_idx, float2 theta_phi)
{
    if (bm_idx == 0) return float4(0, 0, 0, 0); // invalid bsdf measurement

    const float2 albedo_refl = df_bsdf_measurement_albedo(
        bm_idx, theta_phi, MBSDF_DATA_REFLECTION);
    const float2 albedo_trans = df_bsdf_measurement_albedo(
        bm_idx, theta_phi, MBSDF_DATA_TRANSMISSION);

    return float4(albedo_refl[0], albedo_refl[1], albedo_trans[0], albedo_trans[1]);
}


// ------------------------------------------------------------------------------------------------
// Scene Data API
// ------------------------------------------------------------------------------------------------

bool scene_data_isvalid_internal(
    Shading_state_material state,   // MDL state that also contains a custom renderer state
    int scene_data_id,              // the scene_data_id (from target code or manually added)
    bool uniform_lookup)
{
    // invalid id
    if (scene_data_id == 0)
        return false;

    // get scene data buffer layout and access infos
    SceneDataInfo info = scene_data_infos[
        state.renderer_state.scene_data_info_offset + scene_data_id];

    SceneDataKind kind = info.GetKind();
    switch (kind)
    {
        case SCENE_DATA_KIND_VERTEX:
        case SCENE_DATA_KIND_INSTANCE:
            return true;

        default:
            return false;
    }
}

bool scene_data_isvalid(
    inout Shading_state_material state, // MDL state that also contains a custom renderer state
    int scene_data_id)                  // the scene_data_id (from target code or manually added)
{
    return scene_data_isvalid_internal(state, scene_data_id, false);
}

// try to avoid a lot of redundant code, always return float4 but (statically) switch on components
float4 scene_data_lookup_floatX(
    Shading_state_material state,   // MDL state that also contains a custom renderer state
    int scene_data_id,              // the scene_data_id (from target code or manually added)
    float4 default_value,           // default value in case the requested data is not valid
    bool uniform_lookup,            // true if a uniform lookup is requested
    int number_of_components)       // 1, 2, 3, or 4
{
    // invalid id
    if (scene_data_id == 0)
        return default_value;

    // get scene data buffer layout and access infos
    SceneDataInfo info = scene_data_infos[
        state.renderer_state.scene_data_info_offset + scene_data_id];

    if (uniform_lookup && !info.GetUniform())
        return default_value;

    SceneDataKind kind = info.GetKind();
    SceneDataInterpolationMode mode = info.GetInterpolationMode();

    // access data depending of the scope (per scene, object, or vertex)
    switch (kind)
    {
    case SCENE_DATA_KIND_VERTEX:
    {
        // address of the per vertex data (for each index)
        uint3 addresses =
            state.renderer_state.scene_data_geometry_byte_offset + // address of the geometry
            state.renderer_state.hit_vertex_indices * info.GetByteStride() + // element offset
            info.GetByteOffset(); // offset within the vertex or to first element

        // raw data read from the buffer
        uint4 value_a_raw = uint4(0, 0, 0, 0);
        uint4 value_b_raw = uint4(0, 0, 0, 0);
        uint4 value_c_raw = uint4(0, 0, 0, 0);
        if (number_of_components == 1)
        {
            value_a_raw.x = vertices.Load(addresses.x);
            value_b_raw.x = vertices.Load(addresses.y);
            value_c_raw.x = vertices.Load(addresses.z);
        }
        else if (number_of_components == 2)
        {
            value_a_raw.xy = vertices.Load2(addresses.x);
            value_b_raw.xy = vertices.Load2(addresses.y);
            value_c_raw.xy = vertices.Load2(addresses.z);
        }
        else if (number_of_components == 3)
        {
            value_a_raw.xyz = vertices.Load3(addresses.x);
            value_b_raw.xyz = vertices.Load3(addresses.y);
            value_c_raw.xyz = vertices.Load3(addresses.z);
        }
        else if (number_of_components == 4)
        {
            value_a_raw = vertices.Load4(addresses.x);
            value_b_raw = vertices.Load4(addresses.y);
            value_c_raw = vertices.Load4(addresses.z);
        }

        // convert to float, int or color data
        float4 value_a = float4(0, 0, 0, 0);
        float4 value_b = float4(0, 0, 0, 0);
        float4 value_c = float4(0, 0, 0, 0);
        if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_FLOAT     // reinterpret as float
            || info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_COLOR) // (handled as float3, no spectral support)
        {
            value_a = asfloat(value_a_raw);
            value_b = asfloat(value_b_raw);
            value_c = asfloat(value_c_raw);
        }
        else if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_INT)
        {
            // reinterpret as signed int and convert from integer to float
            value_a = float4(asint(value_a_raw));
            value_b = float4(asint(value_b_raw));
            value_c = float4(asint(value_c_raw));
        }

        // interpolate across the triangle
        const float3 barycentric = state.renderer_state.barycentric;
        if (mode == SCENE_DATA_INTERPOLATION_MODE_LINEAR)
            return value_a * barycentric.x +
                value_b * barycentric.y +
                value_c * barycentric.z;
        else if (mode == SCENE_DATA_INTERPOLATION_MODE_NEAREST)
            if (barycentric.x > barycentric.y)
                return barycentric.x > barycentric.z ? value_a : value_c;
            else
                return barycentric.y > barycentric.z ? value_b : value_c;
        else // SCENE_DATA_INTERPOLATION_MODE_NONE
             // or unsupported interpolation mode
            return default_value;
    }

    case SCENE_DATA_KIND_INSTANCE:
    {
        // raw data read from the buffer
        uint address = info.GetByteOffset();
        uint4 value_raw = uint4(0, 0, 0, 0);
        if (number_of_components == 1)      value_raw.x = scene_data.Load(address);
        else if (number_of_components == 2) value_raw.xy = scene_data.Load2(address);
        else if (number_of_components == 3) value_raw.xyz = scene_data.Load3(address);
        else if (number_of_components == 4) value_raw = scene_data.Load4(address);

        // convert to float, int or color data
        // do not interpolate as all currently available modes would result in the same value
        if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_FLOAT     // reinterpret as float
            || info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_COLOR) // (handled as float3, no spectral support)
        {
            return asfloat(value_raw);
        }
        else if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_INT)
        {
            // reinterpret as signed int and convert to float
            return float4(asint(value_raw));
        }
    }

    case SCENE_DATA_KIND_NONE:
    default:
        return default_value;
    }
}

float4 scene_data_lookup_float4(
    inout Shading_state_material state,
    int scene_data_id,
    float4 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value, uniform_lookup, 4);
}

float3 scene_data_lookup_float3(
    inout Shading_state_material state,
    int scene_data_id,
    float3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xyzx, uniform_lookup, 3).xyz;
}

float3 scene_data_lookup_color(
    inout Shading_state_material state,
    int scene_data_id,
    float3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xyzx, uniform_lookup, 3).xyz;
}

float2 scene_data_lookup_float2(
    inout Shading_state_material state,
    int scene_data_id,
    float2 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xyxx, uniform_lookup, 2).xy;
}

float scene_data_lookup_float(
    inout Shading_state_material state,
    int scene_data_id,
    float default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xxxx, uniform_lookup, 1).x;
}

float4x4 scene_data_lookup_float4x4(
    inout Shading_state_material state,
    int scene_data_id,
    float4x4 default_value,
    bool uniform_lookup)
{
    // dummy implementation
    return default_value;
}

int4 scene_data_lookup_intX(
    Shading_state_material state,
    int scene_data_id,
    int4 default_value,
    bool uniform_lookup,
    int number_of_components)
{
    // invalid id
    if (scene_data_id == 0)
        return default_value;

    // get scene data buffer layout and access infos
    SceneDataInfo info = scene_data_infos[
        state.renderer_state.scene_data_info_offset + scene_data_id];

    if (uniform_lookup && !info.GetUniform())
        return default_value;

    SceneDataKind kind = info.GetKind();
    SceneDataInterpolationMode mode = info.GetInterpolationMode();

    // access data depending of the scope (per scene, object, or vertex)
    switch (kind)
    {
    case SCENE_DATA_KIND_VERTEX:
    {
        // address of the per vertex data (for each index)
        uint3 addresses =
            state.renderer_state.scene_data_geometry_byte_offset + // address of the geometry
            state.renderer_state.hit_vertex_indices * info.GetByteStride() + // element offset
            info.GetByteOffset(); // offset within the vertex or to first element

        // raw data read from the buffer
        uint4 value_a_raw = uint4(0, 0, 0, 0);
        uint4 value_b_raw = uint4(0, 0, 0, 0);
        uint4 value_c_raw = uint4(0, 0, 0, 0);
        if (number_of_components == 1)
        {
            value_a_raw.x = vertices.Load(addresses.x);
            value_b_raw.x = vertices.Load(addresses.y);
            value_c_raw.x = vertices.Load(addresses.z);
        }
        else if (number_of_components == 2)
        {
            value_a_raw.xy = vertices.Load2(addresses.x);
            value_b_raw.xy = vertices.Load2(addresses.y);
            value_c_raw.xy = vertices.Load2(addresses.z);
        }
        else if (number_of_components == 3)
        {
            value_a_raw.xyz = vertices.Load3(addresses.x);
            value_b_raw.xyz = vertices.Load3(addresses.y);
            value_c_raw.xyz = vertices.Load3(addresses.z);
        }
        else if (number_of_components == 4)
        {
            value_a_raw = vertices.Load4(addresses.x);
            value_b_raw = vertices.Load4(addresses.y);
            value_c_raw = vertices.Load4(addresses.z);
        }

        // convert to float, int or color data
        int4 value_a = int4(0, 0, 0, 0);
        int4 value_b = int4(0, 0, 0, 0);
        int4 value_c = int4(0, 0, 0, 0);
        if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_FLOAT     // reinterpret as float and convert to int
            || info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_COLOR) // (handled as float3, no spectral support)
        {
            value_a = int4(asfloat(value_a_raw));
            value_b = int4(asfloat(value_b_raw));
            value_c = int4(asfloat(value_c_raw));
        }
        else if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_INT)
        {
            value_a = asint(value_a_raw);
            value_b = asint(value_b_raw);
            value_c = asint(value_c_raw);
        }

        // interpolate across the triangle
        const float3 barycentric = state.renderer_state.barycentric;
        if (mode == SCENE_DATA_INTERPOLATION_MODE_LINEAR)
            return int4(float4(value_a)*barycentric.x +
                float4(value_b)*barycentric.y +
                float4(value_c)*barycentric.z);
        else if (mode == SCENE_DATA_INTERPOLATION_MODE_NEAREST)
            if (barycentric.x > barycentric.y)
                return barycentric.x > barycentric.z ? value_a : value_c;
            else
                return barycentric.y > barycentric.z ? value_b : value_c;
        else // SCENE_DATA_INTERPOLATION_MODE_NONE:
             // or unsupported interpolation mode
            return default_value;
    }

    case SCENE_DATA_KIND_INSTANCE:
    {
        // raw data read from the buffer
        uint address = info.GetByteOffset();
        uint4 value_raw = uint4(0, 0, 0, 0);
        if (number_of_components == 1)      value_raw.x = scene_data.Load(address);
        else if (number_of_components == 2) value_raw.xy = scene_data.Load2(address);
        else if (number_of_components == 3) value_raw.xyz = scene_data.Load3(address);
        else if (number_of_components == 4) value_raw = scene_data.Load4(address);

        // convert to float, int or color data
        // do not interpolate as all currently available modes would result in the same value
        if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_FLOAT     // reinterpret as float and convert to int
            || info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_COLOR) // (handled as float3, no spectral support)
        {
            return int4(asfloat(value_raw));
        }
        else if (info.GetElementType() == SCENE_DATA_ELEMENT_TYPE_INT)
        {
            // reinterpret as signed int
            return asint(value_raw);
        }
    }

    case SCENE_DATA_KIND_NONE:
    default:
        return default_value;
    }
}

int4 scene_data_lookup_int4(
    inout Shading_state_material state,
    int scene_data_id,
    int4 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value, uniform_lookup, 4);
}

int3 scene_data_lookup_int3(
    inout Shading_state_material state,
    int scene_data_id,
    int3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value.xyzx, uniform_lookup, 3).xyz;
}

int2 scene_data_lookup_int2(
    inout Shading_state_material state,
    int scene_data_id,
    int2 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value.xyxx, uniform_lookup, 2).xy;
}

int scene_data_lookup_int(
    inout Shading_state_material state,
    int scene_data_id,
    int default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value.xxxx, uniform_lookup, 1).x;
}

// currently no scene data with derivatives is supported
Derived_float4 scene_data_lookup_deriv_float4(
    inout Shading_state_material state,
    int scene_data_id,
    Derived_float4 default_value,
    bool uniform_lookup)
{
    if (!scene_data_isvalid_internal(state, scene_data_id, uniform_lookup))
        return default_value;

    Derived_float4 res;
    res.val = scene_data_lookup_float4(
        state, scene_data_id, default_value.val, uniform_lookup);
    res.dx = float4(0, 0, 0, 0);
    res.dy = float4(0, 0, 0, 0);
    return res;
}

Derived_float3 scene_data_lookup_deriv_float3(
    inout Shading_state_material state,
    int scene_data_id,
    Derived_float3 default_value,
    bool uniform_lookup)
{
    if (!scene_data_isvalid_internal(state, scene_data_id, uniform_lookup))
        return default_value;

    Derived_float3 res;
    res.val = scene_data_lookup_float3(
        state, scene_data_id, default_value.val, uniform_lookup);
    res.dx = float3(0, 0, 0);
    res.dy = float3(0, 0, 0);
    return res;
}

Derived_float3 scene_data_lookup_deriv_color(
    inout Shading_state_material state,
    int scene_data_id,
    Derived_float3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_deriv_float3(
        state, scene_data_id, default_value, uniform_lookup);
}

Derived_float2 scene_data_lookup_deriv_float2(
    inout Shading_state_material state,
    int scene_data_id,
    Derived_float2 default_value,
    bool uniform_lookup)
{
    if (!scene_data_isvalid_internal(state, scene_data_id, uniform_lookup))
        return default_value;

    Derived_float2 res;
    res.val = scene_data_lookup_float2(
        state, scene_data_id, default_value.val, uniform_lookup);
    res.dx = float2(0, 0);
    res.dy = float2(0, 0);
    return res;
}

Derived_float scene_data_lookup_deriv_float(
    inout Shading_state_material state,
    int scene_data_id,
    Derived_float default_value,
    bool uniform_lookup)
{
    if (!scene_data_isvalid_internal(state, scene_data_id, uniform_lookup))
        return default_value;

    Derived_float res;
    res.val = scene_data_lookup_float(
        state, scene_data_id, default_value.val, uniform_lookup);
    res.dx = 0;
    res.dy = 0;
    return res;
}

#endif // MDL_RENDERER_RUNTIME_HLSLI
