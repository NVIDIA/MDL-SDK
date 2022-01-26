/******************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
// - MDL_MATERIAL_RESOURCE_INFO_SLOT

// - MDL_MATERIAL_TEXTURE_2D_REGISTER_SPACE
// - MDL_MATERIAL_TEXTURE_3D_REGISTER_SPACE
// - MDL_MATERIAL_TEXTURE_SLOT_BEGIN
//
// - MDL_TEXTURE_SAMPLER_SLOT


/// Information passed to GPU for mapping id requested in the runtime functions to buffer
/// views of the corresponding type.
struct Mdl_resource_info
{
    // index into the tex2d, tex3d, ... buffers, depending on the type requested
    uint gpu_resource_array_start;

    // number resources (e.g. uv-tiles) that belong to this resource
    uint gpu_resource_array_size;

    // frame number of the first texture/uv-tile
    int gpu_resource_frame_first;

    // coordinate of the left bottom most uv-tile (also bottom left corner)
    int2 gpu_resource_uvtile_min;

    // in case of uv-tiled textures,  required to calculate a linear index (u + v * width
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

// per target data
ByteAddressBuffer mdl_ro_data_segment : register(MDL_TARGET_RO_DATA_SEGMENT_SLOT, MDL_TARGET_REGISTER_SPACE);

// per material data
// - argument block contains dynamic parameter data exposed in class compilation mode
ByteAddressBuffer mdl_argument_block : register(MDL_MATERIAL_ARGUMENT_BLOCK_SLOT, MDL_MATERIAL_REGISTER_SPACE);
// - resource infos map resource IDs, generated by the SDK, to actual buffer views
StructuredBuffer<Mdl_resource_info> mdl_resource_infos : register(MDL_MATERIAL_RESOURCE_INFO_SLOT, MDL_MATERIAL_REGISTER_SPACE);
// - texture views, unbound and overlapping for 2D and 3D resources
Texture2D mdl_textures_2d[] : register(MDL_MATERIAL_TEXTURE_SLOT_BEGIN, MDL_MATERIAL_TEXTURE_2D_REGISTER_SPACE);
Texture3D mdl_textures_3d[] : register(MDL_MATERIAL_TEXTURE_SLOT_BEGIN, MDL_MATERIAL_TEXTURE_3D_REGISTER_SPACE);

// global samplers
SamplerState mdl_sampler_tex : register(MDL_TEXTURE_SAMPLER_SLOT);

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

float mdl_read_argblock_as_float(uint offs)
{
    return asfloat(mdl_argument_block.Load(offs));
}

double mdl_read_argblock_as_double(uint offs)
{
    return asdouble(mdl_argument_block.Load(offs), mdl_argument_block.Load(offs + 4));
}

int mdl_read_argblock_as_int(uint offs)
{
    return asint(mdl_argument_block.Load(offs));
}

uint mdl_read_argblock_as_uint(uint offs)
{
    return mdl_argument_block.Load(offs);
}

bool mdl_read_argblock_as_bool(uint offs)
{
    uint val = mdl_argument_block.Load(offs & ~3);
    return (val & (0xffU << (8 * (offs & 3)))) != 0;
}


float mdl_read_rodata_as_float(uint offs)
{
    return asfloat(mdl_ro_data_segment.Load(offs));
}

double mdl_read_rodata_as_double(uint offs)
{
    return asdouble(mdl_ro_data_segment.Load(offs), mdl_ro_data_segment.Load(offs + 4));
}

int mdl_read_rodata_as_int(uint offs)
{
    return asint(mdl_ro_data_segment.Load(offs));
}

uint mdl_read_rodata_as_uint(uint offs)
{
    return mdl_ro_data_segment.Load(offs);
}

bool mdl_read_rodata_as_bool(uint offs)
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
bool tex_texture_isvalid(RES_DATA_PARAM_DECL uint tex)
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

uint2 tex_res_2d(RES_DATA_PARAM_DECL uint tex, int2 uv_tile, float frame)
{
    if (tex == 0) return uint2(0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

    int array_index = info.compute_uvtile_id(frame, uv_tile);
    if (array_index < 0) return uint2(0, 0); // out of bounds or no uv-tile

    uint2 res;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y);
    return res;
}

// corresponds to ::tex::width(uniform texture_2d tex, int2 uv_tile, float frame)
uint tex_width_2d(RES_DATA_PARAM_DECL uint tex, int2 uv_tile, float frame)
{
    return tex_res_2d(RES_DATA_PARAM tex, uv_tile, frame).x;
}

// corresponds to ::tex::height(uniform texture_2d tex, int2 uv_tile)
uint tex_height_2d(RES_DATA_PARAM_DECL uint tex, int2 uv_tile, float frame)
{
    return tex_res_2d(RES_DATA_PARAM tex, uv_tile, frame).y;
}

// corresponds to ::tex::first__frame(uniform texture_2d)
int tex_first_frame_2d(RES_DATA_PARAM_DECL uint tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds
    return info.gpu_resource_frame_first;
}

// corresponds to ::tex::last_frame(uniform texture_2d)
int tex_last_frame_2d(RES_DATA_PARAM_DECL uint tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds
    return info.get_last_frame();
}

// corresponds to ::tex::lookup_float4(uniform texture_2d tex, float2 coord, ...)
float4 tex_lookup_float4_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

    // handle uv-tiles and/or get texture array index
    int array_index = info.compute_uvtile_and_update_uv(frame, coord);
    if (array_index < 0) return float4(0, 0, 0, 0); // out of bounds or no uv-tile

    if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0.0 || coord.x >= 1.0))
        return float4(0, 0, 0, 0);
    if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0.0 || coord.y >= 1.0))
        return float4(0, 0, 0, 0);

    uint width, height;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(width, height);
    coord.x = apply_wrap_and_crop(coord.x, wrap_u, crop_u, width);
    coord.y = apply_wrap_and_crop(coord.y, wrap_v, crop_v, height);

    coord = apply_smootherstep_filter(coord, uint2(width, height));

    // Note, since we don't have ddx and ddy in the compute pipeline, TextureObject::Sample() is not
    // available, we use SampleLevel instead and go for the most detailed level. Therefore, we don't
    // need mipmaps. Manual mip level computation is possible though.
    return mdl_textures_2d[NonUniformResourceIndex(array_index)].SampleLevel(
        mdl_sampler_tex, coord, /*lod=*/ 0.0f, /*offset=*/ int2(0, 0));
}

float3 tex_lookup_float3_2d(RES_DATA_PARAM_DECL uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float3 tex_lookup_color_2d(RES_DATA_PARAM_DECL uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float2 tex_lookup_float2_2d(RES_DATA_PARAM_DECL uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_float_2d(RES_DATA_PARAM_DECL uint tex, float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}

// corresponds to ::tex::lookup_float4(uniform texture_2d tex, float2 coord, ...) when derivatives are enabled
float4 tex_lookup_deriv_float4_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    Derived_float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

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

    coord.val = apply_smootherstep_filter(coord.val, uint2(res.x, res.y));

    // Note, since we don't have ddx and ddy in the compute pipeline, TextureObject::Sample() is not
    // available, we use SampleLevel instead and go for the most detailed level. Therefore, we don't
    // need mipmaps. Manual mip level computation is possible though.
    return mdl_textures_2d[NonUniformResourceIndex(array_index)].SampleGrad(
        mdl_sampler_tex, coord.val, coord.dx, coord.dy, /*offset=*/ int2(0, 0));
}

float3 tex_lookup_deriv_float3_2d(RES_DATA_PARAM_DECL uint tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float3 tex_lookup_deriv_color_2d(RES_DATA_PARAM_DECL uint tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xyz;
}

float2 tex_lookup_deriv_float2_2d(RES_DATA_PARAM_DECL uint tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).xy;
}

float tex_lookup_deriv_float_2d(RES_DATA_PARAM_DECL uint tex, Derived_float2 coord, int wrap_u, int wrap_v, float2 crop_u, float2 crop_v, float frame)
{
    return tex_lookup_deriv_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}


// corresponds to ::tex::texel_float4(uniform texture_2d tex, float2 coord, int2 uv_tile)
float4 tex_texel_float4_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    int2 coord,
    int2 uv_tile,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

    // handle uv-tiles and/or get texture array index
    int array_index = info.compute_uvtile_and_update_uv(frame, uv_tile);
    if (array_index < 0) return float4(0, 0, 0, 0); // out of bounds or no uv-tile

    uint2 res;
    mdl_textures_2d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y);
    if (0 > coord.x || res.x <= coord.x || 0 > coord.y || res.y <= coord.y)
        return float4(0, 0, 0, 0); // out of bounds

    return mdl_textures_2d[NonUniformResourceIndex(array_index)].Load(int3(coord, /*mipmaplevel=*/ 0));
}

float3 tex_texel_float3_2d(RES_DATA_PARAM_DECL uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile, frame).xyz;
}

float3 tex_texel_color_2d(RES_DATA_PARAM_DECL uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float3_2d(RES_DATA_PARAM tex, coord, uv_tile, frame);
}

float2 tex_texel_float2_2d(RES_DATA_PARAM_DECL uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile, frame).xy;
}

float tex_texel_float_2d(RES_DATA_PARAM_DECL uint tex, int2 coord, int2 uv_tile, float frame)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile, frame).x;
}


// ------------------------------------------------------------------------------------------------
// Texturing functions, 3D
// ------------------------------------------------------------------------------------------------

uint3 tex_res_3d(RES_DATA_PARAM_DECL uint tex, float frame)
{
    if (tex == 0) return uint3(0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

    // no uv-tiles for 3D textures (shortcut the index calculation)
    int array_index = info.gpu_resource_array_start;

    uint3 res;
    mdl_textures_3d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y, res.z);
    return res;
}

// corresponds to ::tex::first__frame(uniform texture_3d)
int tex_first_frame_3d(RES_DATA_PARAM_DECL uint tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds
    return info.gpu_resource_frame_first;
}

// corresponds to ::tex::last_frame(uniform texture_3d)
int tex_last_frame_3d(RES_DATA_PARAM_DECL uint tex)
{
    if (tex == 0) return 0; // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds
    return info.get_last_frame();
}

// corresponds to ::tex::width(uniform texture_3d tex, int2 uv_tile)
uint tex_width_3d(RES_DATA_PARAM_DECL uint tex, float frame) { return tex_res_3d(RES_DATA_PARAM tex, frame).x; }

// corresponds to ::tex::height(uniform texture_3d tex, int2 uv_tile)
uint tex_height_3d(RES_DATA_PARAM_DECL uint tex, float frame) { return tex_res_3d(RES_DATA_PARAM tex, frame).y; }

// corresponds to ::tex::depth(uniform texture_3d tex, int2 uv_tile)
uint tex_depth_3d(RES_DATA_PARAM_DECL uint tex, float frame) { return tex_res_3d(RES_DATA_PARAM tex, frame).z; }

// corresponds to ::tex::lookup_float4(uniform texture_3d tex, float2 coord, ...)
float4 tex_lookup_float4_3d(
    RES_DATA_PARAM_DECL
    uint tex,
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
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

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

float3 tex_lookup_float3_3d(RES_DATA_PARAM_DECL uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

float3 tex_lookup_color_3d(RES_DATA_PARAM_DECL uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xyz;
}

float2 tex_lookup_float2_3d(RES_DATA_PARAM_DECL uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).xy;
}

float tex_lookup_float_3d(RES_DATA_PARAM_DECL uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w, float2 crop_u, float2 crop_v, float2 crop_w, float frame)
{
    return tex_lookup_float4_3d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame).x;
}

// corresponds to ::tex::texel_float4(uniform texture_3d tex, float3 coord)
float4 tex_texel_float4_3d(
    RES_DATA_PARAM_DECL
    uint tex,
    int3 coord,
    float frame)
{
    if (tex == 0) return float4(0, 0, 0, 0); // invalid texture

    // fetch the infos about this resource
    Mdl_resource_info info = mdl_resource_infos[tex - 1]; // assuming this is in bounds

    // no uv-tiles for 3D textures (shortcut the index calculation)
    int array_index = info.gpu_resource_array_start;

    uint3 res;
    mdl_textures_3d[NonUniformResourceIndex(array_index)].GetDimensions(res.x, res.y, res.z);
    if (0 > coord.x || res.x <= coord.x || 0 > coord.y || res.y <= coord.y || 0 > coord.z || res.z <= coord.z)
        return float4(0, 0, 0, 0); // out of bounds

    return mdl_textures_3d[NonUniformResourceIndex(array_index)].Load(int4(coord, /*mipmaplevel=*/ 0));
}

float3 tex_texel_float3_3d(RES_DATA_PARAM_DECL uint tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord, frame).xyz;
}

float3 tex_texel_color_3d(RES_DATA_PARAM_DECL uint tex, int3 coord, float frame)
{
    return tex_texel_float3_3d(RES_DATA_PARAM tex, coord, frame);
}

float2 tex_texel_float2_3d(RES_DATA_PARAM_DECL uint tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord, frame).xy;
}

float tex_texel_float_3d(RES_DATA_PARAM_DECL uint tex, int3 coord, float frame)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord, frame).x;
}


// ------------------------------------------------------------------------------------------------
// Texturing functions, Cube (not supported by this example)
// ------------------------------------------------------------------------------------------------

uint tex_width_cube(RES_DATA_PARAM_DECL uint tex) { return 0; }
uint tex_height_cube(RES_DATA_PARAM_DECL uint tex) { return 0; }

float4 tex_lookup_float4_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return float4(0, 0, 0, 0);
}

float3 tex_lookup_float3_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float3 tex_lookup_color_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float2 tex_lookup_float2_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xy;
}

float tex_lookup_float_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).x;
}

float4 tex_texel_float4_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return float4(0, 0, 0, 0);
}

float3 tex_texel_float3_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float3 tex_texel_color_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xyz;
}

float2 tex_texel_float2_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xy;
}

float tex_texel_float_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).x;
}

// ------------------------------------------------------------------------------------------------
// Texturing functions, PTEX (not supported by this example)
// ------------------------------------------------------------------------------------------------


float4 tex_lookup_float4_ptex(RES_DATA_PARAM_DECL uint tex, int channel)
{
    return float4(0, 0, 0, 0);
}

float3 tex_lookup_float3_ptex(RES_DATA_PARAM_DECL uint tex, int channel)
{
    return tex_lookup_float4_ptex(RES_DATA_PARAM tex, channel).xyz;
}

float3 tex_lookup_color_ptex(RES_DATA_PARAM_DECL uint tex, int channel)
{
    return tex_lookup_float3_ptex(RES_DATA_PARAM tex, channel);
}

float2 tex_lookup_float2_ptex(RES_DATA_PARAM_DECL uint tex, int channel)
{
    return tex_lookup_float4_ptex(RES_DATA_PARAM tex, channel).xy;
}

float tex_lookup_float_ptex(RES_DATA_PARAM_DECL uint tex, int channel)
{
    return tex_lookup_float4_ptex(RES_DATA_PARAM tex, channel).x;
}

// ------------------------------------------------------------------------------------------------
// Light Profiles (not supported by this example)
// ------------------------------------------------------------------------------------------------

bool df_light_profile_isvalid(RES_DATA_PARAM_DECL uint lp_idx)
{
    return false;
}

float df_light_profile_power(RES_DATA_PARAM_DECL uint lp_idx)
{
    return 0;
}

float df_light_profile_maximum(RES_DATA_PARAM_DECL uint lp_idx)
{
    return 0;
}

float df_light_profile_evaluate(
    RES_DATA_PARAM_DECL
    uint   lp_idx,
    float2 theta_phi)
{
    return 0;
}

float3 df_light_profile_sample(
    RES_DATA_PARAM_DECL
    uint   lp_idx,
    float3 xi)
{
    return 0;
}

float df_light_profile_pdf(
    RES_DATA_PARAM_DECL
    uint   lp_idx,
    float2 theta_phi)
{
    return 0;
}

// ------------------------------------------------------------------------------------------------
// Measured BSDFs (not supported by this example)
// ------------------------------------------------------------------------------------------------

// The example does not support BSDF measurements
int3 df_bsdf_measurement_resolution(RES_DATA_PARAM_DECL uint bm_idx, int part)
{
    return int3(0, 0, 0);
}

float3 df_bsdf_measurement_evaluate(
    RES_DATA_PARAM_DECL
    uint   bm_idx,
    float2 theta_phi_in,
    float2 theta_phi_out,
    int    part)
{
    return float3(0, 0, 0);
}

float3 df_bsdf_measurement_sample(
    RES_DATA_PARAM_DECL
    uint   bm_idx,
    float2 theta_phi_out,
    float3 xi,
    int    part)
{
    return float3(0, 0, 0);
}

float df_bsdf_measurement_pdf(
    RES_DATA_PARAM_DECL
    uint   bm_idx,
    float2 theta_phi_in,
    float2 theta_phi_out,
    int    part)
{
    return 0;
}

float4 df_bsdf_measurement_albedos(RES_DATA_PARAM_DECL uint bm_idx, float2 theta_phi)
{
    return float4(0, 0, 0, 0);
}

bool df_bsdf_measurement_isvalid(RES_DATA_PARAM_DECL uint bm_idx)
{
    return false;
}


// ------------------------------------------------------------------------------------------------
// Scene Data API
// ------------------------------------------------------------------------------------------------

bool scene_data_isvalid_internal(
    Shading_state_material state,   // MDL state that also contains a custom renderer state
    uint scene_data_id,             // the scene_data_id (from target code or manually added)
    bool uniform_lookup)
{
    // invalid id
    if (scene_data_id == 0)
        return false;

    // get scene data buffer layout and access infos
    SceneDataInfo info = state.renderer_state.scene_data_infos[
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
    Shading_state_material state,   // MDL state that also contains a custom renderer state
    uint scene_data_id)             // the scene_data_id (from target code or manually added)
{
    return scene_data_isvalid_internal(state, scene_data_id, false);
}

// try to avoid a lot of redundant code, always return float4 but (statically) switch on components
float4 scene_data_lookup_floatX(
    Shading_state_material state,   // MDL state that also contains a custom renderer state
    uint scene_data_id,             // the scene_data_id (from target code or manually added)
    float4 default_value,           // default value in case the requested data is not valid
    bool uniform_lookup,            // true if a uniform lookup is requested
    int number_of_components)       // 1, 2, 3, or 4
{
    // invalid id
    if (scene_data_id == 0)
        return default_value;

    // get scene data buffer layout and access infos
    SceneDataInfo info = state.renderer_state.scene_data_infos[
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
        switch (number_of_components)
        {
            case 1:
                value_a_raw.x = state.renderer_state.scene_data_vertex.Load(addresses.x);
                value_b_raw.x = state.renderer_state.scene_data_vertex.Load(addresses.y);
                value_c_raw.x = state.renderer_state.scene_data_vertex.Load(addresses.z);
                break;
            case 2:
                value_a_raw.xy = state.renderer_state.scene_data_vertex.Load2(addresses.x);
                value_b_raw.xy = state.renderer_state.scene_data_vertex.Load2(addresses.y);
                value_c_raw.xy = state.renderer_state.scene_data_vertex.Load2(addresses.z);
                break;
            case 3:
                value_a_raw.xyz = state.renderer_state.scene_data_vertex.Load3(addresses.x);
                value_b_raw.xyz = state.renderer_state.scene_data_vertex.Load3(addresses.y);
                value_c_raw.xyz = state.renderer_state.scene_data_vertex.Load3(addresses.z);
                break;
            case 4:
                value_a_raw = state.renderer_state.scene_data_vertex.Load4(addresses.x);
                value_b_raw = state.renderer_state.scene_data_vertex.Load4(addresses.y);
                value_c_raw = state.renderer_state.scene_data_vertex.Load4(addresses.z);
                break;
        }

        // convert to float, int or color data
        float4 value_a = float4(0, 0, 0, 0);
        float4 value_b = float4(0, 0, 0, 0);
        float4 value_c = float4(0, 0, 0, 0);
        switch (info.GetElementType())
        {
            case SCENE_DATA_ELEMENT_TYPE_FLOAT: // reinterpret as float
            case SCENE_DATA_ELEMENT_TYPE_COLOR: // (handled as float3, no spectral support)
                value_a = asfloat(value_a_raw);
                value_b = asfloat(value_b_raw);
                value_c = asfloat(value_c_raw);
                break;

            case SCENE_DATA_ELEMENT_TYPE_INT:
                // reinterpret as signed int and convert from integer to float
                value_a = float4(asint(value_a_raw));
                value_b = float4(asint(value_b_raw));
                value_c = float4(asint(value_c_raw));
                break;
        }

        // interpolate across the triangle
        const float3 barycentric = state.renderer_state.barycentric;
        switch (mode)
        {
            case SCENE_DATA_INTERPOLATION_MODE_LINEAR:
                return value_a * barycentric.x +
                    value_b * barycentric.y +
                    value_c * barycentric.z;

            case SCENE_DATA_INTERPOLATION_MODE_NEAREST:
                if (barycentric.x > barycentric.y)
                    return barycentric.x > barycentric.z ? value_a : value_c;
                else
                    return barycentric.y > barycentric.z ? value_b : value_c;

            case SCENE_DATA_INTERPOLATION_MODE_NONE:
            default: // unsupported interpolation mode
                return default_value;
        }
    }

    case SCENE_DATA_KIND_INSTANCE:
    {
        // raw data read from the buffer
        uint address = info.GetByteOffset();
        uint4 value_raw = uint4(0, 0, 0, 0);
        switch (number_of_components)
        {
            case 1: value_raw.x = state.renderer_state.scene_data_instance.Load(address); break;
            case 2: value_raw.xy = state.renderer_state.scene_data_instance.Load2(address); break;
            case 3: value_raw.xyz = state.renderer_state.scene_data_instance.Load3(address); break;
            case 4: value_raw = state.renderer_state.scene_data_instance.Load4(address); break;
        }

        // convert to float, int or color data
        // do not interpolate as all currently available modes would result in the same value
        switch (info.GetElementType())
        {
            case SCENE_DATA_ELEMENT_TYPE_FLOAT: // reinterpret as float
            case SCENE_DATA_ELEMENT_TYPE_COLOR: // (handled as float3, no spectral support)
                return asfloat(value_raw);

            case SCENE_DATA_ELEMENT_TYPE_INT:   // reinterpret as signed int and convert to float
                return float4(asint(value_raw));
        }
    }

    case SCENE_DATA_KIND_NONE:
    default:
        return default_value;
    }
}

float4 scene_data_lookup_float4(
    Shading_state_material state,
    uint scene_data_id,
    float4 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value, uniform_lookup, 4);
}

float3 scene_data_lookup_float3(
    Shading_state_material state,
    uint scene_data_id,
    float3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xyzx, uniform_lookup, 3).xyz;
}

float3 scene_data_lookup_color(
    Shading_state_material state,
    uint scene_data_id,
    float3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xyzx, uniform_lookup, 3).xyz;
}

float2 scene_data_lookup_float2(
    Shading_state_material state,
    uint scene_data_id,
    float2 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xyxx, uniform_lookup, 2).xy;
}

float scene_data_lookup_float(
    Shading_state_material state,
    uint scene_data_id,
    float default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_floatX(state, scene_data_id, default_value.xxxx, uniform_lookup, 1).x;
}


int4 scene_data_lookup_intX(
    Shading_state_material state,
    uint scene_data_id,
    int4 default_value,
    bool uniform_lookup,
    int number_of_components)
{
    // invalid id
    if (scene_data_id == 0)
        return default_value;

    // get scene data buffer layout and access infos
    SceneDataInfo info = state.renderer_state.scene_data_infos[
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
        switch (number_of_components)
        {
            case 1:
                value_a_raw.x = state.renderer_state.scene_data_vertex.Load(addresses.x);
                value_b_raw.x = state.renderer_state.scene_data_vertex.Load(addresses.y);
                value_c_raw.x = state.renderer_state.scene_data_vertex.Load(addresses.z);
                break;
            case 2:
                value_a_raw.xy = state.renderer_state.scene_data_vertex.Load2(addresses.x);
                value_b_raw.xy = state.renderer_state.scene_data_vertex.Load2(addresses.y);
                value_c_raw.xy = state.renderer_state.scene_data_vertex.Load2(addresses.z);
                break;
            case 3:
                value_a_raw.xyz = state.renderer_state.scene_data_vertex.Load3(addresses.x);
                value_b_raw.xyz = state.renderer_state.scene_data_vertex.Load3(addresses.y);
                value_c_raw.xyz = state.renderer_state.scene_data_vertex.Load3(addresses.z);
                break;
            case 4:
                value_a_raw = state.renderer_state.scene_data_vertex.Load4(addresses.x);
                value_b_raw = state.renderer_state.scene_data_vertex.Load4(addresses.y);
                value_c_raw = state.renderer_state.scene_data_vertex.Load4(addresses.z);
                break;
        }

        // convert to float, int or color data
        int4 value_a = int4(0, 0, 0, 0);
        int4 value_b = int4(0, 0, 0, 0);
        int4 value_c = int4(0, 0, 0, 0);
        switch (info.GetElementType())
        {
            case SCENE_DATA_ELEMENT_TYPE_FLOAT: // reinterpret as float and convert to int
            case SCENE_DATA_ELEMENT_TYPE_COLOR: // (handled as float3, no spectral support)
                value_a = int4(asfloat(value_a_raw));
                value_b = int4(asfloat(value_b_raw));
                value_c = int4(asfloat(value_c_raw));
                break;

            case SCENE_DATA_ELEMENT_TYPE_INT:
                // reinterpret as signed int
                value_a = asint(value_a_raw);
                value_b = asint(value_b_raw);
                value_c = asint(value_c_raw);
                break;
        }

        // interpolate across the triangle
        const float3 barycentric = state.renderer_state.barycentric;
        switch (mode)
        {
            case SCENE_DATA_INTERPOLATION_MODE_LINEAR:
                return int4(float4(value_a) * barycentric.x +
                            float4(value_b) * barycentric.y +
                            float4(value_c) * barycentric.z);

            case SCENE_DATA_INTERPOLATION_MODE_NEAREST:
                if (barycentric.x > barycentric.y)
                    return barycentric.x > barycentric.z ? value_a : value_c;
                else
                    return barycentric.y > barycentric.z ? value_b : value_c;

            case SCENE_DATA_INTERPOLATION_MODE_NONE:
            default: // unsupported interpolation mode
                return default_value;
        }
    }

    case SCENE_DATA_KIND_INSTANCE:
    {
        // raw data read from the buffer
        uint address = info.GetByteOffset();
        uint4 value_raw = uint4(0, 0, 0, 0);
        switch (number_of_components)
        {
            case 1: value_raw.x = state.renderer_state.scene_data_instance.Load(address); break;
            case 2: value_raw.xy = state.renderer_state.scene_data_instance.Load2(address); break;
            case 3: value_raw.xyz = state.renderer_state.scene_data_instance.Load3(address); break;
            case 4: value_raw = state.renderer_state.scene_data_instance.Load4(address); break;
        }

        // convert to float, int or color data
        // do not interpolate as all currently available modes would result in the same value
        switch (info.GetElementType())
        {
            case SCENE_DATA_ELEMENT_TYPE_FLOAT: // reinterpret as float and convert to int
            case SCENE_DATA_ELEMENT_TYPE_COLOR: // (handled as float3, no spectral support)
                return int4(asfloat(value_raw));

            case SCENE_DATA_ELEMENT_TYPE_INT: // reinterpret as signed int
                return asint(value_raw);
        }
    }

    case SCENE_DATA_KIND_NONE:
    default:
        return default_value;
    }
}

int4 scene_data_lookup_int4(
    Shading_state_material state,
    uint scene_data_id,
    int4 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value, uniform_lookup, 4);
}

int3 scene_data_lookup_int3(
    Shading_state_material state,
    uint scene_data_id,
    int3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value.xyzx, uniform_lookup, 3).xyz;
}

int2 scene_data_lookup_int2(
    Shading_state_material state,
    uint scene_data_id,
    int2 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value.xyxx, uniform_lookup, 2).xy;
}

int scene_data_lookup_int(
    Shading_state_material state,
    uint scene_data_id,
    int default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_intX(state, scene_data_id, default_value.xxxx, uniform_lookup, 1).x;
}

// currently no scene data with derivatives is supported
Derived_float4 scene_data_lookup_deriv_float4(
    Shading_state_material state,
    uint scene_data_id,
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
    Shading_state_material state,
    uint scene_data_id,
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
    Shading_state_material state,
    uint scene_data_id,
    Derived_float3 default_value,
    bool uniform_lookup)
{
    return scene_data_lookup_deriv_float3(
        state, scene_data_id, default_value, uniform_lookup);
}

Derived_float2 scene_data_lookup_deriv_float2(
    Shading_state_material state,
    uint scene_data_id,
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
    Shading_state_material state,
    uint scene_data_id,
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
