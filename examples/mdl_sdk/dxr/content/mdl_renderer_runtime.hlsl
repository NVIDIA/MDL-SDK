/******************************************************************************
 * Copyright 2019 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

#ifndef MDL_RENDERER_RUNTIME_HLSLI
#define MDL_RENDERER_RUNTIME_HLSLI

// compiler constants defined from outside:
// - MDL_RO_DATA_SEGMENT_SLOT
// - MDL_ARGUMENT_BLOCK_SLOT
// - MDL_TEXTURE_SLOT_BEGIN
// - MDL_TEXTURE_SLOT_COUNT
// - MDL_TEXTURE_SAMPLER_SLOT

ByteAddressBuffer mdl_ro_data_segment : register( MDL_RO_DATA_SEGMENT_SLOT );
ByteAddressBuffer mdl_argument_block : register( MDL_ARGUMENT_BLOCK_SLOT );

Texture2D mdl_textures[MDL_TEXTURE_SLOT_COUNT] : register( MDL_TEXTURE_SLOT_BEGIN );

SamplerState mdl_sampler_2d : register( MDL_TEXTURE_SAMPLER_SLOT );

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
    return (val & (0xff << (8 * (offs & 3)))) != 0;
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
    return (val & (0xff << (8 * (offs & 3)))) != 0;
}

// Note: UV tiles are not supported in this example
uint tex_width_2d(RES_DATA_PARAM_DECL uint tex, int2 uv_tile)
{
    if (tex == 0)
        return 0;

    uint width, height;
    mdl_textures[NonUniformResourceIndex(tex - 1)].GetDimensions(width, height);
    return width;
}

// Note: UV tiles are not supported in this example
uint tex_height_2d(RES_DATA_PARAM_DECL uint tex, int2 uv_tile)
{
    if (tex == 0)
        return 0;

    uint width, height;
    mdl_textures[NonUniformResourceIndex(tex - 1)].GetDimensions(width, height);
    return height;
}

// The example does not support 3D and cube textures
uint tex_width_3d(RES_DATA_PARAM_DECL uint tex)     { return 0; }
uint tex_height_3d(RES_DATA_PARAM_DECL uint tex)    { return 0; }
uint tex_depth_3d(RES_DATA_PARAM_DECL uint tex)     { return 0; }
uint tex_width_cube(RES_DATA_PARAM_DECL uint tex)   { return 0; }
uint tex_height_cube(RES_DATA_PARAM_DECL uint tex)  { return 0; }

bool tex_texture_isvalid(RES_DATA_PARAM_DECL uint tex)
{
    return tex != 0;  // TODO: need to check number of available textures
}

float4 tex_lookup_float4_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v)
{
    if (tex == 0)
        return float4(0, 0, 0, 0);  // invalid texture

    if (wrap_u != TEX_WRAP_REPEAT || any(crop_u != float2(0, 1))) {
        if (wrap_u == TEX_WRAP_REPEAT) {
            coord.x -= floor(coord.x);
        } else {
            if (wrap_u == TEX_WRAP_CLIP && (coord.x < 0 || coord.x >= 1))
                return float4(0, 0, 0, 0);
            if (wrap_u == TEX_WRAP_MIRRORED_REPEAT) {
                float floored_val = floor(coord.x);
                if ((int(floored_val) & 1) != 0)
                    coord.x = 1 - (coord.x - floored_val);
                else
                    coord.x -= floored_val;
            }
            float inv_hdim = 0.5f / tex_width_2d(RES_DATA_PARAM tex, int2(0, 0));
            coord.x = min(max(coord.x, inv_hdim), 1.f - inv_hdim);
        }
        coord.x = coord.x * (crop_u.y - crop_u.x) + crop_u.x;
    }

    if (wrap_v != TEX_WRAP_REPEAT || any(crop_v != float2(0, 1))) {
        if (wrap_v == TEX_WRAP_REPEAT) {
            coord.y -= floor(coord.y);
        } else {
            if (wrap_v == TEX_WRAP_CLIP && (coord.y < 0 || coord.y >= 1))
                return float4(0, 0, 0, 0);
            if (wrap_v == TEX_WRAP_MIRRORED_REPEAT) {
                float floored_val = floor(coord.y);
                if ((int(floored_val) & 1) != 0)
                    coord.y = 1 - (coord.y - floored_val);
                else
                    coord.y -= floored_val;
            }
            float inv_hdim = 0.5f / tex_height_2d(RES_DATA_PARAM tex, int2(0, 0));
            coord.y = min(max(coord.y, inv_hdim), 1.f - inv_hdim);
        }
        coord.y = coord.y * (crop_v.y - crop_v.x) + crop_v.x;
    }

    // With HLSL 5.1
    // Note, since we don't have ddx and ddy in the compute pipeline, TextureObject::Sample() is not
    // available, we use SampleLevel instead and go for the most detailed level. Therefore, we don't
    // need mipmaps. Manual mip level computation is possible though.

    return mdl_textures[NonUniformResourceIndex(tex - 1)].SampleLevel(
        mdl_sampler_2d, coord, /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0);
}

float3 tex_lookup_float3_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v).xyz;
}

float3 tex_lookup_color_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v)
{
    return tex_lookup_float3_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v);
}

float2 tex_lookup_float2_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v).xy;
}

float tex_lookup_float_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    float2 coord,
    int wrap_u,
    int wrap_v,
    float2 crop_u,
    float2 crop_v)
{
    return tex_lookup_float4_2d(RES_DATA_PARAM tex, coord, wrap_u, wrap_v, crop_u, crop_v).x;
}

// The example does not support 3D textures
float4 tex_lookup_float4_3d(
    RES_DATA_PARAM_DECL
    uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,
    float2 crop_u, float2 crop_v, float2 crop_w)
{
    return float4(0, 0, 0, 0);
}

float3 tex_lookup_float3_3d(
    RES_DATA_PARAM_DECL
    uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,
    float2 crop_u, float2 crop_v, float2 crop_w)
{
    return tex_lookup_float4_3d(
        RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w).xyz;
}

float3 tex_lookup_color_3d(
    RES_DATA_PARAM_DECL
    uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,
    float2 crop_u, float2 crop_v, float2 crop_w)
{
    return tex_lookup_float3_3d(
        RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w);
}

float2 tex_lookup_float2_3d(
    RES_DATA_PARAM_DECL
    uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,
    float2 crop_u, float2 crop_v, float2 crop_w)
{
    return tex_lookup_float4_3d(
        RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w).xy;
}

float tex_lookup_float_3d(
    RES_DATA_PARAM_DECL
    uint tex, float3 coord, int wrap_u, int wrap_v, int wrap_w,
    float2 crop_u, float2 crop_v, float2 crop_w)
{
    return tex_lookup_float4_3d(
        RES_DATA_PARAM tex, coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w).x;
}

// The example does not support cube textures
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
    return tex_lookup_float3_cube(RES_DATA_PARAM tex, coord);
}

float2 tex_lookup_float2_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).xy;
}

float tex_lookup_float_cube(RES_DATA_PARAM_DECL uint tex, float3 coord)
{
    return tex_lookup_float4_cube(RES_DATA_PARAM tex, coord).x;
}

// The example does not support ptex textures
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

// Note: UV tiles are not supported in this example
float4 tex_texel_float4_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    int2 coord,
    int2 uv_tile)
{
    if (tex == 0)
        return float4(0, 0, 0, 0);  // invalid texture

    return mdl_textures[NonUniformResourceIndex(tex - 1)].Load(int3(coord, /*mipmaplevel=*/ 0));
}

// Note: UV tiles are not supported in this example
float3 tex_texel_float3_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    int2 coord,
    int2 uv_tile)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile).xyz;
}

// Note: UV tiles are not supported in this example
float3 tex_texel_color_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    int2 coord,
    int2 uv_tile)
{
    return tex_texel_float3_2d(RES_DATA_PARAM tex, coord, uv_tile);
}

// Note: UV tiles are not supported in this example
float2 tex_texel_float2_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    int2 coord,
    int2 uv_tile)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile).xy;
}

// Note: UV tiles are not supported in this example
float tex_texel_float_2d(
    RES_DATA_PARAM_DECL
    uint tex,
    int2 coord,
    int2 uv_tile)
{
    return tex_texel_float4_2d(RES_DATA_PARAM tex, coord, uv_tile).x;
}


// The example does not support 3D textures
float4 tex_texel_float4_3d(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return float4(0, 0, 0, 0);
}

float3 tex_texel_float3_3d(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord).xyz;
}

float3 tex_texel_color_3d(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float3_3d(RES_DATA_PARAM tex, coord);
}

float2 tex_texel_float2_3d(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord).xy;
}

float tex_texel_float_3d(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_3d(RES_DATA_PARAM tex, coord).x;
}

// The example does not support cube textures
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
    return tex_texel_float3_cube(RES_DATA_PARAM tex, coord);
}

float2 tex_texel_float2_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).xy;
}

float tex_texel_float_cube(RES_DATA_PARAM_DECL uint tex, int3 coord)
{
    return tex_texel_float4_cube(RES_DATA_PARAM tex, coord).x;
}

// The example does not support light profiles
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

#endif // MDL_RENDERER_RUNTIME_HLSLI
