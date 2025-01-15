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

#ifndef MDL_DXR_EXAMPLE_COMMON_HLSL
#define MDL_DXR_EXAMPLE_COMMON_HLSL

static const float M_PI =          3.14159265358979323846;
static const float M_ONE_OVER_PI = 0.318309886183790671538;
static const float DIRAC = -1.0f;

//-------------------------------------------------------------------------------------------------
// Renderer State
//-------------------------------------------------------------------------------------------------

// Ray typed, has to match with CPU version
#define RayType uint
#define RAY_TYPE_RADIANCE   0
#define RAY_TYPE_SHADOW     1
#define RAY_TYPE_COUNT      (RAY_TYPE_SHADOW + 1)

// Ray state expressed using a few flags
#define RadianceHitInfoFlags uint
#define FLAG_NONE               0
#define FLAG_INSIDE             (1 << 0)
#define FLAG_DONE               (1 << 1)
#define FLAG_FIRST_PATH_SEGMENT (1 << 2)
#define FLAG_LAST_PATH_SEGMENT  (1 << 3)
#define FLAG_CAMERA_RAY         (1 << 4)

void add_flag(inout RadianceHitInfoFlags flags, RadianceHitInfoFlags to_add) { flags |= to_add; }
void toggle_flag(inout RadianceHitInfoFlags flags, RadianceHitInfoFlags to_toggle) { flags ^= to_toggle; }
void remove_flag(inout RadianceHitInfoFlags flags, RadianceHitInfoFlags to_remove) { flags &= ~to_remove; }
bool has_flag(RadianceHitInfoFlags flags, RadianceHitInfoFlags to_check) { return (flags & to_check) != 0; }

// payload for RAY_TYPE_RADIANCE
struct RadianceHitInfo
{
    float3 contribution;
    float3 weight;

    float3 ray_origin_next;
    float3 ray_direction_next;

    uint seed;
    float last_bsdf_pdf;
    uint flags;
};

// payload for RAY_TYPE_SHADOW
struct ShadowHitInfo
{
    bool isHit;
    uint seed;
};

// Attributes output by the ray tracing when hitting a surface
struct Attributes
{
    float2 bary;
};

// Helper to make NaN and INF values visible in the output image.
float3 encode_errors(float3 color)
{
    return any(isnan(color) | isinf(color)) ? float3(0.0f, 0.0f, 1.0e+30f) : color;
}


// renderer state object that is passed to mdl runtime functions
struct DXRRendererState
{
    // index offset for the first info object relevant for this geometry
    uint scene_data_info_offset;

    // global offset in the data buffer (for object, geometry, ...)
    uint scene_data_geometry_byte_offset;

    // vertex indices if the hit triangle (from index buffer)
    uint3 hit_vertex_indices;

    // barycentric coordinates of the hit point within the triangle
    float3 barycentric;

    // true if the hit point was on the backside of a triangle, based on geom normal and ray direction
    bool hit_backface;
};
// use this structure as renderer state in the MDL shading state material
#define RENDERER_STATE_TYPE DXRRendererState


// Positions, normals, and tangents are mandatory for this renderer. The vertex buffer always
// contains this data at the beginning of the (interleaved) per vertex data.
#define VertexByteOffset uint
#define VERT_BYTEOFFSET_POSITION    0
#define VERT_BYTEOFFSET_NORMAL      12
#define VERT_BYTEOFFSET_TANGENT     24


// include the target types here, as it depends on RENDERER_STATE_TYPE
#include "content/mdl_target_code_types.hlsl"


//-------------------------------------------------------------------------------------------------
// Scene Constants mostly mapped to the UI
//-------------------------------------------------------------------------------------------------

struct SceneConstants
{
    float total_time;
    float delta_time;

    // (progressive) rendering
    uint progressive_iteration;
    uint max_ray_depth;
    uint iterations_per_frame;

    // tone mapping
    float exposure_compensation;
    float firefly_clamp_threshold;
    float burn_out;

    // one additional point light for illustration
    uint point_light_enabled;
    float3 point_light_position;
    float3 point_light_intensity;

    // gamma correction while rendering to the frame buffer
    float output_gamma_correction;

    // environment light
    float environment_intensity_factor;
    float environment_inv_integral;

    // when auxiliary buffers are enabled, this index is used to select to one to display
    uint display_buffer_index;

    // enable animation mode, progression is limited, mdl state will have an animation_time
    uint enable_animiation;

    /// replace the background with a constant color when visible to the camera
    uint background_color_enabled;
    float3 background_color;

    /// uv transformations
    float2 uv_scale;
    float2 uv_offset;
    uint uv_repeat;
    uint uv_saturate;

    // rotation of the environment [0, 1]
    float environment_rotation;

    // defines the scale of the scene
    float meters_per_scene_unit;

    // far plane that defines the maximum ray travel distance
    float far_plane_distance;

    // if >= 0, a visual representation of the selected AOV is displayed instead of the
    // regular PBR rendering
    int aov_index_to_render;

    // the BSDF data flags to use when executing BSDF functions
    Df_flags bsdf_data_flags;
};

//-------------------------------------------------------------------------------------------------
// Camera
//-------------------------------------------------------------------------------------------------

struct CameraParams
{
    float4x4 viewI;
    float4x4 projectionI;
};


//-------------------------------------------------------------------------------------------------
// Environment
//-------------------------------------------------------------------------------------------------

// Element of the environment sampling data
struct Env_Sample
{
    uint alias;
    float q;
};


//-------------------------------------------------------------------------------------------------
// Materials
//-------------------------------------------------------------------------------------------------


#define MATERIAL_CODE_FEATURE_HAS_INIT              (1 << 0)
#define MATERIAL_CODE_FEATURE_SURFACE_SCATTERING    (1 << 1)
#define MATERIAL_CODE_FEATURE_SURFACE_EMISSION      (1 << 2)
#define MATERIAL_CODE_FEATURE_BACKFACE_SCATTERING   (1 << 3)
#define MATERIAL_CODE_FEATURE_BACKFACE_EMISSION     (1 << 4)
#define MATERIAL_CODE_FEATURE_VOLUME_ABSORPTION     (1 << 5)
#define MATERIAL_CODE_FEATURE_CUTOUT_OPACITY        (1 << 6)
#define MATERIAL_CODE_FEATURE_CAN_BE_THIN_WALLED    (1 << 7)
#define MATERIAL_CODE_FEATURE_HAS_AOVS              (1 << 8)

#define MATERIAL_FLAG_SINGLE_SIDED                  (1 << 0)

bool has_feature(uint flags, uint to_check) { return (flags & to_check) != 0; }



struct Material_constants
{
    // shared for all material compiled from the same MDL material
    // - none -

    // individual properties of the different material instances
    // ------------------------------------------------------------------------
    int material_id;    // id of the material in scene
    uint features;      // material code features encoded in bit mask
    uint flags;         // material features indpendent of MDL

    // true if the MDL material code has a single-init function
    bool has_init()
    {
        #ifdef MDL_HAS_INIT
            // if known at compile time
            return MDL_HAS_INIT == 1;
        #else
            // if not known at compile time we check the flags passed in the constants
            return has_feature(features, MATERIAL_CODE_FEATURE_HAS_INIT);
        #endif
    }

    bool has_volume_absorption()
    {
        #ifdef MDL_HAS_VOLUME_ABSORPTION
            return MDL_HAS_VOLUME_ABSORPTION == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_VOLUME_ABSORPTION);
        #endif
    }

    bool has_surface_scattering()
    {
        #ifdef MDL_HAS_SURFACE_SCATTERING
            return MDL_HAS_SURFACE_SCATTERING == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_SURFACE_SCATTERING);
        #endif
    }

    bool has_surface_emission()
    {
        #ifdef MDL_HAS_SURFACE_EMISSION
            return MDL_HAS_SURFACE_EMISSION == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_SURFACE_EMISSION);
        #endif
    }

    bool has_backface_scattering()
    {
        #ifdef MDL_HAS_BACKFACE_SCATTERING
            return MDL_HAS_BACKFACE_SCATTERING == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_BACKFACE_SCATTERING);
        #endif
    }


    bool has_backface_emission()
    {
        #ifdef MDL_HAS_BACKFACE_EMISSION
            return MDL_HAS_BACKFACE_EMISSION == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_BACKFACE_EMISSION);
        #endif
    }

    bool has_cutout_opacity()
    {
        #ifdef MDL_HAS_CUTOUT_OPACITY
            return MDL_HAS_CUTOUT_OPACITY == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_CUTOUT_OPACITY);
        #endif
    }

    bool can_be_thin_walled()
    {
        #ifdef MDL_CAN_BE_THIN_WALLED
            return MDL_CAN_BE_THIN_WALLED == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_CAN_BE_THIN_WALLED);
        #endif
    }

    bool has_aovs()
    {
        #ifdef MDL_HAS_AOVS
            return MDL_HAS_AOVS == 1;
        #else
            return has_feature(flags, MATERIAL_CODE_FEATURE_HAS_AOVS);
        #endif
    }

    bool is_single_sided()
    {
        return has_feature(flags, MATERIAL_FLAG_SINGLE_SIDED);
    }

    // TODO for testing only init, add the rest as well
};


//-------------------------------------------------------------------------------------------------
// MDL Resources
//-------------------------------------------------------------------------------------------------

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


//-------------------------------------------------------------------------------------------------
// Scene Data API
//-------------------------------------------------------------------------------------------------

/// interpolation of the data over the primitive
#define SceneDataInterpolationMode uint
#define SCENE_DATA_INTERPOLATION_MODE_NONE      0
#define SCENE_DATA_INTERPOLATION_MODE_LINEAR    1
#define SCENE_DATA_INTERPOLATION_MODE_NEAREST   2

/// Scope a scene data element belongs to
#define SceneDataKind uint
#define SCENE_DATA_KIND_NONE        0
#define SCENE_DATA_KIND_VERTEX      1
#define SCENE_DATA_KIND_INSTANCE    2

/// Basic element type of the scene data
#define SceneDataElementType uint
#define SCENE_DATA_ELEMENT_TYPE_FLOAT   0
#define SCENE_DATA_ELEMENT_TYPE_INT     1
#define SCENE_DATA_ELEMENT_TYPE_COLOR   2

// Infos about the interleaved vertex layout (compressed)
struct SceneDataInfo
{
    // use getter function to unpack, see scene.cpp for corresponding c++ code for packing
    uint2 packed_data;

    /// Scope a scene data element belongs to (4 bits)
    inline SceneDataKind GetKind()
    {
        return (SceneDataKind)((packed_data.x & 0xF0000000u) >> 28);
    }

    /// Basic element type of the scene data (4 bits)
    inline SceneDataElementType GetElementType()
    {
        return (SceneDataElementType)((packed_data.x & 0x0F000000u) >> 24);
    }

    /// Interpolation of the data over the primitive (4 bits)
    SceneDataInterpolationMode GetInterpolationMode()
    {
        return (SceneDataInterpolationMode)((packed_data.x & 0x00F00000u) >> 20);
    }

    /// Indicates whether there the scene data is uniform. (1 bit)
    bool GetUniform()
    {
        return (packed_data.x & 0x00010000u) > 0;
    }

    /// Offset between two elements. For interleaved vertex buffers, this is the vertex size in byte.
    /// For non-interleaved buffers, this is the element size in byte. (16 bit)
    uint GetByteStride()
    {
        return (packed_data.x & 0x0000FFFFu);
    }

    /// The offset to the data element within an interleaved vertex buffer, or the absolute
    /// offset to the base (e.g. of the geometry data) in non-interleaved buffers
    uint GetByteOffset()
    {
        return packed_data.y;
    }

};


//-------------------------------------------------------------------------------------------------
// random number generator based on the Optix SDK
//-------------------------------------------------------------------------------------------------
uint tea(uint N, uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    for (uint n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random uint in [0, 2^24)
uint lcg(inout uint prev)
{
    const uint LCG_A = 1664525u;
    const uint LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
float rnd(inout uint prev)
{
    return ((float) lcg(prev) / (float) 0x01000000);
}

float2 rnd2(inout uint prev)
{
    return float2((float) lcg(prev) / (float) 0x01000000,
                  (float) lcg(prev) / (float) 0x01000000);
}

float3 rnd3(inout uint prev)
{
    return float3((float) lcg(prev) / (float) 0x01000000,
                  (float) lcg(prev) / (float) 0x01000000,
                  (float) lcg(prev) / (float) 0x01000000);
}

float4 rnd4(inout uint prev)
{
    return float4((float) lcg(prev) / (float) 0x01000000,
                  (float) lcg(prev) / (float) 0x01000000,
                  (float) lcg(prev) / (float) 0x01000000,
                  (float) lcg(prev) / (float) 0x01000000);
}

//-------------------------------------------------------------------------------------------------
// Math helper
//-------------------------------------------------------------------------------------------------

// convert float4x3 to 4x4, to be compatible with the slang compiler
float4x4 to4x4(float3x4 source)
{
    return float4x4(source[0], source[1], source[2], float4(0.0f, 0.0f, 0.0f, 1.0f));
}

//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

float3 offset_ray(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    const int3 of_i = int3(int_scale * n);

    float3 p_i = float3(asfloat(asint(p.x) + ((p.x < 0.0f) ? -of_i.x : of_i.x)),
                        asfloat(asint(p.y) + ((p.y < 0.0f) ? -of_i.y : of_i.y)),
                        asfloat(asint(p.z) + ((p.z < 0.0f) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

//-------------------------------------------------------------------------------------------------
// make all global resources available to all shaders
//-------------------------------------------------------------------------------------------------
#include "content/resource_bindings.hlsl"

#endif