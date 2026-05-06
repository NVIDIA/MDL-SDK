/******************************************************************************
 * Copyright (c) 2019-2026, NVIDIA CORPORATION. All rights reserved.
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
#define FLAG_SSS                (1 << 5)
#define FLAG_SSS_R              (1 << 6)
#define FLAG_SSS_G              (1 << 7)
#define FLAG_SSS_B              (1 << 8)

void add_flag(inout RadianceHitInfoFlags flags, RadianceHitInfoFlags to_add) { flags |= to_add; }
void toggle_flag(inout RadianceHitInfoFlags flags, RadianceHitInfoFlags to_toggle) { flags ^= to_toggle; }
void remove_flag(inout RadianceHitInfoFlags flags, RadianceHitInfoFlags to_remove) { flags &= ~to_remove; }
bool has_flag(RadianceHitInfoFlags flags, RadianceHitInfoFlags to_check) { return (flags & to_check) != 0; }

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


// include the target types here, as it depends on RENDERER_STATE_TYPE
#include "content/mdl_target_code_types.hlsl"


// Positions, normals, and tangents are mandatory for this renderer. The vertex buffer always
// contains this data at the beginning of the (interleaved) per vertex data.
#define VertexByteOffset uint
#define VERT_BYTEOFFSET_POSITION    0
#define VERT_BYTEOFFSET_NORMAL      12
#define VERT_BYTEOFFSET_TANGENT     24


// payload for RAY_TYPE_RADIANCE
struct RadianceHitInfo
{
    Color_sample weight;
    float3 contribution;

    float3 ray_origin_next;
    float3 ray_direction_next;

    uint seed;
    float last_bsdf_pdf;
    uint flags;

#if defined(MDL_SPECTRAL_RENDERING)
    // Active wavelengths (in nm) for spectral rendering.
    Spectral_sample spectral_wavelengths;
    float spectral_pdf_ratios[MDL_DF_SPECTRAL_SAMPLES - 1];

    float update_spectral_pdf_ratios(Spectral_sample pdfs, bool specular, bool specular_dispersion)
    {
        // main wavelength has been used for sampling, so MIS weight is
        // w = p[0] / sum(p) = 1 / (sum(p) / p[0])
        // for pdf p up to this point in the path

        // here we:
        // - update the pdf ratios
        // - compute the new weight
        // - return factor that changes from old to new weight

        if (specular && !specular_dispersion) // specular without dispersion: nothing to do
            return 1.0f;

        if (!specular_dispersion && pdfs.values[0] <= 0.0f) // this really has zero probablity
            return 0.0f;

        float inv_w_old = 1.0f;
        float inv_w_new = 1.0f;
        const float inv_p0 = specular_dispersion ? 0.0f : (1.0f / pdfs.values[0]);
        [unroll] for (int i = 1; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
        {
            inv_w_old += spectral_pdf_ratios[i - 1];
            spectral_pdf_ratios[i - 1] *= pdfs.values[i] * inv_p0;
            inv_w_new += spectral_pdf_ratios[i - 1];
        }

        return inv_w_old / inv_w_new;
    }
#endif
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

    // for counting SSS steps separate from 'max_ray_depth'
    uint max_sss_depth;

    // spectral wavelength range [nm]
    float spectral_min_wavelength;
    float spectral_max_wavelength;
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


#define MATERIAL_CODE_FEATURE_HAS_INIT                      (1 << 0)
#define MATERIAL_CODE_FEATURE_SURFACE_SCATTERING            (1 << 1)
#define MATERIAL_CODE_FEATURE_SURFACE_EMISSION              (1 << 2)
#define MATERIAL_CODE_FEATURE_BACKFACE_SCATTERING           (1 << 3)
#define MATERIAL_CODE_FEATURE_BACKFACE_EMISSION             (1 << 4)
#define MATERIAL_CODE_FEATURE_VOLUME_ABSORPTION             (1 << 5)
#define MATERIAL_CODE_FEATURE_VOLUME_SCATTERING             (1 << 6)
#define MATERIAL_CODE_FEATURE_VOLUME_SCATTERING_DIR_BIAS    (1 << 7)
#define MATERIAL_CODE_FEATURE_CUTOUT_OPACITY                (1 << 8)
#define MATERIAL_CODE_FEATURE_CAN_BE_THIN_WALLED            (1 << 9)
#define MATERIAL_CODE_FEATURE_HAS_AOVS                      (1 << 10)

#define MATERIAL_FLAG_SINGLE_SIDED                          (1 << 0)

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

    bool has_volume_scattering()
    {
        #ifdef MDL_HAS_VOLUME_SCATTERING
            return MDL_HAS_VOLUME_SCATTERING == 1;
        #else
            return has_feature(features, MATERIAL_CODE_FEATURE_VOLUME_SCATTERING);
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
// make all global resources available to all shaders
//-------------------------------------------------------------------------------------------------
#include "content/resource_bindings.hlsl"


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

// Duff et al - "Building an Orthonormal Basis, Revisited"
void create_basis(float3 n, out float3 b1, out float3 b2)
{
    const float sign = (n.z >= 0.0f) ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
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
// Spectral rendering helpers
//-------------------------------------------------------------------------------------------------

#if defined(MDL_SPECTRAL_RENDERING)

// D65 standard illuminant spectral power distribution, 360-830 nm in 5 nm steps (95 entries).
// Scaled so that the luminance integral (Y channel) equals 1.
static const float s_cie_d65[95] = {
    6.462114e-06f, 6.839740e-06f, 7.217367e-06f, 7.070938e-06f,
    6.924510e-06f, 7.248223e-06f, 7.571951e-06f, 9.519149e-06f,
    1.146636e-05f, 1.207124e-05f, 1.267613e-05f, 1.281093e-05f,
    1.294573e-05f, 1.247813e-05f, 1.201053e-05f, 1.327021e-05f,
    1.452989e-05f, 1.537108e-05f, 1.621241e-05f, 1.626811e-05f,
    1.632381e-05f, 1.611929e-05f, 1.591492e-05f, 1.598850e-05f,
    1.606207e-05f, 1.556936e-05f, 1.507664e-05f, 1.511419e-05f,
    1.515188e-05f, 1.504436e-05f, 1.493684e-05f, 1.472817e-05f,
    1.451950e-05f, 1.472027e-05f, 1.492118e-05f, 1.469367e-05f,
    1.446616e-05f, 1.444122e-05f, 1.441642e-05f, 1.413611e-05f,
    1.385581e-05f, 1.360185e-05f, 1.334788e-05f, 1.331004e-05f,
    1.327220e-05f, 1.278016e-05f, 1.228811e-05f, 1.237960e-05f,
    1.247109e-05f, 1.244288e-05f, 1.241468e-05f, 1.228302e-05f,
    1.215136e-05f, 1.184583e-05f, 1.154031e-05f, 1.156876e-05f,
    1.159720e-05f, 1.134278e-05f, 1.108836e-05f, 1.110137e-05f,
    1.111438e-05f, 1.125732e-05f, 1.140026e-05f, 1.112358e-05f,
    1.084691e-05f, 1.025367e-05f, 9.660451e-06f, 9.791236e-06f,
    9.922021e-06f, 1.011183e-05f, 1.030166e-05f, 9.418694e-06f,
    8.535733e-06f, 9.109474e-06f, 9.683216e-06f, 1.004356e-05f,
    1.040391e-05f, 9.607591e-06f, 8.811283e-06f, 7.621443e-06f,
    6.431617e-06f, 7.844023e-06f, 9.256429e-06f, 9.019315e-06f,
    8.782200e-06f, 8.846020e-06f, 8.909840e-06f, 8.573684e-06f,
    8.237542e-06f, 7.718434e-06f, 7.199340e-06f, 7.579100e-06f,
    7.958860e-06f, 8.157816e-06f, 8.356785e-06f
};

static float lookup_d65(float lambda)
{
    float f = (lambda - 360.0f) / (830.0f - 360.0f);
    if (f < 0.0f || f > 1.0f)
        return 0.0f;
    f *= float(95 - 1);
    uint b0 = min((uint)f, 95u - 1u);
    uint b1 = (b0 < (95u - 1u)) ? (b0 + 1u) : b0;
    float w1 = f - float(b0);
    return s_cie_d65[b0] * (1.0f - w1) + s_cie_d65[b1] * w1;
}

// RGB-to-spectral conversion.
// Uses Jendersie - "Fast Spectral Upsampling of Volume Attenuation Coefficients".
Spectral_sample rgb_to_spectral(float3 rgb, Spectral_sample lambdas, bool is_emission)
{
    Spectral_sample s;
    [unroll] for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        float lambda = lambdas.values[i];
        s.values[i] = (lambda < 485.0f) ? rgb.b : ((lambda < 595.9f) ? rgb.g : rgb.r);

        // for emission, apply spectral illuminant
        if (is_emission)
            s.values[i] *= lookup_d65(lambda);
    }
    return s;
}

// Convert a spectral radiance array to CIE XYZ and then to linear sRGB.
// Uses Wyman et al. - "Simple Analytic Approximations to the CIE XYZ Color Matching Functions".
float3 spectral_to_rgb(Spectral_sample values, Spectral_sample lambdas, bool is_reflectivity)
{
    float3 xyz = float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        float lambda = lambdas.values[i];
        if (lambda < 360.0f || lambda > 830.0f)
            continue;

        // for reflectivity values we need to multiply by the spectral whitepoint of the RGB color space (normalized to luminance 1)
        const float factor = is_reflectivity ? lookup_d65(lambda) : 1.0f;
        {
            const float p1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
            const float p2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
            const float p3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);
            xyz.x += (0.362f * exp(-0.5f * p1 * p1)
                    + 1.056f * exp(-0.5f * p2 * p2)
                    - 0.065f * exp(-0.5f * p3 * p3)) * values.values[i] * factor;
        }
        {
            const float p1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
            const float p2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);
            xyz.y += (0.821f * exp(-0.5f * p1 * p1)
                    + 0.286f * exp(-0.5f * p2 * p2)) * values.values[i] * factor;
        }
        {
            const float p1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
            const float p2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
            xyz.z += (1.217f * exp(-0.5f * p1 * p1)
                    + 0.681f * exp(-0.5f * p2 * p2)) * values.values[i] * factor;
        }
    }

    // apply scaling from radiometric to photometric units
    xyz *= 683.002f;

    // MDL_DF_SPECTRAL_SAMPLES samples uniformly on wavelength range
    if (scene_constants.spectral_max_wavelength != scene_constants.spectral_min_wavelength) {
        xyz *= (scene_constants.spectral_max_wavelength - scene_constants.spectral_min_wavelength)
            / float(MDL_DF_SPECTRAL_SAMPLES);
    }

    // XYZ -> linear sRGB
    return float3(
        dot(xyz, float3( 3.240600f, -1.537200f, -0.498600f)),
        dot(xyz, float3(-0.968900f,  1.875800f,  0.041500f)),
        dot(xyz, float3( 0.055700f, -0.204000f,  1.057000f)));
}

#endif // MDL_SPECTRAL_RENDERING

#endif
