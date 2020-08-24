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

#ifndef MDL_DXR_EXAMPLE_COMMON_HLSL
#define MDL_DXR_EXAMPLE_COMMON_HLSL

// macros the append the target code ID to the function name.
// this is required because the resulting DXIL libraries will be linked to same pipeline object
// and for that, the entry point names have to be unique.
#define export_name_impl(name, id) name ## _ ## id
#define export_name_impl_2(name, id) export_name_impl(name, id)
#define export_name(name) export_name_impl_2(name, TARGET_CODE_ID)

static const float M_PI =          3.14159265358979323846;
static const float M_ONE_OVER_PI = 0.318309886183790671538;
static const float DIRAC = -1.0f;

SamplerState sampler_latlong : register(s1);

cbuffer SceneConstants : register(b1)
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
}

// Ray typed, has to match with CPU version
#if defined(WITH_ENUM_SUPPORT)
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW,

    RAY_TYPE_COUNT,
    RAY_TYPE_FORCE_32_BIT = 0xffffffffU
};
#else
    #define RayType uint
    #define RAY_TYPE_RADIANCE   0
    #define RAY_TYPE_SHADOW     1
    #define RAY_TYPE_COUNT      (RAY_TYPE_SHADOW + 1)
#endif

#if defined(WITH_ENUM_SUPPORT)
enum RadianceHitInfoFlags
{
    FLAG_NONE = 0,
    FLAG_INSIDE = 1,
    FLAG_DONE = 2,
    FLAG_FIRST_PATH_SEGMENT = 4
};
#else
    #define RadianceHitInfoFlags uint
    #define FLAG_NONE               0
    #define FLAG_INSIDE             1
    #define FLAG_DONE               2
    #define FLAG_FIRST_PATH_SEGMENT 4
#endif

void add_flag(inout uint flags, uint to_add) { flags |= to_add; }
void toggle_flag(inout uint flags, uint to_toggle) { flags ^= to_toggle; }
void remove_flag(inout uint flags, uint to_remove) { flags &= ~to_remove; }
bool has_flag(int flags, uint to_check) { return (flags & to_check) != 0; }

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

//-------------------------------------------------------------------------------------------------
// Scene Data API
//-------------------------------------------------------------------------------------------------

/// interpolation of the data over the primitive
#if defined(WITH_ENUM_SUPPORT)
    enum SceneDataInterpolationMode
    {
        SCENE_DATA_INTERPOLATION_MODE_NONE = 0,
        SCENE_DATA_INTERPOLATION_MODE_LINEAR = 1,
        SCENE_DATA_INTERPOLATION_MODE_NEAREST = 2,
    };
#else
    #define SceneDataInterpolationMode uint
    #define SCENE_DATA_INTERPOLATION_MODE_NONE      0
    #define SCENE_DATA_INTERPOLATION_MODE_LINEAR    1
    #define SCENE_DATA_INTERPOLATION_MODE_NEAREST   2
#endif

/// Scope a scene data element belongs to
#if defined(WITH_ENUM_SUPPORT)
    enum SceneDataKind
    {
        SCENE_DATA_KIND_NONE = 0,
        SCENE_DATA_KIND_VERTEX = 1,
        SCENE_DATA_KIND_INSTANCE = 2,
    };
#else
    #define SceneDataKind uint
    #define SCENE_DATA_KIND_NONE        0
    #define SCENE_DATA_KIND_VERTEX      1
    #define SCENE_DATA_KIND_INSTANCE    2
#endif

/// Basic element type of the scene data
#if defined(WITH_ENUM_SUPPORT)
    enum SceneDataElementType
    {
        SCENE_DATA_ELEMENT_TYPE_FLOAT = 0,
        SCENE_DATA_ELEMENT_TYPE_INT = 1,
        SCENE_DATA_ELEMENT_TYPE_COLOR = 2
    };
#else
    #define SceneDataElementType uint
    #define SCENE_DATA_ELEMENT_TYPE_FLOAT   0
    #define SCENE_DATA_ELEMENT_TYPE_INT     1
    #define SCENE_DATA_ELEMENT_TYPE_COLOR   2
#endif

// Infos about the interleaved vertex layout (compressed)
struct SceneDataInfo
{
    /// Scope a scene data element belongs to (4 bits)
    inline SceneDataKind GetKind()
    {
        return (SceneDataKind) ((packed_data.x & 0xF0000000u) >> 28);
    }

    /// Basic element type of the scene data (4 bits)
    inline SceneDataElementType GetElementType()
    {
        return (SceneDataElementType) ((packed_data.x & 0x0F000000u) >> 24);
    }

    /// Interpolation of the data over the primitive (4 bits)
    SceneDataInterpolationMode GetInterpolationMode()
    {
        return (SceneDataInterpolationMode) ((packed_data.x & 0x00F00000u) >> 20);
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

    // use getter function to unpack, see scene.cpp for corresponding c++ code for packing
    uint2 packed_data;
};

// renderer state object that is passed to mdl runtime functions
struct DXRRendererState
{
    // scene data buffer for object/instance data
    ByteAddressBuffer scene_data_instance;

    // The mapping between scene_data_id and scene data buffer layout
    StructuredBuffer<SceneDataInfo> scene_data_infos;

    // index offset for the first info object relevant for this geometry
    uint scene_data_info_offset;

    // the per mesh scene data buffer, includes the vertex buffer
    ByteAddressBuffer scene_data_vertex;

    // global offset in the data buffer (for object, geometry, ...)
    uint scene_data_geometry_byte_offset;

    // vertex indices if the hit triangle (from index buffer)
    uint3 hit_vertex_indices;

    // barycentric coordinates of the hit point within the triangle
    float3 barycentric;
};
// use this structure as renderer state int the MDL shading state material
#define RENDERER_STATE_TYPE DXRRendererState


// Positions, normals, and tangents are mandatory for this renderer. The vertex buffer always
// contains this data at the beginning of the (interleaved) per vertex data.
#if defined(WITH_ENUM_SUPPORT)
    enum VertexByteOffset
    {
        VERT_BYTEOFFSET_POSITION = 0,
        VERT_BYTEOFFSET_NORMAL = 12,
        VERT_BYTEOFFSET_TANGENT = 24,
    };
#else
    #define VertexByteOffset uint
    #define VERT_BYTEOFFSET_POSITION    0
    #define VERT_BYTEOFFSET_NORMAL      12
    #define VERT_BYTEOFFSET_TANGENT     24
#endif

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
// Environment
//-------------------------------------------------------------------------------------------------

struct Environment_sample_data
{
    uint alias;
    float q;
};

float3 environment_evaluate(
    Texture2D<float4> lat_long_tex,
    StructuredBuffer<Environment_sample_data> sample_buffer,
    float3 normalized_dir,
    out float pdf)
{
    // assuming lat long
    const float u = atan2(normalized_dir.z, normalized_dir.x) * 0.5f * M_ONE_OVER_PI + 0.5f;
    const float v = acos(-normalized_dir.y) * M_ONE_OVER_PI;

    // get radiance and calculate pdf
    float3 t = lat_long_tex.SampleLevel(
        sampler_latlong, float2(u, v), /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0).xyz;
    pdf = max(t.x, max(t.y, t.z)) * environment_inv_integral;
    return t * environment_intensity_factor;
}

float3 environment_sample(
    Texture2D<float4> lat_long_tex,
    StructuredBuffer<Environment_sample_data> sample_buffer,
    inout uint seed,
    out float3 to_light,
    out float pdf)
{
    float3 xi;
    xi.x = rnd(seed);
    xi.y = rnd(seed);
    xi.z = rnd(seed);


    uint width;
    uint height;
    lat_long_tex.GetDimensions(width, height);

    const uint size = width * height;
    const uint idx = min(uint(xi.x * float(size)), size - 1);

    uint env_idx;
    if (xi.y < sample_buffer[idx].q)
    {
        env_idx = idx;
        xi.y /= sample_buffer[idx].q;
    }
    else
    {
        env_idx = sample_buffer[idx].alias;
        xi.y = (xi.y - sample_buffer[idx].q) / (1.0f - sample_buffer[idx].q);
    }

    const uint py = env_idx / width;
    const uint px = env_idx % width;

    // uniformly sample spherical area of pixel
    const float u = float(px + xi.y) / float(width);
    const float phi = u * (2.0f * M_PI) - M_PI;
    float sin_phi;
    float cos_phi;
    sincos(phi, sin_phi, cos_phi);

    const float step_theta = M_PI / float(height);
    const float theta0 = float(py) * step_theta;
    const float cos_theta = cos(theta0) * (1.0f - xi.z) + cos(theta0 + step_theta) * xi.z;
    const float theta = acos(cos_theta);
    const float sin_theta = sin(theta);
    to_light = float3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

    // lookup filtered value and calculate pdf
    const float v = theta * M_ONE_OVER_PI;
    float3 t = lat_long_tex.SampleLevel(
        sampler_latlong, float2(u, v), /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0).xyz;
    pdf = max(t.x, max(t.y, t.z)) * environment_inv_integral;
    return t * environment_intensity_factor;
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

#endif