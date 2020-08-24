/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/optix7/optix7_mdl.h
//
// Header file containing definitions used by host and device code.

#ifndef OPTIX7_MDL_H
#define OPTIX7_MDL_H

#include <stdint.h>


// If enabled, the no-direct-call mode will be used. In this mode, the generated MDL code will
// be linked, inlined and optimized with the radiance closest hit shader. This makes it possible
// to get rid of most MDL state related memory reads and writes, leading to much better performance
// at the cost of higher compile times.
// If disabled, there will only be one radiance closest hit shader, which dynamically calls
// direct callables generated for the different functions generated from the MDL code
#define NO_DIRECT_CALL


// If enabled, the ray contribution is stored in the payload registers instead of the RadiancePRD
// structure, which leads to better performance due to less memory reads and writes
#define CONTRIB_IN_PAYLOAD


// The maximum path depth of the ray.
// Keep in sync with RAY_FLAGS_DEPTH_MASK
#define MAX_DEPTH 3


//------------------------------------------------------------------------------
// enum types
//------------------------------------------------------------------------------

enum RayFlags
{
    RAY_FLAGS_NONE       = 0,
    RAY_FLAGS_DEPTH_MASK = 3,
    RAY_FLAGS_INSIDE     = 4         // if set, the ray is inside a material
};


enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


//------------------------------------------------------------------------------
// scene structs
//------------------------------------------------------------------------------

struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};


struct EnvironmentAccel
{
    unsigned int alias;
    float        q;
};


struct EnvironmentLight
{
    uint2               size;
    cudaTextureObject_t texture;
    EnvironmentAccel*   accel;
    float               intensity;
    float               inv_env_integral;
};


struct Params
{
    float3*      accum_buffer;
    uchar4*      frame_buffer;
    uint32_t     subframe_index;
    uint32_t     width;
    uint32_t     height;
    uint32_t     samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    ParallelogramLight     light;
    EnvironmentLight       env_light;

    OptixTraversableHandle handle;
};


// Struct representing a vertex of a scene object.
struct MeshVertex
{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 binormal;
    float2 tex_coord;

    // Example scene data fields
    float3 color;
    int2   row_column;
    int    pad;

    MeshVertex(
        float3 const &p,
        float3 const &n,
        float3 const &t,
        float3 const &b,
        float2 const &uv)
    : position(p)
    , normal(n)
    , tangent(t)
    , binormal(b)
    , tex_coord(uv)
    , color(make_float3(0, 0, 0))
    , row_column(make_int2(0, 0))
    , pad(0)
    { }

    MeshVertex() {}
};


struct Texture_handler;


struct SceneDataInfo
{
    enum Data_kind
    {
        DK_NONE,
        DK_VERTEX_COLOR,
        DK_ROW_COLUMN
    };

    enum Interpolation_mode
    {
        IM_NEAREST,
        IM_LINEAR
    };

    SceneDataInfo()
    : data_kind(DK_NONE)
    , interpolation_mode(IM_NEAREST)
    , is_uniform(false)
    {
    }

    Data_kind           data_kind;
    Interpolation_mode  interpolation_mode;

    bool                is_uniform;
};


struct HitGroupData
{
    MeshVertex      *vertices;
    short3          *indices;
    char            *arg_block;
    Texture_handler *texture_handler;
    SceneDataInfo   *scene_data_info;
    unsigned         mdl_callable_base_index;
};


//------------------------------------------------------------------------------
// vector helper functions
//------------------------------------------------------------------------------

static __host__ __device__ __inline__ float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
static __host__ __device__ __inline__ float2 operator-(const float2& a, float s)
{
    return make_float2(a.x - s, a.y - s);
}
static __host__ __device__ __inline__ float2 operator*(const float2& a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
static __host__ __device__ __inline__ float2 operator*(float s, const float2& a)
{
    return make_float2(a.x * s, a.y * s);
}

static __host__ __device__ __inline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __host__ __device__ __inline__ float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
static __host__ __device__ __inline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __host__ __device__ __inline__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static __host__ __device__ __inline__ float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __host__ __device__ __inline__ float3 operator*(float s, const float3& a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __host__ __device__ __inline__ float3 operator/(const float3& a, float s)
{
    float d = 1.0f / s;
    return make_float3(a.x * d, a.y * d, a.z * d);
}
static __host__ __device__ __inline__ void operator*=(float3& a, const float3& b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
static __host__ __device__ __inline__ void operator*=(float3& a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
static __host__ __device__ __inline__ void operator/=(float3& a, float s)
{
    float d = 1.0f / s;
    a.x *= d; a.y *= d; a.z *= d;
}

static __host__ __device__ __inline__ float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
static __host__ __device__ __inline__ float4 operator*(const float4& a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}


static __host__ __device__ __inline__ float length(const float3 &d)
{
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

static __host__ __device__ __inline__ float3 normalize(const float3 &d)
{
    const float inv_len = 1.0f / length(d);
    return make_float3(d.x * inv_len, d.y * inv_len, d.z * inv_len);
}

static __host__ __device__ __inline__ float dot(const float3 &u, const float3 &v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

static __host__ __device__ __inline__ float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static __host__ __device__ __inline__ float3 lerp(const float3 &a, const float3 &b, float t)
{
    return a + (b - a) * t;
}

static __host__ __device__ __inline__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

static __host__ __device__ __inline__ float4 make_float4(const int4 &v0)
{
    return make_float4(float(v0.x), float(v0.y), float(v0.z), float(v0.w));
}

static __host__ __device__ __inline__ int4 make_int4(const float4 &v0)
{
    return make_int4(int(v0.x), int(v0.y), int(v0.z), int(v0.w));
}


// Code only used on device
#ifdef __CUDACC__

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Per-ray-data for radiance rays
struct RadiancePRD
{
#ifndef CONTRIB_IN_PAYLOAD
    float3   contribution;
#endif
    float3   weight;
    float3   origin;
    float3   direction;
    uint32_t seed;
    float    last_pdf;
};


extern "C" __constant__ Params params;

__device__ static const float DIRAC = -1.0f;


//-------------------------------------------------------------------------------------------------
// Random number generator based on the OptiX SDK
//-------------------------------------------------------------------------------------------------
template<uint32_t N>
static __forceinline__ __device__ uint32_t tea(uint32_t v0, uint32_t v1)
{
    uint32_t s0 = 0;

    for (uint32_t n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}


// Generate random uint32_t in [0, 2^24)
static __forceinline__ __device__ uint32_t lcg(uint32_t &prev)
{
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}


// Generate random float in [0, 1)
static __forceinline__ __device__ float rnd(uint32_t &prev)
{
    return ((float)lcg(prev) / (float)0x01000000);
}


//-------------------------------------------------------------------------------------------------
// Payload access
//-------------------------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpack_pointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void pack_pointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* get_radiance_prd()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(unpack_pointer(u0, u1));
}


static __forceinline__ __device__ RayFlags get_radiance_payload_ray_flags()
{
    return static_cast<RayFlags>(optixGetPayload_2());
}


static __forceinline__ __device__ void set_radiance_payload_ray_flags(RayFlags ray_flags)
{
    optixSetPayload_2(static_cast<uint32_t>(ray_flags));
}


static __forceinline__ __device__ void set_radiance_payload_depth(int depth)
{
    RayFlags flags = get_radiance_payload_ray_flags();
    flags = RayFlags(int(flags & ~RAY_FLAGS_DEPTH_MASK) | (depth & RAY_FLAGS_DEPTH_MASK));
    set_radiance_payload_ray_flags(flags);
}


#ifdef CONTRIB_IN_PAYLOAD
static __forceinline__ __device__ float3 get_radiance_payload_contrib()
{
    return make_float3(
        int_as_float(optixGetPayload_3()),
        int_as_float(optixGetPayload_4()),
        int_as_float(optixGetPayload_5())
    );
}


static __forceinline__ __device__ void set_radiance_payload_contrib(float3 val)
{
    optixSetPayload_3(float_as_int(val.x));
    optixSetPayload_4(float_as_int(val.y));
    optixSetPayload_5(float_as_int(val.z));
}
#endif


static __forceinline__ __device__ void set_occlusion_payload(bool occluded)
{
    optixSetPayload_0(static_cast<uint32_t>(occluded));
}


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

static __device__ float3 offset_ray(const float3 &p, const float3 &n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    const int3 of_i = make_int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = make_float3(int_as_float(float_as_int(p.x) + ((p.x < 0.0f) ? -of_i.x : of_i.x)),
        int_as_float(float_as_int(p.y) + ((p.y < 0.0f) ? -of_i.y : of_i.y)),
        int_as_float(float_as_int(p.z) + ((p.z < 0.0f) ? -of_i.z : of_i.z)));

    return make_float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}


//-------------------------------------------------------------------------------------------------
// Environment sampling code
//-------------------------------------------------------------------------------------------------

// direction to environment map texture coordinates
static __forceinline__ __device__ float2 environment_coords(const float3 &dir)
{
    const float u = atan2f(dir.z, dir.x) * (float)(0.5 / M_PI) + 0.5f;
    const float v = acosf(fmaxf(fminf(-dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
    return make_float2(u, v);
}


// importance sample the environment
static __forceinline__ __device__ float3 environment_sample(
    uint32_t &seed,
    float3   &dir,
    float    &pdf)
{
    float3 xi;
    xi.x = rnd(seed);
    xi.y = rnd(seed);
    xi.z = rnd(seed);

    // importance sample an envmap pixel using an alias map
    const unsigned int size = params.env_light.size.x * params.env_light.size.y;
    const unsigned int idx = min((unsigned int)(xi.x * (float)size), size - 1);
    unsigned int env_idx;
    float xi_y = xi.y;
    if (xi_y < params.env_light.accel[idx].q) {
        env_idx = idx;
        xi_y /= params.env_light.accel[idx].q;
    } else {
        env_idx = params.env_light.accel[idx].alias;
        xi_y = (xi_y - params.env_light.accel[idx].q) / (1.0f - params.env_light.accel[idx].q);
    }

    const unsigned int py = env_idx / params.env_light.size.x;
    const unsigned int px = env_idx % params.env_light.size.x;

    // uniformly sample spherical area of pixel
    const float u = (float)(px + xi_y) / (float)params.env_light.size.x;
    const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
    float sin_phi, cos_phi;
    sincosf(phi, &sin_phi, &cos_phi);
    const float step_theta = (float)M_PI / (float)params.env_light.size.y;
    const float theta0 = (float)(py)* step_theta;
    const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
    const float theta = acosf(cos_theta);
    const float sin_theta = sinf(theta);
    dir = make_float3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

    // lookup filtered beauty
    const float v = theta * (float)(1.0 / M_PI);
    const float4 t = tex2D<float4>(params.env_light.texture, u, v);
    pdf = max(t.x, max(t.y, t.z)) * params.env_light.inv_env_integral;
    return make_float3(t.x, t.y, t.z) * params.env_light.intensity;
}


// evaluate the environment
static __forceinline__ __device__ float3 environment_eval(
    float &pdf,
    const float3 &dir)
{
    const float2 uv = environment_coords(dir);
    const float4 t = tex2DLod<float4>(params.env_light.texture, uv.x, uv.y, 0.f);
    pdf = max(t.x, max(t.y, t.z)) * params.env_light.inv_env_integral;
    return make_float3(t.x, t.y, t.z) * params.env_light.intensity;
}


// selects one light source randomly
static __forceinline__ __device__ float3 sample_lights(
    float3 const &P, float3 &to_light, float &light_dist, float &pdf, uint32_t &seed)
{
    float p_select_light = 1.0f;
    if (params.light.emission.x > 0.0f ||
            params.light.emission.y > 0.0f ||
            params.light.emission.z > 0.0f) {
        // keep it simple and use either point light or environment light, each with the same
        // probability. If the environment factor is zero, we always use the point light
        // Note: see also miss shader
        float p_select_light = params.env_light.intensity > 0.0f ? 0.5f : 1.0f;

        // in general, you would select the light depending on the importance of it
        // e.g. by incorporating their luminance

        // randomly select one of the lights
        if (rnd(seed) <= p_select_light) {
            pdf = DIRAC; // infinity

            // compute light direction and distance
            const float z1 = rnd(seed);
            const float z2 = rnd(seed);

            ParallelogramLight light = params.light;
            const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

            to_light = light_pos - P;
            const float light_dist2 = dot(to_light, to_light);
            light_dist = sqrt(light_dist2);
            to_light /= light_dist;

            const float A = length(cross(light.v1, light.v2));

            return A * light.emission / light_dist2 * 0.25f / M_PI / p_select_light;
        }

        // probability to select the environment instead
        p_select_light = (1.0f - p_select_light);
    }

    // light from the environment
    float3 radiance = environment_sample(seed, to_light, pdf);

    pdf *= p_select_light;
    light_dist = 1e16f;
    return radiance / pdf;
}

#endif  // __CUDACC__

#endif // OPTIX7_MDL_H
