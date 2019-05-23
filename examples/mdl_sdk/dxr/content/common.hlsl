/******************************************************************************
 * Copyright 2019 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

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

    // environment light
    float environment_intensity_factor;

    // gamma correction while rendering to the frame buffer
    float output_gamma_correction;
}

// Ray typed, has to match with CPU version
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW,

    RAY_TYPE_COUNT,
    RAY_TYPE_FORCE_32_BIT = 0xffffffffU
};

enum RadianceHitInfoFlags
{
    FLAG_NONE = 0,
    FLAG_INSIDE = 1,
    FLAG_DONE = 2,
};

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
    float last_pdf;
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

// standard vertex format for this example
struct Vertex
{
    float3 position;
    float3 normal;
    float2 texcoord0;
    float4 tangent0;
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

//-------------------------------------------------------------------------------------------------
// Environment
//-------------------------------------------------------------------------------------------------

struct Environment_sample_data
{
    uint alias;
    float q;
    float pdf;
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

    // get pdf
    uint width, height;
    lat_long_tex.GetDimensions(width, height);
    const uint x = min(uint(u * float(width)), width - 1);
    const uint y = min(uint(v * float(height)), height - 1);
    pdf = sample_buffer[y * width + x].pdf;

    // get radiance
    return environment_intensity_factor * lat_long_tex.SampleLevel(
        sampler_latlong, float2(u, v), /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0).xyz;
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
    pdf = sample_buffer[env_idx].pdf;

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

    // lookup filtered value
    const float v = theta * M_ONE_OVER_PI;
    return environment_intensity_factor * lat_long_tex.SampleLevel(
        sampler_latlong, float2(u, v), /*mipmaplevel=*/ 0.0f, /*mipoffset=*/0).xyz;
}

//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//-------------------------------------------------------------------------------------------------

float3 offset_ray(float3 p, float3 n)
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
