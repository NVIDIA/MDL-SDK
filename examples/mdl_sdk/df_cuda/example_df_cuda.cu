/******************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "example_df_cuda.h"
#include "texture_support_cuda.h"

// To reuse this sample code for the MDL SDK and MDL Core the corresponding namespaces are used.

// when this CUDA code is used in the context of an SDK sample.
#if defined(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR)
    #define BSDF_USE_MATERIAL_IOR MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR
    using namespace mi::neuraylib;
// when this CUDA code is used in the context of an Core sample.
#elif defined(MDL_CORE_BSDF_USE_MATERIAL_IOR)
    #define BSDF_USE_MATERIAL_IOR MDL_CORE_BSDF_USE_MATERIAL_IOR
    using namespace mi::mdl;
#endif

// for LPE support there different options for the renderer, for CUDA a renderer provided buffer
// can be used to retrieve the contributions of the individual handles (named lobes)
// Note, a real renderer would go for one specific option only
#define DF_HSM_POINTER -2
#define DF_HSM_NONE    -1
#define DF_HSM_FIXED_1  1
#define DF_HSM_FIXED_2  2
#define DF_HSM_FIXED_4  4
#define DF_HSM_FIXED_8  8
// this is the one that is used, 
// Note, this has to match with code backend option "df_handle_slot_mode"
#define DF_HANDLE_SLOTS DF_HSM_POINTER

// If enabled, math::DX(state::texture_coordinates(0).xy) = float2(1, 0) and
// math::DY(state::texture_coordinates(0).xy) = float2(0, 1) will be used.
// #define USE_FAKE_DERIVATIVES

#ifdef ENABLE_DERIVATIVES
typedef Material_expr_function_with_derivs                  Mat_expr_func;
typedef Bsdf_init_function_with_derivs                      Bsdf_init_func;
typedef Bsdf_sample_function_with_derivs                    Bsdf_sample_func;
typedef Bsdf_evaluate_function_with_derivs                  Bsdf_evaluate_func;
typedef Bsdf_pdf_function_with_derivs                       Bsdf_pdf_func;
typedef Bsdf_auxiliary_function_with_derivs                 Bsdf_auxiliary_func;
typedef Edf_init_function_with_derivs                       Edf_init_func;
typedef Edf_sample_function_with_derivs                     Edf_sample_func;
typedef Edf_evaluate_function_with_derivs                   Edf_evaluate_func;
typedef Edf_pdf_function_with_derivs                        Edf_pdf_func;
typedef Edf_auxiliary_function_with_derivs                  Edf_auxiliary_func;
typedef Shading_state_material_with_derivs                  Mdl_state;
typedef Texture_handler_deriv                               Tex_handler;
#define TEX_VTABLE                                          tex_deriv_vtable
#else
typedef Material_expr_function                              Mat_expr_func;
typedef Bsdf_init_function                                  Bsdf_init_func;
typedef Bsdf_sample_function                                Bsdf_sample_func;
typedef Bsdf_evaluate_function                              Bsdf_evaluate_func;
typedef Bsdf_pdf_function                                   Bsdf_pdf_func;
typedef Bsdf_auxiliary_function                             Bsdf_auxiliary_func;
typedef Edf_init_function                                   Edf_init_func;
typedef Edf_sample_function                                 Edf_sample_func;
typedef Edf_evaluate_function                               Edf_evaluate_func;
typedef Edf_pdf_function                                    Edf_pdf_func;
typedef Edf_auxiliary_function                              Edf_auxiliary_func;
typedef Shading_state_material                              Mdl_state;
typedef Texture_handler                                     Tex_handler;
#define TEX_VTABLE                                          tex_vtable
#endif

// Custom structure representing the resources used by the generated code of a target code object.
struct Target_code_data
{
    size_t       num_textures;      // number of elements in the textures field
    Texture      *textures;         // a list of Texture objects, if used

    size_t       num_mbsdfs;        // number of elements in the mbsdfs field
    Mbsdf        *mbsdfs;           // a list of mbsdfs objects, if used

    size_t       num_lightprofiles; // number of elements in the lightprofiles field
    Lightprofile *lightprofiles;    // a list of lightprofiles objects, if used

    char const   *ro_data_segment;  // the read-only data segment, if used
};


// all function types
union Mdl_function_ptr
{
    Mat_expr_func           *expression;
    Bsdf_init_func          *bsdf_init;
    Bsdf_sample_func        *bsdf_sample;
    Bsdf_evaluate_func      *bsdf_evaluate;
    Bsdf_pdf_func           *bsdf_pdf;
    Bsdf_auxiliary_func     *bsdf_auxiliary;
    Edf_init_func           *edf_init;
    Edf_sample_func         *edf_sample;
    Edf_evaluate_func       *edf_evaluate;
    Edf_pdf_func            *edf_pdf;
    Edf_auxiliary_func      *edf_auxiliary;
};

// function index offset depending on the target code
extern __constant__ unsigned int     mdl_target_code_offsets[];

// number of generated functions
extern __constant__ unsigned int     mdl_functions_count;

// the following arrays are indexed by an mdl_function_index
extern __constant__ Mdl_function_ptr mdl_functions[];
extern __constant__ unsigned int     mdl_arg_block_indices[];

// Identity matrix.
// The last row is always implied to be (0, 0, 0, 1).
__constant__ const float4 identity[3] = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f}
};

// the material provides pairs for each generated function to evaluate
// the functions and arg blocks array are indexed by:
// mdl_target_code_offsets[target_code_index] + function_index
typedef uint3 Mdl_function_index;
__device__ inline Mdl_function_index get_mdl_function_index(const uint2& index_pair)
{
    return make_uint3(
        index_pair.x,   // target_code_index
        index_pair.y,   // function_index inside target code
        mdl_target_code_offsets[index_pair.x] + index_pair.y); // global function index
}

// resource handler for accessing textures and other data
// depends on the target code (link unit)
struct Mdl_resource_handler
{
    __device__ Mdl_resource_handler()
    {
        m_tex_handler.vtable = &TEX_VTABLE;   // only required in 'vtable' mode, otherwise NULL
        data.shared_data = NULL;
        data.texture_handler = reinterpret_cast<Texture_handler_base *>(&m_tex_handler);
    }

    // reuse the handler with a different target code index
    __device__ inline void set_target_code_index(
        const Kernel_params& params, const Mdl_function_index& index)
    {
        m_tex_handler.num_textures = params.tc_data[index.x].num_textures;
        m_tex_handler.textures = params.tc_data[index.x].textures;
        m_tex_handler.num_mbsdfs = params.tc_data[index.x].num_mbsdfs;
        m_tex_handler.mbsdfs = params.tc_data[index.x].mbsdfs;
        m_tex_handler.num_lightprofiles = params.tc_data[index.x].num_lightprofiles;
        m_tex_handler.lightprofiles = params.tc_data[index.x].lightprofiles;
    }

    // a pointer to this data is passed to all generated functions
    Resource_data data;

private:
    Tex_handler m_tex_handler;
};


// checks if the indexed function can be evaluated or not
__device__ inline bool is_valid(const Mdl_function_index& index)
{
    return index.y != 0xFFFFFFFFu;
}

// get a pointer to the material parameters which is passed to all generated functions
__device__ inline const char* get_arg_block(
    const Kernel_params& params,
    const Mdl_function_index& index)
{
    return params.arg_block_list[mdl_arg_block_indices[index.z]];
}

// Init function
__device__ inline Bsdf_init_func* as_init(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].bsdf_init;
}

// Expression functions
__device__ inline Mat_expr_func* as_expression(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].expression;
}

// BSDF functions
__device__ inline Bsdf_sample_func* as_bsdf_sample(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].bsdf_sample;
}

__device__ inline Bsdf_evaluate_func* as_bsdf_evaluate(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 1].bsdf_evaluate;
}

__device__ inline Bsdf_pdf_func* as_bsdf_pdf(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 2].bsdf_pdf;
}

__device__ inline Bsdf_auxiliary_func* as_bsdf_auxiliary(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 3].bsdf_auxiliary;
}

// EDF functions
__device__ inline Edf_sample_func* as_edf_sample(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].edf_sample;
}

__device__ inline Edf_evaluate_func* as_edf_evaluate(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 1].edf_evaluate;
}

__device__ inline Edf_pdf_func* as_edf_pdf(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 2].edf_pdf;
}

__device__ inline Edf_auxiliary_func* as_edf_auxiliary(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 3].edf_auxiliary;
}


// 3d vector math utilities
__device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ inline float3 operator*(const float3& a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float3 operator/(const float3& a, const float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ inline void operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
__device__ inline void operator-=(float3& a, const float3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
__device__ inline void operator*=(float3& a, const float3& b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
__device__ inline void operator*=(float3& a, const float& s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
__device__ inline float squared_length(const float3 &d)
{
    return d.x * d.x + d.y * d.y + d.z * d.z;
}
__device__ inline float3 normalize(const float3 &d)
{
    const float inv_len = 1.0f / sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    return make_float3(d.x * inv_len, d.y * inv_len, d.z * inv_len);
}
__device__ inline float dot(const float3 &u, const float3 &v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
__device__ inline float3 cross(const float3 &u, const float3 &v)
{
    return make_float3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x);
}

typedef curandStatePhilox4_32_10_t Rand_state;

// direction to environment map texture coordinates
__device__ inline float2 environment_coords(const float3 &dir, const Kernel_params& params)
{
    const float u = atan2f(dir.z, dir.x) * (float)(0.5 / M_PI) + 0.5f;
    const float v = acosf(fmax(fminf(-dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
    return make_float2(fmodf(u + params.env_rotation * 0.5f * M_1_PI, 1.0f), v);
}

// importance sample the environment
__device__ inline float3 environment_sample(
    float3 &dir,
    float  &pdf,
    const  float3 &xi,
    const  Kernel_params &params)
{
    // importance sample an envmap pixel using an alias map
    const unsigned int size = params.env_size.x * params.env_size.y;
    const unsigned int idx = min((unsigned int)(xi.x * (float)size), size - 1);
    unsigned int env_idx;
    float xi_y = xi.y;
    if (xi_y < params.env_accel[idx].q) {
        env_idx = idx ;
        xi_y /= params.env_accel[idx].q;
    } else {
        env_idx = params.env_accel[idx].alias;
        xi_y = (xi_y - params.env_accel[idx].q) / (1.0f - params.env_accel[idx].q);
    }

    const unsigned int py = env_idx / params.env_size.x;
    const unsigned int px = env_idx % params.env_size.x;
    pdf = params.env_accel[env_idx].pdf;

    // uniformly sample spherical area of pixel
    const float u = (float)(px + xi_y) / (float)params.env_size.x;
    const float phi = u * (float)(2.0 * M_PI) - (float)M_PI - params.env_rotation;
    float sin_phi, cos_phi;
    sincosf(phi > float(-M_PI) ? phi : (phi + (float)(2.0 * M_PI)), &sin_phi, &cos_phi);
    const float step_theta = (float)M_PI / (float)params.env_size.y;
    const float theta0 = (float)(py) * step_theta;
    const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
    const float theta = acosf(cos_theta);
    const float sin_theta = sinf(theta);
    dir = make_float3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

    // lookup filtered beauty
    const float v = theta * (float)(1.0 / M_PI);
    const float4 t = tex2D<float4>(params.env_tex, u, v);
    return make_float3(t.x, t.y, t.z) * params.env_intensity / pdf;
}

// evaluate the environment
__device__ inline float3 environment_eval(
    float &pdf,
    const float3 &dir,
    const Kernel_params &params)
{
    const float2 uv = environment_coords(dir, params);
    const unsigned int x =
        min((unsigned int)(uv.x * (float)params.env_size.x), params.env_size.x - 1);
    const unsigned int y =
        min((unsigned int)(uv.y * (float)params.env_size.y), params.env_size.y - 1);

    pdf = params.env_accel[y * params.env_size.x + x].pdf;
    const float4 t = tex2D<float4>(params.env_tex, uv.x, uv.y) ;
    return make_float3(t.x, t.y, t.z) * params.env_intensity;
}

//-------------------------------------------------------------------------------------------------

struct auxiliary_data
{
    float3 albedo;
    float3 normal;
    int num; // multiple elements can contribute to the aux buffer with equal weight 

    __device__ inline auxiliary_data& operator+=(const auxiliary_data& b)
    {
        albedo += b.albedo;
        normal += b.normal;
        num += b.num;
        return *this;
    }
};

__device__ inline static void clear(auxiliary_data& data)
{
    data.albedo = make_float3(0.0f, 0.0f, 0.0f);
    data.normal = make_float3(0.0f, 0.0f, 0.0f);
    data.num = 0;
}

__device__ inline void normalize(auxiliary_data& data)
{
    data.albedo = data.albedo / fmaxf(1.0f, float(data.num));

    if (dot(data.normal, data.normal) > 0.0f)
        data.normal = normalize(data.normal);

    data.num = min(1, data.num);
}

//-------------------------------------------------------------------------------------------------

struct Ray_state
{
    float3 contribution;
    float3 weight;
    float3 pos, pos_rx, pos_ry;
    float3 dir, dir_rx, dir_ry;
    bool inside;
    int intersection;
    uint32_t lpe_current_state;
    auxiliary_data* aux;
};


struct Ray_hit_info
{
    float distance;
    #ifdef ENABLE_DERIVATIVES
        tct_deriv_float3 position;
    #else
        float3 position;
    #endif
    float3 normal;
    float3 tangent_u;
    float3 tangent_v;
    #ifdef ENABLE_DERIVATIVES
        tct_deriv_float3 texture_coords[1];
    #else
        tct_float3 texture_coords[1];
    #endif
};


#define GT_SPHERE_RADIUS 1.0f
__device__ inline bool intersect_sphere(
    const Ray_state &ray_state,
    const Kernel_params &params,
    Ray_hit_info& out_hit)
{
    const float r = GT_SPHERE_RADIUS;
    const float b = 2.0f * dot(ray_state.dir, ray_state.pos);
    const float c = dot(ray_state.pos, ray_state.pos) - r * r;

    float tmp = b * b - 4.0f * c;
    if (tmp < 0.0f)
        return false;

    tmp = sqrtf(tmp);
    const float t0 = (((b < 0.0f) ? -tmp : tmp) - b) * 0.5f;
    const float t1 = c / t0;

    const float m = fminf(t0, t1);
    out_hit.distance = m > 0.0f ? m : fmaxf(t0, t1);
    if (out_hit.distance < 0.0f)
        return false;

    // compute geometry state
    #ifdef ENABLE_DERIVATIVES
        out_hit.position.val = ray_state.pos + ray_state.dir * out_hit.distance;
        out_hit.position.dx = make_float3(0.0f, 0.0f, 0.0f);
        out_hit.position.dy = make_float3(0.0f, 0.0f, 0.0f);
        const float3 &posval = out_hit.position.val;
    #else
        out_hit.position = ray_state.pos + ray_state.dir * out_hit.distance;
        const float3 &posval = out_hit.position;
    #endif
    out_hit.normal = normalize(posval);

    const float phi = atan2f(out_hit.normal.x, out_hit.normal.z);
    const float theta = acosf(out_hit.normal.y);

    const float3 uvw = make_float3(
        (phi * (float) (0.5 / M_PI) + 0.5f) * 2.0f,
        1.0f - theta * (float) (1.0 / M_PI),
        0.0f);

    // compute surface derivatives
    float sp, cp;
    sincosf(phi, &sp, &cp);
    const float st = sinf(theta);
    out_hit.tangent_u = make_float3(cp * st, 0.0f, -sp * st) * (float) M_PI * r;
    out_hit.tangent_v = make_float3(sp * out_hit.normal.y, -st, cp * out_hit.normal.y) * (float) (-M_PI) * r;

    #ifdef ENABLE_DERIVATIVES
        out_hit.texture_coords[0].val = uvw;
        out_hit.texture_coords[0].dx = make_float3(0.0f, 0.0f, 0.0f);
        out_hit.texture_coords[0].dy = make_float3(0.0f, 0.0f, 0.0f);
    #else
        out_hit.texture_coords[0] = uvw;
    #endif

    return true;
}

#define GT_HAIR_RADIUS 0.35f
#define GT_HAIR_LENGTH 3.0f
__device__ inline bool intersect_hair(
    const Ray_state &ray_state,
    const Kernel_params &params,
    Ray_hit_info& out_hit)
{
    const float r = GT_HAIR_RADIUS;
    const float a = ray_state.dir.x * ray_state.dir.x + ray_state.dir.z * ray_state.dir.z;
    const float b = 2.0f * (ray_state.dir.x * ray_state.pos.x + ray_state.dir.z * ray_state.pos.z);
    const float c = ray_state.pos.x * ray_state.pos.x + ray_state.pos.z * ray_state.pos.z - r * r;

    float tmp = b * b - 4.0f * a * c;
    if (tmp < 0.0f)
        return false;

    tmp = sqrtf(tmp);
    const float q = (((b < 0.0f) ? -tmp : tmp) - b) * 0.5f;
    const float t0 = q / a;
    const float t1 = c / q;

    const float m = fminf(t0, t1);
    out_hit.distance = m > 0.0f ? m : fmaxf(t0, t1);
    if (out_hit.distance < 0.0f)
        return false;

    // compute geometry state
    #ifdef ENABLE_DERIVATIVES
        out_hit.position.val = ray_state.pos + ray_state.dir * out_hit.distance;
        out_hit.position.dx = make_float3(0.0f, 0.0f, 0.0f);
        out_hit.position.dy = make_float3(0.0f, 0.0f, 0.0f);
        const float3 &posval = out_hit.position.val;
    #else
        out_hit.position = ray_state.pos + ray_state.dir * out_hit.distance;
        const float3 &posval = out_hit.position;
    #endif
    out_hit.normal = normalize(make_float3(posval.x, 0.0f, posval.z));

    if (fabsf(posval.y) > GT_HAIR_LENGTH * 0.5f)
        return false;

    const float phi = atan2f(posval.z, posval.x);

    const float3 uvw = make_float3(
        (posval.y + GT_HAIR_LENGTH * 0.5f) / GT_HAIR_LENGTH, // position along the hair
        phi * (float) (0.5f / M_PI) + 0.5f, // position around the hair in the range [0, 1]
        2.0f * GT_HAIR_RADIUS); // thickness of the hair

    // compute surface derivatives
    out_hit.tangent_u = make_float3(0.0, 1.0, 0.0);
    out_hit.tangent_v = cross(out_hit.normal, out_hit.tangent_u);

    #ifdef ENABLE_DERIVATIVES
        out_hit.texture_coords[0].val = uvw;
        out_hit.texture_coords[0].dx = make_float3(0.0f, 0.0f, 0.0f);
        out_hit.texture_coords[0].dy = make_float3(0.0f, 0.0f, 0.0f);
    #else
        out_hit.texture_coords[0] = uvw;
    #endif

    return true;
}

__device__ inline bool intersect_geometry(
    Ray_state &ray_state,
    const Kernel_params &params,
    Ray_hit_info& out_hit)
{
    switch (Geometry_type(params.geometry))
    {
        case GT_SPHERE:
            if (!intersect_sphere(ray_state, params, out_hit))
                return false;
            break;
        case GT_HAIR:
            if (!intersect_hair(ray_state, params, out_hit))
                return false;
            break;
        default:
            return false;
    }

    #ifndef ENABLE_DERIVATIVES
    ray_state.pos = out_hit.position;
    #else
    ray_state.pos = out_hit.position.val;
    if (params.use_derivatives && ray_state.intersection == 0)
    {
#ifdef USE_FAKE_DERIVATIVES
        out_hit.position.dx = make_float3(1.0f, 0.0f, 0.0f);
        out_hit.position.dy = make_float3(0.0f, 1.0f, 0.0f);
        out_hit.texture_coords[0].dx = make_float3(1.0f, 0.0f, 0.0f);
        out_hit.texture_coords[0].dy = make_float3(0.0f, 1.0f, 0.0f);
#else
        // compute ray differential for one-pixel offset rays
        // ("Physically Based Rendering", 3rd edition, chapter 10.1.1)
        const float d = dot(out_hit.normal, ray_state.pos);
        const float tx = (d - dot(out_hit.normal, ray_state.pos_rx)) / dot(out_hit.normal, ray_state.dir_rx);
        const float ty = (d - dot(out_hit.normal, ray_state.pos_ry)) / dot(out_hit.normal, ray_state.dir_ry);
        ray_state.pos_rx += ray_state.dir_rx * tx;
        ray_state.pos_ry += ray_state.dir_ry * ty;

        out_hit.position.dx = ray_state.pos_rx - ray_state.pos;
        out_hit.position.dy = ray_state.pos_ry - ray_state.pos;

        float4 A;
        float2 B_x, B_y;
        if (fabsf(out_hit.normal.x) > fabsf(out_hit.normal.y) && fabsf(out_hit.normal.x) > fabsf(out_hit.normal.z))
        {
            B_x = make_float2(
                ray_state.pos_rx.y - ray_state.pos.y,
                ray_state.pos_rx.z - ray_state.pos.z);
            B_y = make_float2(
                ray_state.pos_ry.y - ray_state.pos.y,
                ray_state.pos_ry.z - ray_state.pos.z);
            A = make_float4(
                out_hit.tangent_u.y, out_hit.tangent_u.z, out_hit.tangent_v.y, out_hit.tangent_v.z);
        }
        else if (fabsf(out_hit.normal.y) > fabsf(out_hit.normal.z))
        {
            B_x = make_float2(
                ray_state.pos_rx.x - ray_state.pos.x,
                ray_state.pos_rx.z - ray_state.pos.z);
            B_y = make_float2(
                ray_state.pos_ry.x - ray_state.pos.x,
                ray_state.pos_ry.z - ray_state.pos.z);
            A = make_float4(
                out_hit.tangent_u.x, out_hit.tangent_u.z, out_hit.tangent_v.x, out_hit.tangent_v.z);
        }
        else
        {
            B_x = make_float2(
                ray_state.pos_rx.x - ray_state.pos.x,
                ray_state.pos_rx.y - ray_state.pos.y);
            B_y = make_float2(
                ray_state.pos_ry.x - ray_state.pos.x,
                ray_state.pos_ry.y - ray_state.pos.y);
            A = make_float4(
                out_hit.tangent_u.x, out_hit.tangent_u.y, out_hit.tangent_v.x, out_hit.tangent_v.y);
        }

        const float det = A.x * A.w - A.y * A.z;
        if (fabsf(det) > 1e-10f)
        {
            const float inv_det = 1.0f / det;

            out_hit.texture_coords[0].dx.x = inv_det * (A.w * B_x.x - A.z * B_x.y);
            out_hit.texture_coords[0].dx.y = inv_det * (A.x * B_x.y - A.y * B_x.x);

            out_hit.texture_coords[0].dy.x = inv_det * (A.w * B_y.x - A.z * B_y.y);
            out_hit.texture_coords[0].dy.y = inv_det * (A.x * B_y.y - A.y * B_y.x);
        }
#endif
    }
    #endif

    out_hit.tangent_u = normalize(out_hit.tangent_u);
    out_hit.tangent_v = normalize(out_hit.tangent_v);
    return true;
}


__device__ bool cull_point_light(
    const Kernel_params &params,
    const float3 &light_position,
    const float3 &light_direction /*to light*/,
    const float3 &normal)
{
    switch (params.geometry)
    {
        case GT_SPHERE:
        {
            // same as default, but allow lights inside the sphere
            const float inside = (squared_length(light_position) < GT_SPHERE_RADIUS) ? -1.f : 1.f;
            return (dot(light_direction, normal) * inside) <= 0.0f;
        }
        case GT_HAIR:
            // ignore light sources within the volume
            return (light_position.x * light_position.x +
                    light_position.z * light_position.z) < GT_SPHERE_RADIUS;
        default:
            // ignore light from behind
            return dot(light_direction, normal) <= 0.0f;
    }
}

__device__ bool cull_env_light(
    const Kernel_params &params,
    const float3 &light_direction /*to light*/,
    const float3 &normal)
{
    switch (params.geometry)
    {
        case GT_HAIR:
            // allow light from behind
            return false;

        case GT_SPHERE:
        default:
            // ignore light from behind
            return dot(light_direction, normal) <= 0.0f;
    }
}


__device__ void continue_ray(
    Ray_state& ray_state,
    const Ray_hit_info &hit_indo,
    unsigned int event_type,
    const Kernel_params &params)
{
    switch (params.geometry)
    {

    case GT_HAIR:
        if (event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)
        {
            // conservative
            ray_state.pos += ray_state.dir * 2.0f * GT_HAIR_RADIUS;
            ray_state.inside = false;
        }
        break;

    default:
        return;
    }
}

//-------------------------------------------------------------------------------------------------

// events that are define a transition between states, along with tag IDs
enum Transition_type
{
    TRANSITION_CAMERA = 0,
    TRANSITION_LIGHT,
    TRANSITION_EMISSION,
    TRANSITION_SCATTER_DR,
    TRANSITION_SCATTER_DT,
    TRANSITION_SCATTER_GR,
    TRANSITION_SCATTER_GT,
    TRANSITION_SCATTER_SR,
    TRANSITION_SCATTER_ST,

    TRANSITION_COUNT,
};

// go to the next state, given the current state and a transition token.
__device__ inline uint32_t lpe_transition(
    uint32_t current_state,
    Transition_type event,
    uint32_t global_tag_id,
    const Kernel_params &params)
{
    if(current_state == static_cast<uint32_t>(-1))
        return static_cast<uint32_t>(-1);

    return params.lpe_state_table[
        current_state * params.lpe_num_transitions +
        static_cast<uint32_t>(TRANSITION_COUNT) * global_tag_id +
        static_cast<uint32_t>(event)];
}

// add direct contribution, e.g., for emission, direct light hits
__device__ inline void accumulate_contribution(
    Transition_type light_event,
    uint32_t light_global_tag_id,
    const float3& contrib,
    Ray_state &ray_state,
    const Kernel_params &params)
{
    // check if there is a valid transition to that light source
    uint32_t next_state = lpe_transition(
        ray_state.lpe_current_state, light_event, light_global_tag_id, params);
    if (next_state == static_cast<uint32_t>(-1)) return;

    // add contribution the when the reached state is a final state for the selected LPE
    // here we only have one LPE buffer, but more can be added easily by checking different LPEs
    if ((params.lpe_final_mask[next_state] & (1 << params.lpe_ouput_expression)) != 0)
        ray_state.contribution += contrib;
}

// add contribution for next event estimations
__device__ inline void accumulate_next_event_contribution(
    Transition_type scatter_event, uint32_t material_global_tag_id,
    Transition_type light_event, uint32_t light_global_tag_id,
    const float3& contrib,
    Ray_state &ray_state,
    const Kernel_params &params)
{
    // transition following the scatter event
    uint32_t next_state = lpe_transition(
        ray_state.lpe_current_state, scatter_event, material_global_tag_id, params);
    if (next_state == static_cast<uint32_t>(-1)) return;

    // check if there is a valid transition to the light source
    next_state = lpe_transition(
        next_state, light_event, light_global_tag_id, params);
    if (next_state == static_cast<uint32_t>(-1)) return;

    // add contribution the when the reached state is a final state for the selected LPE
    // here we only have one LPE buffer, but more can be added easily by checking different LPEs
    if ((params.lpe_final_mask[next_state] & (1 << params.lpe_ouput_expression)) != 0)
        ray_state.contribution += contrib;
}

//-------------------------------------------------------------------------------------------------


__device__ inline bool trace_scene(
    Rand_state &rand_state,
    Ray_state &ray_state,
    const Kernel_params &params)
{
    // stop at invalid states
    if (ray_state.lpe_current_state == static_cast<uint32_t>(-1))
        return false;

    // intersect with geometry
    Ray_hit_info hit;
    if (!intersect_geometry(ray_state, params, hit)) {
        if (ray_state.intersection == 0 && params.mdl_test_type != MDL_TEST_NO_ENV) {
            // primary ray miss, add environment contribution
            const float2 uv = environment_coords(ray_state.dir, params);
            const float4 texval = tex2D<float4>(params.env_tex, uv.x, uv.y);

            // add contribution, if `CL` is a valid path
            accumulate_contribution(
                TRANSITION_LIGHT, params.env_gtag /* light group 'env' */,
                make_float3(texval.x, texval.y, texval.z) * params.env_intensity,
                ray_state, params);
        }
        return false;
    }

    float4 texture_results[16];

    // material of the current object
    Df_cuda_material material = params.material_buffer[params.current_material];

    Mdl_function_index func_idx;
    func_idx = get_mdl_function_index(material.init);
    if (!is_valid(func_idx))
        return false;

    // create state
    Mdl_state state = {
        hit.normal,
        hit.normal,
        hit.position,
        0.0f,
        hit.texture_coords,
        &hit.tangent_u,
        &hit.tangent_v,
        texture_results,
        params.tc_data[func_idx.x].ro_data_segment,
        identity,
        identity,
        0,
        1.0f
    };

    // access textures and other resource data
    // expect that the target code index is the same for all functions of a material
    Mdl_resource_handler mdl_resources;
    mdl_resources.set_target_code_index(params, func_idx);    // init resource handler

    const char* arg_block = get_arg_block(params, func_idx);  // get material parameters

    // initialize the state
    as_init(func_idx)(&state, &mdl_resources.data, NULL, arg_block);

    // for evaluating parts of the BSDF individually, e.g. for implementing LPEs
    // the MDL SDK provides several options to pass out the BSDF, EDF, and auxiliary data
    #if DF_HANDLE_SLOTS == DF_HSM_POINTER
        // application provided memory
        // the data structs will get only a pointer to a buffer, along with size and offset 
        const unsigned df_eval_slots = 4;       // number of handles (parts) that can be evaluated
                                                // at once. 4 is an arbitrary choice. However, it 
                                                // has to match eval_data.handle_count and 
                                                // aux_data.handle_count)

        float3 result_buffer_0[df_eval_slots];  // used for bsdf_diffuse, edf, and albedo
        float3 result_buffer_1[df_eval_slots];  // used for bsdf_specular and normal
    #elif DF_HANDLE_SLOTS == DF_HSM_NONE
        // handles are ignored, all parts of the BSDF are returned at once without loops (fastest)
        const unsigned df_eval_slots = 1;
    #else
        // eval_data and auxiliary_data have a fixed size array to pass the data. Only an offset
        // is required if there are more handles (parts) than slots.
        const unsigned df_eval_slots = DF_HANDLE_SLOTS;
    #endif


    // apply volume attenuation after first bounce
    // (assuming uniform absorption coefficient and ignoring scattering coefficient)
    if (ray_state.intersection > 0)
    {
        func_idx = get_mdl_function_index(material.volume_absorption);
        if (is_valid(func_idx)) {
            float3 abs_coeff;
            as_expression(func_idx)(
                &abs_coeff, &state, &mdl_resources.data, NULL, arg_block);

            ray_state.weight.x *= abs_coeff.x > 0.0f ? expf(-abs_coeff.x * hit.distance) : 1.0f;
            ray_state.weight.y *= abs_coeff.y > 0.0f ? expf(-abs_coeff.y * hit.distance) : 1.0f;
            ray_state.weight.z *= abs_coeff.z > 0.0f ? expf(-abs_coeff.z * hit.distance) : 1.0f;
        }
    }

    // add emission
    func_idx = get_mdl_function_index(material.edf);
    if (is_valid(func_idx))
    {
        // evaluate intensity expression
        float3 emission_intensity = make_float3(0.0, 0.0, 0.0);
        Mdl_function_index intensity_func_idx = get_mdl_function_index(material.emission_intensity);
        if (is_valid(intensity_func_idx))
        {
            as_expression(intensity_func_idx)(
                &emission_intensity, &state, &mdl_resources.data, NULL, arg_block);
        }

        // evaluate EDF
        Edf_evaluate_data<(Df_handle_slot_mode) DF_HANDLE_SLOTS> eval_data;
        eval_data.k1 = make_float3(-ray_state.dir.x, -ray_state.dir.y, -ray_state.dir.z);

        #if DF_HANDLE_SLOTS == DF_HSM_POINTER
            eval_data.edf = result_buffer_0;
            eval_data.handle_count = df_eval_slots;
        #endif

        // outer loop in case the are more material tags than slots in the evaluate struct
        unsigned offset = 0;
        #if DF_HANDLE_SLOTS != DF_HSM_NONE
        for (; offset < material.edf_mtag_to_gtag_map_size; offset += df_eval_slots)
        {
            eval_data.handle_offset = offset;
        #endif

            // evaluate the materials EDF
            as_edf_evaluate(func_idx)(&eval_data, &state, &mdl_resources.data, NULL, arg_block);

            // iterate over all lobes (tags that appear in the df)
            for (unsigned lobe = 0; (lobe < df_eval_slots) &&
                ((offset + lobe) < material.edf_mtag_to_gtag_map_size); ++lobe)
            {
                // add emission contribution
                accumulate_contribution(
                    TRANSITION_EMISSION, material.edf_mtag_to_gtag_map[offset + lobe],
                    #if DF_HANDLE_SLOTS == DF_HSM_NONE
                        eval_data.edf * emission_intensity,
                    #else
                        eval_data.edf[lobe] * emission_intensity,
                    #endif
                    ray_state, params);
            }

        #if DF_HANDLE_SLOTS != DF_HSM_NONE
        }
        #endif
    }


    func_idx = get_mdl_function_index(material.bsdf);
    if (is_valid(func_idx))
    {
        // reuse memory for function data
        union
        {
            Bsdf_sample_data                                            sample_data;
            Bsdf_evaluate_data<(Df_handle_slot_mode)DF_HANDLE_SLOTS>    eval_data;
            Bsdf_pdf_data                                               pdf_data;
            Bsdf_auxiliary_data<(Df_handle_slot_mode)DF_HANDLE_SLOTS>   aux_data;
        };

        // for thin_walled materials there is no 'inside'
        bool thin_walled = false;
        Mdl_function_index thin_walled_func_idx = get_mdl_function_index(material.thin_walled);
        if (is_valid(thin_walled_func_idx))
            as_expression(thin_walled_func_idx)(
                &thin_walled, &state, &mdl_resources.data, NULL, arg_block);

        // initialize shared fields
        if (ray_state.inside && !thin_walled)
        {
            sample_data.ior1.x = BSDF_USE_MATERIAL_IOR;
            sample_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
        }
        else
        {
            sample_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
            sample_data.ior2.x = BSDF_USE_MATERIAL_IOR;
        }
        sample_data.k1 = make_float3(-ray_state.dir.x, -ray_state.dir.y, -ray_state.dir.z);

        // if requested, fill auxiliary buffers
        if (params.enable_auxiliary_output && ray_state.intersection == 0)
        {
            #if DF_HANDLE_SLOTS == DF_HSM_POINTER
                aux_data.albedo = result_buffer_0;
                aux_data.normal = result_buffer_1;
                aux_data.handle_count = df_eval_slots;
            #endif

            // outer loop in case the are more material tags than slots in the evaluate struct
            unsigned offset = 0;
            #if DF_HANDLE_SLOTS != DF_HSM_NONE
            for (; offset < material.bsdf_mtag_to_gtag_map_size; offset += df_eval_slots)
            {
                aux_data.handle_offset = offset;
            #endif

                // evaluate the materials auxiliary
                as_bsdf_auxiliary(func_idx)(&aux_data, &state, &mdl_resources.data, NULL, arg_block);

                // iterate over all lobes (tags that appear in the df)
                for (unsigned lobe = 0; (lobe < df_eval_slots) &&
                    ((offset + lobe) < material.bsdf_mtag_to_gtag_map_size); ++lobe)
                {
                    // to keep it simpler, the individual albedo and normals are averaged
                    // however, the parts can also be used separately, e.g. for LPEs
                    #if DF_HANDLE_SLOTS == DF_HSM_NONE
                        ray_state.aux->albedo += aux_data.albedo;
                        ray_state.aux->normal += aux_data.normal;
                    #else
                        ray_state.aux->albedo += aux_data.albedo[lobe];
                        ray_state.aux->normal += aux_data.normal[lobe];
                    #endif
                    ray_state.aux->num++;
                }

            #if DF_HANDLE_SLOTS != DF_HSM_NONE
            }
            #endif
        }

        // compute direct lighting for point light
        Transition_type transition_glossy, transition_diffuse;
        if (params.light_intensity > 0.0f)
        {
            float3 to_light = params.light_pos - ray_state.pos;
            if(!cull_point_light(params, params.light_pos, to_light, hit.normal))
            {
                const float inv_squared_dist = 1.0f / squared_length(to_light);
                const float3 f = params.light_color * params.light_intensity *
                                 inv_squared_dist * (float) (0.25 / M_PI);

                eval_data.k2 = to_light * sqrtf(inv_squared_dist);
                #if DF_HANDLE_SLOTS == DF_HSM_POINTER
                    eval_data.bsdf_diffuse = result_buffer_0;
                    eval_data.bsdf_glossy = result_buffer_1;
                    eval_data.handle_count = df_eval_slots;
                #endif

                // outer loop in case the are more material tags than slots in the evaluate struct
                unsigned offset = 0;
                #if DF_HANDLE_SLOTS != DF_HSM_NONE
                for (; offset < material.bsdf_mtag_to_gtag_map_size; offset += df_eval_slots)
                {
                    eval_data.handle_offset = offset;
                #endif

                    // evaluate the materials BSDF
                    as_bsdf_evaluate(func_idx)(
                        &eval_data, &state, &mdl_resources.data, NULL, arg_block);

                    // we know if we reflect or transmit
                    if (dot(to_light, hit.normal) > 0.0f) {
                        transition_glossy = TRANSITION_SCATTER_GR;
                        transition_diffuse = TRANSITION_SCATTER_DR;
                    } else {
                        transition_glossy = TRANSITION_SCATTER_GT;
                        transition_diffuse = TRANSITION_SCATTER_DT;
                    }

                    // sample weight
                    const float3 w = ray_state.weight * f;

                    // iterate over all lobes (tags that appear in the df)
                    for (unsigned lobe = 0; (lobe < df_eval_slots) &&
                         ((offset + lobe) < material.bsdf_mtag_to_gtag_map_size); ++lobe)
                    {
                        // get the `global tag` of the lobe
                        unsigned material_lobe_gtag = material.bsdf_mtag_to_gtag_map[offset + lobe];

                        // add diffuse contribution
                        accumulate_next_event_contribution(
                            transition_diffuse, material_lobe_gtag,
                            TRANSITION_LIGHT, params.point_light_gtag, // light group
                            #if DF_HANDLE_SLOTS == DF_HSM_NONE
                                eval_data.bsdf_diffuse * w,
                            #else
                                eval_data.bsdf_diffuse[lobe] * w,
                            #endif
                            ray_state, params);

                        // add glossy contribution
                        accumulate_next_event_contribution(
                            transition_glossy, material_lobe_gtag,
                            TRANSITION_LIGHT, params.point_light_gtag, // light group
                            #if DF_HANDLE_SLOTS == DF_HSM_NONE
                                eval_data.bsdf_glossy * w,
                            #else
                                eval_data.bsdf_glossy[lobe] * w,
                            #endif
                            ray_state, params);
                    }

                #if DF_HANDLE_SLOTS != DF_HSM_NONE
                }
                #endif
            }
        }

        // importance sample environment light
        if (params.mdl_test_type != MDL_TEST_SAMPLE && params.mdl_test_type != MDL_TEST_NO_ENV)
        {
            const float xi0 = curand_uniform(&rand_state);
            const float xi1 = curand_uniform(&rand_state);
            const float xi2 = curand_uniform(&rand_state);

            float3 light_dir;
            float pdf;
            const float3 f = environment_sample(light_dir, pdf, make_float3(xi0, xi1, xi2), params);

            if (!cull_env_light(params, light_dir, hit.normal) && pdf > 0.0f)
            {
                eval_data.k2 = light_dir;

                #if DF_HANDLE_SLOTS == DF_HSM_POINTER
                    eval_data.bsdf_diffuse = result_buffer_0;
                    eval_data.bsdf_glossy = result_buffer_1;
                    eval_data.handle_count = df_eval_slots;
                #endif

                // outer loop in case the are more material tags than slots in the evaluate struct
                unsigned offset = 0;
                #if DF_HANDLE_SLOTS != DF_HSM_NONE
                for (; offset < material.bsdf_mtag_to_gtag_map_size; offset += df_eval_slots)
                {
                    eval_data.handle_offset = offset;
                #endif

                    // evaluate the materials BSDF
                    as_bsdf_evaluate(func_idx)(
                        &eval_data, &state, &mdl_resources.data, NULL, arg_block);

                    const float mis_weight =
                        (params.mdl_test_type == MDL_TEST_EVAL) ? 1.0f : pdf / (pdf + eval_data.pdf);

                    // we know if we reflect or transmit
                    if (dot(light_dir, hit.normal) > 0.0f) {
                        transition_glossy = TRANSITION_SCATTER_GR;
                        transition_diffuse = TRANSITION_SCATTER_DR;
                    } else {
                        transition_glossy = TRANSITION_SCATTER_GT;
                        transition_diffuse = TRANSITION_SCATTER_DT;
                    }

                    // sample weight
                    const float3 w = ray_state.weight * f * mis_weight;

                    // iterate over all lobes (tags that appear in the df)
                    for (unsigned lobe = 0; (lobe < df_eval_slots) &&
                        ((offset + lobe) < material.bsdf_mtag_to_gtag_map_size); ++lobe)
                    {
                        // get the `global tag` of the lobe
                        unsigned material_lobe_gtag = material.bsdf_mtag_to_gtag_map[offset + lobe];

                        // add diffuse contribution
                        accumulate_next_event_contribution(
                            transition_diffuse, material_lobe_gtag,
                            TRANSITION_LIGHT, params.env_gtag, // light group 'env'
                            #if DF_HANDLE_SLOTS == DF_HSM_NONE
                                (eval_data.bsdf - eval_data.bsdf_glossy) * w,
                            #else
                                eval_data.bsdf_diffuse[lobe] * w,
                            #endif
                            ray_state, params);

                        // add glossy contribution
                        accumulate_next_event_contribution(
                            transition_glossy, material_lobe_gtag,
                            TRANSITION_LIGHT, params.env_gtag, // light group 'env'
                            #if DF_HANDLE_SLOTS == DF_HSM_NONE
                                eval_data.bsdf_glossy * w,
                            #else
                                eval_data.bsdf_glossy[lobe] * w,
                            #endif
                            ray_state, params);
                    }

                #if DF_HANDLE_SLOTS != DF_HSM_NONE
                }
                #endif
            }
        }

        // importance sample BSDF
        {
            sample_data.xi.x = curand_uniform(&rand_state);
            sample_data.xi.y = curand_uniform(&rand_state);
            sample_data.xi.z = curand_uniform(&rand_state);
            sample_data.xi.w = curand_uniform(&rand_state);


            // sample the materials BSDF
            as_bsdf_sample(func_idx)(&sample_data, &state, &mdl_resources.data, NULL, arg_block);

            if (sample_data.event_type == BSDF_EVENT_ABSORB)
                return false;

            ray_state.dir = sample_data.k2;
            ray_state.weight *= sample_data.bsdf_over_pdf;


            const bool transmission = (sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0;
            if (transmission)
                ray_state.inside = !ray_state.inside;

            const bool is_specular = (sample_data.event_type & BSDF_EVENT_SPECULAR) != 0;

            Transition_type next;
            if (is_specular)
                next = transmission ? TRANSITION_SCATTER_ST : TRANSITION_SCATTER_SR;
            else if ((sample_data.event_type & BSDF_EVENT_DIFFUSE) != 0)
                next = transmission ? TRANSITION_SCATTER_DT : TRANSITION_SCATTER_DR;
            else
                next = transmission ? TRANSITION_SCATTER_GT : TRANSITION_SCATTER_GR;

            // move ray to the next sampled state
            ray_state.lpe_current_state = lpe_transition(
                ray_state.lpe_current_state, next,
                #if DF_HANDLE_SLOTS == DF_HSM_NONE
                    // ill-defined case: the LPE machine expects tags but the renderer ignores them
                    // -> the resulting image of LPEs with tags is undefined in this case
                    params.default_gtag,
                #else
                    material.bsdf_mtag_to_gtag_map[sample_data.handle], // sampled lobe
                #endif
                params);

            // depending on the geometry, the ray might be displaced before continuing
            continue_ray(ray_state, hit, sample_data.event_type, params);

            if (ray_state.inside)
            {
                // avoid self-intersections
                ray_state.pos -= hit.normal * 0.001f;

                return true; // continue bouncing in sphere
            }
            else if (params.mdl_test_type != MDL_TEST_NO_ENV &&
                params.mdl_test_type != MDL_TEST_EVAL)
            {
                // leaving sphere, add contribution from environment hit

                float pdf;
                const float3 f = environment_eval(pdf, sample_data.k2, params);

                float bsdf_pdf;
                if (params.mdl_test_type == MDL_TEST_MIS_PDF)
                {
                    const float3 k2 = sample_data.k2;
                    pdf_data.k2 = k2;

                    // get pdf corresponding to the materials BSDF
                    as_bsdf_pdf(func_idx)(&pdf_data, &state, &mdl_resources.data, NULL, arg_block);

                    bsdf_pdf = pdf_data.pdf;
                }
                else
                    bsdf_pdf = sample_data.pdf;

                if (is_specular || bsdf_pdf > 0.0f)
                {
                    const float mis_weight = is_specular ||
                        (params.mdl_test_type == MDL_TEST_SAMPLE) ? 1.0f :
                            bsdf_pdf / (pdf + bsdf_pdf);

                    float3 specular_contrib = ray_state.weight * f * mis_weight;
                    accumulate_contribution(
                        TRANSITION_LIGHT, params.env_gtag /* light group 'env' */,
                        specular_contrib,
                        ray_state, params);
                }
            }
        }
    }

    return false;
}

struct render_result
{
    float3 beauty;
    auxiliary_data aux;
};

__device__ inline render_result render_scene(
    Rand_state &rand_state,
    const Kernel_params &params,
    const unsigned x,
    const unsigned y)
{
    const float inv_res_x = 1.0f / (float)params.resolution.x;
    const float inv_res_y = 1.0f / (float)params.resolution.y;

    const float dx = params.disable_aa ? 0.5f : curand_uniform(&rand_state);
    const float dy = params.disable_aa ? 0.5f : curand_uniform(&rand_state);

    const float2 screen_pos = make_float2(
        ((float)x + dx) * inv_res_x,
        ((float)y + dy) * inv_res_y);

    const float r    = (2.0f * screen_pos.x               - 1.0f);
    const float r_rx = (2.0f * (screen_pos.x + inv_res_x) - 1.0f);
    const float u    = (2.0f * screen_pos.y               - 1.0f);
    const float u_ry = (2.0f * (screen_pos.y + inv_res_y) - 1.0f);
    const float aspect = (float)params.resolution.y / (float)params.resolution.x;

    render_result res;
    clear(res.aux);

    Ray_state ray_state;
    ray_state.contribution = make_float3(0.0f, 0.0f, 0.0f);
    ray_state.weight = make_float3(1.0f, 1.0f, 1.0f);
    ray_state.pos = ray_state.pos_rx = ray_state.pos_ry = params.cam_pos;
    ray_state.dir = normalize(
        params.cam_dir * params.cam_focal + params.cam_right * r    + params.cam_up * aspect * u);
    ray_state.dir_rx = normalize(
        params.cam_dir * params.cam_focal + params.cam_right * r_rx + params.cam_up * aspect * u);
    ray_state.dir_ry = normalize(
        params.cam_dir * params.cam_focal + params.cam_right * r    + params.cam_up * aspect * u_ry);
    ray_state.inside = false;
    ray_state.lpe_current_state = 1; // already at the camera so state 0 to 1 is free as long as
                                     // there is only one camera
    ray_state.aux = &res.aux;
    const unsigned int max_inters = params.max_path_length - 1;
    for (ray_state.intersection = 0; ray_state.intersection < max_inters; ++ray_state.intersection)
    {
        if (!trace_scene(rand_state, ray_state, params))
            break;
    }

    res.beauty =
        isfinite(ray_state.contribution.x) &&
        isfinite(ray_state.contribution.y) &&
        isfinite(ray_state.contribution.z) ? ray_state.contribution : make_float3(0.0f, 0.0f, 0.0f);
    normalize(res.aux);
    return res;
}


// quantize + gamma
__device__ inline unsigned int float3_to_rgba8(float3 val)
{
    const unsigned int r = (unsigned int) (255.0 * powf(saturate(val.x), 1.0f / 2.2f));
    const unsigned int g = (unsigned int) (255.0 * powf(saturate(val.y), 1.0f / 2.2f));
    const unsigned int b = (unsigned int) (255.0 * powf(saturate(val.z), 1.0f / 2.2f));
    return 0xff000000 | (b << 16) | (g << 8) | r;
}

// exposure + simple Reinhard tonemapper + gamma
__device__ inline unsigned int display(float3 val, const float tonemap_scale)
{
    val *= tonemap_scale;
    const float burn_out = 0.1f;
    val.x *= (1.0f + val.x * burn_out) / (1.0f + val.x);
    val.y *= (1.0f + val.y * burn_out) / (1.0f + val.y);
    val.z *= (1.0f + val.z * burn_out) / (1.0f + val.z);
    return float3_to_rgba8(val);
}


// CUDA kernel rendering simple geometry with IBL
extern "C" __global__ void render_scene_kernel(
    const Kernel_params kernel_params)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernel_params.resolution.x || y >= kernel_params.resolution.y)
        return;

    const unsigned int idx = y * kernel_params.resolution.x + x;
    Rand_state rand_state;
    const unsigned int num_dim = kernel_params.disable_aa ? 6 : 8; // 2 camera, 3 BSDF, 3 environment
    curand_init(idx, /*subsequence=*/0, kernel_params.iteration_start * num_dim, &rand_state);

    render_result res;
    float3 beauty = make_float3(0.0f, 0.0f, 0.0f);
    auxiliary_data aux;
    clear(aux);
    for (unsigned int s = 0; s < kernel_params.iteration_num; ++s)
    {
        res = render_scene(
            rand_state,
            kernel_params,
            x, y);

        beauty += res.beauty;
        aux += res.aux;

    }
    beauty *= 1.0f / (float)kernel_params.iteration_num;
    normalize(aux);

    // accumulate
    if (kernel_params.iteration_start == 0) {
        kernel_params.accum_buffer[idx] = beauty;

        if (kernel_params.enable_auxiliary_output) {
            kernel_params.albedo_buffer[idx] = aux.albedo;
            kernel_params.normal_buffer[idx] = aux.normal;
        }
    } else {
        float iteration_weight = (float) kernel_params.iteration_num /
            (float) (kernel_params.iteration_start + kernel_params.iteration_num);

        float3 buffer_val = kernel_params.accum_buffer[idx] +
            (beauty - kernel_params.accum_buffer[idx]) * iteration_weight;

        kernel_params.accum_buffer[idx] =
            (isinf(buffer_val.x) || isnan(buffer_val.y) || isinf(buffer_val.z) ||
             isnan(buffer_val.x) || isinf(buffer_val.y) || isnan(buffer_val.z))
                ? make_float3(0.0f, 0.0f, 1.0e+30f)
                : buffer_val;

        if (kernel_params.enable_auxiliary_output) {

            // albedo
            kernel_params.albedo_buffer[idx] = kernel_params.albedo_buffer[idx] +
                (aux.albedo - kernel_params.albedo_buffer[idx]) * iteration_weight;

            // normal, check for zero length first
            float3 weighted_normal = kernel_params.normal_buffer[idx] +
                (aux.normal - kernel_params.normal_buffer[idx]) * iteration_weight;
            if (dot(weighted_normal, weighted_normal) > 0.0f)
                weighted_normal = normalize(weighted_normal);
            kernel_params.normal_buffer[idx] = weighted_normal;
        }
    }

    // update display buffer
    if (kernel_params.display_buffer)
    {
        switch (kernel_params.enable_auxiliary_output ? kernel_params.display_buffer_index : 0)
        {
        case 1: /* albedo */
            kernel_params.display_buffer[idx] = float3_to_rgba8(kernel_params.albedo_buffer[idx]);
            break;

        case 2: /* normal */
        {
            float3 display_normal = kernel_params.normal_buffer[idx];
            if (dot(display_normal, display_normal) > 0) {
                display_normal.x = display_normal.x * 0.5f + 0.5f;
                display_normal.y = display_normal.y * 0.5f + 0.5f;
                display_normal.z = display_normal.z * 0.5f + 0.5f;
            }
            kernel_params.display_buffer[idx] = float3_to_rgba8(display_normal);
            break;
        }
        default: /* beauty */
            kernel_params.display_buffer[idx] =
                display(kernel_params.accum_buffer[idx], kernel_params.exposure_scale);
            break;
        }
    }


}
