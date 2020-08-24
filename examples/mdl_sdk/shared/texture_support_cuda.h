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

// examples/mdl_sdk/shared/texture_support_cuda.h
//
// This file contains the implementations and the vtables of the texture access functions.

#ifndef TEXTURE_SUPPORT_CUDA_H
#define TEXTURE_SUPPORT_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <mi/neuraylib/target_code_types.h>

#define USE_SMOOTHERSTEP_FILTER
#ifndef M_PI
    #define M_PI            3.14159265358979323846
#endif
#define M_ONE_OVER_PI       0.318309886183790671538


typedef mi::neuraylib::tct_deriv_float                     tct_deriv_float;
typedef mi::neuraylib::tct_deriv_float2                    tct_deriv_float2;
typedef mi::neuraylib::tct_deriv_arr_float_2               tct_deriv_arr_float_2;
typedef mi::neuraylib::tct_deriv_arr_float_3               tct_deriv_arr_float_3;
typedef mi::neuraylib::tct_deriv_arr_float_4               tct_deriv_arr_float_4;
typedef mi::neuraylib::Shading_state_material_with_derivs  Shading_state_material_with_derivs;
typedef mi::neuraylib::Shading_state_material              Shading_state_material;
typedef mi::neuraylib::Texture_handler_base                Texture_handler_base;
typedef mi::neuraylib::Tex_wrap_mode                       Tex_wrap_mode;
typedef mi::neuraylib::Mbsdf_part                          Mbsdf_part;


// Custom structure representing an MDL texture, containing filtered and unfiltered CUDA texture
// objects and the size of the texture.
struct Texture
{
    explicit Texture()
        : filtered_object(0)
        , unfiltered_object(0)
        , size(make_uint3(0, 0, 0))
        , inv_size(make_float3(0.0f, 0.0f, 0.0f))
    {}

    explicit Texture(
        cudaTextureObject_t  filtered_object,
        cudaTextureObject_t  unfiltered_object,
        uint3                size)
        : filtered_object(filtered_object)
        , unfiltered_object(unfiltered_object)
        , size(size)
        , inv_size(make_float3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z))
    {}

    cudaTextureObject_t  filtered_object;    // uses filter mode cudaFilterModeLinear
    cudaTextureObject_t  unfiltered_object;  // uses filter mode cudaFilterModePoint
    uint3                size;               // size of the texture, needed for texel access
    float3               inv_size;           // the inverse values of the size of the texture
};

// Custom structure representing an MDL BSDF measurement.
struct Mbsdf
{
    unsigned            has_data[2];            // true if there is a measurement for this part
    cudaTextureObject_t eval_data[2];           // uses filter mode cudaFilterModeLinear
    float               max_albedo[2];          // max albedo used to limit the multiplier
    float*              sample_data[2];         // CDFs for sampling a BSDF measurement
    float*              albedo_data[2];         // max albedo for each theta (isotropic)

    uint2           angular_resolution[2];      // size of the dataset, needed for texel access
    float2          inv_angular_resolution[2];  // the inverse values of the size of the dataset
    unsigned        num_channels[2];            // number of color channels (1 or 3)
};

// Structure representing a Light Profile
struct Lightprofile
{
    explicit Lightprofile()
        : angular_resolution(make_uint2(0, 0))
        , theta_phi_start(make_float2(0.0f, 0.0f))
        , theta_phi_delta(make_float2(0.0f, 0.0f))
        , theta_phi_inv_delta(make_float2(0.0f, 0.0f))
        , candela_multiplier(0.0f)
        , total_power(0.0f)
        , eval_data(0)
    {
    }

    uint2           angular_resolution;     // angular resolution of the grid
    float2          theta_phi_start;        // start of the grid
    float2          theta_phi_delta;        // angular step size
    float2          theta_phi_inv_delta;    // inverse step size
    float           candela_multiplier;     // factor to rescale the normalized data
    float           total_power;

    cudaTextureObject_t eval_data;          // normalized data sampled on grid
    float*              cdf_data;           // CDFs for sampling a light profile
};


// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler : Texture_handler_base {
    // additional data for the texture access functions can be provided here

    size_t         num_textures;        // the number of textures used by the material
                                        // (without the invalid texture)
    Texture const *textures;            // the textures used by the material
                                        // (without the invalid texture)

    size_t         num_mbsdfs;          // the number of mbsdfs used by the material
                                        // (without the invalid mbsdf)
    Mbsdf const    *mbsdfs;             // the mbsdfs used by the material
                                        // (without the invalid mbsdf)

    size_t          num_lightprofiles;  // number of elements in the lightprofiles field
                                        // (without the invalid light profile)
    Lightprofile const *lightprofiles;  // a device pointer to a list of mbsdfs objects, if used
                                        // (without the invalid light profile)
};

// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler_deriv : mi::neuraylib::Texture_handler_deriv_base {
    // additional data for the texture access functions can be provided here

    size_t         num_textures;        // the number of textures used by the material
                                        // (without the invalid texture)
    Texture const *textures;            // the textures used by the material
                                        // (without the invalid texture)

    size_t         num_mbsdfs;          // the number of mbsdfs used by the material
                                        // (without the invalid texture)
    Mbsdf const    *mbsdfs;             // the mbsdfs used by the material
                                        // (without the invalid texture)

    size_t          num_lightprofiles;  // number of elements in the lightprofiles field
                                        // (without the invalid light profile)
    Lightprofile const *lightprofiles;  // a device pointer to a list of mbsdfs objects, if used
                                        // (without the invalid light profile)
};


#if defined(__CUDACC__)

// Stores a float4 in a float[4] array.
__device__ inline void store_result4(float res[4], const float4 &v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
    res[3] = v.w;
}

// Stores a float in all elements of a float[4] array.
__device__ inline void store_result4(float res[4], float s)
{
    res[0] = res[1] = res[2] = res[3] = s;
}

// Stores the given float values in a float[4] array.
__device__ inline void store_result4(
    float res[4], float v0, float v1, float v2, float v3)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
    res[3] = v3;
}

// Stores a float3 in a float[3] array.
__device__ inline void store_result3(float res[3], float3 const&v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
}

// Stores a float4 in a float[3] array, ignoring v.w.
__device__ inline void store_result3(float res[3], const float4 &v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
}

// Stores a float in all elements of a float[3] array.
__device__ inline void store_result3(float res[3], float s)
{
    res[0] = res[1] = res[2] = s;
}

// Stores the given float values in a float[3] array.
__device__ inline void store_result3(float res[3], float v0, float v1, float v2)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
}

// Stores the luminance if a given float[3] in a float.
__device__ inline void store_result1(float* res, float3 const& v)
{
    // store luminance
    *res = 0.212671 * v.x + 0.715160 * v.y + 0.072169 * v.z;
}


// Stores the luminance if a given float[3] in a float.
__device__ inline void store_result1(float* res, float v0, float v1, float v2)
{
    // store luminance
    *res = 0.212671 * v0 + 0.715160 * v1 + 0.072169 * v2;
}

// Stores a given float in a float
__device__ inline void store_result1(float* res, float s)
{
    *res = s;
}


// ------------------------------------------------------------------------------------------------
// Textures
// ------------------------------------------------------------------------------------------------

// Applies wrapping and cropping to the given coordinate.
// Note: This macro returns if wrap mode is clip and the coordinate is out of range.
#define WRAP_AND_CROP_OR_RETURN_BLACK(val, inv_dim, wrap_mode, crop_vals, store_res_func)    \
  do {                                                                                       \
    if ( (wrap_mode) == mi::neuraylib::TEX_WRAP_REPEAT &&                                    \
        (crop_vals)[0] == 0.0f && (crop_vals)[1] == 1.0f ) {                                 \
      /* Do nothing, use texture sampler default behavior */                                 \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
      if ( (wrap_mode) == mi::neuraylib::TEX_WRAP_REPEAT )                                   \
        val = val - floorf(val);                                                             \
      else {                                                                                 \
        if ( (wrap_mode) == mi::neuraylib::TEX_WRAP_CLIP && (val < 0.0f || val >= 1.0f) ) {  \
          store_res_func(result, 0.0f);                                                      \
          return;                                                                            \
        }                                                                                    \
        else if ( (wrap_mode) == mi::neuraylib::TEX_WRAP_MIRRORED_REPEAT ) {                 \
          float floored_val = floorf(val);                                                   \
          if ( (int(floored_val) & 1) != 0 )                                                 \
            val = 1.0f - (val - floored_val);                                                \
          else                                                                               \
            val = val - floored_val;                                                         \
        }                                                                                    \
        float inv_hdim = 0.5f * (inv_dim);                                                   \
        val = fminf(fmaxf(val, inv_hdim), 1.f - inv_hdim);                                   \
      }                                                                                      \
      val = val * ((crop_vals)[1] - (crop_vals)[0]) + (crop_vals)[0];                        \
    }                                                                                        \
  } while ( 0 )


#ifdef USE_SMOOTHERSTEP_FILTER
// Modify texture coordinates to get better texture filtering,
// see http://www.iquilezles.org/www/articles/texture/texture.htm
#define APPLY_SMOOTHERSTEP_FILTER()                                                         \
    do {                                                                                    \
        u = u * tex.size.x + 0.5f;                                                          \
        v = v * tex.size.y + 0.5f;                                                          \
                                                                                            \
        float u_i = floorf(u), v_i = floorf(v);                                             \
        float u_f = u - u_i;                                                                \
        float v_f = v - v_i;                                                                \
        u_f = u_f * u_f * u_f * (u_f * (u_f * 6.f - 15.f) + 10.f);                          \
        v_f = v_f * v_f * v_f * (v_f * (v_f * 6.f - 15.f) + 10.f);                          \
        u = u_i + u_f;                                                                      \
        v = v_i + v_f;                                                                      \
                                                                                            \
        u = (u - 0.5f) * tex.inv_size.x;                                                    \
        v = (v - 0.5f) * tex.inv_size.y;                                                    \
    } while ( 0 )
#else
#define APPLY_SMOOTHERSTEP_FILTER()
#endif


// Implementation of tex::lookup_float4() for a texture_2d texture.
extern "C" __device__ void tex_lookup_float4_2d(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    float const                 coord[2],
    Tex_wrap_mode const         wrap_u,
    Tex_wrap_mode const         wrap_v,
    float const                 crop_u[2],
    float const                 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];
    float u = coord[0], v = coord[1];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, tex.inv_size.x, wrap_u, crop_u, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, tex.inv_size.y, wrap_v, crop_v, store_result4);

    APPLY_SMOOTHERSTEP_FILTER();

    store_result4(result, tex2D<float4>(tex.filtered_object, u, v));
}

// Implementation of tex::lookup_float4() for a texture_2d texture.
extern "C" __device__ void tex_lookup_deriv_float4_2d(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    tct_deriv_float2 const     *coord,
    Tex_wrap_mode const         wrap_u,
    Tex_wrap_mode const         wrap_v,
    float const                 crop_u[2],
    float const                 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];
    float u = coord->val.x, v = coord->val.y;
    WRAP_AND_CROP_OR_RETURN_BLACK(u, tex.inv_size.x, wrap_u, crop_u, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, tex.inv_size.y, wrap_v, crop_v, store_result4);

    APPLY_SMOOTHERSTEP_FILTER();

    store_result4(result, tex2DGrad<float4>(tex.filtered_object, u, v, coord->dx, coord->dy));
}

// Implementation of tex::lookup_float3() for a texture_2d texture.
extern "C" __device__ void tex_lookup_float3_2d(
    float                       result[3],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    float const                 coord[2],
    Tex_wrap_mode const         wrap_u,
    Tex_wrap_mode const         wrap_v,
    float const                 crop_u[2],
    float const                 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];
    float u = coord[0], v = coord[1];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, tex.inv_size.x, wrap_u, crop_u, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, tex.inv_size.y, wrap_v, crop_v, store_result3);

    APPLY_SMOOTHERSTEP_FILTER();

    store_result3(result, tex2D<float4>(tex.filtered_object, u, v));
}

// Implementation of tex::lookup_float3() for a texture_2d texture.
extern "C" __device__ void tex_lookup_deriv_float3_2d(
    float                       result[3],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    tct_deriv_float2 const     *coord,
    Tex_wrap_mode const         wrap_u,
    Tex_wrap_mode const         wrap_v,
    float const                 crop_u[2],
    float const                 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];
    float u = coord->val.x, v = coord->val.y;
    WRAP_AND_CROP_OR_RETURN_BLACK(u, tex.inv_size.x, wrap_u, crop_u, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, tex.inv_size.y, wrap_v, crop_v, store_result3);

    APPLY_SMOOTHERSTEP_FILTER();

    store_result3(result, tex2DGrad<float4>(tex.filtered_object, u, v, coord->dx, coord->dy));
}

// Implementation of tex::texel_float4() for a texture_2d texture.
// Note: uvtile textures are not supported
extern "C" __device__ void tex_texel_float4_2d(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    int const                   coord[2],
    int const                   /*uv_tile*/[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    store_result4(result, tex2D<float4>(
        tex.unfiltered_object,
        float(coord[0]) * tex.inv_size.x,
        float(coord[1]) * tex.inv_size.y));
}

// Implementation of tex::lookup_float4() for a texture_3d texture.
extern "C" __device__ void tex_lookup_float4_3d(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    float const                 coord[3],
    Tex_wrap_mode               wrap_u,
    Tex_wrap_mode               wrap_v,
    Tex_wrap_mode               wrap_w,
    float const                 crop_u[2],
    float const                 crop_v[2],
    float const                 crop_w[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    float u = coord[0], v = coord[1], w = coord[2];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, tex.inv_size.x, wrap_u, crop_u, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, tex.inv_size.y, wrap_v, crop_v, store_result4);
    WRAP_AND_CROP_OR_RETURN_BLACK(w, tex.inv_size.z, wrap_w, crop_w, store_result4);

    store_result4(result, tex3D<float4>(tex.filtered_object, u, v, w));
}

// Implementation of tex::lookup_float3() for a texture_3d texture.
extern "C" __device__ void tex_lookup_float3_3d(
    float                       result[3],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    float const                 coord[3],
    Tex_wrap_mode               wrap_u,
    Tex_wrap_mode               wrap_v,
    Tex_wrap_mode               wrap_w,
    float const                 crop_u[2],
    float const                 crop_v[2],
    float const                 crop_w[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    float u = coord[0], v = coord[1], w = coord[2];
    WRAP_AND_CROP_OR_RETURN_BLACK(u, tex.inv_size.x, wrap_u, crop_u, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(v, tex.inv_size.y, wrap_v, crop_v, store_result3);
    WRAP_AND_CROP_OR_RETURN_BLACK(w, tex.inv_size.z, wrap_w, crop_w, store_result3);

    store_result3(result, tex3D<float4>(tex.filtered_object, u, v, w));
}

// Implementation of tex::texel_float4() for a texture_3d texture.
extern "C" __device__ void tex_texel_float4_3d(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    const int                   coord[3])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    store_result4(result, tex3D<float4>(
        tex.unfiltered_object,
        float(coord[0]) * tex.inv_size.x,
        float(coord[1]) * tex.inv_size.y,
        float(coord[2]) * tex.inv_size.z));
}

// Implementation of tex::lookup_float4() for a texture_cube texture.
extern "C" __device__ void tex_lookup_float4_cube(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    float const                 coord[3])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    store_result4(result, texCubemap<float4>(tex.filtered_object, coord[0], coord[1], coord[2]));
}

// Implementation of tex::lookup_float3() for a texture_cube texture.
extern "C" __device__ void tex_lookup_float3_cube(
    float                       result[3],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    float const                 coord[3])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    store_result3(result, texCubemap<float4>(tex.filtered_object, coord[0], coord[1], coord[2]));
}

// Implementation of resolution_2d function needed by generated code.
// Note: uvtile textures are not supported
extern "C" __device__ void tex_resolution_2d(
    int                         result[2],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    int const                   /*uv_tile*/[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if ( texture_idx == 0 || texture_idx - 1 >= self->num_textures ) {
        // invalid texture returns zero
        result[0] = 0;
        result[1] = 0;
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];
    result[0] = tex.size.x;
    result[1] = tex.size.y;
}

// Implementation of resolution_3d function needed by generated code.
// Note: 3d textures are not supported
extern "C" __device__ void tex_resolution_3d(
    int                         result[3],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx)
{
    // invalid texture returns zero
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
}

// Implementation of texture_isvalid().
extern "C" __device__ bool tex_texture_isvalid(
    Texture_handler_base const *self_base,
    unsigned                    texture_idx)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    return texture_idx != 0 && texture_idx - 1 < self->num_textures;
}


// ------------------------------------------------------------------------------------------------
// Light Profiles
// ------------------------------------------------------------------------------------------------


// Implementation of light_profile_power() for a light profile.
extern "C" __device__ float df_light_profile_power(
    Texture_handler_base const *self_base,
    unsigned                    light_profile_idx)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (light_profile_idx == 0 || light_profile_idx - 1 >= self->num_lightprofiles)
        return 0.0f; // invalid light profile returns zero

    const Lightprofile& lp = self->lightprofiles[light_profile_idx - 1];
    return lp.total_power;
}

// Implementation of light_profile_maximum() for a light profile.
extern "C" __device__ float df_light_profile_maximum(
    Texture_handler_base const *self_base,
    unsigned                    light_profile_idx)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (light_profile_idx == 0 || light_profile_idx - 1 >= self->num_lightprofiles)
        return 0.0f; // invalid light profile returns zero

    const Lightprofile& lp = self->lightprofiles[light_profile_idx - 1];
    return lp.candela_multiplier;
}

// Implementation of light_profile_isvalid() for a light profile.
extern "C" __device__ bool df_light_profile_isvalid(
    Texture_handler_base const *self_base,
    unsigned                    light_profile_idx)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    return light_profile_idx != 0 && light_profile_idx - 1 < self->num_lightprofiles;
}

// binary search through CDF
__device__ inline unsigned sample_cdf(
    const float* cdf,
    unsigned cdf_size,
    float xi)
{
    unsigned li = 0;
    unsigned ri = cdf_size - 1;
    unsigned m = (li + ri) / 2;
    while (ri > li)
    {
        if (xi < cdf[m])
            ri = m;
        else
            li = m + 1;

        m = (li + ri) / 2;
    }

    return m;
}


// Implementation of df::light_profile_evaluate() for a light profile.
extern "C" __device__ float df_light_profile_evaluate(
    Texture_handler_base const  *self_base,
    unsigned                    light_profile_idx,
    float const                 theta_phi[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (light_profile_idx == 0 || light_profile_idx - 1 >= self->num_lightprofiles)
        return 0.0f; // invalid light profile returns zero

    const Lightprofile& lp = self->lightprofiles[light_profile_idx - 1];

    // map theta to 0..1 range
    const float u = (theta_phi[0] - lp.theta_phi_start.x) *
        lp.theta_phi_inv_delta.x / float(lp.angular_resolution.x - 1);

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi[1] > 0.0f) ? theta_phi[1] : (float(2.0 * M_PI) + theta_phi[1]);

    // floorf wraps phi range into 0..2pi
    phi = phi - lp.theta_phi_start.y -
        floorf((phi - lp.theta_phi_start.y) * float(0.5 / M_PI)) * float(2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border
    // since it implies lp.theta_phi_start.y > 0 (and we really have "no data" below that)
    const float v = phi * lp.theta_phi_inv_delta.y / float(lp.angular_resolution.y - 1);

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return 0.0f;

    return tex2D<float>(lp.eval_data, u, v) * lp.candela_multiplier;
}

// Implementation of df::light_profile_sample() for a light profile.
extern "C" __device__ void df_light_profile_sample(
    float                       result[3],          // output: theta, phi, pdf
    Texture_handler_base const  *self_base,
    unsigned                    light_profile_idx,
    float const                 xi[3])              // uniform random values
{
    result[0] = -1.0f;  // negative theta means no emission
    result[1] = -1.0f;
    result[2] = 0.0f;

    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (light_profile_idx == 0 || light_profile_idx - 1 >= self->num_lightprofiles)
        return;  // invalid light profile returns zero

    const Lightprofile& lp = self->lightprofiles[light_profile_idx - 1];
    uint2 res = lp.angular_resolution;


    // sample theta_out
    //-------------------------------------------
    float xi0 = xi[0];
    const float* cdf_data_theta = lp.cdf_data;                          // CDF theta
    unsigned idx_theta = sample_cdf(cdf_data_theta, res.x - 1, xi0);    // binary search

    float prob_theta = cdf_data_theta[idx_theta];
    if (idx_theta > 0)
    {
        const float tmp = cdf_data_theta[idx_theta - 1];
        prob_theta -= tmp;
        xi0 -= tmp;
    }
    xi0 /= prob_theta;  // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi[1];
    const float* cdf_data_phi = cdf_data_theta + (res.x - 1)            // CDF theta block
                              + (idx_theta * (res.y - 1));              // selected CDF for phi

    const unsigned idx_phi = sample_cdf(cdf_data_phi, res.y - 1, xi1);        // binary search
    float prob_phi = cdf_data_phi[idx_phi];
    if (idx_phi > 0)
    {
        const float tmp = cdf_data_phi[idx_phi - 1];
        prob_phi -= tmp;
        xi1 -= tmp;
    }
    xi1 /= prob_phi;  // rescale for re-usage

    // compute theta and phi
    //-------------------------------------------
    // sample uniformly within the patch (grid cell)
    const float2 start = lp.theta_phi_start;
    const float2 delta = lp.theta_phi_delta;

    const float cos_theta_0 = cosf(start.x + float(idx_theta)      * delta.x);
    const float cos_theta_1 = cosf(start.x + float(idx_theta + 1u) * delta.x);

    //               n = \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    //                 = 1 / (\cos{\theta_0} - \cos{\theta_1})
    //
    //             \xi = n * \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    // => \cos{\theta} = (1 - \xi) \cos{\theta_0} + \xi \cos{\theta_1}

    const float cos_theta = (1.0f - xi1) * cos_theta_0 + xi1 * cos_theta_1;
    result[0] = acosf(cos_theta);
    result[1] = start.y + (float(idx_phi) + xi0) * delta.y;

    // align phi
    if (result[1] > float(2.0 * M_PI)) result[1] -= float(2.0 * M_PI);              // wrap
    if (result[1] > float(1.0 * M_PI)) result[1] = float(-2.0 * M_PI) + result[1];  // to [-pi, pi]

    // compute pdf
    //-------------------------------------------
    result[2] = prob_theta * prob_phi / (delta.y * (cos_theta_0 - cos_theta_1));
}


// Implementation of df::light_profile_pdf() for a light profile.
extern "C" __device__ float df_light_profile_pdf(
    Texture_handler_base const  *self_base,
    unsigned                    light_profile_idx,
    float const                 theta_phi[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (light_profile_idx == 0 || light_profile_idx - 1 >= self->num_lightprofiles)
        return 0.0f;  // invalid light profile returns zero

    const Lightprofile& lp = self->lightprofiles[light_profile_idx - 1];

    // CDF data
    const uint2 res = lp.angular_resolution;
    const float* cdf_data_theta = lp.cdf_data;


    // map theta to 0..1 range
    const float theta = theta_phi[0] - lp.theta_phi_start.x;
    const int idx_theta = int(theta * lp.theta_phi_inv_delta.x);

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi[1] > 0.0f) ? theta_phi[1] : (float(2.0 * M_PI) + theta_phi[1]);

    // floorf wraps phi range into 0..2pi
    phi = phi - lp.theta_phi_start.y -
        floorf((phi - lp.theta_phi_start.y) * float(0.5 / M_PI)) * float(2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border
    // since it implies lp.theta_phi_start.y > 0 (and we really have "no data" below that)
    const int idx_phi = int(phi * lp.theta_phi_inv_delta.y);

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (idx_theta < 0 || idx_theta > (res.x - 2) || idx_phi < 0 || idx_phi >(res.x - 2))
        return 0.0f;


    // get probability for theta
    //-------------------------------------------

    float prob_theta = cdf_data_theta[idx_theta];
    if (idx_theta > 0)
    {
        const float tmp = cdf_data_theta[idx_theta - 1];
        prob_theta -= tmp;
    }

    // get probability for phi
    //-------------------------------------------
    const float* cdf_data_phi = cdf_data_theta
        + (res.x - 1)                             // CDF theta block
        + (idx_theta * (res.y - 1));              // selected CDF for phi


    float prob_phi = cdf_data_phi[idx_phi];
    if (idx_phi > 0)
    {
        const float tmp = cdf_data_phi[idx_phi - 1];
        prob_phi -= tmp;
    }

    // compute probability to select a position in the sphere patch
    const float2 start = lp.theta_phi_start;
    const float2 delta = lp.theta_phi_delta;

    const float cos_theta_0 = cos(start.x + float(idx_theta)      * delta.x);
    const float cos_theta_1 = cos(start.x + float(idx_theta + 1u) * delta.x);

    return prob_theta * prob_phi / (delta.y * (cos_theta_0 - cos_theta_1));
}


// ------------------------------------------------------------------------------------------------
// BSDF Measurements
// ------------------------------------------------------------------------------------------------

// Implementation of bsdf_measurement_isvalid() for an MBSDF.
extern "C" __device__ bool df_bsdf_measurement_isvalid(
    Texture_handler_base const *self_base,
    unsigned                    bsdf_measurement_index)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    return bsdf_measurement_index != 0 && bsdf_measurement_index - 1 < self->num_mbsdfs;
}

// Implementation of df::bsdf_measurement_resolution() function needed by generated code,
// which retrieves the angular and chromatic resolution of the given MBSDF.
// The returned triple consists of: number of equi-spaced steps of theta_i and theta_o,
// number of equi-spaced steps of phi, and number of color channels (1 or 3).
extern "C" __device__ void df_bsdf_measurement_resolution(
    unsigned                    result[3],
    Texture_handler_base const  *self_base,
    unsigned                    bsdf_measurement_index,
    Mbsdf_part                  part)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (bsdf_measurement_index == 0 || bsdf_measurement_index - 1 >= self->num_mbsdfs)
    {
        // invalid MBSDF returns zero
        result[0] = 0;
        result[1] = 0;
        result[2] = 0;
        return;
    }

    Mbsdf const &bm = self->mbsdfs[bsdf_measurement_index - 1];
    const unsigned part_index = static_cast<unsigned>(part);

    // check for the part
    if (bm.has_data[part_index] == 0)
    {
        result[0] = 0;
        result[1] = 0;
        result[2] = 0;
        return;
    }

    // pass out the information
    result[0] = bm.angular_resolution[part_index].x;
    result[1] = bm.angular_resolution[part_index].y;
    result[2] = bm.num_channels[part_index];
}



__device__ inline float3 bsdf_compute_uvw(const float theta_phi_in[2],
                                          const float theta_phi_out[2])
{
    // assuming each phi is between -pi and pi
    float u = theta_phi_out[1] - theta_phi_in[1];
    if (u < 0.0) u += float(2.0 * M_PI);
    if (u > float(1.0 * M_PI)) u = float(2.0 * M_PI) - u;
    u *= M_ONE_OVER_PI;

    const float v = theta_phi_out[0] * float(2.0 / M_PI);
    const float w = theta_phi_in[0] * float(2.0 / M_PI);

    return make_float3(u, v, w);
}

template<typename T>
__device__ inline T bsdf_measurement_lookup(const cudaTextureObject_t& eval_volume,
                                            const float theta_phi_in[2],
                                            const float theta_phi_out[2])
{
    // 3D volume on the GPU (phi_delta x theta_out x theta_in)
    const float3 uvw = bsdf_compute_uvw(theta_phi_in, theta_phi_out);
    return tex3D<T>(eval_volume, uvw.x, uvw.y, uvw.z);
}

// Implementation of df::bsdf_measurement_evaluate() for an MBSDF.
extern "C" __device__ void df_bsdf_measurement_evaluate(
    float                       result[3],
    Texture_handler_base const  *self_base,
    unsigned                    bsdf_measurement_index,
    float const                 theta_phi_in[2],
    float const                 theta_phi_out[2],
    Mbsdf_part                  part)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (bsdf_measurement_index == 0 || bsdf_measurement_index - 1 >= self->num_mbsdfs)
    {
        // invalid MBSDF returns zero
        store_result3(result, 0.0f);
        return;
    }

    const Mbsdf& bm = self->mbsdfs[bsdf_measurement_index - 1];
    const unsigned part_index = static_cast<unsigned>(part);

    // check for the parta
    if (bm.has_data[part_index] == 0)
    {
        store_result3(result, 0.0f);
        return;
    }

    // handle channels
    if (bm.num_channels[part_index] == 3)
    {
        const float4 sample = bsdf_measurement_lookup<float4>(
            bm.eval_data[part_index], theta_phi_in, theta_phi_out);
        store_result3(result, sample.x, sample.y, sample.z);
    }
    else
    {
        const float sample = bsdf_measurement_lookup<float>(
            bm.eval_data[part_index], theta_phi_in, theta_phi_out);
        store_result3(result, sample);
    }
}

// Implementation of df::bsdf_measurement_sample() for an MBSDF.
extern "C" __device__ void df_bsdf_measurement_sample(
    float                       result[3],          // output: theta, phi, pdf
    Texture_handler_base const  *self_base,
    unsigned                    bsdf_measurement_index,
    float const                 theta_phi_out[2],
    float const                 xi[3],              // uniform random values
    Mbsdf_part                  part)
{
    result[0] = -1.0f;  // negative theta means absorption
    result[1] = -1.0f;
    result[2] = 0.0f;

    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (bsdf_measurement_index == 0 || bsdf_measurement_index - 1 >= self->num_mbsdfs)
        return;  // invalid MBSDFs returns zero

    const Mbsdf& bm = self->mbsdfs[bsdf_measurement_index - 1];
    unsigned part_index = static_cast<unsigned>(part);

    if (bm.has_data[part_index] == 0)
        return;  // check for the part

    // CDF data
    uint2 res = bm.angular_resolution[part_index];
    const float* sample_data = bm.sample_data[part_index];

    // compute the theta_in index (flipping input and output, BSDFs are symmetric)
    unsigned idx_theta_in = unsigned(theta_phi_out[0] * M_ONE_OVER_PI * 2.0f * float(res.x));
    idx_theta_in = min(idx_theta_in, res.x - 1);

    // sample theta_out
    //-------------------------------------------
    float xi0 = xi[0];
    const float* cdf_theta = sample_data + idx_theta_in * res.x;
    unsigned idx_theta_out = sample_cdf(cdf_theta, res.x, xi0);       // binary search

    float prob_theta = cdf_theta[idx_theta_out];
    if (idx_theta_out > 0)
    {
        const float tmp = cdf_theta[idx_theta_out - 1];
        prob_theta -= tmp;
        xi0 -= tmp;
    }
    xi0 /= prob_theta;  // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi[1];
    const float* cdf_phi = sample_data +
                           (res.x * res.x) +                                // CDF theta block
                           (idx_theta_in * res.x + idx_theta_out) * res.y;  // selected CDF phi

    // select which half-circle to choose with probability 0.5
    const bool flip = (xi1 > 0.5f);
    if (flip)
        xi1 = 1.0f - xi1;
    xi1 *= 2.0f;

    unsigned idx_phi_out = sample_cdf(cdf_phi, res.y, xi1);           // binary search
    float prob_phi = cdf_phi[idx_phi_out];
    if (idx_phi_out > 0)
    {
        const float tmp = cdf_phi[idx_phi_out - 1];
        prob_phi -= tmp;
        xi1 -= tmp;
    }
    xi1 /= prob_phi;  // rescale for re-usage

    // compute theta and phi out
    //-------------------------------------------
    const float2 inv_res = bm.inv_angular_resolution[part_index];

    const float s_theta = float(0.5 * M_PI) * inv_res.x;
    const float s_phi   = float(1.0 * M_PI) * inv_res.y;

    const float cos_theta_0 = cosf(float(idx_theta_out)      * s_theta);
    const float cos_theta_1 = cosf(float(idx_theta_out + 1u) * s_theta);

    const float cos_theta = cos_theta_0 * (1.0f - xi1) + cos_theta_1 * xi1;
    result[0] = acosf(cos_theta);
    result[1] = (float(idx_phi_out) + xi0) * s_phi;

    if (flip)
        result[1] = float(2.0 * M_PI) - result[1];  // phi \in [0, 2pi]

    // align phi
    result[1] += (theta_phi_out[1] > 0) ? theta_phi_out[1] : (float(2.0 * M_PI) + theta_phi_out[1]);
    if (result[1] > float(2.0 * M_PI)) result[1] -= float(2.0 * M_PI);
    if (result[1] > float(1.0 * M_PI)) result[1] = float(-2.0 * M_PI) + result[1];  // to [-pi, pi]

    // compute pdf
    //-------------------------------------------
    result[2] = prob_theta * prob_phi * 0.5f
                / (s_phi * (cos_theta_0 - cos_theta_1));
}

// Implementation of df::bsdf_measurement_pdf() for an MBSDF.
extern "C" __device__ float df_bsdf_measurement_pdf(
    Texture_handler_base const  *self_base,
    unsigned                    bsdf_measurement_index,
    float const                 theta_phi_in[2],
    float const                 theta_phi_out[2],
    Mbsdf_part                  part)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (bsdf_measurement_index == 0 || bsdf_measurement_index - 1 >= self->num_mbsdfs)
        return 0.0f;  // invalid MBSDF returns zero

    const Mbsdf& bm = self->mbsdfs[bsdf_measurement_index - 1];
    unsigned part_index = static_cast<unsigned>(part);

    // check for the part
    if (bm.has_data[part_index] == 0)
        return 0.0f;

    // CDF data and resolution
    const float* sample_data = bm.sample_data[part_index];
    uint2 res = bm.angular_resolution[part_index];

    // compute indices in the CDF data
    float3 uvw = bsdf_compute_uvw(theta_phi_in, theta_phi_out); // phi_delta, theta_out, theta_in
    unsigned idx_theta_in  = unsigned(theta_phi_in[0]  * M_ONE_OVER_PI * 2.0f * float(res.x));
    unsigned idx_theta_out = unsigned(theta_phi_out[0] * M_ONE_OVER_PI * 2.0f * float(res.x));
    unsigned idx_phi_out   = unsigned(uvw.x * float(res.y));
    idx_theta_in  = min(idx_theta_in, res.x - 1);
    idx_theta_out = min(idx_theta_out, res.x - 1);
    idx_phi_out   = min(idx_phi_out, res.y - 1);

    // get probability to select theta_out
    const float* cdf_theta = sample_data + idx_theta_in * res.x;
    float prob_theta = cdf_theta[idx_theta_out];
    if (idx_theta_out > 0)
    {
        const float tmp = cdf_theta[idx_theta_out - 1];
        prob_theta -= tmp;
    }

    // get probability to select phi_out
    const float* cdf_phi = sample_data +
        (res.x * res.x) +                                // CDF theta block
        (idx_theta_in * res.x + idx_theta_out) * res.y;  // selected CDF phi
    float prob_phi = cdf_phi[idx_phi_out];
    if (idx_phi_out > 0)
    {
        const float tmp = cdf_phi[idx_phi_out - 1];
        prob_phi -= tmp;
    }

    // compute probability to select a position in the sphere patch
    float2 inv_res = bm.inv_angular_resolution[part_index];

    const float s_theta = float(0.5 * M_PI) * inv_res.x;
    const float s_phi = float(1.0 * M_PI) * inv_res.y;

    const float cos_theta_0 = cosf(float(idx_theta_out)      * s_theta);
    const float cos_theta_1 = cosf(float(idx_theta_out + 1u) * s_theta);

    return prob_theta * prob_phi * 0.5f
        / (s_phi * (cos_theta_0 - cos_theta_1));
}


__device__ inline void df_bsdf_measurement_albedo(
    float                       result[2],          // output: max (in case of color) albedo
                                                    // for the selected direction ([0]) and
                                                    // global ([1])
    Texture_handler const       *self,
    unsigned                    bsdf_measurement_index,
    float const                 theta_phi[2],
    Mbsdf_part                  part)
{
    const Mbsdf& bm = self->mbsdfs[bsdf_measurement_index - 1];
    const unsigned part_index = static_cast<unsigned>(part);

    // check for the part
    if (bm.has_data[part_index] == 0)
        return;

    const uint2 res = bm.angular_resolution[part_index];
    unsigned idx_theta = unsigned(theta_phi[0] * float(2.0 / M_PI) * float(res.x));
    idx_theta = min(idx_theta, res.x - 1u);
    result[0] = bm.albedo_data[part_index][idx_theta];
    result[1] = bm.max_albedo[part_index];
}

// Implementation of df::bsdf_measurement_albedos() for an MBSDF.
extern "C" __device__ void df_bsdf_measurement_albedos(
    float                       result[4],          // output: [0] albedo refl. for theta_phi
                                                    //         [1] max albedo refl. global
                                                    //         [2] albedo trans. for theta_phi
                                                    //         [3] max albedo trans. global
    Texture_handler_base const  *self_base,
    unsigned                    bsdf_measurement_index,
    float const                 theta_phi[2])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 0.0f;

    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    if (bsdf_measurement_index == 0 || bsdf_measurement_index - 1 >= self->num_mbsdfs)
        return;  // invalid MBSDF returns zero

    df_bsdf_measurement_albedo(
        &result[0],
        self,
        bsdf_measurement_index,
        theta_phi,
        mi::neuraylib::MBSDF_DATA_REFLECTION);

    df_bsdf_measurement_albedo(
        &result[2],
        self,
        bsdf_measurement_index,
        theta_phi,
        mi::neuraylib::MBSDF_DATA_TRANSMISSION);
}


// ------------------------------------------------------------------------------------------------
// Scene data (dummy functions)
// ------------------------------------------------------------------------------------------------

#ifndef TEX_SUPPORT_NO_DUMMY_SCENEDATA

// Implementation of scene_data_isvalid().
extern "C" __device__ bool scene_data_isvalid(
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id)
{
    return false;
}

// Implementation of scene_data_lookup_float4().
extern "C" __device__ void scene_data_lookup_float4(
    float                                  result[4],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    float const                            default_value[4],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
    result[3] = default_value[3];
}

// Implementation of scene_data_lookup_float3().
extern "C" __device__ void scene_data_lookup_float3(
    float                                  result[3],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    float const                            default_value[3],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
}

// Implementation of scene_data_lookup_color().
extern "C" __device__ void scene_data_lookup_color(
    float                                  result[3],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    float const                            default_value[3],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
}

// Implementation of scene_data_lookup_float2().
extern "C" __device__ void scene_data_lookup_float2(
    float                                  result[2],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    float const                            default_value[2],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
}

// Implementation of scene_data_lookup_float().
extern "C" __device__ float scene_data_lookup_float(
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    float const                            default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    return default_value;
}

// Implementation of scene_data_lookup_int4().
extern "C" __device__ void scene_data_lookup_int4(
    int                                    result[4],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    int const                              default_value[4],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
    result[3] = default_value[3];
}

// Implementation of scene_data_lookup_int3().
extern "C" __device__ void scene_data_lookup_int3(
    int                                    result[3],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    int const                              default_value[3],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
}

// Implementation of scene_data_lookup_int2().
extern "C" __device__ void scene_data_lookup_int2(
    int                                    result[2],
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    int const                              default_value[2],
    bool                                   uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
}

// Implementation of scene_data_lookup_int().
extern "C" __device__ int scene_data_lookup_int(
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    int                                    default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    return default_value;
}

// Implementation of scene_data_lookup_float4() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float4(
    tct_deriv_arr_float_4                 *result,
    Texture_handler_base const            *self_base,
    Shading_state_material_with_derivs    *state,
    unsigned                               scene_data_id,
    tct_deriv_arr_float_4 const           *default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_float3() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float3(
    tct_deriv_arr_float_3                 *result,
    Texture_handler_base const            *self_base,
    Shading_state_material_with_derivs    *state,
    unsigned                               scene_data_id,
    tct_deriv_arr_float_3 const           *default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_color() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_color(
    tct_deriv_arr_float_3                 *result,
    Texture_handler_base const            *self_base,
    Shading_state_material_with_derivs    *state,
    unsigned                               scene_data_id,
    tct_deriv_arr_float_3 const           *default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_float2() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float2(
    tct_deriv_arr_float_2                 *result,
    Texture_handler_base const            *self_base,
    Shading_state_material_with_derivs    *state,
    unsigned                               scene_data_id,
    tct_deriv_arr_float_2 const           *default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_float() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float(
    tct_deriv_float                       *result,
    Texture_handler_base const            *self_base,
    Shading_state_material_with_derivs    *state,
    unsigned                               scene_data_id,
    tct_deriv_float const                 *default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

#endif  // TEX_SUPPORT_NO_DUMMY_SCENEDATA


// ------------------------------------------------------------------------------------------------
// Vtables
// ------------------------------------------------------------------------------------------------

#ifndef TEX_SUPPORT_NO_VTABLES
// The vtable containing all texture access handlers required by the generated code
// in "vtable" mode.
__device__ mi::neuraylib::Texture_handler_vtable tex_vtable = {
    tex_lookup_float4_2d,
    tex_lookup_float3_2d,
    tex_texel_float4_2d,
    tex_lookup_float4_3d,
    tex_lookup_float3_3d,
    tex_texel_float4_3d,
    tex_lookup_float4_cube,
    tex_lookup_float3_cube,
    tex_resolution_2d,
    tex_resolution_3d,
    tex_texture_isvalid,
    df_light_profile_power,
    df_light_profile_maximum,
    df_light_profile_isvalid,
    df_light_profile_evaluate,
    df_light_profile_sample,
    df_light_profile_pdf,
    df_bsdf_measurement_isvalid,
    df_bsdf_measurement_resolution,
    df_bsdf_measurement_evaluate,
    df_bsdf_measurement_sample,
    df_bsdf_measurement_pdf,
    df_bsdf_measurement_albedos,
    scene_data_isvalid,
    scene_data_lookup_float,
    scene_data_lookup_float2,
    scene_data_lookup_float3,
    scene_data_lookup_float4,
    scene_data_lookup_int,
    scene_data_lookup_int2,
    scene_data_lookup_int3,
    scene_data_lookup_int4,
    scene_data_lookup_color,
};

// The vtable containing all texture access handlers required by the generated code
// in "vtable" mode with derivatives.
__device__ mi::neuraylib::Texture_handler_deriv_vtable tex_deriv_vtable = {
    tex_lookup_deriv_float4_2d,
    tex_lookup_deriv_float3_2d,
    tex_texel_float4_2d,
    tex_lookup_float4_3d,
    tex_lookup_float3_3d,
    tex_texel_float4_3d,
    tex_lookup_float4_cube,
    tex_lookup_float3_cube,
    tex_resolution_2d,
    tex_resolution_3d,
    tex_texture_isvalid,
    df_light_profile_power,
    df_light_profile_maximum,
    df_light_profile_isvalid,
    df_light_profile_evaluate,
    df_light_profile_sample,
    df_light_profile_pdf,
    df_bsdf_measurement_isvalid,
    df_bsdf_measurement_resolution,
    df_bsdf_measurement_evaluate,
    df_bsdf_measurement_sample,
    df_bsdf_measurement_pdf,
    df_bsdf_measurement_albedos,
    scene_data_isvalid,
    scene_data_lookup_float,
    scene_data_lookup_float2,
    scene_data_lookup_float3,
    scene_data_lookup_float4,
    scene_data_lookup_int,
    scene_data_lookup_int2,
    scene_data_lookup_int3,
    scene_data_lookup_int4,
    scene_data_lookup_color,
    scene_data_lookup_deriv_float,
    scene_data_lookup_deriv_float2,
    scene_data_lookup_deriv_float3,
    scene_data_lookup_deriv_float4,
    scene_data_lookup_deriv_color,
};
#endif  // TEX_SUPPORT_NO_VTABLES

#endif  // __CUDACC__

#endif  // TEXTURE_SUPPORT_CUDA_H
