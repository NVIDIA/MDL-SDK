/******************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/shared/texture_support_cuda.h
//
// This file contains the implementations and the vtables of the texture access functions.

#ifndef TEXTURE_SUPPORT_CUDA_H
#define TEXTURE_SUPPORT_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <mi/mdl/mdl_target_types.h>

#define USE_SMOOTHERSTEP_FILTER

typedef mi::mdl::tct_deriv_float                     tct_deriv_float;
typedef mi::mdl::tct_deriv_float2                    tct_deriv_float2;
typedef mi::mdl::tct_deriv_arr_float_2               tct_deriv_arr_float_2;
typedef mi::mdl::tct_deriv_arr_float_3               tct_deriv_arr_float_3;
typedef mi::mdl::tct_deriv_arr_float_4               tct_deriv_arr_float_4;
typedef mi::mdl::Shading_state_material_with_derivs  Shading_state_material_with_derivs;
typedef mi::mdl::Shading_state_material              Shading_state_material;
typedef mi::mdl::Texture_handler_base                Texture_handler_base;
typedef mi::mdl::stdlib::Tex_wrap_mode               Tex_wrap_mode;
typedef mi::mdl::stdlib::Mbsdf_part                  Mbsdf_part;


// Custom structure representing an MDL texture, containing filtered and unfiltered CUDA texture
// objects and the size of the texture.
struct Texture
{
    cudaTextureObject_t  filtered_object;    // uses filter mode cudaFilterModeLinear
    cudaTextureObject_t  unfiltered_object;  // uses filter mode cudaFilterModePoint
    uint3                size;               // size of the texture, needed for texel access
    float3               inv_size;           // the inverse values of the size of the texture
};


// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler : Texture_handler_base {
    // additional data for the texture access functions can be provided here
    size_t         num_textures;        // the number of textures used by the material
                                        // (without the invalid texture)
    Texture const *textures;            // the textures used by the material
                                        // (without the invalid texture)
};

// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler_deriv : mi::mdl::Texture_handler_deriv_base {
    // additional data for the texture access functions can be provided here
    size_t         num_textures;       // the number of textures used by the material
                                       // (without the invalid texture)
    Texture const *textures;           // the textures used by the material
                                       // (without the invalid texture)
};


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


// ------------------------------------------------------------------------------------------------
// Textures
// ------------------------------------------------------------------------------------------------

// Applies wrapping and cropping to the given coordinate.
// Note: This macro returns if wrap mode is clip and the coordinate is out of range.
#define WRAP_AND_CROP_OR_RETURN_BLACK(val, inv_dim, wrap_mode, crop_vals, store_res_func)    \
  do {                                                                                       \
    if ( (wrap_mode) == mi::mdl::stdlib::wrap_repeat &&                                      \
        (crop_vals)[0] == 0.0f && (crop_vals)[1] == 1.0f ) {                                 \
      /* Do nothing, use texture sampler default behavior */                                 \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
      if ( (wrap_mode) == mi::mdl::stdlib::wrap_repeat )                                     \
        val = val - floorf(val);                                                             \
      else {                                                                                 \
        if ( (wrap_mode) == mi::mdl::stdlib::wrap_clip && (val < 0.0f || val >= 1.0f) ) {    \
          store_res_func(result, 0.0f);                                                      \
          return;                                                                            \
        }                                                                                    \
        else if ( (wrap_mode) == mi::mdl::stdlib::wrap_mirrored_repeat ) {                   \
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
    float const                 crop_v[2],
    float                       frame)
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
    float const                 crop_v[2],
    float                       frame)
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
    float const                 crop_v[2],
    float                       frame)
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
    float const                 crop_v[2],
    float                       frame)
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
// Note: uvtile and/or animated textures are not supported
extern "C" __device__ void tex_texel_float4_2d(
    float                       result[4],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    int const                   coord[2],
    int const                   /*uv_tile*/[2],
    float                       /*frame*/)
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
    float const                 crop_w[2],
    float                       frame)
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
    float const                 crop_w[2],
    float                       frame)
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
    const int                   coord[3],
    float                       /*frame*/)
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
// Note: uvtile and/or animated textures are not supported
extern "C" __device__ void tex_resolution_2d(
    int                         result[2],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx,
    int const                   /*uv_tile*/[2],
    float                       frame)
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
    unsigned                    texture_idx,
    float                       frame)
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

// Implementation of frame function needed by generated code.
// Not supported
extern "C" __device__ void tex_frame(
    int                         result[2],
    Texture_handler_base const *self_base,
    unsigned                    texture_idx)
{
    result[0] = 0;
    result[1] = 0;
}

// ------------------------------------------------------------------------------------------------
// Light Profiles
//
// Note:  Light profiles are not implemented for the MDL Core examples.
//        See the MDL SDK counterpart for details.
// ------------------------------------------------------------------------------------------------

// Implementation of light_profile_power() for a light profile.
extern "C" __device__ float df_light_profile_power(
    Texture_handler_base const *self_base,
    unsigned                    light_profile_idx)
{
    return 0.0f;
}

// Implementation of light_profile_maximum() for a light profile.
extern "C" __device__ float df_light_profile_maximum(
    Texture_handler_base const *self_base,
    unsigned                    light_profile_idx)
{
    return 0.0f;
}

// Implementation of light_profile_isvalid() for a light profile.
extern "C" __device__ bool df_light_profile_isvalid(
    Texture_handler_base const *self_base,
    unsigned                    light_profile_idx)
{
    return false;
}

// Implementation of df::light_profile_evaluate() for a light profile.
extern "C" __device__ float df_light_profile_evaluate(
    Texture_handler_base const  *self_base,
    unsigned                    light_profile_idx,
    float const                 theta_phi[2])
{
    return 0.0f;
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
}

// Implementation of df::light_profile_pdf() for a light profile.
extern "C" __device__ float df_light_profile_pdf(
    Texture_handler_base const  *self_base,
    unsigned                    light_profile_idx,
    float const                 theta_phi[2])
{
    return 0.0f;
}

// ------------------------------------------------------------------------------------------------
// BSDF Measurements
//
// Note:  Measured BSDFs are not implemented for the MDL Core examples.
//        See the MDL SDK counterpart for details.
// ------------------------------------------------------------------------------------------------

// Implementation of df::bsdf_measurement_isvalid() for an MBSDF.
extern "C" __device__ bool df_bsdf_measurement_isvalid(
    Texture_handler_base const *self_base,
    unsigned                    bsdf_measurement_index)
{
    return false;
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
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
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
    store_result3(result, 0.0f);
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
}

// Implementation of df::bsdf_measurement_pdf() for an MBSDF.
extern "C" __device__ float df_bsdf_measurement_pdf(
    Texture_handler_base const  *self_base,
    unsigned                    bsdf_measurement_index,
    float const                 theta_phi_in[2],
    float const                 theta_phi_out[2],
    Mbsdf_part                  part)
{
    return 0.0f;
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
    store_result4(result, 0.0f);
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

// The vtable containing all texture access handlers required by the generated code
// in "vtable" mode.
__device__ mi::mdl::Texture_handler_vtable tex_vtable = {
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
    tex_frame,
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
__device__ mi::mdl::Texture_handler_deriv_vtable tex_deriv_vtable = {
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
    tex_frame,
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

#endif  // TEXTURE_SUPPORT_CUDA_H
