/******************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

// examples/texture_support_cuda.h
//
// This file contains the implementations and the vtable of the texture access functions.

#ifndef TEXTURE_SUPPORT_CUDA_H
#define TEXTURE_SUPPORT_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <mi/neuraylib/target_code_types.h>

#define USE_SMOOTHERSTEP_FILTER


typedef mi::neuraylib::Texture_handler_base Texture_handler_base;
typedef mi::neuraylib::Tex_wrap_mode Tex_wrap_mode;


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
__device__ inline void store_result4(float res[4], const float v)
{
    res[0] = res[1] = res[2] = res[3] = v;
}

// Stores the given float values in a float[4] array.
__device__ inline void store_result4(
    float res[4], const float v0, const float v1, const float v2, const float v3)
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
__device__ inline void store_result3(float res[3], const float v)
{
    res[0] = res[1] = res[2] = v;
}

// Stores the given float values in a float[3] array.
__device__ inline void store_result3(float res[3], const float v0, const float v1, const float v2)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
}


// Applies wrapping and cropping to the given coordinate.
// Note: This macro returns if wrap mode is clip and the coordinate is out of range.
#define WRAP_AND_CROP_OR_RETURN_BLACK(val, inv_dim, wrap_mode, crop_vals, store_res_func)    \
  do {                                                                                       \
    if ( (wrap_mode) == mi::neuraylib::TEX_WRAP_REPEAT &&                                    \
        (crop_vals)[0] == 0.0f && (crop_vals)[1] == 1.0f ) {                                 \
      /* Do nothing, use texture sampler default behaviour */                                \
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
        self->textures[texture_idx - 1].unfiltered_object,
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
        self->textures[texture_idx - 1].unfiltered_object,
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
    tex_resolution_2d
};

#endif  // TEXTURE_SUPPORT_CUDA_H
