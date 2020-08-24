/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

// This file contains the implementations and the vtable of the texture access functions.

#ifndef TEXTURE_SUPPORT_H
#define TEXTURE_SUPPORT_H

#include "example_shared.h"

#define USE_SMOOTHERSTEP_FILTER

typedef mi::neuraylib::tct_deriv_float                     tct_deriv_float;
typedef mi::neuraylib::tct_deriv_float2                    tct_deriv_float2;
typedef mi::neuraylib::tct_deriv_arr_float_2               tct_deriv_arr_float_2;
typedef mi::neuraylib::tct_deriv_arr_float_3               tct_deriv_arr_float_3;
typedef mi::neuraylib::tct_deriv_arr_float_4               tct_deriv_arr_float_4;
typedef mi::neuraylib::Shading_state_material_with_derivs  Shading_state_material_with_derivs;
typedef mi::neuraylib::Shading_state_material              Shading_state_material;
typedef mi::neuraylib::Texture_handler_base                Texture_handler_base;
typedef mi::neuraylib::Texture_handler_deriv_base          Texture_handler_deriv_base;
typedef mi::neuraylib::Tex_wrap_mode                       Tex_wrap_mode;

// Custom structure representing an MDL texture
struct Texture
{
    Texture(mi::base::Handle<const mi::neuraylib::ICanvas> c)
        : canvas(c)
        , ncomp(4)
    {
        // for now, we only support floating point rgba
        check_success(strcmp(canvas->get_type(), "Color") == 0);

        mi::base::Handle < const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));

        size.x = canvas->get_resolution_x();
        size.y = canvas->get_resolution_y();
        size.z = canvas->get_layers_size();

        data = static_cast<const mi::Float32*> (tile->get_data());
    }

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas;

    mi::Float32 const       *data;  // texture data for fast access

    mi::Uint32_3_struct     size;   // size of the texture

    mi::Uint32              ncomp;  // components per pixel

};

// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler : Texture_handler_base {
    // additional data for the texture access functions can be provided here
    size_t         num_textures;       // the number of textures used by the material
                                       // (without the invalid texture)
    Texture const *textures;           // the textures used by the material
                                       // (without the invalid texture)
};

// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler_deriv : Texture_handler_deriv_base {
    // additional data for the texture access functions can be provided here
    size_t         num_textures;       // the number of textures used by the material
                                       // (without the invalid texture)
    Texture const *textures;           // the textures used by the material
                                       // (without the invalid texture)
};

// Stores a float4 in a float[4] array.
inline static void store_result4(float res[4], const mi::Float32_4_struct &v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
    res[3] = v.w;
}

// Stores a float in all elements of a float[4] array.
static inline void store_result4(float res[4], const float v)
{
    res[0] = res[1] = res[2] = res[3] = v;
}

// Stores the given float values in a float[4] array.
static inline void store_result4(
    float res[4], const float v0, const float v1, const float v2, const float v3)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
    res[3] = v3;
}

// Stores a float4 in a float[3] array, ignoring v.w.
static inline void store_result3(float res[3], const mi::Float32_4_struct &v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
}

// Stores a float in all elements of a float[3] array.
static inline void store_result3(float res[3], const float v)
{
    res[0] = res[1] = res[2] = v;
}

// Stores the given float values in a float[3] array.
static inline void store_result3(float res[3], const float v0, const float v1, const float v2)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
}


// ------------------------------------------------------------------------------------------------
// Textures
// ------------------------------------------------------------------------------------------------

static inline mi::Float32 u2f_rn(const mi::Uint32 u) {
    return (mi::Float32) u;
}

static inline mi::Uint32 f2u_rz(const mi::Float32 f) {
    return (mi::Uint32) f;
}

static inline mi::Sint64 f2ll_rz(const mi::Float32 f) {
    return (mi::Sint64) f;
}

static inline mi::Sint64 f2ll_rd(const mi::Float32 f) {
    return (mi::Sint64) mi::math::floor(f);
}

mi::Uint32 texremap(
    mi::Uint32 tex_size, Tex_wrap_mode wrap_mode, const mi::Sint32 crop_offset, float texf)
{
    mi::Sint32 texi = 0;
    const mi::Sint64 texil = f2ll_rz(texf);
    const mi::Sint64 texsizel = mi::Sint64(tex_size);

    if (mi::Uint64(f2ll_rd(texf)) >= mi::Uint64(tex_size)) {

        // Wrap or clamp
        if (wrap_mode == Tex_wrap_mode::TEX_WRAP_CLAMP || wrap_mode == Tex_wrap_mode::TEX_WRAP_CLIP)
            texi = (int) std::min(std::max(texil, 0ll), (texsizel - 1ll));
        // Repeat
        else {

            texi = mi::Sint32(texil % texsizel);

            const mi::Sint32 s = mi::math::sign_bit(texf);
            const mi::Sint64 d = texil / (mi::Sint64) tex_size;
            const mi::Sint32 a =
                (mi::Sint32) (wrap_mode == Tex_wrap_mode::TEX_WRAP_MIRRORED_REPEAT) &
                ((mi::Sint32) d^s) & 1;
            const bool alternate = (a != 0);
            if (alternate)   // Flip negative tex
                texi = -texi;
            if (s != a) // "Otherwise" pad negative tex back to positive
                texi += (mi::Sint32) tex_size - 1;
        }
    }
    else texi = (int) texil;

    // Crop
    texi += crop_offset;
    return mi::Uint32(texi);
}

void tex_lookup2D(
    mi::Float32                   res[4],
    Texture const                 &tex,
    const mi::Float32             uv[2],
    mi::neuraylib::Tex_wrap_mode  wrap_u,
    mi::neuraylib::Tex_wrap_mode  wrap_v,
    const mi::Float32             crop_u[2],
    const mi::Float32             crop_v[2])
{
    const mi::Float32 crop_w = crop_u[1] - crop_u[0];
    const mi::Float32 crop_h = crop_v[1] - crop_v[0];

    const mi::Sint32_2 crop_offset(
        f2u_rz(u2f_rn(tex.size.x-1) * crop_u[0]),
        f2u_rz(u2f_rn(tex.size.y-1) * crop_v[0]));

    const mi::Uint32_2 crop_texres(
        std::max(f2u_rz(u2f_rn(tex.size.x) * crop_w), 1u),
        std::max(f2u_rz(u2f_rn(tex.size.y) * crop_h), 1u));

    const float U = uv[0] * crop_texres.x - 0.5f;
    const float V = uv[1] * crop_texres.y - 0.5f;

    const mi::Uint32 U0 = texremap(crop_texres.x, wrap_u, crop_offset[0], U);
    const mi::Uint32 U1 = texremap(crop_texres.x, wrap_u, crop_offset[0], U+1.0f);
    const mi::Uint32 V0 = texremap(crop_texres.y, wrap_v, crop_offset[1], V);
    const mi::Uint32 V1 = texremap(crop_texres.y, wrap_v, crop_offset[1], V+1.0f);

    const mi::Uint32 i00 = (tex.size.x * V0 + U0) * tex.ncomp;
    const mi::Uint32 i01 = (tex.size.x * V0 + U1) * tex.ncomp;
    const mi::Uint32 i10 = (tex.size.x * V1 + U0) * tex.ncomp;
    const mi::Uint32 i11 = (tex.size.x * V1 + U1) * tex.ncomp;

    mi::Float32 ufrac = U - mi::math::floor(U);
    mi::Float32 vfrac = V - mi::math::floor(V);

#ifdef USE_SMOOTHERSTEP_FILTER
    ufrac *= ufrac*ufrac*(ufrac*(ufrac*6.0f - 15.0f) + 10.0f); // smoother step
    vfrac *= vfrac*vfrac*(vfrac*(vfrac*6.0f - 15.0f) + 10.0f);
#endif

    const mi::Float32_4 c1 = mi::math::lerp(
        mi::Float32_4(tex.data[i00 + 0], tex.data[i00 + 1], tex.data[i00 + 2], tex.data[i00 + 3]),
        mi::Float32_4(tex.data[i01 + 0], tex.data[i01 + 1], tex.data[i01 + 2], tex.data[i01 + 3]),
        ufrac);

    const mi::Float32_4 c2 = mi::math::lerp(
        mi::Float32_4(tex.data[i10 + 0], tex.data[i10 + 1], tex.data[i10 + 2], tex.data[i10 + 3]),
        mi::Float32_4(tex.data[i11 + 0], tex.data[i11 + 1], tex.data[i11 + 2], tex.data[i11 + 3]),
        ufrac);

    store_result4(res, mi::math::lerp(c1, c2, vfrac));
}

/// Implementation of \c tex::lookup_float4() for a texture_2d texture.
void tex_lookup_float4_2d(
    mi::Float32 result[4],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx,
    const mi::Float32 coord[2],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    const mi::Float32 crop_u[2],
    const mi::Float32 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (texture_idx == 0 || texture_idx - 1 >= self->num_textures) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];
    tex_lookup2D(result, tex, coord, wrap_u, wrap_v, crop_u, crop_v);
}

/// Implementation of \c tex::lookup_float4() for a texture_2d texture with derivatives.
/// Note: derivatives are just ignored in this example runtime
void tex_lookup_deriv_float4_2d(
    mi::Float32 result[4],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx,
    const tct_deriv_float2 *coord,
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    const mi::Float32 crop_u[2],
    const mi::Float32 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (texture_idx == 0 || texture_idx - 1 >= self->num_textures) {
        // invalid texture returns zero
        store_result4(result, 0.0f);
        return;
    }

    mi::Float32 uv[2] = { coord->val.x, coord->val.y };
    Texture const &tex = self->textures[texture_idx - 1];
    tex_lookup2D(result, tex, uv, wrap_u, wrap_v, crop_u, crop_v);
}

/// Implementation of \c tex::lookup_float3() for a texture_2d texture.
void tex_lookup_float3_2d(
    mi::Float32 result[3],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx,
    const mi::Float32 coord[2],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    const mi::Float32 crop_u[2],
    const mi::Float32 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (texture_idx == 0 || texture_idx - 1 >= self->num_textures) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    mi::Float32 c[4];
    tex_lookup_float4_2d(c, self, texture_idx, coord, wrap_u, wrap_v, crop_u, crop_v);

    result[0] = c[0];
    result[1] = c[1];
    result[2] = c[2];
}

/// Implementation of \c tex::lookup_float3() for a texture_2d texture with derivatives.
/// Note: derivatives are just ignored in this example runtime
void tex_lookup_deriv_float3_2d(
    mi::Float32 result[3],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx,
    const tct_deriv_float2 *coord,
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    const mi::Float32 crop_u[2],
    const mi::Float32 crop_v[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (texture_idx == 0 || texture_idx - 1 >= self->num_textures) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    mi::Float32 c[4];
    mi::Float32 uv[2] = { coord->val.x, coord->val.y };
    tex_lookup_float4_2d(c, self, texture_idx, uv, wrap_u, wrap_v, crop_u, crop_v);

    result[0] = c[0];
    result[1] = c[1];
    result[2] = c[2];
}

/// Implementation of \c tex::texel_float4() for a texture_2d texture.
void tex_texel_float4_2d(
    mi::Float32 result[4],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx,
    const mi::Sint32 coord[2],
    const mi::Sint32 uv_tile[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (texture_idx == 0 || texture_idx - 1 >= self->num_textures) {
        // invalid texture returns zero
        store_result3(result, 0.0f);
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    const mi::Uint32 idx = (tex.size.x * coord[1] + coord[0]) * tex.ncomp;

    store_result4(result,
        mi::Float32_4(tex.data[idx + 0], tex.data[idx + 1], tex.data[idx + 2], tex.data[idx + 3]));
}

/// Implementation of \c tex::lookup_float4() for a texture_3d texture.
void tex_lookup_float4_3d(
    mi::Float32 result[4],
    const mi::neuraylib::Texture_handler_base *self,
    mi::Uint32 texture_idx,
    const mi::Float32 coord[3],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    mi::neuraylib::Tex_wrap_mode wrap_w,
    const mi::Float32 crop_u[2],
    const mi::Float32 crop_v[2],
    const mi::Float32 crop_w[2])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 1.0f;
}

/// Implementation of \c tex::lookup_float3() for a texture_3d texture.
void tex_lookup_float3_3d(
    mi::Float32 result[3],
    const mi::neuraylib::Texture_handler_base *self,
    mi::Uint32 texture_idx,
    const mi::Float32 coord[3],
    mi::neuraylib::Tex_wrap_mode wrap_u,
    mi::neuraylib::Tex_wrap_mode wrap_v,
    mi::neuraylib::Tex_wrap_mode wrap_w,
    const mi::Float32 crop_u[2],
    const mi::Float32 crop_v[2],
    const mi::Float32 crop_w[2])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
}

/// Implementation of \c tex::texel_float4() for a texture_3d texture.
void tex_texel_float4_3d(
    mi::Float32 result[4],
    const mi::neuraylib::Texture_handler_base *self,
    mi::Uint32 texture_idx,
    const mi::Sint32 coord[3])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 1.0f;
}

/// Implementation of \c tex::lookup_float4() for a texture_cube texture.
void tex_lookup_float4_cube(
    mi::Float32 result[4],
    const mi::neuraylib::Texture_handler_base *self,
    mi::Uint32 texture_idx,
    const mi::Float32 coord[3])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 1.0f;
}

/// Implementation of \c tex::lookup_float3() for a texture_cube texture.
void tex_lookup_float3_cube(
    mi::Float32 result[3],
    const mi::neuraylib::Texture_handler_base *self,
    mi::Uint32 texture_idx,
    const mi::Float32 coord[3])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
}

/// Implementation of \c resolution_2d function needed by generated code,
/// which retrieves the width and height of the given texture.
void tex_resolution_2d(
    mi::Sint32 result[2],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx,
    const mi::Sint32 uv_tile[2])
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);

    if (texture_idx == 0 || texture_idx - 1 >= self->num_textures) {
        // invalid texture returns zero
        result[0] = 0;
        result[1] = 0;
        return;
    }

    Texture const &tex = self->textures[texture_idx - 1];

    result[0] = tex.size.x;
    result[1] = tex.size.y;
}

/// Implementation of \c resolution_3d function needed by generated code,
/// which retrieves the width, height and depth of the given texture.
void tex_resolution_3d(
    mi::Sint32 result[3],
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx)
{
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
}

/// Implementation of \c tex::texture_isvalid() function.
bool tex_texture_isvalid(
    const mi::neuraylib::Texture_handler_base *self_base,
    mi::Uint32 texture_idx)
{
    Texture_handler const *self = static_cast<Texture_handler const *>(self_base);
    return texture_idx != 0 && texture_idx - 1 < self->num_textures;
}


// ------------------------------------------------------------------------------------------------
// Light Profiles (dummy functions)
// ------------------------------------------------------------------------------------------------

/// Implementation of \c df::light_profile_power() for a light profile.
/// Note: The example does not support light profiles.
mi::Float32 df_light_profile_power(
    const Texture_handler_base *self,
    mi::Uint32                  light_profile_idx)
{
    return 0.0f;
}

/// Implementation of \c df::light_profile_maximum() for a light profile.
mi::Float32 df_light_profile_maximum(
    const Texture_handler_base *self,
    mi::Uint32                  light_profile_idx)
{
    return 0.0f;
}

/// Implementation of \c df::light_profile_isvalid() for a light profile.
bool df_light_profile_isvalid(
    const Texture_handler_base *self,
    mi::Uint32                  light_profile_idx)
{
    return false;
}

/// Implementation of \c df::light_profile_evaluate() for a light profile.
mi::Float32 df_light_profile_evaluate(
    const Texture_handler_base *self,
    mi::Uint32 light_profile_idx,
    const float theta_phi[2])
{
    return 0.0f;
}

/// Implementation of \c df::light_profile_sample() for a light profile.
void df_light_profile_sample(
    mi::Float32 result[3], // theta, phi, pdf
    const Texture_handler_base *self,
    mi::Uint32 light_profile_idx,
    const float xi[3])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
}

/// Implementation of \c df::light_profile_pdf() for a light profile.
mi::Float32 df_light_profile_pdf(
    const Texture_handler_base *self,
    mi::Uint32 light_profile_idx,
    const float theta_phi[2])
{
    return 0.0f;
}


// ------------------------------------------------------------------------------------------------
// BSDF measurements (dummy functions)
// ------------------------------------------------------------------------------------------------

/// Implementation of \c df::bsdf_measurement_isvalid() for a light profile.
/// Note: The example does not support BSDF measurements.
bool df_bsdf_measurement_isvalid(
    const Texture_handler_base *self,
    mi::Uint32                  bsdf_measurement_index)
{
    return false;
}

/// Implementation of \c df::bsdf_measurement_resolution().
void df_bsdf_measurement_resolution(
    mi::Uint32 result[3],
    const Texture_handler_base *self,
    mi::Uint32 bsdf_measurement_index,
    mi::neuraylib::Mbsdf_part part)
{
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
}

/// Implementation of \c df::bsdf_measurement_evaluate() for an MBSDF.
void df_bsdf_measurement_evaluate(
    mi::Float32 result[3],
    const Texture_handler_base  *self,
    mi::Uint32 bsdf_measurement_index,
    const mi::Float32 theta_phi_in[2],
    const mi::Float32 theta_phi_out[2],
    mi::neuraylib::Mbsdf_part part)
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
}

/// Implementation of \c df::bsdf_measurement_sample() for an MBSDF.
void df_bsdf_measurement_sample(
    mi::Float32 result[3],
    const Texture_handler_base *self,
    mi::Uint32 bsdf_measurement_index,
    const mi::Float32 theta_phi_out[2],
    const mi::Float32 xi[3],
    mi::neuraylib::Mbsdf_part part)
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
}

/// Implementation of \c df::bsdf_measurement_pdf() for an MBSDF.
mi::Float32 df_bsdf_measurement_pdf(
    const Texture_handler_base *self,
    mi::Uint32 bsdf_measurement_index,
    const mi::Float32 theta_phi_in[2],
    const mi::Float32 theta_phi_out[2],
    mi::neuraylib::Mbsdf_part part)
{
    return 0.0f;
}

/// Implementation of \c df::bsdf_measurement_albedos() for an MBSDF.
void df_bsdf_measurement_albedos(
    mi::Float32 result[4],
    const Texture_handler_base *self,
    mi::Uint32 bsdf_measurement_index,
    const mi::Float32 theta_phi[2])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 0.0f;
}


// ------------------------------------------------------------------------------------------------
// Scene data (dummy functions)
// ------------------------------------------------------------------------------------------------

/// Implementation of scene_data_isvalid().
bool scene_data_isvalid(
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id)
{
    return false;
}

/// Implementation of scene_data_lookup_float4().
void scene_data_lookup_float4(
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

/// Implementation of scene_data_lookup_float3().
void scene_data_lookup_float3(
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

/// Implementation of scene_data_lookup_color().
void scene_data_lookup_color(
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

/// Implementation of scene_data_lookup_float2().
void scene_data_lookup_float2(
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

/// Implementation of scene_data_lookup_float().
float scene_data_lookup_float(
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    float const                            default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    return default_value;
}

/// Implementation of scene_data_lookup_int4().
void scene_data_lookup_int4(
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

/// Implementation of scene_data_lookup_int3().
void scene_data_lookup_int3(
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

/// Implementation of scene_data_lookup_int2().
void scene_data_lookup_int2(
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

/// Implementation of scene_data_lookup_int().
int scene_data_lookup_int(
    Texture_handler_base const            *self_base,
    Shading_state_material                *state,
    unsigned                               scene_data_id,
    int                                    default_value,
    bool                                   uniform_lookup)
{
    // just return default value
    return default_value;
}

/// Implementation of scene_data_lookup_float4() with derivatives.
void scene_data_lookup_deriv_float4(
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

/// Implementation of scene_data_lookup_float3() with derivatives.
void scene_data_lookup_deriv_float3(
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

/// Implementation of scene_data_lookup_color() with derivatives.
void scene_data_lookup_deriv_color(
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

/// Implementation of scene_data_lookup_float2() with derivatives.
void scene_data_lookup_deriv_float2(
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

/// Implementation of scene_data_lookup_float() with derivatives.
void scene_data_lookup_deriv_float(
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


// ------------------------------------------------------------------------------------------------
// Vtables
// ------------------------------------------------------------------------------------------------

mi::neuraylib::Texture_handler_vtable tex_vtable = {
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

mi::neuraylib::Texture_handler_deriv_vtable tex_deriv_vtable = {
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

#endif
