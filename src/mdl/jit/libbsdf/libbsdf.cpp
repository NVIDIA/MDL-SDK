/***************************************************************************************************
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
 **************************************************************************************************/

#include "libbsdf_internal.h"

// These functions are not implemented and will be replaced during DF instantiation
BSDF_PARAM bool get_material_thin_walled(State *state);
BSDF_PARAM float3 get_material_ior(State *state);

#include "libbsdf_utilities.h"
#include "libbsdf_multiscatter.h"
namespace multiscatter = mi::libdf::multiscatter;


namespace
{
    template<typename DF_sample_data>
    BSDF_INLINE float3 get_df_over_pdf(const DF_sample_data* data);
    template<> BSDF_INLINE float3 get_df_over_pdf(const BSDF_sample_data* data) {
        return data->bsdf_over_pdf; }
    template<> BSDF_INLINE float3 get_df_over_pdf(const EDF_sample_data* data) {
        return data->edf_over_pdf; }
    template<typename DF_sample_data>
    BSDF_INLINE void set_df_over_pdf(DF_sample_data* data, float3 df_over_pdf);
    template<> BSDF_INLINE void set_df_over_pdf(BSDF_sample_data* data, float3 df_over_pdf) {
        data->bsdf_over_pdf = df_over_pdf; }
    template<> BSDF_INLINE void set_df_over_pdf(EDF_sample_data* data, float3 df_over_pdf) {
        data->edf_over_pdf = df_over_pdf;}

    template<typename DF_evaluate_data>
    BSDF_INLINE float get_cos(const DF_evaluate_data* data);
    template<> BSDF_INLINE float get_cos(const BSDF_evaluate_data* data) { return 0.0f; }
    template<> BSDF_INLINE float get_cos(const EDF_evaluate_data* data) { return data->cos; }
    template<typename DF_evaluate_data>
    BSDF_INLINE void set_cos(DF_evaluate_data* data, float cos);
    template<> BSDF_INLINE void set_cos(BSDF_evaluate_data* data, float cos) { }
    template<> BSDF_INLINE void set_cos(EDF_evaluate_data* data, float cos) { data->cos = cos; }
    template<> BSDF_INLINE void set_cos(EDF_pdf_data* data, float cos) { }

    template<typename DF_pdf_data>
    BSDF_INLINE void set_pdf(DF_pdf_data* data, float pdf);
    template<> BSDF_INLINE void set_pdf(BSDF_pdf_data* data, float pdf) { data->pdf = pdf; }
    template<> BSDF_INLINE void set_pdf(BSDF_evaluate_data* data, float pdf) { data->pdf = pdf; }
    template<> BSDF_INLINE void set_pdf(EDF_pdf_data* data, float pdf) { data->pdf = pdf; }
    template<> BSDF_INLINE void set_pdf(EDF_evaluate_data* data, float pdf) { data->pdf = pdf; }
}


/////////////////////////////////////////////////////////////////////
// general bsdf
/////////////////////////////////////////////////////////////////////

BSDF_INLINE void add_elemental_bsdf_evaluate_contribution(
    BSDF_evaluate_data *data,
    const int handle,
    const float3 &diffuse,
    const float3 &glossy)
{
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        data->bsdf_diffuse += diffuse;
        data->bsdf_glossy += glossy;
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        const int index = handle - data->handle_offset;
        if (index >= 0 && index < data->handle_count) {
            data->bsdf_diffuse[index] += diffuse;
            data->bsdf_glossy[index] += glossy;
        }
    #else
        const int index = handle - data->handle_offset;
        if (index >= 0 && index < MDL_DF_HANDLE_SLOT_MODE) {
            data->bsdf_diffuse[index] += diffuse;
            data->bsdf_glossy[index] += glossy;
        }
    #endif
}

BSDF_INLINE void add_elemental_bsdf_auxiliary_contribution(
    BSDF_auxiliary_data *data,
    const int handle,
    const float3 &albedo,
    const float3 &normal)
{
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        data->albedo += albedo;
        data->normal += normal;
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
    const int index = handle - data->handle_offset;
    if (index >= 0 && index < data->handle_count)
        data->albedo[index] += albedo;
        data->normal[index] += normal;
    #else
    const int index = handle - data->handle_offset;
    if (index >= 0 && index < MDL_DF_HANDLE_SLOT_MODE)
        data->albedo[index] += albedo;
        data->normal[index] += normal;
    #endif
    // (safe) normalization has to happen before reaching the data back to application
}

BSDF_INLINE void add_elemental_edf_evaluate_contribution(
    EDF_evaluate_data *data,
    const int handle,
    const float3 &edf)
{
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        data->edf += edf;
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        const int index = handle - data->handle_offset;
        if (index >= 0 && index < data->handle_count)
            data->edf[index] += edf;
    #else
        const int index = handle - data->handle_offset;
        if (index >= 0 && index < MDL_DF_HANDLE_SLOT_MODE) {
            data->edf[index] += edf;
        }
    #endif
}

BSDF_INLINE void diffuse_sample(
    BSDF_sample_data *data,
    State *state,
    const Geometry &g,
    const float3 &tint,
    const float roughness,
    const bool transmit,
    const int handle)
{
    // sample direction and transform to world coordinates
    const float3 cosh = cosine_hemisphere_sample(make_float2(data->xi.x, data->xi.y));

    const float sign = transmit ? -1.0f : 1.0f;

    data->k2 = math::normalize(
        g.x_axis * cosh.x + sign * g.n.shading_normal * cosh.y + g.z_axis * cosh.z);

    if (cosh.y <= 0.0f || sign * math::dot(data->k2, g.n.geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }

    data->bsdf_over_pdf = tint;
    if (roughness > 0.0f)
        data->bsdf_over_pdf *= eval_oren_nayar(data->k2, data->k1, g.n.shading_normal, roughness);

    data->pdf = cosh.y * (float)(1.0 / M_PI);
    data->event_type = transmit ? BSDF_EVENT_DIFFUSE_TRANSMISSION : BSDF_EVENT_DIFFUSE_REFLECTION;
    data->handle = handle;
}

BSDF_INLINE void diffuse_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const float roughness,
    const bool transmit,
    const int handle)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float sign = transmit ? -1.0f : 1.0f;
    const float nk2 = math::max(sign * math::dot(data->k2, shading_normal), 0.0f);
    const float pdf = nk2 * (float)(1.0f / M_PI);

    float3 bsdf_diffuse = pdf * tint;
    if (nk2 > 0.0f && roughness > 0.0f)
        bsdf_diffuse *= eval_oren_nayar(data->k2, data->k1, shading_normal, roughness);

    add_elemental_bsdf_evaluate_contribution(
        data, handle, bsdf_diffuse * inherited_weight, make<float3>(0.0f));
    data->pdf = pdf;
}

template <typename Data>
BSDF_INLINE void diffuse_pdf(
    Data *data,
    State *state,
    const float3 &inherited_normal,
    const bool transmit)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float sign = transmit ? -1.0f : 1.0f;
    const float nk2 = math::max(sign * math::dot(data->k2, shading_normal), 0.0f);
    data->pdf = nk2 * (float)(1.0f / M_PI);
}

template <typename Data>
BSDF_INLINE void elemental_bsdf_auxiliary(
    Data *data,
    State *state, 
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        absorb(data);
        return;
    }

    const float3 albedo = math::saturate(tint);
    add_elemental_bsdf_auxiliary_contribution(
        data, 
        handle, 
        inherited_weight * albedo, 
        math::average(inherited_weight) * g.n.shading_normal);
}

/////////////////////////////////////////////////////////////////////
// Hair BSDFs
/////////////////////////////////////////////////////////////////////
#include "libbsdf_hair.h"

/////////////////////////////////////////////////////////////////////
// bsdf()
/////////////////////////////////////////////////////////////////////

BSDF_API void black_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal)
{
    absorb(data);
}

BSDF_API void black_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight)
{
    absorb(data);
}

BSDF_API void black_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal)
{
    absorb(data);
}

BSDF_API void black_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const int handle)
{
    elemental_bsdf_auxiliary(
        data, state, inherited_normal, inherited_weight, make<float3>(0.0f), handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf diffuse_reflection_bsdf(
//     color           tint      = color(1.0),
//     float           roughness = 0.0,
//     uniform string  handle    = ""
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void diffuse_reflection_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const float roughness,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        absorb(data);
        return;
    }

    diffuse_sample(data, state, g, math::saturate(tint), roughness, false, handle);
}

BSDF_API void diffuse_reflection_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const float roughness,
    const int handle)
{
    diffuse_evaluate(
        data, state, inherited_normal, inherited_weight, 
        math::saturate(tint), roughness, false, handle);
}

BSDF_API void diffuse_reflection_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const float roughness,
    const int handle)
{
    diffuse_pdf(data, state, inherited_normal, false);
}

BSDF_API void diffuse_reflection_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const float roughness,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf diffuse_transmission_bsdf(
//     color           tint   = color(1.0),
//     uniform string  handle = ""
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void diffuse_transmission_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        absorb(data);
        return;
    }

    diffuse_sample(data, state, g, math::saturate(tint), 0.0f, true, handle);
}

BSDF_API void diffuse_transmission_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const int handle)
{
    diffuse_evaluate(
        data, state, inherited_normal, inherited_weight, 
        math::saturate(tint), 0.0f, true, handle);
}

BSDF_API void diffuse_transmission_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const int handle)
{
    diffuse_pdf(data, state, inherited_normal, true);
}

BSDF_API void diffuse_transmission_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf specular_bsdf(
//     color           tint = color(1.0),
//     scatter_mode    mode = scatter_reflect,
//     uniform string  handle = ""
// )
/////////////////////////////////////////////////////////////////////

BSDF_INLINE void specular_sample(
    BSDF_sample_data *data,
    State *state,
    const Normals &n,
    const float2 &ior,
    const bool thin_walled,
    const float3 &tint,
    const scatter_mode mode,
    const int handle)
{
    const float nk1 = math::dot(n.shading_normal, data->k1);
    if (nk1 < 0.0f) {
        absorb(data);
        return;
    }

    data->bsdf_over_pdf = tint;
    data->pdf = 0.0f;

    // reflection
    if ((mode == scatter_reflect) ||
        ((mode == scatter_reflect_transmit) &&
         data->xi.x < ior_fresnel(ior.y / ior.x,  nk1)))
    {
        data->k2 = (nk1 + nk1) * n.shading_normal - data->k1;

        data->event_type = BSDF_EVENT_SPECULAR_REFLECTION;
    } else {
        // refraction
        // total internal reflection should only be triggered for scatter_transmit
        // (since we should fall in the code-path above otherwise)
        bool tir = false;
        if (thin_walled) // single-sided -> propagate old direction
            data->k2 = -data->k1;
        else
            data->k2 = refract(data->k1, n.shading_normal, ior.x / ior.y, nk1, tir);

        data->event_type = tir ? BSDF_EVENT_SPECULAR_REFLECTION : BSDF_EVENT_SPECULAR_TRANSMISSION;
    }

    // check if the resulting direction is on the correct side of the actual geometry
    const float gnk2 = math::dot(data->k2, n.geometry_normal) * (
        data->event_type == BSDF_EVENT_SPECULAR_REFLECTION ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f) {
        absorb(data);
        return;
    }
    data->handle = handle;
}

BSDF_API void specular_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const scatter_mode mode,
    const int handle)
{
    Normals n;
    get_oriented_normals(
        n.shading_normal, n.geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float2 ior = process_ior(data, state);
    const bool thin_walled = get_material_thin_walled(state);

    specular_sample(data, state, n, ior, thin_walled, math::saturate(tint), mode, handle);
}

BSDF_API void specular_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const scatter_mode mode,
    const int handle)
{
    absorb(data);
}

BSDF_API void specular_bsdf_pdf(
    BSDF_pdf_data *data,
    const State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const scatter_mode mode,
    const int handle)
{
    absorb(data);
}

BSDF_API void specular_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const scatter_mode mode,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


//
// Most glossy BSDF models in MDL are microfacet-theory based along the lines of
// "Bruce Walter, Stephen R. Marschner, Hongsong Li, Kenneth E. Torrance - Microfacet Models For
// Refraction through Rough Surfaces" and "Eric Heitz - Understanding the Masking-Shadowing
// Function in Microfacet-Based BRDFs"
//
// The common utility code uses "Distribution", which has to provide:
// sample():      importance sample visible microfacet normals (i.e. including masking)
// eval():        evaluate microfacet distribution
// mask():        compute masking
// shadow_mask(): combine masking for incoming and outgoing directions
//

template <typename Distribution>
BSDF_INLINE void microfacet_sample(
    const Distribution &ph,
    BSDF_sample_data *data,
    State *state,
    const Geometry &g,
    const float2 &ior,
    const bool thin_walled,
    const scatter_mode mode,
    const int handle,
    float& nk1)
{
    nk1 = math::abs(math::dot(data->k1, g.n.shading_normal));

    const float3 k10 = make_float3(
        math::dot(data->k1, g.x_axis),
        nk1,
        math::dot(data->k1, g.z_axis));

    // sample half vector / microfacet normal
    const float3 h0 = ph.sample(data->xi, k10);
    if (math::abs(h0.y) == 0.0f)
    {
        absorb(data);
        return;
    }

    // transform to world
    const float3 h = g.n.shading_normal * h0.y + g.x_axis * h0.x + g.z_axis * h0.z;
    const float kh = math::dot(data->k1, h);

    if (kh <= 0.0f) {
        absorb(data);
        return;
    }

    // compute probability of selection refraction over reflection
    float f_refl;
    switch (mode) {
        case scatter_reflect:
            f_refl = 1.0f;
            break;
        case scatter_transmit:
            f_refl = 0.0f;
            break;
        case scatter_reflect_transmit:
            f_refl = ior_fresnel(ior.y / ior.x, kh);
            break;
    }

    if ((mode == scatter_reflect) ||
        ((mode == scatter_reflect_transmit) &&
         data->xi.z < f_refl))
    {
        // BRDF: reflect
        data->k2 = (2.0f * kh) * h - data->k1;
        data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
        data->xi.z /= f_refl;
    } else {
        bool tir = false;
        if (thin_walled) {
            // pseudo-BTDF: flip a reflected reflection direction to the back side
            data->k2 = (2.0f * kh) * h - data->k1;
            data->k2 = math::normalize(
                data->k2 - 2.0f * g.n.shading_normal * math::dot(data->k2, g.n.shading_normal));
        } else {
            // BTDF: refract
            data->k2 = refract(data->k1, h, ior.x / ior.y, kh, tir);
        }
        data->event_type = tir ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;
        data->xi.z = (data->xi.z - f_refl) / (1.0f - f_refl);
    }

    // check if the resulting direction is on the correct side of the actual geometry
    const float gnk2 = math::dot(data->k2, g.n.geometry_normal) * (
        data->event_type == BSDF_EVENT_GLOSSY_REFLECTION ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f) {
        absorb(data);
        return;
    }

    const bool refraction = !thin_walled && (data->event_type == BSDF_EVENT_GLOSSY_TRANSMISSION);

    // compute weight
    data->bsdf_over_pdf = make<float3>(1.0f);

    const float nk2 = math::abs(math::dot(data->k2, g.n.shading_normal));
    const float k2h = math::abs(math::dot(data->k2, h));

    float G1, G2;
    const float G12 = ph.shadow_mask(
        G1, G2, h0.y,
        k10, kh,
        make_float3(math::dot(data->k2, g.x_axis), nk2, math::dot(data->k1, g.z_axis)), k2h,
        refraction);

    if (G12 <= 0.0f) {
        absorb(data);
        return;
    }
    data->bsdf_over_pdf *= G12 / G1;

    // compute pdf
    {
        data->pdf = ph.eval(h0) * G1;

        if (refraction) {
            const float tmp = kh * ior.x - k2h * ior.y;
            data->pdf *= kh * k2h / (nk1 * h0.y * tmp * tmp);
        }
        else
            data->pdf *= 0.25f / (nk1 * h0.y);
    }
    data->handle = handle;
}

template <typename Distribution, typename Data>
BSDF_INLINE float microfacet_evaluate(
    const Distribution &ph,
    Data *data,
    State *state,
    const Geometry &g,
    const float2 &ior,
    const bool thin_walled,
    const scatter_mode mode,
    float &nk1,
    float &nk2)
{
    // BTDF or BRDF eval?
    const bool backside_eval = math::dot(data->k2, g.n.geometry_normal) < 0.0f;

    // nothing to evaluate for given directions?
    if (backside_eval && (mode == scatter_reflect)) {
        absorb(data);
        return 0.0f;
    }

    nk1 = math::abs(math::dot(g.n.shading_normal, data->k1));
    nk2 = math::abs(math::dot(g.n.shading_normal, data->k2));

    const float3 h = compute_half_vector(
        data->k1, data->k2, g.n.shading_normal, ior, nk2,
        backside_eval, thin_walled);

    // invalid for reflection / refraction?
    const float nh = math::dot(g.n.shading_normal, h);
    const float k1h = math::dot(data->k1, h);
    const float k2h = math::dot(data->k2, h) * (backside_eval ? -1.0f : 1.0f);
    if (nh < 0.0f || k1h < 0.0f || k2h < 0.0f) {
        absorb(data);
        return 0.0f;
    }

    const float fresnel_refl = ior_fresnel(ior.y / ior.x, k1h);

    // for scatter_transmit: only allow TIR (fresnel_refl == 1) with BRDF eval
    if (!backside_eval && (mode == scatter_transmit) && (fresnel_refl < 1.0f)) {
        absorb(data);
        return 0.0f;
    }

    // compute BSDF and pdf
    const float weight = mode == scatter_reflect_transmit ?
                         (backside_eval ? (1.0f - fresnel_refl) : fresnel_refl) :
                         1.0f;

    data->pdf = ph.eval(make_float3(math::dot(g.x_axis, h), nh, math::dot(g.z_axis, h)));

    float G1, G2;
    //const float k2h = math::abs(math::dot(data->k2, h));
    const bool refraction = !thin_walled && backside_eval;
    const float G12 = ph.shadow_mask(
        G1, G2, nh,
        make_float3(math::dot(g.x_axis, data->k1), nk1, math::dot(g.z_axis, data->k1)), k1h,
        make_float3(math::dot(g.x_axis, data->k2), nk2, math::dot(g.z_axis, data->k2)), k2h,
        refraction);

    if (refraction) {
        // refraction pdf and BTDF
        const float tmp = k1h * ior.x - k2h * ior.y;
        data->pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
    } else {
        // reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
        data->pdf *= 0.25f / (nk1 * nh);
    }

    const float bsdf = data->pdf * weight * G12;
    data->pdf *= G1;
    return bsdf;
}

template <typename Distribution>
BSDF_INLINE void microfacet_sample(
    const Distribution &ph,
    BSDF_sample_data *data,
    State *state,
    const Geometry& g,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const scatter_mode mode,
    const int handle,
    const BSDF_type type,
    const float roughness_u,
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    const float2 ior = process_ior(data, state);
    const bool thin_walled = get_material_thin_walled(state);

    // sample the single scattering (glossy) BSDF
    float nk1;
    microfacet_sample(ph, data, state, g, ior, thin_walled, mode, handle, nk1);

    if (multiscatter_texture_id == 0 || mode == scatter_transmit) {
        data->bsdf_over_pdf *= tint;
        return;
    }

    // sample, in case the multi-scattering part is sampled, k2 will change and rho1 will be > 0
    const float rho1 = multiscatter::sample(
        state, type, roughness_u, roughness_v, nk1,
        (!thin_walled && (mode != scatter_reflect)) ? (ior.x / ior.y) : -1.0f, multiscatter_texture_id,
        data, g, tint, multiscatter_tint);

    // recompute glossy pdf for new direction
    if (rho1 > 0.0f)
    {
        BSDF_pdf_data pdf_data = to_pdf_data(data);
        float nk2;
        microfacet_evaluate(ph, &pdf_data, state, g, ior, thin_walled, mode, nk1, nk2);

        // incorporate multi-scatter part to pdf for the new k2
        multiscatter::sample_update_single_scatter_probability(data, pdf_data.pdf, rho1);
    }
}

template <typename Distribution>
BSDF_INLINE void microfacet_sample(
    const Distribution &ph,
    BSDF_sample_data *data,
    State *state,
    const float3 &normal,
    const float3 &tangent_u,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const scatter_mode mode,
    const int handle,
    const BSDF_type type,
    const float roughness_u, 
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    Geometry g;
    if (!get_geometry(g, normal, tangent_u, data->k1, state))
    {
        absorb(data);
        return;
    }

    microfacet_sample(
        ph, data, state, g, tint, multiscatter_tint, mode, handle, type,
        roughness_u, roughness_v, multiscatter_texture_id);
}

template <typename Distribution>
BSDF_INLINE float2 microfacet_evaluate(
    const Distribution &ph,
    BSDF_evaluate_data *data,
    State *state,
    const Geometry& g,
    const scatter_mode mode,
    const BSDF_type type,
    const float roughness_u,
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    const float2 ior = process_ior(data, state);
    const bool thin_walled = get_material_thin_walled(state);

    float nk1, nk2;
    const float contrib_single = 
        microfacet_evaluate(ph, data, state, g, ior, thin_walled, mode, nk1, nk2);

    if (multiscatter_texture_id == 0 || mode == scatter_transmit)
        return make<float2>(contrib_single, 0.0f);

    float2 contrib_multi = multiscatter::evaluate(
        state, type, roughness_u, roughness_v, nk1, nk2,
        (!thin_walled && (mode != scatter_reflect)) ? (ior.x / ior.y) : -1.0f, multiscatter_texture_id);

    data->pdf *= contrib_multi.x; // * rho1
    if (math::dot(g.n.geometry_normal, data->k2) >= 0.0f) 
    {
        data->pdf += (1.0f - contrib_multi.x) * (float)(1.0 / M_PI);
    }
    else
        contrib_multi.y = 0.0f; // backside eval

    return make<float2>(contrib_single, contrib_multi.y);
}

template <typename Distribution>
BSDF_INLINE float2 microfacet_evaluate(
    const Distribution &ph,
    BSDF_evaluate_data *data,
    State *state,
    const float3 &normal,
    const float3 &tangent_u,
    const scatter_mode mode,
    const BSDF_type type,
    const float roughness_u, 
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    Geometry g;
    if (!get_geometry(g, normal, tangent_u, data->k1, state)) {
        absorb(data);
        return make<float2>(0.0f);
    }
    return microfacet_evaluate(
        ph, data, state, g, mode, type, roughness_u, roughness_v, multiscatter_texture_id);
}

template <typename Distribution>
BSDF_INLINE void microfacet_pdf(
    const Distribution &ph,
    BSDF_pdf_data *data,
    State *state,
    const Geometry& g,
    const scatter_mode mode,
    const BSDF_type type,
    const float roughness_u,
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    const float2 ior = process_ior(data, state);
    const bool thin_walled = get_material_thin_walled(state);

    float nk1, nk2;
    microfacet_evaluate(ph, data, state, g, ior, thin_walled, mode, nk1, nk2);

    if (multiscatter_texture_id == 0 || mode == scatter_transmit)
        return;

    data->pdf = multiscatter::pdf(
        data->pdf,
        state, type, roughness_u, roughness_v, nk1, nk2,
        (!thin_walled && (mode != scatter_reflect)) ? (ior.x / ior.y) : -1.0f, multiscatter_texture_id);
}

template <typename Distribution>
BSDF_INLINE void microfacet_pdf(
    const Distribution &ph,
    BSDF_pdf_data *data,
    State *state,
    const float3 &normal,
    const float3 &tangent_u,
    const scatter_mode mode,
    const BSDF_type type,
    const float roughness_u,
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    Geometry g;
    if (!get_geometry(g, normal, tangent_u, data->k1, state))
    {
        absorb(data);
        return;
    }
    microfacet_pdf(
        ph, data, state, g, mode, type, roughness_u, roughness_v, multiscatter_texture_id);
}

template <typename Distribution>
BSDF_INLINE void microfacet_pdf(
    const Distribution &ph,
    BSDF_pdf_data *data,
    State *state,
    const float3 &normal,
    const scatter_mode mode,
    const BSDF_type type,
    const float roughness_u, 
    const float roughness_v,
    const unsigned multiscatter_texture_id)
{
    Geometry g;
    get_oriented_normals(
        g.n.shading_normal, g.n.geometry_normal, normal, state->geometry_normal(), data->k1);
    microfacet_pdf(
        ph, data, state, g, mode, type, roughness_u, roughness_v, multiscatter_texture_id);
}

/////////////////////////////////////////////////////////////////////
// bsdf simple_glossy_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     scatter_mode    mode              = scatter_reflect,
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////


// v-cavities masking and and importance sampling utility
// (see "Eric Heitz and Eugene d'Eon - Importance Sampling Microfacet-Based BSDFs with the
// Distribution of Visible Normals")
class Vcavities_masking {
public:
    BSDF_INLINE float3 flip(const float3 &h, const float3 &k, float &xi) const
    {
        const float a = h.y * k.y;
        const float b = h.x * k.x + h.z * k.z;
        const float kh   = math::max(a + b, 0.0f);
        const float kh_f = math::max(a - b, 0.0f);

        const float p_flip = kh_f / (kh + kh_f);
        if (xi < p_flip) {
            xi /= p_flip;
            return make_float3(-h.x, h.y, -h.z);
        } else {
            xi = (xi - p_flip) / (1.0f - p_flip);
            return h;
        }
    }

    BSDF_INLINE float shadow_mask(
        float &G1, float &G2,
        const float nh,
        const float3 &k1, const float k1h,
        const float3 &k2, const float k2h,
        const bool refraction) const
    {
        G1 = microfacet_mask_v_cavities(nh, k1h, k1.y);
        G2 = microfacet_mask_v_cavities(nh, k2h, k2.y);
        return refraction ?  math::max(G1 + G2 - 1.0f, 0.0f) : math::min(G1, G2);
    }
};

// simple_glossy_bsdf uses a v-cavities-masked Phong distribution
class Distribution_phong_vcavities : public Vcavities_masking {
public:
    BSDF_INLINE Distribution_phong_vcavities(const float roughness_u, const float roughness_v) {
        m_exponent = roughness_to_exponent(
            clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    BSDF_INLINE float3 sample(float4 &xi, const float3 &k) const {
        return flip(hvd_phong_sample(make_float2(xi.x, xi.y), m_exponent), k, xi.z);
    }

    BSDF_INLINE float eval(const float3 &h) const {
        return hvd_phong_eval(m_exponent, h.y, h.x, h.z);
    }

private:
    float2 m_exponent;
};

BSDF_API void simple_glossy_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_phong_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_SIMPLE_GLOSSY_MULTISCATTER);

    microfacet_sample(
        ph, data, state, inherited_normal, tangent_u, 
        math::saturate(tint), math::saturate(multiscatter_tint), mode, handle,
        SIMPLE_GLOSSY_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void simple_glossy_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_phong_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_SIMPLE_GLOSSY_MULTISCATTER);

    const float2 contrib = microfacet_evaluate(
        ph, data, state, inherited_normal, tangent_u, mode,
        SIMPLE_GLOSSY_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);

    add_elemental_bsdf_evaluate_contribution(
        data, handle, 
        contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        contrib.x * math::saturate(tint) * inherited_weight);
}


BSDF_API void simple_glossy_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_phong_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_SIMPLE_GLOSSY_MULTISCATTER);

    microfacet_pdf(ph, data, state, inherited_normal, tangent_u, mode,
        SIMPLE_GLOSSY_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void simple_glossy_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf sheen_bsdf(
//     float           roughness,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////

// the sheen bsdf uses a v-cavities-masked sheen (sin^k) distribution
class Distribution_sheen_vcavities : public Vcavities_masking
{
public:
    BSDF_INLINE Distribution_sheen_vcavities(const float roughness)
    {
        m_inv_roughness = 1.0f / clamp_roughness(roughness);
    }

    BSDF_INLINE float3 sample(float4 &xi, const float3 &k) const
    {
        return flip(hvd_sheen_sample(make_float2(xi.x, xi.y), m_inv_roughness), k, xi.z);
    }

    BSDF_INLINE float eval(const float3 &h) const
    {
        return hvd_sheen_eval(m_inv_roughness, h.y);
    }

private:
    float m_inv_roughness;
};


BSDF_API void sheen_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const int handle)
{
    const float adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness, roughness)).x;
    const Distribution_sheen_vcavities ph(adapted_roughness);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_SHEEN_MULTISCATTER);

    microfacet_sample(
        ph, data, state, inherited_normal, state->texture_tangent_u(0),
        math::saturate(tint), math::saturate(multiscatter_tint),
        scatter_reflect, handle, SHEEN_BSDF,
        adapted_roughness, adapted_roughness, multiscatter_texture_id);
}

BSDF_API void sheen_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const int handle)
{
    const float adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness, roughness)).x;
    const Distribution_sheen_vcavities ph(adapted_roughness);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_SHEEN_MULTISCATTER);

    const float2 contrib = microfacet_evaluate(
        ph, data, state, inherited_normal, state->texture_tangent_u(0), scatter_reflect,
        SHEEN_BSDF, adapted_roughness, adapted_roughness, multiscatter_texture_id);

    add_elemental_bsdf_evaluate_contribution(
        data, handle,
        contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        contrib.x * math::saturate(tint) * inherited_weight);
}


BSDF_API void sheen_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const int handle)
{
    const float adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness, roughness)).x;
    const Distribution_sheen_vcavities ph(adapted_roughness);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_SHEEN_MULTISCATTER);

    microfacet_pdf(
        ph, data, state, inherited_normal, scatter_reflect,
        SHEEN_BSDF, adapted_roughness, adapted_roughness, multiscatter_texture_id);
}

BSDF_API void sheen_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf backscattering_glossy_reflection_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////

// the backscattering glossy BRDF is inspired by a BRDF model published in
// "The Halfway Vector Disk for BRDF modelling" by Dave Edwards et al.
//
// - it uses a half vector distribution centered around incoming direction constructed using
//   the "scaling projection" and a distribution on the unit disk
// - the distribution is made symmetric by using the minimum of such a half vector distribution
//   constructed from both incoming and outgoing directions
// - further, for the normalization term symmetry is obtained by replacing a division by nk2
//   with a division by max(nk1, nk2)

BSDF_INLINE void backscattering_glossy_sample(
    BSDF_sample_data *data,
    State *state,
    const Geometry &g,
    const float roughness_u,
    const float roughness_v,
    const int handle,
    float &nk1)
{
    const float2 exponent = roughness_to_exponent(
        clamp_roughness(roughness_u), clamp_roughness(roughness_v));

    // sample half vector
    const float2 u = sample_disk_distribution(make_float2(data->xi.x, data->xi.y), exponent);

    const float xk1 = math::dot(g.x_axis, data->k1);
    const float zk1 = math::dot(g.z_axis, data->k1);
    const float2 u1 = u + make_float2(xk1, zk1);

    nk1 = math::dot(data->k1, g.n.shading_normal);
    const float3 h = math::normalize(g.n.shading_normal * nk1 + g.x_axis * u1.x + g.z_axis * u1.y);

    // compute reflection direction
    const float kh = math::dot(data->k1, h);
    data->k2 = h * (kh + kh) - data->k1;

    // check if the resulting direction is on the correct side of the actual geometry
    const float nk2 = math::dot(data->k2, g.n.shading_normal);
    if (nk2 <= 0.0f || math::dot(data->k2, g.n.geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }

    // compute weight and pdf
    const float nh = math::dot(g.n.shading_normal, h);
    const float inv_nh = 1.0f / nh;
    const float nk1_nh = nk1 * inv_nh;
    const float ph1 = eval_disk_distribution(u.x, u.y, exponent) * nk1_nh * nk1_nh * inv_nh;

    const float nk2_nh = nk2 * inv_nh;
    const float xh = math::dot(g.x_axis, h);
    const float zh = math::dot(g.z_axis, h);
    const float x2 = nk2_nh * xh;
    const float y2 = nk2_nh * zh;
    const float xk2 = math::dot(g.x_axis, data->k2);
    const float zk2 = math::dot(g.z_axis, data->k2);
    const float ph2 = eval_disk_distribution(
        x2 - xk2, y2 - zk2, exponent) * nk2_nh * nk2_nh * inv_nh;

    data->bsdf_over_pdf = make<float3>(
        nk2 * math::min(ph1, ph2) / (ph1 * math::max(nk1, nk2)));
    data->pdf = ph1 * 0.25f / kh;

    data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    data->handle = handle;
}

BSDF_INLINE float backscattering_glossy_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const Geometry &g,
    const float roughness_u,
    const float roughness_v,
    float &nk1,
    float &nk2)
{
    nk1 = math::dot(data->k1, g.n.shading_normal);
    nk2 = math::dot(data->k2, g.n.shading_normal);
    if (nk2 <= 0.0f) {
        absorb(data);
        return 0.0f;
    }

    const float2 exponent = roughness_to_exponent(
        clamp_roughness(roughness_u), clamp_roughness(roughness_v));

    const float3 h = math::normalize(data->k1 + data->k2);

    const float nh = math::dot(g.n.shading_normal, h);
    const float xh = math::dot(g.x_axis, h);
    const float zh = math::dot(g.z_axis, h);

    const float inv_nh = 1.0f / nh;
    const float nk1_nh = nk1 * inv_nh;
    const float nk2_nh = nk2 * inv_nh;
    const float x1 = nk1_nh * xh;
    const float y1 = nk1_nh * zh;
    const float x2 = nk2_nh * xh;
    const float y2 = nk2_nh * zh;

    const float xk1 = math::dot(data->k1, g.x_axis);
    const float zk1 = math::dot(data->k1, g.z_axis);
    const float ph1 = eval_disk_distribution(
        x1 - xk1, y1 - zk1, exponent) * nk1_nh * nk1_nh * inv_nh;

    const float xk2 = math::dot(data->k2, g.x_axis);
    const float zk2 = math::dot(data->k2, g.z_axis);
    const float ph2 = eval_disk_distribution(
        x2 - xk2, y2 - zk2, exponent) * nk2_nh * nk2_nh * inv_nh;

    const float kh = math::dot(data->k1, h);
    const float f = (0.25f / kh);
    data->pdf = f * ph1;

    return (f * math::min(ph1, ph2) * nk2 / math::max(nk2, nk1));
}

BSDF_API void backscattering_glossy_reflection_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, tangent_u, data->k1, state)) {
        absorb(data);
        return;
    }

    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    float nk1, nk2;
    const float glossy_contrib = backscattering_glossy_evaluate(
        data, state, g, adapted_roughness.x, adapted_roughness.y, nk1, nk2);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BACKSCATTERING_GLOSSY_MULTISCATTER);

    float2 multiscatter_contrib;
    if (multiscatter_texture_id == 0)
    {
        multiscatter_contrib = make<float2>(0.0f);
    }
    else
    {
        multiscatter_contrib = multiscatter::evaluate(
            state, BACKSCATTERING_GLOSSY_BSDF,
            adapted_roughness.x, adapted_roughness.y,
            nk1, nk2, -1.0f, multiscatter_texture_id);

        data->pdf *= multiscatter_contrib.x; // * rho1
        if (math::dot(g.n.geometry_normal, data->k2) >= 0.0f)
        {
            data->pdf += (1.0f - multiscatter_contrib.x) * (float)(1.0 / M_PI);
        }
        else
            multiscatter_contrib.y = 0.0f; // backside eval
    }

    add_elemental_bsdf_evaluate_contribution(
        data, handle,
        multiscatter_contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        glossy_contrib * math::saturate(tint) * inherited_weight);
}

template <typename Data>
BSDF_INLINE void backscattering_glossy_pdf(
    Data *data,
    State *state,
    const Geometry &g,
    const float roughness_u,
    const float roughness_v,
    float &nk1,
    float &nk2)
{
    nk1 = math::dot(data->k1, g.n.shading_normal);
    nk2 = math::dot(data->k2, g.n.shading_normal);
    if (nk2 <= 0.0f) {
        absorb(data);
        return;
    }

    const float3 h = math::normalize(data->k1 + data->k2);

    const float nh = math::dot(g.n.shading_normal, h);
    const float xh = math::dot(g.x_axis, h);
    const float zh = math::dot(g.z_axis, h);

    const float inv_nh = 1.0f / nh;
    const float nk1_nh = nk1 * inv_nh;
    const float x1 = nk1_nh * xh;
    const float y1 = nk1_nh * zh;

    const float xk1 = math::dot(data->k1, g.x_axis);
    const float zk1 = math::dot(data->k1, g.z_axis);
    const float2 exponent = roughness_to_exponent(
        clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    const float ph1 = eval_disk_distribution(
        x1 - xk1, y1 - zk1, exponent) * nk1_nh * nk1_nh * inv_nh;

    const float kh = math::dot(data->k1, h);
    data->pdf = (0.25f / kh) * nk2 * ph1;
}


BSDF_API void backscattering_glossy_reflection_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, tangent_u, data->k1, state))
    {
        absorb(data);
        return;
    }

    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    // sample the single scattering (glossy) bsdf
    float nk1, nk2;
    backscattering_glossy_sample(
        data, state, g, adapted_roughness.x, adapted_roughness.y, handle, nk1);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BACKSCATTERING_GLOSSY_MULTISCATTER);

    if (multiscatter_texture_id == 0) {
        data->bsdf_over_pdf *= tint;
        return;
    }

    // sample, in case the multi-scattering part is sampled, k2 will change and rho1 will be > 0
    const float rho1 = multiscatter::sample(
        state, BACKSCATTERING_GLOSSY_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        nk1, -1.0f, multiscatter_texture_id, data, g, tint, multiscatter_tint);

    // recompute glossy pdf for new direction
    if (rho1 > 0.0f)
    {
        BSDF_pdf_data pdf_data = to_pdf_data(data);
        backscattering_glossy_pdf(
            &pdf_data, state, g, adapted_roughness.x, adapted_roughness.y, nk1, nk2);

        // incorporate multi-scatter part to pdf for the new k2
        multiscatter::sample_update_single_scatter_probability(data, pdf_data.pdf, rho1);
    }
}

BSDF_API void backscattering_glossy_reflection_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, tangent_u, data->k1, state)) {
        absorb(data);
        return;
    }

    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    float nk1, nk2;
    backscattering_glossy_pdf(data, state, g, adapted_roughness.x, adapted_roughness.y, nk1, nk2);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BACKSCATTERING_GLOSSY_MULTISCATTER);

    if (multiscatter_texture_id == 0)
        return;

    data->pdf = multiscatter::pdf(
        data->pdf,
        state, BACKSCATTERING_GLOSSY_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        nk1, nk2, -1.0f, multiscatter_texture_id);
}

BSDF_API void backscattering_glossy_reflection_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_beckmann_vcavities_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     scatter_mode    mode              = scatter_reflect,
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////

class Distribution_beckmann_vcavities : public Vcavities_masking {
public:
    BSDF_INLINE Distribution_beckmann_vcavities(const float roughness_u, const float roughness_v) {
        m_inv_roughness = make_float2(
            1.0f / clamp_roughness(roughness_u), 1.0f / clamp_roughness(roughness_v));
    }

    BSDF_INLINE float3 sample(float4 &xi, const float3 &k) const {
        return flip(hvd_beckmann_sample(make_float2(xi.x, xi.y), m_inv_roughness), k, xi.z);
    }

    BSDF_INLINE float eval(const float3 &h) const {
        return hvd_beckmann_eval(m_inv_roughness, h.y, h.x, h.z);
    }

private:
    float2 m_inv_roughness;
};


BSDF_API void microfacet_beckmann_vcavities_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_beckmann_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BECKMANN_VC_MULTISCATTER);

    microfacet_sample(
        ph, data, state, inherited_normal, tangent_u, 
        math::saturate(tint), math::saturate(multiscatter_tint), mode, handle,
        MICROFACET_BECKMANN_VCAVITIES_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_beckmann_vcavities_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_beckmann_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BECKMANN_VC_MULTISCATTER);

    const float2 contrib = microfacet_evaluate(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_BECKMANN_VCAVITIES_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);

    add_elemental_bsdf_evaluate_contribution(
        data, handle,
        contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        contrib.x * math::saturate(tint) * inherited_weight);
}


BSDF_API void microfacet_beckmann_vcavities_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_beckmann_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BECKMANN_VC_MULTISCATTER);

    microfacet_pdf(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_BECKMANN_VCAVITIES_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_beckmann_vcavities_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_ggx_vcavities_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     scatter_mode    mode              = scatter_reflect,
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////

class Distribution_ggx_vcavities : public Vcavities_masking {
public:
    BSDF_INLINE Distribution_ggx_vcavities(const float roughness_u, const float roughness_v) {
        m_inv_roughness = make_float2(
            1.0f / clamp_roughness(roughness_u), 1.0f / clamp_roughness(roughness_v));
    }

    BSDF_INLINE float3 sample(float4 &xi, const float3 &k) const {
        return flip(hvd_ggx_sample(make_float2(xi.x, xi.y), m_inv_roughness), k, xi.z);
    }

    BSDF_INLINE float eval(const float3 &h) const {
        return hvd_ggx_eval(m_inv_roughness, h.y, h.x, h.z);
    }

private:
    float2 m_inv_roughness;
};

BSDF_API void microfacet_ggx_vcavities_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_ggx_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_GGX_VC_MULTISCATTER);

    microfacet_sample(
        ph, data, state, inherited_normal, tangent_u, 
        math::saturate(tint), math::saturate(multiscatter_tint), mode, handle,
        MICROFACET_GGX_VCAVITIES_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_ggx_vcavities_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_ggx_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_GGX_VC_MULTISCATTER);

    const float2 contrib = microfacet_evaluate(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_GGX_VCAVITIES_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);

    add_elemental_bsdf_evaluate_contribution(
        data, handle,
        contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        contrib.x * math::saturate(tint) * inherited_weight);
}


BSDF_API void microfacet_ggx_vcavities_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_ggx_vcavities ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_GGX_VC_MULTISCATTER);
    
    microfacet_pdf(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_GGX_VCAVITIES_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_ggx_vcavities_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_beckmann_smith_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     scatter_mode    mode              = scatter_reflect,
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////


class Distribution_beckmann_smith {
public:
    BSDF_INLINE Distribution_beckmann_smith(const float roughness_u, const float roughness_v) {
        m_roughness = make_float2(clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    BSDF_INLINE float3 sample(const float4 &xi, const float3 &k) const {
        return hvd_beckmann_sample_vndf(k, m_roughness, make_float2(xi.x, xi.y));
    }

    BSDF_INLINE float shadow_mask(
        float &G1, float &G2,
        const float nh,
        const float3 &k1, const float k1h,
        const float3 &k2, const float k2h,
        const bool refraction) const {
        G1 = microfacet_mask_smith_beckmann(m_roughness.x, m_roughness.y, k1);
        G2 = microfacet_mask_smith_beckmann(m_roughness.x, m_roughness.y, k2);
        return G1 * G2;
    }


    BSDF_INLINE float eval(const float3 &h) const {
        return hvd_beckmann_eval(
            make_float2(1.0f / m_roughness.x, 1.0f / m_roughness.y), h.y, h.x, h.z);
    }

private:
    float2 m_roughness;
};

BSDF_API void microfacet_beckmann_smith_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_beckmann_smith ph(adapted_roughness.x, adapted_roughness.y);
    
    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BECKMANN_SMITH_MULTISCATTER);

    microfacet_sample(
        ph, data, state, inherited_normal, tangent_u, 
        math::saturate(tint), math::saturate(multiscatter_tint), mode, handle,
        MICROFACET_BECKMANN_SMITH_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_beckmann_smith_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_beckmann_smith ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BECKMANN_SMITH_MULTISCATTER);

    const float2 contrib = microfacet_evaluate(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_BECKMANN_SMITH_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);

    add_elemental_bsdf_evaluate_contribution(
        data, handle,
        contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        contrib.x * math::saturate(tint) * inherited_weight);
}


BSDF_API void microfacet_beckmann_smith_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_beckmann_smith ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BECKMANN_SMITH_MULTISCATTER);

    microfacet_pdf(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_BECKMANN_SMITH_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_beckmann_smith_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_ggx_smith_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     scatter_mode    mode              = scatter_reflect,
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////


class Distribution_ggx_smith {
public:
    BSDF_INLINE Distribution_ggx_smith(const float roughness_u, const float roughness_v) {
        m_roughness = make_float2(clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    BSDF_INLINE float3 sample(const float4 &xi, const float3 &k) const {
        return hvd_ggx_sample_vndf(k, m_roughness, make_float2(xi.x, xi.y));
    }

    BSDF_INLINE float shadow_mask(
        float &G1, float &G2,
        const float nh,
        const float3 &k1, const float k1h,
        const float3 &k2, const float k2h,
        const bool refraction) const {
        G1 = microfacet_mask_smith_ggx(m_roughness.x, m_roughness.y, k1);
        G2 = microfacet_mask_smith_ggx(m_roughness.x, m_roughness.y, k2);
        return G1 * G2;
    }

    BSDF_INLINE float eval(const float3 &h) const {
        return hvd_ggx_eval(make_float2(1.0f / m_roughness.x, 1.0f / m_roughness.y), h.y, h.x, h.z);
    }


private:
    float2 m_roughness;
};


BSDF_API void microfacet_ggx_smith_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_ggx_smith ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_GGX_SMITH_MULTISCATTER);
    
    microfacet_sample(
        ph, data, state, inherited_normal, tangent_u, 
        math::saturate(tint), math::saturate(multiscatter_tint), mode, handle,
        MICROFACET_GGX_SMITH_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_ggx_smith_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_ggx_smith ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_GGX_SMITH_MULTISCATTER);

    const float2 contrib = microfacet_evaluate(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_GGX_SMITH_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);

    add_elemental_bsdf_evaluate_contribution(
        data, handle,
        contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        contrib.x * math::saturate(tint) * inherited_weight);
}


BSDF_API void microfacet_ggx_smith_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    const Distribution_ggx_smith ph(adapted_roughness.x, adapted_roughness.y);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_GGX_SMITH_MULTISCATTER);

    microfacet_pdf(
        ph, data, state, inherited_normal, tangent_u, mode,
        MICROFACET_GGX_SMITH_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        multiscatter_texture_id);
}

BSDF_API void microfacet_ggx_smith_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const scatter_mode mode,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf ward_geisler_moroder_bsdf(
//     float           roughness_u,
//     float           roughness_v       = roughness_u,
//     color           tint              = color(1.0),
//     color           multiscatter_tint = color(0.0),
//     float3          tangent_u         = state->texture_tangent_u(0),
//     uniform string  handle            = ""
// )
/////////////////////////////////////////////////////////////////////

// "A New Ward BRDF Model with Bounded Albedo" by David Geisler-Moroder and Arne Duer

BSDF_INLINE void ward_geisler_moroder_sample(
    BSDF_sample_data *data,
    State *state,
    const Geometry &g,
    const float roughness_u,
    const float roughness_v,
    const int handle,
    float &nk1)
{
    const float2 inv_roughness = make_float2(
        1.0f / clamp_roughness(roughness_u), 1.0f / clamp_roughness(roughness_v));

    // importance sampling uses an (unpublished) trick somewhat similar to Heitz' technique for
    // v-cavities masking (it also considers both the sampled half vector and its flipped variant):
    //
    // - the straightforward way of sampling a half vector may yield weights w = brdf / pdf both
    //   greater than 1 and smaller than 1.
    // - if w < 1 we have "weight to spare" and just use the flipped half vector with probability
    //   q = 1 - w
    // - with that, if w would happen to be > 1 on the flipped side, we increase the pdf of sampling
    //   there, decreasing the weight
    // - in fact, using this trick all weights are guaranteed to be <= 1 for the
    //   Ward-Geisler-Moroder BRDF
    // - using the flipped half vector is possible if the half vector distribution is symmetric with
    //   respect to flipping along the normal (which is the case for the anisotropic Beckmann,
    //   Phong, and GGX distributions)

    // sample half vector
    float3 h0 = hvd_beckmann_sample(make_float2(data->xi.x, data->xi.y), inv_roughness);

    nk1 = math::abs(math::dot(data->k1, g.n.shading_normal));
    const float xk1 = math::dot(data->k1, g.x_axis);
    const float zk1 = math::dot(data->k1, g.z_axis);

    // compute weight for flipped and non-flipped variant
    const float a = nk1 * h0.y;
    const float b = xk1 * h0.x + zk1 * h0.z;
          float kh   = a + b;
    const float kh_f = a - b;
          float w   = math::max(2.0f - nk1 / (h0.y * kh  ), 0.0f);
    const float w_f = math::max(2.0f - nk1 / (h0.y * kh_f), 0.0f);

    // probabilities of not flipping each variant
    const float q   = math::min(w  , 1.0f);
    const float q_f = math::min(w_f, 1.0f);

    // actual probability of getting the used half vector both flipped and not flipped
    float prob_total;
    if (data->xi.z >= q) { // flip?
        h0.x = -h0.x;
        h0.z = -h0.z;
        kh = kh_f;
        w = w_f;

        prob_total = q_f + (1.0f - q);
    }
    else
        prob_total = q + (1.0f - q_f);


    const float3 h = g.x_axis * h0.x + g.n.shading_normal * h0.y + g.z_axis * h0.z;
    data->k2 = h * (kh + kh) - data->k1;

    const float nk2 = math::dot(data->k2, g.n.shading_normal);
    if (nk2 <= 0.0f || math::dot(data->k2, g.n.geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }

    // compute final weight and pdf
    data->bsdf_over_pdf = make<float3>(w / prob_total);
    data->pdf = hvd_beckmann_eval(inv_roughness, h0.y, h0.x, h0.z) * 0.25f / kh * prob_total;

    data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    data->handle = handle;
}

template <typename Data>
BSDF_INLINE float ward_geisler_moroder_shared_eval(
    Data *data,
    State *state,
    const Geometry &g,
    const float roughness_u,
    const float roughness_v,
    float &nk1,
    float &nk2)
{
    nk1 = math::dot(data->k1, g.n.shading_normal);
    nk2 = math::dot(data->k2, g.n.shading_normal);
    if (nk2 <= 0.0f) {
        absorb(data);
        return 0.0f;
    }

    const float2 inv_roughness = make_float2(
        1.0f / clamp_roughness(roughness_u), 1.0f / clamp_roughness(roughness_v));

    const float3 h = math::normalize(data->k1 + data->k2);
    const float3 h0 = make_float3(
        math::dot(g.x_axis, h),
        math::dot(g.n.shading_normal, h),
        math::dot(g.z_axis, h));


    // compute flipping probabilities for pdf
    const float xk1 = math::dot(g.x_axis, data->k1);
    const float zk1 = math::dot(g.z_axis, data->k1);
    const float kh   = h0.y * nk1 + h0.x * xk1 + h0.z * zk1;
    const float kh_f = h0.y * nk1 - h0.x * xk1 - h0.z * zk1;
    const float q   = math::saturate(2.0f - nk1 / (h0.y * kh));
    const float q_f = math::saturate(2.0f - nk1 / (h0.y * kh_f));


    const float ph = hvd_beckmann_eval(inv_roughness, h0.y, h0.x, h0.z) * 0.25f / kh;
    data->pdf = ph * (q + (1.0f - q_f));

    return ph * nk2 / (kh * h0.y);
}

BSDF_API void ward_geisler_moroder_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, tangent_u, data->k1, state))
    {
        absorb(data);
        return;
    }

    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    // sample the single scattering (glossy) bsdf
    float nk1, nk2;
    ward_geisler_moroder_sample(
        data, state, g, adapted_roughness.x, adapted_roughness.y, handle, nk1);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_WARD_GEISLER_MORODER_MULTISCATTER);
    
    if (multiscatter_texture_id == 0) {
        data->bsdf_over_pdf *= tint;
        return;
    }

    // sample, in case the multi-scattering part is sampled, k2 will change and rho1 will be > 0
    const float rho1 = multiscatter::sample(
        state, WARD_GEISLER_MORODER_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        nk1, -1.0f, multiscatter_texture_id, data, g, tint, multiscatter_tint);

    // recompute glossy pdf for new direction
    if (rho1 > 0.0f)
    {
        BSDF_pdf_data pdf_data = to_pdf_data(data);
        ward_geisler_moroder_shared_eval(
            &pdf_data, state, g, adapted_roughness.x, adapted_roughness.y, nk1, nk2);

        // incorporate multi-scatter part to pdf for the new k2
        multiscatter::sample_update_single_scatter_probability(data, pdf_data.pdf, rho1);
    }
}


BSDF_API void ward_geisler_moroder_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, tangent_u, data->k1, state)) {
        absorb(data);
        return;
    }

    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    float nk1, nk2;
    const float glossy_contrib = ward_geisler_moroder_shared_eval(
        data, state, g, adapted_roughness.x, adapted_roughness.y, nk1, nk2);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_WARD_GEISLER_MORODER_MULTISCATTER);

    float2 multiscatter_contrib;
    if (multiscatter_texture_id == 0)
    {
        multiscatter_contrib = make<float2>(0.0f);
    }
    else
    {
        multiscatter_contrib = multiscatter::evaluate(
            state, WARD_GEISLER_MORODER_BSDF,
            adapted_roughness.x, adapted_roughness.y,
            nk1, nk2, -1.0f, multiscatter_texture_id);
        data->pdf *= multiscatter_contrib.x; // * rho1
        if (math::dot(g.n.geometry_normal, data->k2) >= 0.0f)
        {
            data->pdf += (1.0f - multiscatter_contrib.x) * (float)(1.0 / M_PI);
        }
        else
            multiscatter_contrib.y = 0.0f; // backside eval
    }

    add_elemental_bsdf_evaluate_contribution(
        data, handle, 
        multiscatter_contrib.y * math::saturate(multiscatter_tint) * inherited_weight,
        glossy_contrib * math::saturate(tint) * inherited_weight);
}

BSDF_API void ward_geisler_moroder_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, tangent_u, data->k1, state)) {
        absorb(data);
        return;
    }

    const float2 adapted_roughness = state->adapt_microfacet_roughness(
        make_float2(roughness_u, roughness_v));

    float nk1, nk2;
    ward_geisler_moroder_shared_eval(
        data, state, g, adapted_roughness.x, adapted_roughness.y, nk1, nk2);

    const unsigned int multiscatter_texture_id = 
        (multiscatter_tint.x <= 0.0f && multiscatter_tint.y <= 0.0f && multiscatter_tint.z <= 0.0f) ? 0 :
        state->get_bsdf_data_texture_id(BDK_BACKSCATTERING_GLOSSY_MULTISCATTER);

    if (multiscatter_texture_id == 0)
        return;

    data->pdf = multiscatter::pdf(
        data->pdf,
        state, WARD_GEISLER_MORODER_BSDF,
        adapted_roughness.x, adapted_roughness.y,
        nk1, nk2, -1.0f, multiscatter_texture_id);
}

BSDF_API void ward_geisler_moroder_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &multiscatter_tint,
    const float3 &tangent_u,
    const int handle)
{
    elemental_bsdf_auxiliary(data, state, inherited_normal, inherited_weight, tint, handle);
}


/////////////////////////////////////////////////////////////////////
// bsdf measured_bsdf(
//     bsdf_measurement measurement,
//     float            multiplier = 1.0f,
//     scatter_mode     mode       = scatter_reflect,
//     uniform string   handle     = ""
// )
/////////////////////////////////////////////////////////////////////

BSDF_INLINE void measured_sample(
    BSDF_sample_data *data,
    State *state,
    const Geometry &g,
    const unsigned measurement_id,
    const float multiplier,
    const scatter_mode mode,
    const int handle)
{
    // world to local (assuming an orthonormal base)
    const float3 outgoing = math::normalize(make_float3(
        math::dot(data->k1, g.x_axis),
        math::dot(data->k1, g.n.shading_normal),
        math::dot(data->k1, g.z_axis)));

    // local Cartesian to polar
    const float2 outgoing_polar = make_float2(
        math::acos(outgoing.y),
        math::atan2(outgoing.z, outgoing.x));


    // x - albedo reflectance for outgoing_polar (maximum in case of color channels)
    // y - albedo reflectance for globally over all directions (maximum in case of color channels)
    // z - albedo transmittance for outgoing_polar (maximum in case of color channels)
    // w - albedo transmittance for globally over all directions (maximum in case of color channels)
    float4 max_albedos = state->bsdf_measurement_albedos(measurement_id, outgoing_polar);

    // disable the not selected parts
    if (mode == scatter_mode::scatter_reflect) max_albedos.z = 0.0f;
    if (mode == scatter_mode::scatter_transmit) max_albedos.x = 0.0f;

    // scale based on the global albedo
    float scale = math::max(0.0f, multiplier);
    if (mode == scatter_mode::scatter_reflect || mode == scatter_mode::scatter_reflect_transmit)
        scale = math::min(scale, 1.0f / max_albedos.y);
    if (mode == scatter_mode::scatter_transmit || mode == scatter_mode::scatter_reflect_transmit)
        scale = math::min(scale, 1.0f / max_albedos.w);

    const float sum = max_albedos.x + max_albedos.z;
    if (sum == 0.0f || scale == 0.0f) {
        absorb(data);
        return;
    }

    Mbsdf_part selected_part;
    float prob = max_albedos.x / sum;
    if (data->xi.z < prob) {
        data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
        selected_part = Mbsdf_part::mbsdf_data_reflection;
        data->xi.z /= prob;
    } else {
        data->event_type = BSDF_EVENT_GLOSSY_TRANSMISSION;
        selected_part = Mbsdf_part::mbsdf_data_transmission;
        data->xi.z = (data->xi.z - prob) / (1.0f - prob);
        prob = (1.0f - prob);
    }

    // sample incoming direction
    const float3 incoming_polar_pdf = state->bsdf_measurement_sample(
        measurement_id,
        outgoing_polar,
        xyz(data->xi),
        selected_part);

    const float sign = ((data->event_type == BSDF_EVENT_GLOSSY_TRANSMISSION) ? -1.0f : 1.0f);

    // transform to world
    float2 incoming_polar_sin, incoming_polar_cos;
    math::sincos(make<float2>(incoming_polar_pdf.x, incoming_polar_pdf.y),
                 &incoming_polar_sin,
                 &incoming_polar_cos);
    data->k2 = math::normalize(
        g.x_axis            * incoming_polar_sin.x * incoming_polar_cos.y   +
        sign * g.n.shading_normal  * incoming_polar_cos.x                   +
        g.z_axis            * incoming_polar_sin.x * incoming_polar_sin.y);

    // check for valid a result and sampling beneath the surface
    if (incoming_polar_pdf.x < 0.0f ||
        incoming_polar_cos.x <= 0.0f || math::dot(data->k2, g.n.geometry_normal) * sign <= 0.0f)
    {
        absorb(data);
        return;
    }

    data->pdf = incoming_polar_pdf.z * prob;
    data->bsdf_over_pdf = state->bsdf_measurement_evaluate(
        measurement_id,
        make<float2>(incoming_polar_pdf.x, incoming_polar_pdf.y),
        outgoing_polar,
        selected_part) * (scale * incoming_polar_cos.x / data->pdf);
    data->handle = handle;
}

BSDF_API void measured_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const unsigned measurement_id,
    const float multiplier,
    const scatter_mode mode,
    const int handle)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        absorb(data);
        return;
    }
    measured_sample(data, state, g, measurement_id, multiplier, mode, handle);
}


template<typename BSDF_x_data>
BSDF_INLINE bool measured_compute_polar(
    BSDF_x_data *data,
    State *state,
    const Normals &n,
    const scatter_mode mode,
    float3& out_incoming,
    float2& out_incoming_polar,
    float2& out_outgoing_polar,
    bool& out_backside_eval)
{
    // local (tangent) space
    float3 x_axis, z_axis;
    float3 y_axis = n.shading_normal;

    // BTDF or BRDF eval?
    out_backside_eval = math::dot(data->k2, n.geometry_normal) < 0.0f;

    if (!get_bumped_basis(x_axis, z_axis, state->texture_tangent_u(0), y_axis)) {
        absorb(data);
        return false;
    }

    // nothing to evaluate for given directions?
    if ((out_backside_eval && (mode == scatter_reflect)) ||
        (!out_backside_eval && (mode == scatter_transmit)))
    {
        absorb(data);
        return false;
    }

    // world to local (assuming an orthonormal base)
    out_incoming = math::normalize(
        make_float3(
        math::dot(data->k2, x_axis),
        math::abs(math::dot(data->k2, y_axis)),
        math::dot(data->k2, z_axis)));

    const float3 outgoing = math::normalize(
        make_float3(
        math::dot(data->k1, x_axis),
        math::dot(data->k1, y_axis),
        math::dot(data->k1, z_axis)));

    // filter rays below the surface
    if (out_incoming.y < 0.0f || outgoing.y < 0.0f)
    {
        absorb(data);
        return false;
    }

    // local Cartesian to polar
    out_incoming_polar = make_float2(
        math::acos(out_incoming.y),
        math::atan2(out_incoming.z, out_incoming.x));

    out_outgoing_polar = make_float2(
        math::acos(outgoing.y),
        math::atan2(outgoing.z, outgoing.x));

    return true;
}


BSDF_INLINE void measured_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const Normals &n,
    const float3 &inherited_weight,
    const unsigned measurement_id,
    const float multiplier,
    const scatter_mode mode,
    const int handle)
{
    float3 incoming;
    float2 incoming_polar;
    float2 outgoing_polar;
    bool backside_eval;
    if(!measured_compute_polar(
        data, state, n, mode, incoming, incoming_polar, outgoing_polar, backside_eval))
            return;

    // x - albedo reflectance for outgoing_polar (maximum in case of color channels)
    // y - albedo reflectance for globally over all directions (maximum in case of color channels)
    // z - albedo transmittance for outgoing_polar (maximum in case of color channels)
    // w - albedo transmittance for globally over all directions (maximum in case of color channels)
    const float4 max_albedos = state->bsdf_measurement_albedos(measurement_id, outgoing_polar);
    const float sum = max_albedos.x + max_albedos.z;
    Mbsdf_part selected_part;
    float prob = max_albedos.x / sum;
    if (backside_eval) {
        selected_part = Mbsdf_part::mbsdf_data_transmission;
        prob = (1.0f - prob);
    } else {
        selected_part = Mbsdf_part::mbsdf_data_reflection;
    }

    if (prob == 0.0f) {
        absorb(data);
        return;
    }

    // scale based on the global albedo
    float scale = math::max(0.0f, multiplier);
    if (mode == scatter_mode::scatter_reflect || mode == scatter_mode::scatter_reflect_transmit)
        scale = math::min(scale, 1.0f / max_albedos.y);
    if (mode == scatter_mode::scatter_transmit || mode == scatter_mode::scatter_reflect_transmit)
        scale = math::min(scale, 1.0f / max_albedos.w);

    data->pdf = prob * state->bsdf_measurement_pdf(
        measurement_id,
        incoming_polar,
        outgoing_polar,
        selected_part);

    // assuming measured material is glossy
    const float3 bsdf_glossy = (scale * incoming.y) * state->bsdf_measurement_evaluate(
        measurement_id,
        incoming_polar,
        outgoing_polar,
        selected_part);

    add_elemental_bsdf_evaluate_contribution(
        data, handle, make<float3>(0.0f), bsdf_glossy * inherited_weight);
}

BSDF_API void measured_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const unsigned measurement_id,
    const float multiplier,
    const scatter_mode mode,
    const int handle)
{
    Normals n;
    get_oriented_normals(
        n.shading_normal, n.geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    measured_evaluate(data, state, n, inherited_weight, measurement_id, multiplier, mode, handle);
}

template <typename BSDF_x_data>
BSDF_INLINE void measured_pdf(
    BSDF_x_data *data,
    State *state,
    const Normals &n,
    const unsigned measurement_id,
    const scatter_mode mode)
{
    float3 incoming;
    float2 incoming_polar;
    float2 outgoing_polar;
    bool backside_eval;
    if(!measured_compute_polar(
        data, state, n, mode, incoming, incoming_polar, outgoing_polar, backside_eval))
            return;

    // x - albedo reflectance for outgoing_polar (maximum in case of color channels)
    // y - albedo reflectance for globally over all directions (maximum in case of color channels)
    // z - albedo transmittance for outgoing_polar (maximum in case of color channels)
    // w - albedo transmittance for globally over all directions (maximum in case of color channels)
    const float4 max_albedos = state->bsdf_measurement_albedos(measurement_id, outgoing_polar);
    const float sum = max_albedos.x + max_albedos.z;
    Mbsdf_part selected_part;
    float prob = max_albedos.x / sum;
    if (backside_eval) {
        selected_part = Mbsdf_part::mbsdf_data_transmission;
        prob = (1.0f - prob);
    } else {
        selected_part = Mbsdf_part::mbsdf_data_reflection;
    }

    if (prob == 0.0f) {
        absorb(data);
        return;
    }

    data->pdf = prob * state->bsdf_measurement_pdf(
        measurement_id,
        incoming_polar,
        outgoing_polar,
        selected_part);
}


BSDF_API void measured_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const unsigned measurement_id,
    const float multiplier,
    const scatter_mode mode,
    const int handle)
{
    Normals n;
    get_oriented_normals(
        n.shading_normal, n.geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    measured_pdf(data, state, n, measurement_id, mode);
}


BSDF_API void measured_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const unsigned measurement_id,
    const float multiplier,
    const scatter_mode mode,
    const int handle)
{
    Normals n;
    get_oriented_normals(
        n.shading_normal, n.geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    // local (tangent) space
    float3 x_axis, z_axis;
    float3 y_axis = n.shading_normal;
    if (!get_bumped_basis(x_axis, z_axis, state->texture_tangent_u(0), y_axis)) {
        absorb(data);
        return;
    }

    // world to local (assuming an orthonormal base)
    const float3 outgoing = math::normalize(
        make_float3(
        math::dot(data->k1, x_axis),
        math::dot(data->k1, y_axis),
        math::dot(data->k1, z_axis)));

    // local Cartesian to polar
    const float2 outgoing_polar = make_float2(
        math::acos(outgoing.y),
        math::atan2(outgoing.z, outgoing.x));

    // filter rays below the surface
    if (outgoing.y < 0.0f)
    {
        absorb(data);
        return;
    }

    // x - albedo reflectance for outgoing_polar (maximum in case of color channels)
    // y - albedo reflectance for globally over all directions (maximum in case of color channels)
    // z - albedo transmittance for outgoing_polar (maximum in case of color channels)
    // w - albedo transmittance for globally over all directions (maximum in case of color channels)
    const float4 max_albedos = state->bsdf_measurement_albedos(measurement_id, outgoing_polar);
    const float sum = max_albedos.x + max_albedos.z;

    float w_reflection = max_albedos.x / sum;
    float w_transmission = 1.0f - w_reflection;

    // scale based on the global albedo
    float scale = math::max(0.0f, multiplier);
    w_reflection *= math::min(scale, 1.0f / max_albedos.y);
    w_transmission *= math::min(scale, 1.0f / max_albedos.w);

    // TODO evaluate to color RGB albedo
    add_elemental_bsdf_auxiliary_contribution(
        data,
        handle,
        inherited_weight * (w_reflection * max_albedos.x + w_transmission * max_albedos.w),
        math::average(inherited_weight) * n.shading_normal);
}


/////////////////////////////////////////////////////////////////////
// bsdf tint(
//     color  tint,
//     bsdf   base
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void tint_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    data->bsdf_over_pdf *= math::saturate(tint);
}

BSDF_API void tint_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const BSDF &base)
{
    const float3 factor = math::saturate(tint);
    base.evaluate(data, state, inherited_normal, factor * inherited_weight);
}

BSDF_API void tint_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void tint_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const BSDF &base)
{
    const float3 factor = math::saturate(tint);
    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf tint(
//     color  reflection_tint,
//     color  transmission_tint,
//     bsdf   base
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void tint_rt_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &reflection_tint,
    const float3 &transmission_tint,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    const float3 tint = (data->event_type & BSDF_EVENT_TRANSMISSION) == 0 
        ? reflection_tint 
        : transmission_tint;

    data->bsdf_over_pdf *= math::saturate(tint);
}

BSDF_API void tint_rt_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &reflection_tint,
    const float3 &transmission_tint,
    const BSDF &base)
{
    // get a shading normal on the side of k1
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk2 = math::dot(data->k2, shading_normal);

    const float3& tint = (nk2 < 0.0f) ? transmission_tint : reflection_tint;
    base.evaluate(data, state, inherited_normal, math::saturate(tint) * inherited_weight);
}

BSDF_API void tint_rt_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &reflection_tint,
    const float3 &transmission_tint,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void tint_rt_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &reflection_tint,
    const float3 &transmission_tint,
    const BSDF &base)
{
    const float3 factor = math::saturate((reflection_tint + reflection_tint) * 0.5f);
    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// edf tint(
//     color  tint,
//     edf   base
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void tint_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const EDF &base)
{
    base.sample(data, state, inherited_normal);
    data->edf_over_pdf *= math::saturate(tint);
}

BSDF_API void tint_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const EDF &base)
{
    const float3 factor = math::saturate(tint);
    base.evaluate(data, state, inherited_normal, factor * inherited_weight);
}

BSDF_API void tint_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const EDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void tint_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &tint,
    const EDF &base)
{
    const float3 factor = math::saturate(tint);
    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf thin_film(
//     float  thickness,
//     color  ior,
//     bsdf   base
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void thin_film_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data, state);
    const float coating_ior = math::luminance(ior); //!!TODO: no color support here

    data->bsdf_over_pdf *= thin_film_factor(
        coating_ior, thickness, mat_ior, data->k1, data->k2, shading_normal,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0, get_material_thin_walled(state));
}

BSDF_API void thin_film_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data, state);
    const float coating_ior = math::luminance(ior); //!!TODO: no color support here

    const float3 factor = thin_film_factor(
        coating_ior, thickness, mat_ior, data->k1, data->k2, shading_normal,
        (math::dot(data->k2, geometry_normal) < 0.0f), get_material_thin_walled(state));

    base.evaluate(data, state, inherited_normal, factor * inherited_weight); 
}

BSDF_API void thin_film_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void thin_film_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data, state);
    const float coating_ior = math::luminance(ior); //!!TODO: no color support here

    //!! clamped to avoid total internal reflection
    const float eta = math::min(mat_ior.x, coating_ior) / coating_ior;

    // assuming perfect reflection of k1, so the half-vector equals the normal
    const float kh = math::abs(math::dot(data->k1, shading_normal));
    const float costheta_refr = refraction_cosine(kh, eta);
    const float phase_shift_dist = 2.0f * coating_ior * costheta_refr * thickness;
    const float coating_fresnel = ior_fresnel(1.0f / eta, kh);

    const float3 factor = compute_interference_factor(
        coating_fresnel, 1.0f - coating_fresnel, phase_shift_dist);

    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf directional_factor(
//     color  normal_tint  = color(1.0),
//     color  grazing_tint = color(1.0),
//     float  exponent     = 5.0,
//     bsdf   base         = bsdf()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void directional_factor_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0);
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    data->bsdf_over_pdf *= custom_curve_factor(
        kh, exponent, math::saturate(normal_tint), math::saturate(grazing_tint));
}

BSDF_API void directional_factor_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f);
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    const float3 factor = custom_curve_factor(
        kh, exponent, math::saturate(normal_tint), math::saturate(grazing_tint));

    base.evaluate(data, state, inherited_normal, factor * inherited_weight);
}

BSDF_API void directional_factor_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void directional_factor_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    if (nk1 < 0.0f) {
        absorb(data);
        return;
    }

    const float3 factor = custom_curve_factor(
        nk1, exponent, math::saturate(normal_tint), math::saturate(grazing_tint));

    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf fresnel_factor(
//     color  ior,
//     color  extinction_coefficent,
//     bsdf   base = bsdf()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void fresnel_factor_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0);
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    const float2 material_ior = process_ior(data, state);
    const float inv_eta_i = 1.0f / material_ior.x;
    const float3 eta = ior * inv_eta_i;
    const float3 eta_k = extinction_coefficient * inv_eta_i;
    data->bsdf_over_pdf *= complex_ior_fresnel(eta, eta_k, kh);
}

BSDF_API void fresnel_factor_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f);
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    const float2 material_ior = process_ior(data, state);
    const float inv_eta_i = 1.0f / material_ior.x;
    const float3 eta = ior * inv_eta_i;
    const float3 eta_k = extinction_coefficient * inv_eta_i;

    const float3 factor = complex_ior_fresnel(eta, eta_k, kh);
    base.evaluate(data, state, inherited_normal, factor * inherited_weight);
}

BSDF_API void fresnel_factor_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void fresnel_factor_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    if (nk1 < 0.0f) {
        absorb(data);
        return;
    }

    const float2 material_ior = process_ior(data, state);
    const float inv_eta_i = 1.0f / material_ior.x;
    const float3 eta = ior * inv_eta_i;
    const float3 eta_k = extinction_coefficient * inv_eta_i;

    const float3 factor = complex_ior_fresnel(eta, eta_k, nk1);
    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf measured_curve_factor(
//     color[<N>] curve_values,
//     bsdf   base = bsdf()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void measured_curve_factor_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0);
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    data->bsdf_over_pdf *=
        math::saturate(measured_curve_factor(kh, curve_values, num_curve_values));
}

BSDF_API void measured_curve_factor_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f);
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    const float3 factor = math::saturate(measured_curve_factor(kh, curve_values, num_curve_values));
    base.evaluate(data, state, inherited_normal, factor * inherited_weight);
}

BSDF_API void measured_curve_factor_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void measured_curve_factor_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    if (nk1 < 0.0f) {
        absorb(data);
        return;
    }

    const float3 factor = math::saturate(measured_curve_factor(nk1, curve_values, num_curve_values));
    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf measured_factor(
//     texture_2d values,
//     bsdf       base = bsdf()
// )
/////////////////////////////////////////////////////////////////////

BSDF_INLINE float3 measured_factor(
    State *state,
    const float3 &shading_normal,
    const float3 &k2,
    const float3 &h,
    const unsigned value_texture_index)
{
    const float alpha = math::abs(math::dot(k2, h));
    const float beta = math::abs(math::dot(shading_normal, h));
    const float2 coord = make<float2>(
        math::acos(alpha) * (float)(2.0 / M_PI),
        math::acos(beta) * (float)(2.0 / M_PI));
    const float2 clamp = make<float2>(0.0f, 1.0f);

    const float3 f = state->tex_lookup_float3_2d(value_texture_index, coord, 0, 0, clamp, clamp);
    return math::saturate(f);
}


BSDF_API void measured_factor_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const unsigned value_texture_index,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0);

    float3 factor = measured_factor(state, shading_normal, data->k2, h, value_texture_index);
    data->bsdf_over_pdf *= factor;
}

BSDF_API void measured_factor_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const unsigned value_texture_index,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f);

    float3 factor = measured_factor(state, shading_normal, data->k2, h, value_texture_index);
    base.evaluate(data, state, inherited_normal, factor * inherited_weight);
}


BSDF_API void measured_factor_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const unsigned value_texture_index,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void measured_factor_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const unsigned value_texture_index,
    const BSDF &base)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state->geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    if (nk1 < 0.0f)
    {
        absorb(data);
        return;
    }

    // assuming nk1 == nk2, -> h = n
    float3 factor = measured_factor(
        state, shading_normal, data->k1, shading_normal, value_texture_index);
    base.auxiliary(data, state, inherited_normal, factor * inherited_weight);
}

/////////////////////////////////////////////////////////////////////
// bsdf weighted_layer(
//     float   weight,
//     bsdf    layer,
//     bsdf    base   = bsdf(),
//     float3  normal = state->normal()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void weighted_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);

    if (data->xi.z < weight) {
        data->xi.z /= weight;

        layer.sample(data, state, normal);
        if (data->event_type == BSDF_EVENT_ABSORB)
            return;

        if (weight < 1.0f) {
            BSDF_pdf_data pdf_data = to_pdf_data(data);
            base.pdf(&pdf_data, state, inherited_normal);
            data->pdf = pdf_data.pdf * (1.0f - weight) + data->pdf * weight;
        }
    } else {
        data->xi.z = (data->xi.z - weight) / (1.0f - weight);

        base.sample(data, state, inherited_normal);
        if (data->event_type == BSDF_EVENT_ABSORB)
            return;

        if (weight > 0.0f) {
            BSDF_pdf_data pdf_data = to_pdf_data(data);
            layer.pdf(&pdf_data, state, normal);
            data->pdf = pdf_data.pdf * weight + data->pdf * (1.0f - weight);
        }
    }
}

BSDF_API void weighted_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);

    float pdf = 0.0f;
    if (weight > 0.0f) {
        layer.evaluate(data, state, normal, weight * inherited_weight); 
        pdf = weight * data->pdf;
    }
    if (weight < 1.0f) {
        const float inv_weight = 1.0f - weight;
        base.evaluate(data, state, inherited_normal, inv_weight * inherited_weight);
        pdf += inv_weight * data->pdf;
    }
    data->pdf = pdf;
}

BSDF_API void weighted_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);

    float pdf = 0.0f;
    if (weight > 0.0f) {
        layer.pdf(data, state, normal);
        pdf += weight * data->pdf;
    }
    if (weight < 1.0f) {
        base.pdf(data, state, inherited_normal);
        pdf += (1.0f - weight) * data->pdf;
    }
    data->pdf = pdf;
}

BSDF_API void weighted_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    layer.auxiliary(data, state, normal, weight * inherited_weight);
    base.auxiliary(data, state, normal, (1.0f - weight) * inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf color_weighted_layer(
//     color   weight,
//     bsdf    layer,
//     bsdf    base   = bsdf(),
//     float3  normal = state->normal()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void color_weighted_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    float3 weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    const float p = math::average(weight);

    if (data->xi.z < p) {
        const float p_inv = 1.0f / p;
        data->xi.z *= p_inv;

        layer.sample(data, state, normal);
        if (data->event_type == BSDF_EVENT_ABSORB)
            return;

        data->bsdf_over_pdf *= weight * p_inv;

        if (p < 1.0f) {
            BSDF_pdf_data pdf_data = to_pdf_data(data);
            base.pdf(&pdf_data, state, inherited_normal);
            data->pdf = pdf_data.pdf * (1.0f - p) + data->pdf * p;
        }
    } else {
        const float p_inv = 1.0f / (1.0f - p);
        data->xi.z = (data->xi.z - p) * p_inv;

        base.sample(data, state, inherited_normal);
        if (data->event_type == BSDF_EVENT_ABSORB)
            return;

        data->bsdf_over_pdf *= (make_float3(1.0f, 1.0f, 1.0f) - weight) * p_inv;

        if (p > 0.0f) {
            BSDF_pdf_data pdf_data = to_pdf_data(data);
            layer.pdf(&pdf_data, state, normal);
            data->pdf = pdf_data.pdf * p + data->pdf * (1.0f - p);
        }
    }
}

BSDF_API void color_weighted_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    float3 weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    const float p = math::average(weight);

    float pdf = 0.0f;
    if (p > 0.0f) {
        layer.evaluate(data, state, normal, weight * inherited_weight); 
        pdf = p * data->pdf;
    }
    if (p < 1.0f) {
        base.evaluate(data, state, inherited_normal, (1.0f - weight) * inherited_weight);
        pdf += (1.0f - p) * data->pdf;
    }
    data->pdf  = pdf;
}

BSDF_API void color_weighted_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    const float p = math::average(math::saturate(weight));

    float pdf = 0.0f;
    if (p > 0.0f) {
        layer.pdf(data, state, normal);
        pdf = p * data->pdf;
    }
    if (p < 1.0f) {
        base.pdf(data, state, inherited_normal);
        pdf += (1.0f - p) * data->pdf;
    }
    data->pdf = pdf;
}

BSDF_API void color_weighted_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    float3 weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);

    layer.auxiliary(data, state, normal, weight * inherited_weight);
    base.auxiliary(data, state, inherited_normal, (1.0f - weight) * inherited_weight);
}


//
// Common helper code for directionally dependent layering where the weight of the upper layer is
// determined by a function of the angle between direction and half vector f(dot(k, h)).
//
// - For importance sampling we estimate the upper layer weight by f(dot(k1, normal))
//   (motivated by the assumption, that the upper layer is typically rather specular).
// - For the weight of the lower layer we use 1 - max(f(dot(k1, normal)), f(dot(k2, normal))).
//   This heuristic is motivated by the assumption of a rather specular upper layer, symmetry
//   constraints, and energy conservation.
//

template <typename Curve_eval>
BSDF_INLINE void curve_layer_sample(
    const Curve_eval &c,
    BSDF_sample_data *data,
    State *state,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal)
{
    weight = math::saturate(weight);

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float estimated_curve_factor = c.estimate(nk1);

    const bool no_base = base.is_black();

    const float prob_layer = no_base ? 1.0f : estimated_curve_factor * weight;
    const bool sample_layer = no_base || (data->xi.z < prob_layer);
    if (sample_layer) {
        data->xi.z /= prob_layer;
        layer.sample(data, state, layer_normal);
    } else {
        data->xi.z = (1.0f - data->xi.z) / (1.0f - prob_layer);
        base.sample(data, state, base_normal);
    }

    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    const bool transmission = (data->event_type & BSDF_EVENT_TRANSMISSION) != 0;
    const float2 ior = c.ior();
    const bool no_refraction = (ior.x < 0.0f) || !transmission || get_material_thin_walled(state);

    const float nk2 = math::abs(math::dot(data->k2, layer_normal));
    const float3 h =
        compute_half_vector(
            data->k1, data->k2, layer_normal, ior, nk2,
            transmission, no_refraction);

    const float kh = math::abs(math::dot(data->k1, h));
    BSDF_pdf_data pdf_data = to_pdf_data(data);
    if (sample_layer) {
        const float3 curve_factor = c.eval(kh);
        data->bsdf_over_pdf *= curve_factor * weight / prob_layer;

        base.pdf(&pdf_data, state, base_normal);
        data->pdf = pdf_data.pdf * (1.0f - prob_layer) + data->pdf * prob_layer;
    } else {
        const float nk2_refl = no_refraction ? nk2 :
            2.0f * kh * math::dot(layer_normal, h) - nk1;
        const float3 w_base =
            make_float3(1.0f, 1.0f, 1.0f) - weight * math::max(c.eval(nk1), c.eval(nk2_refl));
        data->bsdf_over_pdf *= w_base / (1.0f - prob_layer);

        layer.pdf(&pdf_data, state, layer_normal);
        data->pdf = pdf_data.pdf * prob_layer + data->pdf * (1.0f - prob_layer);
    }
}

template <typename Curve_eval>
BSDF_INLINE void curve_layer_evaluate(
    const Curve_eval &c,
    BSDF_evaluate_data *data,
    State *state,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal,
    const float3 &geometry_normal,
    const float3 &inherited_weight)
{
    weight = math::saturate(weight);

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float nk2 = math::abs(math::dot(data->k2, layer_normal));

    const bool transmission = math::dot(data->k2, geometry_normal) < 0.0f;
    const float2 ior = c.ior();
    const bool no_refraction = (ior.x < 0.0f) || !transmission || get_material_thin_walled(state);

    const float3 h =
        compute_half_vector(
            data->k1, data->k2, layer_normal, ior, nk2,
            transmission, no_refraction);

    const float kh = math::abs(math::dot(data->k1, h));
    const float3 curve_factor = c.eval(kh);
    const float3 cf1 = c.eval(nk1);
    
    const float nk2_refl = no_refraction ? nk2 : 2.0f * kh * math::dot(layer_normal, h) - nk1;
    const float3 cf2 = c.eval(nk2);

    layer.evaluate(data, state, layer_normal, weight * curve_factor * inherited_weight);
    if (base.is_black())
        return;

    const float prob_layer = weight * c.estimate(nk1);
    const float pdf_layer = data->pdf * prob_layer;

    base.evaluate(
        data, state, base_normal, (1.0f - weight * math::max(cf1, cf2)) * inherited_weight);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}

template <typename Curve_eval>
BSDF_INLINE void curve_layer_auxiliary(
    const Curve_eval &c,
    BSDF_auxiliary_data *data,
    State *state,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal,
    const float3 &geometry_normal,
    const float3 &inherited_weight)
{
    weight = math::saturate(weight);

    // assuming perfect reflection
    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float3 curve_factor = weight * c.eval(nk1);

    layer.auxiliary(data, state, layer_normal, curve_factor * inherited_weight);
    if (base.is_black())
        return;

    base.auxiliary(data, state, base_normal, (1.0f - curve_factor) * inherited_weight);
}

template <typename Curve_eval>
BSDF_INLINE void curve_layer_pdf(
    const Curve_eval &c,
    BSDF_pdf_data *data,
    State *state,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal,
    const float3 &geometry_normal)
{
    weight = math::saturate(weight);

    layer.pdf(data, state, layer_normal);
    if (base.is_black())
        return;

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float prob_layer = weight * c.estimate(nk1);
    const float pdf_layer = data->pdf * prob_layer;

    base.pdf(data, state, base_normal);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}


/////////////////////////////////////////////////////////////////////
// bsdf fresnel_layer(
//     float   ior,
//     float   weight = 1.0,
//     bsdf    layer  = bsdf(),
//     bsdf    base   = bsdf(),
//     float3  normal = state->normal()
// )
/////////////////////////////////////////////////////////////////////

class Fresnel_curve_eval {
public:
    Fresnel_curve_eval(const float2 &ior) :
        m_eta(ior.y / ior.x), m_ior(ior) {
    }

    float estimate(const float cosine) const {
        return ior_fresnel(m_eta, cosine);
    }

    float3 eval(const float cosine) const {
        const float f = ior_fresnel(m_eta, cosine);
        return make_float3(f, f, f);
    }

    float2 ior() const {
        return m_ior;
    }
private:
    float m_eta;
    float2 m_ior;
};

BSDF_API void fresnel_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior_fresnel_layer(data, state, ior);
    const Fresnel_curve_eval c(mat_ior);
    curve_layer_sample(
        c, data, state, weight, layer, base, shading_normal, inherited_normal);
}

BSDF_API void fresnel_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior_fresnel_layer(data, state, ior);
    const Fresnel_curve_eval c(mat_ior);
    curve_layer_evaluate(
        c, data, state, weight, layer, base,
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

BSDF_API void fresnel_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior_fresnel_layer(data, state, ior);
    const Fresnel_curve_eval c(mat_ior);
    curve_layer_pdf(
        c, data, state, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void fresnel_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float2 mat_ior = process_ior_fresnel_layer(data, state, ior);
    const Fresnel_curve_eval c(mat_ior);
    curve_layer_auxiliary(
        c, data, state, weight, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf color_fresnel_layer(
//     color   ior,
//     color   weight = 1.0,
//     bsdf    layer  = bsdf(),
//     bsdf    base   = bsdf(),
//     float3  normal = state->normal()
// )
/////////////////////////////////////////////////////////////////////

class Color_fresnel_curve_eval {
public:
    Color_fresnel_curve_eval(const float3 &eta, const float3 &weight, const float2 &ior) :
        m_eta(eta), m_weight(math::saturate(weight)), m_ior(ior) {
    }

    float estimate(const float cosine) const {
        return math::luminance(eval(cosine));
    }

    float3 eval(const float cosine) const {
        return m_weight * make_float3(
            ior_fresnel(m_eta.x, cosine),
            ior_fresnel(m_eta.y, cosine),
            ior_fresnel(m_eta.z, cosine));
    }
    float2 ior() const {
        return m_ior;
    }

private:
    float3 m_eta;
    float3 m_weight;
    float2 m_ior;
};

BSDF_API void color_fresnel_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_fresnel_ior mat_ior = process_ior_color_fresnel_layer(data, state, ior);
    const Color_fresnel_curve_eval c(mat_ior.eta, weight, mat_ior.ior);
    curve_layer_sample(
        c, data, state, 1.0f, layer, base, shading_normal, inherited_normal);
}

BSDF_API void color_fresnel_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_fresnel_ior mat_ior = process_ior_color_fresnel_layer(data, state, ior);
    const Color_fresnel_curve_eval c(mat_ior.eta, weight, mat_ior.ior);
    curve_layer_evaluate(
        c, data, state, 1.0f, layer, base,
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

BSDF_API void color_fresnel_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_fresnel_ior mat_ior = process_ior_color_fresnel_layer(data, state, ior);
    const Color_fresnel_curve_eval c(mat_ior.eta, weight, mat_ior.ior);
    curve_layer_pdf(
        c, data, state, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void color_fresnel_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_fresnel_ior mat_ior = process_ior_color_fresnel_layer(data, state, ior);
    const Color_fresnel_curve_eval c(mat_ior.eta, weight, mat_ior.ior);
    curve_layer_auxiliary(
        c, data, state, 1.0f, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf custom_curve_layer(
//     float   normal_reflectivity,
//     float   grazing_reflectivity = 1.0,
//     float   exponent             = 5.0,
//     float   weight               = 1.0,
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state->normal()
// )
/////////////////////////////////////////////////////////////////////

class Custom_curve_eval {
public:
    Custom_curve_eval(
        const float r0, const float r90, const float exponent) :
        m_r0(math::saturate(r0)), m_r90(math::saturate(r90)), m_exponent(exponent) {
    }

    float estimate(const float cosine) const {
        return custom_curve_factor(cosine, m_exponent, m_r0, m_r90);
    }

    float3 eval(const float cosine) const {
        const float f = custom_curve_factor(cosine, m_exponent, m_r0, m_r90);
        return make_float3(f, f, f);
    }

    float2 ior() const {
        return make<float2>(-1.0f);
    }

private:
    float m_r0, m_r90, m_exponent;
};

BSDF_API void custom_curve_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float normal_reflectivity,
    const float grazing_reflectivity,
    const float exponent,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_sample(c, data, state, weight, layer, base, shading_normal, inherited_normal);
}

BSDF_API void custom_curve_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float normal_reflectivity,
    const float grazing_reflectivity,
    const float exponent,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_evaluate(
        c, data, state, weight, layer, base,
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

BSDF_API void custom_curve_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float normal_reflectivity,
    const float grazing_reflectivity,
    const float exponent,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_pdf(
        c, data, state, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}


BSDF_API void custom_curve_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float normal_reflectivity,
    const float grazing_reflectivity,
    const float exponent,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_auxiliary(
        c, data, state, weight, layer, base,
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf color_custom_curve_layer(
//     color   normal_reflectivity,
//     color   grazing_reflectivity = color(1.0),
//     float   exponent             = 5.0,
//     color   weight               = color(1.0),
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state->normal()
// )
/////////////////////////////////////////////////////////////////////

class Color_custom_curve_eval {
public:
    Color_custom_curve_eval(
        const float3 &r0, const float3 &r90, const float3 &weight, const float exponent) :
        m_r0(math::saturate(r0)),
        m_r90(math::saturate(r90)),
        m_weight(math::saturate(weight)),
        m_exponent(exponent) {
    }

    float estimate(const float cosine) const {
        return math::luminance(eval(cosine));
    }

    float3 eval(const float cosine) const {
        return m_weight * custom_curve_factor(cosine, m_exponent, m_r0, m_r90);
    }

    float2 ior() const {
        return make<float2>(-1.0f);
    }
private:
    float3 m_r0, m_r90, m_weight;
    float m_exponent;
};

BSDF_API void color_custom_curve_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &normal_reflectivity,
    const float3 &grazing_reflectivity,
    const float exponent,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_sample(c, data, state, 1.0f, layer, base, shading_normal, inherited_normal);
}

BSDF_API void color_custom_curve_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &normal_reflectivity,
    const float3 &grazing_reflectivity,
    const float exponent,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_evaluate(
        c, data, state, 1.0f, layer, base,
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

BSDF_API void color_custom_curve_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &normal_reflectivity,
    const float3 &grazing_reflectivity,
    const float exponent,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_pdf(
        c, data, state, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void color_custom_curve_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 &normal_reflectivity,
    const float3 &grazing_reflectivity,
    const float exponent,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_auxiliary(
        c, data, state, 1.0f, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}


/////////////////////////////////////////////////////////////////////
// bsdf measured_curve_layer(
//     color[<N>] curve_values,
//     float   weight               = 1.0,
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state->normal()
// )
/////////////////////////////////////////////////////////////////////

// The HLSL backend does not support storing pointers, so we cannot use the
// templated curve_layer functions, but need to duplicate their code.
#if 0

class Measured_curve_eval {
public:
    Measured_curve_eval(
        const float3 *const values, const unsigned int num_values) :
        m_values(values), m_num_values(num_values) {
    }

    float estimate(const float cosine) const {
        return math::luminance(eval(cosine));
    }

    float3 eval(const float cosine) const {
        if (m_num_values == 0)
            return make_float3(0.0f, 0.0f, 0.0f);
        else
            return math::saturate(measured_curve_factor(cosine, m_values, m_num_values));
    }

    float2 ior() const {
        return make<float2>(-1.0f);
    }
private:
    const float3 *m_values;
    unsigned int m_num_values;
};

BSDF_API void measured_curve_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_sample(c, data, state, weight, layer, base, shading_normal, inherited_normal);
}

BSDF_API void measured_curve_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_evaluate(
        c, data, state, weight, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

BSDF_API void measured_curve_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_pdf(
        c, data, state, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void measured_curve_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_auxiliary(
        c, data, state, weight, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

#else

BSDF_API void measured_curve_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &base_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    weight = math::saturate(weight);

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float estimated_curve_factor = measured_curve_factor_estimate(nk1, curve_values, num_curve_values);

    const bool no_base = base.is_black();

    const float prob_layer = no_base ? 1.0f : estimated_curve_factor * weight;
    const bool sample_layer = no_base || (data->xi.z < prob_layer);
    if (sample_layer) {
        data->xi.z /= prob_layer;
        layer.sample(data, state, layer_normal);
    } else {
        data->xi.z = (1.0f - data->xi.z) / (1.0f - prob_layer);
        base.sample(data, state, base_normal);
    }

    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    const float nk2 = math::abs(math::dot(data->k2, layer_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, layer_normal, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0);

    BSDF_pdf_data pdf_data = to_pdf_data(data);
    if (sample_layer) {
        const float kh = math::abs(math::dot(data->k1, h));
        const float3 curve_factor = measured_curve_factor_eval(kh, curve_values, num_curve_values);
        data->bsdf_over_pdf *= curve_factor * weight / prob_layer;

        base.pdf(&pdf_data, state, base_normal);
        data->pdf = pdf_data.pdf * (1.0f - prob_layer) + data->pdf * prob_layer;
    } else {
        const float3 w_base =
            make_float3(1.0f, 1.0f, 1.0f) - weight * math::max(
                measured_curve_factor_eval(nk1, curve_values, num_curve_values),
                measured_curve_factor_eval(nk2, curve_values, num_curve_values));
        data->bsdf_over_pdf *= w_base / (1.0f - prob_layer);

        layer.pdf(&pdf_data, state, layer_normal);
        data->pdf = pdf_data.pdf * prob_layer + data->pdf * (1.0f - prob_layer);
    }
}

BSDF_API void measured_curve_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &base_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    weight = math::saturate(weight);

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float nk2 = math::abs(math::dot(data->k2, layer_normal));

    const bool backside_eval = math::dot(data->k2, geometry_normal) < 0.0f;
    const float3 h = compute_half_vector(
        data->k1, data->k2, layer_normal, nk2, backside_eval);

    const float kh = math::abs(math::dot(data->k1, h));
    const float3 curve_factor = measured_curve_factor_eval(kh, curve_values, num_curve_values);
    const float3 cf1 = measured_curve_factor_eval(nk1, curve_values, num_curve_values);
    const float3 cf2 = measured_curve_factor_eval(nk2, curve_values, num_curve_values);

    layer.evaluate(data, state, layer_normal, (weight * curve_factor) * inherited_weight); 
    if (base.is_black())
        return;

    const float prob_layer = weight * measured_curve_factor_estimate(
        nk1, curve_values, num_curve_values);
    const float pdf_layer = data->pdf * prob_layer;

    base.evaluate(
        data, state, base_normal, (1.0f - weight * math::max(cf1, cf2)) * inherited_weight); 
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}

BSDF_API void measured_curve_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &base_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    weight = math::saturate(weight);

    layer.pdf(data, state, layer_normal);
    if (base.is_black())
        return;

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float prob_layer = weight * measured_curve_factor_estimate(nk1, curve_values, num_curve_values);
    const float pdf_layer = data->pdf * prob_layer;

    base.pdf(data, state, base_normal);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}

BSDF_API void measured_curve_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &base_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    weight = math::saturate(weight);

    // assuming perfect reflection
    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float3 curve_factor = weight * measured_curve_factor_eval(nk1, curve_values, num_curve_values);

    layer.auxiliary(data, state, layer_normal, curve_factor * inherited_weight);
    if (base.is_black())
        return;

    base.auxiliary(data, state, base_normal, (1.0f - curve_factor) * inherited_weight);
}

#endif


/////////////////////////////////////////////////////////////////////
// bsdf color_measured_curve_layer(
//     color[<N>] curve_values,
//     color   weight               = color(1.0),
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state->normal()
// )
/////////////////////////////////////////////////////////////////////

// The HLSL backend does not support storing pointers, so we cannot use the
// templated curve_layer functions, but need to duplicate their code.
#if 0

class Color_measured_curve_eval {
public:
    Color_measured_curve_eval(
        const float3 *const values, const unsigned int num_values, const float3 &weight) :
        m_values(values), m_num_values(num_values), m_weight(math::saturate(weight)) {
    }

    float estimate(const float cosine) const {
        return math::luminance(eval(cosine));
    }

    float3 eval(const float cosine) const {
        if (m_num_values == 0)
            return make_float3(0.0f, 0.0f, 0.0f);
        else
            return m_weight * math::saturate(measured_curve_factor(cosine, m_values, m_num_values));
    }
    
    float2 ior() const {
        return make<float2>(-1.0f);
    }
private:
    const float3 *m_values;
    unsigned int m_num_values;
    float3 m_weight;

};

BSDF_API void color_measured_curve_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_sample(c, data, state, 1.0f, layer, base, shading_normal, inherited_normal);
}

BSDF_API void color_measured_curve_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_evaluate(
        c, data, state, 1.0f, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}

BSDF_API void color_measured_curve_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_pdf(
        c, data, state, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void color_measured_curve_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_auxiliary(
        c, data, state, 1.0f, layer, base, 
        shading_normal, inherited_normal, geometry_normal, inherited_weight);
}


#else

BSDF_API void color_measured_curve_layer_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &base_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &color_weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float weight = 1.0f; // TODO check if that is right

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float estimated_curve_factor = color_measured_curve_factor_estimate(
        nk1, curve_values, num_curve_values, color_weight);

    const bool no_base = base.is_black();

    const float prob_layer = no_base ? 1.0f : estimated_curve_factor * weight;
    const bool sample_layer = no_base || (data->xi.z < prob_layer);
    if (sample_layer) {
        data->xi.z /= prob_layer;
        layer.sample(data, state, layer_normal);
    } else {
        data->xi.z = (1.0f - data->xi.z) / (1.0f - prob_layer);
        base.sample(data, state, base_normal);
    }

    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    const float nk2 = math::abs(math::dot(data->k2, layer_normal));
    const float3 h = compute_half_vector(
        data->k1, data->k2, layer_normal, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0);

    BSDF_pdf_data pdf_data = to_pdf_data(data);
    if (sample_layer) {
        const float kh = math::abs(math::dot(data->k1, h));
        const float3 curve_factor = color_measured_curve_factor_eval(
            kh, curve_values, num_curve_values, color_weight);
        data->bsdf_over_pdf *= curve_factor * weight / prob_layer;

        base.pdf(&pdf_data, state, base_normal);
        data->pdf = pdf_data.pdf * (1.0f - prob_layer) + data->pdf * prob_layer;
    } else {
        const float3 w_base =
            make_float3(1.0f, 1.0f, 1.0f) - weight * math::max(
                color_measured_curve_factor_eval(
                    nk1, curve_values, num_curve_values, color_weight),
                color_measured_curve_factor_eval(
                    nk2, curve_values, num_curve_values, color_weight));
        data->bsdf_over_pdf *= w_base / (1.0f - prob_layer);

        layer.pdf(&pdf_data, state, layer_normal);
        data->pdf = pdf_data.pdf * prob_layer + data->pdf * (1.0f - prob_layer);
    }
}

BSDF_API void color_measured_curve_layer_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &base_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &color_weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float weight = 1.0f; // TODO check if that is right

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float nk2 = math::abs(math::dot(data->k2, layer_normal));

    const bool backside_eval = math::dot(data->k2, geometry_normal) < 0.0f;
    const float3 h = compute_half_vector(
        data->k1, data->k2, layer_normal, nk2, backside_eval);

    const float kh = math::abs(math::dot(data->k1, h));
    const float3 curve_factor = color_measured_curve_factor_eval(
            kh, curve_values, num_curve_values, color_weight);
    const float3 cf1 = color_measured_curve_factor_eval(
            nk1, curve_values, num_curve_values, color_weight);
    const float3 cf2 = color_measured_curve_factor_eval(
            nk2, curve_values, num_curve_values, color_weight);

    layer.evaluate(data, state, layer_normal, (weight * curve_factor) * inherited_weight); 
    if (base.is_black())
        return;

    const float prob_layer = weight * color_measured_curve_factor_estimate(
            nk1, curve_values, num_curve_values, color_weight);
    const float pdf_layer = data->pdf * prob_layer;

    base.evaluate(
        data, state, base_normal, (1.0f - weight * math::max(cf1, cf2)) * inherited_weight);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}

BSDF_API void color_measured_curve_layer_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &base_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &color_weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float weight = 1.0f; // TODO check if that is right

    layer.pdf(data, state, layer_normal);
    if (base.is_black())
        return;

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float prob_layer = weight * color_measured_curve_factor_estimate(
            nk1, curve_values, num_curve_values, color_weight);
    const float pdf_layer = data->pdf * prob_layer;

    base.pdf(data, state, base_normal);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}

BSDF_API void color_measured_curve_layer_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const float3 &color_weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 layer_normal, geometry_normal;
    get_oriented_normals(
        layer_normal, geometry_normal, normal, state->geometry_normal(), data->k1);

    const float weight = 1.0f; // TODO check if that is right

    const float nk1 = math::saturate(math::dot(data->k1, layer_normal));
    const float3 curve_factor = weight * color_measured_curve_factor_eval(
        nk1, curve_values, num_curve_values, color_weight);

    layer.auxiliary(data, state, layer_normal, curve_factor * inherited_weight);
    if (base.is_black())
        return;

    base.auxiliary(data, state, normal, (1.0f - curve_factor) * inherited_weight);
}

#endif


/////////////////////////////////////////////////////////////////////
// df normalized_mix(
//     df_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

template<typename TDF_sample_data, typename TDF_component>
BSDF_INLINE void normalized_mix_df_sample(
    TDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    float p;
    for (sampled_idx = 0; ; ++sampled_idx) {
        p = math::saturate(components[sampled_idx].weight) * inv_w_sum;
        const float cdf = prev_cdf + p;
        if (data->xi.z < cdf || sampled_idx == num_components - 1) {
            data->xi.z = (data->xi.z - prev_cdf) / p;
            break;
        }
        prev_cdf = cdf;
    }

    components[sampled_idx].component.sample(data, state, inherited_normal);
    if (data->event_type == 0) // BSDF_EVENT_ABSORB or EDF_NO_EMISSION or ...
        return;

    if (w_sum < 1.0f)
        set_df_over_pdf(data, get_df_over_pdf(data) * w_sum);

    data->pdf *= p;
    auto pdf_data = to_pdf_data(data);
    for (unsigned int i = 0; i < num_components; ++i) {
        if (i == sampled_idx)
            continue;
        const float q = math::saturate(components[i].weight) * inv_w_sum;
        if (q > 0.0f) {
            components[i].component.pdf(&pdf_data, state, inherited_normal);
            data->pdf += q * pdf_data.pdf;
        }
    }
}

template<typename TDF_evaluate_data, typename TDF_component>
BSDF_INLINE void normalized_mix_df_evaluate(
    TDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;
    const float normalize = w_sum > 1.0f ? inv_w_sum : 1.0f;

    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        const float w = math::saturate(components[i].weight);
        components[i].component.evaluate(
            data, state, inherited_normal, (w * normalize) * inherited_weight);
        pdf += data->pdf * (w * inv_w_sum);
    }

    set_cos(data, num_components == 0 ? 0.0f : get_cos(data));
    data->pdf = pdf;
}

template<typename TDF_pdf_data, typename TDF_component>
BSDF_INLINE void normalized_mix_df_pdf(
    TDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        components[i].component.pdf(data, state, inherited_normal);
        pdf += data->pdf * math::saturate(components[i].weight) * inv_w_sum;
    }
    data->pdf = pdf;
}

template<typename TDF_auxiliary_data, typename TDF_component>
BSDF_INLINE void normalized_mix_df_auxiliary(
    TDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    if (w_sum <= 0.0f)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;
    const float normalize = w_sum > 1.0f ? inv_w_sum : 1.0f;

    float3 albedo = make<float3>(0.0f);
    float3 normal = make<float3>(0.0f);
    for (unsigned int i = 0; i < num_components; ++i)
    {
        const float w = math::saturate(components[i].weight);
        components[i].component.auxiliary(
            data, state, inherited_normal, (w * normalize) * inherited_weight);
    }
}


/////////////////////////////////////////////////////////////////////
// df clamped_mix(
//     df_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

template<typename TDF_sample_data, typename TDF_component>
BSDF_INLINE void clamped_mix_df_sample(
    TDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active) {
        final_weight = math::saturate(components[num_active].weight);
        const float f = w_sum + final_weight;
        if (f > 1.0f) {
            final_weight = 1.0f - w_sum;
            w_sum = 1.0f;
            ++num_active;
            break;
        }
        w_sum = f;
    }

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    float p;
    for (sampled_idx = 0; ; ++sampled_idx) {
        p = (sampled_idx == num_active - 1 ? final_weight :
             math::saturate(components[sampled_idx].weight)) * inv_w_sum;
        const float cdf = prev_cdf + p;
        if (data->xi.z < cdf || sampled_idx == num_active - 1) {
            data->xi.z = (data->xi.z - prev_cdf) / p;
            break;
        }
        prev_cdf = cdf;
    }

    components[sampled_idx].component.sample(data, state, inherited_normal);
    if (data->event_type == 0) // BSDF_EVENT_ABSORB or EDF_NO_EMISSION or ...
        return;

    set_df_over_pdf(data, get_df_over_pdf(data) * w_sum);

    data->pdf *= p;
    auto pdf_data = to_pdf_data(data);
    for (unsigned int i = 0; i < num_active; ++i) {
        if (i == sampled_idx)
            continue;
        const float q = (i == num_active - 1 ? final_weight :
                         math::saturate(components[i].weight)) * inv_w_sum;
        if (q > 0.0f) {
            components[i].component.pdf(&pdf_data, state, inherited_normal);
            data->pdf += q * pdf_data.pdf;
        }
    }
}

template<typename TDF_evaluate_data, typename TDF_component>
BSDF_INLINE void clamped_mix_df_evaluate(
    TDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active) {
        final_weight = math::saturate(components[num_active].weight);
        const float f = w_sum + final_weight;
        if (f > 1.0f) {
            final_weight = 1.0f - w_sum;
            w_sum = 1.0f;
            ++num_active;
            break;
        }
        w_sum = f;
    }

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    float3 df = make_float3(0.0f, 0.0f, 0.0f);
    float3 df_glossy = make_float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_active; ++i) {
        float weight = i == num_active - 1 ? final_weight : math::saturate(components[i].weight);
        components[i].component.evaluate(data, state, inherited_normal, weight * inherited_weight);
        pdf += data->pdf * weight * inv_w_sum;
    }

    set_cos(data, num_components == 0 ? 0.0f : get_cos(data));
    data->pdf = pdf;
}

template<typename TDF_pdf_data, typename TDF_component>
BSDF_INLINE void clamped_mix_df_pdf(
    TDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active) {
        final_weight = math::saturate(components[num_active].weight);
        const float f = w_sum + final_weight;
        if (f > 1.0f) {
            final_weight = 1.0f - w_sum;
            w_sum = 1.0f;
            ++num_active;
            break;
        }
        w_sum = f;
    }

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_active; ++i) {
        components[i].component.pdf(data, state, inherited_normal);
        float weight = i == num_active - 1 ? final_weight : math::saturate(components[i].weight);
        pdf += data->pdf * weight * inv_w_sum;
    }
    data->pdf = pdf;
}


template<typename TDF_auxiliary_data, typename TDF_component>
BSDF_INLINE void clamped_mix_df_auxiliary(
    TDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active)
    {
        final_weight = math::saturate(components[num_active].weight);
        const float f = w_sum + final_weight;
        if (f > 1.0f)
        {
            final_weight = 1.0f - w_sum;
            w_sum = 1.0f;
            ++num_active;
            break;
        }
        w_sum = f;
    }

    if (w_sum <= 0.0f)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    float3 albedo = make<float3>(0.0f);
    float3 normal = make<float3>(0.0f);
    for (unsigned int i = 0; i < num_active; ++i)
    {
        float weight = i == num_active - 1 ? final_weight : math::saturate(components[i].weight);
        components[i].component.auxiliary(data, state, inherited_normal, weight * inherited_weight);
    }
}


/////////////////////////////////////////////////////////////////////
// df color_normalized_mix(
//     color_df_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

template<typename TDF_sample_data, typename TDF_component>
BSDF_INLINE void color_normalized_mix_df_sample(
    TDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float3 w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    if (w_sum.x <= 0.0f && w_sum.y <= 0.0f && w_sum.z <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / math::luminance(w_sum);

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    float p;
    for (sampled_idx = 0; ; ++sampled_idx) {
        p = math::luminance(math::saturate(components[sampled_idx].weight)) * inv_w_sum;
        const float cdf = prev_cdf + p;
        if (data->xi.z < cdf || sampled_idx == num_components - 1) {
            data->xi.z = (data->xi.z - prev_cdf) / (cdf - prev_cdf);
            break;
        }
        prev_cdf = cdf;
    }

    components[sampled_idx].component.sample(data, state, inherited_normal);
    if (data->event_type == 0) // BSDF_EVENT_ABSORB or EDF_NO_EMISSION or ...
        return;

    set_df_over_pdf(data, get_df_over_pdf<TDF_sample_data>(data) *
                    math::saturate(components[sampled_idx].weight) /
                    (math::max(w_sum, make_float3(1.0f, 1.0f, 1.0f)) * p));

    data->pdf *= p;
    auto pdf_data = to_pdf_data(data);
    for (unsigned int i = 0; i < num_components; ++i) {
        if (i == sampled_idx)
            continue;
        const float q = math::luminance(math::saturate(components[i].weight)) * inv_w_sum;
        if (q > 0.0f) {
            components[i].component.pdf(&pdf_data, state, inherited_normal);
            data->pdf += q * pdf_data.pdf;
        }
    }
}

template<typename TDF_evaluate_data, typename TDF_component>
BSDF_INLINE void color_normalized_mix_df_evaluate(
    TDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float3 w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    w_sum = math::max(make_float3(0.0f, 0.0f, 0.0f), w_sum);

    if (w_sum.x <= 0.0f && w_sum.y <= 0.0f && w_sum.z <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / math::luminance(w_sum);
    const float3 normalize = make_float3(
        w_sum.x > 1.0f ? 1.0f / w_sum.x : 1.0f,
        w_sum.y > 1.0f ? 1.0f / w_sum.y : 1.0f,
        w_sum.z > 1.0f ? 1.0f / w_sum.z : 1.0f);

    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        const float3 w = math::saturate(components[i].weight);
        components[i].component.evaluate(
            data, state, inherited_normal, w * normalize * inherited_weight); 
        pdf += data->pdf * math::luminance(w) * inv_w_sum;
    }

    set_cos(data, num_components == 0 ? 0.0f : get_cos(data));
    data->pdf = pdf;
}

template<typename TDF_pdf_data, typename TDF_component>
BSDF_INLINE void color_normalized_mix_df_pdf(
    TDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float w_sum = math::luminance(math::saturate(components[0].weight));
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::luminance(math::saturate(components[i].weight));

    if (w_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        components[i].component.pdf(data, state, inherited_normal);
        pdf += data->pdf * math::luminance(math::saturate(components[i].weight)) * inv_w_sum;
    }
    data->pdf = pdf;
}

template<typename TDF_auxiliary_data, typename TDF_component>
BSDF_INLINE void color_normalized_mix_df_auxiliary(
    TDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float3 w_sum = math::saturate(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::saturate(components[i].weight);

    w_sum = math::max(make_float3(0.0f, 0.0f, 0.0f), w_sum);

    if (w_sum.x <= 0.0f && w_sum.y <= 0.0f && w_sum.z <= 0.0f)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    const float inv_w_sum = 1.0f / math::luminance(w_sum);
    const float3 normalize = make_float3(
        w_sum.x > 1.0f ? 1.0f / w_sum.x : 1.0f,
        w_sum.y > 1.0f ? 1.0f / w_sum.y : 1.0f,
        w_sum.z > 1.0f ? 1.0f / w_sum.z : 1.0f);

    float3 albedo = make<float3>(0.0f);
    float3 normal = make<float3>(0.0f);
    for (unsigned int i = 0; i < num_components; ++i)
    {
        const float3 w = math::saturate(components[i].weight);
        components[i].component.auxiliary(
            data, state, inherited_normal,  w * normalize * inherited_weight);
    }

}


/////////////////////////////////////////////////////////////////////
// df color_clamped_mix(
//     color_df_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

template<typename TDF_sample_data, typename TDF_component>
BSDF_INLINE void color_clamped_mix_df_sample(
    TDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int num_active = 0;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components && clamp_mask != 7; ++i) {
        ++num_active;
        float3 w = math::saturate(components[i].weight);
        if (clamp_mask & 1)
            w.x = 0.0f;
        if (clamp_mask & 2)
            w.y = 0.0f;
        if (clamp_mask & 4)
            w.z = 0.0f;

        mix_sum += w;

        if ((clamp_mask & 1) == 0 && mix_sum.x > 1.0f) {
            w.x += 1.0f - mix_sum.x;
            mix_sum.x = 1.0f;
            clamp_mask |= 1;
        }
        if ((clamp_mask & 2) == 0 && mix_sum.y > 1.0f) {
            w.y += 1.0f - mix_sum.y;
            mix_sum.y = 1.0f;
            clamp_mask |= 2;
        }
        if ((clamp_mask & 4) == 0 && mix_sum.z > 1.0f) {
            w.z += 1.0f - mix_sum.z;
            mix_sum.z = 1.0f;
            clamp_mask |= 4;
        }

        lw_sum += math::luminance(w);
    }

    if (lw_sum <= 0.0f) {
        no_contribution(data, g.n.shading_normal);
        return;
    }


    const float inv_lw_sum = 1.0f / lw_sum;

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    float p;
    float3 w;
    mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    clamp_mask = 0;
    for (sampled_idx = 0; ; ++sampled_idx) {

        w = math::saturate(components[sampled_idx].weight);
        if (clamp_mask & 1)
            w.x = 0.0f;
        if (clamp_mask & 2)
            w.y = 0.0f;
        if (clamp_mask & 4)
            w.z = 0.0f;

        mix_sum += w;

        if ((clamp_mask & 1) == 0 && mix_sum.x > 1.0f) {
            w.x += 1.0f - mix_sum.x;
            mix_sum.x = 1.0f;
            clamp_mask |= 1;
        }
        if ((clamp_mask & 2) == 0 && mix_sum.y > 1.0f) {
            w.y += 1.0f - mix_sum.y;
            mix_sum.y = 1.0f;
            clamp_mask |= 2;
        }
        if ((clamp_mask & 4) == 0 && mix_sum.z > 1.0f) {
            w.z += 1.0f - mix_sum.z;
            mix_sum.z = 1.0f;
            clamp_mask |= 4;
        }

        p = math::luminance(w) * inv_lw_sum;

        const float cdf = prev_cdf + p;
        if (data->xi.z < cdf || sampled_idx == num_active - 1) {
            data->xi.z = (data->xi.z - prev_cdf) / p;
            break;
        }
        prev_cdf = cdf;
    }

    components[sampled_idx].component.sample(data, state, inherited_normal);
    if (data->event_type == 0) // BSDF_EVENT_ABSORB or DF_NO_EMISSION or ...
        return;

    set_df_over_pdf(data, get_df_over_pdf(data) * w / p);

    data->pdf *= p;
    mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    clamp_mask = 0;
    auto pdf_data = to_pdf_data(data);
    for (unsigned int i = 0; i < num_active && clamp_mask != 7; ++i) {

        w = math::saturate(components[i].weight);
        if (clamp_mask & 1)
            w.x = 0.0f;
        if (clamp_mask & 2)
            w.y = 0.0f;
        if (clamp_mask & 4)
            w.z = 0.0f;

        mix_sum += w;

        if ((clamp_mask & 1) == 0 && mix_sum.x > 1.0f) {
            w.x += 1.0f - mix_sum.x;
            mix_sum.x = 1.0f;
            clamp_mask |= 1;
        }
        if ((clamp_mask & 2) == 0 && mix_sum.y > 1.0f) {
            w.y += 1.0f - mix_sum.y;
            mix_sum.y = 1.0f;
            clamp_mask |= 2;
        }
        if ((clamp_mask & 4) == 0 && mix_sum.z > 1.0f) {
            w.z += 1.0f - mix_sum.z;
            mix_sum.z = 1.0f;
            clamp_mask |= 4;
        }

        const float q = math::luminance(w) * inv_lw_sum;
        if (q > 0.0f) {
            components[i].component.pdf(&pdf_data, state, inherited_normal);
            data->pdf += q * pdf_data.pdf;
        }
    }
}

template<typename TDF_evaluate_data, typename TDF_component>
BSDF_INLINE void color_clamped_mix_df_evaluate(
    TDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float pdf = 0.0f;
    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components && clamp_mask != 7; ++i) {
        float3 w = math::saturate(components[i].weight);
        if (clamp_mask & 1)
            w.x = 0.0f;
        if (clamp_mask & 2)
            w.y = 0.0f;
        if (clamp_mask & 4)
            w.z = 0.0f;

        mix_sum += w;

        if ((clamp_mask & 1) == 0 && mix_sum.x > 1.0f) {
            w.x += 1.0f - mix_sum.x;
            mix_sum.x = 1.0f;
            clamp_mask |= 1;
        }
        if ((clamp_mask & 2) == 0 && mix_sum.y > 1.0f) {
            w.y += 1.0f - mix_sum.y;
            mix_sum.y = 1.0f;
            clamp_mask |= 2;
        }
        if ((clamp_mask & 4) == 0 && mix_sum.z > 1.0f) {
            w.z += 1.0f - mix_sum.z;
            mix_sum.z = 1.0f;
            clamp_mask |= 4;
        }

        components[i].component.evaluate(data, state, inherited_normal, w * inherited_weight);

        const float lw = math::luminance(w);
        lw_sum += lw;
        pdf += data->pdf * lw;
    }

    set_cos(data, num_components == 0 ? 0.0f : get_cos(data));
    if (lw_sum > 0.0f)
        data->pdf = pdf / lw_sum;
}

template<typename TDF_pdf_data, typename TDF_component>
BSDF_INLINE void color_clamped_mix_df_pdf(
    TDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float pdf = 0.0f;

    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components && clamp_mask != 7; ++i) {
        float3 w = math::saturate(components[i].weight);
        if (clamp_mask & 1)
            w.x = 0.0f;
        if (clamp_mask & 2)
            w.y = 0.0f;
        if (clamp_mask & 4)
            w.z = 0.0f;

        mix_sum += w;

        if ((clamp_mask & 1) == 0 && mix_sum.x > 1.0f) {
            w.x += 1.0f - mix_sum.x;
            mix_sum.x = 1.0f;
            clamp_mask |= 1;
        }
        if ((clamp_mask & 2) == 0 && mix_sum.y > 1.0f) {
            w.y += 1.0f - mix_sum.y;
            mix_sum.y = 1.0f;
            clamp_mask |= 2;
        }
        if ((clamp_mask & 4) == 0 && mix_sum.z > 1.0f) {
            w.z += 1.0f - mix_sum.z;
            mix_sum.z = 1.0f;
            clamp_mask |= 4;
        }

        components[i].component.pdf(data, state, inherited_normal);
        const float lw = math::luminance(w);
        lw_sum += lw;
        pdf += data->pdf * lw;
    }

    if (lw_sum > 0.0f)
        pdf /= lw_sum;

    data->pdf = pdf;
}


template<typename TDF_auxiliary_data, typename TDF_component>
BSDF_INLINE void color_clamped_mix_df_auxiliary(
    TDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const TDF_component *components,
    const unsigned int num_components)
{
    Geometry g;
    if (!get_geometry(g, inherited_normal, state->texture_tangent_u(0), data->k1, state)) {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    if (num_components == 0)
    {
        no_contribution(data, g.n.shading_normal);
        return;
    }

    float3 albedo = make<float3>(0.0f);
    float3 normal = make<float3>(0.0f);

    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components && clamp_mask != 7; ++i)
    {
        float3 w = math::saturate(components[i].weight);
        if (clamp_mask & 1)
            w.x = 0.0f;
        if (clamp_mask & 2)
            w.y = 0.0f;
        if (clamp_mask & 4)
            w.z = 0.0f;

        mix_sum += w;

        if ((clamp_mask & 1) == 0 && mix_sum.x > 1.0f)
        {
            w.x += 1.0f - mix_sum.x;
            mix_sum.x = 1.0f;
            clamp_mask |= 1;
        }
        if ((clamp_mask & 2) == 0 && mix_sum.y > 1.0f)
        {
            w.y += 1.0f - mix_sum.y;
            mix_sum.y = 1.0f;
            clamp_mask |= 2;
        }
        if ((clamp_mask & 4) == 0 && mix_sum.z > 1.0f)
        {
            w.z += 1.0f - mix_sum.z;
            mix_sum.z = 1.0f;
            clamp_mask |= 4;
        }

        components[i].component.auxiliary(data, state, inherited_normal, w * inherited_weight);
    }
}


//-------------------------------------------------------------------
// mixing BSDFs
//-------------------------------------------------------------------

BSDF_API void normalized_mix_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void normalized_mix_bsdf_evaluate (
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const BSDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void normalized_mix_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void normalized_mix_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const BSDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_auxiliary(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_normalized_mix_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_normalized_mix_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_normalized_mix_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_normalized_mix_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_auxiliary(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void clamped_mix_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void clamped_mix_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const BSDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void clamped_mix_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void clamped_mix_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const BSDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_auxiliary(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_clamped_mix_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_clamped_mix_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_clamped_mix_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_clamped_mix_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_auxiliary(
        data, state, inherited_normal, inherited_weight, components, num_components);
}


/////////////////////////////////////////////////////////////////////
// edf()
/////////////////////////////////////////////////////////////////////

BSDF_API void black_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal)
{
    no_emission(data);
}

BSDF_API void black_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight)
{
    no_emission(data);
}

BSDF_API void black_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal)
{
    no_emission(data);
}

BSDF_API void black_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight)
{
    no_emission(data);
}


/////////////////////////////////////////////////////////////////////
// EDF Utilities
/////////////////////////////////////////////////////////////////////
namespace
{
    // compute cosine between outgoing direction and main direction (normal)
    // returns true if the cosine is between 0.0 and 1.0
    BSDF_INLINE bool edf_compute_cos(
        const float3&   outgoing_dir,
        const State*    state,
        const float3&   inherited_normal,
        float&          out_cos)
    {
        float3 shading_normal, geometry_normal;
        get_oriented_normals(
            shading_normal, geometry_normal,
            inherited_normal, state->geometry_normal(),
            outgoing_dir);

        out_cos = math::dot(outgoing_dir, shading_normal);
        bool valid = out_cos > 0.0f &&  out_cos <= 1.0f;
        out_cos = math::max(out_cos, 0.0f);
        return valid;
    }

    // compute cosine between outgoing direction and main direction (y-axis of the global frame)
    // returns true if the cosine is between 0.0 and 1.0
    BSDF_INLINE bool edf_compute_cos(
        const float3&   outgoing_dir,
        const float3x3& global_frame,
        float&          out_cos)
    {
        out_cos = math::dot(outgoing_dir, global_frame.col1);
        bool valid = out_cos > 0.0f &&  out_cos <= 1.0f;
        out_cos = math::max(out_cos, 0.0f);
        return valid;
    }

    // compute the outgoing direction in the world space from theta, phi in tangent space.
    BSDF_INLINE bool edf_compute_outgoing_direction(
        EDF_sample_data    *data,
        const State*        state,
        const float3&       inherited_normal,
        const float         sin_theta, const float cos_theta,
        const float         sin_phi, const float cos_phi,
        float3&             out_world_dir)
    {
        // to indicate which normals are used. get_oriented_normals not possible without given k1
        float3 shading_normal = inherited_normal;
        float3 geometry_normal = state->geometry_normal();

        // get world coordinate basis
        float3 x_axis, z_axis;
        if (!get_bumped_basis(x_axis, z_axis, state->texture_tangent_u(0), shading_normal))
            return false;

        // transform to world
        out_world_dir = math::normalize(
            x_axis * sin_theta * cos_phi +
            shading_normal * cos_theta +
            z_axis * sin_theta * sin_phi);
        return true;
    }

    // compute the outgoing direction in the world space from theta, phi in global frame space.
    BSDF_INLINE bool edf_compute_outgoing_direction(
        const float3x3&     global_frame,
        const float         sin_theta, const float cos_theta,
        const float         sin_phi, const float cos_phi,
        float3&             out_world_dir)
    {
        // transform to world
        out_world_dir = math::normalize(
            global_frame.col0 * sin_theta * cos_phi +
            global_frame.col1 * cos_theta +
            global_frame.col2 * sin_theta * sin_phi);
        return true;
    }
}


/////////////////////////////////////////////////////////////////////
// edf diffuse_edf(
//     uniform string handle = ""
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void diffuse_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const int handle)
{
    // to indicate which normals are used. get_oriented_normals not possible without given k1
    float3 shading_normal = inherited_normal;
    float3 geometry_normal = state->geometry_normal();

    // sample direction and transform to world coordinates
    const float3 cosh = cosine_hemisphere_sample(make_float2(data->xi.x, data->xi.y));
    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, state->texture_tangent_u(0), shading_normal)) {
        no_emission(data);
        return;
    }
    data->k1 = math::normalize(x_axis * cosh.x + shading_normal * cosh.y + z_axis * cosh.z);

    if (cosh.y <= 0.0f || math::dot(data->k1, geometry_normal) <= 0.0f) {
        no_emission(data);
        return;
    }

    data->pdf = cosh.y * float(M_ONE_OVER_PI);
    data->edf_over_pdf = make<float3>(1.0f);
    data->event_type = EDF_EVENT_EMISSION;
    data->handle = handle;
}

BSDF_API void diffuse_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const int handle)
{
    float cos;
    edf_compute_cos(data->k1, state, inherited_normal, cos);

    data->cos = cos;
    data->pdf = cos * float(M_ONE_OVER_PI);
    add_elemental_edf_evaluate_contribution(data, handle, make<float3>(float(M_ONE_OVER_PI)) * inherited_weight);
}

BSDF_API void diffuse_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const int handle)
{
    float cos;
    edf_compute_cos(data->k1, state, inherited_normal, cos);
    data->pdf = cos * float(M_ONE_OVER_PI);
}

BSDF_API void diffuse_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const int handle)
{
    no_emission(data);
}


/////////////////////////////////////////////////////////////////////
// edf spot_edf(
//     uniform float     exponent,
//     uniform float     spread              = math::PI,
//     uniform bool      global_distribution = true,
//     uniform float3x3  global_frame        = float3x3(1.0),
//     uniform string    handle              = ""
// )
/////////////////////////////////////////////////////////////////////

namespace
{
    inline float spot_edf_prepare_exponent(float exponent)
    {
        return math::max(0.0f, exponent); // limit exponent to meaningful range
    }

    inline float spot_edf_prepare_spread(float spread)
    {
        // limit spread to meaningful range
        spread = math::clamp(spread, 0.0f, float(2.0 * M_PI));

        // spread - Angle of the cone to which the cosine distribution is restricted.
        //          The hemispherical domain for the distribution is rescaled to this cone.
        //          default is math::PI
        return math::cos(spread * 0.5f); // to compare against the cosine at the normal
    }

    BSDF_INLINE float spot_edf_pdf(float s, float k, float cos_theta)
    {
        return (k + 1.0f) * math::pow(cos_theta - s, k)
            / (math::pow(1.0f - s, k + 1.0f) * (float) (2.0 * M_PI));
    }
}


BSDF_API void spot_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float exponent,
    const float spread,
    const bool global_distribution,
    const float3x3 &global_frame,
    const int handle)
{
    // limit and convert input parameters
    float s = spot_edf_prepare_spread(spread);
    float k = spot_edf_prepare_exponent(exponent);

    // sample direction
    const float phi = (float) (2.0 * M_PI) * data->xi.x;
    const float cos_theta = s + (1.0f - s) * math::pow(1.0f - data->xi.y, 1.0f / (k + 1.0f));
    const float sin_theta = math::sin(math::acos(cos_theta));
    if ((cos_theta - s) < 0.0f) {
        no_emission(data);
        return;
    }

    if (global_distribution) {
        no_emission(data);
        return;
    } else {
        // transform to world  coordinates
        if (!edf_compute_outgoing_direction(data, state, inherited_normal,
            sin_theta, cos_theta, math::sin(phi), math::cos(phi), data->k1))
        {
            no_emission(data);
            return;
        }

        // check for lower hemisphere sample
        if (math::dot(data->k1, state->geometry_normal()) <= 0.0f) {
            no_emission(data);
            return;
        }

        // edf * cos_theta / pdf
        data->edf_over_pdf = make<float3>((k * k + 3.0f * k + 2.0f) * cos_theta
                                          / (k * k + k * (2.0f + s) + s + 1.0f));
    }

    data->pdf = spot_edf_pdf(s, k, cos_theta);
    data->event_type = EDF_EVENT_EMISSION;
    data->handle = handle;
}

BSDF_API void spot_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float exponent,
    const float spread,
    const bool global_distribution,
    const float3x3 &global_frame,
    const int handle)
{
    // limit and convert input parameters
    float s = spot_edf_prepare_spread(spread);
    float k = spot_edf_prepare_exponent(exponent);

    // get angle to the main emission direction
    float cos_theta;
    if (global_distribution) {
        no_emission(data);
        return;
    } else {
        edf_compute_cos(data->k1, state, inherited_normal, cos_theta);
    }

    // lobe cut off because of the spread
    if ((cos_theta - s) < 0.0f) {
        no_emission(data);
        return;
    }

    // un-normalized edf
    float edf = math::pow((cos_theta - s) / (1.0f - s), k);

    // normalization term
    float normalization;
    if (global_distribution) {
        normalization = 1.0f;
    } else {
        //edf *= cos_theta; // projection
        normalization = (k*k + 3.0f*k + 2.0f) / (2.0f * (float) M_PI * (k + 1 - k * s - s * s));
    }

    data->cos = cos_theta;
    data->pdf = spot_edf_pdf(s, k, cos_theta);
    
    // normalized edf (not cosine corrected)
    add_elemental_edf_evaluate_contribution(
        data, handle, make<float3>(edf * normalization) * inherited_weight);
}

BSDF_API void spot_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float exponent,
    const float spread,
    const bool global_distribution,
    const float3x3 &global_frame,
    const int handle)
{
    // limit and convert input parameters
    float s = spot_edf_prepare_spread(spread);
    float k = spot_edf_prepare_exponent(exponent);

    // get angle to the main emission direction
    float cos_theta;
    if (global_distribution) {
        no_emission(data);
        return;
    } else {
        edf_compute_cos(data->k1, state, inherited_normal, cos_theta);
    }

    data->pdf = spot_edf_pdf(s, k, cos_theta);
}

BSDF_API void spot_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float exponent,
    const float spread,
    const bool global_distribution,
    const float3x3 &global_frame,
    const int handle)
{
    no_emission(data);
}


/////////////////////////////////////////////////////////////////////
// edf measured_edf(
//     uniform light_profile  profile,
//     uniform float          multiplier = 1.0,
//     uniform bool           global_distribution = true,
//     uniform float3x3       global_frame        = float3x3(1.0),
//     float3                 tangent_u           = state::texture_tangent_u(0),
//     uniform string         handle = ""
// )
/////////////////////////////////////////////////////////////////////

BSDF_INLINE void lightprofile_sample(
    EDF_sample_data *data,
    State *state,
    const Geometry &g,
    const unsigned light_profile_id,
    const float multiplier,
    const bool global_distribution,
    const float3x3 &global_frame)
{
    // sample and check for valid a result
    float3 polar_pdf = state->light_profile_sample(light_profile_id, xyz(data->xi));
    if (polar_pdf.x < 0.0f) {
        no_emission(data);
        return;
    }

    // transform to world
    float2 sin_theta_phi, cos_theta_phi;
    math::sincos(make<float2>(polar_pdf.x, polar_pdf.y), &sin_theta_phi, &cos_theta_phi);

    // local to world space
    float scale;
    if (global_distribution)  {
        no_emission(data);
        return;
    } else {
        data->k1 = math::normalize(
            g.x_axis            * sin_theta_phi.x * cos_theta_phi.y +
            g.n.shading_normal  * cos_theta_phi.x +
            g.z_axis            * sin_theta_phi.x * sin_theta_phi.y);

        // check for lower hemisphere sample
        scale = math::dot(data->k1, state->geometry_normal());
        if (scale <= 0.0f) {
            no_emission(data);
            return;
        }
    }

    // evaluate the light profile for the sampled direction
    float edf = math::max(0.0f, multiplier)
              * state->light_profile_evaluate(light_profile_id,
                                              make<float2>(polar_pdf.x, polar_pdf.y));

    data->pdf = polar_pdf.z;
    data->edf_over_pdf = make<float3>(edf * scale / polar_pdf.z);
    data->event_type = EDF_EVENT_EMISSION;
}



BSDF_API void measured_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    //
    const unsigned light_profile_id,
    const float multiplier,
    const bool global_distribution,
    const float3x3 &global_frame,
    const float3 &tangent_u,
    const int handle)
{

    Geometry g;
    g.n.shading_normal = inherited_normal;
    g.n.geometry_normal = math::dot(state->geometry_normal(), inherited_normal) > 0 ?
        state->geometry_normal() : -state->geometry_normal();

    get_bumped_basis(g.x_axis, g.z_axis, tangent_u, inherited_normal);

    lightprofile_sample(data, state, g, light_profile_id, multiplier,
                        global_distribution, global_frame);
}


template<typename TEDF_data>
BSDF_INLINE float3 lightprofile_eval_and_pdf(
    TEDF_data *data,
    State *state,
    const Geometry &g,
    const unsigned light_profile_id,
    const float multiplier,
    const bool global_distribution,
    const float3x3 &global_frame)
{
    float2 outgoing_polar;
    float cos;
    if (global_distribution) {
        no_emission(data);
        return make<float3>(0.0f);
    } else {
        // internal to local (assuming an orthonormal base)
        const float3 outgoing = math::normalize(make_float3(
            math::dot(data->k1, g.x_axis),
            math::dot(data->k1, g.n.shading_normal),
            math::dot(data->k1, g.z_axis)));

        // local to polar coords
        outgoing_polar.x = math::acos(outgoing.y);
        outgoing_polar.y = math::atan2(outgoing.z, outgoing.x);

        cos = math::dot(data->k1, g.n.shading_normal);
    }

    float intensity = math::max(0.0f, multiplier)
                    * state->light_profile_evaluate(light_profile_id, outgoing_polar);

    set_cos(data, cos);
    set_pdf(data, state->light_profile_pdf(light_profile_id, outgoing_polar));
    return make<float3>(intensity);
}

BSDF_API void measured_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    //
    const unsigned light_profile_id,
    const float multiplier,
    const bool global_distribution,
    const float3x3 &global_frame,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    g.n.shading_normal = inherited_normal;
    g.n.geometry_normal = state->geometry_normal() *
        (math::dot(state->geometry_normal(), inherited_normal) > 0.0f ? 1.0f : -1.0f);

    if (!global_distribution && math::dot(data->k1, g.n.geometry_normal) <= 0.0f) {
        no_emission(data);
        return;
    }

    get_bumped_basis(g.x_axis, g.z_axis, tangent_u, g.n.shading_normal);

    const float3 edf = lightprofile_eval_and_pdf<EDF_evaluate_data>(
        data, state, g, light_profile_id, multiplier, global_distribution, global_frame);

    add_elemental_edf_evaluate_contribution(data, handle, edf * inherited_weight);
}

BSDF_API void measured_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    //
    const unsigned light_profile_id,
    const float multiplier,
    const bool global_distribution,
    const float3x3 &global_frame,
    const float3 &tangent_u,
    const int handle)
{
    Geometry g;
    g.n.shading_normal = inherited_normal;
    g.n.geometry_normal = state->geometry_normal() *
        (math::dot(state->geometry_normal(), inherited_normal) > 0.0f ? 1.0f : -1.0f);

    if (!global_distribution && math::dot(data->k1, g.n.geometry_normal) <= 0.0f) {
        no_emission(data);
        return;
    }

    get_bumped_basis(g.x_axis, g.z_axis, tangent_u, g.n.shading_normal);

    lightprofile_eval_and_pdf<EDF_pdf_data>(
        data, state, g, light_profile_id, multiplier, global_distribution, global_frame);
}

BSDF_API void measured_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    //
    const unsigned light_profile_id,
    const float multiplier,
    const bool global_distribution,
    const float3x3 &global_frame,
    const float3 &tangent_u,
    const int handle)
{
    no_emission(data);
}


/////////////////////////////////////////////////////////////////////
// mixing EDFs
/////////////////////////////////////////////////////////////////////

BSDF_API void normalized_mix_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const EDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void normalized_mix_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const EDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void normalized_mix_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const EDF_component *components,
    const unsigned int num_components)
{
    normalized_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void normalized_mix_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const EDF_component *components,
    const unsigned int num_components)
{
    no_emission(data);
    //normalized_mix_df_auxiliary(
    //  data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_normalized_mix_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_normalized_mix_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_normalized_mix_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    color_normalized_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_normalized_mix_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    no_emission(data);
    //color_normalized_mix_df_auxiliary(
    //  data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void clamped_mix_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const EDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void clamped_mix_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const EDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void clamped_mix_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const EDF_component *components,
    const unsigned int num_components)
{
    clamped_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void clamped_mix_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const EDF_component *components,
    const unsigned int num_components)
{
    no_emission(data);
    //clamped_mix_df_auxiliary(
    //  data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_clamped_mix_edf_sample(
    EDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_sample(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_clamped_mix_edf_evaluate(
    EDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_evaluate(
        data, state, inherited_normal, inherited_weight, components, num_components);
}

BSDF_API void color_clamped_mix_edf_pdf(
    EDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    color_clamped_mix_df_pdf(data, state, inherited_normal, components, num_components);
}

BSDF_API void color_clamped_mix_edf_auxiliary(
    EDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const color_EDF_component *components,
    const unsigned int num_components)
{
    no_emission(data);
    //color_clamped_mix_df_auxiliary(
    //  data, state, inherited_normal, inherited_weight, components, num_components);
}


