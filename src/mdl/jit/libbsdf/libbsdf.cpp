/***************************************************************************************************
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
 **************************************************************************************************/

#include "libbsdf_internal.h"

BSDF_PARAM bool get_material_thin_walled();
BSDF_PARAM float3 get_material_ior();

#include "libbsdf_utilities.h"

/////////////////////////////////////////////////////////////////////
// bsdf()
/////////////////////////////////////////////////////////////////////

BSDF_API void black_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal)
{
    absorb(data);
}

BSDF_API void black_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal)
{
    absorb(data);
}

BSDF_API void black_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal)
{
    absorb(data);
}


/////////////////////////////////////////////////////////////////////
// bsdf diffuse_reflection_bsdf(
//     color   tint      = color(1.0),
//     float   roughness = 0.0
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void diffuse_reflection_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    float3 tint,
    const float roughness)
{
    tint = math::saturate(tint);
    
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    // sample direction and transform to world coordinates
    const float3 cosh = cosine_hemisphere_sample(make_float2(data->xi.x, data->xi.y));
    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, state::texture_tangent_u(0), shading_normal)) {
        absorb(data);
        return;
    }
    data->k2 = math::normalize(x_axis * cosh.x + shading_normal * cosh.y + z_axis * cosh.z);

    if (cosh.y <= 0.0f || math::dot(data->k2, geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }
    
    if (roughness > 0.0f)
        tint *= eval_oren_nayar(data->k2, data->k1, shading_normal, roughness);

    data->pdf = cosh.y * (float)(1.0 / M_PI);
    data->bsdf_over_pdf = tint;
    data->event_type = BSDF_EVENT_DIFFUSE_REFLECTION;
}

BSDF_API void diffuse_reflection_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    float3 tint,
    const float roughness)
{
    tint = math::saturate(tint);
    
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk2 = math::max(math::dot(data->k2, shading_normal), 0.0f);
    const float pdf = nk2 * (float)(1.0f / M_PI);

    if (nk2 > 0.0f && roughness > 0.0f)
        tint *= eval_oren_nayar(data->k2, data->k1, shading_normal, roughness);
    
    data->bsdf = pdf * tint;
    data->pdf = pdf;
}

BSDF_API void diffuse_reflection_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    float3 tint,
    const float roughness)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);
    
    const float nk2 = math::max(math::dot(data->k2, shading_normal), 0.0f);
    data->pdf = nk2 * (float)(1.0f / M_PI);
}


/////////////////////////////////////////////////////////////////////
// bsdf diffuse_transmission_bsdf(
//     color   tint = color(1.0)
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void diffuse_transmission_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float3 &tint)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    // sample direction and transform to world coordinate
    const float3 cosh = cosine_hemisphere_sample(make_float2(data->xi.x, data->xi.y));
    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, state::texture_tangent_u(0), shading_normal)) {
        absorb(data);
        return;
    }
    data->k2 = math::normalize(x_axis * cosh.x - shading_normal * cosh.y + z_axis * cosh.z);

    if (math::dot(data->k2, geometry_normal) > 0.0f) {
        absorb(data);
        return;
    }
    
    data->pdf = cosh.y * (float)(1.0 / M_PI);
    data->bsdf_over_pdf =  math::saturate(tint);
    data->event_type = BSDF_EVENT_DIFFUSE_TRANSMISSION;
}

BSDF_API void diffuse_transmission_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float3 &tint)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk2 = math::max(-math::dot(data->k2, shading_normal), 0.0f);
    const float pdf = nk2 * (float)(1.0f / M_PI);
    
    data->bsdf = pdf * math::saturate(tint);
    data->pdf = pdf;
}

BSDF_API void diffuse_transmission_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &tint)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk2 = math::max(-math::dot(data->k2, shading_normal), 0.0f);
    data->pdf = nk2 * (float)(1.0f / M_PI);
}


/////////////////////////////////////////////////////////////////////
// bsdf specular_bsdf(
//     color         tint = color(1.0),
//     scatter_mode  mode = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void specular_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float3 &tint,
    const scatter_mode mode)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(shading_normal, data->k1);
    if (nk1 < 0.0f) {
        absorb(data);
        return;
    }

    data->bsdf_over_pdf = math::saturate(tint);
    data->pdf = 0.0f;

    const float2 ior = process_ior(data);

    // reflection
    if ((mode == scatter_reflect) ||
        ((mode == scatter_reflect_transmit) &&
         data->xi.x < ior_fresnel(ior.y / ior.x,  nk1)))
    {
        data->k2 = (nk1 + nk1) * shading_normal - data->k1;

        data->event_type = BSDF_EVENT_SPECULAR_REFLECTION;
    }
    else // refraction
    {
        // total internal reflection should only be triggered for scatter_transmit
        // (since we should fall in the code-path above otherwise)
        bool tir = false; 
        const bool thin_walled = get_material_thin_walled();
        if (thin_walled) // single-sided -> propagate old direction
            data->k2 = -data->k1;
        else
            data->k2 = refract(data->k1, shading_normal, ior.x / ior.y, nk1, tir);

        data->event_type = tir ? BSDF_EVENT_SPECULAR_REFLECTION : BSDF_EVENT_SPECULAR_TRANSMISSION;
    }

    // check if the resulting direction is on the correct side of the actual geometry
    const float gnk2 = math::dot(data->k2, geometry_normal) * (
        data->event_type == BSDF_EVENT_SPECULAR_REFLECTION ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f) {
        absorb(data);
        return;
    }
}

BSDF_API void specular_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float3 &tint,
    const scatter_mode mode)
{
    absorb(data);
}

BSDF_API void specular_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &tint,
    const scatter_mode mode)
{
    absorb(data);
}

//
// Most glossy BSDF models in MDL are microfacet-theory based along the lines of
// "Bruce Walter, Stephen R. Marschner, Hongsong Li, Kenneth E. Torrance - Microfacet Models For
// Refraction through Rough Surfaces" and "Eric Heitz - Understanding the Masking-Shadowing
// Function in Microfacet-Based BRDFs
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
    const float3 &normal,
    const float3 &tangent_u,
    const float3 &tint,
    const scatter_mode mode)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return;
    }

    const float nk1 = math::abs(math::dot(data->k1, shading_normal));

    const float3 k10 = make_float3(
        math::dot(data->k1, x_axis),
        nk1,
        math::dot(data->k1, z_axis));

    // sample half vector / microfacet normal
    const float3 h0 = ph.sample(data->xi, k10);

    // transform to world
    const float3 h = shading_normal * h0.y + x_axis * h0.x + z_axis * h0.z;
    const float kh = math::dot(data->k1, h);

    if (kh <= 0.0f) {
        absorb(data);
        return;
    }

    // compute probability of selection refraction over reflection
    const float2 ior = process_ior(data);
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

    const bool thin_walled = get_material_thin_walled();
    if (data->xi.z < f_refl)
    {
        // BRDF: reflect
        data->k2 = (2.0f * kh) * h - data->k1;
        data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    }
    else
    {
        bool tir = false;
        if (thin_walled) {
            // pseudo-BTDF: flip a reflected reflection direction to the back side
            data->k2 = (2.0f * kh) * h - data->k1;
            data->k2 = math::normalize(
                data->k2 - 2.0f * shading_normal * math::dot(data->k2, shading_normal));
        }
        else
            // BTDF: refract
            data->k2 = refract(data->k1, h, ior.x / ior.y, kh, tir);

        data->event_type = tir ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;
    }
    
    // check if the resulting direction is on the correct side of the actual geometry
    const float gnk2 = math::dot(data->k2, geometry_normal) * (
        data->event_type == BSDF_EVENT_GLOSSY_REFLECTION ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f) {
        absorb(data);
        return;
    }

    const bool refraction = !thin_walled && (data->event_type == BSDF_EVENT_GLOSSY_TRANSMISSION);

    // compute weight
    data->bsdf_over_pdf = math::saturate(tint);

    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float k2h = math::abs(math::dot(data->k2, h));

    const float G1 = ph.mask(k10, kh, h0.y);
    const float G2 = ph.mask(
        make_float3(math::dot(data->k2, x_axis), nk2, math::dot(data->k1, z_axis)), k2h, h0.y);

    const float G12 = ph.shadow_mask(G1, G2, refraction);
    if (G12 <= 0.0f) {
        absorb(data);
        return;
    }
    data->bsdf_over_pdf *= G12 / G1;
    
    // compute pdf
    {
        data->pdf = ph.eval(h0) * G1;

        if (refraction) {
            const float tmp = kh * ior.x + k2h * ior.y;
            data->pdf *= kh * k2h / (nk1 * h0.y * tmp * tmp);
        }
        else
            data->pdf *= 0.25f / (nk1 * h0.y);
    }
}



template <typename Distribution, typename Data>
BSDF_INLINE float3 microfacet_evaluate(
    const Distribution &ph,
    Data *data,
    const float3 &normal,
    const float3 &tangent_u,
    const float3 &tint,
    const scatter_mode mode)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // BTDF or BRDF eval?
    const bool backside_eval = math::dot(data->k2, geometry_normal) < 0.0f;

    // nothing to evaluate for given directions?
    if (( backside_eval && (mode == scatter_reflect )) ||
        (!backside_eval && (mode == scatter_transmit)) ) {
        absorb(data);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float nk1 = math::abs(math::dot(shading_normal, data->k1));
    const float nk2 = math::abs(math::dot(shading_normal, data->k2));
    const bool thin_walled = get_material_thin_walled();

    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, ior, nk1, nk2,
        backside_eval, thin_walled);

    // invalid for reflection / refraction?
    const float nh = math::dot(shading_normal, h);
    const float k1h = math::dot(data->k1, h);
    if (nh < 0.0f || k1h < 0.0f) {
        absorb(data);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // compute BSDF and pdf
    const float fresnel_refl = ior_fresnel(ior.y / ior.x, k1h);
    const float weight = mode == scatter_reflect_transmit ?  
                         (backside_eval ? (1.0f - fresnel_refl) : fresnel_refl) :
                         1.0f;
    
    data->pdf = ph.eval(make_float3(math::dot(x_axis, h), nh, math::dot(z_axis, h)));

    const float G1 = ph.mask(
        make_float3(math::dot(x_axis, data->k1), nk1, math::dot(z_axis, data->k1)), k1h, nh);
    const float k2h = math::abs(math::dot(data->k2, h));
    const float G2 = ph.mask(
        make_float3(math::dot(x_axis, data->k2), nk2, math::dot(z_axis, data->k2)), k2h, nh);

    const bool refraction = !thin_walled && backside_eval;
    if (refraction) {
        // refraction pdf and BTDF
        const float tmp = k1h * ior.x + k2h * ior.y;
        data->pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
    }
    else {
        // reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
        data->pdf *= 0.25f / (nk1 * nh);
    }
    
    const float3 bsdf = math::saturate(tint) *
        (data->pdf * weight * ph.shadow_mask(G1, G2, refraction));
    data->pdf *= G1;    
    return bsdf;
}



/////////////////////////////////////////////////////////////////////
// bsdf simple_glossy_bsdf(
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0),
//     scatter_mode  mode        = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////


// v-cavities masking and and importance sampling utility
// (see "Eric Heitz and Eugene d'Eon - Importance Sampling Microfacet-Based BSDFs with the
// Distribution of Visible Normals")
class Vcavities_masking {
public:
    float3 flip(const float3 &h, const float3 &k, const float xi) const {
        const float a = h.y * k.y;
        const float b = h.x * k.x + h.z * k.z;
        const float kh   = math::max(a + b, 0.0f);
        const float kh_f = math::max(a - b, 0.0f);

        if (xi < kh_f / (kh + kh_f)) {
            return make_float3(-h.x, h.y, -h.z);
        }
        else
            return h;
    }
    
    float mask(const float3 &k, const float kh, const float nh) const {
        return microfacet_mask_v_cavities(nh, kh, k.y);
    }
    float shadow_mask(const float G1, const float G2, const bool refraction) const {
        return refraction ?  math::max(G1 + G2 - 1.0f, 0.0f) : math::min(G1, G2);
    }
};

// simple_glossy_bsdf uses a v-cavities-masked Phong distribution
class Distribution_phong_vcavities : public Vcavities_masking {
public:
    Distribution_phong_vcavities(const float roughness_u, const float roughness_v) {
        m_exponent = roughness_to_exponent(
            clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    float3 sample(const float3 &xi, const float3 &k) const {
        return flip(hvd_phong_sample(make_float2(xi.x, xi.y), m_exponent), k, xi.z);
    }
    
    float eval(const float3 &h) const {
        return hvd_phong_eval(m_exponent, h.y, h.x, h.z);
    }

private:
    float2 m_exponent;
};

BSDF_API void simple_glossy_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_phong_vcavities ph(roughness_u, roughness_v);
    microfacet_sample(ph, data, inherited_normal, tangent_u, tint, mode);
}

BSDF_API void simple_glossy_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_phong_vcavities ph(roughness_u, roughness_v);
    data->bsdf = microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);
}


BSDF_API void simple_glossy_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_phong_vcavities ph(roughness_u, roughness_v);
    microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);
}

/////////////////////////////////////////////////////////////////////
// bsdf backscattering_glossy_reflection_bsdf
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0)
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

BSDF_API void backscattering_glossy_reflection_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return;
    }

    const float2 exponent = roughness_to_exponent(
        clamp_roughness(roughness_u), clamp_roughness(roughness_v));

    // sample half vector
    const float2 u = sample_disk_distribution(make_float2(data->xi.x, data->xi.y), exponent);

    const float xk1 = math::dot(x_axis, data->k1);
    const float zk1 = math::dot(z_axis, data->k1);
    const float2 u1 = u + make_float2(xk1, zk1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float3 h = math::normalize(shading_normal * nk1 + x_axis * u1.x + z_axis * u1.y);

    // compute reflection direction
    const float kh = math::dot(data->k1, h);
    data->k2 = h * (kh + kh) - data->k1;

    // check if the resulting direction is on the correct side of the actual geometry
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    if (nk2 <= 0.0f || math::dot(data->k2, geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }

    // compute weight and pdf
    const float nh = math::dot(shading_normal, h);
    const float inv_nh = 1.0f / nh;
    const float nk1_nh = nk1 * inv_nh;
    const float ph1 = eval_disk_distribution(u.x, u.y, exponent) * nk1_nh * nk1_nh * inv_nh;

    const float nk2_nh = nk2 * inv_nh;
    const float xh = math::dot(x_axis, h);
    const float zh = math::dot(z_axis, h);
    const float x2 = nk2_nh * xh;
    const float y2 = nk2_nh * zh;
    const float xk2 = math::dot(x_axis, data->k2);
    const float zk2 = math::dot(z_axis, data->k2);
    const float ph2 = eval_disk_distribution(
        x2 - xk2, y2 - zk2, exponent) * nk2_nh * nk2_nh * inv_nh;

    data->bsdf_over_pdf = math::saturate(tint) *
        nk2 * math::min(ph1, ph2) / (ph1 * math::max(nk1, nk2));
    data->pdf = ph1 * 0.25f * nk2 / kh;

    data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
}

BSDF_API void backscattering_glossy_reflection_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return;
    }

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::dot(data->k2, shading_normal);
    if (nk2 <= 0.0f) {
        absorb(data);
        return;
    }
    
    const float2 exponent = roughness_to_exponent(
        clamp_roughness(roughness_u), clamp_roughness(roughness_v));
        
    const float3 h = math::normalize(data->k1 + data->k2);

    const float nh = math::dot(shading_normal, h);
    const float xh = math::dot(x_axis, h);
    const float zh = math::dot(z_axis, h);

    const float inv_nh = 1.0f / nh;
    const float nk1_nh = nk1 * inv_nh;
    const float nk2_nh = nk2 * inv_nh;
    const float x1 = nk1_nh * xh;
    const float y1 = nk1_nh * zh;
    const float x2 = nk2_nh * xh;
    const float y2 = nk2_nh * zh;

    const float xk1 = math::dot(data->k1, x_axis);
    const float zk1 = math::dot(data->k1, z_axis);
    const float ph1 = eval_disk_distribution(
        x1 - xk1, y1 - zk1, exponent) * nk1_nh * nk1_nh * inv_nh;

    const float xk2 = math::dot(data->k2, x_axis);
    const float zk2 = math::dot(data->k2, z_axis);
    const float ph2 = eval_disk_distribution(
        x2 - xk2, y2 - zk2, exponent) * nk2_nh * nk2_nh * inv_nh;

    const float kh = math::dot(data->k1, h);
    const float f = (0.25f / kh) * nk2;
    data->pdf = f * ph1;
    data->bsdf = math::saturate(tint) * (f * math::min(ph1, ph2) / math::max(nk2, nk1));
}

BSDF_API void backscattering_glossy_reflection_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return;
    }

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::dot(data->k2, shading_normal);
    if (nk2 <= 0.0f) {
        absorb(data);
        return;
    }

    const float3 h = math::normalize(data->k1 + data->k2);

    const float nh = math::dot(shading_normal, h);
    const float xh = math::dot(x_axis, h);
    const float zh = math::dot(z_axis, h);

    const float inv_nh = 1.0f / nh;
    const float nk1_nh = nk1 * inv_nh;
    const float x1 = nk1_nh * xh;
    const float y1 = nk1_nh * zh;
       
    const float xk1 = math::dot(data->k1, x_axis);
    const float zk1 = math::dot(data->k1, z_axis);
    const float2 exponent = roughness_to_exponent(
        clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    const float ph1 = eval_disk_distribution(
        x1 - xk1, y1 - zk1, exponent) * nk1_nh * nk1_nh * inv_nh;

    const float kh = math::dot(data->k1, h);
    data->pdf = (0.25f / kh) * nk2 * ph1;    
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_beckmann_vcavities_bsdf
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0),
//     scatter_mode  mode        = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////

class Distribution_beckmann_vcavities : public Vcavities_masking {
public:
    Distribution_beckmann_vcavities(const float roughness_u, const float roughness_v) {
        m_inv_roughness = make_float2(
            1.0f / clamp_roughness(roughness_u), 1.0 / clamp_roughness(roughness_v));
    }

    float3 sample(const float3 &xi, const float3 &k) const {
        return flip(hvd_beckmann_sample(make_float2(xi.x, xi.y), m_inv_roughness), k, xi.z);
    }
    
    float eval(const float3 &h) const {
        return hvd_beckmann_eval(m_inv_roughness, h.y, h.x, h.z);
    }

private:
    float2 m_inv_roughness;
};


BSDF_API void microfacet_beckmann_vcavities_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_beckmann_vcavities ph(roughness_u, roughness_v);
    microfacet_sample(ph, data, inherited_normal, tangent_u, tint, mode);
}

BSDF_API void microfacet_beckmann_vcavities_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_beckmann_vcavities ph(roughness_u, roughness_v);
    data->bsdf = microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);
}


BSDF_API void microfacet_beckmann_vcavities_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_beckmann_vcavities ph(roughness_u, roughness_v);
    microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);    
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_ggx_vcavities_bsdf
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0),
//     scatter_mode  mode        = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////

class Distribution_ggx_vcavities : public Vcavities_masking {
public:
    Distribution_ggx_vcavities(const float roughness_u, const float roughness_v) {
        m_inv_roughness = make_float2(
            1.0f / clamp_roughness(roughness_u), 1.0 / clamp_roughness(roughness_v));
    }

    float3 sample(const float3 &xi, const float3 &k) const {
        return flip(hvd_ggx_sample(make_float2(xi.x, xi.y), m_inv_roughness), k, xi.z);
    }
    
    float eval(const float3 &h) const {
        return hvd_ggx_eval(m_inv_roughness, h.y, h.x, h.z);
    }

private:
    float2 m_inv_roughness;
};

BSDF_API void microfacet_ggx_vcavities_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_ggx_vcavities ph(roughness_u, roughness_v);
    microfacet_sample(ph, data, inherited_normal, tangent_u, tint, mode);
}

BSDF_API void microfacet_ggx_vcavities_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_ggx_vcavities ph(roughness_u, roughness_v);
    data->bsdf = microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);
}


BSDF_API void microfacet_ggx_vcavities_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_ggx_vcavities ph(roughness_u, roughness_v);
    microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);    
}


/////////////////////////////////////////////////////////////////////
// bsdf microfacet_beckmann_smith_bsdf
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0),
//     scatter_mode  mode        = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////


class Distribution_beckmann_smith {
public:
    Distribution_beckmann_smith(const float roughness_u, const float roughness_v) {
        m_roughness = make_float2(clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    float3 sample(const float3 &xi, const float3 &k) const {
        return hvd_beckmann_sample_vndf(k, m_roughness, make_float2(xi.x, xi.y));
    }
    
    float mask(const float3 &k, const float, const float) const {
        return microfacet_mask_smith_beckmann(m_roughness.x, m_roughness.y, k);
    }
    float shadow_mask(const float G1, const float G2, const bool refraction) const {
        return G1 * G2;
    }

    float eval(const float3 &h) const {
        return hvd_beckmann_eval(
            make_float2(1.0f / m_roughness.x, 1.0f / m_roughness.y), h.y, h.x, h.z);
    }

private:
    float2 m_roughness;
};

BSDF_API void microfacet_beckmann_smith_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_beckmann_smith ph(roughness_u, roughness_v);
    microfacet_sample(ph, data, inherited_normal, tangent_u, tint, mode);
}

BSDF_API void microfacet_beckmann_smith_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_beckmann_smith ph(roughness_u, roughness_v);
    data->bsdf = microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);
}


BSDF_API void microfacet_beckmann_smith_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_beckmann_smith ph(roughness_u, roughness_v);
    microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);    
}

/////////////////////////////////////////////////////////////////////
// bsdf microfacet_ggx_smith_bsdf
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0),
//     scatter_mode  mode        = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////


class Distribution_ggx_smith {
public:
    Distribution_ggx_smith(const float roughness_u, const float roughness_v) {
        m_roughness = make_float2(clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    float3 sample(const float3 &xi, const float3 &k) const {
        return hvd_ggx_sample_vndf(k, m_roughness, make_float2(xi.x, xi.y));
    }
    
    float mask(const float3 &k, const float, const float) const {
        return microfacet_mask_smith_ggx(m_roughness.x, m_roughness.y, k);
    }
    float shadow_mask(const float G1, const float G2, const bool refraction) const {
        return G1 * G2;
    }

    float eval(const float3 &h) const {
        return hvd_ggx_eval(make_float2(1.0f / m_roughness.x, 1.0f / m_roughness.y), h.y, h.x, h.z);
    }


private:
    float2 m_roughness;
};


BSDF_API void microfacet_ggx_smith_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_ggx_smith ph(roughness_u, roughness_v);
    microfacet_sample(ph, data, inherited_normal, tangent_u, tint, mode);
}

BSDF_API void microfacet_ggx_smith_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_ggx_smith ph(roughness_u, roughness_v);
    data->bsdf = microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);
}


BSDF_API void microfacet_ggx_smith_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u,
    const scatter_mode mode)
{
    const Distribution_ggx_smith ph(roughness_u, roughness_v);
    microfacet_evaluate(ph, data, inherited_normal, tangent_u, tint, mode);    
}


/////////////////////////////////////////////////////////////////////
// bsdf ward_geisler_moroder_bsdf
//     float         roughness_u,
//     float         roughness_v = roughness_u,
//     color         tint        = color(1.0),
//     float3        tangent_u   = state::texture_tangent_u(0)
// )
/////////////////////////////////////////////////////////////////////

// "A New Ward BRDF Model with Bounded Albedo" by David Geisler-Moroder and Arne Duer

BSDF_API void ward_geisler_moroder_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return;
    }

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

    const float nk1 = math::abs(math::dot(data->k1, shading_normal));
    const float xk1 = math::dot(data->k1, x_axis);
    const float zk1 = math::dot(data->k1, z_axis);

    // compute weight for flipped and non-flipped variant
    const float a = nk1 * h0.y;
    const float b = xk1 * h0.x + zk1 * h0.z;
          float kh   = a + b;
    const float kh_f = a - b;
          float w   = math::max(2.0f - nk1 / (h0.y * kh  ), 0.0f);
    const float w_f = math::max(2.0f - nk1 / (h0.y * kh_f), 0.0f);

    // probabilities of not flipping each variant
    float q   = math::min(w  , 1.0f);
    float q_f = math::min(w_f, 1.0f);

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


    const float3 h = x_axis * h0.x + shading_normal * h0.y + z_axis * h0.z;
    data->k2 = h * (kh + kh) - data->k1;

    const float nk2 = math::dot(data->k2, shading_normal);
    if (nk2 <= 0.0f || math::dot(data->k2, geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }

    // compute final weight and pdf
    data->bsdf_over_pdf = math::saturate(tint) * w / prob_total;
    data->pdf = hvd_beckmann_eval(inv_roughness, h0.y, h0.x, h0.z) * 0.25f * nk2 / kh * prob_total;

    data->event_type = BSDF_EVENT_GLOSSY_REFLECTION;
}


template <typename Data>
BSDF_INLINE float3 ward_geisler_moroder_shared_eval(
    Data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, tangent_u, shading_normal)) {
        absorb(data);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::dot(data->k2, shading_normal);
    if (nk2 <= 0.0f) {
        absorb(data);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float2 inv_roughness = make_float2(
        1.0f / clamp_roughness(roughness_u), 1.0f / clamp_roughness(roughness_v));
        
    const float3 h = math::normalize(data->k1 + data->k2);
    const float3 h0 = make_float3(
        math::dot(x_axis, h),
        math::dot(shading_normal, h),
        math::dot(z_axis, h));


    // compute flipping probabilities for pdf
    const float xk1 = math::dot(x_axis, data->k1);
    const float zk1 = math::dot(z_axis, data->k1);
    const float kh   = h0.y * nk1 + h0.x * xk1 + h0.z * zk1;
    const float kh_f = h0.y * nk1 - h0.x * xk1 - h0.z * zk1;
    const float q   = math::saturate((2.0f - nk1) / (h0.y * kh));
    const float q_f = math::saturate((2.0f - nk1) / (h0.y * kh_f));
    

    const float ph = hvd_beckmann_eval(inv_roughness, h0.y, h0.x, h0.z) * 0.25f * nk2 / kh;
    data->pdf = ph * (q + (1.0f - q_f));

    return math::saturate(tint) * ph / (kh * h0.y * h0.y);
}

BSDF_API void ward_geisler_moroder_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    data->bsdf = ward_geisler_moroder_shared_eval(
        data, inherited_normal, roughness_u, roughness_v, tint, tangent_u);
}

BSDF_API void ward_geisler_moroder_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float roughness_u,
    const float roughness_v,
    const float3 &tint,
    const float3 &tangent_u)
{
    ward_geisler_moroder_shared_eval(
        data, inherited_normal, roughness_u, roughness_v, tint, tangent_u);
}

/////////////////////////////////////////////////////////////////////
// bsdf measured_bsdf
//     bsdf_measurement measurement,
//     float            multiplier = 1.0f,
//     scatter_mode     mode       = scatter_reflect
// )
/////////////////////////////////////////////////////////////////////

//!! TODO: measured bsdf is just a diffuse grey dummy for now

BSDF_API void measured_bsdf_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const void *measurement,
    const float multiplier,
    const scatter_mode mode)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    // sample direction and transform to world coordinates
    const float3 cosh = cosine_hemisphere_sample(make_float2(data->xi.x, data->xi.y));
    float3 x_axis, z_axis;
    if (!get_bumped_basis(x_axis, z_axis, state::texture_tangent_u(0), shading_normal)) {
        absorb(data);
        return;
    }
    data->k2 = math::normalize(x_axis * cosh.x + shading_normal * cosh.y + z_axis * cosh.z);

    if (cosh.y <= 0.0f || math::dot(data->k2, geometry_normal) <= 0.0f) {
        absorb(data);
        return;
    }

    data->pdf = cosh.y * (float)(1.0 / M_PI);
    data->bsdf_over_pdf = make_float3(0.5f, 0.5f, 0.5f);
    data->event_type = BSDF_EVENT_DIFFUSE_REFLECTION;
}

BSDF_API void measured_bsdf_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const void *measurement,
    const float multiplier,
    const scatter_mode mode)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk2 = math::max(math::dot(data->k2, shading_normal), 0.0f);
    const float pdf = nk2 * (float)(1.0f / M_PI);
    
    data->bsdf = pdf * make_float3(0.5f, 0.5f, 0.5f);
    data->pdf = pdf;
}

BSDF_API void measured_bsdf_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const void *measurement,
    const float multiplier,
    const scatter_mode mode)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);
    
    const float nk2 = math::max(math::dot(data->k2, shading_normal), 0.0f);
    data->pdf = nk2 * (float)(1.0f / M_PI);
}


/////////////////////////////////////////////////////////////////////
// bsdf tint(
//     color  tint,
//     bsdf   base
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void tint_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.sample(data, inherited_normal);
    data->bsdf_over_pdf *= math::saturate(tint);
}

BSDF_API void tint_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.evaluate(data, inherited_normal);
    data->bsdf *= math::saturate(tint);
}

BSDF_API void tint_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.pdf(data, inherited_normal);
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
    const float3 &inherited_normal,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    base.sample(data, inherited_normal);

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const float coating_ior = math::luminance(ior); //!!TODO: no color support here

    data->bsdf_over_pdf *= thin_film_factor(
        coating_ior, thickness, mat_ior, data->k1, data->k2, shading_normal,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0, get_material_thin_walled());
}

BSDF_API void thin_film_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    base.evaluate(data, inherited_normal);

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const float coating_ior = math::luminance(ior); //!!TODO: no color support here

    data->bsdf *= thin_film_factor(
        coating_ior, thickness, mat_ior, data->k1, data->k2, shading_normal,
        (math::dot(data->k2, geometry_normal) < 0.0f), get_material_thin_walled());
}

BSDF_API void thin_film_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float thickness,
    const float3 &ior,
    const BSDF &base)
{
    base.pdf(data, inherited_normal);
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
    const float3 &inherited_normal,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{
    base.sample(data, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, ior, nk1, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0, get_material_thin_walled());
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
    const float3 &inherited_normal,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{
    base.evaluate(data, inherited_normal);

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, ior, nk1, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f, get_material_thin_walled());
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }   
       
    data->bsdf *= custom_curve_factor(
        kh, exponent, math::saturate(normal_tint), math::saturate(grazing_tint));
}

BSDF_API void directional_factor_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &normal_tint,
    const float3 &grazing_tint,
    const float exponent,
    const BSDF &base)
{
    base.pdf(data, inherited_normal);
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
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    base.sample(data, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float2 material_ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, material_ior, nk1, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0, get_material_thin_walled());
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }

    const float inv_eta_i = 1.0f / material_ior.x;
    const float3 eta = ior * inv_eta_i;
    const float3 eta_k = extinction_coefficient * inv_eta_i;
    data->bsdf_over_pdf *= complex_ior_fresnel(eta, eta_k, kh);
}

BSDF_API void fresnel_factor_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    base.evaluate(data, inherited_normal);

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float2 material_ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, material_ior, nk1, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f, get_material_thin_walled());
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }   

    const float inv_eta_i = 1.0f / material_ior.x;
    const float3 eta = ior * inv_eta_i;
    const float3 eta_k = extinction_coefficient * inv_eta_i;
    data->bsdf *= complex_ior_fresnel(eta, eta_k, kh);
}

BSDF_API void fresnel_factor_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &extinction_coefficient,
    const BSDF &base)
{
    base.pdf(data, inherited_normal);
}

/////////////////////////////////////////////////////////////////////
// bsdf measured_curve_factor(
//     color[<N>] curve_values,
//     bsdf   base = bsdf()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void measured_curve_factor_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{
    base.sample(data, inherited_normal);
    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, ior, nk1, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0, get_material_thin_walled());
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
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{
    base.evaluate(data, inherited_normal);

    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, inherited_normal, state::geometry_normal(), data->k1);

    const float nk1 = math::dot(data->k1, shading_normal);
    const float nk2 = math::abs(math::dot(data->k2, shading_normal));
    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, shading_normal, ior, nk1, nk2,
        math::dot(data->k2, geometry_normal) < 0.0f, get_material_thin_walled());
    const float kh = math::dot(data->k1, h);
    if (kh < 0.0f) {
        absorb(data);
        return;
    }   

    data->bsdf *= math::saturate(measured_curve_factor(kh, curve_values, num_curve_values));
}

BSDF_API void measured_curve_factor_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 *curve_values,
    const unsigned int num_curve_values,
    const BSDF &base)
{
    base.pdf(data, inherited_normal);
}

/////////////////////////////////////////////////////////////////////
// bsdf weighted_layer(
//     float   weight,
//     bsdf    layer,
//     bsdf    base   = bsdf(),
//     float3  normal = state::normal()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void weighted_layer_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    
    if (data->xi.z < weight)
    {
        const float weight_inv = 1.0f / weight;
        data->xi.z *= weight_inv;

        layer.sample(data, normal);
    }
    else
    {
        const float weight_inv = 1.0f / (1.0f - weight);
        data->xi.z = (data->xi.z - weight) * weight_inv;

        base.sample(data, inherited_normal);
    }
}

BSDF_API void weighted_layer_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    
    layer.evaluate(data, normal);
    float3 bsdf0 = data->bsdf;
    float  pdf0  = data->pdf;

    base.evaluate(data, inherited_normal);

    const float w1 = 1.0f - weight;
    data->bsdf = data->bsdf * w1 + bsdf0 * weight;
    data->pdf  = data->pdf  * w1 + pdf0  * weight;
}

BSDF_API void weighted_layer_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    
    layer.pdf(data, normal);
    float pdf0 = data->pdf;

    base.pdf(data, inherited_normal);

    data->pdf = data->pdf * (1.0f - weight) + pdf0 * weight;
}

/////////////////////////////////////////////////////////////////////
// bsdf color_weighted_layer(
//     color   weight,
//     bsdf    layer,
//     bsdf    base   = bsdf(),
//     float3  normal = state::normal()
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void color_weighted_layer_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    float3 weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    const float p = math::average(weight);
    
    if (data->xi.z < p)
    {
        const float p_inv = 1.0f / p;
        data->xi.z *= p_inv;

        layer.sample(data, normal);
        data->bsdf_over_pdf *= weight * p_inv;        
    }
    else
    {
        const float p_inv = 1.0f / (1.0f - p);
        data->xi.z = (data->xi.z - p) * p_inv;

        base.sample(data, inherited_normal);
        data->bsdf_over_pdf *= weight * p_inv;
    }
}

BSDF_API void color_weighted_layer_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    float3 weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    weight = math::saturate(weight);
    const float p = math::average(weight);
    
    layer.evaluate(data, normal);
    const float3 bsdf0 = data->bsdf;
    const float  pdf0  = data->pdf;

    base.evaluate(data, inherited_normal);

    data->bsdf = data->bsdf * (make_float3(1.0f, 1.0f, 1.0f) - weight) + bsdf0 * weight;
    data->pdf  = data->pdf  * (1.0f - p) + pdf0  * p;
}

BSDF_API void color_weighted_layer_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    const float p = math::average(math::saturate(weight));
    
    layer.pdf(data, normal);
    const float pdf0 = data->pdf;

    base.pdf(data, inherited_normal);

    data->pdf = data->pdf * (1.0f - p) + pdf0 * p;
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
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal)
{
    weight = math::saturate(weight);
    
    const float nk1 = math::dot(data->k1, layer_normal);
    const float estimated_curve_factor = c.estimate(nk1);

    const bool no_base = base.is_black();
    
    const float prob_layer = no_base ? 1.0f : estimated_curve_factor * weight;
    const bool sample_layer = no_base || (data->xi.z < prob_layer);
    if (sample_layer) {
        data->xi.z /= prob_layer;
        layer.sample(data, layer_normal);
    } else {
        data->xi.z = (1.0f - data->xi.z) / (1.0f - prob_layer);
        base.sample(data, base_normal);
    }

    if (data->event_type == BSDF_EVENT_ABSORB)
        return;

    const float nk2 = math::abs(math::dot(data->k2, layer_normal));
    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, layer_normal, ior, nk1, nk2,
        (data->event_type & BSDF_EVENT_TRANSMISSION) != 0, get_material_thin_walled());

    if (sample_layer) {
        const float kh = math::abs(math::dot(data->k1, h));
        const float3 curve_factor = c.eval(kh);
        data->bsdf_over_pdf *= curve_factor * weight / prob_layer;
    }
    else{
        const float3 w_base =
            make_float3(1.0f, 1.0f, 1.0f) - weight * math::max(c.eval(nk1), c.eval(nk2));
        data->bsdf_over_pdf *= w_base / (1.0f - prob_layer);
    }
}

template <typename Curve_eval>
BSDF_INLINE void curve_layer_evaluate(
    const Curve_eval &c,
    BSDF_evaluate_data *data,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal,
    const float3 &geometry_normal)
{
    weight = math::saturate(weight);
    
    const float nk1 = math::dot(data->k1, layer_normal);
    const float nk2 = math::abs(math::dot(data->k2, layer_normal));

    const bool backside_eval = math::dot(data->k2, geometry_normal) < 0.0f;
    const float2 ior = process_ior(data);
    const float3 h = compute_half_vector(
        data->k1, data->k2, layer_normal, ior, nk1, nk2,
        backside_eval, get_material_thin_walled());
    
    const float kh = math::abs(math::dot(data->k1, h));
    const float3 curve_factor = c.eval(kh);
    const float3 cf1 = c.eval(nk1);
    const float3 cf2 = c.eval(nk2);

    layer.evaluate(data, layer_normal);
    data->bsdf *= (weight * curve_factor);
    if (base.is_black())
        return;
    
    const float3 bsdf_layer = data->bsdf;
    const float prob_layer = weight * c.estimate(nk1);
    const float pdf_layer = data->pdf * prob_layer;
    
    base.evaluate(data, base_normal);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
    data->bsdf = (1.0f - weight * math::max(cf1, cf2)) * data->bsdf + bsdf_layer;    

}

template <typename Curve_eval>
BSDF_INLINE void curve_layer_pdf(
    const Curve_eval &c,
    BSDF_pdf_data *data,
    float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &layer_normal,
    const float3 &base_normal,
    const float3 &geometry_normal)
{
    weight = math::saturate(weight);
    
    layer.pdf(data, layer_normal);
    if (base.is_black())
        return;

    const float nk1 = math::dot(data->k1, layer_normal);
    const float prob_layer = weight * c.estimate(nk1);
    const float pdf_layer = data->pdf * prob_layer;
    
    base.pdf(data, base_normal);
    data->pdf = (1.0f - prob_layer) * data->pdf + pdf_layer;
}



/////////////////////////////////////////////////////////////////////
// bsdf fresnel_layer(
//     float   ior,
//     float   weight = 1.0,
//     bsdf    layer  = bsdf(),
//     bsdf    base   = bsdf(),
//     float3  normal = state::normal()
// )
/////////////////////////////////////////////////////////////////////

class Fresnel_curve_eval {
public:
    Fresnel_curve_eval(const float eta) :
        m_eta(eta) {
    }

    float estimate(const float cosine) const {
        return ior_fresnel(m_eta, cosine);
    }

    float3 eval(const float cosine) const {
        const float f = ior_fresnel(m_eta, cosine);
        return make_float3(f, f, f);
    }
private:
    float m_eta;
};

BSDF_API void fresnel_layer_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const Fresnel_curve_eval c(ior / mat_ior.x);
    curve_layer_sample(
        c, data, weight, layer, base, shading_normal, inherited_normal);
}

BSDF_API void fresnel_layer_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const Fresnel_curve_eval c(ior / mat_ior.x);
    curve_layer_evaluate(
        c, data, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void fresnel_layer_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float ior,
    const float weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const Fresnel_curve_eval c(ior / mat_ior.x);
    curve_layer_pdf(
        c, data, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

/////////////////////////////////////////////////////////////////////
// bsdf color_fresnel_layer(
//     color   ior,
//     color   weight = 1.0,
//     bsdf    layer  = bsdf(),
//     bsdf    base   = bsdf(),
//     float3  normal = state::normal()
// )
/////////////////////////////////////////////////////////////////////

class Color_fresnel_curve_eval {
public:
    Color_fresnel_curve_eval(const float3 &eta, const float3 &weight) :
        m_eta(eta), m_weight(math::saturate(weight)) {
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
private:
    float3 m_eta;
    float3 m_weight;
};

BSDF_API void color_fresnel_layer_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const Color_fresnel_curve_eval c(ior / mat_ior.x, weight);
    curve_layer_sample(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal);
}

BSDF_API void color_fresnel_layer_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const Color_fresnel_curve_eval c(ior / mat_ior.x, weight);
    curve_layer_evaluate(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void color_fresnel_layer_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const float3 &ior,
    const float3 &weight,
    const BSDF &layer,
    const BSDF &base,
    const float3 &normal)
{
    float3 shading_normal, geometry_normal;
    get_oriented_normals(
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const float2 mat_ior = process_ior(data);
    const Color_fresnel_curve_eval c(ior / mat_ior.x, weight);
    curve_layer_pdf(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}



/////////////////////////////////////////////////////////////////////
// bsdf custom_curve_layer(
//     float   normal_reflectivity,
//     float   grazing_reflectivity = 1.0,
//     float   exponent             = 5.0,
//     float   weight               = 1.0,
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state::normal()
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
private:
    float m_r0, m_r90, m_exponent;
};

BSDF_API void custom_curve_layer_sample(
    BSDF_sample_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_sample(c, data, weight, layer, base, shading_normal, inherited_normal);
}

BSDF_API void custom_curve_layer_evaluate(
    BSDF_evaluate_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_evaluate(
        c, data, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void custom_curve_layer_pdf(
    BSDF_pdf_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    const Custom_curve_eval c(normal_reflectivity, grazing_reflectivity, exponent);
    curve_layer_pdf(
        c, data, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

/////////////////////////////////////////////////////////////////////
// bsdf color_custom_curve_layer(
//     color   normal_reflectivity,
//     color   grazing_reflectivity = color(1.0),
//     float   exponent             = 5.0,
//     color   weight               = color(1.0),
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state::normal()
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
private:
    float3 m_r0, m_r90, m_weight;
    float m_exponent;
};

BSDF_API void color_custom_curve_layer_sample(
    BSDF_sample_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_sample(c, data, 1.0f, layer, base, shading_normal, inherited_normal);
}

BSDF_API void color_custom_curve_layer_evaluate(
    BSDF_evaluate_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_evaluate(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void color_custom_curve_layer_pdf(
    BSDF_pdf_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    const Color_custom_curve_eval c(normal_reflectivity, grazing_reflectivity, weight, exponent);
    curve_layer_pdf(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}


/////////////////////////////////////////////////////////////////////
// bsdf measured_curve_layer(
//     color[<N>] curve_values,
//     float   weight               = 1.0,
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state::normal()
// )
/////////////////////////////////////////////////////////////////////

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
private:    
    const float3 *m_values;
    unsigned int m_num_values;
};

BSDF_API void measured_curve_layer_sample(
    BSDF_sample_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_sample(c, data, weight, layer, base, shading_normal, inherited_normal);
}

BSDF_API void measured_curve_layer_evaluate(
    BSDF_evaluate_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_evaluate(
        c, data, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void measured_curve_layer_pdf(
    BSDF_pdf_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const Measured_curve_eval c(curve_values, num_curve_values);
    curve_layer_pdf(
        c, data, weight, layer, base, shading_normal, inherited_normal, geometry_normal);
}

/////////////////////////////////////////////////////////////////////
// bsdf color_measured_curve_layer(
//     color[<N>] curve_values,
//     color   weight               = color(1.0),
//     bsdf    layer                = bsdf(),
//     bsdf    base                 = bsdf(),
//     float3  normal               = state::normal()
// )
/////////////////////////////////////////////////////////////////////

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
private:    
    const float3 *m_values;
    unsigned int m_num_values;
    float3 m_weight;
    
};

BSDF_API void color_measured_curve_layer_sample(
    BSDF_sample_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);
    
    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_sample(c, data, 1.0f, layer, base, shading_normal, inherited_normal);
}

BSDF_API void color_measured_curve_layer_evaluate(
    BSDF_evaluate_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_evaluate(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

BSDF_API void color_measured_curve_layer_pdf(
    BSDF_pdf_data *data,
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
        shading_normal, geometry_normal, normal, state::geometry_normal(), data->k1);

    const Color_measured_curve_eval c(curve_values, num_curve_values, weight);
    curve_layer_pdf(
        c, data, 1.0f, layer, base, shading_normal, inherited_normal, geometry_normal);
}

/////////////////////////////////////////////////////////////////////
// bsdf normalized_mix(
//     bsdf_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void normalized_mix_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }
    
    float w_sum = components[0].weight;
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += components[i].weight;

    if (w_sum <= 0.0f) {
        absorb(data);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    for (sampled_idx = 0; ; ++sampled_idx) {
        const float cdf = prev_cdf + components[sampled_idx].weight * inv_w_sum;
        if (data->xi.z < cdf || sampled_idx == num_components - 1)
        {
            data->xi.z = (data->xi.z - prev_cdf) / (cdf - prev_cdf);
            break;
        }
        prev_cdf = cdf;
    }
    
    components[sampled_idx].component.sample(data, inherited_normal);
    if (w_sum < 1.0f)
        data->bsdf_over_pdf *= w_sum;
}

BSDF_API void normalized_mix_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float w_sum = components[0].weight;
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += components[i].weight;
    
    if (w_sum <= 0.0f) {
        absorb(data);
        return;
    }
    
    const float inv_w_sum = 1.0f / w_sum;
    const float normalize = w_sum > 1.0f ? inv_w_sum : 1.0f;

    float3 bsdf = make_float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        components[i].component.evaluate(data, inherited_normal);
        const float w = components[i].weight;
        bsdf += data->bsdf * (w * normalize);
        pdf += data->pdf * (w * inv_w_sum);
    }
    
    data->bsdf = bsdf;
    data->pdf = pdf;
}

BSDF_API void normalized_mix_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float w_sum = components[0].weight;
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += components[i].weight;
    
    if (w_sum <= 0.0f) {
        absorb(data);
        return;
    }
    
    const float inv_w_sum = 1.0f / w_sum;
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        components[i].component.pdf(data, inherited_normal);
        pdf += data->pdf * components[i].weight * inv_w_sum;
    }
    data->pdf = pdf;
}


/////////////////////////////////////////////////////////////////////
// bsdf clamped_mix(
//     bsdf_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void clamped_mix_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active) {
        final_weight = components[num_active].weight;
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
        absorb(data);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    for (sampled_idx = 0; ; ++sampled_idx) {
        const float cdf = prev_cdf +
            (sampled_idx == num_active - 1 ? final_weight :
             components[sampled_idx].weight) * inv_w_sum;
        if (data->xi.z < cdf || sampled_idx == num_active - 1) {
            data->xi.z = (data->xi.z - prev_cdf) / (cdf - prev_cdf);
            break;
        }
        prev_cdf = cdf;
    }
    
    components[sampled_idx].component.sample(data, inherited_normal);
    data->bsdf_over_pdf *= w_sum;
;
}

BSDF_API void clamped_mix_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active) {
        final_weight = components[num_active].weight;
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
        absorb(data);
        return;
    }

    const float inv_w_sum = 1.0f / w_sum;

    float3 bsdf = make_float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_active; ++i) {
        components[i].component.evaluate(data, inherited_normal);
        float weight = i == num_active - 1 ? final_weight : components[i].weight;
        bsdf += data->bsdf * weight;
        pdf += data->pdf * weight * inv_w_sum;
    }
    data->bsdf = bsdf;
    data->pdf = pdf;
}

BSDF_API void clamped_mix_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float w_sum = 0.0f;
    unsigned int num_active;
    float final_weight = 0.0f;
    for (num_active = 0; num_active < num_components; ++num_active) {
        final_weight = components[num_active].weight;
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
        absorb(data);
        return;
    }
    
    const float inv_w_sum = 1.0f / w_sum;
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_active; ++i) {
        components[i].component.pdf(data, inherited_normal);       
        float weight = i == num_active - 1 ? final_weight : components[i].weight;
        pdf += data->pdf * weight * inv_w_sum;
    }
    data->pdf = pdf;
}

/////////////////////////////////////////////////////////////////////
// bsdf color_normalized_mix(
//     color_bsdf_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void color_normalized_mix_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float3 w_sum = components[0].weight;
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += components[i].weight;

    if (w_sum.x <= 0.0f && w_sum.y <= 0.0f && w_sum.z <= 0.0f) {
        absorb(data);
        return;
    }

    const float inv_w_sum = 1.0f / math::luminance(w_sum);

    unsigned int sampled_idx;
    float prev_cdf = 0.0f;
    float p;
    for (sampled_idx = 0; ; ++sampled_idx) {
        p = math::luminance(components[sampled_idx].weight) * inv_w_sum;
        const float cdf = prev_cdf + p;
        if (data->xi.z < cdf || sampled_idx == num_components - 1)
        {
            data->xi.z = (data->xi.z - prev_cdf) / (cdf - prev_cdf);
            break;
        }
        prev_cdf = cdf;
    }
    
    components[sampled_idx].component.sample(data, inherited_normal);

    data->bsdf_over_pdf *= components[sampled_idx].weight /
        (math::max(w_sum, make_float3(1.0f, 1.0f, 1.0f)) * p);
}

BSDF_API void color_normalized_mix_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float3 w_sum = components[0].weight;
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += components[i].weight;

    w_sum = math::max(make_float3(0.0f, 0.0f, 0.0f), w_sum);
    
    if (w_sum.x <= 0.0f && w_sum.y <= 0.0f && w_sum.z <= 0.0f) {
        absorb(data);
        return;
    }
    
    const float inv_w_sum = 1.0f / math::luminance(w_sum);
    const float3 normalize = make_float3(
        w_sum.x > 1.0f ? 1.0f / w_sum.x : 1.0f,
        w_sum.y > 1.0f ? 1.0f / w_sum.y : 1.0f,
        w_sum.z > 1.0f ? 1.0f / w_sum.z : 1.0f);

    float3 bsdf = make_float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        components[i].component.evaluate(data, inherited_normal);
        bsdf += data->bsdf * components[i].weight * normalize;
        pdf += data->pdf * (math::luminance(components[i].weight) * inv_w_sum);
    }
    
    data->bsdf = bsdf;
    data->pdf = pdf;
}

BSDF_API void color_normalized_mix_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float w_sum = math::luminance(components[0].weight);
    for (unsigned int i = 1; i < num_components; ++i)
        w_sum += math::luminance(components[i].weight);
    
    if (w_sum <= 0.0f) {
        absorb(data);
        return;
    }
    
    const float inv_w_sum = 1.0f / w_sum;
    float pdf = 0.0f;
    for (unsigned int i = 0; i < num_components; ++i) {
        components[i].component.pdf(data, inherited_normal);
        pdf += data->pdf * math::luminance(components[i].weight) * inv_w_sum;
    }
    data->pdf = pdf;
}

/////////////////////////////////////////////////////////////////////
// bsdf color_clamped_mix(
//     color_bsdf_component[<N>] components
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void color_clamped_mix_sample(
    BSDF_sample_data *data,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int num_active = 0;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components; ++i) {
        ++num_active;
        float3 w = components[i].weight;
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

        if (clamp_mask == 7) // all clamped?
            break;

        lw_sum += math::luminance(w);
    }

    if (lw_sum <= 0.0f) {
        absorb(data);
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
        
        w = components[sampled_idx].weight;
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
        if (data->xi.z < cdf || sampled_idx == num_active - 1)
        {
            data->xi.z = (data->xi.z - prev_cdf) / (cdf - prev_cdf);
            break;
        }
        prev_cdf = cdf;
    }

    components[sampled_idx].component.sample(data, inherited_normal);
    data->bsdf_over_pdf *= w / p;
}

BSDF_API void color_clamped_mix_evaluate(
    BSDF_evaluate_data *data,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float3 bsdf = make_float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;

    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components; ++i) {
        float3 w = components[i].weight;
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

        if (clamp_mask == 7) // all clamped?
            break;

        components[i].component.evaluate(data, inherited_normal);
        bsdf += data->bsdf * w;

        const float lw = math::luminance(w);
        lw_sum += lw;
        pdf += data->pdf * lw;
    }

    data->bsdf = bsdf;
    if (lw_sum > 0.0f)
        data->pdf = pdf / lw_sum;
}

BSDF_API void color_clamped_mix_pdf(
    BSDF_pdf_data *data,
    const float3 &inherited_normal,
    const color_BSDF_component *components,
    const unsigned int num_components)
{
    if (num_components == 0) {
        absorb(data);
        return;
    }

    float pdf = 0.0f;

    float3 mix_sum = make_float3(0.0f, 0.0f, 0.0f);
    float lw_sum = 0.0f;
    unsigned int clamp_mask = 0;
    for (unsigned int i = 0; i < num_components; ++i) {
        float3 w = components[i].weight;
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

        if (clamp_mask == 7) // all clamped?
            break;

        components[i].component.pdf(data, inherited_normal);
        const float lw = math::luminance(w);
        pdf += data->pdf * lw;
    }

    if (lw_sum > 0.0f)
        data->pdf = pdf / lw_sum;
}



