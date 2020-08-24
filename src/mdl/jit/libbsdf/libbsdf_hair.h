/***************************************************************************************************
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
 **************************************************************************************************/

#ifndef MDL_LIBBSDF_HAIR_H
#define MDL_LIBBSDF_HAIR_H

#include "libbsdf.h"
#include "libbsdf_internal.h"

namespace mi
{
namespace libdf
{
namespace hair
{
    BSDF_INLINE float hair_I0(
        const float x)
    {
        float v = 1.0f;
        float n = 1.0f;
        float d = 1.0f; //!! div by 0 if uint, as overflow, but does not matter in practice as float is suitable enough (only last iterations exceed precision but these are huuuge numbers in d then anyway)
        float f = 1.0f;
        const float x2 = x * x;
        for (unsigned int i = 0; i < 9; ++i)
        {
            d *= 4.0f * (f * f);
            n *= x2;
            v += n / d;
            f += 1.0f;
        }
        return v;
    }

    BSDF_INLINE float hair_log_I0(
        const float x)
    {
        return (x > 12.0f) ?
            x + 0.5f * (-1.837877066409345483560659472811235279722794947275566825634f + math::log(1.0f / x) + 1.0f / (8.0f + x)) // = (float)(log(2.0 * M_PI))
            :
            math::log(hair_I0(x));
    }

    BSDF_INLINE float hair_M_p(
        const float cos_theta_o,
        const float cos_theta_i,
        const float sin_theta_i,
        const float sin_theta_o,
        const float v)
    {
        const float inv_v = 1.0f / v;
        const float a = cos_theta_o * cos_theta_i * inv_v;
        const float b = sin_theta_o * sin_theta_i * inv_v;
        return (v < 0.1f) 
            ? math::exp(hair_log_I0(a) - b - inv_v + 0.6931f + math::log(0.5f * inv_v))
            : (math::exp(-b) * hair_I0(a) / (2.0f * v * sinh(inv_v)));
    }

    BSDF_INLINE float hair_Phi(
        const unsigned int p,
        const float gamma_o,
        const float gamma_t)
    {
        const float fp = float(p);
        return 2.0f * fp * gamma_t - 2.0f * gamma_o + fp * (float) M_PI;
    }


    BSDF_INLINE float hair_logistic(float x, const float s)
    {
        if (x > 0.f)
            x = -x;
        const float f = math::exp(x / s);
        return f / (s * (1.0f + f) * (1.0f + f));
    }

    BSDF_INLINE float hair_logistic_cdf(const float x, const float s)
    {
        return 1.0f / (1.0f + math::exp(-x / s));
    }

    BSDF_INLINE float hair_trimmed_logistic(
        const float x, const float s,
        const float a, const float b)
    {
        //!! TODO: opt, always -pi..pi
        return hair_logistic(x, s) / (hair_logistic_cdf(b, s) - hair_logistic_cdf(a, s));
    }


    BSDF_INLINE float hair_sample_trimmed_logistic(
        const float xi,
        const float s,
        const float a,
        const float b)
    {
        //!! TODO: opt, always -pi..pi
        const float k = hair_logistic_cdf(b, s) - hair_logistic_cdf(a, s);
        const float x = -s * math::log(1.0f / (xi * k + hair_logistic_cdf(a, s)) - 1.0f);
        return math::clamp(x, a, b);
    }

    BSDF_INLINE float hair_N_p(
        const float phi,
        const unsigned int p,
        const float s,
        const float gamma_o,
        const float gamma_t)
    {
        if (p >= 3)
            return (float) (0.5 / M_PI);

        float dphi = phi - hair_Phi(p, gamma_o, gamma_t);
        if (!math::isfinite(dphi)) //!! HAIR TODO
            return (float) (0.5 / M_PI);

        while (dphi > (float) M_PI)             dphi -= (float) (2.0 * M_PI);
        while (dphi < (float) (-1.0 * M_PI))    dphi += (float) (2.0 * M_PI);

        return hair_trimmed_logistic(dphi, s, (float) (-1.0f * M_PI), (float) M_PI);
    }


    BSDF_INLINE void hair_tilt_angles(
        float &sin_theta_i_p,
        float &cos_theta_i_p,
        const unsigned int i,
        const float alpha,
        const float sin_theta_i, const float cos_theta_i)
    {
        if (i >= 3 || alpha == 0.0f)
        {
            sin_theta_i_p = sin_theta_i;
            cos_theta_i_p = cos_theta_i;
            return;
        }
        const float m = 2.0f - float(i) * 3.0f;

        float sa = 0.0f;
        float ca = 0.0f;
        math::sincos(m * alpha, &sa, &ca);
        sin_theta_i_p = sin_theta_i * ca + cos_theta_i * sa;
        cos_theta_i_p = cos_theta_i * ca - sin_theta_i * sa;
    }


    BSDF_INLINE float2 hair_prepare_roughness(const float2 &f)
    {
        const float sqrt_pi_8 = 0.626657068657750060403088809835026040673255920410156250f; // sqrt(M_PI / 8.0);
        return make_float2(math::max(f.x, 1e-7f), math::max(f.y * sqrt_pi_8, 1e-7f)); //!! magic
    }
}

BSDF_API void chiang_hair_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    float diffuse_reflection_weight,
    const float3 &diffuse_reflection_tint,
    const float2 &roughness_R,
    const float2 &roughness_TT,
    const float2 &roughness_TRT,
    const float cuticle_angle,
    const float3 &absorption_coefficient,
    const float ior,
    const int handle)
{
    diffuse_reflection_weight = math::saturate(diffuse_reflection_weight);
    data->event_type = BSDF_EVENT_ABSORB;
    data->pdf = 0.0f;

    const float3 tangent_u = state->texture_tangent_u(0);
    const float3 tangent_v = math::cross(inherited_normal, tangent_u);

    // sample diffuse
    float xi_z = data->xi.z;
    if (xi_z < diffuse_reflection_weight)
    {
        // sample diffuse direction
        const float phi = data->xi.x * (float) (2.0 * M_PI);
        float sin_phi, cos_phi;
        math::sincos(phi, &sin_phi, &cos_phi);
        const float sin_theta = math::sqrt(1.0f - data->xi.y);
        const float nk2 = math::sqrt(data->xi.y);

        data->k2 =
            tangent_u * cos_phi * sin_theta +
            inherited_normal * nk2 +
            tangent_v * sin_phi * sin_theta;

        data->pdf = nk2 * (float) (1.0f / M_PI) * diffuse_reflection_weight;

        data->bsdf_over_pdf = math::saturate(diffuse_reflection_tint);

        data->event_type = BSDF_EVENT_DIFFUSE_REFLECTION;

        if (diffuse_reflection_weight >= 1.0f)
            return;
    }
    else
    {
        // continue sample hair bsdf, xi_z will be used again, so rescale
        xi_z = math::saturate((xi_z - diffuse_reflection_weight) / (1.0f - diffuse_reflection_weight));
    }

    const float sin_theta_o = math::dot(data->k1, tangent_u);
    const float cos_theta_o = math::sqrt(math::max(1.0f - sin_theta_o * sin_theta_o, 0.0f));

    const float phi_o = math::atan2(
        math::dot(data->k1, inherited_normal),
        math::dot(data->k1, tangent_v));

    const float3 k1_p = math::normalize(data->k1 - tangent_u * math::dot(data->k1, tangent_u));
    const float cos_gamma_o = math::dot(inherited_normal, k1_p);
    float sin_gamma_o = math::sqrt(math::max(1.0f - cos_gamma_o * cos_gamma_o, 0.0f));
    if (math::dot(k1_p, tangent_v) > 0.0f)
        sin_gamma_o = -sin_gamma_o;
    const float gamma_o = math::asin(sin_gamma_o);


    const float eta = ior / 1.0f /* incoming_ior */;

    const float sin_theta_t = sin_theta_o / eta;
    // all the clamping below avoids issues with total internal reflection (eta < 1)
    // (fresnel will be 1.0 and angles computed here will have no actual effect in computations)
    const float cos_theta_t = math::sqrt(math::max(1.0f - sin_theta_t * sin_theta_t, 0.0f));
    const float eta_p = math::sqrt(math::max(eta * eta - sin_theta_o * sin_theta_o, 0.0f)) / cos_theta_o;
    const float sin_gamma_t = math::max(math::min(sin_gamma_o / eta_p, 1.0f), -1.0f);
    const float cos_gamma_t = math::sqrt(1.0f - sin_gamma_t * sin_gamma_t);
    const float gamma_t = math::asin(sin_gamma_t);


    const float fresnel = ior_fresnel(eta, cos_theta_o * cos_gamma_o);
    float3 T;
    float T_min;
    if (fresnel < 1.0f)
    {
        const float3 sigma_a = absorption_coefficient;

        const float dist = 2.0f * cos_gamma_t / cos_theta_t;
        T = make_float3(
            (sigma_a.x > 0.0f) ? math::exp(-sigma_a.x * dist) : 1.0f,
            (sigma_a.y > 0.0f) ? math::exp(-sigma_a.y * dist) : 1.0f,
            (sigma_a.z > 0.0f) ? math::exp(-sigma_a.z * dist) : 1.0f);
        T_min = math::min(T.x, math::min(T.y, T.z));
    }
    else
    {
        T = make_float3(0.0f, 0.0f, 0.0f);
        T_min = 0.0f;
    }

    const float T_min_f = T_min * fresnel;

    const float sum = fresnel + (1.0f - fresnel) * (1.0f - fresnel) * (T_min + T_min * T_min_f + T_min * T_min_f * T_min_f / (1.0f - T_min_f));

    float sin_theta_i, cos_theta_i, dphi;
    if (data->event_type != BSDF_EVENT_DIFFUSE_REFLECTION) // sample hair bsdf
    {
        float weight_s = fresnel;
        float3 weight = make_float3(fresnel, fresnel, fresnel);
        float prev_cdf = 0.f;
        unsigned int p;
        for (p = 0; ; ++p)
        {
            const float cdf = prev_cdf + weight_s;
            if (xi_z * sum < cdf || p == 3 /* || fresnel >= 1.0*/)
            {
                if (weight_s <= 0.0f) // protect against numerical instabiliy (if sum is close to zero)
                {
                    absorb(data);
                    return;
                }
                
                data->bsdf_over_pdf = weight * sum / weight_s;
                
                xi_z = (xi_z * sum - prev_cdf) / weight_s;
                break;
            }
            
            if (p == 0)
            {
                const float f = (1.0f - fresnel) * (1.0f - fresnel);
                weight_s = T_min * f;
                weight = T * f;
            }
            else if (p == 1)
            {
                weight_s *= T_min_f;
                weight *= T * fresnel;
            }
            else if (p == 2)
            {
                weight_s *= T_min_f / (1.0f - T_min_f);
                weight *= T * fresnel / (make_float3(1.0f, 1.0f, 1.0f) - T * fresnel);
            }
            
            prev_cdf = cdf;
        }

        float2 vs;
        switch(p)
        {
            case 0:
                vs = mi::libdf::hair::hair_prepare_roughness(roughness_R);
                break;
            case 1:
                vs = mi::libdf::hair::hair_prepare_roughness(roughness_TT);
                break;
            default:
                vs = mi::libdf::hair::hair_prepare_roughness(roughness_TRT);
                break;
        }

        // sample N_p
        dphi = mi::libdf::hair::hair_Phi(p, gamma_o, gamma_t);
        if (p >= 3)
            dphi += (data->xi.x * 2.0f - 1.0f) * (float) M_PI;
        else
            dphi += mi::libdf::hair::hair_sample_trimmed_logistic(data->xi.x, vs.y, (float) -M_PI, (float) M_PI);
        const float phi_i = phi_o + dphi;

        // sample M_p
        float xi_y = data->xi.y;
        xi_y = math::max(xi_y, 1e-5f); //!! magic
        const float cos_theta = 1.0f + vs.x * math::log(xi_y + (1.0f - xi_y) * math::exp(-2.0f / vs.x));
        const float sin_theta = math::sqrt(math::max(1.0f - cos_theta * cos_theta, 0.0f));
        const float cos_phi = math::cos(float(2.0 * M_PI) * xi_z);
        sin_theta_i = -cos_theta * sin_theta_o + sin_theta * cos_phi * cos_theta_o;
        cos_theta_i = math::sqrt(1.0f - sin_theta_i * sin_theta_i);
        
        {
            float sti, cti;
            mi::libdf::hair::hair_tilt_angles(sti, cti, p, -cuticle_angle, sin_theta_i, cos_theta_i);
            sin_theta_i = sti;
            cos_theta_i = cti;
        }

        {
            float s_phi_i, c_phi_i;
            math::sincos(phi_i, &s_phi_i, &c_phi_i);
            const float nk2 = s_phi_i * cos_theta_i;
            data->k2 =
                tangent_u * sin_theta_i +
                tangent_v * (c_phi_i * cos_theta_i) +
                inherited_normal * nk2;

            data->event_type = (nk2 > 0.0f) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;

        }
    }
    else
    {
        //!! keep variables alive from diffuse sampling instead of recompute?
        sin_theta_i = math::dot(data->k2, tangent_u);
        cos_theta_i = math::sqrt(math::max(1.0f - sin_theta_i * sin_theta_i, 0.0f));

        const float y1 = math::dot(data->k2, inherited_normal);
        const float x1 = math::dot(data->k2, tangent_v);
        const float y2 = math::dot(data->k1, inherited_normal);
        const float x2 = math::dot(data->k1, tangent_v);
        dphi = math::atan2(y1 * x2 - y2 * x1, x1 * x2 + y1 * y2);
    }

    // compute pdf
    float pdf = 0.0f;
    float weight_s = fresnel;
    float2 vs = mi::libdf::hair::hair_prepare_roughness(roughness_R);
    for (unsigned int p = 0; p <= 3; ++p)
    {

        float sti, cti;
        mi::libdf::hair::hair_tilt_angles(sti, cti, p, cuticle_angle, sin_theta_i, cos_theta_i);
        const float pdf_p = 
            mi::libdf::hair::hair_M_p(cos_theta_o, math::abs(cti), sti, sin_theta_o, vs.x) * 
            mi::libdf::hair::hair_N_p(dphi, p, vs.y, gamma_o, gamma_t);

        pdf += weight_s * pdf_p;

        if (p == 0)
        {
            if (fresnel >= 1.0f)
                break;
            const float f = (1.0f - fresnel) * (1.0f - fresnel);
            weight_s = T_min * f;
            vs = mi::libdf::hair::hair_prepare_roughness(roughness_TT);
        }
        else if (p == 1)
        {
            weight_s *= T_min_f;
            vs = mi::libdf::hair::hair_prepare_roughness(roughness_TRT);
        }
        else if (p == 2)
        {
            weight_s *= T_min_f / (1.0f - T_min_f);
        }
    }
    pdf /= sum;
    data->pdf += pdf * (1.0f - diffuse_reflection_weight);
}

template<typename TBSDF_data>
BSDF_INLINE void chiang_hair_bsdf_evaluate_and_pdf(
    TBSDF_data *data,
    State *state,
    const float3 &inherited_normal,
    const float2 &roughness_R,
    const float2 &roughness_TT,
    const float2 &roughness_TRT,
    const float cuticle_angle,
    const float3 &absorption_coefficient,
    const float ior,
    float& pdf,
    float3& bsdf)
{
    const float3 tangent_u = state->texture_tangent_u(0);
    const float3 tangent_v = math::cross(inherited_normal, tangent_u);

    // compute angles
    const float sin_theta_o = math::dot(data->k1, tangent_u);
    const float sin_theta_i = math::dot(data->k2, tangent_u);
    const float cos_theta_o = math::sqrt(math::max(1.0f - sin_theta_o * sin_theta_o, 0.0f));
    const float cos_theta_i = math::sqrt(math::max(1.0f - sin_theta_i * sin_theta_i, 0.0f));

    const float y1 = math::dot(data->k2, inherited_normal);
    const float x1 = math::dot(data->k2, tangent_v);
    const float y2 = math::dot(data->k1, inherited_normal);
    const float x2 = math::dot(data->k1, tangent_v);
    const float phi = math::atan2(y1 * x2 - y2 * x1, x1 * x2 + y1 * y2);

    const float3 k1_p = math::normalize(data->k1 - tangent_u * math::dot(data->k1, tangent_u));
    const float cos_gamma_o = math::dot(inherited_normal, k1_p);
    float sin_gamma_o = math::sqrt(math::max(1.0f - cos_gamma_o * cos_gamma_o, 0.0f));
    if (math::dot(k1_p, tangent_v) > 0.0f)
        sin_gamma_o = -sin_gamma_o;
    const float gamma_o = math::asin(sin_gamma_o);

    const float eta = ior / 1.0f /* incoming_ior */;

    const float sin_theta_t = sin_theta_o / eta;
    // all the clamping below avoids issues with total internal reflection (eta < 1)
    // (fresnel will be 1.0 and angles computed here will have no actual effect in computations)
    const float cos_theta_t = math::sqrt(math::max(1.0f - sin_theta_t * sin_theta_t, 0.0f));
    const float eta_p = math::sqrt(math::max(eta * eta - sin_theta_o * sin_theta_o, 0.0f)) / cos_theta_o;
    const float sin_gamma_t = math::max(math::min(sin_gamma_o / eta_p, 1.0f), -1.0f);
    const float cos_gamma_t = math::sqrt(1.0f - sin_gamma_t * sin_gamma_t);
    const float gamma_t = math::asin(sin_gamma_t);

    const float fresnel = ior_fresnel(eta, cos_theta_o * cos_gamma_o);
    float3 T = make_float3(0.0f, 0.0f, 0.0f);
    float T_min = 0.0f;
    if (fresnel < 1.0f)
    {
        const float3 sigma_a = absorption_coefficient;
        const float dist = 2.0f * cos_gamma_t / cos_theta_t;
        T = make_float3(
            (sigma_a.x > 0.0f) ? math::exp(-sigma_a.x * dist) : 1.0f,
            (sigma_a.y > 0.0f) ? math::exp(-sigma_a.y * dist) : 1.0f,
            (sigma_a.z > 0.0f) ? math::exp(-sigma_a.z * dist) : 1.0f);
        T_min = math::min(T.x, math::min(T.y, T.z));
    }

    bsdf = make_float3(0.0f, 0.0f, 0.0f);
    pdf = 0.0f;

    float3 weight = make_float3(fresnel, fresnel, fresnel);
    float weight_s = fresnel;
    
    float2 vs = mi::libdf::hair::hair_prepare_roughness(roughness_R);
    float sum = 0.0f;
    for (unsigned int p = 0; p <= 3; ++p)
    {
        float sti = 0.0f;
        float cti = 0.0f;
        mi::libdf::hair::hair_tilt_angles(sti, cti, p, cuticle_angle, sin_theta_i, cos_theta_i);
        
        const float pdf_p =
            mi::libdf::hair::hair_M_p(cos_theta_o, math::abs(cti), sti, sin_theta_o, vs.x) *
            mi::libdf::hair::hair_N_p(phi, p, vs.y, gamma_o, gamma_t);

        bsdf += weight * pdf_p;
        pdf += weight_s * pdf_p;
        sum += weight_s;

        if (p == 0)
        {
            if (fresnel >= 1.0f)
                break;
            const float f = (1.0f - fresnel) * (1.0f - fresnel);
            weight = T * f;
            weight_s = T_min * f;
            vs = mi::libdf::hair::hair_prepare_roughness(roughness_TT);
        }
        else if (p == 1)
        {
            weight *= T * fresnel;
            weight_s *= T_min * fresnel;
            vs = mi::libdf::hair::hair_prepare_roughness(roughness_TRT);
        }
        else if (p == 2)
        {
            weight *= T * fresnel / (make_float3(1.0f, 1.0f, 1.0f) - T * fresnel);
            weight_s *= T_min * fresnel / (1.0f - T_min * fresnel);
        }
    }
    pdf /= sum;
}

BSDF_API void chiang_hair_bsdf_evaluate(
    BSDF_evaluate_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    float diffuse_reflection_weight,
    const float3 &diffuse_reflection_tint,
    const float2 &roughness_R,
    const float2 &roughness_TT,
    const float2 &roughness_TRT,
    const float cuticle_angle,
    const float3 &absorption_coefficient,
    const float ior,
    const int handle)
{
    diffuse_reflection_weight = math::saturate(diffuse_reflection_weight);

    float3 bsdf_glossy;
    chiang_hair_bsdf_evaluate_and_pdf(
        data, state, inherited_normal,
        roughness_R, roughness_TT, roughness_TRT,
        cuticle_angle, absorption_coefficient, ior, data->pdf, bsdf_glossy);

    // add diffuse term (only for light from the front side)
    const float nk2 = math::max(math::dot(data->k2, inherited_normal), 0.0f);
    const float pdf_diffuse = nk2 * (float) (1.0f / M_PI);

    data->pdf =
        pdf_diffuse * diffuse_reflection_weight +
        data->pdf * (1.0f - diffuse_reflection_weight);

    const float3 bsdf_diffuse = math::saturate(diffuse_reflection_tint) * pdf_diffuse;
    add_elemental_bsdf_evaluate_contribution(
        data, handle, 
        bsdf_diffuse * inherited_weight * diffuse_reflection_weight,
        bsdf_glossy * inherited_weight * (1.0f - diffuse_reflection_weight));
}

BSDF_API void chiang_hair_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    float diffuse_reflection_weight,
    const float3 &diffuse_reflection_tint,
    const float2 &roughness_R,
    const float2 &roughness_TT,
    const float2 &roughness_TRT,
    const float cuticle_angle,
    const float3 &absorption_coefficient,
    const float ior,
    const int handle)
{
    diffuse_reflection_weight = math::saturate(diffuse_reflection_weight);

    float3 bsdf;
    chiang_hair_bsdf_evaluate_and_pdf(
        data, state, inherited_normal,
        roughness_R, roughness_TT, roughness_TRT,
        cuticle_angle, absorption_coefficient, ior, data->pdf, bsdf);

    // add diffuse term (only for light from the front side)
    const float nk2 = math::max(math::dot(data->k2, inherited_normal), 0.0f);
    const float pdf_diffuse = nk2 * (float) (1.0f / M_PI);

    data->pdf =
        pdf_diffuse * diffuse_reflection_weight +
        data->pdf * (1.0f - diffuse_reflection_weight);
}

BSDF_API void chiang_hair_bsdf_auxiliary(
    BSDF_auxiliary_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &inherited_weight,
    const float diffuse_reflection_weight,
    const float3 &diffuse_reflection_tint,
    const float2 &roughness_R,
    const float2 &roughness_TT,
    const float2 &roughness_TRT,
    const float cuticle_angle,
    const float3 &absorption_coefficient,
    const float ior,
    const int handle)
{
    add_elemental_bsdf_auxiliary_contribution(
        data,
        handle,
        inherited_weight * diffuse_reflection_weight * diffuse_reflection_tint,
        math::average(inherited_weight) * inherited_normal);
}


/////////////////////////////////////////////////////////////////////
// hair_bsdf tint(
//     color  tint,
//     hair_bsdf   base
// )
/////////////////////////////////////////////////////////////////////

BSDF_API void tint_hair_bsdf_sample(
    BSDF_sample_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.sample(data, state, inherited_normal);
    data->bsdf_over_pdf *= math::saturate(tint);
}

BSDF_API void tint_hair_bsdf_evaluate(
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

BSDF_API void tint_hair_bsdf_pdf(
    BSDF_pdf_data *data,
    State *state,
    const float3 &inherited_normal,
    const float3 &tint,
    const BSDF &base)
{
    base.pdf(data, state, inherited_normal);
}

BSDF_API void tint_hair_bsdf_auxiliary(
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

}
}
#endif // MDL_LIBBSDF_MULTISCATTER_H
