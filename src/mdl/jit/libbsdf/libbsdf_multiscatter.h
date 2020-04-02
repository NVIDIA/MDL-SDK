/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_LIBBSDF_MULTISCATTER_H
#define MDL_LIBBSDF_MULTISCATTER_H

#include "libbsdf.h"
#include "libbsdf_internal.h"

namespace mi
{
namespace libdf
{
namespace multiscatter
{
    BSDF_INLINE unsigned get_lookup_resolution_theta(const BSDF_type /*type*/)
    {
        return 64;
    }

    BSDF_INLINE unsigned get_lookup_resolution_roughness(const BSDF_type /*type*/)
    {
        return 64;
    }

    BSDF_INLINE unsigned get_lookup_resolution_ior(const BSDF_type type)
    {
        switch (type)
        {
            case SHEEN_BSDF:
            case BACKSCATTERING_GLOSSY_BSDF:
            case WARD_GEISLER_MORODER_BSDF:
                return 0;
            default:
                return 16;
        }
    }

    BSDF_INLINE float get_lookup_max_ior(const BSDF_type /*type*/)
    {
        return 3.0f;
    }

    BSDF_INLINE float get_lookup_min_ior(const BSDF_type type)
    {
        return 1.0f + (get_lookup_max_ior(type) - 1.0f) / float(get_lookup_resolution_ior(type));
    }

    BSDF_INLINE float compute_lookup_coordinate_x(
        const BSDF_type type,
        const float cos_theta) // between 0.0 and 1.0
    {
        const float res = float(get_lookup_resolution_theta(type) + 1);

        const float x = math::saturate(math::acos(math::saturate(cos_theta)) * (float)(2.0 / M_PI));
        return (x * (res - 2.0f) + 0.5f) / res;
    }

    BSDF_INLINE float compute_lookup_coordinate_x_nrm(const BSDF_type type)
    {
        return 1.0f; // take very last row entry, without interpolation
    }

    BSDF_INLINE float compute_lookup_coordinate_y(
        const BSDF_type type,
        const float roughness_u, // between min_roughness and 1.0
        const float roughness_v) // between min_roughness and 1.0
    {
        const float res = float(get_lookup_resolution_roughness(type));

        const float r = math::saturate(math::sqrt(roughness_u * roughness_v));
        return (r * (res - 1.0f) + 0.5f) / res;
    }

    BSDF_INLINE float compute_lookup_coordinate_z(
        BSDF_type type, 
        float eta) // -1 or between 1.0/max_ior and ior_min or ior_min and max_ior
    {
        const unsigned res_ior_section = get_lookup_resolution_ior(type);
        if (res_ior_section == 0 || eta < 0.0f)
            return 0.0f; // take very first data layer, without interpolation

        const float res = float(res_ior_section * 2 + 1);
        float offset = 1.5f; // 1.5 pixels (skip the very first layer)
        if (eta < 1.0f)
            eta = 1.0f / eta;                   // first half: eta < 1
        else
            offset += float(res_ior_section);   // second half: eta > 1

        const float a = get_lookup_min_ior(type);
        const float b = get_lookup_max_ior(type);

        float z = math::saturate((eta - a) / (b - a));
        return (z * float(res_ior_section - 1) + offset) / res;
    }

    // called after sampling glossy BSDF
    // returns -1 if the sample was excepted and rho1 if not
    // in case of the latter, the pdf for the glossy BSDF has to be updated (k2 changed)
    // after that, call sample_update_single_scatter_probability.
    BSDF_INLINE float sample(
        const State *state,
        const BSDF_type type,
        const float roughness_u,
        const float roughness_v,
        const float nk1,
        const float eta,
        const unsigned texture_id,
        BSDF_sample_data *data,
        const Geometry &g,
        const float3 &tint,
        const float3 &multiscatter_tint)
    {
        // assuming the glossy BSDF was sampled before
        float w = (data->event_type != BSDF_EVENT_ABSORB) ? data->bsdf_over_pdf.x : 0.0f; // uniform at this point (no tint applied)

        // assuming 0 <= w <= 1, we can reject samples with probability w, rejection basically means 
        // that we use multi-scattering then

        // compute rho1, needed in both cases
        const float2 clamp = make<float2>(0.0f, 1.0f);
        float3 coord = make<float3>(
            compute_lookup_coordinate_x(type, nk1),
            compute_lookup_coordinate_y(type, roughness_u, roughness_v),
            compute_lookup_coordinate_z(type, eta));

        const float rho1 = 
            state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

        // accept glossy sample        
        if (data->xi.w <= w) {

            // incorporate multi-scatter part to pdf
            data->pdf *= rho1;

            // currently, no transmission
            if(data->event_type != BSDF_EVENT_GLOSSY_TRANSMISSION)
                data->pdf += (1.0f - rho1) * (float)(1.0 / M_PI);

            data->bsdf_over_pdf = tint;
            return -1.0f;

        // reject glossy sample
        } else {

            // sample diffuse direction
            const float xi4 = math::saturate((data->xi.w - w) / (1.0f - w));
            const float phi = data->xi.z * (float)(2.0 * M_PI);
            float sin_phi, cos_phi;
            math::sincos(phi, &sin_phi, &cos_phi);

            // could also implement importance sampling based on table
            const float sin_theta = math::sqrt(1.0f - xi4);
            const float nk2 = math::sqrt(xi4);

            data->k2 = math::normalize(g.x_axis * cos_phi * sin_theta + 
                                       g.n.shading_normal * nk2 + 
                                       g.z_axis * sin_phi * sin_theta);

            data->event_type = BSDF_EVENT_DIFFUSE_REFLECTION;

            // compute weight for diffuse event
            coord.x = compute_lookup_coordinate_x(type, nk2);
            const float rho2 =
                state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

            coord.x = compute_lookup_coordinate_x_nrm(type);
            const float nrm =
                state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

            w = (1.0f - rho2) / nrm * (float)M_PI;

            data->bsdf_over_pdf = multiscatter_tint * make<float3>(w);
            return rho1; // for updating the pdf after recomputing the single scatter probability
        }
    }

    // has to be called when the multi-scatter part was sampled before to incorporate the pdf of
    // the new k2 with respect to the glossy BSDF
    BSDF_INLINE void sample_update_single_scatter_probability(
        BSDF_sample_data *data,
        const float pdf,  // updated pdf after sample has returned a non-negative rho1 
        const float rho1)  // the value returned by sample
    {
        // incorporate multi-scatter part to pdf
        data->pdf = rho1 * pdf + (1.0f - rho1) * (float)(1.0 / M_PI);
    }

    BSDF_INLINE float2 evaluate(
        const State *state,
        const BSDF_type type,
        const float roughness_u,
        const float roughness_v,
        const float nk1,
        const float nk2,
        const float eta,
        const unsigned texture_id)
    {
        const float2 clamp = make<float2>(0.0f, 1.0f);
        float3 coord = make<float3>(
            compute_lookup_coordinate_x(type, nk1),
            compute_lookup_coordinate_y(type, roughness_u, roughness_v),
            compute_lookup_coordinate_z(type, eta)
            );

        const float rho1 = 
            state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

        coord.x = compute_lookup_coordinate_x(type, nk2);
        const float rho2 =
            state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

        coord.x = compute_lookup_coordinate_x_nrm(type);
        const float nrm = 
            state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

        return make<float2>(rho1, (1.0f - rho1) * (1.0f - rho2) / nrm * nk2);
    }

    BSDF_INLINE float pdf(
        const float pdf,
        const State *state,
        const BSDF_type type,
        const float roughness_u,
        const float roughness_v,
        const float nk1,
        const float nk2,
        const float eta,
        const unsigned texture_id)
    {
        const float2 clamp = make<float2>(0.0f, 1.0f);
        const float3 coord = make<float3>(
            compute_lookup_coordinate_x(type, nk1),
            compute_lookup_coordinate_y(type, roughness_u, roughness_v),
            compute_lookup_coordinate_z(type, eta));

        const float rho1 =
            state->tex_lookup_float3_3d(texture_id, coord, 0, 0, 0, clamp, clamp, clamp).x;

        return rho1 * pdf + (1.0f - rho1) * (float)(1.0 / M_PI) * nk2;
    }

}
}
}

#endif // MDL_LIBBSDF_MULTISCATTER_H
