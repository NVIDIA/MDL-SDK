/***************************************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

// utility code to bake the sub-clearcoat BRDF of AxF carpaint into a measured BRDF

#ifndef EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_CLEARCOAT_BRDF_UTILS_H
#define EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_CLEARCOAT_BRDF_UTILS_H

#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif
#define M_ONE_OVER_PI       0.318309886183790671538

namespace mi {
namespace examples {
namespace impaxf {

// evaluate isotropic Beckmann distribution on the hemisphere
static float eval_beckmann(
    const float alpha, // roughness
    const float nh)
{     
    const float nh_2 = nh * nh;
    const float tan2 = (1.0f - nh_2) / nh_2; // tan^2(nh)
    const float alpha2 = alpha * alpha;
    
    return exp(-tan2 / alpha2) / (alpha2 * (float)M_PI * nh_2 * nh);
}

static float dot(const float u[3], const float v[3]) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

static void normalize(float k[3]) {
    const float inv_len = 1.0f / sqrt(dot(k, k));
    k[0] *= inv_len;
    k[1] *= inv_len;
    k[2] *= inv_len;
}

// compute cosine of refracted direction
static float refract_cosine(
    const float costheta_i,
    const float eta) // incoming IOR / transmitted IOR
{
    const float sintheta_t = eta * sqrt(1.0f - costheta_i * costheta_i);
    return (sintheta_t >= 1.0f)
        ? -1.0f /* TIR */
        : sqrt(1.0f - sintheta_t * sintheta_t);
}

// Fresnel for unpolarized light
static float fresnel(
    const float costheta_i,
    const float costheta_t, // compute via refract_cosine()
    const float n_i, // incoming IOR
    const float n_t) // transmitted IOR
{
    const float nici = n_i * costheta_i;
    const float ntct = n_t * costheta_t;
    const float rs = (nici - ntct) / (nici + ntct);
    const float nict = n_i * costheta_t;
    const float ntci = n_t * costheta_i;
    const float rp = (nict - ntci) / (nict + ntci);

    return 0.5f * (rs * rs + rp * rp);
}


// refract direction
static int refract_direction(
    float k1_refr[3],
    const float k1[3],
    const float n) // incoming / transmitted IOR
{
    const float nk1 = k1[1];
    assert(nk1 >= 0.0f);
    const float tmp = 1.0f - n * n * (1.0f - nk1 * nk1);
    const int tir = tmp < 0.0f;
    if (tir)
        return 0;

    const float costheta_t = sqrt(tmp);
    const float f = n * nk1 - costheta_t;
    k1_refr[0] = n * k1[0];
    k1_refr[1] = n * k1[1] - f;
    k1_refr[2] = n * k1[2];
    normalize(k1_refr);
    assert(k1_refr[1] >= 0.0f);
    return 1;
}

// parameters of the sub-clearcoat BRDF model of AxF CPA2
struct Brdf_params {
    float diffuse;
    float ct_weight[3];
    float ct_roughness[3];
    float ct_f0[3];
    float ct_f90[3];
    float ior;
    bool refr;
    const float *brdf_colors;
    unsigned brdf_colors_rx;
    unsigned brdf_colors_ry;
};

// Cook-Torrance BRDF model
static float eval_ct(const float k1[3], const float k2[3], const Brdf_params *p) {

    float res = p->diffuse * (float)(1.0 / M_PI);

    float h[3] = {
        k1[0] + k2[0],
        k1[1] + k2[1],
        k1[2] + k2[2]
    };
    normalize(h);
    const float kh = dot(h, k1);
    
    for (unsigned int i = 0; i < 3; ++i) {

        if (p->ct_weight[i] <= 0.0f)
            continue;

        const float ph = eval_beckmann(p->ct_roughness[i], h[1]);

        const float G =
            std::min(std::min(1.0f, 2.0f * h[1] * k1[1] / kh), 2.0f * h[1] * k2[1] / kh);

        const float f = 1.0f - kh;
        const float f2 = f * f;
        const float f5 = f2 * f2 * f;
        const float F = p->ct_f0[i] + (1.0f - p->ct_f0[i]) * f5;

        res += p->ct_weight[i] * ph / (4.0f * k1[1] * k2[1] * h[1]) * G * F;
    }
    return res;
}

// simplistic texture lookup
static void tex_lookup(
    const unsigned int NCOMP,
    float *value,
    const float u, const float v,
    unsigned int rx, unsigned int ry,
    const float *p)
{
    assert(u >= 0.0f && v >= 0.0f);
    float fu = rx * u;
    const float ffu = floorf(fu);
    float fv = ry * v;
    const float ffv = floorf(fv);

    fu -= floorf(fu);
    fv -= floorf(fv);
    
    unsigned int x0 = (unsigned int)ffu;
    unsigned int y0 = (unsigned int)ffv;
    unsigned int x1 = x0 + 1;
    unsigned int y1 = y0 + 1;
    x0 = x0 >= rx ? rx - 1: x0;
    x1 = x1 >= rx ? rx - 1: x1;
    y0 = y0 >= ry ? ry - 1: y0;
    y1 = y1 >= ry ? ry - 1: y1;

    for (unsigned int i = 0; i < NCOMP; ++i) {
        const float v00 = p[NCOMP * (y0 * rx + x0) + i];
        const float v01 = p[NCOMP * (y0 * rx + x1) + i];
        const float v10 = p[NCOMP * (y1 * rx + x0) + i];
        const float v11 = p[NCOMP * (y1 * rx + x1) + i];
        value[i] =
            (1.0f - fu) * ((1.0f - fv) * v00 + fv * v01 ) + 
            (       fu) * ((1.0f - fv) * v10 + fv * v11 );

    }
}

// lookup BRDF coloring curve value
static void apply_brdf_colors(
    float value[3],
    const float k1[3],
    const float k2[3],
    const Brdf_params &params)
{
    float h[3] = {
        k1[0] + k2[0],
        k1[1] + k2[1],
        k1[2] + k2[2]
    };
    normalize(h);
    const float kh = dot(h, k1);
    
    const float u = acos(std::max(std::min(h[1], 1.0f), -1.0f)) * (float)(2.0 / M_PI);
    const float v = acos(std::max(std::min(kh, 1.0f), -1.0f)) * (float)(2.0 / M_PI);
    float rgb[3];
    tex_lookup(3, rgb, u, v, params.brdf_colors_rx, params.brdf_colors_ry, params.brdf_colors);

    for (int i = 0; i < 3; ++i)
        value[i] *= rgb[i];
}

// radical inverse for oversampling
static inline float sample(const unsigned int dim, unsigned int i, const float shift)
{
    static const unsigned int primes[5] = {2, 3, 5, 7, 11};
    const unsigned int base = primes[dim % 5];

    unsigned int base_tmp = 1;
    unsigned int result = i % base;
    i /= base;
    
    while (i) {
        result = result * base + i % base;
        i /= base;
        base_tmp *= base;
    }

    const float res = (float)result / ((float)base_tmp * (float)base) + shift;
    return res > 1.0f ? res - 1.0f : res;
}

// transform the "BRDFcolors" color curve from the domain of refracted to
// the domain of non-refracted directions
// (note: this is approximate as there's no 1:1 relation of the two domains, however,
//  renderings of various AxF carpaints didn't show any difference so far)
inline void recode_brdf_colors(
    std::vector<float> &brdf_colors,
    const unsigned int rx,
    const unsigned int ry,
    const unsigned int num_channels,
    const float ior,
    const bool spectral)
{
    std::vector<float> brdf_colors_nrefr(rx * ry * num_channels, 0.0f);
    std::vector<float> buf(num_channels * 2);

    const float step_y = (float)(M_PI * 0.5) / (float)ry;
    const float step_x = (float)(M_PI * 0.5) / (float)rx;

    constexpr unsigned int num_samples = 4;
    float offsets[2 * num_samples];
    constexpr float inv_num_samples = (float)(1.0 / (double)num_samples);
    for (unsigned int s = 0; s < num_samples; ++s) {
        offsets[2 * s] = ((float)s + 0.5f) * inv_num_samples;
        offsets[2 * s + 1] = sample(0, s, 0.5f * inv_num_samples);
    }

    const float eta = 1.0f / ior;

    for (unsigned int y = 0; y < ry; ++y) {
        for (unsigned int x = 0; x < rx; ++x) {

            const size_t idx = (y * (size_t)rx + x) * num_channels;
            
            for (unsigned int s = 0; s < num_samples; ++s) {
                const float theta_kh = ((float)y + offsets[0]) * step_y;
                const float theta_nh = ((float)x + offsets[1]) * step_x;

                const float cos_theta_kh = cosf(theta_kh);
                const float sin_theta_kh = sinf(theta_kh);

                // wlog: h = [sin_theta_nh, cos_theta_nh, 0]
                const float cos_theta_nh = cosf(theta_nh);
                const float sin_theta_nh = sinf(theta_nh);

                // create a set of differently oriented outgoing directions k1,
                // average results from those valid (for for given half vector elevation)
                constexpr unsigned int num_samples_k1 = 64;
                // half circle is sufficient due to symmetry
                constexpr float step_phi_k1 = (float)(M_PI / (double)num_samples_k1);

                unsigned int valid_samples = 0;
                float *avg_value = buf.data();
                for (unsigned int c = 0; c < num_channels; ++c)
                    avg_value[c] = 0.0f;
                for (unsigned int i = 0; i < num_samples_k1; ++i) {

                    const float phi = (float)i * step_phi_k1;
                    const float cos_phi = cosf(phi);
                    const float sin_phi = sinf(phi);

                    // x_rot, h, z form the basis we create k1 in
                    // x_rot = [cos_theta_nh, -sin_theta_nh, 0]
                    // k1 = x_rot * cos(phi) * sin_theta_kh + h * cos_theta_kh + z * sin(phi) * sin_theta_kh
                    const float k1[3] = {
                        cos_theta_nh * cos_phi * sin_theta_kh + sin_theta_nh * cos_theta_kh,
                        -sin_theta_nh * cos_phi * sin_theta_kh + cos_theta_nh * cos_theta_kh,
                        sin_phi * sin_theta_kh
                    };

                    if (k1[1] < 0.0f) // below surface?
                        continue;

                    // incoming direction by reflecting k1 on h
                    const float k2[3] = {
                        2.0f * cos_theta_kh * sin_theta_nh - k1[0],
                        2.0f * cos_theta_kh * cos_theta_nh - k1[1],
                                                           - k1[2]
                    };
                    if (k2[1] < 0.0f) // below surface?
                        continue;

                    // compute refracted direction and corresponding half vector
                    float k1_refr[3], k2_refr[3];
                    if (!refract_direction(k1_refr, k1, eta))
                        continue;
                    if (!refract_direction(k2_refr, k2, eta))
                        continue;
                    float h_refr[3] = {
                        k1_refr[0] + k2_refr[0],
                        k1_refr[1] + k2_refr[1],
                        k1_refr[2] + k2_refr[2]
                    };
                    normalize(h_refr);

                    const float kh_refr = dot(h_refr, k1_refr);

                    // lookup refracted brdf color values
                    const float v = acos(fmaxf(fminf(kh_refr, 1.0f), 0.0f)) * (float)(2.0 / M_PI);
                    const float u = acos(fmaxf(fminf(h_refr[1], 1.0f), 0.0f)) * (float)(2.0 / M_PI);
                    float *val = &buf[num_channels];
                    tex_lookup(num_channels, val, u, v, rx, ry, brdf_colors.data());

                    ++valid_samples;
                    for (unsigned int c = 0; c < num_channels; ++c)
                        avg_value[c] += val[c];
                }
                
                if (valid_samples > 1) {
                    float scale = 1.0f / (float)valid_samples;
                    if (!spectral) {
                        // - for a non-spectral representation, the BRDFcolors data is only used
                        //   for the fallback model (without measured BSDF)
                        // - in that case we can heuristically darken the curve a bit to better
                        //   match the albedo of the fallback for grazing angles
                        scale *= 0.62f + 0.38f * cos_theta_kh * cos_theta_kh;
                    }
                    for (unsigned int c = 0; c < num_channels; ++c)
                        brdf_colors_nrefr[idx + c] += avg_value[c] * scale;
                }
            }
        }
    }
    for (size_t i = 0; i < brdf_colors_nrefr.size(); ++i)
        brdf_colors_nrefr[i] *= inv_num_samples;
    brdf_colors.swap(brdf_colors_nrefr);
}

// compute the sub-clearcoat BRDF 
static void compute_clearcoat_brdf(
    float val[3],
    const float theta_in,
    const float theta,
    const float phi,
    const Brdf_params &params)    
{
    // incoming direction
    const float k1[3] = {
        sinf(theta_in),
        cosf(theta_in),
        0.0f
    };

    // outgoing direction
    const float sintheta = sinf(theta);
    const float k2[3] = {
        cosf(phi) * sintheta,
        cosf(theta),
        sinf(phi) * sintheta
    };

    val[0] = val[1] = val[2] = 0.0f;
    
    const float eta = 1.0f / params.ior;
    
    const float costheta_t1 = refract_cosine(k1[1], eta);
    if (costheta_t1 < 0.0f)
        return;
    const float F1 = fresnel(k1[1], costheta_t1, 1.0f, params.ior);

    const float costheta_t2 = refract_cosine(k2[1], eta);
    if (costheta_t2 < 0.0f)
        return;
    const float F2 = fresnel(k2[1], costheta_t2, 1.0f, params.ior);

    // note: carpaint representations always use
    // - AXF_TRANSMISSION_VARIANT_NO_SOLID_ANGLE_COMPRESSION: for refracting clearcoat
    // - AXF_TRANSMISSION_VARIANT_DEFAULT: for non-refracting clearcoat
    // so (1 - F1) * (1 - F2) is the base layer should be weighted with (according to the doc)
    // then we divide by the term the MDL Fresnel layerer will apply (1 - max(F1, F2))
    if (params.refr) {
        float k1_refr[3], k2_refr[3];
        if (!refract_direction(k1_refr, k1, eta))
            return;
        if (!refract_direction(k2_refr, k2, eta))
            return;

        const float value = eval_ct(k1_refr, k2_refr, &params) *
            (1.0f - F1) * (1.0f - F2) / (1.0f - std::max(F1, F2));

        val[0] = val[1] = val[2] = value;
        if (params.brdf_colors)
            apply_brdf_colors(val, k1_refr, k2_refr, params);
    } else {
        const float value = eval_ct(k1, k2, &params) *
            (1.0f - F1) * (1.0f - F2) / (1.0f - std::max(F1, F2));

        val[0] = val[1] = val[2] = value;
        if (params.brdf_colors)
            apply_brdf_colors(val, k1, k2, params);
    }
}

// bake sub-clearcoat BRDF model of AxF CPA2 into a measured BRDF data block
inline void create_measured_subclearcoat(
    float *bsdf_data,
    const unsigned int res_theta,
    const unsigned int res_phi,
    const Brdf_params &params)
{
    const float s_theta = (float)(0.5 * M_PI) / (float)res_theta;
    const float s_phi = (float)M_PI / (float)res_phi;

    const unsigned int num_samples = 4;
    float offsets[3 * num_samples];
    const float shift = 0.5f / (float)num_samples;
    for (unsigned int s = 0; s < num_samples; ++s) {
        offsets[3 * s] = sample(0, s, shift);
        offsets[3 * s + 1] = sample(1, s, shift);
        offsets[3 * s + 2] = sample(2, s, shift);
    }        

    const unsigned int num_channels = params.brdf_colors ? 3 : 1;
    
    for (unsigned int tin = 0; tin < res_theta; ++tin)
    {
        const float theta_in0 = (float)tin * s_theta;
        for (unsigned int t = 0; t < res_theta; ++t)
        {
            const float theta0 = (float)t * s_theta;
            for (unsigned int p = 0; p < res_phi; ++p)
            {
                const float phi0 = (float)p * s_phi;

                const unsigned int idx = num_channels * (tin * res_theta * res_phi + t * res_phi + p);
                for (unsigned int i = 0; i < num_channels; ++i)
                    bsdf_data[idx + i] = 0.0f;

                for (unsigned int s = 0; s < num_samples; ++s)
                {
                    float theta_in = theta_in0;
                    float theta = theta0;
                    float phi = phi0;
                    
                    theta_in += offsets[3 * s] * s_theta;
                    theta += offsets[3 * s + 1] * s_theta;
                    phi += offsets[3 * s + 2] * s_phi;

                    float val[3];
                    compute_clearcoat_brdf(val, theta_in, theta, phi, params);

                    for (unsigned int i = 0; i < num_channels; ++i)
                        bsdf_data[idx + i] += val[i];
                }
            }
        }
    }

    const float inv_num_samples = 1.0f / (float)num_samples;
    for (unsigned int i = 0; i < res_theta * res_phi * res_theta * num_channels; ++i)
        bsdf_data[i] *= inv_num_samples;
}

} // namespace impaxf
} // namespace examples
} // namespace mi

#endif //SHADERS_PLUGIN_AXF_IMPORTER_AXF_IMPORTER_CLEARCOAT_BRDF_UTILS_H
