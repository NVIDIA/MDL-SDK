/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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
/** \file
 ** \brief
 **/

#include "pch.h"

#include <base/data/db/i_db_access.h>
#include "i_mdlrt_light_profile.h"

#ifndef M_PI
    #define M_PI            3.14159265358979323846
#endif
#define M_ONE_OVER_PI       0.318309886183790671538

namespace MI {
namespace MDLRT {

Light_profile::Light_profile()
{
}

Light_profile::Light_profile(
    Tag_type const  &tex_t,
    DB::Transaction *trans)
: m_light_profile(tex_t, trans)
, m_light_profile_impl(m_light_profile->get_impl_tag(), trans)
{
    m_res_t = m_light_profile->get_resolution_theta();
    m_res_p = m_light_profile->get_resolution_phi();

    m_start_t = m_light_profile->get_theta(0);
    m_start_p = m_light_profile->get_phi(0);
    
    m_delta_t = m_light_profile->get_theta(1) - m_start_t;
    m_delta_p = m_light_profile->get_phi(1) - m_start_p;

    m_inv_delta_t = m_delta_t ? (1.f / m_delta_t) : 0.f;
    m_inv_delta_p = m_delta_p ? (1.f / m_delta_p) : 0.f;

    // phi-mayor: [m_m_res_t x m_res_p]
    const float* m_data = m_light_profile_impl->get_data();
    m_candela_multiplier = m_light_profile->get_candela_multiplier();
    m_total_power = 0.0f;

    // -------------------------------------------------------------------------------------------- 
    // compute total power
    // compute inverse CDF data for sampling
    // sampling will work on cells rather than grid nodes (used for evaluation)

    // first (m_m_res_t-1) for the cdf for sampling theta
    // rest (rex_t-1) * (m_res_p-1) for the individual cdfs for sampling phi (after theta)
    size_t cdf_data_size = (m_res_t - 1) + (m_res_t - 1) * (m_res_p - 1);
    this->m_cdf_data = new float[cdf_data_size];

    float sum_theta = 0.0;
    float cos_theta0 = cosf(m_start_t);
    for (unsigned int t = 0; t < m_res_t - 1; ++t)
    {
        const float cos_theta1 = cosf(m_start_t + float(t + 1) * m_delta_t);

        // area of the patch (grid cell)
        // \mu = int_{theta0}^{theta1} sin{theta} \delta theta
        const float mu = cos_theta0 - cos_theta1;
        cos_theta0 = cos_theta1;

        // build CDF for phi
        float* cdf_data_phi = m_cdf_data + (m_res_t - 1) + t * (m_res_p - 1);
        float sum_phi = 0.0f;
        for (unsigned int p = 0; p < m_res_p - 1; ++p)
        {
            // the probability to select a patch corresponds to the value times area
            // the value of a cell is the average of the corners
            // omit the *1/4 as we normalize in the end
            float value = m_data[p * m_res_t + t]
                        + m_data[p * m_res_t + t + 1]
                        + m_data[(p + 1) * m_res_t + t]
                        + m_data[(p + 1) * m_res_t + t + 1];

            sum_phi += value * mu;
            cdf_data_phi[p] = sum_phi;
        }

        // normalize CDF for phi
        for (unsigned int p = 0; p < m_res_p - 2; ++p)
            cdf_data_phi[p] = sum_phi ? (cdf_data_phi[p] / sum_phi) : 0.0f;

        cdf_data_phi[m_res_p - 2] = 1.0f;

        // build CDF for theta
        sum_theta += sum_phi;
        m_cdf_data[t] = sum_theta;
    }

    m_total_power = m_candela_multiplier * sum_theta * 0.25f * m_delta_p;
    // equals m_light_profile->get_power();

    // normalize CDF for theta
    for (unsigned int t = 0; t < m_res_t - 2; ++t)
        m_cdf_data[t] = sum_theta ? (m_cdf_data[t] / sum_theta) : m_cdf_data[t];

    m_cdf_data[m_res_t - 2] = 1.0f;
}

Light_profile::~Light_profile()
{
    if (m_cdf_data) 
        delete[] m_cdf_data;
}


inline float lerp(float a, float b, float t)
{
    return (1.0f - t) * a + t * b;
}


mi::Float32 Light_profile::evaluate(const mi::Float32_2& theta_phi) const
{
    // map theta to 0..m_res_t range
    float u = (theta_phi.x - m_start_t) * m_inv_delta_t;

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi.y > 0.0f) ? theta_phi.y : (float(2.0 * M_PI) + theta_phi.y);

    // floorf wraps phi range into 0..2pi
    phi = phi - m_start_p - floorf((phi - m_start_p) * float(0.5 / M_PI)) * float(2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border 
    // since it implies lp.theta_phi_start.y > 0 (and we really have "no data" below that)
    float v = phi * m_inv_delta_p; // range 0..m_res_p

    const int idx_theta = int(u);
    const int idx_phi = int(v);

    // outside the measured patch the value is zero
    if (idx_theta < 0 || idx_theta >= m_res_t || 
        idx_phi   < 0 || idx_phi   >= m_res_p) 
            return 0.0f;

    // map to 0..1 range
    u -= float(idx_theta); // <=> fract(...)
    v -= float(idx_phi);

    // lerp neighboring table values
    const unsigned int k = idx_phi * m_res_t + idx_theta;
    const unsigned int idx_theta_p1 = (idx_theta + 1 >= m_res_t) ? 0 : 1;
    const unsigned int idx_phi_p1   = (idx_phi   + 1 >= m_res_p) ? 0 : m_res_t;

    const float* eval_data = m_light_profile_impl->get_data();
    float value = lerp(
        lerp(eval_data[k + 0          + 0], eval_data[k + 0          + idx_theta_p1], u),
        lerp(eval_data[k + idx_phi_p1 + 0], eval_data[k + idx_phi_p1 + idx_theta_p1], u),
        v);

    return value * m_candela_multiplier;
}

// binary search through CDF
inline unsigned sample_cdf(
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

mi::Float32_3 Light_profile::sample(const mi::Float32_3& xi) const
{
    mi::Float32_3 result;
    result.x = -1.0f;
    result.y = -1.0f;
    result.z = 0.0f;

    // sample theta_out
    //-------------------------------------------
    float xi0 = xi.x;
    const float* cdf_data_theta = m_cdf_data;                           // CDF theta
    unsigned idx_theta = sample_cdf(cdf_data_theta, m_res_t - 1, xi0);  // binary search

    float prob_theta = cdf_data_theta[idx_theta];
    if (idx_theta > 0)
    {
        const float tmp = cdf_data_theta[idx_theta - 1];
        prob_theta -= tmp;
        xi0 -= tmp;
    }
    xi0 /= prob_theta; // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi.y;
    const float* cdf_data_phi = cdf_data_theta + (m_res_t - 1)          // CDF theta block
        + (idx_theta * (m_res_p - 1));                                  // selected CDF for phi

    unsigned idx_phi = sample_cdf(cdf_data_phi, m_res_p - 1, xi1);      // binary search
    float prob_phi = cdf_data_phi[idx_phi];
    if (idx_phi > 0)
    {
        const float tmp = cdf_data_phi[idx_phi - 1];
        prob_phi -= tmp;
        xi1 -= tmp;
    }
    xi1 /= prob_phi; // rescale for re-usage

    // compute theta and phi
    //-------------------------------------------
    // sample uniformly within the patch (grid cell)

    const float cos_theta_0 = cos(m_start_t + float(idx_theta)      * m_delta_t);
    const float cos_theta_1 = cos(m_start_t + float(idx_theta + 1u) * m_delta_t);

    //               n = \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    //                 = 1 / (\cos{\theta_0} - \cos{\theta_1})
    //
    //             \xi = n * \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    // => \cos{\theta} = (1 - \xi) \cos{\theta_0} + \xi \cos{\theta_1}

    const float cos_theta = (1.0f - xi.z) * cos_theta_0 + xi.z * cos_theta_1;
    result.x = acos(cos_theta);
    result.y = m_start_p + (float(idx_phi) + xi1) * m_delta_p;

    // align phi 
    if (result.y > float(2.0 * M_PI)) result.y -= float(2.0 * M_PI);                // wrap
    if (result.y > float(1.0 * M_PI)) result.y = float(-2.0 * M_PI) + result.y;     // to [-pi, pi]

    // compute pdf
    //-------------------------------------------
    result.z = prob_theta * prob_phi / (m_delta_p * (cos_theta_0 - cos_theta_1));
    
    return result;
}

mi::Float32 Light_profile::pdf(const mi::Float32_2& theta_phi) const
{
    // map theta to 0..1 range
    float theta = theta_phi.x - m_start_t;
    const int idx_theta = int(theta * m_inv_delta_t);

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi.y > 0.0f) ? theta_phi.y : (float(2.0 * M_PI) + theta_phi.y);

    // floorf wraps phi range into 0..2pi
    phi = phi - m_start_p -
        floorf((phi -m_start_p) * float(0.5 / M_PI)) * float(2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border 
    // since it implies lp.theta_phi_m_start_p > 0 (and we really have "no data" below that)
    const int idx_phi = int(phi * m_inv_delta_p);

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (idx_theta < 0 || idx_theta >(m_res_t - 2) || idx_phi < 0 || idx_phi >(m_res_t - 2))
        return 0.0f;

    // get probability for theta
    //-------------------------------------------
    float prob_theta = m_cdf_data[idx_theta];
    if (idx_theta > 0)
    {
        const float tmp = m_cdf_data[idx_theta - 1];
        prob_theta -= tmp;
    }

    // get probability for phi
    //-------------------------------------------
    const float* cdf_data_phi = m_cdf_data
        + (m_res_t - 1)                             // CDF theta block
        + (idx_theta * (m_res_p - 1));              // selected CDF for phi


    float prob_phi = cdf_data_phi[idx_phi];
    if (idx_phi > 0)
    {
        const float tmp = cdf_data_phi[idx_phi - 1];
        prob_phi -= tmp;
    }

    // compute probability to select a position in the sphere patch 
    const float cos_theta_0 = cos(m_start_t + float(idx_theta)      * m_delta_t);
    const float cos_theta_1 = cos(m_start_t + float(idx_theta + 1u) * m_delta_t);

    return prob_theta * prob_phi / (m_delta_p * (cos_theta_0 - cos_theta_1));
}

}  // MDLRT
}  // MI
