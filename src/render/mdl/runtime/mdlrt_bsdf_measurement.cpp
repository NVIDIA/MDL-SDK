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
#include "i_mdlrt_bsdf_measurement.h"
#include <mi/neuraylib/ibsdf_isotropic_data.h>

#ifndef M_PI
    #define M_PI            3.14159265358979323846
#endif
#define M_ONE_OVER_PI       0.318309886183790671538

namespace MI {
namespace MDLRT {


Bsdf_measurement::Bsdf_measurement()
{
    for (unsigned i = 0; i < 2; ++i)
    {
        m_has_data[i] = 0u;
        m_eval_data[i] = nullptr;
        m_sample_data[i] = nullptr;
        m_albedo_data[i] = nullptr;
        m_max_albedo[i] = 0.0f;
        m_angular_resolution[i] = mi::Uint32_2{0u, 0u};
        m_inv_angular_resolution[i] = mi::Float32_2{0.0f, 0.0f};
        m_num_channels[i] = 0;
    }
}

Bsdf_measurement::Bsdf_measurement(Tag_type const  &bm_t, DB::Transaction *trans)
    : Bsdf_measurement()
{
    m_bsdf_measurement = DB::Access<BSDFM::Bsdf_measurement>(bm_t, trans);
    m_bsdf_measurement_impl
        = DB::Access<BSDFM::Bsdf_measurement_impl>(m_bsdf_measurement->get_impl_tag(), trans);

    // handle reflection
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> dataset(
        m_bsdf_measurement_impl->get_reflection<const mi::neuraylib::IBsdf_isotropic_data>());
    if(dataset)
        prepare_mbsdfs_part(mi::mdl::stdlib::mbsdf_data_reflection, dataset.get());

    // handle transmission
    dataset = mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data>(
        m_bsdf_measurement_impl->get_transmission<const mi::neuraylib::IBsdf_isotropic_data>());
    if (dataset)
        prepare_mbsdfs_part(mi::mdl::stdlib::mbsdf_data_reflection, dataset.get());
}

Bsdf_measurement::~Bsdf_measurement()
{
    for (unsigned i = 0; i < 2; ++i)
    {
        if(m_has_data[i] == 1u)
        {
            delete[] m_eval_data[i];
            delete[] m_sample_data[i];
            delete[] m_albedo_data[i];
        }
    }
}

void Bsdf_measurement::prepare_mbsdfs_part(Mbsdf_part part, 
                                           const mi::neuraylib::IBsdf_isotropic_data* dataset)
{
    unsigned part_idx = static_cast<unsigned>(part);

    // get dimensions
    mi::Uint32_2 res;
    res.x = dataset->get_resolution_theta();
    res.y = dataset->get_resolution_phi();
    unsigned num_channels = dataset->get_type() == mi::neuraylib::BSDF_SCALAR ? 1 : 3;

    m_has_data[part_idx] = 1u;
    m_angular_resolution[part_idx] = res;
    m_inv_angular_resolution[part_idx] = mi::Float32_2{1.0f/float(res.x), 1.0f/float(res.y)};
    m_num_channels[part_idx] = num_channels;

    // get data
    mi::base::Handle<const mi::neuraylib::IBsdf_buffer> buffer(dataset->get_bsdf_buffer());
    // {1,3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)

    const mi::Float32* src_data = buffer->get_data();

    // ----------------------------------------------------------------------------------------
    // prepare importance sampling data:
    // - for theta_in we will be able to perform a two stage CDF, first to select theta_out,
    //   and second to select phi_out
    // - maximum component is used to "probability" in case of colored measurements

    // CDF of the probability to select a certain theta_out for a given theta_in
    const unsigned int cdf_theta_size = res.x * res.x;

    // for each of theta_in x theta_out combination, a CDF of the probabilities to select a
    // a certain theta_out is stored
    const unsigned sample_data_size = cdf_theta_size + cdf_theta_size * res.y;
    float* sample_data = new float[sample_data_size];

    float* albedo_data = new float[res.x]; // albedo for sampling reflection and transmission

    float* sample_data_theta = sample_data;                // begin of the first (theta) CDF
    float* sample_data_phi = sample_data + cdf_theta_size; // begin of the second (phi) CDFs

    const float s_theta = (float) (M_PI * 0.5) / float(res.x);  // step size
    const float s_phi = (float) (M_PI) / float(res.y);          // step size

    float max_albedo = 0.0f;
    for (unsigned int t_in = 0; t_in < res.x; ++t_in)
    {
        float sum_theta = 0.0f;
        float sintheta0_sqd = 0.0f;
        for (unsigned int t_out = 0; t_out < res.x; ++t_out)
        {
            const float sintheta1 = sinf(float(t_out + 1) * s_theta);
            const float sintheta1_sqd = sintheta1 * sintheta1;

            // BSDFs are symmetric: f(w_in, w_out) = f(w_out, w_in)
            // take the average of both measurements

            // area of two the surface elements (the ones we are averaging) 
            const float mu = (sintheta1_sqd - sintheta0_sqd) * s_phi * 0.5f;
            sintheta0_sqd = sintheta1_sqd;

            // offset for both the thetas into the measurement data (select row in the volume) 
            const unsigned int offset_phi  = (t_in  * res.x + t_out) * res.y;
            const unsigned int offset_phi2 = (t_out * res.x + t_in)  * res.y;

            // build CDF for phi
            float sum_phi = 0.0f;
            for (unsigned int p_out = 0; p_out < res.y; ++p_out)
            {
                const unsigned int idx  = offset_phi  + p_out;
                const unsigned int idx2 = offset_phi2 + p_out;

                float value = 0.0f;
                if (num_channels == 3)
                {
                    value = fmax(fmaxf(src_data[3 * idx + 0], src_data[3 * idx + 1]),
                                 fmaxf(src_data[3 * idx + 2], 0.0f))
                          + fmax(fmaxf(src_data[3 * idx2 + 0], src_data[3 * idx2 + 1]),
                                 fmaxf(src_data[3 * idx2 + 2], 0.0f));
                }
                else /* num_channels == 1 */
                {
                    value = fmaxf(src_data[idx], 0.0f) + fmaxf(src_data[idx2], 0.0f);
                }

                sum_phi += value * mu;
                sample_data_phi[idx] = sum_phi;
            }

            // normalize CDF for phi
            for (unsigned int p_out = 0; p_out < res.y; ++p_out)
            {
                const unsigned int idx = offset_phi + p_out;
                sample_data_phi[idx] = sample_data_phi[idx] / sum_phi;
            }

            // build CDF for theta
            sum_theta += sum_phi;
            sample_data_theta[t_in * res.x + t_out] = sum_theta;
        }

        if (sum_theta > max_albedo)
            max_albedo = sum_theta;

        albedo_data[t_in] = sum_theta;

        // normalize CDF for theta 
        for (unsigned int t_out = 0; t_out < res.x; ++t_out)
        {
            const unsigned int idx = t_in * res.x + t_out;
            sample_data_theta[idx] = sample_data_theta[idx] / sum_theta;
        }
    }

    m_sample_data[part_idx] = sample_data;
    m_albedo_data[part_idx] = albedo_data;
    m_max_albedo[part_idx] = max_albedo;


    // ----------------------------------------------------------------------------------------
    // prepare evaluation data:

    // make lookup data symmetric
    float* lookup_data = new float[num_channels * res.y * res.x * res.x];
    for (unsigned t_in = 0; t_in < res.x; ++t_in)
    {
        for (unsigned t_out = 0; t_out < res.x; ++t_out)
        {
            const unsigned offset_phi = (t_in * res.x + t_out) * res.y;
            const unsigned offset_phi2 = (t_out * res.x + t_in) * res.y;
            for (unsigned p_out = 0; p_out < res.y; ++p_out)
            {
                const unsigned idx = offset_phi + p_out;
                const unsigned idx2 = offset_phi2 + p_out;

                for(unsigned c = 0; c < num_channels; ++c)
                {
                    lookup_data[num_channels * idx+c] = 
                        (src_data[num_channels * idx+c] + src_data[num_channels * idx2+c]) * 0.5f;
                }
            }
        }
    }
    m_eval_data[part_idx] = lookup_data;
}

mi::Uint32_3 Bsdf_measurement::get_resolution(Mbsdf_part part) const
{
    unsigned part_idx = static_cast<unsigned>(part);

    mi::Uint32_3 res;
    res.x = m_angular_resolution[part_idx].x;   // steps of theta
    res.y = m_angular_resolution[part_idx].y;   // steps of phi
    res.y = m_num_channels[part_idx];           // number of channels (1 or 3)
    return res;
}

namespace 
{
    inline float lerp(float a, float b, float t)
    {
        return (1.0f - t) * a + t * b;
    }

    inline void bsdf_compute_uvw(const mi::Float32_2& theta_phi_in,
                                 const mi::Float32_2& theta_phi_out,
                                 float& u, float& v, float& w)
    {
        // assuming each phi is between -pi and pi
        u = theta_phi_out.y - theta_phi_in.y;
        if (u < 0.0) u += float(2.0 * M_PI);
        if (u > float(1.0 * M_PI)) u = float(2.0 * M_PI) - u;
        u *= float(M_ONE_OVER_PI);

        v = theta_phi_out.x * float(2.0 / M_PI);
        w = theta_phi_in.x * float(2.0 / M_PI);
    }
}

mi::Float32_3 Bsdf_measurement::evaluate(const mi::Float32_2& theta_phi_in,
                                         const mi::Float32_2& theta_phi_out,
                                         Mbsdf_part part) const
{
    const unsigned part_index = static_cast<unsigned>(part);

    // check for the part
    if (m_has_data[part_index] == 0u)
        return mi::Float32_3{0.0f, 0.0f, 0.0f};

    // assuming each phi is between -pi and pi
    float u, v, w;
    bsdf_compute_uvw(theta_phi_in, theta_phi_out, u, v, w);

    // check boundaries
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f || w < 0.0f || w > 1.0f)
        return mi::Float32_3{0.0f, 0.0f, 0.0f};

   // assuming uvw in [0, 1]^3
    // and the sample data to be valid for the part
    const unsigned part_idx = static_cast<unsigned>(part);

    unsigned res_u = m_angular_resolution[part_idx].y;
    unsigned res_v = m_angular_resolution[part_idx].x;
    unsigned res_w = m_angular_resolution[part_idx].x;
    unsigned num_c = m_num_channels[part_idx];

    u *= float(res_u);
    v *= float(res_v);
    w *= float(res_w);
    unsigned iu = std::min(unsigned(u), res_u - 1u);
    unsigned iv = std::min(unsigned(v), res_v - 1u);
    unsigned iw = std::min(unsigned(w), res_w - 1u);
    u = u - floorf(u);
    v = v - floorf(v);
    w = w - floorf(w);

    // phi_delta x theta_out x theta_in 
    float* volume = m_eval_data[part_idx];

    const unsigned base = iu
                        + iv * res_u
                        + iw * res_u * res_v;

    const unsigned iu_p1 = (iu + 1 >= res_u) ? 0 : 1;
    const unsigned iv_p1 = (iv + 1 >= res_v) ? 0 : res_u;
    const unsigned iw_p1 = (iw + 1 >= res_w) ? 0 : res_u * res_v;

    // lerp neighboring values
    float res[3];
    for (unsigned i = 0; i < 3; ++i)
    {
        res[i] = lerp(
            lerp(
                lerp(
                    volume[num_c * (base + 0       + 0         + 0    ) + i], 
                    volume[num_c * (base + iu_p1   + 0         + 0    ) + i],
                    u),
                lerp(
                    volume[num_c * (base + 0       + iv_p1     + 0    ) + i],
                    volume[num_c * (base + iu_p1   + iv_p1     + 0    ) + i],
                    u),
                v),
            lerp(
                lerp(
                    volume[num_c * (base + 0       + 0         + iw_p1) + i],
                    volume[num_c * (base + iu_p1   + 0         + iw_p1) + i],
                    u),
                lerp(
                    volume[num_c * (base + 0       + iv_p1     + iw_p1) + i],
                    volume[num_c * (base + iu_p1   + iv_p1     + iw_p1) + i],
                    u),
                v),
            w);

        if (num_c == 1u) // same for all channels in case of monochrome data
        {
            res[1] = res[0];
            res[2] = res[0];
            break;
        }
    }
    return mi::Float32_3{res[0], res[1], res[2]};
};


// binary search through CDF
namespace
{
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
}

mi::Float32_3 Bsdf_measurement::sample(const mi::Float32_2& theta_phi_out, 
                                       const mi::Float32_3& xi,
                                       Mbsdf_part part) const
{
    mi::Float32_3 result;
    result.x = -1.0f;
    result.y = -1.0f;
    result.z = 0.0f;

    const unsigned part_index = static_cast<unsigned>(part);
    if (m_has_data[part_index] == 0u)
        return result; // check for the part

    // CDF data
    mi::Uint32_2 res = m_angular_resolution[part_index];
    const float* sample_data = m_sample_data[part_index];

    // compute the theta_in index (flipping input and output, BSDFs are symmetric)
    unsigned idx_theta_in = unsigned(theta_phi_out.x * M_ONE_OVER_PI * 2.0f * float(res.x));
    idx_theta_in = std::min(idx_theta_in, res.x - 1);

    // sample theta_out
    //-------------------------------------------
    float xi0 = xi.x;
    const float* cdf_theta = sample_data + idx_theta_in * res.x;
    unsigned idx_theta_out = sample_cdf(cdf_theta, res.x, xi0);       // binary search

    float prob_theta = cdf_theta[idx_theta_out];
    if (idx_theta_out > 0)
    {
        const float tmp = cdf_theta[idx_theta_out - 1];
        prob_theta -= tmp;
        xi0 -= tmp;
    }
    xi0 /= prob_theta; // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi.y;
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
    xi1 /= prob_phi; // rescale for re-usage

    // compute theta and phi out
    //-------------------------------------------
    const mi::Float32_2 inv_res = m_inv_angular_resolution[part_index];

    const float s_theta = float(0.5 * M_PI) * inv_res.x;
    const float s_phi = float(1.0 * M_PI) * inv_res.y;

    const float cos_theta_0 = cosf(float(idx_theta_out)      * s_theta);
    const float cos_theta_1 = cosf(float(idx_theta_out + 1u) * s_theta);

    const float cos_theta = cos_theta_0 * (1.0f - xi1) + cos_theta_1 * xi1;
    result.x = acosf(cos_theta);
    result.y = (float(idx_phi_out) + xi0) * s_phi;

    if (flip)
        result.y = float(2.0 * M_PI) - result.y; // phi \in [0, 2pi]

    // align phi 
    result.y += (theta_phi_out.y > 0) ? theta_phi_out.y : (float(2.0 * M_PI) + theta_phi_out.y);
    if (result.y > float(2.0 * M_PI)) result.y -= float(2.0 * M_PI);
    if (result.y > float(1.0 * M_PI)) result.y = float(-2.0 * M_PI) + result.y; // to [-pi, pi]

    // compute pdf
    //-------------------------------------------
    result.z = prob_theta * prob_phi * 0.5f
        / (s_phi * (cos_theta_0 - cos_theta_1));

    return result;
}

mi::Float32 Bsdf_measurement::pdf(const mi::Float32_2& theta_phi_in,
                                  const mi::Float32_2& theta_phi_out,
                                  Mbsdf_part part) const
{ 
    const unsigned part_index = static_cast<unsigned>(part);

    // check for the part
    if (m_has_data[part_index] == 0u)
        return 0.0f;

    // CDF data and resolution
    const float* sample_data = m_sample_data[part_index];
    mi::Uint32_2 res = m_angular_resolution[part_index];

    // compute indices in the CDF data
    float u, v, w; // phi_delta, theta_out, theta_in
    bsdf_compute_uvw(theta_phi_in, theta_phi_out, u, v, w); 
    unsigned idx_theta_in  = unsigned(theta_phi_in.x  * M_ONE_OVER_PI * 2.0f * float(res.x));
    unsigned idx_theta_out = unsigned(theta_phi_out.x * M_ONE_OVER_PI * 2.0f * float(res.x));
    unsigned idx_phi_out   = unsigned(u * float(res.y));
    idx_theta_in  = std::min(idx_theta_in, res.x - 1);
    idx_theta_out = std::min(idx_theta_out, res.x - 1);
    idx_phi_out   = std::min(idx_phi_out, res.y - 1);

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
    mi::Float32_2 inv_res = m_inv_angular_resolution[part_index];

    const float s_theta = float(0.5 * M_PI) * inv_res.x;
    const float s_phi = float(1.0 * M_PI) * inv_res.y;

    const float cos_theta_0 = cosf(float(idx_theta_out)      * s_theta);
    const float cos_theta_1 = cosf(float(idx_theta_out + 1u) * s_theta);

    return prob_theta * prob_phi * 0.5f
        / (s_phi * (cos_theta_0 - cos_theta_1));
}

mi::Float32_2 Bsdf_measurement::albedo(const mi::Float32_2& theta_phi, Mbsdf_part part) const
{   
    mi::Float32_2 result{0.0f, 0.0f};
    const unsigned part_index = static_cast<unsigned>(part);

    // check for the part
    if (m_has_data[part_index] == 0u)
        return result;

    const mi::Uint32_2 res = m_angular_resolution[part_index];
    unsigned idx_theta = unsigned(theta_phi.x * float(2.0 / M_PI) * float(res.x));
    idx_theta = std::min(idx_theta, res.x - 1u);
    result.x = m_albedo_data[part_index][idx_theta];
    result.y = m_max_albedo[part_index];
    return result;
}


mi::Float32_4 Bsdf_measurement::albedos(const mi::Float32_2& theta_phi) const
{
    mi::Float32_4 result;

    mi::Float32_2 part_albedo = albedo(theta_phi, mi::mdl::stdlib::mbsdf_data_reflection);
    result.x = part_albedo.x;
    result.y = part_albedo.y;

    part_albedo = albedo(theta_phi, mi::mdl::stdlib::mbsdf_data_transmission);
    result.z = part_albedo.x;
    result.w = part_albedo.y;

    return result;
}

}  // MDLRT
}  // MI
