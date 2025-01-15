/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "bsdf_measurement.h"
#include "buffer.h"
#include "texture.h"
#include "command_queue.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Bsdf_measurement::Bsdf_measurement(
        Base_application* app,
        const mi::neuraylib::IBsdf_measurement* bsdf_measurement,
        const std::string& debug_name)
        : m_app(app)
        , m_debug_name(debug_name)
{
        for (mi::neuraylib::Mbsdf_part part :
                { mi::neuraylib::MBSDF_DATA_REFLECTION, mi::neuraylib::MBSDF_DATA_TRANSMISSION })
        {
                if (!prepare_mbsdf_part(app, part, bsdf_measurement, debug_name))
                {
                        delete m_parts[part].sample_data;
                        m_parts[part].sample_data = nullptr;

                        delete m_parts[part].albedo_data;
                        m_parts[part].albedo_data = nullptr;

                        delete m_parts[part].evaluation_data;
                        m_parts[part].evaluation_data = nullptr;
                }
        }
}

Bsdf_measurement::~Bsdf_measurement()
{
        for (mi::neuraylib::Mbsdf_part part :
                { mi::neuraylib::MBSDF_DATA_REFLECTION, mi::neuraylib::MBSDF_DATA_TRANSMISSION })
        {
                if (m_parts[part].sample_data) delete m_parts[part].sample_data;
                if (m_parts[part].albedo_data) delete m_parts[part].albedo_data;
                if (m_parts[part].evaluation_data) delete m_parts[part].evaluation_data;
        }
}

bool Bsdf_measurement::prepare_mbsdf_part(
        Base_application* app,
        mi::neuraylib::Mbsdf_part part,
        const mi::neuraylib::IBsdf_measurement* bsdf_measurement,
        const std::string& debug_name)
{
        mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> bsdf_data;
        if (part == mi::neuraylib::MBSDF_DATA_REFLECTION)
                bsdf_data = bsdf_measurement->get_reflection<mi::neuraylib::IBsdf_isotropic_data>();
        else if (part == mi::neuraylib::MBSDF_DATA_TRANSMISSION)
                bsdf_data = bsdf_measurement->get_transmission<mi::neuraylib::IBsdf_isotropic_data>();

        // no, data fine
        if (!bsdf_data)
                return true;

        // get dimensions
        mi::Uint32 res_x = bsdf_data->get_resolution_theta();
        mi::Uint32 res_y = bsdf_data->get_resolution_phi();
        mi::Uint32 num_channels = (bsdf_data->get_type() == mi::neuraylib::BSDF_SCALAR) ? 1 : 3;

        m_parts[part].angular_resolution_theta = res_x;
        m_parts[part].angular_resolution_phi = res_y;
        m_parts[part].num_channels = num_channels;

        // get data
        mi::base::Handle<const mi::neuraylib::IBsdf_buffer> bsdf_buffer(bsdf_data->get_bsdf_buffer());
        // {1,3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)

        const mi::Float32* data = bsdf_buffer->get_data();

        // ----------------------------------------------------------------------------------------
        // prepare importance sampling data:
        // - for theta_in we will be able to perform a two stage CDF, first to select theta_out,
        //   and second to select phi_out
        // - maximum component is used to "probability" in case of colored measurements

        // CDF of the probability to select a certain theta_out for a given theta_in
        const mi::Size cdf_theta_size = res_x * res_x;

        // for each theta_in x theta_out combination, a CDF of the probabilities to select
        // a certain theta_out is stored
        const mi::Size sample_data_size = cdf_theta_size + cdf_theta_size * res_y;
        std::vector<float> sample_data(sample_data_size);
        
        std::vector<float> albedo_data(res_x); // albedo for sampling reflection and transmission

        float* sample_data_theta = sample_data.data();                // begin of the first (theta) CDF
        float* sample_data_phi = sample_data.data() + cdf_theta_size; // begin of the second (phi) CDF

        const float step_theta = (mdl_d3d12::PI * 0.5f) / float(res_x);
        const float step_phi = mdl_d3d12::PI / float(res_y);

        float max_albedo = 0.0f;
        for (mi::Uint32 t_in = 0; t_in < res_x; ++t_in)
        {
                /*
                the projected area of a sphere patch (in lat-long parameterization):
                        \int_{\phi_0}^{\phi_1} \int_{\theta_0}^{\theta_1} sin(\theta) cos(\theta) d\theta d\phi
                                = 1/2 (\phi_1-\phi_0) (sin^2(\theta_1) - sin^2(\theta_0))

                combining the double angular formula:
                        \cos(2\theta) = cos^2(\theta) - sin^2(\theta)

                and the Pythagorean identity
                        cos^2(\theta)+sin^2(\theta) = 1
                ->      cos^2(\theta) = 1 - sin^2(\theta)

                gives:
                        cos(2\theta) = 1 - 2 sin^2(\theta)
                ->      sin^2(\theta) = -1/2 (cos(2\theta) - 1)

                inserting twice into the patch formula gives:
                          1/2 (\phi_1-\phi_0) (-1/2 (cos(2\theta_1) - 1)) - (-1/2 (cos(2\theta_0) - 1))
                        = 1/2 (\phi_1-\phi_0) -1/2 ((cos(2\theta_1) - 1) - (cos(2\theta_0) - 1))
                        = 1/2 (\phi_1-\phi_0) 1/2 (cos(2\theta_0) - 1 - cos(2\theta_1) + 1)
                        = 1/4 (\phi_1-\phi_0) (cos(2\theta_0) - cos(2\theta_1))
                */

                float sum_theta = 0.0f;
                float cos_2theta0 = 1.0f;
                for (mi::Uint32 t_out = 0; t_out < res_x; ++t_out)
                {
                        // BSDFs are symmetric: f(w_in, w_out) = f(w_out, w_in)
                        // take the average of both measurements

                        // projected area of the surface elements (the ones we are averaging)
                        // we are integrating only half of the sphere and sum up the values of two patches
                        const float cos_2theta1 = std::cos(2 * float(t_out + 1) * step_theta);
                        const float proj_area = (cos_2theta0 - cos_2theta1) * step_phi * 0.25f;
                        cos_2theta0 = cos_2theta1;

                        // offset for both the thetas into the measurement data (select row in the volume)
                        const mi::Uint32 offset_phi = (t_in * res_x + t_out) * res_y;
                        const mi::Uint32 offset_phi2 = (t_out * res_x + t_in) * res_y;

                        // build CDF for phi
                        float sum_phi = 0.0f;
                        for (mi::Uint32 p_out = 0; p_out < res_y; ++p_out)
                        {
                                const mi::Uint32 idx = offset_phi + p_out;
                                const mi::Uint32 idx2 = offset_phi2 + p_out;

                                float value = 0.0f;
                                if (num_channels == 3)
                                {
                                        value = std::max(std::max(data[3 * idx + 0], data[3 * idx + 1]),
                                                                         std::max(data[3 * idx + 2], 0.0f))
                                                  + std::max(std::max(data[3 * idx2 + 0], data[3 * idx2 + 1]),
                                                                 std::max(data[3 * idx2 + 2], 0.0f));
                                }
                                else // num_channels == 1
                                {
                                        value = std::max(data[idx], 0.0f) + std::max(data[idx2], 0.0f);
                                }

                                sum_phi += value * proj_area;
                                sample_data_phi[idx] = sum_phi;
                        }

                        // normalize CDF for phi
                        for (mi::Uint32 p_out = 0; p_out < res_y; ++p_out)
                        {
                                const mi::Uint32 idx = offset_phi + p_out;
                                sample_data_phi[idx] /= sum_phi;
                        }

                        // build CDF for theta
                        sum_theta += sum_phi;
                        sample_data_theta[t_in * res_x + t_out] = sum_theta;
                }

                if (sum_theta > max_albedo)
                        max_albedo = sum_theta;

                albedo_data[t_in] = sum_theta;

                // normalize CDF for theta
                for (unsigned int t_out = 0; t_out < res_x; ++t_out)
                {
                        const mi::Uint32 idx = t_in * res_x + t_out;
                        sample_data_theta[idx] /= sum_theta;
                }
        }
        m_parts[part].max_albedo = max_albedo;

        std::string debug_name_postfix = (part == mi::neuraylib::MBSDF_DATA_REFLECTION) ? "_Refl" : "_Trans";

        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        m_parts[part].sample_data = new Structured_buffer<float>(
                app, sample_data_size, debug_name + "_SampleData" + debug_name_postfix);
        if (!m_parts[part].sample_data->set_data(sample_data.data(), sample_data.size()))
                return false;
        if (!m_parts[part].sample_data->upload(command_list))
                return false;

        m_parts[part].albedo_data = new Structured_buffer<float>(
                app, res_x, debug_name + "_AlbedoData" + debug_name_postfix);
        if (!m_parts[part].albedo_data->set_data(albedo_data.data(), albedo_data.size()))
                return false;
        if (!m_parts[part].albedo_data->upload(command_list))
                return false;

        // ----------------------------------------------------------------------------------------
        // prepare evaluation data:
        // - simply store the measured data in a volume texture
        // - in case of color data, we store each sample in a vector4 to get texture support
        mi::Uint32 lookup_channels = (num_channels == 3) ? 4 : 1;

        // make lookup data symmetric
        std::vector<float> lookup_data(lookup_channels * res_y * res_x * res_x);
        for (mi::Uint32 t_in = 0; t_in < res_x; ++t_in)
        {
                for (mi::Uint32 t_out = 0; t_out < res_x; ++t_out)
                {
                        const mi::Uint32 offset_phi = (t_in * res_x + t_out) * res_y;
                        const mi::Uint32 offset_phi2 = (t_out * res_x + t_in) * res_y;
                        for (mi::Uint32 p_out = 0; p_out < res_y; ++p_out)
                        {
                                const mi::Uint32 idx = offset_phi + p_out;
                                const mi::Uint32 idx2 = offset_phi2 + p_out;

                                if (num_channels == 3)
                                {
                                        lookup_data[4 * idx + 0] = (data[3 * idx + 0] + data[3 * idx2 + 0]) * 0.5f;
                                        lookup_data[4 * idx + 1] = (data[3 * idx + 1] + data[3 * idx2 + 1]) * 0.5f;
                                        lookup_data[4 * idx + 2] = (data[3 * idx + 2] + data[3 * idx2 + 2]) * 0.5f;
                                        lookup_data[4 * idx + 3] = 1.0f;
                                }
                                else
                                {
                                        lookup_data[idx] = (data[idx] + data[idx2]) * 0.5f;
                                }
                        }
                }
        }

        DXGI_FORMAT texture_format = (num_channels == 3)
                ? DXGI_FORMAT_R32G32B32A32_FLOAT // float3 is not always supported
                : DXGI_FORMAT_R32_FLOAT;
        m_parts[part].evaluation_data = Texture::create_texture_3d(
                app, GPU_access::shader_resource, res_y, res_x, res_x, texture_format,
                debug_name + "_EvalData" + debug_name_postfix);

        if (m_parts[part].evaluation_data->upload(command_list, (const uint8_t*)lookup_data.data()))
        {
                // .. since the compute pipeline is used for ray tracing
                m_parts[part].evaluation_data->transition_to(
                        command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else
                return false;

        command_queue->execute_command_list(command_list);
        return true;
}

}}} // mi::examples::mdl_d3d12
