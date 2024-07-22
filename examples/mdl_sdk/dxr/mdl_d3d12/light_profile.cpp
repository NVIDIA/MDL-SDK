/******************************************************************************
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
 *****************************************************************************/

#include "light_profile.h"
#include "buffer.h"
#include "texture.h"
#include "command_queue.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Light_profile::Light_profile(
        Base_application* app,
        const mi::neuraylib::ILightprofile* light_profile,
        const std::string& debug_name)
        : m_app(app)
        , m_debug_name(debug_name)
        , m_sample_data(nullptr)
        , m_evaluation_data(nullptr)
{
        if (!create(app, light_profile, debug_name))
        {
                delete m_sample_data;
                m_sample_data = nullptr;

                delete m_evaluation_data;
                m_evaluation_data = nullptr;
        }
}

Light_profile::~Light_profile()
{
        if (m_sample_data) delete m_sample_data;
        if (m_evaluation_data) delete m_evaluation_data;
}

bool Light_profile::create(
        Base_application* app,
        const mi::neuraylib::ILightprofile* light_profile,
        const std::string& debug_name)
{
        if (!light_profile)
                return true;

        mi::Uint32 res_x = light_profile->get_resolution_theta();
        mi::Uint32 res_y = light_profile->get_resolution_phi();
        float start_x = light_profile->get_theta(0);
        float start_y = light_profile->get_phi(0);
        float delta_x = light_profile->get_theta(1) - start_x;
        float delta_y = light_profile->get_phi(1) - start_y;

        m_angular_resolution = mi::Uint32_2(res_x, res_y);
        m_theta_phi_start = mi::Float32_2(start_x, start_y);
        m_theta_phi_delta = mi::Float32_2(delta_x, delta_y);
        m_candela_multiplier = float(light_profile->get_candela_multiplier());

        // phi-mayor: [res.x x res.y]
        const float* data = light_profile->get_data();

        // --------------------------------------------------------------------------------------------
        // compute total power
        // compute inverse CDF data for sampling
        // sampling will work on cells rather than grid nodes (used for evaluation)

        // first (res.x-1) for the cdf for sampling theta
        // rest (rex.x-1) * (res.y-1) for the individual cdfs for sampling phi (after theta)
        size_t cdf_data_size = (res_x - 1) + (res_x - 1) * (res_y - 1);
        std::vector<float> cdf_data(cdf_data_size);

        float sum_theta = 0.0f;
        float cos_theta0 = std::cos(start_x);
        for (mi::Uint32 t = 0; t < res_x - 1; ++t)
        {
                const float cos_theta1 = cosf(start_x + float(t + 1) * delta_x);

                // area of the patch (grid cell)
                // \mu = int_{theta0}^{theta1} sin{theta} \delta theta
                const float mu = cos_theta0 - cos_theta1;
                cos_theta0 = cos_theta1;

                // build CDF for phi
                float* cdf_data_phi = cdf_data.data() + (res_x - 1) + t * (res_y - 1);
                float sum_phi = 0.0f;
                for (mi::Uint32 p = 0; p < res_y - 1; ++p)
                {
                        // the probability to select a patch corresponds to the value times area
                        // the value of a cell is the average of the corners
                        // omit the *1/4 as we normalize in the end
                        const float value = data[p * res_x + t]
                                              + data[p * res_x + t + 1]
                                              + data[(p + 1) * res_x + t]
                                              + data[(p + 1) * res_x + t + 1];

                        sum_phi += value * mu;
                        cdf_data_phi[p] = sum_phi;
                }

                // normalize CDF for phi
                for (mi::Uint32 p = 0; p < res_y - 2; ++p)
                        cdf_data_phi[p] = sum_phi ? (cdf_data_phi[p] / sum_phi) : 0.0f;

                cdf_data_phi[res_y - 2] = 1.0f;

                // build CDF for theta
                sum_theta += sum_phi;
                cdf_data[t] = sum_theta;
        }
        m_total_power = sum_theta * 0.25f * delta_y * m_candela_multiplier;

        // normalize CDF for theta
        for (unsigned int t = 0; t < res_x - 2; ++t)
                cdf_data[t] = sum_theta ? (cdf_data[t] / sum_theta) : cdf_data[t];

        cdf_data[res_x - 2] = 1.0f;

        // copy entire CDF data buffer to GPU
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();
        
        m_sample_data = new Structured_buffer<float>(
                app, cdf_data_size, m_debug_name + "_SampleData");
        if (!m_sample_data->set_data(cdf_data.data(), cdf_data.size()))
                return false;
        if (!m_sample_data->upload(command_list))
                return false;

        // --------------------------------------------------------------------------------------------
        // prepare evaluation data
        //  - use a 2d texture that allows bilinear interpolation
        m_evaluation_data = Texture::create_texture_2d(
                app, GPU_access::shader_resource, res_x, res_y, DXGI_FORMAT_R32_FLOAT, m_debug_name + "_EvalData");

        if (m_evaluation_data->upload(command_list, (const uint8_t*)data))
        {
                // .. since the compute pipeline is used for ray tracing
                m_evaluation_data->transition_to(
                        command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else
                return false;

        command_queue->execute_command_list(command_list);
        return true;
}

}}} // mi::examples::mdl_d3d12
