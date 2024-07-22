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

// examples/mdl_sdk/dxr/mdl_d3d12/light_profile.h

#ifndef MDL_D3D12_LIGHT_PROFILE_H
#define MDL_D3D12_LIGHT_PROFILE_H

#include "common.h"
#include "example_shared.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
class Base_application;
class Texture;
template <typename T>
class Structured_buffer;

class Light_profile
{
public:
        Light_profile(
                Base_application* app,
                const mi::neuraylib::ILightprofile* light_profile,
                const std::string& debug_name);
        virtual ~Light_profile();

        Structured_buffer<float>* get_sample_data() const { return m_sample_data; }
        Texture* get_evaluation_data() const { return m_evaluation_data; }
        mi::Uint32_2 get_angular_resolution() const { return m_angular_resolution; }
        mi::Float32_2 get_theta_phi_start() const { return m_theta_phi_start; }
        mi::Float32_2 get_theta_phi_delta() const { return m_theta_phi_delta; }
        float get_candela_multiplier() const { return m_candela_multiplier; }
        float get_total_power() const { return m_total_power; }

private:
        bool create(
                Base_application* app,
                const mi::neuraylib::ILightprofile* light_profile,
                const std::string& debug_name);

        Base_application* m_app;
        std::string m_debug_name;

        Structured_buffer<float>* m_sample_data; // CDF data for sampling
        Texture* m_evaluation_data;
        mi::Uint32_2 m_angular_resolution; // angular resolutio of the grid
        mi::Float32_2 m_theta_phi_start;   // start angles of the grid
        mi::Float32_2 m_theta_phi_delta;   // angular step size
        float m_candela_multiplier;        // factor to rescale the normalized data 
        float m_total_power;
};

}}} // mi::examples::mdl_d3d12
#endif
