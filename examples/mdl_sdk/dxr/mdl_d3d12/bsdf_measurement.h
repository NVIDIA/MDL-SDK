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

// examples/mdl_sdk/dxr/mdl_d3d12/bsdf_measurement.h

#ifndef MDL_D3D12_BSDF_MEASURMENT_H
#define MDL_D3D12_BSDF_MEASURMENT_H

#include "common.h"
#include "example_shared.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
class Base_application;
class Texture;
template <typename T>
class Structured_buffer;

class Bsdf_measurement
{
public:
        struct Part
        {
                Structured_buffer<float>* sample_data = nullptr;
                Structured_buffer<float>* albedo_data = nullptr;
                Texture* evaluation_data = nullptr;
                float max_albedo;
                uint32_t angular_resolution_theta;
                uint32_t angular_resolution_phi;
                uint32_t num_channels;
        };

        Bsdf_measurement(
                Base_application* app,
                const mi::neuraylib::IBsdf_measurement* bsdf_measurement,
                const std::string& debug_name);
        virtual ~Bsdf_measurement();

        bool has_part(mi::neuraylib::Mbsdf_part part) const { return !!m_parts[part].sample_data; }

        const Part& get_part(mi::neuraylib::Mbsdf_part part) const { return m_parts[part]; }

private:
        bool prepare_mbsdf_part(
                Base_application* app,
                mi::neuraylib::Mbsdf_part part,
                const mi::neuraylib::IBsdf_measurement* bsdf_measurement,
                const std::string& debug_name);

        Base_application* m_app;
        std::string m_debug_name;

        Part m_parts[2];
};

}}} // mi::examples::mdl_d3d12
#endif
