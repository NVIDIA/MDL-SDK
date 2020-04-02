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

#ifndef RENDER_MDL_RUNTIME_I_MDLRT_LIGHT_PROFILE_H
#define RENDER_MDL_RUNTIME_I_MDLRT_LIGHT_PROFILE_H

#include <mi/neuraylib/typedefs.h>

#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_access.h>
#include <io/scene/lightprofile/i_lightprofile.h>

namespace MI {

namespace DB { class Transaction; }

namespace MDLRT {

class Light_profile
{
public:
    typedef DB::Typed_tag<LIGHTPROFILE::Lightprofile> Tag_type;

    Light_profile();

    Light_profile(Tag_type const &tag, DB::Transaction *trans);
    virtual ~Light_profile();

    float get_power() const { return m_light_profile->get_power(); }

    float get_maximum() const { return m_light_profile->get_maximum(); }

    bool is_valid() const { return m_light_profile->is_valid(); }

    mi::Float32 evaluate(const mi::Float32_2& theta_phi) const;
    mi::Float32_3 sample(const mi::Float32_3& xi) const;
    mi::Float32 pdf(const mi::Float32_2& theta_phi) const;

protected:
    DB::Access<LIGHTPROFILE::Lightprofile>       m_light_profile;        // the underlying light profile
    DB::Access<LIGHTPROFILE::Lightprofile_impl>  m_light_profile_impl;   // the underlying light profile

    size_t  m_res_t, m_res_p;               // angular resolution of the grid
    float   m_start_t, m_start_p;           // start of the grid
    float   m_delta_t, m_delta_p;           // angular step size
    float   m_inv_delta_t, m_inv_delta_p;   // inverse step size
    float   m_candela_multiplier;           // factor to rescale the normalized data
    float   m_total_power;                  // power of the light source to be able to rescale

    float*  m_cdf_data;                     // CDFs for sampling a light profile
};

}  // MDLRT
}  // MI

#endif //RENDER_MDL_RUNTIME_I_MDLRT_LIGHT_PROFILE_H
