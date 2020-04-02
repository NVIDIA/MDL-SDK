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

#ifndef RENDER_MDL_RUNTIME_I_MDLRT_BSDF_MEASUREMENT_H
#define RENDER_MDL_RUNTIME_I_MDLRT_BSDF_MEASUREMENT_H

#include <mi/neuraylib/typedefs.h>
#include <mi/mdl/mdl_stdlib_types.h>

#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_access.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>

namespace MI {

namespace DB { class Transaction; }

namespace MDLRT {

class Bsdf_measurement
{
public:
    typedef DB::Typed_tag<BSDFM::Bsdf_measurement> Tag_type;
    typedef mi::mdl::stdlib::Mbsdf_part Mbsdf_part;

    Bsdf_measurement();
    Bsdf_measurement(Tag_type const &tag, DB::Transaction *trans);
    virtual ~Bsdf_measurement();

    bool is_valid() const { return m_bsdf_measurement->is_valid(); }

    mi::Uint32_3 get_resolution(Mbsdf_part part) const;

    mi::Float32_3 evaluate(const mi::Float32_2& theta_phi_in,
                           const mi::Float32_2& theta_phi_out,
                           Mbsdf_part part) const;

    mi::Float32_3 sample(const mi::Float32_2& theta_phi_out, 
                         const mi::Float32_3& xi,
                         Mbsdf_part part) const;

    mi::Float32 pdf(const mi::Float32_2& theta_phi_in,
                    const mi::Float32_2& theta_phi_out,
                    Mbsdf_part part) const;

    mi::Float32_4 albedos(const mi::Float32_2& theta_phi) const;

protected:

    void prepare_mbsdfs_part(Mbsdf_part part, const mi::neuraylib::IBsdf_isotropic_data*);
    mi::Float32_2 albedo(const mi::Float32_2& theta_phi, Mbsdf_part part) const;

    DB::Access<BSDFM::Bsdf_measurement>      m_bsdf_measurement;      // the underlying bsdf meas.
    DB::Access<BSDFM::Bsdf_measurement_impl> m_bsdf_measurement_impl; // the underlying bsdf meas.

    unsigned        m_has_data[2];                // true if there is a measurement for this part
    float*          m_eval_data[2];               // uses filter mode cudaFilterModeLinear
    float           m_max_albedo[2];              // max albedo used to limit the multiplier
    float*          m_sample_data[2];             // CDFs for sampling a BSDF measurement
    float*          m_albedo_data[2];             // max albedo for each theta (isotropic)

    mi::Uint32_2    m_angular_resolution[2];      // size of the dataset, needed for texel access
    mi::Float32_2   m_inv_angular_resolution[2];  // the inverse values of the size of the dataset
    unsigned        m_num_channels[2];            // number of color channels (1 or 3)
};

}  // MDLRT
}  // MI

#endif //RENDER_MDL_RUNTIME_I_MDLRT_BSDF_MEASUREMENT_H
