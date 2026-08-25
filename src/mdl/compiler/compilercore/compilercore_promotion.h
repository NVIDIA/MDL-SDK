/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#ifndef MDL_COMPILERCORE_PROMOTION_H
#define MDL_COMPILERCORE_PROMOTION_H 1

#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// Describes possible effects on the call semantics beyond parameter transforms.
enum Promotion_semantic {
    PS_NONE,                          ///< no semantic change
    PS_UPD_TO_COLOR_FRESNEL_LAYER     ///< call color_fresnel_layer() instead of fresnel_layer()
};

/// Parameter actions: inserts or deletions of parameters with default values.
enum Parameter_action {
    PA_NOTHING                =  0,  ///< no action
    PA_INS_HAIR               =  1,  ///< insert hair_bsdf hair = hair_bsdf()
    PA_INS_EMISSION_INTENSITY =  2,  ///< insert color emission_intensity = color(0.0)
    PA_INS_SELECTOR           =  3,  ///< insert string selector = ""
    PA_INS_MULTISCATTER_TINT  =  4,  ///< insert color multiscatter_tint = color(0.0)
    PA_INS_MULTISCATTER       =  5,  ///< insert bsdf multiscatter= diffuse_reflection_bsdf()
    PA_INS_F82_FACTOR         =  6,  ///< insert color f82_factor = color(1.0)
    PA_INS_MULTIPLIER         =  7,  ///< insert float multiplier = 1.0
    PA_INS_TANGENT_U          =  8,  ///< insert float3 tangent_u = state::texture_tangent_u(0)
    PA_INS_SPREAD             =  9,  ///< insert float spread = math::PI
    PS_INS_ROUNDNESS          = 10,  ///< insert float roundness = 1.0
    PA_INS_UV_TILE            = 11,  ///< insert int2 uv_tile = int2(0,0)
    PA_INS_FRAME              = 12,  ///< insert float frame = 0.0
    PA_INS_BACKSCATTER        = 13,  ///< insert backscatter_modifier backscatter = ..._NONE
    PA_INS_COLLAPSED          = 14,  ///< insert bool collapsed = false
    PA_INS_INTENSITY_MODE     = 15,  ///< insert intensity_mode mode = intensity_radiant_exitance
    PA_WRP_COLOR_CONSTR       = 16,  ///< wrap argument a by color(a)
    PA_UNWRP_COLOR_CONSTR     = 17,  ///< unwrap argument a from color(a)
    PA_REM_PARAM              = 18,  ///< remove a parameter
};

/// Helper struct to denote a parameter transformation.
class Param_transform {
public:
    /// Constructor.
    ///
    /// \param index   the index of the parameter to modify
    /// \param action  the action to apply
    Param_transform(size_t idx = 0, Parameter_action action = PA_NOTHING)
    : m_code((unsigned(idx) << 8u) | action)
    {}

    /// Get the index of the parameter to modify.
    size_t get_index() const { return m_code >> 8u; }

    /// Get the action to apply.
    Parameter_action get_action() const { return Parameter_action(m_code & 0xFF); }

private:
    unsigned m_code;
};

typedef vector<Param_transform>::Type Param_transform_vec;

}  // mdl
}  // mi

#endif  // MDL_COMPILERCORE_PROMOTION_H
