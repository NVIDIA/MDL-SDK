/***************************************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_ARNOLD_BSDF_H
#define MDL_ARNOLD_BSDF_H

#include "mdl_arnold.h"

#include <ai_shader_closure.h>
#include <ai_shader_bsdf.h>
#include <ai_shaderglobals.h>

#include <mi/neuraylib/target_code_types.h>

namespace mi
{
    namespace neuraylib
    {
        class ITarget_code;
    }
}

// more convenient typedefs
typedef mi::neuraylib::tct_float4                               float4;
typedef mi::neuraylib::tct_float3                               float3;
typedef mi::neuraylib::tct_float2                               float2;
typedef mi::neuraylib::tct_float4 const *                       float4x4;

// node constants rendering data when rendering on CPU
struct MdlShaderNodeDataCPU
{
    const mi::neuraylib::ITarget_code* target_code = {nullptr};
    uint64_t surface_bsdf_function_index = {0};
    uint64_t surface_edf_function_index = {0};
    uint64_t surface_emission_intensity_function_index = {0};
    uint64_t backface_bsdf_function_index = {0};
    uint64_t backface_edf_function_index = {0};
    uint64_t backface_emission_intensity_function_index = {0};
    uint64_t cutout_opacity_function_index = {0};
    uint64_t thin_walled_function_index = {0};
    bool thin_walled_constant = {false};    // used instead of generated code in case of a constant
    float cutout_opacity_constant = {1.0f}; // used instead of generated code in case of a constant
};

// typedef some potentially configurable types (e.g., with/without derivatives)
#ifdef ENABLE_DERIVATIVES
    typedef mi::neuraylib::Shading_state_material_with_derivs   Mdl_state;
    typedef mi::neuraylib::tct_deriv_float3                     Tex_coord;
#else
    typedef mi::neuraylib::Shading_state_material               Mdl_state;
    typedef mi::neuraylib::tct_float3                           Tex_coord;
#endif

// Hit point specific data required by the code generated from MDL
struct Mdl_extended_state
{
    // the mdl state and data for which the state only holds a pointer
    float4 texture_results[NUM_TEXTURE_RESULTS + 1];
    Mdl_state state;
    Tex_coord uvw;
    float3 tangent_u;
    float3 tangent_v;

    // parameters
    float3 outgoing;
#ifdef APPLY_BUMP_SHADOW_WEIGHT
    AtVector forward_facing_smooth_normal;
#endif
    bool is_inside;
    bool is_thin_walled;
};

// Hit point specific data required by the code generated from MDL and additional
// fields required by the Arnold BSDF API
struct MdlBSDFData : public Mdl_extended_state
{
    // constant for the material node
    MdlShaderNodeDataCPU shader;

    // lobes, currently diffuse and glossy+specular
    AtBSDFLobeInfo lobe_info[1];
    // TODO: AtBSDFLobeInfo lobe_info[2];

    static AtBSDFMethods* methods;
};

// Fills the MDL state using information given by Arnold
void setup_mdl_state(
    const AtShaderGlobals* sg, 
    Mdl_extended_state& state);

// Forward declaration for the EDF Shader
AtClosure MdlEDFCreate(
    const AtShaderGlobals* sg, 
    const MdlShaderNodeDataCPU* shader_data);

// Forward declaration for the EDF Shader
float MdlOpacityCreate(
    const AtShaderGlobals* sg, 
    const MdlShaderNodeDataCPU* shader_data);

// Forward declaration for the BSDF
AtBSDF* MdlBSDFCreate(
    const AtShaderGlobals* sg, 
    const MdlShaderNodeDataCPU* shader_data);

#endif
