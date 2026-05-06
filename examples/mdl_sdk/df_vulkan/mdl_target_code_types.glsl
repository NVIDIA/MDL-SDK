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

// examples/mdl_sdk/df_vulkan/mdl_target_code_types.glsl

#ifndef MDL_TARGET_CODE_TYPES_GLSL
#define MDL_TARGET_CODE_TYPES_GLSL

//-----------------------------------------------------------------------------
// MDL data types and constants
//-----------------------------------------------------------------------------
#define Tex_wrap_mode            int
#define TEX_WRAP_CLAMP           0
#define TEX_WRAP_REPEAT          1
#define TEX_WRAP_MIRRORED_REPEAT 2
#define TEX_WRAP_CLIP            3

#define Bsdf_event_type          int
#define BSDF_EVENT_ABSORB        0
#define BSDF_EVENT_DIFFUSE       1
#define BSDF_EVENT_GLOSSY       (1 << 1)
#define BSDF_EVENT_SPECULAR     (1 << 2)
#define BSDF_EVENT_REFLECTION   (1 << 3)
#define BSDF_EVENT_TRANSMISSION (1 << 4)

#define BSDF_EVENT_DIFFUSE_REFLECTION    (BSDF_EVENT_DIFFUSE  | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_DIFFUSE_TRANSMISSION  (BSDF_EVENT_DIFFUSE  | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_GLOSSY_REFLECTION     (BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_GLOSSY_TRANSMISSION   (BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_SPECULAR_REFLECTION   (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_SPECULAR_TRANSMISSION (BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION)

#define Edf_event_type         int
#define EDF_EVENT_NONE         0
#define EDF_EVENT_EMISSION     1

#define BSDF_USE_MATERIAL_IOR (-1.0)

// Spectral rendering support: type definitions.
// MDL_DF_SPECTRAL_SAMPLES must be propagated by the host (see example_df_vulkan.cpp) so
// that it matches the C++ value from <mi/neuraylib/target_code_types.h>, which can be
// configured via the CMake option MDL_DF_SPECTRAL_SAMPLES_OVERRIDE. Fail loudly if this
// propagation is missing.
#if defined(MDL_SPECTRAL_RENDERING) && !defined(MDL_DF_SPECTRAL_SAMPLES)
#error "MDL_SPECTRAL_RENDERING is defined but MDL_DF_SPECTRAL_SAMPLES was not propagated by the host."
#endif

#ifdef MDL_SPECTRAL_RENDERING

struct Spectral_sample
{
    float values[MDL_DF_SPECTRAL_SAMPLES];
};

#define Color_sample Spectral_sample
#define Pdf_sample   Spectral_sample

#else

#define Color_sample vec3
#define Pdf_sample   float

#endif

/// Flags controlling the calculation of DF results.
/// This cannot be represented as a real enum, because the MDL SDK GLSL backend only sees enums
/// as ints on LLVM level and would create wrong types for temporary variables
#define Df_flags                             int
#define DF_FLAGS_NONE                        0               ///< allows nothing -> black
#define DF_FLAGS_ALLOW_REFLECT               1
#define DF_FLAGS_ALLOW_TRANSMIT              2
#define DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT  (DF_FLAGS_ALLOW_REFLECT | DF_FLAGS_ALLOW_TRANSMIT)
#define DF_FLAGS_ALLOWED_SCATTER_MODE_MASK   (DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT)

struct State
{
    vec3   normal;
    vec3   geom_normal;
    vec3   position;
    float  animation_time;
    vec3   text_coords[1];
    vec3   tangent_u[1];
    vec3   tangent_v[1];
#ifdef NUM_TEX_RESULTS
    vec4   text_results[NUM_TEX_RESULTS];
#endif
    int    ro_data_segment_offset;
    mat4   world_to_object;
    mat4   object_to_world;
    int    object_id;
    float  meters_per_scene_unit;
    int    arg_block_offset;
#ifdef MDL_SPECTRAL_RENDERING
    Spectral_sample spectral_wavelengths;
#endif
};

struct Bsdf_sample_data
{
    /*Input*/  Color_sample ior1;            // IOR current med
    /*Input*/  Color_sample ior2;            // IOR other side
    /*Input*/  vec3         k1;              // outgoing direction
    /*Output*/ vec3         k2;              // incoming direction
    /*Input*/  vec4         xi;              // pseudo-random sample numbers in range [0, 1)
    /*Output*/ Pdf_sample   pdf;             // pdf (non-projected hemisphere)
    /*Output*/ Color_sample bsdf_over_pdf;   // bsdf * dot(normal, k2) / pdf
    /*Output*/ int          event_type;      // the type of event for the generated sample
    /*Output*/ int          handle;          // handle of the sampled elemental BSDF (lobe)
    /*Input*/  Df_flags     flags;           // flags controlling calculation of result
                                             // (optional depending on backend options)
};

struct Bsdf_evaluate_data
{
    /*Input*/  Color_sample ior1;            // IOR current medium
    /*Input*/  Color_sample ior2;            // IOR other side
    /*Input*/  vec3         k1;              // outgoing direction
    /*Input*/  vec3         k2;              // incoming direction
    /*Output*/ Color_sample bsdf_diffuse;    // bsdf_diffuse * dot(normal, k2)
    /*Output*/ Color_sample bsdf_glossy;     // bsdf_glossy * dot(normal, k2)
    /*Output*/ Pdf_sample   pdf;             // pdf (non-projected hemisphere)
    /*Input*/  Df_flags     flags;           // flags controlling calculation of result
                                             // (optional depending on backend options)
};

struct Bsdf_pdf_data
{
    /*Input*/  Color_sample ior1;            // IOR current medium
    /*Input*/  Color_sample ior2;            // IOR other side
    /*Input*/  vec3         k1;              // outgoing direction
    /*Input*/  vec3         k2;              // incoming direction
    /*Output*/ Pdf_sample   pdf;             // pdf (non-projected hemisphere)
    /*Input*/  Df_flags     flags;           // flags controlling calculation of result
                                             // (optional depending on backend options)
};

struct Bsdf_auxiliary_data
{
    /*Input*/  Color_sample ior1;            // IOR current medium
    /*Input*/  Color_sample ior2;            // IOR other side
    /*Input*/  vec3         k1;              // outgoing direction
    /*Output*/ Color_sample albedo_diffuse;  // (diffuse part of the) albedo
    /*Output*/ Color_sample albedo_glossy;   // (glossy part of the) albedo
    /*Output*/ vec3         normal;          // normal
    /*Output*/ vec3         roughness;       // glossy roughness_u, glossy roughness_v, bsdf_weight
    /*Input*/  Df_flags     flags;           // flags controlling calculation of result
                                             // (optional depending on backend options)
};

struct Edf_sample_data
{
    /*Input*/  vec4         xi;              // pseudo-random sample numbers in range [0, 1)
    /*Output*/ vec3         k1;              // outgoing direction
    /*Output*/ Pdf_sample   pdf;             // pdf (non-projected hemisphere)
    /*Output*/ Color_sample edf_over_pdf;    // edf * dot(normal,k1) / pdf
    /*Output*/ int          event_type;      // the type of event for the generated sample
    /*Output*/ int          handle;          // handle of the sampled elemental EDF (lobe)
};

struct Edf_evaluate_data
{
    /*Input*/  vec3         k1;              // outgoing direction
    /*Output*/ float        cos;             // dot(normal, k1)
    /*Output*/ Color_sample edf;             // edf
    /*Output*/ Pdf_sample   pdf;             // pdf (non-projected hemisphere)
};

struct Edf_pdf_data
{
    /*Input*/  vec3       k1;                // outgoing direction
    /*Output*/ Pdf_sample pdf;               // pdf (non-projected hemisphere)
};

struct Edf_auxiliary_data
{
    /*Input*/  vec3   k1;                 // outgoing direction
};

#endif // MDL_TARGET_CODE_TYPES_GLSL
