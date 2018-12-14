/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_LIBBSDF_H
#define MDL_LIBBSDF_H

/// The MDL material state inside the MDL SDK.
struct MDL_SDK_State
{
    float3                normal;                  ///< state::normal() result
    float3                geom_normal;             ///< state::geom_normal() result
    float3                position;                ///< state::position() result
    float                 animation_time;          ///< state::animation_time() result
    float3 const         *text_coords;             ///< state::texture_coordinate() table
    float3 const         *tangent_u;               ///< state::texture_tangent_u() table
    float3 const         *tangent_v;               ///< state::texture_tangent_v() table
    float4 const         *text_results;            ///< texture results lookup table
    unsigned char const  *ro_data_segment;         ///< read only data segment

    // these fields are used only if the uniform state is included
    float4 const         *world_to_object;         ///< world-to-object transform matrix
    float4 const         *object_to_world;         ///< object-to-world transform matrix
    int                   object_id;               ///< state::object_id() result
};

struct MDL_SDK_Res_data_pair
{
    void             *shared_data;
    void             *thread_data;
};

enum BSDF_event_flags
{
    BSDF_EVENT_DIFFUSE      = 1,
    BSDF_EVENT_GLOSSY       = 1 << 1,
    BSDF_EVENT_SPECULAR	    = 1 << 2,
    BSDF_EVENT_REFLECTION   = 1 << 3,
    BSDF_EVENT_TRANSMISSION = 1 << 4
};

// type of events created by BSDF importance sampling
enum BSDF_event_type
{
    BSDF_EVENT_ABSORB = 0,
    BSDF_EVENT_DIFFUSE_REFLECTION    = BSDF_EVENT_DIFFUSE  | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_DIFFUSE_TRANSMISSION  = BSDF_EVENT_DIFFUSE  | BSDF_EVENT_TRANSMISSION,
    BSDF_EVENT_GLOSSY_REFLECTION     = BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_GLOSSY_TRANSMISSION   = BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION,
    BSDF_EVENT_SPECULAR_REFLECTION   = BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_SPECULAR_TRANSMISSION = BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION
};

// the calling code can mark an IOR field in *_data with BSDF_USE_MATERIAL_IOR, which will then
// be replaced by BSDF functions with the MDL material's IOR
#define BSDF_USE_MATERIAL_IOR   -1.0f

struct BSDF_sample_data
{
    // Input fields
    float3 ior1; // IOR current medium
    float3 ior2; // IOR other side
    float3 k1;  // outgoing direction
    float3 xi;  // pseudo-random sample number

    // Output fields
    float3 k2;            // incoming direction
    float  pdf;           // pdf (non-projected hemisphere)
    float3 bsdf_over_pdf; // bsdf * dot(normal, k2) / pdf
    int event_type;       // BSDF_event_type
};

struct BSDF_evaluate_data
{
    // Input fields
    float3 ior1;
    float3 ior2;
    float3 k1;
    float3 k2;

    // Output fields
    float3 bsdf;
    float  pdf;
};

struct BSDF_pdf_data
{
    // Input fields
    float3 ior1;
    float3 ior2;
    float3 k1;
    float3 k2;

    // Output fields
    float pdf;
};

// functions provided by libbsdf
typedef void (*mdl_bsdf_sample_function)  (BSDF_sample_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void (*mdl_bsdf_evaluate_function)(BSDF_evaluate_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void (*mdl_bsdf_pdf_function)     (BSDF_pdf_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);



// type of events created by EDF importance sampling
enum EDF_event_type
{
    EDF_EVENT_NONE = 0,
    EDF_EVENT_EMISSION = 1,
};

struct EDF_sample_data
{
    // Input fields
    float3 xi;              // pseudo-random sample number

    // Output fields
    float3 k1;              // outgoing direction
    float pdf;              // pdf (non-projected hemisphere)
    float3 edf_over_pdf;    // edf * dot(normal,k1) / pdf
    int event_type;
};

struct EDF_evaluate_data
{
    // Input fields
    float3 k1;              // outgoing direction

    // Output fields
    float cos;              // dot(normal, k1)
    float3 edf;             // edf
    float pdf;              // pdf (non-projected hemisphere)
};

struct EDF_pdf_data
{
    // Input fields
    float3 k1;              // outgoing direction

    // Output fields
    float pdf;              // pdf (non-projected hemisphere)
};

typedef void(*mdl_edf_sample_function)  (EDF_sample_data *,
                                         MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void(*mdl_edf_evaluate_function)(EDF_evaluate_data *,
                                         MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void(*mdl_edf_pdf_function)     (EDF_pdf_data *,
                                         MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);


#endif // MDL_LIBBSDF_H
