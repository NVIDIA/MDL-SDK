/***************************************************************************************************
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
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

/// Type of Bsdf_evaluate_data variants, depending on the backend and its configuration.
#define BSDF_HSMP -2   ///< renderer defined buffers; not supported by all backends
#define BSDF_HSMN -1   ///< No slots, handles are ignored completely
#define BSDF_HSM1  1   ///< fixed size array for processing 1 handle at a time
#define BSDF_HSM2  2   ///< fixed size array for processing 2 handle at a time
#define BSDF_HSM4  4   ///< fixed size array for processing 4 handle at a time
#define BSDF_HSM8  8   ///< fixed size array for processing 8 handle at a time

// define has to be set by the compiler
#ifndef MDL_DF_HANDLE_SLOT_MODE
#error "Compile constant 'MDL_DF_HANDLE_SLOT_MODE' not defined."
#endif

struct __align__(16) BSDF_sample_data
{
    float3 ior1;                // mutual input: IOR current medium
    float3 ior2;                // mutual input: IOR other side
    float3 k1;                  // mutual input: outgoing direction

    float3 k2;                  // output:  incoming direction
    float4 xi;                  // input:   pseudo-random sample numbers
    
    float  pdf;                 // output:  pdf (non-projected hemisphere)
    float3 bsdf_over_pdf;       // output:  bsdf * dot(normal, k2) / pdf
    int event_type;             // output:  BSDF_event_type
    int handle;                 // output:  handle of the sampled elemental BSDF (lobe)
};

struct __align__(16) BSDF_evaluate_data
{
    float3 ior1;                // mutual input: IOR current medium
    float3 ior2;                // mutual input: IOR other side
    float3 k1;                  // mutual input: outgoing direction

    float3 k2;
    #if MDL_DF_HANDLE_SLOT_MODE != BSDF_HSMN
        int handle_offset;      // input: handle offset to allow the evaluation of multiple handles 
    #endif
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        int handle_count;       // input: number of elements of 'bsdf_diffuse' and 'bsdf_glossy'
    #endif

    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        float3 bsdf_diffuse;    // output: (diffuse part of the) bsdf * dot(normal, k2)
        float3 bsdf_glossy;     // output: (glossy part of the) bsdf * dot(normal, k2)
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        float3* bsdf_diffuse;   // output: (diffuse part of the) bsdf * dot(normal, k2)
        float3* bsdf_glossy;    // output: (glossy part of the) bsdf * dot(normal, k2)
    #else
        float3 bsdf_diffuse[MDL_DF_HANDLE_SLOT_MODE]; // output: (diffuse) bsdf * dot(normal, k2)
        float3 bsdf_glossy [MDL_DF_HANDLE_SLOT_MODE]; // output: (glossy) bsdf * dot(normal, k2)
    #endif
    float  pdf;                 // output: pdf (non-projected hemisphere)
};

struct __align__(16) BSDF_pdf_data
{
    float3 ior1;                // mutual input: IOR current medium
    float3 ior2;                // mutual input: IOR other side
    float3 k1;                  // mutual input: outgoing direction

    float3 k2;                  // input:   incoming direction
    float pdf;                  // output:  pdf (non-projected hemisphere)
};

struct __align__(16) BSDF_auxiliary_data
{
    float3 ior1;                // mutual input: IOR current medium
    float3 ior2;                // mutual input: IOR other side
    float3 k1;                  // mutual input: outgoing direction

    #if MDL_DF_HANDLE_SLOT_MODE != BSDF_HSMN
        int handle_offset;      // input: handle offset to allow the evaluation of multiple handles 
    #endif
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        int handle_count;       // input: number of elements of 'albedo' and 'normal'
    #endif

    // output: albedo 2x float3
    // The albedo output is split into diffuse and glossy albedo corresponding the BSDF_evaluate_data.
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        float3 albedo_diffuse;
        float3 albedo_glossy;
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        float3* albedo_diffuse;
        float3* albedo_glossy;
    #else
        float3 albedo_diffuse[MDL_DF_HANDLE_SLOT_MODE];
        float3 albedo_glossy[MDL_DF_HANDLE_SLOT_MODE];
    #endif

    // output: normal
    // The normal is normlized if not a zero vector. It is computed as a weighted linear combination
    // based on the weights of the BSDF in the layering tree.
    // For a weighted combination of normals from multiple calls to the auxiliary function,
    // one could aggregate the slot normals using the wights provided in the last component of the
    // `roughness` output of the corresponding slot.
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        float3 normal;
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        float3* normal;
    #else
        float3 normal[MDL_DF_HANDLE_SLOT_MODE];
    #endif

    // output: roughness (float3)
    // The components x and y contain roughness values corresponding to roughness_u and roughness_v
    // as specified by glossy BSDFs. Diffuse BSDFs are considered to have a roughness of 1.0 in this
    // context. Specular BSDFs have a roughness of 0.0. The invalid black BSDF as well as BSDF
    // measurements also report a roughness of 0.0.
    // The z component carries the cumulated weight of the BSDFs in the layering tree.
    //
    // If only a single BSDF is used in the material or is evaluated using handles, x and y reproduce
    // exactly the arguments passed to the BSDF. The z component will hold the weight of the evaluated
    // BSDF in the layering tree. When aggregating roughness values over multiple calls to the auxiliary
    // function, one could compute a weighted average by: 
    //      sum(roughness_i.xy * roughness_i.z) / sum(roughness_i.z)
    //
    // When multiple elemental BSDF are evaluated at once (e.g. without handles), the components x and y
    // will already contain the weighted average computed by the formula above. The z component will
    // hold the cumulated weights in order to allow the same external aggregation.
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        float3 roughness;
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        float3* roughness;
    #else
        float3 roughness[MDL_DF_HANDLE_SLOT_MODE];
    #endif
};


// functions provided by libbsdf
typedef void (*mdl_bsdf_sample_function)  (BSDF_sample_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void (*mdl_bsdf_evaluate_function)(BSDF_evaluate_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void (*mdl_bsdf_pdf_function)     (BSDF_pdf_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void(*mdl_bsdf_auxiliary_function)(BSDF_auxiliary_data *,
                                           MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);

// type of events created by EDF importance sampling
enum EDF_event_type
{
    EDF_EVENT_NONE = 0,
    EDF_EVENT_EMISSION = 1,
};

struct __align__(16) EDF_sample_data
{
    float4 xi;                  // input: pseudo-random sample numbers

    float3 k1;                  // output: outgoing direction
    float pdf;                  // output: pdf (non-projected hemisphere)
    float3 edf_over_pdf;        // output: edf * dot(normal,k1) / pdf
    int event_type;
    int handle;                 // output: handle of the sampled elemental EDF (lobe)
};

struct __align__(16) EDF_evaluate_data
{
    float3 k1;                  // input: outgoing direction
    #if MDL_DF_HANDLE_SLOT_MODE != BSDF_HSMN
        int handle_offset;      // input: handle offset to allow the evaluation of multiple handles 
    #endif
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        int handle_count;       // input: number of elements of 'edf'
    #endif

    float cos;                  // output: dot(normal, k1)
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMN
        float3 edf;             // output: edf
    #elif MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        float3* edf;            // output: edf
    #else
        float3 edf[MDL_DF_HANDLE_SLOT_MODE]; // output: edf
    #endif
    float pdf;                  // output: pdf (non-projected hemisphere)
};

struct __align__(16) EDF_pdf_data
{
    float3 k1;                  // input: outgoing direction

    float pdf;                  // output: pdf (non-projected hemisphere)
};

struct __align__(16) EDF_auxiliary_data
{
    float3 k1;                  // input: outgoing direction
    #if MDL_DF_HANDLE_SLOT_MODE != BSDF_HSMN
        int handle_offset;      // input: handle offset to allow the evaluation of multiple handles 
    #endif
    #if MDL_DF_HANDLE_SLOT_MODE == BSDF_HSMP
        int handle_count;       // input: number of elements of 'output buffer fields'
    #endif

    // reserved for future use
};

typedef void(*mdl_edf_sample_function)      (EDF_sample_data *,
                                             MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void(*mdl_edf_evaluate_function)    (EDF_evaluate_data *,
                                             MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void(*mdl_edf_pdf_function)         (EDF_pdf_data *,
                                             MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);
typedef void(*mdl_edf_auxiliary_function)   (EDF_auxiliary_data *,
                                             MDL_SDK_State *, MDL_SDK_Res_data_pair *, void *);

enum BSDF_type
{
    ELEMENTAL_START,

    DIFFUSE_REFLECTION_BSDF = ELEMENTAL_START,
    DIFFUSE_TRANSMISSION_BSDF,
    SPECULAR_BSDF,
    SIMPLE_GLOSSY_BSDF,
    MICROFACET_BECKMANN_VCAVITIES_BSDF,
    MICROFACET_GGX_VCAVITIES_BSDF,
    MICROFACET_BECKMANN_SMITH_BSDF,
    MICROFACET_GGX_SMITH_BSDF,
    BACKSCATTERING_GLOSSY_BSDF,
    WARD_GEISLER_MORODER_BSDF,
    SHEEN_BSDF,
    MEASURED_BSDF,
    BLACK_BSDF,

    NON_ELEMENTAL_START,

    TINT = NON_ELEMENTAL_START,
    WEIGHTED_LAYER,
    COLOR_WEIGHTED_LAYER,

    FRESNEL_LAYER,
    COLOR_FRESNEL_LAYER,
    CUSTOM_CURVE_LAYER,
    COLOR_CUSTOM_CURVE_LAYER,
    DIRECTIONAL_FACTOR,
    MEASURED_CURVE_FACTOR,
    MEASURED_CURVE_LAYER,
    COLOR_MEASURED_CURVE_LAYER,
    THIN_FILM,
    FRESNEL_FACTOR,
    MEASURED_FACTOR,  // TODO

    MIX_START,

    NORMALIZED_MIX = MIX_START,
    COLOR_NORMALIZED_MIX,
    CLAMPED_MIX,
    COLOR_CLAMPED_MIX,
    UNBOUNDED_MIX,
    COLOR_UNBOUNDED_MIX,

    TERNARY_OPERATOR,
};

#endif // MDL_LIBBSDF_H
