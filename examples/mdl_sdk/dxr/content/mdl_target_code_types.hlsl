/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#if !defined(MDL_TARGET_CODE_TYPES_HLSLI)
#define MDL_TARGET_CODE_TYPES_HLSLI

// compiler constants defined from outside:
// - MDL_NUM_TEXTURE_RESULTS
// - USE_DERIVS
// - MDL_DF_HANDLE_SLOT_MODE (-1, 1, 2, 4, or 8)
#if !defined(MDL_DF_HANDLE_SLOT_MODE)
    #define MDL_DF_HANDLE_SLOT_MODE -1
#endif

#if defined(MDL_NUM_TEXTURE_RESULTS) && (MDL_NUM_TEXTURE_RESULTS > 0)
    #define USE_TEXTURE_RESULTS
#endif


struct Derived_float {
    float val;
    float dx;
    float dy;
};

struct Derived_float2 {
    float2 val;
    float2 dx;
    float2 dy;
};

struct Derived_float3 {
    float3 val;
    float3 dx;
    float3 dy;
};

struct Derived_float4 {
    float4 val;
    float4 dx;
    float4 dy;
};


struct Shading_state_material
{
    /// The result of state::normal().
    /// It represents the shading normal as determined by the renderer.
    /// This field will be updated to the result of \c "geometry.normal" by BSDF init functions,
    /// if requested during code generation.
    float3            normal;

    /// The result of state::geometry_normal().
    /// It represents the geometry normal as determined by the renderer.
    float3            geom_normal;

    /// The result of state::position().
    /// It represents the position where the material should be evaluated.
#if defined(USE_DERIVS)
    Derived_float3    position;
#else
    float3            position;
#endif

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    float             animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
#if defined(USE_DERIVS)
    Derived_float3    text_coords[1];
#else
    float3            text_coords[1];
#endif

    /// An array containing the results of state::texture_tangent_u(i).
    /// The i-th entry represents the texture tangent vector of the i-th texture space at the
    /// current position, which points in the direction of the projection of the tangent to the
    /// positive u axis of this texture space onto the plane defined by the original
    /// surface normal.
    float3            tangent_u[1];

    /// An array containing the results of state::texture_tangent_v(i).
    /// The i-th entry represents the texture bitangent vector of the i-th texture space at the
    /// current position, which points in the general direction of the positive v axis of this
    /// texture space, but is orthogonal to both the original surface normal and the tangent
    /// of this texture space.
    float3            tangent_v[1];

    /// The texture results lookup table.
    /// Values will be modified by BSDF init functions to avoid duplicate texture fetches
    /// and duplicate calculation of values.
    /// This field is only relevant for code generated with
    /// #mi::neuraylib::IMdl_backend::translate_material_df() or
    /// #mi::neuraylib::ILink_unit::add_material_df(). In other cases this may be NULL.
#if defined(USE_TEXTURE_RESULTS)
    float4            text_results[MDL_NUM_TEXTURE_RESULTS];
#endif
    /// An offset for accesses to the read-only data segment. Will be added before
    /// calling any "mdl_read_rodata_as_*" function.
    /// The data of the read-only data segment is accessible as the first segment
    /// (index 0) returned by #mi::neuraylib::ITarget_code::get_ro_data_segment_data().
    uint ro_data_segment_offset;

    /// A 4x4 transformation matrix transforming from world to object coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    float4x4          world_to_object;

    /// A 4x4 transformation matrix transforming from object to world coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    float4x4          object_to_world;

    /// The result of state::object_id().
    /// It is an application-specific identifier of the hit object as provided in a scene.
    /// It can be used to make instanced objects look different in spite of the same used material.
    /// This field is only used if the uniform state is included.
    uint              object_id;

    /// The result of state::meters_per_scene_unit().
    /// The field is only used if the \c "fold_meters_per_scene_unit" option is set to false.
    /// Otherwise, the value of the \c "meters_per_scene_unit" option will be used in the code.
    float             meters_per_scene_unit;

    /// An offset to add to any argument block read accesses.
    uint              arg_block_offset;

#if defined(RENDERER_STATE_TYPE)
    /// A user-defined structure that allows to pass renderer information; for instance about the
    /// hit-point or buffer references; to mdl run-time functions. This is especially required for
    /// the scene data access. The fields of this structure are not altered by generated code.
    RENDERER_STATE_TYPE renderer_state;
#endif
};

#if defined(WITH_ENUM_SUPPORT)  // HLSL 2017 and above support enums

/// The texture wrap modes as defined by \c tex::wrap_mode in the MDL specification.
/// It determines the texture lookup behavior if a lookup coordinate
/// is exceeding the normalized half-open texture space range of [0, 1).
enum Tex_wrap_mode {
    /// \c tex::wrap_clamp: clamps the lookup coordinate to the range
    TEX_WRAP_CLAMP           = 0,

    /// \c tex::wrap_repeat: takes the fractional part of the lookup coordinate
    /// effectively repeating the texture along this axis
    TEX_WRAP_REPEAT          = 1,

    /// \c tex::wrap_mirrored_repeat: like wrap_repeat but takes one minus the fractional part
    /// every other interval to mirror every second instance of the texture
    TEX_WRAP_MIRRORED_REPEAT = 2,

    /// \c tex::wrap_clip: makes the texture lookup return zero for texture coordinates outside
    /// of the range
    TEX_WRAP_CLIP            = 3
};

/// The type of events created by BSDF importance sampling.
enum Bsdf_event_type {
    BSDF_EVENT_ABSORB       = 0,

    BSDF_EVENT_DIFFUSE      = 1,
    BSDF_EVENT_GLOSSY       = 1 << 1,
    BSDF_EVENT_SPECULAR     = 1 << 2,
    BSDF_EVENT_REFLECTION   = 1 << 3,
    BSDF_EVENT_TRANSMISSION = 1 << 4,

    BSDF_EVENT_DIFFUSE_REFLECTION    = BSDF_EVENT_DIFFUSE  | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_DIFFUSE_TRANSMISSION  = BSDF_EVENT_DIFFUSE  | BSDF_EVENT_TRANSMISSION,
    BSDF_EVENT_GLOSSY_REFLECTION     = BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_GLOSSY_TRANSMISSION   = BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION,
    BSDF_EVENT_SPECULAR_REFLECTION   = BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_SPECULAR_TRANSMISSION = BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION,

    BSDF_EVENT_FORCE_32_BIT = 0xffffffffU
};

/// The type of events created by EDF importance sampling.
enum Edf_event_type {
    EDF_EVENT_NONE          = 0,
    EDF_EVENT_EMISSION      = 1,

    EDF_EVENT_FORCE_32_BIT  = 0xffffffffU
};

/// MBSDFs can consist of two parts, which can be selected using this enumeration.
enum Mbsdf_part
{
    /// the bidirectional reflection distribution function (BRDF)
    MBSDF_DATA_REFLECTION = 0,

    /// the bidirectional transmission distribution function (BTDF)
    MBSDF_DATA_TRANSMISSION = 1
};

#else

/// The texture wrap modes as defined by \c tex::wrap_mode in the MDL specification.
#define Tex_wrap_mode             int
#define TEX_WRAP_CLAMP            0
#define TEX_WRAP_REPEAT           1
#define TEX_WRAP_MIRRORED_REPEAT  2
#define TEX_WRAP_CLIP             3

/// The type of events created by BSDF importance sampling.
#define Bsdf_event_type         int
#define BSDF_EVENT_ABSORB       0

#define BSDF_EVENT_DIFFUSE      1
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

#define BSDF_EVENT_FORCE_32_BIT 0xffffffffU

#define Edf_event_type          int
#define EDF_EVENT_NONE          0

#define EDF_EVENT_EMISSION      1
#define EDF_EVENT_FORCE_32_BIT  0xffffffffU

/// MBSDFs can consist of two parts, which can be selected using this enumeration.
#define Mbsdf_part               int
#define MBSDF_DATA_REFLECTION    0
#define MBSDF_DATA_TRANSMISSION  1

#endif

/// The calling code can mark the \c x component of an IOR field in *_data with
/// \c BSDF_USE_MATERIAL_IOR, to make the BSDF functions use the MDL material's IOR
/// for this IOR field.
#define BSDF_USE_MATERIAL_IOR (-1.0f)

/// Input and output structure for BSDF sampling data.
struct Bsdf_sample_data {
    float3 ior1;                    ///< mutual input: IOR current medium
    float3 ior2;                    ///< mutual input: IOR other side
    float3 k1;                      ///< mutual input: outgoing direction

    float3 k2;                      ///< output: incoming direction
    float4 xi;                      ///< input: pseudo-random sample numbers
    float pdf;                      ///< output: pdf (non-projected hemisphere)
    float3 bsdf_over_pdf;           ///< output: bsdf * dot(normal, k2) / pdf
    Bsdf_event_type event_type;     ///< output: the type of event for the generated sample
    int handle;                     ///< output: handle of the sampled elemental BSDF (lobe)
};

/// Input and output structure for BSDF evaluation data.
struct Bsdf_evaluate_data {
    float3 ior1;                    ///< mutual input: IOR current medium
    float3 ior2;                    ///< mutual input: IOR other side
    float3 k1;                      ///< mutual input: outgoing direction

    float3 k2;                      ///< input: incoming direction
    #if (MDL_DF_HANDLE_SLOT_MODE != -1)
        int handle_offset;          ///< output: handle offset to allow the evaluation of more then
                                    ///  DF_HANDLE_SLOTS handles, calling 'evaluate' multiple times
    #endif
    #if (MDL_DF_HANDLE_SLOT_MODE == -1)
        float3 bsdf_diffuse;        ///< output: (diffuse part of the) bsdf * dot(normal, k2)
        float3 bsdf_glossy;         ///< output: (glossy part of the) bsdf * dot(normal, k2)
    #else
        float3 bsdf_diffuse[MDL_DF_HANDLE_SLOT_MODE]; ///< output: (diffuse) bsdf * dot(normal, k2)
        float3 bsdf_glossy[MDL_DF_HANDLE_SLOT_MODE];  ///< output: (glossy) bsdf * dot(normal, k2)
    #endif
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF PDF calculation data.
struct Bsdf_pdf_data {
    float3 ior1;                    ///< mutual input: IOR current medium
    float3 ior2;                    ///< mutual input: IOR other side
    float3 k1;                      ///< mutual input: outgoing direction

    float3 k2;                      ///< input: incoming direction
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF auxiliary calculation data.
struct Bsdf_auxiliary_data {
    float3 ior1;                    ///< mutual input: IOR current medium
    float3 ior2;                    ///< mutual input: IOR other side
    float3 k1;                      ///< mutual input: outgoing direction

    #if (MDL_DF_HANDLE_SLOT_MODE != -1)
        int handle_offset;          ///< output: handle offset to allow the evaluation of more then
                                    ///  DF_HANDLE_SLOTS handles, calling 'auxiliary' multiple times
    #endif
    #if (MDL_DF_HANDLE_SLOT_MODE == -1)
        float3 albedo;              ///< output: albedo
        float3 normal;              ///< output: normal
    #else
        float3 albedo[MDL_DF_HANDLE_SLOT_MODE]; ///< output: albedo
        float3 normal[MDL_DF_HANDLE_SLOT_MODE]; ///< output: normal
    #endif
};

/// Input and output structure for EDF sampling data.
struct Edf_sample_data
{
    float4 xi;                      ///< input: pseudo-random sample numbers
    float3 k1;                      ///< output: outgoing direction
    float pdf;                      ///< output: pdf (non-projected hemisphere)
    float3 edf_over_pdf;            ///< output: edf * dot(normal,k1) / pdf
    Edf_event_type event_type;      ///< output: the type of event for the generated sample
    int handle;                     ///< output: handle of the sampled elemental EDF (lobe)
};

/// Input and output structure for EDF evaluation data.
struct Edf_evaluate_data
{
    float3 k1;                      ///< input: outgoing direction
    #if (MDL_DF_HANDLE_SLOT_MODE != -1)
        int handle_offset;          ///< output: handle offset to allow the evaluation of more then
                                    ///  DF_HANDLE_SLOTS handles, calling 'evaluate' multiple times
    #endif
    float cos;                      ///< output: dot(normal, k1)
    #if (MDL_DF_HANDLE_SLOT_MODE == -1)
        float3 edf;                 ///< output: edf
    #else
        float3 edf[MDL_DF_HANDLE_SLOT_MODE]; ///< output: edf
    #endif
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_pdf_data
{
    float3 k1;                      ///< input: outgoing direction
    float pdf;                      ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_auxiliary_data
{
    float3 k1;                      ///< input: outgoing direction
    #if (MDL_DF_HANDLE_SLOT_MODE != -1)
        int handle_offset;          ///< output: handle offset to allow the evaluation of more then
                                    ///  DF_HANDLE_SLOTS handles, calling 'auxiliary' multiple times
    #endif

    // reserved for future use
};

// Modifies state.normal with the result of "geometry.normal" of the material.
/*void Bsdf_init_function(
    inout Shading_state_material state,
    out float4 texture_results[16],
    uint arg_block_index);

void Bsdf_sample_function(
    inout Bsdf_sample_data data,
    Shading_state_material state,
    float4 texture_results[16],
    uint arg_block_index);

void Bsdf_evaluate_function(
    inout Bsdf_evaluate_data data,
    Shading_state_material state,
    float4 texture_results[16],
    uint arg_block_index);

void Bsdf_pdf_function(
    inout Bsdf_evaluate_data data,
    Shading_state_material state,
    float4 texture_results[16],
    uint arg_block_index);*/

#endif  // MDL_TARGET_CODE_TYPES_HLSLI
