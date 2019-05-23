/******************************************************************************
 * Copyright 2019 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

#ifndef MDL_TARGET_CODE_TYPES_HLSLI
#define MDL_TARGET_CODE_TYPES_HLSLI

 // compiler constants defined from outside:
 // - MDL_NUM_TEXTURE_RESULTS

#ifdef USE_DERIVS
 // Used by the texture runtime.
struct Derived_float2
{
    float2 val;
    float2 dx;
    float2 dy;
};

// Used for the texture coordinates.
struct Derived_float3
{
    float3 val;
    float3 dx;
    float3 dy;
};
#endif

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
    float3            position;

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    float             animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
#ifdef USE_DERIVS
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
    float4            text_results[MDL_NUM_TEXTURE_RESULTS];

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

    /// An offset to add to any argument block read accesses.
    uint              arg_block_offset;
};

#ifdef WITH_ENUM_SUPPORT  // HLSL 2017 and above support enums

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
    // Input fields
    float3           ior1;           ///< IOR current medium
    float3           ior2;           ///< IOR other side
    float3           k1;             ///< outgoing direction
    float3           xi;             ///< pseudo-random sample number

    // Output fields
    float3           k2;             ///< incoming direction
    float            pdf;            ///< pdf (non-projected hemisphere)
    float3           bsdf_over_pdf;  ///< bsdf * dot(normal, k2) / pdf
    Bsdf_event_type  event_type;     ///< the type of event for the generated sample
};

/// Input and output structure for BSDF evaluation data.
struct Bsdf_evaluate_data {
    // Input fields
    float3       ior1;           ///< IOR current medium
    float3       ior2;           ///< IOR other side
    float3       k1;             ///< outgoing direction
    float3       k2;             ///< incoming direction

    // Output fields
    float3       bsdf;           ///< bsdf * dot(normal, k2)
    float        pdf;            ///< pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF PDF calculation data.
struct Bsdf_pdf_data {
    // Input fields
    float3       ior1;           ///< IOR current medium
    float3       ior2;           ///< IOR other side
    float3       k1;             ///< outgoing direction
    float3       k2;             ///< incoming direction

    // Output fields
    float        pdf;            ///< pdf (non-projected hemisphere)
};



/// Input and output structure for EDF sampling data.
struct Edf_sample_data
{
    // Input fields
    float3          xi;             ///< pseudo-random sample number

    // Output fields
    float3          k1;             ///< outgoing direction
    float           pdf;            ///< pdf (non-projected hemisphere)
    float3          edf_over_pdf;   ///< edf * dot(normal,k1) / pdf
    Edf_event_type  event_type;     ///< the type of event for the generated sample
};

/// Input and output structure for EDF evaluation data.
struct Edf_evaluate_data
{
    // Input fields
    float3          k1;             ///< outgoing direction

    // Output fields
    float           cos;            ///< dot(normal, k1)
    float3          edf;            ///< edf
    float           pdf;            ///< pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_pdf_data
{
    // Input fields
    float3          k1;             ///< outgoing direction

    // Output fields
    float           pdf;            ///< pdf (non-projected hemisphere)
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
