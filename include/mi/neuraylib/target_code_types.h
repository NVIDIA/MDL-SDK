/***************************************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief   Types required for execution of generated native and CUDA code

#ifndef MI_NEURAYLIB_TARGET_CODE_TYPES_H
#define MI_NEURAYLIB_TARGET_CODE_TYPES_H


// If neither TARGET_CODE_USE_CUDA_TYPES nor TARGET_CODE_USE_NEURAY_TYPES is set,
// it will default to CUDA types when compiled by a CUDA compiler and use Neuray types otherwise.

#if defined(TARGET_CODE_USE_CUDA_TYPES) && defined(TARGET_CODE_USE_NEURAY_TYPES)
#error "Only one of TARGET_CODE_USE_CUDA_TYPES and TARGET_CODE_USE_NEURAY_TYPES may be defined."
#endif

#if !defined(TARGET_CODE_USE_NEURAY_TYPES) && \
    (defined(TARGET_CODE_USE_CUDA_TYPES) || defined(__CUDA_ARCH__))

#include <vector_types.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_compiler
@{
*/

typedef float3     tct_float3;
typedef float4     tct_float4;
typedef float      tct_float;
typedef int        tct_int;
typedef unsigned   tct_uint;

#else

#include <mi/neuraylib/typedefs.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_compiler
@{
*/

typedef mi::Float32_3_struct   tct_float3;
typedef mi::Float32_4_struct   tct_float4;
typedef mi::Float32            tct_float;
typedef mi::Sint32             tct_int;
typedef mi::Uint32             tct_uint;

#endif


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

// Forward declaration of texture handler structure.
struct Texture_handler_base;

/// The runtime for bitmap texture access for the generated target code
/// can optionally be implemented in form of a vtable as specified by this structure.
struct Texture_handler_vtable {
    /// Implementation of \c tex::lookup_float4() for a texture_2d texture.
    void (*m_tex_lookup_float4_2d)(
        tct_float                  result[4],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_float            coord[2],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        const tct_float            crop_u[2],
        const tct_float            crop_v[2]);

    /// Implementation of \c tex::lookup_float3() for a texture_2d texture.
    void (*m_tex_lookup_float3_2d)(
        tct_float                  result[3],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_float            coord[2],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        const tct_float            crop_u[2],
        const tct_float            crop_v[2]);

    /// Implementation of \c tex::texel_float4() for a texture_2d texture.
    void (*m_tex_texel_float4_2d)(
        tct_float                  result[4],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_int              coord[2],
        const tct_int              uv_tile[2]);

    /// Implementation of \c tex::lookup_float4() for a texture_3d texture.
    void (*m_tex_lookup_float4_3d)(
        tct_float                  result[4],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_float            coord[3],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        Tex_wrap_mode              wrap_w,
        const tct_float            crop_u[2],
        const tct_float            crop_v[2],
        const tct_float            crop_w[2]);

    /// Implementation of \c tex::lookup_float3() for a texture_3d texture.
    void (*m_tex_lookup_float3_3d)(
        tct_float                  result[3],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_float            coord[3],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        Tex_wrap_mode              wrap_w,
        const tct_float            crop_u[2],
        const tct_float            crop_v[2],
        const tct_float            crop_w[2]);

    /// Implementation of \c tex::texel_float4() for a texture_3d texture.
    void (*m_tex_texel_float4_3d)(
        tct_float                  result[4],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_int              coord[3]);

    /// Implementation of \c tex::lookup_float4() for a texture_cube texture.
    void (*m_tex_lookup_float4_cube)(
        tct_float                  result[4],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_float            coord[3]);

    /// Implementation of \c tex::lookup_float3() for a texture_cube texture.
    void (*m_tex_lookup_float3_cube)(
        tct_float                  result[3],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_float            coord[3]);

    /// Implementation of \c resolution_2d function needed by generated code,
    /// which retrieves the width and height of the given texture.
    void (*m_tex_resolution_2d)(
        tct_int                    result[2],
        const Texture_handler_base *self,
        tct_uint                   texture_idx,
        const tct_int              uv_tile[2]);
};

/// The texture handler structure that is passed to the texturing functions.
/// A user can derive from this structure and add custom fields as required by the texturing
/// function implementations.
struct Texture_handler_base {
    /// In vtable-mode, the vtable field is used to call the texturing functions.
    /// Otherwise, this field may be NULL.
    const Texture_handler_vtable *vtable;
};


/// The MDL material state structure inside the MDL SDK is a representation of the renderer state
/// as defined in section 19 "Renderer state" in the MDL specification.
///
/// It is used to make the state of the renderer (like the position of an intersection point on
/// the surface, the shading normal and the texture coordinates) available to the generated code.
///
/// All spatial values in this structure, i.e. scales, vectors, points and normals, have to be
/// given in internal space (see section 19.2 "Coordinate space transformations" in the MDL
/// specification for more information about the internal space).
/// You can choose the meaning of internal space by setting the \c "internal_space" option via
/// the #mi::neuraylib::IMdl_backend::set_option() method to \c "world" or \c "object".
/// The default is world space.
///
/// The number of available texture spaces should be set with the \c "num_texture_spaces" option
/// of the #mi::neuraylib::IMdl_backend::set_option() method.
///
/// The number of available texture results should be set with the \c "num_texture_results" option
/// of the #mi::neuraylib::IMdl_backend::set_option() method. This option is only relevant if
/// code is generated via #mi::neuraylib::IMdl_backend::translate_material_df() or
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// The MDL specification also mentions some methods which are not or only partially implemented:
///
/// - \c state::motion(), \c state::geometry_tangent_u() and \c state::geometry_tangent_v() are
///   currently only implemented in the GLSL backend,
/// - \c state::wavelength_base() is currently not implemented in any backend,
/// - \c state::rounded_corner_normal() currently just returns the \c normal field of the state.
struct Shading_state_material {
    /// The result of state::normal().
    /// It represents the shading normal as determined by the renderer.
    /// This field will be updated to the result of \c "geometry.normal" by BSDF init functions,
    /// if requested during code generation.
    tct_float3            normal;

    /// The result of state::geometry_normal().
    /// It represents the geometry normal as determined by the renderer.
    tct_float3            geom_normal;

    /// The result of state::position().
    /// It represents the position where the material should be evaluated.
    tct_float3            position;

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    tct_float             animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
    const tct_float3     *text_coords;

    /// An array containing the results of state::texture_tangent_u(i).
    /// The i-th entry represents the texture tangent vector of the i-th texture space at the
    /// current position, which points in the direction of the projection of the tangent to the
    /// positive u axis of this texture space onto the plane defined by the original
    /// surface normal.
    const tct_float3     *tangent_u;

    /// An array containing the results of state::texture_tangent_v(i).
    /// The i-th entry represents the texture bitangent vector of the i-th texture space at the
    /// current position, which points in the general direction of the positive v axis of this
    /// texture space, but is orthogonal to both the original surface normal and the tangent
    /// of this texture space.
    const tct_float3     *tangent_v;

    /// The texture results lookup table.
    /// Values will be modified by BSDF init functions to avoid duplicate texture fetches
    /// and duplicate calculation of values.
    /// This field is only relevant for code generated with
    /// #mi::neuraylib::IMdl_backend::translate_material_df() or
    /// #mi::neuraylib::ILink_unit::add_material_df(). In other cases this may be NULL.
    tct_float4           *text_results;

    /// A pointer to a read-only data segment.
    /// For "PTX" and "LLVM-IR" backend:
    /// - If the MDL code contains large data arrays, compilation time may increase noticeably,
    ///   as a lot of source code will be generated for the arrays.
    ///   To avoid this, you can set the \c "enable_ro_segment" option to \c "on" via the
    ///   #mi::neuraylib::IMdl_backend::set_option() method. Then, data of arrays larger than 1024
    ///   bytes will be stored in a read-only data segment, which is accessible as the first
    ///   segment (index 0) returned by #mi::neuraylib::ITarget_code::get_ro_data_segment_data().
    ///   The generated code will expect, that you make this data available via the
    ///   \c ro_data_segment field of the MDL material state. Depending on the target platform
    ///   this may require copying the data to the GPU.
    ///
    /// For other backends, this should be NULL.
    const char           *ro_data_segment;

    /// A 4x4 transformation matrix transforming from world to object coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    const tct_float4     *world_to_object;

    /// A 4x4 transformation matrix transforming from object to world coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    const tct_float4     *object_to_world;

    /// The result of state::object_id().
    /// It is an application-specific identifier of the hit object as provided in a scene.
    /// It can be used to make instanced objects look different in spite of the same used material.
    /// This field is only used if the uniform state is included.
    tct_int                object_id;
};

/// The MDL environment state structure inside the MDL SDK is a representation of the renderer
/// state in the context of an environment lookup as defined in section 19 "Renderer state" in the
/// MDL specification.
///
/// It is used to make the state of the renderer (like the evaluation direction for the
/// environment) available to the generated code.
///
/// All spatial values in this structure, i.e. scales, vectors, points and normals, have to be
/// given in internal space (see section 19.2 "Coordinate space transformations" in the MDL
/// specification for more information about the internal space).
/// You can choose the meaning of internal space by setting the \c "internal_space" option via
/// the #mi::neuraylib::IMdl_backend::set_option() method to \c "world" or \c "object".
/// The default is world space.
struct Shading_state_environment {
    /// The result of state::direction().
    /// It represents the lookup direction for the environment lookup.
    tct_float3            direction;
};

/// The data structure providing access to resources for generated code.
struct Resource_data {
    const void                 *shared_data;      ///< currently unused, should be NULL
    const Texture_handler_base *texture_handler;  ///< will be provided as "self" parameter to
                                                  ///< texture functions
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

/// The calling code can mark the \c x component of an IOR field in *_data with
/// \c MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR, to make the BSDF functions use the MDL material's IOR
/// for this IOR field.
#define MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR (-1.0f)

/// Input and output structure for BSDF sampling data.
struct Bsdf_sample_data {
    // Input fields
    tct_float3       ior1;           ///< IOR current medium
    tct_float3       ior2;           ///< IOR other side
    tct_float3       k1;             ///< outgoing direction
    tct_float3       xi;             ///< pseudo-random sample number

    // Output fields
    tct_float3       k2;             ///< incoming direction
    tct_float        pdf;            ///< pdf (non-projected hemisphere)
    tct_float3       bsdf_over_pdf;  ///< bsdf * dot(normal, k2) / pdf
    Bsdf_event_type  event_type;     ///< the type of event for the generated sample
};

/// Input and output structure for BSDF evaluation data.
struct Bsdf_evaluate_data {
    // Input fields
    tct_float3       ior1;           ///< IOR current medium
    tct_float3       ior2;           ///< IOR other side
    tct_float3       k1;             ///< outgoing direction
    tct_float3       k2;             ///< incoming direction

    // Output fields
    tct_float3       bsdf;           ///< bsdf * dot(normal, k2)
    tct_float        pdf;            ///< pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF PDF calculation data.
struct Bsdf_pdf_data {
    // Input fields
    tct_float3       ior1;           ///< IOR current medium
    tct_float3       ior2;           ///< IOR other side
    tct_float3       k1;             ///< outgoing direction
    tct_float3       k2;             ///< incoming direction

    // Output fields
    tct_float        pdf;            ///< pdf (non-projected hemisphere)
};


// Signatures for generated target code functions.

/// Signature of environment functions created via
/// #mi::neuraylib::IMdl_backend::translate_environment() and
/// #mi::neuraylib::ILink_unit::add_environment().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   unused, should be NULL
typedef void (Environment_function)  (void *result,
                                      const Shading_state_environment *state,
                                      const Resource_data *res_data,
                                      const void *exception_state,
                                      const char *arg_block_data);

/// Signature of material expression functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_expression() and
/// #mi::neuraylib::ILink_unit::add_material_expression().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Material_expr_function)  (void *result,
                                        const Shading_state_material *state,
                                        const Resource_data *res_data,
                                        const void *exception_state,
                                        const char *arg_block_data);

/// Signature of the initialization function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// This function updates the normal field of the shading state with the result of
/// \c "geometry.normal" and, if the \c "num_texture_results" backend option has been set to
/// non-zero, fills the text_results fields of the state.
///
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_init_function)    (Shading_state_material *state,
                                      const Resource_data *res_data,
                                      const void *exception_state,
                                      const char *arg_block_data);

/// Signature of the importance sampling function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_sample_function)  (Bsdf_sample_data *data,
                                      const Shading_state_material *state,
                                      const Resource_data *res_data,
                                      const void *exception_state,
                                      const char *arg_block_data);

/// Signature of the evaluation function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_evaluate_function)(Bsdf_evaluate_data *data,
                                      const Shading_state_material *state,
                                      const Resource_data *res_data,
                                      const void *exception_state,
                                      const char *arg_block_data);

/// Signature of the probability density function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_pdf_function)     (Bsdf_pdf_data *data,
                                      const Shading_state_material *state,
                                      const Resource_data *res_data,
                                      const void *exception_state,
                                      const char *arg_block_data);

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_TARGET_CODE_TYPES_H
