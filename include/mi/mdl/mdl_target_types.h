/******************************************************************************
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
 *****************************************************************************/
/// \file mi/mdl/mdl_target_types.h
/// \brief Declaration of types used by the generated target code
#ifndef MDL_TARGET_TYPES_H
#define MDL_TARGET_TYPES_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_stdlib_types.h>

namespace mi {
namespace mdl {

/// A simple float2 type.
struct Float2_struct {
    float x, y;
};

/// A simple float3 type.
struct Float3_struct {
    float x, y, z;
};

/// A simple float4 type.
struct Float4_struct {
    float x, y, z, w;
};

/// A simple float4x4 matrix.
struct Matrix4x4_struct {
    float elements[4*4];
};

/// A simple float3x3 matrix.
struct Matrix3x3_struct {
    float elements[3*3];
};

/// A simple RGB color.
struct RGB_color {
    float r, g, b;
};

#if (defined(MDL_CORE_TARGET_CODE_USE_CUDA_TYPES) || defined(__CUDA_ARCH__))
/// Inside CUDA, remap the CUDA floatX type to our tct_floatX types.
/// \{
typedef float2 tct_float2;
typedef float3 tct_float3;
typedef float4 tct_float4;
/// \}
#else
/// On native code, use our simple struct type to represent tct_floatX types.
/// \{
typedef Float2_struct tct_float2;
typedef Float3_struct tct_float3;
typedef Float4_struct tct_float4;
/// \}
#endif

/// The MDL environment state structure inside MDL Core is a representation of the renderer
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
/// the #mi::mdl::ICode_generator::access_options() method to \c "coordinate_object" or
/// \c "coordinate_world". The default is world space.
struct Shading_state_environment {
    /// The result of state::direction().
    /// It represents the lookup direction for the environment lookup.
    tct_float3 direction;
};

/// The MDL material state structure inside MDL Core is a representation of the renderer state
/// as defined in section 19 "Renderer state" in the MDL specification.
///
/// It is used to make the state of the renderer (like the position of an intersection point on
/// the surface, the shading normal and the texture coordinates) available to the generated code.
///
/// All spatial values in this structure, i.e. scales, vectors, points and normals, have to be
/// given in internal space (see section 19.2 "Coordinate space transformations" in the MDL
/// specification for more information about the internal space).
/// You can choose the meaning of internal space by setting the \c "internal_space" option via
/// the #mi::mdl::ICode_generator::access_options() method to \c "coordinate_object" or
/// \c "coordinate_world". The default is world space.
///
/// The number of available texture spaces should be set with the \c "num_texture_spaces" parameter
/// of the various MDL Core backends.
///
/// The number of available texture results should be set with the \c "num_texture_results"
/// parameter of the various MDL Core backends.
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
    tct_float3          normal;

    /// The result of state::geometry_normal().
    /// It represents the geometry normal as determined by the renderer.
    tct_float3          geom_normal;

    /// The result of state::position().
    /// It represents the position where the material should be evaluated.
    tct_float3          position;

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    float               animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
    tct_float3 const    *text_coords;

    /// An array containing the results of state::texture_tangent_u(i).
    /// The i-th entry represents the texture tangent vector of the i-th texture space at the
    /// current position, which points in the direction of the projection of the tangent to the
    /// positive u axis of this texture space onto the plane defined by the original
    /// surface normal.
    tct_float3 const    *tangent_u;

    /// An array containing the results of state::texture_tangent_v(i).
    /// The i-th entry represents the texture bitangent vector of the i-th texture space at the
    /// current position, which points in the general direction of the positive v axis of this
    /// texture space, but is orthogonal to both the original surface normal and the tangent
    /// of this texture space.
    tct_float3 const    *tangent_v;

    /// The texture results lookup table.
    /// Values will be modified by BSDF init functions to avoid duplicate texture fetches
    /// and duplicate calculation of values.
    tct_float4 const    *text_results;

    /// A pointer to a read-only data segment.
    /// For "PTX" and "native" JIT backend:
    /// - If the MDL code contains large data arrays, compilation time may increase noticeably,
    ///   as a lot of source code will be generated for the arrays.
    ///   To avoid this, you can set the \c "jit_enable_ro_segment" option to \c "true" via the
    ///   #mi::mdl::ICode_generator::access_options() method. Then, data of arrays larger than 1024
    ///   bytes will be stored in a read-only data segment, which is accessible as the first
    ///   segment (index 0) returned by #mi::mdl::IGenerated_code_executable::get_ro_data_segment().
    ///   The generated code will expect, that you make this data available via the
    ///   \c ro_data_segment field of the MDL material state. Depending on the target platform
    ///   this may require copying the data to the GPU.
    ///
    /// For other backends, this should be NULL.
    char const          *ro_data_segment;

    // these fields are used only if the uniform state is included

    /// A 4x4 transformation matrix transforming from world to object coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    tct_float4 const    *world_to_object;

    /// A 4x4 transformation matrix transforming from object to world coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    tct_float4 const    *object_to_world;

    /// The result of state::object_id().
    /// It is an application-specific identifier of the hit object as provided in a scene.
    /// It can be used to make instanced objects look different in spite of the same used material.
    /// This field is only used if the uniform state is included.
    int                  object_id;
};

/// The MDL material state structure inside MDL Core is a representation of the renderer state
/// as defined in section 19 "Renderer state" in the MDL specification.
///
/// It is used to make the state of the renderer (like the position of an intersection point on
/// the surface, the shading normal and the texture coordinates) available to the generated code.
///
/// All spatial values in this structure, i.e. scales, vectors, points and normals, have to be
/// given in internal space (see section 19.2 "Coordinate space transformations" in the MDL
/// specification for more information about the internal space).
/// You can choose the meaning of internal space by setting the \c "internal_space" option via
/// the #mi::mdl::ICode_generator::access_options() method to \c "coordinate_object" or
/// \c "coordinate_world". The default is world space.
///
/// This variant of the state structure uses bi-tangents instead of separate u and v tangents.
/// To enable the use of this structure, you have to set the \c "jit_use_bitangent" option
/// to \c "true".
///
/// The number of available texture spaces should be set with the \c "num_texture_spaces" parameter
/// of the various MDL Core backends.
///
/// The number of available texture results should be set with the \c "num_texture_results"
/// parameter of the various MDL Core backends.
///
/// The MDL specification also mentions some methods which are not or only partially implemented:
///
/// - \c state::motion(), \c state::geometry_tangent_u() and \c state::geometry_tangent_v() are
///   currently only implemented in the GLSL backend,
/// - \c state::wavelength_base() is currently not implemented in any backend,
/// - \c state::rounded_corner_normal() currently just returns the \c normal field of the state.
struct Shading_state_material_bitangent {
    /// The result of state::normal().
    /// It represents the shading normal as determined by the renderer.
    /// This field will be updated to the result of \c "geometry.normal" by BSDF init functions,
    /// if requested during code generation.
    tct_float3          normal;

    /// The result of state::geometry_normal().
    /// It represents the geometry normal as determined by the renderer.
    tct_float3          geom_normal;

    /// The result of state::position().
    /// It represents the position where the material should be evaluated.
    tct_float3          position;

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    float               animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
    tct_float3 const    *text_coords;

    /// An array containing the combined results of state::texture_tangent[_u|_v](i).
    /// The i-th entry represents the texture tangent vector of the i-th texture space at the
    /// current position, which points in the direction of the projection of the tangent to the
    /// positive u/v axis of this texture space onto the plane defined by the original
    /// surface normal.
    tct_float4 const    *tangents_bitangentssign;

    /// The texture results lookup table.
    /// Values will be modified by BSDF init functions to avoid duplicate texture fetches
    /// and duplicate calculation of values.
    tct_float4 const    *text_results;

    /// A pointer to a read-only data segment.
    /// For "PTX" and "native" JIT backend:
    /// - If the MDL code contains large data arrays, compilation time may increase noticeably,
    ///   as a lot of source code will be generated for the arrays.
    ///   To avoid this, you can set the \c "jit_enable_ro_segment" option to \c "true" via the
    ///   #mi::mdl::ICode_generator::access_options() method. Then, data of arrays larger than 1024
    ///   bytes will be stored in a read-only data segment, which is accessible as the first
    ///   segment (index 0) returned by #mi::mdl::IGenerated_code_executable::get_ro_data_segment().
    ///   The generated code will expect, that you make this data available via the
    ///   \c ro_data_segment field of the MDL material state. Depending on the target platform
    ///   this may require copying the data to the GPU.
    ///
    /// For other backends, this should be NULL.
    char const          *ro_data_segment;

    // these fields are used only if the uniform state is included

    /// A 4x4 transformation matrix transforming from world to object coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    tct_float4 const    *world_to_object;

    /// A 4x4 transformation matrix transforming from object to world coordinates.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    tct_float4 const    *object_to_world;

    /// The result of state::object_id().
    /// It is an application-specific identifier of the hit object as provided in a scene.
    /// It can be used to make instanced objects look different in spite of the same used material.
    /// This field is only used if the uniform state is included.
    int                  object_id;
};

// Forward declaration of texture handler structure.
struct Texture_handler_base;

/// The runtime for bitmap texture access for the generated target code
/// can optionally be implemented in form of a vtable as specified by this structure.
struct Texture_handler_vtable {
    typedef mi::mdl::stdlib::Tex_wrap_mode Tex_wrap_mode;

    /// Implementation of \c tex::lookup_float4() for a texture_2d texture.
    void (*m_tex_lookup_float4_2d)(
        float                      result[4],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        float const                coord[2],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        float const                crop_u[2],
        float const                crop_v[2]);

    /// Implementation of \c tex::lookup_float3() for a texture_2d texture.
    void (*m_tex_lookup_float3_2d)(
        float                      result[3],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        float const                coord[2],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        float const                crop_u[2],
        float const                crop_v[2]);

    /// Implementation of \c tex::texel_float4() for a texture_2d texture.
    void (*m_tex_texel_float4_2d)(
        float                      result[4],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        int const                  coord[2],
        int const                  uv_tile[2]);

    /// Implementation of \c tex::lookup_float4() for a texture_3d texture.
    void (*m_tex_lookup_float4_3d)(
        float                      result[4],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        float const                coord[3],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        Tex_wrap_mode              wrap_w,
        float const                crop_u[2],
        float const                crop_v[2],
        float const                crop_w[2]);

    /// Implementation of \c tex::lookup_float3() for a texture_3d texture.
    void (*m_tex_lookup_float3_3d)(
        float                      result[3],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        float const                coord[3],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        Tex_wrap_mode              wrap_w,
        float const                crop_u[2],
        float const                crop_v[2],
        float const                crop_w[2]);

    /// Implementation of \c tex::texel_float4() for a texture_3d texture.
    void (*m_tex_texel_float4_3d)(
        float                      result[4],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        int const                  coord[3]);

    /// Implementation of \c tex::lookup_float4() for a texture_cube texture.
    void (*m_tex_lookup_float4_cube)(
        float                      result[4],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        float const                coord[3]);

    /// Implementation of \c tex::lookup_float3() for a texture_cube texture.
    void (*m_tex_lookup_float3_cube)(
        float                      result[3],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        float const                coord[3]);

    /// Implementation of \c resolution_2d function needed by generated code,
    /// which retrieves the width and height of the given texture.
    void (*m_tex_resolution_2d)(
        int                        result[2],
        Texture_handler_base const *self,
        unsigned                   texture_idx,
        int const                  uv_tile[2]);
};

/// The texture handler structure that is passed to the texturing functions.
/// A user can derive from this structure and add custom fields as required by the texturing
/// function implementations.
struct Texture_handler_base {
    /// In vtable-mode, the vtable field is used to call the texturing functions.
    /// Otherwise, this field may be NULL.
    Texture_handler_vtable const *vtable;
};

/// The data structure providing access to resources for generated code.
struct Resource_data {
    void const                 *shared_data;      ///< currently unused, should be NULL
    Texture_handler_base const *texture_handler;  ///< will be provided as "self" parameter to
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
/// \c MDL_CORE_BSDF_USE_MATERIAL_IOR, to make the BSDF functions use the MDL material's IOR
/// for this IOR field.
#define MDL_CORE_BSDF_USE_MATERIAL_IOR (-1.0f)

/// Input and output structure for BSDF sampling data.
struct Bsdf_sample_data {
    // Input fields
    tct_float3      ior1;           ///< IOR current medium
    tct_float3      ior2;           ///< IOR other side
    tct_float3      k1;             ///< outgoing direction
    tct_float3      xi;             ///< pseudo-random sample number

    // Output fields
    tct_float3      k2;             ///< incoming direction
    float           pdf;            ///< pdf (non-projected hemisphere)
    tct_float3      bsdf_over_pdf;  ///< bsdf * dot(normal, k2) / pdf
    Bsdf_event_type event_type;     ///< the type of event for the generated sample
};

/// Input and output structure for BSDF evaluation data.
struct Bsdf_evaluate_data {
    // Input fields
    tct_float3      ior1;           ///< IOR current medium
    tct_float3      ior2;           ///< IOR other side
    tct_float3      k1;             ///< outgoing direction
    tct_float3      k2;             ///< incoming direction

    // Output fields
    tct_float3      bsdf;           ///< bsdf * dot(normal, k2)
    float           pdf;            ///< pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF PDF calculation data.
struct Bsdf_pdf_data {
    // Input fields
    tct_float3      ior1;           ///< IOR current medium
    tct_float3      ior2;           ///< IOR other side
    tct_float3      k1;             ///< outgoing direction
    tct_float3      k2;             ///< incoming direction

    // Output fields
    float           pdf;            ///< pdf (non-projected hemisphere)
};


// Signatures for generated target code functions.

/// Signature of environment functions created via
/// #mi::mdl::ICode_generator_jit::compile_into_environment().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   unused, should be NULL
typedef void (Lambda_environment_function)(
    void                            *result,
    Shading_state_environment const *state,
    Resource_data const             *res_data,
    void const                      *exception_state,
    char const                      *arg_block_data);

/// Signature of constant expression functions created via
/// #mi::mdl::ICode_generator_jit::compile_into_const_function().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Lambda_const_function)(
    void                *result,
    Resource_data const *res_data,
    void const          *exception_state,
    char const          *arg_block_data);

/// Signature of material expression functions created via
/// #mi::mdl::ICode_generator_jit::compile_into_switch_function(),
/// #mi::mdl::ICode_generator_jit::compile_into_switch_function_for_gpu() and for switch lambdas via
/// #mi::mdl::ICode_generator_jit::compile_into_llvm_ir() and
/// #mi::mdl::ICode_generator_jit::compile_into_ptx().
///
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param index            the index of the expression to compute
typedef void (Lambda_switch_function)(
    Shading_state_material *state,
    Resource_data const    *res_data,
    void const             *exception_state,
    char const             *arg_block_data,
    void                   *result,
    unsigned                index);

/// Signature of material expression functions created via
/// #mi::mdl::ICode_generator_jit::compile_into_generic_function(),
/// #mi::mdl::ICode_generator_jit::compile_into_ptx() and for generic lambdas via
/// #mi::mdl::ICode_generator_jit::compile_into_llvm_ir() and #mi::mdl::ILink_unit::add().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Lambda_generic_function)(
    void                   *result,
    Shading_state_material *state,
    Resource_data const    *res_data,
    void const             *exception_state,
    char const             *arg_block_data);

typedef Lambda_generic_function Material_expr_function;

/// Signature of the initialization function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu(),
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// This function updates the normal field of the shading state with the result of
/// \c "geometry.normal" and, if the \c "num_texture_results" backend option has been set to
/// non-zero, fills the text_results fields of the state.
///
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_init_function)(
    Shading_state_material *state,
    Resource_data const    *res_data,
    void const             *exception_state,
    char const             *arg_block_data);

/// Signature of the importance sampling function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu() and
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_sample_function)(
    Bsdf_sample_data             *data,
    Shading_state_material const *state,
    Resource_data const          *res_data,
    void const                   *exception_state,
    char const                   *arg_block_data);

/// Signature of the evaluation function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu() and
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_evaluate_function)(
    Bsdf_evaluate_data           *data,
    Shading_state_material const *state,
    Resource_data const          *res_data,
    void const                   *exception_state,
    char const                   *arg_block_data);

/// Signature of the probability density function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu() and
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_pdf_function)(
    Bsdf_pdf_data                *data,
    Shading_state_material const *state,
    Resource_data const          *res_data,
    void const                   *exception_state,
    char const                   *arg_block_data);


/// The type of events created by EDF importance sampling.
enum Edf_event_type
{
    EDF_EVENT_NONE = 0,
    EDF_EVENT_EMISSION = 1,

    BSF_EVENT_FORCE_32_BIT = 0xffffffffU
};


/// Input and output structure for EDF sampling data.
struct Edf_sample_data
{
    // Input fields
    tct_float3      xi;             ///< pseudo-random sample number

    // Output fields
    tct_float3      k1;             /// < outgoing direction
    float           pdf;            /// < pdf (non-projected hemisphere)
    tct_float3      edf_over_pdf;   /// < edf * dot(normal,k1) / pdf
    Edf_event_type  event_type;
};

/// Input and output structure for EDF evaluation data.
struct Edf_evaluate_data
{
    // Input fields
    tct_float3      k1;            ///< outgoing direction

    // Output fields
    float           cos;            ///< dot(normal, k1)
    tct_float3      edf;            ///< edf
    float           pdf;            ///< pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_pdf_data
{
    // Input fields
    tct_float3      k1;             ///< outgoing direction

    // Output fields
    float           pdf;            ///< pdf (non-projected hemisphere)
};


/// Signature of the initialization function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu(),
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// This function updates the normal field of the shading state with the result of
/// \c "geometry.normal" and, if the \c "num_texture_results" backend option has been set to
/// non-zero, fills the text_results fields of the state.
///
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_init_function)(
    Shading_state_material *state,
    Resource_data const    *res_data,
    void const             *exception_state,
    char const             *arg_block_data);

/// Signature of the importance sampling function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu() and
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_sample_function)(
    Edf_sample_data             *data,
    Shading_state_material const *state,
    Resource_data const          *res_data,
    void const                   *exception_state,
    char const                   *arg_block_data);

/// Signature of the evaluation function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu() and
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_evaluate_function)(
    Edf_evaluate_data           *data,
    Shading_state_material const *state,
    Resource_data const          *res_data,
    void const                   *exception_state,
    char const                   *arg_block_data);

/// Signature of the probability density function for material distribution functions created via
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_cpu() and
/// #mi::mdl::ICode_generator_jit::compile_distribution_function_gpu() and
/// #mi::mdl::ILink_unit::add() for distribution functions.
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_pdf_function)(
    Edf_pdf_data                *data,
    Shading_state_material const *state,
    Resource_data const          *res_data,
    void const                   *exception_state,
    char const                   *arg_block_data);

}  // mdl
}  // mi

#endif
