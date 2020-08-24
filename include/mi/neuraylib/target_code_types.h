/***************************************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file    mi/neuraylib/target_code_types.h
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

/// A float.
typedef float      tct_float;

/// A float2.
typedef float2     tct_float2;

/// A float3.
typedef float3     tct_float3;

/// A float4.
typedef float4     tct_float4;

/// An int.
typedef int        tct_int;

/// An unsigned int.
typedef unsigned   tct_uint;

#else

#include <mi/neuraylib/typedefs.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_compiler
@{
*/

/// A float.
typedef float                  tct_float;

/// A float2.
typedef mi::Float32_2_struct   tct_float2;

/// A float3.
typedef mi::Float32_3_struct   tct_float3;

/// A float4.
typedef mi::Float32_4_struct   tct_float4;

/// An int.
typedef mi::Sint32             tct_int;

/// An unsigned int.
typedef mi::Uint32             tct_uint;

#endif

/// A bool.
typedef bool  tct_bool;


/// A template struct with derivatives.
template<typename T>
struct tct_deriv
{
    T val, dx, dy;
};

/// Helper traits struct to switch between derivative and non-derivative types.
template<bool with_derivatives>
struct tct_traits;

template<>
struct tct_traits<false>
{
    typedef tct_float        tct_derivable_float;
    typedef tct_float2       tct_derivable_float2;
    typedef tct_float3       tct_derivable_float3;
    typedef tct_float4       tct_derivable_float4;
    typedef tct_float const  tct_coord2_type[2];
};

template<>
struct tct_traits<true>
{
    typedef tct_deriv<tct_float>          tct_derivable_float;
    typedef tct_deriv<tct_float2>         tct_derivable_float2;
    typedef tct_deriv<tct_float3>         tct_derivable_float3;
    typedef tct_deriv<tct_float4>         tct_derivable_float4;
    typedef tct_derivable_float2 const *  tct_coord2_type;
};

/// A float with derivatives.
typedef tct_traits<true>::tct_derivable_float  tct_deriv_float;

/// A float2 with derivatives.
typedef tct_traits<true>::tct_derivable_float2 tct_deriv_float2;

/// A float3 with derivatives.
typedef tct_traits<true>::tct_derivable_float3 tct_deriv_float3;

/// A float4 with derivatives.
typedef tct_traits<true>::tct_derivable_float4 tct_deriv_float4;

/// A float[2] with derivatives (needed to avoid problems with wrong alignment).
typedef tct_deriv<float[2]> tct_deriv_arr_float_2;

/// A float[3] with derivatives (needed to avoid problems with wrong alignment).
typedef tct_deriv<float[3]> tct_deriv_arr_float_3;

/// A float[4] with derivatives (needed to avoid problems with wrong alignment).
typedef tct_deriv<float[4]> tct_deriv_arr_float_4;


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

    /// A pointer to a read-only data segment.
    /// For "PTX", "LLVM-IR" and "native" JIT backend:
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
    char const           *ro_data_segment;
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
template<bool with_derivatives = false>
struct Shading_state_material_impl {
    typedef tct_traits<with_derivatives> traits;

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
    typename traits::tct_derivable_float3 position;

    /// The result of state::animation_time().
    /// It represents the time of the current sample in seconds.
    tct_float             animation_time;

    /// An array containing the results of state::texture_coordinate(i).
    /// The i-th entry represents the texture coordinates of the i-th texture space at the
    /// current position.
    typename traits::tct_derivable_float3 const *text_coords;

    /// An array containing the results of state::texture_tangent_u(i).
    /// The i-th entry represents the texture tangent vector of the i-th texture space at the
    /// current position, which points in the direction of the projection of the tangent to the
    /// positive u axis of this texture space onto the plane defined by the original
    /// surface normal.
    tct_float3 const     *tangent_u;

    /// An array containing the results of state::texture_tangent_v(i).
    /// The i-th entry represents the texture bitangent vector of the i-th texture space at the
    /// current position, which points in the general direction of the positive v axis of this
    /// texture space, but is orthogonal to both the original surface normal and the tangent
    /// of this texture space.
    tct_float3 const     *tangent_v;

    /// The texture results lookup table.
    /// Values will be modified by BSDF init functions to avoid duplicate texture fetches
    /// and duplicate calculation of values.
    /// This field is only relevant for code generated with
    /// #mi::neuraylib::IMdl_backend::translate_material_df() or
    /// #mi::neuraylib::ILink_unit::add_material_df(). In other cases this may be NULL.
    tct_float4           *text_results;

    /// A pointer to a read-only data segment.
    /// For "PTX", "LLVM-IR" and "native" JIT backend:
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
    char const           *ro_data_segment;

    /// A 4x4 transformation matrix in row-major order transforming from world to object
    /// coordinates.
    /// The last row is always implied to be (0, 0, 0, 1) and does not have to be provided.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    tct_float4 const     *world_to_object;

    /// A 4x4 transformation matrix in row-major order transforming from object to world
    /// coordinates.
    /// The last row is always implied to be (0, 0, 0, 1) and does not have to be provided.
    /// It is used by the state::transform_*() methods.
    /// This field is only used if the uniform state is included.
    tct_float4 const     *object_to_world;

    /// The result of state::object_id().
    /// It is an application-specific identifier of the hit object as provided in a scene.
    /// It can be used to make instanced objects look different in spite of the same used material.
    /// This field is only used if the uniform state is included.
    tct_int               object_id;

    /// The result of state::meters_per_scene_unit().
    /// The field is only used if the \c "fold_meters_per_scene_unit" option is set to false.
    /// Otherwise, the value of the \c "meters_per_scene_unit" option will be used in the code.
    tct_float             meters_per_scene_unit;
};

/// The MDL material state structure.
typedef struct Shading_state_material_impl<false> Shading_state_material;

/// The MDL material state structure with derivatives for the texture coordinates.
typedef struct Shading_state_material_impl<true> Shading_state_material_with_derivs;


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

/// MBSDFs can consist of two parts, which can be selected using this enumeration.
enum Mbsdf_part
{
    /// the bidirectional reflection distribution function (BRDF)
    MBSDF_DATA_REFLECTION = 0,

    /// the bidirectional transmission distribution function (BTDF)
    MBSDF_DATA_TRANSMISSION = 1
};


// Forward declaration of texture handler structure.
struct Texture_handler_base;


/// The runtime for bitmap texture access for the generated target code
/// can optionally be implemented in form of a vtable as specified by this structure.
template<bool with_derivatives = false>
struct Texture_handler_vtable_impl {
    typedef tct_traits<with_derivatives> traits;

    /// Implementation of \c tex::lookup_float4() for a texture_2d texture.
    void (*m_tex_lookup_float4_2d)(
        tct_float                        result[4],
        Texture_handler_base const       *self,
        tct_uint                         texture_idx,
        typename traits::tct_coord2_type coord,
        Tex_wrap_mode                    wrap_u,
        Tex_wrap_mode                    wrap_v,
        tct_float const                  crop_u[2],
        tct_float const                  crop_v[2]);

    /// Implementation of \c tex::lookup_float3() for a texture_2d texture.
    void (*m_tex_lookup_float3_2d)(
        tct_float                        result[3],
        Texture_handler_base const       *self,
        tct_uint                         texture_idx,
        typename traits::tct_coord2_type coord,
        Tex_wrap_mode                    wrap_u,
        Tex_wrap_mode                    wrap_v,
        tct_float const                  crop_u[2],
        tct_float const                  crop_v[2]);

    /// Implementation of \c tex::texel_float4() for a texture_2d texture.
    void (*m_tex_texel_float4_2d)(
        tct_float                  result[4],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_int const              coord[2],
        tct_int const              uv_tile[2]);

    /// Implementation of \c tex::lookup_float4() for a texture_3d texture.
    void (*m_tex_lookup_float4_3d)(
        tct_float                  result[4],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_float const            coord[3],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        Tex_wrap_mode              wrap_w,
        tct_float const            crop_u[2],
        tct_float const            crop_v[2],
        tct_float const            crop_w[2]);

    /// Implementation of \c tex::lookup_float3() for a texture_3d texture.
    void (*m_tex_lookup_float3_3d)(
        tct_float                  result[3],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_float const            coord[3],
        Tex_wrap_mode              wrap_u,
        Tex_wrap_mode              wrap_v,
        Tex_wrap_mode              wrap_w,
        tct_float const            crop_u[2],
        tct_float const            crop_v[2],
        tct_float const            crop_w[2]);

    /// Implementation of \c tex::texel_float4() for a texture_3d texture.
    void (*m_tex_texel_float4_3d)(
        tct_float                  result[4],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_int const              coord[3]);

    /// Implementation of \c tex::lookup_float4() for a texture_cube texture.
    void (*m_tex_lookup_float4_cube)(
        tct_float                  result[4],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_float const            coord[3]);

    /// Implementation of \c tex::lookup_float3() for a texture_cube texture.
    void (*m_tex_lookup_float3_cube)(
        tct_float                  result[3],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_float const            coord[3]);

    /// Implementation of \c resolution_2d() function needed by generated code,
    /// which retrieves the width and height of the given texture.
    /// Will only be called for texture_2d and texture_cube textures.
    void (*m_tex_resolution_2d)(
        tct_int                    result[2],
        Texture_handler_base const *self,
        tct_uint                   texture_idx,
        tct_int const              uv_tile[2]);

    /// Implementation of \c resolution_3d() function needed by generated code,
    /// which retrieves the width, height and depth of the given texture.
    /// Will only be called for texture_3d textures.
    void (*m_tex_resolution_3d)(
        tct_int                    result[3],
        Texture_handler_base const *self,
        tct_uint                   texture_idx);

    /// Implementation of \c texture_isvalid() for any texture type.
    tct_bool (*m_tex_texture_isvalid)(
        Texture_handler_base const *self,
        tct_uint                   texture_idx);

    /// Implementation of \c light_profile_power() for a light profile.
    tct_float (*m_df_light_profile_power)(
        Texture_handler_base const *self,
        tct_uint                   light_profile_index);

    /// Implementation of \c light_profile_maximum() for a light profile.
    tct_float (*m_df_light_profile_maximum)(
        Texture_handler_base const *self,
        tct_uint                   light_profile_index);

    /// Implementation of \c light_profile_isvalid() for a light profile.
    tct_bool (*m_df_light_profile_isvalid)(
        Texture_handler_base const *self,
        tct_uint                   light_profile_index);

    /// Implementation of \c light_profile_evaluate() for a light profile.
    tct_float (*m_df_light_profile_evaluate)(
        Texture_handler_base const *self,
        tct_uint                   light_profile_index,
        tct_float const            theta_phi[2]);       //!< theta in [0, pi/2] and phi in [-pi, pi]

    /// Implementation of \c light_profile_sample() for a light profile.
    void (*m_df_light_profile_sample)(
        tct_float                  result[3],           /*!< output: theta in [0, pi/2],
                                                             phi in [-pi, pi], and pdf */
        Texture_handler_base const *self,
        tct_uint                   light_profile_index,
        tct_float const            xi[3]);

    /// Implementation of \c light_profile_pdf() for a light profile.
    tct_float (*m_df_light_profile_pdf)(
        Texture_handler_base const *self,
        tct_uint                   light_profile_index,
        tct_float const            theta_phi[2]);       //!< theta in [0, pi/2] and phi in [-pi, pi]

    /// Implementation of \c bsdf_measurement_isvalid() for an MBSDF.
    tct_bool (*m_df_bsdf_measurement_isvalid)(
        Texture_handler_base const *self,
        tct_uint                   bsdf_measurement_index);

    /// Implementation of \c bsdf_measurement_resolution() function needed by generated code,
    /// which retrieves the angular and chromatic resolution of the given MBSDF.
    /// The returned triple consists of: number of equi-spaced steps of theta_i and theta_o,
    /// number of equi-spaced steps of phi, and number of color channels (1 or 3).
    void (*m_df_bsdf_measurement_resolution)(
        tct_uint                    result[3],
        Texture_handler_base const  *self,
        tct_uint                    bsdf_measurement_index,
        Mbsdf_part                  part);              //!< reflection or transmission

    /// Implementation of \c bsdf_measurement_evaluate() for an MBSDF.
    void (*m_df_bsdf_measurement_evaluate)(
        tct_float                   result[3],
        Texture_handler_base const  *self,
        tct_uint                    bsdf_measurement_index,
        tct_float const             theta_phi_in[2],    //!< theta in [0, pi/2] and phi in [-pi, pi]
        tct_float const             theta_phi_out[2],   //!< theta in [0, pi/2] and phi in [-pi, pi]
        Mbsdf_part                  part);              //!< reflection or transmission

    /// Implementation of \c bsdf_measurement_sample() for an MBSDF.
    void (*m_df_bsdf_measurement_sample)(
        tct_float                   result[3],          /*!< output: theta in [0, pi/2],
                                                             phi in [-pi, pi], and pdf */
        Texture_handler_base const  *self,
        tct_uint                    bsdf_measurement_index,
        tct_float const             theta_phi_out[2],   //!< theta in [0, pi/2] and phi in [-pi, pi]
        tct_float const             xi[3],              //!< uniform random values
        Mbsdf_part                  part);              //!< reflection or transmission

    /// Implementation of \c bsdf_measurement_pdf() for an MBSDF.
    tct_float (*m_df_bsdf_measurement_pdf)(
        Texture_handler_base const  *self,
        tct_uint                    bsdf_measurement_index,
        tct_float const             theta_phi_in[2],    //!< theta in [0, pi/2] and phi in [-pi, pi]
        tct_float const             theta_phi_out[2],   //!< theta in [0, pi/2] and phi in [-pi, pi]
        Mbsdf_part                  part);              //!< reflection or transmission

    /// Implementation of \c bsdf_measurement_albedos() for an MBSDF.
    void (*m_df_bsdf_measurement_albedos)(
        tct_float                   result[4],          /*!< output: maximum (in case of colored)
                                                            albedo for reflection and transmission
                                                            [0] albedo refl. for theta_phi
                                                            [1] max albedo refl. global
                                                            [2] albedo trans. for theta_phi
                                                            [3] max albedo trans. global */
        Texture_handler_base const  *self,
        tct_uint                    bsdf_measurement_index,
        tct_float const             theta_phi[2]);      //!< theta in [0, pi/2] and phi in [-pi, pi]

    /// Implementation of \c scene_data_isvalid().
    tct_bool (*m_scene_data_isvalid)(
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id);

    /// Implementation of \c scene_data_lookup_float().
    tct_float (*m_scene_data_lookup_float)(
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_float                              default_value,
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_float2().
    void (*m_scene_data_lookup_float2)(
        tct_float                              result[2],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_float const                        default_value[2],
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_float3().
    void (*m_scene_data_lookup_float3)(
        tct_float                              result[3],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_float const                        default_value[3],
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_float4().
    void (*m_scene_data_lookup_float4)(
        tct_float                              result[4],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_float const                        default_value[4],
        tct_bool                               uniform_lookup);

    /// Implementation of \c scene_data_lookup_int().
    tct_int (*m_scene_data_lookup_int)(
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_int                                default_value,
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_int2().
    void (*m_scene_data_lookup_int2)(
        tct_int                                result[2],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_int const                          default_value[2],
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_int3().
    void (*m_scene_data_lookup_int3)(
        tct_int                                result[3],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_int const                          default_value[3],
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_int4().
    void (*m_scene_data_lookup_int4)(
        tct_int                                result[4],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_int const                          default_value[4],
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_color().
    void (*m_scene_data_lookup_color)(
        tct_float                              result[3],
        Texture_handler_base const            *self_base,
        Shading_state_material                *state,
        tct_uint                               scene_data_id,
        tct_float const                        default_value[3],
        tct_bool                               uniform_lookup);

    //
    // The following functions are only used in the derivative variant,
    // and can be nullptr in the non-derivative variant
    //

    /// Implementation of \c scene_data_lookup_float() with derivatives.
    void (*m_scene_data_lookup_deriv_float)(
        tct_deriv_float                       *result,
        Texture_handler_base const            *self_base,
        Shading_state_material_with_derivs     *state,
        tct_uint                               scene_data_id,
        tct_deriv_float const                 *default_value,
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_float2() with derivatives.
    void (*m_scene_data_lookup_deriv_float2)(
        tct_deriv_arr_float_2                 *result,
        Texture_handler_base const            *self_base,
        Shading_state_material_with_derivs    *state,
        tct_uint                               scene_data_id,
        tct_deriv_arr_float_2 const           *default_value,
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_float3() with derivatives.
    void (*m_scene_data_lookup_deriv_float3)(
        tct_deriv_arr_float_3                 *result,
        Texture_handler_base const            *self_base,
        Shading_state_material_with_derivs    *state,
        tct_uint                               scene_data_id,
        tct_deriv_arr_float_3 const           *default_value,
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_float4() with derivatives.
    void (*m_scene_data_lookup_deriv_float4)(
        tct_deriv_arr_float_4                 *result,
        Texture_handler_base const            *self_base,
        Shading_state_material_with_derivs    *state,
        tct_uint                               scene_data_id,
        tct_deriv_arr_float_4 const           *default_value,
        tct_bool                               uniform_lookup);

    /// Implementation of scene_data_lookup_color() with derivatives.
    void (*m_scene_data_lookup_deriv_color)(
        tct_deriv_arr_float_3                 *result,
        Texture_handler_base const            *self_base,
        Shading_state_material_with_derivs    *state,
        tct_uint                               scene_data_id,
        tct_deriv_arr_float_3 const           *default_value,
        tct_bool                               uniform_lookup);
};

/// The texture handler vtable struct.
typedef Texture_handler_vtable_impl<false> Texture_handler_vtable;

/// The texture handler vtable struct with derivatives for the texture coordinates.
typedef Texture_handler_vtable_impl<true> Texture_handler_deriv_vtable;

/// The texture handler structure that is passed to the texturing functions.
/// A user can derive from this structure and add custom fields as required by the texturing
/// function implementations.
struct Texture_handler_base {
    /// In vtable-mode, the vtable field is used to call the texturing functions.
    /// Otherwise, this field may be NULL.
    Texture_handler_vtable const  *vtable;
};

/// The texture handler structure that is passed to the texturing functions with derivative support.
/// A user can derive from this structure and add custom fields as required by the texturing
/// function implementations.
struct Texture_handler_deriv_base {
    /// In vtable-mode, the vtable field is used to call the texturing functions.
    /// Otherwise, this field may be NULL.
    Texture_handler_deriv_vtable const  *vtable;
};


/// The data structure providing access to resources for generated code.
struct Resource_data {
    void const                  *shared_data;      ///< currently unused, should be NULL
    Texture_handler_base const  *texture_handler;  ///< will be provided as "self" parameter to
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
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_float3       k2;             ///< output: incoming direction
    tct_float4       xi;             ///< input: pseudo-random sample numbers
    tct_float        pdf;            ///< output: pdf (non-projected hemisphere)
    tct_float3       bsdf_over_pdf;  ///< output: bsdf * dot(normal, k2) / pdf
    Bsdf_event_type  event_type;     ///< output: the type of event for the generated sample
    tct_int          handle;         ///< output: handle of the sampled elemental BSDF (lobe)
};

/// Type of Bsdf_evaluate_data variants, depending on the backend and its configuration.
enum Df_handle_slot_mode
{
    DF_HSM_POINTER = -2,    ///< Uses renderer defined buffers; not supported by all backends
    DF_HSM_NONE    = -1,    ///< No slots, handles are ignored completely
    DF_HSM_FIXED_1 =  1,    ///< fixed size array for processing 1 handle at a time
    DF_HSM_FIXED_2 =  2,    ///< fixed size array for processing 2 handle at a time
    DF_HSM_FIXED_4 =  4,    ///< fixed size array for processing 4 handle at a time
    DF_HSM_FIXED_8 =  8,    ///< fixed size array for processing 8 handle at a time
};

/// Input and output structure for BSDF evaluation data.
struct Bsdf_evaluate_data_base {};

template<Df_handle_slot_mode N>
struct Bsdf_evaluate_data : public Bsdf_evaluate_data_base
{
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_float3       k2;             ///< input: incoming direction
    tct_int          handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                     ///  DF_HANDLE_SLOTS handles, calling 'evaluate' multiple times
    tct_float3       bsdf_diffuse[static_cast<size_t>(N)]; ///< output: (diffuse part of the)
                                                           ///  bsdf * dot(normal, k2)
    tct_float3       bsdf_glossy[static_cast<size_t>(N)];  ///< output: (glossy part of the)
                                                           ///  bsdf * dot(normal, k2)
    tct_float        pdf;            ///< output: pdf (non-projected hemisphere)
};

template<>
struct Bsdf_evaluate_data<DF_HSM_POINTER> : public Bsdf_evaluate_data_base
{
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_float3       k2;             ///< input: incoming direction
    tct_int          handle_offset;  ///< input: handle offset to allow the evaluation of many 
                                     ///  handles using in multiple steps
    tct_int          handle_count;   ///< input: number of elements of 'bsdf_diffuse', 'bsdf_glossy'
    tct_float3*      bsdf_diffuse;   ///< output: (diffuse part of the) bsdf * dot(normal, k2)
    tct_float3*      bsdf_glossy;    ///< output: (glossy part of the) bsdf * dot(normal, k2)
    tct_float        pdf;            ///< output: pdf (non-projected hemisphere)
};

template<>
struct Bsdf_evaluate_data<DF_HSM_NONE> : public Bsdf_evaluate_data_base
{
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_float3       k2;             ///< input: incoming direction
    tct_float3       bsdf_diffuse;   ///< output: (diffuse part of the) bsdf * dot(normal, k2)
    tct_float3       bsdf_glossy;    ///< output: (glossy part of the) bsdf * dot(normal, k2)
    tct_float        pdf;            ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF PDF calculation data.
struct Bsdf_pdf_data {
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_float3       k2;             ///< input: incoming direction
    tct_float        pdf;            ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for BSDF auxiliary calculation data.
struct Bsdf_auxiliary_data_base {};

template<Df_handle_slot_mode N>
struct Bsdf_auxiliary_data : public Bsdf_auxiliary_data_base
{
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_int          handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                     ///  DF_HANDLE_SLOTS handles, calling 'auxiliary' multiple times
    tct_float3       albedo[static_cast<size_t>(N)];    ///< output: albedo
    tct_float3       normal[static_cast<size_t>(N)];    ///< output: normal
};

template<>
struct Bsdf_auxiliary_data<DF_HSM_POINTER> : public Bsdf_auxiliary_data_base
{
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction
    
    tct_int          handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                     ///  DF_HANDLE_SLOTS handles, calling 'auxiliary' multiple times
    tct_int          handle_count;   ///< input: number of elements of 'albedo' and 'normal'
    tct_float3*      albedo;         ///< output: albedo
    tct_float3*      normal;         ///< output: normal
};

template<>
struct Bsdf_auxiliary_data<DF_HSM_NONE> : public Bsdf_auxiliary_data_base
{
    tct_float3       ior1;           ///< mutual input: IOR current medium
    tct_float3       ior2;           ///< mutual input: IOR other side
    tct_float3       k1;             ///< mutual input: outgoing direction

    tct_float3       albedo;         ///< output: albedo
    tct_float3       normal;         ///< output: normal
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
typedef void (Environment_function)(
    void                             *result,
    Shading_state_environment const  *state,
    Resource_data const              *res_data,
    void const                       *exception_state,
    char const                       *arg_block_data);


/// Signature of material expression functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_expression() and
/// #mi::neuraylib::ILink_unit::add_material_expression().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Material_expr_function)(
    void                          *result,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of material expression functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_expression() and
/// #mi::neuraylib::ILink_unit::add_material_expression().
///
/// \param result           pointer to the result buffer which must be large enough for the result
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Material_expr_function_with_derivs)(
    void                                      *result,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);


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
typedef void (Bsdf_init_function)(
    Shading_state_material  *state,
    Resource_data const     *res_data,
    void const              *exception_state,
    char const              *arg_block_data);


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
typedef void (Bsdf_init_function_with_derivs)(
    Shading_state_material_with_derivs  *state,
    Resource_data const                 *res_data,
    void const                          *exception_state,
    char const                          *arg_block_data);


/// Signature of the importance sampling function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_sample_function)(
    Bsdf_sample_data              *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the importance sampling function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_sample_function_with_derivs)(
    Bsdf_sample_data                          *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);


/// Signature of the evaluation function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_evaluate_function)(
    Bsdf_evaluate_data_base               *data,
    Shading_state_material const          *state,
    Resource_data const                   *res_data,
    void const                            *exception_state,
    char const                            *arg_block_data);


/// Signature of the evaluation function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_evaluate_function_with_derivs)(
    Bsdf_evaluate_data_base                     *data,
    Shading_state_material_with_derivs const    *state,
    Resource_data const                         *res_data,
    void const                                  *exception_state,
    char const                                  *arg_block_data);


/// Signature of the probability density function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_pdf_function)(
    Bsdf_pdf_data                 *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the probability density function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_pdf_function_with_derivs)(
    Bsdf_pdf_data                             *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);

/// Signature of the auxiliary function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_auxiliary_function)(
    Bsdf_auxiliary_data_base      *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the auxiliary function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Bsdf_auxiliary_function_with_derivs)(
    Bsdf_auxiliary_data_base                  *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);

/// The type of events created by EDF importance sampling.
enum Edf_event_type
{
    EDF_EVENT_NONE = 0,
    EDF_EVENT_EMISSION = 1,

    EDF_EVENT_FORCE_32_BIT = 0xffffffffU
};


/// Input and output structure for EDF sampling data.
struct Edf_sample_data
{
    tct_float4      xi;             ///< input: pseudo-random sample number
    tct_float3      k1;             ///< output: outgoing direction
    tct_float       pdf;            ///< output: pdf (non-projected hemisphere)
    tct_float3      edf_over_pdf;   ///< output: edf * dot(normal,k1) / pdf
    Edf_event_type  event_type;     ///< output: the type of event for the generated sample
    tct_int         handle;         ///< output: handle of the sampled elemental EDF (lobe)
};

/// Input and output structure for EDF evaluation data.
struct Edf_evaluate_data_base {};

template<Df_handle_slot_mode N>
struct Edf_evaluate_data : public Edf_evaluate_data_base
{
    tct_float3      k1;             ///< input: outgoing direction
    tct_int         handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                    ///  DF_HANDLE_SLOTS handles, calling 'evaluate' multiple times
    tct_float       cos;                            ///< output: dot(normal, k1)
    tct_float3      edf[static_cast<size_t>(N)];    ///< output: edf
    tct_float       pdf;                            ///< output: pdf (non-projected hemisphere)
};

template<>
struct Edf_evaluate_data<DF_HSM_POINTER> : public Edf_evaluate_data_base
{
    tct_float3      k1;             ///< input: outgoing direction
    tct_int         handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                    ///  DF_HANDLE_SLOTS handles, calling 'evaluate' multiple times
    tct_int         handle_count;   ///< input: number of elements of 'edf'
    tct_float       cos;            ///< output: dot(normal, k1)
    tct_float3*     edf;            ///< output: edf
    tct_float       pdf;            ///< output: pdf (non-projected hemisphere)
};

template<>
struct Edf_evaluate_data<DF_HSM_NONE> : public Edf_evaluate_data_base
{
    tct_float3      k1;             ///< input: outgoing direction
    tct_float       cos;            ///< output: dot(normal, k1)
    tct_float3      edf;            ///< output: edf
    tct_float       pdf;            ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for EDF PDF calculation data.
struct Edf_pdf_data
{
    tct_float3      k1;             ///< input: outgoing direction
    tct_float       pdf;            ///< output: pdf (non-projected hemisphere)
};

/// Input and output structure for EDF auxiliary calculation data.
struct Edf_auxiliary_data_base {};

template<Df_handle_slot_mode N>
struct Edf_auxiliary_data : public Edf_auxiliary_data_base
{
    tct_float3      k1;             ///< input: outgoing direction
    tct_int         handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                    ///  DF_HANDLE_SLOTS handles, calling 'auxiliary' multiple times
    // reserved for future use
};

template<>
struct Edf_auxiliary_data<DF_HSM_POINTER> : public Edf_auxiliary_data_base
{
    tct_float3      k1;             ///< input: outgoing direction
    tct_int         handle_offset;  ///< input: handle offset to allow the evaluation of more then  
                                    ///  DF_HANDLE_SLOTS handles, calling 'auxiliary' multiple times
    tct_int         handle_count;   ///< input: number of elements of 'edf'

    // reserved for future use
};

template<>
struct Edf_auxiliary_data<DF_HSM_NONE> : public Edf_auxiliary_data_base
{
    tct_float3      k1;             ///< input: outgoing direction

    // reserved for future use
};

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
typedef void (Edf_init_function)(
    Shading_state_material  *state,
    Resource_data const     *res_data,
    void const              *exception_state,
    char const              *arg_block_data);


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
typedef void (Edf_init_function_with_derivs)(
    Shading_state_material_with_derivs  *state,
    Resource_data const                 *res_data,
    void const                          *exception_state,
    char const                          *arg_block_data);


/// Signature of the importance sampling function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_sample_function)(
    Edf_sample_data               *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the importance sampling function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_sample_function_with_derivs)(
    Edf_sample_data                           *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);


/// Signature of the evaluation function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_evaluate_function)(
    Edf_evaluate_data_base       *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the evaluation function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_evaluate_function_with_derivs)(
    Edf_evaluate_data_base                    *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);


/// Signature of the probability density function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_pdf_function)(
    Edf_pdf_data                  *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the probability density function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_pdf_function_with_derivs)(
    Edf_pdf_data                              *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);

/// Signature of the auxiliary function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_auxiliary_function)(
    Edf_auxiliary_data_base       *data,
    Shading_state_material const  *state,
    Resource_data const           *res_data,
    void const                    *exception_state,
    char const                    *arg_block_data);


/// Signature of the auxiliary function for material distribution functions created via
/// #mi::neuraylib::IMdl_backend::translate_material_df() and
/// #mi::neuraylib::ILink_unit::add_material_df().
///
/// \param data             the input and output structure
/// \param state            the shading state
/// \param res_data         the resources
/// \param exception_state  unused, should be NULL
/// \param arg_block_data   the target argument block data, if class compilation was used
typedef void (Edf_auxiliary_function_with_derivs)(
    Edf_auxiliary_data_base                   *data,
    Shading_state_material_with_derivs const  *state,
    Resource_data const                       *res_data,
    void const                                *exception_state,
    char const                                *arg_block_data);

/*@}*/ // end group mi_neuray_mdl_compiler

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_TARGET_CODE_TYPES_H
