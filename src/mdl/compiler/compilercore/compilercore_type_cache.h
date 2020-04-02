/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_TYPE_CACHE_H
#define MDL_COMPILERCORE_TYPE_CACHE_H 1

#include <mi/mdl/mdl_types.h>
#include "compilercore_factories.h"
#include "compilercore_array_ref.h"

namespace mi {
namespace mdl {

class ISymbol;
class IDefinition;

/// A Type cache is a wrapper around a type factory that simplifies the type access.
class Type_cache {
public:
    typedef IType_factory::Function_parameter Function_parameter;
    typedef Array_ref<Function_parameter>     Function_parameters;

    Type_cache(Type_factory &tf)
        : error_type(tf.create_error())
        , incomplete_type(tf.create_incomplete())
        , bool_type(tf.create_bool())
        , bool2_type(tf.create_vector(bool_type, 2))
        , bool3_type(tf.create_vector(bool_type, 3))
        , bool4_type(tf.create_vector(bool_type, 4))
        , int_type(tf.create_int())
        , int2_type(tf.create_vector(int_type, 2))
        , int3_type(tf.create_vector(int_type, 3))
        , int4_type(tf.create_vector(int_type, 4))
        , float_type(tf.create_float())
        , float2_type(tf.create_vector(float_type, 2))
        , float3_type(tf.create_vector(float_type, 3))
        , float4_type(tf.create_vector(float_type, 4))
        , double_type(tf.create_double())
        , double2_type(tf.create_vector(double_type, 2))
        , double3_type(tf.create_vector(double_type, 3))
        , double4_type(tf.create_vector(double_type, 4))
        , float2x2_type(tf.create_matrix(float2_type, 2))
        , float2x3_type(tf.create_matrix(float3_type, 2))
        , float2x4_type(tf.create_matrix(float4_type, 2))
        , float3x2_type(tf.create_matrix(float2_type, 3))
        , float3x3_type(tf.create_matrix(float3_type, 3))
        , float3x4_type(tf.create_matrix(float4_type, 3))
        , float4x2_type(tf.create_matrix(float2_type, 4))
        , float4x3_type(tf.create_matrix(float3_type, 4))
        , float4x4_type(tf.create_matrix(float4_type, 4))
        , double2x2_type(tf.create_matrix(double2_type, 2))
        , double2x3_type(tf.create_matrix(double3_type, 2))
        , double2x4_type(tf.create_matrix(double4_type, 2))
        , double3x2_type(tf.create_matrix(double2_type, 3))
        , double3x3_type(tf.create_matrix(double3_type, 3))
        , double3x4_type(tf.create_matrix(double4_type, 3))
        , double4x2_type(tf.create_matrix(double2_type, 4))
        , double4x3_type(tf.create_matrix(double3_type, 4))
        , double4x4_type(tf.create_matrix(double4_type, 4))
        , string_type(tf.create_string())
        , light_profile_type(tf.create_light_profile())
        , color_type(tf.create_color())
        , bsdf_type(tf.create_bsdf())
        , hair_bsdf_type(tf.create_hair_bsdf())
        , edf_type(tf.create_edf())
        , vdf_type(tf.create_vdf())
        , texture_2d_type(tf.create_texture(IType_texture::TS_2D))
        , texture_3d_type(tf.create_texture(IType_texture::TS_3D))
        , texture_cube_type(tf.create_texture(IType_texture::TS_CUBE))
        , texture_ptex_type(tf.create_texture(IType_texture::TS_PTEX))
        , bsdf_measurement_type(tf.create_bsdf_measurement())
        , material_emission_type(tf.get_predefined_struct(IType_struct::SID_MATERIAL_EMISSION))
        , material_surface_type(tf.get_predefined_struct(IType_struct::SID_MATERIAL_SURFACE))
        , material_volume_type(tf.get_predefined_struct(IType_struct::SID_MATERIAL_VOLUME))
        , material_geometry_type(tf.get_predefined_struct(IType_struct::SID_MATERIAL_GEOMETRY))
        , material_type(tf.get_predefined_struct(IType_struct::SID_MATERIAL))
        , tex_gamma_mode_type(tf.get_predefined_enum(IType_enum::EID_TEX_GAMMA_MODE))
        , intensity_mode_type(tf.get_predefined_enum(IType_enum::EID_INTENSITY_MODE))
        , m_tf(tf)
    {}

    /// Return a decorated type by adding type modifier.
    ///
    /// \param type  the base type
    /// \param mod   modifiers to ass
    IType const *decorate_type(IType const *type, IType::Modifiers mod) {
        if (mod == IType::MK_NONE)
            return type;
        return m_tf.create_alias(type, NULL, mod);
    }

    /// Create a new type function type instance.
    ///
    /// \param return_type   The return type of the function.
    /// \param parameters    The parameters of the function.
    IType_function const *create_function(
        IType const               *return_type,
        Function_parameters const &parameters)
    {
        return m_tf.create_function(return_type, parameters.data(), parameters.size());
    }

    /// Create a new type struct instance.
    ///
    /// \param name The name of the struct.
    IType_struct *create_struct(ISymbol const *name) {
        return m_tf.create_struct(name);
    }

    /// Create a new type enum instance.
    /// \param name The name of the enum.
    IType_enum *create_enum(ISymbol const *name) {
        return m_tf.create_enum(name);
    }

    /// Create a new type alias instance.
    ///
    /// \param type       The aliased type.
    /// \param name       The alias name.
    /// \param modifiers  The type modifiers.
    IType const *create_alias(
        IType const      *type,
        ISymbol const    *name,
        IType::Modifiers modifiers)
    {
        return m_tf.create_alias(type, name, modifiers);
    }

    /// Create a new deferred sized array type instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The deferred array size.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    IType const *create_array(
        IType const            *element_type,
        IType_array_size const *size)
    {
        return m_tf.create_array(element_type, size);
    }

    /// Create a new immediate array type instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The immediate size of the array.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    IType const *create_array(
        IType const *element_type,
        size_t      size)
    {
        return m_tf.create_array(element_type, size);
    }

    /// Get the array size for a given absolute deferred array length name.
    ///
    /// \param abs_name       The absolute name of the array size.
    /// \param sym            The symbol of the deferred array size.
    IType_array_size const *get_array_size(
        ISymbol const *abs_name,
        ISymbol const *sym)
    {
        return m_tf.get_array_size(abs_name, sym);
    }

public:
    IType_error const            * const error_type;
    IType_incomplete const       * const incomplete_type;
    IType_bool const             * const bool_type;
    IType_vector const           * const bool2_type;
    IType_vector const           * const bool3_type;
    IType_vector const           * const bool4_type;
    IType_int const              * const int_type;
    IType_vector const           * const int2_type;
    IType_vector const           * const int3_type;
    IType_vector const           * const int4_type;
    IType_float const            * const float_type;
    IType_vector const           * const float2_type;
    IType_vector const           * const float3_type;
    IType_vector const           * const float4_type;
    IType_double const           * const double_type;
    IType_vector const           * const double2_type;
    IType_vector const           * const double3_type;
    IType_vector const           * const double4_type;
    IType_matrix const           * const float2x2_type;
    IType_matrix const           * const float2x3_type;
    IType_matrix const           * const float2x4_type;
    IType_matrix const           * const float3x2_type;
    IType_matrix const           * const float3x3_type;
    IType_matrix const           * const float3x4_type;
    IType_matrix const           * const float4x2_type;
    IType_matrix const           * const float4x3_type;
    IType_matrix const           * const float4x4_type;
    IType_matrix const           * const double2x2_type;
    IType_matrix const           * const double2x3_type;
    IType_matrix const           * const double2x4_type;
    IType_matrix const           * const double3x2_type;
    IType_matrix const           * const double3x3_type;
    IType_matrix const           * const double3x4_type;
    IType_matrix const           * const double4x2_type;
    IType_matrix const           * const double4x3_type;
    IType_matrix const           * const double4x4_type;
    IType_string const           * const string_type;
    IType_light_profile const    * const light_profile_type;
    IType_color const            * const color_type;
    IType_bsdf const             * const bsdf_type;
    IType_hair_bsdf const        * const hair_bsdf_type;
    IType_edf const              * const edf_type;
    IType_vdf const              * const vdf_type;
    IType_texture const          * const texture_2d_type;
    IType_texture const          * const texture_3d_type;
    IType_texture const          * const texture_cube_type;
    IType_texture const          * const texture_ptex_type;
    IType_bsdf_measurement const * const bsdf_measurement_type;
    IType_struct const           * const material_emission_type;
    IType_struct const           * const material_surface_type;
    IType_struct const           * const material_volume_type;
    IType_struct const           * const material_geometry_type;
    IType_struct const           * const material_type;
    IType_enum const             * const tex_gamma_mode_type;
    IType_enum const             * const intensity_mode_type;

private:
    Type_factory &m_tf;
};


}  // mdl
}  // mi

#endif
