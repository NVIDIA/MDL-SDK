/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_definitions.h
/// \brief Interfaces for the MDL definition table of modules
#ifndef MDL_DEFINITIONS_H
#define MDL_DEFINITIONS_H 1

#include <cstddef>

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_expressions.h>

namespace mi {
namespace mdl {

class IDeclaration;
class ISymbol;
class IType;
class IValue;
class Position;

/// A definition of an MDL entity.
///
/// A definition is a unique entry for every named MDL object inside the definition table.
/// Definitions are created by the compiler when a module is loaded or analyzed.
class IDefinition : public Interface_owned
{
public:
    /// Definition kinds.
    enum Kind {
        DK_ERROR,         ///< This is an error definition.
        DK_CONSTANT,      ///< This is a constant entity.
        DK_ENUM_VALUE,    ///< This is an enum value.
        DK_ANNOTATION,    ///< This is an annotation.
        DK_TYPE,          ///< This is a type.
        DK_FUNCTION,      ///< This is a function.
        DK_VARIABLE,      ///< This is a variable.
        DK_MEMBER,        ///< This is a field member.
        DK_CONSTRUCTOR,   ///< This is a constructor.
        DK_PARAMETER,     ///< This is a parameter.
        DK_ARRAY_SIZE,    ///< This is a constant array size.
        DK_OPERATOR,      ///< This is an operator.
        DK_NAMESPACE,     ///< This is a namespace.
    };

    /// Boolean properties of definitions.
    enum Property {
        DP_IS_OVERLOADED,       ///< True, if a function or constructor has an overload.
        DP_IS_UNIFORM,          ///< True, if this definition is uniform.
        DP_NOT_WRITTEN,         ///< True, if a variable or a parameter is not written.
        DP_IS_IMPORTED,         ///< True, if this definition was imported from another module.
        DP_NEED_REFERENCE,      ///< True, if this definition needs argument passing by reference.
        DP_ALLOW_INLINE,        ///< True, if it is legal to inline this function.
        DP_IS_EXPORTED,         ///< True, if this definition is exported.
        DP_IS_LOCAL_FUNCTION,   ///< True, if the definition is a local function.
        DP_IS_WRITTEN,          ///< True, if the entity belonging to this definition is written.
        DP_USES_STATE,          ///< True, if this function uses the state (either directly
                                ///  or by calling another function that uses the state).
        DP_USES_TEXTURES,       ///< True, if this function uses the texture functions (either
                                ///  directly or by calling another function that uses textures).
        DP_CAN_THROW_BOUNDS,    ///< True, if this function can throw a bounds exception.
        DP_CAN_THROW_DIVZERO,   ///< True, if this function can throw a division by zero exception.
        DP_IS_VARYING,          ///< True, if this function is varying.
        DP_READ_TEX_ATTR,       ///< True, if this function reads texture attributes (width etc.).
        DP_READ_LP_ATTR,        ///< True, if this function reads light profile attributes.
        DP_USES_VARYING_STATE,  ///< True, if this function uses the varying state (either directly
                                ///  or by calling another function that uses the varying state).
                                ///  Always a subset of DP_USES_STATE.
        DP_CONTAINS_DEBUG,      ///< True, if this function contains debug statements.
        DP_USES_OBJECT_ID,      ///< True, if this function may call state::object_id().
        DP_USES_TRANSFORM,      ///< True, if this function may call state::transform*().
        DP_USES_NORMAL,         ///< True, if this function may call state::normal().
        DP_IS_NATIVE,           ///< True, if this function was declared native.
        DP_IS_CONST_EXPR,       ///< True, if this function is declared as const_expr.
        DP_USES_DERIVATIVES,    ///< True, if this functions uses derivatives directly
        DP_USES_SCENE_DATA,     ///< True, if this function uses the scene data functions (either
                                ///  directly or by calling another function that uses scene data).
    };

    /// Built-in semantics.
    enum Semantics {
        DS_UNKNOWN = 0,                          ///< Unknown semantics.
        DS_COPY_CONSTRUCTOR,                     ///< This is a copy constructor.
        DS_CONV_CONSTRUCTOR,                     ///< This is a conversion constructor.
        DS_ELEM_CONSTRUCTOR,                     ///< This is a elemental constructor.
        DS_COLOR_SPECTRUM_CONSTRUCTOR,           ///< This is the color from spectrum constructor.
        DS_MATRIX_ELEM_CONSTRUCTOR,              ///< This is a matrix elemental constructor.
        DS_MATRIX_DIAG_CONSTRUCTOR,              ///< This is a matrix diagonal constructor.
        DS_INVALID_REF_CONSTRUCTOR,              ///< This is a invalid reference constructor.
        DS_DEFAULT_STRUCT_CONSTRUCTOR,           ///< This is a default constructor for a struct.
        DS_TEXTURE_CONSTRUCTOR,                  ///< This is a texture constructor.
        DS_CONV_OPERATOR,                        ///< This is a type conversion operator.

        // annotation semantics
        DS_ANNOTATION_FIRST = 0x0100,

        DS_INTRINSIC_ANNOTATION                  ///< This is the internal intrinsic() annotation.
            = DS_ANNOTATION_FIRST,
        DS_THROWS_ANNOTATION,                    ///< This is the internal throws() annotation.
        DS_SINCE_ANNOTATION,                     ///< This is the internal since() annotation.
        DS_REMOVED_ANNOTATION,                   ///< This is the internal removed() annotation.
        DS_CONST_EXPR_ANNOTATION,                ///< This is the internal const_expr() annotation.
        DS_DERIVABLE_ANNOTATION,                 ///< This is the internal derivable() annotation.
        DS_NATIVE_ANNOTATION,                    ///< This is the internal native() annotation.
        DS_EXPERIMENTAL_ANNOTATION,              ///< This is the internal experimental()
                                                 ///  annotation.
        DS_LITERAL_PARAM_ANNOTATION,             ///< This is the internal literal_param()
                                                 ///  annotation.

        DS_UNUSED_ANNOTATION,                    ///< This is the unused() annotation.
        DS_NOINLINE_ANNOTATION,                  ///< This is the noinline() annotation.
        DS_SOFT_RANGE_ANNOTATION,                ///< This is the soft_range() annotation.
        DS_HARD_RANGE_ANNOTATION,                ///< This is the hard_range() annotation.
        DS_HIDDEN_ANNOTATION,                    ///< This is the hidden() annotation.
        DS_DEPRECATED_ANNOTATION,                ///< This is the deprecated() annotation.
        DS_VERSION_NUMBER_ANNOTATION,            ///< This is the (old) version_number() annotation.
        DS_VERSION_ANNOTATION,                   ///< This is the version() annotation.
        DS_DEPENDENCY_ANNOTATION,                ///< This is the dependency() annotation.
        DS_UI_ORDER_ANNOTATION,                  ///< This is the ui_order() annotation.
        DS_USAGE_ANNOTATION,                     ///< This is the usage() annotation.
        DS_ENABLE_IF_ANNOTATION,                 ///< This is the enable_if() annotation.
        DS_THUMBNAIL_ANNOTATION,                 ///< This is the thumbnail() annotation.
        DS_DISPLAY_NAME_ANNOTATION,              ///< This is the display_name() annotation.
        DS_IN_GROUP_ANNOTATION,                  ///< This is the in_group() annotation.
        DS_DESCRIPTION_ANNOTATION,               ///< This is the description() annotation.
        DS_AUTHOR_ANNOTATION,                    ///< This is the author() annotation.
        DS_CONTRIBUTOR_ANNOTATION,               ///< This is the contributor() annotation.
        DS_COPYRIGHT_NOTICE_ANNOTATION,          ///< This is the copyright_notice() annotation.
        DS_CREATED_ANNOTATION,                   ///< This is the created() annotation.
        DS_MODIFIED_ANNOTATION,                  ///< This is the modified() annotation.
        DS_KEYWORDS_ANNOTATION,                  ///< This is the key_words() annotation.
        DS_ORIGIN_ANNOTATION,                    ///< This is the origin() annotation.

        DS_ANNOTATION_LAST = DS_ORIGIN_ANNOTATION,

        // operator semantics
        DS_OP_BASE = 0x0200,                     ///< Base offset for operator semantics.
        DS_OP_END =
            DS_OP_BASE +
            IExpression::OK_LAST,                ///< Last operator semantic.

        // math module intrinsics
        DS_INTRINSIC_MATH_FIRST = 0x0300,

        DS_INTRINSIC_MATH_ABS                    ///< The math::abs() intrinsic function.
            = DS_INTRINSIC_MATH_FIRST,
        DS_INTRINSIC_MATH_ACOS,                  ///< The math::acos() intrinsic function.
        DS_INTRINSIC_MATH_ALL,                   ///< The math::all() intrinsic function.
        DS_INTRINSIC_MATH_ANY,                   ///< The math::any() intrinsic function.
        DS_INTRINSIC_MATH_ASIN,                  ///< The math::asin() intrinsic function.
        DS_INTRINSIC_MATH_ATAN,                  ///< The math::atan() intrinsic function.
        DS_INTRINSIC_MATH_ATAN2,                 ///< The math::atan2() intrinsic function.
        DS_INTRINSIC_MATH_AVERAGE,               ///< The math::average() intrinsic function.
        DS_INTRINSIC_MATH_CEIL,                  ///< The math::ceil() intrinsic function.
        DS_INTRINSIC_MATH_CLAMP,                 ///< The math::clamp() intrinsic function.
        DS_INTRINSIC_MATH_COS,                   ///< The math::cos() intrinsic function.
        DS_INTRINSIC_MATH_CROSS,                 ///< The math::cross() intrinsic function.
        DS_INTRINSIC_MATH_DEGREES,               ///< The math::degrees() intrinsic function.
        DS_INTRINSIC_MATH_DISTANCE,              ///< The math::distance() intrinsic function.
        DS_INTRINSIC_MATH_DOT,                   ///< The math::dot() intrinsic function.
        DS_INTRINSIC_MATH_EVAL_AT_WAVELENGTH,    ///< The math::eval_at_wavelength()
                                                 ///  intrinsic function.
        DS_INTRINSIC_MATH_EXP,                   ///< The math::exp() intrinsic function.
        DS_INTRINSIC_MATH_EXP2,                  ///< The math::exp2() intrinsic function.
        DS_INTRINSIC_MATH_FLOOR,                 ///< The math::floor() intrinsic function.
        DS_INTRINSIC_MATH_FMOD,                  ///< The math::fmod() intrinsic function.
        DS_INTRINSIC_MATH_FRAC,                  ///< The math::frac() intrinsic function.
        DS_INTRINSIC_MATH_ISNAN,                 ///< The math::isnan() intrinsic function.
        DS_INTRINSIC_MATH_ISFINITE,              ///< The math::isfinite() intrinsic function.
        DS_INTRINSIC_MATH_LENGTH,                ///< The math::length() intrinsic function.
        DS_INTRINSIC_MATH_LERP,                  ///< The math::lerp() intrinsic function.
        DS_INTRINSIC_MATH_LOG,                   ///< The math::log() intrinsic function.
        DS_INTRINSIC_MATH_LOG2,                  ///< The math::log2() intrinsic function.
        DS_INTRINSIC_MATH_LOG10,                 ///< The math::log10() intrinsic function.
        DS_INTRINSIC_MATH_LUMINANCE,             ///< The math::luminance() intrinsic function.
        DS_INTRINSIC_MATH_MAX,                   ///< The math::max() intrinsic function.
        DS_INTRINSIC_MATH_MAX_VALUE,             ///< The math::max_value() intrinsic function.
        DS_INTRINSIC_MATH_MAX_VALUE_WAVELENGTH,  ///< The math::max_value_wavelength()
                                                 ///  intrinsic function.
        DS_INTRINSIC_MATH_MIN,                   ///< The math::min() intrinsic function.
        DS_INTRINSIC_MATH_MIN_VALUE,             ///< The math::min_value() intrinsic function.
        DS_INTRINSIC_MATH_MIN_VALUE_WAVELENGTH,  ///< The math::min_value_wavelength()
                                                 ///  intrinsic function.
        DS_INTRINSIC_MATH_MODF,                  ///< The math::modf() intrinsic function.
        DS_INTRINSIC_MATH_NORMALIZE,             ///< The math::normalize() intrinsic function.
        DS_INTRINSIC_MATH_POW,                   ///< The math::pow() intrinsic function.
        DS_INTRINSIC_MATH_RADIANS,               ///< The math::radians() intrinsic function.
        DS_INTRINSIC_MATH_ROUND,                 ///< The math::round() intrinsic function.
        DS_INTRINSIC_MATH_RSQRT,                 ///< The math::rsqrt() intrinsic function.
        DS_INTRINSIC_MATH_SATURATE,              ///< The math::saturate() intrinsic function.
        DS_INTRINSIC_MATH_SIGN,                  ///< The math::sign() intrinsic function.
        DS_INTRINSIC_MATH_SIN,                   ///< The math::sin() intrinsic function.
        DS_INTRINSIC_MATH_SINCOS,                ///< The math::sincos() intrinsic function.
        DS_INTRINSIC_MATH_SMOOTHSTEP,            ///< The math::smoothstep() intrinsic function.
        DS_INTRINSIC_MATH_SQRT,                  ///< The math::sqrt() intrinsic function.
        DS_INTRINSIC_MATH_STEP,                  ///< The math::step() intrinsic function.
        DS_INTRINSIC_MATH_TAN,                   ///< The math::tan() intrinsic function.
        DS_INTRINSIC_MATH_TRANSPOSE,             ///< The math::transpose() intrinsic function.
        DS_INTRINSIC_MATH_BLACKBODY,             ///< The math::blackbody() intrinsic function.
        DS_INTRINSIC_MATH_EMISSION_COLOR,        ///< The math::emission_color() intrinsic function.
        DS_INTRINSIC_MATH_DX,                    ///< The math::DX() intrinsic function.
        DS_INTRINSIC_MATH_DY,                    ///< The math::DY() intrinsic function.
        DS_INTRINSIC_MATH_LAST = DS_INTRINSIC_MATH_DY,

        // state module intrinsics
        DS_INTRINSIC_STATE_FIRST = 0x0400,

        /// The state::position() function.
        DS_INTRINSIC_STATE_POSITION = DS_INTRINSIC_STATE_FIRST,
        DS_INTRINSIC_STATE_NORMAL,               ///< The state::normal() function.
        DS_INTRINSIC_STATE_GEOMETRY_NORMAL,      ///< The state::geometry_normal() function.
        DS_INTRINSIC_STATE_MOTION,               ///< The state::motion() function.
        DS_INTRINSIC_STATE_TEXTURE_SPACE_MAX,    ///< The state::texture_space_max() function.
        DS_INTRINSIC_STATE_TEXTURE_COORDINATE,   ///< The state::texture_coordinate() function.
        DS_INTRINSIC_STATE_TEXTURE_TANGENT_U,    ///< The state::texture_tangent_u() function.
        DS_INTRINSIC_STATE_TEXTURE_TANGENT_V,    ///< The state::texture_tangent_v() function.
        DS_INTRINSIC_STATE_TANGENT_SPACE,        ///< The state::tangent_space() function.
        DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U,   ///< The state::geometry_tangent_u() function.
        DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V,   ///< The state::geometry_tangent_v() function.
        DS_INTRINSIC_STATE_DIRECTION,            ///< The state::direction() function.
        DS_INTRINSIC_STATE_ANIMATION_TIME,       ///< The state::animation_time() function.
        DS_INTRINSIC_STATE_WAVELENGTH_BASE,      ///< The state::wavelength_base() function.
        DS_INTRINSIC_STATE_TRANSFORM,            ///< The state::transform() function.
        DS_INTRINSIC_STATE_TRANSFORM_POINT,      ///< The state::transform_point() function.
        DS_INTRINSIC_STATE_TRANSFORM_VECTOR,     ///< The state::transform_vector() function.
        DS_INTRINSIC_STATE_TRANSFORM_NORMAL,     ///< The state::transform_normal() function.
        DS_INTRINSIC_STATE_TRANSFORM_SCALE,      ///< The state::transform_scale() function.
        DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL,///< The state::rounded_corner_normal() function.
        DS_INTRINSIC_STATE_METERS_PER_SCENE_UNIT,///< The state::meters_per_scene_unit() function.
        DS_INTRINSIC_STATE_SCENE_UNITS_PER_METER,///< The state::scene_units_per_meter() function.
        DS_INTRINSIC_STATE_OBJECT_ID,            ///< The state::object_id() function.
        DS_INTRINSIC_STATE_WAVELENGTH_MIN,       ///< The state::wavelength_min() function.
        DS_INTRINSIC_STATE_WAVELENGTH_MAX,       ///< The state::wavelength_max() function.
        DS_INTRINSIC_STATE_LAST = DS_INTRINSIC_STATE_WAVELENGTH_MAX,

        // tex module intrinsics
        DS_INTRINSIC_TEX_FIRST = 0x0500,

        /// The tex::width() function.
        DS_INTRINSIC_TEX_WIDTH = DS_INTRINSIC_TEX_FIRST,
        DS_INTRINSIC_TEX_HEIGHT,          ///< The tex::height() function.
        DS_INTRINSIC_TEX_DEPTH,           ///< The tex::depth() function.
        DS_INTRINSIC_TEX_LOOKUP_FLOAT,    ///< The tex::lookup_float() function.
        DS_INTRINSIC_TEX_LOOKUP_FLOAT2,   ///< The tex::lookup_float2() function.
        DS_INTRINSIC_TEX_LOOKUP_FLOAT3,   ///< The tex::lookup_float3() function.
        DS_INTRINSIC_TEX_LOOKUP_FLOAT4,   ///< The tex::lookup_float4() function.
        DS_INTRINSIC_TEX_LOOKUP_COLOR,    ///< The tex::lookup_color() function.
        DS_INTRINSIC_TEX_TEXEL_FLOAT,     ///< The tex::texel_float() function.
        DS_INTRINSIC_TEX_TEXEL_FLOAT2,    ///< The tex::texel_float2() function.
        DS_INTRINSIC_TEX_TEXEL_FLOAT3,    ///< The tex::texel_float3() function.
        DS_INTRINSIC_TEX_TEXEL_FLOAT4,    ///< The tex::texel_float4() function.
        DS_INTRINSIC_TEX_TEXEL_COLOR,     ///< The tex::texel_color() function.
        DS_INTRINSIC_TEX_TEXTURE_ISVALID, ///< The tex::texture_isvalid() function.
        DS_INTRINSIC_TEX_LAST = DS_INTRINSIC_TEX_TEXTURE_ISVALID,

        // df module intrinsics
        DS_INTRINSIC_DF_FIRST = 0x0600,

        DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF = DS_INTRINSIC_DF_FIRST,
        DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF,
        DS_INTRINSIC_DF_SPECULAR_BSDF,
        DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF,
        DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF,
        DS_INTRINSIC_DF_MEASURED_BSDF,
        DS_INTRINSIC_DF_DIFFUSE_EDF,
        DS_INTRINSIC_DF_MEASURED_EDF,
        DS_INTRINSIC_DF_SPOT_EDF,
        DS_INTRINSIC_DF_ANISOTROPIC_VDF,
        DS_INTRINSIC_DF_NORMALIZED_MIX,
        DS_INTRINSIC_DF_CLAMPED_MIX,
        DS_INTRINSIC_DF_WEIGHTED_LAYER,
        DS_INTRINSIC_DF_FRESNEL_LAYER,
        DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER,
        DS_INTRINSIC_DF_MEASURED_CURVE_LAYER,
        DS_INTRINSIC_DF_THIN_FILM,
        DS_INTRINSIC_DF_TINT,
        DS_INTRINSIC_DF_DIRECTIONAL_FACTOR,
        DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR,
        DS_INTRINSIC_DF_LIGHT_PROFILE_POWER,
        DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM,
        DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID,
        DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID,
        DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF,
        DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF,
        DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF,
        DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF,
        DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF,
        DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
        DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
        DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER,
        DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER,
        DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER,
        DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER,
        DS_INTRINSIC_DF_FRESNEL_FACTOR,
        DS_INTRINSIC_DF_MEASURED_FACTOR,
        DS_INTRINSIC_DF_CHIANG_HAIR_BSDF,
        DS_INTRINSIC_DF_SHEEN_BSDF,
        DS_INTRINSIC_DF_LAST = DS_INTRINSIC_DF_SHEEN_BSDF,

        // scene module intrinsics
        DS_INTRINSIC_SCENE_FIRST = 0x0800,

        DS_INTRINSIC_SCENE_DATA_ISVALID = DS_INTRINSIC_SCENE_FIRST,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_INT,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4,
        DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR,
        DS_INTRINSIC_SCENE_LAST = DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR,

        // debug module
        DS_INTRINSIC_DEBUG_FIRST = 0x0900,
        DS_INTRINSIC_DEBUG_BREAKPOINT = DS_INTRINSIC_DEBUG_FIRST,
        DS_INTRINSIC_DEBUG_ASSERT,
        DS_INTRINSIC_DEBUG_PRINT,
        DS_INTRINSIC_DEBUG_LAST = DS_INTRINSIC_DEBUG_PRINT,

        // DAG backend intrinsic functions
        DS_INTRINSIC_DAG_FIRST = 0x0A00,

        /// This is a structure field access function.
        DS_INTRINSIC_DAG_FIELD_ACCESS = DS_INTRINSIC_DAG_FIRST,
        DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR, ///< This is an array constructor.
        DS_INTRINSIC_DAG_ARRAY_LENGTH,      ///< This is the array length operator.
        DS_INTRINSIC_DAG_SET_OBJECT_ID,     ///< Specifies the used object id.
        DS_INTRINSIC_DAG_SET_TRANSFORMS,    ///< Specifies the transform (w2o and o2w) matrices.
        DS_INTRINSIC_DAG_CALL_LAMBDA,       ///< Calls the lambda function specified by the name.
        DS_INTRINSIC_DAG_GET_DERIV_VALUE,   ///< Extract value part of derivative value.
        DS_INTRINSIC_DAG_MAKE_DERIV,        ///< Create a derivative value from a non-derivative
                                            ///< value, setting dx and dy to zero.
        DS_INTRINSIC_DAG_LAST = DS_INTRINSIC_DAG_MAKE_DERIV,

        // JIT Backend intrinsic functions
        DS_INTRINSIC_JIT_LOOKUP = 0x8000,   ///< Texture result lookup.
    };

    /// Returns the kind of this definition.
    virtual Kind get_kind() const = 0;

    /// Get the symbol of the definition.
    virtual ISymbol const *get_symbol() const = 0;

    /// Get the type of the definition.
    virtual IType const *get_type() const = 0;

    /// Get the declaration of the definition if any.
    ///
    /// \note That imported definitions have no declaration (in the current module).
    ///       Additionally compiler generated definitions might have no declaration at all.
    virtual IDeclaration const *get_declaration() const = 0;

    /// Get the default expression of a parameter of a function, constructor or annotation.
    ///
    /// \param index  the index of the parameter
    virtual IExpression const *get_default_param_initializer(int index) const = 0;

    /// Return the value of an enum constant or a global constant.
    virtual IValue const *get_constant_value() const = 0;

    /// Return the field index of a field member.
    virtual int get_field_index() const = 0;

    /// Return the semantics of a function/constructor.
    virtual Semantics get_semantics() const = 0;

    /// Return the parameter index of a parameter.
    virtual int get_parameter_index() const = 0;

    /// Return the namespace of a namespace alias.
    virtual ISymbol const *get_namespace() const = 0;

    /// Get the prototype declaration of the definition if any.
    virtual IDeclaration const *get_prototype_declaration() const = 0;

    /// Get a boolean property of this definition.
    ///
    /// \param prop  the requested property
    virtual bool get_property(Property prop) const = 0;

    /// Return the position of this definition if any.
    virtual Position const *get_position() const = 0;

    /// Set the position of this definition if any.
    ///
    /// \param pos  the new position
    virtual void set_position(Position const *pos) = 0;

    /// Return the mask specifying which parameters of a function are derivable.
    ///
    /// For example, if bit 0 is set, a backend supporting derivatives may provide derivative
    /// values as the first parameter of the function.
    virtual unsigned get_parameter_derivable_mask() const = 0;
};

/// Check if the given semantic describes a constructor.
///
/// \param sema  the semantics
///
/// \return True is the given semantics is a constructor semantic.
inline bool is_constructor(IDefinition::Semantics sema)
{
    switch (sema) {
    case IDefinition::DS_COPY_CONSTRUCTOR:
    case IDefinition::DS_CONV_CONSTRUCTOR:
    case IDefinition::DS_ELEM_CONSTRUCTOR:
    case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
    case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
    case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
    case IDefinition::DS_INVALID_REF_CONSTRUCTOR:
    case IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR:
    case IDefinition::DS_TEXTURE_CONSTRUCTOR:
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        return true;
    default:
        return false;
    }
}

/// Converts an expression operator into the definition semantics.
///
/// \param op  an expression operator
inline IDefinition::Semantics operator_to_semantic(IExpression::Operator op)
{
    return IDefinition::Semantics(IDefinition::DS_OP_BASE + op);
}

/// Converts an expression operator into the definition semantics.
///
/// \param op  an unary operator
inline IDefinition::Semantics operator_to_semantic(IExpression_unary::Operator op)
{
    return IDefinition::Semantics(IDefinition::DS_OP_BASE + op);
}

/// Converts an expression operator into the definition semantics.
///
/// \param op  a binary operator
inline IDefinition::Semantics operator_to_semantic(IExpression_binary::Operator op)
{
    return IDefinition::Semantics(IDefinition::DS_OP_BASE + op);
}

/// Checks if a definition semantics is an expression operator.
///
/// \param sema  the semantics
inline bool semantic_is_operator(IDefinition::Semantics sema)
{
    return IDefinition::DS_OP_BASE <= sema && sema <= IDefinition::DS_OP_END;
}

/// Checks if a definition semantics is an annotation.
inline bool semantic_is_annotation(IDefinition::Semantics sema)
{
    return IDefinition::DS_ANNOTATION_FIRST <= sema && sema <= IDefinition::DS_ANNOTATION_LAST;
}

/// Converts a definition semantics to an expression operator.
///
/// \param sema  the semantics
///
/// \note: Only valid if sema is in range of [DS_OP_BASE, DS_OP_END]
inline IExpression::Operator semantic_to_operator(IDefinition::Semantics sema)
{
    return IExpression::Operator(sema - IDefinition::DS_OP_BASE);
}

/// Check if the given semantics is from the debug module.
///
/// \param sema  the semantics
inline bool is_debug_semantic(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_DEBUG_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_DEBUG_LAST;
}

/// Check if the given semantics is from the state module.
///
/// \param sema  the semantics
inline bool is_state_semantics(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_STATE_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_STATE_LAST;
}

/// Check if the given semantics is from the math module.
///
/// \param sema  the semantics
inline bool is_math_semantics(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_MATH_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_MATH_LAST;
}

/// Check if the given semantics is from the tex module.
///
/// \param sema  the semantics
inline bool is_tex_semantics(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_TEX_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_TEX_LAST;
}

/// Check if the given semantics is from the df module.
///
/// \param sema  the semantics
inline bool is_df_semantics(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_DF_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_DF_LAST;
}


/// Check if the given semantics is from the scene module.
///
/// \param sema  the semantics
inline bool is_scene_semantics(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_SCENE_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_SCENE_LAST;
}

/// Check if the given semantics is an elemental distribution function (i.e. not a modifier or
/// combiner).
inline bool is_elemental_df_semantics(IDefinition::Semantics sema)
{
    if (!is_df_semantics(sema))
        return false;

    switch (sema) {
    case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
    case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
    case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
    case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
    case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
    case IDefinition::DS_INTRINSIC_DF_DIFFUSE_EDF:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_EDF:
    case IDefinition::DS_INTRINSIC_DF_SPOT_EDF:
    case IDefinition::DS_INTRINSIC_DF_ANISOTROPIC_VDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
    case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
    case IDefinition::DS_INTRINSIC_DF_CHIANG_HAIR_BSDF:
    case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
        return true;

    case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
    case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
    case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
    case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
    case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
    case IDefinition::DS_INTRINSIC_DF_TINT:
    case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
    case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
    case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
    case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:

    case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
    case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
    case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
    case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
        return false;

    default:
        return false;
    }
}


/// Check if the given semantics is a generated DAG intrinsic.
///
/// \param sema  the semantics
inline bool is_DAG_semantics(IDefinition::Semantics sema)
{
    return
        IDefinition::DS_INTRINSIC_DAG_FIRST <= sema &&
        sema <= IDefinition::DS_INTRINSIC_DAG_LAST;
}


/// A callback interface to support constant folding of MDL AST expressions.
///
/// The constant folder IExpression::fold() uses this interface to
/// report exception conditions and retrieve helper data.
class IConst_fold_handler
{
public:
    /// Exception reasons.
    enum Reason {
        ER_INT_DIVISION_BY_ZERO,    ///< Integer division by zero.
        ER_INDEX_OUT_OF_BOUND       ///< Index out of bounds.
    };

    /// Called by IExpression::fold() if an exception occurs.
    ///
    /// \param r       the exception reason
    /// \param expr    the erroneous expression
    /// \param index   additional parameter for ER_INDEX_OUT_OF_BOUND
    /// \param length  additional parameter for ER_INDEX_OUT_OF_BOUND
    virtual void exception(
        Reason            r,
        IExpression const *expr,
        int               index = 0,
        int               length = 0) = 0;

    /// Called by IExpression_reference::fold() to lookup a value of a (constant) variable.
    ///
    /// \param var   the definition of the variable
    ///
    /// \return IValue_bad if this variable is not constant, its value otherwise
    virtual IValue const *lookup(
        IDefinition const *var) = 0;

    /// Check whether evaluate_intrinsic_function() should be called for an unhandled
    /// intrinsic functions with the given semantic.
    ///
    /// \param semantic  the semantic to check for
    virtual bool is_evaluate_intrinsic_function_enabled(
        IDefinition::Semantics semantic) const = 0;

    /// Called by IExpression_call::fold() to evaluate unhandled intrinsic functions.
    ///
    /// \param semantic     the semantic of the function to call
    /// \param arguments    the arguments for the call
    /// \param n_arguments  the number of arguments
    ///
    /// \return IValue_bad if this function could not be evaluated, its value otherwise
    virtual IValue const* evaluate_intrinsic_function(
        IDefinition::Semantics semantic,
        IValue const * const   arguments[],
        size_t                 n_arguments) = 0;
};

}  // mdl
}  // mi

#endif

