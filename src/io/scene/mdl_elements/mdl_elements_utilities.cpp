/***************************************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Public and module-internal utilities related to MDL scene
///             elements.

#include "pch.h"

#include "i_mdl_elements_utilities.h"
#include "mdl_elements_utilities.h"

#include "i_mdl_elements_compiled_material.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_material_definition.h"
#include "i_mdl_elements_material_instance.h"
#include "i_mdl_elements_module.h"
#include "mdl_elements_detail.h"

#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/mdl/mdl.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <base/system/main/access_module.h>
#include <boost/core/ignore_unused.hpp>
#include <boost/functional/hash.hpp>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>
#include <mdl/compiler/compilercore/compilercore_visitor.h>
#include <mdl/codegenerators/generator_dag/generator_dag_generated_dag.h>
#include <mdl/codegenerators/generator_dag/generator_dag_tools.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>
#include <mdl/integration/i18n/i_i18n.h>

namespace MI {

namespace MDL {

using mi::mdl::as;
using mi::mdl::is;
using mi::mdl::cast;
using MI::MDL::I18N::Mdl_translator_module;

// ********** Conversion from mi::mdl to mi::neuraylib *********************************************

mi::neuraylib::IFunction_definition::Semantics mdl_semantics_to_ext_semantics(
    mi::mdl::IDefinition::Semantics semantic)
{
    if( mi::mdl::semantic_is_operator( semantic)) {
        mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator( semantic);
        switch( op) {

#define CASE_OK(e) \
            case mi::mdl::IExpression::OK_##e: \
                return mi::neuraylib::IFunction_definition::DS_##e;

            CASE_OK( BITWISE_COMPLEMENT);
            CASE_OK( LOGICAL_NOT);
            CASE_OK( POSITIVE);
            CASE_OK( NEGATIVE);
            CASE_OK( PRE_INCREMENT);
            CASE_OK( PRE_DECREMENT);
            CASE_OK( POST_INCREMENT);
            CASE_OK( POST_DECREMENT);
            CASE_OK( SELECT);
            CASE_OK( ARRAY_INDEX);
            CASE_OK( MULTIPLY);
            CASE_OK( DIVIDE);
            CASE_OK( MODULO);
            CASE_OK( PLUS);
            CASE_OK( MINUS);
            CASE_OK( SHIFT_LEFT);
            CASE_OK( SHIFT_RIGHT);
            CASE_OK( UNSIGNED_SHIFT_RIGHT);
            CASE_OK( LESS);
            CASE_OK( LESS_OR_EQUAL);
            CASE_OK( GREATER_OR_EQUAL);
            CASE_OK( GREATER);
            CASE_OK( EQUAL);
            CASE_OK( NOT_EQUAL);
            CASE_OK( BITWISE_AND);
            CASE_OK( BITWISE_XOR);
            CASE_OK( BITWISE_OR);
            CASE_OK( LOGICAL_AND);
            CASE_OK( LOGICAL_OR);
            CASE_OK( ASSIGN);
            CASE_OK( MULTIPLY_ASSIGN);
            CASE_OK( DIVIDE_ASSIGN);
            CASE_OK( MODULO_ASSIGN);
            CASE_OK( PLUS_ASSIGN);
            CASE_OK( MINUS_ASSIGN);
            CASE_OK( SHIFT_LEFT_ASSIGN);
            CASE_OK( SHIFT_RIGHT_ASSIGN);
            CASE_OK( UNSIGNED_SHIFT_RIGHT_ASSIGN);
            CASE_OK( BITWISE_OR_ASSIGN);
            CASE_OK( BITWISE_XOR_ASSIGN);
            CASE_OK( BITWISE_AND_ASSIGN);
            CASE_OK( SEQUENCE);
            CASE_OK( TERNARY);

#undef CASE_OK

            case mi::mdl::IExpression::OK_CALL: // should not appear in this context
                ASSERT( M_SCENE, false);
                return mi::neuraylib::IFunction_definition::DS_UNKNOWN;
        }
        ASSERT( M_SCENE, false);
        return mi::neuraylib::IFunction_definition::DS_UNKNOWN;
    }

    switch( semantic) {

#define CASE_DS(e) \
        case mi::mdl::IDefinition::DS_##e: \
            return mi::neuraylib::IFunction_definition::DS_##e;

        CASE_DS( UNKNOWN);
        CASE_DS( CONV_CONSTRUCTOR);
        CASE_DS( ELEM_CONSTRUCTOR);
        CASE_DS( COLOR_SPECTRUM_CONSTRUCTOR);
        CASE_DS( MATRIX_ELEM_CONSTRUCTOR);
        CASE_DS( MATRIX_DIAG_CONSTRUCTOR);
        CASE_DS( INVALID_REF_CONSTRUCTOR);
        CASE_DS( DEFAULT_STRUCT_CONSTRUCTOR);
        CASE_DS( TEXTURE_CONSTRUCTOR);
        CASE_DS( CONV_OPERATOR);
        CASE_DS( INTRINSIC_MATH_ABS);
        CASE_DS( INTRINSIC_MATH_ACOS);
        CASE_DS( INTRINSIC_MATH_ALL);
        CASE_DS( INTRINSIC_MATH_ANY);
        CASE_DS( INTRINSIC_MATH_ASIN);
        CASE_DS( INTRINSIC_MATH_ATAN);
        CASE_DS( INTRINSIC_MATH_ATAN2);
        CASE_DS( INTRINSIC_MATH_AVERAGE);
        CASE_DS( INTRINSIC_MATH_CEIL);
        CASE_DS( INTRINSIC_MATH_CLAMP);
        CASE_DS( INTRINSIC_MATH_COS);
        CASE_DS( INTRINSIC_MATH_CROSS);
        CASE_DS( INTRINSIC_MATH_DEGREES);
        CASE_DS( INTRINSIC_MATH_DISTANCE);
        CASE_DS( INTRINSIC_MATH_DOT);
        CASE_DS( INTRINSIC_MATH_EVAL_AT_WAVELENGTH);
        CASE_DS( INTRINSIC_MATH_EXP);
        CASE_DS( INTRINSIC_MATH_EXP2);
        CASE_DS( INTRINSIC_MATH_FLOOR);
        CASE_DS( INTRINSIC_MATH_FMOD);
        CASE_DS( INTRINSIC_MATH_FRAC);
        CASE_DS( INTRINSIC_MATH_ISNAN);
        CASE_DS( INTRINSIC_MATH_ISFINITE);
        CASE_DS( INTRINSIC_MATH_LENGTH);
        CASE_DS( INTRINSIC_MATH_LERP);
        CASE_DS( INTRINSIC_MATH_LOG);
        CASE_DS( INTRINSIC_MATH_LOG2);
        CASE_DS( INTRINSIC_MATH_LOG10);
        CASE_DS( INTRINSIC_MATH_LUMINANCE);
        CASE_DS( INTRINSIC_MATH_MAX);
        CASE_DS( INTRINSIC_MATH_MAX_VALUE);
        CASE_DS( INTRINSIC_MATH_MAX_VALUE_WAVELENGTH);
        CASE_DS( INTRINSIC_MATH_MIN);
        CASE_DS( INTRINSIC_MATH_MIN_VALUE);
        CASE_DS( INTRINSIC_MATH_MIN_VALUE_WAVELENGTH);
        CASE_DS( INTRINSIC_MATH_MODF);
        CASE_DS( INTRINSIC_MATH_NORMALIZE);
        CASE_DS( INTRINSIC_MATH_POW);
        CASE_DS( INTRINSIC_MATH_RADIANS);
        CASE_DS( INTRINSIC_MATH_ROUND);
        CASE_DS( INTRINSIC_MATH_RSQRT);
        CASE_DS( INTRINSIC_MATH_SATURATE);
        CASE_DS( INTRINSIC_MATH_SIGN);
        CASE_DS( INTRINSIC_MATH_SIN);
        CASE_DS( INTRINSIC_MATH_SINCOS);
        CASE_DS( INTRINSIC_MATH_SMOOTHSTEP);
        CASE_DS( INTRINSIC_MATH_SQRT);
        CASE_DS( INTRINSIC_MATH_STEP);
        CASE_DS( INTRINSIC_MATH_TAN);
        CASE_DS( INTRINSIC_MATH_TRANSPOSE);
        CASE_DS( INTRINSIC_MATH_BLACKBODY);
        CASE_DS( INTRINSIC_MATH_EMISSION_COLOR);
        CASE_DS( INTRINSIC_MATH_DX);
        CASE_DS( INTRINSIC_MATH_DY);
        CASE_DS( INTRINSIC_STATE_POSITION);
        CASE_DS( INTRINSIC_STATE_NORMAL);
        CASE_DS( INTRINSIC_STATE_GEOMETRY_NORMAL);
        CASE_DS( INTRINSIC_STATE_MOTION);
        CASE_DS( INTRINSIC_STATE_TEXTURE_SPACE_MAX);
        CASE_DS( INTRINSIC_STATE_TEXTURE_COORDINATE);
        CASE_DS( INTRINSIC_STATE_TEXTURE_TANGENT_U);
        CASE_DS( INTRINSIC_STATE_TEXTURE_TANGENT_V);
        CASE_DS( INTRINSIC_STATE_TANGENT_SPACE);
        CASE_DS( INTRINSIC_STATE_GEOMETRY_TANGENT_U);
        CASE_DS( INTRINSIC_STATE_GEOMETRY_TANGENT_V);
        CASE_DS( INTRINSIC_STATE_DIRECTION);
        CASE_DS( INTRINSIC_STATE_ANIMATION_TIME);
        CASE_DS( INTRINSIC_STATE_WAVELENGTH_BASE);
        CASE_DS( INTRINSIC_STATE_TRANSFORM);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_POINT);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_VECTOR);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_NORMAL);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_SCALE);
        CASE_DS( INTRINSIC_STATE_ROUNDED_CORNER_NORMAL);
        CASE_DS( INTRINSIC_STATE_METERS_PER_SCENE_UNIT);
        CASE_DS( INTRINSIC_STATE_SCENE_UNITS_PER_METER);
        CASE_DS( INTRINSIC_STATE_OBJECT_ID);
        CASE_DS( INTRINSIC_STATE_WAVELENGTH_MIN);
        CASE_DS( INTRINSIC_STATE_WAVELENGTH_MAX);
        CASE_DS( INTRINSIC_TEX_WIDTH);
        CASE_DS( INTRINSIC_TEX_HEIGHT);
        CASE_DS( INTRINSIC_TEX_DEPTH);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT2);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT3);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT4);
        CASE_DS( INTRINSIC_TEX_LOOKUP_COLOR);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT2);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT3);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT4);
        CASE_DS( INTRINSIC_TEX_TEXEL_COLOR);
        CASE_DS( INTRINSIC_TEX_TEXTURE_ISVALID);
        CASE_DS( INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF);
        CASE_DS( INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF);
        CASE_DS( INTRINSIC_DF_SPECULAR_BSDF);
        CASE_DS( INTRINSIC_DF_SIMPLE_GLOSSY_BSDF);
        CASE_DS( INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF);
        CASE_DS( INTRINSIC_DF_MEASURED_BSDF);
        CASE_DS( INTRINSIC_DF_DIFFUSE_EDF);
        CASE_DS( INTRINSIC_DF_MEASURED_EDF);
        CASE_DS( INTRINSIC_DF_SPOT_EDF);
        CASE_DS( INTRINSIC_DF_ANISOTROPIC_VDF);
        CASE_DS( INTRINSIC_DF_NORMALIZED_MIX);
        CASE_DS( INTRINSIC_DF_CLAMPED_MIX);
        CASE_DS( INTRINSIC_DF_WEIGHTED_LAYER);
        CASE_DS( INTRINSIC_DF_FRESNEL_LAYER);
        CASE_DS( INTRINSIC_DF_CUSTOM_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_MEASURED_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_THIN_FILM);
        CASE_DS( INTRINSIC_DF_TINT);
        CASE_DS( INTRINSIC_DF_DIRECTIONAL_FACTOR);
        CASE_DS( INTRINSIC_DF_MEASURED_CURVE_FACTOR);
        CASE_DS( INTRINSIC_DF_LIGHT_PROFILE_POWER);
        CASE_DS( INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM);
        CASE_DS( INTRINSIC_DF_LIGHT_PROFILE_ISVALID);
        CASE_DS( INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID);
        CASE_DS( INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF);
        CASE_DS( INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF);
        CASE_DS( INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF);
        CASE_DS( INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF);
        CASE_DS( INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF);
        CASE_DS( INTRINSIC_DF_COLOR_NORMALIZED_MIX);
        CASE_DS( INTRINSIC_DF_COLOR_CLAMPED_MIX);
        CASE_DS( INTRINSIC_DF_COLOR_WEIGHTED_LAYER);
        CASE_DS( INTRINSIC_DF_COLOR_FRESNEL_LAYER);
        CASE_DS( INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_FRESNEL_FACTOR);
        CASE_DS( INTRINSIC_DEBUG_BREAKPOINT);
        CASE_DS( INTRINSIC_DEBUG_ASSERT);
        CASE_DS( INTRINSIC_DEBUG_PRINT);
        CASE_DS( INTRINSIC_DAG_FIELD_ACCESS);
        CASE_DS( INTRINSIC_DAG_ARRAY_CONSTRUCTOR);
        CASE_DS( INTRINSIC_DAG_INDEX_ACCESS);
        CASE_DS( INTRINSIC_DAG_ARRAY_LENGTH);

#undef CASE_DS

        // handled in first switch statement
        case mi::mdl::IDefinition::DS_OP_BASE:
        case mi::mdl::IDefinition::DS_OP_END:
            ASSERT( M_SCENE, false);
            return mi::neuraylib::IFunction_definition::DS_UNKNOWN;

        // should not appear in this context
        case mi::mdl::IDefinition::DS_COPY_CONSTRUCTOR:
        case mi::mdl::IDefinition::DS_INTRINSIC_ANNOTATION:
        case mi::mdl::IDefinition::DS_THROWS_ANNOTATION:
        case mi::mdl::IDefinition::DS_SINCE_ANNOTATION:
        case mi::mdl::IDefinition::DS_REMOVED_ANNOTATION:
        case mi::mdl::IDefinition::DS_CONST_EXPR_ANNOTATION:
        case mi::mdl::IDefinition::DS_DERIVABLE_ANNOTATION:
        case mi::mdl::IDefinition::DS_NATIVE_ANNOTATION:
        case mi::mdl::IDefinition::DS_UNUSED_ANNOTATION:
        case mi::mdl::IDefinition::DS_NOINLINE_ANNOTATION:
        case mi::mdl::IDefinition::DS_SOFT_RANGE_ANNOTATION:
        case mi::mdl::IDefinition::DS_HARD_RANGE_ANNOTATION:
        case mi::mdl::IDefinition::DS_HIDDEN_ANNOTATION:
        case mi::mdl::IDefinition::DS_DEPRECATED_ANNOTATION:
        case mi::mdl::IDefinition::DS_VERSION_NUMBER_ANNOTATION:
        case mi::mdl::IDefinition::DS_VERSION_ANNOTATION:
        case mi::mdl::IDefinition::DS_DEPENDENCY_ANNOTATION:
        case mi::mdl::IDefinition::DS_UI_ORDER_ANNOTATION:
        case mi::mdl::IDefinition::DS_USAGE_ANNOTATION:
        case mi::mdl::IDefinition::DS_ENABLE_IF_ANNOTATION:
        case mi::mdl::IDefinition::DS_THUMBNAIL_ANNOTATION:
        case mi::mdl::IDefinition::DS_DISPLAY_NAME_ANNOTATION:
        case mi::mdl::IDefinition::DS_IN_GROUP_ANNOTATION:
        case mi::mdl::IDefinition::DS_DESCRIPTION_ANNOTATION:
        case mi::mdl::IDefinition::DS_AUTHOR_ANNOTATION:
        case mi::mdl::IDefinition::DS_CONTRIBUTOR_ANNOTATION:
        case mi::mdl::IDefinition::DS_COPYRIGHT_NOTICE_ANNOTATION:
        case mi::mdl::IDefinition::DS_CREATED_ANNOTATION:
        case mi::mdl::IDefinition::DS_MODIFIED_ANNOTATION:
        case mi::mdl::IDefinition::DS_KEYWORDS_ANNOTATION:
        case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_OBJECT_ID:
        case mi::mdl::IDefinition::DS_INTRINSIC_DAG_SET_TRANSFORMS:
        case mi::mdl::IDefinition::DS_INTRINSIC_DAG_CALL_LAMBDA:
        case mi::mdl::IDefinition::DS_INTRINSIC_DAG_GET_DERIV_VALUE:
        case mi::mdl::IDefinition::DS_INTRINSIC_DAG_MAKE_DERIV:
        case mi::mdl::IDefinition::DS_INTRINSIC_JIT_LOOKUP:
            ASSERT( M_SCENE, false);
            return mi::neuraylib::IFunction_definition::DS_UNKNOWN;
    }

    ASSERT( M_SCENE, false);
    return mi::neuraylib::IFunction_definition::DS_UNKNOWN;
}

mi::mdl::IDefinition::Semantics ext_semantics_to_mdl_semantics(
    mi::neuraylib::IFunction_definition::Semantics semantic)
{
    switch( semantic) {

#define CASE_DS(e) \
        case mi::neuraylib::IFunction_definition::DS_##e: \
            return mi::mdl::IDefinition::DS_##e;
#define CASE_OK(e) \
        case mi::neuraylib::IFunction_definition::DS_##e: \
            return mi::mdl::operator_to_semantic( mi::mdl::IExpression::OK_##e);

        CASE_DS( UNKNOWN);
        CASE_DS( CONV_CONSTRUCTOR);
        CASE_DS( ELEM_CONSTRUCTOR);
        CASE_DS( COLOR_SPECTRUM_CONSTRUCTOR);
        CASE_DS( MATRIX_ELEM_CONSTRUCTOR);
        CASE_DS( MATRIX_DIAG_CONSTRUCTOR);
        CASE_DS( INVALID_REF_CONSTRUCTOR);
        CASE_DS( DEFAULT_STRUCT_CONSTRUCTOR);
        CASE_DS( TEXTURE_CONSTRUCTOR);
        CASE_OK( BITWISE_COMPLEMENT);
        CASE_OK( LOGICAL_NOT);
        CASE_OK( POSITIVE);
        CASE_OK( NEGATIVE);
        CASE_OK( PRE_INCREMENT);
        CASE_OK( PRE_DECREMENT);
        CASE_OK( POST_INCREMENT);
        CASE_OK( POST_DECREMENT);
        CASE_OK( SELECT);
        CASE_OK( ARRAY_INDEX);
        CASE_OK( MULTIPLY);
        CASE_OK( DIVIDE);
        CASE_OK( MODULO);
        CASE_OK( PLUS);
        CASE_OK( MINUS);
        CASE_OK( SHIFT_LEFT);
        CASE_OK( SHIFT_RIGHT);
        CASE_OK( UNSIGNED_SHIFT_RIGHT);
        CASE_OK( LESS);
        CASE_OK( LESS_OR_EQUAL);
        CASE_OK( GREATER_OR_EQUAL);
        CASE_OK( GREATER);
        CASE_OK( EQUAL);
        CASE_OK( NOT_EQUAL);
        CASE_OK( BITWISE_AND);
        CASE_OK( BITWISE_XOR);
        CASE_OK( BITWISE_OR);
        CASE_OK( LOGICAL_AND);
        CASE_OK( LOGICAL_OR);
        CASE_OK( ASSIGN);
        CASE_OK( MULTIPLY_ASSIGN);
        CASE_OK( DIVIDE_ASSIGN);
        CASE_OK( MODULO_ASSIGN);
        CASE_OK( PLUS_ASSIGN);
        CASE_OK( MINUS_ASSIGN);
        CASE_OK( SHIFT_LEFT_ASSIGN);
        CASE_OK( SHIFT_RIGHT_ASSIGN);
        CASE_OK( UNSIGNED_SHIFT_RIGHT_ASSIGN);
        CASE_OK( BITWISE_OR_ASSIGN);
        CASE_OK( BITWISE_XOR_ASSIGN);
        CASE_OK( BITWISE_AND_ASSIGN);
        CASE_OK( SEQUENCE);
        CASE_OK( TERNARY);
        CASE_DS( CONV_OPERATOR);
        CASE_DS( INTRINSIC_MATH_ABS);
        CASE_DS( INTRINSIC_MATH_ACOS);
        CASE_DS( INTRINSIC_MATH_ALL);
        CASE_DS( INTRINSIC_MATH_ANY);
        CASE_DS( INTRINSIC_MATH_ASIN);
        CASE_DS( INTRINSIC_MATH_ATAN);
        CASE_DS( INTRINSIC_MATH_ATAN2);
        CASE_DS( INTRINSIC_MATH_AVERAGE);
        CASE_DS( INTRINSIC_MATH_CEIL);
        CASE_DS( INTRINSIC_MATH_CLAMP);
        CASE_DS( INTRINSIC_MATH_COS);
        CASE_DS( INTRINSIC_MATH_CROSS);
        CASE_DS( INTRINSIC_MATH_DEGREES);
        CASE_DS( INTRINSIC_MATH_DISTANCE);
        CASE_DS( INTRINSIC_MATH_DOT);
        CASE_DS( INTRINSIC_MATH_EVAL_AT_WAVELENGTH);
        CASE_DS( INTRINSIC_MATH_EXP);
        CASE_DS( INTRINSIC_MATH_EXP2);
        CASE_DS( INTRINSIC_MATH_FLOOR);
        CASE_DS( INTRINSIC_MATH_FMOD);
        CASE_DS( INTRINSIC_MATH_FRAC);
        CASE_DS( INTRINSIC_MATH_ISNAN);
        CASE_DS( INTRINSIC_MATH_ISFINITE);
        CASE_DS( INTRINSIC_MATH_LENGTH);
        CASE_DS( INTRINSIC_MATH_LERP);
        CASE_DS( INTRINSIC_MATH_LOG);
        CASE_DS( INTRINSIC_MATH_LOG2);
        CASE_DS( INTRINSIC_MATH_LOG10);
        CASE_DS( INTRINSIC_MATH_LUMINANCE);
        CASE_DS( INTRINSIC_MATH_MAX);
        CASE_DS( INTRINSIC_MATH_MAX_VALUE);
        CASE_DS( INTRINSIC_MATH_MAX_VALUE_WAVELENGTH);
        CASE_DS( INTRINSIC_MATH_MIN);
        CASE_DS( INTRINSIC_MATH_MIN_VALUE);
        CASE_DS( INTRINSIC_MATH_MIN_VALUE_WAVELENGTH);
        CASE_DS( INTRINSIC_MATH_MODF);
        CASE_DS( INTRINSIC_MATH_NORMALIZE);
        CASE_DS( INTRINSIC_MATH_POW);
        CASE_DS( INTRINSIC_MATH_RADIANS);
        CASE_DS( INTRINSIC_MATH_ROUND);
        CASE_DS( INTRINSIC_MATH_RSQRT);
        CASE_DS( INTRINSIC_MATH_SATURATE);
        CASE_DS( INTRINSIC_MATH_SIGN);
        CASE_DS( INTRINSIC_MATH_SIN);
        CASE_DS( INTRINSIC_MATH_SINCOS);
        CASE_DS( INTRINSIC_MATH_SMOOTHSTEP);
        CASE_DS( INTRINSIC_MATH_SQRT);
        CASE_DS( INTRINSIC_MATH_STEP);
        CASE_DS( INTRINSIC_MATH_TAN);
        CASE_DS( INTRINSIC_MATH_TRANSPOSE);
        CASE_DS( INTRINSIC_MATH_BLACKBODY);
        CASE_DS( INTRINSIC_MATH_EMISSION_COLOR);
        CASE_DS( INTRINSIC_MATH_DX);
        CASE_DS( INTRINSIC_MATH_DY);
        CASE_DS( INTRINSIC_STATE_POSITION);
        CASE_DS( INTRINSIC_STATE_NORMAL);
        CASE_DS( INTRINSIC_STATE_GEOMETRY_NORMAL);
        CASE_DS( INTRINSIC_STATE_MOTION);
        CASE_DS( INTRINSIC_STATE_TEXTURE_SPACE_MAX);
        CASE_DS( INTRINSIC_STATE_TEXTURE_COORDINATE);
        CASE_DS( INTRINSIC_STATE_TEXTURE_TANGENT_U);
        CASE_DS( INTRINSIC_STATE_TEXTURE_TANGENT_V);
        CASE_DS( INTRINSIC_STATE_TANGENT_SPACE);
        CASE_DS( INTRINSIC_STATE_GEOMETRY_TANGENT_U);
        CASE_DS( INTRINSIC_STATE_GEOMETRY_TANGENT_V);
        CASE_DS( INTRINSIC_STATE_DIRECTION);
        CASE_DS( INTRINSIC_STATE_ANIMATION_TIME);
        CASE_DS( INTRINSIC_STATE_WAVELENGTH_BASE);
        CASE_DS( INTRINSIC_STATE_TRANSFORM);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_POINT);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_VECTOR);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_NORMAL);
        CASE_DS( INTRINSIC_STATE_TRANSFORM_SCALE);
        CASE_DS( INTRINSIC_STATE_ROUNDED_CORNER_NORMAL);
        CASE_DS( INTRINSIC_STATE_METERS_PER_SCENE_UNIT);
        CASE_DS( INTRINSIC_STATE_SCENE_UNITS_PER_METER);
        CASE_DS( INTRINSIC_STATE_OBJECT_ID);
        CASE_DS( INTRINSIC_STATE_WAVELENGTH_MIN);
        CASE_DS( INTRINSIC_STATE_WAVELENGTH_MAX);
        CASE_DS( INTRINSIC_TEX_WIDTH);
        CASE_DS( INTRINSIC_TEX_HEIGHT);
        CASE_DS( INTRINSIC_TEX_DEPTH);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT2);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT3);
        CASE_DS( INTRINSIC_TEX_LOOKUP_FLOAT4);
        CASE_DS( INTRINSIC_TEX_LOOKUP_COLOR);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT2);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT3);
        CASE_DS( INTRINSIC_TEX_TEXEL_FLOAT4);
        CASE_DS( INTRINSIC_TEX_TEXEL_COLOR);
        CASE_DS( INTRINSIC_TEX_TEXTURE_ISVALID);
        CASE_DS( INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF);
        CASE_DS( INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF);
        CASE_DS( INTRINSIC_DF_SPECULAR_BSDF);
        CASE_DS( INTRINSIC_DF_SIMPLE_GLOSSY_BSDF);
        CASE_DS( INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF);
        CASE_DS( INTRINSIC_DF_MEASURED_BSDF);
        CASE_DS( INTRINSIC_DF_DIFFUSE_EDF);
        CASE_DS( INTRINSIC_DF_MEASURED_EDF);
        CASE_DS( INTRINSIC_DF_SPOT_EDF);
        CASE_DS( INTRINSIC_DF_ANISOTROPIC_VDF);
        CASE_DS( INTRINSIC_DF_NORMALIZED_MIX);
        CASE_DS( INTRINSIC_DF_CLAMPED_MIX);
        CASE_DS( INTRINSIC_DF_WEIGHTED_LAYER);
        CASE_DS( INTRINSIC_DF_FRESNEL_LAYER);
        CASE_DS( INTRINSIC_DF_CUSTOM_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_MEASURED_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_THIN_FILM);
        CASE_DS( INTRINSIC_DF_TINT);
        CASE_DS( INTRINSIC_DF_DIRECTIONAL_FACTOR);
        CASE_DS( INTRINSIC_DF_MEASURED_CURVE_FACTOR);
        CASE_DS( INTRINSIC_DF_LIGHT_PROFILE_POWER);
        CASE_DS( INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM);
        CASE_DS( INTRINSIC_DF_LIGHT_PROFILE_ISVALID);
        CASE_DS( INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID);
        CASE_DS( INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF);
        CASE_DS( INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF);
        CASE_DS( INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF);
        CASE_DS( INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF);
        CASE_DS( INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF);
        CASE_DS( INTRINSIC_DF_COLOR_NORMALIZED_MIX);
        CASE_DS( INTRINSIC_DF_COLOR_CLAMPED_MIX);
        CASE_DS( INTRINSIC_DF_COLOR_WEIGHTED_LAYER);
        CASE_DS( INTRINSIC_DF_COLOR_FRESNEL_LAYER);
        CASE_DS( INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER);
        CASE_DS( INTRINSIC_DF_FRESNEL_FACTOR);
        CASE_DS( INTRINSIC_DEBUG_BREAKPOINT);
        CASE_DS( INTRINSIC_DEBUG_ASSERT);
        CASE_DS( INTRINSIC_DEBUG_PRINT);
        CASE_DS( INTRINSIC_DAG_FIELD_ACCESS);
        CASE_DS( INTRINSIC_DAG_ARRAY_CONSTRUCTOR);
        CASE_DS( INTRINSIC_DAG_INDEX_ACCESS);
        CASE_DS( INTRINSIC_DAG_ARRAY_LENGTH);

#undef CASE_DS
#undef CASE_OK

        case mi::neuraylib::IFunction_definition::DS_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return mi::mdl::IDefinition::DS_UNKNOWN;
    }

    ASSERT( M_SCENE, false);
    return mi::mdl::IDefinition::DS_UNKNOWN;
}

// **********  Computation of references to other DB element ***************************************

void collect_references( const IValue* value, DB::Tag_set* result)
{
    IValue::Kind kind = value->get_kind();

    switch( kind) {

        case IValue::VK_BOOL:
        case IValue::VK_INT:
        case IValue::VK_ENUM:
        case IValue::VK_FLOAT:
        case IValue::VK_DOUBLE:
        case IValue::VK_STRING:
        case IValue::VK_VECTOR:
        case IValue::VK_MATRIX:
        case IValue::VK_COLOR:
            return;
        case IValue::VK_ARRAY:
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            mi::Size n = value_compound->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element(
                    value_compound->get_value( i));
                collect_references( element.get(), result);
            }
            return;
        }
        case IValue::VK_TEXTURE:
        case IValue::VK_LIGHT_PROFILE:
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_resource> value_resource(
                value->get_interface<IValue_resource>());
            DB::Tag tag = value_resource->get_value();
            if( tag)
                result->insert( tag);
            return;
        }
        case IValue::VK_INVALID_DF:
            return;
        case IValue::VK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return;
    }

    ASSERT( M_SCENE, false);
    return;
}

void collect_references( const IValue_list* list, DB::Tag_set* result)
{
    mi::Size n = list->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IValue> element( list->get_value( i));
        collect_references( element.get(), result);
    }
}

void collect_references( const IExpression* expr, DB::Tag_set* result)
{
    IExpression::Kind kind = expr->get_kind();

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            collect_references( value.get(), result);
            return;
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            DB::Tag tag = expr_call->get_call();
            result->insert( tag);
            return;
        }
        case IExpression::EK_PARAMETER:
            return;
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag tag = expr_direct_call->get_definition();
            result->insert( tag);
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());
            collect_references( arguments.get(), result);
            return;
        }
        case IExpression::EK_TEMPORARY:
            return;
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return;
    }

    ASSERT( M_SCENE, false);
    return;
}

void collect_references( const IExpression_list* list, DB::Tag_set* result)
{
    mi::Size n = list->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IExpression> element( list->get_expression( i));
        collect_references( element.get(), result);
    }
}

void collect_references( const IAnnotation* annotation, DB::Tag_set* result)
{
    mi::base::Handle<const IExpression_list> arguments( annotation->get_arguments());
    collect_references( arguments.get(), result);
}

void collect_references( const IAnnotation_block* block, DB::Tag_set* result)
{
    mi::Size n = block ? block->get_size() : 0;
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IAnnotation> element( block->get_annotation( i));
        collect_references( element.get(), result);
    }
}

void collect_references( const IAnnotation_list* list, DB::Tag_set* result)
{
    mi::Size n = list->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IAnnotation_block> element( list->get_annotation_block( i));
        collect_references( element.get(), result);
    }
}


// **********  Misc utility functions **************************************************************

const char* get_array_constructor_db_name() { return "mdl::T[](...)"; }

const char* get_array_constructor_mdl_name() { return "T[](...)"; }

bool is_builtin_module( const std::string& module)
{
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    return mdl->is_builtin_module( module.c_str());
}


Mdl_compiled_material* get_default_compiled_material( DB::Transaction* transaction)
{
    DB::Tag tag = transaction->name_to_tag( "mdl::<neuray>::default_material");

    // The "<neuray>" module is created here on the fly. Note that this needs a transaction and
    // other places like the factories of Mdl_module are not guaranteed to run, e.g., they are
    // never executed on the remote side of the Iray Cloud.
    if( !tag) {
        mi::base::Handle<mi::neuraylib::IReader> reader( create_reader(
            "mdl 1.0; export material default_material() = material();"));
        MDL::Execution_context context;
        mi::Sint32 result = Mdl_module::create_module(
            transaction, "::<neuray>", reader.get(), &context);
        if( result != 0)
            return 0;
        tag = transaction->name_to_tag( "mdl::<neuray>::default_material");
        ASSERT( M_SCENE, tag);
    }

    DB::Access<Mdl_material_definition> md( tag, transaction);
    Mdl_material_instance* mi = md->create_material_instance( transaction, /*arguments*/ 0);
    Execution_context context;
    Mdl_compiled_material* cm = mi->create_compiled_material(
        transaction, /*class_compilation*/ false, &context);
    delete mi;
    return cm;
}


// **********  Traversal of types, values, and expressions *****************************************

namespace {

/// Splits a string at the next separator (dot or bracket pair).
///
/// \param s      The string to split.
/// \param head   The start of the string up to the dot, or the array index, or the entire string
///               if no separator is found.
/// \param tail   The remainder of the string, or empty if no separator is found.
void split( const char* s, std::string& head, std::string& tail)
{
    // find first dot
    const char* dot = strchr( s, '.');

    // find first bracket pair
    const char* left_bracket = strchr( s, '[');
    const char* right_bracket = left_bracket ? strchr( left_bracket, ']') : 0;
    if( left_bracket && !right_bracket)
        left_bracket = 0;

    // handle neither dot nor bracket pair
    if( !dot && !left_bracket) {
        head = s;
        tail = "";
        return;
    }

    // handle array index
    if( left_bracket && left_bracket == s) {
        ASSERT( M_SCENE, right_bracket);
        head = std::string( s+1, right_bracket-(s+1));
        tail = std::string( right_bracket[1] == '.' ? right_bracket+2 : right_bracket+1);
        return;
    }

    // handle field name
    const char* sep = dot ? dot : left_bracket;
    if( left_bracket && sep > left_bracket)
        sep = left_bracket;
    head = std::string( s, sep-s);
    tail = std::string( sep == left_bracket ? sep : sep+1);
}

} // namespace

const mi::mdl::IType* get_field_type( const mi::mdl::IType_struct* type, const char* field_name)
{
    if( !type)
        return 0;

    mi::Uint32 count = type->get_field_count();
    for( mi::Uint32 i = 0; i < count; ++i) {
        const mi::mdl::ISymbol* field_symbol;
        const mi::mdl::IType*   field_type;
        type->get_field( i, field_type, field_symbol);
        if( strcmp( field_name, field_symbol->get_name()) == 0)
            return field_type;
    }

    return 0;
}

const IValue* lookup_sub_value(
    const mi::mdl::IType* type,
    const IValue* value,
    const char* path,
    const mi::mdl::IType** sub_type)
{
    ASSERT( M_SCENE, value && path);
    ASSERT( M_SCENE, (!type && !sub_type) || (type && sub_type));

    // handle empty paths
    if( path[0] == '\0') {
        if( sub_type) *sub_type = type;
        value->retain();
        return value;
    }

    // handle non-compounds
    mi::base::Handle<const IValue_compound> value_compound(
        value->get_interface<IValue_compound>());
    if( !value_compound) {
        if( sub_type) *sub_type = 0;
        return 0;
    }

    std::string head, tail;
    split( path, head, tail);

    // handle structs via field name
    if( value_compound->get_kind() == IValue::VK_STRUCT) {
        const mi::mdl::IType_struct* type_struct = type ? as<mi::mdl::IType_struct>( type) : 0;
        ASSERT( M_SCENE, type_struct || !type);
        mi::base::Handle<const IValue_struct> value_struct(
            value_compound->get_interface<IValue_struct>());
        const mi::mdl::IType* tail_type = get_field_type( type_struct, head.c_str());
        ASSERT( M_SCENE, tail_type || !type);
        const IValue* tail_value = value_struct->get_field( head.c_str());
        return lookup_sub_value( tail_type, tail_value, tail.c_str(), sub_type);
    }

    // handle other compounds via index
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( head);
    if( !index_likely.get_status()) {
        if( sub_type) *sub_type = 0;
        return 0;
    }
    mi::Size index = *index_likely.get_ptr();
    const mi::mdl::IType_compound* type_compound = type ? as<mi::mdl::IType_compound>( type) : 0;
    ASSERT( M_SCENE, type_compound || !type);
    const mi::mdl::IType* tail_type
        = type_compound ? type_compound->get_compound_type( static_cast<mi::Uint32>( index)) : 0;
    ASSERT( M_SCENE, tail_type || !type);
    const IValue* tail_value = value_compound->get_value( index);
    return lookup_sub_value( tail_type, tail_value, tail.c_str(), sub_type);
}

const IExpression* lookup_sub_expression(
    DB::Transaction* transaction,
    const IExpression_factory* ef,
    const IExpression_list* temporaries,
    const mi::mdl::IType* type,
    const IExpression* expr,
    const char* path,
    const mi::mdl::IType** sub_type)
{
    ASSERT( M_SCENE, expr && path);
    ASSERT( M_SCENE, (!transaction && !type && !sub_type) || (transaction && type && sub_type));

    // resolve temporaries
    IExpression::Kind kind = expr->get_kind();
    if( temporaries && kind == IExpression::EK_TEMPORARY) {
        mi::base::Handle<const IExpression_temporary> expr_temporary(
            expr->get_interface<IExpression_temporary>());
        mi::Size index = expr_temporary->get_index();
        expr = temporaries->get_expression( index);
        if( !expr) {
            if( sub_type) *sub_type = 0;
            return 0;
        }
        expr->release(); // rely on refcount in temporaries
        // type is unchanged
        kind = expr->get_kind();
    }

    // handle empty paths
    if (path[0] == '\0') {
        if (sub_type) *sub_type = type;
        expr->retain();
        return expr;
    }

    switch (kind) {

        case IExpression::EK_CONSTANT: {

            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            mi::base::Handle<const IValue> result(
                lookup_sub_value( type, value.get(), path, sub_type));
            return ef->create_constant( const_cast<IValue*>( result.get()));
        }

        case IExpression::EK_DIRECT_CALL: {

            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IExpression_list> arguments(
                expr_direct_call->get_arguments());

            std::string head, tail;
            split( path, head, tail);
            mi::Size index = arguments->get_index( head.c_str());
            if( index == static_cast<mi::Size>( -1)) {
                if( sub_type) *sub_type = 0;
                return 0;
            }

            const mi::mdl::IType* tail_type = 0;
            if( sub_type && transaction && type) {
                DB::Tag tag = expr_direct_call->get_definition();
                DB::Access<Mdl_function_definition> definition( tag, transaction);
                mi::mdl::IDefinition::Semantics sema = definition->get_mdl_semantic();
                if( sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
                    const mi::mdl::IType_array* type_array
                        = mi::mdl::as<mi::mdl::IType_array>( type);
                    tail_type = type_array->get_element_type();
                } else
                    tail_type = definition->get_mdl_parameter_type(
                        transaction, static_cast<mi::Uint32>( index));
                ASSERT( M_SCENE, tail_type);
            }

            mi::base::Handle<const IExpression> argument( arguments->get_expression( index));

            return lookup_sub_expression(
                transaction, ef, temporaries, tail_type, argument.get(), tail.c_str(), sub_type);
        }

        case IExpression::EK_CALL:
        case IExpression::EK_PARAMETER:
        case IExpression::EK_TEMPORARY:
            ASSERT( M_SCENE, false);
            return 0;

        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}


// **********  Resource-related attributes *********************************************************

bool get_texture_attributes(
    DB::Transaction* transaction,
    const mi::mdl::IValue* texture,
    bool& valid,
    bool& is_uvtile,
    int& width,
    int& height,
    int& depth)
{
    valid = false;
    is_uvtile = false;
    width = height = depth = 0;
    if( texture->get_kind() == mi::mdl::IValue::VK_INVALID_REF)
        return true;
    const mi::mdl::IValue_texture* t = as<mi::mdl::IValue_texture>( texture);
    if( !t)
        return false;
    DB::Tag tag = DB::Tag( t->get_tag_value());
#if 1
    // TODO: tag == 0 is forbidden, but neuray uses so far the set_resource_tag() function
    // which modifies a value. If a texture could not be loaded (because an image plugin is
    // missing) this leads to a zero tag. Hence handle this like a invalid resource
    if( !tag)
        return true;
#endif
    if( !tag || transaction->get_class_id( tag) != TEXTURE::ID_TEXTURE)
        return false;
    DB::Access<TEXTURE::Texture> db_texture( tag, transaction);
    tag = db_texture->get_image();
    if( !tag || transaction->get_class_id( tag) != DBIMAGE::ID_IMAGE)
        return false;
    DB::Access<DBIMAGE::Image> db_image( tag, transaction);
    mi::base::Handle<const IMAGE::IMipmap> mipmap( db_image->get_mipmap());
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
        mipmap->get_level( 0));
    valid     = true;
    is_uvtile = db_image->is_uvtile();
    width     = canvas->get_resolution_x();
    height    = canvas->get_resolution_y();
    depth     = canvas->get_layers_size();
    return true;
}

bool get_texture_uvtile_resolution(
    DB::Transaction* transaction,
    const mi::mdl::IValue* texture,
    mi::Sint32_2 const &uv_tile,
    int &width,
    int &height)
{
    width = height = 0;
    if( texture->get_kind() == mi::mdl::IValue::VK_INVALID_REF)
        return true;
    const mi::mdl::IValue_texture* t = as<mi::mdl::IValue_texture>( texture);
    if( !t)
        return false;
    DB::Tag tag = DB::Tag( t->get_tag_value());
#if 1
    // TODO: tag == 0 is forbidden, but neuray uses so far the set_resource_tag() function
    // which modifies a value. If a texture could not be loaded (because an image plugin is
    // missing) this leads to a zero tag. Hence handle this like a invalid resource
    if( !tag)
        return true;
#endif
    if( !tag || transaction->get_class_id( tag) != TEXTURE::ID_TEXTURE)
        return false;
    DB::Access<TEXTURE::Texture> db_texture( tag, transaction);
    tag = db_texture->get_image();
    if( !tag || transaction->get_class_id( tag) != DBIMAGE::ID_IMAGE)
        return false;
    DB::Access<DBIMAGE::Image> db_image( tag, transaction);
    if (!db_image->is_uvtile())
        return false;
    mi::Uint32 uvtile_id = db_image->get_uvtile_id( uv_tile.x, uv_tile.y);
    if (uvtile_id == mi::Uint32( -1)) {
        // return zero for invalid tiles
        width = height = 0;
        return true;
    }

    mi::base::Handle<const IMAGE::IMipmap> mipmap( db_image->get_mipmap( uvtile_id));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
        mipmap->get_level( 0));
    width = canvas->get_resolution_x();
    height = canvas->get_resolution_y();
    return true;
}

bool get_light_profile_attributes(
    DB::Transaction* transaction,
    const mi::mdl::IValue* light_profile,
    bool& valid,
    float& power,
    float& maximum)
{
    valid = false;
    power = maximum = 0.0f;
    if( light_profile->get_kind() == mi::mdl::IValue::VK_INVALID_REF)
        return true;
    const mi::mdl::IValue_light_profile* lp = as<mi::mdl::IValue_light_profile>( light_profile);
    if( !lp)
        return false;
    DB::Tag tag = DB::Tag( lp->get_tag_value());
    if( !tag || transaction->get_class_id( tag) != LIGHTPROFILE::ID_LIGHTPROFILE)
        return false;
    DB::Access<LIGHTPROFILE::Lightprofile> db_lightprofile( tag, transaction);
    valid   = db_lightprofile->is_valid();
    power   = db_lightprofile->get_power();
    maximum = db_lightprofile->get_maximum();
    return true;
}

bool get_bsdf_measurement_attributes(
    DB::Transaction* transaction,
    const mi::mdl::IValue* bsdf_measurement,
    bool& valid)
{
    valid = false;
    if( bsdf_measurement->get_kind() == mi::mdl::IValue::VK_INVALID_REF)
        return true;
    const mi::mdl::IValue_bsdf_measurement* bm =
        as<mi::mdl::IValue_bsdf_measurement>( bsdf_measurement);
    if( !bm)
        return false;
    DB::Tag tag = DB::Tag( bm->get_tag_value());
    if( !tag || transaction->get_class_id( tag) != BSDFM::ID_BSDF_MEASUREMENT)
        return false;
    DB::Access<BSDFM::Bsdf_measurement> db_bsdf_measurement( tag, transaction);
    valid = db_bsdf_measurement->is_valid();
    return true;
}


// **********  Stack_guard *************************************************************************

/// Helper to maintain a call stack and check for cycles.
class Call_stack_guard
{
public:
    /// Adds \p frame to the end of the call stack.
    Call_stack_guard(std::set<MI::Uint32>& call_trace, MI::Uint32 frame)
    : m_call_trace(call_trace)
    , m_frame(frame)
    , m_has_cycle(!call_trace.insert(frame).second)
    {
    }

    /// Removes the last frame from the call stack again.
    ~Call_stack_guard()
    {
        m_call_trace.erase(m_frame);
    }

    /// Checks whether the last frame in the call stack creates a cycle.
    bool last_frame_creates_cycle() const
    {
        return m_has_cycle;
    }

private:
    std::set<MI::Uint32> &m_call_trace;
    MI::Uint32 m_frame;
    bool m_has_cycle;
};

// **********  Mdl_dag_builder *********************************************************************

template <class T>
Mdl_dag_builder<T>::Mdl_dag_builder(
    DB::Transaction* transaction,
    T* dag_builder,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max,
    const Mdl_compiled_material* compiled_material)
  : m_transaction( transaction),
    m_dag_builder( dag_builder),
    m_mdl_meters_per_scene_unit( mdl_meters_per_scene_unit),
    m_mdl_wavelength_min( mdl_wavelength_min),
    m_mdl_wavelength_max( mdl_wavelength_max),
    m_type_factory( dag_builder->get_type_factory()),
    m_value_factory( dag_builder->get_value_factory()),
    m_compiled_material( compiled_material),
    m_temporaries(),
    m_parameter_types(),
    m_call_trace()
{
    if (compiled_material != NULL) {
        m_temporaries.resize(
            compiled_material->get_temporary_count(), (mi::mdl::DAG_node const *)NULL);
    }
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type, const IExpression* expr)
{
    IExpression::Kind kind = expr->get_kind();

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            return int_expr_constant_to_mdl_dag_node( mdl_type, expr_constant.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            return int_expr_call_to_mdl_dag_node( mdl_type, expr_call.get());
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IExpression_parameter> expr_param(
                expr->get_interface<IExpression_parameter>());
            return int_expr_parameter_to_mdl_dag_node( mdl_type, expr_param.get());
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            return int_expr_direct_call_to_mdl_dag_node(
                mdl_type, expr_direct_call.get());
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IExpression_temporary> expr_temporary(
                expr->get_interface<IExpression_temporary>());
            return int_expr_temporary_to_mdl_dag_node(
                mdl_type, expr_temporary.get());
        }
        default:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_constant_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type, const IExpression_constant* expr)
{
    mi::base::Handle<const IValue> value( expr->get_value());
    const mi::mdl::IValue* mdl_value = int_value_to_mdl_value(
        m_transaction, m_value_factory, mdl_type, value.get());
    if( !mdl_value)
        return 0;
    return m_dag_builder->create_constant( mdl_value);
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::clone_dag_node(
    const mi::mdl::DAG_node* node)
{
    switch( node->get_kind()) {
        case mi::mdl::DAG_node::EK_CONSTANT: {
            const mi::mdl::DAG_constant* constant = cast<mi::mdl::DAG_constant>( node);
            const mi::mdl::IValue* value = constant->get_value();
            value = m_value_factory->import( value);
            return m_dag_builder->create_constant( value);
        }
        case mi::mdl::DAG_node::EK_CALL: {
            const mi::mdl::DAG_call* call = cast<mi::mdl::DAG_call>( node);
            const mi::mdl::IType* return_type = call->get_type();
            return_type = m_type_factory->import( return_type);
            mi::Uint32 n = call->get_argument_count();
            Small_VLA<mi::mdl::DAG_call::Call_argument, 8> arguments( n);
            for( mi::Uint32 i = 0; i < n; ++i) {
                const mi::mdl::DAG_node* argument = call->get_argument( i);
                arguments[i].arg        = clone_dag_node( argument);
                arguments[i].param_name = call->get_parameter_name( i);
            }
            return m_dag_builder->create_call(
                call->get_name(),
                call->get_semantic(),
                arguments.data(),
                arguments.size(),
                return_type);
        }
        case mi::mdl::DAG_node::EK_TEMPORARY:
        case mi::mdl::DAG_node::EK_PARAMETER:
            break;
    }
    ASSERT( M_SCENE, false);
    return 0;
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_call_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const IExpression_call* expr)
{
    DB::Tag tag = expr->get_call();

    Call_stack_guard guard( m_call_trace, tag.get_uint());
    if( guard.last_frame_creates_cycle())
        return 0;

    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);

    // handle material instances
    if( class_id == Mdl_material_instance::id) {
        DB::Access<Mdl_material_instance> material_instance( tag, m_transaction);
        DB::Access<Mdl_material_definition> definition(
            material_instance->get_material_definition(), m_transaction);
        mi::base::Handle<const IExpression_list> arguments(material_instance->get_arguments());
        const char* call_name = m_transaction->tag_to_name(tag);
        const char* definition_mdl_name = definition->get_mdl_name();

        return Mdl_dag_builder<T>::int_expr_list_to_mdl_dag_node(
            mdl_type, definition.get_ptr(), definition_mdl_name, call_name, arguments.get());
    }

    // handle non-function calls
    if( class_id != Mdl_function_call::id) {
        const char* name = m_transaction->tag_to_name( tag);
        ASSERT( M_SCENE, name);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Unsupported type for call of \"%s\".", name?name:"");
        return 0;
    }

    // handle function calls

    DB::Access<Mdl_function_call> call( tag, m_transaction);
    DB::Access<Mdl_function_definition> definition( call->get_function_definition(), m_transaction);
    mi::base::Handle<const IExpression_list> arguments( call->get_arguments());
    const char* call_name = m_transaction->tag_to_name( tag);
    const char* definition_mdl_name = definition->get_mdl_name();

    return Mdl_dag_builder<T>::int_expr_list_to_mdl_dag_node(
        mdl_type, definition.get_ptr(), definition_mdl_name, call_name, arguments.get());
}

template <class T>
const mi::mdl::DAG_node*
Mdl_dag_builder<T>::int_expr_parameter_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const IExpression_parameter* expr)
{
    // Check whether we need to adjust a deferred-size libmdl type to the type of the parameter
    if (const mi::mdl::IType_compound* mdl_type_compound = as<mi::mdl::IType_compound>(mdl_type)) {
        mi::base::Handle<const IType> expr_type(expr->get_type());
        mi::base::Handle<const IType_compound> expr_type_compound(
            expr_type->get_interface<IType_compound>());
        ASSERT( M_SCENE, expr_type_compound != NULL);
        mi::Size n = expr_type_compound->get_size();
        mdl_type = convert_deferred_sized_into_immediate_sized_array(
            m_value_factory, mdl_type_compound, n);
    }

    mi::Size index = expr->get_index();

    if( index >= m_parameter_types.size() || !m_parameter_types[index]) {
        if( index >= m_parameter_types.size())
            m_parameter_types.resize( index+1);
        m_parameter_types[index] = mdl_type;
    }

    ASSERT( M_SCENE, index < m_parameter_types.size() && m_parameter_types[index]);
    ASSERT( M_SCENE, m_parameter_types[index] == mdl_type);
    return m_dag_builder->create_parameter( mdl_type, int( index));
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_direct_call_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const IExpression_direct_call* expr)
{
    DB::Tag tag = expr->get_definition();

    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);

    // handle material instances
    if (class_id == Mdl_material_definition::id) {
        DB::Access<Mdl_material_definition> definition( tag, m_transaction);
        mi::base::Handle<const IExpression_list> arguments( expr->get_arguments());
        const char* call_name = m_transaction->tag_to_name( tag);
        const char* definition_mdl_name = definition->get_mdl_name();

        return Mdl_dag_builder<T>::int_expr_list_to_mdl_dag_node(
            mdl_type, definition.get_ptr(), definition_mdl_name, call_name, arguments.get());
    }

    // handle non-function calls
    DB::Access<Mdl_function_definition> definition( tag, m_transaction);
    mi::base::Handle<const IExpression_list> arguments( expr->get_arguments());
    const char* definition_mdl_name = definition->get_mdl_name();

    return Mdl_dag_builder<T>::int_expr_list_to_mdl_dag_node(
        mdl_type, definition.get_ptr(), definition_mdl_name, 0, arguments.get());
}

template <>
const mi::mdl::DAG_node* Mdl_dag_builder<mi::mdl::IDag_builder>::int_expr_temporary_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const IExpression_temporary* expr)
{
    mi::Size index = expr->get_index();

    if( index >= m_temporaries.size() || !m_temporaries[index]) {
        ASSERT( M_SCENE, m_compiled_material);
        mi::base::Handle<const IExpression> referenced_expr(
            m_compiled_material->get_temporary( index));
        ASSERT( M_SCENE, referenced_expr);
        const mi::mdl::DAG_node* result
            = int_expr_to_mdl_dag_node( mdl_type, referenced_expr.get());
        if( !result)
            return 0;
        if( index >= m_temporaries.size())
            m_temporaries.resize( index+1);
        m_temporaries[index] = result;
    }
    // IDag_builder has no create_temporary(), but we don't need to create it again, we *know*
    // it is the same code
    return m_temporaries[index];
}

template <>
const mi::mdl::DAG_node*
Mdl_dag_builder<mi::mdl::IGenerated_code_dag::DAG_node_factory>::int_expr_temporary_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const IExpression_temporary* expr)
{
    mi::Size index = expr->get_index();

    if( index >= m_temporaries.size() || !m_temporaries[index]) {
        ASSERT( M_SCENE, m_compiled_material);
        mi::base::Handle<const IExpression> referenced_expr(
            m_compiled_material->get_temporary( index));
        ASSERT( M_SCENE, referenced_expr);
        const mi::mdl::DAG_node* result
            = int_expr_to_mdl_dag_node( mdl_type, referenced_expr.get());
        if( !result)
            return 0;
        if( index >= m_temporaries.size())
            m_temporaries.resize( index+1);
        m_temporaries[index] = result;
    }

    ASSERT( M_SCENE, index < m_temporaries.size() && m_temporaries[index]);
    // re-create the temporary
    return m_dag_builder->create_temporary( m_temporaries[index], int( index));
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_list_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const Mdl_function_definition* function_definition,
    const char* function_definition_mdl_name,
    const char* function_call_name,
    const IExpression_list* arguments)
{
    DETAIL::Type_binder type_binder( m_type_factory);

    mi::mdl::IDefinition::Semantics semantic = function_definition->get_mdl_semantic();

    const mi::mdl::IType* element_type = 0;
    if( semantic == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
        const mi::mdl::IType_array* mdl_type_array = mi::mdl::as<mi::mdl::IType_array>( mdl_type);
        element_type = mdl_type_array->get_element_type();
    }

    mi::Size n = arguments->get_size();
    Small_VLA<mi::mdl::DAG_call::Call_argument, 8> mdl_arguments( n);

    for( mi::Size i = 0; i < n; ++i) {

        mdl_arguments[i].param_name = arguments->get_name( i);

        const mi::mdl::IType* parameter_type = 0;
        if( semantic == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
            parameter_type = element_type;
        else
            parameter_type = function_definition->get_mdl_parameter_type(
                m_transaction, static_cast<mi::Uint32>( i));
        parameter_type = m_type_factory->import( parameter_type->skip_type_alias());

        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        mdl_arguments[i].arg = int_expr_to_mdl_dag_node( parameter_type, argument.get());
        if( !mdl_arguments[i].arg)
            return 0;

        const mi::mdl::IType* argument_type = mdl_arguments[i].arg->get_type();
        mi::Sint32 result = type_binder.check_and_bind_type( parameter_type, argument_type);
        switch( result) {
            case 0:
                // nothing to do
                break;
            case -1: {
                const std::string& function_definition_name
                    = add_mdl_db_prefix( function_definition_mdl_name);
                const char* s1 = function_call_name ? "" : "of definition ";
                const char* s2
                    = function_call_name ? function_call_name : function_definition_name.c_str();
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Type mismatch for argument \"%s\" of function call %s\"%s\".",
                    mdl_arguments[i].param_name, s1, s2);
                return 0;
            }
            case -2: {
                const std::string& function_definition_name
                    = add_mdl_db_prefix( function_definition_mdl_name);
                const char* s1 = function_call_name ? "" : "of definition ";
                const char* s2
                    = function_call_name ? function_call_name : function_definition_name.c_str();
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Array size mismatch for argument \"%s\" of function call %s\"%s\".",
                    mdl_arguments[i].param_name, s1, s2);
                return 0;
            }
            default:
                ASSERT( M_SCENE, false);
                return 0;
        }
    }

    const mi::mdl::IType* return_type = 0;
    if( semantic == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
        element_type = m_type_factory->import( element_type);
        return_type = m_type_factory->create_array( element_type, mdl_arguments.size());
    } else {
        return_type = function_definition->get_mdl_return_type( m_transaction);
        return_type = m_type_factory->import( return_type);
    }

    const mi::mdl::IType_array* return_type_array = as<mi::mdl::IType_array>( return_type);
    if( return_type_array && !return_type_array->is_immediate_sized()) {
        const mi::mdl::IType* bound_return_type = type_binder.get_bound_type( return_type_array);
        if( bound_return_type)
            return_type = bound_return_type;
    }

    return m_dag_builder->create_call(
        function_definition_mdl_name,
        semantic,
        mdl_arguments.data(),
        static_cast<mi::Uint32>( mdl_arguments.size()),
        return_type);
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_list_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const Mdl_material_definition* material_definition,
    const char* material_definition_mdl_name,
    const char* material_call_name,
    const IExpression_list* arguments)
{
    DETAIL::Type_binder type_binder(m_type_factory);

    mi::Size n = arguments->get_size();
    Small_VLA<mi::mdl::DAG_call::Call_argument, 8> mdl_arguments(n);

    for (mi::Size i = 0; i < n; ++i) {

        mdl_arguments[i].param_name = arguments->get_name(i);

        const mi::mdl::IType* parameter_type = material_definition->get_mdl_parameter_type(
                m_transaction, static_cast<mi::Uint32>(i));
        parameter_type = m_type_factory->import(parameter_type->skip_type_alias());

        mi::base::Handle<const IExpression> argument(arguments->get_expression(i));
        mdl_arguments[i].arg = int_expr_to_mdl_dag_node(parameter_type, argument.get());
        if (!mdl_arguments[i].arg)
            return 0;

        const mi::mdl::IType* argument_type = mdl_arguments[i].arg->get_type();
        mi::Sint32 result = type_binder.check_and_bind_type(parameter_type, argument_type);
        switch (result) {
        case 0:
            // nothing to do
            break;
        case -1: {
                const std::string& material_definition_name
                    = add_mdl_db_prefix(material_definition_mdl_name);
                const char* s1 = material_call_name ? "" : "of definition ";
                const char* s2
                    = material_call_name ? material_call_name : material_definition_name.c_str();
                LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Type mismatch for argument \"%s\" of function call %s\"%s\".",
                    mdl_arguments[i].param_name, s1, s2);
                return 0;
            }
        case -2: {
                const std::string& material_definition_name
                    = add_mdl_db_prefix(material_definition_mdl_name);
                const char* s1 = material_call_name ? "" : "of definition ";
                const char* s2
                    = material_call_name ? material_call_name : material_definition_name.c_str();
                LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Array size mismatch for argument \"%s\" of function call %s\"%s\".",
                    mdl_arguments[i].param_name, s1, s2);
                return 0;
            }
        default:
            ASSERT(M_SCENE, false);
            return 0;
        }
    }

    // materials are always of type material
    const mi::mdl::IType* return_type =
        m_type_factory->get_predefined_struct(mi::mdl::IType_struct::SID_MATERIAL);

    return m_dag_builder->create_call(
        material_definition_mdl_name,
        mi::mdl::IDefinition::DS_UNKNOWN,
        mdl_arguments.data(),
        static_cast<mi::Uint32>(mdl_arguments.size()),
        return_type);
}

template class Mdl_dag_builder<mi::mdl::IDag_builder>;
template class Mdl_dag_builder<mi::mdl::IGenerated_code_dag::DAG_node_factory>;


// **********  Mdl_material_instance_builder *******************************************************

mi::mdl::IGenerated_code_dag::IMaterial_instance*
Mdl_material_instance_builder::create_material_instance(
    DB::Transaction* transaction, const Mdl_compiled_material* cm)
{
    // get IMDL interface and allocator
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    mi::base::Handle<mi::base::IAllocator> allocator(mdl->get_mdl_allocator());

    // find index of material elemental constructor
    DB::Tag module_tag = transaction->name_to_tag( "mdl::<builtins>");
    DB::Access<Mdl_module> builtins_module( module_tag, transaction);
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> builtins_code_dag(
        builtins_module->get_code_dag());
    mi::Size n = builtins_code_dag->get_function_count();
    mi::Size constructor_index = 0;
    for( ; constructor_index < n; ++constructor_index) {
        mi::mdl::IDefinition::Semantics semantic
            = builtins_code_dag->get_function_semantics( int( constructor_index));
        if( semantic != mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR)
            continue;
        const mi::mdl::IType_struct* return_type = as<mi::mdl::IType_struct>(
            builtins_code_dag->get_function_return_type( int( constructor_index)));
        if( !return_type)
            continue;
        mi::mdl::IType_struct::Predefined_id id = return_type->get_predefined_id();
        if( id != mi::mdl::IType_struct::SID_MATERIAL)
            continue;
        break;
    }
    ASSERT( M_SCENE, constructor_index < n);
    if( constructor_index >= n)
        return 0;

    // access material elemental constructor
    DB::Tag fd_tag = builtins_module->get_function( constructor_index);
    if( !fd_tag)
        return 0;
    DB::Access<Mdl_function_definition> fd( fd_tag, transaction);

    // get the DAG code generator and the internal_space option
    mi::base::Handle<mi::mdl::ICode_generator_dag> generator_dag
        = mi::base::make_handle( mdl->load_code_generator( "dag"))
            .get_interface<mi::mdl::ICode_generator_dag>();
    mi::mdl::Options& option = generator_dag->access_options();
    int option_index = option.get_option_index( MDL_CG_OPTION_INTERNAL_SPACE);
    ASSERT( M_SCENE, option_index >= 0);
    const char* internal_space = option.get_option_value( option_index);

    // create Generated_code_dag::Material_instance
    mi::mdl::Allocator_builder allocator_builder(allocator.get());
    mi::base::Handle<mi::mdl::Generated_code_dag::Material_instance> mi(
        allocator_builder.create<mi::mdl::Generated_code_dag::Material_instance>(
            mdl.get(), allocator.get(), int( constructor_index), internal_space));
    ASSERT( M_SCENE, mi);

    // create builder to convert fields and temporaries of compiled material
    Mdl_dag_builder<mi::mdl::IGenerated_code_dag::DAG_node_factory> dag_builder(
        transaction,
        &mi->m_node_factory,
        cm->get_mdl_meters_per_scene_unit(),
        cm->get_mdl_wavelength_min(),
        cm->get_mdl_wavelength_max(),
        cm);

    // convert material body expression and set as constructor
    const mi::mdl::IType* mdl_return_type = fd->get_mdl_return_type( transaction);
    mi::base::Handle<const IExpression_direct_call> call( cm->get_body());
    const mi::mdl::DAG_node* call_node
        = dag_builder.int_expr_to_mdl_dag_node( mdl_return_type, call.get());
    ASSERT( M_SCENE, call_node);
    if( !call_node)
        return 0;
    mi->m_constructor = mi::mdl::as<mi::mdl::DAG_call>( call_node);

    // set temporaries
    const std::vector<const mi::mdl::DAG_node*>& temporaries = dag_builder.get_temporaries();
    n = temporaries.size();
    mi->m_temporaries.resize( n);
    for( mi::Size i = 0; i < n; ++i)
        mi->m_temporaries[i] = temporaries[i];

    // set parameter names/arguments
    mi::mdl::IValue_factory* vf = &mi->m_value_factory;
    mi::base::Handle<const IValue_list> arguments( cm->get_arguments());
    const std::vector<const mi::mdl::IType*>& parameter_types = dag_builder.get_parameter_types();
    n = parameter_types.size();
    mi->m_default_param_values.resize( n);
    mi->m_param_names.resize( n, mi::mdl::string( allocator.get()));
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IValue> argument( arguments->get_value( i));
        const mi::mdl::IValue* mdl_argument
            = int_value_to_mdl_value( transaction, vf, parameter_types[i], argument.get());
        ASSERT( M_SCENE, mdl_argument);
        mi->m_default_param_values[i] = mdl_argument;
        mi->m_param_names[i] = arguments->get_name( i);
    }

    // set properties
    mi->m_properties = cm->get_properties();

    mi->calc_hashes();

    mi->retain();
    return mi.get();
};


// **********  Mdl_call_resolver *******************************************************************

Mdl_call_resolver::~Mdl_call_resolver()
{
    // drop all import entries again
    Module_set::const_iterator end = m_resolved_modules.end();
    for( Module_set::const_iterator it = m_resolved_modules.begin(); it != end; ++it) {
        (*it)->drop_import_entries();
        (*it)->release();
    }
}

const mi::mdl::IModule* Mdl_call_resolver::get_owner_module( char const* name) const
{
    std::string db_name = add_mdl_db_prefix(name);
    DB::Tag tag = m_transaction->name_to_tag(db_name.c_str());
    if (!tag)
        return NULL;

    DB::Tag module_tag;

    SERIAL::Class_id id = m_transaction->get_class_id(tag);
    if (id == Mdl_function_definition::id) {
        DB::Access<Mdl_function_definition> material_definition(tag, m_transaction);
        module_tag = material_definition->get_module(m_transaction);
    } else if (id == Mdl_material_definition::id) {
        DB::Access<Mdl_material_definition> material_definition(tag, m_transaction);
        module_tag = material_definition->get_module(m_transaction);
    } else {
        return NULL;
    }

    if (!module_tag)
        return NULL;

    if (m_transaction->get_class_id(module_tag) != Mdl_module::id)
        return NULL;

    DB::Access<Mdl_module> mdl_module(module_tag, m_transaction);
    const mi::mdl::IModule *module = mdl_module->get_mdl_module();

    if (module != NULL) {
        // ensure that all import entries are restored before the module is returned
        if (m_resolved_modules.find(module) == m_resolved_modules.end()) {
            module->retain();
            m_resolved_modules.insert(module);
            Module_cache cache(m_transaction);
            module->restore_import_entries(&cache);
        }
    }
    return module;
}


// ********** Conversion from mi::mdl to MI::MDL ***************************************************

namespace {

const IType_enum* create_enum(
    IType_factory* tf,
    const char* symbol,
    const mi::mdl::IType_enum* type_enum,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* value_annotations)
{
    mi::mdl::IType_enum::Predefined_id id_mdl = type_enum->get_predefined_id();
    IType_enum::Predefined_id id_int = DETAIL::mdl_enum_id_to_int_enum_id( id_mdl);

    mi::Uint32 count = type_enum->get_value_count();
    IType_enum::Values values( count);

    for( mi::Uint32 i = 0; i < count; ++i) {
        const mi::mdl::ISymbol* symbol;
        int code;
        type_enum->get_value( i, symbol, code);
        values[i] = std::make_pair( symbol->get_name(), code);
    }

    IExpression_factory* ef = get_expression_factory();
    Mdl_dag_converter converter(
        ef,
        /*transaction*/ 0,
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        /*module_filename*/ 0,
        /*module_name*/ 0,
        /*prototype_tag*/ DB::Tag());

    mi::base::Handle<const IAnnotation_block> annotations_int(
        annotations ? converter.mdl_dag_node_vector_to_int_annotation_block(
            *annotations, /*qualified_name*/ symbol) : 0);

    count = value_annotations ? value_annotations->size() : 0;
    IType_enum::Value_annotations value_annotations_int( count);

    for (mi::Uint32 i = 0; i < count; ++i)
        value_annotations_int[i] = converter.mdl_dag_node_vector_to_int_annotation_block(
            (*value_annotations)[i], /*qualified_name*/ symbol);

    mi::Sint32 errors;
    const IType_enum* type_enum_int = tf->create_enum(
        symbol, id_int, values, annotations_int, value_annotations_int, &errors);
    ASSERT( M_SCENE, errors == 0);
    ASSERT( M_SCENE, type_enum_int);
    return type_enum_int;
}

const IType_struct* create_struct(
    IType_factory* tf,
    const char* symbol,
    const mi::mdl::IType_struct* type_struct,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* field_annotations)
{
    mi::mdl::IType_struct::Predefined_id id_mdl = type_struct->get_predefined_id();
    IType_struct::Predefined_id id_int = DETAIL::mdl_struct_id_to_int_struct_id( id_mdl);

    mi::Uint32 count = type_struct->get_compound_size();
    IType_struct::Fields fields( count);

    for( mi::Uint32 i = 0; i < count; ++i) {
        const mi::mdl::IType* type_mdl;
        const mi::mdl::ISymbol* symbol;
        type_struct->get_field( i, type_mdl, symbol);
        mi::base::Handle<const IType> type_int( mdl_type_to_int_type( tf, type_mdl));
        fields[i] = std::make_pair( type_int, symbol->get_name());
    }

    IExpression_factory* ef = get_expression_factory();
    Mdl_dag_converter converter(
        ef,
        /*transaction*/ 0,
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        /*module_filename*/ 0,
        /*module_name*/ 0,
        /*prototype_tag*/ DB::Tag());

    mi::base::Handle<const IAnnotation_block> annotations_int(
        annotations ? converter.mdl_dag_node_vector_to_int_annotation_block(
            *annotations, /*qualified_name*/ symbol) : 0);

    count = field_annotations ? field_annotations->size() : 0;
    IType_enum::Value_annotations field_annotations_int( count);

    for (mi::Uint32 i = 0; i < count; ++i)
        field_annotations_int[i] = converter.mdl_dag_node_vector_to_int_annotation_block(
            (*field_annotations)[i], /*qualified_name*/ symbol);

    mi::Sint32 errors;
    const IType_struct* type_int = tf->create_struct(
        symbol, id_int, fields, annotations_int, field_annotations_int, &errors);
    ASSERT( M_SCENE, errors == 0);
    ASSERT( M_SCENE, type_int);
    return type_int;
}

} // anonymous namespace

const IType* mdl_type_to_int_type(
    IType_factory* tf,
    const mi::mdl::IType* type,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* member_annotations)
{
    mi::mdl::IType::Kind kind = type->get_kind();

    bool enum_or_struct = kind == mi::mdl::IType::TK_ENUM || kind == mi::mdl::IType::TK_STRUCT;
    ASSERT( M_SCENE, enum_or_struct || !annotations);
    ASSERT( M_SCENE, enum_or_struct || !member_annotations);
    boost::ignore_unused( enum_or_struct);

    switch( kind) {
        case mi::mdl::IType::TK_BOOL:             return tf->create_bool();
        case mi::mdl::IType::TK_INT:              return tf->create_int();
        case mi::mdl::IType::TK_FLOAT:            return tf->create_float();
        case mi::mdl::IType::TK_DOUBLE:           return tf->create_double();
        case mi::mdl::IType::TK_STRING:           return tf->create_string();
        case mi::mdl::IType::TK_COLOR:            return tf->create_color();
        case mi::mdl::IType::TK_LIGHT_PROFILE:    return tf->create_light_profile();
        case mi::mdl::IType::TK_BSDF_MEASUREMENT: return tf->create_bsdf_measurement();
        case mi::mdl::IType::TK_BSDF:             return tf->create_bsdf();
        case mi::mdl::IType::TK_EDF:              return tf->create_edf();
        case mi::mdl::IType::TK_VDF:              return tf->create_vdf();

        case mi::mdl::IType::TK_ALIAS: {
            const mi::mdl::IType_alias* type_alias = cast<mi::mdl::IType_alias>( type);
            const mi::mdl::IType* aliased_type = type_alias->get_aliased_type();
            mi::base::Handle<const IType> aliased_type_int(
                mdl_type_to_int_type( tf, aliased_type));
            mi::Uint32 modifiers = type_alias->get_type_modifiers();
            mi::Uint32 modifiers_int = DETAIL::mdl_modifiers_to_int_modifiers( modifiers);
            const mi::mdl::ISymbol* symbol = type_alias->get_symbol();
            const char* symbol_name = symbol ? symbol->get_name() : 0;
            return tf->create_alias( aliased_type_int.get(), modifiers_int, symbol_name);
        }

        case mi::mdl::IType::TK_ENUM: {
            const mi::mdl::IType_enum* type_enum = cast<mi::mdl::IType_enum>( type);
            const mi::mdl::ISymbol* symbol = type_enum->get_symbol();
            const char* symbol_name = symbol->get_name();
            ASSERT( M_SCENE, symbol_name);
            std::string prefixed_symbol_name = prefix_symbol_name( symbol_name);
            symbol_name = symbol_name ? prefixed_symbol_name.c_str() : 0;
            return create_enum( tf, symbol_name, type_enum, annotations, member_annotations);
        }

        case mi::mdl::IType::TK_VECTOR: {
            const mi::mdl::IType_vector* type_vector = cast<mi::mdl::IType_vector>( type);
            const mi::mdl::IType_atomic* element_type = type_vector->get_element_type();
            mi::base::Handle<const IType_atomic> element_type_int(
                mdl_type_to_int_type<IType_atomic>( tf, element_type));
            mi::Size size = type_vector->get_compound_size();
            return tf->create_vector( element_type_int.get(), size);
        }

        case mi::mdl::IType::TK_MATRIX: {
            const mi::mdl::IType_matrix* type_matrix = cast<mi::mdl::IType_matrix>( type);
            const mi::mdl::IType_vector* column_type = type_matrix->get_element_type();
            mi::base::Handle<const IType_vector> column_type_int(
                mdl_type_to_int_type<IType_vector>( tf, column_type));
            mi::Size columns = type_matrix->get_compound_size();
            return tf->create_matrix( column_type_int.get(), columns);
        }

        case mi::mdl::IType::TK_ARRAY: {
            const mi::mdl::IType_array* type_array = as<mi::mdl::IType_array>( type);
            const mi::mdl::IType* element_type = type_array->get_element_type();
            mi::base::Handle<const IType> element_type_int(
                mdl_type_to_int_type( tf, element_type));
            if( type_array->is_immediate_sized()) {
                mi::Size size = type_array->get_size();
                return tf->create_immediate_sized_array( element_type_int.get(), size);
            } else {
                const mi::mdl::ISymbol* size = type_array->get_deferred_size()->get_name();
                return tf->create_deferred_sized_array( element_type_int.get(), size->get_name());
            }
        }

        case mi::mdl::IType::TK_STRUCT: {
            const mi::mdl::IType_struct* type_struct = as<mi::mdl::IType_struct>( type);
            const mi::mdl::ISymbol* symbol = type_struct->get_symbol();
            const char* symbol_name = symbol->get_name();
            ASSERT( M_SCENE, symbol_name);
            std::string prefixed_symbol_name = prefix_symbol_name( symbol_name);
            symbol_name = symbol_name ? prefixed_symbol_name.c_str() : 0;
            return create_struct( tf, symbol_name, type_struct, annotations, member_annotations);
        }

        case mi::mdl::IType::TK_TEXTURE: {
            const mi::mdl::IType_texture* type_texture = as<mi::mdl::IType_texture>( type);
            mi::mdl::IType_texture::Shape shape = type_texture->get_shape();
            IType_texture::Shape shape_int = DETAIL::mdl_shape_to_int_shape( shape);
            return tf->create_texture( shape_int);
        }

        case mi::mdl::IType::TK_FUNCTION:
            ASSERT( M_SCENE, false); return 0;
        case mi::mdl::IType::TK_INCOMPLETE:
            ASSERT( M_SCENE, false); return 0;
        case mi::mdl::IType::TK_ERROR:
            ASSERT( M_SCENE, false); return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

IValue* mdl_value_to_int_value(
    IValue_factory* vf,
    DB::Transaction* transaction,
    const IType* type_int,
    const mi::mdl::IValue* value,
    const char* module_filename,
    const char* module_name)
{
    mi::mdl::IValue::Kind kind = value->get_kind();

    switch( kind) {
        case mi::mdl::IValue::VK_BAD:
            ASSERT( M_SCENE, false);
            return 0;
        case mi::mdl::IValue::VK_BOOL: {
            const mi::mdl::IValue_bool* value_bool = cast<mi::mdl::IValue_bool>( value);
            return vf->create_bool( value_bool->get_value());
        }
        case mi::mdl::IValue::VK_INT: {
            const mi::mdl::IValue_int* value_int = cast<mi::mdl::IValue_int>( value);
            return vf->create_int( value_int->get_value());
        }
        case mi::mdl::IValue::VK_ENUM: {
            const mi::mdl::IValue_enum* value_enum = cast<mi::mdl::IValue_enum>( value);
            const mi::mdl::IType_enum* type_enum = value_enum->get_type();
            mi::base::Handle<IType_factory> tf( vf->get_type_factory());
            mi::base::Handle<const IType_enum> type_enum_int(
                mdl_type_to_int_type<IType_enum>( tf.get(), type_enum));
            return vf->create_enum( type_enum_int.get(), value_enum->get_index());
        }
        case mi::mdl::IValue::VK_FLOAT: {
            const mi::mdl::IValue_float* value_float = cast<mi::mdl::IValue_float>( value);
            return vf->create_float( value_float->get_value());
        }
        case mi::mdl::IValue::VK_DOUBLE: {
            const mi::mdl::IValue_double* value_double = cast<mi::mdl::IValue_double>( value);
            return vf->create_double( value_double->get_value());
        }
        case mi::mdl::IValue::VK_STRING: {
            const mi::mdl::IValue_string* value_string = cast<mi::mdl::IValue_string>( value);
            return vf->create_string( value_string->get_value());
        }
        case mi::mdl::IValue::VK_VECTOR: {
            const mi::mdl::IValue_vector* value_vector = cast<mi::mdl::IValue_vector>( value);
            const mi::mdl::IType_vector* type_vector = value_vector->get_type();
            mi::base::Handle<IType_factory> tf( vf->get_type_factory());
            mi::base::Handle<const IType_vector> type_vector_int(
                mdl_type_to_int_type<IType_vector>( tf.get(), type_vector));
            IValue_vector* value_vector_int = vf->create_vector( type_vector_int.get());
            mi::Size n = value_vector->get_component_count();
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_vector->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    vf, transaction, /* type not relevant */ 0, component, module_filename,
                    module_name));
                mi::Sint32 result = value_vector_int->set_value( i, component_int.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);
            }
            return value_vector_int;
        }
        case mi::mdl::IValue::VK_MATRIX: {
            const mi::mdl::IValue_matrix* value_matrix = cast<mi::mdl::IValue_matrix>( value);
            const mi::mdl::IType_matrix* type_matrix = value_matrix->get_type();
            mi::base::Handle<IType_factory> tf( vf->get_type_factory());
            mi::base::Handle<const IType_matrix> type_matrix_int(
                mdl_type_to_int_type<IType_matrix>( tf.get(), type_matrix));
            IValue_matrix* value_matrix_int = vf->create_matrix( type_matrix_int.get());
            mi::Size n = value_matrix->get_component_count();
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_matrix->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    vf, transaction, /* type not relevant */ 0, component, module_filename,
                    module_name));
                mi::Sint32 result = value_matrix_int->set_value( i, component_int.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);
            }
            return value_matrix_int;
        }
        case mi::mdl::IValue::VK_ARRAY: {
            const mi::mdl::IValue_array* value_array = cast<mi::mdl::IValue_array>( value);
            mi::base::Handle<const IType_array> type_array_int;
            if( type_int) { // use provided type
                mi::base::Handle<const IType> type_aliased_int( type_int->skip_all_type_aliases());
                type_array_int = type_aliased_int->get_interface<IType_array>();
                ASSERT( M_SCENE, type_array_int);
            } else { // else compute it from the value
                const mi::mdl::IType_array* type_array = value_array->get_type();
                mi::base::Handle<IType_factory> tf( vf->get_type_factory());
                type_array_int = mdl_type_to_int_type<IType_array>( tf.get(), type_array);
            }
            IValue_array* value_array_int = vf->create_array( type_array_int.get());
            mi::Size n = value_array->get_component_count();
            if( !type_array_int->is_immediate_sized())
                value_array_int->set_size( n);
            mi::base::Handle<const IType> component_type_int( type_array_int->get_element_type());
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_array->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    vf, transaction, component_type_int.get(), component, module_filename,
                    module_name));
                mi::Sint32 result = value_array_int->set_value( i, component_int.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);
            }
            return value_array_int;
        }
        case mi::mdl::IValue::VK_RGB_COLOR: {
            const mi::mdl::IValue_rgb_color* value_rgb_color
                = cast<mi::mdl::IValue_rgb_color>( value);
            mi::Float32 red   = value_rgb_color->get_value( 0)->get_value();
            mi::Float32 green = value_rgb_color->get_value( 1)->get_value();
            mi::Float32 blue  = value_rgb_color->get_value( 2)->get_value();
            return vf->create_color( red, green, blue);
        }
        case mi::mdl::IValue::VK_STRUCT: {
            const mi::mdl::IValue_struct* value_struct = cast<mi::mdl::IValue_struct>( value);
            mi::base::Handle<const IType_struct> type_struct_int;
            if( type_int) { // use provided type
                mi::base::Handle<const IType> type_aliased_int( type_int->skip_all_type_aliases());
                type_struct_int = type_aliased_int->get_interface<IType_struct>();
                ASSERT( M_SCENE, type_struct_int);
            } else { // else compute it from the value
                const mi::mdl::IType_struct* type_struct = value_struct->get_type();
                mi::base::Handle<IType_factory> tf( vf->get_type_factory());
                type_struct_int = mdl_type_to_int_type<IType_struct>( tf.get(), type_struct);
            }
            IValue_struct* value_struct_int = vf->create_struct( type_struct_int.get());
            mi::Size n = value_struct->get_component_count();
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_struct->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<const IType> component_type_int(
                    type_struct_int->get_field_type( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    vf, transaction, component_type_int.get(), component, module_filename,
                    module_name));
                mi::Sint32 result = value_struct_int->set_value( i, component_int.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);
            }
            return value_struct_int;
        }
        case mi::mdl::IValue::VK_INVALID_REF: {
            const mi::mdl::IValue_invalid_ref* value_invalid_ref
                = cast<mi::mdl::IValue_invalid_ref>( value);
            const mi::mdl::IType_reference* type_reference
                = value_invalid_ref->get_type();
            const mi::mdl::IType_resource* type_resource
                = as<mi::mdl::IType_resource>( type_reference);
            if( type_resource) {
                const mi::mdl::IType_texture* type_texture
                    = as<mi::mdl::IType_texture>( type_resource);
                if( type_texture) {
                    mi::base::Handle<IType_factory> tf( vf->get_type_factory());
                    mi::base::Handle<const IType_texture> type_texture_int(
                        mdl_type_to_int_type<IType_texture>( tf.get(), type_texture));
                    return vf->create_texture( type_texture_int.get(), DB::Tag());
                }
                if( mi::mdl::is<mi::mdl::IType_light_profile>( type_resource))
                    return vf->create_light_profile( DB::Tag());
                if( mi::mdl::is<mi::mdl::IType_bsdf_measurement>( type_resource))
                    return vf->create_bsdf_measurement( DB::Tag());
                ASSERT( M_SCENE, false);
            }
            mi::base::Handle<IType_factory> tf( vf->get_type_factory());
            mi::base::Handle<const IType_reference> type_reference_int(
                mdl_type_to_int_type<IType_reference>( tf.get(), type_reference));
            return vf->create_invalid_df( type_reference_int.get());
        }
        case mi::mdl::IValue::VK_TEXTURE: {
            const mi::mdl::IValue_texture* value_texture
                = cast<mi::mdl::IValue_texture>( value);
            const mi::mdl::IType_texture* type_texture
                = value_texture->get_type();
            mi::base::Handle<IType_factory> tf( vf->get_type_factory());
            mi::base::Handle<const IType_texture> type_texture_int(
                mdl_type_to_int_type<IType_texture>( tf.get(), type_texture));
            DB::Tag tag = DETAIL::mdl_texture_to_tag(
                transaction, value_texture, module_filename, module_name);
            return vf->create_texture( type_texture_int.get(), tag);
        }
        case mi::mdl::IValue::VK_LIGHT_PROFILE: {
            const mi::mdl::IValue_light_profile* value_light_profile
                = cast<mi::mdl::IValue_light_profile>( value);
            DB::Tag tag = DETAIL::mdl_light_profile_to_tag(
                transaction, value_light_profile, module_filename, module_name);
            return vf->create_light_profile( tag);
        }
        case mi::mdl::IValue::VK_BSDF_MEASUREMENT: {
            const mi::mdl::IValue_bsdf_measurement* value_bsdf_measurement
                = cast<mi::mdl::IValue_bsdf_measurement>( value);
            DB::Tag tag = DETAIL::mdl_bsdf_measurement_to_tag(
                transaction, value_bsdf_measurement, module_filename, module_name);
            return vf->create_bsdf_measurement( tag);
        }
    }

    ASSERT( M_SCENE, false);
    return 0;
}

/// Converts mi::mdl::DAG_parameter to MI::MDL::IExpression
static IExpression* mdl_parameter_to_int_expr(
    IExpression_factory* ef,
    const mi::mdl::DAG_parameter* parameter)
{
    mi::base::Handle<IValue_factory> vf( ef->get_value_factory());
    mi::base::Handle<IType_factory> tf( vf->get_type_factory());
    mi::base::Handle<const IType> type( mdl_type_to_int_type( tf.get(), parameter->get_type()));
    mi::Size index = parameter->get_index();
    return ef->create_parameter( type.get(), index);
}

/// Converts mi::mdl::DAG_temporary to MI::MDL::IExpression
static IExpression* mdl_temporary_to_int_expr(
    IExpression_factory* ef,
    const mi::mdl::DAG_temporary* temporary)
{
    mi::base::Handle<IValue_factory> vf( ef->get_value_factory());
    mi::base::Handle<IType_factory> tf( vf->get_type_factory());
    mi::base::Handle<const IType> type( mdl_type_to_int_type( tf.get(), temporary->get_type()));
    mi::Size index = temporary->get_index();
    return ef->create_temporary( type.get(), index);
}

/// Converts mi::mdl::Dag_node to MI::MDL::IExpression/MI::MDL::IAnnotation.
Mdl_dag_converter::Mdl_dag_converter(
    IExpression_factory* ef,
    DB::Transaction* transaction,
    bool immutable_callees,
    bool create_direct_calls,
    const char* module_filename,
    const char* module_name,
    DB::Tag prototype_tag)
    : m_ef(make_handle_dup(ef))
    , m_vf(m_ef->get_value_factory())
    , m_tf(m_vf->get_type_factory())
    , m_transaction(transaction)
    , m_immutable_callees(immutable_callees)
    , m_create_direct_calls(create_direct_calls)
    , m_module_filename(module_filename)
    , m_module_name(module_name)
    , m_prototype_tag(prototype_tag)
{
}

/// Converts mi::mdl::DAG_node to MI::MDL::IExpression.
IExpression* Mdl_dag_converter::mdl_dag_node_to_int_expr(
    const mi::mdl::DAG_node* node, const IType* type_int) const
{
    switch (node->get_kind()) {
    case mi::mdl::DAG_node::EK_CONSTANT: {

        const mi::mdl::DAG_constant* constant = cast<mi::mdl::DAG_constant>(node);
        const mi::mdl::IValue* value = constant->get_value();
        mi::base::Handle<IValue> value_int(mdl_value_to_int_value(
            m_vf.get(), m_transaction, type_int, value, m_module_filename, m_module_name));
        return m_ef->create_constant(value_int.get());
    }
    case mi::mdl::DAG_node::EK_CALL: {

        const mi::mdl::DAG_call* call = cast<mi::mdl::DAG_call>(node);
        return m_create_direct_calls
            ? mdl_call_to_int_expr_direct(call, type_int != NULL)
            : mdl_call_to_int_expr_indirect(call, type_int != NULL);
    }
    case mi::mdl::DAG_node::EK_PARAMETER: {

        const mi::mdl::DAG_parameter* parameter = cast<mi::mdl::DAG_parameter>(node);
        return mdl_parameter_to_int_expr(m_ef.get(), parameter);
    }
    case mi::mdl::DAG_node::EK_TEMPORARY: {

        const mi::mdl::DAG_temporary* temporary = cast<mi::mdl::DAG_temporary>(node);
        return mdl_temporary_to_int_expr(m_ef.get(), temporary);
    }
    }

    ASSERT(M_SCENE, false);
    return 0;
}

/// Converts mi::mdl::DAG_call to MI::MDL::IExpression (creates IExpression_direct_call)
IExpression* Mdl_dag_converter::mdl_call_to_int_expr_direct(
    const mi::mdl::DAG_call* call, bool use_parameter_type) const
{
    const mi::mdl::IType* return_type = call->get_type();
    mi::base::Handle<const IType> return_type_int(
        mdl_type_to_int_type(m_tf.get(), return_type));

    std::string call_name = add_mdl_db_prefix(call->get_name());
    DB::Tag definition_tag = m_transaction->name_to_tag(call_name.c_str());
    ASSERT(M_SCENE, definition_tag.is_valid());

    mi::base::Handle<const IType_list> parameter_types;
    SERIAL::Class_id class_id = m_transaction->get_class_id(definition_tag);
    if (class_id == ID_MDL_FUNCTION_DEFINITION) {

        DB::Access<Mdl_function_definition> function_definition(definition_tag, m_transaction);
        parameter_types = function_definition->get_parameter_types();
    }
    else if (class_id == ID_MDL_MATERIAL_DEFINITION) {

        DB::Access<Mdl_material_definition> material_definition(definition_tag, m_transaction);
        parameter_types = material_definition->get_parameter_types();
    }
    else {
        ASSERT(M_SCENE, false);
        return 0;
    }

    mi::base::Handle<IExpression_list> arguments(m_ef->create_expression_list());
    mi::Uint32 n = call->get_argument_count();
    for (mi::Uint32 i = 0; i < n; i++) {

        const char* parameter_name = call->get_parameter_name(i);
        const mi::mdl::DAG_node* mdl_argument = call->get_argument(i);
        mi::base::Handle<const IType> parameter_type(
            use_parameter_type ? parameter_types->get_type(i) : (const IType *)0);
        mi::base::Handle<const IExpression> argument(mdl_dag_node_to_int_expr(
            mdl_argument, parameter_type.get()));
        arguments->add_expression(parameter_name, argument.get());
    }

    return m_ef->create_direct_call(return_type_int.get(), definition_tag, arguments.get());
}

/// Converts mi::mdl::DAG_call to MI::MDL::IExpression (creates IExpression_call)
IExpression* Mdl_dag_converter::mdl_call_to_int_expr_indirect(
    const mi::mdl::DAG_call* call,
    bool use_parameter_type) const
{
    bool is_array_constructor
        = call->get_semantic() == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;

    // get tag and class ID of call
    std::string definition_name = add_mdl_db_prefix(call->get_name());
    DB::Tag definition_tag = m_transaction->name_to_tag(definition_name.c_str());
    ASSERT(M_SCENE, definition_tag);
    SERIAL::Class_id class_id = m_transaction->get_class_id(definition_tag);

    if (class_id == ID_MDL_FUNCTION_DEFINITION) {

        DB::Access<Mdl_function_definition> function_definition(definition_tag, m_transaction);
        mi::base::Handle<const IType_list> parameter_types(
            function_definition->get_parameter_types());
        // create argument list
        mi::Uint32 n = call->get_argument_count();
        mi::base::Handle<IExpression_list> arguments(m_ef->create_expression_list());

        for (mi::Uint32 i = 0; i < n; ++i) {

            const mi::mdl::DAG_node* argument = call->get_argument(i);
            if (!argument)
                continue;
            mi::base::Handle<const IType> parameter_type(
                use_parameter_type ? parameter_types->get_type(i) : (const IType *)0);
            mi::base::Handle<IExpression> argument_int(
                mdl_dag_node_to_int_expr(argument, parameter_type.get()));
            ASSERT(M_SCENE, argument_int);
            std::string parameter_name;
            if (is_array_constructor)
                parameter_name = std::to_string(i);
            else
                parameter_name = function_definition->get_parameter_name(i);
            arguments->add_expression(parameter_name.c_str(), argument_int.get());
        }
        // create function call from definition
        mi::Sint32 errors = 0;
        Mdl_function_call* function_call = is_array_constructor
            ? function_definition->create_array_constructor_call_internal(m_transaction,
                arguments.get(), m_immutable_callees, &errors)
            : function_definition->create_function_call_internal(m_transaction,
                arguments.get(), /*allow_ek_parameter*/ true, m_immutable_callees, &errors);
        ASSERT(M_SCENE, function_call);
        if (!function_call) {
            LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                "Instantiation of function definition \"%s\" failed.", definition_name.c_str());
            return 0;
        }
        // store function call
        std::string call_name_base = m_immutable_callees ? 
        definition_name + "__default_call__" : definition_name;
        std::string call_name
            = DETAIL::generate_unique_db_name(m_transaction, call_name_base.c_str());
        DB::Tag call_tag = m_transaction->store_for_reference_counting(
            function_call, call_name.c_str(), m_transaction->get_scope()->get_level());

        // create call expression
        const mi::mdl::IType* return_type = call->get_type();
        mi::base::Handle<const IType> return_type_int(
            mdl_type_to_int_type(m_tf.get(), return_type));
        return m_ef->create_call(return_type_int.get(), call_tag);
    }
    else if (class_id == ID_MDL_MATERIAL_DEFINITION) {

        DB::Access<Mdl_material_definition> material_definition(definition_tag, m_transaction);
        mi::base::Handle<const IType_list> parameter_types(
            material_definition->get_parameter_types());
        // create argument list
        mi::Uint32 n = call->get_argument_count();
        mi::base::Handle<IExpression_list> arguments(m_ef->create_expression_list());
        for (mi::Uint32 i = 0; i < n; ++i) {

            const mi::mdl::DAG_node* argument = call->get_argument(i);
            if (!argument)
                continue;
            mi::base::Handle<const IType> parameter_type(parameter_types->get_type(i));
            mi::base::Handle<IExpression> argument_int(
                mdl_dag_node_to_int_expr(argument, parameter_type.get()));
            ASSERT(M_SCENE, argument_int);
            const char* parameter_name = material_definition->get_parameter_name(i);
            arguments->add_expression(parameter_name, argument_int.get());
        }
        // create material instance from definition
        mi::Sint32 errors = 0;
        Mdl_material_instance* material_instance
            = material_definition->create_material_instance_internal(
                m_transaction, arguments.get(), /*allow_ek_parameter*/ true, m_immutable_callees, &errors);
        ASSERT(M_SCENE, material_instance);
        if (!material_instance) {
            LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                "Instantiation of material definition \"%s\" failed.", definition_name.c_str());
            return 0;
        }

        // store material call
        std::string call_name_base = m_immutable_callees ?
            definition_name + "__default_call__" : definition_name;
        std::string call_name
            = DETAIL::generate_unique_db_name(m_transaction, call_name_base.c_str());
        DB::Tag instance_tag = m_transaction->store_for_reference_counting(
            material_instance, call_name.c_str(), m_transaction->get_scope()->get_level());

        // create call expression
        const mi::mdl::IType* return_type = call->get_type();
        mi::base::Handle<const IType> return_type_int(
            mdl_type_to_int_type(m_tf.get(), return_type));

        return m_ef->create_call(return_type_int.get(), instance_tag);
    }
    else
        ASSERT(M_SCENE, false);

    return 0;
}

/// Converts a vector of mi::mdl::DAG_node pointers to MI::MDL::IAnnotation_block.
IAnnotation_block* Mdl_dag_converter::mdl_dag_node_vector_to_int_annotation_block(
    const Mdl_annotation_block& mdl_annotations, const char* qualified_name) const
{
    if (mdl_annotations.empty())
        return 0;

    IAnnotation_block* block = m_ef->create_annotation_block();

    for (mi::Size i = 0; i < mdl_annotations.size(); ++i) {

        const mi::mdl::DAG_call* call = cast<mi::mdl::DAG_call>(mdl_annotations[i]);
        mi::base::Handle<IAnnotation> annotation(
            mdl_dag_call_to_int_annotation(call, qualified_name));
        block->add_annotation(annotation.get());
    }

    return block;
}

/// Converts mi::mdl::DAG_call to MI::MDL::IAnnotation.
IAnnotation* Mdl_dag_converter::mdl_dag_call_to_int_annotation(
    const mi::mdl::DAG_call* call, const char* qualified_name) const
{
    mi::Uint32 n = call->get_argument_count();
    mi::base::Handle<IExpression_list> arguments(m_ef->create_expression_list());
    for (mi::Uint32 i = 0; i < n; ++i) {
        const mi::mdl::DAG_node* argument = call->get_argument(i);
        mi::base::Handle<IExpression> argument_int(mdl_dag_node_to_int_expr_localized(
            argument, call, /*type_int*/ 0, qualified_name));
        ASSERT(M_SCENE, argument_int);
        ASSERT(M_SCENE, argument_int->get_kind() == IExpression::EK_CONSTANT);
        const char* parameter_name = call->get_parameter_name(i);
        arguments->add_expression(parameter_name, argument_int.get());
    }

    const char* name = call->get_name();
    return m_ef->create_annotation(name, arguments.get());
}

void setup_translation_unit(
    Mdl_translator_module::Translation_unit & translation_unit
    , DB::Transaction* transaction
    , DB::Tag prototype_tag
    , const char * module_name
    , const char * qualified_name
    , const char * value_string
)
{
    // Set values when prototype_tag is null (most common case)
    // prototype_tag is used for material and function variants (aka "presets")
    translation_unit.set_module_name(module_name);
    translation_unit.set_context(qualified_name);
    translation_unit.set_source(value_string);

    if (prototype_tag)
    {
        // Search the values in parent material
        DB::Access<MDL::Mdl_material_definition> element(prototype_tag, transaction);
        if (element.is_valid())
        {
            DB::Tag module_tag(element->get_module(transaction));
            DB::Access<MDL::Mdl_module> module(module_tag, transaction);
            if (module.is_valid())
            {
                translation_unit.set_module_name(module->get_mdl_name());
                translation_unit.set_context(element->get_mdl_name());
            }
        }
        else
        {
            // Search the values in parent function
            DB::Access<MDL::Mdl_function_definition> element(prototype_tag, transaction);
            if (element.is_valid())
            {
                DB::Tag module_tag(element->get_module(transaction));
                DB::Access<MDL::Mdl_module> module(module_tag, transaction);
                if (module.is_valid())
                {
                    translation_unit.set_module_name(module->get_mdl_name());
                    translation_unit.set_context(element->get_mdl_name());
                }
            }
        }
    }
}

/// Identical to mdl_dag_node_to_int_expr() but for translation in the context of localization
IExpression* Mdl_dag_converter::mdl_dag_node_to_int_expr_localized(
    const mi::mdl::DAG_node* argument,
    const mi::mdl::DAG_call* call,
    const IType* type_int,
    const char* qualified_name) const
{
    if (argument->get_kind() == mi::mdl::DAG_node::EK_CONSTANT) {
        // If the qualified name is set and the annotation is one we translate the translate it
        MI::SYSTEM::Access_module<Mdl_translator_module> mdl_translator(false);
        if (qualified_name &&  mdl_translator->need_translation(call->get_name())) {
            // Need to translate, check the type of value
            const mi::mdl::DAG_constant* constant = cast<mi::mdl::DAG_constant>(argument);
            const mi::mdl::IValue* value = constant->get_value();
            if (value->get_kind() == mi::mdl::IValue::VK_STRING)
            {
                const mi::mdl::IValue_string* value_string = cast<mi::mdl::IValue_string>(value);
                Mdl_translator_module::Translation_unit translation_unit;
                setup_translation_unit(
                    translation_unit
                    , m_transaction
                    , m_prototype_tag
                    , m_module_name
                    , qualified_name
                    , value_string->get_value()
                );
                if (0 == mdl_translator->translate(translation_unit))
                {
                    std::string translated_string = translation_unit.get_target();
                    mi::base::Handle<IValue> value_int(
                        m_vf->create_string(translated_string.c_str()));
                    return m_ef->create_constant(value_int.get());
                }
            }
            else if (value->get_kind() == mi::mdl::IValue::VK_ARRAY)
            {
                mi::base::Handle<IValue_factory> vf(m_ef->get_value_factory());
                const mi::mdl::IValue_array* value_array = cast<mi::mdl::IValue_array>(value);
                mi::base::Handle<const IType_array> type_array_int;
                // compute type from the value
                const mi::mdl::IType_array* type_array = value_array->get_type();
                if (mi::mdl::IType::TK_STRING == type_array->get_element_type()->get_kind()) {

                    type_array_int = mdl_type_to_int_type<IType_array>(m_tf.get(), type_array);
                    IValue_array* value_array_int = m_vf->create_array(type_array_int.get());
                    mi::Size n = value_array->get_component_count();
                    if (!type_array_int->is_immediate_sized()) {
                        value_array_int->set_size(n);
                    }
                    mi::base::Handle<const IType> component_type_int(
                        type_array_int->get_element_type());
                    for (mi::Size i = 0; i < n; ++i) {
                        const mi::mdl::IValue* component
                            = value_array->get_value(static_cast<mi::Uint32>(i));
                        const mi::mdl::IValue_string* value_string =
                            cast<mi::mdl::IValue_string>(component);
                        Mdl_translator_module::Translation_unit translation_unit;      
                        setup_translation_unit(
                            translation_unit
                            , m_transaction
                            , m_prototype_tag
                            , m_module_name
                            , qualified_name
                            , value_string->get_value()
                        );
                        translation_unit.set_source(value_string->get_value());
                        std::string translated_string(translation_unit.get_source());
                        if (0 == mdl_translator->translate(translation_unit))
                        {
                            translated_string = translation_unit.get_target();
                        }
                        mi::base::Handle<IValue> component_int(
                            vf->create_string(translated_string.c_str()));
                        mi::Sint32 result = value_array_int->set_value(i, component_int.get());
                        ASSERT(M_SCENE, result == 0);
                        boost::ignore_unused(result);
                    }

                    mi::base::Handle<IValue> value_int(value_array_int);
                    return m_ef->create_constant(value_int.get());
                }
            }
        }
    }
    // Fallback to the original code
    return mdl_dag_node_to_int_expr(argument, /*type_int*/ 0);
}

// ********** Conversion from MI::MDL to mi::mdl ***************************************************

namespace {

/// Converts MI::MDL::IValue_texture (given as tag) to mi::mdl::IValue_texture or
/// IValue_invalid_ref
const mi::mdl::IValue* int_value_texture_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_texture* mdl_type,
    DB::Tag tag)
{
    if( !tag)
        return vf->create_invalid_ref( mdl_type);

    DB::Tag_version tag_version = transaction->get_tag_version( tag);

    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    if( class_id != TEXTURE::Texture::id) {
        const char* name = transaction->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for texture resource \"%s\".", name?name:"");
        return vf->create_invalid_ref( mdl_type);
    }

    DB::Access<TEXTURE::Texture> texture( tag, transaction);
    DB::Tag image_tag( texture->get_image());
    if( !image_tag) {
        mi::Uint32 hash
            = get_hash( /*mdl_file_path*/ 0, /*gamma*/ 0.0f, tag_version, DB::Tag_version());
        return vf->create_texture( mdl_type, "",
            mi::mdl::IValue_texture::gamma_default, tag.get_uint(), hash);
    }

    DB::Tag_version image_tag_version = transaction->get_tag_version( image_tag);

    class_id = transaction->get_class_id( image_tag);
    if( class_id != DBIMAGE::Image::id) {
        const char* name = transaction->tag_to_name( image_tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for image resource \"%s\".", name?name:"");
        return vf->create_invalid_ref( mdl_type);
    }

    DB::Access<DBIMAGE::Image> image( image_tag, transaction);
    const std::string& resource_name = image->get_original_filename();

    // try to convert gamma value into the MDL constant
    mi::Float32 gamma_override = texture->get_gamma();
    mi::mdl::IValue_texture::gamma_mode gamma;
    if( gamma_override == 1.0f)
        gamma = mi::mdl::IValue_texture::gamma_linear;
    else if( gamma_override == 2.2f)
        gamma = mi::mdl::IValue_texture::gamma_srgb;
    else
        gamma = mi::mdl::IValue_texture::gamma_default;

    mi::Uint32 hash
        = get_hash( image->get_mdl_file_path(), gamma_override, tag_version, image_tag_version);
    return vf->create_texture( mdl_type, resource_name.c_str(), gamma, tag.get_uint(), hash);
}

/// Converts MI::MDL::IValue_light_profile (given as tag) to mi::mdl::IValue_light_profile or
/// IValue_invalid_ref
const mi::mdl::IValue* int_value_light_profile_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_light_profile* mdl_type,
    DB::Tag tag)
{
    if( !tag)
        return vf->create_invalid_ref( mdl_type);

    DB::Tag_version tag_version = transaction->get_tag_version( tag);

    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    if( class_id != LIGHTPROFILE::Lightprofile::id) {
        const char* name = transaction->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for light profile resource \"%s\".", name?name:"");
        return vf->create_invalid_ref( mdl_type);
    }

    DB::Access<LIGHTPROFILE::Lightprofile> lightprofile( tag, transaction);
    const std::string& resource_name = lightprofile->get_original_filename();
    mi::Uint32 hash = get_hash( lightprofile->get_mdl_file_path(), tag_version);
    return vf->create_light_profile( mdl_type, resource_name.c_str(), tag.get_uint(), hash);
}

/// Converts MI::MDL::IValue_bsdf_measurement (given as tag) to mi::mdl::IValue_bsdf_measurement or
/// IValue_invalid_ref
const mi::mdl::IValue* int_value_bsdf_measurement_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_bsdf_measurement* mdl_type,
    DB::Tag tag)
{
    if( !tag)
        return vf->create_invalid_ref( mdl_type);

    DB::Tag_version tag_version = transaction->get_tag_version( tag);

    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    if( class_id != BSDFM::Bsdf_measurement::id) {
        const char* name = transaction->tag_to_name( tag);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Incorrect type for BSDF measurement resource \"%s\".", name?name:"");
        return vf->create_invalid_ref( mdl_type);
    }

    DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( tag, transaction);
    const std::string& resource_name = bsdf_measurement->get_original_filename();
    mi::Uint32 hash = get_hash( bsdf_measurement->get_mdl_file_path(), tag_version);
    return vf->create_bsdf_measurement( mdl_type, resource_name.c_str(), tag.get_uint(), hash);
}

} // namespace

const mi::mdl::IValue* int_value_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType* mdl_type,
    const IValue* value)
{
    const mi::mdl::IType* stripped_mdl_type = mdl_type->skip_type_alias();
    mi::mdl::IType::Kind stripped_mdl_type_kind = stripped_mdl_type->get_kind();

    IValue::Kind kind = value->get_kind();

    switch( kind) {

        // The type kind checks below might fail if the graph has been broken by overwriting
        // DB elements.

        case IValue::VK_BOOL: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_BOOL)
                return 0;
            mi::base::Handle<const IValue_bool> value_bool(
                value->get_interface<IValue_bool>());
            return vf->create_bool( value_bool->get_value());
        }
        case IValue::VK_INT: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_INT)
                return 0;
            mi::base::Handle<const IValue_int> value_int(
                value->get_interface<IValue_int>());
            return vf->create_int( value_int->get_value());
        }
        case IValue::VK_ENUM: {
            const mi::mdl::IType_enum* mdl_type_enum
                = as<mi::mdl::IType_enum>( stripped_mdl_type);
            if( !mdl_type_enum)
                return 0;
            mi::base::Handle<const IValue_enum> value_enum(
                value->get_interface<IValue_enum>());
            mi::Size index = value_enum->get_index();
            return vf->create_enum( mdl_type_enum, index);
        }
        case IValue::VK_FLOAT: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_FLOAT)
                return 0;
            mi::base::Handle<const IValue_float> value_float(
                value->get_interface<IValue_float>());
            return vf->create_float( value_float->get_value());
        }
        case IValue::VK_DOUBLE: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_DOUBLE)
                return 0;
            mi::base::Handle<const IValue_double> value_double(
                value->get_interface<IValue_double>());
            return vf->create_double( value_double->get_value());
        }
        case IValue::VK_STRING: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_STRING)
                return 0;
            mi::base::Handle<const IValue_string> value_string(
                value->get_interface<IValue_string>());
            return vf->create_string( value_string->get_value());
        }
        case IValue::VK_VECTOR:
        case IValue::VK_MATRIX:
        case IValue::VK_COLOR:
        case IValue::VK_ARRAY:
        case IValue::VK_STRUCT: {
            mi::base::Handle<const IValue_compound> value_compound(
                value->get_interface<IValue_compound>());
            const mi::mdl::IType_compound* mdl_type_compound
                = as<mi::mdl::IType_compound>( stripped_mdl_type);
            if( !mdl_type_compound)
                return 0;
            mi::Size n = value_compound->get_size();
            mdl_type_compound
                = convert_deferred_sized_into_immediate_sized_array( vf, mdl_type_compound, n);
            mi::Size type_n = mdl_type_compound->get_compound_size();
            if( type_n != n)
                return 0;
            std::vector<const mi::mdl::IValue*> mdl_element_values( n);
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IValue> element_value(
                    value_compound->get_value( i));
                const mi::mdl::IType* mdl_element_type
                    = mdl_type_compound->get_compound_type( static_cast<mi::Uint32>( i));
                const mi::mdl::IValue* mdl_element_value = int_value_to_mdl_value(
                    transaction, vf, mdl_element_type, element_value.get());
                mdl_element_values[i] = mdl_element_value;
            }
            return vf->create_compound( mdl_type_compound, n > 0 ? &mdl_element_values[0] : 0, n);
        }
        case IValue::VK_TEXTURE: {
            mi::base::Handle<const IValue_texture> value_texture(
                value->get_interface<IValue_texture>());
            DB::Tag tag = value_texture->get_value();
            const mi::mdl::IType_texture* mdl_type_texture
                = as<mi::mdl::IType_texture>( stripped_mdl_type);
            if( !mdl_type_texture)
                return 0;
            return int_value_texture_to_mdl_value(
                transaction, vf, mdl_type_texture, tag);
        }
        case IValue::VK_LIGHT_PROFILE: {
            mi::base::Handle<const IValue_light_profile> value_light_profile(
                value->get_interface<IValue_light_profile>());
            DB::Tag tag = value_light_profile->get_value();
            const mi::mdl::IType_light_profile* mdl_type_light_profile
                = as<mi::mdl::IType_light_profile>( stripped_mdl_type);
            if( !mdl_type_light_profile)
                return 0;
            return int_value_light_profile_to_mdl_value(
                transaction, vf, mdl_type_light_profile, tag);
        }
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_bsdf_measurement> value_bsdf_measurement(
                value->get_interface<IValue_bsdf_measurement>());
            DB::Tag tag = value_bsdf_measurement->get_value();
            const mi::mdl::IType_bsdf_measurement* mdl_type_bsdf_measurement
                = as<mi::mdl::IType_bsdf_measurement>( stripped_mdl_type);
            if( !mdl_type_bsdf_measurement)
                return 0;
            return int_value_bsdf_measurement_to_mdl_value(
                transaction, vf, mdl_type_bsdf_measurement, tag);
        }
        case IValue::VK_INVALID_DF: {
            const mi::mdl::IType_reference* mdl_type_reference
                = as<mi::mdl::IType_reference>( stripped_mdl_type);
            if( !mdl_type_reference)
                return 0;
            return vf->create_invalid_ref( mdl_type_reference);
        }
        case IValue::VK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

const mi::mdl::IExpression_literal* int_value_to_mdl_literal(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    const mi::mdl::IType* mdl_type,
    const IValue* value)
{
    mi::mdl::IValue_factory* value_factory = module->get_value_factory();
    mi::mdl::IExpression_factory* expression_factory = module->get_expression_factory();

    const mi::mdl::IValue* mdl_value = int_value_to_mdl_value(
        transaction, value_factory, mdl_type, value);
    if( !mdl_value)
        return 0;

    return expression_factory->create_literal( mdl_value);
}

namespace {

/// Creates a field reference from a field access (aka struct getter) function signature.
///
/// \param module      The module on which the expression is created.
/// \param signature   The signature.
/// \return            The created field reference.
const mi::mdl::IExpression_reference* get_field_reference(
    mi::mdl::IModule* module, const char* signature)
{
    mi::mdl::IName_factory &nf = *module->get_name_factory();

    const char* dot = strchr( signature, '.');
    ASSERT( M_SCENE, dot);
    const char* end = strchr( dot, '(');

    std::string field( dot + 1, end - dot - 1);
    const mi::mdl::ISymbol* symbol = nf.create_symbol( field.c_str());
    const mi::mdl::ISimple_name* simple_name = nf.create_simple_name( symbol);
    mi::mdl::IQualified_name* qualified_name = nf.create_qualified_name();
    qualified_name->add_component( simple_name);

    mi::mdl::IType_name* type_name = nf.create_type_name( qualified_name);
    mi::mdl::IExpression_factory &ef = *module->get_expression_factory();
    return ef.create_reference( type_name);
}

/// Creates a deferred array size reference from an array type.
///
/// \param module       The module on which the expression is created.
/// \param array_type   The array type.
/// \return             The created array reference.
const mi::mdl::IExpression_reference* get_array_size_reference(
    mi::mdl::IModule* module, const mi::mdl::IType_array* array_type)
{
    ASSERT( M_SCENE, !array_type->is_immediate_sized());

    mi::mdl::IName_factory &nf = *module->get_name_factory();

    const mi::mdl::IType_array_size* size = array_type->get_deferred_size();

    const mi::mdl::ISymbol* symbol = size->get_size_symbol();
    // import the symbol
    symbol = nf.create_symbol( symbol->get_name());

    const mi::mdl::ISimple_name* simple_name = nf.create_simple_name( symbol);
    mi::mdl::IQualified_name* qualified_name = nf.create_qualified_name();
    qualified_name->add_component( simple_name);

    mi::mdl::IType_name* type_name = nf.create_type_name( qualified_name);
    mi::mdl::IExpression_factory &ef = *module->get_expression_factory();
    mi::mdl::IExpression_reference* ref = ef.create_reference( type_name);
    ref->set_type( module->get_type_factory()->create_int());
    return ref;
}

/// Get a mask of parameters that must be killed due to older MDL version.
///
/// \param signature   The signature of the function call.
/// \param module      The destination module.
unsigned get_parameter_killmask(
    const char* signature,
    const mi::mdl::IModule* module)
{
    // Beware: internally we have ALWAYS the highest possible MDL version,
    // but when we create older MDL modules we must convert it down to
    // older versions ...
    int major = 0, minor = 0;
    module->get_version(major, minor);
    bool has_11 = true, has_12 = true, has_13 = true, has_14 = true;
    switch (major) {
    case 0:
        has_11 = has_12 = has_13 = has_14 = false;
        break;
    case 1:
        switch (minor) {
        case 0:
            has_11 = has_12 = has_13 = has_14 = false;
            break;
        case 1:
            has_12 = has_13 = has_14 = false;
            break;
        case 2:
            has_13 = has_14 = false;
            break;
        case 3:
            has_14 = false;
        case 4:
            // no killmask
            return 0;
        }
    }

    unsigned killmask = 0;

#define KILL_ARG(num) (1 << (num))

    // Convert back several functions. Note that the DAG-IR always uses the highest version,
    // but we export back to the highest version from the original source. Hence, some
    // conversion done by the DAG backend must be reverted.

    // 1.1 => 1.0
    // material_emission(edf,color,intensity_mode) =>
    // material_emission(edf,color)
    if (strcmp("material_emission(edf,color,intensity_mode)", signature) == 0) {
        if (!has_11) {
            killmask |= KILL_ARG(2);
        }
    }

    // 1.2 => 1.0
    // df::measured_edf(light_profile,float,bool,float3x3,float3,string) =>
    // df::measured_edf(light_profile,      bool,float3x3,       string)
    else if (strcmp("df::measured_edf(light_profile,float,bool,float3x3,float3,string)",
        signature) == 0)
    {
        if (!has_11) {
            killmask |= KILL_ARG(1);
        }
        if (!has_12) {
            killmask |= KILL_ARG(4);
        }
    }

    // 1.1 => 1.0
    // df::spot_edf(float,float,bool,float3x3,string) =>
    // df::spot_edf(float,      bool,float3x3,string)
    else if (strcmp("df::spot_edf(float,float,bool,float3x3,string)", signature) == 0) {
        if (!has_11) {
            killmask |= KILL_ARG(1);
        }
    }

    // 1.3 => 1.2
    // state::rounded_corner_normal(float,bool,float) =>
    // state::rounded_corner_normal(float,bool)
    else if (strcmp("state::rounded_corner_normal(float,bool,float)", signature) == 0) {
        if (!has_12) {
            killmask |= KILL_ARG(2);
        }
    }

    // 1.4 => 1.3
    // tex::width(texture_2d,int2) =>             tex::width(texture_2d)
    // tex::height(texture_2d, int2) = >          tex::height(texture_2d)
    // tex::texel_float(texture_2d,int2,int2) =>  tex::texel_float(texture_2d,int2)
    // tex::texel_float2(texture_2d,int2,int2) => tex::texel_float2(texture_2d,int2)
    // tex::texel_float3(texture_2d,int2,int2) => tex::texel_float3(texture_2d,int2)
    // tex::texel_float4(texture_2d,int2,int2) => tex::texel_float4(texture_2d,int2)
    // tex::texel_color(texture_2d,int2,int2) =>  tex::texel_color(texture_2d,int2)
    else if (strcmp("tex::width(texture_2d,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(1);
        }
    } else if (strcmp("tex::height(texture_2d,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(1);
        }
    } else if (strcmp("tex::texel_float(texture_2d,int2,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(2);
        }
    } else if (strcmp("tex::texel_float2(texture_2d,int2,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(2);
        }
    } else if (strcmp("tex::texel_float3(texture_2d,int2,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(2);
        }
    } else if (strcmp("tex::texel_float4(texture_2d,int2,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(2);
        }
    } else if (strcmp("tex::texel_color(texture_2d,int2,int2)", signature) == 0) {
        if (!has_14) {
            killmask |= KILL_ARG(2);
        }
    }

    return killmask;
}

} // namespace

const mi::mdl::IExpression* int_expr_to_mdl_ast_expr(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    const mi::mdl::IType* mdl_type,
    const IExpression* expr,
    std::set<MI::Uint32>& call_trace)
{
    mi::mdl::IName_factory* name_factory = module->get_name_factory();
    mi::mdl::IType_factory* type_factory = module->get_type_factory();
    mi::mdl::IExpression_factory* expr_factory = module->get_expression_factory();

    mdl_type = type_factory->import( mdl_type);

    IExpression::Kind kind = expr->get_kind();

    switch( kind) {
        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            return int_value_to_mdl_literal( transaction, module, mdl_type, value.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            DB::Tag tag = expr_call->get_call();
            ASSERT( M_SCENE, tag);

            Call_stack_guard guard( call_trace, tag.get_uint());
            if( guard.last_frame_creates_cycle())
                return 0;

            // create call expressions
            SERIAL::Class_id class_id = transaction->get_class_id( tag);
            if( class_id == Mdl_function_call::id) {

                // handle function call parameters
                DB::Access<Mdl_function_call> call( tag, transaction);
                mi::base::Handle<const IExpression_list> args( call->get_arguments());

                mi::mdl::IDefinition::Semantics semantic = call->get_mdl_semantic();
                if( mi::mdl::semantic_is_operator( semantic)) {

                    // handle operators
                    mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator( semantic);
                    if( mi::mdl::is_unary_operator( op)) {

                        const mi::mdl::IType* type = call->get_mdl_parameter_type( transaction, 0);
                        mi::base::Handle<const IExpression> arg(
                            args->get_expression( static_cast<mi::Size>( 0)));
                        const mi::mdl::IExpression* expr = int_expr_to_mdl_ast_expr(
                            transaction, module, type, arg.get(), call_trace);
                        return expr_factory->create_unary(
                            mi::mdl::IExpression_unary::Operator( op), expr);

                    } else if( mi::mdl::is_binary_operator( op)) {

                        const mi::mdl::IType* left_type
                            = call->get_mdl_parameter_type( transaction, 0);
                        const mi::mdl::IType* right_type
                            = call->get_mdl_parameter_type( transaction, 1);
                        mi::base::Handle<const IExpression> left_arg(
                             args->get_expression( static_cast<mi::Size>( 0)));
                        mi::base::Handle<const IExpression> right_arg(
                            args->get_expression( 1));
                        const mi::mdl::IExpression* left_expr  = int_expr_to_mdl_ast_expr(
                            transaction, module, left_type,  left_arg.get(), call_trace);
                        const mi::mdl::IExpression* right_expr = int_expr_to_mdl_ast_expr(
                            transaction, module, right_type, right_arg.get(), call_trace);
                        return expr_factory->create_binary(
                            mi::mdl::IExpression_binary::Operator( op), left_expr, right_expr);

                    } else if( op == mi::mdl::IExpression::OK_TERNARY) {

                        const mi::mdl::IType* cond_type
                            = call->get_mdl_parameter_type( transaction, 0);
                        const mi::mdl::IType* true_type
                            = call->get_mdl_parameter_type( transaction, 1);
                        const mi::mdl::IType* false_type
                            = call->get_mdl_parameter_type( transaction, 2);
                        mi::base::Handle<const IExpression> cond_arg(
                            args->get_expression( static_cast<mi::Size>( 0)));
                        mi::base::Handle<const IExpression> true_arg(
                            args->get_expression( 1));
                        mi::base::Handle<const IExpression> false_arg(
                            args->get_expression( 2));
                        const mi::mdl::IExpression* cond_expr  = int_expr_to_mdl_ast_expr(
                            transaction, module, cond_type,  cond_arg.get(), call_trace);
                        const mi::mdl::IExpression* true_expr  = int_expr_to_mdl_ast_expr(
                            transaction, module, true_type,  true_arg.get(), call_trace);
                        const mi::mdl::IExpression* false_expr = int_expr_to_mdl_ast_expr(
                            transaction, module, false_type, false_arg.get(), call_trace);
                        return expr_factory->create_conditional( cond_expr, true_expr, false_expr);

                    } else
                        ASSERT( M_SCENE, false);
                }

                // handle other DAG specific semantics that do not exist in MDL AST
                switch( semantic) {

                    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS: {
                        const mi::mdl::IType* left_type
                            = call->get_mdl_parameter_type( transaction, 0);
                        left_type = type_factory->import( left_type);
                        mi::base::Handle<const IExpression> left_arg(
                            args->get_expression( static_cast<mi::Size>( 0)));
                        const mi::mdl::IExpression* left_expr = int_expr_to_mdl_ast_expr(
                            transaction, module, left_type, left_arg.get(), call_trace);
                        if( !left_expr)
                            return 0;
                        const mi::mdl::IExpression* right_expr
                            = get_field_reference( module, call->get_mdl_function_definition());
                        mi::mdl::IExpression_binary* result = expr_factory->create_binary(
                            mi::mdl::IExpression_binary::OK_SELECT, left_expr, right_expr);
                        result->set_type( call->get_mdl_return_type( transaction));
                        return result;
                    }

                    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR: {
                        const mi::mdl::IType_array* array_type
                            = as<mi::mdl::IType_array>( mdl_type);
                        const mi::mdl::IType* element_type = array_type->get_element_type();
                        element_type = type_factory->import( element_type);
                        // create return type
                        mi::Size count = args->get_size();
                        const mi::mdl::IType* return_type
                            = as<mi::mdl::IType_array>( type_factory->create_array(
                                element_type, count));
                        return_type = type_factory->import( return_type);
                        mi::mdl::IExpression_reference* ref
                            = type_to_reference( module, return_type);
                        mi::mdl::IExpression_call* result = expr_factory->create_call( ref);
                        for( mi::Uint32 i = 0; i < count; ++i) {
                            mi::base::Handle<const IExpression> argument(
                                args->get_expression( i));
                            const mi::mdl::IExpression* argument_expr = int_expr_to_mdl_ast_expr(
                                transaction, module, element_type, argument.get(), call_trace);
                            if( !argument_expr)
                                return 0;
                            const mi::mdl::IArgument* argument_arg
                                = expr_factory->create_positional_argument( argument_expr);
                            result->add_argument( argument_arg);
                        }
                        result->set_type( return_type);
                        return result;
                    }

                    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_INDEX_ACCESS: {
                        const mi::mdl::IType* left_type
                            = call->get_mdl_parameter_type( transaction, 0);
                        left_type = type_factory->import( left_type);
                        mi::base::Handle<const IExpression> left_arg(
                            args->get_expression( static_cast<mi::Size>( 0)));
                        const mi::mdl::IExpression* left_expr = int_expr_to_mdl_ast_expr(
                            transaction, module, left_type, left_arg.get(), call_trace);
                        if( !left_expr)
                            return 0;
                        const mi::mdl::IType* right_type
                            = call->get_mdl_parameter_type( transaction, 1);
                        right_type = type_factory->import( right_type);
                        mi::base::Handle<const IExpression> right_arg( args->get_expression( 1));
                        const mi::mdl::IExpression* right_expr = int_expr_to_mdl_ast_expr(
                            transaction, module, right_type, right_arg.get(), call_trace);
                        if( !right_expr)
                           return 0;
                        mi::mdl::IExpression_binary* result = expr_factory->create_binary(
                            mi::mdl::IExpression_binary::OK_ARRAY_INDEX, left_expr, right_expr);
                        result->set_type( call->get_mdl_return_type( transaction));
                        return result;
                    }

                    case mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH: {
                        // no need to import the type
                        const mi::mdl::IType* type = call->get_mdl_parameter_type( transaction, 0u);
                        const mi::mdl::IType_array* array_type = as<mi::mdl::IType_array>( type);
                        ASSERT( M_SCENE, array_type);
                        return get_array_size_reference( module, array_type);
                    }

                    default: {
                        const char* function_name = call->get_mdl_function_definition();
                        const mi::mdl::IExpression_reference* ref
                            = signature_to_reference( module, function_name);
                        mi::mdl::IExpression_call* expr = expr_factory->create_call( ref);

                        unsigned killmask = get_parameter_killmask( function_name, module);

                        mi::Size n = call->get_parameter_count();
                        for( mi::Size i = 0; i < n; ++i) {
                            if( killmask & (1 << i))
                                continue;
                            const mi::mdl::IType* parameter_type = call->get_mdl_parameter_type(
                                transaction, static_cast<mi::Uint32>( i));
                            parameter_type = type_factory->import( parameter_type);

                            const char* parameter_name = call->get_parameter_name( i);
                            const mi::mdl::ISymbol* parameter_symbol
                                = name_factory->create_symbol( parameter_name);
                            const mi::mdl::ISimple_name* parameter_simple_name
                                = name_factory->create_simple_name( parameter_symbol);
                            mi::base::Handle<const IExpression> argument(
                                args->get_expression( i));
                            const mi::mdl::IExpression* argument_expr = int_expr_to_mdl_ast_expr(
                                transaction, module, parameter_type, argument.get(), call_trace);
                            if( !argument_expr)
                                return 0;
                            const mi::mdl::IArgument* argument_arg
                                = expr_factory->create_named_argument(
                                    parameter_simple_name, argument_expr);
                            expr->add_argument( argument_arg);
                        }
                        expr->set_type( call->get_mdl_return_type( transaction));
                        return expr;
                    }
                }

            } else if( class_id == Mdl_material_instance::id) {

                // handle material instance parameters
                DB::Access<Mdl_material_instance> material_instance( tag, transaction);
                const char* material_name = material_instance->get_mdl_material_definition();
                const mi::mdl::IExpression_reference* ref
                    = signature_to_reference( module, material_name);
                mi::mdl::IExpression_call* call = expr_factory->create_call( ref);

                mi::Size n = material_instance->get_parameter_count();
                for( mi::Size i = 0; i < n; ++i) {
                    const mi::mdl::IType* parameter_type =material_instance->get_mdl_parameter_type(
                        transaction, static_cast<mi::Uint32>( i));
                    parameter_type = type_factory->import( parameter_type);
                    const char* parameter_name = material_instance->get_parameter_name( i);
                    const mi::mdl::ISymbol* parameter_symbol
                        = name_factory->create_symbol( parameter_name);
                    const mi::mdl::ISimple_name* parameter_simple_name
                        = name_factory->create_simple_name( parameter_symbol);
                    mi::base::Handle<const IExpression_list> args(
                        material_instance->get_arguments());
                    mi::base::Handle<const IExpression> argument(
                        args->get_expression( i));
                    const mi::mdl::IExpression* argument_expr = int_expr_to_mdl_ast_expr(
                        transaction, module, parameter_type, argument.get(), call_trace);
                    if( !argument_expr)
                        return 0;
                    const mi::mdl::IArgument* argument_arg = expr_factory->create_named_argument(
                        parameter_simple_name, argument_expr);
                    call->add_argument( argument_arg);
                }
                call->set_type(
                    type_factory->get_predefined_struct( mi::mdl::IType_struct::SID_MATERIAL));
                return call;
            }

            const char* name = transaction->tag_to_name( tag);
            ASSERT( M_SCENE, name);
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "Unsupported type for call of \"%s\".", name?name:"");
            return 0;
        }
        case IExpression::EK_PARAMETER:
        case IExpression::EK_DIRECT_CALL:
        case IExpression::EK_TEMPORARY:
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

const mi::mdl::IExpression* int_expr_to_mdl_ast_expr(
    DB::Transaction* transaction,
    mi::mdl::IModule* module,
    const mi::mdl::IType* mdl_type,
    const IExpression* expr)
{
    std::set<MI::Uint32> call_trace;
    return int_expr_to_mdl_ast_expr( transaction, module, mdl_type, expr, call_trace);
}

const mi::mdl::DAG_node* int_expr_to_mdl_dag_node(
    DB::Transaction* transaction,
    mi::mdl::IDag_builder* builder,
    const mi::mdl::IType* type,
    const IExpression* expr,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max)
{
    return Mdl_dag_builder<mi::mdl::IDag_builder>(
        transaction, builder, mdl_meters_per_scene_unit, mdl_wavelength_min, mdl_wavelength_max,
        /*compiled_material*/ 0)
        .int_expr_to_mdl_dag_node( type, expr);
}

const mi::mdl::DAG_node* int_expr_to_mdl_dag_node(
    DB::Transaction* transaction,
    mi::mdl::IGenerated_code_dag::DAG_node_factory* factory,
    const mi::mdl::IType* type,
    const IExpression* expr,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max)
{
    return Mdl_dag_builder<mi::mdl::IGenerated_code_dag::DAG_node_factory>(
        transaction, factory, mdl_meters_per_scene_unit, mdl_wavelength_min, mdl_wavelength_max,
        /*compiled_material*/ 0)
        .int_expr_to_mdl_dag_node( type, expr);
}


// ********** Misc utility functions around MI::MDL ************************************************

std::string prefix_symbol_name( const char* symbol)
{
    if( !symbol)
        return "";
    if( strncmp( symbol, "::", 2) == 0)
        return symbol;
    ASSERT( M_SCENE, strcmp( symbol, "material_emission") == 0
                  || strcmp( symbol, "material_surface" ) == 0
                  || strcmp( symbol, "material_volume"  ) == 0
                  || strcmp( symbol, "material_geometry") == 0
                  || strcmp( symbol, "material"         ) == 0
                  || strcmp( symbol, "intensity_mode"   ) == 0);
    return std::string( "::") + symbol;
}

bool argument_type_matches_parameter_type(
    IType_factory* tf,
    const IType* argument_type,
    const IType* parameter_type)
{
    // Equal types (without modifiers) succeed.
    mi::base::Handle<const IType> parameter_type_stripped(
        parameter_type->skip_all_type_aliases());
    mi::base::Handle<const IType> argument_type_stripped(
        argument_type->skip_all_type_aliases());
    if( tf->compare( argument_type_stripped.get(), parameter_type_stripped.get()) == 0)
        return true;

    // Parameter type and argument type are different. Let all non-arrays fail since they need to
    // match exactly (modulo modifiers).
    mi::base::Handle<const IType_array> parameter_type_stripped_array(
        parameter_type_stripped->get_interface<IType_array>());
    if( !parameter_type_stripped_array)
        return false;
    mi::base::Handle<const IType_array> argument_type_stripped_array(
        argument_type_stripped->get_interface<IType_array>());
    if( !argument_type_stripped_array)
        return false;

    // Parameter type and argument type are different arrays. Let deferred-sized arguments for
    // immediate-sized parameters fail.
    bool parameter_type_immediate_sized = parameter_type_stripped_array->is_immediate_sized();
    bool argument_type_immediate_sized  = argument_type_stripped_array->is_immediate_sized();
    if( parameter_type_immediate_sized && !argument_type_immediate_sized)
        return false;

    // Let non-matching array lengths for immediate-sized arrays fail.
    if(    parameter_type_immediate_sized
        && parameter_type_stripped_array->get_size() != argument_type_stripped_array->get_size())
        return false;

    // Finally compare element types.
    mi::base::Handle<const IType> parameter_element_type(
        parameter_type_stripped_array->get_element_type());
    mi::base::Handle<const IType> argument_element_type(
        argument_type_stripped_array->get_element_type());
    return argument_type_matches_parameter_type(
        tf, argument_element_type.get(), parameter_element_type.get());
}

bool return_type_is_varying( DB::Transaction* transaction, const IExpression* argument)
{
    mi::base::Handle<const IExpression_call> argument_call(
        argument->get_interface<IExpression_call>());
    if( !argument_call)
        return false;

    DB::Tag tag = argument_call->get_call();
    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    if( class_id != Mdl_function_call::id)
        return false;

    DB::Access<Mdl_function_call> fc( tag, transaction);
    mi::base::Handle<const IType> return_type( fc->get_return_type());
    mi::Uint32 return_type_modifiers = return_type->get_all_type_modifiers();
    bool return_type_uniform = (return_type_modifiers & IType::MK_UNIFORM) != 0;
    if( return_type_uniform)
        return false;
    bool return_type_varying = (return_type_modifiers & IType::MK_VARYING) != 0;
    if( return_type_varying)
        return true;

    DB::Access<Mdl_function_definition> definition( fc->get_function_definition(), transaction);
    return !definition->is_uniform();
}

IExpression* deep_copy(
    const IExpression_factory* ef,
    DB::Transaction* transaction,
    const IExpression* expr,
    const std::vector<mi::base::Handle<const IExpression> >& context)
{
    IExpression::Kind kind = expr->get_kind();

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            mi::base::Handle<IValue_factory> vf( ef->get_value_factory());
            mi::base::Handle<IValue> value_clone( vf->clone( value.get()));
            return ef->create_constant( value_clone.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            DB::Tag tag = expr_call->get_call();
            SERIAL::Class_id class_id = transaction->get_class_id( tag);

            if( class_id == Mdl_function_call::id) {

                DB::Access<Mdl_function_call> original( tag, transaction);
                Mdl_function_call* copy
                    = static_cast<Mdl_function_call*>( original->copy());
                copy->make_mutable(transaction);

                mi::base::Handle<const IExpression_list> arguments( original->get_arguments());
                mi::base::Handle<IExpression_list> copy_arguments( ef->create_expression_list());
                mi::Size n = arguments->get_size();
                for( mi::Size i = 0; i < n; ++i) {
                    mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
                    mi::base::Handle<IExpression> copy_argument(
                        deep_copy( ef, transaction, argument.get(), context));
                    const char* name = arguments->get_name( i);
                    copy_arguments->add_expression( name, copy_argument.get());
                }
                mi::Sint32 result = copy->set_arguments( transaction, copy_arguments.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);

                std::string copy_name_prefix
                    = add_mdl_db_prefix( copy->get_mdl_function_definition());
                std::string copy_name
                    = DETAIL::generate_unique_db_name( transaction, copy_name_prefix.c_str());
                DB::Tag copy_tag = transaction->store_for_reference_counting(
                    copy, copy_name.c_str(), transaction->get_scope()->get_level());
                mi::base::Handle<const IType> type( expr->get_type());
                return ef->create_call( type.get(), copy_tag);

            } else if( class_id == Mdl_material_instance::id) {

                DB::Access<Mdl_material_instance> original( tag, transaction);
                Mdl_material_instance* copy
                    = static_cast<Mdl_material_instance*>( original->copy());
                copy->make_mutable(transaction);

                mi::base::Handle<const IExpression_list> arguments( original->get_arguments());
                mi::base::Handle<IExpression_list> copy_arguments( ef->create_expression_list());
                mi::Size n = arguments->get_size();
                for( mi::Size i = 0; i < n; ++i) {
                    mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
                    mi::base::Handle<IExpression> copy_argument(
                        deep_copy( ef, transaction, argument.get(), context));
                    const char* name = arguments->get_name( i);
                    copy_arguments->add_expression( name, copy_argument.get());
                }
                mi::Sint32 result = copy->set_arguments( transaction, copy_arguments.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);

                std::string copy_name_prefix
                    = add_mdl_db_prefix( copy->get_mdl_material_definition());
                std::string copy_name
                    = DETAIL::generate_unique_db_name( transaction, copy_name_prefix.c_str());
                DB::Tag copy_tag = transaction->store_for_reference_counting(
                    copy, copy_name.c_str(), transaction->get_scope()->get_level());
                mi::base::Handle<const IType> type( expr->get_type());
                return ef->create_call( type.get(), copy_tag);

            } else {

                ASSERT( M_SCENE, false);
                return 0;
            }
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IExpression_parameter> expr_parameter(
                expr->get_interface<IExpression_parameter>());
            mi::Size index = expr_parameter->get_index();
            ASSERT( M_SCENE, index < context.size());
            return deep_copy( ef, transaction, context[index].get(), context);
        }
        case IExpression::EK_DIRECT_CALL:
        case IExpression::EK_TEMPORARY:
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

/// Prefix of DB elements for MDL modules/definitions.
static const char* mdl_db_prefix = "mdl";

std::string add_mdl_db_prefix( const std::string& name)
{
    std::string result( mdl_db_prefix);
    if( name.substr( 0, 2) != "::")
        result += "::";
    result += name;
    return result;
}

namespace {

/// Combines all bits of a size_t into a mi::Uint32.
mi::Uint32 hash_size_t_as_uint32( size_t hash)
{
#ifdef MI_ARCH_64BIT
    return static_cast<mi::Uint32>( (hash >> 32) + (hash & 0xffffffff));
#else
    return hash;
#endif
}

} // namespace

mi::Uint32 get_hash( const std::string& mdl_file_path, DB::Tag_version tv)
{
    if( !mdl_file_path.empty()) {
        const char* begin = mdl_file_path.c_str();
        const char* end   = begin + mdl_file_path.size();
        return hash_size_t_as_uint32( boost::hash_range( begin, end));
    }

    std::vector<mi::Uint32> v;
    v.push_back( tv.m_transaction_id.get_uint());
    v.push_back( tv.m_tag.get_uint());
    v.push_back( tv.m_version);
    return hash_size_t_as_uint32( boost::hash_range( v.begin(), v.end()));
}

mi::Uint32 get_hash( const char* mdl_file_path, DB::Tag_version tv)
{
    std::string s( mdl_file_path != NULL ? mdl_file_path : "");
    return get_hash( s, tv);
}

mi::Uint32 get_hash(
    const std::string& mdl_file_path, mi::Float32 gamma, DB::Tag_version tv1, DB::Tag_version tv2)
{
    if( !mdl_file_path.empty()) {
        const char* begin = mdl_file_path.c_str();
        const char* end   = begin + mdl_file_path.size();
        size_t hash = boost::hash_range( begin, end);
        boost::hash_combine( hash, gamma);
        return hash_size_t_as_uint32( hash);
    }

    std::vector<mi::Uint32> v;
    v.push_back( tv1.m_transaction_id.get_uint());
    v.push_back( tv1.m_tag.get_uint());
    v.push_back( tv1.m_version);
    v.push_back( tv2.m_transaction_id.get_uint());
    v.push_back( tv2.m_tag.get_uint());
    v.push_back( tv2.m_version);
    return hash_size_t_as_uint32( boost::hash_range( v.begin(), v.end()));
}

mi::Uint32 get_hash(
    const char* mdl_file_path, mi::Float32 gamma, DB::Tag_version tv1, DB::Tag_version tv2)
{
    std::string s( mdl_file_path != NULL ? mdl_file_path : "");
    return get_hash( s, gamma, tv1, tv2);
}

namespace {

/// Wraps a string as mi::neuraylib::IBuffer.
class Buffer_wrapper
  : public mi::base::Interface_implement<mi::neuraylib::IBuffer>,
    public boost::noncopyable
{
public:
    Buffer_wrapper( const std::string& data) : m_data( data) { }
    const mi::Uint8* get_data() const { return reinterpret_cast<const mi::Uint8*>( m_data.c_str());}
    mi::Size get_data_size() const { return m_data.size(); }
private:
    const std::string m_data;
};

} // namespace

mi::neuraylib::IReader* create_reader( const std::string& data)
{
    mi::base::Handle<mi::neuraylib::IBuffer> buffer( new Buffer_wrapper( data));
    return new DISK::Memory_reader_impl( buffer.get());
}


// ********** Misc utility functions around mi::mdl ************************************************

const mi::mdl::IType_compound* convert_deferred_sized_into_immediate_sized_array(
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_compound* type,
    mi::Size size)
{
    const mi::mdl::IType_array* type_array = as<mi::mdl::IType_array>( type);
    if( !type_array)
        return type;
    if( type_array->is_immediate_sized())
        return type;
    const mi::mdl::IType* element_type = type_array->get_element_type();
    mi::mdl::IType_factory* tf = vf->get_type_factory();
    return as<mi::mdl::IType_compound>( tf->create_array( element_type, size));
}

const mi::mdl::IExpression_reference* signature_to_reference(
    mi::mdl::IModule* module,
    const char* signature)
{
    mi::mdl::IName_factory &nf = *module->get_name_factory();

    mi::mdl::IQualified_name* qualified_name = nf.create_qualified_name();
    if( signature[0] == ':' && signature[1] == ':') {
        qualified_name->set_absolute();
        signature += 2;
    }

    const mi::mdl::ISymbol* symbol = 0;
    for( const char* pos = signature; signature != 0; ++pos) {
        if( pos[0] == ':' && pos[1] == ':') {
            std::string component( signature, pos - signature);
            symbol = nf.create_symbol( component.c_str());
            signature = pos + 2;
            ++pos;
        } else if( pos[0] == '(') {
            std::string component( signature, pos - signature);
            symbol = nf.create_symbol( component.c_str());
            signature = 0;
        } else if( pos[0] == '\0') {
            symbol = nf.create_symbol( signature);
            signature = 0;
        } else {
            continue;
        }
        const mi::mdl::ISimple_name* simple_name = nf.create_simple_name( symbol);
        qualified_name->add_component( simple_name);
    }

    const mi::mdl::IType_name* type_name = nf.create_type_name( qualified_name);
    mi::mdl::IExpression_factory &ef = *module->get_expression_factory();
    return ef.create_reference( type_name);
}

mi::mdl::IExpression_reference* type_to_reference(
    mi::mdl::IModule* module, const mi::mdl::IType* type)
{
    char buf[32];
    char const *s = 0;

    type = type->skip_type_alias();
    switch( type->get_kind()) {
    case mi::mdl::IType::TK_ALIAS:
    case mi::mdl::IType::TK_INCOMPLETE:
    case mi::mdl::IType::TK_ERROR:
    case mi::mdl::IType::TK_FUNCTION:
        ASSERT( M_SCENE, !"unexpected MDL type kind");
        return NULL;

    case mi::mdl::IType::TK_BOOL:
        s = "bool";
        break;
    case mi::mdl::IType::TK_INT:
        s = "int";
        break;
    case mi::mdl::IType::TK_ENUM:
        s = as<mi::mdl::IType_enum>( type)->get_symbol()->get_name();
        break;
    case mi::mdl::IType::TK_FLOAT:
        s = "float";
        break;
    case mi::mdl::IType::TK_DOUBLE:
        s = "double";
        break;
    case mi::mdl::IType::TK_STRING:
        s = "string";
        break;
    case mi::mdl::IType::TK_LIGHT_PROFILE:
        s = "light_profile";
        break;
    case mi::mdl::IType::TK_BSDF:
        s = "bsdf";
        break;
    case mi::mdl::IType::TK_EDF:
        s = "edf";
        break;
    case mi::mdl::IType::TK_VDF:
        s = "vdf";
        break;
    case mi::mdl::IType::TK_VECTOR:
        {
            const mi::mdl::IType_vector* v_type = as<mi::mdl::IType_vector>( type);
            const mi::mdl::IType_atomic* a_type = v_type->get_element_type();
            int size = v_type->get_size();

            switch( a_type->get_kind()) {
            case mi::mdl::IType::TK_BOOL:
                switch( size) {
                case 2: s = "bool2"; break;
                case 3: s = "bool3"; break;
                case 4: s = "bool4"; break;
                }
                break;
            case mi::mdl::IType::TK_INT:
                switch( size) {
                case 2: s = "int2"; break;
                case 3: s = "int3"; break;
                case 4: s = "int4"; break;
                }
                break;
            case mi::mdl::IType::TK_FLOAT:
                switch( size) {
                case 2: s = "float2"; break;
                case 3: s = "float3"; break;
                case 4: s = "float4"; break;
                }
                break;
            case mi::mdl::IType::TK_DOUBLE:
                switch( size) {
                case 2: s = "double2"; break;
                case 3: s = "double3"; break;
                case 4: s = "double4"; break;
                }
                break;
            default:
                ASSERT( M_SCENE, !"Unexpected type kind");
            }
        }
        break;
    case mi::mdl::IType::TK_MATRIX:
        {
            const mi::mdl::IType_matrix *m_type = as<mi::mdl::IType_matrix>( type);
            const mi::mdl::IType_vector *e_type = m_type->get_element_type();
            const mi::mdl::IType_atomic *a_type = e_type->get_element_type();

            snprintf( buf, sizeof( buf), "%s%dx%d",
                a_type->get_kind() == mi::mdl::IType::TK_FLOAT ? "float" : "double",
                m_type->get_columns(),
                e_type->get_size());
            buf[sizeof( buf) - 1] = '\0';
            s = buf;
        }
        break;
    case mi::mdl::IType::TK_ARRAY:
        {
            const mi::mdl::IType_array* a_type = as<mi::mdl::IType_array>( type);

            mi::mdl::IExpression_reference* ref
                = type_to_reference( module, a_type->get_element_type());
            ref->set_array_constructor();
            return ref;
        }
    case mi::mdl::IType::TK_COLOR:
        s = "color";
        break;
    case mi::mdl::IType::TK_STRUCT:
        s = mi::mdl::as<mi::mdl::IType_struct>(type)->get_symbol()->get_name();
        break;
    case mi::mdl::IType::TK_TEXTURE:
        {
            const mi::mdl::IType_texture *t_type = as<mi::mdl::IType_texture>( type);

            switch( t_type->get_shape()) {
            case mi::mdl::IType_texture::TS_2D:
                s = "texture_2d";
                break;
            case mi::mdl::IType_texture::TS_3D:
                s = "texture_3d";
                break;
            case mi::mdl::IType_texture::TS_CUBE:
                s = "texture_cube";
                break;
            case mi::mdl::IType_texture::TS_PTEX:
                s = "texture_ptex";
                break;
            }
        }
        break;
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        s = "bsdf_measurement";
        break;
    }
    ASSERT( M_SCENE, s);

    mi::mdl::IName_factory &nf = *module->get_name_factory();

    mi::mdl::IQualified_name* qualified_name = nf.create_qualified_name();
    if( s[0] == ':' && s[1] == ':') {
        qualified_name->set_absolute();
        s += 2;
    }

    const mi::mdl::ISymbol* symbol = 0;
    for( const char* pos = s; s != 0; ++pos) {
        if( pos[0] == ':' && pos[1] == ':') {
            std::string component( s, pos - s);
            symbol = nf.create_symbol( component.c_str());
            s = pos + 2;
            ++pos;
        } else if( pos[0] == '\0') {
            symbol = nf.create_symbol( s);
            s = 0;
        } else {
            continue;
        }
        const mi::mdl::ISimple_name* simple_name = nf.create_simple_name( symbol);
        qualified_name->add_component( simple_name);
    }

    const mi::mdl::IType_name* type_name = nf.create_type_name( qualified_name);
    mi::mdl::IExpression_factory &ef = *module->get_expression_factory();
    return ef.create_reference( type_name);
}

namespace {

/// Associates all resource literals inside a subtree of a code DAG with its DB tags.
///
/// \param transaction      The DB transaction to use (to retrieve resource tags).
/// \param code_dag         The code DAG to update.
/// \param module_filename  The file name of the module.
/// \param module_name      The name of the module.
/// \param node             The subtree of \p code_dag to traverse.
void update_resource_literals(
    DB::Transaction* transaction,
    mi::mdl::IGenerated_code_dag* code_dag,
    const char* module_filename,
    const char* module_name,
    const mi::mdl::DAG_node* node)
{
    switch( node->get_kind()) {
        case mi::mdl::DAG_node::EK_CONSTANT: {
            const mi::mdl::DAG_constant* constant = as<mi::mdl::DAG_constant>( node);
            const mi::mdl::IValue_resource* resource
                = as<mi::mdl::IValue_resource>( constant->get_value());
            if( resource) {
                DB::Tag tag = DETAIL::mdl_resource_to_tag(
                    transaction, resource, module_filename, module_name);
                if( tag) {
                    DB::Tag_version tag_version = transaction->get_tag_version( tag);
                    code_dag->set_resource_tag( constant, tag.get_uint(), tag_version.m_version);
                }
            }
            return;
        }
        case mi::mdl::DAG_node::EK_PARAMETER:
            return;
        case mi::mdl::DAG_node::EK_TEMPORARY:
            return; // the referenced temporary will be traversed explicitly
        case mi::mdl::DAG_node::EK_CALL: {
            const mi::mdl::DAG_call* call = as<mi::mdl::DAG_call>( node);
            mi::Uint32 n = call->get_argument_count();
            for( mi::Uint32 i = 0; i < n; ++i)
                update_resource_literals(
                    transaction, code_dag, module_filename, module_name, call->get_argument( i));
            return;
        }
    }
}

} // namespace

void update_resource_literals(
    DB::Transaction* transaction,
    mi::mdl::IGenerated_code_dag* code_dag,
    const char* module_filename,
    const char* module_name)
{
    mi::Uint32 material_count = code_dag->get_material_count();
    for( mi::Uint32 i = 0; i < material_count; ++i) {

        // traverse parameters
        mi::Uint32 parameter_count = code_dag->get_material_parameter_count(i);
        for (mi::Uint32 j = 0; j < parameter_count; ++j) {
            const mi::mdl::DAG_node* parameter_default =
                code_dag->get_material_parameter_default(i, j);
            if (parameter_default)
                update_resource_literals(
                    transaction, code_dag, module_filename, module_name, parameter_default);
        }

        // traverse body
        const mi::mdl::DAG_node* body = code_dag->get_material_value( i);
        update_resource_literals( transaction, code_dag, module_filename, module_name, body);

        // traverse temporaries
        mi::Uint32 temporary_count = code_dag->get_material_temporary_count( i);
        for( mi::Uint32 j = 0; j < temporary_count; ++j) {
            const mi::mdl::DAG_node* temporary = code_dag->get_material_temporary( i, j);
            update_resource_literals(
                transaction, code_dag, module_filename, module_name, temporary);
        }
    }
    mi::Uint32 function_count = code_dag->get_function_count();
    for (mi::Uint32 i = 0; i < function_count; ++i) {

        // traverse parameters
        mi::Uint32 parameter_count = code_dag->get_function_parameter_count(i);
        for (mi::Uint32 j = 0; j < parameter_count; ++j) {
            const mi::mdl::DAG_node* parameter_default =
                code_dag->get_function_parameter_default(i, j);
            if (parameter_default)
                update_resource_literals(
                    transaction, code_dag, module_filename, module_name, parameter_default);
        }
    }
}

namespace {

/// Collects all resource references in a DAG node.
///
/// \param transaction            The DB transaction to use (needed to convert strings into tags).
/// \param code_dag               The DAG node to traverse.
/// \param[out] references        The collected references are added to this container.
static void collect_resource_references(
    const mi::mdl::DAG_node* node,
    std::set<const mi::mdl::IValue_resource*>& references)
{
    switch( node->get_kind()) {
        case mi::mdl::DAG_node::EK_CONSTANT: {
            const mi::mdl::DAG_constant* constant = as<mi::mdl::DAG_constant>( node);
            const mi::mdl::IValue_resource* resource
                = as<mi::mdl::IValue_resource>( constant->get_value());
            if( resource)
                references.insert( resource);
            return;
        }
        case mi::mdl::DAG_node::EK_PARAMETER:
            return; // the referenced parameter will be traversed explicitly
        case mi::mdl::DAG_node::EK_TEMPORARY:
            return; // the referenced temporary will be traversed explicitly
        case mi::mdl::DAG_node::EK_CALL: {
            const mi::mdl::DAG_call* call = as<mi::mdl::DAG_call>( node);
            mi::Uint32 n = call->get_argument_count();
            for( mi::Uint32 i = 0; i < n; ++i)
                collect_resource_references( call->get_argument( i), references);
            return;
        }
    }
}

} // namespace

void collect_resource_references(
    const mi::mdl::IGenerated_code_dag* code_dag,
    std::set<const mi::mdl::IValue_resource*>& references)
{
    mi::Uint32 material_count = code_dag->get_material_count();
    for( mi::Uint32 i = 0; i < material_count; ++i) {

        // traverse parameters
        mi::Uint32 parameter_count = code_dag->get_material_parameter_count( i);
        for (mi::Uint32 j = 0; j < parameter_count; ++j) {
            const mi::mdl::DAG_node* parameter_default = 
                code_dag->get_material_parameter_default( i, j);
            if( parameter_default)
                collect_resource_references( parameter_default, references);
        }

        // traverse body
        const mi::mdl::DAG_node* body = code_dag->get_material_value( i);
        collect_resource_references( body, references);

        // traverse temporaries
        mi::Uint32 temporary_count = code_dag->get_material_temporary_count( i);
        for( mi::Uint32 j = 0; j < temporary_count; ++j) {
            const mi::mdl::DAG_node* temporary = code_dag->get_material_temporary( i, j);
            collect_resource_references( temporary, references);
        }
    }
    mi::Uint32 function_count = code_dag->get_function_count();
    for (mi::Uint32 i = 0; i < function_count; ++i) {

        // traverse parameters
        mi::Uint32 parameter_count = code_dag->get_function_parameter_count(i);
        for (mi::Uint32 j = 0; j < parameter_count; ++j) {
            const mi::mdl::DAG_node* parameter_default =
                code_dag->get_function_parameter_default(i, j);
            if (parameter_default)
                collect_resource_references(parameter_default, references);
        }
    }
}

namespace {

/// Collects all call references in a DAG node.
///
/// Parameter and temporary references are skipped.
///
/// \param transaction       The DB transaction to use (needed to convert strings into tags).
/// \param node              The DAG node to traverse.
/// \param[out] references   The collected references are added to this set.
static void collect_material_references(
    DB::Transaction* transaction,
    const mi::mdl::DAG_node* node,
    DB::Tag_set& references)
{
    switch( node->get_kind()) {
        case mi::mdl::DAG_node::EK_CONSTANT:
            return; // nothing to do
        case mi::mdl::DAG_node::EK_PARAMETER:
            return; // the referenced parameter will be traversed explicitly
        case mi::mdl::DAG_node::EK_TEMPORARY:
            return; // the referenced temporary will be traversed explicitly
        case mi::mdl::DAG_node::EK_CALL: {
            const mi::mdl::DAG_call* call = as<mi::mdl::DAG_call>( node);
            std::string db_name = add_mdl_db_prefix( call->get_name());
            DB::Tag tag = transaction->name_to_tag( db_name.c_str());
            ASSERT( M_SCENE, tag);
            references.insert( tag);
            mi::Uint32 n = call->get_argument_count();
            for( mi::Uint32 i = 0; i < n; ++i)
                collect_material_references( transaction, call->get_argument( i), references);
            return;
        }
    }
}

} // namespace

void collect_material_references(
    DB::Transaction* transaction,
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Uint32 material_index,
    DB::Tag_set& references)
{
    // traverse body
    const mi::mdl::DAG_node* body = code_dag->get_material_value( material_index);
    collect_material_references( transaction, body, references);

    // traverse temporaries
    mi::Uint32 n = code_dag->get_material_temporary_count( material_index);
    for( mi::Uint32 i = 0; i < n; ++i) {
        const mi::mdl::DAG_node* temporary = code_dag->get_material_temporary( material_index, i);
        collect_material_references( transaction, temporary, references);
    }
}

void collect_function_references(
    DB::Transaction* transaction,
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Uint32 function_index,
    DB::Tag_set& references)
{
    mi::Uint32 n = code_dag->get_function_references_count( function_index);
    for( mi::Uint32 i = 0; i < n; ++i) {
        char const* reference = code_dag->get_function_reference( function_index, i);
        std::string db_name = add_mdl_db_prefix( reference);
        DB::Tag tag = transaction->name_to_tag( db_name.c_str());
        ASSERT( M_SCENE, tag);
        references.insert( tag);
    }
}

namespace {

/// Outputs a given compiler message plus severity to the logger.
///
/// Also puts it into \p context (unless \p context is \c NULL).
void log_compiler_message(const Message& message, Execution_context* context)
{
    switch(message.m_severity) {
        case mi::base::MESSAGE_SEVERITY_ERROR:
            LOG::mod_log->error(
                M_MDLC, LOG::Mod_log::C_COMPILER, "%s", message.m_message.c_str());
            if (context) {
                context->add_message(message);
                context->add_error_message(message);
            }
            return;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            LOG::mod_log->warning(
                M_MDLC, LOG::Mod_log::C_COMPILER, "%s", message.m_message.c_str());
            if(context)
                context->add_message(message);
            return;
        case mi::base::MESSAGE_SEVERITY_INFO:
            LOG::mod_log->info(
                M_MDLC, LOG::Mod_log::C_COMPILER, "%s", message.m_message.c_str());
            if(context)
                context->add_message(message);
            return;
        default:
            break;
    }
}

} // namespace
void report_messages(const mi::mdl::Messages& in_messages, Execution_context* context)
{
    mi::Uint32 message_count = in_messages.get_message_count();
    for (mi::Uint32 i = 0; i < message_count; i++) {

        Message message(in_messages.get_message(i));
        log_compiler_message(message, context);

        mi::Uint32 note_count = message.m_notes.size();
        for (mi::Uint32 j = 0; j < note_count; j++) {

            log_compiler_message(message.m_notes[j], context);
        }
    }
}


mi::neuraylib::IReader* get_reader( mi::mdl::IMDL_resource_reader* reader)
{
    return new DETAIL::File_reader_impl( reader);
}

IMAGE::IMdr_callback* create_mdr_callback()
{
    return new DETAIL::Mdr_callback();
}

// ********** Name_importer ************************************************************************

/// Helper class to find all necessary imports.
class Name_importer : public mi::mdl::Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param module  the module which will be processed
    Name_importer( mi::mdl::IModule* module) : m_module( module) {}

    /// Destructor.
    virtual ~Name_importer() {}

    /// Inserts a fully qualified name.
    void insert( const char* name) { m_imports.insert( name); }

    /// Inserts a fully qualified name.
    void insert( const std::string& name) { m_imports.insert( name); }

    /// Post-visits a reference expression.
    virtual void post_visit( mi::mdl::IType_name* name)
    {
        const mi::mdl::IType* type = name->get_type();
        if( type && type->get_kind() == mi::mdl::IType::TK_ENUM) {
            // for enum values, the type must be imported
            handle_type( type);
            return;
        }

        // We assume that every absolute name must be imported:
        // Function and material names are taken from the neuray DB, all of them are absolute
        // Enum values are converted into scoped literals, they have no name.
        if( !name->is_absolute())
            return;

        const mi::mdl::IQualified_name* qualified_name = name->get_qualified_name();
        std::string absolute_name;
        mi::Uint32 n = qualified_name->get_component_count();
        for( mi::Uint32 i = 0; i < n; ++i) {
            const mi::mdl::ISimple_name* component = qualified_name->get_component( i);
            absolute_name += "::";
            absolute_name += component->get_symbol()->get_name();
        }
        insert( absolute_name);
    }

    /// Post-visits an annotation.
    virtual void post_visit( mi::mdl::IAnnotation* annotation)
    {
        const mi::mdl::IQualified_name* qualified_name = annotation->get_name();

        // We assume that every absolute name must be imported:
        // Function and material names are taken from the neuray DB, all of them are absolute
        // Enum values are converted into scoped literals, they have no name.
        if( !qualified_name->is_absolute())
            return;

        std::string absolute_name;
        mi::Uint32 n = qualified_name->get_component_count();
        for( mi::Uint32 i = 0; i < n; ++i) {
            const mi::mdl::ISimple_name* component = qualified_name->get_component( i);
            absolute_name += "::";
            absolute_name += component->get_symbol()->get_name();
        }
        insert( absolute_name);

    }

    /// Post-visits literals.
    virtual void post_visit( mi::mdl::IExpression_literal* literal)
    {
        const mi::mdl::IValue* value = literal->get_value();
        handle_type( value->get_type());
    }

    /// Post-visits binary expressions.
    virtual void post_visit( mi::mdl::IExpression_binary* expr)
    {
        if( expr->get_operator() == mi::mdl::IExpression_binary::OK_SELECT) {
            const mi::mdl::IExpression* left = expr->get_left_argument();
            if( const mi::mdl::IType* left_type = left->get_type()) {
                if( const mi::mdl::IType_struct* struct_type =
                    as<mi::mdl::IType_struct>( left_type))
                {
                    handle_struct_types( struct_type);
                }
            }
        }
    }

    /// Returns the found imports.
    const std::set<std::string>& get_imports() const { return m_imports; }

    /// Returns the module.
    mi::mdl::IModule* get_module() { return m_module; }

private:
    /// Handle all necessary imports of a struct type.
    void handle_struct_types( const mi::mdl::IType_struct* struct_type)
    {
        const char* struct_name = struct_type->get_symbol()->get_name();
        insert( struct_name);

        for( int i = 0, n = struct_type->get_field_count(); i < n; ++i) {
            const mi::mdl::ISymbol* mem_sym;
            const mi::mdl::IType* mem_type;

            struct_type->get_field( i, mem_type, mem_sym);
            handle_type( mem_type);
        }
    }

    /// Handle all necessary imports of a type.
    void handle_type( const mi::mdl::IType* type)
    {
        switch( type->get_kind()) {
        case mi::mdl::IType::TK_ALIAS:
            return handle_type( type->skip_type_alias());
        case mi::mdl::IType::TK_ENUM:
            {
                const mi::mdl::IType_enum* enum_type = as<mi::mdl::IType_enum>( type);
                if( enum_type->get_predefined_id() != mi::mdl::IType_enum::EID_INTENSITY_MODE) {
                    const char* enum_name = enum_type->get_symbol()->get_name();
                    insert( enum_name);
                }
            }
            break;
        case mi::mdl::IType::TK_TEXTURE:
            // for texture values, import tex::gamma_mode
            insert( "::tex::gamma_mode");
            break;
        case mi::mdl::IType::TK_STRUCT:
            {
                const mi::mdl::IType_struct* struct_type = as<mi::mdl::IType_struct>( type);
                if( struct_type->get_predefined_id() == mi::mdl::IType_struct::SID_USER) {
                    // only import user types, others are built-in
                    handle_struct_types( struct_type);
                }
            }
            break;
        case mi::mdl::IType::TK_ARRAY:
            {
                const mi::mdl::IType_array* array_type = as<mi::mdl::IType_array>( type);
                handle_type( array_type->get_element_type());
            }
            break;
        default:
            break;
        }
    }

private:
    /// The module.
    mi::mdl::IModule* m_module;

    /// The import set.
    std::set<std::string> m_imports;
};


// ********** Symbol_importer **********************************************************************

Symbol_importer::Symbol_importer( mi::mdl::IModule* module)
  : m_name_importer( new Name_importer( module))
{
}

Symbol_importer::~Symbol_importer()
{
    delete m_name_importer;
}

void Symbol_importer::collect_imports( const mi::mdl::IExpression* expr)
{
    m_name_importer->visit( expr);
}

void Symbol_importer::collect_imports( const mi::mdl::IAnnotation_block* annotation_block)
{
    m_name_importer->visit( annotation_block);
}

// Add names from a list.
void Symbol_importer::add_names( const Name_list& names)
{
    for (Name_list::const_iterator it(names.begin()), end(names.end()); it != end; ++it) {
        m_name_importer->insert( *it);
    }
}

void Symbol_importer::add_imports()
{
    typedef std::set<std::string> String_set;

    mi::mdl::IModule* module = m_name_importer->get_module();
    const String_set& imports = m_name_importer->get_imports();
    for( String_set::const_iterator it = imports.begin(); it != imports.end(); ++it) {
        const std::string& import = *it;
        // do not import intensity_mode, this is a keyword in MDL 1.1
        if( import == "intensity_mode")
            continue;
        module->add_import( import.c_str());
    }
}


// ********** Module_cache *************************************************************************

Module_cache::~Module_cache()
{
}

const mi::mdl::IModule* Module_cache::lookup( const char* module_name) const
{
    std::string db_name = add_mdl_db_prefix( module_name);
    DB::Tag tag = m_transaction->name_to_tag( db_name.c_str());
    if( !tag)
        return 0;

    if( m_transaction->get_class_id( tag) != Mdl_module::id)
        return 0;

    DB::Access<Mdl_module> module( tag, m_transaction);
    return module->get_mdl_module();
}


// ********** Call_evaluator ***********************************************************************

const mi::mdl::IValue* Call_evaluator::evaluate_intrinsic_function(
    mi::mdl::IValue_factory* value_factory,
    mi::mdl::IDefinition::Semantics semantic,
    const mi::mdl::IValue* const arguments[],
    size_t n_arguments) const
{
    switch( semantic) {
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_df_light_profile_power( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_df_light_profile_maximum( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_df_light_profile_isvalid( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_df_bsdf_measurement_isvalid( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_WIDTH:
            ASSERT( M_SCENE, arguments && (n_arguments == 1 || n_arguments == 2));
            return fold_tex_width(
                value_factory,
                arguments[0],
                n_arguments == 2 ? arguments[1] : NULL);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT:
            ASSERT( M_SCENE, arguments && (n_arguments == 1 || n_arguments == 2));
            return fold_tex_height(
                value_factory,
                arguments[0],
                n_arguments == 2 ? arguments[1] : NULL);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_tex_depth( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_tex_texture_isvalid( value_factory, arguments[0]);

        default:
            return value_factory->create_bad();
    }
}

const mi::mdl::IValue* Call_evaluator::fold_df_light_profile_power(
    mi::mdl::IValue_factory* value_factory, const mi::mdl::IValue* argument) const
{
    bool valid;
    float power, maximum;
    if( get_light_profile_attributes( m_transaction, argument, valid, power, maximum))
        return value_factory->create_float( power);
    ASSERT( M_SCENE, !"not a light profile value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_df_light_profile_maximum(
    mi::mdl::IValue_factory* value_factory, const mi::mdl::IValue* argument) const
{
    bool valid;
    float power, maximum;
    if( get_light_profile_attributes( m_transaction, argument, valid, power, maximum))
        return value_factory->create_float( maximum);
    ASSERT( M_SCENE, !"not a light profile value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_df_light_profile_isvalid(
    mi::mdl::IValue_factory* value_factory, const mi::mdl::IValue* argument) const
{
    bool valid;
    float power, maximum;
    if( get_light_profile_attributes( m_transaction, argument, valid, power, maximum))
        return value_factory->create_bool( valid);
    ASSERT( M_SCENE, !"not a light profile value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_df_bsdf_measurement_isvalid(
    mi::mdl::IValue_factory* value_factory, const mi::mdl::IValue* argument) const
{
    bool valid;
    if( get_bsdf_measurement_attributes( m_transaction, argument, valid))
        return value_factory->create_bool( valid);
    ASSERT( M_SCENE, !"not a bsdf measurement value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_tex_width(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument,
    const mi::mdl::IValue* uvtile_arg) const
{
    bool valid, is_uvtile;
    int width, height, depth;
    if( get_texture_attributes( m_transaction, argument, valid, is_uvtile, width, height, depth)) {
        if ( !is_uvtile) {
            return value_factory->create_int( width);
        } else if ( uvtile_arg != NULL && is<mi::mdl::IValue_vector>( uvtile_arg)) {
            mi::mdl::IValue_vector const *uvtile = cast<mi::mdl::IValue_vector>( uvtile_arg);
            mi::mdl::IValue_int const *x = cast<mi::mdl::IValue_int>( uvtile->get_value( 0));
            mi::mdl::IValue_int const *y = cast<mi::mdl::IValue_int>( uvtile->get_value( 1));
            if ( get_texture_uvtile_resolution(
                m_transaction,
                argument,
                mi::Sint32_2( x->get_value(), y->get_value()),
                width,
                height))
            {
                return value_factory->create_int( width);
            }
        }
    }
    ASSERT( M_SCENE, !"not a texture value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_tex_height(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument,
    const mi::mdl::IValue* uvtile_arg) const
{
    bool valid, is_uvtile;
    int width, height, depth;
    if( get_texture_attributes( m_transaction, argument, valid, is_uvtile, width, height, depth))
    {
        if ( !is_uvtile) {
            return value_factory->create_int( height);
        } else if ( uvtile_arg != NULL && is<mi::mdl::IValue_vector>( uvtile_arg)) {
            mi::mdl::IValue_vector const *uvtile = cast<mi::mdl::IValue_vector>( uvtile_arg);
            mi::mdl::IValue_int const *x = cast<mi::mdl::IValue_int>( uvtile->get_value( 0));
            mi::mdl::IValue_int const *y = cast<mi::mdl::IValue_int>( uvtile->get_value( 1));
            if ( get_texture_uvtile_resolution(
                m_transaction,
                argument,
                mi::Sint32_2( x->get_value(), y->get_value()),
                width,
                height))
            {
                return value_factory->create_int( height);
            }
        }
    }
    ASSERT( M_SCENE, !"not a texture value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_tex_depth(
    mi::mdl::IValue_factory* value_factory, const mi::mdl::IValue* argument) const
{
    bool valid, is_uvtile;
    int width, height, depth;
    if( get_texture_attributes( m_transaction, argument, valid, is_uvtile, width, height, depth))
        return value_factory->create_int( depth);
    ASSERT( M_SCENE, !"not a texture value");
    return value_factory->create_bad();
}

const mi::mdl::IValue* Call_evaluator::fold_tex_texture_isvalid(
    mi::mdl::IValue_factory* value_factory, const mi::mdl::IValue* argument) const
{
    bool valid, is_uvtile;
    int width, height, depth;
    if( get_texture_attributes( m_transaction, argument, valid, is_uvtile, width, height, depth))
        return value_factory->create_bool( valid);
    ASSERT( M_SCENE, !"not a texture value");
    return value_factory->create_bad();
}

namespace {

mi::base::Message_severity convert_severity(mi::mdl::IMessage::Severity severity)
{
    switch (severity) {

    case mi::mdl::IMessage::MS_INFO:
        return mi::base::MESSAGE_SEVERITY_INFO;
    case mi::mdl::IMessage::MS_WARNING:
        return mi::base::MESSAGE_SEVERITY_WARNING;
    case mi::mdl::IMessage::MS_ERROR:
        return mi::base::MESSAGE_SEVERITY_ERROR;
    default:
        break;
    }
    return mi::base::MESSAGE_SEVERITY_FORCE_32_BIT;
}
} // anonymous

Message::Message(const mi::mdl::IMessage *message) 
    : m_severity(convert_severity(message->get_severity()))
    , m_code(message->get_code())
    , m_message(message->get_string())
{
    switch (message->get_class())
    {
    case 'A':
        m_kind = MSG_COMPILER_ARCHIVE_TOOL;
        break;
    case 'C':
        m_kind = MSG_COMILER_CORE;
        break;
    case 'J':
        m_kind = MSG_COMILER_BACKEND;
        break;
    default:
        m_kind = MSG_UNCATEGORIZED;
        break;
    }

    std::string msg;
    const char* file = message->get_file();
    if (file && file[0])
        msg += file;

    const mi::mdl::Position* position = message->get_position();
    mi::Uint32 line = position->get_start_line();
    mi::Uint32 column = position->get_start_column();
    if (line > 0)
        msg += "(" + std::to_string(line) + "," + std::to_string(column) + ")";
    if ((file && file[0]) || (line > 0))
        msg += ": ";

    m_message = msg + m_message;

    for (int i = 0; i < message->get_note_count(); ++i)
        m_notes.push_back(Message(message->get_note(i)));
}

namespace {

bool validate_space(const STLEXT::Any& value)
{
    if (value.type() != typeid(std::string))
        return false;
    const std::string& s = STLEXT::any_cast<const std::string&>(value);
    if (s == "coordinate_object" ||
        s == "coordinate_world")
        return true;
    return false;
}
}

Execution_context::Execution_context() : m_result(0)
{
    add_option(Option(MDL_CTX_OPTION_INTERNAL_SPACE, std::string("coordinate_world"),
        validate_space));
    add_option(Option(MDL_CTX_OPTION_METERS_PER_SCENE_UNIT, 1.0f));
    add_option(Option(MDL_CTX_OPTION_WAVELENGTH_MIN, 380.f));
    add_option(Option(MDL_CTX_OPTION_WAVELENGTH_MAX, 780.f));
    add_option(Option(MDL_CTX_OPTION_INCLUDE_GEO_NORMAL, true));
    add_option(Option(MDL_CTX_OPTION_BUNDLE_RESOURCES, false));
    add_option(Option(MDL_CTX_OPTION_EXPERIMENTAL, false));
}

mi::Size Execution_context::get_messages_count() const
{
    return m_messages.size();
}

mi::Size Execution_context::get_error_messages_count() const
{
    return m_error_messages.size();
}

const Message& Execution_context::get_message(mi::Size index) const
{
    ASSERT(M_SCENE, index < m_messages.size());
    
    return m_messages[index];
}

const Message& Execution_context::get_error_message(mi::Size index) const
{
    ASSERT(M_SCENE, index < m_error_messages.size());
    
    return m_error_messages[index];
}

void Execution_context::add_message(const mi::mdl::IMessage* message)
{
    m_messages.push_back(Message(message));
}

void Execution_context::add_error_message(const mi::mdl::IMessage* message)
{
    m_error_messages.push_back(Message(message));
}

void Execution_context::add_message(const Message& message)
{
    m_messages.push_back(message);
}

void Execution_context::add_error_message(const Message& message)
{
    m_error_messages.push_back(message);
}

void Execution_context::add_messages(const mi::mdl::Messages& messages) 
{
    for(int i = 0; i < messages.get_message_count(); ++i) {
        m_messages.push_back(messages.get_message(i));
    }
    for(int i = 0; i < messages.get_error_message_count(); ++i) {
        m_error_messages.push_back(messages.get_error_message(i));
    }
}

void Execution_context::clear_messages()
{
    m_messages.clear();
    m_error_messages.clear();
}

mi::Size Execution_context::get_option_count() const
{
    return static_cast<mi::Size>(m_options.size());
}

mi::Size Execution_context::get_option_index(const std::string& name) const
{
    const auto& option = m_options_2_index.find(name);
    if (option != m_options_2_index.end())
        return option->second;
    return static_cast<mi::Size>(-1);
}

const char* Execution_context::get_option_name(mi::Size index) const
{
    ASSERT(M_SCENE, m_options.size() > index);

    return m_options[index].get_name();
}

mi::Sint32 Execution_context::get_option(const std::string& name, STLEXT::Any& value) const
{
    mi::Size index = get_option_index(name);
    if (index == static_cast<mi::Size>(-1))
        return -1;
    
    const Option& option = m_options[index];
    value = option.get_value();
    return 0;
}

mi::Sint32 Execution_context::set_option(const std::string& name, const STLEXT::Any& value)
{
    mi::Size index = get_option_index(name);
    if (index == static_cast<mi::Size>(-1))
        return -1;

    Option& option = m_options[index];

    const STLEXT::Any& old_value = option.get_value();
    if(old_value.type() != value.type())
        return -2;

    if (option.set_value(value))
        return 0;
    else
        return -3;
}

void Execution_context::set_result(mi::Sint32 result)
{
    m_result = result;
}

mi::Sint32 Execution_context::get_result() const
{
    return m_result;
}

void Execution_context::add_option(const Option& option)
{
    m_options.push_back(option);
    m_options_2_index[option.get_name()] = m_options.size() - 1;
}

} // namespace MDL

} // namespace MI

