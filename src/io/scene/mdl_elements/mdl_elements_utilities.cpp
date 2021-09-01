/***************************************************************************************************
 * Copyright (c) 2012-2021, NVIDIA CORPORATION. All rights reserved.
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
#include "i_mdl_elements_module.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_type.h"

#include <regex>

#include <boost/core/ignore_unused.hpp>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/istring.h>
#include <mi/mdl/mdl.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_encapsulator.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_generated_dag.h>

#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/hal/disk/disk.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/compiler/compilercore/compilercore_visitor.h>
#include <mdl/codegenerators/generator_code/generator_code.h>
#include <mdl/codegenerators/generator_dag/generator_dag_tools.h>
#include <mdl/jit/generator_jit/generator_jit_libbsdf_data.h>
#include <mdl/integration/i18n/i_i18n.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>


// Disable false positives (claiming expressions involving "class_id" always being true or false)
//-V:class_id:547 PVS

namespace MI {

namespace MDL {

using mi::mdl::as;
using mi::mdl::is;
using mi::mdl::cast;

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
            CASE_OK( CAST);
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
        CASE_DS( COPY_CONSTRUCTOR);
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
        CASE_DS( INTRINSIC_MATH_COSH);
        CASE_DS( INTRINSIC_MATH_SINH);
        CASE_DS( INTRINSIC_MATH_TANH);
        CASE_DS( INTRINSIC_MATH_INT_BITS_TO_FLOAT);
        CASE_DS( INTRINSIC_MATH_FLOAT_BITS_TO_INT);
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
        CASE_DS( INTRINSIC_TEX_WIDTH_OFFSET);
        CASE_DS( INTRINSIC_TEX_HEIGHT_OFFSET);
        CASE_DS( INTRINSIC_TEX_DEPTH_OFFSET);
        CASE_DS( INTRINSIC_TEX_FIRST_FRAME);
        CASE_DS( INTRINSIC_TEX_LAST_FRAME);
        CASE_DS( INTRINSIC_TEX_GRID_TO_OBJECT_SPACE);

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
        CASE_DS( INTRINSIC_DF_MEASURED_FACTOR);
        CASE_DS( INTRINSIC_DF_CHIANG_HAIR_BSDF);
        CASE_DS( INTRINSIC_DF_SHEEN_BSDF);
        CASE_DS( INTRINSIC_DF_UNBOUNDED_MIX);
        CASE_DS( INTRINSIC_DF_COLOR_UNBOUNDED_MIX);
        CASE_DS( INTRINSIC_SCENE_DATA_ISVALID);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_COLOR);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR);

        CASE_DS( INTRINSIC_DEBUG_BREAKPOINT);
        CASE_DS( INTRINSIC_DEBUG_ASSERT);
        CASE_DS( INTRINSIC_DEBUG_PRINT);
        CASE_DS( INTRINSIC_DAG_FIELD_ACCESS);
        CASE_DS( INTRINSIC_DAG_ARRAY_CONSTRUCTOR);
        CASE_DS( INTRINSIC_DAG_ARRAY_LENGTH);

#undef CASE_DS

        // handled in first switch statement
        case mi::mdl::IDefinition::DS_OP_BASE:
        case mi::mdl::IDefinition::DS_OP_END:
            ASSERT( M_SCENE, false);
            return mi::neuraylib::IFunction_definition::DS_UNKNOWN;

        // should not appear in this context
        case mi::mdl::IDefinition::DS_INTRINSIC_ANNOTATION:
        case mi::mdl::IDefinition::DS_THROWS_ANNOTATION:
        case mi::mdl::IDefinition::DS_SINCE_ANNOTATION:
        case mi::mdl::IDefinition::DS_REMOVED_ANNOTATION:
        case mi::mdl::IDefinition::DS_CONST_EXPR_ANNOTATION:
        case mi::mdl::IDefinition::DS_DERIVABLE_ANNOTATION:
        case mi::mdl::IDefinition::DS_NATIVE_ANNOTATION:
        case mi::mdl::IDefinition::DS_EXPERIMENTAL_ANNOTATION:
        case mi::mdl::IDefinition::DS_LITERAL_PARAM_MASK_ANNOTATION:
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
        case mi::mdl::IDefinition::DS_ORIGIN_ANNOTATION:
        case mi::mdl::IDefinition::DS_BAKING_TMM_ANNOTATION:
        case mi::mdl::IDefinition::DS_BAKING_BAKE_TO_TEXTURE_ANNOTATION:
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

mi::neuraylib::IAnnotation_definition::Semantics mdl_semantics_to_ext_annotation_semantics(
    mi::mdl::IDefinition::Semantics semantic)
{
    if (!mi::mdl::semantic_is_annotation(semantic) &&
        semantic != mi::mdl::IDefinition::DS_UNKNOWN) {

        ASSERT(M_SCENE, false);
        return mi::neuraylib::IAnnotation_definition::AS_UNKNOWN;
    }

    switch (semantic) {

#define CASE_AS(e) \
        case mi::mdl::IDefinition::DS_##e: \
            return mi::neuraylib::IAnnotation_definition::AS_##e;

        CASE_AS(INTRINSIC_ANNOTATION);
        CASE_AS(THROWS_ANNOTATION);
        CASE_AS(SINCE_ANNOTATION);
        CASE_AS(REMOVED_ANNOTATION);
        CASE_AS(CONST_EXPR_ANNOTATION);
        CASE_AS(DERIVABLE_ANNOTATION);
        CASE_AS(NATIVE_ANNOTATION);
        CASE_AS(UNUSED_ANNOTATION);
        CASE_AS(NOINLINE_ANNOTATION);
        CASE_AS(SOFT_RANGE_ANNOTATION);
        CASE_AS(HARD_RANGE_ANNOTATION);
        CASE_AS(HIDDEN_ANNOTATION);
        CASE_AS(DEPRECATED_ANNOTATION);
        CASE_AS(VERSION_NUMBER_ANNOTATION);
        CASE_AS(VERSION_ANNOTATION);
        CASE_AS(DEPENDENCY_ANNOTATION);
        CASE_AS(UI_ORDER_ANNOTATION);
        CASE_AS(USAGE_ANNOTATION);
        CASE_AS(ENABLE_IF_ANNOTATION);
        CASE_AS(THUMBNAIL_ANNOTATION);
        CASE_AS(DISPLAY_NAME_ANNOTATION);
        CASE_AS(IN_GROUP_ANNOTATION);
        CASE_AS(DESCRIPTION_ANNOTATION);
        CASE_AS(AUTHOR_ANNOTATION);
        CASE_AS(CONTRIBUTOR_ANNOTATION);
        CASE_AS(COPYRIGHT_NOTICE_ANNOTATION);
        CASE_AS(CREATED_ANNOTATION);
        CASE_AS(MODIFIED_ANNOTATION);
        CASE_AS(KEYWORDS_ANNOTATION);
        CASE_AS(ORIGIN_ANNOTATION);

#undef CASE_AS
    default:
        break;
    }
    return mi::neuraylib::IAnnotation_definition::AS_UNKNOWN;
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
        CASE_DS( COPY_CONSTRUCTOR);
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
        CASE_OK( CAST);
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
        CASE_DS( INTRINSIC_MATH_COSH);
        CASE_DS( INTRINSIC_MATH_SINH);
        CASE_DS( INTRINSIC_MATH_TANH);
        CASE_DS( INTRINSIC_MATH_INT_BITS_TO_FLOAT);
        CASE_DS( INTRINSIC_MATH_FLOAT_BITS_TO_INT);
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
        CASE_DS( INTRINSIC_TEX_WIDTH_OFFSET);
        CASE_DS( INTRINSIC_TEX_HEIGHT_OFFSET);
        CASE_DS( INTRINSIC_TEX_DEPTH_OFFSET);
        CASE_DS( INTRINSIC_TEX_FIRST_FRAME);
        CASE_DS( INTRINSIC_TEX_LAST_FRAME);
        CASE_DS( INTRINSIC_TEX_GRID_TO_OBJECT_SPACE);
        CASE_DS( INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF);
        CASE_DS( INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF);
        CASE_DS( INTRINSIC_DF_SPECULAR_BSDF);
        CASE_DS( INTRINSIC_DF_SIMPLE_GLOSSY_BSDF);
        CASE_DS( INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF);
        CASE_DS( INTRINSIC_DF_MEASURED_BSDF);
        CASE_DS( INTRINSIC_DF_CHIANG_HAIR_BSDF);
        CASE_DS( INTRINSIC_DF_SHEEN_BSDF);
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
        CASE_DS( INTRINSIC_DF_MEASURED_FACTOR);
        CASE_DS( INTRINSIC_DF_UNBOUNDED_MIX);
        CASE_DS( INTRINSIC_DF_COLOR_UNBOUNDED_MIX);
        CASE_DS( INTRINSIC_SCENE_DATA_ISVALID);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_INT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_COLOR);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4);
        CASE_DS( INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR);

        CASE_DS( INTRINSIC_DEBUG_BREAKPOINT);
        CASE_DS( INTRINSIC_DEBUG_ASSERT);
        CASE_DS( INTRINSIC_DEBUG_PRINT);
        CASE_DS( INTRINSIC_DAG_FIELD_ACCESS);
        CASE_DS( INTRINSIC_DAG_ARRAY_CONSTRUCTOR);
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
            return; //-V1037 PVS
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
            return; //-V1037 PVS
        case IValue::VK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return;
    }

    ASSERT( M_SCENE, false);
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
            return; //-V1037 PVS
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag tag = expr_direct_call->get_definition(nullptr);
            result->insert( tag);
            DB::Tag module_tag = expr_direct_call->get_module();
            result->insert(module_tag);
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());
            collect_references( arguments.get(), result);
            return;
        }
        case IExpression::EK_TEMPORARY:
            return; //-V1037 PVS
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return;
    }

    ASSERT( M_SCENE, false);
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

const char* get_index_operator_db_name()
{
    return get_encoded_names_enabled() ? "mdl::operator[](%3C0%3E[],int)" : "mdl::operator[](<0>[],int)";
}

const char* get_index_operator_mdl_name()
{
    return get_encoded_names_enabled() ? "operator[](%3C0%3E[],int)" : "operator[](<0>[],int)";
}

const char* get_array_length_operator_db_name()
{
    return get_encoded_names_enabled() ? "mdl::operator_len(%3C0%3E[])" : "mdl::operator_len(<0>[])";
}

const char* get_array_length_operator_mdl_name()
{
    return get_encoded_names_enabled() ? "operator_len(%3C0%3E[])" : "operator_len(<0>[])";
}

const char* get_ternary_operator_db_name()
{
    return get_encoded_names_enabled() ? "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)" : "mdl::operator?(bool,<0>,<0>)";
}

const char* get_ternary_operator_mdl_name()
{
    return get_encoded_names_enabled() ? "operator%3F(bool,%3C0%3E,%3C0%3E)" : "operator?(bool,<0>,<0>)";
}

const char* get_cast_operator_db_name()
{
    return get_encoded_names_enabled() ? "mdl::operator_cast(%3C0%3E)" : "mdl::operator_cast(<0>)";
}

const char* get_cast_operator_mdl_name()
{
    return get_encoded_names_enabled() ? "operator_cast(%3C0%3E)" : "operator_cast(<0>)";
}

const char* get_builtins_module_db_name()
{
    return get_encoded_names_enabled() ? "mdl::%3Cbuiltins%3E" : "mdl::<builtins>";
}

const char* get_builtins_module_mdl_name()
{
    return get_encoded_names_enabled() ? "::%3Cbuiltins%3E" : "::<builtins>";
}

const char* get_builtins_module_simple_name()
{
    return get_encoded_names_enabled() ? "%3Cbuiltins%3E" : "<builtins>";
}

const char* get_neuray_module_db_name()
{
    return get_encoded_names_enabled() ? "mdl::%3Cneuray%3E" : "mdl::<neuray>";
}

const char* get_neuray_module_mdl_name()
{
    return get_encoded_names_enabled() ? "::%3Cneuray%3E" : "::<neuray>";
}

namespace { bool g_encoded_names_enabled = true; }

bool get_encoded_names_enabled() { return g_encoded_names_enabled; }

void set_encoded_names_enabled( bool value) { g_encoded_names_enabled = value; }

namespace {

unsigned char int_to_hex( int x)
{
    assert( x >= 0 && x < 16);
    return x < 10 ? '0' + x : 'A' + (x-10);
}

int hex_to_int( unsigned char c)
{
    if( (c >= '0') && (c <= '9'))
        return static_cast<int>( c - '0');

    if ((c >= 'A') && (c <= 'F'))
        return static_cast<int>( c - 'A') + 10;

    return -1;
}

size_t strfind( const char* s, size_t len_s, const char* what, size_t len_what)
{
    if( len_what > len_s)
        return std::string::npos;

    for( size_t i = 0; i < len_s - len_what; ++i) {

        if (s[i] != what[0])
            continue;

        size_t j = 1;
        for( ; j < len_what; ++j)
            if (s[i + j] != what[j])
                break;
        if( j == len_what)
            return i;
    }

    return std::string::npos;
}

size_t chrfind( const char* s, size_t n, char what)
{
    for( size_t i = 0; i < n; ++i)
        if( s[i] == what)
            return i;

    return std::string::npos;
}

void encode( const char* s, size_t n, std::string& result)
{
    for( size_t i = 0; i < n; ++i) {
        unsigned char c = s[i];

        switch( s[i]) {
            case '(':
            case ')':
            case '<':
            case '>':
            case ',':
            case ':':
            case '$':
            case '%':
            case '#':
            case '?':
            case '@':
                result += '%';
                result += int_to_hex( c >> 4);
                result += int_to_hex( c % 16);
                break;
            default:
                result += c;
        }
    }
}

} // namespace

std::string encode( const char* s)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = strlen( s);
    result.reserve( 2*n);

    encode( s, n, result);
    return result;
}

std::string decode( const std::string& s, bool strict, Execution_context* context)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = s.size();
    result.reserve( n);

    for( size_t i = 0; i < n; ) {

        if( s[i] != '%') {

            if( strict) {
                switch( s[i]) {
                    case '(':
                    case ')':
                    case '<':
                    case '>':
                    case ',':
                    case ':':
                    case '$':
                    case '#':
                    case '?':
                    case '@':
                        add_error_message( context,
                            STRING::formatted_string( "Invalid unescaped character \"%c\".", s[i]),
                                -1);
                        return std::string();
                    default:
                        break;
                }
            }

            result += s[i];
            ++i;
            continue;
        }

        if( i+2 >= n) {

            if( strict) {
                add_error_message( context,
                    STRING::formatted_string( "Invalid escape sequence at end of string \"%s\".",
                        s.c_str()), -1);
                return std::string();
            }

            result += s[i];
            ++i;
            continue;
        }

        int x1 = hex_to_int( s[i+1]);
        int x2 = hex_to_int( s[i+2]);
        if( x1 == -1 || x2 == -1) {

            if( strict) {
                add_error_message( context,
                    STRING::formatted_string( "Invalid hex character \"%c\".",
                        x1 == -1 ? s[i+1] : s[i+2]), -1);
                return std::string();
            }

            result += s[i];
            ++i;
            continue;
        }

        char c = (x1 << 4) + x2;

        switch( c) {
            case '(':
            case ')':
            case '<':
            case '>':
            case ',':
            case ':':
            case '$':
            case '#':
            case '?':
            case '@':
            case '%':
                break;
            default:
                if( strict) {
                    add_error_message( context,
                        STRING::formatted_string( "Unexpected escape sequence %%\"%c%c\".",
                            s[i+1], s[i+2]), -1);
                    return std::string();
                }
                result += c;
                i += 3;
                continue;
        }

        result += c;
        i += 3;
    }

    return result;
}

std::string decode_for_error_msg( const std::string& s)
{
    return decode( s, /*strict*/ false);
}

// Note the difference between "j" and "k" in methods below. "j" counts from the start of the
// string, whereas "k" is the offset w.r.t. "i", i.e., j == i+k.

namespace {

void encode_module_name( const char* s, size_t n, std::string& result)
{
    size_t i = 0;
    size_t k = strfind( s, n, "::", 2);

    while( k != std::string::npos) {
        encode( s + i, k, result);
        result += "::";
        i += k + 2;
        k = strfind( s + i, n - i, "::", 2);
    }

    encode( s + i, n - i, result);
}

} // namespace

std::string encode_module_name( const std::string& s)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = s.size();
    result.reserve( 2*n);

    encode_module_name( s.c_str(), n, result);
    return result;
}

std::string decode_module_name( const std::string& s)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = s.size();
    result.reserve( n);

    size_t i = 0;
    size_t j = s.find( "::");

    while( j != std::string::npos) {

        std::string tmp = decode( s.substr( i, j-i));
        if( tmp.empty() && (j > i))
            return std::string();

        result += tmp;
        result += "::";
        i = j + 2;
        j = s.find( "::", i);
    }

    std::string tmp = decode( s.substr( i));
    if( tmp.empty())
        return std::string();

    result += tmp;
    return result;
}

namespace {

void encode_name_without_signature( const char* s, size_t n, std::string& result)
{
    size_t i = 0;
    size_t k = strfind( s, n, "::", 2);

    while( k != std::string::npos) {
        encode( s + i, k, result);
        result += "::";
        i += k + 2;
        k = strfind( s + i, n - i, "::", 2);
    }

    k = chrfind( s + i, n - i, '$');
    if( k != std::string::npos) {
        encode( s + i, k, result);
        result += '$';
        i += k + 1;
    }

    encode( s + i, n - i, result);
}

} // namespace

std::string encode_name_without_signature( const std::string& s)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = s.size();
    result.reserve( 2*n);

    encode_name_without_signature( s.c_str(), n, result);
    return result;
}

std::string decode_name_without_signature( const std::string& s)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = s.size();
    result.reserve( n);

    size_t i = 0;
    size_t j = s.find( "::");

    while( j != std::string::npos) {

        std::string tmp = decode( s.substr( i, j-i));
        if( tmp.empty() && (j > i))
            return std::string();

        result += tmp;
        result += "::";
        i = j + 2;
        j = s.find( "::", i);
    }

    j = s.find( '$', i);
    if( j != std::string::npos) {

        std::string tmp = decode( s.substr( i, j-i));
        if( tmp.empty())
            return std::string();

        result += tmp;
        result += '$';
        i = j + 1;
    }

    std::string tmp = decode( s.substr( i));
    if( tmp.empty())
        return tmp;

    result += tmp;
    return result;
}

namespace {

void encode_name_with_signature( const char* s, size_t n, std::string& result)
{
    size_t i = 0;
    size_t k = chrfind( s + i, n - i, '(');
    ASSERT( M_SCENE, (k != std::string::npos) || !"missing signature");
    encode_name_without_signature( s + i, k, result);
    result += '(';
    i += k + 1;

    k = chrfind( s + i, n - i, ',');
    while( k != std::string::npos) {
        encode_name_without_signature( s + i, k, result);
        result += ',';
        i += k + 1;
        k = chrfind( s + i, n - i, ',');
    }

    k = chrfind( s + i, n - i, ')');
    ASSERT( M_SCENE, (k != std::string::npos) || !"broken signature");
    encode_name_without_signature( s + i, k, result);
    result += ')';
}

} // namespace

std::string encode_name_with_signature( const std::string& s)
{
    if( !get_encoded_names_enabled())
        return s;

    std::string result;
    size_t n = s.size();
    result.reserve( 2*n);

    encode_name_with_signature( s.c_str(), n, result);
    return result;
}

std::string decode_name_with_signature( const std::string& s)
{
    if( !get_encoded_names_enabled())
        return s;

    return decode( s, false);
}

std::string encode_name_plus_signature(
    const std::string& s, const std::vector<std::string>& parameter_types)
{
    std::string result = encode_name_without_signature( s);

    result += '(';
    for( size_t i = 0, n = parameter_types.size(); i < n; ++i) {
        const std::string& s = parameter_types[i];
        encode_name_without_signature( s.c_str(), s.size(), result);
        if( i+1 < n)
            result += ',';
    }
    result += ')';

    return result;
}

namespace {

// Assumes that parentheses are always meta-characters and do not appear as part of a simple name.
// TODO encoded names: Remove this limitation.
std::string get_dag_name_without_signature(
    const mi::mdl::IGenerated_code_dag* code_dag, bool is_material, mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    if( is_material)
        return code_dag->get_material_name( index_uint32);

    std::string result = code_dag->get_function_name( index_uint32);
    size_t left_paren = result.find( '(');
    return result.substr( 0, left_paren);
}

} // namespace

std::string get_mdl_name(
    const mi::mdl::IGenerated_code_dag* code_dag, bool is_material, mi::Size index)
{
    Code_dag dag( code_dag, is_material);

    if( !get_encoded_names_enabled())
        return dag.get_name( index);

    mi::mdl::IDefinition::Semantics sema = dag.get_semantics( index);
    mi::mdl::IExpression::Operator op    = mi::mdl::semantic_to_operator( sema);

    if( sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
        return get_array_constructor_mdl_name();
    else if( op == mi::mdl::IExpression::OK_CAST)
        return get_cast_operator_mdl_name();
    else if( op == mi::mdl::IExpression::OK_TERNARY)
        return get_ternary_operator_mdl_name();
    else if( op == mi::mdl::IExpression::OK_ARRAY_INDEX)
        return get_index_operator_mdl_name();
    else if( sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH)
        return get_array_length_operator_mdl_name();

    std::string s = get_dag_name_without_signature( code_dag, is_material, index);

    std::string result = encode_name_without_signature( s);
    result += '(';

    for( mi::Size i = 0, n = dag.get_parameter_count( index); i < n; ++i) {
        std::string s = dag.get_parameter_type_name( index, i);
        result += encode_name_without_signature( s);
        if( i+1 < n)
            result += ',';
    }

    result += ')';
    return result;
}

std::string get_mdl_annotation_name( const mi::mdl::IGenerated_code_dag* code_dag, mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    if( !get_encoded_names_enabled())
        return code_dag->get_annotation_name( index_uint32);

    // get annotation name without signature
    std::string s = code_dag->get_annotation_name( index_uint32);
    size_t left_paren = s.find( '(');
    s = s.substr( 0, left_paren);

    std::string result = encode_name_without_signature( s);
    result += '(';

    for( mi::Size i = 0, n = code_dag->get_annotation_parameter_count( index_uint32); i < n; ++i) {
        std::string s = code_dag->get_annotation_parameter_type_name( index_uint32, i);
        result += encode_name_without_signature( s);
        if( i+1 < n)
            result += ',';
    }

    result += ')';
    return result;
}

std::string encode_name_add_missing_signature(
    DB::Transaction* transaction,
    const mi::mdl::IGenerated_code_dag* m_code_dag,
    const std::string& name)
{
    if( !get_encoded_names_enabled() || name.empty())
        return name;

    if( name.back() == ')')
        return encode_name_with_signature( name);

    // Search material in the code DAG of the current module (if present, used during module
    // loading).
    mi::Size i = 0;
    mi::Size n = m_code_dag ? m_code_dag->get_material_count() : 0;
    for( ; i < n; ++i)
        if( name == m_code_dag->get_material_name( i))
            break;

    if( i < n) {

        std::string result = encode_name_without_signature( name);
        result += '(';
        for( mi::Size j = 0, n = m_code_dag->get_material_parameter_count( i); j < n; ++j) {
            result += encode_name_without_signature( m_code_dag->get_material_parameter_type_name( i, j));
            if( j+1 < n)
                result += ',';
        }
        result += ')';
        return result;
    }

    // Use DB access if code DAG is not present or material is in a different module (during
    // creation of compiled materials).
    std::string mdl_name = encode_name_without_signature( name);
    std::string mdl_module_name = get_mdl_module_name( mdl_name);
    DB::Tag tag = transaction->name_to_tag( get_db_name( mdl_module_name).c_str());
    DB::Access<Mdl_module> module( tag, transaction);
    ASSERT( M_SCENE, module.is_valid() || !"no module nor code DAG available to reconstruct signature");

    std::string prefix = mdl_name;
    prefix += '(';
    mi::Size l = prefix.size();

    for( mi::Size i = 0, n = module->get_material_count(); i < n; ++i) {

        DB::Access<Mdl_function_definition> fd( module->get_material( i), transaction);
        std::string s = fd->get_mdl_name();
        if( s.substr( 0, l) == prefix)
            return s;
    }

    ASSERT( M_SCENE, !"failed to find material in its module");
    return std::string();
}

namespace {

// Implementation of mi::neuraylib::ISerialized_function_name
class Serialized_function_name
  : public mi::base::Interface_implement<mi::neuraylib::ISerialized_function_name>
{
public:
    Serialized_function_name(
        const char* function_name,
        const char* module_name,
        const char* function_name_without_module_name)
      : m_function_name( function_name),
        m_module_name( module_name),
        m_function_name_without_module_name( function_name_without_module_name) { }

    const char* get_function_name() const { return m_function_name.c_str(); }

    const char* get_module_name() const { return m_module_name.c_str(); }

    const char* get_function_name_without_module_name() const
    { return m_function_name_without_module_name.c_str(); }

private:
    std::string m_function_name;
    std::string m_module_name;
    std::string m_function_name_without_module_name;
};

/// Invokes the MDLE callback for serialization on a module DB name.
///
/// Performs the necessary translation between module DB name (this method) and filename (used by
/// the callback).
///
/// \param module_name    The DB name of the MDLE module.
/// \param mdle_callback  A callback to map the filename of MDLE modules.
/// \return               The serialized module name (not necessarily a syntactically valid
///                       DB module name), or the empty string in case of errors.
std::string serialize_mdle_module_name(
    const std::string& module_name,
    mi::neuraylib::IMdle_serialization_callback* mdle_callback,
    Execution_context* context)
{
    ASSERT( M_SCENE, mdle_callback);
    ASSERT( M_SCENE, context);
    ASSERT( M_SCENE, get_encoded_names_enabled());

    // Compute filename from module name.
    if( module_name.substr( 0, 6) != "mdle::") {
        add_error_message( context,
           STRING::formatted_string( "Invalid MDLE definition name \"%s\"",
           module_name.c_str()), -1);
        return std::string();
    }
    std::string filename = decode_module_name( module_name.substr( 6));
    filename = remove_slash_in_front_of_drive_letter( filename);
    filename = HAL::Ospath::convert_to_platform_specific_path( filename);

    // Invoke callback.
    mi::base::Handle<const mi::IString> tmp(
        mdle_callback->get_serialized_filename( filename.c_str()));
    if( !tmp) {
        add_error_message( context,
            "Invalid result (NULL pointer) from MDLE callback.", -5);
        return std::string();
    }

    // Check that the result is still recognized as MDLE.
    std::string result = tmp->get_c_str();
    if( !is_mdle( result)) {
        add_error_message( context,
            STRING::formatted_string(
                "Invalid result \"%s\" from MDLE callback.", result.c_str()), -5);
        return std::string();
    }

    // Convert filename from callback result to "module" name, update definition name.
    //
    // Do \em not use get_db_name() here since "result" is not guaranteed to be a
    // syntactical valid name, e.g., it is not required to start with a slash.
    result = "mdle::" + MDL::encode_module_name( result);
    return result;
}

/// Invokes the MDLE callback for serialization on a (core) type name.
///
/// Performs the necessary translation between (core) type name (this method) and filename (used by
/// the callback).
///
/// \param type_name      The (core) name of the MDLE type.
/// \param mdle_callback  A callback to map the filename of MDLE modules.
/// \return               The serialized type name (not necessarily a syntactically valid
///                       (core) type name, or the empty string in case of errors.
std::string serialize_mdle_type_name(
    const std::string& type_name,
    mi::neuraylib::IMdle_serialization_callback* mdle_callback,
    Execution_context* context)
{
    ASSERT( M_SCENE, mdle_callback);
    ASSERT( M_SCENE, context);
    ASSERT( M_SCENE, get_encoded_names_enabled());

    // Compute module name from type name.
    size_t scope = type_name.rfind( "::");
    ASSERT( M_SCENE, scope != std::string::npos);
    ASSERT( M_SCENE, type_name.substr( 0, 2) == "::");
    std::string module_name = "mdle" + type_name.substr( 0, scope);

    std::string result = serialize_mdle_module_name( module_name, mdle_callback, context);
    if( context->get_error_messages_count() > 0)
        return std::string();

    // Compute type name from module name.
    ASSERT( M_SCENE, result.substr( 0, 6) == "mdle::");
    result = result.substr( 4) + "::" + type_name.substr( scope+2);
    return result;
}

/// Joins the vector elements by commas and add parentheses around the result.
std::string get_signature( const std::vector<std::string>& type_names)
{
    if( type_names.empty())
       return std::string( "()");

    std::string result = "(";
    for( const auto& type_name: type_names)
        result += type_name + ",";
    result.back() = ')';

    return result;
}

} // namespace

const mi::neuraylib::ISerialized_function_name* serialize_function_name(
    const char* definition_name,
    const IType_list* argument_types,
    const IType* return_type,
    mi::neuraylib::IMdle_serialization_callback* mdle_callback,
    Execution_context* context)
{
    ASSERT( M_SCENE, definition_name);
    ASSERT( M_SCENE, context);
    ASSERT( M_SCENE, get_encoded_names_enabled());

    std::string s = definition_name;

    // Catch some misuse. Also important for deriving the module name.
    if( s.substr( 0, 5) != "mdl::" && s.substr( 0, 6) != "mdle::") {
        add_error_message( context,
            STRING::formatted_string( "Invalid definition name \"%s\"", s.c_str()), -1);
        return nullptr;
    }

    // Decompose definition name.
    size_t left_paren = s.find( '(');
    if( left_paren == std::string::npos) {
        add_error_message( context,
            STRING::formatted_string( "Invalid definition name \"%s\"", s.c_str()), -1);
        return nullptr;
    }
    size_t scope = s.rfind( "::", left_paren);
    ASSERT( M_SCENE, scope == 3 || scope >= 6);
    std::string module_name
        = scope == 3 ? get_builtins_module_db_name() : s.substr( 0, scope);
    std::string function_name_without_module_name = s.substr( scope+2);
    std::string simple_name = s.substr( scope+2, left_paren-scope-2);

    bool is_array_constructor     = s == get_array_constructor_db_name();
    bool is_array_index_operator  = s == get_index_operator_db_name();
    bool is_array_length_operator = s == get_array_length_operator_db_name();
    bool is_ternary_operator      = s == get_ternary_operator_db_name();
    bool is_cast_operator         = s == get_cast_operator_db_name();

    if(    !is_array_constructor
        && !is_array_index_operator
        && !is_array_length_operator
        && !is_ternary_operator
        && !is_cast_operator) {

        // Invoke MDLE callback if present.
        if( mdle_callback && is_mdle( module_name.c_str())) {

            // Split signature.
            std::string type_names_str( s.substr( left_paren+1, s.size()-1-left_paren-1));
            std::vector<std::string> type_names;
            boost::algorithm::split(
                type_names, type_names_str, boost::algorithm::is_any_of( ","));
            if( type_names_str.empty())
                type_names.clear();

            // Translate module name.
            module_name = serialize_mdle_module_name( module_name.c_str(), mdle_callback, context);
            if( context->get_error_messages_count() > 0)
                return nullptr;

            // Translate type names.
            for( auto& type_name: type_names) {
                if( is_mdle( type_name))
                    type_name = serialize_mdle_type_name( type_name, mdle_callback, context);
                if( context->get_error_messages_count() > 0)
                    return nullptr;
            }

            // Update definition names.
            function_name_without_module_name = simple_name + get_signature( type_names);
            s = module_name + "::" + function_name_without_module_name;
        }

        return new Serialized_function_name(
            s.c_str(), module_name.c_str(), function_name_without_module_name.c_str());
    }

    if( !argument_types) {
        add_error_message( context,
            "Argument types are required for template-like functions", -2);
        return nullptr;
    }

    if( is_cast_operator && !return_type) {
        add_error_message( context,
            "Return type is required for the cast operator", -3);
        return nullptr;
    }

    s += '<';

    // Avoids type ambiguity of literial "0".
    const mi::Size zero = 0;

    if( is_array_constructor) {

        mi::base::Handle<const IType> arg0( argument_types->get_type( zero));
        if( !arg0) {
            add_error_message( context, "Invalid argument types.", -4);
            return nullptr;
        }

        s += Type_factory::get_serialization_type_name( arg0.get());
        s += ',';
        std::ostringstream ss;
        ss << argument_types->get_size();
        s += ss.str();

    } else if( is_array_index_operator) {

        mi::base::Handle<const IType> arg0( argument_types->get_type( zero));
        mi::base::Handle<const IType> stripped_arg0( arg0->skip_all_type_aliases());
        IType::Kind kind = stripped_arg0->get_kind();
        if(    kind != IType::TK_ARRAY
            && kind != IType::TK_VECTOR
            && kind != IType::TK_MATRIX) {
            add_error_message( context, "Invalid argument types.", -4);
            return nullptr;
        }

        s += Type_factory::get_serialization_type_name( arg0.get());

    } else if( is_array_length_operator) {

        mi::base::Handle<const IType> arg0( argument_types->get_type( zero));
        mi::base::Handle<const IType> stripped_arg0( arg0->skip_all_type_aliases());
        IType::Kind kind = stripped_arg0->get_kind();
        if( kind != IType::TK_ARRAY) {
            add_error_message( context, "Invalid argument types.", -4);
            return nullptr;
        }

        s += Type_factory::get_serialization_type_name( arg0.get());

    } else if( is_ternary_operator) {

        mi::base::Handle<const IType> arg1( argument_types->get_type( 1));
        if( !arg1) {
            add_error_message( context, "Invalid argument types.", -4);
            return nullptr;
        }

        s += Type_factory::get_serialization_type_name( arg1.get());

    } else if( is_cast_operator) {

        mi::base::Handle<const IType> arg0( argument_types->get_type( zero));
        if( !arg0) {
            add_error_message( context, "Invalid argument types.", -4);
            return nullptr;
        }

        s += Type_factory::get_serialization_type_name( arg0.get());
        s += ',';
        s += Type_factory::get_serialization_type_name( return_type);

    } else {
        ASSERT( M_SCENE, false);
    }

    s += '>';

    function_name_without_module_name = s.substr( scope+2);
    return new Serialized_function_name(
        s.c_str(), module_name.c_str(), function_name_without_module_name.c_str());
}

namespace {

// Implementation of IDeserialized_function_name
class Deserialized_function_name
  : public mi::base::Interface_implement<MDL::IDeserialized_function_name>
{
public:
    Deserialized_function_name( const char* db_name, const IType_list* argument_types)
      : m_db_name( db_name), m_argument_types( argument_types, mi::base::DUP_INTERFACE) { }

    const char* get_db_name() const { return m_db_name.c_str(); }

    const IType_list* get_argument_types() const
    { m_argument_types->retain(); return m_argument_types.get(); }

private:
    std::string m_db_name;
    mi::base::Handle<const IType_list> m_argument_types;
};

/// Invokes the MDLE callback for deserialization on a serialized module name.
///
/// Performs the necessary translation between module DB name (this method) and filename (used by
/// the callback).
///
/// \param module_name    The serialized name of the MDLE module.
/// \param mdle_callback  A callback to map the filename of MDLE modules.
/// \return               The DB module name, or the empty string in case of errors.
std::string deserialize_mdle_module_name(
    const std::string& module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context)
{
    ASSERT( M_SCENE, mdle_callback);
    ASSERT( M_SCENE, context);
    ASSERT( M_SCENE, get_encoded_names_enabled());

    // Compute filename from "module" name.
    if( module_name.substr( 0, 6) != "mdle::") {
        add_error_message( context,
            STRING::formatted_string(
                "Invalid serialized MDLE module name \"%s\" (wrong prefix).",
                module_name.c_str()), -6);
        return std::string();
    }
    ASSERT( M_SCENE, module_name.substr( 0, 6) == "mdle::");
    std::string filename = decode_module_name( module_name.substr( 6));
    filename = remove_slash_in_front_of_drive_letter( filename);
    filename = HAL::Ospath::convert_to_platform_specific_path( filename);

    // Invoke callback.
    mi::base::Handle<const mi::IString> tmp(
        mdle_callback->get_deserialized_filename( filename.c_str()));
    if( !tmp) {
        add_error_message( context,
            "Invalid result (NULL pointer) from MDLE callback.", -10);
        return std::string();
    }

    // Check that the result is still recognized as MDLE.
    std::string result = tmp->get_c_str();
    if( !is_mdle( result)) {
        add_error_message( context,
            STRING::formatted_string(
                "Invalid result \"%s\" from MDLE callback.", result.c_str()), -10);
        return std::string();
    }

    // Convert filename from callback result to module name, update definition names.
    result = get_mdl_name_from_load_module_arg( result, /*is_mdle*/ true);
    result = get_db_name( result);
    return result;
}

/// Invokes the MDLE callback for deserialization on a serialized type name.
///
/// Performs the necessary translation between (core) type name (this method) and filename (used by
/// the callback).
///
/// \param type_name      The serialized name of the MDLE type.
/// \param mdle_callback  A callback to map the filename of MDLE modules.
/// \return               The (core) type name, or the empty string in case of errors.
std::string deserialize_mdle_type_name(
    const std::string& type_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context)
{
    ASSERT( M_SCENE, mdle_callback);
    ASSERT( M_SCENE, context);
    ASSERT( M_SCENE, get_encoded_names_enabled());

    // Compute module name from type name.
    size_t scope = type_name.rfind( "::");
    ASSERT( M_SCENE, scope != std::string::npos);
    ASSERT( M_SCENE, type_name.substr( 0, 2) == "::");
    std::string module_name = "mdle" + type_name.substr( 0, scope);

    std::string result = deserialize_mdle_module_name( module_name, mdle_callback, context);
    if( context->get_error_messages_count() > 0)
        return std::string();

    // Compute type name from module name.
    ASSERT( M_SCENE, result.substr( 0, 6) == "mdle::");
    result = result.substr( 4) + "::" + type_name.substr( scope+2);
    return result;
}

} // namespace

const IDeserialized_function_name* deserialize_function_name(
    DB::Transaction* transaction,
    const char* function_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context)
{
    if( !function_name)
        return nullptr;

    std::string s = function_name;
    if( s.empty())
        return nullptr;

    mi::base::Handle<IType_factory> tf( get_type_factory());

    if( s.back() == '>') {

        size_t left_bracket = s.rfind( '<');
        if( left_bracket == std::string::npos) {
            add_error_message( context, "Invalid serialized function name (missing '<').", -1);
            return nullptr;
        }

        // All template-like functions are from the builtins module. Thus, we need to ensure that
        // this module is loaded, even if we do not need it here directly.
        load_neuray_module( transaction);

        std::string db_name = s.substr( 0, left_bracket);
        std::string template_type_names_str( s.substr( left_bracket+1, s.size()-1-left_bracket-1));
        std::vector<std::string> template_type_names;
        boost::algorithm::split(
            template_type_names, template_type_names_str, boost::algorithm::is_any_of( ","));
        if( template_type_names_str.empty())
            template_type_names.clear();

        if( db_name == get_array_constructor_db_name()) {

            if( template_type_names.size() != 2) {
                add_error_message( context,
                    "The array constructor requires two template parameters.", -2);
                return nullptr;
            }

            const char* element_type = template_type_names[0].c_str();
            mi::base::Handle<const IType> arg0( tf->create_type( element_type));
            if( !arg0) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid type "
                        "name.", element_type), -3);
                return nullptr;
            }

            const char* size_str = template_type_names[1].c_str();
            STLEXT::Likely<mi::Size> size_likely = STRING::lexicographic_cast_s<mi::Size>( size_str);
            if( !size_likely.get_status()) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid array "
                        "size.", size_str), -4);
                return nullptr;
            }

            mi::Size size = *size_likely.get_ptr();
            mi::base::Handle<IType_list> argument_types( tf->create_type_list());
            for( mi::Size i = 0; i < size; ++i)
                argument_types->add_type( std::to_string( i).c_str(), arg0.get());

            return new Deserialized_function_name( db_name.c_str(), argument_types.get());

        } else if( db_name == get_index_operator_db_name()) {

            if( template_type_names.size() != 1) {
                add_error_message( context,
                    "The array index operator requires one template parameter.", -2);
                return nullptr;
            }

            const char* indexable_type = template_type_names[0].c_str();
            mi::base::Handle<const IType> arg0( tf->create_type( indexable_type));
            if( !arg0) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid type "
                        "name.", indexable_type), -3);
                return nullptr;
            }

            IType::Kind kind = arg0->get_kind();
            if(    kind != IType::TK_ARRAY
                && kind != IType::TK_VECTOR
                && kind != IType::TK_MATRIX) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid array, "
                        "vector, or matrix type name.", indexable_type), -5);
                return nullptr;
            }

            mi::base::Handle<IType_list> argument_types( tf->create_type_list());
            argument_types->add_type( "a", arg0.get());
            mi::base::Handle<const IType> arg1( tf->create_int());
            argument_types->add_type( "i", arg1.get());

            return new Deserialized_function_name( db_name.c_str(), argument_types.get());

        } else if( db_name == get_array_length_operator_db_name()) {

            if( template_type_names.size() != 1) {
                add_error_message( context,
                    "The array length operator requires one template parameter.", -2);
                return nullptr;
            }

            const char* array_type = template_type_names[0].c_str();
            mi::base::Handle<const IType> arg0( tf->create_type( array_type));
            if( !arg0) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid type "
                        "name.", array_type), -3);
                return nullptr;
            }

            if( arg0->get_kind() != IType::TK_ARRAY) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid array "
                        "type name.", array_type), -5);
                return nullptr;
            }

            mi::base::Handle<IType_list> argument_types( tf->create_type_list());
            argument_types->add_type( "a", arg0.get());

            return new Deserialized_function_name( db_name.c_str(), argument_types.get());

        } else if( db_name == get_ternary_operator_db_name()) {

            if( template_type_names.size() != 1) {
                add_error_message( context,
                    "The ternary operator requires one template parameter.", -2);
                return nullptr;
            }

            const char* true_type = template_type_names[0].c_str();
            mi::base::Handle<const IType> arg1( tf->create_type( true_type));
            if( !arg1) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid type "
                        "name.", true_type), -3);
                return nullptr;
            }

            mi::base::Handle<IType_list> argument_types( tf->create_type_list());
            mi::base::Handle<const IType> arg0( tf->create_bool());
            argument_types->add_type( "cond",      arg0.get());
            argument_types->add_type( "true_exp",  arg1.get());
            argument_types->add_type( "false_exp", arg1.get()); // same type as true_exp

            return new Deserialized_function_name( db_name.c_str(), argument_types.get());

        } else if( db_name == get_cast_operator_db_name()) {

            if( template_type_names.size() != 2) {
                add_error_message( context,
                    "The cast operator requires two template parameters.", -2);
                return nullptr;
            }

            const char* cast_type = template_type_names[0].c_str();
            mi::base::Handle<const IType> arg0( tf->create_type( cast_type));
            if( !arg0) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid type "
                        "name.", cast_type), -3);
                return nullptr;
            }

            const char* cast_return_type = template_type_names[1].c_str();
            mi::base::Handle<const IType> arg1( tf->create_type( cast_return_type));
            if( !arg1) {
                add_error_message( context,
                    STRING::formatted_string( "Template parameter \"%s\" is not a valid type "
                        "name.", cast_return_type), -3);
                return nullptr;
            }

            mi::base::Handle<IType_list> argument_types( tf->create_type_list());
            argument_types->add_type( "cast",        arg0.get());
            argument_types->add_type( "cast_return", arg1.get());

            return new Deserialized_function_name( db_name.c_str(), argument_types.get());

        } else {

            add_error_message( context,
                STRING::formatted_string(
                    "Invalid serialized function name (containing '<' and '>')."), -5);
            return nullptr;

        }
    }

    // We might need the builtins module (for field selection operators) or one of the standard
    // modules (why actually?). The easiest way to ensure that they are present in the DB is to
    // load the neuray module, which loads the other ones implicitly.
    load_neuray_module( transaction);

    if( s.back() != ')') {
        add_error_message( context, "Invalid serialized function name (missing ')').", -6);
        return nullptr;
    }

    size_t left_paren = s.rfind( '(');
    if( left_paren == std::string::npos) {
        add_error_message( context, "Invalid serialized function name (missing '(').", -6);
        return nullptr;
    }

    // Split off signature and type names in the signature.
    std::string db_name = s;
    std::string db_name_without_signature = s.substr( 0, left_paren);
    std::string type_names_str( s.substr( left_paren+1, s.size()-1-left_paren-1));
    std::vector<std::string> type_names;
    boost::algorithm::split(
        type_names, type_names_str, boost::algorithm::is_any_of( ","));
    if( type_names_str.empty())
        type_names.clear();

    // Find scope operator before the simple name.
    size_t scope = db_name_without_signature.rfind( "::", left_paren);
    ASSERT( M_SCENE, scope == 3 || scope >= 6);
    if( scope == std::string::npos) {
        add_error_message( context, "Invalid serialized function name (missing '::').", -6);
        return nullptr;
    }

    // Compute module name and load it if not yet loaded.
    std::string module_db_name
        = scope == 3 ? get_builtins_module_db_name() : db_name.substr( 0, scope);

    // Invoke MDLE callback if present.
    if( mdle_callback && is_mdle( module_db_name)) {

        // Translate module name.
        std::string result = deserialize_mdle_module_name( module_db_name, mdle_callback, context);
        if( context->get_error_messages_count() > 0)
            return nullptr;

        // Translate type names.
        for( auto& type_name: type_names) {
            if( is_mdle( type_name))
                type_name = deserialize_mdle_type_name( type_name, mdle_callback, context);
            if( context->get_error_messages_count() > 0)
                return nullptr;
        }

        // Update definition names.
        size_t n = module_db_name.size();
        module_db_name = result;
        db_name_without_signature = module_db_name + db_name_without_signature.substr( n);
        scope = db_name_without_signature.rfind( "::");
        ASSERT( M_SCENE, scope == 3 || scope >= 6);
        db_name = s = db_name_without_signature + get_signature( type_names);
    }

    // Load module if not yet loaded.
    DB::Tag module_tag = transaction->name_to_tag( module_db_name.c_str());
    if( !module_tag) {
        std::string load_module_arg
            = decode_module_name( strip_mdl_or_mdle_prefix( module_db_name));
        if( is_mdle( load_module_arg)) {
            load_module_arg = load_module_arg.substr( 2);
            load_module_arg = remove_slash_in_front_of_drive_letter( load_module_arg);
        }
        mi::Sint32 result
            = Mdl_module::create_module( transaction, load_module_arg.c_str(), context);
        if( result < 0)
            return nullptr;
        module_tag = transaction->name_to_tag( module_db_name.c_str());
        ASSERT( M_SCENE, module_tag);
    }

    SERIAL::Class_id class_id = transaction->get_class_id( module_tag);
    if( class_id != Mdl_module::id) {
        add_error_message( context,
            STRING::formatted_string( "DB name for module \"%s\" already in use.",
                module_db_name.c_str()), -7);
        return nullptr;
    }

    // Compute simple name and check for field access functions.
    std::string simple_name = db_name_without_signature.substr( scope+2);
    bool field_access = simple_name.find( '.') != std::string::npos;

    DB::Tag fd_tag;
    if( !field_access) {

        // Run overload resolution.
        DB::Access<Mdl_module> module( module_tag, transaction);

        std::vector<const char*> type_names_cstr;
        for( const auto& type_name: type_names)
            type_names_cstr.push_back( type_name.c_str());
        std::vector<std::string> result( module->get_function_overloads_by_signature(
            db_name_without_signature.c_str(), type_names_cstr));
        if( result.empty()) {
            add_error_message( context,
                STRING::formatted_string( "No matching overload found for \"%s\".",
                    db_name.c_str()), -8);
            return nullptr;
        }
        if( result.size() > 1) {
            add_error_message( context,
                STRING::formatted_string( "No unambiguous overload found for \"%s\".",
                    db_name.c_str()), -8);
            return nullptr;
        }

        db_name = result[0];
        fd_tag = transaction->name_to_tag( db_name.c_str());
         ASSERT( M_SCENE, fd_tag);

    } else {

        // Skip overload resolution for field access methods (not supported yet).
        fd_tag = transaction->name_to_tag( db_name.c_str());
        if( !fd_tag) {
            add_error_message( context,
                STRING::formatted_string( "No matching overload found for \"%s\".",
                    db_name.c_str()), -8);
            return nullptr;
        }

    }

    class_id = transaction->get_class_id( fd_tag);
    if( class_id != Mdl_function_definition::id) {
        add_error_message( context,
            STRING::formatted_string( "DB name for function defnition \"%s\" already in use.",
                db_name.c_str()), -9);
        return nullptr;
    }

    DB::Access<Mdl_function_definition> fd( fd_tag, transaction);
    mi::base::Handle<const IType_list> parameter_types( fd->get_parameter_types());
    return new Deserialized_function_name( db_name.c_str(), parameter_types.get());
}

const IDeserialized_function_name* deserialize_function_name(
    DB::Transaction* transaction,
    const char* module_name,
    const char* function_name_without_module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context)
{
    if( !module_name || !function_name_without_module_name)
        return nullptr;

    std::string function_name;
    if( strcmp( module_name, get_builtins_module_db_name()) == 0)
        function_name = get_db_name( function_name_without_module_name);
    else
        function_name = std::string( module_name) + "::" + function_name_without_module_name;

    return deserialize_function_name( transaction, function_name.c_str(), mdle_callback, context);
}

namespace {

class Deserialized_module_name
  : public mi::base::Interface_implement<mi::neuraylib::IDeserialized_module_name>
{
public:
    Deserialized_module_name( const char* db_name, const char* load_module_argument)
      : m_db_name( db_name),
        m_load_module_argument( load_module_argument) { }

    const char* get_db_name() const { return m_db_name.c_str(); }

    const char* get_load_module_argument() const { return m_load_module_argument.c_str(); }

private:
    std::string m_db_name;
    std::string m_load_module_argument;
};

} // namespace

const mi::neuraylib::IDeserialized_module_name* deserialize_module_name(
    const char* module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context)
{
    if( !module_name)
        return nullptr;

    std::string db_name = module_name;

    // Invoke MDLE callback if present.
    if( mdle_callback && is_mdle( db_name)) {

        // Translate module name.
        db_name = deserialize_mdle_module_name( db_name, mdle_callback, context);
        if( context->get_error_messages_count() > 0)
            return nullptr;
    }

    std::string load_module_argument = strip_mdl_or_mdle_prefix( db_name);

    if( is_mdle( load_module_argument)) {

        load_module_argument = decode_module_name( load_module_argument);
        if( load_module_argument.substr( 0, 2) != "::") {
            add_error_message( context,
                STRING::formatted_string(
                    "Invalid serialized module name \"%s\".", db_name.c_str()), -11);
            return nullptr;
        }

        load_module_argument = load_module_argument.substr( 2);
        load_module_argument = remove_slash_in_front_of_drive_letter( load_module_argument);
        load_module_argument
            = HAL::Ospath::convert_to_platform_specific_path( load_module_argument);
    }

    return new Deserialized_module_name( db_name.c_str(), load_module_argument.c_str());
}

bool is_builtin_module( const std::string& module)
{
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    return mdl->is_builtin_module( module.c_str());
}


bool is_supported_prototype( mi::neuraylib::IFunction_definition::Semantics sema, bool for_variant)
{
    if(    sema == mi::neuraylib::IFunction_definition::DS_CAST
        || sema == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
        || sema == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH
        || (   sema >= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_FIRST
            && sema <= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_LAST))
        return false;


    if( !for_variant)
        return true;

    if( sema == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_FIELD_ACCESS
        || (   sema >= mi::neuraylib::IFunction_definition::DS_OPERATOR_FIRST
            && sema <= mi::neuraylib::IFunction_definition::DS_OPERATOR_LAST))
        return false;

    return true;
}

void load_neuray_module( DB::Transaction* transaction)
{
    DB::Tag tag = transaction->name_to_tag( get_neuray_module_db_name());
    if( tag)
        return;

    // The "<neuray>" module is created here on the fly. Note that this needs a transaction and
    // other places like the factories of Mdl_module are not guaranteed to run, e.g., they are
    // never executed on the remote side of the Iray Cloud.
    if( !tag) {
        mi::base::Handle<mi::neuraylib::IReader> reader( create_reader(
            "mdl 1.0; export material default_material() = material();"));
        MDL::Execution_context context;
        mi::Sint32 result = Mdl_module::create_module(
            transaction, get_neuray_module_mdl_name(), reader.get(), &context);
        ASSERT( M_SCENE, result == 0);
        boost::ignore_unused( result);
    }
}

Mdl_compiled_material* get_default_compiled_material( DB::Transaction* transaction)
{
    load_neuray_module( transaction);

    std::string name = std::string( get_neuray_module_db_name()) + "::default_material";
    if( get_encoded_names_enabled())
        name += "()";

    DB::Tag tag = transaction->name_to_tag( name.c_str());
    DB::Access<Mdl_function_definition> md( tag, transaction);
    Mdl_function_call* mi = md->create_function_call( transaction, /*arguments*/ nullptr);
    Execution_context context;
    Mdl_compiled_material* cm = mi->create_compiled_material(
        transaction, /*class_compilation*/ false, &context);
    delete mi;
    return cm;
}


// **********  Traversal of types, values, and expressions *****************************************

const mi::mdl::IType* get_field_type( const mi::mdl::IType_struct* type, const char* field_name)
{
    if( !type)
        return nullptr;

    mi::Uint32 count = type->get_field_count();
    for( mi::Uint32 i = 0; i < count; ++i) {
        const mi::mdl::ISymbol* field_symbol;
        const mi::mdl::IType*   field_type;
        type->get_field( i, field_type, field_symbol);
        if( strcmp( field_name, field_symbol->get_name()) == 0)
            return field_type;
    }

    return nullptr;
}

const IValue* lookup_sub_value( const IValue* value, const char* path)
{
    ASSERT( M_SCENE, value && path);

    // handle empty paths
    if( path[0] == '\0') {
        value->retain();
        return value;
    }

    // handle non-compounds
    mi::base::Handle<const IValue_compound> value_compound(
        value->get_interface<IValue_compound>());
    if( !value_compound)
        return nullptr;

    std::string head, tail;
    split_next_dot_or_bracket( path, head, tail);

    // handle structs via field name
    if( value_compound->get_kind() == IValue::VK_STRUCT) {
        mi::base::Handle<const IValue_struct> value_struct(
            value_compound->get_interface<IValue_struct>());
        mi::base::Handle<const IValue> tail_value( value_struct->get_field( head.c_str()));
        if( !tail_value)
            return nullptr;
        return lookup_sub_value( tail_value.get(), tail.c_str());
    }

    // handle other compounds via index
    STLEXT::Likely<mi::Size> index_likely = STRING::lexicographic_cast_s<mi::Size>( head);
    if( !index_likely.get_status())
        return nullptr;
    mi::Size index = *index_likely.get_ptr(); //-V522 PVS
    mi::base::Handle<const IValue> tail_value( value_compound->get_value( index));
    if( !tail_value)
        return nullptr;
    return lookup_sub_value( tail_value.get(), tail.c_str());
}

const IExpression* lookup_sub_expression(
    const IExpression_factory* ef,
    const IExpression_list* temporaries,
    const IExpression* expr,
    const char* path)
{
    ASSERT( M_SCENE, ef && expr && path);

    // resolve temporaries
    IExpression::Kind kind = expr->get_kind();
    if( temporaries && kind == IExpression::EK_TEMPORARY) {
        mi::base::Handle<const IExpression_temporary> expr_temporary(
            expr->get_interface<IExpression_temporary>());
        mi::Size index = expr_temporary->get_index();
        expr = temporaries->get_expression( index);
        if( !expr)
            return nullptr;
        expr->release(); // rely on refcount in temporaries
        // type is unchanged
        kind = expr->get_kind();
    }

    // handle empty paths
    if( path[0] == '\0') {
        expr->retain();
        return expr;
    }

    switch( kind) {

        case IExpression::EK_CONSTANT: {

            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            mi::base::Handle<const IValue> result(
                lookup_sub_value( value.get(), path));
            if( !result)
                return nullptr;
            return ef->create_constant( result.get());
        }

        case IExpression::EK_DIRECT_CALL: {

            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IExpression_list> arguments(
                expr_direct_call->get_arguments());

            std::string head, tail;
            split_next_dot_or_bracket( path, head, tail);
            mi::Size index = arguments->get_index( head.c_str());
            if( index == static_cast<mi::Size>( -1))
                return nullptr;
            mi::base::Handle<const IExpression> argument( arguments->get_expression( index));
            return lookup_sub_expression(
                ef, temporaries, argument.get(), tail.c_str());
        }

        case IExpression::EK_CALL:
        case IExpression::EK_PARAMETER:
        case IExpression::EK_TEMPORARY:
            ASSERT( M_SCENE, false);
            return nullptr;

        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}


// **********  Resource-related attributes *********************************************************

void get_texture_attributes(
    DB::Transaction* transaction,
    DB::Tag tag,
    mi::Sint32 uv_tile_x,
    mi::Sint32 uv_tile_y,
    bool& valid,
    int& width,
    int& height,
    int& depth)
{
    valid     = false;
    width     = 0;
    height    = 0;
    depth     = 0;

    if( !tag || transaction->get_class_id( tag) != TEXTURE::ID_TEXTURE)
        return;
    DB::Access<TEXTURE::Texture> db_texture( tag, transaction);
    tag = db_texture->get_image();
    if( !tag || transaction->get_class_id( tag) != DBIMAGE::ID_IMAGE)
        return;
    DB::Access<DBIMAGE::Image> db_image( tag, transaction);
    if( !db_image->is_valid())
        return;

    mi::Uint32 uvtile_id = 0;
    if( db_image->is_uvtile())
        uvtile_id = db_image->get_uvtile_id( uv_tile_x, uv_tile_y);
    if( uvtile_id == static_cast<mi::Uint32>( -1))
        return;

    mi::base::Handle<const IMAGE::IMipmap> mipmap( db_image->get_mipmap( transaction, uvtile_id));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    valid     = true;
    width     = canvas->get_resolution_x();
    height    = canvas->get_resolution_y();
    depth     = canvas->get_layers_size();
}

void get_light_profile_attributes(
    DB::Transaction* transaction,
    DB::Tag tag,
    bool& valid,
    float& power,
    float& maximum)
{
    valid   = false;
    power   = 0.0f;
    maximum = 0.0f;

    if( !tag || transaction->get_class_id( tag) != LIGHTPROFILE::ID_LIGHTPROFILE)
        return;

    DB::Access<LIGHTPROFILE::Lightprofile> db_lightprofile( tag, transaction);
    if( !db_lightprofile->is_valid())
        return;

    valid   = true;
    power   = db_lightprofile->get_power();
    maximum = db_lightprofile->get_maximum();
}

void get_bsdf_measurement_attributes(
    DB::Transaction* transaction,
    DB::Tag tag,
    bool& valid)
{
    valid = false;

    if( !tag || transaction->get_class_id( tag) != BSDFM::ID_BSDF_MEASUREMENT)
        return;

    DB::Access<BSDFM::Bsdf_measurement> db_bsdf_measurement( tag, transaction);
    if( !db_bsdf_measurement->is_valid())
        return;

    valid = true;
}

// **********  Mdl_dag_builder *********************************************************************

template <class T>
Mdl_dag_builder<T>::Mdl_dag_builder(
    DB::Transaction* transaction,
    T* dag_builder,
    const Mdl_compiled_material* compiled_material)
  : m_transaction( transaction),
    m_dag_builder( dag_builder),
    m_type_factory( dag_builder->get_type_factory()),
    m_value_factory( dag_builder->get_value_factory()),
    m_compiled_material( compiled_material),
    m_temporaries(),
    m_parameter_types()
{
    if( compiled_material)
        m_temporaries.resize( compiled_material->get_temporary_count());
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
            return nullptr;
    }

    ASSERT( M_SCENE, false); //-V779 PVS
    return nullptr;
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_constant_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type, const IExpression_constant* expr)
{
    mi::base::Handle<const IValue> value( expr->get_value());
    const mi::mdl::IValue* mdl_value = int_value_to_mdl_value(
        m_transaction, m_value_factory, mdl_type, value.get());
    if( !mdl_value)
        return nullptr;
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
    return nullptr;
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::add_cache_entry(
    DB::Tag tag, const mi::mdl::DAG_node* value)
{
    ASSERT( M_SCENE, m_converted_call_expressions.count( tag) == 0);
    m_converted_call_expressions[tag] = value;
    return value;
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_call_to_mdl_dag_node(
    const mi::mdl::IType* mdl_type,
    const IExpression_call* expr)
{
    DB::Tag tag = expr->get_call();
    Call_stack_guard guard( m_set_indirect_calls, tag);
    if( guard.last_frame_creates_cycle())
        return nullptr;

    auto it = m_converted_call_expressions.find( tag);
    if( it != m_converted_call_expressions.end())
        return it->second;

    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id != Mdl_function_call::id) {
        const char* name = m_transaction->tag_to_name( tag);
        ASSERT( M_SCENE, name);
        LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
            "Unsupported type for call of \"%s\".", name?name:"");
        return add_cache_entry( tag, nullptr);
    }

    DB::Access<Mdl_function_call> call( tag, m_transaction); //-V779 PVS
    DB::Tag module_tag = call->get_module( m_transaction);
    ASSERT( M_SCENE, module_tag.is_valid());
    if( !module_tag.is_valid())
        return add_cache_entry( tag, m_dag_builder->create_constant( m_value_factory->create_bad()));

    DB::Access<Mdl_module> module( module_tag, m_transaction);
    mi::base::Handle<const IExpression_list> arguments( call->get_arguments());
    const char* call_name = m_transaction->tag_to_name( tag);

    bool is_material = call->is_material();
    mi::Size index = module->get_definition_index(
        is_material, call->get_definition_db_name(), call->get_definition_ident());
    ASSERT( M_SCENE, index != mi::Size(-1));

    return add_cache_entry( tag, Mdl_dag_builder<T>::int_expr_call_to_mdl_dag_node_shared(
        mdl_type,
        module.get_ptr(),
        is_material,
        index,
        call_name,
        arguments.get()));
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
        ASSERT( M_SCENE, expr_type_compound != nullptr);
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
    DB::Tag tag = expr->get_definition( m_transaction);
    if( !tag)
        return m_dag_builder->create_constant( m_value_factory->create_bad());
    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id != Mdl_function_definition::id)
        return nullptr;

    DB::Access<Mdl_function_definition> definition( tag, m_transaction);
    DB::Tag module_tag = definition->get_module( m_transaction);
    if( !module_tag)
        return m_dag_builder->create_constant( m_value_factory->create_bad());

    DB::Access<Mdl_module> module( module_tag, m_transaction);
    bool is_material = definition->is_material();
    const char* definition_db_name = m_transaction->tag_to_name( tag);
    mi::Size index = module->get_definition_index(
        is_material, definition_db_name, definition->get_ident());
    ASSERT( M_SCENE, index != mi::Size(-1));

    const char* call_name = m_transaction->tag_to_name( tag);
    mi::base::Handle<const IExpression_list> arguments( expr->get_arguments());

    return Mdl_dag_builder<T>::int_expr_call_to_mdl_dag_node_shared(
        mdl_type, module.get_ptr(), is_material, index, call_name, arguments.get());
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
            return nullptr;
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
            return nullptr;
        if( index >= m_temporaries.size())
            m_temporaries.resize( index+1);
        m_temporaries[index] = result;
    }

    ASSERT( M_SCENE, index < m_temporaries.size() && m_temporaries[index]);
    // re-create the temporary
    return m_dag_builder->create_temporary( m_temporaries[index], int( index));
}

template <class T>
const mi::mdl::DAG_node* Mdl_dag_builder<T>::int_expr_call_to_mdl_dag_node_shared(
    const mi::mdl::IType* mdl_type,
    const Mdl_module* module,
    bool is_material,
    mi::Size definition_index,
    const char* call_name,
    const IExpression_list* arguments)
{
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), is_material);
    const char* definition_core_name = code_dag.get_name( definition_index);

    mi::mdl::IDefinition::Semantics sema = code_dag.get_semantics( definition_index);
    bool is_array_constructor = sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
    bool is_array_length      = sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH;

    mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator( sema);
    bool is_cast_operator        = op == mi::mdl::IExpression::OK_CAST;
    bool is_ternary_operator     = op == mi::mdl::IExpression::OK_TERNARY;
    bool is_array_index_operator = op == mi::mdl::IExpression::OK_ARRAY_INDEX;

    const mi::mdl::IType* element_type = nullptr;
    if( is_array_constructor) {
        const mi::mdl::IType_array* mdl_type_array = mi::mdl::as<mi::mdl::IType_array>( mdl_type);
        element_type = mdl_type_array->get_element_type();
    }

    DETAIL::Type_binder type_binder( m_type_factory);

    mi::Size n = arguments->get_size();
    Small_VLA<mi::mdl::DAG_call::Call_argument, 8> mdl_arguments( n);

    for( mi::Size i = 0; i < n; ++i) {

        mdl_arguments[i].param_name = arguments->get_name( i);

        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        const mi::mdl::IType* parameter_type = nullptr;
        if( is_array_constructor) {
            parameter_type = element_type;
        } else if( is_array_length || is_cast_operator || is_ternary_operator || is_array_index_operator) {
            mi::base::Handle<const IType> type( argument->get_type());
            parameter_type = int_type_to_mdl_type( type.get(), *m_type_factory);
        } else {
            parameter_type = code_dag.get_parameter_type( definition_index, i);
        }
        parameter_type = m_type_factory->import( parameter_type->skip_type_alias()); //-V522 PVS

        mdl_arguments[i].arg = int_expr_to_mdl_dag_node( parameter_type, argument.get());
        if( !mdl_arguments[i].arg)
            return nullptr;

        const mi::mdl::IType* argument_type = mdl_arguments[i].arg->get_type();
        mi::Sint32 result = type_binder.check_and_bind_type( parameter_type, argument_type);

        switch( result) {
            case 0:
                // nothing to do
                break;
            case -1: {
                const std::string& definition_name
                    = get_db_name( definition_core_name);
                const char* s1 = call_name ? "" : "of definition ";
                const char* s2
                    = call_name ? call_name : definition_name.c_str();
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Type mismatch for argument \"%s\" of function call %s\"%s\".",
                    mdl_arguments[i].param_name, s1, s2);
                return nullptr;
            }
            case -2: {
                const std::string& definition_name
                    = get_db_name( definition_core_name);
                const char* s1 = call_name ? "" : "of definition ";
                const char* s2
                    = call_name ? call_name : definition_name.c_str();
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Array size mismatch for argument \"%s\" of function call %s\"%s\".",
                    mdl_arguments[i].param_name, s1, s2);
                return nullptr;
            }
            default:
                ASSERT( M_SCENE, false);
                return nullptr;
            }
    }

    const mi::mdl::IType* return_type = nullptr;
    if( is_material) {
        return_type = m_type_factory->get_predefined_struct( mi::mdl::IType_struct::SID_MATERIAL);
    } else if( is_array_constructor) {
        element_type = m_type_factory->import( element_type);
        return_type = m_type_factory->create_array( element_type, mdl_arguments.size());
    } else if( is_cast_operator || is_ternary_operator || is_array_index_operator) {
        return_type = m_type_factory->import( mdl_type);
    } else {
        return_type = code_dag.get_return_type( definition_index);
        return_type = m_type_factory->import( return_type);
    }

    const mi::mdl::IType_array* return_type_array = as<mi::mdl::IType_array>( return_type);
    if( return_type_array && !return_type_array->is_immediate_sized()) {
        const mi::mdl::IType* bound_return_type = type_binder.get_bound_type( return_type_array);
        if( bound_return_type)
            return_type = bound_return_type;
    }

    return m_dag_builder->create_call(
        definition_core_name,
        sema,
        mdl_arguments.data(),
        static_cast<mi::Uint32>( mdl_arguments.size()),
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
    DB::Tag module_tag = transaction->name_to_tag( get_builtins_module_db_name());
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
        return nullptr;

    // access material elemental constructor
    DB::Tag fd_tag = builtins_module->get_definition( /*is_material*/ false, constructor_index);
    if( !fd_tag)
        return nullptr;
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
            mdl.get(),
            allocator.get(),
            int( constructor_index),
            internal_space,
            /*unsafe_math_opt=*/true));
    ASSERT( M_SCENE, mi);

    // create builder to convert fields and temporaries of compiled material
    Mdl_dag_builder<mi::mdl::IGenerated_code_dag::DAG_node_factory> dag_builder(
        transaction,
        &mi->m_node_factory,
        cm);

    // convert material body expression and set as constructor
    const mi::mdl::IType* mdl_return_type = fd->get_mdl_return_type( transaction);
    mi::base::Handle<const IExpression_direct_call> call( cm->get_body());
    const mi::mdl::DAG_node* call_node
        = dag_builder.int_expr_to_mdl_dag_node( mdl_return_type, call.get());
    ASSERT( M_SCENE, call_node);
    if( !call_node)
        return nullptr;
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

DB::Tag Mdl_call_resolver::get_module_tag( const char* name) const
{
    if( get_encoded_names_enabled()) {

        std::string s = name;

        // strip signature
        size_t i = s.find( '(');
        if( i != std::string::npos)
            s = s.substr( 0, i);

        // lookup name for DB element
        std::string mdl_name = encode_name_without_signature( s);
        std::string db_name  = get_db_name( mdl_name);
        DB::Tag tag = m_transaction->name_to_tag( db_name.c_str());
        if( tag && (m_transaction->get_class_id( tag) == Mdl_module::id))
            return tag;

        // strip last component and lookup name of that DB element
        mdl_name = get_mdl_module_name( mdl_name);
        db_name  = get_db_name( mdl_name);
        tag = m_transaction->name_to_tag( db_name.c_str());
        if( tag && (m_transaction->get_class_id( tag) == Mdl_module::id))
            return tag;

        return DB::Tag();
    }

    std::string db_name = get_db_name( name);
    DB::Tag tag = m_transaction->name_to_tag( db_name.c_str());
    if( !tag)
        return DB::Tag();

    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id == Mdl_module::id)
        return tag;

    if( class_id != Mdl_function_definition::id)
        return DB::Tag();

    DB::Access<Mdl_function_definition> function_definition( tag, m_transaction); //-V779 PVS
    DB::Tag module_tag = function_definition->get_module( m_transaction);
    if( m_transaction->get_class_id( module_tag) != Mdl_module::id)
        return DB::Tag();

    return module_tag;
}

const mi::mdl::IModule* Mdl_call_resolver::get_owner_module( const char* name) const
{
    DB::Tag module_tag = get_module_tag(name);
    if (!module_tag)
        return nullptr;

    DB::Access<Mdl_module> mdl_module(module_tag, m_transaction);
    const mi::mdl::IModule *module = mdl_module->get_mdl_module();

    // ensure that all import entries are restored before the module is returned
    if (m_resolved_modules.find(module) == m_resolved_modules.end()) {
        module->retain();
        m_resolved_modules.insert(module);
        SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
        Module_cache cache(m_transaction, mdlc_module->get_module_wait_queue(), {});
        module->restore_import_entries(&cache);
    }

    return module;
}

mi::mdl::IGenerated_code_dag const *Mdl_call_resolver::get_owner_dag(const char* name) const
{
    DB::Tag module_tag = get_module_tag(name);
    if (!module_tag)
        return nullptr;

    DB::Access<Mdl_module> mdl_module(module_tag, m_transaction);
    return mdl_module->get_code_dag();
}

// Constructor.
Mdl_call_resolver_ext::Mdl_call_resolver_ext(
    DB::Transaction* transaction,
    mi::mdl::IModule const *module)
    : Base(transaction)
    , m_module(module)
    , m_module_core_name(module->get_name())
{
}

const mi::mdl::IModule* Mdl_call_resolver_ext::get_owner_module(const char* name) const
{
    if( !is_in_module( name, m_module_core_name))
        return Base::get_owner_module( name);

    // This test is not strictly necessary, but matches the behavior of the base class.
    const mi::mdl::Module* module = mi::mdl::impl_cast<mi::mdl::Module>( m_module);
    auto result = module->find_signature( name + m_module_core_name.size() + 2, /*only_exported=*/false);
    if( result != nullptr) {
        m_module->retain();
        return m_module;
    }

    ASSERT( M_SCENE, false);
    return Base::get_owner_module( name);
}


// ********** Conversion from mi::mdl to MI::MDL ***************************************************

namespace {

const IType_enum* create_enum(
    IType_factory* tf,
    const char* symbol,
    const mi::mdl::IType_enum* type_enum,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* value_annotations,
    bool can_fail)
{
    mi::mdl::IType_enum::Predefined_id id_mdl = type_enum->get_predefined_id();
    IType_enum::Predefined_id id_int = DETAIL::mdl_enum_id_to_int_enum_id( id_mdl);

    mi::Uint32 count = type_enum->get_value_count();
    IType_enum::Values values( count);

    for( mi::Uint32 i = 0; i < count; ++i) {
        const mi::mdl::ISymbol* value_symbol;
        int code;
        type_enum->get_value( i, value_symbol, code);
        values[i] = std::make_pair( value_symbol->get_name(), code);
    }

    IExpression_factory* ef = get_expression_factory();

    // note: annotations cannot contain resources, hence we can safely ignore the mode
    Mdl_dag_converter anno_converter(
        ef,
        /*transaction*/ nullptr,
        /*resource_tagger*/ nullptr,
        /*code_dag*/ nullptr,
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        /*module_filename*/ nullptr,
        /*module_mdl_name*/ nullptr,
        /*prototype_tag*/ DB::Tag(),
        /*load_resources*/ false,
        /*user_modules_seen*/ nullptr);

    mi::base::Handle<const IAnnotation_block> annotations_int(
        annotations ? anno_converter.mdl_dag_node_vector_to_int_annotation_block(
            *annotations, /*qualified_name*/ symbol) : nullptr);

    count = value_annotations ? value_annotations->size() : 0;
    IType_enum::Value_annotations value_annotations_int( count);

    for (mi::Uint32 i = 0; i < count; ++i)
        value_annotations_int[i] = anno_converter.mdl_dag_node_vector_to_int_annotation_block(
            (*value_annotations)[i], /*qualified_name*/ symbol);

    mi::Sint32 errors;
    std::string mdl_symbol = MDL::encode_name_without_signature( symbol);
    const IType_enum* type_enum_int = tf->create_enum(
        mdl_symbol.c_str(), id_int, values, annotations_int, value_annotations_int, &errors);
    boost::ignore_unused(can_fail);
    ASSERT( M_SCENE, errors == 0 || can_fail);
    ASSERT( M_SCENE, type_enum_int || can_fail);
    return type_enum_int;
}

const IType_struct* create_struct(
    IType_factory* tf,
    const char* symbol,
    const mi::mdl::IType_struct* type_struct,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* field_annotations,
    bool can_fail)
{
    mi::mdl::IType_struct::Predefined_id id_mdl = type_struct->get_predefined_id();
    IType_struct::Predefined_id id_int = DETAIL::mdl_struct_id_to_int_struct_id( id_mdl);

    mi::Uint32 count = type_struct->get_compound_size();
    IType_struct::Fields fields( count);

    for( mi::Uint32 i = 0; i < count; ++i) {
        const mi::mdl::IType* type_mdl;
        const mi::mdl::ISymbol* field_symbol;
        type_struct->get_field( i, type_mdl, field_symbol);
        mi::base::Handle<const IType> type_int( mdl_type_to_int_type( tf, type_mdl));
        fields[i] = std::make_pair( type_int, field_symbol->get_name());
    }

    IExpression_factory* ef = get_expression_factory();

    // note: annotations cannot contain resources, hence we can safely ignore the mode
    Mdl_dag_converter anno_converter(
        ef,
        /*transaction*/ nullptr,
        /*resource_tagger*/ nullptr,
        /*code_dag*/ nullptr,
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        /*module_filename*/ nullptr,
        /*module_mdl_name*/ nullptr,
        /*prototype_tag*/ DB::Tag(),
        /*load_resources*/ false,
        /*user_modules_seen*/ nullptr);

    mi::base::Handle<const IAnnotation_block> annotations_int(
        annotations ? anno_converter.mdl_dag_node_vector_to_int_annotation_block(
            *annotations, /*qualified_name*/ symbol) : nullptr);

    count = field_annotations ? field_annotations->size() : 0;
    IType_enum::Value_annotations field_annotations_int( count);

    for (mi::Uint32 i = 0; i < count; ++i)
        field_annotations_int[i] = anno_converter.mdl_dag_node_vector_to_int_annotation_block(
            (*field_annotations)[i], /*qualified_name*/ symbol);

    mi::Sint32 errors;
    std::string mdl_symbol = MDL::encode_name_without_signature( symbol);
    const IType_struct* type_int = tf->create_struct(
        mdl_symbol.c_str(), id_int, fields, annotations_int, field_annotations_int, &errors);
    boost::ignore_unused(can_fail);
    ASSERT( M_SCENE, errors == 0 || can_fail);
    ASSERT( M_SCENE, type_int || can_fail);
    return type_int;
}

} // anonymous namespace

bool mdl_type_enum_to_int_type_test(
    IType_factory* tf,
    const mi::mdl::IType_enum* type,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* member_annotations)
{
    const mi::mdl::ISymbol* symbol = type->get_symbol();
    const char* symbol_name = symbol->get_name();
    ASSERT(M_SCENE, symbol_name);
    std::string prefixed_symbol_name = prefix_builtin_type_name(symbol_name);
    symbol_name = symbol_name ? prefixed_symbol_name.c_str() : nullptr;
    mi::base::Handle<const IType_enum> test_enum(
        create_enum(tf, symbol_name, type, annotations, member_annotations,
            /*can_fail=*/true));
    return test_enum.is_valid_interface();
}

bool mdl_type_struct_to_int_type_test(
    IType_factory* tf,
    const mi::mdl::IType_struct* type,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* member_annotations)
{
    const mi::mdl::ISymbol* symbol = type->get_symbol();
    const char* symbol_name = symbol->get_name();
    ASSERT(M_SCENE, symbol_name);
    std::string prefixed_symbol_name = prefix_builtin_type_name(symbol_name);
    symbol_name = symbol_name ? prefixed_symbol_name.c_str() : nullptr;
    mi::base::Handle<const IType_struct> test_struct(
        create_struct(tf, symbol_name, type, annotations, member_annotations,
            /*can_fail=*/true));
    return test_struct.is_valid_interface();
}

const IType* mdl_type_to_int_type(
    IType_factory* tf,
    const mi::mdl::IType* type,
    const Mdl_annotation_block* annotations,
    const Mdl_annotation_block_vector* member_annotations)
{
    mi::mdl::IType::Kind kind = type->get_kind();
    mi::mdl::IType::Kind skipped_kind = type->skip_type_alias()->get_kind();

    bool enum_or_struct = skipped_kind == mi::mdl::IType::TK_ENUM ||
        skipped_kind == mi::mdl::IType::TK_STRUCT;
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
        case mi::mdl::IType::TK_HAIR_BSDF:        return tf->create_hair_bsdf();
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
            const char* symbol_name = symbol ? symbol->get_name() : nullptr;
            return tf->create_alias( aliased_type_int.get(), modifiers_int, symbol_name);
        }

        case mi::mdl::IType::TK_ENUM: {
            const mi::mdl::IType_enum* type_enum = cast<mi::mdl::IType_enum>( type);
            const mi::mdl::ISymbol* symbol = type_enum->get_symbol();
            const char* symbol_name = symbol->get_name();
            ASSERT( M_SCENE, symbol_name);
            std::string prefixed_symbol_name = prefix_builtin_type_name( symbol_name);
            symbol_name = symbol_name ? prefixed_symbol_name.c_str() : nullptr;
            return create_enum(
                tf, symbol_name, type_enum, annotations, member_annotations, /*can_fail=*/false);
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
            std::string prefixed_symbol_name = prefix_builtin_type_name( symbol_name);
            symbol_name = symbol_name ? prefixed_symbol_name.c_str() : nullptr;
            return create_struct(
                tf, symbol_name, type_struct, annotations, member_annotations, /*can_fail=*/false);
        }

        case mi::mdl::IType::TK_TEXTURE: {
            const mi::mdl::IType_texture* type_texture = as<mi::mdl::IType_texture>( type);
            mi::mdl::IType_texture::Shape shape = type_texture->get_shape();
            IType_texture::Shape shape_int = DETAIL::mdl_shape_to_int_shape( shape);
            return tf->create_texture( shape_int);
        }

        case mi::mdl::IType::TK_FUNCTION:
            ASSERT( M_SCENE, false); return nullptr;
        case mi::mdl::IType::TK_AUTO:
            ASSERT( M_SCENE, false); return nullptr;
        case mi::mdl::IType::TK_ERROR:
            ASSERT( M_SCENE, false); return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

IValue* Mdl_dag_converter::mdl_value_to_int_value(
    const IType* type_int,
    const mi::mdl::IValue* value) const
{
    mi::mdl::IValue::Kind kind = value->get_kind();

    switch( kind) {
        case mi::mdl::IValue::VK_BAD:
            ASSERT( M_SCENE, false);
            return nullptr;
        case mi::mdl::IValue::VK_BOOL: {
            const mi::mdl::IValue_bool* value_bool = cast<mi::mdl::IValue_bool>( value);
            return m_vf->create_bool( value_bool->get_value());
        }
        case mi::mdl::IValue::VK_INT: {
            const mi::mdl::IValue_int* value_int = cast<mi::mdl::IValue_int>( value);
            return m_vf->create_int( value_int->get_value());
        }
        case mi::mdl::IValue::VK_ENUM: {
            const mi::mdl::IValue_enum* value_enum = cast<mi::mdl::IValue_enum>( value);
            const mi::mdl::IType_enum* type_enum = value_enum->get_type();
            mi::base::Handle<const IType_enum> type_enum_int(
                mdl_type_to_int_type<IType_enum>( m_tf.get(), type_enum));
            return m_vf->create_enum( type_enum_int.get(), value_enum->get_index());
        }
        case mi::mdl::IValue::VK_FLOAT: {
            const mi::mdl::IValue_float* value_float = cast<mi::mdl::IValue_float>( value);
            return m_vf->create_float( value_float->get_value());
        }
        case mi::mdl::IValue::VK_DOUBLE: {
            const mi::mdl::IValue_double* value_double = cast<mi::mdl::IValue_double>( value);
            return m_vf->create_double( value_double->get_value());
        }
        case mi::mdl::IValue::VK_STRING: {
            const mi::mdl::IValue_string* value_string = cast<mi::mdl::IValue_string>( value);
            return m_vf->create_string( value_string->get_value());
        }
        case mi::mdl::IValue::VK_VECTOR: {
            const mi::mdl::IValue_vector* value_vector = cast<mi::mdl::IValue_vector>( value);
            const mi::mdl::IType_vector* type_vector = value_vector->get_type();
            mi::base::Handle<const IType_vector> type_vector_int(
                mdl_type_to_int_type<IType_vector>( m_tf.get(), type_vector));
            IValue_vector* value_vector_int = m_vf->create_vector( type_vector_int.get());
            mi::Size n = value_vector->get_component_count();
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_vector->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    /* type not relevant */ nullptr, component));
                mi::Sint32 result = value_vector_int->set_value( i, component_int.get());
                ASSERT( M_SCENE, result == 0);
                boost::ignore_unused( result);
            }
            return value_vector_int;
        }
        case mi::mdl::IValue::VK_MATRIX: {
            const mi::mdl::IValue_matrix* value_matrix = cast<mi::mdl::IValue_matrix>( value);
            const mi::mdl::IType_matrix* type_matrix = value_matrix->get_type();
            mi::base::Handle<const IType_matrix> type_matrix_int(
                mdl_type_to_int_type<IType_matrix>( m_tf.get(), type_matrix));
            IValue_matrix* value_matrix_int = m_vf->create_matrix( type_matrix_int.get());
            mi::Size n = value_matrix->get_component_count();
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_matrix->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    /* type not relevant */ nullptr, component));
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
                type_array_int = mdl_type_to_int_type<IType_array>( m_tf.get(), type_array);
            }
            IValue_array* value_array_int = m_vf->create_array( type_array_int.get());
            mi::Size n = value_array->get_component_count();
            if( !type_array_int->is_immediate_sized())
                value_array_int->set_size( n);
            mi::base::Handle<const IType> component_type_int( type_array_int->get_element_type());
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_array->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    component_type_int.get(), component));
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
            return m_vf->create_color( red, green, blue);
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
                type_struct_int = mdl_type_to_int_type<IType_struct>( m_tf.get(), type_struct);
            }
            IValue_struct* value_struct_int = m_vf->create_struct( type_struct_int.get());
            mi::Size n = value_struct->get_component_count();
            for( mi::Size i = 0; i < n; ++i) {
                const mi::mdl::IValue* component
                    = value_struct->get_value( static_cast<mi::Uint32>( i));
                mi::base::Handle<const IType> component_type_int(
                    type_struct_int->get_field_type( i));
                mi::base::Handle<IValue> component_int( mdl_value_to_int_value(
                    component_type_int.get(), component));
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
                    mi::base::Handle<const IType_texture> type_texture_int(
                        mdl_type_to_int_type<IType_texture>( m_tf.get(), type_texture));
                    return m_vf->create_texture( type_texture_int.get(), DB::Tag());
                }
                if( mi::mdl::is<mi::mdl::IType_light_profile>( type_resource))
                    return m_vf->create_light_profile( DB::Tag());
                if( mi::mdl::is<mi::mdl::IType_bsdf_measurement>( type_resource))
                    return m_vf->create_bsdf_measurement( DB::Tag());
                ASSERT( M_SCENE, false);
            }
            mi::base::Handle<const IType_reference> type_reference_int(
                mdl_type_to_int_type<IType_reference>( m_tf.get(), type_reference));
            return m_vf->create_invalid_df( type_reference_int.get());
        }
        case mi::mdl::IValue::VK_TEXTURE: {
            const mi::mdl::IValue_texture* value_texture = cast<mi::mdl::IValue_texture>( value);
            const mi::mdl::IType_texture* type_texture = value_texture->get_type();
            mi::base::Handle<const IType_texture> type_texture_int(
                mdl_type_to_int_type<IType_texture>( m_tf.get(), type_texture));
            DB::Tag tag;
            Float32 gamma = 0.0f;
            bool needs_owner = false;
            std::string string_value_buf;
            if (m_load_resources) {
                tag = find_resource_tag(value_texture);
                if (tag.is_valid()) {
                    DB::Access<TEXTURE::Texture> tex(tag, m_transaction);
                    gamma = tex->get_effective_gamma(m_transaction);
                }
            }  else {
                const char* string_value = value_texture->get_string_value();
                tag = DB::Tag(value_texture->get_tag_value());
                gamma = value_texture->get_gamma_mode() == mi::mdl::IValue_texture::gamma_default ? 0.0f :
                    (value_texture->get_gamma_mode() == mi::mdl::IValue_texture::gamma_linear ? 1.0f : 2.2f);
                string_value_buf = strip_resource_owner_prefix( string_value);
                needs_owner = string_value_buf != string_value;
            }
            return m_vf->create_texture(
                type_texture_int.get(), tag, string_value_buf.c_str(),
                needs_owner ? m_module_mdl_name : nullptr, gamma);
        }
        case mi::mdl::IValue::VK_LIGHT_PROFILE: {
            const mi::mdl::IValue_light_profile* value_light_profile
                = cast<mi::mdl::IValue_light_profile>( value);
            DB::Tag tag = m_load_resources ?
                find_resource_tag(value_light_profile) :
                DB::Tag(value_light_profile->get_tag_value());
            return m_vf->create_light_profile( tag);
        }
        case mi::mdl::IValue::VK_BSDF_MEASUREMENT: {
            const mi::mdl::IValue_bsdf_measurement* value_bsdf_measurement
                = cast<mi::mdl::IValue_bsdf_measurement>( value);
            DB::Tag tag = m_load_resources ?
                find_resource_tag(value_bsdf_measurement) :
                DB::Tag(value_bsdf_measurement->get_tag_value());
            return m_vf->create_bsdf_measurement( tag);
        }
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

Mdl_dag_converter::Mdl_dag_converter(
    IExpression_factory* ef,
    DB::Transaction* transaction,
    mi::mdl::IResource_tagger* tagger,
    const mi::mdl::IGenerated_code_dag* code_dag,
    bool immutable_callees,
    bool create_direct_calls,
    const char* module_filename,
    const char* module_mdl_name,
    DB::Tag prototype_tag,
    bool load_resources,
    std::set<Mdl_tag_ident>* user_modules_seen)
    : m_ef(ef, mi::base::DUP_INTERFACE)
    , m_vf(m_ef->get_value_factory())
    , m_tf(m_vf->get_type_factory())
    , m_transaction(transaction)
    , m_tagger(tagger)
    , m_code_dag(code_dag)
    , m_immutable_callees(immutable_callees)
    , m_create_direct_calls(create_direct_calls)
    , m_module_filename(module_filename)
    , m_module_mdl_name(module_mdl_name)
    , m_prototype_tag(prototype_tag)
    , m_load_resources(load_resources)
    , m_user_modules_seen(user_modules_seen)
{
}

DB::Tag Mdl_dag_converter::find_resource_tag( const mi::mdl::IValue_resource* res) const
{
    int tag = res->get_tag_value();
    if( tag == 0 && m_tagger != nullptr)
        tag = m_tagger->get_resource_tag( res);
    return DB::Tag( tag);
}

IExpression* Mdl_dag_converter::mdl_dag_node_to_int_expr(
    const mi::mdl::DAG_node* node, const IType* type_int) const
{
    switch (node->get_kind()) {
    case mi::mdl::DAG_node::EK_CONSTANT:
        {
            const mi::mdl::DAG_constant* constant = cast<mi::mdl::DAG_constant>(node);
            const mi::mdl::IValue* value = constant->get_value();
            mi::base::Handle<IValue> value_int(mdl_value_to_int_value(type_int, value));

            return m_ef->create_constant(value_int.get());
        }
    case mi::mdl::DAG_node::EK_CALL:
        {
            const mi::mdl::DAG_call* call = cast<mi::mdl::DAG_call>(node);
            return m_create_direct_calls
                ? mdl_call_to_int_expr_direct(call, /*use_parameter_type*/ type_int != nullptr)
                : mdl_call_to_int_expr_indirect(call, /*use_parameter_type*/ type_int != nullptr);
        }
    case mi::mdl::DAG_node::EK_PARAMETER:
        {
            const mi::mdl::DAG_parameter* parameter = cast<mi::mdl::DAG_parameter>(node);
            mi::base::Handle<const IType> type(
                mdl_type_to_int_type(m_tf.get(), parameter->get_type()));
            mi::Size index = parameter->get_index();
            return m_ef->create_parameter(type.get(), index);
        }
    case mi::mdl::DAG_node::EK_TEMPORARY:
        {
            const mi::mdl::DAG_temporary* temporary = cast<mi::mdl::DAG_temporary>(node);
            mi::base::Handle<const IType> type(
                mdl_type_to_int_type(m_tf.get(), temporary->get_type()));
            mi::Size index = temporary->get_index();
            return m_ef->create_temporary(type.get(), index);
        }
    }

    ASSERT(M_SCENE, false);
    return nullptr;
}

IExpression* Mdl_dag_converter::mdl_call_to_int_expr_direct(
    const mi::mdl::DAG_call* call, bool use_parameter_type) const
{
    const mi::mdl::IType* return_type = call->get_type();
    mi::base::Handle<const IType> return_type_int(
        mdl_type_to_int_type(m_tf.get(), return_type));

    std::string mdl_name = encode_name_add_missing_signature( m_transaction, m_code_dag, call->get_name());
    std::string definition_name = get_db_name( mdl_name);
    DB::Tag definition_tag = m_transaction->name_to_tag( definition_name.c_str());

    ASSERT(M_SCENE, definition_tag.is_valid());

    SERIAL::Class_id class_id = m_transaction->get_class_id(definition_tag);
    if( class_id != ID_MDL_FUNCTION_DEFINITION) {
        ASSERT(M_SCENE, false);
        return nullptr;
    }

    DB::Access<Mdl_function_definition> function_definition(definition_tag, m_transaction);
    mi::base::Handle<const IType_list> parameter_types( function_definition->get_parameter_types());
    DB::Tag module_tag = function_definition->get_module(m_transaction);
    ASSERT(M_SCENE, module_tag.is_valid());
    Mdl_ident def_ident = function_definition->get_ident();
    std::string def_name = get_db_name(function_definition->get_mdl_name());

    mi::base::Handle<IExpression_list> arguments(m_ef->create_expression_list());
    mi::Uint32 n = call->get_argument_count();
    for (mi::Uint32 i = 0; i < n; i++) {

        const char* parameter_name = call->get_parameter_name(i);
        const mi::mdl::DAG_node* mdl_argument = call->get_argument(i);
        mi::base::Handle<const IType> parameter_type(
            use_parameter_type ? parameter_types->get_type(i) : (const IType *)nullptr);
        mi::base::Handle<const IExpression> argument(mdl_dag_node_to_int_expr(
            mdl_argument, parameter_type.get()));
        arguments->add_expression(parameter_name, argument.get());
    }

    if (m_user_modules_seen) {
        DB::Access<Mdl_module> module(module_tag, m_transaction);
        if (!module->is_standard_module())
            m_user_modules_seen->insert(Mdl_tag_ident(module_tag, module->get_ident()));
    }

    return m_ef->create_direct_call(
        return_type_int.get(),
        module_tag,
        Mdl_tag_ident(definition_tag, def_ident), def_name, arguments.get());
}

IExpression* Mdl_dag_converter::mdl_call_to_int_expr_indirect(
    const mi::mdl::DAG_call* call, bool use_parameter_type) const
{
    mi::mdl::IDefinition::Semantics sema =  call->get_semantic();
    bool is_array_constructor = sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
    bool is_array_length      = sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH;

    mi::mdl::IExpression::Operator op = mi::mdl::semantic_to_operator( sema);
    bool is_cast_operator        = op == mi::mdl::IExpression::OK_CAST;
    bool is_ternary_operator     = op == mi::mdl::IExpression::OK_TERNARY;
    bool is_array_index_operator = op == mi::mdl::IExpression::OK_ARRAY_INDEX;

    // get tag and class ID of call
    std::string definition_name;
    if( is_array_constructor) {
        definition_name = get_array_constructor_db_name();
        use_parameter_type = false;
    } else if( is_array_length) {
        definition_name = get_array_length_operator_db_name();
        use_parameter_type = false;
    } else if( is_cast_operator) {
        definition_name = get_cast_operator_db_name();
        use_parameter_type = false;
    } else if( is_ternary_operator) {
        definition_name = get_ternary_operator_db_name();
        use_parameter_type = false;
    } else if( is_array_index_operator) {
        definition_name = get_index_operator_db_name();
        use_parameter_type = false;
    } else {
        std::string mdl_name = encode_name_add_missing_signature( m_transaction, m_code_dag, call->get_name());
        definition_name = get_db_name( mdl_name);
    }

    DB::Tag definition_tag = m_transaction->name_to_tag( definition_name.c_str());
    ASSERT( M_SCENE, definition_tag);

    SERIAL::Class_id class_id = m_transaction->get_class_id( definition_tag);
    if( class_id != ID_MDL_FUNCTION_DEFINITION) {
        ASSERT( M_SCENE, false);
        return nullptr;
    }

    DB::Access<Mdl_function_definition> function_definition( definition_tag, m_transaction);
    mi::base::Handle<const IType_list> parameter_types( function_definition->get_parameter_types());

    // create argument list
    mi::base::Handle<IExpression_list> arguments( m_ef->create_expression_list());
    mi::Uint32 n = call->get_argument_count();
    ASSERT( M_SCENE, !is_cast_operator     || n == 1);
    ASSERT( M_SCENE, !is_array_constructor || n > 0);

    for( mi::Uint32 i = 0; i < n; ++i) {

        const mi::mdl::DAG_node* argument = call->get_argument(i);
        if( !argument)
            continue;
        mi::base::Handle<const IType> parameter_type(
            use_parameter_type ? parameter_types->get_type( i) : (const IType *) nullptr);
        mi::base::Handle<IExpression> argument_int(
            mdl_dag_node_to_int_expr( argument, parameter_type.get()));
        ASSERT( M_SCENE, argument_int);
        std::string parameter_name;
        if( is_array_constructor)
            parameter_name = std::to_string( i);
        else
            parameter_name = function_definition->get_parameter_name( i);
        arguments->add_expression( parameter_name.c_str(), argument_int.get());
    }

    // create function call from definition
    mi::Sint32 errors = 0;
    Mdl_function_call* function_call = nullptr;
    if( is_array_constructor) {
        function_call = function_definition->create_array_constructor_call_internal(
            m_transaction, arguments.get(), /*allow_ek_parameter*/ true, m_immutable_callees,
            &errors);
    } else if( is_array_length) {
        function_call = function_definition->create_array_length_operator_call_internal(
            m_transaction, arguments.get(), m_immutable_callees, &errors);
    } else if( is_cast_operator) {
        // add a dummy argument to pass the return type
        mi::base::Handle<const IType> ret_type(
            mdl_type_to_int_type( m_tf.get(), call->get_type()));
        mi::base::Handle<IValue> value( m_vf->create( ret_type.get()));
        mi::base::Handle<IExpression> expr( m_ef->create_constant( value.get()));
        arguments->add_expression( "cast_return", expr.get());
        function_call = function_definition->create_cast_operator_call_internal(
            m_transaction, arguments.get(), m_immutable_callees, &errors);
    } else if( is_ternary_operator) {
        function_call = function_definition->create_ternary_operator_call_internal(
            m_transaction, arguments.get(), m_immutable_callees, &errors);
    } else if( is_array_index_operator) {
        function_call = function_definition->create_array_index_operator_call_internal(
            m_transaction, arguments.get(), m_immutable_callees, &errors);
    } else {
        function_call = function_definition->create_function_call_internal( m_transaction,
            arguments.get(), /*allow_ek_parameter*/ true, m_immutable_callees, &errors);
    }
    ASSERT( M_SCENE, function_call);
    if( !function_call) {
        LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
            "Instantiation of function definition \"%s\" failed.", definition_name.c_str());
        return nullptr;
    }

    if( m_user_modules_seen) {
        DB::Tag module_tag = function_definition->get_module( m_transaction);
        DB::Access<Mdl_module> module( module_tag, m_transaction);
        if( !module->is_standard_module())
            m_user_modules_seen->insert( Mdl_tag_ident( module_tag, module->get_ident()));
    }

    // store function call
    std::string call_name_base = m_immutable_callees
        ? definition_name + "__default_call__" : definition_name;
    std::string call_name
        = DETAIL::generate_unique_db_name( m_transaction, call_name_base.c_str());
    DB::Tag call_tag = m_transaction->store_for_reference_counting(
        function_call, call_name.c_str(), m_transaction->get_scope()->get_level());

    // create call expression
    const mi::mdl::IType* return_type = call->get_type();
    mi::base::Handle<const IType> return_type_int(
        mdl_type_to_int_type( m_tf.get(), return_type));

    return m_ef->create_call( return_type_int.get(), call_tag);
}

IAnnotation_block* Mdl_dag_converter::mdl_dag_node_vector_to_int_annotation_block(
    const Mdl_annotation_block& mdl_annotations,
    const char* qualified_name) const
{
    if (mdl_annotations.empty())
        return nullptr;

    IAnnotation_block* block = m_ef->create_annotation_block();

    for (mi::Size i = 0; i < mdl_annotations.size(); ++i) {

        const mi::mdl::DAG_call* call = cast<mi::mdl::DAG_call>(mdl_annotations[i]);
        mi::base::Handle<IAnnotation> annotation(
            mdl_dag_call_to_int_annotation(call, qualified_name));
        block->add_annotation(annotation.get());
    }

    return block;
}

IAnnotation* Mdl_dag_converter::mdl_dag_call_to_int_annotation(
    const mi::mdl::DAG_call* call, const char* qualified_name) const
{
    mi::Uint32 n = call->get_argument_count();
    mi::base::Handle<IExpression_list> arguments(m_ef->create_expression_list());
    for (mi::Uint32 i = 0; i < n; ++i) {
        const mi::mdl::DAG_node* argument = call->get_argument(i);
        mi::base::Handle<IExpression> argument_int(mdl_dag_node_to_int_expr_localized(
            argument, call, /*type_int*/ nullptr, qualified_name));
        ASSERT(M_SCENE, argument_int);
        ASSERT(M_SCENE, argument_int->get_kind() == IExpression::EK_CONSTANT);
        const char* parameter_name = call->get_parameter_name(i);
        arguments->add_expression(parameter_name, argument_int.get());
    }

    const char* name = call->get_name();
    std::string mdl_name = encode_name_add_missing_signature( m_transaction, m_code_dag, name);
    return m_ef->create_annotation( m_transaction, mdl_name.c_str(), arguments.get());
}

namespace {

void setup_translation_unit(
    I18N::Mdl_translator_module::Translation_unit & translation_unit
    , DB::Transaction* transaction
    , DB::Tag prototype_tag
    , const char* module_name
    , const char* qualified_name
    , const char* value_string
)
{
    // Set values when prototype_tag is null (most common case)
    // prototype_tag is used for material and function variants
    translation_unit.set_module_name(module_name);
    translation_unit.set_context(qualified_name);
    translation_unit.set_source(value_string);

    if (prototype_tag)
    {
        // Search the values in parent material/function
        SERIAL::Class_id class_id = transaction->get_class_id(prototype_tag);
        if (class_id == Mdl_function_definition::id)
        {
            DB::Access<MDL::Mdl_function_definition> element(prototype_tag, transaction);
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

} // namespace

IExpression* Mdl_dag_converter::mdl_dag_node_to_int_expr_localized(
    const mi::mdl::DAG_node* argument,
    const mi::mdl::DAG_call* call,
    const IType* type_int,
    const char* qualified_name) const
{
    if (argument->get_kind() == mi::mdl::DAG_node::EK_CONSTANT) {
        // If the qualified name is set and the annotation is one we translate the translate it
        SYSTEM::Access_module<I18N::Mdl_translator_module> mdl_translator(false);
        if (qualified_name &&  mdl_translator->need_translation(call->get_name())) {
            // Need to translate, check the type of value
            const mi::mdl::DAG_constant* constant = cast<mi::mdl::DAG_constant>(argument);
            const mi::mdl::IValue* value = constant->get_value();
            if (value->get_kind() == mi::mdl::IValue::VK_STRING)
            {
                const mi::mdl::IValue_string* value_string = cast<mi::mdl::IValue_string>(value);
                I18N::Mdl_translator_module::Translation_unit translation_unit;
                setup_translation_unit(
                    translation_unit
                    , m_transaction
                    , m_prototype_tag
                    , m_module_mdl_name
                    , qualified_name
                    , value_string->get_value()
                );
                if (0 == mdl_translator->translate(translation_unit))
                {
                    const std::string& translated_string = translation_unit.get_target();
                    mi::base::Handle<IValue> value_int(
                        m_vf->create_string_localized(translated_string.c_str(), value_string->get_value()));
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
                        I18N::Mdl_translator_module::Translation_unit translation_unit;
                        setup_translation_unit(
                            translation_unit
                            , m_transaction
                            , m_prototype_tag
                            , m_module_mdl_name
                            , qualified_name
                            , value_string->get_value()
                        );
                        translation_unit.set_source(value_string->get_value());
                        std::string translated_string(translation_unit.get_source());
                        mi::Sint32 result;
                        if (0 == mdl_translator->translate(translation_unit))
                        {
                            translated_string = translation_unit.get_target();
                            mi::base::Handle<IValue> component_int(
                                m_vf->create_string_localized(translated_string.c_str(), value_string->get_value()));
                            result = value_array_int->set_value(i, component_int.get());
                        }
                        else
                        {
                            mi::base::Handle<IValue> component_int(
                                vf->create_string(translated_string.c_str()));
                            result = value_array_int->set_value(i, component_int.get());
                        }
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
    return mdl_dag_node_to_int_expr(argument, /*type_int*/ nullptr);
}

// **********  Code_dag ****************************************************************************

const char* Code_dag::get_name( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_name( index_uint32)
        : m_code_dag->get_function_name( index_uint32);
}

const char* Code_dag::get_simple_name( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_simple_material_name( index_uint32)
        : m_code_dag->get_simple_function_name( index_uint32);
}

mi::mdl::IDefinition::Semantics Code_dag::get_semantics( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? mi::mdl::IDefinition::DS_UNKNOWN
        : m_code_dag->get_function_semantics( index_uint32);
}

bool Code_dag::get_exported( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_exported( index_uint32)
        : m_code_dag->get_function_property(
            index_uint32, mi::mdl::IGenerated_code_dag::FP_IS_EXPORTED);
}

bool Code_dag::get_uniform( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? false
        : m_code_dag->get_function_property(
            index_uint32, mi::mdl::IGenerated_code_dag::FP_IS_UNIFORM);
}

const char* Code_dag::get_cloned_name( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_cloned_material_name( index_uint32)
        : m_code_dag->get_cloned_function_name( index_uint32);
    }

const char* Code_dag::get_original_name( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_original_material_name( index_uint32)
        : m_code_dag->get_original_function_name( index_uint32);
}

const mi::mdl::IType* Code_dag::get_return_type( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? nullptr
        : m_code_dag->get_function_return_type( index_uint32);
}

mi::Size Code_dag::get_parameter_count( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_parameter_count( index_uint32)
        : m_code_dag->get_function_parameter_count( index_uint32);
}

const mi::mdl::IType* Code_dag::get_parameter_type( mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_type( index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_type( index_uint32, parameter_index_uint32);
}

const char* Code_dag::get_parameter_type_name( mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_type_name( index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_type_name( index_uint32, parameter_index_uint32);
}

const char* Code_dag::get_parameter_name( mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_name( index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_name( index_uint32, parameter_index_uint32);
}

mi::Size Code_dag::get_parameter_index( mi::Size index, const char* parameter_name)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_parameter_index( index_uint32, parameter_name)
        : m_code_dag->get_function_parameter_index( index_uint32, parameter_name);
}

const mi::mdl::DAG_node* Code_dag::get_parameter_enable_if_condition(
    mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_enable_if_condition( index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_enable_if_condition( index_uint32, parameter_index_uint32);
}

mi::Size Code_dag::get_parameter_enable_if_condition_users(
    mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_enable_if_condition_users(
            index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_enable_if_condition_users(
            index_uint32, parameter_index_uint32);
}

mi::Size Code_dag::get_parameter_enable_if_condition_user(
    mi::Size index, mi::Size parameter_index, mi::Size user_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    mi::Uint32 user_index_uint32      = static_cast<mi::Uint32>( user_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_enable_if_condition_user(
            index_uint32, parameter_index_uint32, user_index_uint32)
        : m_code_dag->get_function_parameter_enable_if_condition_user(
            index_uint32, parameter_index_uint32, user_index_uint32);
}

const mi::mdl::DAG_node* Code_dag::get_parameter_default( mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_default( index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_default( index_uint32, parameter_index_uint32);
}

mi::Size Code_dag::get_parameter_annotation_count( mi::Size index, mi::Size parameter_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32 = static_cast<mi::Uint32>( parameter_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_annotation_count( index_uint32, parameter_index_uint32)
        : m_code_dag->get_function_parameter_annotation_count( index_uint32, parameter_index_uint32);
}

const mi::mdl::DAG_node* Code_dag::get_parameter_annotation(
    mi::Size index, mi::Size parameter_index, mi::Size annotation_index)
{
    mi::Uint32 index_uint32            = static_cast<mi::Uint32>( index);
    mi::Uint32 parameter_index_uint32  = static_cast<mi::Uint32>( parameter_index);
    mi::Uint32 annotation_index_uint32 = static_cast<mi::Uint32>( annotation_index);
    return m_is_material
        ? m_code_dag->get_material_parameter_annotation(
            index_uint32, parameter_index_uint32, annotation_index_uint32)
        : m_code_dag->get_function_parameter_annotation(
            index_uint32, parameter_index_uint32, annotation_index_uint32);
}

mi::Size Code_dag::get_temporary_count( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_temporary_count( index_uint32)
        : m_code_dag->get_function_temporary_count( index_uint32);
}

const mi::mdl::DAG_node* Code_dag::get_temporary( mi::Size index, mi::Size temporary_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 temporary_index_uint32 = static_cast<mi::Uint32>( temporary_index);
    return m_is_material
        ? m_code_dag->get_material_temporary( index_uint32, temporary_index_uint32)
        : m_code_dag->get_function_temporary( index_uint32, temporary_index_uint32);
}

const char* Code_dag::get_temporary_name( mi::Size index, mi::Size temporary_index)
{
    mi::Uint32 index_uint32           = static_cast<mi::Uint32>( index);
    mi::Uint32 temporary_index_uint32 = static_cast<mi::Uint32>( temporary_index);
    return m_is_material
        ? m_code_dag->get_material_temporary_name( index_uint32, temporary_index_uint32)
        : m_code_dag->get_function_temporary_name( index_uint32, temporary_index_uint32);
}

mi::Size Code_dag::get_annotation_count( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_annotation_count( index_uint32)
        : m_code_dag->get_function_annotation_count( index_uint32);
}

const mi::mdl::DAG_node* Code_dag::get_annotation( mi::Size index, mi::Size annotation_index)
{
    mi::Uint32 index_uint32            = static_cast<mi::Uint32>( index);
    mi::Uint32 annotation_index_uint32 = static_cast<mi::Uint32>( annotation_index);
    return m_is_material
        ? m_code_dag->get_material_annotation( index_uint32, annotation_index_uint32)
        : m_code_dag->get_function_annotation( index_uint32, annotation_index_uint32);
}

mi::Size Code_dag::get_return_annotation_count( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_return_annotation_count( index_uint32)
        : m_code_dag->get_function_return_annotation_count( index_uint32);
}

const mi::mdl::DAG_node* Code_dag::get_return_annotation( mi::Size index, mi::Size annotation_index)
{
    mi::Uint32 index_uint32            = static_cast<mi::Uint32>( index);
    mi::Uint32 annotation_index_uint32 = static_cast<mi::Uint32>( annotation_index);
    return m_is_material
        ? m_code_dag->get_material_return_annotation( index_uint32, annotation_index_uint32)
        : m_code_dag->get_function_return_annotation( index_uint32, annotation_index_uint32);
}

const mi::mdl::DAG_node* Code_dag::get_body( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? m_code_dag->get_material_value( index_uint32)
        : m_code_dag->get_function_body( index_uint32);
}

const mi::mdl::DAG_hash* Code_dag::get_hash( mi::Size index)
{
    mi::Uint32 index_uint32 = static_cast<mi::Uint32>( index);
    return m_is_material
        ? nullptr
        : m_code_dag->get_function_hash( index_uint32);
}

// ********** Conversion from MI::MDL to mi::mdl ***************************************************

namespace {

/// Converts MI::MDL::IValue_texture to mi::mdl::IValue_texture or IValue_invalid_ref.
const mi::mdl::IValue* int_value_texture_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_texture* mdl_type,
    const IValue_texture* texture)
{
    std::string resource_name;
    DB::Tag_version tag_version, image_tag_version;
    mi::Float32 gamma;
    DB::Tag image_tag, volume_tag;
    DB::Tag tag = texture->get_value();
    if( tag) {

        SERIAL::Class_id class_id = transaction->get_class_id( tag);
        if( class_id != TEXTURE::Texture::id) {
            const char* name = transaction->tag_to_name( tag);
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "Incorrect type for texture resource \"%s\".", name ? name : "");
            return vf->create_invalid_ref( mdl_type);
        }

        DB::Access<TEXTURE::Texture> db_texture( tag, transaction);
        image_tag = db_texture->get_image();
        if( image_tag) {
            class_id = transaction->get_class_id( image_tag);
            if( class_id != DBIMAGE::Image::id) {
                const char* name = transaction->tag_to_name(image_tag);
                LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Incorrect type for image resource \"%s\".", name ? name : "");
                return vf->create_invalid_ref( mdl_type);
            }

            DB::Access<DBIMAGE::Image> image( image_tag, transaction);
            resource_name = image->get_mdl_file_path();
            tag_version = transaction->get_tag_version( tag);
            image_tag_version = transaction->get_tag_version( image_tag);
            gamma = db_texture->get_gamma();
        }
        else
            return vf->create_invalid_ref( mdl_type);

    } else {

        const char* unresolved_mdl_url = texture->get_unresolved_mdl_url();
        if( unresolved_mdl_url == nullptr || unresolved_mdl_url[0] == '\0')
            return vf->create_invalid_ref( mdl_type);

        // prepend the resource url with its owner module (if available)
        const char* owner_name = texture->get_owner_module();
        if( owner_name != nullptr && owner_name[0] != '\0')
            resource_name = std::string( owner_name) + "::" + unresolved_mdl_url;
        else
            resource_name = unresolved_mdl_url;

        gamma = texture->get_gamma();
    }

    // convert gamma value
    mi::mdl::IValue_texture::gamma_mode mdl_gamma;
    if( gamma == 1.0f)
        mdl_gamma = mi::mdl::IValue_texture::gamma_linear;
    else if( gamma == 2.2f)
        mdl_gamma = mi::mdl::IValue_texture::gamma_srgb;
    else
        mdl_gamma = mi::mdl::IValue_texture::gamma_default;

    // FIXME:
    const char* selector = "";
    mi::Uint32 hash = volume_tag ? 0 : get_hash( resource_name, gamma, tag_version, image_tag_version);
    return vf->create_texture(
        mdl_type, resource_name.c_str(), mdl_gamma, selector, tag.get_uint(), hash);
}

/// Converts MI::MDL::IValue_light_profile to mi::mdl::IValue_light_profile or IValue_invalid_ref.
const mi::mdl::IValue* int_value_light_profile_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_light_profile* mdl_type,
    const IValue_light_profile* light_profile)
{
    std::string resource_name;
    DB::Tag_version tag_version;

    DB::Tag tag = light_profile->get_value();
    if( tag) {

        SERIAL::Class_id class_id = transaction->get_class_id( tag);
        if( class_id != LIGHTPROFILE::Lightprofile::id) {
            const char* name = transaction->tag_to_name( tag);
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "Incorrect type for light profile resource \"%s\".", name ? name : "");
            return vf->create_invalid_ref( mdl_type);
        }

        DB::Access<LIGHTPROFILE::Lightprofile> db_lightprofile( tag, transaction);
        resource_name = db_lightprofile->get_mdl_file_path();
        tag_version = transaction->get_tag_version( tag);

    } else {

        const char* unresolved_mdl_url = light_profile->get_unresolved_mdl_url();
        if( unresolved_mdl_url == nullptr || unresolved_mdl_url[0] == '\0')
            return vf->create_invalid_ref( mdl_type);

        // prepend the resource url with its owner module (if available)
        const char* owner_name = light_profile->get_owner_module();
        if( owner_name != nullptr && owner_name[0] != '\0')
            resource_name = std::string( owner_name) + "::" + unresolved_mdl_url;
        else
            resource_name = unresolved_mdl_url;

    }

    mi::Uint32 hash = get_hash( resource_name, tag_version);
    return vf->create_light_profile( mdl_type, resource_name.c_str(), tag.get_uint(), hash);
}

/// Converts MI::MDL::IValue_bsdf_measurement to mi::mdl::IValue_bsdf_measurement or IValue_invalid_ref.
const mi::mdl::IValue* int_value_bsdf_measurement_to_mdl_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_bsdf_measurement* mdl_type,
    const IValue_bsdf_measurement* bsdf_measurement)
{
    std::string resource_name;
    DB::Tag_version tag_version;

    DB::Tag tag = bsdf_measurement->get_value();
    if( tag) {

        SERIAL::Class_id class_id = transaction->get_class_id( tag);
        if( class_id != BSDFM::Bsdf_measurement::id) {
            const char* name = transaction->tag_to_name( tag);
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "Incorrect type for BSDF measurement resource \"%s\".", name ? name : "");
            return vf->create_invalid_ref( mdl_type);
        }

        DB::Access<BSDFM::Bsdf_measurement> db_bsdf_measurement( tag, transaction);
        resource_name = db_bsdf_measurement->get_mdl_file_path();
        tag_version = transaction->get_tag_version( tag);

    } else {

        const char* unresolved_mdl_url = bsdf_measurement->get_unresolved_mdl_url();
        if( unresolved_mdl_url == nullptr || unresolved_mdl_url[0] == '\0')
            return vf->create_invalid_ref( mdl_type);

        // prepend the resource url with its owner module (if available)
        const char* owner_name = bsdf_measurement->get_owner_module();
        if( owner_name != nullptr && owner_name[0] != '\0')
            resource_name = std::string( owner_name) + "::" + unresolved_mdl_url;
        else
            resource_name = unresolved_mdl_url;

    }

    mi::Uint32 hash = get_hash( resource_name, tag_version);
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
                return nullptr;
            mi::base::Handle<const IValue_bool> value_bool(
                value->get_interface<IValue_bool>());
            return vf->create_bool( value_bool->get_value());
        }
        case IValue::VK_INT: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_INT)
                return nullptr;
            mi::base::Handle<const IValue_int> value_int(
                value->get_interface<IValue_int>());
            return vf->create_int( value_int->get_value());
        }
        case IValue::VK_ENUM: {
            const mi::mdl::IType_enum* mdl_type_enum
                = as<mi::mdl::IType_enum>( stripped_mdl_type);
            if( !mdl_type_enum)
                return nullptr;
            mi::base::Handle<const IValue_enum> value_enum(
                value->get_interface<IValue_enum>());
            mi::Size index = value_enum->get_index();
            return vf->create_enum( mdl_type_enum, index);
        }
        case IValue::VK_FLOAT: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_FLOAT)
                return nullptr;
            mi::base::Handle<const IValue_float> value_float(
                value->get_interface<IValue_float>());
            return vf->create_float( value_float->get_value());
        }
        case IValue::VK_DOUBLE: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_DOUBLE)
                return nullptr;
            mi::base::Handle<const IValue_double> value_double(
                value->get_interface<IValue_double>());
            return vf->create_double( value_double->get_value());
        }
        case IValue::VK_STRING: {
            if( stripped_mdl_type_kind != mi::mdl::IType::TK_STRING)
                return nullptr;
            {
                mi::base::Handle<const IValue_string_localized> value_string(
                    value->get_interface<IValue_string_localized>());
                if (value_string) {
                    return vf->create_string(value_string->get_original_value());
                }
            }
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
                return nullptr;
            mi::Size n = value_compound->get_size();
            mdl_type_compound
                = convert_deferred_sized_into_immediate_sized_array( vf, mdl_type_compound, n);
            mi::Size type_n = mdl_type_compound->get_compound_size();
            if( type_n != n)
                return nullptr;
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
            return vf->create_compound(
                mdl_type_compound, mdl_element_values.data(), n);
        }
        case IValue::VK_TEXTURE: {
            mi::base::Handle<const IValue_texture> value_texture(
                value->get_interface<IValue_texture>());
            const mi::mdl::IType_texture* mdl_type_texture
                = as<mi::mdl::IType_texture>( stripped_mdl_type);
            if( !mdl_type_texture)
                return nullptr;
            return int_value_texture_to_mdl_value(
                transaction, vf, mdl_type_texture, value_texture.get());
        }
        case IValue::VK_LIGHT_PROFILE: {
            mi::base::Handle<const IValue_light_profile> value_light_profile(
                value->get_interface<IValue_light_profile>());
            const mi::mdl::IType_light_profile* mdl_type_light_profile
                = as<mi::mdl::IType_light_profile>( stripped_mdl_type);
            if( !mdl_type_light_profile)
                return nullptr;
            return int_value_light_profile_to_mdl_value(
                transaction, vf, mdl_type_light_profile, value_light_profile.get());
        }
        case IValue::VK_BSDF_MEASUREMENT: {
            mi::base::Handle<const IValue_bsdf_measurement> value_bsdf_measurement(
                value->get_interface<IValue_bsdf_measurement>());
            const mi::mdl::IType_bsdf_measurement* mdl_type_bsdf_measurement
                = as<mi::mdl::IType_bsdf_measurement>( stripped_mdl_type);
            if( !mdl_type_bsdf_measurement)
                return nullptr;
            return int_value_bsdf_measurement_to_mdl_value(
                transaction, vf, mdl_type_bsdf_measurement, value_bsdf_measurement.get());
        }
        case IValue::VK_INVALID_DF: {
            const mi::mdl::IType_reference* mdl_type_reference
                = as<mi::mdl::IType_reference>( stripped_mdl_type);
            if( !mdl_type_reference)
                return nullptr;
            return vf->create_invalid_ref( mdl_type_reference);
        }
        case IValue::VK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

const mi::mdl::IType* int_type_to_mdl_type(
    const IType *type,
    mi::mdl::IType_factory &tf)
{
    // the type factory from mi::mdl::IMDL itself has no valid symbol table
    mi::mdl::ISymbol_table* symtab = tf.get_symbol_table();
    ASSERT( M_SCENE, symtab && "type factory has no valid symbol table");

    switch (type->get_kind()) {
    case IType::TK_ALIAS: {
        mi::base::Handle<const IType_alias> int_alias_type(
            type->get_interface<IType_alias>());
        mi::base::Handle<const IType> int_aliased_type(int_alias_type->skip_all_type_aliases());
        return int_type_to_mdl_type(int_aliased_type.get(), tf);
    }
    case IType::TK_ENUM: {
        mi::base::Handle<const IType_enum> int_enum_type(
            type->get_interface<IType_enum>());

        switch (int_enum_type->get_predefined_id()) {
        case IType_enum::EID_INTENSITY_MODE:
            return tf.get_predefined_enum(mi::mdl::IType_enum::EID_INTENSITY_MODE);
        case IType_enum::EID_TEX_GAMMA_MODE:
            return tf.get_predefined_enum(mi::mdl::IType_enum::EID_TEX_GAMMA_MODE);
        case IType_enum::EID_USER:
            break;
        default:
            ASSERT(M_SCENE, !"unhandled predefinied enum type.");
            return nullptr;
        }

        if (mi::mdl::IType_enum const *te = tf.lookup_enum(int_enum_type->get_symbol())) {
            // an enum with this name already exists, assume it's the right one
            return te;
        }

        const mi::mdl::ISymbol *s = symtab->create_symbol(int_enum_type->get_symbol());
        mi::mdl::IType_enum* te = tf.create_enum(s);
        for (mi::Size i = 0, n = int_enum_type->get_size(); i < n; ++i) {

            const mi::mdl::ISymbol *evs = symtab->create_symbol(int_enum_type->get_value_name(i));
            te->add_value(evs, int_enum_type->get_value_code(i));
        }
        return te;
    }
    case IType::TK_ARRAY: {
        mi::base::Handle<const IType_array> int_array_type(
            type->get_interface<IType_array>());

        mi::base::Handle<const IType> int_elem_type(
            int_array_type->get_element_type());

        const mi::mdl::IType* element_type = int_type_to_mdl_type(int_elem_type.get(), tf);
        if (int_array_type->is_immediate_sized()) {

            return tf.create_array(element_type, int_array_type->get_size());
        }
        else {
            // deferred size arrays are not supported, here
            return tf.create_error();
        }
    }
    case IType::TK_STRUCT:
    {
        mi::base::Handle<const IType_struct> int_struct_type(
            type->get_interface<IType_struct>());
        switch (int_struct_type->get_predefined_id()) {
        case IType_struct::SID_MATERIAL:
            return tf.get_predefined_struct(mi::mdl::IType_struct::SID_MATERIAL);
        case IType_struct::SID_MATERIAL_EMISSION:
            return tf.get_predefined_struct(mi::mdl::IType_struct::SID_MATERIAL_EMISSION);
        case IType_struct::SID_MATERIAL_GEOMETRY:
            return tf.get_predefined_struct(mi::mdl::IType_struct::SID_MATERIAL_GEOMETRY);
        case IType_struct::SID_MATERIAL_SURFACE:
            return tf.get_predefined_struct(mi::mdl::IType_struct::SID_MATERIAL_SURFACE);
        case IType_struct::SID_MATERIAL_VOLUME:
            return tf.get_predefined_struct(mi::mdl::IType_struct::SID_MATERIAL_VOLUME);
        case IType_struct::SID_USER:
            break;
        default:
            ASSERT(M_SCENE, !"unhandled predefinied struct type.");
            return nullptr;
        }

        if (mi::mdl::IType_struct const *st = tf.lookup_struct(int_struct_type->get_symbol())) {
            // an struct with this name already exists, assume it's the right one
            return st;
        }

        const mi::mdl::ISymbol *s = symtab->create_symbol(int_struct_type->get_symbol());

        mi::mdl::IType_struct* st = tf.create_struct(s);
        for (mi::Size i = 0, n = int_struct_type->get_size(); i < n; ++i) {

            mi::base::Handle<const IType> field(int_struct_type->get_field_type(i));
            const mi::mdl::ISymbol *fs = symtab->create_symbol(int_struct_type->get_field_name(i));

            st->add_field(int_type_to_mdl_type(field.get(), tf), fs);
        }
        return st;
    }
    case IType::TK_BOOL:
        return tf.create_bool();
    case IType::TK_INT:
        return tf.create_int();
    case IType::TK_FLOAT:
        return tf.create_float();
    case IType::TK_DOUBLE:
        return tf.create_double();
    case IType::TK_STRING:
        return tf.create_string();
    case IType::TK_VECTOR:
    {
        mi::base::Handle<IType_vector const> v_tp(type->get_interface<IType_vector>());
        mi::base::Handle<IType const>        e_tp(v_tp->get_element_type());

        mi::mdl::IType_atomic const *a_tp = cast<mi::mdl::IType_atomic>(
            int_type_to_mdl_type(e_tp.get(), tf));
        return tf.create_vector(a_tp, int(v_tp->get_size()));
    }
    case IType::TK_MATRIX:
    {
        mi::base::Handle<IType_matrix const> m_tp(type->get_interface<IType_matrix>());
        mi::base::Handle<IType const>        e_tp(m_tp->get_element_type());

        mi::mdl::IType_vector const *v_tp = cast<mi::mdl::IType_vector>(
            int_type_to_mdl_type(e_tp.get(), tf));
        return tf.create_matrix(v_tp, int(m_tp->get_size()));
    }
    case IType::TK_COLOR:
        return tf.create_color();
    case IType::TK_TEXTURE:
    {
        mi::base::Handle<IType_texture const> t_tp(type->get_interface<IType_texture>());

        switch (t_tp->get_shape()) {
        case IType_texture::TS_2D:
            return tf.create_texture(mi::mdl::IType_texture::TS_2D);
        case IType_texture::TS_3D:
            return tf.create_texture(mi::mdl::IType_texture::TS_3D);
        case IType_texture::TS_CUBE:
            return tf.create_texture(mi::mdl::IType_texture::TS_CUBE);
        case IType_texture::TS_PTEX:
            return tf.create_texture(mi::mdl::IType_texture::TS_PTEX);
        case IType_texture::TS_BSDF_DATA:
            return tf.create_texture(mi::mdl::IType_texture::TS_BSDF_DATA);
        case IType_texture::TS_FORCE_32_BIT:
            // not a real shape
            break;
        }
    }
    break;
    case IType::TK_LIGHT_PROFILE:
        return tf.create_light_profile();
    case IType::TK_BSDF_MEASUREMENT:
        return tf.create_bsdf_measurement();
    case IType::TK_BSDF:
        return tf.create_bsdf();
    case IType::TK_HAIR_BSDF:
        return tf.create_hair_bsdf();
    case IType::TK_EDF:
        return tf.create_edf();
    case IType::TK_VDF:
        return tf.create_vdf();
    case IType::TK_FORCE_32_BIT:
        // not a real type
        break;
    }
    ASSERT(M_SCENE, !"unsupported type kind");
    return tf.create_error();
}

const mi::mdl::DAG_node* int_expr_to_mdl_dag_node(
    DB::Transaction* transaction,
    mi::mdl::IDag_builder* builder,
    const mi::mdl::IType* type,
    const IExpression* expr)
{
    return Mdl_dag_builder<mi::mdl::IDag_builder>(
        transaction, builder, /*compiled_material*/ nullptr)
        .int_expr_to_mdl_dag_node( type, expr);
}

const mi::mdl::DAG_node* int_expr_to_mdl_dag_node(
    DB::Transaction* transaction,
    mi::mdl::IGenerated_code_dag::DAG_node_factory* factory,
    const mi::mdl::IType* type,
    const IExpression* expr)
{
    return Mdl_dag_builder<mi::mdl::IGenerated_code_dag::DAG_node_factory>(
        transaction, factory, /*compiled_material*/ nullptr)
        .int_expr_to_mdl_dag_node( type, expr);
}


// ********** Conversion from MI::MDL to MI::MDL ***************************************************

IExpression* int_expr_call_to_int_expr_direct_call(
    DB::Transaction* transaction,
    IExpression_factory* ef,
    const IExpression* expr,
    const std::vector<mi::base::Handle<const IExpression>>& parameters,
    Execution_context* context)
{
    for( mi::Size i = 0, n = parameters.size(); i < n; ++i)
        if( expr == parameters[i].get()) {
            mi::base::Handle<const IType> type( expr->get_type());
            return ef->create_parameter( type.get(), i);
        }

    switch( expr->get_kind()) {

        case IExpression::EK_CONSTANT:
        case IExpression::EK_DIRECT_CALL:
        case IExpression::EK_PARAMETER:
            return ef->clone( expr, transaction, /*copy_immutable_calls*/ false);

        case IExpression::EK_CALL: {

            mi::base::Handle<const IExpression_call> call(
                expr->get_interface<IExpression_call>());
            mi::base::Handle<const IType> type( call->get_type());
            DB::Tag tag = call->get_call();
            SERIAL::Class_id class_id = transaction->get_class_id( tag);
            if( class_id != ID_MDL_FUNCTION_CALL) {
               add_error_message(
                   context, "The call expression refers to an unsupported type.", -56);
               return nullptr;
            }

            DB::Access<Mdl_function_call> function_call( tag, transaction);
            return int_expr_call_to_int_expr_direct_call(
                transaction, ef, type.get(), function_call.get_ptr(), parameters, context);
        }

        default:
            add_error_message(
               context, "The expression contains an unsupported kind of sub-expression.", -57);
            return nullptr;
    }
}

IExpression* int_expr_call_to_int_expr_direct_call(
    DB::Transaction* transaction,
    IExpression_factory* ef,
    const IType* type,
    const Mdl_function_call* call,
    const std::vector<mi::base::Handle<const IExpression>>& parameters,
    Execution_context* context)
{
    mi::base::Handle<const IExpression_list> arguments( call->get_arguments());
    mi::base::Handle<IExpression_list> converted_arguments(
        ef->create_expression_list());

    for( mi::Size i = 0, n = arguments->get_size(); i < n; ++i) {
        const char* parameter_name = arguments->get_name( i);
        mi::base::Handle<const IExpression> arg( arguments->get_expression( i));
        mi::base::Handle<IExpression> converted_arg(
            int_expr_call_to_int_expr_direct_call( transaction, ef, arg.get(), parameters, context));
        if( !converted_arg)
            return nullptr;
        converted_arguments->add_expression( parameter_name, converted_arg.get());
    }

    DB::Tag module_tag = call->get_module( transaction);
    DB::Tag definition_tag =  call->get_function_definition( transaction);
    if( !definition_tag) {
        add_error_message(
            context, "Failed to obtain definition for a call expression.", -58);
        return nullptr;
    }

    SERIAL::Class_id class_id = transaction->get_class_id( definition_tag);
    if( class_id != ID_MDL_FUNCTION_DEFINITION) {
        add_error_message(
            context, "The call expression refers to an unsupported type.", -56);
        return nullptr;
    }

    DB::Access<Mdl_function_definition> function_definition( definition_tag, transaction);
    Mdl_ident ident = function_definition->get_ident();
    std::string definition_db_name = transaction->tag_to_name( definition_tag);

    return ef->create_direct_call(
        type, module_tag, {definition_tag, ident}, definition_db_name, converted_arguments.get());
}

// ********** Misc utility functions around MI::MDL ************************************************

bool argument_type_matches_parameter_type(
    IType_factory* tf,
    const IType* argument_type,
    const IType* parameter_type,
    bool allow_cast,
    bool& needs_cast)
{
    // Equal types (without modifiers) succeed.
    mi::base::Handle<const IType> parameter_type_stripped(
        parameter_type->skip_all_type_aliases());
    mi::base::Handle<const IType> argument_type_stripped(
        argument_type->skip_all_type_aliases());
    if( allow_cast) {
        mi::Sint32 r = tf->is_compatible(
            argument_type_stripped.get(),
            parameter_type_stripped.get());
        if( r == 1) // identical
            return true;
        if( r == 0) { // compatible
            needs_cast = true;
            return true;
        }
    } else {
        if( tf->compare( argument_type_stripped.get(), parameter_type_stripped.get()) == 0)
            return true;
    }

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
        tf, argument_element_type.get(), parameter_element_type.get(), allow_cast, needs_cast);
}

bool return_type_is_varying( DB::Transaction* transaction, const IExpression* expr)
{
    switch( expr->get_kind()) {

        case IExpression::EK_CONSTANT:
            return false;

        case IExpression::EK_TEMPORARY:
            ASSERT( M_SCENE, !"temporaries are not supported");
            return false;

        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::Uint32 type_modifiers = type->get_all_type_modifiers();
            bool type_varying = (type_modifiers & IType::MK_VARYING) != 0;
            return type_varying;
        }

        case IExpression::EK_CALL: {

            // TODO type of expr, return_type, and return type of definition should agree
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            DB::Tag tag = expr_call->get_call();
            if( transaction->get_class_id( tag) != Mdl_function_call::id)
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

            DB::Tag definition_tag = fc->get_function_definition( transaction);
            if( transaction->get_class_id( definition_tag) != Mdl_function_definition::id)
                return false;
            DB::Access<Mdl_function_definition> definition( definition_tag, transaction);
            return !definition->is_uniform();
        }

        case IExpression::EK_DIRECT_CALL: {

            // TODO type of expr and return_type should agree
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag tag = expr_direct_call->get_definition( transaction);
            if( transaction->get_class_id( tag) != Mdl_function_definition::id)
                return false;

            DB::Access<Mdl_function_definition> definition( tag, transaction);
            mi::base::Handle<const IType> return_type( definition->get_return_type());
            mi::Uint32 return_type_modifiers = return_type->get_all_type_modifiers();
            bool return_type_uniform = (return_type_modifiers & IType::MK_UNIFORM) != 0;
            if( return_type_uniform)
                return false;
            bool return_type_varying = (return_type_modifiers & IType::MK_VARYING) != 0;
            if( return_type_varying)
                return true;
            return !definition->is_uniform();
        }

        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return false;
    }

    ASSERT( M_SCENE, false);
    return false;
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
            if( class_id != Mdl_function_call::id) {
                ASSERT( M_SCENE, false);
                return nullptr;
            }

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

            DB::Access<Mdl_function_definition> definition(
                original->get_function_definition( transaction), transaction);
            std::string copy_name_prefix = get_db_name(
                definition->get_mdl_name_without_parameter_types());
            std::string copy_name
                = DETAIL::generate_unique_db_name( transaction, copy_name_prefix.c_str());

            DB::Tag copy_tag = transaction->store_for_reference_counting(
                copy, copy_name.c_str(), transaction->get_scope()->get_level());
            mi::base::Handle<const IType> type( expr->get_type());
            return ef->create_call( type.get(), copy_tag);
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IExpression_parameter> expr_parameter(
                expr->get_interface<IExpression_parameter>());
            mi::Size index = expr_parameter->get_index();
            ASSERT( M_SCENE, index < context.size());
            return deep_copy( ef, transaction, context[index].get(), context);
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IType> type( expr->get_type());
            DB::Tag definition_tag = expr_direct_call->get_definition( transaction);
            DB::Tag module_tag = expr_direct_call->get_module();
            Mdl_ident ident = expr_direct_call->get_definition_ident();
            const char* definition_db_name = expr_direct_call->get_definition_db_name();
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());

            mi::base::Handle<IExpression_list> copy_arguments( ef->create_expression_list());
            mi::Size n = arguments->get_size();
            for( mi::Size i = 0; i < n; ++i) {
                mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
                mi::base::Handle<IExpression> copy_argument(
                    deep_copy( ef, transaction, argument.get(), context));
                const char* name = arguments->get_name( i);
                copy_arguments->add_expression( name, copy_argument.get());
            }

            return ef->create_direct_call(
                type.get(), module_tag, {definition_tag, ident}, definition_db_name, copy_arguments.get());
        }
        case IExpression::EK_TEMPORARY:
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

namespace {

/// Combines all bits of a size_t into a mi::Uint32.
mi::Uint32 hash_size_t_as_uint32( size_t hash)
{
    return static_cast<mi::Uint32>( (hash >> 32) + (hash & 0xffffffff));
}

} // namespace

mi::Uint32 get_hash( const std::string& mdl_file_path, const DB::Tag_version& tv)
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

mi::Uint32 get_hash( const char* mdl_file_path, const DB::Tag_version& tv)
{
    std::string s( mdl_file_path ? mdl_file_path : "");
    return get_hash( s, tv);
}

mi::Uint32 get_hash(
    const std::string& mdl_file_path, mi::Float32 gamma, const DB::Tag_version& tv1, const DB::Tag_version& tv2)
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
    const char* mdl_file_path, mi::Float32 gamma, const DB::Tag_version& tv1, const DB::Tag_version& tv2)
{
    std::string s( mdl_file_path ? mdl_file_path : "");
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

mi::mdl::IMDL::MDL_version get_min_required_mdl_version(
    DB::Transaction* transaction, const Mdl_function_definition* definition)
{
    mi::mdl::IMDL::MDL_version since, removed;
    definition->get_mdl_version( since, removed);
    return since;
}

mi::mdl::IMDL::MDL_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IExpression* expr)
{
    mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
    if( !expr)
        return version;

    switch( expr->get_kind()) {

        case IExpression::EK_CONSTANT:
        case IExpression::EK_PARAMETER:
            // smallest version is fine
            break;

        case IExpression::EK_CALL: {

            mi::base::Handle<const IExpression_call> call(
                expr->get_interface<IExpression_call>());
            DB::Tag call_tag = call->get_call();
            if( !call_tag)
                return mi_mdl_IMDL_MDL_VERSION_INVALID;

            SERIAL::Class_id class_id = transaction->get_class_id( call_tag);
            if( class_id != ID_MDL_FUNCTION_CALL) {
                ASSERT( M_SCENE, !"call to unknown entity class");
                return mi_mdl_IMDL_MDL_VERSION_INVALID;
            }

            DB::Access<Mdl_function_call> fcall( call_tag, transaction);
            DB::Tag def_tag = fcall->get_function_definition( transaction);
            if( !def_tag)
                return mi_mdl_IMDL_MDL_VERSION_INVALID;

            DB::Access<Mdl_function_definition> fdef( def_tag, transaction);
            version = get_min_required_mdl_version( transaction, fdef.get_ptr());
            mi::base::Handle<const IExpression_list> args( fcall->get_arguments());

            mi::mdl::IMDL::MDL_version v = get_min_required_mdl_version( transaction, args.get());
            if( v > version)
                version = v;
            break;
        }

        case IExpression::EK_DIRECT_CALL: {

           mi::base::Handle<const IExpression_direct_call> call(
               expr->get_interface<IExpression_direct_call>());
           DB::Tag def_tag = call->get_definition( transaction);
           if( !def_tag)
               return mi_mdl_IMDL_MDL_VERSION_INVALID;

           SERIAL::Class_id class_id = transaction->get_class_id( def_tag);
           if( class_id != ID_MDL_FUNCTION_DEFINITION) {
               ASSERT( M_SCENE, !"call to unknown entity class");
               return mi_mdl_IMDL_MDL_VERSION_INVALID;
           }

           DB::Access<Mdl_function_definition> fdef( def_tag, transaction);
           version = get_min_required_mdl_version( transaction, fdef.get_ptr());

           mi::base::Handle<const IExpression_list> args( call->get_arguments());
           mi::mdl::IMDL::MDL_version v = get_min_required_mdl_version( transaction, args.get());
           if( v > version)
               version = v;
           break;
        }

        case IExpression::EK_TEMPORARY:
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, !"Unsupported expression kind");
            return mi_mdl_IMDL_MDL_VERSION_INVALID;
            break;
    }

    return version;
}

mi::mdl::IMDL::MDL_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IExpression_list* expr_list)
{
    mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
    if( !expr_list)
        return version;

    for( mi::Size i = 0, n = expr_list->get_size(); i < n; ++i) {
        mi::base::Handle<const IExpression> expr( expr_list->get_expression( i));
        mi::mdl::IMDL::MDL_version v = get_min_required_mdl_version( transaction, expr.get());
        if( v > version)
            version = v;
    }

    return version;
}

mi::mdl::IMDL::MDL_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IAnnotation* annotation)
{
    mi::base::Handle<const IAnnotation_definition> definition(
        annotation->get_definition( transaction));
    mi::mdl::IMDL::MDL_version def_since, def_removed;
    definition->get_mdl_version( def_since, def_removed);

    mi::base::Handle<const IExpression_list> args( annotation->get_arguments());
    mi::mdl::IMDL::MDL_version args_since = get_min_required_mdl_version( transaction, args.get());

    return std::max( def_since, args_since);
}

mi::mdl::IMDL::MDL_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IAnnotation_block* block)
{
    mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
    if( !block)
        return version;


    for( mi::Size i = 0, n = block->get_size(); i < n; ++i) {
        mi::base::Handle<const IAnnotation> annotation( block->get_annotation( i));
        mi::mdl::IMDL::MDL_version v = get_min_required_mdl_version( transaction, annotation.get());
        if( v > version)
            version = v;
    }

    return version;
}

mi::mdl::IMDL::MDL_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IAnnotation_list* list)
{
    mi::mdl::IMDL::MDL_version version = mi::mdl::IMDL::MDL_VERSION_1_0;
    if( !list)
        return version;


    for( mi::Size i = 0, n = list->get_size(); i < n; ++i) {
        mi::base::Handle<const IAnnotation_block> block( list->get_annotation_block( i));
        mi::mdl::IMDL::MDL_version v = get_min_required_mdl_version( transaction, block.get());
        if( v > version)
            version = v;
    }

    return version;
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

mi::mdl::IQualified_name* signature_to_qualified_name(
    mi::mdl::IName_factory* nf, const char* signature, Name_mangler* name_mangler)
{
    mi::mdl::IQualified_name* qualified_name = nf->create_qualified_name();
    if( is_absolute( signature) && !is_in_module( signature, get_builtins_module_mdl_name()))
        qualified_name->set_absolute();

    std::string qual_module        = get_mdl_module_name( signature);
    std::string simple_definition  = get_mdl_simple_definition_name( signature);

    std::vector<std::string> package_components = get_mdl_package_component_names( qual_module);
    std::string simple_module                   = get_mdl_simple_module_name( qual_module);

    std::string s;
    const mi::mdl::ISymbol*      symbol      = nullptr;
    const mi::mdl::ISimple_name* simple_name = nullptr;

    for( const auto& pc: package_components) {
        s = name_mangler ? name_mangler->mangle( pc.c_str()) : pc;
        symbol = nf->create_symbol( s.c_str());
        simple_name = nf->create_simple_name( symbol);
        qualified_name->add_component( simple_name);
    }

    if( simple_module != get_builtins_module_simple_name()) {
        s = name_mangler ? name_mangler->mangle( simple_module.c_str()) : simple_module;
        symbol = nf->create_symbol( s.c_str());
        simple_name = nf->create_simple_name( symbol);
        qualified_name->add_component( simple_name);
    }

    symbol = nf->create_symbol( simple_definition.c_str());
    simple_name = nf->create_simple_name( symbol);
    qualified_name->add_component( simple_name);

    return qualified_name;
}

const mi::mdl::IExpression_reference* signature_to_reference(
    mi::mdl::IModule* module,
    const char* signature,
    Name_mangler* name_mangler)
{
    mi::mdl::IName_factory* nf = module->get_name_factory();
    mi::mdl::IQualified_name* qualified_name
        = signature_to_qualified_name( nf, signature, name_mangler);

    const mi::mdl::IType_name* type_name = nf->create_type_name( qualified_name);
    mi::mdl::IExpression_factory* ef = module->get_expression_factory();
    return ef->create_reference( type_name);
}

// TODO: unify with code in AST builder
mi::mdl::IType_name* type_to_type_name(
    mi::mdl::IModule* module, const mi::mdl::IType* type, Name_mangler* name_mangler)
{
    char buf[32];
    const char* s = nullptr;

    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case mi::mdl::IType::TK_ALIAS:
    case mi::mdl::IType::TK_AUTO:
    case mi::mdl::IType::TK_ERROR:
    case mi::mdl::IType::TK_FUNCTION:
        ASSERT(M_SCENE, !"unexpected MDL type kind");
        return nullptr;

    case mi::mdl::IType::TK_BOOL:
        s = "bool";
        break;
    case mi::mdl::IType::TK_INT:
        s = "int";
        break;
    case mi::mdl::IType::TK_ENUM:
        s = as<mi::mdl::IType_enum>(type)->get_symbol()->get_name();
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
    case mi::mdl::IType::TK_HAIR_BSDF:
        s = "hair_bsdf";
        break;
    case mi::mdl::IType::TK_EDF:
        s = "edf";
        break;
    case mi::mdl::IType::TK_VDF:
        s = "vdf";
        break;
    case mi::mdl::IType::TK_VECTOR:
        {
            const mi::mdl::IType_vector* v_type = as<mi::mdl::IType_vector>(type);
            const mi::mdl::IType_atomic* a_type = v_type->get_element_type();
            int size = v_type->get_size();

            switch (a_type->get_kind()) {
            case mi::mdl::IType::TK_BOOL:
                switch (size) {
                case 2: s = "bool2"; break;
                case 3: s = "bool3"; break;
                case 4: s = "bool4"; break;
                }
                break;
            case mi::mdl::IType::TK_INT:
                switch (size) {
                case 2: s = "int2"; break;
                case 3: s = "int3"; break;
                case 4: s = "int4"; break;
                }
                break;
            case mi::mdl::IType::TK_FLOAT:
                switch (size) {
                case 2: s = "float2"; break;
                case 3: s = "float3"; break;
                case 4: s = "float4"; break;
                }
                break;
            case mi::mdl::IType::TK_DOUBLE:
                switch (size) {
                case 2: s = "double2"; break;
                case 3: s = "double3"; break;
                case 4: s = "double4"; break;
                }
                break;
            default:
                ASSERT(M_SCENE, !"Unexpected type kind");
            }
        }
        break;
    case mi::mdl::IType::TK_MATRIX:
        {
            const mi::mdl::IType_matrix *m_type = as<mi::mdl::IType_matrix>(type);
            const mi::mdl::IType_vector *e_type = m_type->get_element_type();
            const mi::mdl::IType_atomic *a_type = e_type->get_element_type();

            snprintf(buf, sizeof(buf), "%s%dx%d",
                a_type->get_kind() == mi::mdl::IType::TK_FLOAT ? "float" : "double",
                m_type->get_columns(),
                e_type->get_size());
            buf[sizeof(buf) - 1] = '\0';
            s = buf;
        }
        break;
    case mi::mdl::IType::TK_ARRAY:
        {
            const mi::mdl::IType_array *a_type = as<mi::mdl::IType_array>(type);

            mi::mdl::IType_name* tn
                = type_to_type_name(module, a_type->get_element_type(), name_mangler);

            if (a_type->is_immediate_sized()) {
                int size = a_type->get_size();
                mi::mdl::IValue const *v = module->get_value_factory()->create_int(size);
                mi::mdl::IExpression const *lit =
                    module->get_expression_factory()->create_literal(v);
                tn->set_array_size(lit);
            } else {
                // we should not be here, but if, we create an incomplete array
                tn->set_incomplete_array();
            }
            return tn;
        }
    case mi::mdl::IType::TK_COLOR:
        s = "color";
        break;
    case mi::mdl::IType::TK_STRUCT:
        s = mi::mdl::as<mi::mdl::IType_struct>(type)->get_symbol()->get_name();
        break;
    case mi::mdl::IType::TK_TEXTURE:
        {
            mi::mdl::IType_texture const *t_type = as<mi::mdl::IType_texture>(type);

            switch (t_type->get_shape()) {
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
            case mi::mdl::IType_texture::TS_BSDF_DATA:
                ASSERT(M_SCENE, !"bsdf data textures cannot be expression in MDL source");
                break;
            }
        }
        break;
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        s = "bsdf_measurement";
        break;
    }
    ASSERT(M_SCENE, s);

    mi::mdl::IName_factory* nf = module->get_name_factory();
    mi::mdl::IQualified_name* qualified_name
        = signature_to_qualified_name( nf, s, name_mangler);

    return nf->create_type_name(qualified_name);
}

Resource_updater::Resource_updater(
    DB::Transaction* transaction,
    mi::mdl::ICall_name_resolver& resolver,
    mi::mdl::IGenerated_code_dag* code_dag,
    const char* module_filename,
    const char* module_mdl_name,
    bool callback_under_mutex)
  : m_transaction( transaction),
    m_resolver( resolver),
    m_code_dag( code_dag),
    m_module_filename( module_filename),
    m_module_mdl_name( module_mdl_name),
    m_callback_under_mutex( callback_under_mutex)
{
}

void Resource_updater::update_resource_literals()
{
    // materials
    mi::Size material_count = m_code_dag->get_material_count();
    for( mi::Size i = 0; i < material_count; ++i) {

        // traverse parameters
        mi::Size parameter_count = m_code_dag->get_material_parameter_count( i);
        for( mi::Size j = 0; j < parameter_count; ++j) {
            const mi::mdl::DAG_node* parameter_default
                = m_code_dag->get_material_parameter_default( i, j);
            update_resource_literals( parameter_default);
        }

        // traverse body
        const mi::mdl::DAG_node* body = m_code_dag->get_material_value( i);
        update_resource_literals( body);

        // traverse temporaries
        mi::Size temporary_count = m_code_dag->get_material_temporary_count( i);
        for( mi::Size j = 0; j < temporary_count; ++j) {
            const mi::mdl::DAG_node* temporary = m_code_dag->get_material_temporary( i, j);
            update_resource_literals( temporary);
        }
    }

    // functions
    mi::Size function_count = m_code_dag->get_function_count();

    for( mi::Size i = 0; i < function_count; ++i) {

        // traverse parameters
        mi::Size parameter_count = m_code_dag->get_function_parameter_count( i);
        for( mi::Size j = 0; j < parameter_count; ++j) {
            const mi::mdl::DAG_node* parameter_default
                = m_code_dag->get_function_parameter_default( i, j);
            update_resource_literals( parameter_default);
        }
    }
}

void Resource_updater::update_resource_literals( const mi::mdl::DAG_node* node)
{
    if( !node)
        return;

    switch( node->get_kind()) {

        case mi::mdl::DAG_node::EK_CONSTANT: {
            const mi::mdl::DAG_constant* constant = cast<mi::mdl::DAG_constant>( node);
            const mi::mdl::IValue_resource* resource
                = as<mi::mdl::IValue_resource>( constant->get_value());
            update_resource_literals( resource);
            return;
        }

        case mi::mdl::DAG_node::EK_PARAMETER:
            return; //-V1037 PVS

        case mi::mdl::DAG_node::EK_TEMPORARY:
            return; //-V1037 PVS the referenced temporary will be traversed explicitly

        case mi::mdl::DAG_node::EK_CALL: {
            const mi::mdl::DAG_call* call = cast<mi::mdl::DAG_call>( node);

            mi::Uint32 n = call->get_argument_count();
            for( mi::Uint32 i = 0; i < n; ++i)
                update_resource_literals( call->get_argument( i));

            if( call->get_semantic() != mi::mdl::IDefinition::DS_UNKNOWN)
                return;

            const char* signature = nullptr;
            mi::base::Handle<const mi::mdl::IModule> owner;

            {
                std::unique_lock<std::mutex> lock( DETAIL::g_transaction_mutex);
                signature = call->get_name();
                owner = m_resolver.get_owner_module( signature);
                if( !owner)
                    return;
            }

            const mi::mdl::Module* owner_impl = mi::mdl::impl_cast<mi::mdl::Module>( owner.get());
            const mi::mdl::IDefinition* def
                = owner_impl->find_signature( signature, /*only_exported*/ false);
            update_resource_literals( owner_impl, def);
            return;
        }
    }
}

void Resource_updater::update_resource_literals(
    const mi::mdl::IModule* owner, const mi::mdl::IDefinition* def)
{
    if( !def)
        return;

    const mi::mdl::IDeclaration* decl = def->get_declaration();
    if( !decl)
        return;

    if( !m_definition_set.insert( def).second)
        return;

    mi::mdl::Store<const mi::mdl::IModule*> store( m_def_owner, owner);
    visit( decl);
}

void Resource_updater::update_resource_literals( const mi::mdl::IDefinition* def)
{
    if( !def)
        return;

    if( !def->get_declaration())
        return;

    if( !m_definition_set.insert( def).second)
        return;

    def = m_def_owner->get_original_definition( def);
    if( !m_definition_set.insert( def).second)
        return;

    const mi::mdl::IDeclaration* decl = def->get_declaration();
    if( !decl)
        return;

    mi::base::Handle<const mi::mdl::IModule> owner( m_def_owner->get_owner_module( def));

    mi::mdl::Store<const mi::mdl::IModule*> store( m_def_owner, owner.get());
    visit( decl);
}

void Resource_updater::update_resource_literals( const mi::mdl::IValue_resource* resource)
{
    if( !resource)
        return;

    auto it = m_resource_tag_map.find( resource);
    if( it == m_resource_tag_map.end()) {
        DB::Tag tag = DETAIL::mdl_resource_to_tag(
            m_transaction, resource, m_module_filename, m_module_mdl_name, m_callback_under_mutex);
        it = m_resource_tag_map.insert( Resource_tag_map::value_type( resource, tag)).first;
    }

    DB::Tag tag = it->second;
    if( !tag)
        return;

    m_code_dag->set_resource_tag( resource, tag.get_uint());
}

mi::mdl::IExpression* Resource_updater::post_visit( mi::mdl::IExpression_literal* expr)
{
    const mi::mdl::IValue_resource* resource = as<mi::mdl::IValue_resource>( expr->get_value());
    update_resource_literals( resource);
    return expr;
}

mi::mdl::IExpression* Resource_updater::post_visit( mi::mdl::IExpression_call* expr)
{
    const mi::mdl::IExpression_reference* ref
        = as<mi::mdl::IExpression_reference>( expr->get_reference());
    if( !ref)
        return expr;

    const mi::mdl::IDefinition* def = ref->get_definition();
    if( !def)
        return expr;

    update_resource_literals( def);
    return expr;
}

namespace {

/// Collects all call references in a DAG node.
///
/// Parameter and temporary references are skipped.
///
/// \param transaction       The DB transaction to use (needed to convert strings into tags).
/// \param node              The DAG node to traverse.
/// \param[out] references   The collected references are added to this set.
void collect_material_references(
    DB::Transaction* transaction,
    const mi::mdl::DAG_node* node,
    DB::Tag_set& references)
{
    switch( node->get_kind()) {
        case mi::mdl::DAG_node::EK_CONSTANT:
            return; //-V1037 PVS nothing to do
        case mi::mdl::DAG_node::EK_PARAMETER:
            return; //-V1037 PVS  the referenced parameter will be traversed explicitly
        case mi::mdl::DAG_node::EK_TEMPORARY:
            return; //-V1037 PVS  the referenced temporary will be traversed explicitly
        case mi::mdl::DAG_node::EK_CALL: {
            const mi::mdl::DAG_call* call = as<mi::mdl::DAG_call>( node);
            std::string db_name = get_db_name( call->get_name());
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
    mi::Size material_index,
    DB::Tag_set& references)
{
    // traverse body
    const mi::mdl::DAG_node* body = code_dag->get_material_value( material_index);
    collect_material_references( transaction, body, references);

    // traverse temporaries
    mi::Size n = code_dag->get_material_temporary_count( material_index);
    for( mi::Size i = 0; i < n; ++i) {
        const mi::mdl::DAG_node* temporary = code_dag->get_material_temporary( material_index, i);
        collect_material_references( transaction, temporary, references);
    }
}

void collect_function_references(
    DB::Transaction* transaction,
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Size function_index,
    DB::Tag_set& references)
{
    mi::Size n = code_dag->get_function_references_count( function_index);
    for( mi::Size i = 0; i < n; ++i) {
        const char* reference = code_dag->get_function_reference( function_index, i);
        std::string db_name = get_db_name( reference);
        DB::Tag tag = transaction->name_to_tag( db_name.c_str());
        ASSERT( M_SCENE, tag);
        references.insert( tag);
    }
}

namespace {

/// Converts a given core message to an execution context message.
void convert_message( const Message& message, Execution_context* context)
{
    switch( message.m_severity) {
        case mi::base::MESSAGE_SEVERITY_ERROR:
            context->add_error_message( message);
            // fallthrough
        case mi::base::MESSAGE_SEVERITY_WARNING:
        case mi::base::MESSAGE_SEVERITY_INFO:
            context->add_message( message);
            return;
        default:
            break;
    }
}

mi::base::Message_severity convert_severity( mi::mdl::IMessage::Severity severity)
{
    switch( severity) {
        case mi::mdl::IMessage::MS_INFO:
            return mi::base::MESSAGE_SEVERITY_INFO;
        case mi::mdl::IMessage::MS_WARNING:
            return mi::base::MESSAGE_SEVERITY_WARNING;
        case mi::mdl::IMessage::MS_ERROR:
            return mi::base::MESSAGE_SEVERITY_ERROR;
        default:
            break;
    }

    return mi::base::MESSAGE_SEVERITY_ERROR;
}

mi::mdl::IMessage::Severity convert_severity( mi::base::Message_severity severity)
{
    switch( severity) {
        case mi::base::MESSAGE_SEVERITY_INFO:
            return mi::mdl::IMessage::MS_INFO;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            return mi::mdl::IMessage::MS_WARNING;
        case mi::base::MESSAGE_SEVERITY_ERROR:
            return mi::mdl::IMessage::MS_ERROR;
        default:
            break;
    }

    return mi::mdl::IMessage::MS_ERROR;
}

} // namespace

void convert_messages( const mi::mdl::Messages& in_messages, Execution_context* context)
{
    mi::Size message_count = in_messages.get_message_count();
    for( mi::Size i = 0; i < message_count; i++) {

        Message message( in_messages.get_message( i));
        convert_message( message, context);

        mi::Size note_count = message.m_notes.size();
        for( mi::Size j = 0; j < note_count; j++)
            convert_message( message.m_notes[j], context);
    }
}

void log_messages( const Execution_context* context)
{
    for( mi::Size i = 0, n = context->get_messages_count(); i < n; ++i) {

        const Message& message = context->get_message( i);

        switch( message.m_severity) {
            case mi::base::MESSAGE_SEVERITY_ERROR:
            LOG::mod_log->error(
                M_MDLC, LOG::Mod_log::C_COMPILER, "%s", message.m_message.c_str());
            break;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            LOG::mod_log->warning(
                M_MDLC, LOG::Mod_log::C_COMPILER, "%s", message.m_message.c_str());
            break;
        case mi::base::MESSAGE_SEVERITY_INFO:
            LOG::mod_log->info(
                M_MDLC, LOG::Mod_log::C_COMPILER, "%s", message.m_message.c_str());
            break;
        default:
            break;
        }
    }
}

void convert_and_log_messages( const mi::mdl::Messages& in_messages, Execution_context* context)
{
    Execution_context dummy_execution_context;
    if( !context)
        context = &dummy_execution_context;

    convert_messages( in_messages, context);
    log_messages( context);
}

void convert_messages( const Execution_context* context, mi::mdl::Messages& out_messages)
{
    mi::Size message_count = context->get_messages_count();
    for( mi::Size i = 0; i < message_count; ++i) {

        const Message& msg = context->get_message( i);

        mi::mdl::IMessage::Severity severity = convert_severity( msg.m_severity);
        mi::Sint32 code = msg.m_code;
        const char* str = msg.m_message.c_str();
        int msg_index = out_messages.add_message( severity, code, /*class*/ 'x', str, /*file*/ "",
            /*start_line*/ 0, /*start_column*/ 0, /*end_line*/ 0, /*end_column*/ 0);

        mi::Size note_count = msg.m_notes.size();
        for( mi::Size j = 0; j < note_count; ++j) {

            const Message& note = msg.m_notes[j];

            mi::mdl::IMessage::Severity severity = convert_severity( note.m_severity);
            mi::Sint32 code = note.m_code;
            const char* str = note.m_message.c_str();
            out_messages.add_note( msg_index, severity, code, /*class*/ 'x', str, /*file*/ "",
                /*start_line*/ 0, /*start_column*/ 0, /*end_line*/ 0, /*end_column*/ 0);
        }
    }
}

mi::Sint32 add_message( Execution_context* context, const Message& message, mi::Sint32 result)
{
    if( !context)
        return result;

    const mi::base::Message_severity& severity = message.m_severity;
    if( severity == mi::base::MESSAGE_SEVERITY_ERROR || severity == mi::base::MESSAGE_SEVERITY_FATAL) {
        context->add_error_message( message);
        context->set_result( result);
    }
    context->add_message( message);
    return result;
}

mi::Sint32 add_error_message(
    Execution_context* context, const std::string& message, mi::Sint32 result)
{
    Message msg( mi::base::MESSAGE_SEVERITY_ERROR, message, -1, Message::MSG_INTEGRATION);
    return add_message( context, msg, result);
}

void add_warning_message(
    MDL::Execution_context* context, const std::string& message)
{
    Message msg( mi::base::MESSAGE_SEVERITY_WARNING, message, -1, Message::MSG_INTEGRATION);
    add_message( context, msg, /*result, unused*/ 0);
}

void add_info_message(
    MDL::Execution_context* context, const std::string& message)
{
    Message msg( mi::base::MESSAGE_SEVERITY_INFO, message, -1, Message::MSG_INTEGRATION);
    add_message( context, msg, /*result, unused*/ 0);
}

mi::neuraylib::IReader* get_reader( mi::mdl::IInput_stream* stream)
{
    return new DETAIL::Input_stream_reader_impl( stream);
}

mi::neuraylib::IReader* get_reader( mi::mdl::IMDL_resource_reader* resource_reader)
{
    return new DETAIL::Resource_reader_impl( resource_reader);
}

mi::mdl::IInput_stream* get_input_stream(
    mi::neuraylib::IReader* reader, const std::string& filename)
{
     return new DETAIL::Input_stream_impl( reader, filename);
}

mi::mdl::IMdle_input_stream* get_mdle_input_stream(
    mi::neuraylib::IReader* reader, const std::string& filename)
{
     return new DETAIL::Mdle_input_stream_impl( reader, filename);
}

mi::mdl::IMDL_resource_reader* get_resource_reader(
    mi::neuraylib::IReader* reader,
    const std::string& file_path,
    const std::string& filename,
    const mi::base::Uuid& hash)
{
    return new DETAIL::Mdl_resource_reader_impl( reader, file_path, filename, hash);
}

mi::neuraylib::IReader* get_container_resource_reader(
    const std::string& resolved_container_filename,
    const std::string& container_member_name)
{
    mi::base::Handle<mi::mdl::IInput_stream> input_stream;

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());

    if (is_archive_filename(resolved_container_filename)) {
        mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());
        input_stream = archive_tool->get_file_content(
            resolved_container_filename.c_str(), container_member_name.c_str());
    } else if (is_mdle_filename(resolved_container_filename)) {
        mi::base::Handle<mi::mdl::IEncapsulate_tool> encapsulator(mdl->create_encapsulate_tool());
        input_stream = encapsulator->get_file_content(
            resolved_container_filename.c_str(), container_member_name.c_str());
    }

    if (!input_stream)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        input_stream->get_interface<mi::mdl::IMDL_resource_reader>());
    return get_reader(reader.get());
}

IMAGE::IMdl_container_callback* create_mdl_container_callback()
{
    return new DETAIL::Mdl_container_callback();
}

namespace {

/// Checks, if the given character is a valid MDL letter.
bool is_mdl_letter(char c)
{
    if ('A' <= c && c <= 'Z')
        return true;
    if ('a' <= c && c <= 'z')
        return true;
    return false;
}

/// Checks, if the given character is a valid MDL digit.
bool is_mdl_digit(char c)
{
    if ('0' <= c && c <= '9')
        return true;
    return false;
}

} // end namespace

Name_mangler::Name_mangler( mi::mdl::IMDL* mdl, mi::mdl::IModule* module)
  : m_mdl( mdl, mi::base::DUP_INTERFACE),
    m_module( module, mi::base::DUP_INTERFACE)
{
    for( int i = 0, n = m_module->get_declaration_count(); i < n; ++i) {

        const mi::mdl::IDeclaration* decl = m_module->get_declaration( i);
        if( decl->get_kind() == mi::mdl::IDeclaration::DK_IMPORT)
            continue;
        if( decl->get_kind() != mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS)
            break;

        const mi::mdl::IDeclaration_namespace_alias* decl_alias
            = cast<mi::mdl::IDeclaration_namespace_alias>( decl);
        std::string namespace_name = stringify( decl_alias->get_namespace());
        std::string alias_name     = decl_alias->get_alias()->get_symbol()->get_name();
        m_name_to_alias[namespace_name] = alias_name;
        m_aliases.insert( alias_name);
    }
}

const char* Name_mangler::mangle( const char* symbol)
{
    // Return mangled name of symbols that have been mangled before
    auto it = m_name_to_alias.find( symbol);
    if( it != m_name_to_alias.end())
        return it->second.c_str();

    // Return symbols that don't require mangling as is.
    if( is_mdle( symbol) || m_mdl->is_valid_mdl_identifier( symbol))
        return symbol;

    // Compute some alias name for symbol.
    size_t n = strlen( symbol);
    std::string alias;
    alias.reserve( n);

    // Replace invalid characters by underscore and compress sequence of underscores.
    for( size_t i = 0; i < n; ++i) {
        const char c = symbol[i];
        if( is_mdl_digit( c) || is_mdl_letter( c) || c == '_') {
            alias.push_back( c);
        } else {
            size_t l = alias.size();
            if( l == 0 || alias[l - 1] != '_')
                alias.push_back('_');
        }
    }

    // Add "m_" prefix if not a valid identifier, i.e., a keyword. Make unique.
    if( !m_mdl->is_valid_mdl_identifier( alias.c_str()))
        alias = "m_" + alias;
    alias = make_unique( alias);

    m_aliases.insert( alias);
    m_to_add.push_back( symbol);
    auto result = m_name_to_alias.insert( std::make_pair( symbol, alias));
    return result.first->second.c_str();
}

std::string Name_mangler::mangle_scoped_name( const std::string& name)
{
    std::string result;

    size_t start = 0;
    if( name.substr( 0, 2) == "::") {
        result += "::";
        start = 2;
    }
    size_t end   = name.find( "::", start);

    while( end != std::string::npos) {
        result += mangle( name.substr( start, end - start).c_str());
        result += "::";
        start = end + 2;
        end   = name.find( "::", start);
    }

    result += mangle( name.substr( start).c_str());
    return result;
}

void Name_mangler::add_namespace_aliases( mi::mdl::IModule* module)
{
    for( const auto& symbol_name : m_to_add) {
        const std::string& alias_name = m_name_to_alias[symbol_name];
        module->add_namespace_alias( alias_name.c_str(), symbol_name.c_str());
    }

    m_to_add.clear();
}

std::string Name_mangler::stringify( const mi::mdl::IQualified_name* name)
{
    std::string result;
    if( name->is_absolute())
        result += "::";
    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISimple_name* simple = name->get_component( i);
        if( i > 0)
            result += "::";
        result += simple->get_symbol()->get_name();
    }
    return result;
}

std::string Name_mangler::make_unique( const std::string& ident) const
{
    static int counter = 0;

    std::string test = ident;
    while( m_aliases.find( test) != m_aliases.end())
        test = ident + std::to_string( counter++);

    return test;
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
    void post_visit( mi::mdl::IType_name* name) final
    {
        const mi::mdl::IType* type = name->get_type();
        if( type && type->get_kind() == mi::mdl::IType::TK_ENUM) {
            // for enum values, the type must be imported.
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
    void post_visit( mi::mdl::IAnnotation* annotation) final
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
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_literal* literal) final
    {
        const mi::mdl::IValue* value = literal->get_value();
        handle_type( value->get_type());
        return literal;
    }

    /// Post-visits binary expressions.
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_binary* expr) final
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
        return expr;
    }

    /// Returns the found imports (core names).
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
  , m_module_core_name( module->get_name())
{
    for( int i = 0, n = module->get_declaration_count(); i < n; ++i) {
        const mi::mdl::IDeclaration* decl = module->get_declaration( i);
        switch( decl->get_kind()) {

            case mi::mdl::IDeclaration::DK_MODULE:
            case mi::mdl::IDeclaration::DK_NAMESPACE_ALIAS:
               continue;

            case  mi::mdl::IDeclaration::DK_IMPORT: {
                const mi::mdl::IDeclaration_import* import
                    = mi::mdl::cast<mi::mdl::IDeclaration_import>( decl);
                const mi::mdl::IQualified_name* module_name = import->get_module_name();
                if( module_name)
                    continue;
                for( int j = 0, n2 = import->get_name_count(); j < n2; ++j) {
                    const mi::mdl::IQualified_name* import_name = import->get_name( j);
                    const std::string& s = stringify( import_name);
                    m_existing_imports.insert( s);
                }
            }

            default:
                break;
        }
    }
}

Symbol_importer::~Symbol_importer()
{
    delete m_name_importer;
}

void Symbol_importer::collect_imports( const mi::mdl::IExpression* expr)
{
    m_name_importer->visit( expr);
}

void Symbol_importer::collect_imports( const mi::mdl::IType_name* tn)
{
    m_name_importer->visit( tn);
}

void Symbol_importer::collect_imports( const mi::mdl::IAnnotation_block* annotation_block)
{
    m_name_importer->visit( annotation_block);
}

void Symbol_importer::add_names( const std::set<std::string>& names)
{
    for( const auto& name: names)
        m_name_importer->insert( name);
}

void Symbol_importer::add_imports()
{
    using String_set = std::set<std::string>;

    mi::mdl::IModule* module = m_name_importer->get_module();
    const String_set& imports = m_name_importer->get_imports();
    for( String_set::const_iterator it = imports.begin(); it != imports.end(); ++it) {
        const std::string& import = *it;
        // do not import intensity_mode, this is a keyword in MDL 1.1
        if( import == "intensity_mode" || import == "::intensity_mode")
            continue;
        // do not import entities from the same module
        if( is_in_module( import, m_module_core_name))
            continue;
        if( m_existing_imports.insert( import).second)
            module->add_import( import.c_str());
    }
}

bool Symbol_importer::imports_mdle() const
{
    using String_set = std::set<std::string>;

    const String_set& imports = m_name_importer->get_imports();
    for( String_set::const_iterator it = imports.begin(); it != imports.end(); ++it)
        if( is_mdle( *it))
            return true;

    return false;
}

std::string Symbol_importer::stringify( const mi::mdl::IQualified_name* name)
{
    std::string result;
    if( name->is_absolute())
        result += "::";
    for( mi::Uint32 i = 0, n = name->get_component_count(); i < n; ++i) {
        const mi::mdl::ISimple_name* simple = name->get_component( i);
        if( i > 0)
            result += "::";
        result += simple->get_symbol()->get_name();
    }
    return result;
}

// **********  Mdl_module_wait_queue  **************************************************************

Mdl_module_wait_queue::Entry::Entry(
    const std::string& name,
    const Module_cache* cache,
    Mdl_module_wait_queue::Table* parent_table)
    : m_core_name(name)
    , m_cache_context_id(cache->get_loading_context_id())
    , m_handle(nullptr)
    , m_parent_table(parent_table)
    , m_usage_count(1 /* one for the creator */)
{
    mi::base::Handle<const mi::neuraylib::IMdl_loading_wait_handle_factory> factory(
        cache->get_wait_handle_factory());

    m_handle = factory->create_wait_handle();

    // printf_s(" created entry: %s on context: %d\n",
    //          m_core_name.c_str(), int(m_cache_context_id));
}

Mdl_module_wait_queue::Entry::~Entry()
{
    m_handle = nullptr;
}

// Called when the module is currently loaded by another threads.
mi::Sint32 Mdl_module_wait_queue::Entry::wait(const Module_cache* cache)
{
    assert(!processed_in_current_context(cache) &&
           "Wait must not be called from the creating thread.");

    m_handle->wait();

    // printf_s(" waited for entry: %s on context: %d\n",
    //          m_core_name.c_str(), int(cache->get_loading_context_id()));

    size_t res = m_handle->get_result_code();

    bool do_cleanup = false;
    {
        std::unique_lock<std::mutex> lck(m_usage_count_mutex);
        m_usage_count--;
        do_cleanup = (m_usage_count == 0);
    }
    if (do_cleanup)
        cleanup();

    return res;
}

// Called by the loading thread after loading is done to wake the waiting threads.
void Mdl_module_wait_queue::Entry::notify(
    const Module_cache* cache,
    mi::Sint32 result_code)
{
    assert(processed_in_current_context(cache) &&
           "Wait Entry notify can only be called from the creating thread.");

    m_handle->notify(result_code);

    // printf_s(" notified entry: %s on context: %d\n",
    //          m_core_name.c_str(), int(cache->get_loading_context_id()));

    // also cleanup because it is possible that no one waited at all
    bool do_cleanup = false;
    {
        std::unique_lock<std::mutex> lck(m_usage_count_mutex);
        m_usage_count--;
        do_cleanup = (m_usage_count == 0);
    }
    if (do_cleanup)
        cleanup();
}

// Check if this module is loaded by the current thread.
bool Mdl_module_wait_queue::Entry::processed_in_current_context(const Module_cache* cache) const
{
    return m_cache_context_id == cache->get_loading_context_id();
}

// Increments the usage counter of the entry.
void Mdl_module_wait_queue::Entry::increment_usage_count()
{
    std::unique_lock<std::mutex> lck(m_usage_count_mutex);
    m_usage_count++;
}

// Erases this entry from the parent table and self-destructs.
void Mdl_module_wait_queue::Entry::cleanup()
{
    m_parent_table->erase(m_core_name);

    // printf_s(" delete entry: %s\n", m_core_name.c_str());

    delete this;
}

//---------------------------------------------------------------------------------------------

Mdl_module_wait_queue::Table::Table()
    : m_mutex()
    , m_elements()
{
}

// Removes an entry from the table.
void Mdl_module_wait_queue::Table::erase(const std::string& name)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_elements.erase(name);
}

// Get or create a waiting entry for a module to load.
Mdl_module_wait_queue::Entry* Mdl_module_wait_queue::Table::get_waiting_entry(
    const Module_cache* cache,
    const std::string& name,
    bool& out_created)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    // check if the table contains an entry or create one
    auto found_entry = m_elements.find(name);
    if (found_entry != m_elements.end())
    {
        out_created = false;
        return found_entry->second;
    }

    out_created = true;
    Entry* entry = new Entry(name, cache, this);
    m_elements[name] = entry;
    return entry;
}

// Get the number of entries in the table.
mi::Size Mdl_module_wait_queue::Table::size() const
{
    return m_elements.size();
}

// Check if this module is loaded by the current thread.
bool Mdl_module_wait_queue::Table::processed_in_current_context(
    const Module_cache* cache,
    const std::string& module_name)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    auto found_entry = m_elements.find(module_name);
    if (found_entry == m_elements.end())
        return false;

    return found_entry->second->processed_in_current_context(cache);
}

//---------------------------------------------------------------------------------------------

// Wraps the cache lookup_db and creates a waiting entry for a module to load.
Mdl_module_wait_queue::Queue_lockup Mdl_module_wait_queue::lookup(
    const Module_cache* cache,
    size_t transaction,
    const std::string& name)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    Queue_lockup result{nullptr, nullptr};

    // check if the module is already in the cache
    result.cached_module = cache->lookup_db(name.c_str());
    if (result.cached_module)
        return result;

    // create the table for the current transaction if it is not existing yet
    Table* table = nullptr;
    auto found = m_tables.find(transaction);
    if (found == m_tables.end())
    {
        table = new Table();
        m_tables[transaction] = table;
    }
    else
    {
        table = found->second;
    }

    // create an entry in the table
    bool created_entry = false;
    result.queue_entry = table->get_waiting_entry(cache, name, created_entry);

    // the first thread that requests this module is responsible for loading (must not wait)
    // return the same result when the compiler requests the module again during loading
    if (created_entry || result.queue_entry->processed_in_current_context(cache))
        result.queue_entry = nullptr;

    if (result.queue_entry != nullptr)
        result.queue_entry->increment_usage_count();

    return result;
}

// Check if this module is loaded by the current thread.
bool Mdl_module_wait_queue::processed_in_current_context(
    const Module_cache* cache,
    size_t transaction,
    const std::string& module_name)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    auto found_table = m_tables.find(transaction);
    if (found_table == m_tables.end())
        return false;

    return found_table->second->processed_in_current_context(cache, module_name);
}

// Notify waiting threads about the finishing of the loading process of module.
void Mdl_module_wait_queue::notify(
    Module_cache* cache,
    size_t transaction,
    const std::string& module_name,
    int result_code)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    // get the table for the current transaction
    auto found = m_tables.find(transaction);
    assert(found != m_tables.end() && "Transaction table not found in wait queue.");

    // get the entry for the module
    bool created = false;
    Mdl_module_wait_queue::Entry* entry = found->second->get_waiting_entry(
        cache, module_name, created);
    assert(!created && "The entry should have been there before, created by this thread.");

    // wake all waiting threads
    entry->notify(cache, result_code);
}

// Try free this table when the transaction is not used anymore
void Mdl_module_wait_queue::cleanup_table(size_t transaction)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    // look for the table to clean up. we have been working with this before
    // note, size() is save here, since the m_mutex above is required to add elements
    auto found_table = m_tables.find(transaction);
    if (found_table != m_tables.end() && found_table->second->size() == 0)
    {
        // printf_s(" delete table: %d\n", static_cast<int>(transaction));
        delete found_table->second;
        m_tables.erase(found_table);
    }
}
// ********** Module_cache_lookup_handle***********************************************************

Module_cache_lookup_handle::Module_cache_lookup_handle()
    : m_lookup_name("")
    , m_is_processing(false)
{
}

void Module_cache_lookup_handle::set_lookup_name(const char* name)
{
    m_lookup_name = name;
}

const char* Module_cache_lookup_handle::get_lookup_name() const
{
    return m_lookup_name.empty() ? nullptr : m_lookup_name.c_str();
}

bool Module_cache_lookup_handle::is_processing() const
{
    return m_is_processing;
}

void Module_cache_lookup_handle::set_is_processing(bool value)
{
    m_is_processing = value;
}

// ********** Module_cache ************************************************************************

Module_cache::Wait_handle::Wait_handle()
    : m_conditional()
    , m_conditional_mutex()
    , m_processed(false)
    , m_result_code(-10 /* some invalid code */)
{
}

// Called when the module is currently loaded by another threads.
void Module_cache::Wait_handle::wait()
{
    std::unique_lock<std::mutex> lck(m_conditional_mutex);
    while (!m_processed) m_conditional.wait(lck);
}

// Called by the loading thread after loading is done to wake the waiting threads.
void Module_cache::Wait_handle::notify(mi::Sint32 result_code)
{
    std::unique_lock<std::mutex> lck(m_conditional_mutex);
    m_processed = true;
    m_result_code = result_code;
    m_conditional.notify_all();
}

// Gets the result code that was passed to \c notify.
mi::Sint32 Module_cache::Wait_handle::get_result_code() const
{
    return m_result_code;
}

// Creates a module cache wait handle.
mi::neuraylib::IMdl_loading_wait_handle*
Module_cache::Wait_handle_factory::create_wait_handle() const
{
    return new Module_cache::Wait_handle();
}

std::atomic<size_t> Module_cache::s_context_counter(0);

Module_cache::Module_cache(
    DB::Transaction* transaction,
    Mdl_module_wait_queue* queue,
    const DB::Tag_set& module_ignore_list)
    : m_context_id(s_context_counter++)
    , m_transaction(transaction)
    , m_queue(queue)
    , m_module_load_callback(nullptr)
    , m_default_wait_handle_factory(new Wait_handle_factory())
    , m_user_wait_handle_factory(nullptr)
    , m_ignore_list(module_ignore_list)
{
}

Module_cache::~Module_cache()
{
    m_queue->cleanup_table(m_transaction->get_id().get_uint());
    m_default_wait_handle_factory = nullptr;
    m_user_wait_handle_factory = nullptr;
}

mi::mdl::IModule_cache_lookup_handle* Module_cache::create_lookup_handle() const
{
    return new Module_cache_lookup_handle();
}

void Module_cache::free_lookup_handle(mi::mdl::IModule_cache_lookup_handle* handle) const
{
    delete static_cast<Module_cache_lookup_handle*>(handle);
}

const mi::mdl::IModule* Module_cache::lookup(
    const char* module_name,
    mi::mdl::IModule_cache_lookup_handle *handle) const
{
    // check for entries in the ignore list
    std::string fixed_module_name = module_name;
    if (is_mdle(module_name))
        fixed_module_name = "::" + add_slash_in_front_of_drive_letter(module_name + 2);

    std::string mdl_name = encode_module_name(fixed_module_name);
    std::string db_name = get_db_name(mdl_name);
    DB::Tag tag = m_transaction->name_to_tag(db_name.c_str());
    if (m_ignore_list.find(tag) != m_ignore_list.end())
        return nullptr;

    // simple check for existence
    if (!handle)
        return lookup_db(tag);

    const mi::mdl::IModule* dep = nullptr;
    size_t transaction_id = m_transaction->get_id().get_uint();

    // get a look first without creating a waiting entry
    // if the module is not in the cache, get a wait entry to wait until the (other)
    // loading thread is finished. If the module is not currently loaded, this thread is
    // responsible for loading
    Mdl_module_wait_queue::Queue_lockup lookup =
        m_queue->lookup(this, transaction_id, module_name);

    // pass out information about the lookup to make sure reporting about success and failure
    // can use the same module_name
    Module_cache_lookup_handle* handle_internal =
        static_cast<Module_cache_lookup_handle*>(handle);
    handle_internal->set_lookup_name(module_name);

    // module is loaded already
    if (lookup.cached_module)
    {
        // printf_s("[info] fetched \"%s\" from cache\n", module_name);
        return lookup.cached_module;
    }

    // this thread is supposed to load the module, do not wait, start loading instead
    if (!lookup.queue_entry)
    {
        // printf_s("[info] loading module on this thread \"%s\"\n", module_name);
        handle_internal->set_lookup_name(module_name);
        handle_internal->set_is_processing(true);
        return nullptr;
    }

    // wait until the module is loaded
    // printf_s("[info] waiting for thread loading \"%s\"\n", module_name);
    mi::Sint32 result_code = lookup.queue_entry->wait(this);

    // loading thread reported success
    if (result_code >= 0)
    {
        //printf_s("[info] waited for thread loading \"%s\"\n", module_name);

        // the module definitions can not be edited, otherwise there has to be more global
        // mechanism to protect against concurrent changes, same for reloads
        dep = lookup_db(module_name);
        assert(dep && "Module should be in the DB, as the loading thread reported success.");
        return dep;
    }

    // loading failed on a different thread
    //printf_s("[error] loading \"%s\" failed on a different thread.\n", module_name);
    return nullptr;
}

const mi::mdl::IModule* Module_cache::lookup_db(const char* module_name) const
{
    std::string mdl_name = encode_module_name(module_name);
    std::string db_name = get_db_name(mdl_name);
    DB::Tag tag = m_transaction->name_to_tag(db_name.c_str());
    return lookup_db(tag);
}

const mi::mdl::IModule* Module_cache::lookup_db(DB::Tag tag) const
{
    if (!tag)
        return nullptr;
    if (m_transaction->get_class_id(tag) != Mdl_module::id)
        return nullptr;

    DB::Access<Mdl_module> module(tag, m_transaction);
    return module->get_mdl_module();
}

/// Check if this module is loaded by the current thread.
bool Module_cache::loading_process_started_in_current_context(const char* module_name) const
{
    size_t transaction_id = m_transaction->get_id().get_uint();
    return m_queue->processed_in_current_context(this, transaction_id, module_name);
}

/// Notify waiting threads about the finishing of the loading process of module.
void Module_cache::notify( const char* module_name, int result_code)
{
    size_t transaction_id = m_transaction->get_id().get_uint();
    m_queue->notify(this, transaction_id, module_name, result_code);
}

/// Get the module loading callback
mi::mdl::IModule_loaded_callback* Module_cache::get_module_loading_callback() const
{
    return m_module_load_callback;
}

/// Set the module loading callback.
void Module_cache::set_module_loading_callback(mi::mdl::IModule_loaded_callback* callback)
{
    m_module_load_callback = callback;
}


const mi::neuraylib::IMdl_loading_wait_handle_factory* Module_cache::get_wait_handle_factory() const
{
    const mi::neuraylib::IMdl_loading_wait_handle_factory* factory = m_user_wait_handle_factory
        ? m_user_wait_handle_factory.get()
        : m_default_wait_handle_factory.get();

    factory->retain();
    return factory;
}

/// Set the module cache wait handle factory.
void Module_cache::set_wait_handle_factory(const mi::neuraylib::IMdl_loading_wait_handle_factory* factory)
{
    m_user_wait_handle_factory = mi::base::make_handle_dup(factory);
}


// ********** Call_evaluator ***********************************************************************

// Check whether evaluate_intrinsic_function should be called for an unhandled
// intrinsic functions with the given semantic.
template<typename T>
bool Call_evaluator<T>::is_evaluate_intrinsic_function_enabled(
    mi::mdl::IDefinition::Semantics semantic) const
{
    if (!m_has_resource_attributes)
        return false;

    switch (semantic) {
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_POWER:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_WIDTH:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_WIDTH_OFFSET:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT_OFFSET:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH_OFFSET:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_FIRST_FRAME:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LAST_FRAME:
        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_GRID_TO_OBJECT_SPACE:
            return true;
        default:
            return false;
    }
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::evaluate_intrinsic_function(
    mi::mdl::IValue_factory* value_factory,
    mi::mdl::IDefinition::Semantics semantic,
    const mi::mdl::IValue* const arguments[],
    size_t n_arguments) const
{
    // Disable evaluation if no resources are available, i.e., resolve_resources is set to false.
    if( !m_has_resource_attributes)
        return value_factory->create_bad();

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
            // TODO take frame parameter into account
            ASSERT( M_SCENE, arguments && (n_arguments >= 1 || n_arguments <= 3));
            return fold_tex_width(
                value_factory,
                arguments[0],
                n_arguments >= 2 ? arguments[1] : nullptr);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT:
            // TODO take frame parameter into account
            ASSERT( M_SCENE, arguments && (n_arguments >= 1 || n_arguments <= 3));
            return fold_tex_height(
                value_factory,
                arguments[0],
                n_arguments >= 2 ? arguments[1] : nullptr);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH:
            // TODO take frame parameter into account
            ASSERT( M_SCENE, arguments && (n_arguments >= 1 || n_arguments <= 2));
            return fold_tex_depth( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_tex_texture_isvalid( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_WIDTH_OFFSET:
            ASSERT(M_SCENE, arguments && n_arguments == 2);
            return fold_tex_width_offset( value_factory, arguments[0], arguments[1]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_HEIGHT_OFFSET:
            ASSERT(M_SCENE, arguments && n_arguments == 2);
            return fold_tex_height_offset(value_factory, arguments[0], arguments[1]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_DEPTH_OFFSET:
            ASSERT(M_SCENE, arguments && n_arguments == 2);
            return fold_tex_depth_offset(value_factory, arguments[0], arguments[1]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_FIRST_FRAME:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_tex_first_frame( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_LAST_FRAME:
            ASSERT( M_SCENE, arguments && n_arguments == 1);
            return fold_tex_last_frame( value_factory, arguments[0]);

        case mi::mdl::IDefinition::DS_INTRINSIC_TEX_GRID_TO_OBJECT_SPACE:
            ASSERT(M_SCENE, arguments && n_arguments == 2);
            return fold_tex_grid_to_object_space(value_factory, arguments[0], arguments[1]);

        default:
            return value_factory->create_bad();
    }
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_df_light_profile_power(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_float( 0.0f);

    const mi::mdl::IValue_resource* res = as<mi::mdl::IValue_resource>( argument);
    DB::Tag tag( this->get_resource_tag( res));

    bool valid;
    float power, maximum;
    get_light_profile_attributes( this->m_transaction, tag, valid, power, maximum);
    return value_factory->create_float( power);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_df_light_profile_maximum(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_float( 0.0f);

    const mi::mdl::IValue_resource* res = as<mi::mdl::IValue_resource>( argument);
    DB::Tag tag( this->get_resource_tag( res));

    bool valid;
    float power, maximum;
    get_light_profile_attributes( this->m_transaction, tag, valid, power, maximum);
    return value_factory->create_float( maximum);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_df_light_profile_isvalid(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_bool( false);

    const mi::mdl::IValue_resource* res = as<mi::mdl::IValue_resource>( argument);
    DB::Tag tag( this->get_resource_tag( res));

    bool valid;
    float power, maximum;
    get_light_profile_attributes( this->m_transaction, tag, valid, power, maximum);
    return value_factory->create_bool( valid);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_df_bsdf_measurement_isvalid(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_bool( false);

    const mi::mdl::IValue_resource* res = as<mi::mdl::IValue_resource>( argument);
    DB::Tag tag( this->get_resource_tag( res));

    bool valid;
    get_bsdf_measurement_attributes( this->m_transaction, tag, valid);
    return value_factory->create_bool( valid);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_tex_width(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument,
    const mi::mdl::IValue* uvtile_arg) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_int( 0);

    const mi::mdl::IValue_texture* tex = as<mi::mdl::IValue_texture>( argument);
    if( tex->get_bsdf_data_kind() != mi::mdl::IValue_texture::BDK_NONE) {
        size_t rx = 0, ry = 0, rz = 0;
        mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data_resolution(
            tex->get_bsdf_data_kind(), rx, ry, rz);
        return value_factory->create_int( rx);
    }

    DB::Tag tag( this->get_resource_tag( tex));
    mi::Sint32 uvtile_x = 0;
    mi::Sint32 uvtile_y = 0;
    if( const mi::mdl::IValue_vector* uvtile_vector = cast<mi::mdl::IValue_vector>( uvtile_arg)) {
        const mi::mdl::IValue_int* x = cast<mi::mdl::IValue_int>( uvtile_vector->get_value( 0));
        const mi::mdl::IValue_int* y = cast<mi::mdl::IValue_int>( uvtile_vector->get_value( 1));
        uvtile_x = x ? x->get_value() : 0;
        uvtile_y = y ? y->get_value() : 1;
    }

    bool valid;
    int width, height, depth;
    get_texture_attributes(
        this->m_transaction, tag, uvtile_x, uvtile_y, valid, width, height, depth);
    return value_factory->create_int( width);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_tex_height(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument,
    const mi::mdl::IValue* uvtile_arg) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_int( 0);

    const mi::mdl::IValue_texture* tex = as<mi::mdl::IValue_texture>( argument);
    if( tex->get_bsdf_data_kind() != mi::mdl::IValue_texture::BDK_NONE) {
        size_t rx = 0, ry = 0, rz = 0;
        mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data_resolution(
            tex->get_bsdf_data_kind(), rx, ry, rz);
        return value_factory->create_int( ry);
    }

    DB::Tag tag( this->get_resource_tag( tex));
    mi::Sint32 uvtile_x = 0;
    mi::Sint32 uvtile_y = 0;
    if( const mi::mdl::IValue_vector* uvtile_vector = cast<mi::mdl::IValue_vector>( uvtile_arg)) {
        const mi::mdl::IValue_int* x = cast<mi::mdl::IValue_int>( uvtile_vector->get_value( 0));
        const mi::mdl::IValue_int* y = cast<mi::mdl::IValue_int>( uvtile_vector->get_value( 1));
        uvtile_x = x ? x->get_value() : 0;
        uvtile_y = y ? y->get_value() : 1;
    }

    bool valid;
    int width, height, depth;
    get_texture_attributes(
        this->m_transaction, tag, uvtile_x, uvtile_y, valid, width, height, depth);
    return value_factory->create_int( height);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_tex_depth(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_int( 0);

    const mi::mdl::IValue_texture* tex = as<mi::mdl::IValue_texture>( argument);
    if( tex->get_bsdf_data_kind() != mi::mdl::IValue_texture::BDK_NONE) {
        size_t rx = 0, ry = 0, rz = 0;
        mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data_resolution(
            tex->get_bsdf_data_kind(), rx, ry, rz);
        return value_factory->create_int( rz);
    }

    DB::Tag tag( this->get_resource_tag( tex));
    mi::Sint32 uvtile_x = 0;
    mi::Sint32 uvtile_y = 0;

    bool valid;
    int width, height, depth;
    get_texture_attributes(
        this->m_transaction, tag, uvtile_x, uvtile_y, valid, width, height, depth);
    return value_factory->create_int( height);
}

template<typename T>
const mi::mdl::IValue* Call_evaluator<T>::fold_tex_texture_isvalid(
    mi::mdl::IValue_factory* value_factory,
    const mi::mdl::IValue* argument) const
{
    if( is<mi::mdl::IValue_invalid_ref>( argument))
        return value_factory->create_bool( false);

    const mi::mdl::IValue_texture* tex = as<mi::mdl::IValue_texture>( argument);
    if( tex->get_bsdf_data_kind() != mi::mdl::IValue_texture::BDK_NONE) {
        // bsdf data is always valid
        return value_factory->create_bool( true);
    }

    DB::Tag tag( this->get_resource_tag( tex));
    mi::Sint32 uvtile_x = 0;
    mi::Sint32 uvtile_y = 0;

    bool valid;
    int width, height, depth;
    get_texture_attributes(
        this->m_transaction, tag, uvtile_x, uvtile_y, valid, width, height, depth);
    return value_factory->create_bool( valid);
}

template<typename T>
const mi::mdl::IValue *Call_evaluator<T>::fold_tex_width_offset(
    mi::mdl::IValue_factory *value_factory,
    const mi::mdl::IValue *tex,
    const mi::mdl::IValue *offset) const
{
    // FIXME: default value currently
    return value_factory->create_int(0);
}

template<typename T>
const mi::mdl::IValue *Call_evaluator<T>::fold_tex_height_offset(
    mi::mdl::IValue_factory *value_factory,
    const mi::mdl::IValue *tex,
    const mi::mdl::IValue *offset) const
{
    // FIXME: default value currently
    return value_factory->create_int(0);
}

template<typename T>
const mi::mdl::IValue *Call_evaluator<T>::fold_tex_depth_offset(
    mi::mdl::IValue_factory *value_factory,
    const mi::mdl::IValue *tex,
    const mi::mdl::IValue *offset) const
{
    // FIXME: default value currently
    return value_factory->create_int(0);
}

template<typename T>
const mi::mdl::IValue *Call_evaluator<T>::fold_tex_first_frame(
    mi::mdl::IValue_factory *value_factory,
    const mi::mdl::IValue *tex) const
{
    // FIXME: default value currently
    return value_factory->create_int(0);
}

template<typename T>
const mi::mdl::IValue *Call_evaluator<T>::fold_tex_last_frame(
    mi::mdl::IValue_factory *value_factory,
    const mi::mdl::IValue *tex) const
{
    // FIXME: default value currently
    return value_factory->create_int(0);
}

template<typename T>

const mi::mdl::IValue *Call_evaluator<T>::fold_tex_grid_to_object_space(
    mi::mdl::IValue_factory *value_factory,
    const mi::mdl::IValue *tex,
    const mi::mdl::IValue *offset) const
{
    // FIXME: default value currently
    mi::mdl::IType_factory *tp_factory = value_factory->get_type_factory();
    mi::mdl::IType_float const *f_tp = tp_factory->create_float();
    mi::mdl::IType_vector const *f4_tp = tp_factory->create_vector(f_tp, 4);
    mi::mdl::IType_matrix const *f4x4_tp = tp_factory->create_matrix(f4_tp, 4);

    mi::mdl::IValue const *zero = value_factory->create_zero(f4x4_tp);
    return zero;
}

// explicit instantiate the two necessary cases
template class Call_evaluator<mi::mdl::IGenerated_code_dag>;
template class Call_evaluator<mi::mdl::ILambda_function>;

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

    // add message number
    msg += message->get_class() + std::to_string(message->get_code()) + ' ';

    m_message = msg + m_message;

    for (size_t i = 0; i < message->get_note_count(); ++i)
        m_notes.emplace_back(message->get_note(i));
}

namespace {

bool validate_warning(const boost::any& value)
{
    if (value.type() != typeid(std::string))
        return false;
    const std::string& s = boost::any_cast<const std::string&>(value);
    const char *opt = s.c_str();
    for (;;) {
        if (opt[0] == 'e' && opt[1] == 'r' && opt[2] == 'r') {
            // "err" : all warnings are errors
            opt += 3;
        } else {
            // digit+ = (on|off|err)
            while (::isdigit(opt[0])) {
                ++opt;
            }
            if (opt[0] == '=') {
                ++opt;
            }
            if (opt[0] == 'o') {
                if (opt[1] == 'f' && opt[2] == 'f') {
                    opt += 3;
                } else if (opt[1] == 'n') {
                    opt += 2;
                }
            }
            if (opt[0] == 'e' && opt[1] == 'r' && opt[2] == 'r') {
                opt += 3;
            }
        }

        if (opt[0] != ',') {
            break;
        }
        ++opt;
    }
    return opt[0] == '\0';
}

bool validate_optimization_level(const boost::any& value)
{
    if (value.type() != typeid(mi::Sint32))
        return false;
    const mi::Sint32& s = boost::any_cast<const mi::Sint32&>(value);
    return s >= 0 && s <= 2;
}

bool validate_internal_space(const boost::any& value)
{
    if (value.type() != typeid(std::string))
        return false;
    const std::string& s = boost::any_cast<const std::string&>(value);
    if (s == "coordinate_object" ||
        s == "coordinate_world")
        return true;
    return false;
}

}

Execution_context::Execution_context() : m_result(0)
{
    add_option(Option(MDL_CTX_OPTION_WARNING, std::string(""), false,
        validate_warning));
    add_option(Option(MDL_CTX_OPTION_OPTIMIZATION_LEVEL, static_cast<mi::Sint32>( 2), false,
        validate_optimization_level));
    add_option(Option(MDL_CTX_OPTION_INTERNAL_SPACE, std::string("coordinate_world"), false,
        validate_internal_space));
    add_option(Option(MDL_CTX_OPTION_FOLD_METERS_PER_SCENE_UNIT, true, false));
    add_option(Option(MDL_CTX_OPTION_METERS_PER_SCENE_UNIT, 1.0f, false));
    add_option(Option(MDL_CTX_OPTION_WAVELENGTH_MIN, 380.f, false));
    add_option(Option(MDL_CTX_OPTION_WAVELENGTH_MAX, 780.f, false));
    add_option(Option(MDL_CTX_OPTION_INCLUDE_GEO_NORMAL, true, false));
    add_option(Option(MDL_CTX_OPTION_BUNDLE_RESOURCES, false, false));
    add_option(Option(MDL_CTX_OPTION_EXPERIMENTAL, false, false));
    add_option(Option(MDL_CTX_OPTION_RESOLVE_RESOURCES, true, false));
    add_option(Option(MDL_CTX_OPTION_FOLD_TERNARY_ON_DF, false, false));
    add_option(Option(MDL_CTX_OPTION_IGNORE_NOINLINE, false, false));
    add_option(Option(MDL_CTX_OPTION_REMOVE_DEAD_PARAMETERS, true, false));
    add_option(Option(MDL_CTX_OPTION_FOLD_ALL_BOOL_PARAMETERS, false, false));
    add_option(Option(MDL_CTX_OPTION_FOLD_ALL_ENUM_PARAMETERS, false, false));
    add_option(Option(MDL_CTX_OPTION_FOLD_PARAMETERS,
        mi::base::Handle<const mi::base::IInterface>(), true));
    add_option(Option(MDL_CTX_OPTION_FOLD_TRIVIAL_CUTOUT_OPACITY, false, false));
    add_option(Option(MDL_CTX_OPTION_FOLD_TRANSPARENT_LAYERS, false, false));
    add_option(Option(MDL_CTX_OPTION_SERIALIZE_CLASS_INSTANCE_DATA, true, false));
    add_option(Option(MDL_CTX_OPTION_LOADING_WAIT_HANDLE_FACTORY,
        mi::base::Handle<const mi::base::IInterface>(), true));
    add_option(Option(MDL_CTX_OPTION_DEPRECATED_REPLACE_EXISTING, false, false));
    add_option(Option(MDL_CTX_OPTION_TARGET_MATERIAL_MODEL_MODE, false, false));
    add_option(Option(MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS, false, false));
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
    m_messages.emplace_back(message);
}

void Execution_context::add_error_message(const mi::mdl::IMessage* message)
{
    m_error_messages.emplace_back(message);
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
    for(mi::Size i = 0; i < messages.get_message_count(); ++i)
        m_messages.emplace_back(messages.get_message(i));
    for(mi::Size i = 0; i < messages.get_error_message_count(); ++i)
        m_error_messages.emplace_back(messages.get_error_message(i));
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

mi::Sint32 Execution_context::get_option(const std::string& name, boost::any& value) const
{
    mi::Size index = get_option_index(name);
    if (index == static_cast<mi::Size>(-1))
        return -1;

    const Option& option = m_options[index];
    value = option.get_value();
    return 0;
}

mi::Sint32 Execution_context::set_option(const std::string& name, const boost::any& value)
{
    mi::Size index = get_option_index(name);
    if (index == static_cast<mi::Size>(-1))
        return -1;

    Option& option = m_options[index];
    if (option.is_interface()) {

        // check that the value is a handle
        try {
            mi::base::Handle<const mi::base::IInterface> is_value_handle(
                boost::any_cast<mi::base::Handle<const mi::base::IInterface>>(value));
        } catch( ...) {
            return -2;
        }

    } else {

        // check that the tyoe of value matches exactly
        const boost::any& old_value = option.get_value();
        if (old_value.type() != value.type())
            return -2;

    }

    if (!option.set_value(value))
        return -3;

    return 0;
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

mi::mdl::IThread_context* create_thread_context( mi::mdl::IMDL* mdl, Execution_context* context)
{
    mi::mdl::IThread_context* thread_context = mdl->create_thread_context();

    if( context) {
        mi::mdl::Options& options = thread_context->access_options();

        std::string warnings = context->get_option<std::string>( MDL_CTX_OPTION_WARNING);
        options.set_option( MDL_OPTION_WARN, warnings.c_str());

        mi::Sint32 optimization_level = context->get_option<mi::Sint32>( MDL_CTX_OPTION_OPTIMIZATION_LEVEL);
        std::ostringstream optimization_level_str;
        optimization_level_str << optimization_level;
        options.set_option( MDL_OPTION_OPT_LEVEL, optimization_level_str.str().c_str());

        bool resolve_resources = context->get_option<bool>( MDL_CTX_OPTION_RESOLVE_RESOURCES);
        options.set_option( MDL_OPTION_RESOLVE_RESOURCES, resolve_resources ? "true" : "false");

        bool experimental = context->get_option<bool>( MDL_CTX_OPTION_EXPERIMENTAL);
        options.set_option( MDL_OPTION_EXPERIMENTAL_FEATURES, experimental ? "true" : "false");

        bool keep_original_resource_file_paths
            = context->get_option<bool>( MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS);
        options.set_option( MDL_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS,
            keep_original_resource_file_paths ? "true" : "false");
    }

    return thread_context;
}

namespace {

/// Skips the cast expression by extracting and returning its argument.
/// Returns the original expression, if \p expr is not a cast.
const IExpression* skip_cast(
    DB::Transaction* transaction,
    const IExpression* expr,
    const std::string &param_name) {

    if (!expr) return nullptr;
    mi::base::Handle<const IExpression_call > expr_call(
        expr->get_interface<IExpression_call>());
    if (expr_call) {
        DB::Tag tag = expr_call->get_call();
        SERIAL::Class_id class_id = transaction->get_class_id(tag);
        if (class_id == Mdl_function_call::id) {
            DB::Access<Mdl_function_call> fcall(tag, transaction);
            DB::Tag def_tag = fcall->get_function_definition(transaction);
            if (!def_tag.is_valid())
                return nullptr;
            DB::Access<Mdl_function_definition> fdef(
                def_tag, transaction);
            if (fdef->get_semantic() == mi::neuraylib::IFunction_definition::DS_CAST) {
                if (param_name != "cast") {
                    mi::base::Handle<const IExpression_list> args(fcall->get_arguments());
                    return args->get_expression("cast");
                }
            }
        }
    }
    expr->retain();
    return expr;
}

} // end namespace

/// Find the expression a path is pointing on.
mi::base::Handle<const IExpression> find_path(
    DB::Transaction* transaction,
    const std::string& path,
    const mi::base::Handle<const IExpression_list>& args)
{
    size_t pos = path.find('.');
    std::string param(path.substr(0, pos));

    mi::base::Handle<const IExpression> expr(args->get_expression(param.c_str()));
    if (!expr.is_valid_interface())
        return mi::base::Handle<const IExpression>();

    for (; pos != std::string::npos;) {
        size_t p = path.find('.', pos + 1);

        std::string arg(path.substr(pos + 1, p != std::string::npos ? p - pos - 1 : p));
        pos = p;

        expr = skip_cast(transaction, expr.get(), arg);

        IExpression::Kind kind = expr->get_kind();
        if (kind == IExpression::EK_CALL) {
            const mi::base::Handle<const IExpression_call> call(
                expr->get_interface<IExpression_call>());

            DB::Tag tag = call->get_call();
            SERIAL::Class_id class_id = transaction->get_class_id(tag);

            if (class_id != Mdl_function_call::id)
                return mi::base::Handle<const IExpression>();

            DB::Access<Mdl_function_call> fcall(tag, transaction);
            mi::base::Handle<const IExpression_list> args(fcall->get_arguments());
            expr = mi::base::make_handle(args->get_expression(arg.c_str()));
        } else if (kind == IExpression::EK_DIRECT_CALL) {
            const mi::base::Handle<const IExpression_direct_call> call(
                expr->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IExpression_list> args(call->get_arguments());
            expr = mi::base::make_handle(args->get_expression(arg.c_str()));
        } else {
            return mi::base::Handle<const IExpression>();
        }

        if (!expr.is_valid_interface())
            return mi::base::Handle<const IExpression>();
    }
    return expr;
}

mi::base::Uuid convert_hash( const mi::mdl::DAG_hash& h)
{
    mi::base::Uuid result;
    result.m_id1 = (h[ 0] << 24) | (h[ 1] << 16) | (h[ 2] << 8) | h[ 3];
    result.m_id2 = (h[ 4] << 24) | (h[ 5] << 16) | (h[ 6] << 8) | h[ 7];
    result.m_id3 = (h[ 8] << 24) | (h[ 9] << 16) | (h[10] << 8) | h[11];
    result.m_id4 = (h[12] << 24) | (h[13] << 16) | (h[14] << 8) | h[15];
    return result;
}

mi::base::Uuid convert_hash( const unsigned char h[16])
{
    mi::base::Uuid result;
    result.m_id1 = (h[ 0] << 24) | (h[ 1] << 16) | (h[ 2] << 8) | h[ 3];
    result.m_id2 = (h[ 4] << 24) | (h[ 5] << 16) | (h[ 6] << 8) | h[ 7];
    result.m_id3 = (h[ 8] << 24) | (h[ 9] << 16) | (h[10] << 8) | h[11];
    result.m_id4 = (h[12] << 24) | (h[13] << 16) | (h[14] << 8) | h[15];
    return result;
}

bool convert_hash( const mi::base::Uuid& hash_in, unsigned char hash_out[16])
{
    if( hash_in == mi::base::Uuid{0,0,0,0})
        return false;

    for( int i = 0; i < 4; ++i)
        hash_out[ 0 + i] = (hash_in.m_id1 >> (24 - 8*i)) & 0xff;
    for( int i = 0; i < 4; ++i)
        hash_out[ 4 + i] = (hash_in.m_id2 >> (24 - 8*i)) & 0xff;
    for( int i = 0; i < 4; ++i)
        hash_out[ 8 + i] = (hash_in.m_id3 >> (24 - 8*i)) & 0xff;
    for( int i = 0; i < 4; ++i)
        hash_out[12 + i] = (hash_in.m_id4 >> (24 - 8*i)) & 0xff;
    return true;
}

mi::base::Uuid get_hash( mi::mdl::IMDL_resource_reader* reader)
{
    unsigned char h[16];
    bool valid_hash = reader->get_resource_hash( h);
    if( !valid_hash)
        return mi::base::Uuid{0,0,0,0};

    return convert_hash( h);
}

mi::base::Uuid get_hash(mi::mdl::IMDL_resource_set const *set)
{
    size_t count = set->get_count();
    size_t overall_hash = 0;

    for (size_t i = 0; i < count; ++i) {
        unsigned char mdl_hash[16];
        bool valid_hash = set->get_resource_hash(i, mdl_hash);
        if (!valid_hash)
            return mi::base::Uuid{ 0,0,0,0 };
        size_t hash = boost::hash_range(mdl_hash, mdl_hash + 16);
        boost::hash_combine(overall_hash, hash);
    }

    mi::Uint32 low  = overall_hash & 0xFFFFFFFFu;
    mi::Uint32 high = overall_hash >> 32u;
    return mi::base::Uuid{ 0, 0, high, low };
}

Uint64 generate_unique_id()
{
    boost::uuids::uuid u = boost::uuids::random_generator()();
    std::string name = boost::uuids::to_string(u);
    return std::hash<std::string>{}(name);
}

mi::Sint32 repair_call(
    DB::Transaction* transaction,
    const DB::Access<Mdl_function_call>& call_access,
    mi::base::Handle<IExpression>& new_expr,
    const IExpression *default_expr,
    const IExpression_call* arg_call,
    IExpression_factory* ef,
    IValue_factory* vf,
    bool repair_invalid_calls,
    bool remove_invalid_calls,
    mi::Uint32 level,
    Execution_context* context
)
{
    DB::Edit<Mdl_function_call> call_edit(call_access);
    mi::Sint32 res = call_edit->repair(
        transaction, repair_invalid_calls, remove_invalid_calls, level, context);
    if (res == 0)
        return 0;

    if (remove_invalid_calls) {

        if (default_expr) {
            new_expr =
                ef->clone(default_expr, transaction, /*copy_immutable_calls=*/true);
        }
        else { // create a value
            mi::base::Handle<const IType> arg_type(arg_call->get_type());
            mi::base::Handle<IValue> new_val(vf->create(arg_type.get()));
            new_expr = ef->create_constant(new_val.get());
        }
        return 0;
    }
    return -1;
}

mi::Sint32 repair_arguments(
    DB::Transaction* transaction,
    IExpression_list* arguments,
    const IExpression_list* defaults,
    bool repair_invalid_calls,
    bool remove_invalid_calls,
    mi::Uint32 level,
    Execution_context* context)
{
    mi::base::Handle<IExpression_factory> ef(get_expression_factory());
    mi::base::Handle<IValue_factory> vf(get_value_factory());

    for (mi::Size i = 0, n = arguments->get_size(); i < n; ++i) {

        mi::base::Handle<const IExpression_call> arg_call(
            arguments->get_expression<IExpression_call>(i));
        if (arg_call.is_valid_interface()) {
            DB::Tag call_tag = arg_call->get_call();
            if (!call_tag.is_valid())
                continue;

            SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
            if (class_id != ID_MDL_FUNCTION_CALL)
                continue;

            const char* arg_name = arguments->get_name(i);
            mi::base::Handle<const IExpression> default_expr(
                defaults->get_expression(defaults->get_index(arg_name)));

            mi::base::Handle<IExpression> new_expr;
            DB::Access<Mdl_function_call> fcall(call_tag, transaction);
            if (!fcall->is_valid(transaction, context)) {
                if (repair_call(
                    transaction,
                    fcall,
                    new_expr,
                    default_expr.get(),
                    arg_call.get(),
                    ef.get(),
                    vf.get(),
                    repair_invalid_calls,
                    remove_invalid_calls,
                    level,
                    context) != 0) {
                        add_context_error( context,
                            STRING::formatted_string(
                                "The call \"%s\" attached to argument \"%s\" could not be "
                                "repaired.", transaction->tag_to_name(call_tag), arg_name), -1);
                    return -1;
                }
            }
            if (new_expr)
                arguments->set_expression(i, new_expr.get());
        }
    }
    return 0;
}

mi::Sint32 add_context_error(
    MDL::Execution_context* context,
    const std::string& message,
    mi::Sint32 result)
{
    if (!context)
        return -1;
    MDL::Message err(mi::base::MESSAGE_SEVERITY_ERROR, message);
    context->add_error_message(err);
    context->add_message(err);
    context->set_result(result);

    return -1;
}

mi::neuraylib::Mdl_version convert_mdl_version( mi::mdl::IMDL::MDL_version version)
{
    if( version == mi_mdl_IMDL_MDL_VERSION_INVALID)
        return mi::neuraylib::MDL_VERSION_INVALID;

    switch( version) {
        case mi::mdl::IMDL::MDL_VERSION_1_0: return mi::neuraylib::MDL_VERSION_1_0;
        case mi::mdl::IMDL::MDL_VERSION_1_1: return mi::neuraylib::MDL_VERSION_1_1;
        case mi::mdl::IMDL::MDL_VERSION_1_2: return mi::neuraylib::MDL_VERSION_1_2;
        case mi::mdl::IMDL::MDL_VERSION_1_3: return mi::neuraylib::MDL_VERSION_1_3;
        case mi::mdl::IMDL::MDL_VERSION_1_4: return mi::neuraylib::MDL_VERSION_1_4;
        case mi::mdl::IMDL::MDL_VERSION_1_5: return mi::neuraylib::MDL_VERSION_1_5;
        case mi::mdl::IMDL::MDL_VERSION_1_6: return mi::neuraylib::MDL_VERSION_1_6;
        case mi::mdl::IMDL::MDL_VERSION_1_7: return mi::neuraylib::MDL_VERSION_1_7;
        // Adapt check in strip_deprecated_suffix() when new versions are added.
        case mi::mdl::IMDL::MDL_VERSION_1_8: return mi::neuraylib::MDL_VERSION_INVALID;
    }

    ASSERT( M_SCENE, false);
    return mi::neuraylib::MDL_VERSION_INVALID;
}

mi::mdl::IMDL::MDL_version convert_mdl_version( mi::neuraylib::Mdl_version version)
{
    switch( version) {
        case mi::neuraylib::MDL_VERSION_1_0:     return mi::mdl::IMDL::MDL_VERSION_1_0;
        case mi::neuraylib::MDL_VERSION_1_1:     return mi::mdl::IMDL::MDL_VERSION_1_1;
        case mi::neuraylib::MDL_VERSION_1_2:     return mi::mdl::IMDL::MDL_VERSION_1_2;
        case mi::neuraylib::MDL_VERSION_1_3:     return mi::mdl::IMDL::MDL_VERSION_1_3;
        case mi::neuraylib::MDL_VERSION_1_4:     return mi::mdl::IMDL::MDL_VERSION_1_4;
        case mi::neuraylib::MDL_VERSION_1_5:     return mi::mdl::IMDL::MDL_VERSION_1_5;
        case mi::neuraylib::MDL_VERSION_1_6:     return mi::mdl::IMDL::MDL_VERSION_1_6;
        case mi::neuraylib::MDL_VERSION_1_7:     return mi::mdl::IMDL::MDL_VERSION_1_7;
        case mi::neuraylib::MDL_VERSION_INVALID: return mi_mdl_IMDL_MDL_VERSION_INVALID;
    }

    ASSERT( M_SCENE, false);
    return mi_mdl_IMDL_MDL_VERSION_INVALID;
}

const char* stringify_mdl_version( mi::mdl::IMDL::MDL_version version)
{
    switch( version) {
        case mi::mdl::IMDL::MDL_VERSION_1_0: return "1.0";
        case mi::mdl::IMDL::MDL_VERSION_1_1: return "1.1";
        case mi::mdl::IMDL::MDL_VERSION_1_2: return "1.2";
        case mi::mdl::IMDL::MDL_VERSION_1_3: return "1.3";
        case mi::mdl::IMDL::MDL_VERSION_1_4: return "1.4";
        case mi::mdl::IMDL::MDL_VERSION_1_5: return "1.5";
        case mi::mdl::IMDL::MDL_VERSION_1_6: return "1.6";
        case mi::mdl::IMDL::MDL_VERSION_1_7: return "1.7";
        case mi::mdl::IMDL::MDL_VERSION_1_8: return "1.8";
    }

    ASSERT( M_SCENE, false);
    return "unknown";
}

// **********  Name parsing/splitting **************************************************************

bool is_valid_simple_package_or_module_name( const std::string& name)
{
    size_t n = name.size();
    if( n == 0)
        return false;

    for( size_t i = 0; i < n; ++i) {

        unsigned char c = name[i];
        // These characters are not permitted per MDL spec.
        if( c == '/' || c == '\\' || c < 32 || c == 127 || c == ':')
            return false;
    }

    return true;
}

bool is_valid_module_name( const std::string& name)
{
    if( name[0] != ':' || name[1] != ':')
        return false;

    size_t start = 2;
    size_t end   = name.find( "::", start);

    while( end != std::string::npos) {
        if( !is_valid_simple_package_or_module_name( name.substr( start, end - start)))
            return false;
        start = end + 2;
        end   = name.find( "::", start);
    }

    if( !is_valid_simple_package_or_module_name( name.substr( start)))
        return false;

    return true;
}

bool is_absolute( const std::string& name)
{
    return (name[0] == ':' && name[1] == ':' && name[2]) || is_mdle( name);
}

bool starts_with_scope( const std::string& name)
{
    return name[0] == ':' && name[1] == ':';
}

bool starts_with_slash( const std::string& name)
{
    return name[0] == '/';
}

bool starts_with_mdl_or_mdle( const std::string& name)
{
    return name.substr( 0, 5) == "mdl::" || name.substr( 0, 6) == "mdle::";
}

std::string strip_mdl_or_mdle_prefix( const std::string& name)
{
    if( name.substr( 0, 5) == "mdl::")
        return name.substr( 3);

    if( name.substr( 0, 6) == "mdle::") {

        if( get_encoded_names_enabled())
            return name.substr( 4);

        if( name.size() >= 10 && name[6] == '/' && name[8] == ':' && name[9] == '/')
            return "::" + name.substr( 7);
        return name.substr( 4);
    }

    return name;
}

bool is_mdle( const std::string& name)
{
    size_t n = name.size();
    if (n > 5 && name.substr( n-5, 5) == ".mdle")
        return true;

    return name.find( ".mdle::") != std::string::npos;
}

bool is_deprecated( const std::string& name)
{
    size_t dollar = name.rfind( '$');
    size_t scope  = name.rfind( "::");
    return (dollar != std::string::npos) && ((scope == std::string::npos) || (dollar > scope));
}

bool is_in_module( const std::string& name, const std::string& module_name)
{
    size_t offset;

    if( module_name == get_builtins_module_mdl_name()) {
        // initial "::" optional
        offset = name.substr( 0, 2) == "::" ? 2 : 0;
    } else {
        size_t l = module_name.length();
        if( name.substr( 0, l) != module_name || name.substr( l, 2) != "::")
            return false;
        offset = l+2;
    }

    size_t scope  = name.find( "::", offset);
    size_t left   = name.find( '(' , offset);

    if( scope == std::string::npos)
        return true;  // name is in module
    if( left == std::string::npos)
        return false; // name has no signature and is in submodule
    if( scope < left)
        return false; // name has signature and is in submodule

    return true;      // name is in module (scope appears only in signature)
}

std::string remove_qualifiers_if_from_module(
    const std::string& name, const std::string& module_name)
{
    size_t offset;

    if( module_name == get_builtins_module_mdl_name()) {
        // initial "::" optional
        offset = name.substr( 0, 2) == "::" ? 2 : 0;
    } else {
        size_t l = module_name.length();
        if( name.substr( 0, l) != module_name || name.substr( l, 2) != "::")
            return name;
        offset = l+2;
    }

    size_t scope  = name.find( "::", offset);
    size_t left   = name.find( '(' , offset);

    if( scope == std::string::npos)
        return name.substr( offset); // name is in module
    if( left == std::string::npos)
        return name;                 // name has no signature and is in submodule
    if( scope < left)
        return name;                 // name has signature and is in submodule

    return name.substr( offset);     // name is in module (scope appears only in signature)
}

std::string strip_deprecated_suffix( const std::string& name)
{
    size_t dollar = name.rfind( '$');
    size_t scope  = name.rfind( "::");

    if( dollar == std::string::npos)
        return name;
    if( (scope != std::string::npos) && (dollar < scope))
        return name;

    std::string suffix = name.substr( dollar+1);
    if( suffix.size() == 3
        && suffix[0] == '1'
        && suffix[1] == '.'
        && (suffix[2] >= '0' && suffix[2] <= '7'))
        return name.substr( 0, dollar);

    return name;
}

std::string strip_resource_owner_prefix( const std::string& name)
{
    size_t scope = name.rfind( "::");
    return scope != std::string::npos ? name.substr( scope + 2) : name;
}

std::string get_resource_owner_prefix( const std::string& name)
{
    size_t scope = name.rfind( "::");
    return scope != std::string::npos ? name.substr( 0, scope) : std::string();
}

std::string get_db_name( const std::string& name)
{
    // Prefix of DB elements for MDL modules/definitions.
    const char* mdl_db_prefix  = "mdl::";
    const char* mdle_db_prefix = "mdle::";

    std::string result;
    size_t start = 0;

    // Skip "::" if already part of name
    if( starts_with_scope( name))
        start += 2;

    if( is_mdle( name)) {

        result += mdle_db_prefix;
        result += '/';

        ASSERT( M_SCENE, start == 2);

        // If \p name is an MDLE module (or an entity from an MDLE module), then it might start
        // with a drive letter and we need to insert a slash here. This case should not occur
        // anymore.
        ASSERT( M_SCENE, starts_with_slash( &name[start]));

        if( starts_with_slash( &name[start]))
           start += 1;

    } else {

        // If there is no leading scope, then \p name is the builtins module (or an entity from the
        // builtins module). Check this by verifying that there is no scope at all (excluding the
        // signature). This check is only reliable with encoded names enabled.
        if( (start == 0) && get_encoded_names_enabled()) {
            size_t left_paren = name.find( '(');
            size_t scope      = name.find( "::");
            ASSERT( M_SCENE, scope == std::string::npos
                             || ((left_paren != std::string::npos) && (scope > left_paren)));
            boost::ignore_unused( left_paren);
            boost::ignore_unused( scope);
        }

        result += mdl_db_prefix;
    }

    result += &name[start];
    return result;
}

std::string prefix_builtin_type_name( const char* name)
{
    if( starts_with_scope( name))
        return name;

    ASSERT( M_SCENE, strcmp( name, "material_emission") == 0
                  || strcmp( name, "material_surface" ) == 0
                  || strcmp( name, "material_volume"  ) == 0
                  || strcmp( name, "material_geometry") == 0
                  || strcmp( name, "material"         ) == 0
                  || strcmp( name, "intensity_mode"   ) == 0);

    return std::string( "::") + name;
}

std::string get_mdl_simple_module_name( const std::string& name)
{
    // Precondition: starts with "::" or from <builtins> module or from MDLE

    size_t scope = name.rfind( "::");
    if( scope == std::string::npos)
        return name;

    return name.substr( scope + 2);
}

std::vector<std::string> get_mdl_package_component_names( const std::string& name)
{
    // Precondition: starts with "::" or from <builtins> module or from MDLE

    std::vector<std::string> result;

    size_t start = 2;
    size_t end   = name.find( "::", start);

    while( end != std::string::npos) {
        result.push_back( name.substr( start, end - start));
        start = end + 2;
        end   = name.find( "::", start);
    }

    return result;
}

std::string get_mdl_simple_definition_name( const std::string& name)
{
    // Precondition: starts with "::" or from <builtins> module or from MDLE

    size_t mdle = name.find( ".mdle::");
    if( mdle != std::string::npos)
        return name.substr( mdle + 7);

    // Assert: starts with "::" or from <builtins> module

    size_t scope = name.rfind( "::");
    if( scope == std::string::npos)
        return name;

    return name.substr( scope + 2);
}

std::string get_mdl_module_name( const std::string& name)
{
    // Precondition: starts with "::" or from <builtins> module or from MDLE

    size_t scope = name.rfind( "::");
    if( scope == 0 || scope == std::string::npos)
        return get_builtins_module_mdl_name();

    return name.substr( 0, scope);
}

std::string get_mdl_field_name( const std::string& name)
{
    // Precondition:    (a) starts with "::" or from <builtins> module or from MDLE
    //               or (b) is a simple MDL name
    //
    // Precondition: does not contain parameter types
    ASSERT( M_SCENE, !name.empty() && name.back() != ')');

    size_t dot  = name.rfind( '.');
    ASSERT( M_SCENE, dot > 0);
    return name.substr( dot + 1);
}

void split_next_dot_or_bracket( const char* s, std::string& head, std::string& tail)
{
    // find first dot
    const char* dot = strchr( s, '.');

    // find first bracket pair
    const char* left_bracket = strchr( s, '[');
    const char* right_bracket = left_bracket ? strchr( left_bracket, ']') : nullptr;
    if( left_bracket && !right_bracket)
        left_bracket = nullptr;

    // handle neither dot nor bracket pair
    if( !dot && !left_bracket) {
        head = s;
        tail = "";
        return;
    }

    // handle leading array index
    if( left_bracket && left_bracket == s) {
        ASSERT( M_SCENE, right_bracket);
        head = std::string( s+1, right_bracket-(s+1)); //-V769 PVS
        tail = std::string( right_bracket[1] == '.' ? right_bracket+2 : right_bracket+1);
        return;
    }

    // handle non-leading array index or field name
    const char* sep = dot ? dot : left_bracket;
    if( left_bracket && sep > left_bracket)
        sep = left_bracket;
    head = std::string( s, sep-s); //-V769 PVS
    tail = std::string( sep == left_bracket ? sep : sep+1);
}

std::string get_mdl_name_from_load_module_arg( const std::string& name, bool is_mdle)
{
    if( is_mdle) {

        // make path absolute
        std::string result = name;
        if( !DISK::is_path_absolute( result.c_str()))
            result = HAL::Ospath::join( DISK::get_cwd(), result);

        // normalize path to make it unique
        result = HAL::Ospath::normpath( result);

        // use forward slashes
        result = HAL::Ospath::convert_to_forward_slashes( result);

        // convert to MDL name
        result = add_slash_in_front_of_drive_letter( result);
        return encode_module_name( "::" + result);
    }

    std::string result;

    // Add leading "::" if missing. This is deprecated (with warning) if encoded names disabled, and
    // an error if enabled.
    if( !starts_with_scope( name)) {
        ASSERT( M_SCENE, !MDL::get_encoded_names_enabled());
        result += "::";
    }

    result += name;
    return result;
}

std::string get_db_name_annotation_definition( const std::string& name)
{
    return "mdla" + name;
}

bool is_absolute_mdl_file_path( const std::string& name)
{
    return name[0] == '/';
}

bool has_mdl_suffix(const std::string& filename)
{
    size_t n = filename.size();
    return n >= 4 && filename.substr(n - 4) == ".mdl";
}

std::string strip_dot_mdl_suffix( const std::string& filename)
{
    size_t n = filename.size();
    ASSERT( M_SCENE, n >= 4 && filename.substr( n - 4) == ".mdl");
    return filename.substr( 0, n - 4);
}

bool is_archive_filename( const std::string& filename)
{
    size_t n = filename.size();
    return n > 4 && filename.substr( filename.size() - 4) == ".mdr";
}

bool is_mdle_filename( const std::string& filename)
{
    size_t n = filename.size();
    return n > 5 && filename.substr( filename.size() - 5) == ".mdle";
}

bool is_container_member( const char* filename)
{
    return filename && ( strstr( filename, ".mdr:") != nullptr || strstr( filename, ".mdle:") != nullptr);
}

std::string get_container_filename( const char* filename)
{
    if( !filename)
        return std::string();

    // archive
    if( const char* mdr = strstr( filename, ".mdr:"))
        return std::string( filename, mdr + 4);

    // MDLE
    if( const char* mdle = strstr( filename, ".mdle:"))
        return std::string( filename, mdle + 5);

    return std::string();
}

const char* get_container_membername( const char* filename)
{
    if( !filename)
        return "";

    // archive
    if( const char* mdr = strstr( filename, ".mdr:"))
        return mdr + 5;

    // MDLE
    if( const char* mdle = strstr( filename, ".mdle:"))
        return mdle + 6;

    return "";
}

std::string add_slash_in_front_of_drive_letter( const std::string& filename)
{
#ifdef MI_PLATFORM_WINDOWS
    if( filename.size() >= 2 && is_mdl_letter( filename[0]) && filename[1] == ':')
        return "/" + filename;
    return filename;
#else // MI_PLATFORM_WINDOWS
    return filename;
#endif // MI_PLATFORM_WINDOWS
}

std::string remove_slash_in_front_of_drive_letter( const std::string& input)
{
#ifdef MI_PLATFORM_WINDOWS
    if( input.size() >= 3 && input[0] == '/' && is_mdl_letter( input[1]) && input[2] == ':')
        return input.substr( 1);
    return input;
#else // MI_PLATFORM_WINDOWS
    return input;
#endif // MI_PLATFORM_WINDOWS
}

// *************************************************************************************************

std::string uvtile_marker_to_string( const std::string& s, mi::Sint32 u, mi::Sint32 v)
{
    ASSERT( M_SCENE, !s.empty());

    std::stringstream result;

    size_t p = s.find( "<UVTILE0>");
    if( p != std::string::npos) {
        result << s.substr( 0, p) << "_u" << u << "_v" << v << s.substr( p+9);
        return result.str();
    }

    p = s.find( "<UVTILE1>");
    if( p != std::string::npos) {
        result << s.substr( 0, p) << "_u" << u+1 << "_v" << v+1 << s.substr( p+9);
        return result.str();
    }

    p = s.find( "<UDIM>");
    if( p != std::string::npos) {
        if( u < 0 || v < 0)
            return std::string();
        result << s.substr( 0, p) << 1000 + 10*v + u + 1 << s.substr( p+6);
        return result.str();
    }

    return std::string();
}

std::string uvtile_string_to_marker( const std::string& s, const std::string& marker)
{
    ASSERT( M_SCENE, !s.empty() && !marker.empty());

    std::regex regex;
    if( marker == "<UVTILE0>" || marker == "<UVTILE1>")
        regex = ".*(_u-?[0-9]+_v-?[0-9]+)(.*)";
    else if( marker == "<UDIM>")
        regex = ".*([1-9][0-9][0-9][0-9])(.*)";
    else
        return std::string();

    std::smatch matches;
    if( !regex_match( s, matches, regex))
        return std::string();

    ASSERT( M_SCENE, matches.size() == 3);
    auto p0 = matches.position( 1);
    auto p1 = matches.position( 2);

    std::string result( s);
    result.replace( p0, p1-p0, marker);
    return result;
}

std::string get_file_path(
    std::string filename, mi::neuraylib::IMdl_impexp_api::Search_option option)
{
    ASSERT( M_SCENE, !filename.empty());

    SYSTEM::Access_module<PATH::Path_module> path_module( false);
    PATH::Path_module::Search_path mdl_paths = path_module->get_search_path( PATH::MDL);

    // Apply the same normalization as the PATH module (and make sure that "sep" matches).
    filename = path_module->normalize( filename);
    // Normalize input further for better matches.
    filename = HAL::Ospath::normpath_v2( filename);

    const std::string& sep = HAL::Ospath::sep();
    ASSERT( M_NEURAY_API, sep.size() == 1);

    std::string result;

    for( auto& mdl_path: mdl_paths) {

        // Normalize search path for better matches.
        mdl_path = HAL::Ospath::normpath_v2( mdl_path);
        // Add trailing separator.
        if( mdl_path.empty() || (mdl_path.back() != sep[0]))
            mdl_path += sep;

        size_t n = mdl_path.size();
        if( filename.substr( 0, n) != mdl_path)
            continue;

        const std::string& candidate = filename.substr( n-1);

        switch( option) {

            case mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_FIRST:
                if( result.empty())
                    result = candidate;
                break;

            // shortest search path => longest module name
            case mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_SHORTEST:
                if( candidate.size() > result.size())
                    result = candidate;
                break;

            // longest search path => shortest module name
            case mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_LONGEST:
                if( result.empty() || (candidate.size() < result.size()))
                    result = candidate;
                break;

            case mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_FORCE_32_BIT:
                return nullptr;
        }
    }

    return HAL::Ospath::convert_to_forward_slashes( result);
}

} // namespace MDL

} // namespace MI

