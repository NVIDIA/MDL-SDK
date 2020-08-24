/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <climits>

#include <mi/mdl/mdl_generated_executable.h>

#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Metadata.h>

#include "generator_jit_generated_code.h"
#include "generator_jit_type_map.h"

namespace mi {
namespace mdl {

/// Helper: get the index of a matrix type in the matrix type cache.
///
/// \param col  number of columns
/// \param row  number of rows
static int get_matric_index(int col, int row)
{
    MDL_ASSERT(2 <= col && col <= 4);
    MDL_ASSERT(2 <= row && row <= 4);

    return (col - 2) * 3 + row - 2;
}

// Constructor.
Type_mapper::Type_mapper(
    IAllocator             *alloc,
    llvm::LLVMContext      &context,
    llvm::DataLayout const *target_data,
    unsigned               state_mapping,
    Type_mapping_mode      tm_mode,
    unsigned               num_texture_spaces,
    unsigned               num_texture_results)
: m_state_mapping(state_mapping)
, m_context(context)
, m_data_layout(*target_data)
, m_tm_mode(tm_mode)

, m_type_size_t(llvm::IntegerType::get(context, sizeof(size_t) * CHAR_BIT))

, m_type_void(llvm::Type::getVoidTy(context))
// the LLVM void type is not allowed to be used for get_ptr, so create a int8 * here
, m_type_void_ptr(get_ptr(llvm::Type::getInt8Ty(context)))

// the type of LLVM predicates (aka compare results)
, m_type_predicate(llvm::Type::getInt1Ty(context))

, m_type_bool(llvm::IntegerType::get(context, tm_mode & TM_BOOL1_SUPPORTED ? 1 : 8))
, m_type_bool2(llvm::VectorType::get(m_type_bool, 2))
, m_type_bool3(llvm::VectorType::get(m_type_bool, 3))
, m_type_bool4(llvm::VectorType::get(m_type_bool, 4))

// int is 32bit in MDL
, m_type_int(llvm::Type::getInt32Ty(context))
, m_type_int2(llvm::VectorType::get(m_type_int, 2))
, m_type_int3(llvm::VectorType::get(m_type_int, 3))
, m_type_int4(llvm::VectorType::get(m_type_int, 4))

, m_type_float(llvm::Type::getFloatTy(context))
, m_type_float2(llvm::VectorType::get(m_type_float, 2))
, m_type_float3(llvm::VectorType::get(m_type_float, 3))
, m_type_float4(llvm::VectorType::get(m_type_float, 4))

, m_type_double(llvm::Type::getDoubleTy(context))
, m_type_double2(llvm::VectorType::get(m_type_double, 2))
, m_type_double3(llvm::VectorType::get(m_type_double, 3))
, m_type_double4(llvm::VectorType::get(m_type_double, 4))

, m_type_color(NULL)

// int/float arrays needed in glue code / state interface
, m_type_arr_int_2(llvm::ArrayType::get(m_type_int, 2))
, m_type_arr_int_3(llvm::ArrayType::get(m_type_int, 3))
, m_type_arr_int_4(llvm::ArrayType::get(m_type_int, 4))

, m_type_arr_float_2(llvm::ArrayType::get(m_type_float, 2))
, m_type_arr_float_3(llvm::ArrayType::get(m_type_float, 3))
, m_type_arr_float_4(llvm::ArrayType::get(m_type_float, 4))

, m_type_deriv_float(NULL)
, m_type_deriv_float2(NULL)
, m_type_deriv_float3(NULL)
, m_type_deriv_arr_float_2(NULL)
, m_type_deriv_arr_float_3(NULL)
, m_type_deriv_arr_float_4(NULL)

// matrix types ...
, m_type_float2x2(NULL)
, m_type_float3x2(NULL)
, m_type_float4x2(NULL)
, m_type_float2x3(NULL)
, m_type_float3x3(NULL)
, m_type_float4x3(NULL)
, m_type_float2x4(NULL)
, m_type_float3x4(NULL)
, m_type_float4x4(NULL)
, m_type_double2x2(NULL)
, m_type_double3x2(NULL)
, m_type_double4x2(NULL)
, m_type_double2x3(NULL)
, m_type_double3x3(NULL)
, m_type_double4x3(NULL)
, m_type_double2x4(NULL)
, m_type_double3x4(NULL)
, m_type_double4x4(NULL)

// we represent MDL strings as C-strings, i.e. char pointer
, m_type_char(llvm::Type::getInt8Ty(context))
, m_type_cstring(get_ptr(m_type_char))

, m_type_tag(llvm::IntegerType::get(context, sizeof(Tag) * CHAR_BIT))

// State types constructed later
, m_type_state_environment(NULL)
, m_type_state_environment_ptr(NULL)
, m_type_state_core(NULL)
, m_type_state_core_ptr(NULL)

// Exception state type constructed later
, m_type_exc_state(NULL)
, m_type_exc_state_ptr(NULL)

// res_data_pair type constructed later
, m_type_res_data_pair(NULL)
, m_type_res_data_pair_ptr(NULL)

// core texture handler type constructed later
, m_type_core_tex_handler(NULL)
, m_type_core_tex_handler_ptr(NULL)

// attribute entry types constructed later
, m_type_texture_attribute_entry(NULL)
, m_type_light_profile_attribute_entry(NULL)
, m_type_bsdf_measurement_attribute_entry(NULL)

// optix types constructed on demand
, m_type_optix_type_info(NULL)

, m_type_struct_cache(0, Type_struct_map::hasher(), Type_struct_map::key_equal(), alloc)
, m_deriv_type_cache(0, Deriv_type_map::hasher(), Deriv_type_map::key_equal(), alloc)
, m_deriv_type_set(0, Deriv_type_set::hasher(), Deriv_type_set::key_equal(), alloc)
, m_type_arr_cache(0, Type_array_map::hasher(), Type_array_map::key_equal(), alloc)
{
    switch (tm_mode & TM_VECTOR_MASK) {
    case TM_ALL_SCALAR:
        // don't use vector types at all
        m_type_bool2     = llvm::ArrayType::get(m_type_bool,   2);
        m_type_bool3     = llvm::ArrayType::get(m_type_bool,   3);
        m_type_bool4     = llvm::ArrayType::get(m_type_bool,   4);

        m_type_int2      = llvm::ArrayType::get(m_type_int,    2);
        m_type_int3      = llvm::ArrayType::get(m_type_int,    3);
        m_type_int4      = llvm::ArrayType::get(m_type_int,    4);

        m_type_float2    = llvm::ArrayType::get(m_type_float,  2);
        m_type_float3    = llvm::ArrayType::get(m_type_float,  3);
        m_type_float4    = llvm::ArrayType::get(m_type_float,  4);

        m_type_double2   = llvm::ArrayType::get(m_type_double, 2);
        m_type_double3   = llvm::ArrayType::get(m_type_double, 3);
        m_type_double4   = llvm::ArrayType::get(m_type_double, 4);

        m_type_float2x2  = llvm::ArrayType::get(m_type_float,  2 * 2);
        m_type_float3x2  = llvm::ArrayType::get(m_type_float,  3 * 2);
        m_type_float4x2  = llvm::ArrayType::get(m_type_float,  4 * 2);
        m_type_float2x3  = llvm::ArrayType::get(m_type_float,  2 * 3);
        m_type_float3x3  = llvm::ArrayType::get(m_type_float,  3 * 3);
        m_type_float4x3  = llvm::ArrayType::get(m_type_float,  4 * 3);
        m_type_float2x4  = llvm::ArrayType::get(m_type_float,  2 * 4);
        m_type_float3x4  = llvm::ArrayType::get(m_type_float,  3 * 4);
        m_type_float4x4  = llvm::ArrayType::get(m_type_float,  4 * 4);

        m_type_double2x2 = llvm::ArrayType::get(m_type_double, 2 * 2);
        m_type_double3x2 = llvm::ArrayType::get(m_type_double, 3 * 2);
        m_type_double4x2 = llvm::ArrayType::get(m_type_double, 4 * 2);
        m_type_double2x3 = llvm::ArrayType::get(m_type_double, 2 * 3);
        m_type_double3x3 = llvm::ArrayType::get(m_type_double, 3 * 3);
        m_type_double4x3 = llvm::ArrayType::get(m_type_double, 4 * 3);
        m_type_double2x4 = llvm::ArrayType::get(m_type_double, 2 * 4);
        m_type_double3x4 = llvm::ArrayType::get(m_type_double, 3 * 4);
        m_type_double4x4 = llvm::ArrayType::get(m_type_double, 4 * 4);
        break;

    case TM_SMALL_VECTORS:
        // matrix types are arrays of vectors ...
        m_type_float2x2  = llvm::ArrayType::get(m_type_float2,  2);
        m_type_float3x2  = llvm::ArrayType::get(m_type_float2,  3);
        m_type_float4x2  = llvm::ArrayType::get(m_type_float2,  4);
        m_type_float2x3  = llvm::ArrayType::get(m_type_float3,  2);
        m_type_float3x3  = llvm::ArrayType::get(m_type_float3,  3);
        m_type_float4x3  = llvm::ArrayType::get(m_type_float3,  4);
        m_type_float2x4  = llvm::ArrayType::get(m_type_float4,  2);
        m_type_float3x4  = llvm::ArrayType::get(m_type_float4,  3);
        m_type_float4x4  = llvm::ArrayType::get(m_type_float4,  4);

        m_type_double2x2 = llvm::ArrayType::get(m_type_double2, 2);
        m_type_double3x2 = llvm::ArrayType::get(m_type_double2, 3);
        m_type_double4x2 = llvm::ArrayType::get(m_type_double2, 4);
        m_type_double2x3 = llvm::ArrayType::get(m_type_double3, 2);
        m_type_double3x3 = llvm::ArrayType::get(m_type_double3, 3);
        m_type_double4x3 = llvm::ArrayType::get(m_type_double3, 4);
        m_type_double2x4 = llvm::ArrayType::get(m_type_double4, 2);
        m_type_double3x4 = llvm::ArrayType::get(m_type_double4, 3);
        m_type_double4x4 = llvm::ArrayType::get(m_type_double4, 4);
        break;
    case TM_BIG_VECTORS:
        // matrix types are big vectors ...
        m_type_float2x2  = llvm::VectorType::get(m_type_float,  2 * 2);
        m_type_float3x2  = llvm::VectorType::get(m_type_float,  3 * 2);
        m_type_float4x2  = llvm::VectorType::get(m_type_float,  4 * 2);
        m_type_float2x3  = llvm::VectorType::get(m_type_float,  2 * 3);
        m_type_float3x3  = llvm::VectorType::get(m_type_float,  3 * 3);
        m_type_float4x3  = llvm::VectorType::get(m_type_float,  4 * 3);
        m_type_float2x4  = llvm::VectorType::get(m_type_float,  2 * 4);
        m_type_float3x4  = llvm::VectorType::get(m_type_float,  3 * 4);
        m_type_float4x4  = llvm::VectorType::get(m_type_float,  4 * 4);

        m_type_double2x2 = llvm::VectorType::get(m_type_double, 2 * 2);
        m_type_double3x2 = llvm::VectorType::get(m_type_double, 3 * 2);
        m_type_double4x2 = llvm::VectorType::get(m_type_double, 4 * 2);
        m_type_double2x3 = llvm::VectorType::get(m_type_double, 2 * 3);
        m_type_double3x3 = llvm::VectorType::get(m_type_double, 3 * 3);
        m_type_double4x3 = llvm::VectorType::get(m_type_double, 4 * 3);
        m_type_double2x4 = llvm::VectorType::get(m_type_double, 2 * 4);
        m_type_double3x4 = llvm::VectorType::get(m_type_double, 3 * 4);
        m_type_double4x4 = llvm::VectorType::get(m_type_double, 4 * 4);
        break;
    }

#define ENTRY(ty, col, row) m_##ty##_matrix[get_matric_index(col, row)] = m_type_##ty##col##x##row

    ENTRY(float, 2, 2);
    ENTRY(float, 3, 2);
    ENTRY(float, 4, 2);
    ENTRY(float, 2, 3);
    ENTRY(float, 3, 3);
    ENTRY(float, 4, 3);
    ENTRY(float, 2, 4);
    ENTRY(float, 3, 4);
    ENTRY(float, 4, 4);

    ENTRY(double, 2, 2);
    ENTRY(double, 3, 2);
    ENTRY(double, 4, 2);
    ENTRY(double, 2, 3);
    ENTRY(double, 3, 3);
    ENTRY(double, 4, 3);
    ENTRY(double, 2, 4);
    ENTRY(double, 3, 4);
    ENTRY(double, 4, 4);

#undef ENTRY

    // the following types depend on float3 representation ...

    // for now, a color is a RGB float
    m_type_color = m_type_float3;

    if (use_derivatives()) {
        m_type_deriv_float = lookup_deriv_type(m_type_float);

        m_type_deriv_arr_float_2 = lookup_deriv_type(m_type_arr_float_2);
        m_type_deriv_float2 = lookup_deriv_type(m_type_float2);

        m_type_deriv_arr_float_3 = lookup_deriv_type(m_type_arr_float_3);
        m_type_deriv_float3 = lookup_deriv_type(m_type_float3);

        m_type_deriv_arr_float_4 = lookup_deriv_type(m_type_arr_float_4);
    }

    bool vec_in_structs = !target_supports_pointers();

    // built state types
    m_type_state_environment     = construct_state_environment_type(
        context,
        vec_in_structs ? m_type_float3 : m_type_arr_float_3);
    m_type_state_environment_ptr = get_ptr(m_type_state_environment);
    m_type_state_core            = construct_state_core_type(
        context,
        m_type_int,
        vec_in_structs ? m_type_float3 : m_type_arr_float_3,
        vec_in_structs ? m_type_float4 : m_type_arr_float_4,
        m_type_float, m_type_cstring,
        vec_in_structs ? m_type_deriv_float3 : m_type_deriv_arr_float_3,
        vec_in_structs ? m_type_float4x4 : get_ptr(m_type_arr_float_4),
        num_texture_spaces,
        num_texture_results);
    m_type_state_core_ptr        = get_ptr(m_type_state_core);

    m_type_exc_state             = construct_exception_state_type(
        context, m_type_void_ptr, llvm::Type::getInt32Ty(context));
    m_type_exc_state_ptr         = get_ptr(m_type_exc_state);

    m_type_res_data_pair         = construct_res_data_pair_type(context);
    m_type_res_data_pair_ptr     = get_ptr(m_type_res_data_pair);

    m_type_exec_ctx              = construct_exec_ctx_type(
        context,
        m_type_state_core_ptr,
        m_type_res_data_pair_ptr,
        m_type_exc_state_ptr,
        m_type_void_ptr);
    m_type_exec_ctx_ptr          = get_ptr(m_type_exec_ctx);


    // these must be run last, as they expect fully initialized upper types
    m_type_core_tex_handler      = construct_core_texture_handler_type(context);
    m_type_core_tex_handler_ptr  = get_ptr(m_type_core_tex_handler);

    m_type_texture_attribute_entry =
        construct_texture_attribute_entry_type(context);
    m_type_texture_attribute_entry_ptr =
        get_ptr(m_type_texture_attribute_entry);

    m_type_light_profile_attribute_entry =
        construct_light_profile_attribuute_entry_type(context);
    m_type_light_profile_attribute_entry_ptr =
        get_ptr(m_type_light_profile_attribute_entry);

    m_type_bsdf_measurement_attribute_entry =
        construct_bsdf_measurement_attribuute_entry_type(context);
    m_type_bsdf_measurement_attribute_entry_ptr =
        get_ptr(m_type_bsdf_measurement_attribute_entry);
}

// Get the index of a state field in the current state struct.
int Type_mapper::get_state_index(
    State_field state_field)
{
    switch (state_field) {

    // Environment context
    case STATE_ENV_DIRECTION:
        // always 0
        return 0;
    case STATE_ENV_RO_DATA_SEG:
        // always 1
        return 1;

    // Core context
    case STATE_CORE_NORMAL:
        return 0;
    case STATE_CORE_GEOMETRY_NORMAL:
        return 1;
    case STATE_CORE_POSITION:
        return 2;
    case STATE_CORE_ANIMATION_TIME:
        return 3;
    case STATE_CORE_TEXTURE_COORDINATE:
        return 4;
    case STATE_CORE_TANGENT_U:
        if (use_bitangents())
            return -1;
        return 5;
    case STATE_CORE_TANGENT_V:
        if (use_bitangents())
            return -1;
        return 6;
    case STATE_CORE_BITANGENTS:
        if (use_bitangents())
            return 5;
        return -1;
    case STATE_CORE_TEXT_RESULTS:
        if (use_bitangents())
            return 6;
        return 7;
    case STATE_CORE_RO_DATA_SEG:
        if (use_bitangents())
            return 7;
        return 8;
    case STATE_CORE_W2O_TRANSFORM:
        if (state_includes_uniform_state()) {
            if (use_bitangents())
                return 8;
            return 9;
        }
        return -1;
    case STATE_CORE_O2W_TRANSFORM:
        if (state_includes_uniform_state()) {
            if (use_bitangents())
                return 9;
            return 10;
        }
        return -1;
    case STATE_CORE_OBJECT_ID:
        if (state_includes_uniform_state()) {
            if (use_bitangents())
                return 10;
            return 11;
        }
        return -1;
    case STATE_CORE_METERS_PER_SCENE_UNIT:
        if (state_includes_uniform_state()) {
            if (use_bitangents())
                return 11;
            return 12;
        }
        return -1;
    case STATE_CORE_ARG_BLOCK_OFFSET:
        if (state_includes_arg_block_offset()) {
            if (use_bitangents())
                return 12;
            return 13;
        }
        return -1;
    }
    return -1;
}

// Get an llvm type for an MDL type.
llvm::Type *Type_mapper::lookup_type(
    llvm::LLVMContext &context,
    mdl::IType const  *type,
    int               arr_size) const
{
    switch (type->get_kind()) {
    case mdl::IType::TK_ALIAS:
        return lookup_type(context, cast<mdl::IType_alias>(type)->get_aliased_type());
    case mdl::IType::TK_BOOL:
        return m_type_bool;
    case mdl::IType::TK_INT:
        return m_type_int;
    case mdl::IType::TK_ENUM:
        // map to int
        return m_type_int;
    case mdl::IType::TK_FLOAT:
        return m_type_float;
    case mdl::IType::TK_DOUBLE:
        return m_type_double;
    case mdl::IType::TK_STRING:
        return get_string_type();
    case mdl::IType::TK_LIGHT_PROFILE:
    case mdl::IType::TK_BSDF:
    case mdl::IType::TK_HAIR_BSDF:
    case mdl::IType::TK_EDF:
    case mdl::IType::TK_VDF:
        // handled as tags for now
        return m_type_tag;
    case mdl::IType::TK_VECTOR:
        {
            mdl::IType_vector const *v_type = cast<mdl::IType_vector>(type);
            mdl::IType_atomic const *e_type = v_type->get_element_type();
            int size = v_type->get_size();

            switch (e_type->get_kind()) {
            case mdl::IType::TK_BOOL:
                switch (size) {
                case 2: return m_type_bool2;
                case 3: return m_type_bool3;
                case 4: return m_type_bool4;
                }
                break;
            case mdl::IType::TK_INT:
                switch (size) {
                case 2: return m_type_int2;
                case 3: return m_type_int3;
                case 4: return m_type_int4;
                }
                break;
            case mdl::IType::TK_FLOAT:
                switch (size) {
                case 2: return m_type_float2;
                case 3: return m_type_float3;
                case 4: return m_type_float4;
                }
                break;
            case mdl::IType::TK_DOUBLE:
                switch (size) {
                case 2: return m_type_double2;
                case 3: return m_type_double3;
                case 4: return m_type_double4;
                }
                break;
            default:
                break;
            }
            MDL_ASSERT(!"Unsupported atomic type");
            return NULL;
        }
    case mdl::IType::TK_MATRIX:
        {
            mdl::IType_matrix const *m_type = cast<mdl::IType_matrix>(type);
            mdl::IType_vector const *v_type = m_type->get_element_type();
            int col = m_type->get_columns();
            int row = v_type->get_size();
            int idx = get_matric_index(col, row);

            switch (v_type->get_element_type()->get_kind()) {
            case mdl::IType::TK_FLOAT:
                return m_float_matrix[idx];
            case mdl::IType::TK_DOUBLE:
                return m_double_matrix[idx];
            default:
                break;
            }
            MDL_ASSERT(!"Unsupported matrix type");
            return NULL;
        }
    case mdl::IType::TK_ARRAY:
        {
            // lookup in the array cache
            Type_array_map::const_iterator it = m_type_arr_cache.find(
                Array_type_cache_key(type, arr_size));
            if (it != m_type_arr_cache.end())
                return it->second;

            mdl::IType_array const *a_type = cast<mdl::IType_array>(type);

            mdl::IType const *e_type = a_type->get_element_type();
            llvm::Type       *res    = lookup_type(context, e_type);

            if (a_type->is_immediate_sized()) {
                res = llvm::ArrayType::get(res, a_type->get_size());
            } else if (arr_size >= 0) {
                // instantiated array
                res = llvm::ArrayType::get(res, arr_size);
            } else {
                // uninstantiated deferred size array, create an array_desc<T> struct
                llvm::Type *types[2];
                types[ARRAY_DESC_BASE] = get_ptr(res);
                types[ARRAY_DESC_SIZE] = m_type_size_t;
                res = llvm::StructType::get(context, types, /*isPacked=*/false);
            }
            m_type_arr_cache[Array_type_cache_key(type, arr_size)] = res;
            return res;
        }
    case mdl::IType::TK_COLOR:
        return m_type_color;
    case mdl::IType::TK_FUNCTION:
        // should never be needed
        MDL_ASSERT(!"requested function type");
        return NULL;
    case mdl::IType::TK_STRUCT:
        {
            mdl::IType_struct const *s_type = cast<mdl::IType_struct>(type);
            char const              *s_name = s_type->get_symbol()->get_name();

            // the MDL type is a derivative type?
            if (s_name[0] == '#') {
                // retrieve original type and call derivative type lookup function
                type = s_type->get_compound_type(0);
                return lookup_deriv_type(type, arr_size);
            }

            // lookup in the struct cache
            Type_struct_map::const_iterator it = m_type_struct_cache.find(s_name);
            if (it != m_type_struct_cache.end())
                return it->second;

            // build the struct
            int n_fields = s_type->get_field_count();

            llvm::SmallVector<llvm::Type *, 16> member_types;
            member_types.resize(n_fields);

            for (int i = 0; i < n_fields; ++i) {
                mdl::IType const   *m_type;
                mdl::ISymbol const *m_sym;

                s_type->get_field(i, m_type, m_sym);

                member_types[i] = lookup_type(context, m_type);
            }

            llvm::Type *res = llvm::StructType::create(
                context, member_types, s_name, /*isPacked=*/false);

            m_type_struct_cache[s_name] = res;
            return res;
        }
    case mdl::IType::TK_TEXTURE:
    case mdl::IType::TK_BSDF_MEASUREMENT:
        // handled as tags for now
        return m_type_tag;
    case mdl::IType::TK_INCOMPLETE:
    case mdl::IType::TK_ERROR:
        // should never be needed
        MDL_ASSERT(!"requested error type");
        return NULL;
    }
    MDL_ASSERT(!"Unsupported type");
    return NULL;
}

// Get an LLVM type for an MDL type with derivatives.
llvm::StructType *Type_mapper::lookup_deriv_type(
    mdl::IType const *type,
    int arr_size) const
{
    // the MDL type already is a derivative type?
    if (IType_struct const *struct_type = as<IType_struct>(type)) {
        if (struct_type->get_symbol()->get_name()[0] == '#') {
            // retrieve original type
            type = struct_type->get_compound_type(0);
        }
    }

    llvm::Type *llvm_type = lookup_type(m_context, type, arr_size);
    return lookup_deriv_type(llvm_type);
}

// Get an LLVM type with derivatives for an LLVM type.
llvm::StructType *Type_mapper::lookup_deriv_type(llvm::Type *type) const
{
    Deriv_type_map::const_iterator it(m_deriv_type_cache.find(type));
    if (it != m_deriv_type_cache.end())
        return it->second;

    llvm::Type *member_types[3] = { type, type, type };
    llvm::StructType *res = llvm::StructType::get(m_context, member_types, /*isPacked=*/ false);
    m_deriv_type_cache[type] = res;
    m_deriv_type_set.insert(res);

    return res;
}

// Checks if the given LLVM type is a derivative type.
bool Type_mapper::is_deriv_type(llvm::Type *type) const
{
    Deriv_type_set::const_iterator it(m_deriv_type_set.find(type));
    return it != m_deriv_type_set.end();
}

// Checks if the given MDL type is a derivative type.
bool Type_mapper::is_deriv_type(mi::mdl::IType const *type) const
{
    if (IType_struct const *struct_type = as<IType_struct>(type)) {
        return struct_type->get_symbol()->get_name()[0] == '#';
    }
    return false;
}

// Check whether the given type is based on a floating point type.
bool Type_mapper::is_floating_point_based_type(IType const *type) const
{
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_COLOR:
    case IType::TK_MATRIX:  // there are only float and double matrix types
        return true;

    case IType::TK_VECTOR:
        {
            IType_vector const *vec_type = as<IType_vector>(type);
            IType_atomic const *elem_type = vec_type->get_element_type();
            IType::Kind elem_kind = elem_type->get_kind();
            return elem_kind == IType::TK_FLOAT || elem_kind == IType::TK_DOUBLE;
        }

    case IType::TK_ARRAY:
        {
            IType_array const *array_type = as<IType_array>(type);
            IType const *elem_type = array_type->get_element_type();
            return is_floating_point_based_type(elem_type);
        }

    default:
        return false;
    }
}

// Get the base value type of a derivative type.
mi::mdl::IType const *Type_mapper::get_deriv_base_type(mi::mdl::IType const *type) const
{
    if (!is_deriv_type(type))
        return NULL;
    return cast<mi::mdl::IType_struct>(type)->get_compound_type(0);
}

// Get the base value LLVM type of a derivative LLVM type.
llvm::Type *Type_mapper::get_deriv_base_type(llvm::Type *type) const
{
    if (!is_deriv_type(type))
        return NULL;
    return type->getStructElementType(0);
}

// Skip to the base value type of a derivative type or just return the type itself for
// non-derivative types.
mi::mdl::IType const *Type_mapper::skip_deriv_type(mi::mdl::IType const *type) const
{
    if (!is_deriv_type(type))
        return type;
    return cast<mi::mdl::IType_struct>(type)->get_compound_type(0);
}

// Skip to the base value LLVM type of a derivative LLVM type or just return the type itself
// for non-derivative types.
llvm::Type *Type_mapper::skip_deriv_type(llvm::Type *type) const
{
    if (!is_deriv_type(type))
        return type;
    return type->getStructElementType(0);
}

// Checks if a given type needs reference return calling convention.
bool Type_mapper::need_reference_return(mi::mdl::IType const *type) const
{
    if (m_tm_mode & TM_NO_REFERENCE)
        return false;

    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case mi::mdl::IType::TK_ALIAS:
        // should not happen
        break;
    case mi::mdl::IType::TK_BOOL:
    case mi::mdl::IType::TK_INT:
    case mi::mdl::IType::TK_ENUM:
    case mi::mdl::IType::TK_FLOAT:
    case mi::mdl::IType::TK_DOUBLE:
        // all atomic
        return false;
    case mi::mdl::IType::TK_STRING:
        // because we "known" all possible string literals in advance
        // we use char * for strings, so it is atomic
        return false;
    case mi::mdl::IType::TK_LIGHT_PROFILE:
    case mi::mdl::IType::TK_BSDF:
    case mi::mdl::IType::TK_HAIR_BSDF:
    case mi::mdl::IType::TK_EDF:
    case mi::mdl::IType::TK_VDF:
        // returned as atomic tags
        return false;
    case mi::mdl::IType::TK_VECTOR:
        if ((m_tm_mode & TM_VECTOR_MASK) == TM_ALL_SCALAR) {
            // returned as an array, needs reference
            return true;
        }
        // else returned as a LLVM vector type
        return false;
    case mi::mdl::IType::TK_MATRIX:
        if ((m_tm_mode & (TM_VECTOR_MASK|TM_BIG_VECTOR_RETURN)) ==
            (TM_BIG_VECTORS|TM_BIG_VECTOR_RETURN))
        {
            // returned as a LLVM vector type
            return false;
        }
        // else is either returned as an array, or the BE does not support returning big
        // vectors, needs reference
        return true;
    case mi::mdl::IType::TK_ARRAY:
        // use reference for arrays
        return true;
    case mi::mdl::IType::TK_COLOR:
        if ((m_tm_mode & TM_VECTOR_MASK) == TM_ALL_SCALAR) {
            // returned as an array, needs reference
            return true;
        }
        // else returned as a LLVM vector type
        return false;
    case mi::mdl::IType::TK_FUNCTION:
        // should not happen
        break;
    case mi::mdl::IType::TK_STRUCT:
        // return structs by value
        return false;
    case mi::mdl::IType::TK_TEXTURE:
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        // returned as atomic tags
        return false;
    case mi::mdl::IType::TK_INCOMPLETE:
    case mi::mdl::IType::TK_ERROR:
        // should not happen
        break;
    }
    MDL_ASSERT(!"Unexpected type kind");
    return false;
}

// Check if the given parameter type must be passed by reference.
bool Type_mapper::is_passed_by_reference(mi::mdl::IType const *type) const
{
    if (m_tm_mode & TM_NO_REFERENCE)
        return false;
restart:
    switch (type->get_kind()) {
    case mi::mdl::IType::TK_ALIAS:
        {
            mi::mdl::IType_alias const *a_type = cast<mi::mdl::IType_alias>(type);
            type = a_type->get_aliased_type();
            goto restart;
        }
    case mi::mdl::IType::TK_BOOL:
    case mi::mdl::IType::TK_INT:
    case mi::mdl::IType::TK_ENUM:
    case mi::mdl::IType::TK_FLOAT:
    case mi::mdl::IType::TK_DOUBLE:
        // simple atomic types, pass by value
        return false;
    case mi::mdl::IType::TK_STRING:
        // only string literals represented by C-strings, pass by value
        return false;
    case mi::mdl::IType::TK_LIGHT_PROFILE:
    case mi::mdl::IType::TK_BSDF:
    case mi::mdl::IType::TK_HAIR_BSDF:
    case mi::mdl::IType::TK_EDF:
    case mi::mdl::IType::TK_VDF:
        // pass tags by value
        return false;
    case mi::mdl::IType::TK_VECTOR:
        if ((m_tm_mode & TM_VECTOR_MASK) == TM_ALL_SCALAR) {
            // returned as an array, needs reference
            return true;
        }
        // else represented by LLVM vector types, pass by value
        return false;
    case mi::mdl::IType::TK_MATRIX:
        if ((m_tm_mode & TM_VECTOR_MASK) == TM_BIG_VECTORS) {
            // represented by LLVM vector types, pass by value
            return false;
        }
        // else represented by LLVM array types, pass by reference
        return true;
    case mi::mdl::IType::TK_ARRAY:
        // try by reference to safe memory and CPU cycles
        return true;
    case mi::mdl::IType::TK_COLOR:
        if ((m_tm_mode & TM_VECTOR_MASK) == TM_ALL_SCALAR) {
            // represented by LLVM array type, needs reference
            return true;
        }
        // else represented by a LLVM vector type, pass by value
        return false;
    case mi::mdl::IType::TK_FUNCTION:
        MDL_ASSERT(!"unhandled function type");
        return false;
    case mi::mdl::IType::TK_STRUCT:
        // pass by reference if possible, let LLVM decide
        return true;
    case mi::mdl::IType::TK_TEXTURE:
    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        // pass tags by value
        return false;
    case mi::mdl::IType::TK_INCOMPLETE:
        MDL_ASSERT(!"incomplete type occured");
        return false;
    case mi::mdl::IType::TK_ERROR:
        MDL_ASSERT(!"error type occured");
        return false;
    }
    MDL_ASSERT(!"unsupported type kind");
    return false;
}

// Get a pointer type from a base type.
llvm::PointerType *Type_mapper::get_ptr(llvm::Type *type)
{
    // use default memory space
    return llvm::PointerType::get(type, 0);
}

// Get the LLVM State * type for the given state context.
llvm::PointerType *Type_mapper::get_state_ptr_type(
    State_subset_mode mode)
{
    switch (mode) {
    case SSM_NO_STATE:
        MDL_ASSERT(!"state type should not be requested in empty state mode");
        return NULL;

    case SSM_ENVIRONMENT:
        return m_type_state_environment_ptr;

    case SSM_CORE:
        return m_type_state_core_ptr;

    case SSM_FULL_SET:
        // FIXME: no bigger set available yet
        return m_type_state_core_ptr;
    }
    MDL_ASSERT(!"unsupported state subset");
    return NULL;
}

// Get the Optix rti_internal_typeinfo::rti_typeinfo type.
llvm::StructType *Type_mapper::get_optix_typeinfo_type()
{
    if (m_type_optix_type_info == NULL) {
        /*
        struct rti_typeinfo {
            unsigned int kind;
            unsigned int size;
        };
        */
        llvm::Type *members[] = {
            m_type_int,
            m_type_int
        };

        m_type_optix_type_info = llvm::StructType::create(
            m_context, members, "rti_internal_typeinfo::rti_typeinfo", /*is_packed=*/false);
    }
    return m_type_optix_type_info;
}

// Get the debug info type for an MDL type.
llvm::DIType *Type_mapper::get_debug_info_type(
    llvm::DIBuilder      *diBuilder,
    llvm::DIFile         *file,
    llvm::DIScope        *scope,
    mi::mdl::IType const *type) const
{
    switch (type->get_kind()) {
    case mi::mdl::IType::TK_ALIAS:
        {
            mi::mdl::IType_alias const *a_tp = cast<mi::mdl::IType_alias>(type);
            mi::mdl::IType const       *d_tp = a_tp->get_aliased_type();

            llvm::DIType *di_type = get_debug_info_type(diBuilder, file, scope, d_tp);

            // Note: LLVM 3.3 does not handle typedefs well, so do not try to generate them
            return di_type;

            mi::mdl::IType::Modifiers mod = a_tp->get_type_modifiers();
            if (mod == 0)
                return di_type;

            if (mod & mi::mdl::IType::MK_CONST)
                di_type = diBuilder->createQualifiedType(llvm::dwarf::DW_TAG_const_type, di_type);

            // FIXME: handle uniform and varying
            return diBuilder->createQualifiedType(llvm::dwarf::DW_TAG_typedef, di_type);
        }

    case mi::mdl::IType::TK_BOOL:
        return diBuilder->createBasicType(
            "bool",
            /*SizeInBits=*/m_type_bool->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_INT:
        return diBuilder->createBasicType(
            "int",
            /*SizeInBits=*/m_type_int->getBitWidth(),
            llvm::dwarf::DW_ATE_signed);

    case mi::mdl::IType::TK_ENUM:
        {
            mi::mdl::IType_enum const *e_type = cast<mi::mdl::IType_enum>(type);

            std::vector<llvm::Metadata *> enumeratorDescriptors;
            for (int i = 0, n = e_type->get_value_count(); i < n; ++i) {
                mi::mdl::ISymbol const *sym;
                int                    code;

                e_type->get_value(i, sym, code);

                llvm::Metadata *descriptor = diBuilder->createEnumerator(sym->get_name(), code);
                enumeratorDescriptors.push_back(descriptor);
            }
            llvm::DINodeArray elementArray = diBuilder->getOrCreateArray(enumeratorDescriptors);

            // FIXME: get the type position here
            int start_pos = 0;

            llvm::DIFile *diFile = diBuilder->createFile("", "");

            return diBuilder->createEnumerationType(
                scope,
                e_type->get_symbol()->get_name(),
                diFile,
                start_pos,
                /*SizeInBits=*/m_type_int->getBitWidth(),
                /*AlignInBits=*/32,
                elementArray,
                /*UnderlyingType=*/nullptr
            );
        }

    case mi::mdl::IType::TK_FLOAT:
        return diBuilder->createBasicType(
            "float",
            /*SizeInBits=*/m_type_float->getPrimitiveSizeInBits(),
            llvm::dwarf::DW_ATE_float);

    case mi::mdl::IType::TK_DOUBLE:
        return diBuilder->createBasicType(
            "double",
            /*SizeInBits=*/m_type_double->getPrimitiveSizeInBits(),
            llvm::dwarf::DW_ATE_float);

    case mi::mdl::IType::TK_STRING:
        if (strings_mapped_to_ids())
            return diBuilder->createBasicType(
                "string",
                /*SizeInBits=*/ m_type_tag->getPrimitiveSizeInBits(),
                llvm::dwarf::DW_ATE_unsigned);
        else {
            uint64_t pointer_size = m_data_layout.getTypeAllocSizeInBits(m_type_cstring);
            llvm::DIType *pointee_type = diBuilder->createBasicType(
                "char",
                /*SizeInBits=*/ 8,
                llvm::dwarf::DW_ATE_unsigned_char);

            return diBuilder->createPointerType(
                pointee_type,
                /*SizeInBits=*/  pointer_size,
                /*AlignInBits=*/ uint32_t(pointer_size),
                llvm::None,
                "string");
        }

    case mi::mdl::IType::TK_LIGHT_PROFILE:
        return diBuilder->createBasicType(
            "light_profile",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_BSDF:
        return diBuilder->createBasicType(
            "bsdf",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_HAIR_BSDF:
        return diBuilder->createBasicType(
            "hair_bsdf",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_EDF:
        return diBuilder->createBasicType(
            "edf",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_VDF:
        return diBuilder->createBasicType(
            "vdf",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_VECTOR:
        {
            mi::mdl::IType_vector const *v_tp = cast<mi::mdl::IType_vector>(type);
            mi::mdl::IType const        *e_tp = v_tp->get_element_type();
            int                         size  = v_tp->get_size();

            llvm::DIType      *eltType  = get_debug_info_type(diBuilder, file, scope, e_tp);
            llvm::Metadata    *sub      = diBuilder->getOrCreateSubrange(0, size - 1);
            llvm::DINodeArray  subArray = diBuilder->getOrCreateArray(sub);

            uint64_t sizeBits = eltType->getSizeInBits() * size;
            uint32_t align    = eltType->getAlignInBits();

            return diBuilder->createVectorType(sizeBits, align, eltType, subArray);
        }

    case mi::mdl::IType::TK_MATRIX:
        {
            mi::mdl::IType_matrix const *m_tp = cast<mi::mdl::IType_matrix>(type);
            mi::mdl::IType const        *e_tp = m_tp->get_element_type();
            int                         cols  = m_tp->get_columns();

            llvm::DIType      *eltType  = get_debug_info_type(diBuilder, file, scope, e_tp);
            llvm::Metadata    *sub      = diBuilder->getOrCreateSubrange(0, cols - 1);
            llvm::DINodeArray  subArray = diBuilder->getOrCreateArray(sub);

            uint64_t sizeBits = eltType->getSizeInBits() * cols;
            uint32_t align    = eltType->getAlignInBits();

            return diBuilder->createVectorType(sizeBits, align, eltType, subArray);
        }

    case mi::mdl::IType::TK_ARRAY:
        {
            mi::mdl::IType_array const *a_type = cast<mi::mdl::IType_array>(type);
            mi::mdl::IType const       *e_type = a_type->get_element_type();

            llvm::DIType *eltType = get_debug_info_type(diBuilder, file, scope, e_type);

            int lowerBound, upperBound;
            unsigned count = 0;
            if (a_type->is_immediate_sized()) {
                count      = a_type->get_size();
                lowerBound = 0;
                upperBound = (int)count - 1;
            } else {
                // deferred size array -> indicate with low > high
                lowerBound = 1;
                upperBound = 0;
            }

            llvm::Metadata    *sub      = diBuilder->getOrCreateSubrange(lowerBound, upperBound);
            llvm::DINodeArray  subArray = diBuilder->getOrCreateArray(sub);

            uint64_t size  = eltType->getSizeInBits() * count;
            uint32_t align = eltType->getAlignInBits();

            return diBuilder->createArrayType(size, align, eltType, subArray);
        }

    case mi::mdl::IType::TK_COLOR:
        {
            // modell as <3 * float>
            llvm::DIType *eltType =
                diBuilder->createBasicType(
                    "float",
                    /*SizeInBits=*/m_type_float->getPrimitiveSizeInBits(),
                    llvm::dwarf::DW_ATE_float);

            llvm::Metadata    *sub      = diBuilder->getOrCreateSubrange(0, 3 - 1);
            llvm::DINodeArray  subArray = diBuilder->getOrCreateArray(sub);

            uint64_t sizeBits = eltType->getSizeInBits() * 3;
            uint32_t align    = eltType->getAlignInBits();

            return diBuilder->createVectorType(sizeBits, align, eltType, subArray);
        }

    case mi::mdl::IType::TK_FUNCTION:
        break;

    case mi::mdl::IType::TK_STRUCT:
        {
            mi::mdl::IType_struct const *s_tp = cast<mi::mdl::IType_struct>(type);

            std::vector<llvm::Metadata *> field_types;
            uint64_t currentSize = 0;
            uint32_t align = 0;

            for (int i = 0, n = s_tp->get_field_count(); i < n; ++i) {
                mi::mdl::IType const   *f_tp;
                mi::mdl::ISymbol const *f_sym;

                s_tp->get_field(i, f_tp, f_sym);

                llvm::DIType *eltType = get_debug_info_type(diBuilder, file, scope, f_tp);
                uint32_t eltAlign = eltType->getAlignInBits();
                uint64_t eltSize  = eltType->getSizeInBits();

                // FIXME: should be retrieved from the DataLayout
                if (eltAlign == 0)
                    eltAlign = uint32_t(eltSize);

                MDL_ASSERT(eltAlign != 0);

                // The alignment for the entire structure is the maximum of the
                // required alignments of its elements
                align = std::max(align, eltAlign);

                // Move the current size forward if needed so that the current
                // element starts at an offset that's the correct alignment.
                if (currentSize > 0 && (currentSize % eltAlign) != 0)
                    currentSize += eltAlign - (currentSize % eltAlign);
                MDL_ASSERT((currentSize == 0) || (currentSize % eltAlign) == 0);

                // FIXME: position of the fields
                int line = 0;

                llvm::DIType *fieldType = diBuilder->createMemberType(
                    scope,
                    f_sym->get_name(),
                    file,
                    line,
                    eltSize,
                    eltAlign,
                    currentSize,
                    llvm::DINode::FlagZero,
                    eltType);
                field_types.push_back(fieldType);

                currentSize += eltSize;
            }
            // Round up the struct's entire size so that it's a multiple of the
            // required alignment that we figured out along the way...
            if (currentSize > 0 && (currentSize % align) != 0)
                currentSize += align - (currentSize % align);

            llvm::DINodeArray elements = diBuilder->getOrCreateArray(field_types);

            // FIXME: position of the struct
            int start_pos = 0;

            return diBuilder->createStructType(
                scope,
                s_tp->get_symbol()->get_name(),
                file,
                start_pos,
                currentSize,
                align,
                llvm::DINode::FlagZero,
                /*DerivedFrom=*/ nullptr,
                elements);
        }

    case mi::mdl::IType::TK_TEXTURE:
        return diBuilder->createBasicType(
            "texture",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_BSDF_MEASUREMENT:
        return diBuilder->createBasicType(
            "bsdf_measurement",
            /*SizeInBits=*/m_type_tag->getBitWidth(),
            llvm::dwarf::DW_ATE_unsigned);

    case mi::mdl::IType::TK_INCOMPLETE:
    case mi::mdl::IType::TK_ERROR:
        // should not happen
        break;
    }
    MDL_ASSERT(!"Unexpected type kind");
    return diBuilder->createUnspecifiedType("error");
}

// Get the debug info type for an MDL function type.
llvm::DISubroutineType *Type_mapper::get_debug_info_type(
    llvm::DIBuilder               *diBuilder,
    llvm::DIFile                  *file,
    mi::mdl::IType_function const *func_tp) const
{
    std::vector<llvm::Metadata *> signature_types;

    signature_types.push_back(
        get_debug_info_type(diBuilder, file, file, func_tp->get_return_type()));
    for (int i = 0, n = func_tp->get_parameter_count(); i < n; ++i) {
        mi::mdl::IType const   *p_tp;
        mi::mdl::ISymbol const *p_sym;

        func_tp->get_parameter(i, p_tp, p_sym);
        signature_types.push_back(get_debug_info_type(diBuilder, file, file, p_tp));
    }

    llvm::DITypeRefArray signature_types_array = diBuilder->getOrCreateTypeArray(signature_types);
    return diBuilder->createSubroutineType(signature_types_array);
}

// Construct the State type for the environment context.
llvm::StructType *Type_mapper::construct_state_environment_type(
    llvm::LLVMContext      &context,
    llvm::Type             *float3_type)
{
    llvm::Type *rodatasegment_type = target_supports_pointers()
        ? m_type_cstring
        : static_cast<llvm::Type *>(m_type_int);

    llvm::Type *members[] = {
        float3_type,          // direction
        rodatasegment_type,   // read-only data segment
    };

    llvm::StructType *res = llvm::StructType::create(
        context, members, "State_environment", /*is_packed=*/false);

#if defined(DEBUG) || defined(ENABLE_ASSERT)
    if (target_supports_pointers())
    {
        // check struct layout offsets and size
        // must match between LLVM layout and C++ layout from the native
        // compiler

        typedef mi::mdl::Shading_state_environment State;
        llvm::StructLayout const *sl = m_data_layout.getStructLayout(res);
        MDL_ASSERT(sl->getSizeInBytes() ==
            sizeof(State));
        MDL_ASSERT(sl->getElementOffset(get_state_index(STATE_ENV_DIRECTION)) ==
            offsetof(State, direction));
    }
#endif
    return res;
}

// Construct the State type for the iray core context.
llvm::StructType *Type_mapper::construct_state_core_type(
    llvm::LLVMContext      &context,
    llvm::Type             *int_type,
    llvm::Type             *float3_type,
    llvm::Type             *float4_type,
    llvm::Type             *float_type,
    llvm::Type             *byte_ptr_type,
    llvm::Type             *deriv_float3_type,
    llvm::Type             *float4x4_type,
    unsigned               num_texture_spaces,
    unsigned               num_texture_results)
{
    llvm::StructType *res = NULL;

    llvm::Type *pos_type = float3_type;
    if (use_derivatives()) {
        pos_type = deriv_float3_type;
    }

    llvm::Type *coord_type = target_supports_pointers()
        ? get_ptr(float3_type)
        : static_cast<llvm::Type *>(llvm::ArrayType::get(float3_type, num_texture_spaces));

    llvm::Type *tex_coord_type = coord_type;
    if (use_derivatives()) {
        tex_coord_type = target_supports_pointers()
            ? get_ptr(deriv_float3_type)
            : static_cast<llvm::Type *>(
                llvm::ArrayType::get(deriv_float3_type, num_texture_spaces));
    }

    llvm::Type *texres_type = target_supports_pointers()
        ? get_ptr(float4_type)
        : static_cast<llvm::Type *>(llvm::ArrayType::get(float4_type, num_texture_results));

    llvm::Type *rodatasegment_type = target_supports_pointers()
        ? byte_ptr_type
        : int_type;

    llvm::SmallVector<llvm::Type *, 13> members;
    members.push_back(float3_type);          // normal
    members.push_back(float3_type);          // geom_normal
    members.push_back(pos_type);             // position
    members.push_back(float_type);           // animation time
    members.push_back(tex_coord_type);       // texture_coordinate(index)

    if (use_bitangents()) {
        llvm::Type *bitangents_type = target_supports_pointers()
            ? get_ptr(float4_type)
            : static_cast<llvm::Type *>(llvm::ArrayType::get(float4_type, num_texture_spaces));

        members.push_back(bitangents_type);  // tangents_bitangentssign(index)
    } else {
        members.push_back(coord_type);       // tangent_u(index)
        members.push_back(coord_type);       // tangent_v(index)
    }

    members.push_back(texres_type);          // texture_results
    members.push_back(rodatasegment_type);   // read-only data segment
    members.push_back(float4x4_type);        // world-to-object transform matrix
    members.push_back(float4x4_type);        // object-to-world transform matrix
    members.push_back(int_type);             // state::object_id() result
    members.push_back(float_type);           // state::meters_per_scene_unit() result
    if (state_includes_arg_block_offset()) {
        members.push_back(int_type);
    }

    res = llvm::StructType::create(
        context, members, "State_core", /*is_packed=*/false);


#if defined(DEBUG) || defined(ENABLE_ASSERT)
    if (target_supports_pointers() && !use_derivatives()) {
        // check struct layout offsets and size
        // must match between LLVM layout and C++ layout from the native
        // compiler

        // MDL SDK state
        struct SDK_State {
            Float3_struct       normal;                   ///< state::normal() result
            Float3_struct       geom_normal;              ///< state::geom_normal() result
            Float3_struct       position;                 ///< state::position() result
            float               animation_time;           ///< state::animation_time() result
            Float3_struct const *text_coords;             ///< state::texture_coordinate() result
            Float3_struct const *tangent_u;               ///< state::texture_tangent_u() data
            Float3_struct const *tangent_v;               ///< state::texture_tangent_u() data
            Float4_struct const *text_results;            ///< texture results lookup table
            unsigned char const *ro_data_segment;         ///< read only data segment

            // these fields are used only if the uniform state is included
            Float4_struct const *world_to_object;         ///< world-to-object transform matrix
            Float4_struct const *object_to_world;         ///< object-to-world transform matrix
            int                 object_id;                ///< state::object_id() result
            float               meters_per_scene_unit;    ///< state::meters_per_scene_unit() result
        };

        // IRAY SDK state
        typedef mi::mdl::Shading_state_material_bitangent IRAY_State;

        llvm::StructLayout const *sl = m_data_layout.getStructLayout(res);
        if (use_bitangents()) {
            MDL_ASSERT(sl->getSizeInBytes() == sizeof(IRAY_State));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_NORMAL))
                == offsetof(IRAY_State, normal));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_GEOMETRY_NORMAL))
                == offsetof(IRAY_State, geom_normal));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_POSITION))
                == offsetof(IRAY_State, position));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_TEXTURE_COORDINATE))
                == offsetof(IRAY_State, text_coords));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_BITANGENTS))
                == offsetof(IRAY_State, tangents_bitangentssign));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_TEXT_RESULTS))
                == offsetof(IRAY_State, text_results));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_ANIMATION_TIME))
                == offsetof(IRAY_State, animation_time));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_RO_DATA_SEG))
                == offsetof(IRAY_State, ro_data_segment));

            if (state_includes_uniform_state()) {
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_W2O_TRANSFORM))
                    == offsetof(IRAY_State, world_to_object));
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_O2W_TRANSFORM))
                    == offsetof(IRAY_State, object_to_world));
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_OBJECT_ID))
                    == offsetof(IRAY_State, object_id));
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_METERS_PER_SCENE_UNIT))
                    == offsetof(IRAY_State, meters_per_scene_unit));
            }
        } else {
            MDL_ASSERT(sl->getSizeInBytes() == sizeof(SDK_State));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_NORMAL))
                == offsetof(SDK_State, normal));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_GEOMETRY_NORMAL))
                == offsetof(SDK_State, geom_normal));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_POSITION))
                == offsetof(SDK_State, position));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_TEXTURE_COORDINATE))
                == offsetof(SDK_State, text_coords));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_TANGENT_U))
                == offsetof(SDK_State, tangent_u));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_TANGENT_V))
                == offsetof(SDK_State, tangent_v));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_TEXT_RESULTS))
                == offsetof(SDK_State, text_results));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_ANIMATION_TIME))
                == offsetof(SDK_State, animation_time));
            MDL_ASSERT(
                sl->getElementOffset(get_state_index(STATE_CORE_RO_DATA_SEG))
                == offsetof(SDK_State, ro_data_segment));

            if (state_includes_uniform_state()) {
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_W2O_TRANSFORM))
                    == offsetof(SDK_State, world_to_object));
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_O2W_TRANSFORM))
                    == offsetof(SDK_State, object_to_world));
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_OBJECT_ID))
                    == offsetof(SDK_State, object_id));
                MDL_ASSERT(
                    sl->getElementOffset(get_state_index(STATE_CORE_METERS_PER_SCENE_UNIT))
                    == offsetof(SDK_State, meters_per_scene_unit));
            }
        }
    }
#endif
    return res;
}

// Construct the exception state type.
llvm::StructType *Type_mapper::construct_exception_state_type(
    llvm::LLVMContext &context,
    llvm::Type        *void_ptr_type,
    llvm::Type        *atom32_type)
{
    llvm::Type *members[] = {
        void_ptr_type,        // exception-handler
        get_ptr(atom32_type)  // abort destination
    };

    return llvm::StructType::create(context, members, "Exception_state", /*is_packed=*/false);
}

// Construct the Res_data_pair type.
llvm::StructType *Type_mapper::construct_res_data_pair_type(
    llvm::LLVMContext      &context)
{
    if (!target_supports_pointers()) {
        // generate an opaque struct
        return llvm::StructType::create(context, "Res_data");
    }

    llvm::Type *members[] = {
        m_type_void_ptr,        // shared data
        m_type_void_ptr         // thread data
    };

    llvm::StructType *res =
        llvm::StructType::create(context, members, "Res_data_pair", /*is_packed=*/false);

#if defined(DEBUG) || defined(ENABLE_ASSERT)
    {
        // check struct layout offsets and size
        // must match between LLVM layout and C++ layout from the native
        // compiler

        typedef mi::mdl::Generated_code_lambda_function::Res_data_pair Pair;
        llvm::StructLayout const *sl = m_data_layout.getStructLayout(res);
        MDL_ASSERT(sl->getSizeInBytes() ==
            sizeof(Pair));
        MDL_ASSERT(sl->getElementOffset(RDP_SHARED_DATA) ==
            offsetof(Pair, m_shared_data));
        MDL_ASSERT(sl->getElementOffset(RDP_THREAD_DATA) ==
            offsetof(Pair, m_thread_data));
    }
#endif

    return res;
}

llvm::StructType *Type_mapper::construct_exec_ctx_type(
    llvm::LLVMContext      &context,
    llvm::Type             *state_core_ptr_type,
    llvm::Type             *res_data_pair_ptr_type,
    llvm::Type             *exc_state_ptr_type,
    llvm::Type             *void_ptr_type)
{
    llvm::Type *members[] = {
        state_core_ptr_type,
        res_data_pair_ptr_type,
        exc_state_ptr_type,
        void_ptr_type,   // captured_arguments
        void_ptr_type,   // lambda_results
    };

    return llvm::StructType::create(context, members, "Execution_context", /*is_packed=*/false);
}


// Construct the texture handler vtable type.
llvm::StructType *Type_mapper::construct_core_texture_handler_type(
    llvm::LLVMContext &context)
{
    llvm::StructType *self_type            = llvm::StructType::create(context, "Core_tex_handler");
    llvm::Type *self_ptr_type              = get_ptr(self_type);
    llvm::Type *void_type                  = get_void_type();
    llvm::Type *bool_type                  = get_bool_type();
    llvm::Type *arr_float_2_ptr_type       = get_arr_float_2_ptr_type();
    llvm::Type *arr_float_3_ptr_type       = get_arr_float_3_ptr_type();
    llvm::Type *arr_float_4_ptr_type       = get_arr_float_4_ptr_type();
    llvm::Type *arr_int_2_ptr_type         = get_arr_int_2_ptr_type();
    llvm::Type *arr_int_3_ptr_type         = get_arr_int_3_ptr_type();
    llvm::Type *arr_int_4_ptr_type         = get_arr_int_4_ptr_type();
    llvm::Type *float_type                 = get_float_type();
    llvm::Type *int_type                   = get_int_type();

    llvm::FunctionType *tex_lookup_float4_2d_type   = NULL;
    llvm::FunctionType *tex_lookup_float3_2d_type   = NULL;
    llvm::FunctionType *tex_texel_float4_2d_type    = NULL;
    llvm::FunctionType *tex_lookup_float4_3d_type   = NULL;
    llvm::FunctionType *tex_lookup_float3_3d_type   = NULL;
    llvm::FunctionType *tex_texel_float4_3d_type    = NULL;
    llvm::FunctionType *tex_lookup_float4_cube_type = NULL;
    llvm::FunctionType *tex_lookup_float3_cube_type = NULL;
    llvm::FunctionType *tex_resolution_2d_type      = NULL;
    llvm::FunctionType *tex_resolution_3d_type      = NULL;
    llvm::FunctionType *tex_isvalid_type            = NULL;

    llvm::FunctionType *df_light_profile_power_type    = NULL;
    llvm::FunctionType *df_light_profile_maximum_type  = NULL;
    llvm::FunctionType *df_light_profile_isvalid_type  = NULL;
    llvm::FunctionType *df_light_profile_evaluate_type = NULL;
    llvm::FunctionType *df_light_profile_sample_type   = NULL;
    llvm::FunctionType *df_light_profile_pdf_type      = NULL;

    llvm::FunctionType *df_bsdf_measurement_isvalid_type = NULL;
    llvm::FunctionType *df_bsdf_measurement_resolution_type = NULL;
    llvm::FunctionType *df_bsdf_measurement_evaluate_type   = NULL;
    llvm::FunctionType *df_bsdf_measurement_sample_type     = NULL;
    llvm::FunctionType *df_bsdf_measurement_pdf_type        = NULL;
    llvm::FunctionType *df_bsdf_measurement_albedos_type    = NULL;

    llvm::FunctionType *scene_data_isvalid_type       = NULL;
    llvm::FunctionType *scene_data_lookup_float_type  = NULL;
    llvm::FunctionType *scene_data_lookup_float2_type = NULL;
    llvm::FunctionType *scene_data_lookup_float3_type = NULL;
    llvm::FunctionType *scene_data_lookup_float4_type = NULL;
    llvm::FunctionType *scene_data_lookup_int_type    = NULL;
    llvm::FunctionType *scene_data_lookup_int2_type   = NULL;
    llvm::FunctionType *scene_data_lookup_int3_type   = NULL;
    llvm::FunctionType *scene_data_lookup_int4_type   = NULL;
    llvm::FunctionType *scene_data_lookup_color_type  = NULL;

    llvm::FunctionType *scene_data_lookup_deriv_float_type  = NULL;
    llvm::FunctionType *scene_data_lookup_deriv_float2_type = NULL;
    llvm::FunctionType *scene_data_lookup_deriv_float3_type = NULL;
    llvm::FunctionType *scene_data_lookup_deriv_float4_type = NULL;
    llvm::FunctionType *scene_data_lookup_deriv_color_type  = NULL;

    {
        // virtual void tex_lookup_<T>_2d(
        //     T                result,
        //     Core_tex_handler *self,
        //     unsigned         texture,
        //     float const      coord[2],
        //     MDL_wrap_mode    wrap_u,
        //     MDL_wrap_mode    wrap_v,
        //     float const      crop_u[2],
        //     float const      crop_v[2]) const = 0;

        llvm::Type *args[] = {
            NULL,                   // T
            self_ptr_type,
            int_type,
            use_derivatives()
                ? get_ptr(get_deriv_arr_float_2_type())
                : arr_float_2_ptr_type,
            int_type,
            int_type,
            arr_float_2_ptr_type,
            arr_float_2_ptr_type
        };

        args[0] = arr_float_4_ptr_type;
        tex_lookup_float4_2d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);

        args[0] = arr_float_3_ptr_type;
        tex_lookup_float3_2d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_texel_float4_2d(
        //     T                result,
        //     Core_tex_handler *self,
        //     unsigned         texture,
        //     int const        coord[2],
        //     int const        uv_tile[2]) const = 0;

        llvm::Type *args[] = {
            arr_float_4_ptr_type,
            self_ptr_type,
            int_type,
            arr_int_2_ptr_type,
            arr_int_2_ptr_type
        };

        tex_texel_float4_2d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_lookup_<T>_3d(
        //     T                      result,
        //     Core_tex_handler const *self,
        //     unsigned               texture_idx,
        //     float const            coord[3],
        //     tex_wrap_mode const    wrap_u,
        //     tex_wrap_mode const    wrap_v,
        //     tex_wrap_mode const    wrap_w,
        //     float const            crop_u[2],
        //     float const            crop_v[2],
        //     float const            crop_w[2]);

        llvm::Type *args[] = {
            NULL,                   // T
            self_ptr_type,
            int_type,
            arr_float_3_ptr_type,
            int_type,
            int_type,
            int_type,
            arr_float_2_ptr_type,
            arr_float_2_ptr_type,
            arr_float_2_ptr_type
        };

        args[0] = arr_float_4_ptr_type;
        tex_lookup_float4_3d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);

        args[0] = arr_float_3_ptr_type;
        tex_lookup_float3_3d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_texel_float4_3d(
        //     float                  result[4],
        //     Core_tex_handler const *self,
        //     unsigned               texture_idx,
        //     int const              coord[3]);

        llvm::Type *args[] = {
            arr_float_4_ptr_type,
            self_ptr_type,
            int_type,
            arr_int_3_ptr_type
        };

        tex_texel_float4_3d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_lookup_float4_cube(
        //     float                  result[4],
        //     Core_tex_handler const *self,
        //     unsigned               texture_idx,
        //     float const            coord[3]);

        llvm::Type *args[] = {
            arr_float_4_ptr_type,
            self_ptr_type,
            int_type,
            arr_float_3_ptr_type
        };

        tex_lookup_float4_cube_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_lookup_float3_cube(
        //     float                  result[3],
        //     Core_tex_handler const *self,
        //     unsigned               texture_idx,
        //     float const            coord[3]);

        llvm::Type *args[] = {
            arr_float_3_ptr_type,
            self_ptr_type,
            int_type,
            arr_float_3_ptr_type
        };

        tex_lookup_float3_cube_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_resolution_2d(
        //     int              result[2],
        //     Core_tex_handler *self,
        //     unsigned         texture,
        //     int const        uv_tile[2]);

        llvm::Type *args[] = {
            arr_int_2_ptr_type,
            self_ptr_type,
            int_type,
            arr_int_2_ptr_type
        };

        tex_resolution_2d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void tex_resolution_3d(
        //     int              result[3],
        //     Core_tex_handler *self,
        //     unsigned         texture);

        llvm::Type *args[] = {
            arr_int_3_ptr_type,
            self_ptr_type,
            int_type
        };

        tex_resolution_3d_type = llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual bool tex_texture_isvalid(
        //     Core_tex_handler *self,
        //     unsigned         texture);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type
        };

        tex_isvalid_type = llvm::FunctionType::get(bool_type, args, /*isVarArg=*/false);
    }


    {
        // virtual float df_light_profile_power(
        //     Core_tex_handler const *self,
        //     unsigned               light_profile_index);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type
        };

        df_light_profile_power_type =
            llvm::FunctionType::get(float_type, args, /*isVarArg=*/false);
    }

    {
        // virtual float df_light_profile_maximum(
        //     Core_tex_handler const *self,
        //     unsigned               light_profile_index);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type
        };

        df_light_profile_maximum_type =
            llvm::FunctionType::get(float_type, args, /*isVarArg=*/false);
    }

    {
        // virtual bool df_light_profile_isvalid(
        //     Core_tex_handler *self,
        //     unsigned         light_profile_index);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type
        };

        df_light_profile_isvalid_type =
            llvm::FunctionType::get(bool_type, args, /*isVarArg=*/false);
    }

    {
        // virtual float df_light_profile_evaluate(
        //      Core_tex_handler const *self,
        //      unsigned               light_profile_index,
        //      const float            theta_phi[2]);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type,
            arr_float_2_ptr_type
        };

        df_light_profile_evaluate_type =
            llvm::FunctionType::get(float_type, args, /*isVarArg=*/false);
    }

    {
        // virtual float df_light_profile_sample(
        //      float                  result[3],
        //      Core_tex_handler const *self,
        //      unsigned               light_profile_index,
        //      const float            xi[3]);

        llvm::Type *args[] = {
            arr_float_3_ptr_type,
            self_ptr_type,
            int_type,
            arr_float_3_ptr_type
        };

        df_light_profile_sample_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual float df_light_profile_pdf(
        //      Core_tex_handler const *self,
        //      unsigned               light_profile_index,
        //      const float            theta_phi[2]);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type,
            arr_float_2_ptr_type
        };

        df_light_profile_pdf_type =
            llvm::FunctionType::get(float_type, args, /*isVarArg=*/false);
    }

    {
        // virtual bool df_bsdf_measurement_isvalid(
        //     Core_tex_handler *self,
        //     unsigned         bsdf_measurement_index);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type
        };

        df_bsdf_measurement_isvalid_type =
            llvm::FunctionType::get(bool_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void df_bsdf_measurement_resolution(
        //      int                  result[3],
        //      Core_tex_handler     *self,
        //      unsigned             bsdf_measurement_index,
        //      Mbsdf_part           part);

        llvm::Type *args[] = {
            arr_int_3_ptr_type,
            self_ptr_type,
            int_type,
            int_type
        };

        df_bsdf_measurement_resolution_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void df_bsdf_measurement_evaluate(
        //      float                  result[3],
        //      Core_tex_handler const *self,
        //      unsigned               bsdf_measurement_index,
        //      const float            theta_phi_in[2],
        //      const float            theta_phi_out[2],
        //      Mbsdf_part             part);

        llvm::Type *args[] = {
            arr_float_3_ptr_type,
            self_ptr_type,
            int_type,
            arr_float_2_ptr_type,
            arr_float_2_ptr_type,
            int_type
        };

        df_bsdf_measurement_evaluate_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void df_bsdf_measurement_sample(
        //      float                  result[3],
        //      Core_tex_handler const *self,
        //      unsigned               bsdf_measurement_index,
        //      const float            theta_phi_out[2],
        //      const float            xi[3],
        //      Mbsdf_part             part);

        llvm::Type *args[] = {
            arr_float_3_ptr_type,
            self_ptr_type,
            int_type,
            arr_float_2_ptr_type,
            arr_float_3_ptr_type,
            int_type
        };

        df_bsdf_measurement_sample_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        // virtual float df_bsdf_measurement_pdf(
        //      Core_tex_handler const *self,
        //      unsigned               bsdf_measurement_index,
        //      const float            theta_phi_in[2],
        //      const float            theta_phi_out[2],
        //      Mbsdf_part             part);

        llvm::Type *args[] = {
            self_ptr_type,
            int_type,
            arr_float_2_ptr_type,
            arr_float_2_ptr_type,
            int_type
        };

        df_bsdf_measurement_pdf_type =
            llvm::FunctionType::get(float_type, args, /*isVarArg=*/false);
    }

    {
        // virtual void df_bsdf_measurement_albedos(
        //      float                  result[4],
        //      Core_tex_handler const *self,
        //      unsigned               bsdf_measurement_index,
        //      const float            theta_phi[2]);

        llvm::Type *args[] = {
            arr_float_4_ptr_type,
            self_ptr_type,
            int_type,
            arr_float_2_ptr_type
        };

        df_bsdf_measurement_albedos_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            self_ptr_type,
            m_type_state_core_ptr,
            int_type
        };

        scene_data_isvalid_type =
            llvm::FunctionType::get(bool_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            float_type,
            bool_type
        };

        scene_data_lookup_float_type =
            llvm::FunctionType::get(float_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_float_2_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_float_2_ptr_type,
            bool_type
        };

        scene_data_lookup_float2_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_float_3_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_float_3_ptr_type,
            bool_type
        };

        scene_data_lookup_float3_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_float_4_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_float_4_ptr_type,
            bool_type
        };

        scene_data_lookup_float4_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            int_type,
            bool_type
        };

        scene_data_lookup_int_type =
            llvm::FunctionType::get(int_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_int_2_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_int_2_ptr_type,
            bool_type
        };

        scene_data_lookup_int2_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_int_3_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_int_3_ptr_type,
            bool_type
        };

        scene_data_lookup_int3_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_int_4_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_int_4_ptr_type,
            bool_type
        };

        scene_data_lookup_int4_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    {
        llvm::Type *args[] = {
            arr_float_3_ptr_type,
            self_ptr_type,
            m_type_state_core_ptr,
            int_type,
            arr_float_3_ptr_type,
            bool_type
        };

        scene_data_lookup_color_type =
            llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
    }

    // currently we support only these
    llvm::SmallVector<llvm::Type *, 38> vtable_members;
    vtable_members.append({
        get_ptr(tex_lookup_float4_2d_type),
        get_ptr(tex_lookup_float3_2d_type),
        get_ptr(tex_texel_float4_2d_type),
        get_ptr(tex_lookup_float4_3d_type),
        get_ptr(tex_lookup_float3_3d_type),
        get_ptr(tex_texel_float4_3d_type),
        get_ptr(tex_lookup_float4_cube_type),
        get_ptr(tex_lookup_float3_cube_type),
        get_ptr(tex_resolution_2d_type),
        get_ptr(tex_resolution_3d_type),
        get_ptr(tex_isvalid_type),
        get_ptr(df_light_profile_power_type),
        get_ptr(df_light_profile_maximum_type),
        get_ptr(df_light_profile_isvalid_type),
        get_ptr(df_light_profile_evaluate_type),
        get_ptr(df_light_profile_sample_type),
        get_ptr(df_light_profile_pdf_type),
        get_ptr(df_bsdf_measurement_isvalid_type),
        get_ptr(df_bsdf_measurement_resolution_type),
        get_ptr(df_bsdf_measurement_evaluate_type),
        get_ptr(df_bsdf_measurement_sample_type),
        get_ptr(df_bsdf_measurement_pdf_type),
        get_ptr(df_bsdf_measurement_albedos_type),
        get_ptr(scene_data_isvalid_type),
        get_ptr(scene_data_lookup_float_type),
        get_ptr(scene_data_lookup_float2_type),
        get_ptr(scene_data_lookup_float3_type),
        get_ptr(scene_data_lookup_float4_type),
        get_ptr(scene_data_lookup_int_type),
        get_ptr(scene_data_lookup_int2_type),
        get_ptr(scene_data_lookup_int3_type),
        get_ptr(scene_data_lookup_int4_type),
        get_ptr(scene_data_lookup_color_type),
    });

    if (use_derivatives()) {
        {
            llvm::Type *args[] = {
                get_ptr(get_deriv_float_type()),
                self_ptr_type,
                m_type_state_core_ptr,
                int_type,
                get_ptr(get_deriv_float_type()),
                bool_type
            };

            scene_data_lookup_deriv_float_type =
                llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
        }

        {
            llvm::Type *args[] = {
                get_ptr(get_deriv_arr_float_2_type()),
                self_ptr_type,
                m_type_state_core_ptr,
                int_type,
                get_ptr(get_deriv_arr_float_2_type()),
                bool_type
            };

            scene_data_lookup_deriv_float2_type =
                llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
        }

        {
            llvm::Type *args[] = {
                get_ptr(get_deriv_arr_float_3_type()),
                self_ptr_type,
                m_type_state_core_ptr,
                int_type,
                get_ptr(get_deriv_arr_float_3_type()),
                bool_type
            };

            scene_data_lookup_deriv_float3_type =
                llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
        }

        {
            llvm::Type *args[] = {
                get_ptr(get_deriv_arr_float_4_type()),
                self_ptr_type,
                m_type_state_core_ptr,
                int_type,
                get_ptr(get_deriv_arr_float_4_type()),
                bool_type
            };

            scene_data_lookup_deriv_float4_type =
                llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
        }

        {
            llvm::Type *args[] = {
                get_ptr(get_deriv_arr_float_3_type()),
                self_ptr_type,
                m_type_state_core_ptr,
                int_type,
                get_ptr(get_deriv_arr_float_3_type()),
                bool_type
            };

            scene_data_lookup_deriv_color_type =
                llvm::FunctionType::get(void_type, args, /*isVarArg=*/false);
        }

        vtable_members.append({
            get_ptr(scene_data_lookup_deriv_float_type),
            get_ptr(scene_data_lookup_deriv_float2_type),
            get_ptr(scene_data_lookup_deriv_float3_type),
            get_ptr(scene_data_lookup_deriv_float4_type),
            get_ptr(scene_data_lookup_deriv_color_type),
        });
    };

    llvm::StructType *vtable_type =
        llvm::StructType::create(context, vtable_members, "Core_th_vtable", /*is_packed=*/false);

    // the texture handler interface has only the vtable
     llvm::Type *members[] = {
         get_ptr(vtable_type)
     };

     self_type->setBody(members, /*is_packed=*/false);
     return self_type;
}

// Construct the texture attribute entry type.
llvm::StructType *Type_mapper::construct_texture_attribute_entry_type(
    llvm::LLVMContext &context)
{
    llvm::Type *members[] = {
        m_type_bool,
        m_type_int,
        m_type_int,
        m_type_int
    };

    return llvm::StructType::create(
        context, members, "Texture_attribute_entry", /*is_packed=*/false);
}

// Construct the light profile attribute entry type.
llvm::StructType *Type_mapper::construct_light_profile_attribuute_entry_type(
    llvm::LLVMContext &context)
{
    llvm::Type *members[] = {
        m_type_bool,
        m_type_float,
        m_type_float
    };

    return llvm::StructType::create(
        context, members, "Light_profile_attribute_entry", /*is_packed=*/false);
}

// Construct the bsdf measurement attribute entry type.
llvm::StructType *Type_mapper::construct_bsdf_measurement_attribuute_entry_type(
    llvm::LLVMContext &context)
{
    llvm::Type *members[] = {
        m_type_bool
    };

    return llvm::StructType::create(
        context, members, "Bsdf_measurement_attribute_entry", /*is_packed=*/false);
}

}  // mdl
}  // mi

