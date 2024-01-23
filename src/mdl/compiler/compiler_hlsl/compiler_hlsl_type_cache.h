/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_TYPE_CACHE_H
#define MDL_COMPILER_HLSL_TYPE_CACHE_H 1

#include "../compilercore/compilercore_allocator.h"
#include "../compilercore/compilercore_array_ref.h"

#include "compiler_hlsl_types.h"

namespace mi {
namespace mdl {
namespace hlsl {

class Symbol;
class Definition;

/// A Type cache is a wrapper around a type factory that simplifies the type access.
class Type_cache {
    typedef Type_function::Parameter Parameter;
    typedef Array_ref<Parameter>     Function_parameters;

    typedef Type_struct::Field       Field;
    typedef Array_ref<Field>         Fields;

public:
    /// Constructor.
    ///
    /// \param tf  the type factory
    Type_cache(Type_factory &tf)
    : error_type(tf.get_error())
    , void_type(tf.get_void())
    , bool_type(tf.get_bool())
    , bool2_type(tf.get_vector(bool_type, 2))
    , bool3_type(tf.get_vector(bool_type, 3))
    , bool4_type(tf.get_vector(bool_type, 4))
    , int_type(tf.get_int())
    , int2_type(tf.get_vector(int_type, 2))
    , int3_type(tf.get_vector(int_type, 3))
    , int4_type(tf.get_vector(int_type, 4))
    , uint_type(tf.get_uint())
    , uint2_type(tf.get_vector(uint_type, 2))
    , uint3_type(tf.get_vector(uint_type, 3))
    , uint4_type(tf.get_vector(uint_type, 4))
    , half_type(tf.get_half())
    , half2_type(tf.get_vector(half_type, 2))
    , half3_type(tf.get_vector(half_type, 3))
    , half4_type(tf.get_vector(half_type, 4))
    , float_type(tf.get_float())
    , float2_type(tf.get_vector(float_type, 2))
    , float3_type(tf.get_vector(float_type, 3))
    , float4_type(tf.get_vector(float_type, 4))
    , double_type(tf.get_double())
    , double2_type(tf.get_vector(double_type, 2))
    , double3_type(tf.get_vector(double_type, 3))
    , double4_type(tf.get_vector(double_type, 4))
    , float2x2_type(tf.get_matrix(float2_type, 2))
    , float2x3_type(tf.get_matrix(float2_type, 3))
    , float2x4_type(tf.get_matrix(float2_type, 4))
    , float3x2_type(tf.get_matrix(float3_type, 2))
    , float3x3_type(tf.get_matrix(float3_type, 3))
    , float3x4_type(tf.get_matrix(float3_type, 4))
    , float4x2_type(tf.get_matrix(float4_type, 2))
    , float4x3_type(tf.get_matrix(float4_type, 3))
    , float4x4_type(tf.get_matrix(float4_type, 4))
    , double2x2_type(tf.get_matrix(double2_type, 2))
    , double2x3_type(tf.get_matrix(double2_type, 3))
    , double2x4_type(tf.get_matrix(double2_type, 4))
    , double3x2_type(tf.get_matrix(double3_type, 2))
    , double3x3_type(tf.get_matrix(double3_type, 3))
    , double3x4_type(tf.get_matrix(double3_type, 4))
    , double4x2_type(tf.get_matrix(double4_type, 2))
    , double4x3_type(tf.get_matrix(double4_type, 3))
    , double4x4_type(tf.get_matrix(double4_type, 4))
    , m_tf(tf)
    {}

    /// Return a decorated type by adding type modifier.
    ///
    /// \param type  the base type
    /// \param mod   modifiers to ass
    Type *decorate_type(
        Type            *type,
        Type::Modifiers mod)
    {
        if (mod == Type::MK_NONE) {
            return type;
        }
        return m_tf.get_alias(type, mod);
    }

    /// Get a vector type 1instance.
    ///
    /// \param element_type The type of the vector elements.
    /// \param size         The size of the vector.
    Type_vector *get_vector(
        Type_scalar *element_type,
        size_t size)
    {
        return m_tf.get_vector(element_type, size);
    }

    /// Get a matrix type instance.
    ///
    /// \param element_type The type of the matrix elements.
    /// \param columns      The number of columns.
    Type_matrix *get_matrix(
        Type_vector *element_type,
        size_t      columns)
    {
        return m_tf.get_matrix(element_type, columns);
    }

    /// Create a new type function type instance.
    ///
    /// \param return_type   The return type of the function.
    /// \param parameters    The parameters of the function.
    Type_function *get_function(
        Type                     *return_type,
        Function_parameters const &parameters)
    {
        return m_tf.get_function(return_type, parameters);
    }

    /// Create a new type struct instance.
    ///
    /// \param fields  The fields of the new struct.
    /// \param name    The name of the struct.
    Type_struct *get_struct(
        Fields const &fields,
        Symbol       *name)
    {
        return m_tf.get_struct(fields, name);
    }

    /// Create a new array type instance.
    ///
    /// \param element_type The element type of the array.
    /// \param size         The immediate size of the array.
    ///
    /// \return IType_error if element_type was of IType_error, an IType_array instance else.
    Type *get_array(
        Type  *element_type,
        size_t size)
    {
        return m_tf.get_array(element_type, size);
    }

    /// Create a new type alias instance.
    ///
    /// \param type       The aliased type.
    /// \param modifiers  The type modifiers.
    Type *get_alias(
        Type            *type,
        Type::Modifiers modifiers)
    {
        return m_tf.get_alias(type, modifiers);
    }

    /// If a given type has an unsigned variant, return it.
    ///
    /// \param type  the type that should be converted to unsigned
    ///
    /// \return the corresponding unsigned type or NULL if such type does not exists
    Type *to_unsigned_type(Type *type)
    {
        return m_tf.to_unsigned_type(type);
    }

public:
    Type_error             * const error_type;
    Type_void              * const void_type;
    Type_bool              * const bool_type;
    Type_vector            * const bool2_type;
    Type_vector            * const bool3_type;
    Type_vector            * const bool4_type;
    Type_int               * const int_type;
    Type_vector            * const int2_type;
    Type_vector            * const int3_type;
    Type_vector            * const int4_type;
    Type_uint              * const uint_type;
    Type_vector            * const uint2_type;
    Type_vector            * const uint3_type;
    Type_vector            * const uint4_type;
    Type_half              * const half_type;
    Type_vector            * const half2_type;
    Type_vector            * const half3_type;
    Type_vector            * const half4_type;
    Type_float             * const float_type;
    Type_vector            * const float2_type;
    Type_vector            * const float3_type;
    Type_vector            * const float4_type;
    Type_double            * const double_type;
    Type_vector            * const double2_type;
    Type_vector            * const double3_type;
    Type_vector            * const double4_type;
    Type_matrix            * const float2x2_type;
    Type_matrix            * const float2x3_type;
    Type_matrix            * const float2x4_type;
    Type_matrix            * const float3x2_type;
    Type_matrix            * const float3x3_type;
    Type_matrix            * const float3x4_type;
    Type_matrix            * const float4x2_type;
    Type_matrix            * const float4x3_type;
    Type_matrix            * const float4x4_type;
    Type_matrix            * const double2x2_type;
    Type_matrix            * const double2x3_type;
    Type_matrix            * const double2x4_type;
    Type_matrix            * const double3x2_type;
    Type_matrix            * const double3x3_type;
    Type_matrix            * const double3x4_type;
    Type_matrix            * const double4x2_type;
    Type_matrix            * const double4x3_type;
    Type_matrix            * const double4x4_type;

private:
    Type_factory &m_tf;
};


}  // hlsl
}  // mdl
}  // mi

#endif
