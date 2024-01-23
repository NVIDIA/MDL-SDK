/******************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_TYPE_CACHE_H
#define MDL_COMPILER_GLSL_TYPE_CACHE_H 1

#include "../compilercore/compilercore_allocator.h"
#include "../compilercore/compilercore_array_ref.h"

#include "compiler_glsl_types.h"

namespace mi {
namespace mdl {
namespace glsl {

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
    , bvec2_type(tf.get_vector(bool_type, 2))
    , bvec3_type(tf.get_vector(bool_type, 3))
    , bvec4_type(tf.get_vector(bool_type, 4))
    , int_type(tf.get_int())
    , ivec2_type(tf.get_vector(int_type, 2))
    , ivec3_type(tf.get_vector(int_type, 3))
    , ivec4_type(tf.get_vector(int_type, 4))
    , uint_type(tf.get_uint())
    , uvec2_type(tf.get_vector(uint_type, 2))
    , uvec3_type(tf.get_vector(uint_type, 3))
    , uvec4_type(tf.get_vector(uint_type, 4))
    , atomic_uint_type(tf.get_atomic_uint())
    , half_type(tf.get_half())
    , hvec2_type(tf.get_vector(half_type, 2))
    , hvec3_type(tf.get_vector(half_type, 3))
    , hvec4_type(tf.get_vector(half_type, 4))
    , float_type(tf.get_float())
    , vec2_type(tf.get_vector(float_type, 2))
    , vec3_type(tf.get_vector(float_type, 3))
    , vec4_type(tf.get_vector(float_type, 4))
    , double_type(tf.get_double())
    , dvec2_type(tf.get_vector(double_type, 2))
    , dvec3_type(tf.get_vector(double_type, 3))
    , dvec4_type(tf.get_vector(double_type, 4))
    , mat2x2_type(tf.get_matrix(vec2_type, 2))
    , mat2x3_type(tf.get_matrix(vec2_type, 3))
    , mat2x4_type(tf.get_matrix(vec2_type, 4))
    , mat3x2_type(tf.get_matrix(vec3_type, 2))
    , mat3x3_type(tf.get_matrix(vec3_type, 3))
    , mat3x4_type(tf.get_matrix(vec3_type, 4))
    , mat4x2_type(tf.get_matrix(vec4_type, 2))
    , mat4x3_type(tf.get_matrix(vec4_type, 3))
    , mat4x4_type(tf.get_matrix(vec4_type, 4))
    , mat2_type(mat2x2_type)
    , mat3_type(mat3x3_type)
    , mat4_type(mat4x4_type)
    , dmat2x2_type(tf.get_matrix(dvec2_type, 2))
    , dmat2x3_type(tf.get_matrix(dvec2_type, 3))
    , dmat2x4_type(tf.get_matrix(dvec2_type, 4))
    , dmat3x2_type(tf.get_matrix(dvec3_type, 2))
    , dmat3x3_type(tf.get_matrix(dvec3_type, 3))
    , dmat3x4_type(tf.get_matrix(dvec3_type, 4))
    , dmat4x2_type(tf.get_matrix(dvec4_type, 2))
    , dmat4x3_type(tf.get_matrix(dvec4_type, 3))
    , dmat4x4_type(tf.get_matrix(dvec4_type, 4))
    , dmat2_type(dmat2x2_type)
    , dmat3_type(dmat3x3_type)
    , dmat4_type(dmat4x4_type)
    , sampler1D_type(tf.get_sampler(float_type, SHAPE_1D))
    , sampler2D_type(tf.get_sampler(float_type, SHAPE_2D))
    , sampler3D_type(tf.get_sampler(float_type, SHAPE_3D))
    , samplerCube_type(tf.get_sampler(float_type, SHAPE_CUBE))
    , sampler1DShadow_type(tf.get_sampler(float_type, SHAPE_1D_SHADOW))
    , sampler2DShadow_type(tf.get_sampler(float_type, SHAPE_2D_SHADOW))
    , sampler2DRect_type(tf.get_sampler(float_type, SHAPE_2D_RECT))
    , sampler2DRectShadow_type(tf.get_sampler(float_type, SHAPE_2D_RECT_SHADOW))
    , samplerExternalOES_type(tf.get_sampler(float_type, SHAPE_EXTERNAL_OES))
    , highpfloat_type(tf.get_alias(float_type, Type::MK_HIGHP))
    , int8_t_type(tf.get_int8_t())
    , i8vec2_type(tf.get_vector(int8_t_type, 2))
    , i8vec3_type(tf.get_vector(int8_t_type, 3))
    , i8vec4_type(tf.get_vector(int8_t_type, 4))
    , uint8_t_type(tf.get_uint8_t())
    , u8vec2_type(tf.get_vector(uint8_t_type, 2))
    , u8vec3_type(tf.get_vector(uint8_t_type, 3))
    , u8vec4_type(tf.get_vector(uint8_t_type, 4))
    , int16_t_type(tf.get_int16_t())
    , i16vec2_type(tf.get_vector(int16_t_type, 2))
    , i16vec3_type(tf.get_vector(int16_t_type, 3))
    , i16vec4_type(tf.get_vector(int16_t_type, 4))
    , uint16_t_type(tf.get_uint16_t())
    , u16vec2_type(tf.get_vector(uint16_t_type, 2))
    , u16vec3_type(tf.get_vector(uint16_t_type, 3))
    , u16vec4_type(tf.get_vector(uint16_t_type, 4))
    , int32_t_type(int_type)
    , i32vec2_type(ivec2_type)
    , i32vec3_type(ivec3_type)
    , i32vec4_type(ivec4_type)
    , uint32_t_type(uint_type)
    , u32vec2_type(uvec2_type)
    , u32vec3_type(uvec3_type)
    , u32vec4_type(uvec4_type)
    , int64_t_type(tf.get_int64_t())
    , i64vec2_type(tf.get_vector(int64_t_type, 2))
    , i64vec3_type(tf.get_vector(int64_t_type, 3))
    , i64vec4_type(tf.get_vector(int64_t_type, 4))
    , uint64_t_type(tf.get_uint64_t())
    , u64vec2_type(tf.get_vector(uint64_t_type, 2))
    , u64vec3_type(tf.get_vector(uint64_t_type, 3))
    , u64vec4_type(tf.get_vector(uint64_t_type, 4))
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
        size_t      size)
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

    /// Get the size of a GLSL type in bytes.
    ///
    /// \param type  the type
    size_t get_type_size(Type *type)
    {
        return m_tf.get_type_size(type);
    }

    /// Get the alignment of a GLSL type in bytes.
    ///
    /// \param type  the type
    size_t get_type_alignment(Type *type)
    {
        return m_tf.get_type_alignment(type);
    }

public:
    Type_error             * const error_type;
    Type_void              * const void_type;
    Type_bool              * const bool_type;
    Type_vector            * const bvec2_type;
    Type_vector            * const bvec3_type;
    Type_vector            * const bvec4_type;
    Type_int               * const int_type;
    Type_vector            * const ivec2_type;
    Type_vector            * const ivec3_type;
    Type_vector            * const ivec4_type;
    Type_uint              * const uint_type;
    Type_vector            * const uvec2_type;
    Type_vector            * const uvec3_type;
    Type_vector            * const uvec4_type;
    Type_atomic_uint       * const atomic_uint_type;
    Type_half              * const half_type;
    Type_vector            * const hvec2_type;
    Type_vector            * const hvec3_type;
    Type_vector            * const hvec4_type;
    Type_float             * const float_type;
    Type_vector            * const vec2_type;
    Type_vector            * const vec3_type;
    Type_vector            * const vec4_type;
    Type_double            * const double_type;
    Type_vector            * const dvec2_type;
    Type_vector            * const dvec3_type;
    Type_vector            * const dvec4_type;
    Type_matrix            * const mat2x2_type;
    Type_matrix            * const mat2x3_type;
    Type_matrix            * const mat2x4_type;
    Type_matrix            * const mat3x2_type;
    Type_matrix            * const mat3x3_type;
    Type_matrix            * const mat3x4_type;
    Type_matrix            * const mat4x2_type;
    Type_matrix            * const mat4x3_type;
    Type_matrix            * const mat4x4_type;
    Type_matrix            * const mat2_type;
    Type_matrix            * const mat3_type;
    Type_matrix            * const mat4_type;
    Type_matrix            * const dmat2x2_type;
    Type_matrix            * const dmat2x3_type;
    Type_matrix            * const dmat2x4_type;
    Type_matrix            * const dmat3x2_type;
    Type_matrix            * const dmat3x3_type;
    Type_matrix            * const dmat3x4_type;
    Type_matrix            * const dmat4x2_type;
    Type_matrix            * const dmat4x3_type;
    Type_matrix            * const dmat4x4_type;
    Type_matrix            * const dmat2_type;
    Type_matrix            * const dmat3_type;
    Type_matrix            * const dmat4_type;

    Type_sampler           * const sampler1D_type;
    Type_sampler           * const sampler2D_type;
    Type_sampler           * const sampler3D_type;
    Type_sampler           * const samplerCube_type;
    Type_sampler           * const sampler1DShadow_type;
    Type_sampler           * const sampler2DShadow_type;
    Type_sampler           * const sampler2DRect_type;
    Type_sampler           * const sampler2DRectShadow_type;
    Type_sampler           * const samplerExternalOES_type;
    Type                   * const highpfloat_type;

    Type_int8_t            * const int8_t_type;
    Type_vector            * const i8vec2_type;
    Type_vector            * const i8vec3_type;
    Type_vector            * const i8vec4_type;
    Type_uint8_t           * const uint8_t_type;
    Type_vector            * const u8vec2_type;
    Type_vector            * const u8vec3_type;
    Type_vector            * const u8vec4_type;
    Type_int16_t           * const int16_t_type;
    Type_vector            * const i16vec2_type;
    Type_vector            * const i16vec3_type;
    Type_vector            * const i16vec4_type;
    Type_uint16_t          * const uint16_t_type;
    Type_vector            * const u16vec2_type;
    Type_vector            * const u16vec3_type;
    Type_vector            * const u16vec4_type;
    Type_int32_t           * const int32_t_type;
    Type_vector            * const i32vec2_type;
    Type_vector            * const i32vec3_type;
    Type_vector            * const i32vec4_type;
    Type_uint32_t          * const uint32_t_type;
    Type_vector            * const u32vec2_type;
    Type_vector            * const u32vec3_type;
    Type_vector            * const u32vec4_type;
    Type_int64_t           * const int64_t_type;
    Type_vector            * const i64vec2_type;
    Type_vector            * const i64vec3_type;
    Type_vector            * const i64vec4_type;
    Type_uint64_t          * const uint64_t_type;
    Type_vector            * const u64vec2_type;
    Type_vector            * const u64vec3_type;
    Type_vector            * const u64vec4_type;

private:
    Type_factory &m_tf;
};


}  // glsl
}  // mdl
}  // mi

#endif
