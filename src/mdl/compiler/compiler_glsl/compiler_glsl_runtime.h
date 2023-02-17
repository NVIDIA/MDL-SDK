/******************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef ARG
/// Defines a function IN argument.
///
/// \param type  the type of the argument
/// \param name  the name of the argument
#define ARG(type, name)
#endif
#ifndef OUTARG
/// Defines a function OUT argument.
///
/// \param type  the type of the argument
/// \param name  the name of the argument
#define OUTARG(type, name)
#endif
#ifndef CVIOARG
/// Defines a function coherent volatile INOUT argument.
///
/// \param type  the type of the argument
/// \param name  the name of the argument
#define CVIOARG(type, name)
#endif
#ifndef FUNCTION
/// Defines a function.
///
/// \param ret        the return type of this function
/// \param name       the name of this method
/// \param args       the method arguments
#define FUNCTION(ret, name, args)
#endif
#ifndef CONSTRUCTOR
// Defines a constructor of a builtin type.
///
/// \param kind       either EXPLICIT, or IMPLICIT
/// \param classname  the name of the builtin struct this method belongs to
/// \param args       the constructor arguments
/// \param sema       semantics
/// \param pred       the predicate
#define CONSTRUCTOR(kind, classname, args, sema, pred)
#endif
#ifndef BLOCK
/// Defines block of with a predicate.
///
/// \param name  a unique block name
/// \param pred  the predicate
#define BLOCK(name, pred)
#endif
#ifndef BEND
/// Defines a block end.
///
/// \param name  a unique block name
#define BEND(name)
#endif
#ifndef EXTENSION
/// Defines an extension block.
///
/// \param ext  the necessary extension
#define EXTENSION(ext)
#endif
#ifndef EEND
/// Defines an extension block end.
///
/// \param ext  the necessary extension
#define EEND(ext)
#endif
#ifndef STRUCT_BEGIN
/// Defines a struct type begin.
///
/// \param name  the name of the struct
#define STRUCT_BEGIN(name)
#endif
#ifndef STRUCT_END
/// Defines a struct type end.
///
/// \param name  the name of the struct
#define STRUCT_END(name)
#endif
#ifndef FIELD
/// Defines a struct field.
///
/// \param mod   optional type modifier
/// \param type  the type of the field
/// \param name  the name of the field
#define FIELD(mod, type, name)
#endif
#ifndef UVARIABLE
/// Defines a (global) uniform variable.
///
/// \param type  the type of the variable
/// \param name  the name of the variable
#define UVARIABLE(type, name)
#endif
#ifndef HAS
/// Defines an extension check.
///
/// \param ext  the necessary extension
#define HAS(ext)
#endif
#ifndef TYPE_BEGIN
/// Defines the begin of a builtin type.
///
/// \param typename      the name of the builtin type
#define TYPE_BEGIN(typename)
#endif
#ifndef TYPE_END
/// Defines the end of a builtin type.
#define TYPE_END
#endif
#ifndef VECTOR_TYPE_BEGIN
/// Defines the begin of a builtin vector type.
///
/// \param typename      the name of the builtin vector type
#define VECTOR_TYPE_BEGIN(typename)  TYPE_BEGIN(typename)
#endif
#ifndef VECTOR_TYPE_END
/// Defines the end of a builtin vector type.
#define VECTOR_TYPE_END  TYPE_END
#endif
#ifndef MATRIX_TYPE_BEGIN
/// Defines the begin of a builtin matrix type.
///
/// \param typename      the name of the builtin matrix type
#define MATRIX_TYPE_BEGIN(typename)  TYPE_BEGIN(typename)
#endif
#ifndef MATRIX_TYPE_END
/// Defines the end of a builtin matrix type.
#define MATRIX_TYPE_END  TYPE_END
#endif

// typical predicates

//
// uint supported in GS from 3.00 and GL from 1.30
//
#define PRED_HAS_UINT   (ES && VERSION >= 300) || (!ES && VERSION >= 130)

//
// double is only supported in GL from 4.00 or with the GL_ARB_gpu_shader_fp64 extension
//
#define PRED_HAS_DOUBLE (!ES && VERSION >= 400) || HAS(GL_ARB_gpu_shader_fp64)

//
// half is only supported with the GL_NV_gpu_shader5 or GL_AMD_gpu_shader_half_float extension
//
#define PRED_HAS_HALF   HAS(GL_NV_gpu_shader5) || HAS(GL_AMD_gpu_shader_half_float)

///
/// non quadratic matrices
///
#define PRED_NON_QUADRATIC_MATRICES VERSION > 110

#define PRED_HAS_BITCAST \
    (ES && VERSION >= 300) || (!ES && VERSION >= 330) || \
    HAS(GL_ARB_gpu_shader5) || HAS(GL_ARB_shader_bit_encoding)

/*
                                Can be implicitly
        Type of expression        converted to
        --------------------    -----------------------------------------
        int                     uint, int64_t, uint64_t, float, double(*)
        ivec2                   uvec2, i64vec2, u64vec2, vec2, dvec2(*)
        ivec3                   uvec3, i64vec3, u64vec3, vec3, dvec3(*)
        ivec4                   uvec4, i64vec4, u64vec4, vec4, dvec4(*)

        int8_t   int16_t        int, int64_t, uint, uint64_t, float, double(*)
        i8vec2   i16vec2        ivec2, i64vec2, uvec2, u64vec2, vec2, dvec2(*)
        i8vec3   i16vec3        ivec3, i64vec3, uvec3, u64vec3, vec3, dvec3(*)
        i8vec4   i16vec4        ivec4, i64vec4, uvec4, u64vec4, vec4, dvec4(*)

        int64_t                 uint64_t, double(*)
        i64vec2                 u64vec2, dvec2(*)
        i64vec3                 u64vec3, dvec3(*)
        i64vec4                 u64vec4, dvec4(*)

        uint                    uint64_t, float, double(*)
        uvec2                   u64vec2, vec2, dvec2(*)
        uvec3                   u64vec3, vec3, dvec3(*)
        uvec4                   u64vec4, vec4, dvec4(*)

        uint8_t  uint16_t       uint, uint64_t, float, double(*)
        u8vec2   u16vec2        uvec2, u64vec2, vec2, dvec2(*)
        u8vec3   i16vec3        uvec3, u64vec3, vec3, dvec3(*)
        u8vec4   i16vec4        uvec4, u64vec4, vec4, dvec4(*)

        uint64_t                double(*)
        u64vec2                 dvec2(*)
        u64vec3                 dvec3(*)
        u64vec4                 dvec4(*)

        float                   double(*)
        vec2                    dvec2(*)
        vec3                    dvec3(*)
        vec4                    dvec4(*)

        float16_t               float, double(*)
        f16vec2                 vec2, dvec2(*)
        f16vec3                 vec3, dvec3(*)
        f16vec4                 vec4, dvec4(*)

        (*) if ARB_gpu_shader_fp64 is supported
*/

//
// Basic types, always available in ES and non-ES.
//
BLOCK(basic_types, true)
    TYPE_BEGIN(void)
    TYPE_END

    TYPE_BEGIN(bool)
        CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(int, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(uint, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_UINT)
        CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(half, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(float, x)),  DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, bool, ARG1(ARG(double, x)), DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END

    TYPE_BEGIN(int)
        CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(bool, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(uint, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_UINT)
        CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(half, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(float, x)),  DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, int, ARG1(ARG(double, x)), DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END

    TYPE_BEGIN(float)
        CONSTRUCTOR(IMPLICIT, float, ARG1(ARG(int, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, float, ARG1(ARG(uint, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_UINT)
        CONSTRUCTOR(IMPLICIT, float, ARG1(ARG(half, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(EXPLICIT, float, ARG1(ARG(bool, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, float, ARG1(ARG(double, x)), DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END

    VECTOR_TYPE_BEGIN(vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(vec4)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(bvec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(bvec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(bvec4)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(ivec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(ivec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(ivec4)
    VECTOR_TYPE_END

    MATRIX_TYPE_BEGIN(mat2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat4)
    MATRIX_TYPE_END

    TYPE_BEGIN(sampler2D)
    TYPE_END

    TYPE_BEGIN(samplerCube)
    TYPE_END
BEND(basic_types)

//
// basic non-ES types
//
BLOCK(basic_types_non_es, !ES)

    TYPE_BEGIN(sampler1D)
    TYPE_END

    TYPE_BEGIN(sampler3D)
    TYPE_END

    TYPE_BEGIN(sampler1DShadow)
    TYPE_END

    TYPE_BEGIN(sampler2DShadow)
    TYPE_END
BEND(basic_types_non_es)

//
// non-quadratic matrices
//
BLOCK(non_quadratic_matrices_types, PRED_NON_QUADRATIC_MATRICES)

    MATRIX_TYPE_BEGIN(mat2x2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat2x3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat2x4)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat3x2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat3x3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat3x4)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat4x2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat4x3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(mat4x4)
    MATRIX_TYPE_END

BEND(non_quadratic_matrices_types)

//
// uint supported in GS from 3.00 and GL from 1.30
//
BLOCK(uint_types, PRED_HAS_UINT)
    TYPE_BEGIN(uint)
        CONSTRUCTOR(IMPLICIT, uint, ARG1(ARG(int, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, uint, ARG1(ARG(bool, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, uint, ARG1(ARG(half, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(EXPLICIT, uint, ARG1(ARG(float, x)),  DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, uint, ARG1(ARG(double, x)), DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END

    VECTOR_TYPE_BEGIN(uvec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(uvec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(uvec4)
    VECTOR_TYPE_END

BEND(uint_types)

//
// half is only supported with the GL_NV_gpu_shader5 or GL_AMD_gpu_shader_half_float extension
//
BLOCK(half_types, PRED_HAS_HALF)
    TYPE_BEGIN(half)
        CONSTRUCTOR(EXPLICIT, half, ARG1(ARG(int, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, half, ARG1(ARG(uint, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, half, ARG1(ARG(bool, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, half, ARG1(ARG(float, x)),  DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, half, ARG1(ARG(double, x)), DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END
BEND(half_types)

//
// double is only supported in GL from 4.00 or with the GL_ARB_gpu_shader_fp64 extension
//
BLOCK(double_types, PRED_HAS_DOUBLE)
    TYPE_BEGIN(double)
        CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(int, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(uint, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(half, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(IMPLICIT, double, ARG1(ARG(float, x)),  DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, double, ARG1(ARG(bool, x)),   DS_CONV_CONSTRUCTOR, true)
    TYPE_END

    VECTOR_TYPE_BEGIN(dvec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(dvec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(dvec4)
    VECTOR_TYPE_END

    MATRIX_TYPE_BEGIN(dmat2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat4)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat2x2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat2x3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat2x4)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat3x2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat3x3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat3x4)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat4x2)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat4x3)
    MATRIX_TYPE_END

    MATRIX_TYPE_BEGIN(dmat4x4)
    MATRIX_TYPE_END

BEND(double_types)

//
// 64bit integer types
//
BLOCK(64bit_int_types, HAS_64BIT_INT_TYPES)
    TYPE_BEGIN(int64_t)
        CONSTRUCTOR(IMPLICIT, int64_t, ARG1(ARG(bool, x)),     DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, int64_t, ARG1(ARG(int, x)),      DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, int64_t, ARG1(ARG(uint, x)),     DS_CONV_CONSTRUCTOR, PRED_HAS_UINT)
        CONSTRUCTOR(IMPLICIT, int64_t, ARG1(ARG(uint64_t, x)), DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, int64_t, ARG1(ARG(half, x)),     DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(EXPLICIT, int64_t, ARG1(ARG(float, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, int64_t, ARG1(ARG(double, x)),   DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END

    VECTOR_TYPE_BEGIN(i64vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i64vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i64vec4)
    VECTOR_TYPE_END

    TYPE_BEGIN(uint64_t)
        CONSTRUCTOR(IMPLICIT, uint64_t, ARG1(ARG(bool, x)),    DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, uint64_t, ARG1(ARG(int, x)),     DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(IMPLICIT, uint64_t, ARG1(ARG(uint, x)),    DS_CONV_CONSTRUCTOR, PRED_HAS_UINT)
        CONSTRUCTOR(IMPLICIT, uint64_t, ARG1(ARG(int64_t, x)), DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, uint64_t, ARG1(ARG(half, x)),    DS_CONV_CONSTRUCTOR, PRED_HAS_HALF)
        CONSTRUCTOR(EXPLICIT, uint64_t, ARG1(ARG(float, x)),   DS_CONV_CONSTRUCTOR, true)
        CONSTRUCTOR(EXPLICIT, uint64_t, ARG1(ARG(double, x)),  DS_CONV_CONSTRUCTOR, PRED_HAS_DOUBLE)
    TYPE_END

    VECTOR_TYPE_BEGIN(u64vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u64vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u64vec4)
    VECTOR_TYPE_END
BEND(64bit_int_types)

//
// explicit sized integer types
//
BLOCK(explicit_size_int_types, HAS_EXPLICIT_SIZED_INT_TYPES)
    TYPE_BEGIN(int8_t)
    TYPE_END

    VECTOR_TYPE_BEGIN(i8vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i8vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i8vec4)
    VECTOR_TYPE_END

    TYPE_BEGIN(int16_t)
    TYPE_END

    VECTOR_TYPE_BEGIN(i16vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i16vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i16vec4)
    VECTOR_TYPE_END

    TYPE_BEGIN(int32_t)
    TYPE_END

    VECTOR_TYPE_BEGIN(i32vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i32vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(i32vec4)
    VECTOR_TYPE_END

    TYPE_BEGIN(uint8_t)
    TYPE_END

    VECTOR_TYPE_BEGIN(u8vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u8vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u8vec4)
    VECTOR_TYPE_END

    TYPE_BEGIN(uint16_t)
    TYPE_END

    VECTOR_TYPE_BEGIN(u16vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u16vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u16vec4)
    VECTOR_TYPE_END

    TYPE_BEGIN(uint32_t)
    TYPE_END

    VECTOR_TYPE_BEGIN(u32vec2)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u32vec3)
    VECTOR_TYPE_END

    VECTOR_TYPE_BEGIN(u32vec4)
    VECTOR_TYPE_END
BEND(explicit_size_int_types)

//
// Angle and Trigonometric Functions.
//
BLOCK(angle_and_trigonomic, true)

    FUNCTION(float, radians, ARG1(ARG(float, degrees)))
    FUNCTION(vec2,  radians, ARG1(ARG(vec2,  degrees)))
    FUNCTION(vec3,  radians, ARG1(ARG(vec3,  degrees)))
    FUNCTION(vec4,  radians, ARG1(ARG(vec4,  degrees)))

    FUNCTION(float, degrees, ARG1(ARG(float, radians)))
    FUNCTION(vec2,  degrees, ARG1(ARG(vec2,  radians)))
    FUNCTION(vec3,  degrees, ARG1(ARG(vec3,  radians)))
    FUNCTION(vec4,  degrees, ARG1(ARG(vec4,  radians)))

    FUNCTION(float, sin, ARG1(ARG(float, angle)))
    FUNCTION(vec2,  sin, ARG1(ARG(vec2,  angle)))
    FUNCTION(vec3,  sin, ARG1(ARG(vec3,  angle)))
    FUNCTION(vec4,  sin, ARG1(ARG(vec4,  angle)))

    FUNCTION(float, cos, ARG1(ARG(float, angle)))
    FUNCTION(vec2,  cos, ARG1(ARG(vec2,  angle)))
    FUNCTION(vec3,  cos, ARG1(ARG(vec3,  angle)))
    FUNCTION(vec4,  cos, ARG1(ARG(vec4,  angle)))

    FUNCTION(float, tan, ARG1(ARG(float, angle)))
    FUNCTION(vec2,  tan, ARG1(ARG(vec2,  angle)))
    FUNCTION(vec3,  tan, ARG1(ARG(vec3,  angle)))
    FUNCTION(vec4,  tan, ARG1(ARG(vec4,  angle)))

    FUNCTION(float, asin, ARG1(ARG(float, x)))
    FUNCTION(vec2,  asin, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  asin, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  asin, ARG1(ARG(vec4,  x)))

    FUNCTION(float, acos, ARG1(ARG(float, x)))
    FUNCTION(vec2,  acos, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  acos, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  acos, ARG1(ARG(vec4,  x)))

    FUNCTION(float, atan, ARG2(ARG(float, y), ARG(float, x)))
    FUNCTION(vec2,  atan, ARG2(ARG(vec2,  y), ARG(vec2,  x)))
    FUNCTION(vec3,  atan, ARG2(ARG(vec3,  y), ARG(vec3,  x)))
    FUNCTION(vec4,  atan, ARG2(ARG(vec4,  y), ARG(vec4,  x)))

    FUNCTION(float, atan, ARG1(ARG(float, y_over_x)))
    FUNCTION(vec2,  atan, ARG1(ARG(vec2,  y_over_x)))
    FUNCTION(vec3,  atan, ARG1(ARG(vec3,  y_over_x)))
    FUNCTION(vec4,  atan, ARG1(ARG(vec4,  y_over_x)))

BEND(angle_and_trigonomic)

BLOCK(angle_and_trigonomic_130, VERSION >= 130)

    FUNCTION(float, sinh, ARG1(ARG(float, angle)))
    FUNCTION(vec2,  sinh, ARG1(ARG(vec2,  angle)))
    FUNCTION(vec3,  sinh, ARG1(ARG(vec3,  angle)))
    FUNCTION(vec4,  sinh, ARG1(ARG(vec4,  angle)))

    FUNCTION(float, cosh, ARG1(ARG(float, angle)))
    FUNCTION(vec2,  cosh, ARG1(ARG(vec2,  angle)))
    FUNCTION(vec3,  cosh, ARG1(ARG(vec3,  angle)))
    FUNCTION(vec4,  cosh, ARG1(ARG(vec4,  angle)))

    FUNCTION(float, tanh, ARG1(ARG(float, angle)))
    FUNCTION(vec2,  tanh, ARG1(ARG(vec2,  angle)))
    FUNCTION(vec3,  tanh, ARG1(ARG(vec3,  angle)))
    FUNCTION(vec4,  tanh, ARG1(ARG(vec4,  angle)))

    FUNCTION(float, asinh, ARG1(ARG(float, x)))
    FUNCTION(vec2,  asinh, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  asinh, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  asinh, ARG1(ARG(vec4,  x)))

    FUNCTION(float, acosh, ARG1(ARG(float, x)))
    FUNCTION(vec2,  acosh, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  acosh, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  acosh, ARG1(ARG(vec4,  x)))

    FUNCTION(float, atanh, ARG1(ARG(float, y_over_x)))
    FUNCTION(vec2,  atanh, ARG1(ARG(vec2,  y_over_x)))
    FUNCTION(vec3,  atanh, ARG1(ARG(vec3,  y_over_x)))
    FUNCTION(vec4,  atanh, ARG1(ARG(vec4,  y_over_x)))

BEND(angle_and_trigonomic_130)

//
// Exponential Functions.
//
BLOCK(exponential, true)

    FUNCTION(float, pow, ARG2(ARG(float, x), ARG(float, y)))
    FUNCTION(vec2,  pow, ARG2(ARG(vec2,  x), ARG(vec2,  y)))
    FUNCTION(vec3,  pow, ARG2(ARG(vec3,  x), ARG(vec3,  y)))
    FUNCTION(vec4,  pow, ARG2(ARG(vec4,  x), ARG(vec4,  y)))

    FUNCTION(float, exp, ARG1(ARG(float, x)))
    FUNCTION(vec2,  exp, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  exp, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  exp, ARG1(ARG(vec4,  x)))

    FUNCTION(float, log, ARG1(ARG(float, x)))
    FUNCTION(vec2,  log, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  log, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  log, ARG1(ARG(vec4,  x)))

    FUNCTION(float, exp2, ARG1(ARG(float, x)))
    FUNCTION(vec2,  exp2, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  exp2, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  exp2, ARG1(ARG(vec4,  x)))

    FUNCTION(float, log2, ARG1(ARG(float, x)))
    FUNCTION(vec2,  log2, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  log2, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  log2, ARG1(ARG(vec4,  x)))

    FUNCTION(float, sqrt, ARG1(ARG(float, x)))
    FUNCTION(vec2,  sqrt, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  sqrt, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  sqrt, ARG1(ARG(vec4,  x)))

    FUNCTION(float, inversesqrt, ARG1(ARG(float, x)))
    FUNCTION(vec2,  inversesqrt, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  inversesqrt, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  inversesqrt, ARG1(ARG(vec4,  x)))

BEND(exponential)

BLOCK(exponential_double, PRED_HAS_DOUBLE)

    FUNCTION(double, sqrt, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  sqrt, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  sqrt, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  sqrt, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, inversesqrt, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  inversesqrt, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  inversesqrt, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  inversesqrt, ARG1(ARG(dvec4,  x)))

BEND(exponential_double)

//
// Common Functions.
//
BLOCK(common, true)

    FUNCTION(float, abs, ARG1(ARG(float, x)))
    FUNCTION(vec2,  abs, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  abs, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  abs, ARG1(ARG(vec4,  x)))

    FUNCTION(float, sign, ARG1(ARG(float, x)))
    FUNCTION(vec2,  sign, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  sign, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  sign, ARG1(ARG(vec4,  x)))

    FUNCTION(float, floor, ARG1(ARG(float, x)))
    FUNCTION(vec2,  floor, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  floor, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  floor, ARG1(ARG(vec4,  x)))

    FUNCTION(float, ceil, ARG1(ARG(float, x)))
    FUNCTION(vec2,  ceil, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  ceil, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  ceil, ARG1(ARG(vec4,  x)))

    FUNCTION(float, fract, ARG1(ARG(float, x)))
    FUNCTION(vec2,  fract, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  fract, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  fract, ARG1(ARG(vec4,  x)))

    FUNCTION(float, mod, ARG2(ARG(float, x), ARG(float, y)))
    FUNCTION(vec2,  mod, ARG2(ARG(vec2,  x), ARG(float, y)))
    FUNCTION(vec3,  mod, ARG2(ARG(vec3,  x), ARG(float, y)))
    FUNCTION(vec4,  mod, ARG2(ARG(vec4,  x), ARG(float, y)))
    FUNCTION(vec2,  mod, ARG2(ARG(vec2,  x), ARG(vec2,  y)))
    FUNCTION(vec3,  mod, ARG2(ARG(vec3,  x), ARG(vec3,  y)))
    FUNCTION(vec4,  mod, ARG2(ARG(vec4,  x), ARG(vec4,  y)))

    FUNCTION(float, min, ARG2(ARG(float, x), ARG(float, y)))
    FUNCTION(vec2,  min, ARG2(ARG(vec2,  x), ARG(float, y)))
    FUNCTION(vec3,  min, ARG2(ARG(vec3,  x), ARG(float, y)))
    FUNCTION(vec4,  min, ARG2(ARG(vec4,  x), ARG(float, y)))
    FUNCTION(vec2,  min, ARG2(ARG(vec2,  x), ARG(vec2,  y)))
    FUNCTION(vec3,  min, ARG2(ARG(vec3,  x), ARG(vec3,  y)))
    FUNCTION(vec4,  min, ARG2(ARG(vec4,  x), ARG(vec4,  y)))

    FUNCTION(float, max, ARG2(ARG(float, x), ARG(float, y)))
    FUNCTION(vec2,  max, ARG2(ARG(vec2,  x), ARG(float, y)))
    FUNCTION(vec3,  max, ARG2(ARG(vec3,  x), ARG(float, y)))
    FUNCTION(vec4,  max, ARG2(ARG(vec4,  x), ARG(float, y)))
    FUNCTION(vec2,  max, ARG2(ARG(vec2,  x), ARG(vec2,  y)))
    FUNCTION(vec3,  max, ARG2(ARG(vec3,  x), ARG(vec3,  y)))
    FUNCTION(vec4,  max, ARG2(ARG(vec4,  x), ARG(vec4,  y)))

    FUNCTION(float, clamp, ARG3(ARG(float, x), ARG(float, minVal), ARG(float, maxVal)))
    FUNCTION(vec2,  clamp, ARG3(ARG(vec2,  x), ARG(float, minVal), ARG(float, maxVal)))
    FUNCTION(vec3,  clamp, ARG3(ARG(vec3,  x), ARG(float, minVal), ARG(float, maxVal)))
    FUNCTION(vec4,  clamp, ARG3(ARG(vec4,  x), ARG(float, minVal), ARG(float, maxVal)))
    FUNCTION(vec2,  clamp, ARG3(ARG(vec2,  x), ARG(vec2,  minVal), ARG(vec2,  maxVal)))
    FUNCTION(vec3,  clamp, ARG3(ARG(vec3,  x), ARG(vec3,  minVal), ARG(vec3,  maxVal)))
    FUNCTION(vec4,  clamp, ARG3(ARG(vec4,  x), ARG(vec4,  minVal), ARG(vec4,  maxVal)))

    FUNCTION(float, mix, ARG3(ARG(float, x), ARG(float, y), ARG(float, a)))
    FUNCTION(vec2,  mix, ARG3(ARG(vec2,  x), ARG(vec2,  y), ARG(float, a)))
    FUNCTION(vec3,  mix, ARG3(ARG(vec3,  x), ARG(vec3,  y), ARG(float, a)))
    FUNCTION(vec4,  mix, ARG3(ARG(vec4,  x), ARG(vec4,  y), ARG(float, a)))
    FUNCTION(vec2,  mix, ARG3(ARG(vec2,  x), ARG(vec2,  y), ARG(vec2,  a)))
    FUNCTION(vec3,  mix, ARG3(ARG(vec3,  x), ARG(vec3,  y), ARG(vec3,  a)))
    FUNCTION(vec4,  mix, ARG3(ARG(vec4,  x), ARG(vec4,  y), ARG(vec4,  a)))

    FUNCTION(float, step, ARG2(ARG(float, edge), ARG(float, x)))
    FUNCTION(vec2,  step, ARG2(ARG(float, edge), ARG(vec2,  x)))
    FUNCTION(vec3,  step, ARG2(ARG(float, edge), ARG(vec3,  x)))
    FUNCTION(vec4,  step, ARG2(ARG(float, edge), ARG(vec4,  x)))
    FUNCTION(vec2,  step, ARG2(ARG(vec2,  edge), ARG(vec2,  x)))
    FUNCTION(vec3,  step, ARG2(ARG(vec3,  edge), ARG(vec3,  x)))
    FUNCTION(vec4,  step, ARG2(ARG(vec4,  edge), ARG(vec4,  x)))

    FUNCTION(float, smoothstep, ARG3(ARG(float, edge0), ARG(float, edge1), ARG(float, x)))
    FUNCTION(vec2,  smoothstep, ARG3(ARG(float, edge0), ARG(float, edge1), ARG(vec2,  x)))
    FUNCTION(vec3,  smoothstep, ARG3(ARG(float, edge0), ARG(float, edge1), ARG(vec3,  x)))
    FUNCTION(vec4,  smoothstep, ARG3(ARG(float, edge0), ARG(float, edge1), ARG(vec4,  x)))
    FUNCTION(vec2,  smoothstep, ARG3(ARG(vec2,  edge0), ARG(vec2,  edge1), ARG(vec2,  x)))
    FUNCTION(vec3,  smoothstep, ARG3(ARG(vec3,  edge0), ARG(vec3,  edge1), ARG(vec3,  x)))
    FUNCTION(vec4,  smoothstep, ARG3(ARG(vec4,  edge0), ARG(vec4,  edge1), ARG(vec4,  x)))

BEND(common)

BLOCK(common_130, VERSION >= 130)

    FUNCTION(int,   abs, ARG1(ARG(int,   x)))
    FUNCTION(ivec2, abs, ARG1(ARG(ivec2, x)))
    FUNCTION(ivec3, abs, ARG1(ARG(ivec3, x)))
    FUNCTION(ivec4, abs, ARG1(ARG(ivec4, x)))

    FUNCTION(int,   sign, ARG1(ARG(int,   x)))
    FUNCTION(ivec2, sign, ARG1(ARG(ivec2, x)))
    FUNCTION(ivec3, sign, ARG1(ARG(ivec3, x)))
    FUNCTION(ivec4, sign, ARG1(ARG(ivec4, x)))

    FUNCTION(float, trunc, ARG1(ARG(float, x)))
    FUNCTION(vec2,  trunc, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  trunc, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  trunc, ARG1(ARG(vec4,  x)))

    FUNCTION(float, round, ARG1(ARG(float, x)))
    FUNCTION(vec2,  round, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  round, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  round, ARG1(ARG(vec4,  x)))

    FUNCTION(float, roundEven, ARG1(ARG(float, x)))
    FUNCTION(vec2,  roundEven, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  roundEven, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  roundEven, ARG1(ARG(vec4,  x)))

    FUNCTION(float, modf, ARG2(ARG(float, x), OUTARG(float, i)))
    FUNCTION(vec2,  modf, ARG2(ARG(vec2,  x), OUTARG(vec2,  i)))
    FUNCTION(vec3,  modf, ARG2(ARG(vec3,  x), OUTARG(vec3,  i)))
    FUNCTION(vec4,  modf, ARG2(ARG(vec4,  x), OUTARG(vec4,  i)))

    FUNCTION(int,   min, ARG2(ARG(int,   x), ARG(int,   y)))
    FUNCTION(ivec2, min, ARG2(ARG(ivec2, x), ARG(int,   y)))
    FUNCTION(ivec3, min, ARG2(ARG(ivec3, x), ARG(int,   y)))
    FUNCTION(ivec4, min, ARG2(ARG(ivec4, x), ARG(int,   y)))
    FUNCTION(ivec2, min, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(ivec3, min, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(ivec4, min, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION( uint, min, ARG2(ARG(uint,  x), ARG(uint,  y)))
    FUNCTION(uvec2, min, ARG2(ARG(uvec2, x), ARG(uint,  y)))
    FUNCTION(uvec3, min, ARG2(ARG(uvec3, x), ARG(uint,  y)))
    FUNCTION(uvec4, min, ARG2(ARG(uvec4, x), ARG(uint,  y)))
    FUNCTION(uvec2, min, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(uvec3, min, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(uvec4, min, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(int,   max, ARG2(ARG(int,   x), ARG(int,   y)))
    FUNCTION(ivec2, max, ARG2(ARG(ivec2, x), ARG(int,   y)))
    FUNCTION(ivec3, max, ARG2(ARG(ivec3, x), ARG(int,   y)))
    FUNCTION(ivec4, max, ARG2(ARG(ivec4, x), ARG(int,   y)))
    FUNCTION(ivec2, max, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(ivec3, max, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(ivec4, max, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION( uint, max, ARG2(ARG(uint,  x), ARG(uint,  y)))
    FUNCTION(uvec2, max, ARG2(ARG(uvec2, x), ARG(uint,  y)))
    FUNCTION(uvec3, max, ARG2(ARG(uvec3, x), ARG(uint,  y)))
    FUNCTION(uvec4, max, ARG2(ARG(uvec4, x), ARG(uint,  y)))
    FUNCTION(uvec2, max, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(uvec3, max, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(uvec4, max, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(int,    clamp, ARG3(ARG(int,   x), ARG(int,    minVal), ARG(int,    maxVal)))
    FUNCTION(ivec2,  clamp, ARG3(ARG(ivec2, x), ARG(int,    minVal), ARG(int,    maxVal)))
    FUNCTION(ivec3,  clamp, ARG3(ARG(ivec3, x), ARG(int,    minVal), ARG(int,    maxVal)))
    FUNCTION(ivec4,  clamp, ARG3(ARG(ivec4, x), ARG(int,    minVal), ARG(int,    maxVal)))
    FUNCTION(ivec2,  clamp, ARG3(ARG(ivec2, x), ARG(ivec2,  minVal), ARG(ivec2,  maxVal)))
    FUNCTION(ivec3,  clamp, ARG3(ARG(ivec3, x), ARG(ivec3,  minVal), ARG(ivec3,  maxVal)))
    FUNCTION(ivec4,  clamp, ARG3(ARG(ivec4, x), ARG(ivec4,  minVal), ARG(ivec4,  maxVal)))

    FUNCTION(uint,   clamp, ARG3(ARG(uint,  x), ARG(uint,   minVal), ARG(uint,   maxVal)))
    FUNCTION(uvec2,  clamp, ARG3(ARG(uvec2, x), ARG(uint,   minVal), ARG(uint,   maxVal)))
    FUNCTION(uvec3,  clamp, ARG3(ARG(uvec3, x), ARG(uint,   minVal), ARG(uint,   maxVal)))
    FUNCTION(uvec4,  clamp, ARG3(ARG(uvec4, x), ARG(uint,   minVal), ARG(uint,   maxVal)))
    FUNCTION(uvec2,  clamp, ARG3(ARG(uvec2, x), ARG(uvec2,  minVal), ARG(uvec2,  maxVal)))
    FUNCTION(uvec3,  clamp, ARG3(ARG(uvec3, x), ARG(uvec3,  minVal), ARG(uvec3,  maxVal)))
    FUNCTION(uvec4,  clamp, ARG3(ARG(uvec4, x), ARG(uvec4,  minVal), ARG(uvec4,  maxVal)))

    FUNCTION(float, mix, ARG3(ARG(float, x), ARG(float, y), ARG(bool,  a)))
    FUNCTION(vec2,  mix, ARG3(ARG(vec2,  x), ARG(vec2,  y), ARG(bvec2, a)))
    FUNCTION(vec3,  mix, ARG3(ARG(vec3,  x), ARG(vec3,  y), ARG(bvec3, a)))
    FUNCTION(vec4,  mix, ARG3(ARG(vec4,  x), ARG(vec4,  y), ARG(bvec4, a)))

    FUNCTION(bool,  isnan, ARG1(ARG(float, x)))
    FUNCTION(bvec2, isnan, ARG1(ARG(vec2,  x)))
    FUNCTION(bvec3, isnan, ARG1(ARG(vec3,  x)))
    FUNCTION(bvec4, isnan, ARG1(ARG(vec4,  x)))

    FUNCTION(bool,  isinf, ARG1(ARG(float, x)))
    FUNCTION(bvec2, isinf, ARG1(ARG(vec2,  x)))
    FUNCTION(bvec3, isinf, ARG1(ARG(vec3,  x)))
    FUNCTION(bvec4, isinf, ARG1(ARG(vec4,  x)))

BEND(common_130)

BLOCK(common_double, PRED_HAS_DOUBLE)

    FUNCTION(double, abs, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  abs, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  abs, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  abs, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, sign, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  sign, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  sign, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  sign, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, floor, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  floor, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  floor, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  floor, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, trunc, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  trunc, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  trunc, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  trunc, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, round, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  round, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  round, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  round, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, roundEven, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  roundEven, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  roundEven, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  roundEven, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, ceil, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  ceil, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  ceil, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  ceil, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, fract, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  fract, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  fract, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  fract, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, mod, ARG2(ARG(double, x), ARG(double, y)))
    FUNCTION(dvec2,  mod, ARG2(ARG(dvec2,  x), ARG(double, y)))
    FUNCTION(dvec3,  mod, ARG2(ARG(dvec3,  x), ARG(double, y)))
    FUNCTION(dvec4,  mod, ARG2(ARG(dvec4,  x), ARG(double, y)))
    FUNCTION(dvec2,  mod, ARG2(ARG(dvec2,  x), ARG(dvec2,  y)))
    FUNCTION(dvec3,  mod, ARG2(ARG(dvec3,  x), ARG(dvec3,  y)))
    FUNCTION(dvec4,  mod, ARG2(ARG(dvec4,  x), ARG(dvec4,  y)))

    FUNCTION(double, modf, ARG2(ARG(double, x), OUTARG(double, i)))
    FUNCTION(dvec2,  modf, ARG2(ARG(dvec2,  x), OUTARG(dvec2,  i)))
    FUNCTION(dvec3,  modf, ARG2(ARG(dvec3,  x), OUTARG(dvec3,  i)))
    FUNCTION(dvec4,  modf, ARG2(ARG(dvec4,  x), OUTARG(dvec4,  i)))

    FUNCTION(double, min, ARG2(ARG(double, x), ARG(double, y)))
    FUNCTION(dvec2,  min, ARG2(ARG(dvec2,  x), ARG(double, y)))
    FUNCTION(dvec3,  min, ARG2(ARG(dvec3,  x), ARG(double, y)))
    FUNCTION(dvec4,  min, ARG2(ARG(dvec4,  x), ARG(double, y)))
    FUNCTION(dvec2,  min, ARG2(ARG(dvec2,  x), ARG(dvec2,  y)))
    FUNCTION(dvec3,  min, ARG2(ARG(dvec3,  x), ARG(dvec3,  y)))
    FUNCTION(dvec4,  min, ARG2(ARG(dvec4,  x), ARG(dvec4,  y)))

    FUNCTION(double, max, ARG2(ARG(double, x), ARG(double, y)))
    FUNCTION(dvec2,  max, ARG2(ARG(dvec2,  x), ARG(double, y)))
    FUNCTION(dvec3,  max, ARG2(ARG(dvec3,  x), ARG(double, y)))
    FUNCTION(dvec4,  max, ARG2(ARG(dvec4,  x), ARG(double, y)))
    FUNCTION(dvec2,  max, ARG2(ARG(dvec2,  x), ARG(dvec2,  y)))
    FUNCTION(dvec3,  max, ARG2(ARG(dvec3,  x), ARG(dvec3,  y)))
    FUNCTION(dvec4,  max, ARG2(ARG(dvec4,  x), ARG(dvec4,  y)))

    FUNCTION(double, clamp, ARG3(ARG(double, x), ARG(double, minVal), ARG(double, maxVal)))
    FUNCTION(dvec2,  clamp, ARG3(ARG(dvec2,  x), ARG(double, minVal), ARG(double, maxVal)))
    FUNCTION(dvec3,  clamp, ARG3(ARG(dvec3,  x), ARG(double, minVal), ARG(double, maxVal)))
    FUNCTION(dvec4,  clamp, ARG3(ARG(dvec4,  x), ARG(double, minVal), ARG(double, maxVal)))
    FUNCTION(dvec2,  clamp, ARG3(ARG(dvec2,  x), ARG(dvec2,  minVal), ARG(dvec2,  maxVal)))
    FUNCTION(dvec3,  clamp, ARG3(ARG(dvec3,  x), ARG(dvec3,  minVal), ARG(dvec3,  maxVal)))
    FUNCTION(dvec4,  clamp, ARG3(ARG(dvec4,  x), ARG(dvec4,  minVal), ARG(dvec4,  maxVal)))

    FUNCTION(double, mix, ARG3(ARG(double, x), ARG(double, y), ARG(double, a)))
    FUNCTION(dvec2,  mix, ARG3(ARG(dvec2,  x), ARG(dvec2,  y), ARG(double, a)))
    FUNCTION(dvec3,  mix, ARG3(ARG(dvec3,  x), ARG(dvec3,  y), ARG(double, a)))
    FUNCTION(dvec4,  mix, ARG3(ARG(dvec4,  x), ARG(dvec4,  y), ARG(double, a)))
    FUNCTION(dvec2,  mix, ARG3(ARG(dvec2,  x), ARG(dvec2,  y), ARG(dvec2,  a)))
    FUNCTION(dvec3,  mix, ARG3(ARG(dvec3,  x), ARG(dvec3,  y), ARG(dvec3,  a)))
    FUNCTION(dvec4,  mix, ARG3(ARG(dvec4,  x), ARG(dvec4,  y), ARG(dvec4,  a)))

    FUNCTION(double, step, ARG2(ARG(double, edge), ARG(double, x)))
    FUNCTION(dvec2,  step, ARG2(ARG(double, edge), ARG(dvec2,  x)))
    FUNCTION(dvec3,  step, ARG2(ARG(double, edge), ARG(dvec3,  x)))
    FUNCTION(dvec4,  step, ARG2(ARG(double, edge), ARG(dvec4,  x)))
    FUNCTION(dvec2,  step, ARG2(ARG(dvec2,  edge), ARG(dvec2,  x)))
    FUNCTION(dvec3,  step, ARG2(ARG(dvec3,  edge), ARG(dvec3,  x)))
    FUNCTION(dvec4,  step, ARG2(ARG(dvec4,  edge), ARG(dvec4,  x)))

    FUNCTION(double, smoothstep, ARG3(ARG(double, edge0), ARG(double, edge1), ARG(double, x)))
    FUNCTION(dvec2,  smoothstep, ARG3(ARG(double, edge0), ARG(double, edge1), ARG(dvec2,  x)))
    FUNCTION(dvec3,  smoothstep, ARG3(ARG(double, edge0), ARG(double, edge1), ARG(dvec3,  x)))
    FUNCTION(dvec4,  smoothstep, ARG3(ARG(double, edge0), ARG(double, edge1), ARG(dvec4,  x)))
    FUNCTION(dvec2,  smoothstep, ARG3(ARG(dvec2,  edge0), ARG(dvec2,  edge1), ARG(dvec2,  x)))
    FUNCTION(dvec3,  smoothstep, ARG3(ARG(dvec3,  edge0), ARG(dvec3,  edge1), ARG(dvec3,  x)))
    FUNCTION(dvec4,  smoothstep, ARG3(ARG(dvec4,  edge0), ARG(dvec4,  edge1), ARG(dvec4,  x)))

    FUNCTION(bool,  isnan, ARG1(ARG(double, x)))
    FUNCTION(bvec2, isnan, ARG1(ARG(dvec2,  x)))
    FUNCTION(bvec3, isnan, ARG1(ARG(dvec3,  x)))
    FUNCTION(bvec4, isnan, ARG1(ARG(dvec4,  x)))

    FUNCTION(bool,  isinf, ARG1(ARG(double, x)))
    FUNCTION(bvec2, isinf, ARG1(ARG(dvec2,  x)))
    FUNCTION(bvec3, isinf, ARG1(ARG(dvec3,  x)))
    FUNCTION(bvec4, isinf, ARG1(ARG(dvec4,  x)))

BEND(common_double)

BLOCK(common_atomic, (ES && VERSION >= 310) || (!ES && VERSION >= 430))

    FUNCTION(uint, atomicAdd, ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicAdd, ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicMin, ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicMin, ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicMax, ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicMax, ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicAnd, ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicAnd, ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicOr,  ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicOr,  ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicXor, ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicXor, ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicExchange, ARG2(CVIOARG(uint, mem), ARG(uint, data)))
    FUNCTION(int,  atomicExchange, ARG2(CVIOARG(int,  mem), ARG(int,  data)))

    FUNCTION(uint, atomicCompSwap, ARG3(CVIOARG(uint, mem), ARG(uint, compare), ARG(uint, data)))
    FUNCTION(int,  atomicCompSwap, ARG3(CVIOARG(int,  mem), ARG(int,  compare), ARG(int,  data)))

BEND(common_atomic)

BLOCK(common_int_mix, (ES && VERSION >= 310) || (!ES && VERSION >= 450))

    FUNCTION(int,    mix, ARG3(ARG(int,   x), ARG(int,   y), ARG(bool,  a)))
    FUNCTION(ivec2,  mix, ARG3(ARG(ivec2, x), ARG(ivec2, y), ARG(bvec2, a)))
    FUNCTION(ivec3,  mix, ARG3(ARG(ivec3, x), ARG(ivec3, y), ARG(bvec3, a)))
    FUNCTION(ivec4,  mix, ARG3(ARG(ivec4, x), ARG(ivec4, y), ARG(bvec4, a)))

    FUNCTION(uint,   mix, ARG3(ARG(uint,  x), ARG(uint,  y), ARG(bool,  a)))
    FUNCTION(uvec2,  mix, ARG3(ARG(uvec2, x), ARG(uvec2, y), ARG(bvec2, a)))
    FUNCTION(uvec3,  mix, ARG3(ARG(uvec3, x), ARG(uvec3, y), ARG(bvec3, a)))
    FUNCTION(uvec4,  mix, ARG3(ARG(uvec4, x), ARG(uvec4, y), ARG(bvec4, a)))

    FUNCTION(bool,   mix, ARG3(ARG(bool,  x), ARG(bool,  y), ARG(bool,  a)))
    FUNCTION(bvec2,  mix, ARG3(ARG(bvec2, x), ARG(bvec2, y), ARG(bvec2, a)))
    FUNCTION(bvec3,  mix, ARG3(ARG(bvec3, x), ARG(bvec3, y), ARG(bvec3, a)))
    FUNCTION(bvec4,  mix, ARG3(ARG(bvec4, x), ARG(bvec4, y), ARG(bvec4, a)))

BEND(common_int_mix)

BLOCK(common_bitcast, PRED_HAS_BITCAST)

    FUNCTION(int,   floatBitsToInt, ARG1(ARG(float, value)))
    FUNCTION(ivec2, floatBitsToInt, ARG1(ARG(vec2,  value)))
    FUNCTION(ivec3, floatBitsToInt, ARG1(ARG(vec3,  value)))
    FUNCTION(ivec4, floatBitsToInt, ARG1(ARG(vec4,  value)))

    FUNCTION(uint,  floatBitsToUint, ARG1(ARG(float, value)))
    FUNCTION(uvec2, floatBitsToUint, ARG1(ARG(vec2,  value)))
    FUNCTION(uvec3, floatBitsToUint, ARG1(ARG(vec3,  value)))
    FUNCTION(uvec4, floatBitsToUint, ARG1(ARG(vec4,  value)))

    FUNCTION(float, intBitsToFloat, ARG1(ARG(int,   value)))
    FUNCTION(vec2,  intBitsToFloat, ARG1(ARG(ivec2, value)))
    FUNCTION(vec3,  intBitsToFloat, ARG1(ARG(ivec3, value)))
    FUNCTION(vec4,  intBitsToFloat, ARG1(ARG(ivec4, value)))

    FUNCTION(float, uintBitsToFloat, ARG1(ARG(uint,  value)))
    FUNCTION(vec2,  uintBitsToFloat, ARG1(ARG(uvec2, value)))
    FUNCTION(vec3,  uintBitsToFloat, ARG1(ARG(uvec3, value)))
    FUNCTION(vec4,  uintBitsToFloat, ARG1(ARG(uvec4, value)))

BEND(common_bitcast)

BLOCK(common_fma, !ES && VERSION >= 400)

    FUNCTION(float,  fma, ARG3(ARG(float,  a), ARG(float,  b), ARG(float,  c)))
    FUNCTION(vec2,   fma, ARG3(ARG(vec2,   a), ARG(vec2,   b), ARG(vec2,   c)))
    FUNCTION(vec3,   fma, ARG3(ARG(vec3,   a), ARG(vec3,   b), ARG(vec3,   c)))
    FUNCTION(vec4,   fma, ARG3(ARG(vec4,   a), ARG(vec4,   b), ARG(vec4,   c)))

BEND(common_fma)

BLOCK(common_fma_double, PRED_HAS_DOUBLE)

    FUNCTION(double, fma, ARG3(ARG(double, a), ARG(double, b), ARG(double, c)))
    FUNCTION(dvec2,  fma, ARG3(ARG(dvec2,  a), ARG(dvec2,  b), ARG(dvec2,  c)))
    FUNCTION(dvec3,  fma, ARG3(ARG(dvec3,  a), ARG(dvec3,  b), ARG(dvec3,  c)))
    FUNCTION(dvec4,  fma, ARG3(ARG(dvec4,  a), ARG(dvec4,  b), ARG(dvec4,  c)))

BEND(common_fma_double)

BLOCK(common_frexp, (ES && VERSION >=310) || (!ES && VERSION >= 400))

    FUNCTION(float, frexp, ARG2(ARG(float, x), OUTARG(int,   exp)))
    FUNCTION(vec2,  frexp, ARG2(ARG(vec2,  x), OUTARG(ivec2, exp)))
    FUNCTION(vec3,  frexp, ARG2(ARG(vec3,  x), OUTARG(ivec3, exp)))
    FUNCTION(vec4,  frexp, ARG2(ARG(vec4,  x), OUTARG(ivec4, exp)))

    FUNCTION(float, ldexp, ARG2(ARG(float, x), ARG(int,   exp)))
    FUNCTION(vec2,  ldexp, ARG2(ARG(vec2,  x), ARG(ivec2, exp)))
    FUNCTION(vec3,  ldexp, ARG2(ARG(vec3,  x), ARG(ivec3, exp)))
    FUNCTION(vec4,  ldexp, ARG2(ARG(vec4,  x), ARG(ivec4, exp)))

 BEND(common_frexp)

BLOCK(common_frexp_double, PRED_HAS_DOUBLE)

    FUNCTION(double, frexp, ARG2(ARG(double, x), OUTARG(int,   exp)))
    FUNCTION(dvec2,  frexp, ARG2(ARG(dvec2,  x), OUTARG(ivec2, exp)))
    FUNCTION(dvec3,  frexp, ARG2(ARG(dvec3,  x), OUTARG(ivec3, exp)))
    FUNCTION(dvec4,  frexp, ARG2(ARG(dvec4,  x), OUTARG(ivec4, exp)))

    FUNCTION(double, ldexp, ARG2(ARG(double, x), ARG(int,   exp)))
    FUNCTION(dvec2,  ldexp, ARG2(ARG(dvec2,  x), ARG(ivec2, exp)))
    FUNCTION(dvec3,  ldexp, ARG2(ARG(dvec3,  x), ARG(ivec3, exp)))
    FUNCTION(dvec4,  ldexp, ARG2(ARG(dvec4,  x), ARG(ivec4, exp)))

BEND(common_frexp_double)

BLOCK(common_pack_double, !ES && VERSION >= 400)

    FUNCTION(double, packDouble2x32,   ARG1(ARG(uvec2,  v)))
    FUNCTION(uvec2,  unpackDouble2x32, ARG1(ARG(double, v)))

BEND(common_pack_unorm_16)

BLOCK(common_pack_16, (ES && VERSION >= 300) ||  (!ES && VERSION >= 400))

    FUNCTION(uint, packUnorm2x16,   ARG1(ARG(vec2, p)))
    FUNCTION(vec2, unpackUnorm2x16, ARG1(ARG(uint, p)))

BEND(common_pack_unorm_16)

BLOCK(common_pack_snorm_16, (ES && VERSION >= 300) || (!ES && VERSION >= 420))

    FUNCTION(uint, packSnorm2x16,   ARG1(ARG(vec2, p)))
    FUNCTION(vec2, unpackSnorm2x16, ARG1(ARG(uint, p)))
    FUNCTION(uint, packHalf2x16,    ARG1(ARG(vec2, p)))
    FUNCTION(vec2, unpackHalf2x16,  ARG1(ARG(uint, p)))

BEND(common_pack_snorm_16)

BLOCK(common_pack_8, (ES && VERSION >= 310) || (!ES && VERSION >= 400))

    FUNCTION(uint, packSnorm4x8,   ARG1(ARG(vec4, p)))
    FUNCTION(vec4, unpackSnorm4x8, ARG1(ARG(uint, p)))
    FUNCTION(uint, packUnorm4x8,   ARG1(ARG(vec4, p)))
    FUNCTION(vec4, unpackUnorm4x8, ARG1(ARG(uint, p)))

BEND(common_pack_8)

//
// Geometric Functions.
//
BLOCK(geometric, true)

    FUNCTION(float, length, ARG1(ARG(float, x)))
    FUNCTION(float, length, ARG1(ARG(vec2,  x)))
    FUNCTION(float, length, ARG1(ARG(vec3,  x)))
    FUNCTION(float, length, ARG1(ARG(vec4,  x)))

    FUNCTION(float, distance, ARG2(ARG(float, p0), ARG(float, p1)))
    FUNCTION(float, distance, ARG2(ARG(vec2,  p0), ARG(vec2,  p1)))
    FUNCTION(float, distance, ARG2(ARG(vec3,  p0), ARG(vec3,  p1)))
    FUNCTION(float, distance, ARG2(ARG(vec4,  p0), ARG(vec4,  p1)))

    FUNCTION(float, dot, ARG2(ARG(float, x), ARG(float, y)))
    FUNCTION(float, dot, ARG2(ARG(vec2,  x), ARG(vec2,  y)))
    FUNCTION(float, dot, ARG2(ARG(vec3,  x), ARG(vec3,  y)))
    FUNCTION(float, dot, ARG2(ARG(vec4,  x), ARG(vec4,  y)))

    FUNCTION(vec3,  cross, ARG2(ARG(vec3, x), ARG(vec3, y)))

    FUNCTION(float, normalize, ARG1(ARG(float, x)))
    FUNCTION(vec2,  normalize, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3,  normalize, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4,  normalize, ARG1(ARG(vec4,  x)))

    FUNCTION(float, faceforward, ARG3(ARG(float, N), ARG(float, I), ARG(float, Nref)))
    FUNCTION(vec2,  faceforward, ARG3(ARG(vec2,  N), ARG(vec2,  I), ARG(vec2,  Nref)))
    FUNCTION(vec3,  faceforward, ARG3(ARG(vec3,  N), ARG(vec3,  I), ARG(vec3,  Nref)))
    FUNCTION(vec4,  faceforward, ARG3(ARG(vec4,  N), ARG(vec4,  I), ARG(vec4,  Nref)))

    FUNCTION(float, reflect, ARG2(ARG(float, I), ARG(float, N)))
    FUNCTION(vec2,  reflect, ARG2(ARG(vec2,  I), ARG(vec2,  N)))
    FUNCTION(vec3,  reflect, ARG2(ARG(vec3,  I), ARG(vec3,  N)))
    FUNCTION(vec4,  reflect, ARG2(ARG(vec4,  I), ARG(vec4,  N)))

    FUNCTION(float, refract, ARG3(ARG(float, I), ARG(float, N), ARG(float, eta)))
    FUNCTION(vec2,  refract, ARG3(ARG(vec2,  I), ARG(vec2,  N), ARG(float, eta)))
    FUNCTION(vec3,  refract, ARG3(ARG(vec3,  I), ARG(vec3,  N), ARG(float, eta)))
    FUNCTION(vec4,  refract, ARG3(ARG(vec4,  I), ARG(vec4,  N), ARG(float, eta)))

BEND(geometric)

BLOCK(geometric_double, PRED_HAS_DOUBLE)

    FUNCTION(double, length, ARG1(ARG(double, x)))
    FUNCTION(double, length, ARG1(ARG(dvec2,  x)))
    FUNCTION(double, length, ARG1(ARG(dvec3,  x)))
    FUNCTION(double, length, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, distance, ARG2(ARG(double, p0), ARG(double, p1)))
    FUNCTION(double, distance, ARG2(ARG(dvec2,  p0), ARG(dvec2,  p1)))
    FUNCTION(double, distance, ARG2(ARG(dvec3,  p0), ARG(dvec3,  p1)))
    FUNCTION(double, distance, ARG2(ARG(dvec4,  p0), ARG(dvec4,  p1)))

    FUNCTION(double, dot, ARG2(ARG(double, x), ARG(double, y)))
    FUNCTION(double, dot, ARG2(ARG(dvec2,  x), ARG(dvec2,  y)))
    FUNCTION(double, dot, ARG2(ARG(dvec3,  x), ARG(dvec3,  y)))
    FUNCTION(double, dot, ARG2(ARG(dvec4,  x), ARG(dvec4,  y)))

    FUNCTION(dvec3,  cross, ARG2(ARG(dvec3, x), ARG(dvec3, y)))

    FUNCTION(double, normalize, ARG1(ARG(double, x)))
    FUNCTION(dvec2,  normalize, ARG1(ARG(dvec2,  x)))
    FUNCTION(dvec3,  normalize, ARG1(ARG(dvec3,  x)))
    FUNCTION(dvec4,  normalize, ARG1(ARG(dvec4,  x)))

    FUNCTION(double, faceforward, ARG3(ARG(double, N), ARG(double, I), ARG(double, Nref)))
    FUNCTION(dvec2,  faceforward, ARG3(ARG(dvec2,  N), ARG(dvec2,  I), ARG(dvec2,  Nref)))
    FUNCTION(dvec3,  faceforward, ARG3(ARG(dvec3,  N), ARG(dvec3,  I), ARG(dvec3,  Nref)))
    FUNCTION(dvec4,  faceforward, ARG3(ARG(dvec4,  N), ARG(dvec4,  I), ARG(dvec4,  Nref)))

    FUNCTION(double, reflect, ARG2(ARG(double, I), ARG(double, N)))
    FUNCTION(dvec2,  reflect, ARG2(ARG(dvec2,  I), ARG(dvec2,  N)))
    FUNCTION(dvec3,  reflect, ARG2(ARG(dvec3,  I), ARG(dvec3,  N)))
    FUNCTION(dvec4,  reflect, ARG2(ARG(dvec4,  I), ARG(dvec4,  N)))

    // Note: eta is always float even for double results
    FUNCTION(double, refract, ARG3(ARG(double, I), ARG(double, N), ARG(float, eta)))
    FUNCTION(dvec2,  refract, ARG3(ARG(dvec2,  I), ARG(dvec2,  N), ARG(float, eta)))
    FUNCTION(dvec3,  refract, ARG3(ARG(dvec3,  I), ARG(dvec3,  N), ARG(float, eta)))
    FUNCTION(dvec4,  refract, ARG3(ARG(dvec4,  I), ARG(dvec4,  N), ARG(float, eta)))

BEND(geometric_double)

//
// Matrix Functions.
//
BLOCK(matrix, true)

    FUNCTION(mat2, matrixCompMult, ARG2(ARG(mat2, x), ARG(mat2, y)))
    FUNCTION(mat3, matrixCompMult, ARG2(ARG(mat3, x), ARG(mat3, y)))
    FUNCTION(mat4, matrixCompMult, ARG2(ARG(mat4, x), ARG(mat4, y)))

BEND(matrix)

BLOCK(matrix_non_q, VERSION >= 120)

    FUNCTION(mat2,   outerProduct, ARG2(ARG(vec2, c), ARG(vec2, r)))
    FUNCTION(mat3,   outerProduct, ARG2(ARG(vec3, c), ARG(vec3, r)))
    FUNCTION(mat4,   outerProduct, ARG2(ARG(vec4, c), ARG(vec4, r)))
    FUNCTION(mat2x3, outerProduct, ARG2(ARG(vec3, c), ARG(vec2, r)))
    FUNCTION(mat3x2, outerProduct, ARG2(ARG(vec2, c), ARG(vec3, r)))
    FUNCTION(mat2x4, outerProduct, ARG2(ARG(vec4, c), ARG(vec2, r)))
    FUNCTION(mat4x2, outerProduct, ARG2(ARG(vec2, c), ARG(vec4, r)))
    FUNCTION(mat3x4, outerProduct, ARG2(ARG(vec4, c), ARG(vec3, r)))
    FUNCTION(mat4x3, outerProduct, ARG2(ARG(vec3, c), ARG(vec4, r)))

    FUNCTION(mat2,   transpose, ARG1(ARG(mat2,   m)))
    FUNCTION(mat3,   transpose, ARG1(ARG(mat3,   m)))
    FUNCTION(mat4,   transpose, ARG1(ARG(mat4,   m)))
    FUNCTION(mat2x3, transpose, ARG1(ARG(mat3x2, m)))
    FUNCTION(mat3x2, transpose, ARG1(ARG(mat2x3, m)))
    FUNCTION(mat2x4, transpose, ARG1(ARG(mat4x2, m)))
    FUNCTION(mat4x2, transpose, ARG1(ARG(mat2x4, m)))
    FUNCTION(mat3x4, transpose, ARG1(ARG(mat4x3, m)))
    FUNCTION(mat4x3, transpose, ARG1(ARG(mat3x4, m)))

    FUNCTION(mat2x3, matrixCompMult, ARG2(ARG(mat2x3, x), ARG(mat2x3, y)))
    FUNCTION(mat2x4, matrixCompMult, ARG2(ARG(mat2x4, x), ARG(mat2x4, y)))
    FUNCTION(mat3x2, matrixCompMult, ARG2(ARG(mat3x2, x), ARG(mat3x2, y)))
    FUNCTION(mat3x4, matrixCompMult, ARG2(ARG(mat3x4, x), ARG(mat3x4, y)))
    FUNCTION(mat4x2, matrixCompMult, ARG2(ARG(mat4x2, x), ARG(mat4x2, y)))
    FUNCTION(mat4x3, matrixCompMult, ARG2(ARG(mat4x3, x), ARG(mat4x3, y)))

BEND(matrix_non_q)

BLOCK(matrix_double, PRED_HAS_DOUBLE)

    FUNCTION(dmat2,   outerProduct, ARG2(ARG(dvec2, c), ARG(dvec2, r)))
    FUNCTION(dmat3,   outerProduct, ARG2(ARG(dvec3, c), ARG(dvec3, r)))
    FUNCTION(dmat4,   outerProduct, ARG2(ARG(dvec4, c), ARG(dvec4, r)))
    FUNCTION(dmat2x3, outerProduct, ARG2(ARG(dvec3, c), ARG(dvec2, r)))
    FUNCTION(dmat3x2, outerProduct, ARG2(ARG(dvec2, c), ARG(dvec3, r)))
    FUNCTION(dmat2x4, outerProduct, ARG2(ARG(dvec4, c), ARG(dvec2, r)))
    FUNCTION(dmat4x2, outerProduct, ARG2(ARG(dvec2, c), ARG(dvec4, r)))
    FUNCTION(dmat3x4, outerProduct, ARG2(ARG(dvec4, c), ARG(dvec3, r)))
    FUNCTION(dmat4x3, outerProduct, ARG2(ARG(dvec3, c), ARG(dvec4, r)))

    FUNCTION(dmat2,   transpose, ARG1(ARG(dmat2,   m)))
    FUNCTION(dmat3,   transpose, ARG1(ARG(dmat3,   m)))
    FUNCTION(dmat4,   transpose, ARG1(ARG(dmat4,   m)))
    FUNCTION(dmat2x3, transpose, ARG1(ARG(dmat3x2, m)))
    FUNCTION(dmat3x2, transpose, ARG1(ARG(dmat2x3, m)))
    FUNCTION(dmat2x4, transpose, ARG1(ARG(dmat4x2, m)))
    FUNCTION(dmat4x2, transpose, ARG1(ARG(dmat2x4, m)))
    FUNCTION(dmat3x4, transpose, ARG1(ARG(dmat4x3, m)))
    FUNCTION(dmat4x3, transpose, ARG1(ARG(dmat3x4, m)))

    FUNCTION(dmat2x3, matrixCompMult, ARG2(ARG(dmat2x3, x), ARG(dmat2x3, y)))
    FUNCTION(dmat2x4, matrixCompMult, ARG2(ARG(dmat2x4, x), ARG(dmat2x4, y)))
    FUNCTION(dmat3x2, matrixCompMult, ARG2(ARG(dmat3x2, x), ARG(dmat3x2, y)))
    FUNCTION(dmat3x4, matrixCompMult, ARG2(ARG(dmat3x4, x), ARG(dmat3x4, y)))
    FUNCTION(dmat4x2, matrixCompMult, ARG2(ARG(dmat4x2, x), ARG(dmat4x2, y)))
    FUNCTION(dmat4x3, matrixCompMult, ARG2(ARG(dmat4x3, x), ARG(dmat4x3, y)))

    FUNCTION(double, determinant, ARG1(ARG(dmat2, m)))
    FUNCTION(double, determinant, ARG1(ARG(dmat3, m)))
    FUNCTION(double, determinant, ARG1(ARG(dmat4, m)))

    FUNCTION(dmat2, inverse, ARG1(ARG(dmat2, m)))
    FUNCTION(dmat3, inverse, ARG1(ARG(dmat3, m)))
    FUNCTION(dmat4, inverse, ARG1(ARG(dmat4, m)))

BEND(matrix_double)

BLOCK(matrix_determinant, VERSION >= 150)

    FUNCTION(float, determinant, ARG1(ARG(mat2, m)))
    FUNCTION(float, determinant, ARG1(ARG(mat3, m)))
    FUNCTION(float, determinant, ARG1(ARG(mat4, m)))

    FUNCTION(mat2, inverse, ARG1(ARG(mat2, m)))
    FUNCTION(mat3, inverse, ARG1(ARG(mat3, m)))
    FUNCTION(mat4, inverse, ARG1(ARG(mat4, m)))

BEND(matrix_determinant)

//
// Vector relational functions.
//
BLOCK(vector_compare, true)

    FUNCTION(bvec2, lessThan, ARG2(ARG(vec2, x), ARG(vec2, y)))
    FUNCTION(bvec3, lessThan, ARG2(ARG(vec3, x), ARG(vec3, y)))
    FUNCTION(bvec4, lessThan, ARG2(ARG(vec4, x), ARG(vec4, y)))

    FUNCTION(bvec2, lessThan, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(bvec3, lessThan, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(bvec4, lessThan, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION(bvec2, lessThanEqual, ARG2(ARG(vec2, x), ARG(vec2, y)))
    FUNCTION(bvec3, lessThanEqual, ARG2(ARG(vec3, x), ARG(vec3, y)))
    FUNCTION(bvec4, lessThanEqual, ARG2(ARG(vec4, x), ARG(vec4, y)))

    FUNCTION(bvec2, lessThanEqual, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(bvec3, lessThanEqual, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(bvec4, lessThanEqual, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION(bvec2, greaterThan, ARG2(ARG(vec2, x), ARG(vec2, y)))
    FUNCTION(bvec3, greaterThan, ARG2(ARG(vec3, x), ARG(vec3, y)))
    FUNCTION(bvec4, greaterThan, ARG2(ARG(vec4, x), ARG(vec4, y)))

    FUNCTION(bvec2, greaterThan, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(bvec3, greaterThan, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(bvec4, greaterThan, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION(bvec2, greaterThanEqual, ARG2(ARG(vec2, x), ARG(vec2, y)))
    FUNCTION(bvec3, greaterThanEqual, ARG2(ARG(vec3, x), ARG(vec3, y)))
    FUNCTION(bvec4, greaterThanEqual, ARG2(ARG(vec4, x), ARG(vec4, y)))

    FUNCTION(bvec2, greaterThanEqual, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(bvec3, greaterThanEqual, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(bvec4, greaterThanEqual, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION(bvec2, equal, ARG2(ARG(vec2, x), ARG(vec2, y)))
    FUNCTION(bvec3, equal, ARG2(ARG(vec3, x), ARG(vec3, y)))
    FUNCTION(bvec4, equal, ARG2(ARG(vec4, x), ARG(vec4, y)))

    FUNCTION(bvec2, equal, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(bvec3, equal, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(bvec4, equal, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION(bvec2, equal, ARG2(ARG(bvec2, x), ARG(bvec2, y)))
    FUNCTION(bvec3, equal, ARG2(ARG(bvec3, x), ARG(bvec3, y)))
    FUNCTION(bvec4, equal, ARG2(ARG(bvec4, x), ARG(bvec4, y)))

    FUNCTION(bvec2, notEqual, ARG2(ARG(vec2, x), ARG(vec2, y)))
    FUNCTION(bvec3, notEqual, ARG2(ARG(vec3, x), ARG(vec3, y)))
    FUNCTION(bvec4, notEqual, ARG2(ARG(vec4, x), ARG(vec4, y)))

    FUNCTION(bvec2, notEqual, ARG2(ARG(ivec2, x), ARG(ivec2, y)))
    FUNCTION(bvec3, notEqual, ARG2(ARG(ivec3, x), ARG(ivec3, y)))
    FUNCTION(bvec4, notEqual, ARG2(ARG(ivec4, x), ARG(ivec4, y)))

    FUNCTION(bvec2, notEqual, ARG2(ARG(bvec2, x), ARG(bvec2, y)))
    FUNCTION(bvec3, notEqual, ARG2(ARG(bvec3, x), ARG(bvec3, y)))
    FUNCTION(bvec4, notEqual, ARG2(ARG(bvec4, x), ARG(bvec4, y)))

    FUNCTION(bool, any, ARG1(ARG(bvec2, x)))
    FUNCTION(bool, any, ARG1(ARG(bvec3, x)))
    FUNCTION(bool, any, ARG1(ARG(bvec4, x)))

    FUNCTION(bool, all, ARG1(ARG(bvec2, x)))
    FUNCTION(bool, all, ARG1(ARG(bvec3, x)))
    FUNCTION(bool, all, ARG1(ARG(bvec4, x)))

    FUNCTION(bvec2, not, ARG1(ARG(bvec2, x)))
    FUNCTION(bvec3, not, ARG1(ARG(bvec3, x)))
    FUNCTION(bvec4, not, ARG1(ARG(bvec4, x)))

BEND(vector_compare)

BLOCK(vector_compare_uint, VERSION >= 130)

    FUNCTION(bvec2, lessThan, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(bvec3, lessThan, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(bvec4, lessThan, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(bvec2, lessThanEqual, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(bvec3, lessThanEqual, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(bvec4, lessThanEqual, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(bvec2, greaterThan, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(bvec3, greaterThan, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(bvec4, greaterThan, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(bvec2, greaterThanEqual, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(bvec3, greaterThanEqual, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(bvec4, greaterThanEqual, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(bvec2, equal, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(bvec3, equal, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(bvec4, equal, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

    FUNCTION(bvec2, notEqual, ARG2(ARG(uvec2, x), ARG(uvec2, y)))
    FUNCTION(bvec3, notEqual, ARG2(ARG(uvec3, x), ARG(uvec3, y)))
    FUNCTION(bvec4, notEqual, ARG2(ARG(uvec4, x), ARG(uvec4, y)))

BEND(vector_compare_uint)

//
// Texture Functions.
//
BLOCK(texture, (ES && VERSION == 100) || COMPATIBILITY || (CORE && VERSION < 420) || NOPROFILE)

    FUNCTION(vec4, texture2D, ARG2(ARG(sampler2D, sampler), ARG(vec2, coord)))

    FUNCTION(vec4, texture2DProj, ARG2(ARG(sampler2D, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, texture2DProj, ARG2(ARG(sampler2D, sampler), ARG(vec4, coord)))

EXTENSION(GL_OES_texture_3D)
    FUNCTION(vec4, texture3D,     ARG2(ARG(sampler3D, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, texture3DProj, ARG2(ARG(sampler3D, sampler), ARG(vec4, coord)))
EEND(GL_OES_texture_3D)

    FUNCTION(vec4, textureCube, ARG2(ARG(samplerCube, sampler), ARG(vec3, coord)))

BEND(texture)

BLOCK(texture_2, COMPATIBILITY || (CORE && VERSION < 420) || NOPROFILE)

    FUNCTION(vec4, texture1D, ARG2(ARG(sampler1D, sampler), ARG(float, coord)))

    FUNCTION(vec4, texture1DProj, ARG2(ARG(sampler1D, sampler), ARG(vec2, coord)))
    FUNCTION(vec4, texture1DProj, ARG2(ARG(sampler1D, sampler), ARG(vec4, coord)))

    FUNCTION(vec4, shadow1D,     ARG2(ARG(sampler1DShadow, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, shadow2D,     ARG2(ARG(sampler2DShadow, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, shadow1DProj, ARG2(ARG(sampler1DShadow, sampler), ARG(vec4, coord)))
    FUNCTION(vec4, shadow2DProj, ARG2(ARG(sampler2DShadow, sampler), ARG(vec4, coord)))

EXTENSION(GL_ARB_texture_rectangle)
    FUNCTION(vec4, texture2DRect,     ARG2(ARG(sampler2DRect, sampler), ARG(vec2, coord)))
    FUNCTION(vec4, texture2DRectProj, ARG2(ARG(sampler2DRect, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, texture2DRectProj, ARG2(ARG(sampler2DRect, sampler), ARG(vec4, coord)))
    FUNCTION(vec4, shadow2DRect,      ARG2(ARG(sampler2DRectShadow, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, shadow2DRectProj,  ARG2(ARG(sampler2DRectShadow, sampler), ARG(vec4, coord)))
EEND(GL_ARB_texture_rectangle)

BEND(texture_2)

BLOCK(texture_es, ES)

EXTENSION(GL_OES_EGL_image_external)
    FUNCTION(vec4, texture2D,     ARG2(ARG(samplerExternalOES, sampler), ARG(vec2, coord)))
    FUNCTION(vec4, texture2DProj, ARG2(ARG(samplerExternalOES, sampler), ARG(vec3, coord)))
    FUNCTION(vec4, texture2DProj, ARG2(ARG(samplerExternalOES, sampler), ARG(vec4, coord)))
EEND(GL_OES_EGL_image_external)

EXTENSION(GL_EXT_shader_texture_lod)
    FUNCTION(vec4, texture2DGradEXT,
        ARG4(ARG(sampler2D,   sampler),  ARG(vec2, P), ARG(vec2, dPdx), ARG(vec2, dPdy)))
    FUNCTION(vec4, texture2DProjGradEXT,
        ARG4(ARG(sampler2D,   sampler),  ARG(vec3, P), ARG(vec2, dPdx), ARG(vec2, dPdy)))
    FUNCTION(vec4, texture2DProjGradEXT,
        ARG4(ARG(sampler2D,   sampler),  ARG(vec4, P), ARG(vec2, dPdx), ARG(vec2, dPdy)))
    FUNCTION(vec4, textureCubeGradEXT,
        ARG4(ARG(samplerCube, sampler), ARG(vec3, P), ARG(vec3, dPdx), ARG(vec3, dPdy)))
EEND(GL_EXT_shader_texture_lod)

BEND(texture_es)

//
// Noise Functions.
//
BLOCK(noise, !ES)

    FUNCTION(float, noise1, ARG1(ARG(float, x)))
    FUNCTION(float, noise1, ARG1(ARG(vec2,  x)))
    FUNCTION(float, noise1, ARG1(ARG(vec3,  x)))
    FUNCTION(float, noise1, ARG1(ARG(vec4,  x)))

    FUNCTION(vec2, noise2, ARG1(ARG(float, x)))
    FUNCTION(vec2, noise2, ARG1(ARG(vec2,  x)))
    FUNCTION(vec2, noise2, ARG1(ARG(vec3,  x)))
    FUNCTION(vec2, noise2, ARG1(ARG(vec4,  x)))

    FUNCTION(vec3, noise3, ARG1(ARG(float, x)))
    FUNCTION(vec3, noise3, ARG1(ARG(vec2,  x)))
    FUNCTION(vec3, noise3, ARG1(ARG(vec3,  x)))
    FUNCTION(vec3, noise3, ARG1(ARG(vec4,  x)))

    FUNCTION(vec4, noise4, ARG1(ARG(float, x)))
    FUNCTION(vec4, noise4, ARG1(ARG(vec2,  x)))
    FUNCTION(vec4, noise4, ARG1(ARG(vec3,  x)))
    FUNCTION(vec4, noise4, ARG1(ARG(vec4,  x)))

BEND(noise)

//
// Atomic counter Functions.
//
BLOCK(atomic_counter, (ES && VERSION >= 310) || (!ES && VERSION >= 300))
    FUNCTION(uint, atomicCounterIncrement, ARG1(ARG(atomic_uint, x)))
    FUNCTION(uint, atomicCounterDecrement, ARG1(ARG(atomic_uint, x)))
    FUNCTION(uint, atomicCounter,          ARG1(ARG(atomic_uint, x)))
BEND(atomic_counter)

//
// Bitfield Functions.
//
BLOCK(bitfield, (ES && VERSION >= 310) || (!ES && VERSION >= 400))

    FUNCTION( uint, uaddCarry,  ARG3(ARG( uint, x), ARG( uint, y), OUTARG( uint, carry)))
    FUNCTION(uvec2, uaddCarry,  ARG3(ARG(uvec2, x), ARG(uvec2, y), OUTARG(uvec2, carry)))
    FUNCTION(uvec3, uaddCarry,  ARG3(ARG(uvec3, x), ARG(uvec3, y), OUTARG(uvec3, carry)))
    FUNCTION(uvec4, uaddCarry,  ARG3(ARG(uvec4, x), ARG(uvec4, y), OUTARG(uvec4, carry)))

    FUNCTION( uint, usubBorrow, ARG3(ARG( uint, x), ARG( uint, y), OUTARG( uint, borrow)))
    FUNCTION(uvec2, usubBorrow, ARG3(ARG(uvec2, x), ARG(uvec2, y), OUTARG(uvec2, borrow)))
    FUNCTION(uvec3, usubBorrow, ARG3(ARG(uvec3, x), ARG(uvec3, y), OUTARG(uvec3, borrow)))
    FUNCTION(uvec4, usubBorrow, ARG3(ARG(uvec4, x), ARG(uvec4, y), OUTARG(uvec4, borrow)))

    FUNCTION(void, umulExtended,
        ARG4(ARG( uint, x), ARG( uint, y), OUTARG( uint, msb), OUTARG( uint, lsb)))
    FUNCTION(void, umulExtended,
        ARG4(ARG(uvec2, x), ARG(uvec2, y), OUTARG(uvec2, msb), OUTARG(uvec2, lsb)))
    FUNCTION(void, umulExtended,
        ARG4(ARG(uvec3, x), ARG(uvec3, y), OUTARG(uvec3, msb), OUTARG(uvec3, lsb)))
    FUNCTION(void, umulExtended,
        ARG4(ARG(uvec4, x), ARG(uvec4, y), OUTARG(uvec4, msb), OUTARG(uvec4, lsb)))

    FUNCTION(void, imulExtended,
        ARG4(ARG(  int, x), ARG(  int, y), OUTARG(  int, msb), OUTARG(  int, lsb)))
    FUNCTION(void, imulExtended,
        ARG4(ARG(ivec2, x), ARG(ivec2, y), OUTARG(ivec2, msb), OUTARG(ivec2, lsb)))
    FUNCTION(void, imulExtended,
        ARG4(ARG(ivec3, x), ARG(ivec3, y), OUTARG(ivec3, msb), OUTARG(ivec3, lsb)))
    FUNCTION(void, imulExtended,
        ARG4(ARG(ivec4, x), ARG(ivec4, y), OUTARG(ivec4, msb), OUTARG(ivec4, lsb)))

    FUNCTION(  int, bitfieldExtract, ARG3(ARG(  int, value), ARG(int, offset), ARG(int, bits)))
    FUNCTION(ivec2, bitfieldExtract, ARG3(ARG(ivec2, value), ARG(int, offset), ARG(int, bits)))
    FUNCTION(ivec3, bitfieldExtract, ARG3(ARG(ivec3, value), ARG(int, offset), ARG(int, bits)))
    FUNCTION(ivec4, bitfieldExtract, ARG3(ARG(ivec4, value), ARG(int, offset), ARG(int, bits)))

    FUNCTION( uint, bitfieldExtract, ARG3(ARG( uint, value), ARG(int, offset), ARG(int, bits)))
    FUNCTION(uvec2, bitfieldExtract, ARG3(ARG(uvec2, value), ARG(int, offset), ARG(int, bits)))
    FUNCTION(uvec3, bitfieldExtract, ARG3(ARG(uvec3, value), ARG(int, offset), ARG(int, bits)))
    FUNCTION(uvec4, bitfieldExtract, ARG3(ARG(uvec4, value), ARG(int, offset), ARG(int, bits)))

    FUNCTION(  int, bitfieldInsert,
        ARG4(ARG(  int, base), ARG(  int, insert), ARG(int, offset), ARG(int, bits)))
    FUNCTION(ivec2, bitfieldInsert,
        ARG4(ARG(ivec2, base), ARG(ivec2, insert), ARG(int, offset), ARG(int, bits)))
    FUNCTION(ivec3, bitfieldInsert,
        ARG4(ARG(ivec3, base), ARG(ivec3, insert), ARG(int, offset), ARG(int, bits)))
    FUNCTION(ivec4, bitfieldInsert,
        ARG4(ARG(ivec4, base), ARG(ivec4, insert), ARG(int, offset), ARG(int, bits)))

    FUNCTION( uint, bitfieldInsert,
        ARG4(ARG( uint, base), ARG( uint, insert), ARG(int, offset), ARG(int, bits)))
    FUNCTION(uvec2, bitfieldInsert,
        ARG4(ARG(uvec2, base), ARG(uvec2, insert), ARG(int, offset), ARG(int, bits)))
    FUNCTION(uvec3, bitfieldInsert,
        ARG4(ARG(uvec3, base), ARG(uvec3, insert), ARG(int, offset), ARG(int, bits)))
    FUNCTION(uvec4, bitfieldInsert,
        ARG4(ARG(uvec4, base), ARG(uvec4, insert), ARG(int, offset), ARG(int, bits)))

    FUNCTION(  int, bitfieldReverse, ARG1(ARG(  int, value)))
    FUNCTION(ivec2, bitfieldReverse, ARG1(ARG(ivec2, value)))
    FUNCTION(ivec3, bitfieldReverse, ARG1(ARG(ivec3, value)))
    FUNCTION(ivec4, bitfieldReverse, ARG1(ARG(ivec4, value)))

    FUNCTION( uint, bitfieldReverse, ARG1(ARG( uint, value)))
    FUNCTION(uvec2, bitfieldReverse, ARG1(ARG(uvec2, value)))
    FUNCTION(uvec3, bitfieldReverse, ARG1(ARG(uvec3, value)))
    FUNCTION(uvec4, bitfieldReverse, ARG1(ARG(uvec4, value)))

    FUNCTION(  int, bitCount, ARG1(ARG(  int, value)))
    FUNCTION(ivec2, bitCount, ARG1(ARG(ivec2, value)))
    FUNCTION(ivec3, bitCount, ARG1(ARG(ivec3, value)))
    FUNCTION(ivec4, bitCount, ARG1(ARG(ivec4, value)))

    FUNCTION(  int, bitCount, ARG1(ARG( uint, value)))
    FUNCTION(ivec2, bitCount, ARG1(ARG(uvec2, value)))
    FUNCTION(ivec3, bitCount, ARG1(ARG(uvec3, value)))
    FUNCTION(ivec4, bitCount, ARG1(ARG(uvec4, value)))

    FUNCTION(  int, findLSB, ARG1(ARG(  int, value)))
    FUNCTION(ivec2, findLSB, ARG1(ARG(ivec2, value)))
    FUNCTION(ivec3, findLSB, ARG1(ARG(ivec3, value)))
    FUNCTION(ivec4, findLSB, ARG1(ARG(ivec4, value)))

    FUNCTION(  int, findLSB, ARG1(ARG( uint, value)))
    FUNCTION(ivec2, findLSB, ARG1(ARG(uvec2, value)))
    FUNCTION(ivec3, findLSB, ARG1(ARG(uvec3, value)))
    FUNCTION(ivec4, findLSB, ARG1(ARG(uvec4, value)))

    FUNCTION(  int, findMSB, ARG1(ARG(  int, value)))
    FUNCTION(ivec2, findMSB, ARG1(ARG(ivec2, value)))
    FUNCTION(ivec3, findMSB, ARG1(ARG(ivec3, value)))
    FUNCTION(ivec4, findMSB, ARG1(ARG(ivec4, value)))

    FUNCTION(  int, findMSB, ARG1(ARG( uint, value)))
    FUNCTION(ivec2, findMSB, ARG1(ARG(uvec2, value)))
    FUNCTION(ivec3, findMSB, ARG1(ARG(uvec3, value)))
    FUNCTION(ivec4, findMSB, ARG1(ARG(uvec4, value)))

BEND(bitfield)

BLOCK(texture_lod,
    (VERSION == 100 && VERTEX_LANG) ||
    (ES && VERSION == 100) ||
    COMPATIBILITY ||
    (CORE && VERSION < 420) ||
    NOPROFILE)

EXTENSION(GL_ARB_shader_texture_lod)
    FUNCTION(vec4, texture2DLod,
        ARG3(ARG(sampler2D,   sampler), ARG(vec2, P), ARG(float, lod)))
    FUNCTION(vec4, texture2DProjLod,
        ARG3(ARG(sampler2D,   sampler), ARG(vec3, P), ARG(float, lod)))
    FUNCTION(vec4, texture2DProjLod,
        ARG3(ARG(sampler2D,   sampler), ARG(vec4, P), ARG(float, lod)))
    FUNCTION(vec4, textureCubeLod,
        ARG3(ARG(samplerCube, sampler), ARG(vec3, P), ARG(float, lod)))

    EXTENSION(GL_OES_texture_3D)
        FUNCTION(vec4, texture3DLod,
            ARG3(ARG(sampler3D,   sampler), ARG(vec3, P), ARG(float, lod)))
        FUNCTION(vec4, texture3DProjLod,
            ARG3(ARG(sampler3D,   sampler), ARG(vec4, P), ARG(float, lod)))
    EEND(GL_OES_texture_3D)
EEND(GL_ARB_shader_texture_lod)

BEND(texture_lod)

BLOCK(texture_lod_non_es,
    (VERSION == 100 && VERTEX_LANG) ||
    COMPATIBILITY ||
    (CORE && VERSION < 420) ||
    NOPROFILE)

EXTENSION(GL_ARB_shader_texture_lod)
    FUNCTION(vec4, texture1DLod,
        ARG3(ARG(sampler1D,       sampler), ARG(float, P), ARG(float, lod)))
    FUNCTION(vec4, texture1DProjLod,
        ARG3(ARG(sampler1D,       sampler), ARG(vec2,  P), ARG(float, lod)))
    FUNCTION(vec4, texture1DProjLod,
        ARG3(ARG(sampler1D,       sampler), ARG(vec4,  P), ARG(float, lod)))
    FUNCTION(vec4, shadow1DLod,
        ARG3(ARG(sampler1DShadow, sampler), ARG(vec3,  P), ARG(float, lod)))
    FUNCTION(vec4, shadow2DLod,
        ARG3(ARG(sampler2DShadow, sampler), ARG(vec3,  P), ARG(float, lod)))
    FUNCTION(vec4, shadow1DProjLod,
        ARG3(ARG(sampler1DShadow, sampler), ARG(vec4,  P), ARG(float, lod)))
    FUNCTION(vec4, shadow2DProjLod,
        ARG3(ARG(sampler2DShadow, sampler), ARG(vec4,  P), ARG(float, lod)))

    FUNCTION(vec4, texture1DGradARB,
        ARG4(ARG(sampler1D,           sampler), ARG(float, P), ARG(float, dPdx), ARG(float, dPdy)))
    FUNCTION(vec4, texture1DProjGradARB,
        ARG4(ARG(sampler1D,           sampler), ARG(vec2,  P), ARG(float, dPdx), ARG(float, dPdy)))
    FUNCTION(vec4, texture1DProjGradARB,
        ARG4(ARG(sampler1D,           sampler), ARG(vec4,  P), ARG(float, dPdx), ARG(float, dPdy)))
    FUNCTION(vec4, texture2DGradARB,
        ARG4(ARG(sampler2D,           sampler), ARG(vec2,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, texture2DProjGradARB,
        ARG4(ARG(sampler2D,           sampler), ARG(vec3,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, texture2DProjGradARB,
        ARG4(ARG(sampler2D,           sampler), ARG(vec4,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, texture3DGradARB,
        ARG4(ARG(sampler3D,           sampler), ARG(vec3,  P), ARG(vec3,  dPdx), ARG(vec3,  dPdy)))
    FUNCTION(vec4, texture3DProjGradARB,
        ARG4(ARG(sampler3D,           sampler), ARG(vec4,  P), ARG(vec3,  dPdx), ARG(vec3,  dPdy)))
    FUNCTION(vec4, textureCubeGradARB,
        ARG4(ARG(samplerCube,         sampler), ARG(vec3,  P), ARG(vec3,  dPdx), ARG(vec3,  dPdy)))
    FUNCTION(vec4, shadow1DGradARB,
        ARG4(ARG(sampler1DShadow,     sampler), ARG(vec3,  P), ARG(float, dPdx), ARG(float, dPdy)))
    FUNCTION(vec4, shadow1DProjGradARB,
        ARG4(ARG(sampler1DShadow,     sampler), ARG(vec4,  P), ARG(float, dPdx), ARG(float, dPdy)))
    FUNCTION(vec4, shadow2DGradARB,
        ARG4(ARG(sampler2DShadow,     sampler), ARG(vec3,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, shadow2DProjGradARB,
        ARG4(ARG(sampler2DShadow,     sampler), ARG(vec4,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, texture2DRectGradARB,
        ARG4(ARG(sampler2DRect,       sampler), ARG(vec2,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, texture2DRectProjGradARB,
        ARG4(ARG(sampler2DRect,       sampler), ARG(vec3,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, texture2DRectProjGradARB,
        ARG4(ARG(sampler2DRect,       sampler), ARG(vec4,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, shadow2DRectGradARB,
        ARG4(ARG(sampler2DRectShadow, sampler), ARG(vec3,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
    FUNCTION(vec4, shadow2DRectProjGradARB,
        ARG4(ARG(sampler2DRectShadow, sampler), ARG(vec4,  P), ARG(vec2,  dPdx), ARG(vec2,  dPdy)))
EEND(GL_ARB_shader_texture_lod)

BEND(texture_lod_non_es)

// Prototypes for built-in functions seen by geometry shaders only.
BLOCK(stream_primitive, GEOMETRY_LANG && (!ES && VERSION >= 400))
    FUNCTION(void, EmitStreamVertex,   ARG1(ARG(int, stream)))
    FUNCTION(void, EndStreamPrimitive, ARG1(ARG(int, stream)))
BEND(stream_primitive)

BLOCK(vertex_primitive, GEOMETRY_LANG && (!ES && VERSION >= 150))
    FUNCTION(void, EmitVertex,   ARG0())
    FUNCTION(void, EndPrimitive, ARG0())
BEND(vertex_primitive)

// Prototypes for control functions.
BLOCK(tess_barrier, TESSCONTROL_LANG && (!ES && VERSION >= 150))
    FUNCTION(void, barrier, ARG0())
BEND(tess_barrier)

BLOCK(compute_barrier, COMPUTE_LANG && ((!ES && VERSION >= 430) || (ES && VERSION >= 310)))
    FUNCTION(void, barrier, ARG0())
BEND(compute_barrier)

BLOCK(memory_barrier, (!ES && VERSION >= 130) || (ES && VERSION >= 310))
    FUNCTION(void, memoryBarrier, ARG0())
BEND(memory_barrier)

BLOCK(memory_barrier_ext, (!ES && VERSION >= 430) || (ES && VERSION >= 310))
    FUNCTION(void, memoryBarrierAtomicCounter, ARG0())
    FUNCTION(void, memoryBarrierBuffer, ARG0())
    FUNCTION(void, memoryBarrierImage, ARG0())
BEND(memory_barrier_ext)

BLOCK(memory_barrier_compute, COMPUTE_LANG && ((!ES && VERSION >= 430) || (ES && VERSION >= 310)))
    FUNCTION(void, memoryBarrierShared, ARG0())
    FUNCTION(void, groupMemoryBarrier, ARG0())
BEND(memory_barrier_compute)

// Prototypes for built-in functions seen by fragment shaders only.
BLOCK(texture_fragment_100, FRAGMENT_LANG && (!ES || VERSION == 100))
    FUNCTION(vec4, texture2D,
        ARG3(ARG(sampler2D,   sampler), ARG(vec2, coord), ARG(float, bias)))
    FUNCTION(vec4, texture2DProj,
        ARG3(ARG(sampler2D,   sampler), ARG(vec3, coord), ARG(float, bias)))
    FUNCTION(vec4, texture2DProj,
        ARG3(ARG(sampler2D,   sampler), ARG(vec4, coord), ARG(float, bias)))
    FUNCTION(vec4, textureCube,
        ARG3(ARG(samplerCube, sampler), ARG(vec3, coord), ARG(float, bias)))

EXTENSION(GL_OES_texture_3D)
    FUNCTION(vec4, texture3D,
        ARG3(ARG(sampler3D,   sampler), ARG(vec3, coord), ARG(float, bias)))
    FUNCTION(vec4, texture3DProj,
        ARG3(ARG(sampler3D,   sampler), ARG(vec4, coord), ARG(float, bias)))
EEND(GL_OES_texture_3D)

BEND(texture_fragement_100)

BLOCK(texture_fragment, FRAGMENT_LANG && (!ES || VERSION > 100))
    FUNCTION(vec4, texture1D,
        ARG3(ARG(sampler1D,       sampler), ARG(float, coord), ARG(float, bias)))
    FUNCTION(vec4, texture1DProj,
        ARG3(ARG(sampler1D,       sampler), ARG(vec2,  coord), ARG(float, bias)))
    FUNCTION(vec4, texture1DProj,
        ARG3(ARG(sampler1D,       sampler), ARG(vec4,  coord), ARG(float, bias)))
    FUNCTION(vec4, shadow1D,
        ARG3(ARG(sampler1DShadow, sampler), ARG(vec3,  coord), ARG(float, bias)))
    FUNCTION(vec4, shadow2D,
        ARG3(ARG(sampler2DShadow, sampler), ARG(vec3,  coord), ARG(float, bias)))
    FUNCTION(vec4, shadow1DProj,
        ARG3(ARG(sampler1DShadow, sampler), ARG(vec4,  coord), ARG(float, bias)))
    FUNCTION(vec4, shadow2DProj,
        ARG3(ARG(sampler2DShadow, sampler), ARG(vec4,  coord), ARG(float, bias)))
BEND(texture_fragment)

BLOCK(texture_fragment_es_lod, FRAGMENT_LANG && ES)

EXTENSION(GL_EXT_shader_texture_lod)
    FUNCTION(vec4, texture2DLodEXT,
        ARG3(ARG(sampler2D,   sampler), ARG(vec2, coord), ARG(float, lod)))
    FUNCTION(vec4, texture2DProjLodEXT,
        ARG3(ARG(sampler2D,   sampler), ARG(vec3, coord), ARG(float, lod)))
    FUNCTION(vec4, texture2DProjLodEXT,
        ARG3(ARG(sampler2D,   sampler), ARG(vec4, coord), ARG(float, lod)))
    FUNCTION(vec4, textureCubeLodEXT,
        ARG3(ARG(samplerCube, sampler), ARG(vec3, coord), ARG(float, lod)))
EEND(GL_EXT_shader_texture_lod)

BEND(texture_fragment_es_lod)

BLOCK(derivatives, FRAGMENT_LANG)
    FUNCTION(float, dFdx, ARG1(ARG(float, p)))
    FUNCTION(vec2,  dFdx, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  dFdx, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  dFdx, ARG1(ARG(vec4,  p)))

    FUNCTION(float, dFdy, ARG1(ARG(float, p)))
    FUNCTION(vec2,  dFdy, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  dFdy, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  dFdy, ARG1(ARG(vec4,  p)))

    FUNCTION(float, fwidth, ARG1(ARG(float, p)))
    FUNCTION(vec2,  fwidth, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  fwidth, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  fwidth, ARG1(ARG(vec4,  p)))
BEND(derivatives)

BLOCK(derivative_control, FRAGMENT_LANG && (!ES && VERSION >= 400))

EXTENSION(GL_ARB_derivative_control)
    FUNCTION(float, dFdxFine, ARG1(ARG(float, p)))
    FUNCTION(vec2,  dFdxFine, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  dFdxFine, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  dFdxFine, ARG1(ARG(vec4,  p)))

    FUNCTION(float, dFdyFine, ARG1(ARG(float, p)))
    FUNCTION(vec2,  dFdyFine, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  dFdyFine, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  dFdyFine, ARG1(ARG(vec4,  p)))

    FUNCTION(float, fwidthFine, ARG1(ARG(float, p)))
    FUNCTION(vec2,  fwidthFine, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  fwidthFine, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  fwidthFine, ARG1(ARG(vec4,  p)))

    FUNCTION(float, dFdxCoarse, ARG1(ARG(float, p)))
    FUNCTION(vec2,  dFdxCoarse, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  dFdxCoarse, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  dFdxCoarse, ARG1(ARG(vec4,  p)))

    FUNCTION(float, dFdyCoarse, ARG1(ARG(float, p)))
    FUNCTION(vec2,  dFdyCoarse, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  dFdyCoarse, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  dFdyCoarse, ARG1(ARG(vec4,  p)))

    FUNCTION(float, fwidthCoarse, ARG1(ARG(float, p)))
    FUNCTION(vec2,  fwidthCoarse, ARG1(ARG(vec2,  p)))
    FUNCTION(vec3,  fwidthCoarse, ARG1(ARG(vec3,  p)))
    FUNCTION(vec4,  fwidthCoarse, ARG1(ARG(vec4,  p)))
EEND(GL_ARB_derivative_control)

BEND(derivative_control)

BLOCK(depth_range_parameters, !ES)

STRUCT_BEGIN(gl_DepthRangeParameters)
    FIELD(, float, near)   // n
    FIELD(, float, far)    // f
    FIELD(, float, diff)   // f - n
STRUCT_END(gl_DepthRangeParameters)

UVARIABLE(gl_DepthRangeParameters, gl_DepthRange)

BEND(depth_range_parameters)

BLOCK(depth_range_parameters_es, ES)

STRUCT_BEGIN(gl_DepthRangeParameters)
    FIELD(highp, float, near)   // n
    FIELD(highp, float, far)    // f
    FIELD(highp, float, diff)   // f - n
STRUCT_END(gl_DepthRangeParameters)

UVARIABLE(gl_DepthRangeParameters, gl_DepthRange)

BEND(depth_range_parameters_es)

#undef MATRIX_TYPE_END
#undef MATRIX_TYPE_BEGIN
#undef VECTOR_TYPE_END
#undef VECTOR_TYPE_BEGIN
#undef TYPE_END
#undef TYPE_BEGIN
#undef HAS
#undef UVARIABLE
#undef FIELD
#undef STRUCT_END
#undef STRUCT_BEGIN
#undef EEND
#undef EXTENSION
#undef BEND
#undef BLOCK
#undef CONSTRUCTOR
#undef FUNCTION
#undef CVIOARG
#undef OUTARG 
#undef ARG
