/******************************************************************************
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
 *****************************************************************************/

// can be included more than once ...

#ifndef BUILTIN_TYPE
#define BUILTIN_TYPE(type, name, args)
#endif

#define BUILTIN_SCALAR(scalar) \
    BUILTIN_TYPE(Type_ ## scalar, hlsl_ ## scalar ## _type, )

#define BUILTIN_VECTOR(scalar) \
    BUILTIN_TYPE(Type_vector, hlsl_ ## scalar ## 1_type, (&hlsl_ ## scalar ##_type, 1))  \
    BUILTIN_TYPE(Type_vector, hlsl_ ## scalar ## 2_type, (&hlsl_ ## scalar ##_type, 2))  \
    BUILTIN_TYPE(Type_vector, hlsl_ ## scalar ## 3_type, (&hlsl_ ## scalar ##_type, 3))  \
    BUILTIN_TYPE(Type_vector, hlsl_ ## scalar ## 4_type, (&hlsl_ ## scalar ##_type, 4))

#define BUILTIN_MATRIX(scalar) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 1x1_type, (&hlsl_ ## scalar ## 1_type, 1)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 1x2_type, (&hlsl_ ## scalar ## 1_type, 2)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 1x3_type, (&hlsl_ ## scalar ## 1_type, 3)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 1x4_type, (&hlsl_ ## scalar ## 1_type, 4)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 2x1_type, (&hlsl_ ## scalar ## 2_type, 1)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 2x2_type, (&hlsl_ ## scalar ## 2_type, 2)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 2x3_type, (&hlsl_ ## scalar ## 2_type, 3)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 2x4_type, (&hlsl_ ## scalar ## 2_type, 4)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 3x1_type, (&hlsl_ ## scalar ## 3_type, 1)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 3x2_type, (&hlsl_ ## scalar ## 3_type, 2)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 3x3_type, (&hlsl_ ## scalar ## 3_type, 3)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 3x4_type, (&hlsl_ ## scalar ## 3_type, 4)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 4x1_type, (&hlsl_ ## scalar ## 4_type, 1)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 4x2_type, (&hlsl_ ## scalar ## 4_type, 2)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 4x3_type, (&hlsl_ ## scalar ## 4_type, 3)) \
    BUILTIN_TYPE(Type_matrix, hlsl_ ## scalar ## 4x4_type, (&hlsl_ ## scalar ## 4_type, 4))

BUILTIN_SCALAR(error)
BUILTIN_SCALAR(void)
BUILTIN_SCALAR(bool)
BUILTIN_SCALAR(int)
BUILTIN_SCALAR(uint)
BUILTIN_SCALAR(half)
BUILTIN_SCALAR(float)
BUILTIN_SCALAR(double)
BUILTIN_SCALAR(min12int)
BUILTIN_SCALAR(min16int)
BUILTIN_SCALAR(min16uint)
BUILTIN_SCALAR(min10float)
BUILTIN_SCALAR(min16float)

BUILTIN_VECTOR(bool)
BUILTIN_VECTOR(int)
BUILTIN_VECTOR(uint)
BUILTIN_VECTOR(half)
BUILTIN_VECTOR(float)
BUILTIN_VECTOR(double)
BUILTIN_VECTOR(min12int)
BUILTIN_VECTOR(min16int)
BUILTIN_VECTOR(min16uint)
BUILTIN_VECTOR(min10float)
BUILTIN_VECTOR(min16float)

BUILTIN_MATRIX(bool)
BUILTIN_MATRIX(int)
BUILTIN_MATRIX(uint)
BUILTIN_MATRIX(half)
BUILTIN_MATRIX(float)
BUILTIN_MATRIX(double)
BUILTIN_MATRIX(min12int)
BUILTIN_MATRIX(min16int)
BUILTIN_MATRIX(min16uint)
BUILTIN_MATRIX(min10float)
BUILTIN_MATRIX(min16float)

// textures
BUILTIN_TYPE(Type_texture, hlsl_texture_type,                (SHAPE_UNKNOWN))
BUILTIN_TYPE(Type_texture, hlsl_texture1D_type,              (SHAPE_1D))
BUILTIN_TYPE(Type_texture, hlsl_texture2D_type,              (SHAPE_2D))
BUILTIN_TYPE(Type_texture, hlsl_texture3D_type,              (SHAPE_3D))
BUILTIN_TYPE(Type_texture, hlsl_textureCube_type,            (SHAPE_CUBE))
BUILTIN_TYPE(Type_texture, hlsl_texture1DArray_type,         (SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_texture, hlsl_texture2DArray_type,         (SHAPE_2D_ARRAY))

#undef BUILTIN_MATRIX
#undef BUILTIN_VECTOR
#undef BUILTIN_SCALAR
#undef BUILTIN_TYPE
