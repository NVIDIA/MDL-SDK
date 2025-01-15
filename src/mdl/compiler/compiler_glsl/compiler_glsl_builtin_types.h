/******************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

BUILTIN_TYPE(Type_error,       glsl_error_type, )
BUILTIN_TYPE(Type_void,        glsl_void_type, )
BUILTIN_TYPE(Type_bool,        glsl_bool_type, )
BUILTIN_TYPE(Type_int,         glsl_int_type, )
BUILTIN_TYPE(Type_uint,        glsl_uint_type, )
BUILTIN_TYPE(Type_half,        glsl_half_type, )
BUILTIN_TYPE(Type_float,       glsl_float_type, )
BUILTIN_TYPE(Type_double,      glsl_double_type, )
BUILTIN_TYPE(Type_int8_t,      glsl_int8_t_type, )
BUILTIN_TYPE(Type_uint8_t,     glsl_uint8_t_type, )
BUILTIN_TYPE(Type_int16_t,     glsl_int16_t_type, )
BUILTIN_TYPE(Type_uint16_t,    glsl_uint16_t_type, )
BUILTIN_TYPE(Type_int64_t,     glsl_int64_t_type, )
BUILTIN_TYPE(Type_uint64_t,    glsl_uint64_t_type, )
BUILTIN_TYPE(Type_vector,      glsl_bvec2_type,   (&glsl_bool_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_bvec3_type,   (&glsl_bool_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_bvec4_type,   (&glsl_bool_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_ivec2_type,   (&glsl_int_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_ivec3_type,   (&glsl_int_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_ivec4_type,   (&glsl_int_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_uvec2_type,   (&glsl_uint_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_uvec3_type,   (&glsl_uint_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_uvec4_type,   (&glsl_uint_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_vec2_type,    (&glsl_float_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_vec3_type,    (&glsl_float_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_vec4_type,    (&glsl_float_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_dvec2_type,   (&glsl_double_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_dvec3_type,   (&glsl_double_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_dvec4_type,   (&glsl_double_type, 4))
BUILTIN_TYPE(Type_matrix,      glsl_mat2_type,    (&glsl_vec2_type, 2))
BUILTIN_TYPE(Type_matrix,      glsl_mat2x3_type,  (&glsl_vec2_type, 3))
BUILTIN_TYPE(Type_matrix,      glsl_mat2x4_type,  (&glsl_vec2_type, 4))
BUILTIN_TYPE(Type_matrix,      glsl_mat3x2_type,  (&glsl_vec3_type, 2))
BUILTIN_TYPE(Type_matrix,      glsl_mat3_type,    (&glsl_vec3_type, 3))
BUILTIN_TYPE(Type_matrix,      glsl_mat3x4_type,  (&glsl_vec3_type, 4))
BUILTIN_TYPE(Type_matrix,      glsl_mat4x2_type,  (&glsl_vec4_type, 2))
BUILTIN_TYPE(Type_matrix,      glsl_mat4x3_type,  (&glsl_vec4_type, 3))
BUILTIN_TYPE(Type_matrix,      glsl_mat4_type,    (&glsl_vec4_type, 4))
BUILTIN_TYPE(Type_matrix,      glsl_dmat2_type,   (&glsl_dvec2_type, 2))
BUILTIN_TYPE(Type_matrix,      glsl_dmat2x3_type, (&glsl_dvec2_type, 3))
BUILTIN_TYPE(Type_matrix,      glsl_dmat2x4_type, (&glsl_dvec2_type, 4))
BUILTIN_TYPE(Type_matrix,      glsl_dmat3x2_type, (&glsl_dvec3_type, 2))
BUILTIN_TYPE(Type_matrix,      glsl_dmat3_type,   (&glsl_dvec3_type, 3))
BUILTIN_TYPE(Type_matrix,      glsl_dmat3x4_type, (&glsl_dvec3_type, 4))
BUILTIN_TYPE(Type_matrix,      glsl_dmat4x2_type, (&glsl_dvec4_type, 2))
BUILTIN_TYPE(Type_matrix,      glsl_dmat4x3_type, (&glsl_dvec4_type, 3))
BUILTIN_TYPE(Type_matrix,      glsl_dmat4_type,   (&glsl_dvec4_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_i8vec2_type, (&glsl_int8_t_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_i8vec3_type, (&glsl_int8_t_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_i8vec4_type, (&glsl_int8_t_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_u8vec2_type, (&glsl_uint8_t_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_u8vec3_type, (&glsl_uint8_t_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_u8vec4_type, (&glsl_uint8_t_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_i16vec2_type, (&glsl_int16_t_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_i16vec3_type, (&glsl_int16_t_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_i16vec4_type, (&glsl_int16_t_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_u16vec2_type, (&glsl_uint16_t_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_u16vec3_type, (&glsl_uint16_t_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_u16vec4_type, (&glsl_uint16_t_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_i64vec2_type, (&glsl_int64_t_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_i64vec3_type, (&glsl_int64_t_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_i64vec4_type, (&glsl_int64_t_type, 4))
BUILTIN_TYPE(Type_vector,      glsl_u64vec2_type, (&glsl_uint64_t_type, 2))
BUILTIN_TYPE(Type_vector,      glsl_u64vec3_type, (&glsl_uint64_t_type, 3))
BUILTIN_TYPE(Type_vector,      glsl_u64vec4_type, (&glsl_uint64_t_type, 4))

// Floating-Point Opaque Types
BUILTIN_TYPE(Type_sampler, glsl_sampler1D_type,              (&glsl_float_type, SHAPE_1D))
BUILTIN_TYPE(Type_image,   glsl_image1D_type,                (&glsl_float_type, SHAPE_1D))
BUILTIN_TYPE(Type_sampler, glsl_sampler2D_type,              (&glsl_float_type, SHAPE_2D))
BUILTIN_TYPE(Type_image,   glsl_image2D_type,                (&glsl_float_type, SHAPE_2D))
BUILTIN_TYPE(Type_sampler, glsl_sampler3D_type,              (&glsl_float_type, SHAPE_3D))
BUILTIN_TYPE(Type_image,   glsl_image3D_type,                (&glsl_float_type, SHAPE_3D))
BUILTIN_TYPE(Type_sampler, glsl_samplerCube_type,            (&glsl_float_type, SHAPE_CUBE))
BUILTIN_TYPE(Type_image,   glsl_imageCube_type,              (&glsl_float_type, SHAPE_CUBE))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DRect_type,          (&glsl_float_type, SHAPE_2D_RECT))
BUILTIN_TYPE(Type_image,   glsl_image2DRect_type,            (&glsl_float_type, SHAPE_2D_RECT))
BUILTIN_TYPE(Type_sampler, glsl_sampler1DArray_type,         (&glsl_float_type, SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_image1DArray_type,           (&glsl_float_type, SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DArray_type,         (&glsl_float_type, SHAPE_2D_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_image2DArray_type,           (&glsl_float_type, SHAPE_2D_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_samplerBuffer_type,          (&glsl_float_type, SHAPE_BUFFER))
BUILTIN_TYPE(Type_image,   glsl_imageBuffer_type,            (&glsl_float_type, SHAPE_BUFFER))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DMS_type,            (&glsl_float_type, SHAPE_2DMS))
BUILTIN_TYPE(Type_image,   glsl_image2DMS_type,              (&glsl_float_type, SHAPE_2DMS))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DMSArray_type,       (&glsl_float_type, SHAPE_2DMS_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_image2DMSArray_type,         (&glsl_float_type, SHAPE_2DMS_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_samplerCubeArray_type,       (&glsl_float_type, SHAPE_CUBE_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_imageCubeArray_type,         (&glsl_float_type, SHAPE_CUBE_ARRAY))

BUILTIN_TYPE(Type_sampler, glsl_sampler1DShadow_type,
    (&glsl_float_type, SHAPE_1D_SHADOW))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DShadow_type,
    (&glsl_float_type, SHAPE_2D_SHADOW))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DRectShadow_type,
    (&glsl_float_type, SHAPE_2D_RECT_SHADOW))
BUILTIN_TYPE(Type_sampler, glsl_sampler1DArrayShadow_type,
    (&glsl_float_type, SHAPE_1D_ARRAY_SHADOW))
BUILTIN_TYPE(Type_sampler, glsl_sampler2DArrayShadow_type,
    (&glsl_float_type, SHAPE_2D_ARRAY_SHADOW))
BUILTIN_TYPE(Type_sampler, glsl_samplerCubeShadow_type,
    (&glsl_float_type, SHAPE_CUBE_SHADOW))
BUILTIN_TYPE(Type_sampler, glsl_samplerCubeArrayShadow_type,
    (&glsl_float_type, SHAPE_CUBE_ARRAY_SHADOW))

// GL_OES_EGL_image_external
BUILTIN_TYPE(Type_sampler, glsl_samplerExternalOES_type, (&glsl_float_type, SHAPE_EXTERNAL_OES))

// Signed Integer Opaque Types
BUILTIN_TYPE(Type_sampler, glsl_isampler1D_type,        (&glsl_float_type, SHAPE_1D))
BUILTIN_TYPE(Type_image,   glsl_iimage1D_type,          (&glsl_float_type, SHAPE_1D))
BUILTIN_TYPE(Type_sampler, glsl_isampler2D_type,        (&glsl_float_type, SHAPE_2D))
BUILTIN_TYPE(Type_image,   glsl_iimage2D_type,          (&glsl_float_type, SHAPE_2D))
BUILTIN_TYPE(Type_sampler, glsl_isampler3D_type,        (&glsl_float_type, SHAPE_3D))
BUILTIN_TYPE(Type_image,   glsl_iimage3D_type,          (&glsl_float_type, SHAPE_3D))
BUILTIN_TYPE(Type_sampler, glsl_isamplerCube_type,      (&glsl_float_type, SHAPE_CUBE))
BUILTIN_TYPE(Type_image,   glsl_iimageCube_type,        (&glsl_float_type, SHAPE_CUBE))
BUILTIN_TYPE(Type_sampler, glsl_isampler2DRect_type,    (&glsl_float_type, SHAPE_2D_RECT))
BUILTIN_TYPE(Type_image,   glsl_iimage2DRect_type,      (&glsl_float_type, SHAPE_2D_RECT))
BUILTIN_TYPE(Type_sampler, glsl_isampler1DArray_type,   (&glsl_float_type, SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_iimage1DArray_type,     (&glsl_float_type, SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_isampler2DArray_type,   (&glsl_float_type, SHAPE_2D_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_iimage2DArray_type,     (&glsl_float_type, SHAPE_2D_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_isamplerBuffer_type,    (&glsl_float_type, SHAPE_BUFFER))
BUILTIN_TYPE(Type_image,   glsl_iimageBuffer_type,      (&glsl_float_type, SHAPE_BUFFER))
BUILTIN_TYPE(Type_sampler, glsl_isampler2DMS_type,      (&glsl_float_type, SHAPE_2DMS))
BUILTIN_TYPE(Type_image,   glsl_iimage2DMS_type,        (&glsl_float_type, SHAPE_2DMS))
BUILTIN_TYPE(Type_sampler, glsl_isampler2DMSArray_type, (&glsl_float_type, SHAPE_2DMS_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_iimage2DMSArray_type,   (&glsl_float_type, SHAPE_2DMS_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_isamplerCubeArray_type, (&glsl_float_type, SHAPE_CUBE_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_iimageCubeArray_type,   (&glsl_float_type, SHAPE_CUBE_ARRAY))

// Unsigned Integer Opaque Types
BUILTIN_TYPE(Type_atomic_uint, glsl_atomic_uint_type, )

BUILTIN_TYPE(Type_sampler, glsl_usampler1D_type,        (&glsl_float_type, SHAPE_1D))
BUILTIN_TYPE(Type_image,   glsl_uimage1D_type,          (&glsl_float_type, SHAPE_1D))
BUILTIN_TYPE(Type_sampler, glsl_usampler2D_type,        (&glsl_float_type, SHAPE_2D))
BUILTIN_TYPE(Type_image,   glsl_uimage2D_type,          (&glsl_float_type, SHAPE_2D))
BUILTIN_TYPE(Type_sampler, glsl_usampler3D_type,        (&glsl_float_type, SHAPE_3D))
BUILTIN_TYPE(Type_image,   glsl_uimage3D_type,          (&glsl_float_type, SHAPE_3D))
BUILTIN_TYPE(Type_sampler, glsl_usamplerCube_type,      (&glsl_float_type, SHAPE_CUBE))
BUILTIN_TYPE(Type_image,   glsl_uimageCube_type,        (&glsl_float_type, SHAPE_CUBE))
BUILTIN_TYPE(Type_sampler, glsl_usampler2DRect_type,    (&glsl_float_type, SHAPE_2D_RECT))
BUILTIN_TYPE(Type_image,   glsl_uimage2DRect_type,      (&glsl_float_type, SHAPE_2D_RECT))
BUILTIN_TYPE(Type_sampler, glsl_usampler1DArray_type,   (&glsl_float_type, SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_uimage1DArray_type,     (&glsl_float_type, SHAPE_1D_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_usampler2DArray_type,   (&glsl_float_type, SHAPE_2D_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_uimage2DArray_type,     (&glsl_float_type, SHAPE_2D_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_usamplerBuffer_type,    (&glsl_float_type, SHAPE_BUFFER))
BUILTIN_TYPE(Type_image,   glsl_uimageBuffer_type,      (&glsl_float_type, SHAPE_BUFFER))
BUILTIN_TYPE(Type_sampler, glsl_usampler2DMS_type,      (&glsl_float_type, SHAPE_2DMS))
BUILTIN_TYPE(Type_image,   glsl_uimage2DMS_type,        (&glsl_float_type, SHAPE_2DMS))
BUILTIN_TYPE(Type_sampler, glsl_usampler2DMSArray_type, (&glsl_float_type, SHAPE_2DMS_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_uimage2DMSArray_type,   (&glsl_float_type, SHAPE_2DMS_ARRAY))
BUILTIN_TYPE(Type_sampler, glsl_usamplerCubeArray_type, (&glsl_float_type, SHAPE_CUBE_ARRAY))
BUILTIN_TYPE(Type_image,   glsl_uimageCubeArray_type,   (&glsl_float_type, SHAPE_CUBE_ARRAY))

#undef BUILTIN_TYPE
