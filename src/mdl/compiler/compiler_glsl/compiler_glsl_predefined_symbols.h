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

// can be included more than once ...

#ifndef OPERATOR_SYM
// Define an operator symbol.
#define OPERATOR_SYM(sym_name, id, name)
#endif
#ifndef OPERATOR_SYM_UNARY
// Define an unary operator symbol.
#define OPERATOR_SYM_UNARY(sym_name, id, name) OPERATOR_SYM(sym_name, id, name)
#endif
#ifndef OPERATOR_SYM_BINARY
// Define an unary operator symbol.
#define OPERATOR_SYM_BINARY(sym_name, id, name) OPERATOR_SYM(sym_name, id, name)
#endif
#ifndef OPERATOR_SYM_TERNARY
// Define an unary operator symbol.
#define OPERATOR_SYM_TERNARY(sym_name, id, name) OPERATOR_SYM(sym_name, id, name)
#endif
#ifndef OPERATOR_SYM_VARIADIC
// Define an unary operator symbol.
#define OPERATOR_SYM_VARIADIC(sym_name, id, name) OPERATOR_SYM(sym_name, id, name)
#endif

#ifndef PREDEF_SYM
// Define a predefined symbol.
#define PREDEF_SYM(sym_name, id, name)
#endif

// unary operators
OPERATOR_SYM_UNARY(sym_bitwise_complement,  OK_BITWISE_COMPLEMENT, "operator~")
OPERATOR_SYM_UNARY(sym_logical_not,         OK_LOGICAL_NOT,        "operator!")
OPERATOR_SYM_UNARY(sym_positive,            OK_POSITIVE,           "operator+")
OPERATOR_SYM_UNARY(sym_negative,            OK_NEGATIVE,           "operator-")
OPERATOR_SYM_UNARY(sym_pre_increment,       OK_PRE_INCREMENT,      "operator++")
OPERATOR_SYM_UNARY(sym_pre_decrement,       OK_PRE_DECREMENT,      "operator--")
OPERATOR_SYM_UNARY(sym_post_increment,      OK_POST_INCREMENT,     "operator++")
OPERATOR_SYM_UNARY(sym_post_decrement,      OK_POST_DECREMENT,     "operator--")

// binary operator
OPERATOR_SYM_BINARY(sym_select,             OK_SELECT,             "operator.")
OPERATOR_SYM_BINARY(sym_array_index,        OK_ARRAY_INDEX,        "operator[]")
OPERATOR_SYM_BINARY(sym_multiply,           OK_MULTIPLY,           "operator*")
OPERATOR_SYM_BINARY(sym_divide,             OK_DIVIDE,             "operator/")
OPERATOR_SYM_BINARY(sym_modulo,             OK_MODULO,             "operator%")
OPERATOR_SYM_BINARY(sym_plus,               OK_PLUS,               "operator+")
OPERATOR_SYM_BINARY(sym_minus,              OK_MINUS,              "operator-")
OPERATOR_SYM_BINARY(sym_shift_left,         OK_SHIFT_LEFT,         "operator<<")
OPERATOR_SYM_BINARY(sym_shift_right,        OK_SHIFT_RIGHT,        "operator>>")
OPERATOR_SYM_BINARY(sym_less,               OK_LESS,               "operator<")
OPERATOR_SYM_BINARY(sym_less_or_equal,      OK_LESS_OR_EQUAL,      "operator<=")
OPERATOR_SYM_BINARY(sym_greater_or_equal,   OK_GREATER_OR_EQUAL,   "operator>=")
OPERATOR_SYM_BINARY(sym_greater,            OK_GREATER,            "operator>")
OPERATOR_SYM_BINARY(sym_equal,              OK_EQUAL,              "operator==")
OPERATOR_SYM_BINARY(sym_not_equal,          OK_NOT_EQUAL,          "operator!=")
OPERATOR_SYM_BINARY(sym_bitwise_and,        OK_BITWISE_AND,        "operator&")
OPERATOR_SYM_BINARY(sym_bitwise_xor,        OK_BITWISE_XOR,        "operator^")
OPERATOR_SYM_BINARY(sym_bitwise_or,         OK_BITWISE_OR,         "operator|")
OPERATOR_SYM_BINARY(sym_logical_and,        OK_LOGICAL_AND,        "operator&&")
OPERATOR_SYM_BINARY(sym_logical_or,         OK_LOGICAL_OR,         "operator||")
OPERATOR_SYM_BINARY(sym_assign,             OK_ASSIGN,             "operator=")
OPERATOR_SYM_BINARY(sym_multiply_assign,    OK_MULTIPLY_ASSIGN,    "operator*=")
OPERATOR_SYM_BINARY(sym_divide_assign,      OK_DIVIDE_ASSIGN,      "operator/=")
OPERATOR_SYM_BINARY(sym_modulo_assign,      OK_MODULO_ASSIGN,      "operator%=")
OPERATOR_SYM_BINARY(sym_plus_assign,        OK_PLUS_ASSIGN,        "operator+=")
OPERATOR_SYM_BINARY(sym_minus_assign,       OK_MINUS_ASSIGN,       "operator-=")
OPERATOR_SYM_BINARY(sym_shift_left_assign,  OK_SHIFT_LEFT_ASSIGN,  "operator<<=")
OPERATOR_SYM_BINARY(sym_shift_right_assign, OK_SHIFT_RIGHT_ASSIGN, "operator>>=")
OPERATOR_SYM_BINARY(sym_bitwise_or_assign,  OK_BITWISE_OR_ASSIGN,  "operator|=")
OPERATOR_SYM_BINARY(sym_bitwise_xor_assign, OK_BITWISE_XOR_ASSIGN, "operator^=")
OPERATOR_SYM_BINARY(sym_bitwise_and_assign, OK_BITWISE_AND_ASSIGN, "operator&=")
OPERATOR_SYM_BINARY(sym_sequence,           OK_SEQUENCE,           "operator,")

// ternary operator
OPERATOR_SYM_TERNARY(sym_ternary,           OK_TERNARY,            "operator?")

// variadic operator
OPERATOR_SYM_VARIADIC(sym_call,             OK_CALL,               "operator()")

// predefined names
PREDEF_SYM(sym_error, SYM_ERROR,   "<ERROR>")

// type names
PREDEF_SYM(sym_void,                   SYM_TYPE_VOID,                   "void")
PREDEF_SYM(sym_bool,                   SYM_TYPE_BOOL,                   "bool")
PREDEF_SYM(sym_bvec2,                  SYM_TYPE_BVEC2,                  "bvec2")
PREDEF_SYM(sym_bvec3,                  SYM_TYPE_BVEC3,                  "bvec3")
PREDEF_SYM(sym_bvec4,                  SYM_TYPE_BVEC4,                  "bvec4")
PREDEF_SYM(sym_int,                    SYM_TYPE_INT,                    "int")
PREDEF_SYM(sym_ivec2,                  SYM_TYPE_IVEC2,                  "ivec2")
PREDEF_SYM(sym_ivec3,                  SYM_TYPE_IVEC3,                  "ivec3")
PREDEF_SYM(sym_ivec4,                  SYM_TYPE_IVEC4,                  "ivec4")
PREDEF_SYM(sym_uint,                   SYM_TYPE_UINT,                   "uint")
PREDEF_SYM(sym_uvec2,                  SYM_TYPE_UVEC2,                  "uvec2")
PREDEF_SYM(sym_uvec3,                  SYM_TYPE_UVEC3,                  "uvec3")
PREDEF_SYM(sym_uvec4,                  SYM_TYPE_UVEC4,                  "uvec4")
PREDEF_SYM(sym_float16_t,              SYM_TYPE_FLOAT16_T,              "float16_t")
PREDEF_SYM(sym_float,                  SYM_TYPE_FLOAT,                  "float")
PREDEF_SYM(sym_vec2,                   SYM_TYPE_VEC2,                   "vec2")
PREDEF_SYM(sym_vec3,                   SYM_TYPE_VEC3,                   "vec3")
PREDEF_SYM(sym_vec4,                   SYM_TYPE_VEC4,                   "vec4")
PREDEF_SYM(sym_double,                 SYM_TYPE_DOUBLE,                 "double")
PREDEF_SYM(sym_dvec2,                  SYM_TYPE_DVEC2,                  "dvec2")
PREDEF_SYM(sym_dvec3,                  SYM_TYPE_DVEC3,                  "dvec3")
PREDEF_SYM(sym_dvec4,                  SYM_TYPE_DVEC4,                  "dvec4")
PREDEF_SYM(sym_mat2,                   SYM_TYPE_MAT2,                   "mat2")
PREDEF_SYM(sym_mat2x3,                 SYM_TYPE_MAT2X3,                 "mat2x3")
PREDEF_SYM(sym_mat2x4,                 SYM_TYPE_MAT2X4,                 "mat2x4")
PREDEF_SYM(sym_mat3x2,                 SYM_TYPE_MAT3X2,                 "mat3x2")
PREDEF_SYM(sym_mat3,                   SYM_TYPE_MAT3,                   "mat3")
PREDEF_SYM(sym_mat3x4,                 SYM_TYPE_MAT3X4,                 "mat3x4")
PREDEF_SYM(sym_mat4x2,                 SYM_TYPE_MAT4X2,                 "mat4x2")
PREDEF_SYM(sym_mat4x3,                 SYM_TYPE_MAT4X3,                 "mat4x3")
PREDEF_SYM(sym_mat4,                   SYM_TYPE_MAT4,                   "mat4")
PREDEF_SYM(sym_dmat2,                  SYM_TYPE_DMAT2,                  "dmat2")
PREDEF_SYM(sym_dmat2x3,                SYM_TYPE_DMAT2X3,                "dmat2x3")
PREDEF_SYM(sym_dmat2x4,                SYM_TYPE_DMAT2X4,                "dmat2x4")
PREDEF_SYM(sym_dmat3x2,                SYM_TYPE_DMAT3X2,                "dmat3x2")
PREDEF_SYM(sym_dmat3,                  SYM_TYPE_DMAT3,                  "dmat3")
PREDEF_SYM(sym_dmat3x4,                SYM_TYPE_DMAT3X4,                "dmat3x4")
PREDEF_SYM(sym_dmat4x2,                SYM_TYPE_DMAT4X2,                "dmat4x2")
PREDEF_SYM(sym_dmat4x3,                SYM_TYPE_DMAT4X3,                "dmat4x3")
PREDEF_SYM(sym_dmat4,                  SYM_TYPE_DMAT4,                  "dmat4")
PREDEF_SYM(sym_atomic_uint,            SYM_TYPE_ATOMIC_UINT,            "atomic_uint")
PREDEF_SYM(sym_int8_t,                 SYM_TYPE_INT8_T,                 "int8_t")
PREDEF_SYM(sym_i8vec2,                 SYM_TYPE_I8VEC2,                 "i8vec2")
PREDEF_SYM(sym_i8vec3,                 SYM_TYPE_I8VEC3,                 "i8vec3")
PREDEF_SYM(sym_i8vec4,                 SYM_TYPE_I8VEC4,                 "i8vec4")
PREDEF_SYM(sym_uint8_t,                SYM_TYPE_UINT8_T,                "uint8_t")
PREDEF_SYM(sym_u8vec2,                 SYM_TYPE_U8VEC2,                 "u8vec2")
PREDEF_SYM(sym_u8vec3,                 SYM_TYPE_U8VEC3,                 "u8vec3")
PREDEF_SYM(sym_u8vec4,                 SYM_TYPE_U8VEC4,                 "u8vec4")
PREDEF_SYM(sym_int16_t,                SYM_TYPE_INT16_T,                "int16_t")
PREDEF_SYM(sym_i16vec2,                SYM_TYPE_I16VEC2,                "i16vec2")
PREDEF_SYM(sym_i16vec3,                SYM_TYPE_I16VEC3,                "i16vec3")
PREDEF_SYM(sym_i16vec4,                SYM_TYPE_I16VEC4,                "i16vec4")
PREDEF_SYM(sym_uint16_t,               SYM_TYPE_UINT16_T,               "uint16_t")
PREDEF_SYM(sym_u16vec2,                SYM_TYPE_U16VEC2,                "u16vec2")
PREDEF_SYM(sym_u16vec3,                SYM_TYPE_U16VEC3,                "u16vec3")
PREDEF_SYM(sym_u16vec4,                SYM_TYPE_U16VEC4,                "u16vec4")
PREDEF_SYM(sym_int32_t,                SYM_TYPE_INT32_T,                "int32_t")
PREDEF_SYM(sym_i32vec2,                SYM_TYPE_I32VEC2,                "i32vec2")
PREDEF_SYM(sym_i32vec3,                SYM_TYPE_I32VEC3,                "i32vec3")
PREDEF_SYM(sym_i32vec4,                SYM_TYPE_I32VEC4,                "i32vec4")
PREDEF_SYM(sym_uint32_t,               SYM_TYPE_UINT32_T,               "uint32_t")
PREDEF_SYM(sym_u32vec2,                SYM_TYPE_U32VEC2,                "u32vec2")
PREDEF_SYM(sym_u32vec3,                SYM_TYPE_U32VEC3,                "u32vec3")
PREDEF_SYM(sym_u32vec4,                SYM_TYPE_U32VEC4,                "u32vec4")
PREDEF_SYM(sym_int64_t,                SYM_TYPE_INT64_T,                "int64_t")
PREDEF_SYM(sym_i64vec2,                SYM_TYPE_I64VEC2,                "i64vec2")
PREDEF_SYM(sym_i64vec3,                SYM_TYPE_I64VEC3,                "i64vec3")
PREDEF_SYM(sym_i64vec4,                SYM_TYPE_I64VEC4,                "i64vec4")
PREDEF_SYM(sym_uint64_t,               SYM_TYPE_UINT64_T,               "uint64_t")
PREDEF_SYM(sym_u64vec2,                SYM_TYPE_U64VEC2,                "u64vec2")
PREDEF_SYM(sym_u64vec3,                SYM_TYPE_U64VEC3,                "u64vec3")
PREDEF_SYM(sym_u64vec4,                SYM_TYPE_U64VEC4,                "u64vec4")
PREDEF_SYM(sym_sampler1d,              SYM_TYPE_SAMPLER1D,              "sampler1d")
PREDEF_SYM(sym_sampler2d,              SYM_TYPE_SAMPLER2D,              "sampler2d")
PREDEF_SYM(sym_sampler3d,              SYM_TYPE_SAMPLER3D,              "sampler3d")
PREDEF_SYM(sym_samplercube,            SYM_TYPE_SAMPLERCUBE,            "samplercube")
PREDEF_SYM(sym_sampler1dshadow,        SYM_TYPE_SAMPLER1DSHADOW,        "sampler1dshadow")
PREDEF_SYM(sym_sampler2dshadow,        SYM_TYPE_SAMPLER2DSHADOW,        "sampler2dshadow")
PREDEF_SYM(sym_samplercubeshadow,      SYM_TYPE_SAMPLERCUBESHADOW,      "samplercubeshadow")
PREDEF_SYM(sym_sampler1darray,         SYM_TYPE_SAMPLER1DARRAY,         "sampler1darray")
PREDEF_SYM(sym_sampler2darray,         SYM_TYPE_SAMPLER2DARRAY,         "sampler2darray")
PREDEF_SYM(sym_sampler1darrayshadow,   SYM_TYPE_SAMPLER1DARRAYSHADOW,   "sampler1darrayshadow")
PREDEF_SYM(sym_sampler2darrayshadow,   SYM_TYPE_SAMPLER2DARRAYSHADOW,   "sampler2darrayshadow")
PREDEF_SYM(sym_samplercubearray,       SYM_TYPE_SAMPLERCUBEARRAY,       "samplercubearray")
PREDEF_SYM(sym_samplercubearrayshadow, SYM_TYPE_SAMPLERCUBEARRAYSHADOW, "samplercubearrayshadow")
PREDEF_SYM(sym_isampler1d,             SYM_TYPE_ISAMPLER1D,             "isampler1d")
PREDEF_SYM(sym_isampler2d,             SYM_TYPE_ISAMPLER2D,             "isampler2d")
PREDEF_SYM(sym_isampler3d,             SYM_TYPE_ISAMPLER3D,             "isampler3d")
PREDEF_SYM(sym_isamplercube,           SYM_TYPE_ISAMPLERCUBE,           "isamplercube")
PREDEF_SYM(sym_isampler1darray,        SYM_TYPE_ISAMPLER1DARRAY,        "isampler1darray")
PREDEF_SYM(sym_isampler2darray,        SYM_TYPE_ISAMPLER2DARRAY,        "isampler2darray")
PREDEF_SYM(sym_isamplercubearray,      SYM_TYPE_ISAMPLERCUBEARRAY,      "isamplercubearray")
PREDEF_SYM(sym_usampler1d,             SYM_TYPE_USAMPLER1D,             "usampler1d")
PREDEF_SYM(sym_usampler2d,             SYM_TYPE_USAMPLER2D,             "usampler2d")
PREDEF_SYM(sym_usampler3d,             SYM_TYPE_USAMPLER3D,             "usampler3d")
PREDEF_SYM(sym_usamplercube,           SYM_TYPE_USAMPLERCUBE,           "usamplercube")
PREDEF_SYM(sym_usampler1darray,        SYM_TYPE_USAMPLER1DARRAY,        "usampler1darray")
PREDEF_SYM(sym_usampler2darray,        SYM_TYPE_USAMPLER2DARRAY,        "usampler2darray")
PREDEF_SYM(sym_usamplercubearray,      SYM_TYPE_USAMPLERCUBEARRAY,      "usamplercubearray")
PREDEF_SYM(sym_sampler2drect,          SYM_TYPE_SAMPLER2DRECT,          "sampler2drect")
PREDEF_SYM(sym_sampler2drectshadow,    SYM_TYPE_SAMPLER2DRECTSHADOW,    "sampler2drectshadow")
PREDEF_SYM(sym_isampler2drect,         SYM_TYPE_ISAMPLER2DRECT,         "isampler2drect")
PREDEF_SYM(sym_usampler2drect,         SYM_TYPE_USAMPLER2DRECT,         "usampler2drect")
PREDEF_SYM(sym_samplerbuffer,          SYM_TYPE_SAMPLERBUFFER,          "samplerbuffer")
PREDEF_SYM(sym_isamplerbuffer,         SYM_TYPE_ISAMPLERBUFFER,         "isamplerbuffer")
PREDEF_SYM(sym_usamplerbuffer,         SYM_TYPE_USAMPLERBUFFER,         "usamplerbuffer")
PREDEF_SYM(sym_sampler2dms,            SYM_TYPE_SAMPLER2DMS,            "sampler2dms")
PREDEF_SYM(sym_isampler2dms,           SYM_TYPE_ISAMPLER2DMS,           "isampler2dms")
PREDEF_SYM(sym_usampler2dms,           SYM_TYPE_USAMPLER2DMS,           "usampler2dms")
PREDEF_SYM(sym_sampler2dmsarray,       SYM_TYPE_SAMPLER2DMSARRAY,       "sampler2dmsarray")
PREDEF_SYM(sym_isampler2dmsarray,      SYM_TYPE_ISAMPLER2DMSARRAY,      "isampler2dmsarray")
PREDEF_SYM(sym_usampler2dmsarray,      SYM_TYPE_USAMPLER2DMSARRAY,      "usampler2dmsarray")
PREDEF_SYM(sym_image1d,                SYM_TYPE_IMAGE1D,                "image1d")
PREDEF_SYM(sym_iimage1d,               SYM_TYPE_IIMAGE1D,               "iimage1d")
PREDEF_SYM(sym_uimage1d,               SYM_TYPE_UIMAGE1D,               "uimage1d")
PREDEF_SYM(sym_image2d,                SYM_TYPE_IMAGE2D,                "image2d")
PREDEF_SYM(sym_iimage2d,               SYM_TYPE_IIMAGE2D,               "iimage2d")
PREDEF_SYM(sym_uimage2d,               SYM_TYPE_UIMAGE2D,               "uimage2d")
PREDEF_SYM(sym_image3d,                SYM_TYPE_IMAGE3D,                "image3d")
PREDEF_SYM(sym_iimage3d,               SYM_TYPE_IIMAGE3D,               "iimage3d")
PREDEF_SYM(sym_uimage3d,               SYM_TYPE_UIMAGE3D,               "uimage3d")
PREDEF_SYM(sym_image2drect,            SYM_TYPE_IMAGE2DRECT,            "image2drect")
PREDEF_SYM(sym_iimage2drect,           SYM_TYPE_IIMAGE2DRECT,           "iimage2drect")
PREDEF_SYM(sym_uimage2drect,           SYM_TYPE_UIMAGE2DRECT,           "uimage2drect")
PREDEF_SYM(sym_imagecube,              SYM_TYPE_IMAGECUBE,              "imagecube")
PREDEF_SYM(sym_iimagecube,             SYM_TYPE_IIMAGECUBE,             "iimagecube")
PREDEF_SYM(sym_uimagecube,             SYM_TYPE_UIMAGECUBE,             "uimagecube")
PREDEF_SYM(sym_imagebuffer ,           SYM_TYPE_IMAGEBUFFER ,           "imagebuffer ")
PREDEF_SYM(sym_iimagebuffer,           SYM_TYPE_IIMAGEBUFFER,           "iimagebuffer")
PREDEF_SYM(sym_uimagebuffer,           SYM_TYPE_UIMAGEBUFFER,           "uimagebuffer")
PREDEF_SYM(sym_image1darray,           SYM_TYPE_IMAGE1DARRAY,           "image1darray")
PREDEF_SYM(sym_iimage1darray,          SYM_TYPE_IIMAGE1DARRAY,          "iimage1darray")
PREDEF_SYM(sym_uimage1darray,          SYM_TYPE_UIMAGE1DARRAY,          "uimage1darray")
PREDEF_SYM(sym_image2darray,           SYM_TYPE_IMAGE2DARRAY,           "image2darray")
PREDEF_SYM(sym_iimage2darray,          SYM_TYPE_IIMAGE2DARRAY,          "iimage2darray")
PREDEF_SYM(sym_uimage2darray,          SYM_TYPE_UIMAGE2DARRAY,          "uimage2darray")
PREDEF_SYM(sym_imagecubearray,         SYM_TYPE_IMAGECUBEARRAY,         "imagecubearray")
PREDEF_SYM(sym_iimagecubearray,        SYM_TYPE_IIMAGECUBEARRAY,        "iimagecubearray")
PREDEF_SYM(sym_uimagecubearray,        SYM_TYPE_UIMAGECUBEARRAY,        "uimagecubearray")
PREDEF_SYM(sym_image2dms,              SYM_TYPE_IMAGE2DMS,              "image2dms")
PREDEF_SYM(sym_iimage2dms,             SYM_TYPE_IIMAGE2DMS,             "iimage2dms")
PREDEF_SYM(sym_uimage2dms,             SYM_TYPE_UIMAGE2DMS,             "uimage2dms")
PREDEF_SYM(sym_image2dmsarray,         SYM_TYPE_IMAGE2DMSARRAY,         "image2dmsarray ")
PREDEF_SYM(sym_iimage2dmsarray,        SYM_TYPE_IIMAGE2DMSARRAY,        "iimage2dmsarray")
PREDEF_SYM(sym_uimage2dmsarray,        SYM_TYPE_UIMAGE2DMSARRAY,        "uimage2dmsarray")
// GL_OES_EGL_image_external
PREDEF_SYM(sym_samplerexternaloes,     SYM_TYPE_SAMPLEREXTERNALOES,     "samplerExternalOES")

// constants
PREDEF_SYM(sym_cnst_true,  SYM_CNST_TRUE,  "true")
PREDEF_SYM(sym_cnst_false, SYM_CNST_FALSE, "false")

#undef PREDEF_SYM
#undef OPERATOR_SYM_VARIADIC
#undef OPERATOR_SYM_TERNARY
#undef OPERATOR_SYM_BINARY
#undef OPERATOR_SYM_UNARY
#undef OPERATOR_SYM
