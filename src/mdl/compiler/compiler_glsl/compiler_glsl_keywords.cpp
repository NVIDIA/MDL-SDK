/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <memory.h>
#include <string.h>

#include "compiler_glsl_keywords.h"
#include "compiler_glsl_assert.h"
#include "compiler_glsl_version.h"

struct Parser {
    enum TokenKind {
        TOK_EOF = 0,
        TOK_IDENTIFIER = 1,
        TOK_INTCONSTANT = 2,
        TOK_UINTCONSTANT = 3,
        TOK_INT64CONSTANT = 4,
        TOK_UINT64CONSTANT = 5,
        TOK_FLOATCONSTANT = 6,
        TOK_DOUBLECONSTANT = 7,
        TOK_EQUAL = 8,
        TOK_COMMA = 9,
        TOK_SEMICOLON = 10,
        TOK_DOT = 11,
        TOK_COLON = 12,
        TOK_LEFT_PAREN = 13,
        TOK_RIGHT_PAREN = 14,
        TOK_BANG = 15,
        TOK_TILDE = 16,
        TOK_DASH = 17,
        TOK_PLUS = 18,
        TOK_STAR = 19,
        TOK_SLASH = 20,
        TOK_PERCENT = 21,
        TOK_LEFT_ANGLE = 22,
        TOK_RIGHT_ANGLE = 23,
        TOK_VERTICAL_BAR = 24,
        TOK_CARET = 25,
        TOK_AMPERSAND = 26,
        TOK_QUESTION = 27,
        TOK_ARROW = 28,
        TOK_LE_OP = 29,
        TOK_GE_OP = 30,
        TOK_EQ_OP = 31,
        TOK_NE_OP = 32,
        TOK_AND_OP = 33,
        TOK_XOR_OP = 34,
        TOK_OR_OP = 35,
        TOK_INC_OP = 36,
        TOK_DEC_OP = 37,
        TOK_LEFT_OP = 38,
        TOK_RIGHT_OP = 39,
        TOK_MUL_ASSIGN = 40,
        TOK_DIV_ASSIGN = 41,
        TOK_MOD_ASSIGN = 42,
        TOK_ADD_ASSIGN = 43,
        TOK_SUB_ASSIGN = 44,
        TOK_LEFT_ASSIGN = 45,
        TOK_RIGHT_ASSIGN = 46,
        TOK_AND_ASSIGN = 47,
        TOK_XOR_ASSIGN = 48,
        TOK_OR_ASSIGN = 49,
        TOK_LEFT_BRACKET = 50,
        TOK_RIGHT_BRACKET = 51,
        TOK_LEFT_BRACE = 52,
        TOK_RIGHT_BRACE = 53,
        TOK_ATTRIBUTE = 54,
        TOK_CONST = 55,
        TOK_UNIFORM = 56,
        TOK_VARYING = 57,
        TOK_BUFFER = 58,
        TOK_SHARED = 59,
        TOK_COHERENT = 60,
        TOK_VOLATILE = 61,
        TOK_RESTRICT = 62,
        TOK_READONLY = 63,
        TOK_WRITEONLY = 64,
        TOK_ATOMIC_UINT = 65,
        TOK_LAYOUT = 66,
        TOK_CENTROID = 67,
        TOK_FLAT = 68,
        TOK_SMOOTH = 69,
        TOK_NOPERSPECTIVE = 70,
        TOK_PATCH = 71,
        TOK_SAMPLE = 72,
        TOK_BREAK = 73,
        TOK_CONTINUE = 74,
        TOK_DO = 75,
        TOK_FOR = 76,
        TOK_WHILE = 77,
        TOK_SWITCH = 78,
        TOK_CASE = 79,
        TOK_DEFAULT = 80,
        TOK_IF_ = 81,
        TOK_ELSE = 82,
        TOK_SUBROUTINE = 83,
        TOK_IN = 84,
        TOK_OUT = 85,
        TOK_INOUT = 86,
        TOK_FLOAT = 87,
        TOK_DOUBLE = 88,
        TOK_INT = 89,
        TOK_VOID = 90,
        TOK_BOOL = 91,
        TOK_TRUE = 92,
        TOK_FALSE = 93,
        TOK_INVARIANT = 94,
        TOK_PRECISE = 95,
        TOK_DISCARD = 96,
        TOK_RETURN = 97,
        TOK_MAT2 = 98,
        TOK_MAT3 = 99,
        TOK_MAT4 = 100,
        TOK_DMAT2 = 101,
        TOK_DMAT3 = 102,
        TOK_DMAT4 = 103,
        TOK_MAT2X2 = 104,
        TOK_MAT2X3 = 105,
        TOK_MAT2X4 = 106,
        TOK_DMAT2X2 = 107,
        TOK_DMAT2X3 = 108,
        TOK_DMAT2X4 = 109,
        TOK_MAT3X2 = 110,
        TOK_MAT3X3 = 111,
        TOK_MAT3X4 = 112,
        TOK_DMAT3X2 = 113,
        TOK_DMAT3X3 = 114,
        TOK_DMAT3X4 = 115,
        TOK_MAT4X2 = 116,
        TOK_MAT4X3 = 117,
        TOK_MAT4X4 = 118,
        TOK_DMAT4X2 = 119,
        TOK_DMAT4X3 = 120,
        TOK_DMAT4X4 = 121,
        TOK_VEC2 = 122,
        TOK_VEC3 = 123,
        TOK_VEC4 = 124,
        TOK_IVEC2 = 125,
        TOK_IVEC3 = 126,
        TOK_IVEC4 = 127,
        TOK_BVEC2 = 128,
        TOK_BVEC3 = 129,
        TOK_BVEC4 = 130,
        TOK_DVEC2 = 131,
        TOK_DVEC3 = 132,
        TOK_DVEC4 = 133,
        TOK_UINT = 134,
        TOK_UVEC2 = 135,
        TOK_UVEC3 = 136,
        TOK_UVEC4 = 137,
        TOK_INT8_T = 138,
        TOK_I8VEC2 = 139,
        TOK_I8VEC3 = 140,
        TOK_I8VEC4 = 141,
        TOK_UINT8_T = 142,
        TOK_U8VEC2 = 143,
        TOK_U8VEC3 = 144,
        TOK_U8VEC4 = 145,
        TOK_INT16_T = 146,
        TOK_I16VEC2 = 147,
        TOK_I16VEC3 = 148,
        TOK_I16VEC4 = 149,
        TOK_UINT16_T = 150,
        TOK_U16VEC2 = 151,
        TOK_U16VEC3 = 152,
        TOK_U16VEC4 = 153,
        TOK_INT32_T = 154,
        TOK_I32VEC2 = 155,
        TOK_I32VEC3 = 156,
        TOK_I32VEC4 = 157,
        TOK_UINT32_T = 158,
        TOK_U32VEC2 = 159,
        TOK_U32VEC3 = 160,
        TOK_U32VEC4 = 161,
        TOK_INT64_T = 162,
        TOK_I64VEC2 = 163,
        TOK_I64VEC3 = 164,
        TOK_I64VEC4 = 165,
        TOK_UINT64_T = 166,
        TOK_U64VEC2 = 167,
        TOK_U64VEC3 = 168,
        TOK_U64VEC4 = 169,
        TOK_FLOAT16_T = 170,
        TOK_F16VEC2 = 171,
        TOK_F16VEC3 = 172,
        TOK_F16VEC4 = 173,
        TOK_FLOAT32_T = 174,
        TOK_F32VEC2 = 175,
        TOK_F32VEC3 = 176,
        TOK_F32VEC4 = 177,
        TOK_FLOAT64_T = 178,
        TOK_F64VEC2 = 179,
        TOK_F64VEC3 = 180,
        TOK_F64VEC4 = 181,
        TOK_LOW_PRECISION = 182,
        TOK_MEDIUM_PRECISION = 183,
        TOK_HIGH_PRECISION = 184,
        TOK_PRECISION = 185,
        TOK_SAMPLER1D = 186,
        TOK_SAMPLER2D = 187,
        TOK_SAMPLER3D = 188,
        TOK_SAMPLERCUBE = 189,
        TOK_SAMPLER1DSHADOW = 190,
        TOK_SAMPLER2DSHADOW = 191,
        TOK_SAMPLERCUBESHADOW = 192,
        TOK_SAMPLER1DARRAY = 193,
        TOK_SAMPLER2DARRAY = 194,
        TOK_SAMPLER1DARRAYSHADOW = 195,
        TOK_SAMPLER2DARRAYSHADOW = 196,
        TOK_ISAMPLER1D = 197,
        TOK_ISAMPLER2D = 198,
        TOK_ISAMPLER3D = 199,
        TOK_ISAMPLERCUBE = 200,
        TOK_ISAMPLER1DARRAY = 201,
        TOK_ISAMPLER2DARRAY = 202,
        TOK_USAMPLER1D = 203,
        TOK_USAMPLER2D = 204,
        TOK_USAMPLER3D = 205,
        TOK_USAMPLERCUBE = 206,
        TOK_USAMPLER1DARRAY = 207,
        TOK_USAMPLER2DARRAY = 208,
        TOK_SAMPLER2DRECT = 209,
        TOK_SAMPLER2DRECTSHADOW = 210,
        TOK_ISAMPLER2DRECT = 211,
        TOK_USAMPLER2DRECT = 212,
        TOK_SAMPLERBUFFER = 213,
        TOK_ISAMPLERBUFFER = 214,
        TOK_USAMPLERBUFFER = 215,
        TOK_SAMPLER2DMS = 216,
        TOK_ISAMPLER2DMS = 217,
        TOK_USAMPLER2DMS = 218,
        TOK_SAMPLER2DMSARRAY = 219,
        TOK_ISAMPLER2DMSARRAY = 220,
        TOK_USAMPLER2DMSARRAY = 221,
        TOK_SAMPLERCUBEARRAY = 222,
        TOK_SAMPLERCUBEARRAYSHADOW = 223,
        TOK_ISAMPLERCUBEARRAY = 224,
        TOK_USAMPLERCUBEARRAY = 225,
        TOK_IMAGE1D = 226,
        TOK_IIMAGE1D = 227,
        TOK_UIMAGE1D = 228,
        TOK_IMAGE2D = 229,
        TOK_IIMAGE2D = 230,
        TOK_UIMAGE2D = 231,
        TOK_IMAGE3D = 232,
        TOK_IIMAGE3D = 233,
        TOK_UIMAGE3D = 234,
        TOK_IMAGE2DRECT = 235,
        TOK_IIMAGE2DRECT = 236,
        TOK_UIMAGE2DRECT = 237,
        TOK_IMAGECUBE = 238,
        TOK_IIMAGECUBE = 239,
        TOK_UIMAGECUBE = 240,
        TOK_IMAGEBUFFER = 241,
        TOK_IIMAGEBUFFER = 242,
        TOK_UIMAGEBUFFER = 243,
        TOK_IMAGE1DARRAY = 244,
        TOK_IIMAGE1DARRAY = 245,
        TOK_UIMAGE1DARRAY = 246,
        TOK_IMAGE2DARRAY = 247,
        TOK_IIMAGE2DARRAY = 248,
        TOK_UIMAGE2DARRAY = 249,
        TOK_IMAGECUBEARRAY = 250,
        TOK_IIMAGECUBEARRAY = 251,
        TOK_UIMAGECUBEARRAY = 252,
        TOK_IMAGE2DMS = 253,
        TOK_IIMAGE2DMS = 254,
        TOK_UIMAGE2DMS = 255,
        TOK_IMAGE2DMSARRAY = 256,
        TOK_IIMAGE2DMSARRAY = 257,
        TOK_UIMAGE2DMSARRAY = 258,
        TOK_STRUCT = 259,
        TOK_RESERVED = 260,
        TOK_PACKED = 261,
        TOK_SAMPLEREXTERNALOES = 262,
        TOK_COMMON = 263,
        TOK_PARTITION = 264,
        TOK_ACTIVE = 265,
        TOK_ASM = 266,
        TOK_CLASS = 267,
        TOK_UNION = 268,
        TOK_ENUM = 269,
        TOK_TYPEDEF = 270,
        TOK_TEMPLATE = 271,
        TOK_THIS = 272,
        TOK_RESOURCE = 273,
        TOK_GOTO = 274,
        TOK_INLINE = 275,
        TOK_NOINLINE = 276,
        TOK_PUBLIC = 277,
        TOK_STATIC = 278,
        TOK_EXTERN = 279,
        TOK_EXTERNAL = 280,
        TOK_INTERFACE = 281,
        TOK_LONG = 282,
        TOK_SHORT = 283,
        TOK_HALF = 284,
        TOK_FIXED = 285,
        TOK_UNSIGNED = 286,
        TOK_SUPERP = 287,
        TOK_INPUT = 288,
        TOK_OUTPUT = 289,
        TOK_HVEC2 = 290,
        TOK_HVEC3 = 291,
        TOK_HVEC4 = 292,
        TOK_FVEC2 = 293,
        TOK_FVEC3 = 294,
        TOK_FVEC4 = 295,
        TOK_SAMPLER3DRECT = 296,
        TOK_FILTER = 297,
        TOK_SIZEOF = 298,
        TOK_CAST = 299,
        TOK_NAMESPACE = 300,
        TOK_USING = 301,
        maxT = 302,
        noSym = 302
    };
};

namespace mi {
namespace mdl {
namespace glsl {

// Computes a hash for a C-string.
unsigned glsl_cstring_hash(char const *data, size_t len)
{
    GLSL_ASSERT(data != NULL);

    unsigned h = 0;
    for (size_t i = 0; i < len; ++i) {
        h = (h * 7) ^ unsigned(data[i]);
    }
    return h;
}

// Fill the map initially.
void GLSLKeywordMap::init()
{
    set("attribute", Parser::TOK_ATTRIBUTE);
    set("const", Parser::TOK_CONST);
    set("uniform", Parser::TOK_UNIFORM);
    set("varying", Parser::TOK_VARYING);
    set("buffer", Parser::TOK_BUFFER);
    set("shared", Parser::TOK_SHARED);
    set("coherent", Parser::TOK_COHERENT);
    set("volatile", Parser::TOK_VOLATILE);
    set("restrict", Parser::TOK_RESTRICT);
    set("readonly", Parser::TOK_READONLY);
    set("writeonly", Parser::TOK_WRITEONLY);
    set("atomic_uint", Parser::TOK_ATOMIC_UINT);
    set("layout", Parser::TOK_LAYOUT);
    set("centroid", Parser::TOK_CENTROID);
    set("flat", Parser::TOK_FLAT);
    set("smooth", Parser::TOK_SMOOTH);
    set("noperspective", Parser::TOK_NOPERSPECTIVE);
    set("patch", Parser::TOK_PATCH);
    set("sample", Parser::TOK_SAMPLE);
    set("break", Parser::TOK_BREAK);
    set("continue", Parser::TOK_CONTINUE);
    set("do", Parser::TOK_DO);
    set("for", Parser::TOK_FOR);
    set("while", Parser::TOK_WHILE);
    set("switch", Parser::TOK_SWITCH);
    set("case", Parser::TOK_CASE);
    set("default", Parser::TOK_DEFAULT);
    set("else", Parser::TOK_ELSE);
    set("subroutine", Parser::TOK_SUBROUTINE);
    set("in", Parser::TOK_IN);
    set("out", Parser::TOK_OUT);
    set("inout", Parser::TOK_INOUT);
    set("float", Parser::TOK_FLOAT);
    set("double", Parser::TOK_DOUBLE);
    set("int", Parser::TOK_INT);
    set("void", Parser::TOK_VOID);
    set("bool", Parser::TOK_BOOL);
    set("true", Parser::TOK_TRUE);
    set("false", Parser::TOK_FALSE);
    set("invariant", Parser::TOK_INVARIANT);
    set("precise", Parser::TOK_PRECISE);
    set("discard", Parser::TOK_DISCARD);
    set("return", Parser::TOK_RETURN);
    set("mat2", Parser::TOK_MAT2);
    set("mat3", Parser::TOK_MAT3);
    set("mat4", Parser::TOK_MAT4);
    set("dmat2", Parser::TOK_DMAT2);
    set("dmat3", Parser::TOK_DMAT3);
    set("dmat4", Parser::TOK_DMAT4);
    set("mat2x2", Parser::TOK_MAT2X2);
    set("mat2x3", Parser::TOK_MAT2X3);
    set("mat2x4", Parser::TOK_MAT2X4);
    set("dmat2x2", Parser::TOK_DMAT2X2);
    set("dmat2x3", Parser::TOK_DMAT2X3);
    set("dmat2x4", Parser::TOK_DMAT2X4);
    set("mat3x2", Parser::TOK_MAT3X2);
    set("mat3x3", Parser::TOK_MAT3X3);
    set("mat3x4", Parser::TOK_MAT3X4);
    set("dmat3x2", Parser::TOK_DMAT3X2);
    set("dmat3x3", Parser::TOK_DMAT3X3);
    set("dmat3x4", Parser::TOK_DMAT3X4);
    set("mat4x2", Parser::TOK_MAT4X2);
    set("mat4x3", Parser::TOK_MAT4X3);
    set("mat4x4", Parser::TOK_MAT4X4);
    set("dmat4x2", Parser::TOK_DMAT4X2);
    set("dmat4x3", Parser::TOK_DMAT4X3);
    set("dmat4x4", Parser::TOK_DMAT4X4);
    set("vec2", Parser::TOK_VEC2);
    set("vec3", Parser::TOK_VEC3);
    set("vec4", Parser::TOK_VEC4);
    set("ivec2", Parser::TOK_IVEC2);
    set("ivec3", Parser::TOK_IVEC3);
    set("ivec4", Parser::TOK_IVEC4);
    set("bvec2", Parser::TOK_BVEC2);
    set("bvec3", Parser::TOK_BVEC3);
    set("bvec4", Parser::TOK_BVEC4);
    set("dvec2", Parser::TOK_DVEC2);
    set("dvec3", Parser::TOK_DVEC3);
    set("dvec4", Parser::TOK_DVEC4);
    set("uint", Parser::TOK_UINT);
    set("uvec2", Parser::TOK_UVEC2);
    set("uvec3", Parser::TOK_UVEC3);
    set("uvec4", Parser::TOK_UVEC4);
    set("lowp", Parser::TOK_LOW_PRECISION);
    set("mediump", Parser::TOK_MEDIUM_PRECISION);
    set("highp", Parser::TOK_HIGH_PRECISION);
    set("precision", Parser::TOK_PRECISION);
    set("sampler1D", Parser::TOK_SAMPLER1D);
    set("sampler2D", Parser::TOK_SAMPLER2D);
    set("sampler3D", Parser::TOK_SAMPLER3D);
    set("samplerCube", Parser::TOK_SAMPLERCUBE);
    set("sampler1DShadow", Parser::TOK_SAMPLER1DSHADOW);
    set("sampler2DShadow", Parser::TOK_SAMPLER2DSHADOW);
    set("samplerCubeShadow", Parser::TOK_SAMPLERCUBESHADOW);
    set("sampler1DArray", Parser::TOK_SAMPLER1DARRAY);
    set("sampler2DArray", Parser::TOK_SAMPLER2DARRAY);
    set("sampler1DArrayShadow", Parser::TOK_SAMPLER1DARRAYSHADOW);
    set("sampler2DArrayShadow", Parser::TOK_SAMPLER2DARRAYSHADOW);
    set("isampler1D", Parser::TOK_ISAMPLER1D);
    set("isampler2D", Parser::TOK_ISAMPLER2D);
    set("isampler3D", Parser::TOK_ISAMPLER3D);
    set("isamplerCube", Parser::TOK_ISAMPLERCUBE);
    set("isampler1DArray", Parser::TOK_ISAMPLER1DARRAY);
    set("isampler2DArray", Parser::TOK_ISAMPLER2DARRAY);
    set("usampler1D", Parser::TOK_USAMPLER1D);
    set("usampler2D", Parser::TOK_USAMPLER2D);
    set("usampler3D", Parser::TOK_USAMPLER3D);
    set("usamplerCube", Parser::TOK_USAMPLERCUBE);
    set("usampler1DArray", Parser::TOK_USAMPLER1DARRAY);
    set("usampler2DArray", Parser::TOK_USAMPLER2DARRAY);
    set("sampler2DRect", Parser::TOK_SAMPLER2DRECT);
    set("sampler2DRectShadow", Parser::TOK_SAMPLER2DRECTSHADOW);
    set("isampler2DRect", Parser::TOK_ISAMPLER2DRECT);
    set("usampler2DRect", Parser::TOK_USAMPLER2DRECT);
    set("samplerBuffer", Parser::TOK_SAMPLERBUFFER);
    set("isamplerBuffer", Parser::TOK_ISAMPLERBUFFER);
    set("usamplerBuffer", Parser::TOK_USAMPLERBUFFER);
    set("sampler2DMS", Parser::TOK_SAMPLER2DMS);
    set("isampler2DMS", Parser::TOK_ISAMPLER2DMS);
    set("usampler2DMS", Parser::TOK_USAMPLER2DMS);
    set("sampler2DMSArray", Parser::TOK_SAMPLER2DMSARRAY);
    set("isampler2DMSArray", Parser::TOK_ISAMPLER2DMSARRAY);
    set("usampler2DMSArray", Parser::TOK_USAMPLER2DMSARRAY);
    set("samplerCubeArray", Parser::TOK_SAMPLERCUBEARRAY);
    set("samplerCubeArrayShadow", Parser::TOK_SAMPLERCUBEARRAYSHADOW);
    set("isamplerCubeArray", Parser::TOK_ISAMPLERCUBEARRAY);
    set("usamplerCubeArray", Parser::TOK_USAMPLERCUBEARRAY);
    set("image1D", Parser::TOK_IMAGE1D);
    set("iimage1D", Parser::TOK_IIMAGE1D);
    set("uimage1D", Parser::TOK_UIMAGE1D);
    set("image2D", Parser::TOK_IMAGE2D);
    set("iimage2D", Parser::TOK_IIMAGE2D);
    set("uimage2D", Parser::TOK_UIMAGE2D);
    set("image3D", Parser::TOK_IMAGE3D);
    set("iimage3D", Parser::TOK_IIMAGE3D);
    set("uimage3D", Parser::TOK_UIMAGE3D);
    set("image2DRect", Parser::TOK_IMAGE2DRECT);
    set("iimage2DRect", Parser::TOK_IIMAGE2DRECT);
    set("uimage2DRect", Parser::TOK_UIMAGE2DRECT);
    set("imageCube", Parser::TOK_IMAGECUBE);
    set("iimageCube", Parser::TOK_IIMAGECUBE);
    set("uimageCube", Parser::TOK_UIMAGECUBE);
    set("imageBuffer", Parser::TOK_IMAGEBUFFER);
    set("iimageBuffer", Parser::TOK_IIMAGEBUFFER);
    set("uimageBuffer", Parser::TOK_UIMAGEBUFFER);
    set("image1DArray", Parser::TOK_IMAGE1DARRAY);
    set("iimage1DArray", Parser::TOK_IIMAGE1DARRAY);
    set("uimage1DArray", Parser::TOK_UIMAGE1DARRAY);
    set("image2DArray", Parser::TOK_IMAGE2DARRAY);
    set("iimage2DArray", Parser::TOK_IIMAGE2DARRAY);
    set("uimage2DArray", Parser::TOK_UIMAGE2DARRAY);
    set("imageCubeArray", Parser::TOK_IMAGECUBEARRAY);
    set("iimageCubeArray", Parser::TOK_IIMAGECUBEARRAY);
    set("uimageCubeArray", Parser::TOK_UIMAGECUBEARRAY);
    set("image2DMS", Parser::TOK_IMAGE2DMS);
    set("iimage2DMS", Parser::TOK_IIMAGE2DMS);
    set("uimage2DMS", Parser::TOK_UIMAGE2DMS);
    set("image2DMSArray", Parser::TOK_IMAGE2DMSARRAY);
    set("iimage2DMSArray", Parser::TOK_IIMAGE2DMSARRAY);
    set("uimage2DMSArray", Parser::TOK_UIMAGE2DMSARRAY);
    set("struct", Parser::TOK_STRUCT);
    set("reserved", Parser::TOK_RESERVED);
    set("packed", Parser::TOK_PACKED);
    set("samplerExternalOES", Parser::TOK_SAMPLEREXTERNALOES);
    set("common", Parser::TOK_COMMON);
    set("partition", Parser::TOK_PARTITION);
    set("active", Parser::TOK_ACTIVE);
    set("asm", Parser::TOK_ASM);
    set("class", Parser::TOK_CLASS);
    set("union", Parser::TOK_UNION);
    set("enum", Parser::TOK_ENUM);
    set("typedef", Parser::TOK_TYPEDEF);
    set("template", Parser::TOK_TEMPLATE);
    set("this", Parser::TOK_THIS);
    set("resource", Parser::TOK_RESOURCE);
    set("goto", Parser::TOK_GOTO);
    set("inline", Parser::TOK_INLINE);
    set("noinline", Parser::TOK_NOINLINE);
    set("public", Parser::TOK_PUBLIC);
    set("static", Parser::TOK_STATIC);
    set("extern", Parser::TOK_EXTERN);
    set("external", Parser::TOK_EXTERNAL);
    set("interface", Parser::TOK_INTERFACE);
    set("long", Parser::TOK_LONG);
    set("short", Parser::TOK_SHORT);
    set("half", Parser::TOK_HALF);
    set("fixed", Parser::TOK_FIXED);
    set("unsigned", Parser::TOK_UNSIGNED);
    set("superp", Parser::TOK_SUPERP);
    set("input", Parser::TOK_INPUT);
    set("output", Parser::TOK_OUTPUT);
    set("hvec2", Parser::TOK_HVEC2);
    set("hvec3", Parser::TOK_HVEC3);
    set("hvec4", Parser::TOK_HVEC4);
    set("fvec2", Parser::TOK_FVEC2);
    set("fvec3", Parser::TOK_FVEC3);
    set("fvec4", Parser::TOK_FVEC4);
    set("sampler3DRect", Parser::TOK_SAMPLER3DRECT);
    set("filter", Parser::TOK_FILTER);
    set("sizeof", Parser::TOK_SIZEOF);
    set("cast", Parser::TOK_CAST);
    set("namespace", Parser::TOK_NAMESPACE);
    set("using", Parser::TOK_USING);
    set("if", Parser::TOK_IF_);
}

// Set the token value of a keyword depending on the keyword state.
void GLSLKeywordMap::set_keyword(char const *keyword, int val, GLSL_keyword_state state)
{
    switch (state) {
    case KS_IDENT:
        val = Parser::TOK_IDENTIFIER;
        break;
    case KS_RESERVED:
        val = Parser::TOK_RESERVED;
        break;
    case KS_KEYWORD:
        break;
    }
    set(keyword, val, state == KS_IDENT);
}

// Notify the the GLSLang version has changes.
void GLSLKeywordMap::glsl_version_changed(GLSLang_context &glslang_ctx)
{
    // handle some keywords that were added in later versions
#define KEYWORD(k, K)  set_keyword(k, Parser::TOK##K, glslang_ctx.keyword(KW##K));
#define RESERVED(k)    set_keyword(k, Parser::TOK_RESERVED, KS_RESERVED);

    KEYWORD("switch", _SWITCH);
    KEYWORD("default", _DEFAULT);
    KEYWORD("case", _CASE);
    KEYWORD("attribute", _ATTRIBUTE);
    KEYWORD("varying", _VARYING);
    KEYWORD("buffer", _BUFFER);
    KEYWORD("atomic_uint", _ATOMIC_UINT);
    KEYWORD("coherent", _COHERENT);
    KEYWORD("restrict", _RESTRICT);
    KEYWORD("readonly", _READONLY);
    KEYWORD("writeonly", _WRITEONLY);
    KEYWORD("volatile", _VOLATILE);
    KEYWORD("layout", _LAYOUT);
    KEYWORD("shared", _SHARED);
    KEYWORD("patch", _PATCH);
    KEYWORD("sample", _SAMPLE);
    KEYWORD("subroutine", _SUBROUTINE);
    KEYWORD("high_precision", _HIGH_PRECISION);
    KEYWORD("medium_precision", _MEDIUM_PRECISION);
    KEYWORD("low_precision", _LOW_PRECISION);
    KEYWORD("precision", _PRECISION);
    KEYWORD("mat2x2", _MAT2X2);
    KEYWORD("mat2x3", _MAT2X3);
    KEYWORD("mat2x4", _MAT2X4);
    KEYWORD("mat3x2", _MAT3X2);
    KEYWORD("mat3x3", _MAT3X3);
    KEYWORD("mat3x4", _MAT3X4);
    KEYWORD("mat4x2", _MAT4X2);
    KEYWORD("mat4x3", _MAT4X3);
    KEYWORD("mat4x4", _MAT4X4);
    KEYWORD("dmat2", _DMAT2);
    KEYWORD("dmat3", _DMAT3);
    KEYWORD("dmat4", _DMAT4);
    KEYWORD("dmat2x2", _DMAT2X2);
    KEYWORD("dmat2x3", _DMAT2X3);
    KEYWORD("dmat2x4", _DMAT2X4);
    KEYWORD("dmat3x2", _DMAT3X2);
    KEYWORD("dmat3x3", _DMAT3X3);
    KEYWORD("dmat3x4", _DMAT3X4);
    KEYWORD("dmat4x2", _DMAT4X2);
    KEYWORD("dmat4x3", _DMAT4X3);
    KEYWORD("dmat4x4", _DMAT4X4);
    KEYWORD("image1D", _IMAGE1D);
    KEYWORD("iimage1D", _IIMAGE1D);
    KEYWORD("uimage1D", _UIMAGE1D);
    KEYWORD("image1DArray", _IMAGE1DARRAY);
    KEYWORD("iimage1DArray", _IIMAGE1DARRAY);
    KEYWORD("uimage1DArray", _UIMAGE1DARRAY);
    KEYWORD("image2DRect", _IMAGE2DRECT);
    KEYWORD("iimage2DRect", _IIMAGE2DRECT);
    KEYWORD("uimage2DRect", _UIMAGE2DRECT);
    KEYWORD("imageBuffer", _IMAGEBUFFER);
    KEYWORD("iimageBuffer", _IIMAGEBUFFER);
    KEYWORD("uimageBuffer", _UIMAGEBUFFER);
    KEYWORD("image2D", _IMAGE2D);
    KEYWORD("iimage2D", _IIMAGE2D);
    KEYWORD("uimage2D", _UIMAGE2D);
    KEYWORD("image3D", _IMAGE3D);
    KEYWORD("iimage3D", _IIMAGE3D);
    KEYWORD("uimage3D", _UIMAGE3D);
    KEYWORD("imageCube", _IMAGECUBE);
    KEYWORD("iimageCube", _IIMAGECUBE);
    KEYWORD("uimageCube", _UIMAGECUBE);
    KEYWORD("image2DArray", _IMAGE2DARRAY);
    KEYWORD("iimage2DArray", _IIMAGE2DARRAY);
    KEYWORD("uimage2DArray", _UIMAGE2DARRAY);
    KEYWORD("imageCubeArray", _IMAGECUBEARRAY);
    KEYWORD("iimageCubeArray", _IIMAGECUBEARRAY);
    KEYWORD("uimageCubeArray", _UIMAGECUBEARRAY);
    KEYWORD("image2DMS", _IMAGE2DMS);
    KEYWORD("iimage2DMS", _IIMAGE2DMS);
    KEYWORD("uimage2DMS", _UIMAGE2DMS);
    KEYWORD("image2DMSArray", _IMAGE2DMSARRAY);
    KEYWORD("iimage2DMSArray", _IIMAGE2DMSARRAY);
    KEYWORD("uimage2DMSArray", _UIMAGE2DMSARRAY);
    KEYWORD("double", _DOUBLE);
    KEYWORD("dvec2", _DVEC2);
    KEYWORD("dvec3", _DVEC3);
    KEYWORD("dvec4", _DVEC4);
    KEYWORD("samplerCubeArray", _SAMPLERCUBEARRAY);
    KEYWORD("samplerCubeArrayShadow", _SAMPLERCUBEARRAYSHADOW);
    KEYWORD("isamplerCubeArray", _ISAMPLERCUBEARRAY);
    KEYWORD("usamplerCubeArray", _USAMPLERCUBEARRAY);
    KEYWORD("isampler1D", _ISAMPLER1D);
    KEYWORD("isampler1DArray", _ISAMPLER1DARRAY);
    KEYWORD("sampler1DArrayshadow", _SAMPLER1DARRAYSHADOW);
    KEYWORD("usampler1D", _USAMPLER1D);
    KEYWORD("usampler1DArray", _USAMPLER1DARRAY);
    KEYWORD("samplerBuffer", _SAMPLERBUFFER);
    KEYWORD("uint", _UINT);
    KEYWORD("uvec2", _UVEC2);
    KEYWORD("uvec3", _UVEC3);
    KEYWORD("uvec4", _UVEC4);
    KEYWORD("samplerCubeShadow", _SAMPLERCUBESHADOW);
    KEYWORD("sampler2DArray", _SAMPLER2DARRAY);
    KEYWORD("sampler2DArrayShadow", _SAMPLER2DARRAYSHADOW);
    KEYWORD("isampler2D", _ISAMPLER2D);
    KEYWORD("isampler3D", _ISAMPLER3D);
    KEYWORD("isamplerCube", _ISAMPLERCUBE);
    KEYWORD("isampler2DArray", _ISAMPLER2DARRAY);
    KEYWORD("usampler2D", _USAMPLER2D);
    KEYWORD("usampler3D", _USAMPLER3D);
    KEYWORD("usamplerCube", _USAMPLERCUBE);
    KEYWORD("usampler2DArray", _USAMPLER2DARRAY);
    KEYWORD("isampler2DRect", _ISAMPLER2DRECT);
    KEYWORD("usampler2DRect", _USAMPLER2DRECT);
    KEYWORD("isamplerBuffer", _ISAMPLERBUFFER);
    KEYWORD("usamplerBuffer", _USAMPLERBUFFER);
    KEYWORD("sampler2DMS", _SAMPLER2DMS);
    KEYWORD("isampler2DMS", _ISAMPLER2DMS);
    KEYWORD("usampler2DMS", _USAMPLER2DMS);
    KEYWORD("sampler2DMSArray", _SAMPLER2DMSARRAY);
    KEYWORD("isampler2DMSArray", _ISAMPLER2DMSARRAY);
    KEYWORD("usampler2DMSArray", _USAMPLER2DMSARRAY);
    KEYWORD("sampler1D", _SAMPLER1D);
    KEYWORD("sampler1DShadow", _SAMPLER1DSHADOW);
    KEYWORD("sampler3D", _SAMPLER3D);
    KEYWORD("sampler2DShadow", _SAMPLER2DSHADOW);
    KEYWORD("sampler2DRect", _SAMPLER2DRECT);
    KEYWORD("sampler2DRectShadow", _SAMPLER2DRECTSHADOW);
    KEYWORD("sampler1DArray", _SAMPLER1DARRAY);
    KEYWORD("samplerExternalOES", _SAMPLEREXTERNALOES);
    KEYWORD("noperspective", _NOPERSPECTIVE);
    KEYWORD("smooth", _SMOOTH);
    KEYWORD("flat", _FLAT);
    KEYWORD("centroid", _CENTROID);
    KEYWORD("precise", _PRECISE);
    KEYWORD("invariant", _INVARIANT);
    KEYWORD("packed", _PACKED);
    KEYWORD("resource", _RESOURCE);
    KEYWORD("superp", _SUPERP);
    KEYWORD("unsigned", _UNSIGNED);
    KEYWORD("int8_t", _INT8_T);
    KEYWORD("i8vec2", _I8VEC2);
    KEYWORD("i8vec3", _I8VEC3);
    KEYWORD("i8vec4", _I8VEC4);
    KEYWORD("uint8_t", _UINT8_T);
    KEYWORD("u8vec2", _U8VEC2);
    KEYWORD("u8vec3", _U8VEC3);
    KEYWORD("u8vec4", _U8VEC4);
    KEYWORD("int16_t", _INT16_T);
    KEYWORD("i16vec2", _I16VEC2);
    KEYWORD("i16vec3", _I16VEC3);
    KEYWORD("i16vec4", _I16VEC4);
    KEYWORD("uint16_t", _UINT16_T);
    KEYWORD("u16vec2", _U16VEC2);
    KEYWORD("u16vec3", _U16VEC3);
    KEYWORD("u16vec4", _U16VEC4);
    KEYWORD("int32_t", _INT32_T);
    KEYWORD("i32vec2", _I32VEC2);
    KEYWORD("i32vec3", _I32VEC3);
    KEYWORD("i32vec4", _I32VEC4);
    KEYWORD("uint32_t", _UINT32_T);
    KEYWORD("u32vec2", _U32VEC2);
    KEYWORD("u32vec3", _U32VEC3);
    KEYWORD("u32vec4", _U32VEC4);
    KEYWORD("int64_t", _INT64_T);
    KEYWORD("i64vec2", _I64VEC2);
    KEYWORD("i64vec3", _I64VEC3);
    KEYWORD("i64vec4", _I64VEC4);
    KEYWORD("uint64_t", _UINT64_T);
    KEYWORD("u64vec2", _U64VEC2);
    KEYWORD("u64vec3", _U64VEC3);
    KEYWORD("u64vec4", _U64VEC4);
    KEYWORD("float16_t", _FLOAT16_T);
    KEYWORD("f16vec2", _F16VEC2);
    KEYWORD("f16vec3", _F16VEC3);
    KEYWORD("f16vec4", _F16VEC4);
    KEYWORD("float32_t", _FLOAT32_T);
    KEYWORD("f32vec2", _F32VEC2);
    KEYWORD("f32vec3", _F32VEC3);
    KEYWORD("f32vec4", _F32VEC4);
    KEYWORD("float64_t", _FLOAT64_T);
    KEYWORD("f64vec2", _F64VEC2);
    KEYWORD("f64vec3", _F64VEC3);
    KEYWORD("f64vec4", _F64VEC4);

    RESERVED("common");
    RESERVED("partition");
    RESERVED("active");
    RESERVED("asm");
    RESERVED("class");
    RESERVED("union");
    RESERVED("enum");
    RESERVED("typedef");
    RESERVED("template");
    RESERVED("this");
    RESERVED("resource");
    RESERVED("goto");
    RESERVED("inline");
    RESERVED("noinline");
    RESERVED("public");
    RESERVED("static");
    RESERVED("extern");
    RESERVED("external");
    RESERVED("interface");
    RESERVED("long");
    RESERVED("short");
    RESERVED("half");
    RESERVED("fixed");
    RESERVED("superp");
    RESERVED("input");
    RESERVED("output");
    RESERVED("hvec2");
    RESERVED("hvec3");
    RESERVED("hvec4");
    RESERVED("fvec2");
    RESERVED("fvec3");
    RESERVED("fvec4");
    RESERVED("sampler3DRect");
    RESERVED("filter");
    RESERVED("sizeof");
    RESERVED("cast");
    RESERVED("namespace");
    RESERVED("using");

#undef RESERVED
#undef KEYWORD
}

/// Check if the given identifier is a keyword or a reserved word.
bool GLSLKeywordMap::keyword_or_reserved(
    char const *s,
    size_t len) const
{
    return get(len, s, Parser::TOK_IDENTIFIER) != Parser::TOK_IDENTIFIER;
}

} // namespace
} // namespace
} // namespace


