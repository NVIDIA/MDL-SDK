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

#ifndef KEYWORD
#define KEYWORD(keyword) \
    PREDEF_SYM(sym_keyword ## keyword, SYM_KEYWORD, #keyword)
#endif

#ifndef RESERVED
#define RESERVED(word) \
    PREDEF_SYM(sym_reserved ## word, SYM_RESERVED, #word)
#endif

#define PREDEF_SYM_VECTOR(sym_name, id, name)    \
    PREDEF_SYM(sym_name ## 1, id ## 1, name "1") \
    PREDEF_SYM(sym_name ## 2, id ## 2, name "2") \
    PREDEF_SYM(sym_name ## 3, id ## 3, name "3") \
    PREDEF_SYM(sym_name ## 4, id ## 4, name "4")

#define PREDEF_SYM_MATRIX(sym_name, id, name)            \
    PREDEF_SYM(sym_name ##  1X1, id ##  1X1, name "1x1") \
    PREDEF_SYM(sym_name ##  1X2, id ##  1X2, name "1x2") \
    PREDEF_SYM(sym_name ##  1X3, id ##  1X3, name "1x3") \
    PREDEF_SYM(sym_name ##  1X4, id ##  1X4, name "1x4") \
    PREDEF_SYM(sym_name ##  2X1, id ##  2X1, name "2x1") \
    PREDEF_SYM(sym_name ##  2X2, id ##  2X2, name "2x2") \
    PREDEF_SYM(sym_name ##  2X3, id ##  2X3, name "2x3") \
    PREDEF_SYM(sym_name ##  2X4, id ##  2X4, name "2x4") \
    PREDEF_SYM(sym_name ##  3X1, id ##  3X1, name "3x1") \
    PREDEF_SYM(sym_name ##  3X2, id ##  3X2, name "3x2") \
    PREDEF_SYM(sym_name ##  3X3, id ##  3X3, name "3x3") \
    PREDEF_SYM(sym_name ##  3X4, id ##  3X4, name "3x4") \
    PREDEF_SYM(sym_name ##  4X1, id ##  4X1, name "4x1") \
    PREDEF_SYM(sym_name ##  4X2, id ##  4X2, name "4x2") \
    PREDEF_SYM(sym_name ##  4X3, id ##  4X3, name "4x3") \
    PREDEF_SYM(sym_name ##  4X4, id ##  4X4, name "4x4")

#define PREDEF_SYM_ALL(sym_name, id, name) \
    PREDEF_SYM(sym_name, id, name)         \
    PREDEF_SYM_VECTOR(sym_name, id, name)  \
    PREDEF_SYM_MATRIX(sym_name, id, name)

// keywords
KEYWORD(AppendStructuredBuffer)
KEYWORD(asm)
KEYWORD(asm_fragment)
KEYWORD(BlendState)
// KEYWORD(bool)
KEYWORD(break)
KEYWORD(Buffer)
KEYWORD(ByteAddressBuffer)
KEYWORD(case)
KEYWORD(cbuffer)
KEYWORD(centroid)
KEYWORD(class)
KEYWORD(column_major)
KEYWORD(compile)
KEYWORD(compile_fragment)
KEYWORD(CompileShader)
KEYWORD(const)
KEYWORD(continue)
KEYWORD(ComputeShader)
KEYWORD(ConsumeStructuredBuffer)
KEYWORD(default)
KEYWORD(DepthStencilState)
KEYWORD(DepthStencilView)
KEYWORD(discard)
KEYWORD(do)
// KEYWORD(double)
KEYWORD(DomainShader)
// KEYWORD(dword)
KEYWORD(else)
KEYWORD(export)
KEYWORD(extern)
// KEYWORD(false)
// KEYWORD(float)
KEYWORD(for)
KEYWORD(fxgroup)
KEYWORD(GeometryShader)
KEYWORD(groupshared)
// KEYWORD(half)
KEYWORD(Hullshader)
KEYWORD(if)
KEYWORD(in)
KEYWORD(inline)
KEYWORD(inout)
KEYWORD(InputPatch)
// KEYWORD(int)
KEYWORD(interface)
KEYWORD(line)
KEYWORD(lineadj)
KEYWORD(linear)
KEYWORD(LineStream)
KEYWORD(matrix)
// KEYWORD(min16float)
// KEYWORD(min10float)
// KEYWORD(min16int)
// KEYWORD(min12int)
// KEYWORD(min16uint)
KEYWORD(namespace)
KEYWORD(nointerpolation)
KEYWORD(noperspective)
KEYWORD(NULL)
KEYWORD(out)
KEYWORD(OutputPatch)
KEYWORD(packoffset)
KEYWORD(pass)
KEYWORD(pixelfragment)
KEYWORD(PixelShader)
KEYWORD(point)
KEYWORD(PointStream)
KEYWORD(precise)
KEYWORD(RasterizerState)
KEYWORD(RenderTargetView)
KEYWORD(return)
KEYWORD(register)
KEYWORD(row_major)
KEYWORD(RWBuffer)
KEYWORD(RWByteAddressBuffer)
KEYWORD(RWStructuredBuffer)
KEYWORD(RWTexture1D)
KEYWORD(RWTexture1DArray)
KEYWORD(RWTexture2D)
KEYWORD(RWTexture2DArray)
KEYWORD(RWTexture3D)
KEYWORD(sample)
// KEYWORD(sampler)
KEYWORD(SamplerState)
KEYWORD(SamplerComparisonState)
KEYWORD(shared)
KEYWORD(snorm)
KEYWORD(stateblock)
KEYWORD(stateblock_state)
KEYWORD(static)
KEYWORD(string)
KEYWORD(struct)
KEYWORD(switch)
KEYWORD(StructuredBuffer)
KEYWORD(tbuffer)
KEYWORD(technique)
KEYWORD(technique10)
KEYWORD(technique11)
// KEYWORD(texture)
// KEYWORD(Texture1D)
// KEYWORD(Texture1DArray)
// KEYWORD(Texture2D)
// KEYWORD(Texture2DArray)
KEYWORD(Texture2DMS)
KEYWORD(Texture2DMSArray)
// KEYWORD(Texture3D)
// KEYWORD(TextureCube)
KEYWORD(TextureCubeArray)
//KEYWORD(true)
KEYWORD(typedef)
KEYWORD(triangle)
KEYWORD(triangleadj)
KEYWORD(TriangleStream)
//KEYWORD(uint)
KEYWORD(uniform)
KEYWORD(unorm)
KEYWORD(unsigned)
KEYWORD(vector)
KEYWORD(vertexfragment)
KEYWORD(VertexShader)
//KEYWORD(void)
KEYWORD(volatile)
KEYWORD(while)

// reserved
RESERVED(auto)
RESERVED(catch)
RESERVED(char)
RESERVED(class)
RESERVED(const_cast)
RESERVED(default)
RESERVED(delete)
RESERVED(dynamic_cast)
RESERVED(enum)
RESERVED(explicit)
RESERVED(friend)
RESERVED(goto)
RESERVED(long)
RESERVED(mutable)
RESERVED(new)
RESERVED(operator)
RESERVED(private)
RESERVED(protected)
RESERVED(public)
RESERVED(reinterpret_cast)
RESERVED(short)
RESERVED(signed)
RESERVED(sizeof)
RESERVED(static_cast)
RESERVED(template)
RESERVED(this)
RESERVED(throw)
RESERVED(try)
RESERVED(typename)
RESERVED(union)
RESERVED(unsigned)
RESERVED(using)
RESERVED(virtual)

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
PREDEF_SYM(    sym_void,                   SYM_TYPE_VOID,                   "void")
PREDEF_SYM_ALL(sym_bool,                   SYM_TYPE_BOOL,                   "bool")
PREDEF_SYM_ALL(sym_int,                    SYM_TYPE_INT,                    "int")
PREDEF_SYM_ALL(sym_uint,                   SYM_TYPE_UINT,                   "uint")
PREDEF_SYM_ALL(sym_dword,                  SYM_TYPE_DWORD,                  "dword")
PREDEF_SYM_ALL(sym_half,                   SYM_TYPE_HALF,                   "half")
PREDEF_SYM_ALL(sym_float,                  SYM_TYPE_FLOAT,                  "float")
PREDEF_SYM_ALL(sym_double,                 SYM_TYPE_DOUBLE,                 "double")
PREDEF_SYM_ALL(sym_min12int,               SYM_TYPE_MIN12INT,               "min12int")
PREDEF_SYM_ALL(sym_min16int,               SYM_TYPE_MIN16INT,               "min16int")
PREDEF_SYM_ALL(sym_min16uint,              SYM_TYPE_MIN16UINT,              "min16uint")
PREDEF_SYM_ALL(sym_min10float,             SYM_TYPE_MIN10FLOAT,             "min10float")
PREDEF_SYM_ALL(sym_min16float,             SYM_TYPE_MIN16FLOAT,             "min16float")

PREDEF_SYM(    sym_sampler,                SYM_TYPE_SAMPLER,                "sampler")

PREDEF_SYM(    sym_texture,                SYM_TYPE_TEXTURE,                "texture")
PREDEF_SYM(    sym_texture1d,              SYM_TYPE_TEXTURE1D,              "Texture1D")
PREDEF_SYM(    sym_texture1darray,         SYM_TYPE_TEXTURE1DARRAY,         "Texture1DArray")
PREDEF_SYM(    sym_texture2d,              SYM_TYPE_TEXTURE2D,              "Texture2D")
PREDEF_SYM(    sym_texture2darray,         SYM_TYPE_TEXTURE2DARRAY,         "Texture2DArray")
PREDEF_SYM(    sym_texture3d,              SYM_TYPE_TEXTURE3D,              "Texture3D")
PREDEF_SYM(    sym_texturecube,            SYM_TYPE_TEXTURECUBE,            "TextureCube")

// constants
PREDEF_SYM(sym_cnst_true,  SYM_CNST_TRUE,  "true")
PREDEF_SYM(sym_cnst_false, SYM_CNST_FALSE, "false")

#undef RESERVED
#undef KEYWORD

#undef PREDEF_SYM_ALL
#undef PREDEF_SYM_MATRIX
#undef PREDEF_SYM_VECTOR

#undef PREDEF_SYM
#undef OPERATOR_SYM_VARIADIC
#undef OPERATOR_SYM_TERNARY
#undef OPERATOR_SYM_BINARY
#undef OPERATOR_SYM_UNARY
#undef OPERATOR_SYM
