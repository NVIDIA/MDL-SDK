/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
OPERATOR_SYM_UNARY(sym_bitwise_complement, OK_BITWISE_COMPLEMENT, "operator~")
OPERATOR_SYM_UNARY(sym_logical_not, OK_LOGICAL_NOT, "operator!")
OPERATOR_SYM_UNARY(sym_positive, OK_POSITIVE, "operator+")
OPERATOR_SYM_UNARY(sym_negative, OK_NEGATIVE, "operator-")
OPERATOR_SYM_UNARY(sym_pre_increment, OK_PRE_INCREMENT, "operator++")
OPERATOR_SYM_UNARY(sym_pre_decrement, OK_PRE_DECREMENT, "operator--")
OPERATOR_SYM_UNARY(sym_post_increment, OK_POST_INCREMENT, "operator++")
OPERATOR_SYM_UNARY(sym_post_decrement, OK_POST_DECREMENT, "operator--")
OPERATOR_SYM_UNARY(sym_cast, OK_CAST, "operator_cast")

// binary operator
OPERATOR_SYM_BINARY(sym_select, OK_SELECT, "operator.")
OPERATOR_SYM_BINARY(sym_array_index, OK_ARRAY_INDEX, "operator[]")
OPERATOR_SYM_BINARY(sym_multiply, OK_MULTIPLY, "operator*")
OPERATOR_SYM_BINARY(sym_divide, OK_DIVIDE, "operator/")
OPERATOR_SYM_BINARY(sym_modulo, OK_MODULO, "operator%")
OPERATOR_SYM_BINARY(sym_plus, OK_PLUS, "operator+")
OPERATOR_SYM_BINARY(sym_minus, OK_MINUS, "operator-")
OPERATOR_SYM_BINARY(sym_shift_left, OK_SHIFT_LEFT, "operator<<")
OPERATOR_SYM_BINARY(sym_shift_right, OK_SHIFT_RIGHT, "operator>>")
OPERATOR_SYM_BINARY(sym_unsigned_shift_right, OK_UNSIGNED_SHIFT_RIGHT, "operator>>>")
OPERATOR_SYM_BINARY(sym_less, OK_LESS, "operator<")
OPERATOR_SYM_BINARY(sym_less_or_equal, OK_LESS_OR_EQUAL, "operator<=")
OPERATOR_SYM_BINARY(sym_greater_or_equal, OK_GREATER_OR_EQUAL, "operator>=")
OPERATOR_SYM_BINARY(sym_greater, OK_GREATER, "operator>")
OPERATOR_SYM_BINARY(sym_equal, OK_EQUAL, "operator==")
OPERATOR_SYM_BINARY(sym_not_equal, OK_NOT_EQUAL, "operator!=")
OPERATOR_SYM_BINARY(sym_bitwise_and, OK_BITWISE_AND, "operator&")
OPERATOR_SYM_BINARY(sym_bitwise_xor, OK_BITWISE_XOR, "operator^")
OPERATOR_SYM_BINARY(sym_bitwise_or, OK_BITWISE_OR, "operator|")
OPERATOR_SYM_BINARY(sym_logical_and, OK_LOGICAL_AND, "operator&&")
OPERATOR_SYM_BINARY(sym_logical_or, OK_LOGICAL_OR, "operator||")
OPERATOR_SYM_BINARY(sym_assign, OK_ASSIGN, "operator=")
OPERATOR_SYM_BINARY(sym_multiply_assign, OK_MULTIPLY_ASSIGN, "operator*=")
OPERATOR_SYM_BINARY(sym_divide_assign, OK_DIVIDE_ASSIGN, "operator/=")
OPERATOR_SYM_BINARY(sym_modulo_assign, OK_MODULO_ASSIGN, "operator%=")
OPERATOR_SYM_BINARY(sym_plus_assign, OK_PLUS_ASSIGN, "operator+=")
OPERATOR_SYM_BINARY(sym_minus_assign, OK_MINUS_ASSIGN, "operator-=")
OPERATOR_SYM_BINARY(sym_shift_left_assign, OK_SHIFT_LEFT_ASSIGN, "operator<<=")
OPERATOR_SYM_BINARY(sym_shift_right_assign, OK_SHIFT_RIGHT_ASSIGN, "operator>>=")
OPERATOR_SYM_BINARY(sym_unsigned_shift_right_assign, OK_UNSIGNED_SHIFT_RIGHT_ASSIGN, "operator>>>=")
OPERATOR_SYM_BINARY(sym_bitwise_or_assign, OK_BITWISE_OR_ASSIGN, "operator|=")
OPERATOR_SYM_BINARY(sym_bitwise_xor_assign, OK_BITWISE_XOR_ASSIGN, "operator^=")
OPERATOR_SYM_BINARY(sym_bitwise_and_assign, OK_BITWISE_AND_ASSIGN, "operator&=")
OPERATOR_SYM_BINARY(sym_sequence, OK_SEQUENCE, "operator,")

// ternary operator
OPERATOR_SYM_TERNARY(sym_ternary, OK_TERNARY, "operator?")

// variadic operator
OPERATOR_SYM_VARIADIC(sym_call, OK_CALL, "operator()")

// predefined names
PREDEF_SYM(sym_error, SYM_ERROR, "<ERROR>")
PREDEF_SYM(sym_star, SYM_STAR, "*")
PREDEF_SYM(sym_dot, SYM_DOT, ".")
PREDEF_SYM(sym_dotdot, SYM_DOTDOT, "..")

// type names
PREDEF_SYM(sym_bool, SYM_TYPE_BOOL, "bool")
PREDEF_SYM(sym_bool2, SYM_TYPE_BOOL2, "bool2")
PREDEF_SYM(sym_bool3, SYM_TYPE_BOOL3, "bool3")
PREDEF_SYM(sym_bool4, SYM_TYPE_BOOL4, "bool4")
PREDEF_SYM(sym_int, SYM_TYPE_INT, "int")
PREDEF_SYM(sym_int2, SYM_TYPE_INT2, "int2")
PREDEF_SYM(sym_int3, SYM_TYPE_INT3, "int3")
PREDEF_SYM(sym_int4, SYM_TYPE_INT4, "int4")
PREDEF_SYM(sym_float, SYM_TYPE_FLOAT, "float")
PREDEF_SYM(sym_float2, SYM_TYPE_FLOAT2, "float2")
PREDEF_SYM(sym_float3, SYM_TYPE_FLOAT3, "float3")
PREDEF_SYM(sym_float4, SYM_TYPE_FLOAT4, "float4")
PREDEF_SYM(sym_double, SYM_TYPE_DOUBLE, "double")
PREDEF_SYM(sym_double2, SYM_TYPE_DOUBLE2, "double2")
PREDEF_SYM(sym_double3, SYM_TYPE_DOUBLE3, "double3")
PREDEF_SYM(sym_double4, SYM_TYPE_DOUBLE4, "double4")
PREDEF_SYM(sym_float2x2, SYM_TYPE_FLOAT2X2, "float2x2")
PREDEF_SYM(sym_float2x3, SYM_TYPE_FLOAT2X3, "float2x3")
PREDEF_SYM(sym_float2x4, SYM_TYPE_FLOAT2X4, "float2x4")
PREDEF_SYM(sym_float3x2, SYM_TYPE_FLOAT3X2, "float3x2")
PREDEF_SYM(sym_float3x3, SYM_TYPE_FLOAT3X3, "float3x3")
PREDEF_SYM(sym_float3x4, SYM_TYPE_FLOAT3X4, "float3x4")
PREDEF_SYM(sym_float4x2, SYM_TYPE_FLOAT4X2, "float4x2")
PREDEF_SYM(sym_float4x3, SYM_TYPE_FLOAT4X3, "float4x3")
PREDEF_SYM(sym_float4x4, SYM_TYPE_FLOAT4X4, "float4x4")
PREDEF_SYM(sym_double2x2, SYM_TYPE_DOUBLE2X2, "double2x2")
PREDEF_SYM(sym_double2x3, SYM_TYPE_DOUBLE2X3, "double2x3")
PREDEF_SYM(sym_double2x4, SYM_TYPE_DOUBLE2X4, "double2x4")
PREDEF_SYM(sym_double3x2, SYM_TYPE_DOUBLE3X2, "double3x2")
PREDEF_SYM(sym_double3x3, SYM_TYPE_DOUBLE3X3, "double3x3")
PREDEF_SYM(sym_double3x4, SYM_TYPE_DOUBLE3X4, "double3x4")
PREDEF_SYM(sym_double4x2, SYM_TYPE_DOUBLE4X2, "double4x2")
PREDEF_SYM(sym_double4x3, SYM_TYPE_DOUBLE4X3, "double4x3")
PREDEF_SYM(sym_double4x4, SYM_TYPE_DOUBLE4X4, "double4x4")
PREDEF_SYM(sym_texture_2d, SYM_TYPE_TEXTURE_2D, "texture_2d")
PREDEF_SYM(sym_texture_3d, SYM_TYPE_TEXTURE_3D, "texture_3d")
PREDEF_SYM(sym_texture_cube, SYM_TYPE_TEXTURE_CUBE, "texture_cube")
PREDEF_SYM(sym_texture_ptex, SYM_TYPE_TEXTURE_PTEX, "texture_ptex")
PREDEF_SYM(sym_string, SYM_TYPE_STRING, "string")
PREDEF_SYM(sym_color, SYM_TYPE_COLOR, "color")
PREDEF_SYM(sym_light_profile, SYM_TYPE_LIGHT_PROFILE, "light_profile")
PREDEF_SYM(sym_bsdf_measurement, SYM_TYPE_BSDF_MEASUREMENT, "bsdf_measurement")
PREDEF_SYM(sym_bsdf, SYM_TYPE_BSDF, "bsdf")
PREDEF_SYM(sym_vdf, SYM_TYPE_VDF, "vdf")
PREDEF_SYM(sym_edf, SYM_TYPE_EDF, "edf")
PREDEF_SYM(sym_material_emission, SYM_TYPE_MATERIAL_EMISSION, "material_emission")
PREDEF_SYM(sym_material_surface, SYM_TYPE_MATERIAL_SURFACE, "material_surface")
PREDEF_SYM(sym_material_volume, SYM_TYPE_MATERIAL_VOLUME, "material_volume")
PREDEF_SYM(sym_material_geometry, SYM_TYPE_MATERIAL_GEOMETRY, "material_geometry")
PREDEF_SYM(sym_material, SYM_TYPE_MATERIAL, "material")
PREDEF_SYM(sym_tex_gamma_mode, SYM_TYPE_TEX_GAMMA_MODE, "::tex::gamma_mode")
PREDEF_SYM(sym_intensity_mode, SYM_TYPE_INTENSITY_MODE, "intensity_mode")
PREDEF_SYM(sym_hair_bsdf, SYM_TYPE_HAIR_BSDF, "hair_bsdf")

// enum values
PREDEF_SYM(sym_enum_gamma_default, SYM_ENUM_GAMMA_DEFAULT, "gamma_default")
PREDEF_SYM(sym_enum_gamma_linear, SYM_ENUM_GAMMA_LINEAR, "gamma_linear")
PREDEF_SYM(sym_enum_gamma_srgb, SYM_ENUM_GAMMA_SRGB, "gamma_srgb")
PREDEF_SYM(sym_enum_intensity_radiant_exitance,
           SYM_ENUM_INTENSITY_RADIANT_EXITANCE, "intensity_radiant_exitance")
PREDEF_SYM(sym_enum_intensity_power, SYM_ENUM_INTENSITY_POWER, "intensity_power")

// field names
PREDEF_SYM(sym_field_absorption_coefficient, 
    SYM_FIELD_ABSORPTION_COEFFICIENT, "absorption_coefficient")
PREDEF_SYM(sym_field_backface, SYM_FIELD_BACKFACE, "backface")
PREDEF_SYM(sym_field_cutout_opacity, SYM_FIELD_CUTOUT_OPACITY, "cutout_opacity")
PREDEF_SYM(sym_field_displacement, SYM_FIELD_DISPLACEMENT, "displacement")
PREDEF_SYM(sym_field_emission, SYM_FIELD_EMISSION, "emission")
PREDEF_SYM(sym_field_geometry, SYM_FIELD_GEOMETRY, "geometry")
PREDEF_SYM(sym_field_intensity, SYM_FIELD_INTENSITY, "intensity")
PREDEF_SYM(sym_field_ior, SYM_FIELD_IOR, "ior")
PREDEF_SYM(sym_field_normal, SYM_FIELD_NORMAL, "normal")
PREDEF_SYM(sym_field_rounded_edges_across_materials,
    SYM_FIELD_ROUNDED_EDGES_ACROSS_MATERIALS, "rounded_edges_across_materials")
PREDEF_SYM(sym_field_rounded_edges_radius, SYM_FIELD_ROUNDED_EDGES_RADIUS, "rounded_edges_radius")
PREDEF_SYM(sym_field_scattering, SYM_FIELD_SCATTERING, "scattering")
PREDEF_SYM(sym_field_scaterring_coefficient,
    SYM_FIELD_SCATTERING_COEFFICIENT, "scattering_coefficient")
PREDEF_SYM(sym_field_surface, SYM_FIELD_SURFACE, "surface")
PREDEF_SYM(sym_field_thin_walled, SYM_FIELD_THIN_WALLED, "thin_walled")
PREDEF_SYM(sym_field_volume, SYM_FIELD_VOLUME, "volume")
PREDEF_SYM(sym_field_mode, SYM_FIELD_MODE, "mode")
PREDEF_SYM(sym_field_hair, SYM_FIELD_HAIR, "hair")

// constants
PREDEF_SYM(sym_cnst_true, SYM_CNST_TRUE, "true")
PREDEF_SYM(sym_cnst_false, SYM_CNST_FALSE, "false")
PREDEF_SYM(sym_cnst_state, SYM_CNST_STATE, "state")
PREDEF_SYM(sym_cnst_tex, SYM_CNST_TEX, "tex")

// parameters of constructors
PREDEF_SYM(sym_param_x, SYM_PARAM_X, "x")
PREDEF_SYM(sym_param_y, SYM_PARAM_Y, "y")
PREDEF_SYM(sym_param_z, SYM_PARAM_Z, "z")
PREDEF_SYM(sym_param_w, SYM_PARAM_W, "w")
PREDEF_SYM(sym_param_r, SYM_PARAM_R, "r")
PREDEF_SYM(sym_param_g, SYM_PARAM_G, "g")
PREDEF_SYM(sym_param_b, SYM_PARAM_B, "b")
PREDEF_SYM(sym_param_value, SYM_PARAM_VALUE, "value")
PREDEF_SYM(sym_param_rgb, SYM_PARAM_RGB, "rgb")
PREDEF_SYM(sym_param_name, SYM_PARAM_NAME, "name")
PREDEF_SYM(sym_param_wavelengths, SYM_PARAM_WAVELENGTHS, "wavelengths")
PREDEF_SYM(sym_param_amplitudes, SYM_PARAM_AMPLITUDES, "amplitudes")
PREDEF_SYM(sym_param_m00, SYM_PARAM_M00, "m00")
PREDEF_SYM(sym_param_m01, SYM_PARAM_M01, "m01")
PREDEF_SYM(sym_param_m02, SYM_PARAM_M02, "m02")
PREDEF_SYM(sym_param_m03, SYM_PARAM_M03, "m03")
PREDEF_SYM(sym_param_m10, SYM_PARAM_M10, "m10")
PREDEF_SYM(sym_param_m11, SYM_PARAM_M11, "m11")
PREDEF_SYM(sym_param_m12, SYM_PARAM_M12, "m12")
PREDEF_SYM(sym_param_m13, SYM_PARAM_M13, "m13")
PREDEF_SYM(sym_param_m20, SYM_PARAM_M20, "m20")
PREDEF_SYM(sym_param_m21, SYM_PARAM_M21, "m21")
PREDEF_SYM(sym_param_m22, SYM_PARAM_M22, "m22")
PREDEF_SYM(sym_param_m23, SYM_PARAM_M23, "m23")
PREDEF_SYM(sym_param_m30, SYM_PARAM_M30, "m30")
PREDEF_SYM(sym_param_m31, SYM_PARAM_M31, "m31")
PREDEF_SYM(sym_param_m32, SYM_PARAM_M32, "m32")
PREDEF_SYM(sym_param_m33, SYM_PARAM_M33, "m33")
PREDEF_SYM(sym_param_col0, SYM_PARAM_COL0, "col0")
PREDEF_SYM(sym_param_col1, SYM_PARAM_COL1, "col1")
PREDEF_SYM(sym_param_col2, SYM_PARAM_COL2, "col2")
PREDEF_SYM(sym_param_col3, SYM_PARAM_COL3, "col3")
PREDEF_SYM(sym_param_gamma_mode, SYM_PARAM_GAMMA, "gamma")

PREDEF_SYM(sym_param_big_n, SYM_PARAM_BIG_N, "N")

#undef PREDEF_SYM
#undef OPERATOR_SYM_VARIADIC
#undef OPERATOR_SYM_TERNARY
#undef OPERATOR_SYM_BINARY
#undef OPERATOR_SYM_UNARY
#undef OPERATOR_SYM
