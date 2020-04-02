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

#ifndef BUILTIN_TYPE
#define BUILTIN_TYPE(type, name, args)
#endif

#ifndef NO_FLOAT_TYPE
// The float type is special, because it is needed to create the color type.
BUILTIN_TYPE(Type_float,         the_float_type, )
#endif

BUILTIN_TYPE(Type_error,         the_error_type, )
BUILTIN_TYPE(Type_incomplete,    the_incomplete_type, )
BUILTIN_TYPE(Type_bool,          the_bool_type, )
BUILTIN_TYPE(Type_int,           the_int_type, )
BUILTIN_TYPE(Type_double,        the_double_type, )
BUILTIN_TYPE(Type_string,        the_string_type, )
BUILTIN_TYPE(Type_light_profile, the_light_profile_type, )
BUILTIN_TYPE(Type_color,         the_color_type, )
BUILTIN_TYPE(Type_vector,        the_bool2_type, (&the_bool_type, 2))
BUILTIN_TYPE(Type_vector,        the_bool3_type, (&the_bool_type, 3))
BUILTIN_TYPE(Type_vector,        the_bool4_type, (&the_bool_type, 4))
BUILTIN_TYPE(Type_vector,        the_int2_type, (&the_int_type, 2))
BUILTIN_TYPE(Type_vector,        the_int3_type, (&the_int_type, 3))
BUILTIN_TYPE(Type_vector,        the_int4_type, (&the_int_type, 4))
BUILTIN_TYPE(Type_vector,        the_float2_type, (&the_float_type, 2))
BUILTIN_TYPE(Type_vector,        the_float3_type, (&the_float_type, 3))
BUILTIN_TYPE(Type_vector,        the_float4_type, (&the_float_type, 4))
BUILTIN_TYPE(Type_vector,        the_double2_type, (&the_double_type, 2))
BUILTIN_TYPE(Type_vector,        the_double3_type, (&the_double_type, 3))
BUILTIN_TYPE(Type_vector,        the_double4_type, (&the_double_type, 4))
BUILTIN_TYPE(Type_matrix,        the_float2x2_type, (&the_float2_type, 2))
BUILTIN_TYPE(Type_matrix,        the_float2x3_type, (&the_float2_type, 3))
BUILTIN_TYPE(Type_matrix,        the_float2x4_type, (&the_float2_type, 4))
BUILTIN_TYPE(Type_matrix,        the_float3x2_type, (&the_float3_type, 2))
BUILTIN_TYPE(Type_matrix,        the_float3x3_type, (&the_float3_type, 3))
BUILTIN_TYPE(Type_matrix,        the_float3x4_type, (&the_float3_type, 4))
BUILTIN_TYPE(Type_matrix,        the_float4x2_type, (&the_float4_type, 2))
BUILTIN_TYPE(Type_matrix,        the_float4x3_type, (&the_float4_type, 3))
BUILTIN_TYPE(Type_matrix,        the_float4x4_type, (&the_float4_type, 4))
BUILTIN_TYPE(Type_matrix,        the_double2x2_type, (&the_double2_type, 2))
BUILTIN_TYPE(Type_matrix,        the_double2x3_type, (&the_double2_type, 3))
BUILTIN_TYPE(Type_matrix,        the_double2x4_type, (&the_double2_type, 4))
BUILTIN_TYPE(Type_matrix,        the_double3x2_type, (&the_double3_type, 2))
BUILTIN_TYPE(Type_matrix,        the_double3x3_type, (&the_double3_type, 3))
BUILTIN_TYPE(Type_matrix,        the_double3x4_type, (&the_double3_type, 4))
BUILTIN_TYPE(Type_matrix,        the_double4x2_type, (&the_double4_type, 2))
BUILTIN_TYPE(Type_matrix,        the_double4x3_type, (&the_double4_type, 3))
BUILTIN_TYPE(Type_matrix,        the_double4x4_type, (&the_double4_type, 4))
BUILTIN_TYPE(Type_bsdf,          the_bsdf_type, )
BUILTIN_TYPE(Type_hair_bsdf,     the_hair_bsdf_type, )
BUILTIN_TYPE(Type_edf,           the_edf_type, )
BUILTIN_TYPE(Type_vdf,           the_vdf_type, )
BUILTIN_TYPE(Type_texture,       the_texture_2d_type, (IType_texture::TS_2D,     &the_float2_type))
BUILTIN_TYPE(Type_texture,       the_texture_3d_type, (IType_texture::TS_3D,     &the_float3_type))
BUILTIN_TYPE(Type_texture,       the_texture_cube_type, (IType_texture::TS_CUBE, &the_float3_type))
BUILTIN_TYPE(Type_texture,       the_texture_ptex_type, (IType_texture::TS_PTEX, NULL))
BUILTIN_TYPE(Type_texture,       the_texture_bsdf_data_type,
    (IType_texture::TS_BSDF_DATA, &the_float3_type))
BUILTIN_TYPE(Type_bsdf_measurement, the_bsdf_measurement_type, )

#undef BUILTIN_TYPE
#undef NO_FLOAT_TYPE
