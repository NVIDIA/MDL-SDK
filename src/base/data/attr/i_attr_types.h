/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

/// \file
/// \brief attribute types
/// 
/// An attribute is an extra piece of data that can be attached to a database element. It is named
/// (although names are converted to perfect hashes early on, for faster lookup). All inheritance
/// is based on attributes. For example, a geometric object may have a boolean attribute "trace"
/// that controls whether the object is hit by trace rays; or an array attribute containing a
/// texture space (one vector per vertex); or a shader or instance may have attributes like
/// "ambient" that replace mental ray 3's parameters.

#ifndef BASE_DATA_ATTR_I_ATTR_TYPES_H
#define BASE_DATA_ATTR_I_ATTR_TYPES_H

#include <base/system/main/types.h>

namespace MI {
namespace ATTR {

/// Identifies attribute type. Easier and faster to use than symbolic name.
typedef Uint Attribute_id;

class Type;
/// Convenient typedef for comparison function used in following function.
typedef bool (*Compare)(const Type& typ);

/// For dynamic arrays, a value described by the Type tree contains a reference
/// to a separately allocated array. Note the alignment on 64-bit hosts.
struct Dynamic_array
{
    unsigned int m_count;				///< number of array elements
    char* m_value;					///< m_count array members
};


/// Propagation specifier. Attribute inheritance propagates attributes and their
/// values. That propagation can be performed in several ways - this enumeration
/// specifies in which.
enum Attribute_propagation {
    PROPAGATION_STANDARD,		///< every new attribute overrides the inherited
    PROPAGATION_OVERRIDE,		///< the inherited attribute overrides everything
    PROPAGATION_UNDEF			///< undefined
};


/// see io/scene/*/*.h for render-specific scene types.
enum Scene_type {
    ID_ATTRIBUTE_SET    = 0x5f417453,   ///< '_AtS'
    ID_ATTRIBUTE        = 0x5f417472,   ///< '_Atr'
    ID_ATTRIBUTE_LIST	= 0x5f41744C,	///< '_AtL'
    ID_TYPE             = 0x5f547970    ///< '_Typ'
};


///
/// Type codes, used in every Type (see below).
///
/// Note that type remappings (TYPE_A = TYPE_B) have to
/// be set at the end of the enum listing in order not to
/// have duplicate enum values.
///

enum Type_code {                  ///< see m_typeinfo[] in attr_type.cpp
    TYPE_UNDEF,                   ///< = 0, illegal
    TYPE_BOOLEAN,                 ///< = 1, size 1, bool
    TYPE_INT8,                    ///< = 2, size 1, Uint8/Sint8
    TYPE_INT16,                   ///< = 3, size 2, Uint16/Sint16
    TYPE_INT32,                   ///< = 4, size 4, Uint32/Sint32
    TYPE_INT64,                   ///< = 5, size 8, Uint64/Sint64
    TYPE_SCALAR,                  ///< = 6, size 4, Scalar
    TYPE_VECTOR2,                 ///< = 7, size 8, Vector2
    TYPE_VECTOR3,                 ///< = 8, size 12, Vector3
    TYPE_VECTOR4,                 ///< = 9, size 16, Vector4
    TYPE_DSCALAR,                 ///< = 10, size 8, Dscalar
    TYPE_DVECTOR2,                ///< = 11, size 16, Dvector2
    TYPE_DVECTOR3,                ///< = 12, size 24, Dvector3
    TYPE_DVECTOR4,                ///< = 13, size 32, Dvector4
    TYPE_MATRIX,                  ///< = 14, size 64, Matrix
    TYPE_DMATRIX,                 ///< = 15, size 128, Dmatrix
    TYPE_QUATERNION,              ///< = 16, size 16, Quaternion
    TYPE_STRING,                  ///< = 17, size 4/8, char *
    TYPE_TAG,                     ///< = 18, size 4, Tag
    TYPE_COLOR,                   ///< = 19, size 16, Color
    TYPE_RGB,                     ///< = 20, size 3, Uint8[3]
    TYPE_RGBA,                    ///< = 21, size 4, Uint8[4]
    TYPE_RGBE,                    ///< = 22, size 4, Uint8[4]
    TYPE_RGBEA,                   ///< = 23, size 5, Uint8[5]
    TYPE_RGB_16,                  ///< = 24, size 6, Uint16[3]
    TYPE_RGBA_16,                 ///< = 25, size 8, Uint16[4]
    TYPE_RGB_FP,                  ///< = 26, size 12, Scalar[3]
    TYPE_STRUCT,                  ///< = 27, size  0, struct{...}
    TYPE_ARRAY,                   ///< = 28
    TYPE_RLE_UINT_PTR,            ///< = 29, size  4/8, Rle_array<Uint> *
    TYPE_VECTOR2I,                ///< = 30, size 8, Sint32[2]
    TYPE_VECTOR3I,                ///< = 31, size 12, Sint32[3]
    TYPE_VECTOR4I,                ///< = 32, size 16, Sint32[4]
    TYPE_VECTOR2B,                ///< = 33, size 2, bool[2]
    TYPE_VECTOR3B,                ///< = 34, size 3, bool[3]
    TYPE_VECTOR4B,                ///< = 35, size 4, bool[4]
    TYPE_MATRIX2X2,               ///< = 36, size 16, Scalar[4]
    TYPE_MATRIX2X3,               ///< = 37, size 24, Scalar[6]
    TYPE_MATRIX3X2,               ///< = 38, size 24, Scalar[6]
    TYPE_MATRIX3X3,               ///< = 39, size 36, Scalar[9]
    TYPE_MATRIX4X3,               ///< = 40, size 48, Scalar[12]
    TYPE_MATRIX3X4,               ///< = 41, size 48, Scalar[12]
    TYPE_MATRIX4X2,               ///< = 42, size 32, Scalar[8]
    TYPE_MATRIX2X4,               ///< = 43, size 32, Scalar[8]
    // MetaSL-specific attribute types
    TYPE_TEXTURE1D,               ///< = 44, size 4, Tag, unused
    TYPE_TEXTURE2D,               ///< = 45, size 4, Tag, unused
    TYPE_TEXTURE3D,               ///< = 46, size 4, Tag, unused
    TYPE_TEXTURE_CUBE,            ///< = 47, size 4, Tag, unused
    TYPE_BRDF,                    ///< = 48, size 4, Tag, unused
    TYPE_LIGHT,                   ///< = 49, size 4, Tag, unused
    TYPE_LIGHTPROFILE,            ///< = 50, size 4, Tag, unused
    TYPE_SHADER,                  ///< = 51, size 4, Tag, unused
    TYPE_PARTICLE_MAP,            ///< = 52, size 4, Tag, unused
    TYPE_SPECTRUM,                ///< = 53, size 12, Scalar[3] (currently)
    // MDL-specific attribute types
    TYPE_ID,                      ///< = 54, size 4, Uint32/Sint32, unused
    TYPE_PARAMETER,               ///< = 55, size 4, Uint32/Sint32, unused
    TYPE_TEMPORARY,               ///< = 56, size 4, Uint32/Sint32, unused
    TYPE_ATTACHABLE,              ///< = 57, size 0, struct{...}, unused
    TYPE_CALL,                    ///< = 58, size 2*4/8, Tag + char *, but counted as 2 const char*, unused
    TYPE_TEXTURE,                 ///< = 59, size 4, Tag, unused
    TYPE_BSDF_MEASUREMENT,        ///< = 60, size 4, Tag, unused
    TYPE_ENUM,                    ///< = 61, size 4, Uint32
    // general attribute types
    TYPE_DMATRIX2X2,              ///< = 62, size 32, Dscalar[4]
    TYPE_DMATRIX2X3,              ///< = 63, size 48, Dscalar[6]
    TYPE_DMATRIX3X2,              ///< = 64, size 48, Dscalar[6]
    TYPE_DMATRIX3X3,              ///< = 65, size 72, Dscalar[9]
    TYPE_DMATRIX4X3,              ///< = 66, size 96, Dscalar[12]
    TYPE_DMATRIX3X4,              ///< = 67, size 96, Dscalar[12]
    TYPE_DMATRIX4X2,              ///< = 68, size 64, Dscalar[8]
    TYPE_DMATRIX2X4,              ///< = 69, size 64, Dscalar[8]

    TYPE_NUM,                     ///< = 70, number of types
    //
    // Type remappings. It is important to have this at the end
    // of the enum listing in order not to have duplicate enum values.
    //
    TYPE_MATRIX4X4 = TYPE_MATRIX,   ///< = 14, size 64, Scalar[16]
    TYPE_DMATRIX4X4 = TYPE_DMATRIX, ///< = 15, size 128, Dscalar[16]
    TYPE_RGBA_FP = TYPE_COLOR       ///< = 19, size 16, Scalar[4]
};


}
}

#endif
