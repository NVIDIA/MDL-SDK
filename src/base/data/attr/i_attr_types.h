/***************************************************************************************************
 * Copyright (c) 2006-2026, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Attribute types

#ifndef BASE_DATA_ATTR_I_ATTR_TYPES_H
#define BASE_DATA_ATTR_I_ATTR_TYPES_H

#include <base/system/main/types.h>

namespace MI {
namespace ATTR {

/// Identifies attribute type. Easier and faster to use than symbolic name.
using Attribute_id = mi::Uint32;

/// For dynamic arrays, a value described by the Type tree contains a reference
/// to a separately allocated array. Note the alignment on 64-bit hosts.
struct Dynamic_array
{
    unsigned int m_count; ///< Number of array elements
    char* m_value;        ///< Buffer of array elements
};

/// Propagation specifier.
///
/// Attribute inheritance propagates attributes and their values. That propagation can be performed
/// in several ways - this enumerationspecifies in which.
enum Attribute_propagation {
    PROPAGATION_STANDARD,               ///< every new attribute overrides the inherited
    PROPAGATION_OVERRIDE,               ///< the inherited attribute overrides everything
    PROPAGATION_UNDEF                   ///< undefined
};

/// Class IDs for serialization.
enum Scene_type {
    ID_ATTRIBUTE_SET    = 0x5f417453,   ///< '_AtS'
    ID_ATTRIBUTE        = 0x5f417472,   ///< '_Atr'
    ID_ATTRIBUTE_LIST   = 0x5f41744C,   ///< '_AtL'
    ID_TYPE             = 0x5f547970    ///< '_Typ'
};

/// Type codes
///
/// \see  m_typeinfo[] in attr_type.cpp
enum Type_code {
    TYPE_UNDEF,                    ///< illegal
    TYPE_BOOLEAN,                  ///< size 1, bool
    TYPE_INT8,                     ///< size 1, Uint8/Sint8
    TYPE_INT16,                    ///< size 2, Uint16/Sint16
    TYPE_INT32,                    ///< size 4, Uint32/Sint32
    TYPE_INT64,                    ///< size 8, Uint64/Sint64
    TYPE_SCALAR,                   ///< size 4, Scalar
    TYPE_VECTOR2,                  ///< size 8, Vector2
    TYPE_VECTOR3,                  ///< size 12, Vector3
    TYPE_VECTOR4,                  ///< size 16, Vector4
    TYPE_DSCALAR,                  ///< size 8, Dscalar
    TYPE_DVECTOR2,                 ///< size 16, Dvector2
    TYPE_DVECTOR3,                 ///< size 24, Dvector3
    TYPE_DVECTOR4,                 ///< size 32, Dvector4
    TYPE_MATRIX,                   ///< size 64, Matrix
    TYPE_DMATRIX,                  ///< size 128, Dmatrix
    TYPE_STRING,                   ///< size 8, char *
    TYPE_TAG,                      ///< size 4, Tag
    TYPE_COLOR,                    ///< size 16, Color
    TYPE_RGB_FP,                   ///< size 12, Scalar[3]
    TYPE_STRUCT,                   ///<
    TYPE_ARRAY,                    ///<
    TYPE_VECTOR2I,                 ///< size 8, Sint32[2]
    TYPE_VECTOR3I,                 ///< size 12, Sint32[3]
    TYPE_VECTOR4I,                 ///< size 16, Sint32[4]
    TYPE_VECTOR2B,                 ///< size 2, bool[2]
    TYPE_VECTOR3B,                 ///< size 3, bool[3]
    TYPE_VECTOR4B,                 ///< size 4, bool[4]
    TYPE_MATRIX2X2,                ///< size 16, Scalar[4]
    TYPE_MATRIX2X3,                ///< size 24, Scalar[6]
    TYPE_MATRIX3X2,                ///< size 24, Scalar[6]
    TYPE_MATRIX3X3,                ///< size 36, Scalar[9]
    TYPE_MATRIX4X3,                ///< size 48, Scalar[12]
    TYPE_MATRIX3X4,                ///< size 48, Scalar[12]
    TYPE_MATRIX4X2,                ///< size 32, Scalar[8]
    TYPE_MATRIX2X4,                ///< size 32, Scalar[8]
    TYPE_SPECTRUM,                 ///< size 12, Scalar[3]
    TYPE_ENUM,                     ///< size 4, Uint32
    TYPE_DMATRIX2X2,               ///< size 32, Dscalar[4]
    TYPE_DMATRIX2X3,               ///< size 48, Dscalar[6]
    TYPE_DMATRIX3X2,               ///< size 48, Dscalar[6]
    TYPE_DMATRIX3X3,               ///< size 72, Dscalar[9]
    TYPE_DMATRIX4X3,               ///< size 96, Dscalar[12]
    TYPE_DMATRIX3X4,               ///< size 96, Dscalar[12]
    TYPE_DMATRIX4X2,               ///< size 64, Dscalar[8]
    TYPE_DMATRIX2X4,               ///< size 64, Dscalar[8]

    TYPE_NUM,                      ///< number of types

    // Type aliases
    TYPE_MATRIX4X4 = TYPE_MATRIX,  ///< size 64, Scalar[16]
    TYPE_DMATRIX4X4 = TYPE_DMATRIX ///< size 128, Dscalar[16]
};

} // namespace ATTR

} // namespace MI

#endif // BASE_DATA_ATTR_I_ATTR_TYPES_H

