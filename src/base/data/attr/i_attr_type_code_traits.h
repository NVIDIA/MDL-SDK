/******************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief

#ifndef BASE_DATA_ATTR_I_ATTR_TYPE_CODE_TRAITS_H
#define BASE_DATA_ATTR_I_ATTR_TYPE_CODE_TRAITS_H

#include "i_attr_types.h"
#include <base/system/main/types.h> // for Size

// forward declarations
namespace mi {
namespace math {
    class Color;
    template <typename T, MI::Size DIM> class Vector;
    template <typename T, MI::Size ROW, MI::Size COL> class Matrix;
} // namespace math
} // namespace mi

namespace MI {
namespace DB { class Tag; }

namespace ATTR {

/// mapping from basic types to their respective type codes
template <typename T> struct Type_code_traits;

template <> struct Type_code_traits<bool>
{ static const Type_code type_code = TYPE_BOOLEAN; };
template <> struct Type_code_traits<Uint8>
{ static const Type_code type_code = TYPE_INT8; };
template <> struct Type_code_traits<Sint8>
{ static const Type_code type_code = TYPE_INT8; };
template <> struct Type_code_traits<Uint16>
{ static const Type_code type_code = TYPE_INT16; };
template <> struct Type_code_traits<Sint16>
{ static const Type_code type_code = TYPE_INT16; };
template <> struct Type_code_traits<Uint32>
{ static const Type_code type_code = TYPE_INT32; };
template <> struct Type_code_traits<Sint32>
{ static const Type_code type_code = TYPE_INT32; };
template <> struct Type_code_traits<Uint64>
{ static const Type_code type_code = TYPE_INT64; };
template <> struct Type_code_traits<Sint64>
{ static const Type_code type_code = TYPE_INT64; };
template <> struct Type_code_traits<Scalar>
{ static const Type_code type_code = TYPE_SCALAR; };
template <> struct Type_code_traits<Dscalar>
{ static const Type_code type_code = TYPE_DSCALAR; };
template <> struct Type_code_traits<mi::math::Vector<Scalar, 2> >
{ static const Type_code type_code = TYPE_VECTOR2; };
template <> struct Type_code_traits<mi::math::Vector<Scalar, 3> >
{ static const Type_code type_code = TYPE_VECTOR3; };
template <> struct Type_code_traits<mi::math::Vector<Scalar, 4> >
{ static const Type_code type_code = TYPE_VECTOR4; };
template <> struct Type_code_traits<mi::math::Vector<Dscalar, 2> >
{ static const Type_code type_code = TYPE_DVECTOR2; };
template <> struct Type_code_traits<mi::math::Vector<Dscalar, 3> >
{ static const Type_code type_code = TYPE_DVECTOR3; };
template <> struct Type_code_traits<mi::math::Vector<Dscalar, 4> >
{ static const Type_code type_code = TYPE_DVECTOR4; };
template <> struct Type_code_traits<mi::math::Vector<Sint32, 2> >
{ static const Type_code type_code = TYPE_VECTOR2I; };
template <> struct Type_code_traits<mi::math::Vector<Sint32, 3> >
{ static const Type_code type_code = TYPE_VECTOR3I; };
template <> struct Type_code_traits<mi::math::Vector<Sint32, 4> >
{ static const Type_code type_code = TYPE_VECTOR4I; };
template <> struct Type_code_traits<mi::math::Vector<bool, 2> >
{ static const Type_code type_code = TYPE_VECTOR2B; };
template <> struct Type_code_traits<mi::math::Vector<bool, 3> >
{ static const Type_code type_code = TYPE_VECTOR3B; };
template <> struct Type_code_traits<mi::math::Vector<bool, 4> >
{ static const Type_code type_code = TYPE_VECTOR4B; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 2, 2> >
{ static const Type_code type_code = TYPE_MATRIX2X2; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 2, 3> >
{ static const Type_code type_code = TYPE_MATRIX2X3; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 3, 2> >
{ static const Type_code type_code = TYPE_MATRIX3X2; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 3, 3> >
{ static const Type_code type_code = TYPE_MATRIX3X3; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 4, 3> >
{ static const Type_code type_code = TYPE_MATRIX4X3; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 3, 4> >
{ static const Type_code type_code = TYPE_MATRIX3X4; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 4, 4> >
{ static const Type_code type_code = TYPE_MATRIX4X4; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 4, 2> >
{ static const Type_code type_code = TYPE_MATRIX4X2; };
template <> struct Type_code_traits<mi::math::Matrix<Scalar, 2, 4> >
{ static const Type_code type_code = TYPE_MATRIX2X4; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 4, 4> >
{ static const Type_code type_code = TYPE_DMATRIX; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 2, 2> >
{ static const Type_code type_code = TYPE_DMATRIX2X2; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 2, 3> >
{ static const Type_code type_code = TYPE_DMATRIX2X3; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 3, 2> >
{ static const Type_code type_code = TYPE_DMATRIX3X2; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 3, 3> >
{ static const Type_code type_code = TYPE_DMATRIX3X3; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 4, 3> >
{ static const Type_code type_code = TYPE_DMATRIX4X3; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 3, 4> >
{ static const Type_code type_code = TYPE_DMATRIX3X4; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 4, 2> >
{ static const Type_code type_code = TYPE_DMATRIX4X2; };
template <> struct Type_code_traits<mi::math::Matrix<Dscalar, 2, 4> >
{ static const Type_code type_code = TYPE_DMATRIX2X4; };
template <> struct Type_code_traits<const char*>
{ static const Type_code type_code = TYPE_STRING; };
template <> struct Type_code_traits<DB::Tag>
{ static const Type_code type_code = TYPE_TAG; };
template <> struct Type_code_traits<mi::math::Color>
{ static const Type_code type_code = TYPE_COLOR; };

} // namespace ATTR
} // namespace MI

#endif // BASE_DATA_ATTR_I_ATTR_TYPE_CODE_TRAITS_H

