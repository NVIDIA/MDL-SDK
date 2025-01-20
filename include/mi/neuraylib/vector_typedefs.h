/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Typedefs for types from the math API

#ifndef MI_NEURAYLIB_VECTOR_TYPEDEFS_H
#define MI_NEURAYLIB_VECTOR_TYPEDEFS_H

#include <mi/math/vector.h>

namespace mi {

/** \addtogroup mi_neuray_compounds
@{
*/

/// Vector of two bool.
///
/// \see #mi::Boolean_2_struct for the corresponding POD type and
///      #mi::math::Vector for the underlying template class
using Boolean_2 = math::Vector<bool,2>;

/// Vector of three bool.
///
/// \see #mi::Boolean_3_struct for the corresponding POD type and
///      #mi::math::Vector for the underlying template class
using Boolean_3 = math::Vector<bool,3>;

/// Vector of four bool.
///
/// \see #mi::Boolean_4_struct for the corresponding POD type and
///      #mi::math::Vector for the underlying template class
using Boolean_4 = math::Vector<bool,4>;

/// Vector of two %Sint32.
///
/// \see #mi::Sint32_2_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Sint32 for the type of the vector components
using Sint32_2 = math::Vector<Sint32,2>;

/// Vector of three %Sint32.
///
/// \see #mi::Sint32_3_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Sint32 for the type of the vector components
using Sint32_3 = math::Vector<Sint32,3>;

/// Vector of four %Sint32.
///
/// \see #mi::Sint32_4_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Sint32 for the type of the vector components
using Sint32_4 = math::Vector<Sint32,4>;

/// Vector of two %Uint32.
///
/// \see #mi::Uint32_2_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Uint32 for the type of the vector components
using Uint32_2 = math::Vector<Uint32,2>;

/// Vector of three %Uint32.
///
/// \see #mi::Uint32_3_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Uint32 for the type of the vector components
using Uint32_3 = math::Vector<Uint32,3>;

/// Vector of four %Uint32.
///
/// \see #mi::Uint32_4_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Uint32 for the type of the vector components
using Uint32_4 = math::Vector<Uint32,4>;

/// Vector of two %Float32.
///
/// \see #mi::Float32_2_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Float32 for the type of the vector components
using Float32_2 = math::Vector<Float32,2>;

/// Vector of three %Float32.
///
/// \see #mi::Float32_3_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Float32 for the type of the vector components
using Float32_3 = math::Vector<Float32,3>;

/// Vector of four %Float32.
///
/// \see #mi::Float32_4_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Float32 for the type of the vector components
using Float32_4 = math::Vector<Float32,4>;

/// Vector of two %Float64.
///
/// \see #mi::Float64_2_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Float64 for the type of the vector components
using Float64_2 = math::Vector<Float64,2>;

/// Vector of three %Float64.
///
/// \see #mi::Float64_3_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Float64 for the type of the vector components
using Float64_3 = math::Vector<Float64,3>;

/// Vector of four %Float64.
///
/// \see #mi::Float64_4_struct for the corresponding POD type,
///      #mi::math::Vector for the underlying template class, and
///      #mi::Float64 for the type of the vector components
using Float64_4 = math::Vector<Float64,4>;



/// Vector of two bool (underlying POD type).
///
/// \see #mi::Boolean_2 for the corresponding non-POD type and
///      #mi::math::Vector_struct for the underlying template class
using Boolean_2_struct = math::Vector_struct<bool,2>;

/// Vector of three bool (underlying POD type).
///
/// \see #mi::Boolean_3 for the corresponding non-POD type and
///      #mi::math::Vector_struct for the underlying template class
using Boolean_3_struct = math::Vector_struct<bool,3>;

/// Vector of four bool (underlying POD type).
///
/// \see #mi::Boolean_4 for the corresponding non-POD type and
///      #mi::math::Vector_struct for the underlying template class
using Boolean_4_struct = math::Vector_struct<bool,4>;

/// Vector of two %Sint32 (underlying POD type).
///
/// \see #mi::Sint32_2 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Sint32 for the type of the vector components
using Sint32_2_struct = math::Vector_struct<Sint32,2>;

/// Vector of three %Sint32 (underlying POD type).
///
/// \see #mi::Sint32_3 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Sint32 for the type of the vector components
using Sint32_3_struct = math::Vector_struct<Sint32,3>;

/// Vector of four %Sint32 (underlying POD type).
///
/// \see #mi::Sint32_4 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Sint32 for the type of the vector components
using Sint32_4_struct = math::Vector_struct<Sint32,4>;

/// Vector of two %Uint32 (underlying POD type).
///
/// \see #mi::Uint32_2 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Uint32 for the type of the vector components
using Uint32_2_struct = math::Vector_struct<Uint32,2>;

/// Vector of three %Uint32 (underlying POD type).
///
/// \see #mi::Uint32_3 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Uint32 for the type of the vector components
using Uint32_3_struct = math::Vector_struct<Uint32,3>;

/// Vector of four %Uint32 (underlying POD type).
///
/// \see #mi::Uint32_4 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Uint32 for the type of the vector components
using Uint32_4_struct = math::Vector_struct<Uint32,4>;

/// Vector of two %Float32 (underlying POD type).
///
/// \see #mi::Float32_2 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Float32 for the type of the vector components
using Float32_2_struct = math::Vector_struct<Float32,2>;

/// Vector of three %Float32 (underlying POD type).
///
/// \see #mi::Float32_3 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Float32 for the type of the vector components
using Float32_3_struct = math::Vector_struct<Float32,3>;

/// Vector of four %Float32 (underlying POD type).
///
/// \see #mi::Float32_4 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Float32 for the type of the vector components
using Float32_4_struct = math::Vector_struct<Float32,4>;

/// Vector of two %Float64 (underlying POD type).
///
/// \see #mi::Float64_2 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Float64 for the type of the vector components
using Float64_2_struct = math::Vector_struct<Float64,2>;

/// Vector of three %Float64 (underlying POD type).
///
/// \see #mi::Float64_3 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Float64 for the type of the vector components
using Float64_3_struct = math::Vector_struct<Float64,3>;

/// Vector of four %Float64 (underlying POD type).
///
/// \see #mi::Float64_4 for the corresponding non-POD type,
///      #mi::math::Vector_struct for the underlying template class, and
///      #mi::Float64 for the type of the vector components
using Float64_4_struct = math::Vector_struct<Float64,4>;


/**@}*/ // end group mi_neuray_compounds

} // namespace mi

#endif // MI_NEURAYLIB_VECTOR_TYPEDEFS_H
