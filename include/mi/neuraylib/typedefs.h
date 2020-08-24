/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief Typedefs for types from the math API

#ifndef MI_NEURAYLIB_TYPEDEFS_H
#define MI_NEURAYLIB_TYPEDEFS_H

#include <mi/neuraylib/matrix_typedefs.h>
#include <mi/neuraylib/vector_typedefs.h>

#include <mi/math/bbox.h>
#include <mi/math/color.h>
#include <mi/math/spectrum.h>

namespace mi {

/** \addtogroup mi_neuray_compounds
@{
*/


/// RGBA color class.
///
/// \see #mi::Color_struct for the corresponding POD type and
///      #mi::math::Color for the source of the typedef
typedef math::Color Color;

/// RGBA color class (underlying POD type).
///
/// \see #mi::Color for the corresponding non-POD type and
///      #mi::math::Color_struct for the source of the typedef
typedef math::Color_struct Color_struct;

using mi::math::Clip_mode;
using mi::math::CLIP_RGB;
using mi::math::CLIP_ALPHA;
using mi::math::CLIP_RAW;



/// Spectrum class.
///
/// \see #mi::Spectrum_struct for the corresponding POD type and
///      #mi::math::Spectrum for the source of the typedef
typedef math::Spectrum Spectrum;

/// Spectrum class (underlying POD type).
///
/// \see #mi::Spectrum for the corresponding non-POD type and
///      #mi::math::Spectrum_struct for the source of the typedef
typedef math::Spectrum_struct Spectrum_struct;



/// Three-dimensional bounding box.
///
/// \see #mi::Bbox3_struct for the corresponding POD type and
///      #mi::math::Bbox for the underlying template class
typedef math::Bbox<Float32,3> Bbox3;

/// Three-dimensional bounding box (underlying POD type).
///
/// \see #mi::Bbox3 for the corresponding non-POD type and
///      #mi::math::Bbox_struct for the underlying template class
typedef math::Bbox_struct<Float32,3> Bbox3_struct;



/*@}*/ // end group mi_neuray_compounds

} // namespace mi

#endif // MI_NEURAYLIB_TYPEDEFS_H
