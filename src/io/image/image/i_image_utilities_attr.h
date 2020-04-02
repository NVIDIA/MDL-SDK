/***************************************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_IMAGE_IMAGE_I_IMAGE_UTILITIES_ATTR_H
#define IO_IMAGE_IMAGE_I_IMAGE_UTILITIES_ATTR_H

#include "i_image_utilities.h"

#include <base/data/attr/i_attr_types.h>

namespace MI {

namespace IMAGE {

/// Converts a pixel type into an ATTR type code.
inline ATTR::Type_code convert_pixel_type_to_type_code( Pixel_type pixel_type)
{
    switch( pixel_type) {
        case PT_UNDEF:     return ATTR::TYPE_UNDEF;
        case PT_SINT8:     return ATTR::TYPE_INT8;
        case PT_SINT32:    return ATTR::TYPE_INT32;
        case PT_FLOAT32:   return ATTR::TYPE_SCALAR;
        case PT_FLOAT32_2: return ATTR::TYPE_VECTOR2;
        case PT_FLOAT32_3: return ATTR::TYPE_VECTOR3;
        case PT_FLOAT32_4: return ATTR::TYPE_VECTOR4;
        case PT_RGB:       return ATTR::TYPE_RGB;
        case PT_RGBA:      return ATTR::TYPE_RGBA;
        case PT_RGBE:      return ATTR::TYPE_RGBE;
        case PT_RGBEA:     return ATTR::TYPE_RGBEA;
        case PT_RGB_16:    return ATTR::TYPE_RGB_16;
        case PT_RGBA_16:   return ATTR::TYPE_RGBA_16;
        case PT_RGB_FP:    return ATTR::TYPE_RGB_FP;
        case PT_COLOR:     return ATTR::TYPE_COLOR;
        default:           return ATTR::TYPE_UNDEF;
    }
}

/// Converts an ATTR type code into a pixel type.
///
/// Returns PT_UNDEF for type codes that are not a valid pixel type.
inline Pixel_type convert_type_code_to_pixel_type( ATTR::Type_code type_code)
{
    switch( type_code) {
        case ATTR::TYPE_UNDEF:   return PT_UNDEF;
        case ATTR::TYPE_INT8:    return PT_SINT8;
        case ATTR::TYPE_INT32:   return PT_SINT32;
        case ATTR::TYPE_SCALAR:  return PT_FLOAT32;
        case ATTR::TYPE_VECTOR2: return PT_FLOAT32_2;
        case ATTR::TYPE_VECTOR3: return PT_FLOAT32_3;
        case ATTR::TYPE_VECTOR4: return PT_FLOAT32_4;
        case ATTR::TYPE_RGB:     return PT_RGB;
        case ATTR::TYPE_RGBA:    return PT_RGBA;
        case ATTR::TYPE_RGBE:    return PT_RGBE;
        case ATTR::TYPE_RGBEA:   return PT_RGBEA;
        case ATTR::TYPE_RGB_16:  return PT_RGB_16;
        case ATTR::TYPE_RGBA_16: return PT_RGBA_16;
        case ATTR::TYPE_RGB_FP:  return PT_RGB_FP;
        case ATTR::TYPE_COLOR:   return PT_COLOR;
        default:                 return PT_UNDEF;
    }
}

} // namespace IMAGE

} // namespace MI

#endif // IO_IMAGE_IMAGE_I_IMAGE_UTILITIES_ATTR_H
