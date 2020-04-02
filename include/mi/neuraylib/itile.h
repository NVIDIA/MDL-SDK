/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Abstract interface for tiles

#ifndef MI_NEURAYLIB_ITILE_H
#define MI_NEURAYLIB_ITILE_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

/** \if IRAY_API \addtogroup mi_neuray_rendering
    \elseif DICE_API \addtogroup mi_neuray_rtmp
    \else \addtogroup mi_neuray_misc
    \endif
@{
*/

/// Abstract interface for a tile.
///
/// A tile is a rectangular array of pixels of a certain pixel type.
///
/// \if IRAY_API
/// The #mi::neuraylib::IRender_target, #mi::neuraylib::ICanvas, and #mi::neuraylib::ITile classes
/// are abstract interfaces which can be implemented by the application. For example, this gives
/// the application the ability to tailor the rendering process very specific to its needs. The     
/// render target has to be implemented by the application whereas default implementations for
/// canvases and tiles are available from #mi::neuraylib::IImage_api.
/// \endif
class ITile : public
    mi::base::Interface_declare<0x0f0a0181,0x7640,0x4f60,0x9d,0xa7,0xb0,0xa0,0x09,0x17,0x1a,0xec>
{
public:
    /// Looks up a certain pixel at the given coordinates.
    ///
    /// The offsets are relative to the lower left border of the tile. The \p floats argument must
    /// point to at least 4 floats, e.g., you pass the address of #mi::math::Color.r.
    ///
    /// This method is a rather slow, but convenient access method. Typically, the #get_data()
    /// method is faster and can handle arbitrary data types without pixel type conversion.
    virtual void get_pixel(
        Uint32 x_offset,
        Uint32 y_offset,
        Float32* floats) const = 0;

    /// Stores a certain pixel at the given coordinates.
    ///
    /// The offsets are relative to the lower left border of the tile. The \p floats argument must
    /// point to at least 4 floats, e.g., you pass the address of #mi::math::Color.r.
    ///
    /// This method is a rather slow, but convenient access method. Typically, the #get_data()
    /// method is faster and can handle arbitrary data types without pixel type conversion.
    virtual void set_pixel(
        Uint32 x_offset,
        Uint32 y_offset,
        const Float32* floats) = 0;

    /// Returns the pixel type used by the tile.
    ///
    /// \see \ref mi_neuray_types for a list of supported pixel types
    virtual const char* get_type() const = 0;

    /// Returns the tile size in x direction
    virtual Uint32 get_resolution_x() const = 0;

    /// Returns the tile size in y direction
    virtual Uint32 get_resolution_y() const = 0;

    /// Returns a pointer to the raw tile data according to the pixel type of the tile.
    ///
    /// This methods is used for fast, direct read access to the raw tile data. It is expected that
    /// the data is stored in row-major layout without any padding. In case of #mi::Color, the
    /// components are expected to be stored in RGBA order.
    ///
    /// The total size of the buffer in bytes is \code x * y * bpp \endcode where \c x is the result
    /// of #get_resolution_x(), \c y is the result of #get_resolution_y(), and \c bpp is the number
    /// of bytes per pixel. \ifnot MDL_SDK_API The number of bytes per pixel is the product of
    /// #mi::neuraylib::IImage_api::get_components_per_pixel() and
    /// #mi::neuraylib::IImage_api::get_bytes_per_component() when passing the result of #get_type()
    /// as pixel type. \endif
    virtual const void* get_data() const = 0;

    /// Returns a pointer to the raw tile data according to the pixel type of the tile.
    ///
    /// This methods is used for fast, direct write access to the raw tile data. It is expected that
    /// the data is stored in row-major layout without any padding. In case of #mi::Color, the
    /// components are expected to be stored in RGBA order.
    ///
    /// The total size of the buffer in bytes is \code x * y * bpp \endcode where \c x is the result
    /// of #get_resolution_x(), \c y is the result of #get_resolution_y(), and \c bpp is the number
    /// of bytes per pixel. \ifnot MDL_SDK_API The number of bytes per pixel is the product of
    /// #mi::neuraylib::IImage_api::get_components_per_pixel() and
    /// #mi::neuraylib::IImage_api::get_bytes_per_component() when passing the result of #get_type()
    /// as pixel type. \endif
    virtual void* get_data() = 0;
};

/*@}*/ // end group mi_neuray_rendering

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ITILE_H
