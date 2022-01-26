/***************************************************************************************************
 * Copyright (c) 2008-2022, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Abstract interface for canvases

#ifndef MI_NEURAYLIB_ICANVAS_H
#define MI_NEURAYLIB_ICANVAS_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

class ITile;

/** \if IRAY_API \addtogroup mi_neuray_rendering
    \elseif DICE_API \addtogroup mi_neuray_rtmp
    \else \addtogroup mi_neuray_mdl_sdk_misc
    \endif
@{
*/

/// Abstract interface for a canvas (base class).
///
/// This interface is the base class for all canvases. It holds the common functionality of
/// all different canvas interfaces, i.e., the resolution of the canvas.
///
/// \see #mi::neuraylib::ICanvas
class ICanvas_base : public
    mi::base::Interface_declare<0x649fc7bd,0xc021,0x4aff,0x9e,0xa4,0x5b,0xab,0x18,0xb9,0x25,0x59>
{
public:
    /// Returns the resolution of the canvas in x direction.
    virtual Uint32 get_resolution_x() const = 0;

    /// Returns the resolution of the canvas in y direction.
    virtual Uint32 get_resolution_y() const = 0;

    /// Returns the pixel type used by the canvas.
    ///
    /// \see \ref mi_neuray_types for a list of supported pixel types
    virtual const char* get_type() const = 0;

    /// Returns the number of layers this canvas has.
    virtual Uint32 get_layers_size() const = 0;

    /// Returns the gamma value.
    ///
    /// The gamma value should be a positive number. Typical values are 2.2 for LDR pixel types, and
    /// 1.0 for HDR pixel types.
    virtual Float32 get_gamma() const = 0;

    /// Sets the gamma value.
    ///
    /// \note This method just sets the gamma value. It does \em not change the pixel data itself.
    virtual void set_gamma( Float32 gamma) = 0;

};

/// Abstract interface for a canvas represented by a rectangular array of tiles.
///
/// A canvas represents a two- or three-dimensional array of pixels. The size of this array is given
/// by #mi::neuraylib::ICanvas_base::get_resolution_x() and
/// #mi::neuraylib::ICanvas_base::get_resolution_y().
///
/// Optionally, there can be multiple layers. The number of these layers is
/// given by #mi::neuraylib::ICanvas_base::get_layers_size(). The format a layer, i.e., the type of
/// each pixel in that layer, is described by #mi::neuraylib::ICanvas_base::get_type().
///
/// \if IRAY_API
/// The #mi::neuraylib::IRender_target, #mi::neuraylib::ICanvas, and #mi::neuraylib::ITile classes
/// are abstract interfaces which can to be implemented by the application. For example, this gives
/// the application the ability to tailor the rendering process very specific to its needs. The     
/// render target has to be implemented by the application whereas default implementations for
/// canvases and tiles are available from #mi::neuraylib::IImage_api.
/// \endif
///
/// \see #mi::neuraylib::ICanvas_base
class ICanvas : public
    mi::base::Interface_declare<0x20e5d5de,0x1f61,0x441c,0x88,0x88,0xff,0x85,0x89,0x98,0x7a,0xfa,
                                neuraylib::ICanvas_base>
{
public:

    /// Returns the tile for the given layer.
    ///
    /// \param layer     The layer of the pixel in the canvas.
    /// \return          The tile that contains the pixel, or \c NULL in case of invalid
    ///                  parameters.
    virtual const ITile* get_tile( Uint32 layer = 0) const = 0;

    /// Returns the tile for the given layer.
    ///
    /// \param layer     The layer of the pixel in the canvas.
    /// \return          The tile that contains the pixel, or \c NULL in case of invalid
    ///                  parameters.
    virtual ITile* get_tile( Uint32 layer = 0) = 0;
};

/*@}*/ // end group mi_neuray_rendering

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ICANVAS_H
