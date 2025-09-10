/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief API component for various image-related functions.

#ifndef MI_NEURAYLIB_IIMAGE_API_H
#define MI_NEURAYLIB_IIMAGE_API_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/version.h> // for MI_NEURAYLIB_DEPRECATED_ENUM_VALUE

namespace mi {

class IArray;
class IMap;

namespace neuraylib {

class IBuffer;
class ICanvas;
class ICanvas_cuda;
class IReader;
class ITile;

/**
\if MDL_SDK_API
    \defgroup mi_neuray_mdl_sdk_misc Miscellaneous Interfaces
    \ingroup mi_neuray

    \brief Various utility classes.
\endif
*/

/** \if IRAY_API \addtogroup mi_neuray_rendering
    \elseif MDL_SDK_API \addtogroup mi_neuray_mdl_sdk_misc
    \elseif DICE_API \addtogroup mi_neuray_rtmp
    \endif
@{
*/


/// This interface provides various utilities related to canvases and buffers.
///
/// Note that #create_buffer_from_canvas() and #create_canvas_from_buffer() encode and decode pixel
/// data to/from memory buffers. \if IRAY_API To import images from disk use
/// #mi::neuraylib::IImport_api::import_canvas(). To export images to disk use
/// #mi::neuraylib::IExport_api::export_canvas(). \endif
///
/// \section mi_image_export_options Image export options
///
/// Various methods for image export support options to control details of the export process. The
/// following general options are currently supported:
///
/// - \c bool "force_default_gamma": If enabled, adjusts the gamma value of the exported pixel data
///   according to the pixel type chosen for export (1.0 for HDR pixel types, 2.2 for LDR pixel
///   types). Default: \c false.
///
/// The following format-specific options are currently supported:
///
/// - \c #mi::Uint32 "jpg:quality": The quality of JPG compression in the range from 0 to 100, where
///   0 is the lowest quality, and 100 is the highest. Default: 100.
/// - \c std::string "exr:data_type": Indicates the desired data type of the channels. Possible
///   values: \c "Float16" and \c "Float32". Default: \c "Float32".
/// - \c bool "exr:create_multipart_for_alpha": If enabled, and if the pixel type has an alpha
///   channel, creates an OpenEXR multipart image where the RGB and alpha channels are split into
///   two subimages named "rgb" and "alpha". The advantage is that with a separate subimage the
///   alpha channel can be kept unassociated without violating the OpenEXR specification.
///   \if MDL_SOURCE_RELEASE This option requires OpenImageIO >= 2.5.12. \endif Default: \c false.
class IImage_api : public
    mi::base::Interface_declare<0x4c25a4f0,0x2bac,0x4ce6,0xb0,0xab,0x4d,0x94,0xbf,0xfd,0x97,0xa5>
{
public:
    /// \name Factory methods for canvases and tiles
    //@{

    /// Creates a tile with given pixel type, width, and height.
    ///
    /// This factory function allows to create instances of the abstract interface
    /// #mi::neuraylib::ITile based on an internal default implementation. However, you are not
    /// obligated to use this factory function and the internal default implementation. It is
    /// absolutely fine to use your own (correct) implementation of the #mi::neuraylib::ITile
    /// interface.
    ///
    /// \param pixel_type   The desired pixel type. See \ref mi_neuray_types for a list of supported
    ///                     pixel types.
    /// \param width        The desired width.
    /// \param height       The desired height.
    /// \return             The requested tile, or \c nullptr in case of invalid pixel type, width,
    ///                     or height, or memory allocation failures.
    virtual ITile* create_tile(
        const char* pixel_type,
        Uint32 width,
        Uint32 height) const = 0;

    /// Creates a canvas with given pixel type, resolution, and layers.
    ///
    /// This factory function allows to create instances of the abstract interface
    /// #mi::neuraylib::ICanvas based on an internal default implementation. However, you are not
    /// obligated to use this factory function and the internal default implementation. It is
    /// absolutely fine to use your own (correct) implementation of the #mi::neuraylib::ICanvas
    /// interface.
    ///
    /// \param pixel_type   The desired pixel type. See \ref mi_neuray_types for a list of
    ///                     supported pixel types.
    /// \param width        The desired width.
    /// \param height       The desired height.
    /// \param layers       The desired number of layers (depth). Must be 6 for cubemaps.
    /// \param is_cubemap   Flag that indicates whether this canvas represents a cubemap.
    /// \param gamma        The desired gamma value. The special value 0.0 represents the default
    ///                     gamma which is 1.0 for HDR pixel types and 2.2 for LDR pixel types.
    /// \return             The requested canvas, or \c nullptr in case of invalid pixel type,
    ///                     width, height, layers, or cubemap flag, or memory allocation failures.
    virtual ICanvas* create_canvas(
        const char* pixel_type,
        Uint32 width,
        Uint32 height,
        Uint32 layers = 1,
        bool is_cubemap = false,
        Float32 gamma = 0.0f) const = 0;

#ifndef MI_SKIP_WITH_MDL_SDK_DOXYGEN
    /// Creates a CUDA canvas with given pixel type, width, height, and layers.
    ///
    /// \see #create_canvas()
    /// \see #mi::neuraylib::IGpu_description::get_cuda_device_id()
    ///
    /// \param cuda_device_id The CUDA ID of the device on which the canvas will reside.
    ///                     Note that this is the CUDA device ID, not the 1-based GPU index
    ///                     used in \if IRAY_API #mi::neuraylib::IRendering_configuration.
    ///                     \else #mi::neuraylib::IDice_configuration. \endif
    /// \param pixel_type   The desired pixel type. See \ref mi_neuray_types for a list of
    ///                     supported pixel types.
    /// \param width        The desired width.
    /// \param height       The desired height.
    /// \param layers       The desired number of layers.
    /// \param gamma        The desired gamma value. The special value 0.0 represents the default
    ///                     gamma which is 1.0 for HDR pixel types and 2.2 for LDR pixel types.
    /// \return             The requested canvas, or \c nullptr in case of invalid parameters or
    ///                     CUDA errors.
#else // MI_SKIP_WITH_MDL_SDK_DOXYGEN
    /// Unused.
    ///
    /// This method exists only for technical reasons (ABI compatibility). Calling it results in
    /// unspecified behavior.
#endif // MI_SKIP_WITH_MDL_SDK_DOXYGEN
    virtual ICanvas_cuda* create_canvas_cuda(
        Sint32 cuda_device_id,
        const char* pixel_type,
        Uint32 width,
        Uint32 height,
        Uint32 layers = 1,
        Float32 gamma = 0.0f) const = 0;

    /// Creates a mipmap from the given canvas.
    ///
    /// \note The base level (the canvas that is passed in) is not included in the returned
    /// canvas array.
    ///
    /// \param canvas           The canvas to create the mipmap from.
    /// \param gamma_override   If this parameter is different from zero, it is used instead of the
    ///                         canvas gamma during mipmap creation.
    /// \return                 An array of type #mi::IPointer containing pointers to
    ///                         the miplevels of type #mi::neuraylib::ICanvas.
    ///                         If no mipmap could be created, \c nullptr is returned.
    virtual IArray* create_mipmap(
        const ICanvas* canvas, Float32 gamma_override = 0.0f) const = 0;

    /// Creates a copy of the passed tile.
    virtual ITile* clone_tile( const ITile* tile) const = 0;

    /// Creates a (deep) copy of the passed canvas.
    virtual ICanvas* clone_canvas( const ICanvas* canvas) const = 0;

    //@}
    /// \name Conversion between canvases and raw memory buffers
    //@{

    /// Reads raw pixel data from a canvas.
    ///
    /// Reads a rectangular area of pixels from a canvas (possibly spanning multiple tiles),
    /// converts the pixel type if needed, and writes the pixel data to buffer in memory.
    /// Management of the buffer memory is the responsibility of the caller.
    ///
    /// \param width               The width of the rectangular pixel area.
    /// \param height              The height of the rectangular pixel area.
    /// \param canvas              The canvas to read the pixel data from.
    /// \param canvas_x            The x-coordinate of the lower-left corner of the rectangle.
    /// \param canvas_y            The y-coordinate of the lower-left corner of the rectangle.
    /// \param canvas_layer        The layer of the canvas that holds the rectangular area.
    /// \param buffer              The buffer to write the pixel data to.
    /// \param buffer_topdown      Indicates whether the buffer stores the rows in top-down order.
    /// \param buffer_pixel_type   The pixel type of the buffer. See \ref mi_neuray_types for a
    ///                            list of supported pixel types.
    /// \param buffer_padding      The padding between subsequent rows of the buffer in bytes.
    /// \return
    ///                            -  0: Success.
    ///                            - -1: Invalid parameters (\c nullptr).
    ///                            - -2: \p width or \p height is zero.
    ///                            - -3: Invalid pixel type of the buffer.
    ///                            - -4: The rectangular area [\p canvas_x, \p canvas_x + \p width)
    ///                                  x [\p canvas_y, \p canvas_y + \p height) exceeds the size
    ///                                  of the canvas, or \p canvas_layer is invalid.
    virtual Sint32 read_raw_pixels(
        Uint32 width,
        Uint32 height,
        const ICanvas* canvas,
        Uint32 canvas_x,
        Uint32 canvas_y,
        Uint32 canvas_layer,
        void* buffer,
        bool buffer_topdown,
        const char* buffer_pixel_type,
        Uint32 buffer_padding = 0) const = 0;

    /// Writes raw pixel data to a canvas.
    ///
    /// Reads a rectangular area of pixels from a buffer in memory, converts the pixel type if
    /// needed, and writes the pixel data to a canvas (possibly spanning multiple tiles).
    /// Management of the buffer memory is the responsibility of the caller.
    ///
    /// \param width               The width of the rectangular pixel area.
    /// \param height              The height of the rectangular pixel area.
    /// \param canvas              The canvas to write the pixel data to.
    /// \param canvas_x            The x-coordinate of the lower-left corner of the rectangle.
    /// \param canvas_y            The y-coordinate of the lower-left corner of the rectangle.
    /// \param canvas_layer        The layer of the canvas that holds the rectangular area.
    /// \param buffer              The buffer to read the pixel data from.
    /// \param buffer_topdown      Indicates whether the buffer stores the rows in top-down order.
    /// \param buffer_pixel_type   The pixel type of the buffer. See \ref mi_neuray_types for a
    ///                            list of supported pixel types.
    /// \param buffer_padding      The padding between subsequent rows of the buffer in bytes.
    /// \return
    ///                            -  0: Success.
    ///                            - -1: Invalid parameters (\c nullptr).
    ///                            - -2: \p width or \p height is zero.
    ///                            - -3: Invalid pixel type of the buffer.
    ///                            - -4: The rectangular area [\p canvas_x, \p canvas_x + \p width)
    ///                                  x [\p canvas_y, \p canvas_y + \p height) exceeds the size
    ///                                  of the canvas, or \p canvas_layer is invalid.
    virtual Sint32 write_raw_pixels(
        Uint32 width,
        Uint32 height,
        ICanvas* canvas,
        Uint32 canvas_x,
        Uint32 canvas_y,
        Uint32 canvas_layer,
        const void* buffer,
        bool buffer_topdown,
        const char* buffer_pixel_type,
        Uint32 buffer_padding = 0) const = 0;

    //@}
    /// \name Conversion between canvases and encoded images
    //@{

    /// Encodes the pixel data of a canvas into a memory buffer.
    ///
    /// \param canvas                The canvas whose contents are to be used.
    /// \param image_format          The desired image format of the image, e.g., \c "jpg". Note
    ///                              that support for a given image format requires an image plugin
    ///                              capable of handling that format.
    /// \param pixel_type            The desired pixel type. See \ref mi_neuray_types for a list of
    ///                              supported pixel types. Not every image plugin supports every
    ///                              pixel type. If the requested pixel type is not supported, the
    ///                              argument is ignored and one of the supported formats is chosen
    ///                              instead.
    /// \param export_options        See \ref mi_image_export_options for supported options.
    /// \return                      The created buffer, or \c nullptr in case of failure.
    virtual IBuffer* create_buffer_from_canvas(
        const ICanvas* canvas,
        const char* image_format,
        const char* pixel_type,
        const IMap* export_options = nullptr) const = 0;

    /// Decodes the pixel data of a memory buffer into a canvas.
    ///
    /// \param buffer        The buffer that holds the encoded pixel data.
    /// \param image_format  The image format of the buffer, e.g., \c "jpg". Note that support for
    ///                      a given image format requires an image plugin capable of handling that
    ///                      format.
    /// \param selector      The selector, or \c nullptr. \ifnot DICE_API See section 2.3.1 in
    ///                      [\ref MDLLS] for details. \endif
    /// \return              The canvas with the decoded pixel data, or \c nullptr in case of
    ///                      failure.
    virtual ICanvas* create_canvas_from_buffer(
        const IBuffer* buffer, const char* image_format, const char* selector = nullptr) const = 0;

    /// Decodes the pixel data from a reader into a canvas.
    ///
    /// \param reader        The reader that provides the data for the image. The reader needs to
    ///                      support absolute access.
    /// \param image_format  The image format of the buffer, e.g., \c "jpg". Note that support for
    ///                      a given image format requires an image plugin capable of handling that
    ///                      format.
    /// \param selector      The selector, or \c nullptr. \ifnot DICE_API See section 2.3.1 in
    ///                      [\ref MDLLS] for details. \endif
    /// \return              The canvas with the decoded pixel data, or \c nullptr in case of
    ///                      failure.
    virtual ICanvas* create_canvas_from_reader(
        IReader* reader, const char* image_format, const char* selector = nullptr) const = 0;

    /// Indicates whether a particular image format is supported for decoding.
    ///
    /// Support for a given image format requires an image plugin capable of handling that format.
    /// This method allows to check whether such a plugin has been loaded for a particular format.
    ///
    /// Decoding is used when the image is converted into a canvas from a \if DICE_API memory
    /// buffer. \else memory buffer or a file \endif. Note that even if this method returns \c true,
    /// #create_canvas_from_buffer() \if IRAY_API or
    /// #mi::neuraylib::IImport_api::import_canvas() \endif can still fail for a particular image if
    /// that image uses an unsupported feature.
    ///
    /// \param image_format   The image format in question, e.g., \c "jpg".
    /// \param reader         An optional reader \if IRAY_API used by
    ///                       #mi::neuraylib::IImage_plugin::test(). \endif
    /// \return               \c true if the image format is supported, \c false otherwise
    virtual bool supports_format_for_decoding(
        const char* image_format, IReader* reader = nullptr) const = 0;

    /// Indicates whether a particular image format is supported for encoding.
    ///
    /// Support for a given image format requires an image plugin capable of handling that format.
    /// This method allows to check whether such a plugin has been loaded for a particular format.
    ///
    /// Encoding is used when the image is converted from a canvas into a \if DICE_API memory
    /// buffer. \else memory buffer or a file. \endif. Note that even if this method returns
    /// \c true, #create_buffer_from_canvas() \if IRAY_API or
    /// #mi::neuraylib::IExport_api::export_canvas \endif can still fail if the given canvas
    /// uses an unsupported feature, e.g., multiple layers.
    ///
    /// \param image_format   The image format in question, e.g., \c "jpg".
    /// \return               \c true if the image format is supported, \c false otherwise
    virtual bool supports_format_for_encoding( const char* image_format) const = 0;

    //@}
    /// \name Utility methods for canvases
    //@{

    /// Converts a tile to a different pixel type.
    ///
    /// \note This method creates a copy if the passed-in tiles already has the desired pixel type.
    /// (It cannot return the passed-in tile since this would require a const cast.) If
    /// performance is critical, you should compare pixel types yourself and skip the method call if
    /// pixel type conversion is not needed.)
    ///
    /// See #convert(const ICanvas*,const char*)const for details of the conversion process.
    ///
    /// \param tile         The tile to convert (or to copy).
    /// \param pixel_type   The desired pixel type. See \ref mi_neuray_types for a list of supported
    ///                     pixel types. If this pixel type is the same as the pixel type of \p
    ///                     tile, then a copy of the tile is returned.
    /// \return             A tile with the requested pixel type, or \c nullptr in case of errors
    ///                     (\p tile is \c nullptr, or \p pixel_type is not valid).
    virtual ITile* convert( const ITile* tile, const char* pixel_type) const = 0;

    /// Converts a canvas to a different pixel type.
    ///
    /// \note This method creates a copy if the passed-in canvas already has the desired pixel type.
    /// (It cannot return the passed-in canvas since this would require a const cast.) If
    /// performance is critical, you should compare pixel types yourself and skip the method call if
    /// pixel type conversion is not needed.)
    ///
    /// The conversion converts a given pixel as follows:
    ///
    /// - Floating-point values are linearly mapped to integers as follows: 0.0f is mapped to 0 and
    ///   1.0f is mapped to 255 or 65535, respectively. Note that the pixel type \c "Sint8" is
    ///   treated as the corresponding unsigned integer type \c "Uint8" here. Floating-point values
    ///   are clamped to [0.0f, 1.0f] beforehand. The reverse conversion uses the corresponding
    ///   inverse mapping.
    /// - Single-channel formats are converted to grey-scale RGB formats by duplicating the value
    ///   in each channel.
    /// - RGB formats are converted to single-channel formats by mixing the RGB channels with
    ///   weights 0.27f for red, 0.67f for green, and 0.06f for blue.
    /// - If an alpha channel is added, the values are set to 1.0f, 255, or 65535 respectively.
    /// - The pixel type \c "Float32<4>" is treated in the same way as \c "Color", \c "Float32<3>"
    ///   in the same way as \c "Rgb_fp", and \c "Sint32" in the same way as \c "Rgba".
    /// - The pixel type \c "Rgbe" is converted via \c "Rgb_fp". Similarly, \c "Rgbea" is converted
    ///   via \c "Color".
    /// - \c "Float32<2>" is converted to single-channel formats by averaging the two channels. If
    ///   \c "Float32<2>" is converted to three- or four-channel formats, the blue channel is set to
    ///   0.0f, or 0, respectively. Conversion of single-channel formats to \c "Float32<2>"
    ///   duplicates the channel. Conversion of three- or four-channel formats to \c "Float32<2>"
    ///   drops the third and fourth channel.
    ///
    /// \param canvas       The canvas to convert (or to copy).
    /// \param pixel_type   The desired pixel type. See \ref mi_neuray_types for a list of supported
    ///                     pixel types. If this pixel type is the same as the pixel type of \p
    ///                     canvas, then a copy of the canvas is returned.
    /// \return             A canvas with the requested pixel type, or \c nullptr in case of errors
    ///                     (\p canvas is \c nullptr, or \p pixel_type is not valid).
    virtual ICanvas* convert( const ICanvas* canvas, const char* pixel_type) const = 0;

    /// Sets the gamma value of a tile and adjusts the pixel data accordingly.
    ///
    /// The alpha channel (if present) is always linear and not affected by this operation.
    ///
    /// \note Gamma adjustments are always done in pixel type "Color" or "Rgb_fp". If necessary,
    ///       the pixel data is converted forth and back automatically (which needs temporary
    ///       buffers).
    ///
    /// \param tile             The tile whose pixel data is to be adjusted.
    /// \param old_gamma        The old gamma value.
    /// \param new_gamma        The new gamma value.
    virtual void adjust_gamma( ITile* tile, Float32 old_gamma, Float32 new_gamma) const = 0;

    /// Sets the gamma value of a canvas and adjusts the pixel data accordingly.
    ///
    /// The alpha channel (if present) is always linear and not affected by this operation.
    ///
    /// \note Gamma adjustments are always done in pixel type "Color" or "Rgb_fp". If necessary,
    ///       the pixel data is converted forth and back automatically (which needs temporary
    ///       buffers).
    ///
    /// \param canvas           The canvas whose pixel data is to be adjusted.
    /// \param new_gamma        The new gamma value.
    virtual void adjust_gamma( ICanvas* canvas, Float32 new_gamma) const = 0;

    //@}
    /// \name Utility methods for pixel type characteristics
    //@{

    /// Returns the number of components per pixel type.
    ///
    /// For example, for the pixel type "Color" the method returns 4 because it consists of four
    /// components R, G, B, and A. Returns 0 in case of invalid pixel types.
    ///
    /// \see #get_bytes_per_component()
    virtual Uint32 get_components_per_pixel( const char* pixel_type) const = 0;

    /// Returns the number of bytes used per pixel component.
    ///
    /// For example, for the pixel type "Color" the method returns 4 because its components are of
    /// type #mi::Float32 which needs 4 bytes. Returns 0 in case of invalid pixel types.
    ///
    /// \see #get_components_per_pixel()
    virtual Uint32 get_bytes_per_component( const char* pixel_type) const = 0;

    //@}
    /// \name Utility methods for RGBA channels
    //@{

    /// Returns the pixel type of an RGBA channel.
    ///
    /// Invalid pixel type/selector combinations are:
    /// - \p pixel_type is not an RGB or RGBA pixel type
    /// - \p selector is not an RGBA channel selector
    /// - \p selector is \c "A", but \p pixel_type has no alpha channel
    ///
    /// \param pixel_type   The pixel type of the mipmap/canvas/tile.
    /// \param selector     The RGBA channel selector.
    /// \return             Returns PT_UNDEF for invalid pixel type/selector combinations.
    ///                     Otherwise, returns PT_SINT8 or PT_FLOAT32, depending on
    ///                     \p pixel_type.
    virtual const char* get_pixel_type_for_channel(
        const char* pixel_type, const char* selector) const = 0;

    /// Extracts an RGBA channel from a canvas.
    ///
    /// \param canvas           The canvas to extract a channel from.
    /// \param selector         The RGBA channel selector.
    /// \return                 The extracted channel, or \c nullptr in case of invalid pixel type/
    ///                         channel selector combinations (see #get_pixel_type_for_channel()).
    virtual ICanvas* extract_channel( const ICanvas* canvas, const char* selector) const = 0;

    /// Extracts an RGBA channel from a tile.
    ///
    /// \param tile             The tile to extract a channel from.
    /// \param selector         The RGBA channel selector.
    /// \return                 The extracted channel, or \c nullptr in case of invalid pixel type/
    ///                         channel selector combinations (see #get_pixel_type_for_channel()).
    virtual ITile* extract_channel( const ITile* tile, const char* selector) const = 0;

    //@}

    virtual IBuffer* deprecated_create_buffer_from_canvas(
        const ICanvas* canvas,
        const char* image_format,
        const char* pixel_type,
        const char* quality,
        bool force_default_gamma) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_15_0
    inline IBuffer* create_buffer_from_canvas(
        const ICanvas* canvas,
        const char* image_format,
        const char* pixel_type,
        const char* quality,
        bool force_default_gamma = false) const
    {
        return deprecated_create_buffer_from_canvas(
            canvas, image_format, pixel_type, quality, force_default_gamma);
    }
#endif
};

/**@}*/ // end group mi_neuray_rendering / mi_neuray_rtmp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IIMAGE_API_H

