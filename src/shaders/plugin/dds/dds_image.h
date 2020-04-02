/***************************************************************************************************
 * Copyright (c) 2005-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_IMAGE_DDS_DDS_IMAGE_H
#define IO_IMAGE_DDS_DDS_IMAGE_H

#include "dds_types.h"
#include "dds_texture.h"

#include <mi/base/types.h>
#include <cassert>
#include <io/image/image/i_image_utilities.h>

namespace mi { namespace neuraylib { class IReader; class IWriter; } }

namespace MI {

namespace DDS {

/// An image (in DDS terms) is a wrapper around a DDS texture.
///
/// Note that this classes does not take care of compression. If the data is compressed and you need
/// it uncompressed, you have to apply the Dxt_decompressor yourself.
class Image
{
public:
    /// Creates an invalid image.
    Image();

    /// Creates a valid image from a texture.
    ///
    /// Calls clear() first. The texture is copied. The image becomes valid afterwards.
    void create(
        IMAGE::Pixel_type pixel_type,
        const Texture& texture,
        bool is_cubemap = false,
        Dds_compress_fmt format = DXTC_none);

    /// Destroys an image.
    ///
    /// The image is in the same state as after default construction.
    void clear();

    /// Loads only the header of a DDS image.
    ///
    /// \param reader                 The reader to load the DDS header from.
    /// \param header[out]            The header information is stored here.
    /// \param pixel_type[out]        The pixel type (decoded from the header) is stored here.
    /// \param compress_format[out]   The compression format (dec. from the header) is stored here.
    /// \param for_hw                 Indicates whether the image is intended to be used with
    ///                               hardware (only needed to decide whether the format is
    ///                               supported or not).
    /// \return                       \c true if the file format can be read, \c false otherwise.
    static bool load_header(
        mi::neuraylib::IReader* reader,
        Header& header,
        IMAGE::Pixel_type& pixel_type,
        Dds_compress_fmt& compress_format,
        bool for_hw = false);

    /// Loads a DDS image.
    ///
    /// Calls clear() first. The image becomes valid after loading (unless there is failure).
    ///
    /// \param reader            The reader to load the DDS header from.
    /// \param for_hw            Indicates whether the image is intended to be used with hardware.
    /// \return                  \c true if the file format can be read, \c false otherwise.
    bool load(
        mi::neuraylib::IReader* reader,
        bool for_hw = false);

    /// Saves a DDS image.
    ///
    /// Afterwards the image becomes invalid (otherwise all the in-place transformations would
    /// have to be undone).
    ///
    /// \param writer            The writer to write the DDS image to.
    /// \return                  \c true if the image can be saved, \c false otherwise.
    bool save(
        mi::neuraylib::IWriter* writer);

    /// Indicates whether the image is valid.
    bool is_valid() const { return m_valid; }

    /// Indicates whether the image represents a cubemap.
    bool is_cubemap() const { return m_texture_type == TEXTURE_CUBEMAP; }

    /// Indicates whether the image represents a 3D texture.
    bool is_volume() const { return m_texture_type == TEXTURE_3D; }

    /// Returns the pixel format.
    IMAGE::Pixel_type get_pixel_type() const { return m_pixel_type; }

    /// Returns Get the number of components in the image
    mi::Uint32 get_components() const { return IMAGE::get_components_per_pixel( m_pixel_type); }

    /// Returns the compression format.
    Dds_compress_fmt get_compressed_format() const { return m_compress_format; }

    /// Indicates whether the image is compressed.
    bool is_compressed() const { return m_compress_format != DXTC_none; }

    /// Returns the number of surfaces.
    mi::Uint32 get_num_surfaces() const
    {
        return m_valid ? static_cast<mi::Uint32>( m_texture.get_num_surfaces()) : 0;
    }

    /// Returns the given surface (const).
    ///
    /// \pre The image is not a cubemap.
    const Surface& get_surface( mi::Size level) const { return m_texture.get_surface( level); }

    /// Returns the width of the image.
    mi::Uint32 get_width() const  { return m_valid ? m_texture.get_surface( 0).get_width()  : 0; }

    /// Returns the height of the image.
    mi::Uint32 get_height() const { return m_valid ? m_texture.get_surface( 0).get_height() : 0; }

    /// Returns the depth of the image.
    mi::Uint32 get_depth() const  { return m_valid ? m_texture.get_surface( 0).get_depth()  : 0; }

private:

    /// Returns the size of an surface with the given width and height and depth 1.
    ///
    /// Takes compression into account (if the surface is compressed).
    mi::Uint32 get_layer_size( mi::Uint32 width, mi::Uint32 height);

    /// Flips surface around X axis.
    void flip_surface( Surface& surface);

    /// Flips DXTC1 blocks.
    static void flip_blocks_dxtc1( DXT_color_block* line, mi::Uint32 num_blocks);

    /// Flips DXTC3 blocks.
    static void flip_blocks_dxtc3( DXT_color_block* line, mi::Uint32 num_blocks);

    /// Flips DXTC5 blocks.
    static void flip_blocks_dxtc5( DXT_color_block* line, mi::Uint32 num_blocks);

    /// Flips a DXTC5 alpha block.
    static void flip_dxt5_alpha( DXT5_alpha_block* block);

    /// Reorders uncompressed DDS RGB(A) pixel data into neuray RGB(A) component order
    ///
    /// The DDS RGB(A) pixel data actually might not be in RGB(A) order, but in a different order.
    /// This method brings it into RGB(A) order.
    void reorder_rgb_or_rgba( Header& header);

    /// Swap two memory ranges.
    static void swap( void* addr1, void* addr2, mi::Uint32 size);

    /// Expand half data to float data.
    static void expand_half( std::vector<mi::Uint8>& buffer);

    /// Compression format of the pixel data.
    Dds_compress_fmt m_compress_format;

    /// The pixel type of the pixel data.
    IMAGE::Pixel_type m_pixel_type;

    /// The type of the texture.
    Texture_type m_texture_type;

    /// Is the image valid?
    bool m_valid;

    /// The texture that is wrapped by this image.
    ///
    /// Note that DDS considers the faces of a cubemap as separate textures. But this code converts
    /// them to layers of the same texture (like a 3D texture of depth 6).
    ///
    /// Also notethat  DDS stores images top-down, while neuray stores them bottom-up. Hence, all
    /// images have to be flipped, except for cubemaps, which are stored top-down in neuray.
    Texture m_texture;
};

} // namespace DDS

} // namespace MI

#endif // IO_IMAGE_DDS_DDS_IMAGE_H
