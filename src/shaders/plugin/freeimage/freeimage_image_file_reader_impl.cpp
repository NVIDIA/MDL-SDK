/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "freeimage_image_file_reader_impl.h"
#include "freeimage_utilities.h"

#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/itile.h>

#include <cassert>
#include <cstring>
#include <io/image/image/i_image_utilities.h>


namespace MI {

namespace FREEIMAGE {

Image_file_reader_impl::Image_file_reader_impl(
    mi::neuraylib::IReader* reader, FREE_IMAGE_FORMAT format)
  : m_resolution_x( 1), m_resolution_y( 1), m_format( format), m_bitmap( 0), m_bitmap_pixel_type( 0)
{
    m_reader = reader;
    m_reader->retain();

    FreeImageIO io = construct_io_for_reading();
    m_format = FreeImage_GetFileTypeFromHandle( &io, static_cast<fi_handle>( reader));
    assert( m_format == format);

    int flags = 0;
    // Import JPEGs with full accuracy (to get identical results as with the default settings of
    // the JPEG library).
    if( m_format == FIF_JPEG)
        flags |= JPEG_ACCURATE;
    // Import PNGs ignoring the gamma value (that is what our old PNG plugin did).
    if( m_format == FIF_PNG)
        flags |= PNG_IGNOREGAMMA;
    // Delay pixel loading (if supported for this format). Disable delayed pixel loading for TIFF
    // files (see bug 12086 or https://sourceforge.net/p/freeimage/bugs/233/).
    if( m_format != FIF_TIFF && FreeImage_FIFSupportsNoPixels( m_format))
        flags |= FIF_LOAD_NOPIXELS;

    m_bitmap = FreeImage_LoadFromHandle( m_format, &io, static_cast<fi_handle>( m_reader), flags);

    if( !m_bitmap) {
        m_resolution_x = 1;
        m_resolution_y = 1;
        return;
    }

    bool convert = true; // avoid compiler warning
    m_bitmap_pixel_type = convert_freeimage_pixel_type_to_neuray_pixel_type( m_bitmap, convert);

    if( convert && FreeImage_HasPixels( m_bitmap)) {
        FIBITMAP* new_bitmap;
        if( strcmp( m_bitmap_pixel_type, "Rgb") == 0)
            new_bitmap = FreeImage_ConvertTo24Bits( m_bitmap);
        else if( strcmp( m_bitmap_pixel_type, "Rgba") == 0)
            new_bitmap = FreeImage_ConvertTo32Bits( m_bitmap);
        else if( strcmp( m_bitmap_pixel_type, "Rgb_fp") == 0)
            new_bitmap = FreeImage_ConvertToRGBF( m_bitmap);
        else {
            assert( false);
            new_bitmap = FreeImage_ConvertToRGBF( m_bitmap);
        }
        FreeImage_Unload( m_bitmap);
        m_bitmap = new_bitmap;
    }

    m_resolution_x = FreeImage_GetWidth( m_bitmap);
    m_resolution_y = FreeImage_GetHeight( m_bitmap);

    // For one-dimensional DDS textures the height is incorrectly reported as 0.
    if( format == FIF_DDS && m_resolution_y == 0)
        m_resolution_y = 1;
}

Image_file_reader_impl::~Image_file_reader_impl()
{
    m_reader->release();
    if( m_bitmap)
        FreeImage_Unload( m_bitmap);
}

const char* Image_file_reader_impl::get_type() const
{
    return m_bitmap_pixel_type;
}

mi::Uint32 Image_file_reader_impl::get_resolution_x( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_x;
}

mi::Uint32 Image_file_reader_impl::get_resolution_y( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_y;
}

mi::Uint32 Image_file_reader_impl::get_layers_size( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return 1;
}

mi::Uint32 Image_file_reader_impl::get_tile_resolution_x( mi::Uint32 level) const //-V524 PVS
{
    if( level > 0)
        return 0;
    return m_resolution_x;
}

mi::Uint32 Image_file_reader_impl::get_tile_resolution_y( mi::Uint32 level) const //-V524 PVS
{
    if( level > 0)
        return 0;
    return m_resolution_y;
}

mi::Uint32 Image_file_reader_impl::get_miplevels() const
{
    return 1;
}

bool Image_file_reader_impl::get_is_cubemap() const
{
    return false;
}

mi::Float32 Image_file_reader_impl::get_gamma() const
{
    IMAGE::Pixel_type pixel_type = IMAGE::convert_pixel_type_string_to_enum( m_bitmap_pixel_type);
    return IMAGE::get_default_gamma( pixel_type);
}

bool Image_file_reader_impl::read(
    mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 level) const
{
    if( !m_bitmap || z != 0 || level != 0)
        return false;

#ifndef NDEBUG
    const char* tile_pixel_type = tile->get_type();
    assert( strcmp( tile_pixel_type, m_bitmap_pixel_type) == 0);
#endif

    if( !FreeImage_HasPixels( m_bitmap)) {

        FreeImage_Unload( m_bitmap);

        int flags = 0;
        // Import JPEGs with full accuracy (to get identical results as with the default settings of
        // the JPEG library).
        if( m_format == FIF_JPEG)
            flags |= JPEG_ACCURATE;
        // Import PNGs ignoring the gamma value (that is what our old PNG plugin did).
        if( m_format == FIF_PNG)
            flags |= PNG_IGNOREGAMMA;

        m_reader->seek_absolute( 0);
        FreeImageIO io = construct_io_for_reading();
        m_bitmap = FreeImage_LoadFromHandle(
            m_format, &io, static_cast<fi_handle>( m_reader), flags);

        if( !m_bitmap)
            return false;

        bool convert = true; // avoid compiler warning
        const char* pixel_type
            = convert_freeimage_pixel_type_to_neuray_pixel_type( m_bitmap, convert);
        assert( strcmp( m_bitmap_pixel_type, pixel_type) == 0);
        (void) pixel_type;

        if( convert) {
            FIBITMAP* new_bitmap;
            if( strcmp( m_bitmap_pixel_type, "Rgb") == 0)
                new_bitmap = FreeImage_ConvertTo24Bits( m_bitmap);
            else if( strcmp( m_bitmap_pixel_type, "Rgba") == 0)
                new_bitmap = FreeImage_ConvertTo32Bits( m_bitmap);
            else if( strcmp( m_bitmap_pixel_type, "Rgb_fp") == 0)
                new_bitmap = FreeImage_ConvertToRGBF( m_bitmap);
            else {
                assert( false);
                new_bitmap = FreeImage_ConvertToRGBF( m_bitmap);
            }
            FreeImage_Unload( m_bitmap);
            m_bitmap = new_bitmap;
        }

    }

    return copy_from_bitmap_to_tile( m_bitmap, x, y, tile);
}

bool Image_file_reader_impl::write(
    const mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 level)
{
    return false;
}

} // namespace FREEIMAGE

} // namespace MI
