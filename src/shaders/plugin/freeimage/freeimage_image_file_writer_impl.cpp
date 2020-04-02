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

#include "freeimage_image_file_writer_impl.h"
#include "freeimage_utilities.h"

#include <mi/neuraylib/iwriter.h>
#include <mi/neuraylib/itile.h>

#include <cassert>
#include <cstring>
#include <string>

namespace MI {

namespace FREEIMAGE {

Image_file_writer_impl::Image_file_writer_impl(
    mi::neuraylib::IWriter* writer,
    const char* pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Float32 gamma,
    mi::Uint32 quality,
    FREE_IMAGE_FORMAT format,
    const std::string& plugin_name)
  : m_resolution_x( resolution_x),
    m_resolution_y( resolution_y),
    m_gamma( gamma),
    m_quality( quality),
    m_format( format),
    m_plugin_name( plugin_name)
{
    m_writer = writer;
    m_writer->retain();

    if(      strcmp( pixel_type, "Rgb"    ) == 0)
        m_bitmap = FreeImage_AllocateT( FIT_BITMAP, m_resolution_x, m_resolution_y, 24);
    else if( strcmp( pixel_type, "Rgba"   ) == 0)
        m_bitmap = FreeImage_AllocateT( FIT_BITMAP, m_resolution_x, m_resolution_y, 32);
    else if( strcmp( pixel_type, "Sint32" ) == 0)
        m_bitmap = FreeImage_AllocateT( FIT_INT32,  m_resolution_x, m_resolution_y);
    else if( strcmp( pixel_type, "Float32") == 0)
        m_bitmap = FreeImage_AllocateT( FIT_FLOAT,  m_resolution_x, m_resolution_y);
    else if( strcmp( pixel_type, "Rgb_16" ) == 0)
        m_bitmap = FreeImage_AllocateT( FIT_RGB16,  m_resolution_x, m_resolution_y);
    else if( strcmp( pixel_type, "Rgba_16") == 0)
        m_bitmap = FreeImage_AllocateT( FIT_RGBA16, m_resolution_x, m_resolution_y);
    else if( strcmp( pixel_type, "Rgb_fp" ) == 0)
        m_bitmap = FreeImage_AllocateT( FIT_RGBF,   m_resolution_x, m_resolution_y);
    else if( strcmp( pixel_type, "Color"  ) == 0)
        m_bitmap = FreeImage_AllocateT( FIT_RGBAF,  m_resolution_x, m_resolution_y);
    else {
        assert( false);
        m_bitmap = 0;
    }

    bool unused = true; // avoid compiler warning
    m_bitmap_pixel_type = convert_freeimage_pixel_type_to_neuray_pixel_type( m_bitmap, unused);
}

Image_file_writer_impl::~Image_file_writer_impl()
{
    if( m_bitmap) {

        FreeImageIO io = construct_io_for_writing();

        int flags = 0;
        // Export JPGs with the desired quality.
        if( m_format == FIF_JPEG)
            flags |= m_quality;
        // Export EXRs using full 32bit precision (default is 16bit precision), enable ZIP
        // compression.
        if( m_format == FIF_EXR)
            flags |= EXR_FLOAT | EXR_ZIP;

        if( !FreeImage_SaveToHandle(
            m_format, m_bitmap, &io, static_cast<fi_handle>( m_writer), flags)) {

            std::string message
                = "The image plugin \"" + m_plugin_name + "\" failed to export an image.";
            log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
        }
        FreeImage_Unload( m_bitmap);
    }
    m_writer->release();
}

const char* Image_file_writer_impl::get_type() const
{
    if( !m_bitmap)
        return 0;

    bool unused = true; // avoid compiler warning
    return convert_freeimage_pixel_type_to_neuray_pixel_type( m_bitmap, unused);
}

mi::Uint32 Image_file_writer_impl::get_resolution_x( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_x;
}

mi::Uint32 Image_file_writer_impl::get_resolution_y( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_y;
}

mi::Uint32 Image_file_writer_impl::get_layers_size( mi::Uint32 level) const
{
    return 1;
}

mi::Uint32 Image_file_writer_impl::get_tile_resolution_x( mi::Uint32 level) const //-V524 PVS
{
    if( level > 0)
        return 0;
    return m_resolution_x;
}

mi::Uint32 Image_file_writer_impl::get_tile_resolution_y( mi::Uint32 level) const //-V524 PVS
{
    if( level > 0)
        return 0;
    return m_resolution_y;
}

mi::Uint32 Image_file_writer_impl::get_miplevels() const
{
    return 1;
}

bool Image_file_writer_impl::get_is_cubemap() const
{
    return false;
}

mi::Float32 Image_file_writer_impl::get_gamma() const
{
    return m_gamma;
}

bool Image_file_writer_impl::read(
    mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 level) const
{
    return false;
}

bool Image_file_writer_impl::write(
    const mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 level)
{
    if( z != 0 || level != 0 || !m_bitmap) {
        assert( false);
        return false;
    }

#ifndef NDEBUG
    const char* tile_pixel_type = tile->get_type();
    assert( strcmp( tile_pixel_type, m_bitmap_pixel_type) == 0);
#endif

    return copy_from_tile_to_bitmap( tile, m_bitmap, x, y);
}

} // namespace FREEIMAGE

} // namespace MI
