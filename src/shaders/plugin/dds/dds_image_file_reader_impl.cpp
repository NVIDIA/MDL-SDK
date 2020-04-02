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

#include "dds_image_file_reader_impl.h"

#include "dds_decompress.h"
#include "dds_utilities.h"

#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/itile.h>
#include <io/image/image/i_image_utilities.h>

#include <algorithm>
#include <cassert>

namespace MI {

namespace DDS {

Image_file_reader_impl::Image_file_reader_impl( mi::neuraylib::IReader* reader)
{
    m_reader = reader;
    m_reader->retain();

    Dds_compress_fmt compress_format = DXTC_none; // avoid warning
    m_image.load_header( m_reader, m_header, m_pixel_type, compress_format);
}

Image_file_reader_impl::~Image_file_reader_impl()
{
    m_reader->release();
}

const char* Image_file_reader_impl::get_type() const
{
    return IMAGE::convert_pixel_type_enum_to_string( m_pixel_type);
}

mi::Uint32 Image_file_reader_impl::get_resolution_x( mi::Uint32 level) const
{
    if( level >= m_header.m_mipmap_count)
        return 0;
    return std::max( m_header.m_width >> level, 1u);
}

mi::Uint32 Image_file_reader_impl::get_resolution_y( mi::Uint32 level) const
{
    if( level >= m_header.m_mipmap_count)
        return 0;
    return std::max( m_header.m_height >> level, 1u);
}

mi::Uint32 Image_file_reader_impl::get_layers_size( mi::Uint32 level) const
{
    if( level >= m_header.m_mipmap_count)
        return 0;

    return m_header.m_depth;
}

mi::Uint32 Image_file_reader_impl::get_tile_resolution_x( mi::Uint32 level) const //-V524 PVS
{
    if( level >= m_header.m_mipmap_count)
        return 0;
    return std::max( m_header.m_width >> level, 1u);
}

mi::Uint32 Image_file_reader_impl::get_tile_resolution_y( mi::Uint32 level) const //-V524 PVS
{
    if( level >= m_header.m_mipmap_count)
        return 0;
    return std::max( m_header.m_height >> level, 1u);
}

mi::Uint32 Image_file_reader_impl::get_miplevels() const
{
    return m_header.m_mipmap_count;
}

bool Image_file_reader_impl::get_is_cubemap() const
{
    return (m_header.m_caps2 & DDSF_CUBEMAP) && (m_header.m_depth == 6);
}

mi::Float32 Image_file_reader_impl::get_gamma() const
{
    return IMAGE::get_default_gamma( m_pixel_type);
}

bool Image_file_reader_impl::read(
    mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 level) const
{
    if( level >= get_miplevels() || z >= get_layers_size( level))
        return false;

    if( !m_image.is_valid()) {
        m_reader->seek_absolute( 0);
        if( !m_image.load( m_reader))
            return false;
    }

    const Surface& surface = m_image.get_surface( level);
    mi::Uint32 image_width  = surface.get_width();
    mi::Uint32 image_height = surface.get_height();

    // Non compressed images
    if( !m_image.is_compressed()) {

        const char* pixel_type = tile->get_type();
        mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
        mi::Uint32 bytes_per_layer = image_width * image_height * bytes_per_pixel;
        copy_from_dds_to_tile(
            surface.get_pixels() + z * bytes_per_layer, x, y, image_width, image_height, tile);

    } else {

        // Compressed images
        const char* pixel_type = tile->get_type();
        mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
        mi::Uint32 bytes_per_layer = image_width * image_height * bytes_per_pixel;

        Dxt_decompressor decompressor;
        decompressor.set_source_format( m_image.get_compressed_format(), image_width, image_height);
        decompressor.set_target_format( get_components_per_pixel( pixel_type), image_width);

        mi::Uint32 block_height = decompressor.get_block_dimension();
        mi::Uint32 bytes_per_block = image_width * block_height * bytes_per_pixel;

        const mi::Uint8* src = surface.get_pixels() + z * surface.get_size() / 6;
        const mi::Uint8* buffer1 = decompressor.get_buffer();
        mi::Uint8* buffer2 = new mi::Uint8[bytes_per_layer];

        for( mi::Uint32 block = 0; block < decompressor.get_block_count_y(); ++block) {
            decompressor.decompress_blockline( src, block);
            memcpy( buffer2 + block * bytes_per_block, buffer1, bytes_per_block);
        }

        copy_from_dds_to_tile( buffer2, x, y, image_width, image_height, tile);
        delete[] buffer2;
    }

    return true;
}

bool Image_file_reader_impl::write(
    const mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 level)
{
    return false;
}

} // namespace DDS

} // namespace MI
