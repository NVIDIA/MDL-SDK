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

#include "dds_image_file_writer_impl.h"

#include "dds_image.h"
#include "dds_utilities.h"

#include <mi/neuraylib/iwriter.h>
#include <mi/neuraylib/itile.h>

#include <algorithm>
#include <cassert>

namespace MI {

namespace DDS {

Image_file_writer_impl::Image_file_writer_impl(
    mi::neuraylib::IWriter* writer,
    const char* pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Uint32 nr_of_layers,
    mi::Uint32 miplevels,
    bool is_cubemap,
    mi::Float32 gamma)
  : m_resolution_x( resolution_x),
    m_resolution_y( resolution_y),
    m_nr_of_layers( nr_of_layers),
    m_miplevels( miplevels),
    m_is_cubemap( is_cubemap),
    m_gamma( gamma)
{
    assert( !m_is_cubemap || m_nr_of_layers == 6);

    m_writer = writer;
    m_writer->retain();

    m_pixel_type = IMAGE::convert_pixel_type_string_to_enum( pixel_type);

    m_level.resize( m_miplevels);

    for( mi::Size i = 0; i < m_miplevels; ++i) {
        mi::Uint32 width  = std::max( m_resolution_x >> i, 1u);
        mi::Uint32 height = std::max( m_resolution_y >> i, 1u);
        mi::Uint32 depth  = std::max( m_nr_of_layers >> i, 1u);
        mi::Uint32 bytes_per_level = width * height * depth * get_bytes_per_pixel( m_pixel_type);
        m_level[i].resize( bytes_per_level);
    }
}

Image_file_writer_impl::~Image_file_writer_impl()
{
    mi::Uint32 width  = m_resolution_x;
    mi::Uint32 height = m_resolution_y;
    mi::Uint32 depth  = m_nr_of_layers;
    mi::Uint32 bytes_per_level = width * height * depth * get_bytes_per_pixel( m_pixel_type);

    Texture texture;

    for( mi::Size i = 0; i < m_miplevels; ++i) {

        Surface surface( width, height, depth, bytes_per_level, &m_level[i][0]);
        texture.add_surface( surface);

        width  = std::max( width  / 2, 1u);
        height = std::max( height / 2, 1u);
        depth  = std::max( depth  / 2, 1u);
        bytes_per_level = width * height * depth * get_bytes_per_pixel( m_pixel_type);
    }

    Image image;
    image.create( m_pixel_type, texture, m_is_cubemap);
    if( !image.save( m_writer)) {
        log( mi::base::MESSAGE_SEVERITY_ERROR,
            "The image plugin \"dds\" failed to export an image.");
    }

    m_writer->release();
}

const char* Image_file_writer_impl::get_type() const
{
    return IMAGE::convert_pixel_type_enum_to_string( m_pixel_type);
}

mi::Uint32 Image_file_writer_impl::get_resolution_x( mi::Uint32 level) const
{
    if( level >= m_miplevels)
        return 0;
    return std::max( m_resolution_x >> level, 1u);
}

mi::Uint32 Image_file_writer_impl::get_resolution_y( mi::Uint32 level) const
{
    if( level >= m_miplevels)
        return 0;
    return std::max( m_resolution_y >> level, 1u);
}

mi::Uint32 Image_file_writer_impl::get_layers_size( mi::Uint32 level) const
{
    if( level >= m_miplevels)
        return 0;
    return m_nr_of_layers;
}

mi::Uint32 Image_file_writer_impl::get_tile_resolution_x( mi::Uint32 level) const //-V524 PVS
{
    if( level >= m_miplevels)
        return 0;
    return std::max( m_resolution_x >> level, 1u);
}

mi::Uint32 Image_file_writer_impl::get_tile_resolution_y( mi::Uint32 level) const //-V524 PVS
{
    if( level >= m_miplevels)
        return 0;
    return std::max( m_resolution_y >> level, 1u);
}

mi::Uint32 Image_file_writer_impl::get_miplevels() const
{
    return m_miplevels;
}

bool Image_file_writer_impl::get_is_cubemap() const
{
    return m_is_cubemap;
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
    if( level >= get_miplevels() || z >= get_layers_size( level))
        return false;

#ifndef NDEBUG
    IMAGE::Pixel_type tile_pixel_type
        = IMAGE::convert_pixel_type_string_to_enum( tile->get_type());
    assert( tile_pixel_type == m_pixel_type);
#endif

    mi::Uint32 image_width     = get_resolution_x( level);
    mi::Uint32 image_height    = get_resolution_y( level);
    mi::Uint32 bytes_per_pixel = IMAGE::get_bytes_per_pixel( m_pixel_type);
    mi::Uint32 bytes_per_layer = image_width * image_height * bytes_per_pixel;

    copy_from_tile_to_dds(
        tile, &m_level[level][0] + z * bytes_per_layer, x, y, image_width, image_height);

    return true;
}

} // namespace DDS

} // namespace MI
