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

#include "i_image_access_canvas.h"
#include "i_image_pixel_conversion.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/itile.h>

#include <algorithm>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace IMAGE {

Access_canvas::Access_canvas( const mi::neuraylib::ICanvas* canvas, bool lockless)
  : m_lockless( lockless)
{
    set( canvas);
}

Access_canvas::Access_canvas( const Access_canvas& rhs)
{
    mi::base::Lock::Block block;
    if( !rhs.m_lockless) block.set( &rhs.m_tiles_lock);

    m_lockless = rhs.m_lockless;
    set( rhs.m_canvas.get());
}

Access_canvas& Access_canvas::operator=( const Access_canvas& rhs)
{
    if( this == &rhs)
        return *this;

    mi::base::Lock::Block block;
    if( !rhs.m_lockless) block.set( &rhs.m_tiles_lock);

    m_lockless = rhs.m_lockless;
    set( rhs.m_canvas.get());
    return *this;
}

void Access_canvas::set( const mi::neuraylib::ICanvas* canvas)
{
    mi::base::Lock::Block block;
    if( !m_lockless) block.set( &m_tiles_lock);

    m_canvas = make_handle_dup( canvas);
    if( !m_canvas) {
        m_nr_of_layers      = 0;
        m_canvas_width      = 0;
        m_canvas_height     = 0;
        m_canvas_pixel_type = PT_UNDEF;
        m_tile_width        = 0;
        m_tile_height       = 0;
        m_nr_of_tiles_x     = 0;
        m_nr_of_tiles_y     = 0;
        m_nr_of_tiles       = 0;
        m_tiles.clear();
    } else {
        m_nr_of_layers      = m_canvas->get_layers_size();
        m_canvas_width      = m_canvas->get_resolution_x();
        m_canvas_height     = m_canvas->get_resolution_y();
        m_canvas_pixel_type = convert_pixel_type_string_to_enum( m_canvas->get_type());
        m_tile_width        = m_canvas->get_tile_resolution_x();
        m_tile_height       = m_canvas->get_tile_resolution_y();
        m_nr_of_tiles_x     = (m_canvas_width  + m_tile_width  - 1) / m_tile_width;
        m_nr_of_tiles_y     = (m_canvas_height + m_tile_height - 1) / m_tile_height;
        m_nr_of_tiles       = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;
        m_tiles.clear(); // clear previously cached elements
        m_tiles.resize( m_nr_of_tiles);
    }

    if( !m_lockless)
        return;

    // Prefetch all tiles in the lockless variant.
    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z)
        for( mi::Uint32 tile_y = 0; tile_y < m_nr_of_tiles_y; ++tile_y)
            for( mi::Uint32 tile_x = 0; tile_x < m_nr_of_tiles_x; ++tile_x) {
                mi::Uint32 index = z * m_nr_of_tiles_x*m_nr_of_tiles_y
                                   + tile_y * m_nr_of_tiles_x + tile_x;
                ASSERT( M_IMAGE, index < m_nr_of_tiles);
                mi::Uint32 pixel_x = tile_x * m_tile_width;
                mi::Uint32 pixel_y = tile_y * m_tile_height;
                m_tiles[index] = m_canvas->get_tile( pixel_x, pixel_y, z);
            }
}

const mi::neuraylib::ICanvas* Access_canvas::get() const
{
    mi::base::Lock::Block block;
    if( !m_lockless) block.set( &m_tiles_lock);

    if( !m_canvas)
        return 0;
    m_canvas->retain();
    return m_canvas.get();
}

bool Access_canvas::read_rect(
    mi::Uint8* buffer,
    bool buffer_topdown,
    Pixel_type buffer_pixel_type,
    mi::Uint32 canvas_x,
    mi::Uint32 canvas_y,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 buffer_padding,
    mi::Uint32 canvas_layer) const
{
    if( !buffer || !m_canvas)
        return false;

    if( canvas_layer >= m_nr_of_layers)
        return false;

    if( canvas_x + width > m_canvas_width || canvas_y + height > m_canvas_height)
        return false;

    if( m_canvas_pixel_type == PT_UNDEF)
        return false;

    if( !exists_pixel_conversion( m_canvas_pixel_type, buffer_pixel_type))
        return false;

    mi::Uint32 canvas_bytes_per_pixel = get_bytes_per_pixel( m_canvas_pixel_type);
    mi::Uint32 buffer_bytes_per_pixel = get_bytes_per_pixel( buffer_pixel_type);

    // Compute canvas_x/canvas_y rounded down to multiples of m_tile_width/m_tile_height
    mi::Uint32 canvas_x_rd = (canvas_x / m_tile_width ) * m_tile_width;
    mi::Uint32 canvas_y_rd = (canvas_y / m_tile_height) * m_tile_height;

    // Loop over the affected tiles, (x,y) is the lower left corner of the tile
    for( mi::Uint32 y = canvas_y_rd; y < canvas_y + height; y += m_tile_height) {
        for( mi::Uint32 x = canvas_x_rd; x < canvas_x + width; x += m_tile_width) {

            // Compute height of rectangle that falls into this tile
            mi::Uint32 local_height = m_tile_height;
            if( y < canvas_y)
                local_height -= canvas_y - y;
            if( y + m_tile_height > canvas_y + height)
                local_height -= y + m_tile_height - (canvas_y + height);

            // Compute width of rectangle that falls into this tile
            mi::Uint32 local_width = m_tile_width;
            if( x < canvas_x)
                local_width -= canvas_x - x;
            if( x + m_tile_width > canvas_x + width)
                local_width -= x + m_tile_width - (canvas_x + width);

            // Compute the pointer to the lower left corner of the rectangle that falls into this
            // tile, and stride per row (canvas, source).
            mi::Difference source_stride = m_tile_width * canvas_bytes_per_pixel;
            mi::base::Handle<const mi::neuraylib::ITile> tile(
                m_canvas->get_tile( x, y, canvas_layer));
            const mi::Uint8* tile_data = static_cast<const mi::Uint8*>( tile->get_data());
            mi::Uint32 local_x = std::max( canvas_x, x) % m_tile_width;
            mi::Uint32 local_y = std::max( canvas_y, y) % m_tile_height;
            const mi::Uint8* source
                = tile_data + local_y * source_stride + local_x * canvas_bytes_per_pixel;

            // Compute the pointer to the lower left corner of the rectangle that falls into this
            // tile, and stride per row (buffer, dest).
            mi::Difference dest_stride = width * buffer_bytes_per_pixel + buffer_padding;
            mi::Uint8* dest = buffer;
            if( buffer_topdown) {
                dest += (height - 1) * dest_stride;
                dest_stride = -dest_stride;
            }
            if( y > canvas_y)
                dest += (y - canvas_y) * dest_stride;
            if( x > canvas_x)
                dest += (x - canvas_x) * buffer_bytes_per_pixel;

            // Copy pixel data for rectangle that falls into this tile
            convert( source, dest, m_canvas_pixel_type, buffer_pixel_type, local_width,
                local_height, source_stride, dest_stride);
        }
    }

    return true;
}

bool Access_canvas::lookup(
    mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) const
{
    if( x >= m_canvas_width || y >= m_canvas_height || z >= m_nr_of_layers)
        return false;

    const mi::Uint32 tile_x  = x / m_tile_width;
    const mi::Uint32 local_x = x % m_tile_width;
    const mi::Uint32 tile_y  = y / m_tile_height;
    const mi::Uint32 local_y = y % m_tile_height;
    const mi::Size   tile_index = (z*m_nr_of_tiles_y + tile_y) * m_nr_of_tiles_x + tile_x;
    ASSERT( M_NEURAY_API, tile_index < m_nr_of_tiles);

    mi::base::Lock::Block block;
    if( !m_lockless) {
        block.set( &m_tiles_lock);
        if( !m_tiles[tile_index])
            m_tiles[tile_index] = m_canvas->get_tile( x, y, z);
    }

    m_tiles[tile_index]->get_pixel( local_x, local_y, &color.r);
    return true;
}

Edit_canvas::Edit_canvas( mi::neuraylib::ICanvas* canvas, bool lockless)
  : m_lockless( lockless)
{
    set( canvas);
}

Edit_canvas::Edit_canvas( const Edit_canvas& rhs)
{
    mi::base::Lock::Block block;
    if( !rhs.m_lockless) block.set( &rhs.m_tiles_lock);

    m_lockless = rhs.m_lockless;
    set( rhs.m_canvas.get());
}

Edit_canvas& Edit_canvas::operator=( const Edit_canvas& rhs)
{
    if( this == &rhs)
        return *this;

    mi::base::Lock::Block block;
    if( !rhs.m_lockless) block.set( &rhs.m_tiles_lock);

    m_lockless = rhs.m_lockless;
    set( rhs.m_canvas.get());
    return *this;
}

void Edit_canvas::set( mi::neuraylib::ICanvas* canvas)
{
    mi::base::Lock::Block block;
    if( !m_lockless) block.set( &m_tiles_lock);

    m_canvas = make_handle_dup( canvas);
    if( !m_canvas) {
        m_nr_of_layers      = 0;
        m_canvas_width      = 0;
        m_canvas_height     = 0;
        m_canvas_pixel_type = PT_UNDEF;
        m_tile_width        = 0;
        m_tile_height       = 0;
        m_nr_of_tiles_x     = 0;
        m_nr_of_tiles_y     = 0;
        m_nr_of_tiles       = 0;
        m_tiles.clear();
    } else {
        m_nr_of_layers      = m_canvas->get_layers_size();
        m_canvas_width      = m_canvas->get_resolution_x();
        m_canvas_height     = m_canvas->get_resolution_y();
        m_canvas_pixel_type = convert_pixel_type_string_to_enum( m_canvas->get_type());
        m_tile_width        = m_canvas->get_tile_resolution_x();
        m_tile_height       = m_canvas->get_tile_resolution_y();
        m_nr_of_tiles_x     = (m_canvas_width  + m_tile_width  - 1) / m_tile_width;
        m_nr_of_tiles_y     = (m_canvas_height + m_tile_height - 1) / m_tile_height;
        m_nr_of_tiles       = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;
        m_tiles.clear(); // clear previously cached elements
        m_tiles.resize( m_nr_of_tiles);
    }

    if( !m_lockless)
        return;

    // Prefetch all tiles in the lockless variant.
    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z)
        for( mi::Uint32 tile_y = 0; tile_y < m_nr_of_tiles_y; ++tile_y)
            for( mi::Uint32 tile_x = 0; tile_x < m_nr_of_tiles_x; ++tile_x) {
                mi::Uint32 index = z * m_nr_of_tiles_x*m_nr_of_tiles_y
                                   + tile_y * m_nr_of_tiles_x + tile_x;
                ASSERT( M_IMAGE, index < m_nr_of_tiles);
                mi::Uint32 pixel_x = tile_x * m_tile_width;
                mi::Uint32 pixel_y = tile_y * m_tile_height;
                m_tiles[index] = m_canvas->get_tile( pixel_x, pixel_y, z);
            }
}

mi::neuraylib::ICanvas* Edit_canvas::get() const
{
    mi::base::Lock::Block block;
    if( !m_lockless) block.set( &m_tiles_lock);

    if( !m_canvas)
        return 0;
    m_canvas->retain();
    return m_canvas.get();
}

bool Edit_canvas::read_rect(
    mi::Uint8* buffer,
    bool buffer_topdown,
    Pixel_type buffer_pixel_type,
    mi::Uint32 canvas_x,
    mi::Uint32 canvas_y,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 buffer_padding,
    mi::Uint32 canvas_layer) const
{
    if( !buffer || !m_canvas)
        return false;

    if( canvas_layer >= m_nr_of_layers)
        return false;

    if( canvas_x + width > m_canvas_width || canvas_y + height > m_canvas_height)
        return false;

    if( m_canvas_pixel_type == PT_UNDEF)
        return false;

    if( !exists_pixel_conversion( m_canvas_pixel_type, buffer_pixel_type))
        return false;

    mi::Uint32 canvas_bytes_per_pixel = get_bytes_per_pixel( m_canvas_pixel_type);
    mi::Uint32 buffer_bytes_per_pixel = get_bytes_per_pixel( buffer_pixel_type);

    // Compute canvas_x/canvas_y rounded down to multiples of m_tile_width/m_tile_height
    mi::Uint32 canvas_x_rd = (canvas_x / m_tile_width ) * m_tile_width;
    mi::Uint32 canvas_y_rd = (canvas_y / m_tile_height) * m_tile_height;

    // Loop over the affected tiles, (x,y) is the lower left corner of the tile
    for( mi::Uint32 y = canvas_y_rd; y < canvas_y + height; y += m_tile_height) {
        for( mi::Uint32 x = canvas_x_rd; x < canvas_x + width; x += m_tile_width) {

            // Compute height of rectangle that falls into this tile
            mi::Uint32 local_height = m_tile_height;
            if( y < canvas_y)
                local_height -= canvas_y - y;
            if( y + m_tile_height > canvas_y + height)
                local_height -= y + m_tile_height - (canvas_y + height);

            // Compute width of rectangle that falls into this tile
            mi::Uint32 local_width = m_tile_width;
            if( x < canvas_x)
                local_width -= canvas_x - x;
            if( x + m_tile_width > canvas_x + width)
                local_width -= x + m_tile_width - (canvas_x + width);

            // Compute the pointer to the lower left corner of the rectangle that falls into this
            // tile, and stride per row (canvas, source).
            mi::Difference source_stride = m_tile_width * canvas_bytes_per_pixel;
            mi::base::Handle<const mi::neuraylib::ITile> tile(
                m_canvas->get_tile( x, y, canvas_layer));
            const mi::Uint8* tile_data = static_cast<const mi::Uint8*>( tile->get_data());
            mi::Uint32 local_x = std::max( canvas_x, x) % m_tile_width;
            mi::Uint32 local_y = std::max( canvas_y, y) % m_tile_height;
            const mi::Uint8* source
                = tile_data + local_y * source_stride + local_x * canvas_bytes_per_pixel;

            // Compute the pointer to the lower left corner of the rectangle that falls into this
            // tile, and stride per row (buffer, dest).
            mi::Difference dest_stride = width * buffer_bytes_per_pixel + buffer_padding;
            mi::Uint8* dest = buffer;
            if( buffer_topdown) {
                dest += (height - 1) * dest_stride;
                dest_stride = -dest_stride;
            }
            if( y > canvas_y)
                dest += (y - canvas_y) * dest_stride;
            if( x > canvas_x)
                dest += (x - canvas_x) * buffer_bytes_per_pixel;

            // Copy pixel data for rectangle that falls into this tile
            convert( source, dest, m_canvas_pixel_type, buffer_pixel_type, local_width,
                local_height, source_stride, dest_stride);
        }
    }

    return true;
}

bool Edit_canvas::write_rect(
    const mi::Uint8* buffer,
    bool buffer_topdown,
    Pixel_type buffer_pixel_type,
    mi::Uint32 canvas_x,
    mi::Uint32 canvas_y,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 buffer_padding,
    mi::Uint32 canvas_layer)
{
    if( !buffer || !m_canvas)
        return false;

    if( canvas_layer >= m_nr_of_layers)
        return false;

    if( canvas_x + width > m_canvas_width || canvas_y + height > m_canvas_height)
        return false;

    if( m_canvas_pixel_type == PT_UNDEF)
        return false;

    if( !exists_pixel_conversion( m_canvas_pixel_type, buffer_pixel_type))
        return false;

    mi::Uint32 canvas_bytes_per_pixel = get_bytes_per_pixel( m_canvas_pixel_type);
    mi::Uint32 buffer_bytes_per_pixel = get_bytes_per_pixel( buffer_pixel_type);

    // Compute canvas_x/canvas_y rounded down to multiples of m_tile_width/m_tile_height
    mi::Uint32 canvas_x_rd = (canvas_x / m_tile_width ) * m_tile_width;
    mi::Uint32 canvas_y_rd = (canvas_y / m_tile_height) * m_tile_height;

    // Loop over the affected tiles, (x,y) is the lower left corner of the tile
    for( mi::Uint32 y = canvas_y_rd; y < canvas_y + height; y += m_tile_height) {
        for( mi::Uint32 x = canvas_x_rd; x < canvas_x + width; x += m_tile_width) {

            // Compute height of rectangle that falls into this tile
            mi::Uint32 local_height = m_tile_height;
            if( y < canvas_y)
                local_height -= canvas_y - y;
            if( y + m_tile_height > canvas_y + height)
                local_height -= y + m_tile_height - (canvas_y + height);

            // Compute width of rectangle that falls into this tile
            mi::Uint32 local_width = m_tile_width;
            if( x < canvas_x)
                local_width -= canvas_x - x;
            if( x + m_tile_width > canvas_x + width)
                local_width -= x + m_tile_width - (canvas_x + width);

            // Compute the pointer to the lower left corner of the rectangle that falls into this
            // tile, and stride per row (canvas, dest).
            mi::Difference dest_stride = m_tile_width * canvas_bytes_per_pixel;
            mi::base::Handle<mi::neuraylib::ITile> tile( m_canvas->get_tile( x, y, canvas_layer));
            mi::Uint8* tile_data = static_cast<mi::Uint8*>( tile->get_data());
            mi::Uint32 local_x = std::max( canvas_x, x) % m_tile_width;
            mi::Uint32 local_y = std::max( canvas_y, y) % m_tile_height;
            mi::Uint8* dest
                = tile_data + local_y * dest_stride + local_x * canvas_bytes_per_pixel;

            // Compute the pointer to the lower left corner of the rectangle that falls into this
            // tile, and stride per row (buffer, source).
            mi::Difference source_stride = width * buffer_bytes_per_pixel + buffer_padding;
            const mi::Uint8* source = buffer;
            if( buffer_topdown) {
                source += (height - 1) * source_stride;
                source_stride = -source_stride;
            }
            if( y > canvas_y)
                source += (y - canvas_y) * source_stride;
            if( x > canvas_x)
                source += (x - canvas_x) * buffer_bytes_per_pixel;

            // Copy pixel data for rectangle that falls into this tile
            convert( source, dest, buffer_pixel_type, m_canvas_pixel_type, local_width,
                local_height, source_stride, dest_stride);
        }
    }

    return true;
}

bool Edit_canvas::lookup(
    mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) const
{
    if( x >= m_canvas_width || y >= m_canvas_height || z >= m_nr_of_layers)
        return false;

    const mi::Uint32 tile_x  = x / m_tile_width;
    const mi::Uint32 local_x = x % m_tile_width;
    const mi::Uint32 tile_y  = y / m_tile_height;
    const mi::Uint32 local_y = y % m_tile_height;
    const mi::Size   tile_index = (z * m_nr_of_tiles_y + tile_y) * m_nr_of_tiles_x + tile_x;
    ASSERT( M_NEURAY_API, tile_index < m_nr_of_tiles);

    mi::base::Lock::Block block;
    if( !m_lockless) {
        block.set( &m_tiles_lock);
        if( !m_tiles[tile_index])
            m_tiles[tile_index] = m_canvas->get_tile( x, y, z);
    }

    m_tiles[tile_index]->get_pixel( local_x, local_y, &color.r);
    return true;
}

bool Edit_canvas::store(
    const mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z)
{
    if( x >= m_canvas_width || y >= m_canvas_height || z >= m_nr_of_layers)
        return false;

    const mi::Uint32 tile_x  = x / m_tile_width;
    const mi::Uint32 local_x = x % m_tile_width;
    const mi::Uint32 tile_y  = y / m_tile_height;
    const mi::Uint32 local_y = y % m_tile_height;
    const mi::Size   tile_index = (z * m_nr_of_tiles_y + tile_y) * m_nr_of_tiles_x + tile_x;
    ASSERT( M_NEURAY_API, tile_index < m_nr_of_tiles);

    mi::base::Lock::Block block;
    if( !m_lockless) {
        block.set( &m_tiles_lock);
        if( !m_tiles[tile_index])
            m_tiles[tile_index] = m_canvas->get_tile( x, y, z);
    }

    m_tiles[tile_index]->set_pixel( local_x, local_y, &color.r);
    return true;
}

} // namespace IMAGE

} // namespace MI
