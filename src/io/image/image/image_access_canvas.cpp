/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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
        m_tiles.clear();
    } else {
        m_nr_of_layers      = m_canvas->get_layers_size();
        m_canvas_width      = m_canvas->get_resolution_x();
        m_canvas_height     = m_canvas->get_resolution_y();
        m_canvas_pixel_type = convert_pixel_type_string_to_enum( m_canvas->get_type());
        m_tiles.clear(); // clear previously cached elements
        m_tiles.resize( m_nr_of_layers);
    }

    if( !m_lockless)
        return;

    // Prefetch all tiles in the lockless variant.
    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z) {
        m_tiles[z] = m_canvas->get_tile( z);
    }
}

const mi::neuraylib::ICanvas* Access_canvas::get() const
{
    mi::base::Lock::Block block;
    if( !m_lockless) block.set( &m_tiles_lock);

    if( !m_canvas)
        return nullptr;
    m_canvas->retain();
    return m_canvas.get();
}

bool Access_canvas::read_rect(
    mi::Uint8* buffer,
    const bool buffer_topdown,
    const Pixel_type buffer_pixel_type,
    const mi::Uint32 canvas_x,
    const mi::Uint32 canvas_y,
    const mi::Uint32 width,
    const mi::Uint32 height,
    const mi::Uint32 buffer_padding,
    const mi::Uint32 canvas_layer) const
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

    const mi::Uint32 canvas_bytes_per_pixel = get_bytes_per_pixel( m_canvas_pixel_type);
    const mi::Uint32 buffer_bytes_per_pixel = get_bytes_per_pixel( buffer_pixel_type);

    // Compute the pointer to the lower left corner of the rectangle that falls into this
    // tile, and stride per row (canvas, source).
    const mi::Difference source_stride
        = static_cast<mi::Difference>( m_canvas_width) * canvas_bytes_per_pixel;
    mi::base::Handle<const mi::neuraylib::ITile> tile( m_canvas->get_tile( canvas_layer));
    const mi::Uint8* tile_data = static_cast<const mi::Uint8*>( tile->get_data());
    const mi::Uint8* source = tile_data + canvas_y * source_stride
        + static_cast<mi::Difference>( canvas_x) * canvas_bytes_per_pixel;

    // Compute the pointer to the lower left corner of the rectangle that falls into this
    // tile, and stride per row (buffer, dest).
    mi::Difference dest_stride
        = static_cast<mi::Difference>( width) * buffer_bytes_per_pixel + buffer_padding;
    mi::Uint8* dest = buffer;
    if( buffer_topdown) {
        dest += (height - 1) * dest_stride;
        dest_stride = -dest_stride;
    }

    // Copy pixel data for rectangle that falls into this tile
    convert( source, dest, m_canvas_pixel_type, buffer_pixel_type, width, height,
             source_stride, dest_stride);

    return true;
}

bool Access_canvas::lookup(
    mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) const
{
    if( x >= m_canvas_width || y >= m_canvas_height || z >= m_nr_of_layers)
        return false;

    mi::base::Lock::Block block;
    if( !m_lockless) {
        block.set( &m_tiles_lock);
        if( !m_tiles[z])
            m_tiles[z] = m_canvas->get_tile( z);
    }

    m_tiles[z]->get_pixel( x, y, &color.r);
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
        m_tiles.clear();
    } else {
        m_nr_of_layers      = m_canvas->get_layers_size();
        m_canvas_width      = m_canvas->get_resolution_x();
        m_canvas_height     = m_canvas->get_resolution_y();
        m_canvas_pixel_type = convert_pixel_type_string_to_enum( m_canvas->get_type());
        m_tiles.clear(); // clear previously cached elements
        m_tiles.resize( m_nr_of_layers);
    }

    if( !m_lockless)
        return;

    // Prefetch all tiles in the lockless variant.
    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z) {
        m_tiles[z] = m_canvas->get_tile( z);
    }
}

mi::neuraylib::ICanvas* Edit_canvas::get() const
{
    mi::base::Lock::Block block;
    if( !m_lockless) block.set( &m_tiles_lock);

    if( !m_canvas)
        return nullptr;
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

    const mi::Uint32 canvas_bytes_per_pixel = get_bytes_per_pixel( m_canvas_pixel_type);
    const mi::Uint32 buffer_bytes_per_pixel = get_bytes_per_pixel( buffer_pixel_type);

    // Compute the pointer to the lower left corner of the rectangle that falls into this
    // tile, and stride per row (canvas, source).
    const mi::Difference source_stride
        = static_cast<mi::Difference>( m_canvas_width) * canvas_bytes_per_pixel;
    mi::base::Handle<const mi::neuraylib::ITile> tile( m_canvas->get_tile( canvas_layer));
    const mi::Uint8* const tile_data = static_cast<const mi::Uint8*>( tile->get_data());
    const mi::Uint8* const source = tile_data + canvas_y * source_stride
         + static_cast<mi::Difference>( canvas_x) * canvas_bytes_per_pixel;

    // Compute the pointer to the lower left corner of the rectangle that falls into this
    // tile, and stride per row (buffer, dest).
    mi::Difference dest_stride
        = static_cast<mi::Difference>( width) * buffer_bytes_per_pixel + buffer_padding;
    mi::Uint8* dest = buffer;
    if( buffer_topdown) {
        dest += (height - 1) * dest_stride;
        dest_stride = -dest_stride;
    }

    // Copy pixel data for rectangle that falls into this tile
    convert( source, dest, m_canvas_pixel_type, buffer_pixel_type, width, height,
             source_stride, dest_stride);

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

    const mi::Uint32 canvas_bytes_per_pixel = get_bytes_per_pixel( m_canvas_pixel_type);
    const mi::Uint32 buffer_bytes_per_pixel = get_bytes_per_pixel( buffer_pixel_type);

    // Compute the pointer to the lower left corner of the rectangle that falls into this
    // tile, and stride per row (canvas, dest).
    const mi::Difference dest_stride
        = static_cast<mi::Difference>( m_canvas_width) * canvas_bytes_per_pixel;
    mi::base::Handle<mi::neuraylib::ITile> tile( m_canvas->get_tile( canvas_layer));
    mi::Uint8* const tile_data = static_cast<mi::Uint8*>( tile->get_data());
    mi::Uint8* const dest = tile_data + canvas_y * dest_stride
        + static_cast<mi::Difference>( canvas_x) * canvas_bytes_per_pixel;

    // Compute the pointer to the lower left corner of the rectangle that falls into this
    // tile, and stride per row (buffer, source).
    mi::Difference source_stride
        = static_cast<mi::Difference>( width) * buffer_bytes_per_pixel + buffer_padding;
    const mi::Uint8* source = buffer;
    if( buffer_topdown) {
        source += (height - 1) * source_stride;
        source_stride = -source_stride;
    }

    // Copy pixel data for rectangle that falls into this tile
    convert( source, dest, buffer_pixel_type, m_canvas_pixel_type, width, height,
             source_stride, dest_stride);

    return true;
}

bool Edit_canvas::lookup(
    mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) const
{
    if( x >= m_canvas_width || y >= m_canvas_height || z >= m_nr_of_layers)
        return false;

    mi::base::Lock::Block block;
    if( !m_lockless) {
        block.set( &m_tiles_lock);
        if( !m_tiles[z])
            m_tiles[z] = m_canvas->get_tile( z);
    }

    m_tiles[z]->get_pixel( x, y, &color.r);
    return true;
}

bool Edit_canvas::store(
    const mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z)
{
    if( x >= m_canvas_width || y >= m_canvas_height || z >= m_nr_of_layers)
        return false;

    mi::base::Lock::Block block;
    if( !m_lockless) {
        block.set( &m_tiles_lock);
        if( !m_tiles[z])
            m_tiles[z] = m_canvas->get_tile( z);
    }

    m_tiles[z]->set_pixel( x, y, &color.r);
    return true;
}

} // namespace IMAGE

} // namespace MI
