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

#include "i_image_access_mipmap.h"
#include "i_image_access_canvas.h"
#include "i_image_mipmap.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>

#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace IMAGE {

Access_mipmap::Access_mipmap( const IMipmap* mipmap)
{
    set( mipmap);
}

Access_mipmap::Access_mipmap( const Access_mipmap& rhs)
{
    mi::base::Lock::Block block( &rhs.m_access_canvases_lock);
    set( rhs.m_mipmap.get());
}

Access_mipmap& Access_mipmap::operator=( const Access_mipmap& rhs)
{
    if( this == &rhs)
        return *this;

    mi::base::Lock::Block block( &rhs.m_access_canvases_lock);
    set( rhs.m_mipmap.get());
    return *this;
}

void Access_mipmap::set( const IMipmap* mipmap)
{
    mi::base::Lock::Block block( &m_access_canvases_lock);

    m_mipmap = make_handle_dup( mipmap);
    if( !m_mipmap) {
        m_miplevels      = 0;
        m_access_canvases.clear();
        m_resolution_x_0 = 0;
        m_resolution_y_0 = 0;
        m_nr_of_layers   = 0;
    } else {
        m_miplevels      = m_mipmap->get_nlevels();
        m_access_canvases.clear(); // clear previously cached elements
        m_access_canvases.resize( m_miplevels);
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas_0( mipmap->get_level( 0));
        m_access_canvases[0].set( canvas_0.get());
        m_resolution_x_0 = canvas_0->get_resolution_x();
        m_resolution_y_0 = canvas_0->get_resolution_y();
        m_nr_of_layers   = canvas_0->get_layers_size();
    }
}

const IMipmap* Access_mipmap::get() const
{
    if( !m_mipmap)
        return 0;
    m_mipmap->retain();
    return m_mipmap.get();
}

mi::Uint32 Access_mipmap::get_resolution_x( mi::Uint32 miplevel) const
{
    return miplevel >= m_miplevels ? 0 : m_resolution_x_0 >> miplevel;
}

mi::Uint32 Access_mipmap::get_resolution_y( mi::Uint32 miplevel) const
{
    return miplevel >= m_miplevels ? 0 : m_resolution_y_0 >> miplevel;
}

bool Access_mipmap::read_rect(
    mi::Uint8* buffer,
    bool buffer_topdown,
    Pixel_type buffer_pixel_type,
    mi::Uint32 miplevel_x,
    mi::Uint32 miplevel_y,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 buffer_padding,
    mi::Uint32 miplevel_layer,
    mi::Uint32 miplevel) const
{
    if( !m_mipmap)
        return false;

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( m_mipmap->get_level( miplevel));
    if( !canvas.is_valid_interface())
        return false;

    Access_canvas access_canvas( canvas.get());
    return access_canvas.read_rect(
        buffer, buffer_topdown, buffer_pixel_type, miplevel_x, miplevel_y, width, height,
        buffer_padding, miplevel_layer);
}

bool Access_mipmap::lookup(
    mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 miplevel) const
{
    if( !m_mipmap || miplevel >= m_miplevels)
        return false;

    mi::base::Lock::Block block( &m_access_canvases_lock);

    if( !m_access_canvases[miplevel].is_valid()) {
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas( m_mipmap->get_level( miplevel));
        m_access_canvases[miplevel].set( canvas.get());
    }
    return m_access_canvases[miplevel].lookup( color, x, y, z);
}

Edit_mipmap::Edit_mipmap( IMipmap* mipmap)
{
    set( mipmap);
}

Edit_mipmap::Edit_mipmap( const Edit_mipmap& rhs)
{
    set( rhs.m_mipmap.get());
}

Edit_mipmap& Edit_mipmap::operator=( const Edit_mipmap& rhs)
{
    if( this == &rhs)
        return *this;

    set( rhs.m_mipmap.get());
    return *this;
}

void Edit_mipmap::set( IMipmap* mipmap)
{
    m_mipmap = make_handle_dup( mipmap);
    if( !m_mipmap) {
        m_miplevels      = 0;
        m_edit_canvas_0.set( 0);
        m_resolution_x_0 = 0;
        m_resolution_y_0 = 0;
        m_nr_of_layers         = 0;
    } else {
        m_miplevels      = m_mipmap->get_nlevels();
        mi::base::Handle<mi::neuraylib::ICanvas> canvas_0( mipmap->get_level( 0));
        m_edit_canvas_0.set( canvas_0.get());
        m_resolution_x_0 = canvas_0->get_resolution_x();
        m_resolution_y_0 = canvas_0->get_resolution_y();
        m_nr_of_layers   = canvas_0->get_layers_size();
    }
}

IMipmap* Edit_mipmap::get() const
{
    if( !m_mipmap)
        return 0;
    m_mipmap->retain();
    return m_mipmap.get();
}

mi::Uint32 Edit_mipmap::get_resolution_x( mi::Uint32 miplevel) const
{
    return miplevel >= m_miplevels ? 0 : m_resolution_x_0 >> miplevel;
}

mi::Uint32 Edit_mipmap::get_resolution_y( mi::Uint32 miplevel) const
{
    return miplevel >= m_miplevels ? 0 : m_resolution_y_0 >> miplevel;
}
bool Edit_mipmap::read_rect(
    mi::Uint8* buffer,
    bool buffer_topdown,
    Pixel_type buffer_pixel_type,
    mi::Uint32 miplevel_x,
    mi::Uint32 miplevel_y,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 buffer_padding,
    mi::Uint32 miplevel_layer,
    mi::Uint32 miplevel) const
{
    if( !m_mipmap)
        return false;

    // Use the cached Edit_canvas for level 0
    if( miplevel == 0)
        m_edit_canvas_0.read_rect(
            buffer, buffer_topdown, buffer_pixel_type, miplevel_x, miplevel_y, width, height,
            buffer_padding, miplevel_layer);

    // Get the actual canvas for higher miplevels
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( m_mipmap->get_level( miplevel));
    if( !canvas.is_valid_interface())
        return false;

    Access_canvas access_canvas( canvas.get());
    return access_canvas.read_rect(
        buffer, buffer_topdown, buffer_pixel_type, miplevel_x, miplevel_y, width, height,
        buffer_padding, miplevel_layer);
}

bool Edit_mipmap::write_rect(
    const mi::Uint8* buffer,
    bool buffer_topdown,
    Pixel_type buffer_pixel_type,
    mi::Uint32 miplevel_x,
    mi::Uint32 miplevel_y,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 buffer_padding,
    mi::Uint32 miplevel_layer,
    mi::Uint32 miplevel)
{
    if( !m_mipmap)
        return false;

    // Use the cached Edit_canvas for level 0
    if( miplevel == 0)
        return m_edit_canvas_0.write_rect(
            buffer, buffer_topdown, buffer_pixel_type, miplevel_x, miplevel_y, width, height,
            buffer_padding, miplevel_layer);

    // Get the actual canvas for higher miplevels
    mi::base::Handle<mi::neuraylib::ICanvas> canvas( m_mipmap->get_level( miplevel));
    if( !canvas.is_valid_interface())
        return false;

    Edit_canvas edit_canvas( canvas.get());
    return edit_canvas.write_rect(
        buffer, buffer_topdown, buffer_pixel_type, miplevel_x, miplevel_y, width, height,
        buffer_padding, miplevel_layer);
}

bool Edit_mipmap::lookup(
    mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 miplevel) const
{
    if( !m_mipmap)
        return false;

    // Use the cached Edit_canvas for level 0
    if( miplevel == 0)
        return m_edit_canvas_0.lookup( color, x, y, z);

    // Get the actual canvas for higher miplevels
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( m_mipmap->get_level( miplevel));
    if( !canvas.is_valid_interface())
        return false;

    mi::base::Handle<const mi::neuraylib::ITile> tile( canvas->get_tile( x, y, z));
    if( !tile.is_valid_interface())
        return false;

    mi::Uint32 local_x = x % tile->get_resolution_x();
    mi::Uint32 local_y = y % tile->get_resolution_y();
    tile->get_pixel( local_x, local_y, &color.r);
    return true;
}

bool Edit_mipmap::store(
    const mi::math::Color& color, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z, mi::Uint32 miplevel)
{
    if( !m_mipmap)
        return false;

    // Use the cached Edit_canvas for level 0
    if( miplevel == 0)
        return m_edit_canvas_0.store( color, x, y, z);

    // Get the actual canvas for higher miplevels
    mi::base::Handle<mi::neuraylib::ICanvas> canvas( m_mipmap->get_level( miplevel));
    if( !canvas.is_valid_interface())
        return false;

    mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( x, y, z));
    if( !tile.is_valid_interface())
        return false;

    mi::Uint32 local_x = x % tile->get_resolution_x();
    mi::Uint32 local_y = y % tile->get_resolution_y();
    tile->set_pixel( local_x, local_y, &color.r);
    return true;
}

} // namespace IMAGE

} // namespace MI
