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

#ifndef IO_IMAGE_DDS_DDS_SURFACE_H
#define IO_IMAGE_DDS_DDS_SURFACE_H

#include <mi/base/types.h>

#include <cassert>
#include <cstring>
#include <vector>

namespace MI {

namespace DDS {

/// A surface (in DDS terms) in comparable to a canvas/miplevel in neuray terms.
class Surface
{
public:

    /// Creates an invalid surface (exists only because required(?) for std::vector).
    Surface() : m_width( 0), m_height( 0), m_depth( 0) { assert( false); }

    /// Creates a surface with given pixel data.
    ///
    /// \param w            width of the surface
    /// \param h            height of the surface
    /// \param d            depth of the surface
    /// \param pixel_size   size of the pixel data
    /// \param pixels       the pixel data of the surface (will be copied if not \c NULL)
    Surface(
        mi::Uint32 w,
        mi::Uint32 h,
        mi::Uint32 d,
        mi::Uint32 pixel_size,
        const mi::Uint8* pixels)
    {
        assert( w > 0);
        assert( h > 0);
        assert( d > 0);
        assert( pixel_size > 0);

        m_width  = w;
        m_height = h;
        m_depth  = d;
        m_pixels.resize( pixel_size);
        if( pixels)
            memcpy( &m_pixels[0], pixels, pixel_size);
    }

    /// Destroys the surface.
    ///
    /// The surface is in the same state as after default construction.
    void clear()
    {
        m_width  = 0;
        m_height = 0;
        m_depth  = 0;
        m_pixels.clear();
    }

    /// Returns the pixels (read-only).
    const mi::Uint8* get_pixels() const { return &m_pixels[0]; };

    /// Returns the pixels (mutable).
    mi::Uint8* get_pixels() { return &m_pixels[0]; };

    /// Returns the width of the surface.
    mi::Uint32 get_width() const { return m_width; }

    /// Returns the height of the surface.
    mi::Uint32 get_height() const { return m_height; }

    /// Returns the depth of the surface.
    mi::Uint32 get_depth() const { return m_depth; }

    /// Returns the size of the pixel data of the surface (possibly compressed).
    mi::Uint32 get_size() const { return m_pixels.size(); };

    /// Returns the number of elements of the surface (i.e. width * height * depth).
    mi::Uint32 get_num_elements() const { return m_width * m_height * m_depth; }

private:
    /// Width of the surface
    mi::Uint32 m_width;
    /// Height of the surface
    mi::Uint32 m_height;
    /// Depth of the surface
    mi::Uint32 m_depth;
    /// The pixels of the surface
    std::vector<mi::Uint8> m_pixels;
};

} // namespace DDS

} // namespace MI

#endif // IO_IMAGE_DDS_DDS_SURFACE_H
