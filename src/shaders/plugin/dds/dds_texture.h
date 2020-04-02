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

#ifndef IO_IMAGE_DDS_DDS_TEXTURE_H
#define IO_IMAGE_DDS_DDS_TEXTURE_H

#include "dds_surface.h"

#include <mi/base/types.h>

#include <cassert>
#include <vector>

namespace MI {

namespace DDS {

/// A texture (in DDS terms) in comparable to a mipmap in neuray terms.
class Texture
{
public:
    /// Creates an (invalid) texture with no surfaces.
    ///
    /// A texture becomes valid by adding at least one surface.
    Texture() { }

    /// Destroys the texture.
    ///
    /// The texture is in the same state as after default construction.
    void clear() { m_surfaces.clear(); }

    /// Returns a surface (const).
    ///
    /// \param level   The requested surface.
    /// \return        The surface of level \p level.
    const Surface& get_surface( mi::Size level) const
    {
        assert( level < m_surfaces.size());
        return m_surfaces[level];
    }

    /// Returns a surface (mutable).
    ///
    /// \param level   The requested surface.
    /// \return        The surface of level \p level.
    Surface& get_surface( mi::Size level)
    {
        assert( level < m_surfaces.size());
        return m_surfaces[level];
    }

    /// Adds a surface.
    ///
    /// \param surface   Adds a copy of this surface as last miplevel.
    void add_surface( const Surface& surface) { m_surfaces.push_back( surface); }

    /// Returns the number of surfaces.
    mi::Size get_num_surfaces() const { return m_surfaces.size(); }

private:

    /// The surfaces of this mipmap.
    std::vector<Surface> m_surfaces;
};

} // namespace DDS

} // namespace MI

#endif
