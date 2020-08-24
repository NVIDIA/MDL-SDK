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

#ifndef IO_IMAGE_IMAGE_IMAGE_TILE_IMPL_H
#define IO_IMAGE_IMAGE_IMAGE_TILE_IMPL_H

#include <mi/neuraylib/itile.h>
#include <mi/base/interface_implement.h>

#include "i_image_utilities.h"

#include <boost/core/noncopyable.hpp>

#include <vector>


namespace MI {

namespace IMAGE {

mi::neuraylib::ITile* create_tile( Pixel_type pixel_type, mi::Uint32 width, mi::Uint32 height);

/// IMAGE::ITile is an interface derived from mi::neuraylib::ITile.
///
/// It adds one single method to compute the memory usage of the tile. Always use the public
/// interface, unless you really need this special method.
class ITile : public
    mi::base::Interface_declare<0x61d64832,0x4b8b,0x4c39,0xbd,0x79,0x4f,0x46,0x9a,0x50,0x56,0x03,
                                mi::neuraylib::ITile>
{
public:
    /// Returns the memory used by this element in bytes, including all substructures.
    ///
    /// Used to implement DB::Element_base::get_size() for DBIMAGE::Image.
    virtual mi::Size get_size() const = 0;
};

/// A simple implementation of the ITile interface.
///
/// Note that only a fixed set of types is permitted for the template parameter T.
/// Hence we use explicit template instantiation in the corresponding .cpp file.
template <Pixel_type T>
class Tile_impl
  : public mi::base::Interface_implement<ITile>,
    public boost::noncopyable
{
public:
    /// Constructor.
    ///
    /// Creates a tile of the given width and height.
    Tile_impl( mi::Uint32 width, mi::Uint32 height);

    // methods of mi::neuraylib::ITile

    void set_pixel( mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats);

    void get_pixel( mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const;

    const char* get_type() const;

    mi::Uint32 get_resolution_x() const { return m_width; }

    mi::Uint32 get_resolution_y() const { return m_height; }

    const void* get_data() const { return m_data.data(); }

    void* get_data() { return m_data.data(); }

    // own methods

    /// Returns the memory used by this element in bytes, including all substructures.
    ///
    /// Used to implement DB::Element_base::get_size() for DBIMAGE::Image.
    mi::Size get_size() const;

private:

    /// Number of components per pixel.
    static const int s_components_per_pixel = Pixel_type_traits<T>::s_components_per_pixel;

    /// Width of the tile
    mi::Uint32 m_width;
    /// Height of the tile
    mi::Uint32 m_height;
    /// The data of this tile
    std::vector<typename Pixel_type_traits<T>::Base_type> m_data;
};

} // namespace IMAGE

} // namespace MI

#endif // MI_IO_IMAGE_IMAGE_IMAGE_TILE_IMPL_H
