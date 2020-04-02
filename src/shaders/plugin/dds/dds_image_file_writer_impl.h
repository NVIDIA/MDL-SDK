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

#ifndef SHADERS_PLUGIN_DDS_DDS_IMAGE_FILE_WRITER_IMPL_H
#define SHADERS_PLUGIN_DDS_DDS_IMAGE_FILE_WRITER_IMPL_H

#include <mi/base.h>
#include <mi/neuraylib/iimage_plugin.h>

#include "dds_texture.h"

#include <vector>
#include <io/image/image/i_image_utilities.h>

namespace MI {

namespace DDS {

class Image_file_writer_impl : public mi::base::Interface_implement<mi::neuraylib::IImage_file>
{
public:
    /// Constructor.
    Image_file_writer_impl(
        mi::neuraylib::IWriter* writer,
        const char* pixel_type,
        mi::Uint32 resolution_x,
        mi::Uint32 resolution_y,
        mi::Uint32 nr_of_layers,
        mi::Uint32 miplevels,
        bool is_cubemap,
        mi::Float32 gamma);

    /// Destructor.
    ~Image_file_writer_impl();

    // methods of mi::neuraylib::IImage_file

    const char* get_type() const;

    mi::Uint32 get_resolution_x( mi::Uint32 level) const;

    mi::Uint32 get_resolution_y( mi::Uint32 level) const;

    mi::Uint32 get_layers_size( mi::Uint32 level) const;

    mi::Uint32 get_tile_resolution_x( mi::Uint32 level) const;

    mi::Uint32 get_tile_resolution_y( mi::Uint32 level) const;

    mi::Uint32 get_miplevels() const;

    bool get_is_cubemap() const;

    mi::Float32 get_gamma() const;

    /// Does nothing and returns always \false.
    bool read(
        mi::neuraylib::ITile* tile,
        mi::Uint32 x,
        mi::Uint32 y,
        mi::Uint32 z,
        mi::Uint32 level) const;

    bool write(
        const mi::neuraylib::ITile* tile,
        mi::Uint32 x,
        mi::Uint32 y,
        mi::Uint32 z,
        mi::Uint32 level);

private:

    /// The writer used to export the image.
    mi::neuraylib::IWriter* m_writer;

    /// The width of the image.
    mi::Uint32 m_resolution_x;

    /// The height of the image.
    mi::Uint32 m_resolution_y;

    /// The depth of the image.
    mi::Uint32 m_nr_of_layers;

    /// The number of miplevels of the image.
    mi::Uint32 m_miplevels;

    /// Flag for cubemaps.
    bool m_is_cubemap;

    /// The gamma value of the image.
    mi::Float32 m_gamma;

    /// The pixel type;
    IMAGE::Pixel_type m_pixel_type;

    /// Buffer for tile data
    std::vector<std::vector<mi::Uint8> > m_level;
};

} // namespace DDS

} // namespace MI

#endif // SHADERS_PLUGIN_DDS_DDS_IMAGE_FILE_WRITER_IMPL_H
