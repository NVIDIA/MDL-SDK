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

#ifndef SHADERS_PLUGIN_DDS_DDS_IMAGE_FILE_READER_IMPL_H
#define SHADERS_PLUGIN_DDS_DDS_IMAGE_FILE_READER_IMPL_H

#include <mi/base.h>
#include <mi/neuraylib/iimage_plugin.h>

#include "dds_image.h"
#include "dds_types.h"

#include <io/image/image/i_image_utilities.h>

namespace MI {

namespace DDS {

class Image_file_reader_impl : public mi::base::Interface_implement<mi::neuraylib::IImage_file>
{
public:
    /// Constructs an image file that imports from the given reader
    Image_file_reader_impl( mi::neuraylib::IReader* reader);

    /// Destructor
    ~Image_file_reader_impl();

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

    bool read(
        mi::neuraylib::ITile* tile,
        mi::Uint32 x,
        mi::Uint32 y,
        mi::Uint32 z,
        mi::Uint32 level) const;

    /// Does nothing and returns always \false.
    bool write(
        const mi::neuraylib::ITile* tile,
        mi::Uint32 x,
        mi::Uint32 y,
        mi::Uint32 z,
        mi::Uint32 level);

private:

    /// The reader used to import the image.
    mi::neuraylib::IReader* m_reader;

    /// The DDS header.
    Header m_header;

    /// The DDS image.
    mutable Image m_image;

    /// The pixel type (decoded from the header).
    IMAGE::Pixel_type m_pixel_type;
};

} // namespace DDS

} // namespace MI

#endif // SHADERS_PLUGIN_DDS_DDS_IMAGE_FILE_READER_IMPL_H
