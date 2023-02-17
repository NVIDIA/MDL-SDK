/***************************************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_FILE_WRITER_IMPL_H
#define SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_FILE_WRITER_IMPL_H

#include <mi/base.h>
#include <mi/neuraylib/iimage_plugin.h>

#include <memory>
#include <string>
#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imageio.h>

#include <io/image/image/i_image_utilities.h>

namespace mi { namespace neuraylib { class IImage_api; } }

namespace MI {

namespace MI_OIIO {

class Image_file_writer_impl : public mi::base::Interface_implement<mi::neuraylib::IImage_file>
{
public:
    /// Constructor.
    Image_file_writer_impl(
        const std::string& oiio_format,
        const std::string& plugin_name,
        mi::neuraylib::IImage_api* image_api,
        mi::neuraylib::IWriter* writer,
        IMAGE::Pixel_type pixel_type,
        mi::Uint32 resolution_x,
        mi::Uint32 resolution_y,
        mi::Uint32 resolution_z,
        mi::Float32 gamma,
        mi::Uint32 quality);

    /// Destructor.
    ~Image_file_writer_impl() override;

    // methods of mi::neuraylib::IImage_file

    const char* get_type() const override;

    mi::Uint32 get_resolution_x( mi::Uint32 level) const override;

    mi::Uint32 get_resolution_y( mi::Uint32 level) const override;

    mi::Uint32 get_layers_size( mi::Uint32 level) const override;

    mi::Uint32 get_miplevels() const override;

    bool get_is_cubemap() const override;

    mi::Float32 get_gamma() const override;

    /// Does nothing and returns always \nullptr.
    mi::neuraylib::ITile* read( mi::Uint32 z, mi::Uint32 level) const override;

    bool write( const mi::neuraylib::ITile* tile, mi::Uint32 z, mi::Uint32 level) override;

    // internal methods

    /// Indicates whether the constructor succeeded.
    ///
    /// This method needs to be called before any other method. If it returns \c false, no other
    /// method must be called, and the instance should be destroyed right away.
    bool is_valid() const;

private:

    /// The OIIO format handled by this plugin.
    std::string m_oiio_format;

    /// The plugin name (for error messages).
    std::string m_plugin_name;

    /// API component IImage_api.
    mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;

    /// The writer used to export the image.
    mi::base::Handle<mi::neuraylib::IWriter> m_writer;

    /// Buffer that holds the entire content later passed to the writer.
    std::vector<unsigned char> m_buffer;

    /// I/O proxy for m_buffer
    std::unique_ptr<OIIO::Filesystem::IOProxy> m_io_proxy;

    /// Output object
    std::unique_ptr<OIIO::ImageOutput> m_image_output;

    /// The pixel type of the image.
    IMAGE::Pixel_type m_pixel_type;

    /// Resolution of the image in x-direction.
    mi::Uint32 m_resolution_x;

    /// Resolution of the image in y-direction.
    mi::Uint32 m_resolution_y;

    /// Resolution of the image in z-direction.
    mi::Uint32 m_resolution_z;

    /// The gamma value of the image.
    mi::Float32 m_gamma;

    /// The requested image quality.
    mi::Uint32 m_quality;

    /// Indicates whether we pass unassociated alpha to the OIIO API for this format.
    bool m_pass_unassociated_alpha;
};

} // namespace MI_OIIO

} // namespace MI

#endif // SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_FILE_WRITER_IMPL_H
