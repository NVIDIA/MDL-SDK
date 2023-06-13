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

#ifndef SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_FILE_READER_IMPL_H
#define SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_FILE_READER_IMPL_H

#include <mi/base.h>
#include <mi/neuraylib/iimage_plugin.h>

#include <memory>
#include <string>
#include <vector>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filesystem.h>

#include <io/image/image/i_image_utilities.h>

namespace mi { namespace neuraylib { class IImage_api; } }

namespace MI {

namespace MI_OIIO {

class Image_file_reader_impl : public mi::base::Interface_implement<mi::neuraylib::IImage_file>
{
public:
    /// Constructor.
    Image_file_reader_impl(
        const std::string& oiio_format,
        const std::string& plugin_name,
        mi::neuraylib::IImage_api* image_api,
        mi::neuraylib::IReader* reader,
        const char* selector);

    // methods of mi::neuraylib::IImage_file

    const char* get_type() const override;

    mi::Uint32 get_resolution_x( mi::Uint32 level) const override;

    mi::Uint32 get_resolution_y( mi::Uint32 level) const override;

    mi::Uint32 get_layers_size( mi::Uint32 level) const override;

    mi::Uint32 get_miplevels() const override;

    bool get_is_cubemap() const override;

    mi::Float32 get_gamma() const override;

    mi::neuraylib::ITile* read( mi::Uint32 z, mi::Uint32 level) const override;

    /// Does nothing and returns always \false.
    bool write( const mi::neuraylib::ITile* tile, mi::Uint32 z, mi::Uint32 level) override;

    // internal methods

    /// Indicates whether the constructor succeeded.
    ///
    /// This method needs to be called before any other method. If it returns \c false, no other
    /// method must be called, and the instance should be destroyed right away.
    bool is_valid() const;

private:
    /// Sets up \c m_image_input.
    ///
    /// Modifies \c m_image_input and \c m_nv_header_only. Only const because it is used from read()
    /// (and the constructor).
    bool setup_image_input( bool from_constructor) const;

    /// The OIIO format handled by this plugin.
    std::string m_oiio_format;

    /// The plugin name (for error messages).
    std::string m_plugin_name;

    /// API component IImage_api.
    mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;

    /// The reader used to import the image.
    mi::base::Handle<mi::neuraylib::IReader> m_reader;

     /// Buffer that holds the entire content of the reader.
    std::vector<char> m_buffer;

    /// I/O proxy for m_buffer
    std::unique_ptr<OIIO::Filesystem::IOProxy> m_io_proxy;

    /// Input object
    mutable std::unique_ptr<OIIO::ImageInput> m_image_input;

    /// Indicates whether the "nv:header_only" attribute was set.
    mutable bool m_nv_header_only = false;

    /// \name Various properties derived from the image input and the selector.
    //@{

    /// The subimage index (usually 0, but the selector might affect this).
    mi::Uint32 m_subimage = 0;

    /// Resolution of the subimage in x-direction.
    mi::Uint32 m_resolution_x = 1;

    /// Resolution of the subimage in y-direction.
    mi::Uint32 m_resolution_y = 1;

    /// Resolution of the subimage in z-direction.
    mi::Uint32 m_resolution_z = 1;

    /// The pixel type of the subimage (after applying the selector).
    IMAGE::Pixel_type m_pixel_type = IMAGE::PT_UNDEF;

    /// The channel names (for layers: after stripping the selector prefix plus dot).
    std::vector<std::string> m_channel_names;

    /// The first channel index to import.
    mi::Sint32 m_channel_start = -1;

    /// The last channel index+1 to import.
    mi::Sint32 m_channel_end = -1;

    //@}
};

} // namespace MI_OIIO

} // namespace MI

#endif // SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_FILE_READER_IMPL_H
