/***************************************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_PLUGIN_IMPL_H
#define SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_PLUGIN_IMPL_H

#include <mi/base/handle.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/iimage_plugin.h>

#include <string>
#include <vector>

#include <io/image/image/i_image_utilities.h>

namespace MI {

namespace MI_OIIO {

// The plugin has the following limitations:
// - 3d textures and cubemaps are not supported.
// - Higher miplevels are ignored.
// - Per-channel pixel types are not supported.
// - Deep pixels are not supported.
// - Metadata is mostly ignored, in particular the "oiio:ColorSpace" attribute.
class Image_plugin_impl : public mi::neuraylib::IImage_plugin
{
public:
    /// Constructor.
    ///
    /// \param name          Name of the plugin (start with \c "oiio_").
    /// \param oiio_format   Corresponding OIIO format.
    Image_plugin_impl( const char* name, const char* oiio_format);

    virtual ~Image_plugin_impl() { }

    // methods of mi::base::Plugin

    const char* get_name() const override { return m_name.c_str(); }

    const char* get_type() const override { return MI_NEURAY_IMAGE_PLUGIN_TYPE; }

    mi::Sint32 get_version() const override { return 1; }

    const char* get_compiler() const override { return "unknown"; }

    void release() override { delete this; }

    // methods of mi::neuraylib::IImage_plugin

    bool init( mi::neuraylib::IPlugin_api* plugin_api) override;

    bool exit( mi::neuraylib::IPlugin_api* plugin_api) override;

    const char* get_file_extension( mi::Uint32 index) const override;

    const char* get_supported_type( mi::Uint32 index) const override;

    mi::neuraylib::Impexp_priority get_priority() const override;

    bool supports_selectors() const override { return true; }

    bool test( mi::neuraylib::IReader* reader) const override;

    mi::neuraylib::IImage_file* open_for_reading(
        mi::neuraylib::IReader* reader, const char* selector) const override;

    mi::neuraylib::IImage_file* open_for_writing(
        mi::neuraylib::IWriter* writer,
        const char* pixel_type,
        mi::Uint32 resolution_x,
        mi::Uint32 resolution_y,
        mi::Uint32 nr_of_layers,
        mi::Uint32 miplevels,
        bool is_cubemap,
        mi::Float32 gamma,
        const mi::IMap* export_options) const override;

private:

    /// Indicates whether this plugin supports exporting (files and streams) with at least one of
    /// our pixel types
    bool supports_export() const;

    /// The name of this plugin.
    std::string m_name;

    /// The OIIO format handled by this plugin.
    std::string m_oiio_format;

    /// The file extensions supported by this plugin.
    std::vector<std::string> m_extensions;

    /// The pixel types supported by this plugin for export.
    std::vector<IMAGE::Pixel_type> m_pixel_types;

    /// API component IImage_api.
    mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;
};

} // namespace MI_OIIO

} // namespace MI

#endif // SHADERS_PLUGIN_OPENIMAGEIO_OPENIMAGEIO_IMAGE_PLUGIN_IMPL_H
