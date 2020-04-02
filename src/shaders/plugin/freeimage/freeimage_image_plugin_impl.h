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

#ifndef SHADERS_PLUGIN_FREEIMAGE_FREEIMAGE_IMAGE_PLUGIN_IMPL_H
#define SHADERS_PLUGIN_FREEIMAGE_FREEIMAGE_IMAGE_PLUGIN_IMPL_H

#include <mi/neuraylib/iimage_plugin.h>

#include <string>
#include <vector>

#include <FreeImage.h>

// FreeImage vs neuray pixel types
// ===============================
//
// FreeImage    neuray          bits per pixel
//
// FIT_BITMAP   "Rgb", "Rgba"   24/32 bpp (1)
// FIT_UINT16   -               16 bpp (2)
// FIT_INT16    -               16 bpp (2)
// FIT_UINT32   -               32 bpp (2)
// FIT_INT32    "Sint32"        32 bpp
// FIT_FLOAT    "Float32"       32 bpp
// FIT_DOUBLE   -               64 bpp
// FIT_COMPLEX  -              128 bpp
// FIT_RGB16    "Rgb_16"        48 bpp
// FIT_RGBA16   "Rgba_16"       64 bpp
// FIT_RGBF     "Rgb_fp"        96 bpp
// FIT_RGBAF    "Color"        128 bpp
// -            "Sint8"          8 bpp (1)
// -            "Float32<2>"    64 bpp
// -            "Float32<3>"    96 bpp
// -            "Float32<4>"   128 bpp
//
// (1) Note that FIT_BITMAP supports also bitmaps with 1, 4, 8, and 16 bits per pixel, but for
// 1 to 8 bits per pixel the bitmaps are palletized, i.e., FIT_BITMAP with 8 bits per pixel does
// *not* correspond to the neuray pixel type "Sint8". These pixel types are indirectly supported
// by an implicit conversion to FIT_RGBF.
//
// (2) Indirectly supported by converting the image to FIT_FLOAT first, and then to FIT_RGBF.

namespace MI {

namespace FREEIMAGE {

class Image_plugin_impl : public mi::neuraylib::IImage_plugin
{
public:
    /// Constructs an image plugin that handles images of a given name/format.
    Image_plugin_impl( const char* name, FREE_IMAGE_FORMAT format);

    virtual ~Image_plugin_impl() { }

    // methods of mi::base::Plugin

    const char* get_name() const { return m_name.c_str(); }

    const char* get_type() const { return MI_NEURAY_IMAGE_PLUGIN_TYPE; }

    mi::Sint32 get_version() const { return 1; }

    const char* get_compiler() const { return "unknown"; }

    void release() { delete this; }

    // methods of mi::neuraylib::IImage_plugin

    bool init( mi::neuraylib::IPlugin_api* plugin_api);

    bool exit( mi::neuraylib::IPlugin_api* plugin_api);

    const char* get_file_extension( mi::Uint32 index) const;

    const char* get_supported_type( mi::Uint32 index) const;

    bool test( const mi::Uint8* buffer, mi::Uint32 file_size) const;

    mi::neuraylib::Impexp_priority get_priority() const;

    mi::neuraylib::IImage_file* open_for_writing(
        mi::neuraylib::IWriter* writer,
        const char* pixel_type,
        mi::Uint32 resolution_x,
        mi::Uint32 resolution_y,
        mi::Uint32 nr_of_layers,
        mi::Uint32 miplevels,
        bool is_cubemap,
        mi::Float32 gamma,
        mi::Uint32 quality) const;

    mi::neuraylib::IImage_file* open_for_reading( mi::neuraylib::IReader* reader) const;

private:

    /// Indicates whether this plugin supports exporting (files and streams) with at least one of
    /// our pixel types
    bool supports_export() const;

    /// The name of this plugin.
    std::string m_name;

    /// The format handled by this plugin.
    FREE_IMAGE_FORMAT m_format;

    /// The pixel types supported by this plugin.
    std::vector<const char*> m_supported_types;
};

} // namespace FREEIMAGE

} // namespace MI

#endif // SHADERS_PLUGIN_FREEIMAGE_FREEIMAGE_IMAGE_PLUGIN_IMPL_H
