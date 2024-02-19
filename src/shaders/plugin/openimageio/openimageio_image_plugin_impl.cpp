/***************************************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <mi/neuraylib/iplugin_api.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iwriter.h>

#include "openimageio_image_plugin_impl.h"
#include "openimageio_image_file_reader_impl.h"
#include "openimageio_image_file_writer_impl.h"
#include "openimageio_utilities.h"

#include <cassert>
#include <memory>

#include <base/system/version/i_version.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filesystem.h>

namespace MI {

namespace MI_OIIO {

Image_plugin_impl::Image_plugin_impl( const char* name, const char* oiio_format)
  : m_name( name), m_oiio_format( oiio_format)
{
}

bool Image_plugin_impl::init( mi::neuraylib::IPlugin_api* plugin_api)
{
    assert( plugin_api);

    m_image_api = plugin_api->get_api_component<mi::neuraylib::IImage_api>();

    // Check whether the OIIO installation actually supports this format. Otherwise, fall back to
    // a dummy plugin without supported extensions and pixel types, and skip the log message.
    if( !OIIO::is_imageio_format_name( m_oiio_format)) {
       m_extensions.clear();
       m_pixel_types.clear();
       return true;
    }

    mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_configuration(
        plugin_api->get_api_component<mi::neuraylib::ILogging_configuration>());
    if( logging_configuration)
        g_logger = logging_configuration->get_forwarding_logger();

    std::string message = "Plugin \"";
    message += m_name;
    message += "\" (build " + std::string( VERSION::get_platform_version());
    message += ", " + std::string( VERSION::get_platform_date());
    message += ") initialized";
    log( mi::base::MESSAGE_SEVERITY_INFO, message.c_str());

    // Explicit extensions since we map the OIIO name "jpeg2000" to two different plugins.
    if( m_name == "oiio_jp2")
        m_extensions = { "jp2" };
    else if( m_name == "oiio_j2k")
        m_extensions = { "j2k" };
    // "tga" is missing the extension targa
    else if( m_name == "oiio_tga")
        m_extensions = { "tga", "tpic", "targa" };
    else {
        // Use extensions as reported by OIIO.
        const auto& all_extensions = OIIO::get_extension_map();
        auto it = all_extensions.find( m_oiio_format);
        assert( it != all_extensions.end());
        m_extensions = it->second;
    }

    using namespace IMAGE;

    // Limiting the supported pixel types for export here has the consequence that necessary
    // conversions happen already on our side (which takes care of "force_default_gamma") and not
    // on the OpenImageIO side.
    if( m_name == "oiio_bmp")
        m_pixel_types = { PT_RGBA, PT_RGB };
    else if( m_name == "oiio_exr")
        m_pixel_types = { PT_COLOR, PT_RGB_FP, PT_FLOAT32 };
    else if( m_name == "oiio_gif")
        m_pixel_types = { PT_RGB };
    else if( m_name == "oiio_hdr")
        m_pixel_types = { PT_RGB_FP };
    else if( m_name == "oiio_jpg")
        m_pixel_types = { PT_RGB };
    else if( m_name == "oiio_jp2")
        m_pixel_types = { PT_RGB };
    else if( m_name == "oiio_j2k")
        m_pixel_types = { PT_RGB };
    else if( m_name == "oiio_png")
        m_pixel_types = { PT_RGBA, PT_RGB, PT_RGBA_16, PT_RGB_16 };
    else if( m_name == "oiio_pnm")
        m_pixel_types = { PT_RGB_16, PT_RGB, PT_RGB_FP, PT_FLOAT32 };
    else if( m_name == "oiio_psd")
        m_pixel_types = { };
    else if( m_name == "oiio_tga")
        m_pixel_types = { PT_RGBA, PT_RGB };
    else if( m_name == "oiio_tif")
        m_pixel_types = {
            PT_RGBA, PT_RGB, PT_RGBA_16, PT_RGB_16, PT_COLOR, PT_RGB_FP, PT_FLOAT32, PT_SINT8,
            PT_SINT32 };
    else if( m_name == "oiio_webp")
        m_pixel_types = { PT_RGBA, PT_RGB };
    else
        assert( false);

    // Disable OIIO internal thread pool since it causes shutdown problems in larger integrations.
    // Unfortunately, setting the global property first creates a default thread pool with N
    // threads, and then destroys them again right away. Still better than creating them implicitly
    // later, possibly without destroying them before the plugin is unloaded.
    bool result = OIIO::attribute( "threads", 1);
    assert( result);

    // Disable EXR internal thread pool via the global property. Calling OIIO::ImageInput::threads()
    // is too late since EXR threads might already be created in OIIO::ImageInput::open().
    result = OIIO::attribute( "exr_threads", -1);
    assert( result);
    (void) result;

    // Explicitly request the C++ API from OpenEXR. The C API lacks support for luminance-chroma
    // images. (The default depends on the OpenEXR version and build time flags for OIIO.)
    result = OIIO::attribute( "openexr:core", 0);
    assert( result);
    (void) result;

#ifndef NDEBUG
    // Disable printing of uncaught errors in release builds.
    OIIO::attribute( "oiio:print_uncaught_errors", 0);
    OIIO::attribute( "imagebuf:print_uncaught_errors", 0);
#endif

    return true;
}

bool Image_plugin_impl::exit( mi::neuraylib::IPlugin_api* plugin_api)
{
    m_image_api.reset();

    g_logger.reset();
    return true;
}

const char* Image_plugin_impl::get_file_extension( mi::Uint32 index) const
{
    if( index >= m_extensions.size())
        return nullptr;
    return m_extensions[index].c_str();
}

const char* Image_plugin_impl::get_supported_type( mi::Uint32 index) const
{
    if( index >= m_pixel_types.size())
        return nullptr;
    return IMAGE::convert_pixel_type_enum_to_string( m_pixel_types[index]);
}

mi::neuraylib::Impexp_priority Image_plugin_impl::get_priority() const
{
    return mi::neuraylib::IMPEXP_PRIORITY_WELL_DEFINED;
}

bool Image_plugin_impl::test( mi::neuraylib::IReader* reader) const
{
    if( !reader)
        return false;
    if( !reader->supports_absolute_access())
        return false;

    // Avoid buffering the entire file just for this method (see Image_file_reader_impl
    // constructor). Also alleviates the need to enable the header-only mode for libjpeg (see
    // Image_file_reader_impl::setup_image_input()).
    if( m_name == "oiio_jpg")
        return true;

    auto io_proxy = std::unique_ptr<OIIO::Filesystem::IOProxy>(
        create_input_proxy( reader, /*use_buffer*/ false, /*buffer*/ nullptr));
    std::string ext = m_oiio_format;

    OIIO::ImageSpec config;
    config["oiio:UnassociatedAlpha"] = 1;

    // Workaround for issue #3273: Set proxy also via config.
    OIIO::Filesystem::IOProxy* io_proxy_ptr = io_proxy.get();
    config.attribute( "oiio:ioproxy", OIIO::TypeDesc::PTR, &io_proxy_ptr);

    auto image_input = std::unique_ptr<OIIO::ImageInput>(
        OIIO::ImageInput::open( ext, &config, io_proxy.get()));
    if( !image_input) {
        // Consume any error messages.
        OIIO::geterror();
        return false;
    }

    return true;
}

mi::neuraylib::IImage_file* Image_plugin_impl::open_for_reading(
    mi::neuraylib::IReader* reader, const char* selector) const
{
    if( !reader)
        return nullptr;

    if( !reader->supports_absolute_access())
        return nullptr;

    Image_file_reader_impl* result = new Image_file_reader_impl(
        m_oiio_format,
        m_name,
        m_image_api.get(),
        reader,
        selector);
    if( !result->is_valid()) {
        delete result;
        return nullptr;
    }

    return result;
}

mi::neuraylib::IImage_file* Image_plugin_impl::open_for_writing(
    mi::neuraylib::IWriter* writer,
    const char* pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Uint32 nr_of_layers,
    mi::Uint32 miplevels,
    bool is_cubemap,
    mi::Float32 gamma,
    mi::Uint32 quality) const
{
   if( !writer || !pixel_type)
        return nullptr;

    if( !writer->supports_absolute_access())
        return nullptr;

    if( !supports_export())
        return nullptr;

    // Invalid canvas properties
    if( resolution_x == 0 || resolution_y == 0 || nr_of_layers == 0 || miplevels == 0
        || gamma <= 0.0f)
        return nullptr;

    // Features not implemented
    if( nr_of_layers != 1 || miplevels != 1 || is_cubemap)
        return nullptr;

    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    Image_file_writer_impl* result = new Image_file_writer_impl(
        m_oiio_format,
        m_name,
        m_image_api.get(),
        writer,
        pixel_type_enum,
        resolution_x,
        resolution_y,
        nr_of_layers,
        gamma,
        quality);
    if( !result->is_valid()) {
        delete result;
        return nullptr;
    }

    return result;
}

bool Image_plugin_impl::supports_export() const
{
    return !m_pixel_types.empty();
}

/// Describes a plugin in g_plugin_list below.
class Plugin_description
{
public:
    /// \param name          Our plugin name.
    /// \param oiio_format   The OIIO format name.
    Plugin_description( const char* name, const char* oiio_format)
      : m_name( name), m_oiio_format( oiio_format) { }
    const char* m_name;
    const char* m_oiio_format;
};

/// The list of of all plugins in this DSO.
/// - dds: support is limited to very few subformats, not usable without proper test()
///   implementation, our dedicated DDS plugin is much better
/// - dpx: not relevant for us
/// - jpeg2000: exposed as two different plugins since the plugin name is passed as pseudo-
///   extension and is used to select the compressor
Plugin_description g_plugin_list[] = {
        Plugin_description( "oiio_bmp",    "bmp"      ),
        // Plugin_description( "oiio_dds", "dds"      ),
        // Plugin_description( "oiio_dpx", "dpx"      ),
        Plugin_description( "oiio_exr",    "openexr"  ),
        Plugin_description( "oiio_gif",    "gif"      ),
        Plugin_description( "oiio_hdr",    "hdr"      ),
        Plugin_description( "oiio_jpg",    "jpeg"     ),
        Plugin_description( "oiio_jp2",    "jpeg2000" ),
        Plugin_description( "oiio_j2k",    "jpeg2000" ),
        Plugin_description( "oiio_png",    "png"      ),
        Plugin_description( "oiio_pnm",    "pnm"      ),
        Plugin_description( "oiio_psd",    "psd"      ),
        Plugin_description( "oiio_tga",    "targa"    ),
        Plugin_description( "oiio_tif",    "tiff"     ),
        Plugin_description( "oiio_webp",   "webp"     ),
};

/// Factory to create an instance of Image_plugin_impl.
extern "C"
MI_DLL_EXPORT
mi::base::Plugin* mi_plugin_factory(
    mi::Sint32 index,         // index of the plugin
    void* context)            // context given to the library, ignore
{
    if( static_cast<size_t>( index) >= sizeof( g_plugin_list) / sizeof( g_plugin_list[0]))
        return nullptr;
    return new Image_plugin_impl( g_plugin_list[index].m_name, g_plugin_list[index].m_oiio_format);
}

} // namespace MI_OIIO

} // namespace MI
