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

#include "pch.h"

#include <mi/base.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <mi/neuraylib/iplugin_api.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iwriter.h>

#include "freeimage_image_plugin_impl.h"
#include "freeimage_image_file_reader_impl.h"
#include "freeimage_image_file_writer_impl.h"
#include "freeimage_utilities.h"

#include <cassert>
#include <base/system/version/i_version.h>

namespace MI {

namespace FREEIMAGE {

static mi::base::Atom32 g_freeimage_libray_initialization_counter = 0;

Image_plugin_impl::Image_plugin_impl( const char* name, FREE_IMAGE_FORMAT format)
  : m_name( name), m_format( format)
{
}

bool Image_plugin_impl::init( mi::neuraylib::IPlugin_api* plugin_api)
{
    if( plugin_api) {
        mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_configuration(
            plugin_api->get_api_component<mi::neuraylib::ILogging_configuration>());
        g_logger = logging_configuration->get_forwarding_logger();
    }

    std::string message = "Plugin \"";
    message += m_name;
    message += "\" (build " + std::string( VERSION::get_platform_version());
    message += ", " + std::string( VERSION::get_platform_date());
    message += ") initialized";
    log( mi::base::MESSAGE_SEVERITY_INFO, message.c_str());

    if( ++g_freeimage_libray_initialization_counter == 1)
        FreeImage_Initialise();

    if( FreeImage_FIFSupportsExportType( m_format, FIT_BITMAP)) {
        // Disable pixel type "Rgba" for export for "jpeg". The format claims to support exports
        // with 32 bpp, but this works only for CMYK color types.
        if( FreeImage_FIFSupportsExportBPP( m_format, 32) && m_format != FIF_JPEG)
            m_supported_types.push_back( "Rgba");
        if( FreeImage_FIFSupportsExportBPP( m_format, 24))
            m_supported_types.push_back( "Rgb");
    }

    if( FreeImage_FIFSupportsExportType( m_format, FIT_RGBA16))
        m_supported_types.push_back( convert_freeimage_pixel_type_to_neuray_pixel_type(FIT_RGBA16));
    if( FreeImage_FIFSupportsExportType( m_format, FIT_RGB16))
        m_supported_types.push_back( convert_freeimage_pixel_type_to_neuray_pixel_type( FIT_RGB16));
    // Disable pixel types "Color" and "Rgb_fp" for export for "psd". Re-import looks good, but
    // looks broken in Gimp 2.10.8, and fails with Adobe Photoshop. See
    // https://sourceforge.net/p/freeimage/bugs/302/ .
    if( FreeImage_FIFSupportsExportType( m_format, FIT_RGBAF) && m_format != FIF_PSD)
        m_supported_types.push_back( convert_freeimage_pixel_type_to_neuray_pixel_type( FIT_RGBAF));
    if( FreeImage_FIFSupportsExportType( m_format, FIT_RGBF) && m_format != FIF_PSD)
        m_supported_types.push_back( convert_freeimage_pixel_type_to_neuray_pixel_type( FIT_RGBF));
    // Disable pixel type "Float32" for export for "psd". Re-import fails, looks broken in Gimp
    // 2.10.8, and fails with Adobe Photoshop. See https://sourceforge.net/p/freeimage/bugs/301/ .
    if( FreeImage_FIFSupportsExportType( m_format, FIT_FLOAT) && m_format != FIF_PSD)
        m_supported_types.push_back( convert_freeimage_pixel_type_to_neuray_pixel_type( FIT_FLOAT));
    if( FreeImage_FIFSupportsExportType( m_format, FIT_INT32))
        m_supported_types.push_back( convert_freeimage_pixel_type_to_neuray_pixel_type( FIT_INT32));

    // Disable export for "jp2" (does not work for unknown reasons, see bug 12505).
    if( m_format == FIF_JP2)
        m_supported_types.clear();

    // Disable export for "jxr" (apparently only available on Windows, re-import fails (at least on
    // Linux), import with Gimp 2.10.8 fails). See https://sourceforge.net/p/freeimage/bugs/294/ .
    if( m_format == FIF_JXR)
        m_supported_types.clear();
    

    return true;
}

bool Image_plugin_impl::exit( mi::neuraylib::IPlugin_api* plugin_api)
{
    if( --g_freeimage_libray_initialization_counter == 0)
        FreeImage_DeInitialise();

    g_logger = 0;
    return true;
}

const char* Image_plugin_impl::get_file_extension( mi::Uint32 index) const
{
    // Taken from the appendix of the FreeImage documentation.
    switch( m_format) {
        case FIF_IFF:
            if( index == 0) return "iff";
            if( index == 1) return "lbm";
            return 0;
        case FIF_J2K:
            if( index == 0) return "j2c";
            if( index == 1) return "j2k";
            return 0;
        case FIF_JPEG:
            if( index == 0) return "jpe";
            if( index == 1) return "jpeg";
            if( index == 2) return "jpg";
            if( index == 3) return "jif";
            return 0;
        case FIF_JXR:
            if( index == 0) return "jxr";
            if( index == 1) return "wdp";
            if( index == 2) return "htp";
            return 0;
        case FIF_PICT:
            if( index == 0) return "pct";
            if( index == 1) return "pic";
            if( index == 2) return "pict";
            return 0;
        case FIF_TARGA:
            if( index == 0) return "targa";
            if( index == 1) return "tga";
            return 0;
        case FIF_TIFF:
            if( index == 0) return "tif";
            if( index == 1) return "tiff";
            return 0;
        case FIF_WBMP:
            if( index == 0) return "wap";
            if( index == 1) return "wbmp";
            if( index == 2) return "wbm";
            return 0;
        default:
            if( index == 0) return m_name.c_str() + 3;
            return 0;
    }

    assert( false); //-V779 PVS
    return 0;
}

const char* Image_plugin_impl::get_supported_type( mi::Uint32 index) const
{
    if( index >= m_supported_types.size())
        return 0;
    return m_supported_types[index];
}

bool Image_plugin_impl::test( const mi::Uint8* buffer, mi::Uint32 file_size) const
{
    FIMEMORY* fimemory = FreeImage_OpenMemory(
        const_cast<mi::Uint8*>( buffer), std::min( file_size, mi::Uint32( 512)));
    if( !fimemory)
        return false;

    FREE_IMAGE_FORMAT format = FreeImage_GetFileTypeFromMemory( fimemory);
    FreeImage_CloseMemory( fimemory);
    return format == m_format;
}

mi::neuraylib::Impexp_priority Image_plugin_impl::get_priority() const
{
    // FreeImage cannot import all DDS files, e.g., files with multiple layers or levels.
    if( m_format == FIF_DDS)
        return mi::neuraylib::IMPEXP_PRIORITY_AMBIGUOUS;

    return mi::neuraylib::IMPEXP_PRIORITY_WELL_DEFINED;
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
        return 0;

    if( !writer->supports_absolute_access())
        return 0;

    if( !supports_export())
        return 0;

    // Invalid canvas properties
    if( resolution_x == 0 || resolution_y == 0 || nr_of_layers == 0 || miplevels == 0
        || gamma <= 0.0f)
        return 0;

    // FreeImage cannot handle any of these features
    if( nr_of_layers != 1 || miplevels != 1 || is_cubemap)
        return 0;

    return new Image_file_writer_impl(
        writer, pixel_type, resolution_x, resolution_y, gamma, quality, m_format, m_name);
}

mi::neuraylib::IImage_file* Image_plugin_impl::open_for_reading(
    mi::neuraylib::IReader* reader) const
{
    if( !reader)
        return 0;

    if( !reader->supports_absolute_access())
        return 0;

    return new Image_file_reader_impl( reader, m_format);
}

bool Image_plugin_impl::supports_export() const
{
    return get_supported_type( 0) != 0;
}

/// Describes a plugin in g_plugin_list below.
class Plugin_description
{
public:
    Plugin_description( const char* name, FREE_IMAGE_FORMAT format)
      : m_name( name), m_format( format) { }
    const char* m_name;
    FREE_IMAGE_FORMAT m_format;
};

/// The list of of all plugins in this DSO.
///
/// - "mng", "pbmraw", "pgm", "pgmraw", "ppm", "ppmraw" are disabled because an
///   import of a previously exported image does not work
/// - "raw" is disabled because it does not seem to be useful in our context
/// - "xbm" is disabled because it only supports 2-bit pixel types
///
Plugin_description g_plugin_list[] = {
        Plugin_description( "fi_bmp"   , FIF_BMP   ),
        Plugin_description( "fi_cut"   , FIF_CUT   ),
        Plugin_description( "fi_dds"   , FIF_DDS   ),
        Plugin_description( "fi_exr"   , FIF_EXR   ),
        Plugin_description( "fi_faxg3" , FIF_FAXG3 ),
        Plugin_description( "fi_gif"   , FIF_GIF   ),
        Plugin_description( "fi_hdr"   , FIF_HDR   ),
        Plugin_description( "fi_ico"   , FIF_ICO   ),
        Plugin_description( "fi_iff"   , FIF_IFF   ),
        Plugin_description( "fi_j2k"   , FIF_J2K   ),
        Plugin_description( "fi_jng"   , FIF_JNG   ),
        Plugin_description( "fi_jp2"   , FIF_JP2   ),
        Plugin_description( "fi_jpg"   , FIF_JPEG  ),
        Plugin_description( "fi_jxr"   , FIF_JXR   ),
        Plugin_description( "fi_koala" , FIF_KOALA ),
        Plugin_description( "fi_lbm"   , FIF_LBM   ),
//      Plugin_description( "fi_mng"   , FIF_MNG   ),
        Plugin_description( "fi_pbm"   , FIF_PBM   ),
//      Plugin_description( "fi_pbmraw", FIF_PBMRAW),
        Plugin_description( "fi_pcd"   , FIF_PCD   ),
//      Plugin_description( "fi_pcx"   , FIF_PCX   ), see bug 18655
        Plugin_description( "fi_pfm"   , FIF_PFM   ),
//      Plugin_description( "fi_pgm"   , FIF_PGM   ),
//      Plugin_description( "fi_pgmraw", FIF_PGMRAW),
        Plugin_description( "fi_pict"  , FIF_PICT  ),
        Plugin_description( "fi_png"   , FIF_PNG   ),
//      Plugin_description( "fi_ppm"   , FIF_PPM   ),
//      Plugin_description( "fi_ppmraw", FIF_PPMRAW),
        Plugin_description( "fi_psd"   , FIF_PSD   ),
        Plugin_description( "fi_ras"   , FIF_RAS   ),
//      Plugin_description( "fi_raw"   , FIF_RAW   ),
        Plugin_description( "fi_sgi"   , FIF_SGI   ),
        Plugin_description( "fi_targa" , FIF_TARGA ),
        Plugin_description( "fi_tiff"  , FIF_TIFF  ),
        Plugin_description( "fi_wbmp"  , FIF_WBMP  ),
        Plugin_description( "fi_webp"  , FIF_WEBP  ),
//      Plugin_description( "fi_xbm"   , FIF_XBM   ),
        Plugin_description( "fi_xpm"   , FIF_XPM   )
};

/// Factory to create an instance of Image_plugin_impl.
extern "C"
MI_DLL_EXPORT
mi::base::Plugin* mi_plugin_factory(
    mi::Sint32 index,         // index of the plugin
    void* context)            // context given to the library, ignore
{
    if( static_cast<size_t>( index) >= sizeof( g_plugin_list) / sizeof( g_plugin_list[0]))
        return 0;
    return new Image_plugin_impl( g_plugin_list[index].m_name, g_plugin_list[index].m_format);
}

} // namespace FREEIMAGE

} // namespace MI
