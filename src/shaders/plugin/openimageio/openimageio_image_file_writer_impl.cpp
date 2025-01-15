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

#include "pch.h"

#include "openimageio_image_file_writer_impl.h"
#include "openimageio_utilities.h"

#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/iwriter.h>

#include <cassert>
#include <cstring>
#include <string>
#include <utility>

#ifndef OIIO_SUPPORT_EXR_CREATE_MULTIPART_FOR_ALPHA
#if OIIO_VERSION >= OIIO_MAKE_VERSION(2,5,12)
#define OIIO_SUPPORT_EXR_CREATE_MULTIPART_FOR_ALPHA 1
#else
#define OIIO_SUPPORT_EXR_CREATE_MULTIPART_FOR_ALPHA 0
#endif
#endif

// #define DUMP_PIXEL_X 25
// #define DUMP_PIXEL_Y 1

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
#include <mi/math/color.h>
#include <iostream>
#endif

namespace MI {

namespace MI_OIIO {

Image_file_writer_impl::Image_file_writer_impl(
    std::string oiio_format,
    const std::string& plugin_name,
    mi::neuraylib::IImage_api* image_api,
    mi::neuraylib::IWriter* writer,
    IMAGE::Pixel_type pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Uint32 resolution_z,
    mi::Float32 gamma,
    const mi::IMap* export_options)
  : m_oiio_format( std::move( oiio_format)),
    m_plugin_name( plugin_name),
    m_image_api( image_api, mi::base::DUP_INTERFACE),
    m_writer( writer, mi::base::DUP_INTERFACE),
    m_pixel_type( pixel_type),
    m_resolution_x( resolution_x),
    m_resolution_y( resolution_y),
    m_resolution_z( resolution_z),
    m_gamma( gamma)
{
    // Writing to a local buffer first is 1/3 faster than adapting IWriter to IOProxy (at least for
    // debug builds, no noticeable difference for release builds).
    m_io_proxy = std::make_unique<OIIO::Filesystem::IOVecOutput>( m_buffer);

    // Here the dot is relevant to distinguish the extension from the format. For some formats the
    // extension contains important information, e.g., "jp2" vs "j2k" for JPEG-2000. The dummy
    // filename itself is not used due to the IO proxy.
    std::string ext = std::string( "dummy.") + plugin_name.substr( 5);
    m_image_output = OIIO::ImageOutput::create( ext, m_io_proxy.get());
    if( !m_image_output) {
        std::string message = OIIO::geterror();
        log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
        return;
    }

    OIIO::ImageSpec image_spec = get_image_spec( m_pixel_type, m_resolution_x, m_resolution_y, 1);

    // Pass unassociated alpha for the image handlers that support it.
    if(    plugin_name == "oiio_png"
        || plugin_name == "oiio_tif"
        || plugin_name == "oiio_tga"
        || plugin_name == "oiio_jp2"
        || plugin_name == "oiio_j2k") {
        m_pass_unassociated_alpha = true;
        image_spec["oiio:UnassociatedAlpha"] = 1;
    }

    // It is unclear whether BMP uses associated or unassociated alpha. We use unassociated alpha
    // for historic reasons, in contrast to OIIO. Doing so without support from the OIIO handler
    // for BMP is a misuse of the OIIO API. (There is no point in setting image_spec[...] as above
    // since the BMP handler does not support it.)
    if( plugin_name == "oiio_bmp")
        m_pass_unassociated_alpha = true;

    if( m_plugin_name == "oiio_exr") {

        const char* key = "exr:data_type";
        if( export_options && export_options->has_key( key)) {

            mi::base::Handle value( export_options->get_value<mi::IString>( key));
            if( !value) {
                std::string message = std::string( "Invalid type for option \"") + key + "\".";
                log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
                m_image_output.reset();
                return;
            }

            std::string s = value->get_c_str();
            if( s == "Float16")
                image_spec.format = OIIO::TypeDesc::HALF;
            else if( s == "Float32")
                image_spec.format = OIIO::TypeDesc::FLOAT;
            else {
                std::string message = std::string( "Invalid value \"")
                    + s + "\" for option \"" + key + "\".";
                log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
                m_image_output.reset();
                return;
            }
        }

#if OIIO_SUPPORT_EXR_CREATE_MULTIPART_FOR_ALPHA
        // Handle this option as last EXR option.
        key = "exr:create_multipart_for_alpha";
        if( export_options && export_options->has_key( key)) {

            mi::base::Handle value( export_options->get_value<mi::IBoolean>( key));
            if( !value) {
                std::string message = std::string( "Invalid type for option \"") + key + "\".";
                log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
                m_image_output.reset();
                return;
            }

            if( value->get_value<bool>() && IMAGE::has_alpha( m_pixel_type)) {
                m_exr_create_multipart_for_alpha = true;
                m_pass_unassociated_alpha = true;
                m_exr_multipart_image_spec[0] = image_spec;
                m_exr_multipart_image_spec[0].nchannels -= 1;
                m_exr_multipart_image_spec[0].channelnames.resize(
                    m_exr_multipart_image_spec[0].nchannels);
                m_exr_multipart_image_spec[0]["oiio:subimagename"] = "rgb";
                m_exr_multipart_image_spec[0].alpha_channel = -1;
                m_exr_multipart_image_spec[1] = image_spec;
                m_exr_multipart_image_spec[1].nchannels = 1;
                m_exr_multipart_image_spec[1].channelnames.resize(
                    m_exr_multipart_image_spec[1].nchannels);
                m_exr_multipart_image_spec[1].channelnames[0]
                    = image_spec.channelnames[image_spec.alpha_channel];
                m_exr_multipart_image_spec[1].alpha_channel = 0;
                m_exr_multipart_image_spec[1]["oiio:subimagename"] = "alpha";
            }

        }
#endif // OIIO_SUPPORT_EXR_CREATE_MULTIPART_FOR_ALPHA

    } else if( m_plugin_name == "oiio_jpg") {

        const char* key = "jpg:quality";
        if( export_options && export_options->has_key( key)) {

            mi::base::Handle value( export_options->get_value<mi::IUint32>( key));
            if( !value) {
                std::string message = std::string( "Invalid type for option \"") + key + "\".";
                log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
                m_image_output.reset();
                return;
            }

            mi::Uint32 v = value->get_value<mi::Uint32>();
            if( v <= 100) {
                std::ostringstream s;
                s << "jpeg:" << v;
                image_spec["Compression"] = s.str();
            } else {
                std::string message = std::string( "Invalid value \"")
                    + std::to_string( v) + "\" for option \"" + key + "\".";
                log( mi::base::MESSAGE_SEVERITY_ERROR, message.c_str());
                m_image_output.reset();
                return;
            }

        } else {
            image_spec["Compression"] = "jpeg:100";
        }

    } else if( m_plugin_name == "oiio_png") {
        image_spec["png:compressionLevel"] = 5;
    } else if( m_plugin_name == "oiio_tif") {
        image_spec["compression"] = "lzw";
    }

    if( m_exr_create_multipart_for_alpha) {
        m_image_output->open( ext, 2, m_exr_multipart_image_spec);
    } else
        m_image_output->open( ext, image_spec);
}

Image_file_writer_impl::~Image_file_writer_impl()
{
    if( m_image_output)
        m_image_output->close();

    if( !m_buffer.empty()) {
        mi::Sint64 count = m_writer->write(
            reinterpret_cast<const char*>( &m_buffer[0]), m_buffer.size());
        assert( static_cast<size_t>( count) ==  m_buffer.size());
        (void) count;
    }
}

const char* Image_file_writer_impl::get_type() const
{
    return IMAGE::convert_pixel_type_enum_to_string( m_pixel_type);
}

mi::Uint32 Image_file_writer_impl::get_resolution_x( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_x;
}

mi::Uint32 Image_file_writer_impl::get_resolution_y( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_y;
}

mi::Uint32 Image_file_writer_impl::get_layers_size( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_z;
}

mi::Uint32 Image_file_writer_impl::get_miplevels() const
{
    return 1;
}

bool Image_file_writer_impl::get_is_cubemap() const
{
    return false;
}

mi::Float32 Image_file_writer_impl::get_gamma() const
{
    return m_gamma;
}

mi::neuraylib::ITile* Image_file_writer_impl::read(
    mi::Uint32 z, mi::Uint32 level) const
{
    assert( false);
    return nullptr;
}

bool Image_file_writer_impl::write(
    const mi::neuraylib::ITile* tile, mi::Uint32 z, mi::Uint32 level)
{
    if( level != 0) {
        assert( false);
        return false;
    }

    int cpp = IMAGE::get_components_per_pixel( m_pixel_type);
    int bpc = IMAGE::get_bytes_per_component( m_pixel_type);
    int bytes_per_pixel = cpp * bpc;
    int bytes_per_row = m_resolution_x * cpp * bpc;

    OIIO::TypeDesc format( get_base_type( m_pixel_type));

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
    mi::math::Color c;
    tile->get_pixel( DUMP_PIXEL_X, DUMP_PIXEL_Y, &c.r);
    std::cout << "OIIO plugin writer (before associate_alpha()): "
              << c.r << " " << c.g << " " << c.b << " " << c.a << std::endl;
#endif

    mi::base::Handle<const mi::neuraylib::ITile> tile2;
    if( m_pass_unassociated_alpha) {
        // Avoid redundant conversions and resulting quantization errors.
        tile2 = make_handle_dup( tile);
#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
        std::cout << "OIIO plugin writer (skip associate_alpha()) " << std::endl;
#endif
    } else {
        // Use 1.0f instead of m_gamma. Unclear what is correct here. OIIO itself uses the actual
        // gamma value, whereas FreeImage and GIMP seem to ignore gamma for this and always assume
        // gamma == 1.0f.
        tile2 = associate_alpha( m_image_api.get(), tile, 1.0f);
    }
    tile = nullptr; // prevent accidental misuse

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
    tile2->get_pixel( DUMP_PIXEL_X, DUMP_PIXEL_Y, &c.r);
    std::cout << "OIIO plugin writer (after associate_alpha()): "
              << c.r << " " << c.g << " " << c.b << " " << c.a << std::endl;
#endif

    const auto* data = static_cast<const mi::Uint8*>( tile2->get_data());
    data += (m_resolution_y - 1) * static_cast<size_t>( bytes_per_row);

    try {
        // Use write_scanlines() instead of write_image(). The latter does not allow to specify
        // a single layer.
        bool success = m_image_output->write_scanlines(
            /*ybegin*/ 0,
            /*yend*/ m_resolution_y,
            /*z*/ z,
            format,
            data,
            /*xstride*/ bytes_per_pixel,
            /*ystride*/ -bytes_per_row);
        if( !success)
            return false;
    } catch( const std::bad_alloc&) {
        return false;
    }

    if( m_exr_create_multipart_for_alpha) {

        std::string ext = std::string( "dummy.") + m_plugin_name.substr( 5);
        bool success = m_image_output->open(
            ext, m_exr_multipart_image_spec[1], OIIO::ImageOutput::AppendSubimage);
        if( !success)
            return false;

        try {
            // Use write_scanlines() instead of write_image(). The latter does not allow to specify
            // a single layer.
            success = m_image_output->write_scanlines(
                /*ybegin*/ 0,
                /*yend*/ m_resolution_y,
                /*z*/ z,
                format,
                data + 3 * bpc,
                /*xstride*/ bytes_per_pixel,
                /*ystride*/ -bytes_per_row);
            if( !success)
                return false;
        } catch( const std::bad_alloc&) {
            return false;
        }

    }

    return true;
}

bool Image_file_writer_impl::is_valid() const
{
    return !!m_image_output;
}

} // namespace MI_OIIO

} // namespace MI
