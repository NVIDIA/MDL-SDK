/***************************************************************************************************
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/iwriter.h>

#include <cassert>
#include <cstring>
#include <string>

// #define DUMP_PIXEL_X 25
// #define DUMP_PIXEL_Y 1

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
#include <mi/math/color.h>
#include <iostream>
#endif

namespace MI {

namespace MI_OIIO {

Image_file_writer_impl::Image_file_writer_impl(
    const std::string& oiio_format,
    const std::string& plugin_name,
    mi::neuraylib::IImage_api* image_api,
    mi::neuraylib::IWriter* writer,
    IMAGE::Pixel_type pixel_type,
    mi::Uint32 resolution_x,
    mi::Uint32 resolution_y,
    mi::Uint32 resolution_z,
    mi::Float32 gamma,
    mi::Uint32 quality)
  : m_oiio_format( oiio_format),
    m_plugin_name( plugin_name),
    m_image_api( image_api, mi::base::DUP_INTERFACE),
    m_writer( writer, mi::base::DUP_INTERFACE),
    m_pixel_type( pixel_type),
    m_resolution_x( resolution_x),
    m_resolution_y( resolution_y),
    m_resolution_z( resolution_z),
    m_gamma( gamma),
    m_quality( quality),
    m_pass_unassociated_alpha( false)
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
        // Can this happen at all?
        assert( false);
        return;
    }

    OIIO::ImageSpec image_spec(
        get_image_spec( m_pixel_type, m_resolution_x, m_resolution_y, 1));

    // Do not enable unassociated alpha for "oiio_tif" since imf_diff reports differences. Unclear
    // on which side the error is.
    if(    plugin_name == "oiio_png"
    //  || plugin_name == "oiio_tif"
        || plugin_name == "oiio_tga"
        || plugin_name == "oiio_jp2"
        || plugin_name == "oiio_j2k") {
        m_pass_unassociated_alpha = true;
        image_spec["oiio:UnassociatedAlpha"] = 1;
    }
    
    // It is unclear whether BMP uses associated or unassociated alpha. We use unassociated alpha
    // for historic reasons, in contrast to OIIO. Doing so without support from the OIIO handler
    // for BMP is a misuse of the OIIO API.
    if( plugin_name == "oiio_bmp")
        m_pass_unassociated_alpha = true;

    if( m_plugin_name == "oiio_exr") {
        image_spec.format = m_quality <= 50 ? OIIO::TypeDesc::HALF : OIIO::TypeDesc::FLOAT;
    } else if( m_plugin_name == "oiio_jpg") {
        std::ostringstream s;
        s << "jpeg:" << m_quality;
        image_spec["Compression"] = s.str();
    } else if( m_plugin_name == "oiio_png") {
        image_spec["png:compressionLevel"] = 5;
    } else if( m_plugin_name == "oiio_tif") {
        image_spec["compression"] = "lzw";
    }

    // Disable OIIO internal thread pool since it causes shutdown problems in larger integrations.
    m_image_output->threads( 1);

    m_image_output->open( ext, image_spec);
}

Image_file_writer_impl::~Image_file_writer_impl()
{
    m_image_output->close();

    mi::Sint64 count = m_writer->write(
        reinterpret_cast<const char*>( &m_buffer[0]), m_buffer.size());
    assert( static_cast<size_t>( count) ==  m_buffer.size());
    (void) count;
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
        // OIIO documentation mentions that alpha and depth should be assumed linear.
        tile2 = associate_alpha( m_image_api.get(), tile, 1.0f);
    }
    tile = nullptr; // prevent accidental misuse

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
    tile2->get_pixel( DUMP_PIXEL_X, DUMP_PIXEL_Y, &c.r);
    std::cout << "OIIO plugin writer (after associate_alpha()): "
              << c.r << " " << c.g << " " << c.b << " " << c.a << std::endl;
#endif

    const mi::Uint8* data = static_cast<const mi::Uint8*>( tile2->get_data());
    data += (m_resolution_y - 1) * static_cast<size_t>( bytes_per_row);

    // Use write_scanlines() instead of write_image(). The latter does not allow to specify a
    // single layer.
    bool success = m_image_output->write_scanlines(
        /*ybegin*/ 0,
        /*yend*/ m_resolution_y,
        /*z*/ z,
        format,
        data,
        /*xstride*/ OIIO::AutoStride,
        /*ystride*/ -bytes_per_row);
    return success;
}

bool Image_file_writer_impl::is_valid() const
{
    return !!m_image_output;
}

} // namespace MI_OIIO

} // namespace MI
