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

#include "pch.h"

#include "openimageio_image_file_reader_impl.h"
#include "openimageio_utilities.h"

#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/itile.h>

#include <cassert>
#include <cstring>

#include <io/image/image/i_image_utilities.h>

// #define DUMP_PIXEL_X 25
// #define DUMP_PIXEL_Y 1

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
#include <mi/math/color.h>
#include <iostream>
#endif

namespace MI {

namespace MI_OIIO {

Image_file_reader_impl::Image_file_reader_impl(
    const std::string& oiio_format,
    const std::string& plugin_name,
    mi::neuraylib::IImage_api* image_api,
    mi::neuraylib::IReader* reader)
  : m_oiio_format( oiio_format),
    m_plugin_name( plugin_name),
    m_image_api( image_api, mi::base::DUP_INTERFACE),
    m_reader( reader, mi::base::DUP_INTERFACE),
    m_resolution_x( 1),
    m_resolution_y( 1),
    m_resolution_z( 1),
    m_pixel_type( IMAGE::PT_UNDEF),
    m_nv_header_only( false)
{
    // Some plugins do not accept a generic IOProxy, but require a IOMemReader (using m_buffer).
    // bool use_buffer = plugin_name == "oiio_jpg";
    // Reading into a local buffer first is 5-10% faster than adapting IReader to IOProxy (at least
    // for debug builds, no noticeable difference for release builds).
    constexpr bool use_buffer = true;
    m_io_proxy = std::unique_ptr<OIIO::Filesystem::IOProxy>(
        create_input_proxy( reader, use_buffer, m_buffer));

    if( !setup_image_input( /*from_constructor*/ true))
        return;

    const OIIO::ImageSpec& spec = m_image_input->spec();

    m_resolution_x = spec.width;
    m_resolution_y = spec.height;
    m_resolution_z = spec.depth;
    m_pixel_type   = get_pixel_type( spec);

    assert( m_resolution_z == 1); // see comment in read()
}

const char* Image_file_reader_impl::get_type() const
{
    return IMAGE::convert_pixel_type_enum_to_string( m_pixel_type);
}

mi::Uint32 Image_file_reader_impl::get_resolution_x( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_x;
}

mi::Uint32 Image_file_reader_impl::get_resolution_y( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_y;
}

mi::Uint32 Image_file_reader_impl::get_layers_size( mi::Uint32 level) const
{
    if( level > 0)
        return 0;
    return m_resolution_z;
}

mi::Uint32 Image_file_reader_impl::get_miplevels() const
{
    // Support only one miplevel for now. OpenImageIO does not support retrieving this number
    // without looping through the miplevels.
    return 1;
}

bool Image_file_reader_impl::get_is_cubemap() const
{
    return false;
}

mi::Float32 Image_file_reader_impl::get_gamma() const
{
    return IMAGE::get_default_gamma( m_pixel_type);
}

mi::neuraylib::ITile* Image_file_reader_impl::read( mi::Uint32 z, mi::Uint32 level) const
{
    if( !setup_image_input( /*from_constructor*/ false))
        return nullptr;

    const char* pixel_type = convert_pixel_type_enum_to_string( m_pixel_type);
    mi::base::Handle<mi::neuraylib::ITile> tile(
        m_image_api->create_tile( pixel_type, m_resolution_x, m_resolution_y));

    int cpp = IMAGE::get_components_per_pixel( m_pixel_type);
    int bpc = IMAGE::get_bytes_per_component( m_pixel_type);
    int bytes_per_row = m_resolution_x * cpp * bpc;

    OIIO::TypeDesc format( get_base_type( m_pixel_type));
    mi::Uint8* data = static_cast<mi::Uint8*>( tile->get_data());

    // Note that read_image() does not support specifying a range in z direction. This should not
    // become necessary for the registered file formats. If this changes we need to read the entire
    // image and extract the requested layer. Note that read_scanlines() allows to specify a range
    // in z direction, but does not work for tiled images.
    assert( m_resolution_z == 1);
    if( m_resolution_z != 1)
        return nullptr;
    bool success = m_image_input->read_image(
        /*subimage*/ 0,
        /*miplevel*/ level,
        /*chbegin*/ 0,
        /*chend*/ cpp,
        format,
        data + (m_resolution_y - 1) * static_cast<size_t>( bytes_per_row),
        /*xstride*/ OIIO::AutoStride,
        /*ystride*/ -bytes_per_row,
        /*zstride*/ OIIO::AutoStride);
    if( !success)
        return nullptr;

    const OIIO::ImageSpec& spec = m_image_input->spec();
    if( (spec.nchannels == 2) && (spec.channelnames[0] == "Y") && (spec.channelnames[1] == "A"))
        expand_ya_to_rgba( bpc, m_resolution_x, m_resolution_y, data);

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
    mi::math::Color c;
    tile->get_pixel( DUMP_PIXEL_X, DUMP_PIXEL_Y, &c.r);
    std::cout << "OIIO plugin reader (before unassociate_alpha()): "
              << c.r << " " << c.g << " " << c.b << " " << c.a << std::endl;
#endif

    if( spec.get_int_attribute( "oiio:UnassociatedAlpha") == 0) {
        // Use 1.0f instead of get_gamma().  Unclear what is correct here. OIIO itself uses the
        // actual gamma value, whereas FreeImage and GIMP seem to ignore gamma for this and always
        // assume gamma == 1.0f.
        // OIIO documentation mentions that alpha and depth should be assumed linear.
        tile = unassociate_alpha( m_image_api.get(), tile.get(), 1.0f);
    } else if( m_plugin_name == "oiio_bmp") {
        // It is unclear whether BMP uses associated or unassociated alpha. We use unassociated
        // alpha for historic reasons, in contrast to OIIO. Doing so without support from the
        // OIIO handler for BMP is a misuse of the OIIO API.
        tile = unassociate_alpha( m_image_api.get(), tile.get(), 1.0f);
    } else {
        // Avoid redundant conversions and resulting quantization errors
#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
        std::cout << "OIIO plugin reader (skip unassociate_alpha())" << std::endl;
#endif
    }

#if defined(DUMP_PIXEL_X) && defined(DUMP_PIXEL_Y)
    tile->get_pixel( DUMP_PIXEL_X, DUMP_PIXEL_Y, &c.r);
    std::cout << "OIIO plugin reader (after unassociate_alpha()): "
              << c.r << " " << c.g << " " << c.b << " " << c.a << std::endl;
#endif

    tile->retain();
    return tile.get();
}

bool Image_file_reader_impl::write(
    const mi::neuraylib::ITile* tile, mi::Uint32 z, mi::Uint32 level)
{
    assert( false);
    return false;
}

bool Image_file_reader_impl::is_valid() const
{
    return !!m_image_input;
}

bool Image_file_reader_impl::setup_image_input( bool from_constructor) const
{
    if( !from_constructor) {

        // Nothing to do if the special header-only was not activated.
        if( !m_nv_header_only)
            return true;

        // Disable header-only mode and re-create m_image_input below.
        m_nv_header_only = false;
    }

    // Adding a dot here triggers a code path in OIIO that tries all plugins if the one for the
    // claimed format fails. We do not want to support such misnamed files, so we leave out the
    // dot here.
    std::string ext = /*"." +*/ m_oiio_format;

    OIIO::ImageSpec config;
    config["oiio:UnassociatedAlpha"] = 1;

    if( from_constructor) {
        // Enable header-only mode for libjpeg. Large speedup for progressive JPEGs.
        if( m_plugin_name == "oiio_jpg") {
            config["nv:header_only"] = 1;
            m_nv_header_only = true;
        }
    }

    // Workaround for issue #3273: Set proxy also via config.
    OIIO::Filesystem::IOProxy* io_proxy = m_io_proxy.get();
    config.attribute( "oiio:ioproxy", OIIO::TypeDesc::PTR, &io_proxy);

    m_image_input = OIIO::ImageInput::open( ext, &config, m_io_proxy.get());
    if( !m_image_input)
        return false;

    // Disable OIIO internal thread pool since it causes shutdown problems in larger integrations.
    m_image_input->threads( 1);

    return true;
}

} // namespace MI_OIIO

} // namespace MI
