/***************************************************************************************************
 * Copyright (c) 2009-2023, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Implementation of IImage_api
 **
 ** Implements the IImage_api interface
 **/

#include "pch.h"

#include "neuray_image_api_impl.h"
#include "neuray_array_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/ipointer.h>

#include <io/image/image/i_image.h>

namespace MI {

namespace NEURAY {

Image_api_impl::Image_api_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray),
    m_impl( nullptr)
{
}

Image_api_impl::~Image_api_impl()
{
    m_neuray = nullptr;
}

mi::neuraylib::ITile* Image_api_impl::create_tile(
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height) const
{
    return m_impl.create_tile( pixel_type, width, height);
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas(
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    return m_impl.create_canvas( pixel_type, width, height, layers, is_cubemap, gamma);
}

mi::neuraylib::ICanvas_cuda* Image_api_impl::create_canvas_cuda(
    mi::Sint32 cuda_device_id,
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    mi::Float32 gamma) const
{
    // Not implemented by IMAGE::IMage_api_impl due to the #ifdef here.

    return nullptr;
}

mi::IArray* Image_api_impl::create_mipmaps(
    const mi::neuraylib::ICanvas* canvas, mi::Float32 gamma) const
{
    // Not implemented by IMAGE::IMage_api_impl due to the dependencies on Array_impl and IPointer.

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > mipmaps;
    m_image_module->create_mipmaps( mipmaps, canvas, gamma);
    if( mipmaps.empty())
        return nullptr;

    mi::IArray* array = new Array_impl( nullptr, "Pointer<Interface>", mipmaps.size());
    for( mi::Size i = 0; i < mipmaps.size(); ++i) {
        mi::base::Handle<mi::IPointer> element( array->get_element<mi::IPointer>( i));
        element->set_pointer( mipmaps[i].get());
    }

    return array;
}

mi::neuraylib::ITile* Image_api_impl::clone_tile( const mi::neuraylib::ITile* tile) const
{
    return m_impl.clone_tile( tile);
}

mi::neuraylib::ICanvas* Image_api_impl::clone_canvas( const mi::neuraylib::ICanvas* canvas) const
{
    return m_impl.clone_canvas( canvas);
}

mi::Sint32 Image_api_impl::read_raw_pixels(
    mi::Uint32 width,
    mi::Uint32 height,
    const mi::neuraylib::ICanvas* canvas,
    mi::Uint32 canvas_x,
    mi::Uint32 canvas_y,
    mi::Uint32 canvas_layer,
    void* buffer,
    bool buffer_topdown,
    const char* buffer_pixel_type,
    mi::Uint32 buffer_padding) const
{
    return m_impl.read_raw_pixels(
        width,
        height,
        canvas,
        canvas_x,
        canvas_y,
        canvas_layer,
        buffer,
        buffer_topdown,
        buffer_pixel_type,
        buffer_padding);
}

mi::Sint32 Image_api_impl::write_raw_pixels(
    mi::Uint32 width,
    mi::Uint32 height,
    mi::neuraylib::ICanvas* canvas,
    mi::Uint32 canvas_x,
    mi::Uint32 canvas_y,
    mi::Uint32 canvas_layer,
    const void* buffer,
    bool buffer_topdown,
    const char* buffer_pixel_type,
    mi::Uint32 buffer_padding) const
{
    return m_impl.write_raw_pixels(
        width,
        height,
        canvas,
        canvas_x,
        canvas_y,
        canvas_layer,
        buffer,
        buffer_topdown,
        buffer_pixel_type,
        buffer_padding);
}

mi::neuraylib::IBuffer* Image_api_impl::create_buffer_from_canvas(
    const mi::neuraylib::ICanvas* canvas,
    const char* image_format,
    const char* pixel_type,
    const char* quality,
    bool force_default_gamma) const
{
    return m_impl.create_buffer_from_canvas(
        canvas,
        image_format,
        pixel_type,
        quality,
        force_default_gamma);
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas_from_buffer(
    const mi::neuraylib::IBuffer* buffer,
    const char* image_format) const
{
    return m_impl.create_canvas_from_buffer( buffer, image_format);
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas_from_reader(
    mi::neuraylib::IReader* reader, const char* image_format) const
{
    return m_impl.create_canvas_from_reader( reader, image_format);
}

bool Image_api_impl::supports_format_for_decoding(
    const char* image_format, mi::neuraylib::IReader* reader) const
{
    return m_impl.supports_format_for_decoding( image_format, reader);
}

bool Image_api_impl::supports_format_for_encoding( const char* image_format) const
{
    return m_impl.supports_format_for_encoding( image_format);
}

mi::neuraylib::ITile* Image_api_impl::convert(
    const mi::neuraylib::ITile* tile, const char* pixel_type) const
{
    return m_impl.convert( tile, pixel_type);
}

mi::neuraylib::ICanvas* Image_api_impl::convert(
    const mi::neuraylib::ICanvas* canvas, const char* pixel_type) const
{
    return m_impl.convert( canvas, pixel_type);
}

void Image_api_impl::adjust_gamma(
    mi::neuraylib::ITile* tile, mi::Float32 old_gamma, mi::Float32 new_gamma) const
{
    return m_impl.adjust_gamma( tile, old_gamma, new_gamma);
}

void Image_api_impl::adjust_gamma( mi::neuraylib::ICanvas* canvas, mi::Float32 new_gamma) const
{
    return m_impl.adjust_gamma( canvas, new_gamma);
}

mi::Uint32 Image_api_impl::get_components_per_pixel( const char* pixel_type) const
{
    return m_impl.get_components_per_pixel( pixel_type);
}

mi::Uint32 Image_api_impl::get_bytes_per_component( const char* pixel_type) const
{
    return m_impl.get_bytes_per_component( pixel_type);
}

const char* Image_api_impl::get_pixel_type_for_channel(
    const char* pixel_type, const char* selector) const
{
    return m_impl.get_pixel_type_for_channel( pixel_type, selector);
}

mi::neuraylib::ICanvas* Image_api_impl::extract_channel(
    const mi::neuraylib::ICanvas* canvas, const char* selector) const
{
    return m_impl.extract_channel( canvas, selector);
}

mi::neuraylib::ITile* Image_api_impl::extract_channel(
    const mi::neuraylib::ITile* tile, const char* selector) const
{
   return m_impl.extract_channel( tile, selector);
}

mi::Sint32 Image_api_impl::start()
{
    m_image_module.set();
    m_impl.start();
    return 0;
}

mi::Sint32 Image_api_impl::shutdown()
{
    m_impl.shutdown();
    m_image_module.reset();
    return 0;
}

} // namespace NEURAY

} // namespace MI

