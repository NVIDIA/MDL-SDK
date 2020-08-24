/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "neuray_image_impl.h"
#include "neuray_array_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iimage_plugin.h>
#include <mi/neuraylib/ipointer.h>

#include <base/system/stlext/i_stlext_likely.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/lib/log/i_log_logger.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_access_canvas.h>
#include <io/scene/dbimage/i_dbimage.h>

namespace MI {

namespace NEURAY {

Image_api_impl::Image_api_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray)
{
}

Image_api_impl::~Image_api_impl()
{
    m_neuray = nullptr;
}

mi::neuraylib::ICanvas_cuda* Image_api_impl::create_canvas_cuda(
    mi::Sint32 cuda_device_id,
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    mi::Float32 gamma) const
{
    return nullptr;
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas(
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    if( tile_width == 0)
        tile_width = width;
    if( tile_height == 0)
        tile_height = height;

    return m_image_module->create_canvas(
        pixel_type_enum, width, height, tile_width, tile_height, layers, is_cubemap, gamma);
}

mi::neuraylib::ITile* Image_api_impl::create_tile(
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height) const
{
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    return m_image_module->create_tile( pixel_type_enum, width, height);
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
    if( !canvas || !buffer || !buffer_pixel_type)
        return -1;

    if( width == 0 || height == 0)
        return -2;

    IMAGE::Pixel_type pixel_type = IMAGE::convert_pixel_type_string_to_enum( buffer_pixel_type);
    if( pixel_type == IMAGE::PT_UNDEF)
        return -3;

    IMAGE::Access_canvas access_canvas( canvas);
    bool success = access_canvas.read_rect( static_cast<mi::Uint8*>( buffer), buffer_topdown,
        pixel_type, canvas_x, canvas_y, width, height, buffer_padding, canvas_layer);
    return success ? 0 : -4;
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
    if( !canvas || !buffer || !buffer_pixel_type)
        return -1;

    if( width == 0 || height == 0)
        return -2;

    IMAGE::Pixel_type pixel_type = IMAGE::convert_pixel_type_string_to_enum( buffer_pixel_type);
    if( pixel_type == IMAGE::PT_UNDEF)
        return -3;

    IMAGE::Edit_canvas edit_canvas( canvas);
    bool success = edit_canvas.write_rect( static_cast<const mi::Uint8*>( buffer), buffer_topdown,
        pixel_type, canvas_x, canvas_y, width, height, buffer_padding, canvas_layer);
    return success ? 0 : -4;
}

mi::neuraylib::IBuffer* Image_api_impl::create_buffer_from_canvas(
    const mi::neuraylib::ICanvas* canvas,
    const char* image_format,
    const char* pixel_type,
    const char* quality) const
{
    if( !canvas || !image_format || !pixel_type)
        return nullptr;

    STLEXT::Likely<mi::Uint32> quality_likely = STRING::lexicographic_cast_s<mi::Uint32>( quality);
    if( !quality_likely.get_status())
        return nullptr;

    mi::Uint32 quality_uint32 = *quality_likely.get_ptr(); //-V522 PVS
    if( quality_uint32 > 100)
        return nullptr;

    return m_image_module->create_buffer_from_canvas(
        canvas, image_format, pixel_type, quality_uint32);
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas_from_buffer(
    const mi::neuraylib::IBuffer* buffer,
    const char* image_format) const
{
    if( !buffer || !image_format)
        return nullptr;

    DISK::Memory_reader_impl reader( buffer);
    return m_image_module->create_canvas( IMAGE::Memory_based(), &reader, image_format);
}

bool Image_api_impl::supports_format_for_decoding(
    const char* image_format, mi::neuraylib::IReader* reader) const
{
   const mi::neuraylib::IImage_plugin* plugin =
        m_image_module->find_plugin_for_import( image_format, reader);
    return plugin != nullptr;
}

bool Image_api_impl::supports_format_for_encoding( const char* image_format) const
{
    const mi::neuraylib::IImage_plugin* plugin =
        m_image_module->find_plugin_for_export( image_format);

    return plugin != nullptr;
}

mi::neuraylib::ICanvas* Image_api_impl::convert(
    const mi::neuraylib::ICanvas* canvas, const char* pixel_type) const
{
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    return m_image_module->convert_canvas( canvas, pixel_type_enum);
}

void Image_api_impl::adjust_gamma( mi::neuraylib::ICanvas* canvas, mi::Float32 new_gamma) const
{
    m_image_module->adjust_gamma( canvas, new_gamma);
}

mi::Uint32 Image_api_impl::get_components_per_pixel( const char* pixel_type) const
{
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    return IMAGE::get_components_per_pixel( pixel_type_enum);
}

mi::Uint32 Image_api_impl::get_bytes_per_component( const char* pixel_type) const
{
    IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    return IMAGE::get_bytes_per_component( pixel_type_enum);
}

mi::IArray* Image_api_impl::create_mipmaps(
    const mi::neuraylib::ICanvas* canvas,
    mi::Float32 gamma) const
{
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > mipmaps;
    m_image_module->create_mipmaps(mipmaps, canvas, gamma);

    if (mipmaps.empty())
        return nullptr;

    Array_impl* arr = new Array_impl(nullptr, "Pointer<Interface>", mipmaps.size());
    for(mi::Size i=0; i<mipmaps.size(); ++i)
    {
        mi::base::Handle<mi::base::IInterface> if_p (arr->get_element(i));
        mi::base::Handle<mi::IPointer> p(if_p->get_interface<mi::IPointer>());
        p->set_pointer(mipmaps[i].get());
    }
    return arr;
}

mi::Sint32 Image_api_impl::start()
{
    m_image_module.set();
    return 0;
}

mi::Sint32 Image_api_impl::shutdown()
{
    m_image_module.reset();
    return 0;
}

} // namespace NEURAY

} // namespace MI

