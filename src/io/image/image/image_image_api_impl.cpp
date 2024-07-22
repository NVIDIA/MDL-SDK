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

/** \file
 ** \brief Implementation of IImage_api
 **
 ** Implements the IImage_api interface
 **/

#include "pch.h"

#include "image_image_api_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iimage_plugin.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/version.h>

#include <base/data/idata/i_idata_factory.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/log/i_log_utilities.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>

#include "i_image.h"
#include "i_image_access_canvas.h"

namespace MI {

namespace IMAGE {

Image_api_impl::Image_api_impl( IMAGE::Image_module* image_module)
  : m_image_module( image_module)
{
}

mi::neuraylib::ITile* Image_api_impl::create_tile(
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    return m_image_module->create_tile( pixel_type_enum, width, height);
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas(
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    return m_image_module->create_canvas(
        pixel_type_enum, width, height, layers, is_cubemap, gamma);
}

mi::neuraylib::ICanvas_cuda* Image_api_impl::create_canvas_cuda(
    mi::Sint32 cuda_device_id,
    const char* pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    mi::Float32 gamma) const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return nullptr;
}

mi::IArray* Image_api_impl::create_mipmap(
    const mi::neuraylib::ICanvas* canvas, mi::Float32 gamma) const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return nullptr;
}

mi::neuraylib::ITile* Image_api_impl::clone_tile( const mi::neuraylib::ITile* tile) const
{
    if( !tile)
        return nullptr;

    return m_image_module->copy_tile( tile);
}

mi::neuraylib::ICanvas* Image_api_impl::clone_canvas( const mi::neuraylib::ICanvas* canvas) const
{
    if( !canvas)
        return nullptr;

    return m_image_module->copy_canvas( canvas);
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

    const IMAGE::Pixel_type pixel_type
        = IMAGE::convert_pixel_type_string_to_enum( buffer_pixel_type);
    if( pixel_type == IMAGE::PT_UNDEF)
        return -3;

    IMAGE::Access_canvas access_canvas( canvas);
    const bool success = access_canvas.read_rect(
        static_cast<mi::Uint8*>( buffer),
        buffer_topdown,
        pixel_type,
        canvas_x,
        canvas_y,
        width,
        height,
        buffer_padding,
        canvas_layer);
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

    const IMAGE::Pixel_type pixel_type
        = IMAGE::convert_pixel_type_string_to_enum( buffer_pixel_type);
    if( pixel_type == IMAGE::PT_UNDEF)
        return -3;

    IMAGE::Edit_canvas edit_canvas( canvas);
    const bool success = edit_canvas.write_rect(
        static_cast<const mi::Uint8*>( buffer), buffer_topdown,
        pixel_type,
        canvas_x,
        canvas_y,
        width,
        height,
        buffer_padding,
        canvas_layer);
    return success ? 0 : -4;
}

mi::neuraylib::IBuffer* Image_api_impl::create_buffer_from_canvas(
    const mi::neuraylib::ICanvas* canvas,
    const char* image_format,
    const char* pixel_type,
    const mi::IMap* export_options) const
{
    if( !canvas || !image_format || !pixel_type)
        return nullptr;

    return m_image_module->create_buffer_from_canvas(
        canvas, image_format, pixel_type, export_options);
}

mi::neuraylib::IBuffer* Image_api_impl::deprecated_create_buffer_from_canvas(
    const mi::neuraylib::ICanvas* canvas,
    const char* image_format,
    const char* pixel_type,
    const char* quality,
    bool force_default_gamma) const
{
    std::optional<mi::Uint32> quality_optional = STRING::lexicographic_cast_s<mi::Uint32>( quality);
    if( !quality_optional.has_value())
        return nullptr;

    const mi::Uint32 quality_uint32 = quality_optional.value();
    if( quality_uint32 > 100)
        return nullptr;

    mi::base::Handle<mi::IMap> export_options(
         m_image_module->convert_legacy_options( quality_uint32, force_default_gamma));

    return create_buffer_from_canvas( canvas, image_format, pixel_type, export_options.get());
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas_from_buffer(
    const mi::neuraylib::IBuffer* buffer,
    const char* image_format,
    const char* selector) const
{
    if( !buffer || !image_format)
        return nullptr;

    DISK::Memory_reader_impl reader( buffer);
    return m_image_module->create_canvas(
        IMAGE::Memory_based(), &reader, image_format, selector);
}

mi::neuraylib::ICanvas* Image_api_impl::create_canvas_from_reader(
    mi::neuraylib::IReader* reader, const char* image_format, const char* selector) const
{
    if( !reader || !image_format)
        return nullptr;

    return m_image_module->create_canvas(
        IMAGE::Memory_based(), reader, image_format, selector);
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

void Image_api_impl::adjust_gamma(
    mi::neuraylib::ITile* tile, mi::Float32 old_gamma, mi::Float32 new_gamma) const
{
    m_image_module->adjust_gamma( tile, old_gamma, new_gamma);
}

void Image_api_impl::adjust_gamma( mi::neuraylib::ICanvas* canvas, mi::Float32 new_gamma) const
{
    m_image_module->adjust_gamma( canvas, new_gamma);
}

mi::neuraylib::ITile* Image_api_impl::convert(
    const mi::neuraylib::ITile* tile, const char* pixel_type) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    return m_image_module->convert_tile( tile, pixel_type_enum);
}

mi::neuraylib::ICanvas* Image_api_impl::convert(
    const mi::neuraylib::ICanvas* canvas, const char* pixel_type) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    if( pixel_type_enum == IMAGE::PT_UNDEF)
        return nullptr;

    return m_image_module->convert_canvas( canvas, pixel_type_enum);
}

mi::Uint32 Image_api_impl::get_components_per_pixel( const char* pixel_type) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    return IMAGE::get_components_per_pixel( pixel_type_enum);
}

mi::Uint32 Image_api_impl::get_bytes_per_component( const char* pixel_type) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    return IMAGE::get_bytes_per_component( pixel_type_enum);
}

const char* Image_api_impl::get_pixel_type_for_channel(
    const char* pixel_type, const char* selector) const
{
    const IMAGE::Pixel_type pixel_type_enum = IMAGE::convert_pixel_type_string_to_enum( pixel_type);
    const IMAGE::Pixel_type result = IMAGE::get_pixel_type_for_channel( pixel_type_enum, selector);
    return convert_pixel_type_enum_to_string( result);
}

mi::neuraylib::ICanvas* Image_api_impl::extract_channel(
    const mi::neuraylib::ICanvas* canvas, const char* selector) const
{
    return m_image_module->extract_channel( canvas, selector);
}

mi::neuraylib::ITile* Image_api_impl::extract_channel(
    const mi::neuraylib::ITile* tile, const char* selector) const
{
    return m_image_module->extract_channel( tile, selector);
}

mi::Sint32 Image_api_impl::start()
{
    m_image_module_access.set();
    m_image_module = m_image_module_access.operator->();
    return 0;
}

mi::Sint32 Image_api_impl::shutdown()
{
    m_image_module = nullptr;
    m_image_module_access.reset();
    return 0;
}

Logging_configuration_impl::Logging_configuration_impl()
  : m_forwarding_logger( new LOG::Forwarding_logger)
{
}

void Logging_configuration_impl::set_receiving_logger( mi::base::ILogger* logger)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
}

mi::base::ILogger* Logging_configuration_impl::get_receiving_logger() const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return nullptr;
}

mi::base::ILogger* Logging_configuration_impl::get_forwarding_logger() const
{
    m_forwarding_logger->retain();
    return m_forwarding_logger.get();
}

mi::Sint32 Logging_configuration_impl::set_log_level( mi::base::Message_severity level)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return -1;
}

mi::base::Message_severity Logging_configuration_impl::get_log_level() const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return mi::base::MESSAGE_SEVERITY_INFO;
}

mi::Sint32 Logging_configuration_impl::set_log_level_by_category(
    const char* category, mi::base::Message_severity level)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return -1;
}

mi::base::Message_severity Logging_configuration_impl::get_log_level_by_category(
    const char* category) const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return mi::base::MESSAGE_SEVERITY_INFO;
}

void Logging_configuration_impl::set_log_prefix( mi::Uint32 prefix)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
}

mi::Uint32 Logging_configuration_impl::get_log_prefix() const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return 0;
}

mi::Sint32 Logging_configuration_impl::set_log_priority( mi::Sint32 priority)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return -1;
}

mi::Sint32 Logging_configuration_impl::get_log_priority() const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return 0;
}

mi::Sint32 Logging_configuration_impl::set_log_locally( bool value)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return -1;
}

bool Logging_configuration_impl::get_log_locally() const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return false;
}

Plugin_api_impl::Plugin_api_impl( IMAGE::Image_module* image_module)
{
    m_logging_configuration = new Logging_configuration_impl;
    m_image_api = new Image_api_impl( image_module);
}

mi::Uint32 Plugin_api_impl::get_interface_version() const
{
    return MI_NEURAYLIB_API_VERSION;
}

const char* Plugin_api_impl::get_version() const
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return nullptr;
}

mi::base::IInterface* Plugin_api_impl::get_api_component(
    const mi::base::Uuid& uuid) const
{
    if( uuid == mi::neuraylib::ILogging_configuration::IID()) {
        m_logging_configuration->retain();
        return m_logging_configuration.get();
    }

    if( uuid == mi::neuraylib::IImage_api::IID()) {
        m_image_api->retain();
        return m_image_api.get();
    }

    return nullptr;
}

mi::Sint32 Plugin_api_impl::register_api_component(
    const mi::base::Uuid& uuid, mi::base::IInterface* api_component)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return -1;
}

mi::Sint32 Plugin_api_impl::unregister_api_component( const mi::base::Uuid& uuid)
{
    ASSERT( M_IMAGE, !"not implemented by this implementation");
    return -1;
}

} // namespace IMAGE

} // namespace MI
