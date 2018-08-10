/***************************************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "image_module_impl.h"
#include "i_image_pixel_conversion.h"
#include "i_image_utilities.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/iimage_plugin.h>
#include <mi/neuraylib/iplugin_api.h>

#include <cstring>
#include <sstream>
#include <queue>
#include <base/system/main/module_registration.h>
#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/plug/i_plug.h>
#include <base/util/string_utils/i_string_utils.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/data/serial/i_serializer.h>

#include "image_canvas_impl.h"
#include "image_tile_impl.h"
#include "image_mipmap_impl.h"

#include <iomanip>
#include <limits>

namespace MI {

namespace IMAGE {

/// The less-than functor for plugin selection.
class Plugin_less
{
public:
    bool operator()(
        const mi::neuraylib::IImage_plugin* lhs, const mi::neuraylib::IImage_plugin* rhs)
    {
        return lhs->get_priority() < rhs->get_priority();
    }
};

// Register the module.
static SYSTEM::Module_registration<Image_module_impl> s_module( M_IMAGE, "IMAGE");

Module_registration_entry* Image_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

bool Image_module_impl::init()
{
    m_plug_module.set();

    mi::base::Handle<mi::neuraylib::IPlugin_api> plugin_api( m_plug_module->get_plugin_api());

    // Call IImage_plugin::init() on our type of plugins
    size_t index = 0;
    mi::base::Handle<mi::base::IPlugin_descriptor> descriptor( m_plug_module->get_plugin( index));
    while( descriptor) {

        mi::base::Plugin* plugin = descriptor->get_plugin();
        const char* type = plugin->get_type();
        const char* name = plugin->get_name();
        const char* filename = descriptor->get_plugin_library_path();

        if( is_valid_image_plugin( type, name, filename)) {

            // Call IImage_plugin::init()
            mi::neuraylib::IImage_plugin* image_plugin
                = static_cast<mi::neuraylib::IImage_plugin*>( plugin);
            image_plugin->init( plugin_api.get());

            // Store plugin for exit()
            mi::base::Lock::Block block( &m_plugins_lock);
            m_plugins.push_back( descriptor);
        }

        descriptor = m_plug_module->get_plugin( ++index);
    }

    return true;
}

void Image_module_impl::exit()
{
    mi::base::Handle<mi::neuraylib::IPlugin_api> plugin_api( m_plug_module->get_plugin_api());

    // Call IImage_plugin::exit() on our type of plugins
    mi::base::Lock::Block block( &m_plugins_lock);
    Plugin_vector::reverse_iterator it     = m_plugins.rbegin();
    Plugin_vector::reverse_iterator it_end = m_plugins.rend();
    for( ; it != it_end; ++it) {
        mi::base::Plugin* plugin = (*it)->get_plugin();
        mi::neuraylib::IImage_plugin* image_plugin
            = static_cast<mi::neuraylib::IImage_plugin*>( plugin);
        image_plugin->exit( plugin_api.get());
    }
    m_plugins.clear();

    m_plug_module.reset();
}

IMipmap* Image_module_impl::create_mipmap(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    if( pixel_type == PT_UNDEF || width == 0 || height == 0 || layers == 0 || gamma < 0)
        return 0;

    if( is_cubemap && layers != 6)
        return 0;

    return new Mipmap_impl(
        pixel_type, width, height, tile_width, tile_height, layers, is_cubemap, gamma);
}

IMipmap* Image_module_impl::create_mipmap(
    const std::string& filename,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    bool only_first_level,
    mi::Sint32* errors) const
{
    return new Mipmap_impl( filename, tile_width, tile_height, only_first_level, errors);
}

IMipmap* Image_module_impl::create_mipmap(
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& member_filename,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    bool only_first_level,
    mi::Sint32* errors) const
{
    return new Mipmap_impl(
        reader,
        archive_filename,
        member_filename,
        tile_width,
        tile_height,
        only_first_level,
        errors);
}

IMipmap* Image_module_impl::create_mipmap(
    mi::neuraylib::IReader* reader,
    const char* image_format,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    bool only_first_level,
    mi::Sint32* errors) const
{
    return new Mipmap_impl(
        reader, image_format, tile_width, tile_height, only_first_level, errors);
}

IMipmap* Image_module_impl::create_mipmap(
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> >& canvases, bool is_cubemap) const
{
    mi::Size count = canvases.size();
    if( count == 0)
        return 0;

    for( mi::Uint32 i = 0; i < count; ++i)
        if( !canvases[i])
            return 0;

    return new Mipmap_impl( canvases, is_cubemap);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    if( pixel_type == PT_UNDEF || width == 0 || height == 0 || layers == 0 || gamma < 0)
        return 0;

    if( is_cubemap && layers != 6)
        return 0;

    return new Canvas_impl(
        pixel_type, width, height, tile_width, tile_height, layers, is_cubemap, gamma);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    const std::string& filename,
    mi::Uint32 miplevel,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Sint32* errors) const
{
    return new Canvas_impl( filename, miplevel, tile_width, tile_height, /*image_file*/0, errors);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& member_filename,
    mi::Uint32 miplevel,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Sint32* errors) const
{
    return new Canvas_impl(
        reader,
        archive_filename,
        member_filename,
        miplevel,
        tile_width,
        tile_height,
        /*image_file*/ 0,
        errors);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    mi::neuraylib::IReader* reader,
    const char* image_format,
    mi::Uint32 miplevel,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Sint32* errors) const
{
    if( !reader || !image_format)
        return 0;

    return new Canvas_impl(
        reader,
        image_format,
        miplevel,
        tile_width,
        tile_height,
        /*image_file*/ 0,
        errors);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    mi::neuraylib::ITile* tile, mi::Float32 gamma) const
{
    if( !tile || gamma < 0)
        return 0;

    return new Canvas_impl( tile, gamma);
}

mi::neuraylib::ITile* Image_module_impl::create_tile(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height) const
{
    if( width == 0 || height == 0 || pixel_type == PT_UNDEF)
        return 0;

    return IMAGE::create_tile( pixel_type, width, height);
}

IMipmap* Image_module_impl::copy_mipmap( const IMipmap* other, bool only_first_level) const
{
    mi::Uint32 levels = only_first_level ? 1 : other->get_nlevels();

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( levels);
    for( mi::Uint32 i = 0; i < levels; ++i) {
        mi::base::Handle<const mi::neuraylib::ICanvas> other_canvas( other->get_level( i));
        canvases[i] = copy_canvas( other_canvas.get());
    }

    return create_mipmap( canvases, other->get_is_cubemap());
}

mi::neuraylib::ICanvas* Image_module_impl::copy_canvas( const mi::neuraylib::ICanvas* other) const
{
    Pixel_type pixel_type    = convert_pixel_type_string_to_enum( other->get_type());
    if( pixel_type == PT_UNDEF)
        return 0;
    mi::Uint32 canvas_width  = other->get_resolution_x();
    mi::Uint32 canvas_height = other->get_resolution_y();
    mi::Uint32 tile_width    = other->get_tile_resolution_x();
    mi::Uint32 tile_height   = other->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = other->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = other->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = other->get_layers_size();
    mi::Float32 gamma        = other->get_gamma();

    bool is_cubemap = get_canvas_is_cubemap( other);
    mi::neuraylib::ICanvas* canvas = create_canvas( pixel_type,
        canvas_width, canvas_height, tile_width, tile_height, nr_of_layers, is_cubemap, gamma);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    mi::Size count = static_cast<mi::Size>( tile_width) * tile_height * bytes_per_pixel;

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
        for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
            for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                mi::base::Handle<const mi::neuraylib::ITile> other_tile(
                    other->get_tile( x*tile_width, y*tile_height, z));
                mi::base::Handle<mi::neuraylib::ITile> tile(
                    canvas->get_tile( x*tile_width, y*tile_height, z));
                const void* other_tile_data = other_tile->get_data();
                void* tile_data = tile->get_data();
                memcpy( tile_data, other_tile_data, count);
            }

    return canvas;
}

mi::neuraylib::ITile* Image_module_impl::copy_tile( const mi::neuraylib::ITile* other) const
{
    Pixel_type pixel_type = convert_pixel_type_string_to_enum( other->get_type());
    mi::Uint32 width  = other->get_resolution_x();
    mi::Uint32 height = other->get_resolution_y();
    if( width == 0 || height == 0 || pixel_type == PT_UNDEF)
        return 0;

    mi::neuraylib::ITile* tile = create_tile( pixel_type, width, height);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    mi::Size count = static_cast<mi::Size>( width) * height * bytes_per_pixel;
    const void* other_tile_data = other->get_data();
    void* tile_data = tile->get_data();
    memcpy( tile_data, other_tile_data, count);

    return tile;
}

IMipmap* Image_module_impl::convert_mipmap(
    const IMipmap* old_mipmap, Pixel_type new_pixel_type, bool only_first_level) const
{
    if( !old_mipmap)
        return 0;

    mi::Uint32 converted_levels = only_first_level ? 1 : old_mipmap->get_nlevels();

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > new_canvases( converted_levels);
    for( mi::Uint32 i = 0; i < converted_levels; ++i) {
        mi::base::Handle<const mi::neuraylib::ICanvas> old_canvas(
            old_mipmap->get_level( i));
        new_canvases[i] = convert_canvas( old_canvas.get(), new_pixel_type);
    }

    return create_mipmap( new_canvases, old_mipmap->get_is_cubemap());
}

mi::neuraylib::ICanvas* Image_module_impl::convert_canvas(
    const mi::neuraylib::ICanvas* old_canvas, Pixel_type new_pixel_type) const
{
    if( !old_canvas || new_pixel_type == PT_UNDEF)
        return 0;

    Pixel_type old_pixel_type = convert_pixel_type_string_to_enum( old_canvas->get_type());
    if( old_pixel_type == PT_UNDEF)
        return 0;

    if( old_pixel_type == new_pixel_type)
        return copy_canvas( old_canvas); // faster than the code below

    mi::Uint32 canvas_width  = old_canvas->get_resolution_x();
    mi::Uint32 canvas_height = old_canvas->get_resolution_y();
    mi::Uint32 tile_width    = old_canvas->get_tile_resolution_x();
    mi::Uint32 tile_height   = old_canvas->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = old_canvas->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = old_canvas->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = old_canvas->get_layers_size();
    mi::Size   nr_of_pixels  = tile_width * tile_height;
    mi::Float32 gamma        = old_canvas->get_gamma();

    bool is_cubemap = get_canvas_is_cubemap( old_canvas);
    mi::neuraylib::ICanvas* new_canvas = new Canvas_impl( new_pixel_type,
        canvas_width, canvas_height, tile_width, tile_height, nr_of_layers, is_cubemap, gamma);

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
        for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
            for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                mi::base::Handle<const mi::neuraylib::ITile> old_tile(
                    old_canvas->get_tile( x*tile_width, y*tile_height, z));
                mi::base::Handle<mi::neuraylib::ITile> new_tile(
                    new_canvas->get_tile( x*tile_width, y*tile_height, z));
                const void* old_data = old_tile->get_data();
                void* new_data = new_tile->get_data();
                convert( old_data, new_data, old_pixel_type, new_pixel_type, nr_of_pixels);
            }

    return new_canvas;
}

mi::neuraylib::ITile* Image_module_impl::convert_tile(
    const mi::neuraylib::ITile* old_tile, Pixel_type new_pixel_type) const
{
    if( !old_tile || new_pixel_type == PT_UNDEF)
        return 0;

    Pixel_type old_pixel_type = convert_pixel_type_string_to_enum( old_tile->get_type());
    if( old_pixel_type == PT_UNDEF)
        return 0;

    if( old_pixel_type == new_pixel_type)
        return copy_tile( old_tile); // faster than the code below

    mi::Uint32 tile_width   = old_tile->get_resolution_x();
    mi::Uint32 tile_height  = old_tile->get_resolution_y();
    mi::Size   nr_of_pixels = tile_width * tile_height;

    mi::neuraylib::ITile* new_tile = IMAGE::create_tile( new_pixel_type, tile_width, tile_height);
    if( !new_tile)
        return 0;

    const void* old_data = old_tile->get_data();
    void* new_data = new_tile->get_data();
    convert( old_data, new_data, old_pixel_type, new_pixel_type, nr_of_pixels);

    return new_tile;
}

void Image_module_impl::adjust_gamma(
    mi::neuraylib::ICanvas* canvas, mi::Float32 new_gamma) const
{
    if( !canvas || new_gamma <= 0.0f)
        return;

    Pixel_type pixel_type = convert_pixel_type_string_to_enum( canvas->get_type());
    if( pixel_type == PT_UNDEF)
        return;

    mi::Float32 old_gamma = canvas->get_gamma();
    if( old_gamma <= 0.0f || new_gamma == old_gamma)
        return;

    mi::Float32 exponent = old_gamma/new_gamma;

    mi::Uint32 tile_width    = canvas->get_tile_resolution_x();
    mi::Uint32 tile_height   = canvas->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = canvas->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = canvas->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    mi::Size   nr_of_pixels  = tile_width * tile_height;

    if( pixel_type == PT_COLOR || pixel_type == PT_RGB_FP) {

        mi::Uint32 components = pixel_type == PT_COLOR ? 4 : 3;
        for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
            for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
                for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                    mi::base::Handle<mi::neuraylib::ITile> tile(
                        canvas->get_tile( x*tile_width, y*tile_height, z));
                    mi::Float32* data = static_cast<mi::Float32*>( tile->get_data());
                    IMAGE::adjust_gamma( data, nr_of_pixels, components, exponent);
                }

    } else {

        mi::Float32* buffer = new mi::Float32[4*nr_of_pixels];
        for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
            for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
                for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                    mi::base::Handle<mi::neuraylib::ITile> tile(
                        canvas->get_tile( x*tile_width, y*tile_height, z));
                    void* data = tile->get_data();
                    convert( data, buffer, pixel_type, PT_COLOR, nr_of_pixels);
                    IMAGE::adjust_gamma( buffer, nr_of_pixels, 4, exponent);
                    convert( buffer, data, PT_COLOR, pixel_type, nr_of_pixels);
                }
        delete[] buffer;

    }

    canvas->set_gamma( new_gamma);
}

void Image_module_impl::serialize_mipmap(
    SERIAL::Serializer* serializer, const IMipmap* mipmap, bool only_first_level) const
{
    mi::Uint32 serialized_levels = only_first_level ? 1 : mipmap->get_nlevels();
    serializer->write( serialized_levels);

    for( mi::Uint32 i = 0; i < serialized_levels; ++i) {
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( i));
        serialize_canvas( serializer, canvas.get());
    }

    serializer->write( mipmap->get_is_cubemap());
}

IMipmap* Image_module_impl::deserialize_mipmap( SERIAL::Deserializer* deserializer) const
{
    mi::Uint32 deserialized_levels = 0;
    deserializer->read( &deserialized_levels);

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( deserialized_levels);
    for( mi::Uint32 i = 0; i < deserialized_levels; ++i)
        canvases[i] = deserialize_canvas( deserializer);

    bool is_cubemap;
    deserializer->read( &is_cubemap);

    return create_mipmap( canvases, is_cubemap);
}

void Image_module_impl::serialize_canvas(
    SERIAL::Serializer* serializer, const mi::neuraylib::ICanvas* canvas) const
{
    Pixel_type pixel_type = convert_pixel_type_string_to_enum( canvas->get_type());
    mi::Uint32 canvas_width  = canvas->get_resolution_x();
    mi::Uint32 canvas_height = canvas->get_resolution_y();
    mi::Uint32 tile_width    = canvas->get_tile_resolution_x();
    mi::Uint32 tile_height   = canvas->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = canvas->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = canvas->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    mi::Float32 gamma        = canvas->get_gamma();

    bool is_cubemap = get_canvas_is_cubemap( canvas);

    serializer->write( static_cast<mi::Uint32>( pixel_type));
    serializer->write( canvas_width);
    serializer->write( canvas_height);
    serializer->write( nr_of_layers);
    serializer->write( tile_width);
    serializer->write( tile_height);
    serializer->write( is_cubemap);
    serializer->write( gamma);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    mi::Size count = static_cast<mi::Size>( tile_width) * tile_height * bytes_per_pixel;

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
        for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
            for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                mi::base::Handle<const mi::neuraylib::ITile> tile(
                    canvas->get_tile( x*tile_width, y*tile_height, z));
                const void* tile_data = tile->get_data();
                serializer->write( static_cast<const char*>( tile_data), count);
            }
}

mi::neuraylib::ICanvas* Image_module_impl::deserialize_canvas(
    SERIAL::Deserializer* deserializer) const
{
    Pixel_type pixel_type;
    mi::Uint32 canvas_width;
    mi::Uint32 canvas_height;
    mi::Uint32 tile_width;
    mi::Uint32 tile_height;
    mi::Uint32 nr_of_tiles_x;
    mi::Uint32 nr_of_tiles_y;
    mi::Uint32 nr_of_layers;
    bool is_cubemap;
    mi::Float32 gamma;

    mi::Uint32 pixel_type_as_uint32;
    deserializer->read( &pixel_type_as_uint32);
    pixel_type = static_cast<Pixel_type>( pixel_type_as_uint32);
    deserializer->read( &canvas_width);
    deserializer->read( &canvas_height);
    deserializer->read( &nr_of_layers);
    deserializer->read( &tile_width);
    deserializer->read( &tile_height);
    deserializer->read( &is_cubemap);
    deserializer->read( &gamma);

    nr_of_tiles_x = (canvas_width  + tile_width  - 1) / tile_width;
    nr_of_tiles_y = (canvas_height + tile_height - 1) / tile_height;

    mi::neuraylib::ICanvas* canvas = create_canvas( pixel_type,
        canvas_width, canvas_height, tile_width, tile_height, nr_of_layers, is_cubemap, gamma);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    mi::Size count = static_cast<mi::Size>( tile_width) * tile_height * bytes_per_pixel;

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
        for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
            for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                mi::base::Handle<mi::neuraylib::ITile> tile(
                    canvas->get_tile( x*tile_width, y*tile_height, z));
                void* tile_data = tile->get_data();
                deserializer->read( static_cast<char*>( tile_data), count);
            }

    return canvas;
}

void Image_module_impl::serialize_tile(
    SERIAL::Serializer* serializer, const mi::neuraylib::ITile* tile) const
{
    Pixel_type pixel_type = convert_pixel_type_string_to_enum( tile->get_type());
    mi::Uint32 width  = tile->get_resolution_x();
    mi::Uint32 height = tile->get_resolution_y();

    serializer->write( static_cast<mi::Uint32>( pixel_type));
    serializer->write( width);
    serializer->write( height);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    mi::Size count = static_cast<mi::Size>( width) * height * bytes_per_pixel;
    const void* tile_data = tile->get_data();
    serializer->write( static_cast<const char*>( tile_data), count);
}

mi::neuraylib::ITile* Image_module_impl::deserialize_tile(
    SERIAL::Deserializer* deserializer) const
{
    Pixel_type pixel_type;
    mi::Uint32 width;
    mi::Uint32 height;

    mi::Uint32 pixel_type_as_uint32;
    deserializer->read( &pixel_type_as_uint32);
    pixel_type = static_cast<Pixel_type>( pixel_type_as_uint32);
    deserializer->read( &width);
    deserializer->read( &height);

    mi::neuraylib::ITile* tile = create_tile( pixel_type, width, height);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    mi::Size count = static_cast<mi::Size>( width) * height * bytes_per_pixel;
    void* tile_data = tile->get_data();
    deserializer->read( static_cast<char*>( tile_data), count);

    return tile;
}

bool Image_module_impl::export_canvas(
    const mi::neuraylib::ICanvas* canvas,
    const char* output_filename,
    mi::Uint32 quality) const
{
    std::string root, extension;
    HAL::Ospath::splitext( output_filename, root, extension);
    if( !extension.empty() && extension[0] == '.' )
        extension = extension.substr( 1);

    mi::neuraylib::IImage_plugin* plugin = find_plugin_for_export( extension.c_str());
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle \"%s\".", output_filename);
        return false;
    }

    const char* canvas_pixel_type = canvas->get_type();
    const char* export_pixel_type = find_best_pixel_type_for_export( canvas_pixel_type, plugin);
    if( !export_pixel_type) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" supports only the import, but not the export of images.",
            plugin->get_name());
        return false;
    }

    // If necessary, convert canvas to export_pixel_type
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas2;
    if( strcmp( canvas_pixel_type, export_pixel_type) != 0) {
        canvas2 = convert_canvas( canvas, convert_pixel_type_string_to_enum( export_pixel_type));
        ASSERT( M_IMAGE, canvas2.is_valid_interface());
    } else {
        canvas2 = make_handle_dup( canvas);
        canvas = 0; // only use canvas2 below
    }

    DISK::File_writer_impl writer;
    if( !writer.open( output_filename)) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "Failed to open image file \"%s\".", output_filename);
        return false;
    }

    mi::Uint32 image_width   = canvas2->get_resolution_x();
    mi::Uint32 image_height  = canvas2->get_resolution_y();
    mi::Uint32 tile_width    = canvas2->get_tile_resolution_x();
    mi::Uint32 tile_height   = canvas2->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = canvas2->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = canvas2->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = canvas2->get_layers_size();
    const char* pixel_type   = canvas2->get_type();
    mi::Float32 gamma        = canvas2->get_gamma();
    bool is_cubemap          = get_canvas_is_cubemap( canvas2.get());

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_writing( &writer,
        pixel_type, image_width, image_height, nr_of_layers, 1, is_cubemap, gamma, quality));
    if( !image_file.is_valid_interface()) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to export \"%s\" due to unsupported properties.",
            plugin->get_name(), output_filename);
        return false;
    }

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
        for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
            for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                mi::base::Handle<const mi::neuraylib::ITile> tile(
                    canvas2->get_tile( x*tile_width, y*tile_height, z));
                if( !image_file->write( tile.get(), x*tile_width, y*tile_height, z, 0)) {
                    LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                        "The image plugin \"%s\" failed to export \"%s\" due to unsupported "
                        "properties.", plugin->get_name(), output_filename);
                    return false;
                }
            }

    LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
        "Saving image \"%s\", pixel type \"%s\", %dx%dx%d pixels, 1 miplevel.",
        output_filename, export_pixel_type, image_width, image_height, nr_of_layers);

    return true;
}

bool Image_module_impl::export_mipmap(
    const IMipmap* mipmap,
    const char* output_filename,
    mi::Uint32 quality) const
{
    DISK::File_writer_impl writer;
    if( !writer.open( output_filename)) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "Failed to open image file \"%s\".", output_filename);
        return false;
    }

    std::string root, extension;
    HAL::Ospath::splitext( output_filename, root, extension);
    if( !extension.empty() && extension[0] == '.' )
        extension = extension.substr( 1);

    mi::neuraylib::IImage_plugin* plugin = find_plugin_for_export( extension.c_str());
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle \"%s\".", output_filename);
        return false;
    }

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    if( !canvas.is_valid_interface())
        return false;

    const char* canvas_pixel_type = canvas->get_type();
    const char* export_pixel_type = find_best_pixel_type_for_export( canvas_pixel_type, plugin);
    if( !export_pixel_type) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" supports only the import, but not the export of images.",
            plugin->get_name());
        return false;
    }

    mi::Uint32 image_width   = canvas->get_resolution_x();
    mi::Uint32 image_height  = canvas->get_resolution_y();
    mi::Uint32 tile_width    = canvas->get_tile_resolution_x();
    mi::Uint32 tile_height   = canvas->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = canvas->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = canvas->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    mi::Uint32 nr_of_levels  = mipmap->get_nlevels();
    mi::Float32 gamma        = canvas->get_gamma();
    bool is_cubemap          = get_canvas_is_cubemap( canvas.get());
    canvas = 0;

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_writing( &writer,
        export_pixel_type, image_width, image_height, nr_of_layers, nr_of_levels, is_cubemap, gamma,
        quality));
    if( !image_file.is_valid_interface()) {
        // if multiple levels are not supported try again exporting only the first level
        image_file = plugin->open_for_writing( &writer, export_pixel_type,
            image_width, image_height, nr_of_layers, 1, is_cubemap, gamma, quality);
        if( !image_file.is_valid_interface()) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to export \"%s\" due to unsupported properties.",
                plugin->get_name(), output_filename);
            return false;
        } else {
            nr_of_levels = 1;
        }
    }

    for( mi::Uint32 l = 0; l < nr_of_levels; ++l) {

        mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( l));
        ASSERT( M_IMAGE, canvas.is_valid_interface());
        if( !canvas.is_valid_interface())
            return false;

        tile_width    = canvas->get_tile_resolution_x();
        tile_height   = canvas->get_tile_resolution_y();
        nr_of_tiles_x = canvas->get_tiles_size_x();
        nr_of_tiles_y = canvas->get_tiles_size_y();
        nr_of_layers  = canvas->get_layers_size();

        // If necessary, convert canvas to export_pixel_type
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas2;
        if( strcmp( canvas_pixel_type, export_pixel_type) != 0) {
            canvas2 = convert_canvas( canvas.get(),
                convert_pixel_type_string_to_enum( export_pixel_type));
            ASSERT( M_IMAGE, canvas2.is_valid_interface());
        } else {
            canvas2 = make_handle_dup( canvas.get());
            canvas = 0; // only use canvas2 below
        }

        for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
            for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
                for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                    mi::base::Handle<const mi::neuraylib::ITile> tile(
                        canvas2->get_tile( x*tile_width, y*tile_height, z));
                    if( !image_file->write( tile.get(), x*tile_width, y*tile_height, z, l)) {
                        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                            "The image plugin \"%s\" failed to export \"%s\" due to unsupported "
                            "properties.", plugin->get_name(), output_filename);
                        return false;
                    }
            }
    }

    LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
        "Saving image \"%s\", pixel type \"%s\", %dx%dx%d pixels, %d miplevel%s.",
        output_filename, export_pixel_type, image_width, image_height, nr_of_layers,
        nr_of_levels, nr_of_levels == 1 ? "" : "s");

    return true;
}

mi::neuraylib::IBuffer* Image_module_impl::create_buffer_from_canvas(
    const mi::neuraylib::ICanvas* canvas,
    const char* image_format,
    const char* pixel_type,
    mi::Uint32 quality) const
{
    DISK::Memory_writer_impl writer;

    mi::neuraylib::IImage_plugin* plugin = find_plugin_for_export( image_format);
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle image format \"%s\".", image_format);
        return 0;
    }

    const char* export_pixel_type = find_best_pixel_type_for_export( pixel_type, plugin);
    if( !export_pixel_type) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" supports only the import, but not the export of images.",
            plugin->get_name());
        return 0;
    }

    // If necessary, convert canvas to export_pixel_type
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas2;
    if( strcmp( export_pixel_type, canvas->get_type()) != 0) {
        canvas2 = convert_canvas( canvas, convert_pixel_type_string_to_enum( export_pixel_type));
        ASSERT( M_IMAGE, canvas2.is_valid_interface());
    } else {
        canvas2 = make_handle_dup( canvas);
        canvas = 0; // only use canvas2 below
    }

    mi::Uint32 image_width   = canvas2->get_resolution_x();
    mi::Uint32 image_height  = canvas2->get_resolution_y();
    mi::Uint32 tile_width    = canvas2->get_tile_resolution_x();
    mi::Uint32 tile_height   = canvas2->get_tile_resolution_y();
    mi::Uint32 nr_of_tiles_x = canvas2->get_tiles_size_x();
    mi::Uint32 nr_of_tiles_y = canvas2->get_tiles_size_y();
    mi::Uint32 nr_of_layers  = canvas2->get_layers_size();
    mi::Float32 gamma        = canvas2->get_gamma();
    bool is_cubemap          = get_canvas_is_cubemap( canvas2.get());

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_writing( &writer,
        export_pixel_type, image_width, image_height, nr_of_layers, 1, is_cubemap, gamma, quality));
    if( !image_file.is_valid_interface()) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to encode the canvas due to unsupported properties.",
            plugin->get_name());
        return 0;
    }

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z)
        for( mi::Uint32 y = 0; y < nr_of_tiles_y; ++y)
            for( mi::Uint32 x = 0; x < nr_of_tiles_x; ++x) {
                mi::base::Handle<const mi::neuraylib::ITile> tile(
                    canvas2->get_tile( x*tile_width, y*tile_height, z));
                if( !image_file->write( tile.get(), x*tile_width, y*tile_height, z, 0)) {
                    LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                        "The image plugin \"%s\" failed to encode the canvas due to unsupported "
                        "properties.", plugin->get_name());
                    return 0;
                }
            }

    return writer.get_buffer();
}

mi::neuraylib::IImage_plugin* Image_module_impl::find_plugin_for_import(
    const char* extension, mi::neuraylib::IReader* reader) const
{
    mi::Uint8 buffer[512];
    mi::Sint64 file_size = 0;
    if( reader) {
        mi::Sint64 bytes_read
            = reader->read( static_cast<char*>( static_cast<void*>( &buffer[0])), 512);
        reader->rewind();
        file_size = reader->get_file_size();
        if( bytes_read != 512 && file_size >= 512)
            return 0;
    }

    std::priority_queue<mi::neuraylib::IImage_plugin*,
                        std::vector<mi::neuraylib::IImage_plugin*>, Plugin_less> queue;

    mi::base::Lock::Block block( &m_plugins_lock);
    for( mi::Size plugin_index = 0; plugin_index < m_plugins.size(); ++plugin_index) {

        mi::neuraylib::IImage_plugin* plugin
            = static_cast<mi::neuraylib::IImage_plugin*>( m_plugins[plugin_index]->get_plugin());
        mi::Uint32 extension_index = 0;
        const char* plugin_extension = plugin->get_file_extension( extension_index);

        while( plugin_extension) {
            if( !extension || STRING::compare_case_insensitive( extension, plugin_extension) == 0) {
                if( !reader || plugin->test( buffer, static_cast<mi::Uint32>( file_size)))
                    queue.push( plugin);
            }
            plugin_extension = extension ? plugin->get_file_extension( ++extension_index) : 0;
        }
    }

    if( queue.empty())
        return 0;

    return queue.top();
}

mi::neuraylib::IImage_plugin* Image_module_impl::find_plugin_for_export(
    const char* extension) const
{
    if( !extension)
        return 0;

    std::priority_queue<mi::neuraylib::IImage_plugin*,
                        std::vector<mi::neuraylib::IImage_plugin*>, Plugin_less> queue;

    mi::base::Lock::Block block( &m_plugins_lock);
    for( mi::Size plugin_index = 0; plugin_index < m_plugins.size(); ++plugin_index) {

        mi::neuraylib::IImage_plugin* plugin
            = static_cast<mi::neuraylib::IImage_plugin*>( m_plugins[plugin_index]->get_plugin());
        mi::Uint32 extension_index = 0;
        const char* plugin_extension = plugin->get_file_extension( extension_index);

        while( plugin_extension) {
            if( STRING::compare_case_insensitive( extension, plugin_extension) == 0) {
                if( plugin->get_supported_type( 0) != 0)
                    queue.push( plugin);
            }
            plugin_extension = plugin->get_file_extension( ++extension_index);
        }
    }

    if( queue.empty())
        return 0;

    return queue.top();
}

void Image_module_impl::set_mdr_callback( IMdr_callback* mdr_callback)
{
    m_mdr_callback = make_handle_dup( mdr_callback);
}

IMdr_callback* Image_module_impl::get_mdr_callback() const
{
    if( !m_mdr_callback)
        return 0;

    m_mdr_callback->retain();
    return m_mdr_callback.get();
}

void Image_module_impl::dump() const
{
    mi::Size i = 0;

    mi::base::Lock::Block block( &m_plugins_lock);
    Plugin_vector::const_iterator it     = m_plugins.begin();
    Plugin_vector::const_iterator it_end = m_plugins.end();

    // Dump list of image plugins with extensions and pixel types for export
    for( ; it != it_end; ++it) {

        mi::neuraylib::IImage_plugin* plugin
            = static_cast<mi::neuraylib::IImage_plugin*>( (*it)->get_plugin());

        std::ostringstream line;
        line << "plugin " << i << ": name \"" << plugin->get_name() << "\", ";

        line << "priority " << plugin->get_priority() << ", ";

        line << "file extensions: ";
        mi::Uint32 j = 0;
        const char* file_extension = plugin->get_file_extension( j);
        while( file_extension) {
            line << (j>0 ? ", ": "") << "\"." << file_extension << "\"";
            file_extension = plugin->get_file_extension( ++j);
        }

        line << ", supported pixel types (export): ";
        mi::Uint32 k = 0;
        const char* supported_type = plugin->get_supported_type( k);
        while( supported_type) {
            line << (k>0 ? ", ": "") << "\"" << supported_type << "\"";
            supported_type = plugin->get_supported_type( ++k);
        }

        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, "%s", line.str().c_str());
        i++;
    }

    // Dump pixel type conversion priorities (essentially rows of table g_cost sorted by cost)
    const mi::Uint32 first_pixel_type =  1;
    const mi::Uint32 last_pixel_type  = 14;

    for( mi::Uint32 from = first_pixel_type; from <= last_pixel_type; ++from) {
        Pixel_type from_enum = static_cast<Pixel_type>( from);
        std::ostringstream s;
        s << "conversion priorities for ";
        s << std::left << std::setw( 11) << convert_pixel_type_enum_to_string( from_enum);

        std::vector<mi::Float32> cost( last_pixel_type+1);
        for( mi::Uint32 to = first_pixel_type; to <= last_pixel_type; ++to) {
            Pixel_type to_enum = static_cast<Pixel_type>( to);
            cost[to] = get_conversion_cost( from_enum, to_enum);
        }

        mi::Float32 last_cost = 0.0;
        for( mi::Uint32 i = first_pixel_type; i <= last_pixel_type; ++i) {

            mi::Float32 min_cost = std::numeric_limits<mi::Float32>::max();
            mi::Uint32 min_index = static_cast<mi::Uint32>( -1);
            for( mi::Uint32 j = first_pixel_type; j <= last_pixel_type; ++j) {
                if( cost[j] < min_cost) {
                    min_cost = cost[j];
                    min_index = j;
                }
            }
            ASSERT( M_IMAGE, min_index != static_cast<mi::Uint32>( -1));
            ASSERT( M_IMAGE, min_cost >= last_cost);

            Pixel_type to_enum = static_cast<Pixel_type>( min_index);
            if( i > 1)
                s << ((min_cost > last_cost) ? ", " : "/");
            s << convert_pixel_type_enum_to_string( to_enum);
            last_cost = min_cost;
            cost[min_index] = std::numeric_limits<mi::Float32>::max();
        }
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, "%s", s.str().c_str());
    }
}

bool Image_module_impl::is_valid_image_plugin(
    const char* type, const char* name, const char* filename)
{
    if( !type)
        return false;
    // current version
    if( 0 == strcmp( type, MI_NEURAY_IMAGE_PLUGIN_TYPE))
        return true;
    // previous versions that might still work
    if( false /*(0 == strcmp( type, "version_goes_here"))*/) {
        LOG::mod_log->warning( M_IMAGE, LOG::Mod_log::C_PLUGIN,
            "Image plugin of name \"%s\" from library \"%s\" has different plugin type "
            "\"%s\". If you encounter problems with this plugin, you may want to use a "
            "version of the plugin that has been compiled for the currently supported "
            "plugin type \"%s\".",
            name, filename, type, MI_NEURAY_IMAGE_PLUGIN_TYPE);
        return true;
    }
    // unsupported versions (previous or future)
    if( (0 == strncmp( type, "image", 5))) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_PLUGIN,
            "Image plugin of name \"%s\" from library \"%s\" has unsupported plugin type "
            "\"%s\". Please use a version of the plugin that has been compiled for the "
            "currently supported plugin type \"%s\".",
            name, filename, type, MI_NEURAY_IMAGE_PLUGIN_TYPE);
        return false;
    }
    // other types
    return false;
}

namespace {

mi::Float32 g_cost[14][14] = {
    {  0,  3,  4,  8, 10, 12,  1,  2,  5,  6,  7,  9, 11, 13 },
    { 13,  0,  2,  3,  4,  6,  8,  1,  9, 10, 11, 12,  5,  7 },
    { 13, 12,  0,  1,  2,  4, 10, 11,  8,  9,  6,  7,  3,  5 },
    { 13, 11, 12,  0,  1,  3,  9, 10,  7,  8,  5,  6,  2,  4 },
    { 13, 10, 12, 11,  0,  2,  8,  9,  6,  7,  4,  5,  1,  3 },
    { 13,  5, 12, 11, 10,  0,  9,  4,  8,  3,  7,  2,  6,  1 },
    { 13,  2, 12, 11,  8, 10,  0,  1,  3,  4,  5,  6,  7,  9 },
    { 13,  1, 12, 11, 10,  5,  6,  0,  7,  2,  8,  3,  9,  4 },
    { 13, 10, 12, 11,  5,  7,  8,  9,  0,  1,  2,  3,  4,  6 },
    { 13,  5, 12, 11,  9,  3, 10,  4,  6,  0,  7,  1,  8,  2 },
    { 13, 10, 12, 11,  3,  5,  8,  9,  6,  7,  0,  1,  2,  4 },
    { 13,  5, 12, 11,  8,  2, 10,  4,  9,  3,  6,  0,  7,  1 },
    { 13, 10, 12, 11,  1,  3,  8,  9,  6,  7,  4,  5,  0,  2 },
    { 13,  5, 12, 11,  7,  1, 10,  4,  9,  3,  8,  2,  6,  0 }
};

} // namespace

mi::Float32 Image_module_impl::get_conversion_cost( Pixel_type from, Pixel_type to)
{
    ASSERT( M_IMAGE, from != PT_UNDEF && to != PT_UNDEF);
    ASSERT( M_IMAGE, from <= PT_COLOR && to <= PT_COLOR);

    // The table above heavily depends on the actual values of the enums.
    ASSERT( M_IMAGE, PT_SINT8     ==  1);
    ASSERT( M_IMAGE, PT_SINT32    ==  2);
    ASSERT( M_IMAGE, PT_FLOAT32   ==  3);
    ASSERT( M_IMAGE, PT_FLOAT32_2 ==  4);
    ASSERT( M_IMAGE, PT_FLOAT32_3 ==  5);
    ASSERT( M_IMAGE, PT_FLOAT32_4 ==  6);
    ASSERT( M_IMAGE, PT_RGB       ==  7);
    ASSERT( M_IMAGE, PT_RGBA      ==  8);
    ASSERT( M_IMAGE, PT_RGBE      ==  9);
    ASSERT( M_IMAGE, PT_RGBEA     == 10);
    ASSERT( M_IMAGE, PT_RGB_16    == 11);
    ASSERT( M_IMAGE, PT_RGBA_16   == 12);
    ASSERT( M_IMAGE, PT_RGB_FP    == 13);
    ASSERT( M_IMAGE, PT_COLOR     == 14);

    return g_cost[from-1][to-1];
}

const char* Image_module_impl::find_best_pixel_type_for_export(
    const char* pixel_type,
    mi::neuraylib::IImage_plugin* plugin)
{
    Pixel_type canvas_pixel_type = convert_pixel_type_string_to_enum( pixel_type);
    ASSERT( M_IMAGE, canvas_pixel_type != PT_UNDEF);

    Pixel_type min_pixel_type = PT_UNDEF;
    mi::Float32 min_ratio = 100.0f;

    mi::Uint32 index = 0;
    Pixel_type plugin_pixel_type
        = convert_pixel_type_string_to_enum( plugin->get_supported_type( index));

    while( plugin_pixel_type != PT_UNDEF) {

        mi::Float32 ratio = get_conversion_cost( canvas_pixel_type, plugin_pixel_type);
        if( ratio < min_ratio) {
            min_ratio = ratio;
            min_pixel_type = plugin_pixel_type;
        }
        plugin_pixel_type
            = convert_pixel_type_string_to_enum( plugin->get_supported_type( ++index));
    }

    return convert_pixel_type_enum_to_string( min_pixel_type);
}

bool Image_module_impl::get_canvas_is_cubemap( const mi::neuraylib::ICanvas* canvas)
{
    mi::base::Handle<const ICanvas> canvas_internal( canvas->get_interface<ICanvas>());
    return canvas_internal && canvas_internal->get_is_cubemap();
}

} // namespace IMAGE

} // namespace MI
