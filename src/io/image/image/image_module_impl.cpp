/***************************************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/handle.h>
#include <mi/neuraylib/iimage_plugin.h>
#include <mi/neuraylib/iplugin_api.h>
#include <mi/math/color.h>

#include <cstring>
#include <iomanip>
#include <limits>
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

#include "i_image_pixel_conversion.h"
#include "i_image_utilities.h"
#include "image_canvas_impl.h"
#include "image_image_api_impl.h"
#include "image_mipmap_impl.h"
#include "image_tile_impl.h"


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

    // If no plugin API has been registered, e.g., in some unit tests, then we provide our own
    // which at least provides access to most of IImage_api.
    if( !plugin_api)
        plugin_api = new Plugin_api_impl( this);

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

    // If no plugin API has been registered, e.g., in some unit tests, then we provide our own
    // which at least provides access to most of IImage_api.
    if( !plugin_api)
        plugin_api = new Plugin_api_impl( this);

    // Call IImage_plugin::exit() on our type of plugins
    mi::base::Lock::Block block( &m_plugins_lock);
          Plugin_vector::reverse_iterator it     = m_plugins.rbegin();
    const Plugin_vector::reverse_iterator it_end = m_plugins.rend();
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
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    if( pixel_type == PT_UNDEF || width == 0 || height == 0 || layers == 0 || gamma < 0)
        return nullptr;

    if( is_cubemap && layers != 6)
        return nullptr;

    try {
        return new Mipmap_impl( pixel_type, width, height, layers, is_cubemap, gamma);
    } catch( const std::bad_alloc& e) {
        return nullptr;
    }
}

IMipmap* Image_module_impl::create_mipmap(
    const std::string& filename,
    const char* selector,
    bool only_first_level,
    mi::Sint32* errors) const
{
    return new Mipmap_impl( filename, selector, only_first_level, errors);
}

IMipmap* Image_module_impl::create_mipmap(
    Container_based,
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& member_filename,
    const char* selector,
    bool only_first_level,
    mi::Sint32* errors) const
{
    return new Mipmap_impl(
        Container_based(),
        reader,
        archive_filename,
        member_filename,
        selector,
        only_first_level,
        errors);
}

IMipmap* Image_module_impl::create_mipmap(
    Memory_based,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* selector,
    const char* mdl_file_path,
    bool only_first_level,
    mi::Sint32* errors) const
{
    return new Mipmap_impl(
        Memory_based(),
        reader,
        image_format,
        selector,
        mdl_file_path,
        only_first_level,
        errors);
}

IMipmap* Image_module_impl::create_mipmap(
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> >& canvases, bool is_cubemap) const
{
    const mi::Size count = canvases.size();
    if( count == 0)
        return nullptr;

    for( mi::Uint32 i = 0; i < count; ++i)
        if( !canvases[i])
            return nullptr;

    return new Mipmap_impl( canvases, is_cubemap);
}


void Image_module_impl::create_mipmaps(
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> >& mipmaps,
    const mi::neuraylib::ICanvas* base_canvas,
    mi::Float32 gamma) const
{
    const mi::Uint32 w = base_canvas->get_resolution_x();
    const mi::Uint32 h = base_canvas->get_resolution_y();

    if (w == 1 || h == 1)
        return;

    const mi::Uint32 nr_of_levels = mi::math::log2_int(std::min(w, h));
    mipmaps.resize(nr_of_levels);

    const mi::neuraylib::ICanvas* prev_canvas = base_canvas;
    for (mi::Size i = 0; i < nr_of_levels; ++i)
    {
        mipmaps[i] = mi::base::make_handle(create_miplevel(prev_canvas, gamma));
        prev_canvas = mipmaps[i].get();
    }
}

IMipmap* Image_module_impl::create_dummy_mipmap()
{
    return new Mipmap_impl();
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma) const
{
    if( pixel_type == PT_UNDEF || width == 0 || height == 0 || layers == 0 || gamma < 0)
        return nullptr;

    if( is_cubemap && layers != 6)
        return nullptr;

    try {
        return new Canvas_impl( pixel_type, width, height, layers, is_cubemap, gamma);
    } catch( const std::bad_alloc& e) {
        return nullptr;
    }
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    const std::string& filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::Sint32* errors) const
{
    return new Canvas_impl(
        filename, selector, miplevel, /*image_file*/ nullptr, errors);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    Container_based,
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& member_filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::Sint32* errors) const
{
    return new Canvas_impl(
        Container_based(),
        reader,
        archive_filename,
        member_filename,
        selector,
        miplevel,
        /*image_file*/ nullptr,
        errors);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    Memory_based,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* selector,
    const char* mdl_file_path,
    mi::Uint32 miplevel,
    mi::Sint32* errors) const
{
    if( !reader || !image_format)
        return nullptr;

    return new Canvas_impl(
        Memory_based(),
        reader,
        image_format,
        selector,
        mdl_file_path,
        miplevel,
        /*image_file*/ nullptr,
        errors);
}

mi::neuraylib::ICanvas* Image_module_impl::create_canvas(
    const std::vector<mi::base::Handle<mi::neuraylib::ITile>>& tiles, mi::Float32 gamma) const
{
    if( gamma < 0)
        return nullptr;

    return new Canvas_impl( tiles, gamma);
}

mi::neuraylib::ITile* Image_module_impl::create_tile(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height) const
{
    if( width == 0 || height == 0 || pixel_type == PT_UNDEF)
        return nullptr;

    try {
        return IMAGE::create_tile( pixel_type, width, height);
    } catch( const std::bad_alloc& e) {
        return nullptr;
    }
}

IMipmap* Image_module_impl::copy_mipmap( const IMipmap* other, bool only_first_level) const
{
    const mi::Uint32 levels = only_first_level ? 1 : other->get_nlevels();

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( levels);
    for( mi::Uint32 i = 0; i < levels; ++i) {
        mi::base::Handle<const mi::neuraylib::ICanvas> other_canvas( other->get_level( i));
        canvases[i] = copy_canvas( other_canvas.get());
    }

    return create_mipmap( canvases, other->get_is_cubemap());
}

mi::neuraylib::ICanvas* Image_module_impl::copy_canvas( const mi::neuraylib::ICanvas* other) const
{
    const Pixel_type pixel_type    = convert_pixel_type_string_to_enum( other->get_type());
    if( pixel_type == PT_UNDEF)
        return nullptr;
    const mi::Uint32 canvas_width  = other->get_resolution_x();
    const mi::Uint32 canvas_height = other->get_resolution_y();
    const mi::Uint32 nr_of_layers  = other->get_layers_size();
    const mi::Float32 gamma        = other->get_gamma();

    const bool is_cubemap = get_canvas_is_cubemap( other);
    mi::neuraylib::ICanvas* const canvas = create_canvas( pixel_type,
        canvas_width, canvas_height, nr_of_layers, is_cubemap, gamma);
    if (!canvas)
        return nullptr;

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    const mi::Size count = static_cast<mi::Size>( canvas_width) * canvas_height * bytes_per_pixel;

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> other_tile( other->get_tile( z));
        mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( z));
        const void* const other_tile_data = other_tile->get_data();
        void* const tile_data = tile->get_data();
        memcpy( tile_data, other_tile_data, count);
    }

    return canvas;
}

mi::neuraylib::ITile* Image_module_impl::copy_tile( const mi::neuraylib::ITile* other) const
{
    const Pixel_type pixel_type = convert_pixel_type_string_to_enum( other->get_type());
    const mi::Uint32 width  = other->get_resolution_x();
    const mi::Uint32 height = other->get_resolution_y();
    if( width == 0 || height == 0 || pixel_type == PT_UNDEF)
        return nullptr;

    mi::neuraylib::ITile* tile = create_tile( pixel_type, width, height);

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    const mi::Size count = static_cast<mi::Size>( width) * height * bytes_per_pixel;
    const void* const other_tile_data = other->get_data();
    void* const tile_data = tile->get_data();
    memcpy( tile_data, other_tile_data, count);

    return tile;
}

mi::Sint32 Image_module_impl::copy_mipmap_data( const IMipmap* source, IMipmap* dest) const
{
    const mi::Uint32 source_levels  = source->get_nlevels();
    const mi::Uint32 dest_levels = dest->get_nlevels();
    if( dest_levels != source_levels)
        return -1;

    for( mi::Uint32 i = 0; i < source_levels; ++i) {
        mi::base::Handle<const mi::neuraylib::ICanvas> source_canvas( source->get_level( i));
        mi::base::Handle<mi::neuraylib::ICanvas> dest_canvas( dest->get_level( i));
        const mi::Sint32 result = copy_canvas_data( source_canvas.get(), dest_canvas.get());
        if( result != 0)
            return result;
    }

    return 0;
}

mi::Sint32 Image_module_impl::copy_canvas_data(
    const mi::neuraylib::ICanvas* source, mi::neuraylib::ICanvas* dest) const
{
    const mi::Uint32 source_nr_of_layers  = source->get_layers_size();
    const mi::Float32 source_gamma        = source->get_gamma();
    // is_cubemap does not really matter and we might not be able to detect it reliably

    const mi::Uint32 dest_nr_of_layers  = dest->get_layers_size();
    const mi::Float32 dest_gamma        = dest->get_gamma();
    // is_cubemap does not really matter and we might not be able to detect it reliably

    if(    dest_nr_of_layers  != source_nr_of_layers
        || dest_gamma         != source_gamma)
        return -1;

    for( mi::Uint32 z = 0; z < source_nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> source_tile( source->get_tile( z));
        mi::base::Handle<mi::neuraylib::ITile> dest_tile( dest->get_tile( z));
        if (const int res = copy_tile_data( source_tile.get(), dest_tile.get()))
            return res;
    }

    return 0;
}

mi::Sint32 Image_module_impl::copy_tile_data(
    const mi::neuraylib::ITile* source, mi::neuraylib::ITile* dest) const
{
    const Pixel_type source_pixel_type = convert_pixel_type_string_to_enum( source->get_type());
    if( source_pixel_type == PT_UNDEF)
        return -2;
    const mi::Uint32 source_width  = source->get_resolution_x();
    const mi::Uint32 source_height = source->get_resolution_y();

    const Pixel_type dest_pixel_type = convert_pixel_type_string_to_enum( dest->get_type());
    if( dest_pixel_type == PT_UNDEF)
        return -2;
    const mi::Uint32 dest_width  = dest->get_resolution_x();
    const mi::Uint32 dest_height = dest->get_resolution_y();

    if(    dest_pixel_type != source_pixel_type
        || dest_width      != source_width
        || dest_height     != source_height)
        return -1;

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( source_pixel_type);
    const mi::Size count = static_cast<mi::Size>( source_width) * source_height * bytes_per_pixel;

    const void* const source_data = source->get_data();
    void* const dest_data         = dest->get_data();
    memcpy( dest_data, source_data, count);

    return 0;
}

IMipmap* Image_module_impl::convert_mipmap(
    const IMipmap* old_mipmap, Pixel_type new_pixel_type, bool only_first_level) const
{
    if( !old_mipmap)
        return nullptr;

    const mi::Uint32 converted_levels = only_first_level ? 1 : old_mipmap->get_nlevels();

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
        return nullptr;

    const Pixel_type old_pixel_type = convert_pixel_type_string_to_enum( old_canvas->get_type());
    if( old_pixel_type == PT_UNDEF)
        return nullptr;

    if( old_pixel_type == new_pixel_type)
        return copy_canvas( old_canvas); // faster than the code below

    const mi::Uint32 canvas_width  = old_canvas->get_resolution_x();
    const mi::Uint32 canvas_height = old_canvas->get_resolution_y();
    const mi::Uint32 nr_of_layers  = old_canvas->get_layers_size();
    const mi::Size   nr_of_pixels  = canvas_width * canvas_height;
    const mi::Float32 gamma        = old_canvas->get_gamma();

    const bool is_cubemap = get_canvas_is_cubemap( old_canvas);
    mi::neuraylib::ICanvas* new_canvas = new Canvas_impl( new_pixel_type,
        canvas_width, canvas_height, nr_of_layers, is_cubemap, gamma);

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> old_tile( old_canvas->get_tile( z));
        mi::base::Handle<mi::neuraylib::ITile> new_tile( new_canvas->get_tile( z));
        const void* const old_data = old_tile->get_data();
        void* const new_data = new_tile->get_data();
        convert( old_data, new_data, old_pixel_type, new_pixel_type, nr_of_pixels);
    }

    return new_canvas;
}

mi::neuraylib::ITile* Image_module_impl::convert_tile(
    const mi::neuraylib::ITile* old_tile, Pixel_type new_pixel_type) const
{
    if( !old_tile || new_pixel_type == PT_UNDEF)
        return nullptr;

    const Pixel_type old_pixel_type = convert_pixel_type_string_to_enum( old_tile->get_type());
    if( old_pixel_type == PT_UNDEF)
        return nullptr;

    if( old_pixel_type == new_pixel_type)
        return copy_tile( old_tile); // faster than the code below

    const mi::Uint32 tile_width   = old_tile->get_resolution_x();
    const mi::Uint32 tile_height  = old_tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * tile_height;

    mi::neuraylib::ITile* new_tile = IMAGE::create_tile( new_pixel_type, tile_width, tile_height);
    if( !new_tile)
        return nullptr;

    const void* const old_data = old_tile->get_data();
    void* const new_data = new_tile->get_data();
    convert( old_data, new_data, old_pixel_type, new_pixel_type, nr_of_pixels);

    return new_tile;
}

void Image_module_impl::adjust_gamma(
    mi::neuraylib::ICanvas* canvas, mi::Float32 new_gamma) const
{
    if( !canvas || new_gamma <= 0.0f)
        return;

    const Pixel_type pixel_type = convert_pixel_type_string_to_enum( canvas->get_type());
    const mi::Float32 old_gamma = canvas->get_gamma();
    if( old_gamma <= 0.0f || new_gamma == old_gamma)
        return;

    const mi::Float32 exponent = old_gamma/new_gamma;

    const mi::Uint32 canvas_width  = canvas->get_resolution_x();
    const mi::Uint32 canvas_height = canvas->get_resolution_y();
    const mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    const mi::Size   nr_of_pixels  = canvas_width * canvas_height;

    switch (pixel_type) {
        case PT_COLOR:
        case PT_RGB_FP:
        case PT_FLOAT32:
        case PT_FLOAT32_2:
        case PT_FLOAT32_3:
        case PT_FLOAT32_4: {
            const mi::Uint32 components = get_components_per_pixel(pixel_type);
            for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
                mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( z));
                ASSERT(M_IMAGE, tile->get_resolution_x() == canvas_width);
                ASSERT(M_IMAGE, tile->get_resolution_y() == canvas_height);
                mi::Float32* data = static_cast<mi::Float32*>( tile->get_data());
                IMAGE::adjust_gamma( data, nr_of_pixels, components, exponent);
            }
            break;
        }
        case PT_SINT8:  case PT_RGB:    case PT_RGBA:
        case PT_SINT32: case PT_RGB_16: case PT_RGBA_16:
            LOG::mod_log->warning(M_IMAGE, LOG::Mod_log::C_IO,
                "Adjusting gamma for pixel type \"%s\", which can lead to banding/quantization"
                "artifacts.", canvas->get_type());
            // fallthrough
        case PT_RGBE:
        case PT_RGBEA: {
            std::vector<mi::Float32> buffer(4*nr_of_pixels);
            for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
                mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( z));
                void* data = tile->get_data();
                convert( data, buffer.data(), pixel_type, PT_COLOR, nr_of_pixels);
                IMAGE::adjust_gamma( buffer.data(), nr_of_pixels, 4, exponent);
                convert( buffer.data(), data, PT_COLOR, pixel_type, nr_of_pixels);
            }
            break;
        }
        case PT_UNDEF:
            return;
    }

    canvas->set_gamma( new_gamma);
}

Pixel_type Image_module_impl::get_pixel_type_for_channel(
    Pixel_type pixel_type, const char* selector) const
{
    return IMAGE::get_pixel_type_for_channel( pixel_type, selector);
}

IMipmap* Image_module_impl::extract_channel(
    const IMipmap* old_mipmap, const char* selector, bool only_first_level) const
{
    if( !old_mipmap)
        return nullptr;

    const mi::Uint32 converted_levels = only_first_level ? 1 : old_mipmap->get_nlevels();

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > new_canvases( converted_levels);
    for( mi::Uint32 i = 0; i < converted_levels; ++i) {
        mi::base::Handle<const mi::neuraylib::ICanvas> old_canvas(
            old_mipmap->get_level( i));
        new_canvases[i] = extract_channel( old_canvas.get(), selector);
    }

    return create_mipmap( new_canvases, old_mipmap->get_is_cubemap());
}

mi::neuraylib::ICanvas* Image_module_impl::extract_channel(
    const mi::neuraylib::ICanvas* old_canvas, const char* selector) const
{
    if( !old_canvas)
        return nullptr;

    Pixel_type old_pixel_type = convert_pixel_type_string_to_enum( old_canvas->get_type());
    const Pixel_type new_pixel_type = get_pixel_type_for_channel( old_pixel_type, selector);
    if( new_pixel_type == PT_UNDEF)
        return nullptr;

    mi::base::Handle<mi::neuraylib::ICanvas> tmp_canvas;
    if( old_pixel_type == PT_RGBE || old_pixel_type == PT_RGB_16) {
        tmp_canvas = convert_canvas( old_canvas, PT_RGB_FP);
        old_canvas = tmp_canvas.get();
        old_pixel_type = PT_RGB_FP;
    } else if( old_pixel_type == PT_RGBEA || old_pixel_type == PT_RGBA_16) {
        tmp_canvas = convert_canvas( old_canvas, PT_COLOR);
        old_canvas = tmp_canvas.get();
        old_pixel_type = PT_COLOR;
    }

    const mi::Uint32 canvas_width  = old_canvas->get_resolution_x();
    const mi::Uint32 canvas_height = old_canvas->get_resolution_y();
    const mi::Uint32 nr_of_layers  = old_canvas->get_layers_size();
    const mi::Size   nr_of_pixels  = canvas_width * canvas_height;
    const mi::Float32 gamma        = old_canvas->get_gamma();

    const bool is_cubemap = get_canvas_is_cubemap( old_canvas);
    mi::neuraylib::ICanvas* new_canvas = new Canvas_impl( new_pixel_type,
        canvas_width, canvas_height, nr_of_layers, is_cubemap, gamma);

    const mi::Size channel_index = get_channel_index( selector);

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> old_tile( old_canvas->get_tile( z));
        mi::base::Handle<mi::neuraylib::ITile> new_tile( new_canvas->get_tile( z));
        const char* const old_data = static_cast<const char*>( old_tile->get_data());
        char* const new_data = static_cast<char*>( new_tile->get_data());
        IMAGE::extract_channel( old_data, new_data, nr_of_pixels, old_pixel_type, channel_index);
    }

    return new_canvas;
}

mi::neuraylib::ITile* Image_module_impl::extract_channel(
    const mi::neuraylib::ITile* old_tile, const char* selector) const
{
    if( !old_tile)
        return nullptr;

    Pixel_type old_pixel_type = convert_pixel_type_string_to_enum( old_tile->get_type());
    const Pixel_type new_pixel_type = get_pixel_type_for_channel( old_pixel_type, selector);
    if( new_pixel_type == PT_UNDEF)
        return nullptr;

    mi::base::Handle<mi::neuraylib::ITile> tmp_tile;
    if( old_pixel_type == PT_RGBE || old_pixel_type == PT_RGB_16) {
        tmp_tile = convert_tile( old_tile, PT_RGB_FP);
        old_tile = tmp_tile.get();
        old_pixel_type = PT_RGB_FP;
    } else if( old_pixel_type == PT_RGBEA || old_pixel_type == PT_RGBA_16) {
        tmp_tile = convert_tile( old_tile, PT_COLOR);
        old_tile = tmp_tile.get();
        old_pixel_type = PT_COLOR;
    }

    const mi::Uint32 tile_width   = old_tile->get_resolution_x();
    const mi::Uint32 tile_height  = old_tile->get_resolution_y();
    const mi::Size   nr_of_pixels = tile_width * tile_height;

    mi::neuraylib::ITile* new_tile = IMAGE::create_tile( new_pixel_type, tile_width, tile_height);
    if( !new_tile)
        return nullptr;

    const mi::Size channel_index = get_channel_index( selector);
    const char* const old_data = static_cast<const char*>( old_tile->get_data());
    char* const new_data = static_cast<char*>( new_tile->get_data());
    IMAGE::extract_channel( old_data, new_data, nr_of_pixels, old_pixel_type, channel_index);

    return new_tile;
}

void Image_module_impl::serialize_mipmap(
    SERIAL::Serializer* serializer, const IMipmap* mipmap, bool only_first_level) const
{
    const mi::Uint32 serialized_levels = only_first_level ? 1 : mipmap->get_nlevels();
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
    const Pixel_type pixel_type = convert_pixel_type_string_to_enum( canvas->get_type());
    const mi::Uint32 canvas_width  = canvas->get_resolution_x();
    const mi::Uint32 canvas_height = canvas->get_resolution_y();
    const mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    const mi::Float32 gamma        = canvas->get_gamma();

    const bool is_cubemap = get_canvas_is_cubemap( canvas);

    serializer->write( static_cast<mi::Uint32>( pixel_type));
    serializer->write( canvas_width);
    serializer->write( canvas_height);
    serializer->write( nr_of_layers);
    serializer->write( is_cubemap);
    serializer->write( gamma);

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    const mi::Size count = static_cast<mi::Size>( canvas_width) * canvas_height * bytes_per_pixel;

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> tile( canvas->get_tile( z));
        const void* const tile_data = tile->get_data();
        serializer->write( static_cast<const char*>( tile_data), count);
    }
}

mi::neuraylib::ICanvas* Image_module_impl::deserialize_canvas(
    SERIAL::Deserializer* deserializer) const
{
    mi::Uint32 canvas_width;
    mi::Uint32 canvas_height;
    mi::Uint32 nr_of_layers;
    bool is_cubemap;
    mi::Float32 gamma;

    mi::Uint32 pixel_type_as_uint32;
    deserializer->read( &pixel_type_as_uint32);
    const Pixel_type pixel_type = static_cast<Pixel_type>( pixel_type_as_uint32);
    deserializer->read( &canvas_width);
    deserializer->read( &canvas_height);
    deserializer->read( &nr_of_layers);
    deserializer->read( &is_cubemap);
    deserializer->read( &gamma);

    mi::neuraylib::ICanvas* canvas = create_canvas( pixel_type,
        canvas_width, canvas_height, nr_of_layers, is_cubemap, gamma);

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    const mi::Size count = static_cast<mi::Size>( canvas_width) * canvas_height * bytes_per_pixel;

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( z));
        void* const tile_data = tile->get_data();
        deserializer->read( static_cast<char*>( tile_data), count);
    }

    return canvas;
}

void Image_module_impl::serialize_tile(
    SERIAL::Serializer* serializer, const mi::neuraylib::ITile* tile) const
{
    const Pixel_type pixel_type = convert_pixel_type_string_to_enum( tile->get_type());
    const mi::Uint32 width  = tile->get_resolution_x();
    const mi::Uint32 height = tile->get_resolution_y();

    serializer->write( static_cast<mi::Uint32>( pixel_type));
    serializer->write( width);
    serializer->write( height);

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    const mi::Size count = static_cast<mi::Size>( width) * height * bytes_per_pixel;
    const void* const tile_data = tile->get_data();
    serializer->write( static_cast<const char*>( tile_data), count);
}

mi::neuraylib::ITile* Image_module_impl::deserialize_tile(
    SERIAL::Deserializer* deserializer) const
{
    mi::Uint32 width;
    mi::Uint32 height;

    mi::Uint32 pixel_type_as_uint32;
    deserializer->read( &pixel_type_as_uint32);
    const Pixel_type pixel_type = static_cast<Pixel_type>( pixel_type_as_uint32);
    deserializer->read( &width);
    deserializer->read( &height);

    mi::neuraylib::ITile* tile = create_tile( pixel_type, width, height);

    const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
    const mi::Size count = static_cast<mi::Size>( width) * height * bytes_per_pixel;
    void* const tile_data = tile->get_data();
    deserializer->read( static_cast<char*>( tile_data), count);

    return tile;
}

bool Image_module_impl::export_canvas(
    const mi::neuraylib::ICanvas* canvas_ptr,
    const char* output_filename,
    mi::Uint32 quality,
    bool force_default_gamma) const
{
    // Wrap incoming canvas pointer into handle for easier management later.
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( canvas_ptr, mi::base::DUP_INTERFACE);
    canvas_ptr = nullptr;

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

    const char* const canvas_pixel_type = canvas->get_type();
    const char* const export_pixel_type = find_best_pixel_type_for_export( canvas_pixel_type, plugin);
    if( !export_pixel_type) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" supports only the import, but not the export of images.",
            plugin->get_name());
        return false;
    }

    const Pixel_type export_pixel_type_enum = convert_pixel_type_string_to_enum( export_pixel_type);
    const mi::Float32 export_default_gamma = get_default_gamma( export_pixel_type_enum);

    // If enabled and necessary, adjust gamma to export_default_gamma
    if( force_default_gamma && fabs( canvas->get_gamma() - export_default_gamma) > 0.001) {
        mi::base::Handle<mi::neuraylib::ICanvas> tmp( copy_canvas( canvas.get()));
        adjust_gamma( tmp.get(), export_default_gamma);
        canvas = tmp;
    }

    // If necessary, convert canvas to export_pixel_type
    if( strcmp( canvas_pixel_type, export_pixel_type) != 0) {
        canvas = convert_canvas( canvas.get(), export_pixel_type_enum);
        ASSERT( M_IMAGE, canvas);
    }

    DISK::File_writer_impl writer;
    if( !writer.open( output_filename)) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "Failed to open image file \"%s\".", output_filename);
        return false;
    }

    const mi::Uint32 image_width   = canvas->get_resolution_x();
    const mi::Uint32 image_height  = canvas->get_resolution_y();
    const mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    const char* const pixel_type   = canvas->get_type();
    const mi::Float32 gamma        = canvas->get_gamma();
    const bool is_cubemap          = get_canvas_is_cubemap( canvas.get());

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_writing( &writer,
        pixel_type, image_width, image_height, nr_of_layers, 1, is_cubemap, gamma, quality));
    if( !image_file) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to export \"%s\" due to unsupported properties.",
            plugin->get_name(), output_filename);
        return false;
    }

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> tile( canvas->get_tile( z));
        if( !image_file->write( tile.get(), z, 0)) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to export \"%s\" due to unsupported "
                "properties.", plugin->get_name(), output_filename);
            return false;
        }
    }

    LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
        "Saving image \"%s\", pixel type \"%s\", %ux%ux%u pixels, 1 miplevel.",
        output_filename, export_pixel_type, image_width, image_height, nr_of_layers);

    return true;
}

bool Image_module_impl::export_mipmap(
    const IMipmap* mipmap,
    const char* output_filename,
    mi::Uint32 quality,
    bool force_default_gamma) const
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

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    if( !canvas)
        return false;

    const char* const canvas_pixel_type = canvas->get_type();
    const char* const export_pixel_type = find_best_pixel_type_for_export( canvas_pixel_type, plugin);
    if( !export_pixel_type) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" supports only the import, but not the export of images.",
            plugin->get_name());
        return false;
    }

    const Pixel_type export_pixel_type_enum = convert_pixel_type_string_to_enum( export_pixel_type);
    const mi::Float32 export_default_gamma = get_default_gamma( export_pixel_type_enum);

    DISK::File_writer_impl writer;
    if( !writer.open( output_filename)) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "Failed to open image file \"%s\".", output_filename);
        return false;
    }

    const mi::Uint32 image_width   = canvas->get_resolution_x();
    const mi::Uint32 image_height  = canvas->get_resolution_y();
          mi::Uint32 nr_of_layers  = canvas->get_layers_size();
          mi::Uint32 nr_of_levels  = mipmap->get_nlevels();
    const mi::Float32 gamma        = canvas->get_gamma();
    const bool is_cubemap          = get_canvas_is_cubemap( canvas.get());

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_writing( &writer,
        export_pixel_type, image_width, image_height, nr_of_layers, nr_of_levels, is_cubemap, gamma,
        quality));
    if( !image_file) {
        // if multiple levels are not supported try again exporting only the first level
        image_file = plugin->open_for_writing( &writer, export_pixel_type,
            image_width, image_height, nr_of_layers, 1, is_cubemap, gamma, quality);
        if( !image_file) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to export \"%s\" due to unsupported properties.",
                plugin->get_name(), output_filename);
            return false;
        } else {
            nr_of_levels = 1;
        }
    }

    for( mi::Uint32 l = 0; l < nr_of_levels; ++l) {
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas_l( mipmap->get_level( l));
        ASSERT( M_IMAGE, canvas_l);
        if( !canvas_l)
            return false;

        nr_of_layers             = canvas_l->get_layers_size();

        // If enabled and necessary, adjust gamma to export_default_gamma
        if( force_default_gamma && fabs( canvas_l->get_gamma() - export_default_gamma) > 0.001) {
            mi::base::Handle<mi::neuraylib::ICanvas> tmp( copy_canvas( canvas_l.get()));
            adjust_gamma( tmp.get(), export_default_gamma);
            canvas_l = tmp;
        }

        // If necessary, convert canvas to export_pixel_type
        if( strcmp( canvas_pixel_type, export_pixel_type) != 0) {
            canvas_l = convert_canvas( canvas_l.get(), export_pixel_type_enum);
            ASSERT( M_IMAGE, canvas_l);
        }

        for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
            mi::base::Handle<const mi::neuraylib::ITile> tile( canvas_l->get_tile( z));
            if( !image_file->write( tile.get(), z, l)) {
                LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                    "The image plugin \"%s\" failed to export \"%s\" due to unsupported "
                    "properties.", plugin->get_name(), output_filename);
                return false;
            }
        }
    }

    LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
        "Saving image \"%s\", pixel type \"%s\", %ux%ux%u pixels, %u miplevel%s.",
        output_filename, export_pixel_type, image_width, image_height, nr_of_layers,
        nr_of_levels, nr_of_levels == 1 ? "" : "s");

    return true;
}

mi::neuraylib::IBuffer* Image_module_impl::create_buffer_from_canvas(
    const mi::neuraylib::ICanvas* canvas_ptr,
    const char* image_format,
    const char* pixel_type,
    mi::Uint32 quality,
    bool force_default_gamma) const
{
    // Wrap incoming canvas pointer into handle for easier management later.
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( canvas_ptr, mi::base::DUP_INTERFACE);
    canvas_ptr = nullptr;

    DISK::Memory_writer_impl writer;

    mi::neuraylib::IImage_plugin* plugin = find_plugin_for_export( image_format);
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle image format \"%s\".", image_format);
        return nullptr;
    }

    const char* export_pixel_type = find_best_pixel_type_for_export( pixel_type, plugin);
    if( !export_pixel_type) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" supports only the import, but not the export of images.",
            plugin->get_name());
        return nullptr;
    }

    const Pixel_type export_pixel_type_enum = convert_pixel_type_string_to_enum( export_pixel_type);
    const mi::Float32 export_default_gamma = get_default_gamma( export_pixel_type_enum);

    // If enabled and necessary, adjust gamma to export_default_gamma
    if( force_default_gamma && fabs( canvas->get_gamma() - export_default_gamma) > 0.001) {
        mi::base::Handle<mi::neuraylib::ICanvas> tmp( copy_canvas( canvas.get()));
        adjust_gamma( tmp.get(), export_default_gamma);
        canvas = tmp;
    }

    // If necessary, convert canvas to export_pixel_type
    if( strcmp( export_pixel_type, canvas->get_type()) != 0) {
        canvas = convert_canvas( canvas.get(), export_pixel_type_enum);
        ASSERT( M_IMAGE, canvas);
    }

    const mi::Uint32 image_width   = canvas->get_resolution_x();
    const mi::Uint32 image_height  = canvas->get_resolution_y();
    const mi::Uint32 nr_of_layers  = canvas->get_layers_size();
    const mi::Float32 gamma        = canvas->get_gamma();
    const bool is_cubemap          = get_canvas_is_cubemap( canvas.get());

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_writing( &writer,
        export_pixel_type, image_width, image_height, nr_of_layers, 1, is_cubemap, gamma, quality));
    if( !image_file) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to encode the canvas due to unsupported properties.",
            plugin->get_name());
        return nullptr;
    }

    for( mi::Uint32 z = 0; z < nr_of_layers; ++z) {
        mi::base::Handle<const mi::neuraylib::ITile> tile( canvas->get_tile( z));
        if( !image_file->write( tile.get(), z, 0)) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to encode the canvas due to unsupported "
                "properties.", plugin->get_name());
            return nullptr;
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
        const mi::Sint64 bytes_read
            = reader->read( static_cast<char*>( static_cast<void*>( &buffer[0])), 512);
        reader->rewind();
        file_size = reader->get_file_size();
        if( bytes_read != 512 && file_size >= 512)
            return nullptr;
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
            plugin_extension = extension ? plugin->get_file_extension( ++extension_index) : nullptr;
        }
    }

    if( queue.empty())
        return nullptr;

    return queue.top();
}

mi::neuraylib::IImage_plugin* Image_module_impl::find_plugin_for_export(
    const char* extension) const
{
    if( !extension)
        return nullptr;

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
                if( plugin->get_supported_type( 0) != nullptr)
                    queue.push( plugin);
            }
            plugin_extension = plugin->get_file_extension( ++extension_index);
        }
    }

    if( queue.empty())
        return nullptr;

    return queue.top();
}

void Image_module_impl::set_mdl_container_callback( IMdl_container_callback* mdl_container_callback)
{
    m_mdl_container_callback = make_handle_dup( mdl_container_callback);
}

IMdl_container_callback* Image_module_impl::get_mdl_container_callback() const
{
    if( !m_mdl_container_callback)
        return nullptr;

    m_mdl_container_callback->retain();
    return m_mdl_container_callback.get();
}

void Image_module_impl::dump() const
{
    mi::Size i = 0;

    mi::base::Lock::Block block( &m_plugins_lock);
          Plugin_vector::const_iterator it     = m_plugins.begin();
    const Plugin_vector::const_iterator it_end = m_plugins.end();

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
    constexpr mi::Uint32 first_pixel_type =  1;
    constexpr mi::Uint32 last_pixel_type  = 14;

    for( mi::Uint32 from = first_pixel_type; from <= last_pixel_type; ++from) {
        const Pixel_type from_enum = static_cast<Pixel_type>( from);
        std::ostringstream s;
        s << "conversion priorities for ";
        s << std::left << std::setw( 11) << convert_pixel_type_enum_to_string( from_enum);

        std::vector<mi::Float32> cost( last_pixel_type+1);
        for( mi::Uint32 to = first_pixel_type; to <= last_pixel_type; ++to) {
            const Pixel_type to_enum = static_cast<Pixel_type>( to);
            cost[to] = get_conversion_cost( from_enum, to_enum);
        }

        mi::Float32 last_cost = 0.0;
        for( mi::Uint32 i2 = first_pixel_type; i2 <= last_pixel_type; ++i2) {

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

            const Pixel_type to_enum = static_cast<Pixel_type>( min_index);
            if( i2 > 1)
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

static constexpr mi::Float32 g_cost[14][14] = {
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
    const Pixel_type canvas_pixel_type = convert_pixel_type_string_to_enum( pixel_type);
    ASSERT( M_IMAGE, canvas_pixel_type != PT_UNDEF);

    Pixel_type min_pixel_type = PT_UNDEF;
    mi::Float32 min_ratio = 100.0f;

    mi::Uint32 index = 0;
    Pixel_type plugin_pixel_type
        = convert_pixel_type_string_to_enum( plugin->get_supported_type( index));

    while( plugin_pixel_type != PT_UNDEF) {

        const mi::Float32 ratio = get_conversion_cost( canvas_pixel_type, plugin_pixel_type);
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

namespace {

#if defined(HAS_SSE) || defined(SSE_INTRINSICS)
#ifdef MI_ARCH_X86_64
#include <xmmintrin.h>
#elif defined(MI_ARCH_ARM_64)
//#include <prod/iray-wf/core-cpu/sse2neon.h>
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <base/lib/simde/x86/sse2.h>
#endif

__m128 fast_pow4(
    const __m128 &b,  ///< %base
    const __m128 &e)  ///< exponent
{

    const __m128 x = _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(b)),
        _mm_set1_ps(0.00000011920928955078125f)), _mm_set1_ps(127.f));
    const __m128i xi = _mm_cvttps_epi32(x);
    const __m128 y = _mm_sub_ps(x, _mm_cvtepi32_ps(_mm_sub_epi32(
        xi, _mm_srli_epi32(_mm_castps_si128(x), 31)))); // >>31 to correct x < 0
    const __m128 fl = _mm_mul_ps(e, _mm_add_ps(x, _mm_mul_ps(
        _mm_sub_ps(y, _mm_mul_ps(y, y)), _mm_set1_ps(0.346607f))));

    const __m128i fli = _mm_cvttps_epi32(fl);
    const __m128 y2 = _mm_sub_ps(fl, _mm_cvtepi32_ps(
        _mm_sub_epi32(fli, _mm_srli_epi32(_mm_castps_si128(fl), 31)))); // >>31 to correct fl < 0
    const __m128 z = _mm_max_ps(_mm_sub_ps(_mm_add_ps(
        _mm_mul_ps(fl, _mm_set1_ps(8388608.0f)), _mm_set1_ps((float)(127.0*8388608.0))),
        _mm_mul_ps(_mm_sub_ps(y2, _mm_mul_ps(y2, y2)), _mm_set1_ps((float)(0.33971*8388608.0)))),
        _mm_setzero_ps());

    return _mm_and_ps(_mm_castsi128_ps(
        _mm_cvttps_epi32(z)), _mm_cmpneq_ps(b, _mm_setzero_ps())); // b==0 -> 0
}

#endif

void apply_gamma(mi::math::Color& c, mi::Float32 gamma)
{
    if (gamma == 1.0f)
        return;
#if defined(HAS_SSE) || defined(SSE_INTRINSICS)
    c = mi::base::binary_cast<mi::math::Color_struct>(
        fast_pow4(_mm_set_ps(c.a, c.b, c.g, c.r), _mm_set1_ps(gamma)));
#else
    c.r = mi::math::fast_pow(c.r, gamma);
    c.g = mi::math::fast_pow(c.g, gamma);
    c.b = mi::math::fast_pow(c.b, gamma);
    c.a = mi::math::fast_pow(c.a, gamma);
#endif
}

} // namespace

mi::neuraylib::ICanvas* Image_module_impl::create_miplevel(
    const mi::neuraylib::ICanvas* prev_canvas, float gamma_override) const
{
    // NOTE: This implementation creates the new miplevel tile by tile. For each tile, it retrieves
    // the at most four needed tiles from the previous miplevel *once* (and not for every pixel).
    // Remember that tile lookups require locks (and reference counts).
    ASSERT(M_IMAGE, prev_canvas);

    // Get properties of previous miplevel
    const mi::Uint32 prev_width = prev_canvas->get_resolution_x();
    const mi::Uint32 prev_height = prev_canvas->get_resolution_y();
    const mi::Uint32 prev_layers = prev_canvas->get_layers_size();
    const Pixel_type prev_pixel_type
        = convert_pixel_type_string_to_enum(prev_canvas->get_type());
    const mi::Float32 prev_gamma = gamma_override != 0.0f ? gamma_override : prev_canvas->get_gamma();

    // Compute properties of this miplevel
    const mi::Uint32 width = std::max(prev_width / 2, 1u);
    const mi::Uint32 height = std::max(prev_height / 2, 1u);
    const mi::Uint32 layers = prev_layers;
    const Pixel_type pixel_type = prev_pixel_type;
    const mi::Float32 gamma = prev_gamma;

    // Create the miplevel
    mi::neuraylib::ICanvas* canvas = new Canvas_impl(
        pixel_type, width, height, layers,
        get_canvas_is_cubemap(prev_canvas), prev_canvas->get_gamma());

    constexpr mi::Uint32 offsets_x[4] = { 0, 1, 0, 1 };
    constexpr mi::Uint32 offsets_y[4] = { 0, 0, 1, 1 };

    const mi::Float32 inv_gamma = 1.0f / gamma;

    // Loop over the tiles and compute the pixel data for each tile
    for (mi::Uint32 tile_z = 0; tile_z < layers; ++tile_z) {

        // The current tile covers pixels in the range [0,x_end) x [0,y_end)
        // from the canvas for this miplevel.
        const mi::Uint32 x_end = width;
        const mi::Uint32 y_end = height;

        // Lookup tile for this miplevel
        mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());

        // Lookup involved tiles from the previous miplevel (note that these tiles are not
        // necessarily distinct).
        mi::base::Handle<const mi::neuraylib::ITile> prev_tile( prev_canvas->get_tile());
        ASSERT(M_IMAGE, prev_tile);

        // Loop over the pixels of this tile and compute the value for each pixel
        for (mi::Uint32 y = 0; y < y_end; ++y) {
            for (mi::Uint32 x = 0; x < x_end; ++x) {

                // The current pixel (x,y) corresponds to the four pixels
                // [prev_x, prev_x+1] x [prev_y,prev_y+1] in the tiles of the previous
                // layer. Note that all four pixels might actually be in a different tile.
                const mi::Uint32 prev_x = 2 * x;
                const mi::Uint32 prev_y = 2 * y;

                mi::math::Color color(0.0f, 0.0f, 0.0f, 0.0f);
                mi::Uint32 nr_of_summands = 0;

                // Loop over the at most four pixels corresponding to pixel (x,y)
                for (mi::Uint32 i = 0; i < 4; ++i) {

                    // Find tile of pixel
                    // (prev_x + offsets_x[i], prev_y + offsets_y[i]) and its coordinates
                    // with respect to that tile.
                    const mi::Uint32 prev_actual_x = prev_x + offsets_x[i];
                    if (prev_actual_x >= prev_width)
                        continue;
                    const mi::Uint32 prev_actual_y = prev_y + offsets_y[i];
                    if (prev_actual_y >= prev_height)
                        continue;

                    // The pixel (prev_x + offsets_x[i], prev_y + offsets_y[i]) actually
                    // is pixel (prev_actual_x, prev_actual_y) in tile
                    // prev_tiles[prev_tile_id].
                    mi::math::Color prev_color;
                    prev_tile->get_pixel(
                            prev_actual_x, prev_actual_y, &prev_color.r);
                    if (gamma != 1.0f) {
                        apply_gamma(prev_color, gamma);
                    }
                    color += prev_color;
                    nr_of_summands += 1;
                }
                color /= static_cast<mi::Float32>(nr_of_summands);
                if (gamma != 1.0f)
                    apply_gamma(color, inv_gamma);

                tile->set_pixel(x, y, &color.r);
            }
        }
    }
    return canvas;
}

} // namespace IMAGE

} // namespace MI
