/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/handle.h>
#include <mi/math/color.h>
#include <mi/neuraylib/iimage_plugin.h>

#include <filesystem>

#include "i_image.h"
#include "i_image_utilities.h"
#include "image_canvas_impl.h"
#include "image_tile_impl.h"

#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/disk/disk_utils.h>

namespace fs = std::filesystem;

namespace MI {

namespace IMAGE {

namespace {

/// Returns a readable selector string for log messages.
std::string get_selector_string( const char* selector)
{
    if( !selector || !selector[0])
        return std::string( "no selector");
    return std::string( "selector \"") + selector + '\"';
}

} // namespace

Canvas_impl::Canvas_impl(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma)
{
    // check incorrect arguments
    ASSERT( M_IMAGE, pixel_type != PT_UNDEF);
    ASSERT( M_IMAGE, width > 0 && height > 0 && layers > 0);
    ASSERT( M_IMAGE, !is_cubemap || layers == 6);
    ASSERT( M_IMAGE, gamma >= 0);

    m_pixel_type    = pixel_type;
    m_width         = width;
    m_height        = height;
    m_nr_of_layers  = layers;
    m_miplevel      = 0;
    m_is_cubemap    = is_cubemap;
    m_gamma         = gamma == 0.0f ? get_default_gamma( m_pixel_type) : gamma;

    m_tiles.resize( m_nr_of_layers);
    for( mi::Uint32 i = 0; i < m_nr_of_layers; ++i)
        m_tiles[i] = create_tile( m_pixel_type, m_width, m_height);
}

Canvas_impl::Canvas_impl( const mi::neuraylib::ICanvas* other)
{
    const Pixel_type pixel_type = convert_pixel_type_string_to_enum( other->get_type());
    const mi::Uint32 width = other->get_resolution_x();
    const mi::Uint32 height = other->get_resolution_y();
    const mi::Uint32 layers = other->get_layers_size();
    const mi::Float32 gamma = other->get_gamma();

    mi::base::Handle<const ICanvas> canvas_internal( other->get_interface<ICanvas>());
    const bool is_cubemap = canvas_internal && canvas_internal->get_is_cubemap();

    // check incorrect arguments
    ASSERT( M_IMAGE, pixel_type != PT_UNDEF);
    ASSERT( M_IMAGE, width > 0 && height > 0 && layers > 0);
    ASSERT( M_IMAGE, !is_cubemap || layers == 6);
    ASSERT( M_IMAGE, gamma >= 0);

    m_pixel_type    = pixel_type;
    m_width         = width;
    m_height        = height;
    m_nr_of_layers  = layers;
    m_miplevel      = 0;
    m_is_cubemap    = is_cubemap;
    m_gamma         = gamma == 0.0f ? get_default_gamma( m_pixel_type) : gamma;

    m_tiles.resize( m_nr_of_layers);
    for( mi::Uint32 i = 0; i < m_nr_of_layers; ++i) {
        mi::base::Handle<const mi::neuraylib::ITile> tile( other->get_tile( i));
        m_tiles[i] = copy_tile( tile.get());
    }
}

Canvas_impl::Canvas_impl(
    File_based,
    const std::string& filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<DISK::File_reader_impl> reader(
        new DISK::File_reader_impl);
    if( !reader->open( filename.c_str())) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "Failed to open image file \"%s\".", filename.c_str());
        *errors = -5;
        set_default_pink_dummy_canvas();
        return;
    }

    std::string extension = DISK::get_extension( filename);

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( extension.c_str(), reader.get());
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle \"%s\".", filename.c_str());
        *errors = -3;
        set_default_pink_dummy_canvas();
        return;
    }

    bool plugin_supports_selectors = plugin->supports_selectors();
    const char* plugin_selector = plugin_supports_selectors ? selector : nullptr;
    mi::base::Handle<mi::neuraylib::IImage_file> image_file(
        plugin->open_for_reading( reader.get(), plugin_selector));
    if( !image_file) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            R"(The image plugin "%s" failed to import "%s".)",
            plugin->get_name(), filename.c_str());
        *errors = -7;
        set_default_pink_dummy_canvas();
        return;
    }

    do_init(
        File_based(),
        filename,
        selector,
        miplevel,
        image_file.get(),
        plugin_supports_selectors,
        errors);
}

Canvas_impl::Canvas_impl(
    File_based,
    const std::string& filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::neuraylib::IImage_file* image_file,
    bool plugin_supports_selectors,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    do_init(
        File_based(),
        filename,
        selector,
        miplevel,
        image_file,
        plugin_supports_selectors,
        errors);
}

void Canvas_impl::do_init(
    File_based,
    const std::string& filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::neuraylib::IImage_file* image_file,
    bool plugin_supports_selectors,
    mi::Sint32* errors)
{
    m_filename = filename;

    if( selector)
        m_selector = selector;

    m_pixel_type    = convert_pixel_type_string_to_enum( image_file->get_type());
    m_width         = std::max( image_file->get_resolution_x( 0) >> miplevel, 1u);
    m_height        = std::max( image_file->get_resolution_y( 0) >> miplevel, 1u);
    m_nr_of_layers  = image_file->get_layers_size( 0);
    m_miplevel      = miplevel;
    m_is_cubemap    = image_file->get_is_cubemap();
    m_gamma         = image_file->get_gamma();

    if(    m_pixel_type == PT_UNDEF
        || m_width == 0
        || m_height == 0
        || m_nr_of_layers == 0
        || m_gamma <= 0) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin failed to import \"%s\".", m_filename.c_str());
        *errors = -7;
        set_default_pink_dummy_canvas();
        return;
    }

    if( selector && !plugin_supports_selectors) {
        const Pixel_type new_pixel_type = get_pixel_type_for_channel( m_pixel_type, selector);
        if( new_pixel_type == PT_UNDEF) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                R"(Failed to apply selector "%s" to "%s" with pixel type "%s".)",
                selector, m_filename.c_str(), image_file->get_type());
            *errors = -10;
            set_default_pink_dummy_canvas();
            return;
        }

        m_pixel_type = new_pixel_type;
    }

    if( m_miplevel == 0) {
        const mi::Uint32 miplevels = image_file->get_miplevels();
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
            "Loading image \"%s\", %s, pixel type \"%s\", "
#ifdef MI_IMAGE_LOG_ASSUMED_GAMMA
            "gamma %0.1f, "
#endif
            "%ux%ux%u pixels, %u miplevel%s.",
            m_filename.c_str(),
            get_selector_string( selector).c_str(),
            convert_pixel_type_enum_to_string( m_pixel_type),
#ifdef MI_IMAGE_LOG_ASSUMED_GAMMA
            m_gamma,
#endif
            m_width,
            m_height,
            m_nr_of_layers,
            miplevels,
            miplevels == 1 ? "" : "s");
    }

    m_tiles.resize( m_nr_of_layers);

    *errors = 0;
}

Canvas_impl::Canvas_impl(
    Container_based,
    mi::neuraylib::IReader* reader,
    const std::string& container_filename,
    const std::string& member_filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !reader) {
        *errors = -1;
        set_default_pink_dummy_canvas();
        return;
    }

    if( !reader->supports_absolute_access()) {
        *errors = -6;
        set_default_pink_dummy_canvas();
        return;
    }

    std::string extension = DISK::get_extension( member_filename);

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( extension.c_str(), reader);
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            R"(No image plugin found to handle "%s" in "%s".)",
            member_filename.c_str(), container_filename.c_str());
        *errors = -3;
        set_default_pink_dummy_canvas();
        return;
    }

    bool plugin_supports_selectors = plugin->supports_selectors();
    const char* plugin_selector = plugin_supports_selectors ? selector : nullptr;
    mi::base::Handle<mi::neuraylib::IImage_file> image_file(
        plugin->open_for_reading( reader, plugin_selector));
    if( !image_file) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            R"(The image plugin "%s" failed to import "%s" in "%s".)",
            plugin->get_name(), member_filename.c_str(), container_filename.c_str());
        *errors = -7;
        set_default_pink_dummy_canvas();
        return;
    }

    do_init(
        Container_based(),
        reader,
        container_filename,
        member_filename,
        selector,
        miplevel,
        image_file.get(),
        plugin_supports_selectors,
        errors);
}

Canvas_impl::Canvas_impl(
    Container_based,
    mi::neuraylib::IReader* reader,
    const std::string& container_filename,
    const std::string& member_filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::neuraylib::IImage_file* image_file,
    bool plugin_supports_selectors,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    do_init(
        Container_based(),
        reader,
        container_filename,
        member_filename,
        selector,
        miplevel,
        image_file,
        plugin_supports_selectors,
        errors);
}

void Canvas_impl::do_init(
    Container_based,
    mi::neuraylib::IReader* reader,
    const std::string& container_filename,
    const std::string& member_filename,
    const char* selector,
    mi::Uint32 miplevel,
    mi::neuraylib::IImage_file* image_file,
    bool plugin_supports_selectors,
    mi::Sint32* errors)
{
    m_container_filename = container_filename;
    m_member_filename  = member_filename;

    if( selector)
        m_selector = selector;

    m_pixel_type    = convert_pixel_type_string_to_enum( image_file->get_type());
    m_width         = std::max( image_file->get_resolution_x( 0) >> miplevel, 1u);
    m_height        = std::max( image_file->get_resolution_y( 0) >> miplevel, 1u);
    m_nr_of_layers  = image_file->get_layers_size( 0);
    m_miplevel      = miplevel;
    m_is_cubemap    = image_file->get_is_cubemap();
    m_gamma         = image_file->get_gamma();

    if(    m_pixel_type == PT_UNDEF
        || m_width == 0
        || m_height == 0
        || m_nr_of_layers == 0
        || m_gamma <= 0) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            R"(The image plugin failed to import "%s" in "%s".)",
            member_filename.c_str(), container_filename.c_str());
        *errors = -7;
        set_default_pink_dummy_canvas();
        return;
    }

    if( selector && !plugin_supports_selectors) {
        const Pixel_type new_pixel_type = get_pixel_type_for_channel( m_pixel_type, selector);
        if( new_pixel_type == PT_UNDEF) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                R"(Failed to apply selector "%s" to "%s" in "%s" with pixel type "%s".)",
                selector, member_filename.c_str(), container_filename.c_str(),
                image_file->get_type());
            *errors = -10;
            set_default_pink_dummy_canvas();
            return;
        }

        m_pixel_type = new_pixel_type;
    }

    if( m_miplevel == 0) {
        const mi::Uint32 miplevels = image_file->get_miplevels();
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
            "Loading image \"%s\" in \"%s\", %s, pixel type \"%s\", "
#ifdef MI_IMAGE_LOG_ASSUMED_GAMMA
            "gamma %0.1f, "
#endif
            "%ux%ux%u pixels, %u miplevel%s.",
            member_filename.c_str(),
            container_filename.c_str(),
            get_selector_string( selector).c_str(),
            convert_pixel_type_enum_to_string( m_pixel_type),
#ifdef MI_IMAGE_LOG_ASSUMED_GAMMA
            m_gamma,
#endif
            m_width,
            m_height,
            m_nr_of_layers,
            miplevels,
            miplevels == 1 ? "" : "s");
    }

    m_tiles.resize( m_nr_of_layers);

    if( supports_lazy_loading()) {
        *errors = 0;
        return;
    }

    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z) {
        m_tiles[z] = image_file->read( z, m_miplevel);
        if( !m_tiles[z]) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                    R"(The image plugin failed to import "%s" in "%s".)",
                    member_filename.c_str(), container_filename.c_str());
            *errors = -5;
            set_default_pink_dummy_canvas();
            return;
        }
    }

    *errors = 0;
}

Canvas_impl::Canvas_impl(
    Memory_based,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* selector,
    const char* mdl_file_path,
    mi::Uint32 miplevel,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !reader || !image_format) {
        *errors = -1;
        set_default_pink_dummy_canvas();
        return;
    }

    if( !reader->supports_absolute_access()) {
        *errors = -6;
        set_default_pink_dummy_canvas();
        return;
    }

    if( selector)
        m_selector = selector;

    const std::string log_identifier = mdl_file_path
        ? std::string( "an image from MDL file path \"") + mdl_file_path + '\"'
        : std::string( "a memory-based image with image format \"") + image_format + '\"';

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( image_format, reader);
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle %s.", log_identifier.c_str());
        *errors = -3;
        set_default_pink_dummy_canvas();
        return;
    }

    bool plugin_supports_selectors = plugin->supports_selectors();
    const char* plugin_selector = plugin_supports_selectors ? selector : nullptr;
    mi::base::Handle<mi::neuraylib::IImage_file> image_file(
        plugin->open_for_reading( reader, plugin_selector));
    if( !image_file) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin failed to import %s.", log_identifier.c_str());
        *errors = -7;
        set_default_pink_dummy_canvas();
        return;
    }

    do_init(
        Memory_based(),
        reader,
        image_format,
        selector,
        mdl_file_path,
        miplevel,
        image_file.get(),
        plugin_supports_selectors,
        errors);
}

Canvas_impl::Canvas_impl(
    Memory_based,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* selector,
    const char* mdl_file_path,
    mi::Uint32 miplevel,
    mi::neuraylib::IImage_file* image_file,
    bool plugin_supports_selectors,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    do_init(
        Memory_based(),
        reader,
        image_format,
        selector,
        mdl_file_path,
        miplevel,
        image_file,
        plugin_supports_selectors,
        errors);
}

void Canvas_impl::do_init(
    Memory_based,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* selector,
    const char* mdl_file_path,
    mi::Uint32 miplevel,
    mi::neuraylib::IImage_file* image_file,
    bool plugin_supports_selectors,
    mi::Sint32* errors)
{
    ASSERT( M_IMAGE, image_format);

    if( selector)
        m_selector = selector;

    const std::string log_identifier = mdl_file_path
        ? std::string( "an image from MDL file path \"") + mdl_file_path + '\"'
        : std::string( "a memory-based image with image format \"") + image_format + '\"';

    m_pixel_type    = convert_pixel_type_string_to_enum( image_file->get_type());
    m_width         = std::max( image_file->get_resolution_x( 0) >> miplevel, 1u);
    m_height        = std::max( image_file->get_resolution_y( 0) >> miplevel, 1u);
    m_nr_of_layers  = image_file->get_layers_size( 0);
    m_miplevel      = miplevel;
    m_is_cubemap    = image_file->get_is_cubemap();
    m_gamma         = image_file->get_gamma();

    if(    m_pixel_type == PT_UNDEF
        || m_width == 0
        || m_height == 0
        || m_nr_of_layers == 0
        || m_gamma <= 0) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin failed to import %s.", log_identifier.c_str());
        *errors = -7;
        set_default_pink_dummy_canvas();
        return;
    }

    if( selector && !plugin_supports_selectors) {
        const Pixel_type new_pixel_type = get_pixel_type_for_channel( m_pixel_type, selector);
        if( new_pixel_type == PT_UNDEF) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                R"(Failed to apply selector "%s" to "%s" with pixel type "%s".)",
                selector, log_identifier.c_str(), image_file->get_type());
            *errors = -10;
            set_default_pink_dummy_canvas();
            return;
        }

        m_pixel_type = new_pixel_type;
    }

    if( m_miplevel == 0) {
        const mi::Uint32 miplevels = image_file->get_miplevels();
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
            "Loading %s, %s, pixel type \"%s\", "
#ifdef MI_IMAGE_LOG_ASSUMED_GAMMA
            "gamma %0.1f, "
#endif
            "%ux%ux%u pixels, %u miplevel%s.",
            log_identifier.c_str(),
            get_selector_string( selector).c_str(),
            convert_pixel_type_enum_to_string( m_pixel_type),
#ifdef MI_IMAGE_LOG_ASSUMED_GAMMA
            m_gamma,
#endif
            m_width,
            m_height,
            m_nr_of_layers,
            miplevels,
            miplevels == 1 ? "" : "s");
    }

    m_tiles.resize( m_nr_of_layers);

    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z) {
        m_tiles[z] = image_file->read( z, m_miplevel);
        if( !m_tiles[z]) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin failed to import %s.", log_identifier.c_str());
            *errors = -7;
            set_default_pink_dummy_canvas();
            return;
        }
    }

    *errors = 0;
}

Canvas_impl::Canvas_impl(
    const std::vector<mi::base::Handle<mi::neuraylib::ITile>>& tiles,
    Float32 gamma)
  : m_tiles( tiles)
{
    // check incorrect arguments
    ASSERT( M_IMAGE, !tiles.empty());
    ASSERT( M_IMAGE, tiles[0]);
    ASSERT( M_IMAGE, gamma >= 0);

    m_pixel_type     = convert_pixel_type_string_to_enum( tiles[0]->get_type());
    m_width          = tiles[0]->get_resolution_x();
    m_height         = tiles[0]->get_resolution_y();
    m_nr_of_layers   = tiles.size();
    m_miplevel       = 0;
    m_is_cubemap     = false;
    m_gamma          = gamma == 0.0f ? get_default_gamma( m_pixel_type) : gamma;

    // check incorrect arguments
#ifdef ENABLE_ASSERT
    for( size_t i = 1, cnt = tiles.size(); i < cnt; ++i) {
        ASSERT( M_IMAGE, tiles[i]->get_resolution_x() == m_width);
        ASSERT( M_IMAGE, tiles[i]->get_resolution_y() == m_height);
        ASSERT( M_IMAGE, strcmp( tiles[i]->get_type(), tiles[0]->get_type()) == 0);
    }
#endif
}

void Canvas_impl::reset(
        const mi::Uint32 width,
        const mi::Uint32 height,
        const Pixel_type pt)
{
    for ( auto& tile : m_tiles) {
        static_cast<Tile_impl*>(tile.get())->reset( width, height, pt);
    }

    m_pixel_type = pt;
    m_width = width;
    m_height = height;
}


const char* Canvas_impl::get_type() const
{
    return convert_pixel_type_enum_to_string( m_pixel_type);
}

void Canvas_impl::set_gamma( mi::Float32 gamma)
{
    ASSERT( M_IMAGE, gamma > 0);
    m_gamma = gamma;
}

const mi::neuraylib::ITile* Canvas_impl::get_tile( mi::Uint32 layer) const
{
    if( layer >= m_nr_of_layers)
        return nullptr;

    mi::base::Lock::Block block( &m_lock);

    if( m_tiles[layer] == nullptr) {
        ASSERT( M_IMAGE, supports_lazy_loading());
#ifdef MI_IMAGE_LOAD_ONLY_REQUESTED_TILE
        m_tiles[layer] = load_tile( layer);
#else
        for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z) {
            ASSERT( M_IMAGE, !m_tiles[z]);
            m_tiles[z] = load_tile( z);
        }
#endif
    }

    m_tiles[layer]->retain();
    return m_tiles[layer].get();
}

mi::neuraylib::ITile* Canvas_impl::get_tile( mi::Uint32 layer)
{
    if( layer >= m_nr_of_layers)
        return nullptr;

    mi::base::Lock::Block block( &m_lock);

    if( m_tiles[layer] == nullptr) {

        ASSERT( M_IMAGE, supports_lazy_loading());
#ifdef MI_IMAGE_LOAD_ONLY_REQUESTED_TILE
        m_tiles[layer] = load_tile( layer);
#else
        for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z) {
            ASSERT( M_IMAGE, !m_tiles[z]);
            m_tiles[z] = load_tile( z);
        }
#endif
    }

    m_tiles[layer]->retain();
    return m_tiles[layer].get();
}

mi::Size Canvas_impl::get_size() const
{
    mi::Size size = sizeof( *this);

    size += m_nr_of_layers * sizeof( mi::base::Handle<mi::neuraylib::ITile>); // m_tiles

    for( mi::Uint32 i = 0; i < m_nr_of_layers; ++i)          // m_tiles[i]
        if( m_tiles[i]) {
            mi::base::Handle<ITile> tile_internal( m_tiles[i]->get_interface<ITile>());
            if( tile_internal)         // exact memory usage
                size += tile_internal->get_size();
            else                                            // approximate memory usage
                size +=   static_cast<size_t>( m_width)
                        * static_cast<size_t>( m_height)
                        * get_bytes_per_pixel( m_pixel_type);
        }

    return size;
}

bool Canvas_impl::release_tiles() const
{
    if( !supports_lazy_loading())
        return false;

    mi::base::Lock::Block block( &m_lock);
    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z)
        m_tiles[z] = nullptr;

    return true;
}

bool Canvas_impl::supports_lazy_loading() const
{
    // either both m_container_filename or m_member_filename are set or none
    ASSERT( M_IMAGE,  m_container_filename.empty() || !m_member_filename.empty());
    ASSERT( M_IMAGE, !m_container_filename.empty() ||  m_member_filename.empty());

    // m_filename and m_container_filename (or m_member_filename) are not both set
    ASSERT( M_IMAGE,  m_filename.empty() || m_container_filename.empty());

    if( !m_filename.empty())
        return true;
    if( m_container_filename.empty())
        return false;

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::base::Handle<IMdl_container_callback> callback( image_module->get_mdl_container_callback());
    return callback;
}

mi::neuraylib::ITile* Canvas_impl::load_tile( mi::Uint32 z) const
{
    mi::neuraylib::ITile* tile = do_load_tile( z);
    return tile ? tile : create_tile( m_pixel_type, m_width, m_height);
}

mi::neuraylib::ITile* Canvas_impl::do_load_tile( mi::Uint32 z) const
{
    ASSERT( M_IMAGE, supports_lazy_loading());

    std::string log_identifier;
    mi::base::Handle<mi::neuraylib::IReader> reader( get_reader( log_identifier));
    if( !reader)
        return nullptr;

    std::string extension
        = DISK::get_extension( !m_filename.empty() ? m_filename : m_member_filename);

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( extension.c_str(), reader.get());
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle \"%s\".", log_identifier.c_str());
        return nullptr;
    }

    bool plugin_supports_selectors = plugin->supports_selectors();
    const char* plugin_selector
        = plugin_supports_selectors && !m_selector.empty() ? m_selector.c_str() : nullptr;
    mi::base::Handle<mi::neuraylib::IImage_file> image_file(
        plugin->open_for_reading( reader.get(), plugin_selector));
    if( !image_file) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            R"(The image plugin "%s" failed to import "%s".)",
            plugin->get_name(), log_identifier.c_str());
        return nullptr;
    }

    mi::base::Handle<mi::neuraylib::ITile> tile( image_file->read( z, m_miplevel));
    if( !tile) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            R"(The image plugin "%s" failed to import "%s".)",
            plugin->get_name(), log_identifier.c_str());
        return nullptr;
    }

    const char* pixel_type = tile->get_type();
    const std::string pixel_type_str = pixel_type ? pixel_type : "(invalid)";
    if( !plugin_supports_selectors && !m_selector.empty()) {
        tile = image_module->extract_channel( tile.get(), m_selector.c_str());
        if( !tile) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                R"(Failed to apply selector "%s" to "%s" with pixel type "%s".)",
                m_selector.c_str(), log_identifier.c_str(), pixel_type_str.c_str());
            return nullptr;
        }
        pixel_type = tile->get_type();
    }

    // Check pixel type of the tile
    if( convert_pixel_type_string_to_enum( pixel_type) != m_pixel_type) {
        // We report here the pixel type after selector application, whereas strictly speaking, the
        // plugin might have returned a different pixel (before selector application).
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import \"%s\" (incorrect pixel type \"%s\" "
            "vs expected \"%s\").",
            plugin->get_name(), log_identifier.c_str(), pixel_type,
            convert_pixel_type_enum_to_string( m_pixel_type));
        return nullptr;
    }

    // Check width of the tile
    const mi::Uint32 width = tile->get_resolution_x();
    if( width != m_width) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import \"%s\" (incorrect width %u "
            "vs expected %u).",
            plugin->get_name(), log_identifier.c_str(), width, m_width);
        return nullptr;

    }

    // Check height of the tile
    const mi::Uint32 height = tile->get_resolution_y();
    if( height != m_height) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import \"%s\" (incorrect height %u "
            "vs expected %u).",
            plugin->get_name(), log_identifier.c_str(), height, m_height);
        return nullptr;
    }

    return tile.extract();
}

mi::neuraylib::IReader* Canvas_impl::get_reader( std::string& log_identifier) const
{
    ASSERT( M_IMAGE, supports_lazy_loading());

    log_identifier.clear();

    // file-based
    if( !m_filename.empty()) {

        log_identifier = m_filename;

        mi::base::Handle<DISK::File_reader_impl> reader( new DISK::File_reader_impl);
        if( !reader->open( m_filename.c_str())) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                 "Failed to open image file \"%s\".", log_identifier.c_str());
            return nullptr;
        }

        return reader.extract();

    }

    // container-based
    if( !m_container_filename.empty() && !m_member_filename.empty()) {

        log_identifier = m_container_filename + "\" in \"" + m_member_filename;

        SYSTEM::Access_module<Image_module> image_module( false);
        mi::base::Handle<IMdl_container_callback> callback(
            image_module->get_mdl_container_callback());
        mi::base::Handle<mi::neuraylib::IReader> reader(
            callback->get_reader( m_container_filename.c_str(), m_member_filename.c_str()));
        if( !reader) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                 "Failed to open image file \"%s\".", log_identifier.c_str());
            return nullptr;
        }

        return reader.extract();
    }

    ASSERT( M_IMAGE, false);
    return nullptr;
}

void Canvas_impl::set_default_pink_dummy_canvas()
{
    m_tiles.clear();

    m_filename.clear();
    m_container_filename.clear();
    m_member_filename.clear();

    m_pixel_type    = PT_RGBA;
    m_width         = 1;
    m_height        = 1;
    m_nr_of_layers  = 1;
    m_miplevel      = 0;
    m_is_cubemap    = false;
    m_gamma         = 2.2f;

    m_tiles.resize( 1);
    m_tiles[0] = create_tile( m_pixel_type, m_width, m_height);

    mi::base::Handle<mi::neuraylib::ITile> tile( get_tile());
    const mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
}

} // namespace IMAGE

} // namespace MI
