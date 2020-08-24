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

#include <mi/base/handle.h>
#include <mi/math/color.h>
#include <mi/neuraylib/iimage_plugin.h>

#include "i_image.h"
#include "i_image_utilities.h"
#include "image_canvas_impl.h"
#include "image_tile_impl.h"

#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>

namespace MI {

namespace IMAGE {

Canvas_impl::Canvas_impl(
    Pixel_type pixel_type,
    mi::Uint32 width,
    mi::Uint32 height,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::Uint32 layers,
    bool is_cubemap,
    mi::Float32 gamma)
{
    // check incorrect arguments
    ASSERT( M_IMAGE, pixel_type != PT_UNDEF);
    ASSERT( M_IMAGE, width > 0 && height > 0 && layers > 0);
    ASSERT( M_IMAGE, !is_cubemap || layers == 6);
    ASSERT( M_IMAGE, gamma >= 0);

    // choose default values if requested
#ifdef MI_IMAGE_USE_TILES_BY_DEFAULT
    if( tile_width  == 0)
        tile_width  = std::min( width,  default_tile_width);
    if( tile_height == 0)
        tile_height = std::min( height, default_tile_height);
#else
    if( tile_width  == 0)
        tile_width  = width;
    if( tile_height == 0)
        tile_height = height;
#endif

    m_pixel_type    = pixel_type;
    m_width         = width;
    m_height        = height;
    m_tile_width    = tile_width;
    m_tile_height   = tile_height;
    m_nr_of_layers  = layers;
    m_nr_of_tiles_x = (width  + tile_width  - 1) / tile_width;
    m_nr_of_tiles_y = (height + tile_height - 1) / tile_height;
    m_nr_of_tiles   = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;
    m_miplevel      = 0;
    m_is_cubemap    = is_cubemap;
    m_gamma         = gamma == 0.0f ? get_default_gamma( m_pixel_type) : gamma;

    m_tiles = new mi::neuraylib::ITile*[m_nr_of_tiles];
    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
        m_tiles[i] = create_tile( m_pixel_type, m_tile_width, m_tile_height);
}

Canvas_impl::Canvas_impl(
    const std::string& filename,
    mi::Uint32 miplevel,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::neuraylib::IImage_file* image_file,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    m_filename = filename;

    m_nr_of_tiles = 0;
    m_tiles = 0;

    mi::base::Handle<mi::neuraylib::IImage_file> image_file2;
    mi::base::Handle<DISK::File_reader_impl> reader;

    if( image_file) {
        image_file2 = make_handle_dup( image_file);
        image_file = 0; // only use image_file2 below
    } else {
        reader = new DISK::File_reader_impl;
        if( !reader->open( m_filename.c_str())) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "Failed to open image file \"%s\".", m_filename.c_str());
            *errors = -3;
            set_default_pink_dummy_canvas();
            return;
        }

        std::string root, extension;
        HAL::Ospath::splitext( m_filename, root, extension);
        if( !extension.empty() && extension[0] == '.' )
            extension = extension.substr( 1);

        SYSTEM::Access_module<Image_module> image_module( false);
        mi::neuraylib::IImage_plugin* plugin
            = image_module->find_plugin_for_import( extension.c_str(), reader.get());
        if( !plugin) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "No image plugin found to handle \"%s\".", m_filename.c_str());
            *errors = -4;
            set_default_pink_dummy_canvas();
            return;
        }

        image_file2 = plugin->open_for_reading( reader.get());
        if( !image_file2.is_valid_interface()) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to import \"%s\".",
                plugin->get_name(), m_filename.c_str());
            *errors = -5;
            set_default_pink_dummy_canvas();
            return;
        }
    }

    m_pixel_type    = convert_pixel_type_string_to_enum( image_file2->get_type());
    m_width         = std::max( image_file2->get_resolution_x() >> miplevel, 1u);
    m_height        = std::max( image_file2->get_resolution_y() >> miplevel, 1u);
#ifdef MI_IMAGE_USE_TILES_BY_DEFAULT
    m_tile_width    = tile_width  > 0 ? tile_width  : default_tile_width;
    m_tile_height   = tile_height > 0 ? tile_height : default_tile_height;
#else
    m_tile_width    = tile_width  > 0 ? tile_width  : image_file2->get_tile_resolution_x();
    m_tile_height   = tile_height > 0 ? tile_height : image_file2->get_tile_resolution_y();
#endif
    m_nr_of_layers  = image_file2->get_layers_size();
    m_miplevel      = miplevel;
    m_is_cubemap    = image_file2->get_is_cubemap();
    m_gamma         = image_file2->get_gamma();

    if(    m_pixel_type == PT_UNDEF
        || m_width == 0
        || m_height == 0
        || m_tile_width == 0
        || m_tile_height == 0
        || m_nr_of_layers == 0) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin failed to import \"%s\".", m_filename.c_str());
        set_default_pink_dummy_canvas();
        *errors = -5;
        return;
    }

    if( m_miplevel == 0) {
        mi::Uint32 miplevels = image_file2->get_miplevels();
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
            "Loading image \"%s\", pixel type \"%s\", %ux%ux%u pixels, %u miplevel%s.",
            m_filename.c_str(), convert_pixel_type_enum_to_string( m_pixel_type),
            m_width, m_height, m_nr_of_layers, miplevels, miplevels == 1 ? "" : "s");
    }

    m_nr_of_tiles_x = (m_width  + m_tile_width  - 1) / m_tile_width;
    m_nr_of_tiles_y = (m_height + m_tile_height - 1) / m_tile_height;
    m_nr_of_tiles   = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;

    m_tiles = new mi::neuraylib::ITile*[m_nr_of_tiles];
    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
        m_tiles[i] = 0;

    *errors = 0;
}

Canvas_impl::Canvas_impl(
    Container_based,
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& member_filename,
    mi::Uint32 miplevel,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::neuraylib::IImage_file* image_file,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !reader || !reader->supports_absolute_access()) {
        *errors = -3;
        set_default_pink_dummy_canvas();
        return;
    }

    m_archive_filename = archive_filename;
    m_member_filename  = member_filename;

    m_nr_of_tiles = 0;
    m_tiles = 0;

    mi::base::Handle<mi::neuraylib::IImage_file> image_file2;
    if( image_file) {
        image_file2 = make_handle_dup( image_file);
        image_file = 0; // only use image_file2 below
    } else {
        std::string root, extension;
        HAL::Ospath::splitext( member_filename, root, extension);
        if( !extension.empty() && extension[0] == '.' )
            extension = extension.substr( 1);

        SYSTEM::Access_module<Image_module> image_module( false);
        mi::neuraylib::IImage_plugin* plugin
            = image_module->find_plugin_for_import( extension.c_str(), reader);
        if( !plugin) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "No image plugin found to handle \"%s\" in \"%s\".",
                member_filename.c_str(), archive_filename.c_str());
            *errors = -4;
            set_default_pink_dummy_canvas();
            return;
        }

        image_file2 = plugin->open_for_reading( reader);
        if( !image_file2.is_valid_interface()) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to import \"%s\" in \"%s\".",
                plugin->get_name(), member_filename.c_str(), archive_filename.c_str());
            *errors = -5;
            set_default_pink_dummy_canvas();
            return;
        }
    }

    m_pixel_type    = convert_pixel_type_string_to_enum( image_file2->get_type());
    m_width         = std::max( image_file2->get_resolution_x() >> miplevel, 1u);
    m_height        = std::max( image_file2->get_resolution_y() >> miplevel, 1u);
#ifdef MI_IMAGE_USE_TILES_BY_DEFAULT
    m_tile_width    = tile_width  > 0 ? tile_width  : default_tile_width;
    m_tile_height   = tile_height > 0 ? tile_height : default_tile_height;
#else
    m_tile_width    = tile_width  > 0 ? tile_width  : image_file2->get_tile_resolution_x();
    m_tile_height   = tile_height > 0 ? tile_height : image_file2->get_tile_resolution_y();
#endif
    m_nr_of_layers  = image_file2->get_layers_size();
    m_miplevel      = miplevel;
    m_is_cubemap    = image_file2->get_is_cubemap();
    m_gamma         = image_file2->get_gamma();

    if(    m_pixel_type == PT_UNDEF
        || m_width == 0
        || m_height == 0
        || m_tile_width == 0
        || m_tile_height == 0
        || m_nr_of_layers == 0) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin failed to import \"%s\" in \"%s\".",
            member_filename.c_str(), archive_filename.c_str());
        *errors = -5;
        set_default_pink_dummy_canvas();
        return;
    }

    if( m_miplevel == 0) {
        mi::Uint32 miplevels = image_file2->get_miplevels();
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
            "Loading image \"%s\" in \"%s\", pixel type \"%s\", %ux%ux%u pixels, %u miplevel%s.",
            member_filename.c_str(), archive_filename.c_str(),
            convert_pixel_type_enum_to_string( m_pixel_type),
            m_width, m_height, m_nr_of_layers, miplevels, miplevels == 1 ? "" : "s");
    }

    m_nr_of_tiles_x = (m_width  + m_tile_width  - 1) / m_tile_width;
    m_nr_of_tiles_y = (m_height + m_tile_height - 1) / m_tile_height;
    m_nr_of_tiles   = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;

    if( supports_lazy_loading()) {
        m_tiles = new mi::neuraylib::ITile*[m_nr_of_tiles];
        for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
            m_tiles[i] = 0;
        *errors = 0;
        return;
    }

    m_tiles = new mi::neuraylib::ITile*[m_nr_of_tiles];
    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
        m_tiles[i] = create_tile( m_pixel_type, m_tile_width, m_tile_height);

    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z)
        for( mi::Uint32 tile_y = 0; tile_y < m_nr_of_tiles_y; ++tile_y)
            for( mi::Uint32 tile_x = 0; tile_x < m_nr_of_tiles_x; ++tile_x) {
                mi::Uint32 index = z * m_nr_of_tiles_x*m_nr_of_tiles_y
                                   + tile_y * m_nr_of_tiles_x + tile_x;
                ASSERT( M_IMAGE, index < m_nr_of_tiles);
                mi::Uint32 pixel_x = tile_x * m_tile_width;
                mi::Uint32 pixel_y = tile_y * m_tile_height;
                if( !image_file2->read( m_tiles[index], pixel_x, pixel_y, z, m_miplevel)) {
                    LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                        "The image plugin failed to import \"%s\" in \"%s\".",
                        member_filename.c_str(), archive_filename.c_str());
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
    const char* mdl_file_path,
    mi::Uint32 miplevel,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    mi::neuraylib::IImage_file* image_file,
    mi::Sint32* errors)
{
    ASSERT( M_IMAGE, reader);
    ASSERT( M_IMAGE, image_format);

    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !reader || !reader->supports_absolute_access()) {
        *errors = -3;
        set_default_pink_dummy_canvas();
        return;
    }

    std::string log_identifier = mdl_file_path
        ? std::string( "an image from MDL file path \"") + mdl_file_path + "\""
        : std::string( "a memory-based image with image format \"") + image_format + "\"";

    m_nr_of_tiles = 0;
    m_tiles = 0;

    mi::base::Handle<mi::neuraylib::IImage_file> image_file2;
    if( image_file) {
        image_file2 = make_handle_dup( image_file);
        image_file = 0; // only use image_file2 below
    } else {
        SYSTEM::Access_module<Image_module> image_module( false);
        mi::neuraylib::IImage_plugin* plugin
            = image_module->find_plugin_for_import( image_format, reader);
        if( !plugin) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "No image plugin found to handle %s.", log_identifier.c_str());
            *errors = -4;
            set_default_pink_dummy_canvas();
            return;
        }

        image_file2 = plugin->open_for_reading( reader);
        if( !image_file2.is_valid_interface()) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin failed to import %s.", log_identifier.c_str());
            *errors = -5;
            set_default_pink_dummy_canvas();
            return;
        }
    }

    m_pixel_type    = convert_pixel_type_string_to_enum( image_file2->get_type());
    m_width         = std::max( image_file2->get_resolution_x() >> miplevel, 1u);
    m_height        = std::max( image_file2->get_resolution_y() >> miplevel, 1u);
#ifdef MI_IMAGE_USE_TILES_BY_DEFAULT
    m_tile_width    = tile_width  > 0 ? tile_width  : default_tile_width;
    m_tile_height   = tile_height > 0 ? tile_height : default_tile_height;
#else
    m_tile_width    = tile_width  > 0 ? tile_width  : image_file2->get_tile_resolution_x();
    m_tile_height   = tile_height > 0 ? tile_height : image_file2->get_tile_resolution_y();
#endif
    m_nr_of_layers  = image_file2->get_layers_size();
    m_miplevel      = miplevel;
    m_is_cubemap    = image_file2->get_is_cubemap();
    m_gamma         = image_file2->get_gamma();

    if(    m_pixel_type == PT_UNDEF
        || m_width == 0
        || m_height == 0
        || m_tile_width == 0
        || m_tile_height == 0
        || m_nr_of_layers == 0) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin failed to import %s.", log_identifier.c_str());
        *errors = -5;
        set_default_pink_dummy_canvas();
        return;
    }

    if( m_miplevel == 0) {
        mi::Uint32 miplevels = image_file2->get_miplevels();
        if( mdl_file_path)
            LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
                "Loading image from MDL file path \"%s\", pixel type \"%s\", %ux%ux%u pixels, "
                "%u miplevel%s.",
                mdl_file_path, convert_pixel_type_enum_to_string( m_pixel_type),
                m_width, m_height, m_nr_of_layers, miplevels, miplevels == 1 ? "" : "s");
        else
            LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO, //-V576 PVS
                "Loading memory-based image, pixel type \"%s\", %ux%ux%u pixels, %u miplevel%s.",
                convert_pixel_type_enum_to_string( m_pixel_type),
                m_width, m_height, m_nr_of_layers, miplevels, miplevels == 1 ? "" : "s");
    }

    m_nr_of_tiles_x = (m_width  + m_tile_width  - 1) / m_tile_width;
    m_nr_of_tiles_y = (m_height + m_tile_height - 1) / m_tile_height;
    m_nr_of_tiles   = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;

    m_tiles = new mi::neuraylib::ITile*[m_nr_of_tiles];
    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
        m_tiles[i] = create_tile( m_pixel_type, m_tile_width, m_tile_height);

    for( mi::Uint32 z = 0; z < m_nr_of_layers; ++z)
        for( mi::Uint32 tile_y = 0; tile_y < m_nr_of_tiles_y; ++tile_y)
            for( mi::Uint32 tile_x = 0; tile_x < m_nr_of_tiles_x; ++tile_x) {
                mi::Uint32 index = z * m_nr_of_tiles_x*m_nr_of_tiles_y
                                   + tile_y * m_nr_of_tiles_x + tile_x;
                ASSERT( M_IMAGE, index < m_nr_of_tiles);
                mi::Uint32 pixel_x = tile_x * m_tile_width;
                mi::Uint32 pixel_y = tile_y * m_tile_height;
                if( !image_file2->read( m_tiles[index], pixel_x, pixel_y, z, m_miplevel)) {
                    LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                        "The image plugin failed to import %s.", log_identifier.c_str());
                    *errors = -5;
                    set_default_pink_dummy_canvas();
                    return;
                }
    }

    *errors = 0;
}

Canvas_impl::Canvas_impl( mi::neuraylib::ITile* tile, Float32 gamma)
{
    // check incorrect arguments
    ASSERT( M_IMAGE, tile);
    ASSERT( M_IMAGE, gamma >= 0);

    m_pixel_type     = convert_pixel_type_string_to_enum( tile->get_type());
    m_width          = tile->get_resolution_x();
    m_height         = tile->get_resolution_y();
    m_nr_of_layers   = 1;
    m_tile_width     = m_width;
    m_tile_height    = m_height;
    m_nr_of_tiles_x  = 1;
    m_nr_of_tiles_y  = 1;
    m_nr_of_tiles    = 1;
    m_miplevel       = 0;
    m_is_cubemap     = false;
    m_gamma          = gamma == 0.0f ? get_default_gamma( m_pixel_type) : gamma;

    m_tiles = new mi::neuraylib::ITile*[1];
    m_tiles[0] = tile;
    m_tiles[0]->retain();
}

Canvas_impl::~Canvas_impl()
{
    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
        if( m_tiles[i])
            m_tiles[i]->release();
    delete[] m_tiles;
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

const mi::neuraylib::ITile* Canvas_impl::get_tile(
    mi::Uint32 pixel_x, mi::Uint32 pixel_y, mi::Uint32 layer) const
{
    if( pixel_x >= m_width || pixel_y >= m_height || layer >= m_nr_of_layers)
        return 0;

    mi::Uint32 tile_x = pixel_x / m_tile_width;
    mi::Uint32 tile_y = pixel_y / m_tile_height;
    mi::Uint32 index = (layer * m_nr_of_tiles_y + tile_y) * m_nr_of_tiles_x + tile_x;
    ASSERT( M_IMAGE, index < m_nr_of_tiles);

    mi::base::Lock::Block block( &m_lock);

    if( m_tiles[index] == 0) {
        ASSERT( M_IMAGE, supports_lazy_loading());
#ifdef MI_IMAGE_LOAD_ONLY_REQUESTED_TILE
        m_tiles[index] = create_tile( m_pixel_type, m_tile_width, m_tile_height);
        load_tile( m_tiles[index], tile_x * m_tile_width, tile_y * m_tile_height, layer);
#else
        for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i) {
            ASSERT( M_IMAGE, !m_tiles[i]);
            m_tiles[i] = create_tile( m_pixel_type, m_tile_width, m_tile_height);
        }
        load_tile( 0, 0, 0, 0);
#endif
    }

    m_tiles[index]->retain();
    return m_tiles[index];
}

mi::neuraylib::ITile* Canvas_impl::get_tile(
    mi::Uint32 pixel_x, mi::Uint32 pixel_y, mi::Uint32 layer)
{
    if( pixel_x >= m_width || pixel_y >= m_height || layer >= m_nr_of_layers)
        return 0;

    mi::Uint32 tile_x = pixel_x / m_tile_width;
    mi::Uint32 tile_y = pixel_y / m_tile_height;
    mi::Uint32 index = (layer * m_nr_of_tiles_y + tile_y) * m_nr_of_tiles_x + tile_x;
    ASSERT( M_IMAGE, index < m_nr_of_tiles);

    mi::base::Lock::Block block( &m_lock);

    if( m_tiles[index] == 0) {

        ASSERT( M_IMAGE, supports_lazy_loading());
#ifdef MI_IMAGE_LOAD_ONLY_REQUESTED_TILE
        m_tiles[index] = create_tile( m_pixel_type, m_tile_width, m_tile_height);
        load_tile( m_tiles[index], tile_x * m_tile_width, tile_y * m_tile_height, layer);
#else
        for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i) {
            ASSERT( M_IMAGE, !m_tiles[i]);
            m_tiles[i] = create_tile( m_pixel_type, m_tile_width, m_tile_height);
        }
        load_tile( 0, 0, 0, 0);
#endif
    }

    m_tiles[index]->retain();
    return m_tiles[index];
}

mi::Size Canvas_impl::get_size() const
{
    mi::Size size = sizeof( *this);

    size += m_nr_of_tiles * sizeof( mi::neuraylib::ITile*); // m_tiles

    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)          // m_tiles[i]
        if( m_tiles[i]) {
            mi::base::Handle<ITile> tile_internal( m_tiles[i]->get_interface<ITile>());
            if( tile_internal.is_valid_interface())         // exact memory usage
                size += tile_internal->get_size();
            else                                            // approximate memory usage
                size +=   static_cast<size_t>( m_tile_width)
                        * static_cast<size_t>( m_tile_height)
                        * get_bytes_per_pixel( m_pixel_type);
        }

    return size;
}

bool Canvas_impl::supports_lazy_loading() const
{
    // either both m_archive_filename or m_member_filename are set or none
    ASSERT( M_IMAGE,  m_archive_filename.empty() || !m_member_filename.empty());
    ASSERT( M_IMAGE, !m_archive_filename.empty() ||  m_member_filename.empty());

    // m_filename and m_archive_filename (or m_member_filename) are not both set
    ASSERT( M_IMAGE,  m_filename.empty() || m_archive_filename.empty());

    if( !m_filename.empty())
        return true;
    if( m_archive_filename.empty())
        return false;

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::base::Handle<IMdl_container_callback> callback( image_module->get_mdl_container_callback());
    return callback.is_valid_interface();
}

void Canvas_impl::load_tile(
    mi::neuraylib::ITile* tile, mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) const
{
    ASSERT( M_IMAGE, supports_lazy_loading());

    std::string log_identifier;
    mi::base::Handle<mi::neuraylib::IReader> reader( get_reader( log_identifier));
    if( !reader)
        return;

    std::string root, extension;
    HAL::Ospath::splitext( !m_filename.empty() ? m_filename : m_member_filename, root, extension);
    if( !extension.empty() && extension[0] == '.' )
        extension = extension.substr( 1);

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( extension.c_str(), reader.get());
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle \"%s\".", log_identifier.c_str());
        return;
    }

    mi::base::Handle<mi::neuraylib::IImage_file> image_file(
        plugin->open_for_reading( reader.get()));
    if( !image_file.is_valid_interface()) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import \"%s\".",
            plugin->get_name(), log_identifier.c_str());
        return;
    }

    if( tile) {
        // load only the requested tile
        if( !image_file->read( tile, x, y, z, m_miplevel)) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                "The image plugin \"%s\" failed to import \"%s\".",
                plugin->get_name(), log_identifier.c_str());
            return;
        }
    } else {
        // load all tiles at once
        for( mi::Uint32 layer = 0; layer < m_nr_of_layers; ++layer)
            for( mi::Uint32 tile_y = 0; tile_y < m_nr_of_tiles_y; ++tile_y)
                for( mi::Uint32 tile_x = 0; tile_x < m_nr_of_tiles_x; ++tile_x) {
                    mi::Uint32 index = (layer*m_nr_of_tiles_y + tile_y) * m_nr_of_tiles_x + tile_x;
                    ASSERT( M_IMAGE, index < m_nr_of_tiles);
                    mi::Uint32 x = tile_x * m_tile_width;
                    mi::Uint32 y = tile_y * m_tile_height;
                    if( !image_file->read( m_tiles[index], x, y, layer, m_miplevel)) {
                         LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                             "The image plugin \"%s\" failed to import \"%s\".",
                             plugin->get_name(), log_identifier.c_str());
                         return;
                    }
                }
    }
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
            return 0;
        }

        reader->retain();
        return reader.get();

    }

    // archive-based
    if( !m_archive_filename.empty() && !m_member_filename.empty()) {

        log_identifier = m_archive_filename + "\" in \"" + m_member_filename;

        SYSTEM::Access_module<Image_module> image_module( false);
        mi::base::Handle<IMdl_container_callback> callback(
            image_module->get_mdl_container_callback());
        mi::base::Handle<mi::neuraylib::IReader> reader(
            callback->get_reader( m_archive_filename.c_str(), m_member_filename.c_str()));
        if( !reader) {
            LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
                 "Failed to open image file \"%s\".", log_identifier.c_str());
            return 0;
        }

        reader->retain();
        return reader.get();
    }

    ASSERT( M_IMAGE, false);
    return 0;
}

void Canvas_impl::set_default_pink_dummy_canvas()
{
    for( mi::Uint32 i = 0; m_tiles && i < m_nr_of_tiles; ++i)
        if( m_tiles[i])
            m_tiles[i]->release();
    delete[] m_tiles;

    m_filename.clear();
    m_archive_filename.clear();
    m_member_filename.clear();

    m_pixel_type    = PT_RGBA;
    m_width         = 1;
    m_height        = 1;
    m_tile_width    = 1;
    m_tile_height   = 1;
    m_nr_of_layers  = 1;
    m_nr_of_tiles_x = (m_width  + m_tile_width  - 1) / m_tile_width;
    m_nr_of_tiles_y = (m_height + m_tile_height - 1) / m_tile_height;
    m_nr_of_tiles   = m_nr_of_tiles_x * m_nr_of_tiles_y * m_nr_of_layers;
    m_is_cubemap    = false;

    m_tiles = new mi::neuraylib::ITile*[m_nr_of_tiles];
    for( mi::Uint32 i = 0; i < m_nr_of_tiles; ++i)
        m_tiles[i] = create_tile( m_pixel_type, m_tile_width, m_tile_height);

    mi::base::Handle<mi::neuraylib::ITile> tile( get_tile( 0, 0, 0));
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
}

} // namespace IMAGE

} // namespace MI
