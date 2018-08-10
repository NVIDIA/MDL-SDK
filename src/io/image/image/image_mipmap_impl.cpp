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

#include "i_image.h"
#include "image_canvas_impl.h"
#include "image_mipmap_impl.h"

#include <mi/base/handle.h>
#include <mi/math/color.h>
#include <mi/math/function.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/iimage_plugin.h>

#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>

namespace MI {

namespace IMAGE {

Mipmap_impl::Mipmap_impl(
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

    m_nr_of_levels = 1 + mi::math::log2_int( std::min( width, height));
    m_nr_of_provided_levels = 0;

    m_levels.resize( m_nr_of_levels);
    m_levels[0] = new Canvas_impl(
        pixel_type, width, height, tile_width, tile_height, layers, is_cubemap, gamma);

    m_last_created_level = 0;
    m_is_cubemap = is_cubemap;
}

Mipmap_impl::Mipmap_impl(
    const std::string& filename,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    bool only_first_level,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    m_nr_of_levels = 1;
    m_nr_of_provided_levels = 0;
    m_levels.resize( m_nr_of_levels);
    m_levels[0] = new Canvas_impl( PT_RGBA, 1, 1, 1, 1, 1, false, 0.0f);
    mi::base::Handle<mi::neuraylib::ITile> tile( m_levels[0]->get_tile( 0, 0));
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
    m_last_created_level = 0;
    m_is_cubemap = false;

    DISK::File_reader_impl reader;
    if( !reader.open( filename.c_str())) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "Failed to open image file \"%s\".", filename.c_str());
        *errors = -3;
        return;
    }

    std::string root, extension;
    HAL::Ospath::splitext( filename, root, extension);
    if( !extension.empty() && extension[0] == '.' )
        extension = extension.substr( 1);

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( extension.c_str(), &reader);
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle \"%s\".", filename.c_str());
        *errors = -4;
        return;
    }

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_reading( &reader));
    if( !image_file.is_valid_interface()) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import \"%s\".",
            plugin->get_name(), filename.c_str());
        *errors = -5;
        return;
    }

    m_levels.clear();

    mi::Uint32 width  = image_file->get_resolution_x();
    mi::Uint32 height = image_file->get_resolution_y();

    m_nr_of_levels = 1 + mi::math::log2_int( std::min( width, height));
    m_nr_of_provided_levels = only_first_level ? 1 : image_file->get_miplevels();
    if( m_nr_of_provided_levels > m_nr_of_levels)
        m_nr_of_provided_levels = m_nr_of_levels;
    m_last_created_level = m_nr_of_provided_levels-1;

    m_levels.resize( m_nr_of_levels);

    for( mi::Uint32 i = 0; i < m_nr_of_provided_levels; ++i)
        m_levels[i] = new Canvas_impl( filename, i, tile_width, tile_height, image_file.get());

    m_is_cubemap = false;
    mi::base::Handle<ICanvas> canvas_internal( m_levels[0]->get_interface<ICanvas>());
    if( canvas_internal.is_valid_interface())
        m_is_cubemap = canvas_internal->get_is_cubemap();

    *errors = 0;
}

Mipmap_impl::Mipmap_impl(
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& member_filename,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    bool only_first_level,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    m_nr_of_levels = 1;
    m_nr_of_provided_levels = 0;
    m_levels.resize( m_nr_of_levels);
    m_levels[0] = new Canvas_impl( PT_RGBA, 1, 1, 1, 1, 1, false, 0.0f);
    mi::base::Handle<mi::neuraylib::ITile> tile( m_levels[0]->get_tile( 0, 0));
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
    m_last_created_level = 0;
    m_is_cubemap = false;

    if( !reader || !reader->supports_absolute_access()) {
        *errors = -3;
        return;
    }

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
        return;
    }

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_reading( reader));
    if( !image_file.is_valid_interface()) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import \"%s\" in \"%s\".",
            plugin->get_name(), member_filename.c_str(), archive_filename.c_str());
        *errors = -5;
        return;
    }

    m_levels.clear();

    mi::Uint32 width  = image_file->get_resolution_x();
    mi::Uint32 height = image_file->get_resolution_y();

    m_nr_of_levels = 1 + mi::math::log2_int( std::min( width, height));
    m_nr_of_provided_levels = only_first_level ? 1 : image_file->get_miplevels();
    if( m_nr_of_provided_levels > m_nr_of_levels)
        m_nr_of_provided_levels = m_nr_of_levels;
    m_last_created_level = m_nr_of_provided_levels-1;

    m_levels.resize( m_nr_of_levels);

    for( mi::Uint32 i = 0; i < m_nr_of_provided_levels; ++i)
        m_levels[i] = new Canvas_impl( reader, archive_filename, member_filename,
            i, tile_width, tile_height, image_file.get());

    m_is_cubemap = false;
    mi::base::Handle<ICanvas> canvas_internal( m_levels[0]->get_interface<ICanvas>());
    if( canvas_internal.is_valid_interface())
        m_is_cubemap = canvas_internal->get_is_cubemap();

    *errors = 0;
}

Mipmap_impl::Mipmap_impl(
    mi::neuraylib::IReader* reader,
    const char* image_format,
    mi::Uint32 tile_width,
    mi::Uint32 tile_height,
    bool only_first_level,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    m_nr_of_levels = 1;
    m_nr_of_provided_levels = 0;
    m_levels.resize( m_nr_of_levels);
    m_levels[0] = new Canvas_impl( PT_RGBA, 1, 1, 1, 1, 1, false, 0.0f);
    mi::base::Handle<mi::neuraylib::ITile> tile( m_levels[0]->get_tile( 0, 0));
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
    m_last_created_level = 0;
    m_is_cubemap = false;

    if( !reader || !reader->supports_absolute_access()) {
        *errors = -3;
        return;
    }

    SYSTEM::Access_module<Image_module> image_module( false);
    mi::neuraylib::IImage_plugin* plugin
        = image_module->find_plugin_for_import( image_format, reader);
    if( !plugin) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "No image plugin found to handle a memory-based image with image format \"%s\".",
            image_format);
        *errors = -4;
        return;
    }

    mi::base::Handle<mi::neuraylib::IImage_file> image_file( plugin->open_for_reading( reader));
    if( !image_file.is_valid_interface()) {
        LOG::mod_log->error( M_IMAGE, LOG::Mod_log::C_IO,
            "The image plugin \"%s\" failed to import a memory-based image with image format "
            "\"%s\".", plugin->get_name(), image_format);
        *errors = -5;
        return;
    }

    m_levels.clear();

    mi::Uint32 width  = image_file->get_resolution_x();
    mi::Uint32 height = image_file->get_resolution_y();

    m_nr_of_levels = 1 + mi::math::log2_int( std::min( width, height));
    m_nr_of_provided_levels = only_first_level ? 1 : image_file->get_miplevels();
    if( m_nr_of_provided_levels > m_nr_of_levels)
        m_nr_of_provided_levels = m_nr_of_levels;
    m_last_created_level = m_nr_of_provided_levels-1;

    m_levels.resize( m_nr_of_levels);

    for( mi::Uint32 i = 0; i < m_nr_of_provided_levels; ++i)
        m_levels[i] = new Canvas_impl( reader, image_format,
            i, tile_width, tile_height, image_file.get());

    m_is_cubemap = false;
    mi::base::Handle<ICanvas> canvas_internal( m_levels[0]->get_interface<ICanvas>());
    if( canvas_internal.is_valid_interface())
        m_is_cubemap = canvas_internal->get_is_cubemap();
}

Mipmap_impl::Mipmap_impl(
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> >& canvases, bool is_cubemap)
{
    ASSERT( M_IMAGE, canvases[0]);

    mi::Uint32 base_level_width  = canvases[0]->get_resolution_x();
    mi::Uint32 base_level_height = canvases[0]->get_resolution_y();

    m_nr_of_levels = 1 + mi::math::log2_int( std::min( base_level_width, base_level_height));
    m_nr_of_provided_levels = static_cast<mi::Uint32>( canvases.size());
    if( m_nr_of_provided_levels > m_nr_of_levels)
        m_nr_of_provided_levels = m_nr_of_levels;
    m_last_created_level = m_nr_of_provided_levels-1;

    m_levels.resize( m_nr_of_levels);

    for( mi::Uint32 i = 0; i < m_nr_of_provided_levels; ++i)
        m_levels[i] = make_handle_dup( canvases[i].get());

    m_is_cubemap = is_cubemap;
}

mi::Uint32 Mipmap_impl::get_nlevels() const
{
    return m_nr_of_levels;
}

const mi::neuraylib::ICanvas* Mipmap_impl::get_level( mi::Uint32 level) const
{
    // level 0 is guaranteed to exist, no locking needed
    if( level == 0) {
        m_levels[0]->retain();
        return m_levels[0].get();
    }

    if( level >= m_nr_of_levels)
        return 0;

    mi::base::Lock::Block block( &m_lock);

    // create miplevels if needed
    for( mi::Uint32 i = m_last_created_level+1; i <= level; ++i) {
        create_miplevel( i);
        ASSERT( M_IMAGE, m_last_created_level == i);
    }

    ASSERT( M_IMAGE, m_last_created_level >= level);
    ASSERT( M_IMAGE, m_levels[level]);
    m_levels[level]->retain();
    return m_levels[level].get();
}

mi::neuraylib::ICanvas* Mipmap_impl::get_level( mi::Uint32 level) //-V659 PVS
{
    if( level >= m_nr_of_levels)
        return 0;

    mi::base::Lock::Block block( &m_lock);

    // create miplevels if needed
    for( mi::Uint32 i = m_last_created_level+1; i <= level; ++i) {
        create_miplevel( i);
        ASSERT( M_IMAGE, m_last_created_level == i);
    }
    ASSERT( M_IMAGE, m_last_created_level >= level);

    // destroy higher levels if needed
    mi::Uint32 first_level_to_destroy = std::max( level+1, m_nr_of_provided_levels);
    for( mi::Uint32 i = first_level_to_destroy; i <= m_last_created_level; ++i)
        m_levels[i] = 0;
    m_last_created_level = first_level_to_destroy - 1;

    ASSERT( M_IMAGE, m_last_created_level >= level);
    ASSERT( M_IMAGE, m_levels[level]);
    m_levels[level]->retain();
    return m_levels[level].get();
}

mi::Size Mipmap_impl::get_size() const
{
    mi::Size size = sizeof( *this);

    mi::base::Lock::Block block( &m_lock);

    size += m_nr_of_levels * sizeof( mi::neuraylib::ICanvas*);   // m_levels

    for( mi::Uint32 i = 0; i <= m_last_created_level; ++i) {     // m_level[i]
        mi::base::Handle<ICanvas> canvas_internal( m_levels[i]->get_interface<ICanvas>());
        if( canvas_internal.is_valid_interface())                // exact memory usage
            size += canvas_internal->get_size();
        else  {                                                  // approximate memory usage
            mi::Size width  = m_levels[i]->get_resolution_x();
            mi::Size height = m_levels[i]->get_resolution_y();
            Pixel_type pixel_type
                = convert_pixel_type_string_to_enum( m_levels[i]->get_type());
            size += width * height * get_bytes_per_pixel( pixel_type);
        }
    }

    return size;
}

void Mipmap_impl::create_miplevel( mi::Uint32 level) const
{
    // NOTE: This implementation creates the new miplevel tile by tile. For each tile, it retrieves
    // the at most four needed tiles from the previous miplevel *once* (and not for every pixel).
    // Remember that tile lookups require locks (and reference counts).

    ASSERT( M_IMAGE, level > 0);
    ASSERT( M_IMAGE, m_last_created_level == level-1);

    mi::neuraylib::ICanvas* prev_canvas = m_levels[level-1].get();
    ASSERT( M_IMAGE, prev_canvas);

    // Get properties of previous miplevel
    mi::Uint32 prev_width       = prev_canvas->get_resolution_x();
    mi::Uint32 prev_height      = prev_canvas->get_resolution_y();
    mi::Uint32 prev_layers      = prev_canvas->get_layers_size();
    mi::Uint32 prev_tile_width  = prev_canvas->get_tile_resolution_x();
    mi::Uint32 prev_tile_height = prev_canvas->get_tile_resolution_y();
    Pixel_type prev_pixel_type
        = convert_pixel_type_string_to_enum( prev_canvas->get_type());
    mi::Float32 prev_gamma      =  prev_canvas->get_gamma();

    // Compute properties of this miplevel
    mi::Uint32 width            = std::max( prev_width  / 2, 1u);
    mi::Uint32 height           = std::max( prev_height / 2, 1u);
    mi::Uint32 layers           = prev_layers;
    mi::Uint32 tile_width       = std::min( prev_tile_width,  width);
    mi::Uint32 tile_height      = std::min( prev_tile_height, height);
    Pixel_type pixel_type       = prev_pixel_type;
    mi::Float32 gamma           = prev_gamma;

    // The last level is just a single pixel in at least one dimension (maybe with multiple layers).
    ASSERT( M_IMAGE, (width == 1 || height == 1) || (level < m_nr_of_levels-1));

    // Create the miplevel
    mi::neuraylib::ICanvas* canvas = new Canvas_impl(
        pixel_type, width, height, tile_width, tile_height, layers, m_is_cubemap, gamma);

    mi::Uint32 nr_of_tiles_x = (width  + tile_width  - 1) / tile_width;
    mi::Uint32 nr_of_tiles_y = (height + tile_height - 1) / tile_height;

    mi::Uint32 offsets_x[4] = { 0, 1, 0, 1};
    mi::Uint32 offsets_y[4] = { 0, 0, 1, 1};

    // Loop over the tiles and compute the pixel data for each tile
    for( mi::Uint32 tile_z = 0; tile_z < layers; ++tile_z)
        for( mi::Uint32 tile_y = 0; tile_y < nr_of_tiles_y; ++tile_y)
            for( mi::Uint32 tile_x = 0; tile_x < nr_of_tiles_x; ++tile_x) {

                // The current tile covers pixels in the range [x_begin,x_end) x [y_begin,y_end)
                // from the canvas for this miplevel.
                mi::Uint32 x_begin = tile_x * tile_width;
                mi::Uint32 y_begin = tile_y * tile_height;
                mi::Uint32 x_end   = std::min( x_begin + tile_width,  width);
                mi::Uint32 y_end   = std::min( y_begin + tile_height, height);

                // Lookup tile for this miplevel
                mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( x_begin, y_begin));

                // The current tile corresponds to the range
                // [prev_x_begin,prev_x_end) x [prev_y_begin,prev_y_end) in the previous miplevel.
                mi::Uint32 prev_x_begin = 2 * x_begin;
                mi::Uint32 prev_y_begin = 2 * y_begin;
                mi::Uint32 prev_x_end   = std::min( 2*x_end, 2*x_begin + prev_width);
                mi::Uint32 prev_y_end   = std::min( 2*y_end, 2*y_begin + prev_height);

                // Lookup involved tiles from the previous miplevel (note that these tiles are not
                // necessarily distinct).
                mi::base::Handle<mi::neuraylib::ITile> prev_tiles[4];
                prev_tiles[0] = prev_canvas->get_tile( prev_x_begin, prev_y_begin);
                prev_tiles[1] = prev_canvas->get_tile( prev_x_end-1, prev_y_begin);
                prev_tiles[2] = prev_canvas->get_tile( prev_x_begin, prev_y_end-1);
                prev_tiles[3] = prev_canvas->get_tile( prev_x_end-1, prev_y_end-1);
                ASSERT( M_IMAGE, prev_tiles[0].is_valid_interface());
                ASSERT( M_IMAGE, prev_tiles[1].is_valid_interface());
                ASSERT( M_IMAGE, prev_tiles[2].is_valid_interface());
                ASSERT( M_IMAGE, prev_tiles[3].is_valid_interface());

                // Loop over the pixels of this tile and compute the value for each pixel
                for( mi::Uint32 y = 0; y < y_end - y_begin; ++y)
                    for( mi::Uint32 x = 0; x < x_end - x_begin; ++x) {

                        // The current pixel (x,y) corresponds to the four pixels
                        // [prev_x, prev_x+1] x [prev_y,prev_y+1] in the tiles of the previous
                        // layer. Note that all four pixels might actually be in a different tile.
                        mi::Uint32 prev_x = 2 * x;
                        mi::Uint32 prev_y = 2 * y;

                        mi::math::Color color( 0.0f, 0.0f, 0.0f, 0.0f);
                        mi::Uint32 nr_of_summands = 0;

                        // Loop over the at most four pixels corresponding to pixel (x,y)
                        for( mi::Uint32 i = 0; i < 4; ++i) {

                            // Find tile of pixel
                            // (prev_x + offsets_x[i], prev_y + offsets_y[i]) and its coordinates
                            // with respect to that tile.
                            mi::Uint32 prev_tile_id = 0; // the ID is the index for prev_tiles
                            mi::Uint32 prev_actual_x = prev_x + offsets_x[i];
                            if( prev_x_begin + prev_actual_x >= prev_width)
                                continue;
                            if( prev_actual_x >= prev_tile_width) {
                                prev_actual_x -= prev_tile_width;
                                prev_tile_id += 1;
                            }
                            mi::Uint32 prev_actual_y = prev_y + offsets_y[i];
                            if( prev_y_begin + prev_actual_y >= prev_height)
                                continue;
                            if( prev_actual_y >= prev_tile_height) {
                                prev_actual_y -= prev_tile_height;
                                prev_tile_id += 2;
                            }

                            // The pixel (prev_x + offsets_x[i], prev_y + offsets_y[i]) actually
                            // is pixel (prev_actual_x, prev_actual_y) in tile
                            // prev_tiles[prev_tile_id].
                            mi::math::Color prev_color;
                            prev_tiles[prev_tile_id]->get_pixel(
                                prev_actual_x, prev_actual_y, &prev_color.r);
                            color += prev_color;
                            nr_of_summands += 1;
                        }

                        color /= static_cast<mi::Float32>( nr_of_summands);
                        tile->set_pixel( x, y, &color.r);
                    }
            }

    m_levels[level] = canvas;
    m_last_created_level = level;
}

} // namespace IMAGE

} // namespace MI
