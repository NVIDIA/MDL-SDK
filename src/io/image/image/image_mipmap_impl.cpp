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

#include "i_image.h"
#include "image_canvas_impl.h"
#include "image_mipmap_impl.h"

#include <mi/base/handle.h>
#include <mi/math/color.h>
#include <mi/base/types.h>
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

Mipmap_impl::Mipmap_impl()
{
    m_nr_of_levels = 1;
    m_nr_of_provided_levels = 1;
    m_levels.resize( m_nr_of_levels);
    m_levels[0] = new Canvas_impl( PT_RGBA, 1, 1, 1, 1, 1, false, 0.0f);
    mi::base::Handle<mi::neuraylib::ITile> tile( m_levels[0]->get_tile( 0, 0));
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
    m_last_created_level = 0;
    m_is_cubemap = false;
}

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
    Container_based,
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
        m_levels[i] = new Canvas_impl(
            Container_based(),
            reader,
            archive_filename,
            member_filename,
            i,
            tile_width,
            tile_height,
            image_file.get());

    m_is_cubemap = false;
    mi::base::Handle<ICanvas> canvas_internal( m_levels[0]->get_interface<ICanvas>());
    if( canvas_internal.is_valid_interface())
        m_is_cubemap = canvas_internal->get_is_cubemap();

    *errors = 0;
}

Mipmap_impl::Mipmap_impl(
    Memory_based,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* mdl_file_path,
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
        m_levels[i] = new Canvas_impl(
            Memory_based(),
            reader,
            image_format,
            mdl_file_path,
            i,
            tile_width,
            tile_height,
            image_file.get());

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

    SYSTEM::Access_module<Image_module> image_module(false);
    mi::base::Lock::Block block( &m_lock);

    // create miplevels if needed
    for( mi::Uint32 i = m_last_created_level+1; i <= level; ++i) {

        m_levels[i] = mi::base::make_handle(image_module->create_miplevel(
            m_levels[i-1].get(), m_levels[i-1]->get_gamma()));

        m_last_created_level = i;
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

    SYSTEM::Access_module<Image_module> image_module(false);
    mi::base::Lock::Block block( &m_lock);

    // create miplevels if needed
    for( mi::Uint32 i = m_last_created_level+1; i <= level; ++i) {

        m_levels[i] = mi::base::make_handle(image_module->create_miplevel(
            m_levels[i - 1].get(), m_levels[i - 1]->get_gamma()));

        m_last_created_level = i;
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

} // namespace IMAGE

} // namespace MI
