/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "i_dbimage.h"

#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/itile.h>

#include <boost/core/ignore_unused.hpp>

#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/data/serial/i_serializer.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/i_image_utilities.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

namespace MI {

namespace DBIMAGE {

namespace {

std::string g_empty_str;

// Returns a string representation of mi::base::Uuid
std::string hash_to_string( const mi::base::Uuid& hash)
{
    char buffer[35];
    snprintf( buffer, sizeof( buffer), "0x%08x%08x%08x%08x",
              hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
    return buffer;
}

} // namespace

IMAGE::IMipmap* Image_set::create_mipmap( mi::Size f, mi::Size i, mi::Sint32& errors) const
{
    ASSERT( M_SCENE, f < get_length());
    ASSERT( M_SCENE, i < get_frame_length( f));

    errors = 0;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    // mdl container based
    if( is_mdl_container()) {
        mi::base::Handle<mi::neuraylib::IReader> reader( open_reader( f, i));
        if( !reader) {
            errors = -5;
            return image_module->create_dummy_mipmap();
        }

        const char* container_filename = get_container_filename();
        const char* container_membername = get_container_membername( f, i);
        ASSERT( M_SCENE, get_container_filename());
        ASSERT( M_SCENE, get_container_membername( f, i));
        const char* selector = get_selector();
        return image_module->create_mipmap(
            IMAGE::Container_based(),
            reader.get(),
            container_filename,
            container_membername,
            selector,
            /*only_first_level*/ true,
            &errors);
    }

    // file based
    const char* resolved_filename = get_resolved_filename( f, i);
    if( resolved_filename && (resolved_filename[0] != '\0')) {
        const char* selector = get_selector();
        return image_module->create_mipmap(
            IMAGE::File_based(),
            resolved_filename,
            selector,
            /*only_first_level*/ true,
            &errors);
    }

    // canvas based
    mi::base::Handle<mi::neuraylib::ICanvas> canvas( get_canvas( f, i));
    if( canvas) {
        std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
        canvases[0] = canvas;
        return image_module->create_mipmap( canvases);
    }

    // reader based
    mi::base::Handle<mi::neuraylib::IReader> reader( open_reader( f, i));
    if( reader) {
        const char* image_format = get_image_format();
        const char* mdl_file_path = get_mdl_file_path();
        ASSERT( M_SCENE, image_format);
        const char* selector = get_selector();
        return image_module->create_mipmap(
            IMAGE::Memory_based(),
            reader.get(),
            image_format,
            selector,
            mdl_file_path && (mdl_file_path[0] != '\0') ? mdl_file_path : nullptr,
            /*only_first_level*/ true,
            &errors);
    }

    errors = -99;
    return image_module->create_dummy_mipmap();
}

// Used by File_image_set.
struct Uv_frame_filename
{
    mi::Size frame_number = 0;
    mi::Sint32 u = 0;
    mi::Sint32 v = 0;
    std::string filename;
};

/// Implementation for file-base image sets (no containers).
///
/// Supports single files, animated textures, and uvtiles.
class File_image_set : public Image_set
{
public:
    /// Constructor for a single file (no animated textures nor uvtiles)
    File_image_set(
        const std::string& original_filename,
        const char* selector,
        const std::string& resolved_filename)
      : m_original_filename( original_filename),
        m_selector( selector ? selector : "")
    {
        m_array.push_back( Uv_frame_filename{ 0, 0, 0, resolved_filename});
        m_frame_number_to_frame_index[0] = 0;
        m_frame_index_to_frame_number.push_back( 0);
        m_global_indices_per_frame_index.resize( 1);
        m_global_indices_per_frame_index[0].push_back( 0);
        m_is_animated = false;
        m_is_uvtile = false;
    }

    /// Constructor that supports animated textures and uvtiles (array can be unordered).
    File_image_set(
        const std::string& original_filename,
        const char* selector,
        const std::vector<Uv_frame_filename>& array,
        bool is_animated,
        bool is_uvtile)
      : m_original_filename( original_filename),
        m_selector( selector ? selector : ""),
        m_array( array),
        m_is_animated( is_animated),
        m_is_uvtile( is_uvtile)
    {
        mi::Size n = array.size();
        ASSERT( M_SCENE, n > 0);

        // Collect all frame numbers
        for( mi::Size i = 0; i < n; ++i) {
            mi::Size frame_number = array[i].frame_number;
            m_frame_number_to_frame_index[frame_number] = 0;
        }

        // Assign frame indices
        mi::Size id = 0;
        mi::Size n_frames = m_frame_number_to_frame_index.size();
        m_frame_index_to_frame_number.resize( n_frames);
        for( auto& elem: m_frame_number_to_frame_index) {
            m_frame_index_to_frame_number[id] = elem.first;
            elem.second = id++;
        }

        // Collect global indices per frame index
        m_global_indices_per_frame_index.resize( n_frames);
        for( mi::Size i = 0; i < n; ++i) {
            mi::Size frame_number = array[i].frame_number;
            mi::Size frame_id = m_frame_number_to_frame_index.at( frame_number);
            m_global_indices_per_frame_index[frame_id].push_back( i);
        }
    }

    bool is_mdl_container() const { return false; }

    const char* get_original_filename() const { return m_original_filename.c_str(); }

    const char* get_container_filename() const { return ""; }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_selector() const { return !m_selector.empty() ? m_selector.c_str() : nullptr; }

    const char* get_image_format() const { return ""; }

    bool is_animated() const { return m_is_animated; }

    bool is_uvtile() const { return m_is_uvtile; }

    mi::Size get_length() const { return m_frame_index_to_frame_number.size(); }

    mi::Size get_frame_number( mi::Size f) const
    {
        ASSERT( M_SCENE, f < get_length());
        return m_frame_index_to_frame_number[f];
    }

    mi::Size get_frame_length( mi::Size f) const
    {
        ASSERT( M_SCENE, f < get_length());
        return m_global_indices_per_frame_index[f].size();
    }

    void get_uvtile_uv( mi::Size f, mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        mi::Size index = get_global_index( f, i);
        u = m_array[index].u;
        v = m_array[index].v;
    }

    const char* get_resolved_filename( mi::Size f, mi::Size i) const
    {
        mi::Size index = get_global_index( f, i);
        return m_array[index].filename.c_str();
    }

    const char* get_container_membername( mi::Size f, mi::Size i) const { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size f, mi::Size i) const { return nullptr; }

    mi::neuraylib::ICanvas* get_canvas( mi::Size f, mi::Size i) const { return nullptr; }

private:
    mi::Size get_global_index( mi::Size f, mi::Size i) const
    {
        ASSERT( M_SCENE, f < get_length());
        const std::vector<mi::Size>& element = m_global_indices_per_frame_index[f];
        ASSERT( M_SCENE, i < element.size());
        ASSERT( M_SCENE, element[i] < m_array.size());
        return element[i];
    }

    std::string m_original_filename;
    std::string m_selector;

    std::vector<Uv_frame_filename> m_array;

    /// Maps frame numbers to frame indices.
    std::map<mi::Size, mi::Size> m_frame_number_to_frame_index;
    /// Maps frame index to frame number.
    std::vector<mi::Size> m_frame_index_to_frame_number;
    /// Stores global indices into m_array per frame index.
    std::vector<std::vector<mi::Size>> m_global_indices_per_frame_index;
    /// Indicates whether this is an animated texture.
    bool m_is_animated;
    /// Indicates whether any frame has uvtiles.
    bool m_is_uvtile;
};

/// Implementation for container-based image sets.
///
/// Does not support animated textures nor uvtiles.
class Container_file_image_set : public Image_set
{
public:
    Container_file_image_set(
        const std::string& resolved_container_filename,
        const std::string& container_member_name,
        const char* selector)
      : m_resolved_container_filename( resolved_container_filename),
        m_container_member_name( container_member_name),
        m_selector( selector ? selector : "")
    {
    }

    bool is_mdl_container() const { return true; }

    const char* get_original_filename() const { return ""; }

    const char* get_container_filename() const { return m_resolved_container_filename.c_str(); }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_selector() const { return !m_selector.empty() ? m_selector.c_str() : nullptr; }

    const char* get_image_format() const { return ""; }

    bool is_animated() const { return false; }

    bool is_uvtile() const { return false; }

    mi::Size get_length() const { return 1; }

    mi::Size get_frame_number( mi::Size f) const
    {
        ASSERT( M_SCENE, f < get_length());
        return 0;
    }

    mi::Size get_frame_length( mi::Size f) const
    {
        ASSERT( M_SCENE, f < get_length());
        return 1;
    }

    void get_uvtile_uv( mi::Size f, mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT( M_SCENE, f < get_length());
        ASSERT( M_SCENE, i < get_frame_length( f));
        u = 0;
        v = 0;
    }

    const char* get_resolved_filename( mi::Size f, mi::Size i) const { return ""; }

    const char* get_container_membername( mi::Size f, mi::Size i) const
    {
        ASSERT( M_SCENE, f < get_length());
        ASSERT( M_SCENE, i < get_frame_length( f));
        return m_container_member_name.c_str();
    }

    mi::neuraylib::IReader* open_reader( mi::Size f, mi::Size i) const
    {
        ASSERT( M_SCENE, f < get_length());
        ASSERT( M_SCENE, i < get_frame_length( f));
        return MDL::get_container_resource_reader(
            m_resolved_container_filename, m_container_member_name);
    }

    mi::neuraylib::ICanvas* get_canvas( mi::Size f, mi::Size i) const { return nullptr; }

private:
    std::string m_resolved_container_filename;
    std::string m_container_member_name;
    std::string m_selector;
};

Uv_to_id::Uv_to_id( mi::Sint32 min_u, mi::Sint32 max_u, mi::Sint32 min_v, mi::Sint32 max_v)
{
    ASSERT( M_SCENE, min_u <= max_u);
    ASSERT( M_SCENE, min_v <= max_v);

    m_count_u = max_u - min_u + 1;
    m_count_v = max_v - min_v + 1;
    m_min_u   = min_u;
    m_min_v   = min_v;
    m_ids.resize( (mi::Size)m_count_u * m_count_v, (mi::Size)m_count_u * m_count_v == 1 ? 0 : ~0u);
}

mi::Uint32 Uv_to_id::get( mi::Sint32 u, mi::Sint32 v) const
{
    const mi::Sint32 uu = u - m_min_u;
    const mi::Sint32 vv = v - m_min_v;

    if( /*uu < 0 ||*/ static_cast<Uint32>( uu) >= static_cast<Uint32>( m_count_u))
        return ~0u;
    if( /*vv < 0 ||*/ static_cast<Uint32>( vv) >= static_cast<Uint32>( m_count_v))
        return ~0u;

    return m_ids[vv * m_count_u + uu];
}

bool Uv_to_id::set( mi::Sint32 u, mi::Sint32 v, mi::Uint32 id)
{
    const mi::Sint32 uu = u - m_min_u;
    const mi::Sint32 vv = v - m_min_v;

    if( /*uu < 0 ||*/ static_cast<Uint32>( uu) >= static_cast<Uint32>( m_count_u))
        return false;
    if( /*vv < 0 ||*/ static_cast<Uint32>( vv) >= static_cast<Uint32>( m_count_v))
        return false;

    mi::Uint32& old_id = m_ids[vv * m_count_u + uu];
    // Do not treat old_id == id as failure. This no-op allows to treat
    // single-tile setups in the same way as multi-tile setups.
    if( old_id != ~0u && old_id != id)
        return false;

    old_id = id;
    return true;
}

Image::Image()
  : m_frames_filenames( 1),
    m_impl_tag( DB::Tag()),
    m_impl_hash{0,0,0,0},
    m_cached_is_valid( false),
    m_cached_is_animated( false),
    m_cached_is_uvtile( false),
    m_cached_is_cubemap( false),
    m_cached_frames( 1),
    m_cached_frame_to_id( { 0 })
{
    m_frames_filenames[0].resize( 1);
    m_cached_frames[0].m_uvtiles.resize( 1);

    ASSERT( M_SCENE, m_frames_filenames.size() == m_cached_frames.size());
}

Image::Image( const Image& other)
  : SCENE::Scene_element<Image, ID_IMAGE>( other),
    m_frames_filenames( 1),
    m_impl_tag( DB::Tag()),
    m_impl_hash{0,0,0,0},
    m_cached_is_valid( false),
    m_cached_is_animated( false),
    m_cached_is_uvtile( false),
    m_cached_is_cubemap( false),
    m_cached_frames( 1),
    m_cached_frame_to_id( { 0 })
{
    m_frames_filenames[0].resize( 1);
    m_cached_frames[0].m_uvtiles.resize( 1);

    ASSERT( M_SCENE, m_frames_filenames.size() == m_cached_frames.size());
}

Image::~Image()
{
}

mi::Sint32 Image::reset_file(
    DB::Transaction* transaction,
    const std::string& original_filename,
    const char* selector,
    const mi::base::Uuid& impl_hash)
{
    mi::base::Handle<Image_set> image_set( resolve_filename( original_filename, selector));
    if( !image_set)
        return -4;

    return reset_image_set( transaction, image_set.get(), impl_hash);
}

mi::Sint32 Image::reset_reader(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const char* selector,
    const mi::base::Uuid& impl_hash)
{
    mi::Sint32 errors = 0;
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap(
        IMAGE::Memory_based(),
        reader,
        image_format,
        selector,
        /*mdl_file_path*/ nullptr,
        /*only_first_level*/ true,
        &errors));
    if( errors != 0)
        return errors;

    // Convert data from mipmap into temporary variables
    Frames tmp_frames( 1);
    tmp_frames[0].m_uvtiles.resize( 1);
    tmp_frames[0].m_uvtiles[0].m_mipmap = mipmap;
    Frame_to_id tmp_frame_to_id( { 0 });
    Frames_filenames tmp_frames_filenames( 1);
    tmp_frames_filenames[0].resize( 1);
    bool tmp_is_animated = false;
    bool tmp_is_uvtile = false;

    reset_shared(
        transaction, tmp_is_animated, tmp_is_uvtile, tmp_frames, tmp_frame_to_id, impl_hash);

    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();
    m_frames_filenames = tmp_frames_filenames;
    m_selector = selector ? selector : "";

    ASSERT( M_SCENE, m_frames_filenames.size() == m_cached_frames.size());

    return 0;
}

mi::Sint32 Image::reset_image_set(
    DB::Transaction* transaction, const Image_set* image_set, const mi::base::Uuid& impl_hash)
{
    if( !image_set)
        return -1;

    const bool tmp_is_animated = image_set->is_animated();
    const bool tmp_is_uvtile   = image_set->is_uvtile();

    const std::string tmp_resolved_container_filename = image_set->get_container_filename();
    const std::string tmp_original_filename           = image_set->get_original_filename();
    const std::string tmp_mdl_file_path               = image_set->get_mdl_file_path();
    const char* const tmp_selector_cstr               = image_set->get_selector();

    const mi::Size number_of_frames = image_set->get_length();

    Frames tmp_frames;
    Frames_filenames tmp_frames_filenames;
    Frame_to_id tmp_frame_to_id;

    // Convert data from image set into temporary variables
    for( mi::Size f = 0; f < number_of_frames; ++f) {

        const mi::Size number_of_tiles = image_set->get_frame_length( f);
        if( number_of_tiles == 0)
            return -1;

        const mi::Size frame_number = image_set->get_frame_number( f);
        if( (f > 0) && (frame_number <= tmp_frames.back().m_frame_number)) {
            ASSERT( M_SCENE, !"wrong frame order");
            return -99;
        }

        // Compute min/max u/v value of all tiles for this frame.
        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        image_set->get_uvtile_uv( f, 0, u, v);
        mi::Sint32 min_u = u;
        mi::Sint32 max_u = u;
        mi::Sint32 min_v = v;
        mi::Sint32 max_v = v;
        for( mi::Size i = 1; i < number_of_tiles; ++i) {
            image_set->get_uvtile_uv( f, i, u, v);
            min_u = mi::math::min( min_u, u);
            max_u = mi::math::max( max_u, u);
            min_v = mi::math::min( min_v, v);
            max_v = mi::math::max( max_v, v);
        }

        Frame frame;
        frame.m_frame_number = frame_number;
        frame.m_uvtiles.resize( number_of_tiles);
        frame.m_uv_to_id = Uv_to_id( min_u, max_u, min_v, max_v);
        Frame_filenames frame_filenames;
        frame_filenames.resize( number_of_tiles);

        // Convert data for this frame into temporary variables
        for( mi::Size i = 0; i < number_of_tiles; ++i) {

            image_set->get_uvtile_uv( f, i, u, v);
            if( !frame.m_uv_to_id.set( u, v, static_cast<mi::Uint32>( i)))
                return -12;

            Uvtile& tile = frame.m_uvtiles[i];
            tile.m_u = u;
            tile.m_v = v;
            mi::Sint32 errors = 0;
            tile.m_mipmap = image_set->create_mipmap( f, i, errors);
            if( errors != 0)
                return errors;

            Uvfilenames& filenames = frame_filenames[i];
            filenames.m_resolved_filename    = image_set->get_resolved_filename( f, i);
            filenames.m_container_membername = image_set->get_container_membername( f, i);
            if( !filenames.m_container_membername.empty())
                filenames.m_resolved_container_membername
                    = tmp_resolved_container_filename + ':' + filenames.m_container_membername;
        }

        tmp_frames.push_back( std::move( frame));
        tmp_frames_filenames.push_back( std::move( frame_filenames));
        tmp_frame_to_id.resize(
            std::max( tmp_frame_to_id.size(), static_cast<size_t>( frame_number+1)),
            static_cast<mi::Size>( -1));
        tmp_frame_to_id[frame_number] = f;
    }

    reset_shared(
        transaction, tmp_is_animated, tmp_is_uvtile, tmp_frames, tmp_frame_to_id, impl_hash);

    m_frames_filenames            = tmp_frames_filenames;
    m_resolved_container_filename = tmp_resolved_container_filename;
    m_original_filename           = tmp_original_filename;
    m_mdl_file_path               = tmp_mdl_file_path;
    m_selector                    = tmp_selector_cstr ? tmp_selector_cstr : "";

    ASSERT( M_SCENE, m_frames_filenames.size() == m_cached_frames.size());

    return 0;
}

void Image::set_mipmap(
    DB::Transaction* transaction,
    IMAGE::IMipmap* mipmap,
    const char* selector,
    const mi::base::Uuid& impl_hash)
{
    mi::base::Handle<IMAGE::IMipmap> tmp_mipmap;
    if( !mipmap) {
        SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
        tmp_mipmap = image_module->create_dummy_mipmap();
    } else
        tmp_mipmap = make_handle_dup( mipmap);

    // Convert data from mipmap into temporary variables
    Frames tmp_frames( 1);
    tmp_frames[0].m_uvtiles.resize( 1);
    tmp_frames[0].m_uvtiles[0].m_mipmap = tmp_mipmap;
    Frame_to_id tmp_frame_to_id( { 0 });
    Frames_filenames tmp_frames_filenames( 1);
    tmp_frames_filenames[0].resize( 1);
    bool tmp_is_animated = false;
    bool tmp_is_uvtile = false;

    reset_shared(
        transaction, tmp_is_animated, tmp_is_uvtile, tmp_frames, tmp_frame_to_id, impl_hash);

    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();
    m_frames_filenames = tmp_frames_filenames;
    m_selector = selector ? selector : "";

    ASSERT( M_SCENE, m_frames_filenames.size() == m_cached_frames.size());
}

const IMAGE::IMipmap* Image::get_mipmap(
    DB::Transaction* transaction, mi::Size frame_id, mi::Size uvtile_id) const
{
    if( !m_cached_is_valid || frame_id >= m_cached_frames.size()) {
        SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
        return image_module->create_dummy_mipmap();
    }

    const Frame& frame = m_cached_frames[frame_id];
    if( uvtile_id >= frame.m_uvtiles.size()) {
        SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
        return image_module->create_dummy_mipmap();
    }

    DB::Access<Image_impl> impl( m_impl_tag, transaction);
    return impl->get_mipmap( frame_id, uvtile_id);
}

const std::string& Image::get_filename( mi::Size frame_id, mi::Size uvtile_id) const
{
    if( frame_id >= m_frames_filenames.size())
        return g_empty_str;

    const Frame_filenames& framefn = m_frames_filenames[frame_id];
    if( uvtile_id >= framefn.size())
        return g_empty_str;

    return framefn[uvtile_id].m_resolved_filename;
}

bool Image::is_file_based() const
{
    return !m_frames_filenames.empty()
        && !m_frames_filenames[0].empty()
        && !m_frames_filenames[0][0].m_resolved_filename.empty();
}

const std::string& Image::get_container_membername( mi::Size frame_id, mi::Size uvtile_id) const
{
    if( frame_id >= m_frames_filenames.size())
        return g_empty_str;

    const Frame_filenames& framefn = m_frames_filenames[frame_id];
    if( uvtile_id >= framefn.size())
        return g_empty_str;

    return framefn[uvtile_id].m_container_membername;
}

const std::string& Image::get_resolved_container_membername(
    mi::Size frame_id, mi::Size uvtile_id) const
{
    if( frame_id >= m_frames_filenames.size())
        return g_empty_str;

    const Frame_filenames& framefn = m_frames_filenames[frame_id];
    if( uvtile_id >= framefn.size())
        return g_empty_str;

    return framefn[uvtile_id].m_resolved_container_membername;
}

const std::string& Image::get_original_filename() const
{
    return m_original_filename;
}

const std::string& Image::get_mdl_file_path( ) const
{
    return m_mdl_file_path;
}

const std::string& Image::get_selector( ) const
{
    return m_selector;
}

mi::Size Image::get_frame_number( mi::Size frame_id) const
{
    if( frame_id >= m_cached_frames.size())
        return 0;

    return m_cached_frames[frame_id].m_frame_number;
}

mi::Size Image::get_frame_id( mi::Size frame_number) const
{
    if( frame_number >= m_cached_frame_to_id.size())
        return -1;

    return m_cached_frame_to_id[frame_number];
}

mi::Size Image::get_frame_length( mi::Size frame_id) const
{
    if( frame_id >= m_cached_frames.size())
        return 0;

    return m_cached_frames[frame_id].m_uvtiles.size();
}

void Image::get_uvtile_uv_ranges(
    mi::Size frame_id,
    mi::Sint32& min_u,
    mi::Sint32& min_v,
    mi::Sint32& max_u,
    mi::Sint32& max_v) const
{
    if( frame_id >= m_cached_frames.size()) {
        min_u = min_v = max_u = max_v = 0;
        return;
    }

    const Frame& frame = m_cached_frames[frame_id];

    min_u = frame.m_uv_to_id.m_min_u;
    min_v = frame.m_uv_to_id.m_min_v;
    max_u = frame.m_uv_to_id.m_min_u + frame.m_uv_to_id.m_count_u - 1;
    max_v = frame.m_uv_to_id.m_min_v + frame.m_uv_to_id.m_count_v - 1;
}

mi::Sint32 Image::get_uvtile_uv(
    mi::Size frame_id, mi::Size uvtile_id, mi::Sint32& u, mi::Sint32& v) const
{
    if( frame_id >= m_cached_frames.size())
        return -1;

    const Frame& frame = m_cached_frames[frame_id];
    if( uvtile_id >= frame.m_uvtiles.size())
        return -1;

    u = frame.m_uvtiles[uvtile_id].m_u;
    v = frame.m_uvtiles[uvtile_id].m_v;
    return 0;
}

mi::Size Image::get_uvtile_id( mi::Size frame_id, mi::Sint32 u, mi::Sint32 v) const
{
    if( frame_id >= m_cached_frames.size())
        return -1;

    return m_cached_frames[frame_id].m_uv_to_id.get( u, v);
}

const SERIAL::Serializable* Image::serialize( SERIAL::Serializer* serializer) const
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    const bool remote = serializer->is_remote();

    Scene_element_base::serialize( serializer);

    SERIAL::write( serializer, HAL::Ospath::sep());

    SERIAL::write( serializer, remote ? "" : m_original_filename);
    SERIAL::write( serializer, remote ? "" : m_mdl_file_path);
    SERIAL::write( serializer, remote ? "" : m_resolved_container_filename);

    serializer->write_size_t( m_frames_filenames.size());
    for( const auto& frame_filenames: m_frames_filenames) {

        serializer->write_size_t( frame_filenames.size());
        for( const auto& uvfn: frame_filenames) {
            SERIAL::write( serializer, remote ? "" : uvfn.m_resolved_filename);
            SERIAL::write( serializer, remote ? "" : uvfn.m_container_membername);
            SERIAL::write( serializer, remote ? "" : uvfn.m_resolved_container_membername);
        }
    }

    SERIAL::write( serializer, remote ? "" : m_selector);

    SERIAL::write( serializer, m_impl_tag);
    SERIAL::write( serializer, m_impl_hash);

    SERIAL::write( serializer, m_cached_is_valid);
    SERIAL::write( serializer, m_cached_is_animated);
    SERIAL::write( serializer, m_cached_is_uvtile);
    SERIAL::write( serializer, m_cached_is_cubemap);

    serializer->write_size_t( m_cached_frames.size());
    for( const auto& frame: m_cached_frames) {

        SERIAL::write( serializer, frame.m_frame_number);

        serializer->write_size_t( frame.m_uvtiles.size());
        for( const auto& uvtile: frame.m_uvtiles) {
            SERIAL::write( serializer, uvtile.m_u);
            SERIAL::write( serializer, uvtile.m_v);
            ASSERT( M_SCENE, !uvtile.m_mipmap);
        }

        SERIAL::write( serializer, frame.m_uv_to_id.m_count_u);
        SERIAL::write( serializer, frame.m_uv_to_id.m_count_v);
        SERIAL::write( serializer, frame.m_uv_to_id.m_min_u);
        SERIAL::write( serializer, frame.m_uv_to_id.m_min_v);
        SERIAL::write( serializer, frame.m_uv_to_id.m_ids);
    }

    SERIAL::write( serializer, m_cached_frame_to_id);

    return this + 1;
}

SERIAL::Serializable* Image::deserialize( SERIAL::Deserializer* deserializer)
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    size_t s;

    Scene_element_base::deserialize( deserializer);

    std::string serializer_sep;
    SERIAL::read( deserializer, &serializer_sep);
    const bool convert_path =  serializer_sep != HAL::Ospath::sep();

    SERIAL::read( deserializer, &m_original_filename);
    SERIAL::read( deserializer, &m_mdl_file_path);
    SERIAL::read( deserializer, &m_resolved_container_filename);

    deserializer->read_size_t( &s);
    m_frames_filenames.resize( s);

    for( auto& frame_filenames: m_frames_filenames) {

        deserializer->read_size_t( &s);
        frame_filenames.resize( s);

        for( auto& uvfn: frame_filenames) {

            SERIAL::read( deserializer, &uvfn.m_resolved_filename);
            SERIAL::read( deserializer, &uvfn.m_container_membername);
            SERIAL::read( deserializer, &uvfn.m_resolved_container_membername);
        }
    }

    SERIAL::read( deserializer, &m_selector);

    SERIAL::read( deserializer, &m_impl_tag);
    SERIAL::read( deserializer, &m_impl_hash);

    SERIAL::read( deserializer, &m_cached_is_valid);
    SERIAL::read( deserializer, &m_cached_is_animated);
    SERIAL::read( deserializer, &m_cached_is_uvtile);
    SERIAL::read( deserializer, &m_cached_is_cubemap);

    deserializer->read_size_t( &s);
    m_cached_frames.resize( s);

    for( auto& frame: m_cached_frames) {

        SERIAL::read( deserializer, &frame.m_frame_number);

        deserializer->read_size_t( &s);
        frame.m_uvtiles.resize( s);

        for( auto& uvtile: frame.m_uvtiles) {
            SERIAL::read( deserializer, &uvtile.m_u);
            SERIAL::read( deserializer, &uvtile.m_v);
        }

        SERIAL::read( deserializer, &frame.m_uv_to_id.m_count_u);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_count_v);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_min_u);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_min_v);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_ids);
    }

    SERIAL::read( deserializer, &m_cached_frame_to_id);

    // Adjust filenames for this host
    if( convert_path) {

        m_original_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_original_filename);
        m_resolved_container_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_resolved_container_filename);

        for( auto& frame_filenames: m_frames_filenames)
            for( auto& uvfn: frame_filenames) {

                uvfn.m_resolved_filename
                    = HAL::Ospath::convert_to_platform_specific_path( uvfn.m_resolved_filename);
                uvfn.m_container_membername
                    = HAL::Ospath::convert_to_platform_specific_path( uvfn.m_container_membername);
                uvfn.m_resolved_container_membername = m_resolved_container_filename.empty()
                    ? "" : m_resolved_container_filename + ':' + uvfn.m_container_membername;
        }
    }

    // Re-resolve filenames

    // Re-resolving m_resolved_container_filename is not possible for container filename since we do
    // not have the original container filename.
    if( !m_resolved_container_filename.empty()
        && !DISK::is_file( m_resolved_container_filename.c_str()))
        m_resolved_container_filename.clear();

    for( auto& frame_filenames: m_frames_filenames)
        for( auto& uvfn: frame_filenames) {

            // Update m_resolved_container_membername based on m_resolved_container_filename above.
            uvfn.m_resolved_container_membername = m_resolved_container_filename.empty()
                ? "" : m_resolved_container_filename + ':' + uvfn.m_container_membername;

            // Re-resolve m_resolved_filename
            if( !uvfn.m_resolved_filename.empty()
                    && !DISK::is_file( uvfn.m_resolved_filename.c_str())) {
                // TODO Fix this for files with uvtile or frame markers
                uvfn.m_resolved_filename
                    = path_module->search( PATH::MDL, m_original_filename);
                if( uvfn.m_resolved_filename.empty())
                    uvfn.m_resolved_filename
                        = path_module->search( PATH::RESOURCE, m_original_filename);
            }
        }

    return this + 1;
}

void Image::dump() const
{
    std::ostringstream s;

    s << "Original filename: " << m_original_filename << std::endl;
    s << "MDL file path: " << m_mdl_file_path << std::endl;
    s << "Resolved container filename: " << m_resolved_container_filename << std::endl;

    s << "Implementation tag: " << m_impl_tag.get_uint() << std::endl;
    s << "Implementation hash: " << hash_to_string( m_impl_hash) << std::endl;

    s << "Is valid (cached): " << (m_cached_is_valid ? "true" : "false") << std::endl;
    s << "Is animated (cached): " << (m_cached_is_animated ? "true" : "false") << std::endl;
    s << "Is uvtile (cached): " << (m_cached_is_uvtile ? "true" : "false") << std::endl;
    s << "Is cubemap (cached): " << (m_cached_is_cubemap ? "true" : "false") << std::endl;

    s << "Number of frames (cached): " << m_cached_frames.size() << std::endl;

    for( mi::Size f = 0, n = m_cached_frames.size(); f < n; ++f) {

        const Frame& frame = m_cached_frames[f];
        const Frame_filenames& frame_filenames = m_frames_filenames[f];

        s << "Frame ID (cached): " << f << std::endl;
        s << "  Frame number (cached): " << frame.m_frame_number << std::endl;
        s << "  Number of UV tiles (cached): " << frame.m_uvtiles.size() << std::endl;

        for( mi::Size i = 0, m = frame.m_uvtiles.size(); i < m; ++i) {

            const Uvtile& uvtile = frame.m_uvtiles[i];
            const Uvfilenames& uvfn = frame_filenames[i];

            s << "  UV tile ID (cached): " << i << std::endl;
            s << "    u (cached): " << uvtile.m_u << std::endl;
            s << "    v (cached): " << uvtile.m_v << std::endl;
            s << "  Resolved filename: " << uvfn.m_resolved_filename << std::endl;
            s << "  Container membername: " << uvfn.m_container_membername << std::endl;
            s << "  Resolved container membername: " << uvfn.m_resolved_container_membername
              << std::endl;
        }

        const Uv_to_id& uv_to_id = frame.m_uv_to_id;

        s << "  UV to ID (cached):" << std::endl;
        s << "    Min u (cached): " << uv_to_id.m_min_u << std::endl;
        s << "    Min v (cached): " << uv_to_id.m_min_v << std::endl;
        s << "    Count u (cached): " << uv_to_id.m_count_u << std::endl;
        s << "    Count v (cached): " << uv_to_id.m_count_v << std::endl;

        mi::Sint32 v     = uv_to_id.m_min_v + uv_to_id.m_count_v - 1;
        mi::Sint32 v_end = uv_to_id.m_min_v;
        for( ; v >= v_end; --v) {
            s << "    v=" << v << ": ";
            mi::Sint32 u     = uv_to_id.m_min_u;
            mi::Sint32 u_end = uv_to_id.m_min_u + uv_to_id.m_count_u;
            for( ; u < u_end; ++u) {
                if( u >  uv_to_id.m_min_u)
                    s << ", ";
                s << static_cast<mi::Sint32>( uv_to_id.get( u, v));
            }
            s << std::endl;
        }
    }

    for( mi::Size f = 0, n = m_cached_frame_to_id.size(); f < n; ++f)
        if( m_cached_frame_to_id[f] != static_cast<mi::Size>( -1))
            s << "Frame (cached): " << f << " => ID: " << m_cached_frame_to_id[f] << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Image::get_size() const
{
    size_t s = dynamic_memory_consumption( m_original_filename)
             + dynamic_memory_consumption( m_mdl_file_path)
             + dynamic_memory_consumption( m_resolved_container_filename);

    for( const auto& frame: m_frames_filenames)
        for( const auto& uvtile: frame) {
            s += dynamic_memory_consumption( uvtile.m_resolved_filename);
            s += dynamic_memory_consumption( uvtile.m_container_membername);
            s += dynamic_memory_consumption( uvtile.m_resolved_container_membername);
        }

    s += dynamic_memory_consumption( m_selector);

    for( const auto& frame: m_cached_frames) {
        s += dynamic_memory_consumption( frame.m_uv_to_id.m_ids);
    }

    s += dynamic_memory_consumption( m_cached_frame_to_id);

    return s;
}

DB::Journal_type Image::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

void Image::get_scene_element_references( DB::Tag_set* result) const
{
    if( m_impl_tag)
        result->insert( m_impl_tag);
}

const mi::Uint32* Image::get_uvtile_mapping(
    mi::Size frame_id,
    mi::Uint32& num_u,
    mi::Uint32& num_v,
    mi::Sint32& offset_u,
    mi::Sint32& offset_v) const
{
    if( frame_id >= m_cached_frames.size()) {
        num_u = num_v = offset_u = offset_v = 0;
        return nullptr;
    }

    const Frame& frame = m_cached_frames[frame_id];

    num_u    =   frame.m_uv_to_id.m_count_u;
    num_v    =   frame.m_uv_to_id.m_count_v;
    offset_u = - frame.m_uv_to_id.m_min_u;
    offset_v = - frame.m_uv_to_id.m_min_v;
    return &frame.m_uv_to_id.m_ids[0];
}

/// Declaration repeated in unit test.
enum Uvtile_mode
{
    MODE_OFF,
    MODE_UDIM,
    MODE_UVTILE0,
    MODE_UVTILE1
};

/// Escapes meta-characters for basic regular expressions.
std::string escape_for_regex( const std::string& s)
{
    const char meta_characters[] = "[]\\^$.|?*+(){}";

    std::string result;
    result.reserve( s.size());

    for( const char ch : s) {
        if( std::strchr( meta_characters, ch) != nullptr)
            result.push_back( '\\');
        result.push_back( ch);
    }

    return result;
}

/// Returns the number of hashes in the pattern "<###...>" in s.
///
/// The pattern has to start at offset 0. Returns 0 if the pattern is not found.
size_t get_frame_marker_length( const char* s)
{
    ASSERT( M_SCENE, s[0] == '<');
    const char* p = s + 1;

    while( *p == '#')
        ++p;
    if( *p != '>')
        return 0;

    return p - s - 1;
}

/// Replaces "<UDIM>", "<UVTILE0>", "<UVTILE1>", "<###...>" by the corresponding regular
/// expressions.
///
/// Returns the empty string in case failure (multiple udim and/or frame patterns). Otherwise,
/// returns the replaced string, the indices for matched subexpressions in \p frame_index and \p
/// mode_index (either 1 or 2, or 0 if not present), the uvtile mode in \p mode, and the maximum
/// number of digits for frame numbers in \p frames_max_digits.
///
/// Declaration repeated in unit test.
std::string get_regex(
    const std::string& mask,
    size_t& mode_index,
    size_t& frames_index,
    Uvtile_mode& mode,
    size_t& frames_max_digits)
{
    size_t index = 1;
    std::string result;

    mode_index        = 0;
    frames_index      = 0;
    mode              = MODE_OFF;
    frames_max_digits = 0;

    size_t p = 0;

    while( true) {

        size_t q = mask.find( '<', p);
        result += escape_for_regex( mask.substr( p, q-p));
        if( q == std::string::npos)
            break;

        if( mask.substr( q, 6) == "<UDIM>") {

            if( mode != MODE_OFF)
                return std::string();
            mode        = MODE_UDIM;
            mode_index  = index++;
            result     += "([1-9][0-9][0-9][0-9])";
            q          += 6;

        } else if( mask.substr( q, 9) == "<UVTILE0>") {

            if( mode != MODE_OFF)
                return std::string();
            mode        = MODE_UVTILE0;
            mode_index  = index++;
            result     += "(_u-?[0-9]+_v-?[0-9]+)";
            q          += 9;

        } else if( mask.substr( q, 9) == "<UVTILE1>") {

            if( mode != MODE_OFF)
                return std::string();
            mode        = MODE_UVTILE1;
            mode_index  = index++;
            result     += "(_u-?[0-9]+_v-?[0-9]+)";
            q          += 9;

        } else if( (frames_max_digits = get_frame_marker_length( &mask[q]))) {

            if( frames_index > 0)
                return std::string();
            frames_index  = index++;
            result       += "([0-9]+)";
            q            += frames_max_digits + 2;

        } else {

            result += mask[q];
            ++q;

        }

        p = q;
    }

    return result;
}

/// Parses the (u,v) coordinates from the given uvtile/udim string.
///
/// Assumes that \p str matches the regular expression in get_regex() for \p mode.
///
/// \param mode   uvtile mode
/// \param s      string containing the indices, e.g. 1001 in udim mode
/// \param u      resulting u coordinate
/// \param v      resulting v coordinate
///
/// Declaration repeated in unit test.
void parse_u_v( Uvtile_mode mode, const char* s, mi::Sint32& u, mi::Sint32& v)
{
    u = 0;
    v = 0;

    switch( mode) {

        case MODE_OFF:
            ASSERT( M_SCENE, false);
            break;

        case MODE_UDIM: // UDIM (Mari), expands to four digits calculated as 1000 + (u+1 + 10*v)
            {
                const unsigned num
                    = 1000 * (s[0] - '0') +
                       100 * (s[1] - '0') +
                        10 * (s[2] - '0') +
                         1 * (s[3] - '0') - 1001;
                // assume u in [0..9]
                u = num % 10;
                v = num / 10;

                ASSERT( M_SCENE, s[4]   == '\0');
                break;
            }
        case MODE_UVTILE0: // 0-based (Zbrush), expands to "_u0_v0" for the first tile
        case MODE_UVTILE1: // 1-based (Mudbox), expands to "_u1_v1" for the first tile
            {
                const char* p = s;
                mi::Sint32 sign = 1;

                ASSERT( M_SCENE, p[0] == '_');
                ASSERT( M_SCENE, p[1] == 'u');
                p += 2; // skip "_u"

                if (*p == '-') {
                    sign = -1;
                    ++p;
                }

                while( isdigit (*p)) {
                    u = 10*u + (*p - '0');
                    ++p;
                }
                u *= sign;

                ASSERT( M_SCENE, p[0] == '_');
                ASSERT( M_SCENE, p[1] == 'v');
                p += 2; // skip "_v"
                sign = 1;

                if( *p == '-') {
                    sign = -1;
                    ++p;
                }

                while( isdigit( *p)) {
                    v = 10*v + (*p - '0');
                    ++p;
                }
                v *= sign;

                if( mode == MODE_UVTILE1) {
                    u -= 1;
                    v -= 1;
                }

                ASSERT( M_SCENE, *p == '\0');
                break;
            }
    }
}

Image_set* Image::resolve_filename( const std::string& filename, const char* selector)
{
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    // Get regular expression for filename.
    std::string basename = HAL::Ospath::basename( filename);
    size_t mode_index = 0;
    size_t frames_index = 0;
    Uvtile_mode mode = MODE_OFF;
    size_t frames_max_digits = 0;
    std::string filename_regex
        = get_regex( basename, mode_index, frames_index, mode, frames_max_digits);
    if( filename_regex.empty())
        return nullptr;

    // Handle plain files and containers
    if( (mode_index == 0) && (frames_index == 0)) {

        std::string resolved_filename = path_module->search( PATH::RESOURCE, filename);
        if( !resolved_filename.empty())
            return new File_image_set( filename, selector, resolved_filename);

        std::string lower = STRING::to_lower( filename);

        auto p = lower.find( ".mdr:");
        if( p != std::string::npos) {
            // Archives must be found via the MDL search path. Strip directories since archives
            // have to be at the top-level of the search paths. Member name is not checked.
            // No support for uvtiles nor animated textures.
            std::string archive_path = filename.substr( 0, p + 4);
            std::string archive_name = HAL::Ospath::basename( archive_path);
            std::string resolved_archive_filename
                = path_module->search( PATH::MDL, archive_name);
            std::string member_name = filename.substr( p + 5);
            if( resolved_archive_filename.empty())
                return nullptr;
            return new Container_file_image_set( resolved_archive_filename, member_name, selector);
        }

        p = lower.find( ".mdle:");
        if( p != std::string::npos) {
            std::string mdle_path = filename.substr( 0, p + 5);
            // Try to resolve relative MDLE files via the MDL search path. Member name is not
            // checked. No support for uvtiles nor animated textures.
            std::string resolved_mdle_path
                = path_module->search( PATH::MDL, mdle_path);
            if( resolved_mdle_path.empty())
                return nullptr;
            std::string member_name = filename.substr( p + 6);
            return new Container_file_image_set( resolved_mdle_path, member_name, selector);
        }

        return nullptr;
    }

    // Obtain search paths and directory relative to search paths
    std::string dirname = HAL::Ospath::dirname( filename);
    PATH::Path_module::Search_path search_paths;
    std::string relative_dirname;
    if( DISK::is_path_absolute( dirname)) {
        if( !DISK::access( dirname.c_str()))
            return nullptr;
        search_paths.push_back( dirname);
    } else {
        search_paths = path_module->get_search_path( PATH::RESOURCE);
        relative_dirname = dirname;
    }
    dirname.clear();

    std::wstring filename_wregex( STRING::utf8_to_wchar( filename_regex.c_str()));
    std::wregex regex( filename_wregex);

    std::vector<Uv_frame_filename> result;

    for( auto it = search_paths.begin(); it != search_paths.end(); ++it) {

        std::string current_dir = HAL::Ospath::join( *it, relative_dirname);

        DISK::Directory dir;
        if( !dir.open( current_dir.c_str()))
            continue;

        std::string fn = dir.read();
        while( !fn.empty()) {

            std::wstring filename( STRING::utf8_to_wchar( fn.c_str()));
            std::wsmatch matches;
            if( !std::regex_match( filename, matches, regex)) {
                fn = dir.read();
                continue;
            }

            mi::Sint32 u = 0, v = 0;
            if( mode_index > 0) {
                ASSERT( M_SCENE, matches.size() >= mode_index);
                std::wstring wmatch( matches[mode_index].first, matches[mode_index].second);
                std::string match = STRING::wchar_to_utf8( wmatch.c_str());
                parse_u_v( mode, match.c_str(), u, v);
            }

            mi::Size frame_number = 0;
            if( frames_index > 0) {
                ASSERT( M_SCENE, matches.size() >= frames_index);
                std::wstring wmatch( matches[frames_index].first, matches[frames_index].second);
                std::string match = STRING::wchar_to_utf8( wmatch.c_str());
                if( match.size() > frames_max_digits) {
                    fn = dir.read();
                    continue;
                }
                char* endptr;
                frame_number = std::strtoul( match.c_str(), &endptr, 10);
                ASSERT( M_SCENE, match.c_str() + match.size() == endptr);
            }

            std::string resolved_filename = HAL::Ospath::join( current_dir, fn);
            result.push_back( Uv_frame_filename{ frame_number, u, v, resolved_filename});

            fn = dir.read();
        }

        if( !result.empty())
            return new File_image_set(
                filename, selector, result, frames_index > 0, mode_index > 0);
    }

    return nullptr;
}

void Image::reset_shared(
    DB::Transaction* transaction,
    bool is_animated,
    bool is_uvtile,
    const Frames& frames,
    const Frame_to_id& frame_to_id,
    const mi::base::Uuid& impl_hash)
{
    // If impl_hash is valid, check whether implementation class exists already.
    std::string impl_name;
    if( impl_hash != mi::base::Uuid{0,0,0,0}) {
        impl_name = "MI_default_image_impl_" + hash_to_string( impl_hash);
        m_impl_tag = transaction->name_to_tag( impl_name.c_str());
        if( m_impl_tag) {
            m_impl_hash = impl_hash;
            DB::Access<Image_impl> impl( m_impl_tag, transaction);
            setup_cached_values( impl.get_ptr());
            return;
        }
    }

    Image_impl* impl = new Image_impl( is_animated, is_uvtile, frames, frame_to_id);

    setup_cached_values( impl);

    // We do not know the scope in which the instance of the proxy class ends up. Therefore, we have
    // to pick the global scope for the instance of the implementation class. Make sure to use
    // a DB name for the implementation class exactly for valid hashes.
    ASSERT( M_SCENE, impl_name.empty() ^ (impl_hash != mi::base::Uuid{0,0,0,0}));
    m_impl_tag = transaction->store_for_reference_counting(
        impl, !impl_name.empty() ? impl_name.c_str() : nullptr, /*privacy_level*/ 0);
    m_impl_hash = impl_hash;
}

void Image::setup_cached_values( const Image_impl* impl)
{
    m_cached_is_valid    = impl->is_valid();
    m_cached_is_animated = impl->is_animated();
    m_cached_is_uvtile   = impl->is_uvtile();
    m_cached_is_cubemap  = impl->get_is_cubemap();
    m_cached_frames      = impl->get_frames_vector();
    m_cached_frame_to_id = impl->get_frame_to_id();

    for( auto& frame: m_cached_frames)
        for( auto& uvtile: frame.m_uvtiles)
             uvtile.m_mipmap.reset();
}

Image_impl::Image_impl()
  : m_is_valid( false),
    m_is_animated( false),
    m_is_uvtile( false),
    m_is_cubemap( false)
{
}

Image_impl::Image_impl( const Image_impl& other)
  : SCENE::Scene_element<Image_impl, ID_IMAGE_IMPL>( other),
    m_is_valid( false),
    m_is_animated( false),
    m_is_uvtile( false),
    m_is_cubemap( false)
{
}

Image_impl::Image_impl(
    bool is_animated, bool is_uvtile, const Frames& frames, const Frame_to_id& frame_to_id)
  : m_is_valid( true),
    m_is_animated( is_animated),
    m_is_uvtile( is_uvtile),
    m_frames( frames),
    m_frame_to_id( frame_to_id)
{
    const mi::Size n = frames.size();

    // check that there is at least one frame
    ASSERT( M_SCENE, n > 0);

    // check that non-animated textures have exactly one frame with frame number 0
    if( !is_animated) {
        ASSERT( M_SCENE, n == 1);
        ASSERT( M_SCENE, frames[0].m_frame_number == 0);
    }

    // check that non-uvtile textures have exactly one uvtile per frame with coordinates (0,0)
    if( !is_uvtile) {
        for( mi::Size i = 0; i < n; ++i) {
            ASSERT( M_SCENE, frames[i].m_uvtiles.size() == 1);
            ASSERT( M_SCENE, frames[i].m_uvtiles[0].m_u == 0);
            ASSERT( M_SCENE, frames[i].m_uvtiles[0].m_v == 0);
        }
    }

    // check that every frame in \p frames has a corresponding mapping in \p frame_to_id
    for( mi::Size i = 0; i < n; ++i) {
        ASSERT( M_SCENE, frames[i].m_frame_number < frame_to_id.size());
        ASSERT( M_SCENE, frame_to_id[frames[i].m_frame_number] == i);
    }

    // check that every mapping in \p frame_to_id has a corresponding frame in \p frames
    for( mi::Size i = 0; i < frame_to_id.size(); ++i) {
        if( frame_to_id[i] == static_cast<mi::Size>( -1))
           continue;
        ASSERT( M_SCENE, frame_to_id[i] < n);
        ASSERT( M_SCENE, frames[frame_to_id[i]].m_frame_number == i);
    }

    // check that mapping frame index to frame number is strictly monotonically creasing
    for( mi::Size i = 1; i < n; ++i)
        ASSERT( M_SCENE, frames[i-1].m_frame_number < frames[i].m_frame_number);

    // check that the frames are not empty and that every uvtile in \p frames.m_uvtiles has a
    // corresponding mapping in \p frames.m_uv_to_id
    for( mi::Size i = 0; i < n; ++i) {

        const Frame& frame = frames[i];

        // check that frames are not empty
        mi::Size m = frame.m_uvtiles.size();
        ASSERT( M_SCENE, m > 0);

        // check that every uvtile in \p frame.m_uvtiles has a corresponding mapping in
        // \p frame.m_uv_to_id
        for( mi::Size j = 0; j < m; ++j) {
            mi::Sint32 u  = frame.m_uvtiles[j].m_u;
            mi::Sint32 v  = frame.m_uvtiles[j].m_v;
            mi::Uint32 id = frame.m_uv_to_id.get( u, v);
            ASSERT( M_SCENE, j == static_cast<mi::Size>( id));
            boost::ignore_unused( id);
        }

        // check that every uvtile in \p frame.m_uv_to_id has a corresponding uvtile in
        // \p \p frame.m_uvtiles
        mi::Sint32 min_u = frame.m_uv_to_id.m_min_u;
        mi::Sint32 min_v = frame.m_uv_to_id.m_min_v;
        mi::Sint32 max_u = frame.m_uv_to_id.m_min_u;
        mi::Sint32 max_v = frame.m_uv_to_id.m_min_v;
        for( mi::Sint32 u = min_u; u < max_u; ++u)
            for( mi::Sint32 v = min_v; v < max_v; ++v) {
                mi::Uint32 id = frame.m_uv_to_id.get( u, v);
                if( id == ~0u)
                    continue;
                ASSERT( M_SCENE, id < m);
                ASSERT( M_SCENE, frame.m_uvtiles[id].m_u == u);
                ASSERT( M_SCENE, frame.m_uvtiles[id].m_v == v);
            }

        // check that mipmap pointers are valid
        for( mi::Size j = 0; j < m; ++j)
            ASSERT( M_SCENE, frame.m_uvtiles[j].m_mipmap);
    }

    m_is_cubemap = m_frames[0].m_uvtiles[0].m_mipmap->get_is_cubemap();
}

Image_impl::~Image_impl()
{
}

const IMAGE::IMipmap* Image_impl::get_mipmap( mi::Size frame_id, mi::Size uvtile_id) const
{
    if( frame_id >= m_frames.size())
        return nullptr;

    const Frame& frame = m_frames[frame_id];
    if( uvtile_id >= frame.m_uvtiles.size())
        return nullptr;

    frame.m_uvtiles[uvtile_id].m_mipmap->retain();
    return frame.m_uvtiles[uvtile_id].m_mipmap.get();
}

mi::Size Image_impl::get_frame_number( mi::Size frame_id) const
{
    if( frame_id >= m_frames.size())
        return 0;

    return m_frames[frame_id].m_frame_number;
}

mi::Size Image_impl::get_frame_id( mi::Size frame_number) const
{
    if( frame_number >= m_frame_to_id.size())
        return -1;

    return m_frame_to_id[frame_number];
}

mi::Size Image_impl::get_frame_length( mi::Size frame_id) const
{
    if( frame_id >= m_frames.size())
        return 0;

    return m_frames[frame_id].m_uvtiles.size();
}

void Image_impl::get_uvtile_uv_ranges(
    mi::Size frame_id,
    mi::Sint32& min_u,
    mi::Sint32& min_v,
    mi::Sint32& max_u,
    mi::Sint32& max_v) const
{
    if( frame_id >= m_frames.size()) {
        min_u = min_v = max_u = max_v = 0;
        return;
    }

    const Frame& frame = m_frames[frame_id];

    min_u = frame.m_uv_to_id.m_min_u;
    min_v = frame.m_uv_to_id.m_min_v;
    max_u = frame.m_uv_to_id.m_min_u + frame.m_uv_to_id.m_count_u - 1;
    max_v = frame.m_uv_to_id.m_min_v + frame.m_uv_to_id.m_count_v - 1;
}

mi::Sint32 Image_impl::get_uvtile_uv(
    mi::Size frame_id, mi::Size uvtile_id, mi::Sint32& u, mi::Sint32& v) const
{
    if( frame_id >= m_frames.size())
        return -1;

    const Frame& frame = m_frames[frame_id];
    if( uvtile_id >= frame.m_uvtiles.size())
        return -1;

    u = frame.m_uvtiles[uvtile_id].m_u;
    v = frame.m_uvtiles[uvtile_id].m_v;
    return 0;
}

mi::Size Image_impl::get_uvtile_id( mi::Size frame_id, mi::Sint32 u, mi::Sint32 v) const
{
    if( frame_id >= m_frames.size())
        return -1;

    return m_frames[frame_id].m_uv_to_id.get( u, v);
}

const SERIAL::Serializable* Image_impl::serialize( SERIAL::Serializer* serializer) const
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    Scene_element_base::serialize( serializer);

    SERIAL::write( serializer, m_is_valid);
    SERIAL::write( serializer, m_is_animated);
    SERIAL::write( serializer, m_is_uvtile);
    SERIAL::write( serializer, m_is_cubemap);

    serializer->write_size_t( m_frames.size());
    for( const auto& frame: m_frames) {

        SERIAL::write( serializer, frame.m_frame_number);

        serializer->write_size_t( frame.m_uvtiles.size());
        for( const auto& uvtile: frame.m_uvtiles) {
            SERIAL::write( serializer, uvtile.m_u);
            SERIAL::write( serializer, uvtile.m_v);
            image_module->serialize_mipmap( serializer, uvtile.m_mipmap.get());
        }

        SERIAL::write( serializer, frame.m_uv_to_id.m_count_u);
        SERIAL::write( serializer, frame.m_uv_to_id.m_count_v);
        SERIAL::write( serializer, frame.m_uv_to_id.m_min_u);
        SERIAL::write( serializer, frame.m_uv_to_id.m_min_v);
        SERIAL::write( serializer, frame.m_uv_to_id.m_ids);
    }

    SERIAL::write( serializer, m_frame_to_id);

    return this + 1;
}

SERIAL::Serializable* Image_impl::deserialize( SERIAL::Deserializer* deserializer)
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    size_t s;

    Scene_element_base::deserialize( deserializer);

    SERIAL::read( deserializer, &m_is_valid);
    SERIAL::read( deserializer, &m_is_animated);
    SERIAL::read( deserializer, &m_is_uvtile);
    SERIAL::read( deserializer, &m_is_cubemap);

    deserializer->read_size_t( &s);
    m_frames.resize( s);
    for( auto& frame: m_frames) {

        SERIAL::read( deserializer, &frame.m_frame_number);

        deserializer->read_size_t( &s);
        frame.m_uvtiles.resize( s);
        for( auto& uvtile: frame.m_uvtiles) {

            SERIAL::read( deserializer, &uvtile.m_u);
            SERIAL::read( deserializer, &uvtile.m_v);
            uvtile.m_mipmap = image_module->deserialize_mipmap( deserializer);
        }

        SERIAL::read( deserializer, &frame.m_uv_to_id.m_count_u);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_count_v);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_min_u);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_min_v);
        SERIAL::read( deserializer, &frame.m_uv_to_id.m_ids);
    }

    SERIAL::read( deserializer, &m_frame_to_id);

    return this + 1;
}

void Image_impl::dump() const
{
    std::ostringstream s;

    s << "Is valid: " << (m_is_valid ? "true" : "false") << std::endl;
    s << "Is animate: " << (m_is_animated ? "true" : "false") << std::endl;
    s << "Is uvtile: " << (m_is_uvtile ? "true" : "false") << std::endl;

    s << "Number of frames: " << m_frames.size() << std::endl;

    for( mi::Size f = 0, n = m_frames.size(); f < n; ++f) {

        const Frame& frame = m_frames[f];
        s << "Frame ID: " << f << std::endl;
        s << "  Frame number: " << frame.m_frame_number << std::endl;
        s << "  Number of UV tiles: " << frame.m_uvtiles.size() << std::endl;

        for( mi::Size i = 0, m = frame.m_uvtiles.size(); i < m; ++i) {

            const Uvtile& uvtile = frame.m_uvtiles[i];
            s << "  UV tile ID: " << i << std::endl;
            s << "    u: " << uvtile.m_u << std::endl;
            s << "    v: " << uvtile.m_v << std::endl;
            s << "    Miplevel: " << uvtile.m_mipmap->get_nlevels() << std::endl;
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas( uvtile.m_mipmap->get_level( 0));
            s << "    Pixel type: " << canvas->get_type() << std::endl;
            s << "    Pixels: " << canvas->get_resolution_x()
              << 'x' << canvas->get_resolution_y()
              << 'x' << canvas->get_layers_size() << std::endl;
            s << "    Gamma: " << canvas->get_gamma() << std::endl;
        }

        const Uv_to_id& uv_to_id = frame.m_uv_to_id;

        s << "  UV to ID:" << std::endl;
        s << "    Min u: " << uv_to_id.m_min_u << std::endl;
        s << "    Min v: " << uv_to_id.m_min_v << std::endl;
        s << "    Count u: " << uv_to_id.m_count_u << std::endl;
        s << "    Count v: " << uv_to_id.m_count_v << std::endl;

        mi::Sint32 v     = uv_to_id.m_min_v + uv_to_id.m_count_v - 1;
        mi::Sint32 v_end = uv_to_id.m_min_v;
        for( ; v >= v_end; --v) {
            s << "    v=" << v << ": ";
            mi::Sint32 u     = uv_to_id.m_min_u;
            mi::Sint32 u_end = uv_to_id.m_min_u + uv_to_id.m_count_u;
            for( ; u < u_end; ++u) {
                if( u >  uv_to_id.m_min_u)
                    s << ", ";
                s << static_cast<mi::Sint32>( uv_to_id.get( u, v));
            }
            s << std::endl;
        }
    }

    for( mi::Size f = 0, n = m_frame_to_id.size(); f < n; ++f)
        if( m_frame_to_id[f] != static_cast<mi::Size>( -1))
            s << "Frame: " << f << " => ID: " << m_frame_to_id[f] << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Image_impl::get_size() const
{
    size_t s = sizeof( *this);

    for( const auto& frame: m_frames) {
        for( const auto& uvtile: frame.m_uvtiles)
            s += uvtile.m_mipmap->get_size();
        s += dynamic_memory_consumption( frame.m_uv_to_id.m_ids);
    }

    s += dynamic_memory_consumption( m_frame_to_id);

    return s;
}

DB::Journal_type Image_impl::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

const mi::Uint32* Image_impl::get_uvtile_mapping(
    mi::Size frame_id,
    mi::Uint32& num_u,
    mi::Uint32& num_v,
    mi::Sint32& offset_u,
    mi::Sint32& offset_v) const
{
    if( frame_id >= m_frames.size()) {
        num_u = num_v = offset_u = offset_v = 0;
        return nullptr;
    }

    const Frame& frame = m_frames[frame_id];

    num_u    =   frame.m_uv_to_id.m_count_u;
    num_v    =   frame.m_uv_to_id.m_count_v;
    offset_u = - frame.m_uv_to_id.m_min_u;
    offset_v = - frame.m_uv_to_id.m_min_v;
    return &frame.m_uv_to_id.m_ids[0];
}

} // namespace DBIMAGE

} // namespace MI
