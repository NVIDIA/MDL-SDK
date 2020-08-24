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
#include <mdl/integration/mdlnr/i_mdlnr.h>

namespace MI {

namespace DBIMAGE {

namespace {

std::string empty_str;

// Returns a string representation of mi::base::Uuid
std::string hash_to_string( const mi::base::Uuid& hash)
{
    char buffer[35];
    snprintf( &buffer[0], sizeof( buffer), "0x%08x%08x%08x%08x",
              hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
    return buffer;
}

}

IMAGE::IMipmap* Image_set::create_mipmap( mi::Size i) const
{
    ASSERT( M_SCENE, i < get_length());

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    // mdl container based
    if( is_mdl_container())
    {
        mi::base::Handle<mi::neuraylib::IReader> reader( open_reader( i));
        if( !reader)
            return image_module->create_dummy_mipmap();

        const char* container_filename = get_container_filename();
        const char* container_membername = get_container_membername( i);
        ASSERT( M_SCENE, get_container_filename());
        ASSERT( M_SCENE, get_container_membername(i));
        return image_module->create_mipmap(
            IMAGE::Container_based(),
            reader.get(),
            container_filename,
            container_membername,
            /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true);
    }

    // file based
    const char* resolved_filename = get_resolved_filename( i);
    if( resolved_filename && strlen( resolved_filename) > 0)
    {
        return image_module->create_mipmap(
            resolved_filename,
            /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true);
    }

    // canvas based
    mi::base::Handle<mi::neuraylib::ICanvas> canvas( get_canvas(i));
    if( canvas) {
        std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
        canvases[0] = canvas;
        return image_module->create_mipmap( canvases);
    }

    // reader based
    mi::base::Handle<mi::neuraylib::IReader> reader( open_reader( i));
    if( reader) {
        const char* image_format = get_image_format();
        ASSERT( M_SCENE, image_format);
        return image_module->create_mipmap(
            IMAGE::Memory_based(),
            reader.get(),
            image_format,
            get_mdl_file_path(),
            /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true);
    }

    return image_module->create_dummy_mipmap();
}

class Plain_file_image_set : public DBIMAGE::Image_set
{
public:

    Plain_file_image_set(
        const std::string& original_filename, const std::string& resolved_filename)
      : m_original_filename( original_filename),
        m_resolved_filename( resolved_filename)
    {
    }

    mi::Size get_length() const { return 1; }

    bool is_uvtile() const { return false; }

    bool is_mdl_container() const { return false; }

    void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT( M_SCENE, i <= get_length());
        u = 0;
        v = 0;
    }

    const char* get_original_filename() const { return m_original_filename.c_str(); }

    const char* get_container_filename() const { return ""; }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_resolved_filename( mi::Size i) const
    {
        ASSERT( M_SCENE, i <= get_length());
        return m_resolved_filename.c_str();
    }

    const char* get_container_membername( mi::Size i) const { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size i) const { return nullptr; }

    mi::neuraylib::ICanvas* get_canvas( mi::Size i) const { return nullptr; }

    const char* get_image_format() const { return ""; }

private:
    std::string m_original_filename;
    std::string m_resolved_filename;
};

class Container_file_image_set : public Image_set
{
public:
    Container_file_image_set(
        const std::string& resolved_container_filename,
        const std::string& container_member_name)
      : m_resolved_container_filename( resolved_container_filename),
        m_container_member_name( container_member_name)
    {
    }

    mi::Size get_length() const { return 1; }

    bool is_uvtile() const { return false; }

    bool is_mdl_container() const { return true; }

    void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT( M_SCENE, i <= get_length());
        u = 0;
        v = 0;
    }

    const char* get_original_filename() const { return ""; }

    const char* get_container_filename() const { return m_resolved_container_filename.c_str(); }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_resolved_filename( mi::Size i) const { return ""; }

    const char* get_container_membername( mi::Size i) const
    {
        ASSERT( M_SCENE, i <= get_length());
        return m_container_member_name.c_str();
    }

    mi::neuraylib::IReader* open_reader( mi::Size i) const
    {
        ASSERT( M_SCENE, i <= get_length());
        return MDL::get_container_resource_reader(
            m_resolved_container_filename, m_container_member_name);
    }

    mi::neuraylib::ICanvas* get_canvas( mi::Size i) const { return nullptr; }

    const char* get_image_format() const { return ""; }

private:
    std::string m_resolved_container_filename;
    std::string m_container_member_name;
};

class Udim_image_set : public Image_set
{
public:
    Udim_image_set( const std::string& original_filename)
      : m_original_filename( original_filename)
    {
    }

    mi::Size get_length() const { return  m_resolved_filenames.size(); }

    bool is_uvtile() const { return true; }

    bool is_mdl_container() const { return false; }

    void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT( M_SCENE, i <= get_length());
        u = m_indices[i].first;
        v = m_indices[i].second;
    }

    const char* get_container_filename() const { return ""; }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_original_filename() const { return m_original_filename.c_str(); }

    const char* get_resolved_filename( mi::Size i) const
    {
        ASSERT( M_SCENE, i <= get_length());
        return m_resolved_filenames[i].c_str();
    }

    const char* get_container_membername( mi::Size i) const { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size i) const { return nullptr; }

    mi::neuraylib::ICanvas* get_canvas( mi::Size i) const { return nullptr; }

    const char* get_image_format() const { return ""; }

    std::vector<std::string>& get_resolved_filenames() { return m_resolved_filenames; }

    std::vector<std::pair<mi::Sint32, mi::Sint32> >& get_uvtile_uvs() { return m_indices; }

private:
    const std::string m_original_filename;
    std::vector<std::string> m_resolved_filenames;
    std::vector<std::pair<mi::Sint32, mi::Sint32> > m_indices;
};

Uv_to_id::Uv_to_id( mi::Sint32 min_u, mi::Sint32 max_u, mi::Sint32 min_v, mi::Sint32 max_v)
{
    m_count_u = max_u - min_u + 1;
    m_count_v = max_v - min_v + 1;
    m_min_u   = min_u;
    m_min_v   = min_v;
    m_ids.resize( m_count_u * m_count_v, m_count_u * m_count_v == 1 ? 0 : ~0u);
}

mi::Uint32 Uv_to_id::get( mi::Sint32 u, mi::Sint32 v) const
{
    mi::Sint32 uu = u - m_min_u;
    mi::Sint32 vv = v - m_min_v;

    if( uu < 0 || uu >= static_cast<Sint32>( m_count_u))
        return ~0u;
    if( vv < 0 || vv >= static_cast<Sint32>( m_count_v))
        return ~0u;

    return m_ids[vv * m_count_u + uu];
}

bool Uv_to_id::set( mi::Sint32 u, mi::Sint32 v, mi::Uint32 id)
{
    mi::Sint32 uu = u - m_min_u;
    mi::Sint32 vv = v - m_min_v;

    if( uu < 0 || uu >= static_cast<Sint32>( m_count_u))
        return false;
    if( vv < 0 || vv >= static_cast<Sint32>( m_count_v))
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
  : m_impl_tag( DB::Tag()),
    m_impl_hash{0,0,0,0},
    m_cached_is_valid( false),
    m_cached_is_uvtile( false),
    m_cached_is_cubemap( false)
{
}

Image::Image( const Image& other)
  : SCENE::Scene_element<Image, ID_IMAGE>( other),
    m_impl_tag( DB::Tag()),
    m_impl_hash{0,0,0,0},
    m_cached_is_valid( false),
    m_cached_is_uvtile( false),
    m_cached_is_cubemap( false)
{
}

Image::~Image()
{
}

mi::Sint32 Image::reset_file(
    DB::Transaction* transaction,
    const std::string& original_filename,
    const mi::base::Uuid& impl_hash)
{
    mi::base::Handle<Image_set> image_set( resolve_filename( original_filename));
    if( !image_set)
        return -2;

    return reset_image_set( transaction, image_set.get(), impl_hash);
}

mi::Sint32 Image::reset_reader(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const char* image_format,
    const mi::base::Uuid& impl_hash)
{
    mi::Sint32 result = 0;
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap(
        IMAGE::Memory_based(),
        reader,
        image_format,
        /*mdl_file_path*/ nullptr,
        /*tile_width*/ 0,
        /*tile_height*/ 0,
        /*only_first_level*/ true,
        &result));
    if( result < 0)
        return result;

    // Convert data from mipmap into temorary variables
    std::vector<Uvtile> tmp_uvtiles( 1);
    tmp_uvtiles[0].m_mipmap = std::move( mipmap);
    std::vector<Uvfilenames> tmp_uvfilenames( 1);
    Uv_to_id tmp_uv_to_id( 0, 0, 0, 0);
    bool tmp_is_uvtile = false;

    reset_shared( transaction, tmp_is_uvtile, tmp_uvtiles, tmp_uv_to_id, impl_hash);

    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();
    m_uvfilenames = tmp_uvfilenames;

    ASSERT( M_SCENE, m_uvfilenames.size() == m_cached_uvtiles.size());

    return 0;
}

mi::Sint32 Image::reset_image_set(
    DB::Transaction* transaction, const Image_set* image_set, const mi::base::Uuid& impl_hash)
{
    if( !image_set)
        return -1;

    mi::Size number_of_tiles = image_set->get_length();
    if( number_of_tiles == 0)
        return -1;

    // Compute min/max u/v value of all tiles.
    mi::Sint32 u = 0;
    mi::Sint32 v = 0;
    image_set->get_uv_mapping( 0, u, v);
    mi::Sint32 min_u = u;
    mi::Sint32 max_u = u;
    mi::Sint32 min_v = v;
    mi::Sint32 max_v = v;
    for( mi::Size i = 1; i < number_of_tiles; ++i) {
        image_set->get_uv_mapping( i, u, v);
        min_u = mi::math::min( min_u, u);
        max_u = mi::math::max( max_u, u);
        min_v = mi::math::min( min_v, v);
        max_v = mi::math::max( max_v, v);
    }

    // Convert data from image_set into temorary variables
    std::vector<Uvtile> tmp_uvtiles( number_of_tiles);
    std::vector<Uvfilenames> tmp_uvfilenames( number_of_tiles);
    Uv_to_id tmp_uv_to_id( min_u, max_u, min_v, max_v);
    bool tmp_is_uvtile = image_set->is_uvtile();

    Sint32 result = 0;

    std::string tmp_resolved_container_filename = image_set->get_container_filename();
    std::string tmp_original_filename           = image_set->get_original_filename();
    std::string tmp_mdl_file_path               = image_set->get_mdl_file_path();

    for( mi::Uint32 i = 0; i < number_of_tiles; ++i) {

        mi::Sint32 u = 0;
        mi::Sint32 v = 0;
        image_set->get_uv_mapping( i, u, v);

        if( !tmp_uv_to_id.set( u, v, i)) {
            result = -2;
            break;
        }

        Uvtile& tile = tmp_uvtiles[i];
        tile.m_u = u;
        tile.m_v = v;
        tile.m_mipmap = image_set->create_mipmap( i);
        if( !tile.m_mipmap) {
            result = -3;
            break;
        }

        Uvfilenames& filename = tmp_uvfilenames[i];
        filename.m_resolved_filename    = image_set->get_resolved_filename( i);
        filename.m_container_membername = image_set->get_container_membername( i);
        if( !filename.m_container_membername.empty())
            filename.m_resolved_container_membername
                = tmp_resolved_container_filename + ":" + filename.m_container_membername;
    }

    if( result != 0)
        return result;

    reset_shared( transaction, tmp_is_uvtile, tmp_uvtiles, tmp_uv_to_id, impl_hash);

    m_uvfilenames                 = tmp_uvfilenames;
    m_resolved_container_filename = tmp_resolved_container_filename;
    m_original_filename           = tmp_original_filename;
    m_mdl_file_path               = tmp_mdl_file_path;

    ASSERT( M_SCENE, m_uvfilenames.size() == m_cached_uvtiles.size());

    return 0;
}

void Image::set_mipmap(
    DB::Transaction* transaction, IMAGE::IMipmap* mipmap, const mi::base::Uuid& impl_hash)
{
    mi::base::Handle<IMAGE::IMipmap> tmp_mipmap;
    if( !mipmap) {
        SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
        tmp_mipmap = image_module->create_dummy_mipmap();
    } else
        tmp_mipmap = make_handle_dup( mipmap);

    // Convert data from image_set into temorary variables
    std::vector<Uvtile> tmp_uvtiles( 1);
    tmp_uvtiles[0].m_mipmap = std::move( tmp_mipmap);
    std::vector<Uvfilenames> tmp_uvfilenames( 1);
    Uv_to_id tmp_uv_to_id( 0, 0, 0, 0);
    bool tmp_is_uvtile = false;

    reset_shared( transaction, tmp_is_uvtile, tmp_uvtiles, tmp_uv_to_id, impl_hash);

    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();
    m_uvfilenames = tmp_uvfilenames;

    ASSERT( M_SCENE, m_uvfilenames.size() == m_cached_uvtiles.size());
}

const IMAGE::IMipmap* Image::get_mipmap( DB::Transaction* transaction, mi::Uint32 uvtile_id) const
{
    if( !m_cached_is_valid) {
        SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
        return image_module->create_dummy_mipmap();
    }

    if( uvtile_id >= m_cached_uvtiles.size())
        return nullptr;

    DB::Access<Image_impl> impl( m_impl_tag, transaction);
    return impl->get_mipmap( uvtile_id);
}

const std::string& Image::get_filename( mi::Uint32 uvtile_id) const
{
    if( uvtile_id >= m_uvfilenames.size())
        return empty_str;

    return m_uvfilenames[uvtile_id].m_resolved_filename;
}

const std::string& Image::get_container_membername( mi::Uint32 uvtile_id) const
{
    if( uvtile_id >= m_uvfilenames.size())
        return empty_str;

    return m_uvfilenames[uvtile_id].m_container_membername;
}

const std::string& Image::get_resolved_container_membername( mi::Uint32 uvtile_id) const
{
    if( uvtile_id >= m_uvfilenames.size())
        return empty_str;

    return m_uvfilenames[uvtile_id].m_resolved_container_membername;
}

const std::string& Image::get_original_filename() const
{
    return m_original_filename;
}

const std::string& Image::get_mdl_file_path( ) const
{
    return m_mdl_file_path;
}

void Image::get_uvtile_uv_ranges(
    mi::Sint32& min_u, mi::Sint32& min_v, mi::Sint32& max_u, mi::Sint32& max_v) const
{
    if( m_cached_uv_to_id.m_ids.empty()) {
        min_u = min_v = max_u = max_v = 0;
        return;
    }

    min_u = m_cached_uv_to_id.m_min_u;
    min_v = m_cached_uv_to_id.m_min_v;
    max_u = m_cached_uv_to_id.m_min_u + m_cached_uv_to_id.m_count_u - 1;
    max_v = m_cached_uv_to_id.m_min_v + m_cached_uv_to_id.m_count_v - 1;
}

mi::Size Image::get_uvtile_length() const
{
    return m_cached_uvtiles.size();
}

mi::Sint32 Image::get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const
{
    if( uvtile_id >= m_cached_uvtiles.size())
        return -1;

    const Uvtile& uv_tile = m_cached_uvtiles[uvtile_id];
    u = uv_tile.m_u;
    v = uv_tile.m_v;

    return 0;
}

mi::Uint32 Image::get_uvtile_id( Sint32 u, Sint32 v) const
{
    return m_cached_uv_to_id.get( u, v);
}

const SERIAL::Serializable* Image::serialize( SERIAL::Serializer* serializer) const
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    bool remote = serializer->is_remote();

    Scene_element_base::serialize( serializer);

    serializer->write( HAL::Ospath::sep());

    serializer->write( remote ? "" : m_original_filename);
    serializer->write( remote ? "" : m_mdl_file_path);
    serializer->write( remote ? "" : m_resolved_container_filename);

    serializer->write_size_t( m_uvfilenames.size());
    for( auto it = m_uvfilenames.begin(); it != m_uvfilenames.end(); ++it) {
         serializer->write( remote ? "" : it->m_resolved_filename);
         serializer->write( remote ? "" : it->m_container_membername);
         serializer->write( remote ? "" : it->m_resolved_container_membername);
    }

    serializer->write( m_impl_tag);
    serializer->write( m_impl_hash);

    serializer->write( m_cached_is_valid);
    serializer->write( m_cached_is_uvtile);
    serializer->write( m_cached_is_cubemap);

    serializer->write_size_t( m_cached_uvtiles.size());
    for( auto it = m_cached_uvtiles.begin(); it != m_cached_uvtiles.end(); ++it) {
        serializer->write( it->m_u);
        serializer->write( it->m_v);
        ASSERT( M_SCENE, !it->m_mipmap);
    }

    SERIAL::write( serializer, m_cached_uv_to_id.m_ids);
    serializer->write( m_cached_uv_to_id.m_count_u);
    serializer->write( m_cached_uv_to_id.m_count_v);
    serializer->write( m_cached_uv_to_id.m_min_u);
    serializer->write( m_cached_uv_to_id.m_min_v);

    return this + 1;
}

SERIAL::Serializable* Image::deserialize( SERIAL::Deserializer* deserializer)
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    size_t s;

    Scene_element_base::deserialize( deserializer);

    std::string serializer_sep;
    deserializer->read( &serializer_sep);
    bool convert_path =  serializer_sep != HAL::Ospath::sep();

    deserializer->read( &m_original_filename);
    deserializer->read( &m_mdl_file_path);
    deserializer->read( &m_resolved_container_filename);

    deserializer->read_size_t( &s);
    m_uvfilenames.resize( s);
    for( auto it = m_uvfilenames.begin(); it != m_uvfilenames.end(); ++it) {
        deserializer->read( &it->m_resolved_filename);
        deserializer->read( &it->m_container_membername);
        deserializer->read( &it->m_resolved_container_membername);
    }

    deserializer->read( &m_impl_tag);
    deserializer->read( &m_impl_hash);

    deserializer->read( &m_cached_is_valid);
    deserializer->read( &m_cached_is_uvtile);
    deserializer->read( &m_cached_is_cubemap);

    deserializer->read_size_t( &s);
    m_cached_uvtiles.resize( s);

    for( auto it = m_cached_uvtiles.begin(); it != m_cached_uvtiles.end(); ++it) {
        deserializer->read( &it->m_u);
        deserializer->read( &it->m_v);
        ASSERT( M_SCENE, !it->m_mipmap);
    }

    SERIAL::read( deserializer, &m_cached_uv_to_id.m_ids);
    deserializer->read( &m_cached_uv_to_id.m_count_u);
    deserializer->read( &m_cached_uv_to_id.m_count_v);
    deserializer->read( &m_cached_uv_to_id.m_min_u);
    deserializer->read( &m_cached_uv_to_id.m_min_v);

    // Adjust filenames for this host
    if( convert_path) {

        m_original_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_original_filename);
        m_resolved_container_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_resolved_container_filename);

        for( auto it = m_uvfilenames.begin(); it != m_uvfilenames.end(); ++it) {
           it->m_resolved_filename
               = HAL::Ospath::convert_to_platform_specific_path( it->m_resolved_filename);
           it->m_container_membername
               = HAL::Ospath::convert_to_platform_specific_path( it->m_container_membername);
           it->m_resolved_container_membername = m_resolved_container_filename.empty()
               ? "" : m_resolved_container_filename + ":" + it->m_container_membername;
        }
    }

    // Re-resolve filenames

    // Re-resolving m_resolved_container_filename is not possible for container filename since we do
    // not have the original container filename.
    if( !m_resolved_container_filename.empty()
        && !DISK::is_file( m_resolved_container_filename.c_str()))
        m_resolved_container_filename.clear();

    for( auto it = m_uvfilenames.begin(); it != m_uvfilenames.end(); ++it) {

        // Update m_resolved_container_membername based on m_resolved_container_filename above.
        it->m_resolved_container_membername = m_resolved_container_filename.empty()
            ? "" : m_resolved_container_filename + ":" + it->m_container_membername;

        // Re-resolve m_resolved_filename
        if( !it->m_resolved_filename.empty() && !DISK::is_file( it->m_resolved_filename.c_str())) {
            // TODO Fix this for files with udim markers
            it->m_resolved_filename = path_module->search( PATH::MDL, m_original_filename.c_str());
            if( it->m_resolved_filename.empty())
                it->m_resolved_filename = path_module->search( PATH::RESOURCE, m_original_filename.c_str());
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

    s << "Is valid (cached): " << (m_cached_is_valid  ? "true" : "false") << std::endl;
    s << "Is uvtile (cached): " << (m_cached_is_uvtile ? "true" : "false") << std::endl;
    s << "Is cubemap (cached): " << (m_cached_is_cubemap ? "true" : "false") << std::endl;

    s << "Number of UV tiles (cached): " << m_cached_uvtiles.size() << std::endl;

    for( mi::Size i = 0; i < m_cached_uvtiles.size(); ++i) {
        s << "UV tile ID (cached): " << i << std::endl;
        s << "  u (cached): " << m_cached_uvtiles[i].m_u << std::endl;
        s << "  v (cached): " << m_cached_uvtiles[i].m_v << std::endl;
        const Uvfilenames& f = m_uvfilenames[i];
        s << "  Resolved filename: " << f.m_resolved_filename << std::endl;
        s << "  Container membername: " << f.m_container_membername << std::endl;
        s << "  Resolved container membername: " << f.m_resolved_container_membername
          << std::endl;
    }

    s << "UV to ID (cached):" << std::endl;
    s << "  Min u (cached): " << m_cached_uv_to_id.m_min_u << std::endl;
    s << "  Min v (cached): " << m_cached_uv_to_id.m_min_v << std::endl;
    s << "  Count u (cached): " << m_cached_uv_to_id.m_count_u << std::endl;
    s << "  Count v (cached): " << m_cached_uv_to_id.m_count_v << std::endl;

    mi::Sint32 v = m_cached_uv_to_id.m_min_v + m_cached_uv_to_id.m_count_v - 1;
    for( ; v >= m_cached_uv_to_id.m_min_v; --v) {
        s << "  v=" << v << ": ";
        mi::Sint32 u = m_cached_uv_to_id.m_min_u;
        for( ; u < m_cached_uv_to_id.m_min_u + m_cached_uv_to_id.m_count_u; ++u) {
            if( u > m_cached_uv_to_id.m_min_u)
                s << ", ";
            s << static_cast<mi::Sint32>( m_cached_uv_to_id.get( u, v));
        }
        s << std::endl;
    }

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Image::get_size() const
{
    size_t s = dynamic_memory_consumption( m_original_filename)
             + dynamic_memory_consumption( m_mdl_file_path)
             + dynamic_memory_consumption( m_resolved_container_filename);

    s += dynamic_memory_consumption( m_cached_uv_to_id.m_ids);

    for( auto it = m_uvfilenames.begin(); it != m_uvfilenames.end(); ++it) {
        s += dynamic_memory_consumption( it->m_resolved_filename);
        s += dynamic_memory_consumption( it->m_container_membername);
        s += dynamic_memory_consumption( it->m_resolved_container_membername);
    }

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

const unsigned int* Image::get_tile_mapping(
    mi::Uint32& num_u,
    mi::Uint32& num_v,
    mi::Sint32& offset_u,
    mi::Sint32& offset_v) const
{
    if( m_cached_uv_to_id.m_ids.empty()) {
        num_u = num_v = offset_u = offset_v = 0;
        return nullptr;
    }

    num_u    =   m_cached_uv_to_id.m_count_u;
    num_v    =   m_cached_uv_to_id.m_count_v;
    offset_u = - m_cached_uv_to_id.m_min_u;
    offset_v = - m_cached_uv_to_id.m_min_v;
    return &m_cached_uv_to_id.m_ids[0];
}

void Image::reset_shared(
    DB::Transaction* transaction,
    bool is_uvtile,
    const std::vector<Uvtile>& uvtiles,
    const Uv_to_id& uv_to_id,
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

    Image_impl* impl = new Image_impl( is_uvtile, uvtiles, uv_to_id);

    setup_cached_values( impl);

    // We do not know the scope in which the instance of the proxy class ends up. Therefore, we have
    // to pick the global scope for the instance of the implementation class. Make sure to use
    // a DB name for the implementation class exactly for valid hashes.
    ASSERT( M_SCENE, impl_name.empty() ^ (impl_hash != mi::base::Uuid{0,0,0,0}));
    m_impl_tag = transaction->store_for_reference_counting(
        impl, !impl_name.empty() ? impl_name.c_str() : nullptr, /*privacy_level*/ 0);
    m_impl_hash = impl_hash;
}

namespace {

enum Uvtile_mode
{
    MODE_OFF,
    MODE_UDIM,
    MODE_U0_V0,
    MODE_U1_V1
};

Uvtile_mode get_uvtile_mode( const std::string& str)
{
    if( str.find("<UDIM>") != std::string::npos)
        return MODE_UDIM;
    if( str.find("<UVTILE0>") != std::string::npos)
        return MODE_U0_V0;
    if( str.find("<UVTILE1>") != std::string::npos)
        return MODE_U1_V1;
    return MODE_OFF;
}

/// Parses the (u,v) coordinates from the given uvtile/udim string.
///
/// Assumes that \p str matches the regular expression in resolve_filename() for \p mode.
///
/// \param mode uvtile/udim mode
/// \param str  string containing the indices, e.g. 1001 in udim mode
/// \param u    resulting u coordinate
/// \param v    resulting v coordinate
void parse_u_v( Uvtile_mode mode, const char* str, mi::Sint32& u, mi::Sint32& v)
{
    u = 0;
    v = 0;

    switch( mode) {
        case MODE_OFF:
            break;
        case MODE_UDIM: // UDIM (Mari), expands to four digits calculated as 1000 + (u+1 + 10*v)
            {
                unsigned num
                    = 1000 * (str[0] - '0') +
                       100 * (str[1] - '0') +
                        10 * (str[2] - '0') +
                         1 * (str[3] - '0') - 1001;
                // assume u in [0..9]
                u = num % 10;
                v = num / 10;
                break;
            }
        case MODE_U0_V0: // 0-based (Zbrush), expands to "_u0_v0" for the first tile
        case MODE_U1_V1: // 1-based (Mudbox), expands to "_u1_v1" for the first tile
            {
                mi::Sint32 sign = 1;
                const char* p = str + 2; // skip "_u"

                if (*p == '-') {
                    sign = -1;
                    ++p;
                }

                while( isdigit (*p)) {
                    u = 10*u + (*p - '0');
                    ++p;
                }
                u *= sign;

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

                if( mode == MODE_U1_V1) {
                    u -= 1;
                    v -= 1;
                }
                break;
            }
    }
}

} // namespace

Image_set* Image::resolve_filename( const std::string& filename)
{
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    Uvtile_mode mode = get_uvtile_mode( filename);

    if( mode == MODE_OFF) {

        std::string resolved_filename = path_module->search( PATH::RESOURCE, filename.c_str());
        if( !resolved_filename.empty())
            return new Plain_file_image_set( filename, resolved_filename);

        std::string lower( filename);
        STRING::to_lower( lower);

        auto p = lower.find( ".mdr:");
        if( p != std::string::npos) {
            // Archives must be found via the MDL search path. Strip directories since archives
            // have to be at the top-level of the search paths.
            std::string archive_path = filename.substr( 0, p + 4);
            std::string archive_name = HAL::Ospath::basename( archive_path);
            std::string resolved_archive_filename
                = path_module->search( PATH::MDL, archive_name.c_str());
            if( resolved_archive_filename.empty())
                return nullptr;
            return new Container_file_image_set( resolved_archive_filename, filename.substr( p + 5));
        }

        p = lower.find( ".mdle:");
        if( p != std::string::npos) {
            std::string mdle_path = filename.substr( 0, p + 5);
            // Try to resolve relative MDLE files via the INCLUDE search path.
            std::string resolved_mdle_path
                = path_module->search( PATH::INCLUDE, mdle_path.c_str());
            if( resolved_mdle_path.empty())
                return nullptr;
            return new Container_file_image_set( resolved_mdle_path, filename.substr( p + 6));
        }

        return nullptr;
    }

    // Compute regular expression for file name (without any directories)
    mi::Size p0 = filename.find( '<');
    mi::Size p1 = filename.find( '>');
    ASSERT( M_SCENE, p0 != std::string::npos);
    ASSERT( M_SCENE, p1 != std::string::npos);
    const std::string regexp = mode == MODE_UDIM ? "[0-9][0-9][0-9][0-9]" : "_u-?[0-9]+_v-?[0-9]+";
    std::string filename_regexp( filename);
    filename_regexp.replace( p0, p1-p0+1, regexp);
    filename_regexp = HAL::Ospath::basename( filename_regexp);

    // Obtain search paths and directory relative to search paths
    std::string dirname = HAL::Ospath::dirname( filename);
    PATH::Path_module::Search_path search_paths;
    std::string relative_dirname;
    if( DISK::is_path_absolute( dirname))
    {
        if( !DISK::access( dirname.c_str()))
            return nullptr;
        search_paths.push_back( dirname);
    }
    else
    {
        search_paths = path_module->get_search_path( PATH::RESOURCE);
        relative_dirname = dirname;
    }
    dirname.clear();

    // Length of filename prefix before the udim marker
    const size_t filename_prefix_len = HAL::Ospath::basename( filename.substr( 0, p0)).size();

    mi::base::Handle<Udim_image_set> uvtiles( new Udim_image_set( filename));
    std::vector<std::string>& resolved_filenames         = uvtiles->get_resolved_filenames();
    std::vector<std::pair<mi::Sint32, mi::Sint32> >& uvs = uvtiles->get_uvtile_uvs();

    SYSTEM::Access_module<MDLC::Mdlc_module> mdl_module( false);

    for( auto it = search_paths.begin(); it != search_paths.end(); ++it) {

        std::string current_dir = HAL::Ospath::join( *it, relative_dirname);

        DISK::Directory dir;
        if (!dir.open( current_dir.c_str()))
            continue;

        std::string fn = dir.read();
        while( !fn.empty()) {

            if( mdl_module->utf8_match( filename_regexp.c_str(), fn.c_str())) {
                mi::Sint32 u, v;
                const std::string substr = fn.substr( filename_prefix_len);
                parse_u_v( mode, substr.c_str(), u, v);
                resolved_filenames.push_back( HAL::Ospath::join( current_dir, fn));
                uvs.emplace_back( u, v);
            }

            fn = dir.read();
        }

        if( resolved_filenames.size() > 0) {
            uvtiles->retain();
            return uvtiles.get();
        }
    }

    return nullptr;
}

void Image::setup_cached_values( const Image_impl* impl)
{
    m_cached_is_valid   = impl->is_valid();
    m_cached_is_uvtile  = impl->is_uvtile();
    m_cached_is_cubemap = impl->get_is_cubemap();
    m_cached_uvtiles    = impl->get_uvtiles();
    m_cached_uv_to_id   = impl->get_uv_to_id();

    for( auto it = m_cached_uvtiles.begin(); it != m_cached_uvtiles.end(); ++it)
      it->m_mipmap.reset();
}

Image_impl::Image_impl()
  : m_is_valid( false),
    m_is_uvtile( false),
    m_is_cubemap( false)
{
}

Image_impl::Image_impl( const Image_impl& other)
  : SCENE::Scene_element<Image_impl, ID_IMAGE_IMPL>( other),
    m_is_valid( false),
    m_is_uvtile( false),
    m_is_cubemap( false)
{
}

Image_impl::Image_impl( bool is_uvtile, const std::vector<Uvtile>& uvtiles, const Uv_to_id& uv_to_id)
  : m_is_valid( true),
    m_is_uvtile( is_uvtile),
    m_uvtiles( uvtiles),
    m_uv_to_id( uv_to_id)
{
    ASSERT( M_SCENE, !uvtiles.empty());

    for( const auto& uvtile: uvtiles) {
        ASSERT( M_SCENE, uvtile.m_mipmap);
        boost::ignore_unused( uvtile);
    }

    m_is_cubemap = m_uvtiles[0].m_mipmap->get_is_cubemap();
}

Image_impl::~Image_impl()
{
}

const IMAGE::IMipmap* Image_impl::get_mipmap( mi::Uint32 uvtile_id) const
{
    if( uvtile_id >= m_uvtiles.size())
        return nullptr;

    m_uvtiles[uvtile_id].m_mipmap->retain();
    return m_uvtiles[uvtile_id].m_mipmap.get();
}

const SERIAL::Serializable* Image_impl::serialize( SERIAL::Serializer* serializer) const
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    Scene_element_base::serialize( serializer);

    serializer->write( m_is_valid);
    serializer->write( m_is_uvtile);
    serializer->write( m_is_cubemap);

    serializer->write_size_t( m_uvtiles.size());
    for( auto it = m_uvtiles.begin(); it != m_uvtiles.end(); ++it) {
        serializer->write( it->m_u);
        serializer->write( it->m_v);
        image_module->serialize_mipmap( serializer, it->m_mipmap.get());
    }

    serializer->write( m_uv_to_id.m_count_u);
    serializer->write( m_uv_to_id.m_count_v);
    serializer->write( m_uv_to_id.m_min_u);
    serializer->write( m_uv_to_id.m_min_v);
    SERIAL::write( serializer, m_uv_to_id.m_ids);

    return this + 1;
}

SERIAL::Serializable* Image_impl::deserialize( SERIAL::Deserializer* deserializer)
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    size_t s;

    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_is_valid);
    deserializer->read( &m_is_uvtile);
    deserializer->read( &m_is_cubemap);

    deserializer->read_size_t( &s);
    m_uvtiles.resize(s);
    for( auto it = m_uvtiles.begin(); it != m_uvtiles.end(); ++it) {
        deserializer->read( &it->m_u);
        deserializer->read( &it->m_v);
        it->m_mipmap = image_module->deserialize_mipmap( deserializer);
    }

    deserializer->read( &m_uv_to_id.m_count_u);
    deserializer->read( &m_uv_to_id.m_count_v);
    deserializer->read( &m_uv_to_id.m_min_u);
    deserializer->read( &m_uv_to_id.m_min_v);
    SERIAL::read( deserializer, &m_uv_to_id.m_ids);

    return this + 1;
}

void Image_impl::dump() const
{
    std::ostringstream s;

    s << "Is valid: " << (m_is_valid  ? "true" : "false") << std::endl;
    s << "Is uvtile: " << (m_is_uvtile  ? "true" : "false") << std::endl;

    s << "Number of UV tiles: " << m_uvtiles.size() << std::endl;

    for( auto it = m_uvtiles.begin(); it != m_uvtiles.end(); ++it) {
        s << "UV tile ID: " << it - m_uvtiles.begin() << std::endl;
        s << "  u: " << it->m_u << std::endl;
        s << "  v: " << it->m_v << std::endl;
        s << "  Miplevel: " << it->m_mipmap->get_nlevels() << std::endl;
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas( it->m_mipmap->get_level( 0));
        s << "  Pixel type: " << canvas->get_type() << std::endl;
        s << "  Pixels: " << canvas->get_resolution_x()
          << "x" << canvas->get_resolution_y()
          << "x" << canvas->get_layers_size() << std::endl;
        s << "  Gamma: " << canvas->get_gamma() << std::endl;
    }

    s << "UV to ID:" << std::endl;
    s << "  Min u: " << m_uv_to_id.m_min_u << std::endl;
    s << "  Min v: " << m_uv_to_id.m_min_v << std::endl;
    s << "  Count u: " << m_uv_to_id.m_count_u << std::endl;
    s << "  Count v: " << m_uv_to_id.m_count_v << std::endl;

    mi::Sint32 v = m_uv_to_id.m_min_v + m_uv_to_id.m_count_v - 1;
    for( ; v >= m_uv_to_id.m_min_v; --v) {
        s << "  v=" << v << ": ";
        mi::Sint32 u = m_uv_to_id.m_min_u;
        for( ; u < m_uv_to_id.m_min_u + m_uv_to_id.m_count_u; ++u) {
            if( u >  m_uv_to_id.m_min_u)
                s << ", ";
            s << static_cast<mi::Sint32>( m_uv_to_id.get( u, v));
        }
        s << std::endl;
    }

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Image_impl::get_size() const
{
    size_t size = sizeof( *this);

    for( auto it = m_uvtiles.begin(); it != m_uvtiles.end(); ++it)
        size += it->m_mipmap->get_size();

    return size;
}

DB::Journal_type Image_impl::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

void Image_impl::get_uvtile_uv_ranges(
    mi::Sint32& min_u, mi::Sint32& min_v, mi::Sint32& max_u, mi::Sint32& max_v) const
{
    if( m_uv_to_id.m_ids.empty()) {
        min_u = min_v = max_u = max_v = 0;
        return;
    }

    min_u = m_uv_to_id.m_min_u;
    min_v = m_uv_to_id.m_min_v;
    max_u = m_uv_to_id.m_min_u + m_uv_to_id.m_count_u - 1;
    max_v = m_uv_to_id.m_min_v + m_uv_to_id.m_count_v - 1;
}

mi::Size Image_impl::get_uvtile_length() const
{
    return m_uvtiles.size();
}

mi::Sint32 Image_impl::get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const
{
    if( uvtile_id >= m_uvtiles.size())
        return -1;

    u = m_uvtiles[uvtile_id].m_u;
    v = m_uvtiles[uvtile_id].m_v;
    return 0;
}

mi::Uint32 Image_impl::get_uvtile_id( Sint32 u, Sint32 v) const
{
    return m_uv_to_id.get( u, v);
}

const unsigned int* Image_impl::get_tile_mapping(
    mi::Uint32& num_u,
    mi::Uint32& num_v,
    mi::Sint32& offset_u,
    mi::Sint32& offset_v) const
{
    if( m_uv_to_id.m_ids.empty()) {
        num_u = num_v = offset_u = offset_v = 0;
        return nullptr;
    }

    num_u    =   m_uv_to_id.m_count_u;
    num_v    =   m_uv_to_id.m_count_v;
    offset_u = - m_uv_to_id.m_min_u;
    offset_v = - m_uv_to_id.m_min_v;
    return &m_uv_to_id.m_ids[0];
}

} // namespace DBIMAGE

} // namespace MI
