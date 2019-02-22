/***************************************************************************************************
 * Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/data/serial/i_serializer.h>
#include <base/util/registry/i_config_registry.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/i_image_utilities.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

namespace {

    std::string empty_str;
}

namespace MI {

namespace DBIMAGE {

Uvtile_mode get_uvtile_mode(const std::string &file_name)
{
    if(file_name.find("<UDIM>") != std::string::npos)
        return MODE_UDIM;
    if(file_name.find("<UVTILE0>") != std::string::npos)
        return MODE_U0_V0;
    if(file_name.find("<UVTILE1>") != std::string::npos)
        return MODE_U1_V1;

    // no valid marker found
    return MODE_OFF;
}

class Single_file : public Image_set
{
public:

    Single_file(
        const std::string& original_filename,
        const std::string& resolved_filename)
        : m_original_filename(original_filename)
        , m_resolved_filename(resolved_filename) {}

    bool is_uvtile() const
    {
        return false;
    }

    const char* get_resolved_filename(mi::Size index) const
    {
        ASSERT(M_SCENE, index == 0);
        return m_resolved_filename.c_str();
    }

    const char* get_original_filename() const
    {
        return m_original_filename.c_str();
    }

    bool get_uv_mapping(mi::Size index, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT(M_SCENE, index == 0);
        u = 0; v = 0;
        return true;
    }

    mi::Size get_length() const
    {
        return 1;
    }

private:
    std::string m_original_filename;
    std::string m_resolved_filename;
};

class Uvtile_set : public Image_set
{
public:

    Uvtile_set(const std::string& original_filename)
        : m_original_filename(original_filename) { }

    bool is_uvtile() const
    {
        return true;
    }

    const char* get_original_filename() const
    {
        return m_original_filename.c_str();
    }

    const char* get_resolved_filename(mi::Size i) const
    {
        if (i >= m_filenames.size())
            return NULL;
        return m_filenames[i].c_str();
    }

    bool get_uv_mapping(mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        if (i >= m_uvs.size())
            return false;
        u = m_uvs[i].first;
        v = m_uvs[i].second;
        return true;
    }

    mi::Size get_length() const
    {
        return m_filenames.size();
    }

    std::vector<std::string>& get_uvtile_names()
    {
        return m_filenames;
    }
    std::vector<std::pair<mi::Sint32, mi::Sint32> >& get_uvtile_uvs()
    {
        return m_uvs;
    }

private:
    std::string m_original_filename;
    std::vector<std::string> m_filenames;
    std::vector<std::pair<mi::Sint32, mi::Sint32> > m_uvs;
};

Image::Image() : m_is_uvtile(false)
{
    set_default_pink_dummy_mipmap();
}

Image::Image( const Image& other)
  : SCENE::Scene_element<Image, ID_IMAGE>( other),
    m_is_uvtile(false)
{
    set_default_pink_dummy_mipmap();
}

Image::~Image()
{
}

mi::Sint32 Image::reset_file( const std::string& original_filename)
{
    mi::base::Handle< Image_set> image_set( resolve_filename( original_filename));
    if( !image_set)
        return -2;

    return reset( image_set.get());
}

mi::Sint32 Image::reset_reader( mi::neuraylib::IReader* reader, const char* image_format)
{
    mi::Sint32 result = 0;
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap( reader, image_format,
        /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true, &result));
    if( result < 0)
        return result;
 
    m_is_uvtile = false;
    m_uv_to_index.reset();
    m_uvtiles.resize(1);
    m_uvtiles[0].m_mipmap = mipmap;

    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();

    return 0;
}

mi::Sint32 Image::reset(
    const Image_set* image_set)
{
    if( !image_set)
        return -1;

    mi::Size number_of_tiles = image_set->get_length();
    if( number_of_tiles == 0)
        return -1;

    mi::Sint32 u=0, v=0;
    image_set->get_uv_mapping( 0, u, v);
    mi::Sint32 u_min = u, u_max = u, v_min = v, v_max = v;
    for( mi::Size i=1; i< number_of_tiles; ++i)
    {
        image_set->get_uv_mapping( i, u, v);
        u_min = mi::math::min( u_min, u);
        u_max = mi::math::max( u_max, u);
        v_min = mi::math::min( v_min, v);
        v_max = mi::math::max( v_max, v);
    }

    std::vector<mi::base::Handle<MI::IMAGE::IMipmap> > temp_mipmaps( number_of_tiles);
    Uv_to_index temp_indices;
    temp_indices.reset( u_min, u_max, v_min, v_max);
    Sint32 result = 0;
    for ( mi::Uint32 i = 0; i < number_of_tiles; ++i)
    {
        int u = 0, v = 0;
        image_set->get_uv_mapping( i, u, v);
        if (!temp_indices.set( u, v, i))
        {
            result = -2;
            break;
        }
        temp_mipmaps[i] = image_set->create_mipmap( i);
        if ( !temp_mipmaps[i].is_valid_interface())
        {
            result = -3;
            break;
        }
    }
    if ( result)
    {
        return result;
    }

#define set_str(s) \
    (s) ? (s) : ""

    m_is_uvtile = image_set->is_uvtile();
    m_original_filename = set_str( image_set->get_original_filename());
    m_mdl_file_path = set_str( image_set->get_mdl_file_path());
    m_resolved_container_filename = set_str( image_set->get_container_filename());

    m_uv_to_index = temp_indices;
    m_uvtiles.resize( number_of_tiles);

    for( mi::Uint32 i=0; i< number_of_tiles; ++i)
    {
        m_uvtiles[ i].m_resolved_filename = set_str( image_set->get_resolved_filename( i));
        m_uvtiles[ i].m_mdl_file_path = set_str( image_set->get_mdl_url( i));
        m_uvtiles[ i].m_container_membername =
            set_str( image_set->get_container_membername( i));
        if (!m_uvtiles[i].m_container_membername.empty())
            m_uvtiles[i].m_resolved_container_membername =
            m_resolved_container_filename + ":" + m_uvtiles[i].m_container_membername;

        mi::Sint32 u=0, v=0;
        image_set->get_uv_mapping(i, u, v);

        m_uvtiles[ i].m_u = u;
        m_uvtiles[ i].m_v = v;

        m_uvtiles[ i].m_mipmap = temp_mipmaps[ i];
    }

#undef set_str
    return 0;
}

void Image::set_mipmap( IMAGE::IMipmap* mipmap)
{
    ASSERT( M_SCENE, mipmap);

    m_is_uvtile = false;
    m_uvtiles.resize(1);
    m_uvtiles[0].m_mipmap = make_handle_dup( mipmap);
    m_uvtiles[0].m_mdl_file_path.clear();
    m_uvtiles[0].m_container_membername.clear();
    m_uvtiles[0].m_resolved_container_membername.clear();
    m_uvtiles[0].m_resolved_filename.clear();

    m_uv_to_index.reset();
    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();
}

IMAGE::IMipmap* Image::get_mipmap()
{
    // Mutable access to the mipmap allows potential modifications. Treat mipmap from now on
    // as memory based (even if was originally loaded from file). Some tiles not yet loaded might
    // still get loaded from file.
    // Not allowed for uv tiles
    if(m_is_uvtile)
        return NULL;

    m_uvtiles[0].m_resolved_filename.clear();
    m_uvtiles[0].m_container_membername.clear();
    m_uvtiles[0].m_resolved_container_membername.clear();
    m_uvtiles[0].m_mdl_file_path.clear();
    m_original_filename.clear();
    m_mdl_file_path.clear();
    m_resolved_container_filename.clear();

    m_uvtiles[0].m_mipmap->retain();
    return m_uvtiles[0].m_mipmap.get();
}

const IMAGE::IMipmap* Image::get_mipmap(mi::Uint32 uvtile_id) const //-V659 PVS
{
    if(uvtile_id > m_uvtiles.size())
        return NULL;
    m_uvtiles[uvtile_id].m_mipmap->retain();
    return m_uvtiles[uvtile_id].m_mipmap.get();
}

const std::string& Image::get_filename(mi::Uint32 uvtile_id) const
{
    if(uvtile_id > m_uvtiles.size())
        return empty_str;
    return m_uvtiles[uvtile_id].m_resolved_filename;
}

const std::string& Image::get_container_membername(mi::Uint32 uvtile_id) const
{
    if(uvtile_id > m_uvtiles.size())
        return empty_str;
    return m_uvtiles[uvtile_id].m_container_membername;
}


const std::string& Image::get_resolved_container_membername(mi::Uint32 uvtile_id) const
{
    if (uvtile_id > m_uvtiles.size())
        return empty_str;
    return m_uvtiles[uvtile_id].m_resolved_container_membername;
}

const std::string& Image::get_original_filename() const
{
    if( m_original_filename.empty() && !m_uvtiles[0].m_mdl_file_path.empty()) {
        SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
        const CONFIG::Config_registry& registry = config_module->get_configuration();
        bool flag = false;
        registry.get_value( "deprecated_mdl_file_path_as_original_filename", flag);
        if( flag)
            return m_mdl_file_path;
    }

    return m_original_filename;
}

const std::string& Image::get_mdl_file_path( ) const
{
    return m_mdl_file_path;
}

bool Image::get_is_cubemap() const
{
    // a uv-tile set cannot be a cubemap
    return m_is_uvtile ? false : m_uvtiles[0].m_mipmap->get_is_cubemap();
}

bool Image::is_valid() const
{
    if(m_uvtiles.size() == 0)
        return false;

    if (m_uvtiles.size() > 1)
        return true;

    if( m_uvtiles[0].m_mipmap->get_nlevels() > 1)
        return true;

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( m_uvtiles[0].m_mipmap->get_level( 0));
    if(    canvas->get_resolution_x() > 1
        || canvas->get_resolution_y() > 1
        || canvas->get_layers_size()  > 1)
        return true;

    mi::base::Handle<const mi::neuraylib::ITile> tile( canvas->get_tile( 0, 0));
    mi::math::Color color;
    tile->get_pixel( 0, 0, &color.r);
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    return color != pink;
}

const SERIAL::Serializable* Image::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    serializer->write( serializer->is_remote() ? "" : m_original_filename);
    serializer->write( serializer->is_remote() ? "" : m_mdl_file_path);
    serializer->write( serializer->is_remote() ? "" : m_resolved_container_filename);
    serializer->write( HAL::Ospath::sep());

    serializer->write( m_is_uvtile);
    serializer->write_size_t( m_uvtiles.size());
    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    for(std::vector<Uvtile>::const_iterator it = m_uvtiles.begin(); it<m_uvtiles.end(); ++it)
    {
        serializer->write( serializer->is_remote() ? "" : it->m_resolved_filename);
        serializer->write( serializer->is_remote() ? "" : it->m_container_membername);
        serializer->write( serializer->is_remote() ? "" : it->m_resolved_container_membername);
        serializer->write( serializer->is_remote() ? "" : it->m_mdl_file_path);
        serializer->write( it->m_u);
        serializer->write( it->m_v);
        image_module->serialize_mipmap( serializer, it->m_mipmap.get());
    }

    serializer->write(m_uv_to_index.m_nu);
    serializer->write(m_uv_to_index.m_nv);
    serializer->write(m_uv_to_index.m_offset_u);
    serializer->write(m_uv_to_index.m_offset_v);
    serializer->write_size_t( m_uv_to_index.m_uv.size());
    for(std::vector<mi::Uint32>::const_iterator it = m_uv_to_index.m_uv.begin(); 
        it< m_uv_to_index.m_uv.end(); ++it)
        serializer->write(*it);

    return this + 1;
}

SERIAL::Serializable* Image::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_original_filename);
    deserializer->read( &m_mdl_file_path);
    deserializer->read( &m_resolved_container_filename);
    std::string serializer_sep;
    deserializer->read( &serializer_sep);
    bool convert_path =  serializer_sep != HAL::Ospath::sep();
    
    // Adjust filename for this host
    if( convert_path)
    {
        m_original_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_original_filename);

        m_resolved_container_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_resolved_container_filename);
    }
    if( !( m_resolved_container_filename.empty() ||
        DISK::is_file( m_resolved_container_filename.c_str())))
        m_resolved_container_filename.clear();

    deserializer->read(&m_is_uvtile);
    size_t s;
    deserializer->read_size_t(&s);
    m_uvtiles.resize(s);

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    for( std::vector<Uvtile>::iterator it = m_uvtiles.begin(); it<m_uvtiles.end(); ++it)
    {
        deserializer->read( &it->m_resolved_filename);
        deserializer->read( &it->m_container_membername);
        deserializer->read( &it->m_resolved_container_membername);
        deserializer->read( &it->m_mdl_file_path);
        deserializer->read( &it->m_u);
        deserializer->read( &it->m_v);
        it->m_mipmap = image_module->deserialize_mipmap( deserializer);

        // Adjust filenames for this host
        if( !m_original_filename.empty() || !it->m_mdl_file_path.empty()) {
            if( convert_path)
            {
                it->m_resolved_filename = HAL::Ospath::convert_to_platform_specific_path(
                    it->m_resolved_filename);
                it->m_container_membername = m_resolved_container_filename.empty() ? "" :
                    HAL::Ospath::convert_to_platform_specific_path(
                        it->m_container_membername);

                it->m_resolved_container_membername = m_resolved_container_filename.empty() ? "" :
                    m_resolved_container_filename + ":" + it->m_container_membername;
            }
            // Re-resolve filename if it is not meaningful for this host. If unsuccessful,
            // clear value(no error since we no longer require all resources to be present
            // on all nodes).
            if( !DISK::is_file( it->m_resolved_filename.c_str())) {
                const std::string& s
                    = !m_original_filename.empty() ? m_original_filename : it->m_mdl_file_path;
                // TODO: fix for files with uv-tile/udim markers
                it->m_resolved_filename = path_module->search( PATH::MDL, s.c_str());
                if( it->m_resolved_filename.empty())
                    it->m_resolved_filename = path_module->search( PATH::RESOURCE, s.c_str());
            }
        }
    }
    deserializer->read(&m_uv_to_index.m_nu);
    deserializer->read(&m_uv_to_index.m_nv);
    deserializer->read(&m_uv_to_index.m_offset_u);
    deserializer->read(&m_uv_to_index.m_offset_v);
    deserializer->read_size_t(&s);
    m_uv_to_index.m_uv.resize(s);
    mi::Uint32 index;
    for(std::vector<mi::Uint32>::iterator it = m_uv_to_index.m_uv.begin(); 
        it< m_uv_to_index.m_uv.end(); ++it) {
        deserializer->read(&index);
        *it = index;
    }

    return this + 1;
}

void Image::dump() const
{
    std::ostringstream s;

    s << "Original filename: " << m_original_filename << std::endl;
    s << "Absolute MDL file path: " << m_mdl_file_path << std::endl;
    s << "Resolved container filename: " << m_resolved_container_filename << std::endl;

    s << "Number of mip maps: " << m_uvtiles.size() << std::endl;
    s << "Is uv-tile-set: " << (m_is_uvtile ? "true" : "false") << std::endl;

    for( std::vector<Uvtile>::const_iterator it = m_uvtiles.begin(); it<m_uvtiles.end(); ++it)
    {
        s << "UV-tile " << it - m_uvtiles.begin() << ":" << std::endl;
        s << "u = " << it->m_u << ", v = " << it->m_v << std::endl;
        s << "Resolved filename: " << it->m_resolved_filename << std::endl;
        s << "Resolved container membername: " << it->m_resolved_container_membername << std::endl;
        s << "MDL file url: " << it->m_mdl_file_path << std::endl;

        s << "Miplevel: " << it->m_mipmap->get_nlevels() << std::endl;
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas( it->m_mipmap->get_level( 0));
        s << "Pixel type: " << canvas->get_type() << std::endl;
        s << "Pixels: " << canvas->get_resolution_x()
            << "x" << canvas->get_resolution_y()
            << "x" << canvas->get_layers_size() << std::endl;
        s << "Gamma: " << canvas->get_gamma() << std::endl;
    }

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Image::get_size() const
{
    size_t s = 0;
    for( std::vector<Uvtile>::const_iterator it = m_uvtiles.begin(); it<m_uvtiles.end(); ++it)
    {
        s += dynamic_memory_consumption( it->m_resolved_filename);
        s += dynamic_memory_consumption( it->m_container_membername);
        s += dynamic_memory_consumption( it->m_resolved_container_membername);
        s += dynamic_memory_consumption( it->m_mdl_file_path);
        s +=it->m_mipmap->get_size();
    }
    return sizeof( *this)
        + dynamic_memory_consumption( m_original_filename)
        + dynamic_memory_consumption( m_mdl_file_path)
        + dynamic_memory_consumption( m_resolved_container_filename)
        + s;
}

DB::Journal_type Image::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

Uint Image::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Image::get_scene_element_references( DB::Tag_set* result) const
{
}

void Image::set_default_pink_dummy_mipmap()
{
    m_is_uvtile = false;
    m_uvtiles.resize(1);
    m_uv_to_index.reset();

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    m_uvtiles[0].m_mipmap = image_module->create_mipmap( IMAGE::PT_RGBA, 1, 1);
    mi::base::Handle<mi::neuraylib::ICanvas> canvas( m_uvtiles[0].m_mipmap->get_level( 0));
    mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( 0, 0));
    mi::math::Color pink( 1.0f, 0.0f, 1.0f, 1.0f);
    tile->set_pixel( 0, 0, &pink.r);
}

// Parse a file name and enter it into a resource set.
void Image::parse_u_v(
    const Uvtile_mode mode,
    const char *str,
    mi::Sint32& u,
    mi::Sint32& v)
{
    mi::Sint32 sign = 1;
    u = v = 0;
    switch (mode) {
    case MODE_OFF:
        break;
    case MODE_UDIM:
        // UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
        {
            unsigned num =
                1000 * (str[0] - '0') +
                100 * (str[1] - '0') +
                10 * (str[2] - '0') +
                1 * (str[3] - '0') - 1001;

            // assume u, v [0..9]
            u = num % 10;
            v = num / 10;
        }
        break;
    case MODE_U0_V0:
        // 0-based (Zbrush), expands to "_u0_v0" for the first tile
    case MODE_U1_V1:
        // 1-based (Mudbox), expands to "_u1_v1" for the first tile
        {
            char const *n = str + 2;

            if (*n == '-') {
                sign = -1;
                ++n;
            } else {
                sign = +1;
            }
            while (isdigit(*n)) {
                u = u * 10 + *n - '0';
                ++n;
            }
            u *= sign;

            if (*n == '_')
                ++n;
            if (*n == 'v')
                ++n;

            if (*n == '-') {
                sign = -1;
                ++n;
            } else {
                sign = +1;
            }
            while (isdigit(*n)) {
                v = v * 10 + *n - '0';
                ++n;
            }
            v *= sign;

            if (mode == MODE_U1_V1) {
                u -= 1;
                v -= 1;
            }
        }
        break;
    }
}

Image_set* Image::resolve_filename(
    const std::string& filename) const
{
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    Uvtile_mode mode = get_uvtile_mode(filename);
    if(mode == MODE_OFF)
    {
        std::string resolved_filename
            = path_module->search(PATH::RESOURCE, filename.c_str());
        if( resolved_filename.empty())
            return NULL;

        return new Single_file( filename, resolved_filename);
    }

    const std::string expr = mode == MODE_UDIM ? "[0-9][0-9][0-9][0-9]" :
        "_u-?[0-9]+_v-?[0-9]+";

    mi::Size p0 = filename.find("<");
    mi::Size p1 = filename.find(">");
    ASSERT(M_SCENE, p0 != std::string::npos);
    ASSERT(M_SCENE, p1 != std::string::npos);

    std::string udim_basename = filename.substr(0, p0);
    std::string file_mask(filename);
    file_mask.replace(p0, p1-p0+1, expr);
    file_mask = HAL::Ospath::basename(file_mask);

    // resolve directory, if not absolute 
    std::string udim_dirname = HAL::Ospath::dirname(udim_basename);
    PATH::Path_module::Search_path search_paths;
    if( DISK::is_path_absolute( udim_dirname))
    {
        if (!DISK::access( udim_dirname.c_str()))
            return NULL;
        search_paths.push_back(udim_dirname);
        udim_dirname.clear();
    }
    else
    {
        search_paths = path_module->get_search_path(PATH::RESOURCE);
    }

    udim_basename =  HAL::Ospath::basename(udim_basename);
    const size_t p = udim_basename.size();

    mi::base::Handle<Uvtile_set> uvtiles(new Uvtile_set(filename));
    std::vector<std::string> &filenames = uvtiles->get_uvtile_names();
    std::vector<std::pair<mi::Sint32, mi::Sint32> > &uvs = uvtiles->get_uvtile_uvs();

    SYSTEM::Access_module<MDLC::Mdlc_module> mdl_module(false);

    for(PATH::Path_module::Search_path::const_iterator it=search_paths.begin();
        it != search_paths.end(); ++it)
    {
        std::string current_dir = *it + udim_dirname;

        DISK::Directory dir;
        dir.open(current_dir.c_str());

        std::string fn = dir.read();
        while(!fn.empty())
        {
            if (mdl_module->utf8_match(file_mask.c_str(), fn.c_str()))
            {
                mi::Sint32 u, v;
                const std::string substr = fn.substr(p, fn.size()-p);
                Image::parse_u_v(mode, substr.c_str(), u, v);

                filenames.push_back(HAL::Ospath::join(current_dir, fn));
                uvs.push_back(std::make_pair(u, v));
            }

            fn = dir.read();
        }

        if(filenames.size() > 0)
        {
            uvtiles->retain();
            return uvtiles.get();
        }
    }
    // nothing found
    return NULL;
}

mi::Size Image::get_uvtile_length() const
{
    return m_uvtiles.size();
}

mi::Sint32 Image::get_uvtile_uv(Uint32 uvtile_id, Sint32& u, Sint32& v) const
{
    if(uvtile_id >= m_uvtiles.size())
        return -1;

    const Uvtile& uv_tile = m_uvtiles[uvtile_id];
    u = uv_tile.m_u;
    v = uv_tile.m_v;

    return 0;
}

mi::Uint32 Image::get_uvtile_id( Sint32 u, Sint32 v) const
{
    return m_uv_to_index.get(u, v);
}

bool Image::is_uvtile() const
{
    return m_is_uvtile;
}

Image::Uvtile::Uvtile() : m_u(0), m_v(0) 
{
}

void Image::Uv_to_index::reset( 
    mi::Sint32 u_min, mi::Sint32 u_max, mi::Sint32 v_min, mi::Sint32 v_max)
{
    m_nu = u_max - u_min + 1;
    m_nv = v_max - v_min + 1;
    m_offset_u = 0 - u_min;
    m_offset_v = 0 - v_min;
    m_uv.resize(m_nu * m_nv, -1);
}

mi::Uint32 Image::Uv_to_index::get(mi::Sint32 u, mi::Sint32 v) const
{
    const mi::Sint32 uu = u + m_offset_u;
    const mi::Sint32 vv = v + m_offset_v;
    if(uu < 0 || uu > Sint32(m_nu))
        return static_cast<mi::Uint32>(-1);
    if(vv < 0 || vv > Sint32(m_nv))
        return -1;
    return m_uv[vv * m_nu + uu];
}

bool Image::Uv_to_index::set(mi::Sint32 u, mi::Sint32 v, mi::Uint32 index)
{
    const mi::Sint32 uu = u + m_offset_u;
    const mi::Sint32 vv = v + m_offset_v;
    if(uu < 0 || uu >= Sint32(m_nu))
        return false;
    if(vv < 0 || vv >= Sint32(m_nv))
        return false;
    const mi::Size i = vv * m_nu + uu;
    if(m_uv[i] != static_cast<mi::Uint32>(-1))
        return false;
    m_uv[i] =  index;

    return true;
}

char const* Image_set::get_container_filename() const
{
    return NULL;
}

char const* Image_set::get_original_filename() const
{
    return NULL;
}

char const* Image_set::get_mdl_file_path() const
{
    return NULL;
}

char const* Image_set::get_mdl_url(mi::Size i) const
{
    return NULL;
}

char const* Image_set::get_resolved_filename(mi::Size i) const
{
    return NULL;
}

char const* Image_set::get_container_membername(mi::Size i) const
{
    return NULL;
}

mi::neuraylib::IReader* Image_set::open_reader(mi::Size i) const
{ 
    return NULL;
}

mi::neuraylib::ICanvas* Image_set::get_canvas(mi::Size i) const
{ 
    return NULL;
}

bool Image_set::is_uvtile() const 
{ 
    return true; 
}

bool Image_set::is_mdl_container() const
{ 
    return false; 
}

const char* Image_set::get_image_format() const
{
    return NULL;
}

MI::IMAGE::IMipmap* Image_set::create_mipmap(mi::Size i) const
{
    if( i >= get_length())
        return NULL;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    // mdl container based
    if( is_mdl_container())
    {
        mi::base::Handle<mi::neuraylib::IReader> reader(
            open_reader( i));
        if( !reader)
            return NULL;

        ASSERT( M_SCENE, get_container_filename());
        ASSERT( M_SCENE, get_container_membername(i));

        return image_module->create_mipmap(
            reader.get(),
            get_container_filename(),
            get_container_membername( i),
            /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true);
    }

    // file based
    const char* resolved_filename = get_resolved_filename( i);
    if( resolved_filename && strlen(resolved_filename) > 0)
    {
       return image_module->create_mipmap( 
            resolved_filename,
            /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true);
    }

    // canvas based
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        get_canvas(i));
    if(canvas)
    {
        std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
        canvases[0] = canvas;
        return image_module->create_mipmap( canvases);
    }

    // reader based
    mi::base::Handle<mi::neuraylib::IReader> reader(
        open_reader( i));
    if( !reader)
        return NULL;

    ASSERT(M_SCENE, get_image_format());
    return image_module->create_mipmap(
        reader.get(),
        get_image_format(),
        /*tile_width*/ 0, /*tile_height*/ 0, /*only_first_level*/ true);
}


} // namespace DBIMAGE

} // namespace MI
