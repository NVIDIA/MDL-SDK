/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Handles the DB element Texture.

#include "pch.h"

#include "i_texture.h"

#include <base/lib/log/log.h>
#include <base/data/serial/i_serializer.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/ireader.h>
#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h>

#include <sstream>

namespace MI {

namespace TEXTURE {

Texture::Texture()
{
    m_image       = DB::Tag();
    m_gamma       = 0.0f;
    m_compression = TEXTURE_NO_COMPRESSION;
}

void Texture::set_image( DB::Tag image)
{
    m_image = image;
}

DB::Tag Texture::get_image() const
{
    return m_image;
}

void Texture::set_gamma( mi::Float32 gamma)
{
    m_gamma = gamma;
}

mi::Float32 Texture::get_gamma() const
{
    return m_gamma;
}

mi::Float32 Texture::get_effective_gamma(
    DB::Transaction* transaction,
    mi::Uint32 uvtile_id) const
{
    if( m_gamma != 0.0 || !m_image)
      return m_gamma;

    DB::Access<DBIMAGE::Image> image( m_image, transaction);
    mi::base::Handle<const IMAGE::IMipmap> mipmap( image->get_mipmap( transaction, uvtile_id));
    if( !mipmap)
        return m_gamma;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    return canvas->get_gamma();
}

void Texture::set_compression( Texture_compression compression)
{
    m_compression = compression;
}

Texture_compression Texture::get_compression() const
{
    return m_compression;
}

const SERIAL::Serializable* Texture::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    serializer->write( m_image);
    serializer->write( m_gamma);
    serializer->write( static_cast<Uint>( m_compression));
    return this + 1;
}

SERIAL::Serializable *Texture::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_image);
    deserializer->read( &m_gamma);
    Uint value;
    deserializer->read( &value);
    m_compression = static_cast<Texture_compression>( value);
    return this + 1;
}

void Texture::dump() const
{
    std::ostringstream s;

    s << "Image: tag " << m_image.get_uint() << std::endl;
    s << "Gamma: " << m_gamma << std::endl;
    s << "Compression: " << m_compression << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Texture::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Texture, ID_TEXTURE>::get_size()
            - sizeof( SCENE::Scene_element<Texture, ID_TEXTURE>);
}

DB::Journal_type Texture::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

Uint Texture::bundle( DB::Tag* results, Uint size) const
{
    if( !m_image || size == 0)
        return 0;

    *results = m_image;
    return 1;
}

void Texture::get_scene_element_references( DB::Tag_set* result) const
{
    if( m_image)
        result->insert( m_image);
}

DB::Tag load_mdl_texture(
    DB::Transaction* transaction,
    DBIMAGE::Image_set* image_set,
    const mi::base::Uuid& impl_hash,
    bool shared_proxy,
    mi::Float32 gamma)
{
    if( !image_set || image_set->get_length() == 0)
        return DB::Tag( 0);

    std::string identifier;
    if( image_set->is_mdl_container()) {
        identifier = image_set->get_container_filename() + std::string( "_")
            + image_set->get_container_membername( 0);
    } else {
        identifier = image_set->get_resolved_filename( 0);
        if( identifier.empty()) {
            identifier = "without_name";
            // Never share the proxy for memory-based resources.
            shared_proxy = false;
        }
    }

    std::string db_texture_name = shared_proxy ? "MI_default_" : "";
    db_texture_name += "texture_" + identifier + "_" +
        std::string( STRING::lexicographic_cast_s<std::string>( gamma));
    if( !shared_proxy)
        db_texture_name
            = MDL::DETAIL::generate_unique_db_name( transaction, db_texture_name.c_str());

    DB::Tag texture_tag = transaction->name_to_tag( db_texture_name.c_str());
    if( texture_tag)
        return texture_tag;

    DB::Privacy_level privacy_level = transaction->get_scope()->get_level();

    std::string db_image_name = shared_proxy ? "MI_default_" : "";
    db_image_name += "image_" + identifier;
    if( !shared_proxy)
        db_image_name = MDL::DETAIL::generate_unique_db_name( transaction, db_image_name.c_str());

    DB::Tag image_tag = transaction->name_to_tag( db_image_name.c_str());
    if( !image_tag) {
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_image_set( transaction, image_set, impl_hash);
        image_tag = transaction->store_for_reference_counting(
            image, db_image_name.c_str(), privacy_level);
    }

    Texture* texture = new Texture();
    texture->set_image( image_tag);
    texture->set_gamma( gamma);

    texture_tag = transaction->store_for_reference_counting(
        texture, db_texture_name.c_str(), privacy_level);
    return texture_tag;
}

} // namespace TEXTURE

} // namespace MI
