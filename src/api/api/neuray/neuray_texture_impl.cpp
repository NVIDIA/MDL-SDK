/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the ITexture implementation.
 **/

#include "pch.h"

#include "neuray_texture_impl.h"
#include "neuray_transaction_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/iimage.h>

#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/texture/i_texture.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace NEURAY {

mi_static_assert(
    (int) mi::neuraylib::TEXTURE_NO_COMPRESSION     == (int) TEXTURE::TEXTURE_NO_COMPRESSION);
mi_static_assert(
    (int) mi::neuraylib::TEXTURE_MEDIUM_COMPRESSION == (int) TEXTURE::TEXTURE_MEDIUM_COMPRESSION);
mi_static_assert(
    (int) mi::neuraylib::TEXTURE_HIGH_COMPRESSION   == (int) TEXTURE::TEXTURE_HIGH_COMPRESSION);

DB::Element_base* Texture_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new TEXTURE::Texture;
}

mi::base::IInterface* Texture_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Texture_impl())->cast_to_major();
}

mi::neuraylib::Element_type Texture_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_TEXTURE;
}

mi::Sint32 Texture_impl::set_image( const char* name)
{
    if( !name)
        return -1;

    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( name);
    if( !tag.is_valid())
        return -2;

    if( !can_reference_tag( tag))
        return -3;

    DB::Typed_tag<DBIMAGE::Image> image_tag = tag;
    if( !image_tag)
        return -4;

    get_db_element()->set_image( tag);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return 0;
}

const char* Texture_impl::get_image() const
{
    DB::Tag tag = get_db_element()->get_image();
    return get_db_transaction()->tag_to_name( tag);
}

void Texture_impl::set_gamma( Float32 gamma)
{
    get_db_element()->set_gamma( gamma);
    add_journal_flag( SCENE::JOURNAL_CHANGE_FIELD);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
}

Float32 Texture_impl::get_gamma() const
{
    return get_db_element()->get_gamma();
}

Float32 Texture_impl::get_effective_gamma( mi::Uint32 uvtile_id) const
{
    return get_db_element()->get_effective_gamma( get_db_transaction(), uvtile_id);
}

void Texture_impl::set_compression( mi::neuraylib::Texture_compression compression)
{
    get_db_element()->set_compression( static_cast<TEXTURE::Texture_compression>( compression));
    add_journal_flag( SCENE::JOURNAL_CHANGE_FIELD);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
}

mi::neuraylib::Texture_compression Texture_impl::get_compression() const
{
    return static_cast<mi::neuraylib::Texture_compression>( get_db_element()->get_compression());
}

} // namespace NEURAY

} // namespace MI
