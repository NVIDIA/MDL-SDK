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
 ** \brief Source for the IImage implementation.
 **/

#include "pch.h"

#include "neuray_image_impl.h"

#include "neuray_transaction_impl.h"
#include "neuray_type_utilities.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/ireader.h>

#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/i_image_utilities.h>


namespace MI {

namespace NEURAY {

class Reader_image_set : public DBIMAGE::Image_set
{
public:

    Reader_image_set( mi::IArray* readers, const char* image_format)
      : m_readers( readers), m_image_format( image_format)
    {
        ASSERT( M_NEURAY_API, readers && readers->get_length() > 0);
    }

    mi::Size get_length() const { return m_readers->get_length(); }

    bool is_uvtile() const { return get_length() > 1; }

    bool is_mdl_container() const { return false; }

    void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT( M_SCENE, i <= get_length());

        u = 0;
        v = 0;

        mi::base::Handle<const mi::IStructure> uvtile( m_readers->get_value<const mi::IStructure>( i));
        if( !uvtile)
            return;

        mi::base::Handle< const mi::ISint32> iu( uvtile->get_value<mi::ISint32>( "u"));
        if( !iu)
            return;

        mi::base::Handle< const mi::ISint32> iv( uvtile->get_value<mi::ISint32>( "v"));
        if( !iv)
            return;

        iu->get_value( u);
        iv->get_value( v);
    }

    const char* get_original_filename() const { return ""; }

    const char* get_container_filename() const { return ""; }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_resolved_filename( mi::Size i) const { return ""; }

    const char* get_container_membername( mi::Size i) const { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size i) const
    {
        ASSERT( M_SCENE, i <= get_length());

        mi::base::Handle< mi::IStructure> uvtile( m_readers->get_value< mi::IStructure>( i));
        if( !uvtile)
            return nullptr;

        mi::base::Handle< mi::neuraylib::IReader> reader(
            uvtile->get_value< mi::neuraylib::IReader>( "reader"));
        if( !reader)
            return nullptr;

        reader->retain();
        return reader.get();
    }

    mi::neuraylib::ICanvas* get_canvas( mi::Size i) const { return nullptr; }

    const char* get_image_format() const { return m_image_format.c_str(); }

private:

    mi::base::Handle<mi::IArray> m_readers;
    std::string m_image_format;
};

class Canvas_image_set : public DBIMAGE::Image_set
{
public:

    Canvas_image_set( mi::IArray* canvases)
      : m_mutable_canvases( mi::base::make_handle_dup( canvases)),
        m_const_canvases( mi::base::make_handle_dup( canvases))
    {
        ASSERT( M_NEURAY_API, canvases && canvases->get_length() > 0);
    }

    Canvas_image_set( const mi::IArray* canvases)
      : m_const_canvases( mi::base::make_handle_dup( canvases))
    {
        ASSERT( M_NEURAY_API, canvases && canvases->get_length() > 0);
    }

    mi::Size get_length() const { return m_const_canvases->get_length(); }

    bool is_uvtile() const { return get_length() > 1; }

    bool is_mdl_container() const { return false; }

    void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        ASSERT( M_SCENE, i <= get_length());

        u = 0;
        v = 0;

        mi::base::Handle<const mi::IStructure> uvtile( m_const_canvases->get_value<const mi::IStructure>( i));
        if( !uvtile)
            return;

        mi::base::Handle< const mi::ISint32> iu( uvtile->get_value<mi::ISint32>( "u"));
        if( !iu)
            return;

        mi::base::Handle< const mi::ISint32> iv( uvtile->get_value<mi::ISint32>( "v"));
        if( !iv)
            return;

        iu->get_value( u);
        iv->get_value( v);
    }

    const char* get_original_filename() const { return ""; }

    const char* get_container_filename() const { return ""; }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_resolved_filename( mi::Size i) const { return ""; }

    const char* get_container_membername( mi::Size i) const { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size i) const { return nullptr; }

    mi::neuraylib::ICanvas* get_canvas( mi::Size i) const
    {
        ASSERT( M_SCENE, i <= get_length());

        if( !m_mutable_canvases) {

            mi::base::Handle< const mi::IStructure> uvtile(
                m_const_canvases->get_value< const mi::IStructure>( i));
            if( !uvtile)
                return nullptr;

            mi::base::Handle< const mi::neuraylib::ICanvas> canvas(
                uvtile->get_value< const mi::neuraylib::ICanvas>( "canvas"));
            if( !canvas)
                return nullptr;

            if( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
                return nullptr;

            SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
            return image_module->copy_canvas( canvas.get());
        }

        mi::base::Handle< mi::IStructure> uvtile( m_mutable_canvases->get_value< mi::IStructure>( i));
        if( !uvtile)
            return nullptr;

        mi::base::Handle< mi::neuraylib::ICanvas> canvas(
            uvtile->get_value< mi::neuraylib::ICanvas>( "canvas"));
        if( !canvas)
            return nullptr;

        if( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
            return nullptr;

        canvas->retain();
        return canvas.get();
    }

    const char* get_image_format() const { return ""; }

private:

    mi::base::Handle<mi::IArray> m_mutable_canvases;
    mi::base::Handle<const mi::IArray> m_const_canvases;
};

DB::Element_base* Image_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new DBIMAGE::Image;
}

mi::base::IInterface* Image_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return ( new Image_impl())->cast_to_major();
}

mi::neuraylib::Element_type Image_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_IMAGE;
}

mi::Sint32 Image_impl::reset_file( const char* filename)
{
    if( !filename)
        return -1;

    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result = get_db_element()->reset_file( get_db_transaction(), filename, impl_hash);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Image_impl::reset_reader( mi::neuraylib::IReader* reader, const char* image_format)
{
    if( !reader || !image_format)
        return -1;

    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result
        = get_db_element()->reset_reader( get_db_transaction(), reader, image_format, impl_hash);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Image_impl::reset_reader( mi::IArray* reader_array, const char* image_format)
{
    if( !reader_array || !image_format || reader_array->get_length() == 0)
        return -1;

    Reader_image_set reader_set( reader_array, image_format);
    mi::base::Uuid impl_hash{0,0,0,0};
    if( get_db_element()->reset_image_set( get_db_transaction(), &reader_set, impl_hash) < 0)
        return -3;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return 0;
}

const char* Image_impl::get_filename( mi::Uint32 uvtile_id) const
{
    const DBIMAGE::Image *image = get_db_element();
    const std::string& filename = image->is_file_based()
        ? image->get_filename( uvtile_id)
        : image->get_resolved_container_membername( uvtile_id);

    return filename.empty() ? nullptr : filename.c_str();
}

const char* Image_impl::get_original_filename() const
{
    const std::string& filename = get_db_element()->get_original_filename();
    return filename.empty() ? nullptr : filename.c_str();
}

bool Image_impl::set_from_canvas( const mi::neuraylib::ICanvas* canvas)
{
    if( !canvas)
        return false;
    if ( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
        return false;

    SYSTEM::Access_module<IMAGE::Image_module> m_image_module( false);
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
    canvases[0] = m_image_module->copy_canvas( canvas);
    mi::base::Handle<IMAGE::IMipmap> mipmap( m_image_module->create_mipmap( canvases));
    mi::base::Uuid impl_hash{0,0,0,0};
    get_db_element()->set_mipmap( get_db_transaction(), mipmap.get(), impl_hash);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::set_from_canvas( mi::neuraylib::ICanvas* canvas, bool shared)
{
    if( !shared)
        return set_from_canvas( const_cast<const mi::neuraylib::ICanvas*>( canvas));

    if( !canvas)
        return false;
    if ( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
        return false;

    SYSTEM::Access_module<IMAGE::Image_module> m_image_module( false);
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
    canvases[0] = make_handle_dup( canvas);
    mi::base::Handle<IMAGE::IMipmap> mipmap( m_image_module->create_mipmap( canvases));
    mi::base::Uuid impl_hash{0,0,0,0};
    get_db_element()->set_mipmap( get_db_transaction(), mipmap.get(), impl_hash);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::set_from_canvas( const mi::IArray* canvas_array)
{
    if( !canvas_array || canvas_array->get_length() == 0)
        return false;

    Canvas_image_set canvas_set( canvas_array);
    mi::base::Uuid impl_hash{0,0,0,0};
    if( get_db_element()->reset_image_set( get_db_transaction(), &canvas_set, impl_hash) != 0)
        return false;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::set_from_canvas( mi::IArray* canvas_array, bool shared)
{
    if( !canvas_array)
        return false;

    if( !shared)
        return set_from_canvas( canvas_array);

    Canvas_image_set canvas_set( canvas_array);
    mi::base::Uuid impl_hash{0,0,0,0};
    if( get_db_element()->reset_image_set( get_db_transaction(), &canvas_set, impl_hash) != 0)
        return false;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

const mi::neuraylib::ICanvas* Image_impl::get_canvas(
    mi::Uint32 level, mi::Uint32 uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), uvtile_id));
    if( !mipmap)
        return nullptr;
    return mipmap->get_level( level);
}

const char* Image_impl::get_type( mi::Uint32 uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), uvtile_id));
    if( !mipmap)
        return nullptr;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    return canvas->get_type();
}

mi::Uint32 Image_impl::get_levels( mi::Uint32 uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), uvtile_id));
    if( !mipmap)
        return ~0;
    return mipmap->get_nlevels();
}

mi::Uint32 Image_impl::resolution_x( Uint32 level, mi::Uint32 uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), uvtile_id));
    if( !mipmap)
        return ~0;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    return canvas->get_resolution_x();
}

mi::Uint32 Image_impl::resolution_y( Uint32 level, mi::Uint32 uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), uvtile_id));
    if( !mipmap)
        return ~0;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    return canvas->get_resolution_y();
}

mi::Uint32 Image_impl::resolution_z( Uint32 level, mi::Uint32 uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), uvtile_id));
    if( !mipmap)
        return ~0;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    return canvas->get_layers_size();
}

bool Image_impl::is_uvtile() const
{
    return get_db_element()->is_uvtile();
}

void Image_impl::get_uvtile_uv_ranges(
    mi::Sint32& min_u, mi::Sint32& min_v, mi::Sint32& max_u, mi::Sint32& max_v) const
{
    get_db_element()->get_uvtile_uv_ranges( min_u, min_v, max_u, max_v);
}

mi::Size Image_impl::get_uvtile_length() const
{
    return get_db_element()->get_uvtile_length();
}

mi::Sint32 Image_impl::get_uvtile_uv( Uint32 uvtile_id, Sint32& u, Sint32& v) const
{
    return get_db_element()->get_uvtile_uv( uvtile_id, u, v);
}

mi::Uint32 Image_impl::get_uvtile_id( Sint32 u, Sint32 v) const
{
    return get_db_element()->get_uvtile_id( u, v);
}

} // namespace NEURAY

} // namespace MI
