/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

// Base class for Reader_image_set and Canvas_image_set
class Base_image_set : public DBIMAGE::Image_set
{
public:

    Base_image_set( const mi::IArray* array, const char* selector)
      : m_array( mi::base::make_handle_dup( array)),
        m_selector( selector ? selector : "")
    {
        mi::Size n = array->get_length();
        ASSERT( M_NEURAY_API, n > 0);

        // Collect all frame numbers, and search for non-trivial u/v pairs
        for( mi::Size i = 0; i < n; ++i) {
            mi::base::Handle<const mi::IStructure> uvtile(
                m_array->get_value<const mi::IStructure>( i));
            mi::base::Handle<const mi::ISize> frame(
                uvtile->get_value<mi::ISize>( "frame"));
            auto frame_number = frame->get_value<mi::Size>();
            m_frame_number_to_frame_index[frame_number] = 0;
            if( !m_is_uvtile) {
                mi::base::Handle<const mi::ISint32> iu( uvtile->get_value<mi::ISint32>( "u"));
                mi::base::Handle<const mi::ISint32> iv( uvtile->get_value<mi::ISint32>( "v"));
                mi::Sint32 u = iu->get_value<mi::Sint32>();
                mi::Sint32 v = iv->get_value<mi::Sint32>();
                m_is_uvtile = u != 0 || v != 0;
            }
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
            mi::base::Handle<const mi::IStructure> uvtile(
                m_array->get_value<const mi::IStructure>( i));
            mi::base::Handle<const mi::ISize> frame(
                uvtile->get_value<mi::ISize>( "frame"));
            auto frame_number = frame->get_value<mi::Size>();
            mi::Size frame_id = m_frame_number_to_frame_index.at( frame_number);
            m_global_indices_per_frame_index[frame_id].push_back( i);
        }

        // Flag for animated textures
        m_is_animated = n_frames > 1;
    }

    bool is_mdl_container() const final { return false; }

    const char* get_original_filename() const final { return ""; }

    const char* get_container_filename() const final { return ""; }

    const char* get_mdl_file_path() const final { return ""; }

    const char* get_selector() const final
    { return !m_selector.empty() ? m_selector.c_str() : nullptr; }

    const char* get_image_format() const override { return ""; }

    bool is_animated() const final { return m_is_animated; }

    bool is_uvtile() const final { return m_is_uvtile; }

    mi::Size get_length() const final { return m_frame_index_to_frame_number.size(); }

    mi::Size get_frame_number( mi::Size f) const final
    {
        ASSERT( M_NEURAY_API, f < get_length());
        return m_frame_index_to_frame_number[f];
    }

    mi::Size get_frame_length( mi::Size f) const final
    {
        ASSERT( M_NEURAY_API, f < get_length());
        return m_global_indices_per_frame_index[f].size();
    }

    void get_uvtile_uv( mi::Size f, mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const final
    {
        mi::Size index = get_global_index( f, i);
        mi::base::Handle<const mi::IStructure> uvtile(
            m_array->get_value<const mi::IStructure>( index));
        mi::base::Handle<const mi::ISint32> iu( uvtile->get_value<mi::ISint32>( "u"));
        mi::base::Handle<const mi::ISint32> iv( uvtile->get_value<mi::ISint32>( "v"));
        u = iu->get_value<mi::Sint32>();
        v = iv->get_value<mi::Sint32>();
    }

    const char* get_resolved_filename( mi::Size f, mi::Size i) const final { return ""; }

    const char* get_container_membername( mi::Size f, mi::Size i) const final { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size f, mi::Size i) const override { return nullptr; }

    mi::neuraylib::ICanvas* get_canvas( mi::Size f, mi::Size i) const override { return nullptr; }

protected:

    mi::Size get_global_index( mi::Size f, mi::Size i) const
    {
        ASSERT( M_NEURAY_API, f < get_length());
        const std::vector<mi::Size>& element = m_global_indices_per_frame_index[f];
        ASSERT( M_NEURAY_API, i < element.size());
        ASSERT( M_NEURAY_API, element[i] < m_array->get_length());
        return element[i];
    }

private:

    mi::base::Handle<const mi::IArray> m_array;

    /// Maps frame numbers to frame indices.
    std::map<mi::Size, mi::Size> m_frame_number_to_frame_index;
    /// Maps frame index to frame number.
    std::vector<mi::Size> m_frame_index_to_frame_number;
    /// Stores global indices into m_array per frame index.
    std::vector<std::vector<mi::Size>> m_global_indices_per_frame_index;
    /// Indicates whether this is an animated texture.
    bool m_is_animated = false;
    /// Indicates whether any frame has uvtiles.
    bool m_is_uvtile = false;
    /// The selector.
    std::string m_selector;
};

// Adapts array of "Uvtile_reader" structs to DBIMAGE::Image_set.
class Reader_image_set : public Base_image_set
{
public:

    Reader_image_set( mi::IArray* array, const char* image_format, const char* selector)
      : Base_image_set( array, selector),
        m_readers( mi::base::make_handle_dup( array)),
        m_image_format( image_format)
    { }

    const char* get_image_format() const final { return m_image_format.c_str(); }

    mi::neuraylib::IReader* open_reader( mi::Size f, mi::Size i) const final
    {
        mi::Size index = get_global_index( f, i);
        mi::base::Handle<mi::IStructure> uvtile(
            m_readers->get_value<mi::IStructure>( index));
        mi::base::Handle<mi::neuraylib::IReader> reader(
            uvtile->get_value<mi::neuraylib::IReader>( "reader"));
        return reader.extract();
    }

private:

    mi::base::Handle<mi::IArray> m_readers;
    std::string m_image_format;
};

// Adapts array of "Uvtile" structs to DBIMAGE::Image_set.
class Canvas_image_set : public Base_image_set
{
public:

    Canvas_image_set( mi::IArray* canvases, const char* selector)
      : Base_image_set( canvases, selector),
        m_mutable_canvases( mi::base::make_handle_dup( canvases)),
        m_const_canvases( mi::base::make_handle_dup( canvases))
    { }

    Canvas_image_set( const mi::IArray* canvases, const char* selector)
      : Base_image_set( canvases, selector),
        m_mutable_canvases( {}),
        m_const_canvases( mi::base::make_handle_dup( canvases))
    { }

    mi::neuraylib::ICanvas* get_canvas( mi::Size f, mi::Size i) const final
    {
        if( !m_mutable_canvases) {

            mi::Size index = get_global_index( f, i);
            mi::base::Handle<const mi::IStructure> uvtile(
                m_const_canvases->get_value<const mi::IStructure>( index));
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(
                uvtile->get_value<const mi::neuraylib::ICanvas>( "canvas"));
            if( !canvas)
                return nullptr;

            if( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
                return nullptr;

            SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
            return image_module->copy_canvas( canvas.get());
        }

        mi::Size index = get_global_index( f, i);
        mi::base::Handle<mi::IStructure> uvtile(
            m_mutable_canvases->get_value<mi::IStructure>( index));
        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            uvtile->get_value<mi::neuraylib::ICanvas>( "canvas"));
        if( !canvas)
            return nullptr;

        if( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
            return nullptr;

        return canvas.extract();
    }

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

mi::Sint32 Image_impl::reset_file( const char* filename, const char* selector)
{
    if( !filename)
        return -1;

    auto* db_element = get_db_element();
    if( !db_element)
        return -255;

    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result = db_element->reset_file(
        get_db_transaction(), filename, selector, impl_hash);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Image_impl::reset_reader(
    mi::neuraylib::IReader* reader, const char* image_format, const char* selector)
{
    if( !reader || !image_format)
        return -1;

    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result = get_db_element()->reset_reader(
        get_db_transaction(), reader, image_format, selector, impl_hash);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Image_impl::reset_reader(
    mi::IArray* reader_array, const char* image_format, const char* selector)
{
    if( !reader_array || !image_format)
        return -1;

    mi::Size n = reader_array->get_length();
    if( n == 0)
        return -10;

    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const mi::IStructure> uvtile(
            reader_array->get_value<const mi::IStructure>( i));
        if( !uvtile)
            return -10;
        if( strcmp( uvtile->get_type_name(), "Uvtile_reader") != 0)
            return -10;
    }

    Reader_image_set reader_set( reader_array, image_format, selector);
    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result
        = get_db_element()->reset_image_set( get_db_transaction(), &reader_set, impl_hash);
    if( result == -99)
        result = -10;

    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

bool Image_impl::set_from_canvas( const mi::neuraylib::ICanvas* canvas, const char* selector)
{
    if( !canvas)
        return false;
    if ( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
        return false;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
    canvases[0] = image_module->copy_canvas( canvas);
    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap( canvases));
    mi::base::Uuid impl_hash{0,0,0,0};
    get_db_element()->set_mipmap( get_db_transaction(), mipmap.get(), selector, impl_hash);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::set_from_canvas(
    mi::neuraylib::ICanvas* canvas, const char* selector, bool shared)
{
    if( !shared)
        return set_from_canvas( const_cast<const mi::neuraylib::ICanvas*>( canvas), selector);

    if( !canvas)
        return false;
    if ( IMAGE::convert_pixel_type_string_to_enum( canvas->get_type()) == IMAGE::PT_UNDEF)
        return false;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    std::vector<mi::base::Handle<mi::neuraylib::ICanvas> > canvases( 1);
    canvases[0] = make_handle_dup( canvas);
    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap( canvases));
    mi::base::Uuid impl_hash{0,0,0,0};
    get_db_element()->set_mipmap( get_db_transaction(), mipmap.get(), selector, impl_hash);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::set_from_canvas( const mi::IArray* canvas_array, const char* selector)
{
    if( !canvas_array)
        return false;

    mi::Size n = canvas_array->get_length();
    if( n == 0)
        return false;

    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const mi::IStructure> uvtile(
            canvas_array->get_value<const mi::IStructure>( i));
        if( !uvtile)
            return false;
        if( strcmp( uvtile->get_type_name(), "Uvtile") != 0)
            return false;
    }

    Canvas_image_set canvas_set( canvas_array, selector);
    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result
        = get_db_element()->reset_image_set( get_db_transaction(), &canvas_set, impl_hash);
    if( result != 0)
        return false;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::set_from_canvas( mi::IArray* canvas_array, const char* selector, bool shared)
{
    if( !shared)
        return set_from_canvas( canvas_array, selector);

    if( !canvas_array)
        return false;

    mi::Size n = canvas_array->get_length();
    if( n == 0)
        return false;

    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const mi::IStructure> uvtile(
            canvas_array->get_value<const mi::IStructure>( i));
        if( !uvtile)
            return false;
        if( strcmp( uvtile->get_type_name(), "Uvtile") != 0)
            return false;
    }

    Canvas_image_set canvas_set( canvas_array, selector);
    mi::base::Uuid impl_hash{0,0,0,0};
    mi::Sint32 result
        = get_db_element()->reset_image_set( get_db_transaction(), &canvas_set, impl_hash);
    if( result != 0)
        return false;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return true;
}

bool Image_impl::is_animated() const
{
    return get_db_element()->is_animated();
}

mi::Size Image_impl::get_length() const
{
    return get_db_element()->get_length();
}

mi::Size Image_impl::get_frame_number( mi::Size frame_id) const
{
    return get_db_element()->get_frame_number( frame_id);
}

mi::Size Image_impl::get_frame_id( mi::Size frame_number) const
{
    return get_db_element()->get_frame_id( frame_number);
}

bool Image_impl::is_uvtile() const
{
    return get_db_element()->is_uvtile();
}

mi::Size Image_impl::get_frame_length( mi::Size frame_id) const
{
    return get_db_element()->get_frame_length( frame_id);
}

mi::Sint32 Image_impl::get_uvtile_uv(
    mi::Size frame_id, Size uvtile_id, Sint32& u, Sint32& v) const
{
    return get_db_element()->get_uvtile_uv( frame_id, uvtile_id, u, v);
}

mi::Size Image_impl::get_uvtile_id(  mi::Size frame_id, Sint32 u, Sint32 v) const
{
    return get_db_element()->get_uvtile_id( frame_id, u, v);
}

void Image_impl::get_uvtile_uv_ranges(
    mi::Size frame_id,
    mi::Sint32& min_u,
    mi::Sint32& min_v,
    mi::Sint32& max_u,
    mi::Sint32& max_v) const
{
    get_db_element()->get_uvtile_uv_ranges( frame_id, min_u, min_v, max_u, max_v);
}

const char* Image_impl::get_original_filename() const
{
    const std::string& filename = get_db_element()->get_original_filename();
    return filename.empty() ? nullptr : filename.c_str();
}

const char* Image_impl::get_selector() const
{
    m_cached_selector = get_db_element()->get_selector();
    return !m_cached_selector.empty() ? m_cached_selector.c_str() : nullptr;
}

const char* Image_impl::get_filename( mi::Size frame_id, mi::Size uvtile_id) const
{
    const DBIMAGE::Image *image = get_db_element();
    const std::string& filename = image->is_file_based()
        ? image->get_filename( frame_id, uvtile_id)
        : image->get_resolved_container_membername( frame_id, uvtile_id);

    return filename.empty() ? nullptr : filename.c_str();
}

const mi::neuraylib::ICanvas* Image_impl::get_canvas(
    mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), frame_id, uvtile_id));
    if( !mipmap)
        return nullptr;
    return mipmap->get_level( level);
}

const char* Image_impl::get_type( mi::Size frame_id, mi::Size uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), frame_id, uvtile_id));
    if( !mipmap)
        return nullptr;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
    return canvas->get_type();
}

mi::Uint32 Image_impl::get_levels( mi::Size frame_id, mi::Size uvtile_id) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), frame_id, uvtile_id));
    if( !mipmap)
        return ~0;
    return mipmap->get_nlevels();
}

mi::Uint32 Image_impl::resolution_x(
    mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), frame_id, uvtile_id));
    if( !mipmap)
        return ~0;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( level));
    return canvas->get_resolution_x();
}

mi::Uint32 Image_impl::resolution_y(
    mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), frame_id, uvtile_id));
    if( !mipmap)
        return ~0;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( level));
    return canvas->get_resolution_y();
}

mi::Uint32 Image_impl::resolution_z(
    mi::Size frame_id, mi::Size uvtile_id, mi::Uint32 level) const
{
    mi::base::Handle<const IMAGE::IMipmap> mipmap(
        get_db_element()->get_mipmap( get_db_transaction(), frame_id, uvtile_id));
    if( !mipmap)
        return ~0;
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( level));
    return canvas->get_layers_size();
}

} // namespace NEURAY

} // namespace MI
