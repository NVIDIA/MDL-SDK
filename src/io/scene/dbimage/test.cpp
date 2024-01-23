/******************************************************************************
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
 *****************************************************************************/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/scene/dbimage"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <vector>
#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/itile.h>

#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/config/config.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/plug/i_plug.h>
#include <base/lib/path/i_path.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_transaction.h>

#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/test_shared.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h>

#include <prod/lib/neuray/test_shared.h> // for plugin_path_openimageio
#include <io/scene/mdl_elements/test_shared.h>

using namespace MI;

SYSTEM::Access_module<IMAGE::Image_module> g_image_module;

// Indicates whether \p hackstack ends in \p needle.
bool ends_with( const char* haystack, const char* needle)
{
    size_t h = strlen( haystack);
    size_t n = strlen( needle);
    return h >= n && strcmp( haystack+h-n, needle) == 0;
}

// Implementation of Image_set that puts the canvases on the u-v diagonal.
class Test_canvas_image_set : public DBIMAGE::Image_set
{
public:
    Test_canvas_image_set(
        const std::vector<mi::base::Handle<const mi::neuraylib::ICanvas>>& canvases)
      : m_canvases( canvases) { }

    bool is_mdl_container() const { return false; }

    const char* get_original_filename() const { return ""; }

    const char* get_container_filename() const { return ""; }

    const char* get_mdl_file_path() const { return ""; }

    const char* get_selector() const { return nullptr; }

    const char* get_image_format() const { return ""; }

    bool is_animated() const { return false; }

    bool is_uvtile() const { return true; }

    mi::Size get_length() const { return 1; }

    mi::Size get_frame_number( mi::Size f) const { return 0; }

    mi::Size get_frame_length( mi::Size f) const { return m_canvases.size(); }

    void get_uvtile_uv( mi::Size f, mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
    {
        if( f > 0 || i >= m_canvases.size())
            return;

        // Put tiles on the diagonal (arbitrary choice)
        u = static_cast<mi::Sint32>( i);
        v = static_cast<mi::Sint32>( i);
    }

    const char* get_resolved_filename( mi::Size f, mi::Size i) const { return ""; }

    const char* get_container_membername( mi::Size f, mi::Size i) const { return ""; }

    mi::neuraylib::IReader* open_reader( mi::Size f, mi::Size i) const { return nullptr; }

    mi::neuraylib::ICanvas* get_canvas( mi::Size f, mi::Size i) const
    {
        if( i >= m_canvases.size())
            return nullptr;

        return g_image_module->copy_canvas( m_canvases[i].get());
    }

private:
    const std::vector<mi::base::Handle<const mi::neuraylib::ICanvas>>& m_canvases;
};


// Checks whether \p image represents $MI_DATA/io/image/image/test_mipmap.png.
//
// Checks number of miplevels and compares miplevel 2.
void check_mipmap( DB::Transaction* transaction, const DBIMAGE::Image* image)
{
    MI_CHECK_EQUAL( image->get_selector(), std::string());

    mi::base::Handle<const IMAGE::IMipmap> mipmap( image->get_mipmap( transaction, 0, 0));
    MI_CHECK_EQUAL( 7, mipmap->get_nlevels());

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 2));
    bool result = g_image_module->export_canvas( canvas.get(), "mipmap_level_2.png");
    MI_CHECK( result);

    std::string root_path = TEST::mi_src_path( "io/image/image/tests/");
    MI_CHECK_IMG_DIFF(
        "mipmap_level_2.png", (root_path + "reference/export_of_test_mipmap_level_2.png").c_str());
}

// Checks the uvtiles of frame \p frame_id against \p expected_tiles (indexed by filename).
void check_uvtile(
    const DBIMAGE::Image* image,
    mi::Size frame_id,
    const std::map<std::string, std::pair<mi::Sint32, mi::Sint32>>& expected_tiles)
{
    mi::Size n_uvtiles = image->get_frame_length( frame_id);
    MI_CHECK_EQUAL( expected_tiles.size(), n_uvtiles);

    mi::Sint32 min_u = std::numeric_limits<mi::Sint32>::max();
    mi::Sint32 min_v = std::numeric_limits<mi::Sint32>::max();
    mi::Sint32 max_u = std::numeric_limits<mi::Sint32>::min();
    mi::Sint32 max_v = std::numeric_limits<mi::Sint32>::min();

    for( mi::Size i = 0; i < n_uvtiles; ++i) {

        mi::Sint32 u, v;
        MI_CHECK_EQUAL( image->get_uvtile_uv( frame_id, i, u, v), 0);

        std::string fn = HAL::Ospath::basename( image->get_filename( frame_id, i));
        const std::pair<mi::Sint32, mi::Sint32>& uvs = expected_tiles.at( fn);
        MI_CHECK_EQUAL( u, uvs.first);
        MI_CHECK_EQUAL( v, uvs.second);

        MI_CHECK_EQUAL( image->get_uvtile_id( frame_id, u, v), i);

        min_u = std::min( min_u, u);
        min_v = std::min( min_v, v);
        max_u = std::max( max_u, u);
        max_v = std::max( max_v, v);
    }

    mi::Sint32 min_u2 = 0, min_v2 = 0, max_u2 = 0, max_v2 = 0;
    image->get_uvtile_uv_ranges( frame_id, min_u2, min_v2, max_u2, max_v2);
    MI_CHECK_EQUAL( min_u2, min_u);
    MI_CHECK_EQUAL( min_v2, min_v);
    MI_CHECK_EQUAL( max_u2, max_u);
    MI_CHECK_EQUAL( max_v2, max_v);
}

// Checks the uvtiles of frame \p frame_id against \p uvs of length \p n.
void check_uvtile(
    const DBIMAGE::Image* image, mi::Size frame_id, mi::Size n, mi::Sint32 uvs[][2])
{
    mi::Size n_uvtiles = image->get_frame_length( frame_id);
    MI_CHECK_EQUAL( n, n_uvtiles);

    mi::Sint32 min_u = std::numeric_limits<mi::Sint32>::max();
    mi::Sint32 min_v = std::numeric_limits<mi::Sint32>::max();
    mi::Sint32 max_u = std::numeric_limits<mi::Sint32>::min();
    mi::Sint32 max_v = std::numeric_limits<mi::Sint32>::min();

    for( mi::Size i = 0; i < n_uvtiles; ++i) {
        mi::Sint32 u, v;
        MI_CHECK_EQUAL( image->get_uvtile_uv( frame_id, i, u, v), 0);
        MI_CHECK_EQUAL( u, uvs[i][0]);
        MI_CHECK_EQUAL( v, uvs[i][1]);
        MI_CHECK_EQUAL( image->get_uvtile_id( frame_id, u, v), i);

        min_u = std::min( min_u, u);
        min_v = std::min( min_v, v);
        max_u = std::max( max_u, u);
        max_v = std::max( max_v, v);
    }

    mi::Sint32 min_u2 = 0, min_v2 = 0, max_u2 = 0, max_v2 = 0;
    image->get_uvtile_uv_ranges( frame_id, min_u2, min_v2, max_u2, max_v2);
    MI_CHECK_EQUAL( min_u2, min_u);
    MI_CHECK_EQUAL( min_v2, min_v);
    MI_CHECK_EQUAL( max_u2, max_u);
    MI_CHECK_EQUAL( max_v2, max_v);
}

// Checks that the image holds the pink dummy mipmap.
void check_default_pink_dummy_mipmap( DB::Transaction* transaction, const DBIMAGE::Image* image)
{
    MI_CHECK( image->get_length() <= 1);
    MI_CHECK( image->get_frame_length( 0) <= 1);

    mi::base::Handle<const IMAGE::IMipmap> mipmap( image->get_mipmap( transaction, 0, 0));
    MI_CHECK( g_image_module->is_dummy_mipmap( mipmap.get()));
}

// Used by check sharing().
DB::Tag load_image(
    DB::Transaction* transaction,
    const std::string& filename,
    const mi::base::Uuid& impl_hash,
    bool shared_proxy)
{
    std::string db_name = shared_proxy ? "MI_default_" : "";
    db_name += "image_" + filename;
    if( !shared_proxy)
        db_name = MDL::DETAIL::generate_unique_db_name( transaction, db_name.c_str());

    DB::Tag tag = transaction->name_to_tag( db_name.c_str());
    if( tag)
        return tag;

    DBIMAGE::Image* image = new DBIMAGE::Image();
    mi::Sint32 result = image->reset_file( transaction, filename, /*selector*/ nullptr, impl_hash);
    if( result != 0)
        return DB::Tag();

    MI_CHECK_EQUAL( image->get_selector(), std::string());

    return transaction->store_for_reference_counting(
        image, db_name.c_str(), transaction->get_scope()->get_level());
}

// Used by check sharing().
DB::Tag get_impl_tag( DB::Transaction* transaction, DB::Tag proxy_tag)
{
    DB::Access<DBIMAGE::Image> proxy( proxy_tag, transaction);
    return proxy->get_impl_tag();
}

// Creates a file-base image with a single frame/uvtile and performs various modifications.
void check_simple_creation( DB::Transaction* transaction)
{
    DB::Tag tag;
    std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_mipmap.png");
    mi::base::Uuid unknown_hash{0,0,0,0};
    mi::Sint32 result;

    // File-based
    {
        // Create a file-based image as DB element
        DBIMAGE::Image* image = new DBIMAGE::Image();
        result = image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, 0);
        image->dump();
        check_mipmap( transaction, image);
        tag = transaction->store( image);
    }
    {
        // Access the DB element
        DB::Access<DBIMAGE::Image> image( tag, transaction);
        check_mipmap( transaction, image.get_ptr());
        MI_CHECK( !image->get_filename( 0, 0).empty());
    }
    {
        // Edit the DB element
        DB::Edit<DBIMAGE::Image> image( tag, transaction);
        check_default_pink_dummy_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
        result = image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, 0);
        image->dump();
        check_mipmap( transaction, image.get_ptr());
    }

    // Memory-based (mipmap)
    {
        // Create a memory-based image as DB element
        DB::Access<DBIMAGE::Image> image( tag, transaction);
        mi::base::Handle<const IMAGE::IMipmap> mipmap( image->get_mipmap( transaction, 0, 0));
        mi::base::Handle<IMAGE::IMipmap> mipmap_copy( g_image_module->copy_mipmap( mipmap.get()));
        DBIMAGE::Image* image_copy = new DBIMAGE::Image();
        image_copy->set_mipmap( transaction, mipmap_copy.get(), /*selector*/ nullptr, unknown_hash);
        image_copy->dump();
        check_mipmap( transaction, image_copy);
        tag = transaction->store( image_copy);
    }
    {
        // Access the DB element
        DB::Access<DBIMAGE::Image> image( tag, transaction);
        check_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
    }
    {
        // Edit the DB element
        DB::Edit<DBIMAGE::Image> image( tag, transaction);
        check_default_pink_dummy_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
        result = image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, 0);
        image->dump();
        check_mipmap( transaction, image.get_ptr());
    }

    // Memory-based (reader)
    {
        // Create a memory-based image from a reader as DB element
        DISK::File_reader_impl reader;
        bool success = reader.open( file_path.c_str());
        MI_CHECK( success);
        DBIMAGE::Image* image = new DBIMAGE::Image();
        result = image->reset_reader(
            transaction, &reader, "png", /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, 0);
        image->dump();
        check_mipmap( transaction, image);
        tag = transaction->store( image);
    }
    {
        // Access the DB element
        DB::Access<DBIMAGE::Image> image( tag, transaction);
        check_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
    }
    {
        // Edit the DB element
        DB::Edit<DBIMAGE::Image> image( tag, transaction);
        check_default_pink_dummy_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
        DISK::File_reader_impl reader;
        bool success = reader.open( file_path.c_str());
        MI_CHECK( success);
        result = image->reset_reader(
            transaction, &reader, "png", /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, 0);
        image->dump();
        check_mipmap( transaction, image.get_ptr());
    }

    // Reset
    {
        // Edit the DB element, implicitly set to default state (not valid)
        DB::Edit<DBIMAGE::Image> image( tag, transaction);
        image->dump();
        check_default_pink_dummy_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
    }

    // Default-constructed
    {
        // Create a default-constructed image DB element
        DBIMAGE::Image* image = new DBIMAGE::Image();
        check_default_pink_dummy_mipmap( transaction, image);
        tag = transaction->store( image);
    }
    {
        // Access the DB element
        DB::Access<DBIMAGE::Image> image( tag, transaction);
        check_default_pink_dummy_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
    }
    {
        // Edit the DB element
        DB::Edit<DBIMAGE::Image> image( tag, transaction);
        check_default_pink_dummy_mipmap( transaction, image.get_ptr());
        MI_CHECK( image->get_filename( 0, 0).empty());
    }

    // Failures (-5 difficult to test)
    {
        // No image plugin found to handle the file.
        std::string file_path = TEST::mi_src_path( "io/scene/mdl_elements/resources/test.ies");
        auto image = std::make_unique<DBIMAGE::Image>();
        result = image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, -3);
    }
    {
        // File does not exist.
        std::string file_path = TEST::mi_src_path(
            "io/scene/mdl_elements/resources/test_not_existing.png");
        auto image = std::make_unique<DBIMAGE::Image>();
        result = image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        MI_CHECK_EQUAL( result, -4);
    }
    {
        // Plugin failure (due to invalid selector).
        std::string file_path = TEST::mi_src_path( "io/scene/mdl_elements/resources/test.png");
        auto image = std::make_unique<DBIMAGE::Image>();
        result = image->reset_file( transaction, file_path, "X", unknown_hash);
        MI_CHECK_EQUAL( result, -7);
    }
    {
        // Failure to apply the given selector (the DDS plugin does not support selectors natively,
        // generic RGBA selectors are applied afterwards).
        std::string file_path = TEST::mi_src_path( "io/scene/mdl_elements/resources/test.dds");
        auto image = std::make_unique<DBIMAGE::Image>();
        result = image->reset_file( transaction, file_path, "X", unknown_hash);
        MI_CHECK_EQUAL( result, -10);
    }
}

// Prototypes of internal functions of dbimage.cpp being tested here.
namespace MI {
namespace DBIMAGE {

enum Uvtile_mode
{
    MODE_OFF,
    MODE_UDIM,
    MODE_UVTILE0,
    MODE_UVTILE1
};

size_t get_frame_marker_length( const char* s);

std::string get_regex(
    const std::string& mask,
    size_t& mode_index,
    size_t& frames_index,
    Uvtile_mode& mode,
    size_t& frames_max_digits);

void parse_u_v( Uvtile_mode mode, const char* str, mi::Sint32& u, mi::Sint32& v);

} // namespace
} // namespace

// Checks DBIMAGE::find_frame().
void check_get_frame_marker_length( const char* s, size_t result)
{
    size_t result2 = DBIMAGE::get_frame_marker_length( s);
    MI_CHECK_EQUAL( result2, result);
}

// Checks DBIMAGE::get_regex().
void check_get_regex(
    const std::string& s,
    size_t mi,
    size_t fi,
    DBIMAGE::Uvtile_mode m,
    size_t fmd,
    std::string result)
{
    DBIMAGE::Uvtile_mode m2 = DBIMAGE::MODE_OFF;
    size_t mi2 = 3, fi2 = 3, fmd2 = 3;
    std::string result2 = DBIMAGE::get_regex( s, mi2, fi2, m2, fmd2);
    MI_CHECK_EQUAL( result2, result);

    // Note that the output arguments are unspecified if result2 is empty. We still check against
    // the actual implementation here.
    MI_CHECK_EQUAL( mi2, mi);
    MI_CHECK_EQUAL( fi2, fi);
    MI_CHECK_EQUAL( (int)m2, (int)m);
    MI_CHECK_EQUAL( fmd2, fmd);
}

// Checks DBIMAGE::get_parse_u_v().
void check_parse_u_v( DBIMAGE::Uvtile_mode mode, const char* str, mi::Sint32 u, mi::Sint32 v)
{
    mi::Sint32 u2 = 42, v2 = 42;
    DBIMAGE::parse_u_v( mode, str, u2, v2);
    MI_CHECK_EQUAL( u2, u);
    MI_CHECK_EQUAL( v2, v);
}

// Checks DBIMAGE::get_frame_marker_length(), get_regex() and parse_u_v().
void check_mask_handling()
{
    check_get_frame_marker_length( "<>", 0);
    check_get_frame_marker_length( "<#>", 1);
    check_get_frame_marker_length( "<#><##>", 1);
    check_get_frame_marker_length( "<#####>", 5);
    check_get_frame_marker_length( "<", 0);
    check_get_frame_marker_length( "<a", 0);
    check_get_frame_marker_length( "<<#>", 0);
    check_get_frame_marker_length( "<#<#>", 0);

    const std::string udim   = "([1-9][0-9][0-9][0-9])";
    const std::string uvtile = "(_u-?[0-9]+_v-?[0-9]+)";
    const std::string frame  = "([0-9]+)";

    check_get_regex( "",            0, 0, DBIMAGE::MODE_OFF,     0, "");
    check_get_regex( "ab",          0, 0, DBIMAGE::MODE_OFF,     0, "ab");

    check_get_regex( "a<UDIM>b",    1, 0, DBIMAGE::MODE_UDIM,    0, "a" + udim   + "b");
    check_get_regex( "a<UVTILE0>b", 1, 0, DBIMAGE::MODE_UVTILE0, 0, "a" + uvtile + "b");
    check_get_regex( "a<UVTILE1>b", 1, 0, DBIMAGE::MODE_UVTILE1, 0, "a" + uvtile + "b");

    check_get_regex( "a<UDIM>b<UVTILE0>",    1, 0, DBIMAGE::MODE_UDIM,    0, "");
    check_get_regex( "a<UVTILE0>b<UVTILE1>", 1, 0, DBIMAGE::MODE_UVTILE0, 0, "");
    check_get_regex( "a<UVTILE1>b<UDIM>",    1, 0, DBIMAGE::MODE_UVTILE1, 0, "");

    check_get_regex( "a<#>b",          0, 1, DBIMAGE::MODE_OFF,     1, "a" + frame  + "b");
    check_get_regex( "a<<#>b",         0, 1, DBIMAGE::MODE_OFF,     1, "a<" + frame  + "b");
    check_get_regex( "a<#<#>b",        0, 1, DBIMAGE::MODE_OFF,     1, "a<#" + frame  + "b");

    check_get_regex( "a<#><#>b",       0, 1, DBIMAGE::MODE_OFF,     1, "");

    check_get_regex( "a<UDIM><#>b",    1, 2, DBIMAGE::MODE_UDIM,    1, "a" + udim   + frame + "b");
    check_get_regex( "a<UVTILE0><#>b", 1, 2, DBIMAGE::MODE_UVTILE0, 1, "a" + uvtile + frame + "b");
    check_get_regex( "a<UVTILE1><#>b", 1, 2, DBIMAGE::MODE_UVTILE1, 1, "a" + uvtile + frame + "b");

    check_get_regex( "a<#><UDIM>b",    2, 1, DBIMAGE::MODE_UDIM,    1, "a" + frame + udim   + "b");
    check_get_regex( "a<#><UVTILE0>b", 2, 1, DBIMAGE::MODE_UVTILE0, 1, "a" + frame + uvtile + "b");
    check_get_regex( "a<#><UVTILE1>b", 2, 1, DBIMAGE::MODE_UVTILE1, 1, "a" + frame + uvtile + "b");

    check_get_regex(
        "a(1701)<UDIM>b",   1, 0, DBIMAGE::MODE_UDIM,    0, "a\\(1701\\)" + udim   + "b");
    check_get_regex(
        "a((1701))<UDIM>b", 1, 0, DBIMAGE::MODE_UDIM,    0, "a\\(\\(1701\\)\\)" + udim   + "b");

    check_parse_u_v( DBIMAGE::MODE_UDIM, "1001", 0, 0);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1009", 8, 0);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1010", 9, 0);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1011", 0, 1);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1019", 8, 1);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1020", 9, 1);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1091", 0, 9);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1099", 8, 9);
    check_parse_u_v( DBIMAGE::MODE_UDIM, "1100", 9, 9);

    check_parse_u_v( DBIMAGE::MODE_UVTILE0, "_u0001_v0" ,   1,   0);
    check_parse_u_v( DBIMAGE::MODE_UVTILE0, "_u-20_v-10", -20, -10);
    check_parse_u_v( DBIMAGE::MODE_UVTILE0, "_u-2_v-1"  ,  -2,  -1);
    check_parse_u_v( DBIMAGE::MODE_UVTILE0, "_u0_v0"    ,   0,   0);

    check_parse_u_v( DBIMAGE::MODE_UVTILE1, "_u0001_v0" ,   0,  -1);
    check_parse_u_v( DBIMAGE::MODE_UVTILE1, "_u-20_v-10", -21, -11);
    check_parse_u_v( DBIMAGE::MODE_UVTILE1, "_u-2_v-1"  ,  -3,  -2);
    check_parse_u_v( DBIMAGE::MODE_UVTILE1, "_u0_v0"    ,  -1,  -1);
}

void check_animated_textures( DB::Transaction* transaction)
{
    mi::base::Uuid unknown_hash{0,0,0,0};

    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_frame_<###>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        MI_CHECK_EQUAL(   3, image2->get_length());
        MI_CHECK_EQUAL(   1, image2->get_frame_number( 0));
        MI_CHECK_EQUAL(   5, image2->get_frame_number( 1));
        MI_CHECK_EQUAL( 999, image2->get_frame_number( 2));
        MI_CHECK_EQUAL(   1, image2->get_frame_length( 0));
        MI_CHECK_EQUAL(   1, image2->get_frame_length( 1));
        MI_CHECK_EQUAL(   1, image2->get_frame_length( 2));
        MI_CHECK( ends_with( image2->get_filename( 0, 0).c_str(), "test_frame_1.png"));
        MI_CHECK( ends_with( image2->get_filename( 1, 0).c_str(), "test_frame_05.png"));
        MI_CHECK( ends_with( image2->get_filename( 2, 0).c_str(), "test_frame_999.png"));
    }

    // only two digits
    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_frame_<##>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        MI_CHECK_EQUAL(   2, image2->get_length());
        MI_CHECK_EQUAL(   1, image2->get_frame_number( 0));
        MI_CHECK_EQUAL(   5, image2->get_frame_number( 1));
    }

    // only one digit
    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_frame_<#>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        MI_CHECK_EQUAL(   1, image2->get_length());
        MI_CHECK_EQUAL(   1, image2->get_frame_number( 0));
    }
}

void check_uvtiles( DB::Transaction* transaction)
{
    mi::base::Uuid unknown_hash{0,0,0,0};

    // UVTILE0
    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_uvtile<UVTILE0>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, "R", unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles;
        expected_tiles["test_uvtile_u0001_v0.png" ] = std::make_pair(   1,   0);
        expected_tiles["test_uvtile_u-20_v-10.png"] = std::make_pair( -20, -10);
        expected_tiles["test_uvtile_u-2_v-1.png"  ] = std::make_pair(  -2,  -1);
        expected_tiles["test_uvtile_u0_v0.png"    ] = std::make_pair(   0,   0);

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        check_uvtile( image2.get_ptr(), 0, expected_tiles);
    }

    // UVTILE1
    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_uvtile<UVTILE1>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, "G", unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles;
        expected_tiles["test_uvtile_u0001_v0.png" ] = std::make_pair(   0,  -1);
        expected_tiles["test_uvtile_u-20_v-10.png"] = std::make_pair( -21, -11);
        expected_tiles["test_uvtile_u-2_v-1.png"  ] = std::make_pair(  -3,  -2);
        expected_tiles["test_uvtile_u0_v0.png"    ] = std::make_pair(  -1,  -1);

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        check_uvtile( image2.get_ptr(), 0, expected_tiles);
    }

    // UDIM
    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_udim.<UDIM>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, "B", unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles;
        expected_tiles["test_udim.1011.png"] = std::make_pair( 0,  1);
        expected_tiles["test_udim.1001.png"] = std::make_pair( 0,  0);
        expected_tiles["test_udim.1002.png"] = std::make_pair( 1,  0);
        expected_tiles["test_udim.1102.png"] = std::make_pair( 1, 10);

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        check_uvtile( image2.get_ptr(), 0, expected_tiles);
    }

    // UDIM with metacharacters for regular expressions
    {
        std::string file_path = TEST::mi_src_path( "io/image/image/tests/test_udim_((1701))_<UDIM>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, "B", unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles;
        expected_tiles["test_udim_((1701))_1001.png"] = std::make_pair( 0,  0);

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        check_uvtile( image2.get_ptr(), 0, expected_tiles);
    }

    // Create a memory based uvtile image
    {
        std::vector<mi::base::Handle<const mi::neuraylib::ICanvas>> canvases;
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas0(
            g_image_module->create_canvas( IMAGE::PT_COLOR, 1, 1));
        canvases.push_back( canvas0);
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas1(
            g_image_module->create_canvas( IMAGE::PT_COLOR, 1, 1));
        canvases.push_back( canvas1);

        Test_canvas_image_set canvas_set( canvases);
        DBIMAGE::Image* image = new DBIMAGE::Image();
        MI_CHECK( image->reset_image_set( transaction, &canvas_set, unknown_hash) == 0);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        mi::Sint32 uvs[2][2] = { {0, 0}, {1, 1} };
        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        check_uvtile( image2.get_ptr(), 0, 2, uvs);
    }
}

void check_animated_uvtiles( DB::Transaction* transaction)
{
    mi::base::Uuid unknown_hash{0,0,0,0};

    // UDIM
    {
        std::string file_path
            = TEST::mi_src_path( "io/image/image/tests/test_udim_<UDIM>_frame_<#>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        MI_CHECK_EQUAL( 2, image2->get_length());
        MI_CHECK_EQUAL( 0, image2->get_frame_number( 0));
        MI_CHECK_EQUAL( 1, image2->get_frame_number( 1));
        MI_CHECK_EQUAL( 2, image2->get_frame_length( 0));
        MI_CHECK_EQUAL( 2, image2->get_frame_length( 1));

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles0;
        expected_tiles0["test_udim_1001_frame_0.png"] = std::make_pair( 0, 0);
        expected_tiles0["test_udim_1002_frame_0.png"] = std::make_pair( 1, 0);
        check_uvtile( image2.get_ptr(), 0, expected_tiles0);

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles1;
        expected_tiles1["test_udim_1001_frame_1.png"] = std::make_pair( 0, 0);
        expected_tiles1["test_udim_1011_frame_1.png"] = std::make_pair( 0, 1);
        check_uvtile( image2.get_ptr(), 1, expected_tiles1);
    }

    // UVTILE0
    {
        std::string file_path
            = TEST::mi_src_path( "io/image/image/tests/test_frame_<#>_uvtile_<UVTILE0>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        MI_CHECK_EQUAL( 2, image2->get_length());
        MI_CHECK_EQUAL( 0, image2->get_frame_number( 0));
        MI_CHECK_EQUAL( 1, image2->get_frame_number( 1));
        MI_CHECK_EQUAL( 2, image2->get_frame_length( 0));
        MI_CHECK_EQUAL( 2, image2->get_frame_length( 1));

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles0;
        expected_tiles0["test_frame_0_uvtile_u0_v0.png"] = std::make_pair( 0, 0);
        expected_tiles0["test_frame_0_uvtile_u1_v0.png"] = std::make_pair( 1, 0);
        check_uvtile( image2.get_ptr(), 0, expected_tiles0);

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles1;
        expected_tiles1["test_frame_0_uvtile_u0_v0.png"] = std::make_pair( 0, 0);
        expected_tiles1["test_frame_0_uvtile_u0_v1.png"] = std::make_pair( 0, 1);
        check_uvtile( image2.get_ptr(), 1, expected_tiles1);
    }

    // UVTILE1
    {
        std::string file_path
            = TEST::mi_src_path( "io/image/image/tests/test_frame_<#>_uvtile_<UVTILE1>.png");
        DBIMAGE::Image* image = new DBIMAGE::Image();
        image->reset_file( transaction, file_path, /*selector*/ nullptr, unknown_hash);
        image->dump();
        DB::Tag tag = transaction->store_for_reference_counting( image); // triggers serialization

        DB::Access<DBIMAGE::Image> image2( tag, transaction);
        MI_CHECK_EQUAL( 2, image2->get_length());
        MI_CHECK_EQUAL( 0, image2->get_frame_number( 0));
        MI_CHECK_EQUAL( 1, image2->get_frame_number( 1));
        MI_CHECK_EQUAL( 2, image2->get_frame_length( 0));
        MI_CHECK_EQUAL( 2, image2->get_frame_length( 1));

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles0;
        expected_tiles0["test_frame_0_uvtile_u0_v0.png"] = std::make_pair( -1, -1);
        expected_tiles0["test_frame_0_uvtile_u1_v0.png"] = std::make_pair(  0,  -1);
        check_uvtile( image2.get_ptr(), 0, expected_tiles0);

        std::map<std::string, std::pair<mi::Sint32, mi::Sint32>> expected_tiles1;
        expected_tiles1["test_frame_0_uvtile_u0_v0.png"] = std::make_pair( -1, -1);
        expected_tiles1["test_frame_0_uvtile_u0_v1.png"] = std::make_pair( -1,  0);
        check_uvtile( image2.get_ptr(), 1, expected_tiles1);
    }
}

void check_mdle( DB::Transaction* transaction)
{
    // test loading images from MDLE files
    DBIMAGE::Image image;
    mi::base::Uuid unknown_hash{0,0,0,0};

    std::string search_path = TEST::mi_src_path( "io/scene/mdl_elements");
    std::string file_path = search_path + "/test_resource_sharing1.mdle:resources/test.png";

    mi::Sint32 result = image.reset_file(
        transaction, file_path, /*selector*/ nullptr, unknown_hash);
    MI_CHECK_EQUAL( result, 0);

    // test loading images from MDL archives files
    file_path = search_path + "/test_archives.mdr:test_archives/test_in_archive.png";

    // since the archive path is not set as an mdl search root, this should fail.
    result = image.reset_file( transaction, file_path, "R", unknown_hash);
    MI_CHECK_EQUAL( result, -4);

    // add the path
    SYSTEM::Access_module<PATH::Path_module> path_module( false);
    path_module->add_path( PATH::MDL, search_path);
    // ... and try again
    result = image.reset_file( transaction, file_path, "R", unknown_hash);
    MI_CHECK_EQUAL( result, 0);
    MI_CHECK_EQUAL( image.get_selector(), "R");
    // remove path again
    path_module->remove_path( PATH::MDL, search_path);
}

void check_sharing( DB::Transaction* transaction, const char* simple_filename)
{
    std::string filename = TEST::mi_src_path( "io/image/image/tests/") + simple_filename;

    mi::base::Uuid invalid_hash{0,0,0,0};
    mi::base::Uuid some_hash1{1,1,1,1};
    mi::base::Uuid some_hash2{2,2,2,2};

    // Load twice with invalid hash (proxy not shared)
    DB::Tag tag1_proxy_invalid_notshared = load_image(
        transaction, filename, invalid_hash, /*shared_proxy*/ false);
    DB::Tag tag1_impl_invalid_notshared = get_impl_tag( transaction, tag1_proxy_invalid_notshared);

    DB::Tag tag2_proxy_invalid_notshared = load_image(
        transaction, filename, invalid_hash, /*shared_proxy*/ false);
    DB::Tag tag2_impl_invalid_notshared = get_impl_tag( transaction, tag2_proxy_invalid_notshared);

    // Check that there is no implementation class sharing with invalid hashes
    MI_CHECK_NOT_EQUAL(
        tag1_proxy_invalid_notshared.get_uint(), tag2_proxy_invalid_notshared.get_uint());
    MI_CHECK_NOT_EQUAL(
        tag1_impl_invalid_notshared.get_uint(), tag2_impl_invalid_notshared.get_uint());

    // Load twice with valid hash (proxy not shared)
    DB::Tag tag1_proxy_hash1_notshared = load_image(
        transaction, filename, some_hash1, /*shared_proxy*/ false);
    DB::Tag tag1_impl_hash1_notshared = get_impl_tag( transaction, tag1_proxy_hash1_notshared);

    DB::Tag tag2_proxy_hash1_notshared = load_image(
        transaction, filename, some_hash1, /*shared_proxy*/ false);
    DB::Tag tag2_impl_hash1_notshared = get_impl_tag( transaction, tag2_proxy_hash1_notshared);

    // Check that the implementation class is shared for equal hashes
    MI_CHECK_NOT_EQUAL(
        tag1_proxy_hash1_notshared.get_uint(), tag2_proxy_hash1_notshared.get_uint());
    MI_CHECK_EQUAL( tag1_impl_hash1_notshared.get_uint(), tag2_impl_hash1_notshared.get_uint());

    // Load again with different hash (proxy not shared)
    DB::Tag tag1_proxy_hash2_notshared = load_image(
        transaction, filename, some_hash2, /*shared_proxy*/ false);
    DB::Tag tag1_impl_hash2_notshared = get_impl_tag( transaction, tag1_proxy_hash2_notshared);

    // Check that the implementation class is not shared for unequal hashes
    MI_CHECK_NOT_EQUAL(
        tag1_proxy_hash1_notshared.get_uint(), tag1_proxy_hash2_notshared.get_uint());
    MI_CHECK_NOT_EQUAL(
        tag1_impl_hash1_notshared.get_uint(), tag1_impl_hash2_notshared.get_uint());

    // Check naming scheme
    const char* name_impl_invalid_notshared
        = transaction->tag_to_name( tag1_impl_invalid_notshared);
    MI_CHECK( !name_impl_invalid_notshared);
    const char* name_impl_hash1_notshared = transaction->tag_to_name( tag1_impl_hash1_notshared);
    MI_CHECK_EQUAL_CSTR(
        name_impl_hash1_notshared, "MI_default_image_impl_0x00000001000000010000000100000001");
    const char* name_impl_hash2_notshared = transaction->tag_to_name( tag1_impl_hash2_notshared);
    MI_CHECK_EQUAL_CSTR(
        name_impl_hash2_notshared, "MI_default_image_impl_0x00000002000000020000000200000002");

    // Load again with first hash (proxy shared)
    DB::Tag tag1_proxy_hash1_shared = load_image(
        transaction, filename, some_hash1, /*shared_proxy*/ true);
    DB::Tag tag1_impl_hash1_shared = get_impl_tag( transaction, tag1_proxy_hash1_shared);

    // Check that proxy class is not shared with unshared proxies
    MI_CHECK_NOT_EQUAL( tag1_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_notshared.get_uint());
    MI_CHECK_NOT_EQUAL( tag1_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_notshared.get_uint());

    // Check that implementation class is shared with unshared proxies
    MI_CHECK_EQUAL( tag1_impl_hash1_shared.get_uint(), tag1_impl_hash1_notshared.get_uint());
    MI_CHECK_EQUAL( tag1_impl_hash1_shared.get_uint(), tag1_impl_hash1_notshared.get_uint());

    // Load again with first hash (proxy shared)
    DB::Tag tag2_proxy_hash1_shared = load_image(
        transaction, filename, some_hash1, /*shared_proxy*/ true);
    DB::Tag tag2_impl_hash1_shared = get_impl_tag( transaction, tag2_proxy_hash1_shared);

    // Check that proxy class is shared with shared proxies
    MI_CHECK_EQUAL( tag2_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_shared.get_uint());
    MI_CHECK_EQUAL( tag2_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_shared.get_uint());

    // Check that implementation class is shared with shared proxies
    MI_CHECK_EQUAL( tag2_impl_hash1_shared.get_uint(), tag1_impl_hash1_shared.get_uint());
    MI_CHECK_EQUAL( tag2_impl_hash1_shared.get_uint(), tag1_impl_hash1_shared.get_uint());
}

MI_TEST_AUTO_FUNCTION( test_dbimage )
{
    Unified_database_access db_access;

    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    config_module->override( "check_serializer_store=1");
    config_module->override( "check_serializer_edit=1");

    SYSTEM::Access_module<PLUG::Plug_module> plug_module( false);
    MI_CHECK( plug_module->load_library( plugin_path_openimageio));
    MI_CHECK( plug_module->load_library( plugin_path_dds));

    DB::Database* database = db_access.get_database();
    DB::Scope* scope = database->get_global_scope();
    DB::Transaction* transaction = scope->start_transaction();

    g_image_module.set();

    check_mask_handling();
    check_simple_creation( transaction);
    check_animated_textures( transaction);
    check_uvtiles( transaction);
    check_animated_textures( transaction);
    check_mdle( transaction);
    check_sharing( transaction, "test_simple.png");

    transaction->commit();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
