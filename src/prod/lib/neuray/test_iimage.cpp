/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"

struct Data
{
    mi::Size frame_number;
    mi::Sint32 u, v;
    mi::Uint32 width;
    std::string filename;
    mi::Size frame_index;
    mi::Size uvtile_index;
};

const mi::Size data_ok_N = 7;
struct Data data_ok[data_ok_N] = {
    { 7,  0,  0, 1, "test_iimage_1.png", 3, 0 },
    { 2,  1,  2, 2, "test_iimage_2.png", 0, 0 },
    { 2,  2,  3, 3, "test_iimage_3.png", 0, 1 },
    { 2,  3,  4, 4, "test_iimage_4.png", 0, 2 },
    { 4, -2, -2, 5, "test_iimage_5.png", 2, 0 },
    { 4, -3, -3, 6, "test_iimage_6.png", 2, 1 },
    { 3,  7,  7, 7, "test_iimage_7.png", 1, 0 }
};

const mi::Size data_uv_N = 2;
struct Data data_uv[data_uv_N] = {
    { 0,  0,  0, 1, "test_iimage_1.png", 0, 0 },
    { 0,  0,  0, 2, "test_iimage_2.png", 0, 0 },
};

// Checks that a frame/uvtile of \p image matches the properties in \p d.
void check_uvtile( const mi::neuraylib::IImage* image, const Data& d)
{
    MI_CHECK_EQUAL( image->get_frame_number( d.frame_index), d.frame_number);
    MI_CHECK_EQUAL( image->get_frame_id( d.frame_number), d.frame_index);

    mi::Sint32 u, v;
    image->get_uvtile_uv( d.frame_index, d.uvtile_index, u, v);
    MI_CHECK_EQUAL( u, d.u);
    MI_CHECK_EQUAL( v, d.v);
    mi::Size i = image->get_uvtile_id( d.frame_index, d.u, d.v);
    MI_CHECK_EQUAL( i, d.uvtile_index);

    MI_CHECK_EQUAL( image->resolution_x( d.frame_index, d.uvtile_index, 0), d.width);
}

// Converts an array of Data into mi::IArray of "Uvtile" structs.
mi::IArray* create_array_for_canvases(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IImage_api* image_api,
    const struct Data* data,
    size_t data_N)
{
    std::ostringstream type_name;
    type_name << "Uvtile[" << data_N << "]";
    auto* array = transaction->create<mi::IArray>( type_name.str().c_str());

    for( mi::Size i = 0; i < data_N; ++i) {
        mi::base::Handle<mi::IStructure> elem( array->get_element<mi::IStructure>( i));
        mi::base::Handle<mi::ISize> f( elem->get_value<mi::ISize>( "frame"));
        f->set_value( data[i].frame_number);
        mi::base::Handle<mi::ISint32> u( elem->get_value<mi::ISint32>( "u"));
        u->set_value( data[i].u);
        mi::base::Handle<mi::ISint32> v( elem->get_value<mi::ISint32>( "v"));
        v->set_value( data[i].v);
        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            image_api->create_canvas( "Rgba", data[i].width, 1));
        elem->set_value( "canvas", canvas.get());
    }

    return array;
}

void check_set_from_canvas(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IImage_api* image_api)
{
    mi::base::Handle<mi::IArray> array( create_array_for_canvases(
        transaction, image_api, data_ok, data_ok_N));

    mi::base::Handle<mi::neuraylib::IImage> image(
        transaction->create<mi::neuraylib::IImage>( "Image"));
    MI_CHECK( image->set_from_canvas( array.get()));
    MI_CHECK_EQUAL( transaction->store( image.get(), "image_canvas"), 0); // triggers serialization

    mi::base::Handle<const mi::neuraylib::IImage> c_image(
        transaction->access<mi::neuraylib::IImage>( "image_canvas"));

    MI_CHECK_EQUAL( c_image->is_animated(), true);
    MI_CHECK_EQUAL( c_image->is_uvtile(), true);
    MI_CHECK_EQUAL( c_image->get_length(), 4);

    MI_CHECK_EQUAL( c_image->get_frame_length( 0), 3);
    MI_CHECK_EQUAL( c_image->get_frame_length( 1), 1);
    MI_CHECK_EQUAL( c_image->get_frame_length( 2), 2);
    MI_CHECK_EQUAL( c_image->get_frame_length( 3), 1);

    for( auto& d : data_ok)
        check_uvtile( c_image.get(), d);

    mi::base::Handle<mi::IArray> array_uv( create_array_for_canvases(
        transaction, image_api, data_uv, data_uv_N));

    mi::base::Handle<mi::neuraylib::IImage> image_uv(
        transaction->create<mi::neuraylib::IImage>( "Image"));
    MI_CHECK( !image_uv->set_from_canvas( array_uv.get()));
}


void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            neuray->get_api_component<mi::neuraylib::IImage_api>());

        // reset_file() is tested in io/scene/dbimage/test.cpp
        check_set_from_canvas( transaction.get(), image_api.get());

        transaction->commit();
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_iimage )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));

        // load plugins
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        run_tests( neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

