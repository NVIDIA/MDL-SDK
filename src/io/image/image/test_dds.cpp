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

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/image/image"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include "i_image.h"
#include "i_image_mipmap.h"
#include "i_image_access_canvas.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/itile.h>

#include <base/system/main/access_module.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/plug/i_plug.h>
#include <base/data/serial/i_serial_buffer_serializer.h>

#include "test_shared.h"
#include <prod/lib/neuray/test_shared.h>

using namespace MI;

SYSTEM::Access_module<IMAGE::Image_module> g_image_module;

mi::neuraylib::ICanvas* serialize_deserialize( mi::neuraylib::ICanvas* canvas)
{
    SERIAL::Buffer_serializer serializer;
    g_image_module->serialize_canvas( &serializer, canvas);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return g_image_module->deserialize_canvas( &deserializer);
}

IMAGE::IMipmap* serialize_deserialize( IMAGE::IMipmap* mipmap, bool only_first_level)
{
    SERIAL::Buffer_serializer serializer;
    g_image_module->serialize_mipmap( &serializer, mipmap, only_first_level);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return g_image_module->deserialize_mipmap( &deserializer);
}

void test_dds_layers(
    const char* file, mi::Uint32 expected_layers)
{
    std::cout << "testing layers of " << file << std::endl;

    std::string root_path = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
    MI_CHECK( canvas);

    MI_CHECK_EQUAL( expected_layers, canvas->get_layers_size());

    // serialize and de-serialize the canvas

    canvas = serialize_deserialize( canvas.get());
    MI_CHECK( canvas);

    // export and re-import the file

    std::string output_path = file;
    output_path += ".dds";

    bool result = g_image_module->export_canvas( canvas.get(), output_path.c_str());
    MI_CHECK( result);

    canvas = g_image_module->create_canvas( IMAGE::File_based(), output_path, /*selector*/ nullptr);
    MI_CHECK( canvas);

    MI_CHECK_EQUAL( expected_layers, canvas->get_layers_size());

    // export each layer as PNG

    mi::Uint32 width  = canvas->get_resolution_x();
    mi::Uint32 height = canvas->get_resolution_y();
    auto buffer = std::make_unique<mi::Uint8[]>( 4 * width * height);

    for( mi::Uint32 layer = 0; layer < canvas->get_layers_size(); ++layer) {

        // extract layer

        IMAGE::Access_canvas access_canvas( canvas.get());
        access_canvas.read_rect(
            buffer.get(), false, IMAGE::PT_RGBA, 0, 0, width, height, 0, layer);

        mi::base::Handle<mi::neuraylib::ICanvas> flat_canvas(
            g_image_module->create_canvas( IMAGE::PT_RGBA, width, height, 1));
        IMAGE::Edit_canvas edit_flat_canvas( flat_canvas.get());
        edit_flat_canvas.write_rect(
            buffer.get(), false, IMAGE::PT_RGBA, 0, 0, width, height);

        // export layer as PNG

        std::ostringstream output_path;
        output_path << "export_of_" << file << "_layer_" << layer << ".png";

        bool result = g_image_module->export_canvas( flat_canvas.get(), output_path.str().c_str());
        MI_CHECK( result);

        std::ostringstream reference_path;
        reference_path << root_path << "reference/export_of_" << file << "_layer_" << layer
            << ".png";
        MI_CHECK_IMG_DIFF( output_path.str(), reference_path.str());
    }
}

void test_dds_cubemap( const char* file)
{
    std::cout << "testing cubemap " << file << std::endl;

    std::string root_path = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
    MI_CHECK( canvas);

    MI_CHECK_EQUAL( 6, canvas->get_layers_size());

    // serialize and de-serialize the canvas

    canvas = serialize_deserialize( canvas.get());
    MI_CHECK( canvas);

    // export and re-import the file

    std::string output_path = file;
    output_path += ".dds";

    bool result = g_image_module->export_canvas( canvas.get(), output_path.c_str());
    MI_CHECK( result);

    canvas = g_image_module->create_canvas( IMAGE::File_based(), output_path, /*selector*/ nullptr);
    MI_CHECK( canvas);

    MI_CHECK_EQUAL( 6, canvas->get_layers_size());

    // export each layer as PNG

    mi::Uint32 width  = canvas->get_resolution_x();
    mi::Uint32 height = canvas->get_resolution_y();
    auto buffer = std::make_unique<mi::Uint8[]>( 4 * width * height);

    for( mi::Uint32 layer = 0; layer < canvas->get_layers_size(); ++layer) {

        // extract layer

        IMAGE::Access_canvas access_canvas( canvas.get());
        access_canvas.read_rect(
            buffer.get(), false, IMAGE::PT_RGBA, 0, 0, width, height, 0, layer);

        mi::base::Handle<mi::neuraylib::ICanvas> flat_canvas(
            g_image_module->create_canvas( IMAGE::PT_RGBA, width, height, 1));
        IMAGE::Edit_canvas edit_flat_canvas( flat_canvas.get());
        edit_flat_canvas.write_rect(
            buffer.get(), false, IMAGE::PT_RGBA, 0, 0, width, height);

        // export layer as PNG

        std::ostringstream output_path;
        output_path << "export_of_" << file << "_layer_" << layer << ".png";

        bool result = g_image_module->export_canvas( flat_canvas.get(), output_path.str().c_str());
        MI_CHECK( result);

        // Compare layer with reference image
        std::ostringstream reference_path;
        reference_path << root_path << "reference/export_of_" << file << "_layer_" << layer
            << ".png";
        MI_CHECK_IMG_DIFF( output_path.str(), reference_path.str());
    }
}

void test_dds_miplevels(
    const char* file, mi::Uint32 expected_levels)
{
    std::cout << "testing miplevels of " << file << std::endl;

    std::string root_path = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    mi::base::Handle<IMAGE::IMipmap> mipmap( g_image_module->create_mipmap(
        IMAGE::File_based(), input_path, /*selector*/ nullptr, false));
    MI_CHECK( mipmap);

    MI_CHECK_EQUAL( expected_levels, mipmap->get_nlevels());

    // serialize and de-serialize the canvas

    mipmap = serialize_deserialize( mipmap.get(), false);
    MI_CHECK( mipmap);

    // export and re-import the file

    std::string output_path = file;
    output_path += ".dds";

    bool result = g_image_module->export_mipmap( mipmap.get(), output_path.c_str());
    MI_CHECK( result);

    // export each level as PNG

    mipmap = g_image_module->create_mipmap(
        IMAGE::File_based(), output_path, /*selector*/ nullptr, false);
    MI_CHECK( mipmap);

    MI_CHECK_EQUAL( expected_levels, mipmap->get_nlevels());

    for( mi::Uint32 level = 0; level < mipmap->get_nlevels(); ++level) {

        // export level as PNG

        mi::base::Handle<mi::neuraylib::ICanvas> canvas( mipmap->get_level( level));

        std::ostringstream output_path;
        output_path << "export_of_" << file << "_level_" << level << ".png";

        bool result = g_image_module->export_canvas( canvas.get(), output_path.str().c_str());
        MI_CHECK( result);

        // The reference images here are crucial for this test to ensure that the miplevels have
        // been loaded from file (different color in each level) and not computed as usual.

        if( level >= expected_levels)
            continue;

        std::ostringstream reference_path;
        reference_path << root_path << "reference/export_of_" << file << "_level_" << level
            << ".png";
        MI_CHECK_IMG_DIFF( output_path.str(), reference_path.str());
    }
}

MI_TEST_AUTO_FUNCTION( test_dds )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    SYSTEM::Access_module<PLUG::Plug_module> plug_module( false);
    MI_CHECK( plug_module->load_library( plugin_path_openimageio));
    MI_CHECK( plug_module->load_library( plugin_path_dds));

    g_image_module.set();

    test_dds_cubemap( "test_dds_cubemap1.dds");

    g_image_module.reset();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

