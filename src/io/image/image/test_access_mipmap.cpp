
/******************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "i_image_access_mipmap.h"
#include "image_mipmap_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>

#include <base/system/main/access_module.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/plug/i_plug.h>

#include "test_shared.h"
#include <prod/lib/neuray/test_shared.h>

using namespace MI;

MI_TEST_AUTO_FUNCTION( test_access_mipmap )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    SYSTEM::Access_module<PLUG::Plug_module> plug_module( false);
    MI_CHECK( plug_module->load_library( plugin_path_openimageio));

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    // Tests Access_mipmap by reading the 2nd miplevel split into four buffers in a 2x2 layout and
    // writing the data back from these buffers, but reversing their horizontal and vertical order.
    // The topdown flag is set for reading, but not for writing. Therefore, the resulting image is
    // mirrored vertically (and horizontally scrambled).
    //
    // The buffers have the pixel type of the output mipmap. Hence, for writing memcpy() is used per
    // row of a buffer/tile. The input image has a different pixel type. Hence, each pixel is
    // converted.

    std::string root_path = TEST::mi_src_path( "io/image/image/tests/");

    mi::base::Handle<const IMAGE::IMipmap> m;

    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap(
        IMAGE::File_based(), root_path + "test_mipmap.png", /*selector*/ nullptr));
    MI_CHECK( mipmap);
    mi::base::Handle<mi::neuraylib::ICanvas> canvas( mipmap->get_level( 2));
    MI_CHECK( canvas);
    MI_CHECK_EQUAL_CSTR( canvas->get_type(), "Rgb");

    mi::Uint32 width  = canvas->get_resolution_x();
    mi::Uint32 height = canvas->get_resolution_y();
    MI_CHECK( (width % 2 == 0) && (height % 2 == 0));
    mi::Uint32 buffer_size = height/2 * width/2 * 4;

    auto lower_left  = std::make_unique<mi::Uint8[]>( buffer_size);
    auto lower_right = std::make_unique<mi::Uint8[]>( buffer_size);
    auto upper_left  = std::make_unique<mi::Uint8[]>( buffer_size);
    auto upper_right = std::make_unique<mi::Uint8[]>( buffer_size);

    IMAGE::Access_mipmap access_mipmap( mipmap.get());
    m = access_mipmap.get();
    MI_CHECK_EQUAL( m.get(), mipmap.get());
    access_mipmap.read_rect(
        lower_left.get(),  true, IMAGE::PT_RGBA,       0,        0, width/2, height/2, 0, 0, 2);
    access_mipmap.read_rect(
        lower_right.get(), true, IMAGE::PT_RGBA, width/2,        0, width/2, height/2, 0, 0, 2);
    access_mipmap.read_rect(
        upper_left.get() , true, IMAGE::PT_RGBA,       0, height/2, width/2, height/2, 0, 0, 2);
    access_mipmap.read_rect(
        upper_right.get(), true, IMAGE::PT_RGBA, width/2, height/2, width/2, height/2, 0, 0, 2);

    mi::base::Handle<IMAGE::IMipmap> mipmap2(
        new IMAGE::Mipmap_impl( IMAGE::PT_RGBA, width, height, 1, false, 0.0f));

    IMAGE::Edit_mipmap edit_mipmap2( mipmap2.get());
    m = edit_mipmap2.get();
    MI_CHECK_EQUAL( m.get(), mipmap2.get());
    edit_mipmap2.write_rect(
        upper_right.get(), false, IMAGE::PT_RGBA,       0,        0, width/2, height/2, 0, 0, 0);
    edit_mipmap2.write_rect(
        upper_left.get(),  false, IMAGE::PT_RGBA, width/2,        0, width/2, height/2, 0, 0, 0);
    edit_mipmap2.write_rect(
        lower_right.get(), false, IMAGE::PT_RGBA,       0, height/2, width/2, height/2, 0, 0, 0);
    edit_mipmap2.write_rect(
        lower_left.get(),  false, IMAGE::PT_RGBA, width/2, height/2, width/2, height/2, 0, 0, 0);

    bool result = image_module->export_mipmap( mipmap2.get(), "access_mipmap2.png");
    MI_CHECK( result);

    std::string reference_path = root_path + "reference/access_mipmap.png";
    MI_CHECK_IMG_DIFF( "access_mipmap2.png", reference_path.c_str());

    // Use a mipmap where the tiles have exactly the same sizes as the buffers. Hence, a single
    // memcpy() is used for each buffer.

    mi::base::Handle<IMAGE::IMipmap> mipmap3(
        new IMAGE::Mipmap_impl( IMAGE::PT_RGBA, width, height, 1, false, 0.0f));

    IMAGE::Edit_mipmap edit_mipmap3( mipmap3.get());
    m = edit_mipmap3.get();
    MI_CHECK_EQUAL( m.get(), mipmap3.get());
    edit_mipmap3.write_rect(
        upper_right.get(), false, IMAGE::PT_RGBA,       0,        0, width/2, height/2, 0, 0, 0);
    edit_mipmap3.write_rect(
        upper_left.get() , false, IMAGE::PT_RGBA, width/2,        0, width/2, height/2, 0, 0, 0);
    edit_mipmap3.write_rect(
        lower_right.get(), false, IMAGE::PT_RGBA,       0, height/2, width/2, height/2, 0, 0, 0);
    edit_mipmap3.write_rect(
        lower_left.get(),  false, IMAGE::PT_RGBA, width/2, height/2, width/2, height/2, 0, 0, 0);

    result = image_module->export_mipmap( mipmap3.get(), "access_mipmap3.png");
    MI_CHECK( result);

    MI_CHECK_IMG_DIFF( "access_mipmap3.png", reference_path.c_str());

    // Use another set of buffers with a row padding of 42.

    buffer_size = height/2 * (width/2 * 4 + 42);

    auto lower_left4  = std::make_unique<mi::Uint8[]>( buffer_size);
    auto lower_right4 = std::make_unique<mi::Uint8[]>( buffer_size);
    auto upper_left4  = std::make_unique<mi::Uint8[]>( buffer_size);
    auto upper_right4 = std::make_unique<mi::Uint8[]>( buffer_size);

    IMAGE::Access_mipmap access_mipmap4( mipmap.get());
    m = access_mipmap4.get();
    MI_CHECK_EQUAL( m.get(), mipmap.get());
    access_mipmap4.read_rect(
        lower_left4.get() , true, IMAGE::PT_RGBA,       0,        0, width/2, height/2, 42, 0, 2);
    access_mipmap4.read_rect(
        lower_right4.get(), true, IMAGE::PT_RGBA, width/2,        0, width/2, height/2, 42, 0, 2);
    access_mipmap4.read_rect(
        upper_left4.get() , true, IMAGE::PT_RGBA,       0, height/2, width/2, height/2, 42, 0, 2);
    access_mipmap4.read_rect(
        upper_right4.get(), true, IMAGE::PT_RGBA, width/2, height/2, width/2, height/2, 42, 0, 2);

    mi::base::Handle<IMAGE::IMipmap> mipmap5(
        new IMAGE::Mipmap_impl( IMAGE::PT_RGBA, width, height, 1, false, 0.0f));

    IMAGE::Edit_mipmap edit_mipmap5( mipmap5.get());
    m = edit_mipmap5.get();
    MI_CHECK_EQUAL( m.get(), mipmap5.get());
    edit_mipmap5.write_rect(
        upper_right4.get(), false, IMAGE::PT_RGBA,       0,        0, width/2, height/2, 42, 0, 0);
    edit_mipmap5.write_rect(
        upper_left4.get() , false, IMAGE::PT_RGBA, width/2,        0, width/2, height/2, 42, 0, 0);
    edit_mipmap5.write_rect(
        lower_right4.get(), false, IMAGE::PT_RGBA,       0, height/2, width/2, height/2, 42, 0, 0);
    edit_mipmap5.write_rect(
        lower_left4.get() , false, IMAGE::PT_RGBA, width/2, height/2, width/2, height/2, 42, 0, 0);

    result = image_module->export_mipmap( mipmap5.get(), "access_mipmap5.png");
    MI_CHECK( result);

    MI_CHECK_IMG_DIFF( "access_mipmap5.png", reference_path.c_str());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
