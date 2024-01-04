/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "image_canvas_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/icanvas.h>

#include <sstream>
#include <base/system/main/access_module.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/plug/i_plug.h>

#include "test_shared.h"
#include <prod/lib/neuray/test_shared.h>

using namespace MI;

MI_TEST_AUTO_FUNCTION( test_mipmap )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    SYSTEM::Access_module<PLUG::Plug_module> plug_module( false);
    MI_CHECK( plug_module->load_library( plugin_path_openimageio));

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    std::string root_path = TEST::mi_src_path( "io/image/image/tests/");

    mi::base::Handle<IMAGE::IMipmap> mipmap( image_module->create_mipmap(
        IMAGE::File_based(), root_path + "test_mipmap.png", /*selector*/ nullptr));
    MI_CHECK( mipmap);

    MI_CHECK_EQUAL( 7, mipmap->get_nlevels());

    for( mi::Uint32 level = 0; level < mipmap->get_nlevels(); ++level) {

        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
            "mipmap size before level %u is computed: %llu", level, mipmap->get_size());

        // Compute requested miplevel
        mi::base::Handle<mi::neuraylib::ICanvas> canvas( mipmap->get_level( level));

        // Export  miplevel
        std::ostringstream output_path;
        output_path << "export_of_test_mipmap_level_" << level << ".png";
        bool result = image_module->export_canvas( canvas.get(), output_path.str().c_str());
        MI_CHECK( result);

        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
            "mipmap size after level %u is computed: %llu", level, mipmap->get_size());

        mi::base::Handle<IMAGE::ICanvas> canvas_internal( canvas->get_interface<IMAGE::ICanvas>());
        LOG::mod_log->info( M_IMAGE, LOG::Mod_log::C_IO,
            "canvas size of level %u is: %llu", level, canvas_internal->get_size());

        // Compare miplevel
        std::ostringstream reference_path;
        reference_path << root_path << "reference/export_of_test_mipmap_level_" << level << ".png";
        MI_CHECK_IMG_DIFF( output_path.str(), reference_path.str());
    }
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
