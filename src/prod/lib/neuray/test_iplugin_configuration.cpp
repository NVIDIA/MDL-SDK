/******************************************************************************
 * Copyright (c) 2008-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/ineuray.h>

#include "test_shared.h"

void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
        neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

    // load plugin library -- not available after start
    MI_CHECK_EQUAL( -1, plugin_configuration->load_plugin_library( plugin_path_openimageio));

    // load plugins from directory -- not available after start
    MI_CHECK( plugin_configuration->load_plugins_from_directory( ".") == -1);

    // check enumeration of plugins again
    MI_CHECK_LESS_OR_EQUAL( 13, plugin_configuration->get_plugin_length());

    mi::base::Handle<mi::base::IPlugin_descriptor> descriptor( plugin_configuration->get_plugin_descriptor( 0));
    MI_CHECK( descriptor->get_plugin());
    MI_CHECK( descriptor->get_plugin_library_path());
    descriptor = plugin_configuration->get_plugin_descriptor( plugin_configuration->get_plugin_length());
    MI_CHECK( !descriptor);

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_iplugin_configuration )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

        // load plugin library
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        // load plugin library -- error if it does not exist
        MI_CHECK( plugin_configuration->load_plugin_library( "non-existing.so") == -1);


        // load plugin library -- error if directory does not exist
        MI_CHECK( plugin_configuration->load_plugins_from_directory( "/non-existing/path") == -1);
        MI_CHECK( plugin_configuration->load_plugins_from_directory( "test_iplugin_configuration.o") == -1);

        // check enumeration of plugins
        MI_CHECK_LESS_OR_EQUAL( 13, plugin_configuration->get_plugin_length());

        mi::base::Handle<mi::base::IPlugin_descriptor> descriptor( plugin_configuration->get_plugin_descriptor( 0));
        MI_CHECK( descriptor->get_plugin());
        MI_CHECK( descriptor->get_plugin_library_path());
        descriptor = plugin_configuration->get_plugin_descriptor( plugin_configuration->get_plugin_length());
        MI_CHECK( !descriptor);

        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray.reset();
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

