/******************************************************************************
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/istring.h>

#include "test_shared.h"

MI_TEST_AUTO_FUNCTION( test_imdl_configuration )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray.is_valid_interface());

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());

        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

        std::string path;
        mi::base::Handle<const mi::IString> string;

        // check default paths
        MI_CHECK_EQUAL( 0, mdl_configuration->get_mdl_paths_length());

        string = mdl_configuration->get_mdl_path( 0);
        MI_CHECK( !string.is_valid_interface());

        MI_CHECK_EQUAL( 0, mdl_configuration->get_resource_paths_length());

        string = mdl_configuration->get_resource_path( 0);
        MI_CHECK( !string.is_valid_interface());

        // clear default paths (such that the first added path has index 0)
        mdl_configuration->clear_resource_paths();

        MI_CHECK_EQUAL(0, mdl_configuration->get_mdl_paths_length());
        MI_CHECK_EQUAL(0, mdl_configuration->get_resource_paths_length());

        // add and check shader paths
        MI_CHECK_EQUAL( -2, mdl_configuration->add_mdl_path( "/non-existing"));
        debug_configuration->set_option( "allow_invalid_search_paths=1");
        MI_CHECK_EQUAL( -2, mdl_configuration->add_mdl_path( "/non-existing"));
        MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( "/non-existing"));
        debug_configuration->set_option( "allow_invalid_search_paths=0");
        MI_CHECK_EQUAL( -2, mdl_configuration->add_mdl_path( "/non-existing"));

        path = MI::TEST::mi_src_path( "base/system/main");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        MI_CHECK_EQUAL( 1, mdl_configuration->get_mdl_paths_length());
        string = mdl_configuration->get_mdl_path( 0);
        MI_CHECK( string.is_valid_interface());
        MI_CHECK( string->get_c_str());
        MI_CHECK_EQUAL_CSTR( string->get_c_str(), path.c_str());
        string = mdl_configuration->get_mdl_path( 1);
        MI_CHECK( !string.is_valid_interface());

        path = MI::TEST::mi_src_path( "base/system/version");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        MI_CHECK_EQUAL( 2, mdl_configuration->get_mdl_paths_length());

        MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));
        MI_CHECK_EQUAL( 1, mdl_configuration->get_mdl_paths_length());

        mdl_configuration->clear_mdl_paths();
        MI_CHECK_EQUAL( 0, mdl_configuration->get_mdl_paths_length());

        // add and check resource paths

        path = MI::TEST::mi_src_path( "base/system/test");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_resource_path( path.c_str()));
        MI_CHECK_EQUAL( 1, mdl_configuration->get_resource_paths_length());

        string = mdl_configuration->get_resource_path( 0);
        MI_CHECK( string.is_valid_interface());
        MI_CHECK( string->get_c_str());
        MI_CHECK_EQUAL_CSTR( string->get_c_str(), path.c_str());
        string = mdl_configuration->get_resource_path( 1);
        MI_CHECK( !string.is_valid_interface());

        path = MI::TEST::mi_src_path( "base/data/db");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_resource_path( path.c_str()));
        MI_CHECK_EQUAL( 2, mdl_configuration->get_resource_paths_length());

        MI_CHECK_EQUAL( 0, mdl_configuration->remove_resource_path( path.c_str()));
        MI_CHECK_EQUAL( 1, mdl_configuration->get_resource_paths_length());

        mdl_configuration->clear_resource_paths();
        MI_CHECK_EQUAL( 0, mdl_configuration->get_resource_paths_length());

        // check explicit cast and expose let expressions settings

        MI_CHECK_EQUAL(true, mdl_configuration->get_implicit_cast_enabled());
        MI_CHECK_EQUAL(0, mdl_configuration->set_implicit_cast_enabled(false));
        MI_CHECK_EQUAL(false, mdl_configuration->get_implicit_cast_enabled());


        MI_CHECK_EQUAL(false, mdl_configuration->get_expose_names_of_let_expressions());
        MI_CHECK_EQUAL(0, mdl_configuration->set_expose_names_of_let_expressions(true));
        MI_CHECK_EQUAL(true, mdl_configuration->get_expose_names_of_let_expressions());

        // start neuray

        MI_CHECK_EQUAL( 0, neuray->start());

        // explicit cast and expose let expressions settings cannot be changed after startup.

        MI_CHECK_EQUAL(-1, mdl_configuration->set_implicit_cast_enabled(true));
        MI_CHECK_EQUAL(false, mdl_configuration->get_implicit_cast_enabled());

        MI_CHECK_EQUAL(-1, mdl_configuration->set_expose_names_of_let_expressions(false));
        MI_CHECK_EQUAL(true, mdl_configuration->get_expose_names_of_let_expressions());

        // check path settings again

        MI_CHECK_EQUAL( 0, mdl_configuration->get_mdl_paths_length());
        MI_CHECK_EQUAL( 0, mdl_configuration->get_resource_paths_length());

        // add another mdl path

        path = MI::TEST::mi_src_path( "base/data/dblight");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        MI_CHECK_LESS_OR_EQUAL( 1, mdl_configuration->get_mdl_paths_length());

        // add another resource path

        path = MI::TEST::mi_src_path( "base/data/thread_pool");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_resource_path( path.c_str()));
        MI_CHECK_LESS_OR_EQUAL( 1, mdl_configuration->get_resource_paths_length());
    }
    MI_CHECK_EQUAL( 0, neuray->shutdown());

    neuray = 0;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

