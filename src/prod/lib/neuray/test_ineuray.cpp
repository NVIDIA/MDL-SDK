/******************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/config.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

// X11/Xlib.h defines Status to int
// Include it here to test our workarounds in neuray.h and inetworking_configuration.h
#ifdef MI_PLATFORM_UNIX
#include <X11/Xlib.h>
#endif // MI_PLATFORM_UNIX

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <mi/neuraylib/imdl_archive_api.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_compiler.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_compatibility_api.h>
#include <mi/neuraylib/imdl_discovery_api.h>
#include <mi/neuraylib/imdl_distiller_api.h>
#include <mi/neuraylib/imdl_evaluator_api.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_i18n_configuration.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdle_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iplugin_api.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/iversion.h>


#include "test_shared.h"

class IMy_api_component : public
    mi::base::Interface_declare<0xb683f118,0x62b4,0x4a46,0xb7,0xb0,0x80,0xa1,0x97,0xa6,0xcf,0x27,
                                mi::base::IInterface>
{
};

class My_api_component : public mi::base::Interface_implement<IMy_api_component>
{
};

template<class T>
void check( mi::neuraylib::INeuray* neuray, bool result)
{
    mi::base::Handle<T> iinterface( neuray->get_api_component<T>());
    MI_CHECK_EQUAL( result, !!iinterface);
}

void test_always_available( mi::neuraylib::INeuray* neuray)
{
    check<mi::neuraylib::IDebug_configuration>( neuray, true);
    check<mi::neuraylib::IFactory>( neuray, true);
    check<mi::neuraylib::ILogging_configuration>( neuray, true);
    check<mi::neuraylib::IMdl_compiler>( neuray, true);
    check<mi::neuraylib::IMdl_configuration>( neuray, true);
    check<mi::neuraylib::IMdl_i18n_configuration>( neuray, true);
    check<mi::neuraylib::IPlugin_api>( neuray, true);
    check<mi::neuraylib::IPlugin_configuration>( neuray, true);
    check<mi::neuraylib::IVersion>( neuray, true);
}

void test_sometimes_available( mi::neuraylib::INeuray* neuray, bool result)
{
    check<mi::neuraylib::IDatabase>( neuray, result);
    check<mi::neuraylib::IImage_api>( neuray, result);
    check<mi::neuraylib::IMdl_archive_api>( neuray, result);
    check<mi::neuraylib::IMdl_backend_api>( neuray, result);
    check<mi::neuraylib::IMdl_compatibility_api>( neuray, result);
    check<mi::neuraylib::IMdl_discovery_api>( neuray, result);
    check<mi::neuraylib::IMdl_distiller_api>( neuray, result);
    check<mi::neuraylib::IMdl_evaluator_api>( neuray, result);
    check<mi::neuraylib::IMdl_factory>( neuray, result);
    check<mi::neuraylib::IMdl_impexp_api>( neuray, result);
    check<mi::neuraylib::IMdle_api>( neuray, result);
}

void test_user_api_components( mi::neuraylib::INeuray* neuray)
{
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    test_always_available( neuray);
    test_sometimes_available( neuray, false);
    test_user_api_components( neuray);

    MI_CHECK_EQUAL( 0, neuray->start());
    MI_CHECK_EQUAL( mi::neuraylib::INeuray::STARTED, neuray->get_status());

    test_always_available( neuray);
    test_sometimes_available( neuray, true);
    test_user_api_components( neuray);

    MI_CHECK_EQUAL( 0, neuray->shutdown());
    MI_CHECK_EQUAL( mi::neuraylib::INeuray::SHUTDOWN, neuray->get_status());

    test_always_available( neuray);
    test_sometimes_available( neuray, false);
    test_user_api_components( neuray);
}

MI_TEST_AUTO_FUNCTION( test_neuray )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    MI_CHECK_EQUAL( MI_NEURAYLIB_API_VERSION, neuray->get_interface_version());
    MI_CHECK( neuray->get_version());
    MI_CHECK_EQUAL( mi::neuraylib::INeuray::PRE_STARTING, neuray->get_status());

    run_tests( neuray.get());
    run_tests( neuray.get());

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

