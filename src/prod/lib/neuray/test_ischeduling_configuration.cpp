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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/ischeduling_configuration.h>

#include "test_shared.h"

void run_tests( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IScheduling_configuration> scheduling_configuration(
        neuray->get_api_component<mi::neuraylib::IScheduling_configuration>());

    MI_CHECK_EQUAL( 1.0, scheduling_configuration->get_cpu_load_limit());
    MI_CHECK_EQUAL( 1.0, scheduling_configuration->get_gpu_load_limit());

    MI_CHECK_EQUAL( -2, scheduling_configuration->set_thread_affinity_enabled( true));
    MI_CHECK_EQUAL( false, scheduling_configuration->get_thread_affinity_enabled());

    MI_CHECK_EQUAL( 0, neuray->start());

    // check defaults

    // assume that the machine has 2 or more CPUs, no assumptions about GPUs because
    // GPUs are disabled in unit tests when run via the script run_test.sh
    // (MI_DISABLE_GPU_FOR_UNIT_TEST)
    MI_CHECK_LESS_OR_EQUAL( 2.0, scheduling_configuration->get_cpu_load_limit());

    MI_CHECK_EQUAL( 0, scheduling_configuration->set_cpu_load_limit( 1.25));
    MI_CHECK_EQUAL( 1.25, scheduling_configuration->get_cpu_load_limit());

    MI_CHECK_EQUAL( -1, scheduling_configuration->set_cpu_load_limit(  0.5));
    MI_CHECK_EQUAL( 1.25, scheduling_configuration->get_cpu_load_limit());

    mi::Sint32 result = scheduling_configuration->set_thread_affinity_enabled( true);
    MI_CHECK( result == 0 || result == -1);
    bool enabled = scheduling_configuration->get_thread_affinity_enabled();
    MI_CHECK( enabled || result == -1);

    MI_CHECK_EQUAL( -1, scheduling_configuration->set_accept_delegations( true));
    MI_CHECK( !scheduling_configuration->get_accept_delegations());
    MI_CHECK_EQUAL( -1, scheduling_configuration->set_accept_delegations( false));
    MI_CHECK( !scheduling_configuration->get_accept_delegations());

    MI_CHECK_EQUAL( -1, scheduling_configuration->set_work_delegation_enabled( true));
    MI_CHECK( !scheduling_configuration->get_work_delegation_enabled());
    MI_CHECK_EQUAL( -1, scheduling_configuration->set_work_delegation_enabled( false));
    MI_CHECK( !scheduling_configuration->get_work_delegation_enabled());

    MI_CHECK_EQUAL( -1, scheduling_configuration->set_gpu_work_delegation_enabled( true));
    MI_CHECK( !scheduling_configuration->get_gpu_work_delegation_enabled());
    MI_CHECK_EQUAL( -1, scheduling_configuration->set_gpu_work_delegation_enabled( false));
    MI_CHECK( !scheduling_configuration->get_gpu_work_delegation_enabled());

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_ischeduling_configuration )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray.reset();
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

