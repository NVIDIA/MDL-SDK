/******************************************************************************
 * Copyright (c) 2007-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief regression test infrastructure
///
/// - init_unit_test_suite()          Run all available auto test cases.
///
/// This module provides a generic init_unit_test_suite() function that
/// will run all available automatic test cases.

#ifndef BASE_SYSTEM_TEST_AUTO_DRIVER_H
#define BASE_SYSTEM_TEST_AUTO_DRIVER_H

#include "i_test_auto_case.h"
#include "i_test_driver.h"

#ifndef MI_TEST_AUTO_SUITE_NAME
#  define MI_TEST_AUTO_SUITE_NAME ""
#endif

namespace MI { namespace TEST {

static int                  global_argc;
static char const * const * global_argv;

int                  get_argc() { return global_argc; }
char const * const * get_argv() { return global_argv; }

Test_suite * get_master_test_suite()
{
    static Test_suite * const suite( MI_TEST_SUITE(MI_TEST_AUTO_SUITE_NAME) );
    return suite;
}

Auto_test_case::Auto_test_case(Test_case * test)
{
    get_master_test_suite()->add(test);
}

}} // MI::TEST

MI::TEST::Test_suite * init_unit_test_suite(int argc, char ** argv)
{
    MI::TEST::global_argc = argc;
    MI::TEST::global_argv = argv;
    return MI::TEST::get_master_test_suite();
}

#endif // BASE_SYSTEM_TEST_AUTO_DRIVER_H
