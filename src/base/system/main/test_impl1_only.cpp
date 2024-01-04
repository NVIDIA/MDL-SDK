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
/// \brief  module system test suite

#include "pch.h"
#include "test_module_api.h"
#include "access_module.h"

#define MI_TEST_AUTO_SUITE_NAME "Module System Regression Tests (implementation 1 only)"
#include <base/system/test/i_test_auto_driver.h>

MI_TEST_AUTO_FUNCTION( require_implementation_1_to_be_available )
{
    MI::SYSTEM::Access_module<Test_module> module(false);
    MI_CHECK_EQUAL(module.get_status(), MI::SYSTEM::MODULE_STATUS_INITIALIZED);
}

MI_TEST_AUTO_FUNCTION( verify_that_implementation_1_can_be_run )
{
    MI::SYSTEM::Access_module<Test_module> module;
    MI_CHECK_EQUAL(module.get_status(), MI::SYSTEM::MODULE_STATUS_UNINITIALIZED);

    MI_CHECK_EQUAL(Test_module::n_impl1_runs, 0u);
    module.set();
    MI_CHECK_EQUAL(module.get_status(), MI::SYSTEM::MODULE_STATUS_INITIALIZED);
    module->run();
    MI_CHECK_EQUAL(Test_module::n_impl1_runs, 1u);

    module.reset();
    MI_CHECK_EQUAL(module.get_status(), MI::SYSTEM::MODULE_STATUS_UNINITIALIZED);
}
