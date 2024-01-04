/******************************************************************************
 * Copyright (c) 2004-2023, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************
 * Description:
 *              Each container class comes with its own testing class.
 *              All of these are instantiated and invoked here.
 *****************************************************************************/

#include "pch.h"

#include "test_array.h"
#include "test_bitvector.h"
#include "test_rle_array.h"

#include <base/lib/config/config.h>
#include <base/lib/log/i_log_module.h>
#include <base/system/main/access_module.h>
#include <base/system/test/i_test_auto_driver.h>

using namespace MI;
using namespace MI::CONT;

MI_TEST_AUTO_FUNCTION( verify_container_module )
{
    // initialize used modules first
    Access_module<LOG::Log_module> log_module(false);
    Access_module<CONFIG::Config_module> config_module(false);

    //mod_log->set_severity_by_category(Mod_log::C_MAIN, Mod_log::S_ALL);
    //mod_log->set_severity_by_category(Mod_log::C_MEMORY, Mod_log::S_ALL);
    //mod_log->set_severity_limit(Mod_log::S_ALL);


    Test_Array test_array;
    MI_REQUIRE(test_array.test());


    MI_REQUIRE(test_rle_array());

    Test_Bitvector test_bitvector;
    MI_REQUIRE(test_bitvector.test());


    // finalize used modules finally
    config_module.reset();
}

