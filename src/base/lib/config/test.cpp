/***************************************************************************************************
 * Copyright (c) 2004-2024, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

/// \file
/// \brief Test the config module

#include "pch.h"

#include <base/lib/config/config.h>
#include <base/lib/log/i_log_module.h>
#include <base/util/registry/i_config_registry.h>
#include <base/system/main/access_module.h>
#include <base/system/test/i_test_auto_driver.h>

using namespace MI;
using namespace CONFIG;

MI_TEST_AUTO_FUNCTION( verify_config_module )
{
    SYSTEM::Access_module<LOG::Log_module> log_module(false);
    SYSTEM::Access_module<Config_module> config_module(false);
    Config_registry& registry = config_module->get_configuration();

    int   c_int         = 11;
    int   c_int3        = 333;
    float l_flt         = 11.5;
    float l_flt3        = 111.5;
    std::string d_str   = "ss1";
    std::string d_str3= "sss1";
    bool  c_bool0       = false;
    bool  c_bool1       = true;
    registry.add_value("int", c_int);
    registry.add_value("int3", c_int3);
    registry.add_value("flt", l_flt);
    registry.add_value("flt3", l_flt3);
    registry.add_value("str", d_str);
    registry.add_value("str3", d_str3);
    registry.add_value("bool0", c_bool0);
    registry.add_value("bool1", c_bool1);

    int   c_int_ = 0;
    int   c_int3_ = 0;
    float l_flt_;
    float l_flt3_;
    std::string d_str_;
    std::string d_str3_;
    bool  c_bool0_;
    bool  c_bool1_;

    update_value(registry, "int", c_int_);
    update_value(registry, "int3", c_int3_);
    update_value(registry, "flt", l_flt_);
    update_value(registry, "flt3", l_flt3_);
    update_value(registry, "str", d_str_);
    update_value(registry, "str3", d_str3_);
    update_value(registry, "bool0", c_bool0_);
    update_value(registry, "bool1", c_bool1_);

    MI_REQUIRE_EQUAL(c_int, c_int_);
    MI_REQUIRE_EQUAL(c_int3, c_int3_);
    MI_REQUIRE_EQUAL(l_flt, l_flt_);
    MI_REQUIRE_EQUAL(l_flt3, l_flt3_);
    MI_REQUIRE_EQUAL(d_str, d_str_);
    MI_REQUIRE_EQUAL(d_str3, d_str3_);

    config_module.reset();
}

MI_TEST_AUTO_FUNCTION( verify_config_module_override )
{
    SYSTEM::Access_module<LOG::Log_module> log_module(false);
    SYSTEM::Access_module<Config_module> config_module(false);
    const Config_registry& registry = config_module->get_configuration();
    {
    config_module->override("floating_value=7");
    float value = 0.0f;
    MI_CHECK(update_value(registry, "floating_value", value));
    MI_CHECK_EQUAL(value, 7.f);
    }
    {
    config_module->override("STR_string_value=\"224.1.7.3\"");
    std::string value;
    MI_CHECK(update_value(registry, "string_value", value));
    MI_CHECK_EQUAL(value, "224.1.7.3");
    }
    // note that we can't test for "empty_value=" or "empty_value= " because both cause
    // an error message which causes the unit test to be failed
    {
    config_module->override("STR_path_include_2=\"/foor/bar\"");
    std::string value;
    MI_CHECK(update_value(registry, "path_include_2", value));
    MI_CHECK_EQUAL(value, "/foor/bar");
    }
}
