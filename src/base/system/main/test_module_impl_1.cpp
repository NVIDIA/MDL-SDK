/******************************************************************************
 * Copyright (c) 2007-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "module_registration.h"
#include "module_registration_entry.h"
#include <base/system/test/i_test_current_location.h>
#include <iostream>

#ifdef DEBUG
#  define MI_TRACE(msg) std::cout << MI_CURRENT_FUNCTION_PRETTY << msg << std::endl
#else
#  define MI_TRACE(msg) ((void)0)
#endif

struct Test_module_impl_1 : public Test_module
{
    Test_module_impl_1()        { MI_TRACE(""); }
    ~Test_module_impl_1()       { MI_TRACE(""); }
    void run()                  { MI_TRACE(""); ++n_impl1_runs; }

    bool init()                 { MI_TRACE(""); return true; }
    void exit()                 { MI_TRACE(""); }

    static const char* get_name() { return "Test_module_impl_1"; }
};

static MI::SYSTEM::Module_registration<Test_module_impl_1> const
    register_test_module_1(MI::SYSTEM::M_MAIN, "MAIN");


// Allow link time detection.
MI::SYSTEM::Module_registration_entry* Test_module::get_instance()
{
    return register_test_module_1.init_module(register_test_module_1.get_name());
}

