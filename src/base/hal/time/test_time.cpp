/******************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Time class test and examples.

#include "pch.h"

#include "i_time.h"

#include <base/system/test/i_test_auto_case.h>
#ifdef MI_TEST_VERBOSE
#include <iostream>
#endif
using namespace MI::TIME;

//----------------------------------------------------------------------
/// test for localtime
MI_TEST_AUTO_FUNCTION( test_time )
{
    // how to get the local time.
    struct tm nowtime;
    MI_REQUIRE(Time::localtime(&nowtime) == 0); // syscall
    Time t = Time::mktime(&nowtime);
    (void) t;
#ifdef MI_TEST_VERBOSE
    std::cout << "current local time = " << t.to_string_rfc_2822() << std::endl;
    std::cout << "current local time = " << t.to_string_rfc_2616() << std::endl;
#endif
    Time t2 = get_cached_system_time();
    // sleep two seconds.
    MI::TIME::sleep(2);
    Time t3 = get_cached_system_time();
    // since we did not update the system time this should be the same
    MI_REQUIRE(t2 == t3);
    update_cached_system_time();
    Time t4 = get_cached_system_time();
    // Now, since we called update_cached_system_time(), t3 and t4 must be different.
    MI_REQUIRE(t3 != t4);
#ifdef MI_TEST_VERBOSE
    std::cout << "current local time = " << t4.to_string_rfc_2822() << std::endl;
    std::cout << "current local time = " << t4.to_string_rfc_2616() << std::endl;
#endif
}

//----------------------------------------------------------------------
