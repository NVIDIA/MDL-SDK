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

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for base/lib/path"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>
#include <base/system/test/i_test_environment.h>

#include "i_path.h"

#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_utils.h>
#include <boost/algorithm/string.hpp>

using namespace MI;

MI_TEST_AUTO_FUNCTION( test_module )
{
    SYSTEM::Access_module<PATH::Path_module> path_module( false);

    std::string p1 = TEST::mi_src_path( "base/lib/log");
    std::string p2 = TEST::mi_src_path( "base/lib/path");

    MI_CHECK_EQUAL(  0, path_module->add_path( PATH::MDL, p1));
    MI_CHECK_EQUAL( -2, path_module->add_path( PATH::MDL, "/foo"));
    MI_CHECK_EQUAL(  0, path_module->remove_path( PATH::MDL, p1));
    MI_CHECK_EQUAL( -2, path_module->remove_path( PATH::MDL, p1));

    MI_CHECK_EQUAL(  0, path_module->add_path( PATH::MDL, p1));
    MI_CHECK_EQUAL(  0, path_module->add_path( PATH::MDL, p2));
    MI_CHECK_EQUAL(  0, path_module->add_path( PATH::MDL, p1));
    MI_CHECK_EQUAL(  0, path_module->remove_path( PATH::MDL, p1));
    MI_CHECK_EQUAL(  0, path_module->remove_path( PATH::MDL, p1));
    MI_CHECK_EQUAL( p2, path_module->get_path( PATH::MDL, 0));
    MI_CHECK_EQUAL( "", path_module->get_path( PATH::MDL, 1));
    MI_CHECK_EQUAL(  0, path_module->set_path( PATH::MDL, 0, p1));
    MI_CHECK_EQUAL( -3, path_module->set_path( PATH::MDL, 1, p2));

    std::vector<std::string> v( 1);
    path_module->clear_search_path( PATH::MDL);
    v[0] =  "/foo";
    MI_CHECK_EQUAL( -2, path_module->set_search_path( PATH::MDL, v));
    v[0] = p1;
    v.push_back( p2);
    MI_CHECK_EQUAL(  0, path_module->set_search_path( PATH::MDL, v));

    std::string absolute_file_name = TEST::mi_src_path( "base/lib/path/i_path.h");
    MI_CHECK_EQUAL( absolute_file_name, path_module->search( PATH::MDL, "i_path.h"));
    MI_CHECK_EQUAL( "",                 path_module->search( PATH::MDL, "missing.h"));
    MI_CHECK_EQUAL( absolute_file_name, path_module->search( PATH::MDL, absolute_file_name));


    std::string absolute_file_name2 = "Hello\\World\\Foo";
    std::string absolute_file_name3 = R"(\Hello\World\Foo\)";

    std::vector<std::string> tests;
    tests.emplace_back("Hello\\World\\Foo");
    tests.emplace_back(R"(\Hello\World\Foo\)");
    tests.emplace_back(R"(\\Hello\\World\Foo\\)");
    tests.emplace_back("\\");
    tests.emplace_back("\\\\");
    tests.emplace_back("");
    {
        using namespace boost::algorithm;

        std::string sep = "\\\\/ ,";

        for (auto& test : tests)
        {
            std::vector<std::string> token_list1;
            std::vector<std::string> token_list2;

            // boost
            split(token_list1, test, is_any_of(sep), token_compress_off);
            if (token_list1.back().empty()) token_list1.pop_back();

            // own function based on boost
            MI::STRING::split(test, sep, token_list2);

            MI_CHECK_EQUAL(token_list1.size(), token_list2.size());

            for (int i = 0, n = token_list1.size(); i < n; ++i)
            {
                MI_CHECK_EQUAL_CSTR(token_list1[i].c_str(), token_list2[i].c_str());
            }
        }
    }

}

MI_TEST_MAIN_CALLING_TEST_MAIN();
