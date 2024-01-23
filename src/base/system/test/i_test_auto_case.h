/******************************************************************************
 * Copyright (c) 2007-2024, NVIDIA CORPORATION. All rights reserved.
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
/// - MI_TEST_AUTO_CASE()     define a test case using a class generator
/// - MI_TEST_AUTO_FUNCTION() define a test case using a plain function
/// - get_argc(), get_argv()  access the test program's main() arguments

#ifndef BASE_SYSTEM_TEST_AUTO_CASE_H
#define BASE_SYSTEM_TEST_AUTO_CASE_H

#include "i_test_suite.h"
#include <memory>

namespace MI { namespace TEST {

extern int get_argc();
extern char const * const * get_argv();
extern Test_suite * get_master_test_suite();

struct Auto_test_case
{
    Auto_test_case(Test_case *);
};

}} // MI::TEST

#define MI_TEST_UNIQUE_SYMBOL(X)     MI_TEST_UNIQUE_SYMBOL2(X,__LINE__)
#define MI_TEST_UNIQUE_SYMBOL2(X,Y)  MI_TEST_UNIQUE_SYMBOL3(X,Y)
#define MI_TEST_UNIQUE_SYMBOL3(X,Y)  X##Y

#define MI_TEST_AUTO_CASE(test_case)                                                    \
    static void MI_TEST_UNIQUE_SYMBOL(test_case_runner)()                               \
    {                                                                                   \
       std::unique_ptr<MI::TEST::Test_case> p( test_case );                             \
       MI::TEST::Test_suite* s( dynamic_cast<MI::TEST::Test_suite*>(p.get()) );         \
       if (!s) p->run();                                                                \
       else                                                                             \
       {                                                                                \
            size_t failures( 0u );                                                      \
            s->run(failures);                                                           \
            MI_CHECK_MSG(failures == 0u, s->name());                                    \
       }                                                                                \
    }                                                                                   \
    static MI::TEST::Auto_test_case const MI_TEST_UNIQUE_SYMBOL(auto_add_test_case)     \
    (                                                                                   \
       new MI::TEST::Function_test_case("", MI_TEST_UNIQUE_SYMBOL(test_case_runner))    \
    )

#define MI_TEST_AUTO_FUNCTION(func)                                                     \
    static void func();                                                                 \
    static MI::TEST::Auto_test_case const                                               \
        MI_TEST_UNIQUE_SYMBOL(auto_add_ ## func)( MI_TEST_FUNCTION(func) );             \
    static void func()

#endif // BASE_SYSTEM_TEST_AUTO_CASE_H

