/******************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for base/util/string_utils"

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include "i_string_lexicographic_cast.h"
#include "i_string_utils.h"

using namespace MI;

MI_TEST_AUTO_FUNCTION( test_lexicographic_cast )
{
    STLEXT::Likely<unsigned int> ui;
    STLEXT::Likely<signed int> si;
    STLEXT::Likely<size_t> s;
    STLEXT::Likely<bool> b;

    // Check valid values for unsigned int.
    ui = STRING::lexicographic_cast_s<unsigned int>("42");
    MI_CHECK(ui.get_status());
    MI_CHECK(*ui.get_ptr() == 42);

    // Check that underflow for unsigned int is detected.
    ui = STRING::lexicographic_cast_s<unsigned int>("-42");
    MI_CHECK(!ui.get_status());
    ui = STRING::lexicographic_cast_s<unsigned int>(" -42");
    MI_CHECK(!ui.get_status());

    // Check valid values for size_t.
    s = STRING::lexicographic_cast_s<size_t>("42");
    MI_CHECK(s.get_status());
    MI_CHECK(*s.get_ptr() == 42);

    // Check that underflow for size_t is detected.
    s = STRING::lexicographic_cast_s<size_t>("-42");
    MI_CHECK(!s.get_status());
    s = STRING::lexicographic_cast_s<size_t>(" -42");
    MI_CHECK(!s.get_status());

    // Check valid values for signed int.
    si = STRING::lexicographic_cast_s<signed int>("42");
    MI_CHECK(si.get_status());
    MI_CHECK(*si.get_ptr() == 42);

    // Check valid valued for signed int.
    si = STRING::lexicographic_cast_s<signed int>("-42");
    MI_CHECK(si.get_status());
    MI_CHECK(*si.get_ptr() == -42);

    // Check valid values for bool.
    b = STRING::lexicographic_cast_s<bool>("0");
    MI_CHECK(b.get_status());
    MI_CHECK(*b.get_ptr() == false);
    b = STRING::lexicographic_cast_s<bool>("1");
    MI_CHECK(b.get_status());
    MI_CHECK(*b.get_ptr() == true);

    // Check that underflow for bool is detected (already handled correctly
    // without the special check).
    b = STRING::lexicographic_cast_s<bool>("-1");
    MI_CHECK(!b.get_status());
}

void check_utf8_to_wchar_and_back( const char* input)
{
    std::wstring tmp = STRING::utf8_to_wchar( input);

    // Check that only 2 bytes are used, even if sizeof(wchar_t) == 4.
    for( const wchar_t& c: tmp)
        MI_CHECK( static_cast<unsigned int>( c) < 0x10000);

    std::string output = STRING::wchar_to_utf8( tmp.c_str());
    MI_CHECK_EQUAL( output, input);
}

void check_wchar_to_utf8_and_back( const wchar_t* input, const wchar_t* expected = nullptr)
{
    std::wstring expected_str = expected ? expected : input;

    std::string tmp = STRING::wchar_to_utf8( input);
    std::wstring output = STRING::utf8_to_wchar( tmp.c_str());
    MI_CHECK( output == expected_str);
}

void check_wchar_surrogate_to_utf8_and_back(
    const wchar_t* prefix, uint32_t symbol, wchar_t high_surrogate, wchar_t low_surrogate)
{
    std::wstring input = prefix;
    input += high_surrogate;
    input += low_surrogate;
    check_wchar_to_utf8_and_back( input.c_str());

#ifndef MI_PLATFORM_WINDOWS
    assert( sizeof(wchar_t) >= sizeof(uint32_t));
    std::wstring input2 = prefix;
    input2 += static_cast<wchar_t>( symbol);
    check_wchar_to_utf8_and_back( input2.c_str(), input.c_str());
#else // MI_PLATFORM_WINDOWS
    assert( sizeof(wchar_t) < sizeof(uint32_t));
#endif // MI_PLATFORM_WINDOWS
}

MI_TEST_AUTO_FUNCTION( test_utf8_and_wchar )
{
    check_utf8_to_wchar_and_back( "foo");                        // plain ASCII
    check_utf8_to_wchar_and_back( "first_bmp_\u0001");           // U+0001
    check_utf8_to_wchar_and_back( "last_ascii_\u007F");          // U+007F
    check_utf8_to_wchar_and_back( "first_non_ascii_\u0080");     // U+0080
    check_utf8_to_wchar_and_back( "u_umlaut_ü");                 // U+00FC
    check_utf8_to_wchar_and_back( "leo_♌");                     // U+264C
    check_utf8_to_wchar_and_back( "last_bmp_\uFFFF");            // U+FFFF
    check_utf8_to_wchar_and_back( "first_surrogate_\U00010000"); // U+10000 (U+D800, U+DC00)
    check_utf8_to_wchar_and_back( "unicorn_🦄");                 // U+1F984 (U+D83E, U+DD84)
    check_utf8_to_wchar_and_back( "g_clef_𝄞");                   // U+1D11E (U+D834, U+DD1E)
    check_utf8_to_wchar_and_back( "last_surrogate_\U0010FFFF");  // U+10FFFF (U+DBFF, U+DFFF)

    check_wchar_to_utf8_and_back( L"foo");
    check_wchar_to_utf8_and_back( L"first_bmp_\u0001");
    check_wchar_to_utf8_and_back( L"last_ascii_\u007F");
    check_wchar_to_utf8_and_back( L"first_non_ascii_\u0080");
    check_wchar_to_utf8_and_back( L"u_umlaut_\u00FC");
    check_wchar_to_utf8_and_back( L"leo_\u264C");
    check_wchar_to_utf8_and_back( L"last_bmp_\uFFFF");

    check_wchar_surrogate_to_utf8_and_back( L"first_surrogate_", 0x10000u, 0xD800u, 0xDC00u);
    check_wchar_surrogate_to_utf8_and_back( L"unicorn_", 0x1F984u, 0xD83Eu, 0xDD84u);
    check_wchar_surrogate_to_utf8_and_back( L"g_clef_", 0x1D11Eu, 0xD834u, 0xDD1Eu);
    check_wchar_surrogate_to_utf8_and_back( L"last_surrogate_", 0x10FFFFu, 0xDBFFu, 0xDFFFu);
}
