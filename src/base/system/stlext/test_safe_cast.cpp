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
/// \brief regression test safe_cast() template
///
/// Goes through an extensive list of tests to detect false negatives and false positives.

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>
#include <limits>
#include <mi/base/config.h>

//
// Import i_stlext_safe_cast.h with MI_CAST_ASSERT() redirected to a local
// version. NOTE: This code falls apart when these tests are executed in a
// multi-threaded environment. This is unlikely to happen, though.
//
static bool safe_cast_failure;
#define MI_CAST_ASSERT(X) if (!(X)) safe_cast_failure = true
#include "i_stlext_safe_cast.h"

template <typename TO_T, typename FROM_T>
static bool test_safe_cast(FROM_T v)
{
    safe_cast_failure = false;
    MI::STLEXT::safe_cast<TO_T>( v );
    return !safe_cast_failure;
}

#define MI_SUCCEED_SAFE_CAST_FROM_TO(limit, to_type, from_type)                                 \
    MI_TEST_AUTO_FUNCTION( safe_cast_ ## limit ## imal_ ## from_type ## _to_ ## to_type )       \
    {                                                                                           \
        MI_CHECK( test_safe_cast<to_type>( std::numeric_limits<from_type>::limit() ) );       \
    }

#define MI_FAIL_SAFE_CAST_FROM_TO(limit, to_type, from_type)                                    \
    MI_TEST_AUTO_FUNCTION( do_not_safe_cast_ ## limit ## imal_ ## from_type ## _to_ ## to_type )\
    {                                                                                           \
        MI_CHECK( !test_safe_cast<to_type>( std::numeric_limits<from_type>::limit() ) );      \
    }

#define MI_TEST_SAFE_CAST_MIN_MAX(type)                 \
    MI_SUCCEED_SAFE_CAST_FROM_TO(min, type, type)       \
    MI_SUCCEED_SAFE_CAST_FROM_TO(max, type, type)

#define MI_TEST_SAFE_CAST_SIGNED_UNSIGNED_TYPE(type)    \
    MI_TEST_SAFE_CAST_MIN_MAX(type)                     \
    MI_TEST_SAFE_CAST_MIN_MAX(signed_ ## type)          \
    MI_TEST_SAFE_CAST_MIN_MAX(unsigned_ ## type)

#define MI_TEST_SAFE_CAST_TYPE(type)                    \
    typedef signed type     signed_ ## type;            \
    typedef unsigned type   unsigned_ ## type;          \
    MI_TEST_SAFE_CAST_SIGNED_UNSIGNED_TYPE(type)

MI_TEST_SAFE_CAST_MIN_MAX(float)
MI_TEST_SAFE_CAST_MIN_MAX(double)
MI_TEST_SAFE_CAST_TYPE(char)
MI_TEST_SAFE_CAST_TYPE(short)
MI_TEST_SAFE_CAST_TYPE(int)
MI_TEST_SAFE_CAST_TYPE(long)
using long_long = long long;
using signed_long_long = long long;
using unsigned_long_long = unsigned long long;
MI_TEST_SAFE_CAST_SIGNED_UNSIGNED_TYPE(long_long)

MI_SUCCEED_SAFE_CAST_FROM_TO(max, int, char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, int, short)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, int, unsigned_char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, int, unsigned_short)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, long, char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, long, int)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, long, short)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, long, unsigned_char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, long, unsigned_short)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, short, char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, short, unsigned_char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, unsigned_int, unsigned_char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, unsigned_int, unsigned_short)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, unsigned_long, unsigned_char)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, unsigned_long, unsigned_int)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, unsigned_long, unsigned_short)
MI_SUCCEED_SAFE_CAST_FROM_TO(max, unsigned_short, unsigned_char)
MI_SUCCEED_SAFE_CAST_FROM_TO(min, int, char)
MI_SUCCEED_SAFE_CAST_FROM_TO(min, int, short)
MI_SUCCEED_SAFE_CAST_FROM_TO(min, long, char)
MI_SUCCEED_SAFE_CAST_FROM_TO(min, long, int)
MI_SUCCEED_SAFE_CAST_FROM_TO(min, long, short)
MI_SUCCEED_SAFE_CAST_FROM_TO(min, short, char)

MI_FAIL_SAFE_CAST_FROM_TO(max, char, unsigned_int)
MI_FAIL_SAFE_CAST_FROM_TO(max, char, unsigned_long)
MI_FAIL_SAFE_CAST_FROM_TO(max, char, unsigned_short)
MI_FAIL_SAFE_CAST_FROM_TO(max, short, unsigned_int)
MI_FAIL_SAFE_CAST_FROM_TO(max, short, unsigned_long)
#if !defined(MI_PLATFORM_LINUX) || !defined(MI_ARCH_ARM_64)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_char, char)
#endif
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_char, int)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_char, long)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_char, short)
#if !defined(MI_PLATFORM_LINUX) || !defined(MI_ARCH_ARM_64)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_int, char)
#endif
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_int, int)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_int, long)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_int, short)
#if !defined(MI_PLATFORM_LINUX) || !defined(MI_ARCH_ARM_64)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_long, char)
#endif
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_long, int)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_long, long)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_long, short)
#if !defined(MI_PLATFORM_LINUX) || !defined(MI_ARCH_ARM_64)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_short, char)
#endif
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_short, int)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_short, long)
MI_FAIL_SAFE_CAST_FROM_TO(min, unsigned_short, short)

MI_TEST_AUTO_FUNCTION( verify_handpicked_safe_casting_test_cases )
{
    MI_CHECK( !test_safe_cast<char>( static_cast<int>(std::numeric_limits<char>::max())+1 ) );
    MI_CHECK( !test_safe_cast<char>( static_cast<int>(std::numeric_limits<char>::min())-1 ) );
    MI_CHECK( !test_safe_cast<char>( static_cast<long int>(std::numeric_limits<char>::max())+1 ) );
    MI_CHECK( !test_safe_cast<char>( static_cast<long int>(std::numeric_limits<char>::min())-1 ) );
    MI_CHECK( !test_safe_cast<char>( static_cast<short>(std::numeric_limits<char>::max())+1 ) );
    MI_CHECK( !test_safe_cast<char>( static_cast<short>(std::numeric_limits<char>::min())-1 ) );
    MI_CHECK( !test_safe_cast<short>( static_cast<int>(std::numeric_limits<short>::max())+1 ) );
    MI_CHECK( !test_safe_cast<short>( static_cast<int>(std::numeric_limits<short>::min())-1 ) );
    MI_CHECK( !test_safe_cast<short>( static_cast<long int>(std::numeric_limits<short>::max())+1 ) );
    MI_CHECK( !test_safe_cast<short>( static_cast<long int>(std::numeric_limits<short>::min())-1 ) );
    MI_CHECK( !test_safe_cast<unsigned char>( static_cast<unsigned int>(std::numeric_limits<unsigned char>::max())+1 ) );
    MI_CHECK( !test_safe_cast<unsigned char>( static_cast<unsigned long int>(std::numeric_limits<unsigned char>::max())+1 ) );
    MI_CHECK( !test_safe_cast<unsigned char>( static_cast<unsigned short>(std::numeric_limits<unsigned char>::max())+1 ) );
    MI_CHECK( !test_safe_cast<unsigned short>( static_cast<unsigned int>(std::numeric_limits<unsigned short>::max())+1 ) );
    MI_CHECK( !test_safe_cast<unsigned short>( static_cast<unsigned long int>(std::numeric_limits<unsigned short>::max())+1 ) );
    MI_CHECK( test_safe_cast<char>(  (signed char) -3) );
    MI_CHECK( test_safe_cast<int>(  (signed char) -3) );
    MI_CHECK( test_safe_cast<long long int>(  (signed char) -3) );
    MI_CHECK( test_safe_cast<short>(  (signed char) -3) );
    MI_CHECK( test_safe_cast<signed char>(  (signed char) -3) );

    if ( sizeof(long int) != sizeof(int) )
    {
        MI_CHECK( !test_safe_cast<int>( static_cast<long int>(std::numeric_limits<int>::min())-1 ) );
        MI_CHECK( !test_safe_cast<int>( static_cast<long int>(std::numeric_limits<int>::max())+1 ) );
        MI_CHECK( !test_safe_cast<unsigned int>( static_cast<unsigned long int>(std::numeric_limits<unsigned int>::max())+1 ) );
        MI_CHECK( test_safe_cast<long int>( std::numeric_limits<unsigned int>::max() ) );
        MI_CHECK( !test_safe_cast<int>( std::numeric_limits<unsigned long int>::max() ) );
        MI_CHECK( test_safe_cast<long int>( std::numeric_limits<long int>::min() ) );
        MI_CHECK( test_safe_cast<long int>( std::numeric_limits<long int>::max() ) );
    }
}
