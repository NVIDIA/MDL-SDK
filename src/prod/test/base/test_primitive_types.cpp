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

#include <base/system/test/i_test_auto_case.h>

#include <cmath>
#include <cstdio>
#include <mi/base/assert.h>
#include <mi/base/types.h>

mi_static_assert(sizeof(mi::Sint8)      == 1);
mi_static_assert(sizeof(mi::Uint8)      == 1);
mi_static_assert(sizeof(mi::Sint16)     == 2);
mi_static_assert(sizeof(mi::Uint16)     == 2);
mi_static_assert(sizeof(mi::Sint32)     == 4);
mi_static_assert(sizeof(mi::Uint32)     == 4);
mi_static_assert(sizeof(mi::Sint64)     == 8);
mi_static_assert(sizeof(mi::Uint64)     == 8);
mi_static_assert(sizeof(mi::Float32)    == 4);
mi_static_assert(sizeof(mi::Float64)    == 8);


struct Numeric_traits_test_dummy
{
    int i = 0;
    bool operator==( const Numeric_traits_test_dummy& rhs) const { return i == rhs.i; }
};

std::ostringstream& operator<<( std::ostringstream& out, Numeric_traits_test_dummy val) {
    out << val.i;
    return out;
}

MI_TEST_AUTO_FUNCTION( test_pi_constants )
{
    MI_CHECK_CLOSE( sin( MI_PI),       0.0, 0.00001);
    MI_CHECK_CLOSE( sin( MI_PI_2),     1.0, 0.00001);
    MI_CHECK_CLOSE( sin( 2 * MI_PI_4), 1.0, 0.00001);
}

#define MI_CHECK_STATIC(expr) { bool b = (expr); MI_CHECK( b); }

MI_TEST_AUTO_FUNCTION( test_numeric_traits )
{
    {
        // generic case
        using Traits = mi::base::numeric_traits<Numeric_traits_test_dummy>;
        MI_CHECK_STATIC( ! Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        Numeric_traits_test_dummy null;
        MI_CHECK_EQUAL( Traits::min(), null);
        MI_CHECK_EQUAL( Traits::max(), null);
        MI_CHECK_EQUAL( Traits::negative_max(), null);
        MI_CHECK_EQUAL( Traits::infinity(), null);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), null);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), null);
    }
    {
        // Uint8 specialization
        using Traits = mi::base::numeric_traits<mi::Uint8>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), 0);
        MI_CHECK_EQUAL( Traits::max(), 255);
        MI_CHECK_EQUAL( Traits::negative_max(), 0);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Uint16 specialization
        using Traits = mi::base::numeric_traits<mi::Uint16>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), 0);
        MI_CHECK_EQUAL( Traits::max(), 65535);
        MI_CHECK_EQUAL( Traits::negative_max(), 0);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Uint32 specialization
        using Traits = mi::base::numeric_traits<mi::Uint32>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), 0);
        MI_CHECK_EQUAL( Traits::max(), 4294967295U);
        MI_CHECK_EQUAL( Traits::negative_max(), 0);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Uint64 specialization
        using Traits = mi::base::numeric_traits<mi::Uint64>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), 0);
        MI_CHECK_EQUAL( Traits::max(), 18446744073709551615ULL);
        MI_CHECK_EQUAL( Traits::negative_max(), 0);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Sint8 specialization
        using Traits = mi::base::numeric_traits<mi::Sint8>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), -128);
        MI_CHECK_EQUAL( Traits::max(), 127);
        MI_CHECK_EQUAL( Traits::negative_max(), -128);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Sint16 specialization
        using Traits = mi::base::numeric_traits<mi::Sint16>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), -32767 - 1);
        MI_CHECK_EQUAL( Traits::max(), 32767);
        MI_CHECK_EQUAL( Traits::negative_max(), -32767 - 1);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Sint32 specialization
        using Traits = mi::base::numeric_traits<mi::Sint32>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), -2147483647 - 1);
        MI_CHECK_EQUAL( Traits::max(),  2147483647);
        MI_CHECK_EQUAL( Traits::negative_max(), -2147483647 - 1);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Sint64 specialization
        using Traits = mi::base::numeric_traits<mi::Sint64>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( ! Traits::has_infinity);
        MI_CHECK_STATIC( ! Traits::has_quiet_NaN);
        MI_CHECK_STATIC( ! Traits::has_signaling_NaN);
        MI_CHECK_EQUAL( Traits::min(), -9223372036854775807LL - 1LL);
        MI_CHECK_EQUAL( Traits::max(),  9223372036854775807LL);
        MI_CHECK_EQUAL( Traits::negative_max(), -9223372036854775807LL - 1LL);
        MI_CHECK_EQUAL( Traits::infinity(), 0);
        MI_CHECK_EQUAL( Traits::quiet_NaN(), 0);
        MI_CHECK_EQUAL( Traits::signaling_NaN(), 0);
    }
    {
        // Float32 specialization
        using Traits = mi::base::numeric_traits<mi::Float32>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( Traits::has_infinity);
        MI_CHECK_STATIC( Traits::has_quiet_NaN);
        MI_CHECK_STATIC( Traits::has_signaling_NaN);
        MI_CHECK_GREATER( Traits::min(), 0.0f);
        MI_CHECK_LESS( Traits::min(), 1.0e-37f);
        MI_CHECK_EQUAL( Traits::max(), 3.402823466e+38f);
        MI_CHECK_EQUAL( Traits::negative_max(), - 3.402823466e+38f);
        MI_CHECK_GREATER( Traits::infinity(), Traits::max());
        MI_CHECK( Traits::quiet_NaN()     != Traits::quiet_NaN());
        MI_CHECK( Traits::signaling_NaN() != Traits::signaling_NaN());
    }
    {
        // Float64 specialization
        using Traits = mi::base::numeric_traits<mi::Float64>;
        MI_CHECK_STATIC( Traits::is_specialized);
        MI_CHECK_STATIC( Traits::has_infinity);
        MI_CHECK_STATIC( Traits::has_quiet_NaN);
        MI_CHECK_STATIC( Traits::has_signaling_NaN);
        MI_CHECK_GREATER( Traits::min(), 0.0);
        MI_CHECK_LESS( Traits::min(), 1.0e-307);
        MI_CHECK_EQUAL( Traits::max(), 1.7976931348623158e+308);
        MI_CHECK_EQUAL( Traits::negative_max(), - 1.7976931348623158e+308);
        MI_CHECK_GREATER( Traits::infinity(), Traits::max());
        MI_CHECK( Traits::quiet_NaN()     != Traits::quiet_NaN());
        MI_CHECK( Traits::signaling_NaN() != Traits::signaling_NaN());
    }
}

MI_TEST_AUTO_FUNCTION( test_printf_format_strings )
{
    char buffer[256];

    mi::Uint64 u64 = 42;
    snprintf( &buffer[0], 256, "%" MI_BASE_FMT_MI_UINT64, u64);

    mi::Sint64 s64 = 42;
    snprintf( &buffer[0], 256, "%" MI_BASE_FMT_MI_SINT64, s64);

    mi::Size s = 42;
    snprintf( &buffer[0], 256, "%" MI_BASE_FMT_MI_SIZE, s);

    mi::Difference d = -42;
    snprintf( &buffer[0], 256, "%" MI_BASE_FMT_MI_DIFFERENCE, d);
}
