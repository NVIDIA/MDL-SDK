/******************************************************************************
 * Copyright (c) 2009-2023, NVIDIA CORPORATION. All rights reserved.
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

/**
 ** \file test_spectrum.cpp
 **/

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>
#include <mi/math/spectrum.h>

#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4305 ) // truncation from 'double' to 'mi::Float32'
#endif

///////////////////////////////////////////////////////////////////////////////
// Convenience operators for vector comparison and printing
///////////////////////////////////////////////////////////////////////////////

namespace mi { namespace math {
    //
    // Define ostream output operator for vectors so that test failures can be
    // printed properly.
    //
    std::ostream & operator<< (std::ostream & os, Spectrum const & c)
    {
	return os << "Spectrum(r = " << c[0]
		  << ", g = " << c[1]
		  << ", b = " << c[2]
		  << ")";
    }
}}

static const float eps = 1e-6f;

using namespace mi;
using namespace mi::math;

#define MI_CHECK_CLOSE_SPECTRUM(c1,c2,eps) \
    MI_CHECK_CLOSE((c1)[0], (c2)[0], (eps)); \
    MI_CHECK_CLOSE((c1)[1], (c2)[1], (eps)); \
    MI_CHECK_CLOSE((c1)[2], (c2)[2], (eps))

#define MI_CHECK_CLOSE_SPECTRUM_MSG(c1,c2,eps,s) \
    MI_CHECK_CLOSE_MSG((c1)[0], (c2)[0], (eps), (s)); \
    MI_CHECK_CLOSE_MSG((c1)[1], (c2)[1], (eps), (s)); \
    MI_CHECK_CLOSE_MSG((c1)[2], (c2)[2], (eps), (s))

void check_function (
        Spectrum (*f) (const Spectrum&),
        const char* s,
        const Spectrum& x,
        const Spectrum& y,
        float prec = eps)
{
    Spectrum z = f(x);
    MI_CHECK_CLOSE_SPECTRUM_MSG (z, y, prec, s);
}

MI_TEST_AUTO_FUNCTION( test_spectrum_constructors )
{
    // This test assumes three spectrum components.
    MI_CHECK_EQUAL(Spectrum::size(), 3);

    Spectrum c1;
    c1[0] = 0.0f;
    c1[1] = 0.25f;
    c1[2] = 0.5f;

    Spectrum c3( 0.5f);
    c3[0] = 0.0f;
    c3[1] = 0.25f;

    const Spectrum c4( 0.0f, 0.25f, 0.5f );

    Vector<Float32,4> vector( 0.0f, 0.25f, 0.5f, 0.75f);
    Spectrum c6( vector);

    MI_CHECK_CLOSE_SPECTRUM( c1, c3, eps);
    MI_CHECK_CLOSE_SPECTRUM( c1, c4, eps);
    MI_CHECK_CLOSE_SPECTRUM( c1, c6, eps);
    MI_CHECK_CLOSE_SPECTRUM( c3, c4, eps);
    MI_CHECK_CLOSE_SPECTRUM( c3, c6, eps);
    MI_CHECK_CLOSE_SPECTRUM( c4, c6, eps);

    MI_CHECK (c1.SIZE == 3);
    MI_CHECK (c1.size() == 3);
    MI_CHECK (c1.max_size() == 3);

    MI_CHECK (c1.begin() == & c1[0]);
    MI_CHECK (c4.begin() == & c4[0]);
}

MI_TEST_AUTO_FUNCTION( test_spectrum_assignment_operator )
{
    Spectrum c1( 0.0f, 0.25f, 0.5f);
    Spectrum c2;
    c2 = c1;

    MI_CHECK_CLOSE_SPECTRUM( c1, c2, eps);
}

MI_TEST_AUTO_FUNCTION( test_spectrum_access_operator_get_set )
{
    const Spectrum c1 (0.0f, 0.25f, 0.5f);

    Spectrum c2;
    c2[0] = c1[0];
    c2[1] = c1[1];
    c2[2] = c1[2];

    Spectrum c3;
    c3.set( 0, c1.get( 0));
    c3.set( 1, c1.get( 1));
    c3.set( 2, c1.get( 2));

    MI_CHECK_CLOSE_SPECTRUM( c1, c2, eps);
    MI_CHECK_CLOSE_SPECTRUM( c1, c3, eps);
    MI_CHECK_CLOSE_SPECTRUM( c2, c3, eps);
}

MI_TEST_AUTO_FUNCTION( test_spectrum_misc_functions )
{
    Spectrum c1( 0.2f, 0.4f, 0.3f);
    MI_CHECK_CLOSE( c1.linear_intensity(), 0.3f, eps);

    Spectrum c2( 0.0f, 0.0f, 0.0f);
    MI_CHECK( ! c1.is_black());
    MI_CHECK( c2.is_black());
}

MI_TEST_AUTO_FUNCTION( test_spectrum_free_comparison_operators )
{
    Spectrum c1 (0.0f, 0.0f, 0.0f);
    Spectrum c3 (0.0f, 0.0f, 1.0f);
    Spectrum c4 (0.0f, 1.0f, 1.0f);
    Spectrum c5 (1.0f, 1.0f, 1.0f);

    MI_CHECK ( c1 == c1);
    MI_CHECK ( c3 == c3);
    MI_CHECK ( c4 == c4);
    MI_CHECK ( c5 == c5);
    MI_CHECK ( ! ( c1 == c3));
    MI_CHECK ( ! ( c3 == c4));
    MI_CHECK ( ! ( c4 == c5));
    MI_CHECK ( ! ( c5 == c1));

    MI_CHECK ( c1 != c3);
    MI_CHECK ( c3 != c4);
    MI_CHECK ( c4 != c5);
    MI_CHECK ( c5 != c1);
    MI_CHECK ( ! ( c1 != c1));
    MI_CHECK ( ! ( c3 != c3));
    MI_CHECK ( ! ( c4 != c4));
    MI_CHECK ( ! ( c5 != c5));

    MI_CHECK ( c1 <= c1);
    MI_CHECK ( c3 <= c3);
    MI_CHECK ( c4 <= c4);
    MI_CHECK ( c5 <= c5);
    MI_CHECK ( c1 <= c3);
    MI_CHECK ( c3 <= c4);
    MI_CHECK ( c4 <= c5);
    MI_CHECK ( ! ( c3 <= c1));
    MI_CHECK ( ! ( c4 <= c3));
    MI_CHECK ( ! ( c5 <= c4));

    MI_CHECK ( c1 < c3);
    MI_CHECK ( c3 < c4);
    MI_CHECK ( c4 < c5);
    MI_CHECK ( ! ( c3 < c1));
    MI_CHECK ( ! ( c4 < c3));
    MI_CHECK ( ! ( c5 < c4));

    MI_CHECK ( c1 >= c1);
    MI_CHECK ( c3 >= c3);
    MI_CHECK ( c4 >= c4);
    MI_CHECK ( c5 >= c5);
    MI_CHECK ( c3 >= c1);
    MI_CHECK ( c4 >= c3);
    MI_CHECK ( c5 >= c4);
    MI_CHECK ( ! ( c1 >= c3));
    MI_CHECK ( ! ( c3 >= c4));
    MI_CHECK ( ! ( c4 >= c5));

    MI_CHECK ( c3 > c1);
    MI_CHECK ( c4 > c3);
    MI_CHECK ( c5 > c4);
    MI_CHECK ( ! ( c1 > c3));
    MI_CHECK ( ! ( c3 > c4));
    MI_CHECK ( ! ( c4 > c5));
}

MI_TEST_AUTO_FUNCTION( test_spectrum_free_arithmetic_operators )
{
    Spectrum c1( 0.1f, 0.2f, 0.3f);
    Spectrum c2( 0.8f, 0.7f, 0.6f);

    Spectrum c3 = c1 + c2;
    Spectrum c4( c1);
    c4 += c2;
    Spectrum c5( 0.9f, 0.9f, 0.9f);

    MI_CHECK_CLOSE_SPECTRUM( c3, c4, eps);
    MI_CHECK_CLOSE_SPECTRUM( c3, c5, eps);
    MI_CHECK_CLOSE_SPECTRUM( c4, c5, eps);

    Spectrum c6 = c1 - c2;
    Spectrum c7( c1);
    c7 -= c2;
    Spectrum c8( -0.7f, -0.5f, -0.3f);

    MI_CHECK_CLOSE_SPECTRUM( c6, c7, eps);
    MI_CHECK_CLOSE_SPECTRUM( c6, c8, eps);
    MI_CHECK_CLOSE_SPECTRUM( c7, c8, eps);

    Spectrum c9 = c1 * c2;
    Spectrum c10( c1);
    c10 *= c2;
    Spectrum c11( 0.08f, 0.14f, 0.18f);

    MI_CHECK_CLOSE_SPECTRUM( c9 , c10, eps);
    MI_CHECK_CLOSE_SPECTRUM( c9 , c11, eps);
    MI_CHECK_CLOSE_SPECTRUM( c10, c11, eps);

    Spectrum c12 = c1 / c2;
    Spectrum c13( c1);
    c13 /= c2;
    Spectrum c14( 0.125f, 0.285714f, 0.5f);

    MI_CHECK_CLOSE_SPECTRUM( c12, c13, eps);
    MI_CHECK_CLOSE_SPECTRUM( c12, c14, eps);
    MI_CHECK_CLOSE_SPECTRUM( c13, c14, eps);

    Spectrum c15 = -c1;
    Spectrum c16( -0.1f, -0.2f, -0.3f);

    MI_CHECK_CLOSE_SPECTRUM( c15, c16, eps);

    Spectrum c17 = c1 * 2.0f;
    Spectrum c18 = 2.0f * c1;
    Spectrum c19( c1);
    c19 *= 2.0f;
    Spectrum c20( 0.2f, 0.4f, 0.6f);

    MI_CHECK_CLOSE_SPECTRUM( c17, c18, eps);
    MI_CHECK_CLOSE_SPECTRUM( c17, c19, eps);
    MI_CHECK_CLOSE_SPECTRUM( c17, c20, eps);
    MI_CHECK_CLOSE_SPECTRUM( c18, c19, eps);
    MI_CHECK_CLOSE_SPECTRUM( c18, c20, eps);
    MI_CHECK_CLOSE_SPECTRUM( c19, c20, eps);

    Spectrum c21 = c1 / 2.0f;
    Spectrum c22( c1);
    c22 /= 2.0f;
    Spectrum c23( 0.05f, 0.1f, 0.15f);

    MI_CHECK_CLOSE_SPECTRUM( c21, c22, eps);
    MI_CHECK_CLOSE_SPECTRUM( c21, c23, eps);
    MI_CHECK_CLOSE_SPECTRUM( c22, c23, eps);
}

MI_TEST_AUTO_FUNCTION( test_spectrum_function_overloads )
{
    Spectrum c1( 0.1f, 0.2f, 0.3f);
    Spectrum c2( -0.1f, -0.2f, -0.3f);
    Spectrum c3 (-1.2f, -0.1f, 0.5f);

    check_function (abs, "abs", c2, c1);
    check_function (acos, "acos", c1, Spectrum(1.470628906, 1.369438406, 1.266103673));

    Spectrum c6 ( 0.0f, 0.0f, 0.0f);
    Spectrum c8 ( 0.0f, 0.0f, 1.0f);
    Spectrum c9 ( 0.0f, 1.0f, 0.0f);
    Spectrum c10( 1.0f, 0.0f, 0.0f);
    Spectrum c11( 1.0f, 1.0f, 1.0f);
    Spectrum c12( 1.0f, 1.0f, 0.0f);
    Spectrum c13( 1.0f, 0.0f, 1.0f);
    Spectrum c14( 0.0f, 1.0f, 1.0f);
    Spectrum c15( 1.0f, 1.0f, 1.0f);
    MI_CHECK( ! all( c6));
    MI_CHECK( ! all( c8));
    MI_CHECK( ! all( c9));
    MI_CHECK( ! all( c10));
    MI_CHECK( ! all( c12));
    MI_CHECK( ! all( c13));
    MI_CHECK( ! all( c14));
    MI_CHECK( all( c15));
    MI_CHECK( ! any( c6));
    MI_CHECK( any( c8));
    MI_CHECK( any( c9));
    MI_CHECK( any( c10));
    MI_CHECK( any( c11));
    MI_CHECK( any( c12));
    MI_CHECK( any( c13));
    MI_CHECK( any( c14));
    MI_CHECK( any( c15));

    check_function (asin, "asin", c1, Spectrum(.1001674212, .2013579208, .3046926540));
    check_function (atan, "atan", c1, Spectrum(.9966865249e-1, .1973955598, .2914567945));

    Spectrum c16 = atan2 ( c1, c3);
    Spectrum c17( 3.058451422, 2.034443936, .5404195003);
    MI_CHECK_CLOSE_SPECTRUM( c16, c17, eps);

    check_function (ceil, "ceil", c3, Spectrum ( -1.0f, 0.0f, 1.0f));

    Spectrum c18( -1.0f, 0.0f, 0.0f);
    Spectrum c19( 0.0f, 1.0f, 1.0f);
    Spectrum c20 = clamp (c3, c18, c19);
    Spectrum c21( -1.0f, 0.0f, 0.5f);
    MI_CHECK_CLOSE_SPECTRUM( c20, c21, eps);

    Spectrum c22 = clamp( c3, 0.0f, 1.0f);
    Spectrum c23( 0.0f, 0.0f, 0.5f);
    MI_CHECK_CLOSE_SPECTRUM( c22, c23, eps);

    check_function (cos, "cos", c1, Spectrum(.9950041653, .9800665778, .9553364891));
    check_function (degrees, "degrees", c1, Spectrum ( 5.729577950, 11.45915590, 17.18873385));

    Spectrum c24 = elementwise_max (c2, c3);
    Spectrum c25( -0.1f, -0.1f, 0.5f);
    MI_CHECK_CLOSE_SPECTRUM( c24, c25, eps);

    Spectrum c26 = elementwise_min (c2, c3);
    Spectrum c27( -1.2f, -0.2f, -0.3f);
    MI_CHECK_CLOSE_SPECTRUM( c26, c27, eps);

    check_function (exp, "exp", c1, Spectrum ( 1.105170918, 1.221402758, 1.349858808));
    check_function (exp2, "exp2", c1, Spectrum ( 1.071773463, 1.148698355, 1.231144413), 1e-2);
    check_function (floor, "floor", c3, Spectrum ( -2.0f, -1.0f, 0.0f));

    Spectrum c28 = fmod (c3, c2);
    Spectrum c29( 0.0f, -0.1f, 0.2f);
    MI_CHECK_CLOSE_SPECTRUM( c28, c29, eps);

    Spectrum c30 = fmod (c3, 0.5f);
    Spectrum c31( -0.2f, -0.1f, 0.0f);
    MI_CHECK_CLOSE_SPECTRUM( c30, c31, eps);

    check_function (frac, "frac", c3, Spectrum ( 0.8f, 0.9f, 0.5f));

    Spectrum c32 = gamma_correction (c1, 0.5f);
    Spectrum c33 = c1 * c1;
    MI_CHECK_CLOSE_SPECTRUM( c32, c33, 1e-2);

    Spectrum c34 = c1 + Spectrum( 1e-6);
    MI_CHECK( is_approx_equal( c34, c1, 1e-5));
    MI_CHECK( ! is_approx_equal( c34, c1, 1e-7));

    Spectrum c35 = lerp (c2, c3, c1);
    Spectrum c36( -0.21f, -0.18f, -0.06f);
    MI_CHECK_CLOSE_SPECTRUM( c35, c36, eps);

    Spectrum c37 = lerp (c2, c3, 0.5f);
    Spectrum c38( -0.65f, -0.15f, 0.1f);
    MI_CHECK_CLOSE_SPECTRUM( c37, c38, eps);

    check_function (log, "log", c1, Spectrum ( -2.302585093, -1.609437912, -1.203972804));
    check_function (log2, "log2", c1, Spectrum ( -3.321928095, -2.321928095, -1.736965594));
    check_function (log10, "log10", c1, Spectrum ( -1.000000000, -.6989700043, -.5228787453));

    Spectrum c39;
    Spectrum c40 = modf (c3, c39);
    Spectrum c41( -1.0f, 0.0f, 0.0f);
    Spectrum c42( -0.2f, -0.1f, 0.5f);
    MI_CHECK_CLOSE_SPECTRUM( c39, c41, eps);
    MI_CHECK_CLOSE_SPECTRUM( c40, c42, eps);

    Spectrum c43 = pow (c1, c3);
    Spectrum c44( 15.84893192, 1.174618943, .5477225575);
    MI_CHECK_CLOSE_SPECTRUM( c43, c44, eps);

    Spectrum c45 = pow (c1, 0.5f);
    Spectrum c46( .3162277660, .4472135955, .5477225575);
    MI_CHECK_CLOSE_SPECTRUM( c45, c46, eps);

    check_function (radians, "radians", c1, Spectrum ( .1745329252e-2, .3490658504e-2, .5235987758e-2));
    check_function (round, "round", c3, Spectrum ( -1.0f, 0.0f, 1.0f));
    check_function (rsqrt, "rsqrt", c1, Spectrum ( 3.162277660, 2.236067977, 1.825741858));
    check_function (saturate, "saturate", c3, Spectrum ( 0.0f, 0.0f, 0.5f));
    check_function (sign, "sign", c3, Spectrum ( -1.0f, -1.0f, 1.0f));
    check_function (sin, "sin", c1, Spectrum ( .9983341665e-1, .1986693308, .2955202067));

    Spectrum c47;
    Spectrum c48;
    sincos( c3, c47, c48);
    Spectrum c49 = sin( c3);
    Spectrum c50 = cos( c3);
    MI_CHECK_CLOSE_SPECTRUM ( c47, c49, eps);
    MI_CHECK_CLOSE_SPECTRUM ( c48, c50, eps);

    Spectrum c57( c1);
    c57[0] = -1.0f;
    Spectrum c51 = smoothstep (c2, c3, c57);
    Spectrum c52( 0.0f, 1.0f, .84375);
    MI_CHECK_CLOSE_SPECTRUM( c51, c52, eps);

    Spectrum c53 = smoothstep (c2, c3, 0.5f);
    Spectrum c54( 1.0f, 1.0f, 1.0f);
    MI_CHECK_CLOSE_SPECTRUM( c53, c54, eps);

    check_function (sqrt, "sqrt", c1, Spectrum ( .3162277660, .4472135955, .5477225575));

    Spectrum c55 = step( c3, c2);
    Spectrum c56( 1.0f, 0.0f, 0.0f);
    MI_CHECK_CLOSE_SPECTRUM (c55, c56, eps);

    check_function (tan, "tan", c1, Spectrum ( .1003346721, .2027100355, .3093362496));
}

