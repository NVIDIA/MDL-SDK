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

/**
 ** \file test_color.cpp
 **/

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>
#include <mi/math/color.h>

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
    std::ostream & operator<< (std::ostream & os, Color const & c)
    {
	return os << "Color(r = " << c.r
		  << ", g = " << c.g
		  << ", b = " << c.b
		  << ", a = " << c.a
		  << ")";
    }
}}

static const float eps = 1e-6f;

using namespace mi;
using namespace mi::math;

#define MI_CHECK_CLOSE_COLOR(c1,c2,eps) \
    MI_CHECK_CLOSE((c1).r, (c2).r, (eps)); \
    MI_CHECK_CLOSE((c1).g, (c2).g, (eps)); \
    MI_CHECK_CLOSE((c1).b, (c2).b, (eps)); \
    MI_CHECK_CLOSE((c1).a, (c2).a, (eps))

#define MI_CHECK_CLOSE_COLOR_MSG(c1,c2,eps,s) \
    MI_CHECK_CLOSE_MSG((c1).r, (c2).r, (eps), (s)); \
    MI_CHECK_CLOSE_MSG((c1).g, (c2).g, (eps), (s)); \
    MI_CHECK_CLOSE_MSG((c1).b, (c2).b, (eps), (s)); \
    MI_CHECK_CLOSE_MSG((c1).a, (c2).a, (eps), (s))

void check_function (
        Color (*f) (const Color&),
        const char* s,
        const Color& x,
        const Color& y,
        float prec = eps)
{
    Color z = f(x);
    MI_CHECK_CLOSE_COLOR_MSG (z, y, prec, s);
}

MI_TEST_AUTO_FUNCTION( test_color_constructors )
{
    Color c1;
    c1[0] = 0.0f;
    c1[1] = 0.25f;
    c1[2] = 0.5f;
    c1[3] = 0.75f;

    Color_struct cs;
    cs.r = 0.0f;
    cs.g = 0.25f;
    cs.b = 0.5f;
    cs.a = 0.75f;
    Color c2( cs);

    Color c3( 0.5f);
    c3[0] = 0.0f;
    c3[1] = 0.25f;
    c3[3] = 0.75f;

    const Color c4( 0.0f, 0.25f, 0.5f, 0.75f);

    Float32 array[4] = {0.0f, 0.25f, 0.5f, 0.75f};
    Color c5( array);

    Vector<Float32,4> vector( 0.0f, 0.25f, 0.5f, 0.75f);
    Color c6( vector);

    Color c7;
    c7.r = 0.0f;
    c7.g = 0.25f;
    c7.b = 0.5f;
    c7.a = 0.75;

    MI_CHECK_CLOSE_COLOR( c1, c2, eps);
    MI_CHECK_CLOSE_COLOR( c1, c3, eps);
    MI_CHECK_CLOSE_COLOR( c1, c4, eps);
    MI_CHECK_CLOSE_COLOR( c1, c5, eps);
    MI_CHECK_CLOSE_COLOR( c1, c6, eps);
    MI_CHECK_CLOSE_COLOR( c1, c7, eps);
    MI_CHECK_CLOSE_COLOR( c2, c3, eps);
    MI_CHECK_CLOSE_COLOR( c2, c4, eps);
    MI_CHECK_CLOSE_COLOR( c2, c5, eps);
    MI_CHECK_CLOSE_COLOR( c2, c6, eps);
    MI_CHECK_CLOSE_COLOR( c2, c7, eps);
    MI_CHECK_CLOSE_COLOR( c3, c4, eps);
    MI_CHECK_CLOSE_COLOR( c3, c5, eps);
    MI_CHECK_CLOSE_COLOR( c3, c6, eps);
    MI_CHECK_CLOSE_COLOR( c3, c7, eps);
    MI_CHECK_CLOSE_COLOR( c4, c5, eps);
    MI_CHECK_CLOSE_COLOR( c4, c6, eps);
    MI_CHECK_CLOSE_COLOR( c4, c7, eps);
    MI_CHECK_CLOSE_COLOR( c5, c6, eps);
    MI_CHECK_CLOSE_COLOR( c5, c7, eps);
    MI_CHECK_CLOSE_COLOR( c6, c7, eps);

    MI_CHECK (c1.SIZE == 4);
    MI_CHECK (c1.size() == 4);
    MI_CHECK (c1.max_size() == 4);

    MI_CHECK (c1.begin() == & c1.r);
    MI_CHECK (c4.begin() == & c4.r);
    MI_CHECK (c1.end()-1 == & c1.a);
    MI_CHECK (c4.end()-1 == & c4.a);
}

MI_TEST_AUTO_FUNCTION( test_color_assignment_operator )
{
    Color c1( 0.0f, 0.25f, 0.5f, 0.75f);
    Color c2;
    c2 = c1;
    Vector<Float32,4> vector (0.0f, 0.25f, 0.5f, 0.75f);
    Color c3;
    c3 = vector;

    MI_CHECK_CLOSE_COLOR( c1, c2, eps);
    MI_CHECK_CLOSE_COLOR( c1, c3, eps);
    MI_CHECK_CLOSE_COLOR( c2, c3, eps);
}

MI_TEST_AUTO_FUNCTION( test_color_access_operator_get_set )
{
    const Color c1 (0.0f, 0.25f, 0.5f, 1.0f);

    Color c2;
    c2[0] = c1[0];
    c2[1] = c1[1];
    c2[2] = c1[2];
    c2[3] = c1[3];

    Color c3;
    c3.set( 0, c1.get( 0));
    c3.set( 1, c1.get( 1));
    c3.set( 2, c1.get( 2));
    c3.set( 3, c1.get( 3));

    MI_CHECK_CLOSE_COLOR( c1, c2, eps);
    MI_CHECK_CLOSE_COLOR( c1, c3, eps);
    MI_CHECK_CLOSE_COLOR( c2, c3, eps);
}

MI_TEST_AUTO_FUNCTION( test_color_misc_functions )
{
    Color c1( 0.2f, 0.4f, 0.3f, 0.7f);
    MI_CHECK_CLOSE( c1.linear_intensity(), 0.3f, eps);
    MI_CHECK_CLOSE( c1.ntsc_intensity(), 0.3288f, eps);

    Color c2( 0.0f, 0.0f, 0.0f, 0.0f);
    MI_CHECK( ! c1.is_black());
    MI_CHECK( c2.is_black());

    Color c3( 0.1f, 0.7f, 1.3f, 0.5f);
    Color c4( -0.1f, 0.2f, -0.3f, -0.5f);
    Color c5( -0.1f, 0.2f, -0.3f, 1.5f);
    Color c6 = c3.clip (CLIP_RGB);
    Color c7 = c4.clip (CLIP_RGB);
    Color c8 = c5.clip (CLIP_RGB);
    Color c9( 0.1f, 0.7f, 1.0f, 1.0f);
    Color c10( 0.0f, 0.2f, 0.0f, 0.2f);
    Color c11( 0.0f, 0.2f, 0.0f, 1.0f);
    MI_CHECK_CLOSE_COLOR( c6, c9, eps);
    MI_CHECK_CLOSE_COLOR( c7, c10, eps);
    MI_CHECK_CLOSE_COLOR( c8, c11, eps);

    Color c12( 0.1f, 0.5f, 0.9f, -0.5f);
    Color c13( 0.1f, 0.5f, 0.9f, 1.5f);
    Color c14( 0.1f, 0.5f, 0.9f, 0.2f);
    Color c15 = c12.clip (CLIP_ALPHA);
    Color c16 = c13.clip (CLIP_ALPHA);
    Color c17 = c14.clip (CLIP_ALPHA);
    Color c18( 0.0f, 0.0f, 0.0f, 0.0f);
    Color c19( 0.1f, 0.5f, 0.9f, 1.0f);
    Color c20( 0.1f, 0.2f, 0.2f, 0.2f);
    MI_CHECK_CLOSE_COLOR( c15, c18, eps);
    MI_CHECK_CLOSE_COLOR( c16, c19, eps);
    MI_CHECK_CLOSE_COLOR( c17, c20, eps);

    Color c21( 0.1f, -0.2f, 0.5f, 1.1f);
    Color c22 = c21.clip (CLIP_RAW);
    Color c23( 0.1f, 0.0f,  0.5f, 1.0f);
    MI_CHECK_CLOSE_COLOR( c22, c23, eps);
}

MI_TEST_AUTO_FUNCTION( test_color_free_comparison_operators )
{
    Color c1 (0.0f, 0.0f, 0.0f, 0.0f);
    Color c2 (0.0f, 0.0f, 0.0f, 1.0f);
    Color c3 (0.0f, 0.0f, 1.0f, 1.0f);
    Color c4 (0.0f, 1.0f, 1.0f, 1.0f);
    Color c5 (1.0f, 1.0f, 1.0f, 1.0f);

    MI_CHECK ( c1 == c1);
    MI_CHECK ( c2 == c2);
    MI_CHECK ( c3 == c3);
    MI_CHECK ( c4 == c4);
    MI_CHECK ( c5 == c5);
    MI_CHECK ( ! ( c1 == c2));
    MI_CHECK ( ! ( c2 == c3));
    MI_CHECK ( ! ( c3 == c4));
    MI_CHECK ( ! ( c4 == c5));
    MI_CHECK ( ! ( c5 == c1));

    MI_CHECK ( c1 != c2);
    MI_CHECK ( c2 != c3);
    MI_CHECK ( c3 != c4);
    MI_CHECK ( c4 != c5);
    MI_CHECK ( c5 != c1);
    MI_CHECK ( ! ( c1 != c1));
    MI_CHECK ( ! ( c2 != c2));
    MI_CHECK ( ! ( c3 != c3));
    MI_CHECK ( ! ( c4 != c4));
    MI_CHECK ( ! ( c5 != c5));

    MI_CHECK ( c1 <= c1);
    MI_CHECK ( c2 <= c2);
    MI_CHECK ( c3 <= c3);
    MI_CHECK ( c4 <= c4);
    MI_CHECK ( c5 <= c5);
    MI_CHECK ( c1 <= c2);
    MI_CHECK ( c2 <= c3);
    MI_CHECK ( c3 <= c4);
    MI_CHECK ( c4 <= c5);
    MI_CHECK ( ! ( c2 <= c1));
    MI_CHECK ( ! ( c3 <= c2));
    MI_CHECK ( ! ( c4 <= c3));
    MI_CHECK ( ! ( c5 <= c4));

    MI_CHECK ( c1 < c2);
    MI_CHECK ( c2 < c3);
    MI_CHECK ( c3 < c4);
    MI_CHECK ( c4 < c5);
    MI_CHECK ( ! ( c2 < c1));
    MI_CHECK ( ! ( c3 < c2));
    MI_CHECK ( ! ( c4 < c3));
    MI_CHECK ( ! ( c5 < c4));

    MI_CHECK ( c1 >= c1);
    MI_CHECK ( c2 >= c2);
    MI_CHECK ( c3 >= c3);
    MI_CHECK ( c4 >= c4);
    MI_CHECK ( c5 >= c5);
    MI_CHECK ( c2 >= c1);
    MI_CHECK ( c3 >= c2);
    MI_CHECK ( c4 >= c3);
    MI_CHECK ( c5 >= c4);
    MI_CHECK ( ! ( c1 >= c2));
    MI_CHECK ( ! ( c2 >= c3));
    MI_CHECK ( ! ( c3 >= c4));
    MI_CHECK ( ! ( c4 >= c5));

    MI_CHECK ( c2 > c1);
    MI_CHECK ( c3 > c2);
    MI_CHECK ( c4 > c3);
    MI_CHECK ( c5 > c4);
    MI_CHECK ( ! ( c1 > c2));
    MI_CHECK ( ! ( c2 > c3));
    MI_CHECK ( ! ( c3 > c4));
    MI_CHECK ( ! ( c4 > c5));
}

MI_TEST_AUTO_FUNCTION( test_color_free_arithmetic_operators )
{
    Color c1( 0.1f, 0.2f, 0.3f, 0.4f);
    Color c2( 0.8f, 0.7f, 0.6f, 0.5f);

    Color c3 = c1 + c2;
    Color c4( c1);
    c4 += c2;
    Color c5( 0.9f, 0.9f, 0.9f, 0.9f);

    MI_CHECK_CLOSE_COLOR( c3, c4, eps);
    MI_CHECK_CLOSE_COLOR( c3, c5, eps);
    MI_CHECK_CLOSE_COLOR( c4, c5, eps);

    Color c6 = c1 - c2;
    Color c7( c1);
    c7 -= c2;
    Color c8( -0.7f, -0.5f, -0.3f, -0.1f);

    MI_CHECK_CLOSE_COLOR( c6, c7, eps);
    MI_CHECK_CLOSE_COLOR( c6, c8, eps);
    MI_CHECK_CLOSE_COLOR( c7, c8, eps);

    Color c9 = c1 * c2;
    Color c10( c1);
    c10 *= c2;
    Color c11( 0.08f, 0.14f, 0.18f, 0.2f);

    MI_CHECK_CLOSE_COLOR( c9 , c10, eps);
    MI_CHECK_CLOSE_COLOR( c9 , c11, eps);
    MI_CHECK_CLOSE_COLOR( c10, c11, eps);

    Color c12 = c1 / c2;
    Color c13( c1);
    c13 /= c2;
    Color c14( 0.125f, 0.285714f, 0.5f, 0.8f);

    MI_CHECK_CLOSE_COLOR( c12, c13, eps);
    MI_CHECK_CLOSE_COLOR( c12, c14, eps);
    MI_CHECK_CLOSE_COLOR( c13, c14, eps);

    Color c15 = -c1;
    Color c16( -0.1f, -0.2f, -0.3f, -0.4f);

    MI_CHECK_CLOSE_COLOR( c15, c16, eps);

    Color c17 = c1 * 2.0f;
    Color c18 = 2.0f * c1;
    Color c19( c1);
    c19 *= 2.0f;
    Color c20( 0.2f, 0.4f, 0.6f, 0.8f);

    MI_CHECK_CLOSE_COLOR( c17, c18, eps);
    MI_CHECK_CLOSE_COLOR( c17, c19, eps);
    MI_CHECK_CLOSE_COLOR( c17, c20, eps);
    MI_CHECK_CLOSE_COLOR( c18, c19, eps);
    MI_CHECK_CLOSE_COLOR( c18, c20, eps);
    MI_CHECK_CLOSE_COLOR( c19, c20, eps);

    Color c21 = c1 / 2.0f;
    Color c22( c1);
    c22 /= 2.0f;
    Color c23( 0.05f, 0.1f, 0.15f, 0.2f);

    MI_CHECK_CLOSE_COLOR( c21, c22, eps);
    MI_CHECK_CLOSE_COLOR( c21, c23, eps);
    MI_CHECK_CLOSE_COLOR( c22, c23, eps);
}

MI_TEST_AUTO_FUNCTION( test_color_function_overloads )
{
    Color c1( 0.1f, 0.2f, 0.3f, 0.4f);
    Color c2( -0.1f, -0.2f, -0.3f, -0.4f);
    Color c3 (-1.2f, -0.1f, 0.5f, 1.6f);

    check_function (abs, "abs", c2, c1);
    check_function (acos, "acos", c1, Color ( 1.470628906, 1.369438406, 1.266103673, 1.159279481));

    Color c6 ( 0.0f, 0.0f, 0.0f, 0.0f);
    Color c7 ( 0.0f, 0.0f, 0.0f, 1.0f);
    Color c8 ( 0.0f, 0.0f, 1.0f, 0.0f);
    Color c9 ( 0.0f, 1.0f, 0.0f, 0.0f);
    Color c10( 1.0f, 0.0f, 0.0f, 0.0f);
    Color c11( 1.0f, 1.0f, 1.0f, 0.0f);
    Color c12( 1.0f, 1.0f, 0.0f, 1.0f);
    Color c13( 1.0f, 0.0f, 1.0f, 1.0f);
    Color c14( 0.0f, 1.0f, 1.0f, 1.0f);
    Color c15( 1.0f, 1.0f, 1.0f, 1.0f);
    MI_CHECK( ! all( c6));
    MI_CHECK( ! all( c7));
    MI_CHECK( ! all( c8));
    MI_CHECK( ! all( c9));
    MI_CHECK( ! all( c10));
    MI_CHECK( ! all( c11));
    MI_CHECK( ! all( c12));
    MI_CHECK( ! all( c13));
    MI_CHECK( ! all( c14));
    MI_CHECK( all( c15));
    MI_CHECK( ! any( c6));
    MI_CHECK( any( c7));
    MI_CHECK( any( c8));
    MI_CHECK( any( c9));
    MI_CHECK( any( c10));
    MI_CHECK( any( c11));
    MI_CHECK( any( c12));
    MI_CHECK( any( c13));
    MI_CHECK( any( c14));
    MI_CHECK( any( c15));

    check_function (asin, "asin", c1, Color ( .1001674212, .2013579208, .3046926540, .4115168461));
    check_function (atan, "atan", c1, Color ( .9966865249e-1, .1973955598, .2914567945, .3805063771));

    Color c16 = atan2 ( c1, c3);
    Color c17( 3.058451422, 2.034443936, .5404195003, .2449786631);
    MI_CHECK_CLOSE_COLOR( c16, c17, eps);

    check_function (ceil, "ceil", c3, Color ( -1.0f, 0.0f, 1.0f, 2.0f));

    Color c18( -1.0f, 0.0f, 0.0f, 0.0f);
    Color c19( 0.0f, 1.0f, 1.0f, 1.0f);
    Color c20 = clamp (c3, c18, c19);
    Color c21( -1.0f, 0.0f, 0.5f, 1.0f);
    MI_CHECK_CLOSE_COLOR( c20, c21, eps);

    Color c22 = clamp( c3, 0.0f, 1.0f);
    Color c23( 0.0f, 0.0f, 0.5f, 1.0f);
    MI_CHECK_CLOSE_COLOR( c22, c23, eps);

    check_function (cos, "cos", c1, Color ( .9950041653, .9800665778, .9553364891, .9210609940));
    check_function (degrees, "degrees", c1, Color ( 5.729577950, 11.45915590, 17.18873385, 22.91831180));

    Color c24 = elementwise_max (c2, c3);
    Color c25( -0.1f, -0.1f, 0.5f, 1.6f);
    MI_CHECK_CLOSE_COLOR( c24, c25, eps);

    Color c26 = elementwise_min (c2, c3);
    Color c27( -1.2f, -0.2f, -0.3f, -0.4f);
    MI_CHECK_CLOSE_COLOR( c26, c27, eps);

    check_function (exp, "exp", c1, Color ( 1.105170918, 1.221402758, 1.349858808, 1.491824698));
    check_function (exp2, "exp2", c1, Color ( 1.071773463, 1.148698355, 1.231144413, 1.319507911), 1e-2);
    check_function (floor, "floor", c3, Color ( -2.0f, -1.0f, 0.0f, 1.0f));

    Color c28 = fmod (c3, c2);
    Color c29( 0.0f, -0.1f, 0.2f, 0.0f);
    MI_CHECK_CLOSE_COLOR( c28, c29, eps);

    Color c30 = fmod (c3, 0.5f);
    Color c31( -0.2f, -0.1f, 0.0f, 0.1f);
    MI_CHECK_CLOSE_COLOR( c30, c31, eps);

    check_function (frac, "frac", c3, Color ( 0.8f, 0.9f, 0.5f, 0.6f));

    Color c32 = gamma_correction (c1, 0.5f);
    Color c33 = c1 * c1;
    MI_CHECK_CLOSE_COLOR( c32, c33, 1e-2);

    Color c34 = c1 + Color( 1e-6);
    MI_CHECK( is_approx_equal( c34, c1, 1e-5));
    MI_CHECK( ! is_approx_equal( c34, c1, 1e-7));

    Color c35 = lerp (c2, c3, c1);
    Color c36( -0.21f, -0.18f, -0.06f, 0.4f);
    MI_CHECK_CLOSE_COLOR( c35, c36, eps);

    Color c37 = lerp (c2, c3, 0.5f);
    Color c38( -0.65f, -0.15f, 0.1f, 0.6f);
    MI_CHECK_CLOSE_COLOR( c37, c38, eps);

    check_function (log, "log", c1, Color ( -2.302585093, -1.609437912, -1.203972804, -.9162907319));
    check_function (log2, "log2", c1, Color ( -3.321928095, -2.321928095, -1.736965594, -1.321928095));
    check_function (log10, "log10", c1, Color ( -1.000000000, -.6989700043, -.5228787453, -.3979400087));

    Color c39;
    Color c40 = modf (c3, c39);
    Color c41( -1.0f, 0.0f, 0.0f, 1.0f);
    Color c42( -0.2f, -0.1f, 0.5f, 0.6f);
    MI_CHECK_CLOSE_COLOR( c39, c41, eps);
    MI_CHECK_CLOSE_COLOR( c40, c42, eps);

    Color c43 = pow (c1, c3);
    Color c44( 15.84893192, 1.174618943, .5477225575, .2308319849);
    MI_CHECK_CLOSE_COLOR( c43, c44, eps);

    Color c45 = pow (c1, 0.5f);
    Color c46( .3162277660, .4472135955, .5477225575, .6324555320);
    MI_CHECK_CLOSE_COLOR( c45, c46, eps);

    check_function (radians, "radians", c1, Color ( .1745329252e-2, .3490658504e-2, .5235987758e-2, .6981317008e-2));
    check_function (round, "round", c3, Color ( -1.0f, 0.0f, 1.0f, 2.0f));
    check_function (rsqrt, "rsqrt", c1, Color ( 3.162277660, 2.236067977, 1.825741858, 1.581138830));
    check_function (saturate, "saturate", c3, Color ( 0.0f, 0.0f, 0.5f, 1.0f));
    check_function (sign, "sign", c3, Color ( -1.0f, -1.0f, 1.0f, 1.0f));
    check_function (sin, "sin", c1, Color ( .9983341665e-1, .1986693308, .2955202067, .3894183423));

    Color c47;
    Color c48;
    sincos( c3, c47, c48);
    Color c49 = sin( c3);
    Color c50 = cos( c3);
    MI_CHECK_CLOSE_COLOR ( c47, c49, eps);
    MI_CHECK_CLOSE_COLOR ( c48, c50, eps);

    Color c57( c1);
    c57[0] = -1.0f;
    Color c51 = smoothstep (c2, c3, c57);
    Color c52( 0.0f, 1.0f, .84375, .352f);
    MI_CHECK_CLOSE_COLOR( c51, c52, eps);

    Color c53 = smoothstep (c2, c3, 0.5f);
    Color c54( 1.0f, 1.0f, 1.0f, .42525f);
    MI_CHECK_CLOSE_COLOR( c53, c54, eps);

    check_function (sqrt, "sqrt", c1, Color ( .3162277660, .4472135955, .5477225575, .6324555320));

    Color c55 = step( c3, c2);
    Color c56( 1.0f, 0.0f, 0.0f, 0.0f);
    MI_CHECK_CLOSE_COLOR (c55, c56, eps);

    check_function (tan, "tan", c1, Color ( .1003346721, .2027100355, .3093362496, .4227932187));
}

