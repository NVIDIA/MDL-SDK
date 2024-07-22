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
#include <mi/math/color.h>
#include <mi/math/function.h>
#include <mi/math/spectrum.h>
#include <mi/math/vector.h>

///////////////////////////////////////////////////////////////////////////////
// Convenience operators for vector comparison and printing
///////////////////////////////////////////////////////////////////////////////

namespace mi { namespace math {
    //
    // Define ostream output operator for vectors so that test failures can be
    // printed properly.
    //
    template <class T, Size DIM>
    std::ostream & operator<< (std::ostream & os, Vector<T, DIM> const & v)
    {
        os << "Vector(";
        for (Size i( 0u ); i != DIM; /**/)
        {
            os << v[i];
            if (++i != DIM)
                os << ",";
        }
        os << ")";
        return os;
    }
}}

static const float eps = 1e-6f;

using namespace mi;
using namespace mi::math;

#define MI_CHECK_CLOSE_VECTOR3(v1,v2,eps) \
    MI_CHECK_CLOSE((v1)[0], (v2)[0], (eps)); \
    MI_CHECK_CLOSE((v1)[1], (v2)[1], (eps)); \
    MI_CHECK_CLOSE((v1)[2], (v2)[2], (eps))

template <typename T, Size DIM>
void set_elements (Vector<T,DIM>& v)
{
    for (Size i(0u); i < DIM; ++i)
        v[i] = T(i);
}

struct Operator_plus_equal {
    /// Functor call.
    template <typename T>
    MI_FORCE_INLINE void operator()( T& t1, const T& t2) { t1 += t2; }
};

template<typename T>
void test_function_functors ()
{
    MI_CHECK_EQUAL( functor::Operator_plus() ( T(2), T(1)), T(3));
    MI_CHECK_EQUAL( functor::Operator_minus()( T(2), T(1)), T(1));
    MI_CHECK_EQUAL( functor::Operator_multiply()( T(2), T(3)), T(6));
    MI_CHECK_EQUAL( functor::Operator_divide()( T(42), T(3)), T(14));

    T x( T(42));
    MI_CHECK_EQUAL( functor::Operator_pre_incr()( x), T(43));
    MI_CHECK_EQUAL( functor::Operator_post_incr()( x), T(43));
    MI_CHECK_EQUAL( x, T(44));
    MI_CHECK_EQUAL( functor::Operator_pre_decr()( x), T(43));
    MI_CHECK_EQUAL( functor::Operator_post_decr()( x), T(43));
    MI_CHECK_EQUAL( x, T(42));

    Vector<T,7> v1;
    set_elements( v1);
    Vector<T,7> v2( v1);
    Vector<T,7> v3( 42);
    general::for_each( v2, v3, Operator_plus_equal());
    MI_CHECK_EQUAL( v2, v1+v3);
}

MI_TEST_AUTO_FUNCTION( test_function_functors )
{
    test_function_functors<  Uint8>();
    test_function_functors< Uint16>();
    test_function_functors< Uint32>();
    test_function_functors< Uint64>();
    test_function_functors<  Sint8>();
    test_function_functors< Sint16>();
    test_function_functors< Sint32>();
    test_function_functors< Sint64>();
    test_function_functors<Float32>();
    test_function_functors<Float64>();
}

MI_TEST_AUTO_FUNCTION( test_function_math_functions )
{
    MI_CHECK(   is_approx_equal( 1.0f, 1.0f, 0.0f));
    MI_CHECK(   is_approx_equal( 1.0,  1.0,  0.0));
    MI_CHECK( ! is_approx_equal( 2.0f, 1.0f, 0.9f));
    MI_CHECK( ! is_approx_equal( 2.0,  1.0,  0.9));
    MI_CHECK( ! is_approx_equal( 1.0f, 2.0f, 0.9f));
    MI_CHECK( ! is_approx_equal( 1.0,  2.0,  0.9));
    MI_CHECK(   is_approx_equal( 2.0f, 1.0f, 1.1f));
    MI_CHECK(   is_approx_equal( 2.0,  1.0,  1.1));
    MI_CHECK(   is_approx_equal( 1.0f, 2.0f, 1.1f));
    MI_CHECK(   is_approx_equal( 1.0,  2.0,  1.1));

    using Vector3 = Vector<Float32, 3>;
    Vector3 v2a(11, 22, 33);
    Vector3 v2b(10, 23, 32);
    Vector3 v2c(12, 21, 34);

    MI_CHECK(   is_approx_equal( v2a, v2a, 1e-32f));
    MI_CHECK( ! is_approx_equal( v2a, v2b, 1e-32f));
    MI_CHECK( ! is_approx_equal( v2a, v2c, 1e-32f));
    MI_CHECK( ! is_approx_equal( v2a, v2b, 0.99f));
    MI_CHECK(   is_approx_equal( v2a, v2b, 1.0f));
    MI_CHECK( ! is_approx_equal( v2a, v2c, 0.99f));
    MI_CHECK(   is_approx_equal( v2a, v2c, 1.0f));
}

MI_TEST_AUTO_FUNCTION( test_function_fast_sqrt_exp_pow2_log2_pow )
{
    MI_CHECK_CLOSE( fast_sqrt( 1), 1.0f, 0.001);
    MI_CHECK_CLOSE( fast_sqrt( 100), 10.0f, 0.3);
    MI_CHECK_CLOSE( fast_sqrt( 10000), 100.0f, 4.0);
    MI_CHECK_CLOSE( fast_exp( 1), 2.7182818f, 0.001);
    MI_CHECK_CLOSE( fast_exp( 2), 7.3890561f, 0.015);
    MI_CHECK_CLOSE( fast_exp( 3), 20.085537f, 0.04);
    MI_CHECK_EQUAL( fast_pow2( 0), 1.0f);
    MI_CHECK_EQUAL( fast_pow2( 1), 2.0f);
    MI_CHECK_EQUAL( fast_pow2( 2), 4.0f);
    MI_CHECK_EQUAL( fast_pow2( 3), 8.0f);
    MI_CHECK_EQUAL( fast_log2( 1), 0.0f);
    MI_CHECK_EQUAL( fast_log2( 2), 1.0f);
    MI_CHECK_EQUAL( fast_log2( 4), 2.0f);
    MI_CHECK_EQUAL( fast_log2( 8), 3.0f);
    MI_CHECK_EQUAL( fast_pow( 0, 0), 0.0f);
    MI_CHECK_EQUAL( fast_pow( 0, 1), 0.0f);
    MI_CHECK_EQUAL( fast_pow( 0, 2), 0.0f);
    MI_CHECK_EQUAL( fast_pow( 0, 3), 0.0f);
    MI_CHECK_EQUAL( fast_pow( 1, 0), 1.0f);
    MI_CHECK_EQUAL( fast_pow( 1, 1), 1.0f);
    MI_CHECK_EQUAL( fast_pow( 1, 2), 1.0f);
    MI_CHECK_EQUAL( fast_pow( 1, 3), 1.0f);
    MI_CHECK_EQUAL( fast_pow( 2, 0), 1.0f);
    MI_CHECK_EQUAL( fast_pow( 2, 1), 2.0f);
    MI_CHECK_EQUAL( fast_pow( 2, 2), 4.0f);
    MI_CHECK_EQUAL( fast_pow( 2, 3), 8.0f);
    MI_CHECK_EQUAL( fast_pow( 3, 0), 1.0f);
    MI_CHECK_CLOSE( fast_pow( 3, 1), 3.0f, 0.01);
    MI_CHECK_CLOSE( fast_pow( 3, 2), 9.0f, 0.01);
    MI_CHECK_CLOSE( fast_pow( 3, 3), 27.0f, 0.2);
}

template<typename T>
void test_function_pow ()
{
    MI_CHECK_EQUAL( mi::math::pow( T( 0), T(0)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 0), T(1)), T(0));
    MI_CHECK_EQUAL( mi::math::pow( T( 0), T(2)), T(0));
    MI_CHECK_EQUAL( mi::math::pow( T( 0), T(3)), T(0));
    MI_CHECK_EQUAL( mi::math::pow( T( 1), T(0)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 1), T(1)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 1), T(2)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 1), T(3)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 2), T(0)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 2), T(1)), T(2));
    MI_CHECK_EQUAL( mi::math::pow( T( 2), T(2)), T(4));
    MI_CHECK_EQUAL( mi::math::pow( T( 2), T(3)), T(8));
    MI_CHECK_EQUAL( mi::math::pow( T( 3), T(0)), T(1));
    MI_CHECK_EQUAL( mi::math::pow( T( 3), T(1)), T(3));
    MI_CHECK_EQUAL( mi::math::pow( T( 3), T(2)), T(9));
    MI_CHECK_EQUAL( mi::math::pow( T( 3), T(3)), T(27));
}

MI_TEST_AUTO_FUNCTION( test_function_pow )
{
    test_function_pow< Uint32>();
    test_function_pow< Uint64>();
    test_function_pow< Sint32>();
    test_function_pow< Sint64>();
    test_function_pow<Float32>();
    test_function_pow<Float64>();
}

template<typename T>
void test_function_all_any_clamp ()
{
    T x0( T(0));
    T x1( T(1));
    T x2( T(2));
    T x3( T(3));
    T x4( T(4));
    MI_CHECK( ! all( x0));
    MI_CHECK(   all( x1));
    MI_CHECK(   all( x2));
    MI_CHECK(   all( x3));
    MI_CHECK(   all( x4));
    MI_CHECK( ! any( x0));
    MI_CHECK(   any( x1));
    MI_CHECK(   any( x2));
    MI_CHECK(   any( x3));
    MI_CHECK(   any( x4));
    MI_CHECK_EQUAL( clamp( x0, x1, x3), x1);
    MI_CHECK_EQUAL( clamp( x1, x1, x3), x1);
    MI_CHECK_EQUAL( clamp( x2, x1, x3), x2);
    MI_CHECK_EQUAL( clamp( x3, x1, x3), x3);
    MI_CHECK_EQUAL( clamp( x4, x1, x3), x3);
}

MI_TEST_AUTO_FUNCTION( test_function_all_any_clamp )
{
    test_function_all_any_clamp<  Uint8>();
    test_function_all_any_clamp< Uint16>();
    test_function_all_any_clamp< Uint32>();
    test_function_all_any_clamp< Uint64>();
    test_function_all_any_clamp<  Sint8>();
    test_function_all_any_clamp< Sint16>();
    test_function_all_any_clamp< Sint32>();
    test_function_all_any_clamp< Sint64>();
    test_function_all_any_clamp<Float32>();
    test_function_all_any_clamp<Float64>();
}

template<typename T>
void test_function_lerp ()
{
    MI_CHECK_CLOSE( lerp( T(-0.1), T(-1.2), T(0.1)), T(-0.21), eps);
    MI_CHECK_CLOSE( lerp( T(-0.2), T(-0.1), T(0.2)), T(-0.18), eps);
    MI_CHECK_CLOSE( lerp( T(-0.3), T( 0.5), T(0.3)), T(-0.06), eps);
    MI_CHECK_CLOSE( lerp( T(-0.4), T( 1.6), T(0.4)), T( 0.40), eps);
}

MI_TEST_AUTO_FUNCTION( test_function_lerp )
{
    test_function_lerp<Float32>();
    test_function_lerp<Float64>();
}

template<typename T>
void test_function_sign_signbit ()
{
    T x0( T(-100));
    T x1( T(-1));
    T x2( T(0));
    T x3( T(1));
    T x4( T(100));
    MI_CHECK_EQUAL( sign( x0), T(-1));
    MI_CHECK_EQUAL( sign( x1), T(-1));
    MI_CHECK_EQUAL( sign( x2), T(0));
    MI_CHECK_EQUAL( sign( x3), T(1));
    MI_CHECK_EQUAL( sign( x4), T(1));
    MI_CHECK( sign_bit( x0));
    MI_CHECK( sign_bit( x1));
    MI_CHECK( ! sign_bit( x2));
    MI_CHECK( ! sign_bit( x3));
    MI_CHECK( ! sign_bit( x4));
}

MI_TEST_AUTO_FUNCTION( test_function_sign_signbit )
{
    test_function_sign_signbit<  Sint8>();
    test_function_sign_signbit< Sint16>();
    test_function_sign_signbit< Sint32>();
    test_function_sign_signbit< Sint64>();
    test_function_sign_signbit<Float32>();
    test_function_sign_signbit<Float64>();
}

template<typename T>
void test_function_generic_vector_algorithms ()
{
    T x( T(3));
    T y( T(-14));
    MI_CHECK_EQUAL( dot( x, y), T(-42));
    MI_CHECK_EQUAL( length( x), T(3));
    MI_CHECK_EQUAL( length( y), T(14));

    Vector<T,4> v1;
    set_elements( v1);
    Vector<T,4> v2;
    for (Size i(0u); i < 4; ++i)
        v2[i] = v1[4-1-i];
    Vector<T,4> v3( T(1));
    Vector<T,4> v4( T(2));

    MI_CHECK_EQUAL( dot( v1, v2), T(4));
    MI_CHECK_EQUAL( square_length( v1), T (14));
    MI_CHECK_CLOSE( length( v1), std::sqrt( T(14)), eps);
    MI_CHECK_EQUAL( square_euclidean_distance( v1, v2), T(20));
    MI_CHECK_CLOSE( euclidean_distance( v1, v2), std::sqrt( T(20)), eps);

    Vector<T,4> v5( v1);
    set_bounds( v5, v3, v4);
    Vector<T,4> v6( 1, 1, 2, 2);
    MI_CHECK_EQUAL( v5, v6);

    MI_CHECK( is_equal( v5, v6));
    MI_CHECK( ! is_equal( v1, v6));
}

MI_TEST_AUTO_FUNCTION( test_function_generic_vector_algorithms )
{
    test_function_generic_vector_algorithms<Float32>();
    test_function_generic_vector_algorithms<Float64>();
}

void test_rgbe( Float32 r, Float32 g, Float32 b)
{
    Float32 in_float[3] = { r, g, b };
    Color in_color( r, g, b );
    Spectrum in_spectrum( r, g, b);
    Uint8 rgbe[4];
    Uint32 rgbe32;
    Float32 out_float[3];
    Color out_color;
    Spectrum out_spectrum;

    // test Float32[3]/Uint8[4] version
    to_rgbe( in_float, rgbe);
    from_rgbe( rgbe, out_float);
    MI_CHECK_CLOSE_VECTOR3( in_float, out_float, 0.004);

    // test Float32[3]/Uint32 version
    to_rgbe( in_float, rgbe32);
    from_rgbe( rgbe32, out_float);
    MI_CHECK_CLOSE_VECTOR3( in_float, out_float, 0.004);

    // test Color/Uint8[4] version
    to_rgbe( in_color, rgbe);
    from_rgbe( rgbe, out_color);
    MI_CHECK_CLOSE_VECTOR3( in_color, out_color, 0.004);

    // test Color/Uint32 version
    to_rgbe( in_color, rgbe32);
    from_rgbe( rgbe32, out_color);
    MI_CHECK_CLOSE_VECTOR3( in_color, out_color, 0.004);

    // test Spectrum/Uint8[4] version
    to_rgbe( in_spectrum, rgbe);
    from_rgbe( rgbe, out_spectrum);
    MI_CHECK_CLOSE_VECTOR3( in_spectrum, out_spectrum, 0.004);

    // test Spetrum/Uint32 version
    to_rgbe( in_spectrum, rgbe32);
    from_rgbe( rgbe32, out_spectrum);
    MI_CHECK_CLOSE_VECTOR3( in_spectrum, out_spectrum, 0.004);
}

MI_TEST_AUTO_FUNCTION( test_to_from_rgbe )
{
    for( mi::Size i = 0; i < 1000000; ++i) {
#ifdef MI_PLATFORM_WINDOWS
        Float32 r = (float) rand() / RAND_MAX;
        Float32 g = (float) rand() / RAND_MAX;
        Float32 b = (float) rand() / RAND_MAX;
#else
        Float32 r = drand48();
        Float32 g = drand48();
        Float32 b = drand48();
#endif
        test_rgbe( r, g, b);
    }

    // explicitly test values that have been problematic in past versions
    test_rgbe( base::binary_cast<float>( 0x3f7fffff), 0, 0); // ~1.0
    test_rgbe( base::binary_cast<float>( 0x3effffff), 0, 0); // ~0.5
    test_rgbe( base::binary_cast<float>( 0x3e7fffff), 0, 0); // ~0.25
    test_rgbe( base::binary_cast<float>( 0x32d22f3c), 1, 1);
}
