/******************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \file test_bbox.cpp
 **/

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>
#include <mi/math/bbox.h>
#include <vector>

namespace mi { namespace math {
    //
    // Define ostream output operator for vectors so that test failures can be
    // printed properly.
    //
    template <class T, Size DIM>
    std::ostream & operator<< (std::ostream & os, Vector<T, DIM> const & v)
    {
        os << "{";
        for (Size i( 0u ); i != DIM; /**/)
        {
            os << v[i];
            if (++i != DIM)
                os << ",";
        }
        os << "}";
        return os;
    }
    template <class T, Size DIM>
    std::ostream & operator<< (std::ostream & os, Bbox<T, DIM> const & bb)
    {
        return os << "Bbox(min = " << bb.min << "; max = " << bb.max << ")";
    }
}}

// Verify that a bounding box has the expected size, namely that of two vectors
// of the underlying type and dimension.
mi_static_assert( sizeof(mi::math::Bbox<double,3>)
                  == 2*sizeof(mi::math::Vector<double,3>));

#define MI_CHECK_CLOSE_VECTOR(x,y,eps) \
    MI_CHECK_CLOSE_COLLECTIONS((x).begin(),(x).end(),(y).begin(),(y).end(),eps)

static const float eps = 1e-6f;

using namespace mi;
using namespace mi::math;

template <typename T, Size DIM>
void set_elements (Vector<T,DIM>& v)
{
    for (Size i(0u); i < DIM; ++i)
        v[i] = T(i);
}

template<typename T, Size DIM>
void test_bbox_constructors ()
{
    Bbox<T,DIM> b1;
    MI_CHECK( b1.empty());
    b1.clear();
    MI_CHECK( b1.empty());

    Vector<T,DIM> v2;
    set_elements( v2);
    Bbox<T,DIM> b2( v2);
    MI_CHECK_CLOSE_VECTOR( b2.min, v2, eps);
    MI_CHECK_CLOSE_VECTOR( b2.max, v2, eps);
    MI_CHECK_CLOSE_VECTOR( b2[0], v2, eps);
    MI_CHECK_CLOSE_VECTOR( b2[1], v2, eps);

    Bbox<T,1> b3( T(1), T(2));
    Vector<T,1> v30( 1);
    Vector<T,1> v31( 2);
    MI_CHECK_CLOSE_VECTOR( b3.min, v30, eps);
    MI_CHECK_CLOSE_VECTOR( b3.max, v31, eps);
    MI_CHECK_CLOSE_VECTOR( b3[0], v30, eps);
    MI_CHECK_CLOSE_VECTOR( b3[1], v31, eps);

    Bbox<T,2> b4( 1, 2, 3, 4);
    Vector<T,2> v40( 1, 2);
    Vector<T,2> v41( 3, 4);
    MI_CHECK_CLOSE_VECTOR( b4.min, v40, eps);
    MI_CHECK_CLOSE_VECTOR( b4.max, v41, eps);
    MI_CHECK_CLOSE_VECTOR( b4[0], v40, eps);
    MI_CHECK_CLOSE_VECTOR( b4[1], v41, eps);

    const Bbox<T,3> b5( 1, 2, 3, 4, 5, 6);
    Vector<T,3> v50( 1, 2, 3);
    Vector<T,3> v51( 4, 5, 6);
    MI_CHECK_CLOSE_VECTOR( b5.min, v50, eps);
    MI_CHECK_CLOSE_VECTOR( b5.max, v51, eps);
    MI_CHECK_CLOSE_VECTOR( b5[0], v50, eps);
    MI_CHECK_CLOSE_VECTOR( b5[1], v51, eps);

    Vector<T,DIM> v60( v2 + Vector<T,DIM>( T(1)));
    Vector<T,DIM> v61( v2 + Vector<T,DIM>( T(2)));
    std::vector<Vector<T,DIM> > v;
    v.push_back( v2);
    v.push_back( v60);
    v.push_back( v61);
    Bbox<T,DIM> b6( v.begin(), v.end());
    MI_CHECK_CLOSE_VECTOR( b6.min, v2, eps);
    MI_CHECK_CLOSE_VECTOR( b6.max, v61, eps);

    Vector<T,2> v42 = *  b4.begin();
    Vector<T,2> v43 = * (b4.end()-1);
    Vector<T,3> v52 = *  b5.begin();
    Vector<T,3> v53 = * (b5.end()-1);
    MI_CHECK_CLOSE_VECTOR( v42, v40, eps);
    MI_CHECK_CLOSE_VECTOR( v43, v41, eps);
    MI_CHECK_CLOSE_VECTOR( v52, v50, eps);
    MI_CHECK_CLOSE_VECTOR( v53, v51, eps);

    Bbox_struct<T,DIM> bs1 = b1;
    Bbox_struct<T,DIM> bs2 = b2;
    Bbox_struct<T,1>   bs3 = b3;
    Bbox_struct<T,2>   bs4 = b4;
    Bbox_struct<T,3>   bs5 = b5;
    Bbox_struct<T,DIM> bs6 = b6;
    Bbox<T,DIM> bb1( bs1);
    Bbox<T,DIM> bb2( bs2);
    Bbox<T,1>   bb3( bs3);
    Bbox<T,2>   bb4( bs4);
    Bbox<T,3>   bb5( bs5);
    Bbox<T,DIM> bb6( bs6);
    MI_CHECK_EQUAL( b1, bb1);
    MI_CHECK_EQUAL( b2, bb2);
    MI_CHECK_EQUAL( b3, bb3);
    MI_CHECK_EQUAL( b4, bb4);
    MI_CHECK_EQUAL( b5, bb5);
    MI_CHECK_EQUAL( b6, bb6);

    Bbox<T,DIM> bbb1;
    Bbox<T,DIM> bbb2;
    Bbox<T,1>   bbb3;
    Bbox<T,2>   bbb4;
    Bbox<T,3>   bbb5;
    Bbox<T,DIM> bbb6;
    bbb1 = bs1;
    bbb2 = bs2;
    bbb3 = bs3;
    bbb4 = bs4;
    bbb5 = bs5;
    bbb6 = bs6;
    MI_CHECK_EQUAL( b1, bbb1);
    MI_CHECK_EQUAL( b2, bbb2);
    MI_CHECK_EQUAL( b3, bbb3);
    MI_CHECK_EQUAL( b4, bbb4);
    MI_CHECK_EQUAL( b5, bbb5);
    MI_CHECK_EQUAL( b6, bbb6);

    Bbox<T,3> t1( 1, 2, 3, 4, 5, 6);
    Bbox<mi::Float64,3> t2( t1);
    Bbox<T, 3> t3( t2);
    MI_CHECK_EQUAL( t1, t3);
}

MI_TEST_AUTO_FUNCTION( test_bbox_constructors )
{
    test_bbox_constructors<  Uint8,1>();
    test_bbox_constructors< Uint16,1>();
    test_bbox_constructors< Uint32,1>();
    test_bbox_constructors< Uint64,1>();
    test_bbox_constructors<  Sint8,1>();
    test_bbox_constructors< Sint16,1>();
    test_bbox_constructors< Sint32,1>();
    test_bbox_constructors< Sint64,1>();
    test_bbox_constructors<Float32,1>();
    test_bbox_constructors<Float64,1>();

    test_bbox_constructors<  Uint8,2>();
    test_bbox_constructors< Uint16,2>();
    test_bbox_constructors< Uint32,2>();
    test_bbox_constructors< Uint64,2>();
    test_bbox_constructors<  Sint8,2>();
    test_bbox_constructors< Sint16,2>();
    test_bbox_constructors< Sint32,2>();
    test_bbox_constructors< Sint64,2>();
    test_bbox_constructors<Float32,2>();
    test_bbox_constructors<Float64,2>();

    test_bbox_constructors<  Uint8,3>();
    test_bbox_constructors< Uint16,3>();
    test_bbox_constructors< Uint32,3>();
    test_bbox_constructors< Uint64,3>();
    test_bbox_constructors<  Sint8,3>();
    test_bbox_constructors< Sint16,3>();
    test_bbox_constructors< Sint32,3>();
    test_bbox_constructors< Sint64,3>();
    test_bbox_constructors<Float32,3>();
    test_bbox_constructors<Float64,3>();
}

MI_TEST_AUTO_FUNCTION( test_bbox_basics_operators_methods )
{
    using Bbox = Bbox<double, 3>;

    Bbox bba(-1, -2, -3, 1, 1, 1);
    Bbox bbb( 0,  0,  0, 1, 2, 3);
    Bbox bbc;

    bbc = bba;
    MI_CHECK_EQUAL(bbc, bba);
    MI_CHECK(bbc != bbb);
    bbc = clip(bbc, bbb);
    MI_CHECK_EQUAL(bbc, Bbox(0, 0, 0, 1, 1, 1));
    bbc += 3.0;
    bbc -= 1.0;
    bbc *= 2.0;
    bbc /= 2.0;
    MI_CHECK_EQUAL(bbc, Bbox(-2, -2, -2, 3, 3, 3));
    bbc.clear();
    bbc.insert(bba);
    MI_CHECK_EQUAL(bba, bbc);
    bbc.clear();
    bbc.push_back(bba);
    MI_CHECK_EQUAL(bba, bbc);
    bbc.insert(bbb);
    MI_CHECK_EQUAL(bbc, Bbox(-1, -2, -3, 1, 2, 3));
    bbc = ((bba + 1.0) * 2.0 - 2.0) / 2.0;
    MI_CHECK_EQUAL(bbc, Bbox(-1, -2, -3, 1, 1, 1));
    MI_CHECK_EQUAL(bbc.volume(), 24);
    MI_CHECK_GREATER(bbc.diagonal_length(), 5.3851647);
    MI_CHECK_LESS(bbc.diagonal_length(), 5.3851649);
    MI_CHECK_EQUAL(bbc.largest_extent_index(), 2);
}

MI_TEST_AUTO_FUNCTION( test_bbox_transform )
{
    using Bbox = Bbox<double, 3>;
    Bbox bbc(-1, -2, -3, 1, 1, 1);

    // test transform
    Matrix<double,4,4> mat(
        0.0,1.0,0.0,0.0,
        0.0,0.0,1.0,0.0,
        1.0,0.0,0.0,0.0,
        1.0,1.0,1.0,1.0);
    Bbox bbd = transform_vector( mat, bbc);
    MI_CHECK_EQUAL(bbd, Bbox(-3, -1, -2, 1, 1, 1));
    bbd = transform_point( mat, bbc);
    MI_CHECK_EQUAL(bbd, Bbox(-2, 0, -1, 2, 2, 2));
}

MI_TEST_AUTO_FUNCTION( test_bbox_interpolate )
{
    using Bbox = Bbox<double, 3>;
    Bbox bba(-1, -2, -3, 1, 1, 1);
    Bbox bbb( 0,  0,  0, 1, 2, 3);
    Bbox bbc(-1, -2, -3, 1, 1, 1);
    Bbox bbd(-2,  0, -1, 2, 2, 2);

    // test interpolate
    MI_CHECK_EQUAL(bbc, lerp( bbc, bbb, 0.0));
    MI_CHECK_EQUAL(bbb, lerp( bbc, bbb, 1.0));
    bbd = lerp( bbc, bbb, 0.5);
    MI_CHECK_EQUAL(bbd, Bbox(-0.5,-1,-1.5,1,1.5,2));

    // test add_motionbox
    MI_CHECK_EQUAL(bbc, bbc.add_motionbox(bbb,0));
    bbd = bbc.add_motionbox(bba,1);
    MI_CHECK_EQUAL(bbd, Bbox(-2, -4, -6, 2, 2, 2));
    bbd = bbc.add_motionbox(bba,-2);
    MI_CHECK_EQUAL(bbd, Bbox(-3, -4, -5, 3, 5, 7));
}

MI_TEST_AUTO_FUNCTION( test_bbox_intersection )
{
    using Bbox = Bbox<double, 3>;

    Bbox outer(Bbox::Vector(10, 0, 0), Bbox::Vector(20, 10, 10));
    Bbox inner(Bbox::Vector(13, 3, 3), Bbox::Vector(18, 8, 8));

    Bbox b_01(outer);
    Bbox b_02(inner);
    b_01 = clip(b_01, b_02);
    MI_CHECK_EQUAL(b_01, inner);

    Bbox b_03(inner);
    Bbox b_04(outer);
    b_03 = clip(b_03, b_04);
    MI_CHECK_EQUAL(b_03, inner);

    Bbox b_05(Bbox::Vector(10, 0, 0), Bbox::Vector(20, 10, 10));
    Bbox b_06(Bbox::Vector(0, 10, 0), Bbox::Vector(10, 20, 10));
    b_05 = clip(b_05, b_06);
    MI_CHECK_EQUAL(b_05, Bbox(Bbox::Vector(10, 10, 0), Bbox::Vector(10, 10, 10)));

    Bbox b_07(Bbox::Vector(10, 10, 10), Bbox::Vector(20, 20, 20));
    Bbox b_08(Bbox::Vector(15, 5, 5), Bbox::Vector(25, 25, 15));
    b_07 = clip(b_07, b_08);
    MI_CHECK_EQUAL(b_07, Bbox(Bbox::Vector(15, 10, 10), Bbox::Vector(20, 20, 15)));
}

MI_TEST_AUTO_FUNCTION( test_bbox_union )
{
    using Bbox = Bbox<double, 3>;

    Bbox left(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox right(Bbox::Vector(5, 5, 5), Bbox::Vector(15, 15, 15));
    left.insert(right);
    MI_CHECK_EQUAL(left, Bbox(Bbox::Vector(0, 0, 0), Bbox::Vector(15, 15, 15)));

    Bbox b_01(Bbox::Vector(0, 10, 20), Bbox::Vector(10, 50, 50));
    Bbox b_02(Bbox::Vector(10, 0, 20), Bbox::Vector(10, 80, 80));
    b_01.insert(b_02);
    MI_CHECK_EQUAL(b_01, Bbox(Bbox::Vector(0, 0, 20), Bbox::Vector(10, 80, 80)));
}

MI_TEST_AUTO_FUNCTION( test_bbox_function_contains )
{
    using Bbox = Bbox<double, 3>;
    Bbox bba(-1, -2, -3, 1, 1, 1);
    MI_CHECK(bba.contains(Bbox::Vector(-1, -2, -3)));
    MI_CHECK(bba.contains(Bbox::Vector(1, 1, 1)));
    MI_CHECK(bba.contains(Bbox::Vector(0, 1, -3)));
    MI_CHECK(!bba.contains(Bbox::Vector(2, 1, 1)));
    MI_CHECK(!bba.contains(Bbox::Vector(1, 2, 1)));
    MI_CHECK(!bba.contains(Bbox::Vector(1, 1, 2)));
}

MI_TEST_AUTO_FUNCTION( test_bbox_function_intersects )
{
    using Bbox = Bbox<double, 3>;
    Bbox a0(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox b0 = a0;
    MI_CHECK(a0.intersects(b0));
    MI_CHECK(b0.intersects(a0));

    Bbox a1(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox b1(Bbox::Vector(2, 2, 2), Bbox::Vector(8,  8,  8));
    MI_CHECK(a1.intersects(b1));
    MI_CHECK(b1.intersects(a1));

    Bbox a2(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox b2(Bbox::Vector(-2, 2, 2), Bbox::Vector(12,  8,  8));
    MI_CHECK(a2.intersects(b2));
    MI_CHECK(b2.intersects(a2));

    Bbox a3(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox b3(Bbox::Vector(5, 5, 5), Bbox::Vector(15, 15, 15));
    MI_CHECK(a3.intersects(b3));
    MI_CHECK(b3.intersects(a3));

    Bbox a4(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox b4(Bbox::Vector(-5, -5, -5), Bbox::Vector(5, 5, 5));
    MI_CHECK(a4.intersects(b4));
    MI_CHECK(b4.intersects(a4));

    Bbox a5(Bbox::Vector(5, 5, 5), Bbox::Vector(5, 5, 5));
    Bbox b5(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    MI_CHECK(a5.intersects(b5));
    MI_CHECK(b5.intersects(a5));

    Bbox a6(Bbox::Vector(0, 0, 0), Bbox::Vector(0, 0, 0));
    Bbox b6(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    MI_CHECK(a6.intersects(b6));
    MI_CHECK(b6.intersects(a6));

    Bbox a7(Bbox::Vector(0, 0, 0), Bbox::Vector(10, 10, 10));
    Bbox b7(Bbox::Vector(11, 0, 0), Bbox::Vector(21, 10, 10));
    MI_CHECK(!a7.intersects(b7));
    MI_CHECK(!b7.intersects(a7));
}

template<typename T, Size DIM>
void test_bbox_rank_and_related_functions ()
{
    Bbox<T,DIM> b;
    MI_CHECK_EQUAL( b.rank(), 0);
    MI_CHECK(   b.empty());
    MI_CHECK( ! b.is_point());
    MI_CHECK( ! b.is_line());
    MI_CHECK( ! b.is_plane());
    MI_CHECK( ! b.is_volume());

    Vector<T,DIM> v( T(0));
    b.insert( v);
    MI_CHECK_EQUAL( b.rank(), 0);
    MI_CHECK( ! b.empty());
    MI_CHECK(   b.is_point());
    MI_CHECK( ! b.is_line());
    MI_CHECK( ! b.is_plane());
    MI_CHECK( ! b.is_volume());

    v[0] = 1;
    b.insert( v);
    MI_CHECK_EQUAL( b.rank(), 1);
    MI_CHECK( ! b.empty());
    MI_CHECK( ! b.is_point());
    MI_CHECK(   b.is_line());
    MI_CHECK( ! b.is_plane());
    MI_CHECK( ! b.is_volume());

    v[1] = 1;
    b.insert( v);
    MI_CHECK_EQUAL( b.rank(), 2);
    MI_CHECK( ! b.empty());
    MI_CHECK( ! b.is_point());
    MI_CHECK( ! b.is_line());
    MI_CHECK(   b.is_plane());
    MI_CHECK( ! b.is_volume());

    v[2] = 1;
    b.insert( v);
    MI_CHECK_EQUAL( b.rank(), 3);
    MI_CHECK( ! b.empty());
    MI_CHECK( ! b.is_point());
    MI_CHECK( ! b.is_line());
    MI_CHECK( ! b.is_plane());
    MI_CHECK(   b.is_volume());

    v[3] = 1;
    b.insert( v);
    MI_CHECK_EQUAL( b.rank(), 4);
    MI_CHECK( ! b.empty());
    MI_CHECK( ! b.is_point());
    MI_CHECK( ! b.is_line());
    MI_CHECK( ! b.is_plane());
    MI_CHECK( ! b.is_volume());
}

MI_TEST_AUTO_FUNCTION( test_bbox_rank_and_related_functions )
{
    test_bbox_rank_and_related_functions<  Uint8,5>();
    test_bbox_rank_and_related_functions< Uint16,5>();
    test_bbox_rank_and_related_functions< Uint32,5>();
    test_bbox_rank_and_related_functions< Uint64,5>();
    test_bbox_rank_and_related_functions<  Sint8,5>();
    test_bbox_rank_and_related_functions< Sint16,5>();
    test_bbox_rank_and_related_functions< Sint32,5>();
    test_bbox_rank_and_related_functions< Sint64,5>();
    test_bbox_rank_and_related_functions<Float32,5>();
    test_bbox_rank_and_related_functions<Float64,5>();
}

template<typename T, Size DIM>
void test_bbox_free_comparison_operators ()
{
    Bbox<T,DIM> b[7];
    b[0] = Bbox<T,DIM>( 0, 0, 0, 0, 0, 0);
    b[1] = Bbox<T,DIM>( 0, 0, 0, 0, 0, 1);
    b[2] = Bbox<T,DIM>( 0, 0, 0, 0, 1, 1);
    b[3] = Bbox<T,DIM>( 0, 0, 0, 1, 1, 1);
    b[4] = Bbox<T,DIM>( 0, 0, 1, 1, 1, 1);
    b[5] = Bbox<T,DIM>( 0, 1, 1, 1, 1, 1);
    b[6] = Bbox<T,DIM>( 1, 1, 1, 1, 1, 1);

    for (Size i(0u); i < 7; ++i)
        for (Size j(0u); j < 7; ++j)
            {
                MI_CHECK( ( i != j) ^ ( b[i] == b[j]));
                MI_CHECK( ( i == j) ^ ( b[i] != b[j]));
                MI_CHECK( ( i >= j) ^ ( b[i] <  b[j]));
                MI_CHECK( ( i >  j) ^ ( b[i] <= b[j]));
                MI_CHECK( ( i <= j) ^ ( b[i] >  b[j]));
                MI_CHECK( ( i <  j) ^ ( b[i] >= b[j]));
            }
}

MI_TEST_AUTO_FUNCTION( test_bbox_free_comparison_operators )
{
    test_bbox_free_comparison_operators<  Uint8,3>();
    test_bbox_free_comparison_operators< Uint16,3>();
    test_bbox_free_comparison_operators< Uint32,3>();
    test_bbox_free_comparison_operators< Uint64,3>();
    test_bbox_free_comparison_operators<  Sint8,3>();
    test_bbox_free_comparison_operators< Sint16,3>();
    test_bbox_free_comparison_operators< Sint32,3>();
    test_bbox_free_comparison_operators< Sint64,3>();
    test_bbox_free_comparison_operators<Float32,3>();
    test_bbox_free_comparison_operators<Float64,3>();
}
