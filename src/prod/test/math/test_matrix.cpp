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
 ** \file test_matrix.cpp
 **/

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>
#include <mi/math/matrix.h>
#include <vector>

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
    template <class T, Size ROW, Size COL>
    std::ostream & operator<< (std::ostream & os, Matrix<T,ROW,COL> const & m)
    {
        Size const DIM( ROW * COL );
        os << "Matrix" << ROW << "x" << COL << "(";
        for (Size i( 0u ); i != DIM; /**/)
        {
            os << m.get(i);
            if (++i != DIM)
                os << ",";
        }
        os << ")";
        return os;
    }
}}

#define MI_CHECK_CLOSE_MATRIX(A,B,eps) \
    MI_CHECK_CLOSE_COLLECTIONS((A).begin(),(A).end(),(B).begin(),(B).end(),eps)

#define MI_CHECK_CLOSE_VECTOR(x,y,eps) \
    MI_CHECK_CLOSE_COLLECTIONS((x).begin(),(x).end(),(y).begin(),(y).end(),eps)

using namespace mi;
using namespace mi::math;

template <typename T, Size ROW, Size COL>
void set_elements (Matrix<T,ROW,COL>& m)
{
    for (Size row(0u); row < ROW; ++row)
        for (Size col(0u); col < COL; ++col)
            m[row][col] = T(row * COL + col);
}

template <typename T, Size ROW, Size COL>
bool check_elements (const Matrix<T,ROW,COL>& m)
{
    for (Size row(0u); row < ROW; ++row)
        for (Size col(0u); col < COL; ++col)
            if (m[row][col] != row * COL + col)
                return false;
    return true;
}

MI_TEST_AUTO_FUNCTION( test_matrix_constructors )
{
    Matrix<Float32,4,4> m1;
    set_elements( m1);

    std::vector<Float32> v;
    for (Size i(0u); i < 4*4; ++i)
        v.push_back( Float32(i));
    Matrix<Float32,4,4> m2( FROM_ITERATOR, v.begin());

    Float32 array1[4*4];
    for (Size i(0u); i < 4*4; ++i)
        array1[i] = Float32(i);
    Matrix<Float32,4,4> m3( array1);

    Uint8 array2[4*4];
    for (Size i(0u); i < 4*4; ++i)
        array2[i] = Uint8(i);
    Matrix<Float32,4,4> m4( array2);

    Matrix<Float32,4,4> m5( m1);

    Matrix<Uint8,4,4> m( array2);
    Matrix<Float32,4,4> m6( m);

    MI_CHECK( check_elements( m1));
    MI_CHECK_EQUAL( m1, m2);
    MI_CHECK_EQUAL( m1, m3);
    MI_CHECK_EQUAL( m1, m4);
    MI_CHECK_EQUAL( m1, m5);
    MI_CHECK_EQUAL( m1, m6);
    MI_CHECK_EQUAL( m2, m3);
    MI_CHECK_EQUAL( m2, m4);
    MI_CHECK_EQUAL( m2, m5);
    MI_CHECK_EQUAL( m2, m6);
    MI_CHECK_EQUAL( m3, m4);
    MI_CHECK_EQUAL( m3, m5);
    MI_CHECK_EQUAL( m3, m6);
    MI_CHECK_EQUAL( m4, m5);
    MI_CHECK_EQUAL( m4, m6);
    MI_CHECK_EQUAL( m5, m6);

    Matrix<Float32,4,4> m7( Matrix<Float32,4,4>::TRANSPOSED_COPY_TAG, m1);
    const Matrix<Float32,4,4> m8( Matrix<Float32,4,4>::TRANSPOSED_COPY_TAG, m7);
    Matrix<Sint8,4,4> m9;
    set_elements( m9);
    Matrix<Float32,4,4> m10( Matrix<Float32,4,4>::TRANSPOSED_COPY_TAG, m9);
    Matrix<Float32,4,4> m11( 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    MI_CHECK_EQUAL( m1, m8);
    MI_CHECK_EQUAL( m7, m11);
    MI_CHECK_EQUAL( m10, m11);

    MI_CHECK( m1.SIZE == 4*4);
    MI_CHECK( m1.size() == 4*4);
    MI_CHECK( m1.max_size() == 4*4);

    MI_CHECK( m1.begin() == & m1.xx);
    MI_CHECK( m8.begin() == & m8.xx);
    MI_CHECK( m1.end()-1 == & m1.ww);
    MI_CHECK( m8.end()-1 == & m8.ww);
}

MI_TEST_AUTO_FUNCTION( test_matrix_elementwise_constructors )
{
    Matrix<Uint8,1,1> m11( 1);
    Matrix<Uint8,1,2> m12( 1, 2);
    Matrix<Uint8,2,1> m21( 1, 2);
    Matrix<Uint8,1,3> m13( 1, 2, 3);
    Matrix<Uint8,3,1> m31( 1, 2, 3);
    Matrix<Uint8,1,4> m14( 1, 2, 3, 4);
    Matrix<Uint8,2,2> m22( 1, 2, 3, 4);
    Matrix<Uint8,4,1> m41( 1, 2, 3, 4);
    Matrix<Uint8,2,3> m23( 1, 2, 3, 4, 5, 6);
    Matrix<Uint8,3,2> m32( 1, 2, 3, 4, 5, 6);
    Matrix<Uint8,2,4> m24( 1, 2, 3, 4, 5, 6, 7, 8);
    Matrix<Uint8,4,2> m42( 1, 2, 3, 4, 5, 6, 7, 8);
    Matrix<Uint8,3,3> m33( 1, 2, 3, 4, 5, 6, 7, 8, 9);
    Matrix<Uint8,3,4> m34( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    Matrix<Uint8,4,3> m43( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    Matrix<Uint8,4,4> m44( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    check_elements( m11);
    check_elements( m12);
    check_elements( m21);
    check_elements( m13);
    check_elements( m31);
    check_elements( m14);
    check_elements( m22);
    check_elements( m41);
    check_elements( m24);
    check_elements( m42);
    check_elements( m33);
    check_elements( m34);
    check_elements( m43);
    check_elements( m44);
}

MI_TEST_AUTO_FUNCTION(  test_matrix_rowwise_constructors )
{
    Vector<Uint8,1> v11( 1);
    Vector<Uint8,1> v21( 2);
    Vector<Uint8,1> v31( 3);
    Vector<Uint8,1> v41( 4);
    Vector<Uint8,2> v12( 1, 2);
    Vector<Uint8,2> v22( 3, 4);
    Vector<Uint8,2> v32( 5, 6);
    Vector<Uint8,2> v42( 7, 8);
    Vector<Uint8,3> v13( 1, 2, 3);
    Vector<Uint8,3> v23( 4, 5, 6);
    Vector<Uint8,3> v33( 7, 8, 9);
    Vector<Uint8,3> v43( 10, 11, 12);
    Vector<Uint8,4> v14( 1, 2, 3, 4);
    Vector<Uint8,4> v24( 5, 6, 7, 8);
    Vector<Uint8,4> v34( 9, 10, 11, 12);
    Vector<Uint8,4> v44( 13, 14, 15, 16);
    Matrix<Uint8,1,1> m11( v11);
    Matrix<Uint8,1,2> m12( v12);
    Matrix<Uint8,1,3> m13( v13);
    Matrix<Uint8,1,4> m14( v14);
    Matrix<Uint8,2,1> m21( v11, v21);
    Matrix<Uint8,2,2> m22( v12, v22);
    Matrix<Uint8,2,3> m23( v13, v23);
    Matrix<Uint8,2,4> m24( v14, v24);
    Matrix<Uint8,3,1> m31( v11, v21, v31);
    Matrix<Uint8,3,2> m32( v12, v22, v32);
    Matrix<Uint8,3,3> m33( v13, v23, v33);
    Matrix<Uint8,3,4> m34( v14, v24, v34);
    const Matrix<Uint8,4,1> m41( v11, v21, v31, v41);
    const Matrix<Uint8,4,2> m42( v12, v22, v32, v42);
    const Matrix<Uint8,4,3> m43( v13, v23, v33, v43);
    const Matrix<Uint8,4,4> m44( v14, v24, v34, v44);
    check_elements( m11);
    check_elements( m12);
    check_elements( m13);
    check_elements( m14);
    check_elements( m21);
    check_elements( m22);
    check_elements( m23);
    check_elements( m24);
    check_elements( m31);
    check_elements( m32);
    check_elements( m33);
    check_elements( m34);
    check_elements( m41);
    check_elements( m42);
    check_elements( m43);
    check_elements( m44);

    MI_CHECK( m11[0] == v11);
    MI_CHECK( m12[0] == v12);
    MI_CHECK( m13[0] == v13);
    MI_CHECK( m14[0] == v14);

    MI_CHECK( m21[0] == v11);
    MI_CHECK( m21[1] == v21);
    MI_CHECK( m22[0] == v12);
    MI_CHECK( m22[1] == v22);
    MI_CHECK( m23[0] == v13);
    MI_CHECK( m23[1] == v23);
    MI_CHECK( m24[0] == v14);
    MI_CHECK( m24[1] == v24);

    MI_CHECK( m31[0] == v11);
    MI_CHECK( m31[1] == v21);
    MI_CHECK( m31[2] == v31);
    MI_CHECK( m32[0] == v12);
    MI_CHECK( m32[1] == v22);
    MI_CHECK( m32[2] == v32);
    MI_CHECK( m33[0] == v13);
    MI_CHECK( m33[1] == v23);
    MI_CHECK( m33[2] == v33);
    MI_CHECK( m34[0] == v14);
    MI_CHECK( m34[1] == v24);
    MI_CHECK( m34[2] == v34);

    MI_CHECK( m41[0] == v11);
    MI_CHECK( m41[1] == v21);
    MI_CHECK( m41[2] == v31);
    MI_CHECK( m41[3] == v41);
    MI_CHECK( m42[0] == v12);
    MI_CHECK( m42[1] == v22);
    MI_CHECK( m41[2] == v31);
    MI_CHECK( m41[3] == v41);
    MI_CHECK( m43[0] == v13);
    MI_CHECK( m43[1] == v23);
    MI_CHECK( m43[2] == v33);
    MI_CHECK( m43[3] == v43);
    MI_CHECK( m44[0] == v14);
    MI_CHECK( m44[1] == v24);
    MI_CHECK( m44[2] == v34);
    MI_CHECK( m44[3] == v44);
}

template<typename T, Size ROW, Size COL>
void test_matrix_assignment_operator_get_set ()
{
    Matrix<T,ROW,COL> m1;
    set_elements( m1);
    Matrix<T,ROW,COL> m2;
    m2 = m1;
    MI_CHECK_EQUAL( m1, m2);

    Matrix<T,ROW,COL> m3;
    for (Size row(0u); row < ROW; ++row)
        for (Size col(0u); col < COL; ++col)
            {
                m2( row, col) = m1( row, col);
                m3( row, col) = m2( row, col);
            }
    MI_CHECK_EQUAL( m1, m2);
    MI_CHECK_EQUAL( m1, m3);
    MI_CHECK_EQUAL( m2, m3);

    for (Size row(0u); row < ROW; ++row)
        for (Size col(0u); col < COL; ++col)
            {
                 m2.set( row, col, m1.get( row, col));
                 m3.set( row, col, m1.get( row, col));
             }
    MI_CHECK_EQUAL( m1, m2);
    MI_CHECK_EQUAL( m1, m3);
    MI_CHECK_EQUAL( m2, m3);
}

MI_TEST_AUTO_FUNCTION( test_matrix_assignment_operator_get_set )
{
    test_matrix_assignment_operator_get_set<  Uint8,1,1>();
    test_matrix_assignment_operator_get_set< Uint16,1,1>();
    test_matrix_assignment_operator_get_set< Uint32,1,1>();
    test_matrix_assignment_operator_get_set< Uint64,1,1>();
    test_matrix_assignment_operator_get_set<  Sint8,1,1>();
    test_matrix_assignment_operator_get_set< Sint16,1,1>();
    test_matrix_assignment_operator_get_set< Sint32,1,1>();
    test_matrix_assignment_operator_get_set< Sint64,1,1>();
    test_matrix_assignment_operator_get_set<Float32,1,1>();
    test_matrix_assignment_operator_get_set<Float64,1,1>();

    test_matrix_assignment_operator_get_set<  Uint8,7,9>();
    test_matrix_assignment_operator_get_set< Uint16,7,9>();
    test_matrix_assignment_operator_get_set< Uint32,7,9>();
    test_matrix_assignment_operator_get_set< Uint64,7,9>();
    test_matrix_assignment_operator_get_set<  Sint8,7,9>();
    test_matrix_assignment_operator_get_set< Sint16,7,9>();
    test_matrix_assignment_operator_get_set< Sint32,7,9>();
    test_matrix_assignment_operator_get_set< Sint64,7,9>();
    test_matrix_assignment_operator_get_set<Float32,7,9>();
    test_matrix_assignment_operator_get_set<Float64,7,9>();
}

template<typename T, Size ROW, Size COL>
void test_matrix_free_comparison_operators ()
{
    Matrix<T,ROW,COL> m1;
    set_elements( m1);
    Matrix<T,ROW,COL> m2( m1);
    m2[ROW-1][COL-1] += T(1);

    MI_CHECK( m1 == m1);
    MI_CHECK( m2 == m2);
    MI_CHECK( ! (m1 == m2));
    MI_CHECK( ! (m2 == m1));

    MI_CHECK( ! (m1 != m1));
    MI_CHECK( ! (m2 != m2));
    MI_CHECK( m1 != m2);
    MI_CHECK( m2 != m1);

    MI_CHECK( ! (m1 < m1));
    MI_CHECK( ! (m2 < m2));
    MI_CHECK( m1 < m2);
    MI_CHECK( ! (m2 < m1));

    MI_CHECK( m1 <= m1);
    MI_CHECK( m2 <= m2);
    MI_CHECK( m1 <= m2);
    MI_CHECK( ! (m2 <= m1));

    MI_CHECK( ! (m1 > m1));
    MI_CHECK( ! (m2 > m2));
    MI_CHECK( ! (m1 > m2));
    MI_CHECK( m2 > m1);

    MI_CHECK( m1 >= m1);
    MI_CHECK( m2 >= m2);
    MI_CHECK( ! (m1 >= m2));
    MI_CHECK( m2 >= m1);
}

MI_TEST_AUTO_FUNCTION( test_matrix_free_comparison_operators )
{
    test_matrix_free_comparison_operators<  Uint8,1,1>();
    test_matrix_free_comparison_operators< Uint16,1,1>();
    test_matrix_free_comparison_operators< Uint32,1,1>();
    test_matrix_free_comparison_operators< Uint64,1,1>();
    test_matrix_free_comparison_operators<  Sint8,1,1>();
    test_matrix_free_comparison_operators< Sint16,1,1>();
    test_matrix_free_comparison_operators< Sint32,1,1>();
    test_matrix_free_comparison_operators< Sint64,1,1>();
    test_matrix_free_comparison_operators<Float32,1,1>();
    test_matrix_free_comparison_operators<Float64,1,1>();

    test_matrix_free_comparison_operators<  Uint8,7,9>();
    test_matrix_free_comparison_operators< Uint16,7,9>();
    test_matrix_free_comparison_operators< Uint32,7,9>();
    test_matrix_free_comparison_operators< Uint64,7,9>();
    test_matrix_free_comparison_operators<  Sint8,7,9>();
    test_matrix_free_comparison_operators< Sint16,7,9>();
    test_matrix_free_comparison_operators< Sint32,7,9>();
    test_matrix_free_comparison_operators< Sint64,7,9>();
    test_matrix_free_comparison_operators<Float32,7,9>();
    test_matrix_free_comparison_operators<Float64,7,9>();
}

template<typename T, Size ROW, Size COL>
void test_matrix_free_arithmetic_operators ()
{
    Matrix<T,ROW,COL> m1;
    set_elements( m1);
    Matrix<T,ROW,COL> m2;
    set_elements( m2);
    m2 = -m2;
    Matrix<T,ROW,COL> m3( T(0));

    Matrix<T,ROW,COL> m4 = m1 + m2;
    Matrix<T,ROW,COL> m5 = m2 + m1;
    Matrix<T,ROW,COL> m6 = m1;
    m6 += m2;
    Matrix<T,ROW,COL> m7 = m2;
    m7 += m1;
    MI_CHECK_EQUAL( m3, m4);
    MI_CHECK_EQUAL( m3, m5);
    MI_CHECK_EQUAL( m3, m6);
    MI_CHECK_EQUAL( m3, m7);

    Matrix<T,ROW,COL> m8 = m1 - m1;
    Matrix<T,ROW,COL> m9( m1);
    m9 -= m1;
    MI_CHECK_EQUAL( m3, m8);
    MI_CHECK_EQUAL( m3, m9);

    Matrix<T,COL,COL> m10( T(1));
    Matrix<T,ROW,COL> m11( m1);
    m11 *= m10;
    MI_CHECK_EQUAL( m11, m1);

    Matrix<T,ROW,COL> m12( m1);
    Matrix<T,COL,ROW> m13;
    for (Size row(0u); row < COL; ++row)
        for (Size col(0u); col < ROW; ++col)
            m13[row][col] = ((row % 2) == 0) ? T(-1) : T(+1);
    Matrix<T,ROW,ROW> m14 = m12 * m13;
    for (Size row(0u); row < ROW; ++row)
        for (Size col(0u); col < ROW; ++col)
            {
                T value = ((COL % 2) == 0) ? T(COL/2) : (T(COL-1)/2) - m12[row][COL-1];
                MI_CHECK_EQUAL (m14[row][col], value);
            }

    Matrix<T,ROW,COL> m15( m1);
    m15 *= T(3);
    Matrix<T,ROW,COL> m16;
    m16 = m1 * T(3);
    Matrix<T,ROW,COL> m17;
    m17 = T(3) * m1;
    Matrix<T,ROW,COL> m18 = m1 + m1 + m1;
    MI_CHECK_EQUAL( m15, m18);
    MI_CHECK_EQUAL( m16, m18);
    MI_CHECK_EQUAL( m17, m18);

    Vector<T,COL> c1;
    for (Size row(0u); row < COL; ++row)
        c1[row] = ((row % 2) == 0) ? T(-1) : T(+1);
    Vector<T,ROW> c2 = m12 * c1;
    for (Size row(0u); row < ROW; ++row)
        {
            T value = ((COL % 2) == 0) ? T(COL/2) : (T(COL-1)/2) - m12[row][COL-1];
            MI_CHECK_EQUAL( c2[row], value);
        }

    Vector<T,ROW> r1;
    for (Size col(0u); col < ROW; ++col)
        r1[col] = ((col % 2) == 0) ? T(-1) : T(+1);
    Vector<T,COL> r2 = r1 * m12;

    for (Size col(0u); col < COL; ++col)
        {
           T value = ((ROW % 2) == 0) ? T(ROW/2 * COL) : (T(ROW-1)/2 * COL) - m12[ROW-1][col];
           MI_CHECK_EQUAL( r2[col], value);
        }
}

MI_TEST_AUTO_FUNCTION( test_matrix_free_arithmetic_operators )
{
    test_matrix_free_arithmetic_operators<  Sint8,1,1>();
    test_matrix_free_arithmetic_operators< Sint16,1,1>();
    test_matrix_free_arithmetic_operators< Sint32,1,1>();
    test_matrix_free_arithmetic_operators< Sint64,1,1>();
    test_matrix_free_arithmetic_operators<Float32,1,1>();
    test_matrix_free_arithmetic_operators<Float64,1,1>();

    // test_matrix_free_arithmetic_operators<  Uint8,7,9>();
    test_matrix_free_arithmetic_operators< Sint16,7,9>();
    test_matrix_free_arithmetic_operators< Sint32,7,9>();
    test_matrix_free_arithmetic_operators< Sint64,7,9>();
    test_matrix_free_arithmetic_operators<Float32,7,9>();
    test_matrix_free_arithmetic_operators<Float64,7,9>();
}

template<typename T>
void test_matrix_det_inverse_transpose ()
{
    Matrix<T,4,4> m1;
    set_elements( m1);
    MI_CHECK_EQUAL( m1.det33(), T(0));

    Matrix<T,4,4> m2( 7, 3, 4, 2, 8, 10, 1, 13, 5, 9, 11, 15, 6, 12, 14, 15);
    MI_CHECK_EQUAL( m2.det33(), T(546));

    Matrix<T,4,4> m3( 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1);
    Matrix<T,4,4> m0( m3);
    bool b3 = m3.invert();
    MI_CHECK_EQUAL( b3, false);
    MI_CHECK_EQUAL( m3, m0);

    Matrix<T,1,1> m4( 0);
    Matrix<T,1,1> m5 = m4;
    bool b5 = m5.invert();
    MI_CHECK_EQUAL( b5, false);
    MI_CHECK_EQUAL( m5, m4);

    Matrix<T,1,1> m6( 7);
    bool b6 = m6.invert();
    MI_CHECK_EQUAL( b6, true);
    MI_CHECK_CLOSE( m6[0][0], T(1.0/7.0), 1e-6);

    Matrix<T,2,2> m7( 1, -1, 1, -1);
    Matrix<T,2,2> m8 = m7;
    bool b8 = m8.invert();
    MI_CHECK_EQUAL( b8, false);
    MI_CHECK_EQUAL( m8, m7);

    Matrix<T,2,2> m9( 1, -1, 1, 1);
    Matrix<T,2,2> m10( m9);
    bool b10 = m10.invert();
    MI_CHECK_EQUAL( b10, true);
    Matrix<T,2,2> m11 = m9 * m10;
    Matrix<T,2,2> m12 (1.0);
    MI_CHECK_CLOSE_MATRIX( m11, m12, 1e-5);

    Matrix<T,2,2> m13( 1, 2, 3, 4);
    Matrix<T,2,2> m14( m13);
    bool b14 = m14.invert();
    MI_CHECK_EQUAL( b14, true);
    Matrix<T,2,2> m15 = m13 * m14;
    Matrix<T,2,2> m16 (1.0);
    MI_CHECK_CLOSE_MATRIX( m15, m16, 1e-5);

    Matrix<T,4,4> m17 = m2;
    bool b17 = m17.invert();
    MI_CHECK_EQUAL( b17, true);
    b17 = m17.invert();
    MI_CHECK_EQUAL( b17, true);
    MI_CHECK_CLOSE_MATRIX( m17, m2, 1e-5);

    Matrix<T,3,3> m18 = sub_matrix<3,3>( m2);
    Matrix<T,3,3> m19( 7, 3, 4, 8, 10, 1, 5, 9, 11);
    MI_CHECK_CLOSE_MATRIX( m18, m19, 1e-5);

    Matrix<T,4,4> m20 = transpose( m2);
    Matrix<T,4,4> m21( m2);
    m21.transpose();
    Matrix<T,4,4> m22( 7, 8, 5, 6, 3, 10, 9, 12, 4, 1, 11, 14, 2, 13, 15, 15);
    MI_CHECK_EQUAL( m20, m22);
    MI_CHECK_EQUAL( m21, m22);
}

MI_TEST_AUTO_FUNCTION( test_matrix_det_inverse_transpose )
{
    test_matrix_det_inverse_transpose<Float32>();
    test_matrix_det_inverse_transpose<Float64>();
}

template<typename T>
void test_matrix_transform ()
{
    Matrix<T,4,4> M;
    set_elements( M);
    Vector<T,4> p4( 1, 2, 3, 4);
    Vector<T,3> p3( 1, 2, 3);
    Vector<T,3> v( p3);

    Vector<T,4> Mp4 = transform_point( M, p4);
    Vector<T,3> Mp3 = transform_point( M, p3);
    Vector<T,3> Mv  = transform_vector( M, v);

    Vector<T,4> q4( 80, 90, 100, 110);
    Vector<T,3> q3( 44.0/65.0, 51.0/65.0, 58.0/65.0);
    Vector<T,3> w( 32, 38, 44);
    MI_CHECK_CLOSE_MATRIX (Mp4, q4, 1e-5);
    MI_CHECK_CLOSE_MATRIX (Mp3, q3, 1e-5);
    MI_CHECK_CLOSE_MATRIX (Mv, w, 1e-5);

    Matrix<T,4,4> m2( 7, 3, 4, 2, 8, 10, 1, 13, 5, 9, 11, 15, 6, 12, 14, 15);
    Vector<T,3> n( 1, 2, 3);
    Vector<T,3> n2 = transform_normal_inv( m2, n);

    Matrix<T,3,3> A = sub_matrix<3,3>( m2);
    Matrix<T,4,4> m3 (1.0);
    for (Size row(0u); row < 3; ++row)
        for (Size col(0u); col < 3; ++col)
            m3[row][col] = A[row][col];
    bool b = m3.invert();
    MI_CHECK( b);
    Vector<T,3> n3 = transform_normal( m3, n);
    MI_CHECK_CLOSE_VECTOR( n2, n3, 1e-5);
}

MI_TEST_AUTO_FUNCTION( test_matrix_transform )
{
    test_matrix_transform<Float32>();
    test_matrix_transform<Float64>();
}

template<typename T>
void test_matrix_translation_rotation_lookat ()
{
    Matrix<T,4,4> m1;
    set_elements( m1);

    Matrix<T,4,4> m5( m1);
    m5.set_translation( Vector<T,3>( 1, 2, 3));
    MI_CHECK_EQUAL( m5[3][0], 1);
    MI_CHECK_EQUAL( m5[3][1], 2);
    MI_CHECK_EQUAL( m5[3][2], 3);
    m5.set_translation( 4, 5, 6);
    MI_CHECK_EQUAL( m5[3][0], 4);
    MI_CHECK_EQUAL( m5[3][1], 5);
    MI_CHECK_EQUAL( m5[3][2], 6);
    m5.translate( Vector<T,3>( 1, 2, 3));
    MI_CHECK_EQUAL( m5[3][0], 5);
    MI_CHECK_EQUAL( m5[3][1], 7);
    MI_CHECK_EQUAL( m5[3][2], 9);
    m5.translate( 1, 2, 3);
    MI_CHECK_EQUAL( m5[3][0], 6);
    MI_CHECK_EQUAL( m5[3][1], 9);
    MI_CHECK_EQUAL( m5[3][2], 12);

    Matrix<T,4,4> m6( 1);
    Matrix<T,4,4> m7( 1);

    m7.rotate( M_PI, 0, 0);
    m7.rotate( 2*M_PI/3, 0, 0);
    m7.rotate( Vector<T,3>( M_PI/3, 0, 0));
    MI_CHECK_CLOSE_MATRIX( m7, m6, 1e-5);

    m7.rotate( 0, M_PI, 0);
    m7.rotate( 0, 2*M_PI/3, 0);
    m7.rotate( Vector<T,3>( 0, M_PI/3, 0));
    MI_CHECK_CLOSE_MATRIX( m7, m6, 1e-5);

    m7.rotate( 0, 0, M_PI);
    m7.rotate( 0, 0, 2*M_PI/3);
    m7.rotate( Vector<T,3>( 0, 0, M_PI/3));
    MI_CHECK_CLOSE_MATRIX( m7, m6, 1e-5);

    m7.set_rotation( M_PI/2, M_PI/2, M_PI/2);
    Matrix<T,4,4> m8( 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);
    MI_CHECK_CLOSE_MATRIX( m7, m8, 1e-5);

    m7.set_rotation( Vector<T,3>( 1, 0, 0), M_PI/2);
    m6.set_rotation( M_PI/2, 0, 0);
    MI_CHECK_CLOSE_MATRIX( m7, m6, 1e-5);

    m7.set_rotation( Vector<T,3>( 0, 1, 0), M_PI/2);
    m6.set_rotation( 0, M_PI/2, 0);
    MI_CHECK_CLOSE_MATRIX( m7, m6, 1e-5);

    m7.set_rotation( Vector<T,3>( 0, 0, 1), M_PI/2);
    m6.set_rotation( 0, 0, M_PI/2);
    MI_CHECK_CLOSE_MATRIX( m7, m6, 1e-5);

    Vector<T,3> n( 1, 1, 1);
    n.normalize();
    m7.set_rotation( n, 2*M_PI/3);
    Matrix<T,4,4> m9( 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
    MI_CHECK_CLOSE_MATRIX( m7, m9, 1e-5);

    Vector<T,3> position( 1, 1, 1);
    Vector<T,3> target( 3, 1, 1);
    Vector<T,3> up( 0, -2, 0);
    Matrix<T,4,4> m10;
    m10.lookat( position, target, up);
    Vector<T,3> position2( 0, 0, 0);
    Vector<T,3> target2( 0, 0, -2);
    Vector<T,3> point( target+up);
    Vector<T,3> point2( 0, 2, -2);
    Vector<T,3> position3( transform_point( m10, position));
    Vector<T,3> target3( transform_point( m10, target));
    Vector<T,3> point3( transform_point( m10, point));
    MI_CHECK_CLOSE_MATRIX( position3, position2, 1e-5);
    MI_CHECK_CLOSE_MATRIX( target3, target2, 1e-5);
    MI_CHECK_CLOSE_MATRIX( point3, point2, 1e-5);
}

MI_TEST_AUTO_FUNCTION( test_matrix_translation_rotation_lookat )
{
    test_matrix_translation_rotation_lookat<Float32>();
    test_matrix_translation_rotation_lookat<Float64>();
}
