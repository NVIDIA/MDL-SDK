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

#include "pch.h"

#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <mi/base/config.h>

#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4305 ) // truncation from 'double' to 'mi::Float32'
#pragma warning( disable : 4244 ) // conversion from '...' to '...', possible loss of data
#endif

#include <base/system/test/i_test_auto_case.h>

#include <mi/math/vector.h>


///////////////////////////////////////////////////////////////////////////////
// Convenience operator for vector printing
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

///////////////////////////////////////////////////////////////////////////////
// Pseudo random number generation
///////////////////////////////////////////////////////////////////////////////


std::mt19937 prng;

void randomize(signed char& val)
{
    val = static_cast<signed char>(prng());
    if (prng() % 2)
        val *= -1;
}

void randomize(unsigned char& val)
{
    val = static_cast<unsigned char>(prng());
}

void randomize(int& val)
{
    val = static_cast<int>(prng());
    if (prng() % 2)
        val *= -1;
}

void randomize(unsigned int& val)
{
    val = static_cast<unsigned int>(prng());
}

void randomize(float& val)
{
    val = static_cast<float>(static_cast<double>(prng()) / prng.max());
    if (prng() % 2)
        val *= -1.0f;
}

void randomize(double &val)
{
    val = static_cast<double>(prng()) / prng.max();
    if (prng() % 2)
        val *= -1.0f;
}


template <class T, mi::Size DIM>
void randomize(mi::math::Vector<T,DIM> & v)
{
    for (mi::Size i( 0u ); i != DIM; ++i)
        randomize(v[i]);
}

///////////////////////////////////////////////////////////////////////////////
// Generic vector properties
///////////////////////////////////////////////////////////////////////////////

template <class T, mi::Size DIM>
void single_value_constructed_vector_has_that_value_in_all_elements(T val)
{
    mi::math::Vector<T, DIM> const vec( val );
    for (mi::Size i( 0u ); i != DIM; ++i)
        MI_CHECK_EQUAL(vec[i], val);
}

template <class Lhs, class Rhs>
void commutative_multiplication(Lhs lhs, Rhs rhs)
{
    std::ostringstream s;
    s << "lhs = " << lhs << ", rhs = " << rhs;
    MI_CHECK_EQUAL_MSG( lhs * rhs, rhs * lhs, s.str() );
}

template <class Lhs, class Rhs>
void commutative_addition(Lhs lhs, Rhs rhs)
{
    std::ostringstream s;
    s << "lhs = " << lhs << ", rhs = " << rhs;
    MI_CHECK_EQUAL_MSG( lhs + rhs, rhs + lhs, s.str() );
}

template <class T1, class T2>
void distributivity(T1 sum1, T1 sum2, T2 scalar)
{
    std::ostringstream s;
    s << "sum1 = " << sum1 << ", sum2 = " << sum2 << ", scalar = " << scalar;
    MI_CHECK_EQUAL_MSG( (sum1 + sum2) * scalar, (sum1 * scalar) + (sum2 * scalar), s.str() );
}

///////////////////////////////////////////////////////////////////////////////
// Generic property verification drivers
///////////////////////////////////////////////////////////////////////////////

template <class T>
void check_property(void (*f)(T))
{
    T t;
    for (unsigned i( 0u ); i != 100u; ++i)
    {
        randomize(t);
        (*f)(t);
    }
}

template <class T1, class T2>
void check_property(void (*f)(T1, T2))
{
    T1 t1;
    T2 t2;
    for (unsigned i( 0u ); i != 100u; ++i)
    {
        randomize(t1);
        randomize(t2);
        (*f)(t1, t2);
    }
}

template <class T1, class T2, class T3>
void check_property(void (*f)(T1, T2, T3))
{
    T1 t1;
    T2 t2;
    T3 t3;
    for (unsigned i( 0u ); i != 100u; ++i)
    {
        randomize(t1);
        randomize(t2);
        randomize(t3);
        (*f)(t1, t2, t3);
    }
}

#define TEST_VECTOR_PROPERTY(driver, ltype, rtype)      \
    driver(1,  ltype, rtype)                            \
    driver(2,  ltype, rtype)                            \
    driver(3,  ltype, rtype)                            \
    driver(4,  ltype, rtype)                            \
    driver(5,  ltype, rtype)                            \
    driver(13, ltype, rtype)

///////////////////////////////////////////////////////////////////////////////
// Vector unit tests
///////////////////////////////////////////////////////////////////////////////

#define SINGLE_VALUE_CONSTRUCTION(dim, type, ignore)                                                            \
  MI_TEST_AUTO_FUNCTION( verify_constructor_of_ ## dim ## _dimensional_ ## type ## _vector )                    \
  {                                                                                                             \
      check_property(&single_value_constructed_vector_has_that_value_in_all_elements<type, dim>);               \
  }

TEST_VECTOR_PROPERTY(SINGLE_VALUE_CONSTRUCTION, float,    void)
TEST_VECTOR_PROPERTY(SINGLE_VALUE_CONSTRUCTION, double,   void)
TEST_VECTOR_PROPERTY(SINGLE_VALUE_CONSTRUCTION, int,      void)
TEST_VECTOR_PROPERTY(SINGLE_VALUE_CONSTRUCTION, unsigned, void)

#define SCALAR_MULTIPLICATION_WITH_DIMENSION(dim, vtype, stype)                                                 \
  MI_TEST_AUTO_FUNCTION( multiply_ ## dim ## _dimensional_ ## vtype ## _vector_with_ ## stype ## _scalar )      \
  {                                                                                                             \
      check_property(&commutative_multiplication<mi::math::Vector<vtype,dim>,stype>);                           \
  }

TEST_VECTOR_PROPERTY(SCALAR_MULTIPLICATION_WITH_DIMENSION, float,  float)
TEST_VECTOR_PROPERTY(SCALAR_MULTIPLICATION_WITH_DIMENSION, float,  double)
TEST_VECTOR_PROPERTY(SCALAR_MULTIPLICATION_WITH_DIMENSION, double, double)
TEST_VECTOR_PROPERTY(SCALAR_MULTIPLICATION_WITH_DIMENSION, double, float)


#define VECTOR_MULTIPLICATION_WITH_DIMENSION(dim, type, ignored)                                                \
  MI_TEST_AUTO_FUNCTION( multiply_ ## dim ## _dimensional_ ## type ## _vectors )                                \
  {                                                                                                             \
      check_property(&commutative_multiplication< mi::math::Vector<type,dim>, mi::math::Vector<type,dim> >);    \
  }

TEST_VECTOR_PROPERTY(VECTOR_MULTIPLICATION_WITH_DIMENSION, float,    void)
TEST_VECTOR_PROPERTY(VECTOR_MULTIPLICATION_WITH_DIMENSION, double,   void)
TEST_VECTOR_PROPERTY(VECTOR_MULTIPLICATION_WITH_DIMENSION, int,      void)
TEST_VECTOR_PROPERTY(VECTOR_MULTIPLICATION_WITH_DIMENSION, unsigned, void)

#define VECTOR_ADDITION_WITH_DIMENSION(dim, type, ignored)                                                      \
  MI_TEST_AUTO_FUNCTION( add_ ## dim ## _dimensional_ ## type ## _vectors )                                     \
  {                                                                                                             \
      check_property(&commutative_addition< mi::math::Vector<type,dim>, mi::math::Vector<type,dim> >);          \
  }

TEST_VECTOR_PROPERTY(VECTOR_ADDITION_WITH_DIMENSION, float,    void)
TEST_VECTOR_PROPERTY(VECTOR_ADDITION_WITH_DIMENSION, double,   void)
TEST_VECTOR_PROPERTY(VECTOR_ADDITION_WITH_DIMENSION, int,      void)
TEST_VECTOR_PROPERTY(VECTOR_ADDITION_WITH_DIMENSION, unsigned, void)

#if 0
#define VECTOR_DISTRIBUTIVITY_WITH_DIMENSION(dim, sumtype, scaletype)                                                   \
MI_TEST_AUTO_FUNCTION( distributivity_of_ ## dim ## _dimensional_ ## sumtype ## _vectors_and_ ## scaletype ## _scalars )\
{                                                                                                                       \
    check_property(&distributivity< mi::math::Vector<sumtype,dim>, scaletype >);                                        \
}

TEST_VECTOR_PROPERTY(VECTOR_DISTRIBUTIVITY_WITH_DIMENSION, float,  float)
TEST_VECTOR_PROPERTY(VECTOR_DISTRIBUTIVITY_WITH_DIMENSION, float,  double)
TEST_VECTOR_PROPERTY(VECTOR_DISTRIBUTIVITY_WITH_DIMENSION, double, float)
TEST_VECTOR_PROPERTY(VECTOR_DISTRIBUTIVITY_WITH_DIMENSION, double, double)
#endif

MI_TEST_AUTO_FUNCTION( test_vector_function_normalize )
{
    using namespace mi;
    using Vector3 = math::Vector<Float32, 3>;
    Vector3 v3a(11, 22, 33);

    MI_CHECK( v3a.normalize());
    MI_CHECK_CLOSE( 1.0f, math::length( v3a), 1e-34f);
    MI_CHECK( ! Vector3(0).normalize());
}

#define MI_CHECK_CLOSE_VECTOR(x,y,eps) \
    MI_CHECK_CLOSE_COLLECTIONS((x).begin(),(x).end(),(y).begin(),(y).end(),eps)

#define MI_CHECK_CLOSE_VECTOR4_MSG(x,y,eps,s) \
    MI_CHECK_CLOSE_MSG((x)[0], (y)[0], (eps), (s)); \
    MI_CHECK_CLOSE_MSG((x)[1], (y)[1], (eps), (s)); \
    MI_CHECK_CLOSE_MSG((x)[2], (y)[2], (eps), (s)); \
    MI_CHECK_CLOSE_MSG((x)[3], (y)[3], (eps), (s))

const float eps = 1e-6f;

using namespace mi;
using namespace mi::math;

template <typename T>
void check_function (
        Vector<T,4> (*f) (const Vector_struct<T,4>&),
        const char* s,
        const Vector<T,4>& x,
        const Vector<T,4>& y,
        T prec = 1e-6)
{
    Vector<T,4> z = f(x);
    MI_CHECK_CLOSE_VECTOR4_MSG (z, y, prec, s);
}

template <typename T, Size DIM>
void set_elements (Vector<T,DIM>& v)
{
    for (Size i(0u); i < DIM; ++i)
        v[i] = T(i);
}

template <typename T, Size DIM>
bool check_elements (const Vector<T,DIM>& v)
{
    for (Size i(0u); i < DIM; ++i)
        if (v[i] != T(i))
            return false;
    return true;
}

template<typename T, Size DIM>
void test_vector_constructors ()
{
    Vector<T,DIM> v1;
    set_elements( v1);

    std::vector<T> v;
    for (Size i(0u); i < DIM; ++i)
        v.push_back( static_cast<T>( i));
    Vector<T,DIM> v2( FROM_ITERATOR, v.begin());

    T array1[DIM];
    for (Size i(0u); i < DIM; ++i)
        array1[i] = T(i);
    Vector<T,DIM> v3( array1);

    Uint8 array2[DIM];
    for (Size i(0u); i < DIM; ++i)
        array2[i] = Uint8(i);
    Vector<T,DIM> v4( array2);

    Vector<T,DIM> array3;
    set_elements( array3);
    Vector<T,DIM> v5( array3);

    Vector<Uint8,DIM> array4;
    set_elements( array4);
    const Vector<T,DIM> v6( array4);

    MI_CHECK( check_elements( v1));
    MI_CHECK_EQUAL( v1, v2);
    MI_CHECK_EQUAL( v1, v3);
    MI_CHECK_EQUAL( v1, v4);
    MI_CHECK_EQUAL( v1, v5);
    MI_CHECK_EQUAL( v1, v6);

    MI_CHECK( v1.SIZE == DIM);
    MI_CHECK( v1.size() == DIM);
    MI_CHECK( v1.max_size() == DIM);

    MI_CHECK( v1.begin() == & v1[0]);
    MI_CHECK( v6.begin() == & v6[0]);
    MI_CHECK( v1.end()-1 == & v1[DIM-1]);
    MI_CHECK( v6.end()-1 == & v6[DIM-1]);

    Vector<T,DIM> v7;
    v7 = v1;
    MI_CHECK_EQUAL( v7, v1);

    Vector<T,DIM> v8;
    v8 = v6;
    MI_CHECK_EQUAL( v8, v6);

    Vector<T,DIM> v9;
    v9 = T(42);
    Vector<T,DIM> v10( 42);
    MI_CHECK_EQUAL( v9, v10);
}

template<typename T>
void test_vector_constructors_2 ()
{
    Vector<T,2> v21( 1, 2);
    Vector<T,2> v22( 2, 3);
    Vector<T,2> v23( 3, 4);
    Vector<T,3> v31( 1, 2, 3);
    Vector<T,3> v32( 2, 3, 4);
    Vector<T,4> v41( 1, 2, 3, 4);

    Vector<T,2> v7( 1, 2);

    Vector<T,3> v8( 1, 2, 3);
    Vector<T,3> v9( 1, v22);
    Vector<T,3> v10( v21, T(3));

    Vector<T,4> v11( 1, 2, 3, 4);
    Vector<T,4> v12( 1, 2, v23);
    Vector<T,4> v13( 1, v22, 4);
    Vector<T,4> v14( v21, 3, 4);
    Vector<T,4> v15( v21, v23);
    Vector<T,4> v16( 1, v32);
    Vector<T,4> v17( v31, 4);

    MI_CHECK_EQUAL( v7, v21);
    MI_CHECK_EQUAL( v8, v31);
    MI_CHECK_EQUAL( v9, v31);
    MI_CHECK_EQUAL( v10, v31);
    MI_CHECK_EQUAL( v11, v41);
    MI_CHECK_EQUAL( v12, v41);
    MI_CHECK_EQUAL( v13, v41);
    MI_CHECK_EQUAL( v14, v41);
    MI_CHECK_EQUAL( v15, v41);
    MI_CHECK_EQUAL( v16, v41);
    MI_CHECK_EQUAL( v17, v41);

    Color_struct c;
    c.r = 1.0f;
    c.g = 2.0f;
    c.b = 3.0f;
    c.a = 4.0f;
    Vector<T,4> v18( c);
    Vector<T,4> v19;
    v19 = c;
    Vector<T,4> v20( 1, 2, 3, 4);
    MI_CHECK_EQUAL( v18, v20);
    MI_CHECK_EQUAL( v19, v20);
}

MI_TEST_AUTO_FUNCTION( test_vector_constructors )
{
    // test constructors not yet tested elsewhere
    test_vector_constructors<  Uint8,1>();
    test_vector_constructors< Uint16,1>();
    test_vector_constructors< Uint32,1>();
    test_vector_constructors< Uint64,1>();
    test_vector_constructors<  Sint8,1>();
    test_vector_constructors< Sint16,1>();
    test_vector_constructors< Sint32,1>();
    test_vector_constructors< Sint64,1>();
    test_vector_constructors<Float32,1>();
    test_vector_constructors<Float64,1>();

    test_vector_constructors<  Uint8,2>();
    test_vector_constructors< Uint16,2>();
    test_vector_constructors< Uint32,2>();
    test_vector_constructors< Uint64,2>();
    test_vector_constructors<  Sint8,2>();
    test_vector_constructors< Sint16,2>();
    test_vector_constructors< Sint32,2>();
    test_vector_constructors< Sint64,2>();
    test_vector_constructors<Float32,2>();
    test_vector_constructors<Float64,2>();

    test_vector_constructors<  Uint8,3>();
    test_vector_constructors< Uint16,3>();
    test_vector_constructors< Uint32,3>();
    test_vector_constructors< Uint64,3>();
    test_vector_constructors<  Sint8,3>();
    test_vector_constructors< Sint16,3>();
    test_vector_constructors< Sint32,3>();
    test_vector_constructors< Sint64,3>();
    test_vector_constructors<Float32,3>();
    test_vector_constructors<Float64,3>();

    test_vector_constructors<  Uint8,4>();
    test_vector_constructors< Uint16,4>();
    test_vector_constructors< Uint32,4>();
    test_vector_constructors< Uint64,4>();
    test_vector_constructors<  Sint8,4>();
    test_vector_constructors< Sint16,4>();
    test_vector_constructors< Sint32,4>();
    test_vector_constructors< Sint64,4>();
    test_vector_constructors<Float32,4>();
    test_vector_constructors<Float64,4>();

    test_vector_constructors<  Uint8,7>();
    test_vector_constructors< Uint16,7>();
    test_vector_constructors< Uint32,7>();
    test_vector_constructors< Uint64,7>();
    test_vector_constructors<  Sint8,7>();
    test_vector_constructors< Sint16,7>();
    test_vector_constructors< Sint32,7>();
    test_vector_constructors< Sint64,7>();
    test_vector_constructors<Float32,7>();
    test_vector_constructors<Float64,7>();

    test_vector_constructors_2<  Uint8>();
    test_vector_constructors_2< Uint16>();
    test_vector_constructors_2< Uint32>();
    test_vector_constructors_2< Uint64>();
    test_vector_constructors_2<  Sint8>();
    test_vector_constructors_2< Sint16>();
    test_vector_constructors_2< Sint32>();
    test_vector_constructors_2< Sint64>();
    test_vector_constructors_2<Float32>();
    test_vector_constructors_2<Float64>();
}

template<typename T, Size DIM>
void test_vector_arithmetic_operators ()
{
    Vector<T,DIM> v1;
    set_elements( v1);
    Vector<T,DIM> v2( v1 + v1);
    Vector<T,DIM> v3( v2 / T(2));
    Vector<T,DIM> v4( v2 / Vector<T,DIM> (T(2)));
    v2 /= T(2);
    MI_CHECK_CLOSE_VECTOR( v2, v1, eps);
    MI_CHECK_CLOSE_VECTOR( v3, v1, eps);
    MI_CHECK_CLOSE_VECTOR( v4, v1, eps);

    Vector<T,DIM> v5( -v1);
    v5 = -v5;
    MI_CHECK_CLOSE_VECTOR( v5, v1, eps);

    Vector<T,2> v6( 12, -34);
    Vector<T,2> v7( 56, -78);
    MI_CHECK_CLOSE( cross( v6, v7), T( 12*(-78) + 34*56), eps);

    Vector<T,3> v8( 1, 2, 3);
    Vector<T,3> v9( -4, -5, -6);
    Vector<T,3> v10( 3, -6, 3);
    Vector<T,3> v11 = cross( v8, v9);
    MI_CHECK_CLOSE_VECTOR( v11, v10, eps);
}

MI_TEST_AUTO_FUNCTION( test_vector_arithmetic_operators )
{
    // test arithmetic operators not yet tested elsewhere
    test_vector_arithmetic_operators<  Sint8,1>();
    test_vector_arithmetic_operators< Sint16,1>();
    test_vector_arithmetic_operators< Sint32,1>();
    test_vector_arithmetic_operators< Sint64,1>();
    test_vector_arithmetic_operators<Float32,1>();
    test_vector_arithmetic_operators<Float64,1>();

    test_vector_arithmetic_operators<  Sint8,2>();
    test_vector_arithmetic_operators< Sint16,2>();
    test_vector_arithmetic_operators< Sint32,2>();
    test_vector_arithmetic_operators< Sint64,2>();
    test_vector_arithmetic_operators<Float32,2>();
    test_vector_arithmetic_operators<Float64,2>();

    test_vector_arithmetic_operators<  Sint8,3>();
    test_vector_arithmetic_operators< Sint16,3>();
    test_vector_arithmetic_operators< Sint32,3>();
    test_vector_arithmetic_operators< Sint64,3>();
    test_vector_arithmetic_operators<Float32,3>();
    test_vector_arithmetic_operators<Float64,3>();

    test_vector_arithmetic_operators<  Sint8,4>();
    test_vector_arithmetic_operators< Sint16,4>();
    test_vector_arithmetic_operators< Sint32,4>();
    test_vector_arithmetic_operators< Sint64,4>();
    test_vector_arithmetic_operators<Float32,4>();
    test_vector_arithmetic_operators<Float64,4>();

    test_vector_arithmetic_operators<  Sint8,7>();
    test_vector_arithmetic_operators< Sint16,7>();
    test_vector_arithmetic_operators< Sint32,7>();
    test_vector_arithmetic_operators< Sint64,7>();
    test_vector_arithmetic_operators<Float32,7>();
    test_vector_arithmetic_operators<Float64,7>();
}

template<typename T>
void test_vector_make_basis ()
{
    Vector<T,3> n( 1, 0, 0);
    Vector<T,3> u( 0, 2, 3);
    Vector<T,3> v( 0, 4, 5);
    make_basis( n, &u, &v);
    MI_CHECK( abs( dot(n, u)) < eps);
    MI_CHECK( abs( dot(u, v)) < eps);
    MI_CHECK( abs( dot(v, n)) < eps);
    MI_CHECK_CLOSE( length( n), T(1), eps);
    MI_CHECK_CLOSE( length( u), T(1), eps);
    MI_CHECK_CLOSE( length( v), T(1), eps);

    n = Vector<T,3>( 0, 1, 0);
    u = Vector<T,3>( 2, 0, 3);
    v = Vector<T,3>( 4, 0, 5);
    make_basis( n, &u, &v);
    MI_CHECK( abs( dot(n, u)) < eps);
    MI_CHECK( abs( dot(u, v)) < eps);
    MI_CHECK( abs( dot(v, n)) < eps);
    MI_CHECK_CLOSE( length( n), T(1), eps);
    MI_CHECK_CLOSE( length( u), T(1), eps);
    MI_CHECK_CLOSE( length( v), T(1), eps);

    u = Vector<T,3>( 0, 4, 5);
    v = Vector<T,3>( 0, 2, 3);
    Vector<T,3> t;
    Vector<T,3> b;
    make_basis( n, u, v, &t, &b);
    MI_CHECK( abs( dot(n, t)) < eps);
    MI_CHECK( abs( dot(t, b)) < eps);
    MI_CHECK( abs( dot(b, n)) < eps);
    MI_CHECK_CLOSE( length( n), T(1), eps);
    MI_CHECK_CLOSE( length( t), T(1), eps);
    MI_CHECK_CLOSE( length( b), T(1), eps);
    Vector<T,3> nu( cross( n, u));
    MI_CHECK( abs( dot(t, nu)) < eps);
    MI_CHECK( dot(t, b) >= T(0));

    v = -v;
    make_basis( n, u, v, &t, &b);
    MI_CHECK( abs( dot(n, t)) < eps);
    MI_CHECK( abs( dot(t, b)) < eps);
    MI_CHECK( abs( dot(b, n)) < eps);
    MI_CHECK_CLOSE( length( n), T(1), eps);
    MI_CHECK_CLOSE( length( t), T(1), eps);
    MI_CHECK_CLOSE( length( b), T(1), eps);
    nu = cross( n, u);
    MI_CHECK( abs( dot(t, nu)) < eps);
    MI_CHECK( dot(t, b) >= T(0));
}

MI_TEST_AUTO_FUNCTION( test_vector_make_basis )
{
    test_vector_make_basis<Float32>();
    test_vector_make_basis<Float64>();
}

template<typename T>
void test_vector_function_overloads ()
{
    Vector<T,4> v1( 0.1f, 0.2f, 0.3f, 0.4f);
    Vector<T,4> v2( -0.1f, -0.2f, -0.3f, -0.4f);
    Vector<T,4> v3 (-1.2f, -0.1f, 0.5f, 1.6f);

    check_function<T> (abs, "abs", v2, v1);
    check_function<T> (acos, "acos", v1, Vector<T,4> ( 1.470628906, 1.369438406, 1.266103673, 1.159279481));
    check_function<T> (asin, "asin", v1, Vector<T,4> ( .1001674212, .2013579208, .3046926540, .4115168461));
    check_function<T> (atan, "atan", v1, Vector<T,4> ( .9966865249e-1, .1973955598, .2914567945, .3805063771));

    Vector<T,4> v16 = atan2 ( v1, v3);
    Vector<T,4> v17( 3.058451422, 2.034443936, .5404195003, .2449786631);
    MI_CHECK_CLOSE_VECTOR( v16, v17, eps);

    check_function<T> (ceil, "ceil", v3, Vector<T,4> ( -1.0f, 0.0f, 1.0f, 2.0f));

    Vector<T,4> v18( -1.0f, 0.0f, 0.0f, 0.0f);
    Vector<T,4> v19( 0.0f, 1.0f, 1.0f, 1.0f);
    Vector<T,4> v20 = clamp (v3, v18, v19);
    Vector<T,4> v21( -1.0f, 0.0f, 0.5f, 1.0f);
    MI_CHECK_CLOSE_VECTOR( v20, v21, eps);

    Vector<T,4> v22 = clamp( v3, T(0), T(1));
    Vector<T,4> v23( 0.0f, 0.0f, 0.5f, 1.0f);
    MI_CHECK_CLOSE_VECTOR( v22, v23, eps);

    check_function<T> (cos, "cos", v1, Vector<T,4> ( .9950041653, .9800665778, .9553364891, .9210609940));
    check_function<T> (degrees, "degrees", v1, Vector<T,4> ( 5.729577950, 11.45915590, 17.18873385, 22.91831180));

    Vector<T,4> v24 = elementwise_max (v2, v3);
    Vector<T,4> v25( -0.1f, -0.1f, 0.5f, 1.6f);
    MI_CHECK_CLOSE_VECTOR( v24, v25, eps);

    Vector<T,4> v26 = elementwise_min (v2, v3);
    Vector<T,4> v27( -1.2f, -0.2f, -0.3f, -0.4f);
    MI_CHECK_CLOSE_VECTOR( v26, v27, eps);

    check_function<T> (exp, "exp", v1, Vector<T,4> ( 1.105170918, 1.221402758, 1.349858808, 1.491824698));
    check_function<T> (exp2, "exp2", v1, Vector<T,4> ( 1.071773463, 1.148698355, 1.231144413, 1.319507911), T(1e-2f));
    check_function<T> (floor, "floor", v3, Vector<T,4> ( -2.0f, -1.0f, 0.0f, 1.0f));

    Vector<T,4> v28 = fmod (v3, v2);
    Vector<T,4> v29( 0.0f, -0.1f, 0.2f, 0.0f);
    MI_CHECK_CLOSE_VECTOR( v28, v29, eps);

    Vector<T,4> v30 = fmod (v3, T(0.5f));
    Vector<T,4> v31( -0.2f, -0.1f, 0.0f, 0.1f);
    MI_CHECK_CLOSE_VECTOR( v30, v31, eps);

    check_function<T> (frac, "frac", v3, Vector<T,4> ( 0.8f, 0.9f, 0.5f, 0.6f));

    Vector<T,4> v34 = v1 + Vector<T,4>( 1e-6);
    MI_CHECK( is_approx_equal( v34, v1, T(1e-5f)));
    MI_CHECK( ! is_approx_equal( v34, v1, T(1e-7f)));

    Vector<T,4> v35 = lerp (v2, v3, v1);
    Vector<T,4> v36( -0.21f, -0.18f, -0.06f, 0.4f);
    MI_CHECK_CLOSE_VECTOR( v35, v36, eps);

    Vector<T,4> v38( -0.21f, -0.15f, 0.1f, 0.6f);
    MI_CHECK_CLOSE_VECTOR( v35, v36, eps);

    check_function<T> (log, "log", v1, Vector<T,4> ( -2.302585093, -1.609437912, -1.203972804, -.9162907319));
    check_function<T> (log2, "log2", v1, Vector<T,4> ( -3.321928095, -2.321928095, -1.736965594, -1.321928095));
    check_function<T> (log10, "log10", v1, Vector<T,4> ( -1.000000000, -.6989700043, -.5228787453, -.3979400087));

    Vector<T,4> v39;
    Vector<T,4> v40 = modf (v3, v39);
    Vector<T,4> v41( -1.0f, 0.0f, 0.0f, 1.0f);
    Vector<T,4> v42( -0.2f, -0.1f, 0.5f, 0.6f);
    MI_CHECK_CLOSE_VECTOR( v39, v41, eps);
    MI_CHECK_CLOSE_VECTOR( v40, v42, eps);

    Vector<T,4> v43 = pow (v1, v3);
    Vector<T,4> v44( 15.84893192, 1.174618943, .5477225575, .2308319849);
    MI_CHECK_CLOSE_VECTOR( v43, v44, T(1e-5f));

    Vector<T,4> v45 = pow (v1, T(0.5f));
    Vector<T,4> v46( .3162277660, .4472135955, .5477225575, .6324555320);
    MI_CHECK_CLOSE_VECTOR( v45, v46, eps);

    check_function<T> (radians, "radians", v1, Vector<T,4> ( .1745329252e-2, .3490658504e-2, .5235987758e-2, .6981317008e-2));
    check_function<T> (round, "round", v3, Vector<T,4> ( -1.0f, 0.0f, 1.0f, 2.0f));
    check_function<T> (rsqrt, "rsqrt", v1, Vector<T,4> ( 3.162277660, 2.236067977, 1.825741858, 1.581138830));
    check_function<T> (saturate, "saturate", v3, Vector<T,4> ( 0.0f, 0.0f, 0.5f, 1.0f));
    check_function<T> (sign, "sign", v3, Vector<T,4> ( -1.0f, -1.0f, 1.0f, 1.0f));
    check_function<T> (sin, "sin", v1, Vector<T,4> ( .9983341665e-1, .1986693308, .2955202067, .3894183423));

    Vector<T,4> v47;
    Vector<T,4> v48;
    sincos( v3, v47, v48);
    Vector<T,4> v49 = sin( v3);
    Vector<T,4> v50 = cos( v3);
    MI_CHECK_CLOSE_VECTOR ( v47, v49, eps);
    MI_CHECK_CLOSE_VECTOR ( v48, v50, eps);

    Vector<T,4> v57( v1);
    v57[0] = -1.0f;
    Vector<T,4> v51 = smoothstep (v2, v3, v57);
    Vector<T,4> v52( 0.0f, 1.0f, .84375, .352f);
    MI_CHECK_CLOSE_VECTOR( v51, v52, eps);

    Vector<T,4> v53 = smoothstep (v2, v3, T(0.5f));
    Vector<T,4> v54( 1.0f, 1.0f, 1.0f, .42525f);
    MI_CHECK_CLOSE_VECTOR( v53, v54, eps);

    check_function<T> (sqrt, "sqrt", v1, Vector<T,4> ( .3162277660, .4472135955, .5477225575, .6324555320));

    Vector<T,4> v55 = step( v3, v2);
    Vector<T,4> v56( 1.0f, 0.0f, 0.0f, 0.0f);
    MI_CHECK_CLOSE_VECTOR (v55, v56, eps);

    check_function<T> (tan, "tan", v1, Vector<T,4> ( .1003346721, .2027100355, .3093362496, .4227932187));
}

MI_TEST_AUTO_FUNCTION( test_vector_function_overloads )
{
    test_vector_function_overloads<Float32>();
    test_vector_function_overloads<Float64>();
}

MI_TEST_AUTO_FUNCTION( test_vector_function_access_get_set_dim1 )
{ // test vector accessors, get, set, dim 1
    using namespace mi;
    using Vector1 = math::Vector<Float32, 1>;
    Vector1 v1a(11);
    MI_CHECK_EQUAL( v1a.begin() + 1, v1a.end());
    MI_CHECK_EQUAL( 11, v1a.x);
    MI_CHECK_EQUAL( 11, v1a[0]);
    MI_CHECK_EQUAL( 11, v1a.begin()[0]);
    MI_CHECK_EQUAL( 11, v1a.get(0));
    v1a.set(0,44);
    MI_CHECK_EQUAL( 44, v1a[0]);
    v1a[0] = 11;
    MI_CHECK_EQUAL( 11, v1a[0]);
    v1a.begin()[0] = 44;
    MI_CHECK_EQUAL( 44, v1a[0]);
}

MI_TEST_AUTO_FUNCTION( test_vector_function_access_get_set_dim2 )
{ // test vector accessors, get, set, dim 2
    using namespace mi;
    using Vector2 = math::Vector<Float32, 2>;
    Vector2 v2a(11, 22);
    MI_CHECK_EQUAL( v2a.begin() + 2, v2a.end());
    MI_CHECK_EQUAL( 11, v2a.x);
    MI_CHECK_EQUAL( 22, v2a.y);
    MI_CHECK_EQUAL( 11, v2a[0]);
    MI_CHECK_EQUAL( 22, v2a[1]);
    MI_CHECK_EQUAL( 11, v2a.begin()[0]);
    MI_CHECK_EQUAL( 22, v2a.begin()[1]);
    MI_CHECK_EQUAL( 11, v2a.get(0));
    MI_CHECK_EQUAL( 22, v2a.get(1));
    v2a.set(0,44);
    v2a.set(1,55);
    MI_CHECK_EQUAL( 44, v2a[0]);
    MI_CHECK_EQUAL( 55, v2a[1]);
    v2a[0] = 11;
    v2a[1] = 22;
    MI_CHECK_EQUAL( 11, v2a[0]);
    MI_CHECK_EQUAL( 22, v2a[1]);
    v2a.begin()[0] = 44;
    v2a.begin()[1] = 55;
    MI_CHECK_EQUAL( 44, v2a[0]);
    MI_CHECK_EQUAL( 55, v2a[1]);
}

MI_TEST_AUTO_FUNCTION( test_vector_function_access_get_set_dim3 )
{ // test vector accessors, get, set, dim 3
    using namespace mi;
    using Vector3 = math::Vector<Float32, 3>;
    Vector3 v3a(11, 22, 33);
    MI_CHECK_EQUAL( v3a.begin() + 3, v3a.end());
    MI_CHECK_EQUAL( 11, v3a.x);
    MI_CHECK_EQUAL( 22, v3a.y);
    MI_CHECK_EQUAL( 33, v3a.z);
    MI_CHECK_EQUAL( 11, v3a[0]);
    MI_CHECK_EQUAL( 22, v3a[1]);
    MI_CHECK_EQUAL( 33, v3a[2]);
    MI_CHECK_EQUAL( 11, v3a.begin()[0]);
    MI_CHECK_EQUAL( 22, v3a.begin()[1]);
    MI_CHECK_EQUAL( 33, v3a.begin()[2]);
    MI_CHECK_EQUAL( 11, v3a.get(0));
    MI_CHECK_EQUAL( 22, v3a.get(1));
    MI_CHECK_EQUAL( 33, v3a.get(2));
    v3a.set(0,44);
    v3a.set(1,55);
    v3a.set(2,66);
    MI_CHECK_EQUAL( 44, v3a[0]);
    MI_CHECK_EQUAL( 55, v3a[1]);
    MI_CHECK_EQUAL( 66, v3a[2]);
    v3a[0] = 11;
    v3a[1] = 22;
    v3a[2] = 33;
    MI_CHECK_EQUAL( 11, v3a[0]);
    MI_CHECK_EQUAL( 22, v3a[1]);
    MI_CHECK_EQUAL( 33, v3a[2]);
    v3a.begin()[0] = 44;
    v3a.begin()[1] = 55;
    v3a.begin()[2] = 66;
    MI_CHECK_EQUAL( 44, v3a[0]);
    MI_CHECK_EQUAL( 55, v3a[1]);
    MI_CHECK_EQUAL( 66, v3a[2]);
}

MI_TEST_AUTO_FUNCTION( test_vector_function_access_get_set_dim4 )
{ // test vector accessors, get, set, dim 4
    using namespace mi;
    using Vector4 = math::Vector<Float32, 4>;
    Vector4 v4a(11, 22, 33, 44);
    MI_CHECK_EQUAL( v4a.begin() + 4, v4a.end());
    MI_CHECK_EQUAL( 11, v4a.x);
    MI_CHECK_EQUAL( 22, v4a.y);
    MI_CHECK_EQUAL( 33, v4a.z);
    MI_CHECK_EQUAL( 44, v4a.w);
    MI_CHECK_EQUAL( 11, v4a[0]);
    MI_CHECK_EQUAL( 22, v4a[1]);
    MI_CHECK_EQUAL( 33, v4a[2]);
    MI_CHECK_EQUAL( 44, v4a[3]);
    MI_CHECK_EQUAL( 11, v4a.begin()[0]);
    MI_CHECK_EQUAL( 22, v4a.begin()[1]);
    MI_CHECK_EQUAL( 33, v4a.begin()[2]);
    MI_CHECK_EQUAL( 44, v4a.begin()[3]);
    MI_CHECK_EQUAL( 11, v4a.get(0));
    MI_CHECK_EQUAL( 22, v4a.get(1));
    MI_CHECK_EQUAL( 33, v4a.get(2));
    MI_CHECK_EQUAL( 44, v4a.get(3));
    v4a.set(0,66);
    v4a.set(1,77);
    v4a.set(2,88);
    v4a.set(3,99);
    MI_CHECK_EQUAL( 66, v4a[0]);
    MI_CHECK_EQUAL( 77, v4a[1]);
    MI_CHECK_EQUAL( 88, v4a[2]);
    MI_CHECK_EQUAL( 99, v4a[3]);
    v4a[0] = 11;
    v4a[1] = 22;
    v4a[2] = 33;
    v4a[3] = 44;
    MI_CHECK_EQUAL( 11, v4a[0]);
    MI_CHECK_EQUAL( 22, v4a[1]);
    MI_CHECK_EQUAL( 33, v4a[2]);
    MI_CHECK_EQUAL( 44, v4a[3]);
    v4a.begin()[0] = 66;
    v4a.begin()[1] = 77;
    v4a.begin()[2] = 88;
    v4a.begin()[3] = 99;
    MI_CHECK_EQUAL( 66, v4a[0]);
    MI_CHECK_EQUAL( 77, v4a[1]);
    MI_CHECK_EQUAL( 88, v4a[2]);
    MI_CHECK_EQUAL( 99, v4a[3]);
}

MI_TEST_AUTO_FUNCTION( test_vector_function_access_get_set_dim5 )
{ // test vector accessors, get, set, dim 5
    using namespace mi;
    using Vector5 = math::Vector<Float32, 5>;
    Float32 data[5] = {11, 22, 33, 44, 55};
    Vector5 v5a(data);
    MI_CHECK_EQUAL( v5a.begin() + 5, v5a.end());
    MI_CHECK_EQUAL( 11, v5a[0]);
    MI_CHECK_EQUAL( 22, v5a[1]);
    MI_CHECK_EQUAL( 33, v5a[2]);
    MI_CHECK_EQUAL( 44, v5a[3]);
    MI_CHECK_EQUAL( 55, v5a[4]);
    MI_CHECK_EQUAL( 11, v5a.begin()[0]);
    MI_CHECK_EQUAL( 22, v5a.begin()[1]);
    MI_CHECK_EQUAL( 33, v5a.begin()[2]);
    MI_CHECK_EQUAL( 44, v5a.begin()[3]);
    MI_CHECK_EQUAL( 55, v5a.begin()[4]);
    MI_CHECK_EQUAL( 11, v5a.get(0));
    MI_CHECK_EQUAL( 22, v5a.get(1));
    MI_CHECK_EQUAL( 33, v5a.get(2));
    MI_CHECK_EQUAL( 44, v5a.get(3));
    MI_CHECK_EQUAL( 55, v5a.get(4));
    v5a.set(0,66);
    v5a.set(1,77);
    v5a.set(2,88);
    v5a.set(3,99);
    v5a.set(4,111);
    MI_CHECK_EQUAL( 66, v5a[0]);
    MI_CHECK_EQUAL( 77, v5a[1]);
    MI_CHECK_EQUAL( 88, v5a[2]);
    MI_CHECK_EQUAL( 99, v5a[3]);
    MI_CHECK_EQUAL( 111,v5a[4]);
    v5a[0] = 11;
    v5a[1] = 22;
    v5a[2] = 33;
    v5a[3] = 44;
    v5a[4] = 55;
    MI_CHECK_EQUAL( 11, v5a[0]);
    MI_CHECK_EQUAL( 22, v5a[1]);
    MI_CHECK_EQUAL( 33, v5a[2]);
    MI_CHECK_EQUAL( 44, v5a[3]);
    MI_CHECK_EQUAL( 55, v5a[4]);
    v5a.begin()[0] = 66;
    v5a.begin()[1] = 77;
    v5a.begin()[2] = 88;
    v5a.begin()[3] = 99;
    v5a.begin()[4] = 111;
    MI_CHECK_EQUAL( 66, v5a[0]);
    MI_CHECK_EQUAL( 77, v5a[1]);
    MI_CHECK_EQUAL( 88, v5a[2]);
    MI_CHECK_EQUAL( 99, v5a[3]);
    MI_CHECK_EQUAL( 111,v5a[4]);
}

MI_TEST_AUTO_FUNCTION( test_vector_function_any )
{
    using mi::math::any;
    using Vector3  = mi::math::Vector<mi::Float32, 3>;
    using VectorI3 = mi::math::Vector<mi::Sint32, 3>;
    using VectorU3 = mi::math::Vector<mi::Uint8, 3>;
    using VectorB3 = mi::math::Vector<bool, 3>;
    MI_CHECK(   any(1));
    MI_CHECK( ! any(0));
    MI_CHECK(   any(1.0));
    MI_CHECK( ! any(0.0));
    MI_CHECK(   any(true));
    MI_CHECK( ! any(false));
    MI_CHECK(   any(Vector3(1.0, 0.0, 0.0)));
    MI_CHECK(   any(Vector3(0.0, 1.0, 0.0)));
    MI_CHECK(   any(Vector3(0.0, 0.0, 1.0)));
    MI_CHECK(   any(Vector3(0.0, 1.0, 1.0)));
    MI_CHECK(   any(Vector3(1.0, 1.0, 1.0)));
    MI_CHECK( ! any(Vector3(0.0, 0.0, 0.0)));
    MI_CHECK(   any(VectorI3(1, 0, 0)));
    MI_CHECK(   any(VectorI3(0, 1, 0)));
    MI_CHECK(   any(VectorI3(0, 0, 1)));
    MI_CHECK(   any(VectorI3(0, 1, 1)));
    MI_CHECK(   any(VectorI3(1, 1, 1)));
    MI_CHECK( ! any(VectorI3(0, 0, 0)));
    MI_CHECK(   any(VectorU3(1, 0, 0)));
    MI_CHECK(   any(VectorU3(0, 1, 0)));
    MI_CHECK(   any(VectorU3(0, 0, 1)));
    MI_CHECK(   any(VectorU3(0, 1, 1)));
    MI_CHECK(   any(VectorU3(1, 1, 1)));
    MI_CHECK( ! any(VectorU3(0, 0, 0)));
    MI_CHECK(   any(VectorB3( true, false, false)));
    MI_CHECK(   any(VectorB3(false,  true, false)));
    MI_CHECK(   any(VectorB3(false, false,  true)));
    MI_CHECK(   any(VectorB3(false,  true,  true)));
    MI_CHECK(   any(VectorB3( true,  true,  true)));
    MI_CHECK( ! any(VectorB3(false, false, false)));
}

MI_TEST_AUTO_FUNCTION( test_vector_function_all )
{
    using mi::math::all;
    using Vector3  = mi::math::Vector<mi::Float32, 3>;
    using VectorI3 = mi::math::Vector<mi::Sint32, 3>;
    using VectorU3 = mi::math::Vector<mi::Uint8, 3>;
    using VectorB3 = mi::math::Vector<bool, 3>;
    MI_CHECK(   all(1));
    MI_CHECK( ! all(0));
    MI_CHECK(   all(1.0));
    MI_CHECK( ! all(0.0));
    MI_CHECK(   all(true));
    MI_CHECK( ! all(false));
    MI_CHECK( ! all(Vector3(1.0, 0.0, 0.0)));
    MI_CHECK( ! all(Vector3(0.0, 1.0, 0.0)));
    MI_CHECK( ! all(Vector3(0.0, 0.0, 1.0)));
    MI_CHECK( ! all(Vector3(0.0, 1.0, 1.0)));
    MI_CHECK(   all(Vector3(1.0, 1.0, 1.0)));
    MI_CHECK( ! all(Vector3(0.0, 0.0, 0.0)));
    MI_CHECK( ! all(VectorI3(1, 0, 0)));
    MI_CHECK( ! all(VectorI3(0, 1, 0)));
    MI_CHECK( ! all(VectorI3(0, 0, 1)));
    MI_CHECK( ! all(VectorI3(0, 1, 1)));
    MI_CHECK(   all(VectorI3(1, 1, 1)));
    MI_CHECK( ! all(VectorI3(0, 0, 0)));
    MI_CHECK( ! all(VectorU3(1, 0, 0)));
    MI_CHECK( ! all(VectorU3(0, 1, 0)));
    MI_CHECK( ! all(VectorU3(0, 0, 1)));
    MI_CHECK( ! all(VectorU3(0, 1, 1)));
    MI_CHECK(   all(VectorU3(1, 1, 1)));
    MI_CHECK( ! all(VectorU3(0, 0, 0)));
    MI_CHECK( ! all(VectorB3( true, false, false)));
    MI_CHECK( ! all(VectorB3(false,  true, false)));
    MI_CHECK( ! all(VectorB3(false, false,  true)));
    MI_CHECK( ! all(VectorB3(false,  true,  true)));
    MI_CHECK(   all(VectorB3( true,  true,  true)));
    MI_CHECK( ! all(VectorB3(false, false, false)));
}

MI_TEST_AUTO_FUNCTION( test_vector_function_elementwise_comparisons )
{
    using mi::math::any;
    using Vector3  = mi::math::Vector<mi::Float32, 3>;
    using VectorB3 = mi::math::Vector<bool, 3>;
    Vector3 v1(1.0, 3.0, 4.0);
    Vector3 v2(1.0, 4.0, 4.0);
    Vector3 v3(1.0, 4.0, 5.0);
    Vector3 v4(0.0, 4.0, 5.0);
    VectorB3 b1 = mi::math::elementwise_is_equal( v1, v1);
    VectorB3 b2 = mi::math::elementwise_is_equal( v1, v2);
    VectorB3 b3 = mi::math::elementwise_is_equal( v1, v3);
    VectorB3 b4 = mi::math::elementwise_is_equal( v1, v4);
    MI_CHECK_EQUAL( b1, VectorB3(  true,  true,  true));
    MI_CHECK_EQUAL( b2, VectorB3(  true, false,  true));
    MI_CHECK_EQUAL( b3, VectorB3(  true, false, false));
    MI_CHECK_EQUAL( b4, VectorB3( false, false, false));
    MI_CHECK( all(b1) && any(b2) && any(b3) && ! any(b4));

    MI_CHECK_EQUAL( VectorB3(true,false,true),
                    mi::math::elementwise_is_not_equal( v2, v4));
    MI_CHECK_EQUAL( VectorB3(false,false,true),
                    mi::math::elementwise_is_less_than( v2, v4));
    MI_CHECK_EQUAL( VectorB3(false,true,true),
                    mi::math::elementwise_is_less_than_or_equal( v2, v4));
    MI_CHECK_EQUAL( VectorB3(true,false,false),
                    mi::math::elementwise_is_greater_than( v2, v4));
    MI_CHECK_EQUAL( VectorB3(true,true,false),
                    mi::math::elementwise_is_greater_than_or_equal( v2, v4));
}

MI_TEST_AUTO_FUNCTION( test_vector_free_comparison_operators )
{
    using Vector2 = mi::math::Vector<mi::Float32, 2>;
    Vector2 v1(1.0, 3.0);
    Vector2 v2(1.0, 4.0);
    Vector2 v3(0.0, 4.0);
    MI_CHECK(    v1 == v1);
    MI_CHECK( ! (v1 != v1));
    MI_CHECK( ! (v1 <  v1));
    MI_CHECK(    v1 <= v1);
    MI_CHECK( ! (v1 >  v1));
    MI_CHECK(    v1 >= v1);
    MI_CHECK( ! (v1 == v2));
    MI_CHECK(    v1 != v2);
    MI_CHECK(    v1 <  v2);
    MI_CHECK(    v1 <= v2);
    MI_CHECK( ! (v1 >  v2));
    MI_CHECK( ! (v1 >= v2));
    MI_CHECK( ! (v1 == v3));
    MI_CHECK(    v1 != v3);
    MI_CHECK( ! (v1 <  v3));
    MI_CHECK( ! (v1 <= v3));
    MI_CHECK(    v1 >  v3);
    MI_CHECK(    v1 >= v3);
}

MI_TEST_AUTO_FUNCTION( test_vector_free_logical_operators )
{
    using Vector4 = mi::math::Vector<bool, 4>;
    Vector4 v1( false, false,  true, true);
    Vector4 v2( false,  true, false, true);
    MI_CHECK_EQUAL( Vector4( false, false, false,  true), v1 && v2);
    MI_CHECK_EQUAL( Vector4( false,  true,  true,  true), v1 || v2);
    MI_CHECK_EQUAL( Vector4( false,  true,  true, false), v1 ^ v2);
    MI_CHECK_EQUAL( Vector4(  true,  true, false, false), ! v1);
    MI_CHECK_EQUAL( Vector4( false, false,  true,  true), v1 && true);
    MI_CHECK_EQUAL( Vector4( false, false, false, false), v1 && false);
    MI_CHECK_EQUAL( Vector4(  true,  true,  true,  true), v1 || true);
    MI_CHECK_EQUAL( Vector4( false, false,  true,  true), v1 || false);
    MI_CHECK_EQUAL( Vector4(  true,  true, false, false), v1 ^ true);
    MI_CHECK_EQUAL( Vector4( false, false,  true,  true), v1 ^ false);
    MI_CHECK_EQUAL( Vector4( false,  true, false,  true), true && v2);
    MI_CHECK_EQUAL( Vector4( false, false, false, false), false && v2);
    MI_CHECK_EQUAL( Vector4(  true,  true,  true,  true), true || v2);
    MI_CHECK_EQUAL( Vector4( false,  true, false,  true), false || v2);
    MI_CHECK_EQUAL( Vector4(  true, false,  true, false), true ^ v2);
    MI_CHECK_EQUAL( Vector4( false,  true, false,  true), false ^ v2);
}

MI_TEST_AUTO_FUNCTION( test_vector_incr_decr_operators )
{
    using Vector2 = mi::math::Vector<mi::Sint32, 2>;
    Vector2 v1(22,33);
    Vector2 v2 = ++v1;
    MI_CHECK_EQUAL( Vector2(23,34), v1);
    MI_CHECK_EQUAL( Vector2(23,34), v2);
    Vector2 v3 = --v1;
    MI_CHECK_EQUAL( Vector2(22,33), v1);
    MI_CHECK_EQUAL( Vector2(22,33), v3);
    MI_CHECK_EQUAL( Vector2(23,34), v2);
}

MI_TEST_AUTO_FUNCTION( test_convert_vector )
{
    const mi::math::Vector<mi::Float32, 3> v(1.f, 2.5f, 3.f);
    MI_CHECK_EQUAL((mi::math::convert_vector<mi::Sint32, 2>(v, 1)),
                   (mi::math::Vector<mi::Sint32, 2>(1, 2)));
    MI_CHECK_EQUAL((mi::math::convert_vector<mi::Sint32, 3>(v, 1)),
                   (mi::math::Vector<mi::Sint32, 3>(1, 2, 3)));
    MI_CHECK_EQUAL((mi::math::convert_vector<mi::Sint32, 4>(v, 2)),
                   (mi::math::Vector<mi::Sint32, 4>(1, 2, 3, 2)));
    const Sint32 v5[] = {1, 2, 3, 0, 0};
    MI_CHECK_EQUAL((mi::math::convert_vector<mi::Sint32, 5>(v)),
                   (mi::math::Vector<mi::Sint32, 5>(v5)));
}

