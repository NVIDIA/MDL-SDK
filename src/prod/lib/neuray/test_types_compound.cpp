/******************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Test for mi::ICompound
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <sstream>
#include <vector>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/ibbox.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/ispectrum.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ivector.h>

#include <mi/neuraylib/iattribute_container.h>

#include "test_shared.h"

#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4800 )
#endif

// compare data in source and buffer arrays
template<class T, class U>
bool verify_buffer( U* buffer, T* source, mi::Size N)
{
    for( mi::Size i = 0; i < N; ++i)
        if( static_cast<U>( source[i]) != buffer[i])
            return false;
    return true;
}

// read data from compound via get_values() and verify
template<class T, class U>
void verify_values( mi::ICompound* compound, T* source, mi::Size N)
{
    U* buffer = new U[N];
    compound->get_values( buffer);
    MI_CHECK( verify_buffer( buffer, source, N));
    delete[] buffer;
}

// read data from compound via get_value() and verify
template<class T, class U>
void verify_value( mi::ICompound* compound, T* source, mi::Size m, mi::Size n)
{
    U* buffer = new U[m*n];
    for( mi::Size i = 0; i < m; ++i)
        for( mi::Size j = 0; j < n; ++j) {
            MI_CHECK( compound->get_value( i, j, buffer[i*n+j]));
        }
    MI_CHECK( verify_buffer( buffer, source, m*n));

    for( mi::Size i = 0; i < m; ++i)
        for( mi::Size j = 0; j < n; ++j) {
            buffer[i*n+j] = compound->get_value<U>( i, j);
            MI_CHECK( buffer[i*n+j]);
        }
    MI_CHECK( verify_buffer( buffer, source, m*n));
    delete[] buffer;
}

// test data
const mi::Size N = 16;
double data1[N]
    = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 };
double data2[N]
    = { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.0, 2.1, 2.2, 2.3, 2.4, 2.5 };

// test mi::ICompound interface
template <class T, mi::Size ROWS, mi::Size COLUMNS>
void test_compound(
    mi::neuraylib::ITransaction* transaction,
    mi::ICompound* compound,
    const char* type_name,
    const char* element_type_name)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    // check type name

    MI_CHECK_EQUAL_CSTR( compound->get_type_name(), type_name);

    // check get_key()

    const char* s;
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i) {
        s = compound->get_key( i);
        MI_CHECK( s);
        MI_CHECK( compound->has_key( s));
    }
    MI_CHECK( !compound->get_key( ROWS*COLUMNS));

    // check has_key()

    MI_CHECK( !compound->has_key( nullptr));
    MI_CHECK( !compound->has_key( ""));
    MI_CHECK( !compound->has_key( "c"));
    MI_CHECK( !compound->has_key( "cc"));
    MI_CHECK( !compound->has_key( "ccc"));
    if( strcmp( type_name, "Color") == 0) {
        MI_CHECK( compound->has_key( "r"));
        MI_CHECK( compound->has_key( "g"));
        MI_CHECK( compound->has_key( "b"));
        MI_CHECK( compound->has_key( "a"));
    } else if( strcmp( type_name, "Color3") == 0) {
        MI_CHECK( compound->has_key( "r"));
        MI_CHECK( compound->has_key( "g"));
        MI_CHECK( compound->has_key( "b"));
    } else if( strcmp( type_name, "Spectrum") == 0) {
        MI_CHECK( compound->has_key( "0"));
        MI_CHECK( compound->has_key( "1"));
        MI_CHECK( compound->has_key( "2"));
    } else if( strcmp( type_name, "Bbox3") == 0) {
        MI_CHECK( compound->has_key( "min_x"));
        MI_CHECK( compound->has_key( "min_y"));
        MI_CHECK( compound->has_key( "min_z"));
        MI_CHECK( compound->has_key( "max_x"));
        MI_CHECK( compound->has_key( "max_y"));
        MI_CHECK( compound->has_key( "max_z"));
    } else if( COLUMNS == 1) {
        for( mi::Size row = 0; row < ROWS; ++row) {
            std::string key;
            key += static_cast<char>( ('x' + (row == 3 ? -1 : row)));
            MI_CHECK( compound->has_key( key.c_str()));
        }
    } else {
        for( mi::Size row = 0; row < ROWS; ++row)
            for( mi::Size column = 0; column < COLUMNS; ++column) {
                std::string key;
                key += static_cast<char>( ('x' + (row    == 3 ? -1 : row   )));
                key += static_cast<char>( ('x' + (column == 3 ? -1 : column)));
                MI_CHECK( compound->has_key( key.c_str()));
        }
    }

    // check dimensions

    mi::Size m = compound->get_number_of_rows();
    mi::Size n = compound->get_number_of_columns();
    assert( m*n <= N);
    MI_CHECK_EQUAL( m*n, compound->get_length());
    MI_CHECK_EQUAL( m, ROWS);
    MI_CHECK_EQUAL( n, COLUMNS);

    // check default values

    for( mi::Size i = 0; i < m; ++i)
        for( mi::Size j = 0; j < n; ++j) {
            MI_CHECK_EQUAL( static_cast<T>( 0), compound->template get_value<T>( i, j));
        }

    // check get_values() / set_values() (mi::ICompound)

    compound->set_values( source1);
    verify_values<T, bool>( compound, source1, m*n);
    verify_values<T, mi::Sint32>( compound, source1, m*n);
    verify_values<T, mi::Float32>( compound, source1, m*n);
    verify_values<T, mi::Float64>( compound, source1, m*n);

    // check get_value() / set_value() (mi::ICompound)

    for( mi::Size i = 0; i < m; ++i)
        for( mi::Size j = 0; j < n; ++j) {
            MI_CHECK( compound->set_value( i, j, source2[i*n+j]));
        }
    verify_value<T, bool>( compound, source2, m, n);
    verify_value<T, mi::Sint32>( compound, source2, m, n);
    verify_value<T, mi::Float32>( compound, source2, m, n);
    verify_value<T, mi::Float64>( compound, source2, m, n);

    // check get_value() / set_value() with wrong indices

    T buffer;
    MI_CHECK( !compound->set_value( 1, n, source1[0]));
    MI_CHECK( !compound->set_value( m, 1, source1[0]));
    MI_CHECK( !compound->set_value( m, n, source1[0]));
    MI_CHECK( !compound->get_value( 1, n, buffer));
    MI_CHECK( !compound->get_value( m, 1, buffer));
    MI_CHECK( !compound->get_value( m, n, buffer));

    // check get_value() / set_value() (IData_collection, index only)

    mi::IData_collection* data_collection = compound;

    for( mi::Size i = 0; i < m*n; ++i) {
        mi::base::Handle<mi::INumber> element(
            transaction->create<mi::INumber>( element_type_name));
        MI_CHECK( element);
        element->set_value( source1[i]);
        MI_CHECK_EQUAL( 0, data_collection->set_value( i, element.get()));
    }
    verify_values<T, bool>( compound, source1, m*n);
    verify_values<T, mi::Sint32>( compound, source1, m*n);
    verify_values<T, mi::Float32>( compound, source1, m*n);
    verify_values<T, mi::Float64>( compound, source1, m*n);

    for( mi::Size i = 0; i < m*n; ++i) {
        mi::base::Handle<mi::INumber> element( data_collection->get_value<mi::INumber>( i));
        MI_CHECK( element);
        T value = element->get_value<T>();
        MI_CHECK_EQUAL( value, source1[i]);
    }

    mi::base::Handle<mi::INumber> element(
        transaction->create<mi::INumber>( element_type_name));
    MI_CHECK( element);
    MI_CHECK_EQUAL( -1, data_collection->set_value( zero_size, nullptr));
    MI_CHECK_EQUAL( -2, data_collection->set_value( m*n, element.get()));
    mi::base::Handle<mi::IVoid> void_( transaction->create<mi::IVoid>( "Void"));
    MI_CHECK_EQUAL( -3, data_collection->set_value( zero_size, void_.get()));

    MI_CHECK( !data_collection->get_value( m*n));

    delete[] source1;
    delete[] source2;
}

// test vector interfaces
template <class I, class T, mi::Size ROWS>
void test_vector(
    mi::neuraylib::ITransaction* transaction, const char* type_name, I* icompound = nullptr)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    if( !icompound)
        icompound = transaction->create<I>( type_name);
    else
        icompound->retain();
    mi::base::Handle<I> compound( icompound);
    MI_CHECK( compound);

    // run generic mi::ICompound test

    test_compound<T,ROWS,1>(
        transaction, compound.get(), type_name, compound->get_element_type_name());

    // check vector interfaces

    mi::math::Vector_struct<T,ROWS> vs = compound->get_value();
    verify_buffer<T,T>( &vs.x, source2, ROWS);

    mi::math::Vector<T,ROWS> v;
    for( mi::Size i = 0; i < ROWS; ++i)
        v.set( i, source1[i]);
    compound->set_value( v);

    compound->get_value( vs);
    verify_buffer<T,T>( &vs.x, source1, ROWS);

    delete[] source1;
    delete[] source2;
}

// test matrix interfaces
template <class I, class T, mi::Size ROWS, mi::Size COLUMNS>
void test_matrix(
    mi::neuraylib::ITransaction* transaction, const char* type_name, I* icompound = nullptr)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    if( !icompound)
        icompound = transaction->create<I>( type_name);
    else
        icompound->retain();
    mi::base::Handle<I> compound( icompound);
    MI_CHECK( compound);

    // run generic mi::ICompound test

    test_compound<T,ROWS,COLUMNS>(
        transaction, compound.get(), type_name, compound->get_element_type_name());

    // check matrix interfaces

    mi::math::Matrix_struct<T,ROWS,COLUMNS> ms = compound->get_value();
    verify_buffer<T,T>( &ms.xx, source2, ROWS*COLUMNS);

    mi::math::Matrix<T,ROWS,COLUMNS> m;
    for( mi::Size i = 0; i < ROWS; ++i)
        for( mi::Size j = 0; j < COLUMNS; ++j)
            m.set( i, j, source1[i*COLUMNS+j]);
    compound->set_value( m);

    compound->get_value( ms);
    verify_buffer<T,T>( &ms.xx, source1, ROWS*COLUMNS);

    delete[] source1;
    delete[] source2;
}

// test mi::IColor interface
template <class I, class T>
void test_color(
    mi::neuraylib::ITransaction* transaction, const char* type_name, I* icompound = nullptr)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    if( !icompound)
        icompound = transaction->create<I>( type_name);
    else
        icompound->retain();
    mi::base::Handle<I> compound( icompound);
    MI_CHECK( compound);

    // run generic mi::ICompound test

    test_compound<T,4,1>(
        transaction, compound.get(), type_name, compound->get_element_type_name());

    // check mi::IColor interface

    mi::Color_struct cs = compound->get_value();
    verify_buffer<T,T>( &cs.r, source2, 4);

    mi::Color c;
    for( mi::Size i = 0; i < 4; ++i)
        c.set( i, source1[i]);
    compound->set_value( c);

    compound->get_value( cs);
    verify_buffer<T,T>( &cs.r, source1, 4);

    delete[] source1;
    delete[] source2;
}

// test mi::IColor3 interface
template <class I, class T>
void test_color3(
    mi::neuraylib::ITransaction* transaction, const char* type_name, I* icompound = nullptr)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    if( !icompound)
        icompound = transaction->create<I>( type_name);
    else
        icompound->retain();
    mi::base::Handle<I> compound( icompound);
    MI_CHECK( compound);

    // run generic mi::ICompound test

    test_compound<T,3,1>(
        transaction, compound.get(), type_name, compound->get_element_type_name());

    // check mi::IColor3 interface

    mi::Color_struct cs = compound->get_value();
    verify_buffer<T,T>( &cs.r, source2, 3);

    mi::Color c;
    for( mi::Size i = 0; i < 3; ++i)
        c.set( i, source1[i]);
    compound->set_value( c);

    compound->get_value( cs);
    verify_buffer<T,T>( &cs.r, source1, 3);

    delete[] source1;
    delete[] source2;
}

// test mi::ISpectrum interface
template <class I, class T>
void test_spectrum(
    mi::neuraylib::ITransaction* transaction, const char* type_name, I* icompound = nullptr)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    if( !icompound)
        icompound = transaction->create<I>( type_name);
    else
        icompound->retain();
    mi::base::Handle<I> compound( icompound);
    MI_CHECK( compound);

    // run generic mi::ICompound test

    test_compound<T,3,1>(
        transaction, compound.get(), type_name, compound->get_element_type_name());

    // check mi::ISpectrum interface

    mi::Spectrum_struct ss = compound->get_value();
    verify_buffer<T,T>( &ss.c[0], source2, 3);

    mi::Spectrum s;
    for( mi::Size i = 0; i < 3; ++i)
        s.set( i, source1[i]);
    compound->set_value( s);

    compound->get_value( ss);
    verify_buffer<T,T>( &ss.c[0], source1, 3);

    delete[] source1;
    delete[] source2;
}

// test bbox interface
template <class I, class T>
void test_bbox(
     mi::neuraylib::ITransaction* transaction, const char* type_name, I* icompound = nullptr)
{
    // prepare test data

    T* source1 = new T[N];
    T* source2 = new T[N];
    for( mi::Size i = 0; i < N; ++i) {
        source1[i] = static_cast<T>( data1[i]);
        source2[i] = static_cast<T>( data2[i]);
    }

    if( !icompound)
        icompound = transaction->create<I>( type_name);
    else
        icompound->retain();
    mi::base::Handle<I> compound( icompound);
    MI_CHECK( compound);

    // run generic mi::ICompound test

    test_compound<T,2,3>(
        transaction, compound.get(), type_name, compound->get_element_type_name());

    // check bbox interface

    mi::Bbox3_struct bs = compound->get_value();
    verify_buffer<T,T>( &bs.min.x, source2, 3);
    verify_buffer<T,T>( &bs.max.x, source2+3, 3);

    mi::Bbox3 b;
    b.min.x = source1[0];
    b.min.y = source1[1];
    b.min.z = source1[2];
    b.max.x = source1[3];
    b.max.y = source1[4];
    b.max.z = source1[5];
    compound->set_value( b);

    compound->get_value( bs);
    verify_buffer<T,T>( &bs.min.x, source1, 3);
    verify_buffer<T,T>( &bs.max.x, source1+3, 3);

    b.clear();
    compound->get_value( b);
    verify_buffer<T,T>( &b.min.x, source1, 3);
    verify_buffer<T,T>( &b.max.x, source1+3, 3);

        delete[] source1;
        delete[] source2;
}


// test vector interfaces
template <class I, class T, mi::Size ROWS>
void test_attribute_vector( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( attribute_container);

    mi::base::Handle<I> compound( attribute_container->create_attribute<I>( "the_attribute", type_name));
    MI_CHECK( compound);

    std::string real_type_name = type_name;
    if( real_type_name[0] == 'U')
        real_type_name[0] = 'S';

    test_vector<I,T,ROWS>( transaction, real_type_name.c_str(), compound.get());
}

// test matrix interfaces
template <class I, class T, mi::Size ROWS, mi::Size COLUMNS>
void test_attribute_matrix( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( attribute_container);

    mi::base::Handle<I> compound( attribute_container->create_attribute<I>( "the_attribute", type_name));
    MI_CHECK( compound);

    test_matrix<I,T,ROWS,COLUMNS>( transaction, type_name, compound.get());
}

// test mi::IColor interface
template <class I, class T>
void test_attribute_color( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( attribute_container);

    mi::base::Handle<I> compound( attribute_container->create_attribute<I>( "the_attribute", type_name));
    MI_CHECK( compound);

    test_color<I,T>( transaction, type_name, compound.get());
}

// test mi::IColor3 interface
template <class I, class T>
void test_attribute_color3( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( attribute_container);

    mi::base::Handle<I> compound( attribute_container->create_attribute<I>( "the_attribute", type_name));
    MI_CHECK( compound);

    test_color3<I,T>( transaction, type_name, compound.get());
}

// test mi::ISpectrum interface
template <class I, class T>
void test_attribute_spectrum( mi::neuraylib::ITransaction* transaction, const char* type_name)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> attribute_container(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( attribute_container);

    mi::base::Handle<I> compound( attribute_container->create_attribute<I>( "the_attribute", type_name));
    MI_CHECK( compound);

    test_spectrum<I,T>( transaction, type_name, compound.get());
}


void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope(
            database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        // some test below are disabled to reduce compile time

//        test_vector<mi::IBoolean_2,   bool, 2>( transaction.get(), "Boolean<2>");
//        test_vector<mi::IBoolean_3,   bool, 3>( transaction.get(), "Boolean<3>");
        test_vector<mi::IBoolean_4,   bool, 4>( transaction.get(), "Boolean<4>");

//        test_vector<mi::ISint32_2,    mi::Sint32, 2>( transaction.get(), "Sint32<2>");
//        test_vector<mi::ISint32_3,    mi::Sint32, 3>( transaction.get(), "Sint32<3>");
        test_vector<mi::ISint32_4,    mi::Sint32, 4>( transaction.get(), "Sint32<4>");

//        test_vector<mi::IUint32_2,    mi::Uint32, 2>( transaction.get(), "Uint32<2>");
//        test_vector<mi::IUint32_3,    mi::Uint32, 3>( transaction.get(), "Uint32<3>");
        test_vector<mi::IUint32_4,    mi::Uint32, 4>( transaction.get(), "Uint32<4>");

//        test_vector<mi::IFloat32_2,   mi::Float32, 2>( transaction.get(), "Float32<2>");
//        test_vector<mi::IFloat32_3,   mi::Float32, 3>( transaction.get(), "Float32<3>");
        test_vector<mi::IFloat32_4,   mi::Float32, 4>( transaction.get(), "Float32<4>");

//        test_vector<mi::IFloat64_2,   mi::Float64, 2>( transaction.get(), "Float64<2>");
//        test_vector<mi::IFloat64_3,   mi::Float64, 3>( transaction.get(), "Float64<3>");
        test_vector<mi::IFloat64_4,   mi::Float64, 4>( transaction.get(), "Float64<4>");


//        test_matrix<mi::IBoolean_2_2, bool, 2, 2>( transaction.get(), "Boolean<2,2>");
//        test_matrix<mi::IBoolean_2_3, bool, 2, 3>( transaction.get(), "Boolean<2,3>");
//        test_matrix<mi::IBoolean_2_4, bool, 2, 4>( transaction.get(), "Boolean<2,4>");
//        test_matrix<mi::IBoolean_3_2, bool, 3, 2>( transaction.get(), "Boolean<3,2>");
//        test_matrix<mi::IBoolean_3_3, bool, 3, 3>( transaction.get(), "Boolean<3,3>");
//        test_matrix<mi::IBoolean_3_4, bool, 3, 4>( transaction.get(), "Boolean<3,4>");
//        test_matrix<mi::IBoolean_4_2, bool, 4, 2>( transaction.get(), "Boolean<4,2>");
//        test_matrix<mi::IBoolean_4_3, bool, 4, 3>( transaction.get(), "Boolean<4,3>");
        test_matrix<mi::IBoolean_4_4, bool, 4, 4>( transaction.get(), "Boolean<4,4>");

//        test_matrix<mi::ISint32_2_2, mi::Sint32, 2, 2>( transaction.get(), "Sint32<2,2>");
//        test_matrix<mi::ISint32_2_3, mi::Sint32, 2, 3>( transaction.get(), "Sint32<2,3>");
//        test_matrix<mi::ISint32_2_4, mi::Sint32, 2, 4>( transaction.get(), "Sint32<2,4>");
//        test_matrix<mi::ISint32_3_2, mi::Sint32, 3, 2>( transaction.get(), "Sint32<3,2>");
//        test_matrix<mi::ISint32_3_3, mi::Sint32, 3, 3>( transaction.get(), "Sint32<3,3>");
//        test_matrix<mi::ISint32_3_4, mi::Sint32, 3, 4>( transaction.get(), "Sint32<3,4>");
//        test_matrix<mi::ISint32_4_2, mi::Sint32, 4, 2>( transaction.get(), "Sint32<4,2>");
//        test_matrix<mi::ISint32_4_3, mi::Sint32, 4, 3>( transaction.get(), "Sint32<4,3>");
        test_matrix<mi::ISint32_4_4, mi::Sint32, 4, 4>( transaction.get(), "Sint32<4,4>");

//        test_matrix<mi::IUint32_2_2, mi::Uint32, 2, 2>( transaction.get(), "Uint32<2,2>");
//        test_matrix<mi::IUint32_2_3, mi::Uint32, 2, 3>( transaction.get(), "Uint32<2,3>");
//        test_matrix<mi::IUint32_2_4, mi::Uint32, 2, 4>( transaction.get(), "Uint32<2,4>");
//        test_matrix<mi::IUint32_3_2, mi::Uint32, 3, 2>( transaction.get(), "Uint32<3,2>");
//        test_matrix<mi::IUint32_3_3, mi::Uint32, 3, 3>( transaction.get(), "Uint32<3,3>");
//        test_matrix<mi::IUint32_3_4, mi::Uint32, 3, 4>( transaction.get(), "Uint32<3,4>");
//        test_matrix<mi::IUint32_4_2, mi::Uint32, 4, 2>( transaction.get(), "Uint32<4,2>");
//        test_matrix<mi::IUint32_4_3, mi::Uint32, 4, 3>( transaction.get(), "Uint32<4,3>");
        test_matrix<mi::IUint32_4_4, mi::Uint32, 4, 4>( transaction.get(), "Uint32<4,4>");

//        test_matrix<mi::IFloat32_2_2, mi::Float32, 2, 2>( transaction.get(), "Float32<2,2>");
//        test_matrix<mi::IFloat32_2_3, mi::Float32, 2, 3>( transaction.get(), "Float32<2,3>");
//        test_matrix<mi::IFloat32_2_4, mi::Float32, 2, 4>( transaction.get(), "Float32<2,4>");
//        test_matrix<mi::IFloat32_3_2, mi::Float32, 3, 2>( transaction.get(), "Float32<3,2>");
//        test_matrix<mi::IFloat32_3_3, mi::Float32, 3, 3>( transaction.get(), "Float32<3,3>");
//        test_matrix<mi::IFloat32_3_4, mi::Float32, 3, 4>( transaction.get(), "Float32<3,4>");
//        test_matrix<mi::IFloat32_4_2, mi::Float32, 4, 2>( transaction.get(), "Float32<4,2>");
//        test_matrix<mi::IFloat32_4_3, mi::Float32, 4, 3>( transaction.get(), "Float32<4,3>");
        test_matrix<mi::IFloat32_4_4, mi::Float32, 4, 4>( transaction.get(), "Float32<4,4>");

//        test_matrix<mi::IFloat64_2_2, mi::Float64, 2, 2>( transaction.get(), "Float64<2,2>");
//        test_matrix<mi::IFloat64_2_3, mi::Float64, 2, 3>( transaction.get(), "Float64<2,3>");
//        test_matrix<mi::IFloat64_2_4, mi::Float64, 2, 4>( transaction.get(), "Float64<2,4>");
//        test_matrix<mi::IFloat64_3_2, mi::Float64, 3, 2>( transaction.get(), "Float64<3,2>");
//        test_matrix<mi::IFloat64_3_3, mi::Float64, 3, 3>( transaction.get(), "Float64<3,3>");
//        test_matrix<mi::IFloat64_3_4, mi::Float64, 3, 4>( transaction.get(), "Float64<3,4>");
//        test_matrix<mi::IFloat64_4_2, mi::Float64, 4, 2>( transaction.get(), "Float64<4,2>");
//        test_matrix<mi::IFloat64_4_3, mi::Float64, 4, 3>( transaction.get(), "Float64<4,3>");
        test_matrix<mi::IFloat64_4_4, mi::Float64, 4, 4>( transaction.get(), "Float64<4,4>");

        test_color<mi::IColor, mi::Float32>( transaction.get(), "Color");
        test_color3<mi::IColor3, mi::Float32>( transaction.get(), "Color3");
        test_spectrum<mi::ISpectrum, mi::Float32>( transaction.get(), "Spectrum");
        test_bbox<mi::IBbox3, mi::Float32>( transaction.get(), "Bbox3");

//        test_attribute_vector<mi::IBoolean_2, bool, 2>( transaction.get(), "Boolean<2>");
//        test_attribute_vector<mi::IBoolean_3, bool, 3>( transaction.get(), "Boolean<3>");
        test_attribute_vector<mi::IBoolean_4, bool, 4>( transaction.get(), "Boolean<4>");

//        test_attribute_vector<mi::ISint32_2, mi::Sint32, 2>( transaction.get(), "Sint32<2>");
//        test_attribute_vector<mi::ISint32_3, mi::Sint32, 3>( transaction.get(), "Sint32<3>");
        test_attribute_vector<mi::ISint32_4, mi::Sint32, 4>( transaction.get(), "Sint32<4>");

//        test_attribute_vector<mi::IFloat32_2,   mi::Float32, 2>( transaction.get(), "Float32<2>");
//        test_attribute_vector<mi::IFloat32_3,   mi::Float32, 3>( transaction.get(), "Float32<3>");
        test_attribute_vector<mi::IFloat32_4,   mi::Float32, 4>( transaction.get(), "Float32<4>");

//        test_attribute_vector<mi::IFloat64_2,   mi::Float64, 2>( transaction.get(), "Float64<2>");
//        test_attribute_vector<mi::IFloat64_3,   mi::Float64, 3>( transaction.get(), "Float64<3>");
        test_attribute_vector<mi::IFloat64_4,   mi::Float64, 4>( transaction.get(), "Float64<4>");

        test_attribute_matrix<mi::IFloat32_2_2, mi::Float32, 2, 2>( transaction.get(), "Float32<2,2>");
//        test_attribute_matrix<mi::IFloat32_2_3, mi::Float32, 2, 3>( transaction.get(), "Float32<2,3>");
//        test_attribute_matrix<mi::IFloat32_2_4, mi::Float32, 2, 4>( transaction.get(), "Float32<2,4>");
//        test_attribute_matrix<mi::IFloat32_3_2, mi::Float32, 3, 2>( transaction.get(), "Float32<3,2>");
//        test_attribute_matrix<mi::IFloat32_3_3, mi::Float32, 3, 3>( transaction.get(), "Float32<3,3>");
//        test_attribute_matrix<mi::IFloat32_3_4, mi::Float32, 3, 4>( transaction.get(), "Float32<3,4>");
//        test_attribute_matrix<mi::IFloat32_4_2, mi::Float32, 4, 2>( transaction.get(), "Float32<4,2>");
//        test_attribute_matrix<mi::IFloat32_4_3, mi::Float32, 4, 3>( transaction.get(), "Float32<4,3>");
        test_attribute_matrix<mi::IFloat32_4_4, mi::Float32, 4, 4>( transaction.get(), "Float32<4,4>");

        test_attribute_matrix<mi::IFloat64_2_2, mi::Float64, 2, 2>( transaction.get(), "Float64<2,2>");
//        test_attribute_matrix<mi::IFloat64_2_3, mi::Float64, 2, 3>( transaction.get(), "Float64<2,3>");
//        test_attribute_matrix<mi::IFloat64_2_4, mi::Float64, 2, 4>( transaction.get(), "Float64<2,4>");
//        test_attribute_matrix<mi::IFloat64_3_2, mi::Float64, 3, 2>( transaction.get(), "Float64<3,2>");
//        test_attribute_matrix<mi::IFloat64_3_3, mi::Float64, 3, 3>( transaction.get(), "Float64<3,3>");
//        test_attribute_matrix<mi::IFloat64_3_4, mi::Float64, 3, 4>( transaction.get(), "Float64<3,4>");
//        test_attribute_matrix<mi::IFloat64_4_2, mi::Float64, 4, 2>( transaction.get(), "Float64<4,2>");
//        test_attribute_matrix<mi::IFloat64_4_3, mi::Float64, 4, 3>( transaction.get(), "Float64<4,3>");
        test_attribute_matrix<mi::IFloat64_4_4, mi::Float64, 4, 4>( transaction.get(), "Float64<4,4>");

        test_attribute_color<mi::IColor, mi::Float32>( transaction.get(), "Color");
        test_attribute_color3<mi::IColor3, mi::Float32>( transaction.get(), "Color3");
        test_attribute_spectrum<mi::ISpectrum, mi::Float32>( transaction.get(), "Spectrum");

        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_types_compound )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));

        run_tests( neuray.get());
        run_tests( neuray.get());
    }


    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

