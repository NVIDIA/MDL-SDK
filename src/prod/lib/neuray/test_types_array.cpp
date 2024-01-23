/******************************************************************************
 * Copyright (c) 2009-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Test for mi::IArray, mi::IDynamic_array
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN


#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>
#include <vector>
#include <sstream>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"

#define GET_REFCOUNT(X) ((X) ? (X)->retain(), (X)->release() : 999)


// set element property
void set_element_property( mi::IString* element, mi::Size i)
{
    std::ostringstream str;
    str << i;
    element->set_c_str( str.str().c_str());
}

// set element property
void set_element_property( mi::ISint32* element, mi::Size i)
{
    element->set_value( static_cast<mi::Sint32>( i));
}

// set element property
void set_element_property( mi::IRef* element, mi::Size i)
{
    // skip
}


// check element property
bool check_element_property( const mi::IString* element, mi::Size i)
{
    std::ostringstream str;
    str << i;
    return strcmp( element->get_c_str(), str.str().c_str()) == 0;
}

// check element property
bool check_element_property( const mi::ISint32* element, mi::Size i)
{
    mi::Sint32 tmp;
    element->get_value( tmp);
    return tmp == static_cast<mi::Sint32>( i);
}

// check element property
bool check_element_property( const mi::IRef* element, mi::Size i)
{
    return true;
}

// Does not work if T is derived from mi::IVoid or mi::neuraylib::IGroup because these are hard-coded
// interfaces to test some expected failures.
template<class T>
void test_interface_IData_collection(
    mi::neuraylib::ITransaction* transaction,
    mi::IData_collection* data_collection,
    mi::Size N,
    const char* type_name,
    const char* element_type_name,
    bool untyped,
    bool attribute)
{
    const mi::IData_collection* const_data_collection = data_collection;

    std::ostringstream str, str1;
    str << N;
    std::string N_string = str.str();
    str1 << N-1;
    std::string N_minus_1_string = str1.str();

    // check type name
    MI_CHECK_EQUAL_CSTR( data_collection->get_type_name(), type_name);

    // check collection length
    MI_CHECK_EQUAL( N, data_collection->get_length());

    // check get_key() / has_key()
    const char* s;
    s = data_collection->get_key( -1);
    MI_CHECK( !s);
    MI_CHECK( !data_collection->has_key( "-1"));
    s = data_collection->get_key( N);
    MI_CHECK( !s);
    MI_CHECK( !data_collection->has_key( N_string.c_str()));
    MI_CHECK( !data_collection->has_key( 0));
    MI_CHECK( !data_collection->has_key( "no_number"));

    if( N == 0) {
        s = data_collection->get_key( 0);
        MI_CHECK( !s);
    } else {
        s = data_collection->get_key( 0);
        MI_CHECK( s);
        MI_CHECK( data_collection->has_key( s));
        s = data_collection->get_key( N-1);
        MI_CHECK( s);
        MI_CHECK( data_collection->has_key( s));
    }

    if( N == 0)
        return;

    mi::base::Handle<const T> c_iinterface;
    mi::base::Handle<T> m_iinterface;

    // check get_value() via index
    m_iinterface = data_collection->get_value<T>( zero_size);
    MI_CHECK( m_iinterface.is_valid_interface() ^ untyped);
    m_iinterface = data_collection->get_value<T>( N-1);
    MI_CHECK( m_iinterface.is_valid_interface() ^ untyped);
    MI_CHECK( !data_collection->get_value<T>( N));

    c_iinterface = const_data_collection->get_value<T>( zero_size);
    MI_CHECK( c_iinterface.is_valid_interface() ^ untyped);
    c_iinterface = const_data_collection->get_value<T>( N-1);
    MI_CHECK( c_iinterface.is_valid_interface() ^ untyped);
    MI_CHECK( !const_data_collection->get_value<T>( N));

    // check get_value() via key
    m_iinterface = data_collection->get_value<T>( "0");
    MI_CHECK( m_iinterface.is_valid_interface() ^ untyped);
    m_iinterface = data_collection->get_value<T>( N_minus_1_string.c_str());
    MI_CHECK( m_iinterface.is_valid_interface() ^ untyped);
    MI_CHECK( !data_collection->get_value<T>( "-1"));
    MI_CHECK( !data_collection->get_value<T>( N_string.c_str()));
    MI_CHECK( !data_collection->get_value<T>( zero_string));
    MI_CHECK( !data_collection->get_value<T>( "no_number"));

    c_iinterface = const_data_collection->get_value<T>( "0");
    MI_CHECK( c_iinterface.is_valid_interface() ^ untyped);
    c_iinterface = const_data_collection->get_value<T>( N_minus_1_string.c_str());
    MI_CHECK( c_iinterface.is_valid_interface() ^ untyped);
    MI_CHECK( !const_data_collection->get_value<T>( "-1"));
    MI_CHECK( !const_data_collection->get_value<T>( N_string.c_str()));
    MI_CHECK( !const_data_collection->get_value<T>( zero_string));
    MI_CHECK( !const_data_collection->get_value<T>( "no_number"));

    mi::base::Handle<mi::IVoid> void_( transaction->create<mi::IVoid>( "Void"));
    MI_CHECK( void_.is_valid_interface());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));

    // check set_value() via index
    mi::base::Handle<mi::base::IInterface> tmp;
    tmp = data_collection->get_value<mi::base::IInterface>( N-1);
    MI_CHECK_EQUAL(  0, data_collection->set_value( N-1, tmp.get()));
    MI_CHECK_EQUAL( -1, data_collection->set_value( N-1, 0));
    MI_CHECK_EQUAL( -2, data_collection->set_value( N, void_.get()));
    if( untyped) {
        MI_CHECK_EQUAL( 0, data_collection->set_value( N-1, void_.get()));
        // restore previous state
        MI_CHECK_EQUAL( 0, data_collection->set_value( N-1, tmp.get()));
    } else {
        MI_CHECK_EQUAL( -3, data_collection->set_value( zero_size, void_.get()));
    }

    // check set_value() via key
    tmp = data_collection->get_value<mi::base::IInterface>( N-1);
    MI_CHECK_EQUAL(  0, data_collection->set_value( N_minus_1_string.c_str(), tmp.get()));
    MI_CHECK_EQUAL( -1, data_collection->set_value( N_minus_1_string.c_str(), 0));
    MI_CHECK_EQUAL( -2, data_collection->set_value( N_string.c_str(), void_.get()));
    if( untyped) {
        MI_CHECK_EQUAL( 0, data_collection->set_value( N_minus_1_string.c_str(), void_.get()));
        // restore previous state
        MI_CHECK_EQUAL( 0, data_collection->set_value( N_minus_1_string.c_str(), tmp.get()));
    } else {
        MI_CHECK_EQUAL( -3, data_collection->set_value( "0", void_.get()));
    }
}

// Does not work if T is derived from mi::IVoid or mi::neuraylib::IGroup because these are hard-coded
// interfaces to test some expected failures.
template<class T>
void test_interface_IArray(
    mi::neuraylib::ITransaction* transaction,
    mi::IArray* array,
    mi::Size N,
    const char* type_name,
    const char* element_type_name,
    bool untyped,
    bool attribute)
{
    const mi::IArray* const_array = array;

    // check type name
    MI_CHECK_EQUAL_CSTR( array->get_type_name(), type_name);

    // check array length
    MI_CHECK_EQUAL( N, array->get_length());
    MI_CHECK( N == 0 ? array->empty() : !array->empty());

    // check that all elements get initialized for typed arrays
    for( mi::Size i=0; i < N; ++i) {
        mi::base::Handle<mi::base::IInterface> element( array->get_element( i));
        MI_CHECK( element.is_valid_interface());
    }

    // check that out-of-bound read access returns 0
    MI_CHECK_EQUAL( 0, array->get_element( N));
    MI_CHECK_EQUAL( 0, array->get_element( 1u << 31));

    // prepare array contents
    std::vector<mi::base::Handle<T> > vector;
    for( mi::Size i=0; i < N+1; ++i) { // +1 such that [0] is always valid
        mi::base::Handle<T> element( transaction->create<T>( element_type_name));
        set_element_property( element.get(), i);
        vector.push_back( element);
        element = 0;
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( vector[i].get()));
    }

    // set array contents
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 0, array->set_element( i, vector[i].get()));
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( vector[i].get()));
    }

    // check that out-of-bound write access fails
    MI_CHECK_EQUAL( -1, array->set_element( N, vector[0].get()));
    MI_CHECK_EQUAL( -1, array->set_element( 1u << 31, vector[0].get()));

    // verify array contents
    for( mi::Size i=0; i < N; ++i) {
        mi::base::IInterface* iinterface = array->get_element( i);
        T* element = array->get_element<T>( i);
        mi::IVoid* void_ = array->get_element<mi::IVoid>( i);
        MI_CHECK( attribute || (iinterface == vector[i].get()));
        MI_CHECK( attribute || (element == vector[i].get()));
        MI_CHECK_EQUAL( 0, void_);
        MI_CHECK( check_element_property( element, i));
        element->release();
        iinterface->release();
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( vector[i].get()));
    }

    // verify const array contents
    for( mi::Size i=0; i < N; ++i) {
        const mi::base::IInterface* iinterface = const_array->get_element( i);
        const T* element = const_array->get_element<T>( i);
        const mi::IVoid* void_ = const_array->get_element<mi::IVoid>( i);
        MI_CHECK( attribute || (iinterface == vector[i].get()));
        MI_CHECK( attribute || (element == vector[i].get()));
        MI_CHECK_EQUAL( 0, void_);
        MI_CHECK( check_element_property( element, i));
        element->release();
        iinterface->release();
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( vector[i].get()));
    }

    if( N == 0)
        return;

    mi::base::Handle<mi::IVoid> void_( transaction->create<mi::IVoid>( "Void"));
    MI_CHECK( void_.is_valid_interface());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));
    if( untyped) {
        // check that untyped arrays accept any types
        MI_CHECK_EQUAL( 0, array->set_element( 0, void_.get()));
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( void_.get()));

    } else {
        // check that typed arrays reject other types
        MI_CHECK_EQUAL( -2, array->set_element( 0, void_.get()));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));

    }
}

// Does not work if T is derived from mi::IVoid or mi::neuraylib::IGroup because these are hard-coded
// interfaces to test some expected failures.
template<class T>
void test_interface_IDynamic_array(
    mi::neuraylib::ITransaction* transaction,
    mi::IDynamic_array* array,
    mi::Size N,
    const char* type_name,
    const char* element_type_name,
    bool untyped,
    bool attribute)
{
    const mi::IDynamic_array* const_array = array;

    // set length to N/2
    array->set_length( N/2);
    MI_CHECK_EQUAL( N/2, array->get_length());
    MI_CHECK( N == 0 ? array->empty() : !array->empty());

    // set length to N
    array->set_length( N);
    MI_CHECK_EQUAL( N, array->get_length());
    MI_CHECK( N == 0 ? array->empty() : !array->empty());

    // check that additional elements get initialized for typed arrays
    for( mi::Size i = N/2 + 1; i < N; ++i) {
        mi::base::Handle<mi::base::IInterface> element( array->get_element( i));
        MI_CHECK( element.is_valid_interface());
    }

    // set length to 0
    array->set_length( 0);
    MI_CHECK_EQUAL( 0, array->get_length());
    MI_CHECK( array->empty());

    mi::base::Handle<T> element( transaction->create<T>( element_type_name));
    set_element_property( element.get(), 42);
    MI_CHECK( element.is_valid_interface());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( element.get()));

    // push back element
    MI_CHECK_EQUAL( 0, array->push_back( element.get()));
    MI_CHECK_EQUAL( 1, array->get_length());
    MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( element.get()));

    // check array length
    MI_CHECK_EQUAL( 1, array->get_length());
    MI_CHECK( !array->empty());

    {
        // and retrieve it via back() and get_element( get_length()-1)
        mi::base::IInterface* interface1 = array->back();
        mi::base::IInterface* interface2 = array->get_element( array->get_length()-1);
        T* ielement = array->back<T>();
        mi::IVoid* void_ = array->back<mi::IVoid>();
        MI_CHECK( attribute || (interface1 == element.get()));
        MI_CHECK( attribute || (interface2 == element.get()));
        MI_CHECK( attribute || (ielement == element.get()));
        MI_CHECK_EQUAL( 0, void_);
        MI_CHECK( check_element_property( ielement, 42));
        ielement->release();
        interface2->release();
        interface1->release();
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( element.get()));
    }
    {
        // and retrieve it via back() and get_element( get_length()-1) (const version)
        const mi::base::IInterface* interface1 = const_array->back();
        const mi::base::IInterface* interface2 = const_array->get_element( array->get_length()-1);
        const T* ielement = const_array->back<T>();
        const mi::IVoid* void_ = const_array->back<mi::IVoid>();
        MI_CHECK( attribute || (interface1 == element.get()));
        MI_CHECK( attribute || (interface2 == element.get()));
        MI_CHECK( attribute || (ielement == element.get()));
        MI_CHECK_EQUAL( 0, void_);
        MI_CHECK( check_element_property( ielement, 42));
        ielement->release();
        interface2->release();
        interface1->release();
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( element.get()));
    }

    // pop last element
    MI_CHECK_EQUAL( 0, array->pop_back());
    MI_CHECK_EQUAL( 0, array->get_length());
    MI_CHECK( array->empty());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( element.get()));
    MI_CHECK_EQUAL( -3, array->pop_back());

    // insert some element
    MI_CHECK_EQUAL( 0, array->insert( 0, element.get()));
    MI_CHECK_EQUAL( 1, array->get_length());
    MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( element.get()));

    // check array length
    MI_CHECK_EQUAL( 1, array->get_length());
    MI_CHECK( !array->empty());

    {
        // and retrieve it via front() and get_element( 0)
        mi::base::IInterface* interface1 = array->front();
        mi::base::IInterface* interface2 = array->get_element( 0);
        T* ielement = array->front<T>();
        mi::IVoid* void_ = array->front<mi::IVoid>();
        MI_CHECK( attribute || (interface1 == element.get()));
        MI_CHECK( attribute || (interface2 == element.get()));
        MI_CHECK( attribute || (ielement == element.get()));
        MI_CHECK_EQUAL( 0, void_);
        MI_CHECK( check_element_property( ielement, 42));
        ielement->release();
        interface2->release();
        interface1->release();
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( element.get()));
    }
    {
        // and retrieve it via front() and get_element( 0) (const version)
        const mi::base::IInterface* interface1 = const_array->front();
        const mi::base::IInterface* interface2 = const_array->get_element( 0);
        const T* ielement = const_array->front<T>();
        const mi::IVoid* void_ = const_array->front<mi::IVoid>();
        MI_CHECK( attribute || (interface1 == element.get()));
        MI_CHECK( attribute || (interface2 == element.get()));
        MI_CHECK( attribute || (ielement == element.get()));
        MI_CHECK_EQUAL( 0, void_);
        MI_CHECK( check_element_property( ielement, 42));
        ielement->release();
        interface2->release();
        interface1->release();
        MI_CHECK_EQUAL( attribute ? 1 : 2, GET_REFCOUNT( element.get()));
    }

    // remove element
    MI_CHECK_EQUAL( 0, array->erase( 0));
    MI_CHECK_EQUAL( 0, array->get_length());
    MI_CHECK( array->empty());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( element.get()));

    // fill it with N elements
    for( mi::Size i = 0; i < N; ++i) {
        MI_CHECK_EQUAL( 0, array->push_back( element.get()));
    }
    MI_CHECK_EQUAL( N, array->get_length());
    MI_CHECK_EQUAL( attribute ? 1 : N+1, GET_REFCOUNT( element.get()));

    // insert elements at the begin, in the middle, and at the end
    MI_CHECK_EQUAL( 0, array->insert( 0, element.get()));
    MI_CHECK_EQUAL( 0, array->insert( (N+1)/2, element.get()));
    MI_CHECK_EQUAL( 0, array->insert( N+2, element.get()));
    MI_CHECK_EQUAL( N+3, array->get_length());
    MI_CHECK_EQUAL( attribute ? 1 : N+4, GET_REFCOUNT( element.get()));
    // insert elements at invalid indices
    MI_CHECK_EQUAL( -1, array->insert( N+4, element.get()));
    MI_CHECK_EQUAL( -1, array->insert( 1u << 31, element.get()));

    // remove elements at the begin, in the middle, and at the end
    MI_CHECK_EQUAL( 0, array->erase( N+2));
    MI_CHECK_EQUAL( 0, array->erase( (N+1)/2));
    MI_CHECK_EQUAL( 0, array->erase( 0));
    MI_CHECK_EQUAL( N, array->get_length());
    MI_CHECK_EQUAL( attribute ? 1 : N+1, GET_REFCOUNT( element.get()));
    // remove elements at invalid indices
    MI_CHECK_EQUAL( -1, array->erase( N));
    MI_CHECK_EQUAL( -1, array->erase( 1u << 31));

    // clear it again
    array->clear();
    MI_CHECK_EQUAL( 0, array->get_length());
    MI_CHECK( array->empty());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( element.get()));

    mi::base::Handle<mi::IVoid> void_( transaction->create<mi::IVoid>( "Void"));
    MI_CHECK( void_.is_valid_interface());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));
    if( untyped) {
        // check that untyped arrays accept any types
        MI_CHECK_EQUAL( 0, array->push_back( void_.get()));
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( void_.get()));
        MI_CHECK_EQUAL( 1, array->get_length());
        MI_CHECK_EQUAL( 0, array->insert( 0, void_.get()));
        MI_CHECK_EQUAL( 3, GET_REFCOUNT( void_.get()));
        MI_CHECK_EQUAL( 2, array->get_length());

    } else {
        // check that typed arrays reject other types
        MI_CHECK_EQUAL( -2, array->push_back( void_.get()));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));
        MI_CHECK_EQUAL( 0, array->get_length());
        MI_CHECK_EQUAL( -2, array->insert( 0, void_.get()));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));
        MI_CHECK_EQUAL( 0, array->get_length());

    }
}

// Does not work if T is derived from mi::IVoid or mi::neuraylib::IGroup because these are hard-coded
// interfaces to test some expected failures.
template<class T>
void test(
    mi::neuraylib::ITransaction* transaction,
    mi::Size N,
    const char* element_type_name)
{
    mi::base::Handle<mi::IData> data( transaction->create<mi::IData>( element_type_name));
    bool untyped = !data.is_valid_interface();
    std::string type_name_prefix = untyped ? "Interface" : element_type_name;

    // test static array of size N
    std::ostringstream s1;
    s1 << type_name_prefix << "[" << N << "]";
    std::string type_name1 = s1.str();
    mi::base::Handle<mi::IArray> array( transaction->create<mi::IArray>( type_name1.c_str()));
    MI_CHECK( array.is_valid_interface());
    test_interface_IData_collection<T>( transaction, array.get(), N, type_name1.c_str(), element_type_name, untyped, false);
    test_interface_IArray<T>( transaction, array.get(), N, type_name1.c_str(), element_type_name, untyped, false);

    // test dynamic array
    std::string type_name2 = type_name_prefix;
    type_name2 += "[]";
    mi::base::Handle<mi::IDynamic_array> dynamic_array( transaction->create<mi::IDynamic_array>( type_name2.c_str()));
    MI_CHECK( dynamic_array.is_valid_interface());
    dynamic_array->set_length( N);
    test_interface_IData_collection<T>( transaction, dynamic_array.get(), N, type_name2.c_str(), element_type_name, untyped, false);
    test_interface_IArray<T>( transaction, dynamic_array.get(), N, type_name2.c_str(), element_type_name, untyped, false);
    test_interface_IDynamic_array<T>( transaction, dynamic_array.get(), N, type_name2.c_str(), element_type_name, untyped, false);

    // static arrays of negative size are not allowed
    std::string type_name4 = type_name_prefix;
    type_name4 += "[-42]";
    mi::base::Handle<mi::IArray> array4( transaction->create<mi::IArray>( type_name4.c_str()));
    MI_CHECK( !array4.is_valid_interface());
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

        test<mi::ISint32>( transaction.get(), 10, "Sint32");
        test<mi::IString>( transaction.get(), 11, "String");
        test<mi::IRef>(    transaction.get(), 12, "Ref");
        test<mi::ISint32>( transaction.get(),  0, "Sint32");
        test<mi::IString>( transaction.get(),  0, "String");
        test<mi::IRef>(    transaction.get(),  0, "Ref");

        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_types_array )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray.is_valid_interface());

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));

        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray = 0;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

