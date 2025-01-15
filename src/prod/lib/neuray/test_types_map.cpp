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
 ** \brief Test for mi::IMap
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <map>
#include <sstream>

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"

#define GET_REFCOUNT(X) ((X) ? (X)->retain(), (X)->release() : 999)

bool skip_identity_checks = false;

const mi::Size N = 16;
std::string g_key;

// A hard-coded list of some keys. Note that this is not necessarily the same mapping as the one
// used by mi::IMap. For simplicity, use const char* and cache last result in g_key.
const char* get_key( mi::Size key_id)
{
    if( key_id >= N)
        return nullptr;
    std::ostringstream s;
    s << "key_" << key_id;
    g_key = s.str();
    return g_key.c_str();
}


// set value property
void set_value_property( mi::IString* value, mi::Size i)
{
    std::ostringstream str;
    str << i;
    value->set_c_str( str.str().c_str());
}

// set value property
void set_value_property( mi::ISint32* value, mi::Size i)
{
    value->set_value( static_cast<mi::Sint32>( i));
}


// check value property
bool check_value_property( const mi::IString* value, mi::Size i)
{
    std::ostringstream str;
    str << i;
    return strcmp( value->get_c_str(), str.str().c_str()) == 0;
}

// check value property
bool check_value_property( const mi::ISint32* value, mi::Size i)
{
    mi::Sint32 tmp;
    value->get_value( tmp);
    return tmp == static_cast<mi::Sint32>( i);
}

// Does not work if T is derived from mi::IVoid or mi::neuraylib::IGroup because these are
// hard-coded interfaces to test some expected failures.
//
// This test modifies that the map such that it contains exactly the keys ...
template<class T>
void test_interface_IMap(
    mi::neuraylib::ITransaction* transaction,
    mi::IMap* map,
    const char* type_name,
    const char* value_type_name,
    bool untyped)
{
    const mi::IMap* const_map = map;

    // check type name
    MI_CHECK_EQUAL_CSTR( map->get_type_name(), type_name);

    // check map length
    MI_CHECK_EQUAL( 0, map->get_length());
    MI_CHECK( map->empty());

    // prepare map contents
    std::map<std::string, mi::base::Handle<T> > stl_map;
    for( mi::Size i=0; i < N; ++i) {
        mi::base::Handle<T> value( transaction->create<T>( value_type_name));
        set_value_property( value.get(), i);
        stl_map[ get_key( i)] = value;
        value = nullptr;
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( stl_map[ get_key( i)].get()));
    }
    mi::base::Handle<mi::IVoid> void_( transaction->create<mi::IVoid>( "Void"));
    MI_CHECK( void_.is_valid_interface());
    MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));

    // insert map contents (via key)
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 0, map->insert( get_key( i), stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( i+1, map->get_length());
        MI_CHECK_EQUAL( -2, map->insert( get_key( i), stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( i+1, map->get_length());
        MI_CHECK( !map->empty());
    }

    // check get_key() / has_key()
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK( map->has_key( get_key( i)));
        std::string key = map->get_key( i);
        key.resize( 4);
        MI_CHECK_EQUAL_CSTR( key.c_str(), "key_");
    }
    MI_CHECK( !map->has_key( nullptr));
    MI_CHECK( !map->has_key( ""));
    MI_CHECK( !map->has_key( "foo"));
    MI_CHECK( !map->get_key( N));

    // verify map contents via key, increasing index sequence (cached)
    for( mi::Size i=0; i < N; ++i) {
        mi::base::IInterface* iinterface = map->get_value( get_key( i));
        T* value = map->get_value<T>( get_key( i));
        auto* void_ = map->get_value<mi::IVoid>( get_key( i));
        MI_CHECK( skip_identity_checks || (iinterface == stl_map[ get_key( i)].get()));
        MI_CHECK( skip_identity_checks || (value == stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( nullptr, void_);
        MI_CHECK( check_value_property( value, i));
        value->release();
        iinterface->release();
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i)].get()));
    }

    // verify const map contents via key, decreasing index sequence (uncached)
    for( mi::Size i=N; i > 0; --i) {
        const mi::base::IInterface* iinterface = const_map->get_value( get_key( i-1));
        const T* value = const_map->get_value<T>( get_key( i-1));
        const auto* void_ = const_map->get_value<mi::IVoid>( get_key( i-1));
        MI_CHECK( skip_identity_checks || (iinterface == stl_map[ get_key( i-1)].get()));
        MI_CHECK( skip_identity_checks || (value == stl_map[ get_key( i-1)].get()));
        MI_CHECK_EQUAL( nullptr, void_);
        MI_CHECK( check_value_property( value, i-1));
        value->release();
        iinterface->release();
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i-1)].get()));
    }

    // set map contents via index
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 0, map->set_value( i, stl_map[ get_key( i)].get()));
    }
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i)].get()));
    }

    // verify map contents via key, increasing index sequence (cached)
    for( mi::Size i=0; i < N; ++i) {
        mi::base::IInterface* iinterface = map->get_value( i);
        T* value = map->get_value<T>( i);
        auto* void_ = map->get_value<mi::IVoid>( i);
        MI_CHECK( skip_identity_checks || (iinterface == stl_map[ get_key( i)].get()));
        MI_CHECK( skip_identity_checks || (value == stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( nullptr, void_);
        MI_CHECK( check_value_property( value, i));
        value->release();
        iinterface->release();
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i)].get()));
    }

    // verify const map contents via key, decreasing index sequence (uncached)
    for( mi::Size i=N; i > 0; --i) {
        const mi::base::IInterface* iinterface = const_map->get_value( i-1);
        const T* value = const_map->get_value<T>( i-1);
        const auto* void_ = const_map->get_value<mi::IVoid>( i-1);
        MI_CHECK( skip_identity_checks || (iinterface == stl_map[ get_key( i-1)].get()));
        MI_CHECK( skip_identity_checks || (value == stl_map[ get_key( i-1)].get()));
        MI_CHECK_EQUAL( nullptr, void_);
        MI_CHECK( check_value_property( value, i-1));
        value->release();
        iinterface->release();
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i-1)].get()));
    }

    // check that set_value() fails with invalid key/index/value
    MI_CHECK_EQUAL( -1, map->set_value( zero_string, stl_map[ get_key( 0)].get()));
    MI_CHECK_EQUAL( -2, map->set_value( "", stl_map[ get_key( 0)].get()));
    MI_CHECK_EQUAL( -2, map->set_value( "foo", stl_map[ get_key( 0)].get()));
    MI_CHECK_EQUAL( -2, map->set_value( N, stl_map[ get_key( 0)].get()));
    MI_CHECK_EQUAL( -2, map->set_value( 1u << 31, stl_map[ get_key( 0)].get()));
    MI_CHECK_EQUAL( -1, map->set_value( "key_0", nullptr));

    // check that get_value() fails with invalid key/index
    MI_CHECK_EQUAL( nullptr, map->get_value( zero_string));
    MI_CHECK_EQUAL( nullptr, map->get_value( ""));
    MI_CHECK_EQUAL( nullptr, map->get_value( "foo"));
    MI_CHECK_EQUAL( nullptr, map->get_value( N));
    MI_CHECK_EQUAL( nullptr, map->get_value( 1u << 31));

    // check that set_value()/insert() fail with values of wrong type (for typed maps)
    if( untyped) {
        // check that untyped maps accept any types
        MI_CHECK_EQUAL( 0, map->set_value( zero_size, void_.get()));
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( void_.get()));
        MI_CHECK_EQUAL( 0, map->set_value( zero_size, stl_map[ get_key( 0)].get()));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));

        MI_CHECK_EQUAL( 0, map->insert( "bar", void_.get()));
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( void_.get()));
        MI_CHECK_EQUAL( 0, map->erase( "bar"));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));
    } else {
        // check that typed maps reject other types
        MI_CHECK_EQUAL( -3, map->set_value( zero_size, void_.get()));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));

        MI_CHECK_EQUAL( -3, map->insert( "bar", void_.get()));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( void_.get()));
    }

    // set map contents via key
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 0, map->set_value( get_key( i), stl_map[ get_key( i)].get()));
    }
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 2, GET_REFCOUNT( stl_map[ get_key( i)].get()));
    }

    // erase some map contents (via key)
    for( mi::Size i=0; i < N/2; ++i) {
        MI_CHECK_EQUAL( 0, map->erase( get_key( i)));
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( stl_map[ get_key( i)].get()));
        MI_CHECK_EQUAL( N-i-1, map->get_length());
        MI_CHECK( !map->empty());
    }
    MI_CHECK_EQUAL( -1, map->erase( zero_string));
    MI_CHECK_EQUAL( -2, map->erase( ""));
    MI_CHECK_EQUAL( -2, map->erase( "foo"));

    // clear map
    map->clear();
    MI_CHECK_EQUAL( 0, map->get_length());
    MI_CHECK( map->empty());

    // only the STL vector now holds a reference
    for( mi::Size i=0; i < N; ++i) {
        MI_CHECK_EQUAL( 1, GET_REFCOUNT( stl_map[ get_key( i)].get()));
    }
}

// Does not work if T is derived from mi::IVoid or mi::neuraylib::IGroup because these are
// hard-coded interfaces to test some expected failures.
template<class T>
void test(
    mi::neuraylib::ITransaction* transaction,
    const char* value_type_name)
{
    mi::base::Handle<mi::IData> data( transaction->create<mi::IData>( value_type_name));
    bool untyped = !data.is_valid_interface();

    std::string type_name = "Map<";
    type_name += untyped ? "Interface" : value_type_name;
    type_name += ">";
    mi::base::Handle<mi::IMap> map( transaction->create<mi::IMap>( type_name.c_str()));
    MI_CHECK( map.is_valid_interface());
    test_interface_IMap<T>( transaction, map.get(), type_name.c_str(), value_type_name, untyped);

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

        test<mi::ISint32>( transaction.get(), "Sint32");
        test<mi::IString>( transaction.get(), "String");

        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());

}

MI_TEST_AUTO_FUNCTION( test_types_map )
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

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

