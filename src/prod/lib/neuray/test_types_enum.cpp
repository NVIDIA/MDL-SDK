/******************************************************************************
 * Copyright (c) 2013-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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
 ** \brief Test for mi::IEnum and mi::IEnum_decl
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/iextension_api.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>

#include <mi/neuraylib/iattribute_container.h>

#include "test_shared.h"

const char* decl_enumerator_names[] = { "ONE", "TWO", "EINS", "FORTY_TWO" };

mi::Sint32 decl_enumerator_values[] = { 1, 2, 1, 42 };

void test_decl( mi::neuraylib::IExtension_api* extension_api, mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::IEnum_decl> decl( transaction->create<mi::IEnum_decl>( "Enum_decl"));
    MI_CHECK( decl);

    MI_CHECK_EQUAL( -1, decl->add_enumerator( 0, 42));
    MI_CHECK_EQUAL(  0, decl->add_enumerator( "FOO", 42));
    MI_CHECK_EQUAL( -2, decl->add_enumerator( "FOO", 42));

    MI_CHECK_EQUAL( -1, decl->remove_enumerator( 0));
    MI_CHECK_EQUAL(  0, decl->remove_enumerator( "FOO"));
    MI_CHECK_EQUAL(  -2, decl->remove_enumerator( "FOO"));

    MI_CHECK_EQUAL(  0, decl->add_enumerator( "FOO", 42));
    MI_CHECK_EQUAL(  nullptr, decl->get_enum_type_name());

    // test IExtension_api::register_enum_decl()
    MI_CHECK_EQUAL(  0, extension_api->register_enum_decl( "Test1", decl.get()));
    MI_CHECK_EQUAL( -1, extension_api->register_enum_decl( "Test1", decl.get()));
    MI_CHECK_EQUAL( -2, extension_api->register_enum_decl( 0, decl.get()));
    MI_CHECK_EQUAL( -2, extension_api->register_enum_decl( "Test1", 0));
    MI_CHECK_EQUAL( -4, extension_api->register_enum_decl( "Invalid_enum_name[]<>{};", decl.get()));

    // test IFactory::get_enum_decl()
    mi::base::Handle<const mi::IEnum_decl> copy( factory->get_enum_decl( "Test1"));
    MI_CHECK( copy);
    MI_CHECK_NOT_EQUAL( decl.get(), copy.get());
    MI_CHECK_EQUAL( nullptr, decl->get_enum_type_name());
    MI_CHECK_EQUAL_CSTR( copy->get_enum_type_name(), "Test1");
    MI_CHECK_EQUAL( nullptr, factory->get_enum_decl( "Non_existing"));

    // test instance creation
    mi::base::Handle<mi::IEnum> enum_( transaction->create<mi::IEnum>( "Test1"));
    mi::base::Handle<const mi::IEnum_decl> tmp_decl( enum_->get_enum_decl());
    MI_CHECK_EQUAL( 1, tmp_decl->get_length());

    // add another enumerator, test that registered declaration for "Test1" is not modified
    MI_CHECK_EQUAL(  0, decl->add_enumerator( "BAR", 43));
    MI_CHECK_EQUAL(  0, extension_api->register_enum_decl( "Test2", decl.get()));
    enum_ = transaction->create<mi::IEnum>( "Test1");
    tmp_decl = enum_->get_enum_decl();
    MI_CHECK_EQUAL( 1, tmp_decl->get_length());
    enum_ = transaction->create<mi::IEnum>( "Test2");
    tmp_decl = enum_->get_enum_decl();
    MI_CHECK_EQUAL( 2, tmp_decl->get_length());

    // test IExtension_api::unregister_enum_decl()
    MI_CHECK_EQUAL(  0, extension_api->unregister_enum_decl( "Test1"));
    MI_CHECK_EQUAL(  0, extension_api->unregister_enum_decl( "Test2"));
    MI_CHECK_EQUAL( -1, extension_api->unregister_enum_decl( "Test1"));
    MI_CHECK_EQUAL( -1, extension_api->unregister_enum_decl( "Non_existing"));
    MI_CHECK_EQUAL( -2, extension_api->unregister_enum_decl( 0));
    MI_CHECK_EQUAL( -4, extension_api->unregister_enum_decl( "Invalid_enum_name[]<>{};"));

    // register the real enum decl
    {

        mi::base::Handle<mi::IEnum_decl> decl( transaction->create<mi::IEnum_decl>( "Enum_decl"));
        mi::Size n = sizeof( decl_enumerator_names) / sizeof( const char*);
        for( mi::Size i = 0; i < n; ++i)
            MI_CHECK_EQUAL( 0, decl->add_enumerator( decl_enumerator_names[i], decl_enumerator_values[i]));
        MI_CHECK_EQUAL( 0, extension_api->register_enum_decl( "Test_enum", decl.get()));
    }

    // instantiate the real enum decl
    {
        mi::base::Handle<mi::IEnum> enum_( transaction->create<mi::IEnum>( "Test_enum"));
        MI_CHECK( enum_);

        mi::base::Handle<const mi::IEnum_decl> decl( enum_->get_enum_decl());
        MI_CHECK( decl);

        mi::Size n = decl->get_length();
        for( mi::Size i = 0; i < n; ++i) {
            MI_CHECK_EQUAL_CSTR( decl->get_name( i), decl_enumerator_names[i]);
            MI_CHECK_EQUAL( decl->get_value( i), decl_enumerator_values[i]);
        }
    }
}

void test_generic( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::IEnum> enum_( transaction->create<mi::IEnum>( "Test_enum"));

    // test default value
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE");
    MI_CHECK_EQUAL( enum_->get_value(), 1);

    // set enumerator by name
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "TWO"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "TWO");
    MI_CHECK_EQUAL( enum_->get_value(), 2);
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "EINS"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "EINS");
    MI_CHECK_EQUAL( enum_->get_value(), 1);
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "ONE"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE");
    MI_CHECK_EQUAL( enum_->get_value(), 1);

    // set enumerator by value
    MI_CHECK_EQUAL( 0, enum_->set_value( 2));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "TWO");
    MI_CHECK_EQUAL( enum_->get_value(), 2);
    MI_CHECK_EQUAL( 0, enum_->set_value( 1));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE"); // (!)
    MI_CHECK_EQUAL( enum_->get_value(), 1);
    MI_CHECK_EQUAL( 0, enum_->set_value( 1));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE");
    MI_CHECK_EQUAL( enum_->get_value(), 1);

    // invalid set operations
    MI_CHECK_EQUAL( -1, enum_->set_value_by_name( "INVALID_ENUMERATOR"));
    MI_CHECK_EQUAL( -1, enum_->set_value( 3));
}


void test_attribute( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::neuraylib::IAttribute_container> db_element(
        transaction->create<mi::neuraylib::IAttribute_container>( "Attribute_container"));
    MI_CHECK( db_element);

    mi::base::Handle<mi::IEnum> enum_( db_element->create_attribute<mi::IEnum>( "test_enum", "Test_enum"));

    // test default value
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE");
    MI_CHECK_EQUAL( enum_->get_value(), 1);

    // set enumerator by name
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "TWO"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "TWO");
    MI_CHECK_EQUAL( enum_->get_value(), 2);
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "EINS"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "EINS");
    MI_CHECK_EQUAL( enum_->get_value(), 1);
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "ONE"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE");
    MI_CHECK_EQUAL( enum_->get_value(), 1);

    // set enumerator by value
    MI_CHECK_EQUAL( 0, enum_->set_value( 2));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "TWO");
    MI_CHECK_EQUAL( enum_->get_value(), 2);
    MI_CHECK_EQUAL( 0, enum_->set_value( 1));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE"); // (!)
    MI_CHECK_EQUAL( enum_->get_value(), 1);
    MI_CHECK_EQUAL( 0, enum_->set_value( 1));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "ONE");
    MI_CHECK_EQUAL( enum_->get_value(), 1);

    // invalid set operations
    MI_CHECK_EQUAL( -1, enum_->set_value_by_name( "INVALID_ENUMERATOR"));
    MI_CHECK_EQUAL( -1, enum_->set_value( 3));

    // check serialization
    MI_CHECK_EQUAL( 0, enum_->set_value_by_name( "FORTY_TWO"));
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "FORTY_TWO");
    MI_CHECK_EQUAL( enum_->get_value(), 42);
    enum_.reset();

    MI_CHECK_EQUAL( 0, transaction->store( db_element.get(), "db_element"));
    db_element = transaction->edit<mi::neuraylib::IAttribute_container>( "db_element");

    enum_ = db_element->edit_attribute<mi::IEnum>( "test_enum");
    MI_CHECK_EQUAL_CSTR( enum_->get_value_by_name(), "FORTY_TWO");
    MI_CHECK_EQUAL( enum_->get_value(), 42);
}


void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( global_scope->create_transaction());

        // run tests
        mi::base::Handle<mi::neuraylib::IExtension_api> extension_api(
            neuray->get_api_component<mi::neuraylib::IExtension_api>());
        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK( extension_api);
        test_decl( extension_api.get(), factory.get(), transaction.get());
        test_generic( transaction.get());

        // run attribute tests
        test_attribute( transaction.get());

        MI_CHECK_EQUAL( 0, transaction->commit());

        MI_CHECK_EQUAL( 0, extension_api->unregister_enum_decl( "Test_enum"));
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_types_enum )
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

    neuray.reset();
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

