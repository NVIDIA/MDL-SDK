/******************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Test driver for MI::NEURAY::IDb_element
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"

#define GET_REFCOUNT(X) ((X) ? (X)->retain(), (X)->release() : 999)


void test_texture( mi::neuraylib::IScope* scope)
{
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

    // create dummy members

    mi::base::Handle<mi::neuraylib::IImage> image( transaction->create<mi::neuraylib::IImage>( "Image"));
    MI_CHECK( image.is_valid_interface());
    MI_CHECK_EQUAL( 0, transaction->store( image.get(), "referenced_image_1"));
    image = 0;

    image = transaction->create<mi::neuraylib::IImage>( "Image");
    MI_CHECK( image.is_valid_interface());
    MI_CHECK_EQUAL( 0, transaction->store( image.get(), "referenced_image_2"));
    image = 0;

    // test create() and store()

    mi::base::Handle<mi::neuraylib::ITexture> texture( transaction->create<mi::neuraylib::ITexture>( "Texture"));
    MI_CHECK( texture.is_valid_interface());
    texture->set_image( "referenced_image_1");
    MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "texture_foo"));
    texture = 0;

    // test access()

    mi::base::Handle<const mi::neuraylib::ITexture> c_texture( transaction->access<mi::neuraylib::ITexture>( "texture_foo"));
    MI_CHECK( c_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( c_texture->get_image(), "referenced_image_1");
    c_texture = 0;

    // test edit()

    mi::base::Handle<mi::neuraylib::ITexture> m_texture( transaction->edit<mi::neuraylib::ITexture>( "texture_foo"));
    MI_CHECK( m_texture.is_valid_interface());
    m_texture->set_image( "referenced_image_2");
    m_texture = 0;

    // test access() on modified element

    c_texture = transaction->access<mi::neuraylib::ITexture>( "texture_foo");
    MI_CHECK( c_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( c_texture->get_image(), "referenced_image_2");
    c_texture = 0;

    // test edit() on modified element

    m_texture = transaction->edit<mi::neuraylib::ITexture>( "texture_foo");
    MI_CHECK( m_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( m_texture->get_image(), "referenced_image_2");
    m_texture = 0;

    // test concurrent access()'s and edit()'s

    mi::base::Handle<const mi::neuraylib::ITexture> c_texture2;
    mi::base::Handle<mi::neuraylib::ITexture> m_texture2;
    c_texture  = transaction->access<mi::neuraylib::ITexture>( "texture_foo");
    c_texture2 = transaction->access<mi::neuraylib::ITexture>( "texture_foo");
    m_texture  = transaction->edit<mi::neuraylib::ITexture>( "texture_foo");
    m_texture2 = transaction->edit<mi::neuraylib::ITexture>( "texture_foo");
    c_texture  = 0;
    c_texture2 = 0;
    m_texture  = 0;
    m_texture2 = 0;
    c_texture  = transaction->access<mi::neuraylib::ITexture>( "texture_foo");
    c_texture2 = transaction->access<mi::neuraylib::ITexture>( "texture_foo");
    m_texture  = transaction->edit<mi::neuraylib::ITexture>( "texture_foo");
    m_texture2 = transaction->edit<mi::neuraylib::ITexture>( "texture_foo");
    m_texture2 = 0;
    m_texture  = 0;
    c_texture2 = 0;
    c_texture  = 0;

    // test copy()

    MI_CHECK_EQUAL(  0, transaction->copy( "texture_foo", "texture_copy"));
    MI_CHECK_EQUAL( -5, transaction->copy( "texture_foo", "texture_foo", 0));

    c_texture = transaction->access<mi::neuraylib::ITexture>( "texture_copy");
    MI_CHECK( c_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( c_texture->get_image(), "referenced_image_2");
    c_texture = 0;

    m_texture = transaction->edit<mi::neuraylib::ITexture>( "texture_copy");
    MI_CHECK( m_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( m_texture->get_image(), "referenced_image_2");
    m_texture = 0;

    // test remove()

    // Disabled because reusing the name after removal might trigger an invalid tag access
    // (see also test case in bug 8577).
    // MI_CHECK_EQUAL( 0, transaction->remove( "texture_copy"));

    // test name_of()

    c_texture = transaction->access<mi::neuraylib::ITexture>( "texture_foo");
    MI_CHECK( c_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( "texture_foo", transaction->name_of( c_texture.get()));
    c_texture = 0;

    m_texture = transaction->edit<mi::neuraylib::ITexture>( "texture_foo");
    MI_CHECK( m_texture.is_valid_interface());
    MI_CHECK_EQUAL_CSTR( "texture_foo", transaction->name_of( m_texture.get()));
    m_texture = 0;

    // test get_timestamp() / has_changed_since_timestamp()

    std::string time_stamp1 = transaction->get_time_stamp();

    mi::base::Handle<mi::neuraylib::ITexture> x_texture( transaction->create<mi::neuraylib::ITexture>( "Texture"));
    MI_CHECK( x_texture.is_valid_interface());
    MI_CHECK_EQUAL( 0, transaction->store( x_texture.get(), "texture_bar"));
    x_texture = 0;

    std::string time_stamp2a = transaction->get_time_stamp();
    std::string time_stamp2b = transaction->get_time_stamp( "texture_bar");
    MI_CHECK_EQUAL( time_stamp2a, time_stamp2b);

    MI_CHECK(  transaction->has_changed_since_time_stamp( "texture_bar", time_stamp1.c_str()));
    MI_CHECK( !transaction->has_changed_since_time_stamp( "texture_bar", time_stamp2a.c_str()));

    MI_CHECK_EQUAL( 0, transaction->commit());
    transaction = scope->create_transaction();
    MI_CHECK( transaction.is_valid_interface());

    std::string time_stamp3a = transaction->get_time_stamp();
    std::string time_stamp3b = transaction->get_time_stamp( "texture_bar");
    MI_CHECK_NOT_EQUAL( time_stamp3a, time_stamp3b);
    MI_CHECK_EQUAL( time_stamp2b, time_stamp3b);

    MI_CHECK(  transaction->has_changed_since_time_stamp( "texture_bar", time_stamp1.c_str()));
    MI_CHECK( !transaction->has_changed_since_time_stamp( "texture_bar", time_stamp2a.c_str()));
    MI_CHECK( !transaction->has_changed_since_time_stamp( "texture_bar", time_stamp3a.c_str()));

    MI_CHECK_EQUAL( 0, transaction->commit());
}

void test_string( mi::neuraylib::IScope* scope)
{
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

    // test create() and store()

    mi::base::Handle<mi::IString> str( transaction->create<mi::IString>( "String"));
    MI_CHECK( str.is_valid_interface());
    str->set_c_str( "foo");
    MI_CHECK_EQUAL_CSTR( "foo", str->get_c_str());
    MI_CHECK_EQUAL( -4, transaction->store( str.get(), "string_foo"));
    str = 0;

    MI_CHECK_EQUAL( 0, transaction->commit());
}


void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        MI_CHECK( database.is_valid_interface());

        mi::base::Handle<mi::neuraylib::IScope> global_scope(
            database->get_global_scope());
        MI_CHECK( global_scope.is_valid_interface());
        MI_CHECK_EQUAL( 0, global_scope->get_privacy_level());


        // test an API class which references another DB element by tag
        test_texture( global_scope.get());

        // test an API class without DB class counterpart
        test_string( global_scope.get());

    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_db_elements )
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

