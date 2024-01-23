/******************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/set_get.h>

#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>

#include "test_shared.h"

void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        // dummy object to test IRef
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IImage> image( transaction->create<mi::neuraylib::IImage>( "Image"));
        transaction->store( image.get(), "dummy");

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        mi::Sint32 result;

        {
            // IBoolean (to check that the #pragma in set_get.h works)
            mi::base::Handle<mi::IBoolean> data( factory->create<mi::IBoolean>( "Boolean"));
            bool value = true;
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, false);
            result = set_value( data.get(), true);
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, true);
        }
        {
            // ISint32
            mi::base::Handle<mi::ISint32> data( factory->create<mi::ISint32>( "Sint32"));
            mi::Sint32 value = 1;
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 0);
            result = set_value( data.get(), 42);
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 42);
        }
        {
            // IString
            mi::base::Handle<mi::IData> data( factory->create<mi::IData>( "String"));
            const char* value;
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL_CSTR( value, "");
            result = set_value( data.get(), "foobar");
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL_CSTR( value, "foobar");
        }
        {
            // IUuid
            mi::base::Handle<mi::IData> data( factory->create<mi::IData>( "Uuid"));
            mi::base::Uuid zero  = { 0, 0, 0, 0 };
            mi::base::Uuid one   = { 1, 1, 1, 1 };
            mi::base::Uuid value = one;
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK( value == zero );
            result = set_value( data.get(), one);
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK( value == one);
        }
        {
            // IRef
            mi::base::Handle<mi::IData> data( transaction->create<mi::IData>( "Ref"));
            const char* value = "foo";
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 0);
            result = set_value( data.get(), "dummy");
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL_CSTR( value, "dummy");
            result = set_value( data.get(), zero_string);
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 0);
        }
        {
            // IFloat32_2 via explicit method
            mi::base::Handle<mi::IData> data( factory->create<mi::IData>( "Float32<2>"));
            mi::Float32_2 value( 1.0f, 2.0f);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value[0], 0.0f);
            MI_CHECK_EQUAL( value[1], 0.0f);
            result = set_value( data.get(), mi::Float32_2( 42.0f, 43.0f));
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value[0], 42.0f);
            MI_CHECK_EQUAL( value[1], 43.0f);
        }
        {
            // IFloat64_2_2 via explicit method
            mi::base::Handle<mi::IData> data( factory->create<mi::IData>( "Float64<2,2>"));
            mi::Float64_2_2 value( 1.0, 2.0, 3.0, 4.0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value.get( 0, 0), 0.0);
            MI_CHECK_EQUAL( value.get( 0, 1), 0.0);
            MI_CHECK_EQUAL( value.get( 1, 0), 0.0);
            MI_CHECK_EQUAL( value.get( 1, 1), 0.0);
            result = set_value( data.get(), mi::Float64_2_2( 42.0, 43.0, 44.0, 45.0));
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value.get( 0, 0), 42.0);
            MI_CHECK_EQUAL( value.get( 0, 1), 43.0);
            MI_CHECK_EQUAL( value.get( 1, 0), 44.0);
            MI_CHECK_EQUAL( value.get( 1, 1), 45.0);
        }
        {
            // IFloat32_3 via generic method using an index
            mi::base::Handle<mi::IData> data( factory->create<mi::IData>( "Float32<3>"));
            mi::Float32 value = 1.0f;
            result = get_value( data.get(), 1, value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 0.0f);
            result = set_value( data.get(), 1, 42.0f);
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), 1, value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 42.0f);
        }
        {
            // IFloat32_3 via generic method using a key
            mi::base::Handle<mi::IData> data( factory->create<mi::IData>( "Float32<3>"));
            mi::Float32 value = 1.0f;
            result = get_value( data.get(), "y", value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 0.0f);
            result = set_value( data.get(), "y", 42.0f);
            MI_CHECK_EQUAL( result, 0);
            result = get_value( data.get(), "y", value);
            MI_CHECK_EQUAL( result, 0);
            MI_CHECK_EQUAL( value, 42.0f);
        }

        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_set_get )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray.is_valid_interface());

    {
        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray = 0;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

