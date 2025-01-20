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
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"


void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0,  neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        MI_CHECK( database);

        // test scope creation / privacy levels

        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        MI_CHECK( global_scope);

        mi::base::Handle<mi::neuraylib::IScope> child_scope(
            database->create_scope( global_scope.get()));
        MI_CHECK( child_scope);

        mi::base::Handle<mi::neuraylib::IScope> grandchild_scope(
            database->create_scope( child_scope.get()));
        MI_CHECK( grandchild_scope);

        mi::base::Handle<mi::neuraylib::IScope> grandgrandchild_scope(
            database->create_scope( grandchild_scope.get(), 100));
        MI_CHECK( grandgrandchild_scope);

        MI_CHECK_EQUAL( 0, global_scope->get_privacy_level());
        MI_CHECK_EQUAL( 1, child_scope->get_privacy_level());
        MI_CHECK_EQUAL( 2, grandchild_scope->get_privacy_level());
        MI_CHECK_EQUAL( 100, grandgrandchild_scope->get_privacy_level());

        MI_CHECK( !global_scope->get_parent());
        mi::base::Handle<mi::neuraylib::IScope> temp_scope(grandchild_scope->get_parent());
        MI_CHECK_EQUAL_CSTR( temp_scope->get_id(), child_scope->get_id());

        // test named scopes

        // check that a named scope which does not exist is not found
        mi::base::Handle<mi::neuraylib::IScope> named_scope(database->get_named_scope( "test_scope"));
        MI_CHECK( !named_scope);

        // create a named scope and check for success and check that it can be retrieved
        named_scope = database->create_or_get_named_scope("test_scope");
        MI_CHECK( named_scope);
        std::string scope_id = named_scope->get_id();
        named_scope = database->get_named_scope( "test_scope");
        MI_CHECK( named_scope);

        // check that it is not possible to generate a second named scope with the same name but
        // different properties
        named_scope = database->create_or_get_named_scope("test_scope", named_scope.get(), 3);
        MI_CHECK( !named_scope);

        // check that it is possible to generate a second named scope with the same name but
        // same properties. This must return the original scope
        named_scope = database->create_or_get_named_scope("test_scope");
        MI_CHECK( named_scope);
        MI_CHECK_EQUAL_CSTR( named_scope->get_id(), scope_id.c_str());
        MI_CHECK_EQUAL_CSTR( named_scope->get_name(), "test_scope");

        // check that it is possible to create a descendant of the named scope
        named_scope = database->create_or_get_named_scope("test_scope2", named_scope.get(), 3);
        MI_CHECK( named_scope);
        named_scope = database->get_named_scope( "test_scope2");
        MI_CHECK( named_scope);
        MI_CHECK( named_scope->get_privacy_level() == 3);
        MI_CHECK_EQUAL_CSTR( named_scope->get_name(), "test_scope2");

        // check that the original named scope is not affected by creating a second one with a
        // different name
        named_scope = database->get_named_scope( "test_scope");
        MI_CHECK( named_scope);
        MI_CHECK( named_scope->get_privacy_level() == 1);
        named_scope = database->get_named_scope( "test_scope");

    }

    MI_CHECK_EQUAL( 0,  neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_iscope )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    run_tests( neuray.get());
    run_tests( neuray.get());

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

