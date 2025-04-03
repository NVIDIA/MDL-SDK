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
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"

void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {

        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());

        mi::base::Handle<mi::neuraylib::ITransaction> transaction;

        // check abort() / get_id()
        transaction = scope->create_transaction();
        MI_CHECK_NOT_EQUAL( nullptr, transaction->get_id());
        MI_CHECK( transaction->is_open());
        transaction->abort();
        MI_CHECK( !transaction->is_open());

        // check commit / get_id()
        transaction = scope->create_transaction();
        MI_CHECK_NOT_EQUAL( nullptr, transaction->get_id());
        MI_CHECK( transaction->is_open());
        MI_CHECK_EQUAL( 0, transaction->commit());
        MI_CHECK( !transaction->is_open());

        // handles to texture
        mi::base::Handle<mi::neuraylib::ITexture> texture;
        mi::base::Handle<const mi::neuraylib::ITexture> c_texture1;
        mi::base::Handle<const mi::neuraylib::ITexture> c_texture2;
        mi::base::Handle<mi::neuraylib::ITexture> m_texture1;
        mi::base::Handle<mi::neuraylib::ITexture> m_texture2;
        mi::base::Handle<mi::neuraylib::ITexture> m_texture3;

        {
            // check that access() before an edit() does NOT see the changes
            transaction = scope->create_transaction();

            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = nullptr;

            c_texture1 = transaction->access<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, c_texture1->get_gamma());

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());

            m_texture1->set_gamma( 2.0);

            MI_CHECK_EQUAL( 1.0, c_texture1->get_gamma());
            MI_CHECK_EQUAL( 2.0, m_texture1->get_gamma());

            c_texture1 = nullptr;
            m_texture1 = nullptr;
            c_texture2 = nullptr;
            transaction->commit();
        }

        {
            // check that access() during an active edit() does see the changes
            transaction = scope->create_transaction();

            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = nullptr;

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());

            c_texture1 = transaction->access<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, c_texture1->get_gamma());

            m_texture1->set_gamma( 2.0);

            c_texture2 = transaction->access<mi::neuraylib::ITexture>( "dummy");

            MI_CHECK_EQUAL( 2.0, m_texture1->get_gamma());
            MI_CHECK_EQUAL( 2.0, c_texture1->get_gamma());
            MI_CHECK_EQUAL( 2.0, c_texture2->get_gamma());

            m_texture1 = nullptr;
            c_texture1 = nullptr;
            c_texture2 = nullptr;
            transaction->commit();
        }

        {
            // check that edit() during an active edit() does NOT see the changes
            transaction = scope->create_transaction();

            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = nullptr;

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());

            m_texture2 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture2->get_gamma());

            m_texture1->set_gamma( 2.0);

            m_texture3 = transaction->edit<mi::neuraylib::ITexture>( "dummy");

            MI_CHECK_EQUAL( 2.0, m_texture1->get_gamma());
            MI_CHECK_EQUAL( 1.0, m_texture2->get_gamma());
            MI_CHECK_EQUAL( 1.0, m_texture3->get_gamma());

            m_texture1 = nullptr;
            m_texture2 = nullptr;
            m_texture3 = nullptr;
            transaction->commit();
        }

        {
            // check that the last created, not released edit() wins: release in creation order
            transaction = scope->create_transaction();

            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = nullptr;

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());

            m_texture2 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture2->get_gamma());

            m_texture1->set_gamma( 2.0);
            m_texture2->set_gamma( 3.0);
            m_texture1 = nullptr;
            m_texture2 = nullptr;

            c_texture1 = transaction->access<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 3.0, c_texture1->get_gamma());

            c_texture1 = nullptr;
            transaction->commit();
        }

        {
            // check that the last created, not released edit() wins: release in reverse creation
            // order
            transaction = scope->create_transaction();

            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = nullptr;

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());

            m_texture2 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture2->get_gamma());

            m_texture1->set_gamma( 2.0);
            m_texture2->set_gamma( 3.0);
            m_texture2 = nullptr;
            m_texture1 = nullptr;

            c_texture1 = transaction->access<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 3.0, c_texture1->get_gamma());

            c_texture1 = nullptr;
            transaction->commit();
        }

        mi::base::Handle<mi::neuraylib::ITransaction> transaction1;
        mi::base::Handle<mi::neuraylib::ITransaction> transaction2;

        {
            // check that the last started, not committed transaction wins: commit in start order
            transaction = scope->create_transaction();
            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = 0;
            transaction->commit();

            transaction1 = scope->create_transaction();
            transaction2 = scope->create_transaction();

            m_texture1 = transaction1->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());
            m_texture2 = transaction2->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture2->get_gamma());

            m_texture1->set_gamma( 2.0);
            m_texture2->set_gamma( 3.0);
            m_texture1 = 0;
            m_texture2 = 0;

            transaction1->commit();
            transaction2->commit();

            transaction = scope->create_transaction();

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 3.0, m_texture1->get_gamma());
            m_texture1->set_gamma( 1.0);
            m_texture1 = 0;

            transaction->commit();
        }

        {
            // check that the last started, not committed transaction wins: commit in reverse start
            // order
            transaction = scope->create_transaction();
            texture = transaction->create<mi::neuraylib::ITexture>( "Texture");
            texture->set_gamma( 1.0);
            MI_CHECK_EQUAL( 0, transaction->store( texture.get(), "dummy"));
            texture = 0;
            transaction->commit();

            transaction1 = scope->create_transaction();
            transaction2 = scope->create_transaction();

            m_texture1 = transaction1->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture1->get_gamma());
            m_texture2 = transaction2->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 1.0, m_texture2->get_gamma());

            m_texture1->set_gamma( 2.0);
            m_texture2->set_gamma( 3.0);
            m_texture1 = 0;
            m_texture2 = 0;

            transaction2->commit();
            transaction1->commit();

            transaction = scope->create_transaction();

            m_texture1 = transaction->edit<mi::neuraylib::ITexture>( "dummy");
            MI_CHECK_EQUAL( 3.0, m_texture1->get_gamma());
            m_texture1->set_gamma( 1.0);
            m_texture1 = 0;

            transaction->commit();
        }

        {
            // check storing elements under names eligible for GC

            transaction = scope->create_transaction();
            {
                // Create "texture" referencing "image". Flag "image" for removal.
                mi::base::Handle image( transaction->create<mi::neuraylib::IImage>( "Image"));
                MI_CHECK_EQUAL( 0, transaction->store( image.get(), "image"));
                MI_CHECK_EQUAL( 0, transaction->remove( "image"));

                mi::base::Handle texture2( transaction->create<mi::neuraylib::ITexture>( "Texture"));
                MI_CHECK_EQUAL( 0, texture2->set_image( "image"));
                MI_CHECK_EQUAL( 0, transaction->store( texture2.get(), "texture"));
            }
            transaction->commit();
            transaction = scope->create_transaction();
            {
                // Flag "texture" for removal.
                MI_CHECK_EQUAL( 0, transaction->remove( "texture"));
            }
            transaction->commit();
            // "texture" is now eligible for GC, "image" not yet, but will become eligible as
            // soon as "texture" is garbage collected.
            transaction = scope->create_transaction();
            {
                // Store new element under "image".
                mi::base::Handle image( transaction->create<mi::neuraylib::IImage>( "Image"));
                MI_CHECK_EQUAL( 0, transaction->store( image.get(), "image"));

                // Two cases are possible here (not easily distinguishable via the API):
                // - the GC did not yet process "image", and just a new version was stored above
                //   (which inherited the removal flag)
                // - the GC did already process "image", and a new tag was allocated

                // Invoke the synchronous GC (instead of a delay to let the potentially
                // asynchronous GC run).
                database->garbage_collection();

                // Access the version just stored above under "image".
                mi::base::Handle test( transaction->access<mi::neuraylib::IImage>( "image"));
                MI_CHECK( test);
            }
            transaction->commit();
        }

    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_itransaction )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {


        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

