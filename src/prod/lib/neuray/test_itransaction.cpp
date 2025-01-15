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
            // check that the last created, not released edit() wins -- release in creation order
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
            // check that the last created, not released edit() wins -- release in reverse creation order
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

    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_itransaction )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray.is_valid_interface());

    {


        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

