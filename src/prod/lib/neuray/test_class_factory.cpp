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
 ** \brief
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/iattribute_container.h>
#include <mi/neuraylib/ibsdf_measurement.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/ilightprofile.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/itexture.h>
#include <mi/neuraylib/itransaction.h>


#include "test_shared.h"

#define GET_REFCOUNT(X) ((X) ? (X)->retain(), (X)->release() : 999)

template <typename I>
void test_generic( mi::neuraylib::ITransaction* transaction, const char* class_name)
{
    // test create()

    mi::base::Handle<I> class_( transaction->create<I>( class_name));
    if( !class_.is_valid_interface())
        std::cerr << "Failed to create instance of class " << class_name << std::endl;
    MI_CHECK( class_.is_valid_interface());

    // test store(), access() and edit() if it is a scene element and the name does not start with "__"

    mi::base::Handle<mi::neuraylib::IScene_element> scene_element(
        class_.template get_interface<mi::neuraylib::IScene_element>());
    if( scene_element.is_valid_interface() && class_name[0] != '_' && class_name[1] != '_') {

        std::string db_element_name = "instance_of_";
        db_element_name += class_name;

        MI_CHECK_EQUAL( 0, transaction->store( scene_element.get(), db_element_name.c_str()));

        mi::base::Handle<const I> access( transaction->access<I>( db_element_name.c_str()));
        MI_CHECK( access.is_valid_interface());

        mi::base::Handle<I> edit( transaction->edit<I>( db_element_name.c_str()));
        MI_CHECK( edit.is_valid_interface());
    }
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

        mi::base::Handle<mi::neuraylib::ITransaction> transaction( global_scope->create_transaction());

        // generic create()/store()/access()/edit() test for API classes with DB counterparts
        test_generic<mi::neuraylib::IAttribute_container>(     transaction.get(), "Attribute_container");
        test_generic<mi::neuraylib::IBsdf_measurement>(        transaction.get(), "Bsdf_measurement");
        test_generic<mi::neuraylib::ICompiled_material>(       transaction.get(), "__Compiled_material");
        test_generic<mi::neuraylib::IImage>(                   transaction.get(), "Image");
        test_generic<mi::neuraylib::ILightprofile>(            transaction.get(), "Lightprofile");
        test_generic<mi::neuraylib::IModule>(                  transaction.get(), "__Module");
        test_generic<mi::neuraylib::ITexture>(                 transaction.get(), "Texture");
        // IFunction_definition and IFunction_call need extra arguments


        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_class_factory )
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

