/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/config.h>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imodule.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/istring.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include "test_shared.h"


#define MI_CHECK_CTX( context) \
    if( context->get_error_messages_count() > 0) { \
        for( mi::Size i = 0, n = context->get_messages_count(); i < n; ++i) { \
            mi::base::Handle<const mi::neuraylib::IMessage> message( context->get_message( i)); \
            std::cerr << message->get_string() << std::endl; \
        } \
        MI_CHECK_EQUAL( context->get_error_messages_count(), 0); \
    } else { }

// There used to be several variants of this unit test. To avoid race conditions, each variant used a
// separate subdirectory for all files it creates (possibly with further subdirectories).
#define DIR_PREFIX "output_test_target_materials"


void copy_file(const std::string & filename1, const std::string & filename2)
{
    std::remove(filename2.c_str());
    std::ifstream file1(filename1, std::ios::binary);
    std::ofstream file2(filename2, std::ios::binary);
    file2 << file1.rdbuf();
}

mi::neuraylib::ICompiled_material const* compile_material_tmm(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_execution_context* context,
    const char* module_name,
    const char* definition_name,
    const char* instance_name
) {
    // Access the loaded test material and create a material instance from it.
    mi::base::Handle<const mi::neuraylib::IModule>
        c_module(transaction->access<mi::neuraylib::IModule>(module_name));
    mi::base::Handle<const mi::IArray>
        overloads(c_module->get_function_overloads(definition_name));
    MI_CHECK_EQUAL(1, overloads->get_length());

    mi::base::Handle<const mi::IString> element(overloads->get_element<mi::IString>(0));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
        transaction->access<mi::neuraylib::IFunction_definition>(element->get_c_str()));
    MI_CHECK(md);
    {
        mi::Sint32 result = 0;

        mi::base::Handle<mi::neuraylib::IFunction_call> mi(
            md->create_function_call(0, &result));
        MI_CHECK_EQUAL(0, result);
        MI_CHECK_EQUAL(0, transaction->store(mi.get(), instance_name));
    }

    // Compile the material instance.
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory(transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory(transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_call>
        mi(transaction->access<const mi::neuraylib::IFunction_call>(
            instance_name));
    MI_CHECK(mi.is_valid_interface());
    mi::base::Handle<const mi::neuraylib::IMaterial_instance>
        mi_mi(mi->get_interface<mi::neuraylib::IMaterial_instance>());
    MI_CHECK(mi_mi.is_valid_interface());

    mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;
    mi::base::Handle<const mi::neuraylib::ICompiled_material>
        cm(mi_mi->create_compiled_material(flags, context));
    MI_CHECK_CTX(context);
    MI_CHECK(cm);
    cm->retain();
    return cm.get();
}

void check_icompiled_material_tmm(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    mi::Sint32 result = 0;

    // Prepare input files for loading materials in different modes.
    copy_file(MI::TEST::mi_src_path("prod/lib/neuray") + "/test_mdl_tmm_orig.mdl",
        DIR_PREFIX "/test_mdl_tmm0.mdl");
    copy_file(MI::TEST::mi_src_path("prod/lib/neuray") + "/test_mdl_tmm_orig.mdl",
        DIR_PREFIX "/test_mdl_tmm1.mdl");

    // Load with target material mode enabled.
    {
        // Load the material for testing by setting the option for target material model
        // mode compilation in the execution context.
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        context->set_option("target_material_model_mode", true);
        result = mdl_impexp_api->load_module(transaction, "::test_mdl_tmm0", context.get());
        MI_CHECK_CTX(context.get());
        MI_CHECK_EQUAL(result, 0);

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>("mdl::test_mdl_tmm0"));
        MI_CHECK_EQUAL(0, result);
        MI_CHECK_EQUAL(2, module->get_material_count());

        mi::base::Handle<const mi::neuraylib::ICompiled_material>
            cm(compile_material_tmm(transaction, mdl_factory, context.get(),
                "mdl::test_mdl_tmm0",
                "mdl::test_mdl_tmm0::md_tmm",
                "mdl::test_mdl_tmm0::mi_tmm"));

        // Check expected properties of the DAG compilation result.
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body(cm->get_body());
        const char* def_name = body->get_definition();
        MI_CHECK_EQUAL_CSTR(def_name, "mdl::test_mdl_tmm0::md_target_material()");
        //dump(transaction, mdl_factory, cm.get(), std::cerr);

        mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
            transaction->access<mi::neuraylib::IFunction_definition>(def_name));
        MI_CHECK(md);
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> annos(md->get_annotations());
        MI_CHECK_EQUAL(annos->get_size(), 1);
        mi::base::Handle<const mi::neuraylib::IAnnotation> anno(annos->get_annotation(0));
        const char* anno_name = anno->get_name();
        MI_CHECK_EQUAL_CSTR(anno_name, "::nvidia::baking::target_material_model(string,string)");
    }

    // Load with target material mode disabled. This test is to ensure that the flag is properly
    // passed through the different API layers and recognized by the compiler.
    {
        // Load the material for testing by setting the option for target material model
        // mode compilation in the execution context.
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdl_impexp_api->load_module(transaction, "::test_mdl_tmm1", context.get());
        MI_CHECK_CTX(context.get());
        MI_CHECK_EQUAL(result, 0);

        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>("mdl::test_mdl_tmm1"));
        MI_CHECK_EQUAL(0, result);
        MI_CHECK_EQUAL(2, module->get_material_count());

        mi::base::Handle<const mi::neuraylib::ICompiled_material>
            cm(compile_material_tmm(transaction, mdl_factory, context.get(),
                "mdl::test_mdl_tmm1",
                "mdl::test_mdl_tmm1::md_tmm",
                "mdl::test_mdl_tmm1::mi_tmm"));

        // Check expected properties of the DAG compilation result.
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body(cm->get_body());
        const char* def_name = body->get_definition();
        MI_CHECK_EQUAL_CSTR(def_name,
            "mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)");
        //dump(transaction, mdl_factory, cm.get(), std::cerr);
    }
}

void run_tests(mi::neuraylib::INeuray* neuray)
{
    boost::filesystem::remove_all(DIR_PREFIX);
    boost::filesystem::create_directory(DIR_PREFIX);

    MI_CHECK_EQUAL(0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK(factory);

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());
        MI_CHECK(mdl_factory);

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        MI_CHECK(mdl_impexp_api);

        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK(mdl_configuration);
        MI_CHECK_EQUAL(0, mdl_configuration->add_mdl_path(DIR_PREFIX));
        MI_CHECK_EQUAL(0, mdl_configuration->add_resource_path(DIR_PREFIX));

        check_icompiled_material_tmm(transaction.get(), factory.get(), mdl_factory.get(),
            mdl_impexp_api.get());

        MI_CHECK_EQUAL(0, transaction->commit());
    }
    MI_CHECK_EQUAL(0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_target_materials )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        run_tests(neuray.get());
    }

    neuray = 0;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

