/******************************************************************************
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <vector>

#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/imdl_i18n_configuration.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imodule.h>

#include "test_shared.h"

using namespace mi::neuraylib;

mi::Sint32 result = 0;

void check_imdl_i18n_configuration(INeuray* neuray)
{
    // Get the internationalization configuration component
    using mi::base::Handle;
    Handle<IMdl_i18n_configuration> i18n_configuration(
        neuray->get_api_component<IMdl_i18n_configuration>());
    MI_CHECK(i18n_configuration);
    const char * locale = NULL;

    // Set locale to French language
    const std::string locale_string("fr");
    i18n_configuration->set_locale(locale_string.c_str());

    // Query defined locale
    locale = i18n_configuration->get_locale();
    MI_CHECK(locale_string == locale);

    // Query system defined locale
    locale = i18n_configuration->get_system_locale();

    // Use system defined locale
    i18n_configuration->set_locale(i18n_configuration->get_system_keyword());

    // Disable any translation
    i18n_configuration->set_locale(NULL);

    // Set locale to French language for the rest of the tests
    i18n_configuration->set_locale(locale_string.c_str());
}

class Traversal_context
{
public:
    virtual void push_annotation(const char* name, const char* value, const char* original_value) = 0;
    virtual void push_qualified_name(const char* name) = 0;
};

class Check_annotation : public Traversal_context
{
public:
    void push_annotation(const char* name, const char* value, const char* original_value)
    {
        if (name && value)
        {
            if (original_value)
            {
                // This annotation has been translated
                // std::cout << "New annotation (name/value/original value): " << name << " / " << value << " / " << original_value << std::endl;
                // Check the translation and the original values
                std::string v(value);
                std::string ov(original_value);
                MI_CHECK(v == "French " + ov);
            }
            else
            {
                // This annotation has not been translated
                // std::cout << "New annotation not translated (name/value): " << name << " / " << value << std::endl;
            }
        }
    }
    void push_qualified_name(const char* name)
    {
        // std::cout << "Name: " << name << std::endl;
    }
};

void annotations_from_annotation(const IAnnotation* anno, Traversal_context & context)
{
    const char* name = anno->get_name();

    mi::base::Handle<const IExpression_list> elist(anno->get_arguments());

    for (mi::Size i = 0; i < elist->get_size(); i++)
    {
        mi::base::Handle<const IExpression> expr(elist->get_expression(mi::Size(i)));
        MI_CHECK(expr);
        const IExpression::Kind kind = expr->get_kind();
        if (kind == IExpression::EK_CONSTANT)
        {
            mi::base::Handle<const IExpression_constant> expr_constant(expr->get_interface<IExpression_constant>());
            MI_CHECK(expr_constant);
            mi::base::Handle<const IValue> value(expr_constant->get_value());
            MI_CHECK(value);

            if (value->get_kind() == IValue::VK_STRING)
            {
                const char* char_value(NULL);
                const char* char_original_value(NULL);
                mi::base::Handle<const IValue_string_localized> value_string_localized(value->get_interface<IValue_string_localized>());
                if (value_string_localized)
                {
                    MI_CHECK(value_string_localized);
                    char_value = value_string_localized->get_value();
                    MI_CHECK(char_value);
                    char_original_value = value_string_localized->get_original_value();
                    MI_CHECK(char_original_value);
                }
                else
                {
                    mi::base::Handle<const IValue_string> value_string(value->get_interface<IValue_string>());
                    MI_CHECK(value_string);
                    char_value = value_string->get_value();
                    MI_CHECK(char_value);
                }
                context.push_annotation(name, char_value, char_original_value);
            }
        }
    }
}

void annotations_from_annotation_block(const IAnnotation_block* ablock, Traversal_context & context)
{
    for (mi::Size i = 0; i < ablock->get_size(); i++)
    {
        mi::base::Handle<const IAnnotation> anno(ablock->get_annotation(i));
        if (anno)
        {
            annotations_from_annotation(anno.get(), context);
        }
    }
}

void annotations_from_module(const IModule* module, Traversal_context & context)
{
    mi::base::Handle<const IAnnotation_block> ablock(module->get_annotations());
    if (ablock)
    {
        annotations_from_annotation_block(ablock.get(), context);
    }
}

void annotations_from_annotation_list(const IAnnotation_list* o, Traversal_context & context)
{
    if (o)
    {
        for (mi::Size i = 0; i < o->get_size(); i++)
        {
            mi::base::Handle<const IAnnotation_block> anno(o->get_annotation_block(i));
            if (anno)
            {
                annotations_from_annotation_block(anno.get(), context);
            }
        }
    }
}

void annotations_from_function_definition(const IFunction_definition* o, Traversal_context & context)
{
    context.push_qualified_name(o->get_mdl_name());

    {
        mi::base::Handle<const IAnnotation_block> ablock(o->get_annotations());
        if (ablock)
        {
            annotations_from_annotation_block(ablock.get(), context);
        }
    }
    {
        mi::base::Handle<const IAnnotation_block> ablock(o->get_return_annotations());
        if (ablock)
        {
            annotations_from_annotation_block(ablock.get(), context);
        }
    }
    {
        mi::base::Handle<const IAnnotation_list> alist(o->get_parameter_annotations());
        if (alist)
        {
            annotations_from_annotation_list(alist.get(), context);
        }
    }
    //{
    //    for (mi::Size i = 0; i < o->get_parameter_count(); i++)
    //    {
    //        std::cout << o->get_parameter_name(i) << std::endl;
    //    }
    //}
}

void check_annotations(ITransaction* transaction)
{
    mi::base::Handle<const IModule> module(
        transaction->access<IModule>("mdl::test_mdl%24"));
    MI_CHECK(module);

    Check_annotation context;
    annotations_from_module(module.get(), context);

    std::string fname;
    for (mi::Size i = 0; i < module->get_function_count(); i++)
    {
        std::string name(module->get_function(i));
        mi::base::Handle<const IFunction_definition> o(transaction->access<IFunction_definition>(name.c_str()));
        MI_CHECK(o);
        annotations_from_function_definition(o.get(), context);
    }

    for (mi::Size i = 0; i < module->get_material_count(); i++)
    {
        std::string name(module->get_material(i));
        mi::base::Handle<const IFunction_definition> o(transaction->access<IFunction_definition>(name.c_str()));
        MI_CHECK(o);
        annotations_from_function_definition(o.get(), context);
    }
}

void run_tests(INeuray* neuray)
{
    check_imdl_i18n_configuration(neuray);

    MI_CHECK_EQUAL(0, neuray->start());

    {
        mi::base::Handle<IDatabase> database(
            neuray->get_api_component<IDatabase>());
        mi::base::Handle<IScope> global_scope(database->get_global_scope());
        mi::base::Handle<ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<IMdl_impexp_api> mdl_impexp_api(neuray->get_api_component<IMdl_impexp_api>());
        MI_CHECK(mdl_impexp_api);
        MI_CHECK_EQUAL( 0, mdl_impexp_api->load_module( transaction.get(), "::test_mdl%24"));

        check_annotations(transaction.get());

        MI_CHECK_EQUAL(0, transaction->commit());
    }

    MI_CHECK_EQUAL(0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_i18n )
{
    mi::base::Handle<INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        // set MDL paths
        mi::base::Handle<IMdl_configuration> mdl_configuration(neuray->get_api_component<IMdl_configuration>());
        MI_CHECK(mdl_configuration);
        std::string path = MI::TEST::mi_src_path( "prod/lib/neuray");
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));
        path = MI::TEST::mi_src_path("io/scene");
        MI_CHECK_EQUAL(0, mdl_configuration->add_mdl_path(path.c_str()));

        // load plugins
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        run_tests( neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get());
    }

    neuray = 0;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

