/******************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/neuraylib/annotation_wrapper.h>
#include <mi/neuraylib/ibsdf_measurement.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/ilightprofile.h>
#include <mi/neuraylib/imdl_archive_api.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/imdl_module_builder.h>
#include <mi/neuraylib/imdl_module_transformer.h>
#include <mi/neuraylib/imdle_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/itexture.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_shared.h"

// To avoid race conditions, each unit test uses a separate subdirectory for all the files it
// creates (possibly with further subdirectories).
#define DIR_PREFIX "output_test_imodule_creation"

#include "test_shared_mdl.h" // depends on DIR_PREFIX



mi::Sint32 result = 0;



// === Helper functions ============================================================================

mi::neuraylib::IAnnotation* create_string_annotation(
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    const char* value)
{
    mi::base::Handle<mi::neuraylib::IValue_string> arg_value( vf->create_string( value));
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_2_int_annotation(
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name0,
    const char* parameter_name1,
    mi::Sint32 value0,
    mi::Sint32 value1)
{
    mi::base::Handle<mi::neuraylib::IValue_int> arg_value0( vf->create_int( value0));
    mi::base::Handle<mi::neuraylib::IValue_int> arg_value1( vf->create_int( value1));
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr0(
        ef->create_constant( arg_value0.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr1(
        ef->create_constant( arg_value1.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name0, arg_expr0.get()));
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name1, arg_expr1.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_float2_annotation(
    mi::neuraylib::IType_factory* tf,
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    mi::Float32 value0,
    mi::Float32 value1)
{
    mi::base::Handle<const mi::neuraylib::IType_float> element_type( tf->create_float());
    mi::base::Handle<const mi::neuraylib::IType_vector> arg_type(
        tf->create_vector( element_type.get(), 2));
    mi::base::Handle<mi::neuraylib::IValue_vector> arg_value(
        vf->create_vector( arg_type.get()));
    mi::base::Handle<mi::neuraylib::IValue_float> arg_value0(
        arg_value->get_value<mi::neuraylib::IValue_float>( 0));
    arg_value0->set_value( value0);
    mi::base::Handle<mi::neuraylib::IValue_float> arg_value1(
        arg_value->get_value<mi::neuraylib::IValue_float>( 1));
    arg_value1->set_value( value1);

    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_Enum_annotation(
    mi::neuraylib::IType_factory* tf,
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    mi::Size index)
{
    mi::base::Handle<const mi::neuraylib::IType_enum> arg_type(
        tf->create_enum( "::" TEST_MDL "::Enum"));
    mi::base::Handle<mi::neuraylib::IValue_enum> arg_value(
        vf->create_enum( arg_type.get()));
    arg_value->set_index( index);
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));

    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

mi::neuraylib::IAnnotation* create_foo_struct_annotation(
    mi::neuraylib::IType_factory* tf,
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* annotation,
    const char* parameter_name,
    int value0,
    float value1)
{
    mi::base::Handle<const mi::neuraylib::IType_struct> arg_type(
        tf->create_struct( "::" TEST_MDL "::foo_struct"));
    mi::base::Handle<mi::neuraylib::IValue_struct> arg_value(
        vf->create_struct( arg_type.get()));
    mi::base::Handle<mi::neuraylib::IValue_int> arg_value0(
        arg_value->get_value<mi::neuraylib::IValue_int>( 0));
    arg_value0->set_value( value0);
    mi::base::Handle<mi::neuraylib::IValue_float> arg_value1(
        arg_value->get_value<mi::neuraylib::IValue_float>( 1));
    arg_value1->set_value( value1);
    mi::base::Handle<mi::neuraylib::IExpression_constant> arg_expr(
        ef->create_constant( arg_value.get()));

    mi::base::Handle<mi::neuraylib::IExpression_list> arg_args( ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arg_args->add_expression( parameter_name, arg_expr.get()));
    return ef->create_annotation( annotation, arg_args.get());
}

bool compare_files( const std::string& filename1, const std::string& filename2)
{
    std::ifstream file1( filename1);
    std::ifstream file2( filename2);

    if( !file1 || !file2)
        return false;

    std::string s1, s2;
    while( !file1.eof() && !file2.eof()) {
        std::getline( file1, s1);
        std::getline( file2, s2);
        if( s1 != s2)
            return false;
    }

    return ! (file1.eof() ^ file2.eof());
}

// Check that the ::anno::origin annotation on the main material/function of the MDLE file has the
// expected value.
void check_anno_origin_on_mdle(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    const char* mdle_path,
    const char* expected)
{
    import_mdl_module( transaction, neuray, mdle_path, 0);

    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<const mi::IString> module_db_name(
        mdl_factory->get_db_module_name( mdle_path));
    mi::base::Handle module(
        transaction->access<mi::neuraylib::IModule>( module_db_name->get_c_str()));

    mi::base::Handle<const mi::IArray> overloads( module->get_function_overloads( "main"));
    MI_CHECK_EQUAL( 1, overloads->get_length());
    mi::base::Handle<const mi::IString> overload(  overloads->get_element<mi::IString>( 0));
    mi::base::Handle material(
        transaction->access<mi::neuraylib::IFunction_definition>( overload->get_c_str()));
    MI_CHECK( material);

    mi::base::Handle<const mi::neuraylib::IAnnotation_block> anno_block(
        material->get_annotations());
    bool found = false;
    for( mi::Size i = 0, n = anno_block->get_size(); i < n; ++i) {
        mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
            anno_block->get_annotation( i));
        if( strcmp( anno->get_name(), "::anno::origin(string)") != 0)
            continue;
        found = true;
        mi::base::Handle<const mi::neuraylib::IExpression_list> args(
            anno->get_arguments());
        mi::base::Handle name(
            args->get_expression<mi::neuraylib::IExpression_constant>( "name"));
        mi::base::Handle name_value(
            name->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK_EQUAL_CSTR( name_value->get_value(), expected);
    }

    MI_CHECK( found);
}

// === Tests =======================================================================================

void check_module_builder_variants(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create module with multiple variants with color/enum arguments and annotations

    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            transaction,
            "mdl::variants",
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));

    {
        // create md_1_white with different annotations
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation(
            create_string_annotation(
                vf.get(), ef.get(), "::anno::description(string)", "description",
                "variant description annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation.get()));

        result = module_builder->add_variant(
            "md_1_white",
            "mdl::" TEST_MDL "::md_1(color)",
            /*defaults*/ nullptr,
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 1, c_module->get_material_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::md_1_white(color)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_md->get_defaults());

        // check the default for tint of "variants::md_1_white(color)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_tint_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
        MI_CHECK( c_tint_expr);
        mi::base::Handle<const mi::neuraylib::IValue_color> c_tint_value(
            c_tint_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<mi::neuraylib::IValue_color> m_tint_value(
            vf->create_color( 1.0, 1.0, 1.0));
        MI_CHECK_EQUAL( 0, vf->compare( c_tint_value.get(), m_tint_value.get()));

        // check the annotation of "variants::md_1_white(color)"
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_md->get_annotations());
        MI_CHECK( c_annotations);
        MI_CHECK_EQUAL( 1, c_annotations->get_size());
        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation(
            c_annotations->get_annotation( 0));
        MI_CHECK( c_annotation);
        MI_CHECK_EQUAL_CSTR( "::anno::description(string)", c_annotation->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_args(
            c_annotation->get_arguments());
        MI_CHECK( c_annotation_args);
        MI_CHECK_EQUAL( 1, c_annotation_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_arg(
            c_annotation_args->get_expression<mi::neuraylib::IExpression_constant>(
                "description"));
        MI_CHECK( c_annotation_arg);
        mi::base::Handle<const mi::neuraylib::IValue_string> c_annotation_value(
            c_annotation_arg->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK( c_annotation_value);
        MI_CHECK_EQUAL_CSTR( "variant description annotation", c_annotation_value->get_value());
    }

    // re-create module builder to test modifying modules already existing in the DB
    module_builder = mdl_factory->create_module_builder(
        transaction,
        "mdl::variants",
        mi::neuraylib::MDL_VERSION_1_0,
        mi::neuraylib::MDL_VERSION_LATEST,
        context.get());

    {
        // create md_enum_one with a new default for "param0" and new annotation block

        // create a "one" "param0" argument
        mi::base::Handle<const mi::neuraylib::IType_enum> param0_type(
            tf->create_enum( "::" TEST_MDL "::Enum"));
        mi::base::Handle<mi::neuraylib::IValue_enum> m_param0_value(
            vf->create_enum( param0_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
            ef->create_constant( m_param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "param0", m_param0_expr.get()));

        // create two ::anno::contributor annotations
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_a( create_string_annotation(
            vf.get(), ef.get(), "::anno::author(string)", "name", "variant author annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_b( create_string_annotation(
            vf.get(), ef.get(), "::anno::contributor(string)", "name",
            "variant contributor annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_a.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_b.get()));

        result = module_builder->add_variant(
            "md_enum_one",
            "mdl::" TEST_MDL "::md_enum(::" TEST_MDL "::Enum)",
            defaults.get(),
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 2, c_module->get_material_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::md_enum_one(::" TEST_MDL "::Enum)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_md->get_defaults());

        // check the default for param0 of "variants::md_enum_one(::" TEST_MDL "::Enum)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_param0_expr);
        mi::base::Handle<const mi::neuraylib::IValue_enum> c_param0_value(
            c_param0_expr->get_value<mi::neuraylib::IValue_enum>());
        MI_CHECK_EQUAL( 0, vf->compare( c_param0_value.get(), m_param0_value.get()));

        // check the annotations of "variants::md_enum_one(::" TEST_MDL "::Enum)"
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_md->get_annotations());
        MI_CHECK( c_annotations);
        MI_CHECK_EQUAL( 2, c_annotations->get_size());
        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_a(
            c_annotations->get_annotation( 0));
        MI_CHECK( c_annotation_a);
        MI_CHECK_EQUAL_CSTR( "::anno::author(string)", c_annotation_a->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_a_args(
            c_annotation_a->get_arguments());
        MI_CHECK( c_annotation_a_args);
        MI_CHECK_EQUAL( 1, c_annotation_a_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_a_arg(
            c_annotation_a_args->get_expression<mi::neuraylib::IExpression_constant>( "name"));
        MI_CHECK( c_annotation_a_arg);
        mi::base::Handle<const mi::neuraylib::IValue_string> c_annotation_a_value(
            c_annotation_a_arg->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK( c_annotation_a_value);
        MI_CHECK_EQUAL_CSTR( "variant author annotation", c_annotation_a_value->get_value());
        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_b(
            c_annotations->get_annotation( 1));
        MI_CHECK( c_annotation_b);
        MI_CHECK_EQUAL_CSTR( "::anno::contributor(string)", c_annotation_b->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_b_args(
            c_annotation_b->get_arguments());
        MI_CHECK( c_annotation_b_args);
        MI_CHECK_EQUAL( 1, c_annotation_b_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_b_arg(
            c_annotation_b_args->get_expression<mi::neuraylib::IExpression_constant>( "name"));
        MI_CHECK( c_annotation_b_arg);
        mi::base::Handle<const mi::neuraylib::IValue_string> c_annotation_b_value(
            c_annotation_b_arg->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK( c_annotation_b_value);
        MI_CHECK_EQUAL_CSTR( "variant contributor annotation", c_annotation_b_value->get_value());
    }
    {
        // create fd_1_43 with a new default for "param0" and new annotations

        // create a 43 "param0" argument
        mi::base::Handle<mi::neuraylib::IValue_int> m_variant2_param0_value( vf->create_int( 43));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_variant2_param0_expr(
            ef->create_constant( m_variant2_param0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults( ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "param0", m_variant2_param0_expr.get()));

        // create a ::" TEST_MDL "::anno_2_int, a ::" TEST_MDL "::anno_float2,
        // a ::" TEST_MDL "::anno_Enum, and a ::" TEST_MDL "::anno_foo_struct annotation
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_a( create_2_int_annotation(
            vf.get(), ef.get(), "::" TEST_MDL "::anno_2_int(int,int)", "max", "min", 20, 10));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_b( create_float2_annotation(
            tf.get(), vf.get(), ef.get(), "::" TEST_MDL "::anno_float2(float2)",
            "param0", 42.0f, 43.0f));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_c( create_Enum_annotation(
            tf.get(), vf.get(), ef.get(), "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)",
            "param0", 1));
        mi::base::Handle<mi::neuraylib::IAnnotation> m_annotation_d( create_foo_struct_annotation(
            tf.get(), vf.get(), ef.get(), "::" TEST_MDL "::anno_struct(::" TEST_MDL "::foo_struct)",
            "param0", 42, 43.0f));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_a.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_b.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_c.get()));
        MI_CHECK_EQUAL( 0, m_annotations->add_annotation( m_annotation_d.get()));

        {
            // check annotation wrapper
            mi::neuraylib::Annotation_wrapper anno_wrapper( m_annotations.get());
            MI_CHECK_EQUAL( 4, anno_wrapper.get_annotation_count());
            MI_CHECK_EQUAL_CSTR(
                "::" TEST_MDL "::anno_2_int(int,int)", anno_wrapper.get_annotation_name( 0));
            MI_CHECK_EQUAL( 2, anno_wrapper.get_annotation_param_count( 0));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_INT,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 0, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "max", anno_wrapper.get_annotation_param_name( 0, 0));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_INT,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 0, 1))->get_kind());
            MI_CHECK_EQUAL_CSTR( "min", anno_wrapper.get_annotation_param_name( 0, 1));

            mi::Sint32 test_value = 0;
            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value<mi::Sint32>(
                0, 0, test_value));
            MI_CHECK_EQUAL( 20, test_value);
            test_value = 0;
            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value_by_name<mi::Sint32>(
                "::" TEST_MDL "::anno_2_int(int,int)", 0, test_value ) );
            MI_CHECK_EQUAL( 20, test_value );

            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value<mi::Sint32>(
                0, 1, test_value));
            MI_CHECK_EQUAL( 10, test_value);
            test_value = 0;
            MI_CHECK_EQUAL( 0, anno_wrapper.get_annotation_param_value_by_name<mi::Sint32>(
                "::" TEST_MDL "::anno_2_int(int,int)", 1, test_value ) );
            MI_CHECK_EQUAL( 10, test_value );

            MI_CHECK_EQUAL_CSTR(
                "::" TEST_MDL "::anno_float2(float2)", anno_wrapper.get_annotation_name( 1));
            MI_CHECK_EQUAL( 1, anno_wrapper.get_annotation_param_count( 1));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_VECTOR,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 1, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "param0", anno_wrapper.get_annotation_param_name( 1, 0));

            mi::Size test_index = anno_wrapper.get_annotation_index(
                "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)" );
            MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)",
                anno_wrapper.get_annotation_name( static_cast<mi::Size>( test_index)));
            MI_CHECK_EQUAL( 1, anno_wrapper.get_annotation_param_count( 2));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_ENUM,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 2, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "param0", anno_wrapper.get_annotation_param_name( 2, 0));

            MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_struct(::" TEST_MDL "::foo_struct)",
                anno_wrapper.get_annotation_name( 3));
            MI_CHECK_EQUAL( 1, anno_wrapper.get_annotation_param_count( 3));
            MI_CHECK_EQUAL( mi::neuraylib::IType::Kind::TK_STRUCT,
                mi::base::Handle<const mi::neuraylib::IType>(
                    anno_wrapper.get_annotation_param_type( 3, 0))->get_kind());
            MI_CHECK_EQUAL_CSTR( "param0", anno_wrapper.get_annotation_param_name( 3, 0));

            // test behavior of invalid usage of the annotation wrapper (quite, no crash)
            mi::Float32 test_value_float = 0.0f;
            mi::neuraylib::Annotation_wrapper anno_wrapper_invalid( nullptr);
            MI_CHECK_EQUAL( 0, anno_wrapper_invalid.get_annotation_count());
            MI_CHECK_EQUAL( nullptr, anno_wrapper_invalid.get_annotation_name( 0));
            MI_CHECK_EQUAL( minus_one_size, anno_wrapper_invalid.get_annotation_index(
                "::anno::foo(int)"));
            MI_CHECK_EQUAL( 0, anno_wrapper_invalid.get_annotation_param_count( 0));
            MI_CHECK_EQUAL( nullptr, anno_wrapper_invalid.get_annotation_param_name( 0, 0));
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IType>(
                anno_wrapper_invalid.get_annotation_param_type( 0, 0)).get());
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IValue>(
                anno_wrapper_invalid.get_annotation_param_value( 0, 0)).get());
            MI_CHECK_EQUAL( -3, anno_wrapper_invalid.get_annotation_param_value<mi::Float32>(
                0, 0, test_value_float));

            MI_CHECK_EQUAL( nullptr, anno_wrapper.get_annotation_param_name( 0, 2));
            MI_CHECK_EQUAL( minus_one_size, anno_wrapper.get_annotation_index(
                "::anno::foo(int)"));
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IType>(
                anno_wrapper.get_annotation_param_type( 0, 2)).get());
            MI_CHECK_EQUAL( nullptr, mi::base::Handle<const mi::neuraylib::IValue>(
                anno_wrapper.get_annotation_param_value( 0, 2)).get());
            MI_CHECK_EQUAL( -3, anno_wrapper.get_annotation_param_value<mi::Float32>(
                0, 2, test_value_float));
            MI_CHECK_EQUAL( -1, anno_wrapper.get_annotation_param_value<mi::Float32>(
                0, 1, test_value_float ) );
            MI_CHECK_EQUAL( -3, anno_wrapper.get_annotation_param_value_by_name<mi::Float32>(
                "::anno::foo(int)", 1, test_value_float ) );
            MI_CHECK_EQUAL( -1, anno_wrapper.get_annotation_param_value_by_name<mi::Float32>(
                "::" TEST_MDL "::anno_2_int(int,int)", 1, test_value_float ) );
        }

        result = module_builder->add_variant(
            "fd_1_43",
            "mdl::" TEST_MDL "::fd_1(int)",
            defaults.get(),
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 1, c_module->get_function_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::fd_1_43(int)"));
        MI_CHECK( c_fd);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_fd->get_defaults());

        // check the default for param0 of "variants::fd_1_43"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_param0_expr);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_param0_value(
            c_param0_expr->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK_EQUAL( 0, vf->compare( c_param0_value.get(), m_variant2_param0_value.get()));

        // check the annotations of "variants::fd_1_43"
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_fd->get_annotations());
        MI_CHECK( c_annotations);
        MI_CHECK_EQUAL( 4, c_annotations->get_size());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_a(
            c_annotations->get_annotation( 0));
        MI_CHECK( c_annotation_a);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_2_int(int,int)", c_annotation_a->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_a_args(
            c_annotation_a->get_arguments());
        MI_CHECK( c_annotation_a_args);
        MI_CHECK_EQUAL( 2, c_annotation_a_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_a_arg(
            c_annotation_a_args->get_expression<mi::neuraylib::IExpression_constant>( "min"));
        MI_CHECK( c_annotation_a_arg);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_annotation_a_value(
            c_annotation_a_arg->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK( c_annotation_a_value);
        MI_CHECK_EQUAL( 10, c_annotation_a_value->get_value());
        c_annotation_a_arg
            = c_annotation_a_args->get_expression<mi::neuraylib::IExpression_constant>( "max");
        MI_CHECK( c_annotation_a_arg);
        c_annotation_a_value
            = c_annotation_a_arg->get_value<mi::neuraylib::IValue_int>();
        MI_CHECK( c_annotation_a_value);
        MI_CHECK_EQUAL( 20, c_annotation_a_value->get_value());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_b(
            c_annotations->get_annotation( 1));
        MI_CHECK( c_annotation_b);
        MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::anno_float2(float2)", c_annotation_b->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_b_args(
            c_annotation_b->get_arguments());
        MI_CHECK( c_annotation_b_args);
        MI_CHECK_EQUAL( 1, c_annotation_b_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_b_arg(
            c_annotation_b_args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_annotation_b_arg);
        mi::base::Handle<const mi::neuraylib::IValue_vector> c_annotation_b_value(
            c_annotation_b_arg->get_value<mi::neuraylib::IValue_vector>());
        MI_CHECK( c_annotation_b_value);
        mi::base::Handle<const mi::neuraylib::IValue_float> c_annotation_b_value_element(
            c_annotation_b_value->get_value<mi::neuraylib::IValue_float>( 0));
        MI_CHECK_EQUAL( 42.0f, c_annotation_b_value_element->get_value());
        c_annotation_b_value_element
            = c_annotation_b_value->get_value<mi::neuraylib::IValue_float>( 1);
        MI_CHECK_EQUAL( 43.0f, c_annotation_b_value_element->get_value());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_c(
            c_annotations->get_annotation( 2));
        MI_CHECK( c_annotation_c);
        MI_CHECK_EQUAL_CSTR(
            "::" TEST_MDL "::anno_enum(::" TEST_MDL "::Enum)", c_annotation_c->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_c_args(
            c_annotation_c->get_arguments());
        MI_CHECK( c_annotation_c_args);
        MI_CHECK_EQUAL( 1, c_annotation_c_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_c_arg(
            c_annotation_c_args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_annotation_c_arg);
        mi::base::Handle<const mi::neuraylib::IValue_enum> c_annotation_c_value(
            c_annotation_c_arg->get_value<mi::neuraylib::IValue_enum>());
        MI_CHECK( c_annotation_c_value);
        MI_CHECK_EQUAL( 1, c_annotation_c_value->get_index());

        mi::base::Handle<const mi::neuraylib::IAnnotation> c_annotation_d(
            c_annotations->get_annotation( 3));
        MI_CHECK( c_annotation_d);
        MI_CHECK_EQUAL_CSTR(
            "::" TEST_MDL "::anno_struct(::" TEST_MDL "::foo_struct)", c_annotation_d->get_name());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_annotation_d_args(
            c_annotation_d->get_arguments());
        MI_CHECK( c_annotation_d_args);
        MI_CHECK_EQUAL( 1, c_annotation_d_args->get_size());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_annotation_d_arg(
            c_annotation_d_args->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_annotation_d_arg);
        mi::base::Handle<const mi::neuraylib::IValue_struct> c_annotation_d_value(
            c_annotation_d_arg->get_value<mi::neuraylib::IValue_struct>());
        MI_CHECK( c_annotation_d_value);
        mi::base::Handle<const mi::neuraylib::IValue_int> c_annotation_d_value0(
            c_annotation_d_value->get_value<mi::neuraylib::IValue_int>( 0));
        MI_CHECK_EQUAL( 42, c_annotation_d_value0->get_value());
        mi::base::Handle<const mi::neuraylib::IValue_float> c_annotation_d_value1(
            c_annotation_d_value->get_value<mi::neuraylib::IValue_float>( 1));
        MI_CHECK_EQUAL( 43.0f, c_annotation_d_value1->get_value());
    }
    {
        // create fd_1_42_plus_42 with a new default for "param0"

        // create a 42+42 "param0" argument
        mi::base::Handle<mi::neuraylib::IValue_int> m_variant_lhs_value( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_variant_lhs_expr(
            ef->create_constant( m_variant_lhs_value.get()));
        mi::base::Handle<mi::neuraylib::IValue_int> m_variant_rhs_value( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> m_variant_rhs_expr(
            ef->create_constant( m_variant_rhs_value.get()));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd_variant_plus(
            transaction->access<mi::neuraylib::IFunction_definition>( "mdl::operator+(int,int)"));
        const char* lhs_name = fd_variant_plus->get_parameter_name( static_cast<mi::Size>( 0));
        const char* rhs_name = fd_variant_plus->get_parameter_name( 1);
        mi::base::Handle<mi::neuraylib::IExpression_list> m_variant_plus_args(
            ef->create_expression_list());
        MI_CHECK_EQUAL(
            0, m_variant_plus_args->add_expression( lhs_name, m_variant_lhs_expr.get()));
        MI_CHECK_EQUAL(
            0, m_variant_plus_args->add_expression( rhs_name, m_variant_lhs_expr.get()));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc_variant_plus(
            fd_variant_plus->create_function_call( m_variant_plus_args.get(), &result));
        MI_CHECK_EQUAL( 0, result);
        MI_CHECK_EQUAL( 0, transaction->store( fc_variant_plus.get(), "variant_plus"));
        mi::base::Handle<mi::neuraylib::IExpression_call> m_variant_param0_expr(
            ef->create_call( "variant_plus"));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults( ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "param0", m_variant_param0_expr.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation_block> m_annotations(
            ef->create_annotation_block());

        result = module_builder->add_variant(
            "fd_1_42_plus_42",
            "mdl::" TEST_MDL "::fd_1(int)",
            defaults.get(),
            m_annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::variants"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 2, c_module->get_function_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::variants::fd_1_42_plus_42(int)"));
        MI_CHECK( c_fd);
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_fd->get_defaults());

        // check the default for param0 of "variants::fd_1_42_plus_42" (constant folding)
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
        MI_CHECK( c_param0_expr);

        mi::base::Handle<const mi::neuraylib::IValue_int> c_param0_value(
            c_param0_expr->get_value<mi::neuraylib::IValue_int>());
        MI_CHECK_EQUAL( 84, c_param0_value->get_value());

        // check that the annotation of "" TEST_MDL "::fd_1" is not copied
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> c_annotations(
            c_fd->get_annotations());
        MI_CHECK( !c_annotations);
    }
}

void check_module_builder_resources(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            transaction,
            "mdl::resources",
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));

    std::string prototype_name = "mdl::" TEST_MDL "::md_resource_sharing(texture_2d,texture_2d)";

    {
        // create md_resource_sharing_new_defaults with a new default for "tex0"
        // (using a resource with MDL file path works)

        // create a "tex0" argument
        mi::base::Handle<mi::neuraylib::IValue> tex0_value(
            mdl_factory->create_texture(
                transaction,
                "/mdl_elements/resources/test.png",
                mi::neuraylib::IType_texture::TS_2D,
                2.2f,
                /*selector*/ nullptr,
                /*shared*/ false,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( tex0_value);
        mi::base::Handle<mi::neuraylib::IExpression_constant> tex0_expr(
            ef->create_constant( tex0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "tex0", tex0_expr.get()));

        result = module_builder->add_variant(
            "md_resource_sharing_new_defaults",
            prototype_name.c_str(),
            defaults.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // create md_resource_sharing_new_defaults2 with a new default for "tex0"
        // (using an invalid resource works)

        // create a "tex0" argument
        mi::base::Handle<const mi::neuraylib::IType_texture> tex0_type(
            tf->create_texture( mi::neuraylib::IType_texture::TS_2D));
        mi::base::Handle<mi::neuraylib::IValue_texture> tex0_value(
            vf->create_texture( tex0_type.get(), nullptr));
        MI_CHECK( tex0_value);
        mi::base::Handle<mi::neuraylib::IExpression_constant> tex0_expr(
            ef->create_constant( tex0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "tex0", tex0_expr.get()));

        result = module_builder->add_variant(
            "md_resource_sharing_new_defaults2",
            prototype_name.c_str(),
            defaults.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // create md_resource_sharing_new_defaults3 with a new default for "tex0"
        // (using a resource without MDL file path, but filename inside the search paths work also)

        // create dummy image/texture
        mi::base::Handle<mi::neuraylib::IImage> image(
            transaction->create<mi::neuraylib::IImage>( "Image"));
        result = transaction->store( image.get(), "dummy_image");
        MI_CHECK_EQUAL( result, 0);
        mi::base::Handle<mi::neuraylib::ITexture> texture(
            transaction->create<mi::neuraylib::ITexture>( "Texture"));
        texture->set_image( "dummy_image");
        result = transaction->store( texture.get(), "dummy_texture");
        MI_CHECK_EQUAL( result, 0);

        // create a "tex0" argument
        mi::base::Handle<const mi::neuraylib::IType_texture> tex0_type(
            tf->create_texture( mi::neuraylib::IType_texture::TS_2D));
        mi::base::Handle<mi::neuraylib::IValue_texture> tex0_value(
            vf->create_texture( tex0_type.get(), "dummy_texture"));
        MI_CHECK( tex0_value);
        mi::base::Handle<mi::neuraylib::IExpression_constant> tex0_expr(
            ef->create_constant( tex0_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        MI_CHECK_EQUAL( 0, defaults->add_expression( "tex0", tex0_expr.get()));

        result = module_builder->add_variant(
            "md_resource_sharing_new_defaults3",
            prototype_name.c_str(),
            defaults.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        // remove again to support import/export later (does not work with memory-based image above)
        result = module_builder->remove_entity(
            "md_resource_sharing_new_defaults3",
            0,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }

    // Create module with multiple variants with resource parameters (same defaults as before).
    //
    // As variants the AST representation will contain tags that are visible to the exporter. The
    // definitions from ::" TEST_MDL " use resources from files, the definitions from
    // ::test_archives use resources from an archive.

    const char* variants[] = {
        "md_texture",
        "md_light_profile",
        "md_bsdf_measurement",
        "fd_in_archive_texture_weak_relative",
        "fd_in_archive_texture_strict_relative",
        "fd_in_archive_texture_absolute",
    };

    const char* prototypes[] = {
        "mdl::" TEST_MDL "::md_texture(texture_2d)",
        "mdl::" TEST_MDL "::md_light_profile(light_profile)",
        "mdl::" TEST_MDL "::md_bsdf_measurement(bsdf_measurement)",
        "mdl::test_archives::fd_in_archive_texture_weak_relative(texture_2d)",
        "mdl::test_archives::fd_in_archive_texture_strict_relative(texture_2d)",
        "mdl::test_archives::fd_in_archive_texture_absolute(texture_2d)"
    };

    mi::Size n =       sizeof( prototypes) / sizeof( prototypes[0]);
    MI_CHECK_EQUAL( n, sizeof( variants  ) / sizeof( variants[0]  ));

    for( mi::Size i = 0; i < n; ++i) {
        result = module_builder->add_variant(
            variants[i],
            prototypes[i],
            /*defaults*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( "mdl::resources"));
    MI_CHECK( module);
    MI_CHECK_EQUAL( 5, module->get_import_count());
    MI_CHECK_EQUAL( 5, module->get_material_count());
    MI_CHECK_EQUAL( 3, module->get_function_count());
}

/// A client-side expression cloner to test that given expressions can actually be created by the
/// user.
///
/// Constant expressions are cloned via the expression factory as base case. Indirect calls are
/// not yet supported.
class Test_cloner
{
public:
    Test_cloner( mi::neuraylib::IExpression_factory* ef) : m_ef( ef) { }

    mi::neuraylib::IExpression* clone( const mi::neuraylib::IExpression* expr);

private:
    mi::neuraylib::IExpression_factory* m_ef;
};

mi::neuraylib::IExpression* Test_cloner::clone( const mi::neuraylib::IExpression* expr)
{
    switch( expr->get_kind()) {
        case mi::neuraylib::IExpression::EK_CONSTANT: {
            mi::base::Handle constant(
                expr->get_interface<mi::neuraylib::IExpression_constant>());
            return m_ef->clone( constant.get());
        }
        case mi::neuraylib::IExpression::EK_CALL: {
            MI_CHECK( false);
            return nullptr;
        }
        case mi::neuraylib::IExpression::EK_DIRECT_CALL: {
            mi::base::Handle direct_call(
                expr->get_interface<mi::neuraylib::IExpression_direct_call>());
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                direct_call->get_arguments());
            mi::base::Handle<mi::neuraylib::IExpression_list> cloned_args(
                m_ef->create_expression_list());
            for( mi::Size i = 0, n = args->get_size(); i < n; ++i) {
                mi::base::Handle<const mi::neuraylib::IExpression> arg( args->get_expression( i));
                mi::base::Handle<mi::neuraylib::IExpression> cloned_arg( clone( arg.get()));
                cloned_args->add_expression( args->get_name( i), cloned_arg.get());;
            }
            return m_ef->create_direct_call( direct_call->get_definition(), cloned_args.get());
        }
        case mi::neuraylib::IExpression::EK_PARAMETER: {
            mi::base::Handle parameter(
                expr->get_interface<mi::neuraylib::IExpression_parameter>());
            mi::base::Handle<const mi::neuraylib::IType> type( expr->get_type());
            return m_ef->create_parameter( type.get(), parameter->get_index());
        }
        case mi::neuraylib::IExpression::EK_TEMPORARY: {
            mi::base::Handle temporary(
                expr->get_interface<mi::neuraylib::IExpression_temporary>());
            mi::base::Handle<const mi::neuraylib::IType> type( expr->get_type());
            return m_ef->create_temporary( type.get(), temporary->get_index());
        }
        case mi::neuraylib::IExpression::EK_FORCE_32_BIT: {
            MI_CHECK( false);
            return nullptr;
        }
    }

    MI_CHECK( false);
    return nullptr;
}

void check_module_builder_misc(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create module "::new_materials" with material "md_wrap", based on
        // "::" TEST_MDL "::md_wrap(float,color,color)", with new parameters
        // "sqrt_x", "r", and "g".

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> sqrt_x_type( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType> r_type( tf->create_color());
        mi::base::Handle<const mi::neuraylib::IType> g_type( tf->create_color());
        g_type = tf->create_alias(
            g_type.get(), mi::neuraylib::IType::MK_UNIFORM, /*symbol*/ nullptr);
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "sqrt_x", sqrt_x_type.get());
        parameters->add_type( "r", r_type.get());
        parameters->add_type( "g", g_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> x_x_expr(
            ef->create_parameter( sqrt_x_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> x_args(
            ef->create_expression_list());
        x_args->add_expression( "x", x_x_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> x_direct_call(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_x(float)", x_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression> rhs_r_expr(
            ef->create_parameter( r_type.get(), 1));
        mi::base::Handle<mi::neuraylib::IExpression> rhs_g_expr(
            ef->create_parameter( g_type.get(), 2));
        mi::base::Handle<mi::neuraylib::IExpression_list> rhs_args(
            ef->create_expression_list());
        rhs_args->add_expression( "r", rhs_r_expr.get());
        rhs_args->add_expression( "g", rhs_g_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> rhs_direct_call(
            ef->create_direct_call(
                "mdl::" TEST_MDL "::fd_wrap_rhs(color,color,color)", rhs_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "x", x_direct_call.get());
        body_args->add_expression( "rhs", rhs_direct_call.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call(
                "mdl::" TEST_MDL "::md_wrap(float,color,color)", body_args.get()));

        // create defaults

        mi::base::Handle<mi::neuraylib::IValue> sqrt_x_value(
            vf->create_float( 0.70710678f));
        mi::base::Handle<mi::neuraylib::IExpression> sqrt_x_expr(
           ef->create_constant( sqrt_x_value.get()));

        mi::base::Handle<mi::neuraylib::IValue_color> c_value(
            vf->create_color( 0, 1, 1));
        mi::base::Handle<mi::neuraylib::IExpression> c_expr(
            ef->create_constant( c_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> r_args(
            ef->create_expression_list());
        r_args->add_expression( "c", c_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> r_expr(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_r(color)", r_args.get()));

        mi::base::Handle<mi::neuraylib::IValue_color> g_value(
            vf->create_color( 0, 1, 0));
        mi::base::Handle<mi::neuraylib::IExpression> g_expr(
            ef->create_constant( g_value.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        defaults->add_expression( "sqrt_x", sqrt_x_expr.get());
        defaults->add_expression( "r", r_expr.get());
        defaults->add_expression( "g", g_expr.get());

        // create parameter annotations

        mi::base::Handle<mi::neuraylib::IAnnotation> sqrt_x_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "sqrt(x.x)"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> sqrt_x_anno_block(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, sqrt_x_anno_block->add_annotation( sqrt_x_annotation.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> r_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "-rhs.r"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> r_anno_block(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, r_anno_block->add_annotation( r_annotation.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> g_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "rhs.g"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> g_anno_block(
            ef->create_annotation_block());
        MI_CHECK_EQUAL( 0, g_anno_block->add_annotation( g_annotation.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation_list> parameter_annotations(
            ef->create_annotation_list());
        parameter_annotations->add_annotation_block( "sqrt_x", sqrt_x_anno_block.get());
        parameter_annotations->add_annotation_block( "r", r_anno_block.get());
        parameter_annotations->add_annotation_block( "g", g_anno_block.get());

        // add the function

        result = module_builder->add_function(
            "md_wrap",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            defaults.get(),
            parameter_annotations.get(),
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // check created module

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::new_materials"));
        MI_CHECK( c_module);
        MI_CHECK_EQUAL( 4, c_module->get_import_count());
        MI_CHECK_EQUAL( 1, c_module->get_material_count());
        MI_CHECK_EQUAL( 0, c_module->get_function_count());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::new_materials::md_wrap(float,color,color)"));
        MI_CHECK( c_md);
        mi::base::Handle<const mi::neuraylib::IType_list> c_types( c_md->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_md->get_defaults());

        // check the default for sqrt_x of "new_materials::md_wrap(float,color,color)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_sqrt_x_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "sqrt_x"));
        MI_CHECK( c_sqrt_x_expr);
        mi::base::Handle<const mi::neuraylib::IValue_float> c_sqrt_x_value(
            c_sqrt_x_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_CLOSE( c_sqrt_x_value->get_value(), 0.7071068f, 1e-6);

        mi::base::Handle<const mi::neuraylib::IType> c_sqrt_x_type(
            c_types->get_type( "sqrt_x"));
        mi::Uint32 modifiers = c_sqrt_x_type->get_all_type_modifiers();
        MI_CHECK_EQUAL( modifiers & mi::neuraylib::IType::MK_UNIFORM, 0);

        // check the default for r of "new_materials::md_wrap(float,color,color)"
        mi::base::Handle<const mi::neuraylib::IExpression_call> c_r_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_call>( "r"));
        MI_CHECK( c_r_expr);
        mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
            transaction->access<mi::neuraylib::IFunction_call>( c_r_expr->get_call()));
        MI_CHECK_EQUAL_CSTR(
            c_fc->get_function_definition(), "mdl::" TEST_MDL "::fd_wrap_r(color)");

        mi::base::Handle<const mi::neuraylib::IType> c_r_expr_type(
            c_types->get_type( "r"));
        modifiers = c_r_expr_type->get_all_type_modifiers();
        MI_CHECK_EQUAL( modifiers & mi::neuraylib::IType::MK_UNIFORM, 0);

        mi::base::Handle<const mi::neuraylib::IExpression_list> c_fc_args( c_fc->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_fc_args_c_expr(
            c_fc_args->get_expression<mi::neuraylib::IExpression_constant>( "c"));
        mi::base::Handle<const mi::neuraylib::IValue_color> c_fc_args_c_value(
            c_fc_args_c_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<mi::neuraylib::IValue_color> m_fc_args_c_value(
            vf->create_color( 0, 1, 1));
        MI_CHECK_EQUAL( 0, vf->compare( c_fc_args_c_value.get(), m_fc_args_c_value.get()));

        // check the default for g of "new_materials::md_wrap(float,color,color)"
        mi::base::Handle<const mi::neuraylib::IExpression_constant> c_g_expr(
            c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "g"));
        MI_CHECK( c_g_expr);
        mi::base::Handle<const mi::neuraylib::IValue_color> c_g_value(
            c_g_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<mi::neuraylib::IValue_color> m_g_value(
            vf->create_color( 0, 1, 0));
        MI_CHECK_EQUAL( 0, vf->compare( c_g_value.get(), m_g_value.get()));

        mi::base::Handle<const mi::neuraylib::IType> c_g_type(
            c_types->get_type( "g"));
        MI_CHECK( (c_g_type->get_all_type_modifiers() & mi::neuraylib::IType::MK_UNIFORM) != 0);
    }
    {
        // create module "mdl::new_operators" with function "my_new_float_add", based on
        // "operator+(float,float)", and adds fd_wrap_x() of the new parameter "new_x" and
        // the constant 2.0

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_operators",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> new_x_type( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "new_x", new_x_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> x_x_expr(
            ef->create_parameter( new_x_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> x_args(
            ef->create_expression_list());
        x_args->add_expression( "x", x_x_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> x_direct_call(
            ef->create_direct_call( "mdl::" TEST_MDL "::fd_wrap_x(float)", x_args.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> y_value(
            vf->create_float( 2.0));
        mi::base::Handle<mi::neuraylib::IExpression> y_expr(
            ef->create_constant( y_value.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> op_plus_args(
            ef->create_expression_list());
        op_plus_args->add_expression( "x", x_direct_call.get());
        op_plus_args->add_expression( "y", y_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( "mdl::operator+(float,float)", op_plus_args.get()));

        // add the function

        result = module_builder->add_function(
            "my_new_float_add",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        // add the function, now with explicit declarative keyword

        result = module_builder->add_function(
            "my_new_float_add_declarative",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add material
        //
        // export material md_body_with_temporaries(float param0) { ... }
        //
        // using the body and temporaries(!) from an existing definition

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_named_temporaries(float)"));

        // create body, temporaries, and parameters

        mi::base::Handle<const mi::neuraylib::IExpression> body( fd->get_body());
        // Check that the body is an expression (not a statement).
        MI_CHECK( body);
        mi::base::Handle<const mi::neuraylib::IType_list> parameters( fd->get_parameter_types());
        mi::base::Handle<mi::neuraylib::IExpression_list> temporaries(
            ef->create_expression_list());

        mi::Size n = fd->get_temporary_count();
        // Check that this material actually uses temporaries.
        MI_CHECK( n >= 11);
        Test_cloner cloner( ef.get());
        for( mi::Size i = 0; i < n; ++i) {
            mi::base::Handle<const mi::neuraylib::IExpression> expr( fd->get_temporary( i));
            std::string name = "unused but unique name " + std::to_string( i);;
            mi::base::Handle<mi::neuraylib::IExpression> expr_clone(
                cloner.clone( expr.get()));
            temporaries->add_expression( name.c_str(), expr_clone.get());
        }

        // add the function

        result = module_builder->add_function(
            "md_body_with_temporaries",
            body.get(),
            temporaries.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // check created module

        // Check that the new material has at least as much temporaries as the original one.
        mi::base::Handle<const mi::neuraylib::IFunction_definition> orig_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_named_temporaries(float)"));
        mi::base::Handle<const mi::neuraylib::IFunction_definition> new_fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::new_materials::md_body_with_temporaries(float)"));
        MI_CHECK( new_fd);
        mi::Size orig_n = orig_fd->get_temporary_count();
        mi::Size new_n = new_fd->get_temporary_count();
        MI_CHECK( new_n >= orig_n);
    }
    {
        // edit module "mdl::new_materials" and add function
        //
        // export float fd_pass_parameter(float x) varying { return x; }
        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> x_type( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "x", x_type.get());

        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_parameter( x_type.get(), 0));

        result = module_builder->add_function(
            "fd_pass_parameter",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add material
        //
        // export material md_pass_parameter(material x) = x;
        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> x_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "x", x_type.get());

        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_parameter( x_type.get(), 0));

        result = module_builder->add_function(
            "md_pass_parameter",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add function
        //
        // export float fd_return_constant() uniform { return 42.f; }

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "fd_return_constant",
            body.get(),
            /*temporaries*/ nullptr,
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_UNIFORM,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add material
        //
        // export material md_return_constant() { return material(); }

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType_struct> body_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_struct( body_type.get()));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "md_return_constant",
            body.get(),
            /*temporaries*/ nullptr,
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // Edit module "mdl::new_materials" and add AGAIN material
        //
        // export material md_return_constant() { return material(); }
        //
        // This will fail in analyze_module(). Test that the module builder instance recovers from
        // this error and add annotation
        //
        // export annotation ad_dummy(int x);

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType_struct> body_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_struct( body_type.get()));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "md_return_constant",
            body.get(),
            /*temporaries*/ nullptr,
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_NOT_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IType> x_type( tf->create_string());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "x", x_type.get());

        result = module_builder->add_annotation(
            "ad_dummy",
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add material
        //
        // export ::test_mdl%24::aov_material md_new_aov_material() { return md_0(); }
        //
        // Note that we need to create a call to the decl_cast operator, which does \em not show
        // up in the AST/MDL source code.

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IExpression> call(
            ef->create_direct_call( "mdl::" TEST_MDL "::md_0()", nullptr));
        MI_CHECK( call);

        mi::base::Handle<const mi::neuraylib::IType_struct> body_type(
            tf->create_struct( "::" TEST_MDL "::aov_material"));
        mi::base::Handle<mi::neuraylib::IExpression> body( ef->create_decl_cast(
            call.get(),
            body_type.get(),
            /*cast_db_name*/ nullptr,
            /*force_cast*/ false,
            /*direct_call*/ true));
        MI_CHECK( body);

        result = module_builder->add_function(
            "md_new_aov_material",
            body.get(),
            /*temporaries*/ nullptr,
            /*parameters*/ nullptr,
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add struct category
        //
        // export struct_category Struct_category [[ description("struct category annotation") ]];

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> sc_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description",
            "struct category annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> annotations(
            ef->create_annotation_block());
        annotations->add_annotation( sc_annotation.get());

        result = module_builder->add_struct_category(
            "Struct_category",
            annotations.get(),
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        // check the struct category in the created module

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::new_materials"));
        mi::base::Handle<const mi::neuraylib::IStruct_category_list> sc_list(
            c_module->get_struct_categories());
        MI_CHECK_EQUAL( 1, sc_list->get_size());
        mi::base::Handle<const mi::neuraylib::IStruct_category> sc(
            sc_list->get_struct_category( zero_size));
        MI_CHECK( sc);
        MI_CHECK_EQUAL_CSTR( sc->get_symbol(), "::new_materials::Struct_category");
    }
    {
        // edit module "mdl::new_materials" and add enum
        //
        // export enum Enum [[ description("enum annotation") ]]
        // {
        //     one = 1 [[ description("one annotation") ]],
        //     two = 2 [[ ad_dummy("test_ad_dummy") ]]
        // };

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IValue> one_value( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression> one_expr(
            ef->create_constant( one_value.get()));
        mi::base::Handle<mi::neuraylib::IValue> two_value( vf->create_int( 2));
        mi::base::Handle<mi::neuraylib::IExpression> two_expr(
            ef->create_constant( two_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> enumerators(
            ef->create_expression_list());
        enumerators->add_expression( "one", one_expr.get());
        enumerators->add_expression( "two", two_expr.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> one_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "one annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> one_block(
            ef->create_annotation_block());
        one_block->add_annotation( one_annotation.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> two_annotation( create_string_annotation(
            vf.get(), ef.get(), "::new_materials::ad_dummy(string)", "x", "test ad_dummy"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> two_block(
            ef->create_annotation_block());
        two_block->add_annotation( two_annotation.get());

        mi::base::Handle<mi::neuraylib::IAnnotation_list> enumerator_annotations(
            ef->create_annotation_list());
        enumerator_annotations->add_annotation_block( "one", one_block.get());
        enumerator_annotations->add_annotation_block( "two", two_block.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> enum_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "enum annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> annotations(
            ef->create_annotation_block());
        annotations->add_annotation( enum_annotation.get());

        result = module_builder->add_enum_type(
            "Enum",
            enumerators.get(),
            enumerator_annotations.get(),
            annotations.get(),
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        // check the enum type in the created module

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::new_materials"));
        mi::base::Handle<const mi::neuraylib::IType_list> type_list(
            c_module->get_types());
        MI_CHECK_EQUAL( 1, type_list->get_size());
        mi::base::Handle<const mi::neuraylib::IType_enum> type_enum(
            type_list->get_type<mi::neuraylib::IType_enum>( "::new_materials::Enum"));
        MI_CHECK( type_enum);
    }
    {
        // edit module "mdl::new_materials" and add struct
        //
        // export declarative struct foo_struct in Struct_category
        //         [[ description("struct annotation") ]] {
        //     int param_int [[ description("param_int annotation") ]];
        //     float param_float = 0.0;
        // };

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> param_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType> param_float( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> fields( tf->create_type_list());
        fields->add_type( "param_int", param_int.get());
        fields->add_type( "param_float", param_float.get());

        mi::base::Handle<mi::neuraylib::IValue> param_float_default_value( vf->create_float( 0.0f));
        mi::base::Handle<mi::neuraylib::IExpression> param_float_default_expr(
            ef->create_constant( param_float_default_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> field_defaults(
            ef->create_expression_list());
        field_defaults->add_expression( "param_float", param_float_default_expr.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> param_int_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description",
            "param_int annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> param_int_block(
            ef->create_annotation_block());
        param_int_block->add_annotation( param_int_annotation.get());
        mi::base::Handle<mi::neuraylib::IAnnotation_list> field_annotations(
            ef->create_annotation_list());
        field_annotations->add_annotation_block( "param_int", param_int_block.get());

        mi::base::Handle<mi::neuraylib::IAnnotation> struct_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "struct annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> annotations(
            ef->create_annotation_block());
        annotations->add_annotation( struct_annotation.get());

        mi::base::Handle<const mi::neuraylib::IStruct_category> sc(
            tf->create_struct_category( "::new_materials::Struct_category"));
        MI_CHECK( sc);

        result = module_builder->add_struct_type(
            "foo_struct",
            fields.get(),
            field_defaults.get(),
            field_annotations.get(),
            annotations.get(),
            /*is_exported*/ true,
            /*is_declarative*/ true,
            sc.get(),
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        // check the struct type in the created module

        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( "mdl::new_materials"));
        mi::base::Handle<const mi::neuraylib::IType_list> type_list(
            c_module->get_types());
        MI_CHECK_EQUAL( 2, type_list->get_size());
        mi::base::Handle<const mi::neuraylib::IType_struct> type_struct(
            type_list->get_type<mi::neuraylib::IType_struct>( "::new_materials::foo_struct"));
        MI_CHECK( type_struct);
        mi::base::Handle<const mi::neuraylib::IStruct_category> sc2(
            type_struct->get_struct_category());
        MI_CHECK( sc2);
        MI_CHECK_EQUAL_CSTR( sc2->get_symbol(), "::new_materials::Struct_category");
        MI_CHECK( type_struct->is_declarative());
    }
    {
        // edit module "mdl::new_materials" and add constant (invalid name!)
        //
        // export float roughly two pi = 2 * 3.14;
        //
        // This will fail. Test that the module builder instance recovers from
        // this error and add constant
        //
        // export float roughly_two_pi = 2 * 3.14;

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IValue> two( vf->create_float( 2.0f));
        mi::base::Handle<mi::neuraylib::IExpression> two_expr( ef->create_constant( two.get()));
        mi::base::Handle<mi::neuraylib::IValue> pi( vf->create_float( 3.14f));
        mi::base::Handle<mi::neuraylib::IExpression> pi_expr( ef->create_constant( pi.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "x", two_expr.get());
        args->add_expression( "y", pi_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            ef->create_direct_call( "mdl::operator*(float,float)", args.get()));

        result = module_builder->add_constant(
            "roughly two pi",
            expr.get(),
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_NOT_EQUAL( 0, result);

        result = module_builder->add_constant(
            "roughly_two_pi",
            expr.get(),
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and set module annotations

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<mi::neuraylib::IAnnotation> module_annotation( create_string_annotation(
            vf.get(), ef.get(), "::anno::description(string)", "description", "module annotation"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> module_annotations(
            ef->create_annotation_block());
        module_annotations->add_annotation( module_annotation.get());

        result = module_builder->set_module_annotations(
            module_annotations.get(),
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // edit module "mdl::new_materials" and add an overload of "fd_return_constant" with dummy
        // parameter; then remove "Enum", "foo_struct", the 2nd overload of "fd_return_constant",
        // and "roughly_two_pi", clear module annotations

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        mi::base::Handle<const mi::neuraylib::IType> parameter0( tf->create_int());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "parameter0", parameter0.get());

        mi::base::Handle<mi::neuraylib::IValue> body_value( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression> body(
            ef->create_constant( body_value.get()));

        result = module_builder->add_function(
            "fd_return_constant",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_UNIFORM,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "Enum", 0, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "foo_struct", 0, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "Struct_category", 0, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "ad_dummy", 0, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "fd_return_constant", 1, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->remove_entity(
            "roughly_two_pi", 0, context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);

        result = module_builder->set_module_annotations(
            /*module_annotations*/ nullptr,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
#if 0
        // Disabled to avoid ending up with an empty module for import/export tests.
        result = module_builder->clear_module( context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
#endif
    }
}

void check_module_builder_removed(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create module "mdl::removed_features" with materials "new_mat_0" and "new_mat_1" with
        // calls to "state::rounded_corner_normal$1.2(float,bool)" and
        // "df::fresnel_layer$1.3(color,float,bsdf,bsdf,float3)" in their body

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // new_mat_0
        {
            // create parameters

            mi::base::Handle<const mi::neuraylib::IType> across_materials_type( tf->create_bool());
            mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
            parameters->add_type( "across_materials", across_materials_type.get());

            // create body

            mi::base::Handle<mi::neuraylib::IExpression> n_across_materials_expr(
                ef->create_parameter( across_materials_type.get(), 0));
            mi::base::Handle<mi::neuraylib::IExpression_list> rcn_args(
                ef->create_expression_list());
            rcn_args->add_expression( "across_materials", n_across_materials_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> rcn_direct_call(
                ef->create_direct_call( "mdl::state::rounded_corner_normal$1.2(float,bool)",
                    rcn_args.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
                ef->create_expression_list());
            body_args->add_expression( "n", rcn_direct_call.get());
            mi::base::Handle<const mi::neuraylib::IExpression> body(
                ef->create_direct_call(
                    "mdl::" TEST_MDL "::md_float3_arg(float3)", body_args.get()));

            // run uniform analysis

            mi::base::Handle<const mi::neuraylib::IType> body_type( body->get_type());
            mi::base::Handle<const mi::IArray> uniform( module_builder->analyze_uniform(
                body.get(), mi::neuraylib::IType::MK_NONE, context.get()));
            MI_CHECK_CTX( context);
            MI_CHECK( uniform);

            // adapt parameter types based on uniform analysis

            mi::base::Handle<mi::neuraylib::IType_list> fixed_parameters( tf->create_type_list());
            for( mi::Size i = 0, n = uniform->get_length(); i < n; ++i) {
                mi::base::Handle<const mi::neuraylib::IType> parameter( parameters->get_type( i));
                const char* name = parameters->get_name( i);
                mi::base::Handle<const mi::IBoolean> element(
                    uniform->get_element<mi::IBoolean>( i));
                if( element->get_value<bool>()) {
                    MI_CHECK( i == 0);
                    parameter = tf->create_alias(
                        parameter.get(), mi::neuraylib::IType::MK_UNIFORM, /*symbol*/ nullptr);
                } else {
                    MI_CHECK( i != 0);
                }
                fixed_parameters->add_type( name, parameter.get());
            }
            parameters = fixed_parameters;

            // add the material

            result = module_builder->add_function(
                "new_mat_0",
                body.get(),
                /*temporaries*/ nullptr,
                parameters.get(),
                /*defaults*/ nullptr,
                /*parameter_annotations*/ nullptr,
                /*annotations*/ nullptr,
                /*return_annotations*/ nullptr,
                /*is_exported*/ true,
                /*is_declarative*/ true,
                /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
                context.get());
            MI_CHECK_CTX( context);
            MI_CHECK_EQUAL( 0, result);
        }

        // new_mat_1
        {
            // create parameters

            mi::base::Handle<const mi::neuraylib::IType> ior_type( tf->create_color());
            mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
            parameters->add_type( "ior", ior_type.get());

            // create body

            mi::base::Handle<mi::neuraylib::IExpression> surface_scattering_ior_expr(
                ef->create_parameter( ior_type.get(), 0));
            mi::base::Handle<mi::neuraylib::IExpression_list> surface_scattering_args(
                ef->create_expression_list());
            surface_scattering_args->add_expression( "ior", surface_scattering_ior_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> surface_scattering_expr(
                ef->create_direct_call( "mdl::df::fresnel_layer$1.3(color,float,bsdf,bsdf,float3)",
                    surface_scattering_args.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> surface_args(
                ef->create_expression_list());
            surface_args->add_expression( "scattering", surface_scattering_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> surface_expr(
                ef->create_direct_call( "mdl::material_surface(bsdf,material_emission)",
                    surface_args.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
                ef->create_expression_list());
            body_args->add_expression( "surface", surface_expr.get());
            mi::base::Handle<const mi::neuraylib::IExpression> body(
                ef->create_direct_call(
                    "mdl::material(bool,material_surface,material_surface,color,material_volume,"
                        "material_geometry,hair_bsdf)",
                    body_args.get()));

            // create defaults

            mi::base::Handle<mi::neuraylib::IValue> ior_value(
                vf->create_color( 0, 0, 0));
            mi::base::Handle<mi::neuraylib::IExpression> ior_expr(
               ef->create_constant( ior_value.get()));

            mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
                ef->create_expression_list());
            defaults->add_expression( "ior", ior_expr.get());

            // add the material

            result = module_builder->add_function(
                "new_mat_1",
                body.get(),
                /*temporaries*/ nullptr,
                parameters.get(),
                defaults.get(),
                /*parameter_annotations*/ nullptr,
                /*annotations*/ nullptr,
                /*return_annotations*/ nullptr,
                /*is_exported*/ true,
                /*is_declarative*/ true,
                /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
                context.get());
            MI_CHECK_CTX( context);
            MI_CHECK_EQUAL( 0, result);
        }
    }
}

void check_module_builder_utf8(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // create module "mdl::123_check_unicode_new_materials" with a call in the body to
        // another module with utf8 name

        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::123_check_unicode_new_materials",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));

        // create parameters

        mi::base::Handle<const mi::neuraylib::IType> a_type( tf->create_float());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "a", a_type.get());

        // create body

        mi::base::Handle<mi::neuraylib::IExpression> a_expr(
            ef->create_parameter( a_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "a", a_expr.get());
        std::string body_def_name = std::string( "mdl::") + name_rus + "::1_module::fd_test(float)";
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( body_def_name.c_str(), body_args.get()));

        // add the function

        result = module_builder->add_function(
            "new_fd_test",
            body.get(),
            /*temporaries*/ nullptr,
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ false,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
    {
        // test variants
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
            mdl_factory->create_module_builder(
                transaction,
                "mdl::123_check_unicode_new_variants",
                mi::neuraylib::MDL_VERSION_1_0,
                mi::neuraylib::MDL_VERSION_LATEST,
                context.get()));
        std::string prototype_name
            = std::string( "mdl::") + name_rus + "::1_module::md_test(float)";
        result = module_builder->add_variant(
            "md_1_white",
            prototype_name.c_str(),
            /*defaults*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*is_declarative*/ true,
            context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( 0, result);
    }
}

void check_module_builder_implicit_casts(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Uses MDL >= 1.5 due to the cast operator
    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            transaction,
            "mdl::compatibility_new_material",
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));

    // create parameters

    mi::base::Handle<const mi::neuraylib::IType> param0_type( tf->create_bool());
    mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
    parameters->add_type( "param0", param0_type.get());

    // create body

    mi::base::Handle<mi::neuraylib::IExpression> v_v_expr(
        ef->create_parameter( param0_type.get(), 0));
    mi::base::Handle<mi::neuraylib::IExpression_list> v_args(
        ef->create_expression_list());
    v_args->add_expression( "v", v_v_expr.get());
    mi::base::Handle<const mi::neuraylib::IExpression> v_expr(
        ef->create_direct_call( "mdl::test_compatibility::struct_return(bool)", v_args.get()));

    mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
        ef->create_expression_list());
    body_args->add_expression( "v", v_expr.get());
    mi::base::Handle<const mi::neuraylib::IExpression> body(
        ef->create_direct_call(
            "mdl::test_compatibility::structure_test_mat(::test_compatibility::struct_one)",
            body_args.get()));

    // create defaults

    mi::base::Handle<mi::neuraylib::IValue> param0_value(
        vf->create_bool( false));
    mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
        ef->create_constant( param0_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
        ef->create_expression_list());
    defaults->add_expression( "param0", param0_expr.get());

    // add the material

    result = module_builder->add_function(
        "my_new_material",
        body.get(),
        /*temporaries*/ nullptr,
        parameters.get(),
        defaults.get(),
        /*parameter_annotations*/ nullptr,
        /*annotations*/ nullptr,
        /*return_annotations*/ nullptr,
        /*is_exported*/ true,
        /*is_declarative*/ true,
        /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
        context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( 0, result);
}

void check_module_transformer(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
// The copy command below fails on Windows if source and target are on a network drive and when
// executed from the command-line via CTest (not from Visual Studio).
#ifndef MI_PLATFORM_WINDOWS
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    // For strict relative imports and resource paths, all module and resources must be in the
    // same search paths. Copy the inputs such that the output can be exported into the same
    // directory.
    std::string path = MI::TEST::mi_src_path("prod/lib/neuray/mdl_module_transformer");
    fs::copy(
        fs::u8path( path),
        fs::u8path( DIR_PREFIX "/mdl_module_transformer"),
        fs::copy_options::recursive);

    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // We need to load the modules to be transformed without any optimization and with the original
    // resource paths. Therefore, we use a separate context for such calls.
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> load_context(
        mdl_factory->create_execution_context());
    load_context->set_option( "optimization_level", static_cast<mi::Sint32>( 0));
    MI_CHECK_CTX( load_context);
    load_context->set_option( "keep_original_resource_file_paths", true);
    MI_CHECK_CTX( load_context);

    uninstall_external_resolver( neuray);

    // check that builtin modules cannot be transformed
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::%3Cbuiltins%3E", context.get()));
        MI_CHECK( !module_transformer);
    }
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer( transaction, "mdl::state", context.get()));
        MI_CHECK( !module_transformer);
    }
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer( transaction, "mdl::stdlib", context.get()));
        MI_CHECK( !module_transformer);
    }
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer( transaction, "mdl::base", context.get()));
        MI_CHECK( !module_transformer);
    }

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer/import_declarations";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    mdl_impexp_api->load_module( transaction, "::p1::main_import_declarations", load_context.get());
    MI_CHECK_CTX( load_context);

    // check use_absolute_import_declarations() (without alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->use_absolute_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_absolute.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_absolute", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_absolute.orig.mdl",
            path + "/p1/exported_main_import_declarations_absolute.mdl"));
    }

    // check use_relative_import_declarations() (without alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->use_relative_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_relative.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_relative", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_relative.orig.mdl",
            path + "/p1/exported_main_import_declarations_relative.mdl"));
    }

    mdl_impexp_api->load_module(
        transaction, "::p1::main_import_declarations_alias", load_context.get());
    MI_CHECK_CTX( load_context);

    // check use_absolute_import_declarations() (with alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations_alias", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->use_absolute_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_alias_absolute.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_alias_absolute", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_alias_absolute.orig.mdl",
            path + "/p1/exported_main_import_declarations_alias_absolute.mdl"));
    }

    // check use_relative_import_declarations() (with alias declarations)
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations_alias", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->use_relative_import_declarations(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_alias_relative.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_alias_relative", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_alias_relative.orig.mdl",
            path + "/p1/exported_main_import_declarations_alias_relative.mdl"));
    }

    // check upgrade_mdl_version() w.r.t. alias removal
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_import_declarations_alias", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->upgrade_mdl_version(
            mi::neuraylib::MDL_VERSION_1_8, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_import_declarations_alias_alias_removal.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_import_declarations_alias_alias_removal",
            context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_import_declarations_alias_alias_removal.orig.mdl",
            path + "/p1/exported_main_import_declarations_alias_alias_removal.mdl"));
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer/resource_file_paths";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    mdl_impexp_api->load_module( transaction, "::p1::main_resource_file_paths", load_context.get());
    MI_CHECK_CTX( load_context);

    // check use_absolute_resource_file_paths()
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_resource_file_paths", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->use_absolute_resource_file_paths(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_resource_file_paths_absolute.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_resource_file_paths_absolute", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_resource_file_paths_absolute.orig.mdl",
            path + "/p1/exported_main_resource_file_paths_absolute.mdl"));
    }

    // check use_relative_resource_file_paths()
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_resource_file_paths", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->use_relative_resource_file_paths(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_resource_file_paths_relative.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_resource_file_paths_relative", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_resource_file_paths_relative.orig.mdl",
            path + "/p1/exported_main_resource_file_paths_relative.mdl"));
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer/upgrade_mdl_version";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    // check upgrade_mdl_version() w.r.t. weak relative import declarations and resource file paths
    {
        mdl_impexp_api->load_module(
            transaction, "::p1::main_upgrade_mdl_version", load_context.get());
        MI_CHECK_CTX( load_context);
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::p1::main_upgrade_mdl_version", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->upgrade_mdl_version(
            mi::neuraylib::MDL_VERSION_1_6, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/p1/exported_main_upgrade_mdl_version.mdl").c_str(), context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::p1::exported_main_upgrade_mdl_version", context.get());
        MI_CHECK_CTX( context);
        // Compare against expected output (otherwise a null transformation would also pass).
        MI_CHECK( compare_files(
            path + "/p1/exported_main_upgrade_mdl_version.orig.mdl",
            path + "/p1/exported_main_upgrade_mdl_version.mdl"));
    }

    // check upgrade_mdl_version() w.r.t. updated function signatures

    struct Data { const char* from; const char* to; mi::neuraylib::Mdl_version to_enum; };
    Data data[] = { { "1_0", "1_3",  mi::neuraylib::MDL_VERSION_1_3 },
                    { "1_0", "1_4",  mi::neuraylib::MDL_VERSION_1_4 },
                    { "1_0", "1_5",  mi::neuraylib::MDL_VERSION_1_5 },
                    { "1_0", "1_6",  mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_3", "1_4",  mi::neuraylib::MDL_VERSION_1_4 },
                    { "1_3", "1_5",  mi::neuraylib::MDL_VERSION_1_5 },
                    { "1_3", "1_6",  mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_4", "1_5",  mi::neuraylib::MDL_VERSION_1_5 },
                    { "1_4", "1_6",  mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_5", "1_6",  mi::neuraylib::MDL_VERSION_1_6 },
                    { "1_9", "1_10", mi::neuraylib::MDL_VERSION_1_10 }
                };

    for( const auto& d: data) {

        std::string input_mdl_name   = std::string( "::test_mdl_version_") + d.from;
        std::string input_db_name    = std::string( "mdl::test_mdl_version_") + d.from;
        std::string output_file_name = std::string( DIR_PREFIX)
                                           + "/mdl_module_transformer/upgrade_mdl_version"
                                           + "/exported_test_mdl_version_" + d.from
                                           + "_as_mdl_version_" + d.to + ".mdl";
        std::string output_mdl_name  = std::string( "::exported_test_mdl_version_") + d.from
                                           + "_as_mdl_version_" + d.to;

        mdl_impexp_api->load_module( transaction, input_mdl_name.c_str(), load_context.get());
        MI_CHECK_CTX( load_context);
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, input_db_name.c_str(), context.get()));
        MI_CHECK_CTX( context);
        module_transformer->upgrade_mdl_version( d.to_enum, context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module( output_file_name.c_str(), context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        MI_CHECK_CTX( context);
        mdl_impexp_api->load_module( transaction, output_mdl_name.c_str(), context.get());
        MI_CHECK_CTX( context);
        // No file comparison (too much code whose serialization in text form changes too often in
        // tiny details)
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));

    std::string core1_path = MI::TEST::mi_src_path( "shaders/mdl");
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( core1_path.c_str()));
    std::string core2_path = MI::TEST::mi_src_path( "../examples/mdl");
    mdl_configuration->add_mdl_path( core2_path.c_str());

    path = std::string( DIR_PREFIX) + "/mdl_module_transformer";
    MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( path.c_str()));

    mdl_impexp_api->load_module( transaction, "::nvidia::core_definitions", load_context.get());
    MI_CHECK_CTX( load_context);

    // check inlined_imported_modules()
    //
    // Run on ::nvidia::core_definitions (nothing really to inline, since it uses only stdlib and
    // base).
    {
        mi::base::Handle<mi::neuraylib::IMdl_module_transformer> module_transformer(
            mdl_factory->create_module_transformer(
                transaction, "mdl::nvidia::core_definitions", context.get()));
        MI_CHECK_CTX( context);
        module_transformer->inline_imported_modules(
            /*include_filter*/ nullptr, /*exclude_filter*/ nullptr, /*omit_anno_origin*/ false,
            context.get());
        MI_CHECK_CTX( context);
        module_transformer->export_module(
            (path + "/exported_inlined_core_definitions.mdl").c_str(),
            context.get());
        MI_CHECK_CTX( context);
        // Import again to check for correctness.
        mdl_impexp_api->load_module(
            transaction, "::exported_inlined_core_definitions", context.get());
        MI_CHECK_CTX( context);
        // No file comparison (too much code, formatting changes cause frequent test failures).
    }

    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( path.c_str()));
    MI_CHECK_EQUAL( 0, mdl_configuration->remove_mdl_path( core1_path.c_str()));
    mdl_configuration->remove_mdl_path( core2_path.c_str());

    install_external_resolver( neuray);
#endif // MI_PLATFORM_WINDOWS
}

void check_mdle(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(
        neuray->get_api_component<mi::neuraylib::IMdle_api>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    {
        // test MDLE creation with a simple material
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str( "mdl::" TEST_MDL "::md_1(color)");
        const char* filename = DIR_PREFIX "/md_1.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle( transaction, neuray, filename, "::test_mdl%24::md_1");
    }
    {
        // test MDLE creation with a simple material (explicit annotation via API)
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str( "mdl::" TEST_MDL "::md_1(color)");
        mi::base::Handle<mi::neuraylib::IAnnotation> anno( create_string_annotation(
            vf.get(), ef.get(), "::anno::origin(string)", "name", "explicit_origin"));
        mi::base::Handle<mi::neuraylib::IAnnotation_block> anno_block(
            ef->create_annotation_block());
        anno_block->add_annotation( anno.get());
        mdle_data->set_value( "annotations", anno_block.get());
        const char* filename = DIR_PREFIX "/md_1_explicit.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle( transaction, neuray, filename, "explicit_origin");
    }
    {
        // test MDLE creation with a simple material variant
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str( "mdl::" TEST_MDL "::md_1_green(color)");
        const char* filename = DIR_PREFIX "/md_1_green.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle( transaction, neuray, filename, "::test_mdl%24::md_1_green");
    }
    {
        // test MDLE creation with a simple material variant with existing origin annotation
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str( "mdl::" TEST_MDL "::md_1_blue(color)");
        const char* filename = DIR_PREFIX "/md_1_blue.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle( transaction, neuray, filename, "existing_origin");
    }

    // create struct constant struct_two_expr of type ...::struct_two
    mi::base::Handle<const mi::neuraylib::IType_struct> struct_two_type(
        tf->create_struct( "::test_compatibility::struct_two"));
    mi::base::Handle<mi::neuraylib::IValue_struct> struct_two_value(
        vf->create_struct( struct_two_type.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> struct_two_expr(
        ef->create_constant( struct_two_value.get()));

    {
        // test MDLE creation with a function with implicit cast
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str(
            "mdl::test_compatibility::structure_test(::test_compatibility::struct_one)");
        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        args->add_expression( "v", struct_two_expr.get());
        MI_CHECK_EQUAL( 0, mdle_data->set_value( "defaults", args.get()));
        const char* filename = DIR_PREFIX "/compatibility_func_test.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle(
            transaction, neuray, filename, "::test_compatibility::structure_test");
    }
    {
        // test MDLE creation with a material with implicit cast
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str(
            "mdl::test_compatibility::structure_test_mat(::test_compatibility::struct_one)");
        mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
        MI_CHECK_EQUAL( 0, args->add_expression( "v", struct_two_expr.get()));
        MI_CHECK_EQUAL( 0, mdle_data->set_value( "defaults", args.get()));
        const char* filename = DIR_PREFIX "/compatibility_mat_test.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle(
            transaction, neuray, filename, "::test_compatibility::structure_test_mat");
    }
    {
        // test MDLE creation with a material from a module created by the module builder

        // ::compatibility_new_material was not exported, requires the module cache
        uninstall_external_resolver( neuray);

        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str( "mdl::compatibility_new_material::my_new_material(bool)");
        const char* filename = DIR_PREFIX "/compatibility_new_material.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        check_anno_origin_on_mdle(
            transaction, neuray, filename, "::compatibility_new_material::my_new_material");

        install_external_resolver( neuray);
    }
    {
        // test MDLE creation with a material from a UTF8 module (created by the module builder)
        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        std::string prototype_str
            = std::string( "mdl::") + name_rus + "::1_module::md_test(float)";
        prototype_name->set_c_str( prototype_str.c_str());
        const char* filename = DIR_PREFIX "/123_check_unicode_create_mdle_with_unicode_€.mdle";

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        std::string expected = std::string( "::") + name_rus + "::1_module::md_test";
        check_anno_origin_on_mdle( transaction, neuray, filename, expected.c_str());
    }
    {
        // test MDLE creation with a material from an MDLE UTF8 module
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mdl_configuration->add_mdl_path( DIR_PREFIX);

        result = mdl_impexp_api->load_module(
            transaction,
            DIR_PREFIX "/123_check_unicode_create_mdle_with_unicode_€.mdle",
            context.get());
        MI_CHECK( result >= 0); // check_anno_origin_on_mdle() already loaded the module

        mi::base::Handle<const mi::IString> db_name( mdl_factory->get_db_module_name(
            DIR_PREFIX "/123_check_unicode_create_mdle_with_unicode_€.mdle"));
        std::string module_db_name = db_name->get_c_str();

        mi::base::Handle<mi::IStructure> mdle_data(
            transaction->create<mi::IStructure>( "Mdle_data"));
        mi::base::Handle<mi::IString> prototype_name(
            mdle_data->get_value<mi::IString>( "prototype_name"));
        prototype_name->set_c_str( (module_db_name + "::main(float)").c_str());

#ifndef MI_PLATFORM_WINDOWS
        const char *filename = DIR_PREFIX "/123_check_unicode_create_mdle.mdle";

        // TODO Reported not to work on Windows. Unclear why.
        result = mdle_api->export_mdle(
            transaction, filename, mdle_data.get(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 0);

        std::string expected = std::string( "::") + name_rus + "::1_module::md_test";
        check_anno_origin_on_mdle( transaction, neuray, filename, expected.c_str());
#endif

        mdl_configuration->remove_mdl_path( DIR_PREFIX);
    }
}

void check_create_archive( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> mdl_archive_api(
        neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());

    std::string directory = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/create");

    MI_CHECK_EQUAL_CSTR(
        ".ies,.mbsdf,.txt,.html", mdl_archive_api->get_extensions_for_compression());

    result =  mdl_archive_api->set_extensions_for_compression( nullptr);
    MI_CHECK_EQUAL( -1, result);

    result =  mdl_archive_api->set_extensions_for_compression( "");
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL_CSTR( "", mdl_archive_api->get_extensions_for_compression());

    result =  mdl_archive_api->set_extensions_for_compression( ".doc");
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL_CSTR( ".doc", mdl_archive_api->get_extensions_for_compression());

    uninstall_external_resolver( neuray);

    result = mdl_archive_api->create_archive(
        nullptr, DIR_PREFIX "/test_create_archives.mdr", nullptr);
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->create_archive( directory.c_str(), nullptr, nullptr);
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives_wrong_extension", nullptr);
    MI_CHECK_EQUAL( -2, result);

    mi::base::Handle<mi::IArray> wrong_array_type( transaction->create<mi::IArray>( "Sint32[42]"));
    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives.mdr", wrong_array_type.get());
    MI_CHECK_EQUAL( -3, result);

    wrong_array_type = transaction->create<mi::IArray>( "Interface[1]");
    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives.mdr", wrong_array_type.get());
    MI_CHECK_EQUAL( -3, result);

    result = mdl_archive_api->create_archive(
        directory.c_str(), DIR_PREFIX "/test_create_archives_wrong_package.mdr", nullptr);
    MI_CHECK_EQUAL( -4, result);

    mi::base::Handle<mi::IDynamic_array> fields(
        transaction->create<mi::IDynamic_array>( "Manifest_field[]"));
    fields->set_length( 2);

    mi::base::Handle<mi::IStructure> field0( fields->get_value<mi::IStructure>( zero_size));
    mi::base::Handle<mi::IString> key0( field0->get_value<mi::IString>( "key"));
    key0->set_c_str( "foo");
    mi::base::Handle<mi::IString> value0( field0->get_value<mi::IString>( "value"));
    value0->set_c_str( "bar");

    mi::base::Handle<mi::IStructure> field1( fields->get_value<mi::IStructure>( 1));
    mi::base::Handle<mi::IString> key1( field1->get_value<mi::IString>( "key"));
    key1->set_c_str( "foo");
    mi::base::Handle<mi::IString> value1( field1->get_value<mi::IString>( "value"));
    value1->set_c_str( "baz");

    std::string archive = std::string( DIR_PREFIX) + dir_sep + "test_create_archives.mdr";
    result = mdl_archive_api->create_archive(
        directory.c_str(), archive.c_str(), fields.get());
    MI_CHECK_EQUAL( 0, result);

    fs::remove( fs::u8path( archive));

    install_external_resolver( neuray);

#ifdef EXTERNAL_ENTITY_RESOLVER
    result = mdl_archive_api->create_archive( directory.c_str(), archive.c_str(), fields.get());
    MI_CHECK_EQUAL( -5, result);
#endif
}

void check_extract_archive( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> mdl_archive_api(
        neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());

    std::string archive       = MI::TEST::mi_src_path( "prod/lib/neuray/test_archives.mdr");
    std::string wrong_archive = MI::TEST::mi_src_path( "prod/lib/neuray/" TEST_MDL_FILE ".mdl");

    result = mdl_archive_api->extract_archive( nullptr, ".");
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->extract_archive( archive.c_str(), nullptr);
    MI_CHECK_EQUAL( -1, result);

    result = mdl_archive_api->extract_archive( wrong_archive.c_str(), ".");
    MI_CHECK_EQUAL( -2, result);

    result = mdl_archive_api->extract_archive( "non-existing.mdr", ".");
    MI_CHECK_EQUAL( -2, result);

    result = mdl_archive_api->extract_archive( archive.c_str(), DIR_PREFIX "/extracted");
    MI_CHECK_EQUAL( 0, result);

    result = mdl_archive_api->extract_archive(
        archive.c_str(), DIR_PREFIX "/extracted-non-existing");
    MI_CHECK_EQUAL( 0, result);
}

void check_archive_get_manifest( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> mdl_archive_api(
        neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());

    std::string archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/query/test_archives.mdr");
    std::string wrong_archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/" TEST_MDL_FILE ".mdl");

    mi::base::Handle<const mi::neuraylib::IManifest> manifest;

    manifest = mdl_archive_api->get_manifest( nullptr);
    MI_CHECK( !manifest);

    manifest = mdl_archive_api->get_manifest( "non-existing.mdr");
    MI_CHECK( !manifest);

    manifest = mdl_archive_api->get_manifest( wrong_archive.c_str());
    MI_CHECK( !manifest);

    manifest = mdl_archive_api->get_manifest( archive.c_str());
    MI_CHECK( manifest);

    mi::Size count = manifest->get_number_of_fields();
    MI_CHECK_EQUAL( count, 13);

    const char* key = manifest->get_key( 0);
    MI_CHECK_EQUAL_CSTR( key, "mdl");

    key = manifest->get_key( 13);
    MI_CHECK( !key);

    const char* value = manifest->get_value( 0);
    MI_CHECK_EQUAL_CSTR( value, "1.3");

    value = manifest->get_value( 22);
    MI_CHECK( !value);

    count = manifest->get_number_of_fields( nullptr);
    MI_CHECK_EQUAL( count, 0);

    count = manifest->get_number_of_fields( "non-existing");
    MI_CHECK_EQUAL( count, 0);

    count = manifest->get_number_of_fields( "mdl");
    MI_CHECK_EQUAL( count, 1);

    count = manifest->get_number_of_fields( "exports.function");
    MI_CHECK_EQUAL( count, 10);

    value = manifest->get_value( nullptr, 0);
    MI_CHECK( !value);

    value = manifest->get_value( "non-existing", 0);
    MI_CHECK( !value);

    value = manifest->get_value( "mdl", 0);
    MI_CHECK_EQUAL_CSTR( value, "1.3");

    value = manifest->get_value( "mdl", 1);
    MI_CHECK( !value);

    value = manifest->get_value( "exports.function", 0);
    MI_CHECK_EQUAL_CSTR( value, "::test_archives::fd_in_archive");

    value = manifest->get_value( "exports.function", 9);
    MI_CHECK_EQUAL_CSTR( value, "::test_archives::fd_in_archive_bsdf_measurement_absolute");

    value = manifest->get_value( "exports.function", 10);
    MI_CHECK( !value);

    value = manifest->get_value( "userdefined", 0);
    MI_CHECK( !value);
}

void check_archive_get_file( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_archive_api> mdl_archive_api(
        neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());

    std::string archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/mdl_archives/query/test_archives.mdr");
    std::string wrong_archive
        = MI::TEST::mi_src_path( "prod/lib/neuray/" TEST_MDL_FILE ".mdl");

    mi::base::Handle<mi::neuraylib::IReader> reader;

    reader = mdl_archive_api->get_file( nullptr, "foo");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( "foo", nullptr);
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( "non-existing.mdr", "foo");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( wrong_archive.c_str(), "foo");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( archive.c_str(), "test_archives.mdl");
    MI_CHECK( reader);

    std::string filename = std::string( "test_archives") + dir_sep + "test_in_archive.png";
    reader = mdl_archive_api->get_file( archive.c_str(), filename.c_str());
    MI_CHECK( reader);

    reader = mdl_archive_api->get_file( archive.c_str(), "non-existing.mdl");
    MI_CHECK( !reader);

    reader = mdl_archive_api->get_file( archive.c_str(), "test_archives");
    MI_CHECK( !reader);
}

void check_mdl_export_reimport(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    Exreimp_data modules[] = {
        { "mdl::variants", "::variants", 0, 0 },
        { "mdl::resources", "::resources", 0, 0, false, false, false },
        { "mdl::resources", "::resources2", 0, 6011, true, false, false },
        { "mdl::resources", "::resources3", 0, 6006, false, true, false },
        { "mdl::resources", "::resources4", 0, 0, false, false, true },
        { "mdl::resources", "::resources5", 0, 6006, false, true, true },
        { "mdl::new_materials", "::new_materials", 0, 0 },
        { "mdl::123_check_unicode_new_materials", "::123_check_unicode_new_materials", 0, 0 },
        { "mdl::123_check_unicode_new_variants", "::123_check_unicode_new_variants", 0, 0 },
    };

    for( const auto& module: modules)
        do_check_mdl_export_reimport( transaction, neuray, module);
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    fs::remove_all( fs::u8path( DIR_PREFIX));
    fs::create_directory( fs::u8path( DIR_PREFIX));

    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> global_scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(
            global_scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK_EQUAL( 0, mdl_configuration->add_mdl_path( DIR_PREFIX));
        MI_CHECK_EQUAL( 0, mdl_configuration->add_resource_path( DIR_PREFIX));

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        install_external_resolver( neuray);

        // run the actual tests

        // Load the primary modules.
        load_primary_modules( transaction.get(), neuray);

        // Create function calls used by many later tests.
        create_function_calls( transaction.get(), mdl_factory.get());

        // Load the secondary modules.
        load_secondary_modules( transaction.get(), neuray);

        // Module builder
        check_module_builder_variants( transaction.get(), mdl_factory.get());
        check_module_builder_resources( transaction.get(), mdl_factory.get());
        check_module_builder_misc( transaction.get(), mdl_factory.get());
        check_module_builder_removed( transaction.get(), mdl_factory.get());
        check_module_builder_utf8( transaction.get(), neuray);
        check_module_builder_implicit_casts( transaction.get(), mdl_factory.get());

        // Module transformer
        check_module_transformer( transaction.get(), neuray);

        // MDLE
        check_mdle( transaction.get(), neuray); // after check_module_builder_*()

        // Archives
        check_create_archive( transaction.get(), neuray);
        check_extract_archive( neuray);
        check_archive_get_manifest( neuray);
        check_archive_get_file( neuray);

        // Export/re-import all modules created so far.
        check_mdl_export_reimport( transaction.get(), neuray);

        MI_CHECK_EQUAL( 0, transaction->commit());

        uninstall_external_resolver( neuray);
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_imodule_creation )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));

        // set MDL paths
        std::string path1 = MI::TEST::mi_src_path( "prod/lib/neuray");
        std::string path2 = MI::TEST::mi_src_path( "io/scene");
        std::string path3 = MI::TEST::mi_src_path( "prod");
        std::string path4 = MI::TEST::mi_src_path( "prod/lib");
        set_mdl_paths( neuray.get(), {path1, path2, path3, path4});

        // load plugins
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_dds));
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        // Required by check_module_builder_misc().
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        MI_CHECK_EQUAL( 0, mdl_configuration->set_expose_names_of_let_expressions( true));

        run_tests( neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

