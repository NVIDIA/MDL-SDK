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

#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/iimage.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/iplugin_configuration.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_shared.h"

// To avoid race conditions, each unit test uses a separate subdirectory for all the files it
// creates (possibly with further subdirectories).
#define DIR_PREFIX "output_test_imodule_basics"

#include "test_shared_mdl.h" // depends on DIR_PREFIX



mi::Sint32 result = 0;



// === Helper functions for encoding ===============================================================

void check_decoding(
    mi::neuraylib::IMdl_factory* mdl_factory, const char* input, const char* expected_result)
{
    mi::base::Handle<const mi::IString> s1( mdl_factory->decode_name( input));
    MI_CHECK_EQUAL_CSTR( s1->get_c_str(), expected_result);
}

void check_encoding_module(
    mi::neuraylib::IMdl_factory* mdl_factory, const char* input, const char* expected_result)
{
    mi::base::Handle<const mi::IString> s( mdl_factory->encode_module_name( input));
    MI_CHECK_EQUAL_CSTR( s->get_c_str(), expected_result);
}

void check_encoding_type(
    mi::neuraylib::IMdl_factory* mdl_factory, const char* input, const char* expected_result)
{
    mi::base::Handle<const mi::IString> s( mdl_factory->encode_type_name( input));
    MI_CHECK_EQUAL_CSTR( s->get_c_str(), expected_result);
}

void check_encoding_function(
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const char* name,
    std::initializer_list<const char*> parameter_types,
    const char* expected_result)
{
    mi::base::Handle<const mi::IArray> array( create_istring_array( factory, parameter_types));
    mi::base::Handle<const mi::IString> s(
        mdl_factory->encode_function_definition_name( name, array.get()));
    MI_CHECK_EQUAL_CSTR( s->get_c_str(), expected_result);
}

// === Tests =======================================================================================

void check_imdl_factory( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    {
        // test MDL identifiers

        MI_CHECK( mdl_factory->is_valid_mdl_identifier( "foo"));
        MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "_foo"));   // starts with underscore
        MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "42foo"));  // starts with digit
        MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "module")); // keyword
        MI_CHECK( !mdl_factory->is_valid_mdl_identifier( "new"));    // reserved for future use
    }
    {
        // test MDL to DB name conversion

        mi::base::Handle<const mi::IString> db_name;

        db_name = mdl_factory->get_db_module_name( nullptr);
        MI_CHECK( !db_name);
        db_name = mdl_factory->get_db_module_name( "state");
        MI_CHECK( !db_name);
        db_name = mdl_factory->get_db_module_name( "::state");
        MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdl::state");

        db_name = mdl_factory->get_db_definition_name( nullptr);
        MI_CHECK( !db_name);
        db_name = mdl_factory->get_db_definition_name( "operator+(float,float)");
        MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdl::operator+(float,float)");
        db_name = mdl_factory->get_db_definition_name( "state::normal()");
        MI_CHECK( !db_name);
        db_name = mdl_factory->get_db_definition_name( "::state::normal()");
        MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdl::state::normal()");
        db_name = mdl_factory->get_db_definition_name( "::/path/to/mod.mdle::fd(int,int)");
        MI_CHECK_EQUAL_CSTR( db_name->get_c_str(), "mdle::/path/to/mod.mdle::fd(int,int)");
    }
    {
        // test encoding/decoding

        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());

        check_decoding( mdl_factory.get(), "()<>,:$%",                 "()<>,:$%");
        check_decoding( mdl_factory.get(), "%28%29%3C%3E%2C%3A%24%25", "()<>,:$%");

        check_encoding_module(
            mdl_factory.get(), "::foo_()<>,:$%", "::foo_%28%29%3C%3E%2C%3A%24%25");

        check_encoding_type(
            mdl_factory.get(), "::foo_()<>,:$%$1.3", "::foo_%28%29%3C%3E%2C%3A$%25%241.3");

        check_encoding_function( factory.get(), mdl_factory.get(),
            "::foo::bar",   { "int",  "float" }, "::foo::bar(int,float)" );
        check_encoding_function( factory.get(), mdl_factory.get(),
            "::foo$::bar$", { "int$", "float" }, "::foo%24::bar$(int$,float)" );

        check_encoding_module( mdl_factory.get(), "int",             "int");
        check_encoding_module( mdl_factory.get(), "::foo$::bar",     "::foo%24::bar");
        check_encoding_module( mdl_factory.get(), "::foo$::bar$1.3", "::foo%24::bar%241.3");
    }
}

void check_imdl_impexp_api( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    mi::base::Handle<const mi::IString> mdl_module_name;
    std::string expected_mdl_module_name;

    mdl_module_name = mdl_impexp_api->get_mdl_module_name( nullptr);
    MI_CHECK( !mdl_module_name);

    mdl_module_name = mdl_impexp_api->get_mdl_module_name( "");
    MI_CHECK( !mdl_module_name);

    mdl_module_name = mdl_impexp_api->get_mdl_module_name( mdle_path.c_str());
    MI_CHECK( !mdl_module_name);

    std::string mdl_path = MI::TEST::mi_src_path( "prod/lib/neuray/") + TEST_MDL_FILE ".mdl";

    std::vector<std::string> mdl_paths = get_mdl_paths( neuray);
    mdl_configuration->clear_mdl_paths();
    std::string path1 = MI::TEST::mi_src_path( "prod/lib");
    std::string path2 = MI::TEST::mi_src_path( "prod/lib/neuray");
    std::string path3 = MI::TEST::mi_src_path( "prod");
    std::string path4 = MI::TEST::mi_src_path( "io/scene");
    set_mdl_paths( neuray, {path1, path2, path3, path4});

    expected_mdl_module_name = "::neuray::" TEST_MDL;
    mdl_module_name = mdl_impexp_api->get_mdl_module_name( mdl_path.c_str());
    MI_CHECK( mdl_module_name);
    MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

    expected_mdl_module_name = "::neuray::" TEST_MDL;
    mdl_module_name = mdl_impexp_api->get_mdl_module_name(
        mdl_path.c_str(), mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_FIRST);
    MI_CHECK( mdl_module_name);
    MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

    expected_mdl_module_name = "::lib::neuray::" TEST_MDL;
    mdl_module_name = mdl_impexp_api->get_mdl_module_name(
        mdl_path.c_str(), mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_SHORTEST);
    MI_CHECK( mdl_module_name);
    MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

    expected_mdl_module_name = "::" TEST_MDL;
    mdl_module_name = mdl_impexp_api->get_mdl_module_name(
        mdl_path.c_str(), mi::neuraylib::IMdl_impexp_api::SEARCH_OPTION_USE_LONGEST);
    MI_CHECK( mdl_module_name);
    MI_CHECK_EQUAL_CSTR( mdl_module_name->get_c_str(), expected_mdl_module_name.c_str());

    set_mdl_paths( neuray, mdl_paths);
}

void check_itransaction( mi::neuraylib::ITransaction* transaction)
{
    // modules and definitions

    // check that ITransaction::access() works
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
    MI_CHECK( c_module);
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_0()"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_0()"));
    MI_CHECK( c_md);

    // check that ITransaction::edit() works
    mi::base::Handle<mi::neuraylib::IModule> m_module(
        transaction->edit<mi::neuraylib::IModule>( "mdl::" TEST_MDL));
    MI_CHECK( m_module);
    // check that ITransaction::edit() is prohibited
    mi::base::Handle<mi::neuraylib::IFunction_definition> m_fd(
        transaction->edit<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_0()"));
    MI_CHECK( !m_fd);
    mi::base::Handle<mi::neuraylib::IFunction_definition> m_md(
        transaction->edit<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_0()"));
    MI_CHECK( !m_md);

    // check that ITransaction::copy() is prohibited
    MI_CHECK_EQUAL( -6, transaction->copy( "mdl::" TEST_MDL, "foo"));
    MI_CHECK_EQUAL( -6, transaction->copy( "mdl::" TEST_MDL "::fd_0()", "foo"));
    MI_CHECK_EQUAL( -6, transaction->copy( "mdl::" TEST_MDL "::md_0()", "foo"));

    {
        // ... also on defaults
        mi::base::Handle<const mi::neuraylib::IFunction_definition> m_md_default_call(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::" TEST_MDL "::md_default_call(material)"));

        mi::base::Handle<const mi::neuraylib::IExpression_list> md_defaults(
            m_md_default_call->get_defaults());
        mi::base::Handle<const mi::neuraylib::IExpression_call> md_default_call(
            md_defaults->get_expression<mi::neuraylib::IExpression_call>( "param0"));
        const char* call_name = md_default_call->get_call();
        MI_CHECK_EQUAL( -6, transaction->copy( call_name, "foo"));

        mi::base::Handle<mi::neuraylib::IImage> dummy(
            transaction->create<mi::neuraylib::IImage>( "Image"));
        transaction->store( dummy.get(), "dummy_image");
        MI_CHECK_EQUAL( -9, transaction->copy( "dummy_image", call_name));
    }

    // function calls and material instances

    do_create_function_call(
        transaction, "mdl::" TEST_MDL "::fd_1(int)", "mdl::" TEST_MDL "::fc_1");
    do_create_function_call(
        transaction, "mdl::" TEST_MDL "::md_1(color)", "mdl::" TEST_MDL "::mi_1");

    // check that ITransaction::access() works
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1"));
    MI_CHECK( c_fc);
    mi::base::Handle<const mi::neuraylib::IFunction_call> c_mi(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_1"));
    MI_CHECK( c_mi);

    // check that ITransaction::edit() works
    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1"));
    MI_CHECK( m_fc);
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_1"));
    MI_CHECK( m_mi);

    // check that ITransaction::copy() works
    MI_CHECK_EQUAL(
        0, transaction->copy( "mdl::" TEST_MDL "::fc_1", "mdl::" TEST_MDL "::fc_1_copy"));
    MI_CHECK_EQUAL(
        0, transaction->copy( "mdl::" TEST_MDL "::mi_1", "mdl::" TEST_MDL "::mi_1_copy"));
}

void check_imodule_part1( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    // check the "builtins" module

    mi::base::Handle<const mi::neuraylib::IModule> b_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::%3Cbuiltins%3E"));
    MI_CHECK( b_module);
    MI_CHECK_EQUAL_CSTR( "::%3Cbuiltins%3E", b_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "%3Cbuiltins%3E", b_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, b_module->get_mdl_package_component_count());
    MI_CHECK( b_module->is_standard_module());
    MI_CHECK( !b_module->is_mdle_module());

    // standard material category
    mi::base::Handle<const mi::neuraylib::IStruct_category> sc(
        tf->get_predefined_struct_category(
            mi::neuraylib::IStruct_category::CID_MATERIAL_CATEGORY));
    MI_CHECK_EQUAL_CSTR( "::material_category", sc->get_symbol());
    MI_CHECK_EQUAL( mi::neuraylib::IStruct_category::CID_MATERIAL_CATEGORY,
        sc->get_predefined_id());

    // standard material
    mi::base::Handle<const mi::neuraylib::IType_struct> std_mat(
        tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
    sc = std_mat->get_struct_category();
    MI_CHECK_EQUAL_CSTR( "::material_category", sc->get_symbol());
    MI_CHECK_EQUAL( mi::neuraylib::IStruct_category::CID_MATERIAL_CATEGORY,
        sc->get_predefined_id());
    MI_CHECK_EQUAL_CSTR( "::material", std_mat->get_symbol());
    mi::Size index_ior = std_mat->find_field( "ior");
    mi::base::Handle<const mi::neuraylib::IType> std_mat_ior(
        std_mat->get_field_type( index_ior));
    MI_CHECK_EQUAL( std_mat_ior->get_all_type_modifiers(), mi::neuraylib::IType::MK_NONE);

    // standard structure types
    const char* struct_types[] = {
        "::df::bsdf_component", "::df::edf_component", "::df::vdf_component"
    };

    for( auto& struct_type : struct_types) {
        mi::base::Handle<const mi::neuraylib::IType> type( tf->create_struct( struct_type));
        MI_CHECK( type);
    }

    // standard enum types
    const char* enum_types[] = {
        "::df::scatter_mode", "::state::coordinate_space", "::tex::gamma_mode", "::tex::wrap_mode"
    };

    for( auto& enum_type : enum_types) {
        mi::base::Handle<const mi::neuraylib::IType> type( tf->create_enum( enum_type));
        MI_CHECK( type);
    }

    // check another standard module

    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::state"));
    MI_CHECK( c_module);
    MI_CHECK_EQUAL_CSTR( "::state", c_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "state", c_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, c_module->get_mdl_package_component_count());
    MI_CHECK( c_module->is_standard_module());
    MI_CHECK( !c_module->is_mdle_module());

    // check regular module

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
    MI_CHECK( c_module);
    std::string filename = MI::TEST::mi_src_path( "prod/lib/neuray/")  + TEST_MDL_FILE ".mdl";
    MI_CHECK_EQUAL( filename, c_module->get_filename());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_module->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "" TEST_MDL, c_module->get_mdl_simple_name());
    MI_CHECK_EQUAL( 0, c_module->get_mdl_package_component_count());
    MI_CHECK( !c_module->is_standard_module());
    MI_CHECK( !c_module->is_mdle_module());
    MI_CHECK_EQUAL( mi::neuraylib::MDL_VERSION_1_9, c_module->get_mdl_version());

    {
        const char* const imports[] = {
            "mdl::%3Cbuiltins%3E",
            "mdl::df",
            "mdl::anno",
            "mdl::state",
            "mdl::tex",
            "mdl::base",
            "mdl::math",
        };

        mi::Size count = sizeof( imports) / sizeof( const char*);
        for( mi::Size i = 0; i < count; ++i)
            MI_CHECK_EQUAL_CSTR( imports[i], c_module->get_import( i));
        MI_CHECK( !c_module->get_import( count));
    }
    {
        const char* const materials[] = {
            "mdl::" TEST_MDL "::md_0()",
            "mdl::" TEST_MDL "::md_1(color)",
            "mdl::" TEST_MDL "::md_1_green(color)",
            "mdl::" TEST_MDL "::md_1_blue(color)",
            "mdl::" TEST_MDL "::md_1_no_default(color)",
            "mdl::" TEST_MDL "::md_2_index(color,color)",
            "mdl::" TEST_MDL "::md_2_index_nested(color,color[N])",
            "mdl::" TEST_MDL "::md_tmp_ms(color)",
            "mdl::" TEST_MDL "::md_tmp_bsdf(color)",
            "mdl::" TEST_MDL "::md_float3_arg(float3)",
            "mdl::" TEST_MDL "::md_with_annotations(color,float)",
            "mdl::" TEST_MDL "::md_enum(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::md_thin_walled(material)",
            "mdl::" TEST_MDL "::md_deferred(int[N])",
            "mdl::" TEST_MDL "::md_deferred_2(int[N],int[N])",
            "mdl::" TEST_MDL "::md_default_call(material)",
            "mdl::" TEST_MDL "::md_ternary_operator_argument(bool)",
            "mdl::" TEST_MDL "::md_ternary_operator_default(color)",
            "mdl::" TEST_MDL "::md_ternary_operator_body(color)",
            "mdl::" TEST_MDL "::md_resource_sharing(texture_2d,texture_2d)",
            "mdl::" TEST_MDL "::md_folding()",
            "mdl::" TEST_MDL "::md_folding2(bool,::" TEST_MDL "::Enum,int)",
            "mdl::" TEST_MDL "::md_folding_cutout_opacity(float,float)",
            "mdl::" TEST_MDL "::md_folding_cutout_opacity2(material)",
            "mdl::" TEST_MDL "::md_folding_transparent_layers(float)",
            "mdl::" TEST_MDL "::md_folding_transparent_layers2(color,float,bool)",
            "mdl::" TEST_MDL "::md_folding_transparent_layers3(float,float)",
            "mdl::" TEST_MDL "::md_parameters_uniform_auto_varying(int,int,int)",
            "mdl::" TEST_MDL "::md_parameter_uniform(color)",
            "mdl::" TEST_MDL "::md_uses_non_exported_function()",
            "mdl::" TEST_MDL "::md_uses_non_exported_material()",
            "mdl::" TEST_MDL "::md_jit(texture_2d,string,light_profile)",
            "mdl::" TEST_MDL "::md_reexport(float3)",
            "mdl::" TEST_MDL "::md_wrap(float,color,color)",
            "mdl::" TEST_MDL "::md_texture(texture_2d)",
            "mdl::" TEST_MDL "::md_light_profile(light_profile)",
            "mdl::" TEST_MDL "::md_bsdf_measurement(bsdf_measurement)",
            "mdl::" TEST_MDL "::md_baking(float,::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::md_baking_const()",
            "mdl::" TEST_MDL "::md_named_temporaries(float)",
            "mdl::" TEST_MDL "::md_aov(float,float)"
        };

#if 0
        for( mi::Size i = 0, n = c_module->get_material_count(); i < n; ++i)
            printf( "%llu: \"%s\",\n", i, c_module->get_material( i));
#endif

        mi::Size count = sizeof( materials) / sizeof( const char*);
        for( mi::Size i = 0; i < count; ++i)
            MI_CHECK_EQUAL_CSTR( materials[i], c_module->get_material( i));
        MI_CHECK( !c_module->get_material( count));
    }
    {
        const char* const functions[] = {
            "mdl::" TEST_MDL "::normal()",
            "mdl::" TEST_MDL "::fd_0()",
            "mdl::" TEST_MDL "::fd_1(int)",
            "mdl::" TEST_MDL "::fd_1_44(int)",
            "mdl::" TEST_MDL "::fd_1_no_default(int)",
            "mdl::" TEST_MDL "::fd_2_index(int,float)",
            "mdl::" TEST_MDL "::fd_2_index_nested(int,int[N])",
            "mdl::" TEST_MDL "::fd_with_annotations(color,float)",
            "mdl::" TEST_MDL "::Enum(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::int(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::fd_enum(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::fd_remove(int)",
            "mdl::" TEST_MDL "::fd_overloaded(int)",
            "mdl::" TEST_MDL "::fd_overloaded(float)",
            "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)",
            "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
            "mdl::" TEST_MDL "::fd_overloaded(int[3])",
            "mdl::" TEST_MDL "::fd_overloaded(int[N])",
            "mdl::" TEST_MDL "::fd_overloaded2(int)",
            "mdl::" TEST_MDL "::fd_overloaded2(int,int)",
            "mdl::" TEST_MDL "::fd_overloaded3(int,int)",
            "mdl::" TEST_MDL "::fd_overloaded3(int,float)",
            "mdl::" TEST_MDL "::foo_struct(::" TEST_MDL "::foo_struct)",
            "mdl::" TEST_MDL "::foo_struct()",
            "mdl::" TEST_MDL "::foo_struct(int,float)",
            "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)",
            "mdl::" TEST_MDL "::foo_struct.param_float(::" TEST_MDL "::foo_struct)",
            "mdl::" TEST_MDL "::fd_immediate(int[42])",
            "mdl::" TEST_MDL "::fd_deferred(int[N])",
            "mdl::" TEST_MDL "::fd_deferred_2(int[N],int[N])",
            "mdl::" TEST_MDL "::fd_deferred_struct(::" TEST_MDL "::foo_struct[N])",
            "mdl::" TEST_MDL "::fd_matrix(float3x2)",
            "mdl::" TEST_MDL "::fd_default_call(int)",
            "mdl::" TEST_MDL "::Enum2(::" TEST_MDL "::Enum2)",
            "mdl::" TEST_MDL "::int(::" TEST_MDL "::Enum2)",
            "mdl::" TEST_MDL "::fd_ret_bool()",
            "mdl::" TEST_MDL "::fd_ret_int()",
            "mdl::" TEST_MDL "::fd_ret_float()",
            "mdl::" TEST_MDL "::fd_ret_string()",
            "mdl::" TEST_MDL "::fd_ret_enum()",
            "mdl::" TEST_MDL "::fd_ret_enum2()",
            "mdl::" TEST_MDL "::fd_add(int,int)",
            "mdl::" TEST_MDL "::fd_return_uniform()",
            "mdl::" TEST_MDL "::fd_return_auto()",
            "mdl::" TEST_MDL "::fd_return_varying()",
            "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying(int,int,int)",
            "mdl::" TEST_MDL "::fd_parameters_uniform_auto_varying_color(color,color,color)",
            "mdl::" TEST_MDL "::fd_uniform()",
            "mdl::" TEST_MDL "::fd_varying()",
            "mdl::" TEST_MDL "::fd_auto_uniform()",
            "mdl::" TEST_MDL "::fd_auto_varying()",
            "mdl::" TEST_MDL "::fd_non_exported()",
            "mdl::" TEST_MDL "::lookup(int)",
            "mdl::" TEST_MDL "::fd_jit(float)",
            "mdl::" TEST_MDL "::color_weight(texture_2d,light_profile)",
            "mdl::" TEST_MDL "::color_weight(string)",
            "mdl::" TEST_MDL "::fd_wrap_x(float)",
            "mdl::" TEST_MDL "::fd_wrap_rhs(color,color,color)",
            "mdl::" TEST_MDL "::fd_wrap_r(color)",
            "mdl::" TEST_MDL "::fd_cycle(color)",
            "mdl::" TEST_MDL "::sub_struct(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct()",
            "mdl::" TEST_MDL "::sub_struct(bool,bool,color,float)",
            "mdl::" TEST_MDL "::sub_struct.a(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct.b(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct.c(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::sub_struct.d(::" TEST_MDL "::sub_struct)",
            "mdl::" TEST_MDL "::top_struct(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct()",
            "mdl::" TEST_MDL "::top_struct(float,double,::" TEST_MDL "::sub_struct,int,"
                "::" TEST_MDL "::sub_struct,float)",
            "mdl::" TEST_MDL "::top_struct.a(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.x(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.b(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.c(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.d(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::top_struct.e(::" TEST_MDL "::top_struct)",
            "mdl::" TEST_MDL "::fd_identity(int)",
            "mdl::" TEST_MDL "::fd_named_temporaries(int)",
            "mdl::" TEST_MDL "::fd_parameter_references(float,float,float)",
            "mdl::" TEST_MDL "::fd_global_scope_reference_test(int)",
            "mdl::" TEST_MDL "::bar_struct(::" TEST_MDL "::bar_struct)",
            "mdl::" TEST_MDL "::bar_struct()",
            "mdl::" TEST_MDL "::bar_struct(int,float)",
            "mdl::" TEST_MDL "::bar_struct.param_int(::" TEST_MDL "::bar_struct)",
            "mdl::" TEST_MDL "::bar_struct.param_float(::" TEST_MDL "::bar_struct)",
            "mdl::" TEST_MDL "::fd_bar_struct_to_int(::" TEST_MDL "::bar_struct)",
            "mdl::" TEST_MDL "::fd_accepts_int(int)",
            "mdl::" TEST_MDL "::aov_material(::" TEST_MDL "::aov_material)",
            "mdl::" TEST_MDL "::aov_material(float)",
            "mdl::" TEST_MDL "::aov_material.aov(::" TEST_MDL "::aov_material)"
        };

#if 0
        for( mi::Size i = 0, n = c_module->get_function_count(); i < n; ++i)
            printf( "%llu: \"%s\",\n", i, c_module->get_function( i));
#endif

        mi::Size count = sizeof( functions) / sizeof( const char*);
        for( mi::Size i = 0; i < count; ++i)
            MI_CHECK_EQUAL_CSTR( functions[i], c_module->get_function( i));
        MI_CHECK( !c_module->get_function( count));
    }

    // check user-defined struct categories

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
    mi::base::Handle<const mi::neuraylib::IStruct_category_list> struct_categories(
        c_module->get_struct_categories());
    MI_CHECK_EQUAL( 1, struct_categories->get_size());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::bar_struct_category", struct_categories->get_name( 0));

    sc = struct_categories->get_struct_category( zero_size);
    MI_CHECK( sc);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::bar_struct_category", sc->get_symbol());

    // check user-defined types

    mi::base::Handle<const mi::neuraylib::IType_list> types( c_module->get_types());
    MI_CHECK_EQUAL( 7, types->get_size());

    // custom enum
    mi::base::Handle<const mi::neuraylib::IType_enum> type0(
        types->get_type<mi::neuraylib::IType_enum>( zero_size));
    MI_CHECK( type0);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", types->get_name( 0));

    // custom struct without category
    mi::base::Handle<const mi::neuraylib::IType_struct> type1(
        types->get_type<mi::neuraylib::IType_struct>( 1));
    MI_CHECK( type1);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", types->get_name( 1));

    // just test the API binding, extensive test is in io/scene/mdl_elements/test_misc.cpp
    mi::base::Handle<const mi::IString> module_name( tf->get_mdl_module_name( type1.get()));
    MI_CHECK_EQUAL_CSTR( module_name->get_c_str(), "::" TEST_MDL);
    // just test the API binding, extensive test is in io/scene/mdl_elements/test_types.cpp
    mi::base::Handle<const mi::IString> mdl_type_name( tf->get_mdl_type_name( type1.get()));
    MI_CHECK_EQUAL_CSTR( mdl_type_name->get_c_str(), "::" TEST_MDL "::foo_struct");
    mi::base::Handle<const mi::neuraylib::IType_struct> type2(
        tf->create_from_mdl_type_name<mi::neuraylib::IType_struct>( mdl_type_name->get_c_str()));
    MI_CHECK_EQUAL( tf->compare( type1.get(), type2.get()), 0);

    // custom struct in custom category
    mi::base::Handle<const mi::neuraylib::IType_struct> type5(
        types->get_type<mi::neuraylib::IType_struct>( 5));
    MI_CHECK( type5);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::bar_struct", types->get_name( 5));
    mi::base::Handle<const mi::neuraylib::IStruct_category> sc5(
        type5->get_struct_category());
    MI_CHECK( sc5);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::bar_struct_category", sc5->get_symbol());

    // custom struct in standard material category
    mi::base::Handle<const mi::neuraylib::IType_struct> type6(
        types->get_type<mi::neuraylib::IType_struct>( 6));
    MI_CHECK( type6);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::aov_material", types->get_name( 6));
    mi::base::Handle<const mi::neuraylib::IStruct_category> sc6(
        type6->get_struct_category());
    MI_CHECK( sc6);
    MI_CHECK_EQUAL_CSTR( "::material_category", sc6->get_symbol());

    // check user-defined constants

    mi::base::Handle<const mi::neuraylib::IValue_list> constants( c_module->get_constants());
    MI_CHECK_EQUAL( 1, constants->get_size());
    mi::base::Handle<const mi::neuraylib::IValue_int> constant0(
        constants->get_value<mi::neuraylib::IValue_int>( zero_size));
    MI_CHECK_EQUAL( 42, constant0->get_value());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::some_constant", constants->get_name( 0));
}

void check_enums(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_enum(::" TEST_MDL "::Enum)"));
    MI_CHECK( c_fd);

    mi::base::Handle<const mi::neuraylib::IType_list> types( c_fd->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType_enum> type_enum(
        types->get_type<mi::neuraylib::IType_enum>( "param0"));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", type_enum->get_symbol());

    mi::base::Handle<const mi::neuraylib::IExpression_list> c_defaults( c_fd->get_defaults());
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_param0_expr(
        c_defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
    mi::base::Handle<const mi::neuraylib::IValue_enum> c_param0_value(
        c_param0_expr->get_value<mi::neuraylib::IValue_enum>());
    MI_CHECK_EQUAL( 1, c_param0_value->get_index());

    mi::base::Handle<const mi::neuraylib::IType_enum> param0_type(
        tf->create_enum( "::" TEST_MDL "::Enum"));
    mi::base::Handle<mi::neuraylib::IValue_enum> m_param0_value(
        vf->create_enum( param0_type.get(), 0));
    mi::base::Handle<mi::neuraylib::IExpression_constant> m_param0_expr(
        ef->create_constant( m_param0_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, args->add_expression( "param0", m_param0_expr.get()));

    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
    c_fd->create_function_call( args.get(), &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( m_fc);
    MI_CHECK_EQUAL( 0,
        transaction->store( m_fc.get(), "mdl::" TEST_MDL "::fd_enum(" TEST_MDL "::Enum)_param0"));
}

void check_structs( mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd;
    mi::base::Handle<const mi::neuraylib::IType_struct> type_struct;
    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_list> types;

    // default constructor
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::foo_struct()");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 0, c_fd->get_parameter_count());
    type_struct = c_fd->get_return_type<mi::neuraylib::IType_struct>();
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", type_struct->get_symbol());

    // elemental constructor
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::foo_struct(int,float)");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 2, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
    MI_CHECK_EQUAL_CSTR( "param_int", c_fd->get_parameter_name( zero_size));
    type = types->get_type( 1);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, type->get_kind());
    MI_CHECK_EQUAL_CSTR( "param_float", c_fd->get_parameter_name( 1));
    type_struct = c_fd->get_return_type<mi::neuraylib::IType_struct>();
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", type_struct->get_symbol());

    // field access operator for param_int
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    types = c_fd->get_parameter_types();
    type_struct = types->get_type<mi::neuraylib::IType_struct>( zero_size);
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", type_struct->get_symbol());
    MI_CHECK_EQUAL_CSTR( "s", c_fd->get_parameter_name( zero_size));
    type = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());
}

void check_overload_resolution(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IModule> b_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::%3Cbuiltins%3E"));
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL));

    // check get_function_overloads() (no expression list, retrieve all overloads)

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::" TEST_MDL);
    mi::base::Handle<const mi::IArray> result;
    mi::base::Handle<const mi::IString> element;
    MI_CHECK( !c_module->get_function_overloads( nullptr));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::non-existing");
    MI_CHECK_EQUAL( 0, result->get_length());
    result = c_module->get_function_overloads( "non-existing");
    MI_CHECK_EQUAL( 0, result->get_length());

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded");
    MI_CHECK_EQUAL( 6, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());
    element = result->get_element<mi::IString>( 1);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());
    element = result->get_element<mi::IString>( 2);
    MI_CHECK_EQUAL_CSTR(
        "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());
    element = result->get_element<mi::IString>( 3);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());
    element = result->get_element<mi::IString>( 4);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());
    element = result->get_element<mi::IString>( 5);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    result = c_module->get_function_overloads( "fd_overloaded");
    MI_CHECK_EQUAL( 6, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());
    element = result->get_element<mi::IString>( 1);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());
    element = result->get_element<mi::IString>( 2);
    MI_CHECK_EQUAL_CSTR(
        "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());
    element = result->get_element<mi::IString>( 3);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());
    element = result->get_element<mi::IString>( 4);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());
    element = result->get_element<mi::IString>( 5);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::md_1");
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", element->get_c_str());

    result = c_module->get_function_overloads( "md_1");
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", element->get_c_str());

    result = c_module->get_function_overloads( "md_aov");
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_aov(float,float)", element->get_c_str());

    // check get_function_overloads (expression list, retrieve exact match overload)

    mi::base::Handle<mi::neuraylib::IValue> value_int( vf->create_int( 42));
    mi::base::Handle<mi::neuraylib::IExpression> expr_int(
        ef->create_constant( value_int.get()));
    mi::base::Handle<mi::neuraylib::IValue> value_color( vf->create_color( 1.0, 1.0, 1.0));
    mi::base::Handle<mi::neuraylib::IExpression> expr_color(
        ef->create_constant( value_color.get()));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, arguments->add_expression( "param0", expr_int.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "param0", expr_color.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::fd_overloaded", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "tint", expr_color.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::md_1", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::md_1(color)", element->get_c_str());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "tint", expr_int.get()));

    result = c_module->get_function_overloads( "mdl::" TEST_MDL "::md_1", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    arguments = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, arguments->add_expression( "param0", expr_int.get()));

    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded2", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());

    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded3", arguments.get());
    MI_CHECK_EQUAL( 2, result->get_length());

    // check get_function_overloads() (signature, retrieve best-matching overloads)

    mi::base::Handle<mi::IArray> parameter_types;
    MI_CHECK( !c_module->get_function_overloads( nullptr, parameter_types.get()));

    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::non-existing", parameter_types.get());
    MI_CHECK_EQUAL( 0, result->get_length());
    result = c_module->get_function_overloads( "non-existing", parameter_types.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 6, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());
    element = result->get_element<mi::IString>( 1);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());
    element = result->get_element<mi::IString>( 2);
    MI_CHECK_EQUAL_CSTR(
        "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());
    element = result->get_element<mi::IString>( 3);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());
    element = result->get_element<mi::IString>( 4);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());
    // The overload with "int[N]" is not in the result set because "int[3]" is more specific.

    parameter_types = create_istring_array( factory.get(), { "int" });
    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int)", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "float" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(float)", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "::" TEST_MDL "::Enum" });
    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
        "mdl::" TEST_MDL "::fd_overloaded(::" TEST_MDL "::Enum)", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "::state::coordinate_space" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(::state::coordinate_space)",
        element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "int[3]" });
    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[3])", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "int[6]" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    // The deferred symbol size symbol is meaningless.
    parameter_types = create_istring_array( factory.get(), { "int[XnonexistantX]" });
    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    // The deferred symbol size symbol is meaningless, even empty is allowed.
    parameter_types = create_istring_array( factory.get(), { "int[]" });
    result = c_module->get_function_overloads( "fd_overloaded", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_overloaded(int[N])", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "int" });
    result = c_module->get_function_overloads(
        "mdl::" TEST_MDL "::foo_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::foo_struct(int,float)", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { });
    result = c_module->get_function_overloads( "foo_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::foo_struct()", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { });
    result = b_module->get_function_overloads( "mdl::int", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
     // Best match due to default argument.
    MI_CHECK_EQUAL_CSTR( "mdl::int(int)", element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "float" });
    result = b_module->get_function_overloads( "int", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR( "mdl::int(float)", element->get_c_str());

    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl2");

    // Check if reexported structs can be resolved using the new name.
    parameter_types = create_istring_array( factory.get(), { "::test_mdl2::foo_struct" });
    result = c_module->get_function_overloads(
        "mdl::test_mdl2::fd_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
        "mdl::test_mdl2::fd_reexported_struct(::" TEST_MDL "::foo_struct)", element->get_c_str());

    // Check if reexported structs can be resolved using the original name.
    parameter_types = create_istring_array( factory.get(), { "::" TEST_MDL "::foo_struct" });
    result = c_module->get_function_overloads( "fd_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
        "mdl::test_mdl2::fd_reexported_struct(::" TEST_MDL "::foo_struct)", element->get_c_str());

    // Same as above but this time 'foo_struct' is not re-exported
    c_module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl3");

    parameter_types = create_istring_array( factory.get(), { "::test_mdl3::foo_struct" });
    result = c_module->get_function_overloads(
        "mdl::test_mdl3::fd_using_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
        "mdl::test_mdl3::fd_using_reexported_struct(::" TEST_MDL "::foo_struct)",
        element->get_c_str());

    parameter_types = create_istring_array( factory.get(), { "::" TEST_MDL "::foo_struct" });
    result = c_module->get_function_overloads( "fd_using_reexported_struct", parameter_types.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
        "mdl::test_mdl3::fd_using_reexported_struct(::" TEST_MDL "::foo_struct)",
        element->get_c_str());

    // check ::%3Cbuiltins%3E module

    // array length operator without arguments
    arguments = ef->create_expression_list();
    result = b_module->get_function_overloads( "mdl::operator_len", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    // array length operator with arguments
    mi::base::Handle<const mi::neuraylib::IType> type_int( tf->create_int());
    mi::base::Handle<const mi::neuraylib::IType_array> type_int2(
        tf->create_immediate_sized_array( type_int.get(), 2));
    mi::base::Handle<const mi::neuraylib::IValue> value( vf->create_array( type_int2.get()));
    mi::base::Handle<const mi::neuraylib::IExpression> expr( ef->create_constant( value.get()));
    MI_CHECK_EQUAL( 0, arguments->add_expression( "a", expr.get()));
    result = b_module->get_function_overloads( "operator_len", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
         "mdl::operator_len(%3C0%3E[])", element->get_c_str());

    // array index operator without arguments
    arguments = ef->create_expression_list();
    result = b_module->get_function_overloads( "mdl::operator[]", arguments.get());
    MI_CHECK_EQUAL( 0, result->get_length());

    // array index operator with arguments
    MI_CHECK_EQUAL( 0, arguments->add_expression( "a", expr.get()));
    value = vf->create_int( 0);
    expr = ef->create_constant( value.get());
    MI_CHECK_EQUAL( 0, arguments->add_expression( "i", expr.get()));
    result = b_module->get_function_overloads( "operator[]", arguments.get());
    MI_CHECK_EQUAL( 1, result->get_length());
    element = result->get_element<mi::IString>( 0);
    MI_CHECK_EQUAL_CSTR(
         "mdl::operator[](%3C0%3E[],int)", element->get_c_str());

    // operator with encoded characters
    result = b_module->get_function_overloads(
         "mdl::operator%3C", static_cast<const mi::neuraylib::IExpression_list*>( nullptr));
    MI_CHECK_EQUAL( 3, result->get_length());
}

void check_ifunction_definition(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> t;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults;
    mi::base::Handle<const mi::neuraylib::IExpression> body;

    // check a function definition with no arguments

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_0()"));
    MI_CHECK( c_fd);
    MI_CHECK_EQUAL( c_fd->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_fd->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_0()", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_0", c_fd->get_mdl_simple_name());
    MI_CHECK( !c_fd->get_prototype());
    t = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    body = c_fd->get_body();
    MI_CHECK( body);
    MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_CONSTANT, body->get_kind());

    MI_CHECK_EQUAL( 0, c_fd->get_parameter_count());
    MI_CHECK( !c_fd->get_parameter_name( 0));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( "invalid"));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( nullptr));

    types = c_fd->get_parameter_types();
    MI_CHECK( !types->get_type( zero_size));
    MI_CHECK( !types->get_type( zero_string));
    MI_CHECK( !types->get_type( "invalid"));

    defaults = c_fd->get_defaults();
    MI_CHECK( !defaults->get_expression( zero_size));
    MI_CHECK( !defaults->get_expression( zero_string));
    MI_CHECK( !defaults->get_expression( "invalid"));

    // check a function definition with arguments with defaults

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_1(int)");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_fd->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_1(int)", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_1", c_fd->get_mdl_simple_name());
    MI_CHECK( !c_fd->get_prototype());
    t = c_fd->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    body = c_fd->get_body();
    MI_CHECK( body);
    MI_CHECK_EQUAL( mi::neuraylib::IExpression::EK_PARAMETER, body->get_kind());

    MI_CHECK_EQUAL( 1, c_fd->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "param0", c_fd->get_parameter_name( 0));
    MI_CHECK( !c_fd->get_parameter_name( 1));

    MI_CHECK_EQUAL( 0, c_fd->get_parameter_index( "param0"));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( nullptr));
    MI_CHECK_EQUAL( minus_one_size, c_fd->get_parameter_index( "invalid"));

    types = c_fd->get_parameter_types();
    t = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    MI_CHECK( !types->get_type( 1));
    t = types->get_type( "param0");
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    MI_CHECK( !types->get_type( zero_string));
    MI_CHECK( !types->get_type( "invalid"));

    MI_CHECK_EQUAL_CSTR( "int", c_fd->get_mdl_parameter_type_name( 0));
    MI_CHECK( !c_fd->get_mdl_parameter_type_name( 1));

    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_default;
    mi::base::Handle<const mi::neuraylib::IValue_int> c_value_int;

    defaults = c_fd->get_defaults();
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
    t = c_default->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    c_value_int = c_default->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( c_value_int->get_value(), 42);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( 1);
    MI_CHECK( !c_default);

    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0");
    t = c_default->get_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());
    c_value_int = c_default->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( c_value_int->get_value(), 42);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_string);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "invalid");
    MI_CHECK( !c_default);

    // check a function definition with arguments without defaults

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_1_no_default(int)");
    MI_CHECK( c_fd);

    defaults = c_fd->get_defaults();
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_size);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( 1);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "param0");
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( zero_string);
    MI_CHECK( !c_default);
    c_default = defaults->get_expression<mi::neuraylib::IExpression_constant>( "invalid");
    MI_CHECK( !c_default);

    // check is_uniform() property

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_uniform()");
    MI_CHECK( c_fd->is_uniform());
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_varying()");
    MI_CHECK( !c_fd->is_uniform());
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_auto_uniform()");
    MI_CHECK( c_fd->is_uniform());
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_auto_varying()");
    MI_CHECK( !c_fd->is_uniform());

    // check a function definition which is a variant

    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_1_44(int)");
    MI_CHECK( c_fd);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL, c_fd->get_module());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL, c_fd->get_mdl_module_name());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_1_44(int)", c_fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR( "fd_1_44", c_fd->get_mdl_simple_name());
    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_1(int)", c_fd->get_prototype());

    // check MDL versions

    mi::neuraylib::Mdl_version since, removed;

    // removed in MDL 1.1
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::df::measured_edf$1.0(light_profile,bool,float3x3,string)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_1_1);

    // removed in MDL 1.2
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::df::measured_edf$1.1(light_profile,float,bool,float3x3,string)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_1);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_1_2);

    // not yet removed
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::df::measured_edf(light_profile,float,bool,float3x3,float3,string)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_2);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // not from stdlib
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_0()");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_9);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: array constructor (DAG instrinsic)
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::T[](...)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: array length operator (DAG intrinsic)
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator_len(%3C0%3E[])");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: array index operator
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator[](%3C0%3E[],int)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_0);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: cast operator
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator_cast(%3C0%3E)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_5);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);

    // special case: decl_cast operator (DAG intrinsic)
    c_fd = transaction->access<mi::neuraylib::IFunction_definition>(
         "mdl::operator_decl_cast(%3C0%3E)");
    c_fd->get_mdl_version( since, removed);
    MI_CHECK_EQUAL( since, mi::neuraylib::MDL_VERSION_1_9);
    MI_CHECK_EQUAL( removed, mi::neuraylib::MDL_VERSION_INVALID);
}

void check_ifunction_call( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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

    mi::base::Handle<const mi::neuraylib::IType> t;

    // check a function call with no arguments

    mi::base::Handle<const mi::neuraylib::IFunction_call> c_fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_0"));
    MI_CHECK( c_fc);
    MI_CHECK_EQUAL( c_fc->get_element_type(), mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_0()", c_fc->get_function_definition());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_0()", c_fc->get_mdl_function_definition());
    t = c_fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    MI_CHECK_EQUAL( 0, c_fc->get_parameter_count());
    MI_CHECK( !c_fc->get_parameter_name( 0));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( "invalid"));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( nullptr));

    // check a function call with one argument

    c_fc = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1");
    MI_CHECK( c_fc);

    MI_CHECK_EQUAL_CSTR( "mdl::" TEST_MDL "::fd_1(int)", c_fc->get_function_definition());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::fd_1(int)", c_fc->get_mdl_function_definition());
    t = c_fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, t->get_kind());

    MI_CHECK_EQUAL( 1, c_fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "param0", c_fc->get_parameter_name( 0));
    MI_CHECK( !c_fc->get_parameter_name( 1));

    MI_CHECK_EQUAL( 0, c_fc->get_parameter_index( "param0"));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( nullptr));
    MI_CHECK_EQUAL( minus_one_size, c_fc->get_parameter_index( "invalid"));

    // check serialized names

    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::" TEST_MDL "::fd_1(int)",
            /*argument_types*/ nullptr,
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::" TEST_MDL "::fd_1(int)");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "fd_1(int)");

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        c_fc->get_parameter_types());

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::" TEST_MDL "::fd_1(int)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types( dfn->get_argument_types());
    MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::" TEST_MDL "::fd_1(int)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2( dfn2->get_argument_types());
    MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::" TEST_MDL);
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);

    c_fc.reset();

    // test instantiation with parameters

    mi::base::Handle<mi::neuraylib::IExpression_list> args(
        ef->create_expression_list());

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>( "mdl::" TEST_MDL "::fd_1(int)"));

    // with empty argument list
    c_fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( c_fc);

    // with wrong argument name
    mi::base::Handle<mi::neuraylib::IValue> arg_int_value( vf->create_int());
    mi::base::Handle<mi::neuraylib::IExpression> arg_int(
        ef->create_constant( arg_int_value.get()));
    MI_CHECK_EQUAL( 0, args->add_expression( "wrong_argument_name", arg_int.get()));
    c_fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -1, result);
    MI_CHECK( !c_fc);
    args = ef->create_expression_list();

    // with wrong argument type
    mi::base::Handle<mi::neuraylib::IValue> arg_float_value( vf->create_float());
    mi::base::Handle<mi::neuraylib::IExpression> arg_float(
        ef->create_constant( arg_float_value.get()));
    MI_CHECK_EQUAL( 0, args->add_expression( "param0", arg_float.get()));
    c_fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -2, result);
    MI_CHECK( !c_fc);
    args = ef->create_expression_list();

    // with declarative argument for an imperative function
    fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_bar_struct_to_int(::" TEST_MDL "::bar_struct)");
    MI_CHECK( fd->is_declarative());
    mi::base::Handle<mi::neuraylib::IExpression> arg_non_declarative(
        ef->create_call( "mdl::" TEST_MDL "::fc_bar_struct_to_int"));
    MI_CHECK_EQUAL( 0, args->add_expression( "param0", arg_non_declarative.get()));
    fd = transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::fd_accepts_int(int)");
    MI_CHECK( !fd->is_declarative());
    c_fc = fd->create_function_call( args.get(), &result);
    MI_CHECK_EQUAL( -10, result);
    MI_CHECK( !c_fc);

    // check get_argument()/set_argument() (constants only)

    mi::base::Handle<mi::neuraylib::IFunction_call> m_fc(
        transaction->edit<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_1"));
    MI_CHECK( m_fc);

    mi::base::Handle<mi::neuraylib::IValue_int> m_value_int( vf->create_int( -42));
    mi::base::Handle<mi::neuraylib::IExpression_constant> m_expression(
        ef->create_constant( m_value_int.get()));
    MI_CHECK_EQUAL( 0, m_fc->set_argument( "param0", m_expression.get()));

    mi::base::Handle<const mi::neuraylib::IExpression_list> c_expression_list(
        m_fc->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression_constant> c_expression(
        c_expression_list->get_expression<mi::neuraylib::IExpression_constant>( "param0"));
    mi::base::Handle<const mi::neuraylib::IValue_int> c_value_int(
        c_expression->get_value<mi::neuraylib::IValue_int>());
    MI_CHECK_EQUAL( -42, c_value_int->get_value());

    // check error codes of set_argument()

    MI_CHECK_EQUAL( -1, m_fc->set_argument( zero_string, m_expression.get()));
    MI_CHECK_EQUAL( -1, m_fc->set_argument( "param0", nullptr));
    MI_CHECK_EQUAL( -2, m_fc->set_argument( 1, m_expression.get()));
    MI_CHECK_EQUAL( -2, m_fc->set_argument( "invalid", m_expression.get()));
    mi::base::Handle<mi::neuraylib::IValue> m_value_string( vf->create_string( "foo"));
    m_expression = ef->create_constant( m_value_string.get());
    MI_CHECK_EQUAL( -3, m_fc->set_argument( "param0", m_expression.get()));

    // check set_arguments()/access_arguments()

    m_value_int->set_value( -43);
    m_expression = ef->create_constant( m_value_int.get());
    mi::base::Handle<mi::neuraylib::IExpression_list> m_expression_list(
        ef->create_expression_list());
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "param0", m_expression.get()));
    MI_CHECK_EQUAL( 0, m_fc->set_arguments( m_expression_list.get()));

    c_expression_list = m_fc->get_arguments();
    c_expression = c_expression_list->get_expression<mi::neuraylib::IExpression_constant>(
        "param0");
    c_value_int = c_expression->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( -43, c_value_int->get_value());

    // check error codes of set_arguments()

    MI_CHECK_EQUAL( -1, m_fc->set_arguments( nullptr));

    m_expression_list = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "param0", m_expression.get()));
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "invalid", m_expression.get()));
    MI_CHECK_EQUAL( -2, m_fc->set_arguments( m_expression_list.get()));

    m_expression = ef->create_constant( m_value_string.get());
    m_expression_list = ef->create_expression_list();
    MI_CHECK_EQUAL( 0, m_expression_list->add_expression( "param0", m_expression.get()));
    MI_CHECK_EQUAL( -3, m_fc->set_arguments( m_expression_list.get()));

    // check reset_argument()

    MI_CHECK_EQUAL( 0, m_fc->reset_argument( "param0"));

    c_expression_list = m_fc->get_arguments();
    c_expression = c_expression_list->get_expression<mi::neuraylib::IExpression_constant>(
        "param0");
    c_value_int = c_expression->get_value<mi::neuraylib::IValue_int>();
    MI_CHECK_EQUAL( 42, c_value_int->get_value());

    // check error codes of reset_argument()

    MI_CHECK_EQUAL( -1, m_fc->reset_argument( zero_string));
    MI_CHECK_EQUAL( -2, m_fc->reset_argument( "invalid"));
    MI_CHECK_EQUAL( -2, m_fc->reset_argument( 1));

    // check parameter references

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::fd_parameter_references(float,float,float)"));
    MI_CHECK( c_fd);
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( c_fd->get_defaults());
    m_fc = transaction->edit<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::fc_parameter_references");
    MI_CHECK( m_fc);


    mi::base::Handle<const mi::neuraylib::IExpression> toplevel_reference(
        defaults->get_expression( "b"));
    MI_CHECK_EQUAL( -6, m_fc->set_argument( "b", toplevel_reference.get()));
    MI_CHECK_EQUAL(  0, m_fc->reset_argument( "b"));

    mi::base::Handle<const mi::neuraylib::IExpression> nested_reference(
        defaults->get_expression( "c"));
    MI_CHECK_EQUAL( -6, m_fc->set_argument( "c", nested_reference.get()));
    MI_CHECK_EQUAL(  0, m_fc->reset_argument( "c"));

    {
        // check non-template-like function from builtins module
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::operator+(float,float)"));
        MI_CHECK_EQUAL_CSTR( "operator+(float,float)", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fd->get_parameter_count());

        mi::base::Handle<mi::neuraylib::IExpression_call> expr_x(
            ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
        mi::base::Handle<mi::neuraylib::IValue_float> value_1( vf->create_float( 42.0f));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expr_y(
            ef->create_constant( value_1.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "x", expr_x.get());
        arguments->add_expression( "y", expr_y.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "x", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "y", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            fc->get_parameter_types());
        MI_CHECK_EQUAL( 2, parameter_types->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
            parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_type->get_kind());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
            parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter1_type->get_kind());
        MI_CHECK( !fc->is_declarative());
        MI_CHECK( !fc->is_material());

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator+(float,float)",
                /*argument_types*/ nullptr,
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::operator+(float,float)");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR(
            sfn->get_function_name_without_module_name(), "operator+(float,float)");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator+(float,float)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
            dfn->get_argument_types());
        MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator+(float,float)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
            dfn2->get_argument_types());
        MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module(
            transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check that deserialization loads modules if they have not yet been loaded

        // check that the module has not yet been loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl4"));
        MI_CHECK( !module);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                "mdl::test_mdl4::fd_1(int)",
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::test_mdl4::fd_1(int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
            dfn->get_argument_types());
        MI_CHECK_EQUAL( 1, argument_types->get_size());
        MI_CHECK_EQUAL_CSTR( "param0", argument_types2->get_name( zero_size));
        mi::base::Handle<const mi::neuraylib::IType> arg0(
            argument_types2->get_type<mi::neuraylib::IType>( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, arg0->get_kind());

        // check that the module got loaded as side effect of deserialization
        module = transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl4");
        MI_CHECK( module);
    }
}

void check_field_access( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)",
            /*argument_types*/ nullptr,
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR(
        sfn->get_function_name(),
        "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR(
        sfn->get_function_name_without_module_name(),
        "foo_struct.param_int(::" TEST_MDL "::foo_struct)");

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR(
        dfn->get_db_name(),
                        "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
        dfn->get_argument_types());
    MI_CHECK_EQUAL( 1, argument_types->get_size());
    MI_CHECK_EQUAL_CSTR( "s", argument_types->get_name( zero_size));
    mi::base::Handle<const mi::neuraylib::IType_struct> arg0(
        argument_types->get_type<mi::neuraylib::IType_struct>( zero_size));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", arg0->get_symbol());

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR(
        dfn2->get_db_name(),
        "mdl::" TEST_MDL "::foo_struct.param_int(::" TEST_MDL "::foo_struct)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
        dfn2->get_argument_types());
    MI_CHECK_EQUAL( 1, argument_types2->get_size());
    MI_CHECK_EQUAL_CSTR( "s", argument_types2->get_name( zero_size));
    mi::base::Handle<const mi::neuraylib::IType_struct> arg02(
        argument_types2->get_type<mi::neuraylib::IType_struct>( zero_size));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::foo_struct", arg02->get_symbol());

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::" TEST_MDL);
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::" TEST_MDL);
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);
}

void check_array_constructor(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::T[](...)"));
    MI_CHECK_EQUAL_CSTR( "T[](...)", fd->get_mdl_name());
    mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
    MI_CHECK_EQUAL( 0, fd->get_parameter_count());
    MI_CHECK( !fd->is_material());

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_0(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
    mi::base::Handle<mi::neuraylib::IValue_float> value_1( vf->create_float( 42.0f));
    mi::base::Handle<mi::neuraylib::IExpression_constant> expression_1(
        ef->create_constant( value_1.get()));

    // reversed order to check correct handling of named arguments (was positional arguments
    // some time ago)
    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    arguments->add_expression( "value1", expression_1.get());
    arguments->add_expression( "value0", expression_0.get());

    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( arguments.get(), &result));
    MI_CHECK_EQUAL( 0, result);

    ret_type = fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, ret_type->get_kind());
    MI_CHECK_EQUAL( 2, fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "value0", fc->get_parameter_name( zero_size));
    MI_CHECK_EQUAL_CSTR( "value1", fc->get_parameter_name( 1));
    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        fc->get_parameter_types());
    MI_CHECK_EQUAL( 2, parameter_types->get_size());
    mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
        parameter_types->get_type( zero_size));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_type->get_kind());
    mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
        parameter_types->get_type( 1));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter1_type->get_kind());
    mi::base::Handle<const mi::neuraylib::IExpression_list> c_arguments( fc->get_arguments());
    mi::base::Handle<const mi::neuraylib::IExpression> arg0(
        c_arguments->get_expression( "value0"));
    MI_CHECK_EQUAL( arg0->get_kind(), mi::neuraylib::IExpression::EK_CALL);
    mi::base::Handle<const mi::neuraylib::IExpression> arg1(
        c_arguments->get_expression( "value1"));
    MI_CHECK_EQUAL( arg1->get_kind(), mi::neuraylib::IExpression::EK_CONSTANT);
    MI_CHECK( !fc->is_declarative());
    MI_CHECK( !fc->is_material());

    MI_CHECK_EQUAL( 0, fc->set_argument( zero_size, expression_1.get()));
    MI_CHECK_EQUAL( 0, fc->set_argument( 1, expression_0.get()));

    mi::base::Handle<mi::neuraylib::IValue_double> value_2( vf->create_double( 42.0));
    mi::base::Handle<mi::neuraylib::IExpression_constant> expression_2(
        ef->create_constant( value_2.get()));
    MI_CHECK_EQUAL( -3, fc->set_argument( zero_size, expression_2.get()));
    MI_CHECK_EQUAL( -3, fc->set_argument( 1, expression_2.get()));

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        fc->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::T[](...)",
            orig_argument_types.get(),
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::T[](...)<float,2>");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(), "T[](...)<float,2>");

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::T[](...)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
        dfn->get_argument_types());
    MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::T[](...)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
        dfn2->get_argument_types());
    MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);
}

void check_array_length_operator(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
             "mdl::operator_len(%3C0%3E[])"));
    MI_CHECK_EQUAL_CSTR(  "operator_len(%3C0%3E[])", fd->get_mdl_name());
    mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // really int
    MI_CHECK_EQUAL( 1, fd->get_parameter_count());
    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        fd->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
        parameter_types->get_type( zero_size));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy
    MI_CHECK( !fd->is_material());

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_a(
        ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_int_array"));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    arguments->add_expression( "a", expression_a.get());

    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( arguments.get(), &result));
    MI_CHECK_EQUAL( 0, result);

    ret_type = fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ALIAS, ret_type->get_kind());
    MI_CHECK_EQUAL( mi::neuraylib::IType::MK_UNIFORM, ret_type->get_all_type_modifiers());
    ret_type = ret_type->skip_all_type_aliases();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind());
    MI_CHECK_EQUAL( 1, fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
    parameter_types = fc->get_parameter_types();
    parameter0_type = parameter_types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, parameter0_type->get_kind());
    MI_CHECK( !fc->is_declarative());
    MI_CHECK( !fc->is_material());

    MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_a2(
        ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_struct_array"));
    MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
        ef->create_call( "mdl::" TEST_MDL "::fc_0"));
    MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        fc->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::operator_len(%3C0%3E[])",
            orig_argument_types.get(),
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name(), "mdl::operator_len(%3C0%3E[])<int[N]>");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR(
        sfn->get_function_name_without_module_name(), "operator_len(%3C0%3E[])<int[N]>");

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator_len(%3C0%3E[])");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
        dfn->get_argument_types());
    // IType_factory::compare() does not work due to different array size symbols
    MI_CHECK_EQUAL( 1, argument_types->get_size());
    MI_CHECK_EQUAL_CSTR( "a", argument_types->get_name( zero_size));
    mi::base::Handle<const mi::neuraylib::IType_array> arg0a(
        argument_types->get_type<mi::neuraylib::IType_array>( zero_size));
    MI_CHECK( !arg0a->is_immediate_sized());
    mi::base::Handle<const mi::neuraylib::IType> arg0a_element_type(
        arg0a->get_element_type());
    MI_CHECK_EQUAL( arg0a_element_type->get_kind(), mi::neuraylib::IType::TK_INT);

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator_len(%3C0%3E[])");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
        dfn2->get_argument_types());
    // IType_factory::compare() does not work due to different array size symbols
    MI_CHECK_EQUAL( 1, argument_types2->get_size());
    MI_CHECK_EQUAL_CSTR( "a", argument_types2->get_name( zero_size));
    mi::base::Handle<const mi::neuraylib::IType_array> arg0b(
        argument_types2->get_type<mi::neuraylib::IType_array>( zero_size));
    MI_CHECK( !arg0b->is_immediate_sized());
    mi::base::Handle<const mi::neuraylib::IType> arg0b_element_type(
        arg0b->get_element_type());
    MI_CHECK_EQUAL( arg0b_element_type->get_kind(), mi::neuraylib::IType::TK_INT);

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);
}

void check_array_index_operator(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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
        // check array index operator with immediate-sized array
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_float( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_float3(
            tf->create_vector( type_float.get(), 3));
        mi::base::Handle<const mi::neuraylib::IType_matrix> type_matrix3x3(
            tf->create_matrix( type_float3.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a( vf->create_matrix( type_matrix3x3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a(
            ef->create_constant( value_a.get()));
        mi::base::Handle<mi::neuraylib::IValue_int> value_i( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i(
            ef->create_constant( value_i.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());
        arguments->add_expression( "i", expression_i.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator[](%3C0%3E[],int)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IType> ret_type( fc->get_return_type());
        MI_CHECK_EQUAL( 0, tf->compare( type_float3.get(), ret_type.get()));
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType_matrix> parameter0_type(
            parameter_types->get_type<mi::neuraylib::IType_matrix>( zero_size));
        mi::base::Handle<const mi::neuraylib::IType_vector> parameter0_vector_type(
            parameter0_type->get_element_type());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_elem_type(
            parameter0_vector_type->get_element_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_elem_type->get_kind());
        MI_CHECK_EQUAL( 3, parameter0_vector_type->get_size());
        MI_CHECK_EQUAL( 3, parameter0_type->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
            parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int
        MI_CHECK( !fc->is_declarative());
        MI_CHECK( !fc->is_material());

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<const mi::neuraylib::IType_atomic> type_double( tf->create_double());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_double3(
            tf->create_vector( type_double.get(), 3));
        mi::base::Handle<const mi::neuraylib::IType_matrix> type_double3x3(
            tf->create_matrix( type_double3.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a2( vf->create_matrix( type_double3x3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a2(
            ef->create_constant( value_a2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> value_i2( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i2(
            ef->create_constant( value_i2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "i", expression_i2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator[](%3C0%3E[],int)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator[](%3C0%3E[],int)<float3x3>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator[](%3C0%3E[],int)<float3x3>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
            dfn->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
            dfn2->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module(
            transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array index operator with deferred-sized array
        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator[](%3C0%3E[],int)"));
        MI_CHECK_EQUAL_CSTR(  "operator[](%3C0%3E[],int)", fd->get_mdl_name());
        mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
        MI_CHECK_EQUAL( 2, fd->get_parameter_count());
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            fd->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
            parameter_types->get_type( zero_size));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
            parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int
        MI_CHECK( !fd->is_material());

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_struct_array"));
        mi::base::Handle<mi::neuraylib::IValue_int> value_i( vf->create_int( 42));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i(
            ef->create_constant( value_i.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());
        arguments->add_expression( "i", expression_i.get());

        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        ret_type = fc->get_return_type();
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_STRUCT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", fc->get_parameter_name( 1));
        parameter_types = fc->get_parameter_types();
        parameter0_type = parameter_types->get_type( zero_size);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, parameter0_type->get_kind());
        parameter1_type = parameter_types->get_type( 1);
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int
        MI_CHECK( !fc->is_declarative());
        MI_CHECK( !fc->is_material());

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a2(
            ef->create_call( "mdl::" TEST_MDL "::fc_deferred_returns_int_array"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> value_i2( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i2(
            ef->create_constant( value_i2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "i", expression_i2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator[](%3C0%3E[],int)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator[](%3C0%3E[],int)<::" TEST_MDL "::foo_struct[N]>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator[](%3C0%3E[],int)<::" TEST_MDL "::foo_struct[N]>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
            dfn->get_argument_types());
        // IType_factory::compare() does not work due to different array size symbols
        MI_CHECK_EQUAL( 2, argument_types->get_size());
        MI_CHECK_EQUAL_CSTR( "a", argument_types->get_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", argument_types->get_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_array> arg0a(
            argument_types->get_type<mi::neuraylib::IType_array>( zero_size));
        MI_CHECK( !arg0a->is_immediate_sized());
        mi::base::Handle<const mi::neuraylib::IType> arg0a_element_type(
            arg0a->get_element_type());
        MI_CHECK_EQUAL( arg0a_element_type->get_kind(), mi::neuraylib::IType::TK_STRUCT);
        mi::base::Handle<const mi::neuraylib::IType> arg1a( argument_types->get_type( 1));
        MI_CHECK_EQUAL( arg1a->get_kind(), mi::neuraylib::IType::TK_INT);

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
            dfn2->get_argument_types());
        // IType_factory::compare() does not work due to different array size symbols
        MI_CHECK_EQUAL( 2, argument_types2->get_size());
        MI_CHECK_EQUAL_CSTR( "a", argument_types2->get_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", argument_types2->get_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_array> arg0b(
            argument_types2->get_type<mi::neuraylib::IType_array>( zero_size));
        MI_CHECK( !arg0b->is_immediate_sized());
        mi::base::Handle<const mi::neuraylib::IType> arg0b_element_type(
            arg0b->get_element_type());
        MI_CHECK_EQUAL( arg0b_element_type->get_kind(), mi::neuraylib::IType::TK_STRUCT);
        mi::base::Handle<const mi::neuraylib::IType> arg1b( argument_types2->get_type( 1));
        MI_CHECK_EQUAL( arg1b->get_kind(), mi::neuraylib::IType::TK_INT);

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module(
            transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 1);
    }
    {
        // check array index operator with vector
        mi::base::Handle<const mi::neuraylib::IType_atomic> type_float( tf->create_float());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_float3(
            tf->create_vector( type_float.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a( vf->create_vector( type_float3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a(
            ef->create_constant( value_a.get()));
        mi::base::Handle<mi::neuraylib::IValue_int> value_i( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i(
            ef->create_constant( value_i.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            ef->create_expression_list());
        arguments->add_expression( "a", expression_a.get());
        arguments->add_expression( "i", expression_i.get());

        mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
            transaction->access<mi::neuraylib::IFunction_definition>(
                 "mdl::operator[](%3C0%3E[],int)"));
        mi::base::Handle<mi::neuraylib::IFunction_call> fc(
            fd->create_function_call( arguments.get(), &result));
        MI_CHECK_EQUAL( 0, result);

        mi::base::Handle<const mi::neuraylib::IType> ret_type( fc->get_return_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
        MI_CHECK_EQUAL( 2, fc->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( "a", fc->get_parameter_name( zero_size));
        MI_CHECK_EQUAL_CSTR( "i", fc->get_parameter_name( 1));
        mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType_vector> parameter0_type(
            parameter_types->get_type<mi::neuraylib::IType_vector>( zero_size));
        mi::base::Handle<const mi::neuraylib::IType> parameter0_elem_type(
            parameter0_type->get_element_type());
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter0_elem_type->get_kind());
        MI_CHECK_EQUAL( 3, parameter0_type->get_size());
        mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
            parameter_types->get_type( 1));
        MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // really int
        MI_CHECK( !fc->is_declarative());
        MI_CHECK( !fc->is_material());

        MI_CHECK_EQUAL( 0, fc->set_argument( "a", expression_a.get()));

        mi::base::Handle<const mi::neuraylib::IType_atomic> type_int( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType_vector> type_int3(
            tf->create_vector( type_int.get(), 3));
        mi::base::Handle<mi::neuraylib::IValue> value_a2( vf->create_vector( type_int3.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_a2(
            ef->create_constant( value_a2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a2.get()));

        mi::base::Handle<mi::neuraylib::IExpression_call> expression_a3(
            ef->create_call( "mdl::" TEST_MDL "::fc_0"));
        MI_CHECK_EQUAL( -3, fc->set_argument( "a", expression_a3.get()));

        mi::base::Handle<mi::neuraylib::IValue_float> value_i2( vf->create_float( 42.0));
        mi::base::Handle<mi::neuraylib::IExpression_constant> expression_i2(
            ef->create_constant( value_i2.get()));
        MI_CHECK_EQUAL( -3, fc->set_argument( "i", expression_i2.get()));

        mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
            fc->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
            mdl_impexp_api->serialize_function_name(
                "mdl::operator[](%3C0%3E[],int)",
                orig_argument_types.get(),
                /*return_type*/ nullptr,
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
            "mdl::operator[](%3C0%3E[],int)<float3>");
        MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
            "operator[](%3C0%3E[],int)<float3>");

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
            mdl_impexp_api->deserialize_function_name(
                transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
            dfn->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
            mdl_impexp_api->deserialize_function_name(
                transaction,
                sfn->get_module_name(),
                sfn->get_function_name_without_module_name(),
                /*mdle_callback*/ nullptr,
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator[](%3C0%3E[],int)");
        mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
            dfn2->get_argument_types());
        MI_CHECK_EQUAL( 0, tf->compare( orig_argument_types.get(), argument_types.get()));

        mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
            mdl_impexp_api->deserialize_module_name(
                sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
        MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
        mi::base::Handle<const mi::neuraylib::IModule> c_module(
            transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
        result = mdl_impexp_api->load_module(
            transaction, dmn->get_load_module_argument(), context.get());
        MI_CHECK_CTX( context);
        MI_CHECK_EQUAL( result, 1);
    }
}

void check_ternary_operator(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
             "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)"));
    MI_CHECK_EQUAL_CSTR(  "operator%3F(bool,%3C0%3E,%3C0%3E)", fd->get_mdl_name());
    MI_CHECK_EQUAL_CSTR(  "operator%3F", fd->get_mdl_simple_name());
    mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
    MI_CHECK_EQUAL( 3, fd->get_parameter_count());
    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        fd->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
        parameter_types->get_type( zero_size));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_BOOL, parameter0_type->get_kind());
    mi::base::Handle<const mi::neuraylib::IType> parameter1_type(
        parameter_types->get_type( 1));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter1_type->get_kind()); // dummy
    mi::base::Handle<const mi::neuraylib::IType> parameter2_type(
        parameter_types->get_type( 2));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter2_type->get_kind()); // dummy
    MI_CHECK( !fd->is_material());

    mi::base::Handle<mi::neuraylib::IValue_bool> value_cond( vf->create_bool( true));
    mi::base::Handle<mi::neuraylib::IValue_float> value_true( vf->create_float( 42.0));
    mi::base::Handle<mi::neuraylib::IValue_float> value_false( vf->create_float( 43.0));

    mi::base::Handle<mi::neuraylib::IExpression_constant> expression_cond(
        ef->create_constant( value_cond.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> expression_true(
        ef->create_constant( value_true.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> expression_false(
        ef->create_constant( value_false.get()));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    arguments->add_expression( "cond", expression_cond.get());
    arguments->add_expression( "true_exp", expression_true.get());
    arguments->add_expression( "false_exp", expression_false.get());

    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( arguments.get(), &result));
    MI_CHECK_EQUAL( 0, result);

    ret_type = fc->get_return_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, ret_type->get_kind());
    MI_CHECK_EQUAL( 3, fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "cond", fc->get_parameter_name( zero_size));
    MI_CHECK_EQUAL_CSTR( "true_exp", fc->get_parameter_name( 1));
    MI_CHECK_EQUAL_CSTR( "false_exp", fc->get_parameter_name( 2));
    parameter_types = fc->get_parameter_types();
    parameter0_type = parameter_types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_BOOL, parameter0_type->get_kind());
    parameter1_type = parameter_types->get_type( 1);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter1_type->get_kind());
    parameter2_type = parameter_types->get_type( 2);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_FLOAT, parameter2_type->get_kind());
    MI_CHECK( !fc->is_declarative());
    MI_CHECK( !fc->is_material());

    value_cond->set_value( false);
    MI_CHECK_EQUAL( 0, fc->set_argument( "cond", expression_cond.get()));
    MI_CHECK_EQUAL( -3, fc->set_argument( "cond", expression_true.get()));

    MI_CHECK_EQUAL( 0, fc->set_argument( "true_exp", expression_false.get()));
    MI_CHECK_EQUAL( -3, fc->set_argument( "true_exp", expression_cond.get()));

    MI_CHECK_EQUAL( 0, fc->set_argument( "false_exp", expression_true.get()));
    MI_CHECK_EQUAL( -3, fc->set_argument( "false_exp", expression_cond.get()));

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        fc->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)",
            orig_argument_types.get(),
            /*return_type*/ nullptr,
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR(
        sfn->get_function_name(), "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)<float>");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
        "operator%3F(bool,%3C0%3E,%3C0%3E)<float>");

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types(
        dfn->get_argument_types());
    MI_CHECK( tf->compare( argument_types.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
        dfn2->get_argument_types());
    MI_CHECK( tf->compare( argument_types2.get(), orig_argument_types.get()) == 0);

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);
}

void check_cast_operator(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
             "mdl::operator_cast(%3C0%3E)"));
    MI_CHECK_EQUAL_CSTR(  "operator_cast(%3C0%3E)", fd->get_mdl_name());
    mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
    MI_CHECK_EQUAL( 1, fd->get_parameter_count());
    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        fd->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
        parameter_types->get_type( zero_size));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy
    MI_CHECK( !fd->is_material());

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum"));
    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast_return(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum2"));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    arguments->add_expression( "cast", expression_cast.get());
    arguments->add_expression( "cast_return", expression_cast_return.get());

    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( arguments.get(), &result));
    MI_CHECK_EQUAL( 0, result);

    ret_type = fc->get_return_type();
    mi::base::Handle<const mi::neuraylib::IType_enum> arg1a(
        ret_type->get_interface<mi::neuraylib::IType_enum>());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum2", arg1a->get_symbol());
    MI_CHECK_EQUAL( 1, fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "cast", fc->get_parameter_name( zero_size));
    parameter_types = fc->get_parameter_types();
    MI_CHECK_EQUAL( 1, parameter_types->get_size());
    parameter0_type = parameter_types->get_type( zero_size);
    mi::base::Handle<const mi::neuraylib::IType_enum> arg0a(
        parameter0_type->get_interface<mi::neuraylib::IType_enum>());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", arg0a->get_symbol());
    MI_CHECK( !fc->is_declarative());
    MI_CHECK( !fc->is_material());

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast2(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum"));
    MI_CHECK_EQUAL( 0, fc->set_argument( "cast", expression_cast2.get()));

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast3(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_enum2"));
    MI_CHECK_EQUAL( 0, fc->set_argument( "cast", expression_cast3.get()));

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast4(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
    MI_CHECK_EQUAL( -3, fc->set_argument( "cast", expression_cast4.get()));

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        fc->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> return_type(
        fc->get_return_type());
    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::operator_cast(%3C0%3E)",
            orig_argument_types.get(),
            return_type.get(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name(),
        "mdl::operator_cast(%3C0%3E)<::" TEST_MDL "::Enum,::" TEST_MDL "::Enum2>");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
        "operator_cast(%3C0%3E)<::" TEST_MDL "::Enum,::" TEST_MDL "::Enum2>");

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator_cast(%3C0%3E)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_tpes( dfn->get_argument_types());
    // IType_factory::compare() does not work since the parameter type list of the function
    // definition has only one argument
    MI_CHECK_EQUAL( 2, argument_tpes->get_size());
    MI_CHECK_EQUAL_CSTR( "cast", argument_tpes->get_name( zero_size));
    MI_CHECK_EQUAL_CSTR( "cast_return", argument_tpes->get_name( 1));
    mi::base::Handle<const mi::neuraylib::IType_enum> arg0b(
        argument_tpes->get_type<mi::neuraylib::IType_enum>( zero_size));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", arg0b->get_symbol());
    mi::base::Handle<const mi::neuraylib::IType_enum> arg1b(
        argument_tpes->get_type<mi::neuraylib::IType_enum>( 1));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum2", arg1b->get_symbol());

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator_cast(%3C0%3E)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
        dfn2->get_argument_types());
    // IType_factory::compare() does not work since the parameter type list of the function
    // call has only one argument
    MI_CHECK_EQUAL( 2, argument_types2->get_size());
    MI_CHECK_EQUAL_CSTR( "cast", argument_types2->get_name( zero_size));
    MI_CHECK_EQUAL_CSTR( "cast_return", argument_types2->get_name( 1));
    mi::base::Handle<const mi::neuraylib::IType_enum> arg0c(
        argument_types2->get_type<mi::neuraylib::IType_enum>( zero_size));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum", arg0c->get_symbol());
    mi::base::Handle<const mi::neuraylib::IType_enum> arg1c(
        argument_types2->get_type<mi::neuraylib::IType_enum>( 1));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::Enum2", arg1c->get_symbol());

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);
}

void check_decl_cast_operator(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
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

    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>(
             "mdl::operator_decl_cast(%3C0%3E)"));
    MI_CHECK_EQUAL_CSTR(  "operator_decl_cast(%3C0%3E)", fd->get_mdl_name());
    mi::base::Handle<const mi::neuraylib::IType> ret_type( fd->get_return_type());
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, ret_type->get_kind()); // dummy
    MI_CHECK_EQUAL( 1, fd->get_parameter_count());
    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        fd->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> parameter0_type(
        parameter_types->get_type( zero_size));
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, parameter0_type->get_kind()); // dummy
    MI_CHECK( !fd->is_material());
    MI_CHECK( fd->is_declarative());

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast(
        ef->create_call( "mdl::" TEST_MDL "::mi_0"));
    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast_return(
        ef->create_call( "mdl::" TEST_MDL "::mi_aov"));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        ef->create_expression_list());
    arguments->add_expression( "cast", expression_cast.get());
    arguments->add_expression( "cast_return", expression_cast_return.get());

    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        fd->create_function_call( arguments.get(), &result));
    MI_CHECK_EQUAL( 0, result);

    ret_type = fc->get_return_type();
    mi::base::Handle<const mi::neuraylib::IType_struct> arg1a(
        ret_type->get_interface<mi::neuraylib::IType_struct>());
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::aov_material", arg1a->get_symbol());
    MI_CHECK_EQUAL( 1, fc->get_parameter_count());
    MI_CHECK_EQUAL_CSTR( "cast", fc->get_parameter_name( zero_size));
    parameter_types = fc->get_parameter_types();
    MI_CHECK_EQUAL( 1, parameter_types->get_size());
    parameter0_type = parameter_types->get_type( zero_size);
    mi::base::Handle<const mi::neuraylib::IType_struct> arg0a(
        parameter0_type->get_interface<mi::neuraylib::IType_struct>());
    MI_CHECK_EQUAL_CSTR( "::material", arg0a->get_symbol());
    MI_CHECK( fc->is_declarative());
    MI_CHECK( !fc->is_material());

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast2(
        ef->create_call( "mdl::" TEST_MDL "::mi_1"));
    MI_CHECK_EQUAL( 0, fc->set_argument( "cast", expression_cast2.get()));

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast3(
        ef->create_call( "mdl::" TEST_MDL "::mi_aov"));
    MI_CHECK_EQUAL( -3, fc->set_argument( "cast", expression_cast3.get()));

    mi::base::Handle<mi::neuraylib::IExpression_call> expression_cast4(
        ef->create_call( "mdl::" TEST_MDL "::fc_ret_float"));
    MI_CHECK_EQUAL( -3, fc->set_argument( "cast", expression_cast4.get()));

    mi::base::Handle<const mi::neuraylib::IType_list> orig_argument_types(
        fc->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> return_type(
        fc->get_return_type());
    mi::base::Handle<const mi::neuraylib::ISerialized_function_name> sfn(
        mdl_impexp_api->serialize_function_name(
            "mdl::operator_decl_cast(%3C0%3E)",
            orig_argument_types.get(),
            return_type.get(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR(
        sfn->get_function_name(),
        "mdl::operator_decl_cast(%3C0%3E)<material,::" TEST_MDL "::aov_material>");
    MI_CHECK_EQUAL_CSTR( sfn->get_module_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( sfn->get_function_name_without_module_name(),
        "operator_decl_cast(%3C0%3E)<material,::" TEST_MDL "::aov_material>");

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn(
        mdl_impexp_api->deserialize_function_name(
            transaction, sfn->get_function_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn->get_db_name(), "mdl::operator_decl_cast(%3C0%3E)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_tpes( dfn->get_argument_types());
    // IType_factory::compare() does not work since the parameter type list of the function
    // call has only one argument
    MI_CHECK_EQUAL( 2, argument_tpes->get_size());
    MI_CHECK_EQUAL_CSTR( "cast", argument_tpes->get_name( zero_size));
    MI_CHECK_EQUAL_CSTR( "cast_return", argument_tpes->get_name( 1));
    mi::base::Handle<const mi::neuraylib::IType_struct> arg0b(
        argument_tpes->get_type<mi::neuraylib::IType_struct>( zero_size));
    MI_CHECK_EQUAL_CSTR( "::material", arg0b->get_symbol());
    mi::base::Handle<const mi::neuraylib::IType_struct> arg1b(
        argument_tpes->get_type<mi::neuraylib::IType_struct>( 1));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::aov_material", arg1b->get_symbol());

    mi::base::Handle<const mi::neuraylib::IDeserialized_function_name> dfn2(
        mdl_impexp_api->deserialize_function_name(
            transaction,
            sfn->get_module_name(),
            sfn->get_function_name_without_module_name(),
            /*mdle_callback*/ nullptr,
            context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dfn2->get_db_name(), "mdl::operator_decl_cast(%3C0%3E)");
    mi::base::Handle<const mi::neuraylib::IType_list> argument_types2(
        dfn2->get_argument_types());
    // IType_factory::compare() does not work since the parameter type list of the function
    // definition has only one argument
    MI_CHECK_EQUAL( 2, argument_types2->get_size());
    MI_CHECK_EQUAL_CSTR( "cast", argument_types2->get_name( zero_size));
    MI_CHECK_EQUAL_CSTR( "cast_return", argument_types2->get_name( 1));
    mi::base::Handle<const mi::neuraylib::IType_struct> arg0c(
        argument_tpes->get_type<mi::neuraylib::IType_struct>( zero_size));
    MI_CHECK_EQUAL_CSTR( "::material", arg0c->get_symbol());
    mi::base::Handle<const mi::neuraylib::IType_struct> arg1c(
        argument_tpes->get_type<mi::neuraylib::IType_struct>( 1));
    MI_CHECK_EQUAL_CSTR( "::" TEST_MDL "::aov_material", arg1c->get_symbol());

    mi::base::Handle<const mi::neuraylib::IDeserialized_module_name> dmn(
        mdl_impexp_api->deserialize_module_name(
            sfn->get_module_name(), /*mdle_callback*/ nullptr, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL_CSTR( dmn->get_db_name(), "mdl::%3Cbuiltins%3E");
    MI_CHECK_EQUAL_CSTR( dmn->get_load_module_argument(), "::%3Cbuiltins%3E");
    mi::base::Handle<const mi::neuraylib::IModule> c_module(
        transaction->access<mi::neuraylib::IModule>( dmn->get_db_name()));
    result = mdl_impexp_api->load_module(
        transaction, dmn->get_load_module_argument(), context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 1);
}

void check_create_cast(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType_enum> enum_type(
        tf->create_enum( "::" TEST_MDL "::Enum"));
    mi::base::Handle<const mi::neuraylib::IType_enum> enum2_type(
        tf->create_enum( "::" TEST_MDL "::Enum2"));

    mi::base::Handle<mi::neuraylib::IValue_enum> enum_value(
        vf->create_enum( enum_type.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> enum_expr(
        ef->create_constant( enum_value.get()));

    {
        // regular case
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                enum2_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
    }
    const char* db_name = "create_cast_test";
    {
        // regular case, DB name does not yet exist
        mi::base::Handle<const mi::base::IInterface> db_element(
            transaction->access( db_name));
        MI_CHECK( !db_element);

        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                enum2_type.get(),
                db_name,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        MI_CHECK_EQUAL_CSTR( decl_cast_call->get_call(), db_name);
    }
    {
        // regular case, direct call
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                enum2_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                /*direct_call*/ true,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
    }
    {
        // regular case, DB name exists already
        mi::base::Handle<const mi::base::IInterface> db_element(
            transaction->access( db_name));
        MI_CHECK( db_element);

        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                enum2_type.get(),
                db_name,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        MI_CHECK_NOT_EQUAL_CSTR( decl_cast_call->get_call(), db_name);
    }
    {
        // identical types, force_cast is false
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                enum_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
        MI_CHECK_EQUAL( decl_cast, enum_expr);
    }
    {
        // identical types, force_cast is true
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                enum_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
        MI_CHECK_NOT_EQUAL( decl_cast, enum_expr);
    }
    {
        // invalid input
        result = -2;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                nullptr,
                enum_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !decl_cast);
    }
    {
        // invalid input
        result = -2;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                nullptr,
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !decl_cast);
    }
    {
        // different struct categories
        mi::base::Handle<const mi::neuraylib::IType_struct> bar_struct_type(
            tf->create_struct( "::" TEST_MDL "::bar_struct"));

        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_cast(
                enum_expr.get(),
                bar_struct_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !decl_cast);
    }
}

void check_create_decl_cast(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType_struct> aov_material_type(
        tf->create_struct( "::" TEST_MDL "::aov_material"));
    mi::base::Handle<const mi::neuraylib::IType_struct> material_type(
        tf->create_struct( "::material"));

    mi::base::Handle<mi::neuraylib::IValue_struct> aov_material_value(
        vf->create_struct( aov_material_type.get()));
    mi::base::Handle<mi::neuraylib::IExpression_constant> aov_material_expr(
        ef->create_constant( aov_material_value.get()));

    {
        // regular case
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                material_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
            transaction->access<mi::neuraylib::IFunction_call>( decl_cast_call->get_call()));
        MI_CHECK( !fc->is_material());
        MI_CHECK( fc->is_declarative());
    }
    const char* db_name = "create_decl_cast_test";
    {
        // regular case, DB name does not yet exist
        mi::base::Handle<const mi::base::IInterface> db_element(
            transaction->access( db_name));
        MI_CHECK( !db_element);

        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                material_type.get(),
                db_name,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        MI_CHECK_EQUAL_CSTR( decl_cast_call->get_call(), db_name);
        mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
            transaction->access<mi::neuraylib::IFunction_call>( decl_cast_call->get_call()));
        MI_CHECK( !fc->is_material());
        MI_CHECK( fc->is_declarative());
    }
    {
        // regular case, direct call
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                material_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                /*direct_call*/ true,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
    }
    {
        // regular case, DB name exists already
        mi::base::Handle<const mi::base::IInterface> db_element(
            transaction->access( db_name));
        MI_CHECK( db_element);

        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                material_type.get(),
                db_name,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        MI_CHECK_NOT_EQUAL_CSTR( decl_cast_call->get_call(), db_name);
    }
    {
        // identical types, force_cast is false
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                aov_material_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
        MI_CHECK_EQUAL( decl_cast, aov_material_expr);
    }
    {
        // identical types, force_cast is true
        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                aov_material_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);
        MI_CHECK_NOT_EQUAL( decl_cast, aov_material_expr);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
            transaction->access<mi::neuraylib::IFunction_call>( decl_cast_call->get_call()));
        MI_CHECK( !fc->is_material());
        MI_CHECK( fc->is_declarative());
    }
    {
        // regular case, struct not from material category (simpler setup), force_cast is true
        mi::base::Handle<const mi::neuraylib::IType_struct> bar_struct_type(
            tf->create_struct( "::" TEST_MDL "::bar_struct"));
        mi::base::Handle<mi::neuraylib::IValue_struct> bar_struct_value(
            vf->create_struct( bar_struct_type.get()));
        mi::base::Handle<mi::neuraylib::IExpression_constant> bar_struct_expr(
            ef->create_constant( bar_struct_value.get()));

        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                bar_struct_expr.get(),
                bar_struct_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK( decl_cast);

        mi::base::Handle<mi::neuraylib::IExpression_call> decl_cast_call(
            decl_cast->get_interface<mi::neuraylib::IExpression_call>());
        mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
            transaction->access<mi::neuraylib::IFunction_call>( decl_cast_call->get_call()));
        MI_CHECK( !fc->is_material());
    }
    {
        // invalid input
        result = -2;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                nullptr,
                aov_material_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !decl_cast);
    }
    {
        // invalid input
        result = -2;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                nullptr,
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !decl_cast);
    }
    {
        // different struct categories
        mi::base::Handle<const mi::neuraylib::IType_struct> bar_struct_type(
            tf->create_struct( "::" TEST_MDL "::bar_struct"));

        result = -1;
        mi::base::Handle<mi::neuraylib::IExpression> decl_cast(
            ef->create_decl_cast(
                aov_material_expr.get(),
                bar_struct_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ true,
                /*direct_call*/ false,
                &result));
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !decl_cast);
    }
}

void check_mdl_export_reimport(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    Exreimp_data modules[] = {
        { "mdl::" TEST_MDL, "::" TEST_MDL, 0, 0 },
        { "mdl::test_mdl2", "::test_mdl2", 0, 0 },
        { "mdl::test_mdl3", "::test_mdl3", 0, 0 },
        { "mdl::test_mdl4", "::test_mdl4", 0, 0 },
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

        // Tests that do not need any MDL module loaded.
        check_imdl_factory( neuray);
        check_imdl_impexp_api( neuray);

        // Load the primary modules and check the IModule interface.
        load_primary_modules( transaction.get(), neuray);
        check_itransaction( transaction.get());
        check_imodule_part1( transaction.get(), neuray);
        check_enums( transaction.get(), mdl_factory.get());
        check_structs( transaction.get());
        check_overload_resolution( transaction.get(), neuray);

        // Create function calls used by many later tests.
        create_function_calls( transaction.get(), mdl_factory.get());

        // Check the IFunction_definition and IFunction_call interfaces.
        check_ifunction_definition( transaction.get(), mdl_factory.get());
        check_ifunction_call( transaction.get(), neuray);

        // Check the field access and template functions.
        check_field_access( transaction.get(), neuray);
        check_array_constructor( transaction.get(), neuray);
        check_array_length_operator( transaction.get(), neuray);
        check_array_index_operator( transaction.get(), neuray);
        check_ternary_operator( transaction.get(), neuray);
        check_cast_operator( transaction.get(), neuray);
        check_decl_cast_operator( transaction.get(), neuray);
        check_create_cast( transaction.get(), neuray);
        check_create_decl_cast( transaction.get(), neuray);

        // Export/re-import all modules created so far.
        check_mdl_export_reimport( transaction.get(), neuray);

        MI_CHECK_EQUAL( 0, transaction->commit());

        uninstall_external_resolver( neuray);
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_imodule_basics )
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
        set_mdl_paths( neuray.get(), {path1, path2});

        // load plugins
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_dds));
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        run_tests( neuray.get());
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

