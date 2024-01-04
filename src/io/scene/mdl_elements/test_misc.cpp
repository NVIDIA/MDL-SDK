/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

// Note: most tests for this module are in prod/lib/neuray/test_imdl_module.cpp.
// This directory contains additional tests, sometimes for internal functionality that is not
// directly (or easily) accessible via the API.

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/scene/mdl_elements"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <tuple>

#include "i_mdl_elements_compiled_material.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_utilities.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_utilities.h"
#include "test_shared.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/istring.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_distiller_rules.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/time/i_time.h>
#include <base/hal/thread/i_thread_thread.h>
#include <base/hal/thread/i_thread_condition.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_target.h>
#include <base/lib/path/i_path.h>
#include <base/lib/plug/i_plug.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_scope.h>
#include <base/data/db/i_db_transaction.h>
#include <mdl/compiler/compilercore/compilercore_comparator.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>
#include <prod/lib/neuray/test_shared.h> // for plugin_path_openimageio

using namespace MI;

#undef MI_CHECK_CTX
#undef MI_CHECK_CTX_RESULT

// Checks that there are no error messages (and dumps them otherwise).
#define MI_CHECK_CTX( context) \
    log_messages( context); \
    MI_CHECK_EQUAL( context->get_error_messages_count(), 0);

// Checks that the context result matches the given result.
#define MI_CHECK_CTX_RESULT( context, result) \
    MI_CHECK_EQUAL( context->get_result(), result);

// Check the default compiled material
void test_get_default_compiled_material( DB::Transaction* transaction)
{
    MDL::Mdl_compiled_material* cm = MDL::get_default_compiled_material( transaction);
    MI_CHECK( cm);
    delete cm;
}

void check_texture(
    DB::Transaction* transaction,
    const MDL::IExpression_list* expressions,
    const char* expr_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    mi::base::Handle<const MDL::IExpression> expr( expressions->get_expression( expr_name));
    MI_CHECK( expr);
    mi::base::Handle<const MDL::IExpression_constant> constant(
        expr->get_interface<MDL::IExpression_constant>());
    mi::base::Handle<const MDL::IValue_resource> resource(
        constant->get_value<MDL::IValue_resource>());
    DB::Tag resource_tag = resource->get_value();

    if( !success) {
        MI_CHECK( !resource_tag);
        return;
    }

    MI_CHECK( !!resource_tag == !!resolve_resources);

    if( !file_path)
        MI_CHECK( !resource->get_file_path( transaction));
    else
        MI_CHECK_EQUAL_CSTR( resource->get_file_path( transaction), file_path);

    if( !resolve_resources)
        return;

    DB::Access<TEXTURE::Texture> texture( resource_tag, transaction);
    DB::Tag image_tag = texture->get_image();
    MI_CHECK( image_tag);
    DB::Access<DBIMAGE::Image> image( image_tag, transaction);
    MI_CHECK( image->get_original_filename().empty());
}

void check_texture_def(
    DB::Transaction* transaction,
    const char* db_element_name,
    const char* arg_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    DB::Tag tag = transaction->name_to_tag( db_element_name);
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments( fd->get_defaults());
    check_texture( transaction, arguments.get(), arg_name, success, resolve_resources, file_path);
}

void check_texture_call(
    DB::Transaction* transaction,
    const char* db_element_name,
    const char* arg_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    DB::Tag tag = transaction->name_to_tag( db_element_name);
    DB::Access<MDL::Mdl_function_call> fc( tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments( fc->get_arguments());
    check_texture( transaction, arguments.get(), arg_name, success, resolve_resources, file_path);
}

void check_light_profile(
    DB::Transaction* transaction,
    const MDL::IExpression_list* expressions,
    const char* expr_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    mi::base::Handle<const MDL::IExpression> expr( expressions->get_expression( expr_name));
    MI_CHECK( expr);
    mi::base::Handle<const MDL::IExpression_constant> constant(
        expr->get_interface<MDL::IExpression_constant>());
    mi::base::Handle<const MDL::IValue_resource> resource(
        constant->get_value<MDL::IValue_resource>());
    DB::Tag resource_tag = resource->get_value();

    if( !success) {
        MI_CHECK( !resource_tag);
        return;
    }

    MI_CHECK( !!resource_tag == !!resolve_resources);

    if( !file_path)
        MI_CHECK( !resource->get_file_path( transaction));
    else
        MI_CHECK_EQUAL_CSTR( resource->get_file_path( transaction), file_path);

    if( !resolve_resources)
        return;

    DB::Access<LIGHTPROFILE::Lightprofile> lightprofile( resource_tag, transaction);
    MI_CHECK( lightprofile->get_original_filename().empty());
}

void check_light_profile_def(
    DB::Transaction* transaction,
    const char* db_element_name,
    const char* arg_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    DB::Tag tag = transaction->name_to_tag( db_element_name);
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments( fd->get_defaults());
    check_light_profile(
       transaction, arguments.get(), arg_name, success, resolve_resources, file_path);
}

void check_light_profile_call(
    DB::Transaction* transaction,
    const char* db_element_name,
    const char* arg_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    DB::Tag tag = transaction->name_to_tag( db_element_name);
    DB::Access<MDL::Mdl_function_call> fc( tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments( fc->get_arguments());
    check_light_profile(
       transaction, arguments.get(), arg_name, success, resolve_resources, file_path);
}

void check_bsdf_measurement(
    DB::Transaction* transaction,
    const MDL::IExpression_list* expressions,
    const char* expr_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    mi::base::Handle<const MDL::IExpression> expr( expressions->get_expression( expr_name));
    MI_CHECK( expr);
    mi::base::Handle<const MDL::IExpression_constant> constant(
        expr->get_interface<MDL::IExpression_constant>());
    mi::base::Handle<const MDL::IValue_resource> resource(
        constant->get_value<MDL::IValue_resource>());
    DB::Tag resource_tag = resource->get_value();

    if( !success) {
        MI_CHECK( !resource_tag);
        return;
    }

    MI_CHECK( !!resource_tag == !!resolve_resources);

    if( !file_path)
        MI_CHECK( !resource->get_file_path( transaction));
    else
        MI_CHECK_EQUAL_CSTR( resource->get_file_path( transaction), file_path);

    if( !resolve_resources)
        return;

    DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( resource_tag, transaction);
    MI_CHECK( bsdf_measurement->get_original_filename().empty());
}

void check_bsdf_measurement_def(
    DB::Transaction* transaction,
    const char* db_element_name,
    const char* arg_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    DB::Tag tag = transaction->name_to_tag( db_element_name);
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments( fd->get_defaults());
    check_bsdf_measurement(
       transaction, arguments.get(), arg_name, success, resolve_resources, file_path);
}

void check_bsdf_measurement_call(
    DB::Transaction* transaction,
    const char* db_element_name,
    const char* arg_name,
    bool success,
    bool resolve_resources,
    const char* file_path)
{
    DB::Tag tag = transaction->name_to_tag( db_element_name);
    DB::Access<MDL::Mdl_function_call> fc( tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments( fc->get_arguments());
    check_bsdf_measurement(
       transaction, arguments.get(), arg_name, success, resolve_resources, file_path);
}

void test_resources( DB::Transaction* transaction)
{
    // invalid resources
    check_texture_def( transaction,
        "mdl::mdl_elements::test_resolver_failures::fd_texture_invalid(texture_2d)",
        "t", false, true, nullptr);
    check_light_profile_def( transaction,
        "mdl::mdl_elements::test_resolver_failures::fd_light_profile_invalid(light_profile)",
        "l", false, true, nullptr);
    check_bsdf_measurement_def( transaction,
        "mdl::mdl_elements::test_resolver_failures::fd_bsdf_measurement_invalid(bsdf_measurement)",
        "b", false, true, nullptr);

    // resolver failures
    check_texture_def( transaction,
        "mdl::mdl_elements::test_resolver_failures::fd_texture_failure(texture_2d)",
        "t", false, true, nullptr);
    check_light_profile_def( transaction,
        "mdl::mdl_elements::test_resolver_failures::fd_light_profile_failure(light_profile)",
        "l", false, true, nullptr);
    check_bsdf_measurement_def( transaction,
        "mdl::mdl_elements::test_resolver_failures::fd_bsdf_measurement_failure(bsdf_measurement)",
        "b", false, true, nullptr);

    // resolver succeeds
    check_texture_def( transaction, // weak-relative file path
        "mdl::mdl_elements::test_resolver_success::fd_texture_success(texture_2d)", "t",
        true, true, "/mdl_elements/resources/test.png");
    check_texture_call( transaction,
        "mdl::mdl_elements::test_resolver_success::fc_texture_success", "t",
        true, true, "/mdl_elements/resources/test.png");

    check_texture_def (transaction, // weak-relative file path, udim image sequence
        "mdl::mdl_elements::test_resolver_success::fd_texture_udim_success(texture_2d)", "t",
        true, true, "/mdl_elements/resources/test<UDIM>.png");
    check_texture_call( transaction,
        "mdl::mdl_elements::test_resolver_success::fc_texture_udim_success", "t",
        true, true, "/mdl_elements/resources/test<UDIM>.png");

    check_light_profile_def( transaction, // strict-relative file path
        "mdl::mdl_elements::test_resolver_success::fd_light_profile_success(light_profile)", "l",
        true, true, "/mdl_elements/resources/test.ies");
    check_light_profile_call( transaction,
        "mdl::mdl_elements::test_resolver_success::fc_light_profile_success", "l",
        true, true, "/mdl_elements/resources/test.ies");

    check_bsdf_measurement_def( transaction, // absolute path
        "mdl::mdl_elements::test_resolver_success::fd_bsdf_measurement_success(bsdf_measurement)", "b",
        true, true, "/mdl_elements/resources/test.mbsdf");
    check_bsdf_measurement_call( transaction,
        "mdl::mdl_elements::test_resolver_success::fc_bsdf_measurement_success", "b",
        true, true, "/mdl_elements/resources/test.mbsdf");
}

void test_jitted_environment_function( DB::Transaction* transaction)
{
    DB::Tag fc_tag = transaction->name_to_tag( "mdl::mdl_elements::test_misc::fc_color");
    DB::Access<MDL::Mdl_function_call> fc( fc_tag, transaction);

    mi::Sint32 errors;
    mi::base::Handle<mi::mdl::IGenerated_code_lambda_function> lf( fc->create_jitted_function(
        transaction, /*environment_context*/ true, /*mdl_meters_per_scene_unit*/ 1.0f,
        /*mdl_wavelength_min*/ 380.0f, /*mdl_wavelength_max*/ 780.0f, &errors));
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( lf.is_valid_interface());

    lf->init( transaction, nullptr, nullptr);
    mi::mdl::RGB_color result;
    mi::mdl::Shading_state_environment state;
    state.direction.x = 0.1f;
    state.direction.y = 0.2f;
    state.direction.z = 0.3f;
    lf->run_environment( 0, &result, &state);
    lf->term();
    MI_CHECK( result.r == 0.1f && result.g == 0.2f && result.b == 0.3f);
}

// Compute hash for compiled material of material instance mi_tag in instance and class compilation
// mode.
void get_hash_values(
    DB::Transaction* transaction, DB::Tag mi_tag, mi::base::Uuid& hash_ic, mi::base::Uuid& hash_cc,
    MDL::Execution_context* context)
{
    DB::Access<MDL::Mdl_function_call> mi( mi_tag, transaction);
    MDL::Mdl_compiled_material* cm_ic = mi->create_compiled_material(
        transaction, /*class_compilation*/ false, context);
    MDL::Mdl_compiled_material* cm_cc = mi->create_compiled_material(
        transaction, /*class_compilation*/ true, context);
    MI_CHECK( cm_ic);
    MI_CHECK( cm_cc);
    MI_CHECK_EQUAL( 0, cm_ic->get_parameter_count());
    MI_CHECK_EQUAL( 1, cm_cc->get_parameter_count());
    hash_ic = cm_ic->get_hash();
    hash_cc = cm_cc->get_hash();
    delete cm_ic;
    delete cm_cc;
}

// Check that gamma changes of texture are taken into account for hashes (MDL file path still
// valid).
void test_resources_and_hashes_edit_gamma(
    DB::Transaction* transaction, MDL::Execution_context* context)
{
    DB::Tag mi_tag = transaction->name_to_tag( "mdl::mdl_elements::test_misc::mi_textured");

    // get original hash values
    mi::base::Uuid hash_ic, hash_cc;
    get_hash_values( transaction, mi_tag, hash_ic, hash_cc, context);

    // edit texture to modify gamma
    {
        DB::Access<MDL::Mdl_function_call> mi( mi_tag, transaction);
        mi::base::Handle<const MDL::IExpression_list> arguments( mi->get_arguments());
        mi::base::Handle<const MDL::IExpression> argument( arguments->get_expression( "t"));
        mi::base::Handle<const MDL::IExpression_constant> argument_constant(
            argument->get_interface<MDL::IExpression_constant>());
        mi::base::Handle<const MDL::IValue_texture> value(
            argument_constant->get_value<MDL::IValue_texture>());
        MI_CHECK( value->get_file_path( transaction));
        DB::Tag texture_tag = value->get_value();
        MI_CHECK( texture_tag.is_valid());
        DB::Edit<TEXTURE::Texture> texture( texture_tag, transaction);
        texture->set_gamma( 42.0f);
    }

    // compare hash value
    mi::base::Uuid hash_ic2, hash_cc2;
    get_hash_values( transaction, mi_tag, hash_ic2, hash_cc2, context);
    MI_CHECK( hash_ic != hash_ic2);
    MI_CHECK( hash_cc == hash_cc2);
}

// Check that clearing the MDL file path is taken into account for hashes.
void test_resources_and_hashes_clear_mdl_file_path( DB::Transaction* transaction,
    MDL::Execution_context* context)
{
    DB::Tag mi_tag = transaction->name_to_tag( "mdl::mdl_elements::test_misc::mi_textured");

    // get original hash values
    mi::base::Uuid hash_ic, hash_cc;
    get_hash_values( transaction, mi_tag, hash_ic, hash_cc, context);

    // edit image to clear MDL file path
    {
        DB::Access<MDL::Mdl_function_call> mi( mi_tag, transaction);
        mi::base::Handle<const MDL::IExpression_list> arguments( mi->get_arguments());
        mi::base::Handle<const MDL::IExpression> argument( arguments->get_expression( "t"));
        mi::base::Handle<const MDL::IExpression_constant> argument_constant(
            argument->get_interface<MDL::IExpression_constant>());
        mi::base::Handle<const MDL::IValue_texture> value(
            argument_constant->get_value<MDL::IValue_texture>());
        MI_CHECK( value->get_file_path( transaction));
        DB::Tag texture_tag = value->get_value();
        MI_CHECK( texture_tag.is_valid());
        DB::Access<TEXTURE::Texture> texture( texture_tag, transaction);
        DB::Tag image_tag = texture->get_image();
        {
            DB::Edit<DBIMAGE::Image> image( image_tag, transaction);
        }
        MI_CHECK( !value->get_file_path( transaction));
    }

    // compare hash value
    mi::base::Uuid hash_ic2, hash_cc2;
    get_hash_values( transaction, mi_tag, hash_ic2, hash_cc2, context);
    MI_CHECK( hash_ic != hash_ic2);
    MI_CHECK( hash_cc == hash_cc2);
}

// Check that tag version changes of textures are taken into account for hashes (MDL file path not
// valid).
void test_resources_and_hashes_modify_texture( DB::Transaction* transaction,
    MDL::Execution_context* context)
{
    DB::Tag mi_tag = transaction->name_to_tag( "mdl::mdl_elements::test_misc::mi_textured");

    // get original hash values
    mi::base::Uuid hash_ic, hash_cc;
    get_hash_values( transaction, mi_tag, hash_ic, hash_cc, context);

    // edit texture to change tag version
    {
        DB::Access<MDL::Mdl_function_call> mi( mi_tag, transaction);
        mi::base::Handle<const MDL::IExpression_list> arguments( mi->get_arguments());
        mi::base::Handle<const MDL::IExpression> argument( arguments->get_expression( "t"));
        mi::base::Handle<const MDL::IExpression_constant> argument_constant(
            argument->get_interface<MDL::IExpression_constant>());
        mi::base::Handle<const MDL::IValue_texture> value(
            argument_constant->get_value<MDL::IValue_texture>());
        MI_CHECK( !value->get_file_path( transaction));
        DB::Tag texture_tag = value->get_value();
        MI_CHECK( texture_tag.is_valid());
        DB::Edit<TEXTURE::Texture> texture( texture_tag, transaction);
    }

    // compare hash value
    mi::base::Uuid hash_ic2, hash_cc2;
    get_hash_values( transaction, mi_tag, hash_ic2, hash_cc2, context);
    MI_CHECK( hash_ic != hash_ic2);
    MI_CHECK( hash_cc == hash_cc2);
}

// Check that tag version changes of images are taken into account for hashes (MDL file path not
// valid).
void test_resources_and_hashes_modify_image( DB::Transaction* transaction,
    MDL::Execution_context* context)
{
    DB::Tag mi_tag = transaction->name_to_tag( "mdl::mdl_elements::test_misc::mi_textured");

    // get original hash values
    mi::base::Uuid hash_ic, hash_cc;
    get_hash_values( transaction, mi_tag, hash_ic, hash_cc, context);

    // edit image to change tag version
    {
        DB::Access<MDL::Mdl_function_call> mi( mi_tag, transaction);
        mi::base::Handle<const MDL::IExpression_list> arguments( mi->get_arguments());
        mi::base::Handle<const MDL::IExpression> argument( arguments->get_expression( "t"));
        mi::base::Handle<const MDL::IExpression_constant> argument_constant(
            argument->get_interface<MDL::IExpression_constant>());
        mi::base::Handle<const MDL::IValue_texture> value(
            argument_constant->get_value<MDL::IValue_texture>());
        MI_CHECK( !value->get_file_path( transaction));
        DB::Tag texture_tag = value->get_value();
        MI_CHECK( texture_tag.is_valid());
        DB::Access<TEXTURE::Texture> texture( texture_tag, transaction);
        DB::Tag image_tag = texture->get_image();
        DB::Edit<DBIMAGE::Image> image( image_tag, transaction);
    }

    // compare hash value
    mi::base::Uuid hash_ic2, hash_cc2;
    get_hash_values( transaction, mi_tag, hash_ic2, hash_cc2, context);
    MI_CHECK( hash_ic != hash_ic2);
    MI_CHECK( hash_cc == hash_cc2);
}

void test_module_names( DB::Transaction* transaction, MDL::Execution_context* context)
{
    // check forbidden module names
    mi::Sint32 result = 0;
    result = MDL::Mdl_module::create_module( transaction, "::b:ool", context);
    MI_CHECK_EQUAL( -1, result);
    result = MDL::Mdl_module::create_module( transaction, "::my::b\\sdf", context);
    MI_CHECK_EQUAL( -1, result);
}

void test_body_and_temporaries(
    DB::Transaction* transaction, MDL::IExpression_factory* ef)
{
    {
        DB::Tag md_tag = transaction->name_to_tag( "mdl::mdl_elements::test_misc::md_body(color)");
        DB::Access<MDL::Mdl_function_definition> md( md_tag, transaction);

        mi::base::Handle<const MDL::IExpression> body( md->get_body( transaction));
        MI_CHECK( body);

        mi::Size temporary_count = md->get_temporary_count( transaction);
        MI_CHECK_EQUAL( 1, temporary_count);

        mi::base::Handle<const MDL::IExpression> temporary0( md->get_temporary( transaction, 0));
        MI_CHECK( temporary0);

        mi::base::Handle<const MDL::IExpression> temporary1( md->get_temporary( transaction, 1));
        MI_CHECK( !temporary1);
    }

    {
        DB::Tag fd_tag = transaction->name_to_tag(
            "mdl::mdl_elements::test_misc::fd_body_without_control_flow_direct_call(color)");
        DB::Access<MDL::Mdl_function_definition> fd( fd_tag, transaction);

        mi::base::Handle<const MDL::IExpression> body( fd->get_body( transaction));
        MI_CHECK( body);

        mi::base::Handle<const MDL::IExpression_direct_call> body_direct_call(
            body->get_interface<MDL::IExpression_direct_call>());
        MI_CHECK( body_direct_call);

        mi::Size temporary_count = fd->get_temporary_count( transaction);
        MI_CHECK_EQUAL( 1, temporary_count);

        mi::base::Handle<const MDL::IExpression> temporary0( fd->get_temporary( transaction, 0));
        MI_CHECK( temporary0);

        mi::base::Handle<const MDL::IExpression> temporary1( fd->get_temporary( transaction, 1));
        MI_CHECK( !temporary1);
    }

    {
        DB::Tag fd_tag = transaction->name_to_tag(
            "mdl::mdl_elements::test_misc::fd_body_without_control_flow_parameter(color)");
        DB::Access<MDL::Mdl_function_definition> fd( fd_tag, transaction);

        mi::base::Handle<const MDL::IExpression> body( fd->get_body( transaction));
        MI_CHECK( body);

        mi::base::Handle<const MDL::IExpression_parameter> body_parameter(
           body->get_interface<MDL::IExpression_parameter>());
        MI_CHECK( body_parameter);

        mi::Size temporary_count = fd->get_temporary_count( transaction);
        MI_CHECK_EQUAL( 0, temporary_count);

        mi::base::Handle<const MDL::IExpression> temporary0( fd->get_temporary( transaction, 0));
        MI_CHECK( !temporary0);
    }

    {
        DB::Tag fd_tag = transaction->name_to_tag(
            "mdl::mdl_elements::test_misc::fd_body_without_control_flow_constant()");
        DB::Access<MDL::Mdl_function_definition> fd( fd_tag, transaction);

        mi::base::Handle<const MDL::IExpression> body( fd->get_body( transaction));
        MI_CHECK( body);

        mi::base::Handle<const MDL::IExpression_constant> body_constant(
            body->get_interface<MDL::IExpression_constant>());
        MI_CHECK( body_constant);

        mi::Size temporary_count = fd->get_temporary_count( transaction);
        MI_CHECK_EQUAL( 0, temporary_count);

        mi::base::Handle<const MDL::IExpression> temporary0( fd->get_temporary( transaction, 0));
        MI_CHECK( !temporary0);
    }

    {
        DB::Tag fd_tag = transaction->name_to_tag(
            "mdl::mdl_elements::test_misc::fd_body_with_control_flow(color,int)");
        DB::Access<MDL::Mdl_function_definition> fd( fd_tag, transaction);

        mi::base::Handle<const MDL::IExpression> body( fd->get_body( transaction));
        MI_CHECK( !body);

        mi::Size temporary_count = fd->get_temporary_count( transaction);
        MI_CHECK_EQUAL( 0, temporary_count);

        mi::base::Handle<const MDL::IExpression> temporary0( fd->get_temporary( transaction, 0));
        MI_CHECK( !temporary0);
    }
}

void test_resource_sharing( DB::Transaction* transaction, MDL::Execution_context* context)
{
    std::string mdle_path1
        = TEST::mi_src_path( "io/scene/mdl_elements/test_resource_sharing1.mdle");
    std::string mdle_path2
        = TEST::mi_src_path( "io/scene/mdl_elements/test_resource_sharing2.mdle");

    std::string module_name1
       = MDL::get_db_name( MDL::get_mdl_name_from_load_module_arg( mdle_path1, true));
    std::string module_name2
        = MDL::get_db_name( MDL::get_mdl_name_from_load_module_arg( mdle_path2, true));

    std::string material_name1 = module_name1 + "::main(texture_2d,light_profile,bsdf_measurement)";
    std::string material_name2 = module_name2 + "::main(texture_2d,light_profile,bsdf_measurement)";

    mi::base::Uuid invalid_hash{0,0,0,0};

     // load the MDL "mdl_elements::test_resource_sharing_texture1/2" modules
    mi::Sint32 result;
    result = MDL::Mdl_module::create_module( transaction, mdle_path1.c_str(), context);
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( 0, result);
    result = MDL::Mdl_module::create_module( transaction, mdle_path2.c_str(), context);
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( 0, result);

    // load defaults of material_name1/2
    DB::Tag md1_tag = transaction->name_to_tag( material_name1.c_str());
    DB::Access<MDL::Mdl_function_definition> md1( md1_tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> defaults1( md1->get_defaults());
    DB::Tag md2_tag = transaction->name_to_tag( material_name2.c_str());
    DB::Access<MDL::Mdl_function_definition> md2( md2_tag, transaction);
    mi::base::Handle<const MDL::IExpression_list> defaults2( md2->get_defaults());

    {
        // check default "t" (texture)
       mi::base::Handle<const MDL::IExpression_constant> defaults1_param0(
           defaults1->get_expression<MDL::IExpression_constant>( "t"));
       mi::base::Handle<const MDL::IValue_texture> value1(
           defaults1_param0->get_value<MDL::IValue_texture>());
       DB::Tag texture1_tag = value1->get_value();
       DB::Access<TEXTURE::Texture> texture1( texture1_tag, transaction);
       DB::Tag image1_tag = texture1->get_image();
       DB::Access<DBIMAGE::Image> image1( image1_tag, transaction);
       mi::base::Uuid impl_hash1 = image1->get_impl_hash();
       DB::Tag impl_tag1 = image1->get_impl_tag();

       mi::base::Handle<const MDL::IExpression_constant> defaults2_param0(
           defaults2->get_expression<MDL::IExpression_constant>( "t"));
       mi::base::Handle<const MDL::IValue_texture> value2(
           defaults2_param0->get_value<MDL::IValue_texture>());
       DB::Tag texture2_tag = value2->get_value();
       DB::Access<TEXTURE::Texture> texture2( texture2_tag, transaction);
       DB::Tag image2_tag = texture2->get_image();
       DB::Access<DBIMAGE::Image> image2( image2_tag, transaction);
       mi::base::Uuid impl_hash2 = image2->get_impl_hash();
       DB::Tag impl_tag2 = image2->get_impl_tag();

       MI_CHECK( impl_hash1 != invalid_hash);
       MI_CHECK( impl_hash2 != invalid_hash);
       MI_CHECK( impl_hash1 == impl_hash2);
       MI_CHECK( impl_tag1 != DB::Tag());
       MI_CHECK( impl_tag2 != DB::Tag());
       MI_CHECK( impl_tag1 == impl_tag2);
    }

    {
        // check default "l" (light profile)
       mi::base::Handle<const MDL::IExpression_constant> defaults1_param0(
           defaults1->get_expression<MDL::IExpression_constant>( "l"));
       mi::base::Handle<const MDL::IValue_light_profile> value1(
           defaults1_param0->get_value<MDL::IValue_light_profile>());
       DB::Tag light_profile1_tag = value1->get_value();
       DB::Access<LIGHTPROFILE::Lightprofile> light_profile1( light_profile1_tag, transaction);
       mi::base::Uuid impl_hash1 = light_profile1->get_impl_hash();
       DB::Tag impl_tag1 = light_profile1->get_impl_tag();

       mi::base::Handle<const MDL::IExpression_constant> defaults2_param0(
           defaults2->get_expression<MDL::IExpression_constant>( "l"));
       mi::base::Handle<const MDL::IValue_light_profile> value2(
           defaults2_param0->get_value<MDL::IValue_light_profile>());
       DB::Tag light_profile2_tag = value2->get_value();
       DB::Access<LIGHTPROFILE::Lightprofile> light_profile2( light_profile2_tag, transaction);
       mi::base::Uuid impl_hash2 = light_profile2->get_impl_hash();
       DB::Tag impl_tag2 = light_profile2->get_impl_tag();

       MI_CHECK( impl_hash1 != invalid_hash);
       MI_CHECK( impl_hash2 != invalid_hash);
       MI_CHECK( impl_hash1 == impl_hash2);
       MI_CHECK( impl_tag1 != DB::Tag());
       MI_CHECK( impl_tag2 != DB::Tag());
       MI_CHECK( impl_tag1 == impl_tag2);
    }

    {
        // check default "b" (BSDF measurement)
       mi::base::Handle<const MDL::IExpression_constant> defaults1_param0(
           defaults1->get_expression<MDL::IExpression_constant>( "b"));
       mi::base::Handle<const MDL::IValue_bsdf_measurement> value1(
           defaults1_param0->get_value<MDL::IValue_bsdf_measurement>());
       DB::Tag bsdf_measurement1_tag = value1->get_value();
       DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement1( bsdf_measurement1_tag, transaction);
       mi::base::Uuid impl_hash1 = bsdf_measurement1->get_impl_hash();
       DB::Tag impl_tag1 = bsdf_measurement1->get_impl_tag();

       mi::base::Handle<const MDL::IExpression_constant> defaults2_param0(
           defaults2->get_expression<MDL::IExpression_constant>( "b"));
       mi::base::Handle<const MDL::IValue_bsdf_measurement> value2(
           defaults2_param0->get_value<MDL::IValue_bsdf_measurement>());
       DB::Tag bsdf_measurement2_tag = value2->get_value();
       DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement2( bsdf_measurement2_tag, transaction);
       mi::base::Uuid impl_hash2 = bsdf_measurement2->get_impl_hash();
       DB::Tag impl_tag2 = bsdf_measurement2->get_impl_tag();

       MI_CHECK( impl_hash1 != invalid_hash);
       MI_CHECK( impl_hash2 != invalid_hash);
       MI_CHECK( impl_hash1 == impl_hash2);
       MI_CHECK( impl_tag1 != DB::Tag());
       MI_CHECK( impl_tag2 != DB::Tag());
       MI_CHECK( impl_tag1 == impl_tag2);
    }
}

void test_parsing( MDL::Execution_context* context)
{
    {
        std::vector<std::pair<std::string, bool>> valid_simple_package_or_module_name = {
            { "a",           true  },
            { "0",           true  },
            { "int",         true  },
            { "€",           true  },
            { u8"\u20ac",    true  }, // Euro sign
            { "",            false },
            { "/",           false },
            { "\\",          false },
            { ":",           false },
            { "\x01",        false },
            { "\x1f",        false },
            { "\x7f",        false },
            { ",",           true  },
            { "(",           true  },
            { ")",           true  },
            { "[",           true  },
            { "]",           true  }
        };

        for( const auto& s: valid_simple_package_or_module_name)
            MI_CHECK_EQUAL( MDL::is_valid_simple_package_or_module_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> valid_module_name = {
            { "",           false },
            { ":",          false },
            { "::",         false },
            { ":::",        false },
            { "::::",       false },
            { "::a",        true  },
            { "a",          false },
            { "::a::",      false },
            { "::a::b",     true  }
        };

        for( const auto& s: valid_module_name)
            MI_CHECK_EQUAL( MDL::is_valid_module_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> absolute = {
            { "",                               false },
            { ":",                              false },
            { "::",                             false },
            { "::mod",                          true  },
            { "/path/mod.mdle",                 true  },
            { "/path/mod.mdle::fd(int)",        true  },
            { "foo.mdle.bar",                   false },
            { "mdl::df",                        false },
            { "mdle::/path/mod.mdle",           true  },
            { "mdle::/path/mod.mdle::fd(int)",  true  },
            { "mdl::foo.mdle.bar",              false }
        };

        for( const auto& s: absolute)
            MI_CHECK_EQUAL( MDL::is_absolute( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> scope = {
            { "",              false },
            { ":",             false },
            { "::",            true  },
            { ":::",           true  },
            { "::p1::p2::mod", true  },
        };

        for( const auto& s: scope)
            MI_CHECK_EQUAL( MDL::starts_with_scope( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> slash = {
            { "",                      false },
            { "/",                     true  },
            { "//",                    true  },
            { "/path/to/file",         true  },
            { "relative/path/to/file", false }
        };

        for( const auto& s: slash)
            MI_CHECK_EQUAL( MDL::starts_with_slash( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> mdle = {
            { "::df",                           false },
            { "/path/mod.mdle",                 true  },
            { "/path/mod.mdle::fd(int)",        true  },
            { "foo.mdle.bar",                   false },
            { "mdl::df",                        false },
            { "mdle::/path/mod.mdle",           true  },
            { "mdle::/path/mod.mdle::fd(int)",  true  },
            { "mdl::foo.mdle.bar",              false }
        };

        for( const auto& s: mdle)
            MI_CHECK_EQUAL( MDL::is_mdle( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> deprecated = {
            { "::p1::p2::mod::fd$1.3", true  },
            { "::p1::p2::mod::fd",     false },
            { "::p1::p2::m$d::fd$1.3", true  },
            { "::p1::p2::m$d::fd",     false },
            { "fd$1.3",                true  },
            { "fd",                    false }
        };

        for( const auto& s: deprecated)
            MI_CHECK_EQUAL( MDL::is_deprecated( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> absolute_file_path = {
            { "/dir1/dir2/file.mdl",      true  },
            { "file.mdl",                 false },
            { "::mod",                    false },
            { "C:\\dir1\\dir2\\file.mdl", false }, // This is not an MDL file path!
        };

        for( const auto& s: absolute_file_path)
            MI_CHECK_EQUAL( MDL::is_absolute_mdl_file_path( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> strip_resource_owner_prefix = {
            { "::p1::p2::mod::/foo.png", "/foo.png" },
            { "::p1::p2::mod::foo.png",  "foo.png"  },
            { "::/foo.png",              "/foo.png" },
            { "::foo.png",               "foo.png"  },
            { "::p1::p2::mod::",         ""         },
            { "/foo.png",                "/foo.png" },
            { "foo.png",                 "foo.png"  },
            { "",                        ""         }
        };

        for( const auto& s: strip_resource_owner_prefix)
            MI_CHECK_EQUAL( MDL::strip_resource_owner_prefix( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> get_resource_owner_prefix = {
            { "::p1::p2::mod::/foo.png", "::p1::p2::mod" },
            { "::p1::p2::mod::foo.png",  "::p1::p2::mod" },
            { "::/foo.png",              ""              },
            { "::foo.png",               ""              },
            { "::p1::p2::mod::",         "::p1::p2::mod" },
            { "/foo.png",                ""              },
            { "foo.png",                 ""              },
            { "",                        ""              }
        };

        for( const auto& s: get_resource_owner_prefix)
            MI_CHECK_EQUAL( MDL::get_resource_owner_prefix( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> deprecated_suffix = {
            { "::p1::p2::mod::fd",      "::p1::p2::mod::fd"      },
            { "::p1::p2::mod::fd$1",    "::p1::p2::mod::fd$1"    },
            { "::p1::p2::mod::fd$1.",   "::p1::p2::mod::fd$1."   },
            { "::p1::p2::mod::fd$1.3",  "::p1::p2::mod::fd"      },
            { "::p1::p2::mod::fd$1.39", "::p1::p2::mod::fd$1.39" },
            { "::p1::p2::mod::fd$1.99", "::p1::p2::mod::fd$1.99" },
            { "::p1::p2::m$d::fd",      "::p1::p2::m$d::fd"      },
            { "::p1::p2::m$d::fd$1",    "::p1::p2::m$d::fd$1"    },
            { "::p1::p2::m$d::fd$1.",   "::p1::p2::m$d::fd$1."   },
            { "::p1::p2::m$d::fd$1.3",  "::p1::p2::m$d::fd"      },
            { "::p1::p2::m$d::fd$1.39", "::p1::p2::m$d::fd$1.39" },
            { "::p1::p2::m$d::fd$1.99", "::p1::p2::m$d::fd$1.99" },
            { "fd",                     "fd"                     },
            { "fd$1",                   "fd$1"                   },
            { "fd$1.",                  "fd$1."                  },
            { "fd$1.3",                 "fd"                     },
            { "fd$1.39",                "fd$1.39"                },
            { "fd$1.99",                "fd$1.99"                },
        };

        for( const auto& s: deprecated_suffix)
            MI_CHECK_EQUAL( MDL::strip_deprecated_suffix( s.first), s.second);
    }
    {
        using triple = std::tuple<std::string, std::string, bool>;

        std::string builtins = MDL::get_builtins_module_mdl_name();

        std::vector<triple> in_module = {
            { "::p1::p2::mod::md",                  "::p1::p2::mod", true  },
            { "::p1::p2::mod::sub::md",             "::p1::p2::mod", false },
            { "::p1::p2::mod::sub::fd(int)",        "::p1::p2::mod", false },
            { "::p1::p2::mod::fd(::p1::some_type)", "::p1::p2::mod", true  },
            { "::p1::p2::modulo(int,int)",          "::p1::p2::mod", false },
            { "::material",                         builtins,        true  },
            { "material",                           builtins,        true  },
            { "::sub::material",                    builtins,        false },
            { "sub::material",                      builtins,        false },
            { "::sub::function(int)",               builtins,        false },
            { "sub::function(int)",                 builtins,        false },
            { "::function(::p1::some_type)",        builtins,        true  },
            { "function(::p1::some_type)",          builtins,        true  },
        };

        for( const auto& s: in_module)
            MI_CHECK_EQUAL( MDL::is_in_module( std::get<0>( s), std::get<1>( s)), std::get<2>( s));
    }
    {
        using triple = std::tuple<std::string, std::string, std::string>;

        std::string builtins = MDL::get_builtins_module_mdl_name();

        std::vector<triple> remove_qualifiers = {
            { "::p1::p2::mod::md",                  "::p1::p2::mod", "md"                          },
            { "::p1::p2::mod::sub::md",             "::p1::p2::mod", "::p1::p2::mod::sub::md"      },
            { "::p1::p2::mod::sub::fd(int)",        "::p1::p2::mod", "::p1::p2::mod::sub::fd(int)" },
            { "::p1::p2::mod::fd(::p1::some_type)", "::p1::p2::mod", "fd(::p1::some_type)"         },
            { "::p1::p2::modulo(int,int)",          "::p1::p2::mod", "::p1::p2::modulo(int,int)"   },
            { "::material",                         builtins,        "material"                    },
            { "material",                           builtins,        "material"                    },
            { "::sub::material",                    builtins,        "::sub::material"             },
            { "sub::material",                      builtins,        "sub::material"               },
            { "::sub::function(int)",               builtins,        "::sub::function(int)"        },
            { "sub::function(int)",                 builtins,        "sub::function(int)"          },
            { "::function(::p1::some_type)",        builtins,        "function(::p1::some_type)"   },
            { "function(::p1::some_type)",          builtins,        "function(::p1::some_type)"   },
        };

        for( const auto& s: remove_qualifiers)
            MI_CHECK_EQUAL( MDL::remove_qualifiers_if_from_module(
                std::get<0>( s), std::get<1>( s)), std::get<2>( s));
    }
    {
        std::vector<std::pair<std::string, std::string>> db_name = {
            { "::foo",          "mdl::foo"           },
            { "foo",            "mdl::foo"           }, // deprecated
            { "::/foo.mdle",    "mdle::/foo.mdle"    },
            { "::/C:/foo.mdle", "mdle::/C:/foo.mdle" },
        };

        for( const auto& s: db_name)
            MI_CHECK_EQUAL( MDL::get_db_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> simple_module_name = {
            { "::p1::p2::mod", "mod" },
            { "::mod",         "mod" }
        };

        for( const auto& s: simple_module_name)
            MI_CHECK_EQUAL( MDL::get_mdl_simple_module_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::vector<std::string>>> package_component_names = {
            { "::p1::p2::mod", { "p1", "p2"} },
            { "::mod",         { } }
        };

        for( const auto& s: package_component_names)
            MI_CHECK( MDL::get_mdl_package_component_names( s.first) == s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> simple_definition_name = {
            { "::p1::p2::mod::fd",  "fd"             },
            { "::mod::fd",          "fd"             },
            { "/path/mod.mdle::fd", "fd"             },
            { "operator+",          "operator+"      },
            { "::intensity_mode",   "intensity_mode" }
        };

        for( const auto& s: simple_definition_name)
            MI_CHECK_EQUAL( MDL::get_mdl_simple_definition_name( s.first), s.second);
    }
    {
        std::string builtins = MDL::get_builtins_module_mdl_name();

        std::vector<std::pair<std::string, std::string>> module_name = {
            { "::p1::p2::mod::fd",  "::p1::p2::mod"  },
            { "::mod::fd",          "::mod"          },
            { "/path/mod.mdle::fd", "/path/mod.mdle" },
            { "operator+",          builtins         },
            { "::intensity_mode",   builtins         }
        };

        for( const auto& s: module_name)
            MI_CHECK_EQUAL( MDL::get_mdl_module_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> field_name = {
            { "::p1::p2::mod::struct.field",   "field" },
            { "::mod::struct.field",           "field" },
            { "/path/mod.mdle::struct.field",  "field" },
            { "::p1::p2::m.od::struct.field",  "field" },
            { "::m.od::struct.field",          "field" },
            { "/path/m.od.mdle::struct.field", "field" },
            { "struct.field",                  "field" },
        };

        for( const auto& s: field_name)
            MI_CHECK_EQUAL( MDL::get_mdl_field_name( s.first), s.second);
    }
    {
        using triple = std::tuple<std::string, std::string, std::string>;

        std::vector<triple> dot_or_bracket = {
            { "foo",        "foo", ""        },
            { "foo.bar",    "foo", "bar"     },
            { "foo[0]",     "foo", "[0]"     },
            { "foo[0].bar", "foo", "[0].bar" },
            { "[0]",        "0",   ""        },
            { "[0].bar",    "0",   "bar"     }
        };

        for( const auto& s: dot_or_bracket) {
            std::string head, tail;
            MDL::split_next_dot_or_bracket( std::get<0>( s).c_str(), head, tail);
            MI_CHECK_EQUAL( head, std::get<1>( s));
            MI_CHECK_EQUAL( tail, std::get<2>( s));
        }
    }
    {
        using triple = std::tuple<std::string, bool, std::string>;

        std::vector<triple> load_module_arg = {
            { "::p1::p2::mod",       false, "::p1::p2::mod" },
            { "/mod.mdle",           true,  "::/mod.mdle" },
            { "/./mod.mdle",         true,  "::/mod.mdle" },
            { "/a/../mod.mdle",      true,  "::/mod.mdle" },
#ifdef MI_PLATFORM_WINDOWS
            { "C:\\mod.mdle",        true,  "::/C%3A/mod.mdle" },
            { "C:\\.\\mod.mdle",     true,  "::/C%3A/mod.mdle" },
            { "C:\\a\\..\\mod.mdle", true,  "::/C%3A/mod.mdle" },
#else
            { "C:\\mod.mdle",        true,  "::C%3A/mod.mdle" },
            { "C:\\.\\mod.mdle",     true,  "::C%3A/mod.mdle" },
            { "C:\\a\\..\\mod.mdle", true,  "::C%3A/mod.mdle" },
#endif
        };

        for( const auto& s: load_module_arg)
            MI_CHECK_EQUAL( MDL::get_mdl_name_from_load_module_arg(
                std::get<0>( s), std::get<1>( s)), std::get<2>( s));
    }
    {
        std::vector<std::pair<std::string, bool>> mdl_or_mdle = {
            { "",        false  },
            { "x",       false  },
            { "mdl",     false  },
            { "mdl::",   true   },
            { "mdlfoo",  false  },
            { "mdle",    false  },
            { "mdle::",  true   },
            { "mdlefoo", false  },
        };

        for( const auto& s: mdl_or_mdle)
            MI_CHECK_EQUAL( MDL::starts_with_mdl_or_mdle( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> strip_mdl_or_mdle_prefix = {
            { "",                                         ""                                    },
            { "x",                                        "x"                                   },
            { "mdl",                                      "mdl"                                 },
            { "mdl::",                                    "::"                                  },
            { "mdl::mod",                                 "::mod"                               },
            { "mdl::mod::fd(::some_type)",                "::mod::fd(::some_type)"              },
            { "mdlfoo",                                   "mdlfoo"                              },
            { "mdle",                                     "mdle"                                },
            { "mdle::",                                   "::"                                  },
            { "mdle::/path/mod.mdle",                     "::/path/mod.mdle"                    },
            { "mdle::/path/mod.mdle::fd(::some_type)",    "::/path/mod.mdle::fd(::some_type)"   },
            { "mdle::C:/path/mod.mdle",                   "::C:/path/mod.mdle"                  },
            { "mdle::C:/path/mod.mdle::fd(::some_type)",  "::C:/path/mod.mdle::fd(::some_type)" },
            { "mdlefoo",                                  "mdlefoo"                             },
        };

        for( const auto& s: strip_mdl_or_mdle_prefix)
            MI_CHECK_EQUAL( MDL::strip_mdl_or_mdle_prefix( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> absolute = {
            { "",                        false },
            { "mod",                     false },
            { "mod::fd(int)",            false },
            { ":",                       false },
            { ":mod",                    false },
            { ":mod::fd(int)",           false },
            { "::",                      false },
            { "::mod",                   true  },
            { "::mod::fd(int)",          true  },
            { "/path",                   false },
            { "/path/mod.mdle",          true  },
            { "/path/mod.mdle::fd(int)", true  }
        };

        for( const auto& s: absolute)
            MI_CHECK_EQUAL( MDL::is_absolute( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> dot_mdl_suffix = {
            { "/p1/p2/mod.mdl",           "/p1/p2/mod"           },
            { "/p1/p2/mod.mdle:main.mdl", "/p1/p2/mod.mdle:main" },
            { "/p1/p2/mod.mdr:main.mdl",  "/p1/p2/mod.mdr:main"  }
        };

        for( const auto& s: dot_mdl_suffix)
            MI_CHECK_EQUAL( MDL::strip_dot_mdl_suffix( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> archive_file_path = {
            { "/p1/p2/mod.mdr",     true   },
            { "/p1/p2/mod.mdr.mdl", false  },
            { "foo",                false  }
        };

        for( const auto& s: archive_file_path)
            MI_CHECK_EQUAL( MDL::is_archive_filename( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> mdle_file_path = {
            { "/p1/p2/mod.mdle",     true   },
            { "/p1/p2/mod.mdle.mdl", false  },
            { "foo",                 false  }
        };

        for( const auto& s: mdle_file_path)
            MI_CHECK_EQUAL( MDL::is_mdle_filename( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, bool>> container_member = {
            { "/p1/p2/mod.mdr:main.mdl",       true  },
            { "/p1/p2/mod.mdle:main.mdl",      true  },
            { "/p1/p2/mod.mdl",                false },
            { "C:\\p1\\p2\\mod.mdr:main.mdl",  true  },
            { "C:\\p1\\p2\\mod.mdle:main.mdl", true  },
            { "C:\\p1\\p2\\mod.mdl",           false }
        };

        for( const auto& s: container_member)
            MI_CHECK_EQUAL( MDL::is_container_member( s.first.c_str()), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> container_filename = {
            { "/p1/p2/mod.mdr:main.mdl",       "/p1/p2/mod.mdr"       },
            { "/p1/p2/mod.mdle:main.mdl",      "/p1/p2/mod.mdle"      },
            { "/p1/p2/mod.mdl",                ""     ,               },
            { "C:\\p1\\p2\\mod.mdr:main.mdl",  "C:\\p1\\p2\\mod.mdr"  },
            { "C:\\p1\\p2\\mod.mdle:main.mdl", "C:\\p1\\p2\\mod.mdle" },
            { "C:\\p1\\p2\\mod.mdl",           ""                     }
        };

        for( const auto& s: container_filename)
            MI_CHECK_EQUAL( MDL::get_container_filename( s.first.c_str()), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> container_membername = {
            { "/p1/p2/mod.mdr:main.mdl",       "main.mdl" },
            { "/p1/p2/mod.mdle:main.mdl",      "main.mdl" },
            { "/p1/p2/mod.mdl",                ""         },
            { "C:\\p1\\p2\\mod.mdr:main.mdl",  "main.mdl" },
            { "C:\\p1\\p2\\mod.mdle:main.mdl", "main.mdl" },
            { "C:\\p1\\p2\\mod.mdl",           ""         }
        };

        for( const auto& s: container_membername)
            MI_CHECK_EQUAL( MDL::get_container_membername( s.first.c_str()), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> enc = {
            { "(",  "%28" },
            { ")",  "%29" },
            { "<",  "%3C" },
            { ">",  "%3E" },
            { ",",  "%2C" },
            { ":",  "%3A" },
            { "$",  "%24" },
            { "#",  "%23" },
            { "?",  "%3F" },
            { "@",  "%40" },
            { "%",  "%25" },
        };

        for( const auto& s: enc)
            MI_CHECK_EQUAL( MDL::encode( s.first.c_str()), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> dec1 = {
            { "%28", "("  },
            { "%29", ")"  },
            { "%3C", "<"  },
            { "%3E", ">"  },
            { "%2C", ","  },
            { "%3A", ":"  },
            { "%24", "$"  },
            { "%23", "#"  },
            { "%3F", "?"  },
            { "%40", "@"  },
            { "%25", "%"  },
            { "%",   ""   },
            { "%2",  ""   },
            { "%3c", ""   },
            { "%GG", ""   },
            { "%41", ""   },
            { ",",   ""   }
        };

        for( const auto& s: dec1)
            MI_CHECK_EQUAL( MDL::decode( s.first, /*strict*/ true, context), s.second);

        std::vector<std::pair<std::string, std::string>> dec2 = {
            { "%28", "("   },
            { "%29", ")"   },
            { "%3C", "<"   },
            { "%3E", ">"   },
            { "%2C", ","   },
            { "%3A", ":"   },
            { "%24", "$"   },
            { "%23", "#"   },
            { "%3F", "?"   },
            { "%40", "@"   },
            { "%25", "%"   }, // up to here identical to test above
            { "%",   "%"   }, // from here different from test above
            { "%2",  "%2"  },
            { "%5c", "%5c" },
            { "%GG", "%GG" },
            { "%41", "A"   },
            { ",",   ","   }
        };

        for( const auto& s: dec2)
            MI_CHECK_EQUAL( MDL::decode( s.first, /*strict*/ false, context), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> enc_module_name = {
            { "foo",            "foo"               },
            { "foo::bar",       "foo::bar"          },
            { "::foo",          "::foo"             },
            { "::foo::bar",     "::foo::bar"        },
            { "::C:/mod.mdle",  "::C%3A/mod.mdle" },
            { "::<builtins>",   "::%3Cbuiltins%3E"  },
            { "::<neuray>",     "::%3Cneuray%3E"    },
            { "::foo::bar$1.6", "::foo::bar%241.6"  } // '$' is part of the module name
        };

        for( const auto& s: enc_module_name)
            MI_CHECK_EQUAL( MDL::encode_module_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> dec_module_name = {
            { "foo",              "foo"            },
            { "foo::bar",         "foo::bar"       },
            { "::foo",            "::foo"          },
            { "::foo::bar",       "::foo::bar"     },
            { "::C%3A/mod.mdle",  "::C:/mod.mdle"  },
            { "::%3Cbuiltins%3E", "::<builtins>"   },
            { "::%3Cneuray%3E",   "::<neuray>"     },
            { "::<builtins>",     ""               }  // a decoded name is not valid as input
        };

        for( const auto& s: dec_module_name)
            MI_CHECK_EQUAL( MDL::decode_module_name( s.first), s.second);
    }
    {
        std::vector<std::pair<std::string, std::string>> enc_name_wo_signature = {
            { "foo",           "foo"          },
            { "foo::bar",      "foo::bar"     },
            { "::foo",         "::foo"        },
            { "::foo::bar",    "::foo::bar"   },
            { "foo$::bar$1.6", "foo%24::bar$1.6" },  // 2nd '$' indicates the start of the version suffix
        };

        for( const auto& s: enc_name_wo_signature)
            MI_CHECK_EQUAL( MDL::encode_name_without_signature( s.first), s.second);
    }
    {
        using Parameter_types = std::vector<std::string>;
        using Input           = std::pair<std::string,Parameter_types>;

        std::vector<std::pair<Input, std::string>> enc_name_plus_signature = {
            { { "::foo",     { "int" } }, "::foo(int)"                },
            { { "::foo$1.6", { "int" } }, "::foo$1.6(int)"            },
            { { "::foo(),",  { "()," } }, "::foo%28%29%2C(%28%29%2C)" }, // parameter type not really valid
        };

        for( const auto& s: enc_name_plus_signature) {
            const auto& name            = s.first.first;
            const auto& parameter_types = s.first.second;
            MI_CHECK_EQUAL( MDL::encode_name_plus_signature( name, parameter_types), s.second);
        }
    }
    {
        struct Input { std::string s; mi::Size f; mi::Sint32 u; mi::Sint32 v; };

        std::vector<std::pair<Input, std::string>> marker_to_string = {
            { { "test_<UDIM>.png",         0,  0,  0 }, "test_1001.png"     },
            { { "test_<UDIM>.png",         0, -1,  3 }, ""                  },
            { { "test_<UDIM>.png",         0,  1, 10 }, "test_1102.png"     },
            { { "test_<UVTILE0>.png",      0,  0,  0 }, "test__u0_v0.png"   },
            { { "test_<UVTILE0>.png",      0, -1,  3 }, "test__u-1_v3.png"  },
            { { "test_<UVTILE1>.png",      0,  0,  0 }, "test__u1_v1.png"   },
            { { "test_<UVTILE1>.png",      0, -1,  3 }, "test__u0_v4.png"   },
            { { "test_<>.png",             0,  0,  0 }, ""                  },
            { { "test_<#>.png",            1,  0,  0 }, "test_1.png"        },
            { { "test_<#>.png",           42,  0,  0 }, "test_42.png"       },
            { { "test_<##>.png",          42,  0,  0 }, "test_42.png"       },
            { { "test_<UDIM>_<#>.png",     1,  0,  0 }, "test_1001_1.png"   },
            { { "test_<#>_<UVTILE0>.png",  2,  3,  4 }, "test_2__u3_v4.png" },
            { { "test_<#>_<UVTILE1>.png",  5,  6,  7 }, "test_5__u7_v8.png" },
        };

        for( const auto& s: marker_to_string) {
            const Input& i = s.first;
            MI_CHECK_EQUAL( MDL::frame_uvtile_marker_to_string( i.s, i.f, i.u, i.v), s.second);
        }
    }
    {
        struct Input { std::string s; std::string m; };

        std::vector<std::pair<Input, std::string>> string_to_marker = {
            { { "1001.png",               "<UDIM>"    }, "<UDIM>.png"               },
            { { "test.11.22222.1111.png", "<UDIM>"    }, "test.11.<UDIM>2.1111.png" },
            { { "1001",                   "<UDIM>"    }, "<UDIM>"                   },
            { { "0001.png",               "<UDIM>"    }, ""                         },
            { { "test.11.png",            "<UDIM>"    }, ""                         },
            { { "test_u0_v-4.png",        "<UVTILE0>" }, "test<UVTILE0>.png"        },
            { { "_u-3_v001",              "<UVTILE0>" }, "<UVTILE0>"                },
            { { "test.u_11_.png",         "<UVTILE0>" }, ""                         },
            { { "test_u0_v-4.png",        "<UVTILE1>" }, "test<UVTILE1>.png"        },
            { { "_u-3_v001",              "<UVTILE1>" }, "<UVTILE1>"                },
            { { "test.u_11_.png",         "<UVTILE1>" }, ""                         },
        };

        for( const auto& s: string_to_marker) {
            const Input& i = s.first;
            MI_CHECK_EQUAL( MDL::deprecated_uvtile_string_to_marker( i.s, i.m), s.second);
        }
    }
}

void test_module_comparator( DB::Transaction* transaction, MDL::Execution_context* context)
{
    // Check that a reasonably complex module like ::nvidia::core_definitions can be compared
    // against itself.
    context->clear_messages();
    context->set_result( 0);
    mi::Sint32 result
        = MDL::Mdl_module::create_module( transaction, "::nvidia::core_definitions", context);
    MI_CHECK_EQUAL( 0, result);

    DB::Tag tag = transaction->name_to_tag( "mdl::nvidia::core_definitions");
    DB::Access<MDL::Mdl_module> module( tag, transaction);
    mi::base::Handle<const mi::mdl::IModule> mdl_module( module->get_mdl_module());

    MI_CHECK( mi::mdl::equal( mdl_module.get(), mdl_module.get()));
}

void test_create_value_with_range_annotation(
    DB::Transaction* transaction, MDL::Execution_context* context)
{
    mi::base::Handle<MDL::IType_factory> tf( MDL::get_type_factory());
    mi::base::Handle<MDL::IValue_factory> vf( MDL::get_value_factory());
    mi::base::Handle<MDL::IExpression_factory> ef( MDL::get_expression_factory());

    DB::Tag anno_tag = transaction->name_to_tag( "mdl::anno");
    MI_CHECK( anno_tag);
    DB::Access<MDL::Mdl_module> anno_mod( anno_tag, transaction);
    mi::base::Handle<const MDL::IAnnotation_definition> soft_range_int(
        anno_mod->get_annotation_definition( "::anno::soft_range(int,int)"));
    MI_CHECK( soft_range_int);
    mi::base::Handle<const MDL::IAnnotation_definition> hard_range_color(
        anno_mod->get_annotation_definition( "::anno::hard_range(color,color)"));
    MI_CHECK( hard_range_color);
    mi::base::Handle<const MDL::IAnnotation_definition> hard_range_double2(
        anno_mod->get_annotation_definition( "::anno::hard_range(double2,double2)"));
    MI_CHECK( hard_range_double2);
    mi::base::Handle<const MDL::IAnnotation_definition> hidden(
        anno_mod->get_annotation_definition( "::anno::hidden()"));
    MI_CHECK( hidden);

    // prepare hidden annotation
    mi::base::Handle<const MDL::IAnnotation> hidden_anno( hidden->create_annotation( nullptr));
    MI_CHECK( hidden_anno);

    // prepare soft_range annotation on int
    mi::base::Handle<MDL::IValue> min_int( vf->create_int( 2));
    mi::base::Handle<MDL::IValue> max_int( vf->create_int( 5));
    mi::base::Handle<MDL::IExpression_constant> min_int_expr( ef->create_constant( min_int.get()));
    mi::base::Handle<MDL::IExpression_constant> max_int_expr( ef->create_constant( max_int.get()));
    mi::base::Handle<MDL::IExpression_list> args( ef->create_expression_list( 2));
    args->add_expression( "min", min_int_expr.get());
    args->add_expression( "max", max_int_expr.get());
    mi::base::Handle<const MDL::IAnnotation> soft_anno(
         soft_range_int->create_annotation( args.get()));
    MI_CHECK( soft_anno);

    // test null pointers / wrong annotations
    mi::base::Handle<const MDL::IValue> result;
    result = vf->create( static_cast<const MDL::IAnnotation*>( nullptr));
    MI_CHECK( !result);
    result = vf->create( hidden_anno.get());
    MI_CHECK( !result);

    // test soft_range annotation on int
    result = vf->create( soft_anno.get());
    MI_CHECK( result);
    mi::base::Handle<const MDL::IValue_int> result_int( result->get_interface<MDL::IValue_int>());
    MI_CHECK_EQUAL( result_int->get_value(), 3);

    // prepare hard_range annotation on color
    mi::base::Handle<MDL::IValue> min_color( vf->create_color( 0.1f, 0.1f, 0.7f));
    mi::base::Handle<MDL::IValue> max_color( vf->create_color( 0.3f, 0.5f, 0.1f));
    mi::base::Handle<MDL::IExpression_constant> min_color_expr(
        ef->create_constant( min_color.get()));
    mi::base::Handle<MDL::IExpression_constant> max_color_expr(
        ef->create_constant( max_color.get()));
    args->set_expression( "min", min_color_expr.get());
    args->set_expression( "max", max_color_expr.get());
    mi::base::Handle<const MDL::IAnnotation> hard_anno(
        hard_range_color->create_annotation( args.get()));
    MI_CHECK( hard_anno);

    // test hard_range annotation on color
    result = vf->create( hard_anno.get());
    MI_CHECK( result);
    mi::base::Handle<const MDL::IValue_color> result_color(
        result->get_interface<MDL::IValue_color>());
    mi::base::Handle<const MDL::IValue_float> r( result_color->get_value( 0));
    MI_CHECK_EQUAL( r->get_value(), 0.2f);
    mi::base::Handle<const MDL::IValue_float> g( result_color->get_value( 1));
    MI_CHECK_EQUAL( g->get_value(), 0.3f);
    mi::base::Handle<const MDL::IValue_float> b( result_color->get_value( 2));
    MI_CHECK_EQUAL( b->get_value(), 0.4f);

    // prepare hard_range annotation on double2
    mi::base::Handle<const MDL::IType_double> type_double( tf->create_double());
    mi::base::Handle<const MDL::IType_vector> type_double2(
        tf->create_vector( type_double.get(), 2));
    mi::base::Handle<MDL::IValue_vector> min_double2( vf->create_vector( type_double2.get()));
    mi::base::Handle<MDL::IValue_vector> max_double2( vf->create_vector( type_double2.get()));
    mi::base::Handle<MDL::IValue_double> min_double2_0(
        min_double2->get_value<MDL::IValue_double>( 0));
    mi::base::Handle<MDL::IValue_double> min_double2_1(
        min_double2->get_value<MDL::IValue_double>( 1));
    mi::base::Handle<MDL::IValue_double> max_double2_0(
        max_double2->get_value<MDL::IValue_double>( 0));
    mi::base::Handle<MDL::IValue_double> max_double2_1(
        max_double2->get_value<MDL::IValue_double>( 1));
    min_double2_0->set_value( 0.0);
    min_double2_1->set_value( 42.0);
    max_double2_0->set_value( 1.0);
    max_double2_1->set_value( 43.0);
    mi::base::Handle<MDL::IExpression_constant> min_double2_expr(
        ef->create_constant( min_double2.get()));
    mi::base::Handle<MDL::IExpression_constant> max_double2_expr(
        ef->create_constant( max_double2.get()));
    args->set_expression( "min", min_double2_expr.get());
    args->set_expression( "max", max_double2_expr.get());
    mi::base::Handle<const MDL::IAnnotation> hard2_anno(
        hard_range_double2->create_annotation( args.get()));
    MI_CHECK( hard2_anno);

    // test hard_range annotation on double2
    result = vf->create( hard2_anno.get());
    MI_CHECK( result);
    mi::base::Handle<const MDL::IValue_vector> result_double2(
        result->get_interface<MDL::IValue_vector>());
    mi::base::Handle<const MDL::IValue_double> result_0(
        result_double2->get_value<MDL::IValue_double>( 0));
    mi::base::Handle<const MDL::IValue_double> result_1(
        result_double2->get_value<MDL::IValue_double>( 1));
    MI_CHECK_EQUAL( result_0->get_value(), 0.5);
    MI_CHECK_EQUAL( result_1->get_value(), 42.5);

    mi::base::Handle<const MDL::IValue_double> result_double;

    // prepare annotation blocks
    mi::base::Handle<MDL::IAnnotation_block> empty( ef->create_annotation_block( 0));
    mi::base::Handle<MDL::IAnnotation_block> none( ef->create_annotation_block( 1));
    mi::base::Handle<MDL::IAnnotation_block> soft( ef->create_annotation_block( 1));
    mi::base::Handle<MDL::IAnnotation_block> hard( ef->create_annotation_block( 1));
    mi::base::Handle<MDL::IAnnotation_block> both( ef->create_annotation_block( 2));
    mi::base::Handle<MDL::IAnnotation_block> both2( ef->create_annotation_block( 2));
    none->add_annotation( hidden_anno.get());
    soft->add_annotation( soft_anno.get());
    hard->add_annotation( hard_anno.get());
    both->add_annotation( soft_anno.get());
    both->add_annotation( hard_anno.get());
    both2->add_annotation( hard_anno.get());
    both2->add_annotation( soft_anno.get());

    // test null pointer for annotations
    result = vf->create( nullptr, nullptr);
    MI_CHECK( !result);
    result = vf->create( type_double.get(), nullptr);
    MI_CHECK( result);
    result_double = result->get_interface<MDL::IValue_double>();
    MI_CHECK( result_double);

    // test empty annotation block
    result = vf->create( nullptr, empty.get());
    MI_CHECK( !result);
    result = vf->create( type_double.get(), empty.get());
    MI_CHECK( result);
    result_double = result->get_interface<MDL::IValue_double>();
    MI_CHECK( result_double);

    // test annotation block without range annotation
    result = vf->create( nullptr, none.get());
    MI_CHECK( !result);
    result = vf->create( type_double.get(), none.get());
    MI_CHECK( result);
    result_double = result->get_interface<MDL::IValue_double>();
    MI_CHECK( result_double);

    // test annotation block with soft_range annotation
    result = vf->create( nullptr, soft.get());
    MI_CHECK( result);
    result = vf->create( type_double.get(), soft.get());
    MI_CHECK( result);
    result_int = result->get_interface<MDL::IValue_int>();
    MI_CHECK( result_int);
    MI_CHECK_EQUAL( result_int->get_value(), 3);

    // test annotation block with hard_range annotation
    result = vf->create( nullptr, hard.get());
    MI_CHECK( result);
    result = vf->create( type_double.get(), hard.get());
    MI_CHECK( result);
    result_color = result->get_interface<MDL::IValue_color>();
    MI_CHECK( result_color);
    r = result_color->get_value( 0);
    MI_CHECK_EQUAL( r->get_value(), 0.2f);
    g =result_color->get_value( 1);
    MI_CHECK_EQUAL( g->get_value(), 0.3f);
    b =result_color->get_value( 2);
    MI_CHECK_EQUAL( b->get_value(), 0.4f);

    // test annotation block with soft_range and hard_range annotation
    result = vf->create( nullptr, both.get());
    MI_CHECK( result);
    result = vf->create( type_double.get(), both.get());
    MI_CHECK( result);
    result_int = result->get_interface<MDL::IValue_int>();
    MI_CHECK( result_int);
    MI_CHECK_EQUAL( result_int->get_value(), 3);

    // test annotation block with soft_range and hard_range annotation (reverse order)
    result = vf->create( nullptr, both2.get());
    MI_CHECK( result);
    result = vf->create( type_double.get(), both2.get());
    MI_CHECK( result);
    result_int = result->get_interface<MDL::IValue_int>();
    MI_CHECK( result_int);
    MI_CHECK_EQUAL( result_int->get_value(), 3);
}

void test_factory_compare_deep_call_comparisons(
    DB::Transaction* transaction, MDL::Execution_context* context)
{
    mi::base::Handle<MDL::IType_factory> tf( MDL::get_type_factory());
    mi::base::Handle<MDL::IValue_factory> vf( MDL::get_value_factory());
    mi::base::Handle<MDL::IExpression_factory> ef( MDL::get_expression_factory());

    // create expression lists with "int value = 42", "int value = 43", "bool value = true"
    mi::base::Handle<const MDL::IValue> int42( vf->create_int( 42));
    mi::base::Handle<const MDL::IValue> int43( vf->create_int( 43));
    mi::base::Handle<const MDL::IValue> booltrue( vf->create_bool( true));
    mi::base::Handle<const MDL::IExpression> expr42( ef->create_constant( int42.get()));
    mi::base::Handle<const MDL::IExpression> expr43( ef->create_constant( int43.get()));
    mi::base::Handle<const MDL::IExpression> exprtrue( ef->create_constant( booltrue.get()));
    mi::base::Handle<MDL::IExpression_list> list42( ef->create_expression_list( 1));
    mi::base::Handle<MDL::IExpression_list> list43( ef->create_expression_list( 1));
    mi::base::Handle<MDL::IExpression_list> listtrue( ef->create_expression_list( 1));
    list42->add_expression( "value", expr42.get());
    list43->add_expression( "value", expr43.get());
    listtrue->add_expression( "value", exprtrue.get());

    DB::Tag tag = transaction->name_to_tag( "mdl::int(int)");
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);
    mi::base::Handle<const MDL::IType> type( fd->get_return_type());
    DB::Tag tag2 = transaction->name_to_tag( "mdl::int(bool)");
    DB::Access<MDL::Mdl_function_definition> fd2( tag2, transaction);
    mi::base::Handle<const MDL::IType> type2( fd2->get_return_type());
    MDL::Mdl_function_call* fc;

    // create calls (note that the DB elements do \em not share the arguments -- cloned on creation)
    // call1: int(int), argument 42
    // call2: int(int), argument 42
    // call3: int(int), argument 43
    // call4: int(bool), argument true
    fc = fd->create_function_call( transaction, list42.get());
    MI_CHECK( fc);
    DB::Tag tag_call1 = transaction->store( fc, "call1", 255);
    fc = fd->create_function_call( transaction, list42.get());
    MI_CHECK( fc);
    DB::Tag tag_call2 = transaction->store( fc, "call2", 255);
    fc = fd->create_function_call( transaction, list43.get());
    MI_CHECK( fc);
    DB::Tag tag_call3 = transaction->store( fc, "call3", 255);
    fc = fd2->create_function_call( transaction, listtrue.get());
    MI_CHECK( fc);
    DB::Tag tag_call4 = transaction->store( fc, "call4", 255);

    // create corresponding call expressions
    mi::base::Handle<MDL::IExpression_call> expr_call1, expr_call2, expr_call3, expr_call4;
    expr_call1 = ef->create_call( type.get(), tag_call1);
    expr_call2 = ef->create_call( type.get(), tag_call2);
    expr_call3 = ef->create_call( type.get(), tag_call3);
    expr_call4 = ef->create_call( type2.get(), tag_call4);

    // shallow comparison: all three call expressions are different (not identical)
    // order determined by tag creation above
    MI_CHECK_EQUAL( ef->compare( expr_call1.get(), expr_call2.get()), -1);
    MI_CHECK_EQUAL( ef->compare( expr_call2.get(), expr_call3.get()), -1);
    MI_CHECK_EQUAL( ef->compare( expr_call3.get(), expr_call4.get()), -1);
    MI_CHECK_EQUAL( ef->compare( expr_call4.get(), expr_call1.get()), +1);

#define DEEP mi::neuraylib::IExpression_factory::DEEP_CALL_COMPARISONS, 0.0, transaction

    // deep comparison:
    // - expr_call1 and expr_call2 are equal
    // - expr_call3 is different (different arguments)
    // - expr_call4 is different (different definition), order depends on creation order of
    //   int(int) and  int(bool)
    MI_CHECK_EQUAL( ef->compare( expr_call1.get(), expr_call2.get(), DEEP),  0);
    MI_CHECK_EQUAL( ef->compare( expr_call2.get(), expr_call3.get(), DEEP), -1);
    MI_CHECK_EQUAL( ef->compare( expr_call3.get(), expr_call1.get(), DEEP), +1);
    MI_CHECK_NOT_EQUAL( ef->compare( expr_call1.get(), expr_call4.get(), DEEP), 0);
    MI_CHECK_NOT_EQUAL( ef->compare( expr_call3.get(), expr_call4.get(), DEEP), 0);

#undef DEEP
}

void test_factory_compare_skip_type_aliases(
    DB::Transaction* transaction, MDL::Execution_context* context)
{
    mi::base::Handle<MDL::IType_factory> tf( MDL::get_type_factory());
    mi::base::Handle<MDL::IValue_factory> vf( MDL::get_value_factory());
    mi::base::Handle<MDL::IExpression_factory> ef( MDL::get_expression_factory());

    // create expression lists with "int value = 42"
    mi::base::Handle<const MDL::IValue> int42( vf->create_int( 42));
    mi::base::Handle<const MDL::IExpression> expr42( ef->create_constant( int42.get()));
    mi::base::Handle<MDL::IExpression_list> list42( ef->create_expression_list( 1));
    list42->add_expression( "value", expr42.get());

    DB::Tag tag = transaction->name_to_tag( "mdl::int(int)");
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);
    mi::base::Handle<const MDL::IType> type( fd->get_return_type());
    mi::base::Handle<const MDL::IType> varying( tf->create_alias(
        type.get(), MDL::IType::MK_VARYING, nullptr));
    mi::base::Handle<const MDL::IType> uniform( tf->create_alias(
        type.get(), MDL::IType::MK_UNIFORM, nullptr));

    MDL::Mdl_function_call* fc;

    // create call
    fc = fd->create_function_call( transaction, list42.get());
    MI_CHECK( fc);
    DB::Tag tag_call = transaction->store( fc, "call", 255);

    // create corresponding call expressions
    // expr_call1: (auto) int
    // expr_call2: varying int
    // expr_call3: uniform int
    mi::base::Handle<MDL::IExpression_call> expr_call1, expr_call2, expr_call3;
    expr_call1 = ef->create_call( type.get(),    tag_call);
    expr_call2 = ef->create_call( varying.get(), tag_call);
    expr_call3 = ef->create_call( uniform.get(), tag_call);

    // shallow comparison: all three call expressions are different (not identical)
    // order determined by type comparison
    MI_CHECK_NOT_EQUAL( ef->compare( expr_call1.get(), expr_call2.get()), 0);
    MI_CHECK_NOT_EQUAL( ef->compare( expr_call2.get(), expr_call3.get()), 0);
    MI_CHECK_NOT_EQUAL( ef->compare( expr_call3.get(), expr_call1.get()), 0);

#define SKIP mi::neuraylib::IExpression_factory::SKIP_TYPE_ALIASES, 0.0, transaction

    // deep comparison:
    MI_CHECK_EQUAL( ef->compare( expr_call1.get(), expr_call2.get(), SKIP), 0);
    MI_CHECK_EQUAL( ef->compare( expr_call2.get(), expr_call3.get(), SKIP), 0);
    MI_CHECK_EQUAL( ef->compare( expr_call3.get(), expr_call1.get(), SKIP), 0);

#undef SKIP
}

void test_unresolved_resources_factory(
    DB::Transaction* transaction, MDL::Execution_context* context)
{
    context->set_option( MDL_CTX_OPTION_RESOLVE_RESOURCES, false);

    {
        // with owner module
        mi::base::Handle<MDL::IValue_texture> t1( MDL::create_texture(
            transaction,
            "::test_module::./test.png",
            MDL::IType_texture::TS_2D,
            2.2f,
            "R",
            /*shared*/ false,
            context));
        MI_CHECK_CTX( context);
        MI_CHECK( t1);
        MI_CHECK_EQUAL( t1->get_value(), DB::Tag());
        MI_CHECK_EQUAL_CSTR( t1->get_unresolved_file_path(), "./test.png");
        MI_CHECK_EQUAL_CSTR( t1->get_owner_module(), "::test_module");
        MI_CHECK_EQUAL( t1->get_gamma(), 2.2f);
        MI_CHECK_EQUAL_CSTR( t1->get_selector(), "R");

        // without owner module
        mi::base::Handle<MDL::IValue_texture> t2( MDL::create_texture(
            transaction,
            "/test.png",
            MDL::IType_texture::TS_2D,
            2.2f,
            "R",
            /*shared*/ false,
            context));
        MI_CHECK_CTX( context);
        MI_CHECK( t2);

        // invalid owner module
        mi::base::Handle<MDL::IValue_texture> t3( MDL::create_texture(
            transaction,
            "garbage::/test.png",
            MDL::IType_texture::TS_2D,
            2.2f,
            "R",
            /*shared*/ false,
            context));
        MI_CHECK_CTX_RESULT( context, -2);
        MI_CHECK( !t3);
        context->clear_messages();
        context->set_result( 0);
    }

    {
        // with owner module
        mi::base::Handle<MDL::IValue_light_profile> l1( MDL::create_light_profile(
            transaction,
            "::test_module::./test.ies",
            /*shared*/ false,
            context));
        MI_CHECK_CTX( context);
        MI_CHECK( l1);
        MI_CHECK_EQUAL( l1->get_value(), DB::Tag());
        MI_CHECK_EQUAL_CSTR( l1->get_unresolved_file_path(), "./test.ies");
        MI_CHECK_EQUAL_CSTR( l1->get_owner_module(), "::test_module");

        // without owner module
        mi::base::Handle<MDL::IValue_light_profile> l2( MDL::create_light_profile(
            transaction,
            "/test.ies",
            /*shared*/ false,
            context));
        MI_CHECK_CTX( context);
        MI_CHECK( l2);

        // invalid owner module
        mi::base::Handle<MDL::IValue_light_profile> l3( MDL::create_light_profile(
            transaction,
            "garbage::/test.ies",
            /*shared*/ false,
            context));
        MI_CHECK_CTX_RESULT( context, -2);
        MI_CHECK( !l3);
        context->clear_messages();
        context->set_result( 0);
    }

    {
        // with owner module
        mi::base::Handle<MDL::IValue_bsdf_measurement> b1( MDL::create_bsdf_measurement(
            transaction,
            "::test_module::./test.mbsdf",
            /*shared*/ false,
            context));
        MI_CHECK_CTX( context);
        MI_CHECK( b1);
        MI_CHECK_EQUAL( b1->get_value(), DB::Tag());
        MI_CHECK_EQUAL_CSTR( b1->get_unresolved_file_path(), "./test.mbsdf");
        MI_CHECK_EQUAL_CSTR( b1->get_owner_module(), "::test_module");

        // without owner module
        mi::base::Handle<MDL::IValue_bsdf_measurement> b2( MDL::create_bsdf_measurement(
            transaction,
            "/test.mbsdf",
            /*shared*/ false,
            context));
        MI_CHECK_CTX( context);
        MI_CHECK( b2);

        // without owner module
        mi::base::Handle<MDL::IValue_bsdf_measurement> b3( MDL::create_bsdf_measurement(
            transaction,
            "garbage::/test.mbsdf",
            /*shared*/ false,
            context));
        MI_CHECK_CTX_RESULT( context, -2);
        MI_CHECK( !b3);
        context->clear_messages();
        context->set_result( 0);
    }

    context->set_option( MDL_CTX_OPTION_RESOLVE_RESOURCES, true);
}

void test_unresolved_resources_load( DB::Transaction* transaction, MDL::Execution_context* context)
{
    context->set_option( MDL_CTX_OPTION_RESOLVE_RESOURCES, false);

    mi::Sint32 result = MDL::Mdl_module::create_module(
        transaction, "::mdl_elements::test_resolver_success", context);
    MI_CHECK_EQUAL( 0, result);

    check_texture_def( transaction, // weak-relative file path
        "mdl::mdl_elements::test_resolver_success::fd_texture_success(texture_2d)",
        "t", true, false, "resources/test.png");

    check_texture_def(transaction, // weak-relative file path, udim image sequence
        "mdl::mdl_elements::test_resolver_success::fd_texture_udim_success(texture_2d)",
        "t", true, false, "resources/test<UDIM>.png");

    check_light_profile_def( transaction, // strict-relative file path
        "mdl::mdl_elements::test_resolver_success::fd_light_profile_success(light_profile)",
        "l", true, false, "/mdl_elements/resources/test.ies");

    check_bsdf_measurement_def( transaction, // absolute path
        "mdl::mdl_elements::test_resolver_success::fd_bsdf_measurement_success(bsdf_measurement)",
        "b", true, false, "/mdl_elements/resources/test.mbsdf");

    context->set_option( MDL_CTX_OPTION_RESOLVE_RESOURCES, true);
}

void test_get_mdl_module_name( MDL::IType_factory* tf)
{
    std::string builtins = MDL::get_builtins_module_mdl_name();

    {
        // check builtin atomic type
        mi::base::Handle<const MDL::IType_bool> t( tf->create_bool());
        std::string module = MDL::get_mdl_module_name( t.get());
        MI_CHECK_EQUAL( module, builtins);
    }
    {
        // check array type
        mi::base::Handle<const MDL::IType_bool> t1( tf->create_bool());
        mi::base::Handle<const MDL::IType_array> t2(
            tf->create_immediate_sized_array( t1.get(), 42));
        std::string module = MDL::get_mdl_module_name( t2.get());
        MI_CHECK_EQUAL( module, builtins);
    }
    {
        // check alias type without name
        mi::base::Handle<const MDL::IType_bool> t1( tf->create_bool());
        mi::base::Handle<const MDL::IType_alias> t2(
            tf->create_alias( t1.get(), MDL::IType::MK_UNIFORM, nullptr));
        std::string module = MDL::get_mdl_module_name( t2.get());
        MI_CHECK_EQUAL( module, builtins);
    }
    {
        // check alias type with name
        mi::base::Handle<const MDL::IType_bool> t1( tf->create_bool());
        mi::base::Handle<const MDL::IType_alias> t2(
            tf->create_alias( t1.get(), MDL::IType::MK_NONE, "::my_mod::my_reexported_bool"));
        std::string module = MDL::get_mdl_module_name( t2.get());
        MI_CHECK_EQUAL( module, "::my_mod");
    }
    {
        // check enum type from builtins module
        mi::base::Handle<const MDL::IType_enum> t(
            tf->create_enum( "::intensity_mode"));
        std::string module = MDL::get_mdl_module_name( t.get());
        MI_CHECK_EQUAL( module, builtins);
    }
    {
        // check enum type from non-builtins module
        mi::base::Handle<const MDL::IType_enum> t(
            tf->create_enum( "::tex::gamma_mode"));
        std::string module = MDL::get_mdl_module_name( t.get());
        MI_CHECK_EQUAL( module, "::tex");
    }
    {
        // check struct type from builtins module
        mi::base::Handle<const MDL::IType_struct> t(
            tf->create_struct( "::material"));
        std::string module = MDL::get_mdl_module_name( t.get());
        MI_CHECK_EQUAL( module, builtins);
    }
    {
        // check struct type from non-builtins module
        mi::base::Handle<const MDL::IType_struct> t(
            tf->create_struct( "::df::bsdf_component"));
        std::string module = MDL::get_mdl_module_name( t.get());
        MI_CHECK_EQUAL( module, "::df");
    }
}

void test_direct_call_creation(
    DB::Transaction* transaction, MDL::Execution_context* context)
{
    // Test creation of a direct call using parameter references as a default.
    mi::base::Handle<MDL::IType_factory> tf( MDL::get_type_factory());
    mi::base::Handle<MDL::IExpression_factory> ef( MDL::get_expression_factory());

    DB::Tag tag = transaction->name_to_tag(
        "mdl::df::simple_glossy_bsdf(float,float,color,color,float3,::df::scatter_mode,string)");
    MI_CHECK( tag);
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);

    mi::base::Handle<const MDL::IType> type0( tf->create_float());
    mi::base::Handle<MDL::IExpression> arg0( ef->create_parameter( type0.get(), 0));
    mi::base::Handle<MDL::IExpression_list> args( ef->create_expression_list(
        /*initial_capacity*/ 1));
    MI_CHECK_EQUAL( 0, args->add_expression( "roughness_u", arg0.get()));
    // Use default for roughness_v.

    mi::base::Handle<MDL::IExpression_direct_call> call(
        fd->create_direct_call( transaction, args.get()));
    MI_CHECK( call);

    mi::base::Handle<const MDL::IExpression_list> c_args( call->get_arguments());
    mi::base::Handle<const MDL::IExpression> c_arg0( c_args->get_expression( 1));
    MI_CHECK_EQUAL( c_arg0->get_kind(), MDL::IExpression::EK_PARAMETER);
}

// Thread that repeatedly imports and removes a fixed module (name depending on the thread ID).
class Test_thread : public THREAD::Thread
{
public:
    void initialize( DB::Database* database, DB::Scope* scope,
        mi::Size thread_id, mi::Size thread_count, mi::Size iteration_count,
        THREAD::Condition* before, THREAD::Condition* after)
    {
        m_database = database;
        m_scope = scope;
        m_thread_id = thread_id;
        m_thread_count = thread_count;
        m_iteration_count = iteration_count;
        m_before = before;
        m_after = after;
        std::ostringstream s;
        s << thread_id;
        m_module_mdl_name = std::string( "::M") + s.str();
        m_module_db_name  = std::string( "mdl") + m_module_mdl_name;
    }

    void run()
    {
        MDL::Execution_context context;
        for( mi::Size i = 0; i < m_iteration_count; ++i) {

            DB::Transaction* transaction = m_scope->start_transaction();
            mi::base::Handle<mi::neuraylib::IReader> reader(
                MDL::create_reader( "mdl 1.0; import mdl_elements::test_misc::*;"));
            mi::Sint32 result = MDL::Mdl_module::create_module(
                transaction, m_module_mdl_name.c_str(), reader.get(), &context);
            MI_CHECK_LESS_OR_EQUAL( 0, result);
            DB::Tag tag = transaction->name_to_tag( m_module_db_name.c_str());
            bool result2 = transaction->remove( tag);
            MI_CHECK_EQUAL( true, result2);
            transaction->commit();

            TIME::sleep( 0.02);

            // make sure no thread has an open transaction while the last thread triggers the
            // garbage collection
            if( m_thread_id < m_thread_count-1)
                m_before[m_thread_id].wait();
            if( m_thread_id > 0)
                m_before[m_thread_id-1].signal();

            if( m_thread_id == 0)
                m_database->garbage_collection( 2);

            if( m_thread_id > 0)
                m_after[m_thread_id-1].wait();
            if( m_thread_id < m_thread_count-1)
                m_after[m_thread_id].signal();
        }
    }

private:
    DB::Database* m_database;
    DB::Scope* m_scope;
    mi::Size m_thread_id;
    mi::Size m_thread_count;
    mi::Size m_iteration_count;
    THREAD::Condition* m_before;
    THREAD::Condition* m_after;
    std::string m_module_mdl_name;
    std::string m_module_db_name;
};

// Check that concurrent restoring and dropping of the mi::mdl::IModule pointers for the "test"
// module works.
void test_module_pointers( DB::Database* database, DB::Scope* scope)
{
    mi::Size thread_count = 10;
    mi::Size iteration_count = 10;

    Test_thread* threads = new Test_thread[thread_count];
    THREAD::Condition* before = new THREAD::Condition[thread_count-1];
    THREAD::Condition* after = new THREAD::Condition[thread_count-1];
    for( mi::Size i = 0; i < thread_count; ++i) {
        std::ostringstream s;
        s << i;
        threads[i].initialize( database, scope, i, thread_count, iteration_count, before, after);
        threads[i].start();
    }
    for( mi::Size i = 0; i < thread_count; ++i) {
        threads[i].join();
    }
    delete[] after;
    delete[] before;
    delete[] threads;
}

void create_function_call(
    DB::Transaction* transaction, const char* definition_name, const char* call_name)
{
    DB::Tag tag = transaction->name_to_tag( definition_name);
    DB::Access<MDL::Mdl_function_definition> fd( tag, transaction);
    MDL::Mdl_function_call* fc = fd->create_function_call( transaction, nullptr);
    MI_CHECK( fc);
    transaction->store( fc, call_name, 255);
}

void test_main( DB::Scope* global_scope)
{
    MDL::Execution_context context;
    mi::base::Handle<MDL::IType_factory> tf( MDL::get_type_factory());
    mi::base::Handle<MDL::IValue_factory> vf( MDL::get_value_factory());
    mi::base::Handle<MDL::IExpression_factory> ef( MDL::get_expression_factory());

    DB::Transaction* transaction = global_scope->start_transaction();

    // load the MDL "mdl_elements::test_misc" and "mdl_elements::test_resolver_failures" modules
    mi::Sint32 result = MDL::Mdl_module::create_module(
        transaction, "::mdl_elements::test_misc", &context);
    MI_CHECK_EQUAL( 0, result);
    result = MDL::Mdl_module::create_module(
        transaction, "::mdl_elements::test_resolver_success", &context);
    MI_CHECK_EQUAL( 0, result);
    result = MDL::Mdl_module::create_module(
        transaction, "::mdl_elements::test_resolver_failures", &context);
    MI_CHECK_EQUAL( 0, result);

    // instantiate some definitions
    const char* definitions[] = {
        "mdl::mdl_elements::test_resolver_success::fd_texture_success(texture_2d)",
            "mdl::mdl_elements::test_resolver_success::fc_texture_success",
        "mdl::mdl_elements::test_resolver_success::fd_texture_udim_success(texture_2d)",
            "mdl::mdl_elements::test_resolver_success::fc_texture_udim_success",
        "mdl::mdl_elements::test_resolver_success::fd_light_profile_success(light_profile)",
            "mdl::mdl_elements::test_resolver_success::fc_light_profile_success",
        "mdl::mdl_elements::test_resolver_success::fd_bsdf_measurement_success(bsdf_measurement)",
            "mdl::mdl_elements::test_resolver_success::fc_bsdf_measurement_success",
        "mdl::mdl_elements::test_misc::fd_int(int)",
            "mdl::mdl_elements::test_misc::fc_int",
        "mdl::mdl_elements::test_misc::fd_float(float)",
            "mdl::mdl_elements::test_misc::fc_float",
        "mdl::mdl_elements::test_misc::fd_color()",
            "mdl::mdl_elements::test_misc::fc_color",
        "mdl::mdl_elements::test_misc::fd_body_without_control_flow_direct_call(color)",
            "mdl::mdl_elements::test_misc::fc_body_without_control_flow_direct_call",
        "mdl::mdl_elements::test_misc::fd_body_without_control_flow_parameter(color)",
            "mdl::mdl_elements::test_misc::fc_body_without_control_flow_parameter",
        "mdl::mdl_elements::test_misc::fd_body_without_control_flow_constant()",
            "mdl::mdl_elements::test_misc::fc_body_without_control_flow_constant",
        "mdl::mdl_elements::test_misc::fd_body_with_control_flow(color,int)",
            "mdl::mdl_elements::test_misc::fc_body_with_control_flow",
        "mdl::mdl_elements::test_misc::md_array_literal()",
            "mdl::mdl_elements::test_misc::mi_array_literal",
        "mdl::mdl_elements::test_misc::md_textured(texture_2d)",
            "mdl::mdl_elements::test_misc::mi_textured",
        "mdl::mdl_elements::test_misc::md_body(color)",
            "mdl::mdl_elements::test_misc::mi_body"
    };

    for( mi::Size i = 0; i < sizeof( definitions) / sizeof( const char*); i += 2)
        create_function_call( transaction, definitions[i], definitions[i+1]);

    test_get_default_compiled_material( transaction);
    test_resources( transaction);
    test_jitted_environment_function( transaction);

    // the tests for resource hashes depend on each other and need to be run in this order (3rd and
    // 4th test are independent)
    test_resources_and_hashes_edit_gamma( transaction, &context);
    test_resources_and_hashes_clear_mdl_file_path( transaction, &context);
    test_resources_and_hashes_modify_texture( transaction, &context);
    test_resources_and_hashes_modify_image( transaction, &context);

    test_module_names( transaction, &context);

    SYSTEM::Access_module<PATH::Path_module> path_module( false);
    std::string path = TEST::mi_src_path( "io/scene/mdl_elements");
    MI_CHECK_EQUAL( 0, path_module->add_path( PATH::MDL, path));

    // load the MDL "test_archives" module from test_archives.mdr
    context.clear_messages();
    context.set_result( 0);
    result = MDL::Mdl_module::create_module( transaction, "::test_archives", &context);
    MI_CHECK_EQUAL( 0, result);

    test_body_and_temporaries( transaction, ef.get());

    test_resource_sharing( transaction, &context);

    test_parsing( &context);

    test_module_comparator( transaction, &context);

    test_create_value_with_range_annotation( transaction, &context);

    test_factory_compare_deep_call_comparisons( transaction, &context);
    test_factory_compare_skip_type_aliases( transaction, &context);

    test_get_mdl_module_name( tf.get());

    test_direct_call_creation( transaction, &context);

    transaction->commit();
}


void test_multithreading( DB::Database* database, DB::Scope* global_scope)
{
// Skip test on Windows since the machine might become quite unresponsive.
#ifndef MI_PLATFORM_WINDOWS
    DB::Scope* scope = global_scope;
    DB::Transaction* transaction = scope->start_transaction();

    MDL::Execution_context context;
    mi::Sint32 result = MDL::Mdl_module::create_module(
        transaction, "::mdl_elements::test_misc", &context);
    MI_CHECK( result == 0 || result == 1);
    transaction->commit();

    test_module_pointers( database, scope);
#endif // MI_PLATFORM_WINDOWS
}

MI_TEST_AUTO_FUNCTION( test )
{
    Unified_database_access db_access;

    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    config_module->override( "check_serializer_store=1");
    config_module->override( "check_serializer_edit=1");

    SYSTEM::Access_module<PATH::Path_module> path_module( false);
    path_module->clear_search_path( PATH::MDL);
    std::string path = TEST::mi_src_path( "io/scene");
    MI_CHECK_EQUAL( 0, path_module->add_path( PATH::MDL, path));
    path = TEST::mi_src_path( "shaders/mdl");
    MI_CHECK_EQUAL( 0, path_module->add_path( PATH::MDL, path));
    path = TEST::mi_src_path( "../examples/mdl");
    path_module->add_path( PATH::MDL, path);

    SYSTEM::Access_module<PLUG::Plug_module> plug_module( false);
    MI_CHECK( plug_module->load_library( plugin_path_openimageio));

    DB::Database* database = db_access.get_database();
    DB::Scope* scope = database->get_global_scope();

    test_main( scope);


    test_multithreading( database, scope);
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

