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

#include <mi/neuraylib/argument_editor.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/iimage_api.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_distiller_api.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/itile.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_shared.h"

// To avoid race conditions, each unit test uses a separate subdirectory for all the files it
// creates (possibly with further subdirectories).
#define DIR_PREFIX "output_test_icompiled_material"

#include "test_shared_mdl.h" // depends on DIR_PREFIX

#include <io/image/image/test_shared.h> // MI_CHECK_IMG_DIFF



mi::Sint32 result = 0;



void check_icompiled_material(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    bool class_compilation)
{
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
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

    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    mi::base::Handle<const mi::neuraylib::IFunction_call> fc;
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    context->set_option( "meters_per_scene_unit", 42.0f);
    context->set_option( "wavelength_min", 44.0f);
    context->set_option( "wavelength_max", 46.0f);

    fc = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_tmp_ms");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 42.0f, cm->get_mdl_meters_per_scene_unit());
    MI_CHECK_EQUAL( 44.0f, cm->get_mdl_wavelength_min());
    MI_CHECK_EQUAL( 46.0f, cm->get_mdl_wavelength_max());
    MI_CHECK_EQUAL( 1, cm->get_temporary_count());
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 1, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "tint");
    }

    {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( body->get_arguments());

        // check some default values
        mi::base::Handle<const mi::neuraylib::IExpression_constant> ior_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "ior"));
        mi::base::Handle<const mi::neuraylib::IValue_color> ior_value(
            ior_expr->get_value<mi::neuraylib::IValue_color>());
        mi::base::Handle<const mi::neuraylib::IValue_color> exp_ior_value(
            vf->create_color( 1.0f, 1.0f, 1.0f));
        MI_CHECK_EQUAL( 0, vf->compare( ior_value.get(), exp_ior_value.get()));
        mi::base::Handle<const mi::neuraylib::IExpression_constant> thin_walled_expr(
            args->get_expression<mi::neuraylib::IExpression_constant>( "thin_walled"));
        mi::base::Handle<const mi::neuraylib::IValue_bool> thin_walled(
            thin_walled_expr->get_value<mi::neuraylib::IValue_bool>());
        MI_CHECK_EQUAL( false, thin_walled->get_value());

        // check temporary index for the common sub-expression
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> surface(
            args->get_expression<mi::neuraylib::IExpression_temporary>( "surface"));
        MI_CHECK_EQUAL( 0, surface->get_index());
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> backface(
            args->get_expression<mi::neuraylib::IExpression_temporary>( "backface"));
        MI_CHECK_EQUAL( 0, backface->get_index());

        // check temporary 0
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> temporary(
            cm->get_temporary<mi::neuraylib::IExpression_direct_call>( 0));
        MI_CHECK_EQUAL_CSTR(
            "mdl::material_surface(bsdf,material_emission)", temporary->get_definition());
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments(
            temporary->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> scattering(
            arguments->get_expression<mi::neuraylib::IExpression_direct_call>( "scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments2(
            scattering->get_arguments());
        MI_CHECK_EQUAL_CSTR(
            "mdl::df::diffuse_reflection_bsdf(color,float,color,string)", scattering->get_definition());
        if( !class_compilation) {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> tint_expr(
                arguments2->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
            mi::base::Handle<const mi::neuraylib::IValue_color> tint_value(
                tint_expr->get_value<mi::neuraylib::IValue_color>());
            mi::base::Handle<const mi::neuraylib::IValue_color> exp_tint_value(
                vf->create_color( 1.0f, 1.0f, 1.0f));
             MI_CHECK_EQUAL( 0, vf->compare( ior_value.get(), exp_ior_value.get()));
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression_parameter> tint_expr(
                arguments2->get_expression<mi::neuraylib::IExpression_parameter>( "tint"));
            MI_CHECK_EQUAL( 0, tint_expr->get_index());
        }
        mi::base::Handle<const mi::neuraylib::IExpression_constant> roughness_expr(
            arguments2->get_expression<mi::neuraylib::IExpression_constant>( "roughness"));
        mi::base::Handle<const mi::neuraylib::IValue_float> roughness_value(
            roughness_expr->get_value<mi::neuraylib::IValue_float>());
        MI_CHECK_EQUAL( 0.0f, roughness_value->get_value());
        mi::base::Handle<const mi::neuraylib::IExpression_constant> handle_expr(
            arguments2->get_expression<mi::neuraylib::IExpression_constant>( "handle"));
        mi::base::Handle<const mi::neuraylib::IValue_string> handle_value(
            handle_expr->get_value<mi::neuraylib::IValue_string>());
        MI_CHECK_EQUAL_CSTR( "", handle_value->get_value());

        // check argument 0
        if( class_compilation) {
            mi::base::Handle<const mi::neuraylib::IValue_color> argument(
                cm->get_argument<mi::neuraylib::IValue_color>( 0));
            mi::base::Handle<const mi::neuraylib::IValue_color> exp_argument(
                vf->create_color( 1.0f, 1.0f, 1.0f));
            MI_CHECK_EQUAL( 0, vf->compare( argument.get(), exp_argument.get()));
        }
    }

    // check that the ternary operator is not inlined in class compilation mode
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_ternary_operator_argument");

    context->set_option( "meters_per_scene_unit", 1.0f);
    context->set_option( "wavelength_min", 380.0f);
    context->set_option( "wavelength_max", 780.0f);

    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 1, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "cond");
    }

    // check that resource sharing is disabled in class compilation mode
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_resource_sharing");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 2, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "tex0");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "tex1");
    }

    mi::base::Handle fd( transaction->access<mi::neuraylib::IFunction_definition>(
        "mdl::" TEST_MDL "::md_aov(float,float)"));
    MI_CHECK( fd->is_declarative());

    // no explicit target type
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_aov");
    MI_CHECK( fc->is_declarative());
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);

    mi::base::Uuid hash1 = cm->get_sub_expression_hash( "aov");
    MI_CHECK( hash1 != mi::base::Uuid());
    hash1 = cm->get_sub_expression_hash( "");
    mi::base::Uuid hash2 = cm->get_hash();
    MI_CHECK( hash1 == hash2);
    hash1 = cm->get_sub_expression_hash( "surface.scattering");
    hash2 = cm->get_slot_hash( mi::neuraylib::SLOT_SURFACE_SCATTERING);
    MI_CHECK( hash1 == hash2);

    // convert to target type SID_MATERIAL
    mi::base::Handle<const mi::neuraylib::IType> standard_material_type(
        tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
    result = context->set_option( "target_type", standard_material_type.get());
    MI_CHECK_EQUAL( result, 0);
    MI_CHECK_CTX( context);
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    mi::base::Handle<const mi::neuraylib::IExpression> body( cm->get_body());
    mi::base::Handle<const mi::neuraylib::IType_struct> return_type(
        body->get_type<mi::neuraylib::IType_struct>());
    MI_CHECK_EQUAL( return_type->get_predefined_id(), mi::neuraylib::IType_struct::SID_MATERIAL);

    hash1 = cm->get_sub_expression_hash( "aov");
    MI_CHECK( hash1 == mi::base::Uuid());
    hash1 = cm->get_sub_expression_hash( "");
    hash2 = cm->get_hash();
    MI_CHECK( hash1 == hash2);
    hash1 = cm->get_sub_expression_hash( "surface.scattering");
    hash2 = cm->get_slot_hash( mi::neuraylib::SLOT_SURFACE_SCATTERING);
    MI_CHECK( hash1 == hash2);
}

void check_folding(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    bool class_compilation)
{
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
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

    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    mi::base::Handle<const mi::neuraylib::IFunction_call> fc;
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    context->set_option( "meters_per_scene_unit", 42.0f);
    context->set_option( "wavelength_min", 44.0f);
    context->set_option( "wavelength_max", 46.0f);

    // check that certain functions are folded in instance compilation mode
#ifndef RESOLVE_RESOURCES_FALSE
    fc = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( body->get_arguments());

        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> surface(
            args->get_expression<mi::neuraylib::IExpression_direct_call>( "surface"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_surface(
            surface->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> scattering(
            args_surface->get_expression<mi::neuraylib::IExpression_direct_call>( "scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_scattering(
            scattering->get_arguments());

        if( !class_compilation) {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
            MI_CHECK( tint);
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_direct_call>( "tint"));
            MI_CHECK( tint);
        }
    }
    {
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
        mi::base::Handle<const mi::neuraylib::IExpression_list> args( body->get_arguments());

        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> backface(
            args->get_expression<mi::neuraylib::IExpression_direct_call>( "backface"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_backface(
            backface->get_arguments());
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> scattering(
            args_backface->get_expression<mi::neuraylib::IExpression_direct_call>( "scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression_list> args_scattering(
            scattering->get_arguments());

        if( !class_compilation) {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_constant>( "tint"));
            MI_CHECK( tint);
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> tint(
                args_scattering->get_expression<mi::neuraylib::IExpression_direct_call>( "tint"));
            MI_CHECK( tint);
        }
    }
#else // RESOLVE_RESOURCES_FALSE
    // Skip test if resolve_resources is set to false. The resulting compiled material has a
    // completely different structure.
#endif // RESOLVE_RESOURCES_FALSE

    // check that certain parameters are folded in class compilation mode
    // (1) base case without any context flags
    fc = transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::mi_folding2");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 4, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.x");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param1");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 3), "param0");
    }

    // (2) fold all bool parameters
    context->set_option( "fold_all_bool_parameters", true);
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 3, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.x");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param1");
    }
    context->set_option( "fold_all_bool_parameters", false);

    // (3) fold all enum parameters
    context->set_option( "fold_all_enum_parameters", true);
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 3, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.x");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param0");
    }
    context->set_option( "fold_all_enum_parameters", false);

    // (4) fold a parameter by name
    mi::base::Handle<mi::IArray> array( factory->create<mi::IArray>( "String[1]"));
    mi::base::Handle<mi::IString> element( array->get_element<mi::IString>( zero_size));
    element->set_c_str( "param2.x");
    result = context->set_option( "fold_parameters", array.get());
    MI_CHECK_EQUAL( 0, result);
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        MI_CHECK_EQUAL( 3, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "param2.y");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "param1");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 2), "param0");
    }
    context->set_option( "fold_parameters", static_cast<const mi::base::IInterface*>( nullptr));

    // (5a) fold geometry.cutout_opacity (with temporaries)
    context->set_option( "fold_trivial_cutout_opacity", true);
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_folding_cutout_opacity");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_trivial_cutout_opacity", false);

    // (5b) fold geometry.cutout_opacity (with parameter reference)
    context->set_option( "fold_trivial_cutout_opacity", true);
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_folding_cutout_opacity2");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_trivial_cutout_opacity", false);

    // (5c) fold geometry.cutout_opacity (material wrapped into copy-constructor-like meta-material)
    mi::base::Handle<const mi::neuraylib::IFunction_definition> md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_folding_cutout_opacity2(material)"));
    mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
    mi::base::Handle<mi::neuraylib::IExpression> arg(
        ef->create_call( "mdl::" TEST_MDL "::mi_folding_cutout_opacity"));
    args->add_expression( "m", arg.get());
    fc = md->create_function_call( args.get(), &result);
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    MI_CHECK_EQUAL( 0, result);

    context->set_option( "fold_trivial_cutout_opacity", true);
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    if( !class_compilation)
        MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    else {
        // finding geometry.cutout_opacity fails (a path like m.geometry.cutout_opacity might work)
        MI_CHECK_EQUAL( 2, cm->get_parameter_count());
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 0), "m.o1");
        MI_CHECK_EQUAL_CSTR( cm->get_parameter_name( 1), "m.o2");
    }
    context->set_option( "fold_trivial_cutout_opacity", false);

    // (6a) fold transparent layers (a simple case)
    context->set_option( "fold_transparent_layers", true);
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_folding_transparent_layers");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_transparent_layers", false);

    // (6b) fold transparent layers (more complicated case)
    context->set_option( "fold_transparent_layers", true);
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_folding_transparent_layers2");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_transparent_layers", false);

    // (6c) fold transparent layers (nested case)
    context->set_option( "fold_transparent_layers", true);
    fc = transaction->access<mi::neuraylib::IFunction_call>(
        "mdl::" TEST_MDL "::mi_folding_transparent_layers3");
    mi = fc->get_interface<mi::neuraylib::IMaterial_instance>();
    cm = mi->create_compiled_material( flags, context.get());
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 0, cm->get_parameter_count());
    context->set_option( "fold_transparent_layers", false);
}

void check_type_binding(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Handle<const mi::neuraylib::IType> type;
    mi::base::Handle<const mi::neuraylib::IType_array> type_array;
    mi::base::Handle<const mi::neuraylib::IType_list> types;
    mi::base::Handle<mi::neuraylib::IExpression_list> args;

    // check deferred arrays in a material definition (binding mismatch)

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::base::Handle<const mi::neuraylib::IFunction_definition> c_md(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::" TEST_MDL "::md_deferred_2(int[N],int[N])"));
    MI_CHECK( c_md);
    MI_CHECK_EQUAL( 2, c_md->get_parameter_count());
    types = c_md->get_parameter_types();

    MI_CHECK_EQUAL_CSTR( "param0", c_md->get_parameter_name( zero_size));
    type = types->get_type( zero_size);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( !type_array->is_immediate_sized());
    MI_CHECK_EQUAL( minus_one_size, type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    MI_CHECK_EQUAL_CSTR( "param1", c_md->get_parameter_name( 1));
    type = types->get_type( 1);
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_ARRAY, type->get_kind());
    type_array = type->get_interface<mi::neuraylib::IType_array>();
    MI_CHECK( !type_array->is_immediate_sized());
    MI_CHECK_EQUAL( minus_one_size, type_array->get_size());
    type = type_array->get_element_type();
    MI_CHECK_EQUAL( mi::neuraylib::IType::TK_INT, type->get_kind());

    // prepare arguments
    args = ef->create_expression_list();
    {
        mi::base::Handle<mi::neuraylib::IValue_array> param0_value(
            vf->create_array( type_array.get()));
        MI_CHECK_EQUAL( 0, param0_value->set_size( 1));
        mi::base::Handle<mi::neuraylib::IExpression> param0_expr(
            ef->create_constant( param0_value.get()));
        args->add_expression( "param0", param0_expr.get());

        mi::base::Handle<mi::neuraylib::IValue_array> param1_value(
            vf->create_array( type_array.get()));
        MI_CHECK_EQUAL( 0, param1_value->set_size( 2));
        mi::base::Handle<mi::neuraylib::IExpression> param1_expr(
            ef->create_constant( param1_value.get()));
        args->add_expression( "param1", param1_expr.get());
    }

    // instantiate the definition with that argument
    mi::base::Handle<mi::neuraylib::IFunction_call> m_mi(
        c_md->create_function_call( args.get(), &result));
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK( m_mi);

    // compile it (fails because the deferred-sized arrays of both arguments have different length)
    mi::base::Handle<mi::neuraylib::IMaterial_instance> m_mi_mi(
        m_mi->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::base::Handle<const mi::neuraylib::ICompiled_material> c_cm(
        m_mi_mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
    MI_CHECK_NOT_EQUAL( 0, context->get_error_messages_count());
    MI_CHECK( !c_cm);
}

void dump_compiled_material(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const mi::neuraylib::ICompiled_material* cm,
    std::ostream& s)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));

    for( mi::Size i = 0; i < cm->get_parameter_count(); ++i) {
        mi::base::Handle<const mi::neuraylib::IValue> argument( cm->get_argument( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            vf->dump( argument.get(), name.str().c_str(), 1));
        const char* pn = cm->get_parameter_name( i);
        s << "    argument (original parameter name: " << pn << ") "
          << result->get_c_str() << std::endl;
    }

    for( mi::Uint32 i = 0; i < cm->get_temporary_count(); ++i) {
        mi::base::Handle<const mi::neuraylib::IExpression> temporary( cm->get_temporary( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            ef->dump( temporary.get(), name.str().c_str(), 1));
        s << "    temporary " << result->get_c_str() << std::endl;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());
    mi::base::Handle<const mi::IString> result(
        ef->dump( body.get(), nullptr, 1));
    s << "    body " << result->get_c_str() << std::endl;
}


void check_hashing(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
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
    result = mdl_impexp_api->load_module( transaction, "::test_mdl_hashing", context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 0);

    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>( "mdl::test_mdl_hashing"));
    mi::Size material_count = module->get_material_count();
    MI_CHECK_EQUAL( 0, result);
    MI_CHECK_EQUAL( 119, material_count);

    // Compile the material instances and check hashes for inequality with previous instances.
    std::vector<std::string> all_material_names;
    std::vector<mi::base::Handle<const mi::neuraylib::ICompiled_material>> all_compiled_materials;
    std::vector<mi::base::Uuid> all_hashes;

    for( mi::Size i = 0; i < material_count; i++) {

        const char* material_name = module->get_material( i);
        all_material_names.emplace_back( material_name);
        std::string instance_name = "mdl::test_mdl_hashing::mi_" + std::to_string( i);
        do_create_function_call(
            transaction, material_name, instance_name.c_str());

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<const mi::neuraylib::IMaterial_instance>(
                instance_name.c_str()));
        MI_CHECK( mi);

        mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( cm);
        all_compiled_materials.push_back( cm);

        mi::base::Uuid compiled_hash = cm->get_hash();
        for( size_t j = 0; j < all_hashes.size(); ++j) {

            // Do not compare test base and variant.
            if( i == 1 && j == 0)
                continue;

            const auto& other_hash = all_hashes[j];
            if( compiled_hash != other_hash)
                continue;

            const auto& other_material_name = all_material_names[j];
            const auto& other_cm = all_compiled_materials[j];
            std::cout << "*** " << other_material_name << "\n";
            dump_compiled_material( transaction, mdl_factory.get(), other_cm.get(), std::cout);
            std::cout << "*** " << material_name << ":\n";
            dump_compiled_material( transaction, mdl_factory.get(), cm.get(), std::cout);
        }
        all_hashes.push_back( compiled_hash);
    }
}

// Check that ICompiled_material::get_connected_function_db_name() for the material instance named
// \p instance_db_name and the parameter named \p param_path returns \p expected_connected_db_name
// (including \p nullptr as result for constants). If \p default_call is \c true, the varying suffix
// starting with the last underscore is stripped from the return value first.
void check_path(
    const mi::base::Handle<const mi::neuraylib::ICompiled_material>& compiled_material,
    const std::string& instance_db_name,
    const char* param_path,
    const char* expected_connected_db_name,
    bool default_call = false)
{
    MI_CHECK( compiled_material);
    mi::Size param_count = compiled_material->get_parameter_count();

    // Compute index of param_path.
    // so instead of specifying the
    mi::Size index = -1;
    for( mi::Size i = 0; i < param_count; ++i) {
        const char* current_param_path = compiled_material->get_parameter_name( i);
        if( strcmp( current_param_path, param_path) == 0) {
            index = i;
            break;
        }
    }
    MI_CHECK_NOT_EQUAL( index, -1);

    mi::Sint32 errors = 0;
    mi::base::Handle<const mi::IString> result(
        compiled_material->get_connected_function_db_name(
            instance_db_name.c_str(), index, &errors));
    if( !expected_connected_db_name) { // nullptr expected
        MI_CHECK_EQUAL( errors, -3);
        MI_CHECK( !result);
        return;
    }

    MI_CHECK_ZERO( errors);
    MI_CHECK( result);
    std::string connected_name = result->get_c_str();
    if( default_call) {
        // Strip varying suffix starting with the last underscore.
        size_t pos = connected_name.rfind( '_');
        MI_CHECK_NOT_EQUAL( pos, std::string::npos);
        connected_name = connected_name.substr( 0, pos);
    }
    MI_CHECK_EQUAL_CSTR( connected_name.c_str(), expected_connected_db_name);
};

void check_connected_function_db_name(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    MI_CHECK_ZERO( mdl_impexp_api->load_module(
        transaction, "::test_class_param_paths", context.get()));
    MI_CHECK_CTX( context.get());

    std::string mi_name1 = "mdl::test_class_param_paths::main_defaults";
    {
        // The parameter is a constant, not a call.
        //
        // main_defaults(lookup: some_constant)
        do_create_function_call(
            transaction,
            "mdl::test_class_param_paths::main_defaults(::test_class_param_paths::lookup_value)",
            mi_name1.c_str());

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<const mi::neuraylib::IMaterial_instance>( mi_name1.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags));
        MI_CHECK( cm);
        MI_CHECK_EQUAL( cm->get_parameter_count(), 1);
        check_path( cm, mi_name1, "lookup", nullptr);
    }

    std::string lookup_value_fc_name = "mdl::test_class_param_paths::lookup_value";
    {
        // Attach a call to lookup_value(...) as parameter.
        //
        // main_defaults(lookup: lookup_value(...))
        do_create_function_call(
            transaction,
            "mdl::test_class_param_paths::lookup_value(bool,color,float)",
            lookup_value_fc_name.c_str());
        {
            mi::neuraylib::Argument_editor ae(
                transaction, mi_name1.c_str(), mdl_factory.get(), true);
            MI_CHECK_ZERO( ae.set_call( "lookup", lookup_value_fc_name.c_str()));
        }

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<const mi::neuraylib::IMaterial_instance>( mi_name1.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags));
        MI_CHECK( cm);
        MI_CHECK_EQUAL( cm->get_parameter_count(), 3);
        check_path( cm, mi_name1, "lookup.valid", lookup_value_fc_name.c_str());
        check_path( cm, mi_name1, "lookup.value", lookup_value_fc_name.c_str());
        check_path( cm, mi_name1, "lookup.alpha", lookup_value_fc_name.c_str());
    }

    std::string mi_name2 = "mdl::test_class_param_paths::main_indirect";
    {
        // One more level of indirection (and different base material).
        //
        // main_indirect(tint: extract_value(lookup: lookup_value(...)))
        std::string extract_value_fc_name = "mdl::test_class_param_paths::extract_value";
        do_create_function_call(
            transaction,
            "mdl::test_class_param_paths::extract_value(::test_class_param_paths::lookup_value)",
            extract_value_fc_name.c_str());
        {
            mi::neuraylib::Argument_editor ae(
                transaction, extract_value_fc_name.c_str(), mdl_factory.get(), true);
            MI_CHECK_ZERO( ae.set_call( "lookup", lookup_value_fc_name.c_str()));
        }

        do_create_function_call(
            transaction,
            "mdl::test_class_param_paths::main_indirect(color)",
            mi_name2.c_str());
        {
            mi::neuraylib::Argument_editor ae(
                transaction, mi_name2.c_str(), mdl_factory.get(), true);
            MI_CHECK_ZERO( ae.set_call( "tint", extract_value_fc_name.c_str()));
        }

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<const mi::neuraylib::IMaterial_instance>( mi_name2.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags));
        MI_CHECK( cm);
        MI_CHECK_EQUAL( cm->get_parameter_count(), 3);
        check_path( cm, mi_name2, "tint.lookup.valid", lookup_value_fc_name.c_str());
        check_path( cm, mi_name2, "tint.lookup.value", lookup_value_fc_name.c_str());
        check_path( cm, mi_name2, "tint.lookup.alpha", lookup_value_fc_name.c_str());
    }
    {
        // Attach another call to the value parameter of lookup_value().
        //
        // main_indirect(tint: extract_value(lookup: lookup_value(value: create_value(...))))
        std::string create_value_fc_name = "mdl::test_class_param_paths::create_value";
        do_create_function_call(
            transaction,
            "mdl::test_class_param_paths::create_value(float)",
            create_value_fc_name.c_str());
        {
            mi::neuraylib::Argument_editor ae(
                transaction, lookup_value_fc_name.c_str(), mdl_factory.get(), true);
            MI_CHECK_ZERO( ae.set_call( "value", create_value_fc_name.c_str()));
        }

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<const mi::neuraylib::IMaterial_instance>( mi_name2.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags));
        MI_CHECK( cm);
        MI_CHECK_EQUAL( cm->get_parameter_count(), 3);
        check_path( cm, mi_name2, "tint.lookup.valid", lookup_value_fc_name.c_str());
        check_path( cm, mi_name2, "tint.lookup.value.scale", create_value_fc_name.c_str());
        check_path( cm, mi_name2, "tint.lookup.alpha", lookup_value_fc_name.c_str());
    }
    {
        // Test a material with an array as "data" parameter.
        std::string mi_name3 = "mdl::test_class_param_paths::main_array";
        do_create_function_call(
            transaction,
            "mdl::test_class_param_paths::main_array(float,float,color[4],int)",
            mi_name3.c_str());

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<const mi::neuraylib::IMaterial_instance>( mi_name3.c_str()));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags));
        MI_CHECK( cm);
        MI_CHECK_EQUAL( cm->get_parameter_count(), 5);

        check_path( cm, mi_name3, "data.value0", "mdl::T[]", true);
        check_path( cm, mi_name3, "data.value1.rgb.y", "mdl::operator*", true);
        check_path( cm, mi_name3, "data.value2.scale",
            "mdl::test_class_param_paths::create_value", true);
        check_path( cm, mi_name3, "data.value3.scale_2",
            "mdl::test_class_param_paths::create_value_2", true);
        check_path( cm, mi_name3, "index", nullptr); // constant
    }
}

void check_backends_llvm( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    auto instance_compilation = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    auto class_compilation    = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
        transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
        mi->create_compiled_material( instance_compilation));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_cc(
        mi->create_compiled_material( class_compilation));
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>("mdl::" TEST_MDL "::fd_0()"));
    mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_jit"));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be(
        mdl_backend_api->get_backend( mi::neuraylib::IMdl_backend_api::MB_LLVM_IR));
    MI_CHECK( be);

    MI_CHECK_EQUAL( 0, be->set_option( "num_texture_spaces", "16"));

    {
        // Displacement
        MI_CHECK_EQUAL( 0, be->set_option( "enable_simd", "on"));
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "on"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code->get_texture_count());

        MI_CHECK_EQUAL( 1, code->get_ro_data_segment_count());
        MI_CHECK_EQUAL_CSTR( "RO", code->get_ro_data_segment_name( 0));
        MI_CHECK_EQUAL( 1152, code->get_ro_data_segment_size( 0));
        auto data = reinterpret_cast<const float*>( code->get_ro_data_segment_data( 0));
        MI_CHECK_EQUAL( 0.0f, data[0]);
        MI_CHECK_EQUAL( 1.0f, data[1]);
        MI_CHECK_EQUAL( 2.0f, data[2]);

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());

        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant( 1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
    {
        // Regular material part
        MI_CHECK_EQUAL( 0, be->set_option( "enable_simd", "off"));
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "off"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction,
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint",
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "tint", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 2, code->get_texture_count());
        MI_CHECK_EQUAL_CSTR( "", code->get_texture( 0));
        MI_CHECK_NOT_EQUAL( nullptr, code->get_texture( 1));

        MI_CHECK_EQUAL( 0, code->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 0, code->get_string_constant_count());
    }
    {
        // Environment
        MI_CHECK_EQUAL( 0, be->set_option( "enable_simd", "off"));
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "off"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_environment(
            transaction, fc.get(), "env", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "env", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code->get_texture_count());

        MI_CHECK_EQUAL( 0, code->get_string_constant_count());
    }
    {
        // Link units
        MI_CHECK_EQUAL( 0, be->set_option( "num_texture_spaces", "16"));
        MI_CHECK_EQUAL( 0, be->set_option( "enable_simd", "on"));

        mi::base::Handle<mi::neuraylib::ILink_unit> unit(
            be->create_link_unit( transaction, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( unit);

        result = unit->add_material_expression(
            cm_ic.get(), "geometry.displacement", "displacement", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value0.component.tint",
            "tint_0", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value1.component.tint",
            "tint_1", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value2.component.tint",
            "tint_2", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_function(
            fc.get(), mi::neuraylib::ILink_unit::FEC_ENVIRONMENT, "env", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_function(
            fd.get(), mi::neuraylib::ILink_unit::FEC_CORE, nullptr, context.get());
        MI_CHECK_CTX( context);

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_link_unit( unit.get(), context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 6, code->get_callable_function_count());

        MI_CHECK_EQUAL( 4, code->get_texture_count());
        MI_CHECK_EQUAL( code->get_texture_gamma( 1),
            mi::neuraylib::ITarget_code::GM_GAMMA_DEFAULT);
        MI_CHECK_EQUAL( code->get_texture_gamma( 2),
            mi::neuraylib::ITarget_code::GM_GAMMA_LINEAR);
        MI_CHECK_EQUAL( code->get_texture_gamma( 3),
            mi::neuraylib::ITarget_code::GM_GAMMA_SRGB);

        MI_CHECK_EQUAL( code->get_texture_selector( 1), nullptr);
        MI_CHECK_EQUAL_CSTR( code->get_texture_selector( 2), "G");
        MI_CHECK_EQUAL_CSTR( code->get_texture_selector( 3), "R");

        MI_CHECK_EQUAL( 0, code->get_bsdf_measurement_count());
        MI_CHECK_EQUAL( 2, code->get_light_profile_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());
        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant( 1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
}

void check_backends_ptx( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    auto instance_compilation = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    auto class_compilation    = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
        transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
        mi->create_compiled_material( instance_compilation));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_cc(
        mi->create_compiled_material( class_compilation));
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>("mdl::" TEST_MDL "::fd_0()"));
    mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_jit"));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be(
        mdl_backend_api->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
    MI_CHECK( be);

    MI_CHECK_EQUAL( 0, be->set_option( "num_texture_spaces", "16"));
    MI_CHECK_EQUAL( 0, be->set_option( "sm_version", "50"));
    MI_CHECK_EQUAL( 0, be->set_option( "output_format", "PTX"));

    mi::Size size = 0;
    MI_CHECK( be->get_device_library( size));
    MI_CHECK( size > 0);

    {
        // Displacement, code segment off
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "off"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
            transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code->get_texture_count());

        MI_CHECK_EQUAL( 0, code->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());

        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant (1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
    {
        // Displacement, code segment on
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "on"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
            transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code->get_texture_count());

        MI_CHECK_EQUAL( 1, code->get_ro_data_segment_count());
        MI_CHECK_EQUAL_CSTR( "RO", code->get_ro_data_segment_name( 0));
        MI_CHECK_EQUAL( 1152, code->get_ro_data_segment_size( 0));
        auto data = reinterpret_cast<const float*>( code->get_ro_data_segment_data( 0));
        MI_CHECK_EQUAL( 0.0f, data[0]);
        MI_CHECK_EQUAL( 1.0f, data[1]);
        MI_CHECK_EQUAL( 2.0f, data[2]);

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());

        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant( 1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
    {
        // Regular material part
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "off"));
        MI_CHECK_EQUAL( 0, be->set_option( "output_format", "LLVM-IR"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction,
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint",
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "tint", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 2, code->get_texture_count());
        MI_CHECK_EQUAL_CSTR( "", code->get_texture(0));
        MI_CHECK_NOT_EQUAL( nullptr, code->get_texture( 1));

        MI_CHECK_EQUAL( 0, code->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 0, code->get_string_constant_count());
    }
    {
        // Environment
        MI_CHECK_EQUAL( 0, be->set_option( "enable_ro_segment", "off"));
        MI_CHECK_EQUAL( 0, be->set_option( "output_format", "PTX"));

        mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx(
            be->translate_environment(
            transaction, fc.get(), "env", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code_ptx);
        MI_CHECK( code_ptx->get_code());
        MI_CHECK( code_ptx->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code_ptx->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "env", code_ptx->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code_ptx->get_texture_count());
    }
    {
        // Link units
        mi::base::Handle<mi::neuraylib::ILink_unit> unit(
            be->create_link_unit( transaction, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( unit);

        result = unit->add_material_expression(
            cm_ic.get(), "geometry.displacement", "displacement", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value0.component.tint",
            "tint_0", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value1.component.tint",
            "tint_1", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value2.component.tint",
            "tint_2", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_function(
            fc.get(), mi::neuraylib::ILink_unit::FEC_ENVIRONMENT, "env", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_function(
            fd.get(), mi::neuraylib::ILink_unit::FEC_CORE, nullptr, context.get());
        MI_CHECK_CTX( context);

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_link_unit( unit.get(), context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 6, code->get_callable_function_count());
        MI_CHECK_EQUAL( 4, code->get_texture_count());
        MI_CHECK_EQUAL( 0, code->get_bsdf_measurement_count());
        MI_CHECK_EQUAL( 2, code->get_light_profile_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());
        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant( 1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
}

void check_backends_glsl( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    auto instance_compilation = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    auto class_compilation    = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
        transaction->access<mi::neuraylib::IMaterial_instance>( "mdl::" TEST_MDL "::mi_jit"));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_ic(
        mi->create_compiled_material( instance_compilation));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm_cc(
        mi->create_compiled_material( class_compilation));
    mi::base::Handle<const mi::neuraylib::IFunction_definition> fd(
        transaction->access<mi::neuraylib::IFunction_definition>("mdl::" TEST_MDL "::fd_0()"));
    mi::base::Handle<const mi::neuraylib::IFunction_call> fc(
        transaction->access<mi::neuraylib::IFunction_call>( "mdl::" TEST_MDL "::fc_jit"));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be(
        mdl_backend_api->get_backend( mi::neuraylib::IMdl_backend_api::MB_GLSL));
    MI_CHECK( be);

    MI_CHECK_EQUAL( 0, be->set_option( "glsl_version", "450"));
    MI_CHECK_EQUAL( 0, be->set_option( "glsl_max_const_data", "0"));
    MI_CHECK_EQUAL( 0, be->set_option( "glsl_place_uniforms_into_ssbo", "on"));
    MI_CHECK_EQUAL( 0, be->set_option( "glsl_remap_functions",
        "_ZN4base12file_textureEu10texture_2du5coloru5colorN4base9mono_modeEN4base23"
        "texture_coordinate_infoEu6float2u6float2N3tex9wrap_modeEN3tex9wrap_modeEb"
        " = my_file_texture"));

    {
        // Displacement, instance compilation
        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction, cm_ic.get(), "geometry.displacement", "displacement", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code->get_texture_count());

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());
        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant( 1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
    {
        // Displacement, class compilation
        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction, cm_cc.get(), "geometry.displacement", "displacement", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "displacement", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 1, code->get_ro_data_segment_count());
        MI_CHECK_EQUAL_CSTR( "mdl_buffer", code->get_ro_data_segment_name( 0));
        MI_CHECK_EQUAL( 1152, code->get_ro_data_segment_size( 0));
        auto data = reinterpret_cast<const float*>( code->get_ro_data_segment_data( 0));
        MI_CHECK_EQUAL( 0.0f, data[0]);
        MI_CHECK_EQUAL( 1.0f, data[1]);
        MI_CHECK_EQUAL( 2.0f, data[2]);

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 2, code->get_string_constant_count());
        MI_CHECK_EQUAL_CSTR( "abc", code->get_string_constant( 1));
    }
    {
        // Regular material part, instance compilation
        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction,
                cm_ic.get(),
                "surface.scattering.components.value0.component.tint",
                "tint",
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "tint", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 2, code->get_texture_count());
        MI_CHECK_EQUAL_CSTR( "", code->get_texture( 0));
        MI_CHECK_NOT_EQUAL( nullptr, code->get_texture( 1));

        MI_CHECK_EQUAL( 0, code->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 0, code->get_string_constant_count());
    }
    {
        // Regular material part, class compilation
        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_material_expression(
                transaction,
                cm_cc.get(),
                "surface.scattering.components.value0.component.tint",
                "tint",
                context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "tint", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 2, code->get_texture_count());
        MI_CHECK_EQUAL_CSTR( "", code->get_texture( 0));
        MI_CHECK_NOT_EQUAL( nullptr, code->get_texture( 1));

        MI_CHECK_EQUAL( 0, code->get_ro_data_segment_count());

        MI_CHECK_EQUAL( 0, code->get_code_segment_count());

        MI_CHECK_EQUAL( 0, code->get_string_constant_count());
    }
    {
        // Environment
        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_environment(
            transaction, fc.get(), "env", context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 1, code->get_callable_function_count());
        MI_CHECK_EQUAL_CSTR( "env", code->get_callable_function( 0));

        MI_CHECK_EQUAL( 0, code->get_texture_count());

        MI_CHECK_EQUAL( 1, code->get_ro_data_segment_count());
        MI_CHECK_EQUAL_CSTR( "mdl_buffer", code->get_ro_data_segment_name( 0));
        MI_CHECK_EQUAL( 1152, code->get_ro_data_segment_size( 0));
    }
    {
        // Link units
        MI_CHECK_EQUAL( 0, be->set_option( "num_texture_spaces", "16"));

        mi::base::Handle<mi::neuraylib::ILink_unit> unit(
            be->create_link_unit( transaction, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( unit);

        result = unit->add_material_expression(
            cm_ic.get(), "geometry.displacement", "displacement", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value0.component.tint",
            "tint_0", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value1.component.tint",
            "tint_1", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_material_expression(
            cm_ic.get(),
            "surface.scattering.components.value2.component.tint",
            "tint_2", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_function(
            fc.get(), mi::neuraylib::ILink_unit::FEC_ENVIRONMENT, "env", context.get());
        MI_CHECK_CTX( context);

        result = unit->add_function(
            fd.get(), mi::neuraylib::ILink_unit::FEC_CORE, nullptr, context.get());
        MI_CHECK_CTX( context);

        mi::base::Handle<const mi::neuraylib::ITarget_code> code(
            be->translate_link_unit( unit.get(), context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( code);
        MI_CHECK( code->get_code());
        MI_CHECK( code->get_code_size() > 0);

        MI_CHECK_EQUAL( 6, code->get_callable_function_count());
        MI_CHECK_EQUAL( 4, code->get_texture_count());
        MI_CHECK_EQUAL( 0, code->get_bsdf_measurement_count());
        MI_CHECK_EQUAL( 2, code->get_light_profile_count());

        MI_CHECK_EQUAL( 3, code->get_string_constant_count());
        MI_CHECK_EQUAL_CSTR( "something", code->get_string_constant( 1));
        MI_CHECK_EQUAL_CSTR( "abc",       code->get_string_constant( 2));
    }
}

void check_multiscatter_textures( mi::neuraylib::INeuray* neuray)
{
    struct Df_data
    {
        mi::neuraylib::Df_data_kind kind;
        mi::Size rx;
        mi::Size ry;
        mi::Size rz;
        const char* pixel_type;
    };

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

    std::vector<Df_data> df_data = {
        { mi::neuraylib::DFK_SIMPLE_GLOSSY_MULTISCATTER, 65, 64, 33, "Float32" },
        { mi::neuraylib::DFK_BACKSCATTERING_GLOSSY_MULTISCATTER, 65, 64, 1, "Float32" },
        { mi::neuraylib::DFK_BECKMANN_SMITH_MULTISCATTER, 65, 64, 33, "Float32" },
        { mi::neuraylib::DFK_BECKMANN_VC_MULTISCATTER, 65, 64, 33, "Float32" },
        { mi::neuraylib::DFK_GGX_SMITH_MULTISCATTER, 65, 64, 33, "Float32" },
        { mi::neuraylib::DFK_GGX_VC_MULTISCATTER, 65, 64, 33, "Float32" },
        { mi::neuraylib::DFK_WARD_GEISLER_MORODER_MULTISCATTER, 65, 64, 1, "Float32" },
        { mi::neuraylib::DFK_SHEEN_MULTISCATTER, 65, 64, 1, "Float32" },
        { mi::neuraylib::DFK_MICROFLAKE_SHEEN_GENERAL, 32, 32, 1, "Float32<3>" },
        { mi::neuraylib::DFK_MICROFLAKE_SHEEN_MULTISCATTER, 65, 64, 1, "Float32" }
    };

    const float* data = nullptr;
    const char* pixel_type = nullptr;
    mi::Size rx, ry, rz;

    for( const auto& entry : df_data) {
        data = mdl_backend_api->get_df_data_texture( entry.kind, rx, ry, rz, pixel_type);
        MI_CHECK( data);
        MI_CHECK_EQUAL( entry.rx, rx);
        MI_CHECK_EQUAL( entry.ry, ry);
        MI_CHECK_EQUAL( entry.rz, rz);
        MI_CHECK_EQUAL_CSTR( entry.pixel_type, pixel_type);
    }

    data = mdl_backend_api->get_df_data_texture( mi::neuraylib::DFK_NONE, rx, ry, rz, pixel_type);
    MI_CHECK( !data);

    data = mdl_backend_api->get_df_data_texture( mi::neuraylib::DFK_INVALID, rx, ry, rz, pixel_type);
    MI_CHECK( !data);
}

void check_distiller( mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_distiller_api> mdl_distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // Distill a compiled material from instance compilation
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>(
                "mdl::" TEST_MDL "::mi_resource_sharing"));

        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( cm);

        mi::Sint32 errors;
        mi::base::Handle<const mi::neuraylib::ICompiled_material> new_cm(
            mdl_distiller_api->distill_material(
                cm.get(), "diffuse", /*distiller_options*/ nullptr, &errors));
        MI_CHECK_EQUAL( errors, 0);
        MI_CHECK( new_cm);
    }
    {
        // Distill a compiled material from class compilation
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>(
                "mdl::" TEST_MDL "::mi_resource_sharing"));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
            mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( cm);
        MI_CHECK( cm->get_parameter_count() > 0);

        mi::Sint32 errors;
        mi::base::Handle<const mi::neuraylib::ICompiled_material> new_cm(
            mdl_distiller_api->distill_material(
                cm.get(), "diffuse", /*distiller_options*/ nullptr, &errors));
        MI_CHECK_EQUAL( errors, 0);
        MI_CHECK( new_cm);
    }
}

void check_distiller_for_multiple_targets(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray)
{
    // This test invokes the distiller on several materials, distilling each to all available
    // targets in the mdl_distiller plugin. Then, the hashes of all distilled materials are
    // compared to make sure that they differwhen expected.

    mi::base::Handle<mi::neuraylib::IMdl_distiller_api> mdl_distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Tested materials.
    const char* materials[] = {
        "mdl::" TEST_MDL "::mi_0",
        "mdl::" TEST_MDL "::mi_1"
    };
    size_t n_materials = sizeof(materials) / sizeof(materials[0]);

    // Tested targets. We only test against mdl_distiller plugin for now.
    const char* targets[] = {
        "diffuse",
        "specular_glossy",
        "ue4",
        "transmissive_pbr"
    };
    const size_t n_targets = sizeof( targets) / sizeof( targets[0]);

    mi::base::Uuid hashes[n_targets];

    for( size_t m = 0; m < n_materials; ++m) {

        mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
            transaction->access<mi::neuraylib::IMaterial_instance>( materials[m]));
        mi::base::Handle<const mi::neuraylib::ICompiled_material> orig_cm(
            mi->create_compiled_material(
                mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, context.get()));
        MI_CHECK_CTX( context);
        MI_CHECK( orig_cm);
        mi::base::Uuid orig_hash = orig_cm->get_hash();

        for( size_t i = 0; i < n_targets; i++) {

            mi::Sint32 errors;
            mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_cm(
                mdl_distiller_api->distill_material(
                    orig_cm.get(), targets[i], /*distiller_options*/ nullptr, &errors));
            MI_CHECK_EQUAL( errors, 0);
            MI_CHECK( distilled_cm);
            hashes[i] = distilled_cm->get_hash();

            // Check that the distilled hash is different from the original hash.
            MI_CHECK( hashes[i] != orig_hash);

            // Check that the distilled hash is different from earlier targets.
            for( size_t j = 0; j < i; j++)
                MI_CHECK( hashes[i] != hashes[j]);
        }
    }
}

void check_baker(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    bool first_run,
    const std::string& pixel_type,
    const std::string& type_name,
    bool use_constant_detection,
    bool constant_expression,
    const std::string& instance_name,
    const std::string& path,
    const std::string& file_name_prefix,
    mi::neuraylib::IMaterial_instance::Compilation_options compilation_flags,
    mi::neuraylib::Baker_resource baker_flags)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // create compiled material with the given flags
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
        transaction->access<mi::neuraylib::IMaterial_instance>( instance_name.c_str()));
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( mi->create_compiled_material(
        compilation_flags, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK( cm);

    // create baker for the given path and flags
    mi::base::Handle<mi::neuraylib::IMdl_distiller_api> mdl_distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    mi::base::Handle<const mi::neuraylib::IBaker> baker(
        mdl_distiller_api->create_baker( cm.get(), path.c_str(), baker_flags, /*gpu_device_id*/ 0));
    MI_CHECK( baker);
    MI_CHECK_EQUAL( baker->is_uniform(), false);
    MI_CHECK_EQUAL( pixel_type, baker->get_pixel_type());
    MI_CHECK_EQUAL( type_name, baker->get_type_name());

    // allocate canvas
    mi::Uint32 width = 100;
    mi::Uint32 height = 100;
    mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        neuray->get_api_component<mi::neuraylib::IImage_api>());
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas( pixel_type.c_str(), width, height));

    // allocate constant
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    mi::base::Handle<mi::IData> constant(
        factory->create<mi::IData>( type_name.c_str()));

    // run baker
    bool is_constant = false;
    if( use_constant_detection) {
        result = baker->bake_texture_with_constant_detection(
            canvas.get(), constant.get(), is_constant, /*samples*/ 1);
    } else {
        result = baker->bake_texture( canvas.get(), /*samples*/ 1);
        MI_CHECK_EQUAL( 0, result);
    }

    // export baked texture
    std::string output_name = std::string( DIR_PREFIX) + dir_sep + file_name_prefix;
    output_name += first_run
        ? "_1st_run"
        : "_2nd_run";
    output_name += compilation_flags == mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS
        ? "_instance"
        : "_class";
    output_name += baker_flags == mi::neuraylib::BAKE_ON_CPU
        ? "_cpu"
        : "_gpu_with_cpu_fallback";
    output_name += ".png";

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    result = mdl_impexp_api->export_canvas( output_name.c_str(), canvas.get());
    MI_CHECK_EQUAL( 0, result);

    // check that constant detection recognizes constants
    if( use_constant_detection) {
        MI_CHECK_EQUAL( is_constant, constant_expression);
    }

    // check results
    if( !use_constant_detection || !constant_expression) {

        // compare baked texture with reference texture
        std::string ref_name
            = MI::TEST::mi_src_path( "prod/lib/neuray/reference/") + file_name_prefix + ".png";
        MI_CHECK_IMG_DIFF( output_name, ref_name, /*strict*/ false);

    } else {

        // compare baked constant with reference value
        if( pixel_type == "Rgb_fp") {

            mi::base::Handle color( constant->get_interface<mi::IColor>());
            mi::Color v;
            color->get_value( v);
            MI_CHECK_EQUAL( v.r, 0.0f);
            MI_CHECK_EQUAL( v.g, 1.0f);
            MI_CHECK_EQUAL( v.b, 0.0f);
            MI_CHECK_EQUAL( v.a, 1.0f);

        } else if( pixel_type == "Float32") {

            mi::base::Handle float32( constant->get_interface<mi::IFloat32>());
            MI_CHECK_EQUAL( float32->get_value<mi::Float32>(), 1.0f);

        } else {
            MI_CHECK( false);
        }
    }
}

void check_baker(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    bool first_run,
    const std::string& pixel_type,
    const std::string& type_name,
    bool use_constant_detection,
    bool constant_expression)
{
    MI_CHECK( pixel_type == "Rgb_fp" || pixel_type == "Float32");
    bool is_rgb_fp = pixel_type == "Rgb_fp";

    std::string instance_name = constant_expression
        ? "mdl::" TEST_MDL "::mi_baking_const"
        : "mdl::" TEST_MDL "::mi_baking";

    std::string path = is_rgb_fp
        ? "surface.scattering.tint"
        : "backface.scattering.tint.r";

    std::string file_name_prefix = "baker";
    file_name_prefix += is_rgb_fp
        ? "_rgb_fp"
        : "_float32";
    file_name_prefix += constant_expression
        ? "_constant"
        : "_non_constant";

    // instance compilation / CPU
    check_baker(
        transaction, neuray, first_run, pixel_type, type_name, use_constant_detection,
        constant_expression, instance_name, path, file_name_prefix,
        mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS,
        mi::neuraylib::BAKE_ON_CPU);

    // class compilation / CPU
    check_baker(
        transaction, neuray, first_run, pixel_type, type_name, use_constant_detection,
        constant_expression, instance_name, path, file_name_prefix,
        mi::neuraylib::IMaterial_instance::CLASS_COMPILATION,
        mi::neuraylib::BAKE_ON_CPU);

    if( true) {

    // instance compilation / GPU
    check_baker(
        transaction, neuray, first_run, pixel_type, type_name, use_constant_detection,
        constant_expression, instance_name, path, file_name_prefix,
        mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS,
        mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK);

    // class compilation / GPU
    check_baker(
        transaction, neuray, first_run, pixel_type, type_name, use_constant_detection,
        constant_expression, instance_name, path, file_name_prefix,
        mi::neuraylib::IMaterial_instance::CLASS_COMPILATION,
        mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK);

    }
}

void check_baker(
    mi::neuraylib::ITransaction* transaction, mi::neuraylib::INeuray* neuray, bool first_run)
{
#ifndef RESOLVE_RESOURCES_FALSE

    // Rgb_fp / without constant detection / non-constant expression
    check_baker( transaction, neuray, first_run, "Rgb_fp", "Color", false, false);
    // Rgb_fp / without constant detection / constant expression
    check_baker( transaction, neuray, first_run, "Rgb_fp", "Color", false, true);
    // Rgb_fp / constant detection / non-constant expression
    check_baker( transaction, neuray, first_run, "Rgb_fp", "Color", true, false);
    // Rgb_fp / constant detection / constant expression
    check_baker( transaction, neuray, first_run, "Rgb_fp", "Color", true, true);

    // Float32 / without constant detection / non-constant expression
    check_baker( transaction, neuray, first_run, "Float32", "Float32", false, false);
    // Float32 / without constant detection / constant expression
    check_baker( transaction, neuray, first_run, "Float32", "Float32", false, true);
    // Float32 / constant detection / non-constant expression
    check_baker( transaction, neuray, first_run, "Float32", "Float32", true, false);
    // Float32 / constant detection / constant expression
    check_baker( transaction, neuray, first_run, "Float32", "Float32", true, true);

#endif // RESOLVE_RESOURCES_FALSE
}

const mi::neuraylib::ICompiled_material* compile_material_tmm(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    bool target_material_model_mode,
    bool target_type,
    bool class_compilation)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    context->set_option( "target_material_model_mode", target_material_model_mode);

    if( target_type) {
        mi::base::Handle<mi::neuraylib::IType_factory> tf(
            mdl_factory->create_type_factory( transaction));
        mi::base::Handle<const mi::neuraylib::IType> standard_material_type(
            tf->get_predefined_struct( mi::neuraylib::IType_struct::SID_MATERIAL));
        context->set_option( "target_type", standard_material_type.get());
    }

    result = mdl_impexp_api->load_module(
        transaction, "::test_target_material_model", context.get());
    MI_CHECK_CTX( context);
    MI_CHECK_EQUAL( result, 0);

    do_create_function_call(
        transaction,
        "mdl::test_target_material_model::md_tmm()",
        "mdl::test_target_material_model::mi_tmm");

    mi::base::Handle mi( transaction->access<mi::neuraylib::IMaterial_instance>(
        "mdl::test_target_material_model::mi_tmm"));
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
            mi->create_compiled_material( flags, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK( cm);

    return cm.extract();
}

void check_target_material_mode_w_wo_target_type(
    mi::neuraylib::IDatabase* database,
    mi::neuraylib::INeuray* neuray,
    bool target_type,
    bool class_compilation)
{
    mi::base::Handle<mi::neuraylib::IScope> child_scope( database->create_scope( nullptr, 1));
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(
        child_scope->create_transaction());

    {
        // target material mode enabled
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( compile_material_tmm(
            transaction.get(), neuray, true, target_type, class_compilation));
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());

        const char* def_name = body->get_definition();
        MI_CHECK_EQUAL_CSTR( def_name, "mdl::test_target_material_model::md_target_material()");

        mi::base::Handle md( transaction->access<mi::neuraylib::IFunction_definition>( def_name));
        MI_CHECK( md);
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> annos( md->get_annotations());
        MI_CHECK_EQUAL( annos->get_size(), 1);
        mi::base::Handle<const mi::neuraylib::IAnnotation> anno( annos->get_annotation( 0));
        const char* anno_name = anno->get_name();
        MI_CHECK_EQUAL_CSTR( anno_name, "::nvidia::baking::target_material_model(string,string)");
    }

    transaction->commit();
    child_scope = database->create_scope( nullptr, 1);
    transaction = child_scope->create_transaction();

    {
        // target material mode disabled
        mi::base::Handle<const mi::neuraylib::ICompiled_material> cm( compile_material_tmm(
            transaction.get(), neuray, false, target_type, class_compilation));
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body( cm->get_body());

        const char* def_name = body->get_definition();
        MI_CHECK_EQUAL_CSTR( def_name,
            "mdl::material(bool,material_surface,material_surface,color,material_volume,"
            "material_geometry,hair_bsdf)");
    }

    transaction->commit();
}

void check_target_material_mode( mi::neuraylib::IDatabase* database, mi::neuraylib::INeuray* neuray)
{
    // instance compilation
    check_target_material_mode_w_wo_target_type( database, neuray, false, false);
    check_target_material_mode_w_wo_target_type( database, neuray, true, false);
    // class compilation
    check_target_material_mode_w_wo_target_type( database, neuray, false, true);
    check_target_material_mode_w_wo_target_type( database, neuray, true, true);
}

void run_tests( mi::neuraylib::INeuray* neuray, bool first_run)
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

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        install_external_resolver( neuray);

        // run the actual tests

        // Load the primary modules.
        load_primary_modules( transaction.get(), neuray);

        // Create function calls used by many later tests.
        create_function_calls( transaction.get(), mdl_factory.get());

        // Compiled materials
        check_icompiled_material( transaction.get(), neuray, false);
        check_icompiled_material( transaction.get(), neuray, true);
        check_folding( transaction.get(), neuray, false);
        check_folding( transaction.get(), neuray, true);
        check_type_binding( transaction.get(), mdl_factory.get());
        check_hashing( transaction.get(), neuray);
        check_connected_function_db_name( transaction.get(), neuray);
        check_target_material_mode( database.get(), neuray); // uses separate scopes/transactions

        // Backends
        check_backends_llvm( transaction.get(), neuray);
        check_backends_ptx( transaction.get(), neuray);
        check_backends_glsl( transaction.get(), neuray);
        check_multiscatter_textures( neuray);

        // Distiller and baker
        check_distiller( transaction.get(), neuray);
        check_distiller_for_multiple_targets( transaction.get(), neuray);
        check_baker( transaction.get(), neuray, first_run);

        MI_CHECK_EQUAL( 0, transaction->commit());

        uninstall_external_resolver( neuray);
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_icompiled_material )
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
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_mdl_distiller));
        MI_CHECK_EQUAL( 0, plugin_configuration->load_plugin_library( plugin_path_openimageio));

        run_tests( neuray.get(), /*first_run*/ true);
        // MDL SDK must be able to run the test a second time, test that
        run_tests( neuray.get(), /*first_run*/ false);
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

