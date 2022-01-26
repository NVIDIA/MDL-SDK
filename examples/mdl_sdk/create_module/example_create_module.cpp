/******************************************************************************
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/create_module/example_create_module.cpp
//
// Create a new MDL module

#include <iostream>
#include <string>

#include "example_shared.h"

mi::neuraylib::IAnnotation_block* create_annotations(
    mi::neuraylib::IValue_factory* vf,
    mi::neuraylib::IExpression_factory* ef,
    const char* description)
{
    mi::base::Handle<mi::neuraylib::IValue> arg_value( vf->create_string( description));
    mi::base::Handle<mi::neuraylib::IExpression> arg_expr(
        ef->create_constant( arg_value.get()));
    mi::base::Handle<mi::neuraylib::IExpression_list> args( ef->create_expression_list());
    check_success( 0 == args->add_expression( "description", arg_expr.get()));
    mi::base::Handle<mi::neuraylib::IAnnotation> annotation(
         ef->create_annotation( "::anno::description(string)", args.get()));

    mi::base::Handle<mi::neuraylib::IAnnotation_block> annotation_block(
        ef->create_annotation_block());
    annotation_block->add_annotation( annotation.get());

    annotation_block->retain();
    return annotation_block.get();
}

void create_module(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_factory* mdl_factory)
{
    mi::base::Handle<mi::neuraylib::IType_factory> tf(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
        mdl_factory->create_expression_factory( transaction));
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Load modules.
    check_success( mdl_impexp_api->load_module(
        transaction, "::df", context.get()) >= 0);

    // Create the module builder.
    //
    // We start building a module "::new_module" that does not yet exist in the DB.
    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            transaction,
            "mdl::new_module",
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));

    {
        // (1) Create a simple diffuse material.
        //
        // export material diffuse_material(color tint)
        // = material(
        //     surface: material_surface(scattering: df::diffuse_reflection_bsdf(tint: tint))
        // );

        // Create parameters.
        mi::base::Handle<const mi::neuraylib::IType_color> tint_type( tf->create_color());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "tint", tint_type.get());

        // Create body.
        mi::base::Handle<mi::neuraylib::IExpression> drb_tint(
            ef->create_parameter( tint_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> drb_args(
            ef->create_expression_list());
        drb_args->add_expression( "tint", drb_tint.get());
        mi::base::Handle<const mi::neuraylib::IExpression> drb(
            ef->create_direct_call(
                 "mdl::df::diffuse_reflection_bsdf(color,float,string)", drb_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> surface_args(
            ef->create_expression_list());
        surface_args->add_expression( "scattering", drb.get());
        mi::base::Handle<const mi::neuraylib::IExpression> surface(
            ef->create_direct_call(
                "mdl::material_surface(bsdf,material_emission)", surface_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "surface", surface.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call(
                "mdl::material(bool,material_surface,material_surface,color,material_volume,"
                "material_geometry,hair_bsdf)", body_args.get()));

        // Add the material to the module.
        mi::Sint32 result = module_builder->add_function(
            "diffuse_material",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());

        print_messages( context.get());
        check_success( result == 0);
        check_success( context->get_error_messages_count() == 0);
    }
    {
        // (2) Create a simple red material as a variant of "diffuse_material".
        //
        // export material red_material(*) [[ anno::description("A diffuse red material") ]]
        // = diffuse_material(tint: color(1.f, 0.f, 0.f));

        // Create defaults.
        mi::base::Handle<mi::neuraylib::IValue> tint_value(
            vf->create_color( 1.0f, 0.0f, 0.0f));
        mi::base::Handle<mi::neuraylib::IExpression> tint_expr(
            ef->create_constant( tint_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
            ef->create_expression_list());
        defaults->add_expression( "tint", tint_expr.get());

        // Create annotations.
        mi::base::Handle<mi::neuraylib::IAnnotation_block> annotations(
            create_annotations( vf.get(), ef.get(), "A diffuse red material"));

        // Add the variant to the module.
        mi::Sint32 result = module_builder->add_variant(
            "red_material",
            "mdl::new_module::diffuse_material(color)",
            defaults.get(),
            annotations.get(),
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());

        print_messages( context.get());
        check_success( result == 0);
        check_success( context->get_error_messages_count() == 0);
    }

    // Re-create the module builder.
    //
    // We could continue to use the existing module builder, but for demonstration purposes we
    // ignore how "::new_module" was created and show now how to use the module builder to edit a
    // module that exists already in the DB.
    module_builder = mdl_factory->create_module_builder(
        transaction,
        "mdl::new_module",
        /*ignored*/ mi::neuraylib::MDL_VERSION_1_0,
        mi::neuraylib::MDL_VERSION_LATEST,
        context.get());

    {
        // (3) Use the uniform analysis to figure out required uniform modifiers of parameters.
        //
        // export material uniform_parameter(uniform bool across_materials)
        // = diffuse_material(
        //     tint: state::rounded_corner_normal(0.f, across_materials, 1.f)
        // );

        // Create parameters (without uniform modifier at first).
        mi::base::Handle<const mi::neuraylib::IType> across_materials_type( tf->create_bool());
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "across_materials", across_materials_type.get());

        // Create body.
        mi::base::Handle<mi::neuraylib::IExpression> across_materials_expr(
            ef->create_parameter( across_materials_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression_list> rcn_args(
            ef->create_expression_list());
        rcn_args->add_expression( "across_materials", across_materials_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> rcn(
            ef->create_direct_call( "mdl::state::rounded_corner_normal$1.2(float,bool)",
                rcn_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> color_args(
            ef->create_expression_list());
        color_args->add_expression( "rgb", rcn.get());
        mi::base::Handle<const mi::neuraylib::IExpression> color(
            ef->create_direct_call( "mdl::color(float3)", color_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "tint", color.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( "mdl::new_module::diffuse_material(color)", body_args.get()));

        // Run uniform analysis.
        mi::base::Handle<const mi::IArray> uniform( module_builder->analyze_uniform(
            body.get(), /*root_uniform*/ false, context.get()));
        print_messages( context.get());
        check_success( context->get_error_messages_count() == 0);

        // Adapt parameter types based on uniform analysis.
        mi::base::Handle<mi::neuraylib::IType_list> fixed_parameters( tf->create_type_list());
        for( mi::Size i = 0, n = uniform->get_length(); i < n; ++i) {
            mi::base::Handle<const mi::neuraylib::IType> parameter( parameters->get_type( i));
            mi::base::Handle<const mi::IBoolean> element( uniform->get_element<mi::IBoolean>( i));
            if( element->get_value<bool>()) {
                check_success( i == 0);
                parameter = tf->create_alias(
                    parameter.get(), mi::neuraylib::IType::MK_UNIFORM, /*symbol*/ nullptr);
            } else {
                check_success( i != 0);
            }
            const char* name = parameters->get_name( i);
            fixed_parameters->add_type( name, parameter.get());
        }
        parameters = fixed_parameters;

        // Add the material to the module.
        mi::Sint32 result = module_builder->add_function(
            "uniform_parameter",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());

        print_messages( context.get());
        check_success( result == 0);
        check_success( context->get_error_messages_count() == 0);
    }
    {
        // (4a) Create two new types.
        //
        // export enum Enum { Add = 0, Sub = 1 };
        // export struct Struct { int x; int y; };

        mi::base::Handle<mi::neuraylib::IValue> add_value( vf->create_int( 0));
        mi::base::Handle<mi::neuraylib::IExpression> add_expr(
            ef->create_constant( add_value.get()));
        mi::base::Handle<mi::neuraylib::IValue> sub_value( vf->create_int( 1));
        mi::base::Handle<mi::neuraylib::IExpression> sub_expr(
            ef->create_constant( sub_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> enumerators(
            ef->create_expression_list());
        enumerators->add_expression( "Add", add_expr.get());
        enumerators->add_expression( "Sub", sub_expr.get());

        mi::Sint32 result = module_builder->add_enum_type(
            "Enum",
            enumerators.get(),
            /*enumerator_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());

        print_messages( context.get());
        check_success( result == 0);
        check_success( context->get_error_messages_count() == 0);

        mi::base::Handle<const mi::neuraylib::IType> x( tf->create_int());
        mi::base::Handle<const mi::neuraylib::IType> y( tf->create_int());
        mi::base::Handle<mi::neuraylib::IType_list> fields( tf->create_type_list());
        fields->add_type( "x", x.get());
        fields->add_type( "y", y.get());

        result = module_builder->add_struct_type(
            "Struct",
            fields.get(),
            /*field_defaults*/ nullptr,
            /*field_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*is_exported*/ true,
            context.get());

        print_messages( context.get());
        check_success( result == 0);
        check_success( context->get_error_messages_count() == 0);

        // (4b) Create a function that operates on these types.
        //
        // export int types(Enum e, Struct s) { return e == ADD ? s.x + s.y : s.x - s.y; }
        //
        // Since there are no comparison operators for user-defined types, we actually compute
        // "int(e) == 0" instead of "e == ADD".

        // Create parameters
        mi::base::Handle<const mi::neuraylib::IType_enum> e_type(
            tf->create_enum( "::new_module::Enum"));
        mi::base::Handle<const mi::neuraylib::IType_struct> s_type(
            tf->create_struct( "::new_module::Struct"));
        mi::base::Handle<mi::neuraylib::IType_list> parameters( tf->create_type_list());
        parameters->add_type( "e", e_type.get());
        parameters->add_type( "s", s_type.get());

        // Create body.
        mi::base::Handle<mi::neuraylib::IExpression> parameter_e(
            ef->create_parameter( e_type.get(), 0));
        mi::base::Handle<mi::neuraylib::IExpression> parameter_s(
            ef->create_parameter( s_type.get(), 1));

        mi::base::Handle<mi::neuraylib::IExpression_list> e_int_args(
            ef->create_expression_list());
        e_int_args->add_expression( "x", parameter_e.get());
        mi::base::Handle<const mi::neuraylib::IExpression> e_int(
            ef->create_direct_call( "mdl::new_module::int(::new_module::Enum)", e_int_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> struct_selector_args(
            ef->create_expression_list());
        struct_selector_args->add_expression( "s", parameter_s.get());
        mi::base::Handle<const mi::neuraylib::IExpression> s_x(
            ef->create_direct_call(
                "mdl::new_module::Struct.x(::new_module::Struct)", struct_selector_args.get()));
        mi::base::Handle<const mi::neuraylib::IExpression> s_y(
            ef->create_direct_call(
                "mdl::new_module::Struct.y(::new_module::Struct)", struct_selector_args.get()));

        mi::base::Handle<mi::neuraylib::IValue> equal_y_value( vf->create_int( 0));
        mi::base::Handle<mi::neuraylib::IExpression> equal_y_expr(
            ef->create_constant( equal_y_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> equal_args(
            ef->create_expression_list());
        equal_args->add_expression( "x", e_int.get());
        equal_args->add_expression( "y", equal_y_expr.get());
        mi::base::Handle<const mi::neuraylib::IExpression> equal(
            ef->create_direct_call( "mdl::operator==(int,int)", equal_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> add_args(
            ef->create_expression_list());
        add_args->add_expression( "x", s_x.get());
        add_args->add_expression( "y", s_y.get());
        mi::base::Handle<const mi::neuraylib::IExpression> add(
            ef->create_direct_call( "mdl::operator+(int,int)", add_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> sub_args(
            ef->create_expression_list());
        sub_args->add_expression( "x", s_x.get());
        sub_args->add_expression( "y", s_y.get());
        mi::base::Handle<const mi::neuraylib::IExpression> sub(
            ef->create_direct_call( "mdl::operator-(int,int)", sub_args.get()));

        mi::base::Handle<mi::neuraylib::IExpression_list> body_args(
            ef->create_expression_list());
        body_args->add_expression( "cond", equal.get());
        body_args->add_expression( "true_exp", add.get());
        body_args->add_expression( "false_exp", sub.get());
        mi::base::Handle<const mi::neuraylib::IExpression> body(
            ef->create_direct_call( "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)", body_args.get()));

        // Add the function to the module.
        result = module_builder->add_function(
            "types",
            body.get(),
            parameters.get(),
            /*defaults*/ nullptr,
            /*parameter_annotations*/ nullptr,
            /*annotations*/ nullptr,
            /*return_annotations*/ nullptr,
            /*is_exported*/ true,
            /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
            context.get());

        print_messages( context.get());
        check_success( result == 0);
        check_success( context->get_error_messages_count() == 0);
    }

    // Print the exported MDL source code to the console.
    mi::base::Handle<mi::IString> module_source( transaction->create<mi::IString>( "String"));
    mi::Sint32 result = mdl_impexp_api->export_module_to_string(
        transaction, "mdl::new_module", module_source.get(), context.get());

    print_messages( context.get());
    check_success( result == 0);
    check_success( context->get_error_messages_count() == 0);

    std::cerr << module_source->get_c_str();
}

int MAIN_UTF8( int argc, char* argv[])
{
    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray( mi::examples::mdl::load_and_get_ineuray());
    if( !neuray.is_valid_interface())
        exit_failure( "Failed to load the SDK.");

    // Configure the MDL SDK
    if( !mi::examples::mdl::configure( neuray.get(), /*mdl_paths=*/{}))
        exit_failure( "Failed to initialize the SDK.");

    {
        // Start the MDL SDK
        mi::Sint32 result = neuray->start();
        if( result != 0)
            exit_failure( "Failed to initialize the SDK. Result code: %d", result);

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        // Access the database and create a transaction.
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

        create_module( transaction.get(), mdl_impexp_api.get(), mdl_factory.get());

        transaction->commit();
    }

    // Shut down the MDL SDK
    check_success( neuray->shutdown() == 0);
    neuray = 0;

    // Unload the MDL SDK
    neuray = nullptr;
    if( !mi::examples::mdl::unload())
        exit_failure( "Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
