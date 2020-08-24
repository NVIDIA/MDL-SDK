/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/calls/example_calls.cpp
//
// Uses function calls to create a textured material.

#include <iostream>
#include <string>

#include "example_shared.h"

// Utility function to dump the arguments of a material instance or function call.
template <class T>
void dump_instance(
    mi::neuraylib::IExpression_factory* expression_factory, const T* material, std::ostream& s)
{
    mi::Size count = material->get_parameter_count();
    mi::base::Handle<const mi::neuraylib::IExpression_list> arguments( material->get_arguments());

    for( mi::Size index = 0; index < count; index++) {

        mi::base::Handle<const mi::neuraylib::IExpression> argument(
            arguments->get_expression( index));
        std::string name = material->get_parameter_name( index);
        mi::base::Handle<const mi::IString> argument_text(
            expression_factory->dump( argument.get(), name.c_str(), 1));
        s << "    argument " << argument_text->get_c_str() << std::endl;

    }
    s << std::endl;
}

// Creates a textured material.
void create_textured_material( mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
        mdl_factory->create_value_factory( transaction.get()));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
        mdl_factory->create_expression_factory( transaction.get()));

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    {
        // Create a DB element for the image and the texture referencing it.
        mi::base::Handle<mi::neuraylib::IImage> image(
            transaction->create<mi::neuraylib::IImage>( "Image"));
        check_success( image->reset_file( "nvidia/sdk_examples/resources/example.png") == 0);
        transaction->store( image.get(), "nvidia_image");
        mi::base::Handle<mi::neuraylib::ITexture> texture(
            transaction->create<mi::neuraylib::ITexture>( "Texture"));
        texture->set_image( "nvidia_image");
        transaction->store( texture.get(), "nvidia_texture");
    }
    {
        // Import the "::nvidia::sdk_examples::tutorials" and "base" module.
        // The "::nvidia::sdk_examples::tutorials" module is found via the
        // configured module search path.
        check_success( mdl_impexp_api->load_module(
            transaction.get(), "::nvidia::sdk_examples::tutorials", context.get()) >= 0);
        check_success( print_messages( context.get()));
        check_success( mdl_impexp_api->load_module(
            transaction.get(), "::base", context.get()) >= 0);
        check_success( print_messages( context.get()));
    }
    {
        // Lookup the exact name of the DB element for the MDL function "base::file_texture".
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::base"));
        mi::base::Handle<const mi::IArray> overloads(
            module->get_function_overloads( "mdl::base::file_texture"));
        check_success( overloads->get_length() == 1);
        mi::base::Handle<const mi::IString> file_texture_name(
            overloads->get_element<mi::IString>( static_cast<mi::Uint32>( 0)));

        // Prepare the arguments of the function call for "mdl::base::file_texture": set the
        // "texture" argument to the "nvidia_texture" texture.
        mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
            transaction->access<mi::neuraylib::IFunction_definition>(
                file_texture_name->get_c_str()));
        mi::base::Handle<const mi::neuraylib::IType_list> types(
            function_definition->get_parameter_types());
        mi::base::Handle<const mi::neuraylib::IType> arg_type( types->get_type( "texture"));
        check_success( arg_type.is_valid_interface());
        mi::base::Handle<mi::neuraylib::IValue_texture> arg_value(
            value_factory->create<mi::neuraylib::IValue_texture>( arg_type.get()));
        check_success( arg_value.is_valid_interface());
        check_success( arg_value->set_value( "nvidia_texture") == 0);
        mi::base::Handle<mi::neuraylib::IExpression> arg_expr(
            expression_factory->create_constant( arg_value.get()));
        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            expression_factory->create_expression_list());
        arguments->add_expression( "texture", arg_expr.get());

        // Create a function call from the function definition "mdl::base::file_texture" with the
        // just prepared arguments.
        mi::Sint32 result;
        mi::base::Handle<mi::neuraylib::IFunction_call> function_call(
            function_definition->create_function_call( arguments.get(), &result));
        check_success( result == 0);
        transaction->store( function_call.get(), "call of file_texture");
    }
    {
        // Prepare the arguments of the function call for "mdl::base::texture_return.tint": set the
        // "s" argument to the "call of file_texture" function call.
        mi::base::Handle<mi::neuraylib::IExpression> arg_expr(
            expression_factory->create_call( "call of file_texture"));
        check_success( arg_expr.is_valid_interface());
        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            expression_factory->create_expression_list());
        arguments->add_expression( "s", arg_expr.get());

        // Create a function call from the function definition "mdl::base::file_texture" with the
        // just prepared arguments.
        mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
            transaction->access<mi::neuraylib::IFunction_definition>(
                "mdl::base::texture_return.tint(::base::texture_return)"));
        mi::Sint32 result;
        mi::base::Handle<mi::neuraylib::IFunction_call> function_call(
            function_definition->create_function_call( arguments.get(), &result));
        check_success( result == 0);
        transaction->store( function_call.get(), "call of texture_return.tint");
    }
    {
        // Prepare the arguments of the material instance for 
        // "mdl::nvidia::sdk_examples::tutorials::example_material":
        // set the "tint" argument to the "call of texture_return.tint" function call.
        mi::base::Handle<mi::neuraylib::IExpression> arg_expr(
            expression_factory->create_call( "call of texture_return.tint"));
        check_success( arg_expr.is_valid_interface());
        mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
            expression_factory->create_expression_list());
        arguments->add_expression( "tint", arg_expr.get());

        // Create a material instance from the material definition 
        // "mdl::nvidia::sdk_examples::tutorials::example_material"
        // with the just prepared arguments.
        mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
            transaction->access<mi::neuraylib::IMaterial_definition>(
                "mdl::nvidia::sdk_examples::tutorials::example_material"));
        mi::Sint32 result;
        mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
            material_definition->create_material_instance( arguments.get(), &result));
        check_success( result == 0);
        transaction->store( material_instance.get(), "instance of example_material");
    }
    {
        // Dump the created material instance and function calls.
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
            transaction->access<mi::neuraylib::IMaterial_instance>(
                "instance of example_material"));
        std::cout << "Dumping material instance \"instance of example_material\":" << std::endl;
        dump_instance( expression_factory.get(), material_instance.get(), std::cout);

        mi::base::Handle<const mi::neuraylib::IFunction_call> function_call(
            transaction->access<mi::neuraylib::IFunction_call>(
                "call of texture_return.tint"));
        std::cout << "Dumping function call \"call of texture_return.tint\":" << std::endl;
        dump_instance( expression_factory.get(), function_call.get(), std::cout);

        function_call = transaction->access<mi::neuraylib::IFunction_call>(
            "call of file_texture");
        std::cout << "Dumping function call \"call of file_texture\":" << std::endl;
        dump_instance( expression_factory.get(), function_call.get(), std::cout);
    }

    transaction->commit();
}

int MAIN_UTF8( int argc, char* argv[])
{
    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get()))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    // Create a textured material
    create_textured_material( neuray.get());

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
