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

// examples/mdl_sdk/modules/example_modules.cpp
//
// Loads an MDL module and inspects it contents.

#include <iostream>
#include <string>

#include "example_shared.h"

const char* material_definition_name = "mdl::nvidia::sdk_examples::tutorials::example_material";
const char* function_definition_name =
    "mdl::nvidia::sdk_examples::tutorials::example_function(color,float)";

// Utility function to dump the parameters of a material or function definition.
template <class T>
void dump_definition(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const T* definition,
    mi::Size depth,
    std::ostream& s)
{
    mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
        mdl_factory->create_type_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
        mdl_factory->create_expression_factory( transaction));

    mi::Size count = definition->get_parameter_count();
    mi::base::Handle<const mi::neuraylib::IType_list> types( definition->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults( definition->get_defaults());

    for( mi::Size index = 0; index < count; index++) {

        mi::base::Handle<const mi::neuraylib::IType> type( types->get_type( index));
        mi::base::Handle<const mi::IString> type_text( type_factory->dump( type.get(), depth+1));
        std::string name = definition->get_parameter_name( index);
        s << "    parameter " << type_text->get_c_str() << " " << name;

        mi::base::Handle<const mi::neuraylib::IExpression> default_(
            defaults->get_expression( name.c_str()));
        if( default_.is_valid_interface()) {
            mi::base::Handle<const mi::IString> default_text(
                expression_factory->dump( default_.get(), 0, depth+1));
            s << ", default = " << default_text->get_c_str() << std::endl;
        } else {
            s << " (no default)" << std::endl;
        }

    }

    mi::Size temporary_count = definition->get_temporary_count();
    for( mi::Size i = 0; i < temporary_count; ++i) {
        mi::base::Handle<const mi::neuraylib::IExpression> temporary( definition->get_temporary( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            expression_factory->dump( temporary.get(), name.str().c_str(), 1));
        s << "    temporary " << result->get_c_str() << std::endl;
    }

    mi::base::Handle<const mi::neuraylib::IExpression> body( definition->get_body());
    mi::base::Handle<const mi::IString> result( expression_factory->dump( body.get(), 0, 1));
    if( result)
        s << "    body " << result->get_c_str() << std::endl;
    else
        s << "    body not available for this function" << std::endl;

    s << std::endl;
}

// Loads an MDL module and inspects it contents.
void load_module( mi::neuraylib::INeuray* neuray)
{
    // Access the database and create a transaction.
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

    {
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle < mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        // Load the module "tutorials".
        check_success(mdl_impexp_api->load_module(
            transaction.get(), "::nvidia::sdk_examples::tutorials", context.get()) >= 0);
        print_messages( context.get());

        // Access the module by its name. The name to be used here is the MDL name of the module
        // ("example") plus the "mdl::" prefix.
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>( "mdl::nvidia::sdk_examples::tutorials"));
        check_success( module.is_valid_interface());

        // Print the module name and the file name it was loaded from.
        std::cout << "Loaded file " << module->get_filename() << std::endl;
        std::cout << "Found module " << module->get_mdl_name() << std::endl;
        std::cout << std::endl;

        // Dump imported modules.
        mi::Size module_count = module->get_import_count();
        std::cout << "The module imports the following modules:" << std::endl;
        for( mi::Size i = 0; i < module_count; i++)
            std::cout << "    " <<  module->get_import( i) << std::endl;
        std::cout << std::endl;

        // Dump exported types.
        mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
            mdl_factory->create_type_factory( transaction.get()));
        mi::base::Handle<const mi::neuraylib::IType_list> types( module->get_types());
        std::cout << "The module contains the following types: " << std::endl;
        for( mi::Size i = 0; i < types->get_size(); ++i) {
            mi::base::Handle<const mi::neuraylib::IType> type( types->get_type( i));
            mi::base::Handle<const mi::IString> result( type_factory->dump( type.get(), 1));
            std::cout << "    " << result->get_c_str() << std::endl;
        }
        std::cout << std::endl;

        // Dump exported constants.
        mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
            mdl_factory->create_value_factory( transaction.get()));
        mi::base::Handle<const mi::neuraylib::IValue_list> constants( module->get_constants());
        std::cout << "The module contains the following constants: " << std::endl;
        for( mi::Size i = 0; i < constants->get_size(); ++i) {
            const char* name = constants->get_name( i);
            mi::base::Handle<const mi::neuraylib::IValue> constant( constants->get_value( i));
            mi::base::Handle<const mi::IString> result( value_factory->dump( constant.get(), 0, 1));
            std::cout << "    " << name << " = " << result->get_c_str() << std::endl;
        }
        std::cout << std::endl;

        // Dump function definitions of the module.
        mi::Size function_count = module->get_function_count();
        std::cout << "The module contains the following function definitions:" << std::endl;
        for( mi::Size i = 0; i < function_count; i++)
            std::cout << "    " <<  module->get_function( i) << std::endl;
        std::cout << std::endl;

        // Dump material definitions of the module.
        mi::Size material_count = module->get_material_count();
        std::cout << "The module contains the following material definitions:" << std::endl;
        for( mi::Size i = 0; i < material_count; i++)
            std::cout << "    " <<  module->get_material( i) << std::endl;
        std::cout << std::endl;

        // Dump a function definition from the module.
        std::cout << "Dumping function definition \"" << function_definition_name << "\":"
                  << std::endl;
        mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
            transaction->access<mi::neuraylib::IFunction_definition>( function_definition_name));
        dump_definition(
            transaction.get(), mdl_factory.get(), function_definition.get(), 1, std::cout);

        // Dump a material definition from the module.
        std::cout << "Dumping material definition \"" << material_definition_name << "\":"
                  << std::endl;
        mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
            transaction->access<mi::neuraylib::IMaterial_definition>( material_definition_name));
        dump_definition(
            transaction.get(), mdl_factory.get(), material_definition.get(), 1, std::cout);

        // Dump the resources referenced by this module
        std::cout << "Dumping resources of this module: \n";
        for( mi::Size r = 0, rn = module->get_resources_count(); r < rn; ++r)
        {
            const char* db_name = module->get_resource_name( r);
            const char* mdl_file_path = module->get_resource_mdl_file_path( r);

            if( db_name == nullptr)
            {
                // resource is either not used and therefore has not been loaded or
                // could not be found.
                std::cout << "    db_name:               none" << std::endl;
                std::cout << "    mdl_file_path:         " << mdl_file_path << std::endl 
                          << std::endl;
                continue;
            }
            std::cout << "    db_name:               " << db_name << std::endl;
            std::cout << "    mdl_file_path:         " << mdl_file_path << std::endl;

            const mi::base::Handle<const mi::neuraylib::IType_resource> type(
                module->get_resource_type( r));
            switch( type->get_kind())
            {
                case mi::neuraylib::IType::TK_TEXTURE:
                {
                    const mi::base::Handle<const mi::neuraylib::ITexture> texture(
                        transaction->access<mi::neuraylib::ITexture>( db_name));
                    if( texture)
                    {
                        const mi::base::Handle<const mi::neuraylib::IImage> image(
                            transaction->access<mi::neuraylib::IImage>( texture->get_image()));

                        for( mi::Size t = 0, tn = image->get_uvtile_length(); t < tn; ++t)
                        {
                            const char* system_file_path = image->get_filename(
                                static_cast<mi::Uint32>( t));
                            std::cout << "    resolved_file_path[" << t << "]: " 
                                      << system_file_path << std::endl;
                        }
                    }
                    break;
                }

                case mi::neuraylib::IType::TK_LIGHT_PROFILE:
                {
                    const mi::base::Handle<const mi::neuraylib::ILightprofile> light_profile(
                        transaction->access<mi::neuraylib::ILightprofile>( db_name));
                    if( light_profile)
                    {
                        const char* system_file_path = light_profile->get_filename();
                        std::cout << "    resolved_file_path:    " << system_file_path << std::endl;
                    }
                    break;
                }

                case mi::neuraylib::IType::TK_BSDF_MEASUREMENT:
                {
                    const mi::base::Handle<const mi::neuraylib::IBsdf_measurement> mbsdf(
                        transaction->access<mi::neuraylib::IBsdf_measurement>( db_name));
                    if( mbsdf)
                    {
                        const char* system_file_path = mbsdf->get_filename();
                        std::cout << "    resolved_file_path:    " << system_file_path << std::endl;
                    }
                    break;
                }

                default:
                    break;
            }
            std::cout << std::endl;
        }
    }

    // All transactions need to get committed.
    transaction->commit();
}

int MAIN_UTF8( int /*argc*/, char* /*argv*/[])
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

    // Load an MDL module and dump its contents
    load_module( neuray.get());

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
