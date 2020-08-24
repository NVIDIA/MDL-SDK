/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/mdle/example_mdle.cpp
//
// Access the MDLE API and create MDLE files from existing mdl materials or functions.

#include <iostream>

// Include code shared by all examples.
#include "example_shared.h"

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

    // Access the database and create a transaction.
    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle < mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
            mdl_factory->create_type_factory(transaction.get()));

        mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
            mdl_factory->create_value_factory(transaction.get()));

        mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
            mdl_factory->create_expression_factory(transaction.get()));

        // Load the module "tutorials".
        // There is no need to configure any module search paths since
        // the mdl example folder is by default in the search path.
        check_success(mdl_impexp_api->load_module(
            transaction.get(), "::nvidia::sdk_examples::tutorials", context.get()) >= 0);
        print_messages(context.get());

        // get the MDLE api component
        mi::base::Handle<mi::neuraylib::IMdle_api> mdle_api(
            neuray->get_api_component<mi::neuraylib::IMdle_api>());

        // setup the export to mdle
        mi::base::Handle<mi::IStructure> data(transaction->create<mi::IStructure>("Mdle_data"));

        {
            // specify the material/function that will become the "main" of the MDLE
            mi::base::Handle<mi::IString> prototype(data->get_value<mi::IString>("prototype_name"));
            prototype->set_c_str("mdl::nvidia::sdk_examples::tutorials::example_mod_rough");
        }

        {
            // change a default values

            // create a new set of named parameters
            mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
                expression_factory->create_expression_list());

            // set a new tint value
            mi::base::Handle<mi::neuraylib::IValue_color> tint_value(
                value_factory->create_color(0.25f, 0.5f, 0.75f));

            mi::base::Handle<mi::neuraylib::IExpression_constant> tint_expr(
                expression_factory->create_constant(tint_value.get()));

            defaults->add_expression("tint", tint_expr.get());

            // set a new roughness value
            mi::base::Handle<mi::neuraylib::IValue_float> rough_value(
                value_factory->create_float(0.5f));

            mi::base::Handle<mi::neuraylib::IExpression_constant> rough_expr(
                expression_factory->create_constant(rough_value.get()));

            defaults->add_expression("roughness", rough_expr.get());

            // pass the defaults the Mdle_data struct
            data->set_value("defaults", defaults.get());
        }

        {
            // set thumbnail (files in the search paths or absolute file paths allowed as fall back)
            std::string thumbnail_path = mi::examples::mdl::get_examples_root()
                + "/mdl/nvidia/sdk_examples/resources/example_thumbnail.png";

            mi::base::Handle<mi::IString> thumbnail(data->get_value<mi::IString>("thumbnail_path"));
            thumbnail->set_c_str(thumbnail_path.c_str());
        }

        {
            // add additional files

            // each user file ...
            mi::base::Handle<mi::IStructure> user_file(
                transaction->create<mi::IStructure>("Mdle_user_file"));

            // ... is defined by a source path ...
            std::string readme_path = mi::examples::mdl::get_examples_root()
                + "/mdl/nvidia/sdk_examples/resources/example_readme.txt";

            mi::base::Handle<mi::IString> source_path(
                user_file->get_value<mi::IString>("source_path"));
            source_path->set_c_str(readme_path.c_str());

            // ... and a target path (inside the MDLE)
            mi::base::Handle<mi::IString> target_path(
                user_file->get_value<mi::IString>("target_path"));
            target_path->set_c_str("readme.txt");

            // all user files are passed as array
            mi::base::Handle<mi::IArray> user_file_array(
                transaction->create<mi::IArray>("Mdle_user_file[1]"));
            user_file_array->set_element(0, user_file.get());
            data->set_value("user_files", user_file_array.get());
        }

        // start the actual export
        const char* mdle_file_name = "example_material_blue.mdle";
        mdle_api->export_mdle(transaction.get(), mdle_file_name, data.get(), context.get());
        check_success(print_messages(context.get()));

        {
            // check and load an MDLE
            std::string mdle_path =
                mi::examples::io::get_working_directory() + "/" + mdle_file_name;

            // optional: check integrity of a (the created) MDLE file.
            mdle_api->validate_mdle(mdle_path.c_str(), context.get());
            check_success(print_messages(context.get()));

            // load the MDLE module
            mdl_impexp_api->load_module(transaction.get(), mdle_path.c_str(), context.get());
            check_success(print_messages(context.get()));

            // get database name of MDLE module
            mi::base::Handle<const mi::IString> mdle_db_name(
                mdl_factory->get_db_module_name(mdle_path.c_str()));
            std::cerr << "MDLE DB name: " << mdle_db_name->get_c_str() << std::endl;

            // the main material of an MDLE module is always called "main"
            std::string main_db_name(mdle_db_name->get_c_str());
            main_db_name += "::main";

            std::cerr << "MDLE main DB name: " << main_db_name << std::endl;

            // get the main material
            mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
                transaction->access<mi::neuraylib::IMaterial_definition>(main_db_name.c_str()));
            check_success(material_definition);

            // use the material ...
            std::cerr << "Successfully created and loaded " << mdle_file_name << std::endl
                      << std::endl;

            // access the user file
            mi::base::Handle<mi::neuraylib::IReader> reader(
                mdle_api->get_user_file(mdle_path.c_str(), "readme.txt", context.get()));
            check_success(print_messages(context.get()));

            // print the content to the console
            mi::Sint64 file_size = reader->get_file_size();
            char* content = new char[file_size + 1];
            content[file_size] = '\0';
            reader->read(content, file_size);
            std::cerr << "content of the readme.txt:" << std::endl << content << std::endl
                      << std::endl;
        }

        // ----------------------------------------------------------------------------------------

        // export a function to a second MDLE
        const char* mdle_file_name2 = "example_function.mdle";

        {
            // setup the export to MDLE
            mi::base::Handle<mi::IStructure> data(transaction->create<mi::IStructure>("Mdle_data"));

            // specify the material/function that will become the "main" of the MDLE
            mi::base::Handle<mi::IString> prototype(data->get_value<mi::IString>("prototype_name"));
            prototype->set_c_str(
                "mdl::nvidia::sdk_examples::tutorials::example_function(color,float)");

            {
                // set parameters, the 'example_function' has no defaults

                // create a new set of named parameters
                mi::base::Handle<mi::neuraylib::IExpression_list> defaults(
                    expression_factory->create_expression_list());

                // set a new tint value
                mi::base::Handle<mi::neuraylib::IValue_color> tint_value(
                    value_factory->create_color(1.0f, 0.66f, 0.33f));

                mi::base::Handle<mi::neuraylib::IExpression_constant> tint_expr(
                    expression_factory->create_constant(tint_value.get()));

                defaults->add_expression("tint", tint_expr.get());

                // set a new distance value
                mi::base::Handle<mi::neuraylib::IValue_float> distance_value(
                    value_factory->create_float(0.5f));

                mi::base::Handle<mi::neuraylib::IExpression_constant> distance_expr(
                    expression_factory->create_constant(distance_value.get()));

                defaults->add_expression("distance", distance_expr.get());

                // pass the defaults the Mdle_data struct
                data->set_value("defaults", defaults.get());
            }

            // start the actual export
            mdle_api->export_mdle(transaction.get(), mdle_file_name2, data.get(), context.get());
            check_success(print_messages(context.get()));
        }

        {
            // check and load the function again
            std::string mdle_path =
                mi::examples::io::get_working_directory() + "/" + mdle_file_name2;

            // optional: check integrity of a (the created) MDLE file.
            mdle_api->validate_mdle(mdle_path.c_str(), context.get());
            check_success(print_messages(context.get()));

            // load the MDLE module
            mdl_impexp_api->load_module(transaction.get(), mdle_path.c_str(), context.get());
            check_success(print_messages(context.get()));

            // get database name of MDLE module
            mi::base::Handle<const mi::IString> mdle_db_name(
                mdl_factory->get_db_module_name(mdle_path.c_str()));
            std::cerr << "MDLE DB name: " << mdle_db_name->get_c_str() << std::endl;

            // get database name of main function
            mi::base::Handle<const mi::neuraylib::IModule> mdle_module(
                transaction->access<mi::neuraylib::IModule>(mdle_db_name->get_c_str()));
            check_success(mdle_module.is_valid_interface());

            std::string main_db_name = mdle_db_name->get_c_str() + std::string("::main");
            mi::base::Handle<const mi::IArray> functions(
                mdle_module->get_function_overloads(main_db_name.c_str()));

            check_success(functions.is_valid_interface());
            check_success(functions->get_length() == 1);

            mi::base::Handle<const mi::IString> main_db_name_str(
                functions->get_element<const mi::IString>(0));

            main_db_name = main_db_name_str->get_c_str();
            std::cerr << "MDLE main DB name: " << main_db_name << std::endl;

            // get the main function
            mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
                transaction->access<mi::neuraylib::IFunction_definition>(main_db_name.c_str()));
            check_success(function_definition);

            // use the function ...
            std::cerr << "Successfully created and loaded " << mdle_file_name2 << std::endl
                      << std::endl;
        }

        {
            // since the same MDLE can be stored at various different places with different name
            // it could be valuable to check if the content of two MDLE is equal
            std::string mdle_path_a =
                mi::examples::io::get_working_directory() + "/" + mdle_file_name;
            std::string mdle_path_b =
                mi::examples::io::get_working_directory() + "/" + mdle_file_name2;

            // comparing an MDLE with itself should work
            mi::Sint32 res = mdle_api->compare_mdle(mdle_path_a.c_str(), mdle_path_a.c_str(), NULL);
            std::cerr << "Comparing " << mdle_file_name << " with " << mdle_file_name
                      << " resulted in: " << std::to_string(res) << std::endl;

            // this will fail
            res = mdle_api->compare_mdle(mdle_path_a.c_str(), mdle_path_b.c_str(), NULL);
            std::cerr << "Comparing " << mdle_file_name << " with " << mdle_file_name2
                << " resulted in: " << std::to_string(res) << std::endl;
        }


        // ----------------------------------------------------------------------------------------
        // All transactions need to get committed.
        transaction->commit();
    }

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
