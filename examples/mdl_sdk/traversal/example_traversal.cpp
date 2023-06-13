/******************************************************************************
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/traversal/example_traversal.cpp
//
// Instantiates the materials in a given module, compiles them and recovers mdl code from
// the compiled material. This shows how to traverse a compiled material.

#include <iostream>
#include <fstream>
#include <string>
#include <stack>
#include <set>

#include "example_shared.h"
#include "compiled_material_traverser_print.h"

void print_help();
bool consume_cmd_options(int argc, char *argv[]);

// several test cases:
static const char* MODULE_TO_TRAVERSE = "::nvidia::sdk_examples::tutorials";

static const bool WRITE_TO_FILE = true;

// command line settings
std::string g_qualified_module_name = MODULE_TO_TRAVERSE;
bool g_use_class_compilation = true;
bool g_keep_compiled_structure = false;
std::vector<std::string> g_mdl_paths;
bool g_nostdpath = false;


int MAIN_UTF8(int argc, char* argv[])
{
    // Read command line settings to global variables
    if (!consume_cmd_options(argc, argv))
        exit_failure();

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Apply the search path setup described on the command line
    mi::examples::mdl::Configure_options configure_options;
    configure_options.additional_mdl_paths = g_mdl_paths;
    if (g_nostdpath)
    {
        configure_options.add_admin_space_search_paths = false;
        configure_options.add_user_space_search_paths = false;
        configure_options.add_example_search_path = false;
    }


    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 result = neuray->start();
    if (result != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", result);

    // get transaction
    mi::base::Handle<mi::neuraylib::ITransaction> transaction;
    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        transaction = scope->create_transaction();
    }

    // MDL factory to produce default values if not available in the material definition
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    // load the selected module
    mi::base::Handle<const mi::neuraylib::IModule> mdl_module;
    {
        // API component for module loading
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        // Create execution context
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        // load the module
        if (mdl_impexp_api->load_module(
            transaction.get(), g_qualified_module_name.c_str(), context.get()) < 0)
        {
            std::cerr << "[EXAMPLE] error: Failed to load the module '"
                << g_qualified_module_name << "'." << std::endl
                << "                 Please make sure to specify a qualified name."
                << std::endl;

            print_messages(context.get());
            print_help();
            exit_failure();
        }

        // get the module from the db
        mi::base::Handle<const mi::IString> db_name(
            mdl_factory->get_db_module_name(g_qualified_module_name.c_str()));
        mdl_module = transaction->access<mi::neuraylib::IModule>(db_name->get_c_str());
        if (!mdl_module)
            exit_failure("Module '%s' not found.", db_name->get_c_str());
    }

    // create an example traverser that allows to print mdl code from compiled materials
    Compiled_material_traverser_print printer;

    {
        // setup a user defined context that is passed though while traversing

        // ATTENTION: set the last parameter true to inspect the actual
        // structure of the compiled material. However, this may result in
        // invalid mdl code, that can not be compiled.
        Compiled_material_traverser_print::Context printer_context(
            transaction.get(),  // used to resolve resources
            mdl_factory.get(),
            g_keep_compiled_structure); // show compiler output vs. print valid mdl

        // Iterate over all materials exported by the module.
        mi::Size max_count = mdl_module->get_material_count();
        for (mi::Size i = 0; i < max_count; ++i)
        {
            std::string material_name = mdl_module->get_material(i);
            std::cout << "\n[EXAMPLE] info: Started processing material: "
                << material_name << "\n";

            // Access the material definition
            mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
                transaction->access<mi::neuraylib::IFunction_definition>(
                    material_name.c_str()));

            // assuming the material has parameters without defaults
            mi::neuraylib::Definition_wrapper definition_wrapper(
                transaction.get(), material_name.c_str(), mdl_factory.get());

            mi::base::Handle<mi::neuraylib::IScene_element> material_instance_se(
                definition_wrapper.create_instance(nullptr, &result));

            if (result < 0)
            {
                std::cerr << "[EXAMPLE] error: Failed to create material instance of '"
                    << material_name << "'\n";
                continue;
            }

            const mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
                material_instance_se->get_interface<mi::neuraylib::IMaterial_instance>());

            // Compile the material instance
            const mi::Uint32 flags = g_use_class_compilation
                ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
                : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

            mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
                material_instance->create_compiled_material(
                    flags));


            if (!material_instance)
            {
                std::cerr << "[EXAMPLE] error: Failed to compile material instance of '"
                    << material_name << "'\n";
                continue;
            }

            // generate mdl from a compiled material
            // since not all information is available anymore, we need to pass them manually
            std::stringstream s;
            s << material_definition->get_mdl_simple_name() << "_" << i << "_printed";
            const std::string printed_material_name = s.str();

            // print mdl string
            const std::string mdl = printer.print_mdl(
                compiled_material.get(), // to compiled material to traverse
                printer_context,         // the context passed through while traversing
                mdl_module->get_mdl_name(), // the original module path (for include)
                printed_material_name);  // the name of the output material

            // optional: print directly referenced modules and resources
            /*
            std::cout << "\n";
            std::cout << "Reconstructed Mdl code for '" << material_name << "'\n";
            std::cout << "Modules directly imported by the module "
                        << "and used by the material:\n";
            std::set<std::string>::iterator it = printer_context.get_used_modules().begin();
            std::set<std::string>::iterator end = printer_context.get_used_modules().end();
            for (; it != end; ++it)
                std::cout << " " << it->c_str() << "\n";

            std::cout << "Resources directly imported by the module "
                        << "and used by the material:\n";
            it = printer_context.get_used_resources().begin();
            end = printer_context.get_used_resources().end();
            for (; it != end; ++it)
                std::cout << " " << it->c_str() << "\n";
            std::cout << "\n";
            */

            // write to file if enabled
            if (WRITE_TO_FILE)
            {
                // note the extra underscore: this is used to avoid conflicts while loading
                std::ofstream file_stream;
                file_stream.open((printed_material_name + "_.mdl").c_str());
                if (file_stream)
                {
                    file_stream << mdl;
                    file_stream.close();
                }
            }
            else
            {
                // print to console instead
                std::cout << "\n\n\n" << mdl << "\n\n\n";
            }

            // if the resulting printed file is known to be invalid,
            // we do not try to load it.
            if (!printer_context.get_is_valid_mdl())
                continue;

            // check if the result can be loaded again
            s.str("");
            s << mdl_module->get_mdl_name() << "_" << i << "_printed";
            const std::string printed_module_name = s.str();

            // Create execution context
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
                neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

            result = mdl_impexp_api->load_module_from_string(
                transaction.get(), printed_module_name.c_str(), mdl.c_str(), context.get());

            if (result < 0)
            {
                std::cerr << "[EXAMPLE] error: Failed to load generated module: '"
                    << printed_module_name << "'\n";
                print_messages(context.get());
                continue;
            }

            const mi::base::Handle<const mi::neuraylib::IModule> module_printed(
                transaction->access<mi::neuraylib::IModule>(
                ("mdl" + printed_module_name).c_str()
                    ));

            if (!module_printed)
            {
                std::cerr << "[EXAMPLE] error: Loaded generated module is invalid: '"
                    << printed_module_name << "'\n";
            }
        }
    }

    // free all handles to elements in the database before committing
    mdl_module = nullptr;
    transaction->commit();

    // free all other handles before shutting down neuray
    transaction = nullptr;
    mdl_factory = nullptr;

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

//-----------------------------------------------------------------------------
// Command line processing
//

void print_help()
{
    std::cerr << R"(
traversal [options] <qualified_material_name>

options:

  -h|--help                     Prints this usage message and exits.
  -p|--mdl_path <path>          Adds the given path to the MDL search path.
  -n|--nostdpath                Prevents adding the MDL system and user search
                                path(s) to the MDL search path.
  --class                       Use class compilation (Default).
  --instance                    Use instance mode instead of class compilation.
  --keep                        Keep the structure produced by the compiler
                                (output may not compile!).)";


std::cerr << std::endl;
}

bool consume_cmd_options(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string cmd(argv[i]);
        if (argv[i][0] == '-')
        {
            // compilation mode. "-class" or "-instance", while "-class" is default
            if (cmd == "--instance")
            {
                g_use_class_compilation = false;
                continue;
            }
            if (cmd == "--class")
            {
                g_use_class_compilation = true; // also default
                continue;
            }

            // keep the structure produced by the compiler (output may not compile!)
            if (cmd == "--keep")
            {
                g_keep_compiled_structure = true; // default is false
                continue;
            }

            // add mdl paths
            if (cmd == "-p" || cmd == "--mdl_path")
            {
                if (i == argc - 1)
                {
                    std::cerr << "[EXAMPLE] error: Argument for -p|--mdl_path missing." << std::endl;
                    return false;
                }
                g_mdl_paths.push_back(argv[++i]);
                continue;
            }


            if (cmd == "-n" || cmd == "--nostdpath")
            {
                g_nostdpath = true;
                continue;
            }
            if (cmd == "--help" || cmd == "-h")
            {
                print_help();
                return false;
            }
        }
        else
        {
            // not beginning with dash defined the module to load
            g_qualified_module_name = cmd;

            // very basic test - note that this is not sufficient
            if (g_qualified_module_name.substr(0, 2) != "::")
            {
                std::cerr << "[EXAMPLE] error: the specified module '" << cmd
                    << "' is not a qualified module name" << std::endl;
                print_help();
                return false;
            }
        }
    }

    // print infos
    std::cout << "[EXAMPLE] info: Module to process: " << g_qualified_module_name << "\n";
    std::cout << "[EXAMPLE] info: Use class compilation: "
        << (g_use_class_compilation ? "true" : "false") << "\n";
    std::cout << "[EXAMPLE] info: Keep compiled structure: "
        << (g_keep_compiled_structure ? "true" : "false") << "\n";
    std::cout << "\n";
    return true;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8

