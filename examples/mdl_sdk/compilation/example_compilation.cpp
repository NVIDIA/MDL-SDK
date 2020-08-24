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

// examples/mdl_sdk/compilation/example_compilation.cpp
//
// Introduces compiled materials and highlights differences between different compilation modes.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#include "example_shared.h"

// Command line options structure.
struct Options {
    // Materials to use.
    std::string material_name;

    // Expression path to compile.
    std::string expr_path;

    // If true, changes the arguments of the instantiated material.
    // Will be set to false if the material name or expression path is changed.
    bool change_arguments;

    // The constructor.
    Options()
        : material_name("::nvidia::sdk_examples::tutorials::example_compilation")
        , expr_path("backface.scattering.tint")
        , change_arguments(true)
    {
    }
};

// Utility function to dump the hash, arguments, temporaries, and fields of a compiled material.
void dump_compiled_material(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const mi::neuraylib::ICompiled_material* cm,
    std::ostream& s)
{
    mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
        mdl_factory->create_expression_factory( transaction));

    mi::base::Uuid hash = cm->get_hash();
    char buffer[36];
    snprintf( buffer, sizeof( buffer),
        "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
    s << "    hash overall = " << buffer << std::endl;

    for( mi::Uint32 i = 0; i <= mi::neuraylib::SLOT_GEOMETRY_NORMAL; ++i) {
        hash = cm->get_slot_hash( mi::neuraylib::Material_slot( i));
        snprintf( buffer, sizeof( buffer),
            "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
        s << "    hash slot " << std::setw( 2) << i << " = " << buffer << std::endl;
    }

    mi::Size parameter_count = cm->get_parameter_count();
    for( mi::Size i = 0; i < parameter_count; ++i) {
        mi::base::Handle<const mi::neuraylib::IValue> argument( cm->get_argument( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            value_factory->dump( argument.get(), name.str().c_str(), 1));
        s << "    argument " << result->get_c_str() << std::endl;
    }

    mi::Size temporary_count = cm->get_temporary_count();
    for( mi::Size i = 0; i < temporary_count; ++i) {
        mi::base::Handle<const mi::neuraylib::IExpression> temporary( cm->get_temporary( i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            expression_factory->dump( temporary.get(), name.str().c_str(), 1));
        s << "    temporary " << result->get_c_str() << std::endl;
    }

    mi::base::Handle<const mi::neuraylib::IExpression> body( cm->get_body());
    mi::base::Handle<const mi::IString> result( expression_factory->dump( body.get(), 0, 1));
    s << "    body " << result->get_c_str() << std::endl;

    s << std::endl;
}

// Creates an instance of the given material definition and stores it in the DB.
void create_material_instance(
    mi::neuraylib::IMdl_factory* factory,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_execution_context* context,
    const char* material_name,
    const char* instance_name)
{
    // split module and material name
    std::string module_name, material_simple_name;
    if (!mi::examples::mdl::parse_cmd_argument_material_name(
        material_name, module_name, material_simple_name, true))
        exit_failure();

    // Load the module.
    mdl_impexp_api->load_module(transaction, module_name.c_str(), context);
    if (!print_messages(context))
        exit_failure("Loading module '%s' failed.", module_name.c_str());

    // Get the database name for the module we loaded
    mi::base::Handle<const mi::IString> module_db_name(
        factory->get_db_module_name(module_name.c_str()));

    // attach the material name
    std::string material_db_name =
        std::string(module_db_name->get_c_str()) + "::" + material_simple_name;

    // Get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        transaction->access<mi::neuraylib::IMaterial_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Accessing definition '%s' failed.", material_db_name.c_str());

    // Create a material instance from the material definition with the default arguments.
    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        material_definition->create_material_instance(0, &result));
    if (result != 0)
        exit_failure("Instantiating '%s' failed.", material_db_name.c_str());

    transaction->store(material_instance.get(), instance_name);
}

// Compiles the given material instance in the given compilation modes, dumps the result, and stores
// it in the DB.
void compile_material_instance(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::IMdl_execution_context* context,
    const char* instance_name,
    const char* compiled_material_name,
    bool class_compilation)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
       transaction->access<mi::neuraylib::IMaterial_instance>( instance_name));

    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material( flags, context));
    check_success( print_messages( context));

    std::cout << "Dumping compiled material (" << ( class_compilation ? "class" : "instance")
              << " compilation) for \"" << instance_name << "\":" << std::endl << std::endl;
    dump_compiled_material( transaction, mdl_factory, compiled_material.get(), std::cout);
    std::cout << std::endl;
    transaction->store( compiled_material.get(), compiled_material_name);
}

// Changes the tint parameter of the given material instance to green.
void change_arguments(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const char* instance_name)
{
    mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
        mdl_factory->create_value_factory( transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
        mdl_factory->create_expression_factory( transaction));

    // Edit the instance of the material definition "compilation_material".
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        transaction->edit<mi::neuraylib::IMaterial_instance>( instance_name));
    check_success( material_instance.is_valid_interface());

    // Create the new argument for the "tint" parameter from scratch with the new value, and set it.
    mi::base::Handle<mi::neuraylib::IValue> tint_value(
        value_factory->create_color( 0.0f, 1.0f, 0.0f));
    mi::base::Handle<mi::neuraylib::IExpression> tint_expr(
        expression_factory->create_constant( tint_value.get()));
    check_success( material_instance->set_argument( "tint", tint_expr.get()) == 0);
}

// Generates LLVM IR target code for a subexpression of a given compiled material.
void generate_llvm_ir(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::IMdl_execution_context* context,
    const char* compiled_material_name,
    const char* path,
    const char* fname)
{
    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        transaction->access<mi::neuraylib::ICompiled_material>( compiled_material_name));
    check_success(compiled_material.is_valid_interface());

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_llvm_ir(
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_LLVM_IR));
    check_success(be_llvm_ir.is_valid_interface());

    check_success(be_llvm_ir->set_option( "num_texture_spaces", "16") == 0);
    check_success(be_llvm_ir->set_option( "enable_simd", "on") == 0);

    mi::base::Handle<const mi::neuraylib::ITarget_code> code_llvm_ir(
        be_llvm_ir->translate_material_expression(
            transaction, compiled_material.get(), path, fname, context));
    check_success(print_messages( context));
    check_success(code_llvm_ir);

    std::cout << "Dumping LLVM IR code for \"" << path << "\" of \"" << compiled_material_name
              << "\":" << std::endl << std::endl;
    std::cout << code_llvm_ir->get_code() << std::endl;
}

// Generates CUDA PTX target code for a subexpression of a given compiled material.
void generate_cuda_ptx(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::IMdl_execution_context* context,
    const char* compiled_material_name,
    const char* path,
    const char* fname)
{
    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        transaction->access<mi::neuraylib::ICompiled_material>( compiled_material_name));
    check_success(compiled_material.is_valid_interface());

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_cuda_ptx(
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
    check_success(be_cuda_ptx.is_valid_interface());

    check_success(be_cuda_ptx->set_option( "num_texture_spaces", "16") == 0);
    check_success(be_cuda_ptx->set_option( "sm_version", "50") == 0);

    mi::base::Handle<const mi::neuraylib::ITarget_code> code_cuda_ptx(
        be_cuda_ptx->translate_material_expression(
            transaction, compiled_material.get(), path, fname, context));
    check_success( print_messages( context));
    check_success( code_cuda_ptx);

    std::cout << "Dumping CUDA PTX code for \"" << path << "\" of \"" << compiled_material_name
              << "\":" << std::endl << std::endl;
    std::cout << code_cuda_ptx->get_code() << std::endl;
}

// Generates HLSL target code for a subexpression of a given compiled material.
void generate_hlsl(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::IMdl_execution_context* context,
    const char* compiled_material_name,
    const char* path,
    const char* fname)
{
    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        transaction->access<mi::neuraylib::ICompiled_material>( compiled_material_name));
    check_success(compiled_material.is_valid_interface());

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_hlsl(
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL));
    check_success(be_hlsl.is_valid_interface());

    check_success(be_hlsl->set_option( "num_texture_spaces", "1") == 0);

    mi::base::Handle<const mi::neuraylib::ITarget_code> code_hlsl(
        be_hlsl->translate_material_expression(
            transaction, compiled_material.get(), path, fname, context));
    check_success(print_messages( context));
    check_success(code_hlsl);

    std::cout << "Dumping HLSL code for \"" << path << "\" of \"" << compiled_material_name
              << "\":" << std::endl << std::endl;
    std::cout << code_hlsl->get_code() << std::endl;
}


void usage( char const *prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options] [<material_name>]\n"
        << "Options:\n"
        << "  --mdl_path <path>   mdl search path, can occur multiple times.\n"
        << "  --expr_path         expression path to compile, defaults to "
           "\"backface.scattering.tint\"."
        << "  <material_name>     qualified name of materials to use, defaults to\n"
        << "                      \"::nvidia::sdk_examples::tutorials::example_compilation\"\n"
        << std::endl;
    exit_failure();
}

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line options
    Options options;
    mi::examples::mdl::Configure_options configure_options;

    for (int i = 1; i < argc; ++i) {
        char const *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--mdl_path") == 0 && i < argc - 1) {
                configure_options.additional_mdl_paths.push_back(argv[++i]);
            }
            else if (strcmp(opt, "--expr_path") == 0 && i < argc - 1) {
                options.expr_path = argv[++i];
                options.change_arguments = false;
            }
            else {
                std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        }
        else {
            options.material_name = opt;
            options.change_arguments = false;
        }
    }

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        {
            // Create an execution context for options and error message handling
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            // Create MDL import-export API for importing MDL modules
            mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
                neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

            // Load the "example" module and create a material instance
            std::string instance_name = "instance of compilation_material";
            create_material_instance(
                mdl_factory.get(),
                transaction.get(),
                mdl_impexp_api.get(),
                context.get(),
                options.material_name.c_str(),
                instance_name.c_str());

            // Compile the material instance in instance compilation mode
            std::string instance_compilation_name
                = std::string("instance compilation of ") + instance_name;
            compile_material_instance(
                transaction.get(), mdl_factory.get(), context.get(), instance_name.c_str(),
                instance_compilation_name.c_str(), false);

            // Compile the material instance in class compilation mode
            std::string class_compilation_name
                = std::string("class compilation of ") + instance_name;
            compile_material_instance(
                transaction.get(), mdl_factory.get(), context.get(), instance_name.c_str(),
                class_compilation_name.c_str(), true);

            // Change some material argument and compile again in both modes. Note the differences
            // in instance compilation mode, whereas only the referenced parameter itself changes in
            // class compilation mode.
            if (options.change_arguments) {
                change_arguments(transaction.get(), mdl_factory.get(), instance_name.c_str());
                compile_material_instance(
                    transaction.get(), mdl_factory.get(), context.get(), instance_name.c_str(),
                    instance_compilation_name.c_str(), false);
                compile_material_instance(
                    transaction.get(), mdl_factory.get(), context.get(), instance_name.c_str(),
                    class_compilation_name.c_str(), true);
            }

            // Use the various backends to generate target code for some material expression.

            mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
                neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

            generate_llvm_ir(
                transaction.get(), mdl_backend_api.get(), context.get(),
                instance_compilation_name.c_str(),
                options.expr_path.c_str(), "tint");
            generate_llvm_ir(
                transaction.get(), mdl_backend_api.get(), context.get(),
                class_compilation_name.c_str(),
                options.expr_path.c_str(), "tint");
            generate_cuda_ptx(
                transaction.get(), mdl_backend_api.get(), context.get(),
                instance_compilation_name.c_str(),
                options.expr_path.c_str(), "tint");
            generate_cuda_ptx(
                transaction.get(), mdl_backend_api.get(), context.get(),
                class_compilation_name.c_str(),
                options.expr_path.c_str(), "tint");
            generate_hlsl(
                transaction.get(), mdl_backend_api.get(), context.get(),
                instance_compilation_name.c_str(),
                options.expr_path.c_str(), "tint");
            generate_hlsl(
                transaction.get(), mdl_backend_api.get(), context.get(),
                class_compilation_name.c_str(),
                options.expr_path.c_str(), "tint");
        }

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

