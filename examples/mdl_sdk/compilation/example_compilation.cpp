/******************************************************************************
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

// examples/example_compilation.cpp
//
// Introduces compiled materials and highlights differences between different compilation modes.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <mi/mdl_sdk.h>

#include "example_shared.h"

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

// Creates an instance of "mdl::example::compilation_material".
void create_material_instance(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_compiler* mdl_compiler,
    const char* instance_name)
{
    // Load the "example" module.
    check_success( mdl_compiler->load_module( transaction, "::nvidia::sdk_examples::tutorials") >= 0);

    // Create a material instance from the material definition 
    // "mdl::nvidia::sdk_examples::tutorials::compilation_material" with the default arguments.
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        transaction->access<mi::neuraylib::IMaterial_definition>(
            "mdl::nvidia::sdk_examples::tutorials::example_compilation"));
    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        material_definition->create_material_instance( 0, &result));
    check_success( result == 0);
    transaction->store( material_instance.get(), instance_name);
}

// Compiles the given material instance in the given compilation modes, dumps the result, and stores
// it in the DB.
void compile_material_instance(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* mdl_factory,
    const char* instance_name,
    const char* compiled_material_name,
    bool class_compilation)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
       transaction->access<mi::neuraylib::IMaterial_instance>( instance_name));
    mi::Sint32 result = 0;
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material( flags, 1.0f, 380.0f, 780.0f, &result));
    check_success( result == 0);

    std::cout << "Dumping compiled material (" << (class_compilation?"class":"instance")
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
    mi::neuraylib::IMdl_compiler* mdl_compiler,
    const char* compiled_material_name,
    const char* path,
    const char* fname)
{
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        transaction->edit<mi::neuraylib::ICompiled_material>( compiled_material_name));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_llvm_ir(
        mdl_compiler->get_backend( mi::neuraylib::IMdl_compiler::MB_LLVM_IR));
    check_success( be_llvm_ir->set_option( "num_texture_spaces", "16") == 0);
    check_success( be_llvm_ir->set_option( "enable_simd", "on") == 0);

    mi::Sint32 result = -1;
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_llvm_ir(
        be_llvm_ir->translate_material_expression(
            transaction, compiled_material.get(), path, fname, &result));
    check_success( result == 0);
    check_success( code_llvm_ir);

    std::cout << "Dumping LLVM IR code for \"" << path << "\" of \"" << compiled_material_name
              << "\":" << std::endl << std::endl;
    std::cout << code_llvm_ir->get_code() << std::endl;
}

// Generates CUDA PTX target code for a subexpression of a given compiled material.
void generate_cuda_ptx(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_compiler* mdl_compiler,
    const char* compiled_material_name,
    const char* path,
    const char* fname)
{
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        transaction->edit<mi::neuraylib::ICompiled_material>( compiled_material_name));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_cuda_ptx(
        mdl_compiler->get_backend( mi::neuraylib::IMdl_compiler::MB_CUDA_PTX));
    check_success( be_cuda_ptx->set_option( "num_texture_spaces", "16") == 0);
    check_success( be_cuda_ptx->set_option( "sm_version", "50") == 0);

    mi::Sint32 result = -1;
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_cuda_ptx(
        be_cuda_ptx->translate_material_expression(
            transaction, compiled_material.get(), path, fname, &result));
    check_success( result == 0);
    check_success( code_cuda_ptx);

    std::cout << "Dumping CUDA PTX code for \"" << path << "\" of \"" << compiled_material_name
              << "\":" << std::endl << std::endl;
    std::cout << code_cuda_ptx->get_code() << std::endl;
}


int main( int /*argc*/, char* /*argv*/[])
{
    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    check_success( neuray.is_valid_interface());

    // Configure the MDL SDK
    configure(neuray.get());

    // Start the MDL SDK
    mi::Sint32 result = neuray->start();
    check_start_success( result);

    {
        mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
            neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

        {
            // Load the "example" module and create a material instance
            std::string instance_name = "instance of compilation_material";
            create_material_instance( transaction.get(), mdl_compiler.get(), instance_name.c_str());

            // Compile the material instance in instance compilation mode
            std::string instance_compilation_name
                = std::string( "instance compilation of ") + instance_name;
            compile_material_instance(
                transaction.get(), mdl_factory.get(), instance_name.c_str(),
                instance_compilation_name.c_str(), false);

            // Compile the material instance in class compilation mode
            std::string class_compilation_name
                = std::string( "class compilation of ") + instance_name;
            compile_material_instance(
                transaction.get(), mdl_factory.get(), instance_name.c_str(),
                class_compilation_name.c_str(), true);

            // Change some material argument and compile again in both modes. Note the differences
            // in instance compilation mode, whereas only the referenced parameter itself changes in
            // class compilation mode.
            change_arguments( transaction.get(), mdl_factory.get(), instance_name.c_str());
            compile_material_instance(
                transaction.get(), mdl_factory.get(), instance_name.c_str(),
                instance_compilation_name.c_str(), false);
            compile_material_instance(
                transaction.get(), mdl_factory.get(), instance_name.c_str(),
                class_compilation_name.c_str(), true);

            // Use the various backends to generate target code for some material expression.
            generate_llvm_ir(
                transaction.get(), mdl_compiler.get(), instance_compilation_name.c_str(),
                "backface.scattering.tint", "tint");
            generate_cuda_ptx(
                transaction.get(), mdl_compiler.get(), instance_compilation_name.c_str(),
                "backface.scattering.tint", "tint");
            generate_cuda_ptx(
                transaction.get(), mdl_compiler.get(), class_compilation_name.c_str(),
                "backface.scattering.tint", "tint");
        }

        transaction->commit();
    }

    // Shut down the MDL SDK
    check_success( neuray->shutdown() == 0);
    neuray = 0;

    // Unload the MDL SDK
    check_success( unload());

    keep_console_open();
    return EXIT_SUCCESS;
}

