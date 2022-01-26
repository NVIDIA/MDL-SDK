/******************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/calls/example_calls.cpp
//
// Uses function calls to create a textured material.

#include <iostream>
#include <string>
#include <vector>

#include "example_shared.h"

using Call_argument = mi::mdl::DAG_call::Call_argument;


// Creates a diffuse material with a texture supplied as tint parameter.
void create_textured_material(Material_compiler &mc, bool use_class_compilation)
{
    // Load the "example" and "base" modules found via the configured module search path.
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> example_dag(mc.compile_module("::nvidia::sdk_examples::tutorials"));
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> base_dag(mc.compile_module("::base"));

    // Create a material instance which we will use to create DAG nodes for the material arguments
    Material_instance mat_instance(
        mc.create_material_instance(example_dag.get(), "::nvidia::sdk_examples::tutorials::example_material"));
    check_success(mat_instance);

    // Get factories of the material instance
    mi::mdl::IValue_factory *vf = mat_instance->get_value_factory();
    mi::mdl::IType_factory  *tf = vf->get_type_factory();

    //
    // Build the DAG node for
    //   ::base::file_texture(texture_2d("/nvidia/sdk_examples/resources/example.png",
    //       ::tex::gamma_default)).tint
    //

    // Create the texture value for the ::base::file_texture() call
    mi::mdl::IValue_texture const *tex = vf->create_texture(
        tf->create_texture(mi::mdl::IType_texture::TS_2D),
        "/nvidia/sdk_examples/resources/example.png",
        mi::mdl::IValue_texture::gamma_default,
        /*selector=*/ "",
        /*tag=*/ 0,
        /*tag_version=*/ 0);

    // Create the ::base::file_texture() call.
    // We don't provide a full signature here, but just take the first with a matching name
    mi::mdl::DAG_node const *file_texture_call = mat_instance.create_call(
        base_dag.get(),
        "::base::file_texture(texture_2d,color,color,::base::mono_mode,"
        "::base::texture_coordinate_info,float2,float2,::tex::wrap_mode,::tex::wrap_mode,bool,"
        "float,int2,::tex::wrap_mode,float)",
        { Call_argument(mat_instance->create_constant(tex), "texture") });
    check_success(file_texture_call);

    // Build the ".tint" accessor call. The parameter name of struct accessors is always "s"
    mi::mdl::DAG_node const *tex_ret_tint_call = mat_instance.create_call(
        base_dag.get(),
        "::base::texture_return.tint(::base::texture_return)",
        { Call_argument(file_texture_call, "s")});
    check_success(tex_ret_tint_call);

    // Initialize the material instance with the DAG node we created as "tint" material argument.
    // Note: As we don't specify the "roughness" argument, the default value will be used
    mi::mdl::IGenerated_code_dag::Error_code err =
        mc.initialize_material_instance(
            mat_instance,
            { Call_argument(tex_ret_tint_call, "tint") },
            use_class_compilation);
    check_success(err == mi::mdl::IGenerated_code_dag::EC_NONE);

    // Dump the DAG of the material instance together with the material instance arguments.
    // Note, that they are usually different from the material arguments we specified above.
    // With instance-compilation there will be no arguments, with class-compilation the arguments
    // will basically consist of the constants used in the material arguments. In our case,
    // we would see a lot of the constants from the default parameters of the file_texture() call.
    std::cout << "Dumping material instance:\n" << std::endl;
    mc.get_printer()->print(mat_instance.get_material_instance().get());
}

void usage()
{
    std::cerr
        << "Usage: example_calls [options]\n"
        << "--cc            use class compilation\n"
        << "--mdl_path      mdl search path, can occur multiple times\n"
        << std::endl;

    keep_console_open();
    exit(EXIT_FAILURE);
}

int MAIN_UTF8(int argc, char *argv[])
{
    // Collect command line parameters
    std::vector<std::string> mdl_paths;
    mdl_paths.push_back(get_samples_mdl_root());

    bool use_class_compilation = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cc") == 0)
            use_class_compilation = true;
        else if (strcmp(argv[i], "--mdl_path") == 0) {
            if (i + 1 < argc)
                mdl_paths.push_back(argv[++i]);
            else
                usage();
        }
        else
            usage();
    }

    // Access the MDL Core compiler
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    {
        // Initialize the material compiler
        Material_compiler mc(mdl_compiler.get());
        for (auto path : mdl_paths)
            mc.add_module_path(path.c_str());

        // Create a textured material
        create_textured_material(mc, use_class_compilation);
    }

    // Free MDL compiler before shutting down MDL Core
    mdl_compiler = 0;

    // Unload MDL Core
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
