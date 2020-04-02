/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/execution_cuda/example_execution_cuda.cpp
//
// Introduces execution of the generated code for compiled material sub-expressions
// for the PTX backend with CUDA.

#include <iostream>
#include <vector>

// Enable this to dump the generated PTX code to stdout.
// #define DUMP_PTX

#include "example_cuda_shared.h"


// Command line options structure.
struct Options {
    // An result output file name.
    std::string outputfile;

    // The pattern number representing the combination of materials to display.
    unsigned material_pattern;

    // The resolution of the display / image.
    unsigned res_x, res_y;

    // Whether class compilation should be used for the materials.
    bool use_class_compilation;

    // Disables pixel oversampling.
    bool no_aa;

    // Whether derivative support should be enabled.
    bool enable_derivatives;

    // List of materials to use.
    std::vector<std::string> material_names;

    // List of MDL module paths.
    std::vector<std::string> mdl_paths;

    // The constructor.
    Options()
        : outputfile()
        , material_pattern(0)
        , res_x(700)
        , res_y(520)
        , use_class_compilation(false)
        , no_aa(false)
        , enable_derivatives(false)
    {
    }
};


// Bake the material sub-expressions created with the PTX backend into an image with the given
// resolution and the given number of samples for super-sampling and export it.
void bake_expression_cuda_ptx(
    std::vector<std::unique_ptr<Ptx_code> > const &target_codes,
    Options                                       &options,
    mi::Uint32                                    num_samples)
{
    // Build the full CUDA kernel with all the generated code
    CUfunction  cuda_function;
    char const *ptx_name = options.enable_derivatives ?
        "example_execution_cuda_derivatives.ptx" : "example_execution_cuda.ptx";
    CUmodule    cuda_module = build_linked_kernel(
        target_codes,
        (get_executable_folder() + "/" + ptx_name).c_str(),
        "evaluate_mat_expr",
        &cuda_function);

    // Prepare the needed data of all target codes for the GPU
    Material_gpu_context material_gpu_context(options.enable_derivatives);
    for (size_t i = 0, num_target_codes = target_codes.size(); i < num_target_codes; ++i) {
        if (!material_gpu_context.prepare_target_code_data(target_codes[i].get()))
            return;
    }
    CUdeviceptr device_tc_data_list = material_gpu_context.get_device_target_code_data_list();
    CUdeviceptr device_arg_block_list =
        material_gpu_context.get_device_target_argument_block_list();

    // Allocate GPU output buffer
    CUdeviceptr device_outbuf;
    check_cuda_success(cuMemAlloc(&device_outbuf, options.res_x * options.res_y * sizeof(float3)));

    // Launch kernel for the whole image
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((options.res_x + 15) / 16, (options.res_y + 15) / 16);
    void *kernel_params[] = {
        &device_outbuf,
        &device_tc_data_list,
        &device_arg_block_list,
        &options.res_x,
        &options.res_y,
        &num_samples
    };

    check_cuda_success(cuLaunchKernel(
        cuda_function,
        num_blocks.x, num_blocks.y, num_blocks.z,
        threads_per_block.x, threads_per_block.y, threads_per_block.z,
        0, nullptr, kernel_params, nullptr));

    // Copy the result image data to the host and export it.
    float3 *data = static_cast<float3*>(malloc(options.res_x * options.res_y * sizeof(float3)));
    if (data != nullptr) {
        check_cuda_success(cuMemcpyDtoH(
            data, device_outbuf, options.res_x * options.res_y * sizeof(float3)));
        export_image_rgbf(options.outputfile.c_str(), options.res_x, options.res_y, data);
        free(data);
    }

    // Cleanup resources not handled by Material_gpu_context
    check_cuda_success(cuMemFree(device_outbuf));
    check_cuda_success(cuModuleUnload(cuda_module));
}

void usage(char const *prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options] [(<material_pattern | (<material_name1> ...)]\n"
        << "Options:\n"
        << "  --res <x> <y>       resolution (default: 700x520)\n"
        << "  --cc                use class compilation\n"
        << "  --noaa              disable pixel oversampling\n"
        << "  -d                  enable use of derivatives\n"
        << "  -o <outputfile>     image file to write result to\n"
        << "                      (default: example_cuda_<material_pattern>.png)\n"
        << "  --mdl_path <path>   mdl search path, can occur multiple times.\n"
        << "  <material_pattern>  a number from 1 to 2 ^ num_materials - 1 choosing which\n"
        << "                      material combination to use (default: 2 ^ num_materials - 1)\n"
        << "  <material_name*>    qualified name of materials to use. The example will try to\n"
        << "                      access the path \"surface.scattering.tint\"."
        << std::endl;
    keep_console_open();
    exit(EXIT_FAILURE);
}


//------------------------------------------------------------------------------
//
// Main function
//
//------------------------------------------------------------------------------

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line options
    Options options;
    options.mdl_paths.push_back(get_samples_mdl_root());

    for (int i = 1; i < argc; ++i) {
        char const *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "-o") == 0 && i < argc - 1) {
                options.outputfile = argv[++i];
            } else if (strcmp(opt, "--res") == 0 && i < argc - 2) {
                options.res_x = std::max(atoi(argv[++i]), 1);
                options.res_y = std::max(atoi(argv[++i]), 1);
            } else if (strcmp(opt, "--cc") == 0) {
                options.use_class_compilation = true;
            } else if (strcmp(opt, "--noaa") == 0) {
                options.no_aa = true;
            } else if (strcmp(opt, "-d") == 0) {
                options.enable_derivatives = true;
            } else if (strcmp(opt, "--mdl_path") == 0 && i < argc - 1) {
                options.mdl_paths.push_back(argv[++i]);
            } else {
                std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        } else if (opt[0] >= '0' && opt[0] <= '9') {
            options.material_pattern = unsigned(atoi(opt));
        } else
            options.material_names.push_back(std::string(opt));
    }

    // Use default materials, if non was provided via command line
    if (options.material_names.empty()) {
        options.material_names.push_back("::nvidia::sdk_examples::tutorials::example_execution1");
        options.material_names.push_back("::nvidia::sdk_examples::tutorials::example_execution2");
        options.material_names.push_back("::nvidia::sdk_examples::tutorials::example_execution3");
    }

    if (options.material_pattern == 0)
        options.material_pattern = (1 << options.material_names.size()) - 1;
    else if (options.material_pattern < 1 ||
            options.material_pattern > unsigned(1 << options.material_names.size()) - 1) {
        std::cerr << "Invalid material_pattern parameter." << std::endl;
        usage(argv[0]);
    }

    if (options.outputfile.empty())
        options.outputfile = "example_cuda_" + to_string(options.material_pattern) + ".png";

    // Access the MDL Core compiler
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    FreeImage_Initialise();

    {
        Material_ptx_compiler mc(
            mdl_compiler.get(),
            /*num_texture_results=*/ 0,
            options.enable_derivatives,
            /*df_handle_mode=*/ "none");

        for (std::size_t i = 0; i < options.mdl_paths.size(); ++i)
            mc.add_module_path(options.mdl_paths[i].c_str());

        bool success = true;

        // Generate code for material sub-expressions of different materials
        // according to the requested material pattern
        for (unsigned i = 0, n = unsigned(options.material_names.size()); i < n; ++i) {
            if ((options.material_pattern & (1 << i)) != 0) {
                if (!mc.add_material_subexpr(
                        options.material_names[i].c_str(),
                        "surface.scattering.tint",
                        ("tint_" + to_string(i)).c_str(),
                        options.use_class_compilation)) {
                    success = false;
                    if (!mc.has_errors()) {
                        std::cout << "Failed compiling \"surface.scattering.tint\" of material \""
                            << options.material_names[i].c_str() << "\"." << std::endl;
                    }
                    break;
                }
            }
        }

        if (!success) {
            // Print any compiler messages, if available
            mc.print_messages();
        } else {
            // Generate the CUDA PTX code for the link unit.
            std::vector<std::unique_ptr<Ptx_code> > ptx_codes;
            ptx_codes.push_back(std::unique_ptr<Ptx_code>(mc.generate_cuda_ptx()));

            // Bake the material sub-expressions into a canvas
            CUcontext cuda_context = init_cuda();
            bake_expression_cuda_ptx(
                ptx_codes,
                options,
                options.no_aa ? 1 : 8);
            uninit_cuda(cuda_context);
        }
    }

    FreeImage_DeInitialise();

    // Free MDL compiler before shutting down MDL Core
    mdl_compiler = 0;

    // Unload MDL Core
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
