/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

// examples/example_execution_cuda.cpp
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

    // The MDL search paths.
    std::vector<std::string> mdl_paths;

    // List of materials to use.
    std::vector<std::string> material_names;

    // The constructor.
    Options()
        : outputfile()
        , material_pattern(0)
        , res_x(700)
        , res_y(520)
        , use_class_compilation(false)
    {
    }
};


// Bake the material sub-expressions created with the PTX backend into an image with the given
// resolution and the given number of samples for super-sampling and export it.
void bake_expression_cuda_ptx(
    std::vector<std::unique_ptr<Ptx_code> > const &target_codes,
    mi::Uint32                                    width,
    mi::Uint32                                    height,
    mi::Uint32                                    num_samples,
    char const                                   *out_path)
{
    // Build the full CUDA kernel with all the generated code
    CUfunction  cuda_function;
    CUmodule    cuda_module = build_linked_kernel(
        target_codes,
        (get_executable_folder() + "example_execution_cuda.ptx").c_str(),
        "evaluate_mat_expr",
        &cuda_function);

    // Prepare the needed data of all target codes for the GPU
    Material_gpu_context material_gpu_context;
    for (size_t i = 0, num_target_codes = target_codes.size(); i < num_target_codes; ++i) {
        if (!material_gpu_context.prepare_target_code_data(target_codes[i].get()))
            return;
    }
    CUdeviceptr device_tc_data_list = material_gpu_context.get_device_target_code_data_list();
    CUdeviceptr device_arg_block_list =
        material_gpu_context.get_device_target_argument_block_list();

    // Allocate GPU output buffer
    CUdeviceptr device_outbuf;
    check_cuda_success(cuMemAlloc(&device_outbuf, width * height * sizeof(mi::Uint32)));

    // Launch kernel for the whole image
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + 15) / 16, (height + 15) / 16);
    void *kernel_params[] = {
        &device_outbuf,
        &device_tc_data_list,
        &device_arg_block_list,
        &width,
        &height,
        &num_samples
    };

    check_cuda_success(cuLaunchKernel(
        cuda_function,
        num_blocks.x, num_blocks.y, num_blocks.z,
        threads_per_block.x, threads_per_block.y, threads_per_block.z,
        0, nullptr, kernel_params, nullptr));

    // Copy the result image data to the host and export it.
    mi::Uint32 *data = static_cast<mi::Uint32 *>(malloc(width * height * sizeof(mi::Uint32)));
    if (data != nullptr) {
        check_cuda_success(cuMemcpyDtoH(data, device_outbuf, width * height * sizeof(mi::Uint32)));
        export_image_rgba(out_path, width, height, data);
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

int main(int argc, char* argv[])
{
    // Parse command line options
    Options options;
    options.mdl_paths.push_back(get_samples_mdl_root());

    for (int i = 1; i < argc; ++i) {
        char const *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "-o") == 0) {
                if (i < argc - 1)
                    options.outputfile = argv[++i];
                else
                    usage(argv[0]);
            } else if (strcmp(opt, "--res") == 0) {
                if (i < argc - 2) {
                    options.res_x = std::max(atoi(argv[++i]), 1);
                    options.res_y = std::max(atoi(argv[++i]), 1);
                } else
                    usage(argv[0]);
            } else if (strcmp(opt, "--cc") == 0) {
                options.use_class_compilation = true;
            } else if (strcmp(opt, "--mdl_path") == 0) {
                if (i < argc - 1)
                    options.mdl_paths.push_back(argv[++i]);
                else
                    usage(argv[0]);
            } else
                usage(argv[0]);
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
        Material_ptx_compiler mc(mdl_compiler.get(), 0);
        for (std::size_t i = 0; i < options.mdl_paths.size(); ++i)
            mc.add_module_path(options.mdl_paths[i].c_str());

        bool success = true;

        // Generate code for material sub-expressions of different materials
        // according to the requested material pattern
        for (unsigned i = 0, n = unsigned(options.material_names.size()); i < n; ++i) {
            if ((options.material_pattern & (1 << i)) != 0) {
                if (!mc.add_material_subexpr(
                        options.material_names[i].c_str(),
                        { "surface", "scattering", "tint" },
                        "tint",
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
                options.res_x,
                options.res_y,
                8,
                options.outputfile.c_str());
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
