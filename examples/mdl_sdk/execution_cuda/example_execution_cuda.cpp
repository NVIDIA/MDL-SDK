/******************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/execution_cuda/example_execution_cuda.cpp
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
    // The CUDA device ID.
    int cuda_device;

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

    // Whether terninary operators on *df types are executed at runtime or folded at compile time.
    bool fold_ternary_on_df;

    // List of materials to use.
    std::vector<std::string> material_names;

    // The constructor.
    Options()
        : cuda_device(0)
        , outputfile()
        , material_pattern(0)
        , res_x(700)
        , res_y(520)
        , use_class_compilation(false)
        , no_aa(false)
        , enable_derivatives(false)
        , fold_ternary_on_df(false)
    {
    }
};

// Bake the material sub-expressions created with the PTX backend into a canvas with the given
// resolution and the given number of samples for super-sampling.
mi::neuraylib::ICanvas *bake_expression_cuda_ptx(
    mi::neuraylib::ITransaction       *transaction,
    mi::neuraylib::IImage_api         *image_api,
    std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > const &target_codes,
    std::vector<size_t> const         &arg_block_indices,
    Options                           &options,
    mi::Uint32                         num_samples)
{
    // Build the full CUDA kernel with all the generated code
    CUfunction  cuda_function;
    char const *ptx_name = options.enable_derivatives ?
        "example_execution_cuda_derivatives.ptx" : "example_execution_cuda.ptx";
    CUmodule    cuda_module = build_linked_kernel(
        target_codes,
        (mi::examples::io::get_executable_folder() + "/" + ptx_name).c_str(),
        "evaluate_mat_expr",
        &cuda_function);

    // Prepare the needed data of all target codes for the GPU
    Material_gpu_context material_gpu_context(options.enable_derivatives);
    for (size_t i = 0, num_target_codes = target_codes.size(); i < num_target_codes; ++i) {
        if (!material_gpu_context.prepare_target_code_data(
                transaction, image_api, target_codes[i].get(), arg_block_indices))
            return nullptr;
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

    // Create a canvas (with only one tile) and copy the result image to it
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas("Rgb_fp", options.res_x, options.res_y));
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
    float3 *data = static_cast<float3 *>(tile->get_data());
    check_cuda_success(cuMemcpyDtoH(
        data, device_outbuf, options.res_x * options.res_y * sizeof(float3)));

    // Cleanup resources not handled by Material_gpu_context
    check_cuda_success(cuMemFree(device_outbuf));
    check_cuda_success(cuModuleUnload(cuda_module));

    canvas->retain();
    return canvas.get();
}

void usage(char const *prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options] [(<material_pattern | (<material_name1> ...)]\n"
        << "Options:\n"
        << "  --device <id>        run on CUDA device <id> (default: 0)\n"
        << "  --res <x> <y>        resolution (default: 700x520)\n"
        << "  --cc                 use class compilation\n"
        << "  --noaa               disable pixel oversampling\n"
        << "  -d                   enable use of derivatives\n"
        << "  -o <outputfile>      image file to write result to\n"
        << "                       (default: example_cuda_<material_pattern>.png)\n"
        << "  --mdl_path <path>    mdl search path, can occur multiple times.\n"
        << "  --fold_ternary_on_df fold all ternary operators on *df types\n"
        << "  <material_pattern>   a number from 1 to 2 ^ num_materials - 1 choosing which\n"
        << "                       material combination to use (default: 2 ^ num_materials - 1)\n"
        << "  <material_name*>     qualified name of materials to use. The example will try to\n"
        << "                       access the path \"surface.scattering.tint\"."
        << std::endl;
    exit_failure();
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
    mi::examples::mdl::Configure_options configure_options;

    for (int i = 1; i < argc; ++i) {
        char const *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "-o") == 0 && i < argc - 1) {
                options.outputfile = argv[++i];
            } else if (strcmp(opt, "--device") == 0 && i < argc - 2) {
                options.cuda_device = atoi(argv[++i]);
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
                configure_options.additional_mdl_paths.push_back(argv[++i]);
            } else if (strcmp(opt, "--fold_ternary_on_df") == 0) {
                options.fold_ternary_on_df = true;
            } else {
                std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        } else if (opt[0] >= '0' && opt[0] <= '9') {
            options.material_pattern = unsigned(atoi(opt));
        } else
            options.material_names.push_back(std::string(opt));
    }

    // Use default materials, if none was provided via command line
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
        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        // Access needed API components
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
            neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

        {
            // Generate code for material sub-expressions of different materials
            // according to the requested material pattern
            std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > target_codes;

            Material_compiler mc(
                mdl_impexp_api.get(),
                mdl_backend_api.get(),
                mdl_factory.get(),
                transaction.get(),
                /*num_texture_results=*/ 0,
                options.enable_derivatives,
                options.fold_ternary_on_df,
                /*enable_axuiliary_output*/ false,
                /*df_handle_mode*/ "none");

            for (std::size_t i = 0, n = options.material_names.size(); i < n; ++i) {
                if ((options.material_pattern & (1 << i)) != 0) {
                    // split module and material name
                    std::string module_name, material_simple_name;
                    if (!mi::examples::mdl::parse_cmd_argument_material_name(
                        options.material_names[i], module_name, material_simple_name, true))
                            continue;

                    // add the sub expression
                    mc.add_material_subexpr(
                        module_name, material_simple_name,
                        "surface.scattering.tint", ("tint_" + to_string(i)).c_str(),
                        options.use_class_compilation);
                }
            }

            // Generate target code for link unit
            target_codes.push_back(mc.generate_cuda_ptx());

            // Acquire image API needed to prepare the textures and to create a canvas for baking
            mi::base::Handle<mi::neuraylib::IImage_api> image_api(
                neuray->get_api_component<mi::neuraylib::IImage_api>());

            // Bake the material sub-expressions into a canvas
            CUcontext cuda_context = init_cuda(options.cuda_device);
            mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                bake_expression_cuda_ptx(
                    transaction.get(),
                    image_api.get(),
                    target_codes,
                    mc.get_argument_block_indices(),
                    options,
                    options.no_aa ? 1 : 8));
            uninit_cuda(cuda_context);

            // Export the canvas to an image on disk
            if (canvas)
                mdl_impexp_api->export_canvas(options.outputfile.c_str(), canvas.get());
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

