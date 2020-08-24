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

// examples/mdl_sdk/dxr/example_dxr_options.h

#ifndef MDL_D3D12_EXAMPLE_DXR_OPTIONS_H
#define MDL_D3D12_EXAMPLE_DXR_OPTIONS_H

#include <shellapi.h>
#include "mdl_d3d12/common.h"
#include "mdl_d3d12/base_application.h"

#include <example_shared.h>

namespace mi { namespace examples { namespace dxr
{
    class Example_dxr_options : public mi::examples::mdl_d3d12::Base_options
    {
    public:
        explicit Example_dxr_options()
            : Base_options()
            , initial_scene(
                mi::examples::io::get_executable_folder() +
                "/content/gltf/sphere/sphere.gltf")
            , point_light_enabled(false)
            , point_light_position{ 10.0f, 0.0f, 5.0f }
            , point_light_intensity{ 1.0f, 0.95f, 0.9f }
            , hdr_scale(1.0f)
            , firefly_clamp(true)
            , tone_mapping_burn_out(1.0f)
        {
            user_options["environment"] =
                mi::examples::io::get_executable_folder() +
                "/content/hdri/hdrihaven_teufelsberg_inner_2k.exr";

            user_options["override_material"] = "";
        }

        std::string initial_scene;
        bool point_light_enabled;
        DirectX::XMFLOAT3 point_light_position;
        DirectX::XMFLOAT3 point_light_intensity;

        float hdr_scale;
        bool firefly_clamp;
        float tone_mapping_burn_out;
    };

    inline void print_options()
    {
        Example_dxr_options defaults;
        std::stringstream ss;

        ss << "\n"
        << "usage: " << BINARY_NAME << " [options] [<path_to_gltf_scene>]\n"

        << "-h|--help                 Print this text and exit\n"

        << "-v|--version              Print the MDL SDK version string and exit\n"

        << "-o|--output <outputfile>  Image file to write result to (default: "
                                      << defaults.output_file << ")\n"

        << "--res <res_x> <res_y>     Resolution (default: " << defaults.window_width
                                      << "x" << defaults.window_height << ")\n"

        << "--gpu <num>               Select a specific (non-default GPU) in case there are.\n"
        << "                          multiple available. See log output for option.\n"

        << "--nogui                   Don't open interactive display\n"

        << "--hide_gui                GUI can be toggled by pressing SPACE.\n(default: "
                                      << (defaults.hide_gui ? "hidden" : "visible") << ")\n"

        << "--nocc                    Don't use class-compilation\n"
        << "--noaux                   Don't generate code for albedo and normal buffers\n"
        << "--nothreads               Disable parallel loading, e.g. of textures.\n"

        << "--hdr <filename>          HDR environment map\n"
        << "                          (default: <scene_folder>/" <<
                                      defaults.user_options["environment"] << ")\n"

        << "--hdr_scale <factor>      Environment intensity scale factor\n"
        << "                          (default: " << defaults.hdr_scale << ")\n"

        << "--mdl_path <path>         MDL search path, can occur multiple times.\n"

        << "--max_path_length <num>   Maximum path length (up to one total internal reflection),\n"
        << "                          clamped to 2..100, default " << defaults.ray_depth << "\n"

        << "--iterations <num>        Number of progressive iterations. In GUI-mode, this is the\n"
        << "                          iterations per frame. In NO-GUI-mode it is the total count.\n"

        << "--enable_derivs           Enable automatic derivatives (not used in the ray tracer).\n"

        << "--tex_res <num>           Size of the texture results buffer (in float4). (default: " <<
                                      defaults.texture_results_cache_size <<")\n"

        << "--no_firefly_clamp        Disables firefly clamping used to suppress white pixels\n"
        << "                          because of low probability paths at early iterations.\n"
        << "                          (default: " << (defaults.firefly_clamp ? "on":"off") << ")\n"

        << "--burn_out <factor>       Tone mapping parameter (default: " <<
                                      defaults.tone_mapping_burn_out << ")\n"

        << "-l <x> <y> <z> <r> <g> <b>      Add an isotropic point light with given coordinates\n"
        << "                                and intensity (flux) (default: none)\n"

        << "--mat <qualified_name>    override all materials using a qualified material name.\n"

        << "--z_axis_up               flip coordinate axis while loading the scene to (x, -z, y).\n"

        << "--upm <value>             units per meter. the inverse is applied while loading the\n"
           "                          the scene. (default: " << defaults.units_per_meter << ")\n"

        << "--lpe <value>             LPE expression used on startup. Currently only 'beauty',\n"
           "                          'albedo', and 'normal' are valid options.\n"
           "                          (default: " << defaults.lpe<< ")\n"

        << "--no_console_window       There will be no console window in addition to the main\n"
           "                          window. stdout and stderr streams are also not redirected.\n"

        << "--log_file <path>|0       Target path of the log output or '0' to disable\n"
            "                         the log file. (default: <outputfile-basename>.log)\n"

        << "--enable_shader_cache     Enable shader caching to improve (second) loading times.\n"
        ;

        mdl_d3d12::log_info(ss.str());
    }

    inline bool parse_options(
        Example_dxr_options& options,
        int argc,
        PWSTR* argv,
        int& return_code)
    {
        using namespace mdl_d3d12;
        std::string log_path = "";
        for (int i = 0; i < argc; ++i)
        {
            LPWSTR opt = argv[i];
            if (opt[0] == '-')
            {
                if (wcscmp(opt, L"--no_console_window") == 0)   // handled outside this parser
                    continue;

                if (wcscmp(opt, L"--nocc") == 0)
                {
                    options.use_class_compilation = false;
                }
                else if (wcscmp(opt, L"--noaux") == 0)
                {
                    options.enable_auxiliary = false;
                }
                else if (wcscmp(opt, L"--nogui") == 0)
                {
                    options.no_gui = true;

                    // use a reasonable number of iterations in no-gui mode by default
                    if (options.iterations == 1)
                        options.iterations = 1000;
                }
                else if (wcscmp(opt, L"--hide_gui") == 0)
                {
                    options.hide_gui = true;
                }
                else if (wcscmp(opt, L"--nothreads") == 0)
                {
                    options.force_single_threading = true;
                }
                else if (wcscmp(opt, L"--res") == 0 && i < argc - 2)
                {
                    options.window_width = std::max(_wtoi(argv[++i]), 64);
                    options.window_height = std::max(_wtoi(argv[++i]), 48);
                }
                else if (wcscmp(opt, L"--iterations") == 0 && i < argc - 1)
                {
                    options.iterations = std::max(_wtoi(argv[++i]), 1);
                }
                else if (wcscmp(opt, L"--gpu") == 0 && i < argc - 1)
                {
                    options.gpu = std::max(_wtoi(argv[++i]), -1);
                }
                else if ((wcscmp(opt, L"-o" ) == 0 || wcscmp(opt, L"--output") == 0) && i < argc - 1)
                {
                    options.output_file =
                        mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));

                    if (!mi::examples::strings::remove_quotes(options.output_file))
                    {
                        log_error("Unexpected quotes in: '" + options.output_file + "'.", SRC);
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--log_file") == 0 && i < argc - 1)
                {
                    log_path = mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));
                    if (!mi::examples::strings::remove_quotes(log_path))
                    {
                        log_error("Unexpected quotes in: '" + log_path + "'.", SRC);
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--lpe") == 0 && i < argc - 1)
                {
                    options.lpe = mi::examples::strings::wstr_to_str(argv[++i]);
                    if(options.lpe != "beauty" &&
                        options.lpe != "albedo" &&
                        options.lpe != "normal")
                    {
                        log_error("Invalid LPE option: '" + options.lpe + "'.", SRC);
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--hdr") == 0 && i < argc - 1)
                {
                    std::string environment =
                        mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));

                    if (!mi::examples::strings::remove_quotes(environment))
                    {
                        log_error("Unexpected quotes in: '" + environment + "'.", SRC);
                        return_code = EXIT_FAILURE;
                        return false;
                    }

                    options.user_options["environment"] = environment;
                }
                else if (wcscmp(opt, L"--hdr_scale") == 0 && i < argc - 1)
                {
                    options.hdr_scale = std::max(0.0f, static_cast<float>(_wtof(argv[++i])));
                }
                else if (wcscmp(opt, L"--mat") == 0 && i < argc - 1)
                {
                    options.user_options["override_material"] = mi::examples::strings::wstr_to_str(argv[++i]);
                }
                else if (wcscmp(opt, L"--no_firefly_clamp") == 0)
                {
                    options.firefly_clamp = false;
                }
                else if (wcscmp(opt, L"--burn_out") == 0 && i < argc - 1)
                {
                    options.tone_mapping_burn_out = static_cast<float>(_wtof(argv[++i]));
                }
                else if (wcscmp(opt, L"--enable_derivs") == 0)
                {
                    options.automatic_derivatives = true;
                }
                else if (wcscmp(opt, L"--tex_res") == 0 && i < argc - 1)
                {
                    options.texture_results_cache_size = std::max(_wtoi(argv[++i]), 1);
                }
                else if (wcscmp(opt, L"-h") == 0 || wcscmp(opt, L"--help") == 0)
                {
                    print_options();
                    return_code = EXIT_SUCCESS;
                    return false;
                }
                else if (wcscmp(opt, L"-v") == 0 || wcscmp(opt, L"--version") == 0)
                {
                    // load neuray
                    mi::base::Handle<mi::neuraylib::INeuray> neuray(
                        mi::examples::mdl::load_and_get_ineuray());

                    if (!neuray.is_valid_interface())
                        exit_failure("Error: The MDL SDK library failed to load and to provide "
                                        "the mi::neuraylib::INeuray interface.");

                    // print library version information.
                    mi::base::Handle<const mi::neuraylib::IVersion> version(
                        neuray->get_api_component<const mi::neuraylib::IVersion>());
                    fprintf(stdout, "%s\n", version->get_string());

                    // free the handles and unload the MDL SDK
                    version = nullptr;
                    neuray = nullptr;
                    if (!mi::examples::mdl::unload())
                        exit_failure("Failed to unload the SDK.");

                    return_code = EXIT_SUCCESS;
                    return false;
                }
                else if (wcscmp(opt, L"-l") == 0 && i < argc - 6)
                {
                    options.point_light_enabled = true;
                    options.point_light_position = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                    options.point_light_intensity = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                }
                else if (wcscmp(opt, L"--max_path_length") == 0 && i < argc - 1)
                {
                    options.ray_depth = std::max(2, std::min(_wtoi(argv[++i]), 100));
                }
                else if (wcscmp(opt, L"--mdl_path") == 0 && i < argc - 1)
                {
                    std::string mdl_path = mi::examples::strings::wstr_to_str(argv[++i]);
                    if (!mi::examples::strings::remove_quotes(mdl_path))
                    {
                        log_error("Unexpected quotes in: '" + mdl_path + "'.", SRC);
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                    options.mdl_paths.push_back(mi::examples::io::normalize(mdl_path));
                }
                else if (wcscmp(opt, L"--z_axis_up") == 0)
                {
                    options.handle_z_axis_up = true;
                }
                else if (wcscmp(opt, L"--upm") == 0 && i < argc - 1)
                {
                    options.units_per_meter = static_cast<float>(_wtof(argv[++i]));
                }
                else if (wcscmp(opt, L"--enable_shader_cache") == 0)
                {
                    options.enable_shader_cache = true;
                    mi::examples::io::mkdir(mi::examples::io::get_executable_folder() + "/shader_cache");
                }
                else
                {
                    log_error("Unknown option: \"" + mi::examples::strings::wstr_to_str(argv[i]) + "\"", SRC);
                    print_options();
                    return_code = EXIT_FAILURE;
                    return false;
                }
            }
            else
            {
                // default argument is the GLTF scene to load
                options.initial_scene =
                    mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[i]));

                if (!mi::examples::strings::remove_quotes(options.initial_scene))
                {
                    log_error("Unexpected quotes in: '" + options.initial_scene + "'.", SRC);
                    return_code = EXIT_FAILURE;
                    return false;
                }
            }
        }

        // set log to output
        if (log_path.empty())
        {
            log_path = options.output_file;
            log_path = log_path.substr(0, log_path.find_last_of('.') + 1) + "log";
        }
        log_set_file_path(log_path == "0" ? nullptr : log_path.c_str());

        std::string cwd = mi::examples::io::get_working_directory();
        log_info("Current working directory: " + cwd);


        if (!mi::examples::io::is_absolute_path(options.initial_scene))
            options.initial_scene = cwd + "/" + options.initial_scene;

        log_info("Scene directory: " + mi::examples::io::dirname(options.initial_scene));
        log_info("Scene: " + options.initial_scene);

        return_code = EXIT_SUCCESS;
        return true;
    }
}}}

#endif
