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

// examples/mdl_sdk/dxr/options.h

#ifndef MDL_D3D12_OPTIONS_H
#define MDL_D3D12_OPTIONS_H

#include <shellapi.h>

#include "mdl_d3d12/common.h"
#include "mdl_d3d12/base_application.h"
#include <example_shared.h>
#include <codecvt>
namespace mdl_d3d12
{
    class Example_dxr_options : public Base_options
    {
    public:
        explicit Example_dxr_options()
            : Base_options()
            , scene(get_executable_folder() + "/content/gltf/sphere/sphere.gltf")
            , point_light_enabled(false)
            , point_light_position{0.0f, 0.0f, 0.0f}
            , point_light_intensity{0.0f, 0.0f, 0.0f}
            , hdr_scale(1.0f)
            , firefly_clamp(true)
        {
            user_options["environment"] = 
                get_executable_folder() + "/content/hdri/hdrihaven_teufelsberg_inner_2k.exr";

            user_options["override_material"] = "";
        }

        std::string scene;

        bool point_light_enabled;
        DirectX::XMFLOAT3 point_light_position;
        DirectX::XMFLOAT3 point_light_intensity;

        float hdr_scale;
        bool firefly_clamp;
    };

    inline void print_options()
    {
        Example_dxr_options defaults;
        std::stringstream ss;

        ss << "\n"
        << "usage: " << BINARY_NAME << " [options] [<path_to_gltf_scene>]\n"

        << "-h|--help                 Print this text\n"
        
        << "-o <outputfile>           Image file to write result to (default: "
                                      << defaults.output_file << ")\n"

        << "--res <res_x> <res_y>     Resolution (default: " << defaults.window_width 
                                      << "x" << defaults.window_height << ")\n"

        << "--gpu <num>               Select a specific (non-default GPU) in case there are.\n"
        << "                          multiple available. See log output for option.\n"

        << "--nogui                   Don't open interactive display\n"

        << "--gui_scale <factor>      GUI scaling factor (default: " << defaults.gui_scale << ")\n"

        << "--hide_gui                GUI is hidden by default, press SPACE to show it\n"

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

        << "--no_firefly_clamp        Disables firefly clamping used to suppress white pixels\n"
        << "                          because of low probability paths at early iterations.\n"

        << "-l <x> <y> <z> <r> <g> <b>      Add an isotropic point light with given coordinates\n"
        << "                                and intensity (flux) (default: none)\n"
            
        << "--mat <qualified_name>    override all materials using a qualified material name.\n"
        << "--z_axis_up               flip coordinate axis while loading the scene to (x, -z, y).\n"
        << "--upm <value>             units per meter. the inverse is applied while loading the\n"
           "                          the scene. (default: " << defaults.units_per_meter << ")\n"
        ;

        log_info(ss.str());
    }

    //--------------------------------------------------------------------------------------
    // Conversion from wchar_t to utf8
    //--------------------------------------------------------------------------------------
    inline std::string to_utf8(wchar_t const *str)
    {
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.to_bytes(str);
    }

    inline bool parse_options(
        Example_dxr_options& options,
        LPWSTR command_line_args,
        int& return_code)
    {
        if (command_line_args && *command_line_args)
        {
            int argc;
            LPWSTR *arg_list = CommandLineToArgvW(command_line_args, &argc);

            for (size_t i = 0; i < argc; ++i)
            {
                LPWSTR opt = arg_list[i];
                if (opt[0] == '-')
                {
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
                        options.force_single_theading = true;
                    }
                    else if (wcscmp(opt, L"--gui_scale") == 0 && i < argc - 1)
                    {
                        options.gui_scale = static_cast<float>(_wtof(arg_list[++i]));
                    }
                    else if (wcscmp(opt, L"--res") == 0 && i < argc - 2)
                    {
                        options.window_width = std::max(_wtoi(arg_list[++i]), 64);
                        options.window_height = std::max(_wtoi(arg_list[++i]), 48);
                    }
                    else if (wcscmp(opt, L"--iterations") == 0 && i < argc - 1)
                    {
                        options.iterations = std::max(_wtoi(arg_list[++i]), 1);
                    }
                    else if (wcscmp(opt, L"--gpu") == 0 && i < argc - 1)
                    {
                        options.gpu = std::max(_wtoi(arg_list[++i]), -1);
                    }
                    else if (wcscmp(opt, L"-o") == 0 && i < argc - 1)
                    {
                        options.output_file = to_utf8(arg_list[++i]);
                        std::replace(options.output_file.begin(), options.output_file.end(), 
                                     '\\', '/');

                        if (!str_remove_quotes(options.output_file))
                        {
                            log_error("Unexpected quotes in: '" + options.output_file + "'.", SRC);
                            return_code = EXIT_FAILURE;
                            return false;
                        }
                    }
                    else if (wcscmp(opt, L"--hdr") == 0 && i < argc - 1)
                    {
                        std::string environment = to_utf8(arg_list[++i]);
                        std::replace(environment.begin(), environment.end(), '\\', '/');

                        if (!str_remove_quotes(environment))
                        {
                            log_error("Unexpected quotes in: '" + environment + "'.", SRC);
                            return_code = EXIT_FAILURE;
                            return false;
                        }

                        options.user_options["environment"] = environment;
                    }
                    else if (wcscmp(opt, L"--hdr_scale") == 0 && i < argc - 1)
                    {
                        options.hdr_scale = static_cast<float>(_wtof(arg_list[++i]));
                    }
                    else if (wcscmp(opt, L"--mat") == 0 && i < argc - 1)
                    {
                        options.user_options["override_material"] = to_utf8(arg_list[++i]);
                    }
                    else if (wcscmp(opt, L"--no_firefly_clamp") == 0)
                    {
                        options.firefly_clamp = false;
                    }
                    else if (wcscmp(opt, L"--no_derivs") == 0)
                    {
                        options.automatic_derivatives = false;
                    }
                    else if (wcscmp(opt, L"-h") == 0 || wcscmp(opt, L"--help") == 0)
                    {
                        print_options();
                        return_code = EXIT_SUCCESS;
                        return false;
                    }
                    else if (wcscmp(opt, L"-l") == 0 && i < argc - 6)
                    {
                        options.point_light_enabled = true;
                        options.point_light_position = {
                            static_cast<float>(_wtof(arg_list[++i])),
                            static_cast<float>(_wtof(arg_list[++i])),
                            static_cast<float>(_wtof(arg_list[++i]))
                        };

                        options.point_light_intensity = {
                            static_cast<float>(_wtof(arg_list[++i])),
                            static_cast<float>(_wtof(arg_list[++i])),
                            static_cast<float>(_wtof(arg_list[++i]))
                        };
                    }
                    else if (wcscmp(opt, L"--max_path_length") == 0 && i < argc - 1)
                    {
                        options.ray_depth = std::max(2, std::min(_wtoi(arg_list[++i]), 100));
                    }
                    else if (wcscmp(opt, L"--mdl_path") == 0 && i < argc - 1)
                    {
                        std::string mdl_path = to_utf8(arg_list[++i]);
                        if (!str_remove_quotes(mdl_path))
                        {
                            log_error("Unexpected quotes in: '" + mdl_path + "'.", SRC);
                            return_code = EXIT_FAILURE;
                            return false;
                        }

                        options.mdl_paths.push_back(mdl_path);
                    }
                    else if (wcscmp(opt, L"--z_axis_up") == 0)
                    {
                        options.handle_z_axis_up = true;
                    }
                    else if (wcscmp(opt, L"--upm") == 0 && i < argc - 1)
                    {
                        options.units_per_meter = static_cast<float>(_wtof(arg_list[++i]));
                    }
                    else
                    {
                        log_error("Unknown option: \"" + to_utf8(arg_list[i]) + "\"", SRC);
                        print_options();
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else
                {
                    // default argument is the GLTF scene to load
                    options.scene = to_utf8(arg_list[i]);
                    std::replace(options.scene.begin(), options.scene.end(), '\\', '/');

                    if (!str_remove_quotes(options.scene))
                    {
                        log_error("Unexpected quotes in: '" + options.scene + "'.", SRC);
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
            }
            LocalFree(arg_list);
        }

        // set log to output
        std::string log_path = options.output_file;
        log_path = log_path.substr(0, log_path.find_last_of('.') + 1) + "log";
        log_set_file_path(log_path.c_str());

        std::string cwd = get_working_directory();
        log_info("Current working directory: " + cwd);

        size_t pos = options.scene.find_last_of('/');
        if (!is_absolute_path(options.scene))
        {
            std::string subfolder = options.scene.substr(0, pos);

            options.scene = cwd + "/" + options.scene;
            options.scene_directory = cwd;
            if (pos != std::string::npos)
                options.scene_directory += "/" + subfolder;
        }
        else
            options.scene_directory = options.scene.substr(0, pos);


        log_info("Scene directory: " + options.scene_directory);
        log_info("Scene: " + options.scene);

        // add scene folder to search paths
        options.mdl_paths.push_back(options.scene_directory);

        return_code = EXIT_SUCCESS;
        return true;
    }
}

#endif
