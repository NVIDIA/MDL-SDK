/******************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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
            , hdr_rotate(0.0f)
            , background_color_enabled(false)
            , background_color{ 0.25f, 0.5f, 0.75f }
            , firefly_clamp(true)
            , tone_mapping_burn_out(1.0f)
            , exposure_compensation(0.0f)
            , material_overrides()
        {
            hdr_environment =
                mi::examples::io::get_executable_folder() +
                "/content/hdri/hdrihaven_teufelsberg_inner_2k.exr";
        }

        std::string initial_scene;
        bool point_light_enabled;
        DirectX::XMFLOAT3 point_light_position;
        DirectX::XMFLOAT3 point_light_intensity;

        std::string hdr_environment;
        float hdr_scale;
        float hdr_rotate;
        bool background_color_enabled;
        DirectX::XMFLOAT3 background_color;

        bool firefly_clamp;
        float tone_mapping_burn_out;
        float exposure_compensation;

        std::vector<Material_override> material_overrides;
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

        << "--no-color                Disable colored console output for warnings and errors\n"

        << "-g|--generated <path>     File path to write generated MDL (e.g. from MaterialX) to.\n"
        << "                          If the specified path is a folder, the filename will be \n"
        << "                          generated with names defined by the loader.\n"
        << "                          If empty, no files are written (default: <empty>).\n"

        << "--res <res_x> <res_y>     Resolution (default: " << defaults.window_width
                                      << "x" << defaults.window_height << ")\n"

        << "--gpu <num>               Select a specific (non-default GPU) in case there are.\n"
        << "                          multiple available. See log output for option.\n"

        << "--gpu_debug               Enable the D3D Debug Layer and DRED\n"

        << "--nogui                   Don't open interactive display\n"

        << "--hide_gui                GUI can be toggled by pressing SPACE.\n(default: "
                                      << (defaults.hide_gui ? "hidden" : "visible") << ")\n"

        << "--nocc                    Don't use class-compilation\n"
        << "--fold_all_bool_params    Fold all boolean material parameters in class-compilation\n"
        << "--noaux                   Don't generate code for albedo and normal buffers\n"
        << "--nothreads               Disable parallel loading, e.g. of textures.\n"

        << "--hdr <filename>          HDR environment map\n"
        << "                          (default: <scene_folder>/" <<
                                      defaults.hdr_environment << ")\n"

        << "--hdr_scale <factor>      Environment intensity scale factor\n"
        << "                          (default: " << defaults.hdr_scale << ")\n"

        << "--hdr_rotate <angle>      Environment rotation in degree\n"
        << "                          (default: " << defaults.hdr_rotate << ")\n"

        << "--background <r> <g> <b>  Constant background color to replace the environment only \n"
        << "                          if directly visible to the camera. (default: <empty>).\n"

        << "--camera <px> <py> <pz> <fx> <fy> <fz>  Overrides the camera pose defined in the\n"
        << "                                        scene as well as the computed one if the scene\n"
        << "                                        has no camera. Parameters specify position and\n"
        << "                                        focus point.\n"

        << "-p|--mdl_path <path>      MDL search path, can occur multiple times.\n"

        << "--mdl_next                Enable features from upcoming MDL version.\n"

        << "--max_path_length <num>   Maximum path length (up to one total internal reflection),\n"
        << "                          minimum of 2, default " << defaults.ray_depth << "\n"

        << "--max_sss_steps <num>     Maximum number of volume scattering steps in addition to \n"
        << "                          'max_path_length', default " << defaults.sss_depth << "\n"

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

        << "--cam_exposure <factor>   Exposure compensation of the camera (default: " <<
                                      defaults.exposure_compensation << ")\n"

        << "-l <x> <y> <z> <r> <g> <b>      Add an isotropic point light with given coordinates\n"
        << "                                and intensity (flux) (default: none)\n"

        << "--mat <qualified_name>    Override all materials using a qualified material name.\n"

        << "--mat_selective <selector> <qualified_name> Override material in the scene that is\n"
        << "                                            selected by the material name in gltf.\n"

        << "--z_axis_up               Flip coordinate axis while loading the scene to (x, -z, y).\n"

        << "--uv_flip                 Flip texture coordinates from (u, 1-v) to (u, v).\n"
        << "                          (u, 1-v) is default in order to work with tiled textures\n"

        << "--uv_scale <x> <y>        Scale texture coordinates. (default: (1, 1)).\n"

        << "--uv_offset <x> <y>       Offset the scaled texture coordinates. (default: (0, 0)).\n"

        << "--uv_saturate             Clamps the texture coordinates to (0, 1). (default: false).\n"
        
        << "--uv_repeat               Wraps the texture coordinates to (0, 1). (default: false).\n"

        << "--mpsu <value>            meters per scene unit. Defines the scale of the scene.\n"
           "                          (default: " << defaults.meters_per_scene_unit << ")\n"

        << "--lpe <value>             LPE expression used on startup. Currently only 'beauty',\n"
           "                          'albedo_diffuse', 'albedo_glossy', and 'normal', 'roughness',\n"
           "                          and 'aov' are valid options.\n"
           "                          Also combinations seperated by ',' are valid with '--nogui'.\n"
           "                          (default: " << defaults.lpe[0]<< ")\n"

        << "--no_console_window       There will be no console window in addition to the main\n"
           "                          window. stdout and stderr streams are also not redirected.\n"

        << "--log_file <path>|0       Target path of the log output or '0' to disable\n"
           "                          the log file. (default: <outputfile-basename>.log)\n"

        << "--enable_shader_cache     Enable shader caching to improve (second) loading times.\n"

        << "--error                   Set log level to 'error' (default is 'info').\n"
        << "--warning                 Set log level to 'warning' (default is 'info').\n"
        << "--verbose                 Set log level to 'verbose' (default is 'info').\n"

        << "--shader_opt Od|O0|O1|O2|O3     Set optimization level of the shader compiler (DXC or Slang).\n"
           "                                (default: O3)\n"

        << "--use_slang               Use the Slang shader compiler instead of DXC.\n"

        << "--distill <target>        Distill the material before running the code generation.\n"
        << "--distill_debug           Dumps the original and distilled material (default is 'false').\n"

        << "--slot_mode <mode>        Slot mode for BSDF handles, one of 'none', '1', '2', '4', '8'.\n"
           "                          (default: 'none')\n"

        << "--material_type <type>    Qualified name of the material type (default: <empty>).\n"
           "                          If a type is specified and no --aov option is set,\n"
           "                          it's fields will be made available as AOVs.\n"

        << "--avo <field>             Field name of the material type to render. Return types that are\n"
           "                          not supported for visualization will be filtered out. Combinations,\n"
           "                          separated by ',' are valid, too. The first is used for rendering while\n"
           "                          the following will be available on the UI. (default: <empty>)\n"
           "                          See also the --material_type option for interactions.\n"

        << "--allowed_scatter_mode <m>      Limits the allowed scatter mode to \"none\", \"reflect\", \n"
           "                                \"transmit\" or \"reflect_and_transmit\"\n"
           "                                (default: restriction disabled)\n"

        #if MDL_ENABLE_MATERIALX
        << "--mtlx_path <path>        Specify an additional absolute search path location\n"
           "                          (e.g. '/projects/MaterialX'). This path will be queried when\n"
           "                          locating standard data libraries, XInclude references, and\n"
           "                          referenced images. Can occur multiple times.\n"

        << "--mtlx_library <rel_path> Specify an additional relative path to a custom data\n"
           "                          library folder (e.g. 'libraries/custom'). MaterialX files\n"
           "                          at the root of this folder will be included in all content\n"
           "                          documents. Can occur multiple times.\n"

        << "--mtlx_to_mdl <version>   Specify the MDL version to generate (requires MaterialX 1.38.9).\n"
            "                         Supported values are \"1.6\", \"1.7\", ..., \"1.10\", and \"latest\".\n"
            "                         Later MDL language versions require later MaterialX SDKs, too.\n"
            "                         (default: \"latest\")\n"

        << "--materialxtest_mode      Setup image and tex-coord space to match the MaterialXTest setup.\n"
        #endif
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
                if (wcscmp(opt, L"--no_console_window") == 0)
                {
                    options.no_console_window = true;
                }
                else if (wcscmp(opt, L"--nocc") == 0)
                {
                    options.use_class_compilation = false;
                }
                else if (wcscmp(opt, L"--fold_all_bool_params") == 0)
                {
                    options.fold_all_bool_parameters = true;
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
                else if (wcscmp(opt, L"--gpu_debug") == 0)
                {
                    options.gpu_debug = true;
                }
                else if (wcscmp(opt, L"--gpu-debug") == 0)
                {
                    log_warning("Deprecated argument `--gpu-debug`. Please use `--gpu_debug` instead.");
                    options.gpu_debug = true;
                }
                else if ((wcscmp(opt, L"-o" ) == 0 || wcscmp(opt, L"--output") == 0) && i < argc - 1)
                {
                    options.output_file =
                        mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));

                    if (!mi::examples::strings::remove_quotes(options.output_file))
                    {
                        log_error("Unexpected quotes in: '" + options.output_file + "'.");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if ((wcscmp(opt, L"-g") == 0 || wcscmp(opt, L"--generated") == 0) && i < argc - 1)
                {
                    options.generated_mdl_path =
                        mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));

                    if (!mi::examples::strings::remove_quotes(options.generated_mdl_path))
                    {
                        log_error("Unexpected quotes in: '" + options.generated_mdl_path + "'.");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--log_file") == 0 && i < argc - 1)
                {
                    log_path = mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));
                    if (!mi::examples::strings::remove_quotes(log_path))
                    {
                        log_error("Unexpected quotes in: '" + log_path + "'.");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--no-color") == 0)
                {
                    mi::examples::mdl_d3d12::enable_color_output(false);
                }
                else if (wcscmp(opt, L"--lpe") == 0 && i < argc - 1)
                {
                    std::string argument = mi::examples::strings::wstr_to_str(argv[++i]);
                    auto expressions = mi::examples::strings::split(argument, ",");
                    options.lpe.clear();
                    for (std::string& expr : expressions)
                    {
                        // remove white spaces
                        expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char x) 
                            { return std::isspace(x); }), expr.end());

                        if (expr != "beauty" &&
                            expr != "albedo" &&
                            expr != "albedo_diffuse" &&
                            expr != "albedo_glossy" &&
                            expr != "normal" &&
                            expr != "roughness" &&
                            expr != "aov")
                        {
                            log_error("Invalid LPE option: '" + expr + "'.");
                            return_code = EXIT_FAILURE;
                            return false;
                        }
                        options.lpe.push_back(expr);
                    }
                }
                else if (wcscmp(opt, L"--hdr") == 0 && i < argc - 1)
                {
                    std::string environment =
                        mi::examples::io::normalize(mi::examples::strings::wstr_to_str(argv[++i]));

                    if (!mi::examples::strings::remove_quotes(environment))
                    {
                        log_error("Unexpected quotes in: '" + environment + "'.");
                        return_code = EXIT_FAILURE;
                        return false;
                    }

                    options.hdr_environment = environment;
                }
                else if (wcscmp(opt, L"--hdr_scale") == 0 && i < argc - 1)
                {
                    options.hdr_scale = std::max(0.0f, static_cast<float>(_wtof(argv[++i])));
                }
                else if (wcscmp(opt, L"--hdr_rotate") == 0 && i < argc - 1)
                {
                    options.hdr_rotate =
                        std::max(0.0f, std::min(static_cast<float>(_wtof(argv[++i])), 360.0f)) / 360.0f;
                }
                else if (wcscmp(opt, L"--background") == 0 && i < argc - 3)
                {
                    options.background_color_enabled = true;
                    options.background_color = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                }
                else if (wcscmp(opt, L"--camera") == 0 && i < argc - 6)
                {
                    options.camera_pose_override = true;
                    options.camera_position = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                    options.camera_focus = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                }
                else if (wcscmp(opt, L"--fov") == 0 && i < argc - 1)
                {
                    options.camera_fov = 
                        std::max(0.0f, 
                            std::min(static_cast<float>(_wtof(argv[++i])), 90.0f)) * float(M_PI) / 180.0f;
                }
                else if (wcscmp(opt, L"--mat_selective") == 0 && i < argc - 2)
                {
                    Base_options::Material_override over;
                    over.selector = mi::examples::strings::wstr_to_str(argv[++i]);
                    over.material = mi::examples::strings::wstr_to_str(argv[++i]);
                    options.material_overrides.push_back(over);
                }
                else if (wcscmp(opt, L"--mat") == 0 && i < argc - 1)
                {
                    Base_options::Material_override over;
                    over.selector = "";
                    over.material = mi::examples::strings::wstr_to_str(argv[++i]);
                    options.material_overrides.push_back(over);
                }
                else if (wcscmp(opt, L"--no_firefly_clamp") == 0)
                {
                    options.firefly_clamp = false;
                }
                else if (wcscmp(opt, L"--burn_out") == 0 && i < argc - 1)
                {
                    options.tone_mapping_burn_out = static_cast<float>(_wtof(argv[++i]));
                }
                else if (wcscmp(opt, L"--cam_exposure") == 0 && i < argc - 1)
                {
                    options.exposure_compensation = static_cast<float>(_wtof(argv[++i]));
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
                    options.ray_depth = std::max(2, _wtoi(argv[++i]));
                }
                else if (wcscmp(opt, L"--max_sss_steps") == 0 && i < argc - 1)
                {
                    options.sss_depth = std::max(0, _wtoi(argv[++i]));
                }
                else if ((wcscmp(opt, L"-p") == 0 || wcscmp(opt, L"--mdl_path") == 0) && i < argc - 1)
                {
                    std::string mdl_path = mi::examples::strings::wstr_to_str(argv[++i]);
                    if (!mi::examples::strings::remove_quotes(mdl_path))
                    {
                        log_error("Unexpected quotes in: '" + mdl_path + "'.");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                    if (!mi::examples::io::is_absolute_path(mdl_path))
                    {
                        // make MDL search paths absolute
                        mdl_path = mi::examples::io::get_working_directory() + "/" + mdl_path;
                    }
                    options.mdl_paths.push_back(mi::examples::io::normalize(mdl_path, true));
                }
                else if (wcscmp(opt, L"--mdl_next") == 0)
                {
                    options.mdl_next = true;
                }
                else if (wcscmp(opt, L"--z_axis_up") == 0)
                {
                    options.handle_z_axis_up = true;
                }
                else if (wcscmp(opt, L"--mpsu") == 0 && i < argc - 1)
                {
                    options.meters_per_scene_unit = static_cast<float>(_wtof(argv[++i]));
                }
                else if (wcscmp(opt, L"--uv_flip") == 0)
                {
                    options.uv_flip = false;
                }
                else if (wcscmp(opt, L"--uv_scale") == 0 && i < argc - 2)
                {
                    options.uv_scale = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                }
                else if (wcscmp(opt, L"--uv_offset") == 0 && i < argc - 2)
                {
                    options.uv_offset = {
                        static_cast<float>(_wtof(argv[++i])),
                        static_cast<float>(_wtof(argv[++i]))
                    };
                }
                else if (wcscmp(opt, L"--uv_saturate") == 0)
                {
                    options.uv_saturate = true;
                }
                else if (wcscmp(opt, L"--uv_repeat") == 0)
                {
                    options.uv_repeat = true;
                }
                else if (wcscmp(opt, L"--enable_shader_cache") == 0)
                {
                    options.enable_shader_cache = true;
                }
                else if (wcscmp(opt, L"--error") == 0)
                {
                    set_log_level(Log_level::Error);
                }
                else if (wcscmp(opt, L"--warning") == 0)
                {
                    set_log_level(Log_level::Warning);
                }
                else if (wcscmp(opt, L"--verbose") == 0)
                {
                    set_log_level(Log_level::Verbose);
                }
                else if (wcscmp(opt, L"--shader_opt") == 0 && i < argc - 1)
                {
                    std::string opt = mi::examples::strings::wstr_to_str(argv[++i]);
                    if (opt != "Od" && opt != "O0" && opt != "O1" && opt != "O2" && opt != "O3")
                    {
                        log_error("Unexpected shader optimization level: '" + opt + "'.");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                    options.shader_opt = opt;
                }
                else if (wcscmp(opt, L"--distill") == 0 && i < argc - 1)
                {
                    options.distill_target = mi::examples::strings::wstr_to_str(argv[++i]);
                }
                else if (wcscmp(opt, L"--distill_debug") == 0)
                {
                    options.distill_debug = true;
                }
                else if (wcscmp(opt, L"--use_slang") == 0)
                {
                    #if MDL_ENABLE_SLANG
                        options.use_slang = true;
                    #else
                        log_error("Application not built with slang support. '--use_slang' will be ignored.");
                    #endif
                }
                else if (wcscmp(opt, L"--slot_mode") == 0 && i < argc - 1)
                {
                    ++i;
                    if (wcscmp(argv[i], L"none") == 0) {
                        options.slot_mode = Base_options::SM_NONE;
                    } else if (wcscmp(argv[i], L"1") == 0) {
                        options.slot_mode = Base_options::SM_FIXED_1;
                    } else if (wcscmp(argv[i], L"2") == 0) {
                        options.slot_mode = Base_options::SM_FIXED_2;
                    } else if (wcscmp(argv[i], L"4") == 0) {
                        options.slot_mode = Base_options::SM_FIXED_4;
                    } else if (wcscmp(argv[i], L"8") == 0) {
                        options.slot_mode = Base_options::SM_FIXED_8;
                    } else {
                        log_error("Unsupported slot mode: " + mi::examples::strings::wstr_to_str(argv[i]) + ".");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--material_type") == 0 && i < argc - 1)
                {
                    options.material_type = mi::examples::strings::wstr_to_str(argv[++i]);
                    std::string simple_type_name;
                    if (!mi::examples::mdl::parse_cmd_argument_material_name(
                        options.material_type, options.material_type_module, simple_type_name, false))
                    {
                        log_error("Type specified by --material_type is not a fully qualified name: " +
                            options.material_type);
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }
                else if (wcscmp(opt, L"--aov") == 0 && i < argc - 1)
                {
                    options.aov_to_render = mi::examples::strings::wstr_to_str(argv[++i]);
                }
                else if (wcscmp(opt, L"--allowed_scatter_mode") == 0 && i < argc - 1)
				{
                    options.enable_bsdf_flags = true;
                    LPWSTR mode = argv[++i];
                    if (wcscmp(mode, L"none") == 0) {
                        options.allowed_scatter_mode = Base_options::DF_FLAGS_NONE;
                    } else if (wcscmp(mode, L"reflect") == 0) {
                        options.allowed_scatter_mode = Base_options::DF_FLAGS_ALLOW_REFLECT;
                    } else if (wcscmp(mode, L"transmit") == 0) {
                        options.allowed_scatter_mode = Base_options::DF_FLAGS_ALLOW_TRANSMIT;
                    } else if (wcscmp(mode, L"reflect_and_transmit") == 0) {
                        options.allowed_scatter_mode =
                            Base_options::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
                    } else {
                        log_error("Unknown allowed_scatter_mode: \"" +
                            mi::examples::strings::wstr_to_str(mode) + "\".");
                        return_code = EXIT_FAILURE;
                        return false;
                    }
                }

                #if MDL_ENABLE_MATERIALX
                    else if (wcscmp(opt, L"--mtlx_path") == 0 && i < argc - 1)
                    {
                        std::string path = mi::examples::strings::wstr_to_str(argv[++i]);
                        if (!mi::examples::strings::remove_quotes(path))
                        {
                            log_error("Unexpected quotes in: '" + path + "'.");
                            return_code = EXIT_FAILURE;
                            return false;
                        }
                        options.mtlx_paths.push_back(mi::examples::io::normalize(path));
                    }
                    else if (wcscmp(opt, L"--mtlx_library") == 0 && i < argc - 1)
                    {
                        std::string path = mi::examples::strings::wstr_to_str(argv[++i]);
                        if (!mi::examples::strings::remove_quotes(path))
                        {
                            log_error("Unexpected quotes in: '" + path + "'.");
                            return_code = EXIT_FAILURE;
                            return false;
                        }
                        options.mtlx_libraries.push_back(mi::examples::io::normalize(path));
                    }
                    else if (wcscmp(opt, L"--mtlx_to_mdl") == 0 && i < argc - 1)
                    {
                        std::string version = mi::examples::strings::wstr_to_str(argv[++i]);
                        if (version != "1.6" && version != "1.7" && version != "1.8" && version != "1.9" &&
                            version != "1.10" && version != "latest")
                        {
                            log_error("Unexpected MaterialX to MDL version number: '" + version + "'.");
                            return_code = EXIT_FAILURE;
                            return false;
                        }
                        options.mtlx_to_mdl = version;
                    }
                    else if (wcscmp(opt, L"--materialxtest_mode") == 0)
                    {
                        options.uv_flip = false;
                        options.uv_scale = { 0.5f, 1.0f };
                        options.uv_offset = { 0.0f, 0.0f };
                        options.uv_repeat = true;
                        options.enable_auxiliary = false;
                        options.use_class_compilation = false;
                        options.materialxtest_mode = true;
                    }
                #endif
                else
                {
                    log_error("Unknown option: \"" + mi::examples::strings::wstr_to_str(argv[i]) + "\"");
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
                    log_error("Unexpected quotes in: '" + options.initial_scene + "'.");
                    return_code = EXIT_FAILURE;
                    return false;
                }
            }
        }

        // options that can be set only if combination
        if (options.lpe.size() > 1 && !options.no_gui)
        {
            log_error("Multiple LPE expressions can only be set in combination with '--nogui'.");
            return_code = EXIT_FAILURE;
            return false;
        }

        // set log to output
        if (log_path.empty())
        {
            log_path = options.output_file;
            log_path = log_path.substr(0, log_path.find_last_of('.') + 1) + "log";
        }
        log_set_file_path(log_path == "0" ? nullptr : log_path.c_str());

        // print time and command line options used to run the example
        std::wstring cmdLine = L"";
        for (int i = 0; i < argc; ++i)
        {
            cmdLine += argv[i];
            cmdLine += L" ";
        }
        mi::examples::mdl_d3d12::log_info("Command line arguments passed: " +
            mi::examples::strings::wstr_to_str(cmdLine));
        mi::examples::mdl_d3d12::log_info("Time local: " +
            mi::examples::strings::current_date_time_local());
        mi::examples::mdl_d3d12::log_info("Time UTC:   " +
            mi::examples::strings::current_date_time_UTC());

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
