/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/axf_to_mdl/axf_to_mdl.cpp
//
// Converts a X-Rite AxF(Appearance Exchange Format) physical material representration to MDL format.

#include <iostream>
#include <string>
#include <vector>

#include "example_shared.h"

#include "axf_importer_clearcoat_brdf_utils.h"
#include "axf_importer_options.h"
#include "axf_importer_reader.h"
#include "axf_importer_state.h"

using namespace mi::examples::impaxf;
using namespace std;

// Print command line usage to console and terminate the application.
void usage(char const* prog_name, bool failure = false)
{
    std::cout
        << "Usage: " << prog_name << " [options] <input_AxF_filename> [<output_MDL_filename>]\n"
        << "Options:\n"
        << "  -h   | --help                           Print this text and exit\n"
        << "  -v   | --version                        Print the MDL SDK version string and exit\n"
        << "\n"
        << "  -ap  | --axf_prefix <prefix>            AxF module prefix\n"
        << "                                            (default: \"axf\")\n"
        << "\n"
        << "  -acs | --axf_color_space <color_space>  AxF color space. Options:\n"
        << "                                            \"XYZ\"             : CIE 1931 XYZ.\n"
        << "                                            \"sRGB,E\" (default): Linear sRGB,E.\n"
        << "                                            \"AdobeRGB,E\"      : Linear Adobe RGB,E.\n"
        << "                                            \"WideGamutRGB,E\"  : Linear Adobe wide gamut RGB,E.\n"
        << "                                            \"ProPhotoRGB,E\"   : Linear Prophoto RGB,E.\n"
        // not supported by MDL
        //<< "                                            \"P3,E\"            : Linear P3,E.\n"
        //<< "                                            \"Rec2020,E\"       : Linear Rec2020,E.\n"
        //<< "                                            \"ACEScg,E\"        : ACEScg,E.\n"
        << "\n"
        << "  -acr | --axf_color_repr <color_repr>    AxF color representation. Options:\n"
        << "                                            \"rgb\"             : RGB.\n"
        << "                                            \"spectral\"        : Spectral.\n"
        << "                                            \"all\" (default)   : All.\n"
        << std::endl;

    exit_failure();
}

int MAIN_UTF8( int argc, char* argv[])
{
bool print_version_and_exit = false;
std::string axf_filename;

Axf_importer_options importer_options;

    if (argc == 1)
        usage(argv[0]);

    for (int i = 1; i < argc; ++i)
    {
        char const* opt = argv[i];
        if (opt[0] == '-')
        {
            if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0)
            {
                usage(argv[0]);
            }
            else if (strcmp(opt, "-v") == 0 || strcmp(opt, "--version") == 0)
            {
                print_version_and_exit = true;
            }
            else if ((strcmp(opt, "-ap") == 0 || strcmp(opt, "--axf_prefix") == 0) && i < argc - 1)
            {
                importer_options.axf_module_prefix = std::string(argv[++i]);
            }
            else if ((strcmp(opt, "-acs") == 0 || strcmp(opt, "--axf_color_space") == 0) && i < argc - 1)
            {
                char const* cs = argv[++i];

                if (
                    strcmp(cs, "XYZ") == 0 ||
                    strcmp(cs, "sRGB,E") == 0 ||
                    strcmp(cs, "AdobeRGB,E") == 0 ||
                    strcmp(cs, "WideGamutRGB,E") == 0 ||
                    strcmp(cs, "ProPhotoRGB,E") == 0 
                   )
                {
                    importer_options.axf_color_space = std::string(cs);
                }
                else
                {
                    printf("Wrong color space: \"%s\".\n", cs);
                    usage(argv[0]);
                }
            }
            else if ((strcmp(opt, "-acr") == 0 || strcmp(opt, "--axf_color_repr") == 0) && i < argc - 1)
            {
                char const* cr = argv[++i];

                if (
                    strcmp(cr, "rgb") == 0 ||
                    strcmp(cr, "spectral") == 0 ||
                    strcmp(cr, "all") == 0
                   )
                {
                    importer_options.axf_color_representation = std::string(cr);
                }
                else
                {
                    printf("Wrong color representation: \"%s\".\n", cr);
                    usage(argv[0]);
                }
            }
            else
            {
                usage(argv[0]);
            }
        }
        else
        {
            axf_filename = std::string(argv[i]);
            if(i+1 < argc)
            {
                importer_options.mdl_output_filename = std::string(argv[i + 1]);
            }
            else
            {
                size_t dot_pos = axf_filename.rfind('.');
                if (dot_pos == std::string::npos)
                    importer_options.mdl_output_filename = axf_filename + ".mdl";
                else
                    importer_options.mdl_output_filename = axf_filename.substr(0, dot_pos) + ".mdl";
            }

            printf("AxF Filename: %s\n", axf_filename.c_str());
            printf("MDL Filename: %s\n", importer_options.mdl_output_filename.c_str());

            break;
        }
    }

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Handle the --version flag
    if (print_version_and_exit)
    {
        // print library version information.
        mi::base::Handle<const mi::neuraylib::IVersion> version(
            neuray->get_api_component<const mi::neuraylib::IVersion>());
        fprintf(stdout, "%s\n", version->get_string());

        // free the handles and unload the MDL SDK
        version = nullptr;
        neuray = nullptr;
        if (!mi::examples::mdl::unload())
            exit_failure("Failed to unload the SDK.");

        exit_success();
    }


    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get()))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

        // Create AxF importer state
        Axf_impexp_state* imp_exp_state = new Axf_impexp_state("");
        assert(imp_exp_state);

        // our internal state is holding the parsed options
        Axf_impexp_state* internal_state = static_cast<Axf_impexp_state*>(imp_exp_state);
        internal_state->parse_importer_options(importer_options);

        // now do the import
        {
            Axf_reader axf_reader(neuray.get(), transaction.get());
            axf_reader.read(axf_filename.c_str(), internal_state);
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
