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

// examples/mdl_sdk/discovery/example_discovery.cpp
//
// Discovers MDL files in file system and MDL archives and measures the traversal time

#include <chrono>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "example_shared.h"

using namespace mi::neuraylib;
using namespace std;

//-----------------------------------------------------------------------------
// Converts a kind to a string
//
std::string DK_to_string(IMdl_info::Kind kind)
{
    switch (kind)
    {
        case IMdl_info::DK_PACKAGE: return "DK_PACKAGE";
        case IMdl_info::DK_MODULE: return "DK_MODULE";
        case IMdl_info::DK_XLIFF: return "DK_XLIFF";
        case IMdl_info::DK_LIGHTPROFILE: return "DK_LIGHTPROFILE";
        case IMdl_info::DK_TEXTURE: return "DK_TEXTURE";
        case IMdl_info::DK_MEASURED_BSDF: return "DK_MEASURED_BSDF";
        case IMdl_info::DK_DIRECTORY: return "DK_DIRECTORY";
        default: return "UNKNOWN";
    }
}

//-----------------------------------------------------------------------------
// Converts a string to a kind
//
IMdl_info::Kind string_to_DK(std::string s)
{
    if (strcmp(s.c_str(), "DK_PACKAGE") == 0)
        return IMdl_info::DK_PACKAGE;
    else if (strcmp(s.c_str(), "DK_MODULE") == 0)
        return IMdl_info::DK_MODULE;
    else if (strcmp(s.c_str(), "DK_XLIFF") == 0)
        return IMdl_info::DK_XLIFF;
    else if (strcmp(s.c_str(), "DK_LIGHTPROFILE") == 0)
        return IMdl_info::DK_LIGHTPROFILE;
    else if (strcmp(s.c_str(), "DK_TEXTURE") == 0)
        return IMdl_info::DK_TEXTURE;
    else if (strcmp(s.c_str(), "DK_MEASURED_BSDF") == 0)
        return IMdl_info::DK_MEASURED_BSDF;
    if (strcmp(s.c_str(), "DK_DIRECTORY") == 0)
        return IMdl_info::DK_DIRECTORY;
    else if (strcmp(s.c_str(), "DK_ALL") == 0)
        return IMdl_info::DK_ALL;
    else {
        cerr << "\nWarning: Unexpected kind type " << s.c_str();
        cerr << ". Filtering will be disabled.\n";
        return IMdl_info::DK_ALL;
    }
}

void log_default_attributes(
    const char* shift,
    mi::Size search_path_index,
    const char* search_path,
    const char* resolved_path,
    bool in_archive)
{
    cout << "\n" << shift << "search path index: " << search_path_index;
    cout << "\n" << shift << "search path: " << search_path;
    cout << "\n" << shift << "resolved path: " << resolved_path;
    cout << "\n" << shift << "found in archive: " << in_archive;
    cout << "\n";
}
//-----------------------------------------------------------------------------
// Helper function for discovery logging
//
void log_api_package(const IMdl_info* info, int level)
{
    string shift("  ");
    for (int s = 0; s < level; s++)
        shift.append("  ");

    if (info == nullptr) {
        cerr << "\nError: Unexpected empty kind!\n";
    }
    else { 

        // Retrieve base properties
        cout << "\n" << shift.c_str() << "simple name: " << info->get_simple_name();
        cout << "\n" << shift.c_str() << "qualified name: " << info->get_qualified_name();

        mi::neuraylib::IMdl_info::Kind k = info->get_kind();
        cout << "\n" << shift.c_str() << "kind: " << DK_to_string(k);
        // Retrieve xliff properties
        if (info->get_kind() == IMdl_info::DK_XLIFF) {
            const mi::base::Handle<const IMdl_xliff_info> xliff_info(
                info->get_interface<const IMdl_xliff_info>());
            log_default_attributes(
                shift.c_str(),
                xliff_info->get_search_path_index(),
                xliff_info->get_search_path(),
                xliff_info->get_resolved_path(),
                xliff_info->in_archive());
            return;
        }

        // Retrieve texture properties
        if (info->get_kind() == IMdl_info::DK_TEXTURE) {
            const mi::base::Handle<const IMdl_texture_info> texture_info(
                info->get_interface<const IMdl_texture_info>());
            log_default_attributes(
                shift.c_str(),
                texture_info->get_search_path_index(),
                texture_info->get_search_path(),
                texture_info->get_resolved_path(),
                texture_info->in_archive());
            cout << "\n" << shift.c_str() << "number of shadows: "
                << texture_info->get_shadows_count();

            for (mi::Size s = 0; s < texture_info->get_shadows_count(); s++)
            {
                const mi::base::Handle<const IMdl_resource_info> sh(texture_info->get_shadow(s));
                cout << "\n" << shift.c_str() << "* in search path: " << sh->get_search_path();
            }
            cout << "\n";
            return;
        }

        // Retrieve lightprofile properties
        if (info->get_kind() == IMdl_info::DK_LIGHTPROFILE) {
            const mi::base::Handle<const IMdl_lightprofile_info> lightprofile_info(
                info->get_interface<const IMdl_lightprofile_info>());
            log_default_attributes(
                shift.c_str(),
                lightprofile_info->get_search_path_index(),
                lightprofile_info->get_search_path(),
                lightprofile_info->get_resolved_path(),
                lightprofile_info->in_archive());
            cout << "\n" << shift.c_str() << "number of shadows: "
                << lightprofile_info->get_shadows_count();

            for (mi::Size s = 0; s < lightprofile_info->get_shadows_count(); s++)
            {
                const mi::base::Handle<const IMdl_resource_info> sh(lightprofile_info->get_shadow(s));
                cout << "\n" << shift.c_str() << "* in search path: " << sh->get_search_path();
            }
            cout << "\n";
            return;
        }

        // Retrieve bsdf properties
        if (info->get_kind() == IMdl_info::DK_MEASURED_BSDF) {
            const mi::base::Handle<const IMdl_measured_bsdf_info> bsdf_info(
                info->get_interface<const IMdl_measured_bsdf_info>());
            log_default_attributes(
                shift.c_str(),
                bsdf_info->get_search_path_index(),
                bsdf_info->get_search_path(),
                bsdf_info->get_resolved_path(),
                bsdf_info->in_archive());
            cout << "\n" << shift.c_str() << "number of shadows: "
                << bsdf_info->get_shadows_count();

            for (mi::Size s = 0; s < bsdf_info->get_shadows_count(); s++)
            {
                const mi::base::Handle<const IMdl_resource_info> sh(bsdf_info->get_shadow(s));
                cout << "\n" << shift.c_str() << "* in search path: " << sh->get_search_path();
            }
            cout << "\n";
            return;
        }

        // Retrieve module properties
        if (info->get_kind() == IMdl_info::DK_MODULE)
        {
            const mi::base::Handle<const IMdl_module_info> module_info(
                info->get_interface<const IMdl_module_info>());
            log_default_attributes(
                shift.c_str(),
                module_info->get_search_path_index(),
                module_info->get_search_path(),
                module_info->get_resolved_path()->get_c_str(),
                module_info->in_archive());
            cout << "\n" << shift.c_str() << "number of shadows: "
                << module_info->get_shadows_count();

            for (mi::Size s = 0; s < module_info->get_shadows_count(); s++)
            {
                const mi::base::Handle<const IMdl_module_info> sh(module_info->get_shadow(s));
                cout << "\n" << shift.c_str() << "* in search path: " << sh->get_search_path();
            }
            cout << "\n";
            return;
        }

        // Retrieve package or directory properties
        if ((info->get_kind() == IMdl_info::DK_PACKAGE)||
            (info->get_kind() == IMdl_info::DK_DIRECTORY))
        {
            const mi::base::Handle<const IMdl_package_info> package_info(
                info->get_interface<const IMdl_package_info>());

            const mi::Size spi_count = package_info->get_search_path_index_count();
            if (spi_count > 0)
            {
                cout << "\n" << shift.c_str() << "discovered in " << spi_count << " search paths:";
                for (mi::Size i = 0; i < spi_count; ++i)
                {
                    log_default_attributes(
                        shift.c_str(),
                        package_info->get_search_path_index(i),
                        package_info->get_search_path(i),
                        package_info->get_resolved_path(i)->get_c_str(),
                        package_info->in_archive(i));
                }
            }

            // Recursively iterate over all sub-packages and modules
            const mi::Size child_count = package_info->get_child_count();
            cout << "\n" << shift.c_str() << "number of children: " << child_count;
            if (child_count > 0) cout << "\n";
            for (mi::Size i = 0; i < child_count; i++)
            {
                const mi::base::Handle<const IMdl_info> child(package_info->get_child(i));
                log_api_package(child.get(), level + 1);
            }
            if (child_count == 0) cout << "\n";
            return;
        }
    }
    cerr << "\n Unhandled IMdl_info::Kind found!\n";
}

// Prints the program usage
static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1> ...]\n"
        << "--help, -h            print this text\n"
        << "--filter, -f <kind>   discovery filter, can occur multiple times\n"
        << "                      Valid values are: DK_PACKAGE DK_MODULE DK_XLIFF DK_TEXTURE\n"
        << "                      DK_LIGHTPROFILE DK_MEASURED_BSDF DK_ALL(default)\n"
        << "--mdl_path, -m <path> mdl search path, can occur multiple times\n";
    exit(EXIT_FAILURE);
}

//-----------------------------------------------------------------------------
//
//
int MAIN_UTF8(int argc, char* argv[])
{
    mi::examples::mdl::Configure_options options;
    vector<string>  kind_filter;

    if (argc > 1) {
        // Collect command line arguments, if any argument is set
        for (int i = 1; i < argc; ++i) {
            const char *opt = argv[i];
            if ((strcmp(opt, "--filter") == 0) || (strcmp(opt, "-f") == 0)) {
                if (i < argc - 1)
                    kind_filter.push_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else if ((strcmp(opt, "--help") == 0) || (strcmp(opt, "-h") == 0)) {
                usage(argv[0]);
            }
            else if ((strcmp(opt, "--mdl_path") == 0) || (strcmp(opt, "-m") == 0)) {
                if (i < argc - 1)
                    options.additional_mdl_paths.push_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else {
                std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        }

        // at least one search path is required for the example to work
        if (options.additional_mdl_paths.size() == 0) {
            usage(argv[0]);
        }

        // disable the example search path
        options.add_example_search_path = false;
    }

    // disable admin and user space search paths to limit the output to only
    // the user input or the example content if the example was started without
    // parameters. Note, that real applications should always consider the
    // the admin and user space search paths.
    options.add_admin_space_search_paths = false;
    options.add_user_space_search_paths = false;

    // Configure filtering by IMdl_info::Kind
    mi::Uint32 discover_filter = IMdl_info::DK_ALL;
    if (kind_filter.size() > 0) {
        discover_filter = IMdl_info::DK_PACKAGE;
        for (auto& filter : kind_filter)
            discover_filter |= string_to_DK(filter);
    }

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        vector<string> mdl_files;

        chrono::time_point<chrono::system_clock> start, end;

        // Load discovery API
        mi::base::Handle<mi::neuraylib::IMdl_discovery_api> discovery_api(
            neuray->get_api_component<mi::neuraylib::IMdl_discovery_api>());

        // Create root package
        start = chrono::system_clock::now();

        // Get complete graph
        mi::base::Handle<const mi::neuraylib::IMdl_discovery_result>
            disc_result(discovery_api->discover(discover_filter));

        end = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = end - start;

        if (disc_result != nullptr)
        {
            const mi::base::Handle<const mi::neuraylib::IMdl_package_info> root(
                disc_result->get_graph());

            cout << "\nsearch path";
            mi::Size num = disc_result->get_search_paths_count();
            // Exclude '.'
            if (num > 1)
                cout << "s: \n";
            else
                cout << ": \n";

            for (mi::Size i = 0; i < num; i++)
                cout << disc_result->get_search_path(i) << "\n";

            cout << "\n -------------------- MDL graph --------------------\n";
            log_api_package(root.get(), 0);
            cout << "\n ------------------ \\ MDL graph --------------------\n";

            // Print traverse benchmark result
            stringstream m;
            m << "\nTraversed search path(s) ";
            for (size_t p = 0; p < options.additional_mdl_paths.size(); ++p)
                m << options.additional_mdl_paths[p] << " ";
            m << "in " << elapsed_seconds.count() << " seconds \n\n";
            cerr << m.str();
        }
        else
            cerr << "Failed to create collapsing graph out of search path "
                 << options.additional_mdl_paths[0] << "\n";

        discovery_api = 0;
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
