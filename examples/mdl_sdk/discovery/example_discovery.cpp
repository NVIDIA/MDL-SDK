/******************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

// examples/example_discovery.cpp
//
// Discovers MDL files in file system and MDL archives and measures the traversal time

#include <chrono>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <mi/mdl_sdk.h>
#include "example_shared.h"

using namespace mi::neuraylib;
using namespace std;

//-----------------------------------------------------------------------------
//  Helper function to add the root paths to the MDL SDK
//
void configure(mi::neuraylib::INeuray* neuray, const vector<string> &roots)
{
    // Add mdl search paths
    mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
        neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

    for (size_t p = 0; p < roots.size(); ++p)
    {
        mi::Sint32 res = mdl_compiler->add_module_path(roots[p].c_str());
        if ( res != 0)
            std::cerr << "Error: Issue with adding path " << roots[p] << "\n";
    }
}

//-----------------------------------------------------------------------------
// Helper function for discovery info kind logging
//
std::string DK_to_string(IMdl_info::Kind kind)
{
    switch (kind)
    {
    case IMdl_info::DK_PACKAGE: return "DK_PACKAGE";
    case IMdl_info::DK_MODULE: return "DK_MODULE";
    default: return "UNKNOWN";
    }
}

//-----------------------------------------------------------------------------
// Helper function for discovery logging
//
void log_api_package(const IMdl_info* info, int level)
{
    string shift("  ");
    for (int s = 0; s < level; s++)
        shift.append("  ");

    // Retrieve base properties
    cout << "\n" << shift.c_str() << "simple name: " << info->get_simple_name();
    cout << "\n" << shift.c_str() << "qualified name: " << info->get_qualified_name();
    cout << "\n" << shift.c_str() << "kind: " << DK_to_string(info->get_kind());

    // Retrieve module properties
    if (info->get_kind() == IMdl_info::DK_MODULE)
    {
        const mi::base::Handle<const IMdl_module_info> module_info(
            info->get_interface<const IMdl_module_info>());
        cout << "\n" << shift.c_str() << "search path index: "
            << module_info->get_search_path_index();
        cout << "\n" << shift.c_str() << "search path: "
            << module_info->get_search_path();
        mi::base::Handle<const mi::IString> res_path(module_info->get_resolved_path());
        cout << "\n" << shift.c_str() << "resolved path: " << res_path->get_c_str();
        cout << "\n" << shift.c_str() << "found in archive: "
            << (module_info->in_archive() ? "true" : "false");
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

    // Retrieve package properties
    if (info->get_kind() == IMdl_info::DK_PACKAGE)
    {
        const mi::base::Handle<const IMdl_package_info> package_info(
            info->get_interface<const IMdl_package_info>());

        const mi::Size spi_count = package_info->get_search_path_index_count();
        if (spi_count > 0)
        {
            cout << "\n" << shift.c_str() << "discovered in " << spi_count << " search paths:";
            for (mi::Size i = 0; i < spi_count; ++i)
            {
                cout << "\n" << shift.c_str() << "* search path index: "
                    << package_info->get_search_path_index(i);
                cout << "\n" << shift.c_str() << "  search path: "
                    << package_info->get_search_path(i);
                mi::base::Handle<const mi::IString> res_path(package_info->get_resolved_path(i));
                cout << "\n" << shift.c_str() << "  resolved path: " << res_path->get_c_str();
                cout << "\n" << shift.c_str() << "  found in archive: " 
                     << (package_info->in_archive(i) ? "true" : "false");
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

    cerr << "\n Unhandled IMdl_info::Kind found!\n";
}


//-----------------------------------------------------------------------------
// 
//
int main(int argc, char* argv[])
{
    vector<string>  search_paths;
    if (argc == 1)
        search_paths.push_back(get_samples_mdl_root());
    else
        // Add search paths from command line
        for (int index = 1; index < argc; ++index)
            search_paths.push_back(argv[index]);

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(load_and_get_ineuray());
    check_success(neuray.is_valid_interface());

    // Config root paths and logging
    configure(neuray.get(), search_paths);

    // Start the MDL SDK
    mi::Sint32 result = neuray->start();
    check_start_success(result);
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
            disc_result(discovery_api->discover());

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
            for (size_t p = 0; p < search_paths.size(); ++p)
                m << search_paths[p] << " ";
            m << "in " << elapsed_seconds.count() << " seconds \n\n";
            cerr << m.str();
        }
        else
            cerr << "Failed to create collapsing graph out of search path"<<search_paths[0]<<"\n";

        discovery_api = 0;
    }

    // Shut down the MDL SDK
    check_success(neuray->shutdown() == 0);
    neuray = 0;

    // Unload the MDL SDK
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}
