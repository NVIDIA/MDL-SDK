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


#include "mdl_sdk_wrapper.h"

// MDL
#include <mi/mdl_sdk.h>
#include "example_shared.h"

// system
#include <iostream>

// project
#include "cache/mdl_cache.h"
#include "cache/mdl_cache_serializer_xml_impl.h"
#include "utilities/platform_helper.h"

Mdl_browser_logger::Mdl_browser_logger(bool trace)
    : m_trace(trace)
{
}

void Mdl_browser_logger::message(mi::base::Message_severity level, 
                                 const char* mc, 
                                 const char* message)
{
    if ((m_trace) || (level == mi::base::MESSAGE_SEVERITY_ERROR))
        fprintf(stderr, "%s\n", message);
}

Mdl_sdk_wrapper::Mdl_sdk_wrapper(const std::vector<std::string>& search_paths, bool cache_rebuild)
{
    start(search_paths, cache_rebuild);
}

Mdl_sdk_wrapper::~Mdl_sdk_wrapper()
{
    shutdown();
}


mi::neuraylib::IMdl_compiler* Mdl_sdk_wrapper::get_compiler() const
{
    m_compiler->retain(); 
    return m_compiler.get();
}

mi::neuraylib::IMdl_archive_api* Mdl_sdk_wrapper::get_archive_api() const
{
    m_archive_api->retain();
    return m_archive_api.get();
}

mi::neuraylib::IMdl_discovery_api* Mdl_sdk_wrapper::get_discovery() const
{
    m_discovery->retain(); 
    return m_discovery.get();
}

mi::neuraylib::ITransaction* Mdl_sdk_wrapper::get_transaction() const
{
    m_transaction->retain(); 
    return m_transaction.get();
}

bool Mdl_sdk_wrapper::start(const std::vector<std::string>& search_paths, bool cache_rebuild)
{

    // Access the MDL SDK
    m_neuray = mi::base::Handle<mi::neuraylib::INeuray>(load_and_get_ineuray());
    if (!m_neuray.is_valid_interface()) return false;

    // Add MDL search paths
    m_compiler = mi::base::Handle<mi::neuraylib::IMdl_compiler>(
        m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

    m_logger = mi::base::make_handle(new Mdl_browser_logger(false));
    m_compiler->set_logger(m_logger.get());

    // clear all search paths and add specified default ones
    m_compiler->clear_module_paths(); // contains the current working directory

    // use default paths if none are specified explicitly
    if (search_paths.empty())
    {
        // add admin space search paths before user space paths
        auto admin_space_paths = Platform_helper::get_mdl_admin_space_directories();
        for (const auto& path : admin_space_paths)
            m_compiler->add_module_path(path.c_str());

        auto user_space_paths = Platform_helper::get_mdl_user_space_directories();
        for (const auto& path : user_space_paths)
            m_compiler->add_module_path(path.c_str());
    }
    else
    {
        // add the paths specified on the comand line 
        for(const auto& path : search_paths)
            m_compiler->add_module_path(path.c_str());

    }

    // print paths to check
    for (mi::Size i = 0, n = m_compiler->get_module_paths_length(); i < n; ++i)
        std::cout << "MDL Module Path: " << m_compiler->get_module_path(i)->get_c_str() << "\n";

    // Configure the MDL SDK
    // Load plugin required for loading textures
    if( 0 > m_compiler->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT))
    {
        std::cerr << "[Mdl_sdk] start: failed to load 'nv_freeimage' library.\n";
    }

    // Start the MDL SDK
    check_start_success(m_neuray->start()); // NOTE: this kills the app in case of failure

    // get archive API
    m_archive_api = mi::base::Handle<mi::neuraylib::IMdl_archive_api>(
        m_neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());

    // get discovery API
    m_discovery = mi::base::Handle<mi::neuraylib::IMdl_discovery_api>(
        m_neuray->get_api_component<mi::neuraylib::IMdl_discovery_api>());

    // get transaction
    auto database = mi::base::Handle<mi::neuraylib::IDatabase>(
        m_neuray->get_api_component<mi::neuraylib::IDatabase>());

    auto scope = mi::base::Handle<mi::neuraylib::IScope>(database->get_global_scope());

    m_transaction = mi::base::Handle<mi::neuraylib::ITransaction>(scope->create_transaction());


    // Update MDL Cache
    m_cache = new Mdl_cache();

    const std::string cache_path = "mdl_cache.xml";
    const Mdl_cache_serializer_xml_impl serializer;

    // discard cache if specified on the command line
    if (!cache_rebuild)
    {
        Platform_helper::tic_toc_log("Load Cache: ", [&]()
        {
            m_cache->load_from_disk(serializer, cache_path);
        });
    }

    // timings are measured withing (broken down)
    if (!m_cache->update(this))
        std::cerr << "[Mdl_sdk] start: failed to update the cache.\n";


    Platform_helper::tic_toc_log("Save Cache: ", [&]()
    {
        if (!m_cache->save_to_disk(serializer, cache_path))
            std::cerr << "[Mdl_sdk] start: failed to store the cache.\n";
    });

    
    return true;
}

bool Mdl_sdk_wrapper::shutdown()
{
    delete m_cache;

    m_transaction->commit();

    check_success(m_neuray->shutdown() == 0);

    return true;
}