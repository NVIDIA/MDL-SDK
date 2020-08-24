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

#include "mdl_sdk.h"
#include "base_application.h"
#include "mdl_material.h"
#include "mdl_material_library.h"
#include "mdl_material_target.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{

/// Custom logger
class Mdl_disabled_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity level,
        const char* /*category*/,
        const mi::base::Message_details& /*details*/,
        const char* message) override
    {
        // the application uses the context to report errors in a more local context
        // therefore this global logger is used only for fatal events
        if (level == mi::base::MESSAGE_SEVERITY_FATAL)
        {
            std::string log = "[FATAL] ";
            log += message;
            log_error(log);
        }
    }
};

} // anonymous

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Mdl_sdk::Mdl_sdk(Base_application* app)
    : m_app(app)
    , m_neuray(nullptr)
    , m_database(nullptr)
    , m_image_api(nullptr)
    , m_hlsl_backend(nullptr)
    , m_mdl_impexp_api(nullptr)
    , m_library(nullptr)
    , m_mdl_options()
    , m_valid(false)
{
    // init mdl options
    m_mdl_options.use_class_compilation = app->get_options()->use_class_compilation;
    m_mdl_options.fold_all_bool_parameters = false;
    m_mdl_options.fold_all_enum_parameters = false;
    m_mdl_options.enable_shader_cache = app->get_options()->enable_shader_cache;

    // Access the MDL SDK
    m_neuray = mi::examples::mdl::load_and_get_ineuray();
    if (!m_neuray.is_valid_interface())
    {
        log_error("Failed to load the MDL SDK.", SRC);
        return;
    }

    mi::base::Handle<const mi::neuraylib::IVersion> version(
        m_neuray->get_api_component<const mi::neuraylib::IVersion>());
    log_info("Loaded MDL SDK library version: " + std::string(version->get_string()));

    m_config = m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
    mi::base::Handle<mi::base::ILogger> logger(new Mdl_disabled_logger());
    m_config->set_logger(logger.get());

    // search path setup is done during scene loading as the scene folder is added too
    // reconfigure_search_paths();

    // Load the FreeImage plugin.
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
        m_neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

    if (plugin_conf->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT) != 0)
    {
        log_error("Failed to load the 'nv_freeimage' plugin.", SRC);
        return;
    }

    // Start the MDL SDK
    mi::Sint32 result = m_neuray->start();
    if (result != 0)
    {
        log_error("Failed to start Neuray (MDL SDK) with return code: " +
                    std::to_string(result), SRC);
        return;
    }

    m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();
    m_image_api = m_neuray->get_api_component<mi::neuraylib::IImage_api>();
    m_mdl_factory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();
    m_mdl_impexp_api = m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>();
    m_evaluator_api = m_neuray->get_api_component<mi::neuraylib::IMdl_evaluator_api>();

    // create and setup HLSL backend
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    m_hlsl_backend = mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL);

    if (m_hlsl_backend->set_option(
        "num_texture_results",
        std::to_string(app->get_options()->texture_results_cache_size).c_str()) != 0)
            return;

    if (m_hlsl_backend->set_option(
        "num_texture_spaces", "1") != 0)
        return;

    if (m_hlsl_backend->set_option("texture_runtime_with_derivs",
        m_app->get_options()->automatic_derivatives ? "on" : "off") != 0)
        return;

    if (m_hlsl_backend->set_option("enable_auxiliary",
        m_app->get_options()->enable_auxiliary ? "on" : "off") != 0)
        return;


    // The HLSL backend supports no pointers, which means we need use fixed size arrays
    if (m_hlsl_backend->set_option("df_handle_slot_mode", "none") != 0)
    {
        log_error("Backend option 'df_handle_slot_mode' invalid.", SRC);
        return;
    }

    // Enable scene data queries from MDL to the renderer (also known as Prim-vars)
    // By passing the asterisk (*) all names that appear in MDL, will be forwarded to the
    // renderer. A list of supported names is also possible. Scene data with an unsupported
    // name will be automatically replaces by the corresponding default value.
    if (m_hlsl_backend->set_option("scene_data_names", "*") != 0)
        return;

    m_transaction = new Mdl_transaction(this);
    m_library = new Mdl_material_library(m_app, this);
    m_valid = true;
}

// ------------------------------------------------------------------------------------------------

void Mdl_sdk::reconfigure_search_paths()
{
    // add all search paths from scratch
    m_config->clear_mdl_paths();

    // add admin space search paths before user space paths
    m_config->add_mdl_system_paths();
    m_config->add_mdl_user_paths();

    // also add example search paths.
    // additionally, add the executable folder (if it contains an mdl folder).
    std::string example_root = mi::examples::mdl::get_examples_root() + "/mdl";
    if (!mi::examples::io::directory_exists(example_root) ||
        m_config->add_mdl_path(example_root.c_str()) != 0)
        log_warning("Failed to add the MDL example search path: " + example_root, SRC);

    std::string app_level_mdl_folder = mi::examples::io::get_executable_folder() + "/mdl";
    if (mi::examples::io::directory_exists(app_level_mdl_folder) &&
        m_config->add_mdl_path(app_level_mdl_folder.c_str()) != 0)
        log_warning("Failed to add the Executable directory as search path: " +
            app_level_mdl_folder, SRC);

    // add search paths as defined by the user
    for (auto path : m_app->get_options()->mdl_paths)
        if (m_config->add_mdl_path(path.c_str()) != 0)
            log_error("Failed to add custom search path: " + path, SRC);

    // and finally, with least priority, the scene directory
    std::string scene_directory = mi::examples::io::dirname(m_app->get_scene_path());
    if (!scene_directory.empty() && m_config->add_mdl_path(scene_directory.c_str()) != 0)
        log_error("Failed to add scene directory as search path: " + scene_directory, SRC);

    // list the active set of search paths
    std::string search_paths = "Search paths used in the following order:";
    for (mi::Size i = 0, n = m_config->get_mdl_paths_length(); i < n; i++)
    {
        mi::base::Handle<const mi::IString> path(m_config->get_mdl_path(i));
        search_paths += "\n                      - ";
        search_paths += path->get_c_str();
    }
    log_info(search_paths);
}

// ------------------------------------------------------------------------------------------------

Mdl_sdk::~Mdl_sdk()
{
    delete m_transaction;
    delete m_library;

    m_mdl_impexp_api = nullptr;
    m_mdl_factory = nullptr;
    m_image_api = nullptr;
    m_evaluator_api = nullptr;
    m_hlsl_backend = nullptr;
    m_database = nullptr;
    m_config = nullptr;

    // Shut down the MDL SDK
    if (m_neuray->shutdown() != 0)
    {
        log_error("Failed to shutdown Neuray (MDL SDK).", SRC);
    }
    m_neuray = nullptr;

    if (!mi::examples::mdl::unload())
    {
        log_error("Failed to unload the MDL SDK.", SRC);
    }
}

// ------------------------------------------------------------------------------------------------

bool Mdl_sdk::log_messages(
    const std::string& message,
    const mi::neuraylib::IMdl_execution_context* context,
    const std::string& file,
    int line)
{
    if (context->get_messages_count() == 0)
        return true;

    bool has_errors = context->get_error_messages_count() > 0;
    std::string log = has_errors ? message : "";
    mi::base::Message_severity most_severe = mi::base::MESSAGE_SEVERITY_DEBUG;
    for (mi::Size i = 0, n = context->get_messages_count(); i < n; ++i)
    {
        log += "\n";
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        switch (message->get_severity())
        {
            case mi::base::MESSAGE_SEVERITY_FATAL:      log += "  fatal:   "; break;
            case mi::base::MESSAGE_SEVERITY_ERROR:      log += "  error:   "; break;
            case mi::base::MESSAGE_SEVERITY_WARNING:    log += "  warning: "; break;
            case mi::base::MESSAGE_SEVERITY_INFO:       log += "  info:    "; break;
            case mi::base::MESSAGE_SEVERITY_VERBOSE:    log += "  verbose: "; break;
            case mi::base::MESSAGE_SEVERITY_DEBUG:      log += "  debug:   "; break;
            default:                                    log += "           "; break;
        }
        most_severe = std::min(most_severe, message->get_severity());
        const std::string kind = std::to_string(message->get_kind());
        if (!kind.empty())
            log += kind + ": ";
        log += message->get_string();
    }

    switch (most_severe)
    {
        case mi::base::MESSAGE_SEVERITY_FATAL:
        case mi::base::MESSAGE_SEVERITY_ERROR:
            log_error(log, file, line);
            break;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            log_warning(log, file, line);
            break;
        case mi::base::MESSAGE_SEVERITY_INFO:
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
        case mi::base::MESSAGE_SEVERITY_DEBUG:
        default:
            log_info(log, file, line);
            break;
    }
    return !has_errors;
}

// ------------------------------------------------------------------------------------------------

mi::neuraylib::IMdl_execution_context* Mdl_sdk::create_context()
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        m_mdl_factory->create_execution_context());

    context->set_option("experimental", true);
    context->set_option("fold_ternary_on_df", false);
    context->set_option("internal_space", "coordinate_world");

    context->retain(); // do not free the context right away
    return context.get();
}

// ------------------------------------------------------------------------------------------------

Mdl_transaction::Mdl_transaction(Mdl_sdk* sdk)
    : m_sdk(sdk)
    , m_transaction_mtx()
{
    mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
    m_transaction = scope->create_transaction();
}

// ------------------------------------------------------------------------------------------------

Mdl_transaction::~Mdl_transaction()
{
    std::lock_guard<std::mutex> lock(m_transaction_mtx);
    m_transaction->commit();
    m_transaction = nullptr;
}

// ------------------------------------------------------------------------------------------------

void Mdl_transaction::commit()
{
    std::lock_guard<std::mutex> lock(m_transaction_mtx);
    m_transaction->commit();
    mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
    m_transaction = scope->create_transaction();
}

}}} // mi::examples::mdl_d3d12

