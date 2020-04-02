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

#include "example_shared.h"

namespace mdl_d3d12
{

    Mdl_sdk::Mdl_sdk(Base_application* app)
        : m_app(app)
        , use_class_compilation(app->get_options()->use_class_compilation)
        , m_neuray(nullptr)
        , m_mdl_compiler(nullptr)
        , m_database(nullptr)
        , m_image_api(nullptr)
        , m_hlsl_backend(nullptr)
        , m_library(nullptr)
    {

        // Access the MDL SDK
        m_neuray = load_and_get_ineuray();
        if (!m_neuray.is_valid_interface())
        {
            log_error("Failed to load the MDL SDK.", SRC);
            return;
        }

        m_mdl_compiler = mi::base::Handle<mi::neuraylib::IMdl_compiler>(
            m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

        mi::base::Handle<mi::base::ILogger> logger(new Default_logger());
        m_mdl_compiler->set_logger(logger.get());

        // add admin space search paths before user space paths
        auto admin_space_paths = get_mdl_admin_space_search_paths();
        for (const auto& path : admin_space_paths)
            if (m_mdl_compiler->add_module_path(path.c_str()) != 0)
                log_warning("Failed to add the admin space search path: " + path, SRC);

        auto user_space_paths = get_mdl_user_space_search_paths();
        for (const auto& path : user_space_paths)
            if (m_mdl_compiler->add_module_path(path.c_str()) != 0)
                log_warning("Failed to add the user space search path : " + path, SRC);

        // also add example search paths
        const std::string mdl_root = get_samples_mdl_root();
        if (m_mdl_compiler->add_module_path(mdl_root.c_str()) != 0)
            log_warning("Failed to add the MDL example search path: " + mdl_root, SRC);

        // add search paths
        for (auto path : m_app->get_options()->mdl_paths)
        {
            if (m_mdl_compiler->add_module_path(path.c_str()) != 0)
            {
                log_error("Failed to add custom search path: " + path, SRC);
                return;
            }
        }

        // Load the FreeImage plugin.
        if (m_mdl_compiler->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT) != 0)
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
        m_mdl_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(
            m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // create and setup HLSL backend
        m_hlsl_backend = m_mdl_compiler->get_backend(mi::neuraylib::IMdl_compiler::MB_HLSL);

        if (m_hlsl_backend->set_option(
            "num_texture_results", std::to_string(get_num_texture_results()).c_str()) != 0)
            return;

        if (m_hlsl_backend->set_option(
            "num_texture_spaces", "2") != 0)
            return;

        if (m_hlsl_backend->set_option("texture_runtime_with_derivs",
            m_app->get_options()->automatic_derivatives ? "on" : "off") != 0)
            return;

        if (m_hlsl_backend->set_option("enable_auxiliary",
            m_app->get_options()->enable_auxiliary ? "on" : "off") != 0)
            return;


        // The HLSL backend supports no pointers, which means we need use fixed size arrays
        if (m_hlsl_backend->set_option("df_handle_slot_mode", "none") != 0)
            return;

        m_transaction = new Mdl_transaction(this);
        m_library = new Mdl_material_library(m_app, this, m_app->get_options()->share_target_code);
    }

    Mdl_sdk::~Mdl_sdk()
    {
        delete m_transaction;
        delete m_library;

        m_mdl_factory = nullptr;
        m_image_api = nullptr;
        m_hlsl_backend = nullptr;
        m_mdl_compiler = nullptr;
        m_database = nullptr;

        // Shut down the MDL SDK
        if (m_neuray->shutdown() != 0)
        {
            log_error("Failed to shutdown Neuray (MDL SDK).", SRC);
        }
        m_neuray = nullptr;

        if (!unload())
        {
            log_error("Failed to unload the MDL SDK.", SRC);
        }
    }


    bool Mdl_sdk::log_messages(const mi::neuraylib::IMdl_execution_context* context)
    {
        std::string last_log;
        for (mi::Size i = 0; i < context->get_messages_count(); ++i)
        {
            last_log.clear();

            mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
            last_log += message_kind_to_string(message->get_kind());
            last_log += ": ";
            last_log += message->get_string();

            switch (message->get_severity())
            {
            case mi::base::MESSAGE_SEVERITY_ERROR:
            case mi::base::MESSAGE_SEVERITY_FATAL:
                log_error(last_log, SRC);
                break;

            case mi::base::MESSAGE_SEVERITY_WARNING:
                log_warning(last_log, SRC);
                break;

            default:
                log_info(last_log, SRC);
                break;
            }
        }

        return context->get_error_messages_count() == 0;
    }

    mi::neuraylib::IMdl_execution_context* Mdl_sdk::create_context()
    {
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            m_mdl_factory->create_execution_context());

        context->set_option("experimental", true);
        context->set_option("fold_ternary_on_df", true);
        context->set_option("internal_space", "coordinate_world");

        context->retain(); // do not free the context right away
        return context.get();
    }

    // --------------------------------------------------------------------------------------------

    Mdl_transaction::Mdl_transaction(Mdl_sdk* sdk)
        : m_sdk(sdk)
        , m_transaction_mtx()
    {
        mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
        m_transaction = scope->create_transaction();
    }

    Mdl_transaction::~Mdl_transaction()
    {
        std::lock_guard<std::mutex> lock(m_transaction_mtx);
        m_transaction->commit();
        m_transaction = nullptr;
    }

    void Mdl_transaction::commit()
    {
        std::lock_guard<std::mutex> lock(m_transaction_mtx);
        m_transaction->commit();
        mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
        m_transaction = scope->create_transaction();
    }


}


