/***************************************************************************************************
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
 **************************************************************************************************/

#include "mdl_arnold_interface.h"
#include "mdl_arnold_utils.h"
#include "mdl_arnold.h"

#include <iostream>
#include <set>
#include <string>

#include <ai.h>
#include <ai_msg.h>
#include <ai_nodes.h>
#include <ai_universe.h>
#include <ai_metadata.h>

//-------------------------------------------------------------------------------------------------
// Main entry point
//-------------------------------------------------------------------------------------------------

enum class EMdl_arnold_node
{
    Bsdf,
    //Pattern
};

extern const AtNodeMethods* Mdl_arnold_bsdf_shader_methods;
// extern const AtNodeMethods* Mdl_arnold_pattern_methods;

AI_EXPORT_LIB bool NodeLoader(int i, AtNodeLib* node)
{
    // it is possible to have multiple shader nodes in one plug-in
    // see https://docs.arnoldrenderer.com/display/ARP/Create+a+DLL+that+contains+multiple+shaders
    // this can help to provide multiple interface, i.e. bsdfs, patterns, ...

    switch (static_cast<EMdl_arnold_node>(i))
    {
    case EMdl_arnold_node::Bsdf:
    {
        node->methods = Mdl_arnold_bsdf_shader_methods;
        node->output_type = AI_TYPE_CLOSURE;
        node->name = "mdl";
        node->node_type = AI_NODE_SHADER;
        strcpy(node->version, AI_VERSION);
        return true;
    }

    default:
        return false;
    }
}

//-------------------------------------------------------------------------------------------------
// SDK interface
//-------------------------------------------------------------------------------------------------

namespace
{
    // Returns a string-representation of the given message category
    const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
    {
        switch (message_kind)
        {
        case mi::neuraylib::IMessage::MSG_INTEGRATION:           return "MDL SDK";
        case mi::neuraylib::IMessage::MSG_IMP_EXP:               return "Importer/Exporter";
        case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:       return "Compiler Backend";
        case mi::neuraylib::IMessage::MSG_COMILER_CORE:          return "Compiler Core";
        case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL: return "Compiler Archiver";
        case mi::neuraylib::IMessage::MSG_COMPILER_DAG:          return "Compiler DAG";
        default:                                                 return "";
        }
        return "";
    }

    void print_mdl_message(mi::base::Message_severity level, const char* mc, const char* message)
    {
        switch (level)
        {
        case mi::base::MESSAGE_SEVERITY_FATAL:      AiMsgFatal("%s: %s", mc, message);   break;
        case mi::base::MESSAGE_SEVERITY_ERROR:      AiMsgWarning("%s: %s", mc, message); break; // Print errors as warnings for now.
        case mi::base::MESSAGE_SEVERITY_WARNING:    AiMsgWarning("%s: %s", mc, message); break;
        case mi::base::MESSAGE_SEVERITY_INFO:       AiMsgInfo("%s: %s", mc, message);    break;
        case mi::base::MESSAGE_SEVERITY_DEBUG:
        case mi::base::MESSAGE_SEVERITY_VERBOSE:    AiMsgDebug("%s: %s", mc, message);   break;
        default:                                    return;
        }
    }

} // anonymous


class Internal_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    Internal_logger(Mdl_sdk_interface* mdl_sdk) : m_mdl_sdk(mdl_sdk) {}

    void message(mi::base::Message_severity level,
        const char* mc,
        const mi::base::Message_details&,
        const char* message) override
    {
        // forward important internal messages only
        if (level <= mi::base::MESSAGE_SEVERITY_WARNING)
            print_mdl_message(level, mc, message);
    }

private:
    Mdl_sdk_interface* m_mdl_sdk;
};

namespace
{
    bool exists(const char* filepath)
    {
        if (FILE *file = fopen(filepath, "r"))
        {
            fclose(file);
            return true;
        }
        return false;
    }

    std::string resolve_dynamic_library(
        const std::string& plugin_path,
        const char* filename)
    {
        // check if the library is located next to the Arnold shared library
        // and try to load the dynamic libraries from the same folder
        std::string potential_path = plugin_path + filename;
        if (exists(potential_path.c_str()))
            return potential_path;

        // fall back, let the OS resolve the file based
        return filename;
    }

    inline std::string expand_env_vars(std::string path)
    {
        // check for "[VARNAME]" and expand them
        while (true)
        {
            size_t p0 = path.find_first_of('[');
            size_t p1 = path.find_first_of(']');
            if (p0 == std::string::npos || p1 == std::string::npos)
                break;

            std::string pre = path.substr(0, p0);
            std::string post = (p1 == (path.size() - 1)) ? "" : path.substr(p1 + 1);

            std::string env_name = path.substr(p0 + 1, p1 - p0 - 1);

            path = pre + mi::examples::os::get_environment(env_name.c_str()) + post;
        }
        return path;
    }
}

// ------------------------------------------------------------------------------------------------

namespace
{

// helper to check if a value is in an container
template <typename TValue, typename Alloc, template <typename, typename> class TContainer>
inline bool contains(TContainer<TValue, Alloc>& vector, const TValue& value)
{
    return std::find(vector.begin(), vector.end(), value) != vector.end();
}

// ------------------------------------------------------------------------------------------------

// helper to push back values into a compatible container only if the value is not already
// in that container
template <typename TValue, typename Alloc, template <typename, typename> class TContainer>
inline bool push_back_uniquely(TContainer<TValue, Alloc>& vector, const TValue& value)
{
    if (!contains(vector, value))
    {
        vector.push_back(value);
        return true;
    }
    return false;
}

} // anonymous

// ------------------------------------------------------------------------------------------------

void Mdl_sdk_interface::set_search_paths()
{
    m_mdl_config->clear_mdl_paths();
    m_mdl_config->clear_resource_paths();

    std::vector<std::string> mdl_search_paths;

    // parse options provided by the integrator or the scene file
    AtNode* options = AiUniverseGetOptions(nullptr);

    #ifdef MI_PLATFORM_WINDOWS
        char separator = ';';
    #else
        char separator = ':';
    #endif

    // add default paths
    for (mi::Size i = 0, n = m_mdl_config->get_mdl_system_paths_length(); i < n; ++i)
        push_back_uniquely(mdl_search_paths, std::string(m_mdl_config->get_mdl_system_path(i)));

    for (mi::Size i = 0, n = m_mdl_config->get_mdl_user_paths_length(); i < n; ++i)
        push_back_uniquely(mdl_search_paths, std::string(m_mdl_config->get_mdl_user_path(i)));

    // evaluate MDL_PATHS as additional environment variable to add search paths
    std::string list = mi::examples::os::get_environment("MDL_PATHS");
    std::vector<std::string> paths = mi::examples::strings::split(list, separator);
    for (const auto& p : paths)
        push_back_uniquely(mdl_search_paths, p);

    // add mdl paths specified in the Arnold options
    list = expand_env_vars(AiNodeGetStr(options, AtString("procedural_searchpath")).c_str());
    paths = mi::examples::strings::split(list, separator);
    for (const auto& p : paths)
        push_back_uniquely(mdl_search_paths, p);

    list = expand_env_vars(AiNodeGetStr(options, AtString("texture_searchpath")).c_str());
    paths = mi::examples::strings::split(list, separator);
    for (const auto& p : paths)
        push_back_uniquely(mdl_search_paths, p);

    // add the search paths if they are existing
    // assuming that the Arnold will warn about non existing texture or procedural search paths
    for (const auto& p : mdl_search_paths)
    {
        if (mi::examples::io::directory_exists(p))
        {
            AiMsgInfo("[mdl] Adding MDL Search Path: %s", p.c_str());
            m_mdl_config->add_mdl_path(p.c_str());
        }
        else
        {
            AiMsgInfo("[mdl] Adding MDL Search Path failed. The path does not exist: %s", p.c_str());
        }
    }
}

Mdl_sdk_interface::Mdl_sdk_interface()
    : m_so_handle(nullptr)
    , m_state(EMdl_sdk_state::undefined)
{
    std::string plugin_path = get_current_binary_directory();
    AiMsgInfo("[mdl] Loaded MDL Arnold plugin from: %s", plugin_path.c_str());

    // load the SDK (has to be in the PATH or in the working directory, but could be changed)
    m_mdl_sdk = load_and_get_ineuray(
        resolve_dynamic_library(plugin_path, "libmdl_sdk_ai" MI_BASE_DLL_FILE_EXT).c_str(), &m_so_handle);

    if (!m_mdl_sdk)
    {
        m_state = EMdl_sdk_state::error_libmdl_not_found;
        AiMsgError("[mdl] Loading MDL SDK failed.");
        return;
    }

    // access the MDL SDK compiler component
    m_mdl_config = m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_configuration>();

    // configure search separated_list
    set_search_paths();

    // init logger
    auto logger = mi::base::make_handle(new Internal_logger(this));
    mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_conf(
        m_mdl_sdk->get_api_component<mi::neuraylib::ILogging_configuration>());
    logging_conf->set_receiving_logger(logger.get());

    // load image plug-ins for texture loading
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
        m_mdl_sdk->get_api_component<mi::neuraylib::IPlugin_configuration>());

    mi::Sint32 result = plugin_conf->load_plugin_library(
        resolve_dynamic_library(plugin_path, "nv_openimageio_ai" MI_BASE_DLL_FILE_EXT).c_str());
    if (result != 0)
    {
        m_state = EMdl_sdk_state::error_openimageio_not_found;
        AiMsgWarning("[mdl] Loading MDL SDK OpenImageIO library failed.");
        return;
    }

    result = plugin_conf->load_plugin_library(
        resolve_dynamic_library(plugin_path, "dds_ai" MI_BASE_DLL_FILE_EXT).c_str());
    if (result != 0)
    {
        m_state = EMdl_sdk_state::error_dds_not_found;
        AiMsgWarning("[mdl] Loading MDL SDK DDS library failed.");
        return;
    }

    m_mdl_sdk->start();

    // create a transaction
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        m_mdl_sdk->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    m_transaction = scope->create_transaction();

    // create factories required to create expressions
    m_factory = m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_factory>();
    m_tf = m_factory->create_type_factory(m_transaction.get());
    m_vf = m_factory->create_value_factory(m_transaction.get());
    m_ef = m_factory->create_expression_factory(m_transaction.get());

    // get import export API for loading modules
    m_mdl_impexp_api = m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_impexp_api>();

    // load default module
    m_default_mdl_module_name = AtString("::ai_mdl");
    m_default_mdl_function_name = AtString("not_available()");
    const char* default_module_src =
        "mdl 1.2;\n"
        "import df::*;\n"
        "export material not_available() = material(\n"
        "    surface: material_surface(\n"
        "        df::diffuse_reflection_bsdf(\n"
        "            tint: color(0.8, 0.0, 0.8)\n"
        "        )\n"
        "    )"
        ");";

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(create_context());
    if (m_mdl_impexp_api->load_module_from_string(
        m_transaction.get(), m_default_mdl_module_name.c_str(), default_module_src, context.get()) < 0)
    {
        m_state = EMdl_sdk_state::error_default_module_invalid;
        return;
    }

    // compute the modules DB name using the factory API
    mi::base::Handle<const mi::IString> default_module_db_name(m_factory->get_db_module_name(
        m_default_mdl_module_name.c_str()));
    if (!default_module_db_name)
    {
        m_state = EMdl_sdk_state::error_default_module_invalid;
        return;
    }
    m_default_material_db_name = default_module_db_name->get_c_str();
    m_default_material_db_name += "::" + std::string(m_default_mdl_function_name.c_str());
    m_state = EMdl_sdk_state::loaded;
}


Mdl_sdk_interface::~Mdl_sdk_interface()
{
    m_ef = nullptr;
    m_tf = nullptr;
    m_vf = nullptr;
    m_factory = nullptr;
    m_native_backend = nullptr;

    if(m_transaction)
        m_transaction->commit();
    m_transaction = nullptr;

    m_mdl_impexp_api = nullptr;
    m_mdl_config = nullptr;

    if(m_mdl_sdk)
        m_mdl_sdk->shutdown();
    m_mdl_sdk = nullptr;

    if (m_so_handle && !unload(m_so_handle))
        AiMsgWarning("[mdl] Unloading the plugin failed.\n");
}

void Mdl_sdk_interface::set_be_option(
    mi::neuraylib::IMdl_backend *backend,
    const char *name,
    const char *value)
{
    mi::Sint32 res = backend->set_option(name, value);
    if (res != 0) {
        AiMsgError(
            "[mdl] Setting backend option '%s' = '%s' failed with %d\n", name, value, int(res));
    }
}

mi::neuraylib::IMdl_backend* Mdl_sdk_interface::create_native_backend()
{
    // create backend for native code generation.
    // each thread needs its own backend object
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_backend_api>());
    mi::neuraylib::IMdl_backend* backend =
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE);

    #ifdef ENABLE_DERIVATIVES
        set_be_option(backend, "texture_runtime_with_derivs", "on");
    #else
        set_be_option(backend, "texture_runtime_with_derivs", "off");
    #endif
    set_be_option(backend, "num_texture_results", std::to_string(NUM_TEXTURE_RESULTS).c_str());
    set_be_option(backend, "num_texture_spaces", "1");
    set_be_option(backend, "use_builtin_resource_handler", "on");
    set_be_option(backend, "enable_auxiliary", "on");

    return backend;
}

mi::neuraylib::IMdl_execution_context* Mdl_sdk_interface::create_context()
{
    return m_factory->create_execution_context();
}

bool Mdl_sdk_interface::log_messages(const mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0; i < context->get_messages_count(); ++i)
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));

        print_mdl_message(
            message->get_severity(),
            message_kind_to_string(message->get_kind()),
            message->get_string());
    }

    return context->get_error_messages_count() == 0;
}

// the helper classes currently require the following global variables which
// have to be in one link unit. So either linking the example shared library
// or adding this .cpp file to the build is required.

namespace mi { namespace examples { namespace mdl {

mi::base::Handle<mi::base::ILogger> g_logger;

// required for loading and unloading the SDK
#ifdef MI_PLATFORM_WINDOWS
    HMODULE g_dso_handle = 0;
#else
    void* g_dso_handle = 0;
#endif

}}}
