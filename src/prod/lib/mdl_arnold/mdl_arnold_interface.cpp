/***************************************************************************************************
* Copyright 2020 NVIDIA Corporation. All rights reserved.
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
    AtNode* options = AiUniverseGetOptions();

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
    list = expand_env_vars(AiNodeGetStr(options, "procedural_searchpath").c_str());
    paths = mi::examples::strings::split(list, separator);
    for (const auto& p : paths)
        push_back_uniquely(mdl_search_paths, p);

    list = expand_env_vars(AiNodeGetStr(options, "texture_searchpath").c_str());
    paths = mi::examples::strings::split(list, separator);
    for (const auto& p : paths)
        push_back_uniquely(mdl_search_paths, p);

    // add the search paths if they are existing
    // assuming that the Arnold will warn about non existing texture or procedural search paths
    for (const auto& p : mdl_search_paths)
        if (mi::examples::io::directory_exists(p))
        {
            AiMsgInfo("[mdl] Adding MDL Search Path: %s", p.c_str());
            m_mdl_config->add_mdl_path(p.c_str());
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
        AiMsgWarning("[mdl] Loading MDL SDK failed.");
        return;
    }

    // access the MDL SDK compiler component
    m_mdl_config = m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_configuration>();

    // configure search separated_list
    set_search_paths();

    // init logger
    auto logger = mi::base::make_handle(new Internal_logger(this));
    m_mdl_config->set_logger(logger.get());

    // load image plug-ins for texture loading
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
        m_mdl_sdk->get_api_component<mi::neuraylib::IPlugin_configuration>());

    mi::Sint32 result = plugin_conf->load_plugin_library(
        resolve_dynamic_library(plugin_path, "nv_freeimage_ai" MI_BASE_DLL_FILE_EXT).c_str());
    if (result != 0)
    {
        m_state = EMdl_sdk_state::error_freeimage_not_found;
        AiMsgWarning("[mdl] Loading MDL SDK FreeImage library failed.");
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

    // create back-end for native code generation
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_backend_api>());
    m_native_backend = mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE);

    #ifdef ENABLE_DERIVATIVES
        m_native_backend->set_option("texture_runtime_with_derivs", "on");
    #else
        m_native_backend->set_option("texture_runtime_with_derivs", "off");
    #endif
    m_native_backend->set_option("num_texture_results", std::to_string(NUM_TEXTURE_RESULTS).c_str());
    m_native_backend->set_option("num_texture_spaces", "1");
    m_native_backend->set_option("use_builtin_resource_handler", "on");
    m_native_backend->set_option("enable_auxiliary", "on");

    // create factories required to create expressions
    m_factory = m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_factory>();
    m_tf = m_factory->create_type_factory(m_transaction.get());
    m_vf = m_factory->create_value_factory(m_transaction.get());
    m_ef = m_factory->create_expression_factory(m_transaction.get());

    // get import export API for loading modules
    m_mdl_impexp_api = m_mdl_sdk->get_api_component<mi::neuraylib::IMdl_impexp_api>();

    // load default module
    const char* default_module_name = "::ai_mdl";
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
        m_transaction.get(), default_module_name, default_module_src, context.get()) < 0)
    {
        m_state = EMdl_sdk_state::error_default_module_invalid;
        return;
    }


    mi::base::Handle<const mi::IString> default_module_db_name(m_factory->get_db_module_name(
        default_module_name));

    if (!default_module_db_name)
    {
        m_state = EMdl_sdk_state::error_default_module_invalid;
        return;
    }

    m_default_material_db_name = default_module_db_name->get_c_str();
    m_default_material_db_name += "::not_available";
    m_state = EMdl_sdk_state::loaded;
}


Mdl_sdk_interface::~Mdl_sdk_interface()
{
    m_ef = nullptr;
    m_tf = nullptr;
    m_vf = nullptr;
    m_factory = nullptr;
    m_native_backend = nullptr;

    m_transaction->commit();
    m_transaction = nullptr;

    m_mdl_impexp_api = nullptr;
    m_mdl_config = nullptr;

    m_mdl_sdk->shutdown();
    m_mdl_sdk = nullptr;

    if (!unload(m_so_handle))
        AiMsgWarning("[mdl] Unloading the plugin failed.\n");
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
