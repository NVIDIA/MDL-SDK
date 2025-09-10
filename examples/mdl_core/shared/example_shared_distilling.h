/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/shared/example_shared_distilling.h
//
// Distilling helper functionality shared by all examples

#ifndef EXAMPLE_SHARED_DISTILLING_H
#define EXAMPLE_SHARED_DISTILLING_H

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

#ifdef MI_PLATFORM_WINDOWS
#include <codecvt>
#include <locale>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#endif

#include <mi/base.h>
#include <mi/mdl/mdl_distiller_plugin.h>
#include <mi/mdl/mdl_distiller_plugin_api.h>

using std::pair;
using std::make_pair;
using std::string;

// Simple logger subclass
class Simple_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity /*level*/,
        const char* /*module_category*/,
        const mi::base::Message_details& /*details*/,
        const char* message) override
    {
        fprintf(stderr, "%s\n", message);
#ifdef MI_PLATFORM_WINDOWS
        fflush(stderr);
#endif
    }

    void message(
        mi::base::Message_severity level,
        const char* module_category,
        const char* message) override
    {
        this->message(level, module_category, mi::base::Message_details(), message);
    }
};

// The Distilling helper class loads/unloads a MDL distilling plug-in dynamic library (default: mdl_distiller)
// initializes each distiller plug-in contained in the library and collect all available distiller targets.
// Provides
class Distiller_helper
{
public:
    Distiller_helper();
    ~Distiller_helper();

    const mi::mdl::IMaterial_instance* distill(
        mi::mdl::IMDL* mdl_compiler,
        mi::mdl::ICall_name_resolver& call_resolver,
        mi::mdl::IRule_matcher_event* event_handler,
        const mi::mdl::IMaterial_instance* material_instance,
        const char* target,
        mi::mdl::Distiller_options* options,
        mi::Sint32* p_error);

private:
    mi::base::Plugin_factory* load_distiller_plugin(const char* filename = 0);
    bool unload_distiller_plugin();
    
    std::string get_plugin_filename() const;

    /// The OS-specific handle to the dynamic library.
    void* m_handle;

    /// Plugin factory
    mi::base::Plugin_factory* m_distiller_factory;

    /// Plugin library path
    std::string m_plugin_path;

    /// Logger interface
    mi::base::Handle< mi::base::ILogger> m_logger;
};

Distiller_helper::Distiller_helper()
{
    m_logger = mi::base::make_handle< mi::base::ILogger>(new Simple_logger);

    m_distiller_factory = load_distiller_plugin();
    check_success(m_distiller_factory);
}

Distiller_helper::~Distiller_helper()
{
    m_logger = 0;
    check_success(unload_distiller_plugin());
}

const mi::mdl::IMaterial_instance* Distiller_helper::distill(
    mi::mdl::IMDL* mdl_compiler,
    mi::mdl::ICall_name_resolver& call_resolver,
    mi::mdl::IRule_matcher_event* event_handler,
    const mi::mdl::IMaterial_instance* material_instance,
    const char* target,
    mi::mdl::Distiller_options* options,
    mi::Sint32* p_error)
{
    // Error handling
    bool distilled = false;
    mi::Sint32 dummy;
    mi::Sint32& error = p_error != nullptr ? *p_error : dummy;
    error = -2; // no target found

    const mi::mdl::IMaterial_instance* res = nullptr;
    // The "none" target is the only builtin target. It always exists.
    if (strcmp("none", target) == 0)
    {
        res = material_instance;
        res->retain();
        return res;
    }

    // Invoke plugin factory for all plugins
    for (size_t i = 0; !distilled; ++i)
    {
        using Plugin_ptr = std::unique_ptr<mi::base::Plugin, void (*)(mi::base::Plugin*)>;

        Plugin_ptr plugin(
            m_distiller_factory(i, nullptr), [](mi::base::Plugin* p) {if (p) p->release(); });

        // No plugins, or no plugins that meet the target criteria
        if (!plugin)
            break;

        // Check plug-in system version
        const mi::Sint32 plugin_system_version = plugin->get_plugin_system_version();
        if (plugin_system_version != mi::base::Plugin::s_version)
        {
            // Incompatible system version
            std::string message = "Library \"" + this->get_plugin_filename() + "\": ";
            message += "Found plugin with unsupported system version ";
            message += std::to_string(plugin_system_version) + ", ignoring plugin.";
            m_logger->message(mi::base::MESSAGE_SEVERITY_INFO,
                "DISTILLER:COMPILER", message.c_str());

            continue;
        }

        // Check plug-in validity.
        const char* type = plugin->get_type();
        if (!type)
            continue;

        mi::mdl::Mdl_distiller_plugin* distiller_plugin = static_cast<mi::mdl::Mdl_distiller_plugin*>(plugin.get());
        
        // Check plug-in api version
        if (0 == strcmp(type, MI_DIST_MDL_DISTILLER_PLUGIN_TYPE))
        {
            mi::Size api_version = distiller_plugin->get_api_version();
            if (api_version != MI_MDL_DISTILLER_PLUGIN_API_VERSION)
            {
                // Incompatible API version
                std::stringstream message;
                message << "Plugin \"" << this->get_plugin_filename() << "\" has incompatible API version "
                    << api_version << ", but version " << MI_MDL_DISTILLER_PLUGIN_API_VERSION
                    << " is required.";
                m_logger->message(mi::base::MESSAGE_SEVERITY_ERROR,
                    "DISTILLER:COMPILER", message.str().c_str());

                continue;
            }
        }

        distiller_plugin->init(m_logger.get());

        // Iterate through all targets and store them in the dist module
        mi::Size target_count = distiller_plugin->get_target_count();
        for (mi::Size target_index = 0; target_index < target_count; ++target_index)
        {
            const char* plugin_target = distiller_plugin->get_target_name(target_index);
            if (plugin_target && strcmp(plugin_target, target) == 0) // protect against nullptr from the plugin
            {
                std::string message = "Plugin \"";
                message += plugin->get_name();
                message += "\" registered distiller target \"";
                message += target;
                message += "\"";
                m_logger->message(mi::base::MESSAGE_SEVERITY_INFO,
                    "DISTILLER:COMPILER", message.c_str());

                mi::mdl::IDistiller_plugin_api* api =
                    mdl_compiler->create_distiller_plugin_api(material_instance, &call_resolver);
                res = distiller_plugin->distill(
                    *api, event_handler, material_instance, target_index, options, &error);
                api->release();

                distilled = true;
                break;
            }
        }
    }

    if (res)
        res->retain();

    return res;
}

/// Loads the distiller plugin library and calls the Plugin factory function.
///
/// This convenience function loads the mdl_distiller DSO, locates and calls the #mi_plugin_factory()
/// function. It returns an instance of the main #mi::base::Plugin_factory interface.
/// The function may be called only once.
///
/// \param filename    The file name of the DSO. It is feasible to pass \c nullptr, which uses a
///                    built-in default value.
/// \return            A pointer to an instance of the main #mi::base::Plugin_factory
mi::base::Plugin_factory* Distiller_helper::load_distiller_plugin(const char* filename)
{
    if (!filename)
        filename = "mdl_distiller" MI_BASE_DLL_FILE_EXT;

#ifdef MI_PLATFORM_WINDOWS
    void* handle = LoadLibraryA((LPSTR)filename);
    if (!handle) {
        // fall back to libraries in a relative bin folder, relevant for install targets
        std::string fallback = std::string("../../../bin/") + filename;
        handle = LoadLibraryA(fallback.c_str());
    }
    if (!handle)
    {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
                message = buffer;

        fprintf(stderr, "Failed to load library (%lu): " FMT_LPTSTR, error_code, message);

        if (buffer)
            LocalFree(buffer);

        return nullptr;
    }

    void* symbol = GetProcAddress((HMODULE)handle, "mi_plugin_factory");
    if (!symbol) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
                message = buffer;
        
        fprintf(stderr, "GetProcAddress error (%lu): " FMT_LPTSTR, error_code, message);
        
        if (buffer)
            LocalFree(buffer);
        
        return nullptr;
    }
#else // MI_PLATFORM_WINDOWS
    void* handle = dlopen(filename, RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        return nullptr;
    }
    
    void* symbol = dlsym(handle, "mi_plugin_factory");
    
    if (!symbol)
    {
        fprintf(stderr, "%s\n", dlerror());
        return nullptr;
    }

#endif // MI_PLATFORM_WINDOWS
    m_handle = handle;

    return (mi::base::Plugin_factory*)symbol;
}

/// Unloads the mdl_distiller lib.
bool Distiller_helper::unload_distiller_plugin()
{
#ifdef MI_PLATFORM_WINDOWS
    int result = FreeLibrary((HMODULE)m_handle);
    if (result == 0)
    {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
                message = buffer;
        
        fprintf(stderr, "Failed to unload library (%lu): " FMT_LPTSTR, error_code, message);
        
        if (buffer)
            LocalFree(buffer);
        
        return false;
    }
    return true;
#else
    int result = dlclose(m_handle);
    if (result != 0)
    {
        printf("%s\n", dlerror());
        return false;
    }
    return true;
#endif
}

std::string Distiller_helper::get_plugin_filename() const
{
#ifndef MI_PLATFORM_WINDOWS
    void* symbol = dlsym(m_handle, "mi_plugin_factory");
    if (!symbol)
        return {};
    Dl_info dl_info;
    if (!dladdr(symbol, &dl_info))
        return {};
    if (!dl_info.dli_fname)
        return {};
    return dl_info.dli_fname;
#else // MI_PLATFORM_WINDOWS
    TCHAR filename[MAX_PATH];
    if (!GetModuleFileName((HMODULE)m_handle, filename, MAX_PATH))
        return {};

    std::basic_string<TCHAR> tString = filename;

#ifdef UNICODE
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(tString);
#else
    return tString;
#endif

#endif // MI_PLATFORM_WINDOWS
}


#endif // EXAMPLE_SHARED_DISTILLING_H
