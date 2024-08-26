/******************************************************************************
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/shared/utils/mdl.h
//
// Code shared by all examples

#ifndef EXAMPLE_SHARED_UTILS_MDL_H
#define EXAMPLE_SHARED_UTILS_MDL_H

#ifdef IRAY_SDK
    #include <mi/neuraylib.h>
#else
    #include <mi/mdl_sdk.h>
#endif

#include <fstream>

#include "io.h"
#include "os.h"
#include "strings.h"

#ifdef MI_PLATFORM_WINDOWS
    #include <direct.h>
    #include <mi/base/miwindows.h>
#else
    #include <dlfcn.h>
    #include <unistd.h>
    #include <dirent.h>
#endif

#ifdef MI_PLATFORM_MACOSX
    #include <mach-o/dyld.h>   // _NSGetExecutablePath
#endif

namespace mi { namespace examples { namespace mdl
{
    // have to be defined in a linked unit (see example_shared.cpp)
    extern mi::base::Handle<mi::base::ILogger> g_logger;
#ifdef MI_PLATFORM_WINDOWS
    extern HMODULE g_dso_handle;        // Pointer to the DSO handle. Cached here for unload().
#else
    extern void* g_dso_handle;          // Pointer to the DSO handle. Cached here for unload().
#endif


    /// Loads the MDL SDK and calls the main factory function.
    ///
    /// This convenience function loads the MDL SDK DSO, locates and calls the #mi_factory()
    /// function. It returns an instance of the main #mi::neuraylib::INeuray interface.
    /// The function may be called only once.
    ///
    /// \param filename    The file name of the DSO. It is feasible to pass \c nullptr, which uses
    ///                    a built-in default value.
    /// \return            A pointer to an instance of the main #mi::neuraylib::INeuray interface
    mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename = nullptr);

    /// Unloads the MDL SDK.
    bool unload();

    /// Loads a neuray plugin.
    ///
    /// This convenience functions loads a plugin e.g. for texture format support.
    /// In general this is simple and does not require a lot of logic, but since these examples
    /// are used with different kinds of build setups and binary packaging, it makes sense to wrap
    /// the handing of special cases to support the different packagings in one function.
    mi::Sint32 load_plugin(mi::neuraylib::INeuray* neuray, const char* path);

    /// Get the path specified in the MDL_SAMPLES_ROOT environment variable or if not defined,
    /// the path of the directory where the SDK examples expect their example content.
    /// the latter is the example content folder in the source code directory while building the
    /// example, or if that is not valid, the current working directory.
    std::string get_examples_root();

    /// Returns a directory that contains ::nvidia::core_definitions and ::nvida::axf_to_mdl.
    ///
    /// Might also return "." if that directory is the "mdl" subdirectory of #get_examples_root()
    /// and no extra handling is required.
    std::string get_src_shaders_mdl();

    /// Input to the \c configure function. Allows to control the search path setup for the examples
    /// as well as to control the loaded plugins.
    struct Configure_options
    {
        Configure_options();

        /// additional search paths that are added after admin/user and the example search paths
        std::vector<std::string> additional_mdl_paths;

        /// set to false to not add the admin space search paths. It's recommend to leave this true.
        bool add_admin_space_search_paths;

        /// set to false to not add the user space search paths. It's recommend to leave this true.
        bool add_user_space_search_paths;

        bool add_example_search_path;      ///< set to false to not add the example content mdl path
        bool skip_loading_plugins;         ///< set to true to disable (optional) plugin loading

        /// set a custom logger if we want to use a different one than Default_logger
        mi::base::ILogger* logger;
    };

    /// Configures the MDL SDK by installing a default logger, setting the default MDL search path,
    /// and loading the OpenImageIO and DDS image plugins. This done by many examples so it makes
    /// sense to bundle this here in one place and focus on the actual example.
    ///
    /// \param neuray                   pointer to the main MDL SDK interface
    /// \param options                  see \Configure_options fields
    bool configure(
        mi::neuraylib::INeuray* neuray,
        Configure_options options = Configure_options());

    /// Default logger for the MDL SDK examples.
    ///
    /// This logger is similar to the default logger of the MDL SDK. The only difference is that, on
    /// Windows, it does \em not convert the UTF8 log messages to UTF16 when stderr is connected to
    /// the console. This conversion step is wrong for the MDL SDK examples, since they explicitly
    /// switch the console to UTF8.
    class Default_logger;

    /// Many examples accept material names as command line arguments.
    /// In general, the expected input is fully-qualified absolute MDL material name of the form:
    /// [::<package>]::<module>::<material>
    /// This function splits this input into a module and the material name.
    /// Note, this is not working for function names.
    ///
    /// \param argument                     input, a fully-qualified absolute MDL material name
    /// \param[out] module_name             a fully-qualified absolute MDL module name
    /// \param[out] out_material_name       the materials simple name
    /// \param prepend_colons_if_missing    prepend "::" for non-empty module names, if missing
    inline bool parse_cmd_argument_material_name(
        const std::string& argument,
        std::string& out_module_name,
        std::string& out_material_name,
        bool prepend_colons_if_missing = true);

    /// Adds a missing signature to a material name.
    ///
    /// Specifying material signatures on the command-line can be tedious. Hence, this convenience
    /// method is used to add the missing signature. Since there are no overloads for materials, we
    /// can simply search the module for the given material -- or simpler, let the overload
    /// resolution handle that.
    ///
    /// \param module                       the module containing the material
    /// \param material_name                the DB name of the material without signature
    /// \return                             the DB name of the material including signature, or the
    ///                                     empty string in case of errors.
    inline std::string add_missing_material_signature(
        const mi::neuraylib::IModule* module,
        const std::string& material_name);

#ifdef IRAY_SDK
    /// This is a placeholder for a real authentication key. It is used to
    /// allow easy integration of the authentication code in the examples. If
    /// your variant of the neuray library requires an authentication key you need
    /// to replace this file with a file that contains a valid authentication key.
    ///
    /// Alternatively, you can keep this file and put the key into a file named
    /// "examples.lic" (two lines, first line contains the vendor key, second line
    /// contains the secret key).
    inline mi::Sint32 authenticate(mi::neuraylib::INeuray* neuray);
#endif

    // --------------------------------------------------------------------------------------------
    // Implementations
    // --------------------------------------------------------------------------------------------

    inline Configure_options::Configure_options()
        : additional_mdl_paths()
        , add_admin_space_search_paths(true)
        , add_user_space_search_paths(true)
        , add_example_search_path(true)
        , skip_loading_plugins(false)
        , logger(nullptr)
    {}

    // printf() format specifier for arguments of type LPTSTR (Windows only).
    #ifdef MI_PLATFORM_WINDOWS
        #ifdef UNICODE
            #define FMT_LPTSTR "%ls"
        #else // UNICODE
            #define FMT_LPTSTR "%s"
        #endif // UNICODE
    #endif // MI_PLATFORM_WINDOWS

    inline mi::neuraylib::INeuray* load_and_get_ineuray( const char* filename )
    {
        if( !filename)
            #ifdef IRAY_SDK
                filename = "libneuray" MI_BASE_DLL_FILE_EXT;
            #else
                filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
            #endif

        #ifdef MI_PLATFORM_WINDOWS
            HMODULE handle = LoadLibraryA(filename);
            if( !handle) {
                // fall back to libraries in a relative lib folder, relevant for install targets
                std::string fallback = std::string("../../../lib/") + filename;
                handle = LoadLibraryA(fallback.c_str());
            }
            if( !handle) {
                LPTSTR buffer = 0;
                LPCTSTR message = TEXT("unknown failure");
                DWORD error_code = GetLastError();
                if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                    MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
                    message = buffer;
                fprintf( stderr, "Failed to load %s library (%u): " FMT_LPTSTR,
                    filename, error_code, message);
                if( buffer)
                    LocalFree( buffer);
                return 0;
            }
            void* symbol = GetProcAddress(handle, "mi_factory");
            if( !symbol) {
                LPTSTR buffer = 0;
                LPCTSTR message = TEXT("unknown failure");
                DWORD error_code = GetLastError();
                if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                    MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
                    message = buffer;
                fprintf( stderr, "GetProcAddress error (%u): " FMT_LPTSTR, error_code, message);
                if( buffer)
                    LocalFree( buffer);
                return 0;
            }
        #else // MI_PLATFORM_WINDOWS
            void* handle = dlopen( filename, RTLD_LAZY);
            if( !handle) {
                // fall back to libraries in a relative lib folder, relevant for install targets
                std::string fallback = std::string("../../../lib/") + filename;
                handle = dlopen(fallback.c_str(), RTLD_LAZY);
            }
            if( !handle) {
                fprintf( stderr, "%s\n", dlerror());
                return 0;
            }
            void* symbol = dlsym( handle, "mi_factory");
            if( !symbol) {
                fprintf( stderr, "%s\n", dlerror());
                return 0;
            }
        #endif // MI_PLATFORM_WINDOWS
        g_dso_handle = handle;

        mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>( symbol);
        if( !neuray)
        {
            mi::base::Handle<mi::neuraylib::IVersion> version(
                mi::neuraylib::mi_factory<mi::neuraylib::IVersion>( symbol));
            if( !version)
                fprintf( stderr, "Error: Incompatible library.\n");
            else
                fprintf( stderr, "Error: Library version %s does not match header version %s.\n",
                version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
            return 0;
        }
    #ifdef IRAY_SDK
        if (authenticate(neuray) != 0)
        {
            fprintf(stderr, "Error: Authentication failed.\n");
            unload();
            return 0;
        }

    #endif
        return neuray;
    }

    // --------------------------------------------------------------------------------------------

    inline bool unload()
    {
        // Reset the global logger whose destructor might be defined in the library we are going to
        // unload now.
        g_logger.reset();

    #ifdef MI_PLATFORM_WINDOWS
        BOOL result = FreeLibrary(g_dso_handle);
        if( !result) {
            LPTSTR buffer = 0;
            LPCTSTR message = TEXT("unknown failure");
            DWORD error_code = GetLastError();
            if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
                MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
                message = buffer;
            fprintf( stderr, "Failed to unload library (%u): " FMT_LPTSTR, error_code, message);
            if( buffer)
                LocalFree( buffer);
            return false;
        }
        return true;
    #else
        int result = dlclose( g_dso_handle);
        if( result != 0) {
            printf( "%s\n", dlerror());
            return false;
        }
        return true;
    #endif
    }

    // --------------------------------------------------------------------------------------------

    inline mi::Sint32 load_plugin(mi::neuraylib::INeuray* neuray, const char* path)
    {
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

        // try to load the requested plugin before adding any special handling
        mi::Sint32 res = plugin_conf->load_plugin_library(path);
        if (res == 0)
            return 0;

        // fall back to libraries in a relative lib folder, relevant for install targets
        if (strstr(path, "../../../lib/") != path)
        {
            std::string fallback = std::string("../../../lib/") + path;
            fprintf(stderr, "Falling back to load the plugin library: '%s'\n", fallback.c_str());
            return load_plugin(neuray, fallback.c_str());
        }

        // return the failure code
        fprintf(stderr, "Failed to load the plugin library '%s'\n", path);
        return res;
    }

    // --------------------------------------------------------------------------------------------

    class Default_logger : public mi::base::Interface_implement<mi::base::ILogger>
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

    // --------------------------------------------------------------------------------------------

    inline bool configure(
        mi::neuraylib::INeuray* neuray,
        Configure_options options)
    {
        if (!neuray)
        {
            fprintf(stderr, "INeuray is invalid. Loading the SDK probably failed before.");
            return false;
        }

        mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_config(
            neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

        // set user defined or default logger
        if (options.logger)
        {
            logging_config->set_receiving_logger(options.logger);
        }
        else
        {
            logging_config->set_receiving_logger(mi::base::make_handle(new Default_logger()).get());
        }
        g_logger = logging_config->get_forwarding_logger();

        // collect the search paths to add
        std::vector<std::string> mdl_paths(options.additional_mdl_paths);

        if (options.add_example_search_path)
        {
            const std::string example_search_path1 = mi::examples::mdl::get_examples_root() + "/mdl";
            if (example_search_path1 == "./mdl")
            {
                fprintf(stderr,
                    "MDL Examples path was not found, "
                    "consider setting the environment variable MDL_SAMPLES_ROOT.");
            }
            mdl_paths.push_back(example_search_path1);

            const std::string example_search_path2 = mi::examples::mdl::get_src_shaders_mdl();
            if (example_search_path2 != ".")
                mdl_paths.push_back(example_search_path2);
        }

        // add the search paths for MDL module and resource resolution outside of MDL modules
        for (size_t i = 0, n = mdl_paths.size(); i < n; ++i) {
            if (mdl_config->add_mdl_path(mdl_paths[i].c_str()) != 0 ||
                    mdl_config->add_resource_path(mdl_paths[i].c_str()) != 0) {
                fprintf(stderr,
                    "Warning: Failed to set MDL path \"%s\".\n",
                    mdl_paths[i].c_str());
            }
        }

        // add user and system search paths with lowest priority
        if (options.add_user_space_search_paths)
        {
            mdl_config->add_mdl_user_paths();
        }
        if (options.add_admin_space_search_paths)
        {
            mdl_config->add_mdl_system_paths();
        }

        // load plugins if not skipped
        if (options.skip_loading_plugins)
            return true;

        if (load_plugin(neuray, "nv_openimageio" MI_BASE_DLL_FILE_EXT) != 0)
        {
            fprintf(stderr, "Fatal: Failed to load the nv_openimageio plugin.\n");
            return false;
        }

        if (load_plugin(neuray, "dds" MI_BASE_DLL_FILE_EXT) != 0)
        {
            fprintf(stderr, "Fatal: Failed to load the dds plugin.\n");
            return false;
        }

        return true;
    }

    // --------------------------------------------------------------------------------------------

    inline bool parse_cmd_argument_material_name(
        const std::string& argument,
        std::string& out_module_name,
        std::string& out_material_name,
        bool prepend_colons_if_missing)
    {
        out_module_name = "";
        out_material_name = "";
        std::size_t p_left_paren = argument.rfind('(');
        if (p_left_paren == std::string::npos)
            p_left_paren = argument.size();
        std::size_t p_last = argument.rfind("::", p_left_paren-1);

        bool starts_with_colons = argument.length() > 2 && argument[0] == ':' && argument[1] == ':';

        // check for mdle
        if (!starts_with_colons)
        {
            std::string potential_path = argument;
            std::string potential_material_name = "main";

            // input already has ::main attached (optional)
            if (p_last != std::string::npos)
            {
                potential_path = argument.substr(0, p_last);
                potential_material_name = argument.substr(p_last + 2, argument.size() - p_last);
            }

            // is it an mdle?
            if (mi::examples::strings::ends_with(potential_path, ".mdle"))
            {
                if (potential_material_name != "main")
                {
                    fprintf(stderr, "Error: Material and module name cannot be extracted from "
                        "'%s'.\nThe module was detected as MDLE but the selected material is "
                        "different from 'main'.\n", argument.c_str());
                    return false;
                }
                out_module_name = potential_path;
                out_material_name = potential_material_name;
                return true;
            }
        }

        if (p_last == std::string::npos ||
            p_last == 0 ||
            p_last == argument.length() - 2 ||
            (!starts_with_colons && !prepend_colons_if_missing))
        {
            fprintf(stderr, "Error: Material and module name cannot be extracted from '%s'.\n"
                "An absolute fully-qualified material name of form "
                "'[::<package>]::<module>::<material>' is expected.\n", argument.c_str());
            return false;
        }

        if (!starts_with_colons && prepend_colons_if_missing)
        {
            fprintf(stderr, "Warning: The provided argument '%s' is not an absolute fully-qualified"
                " material name, a leading '::' has been added.\n", argument.c_str());
            out_module_name = "::";
        }

        out_module_name.append(argument.substr(0, p_last));
        out_material_name = argument.substr(p_last + 2, argument.size() - p_last);
        return true;
    }

    // --------------------------------------------------------------------------------------------

    inline std::string add_missing_material_signature(
        const mi::neuraylib::IModule* module,
        const std::string& material_name)
    {
        // Return input if it already contains a signature.
        if (material_name.back() == ')')
            return material_name;

        mi::base::Handle<const mi::IArray> result(
            module->get_function_overloads(material_name.c_str()));
        if (!result || result->get_length() != 1)
            return std::string();

        mi::base::Handle<const mi::IString> overloads(
            result->get_element<mi::IString>(static_cast<mi::Size>(0)));
        return overloads->get_c_str();
    }

    // --------------------------------------------------------------------------------------------

#ifdef IRAY_SDK
    inline mi::Sint32 authenticate(mi::neuraylib::INeuray* neuray)
    {
        auto fix_line_ending = [](std::string& s)
        {
            size_t length = s.length();
            if (length > 0 && s[length - 1] == '\r')
                s.erase(length - 1, 1);
        };

        std::ifstream file("examples.lic");
        if (!file.is_open())
            return -1;

        std::string vendor_key;
        std::string secret_key;
        getline(file, vendor_key);
        getline(file, secret_key);
        fix_line_ending(vendor_key);
        fix_line_ending(secret_key);

        return mi::neuraylib::ILibrary_authenticator::authenticate(
            neuray,
            vendor_key.c_str(),
            vendor_key.size(),
            secret_key.c_str(),
            secret_key.size());
    }
#endif

}}}

#endif
