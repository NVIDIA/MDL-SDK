/******************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifdef MI_PLATFORM_WINDOWS
    #include <direct.h>
    #include <mi/base/miwindows.h>
#else
    #include <dlfcn.h>
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

    /// Returns the root directory of the examples.
    ///
    /// The root directory of the examples is the one that contains the "mdl/nvidia/sdk_examples"
    /// directory. The following steps are performed to find it:
    /// - If the environment variable MDL_SAMPLES_ROOT is set, it is returned (without checking for
    ///   the existence of the subdirectory mentioned above).
    /// - Starting from the directory of the executable all parent directories are considered in
    ///   turn bottom-up, checked for the existence of the subdirectory mentioned above, and the
    ///   first successful directory is returned.
    /// - If that subdirectory of the source tree exists, it is returned.
    /// - Finally, the current working directory is returned (as ".").
    std::string get_examples_root();

    /// Returns a directory that contains ::nvidia::core_definitions and ::nvida::axf_to_mdl.
    ///
    /// Might also return "." if that directory is the "mdl" subdirectory of #get_examples_root()
    /// and no extra handling is required.
    ///
    /// The following steps are performed to find it:
    /// - If the environment variable MDL_SRC_SHADERS_MDL is set, it is returned (without checking
    //    for the existence of the MDL modules mentioned above).
    /// - If that subdirectory of the source tree exists, it is returned.
    /// - Finally, the current working directory is returned (as ".").
    std::string get_src_shaders_mdl();

    /// Input to the \c configure function. Allows to control the search path setup for the examples
    /// as well as to control the loaded plugins.
    struct Configure_options
    {
        /// additional search paths that are added after admin/user and the example search paths
        std::vector<std::string> additional_mdl_paths;

        /// set to false to not add the admin space search paths. It's recommend to leave this true.
        bool add_admin_space_search_paths = true;

        /// set to false to not add the user space search paths. It's recommend to leave this true.
        bool add_user_space_search_paths = true;

        /// set to false to not add the example content mdl path
        bool add_example_search_path = true;

        /// set to true to disable (optional) plugin loading
        bool skip_loading_plugins = false;

        /// if true, render on one thread only
        bool single_threaded = false;

        /// set a custom logger if we want to use a different one than Default_logger
        mi::base::ILogger* logger = nullptr;
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
    bool parse_cmd_argument_material_name(
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
    std::string add_missing_material_signature(
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

    // printf() format specifier for arguments of type LPTSTR (Windows only).
    #ifdef MI_PLATFORM_WINDOWS
        #ifdef UNICODE
            #define FMT_LPTSTR "%ls"
        #else // UNICODE
            #define FMT_LPTSTR "%s"
        #endif // UNICODE
    #endif // MI_PLATFORM_WINDOWS

    // The implementation is inline since it is shared with the Python binding.
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
                // fall back to libraries in a relative bin folder, relevant for install targets
                // (Python binding)
                std::string fallback = std::string("../../bin/") + filename;
                handle = LoadLibraryA(fallback.c_str());
            }
            if( !handle) {
                // fall back to libraries in a relative bin folder, relevant for install targets
                // (examples)
                std::string fallback = std::string("../../../bin/") + filename;
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
                fprintf( stderr, "Failed to load %s library (%lu): " FMT_LPTSTR,
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
                fprintf( stderr, "GetProcAddress error (%lu): " FMT_LPTSTR, error_code, message);
                if( buffer)
                    LocalFree( buffer);
                return 0;
            }
        #else // MI_PLATFORM_WINDOWS
            void* handle = dlopen( filename, RTLD_LAZY);
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

    // The implementation is inline since it is shared with the Python binding.
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
            fprintf( stderr, "Failed to unload library (%lu): " FMT_LPTSTR, error_code, message);
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

    // The implementation is inline since it is shared with the Python binding.
    inline mi::Sint32 load_plugin(mi::neuraylib::INeuray* neuray, const char* path)
    {
        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

        // Temporarily disable warnings. This avoids a potentially confusing warning on Windows
        // where the first attempt with plain "path" might fail if PATH is not set correspondingly,
        // although the second attempt suceeds. If both fail, a suitable error message is generated
        // at the end.
        mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_conf(
            neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
        mi::base::Message_severity old_level = logging_conf->get_log_level();
        logging_conf->set_log_level(std::min(mi::base::MESSAGE_SEVERITY_ERROR, old_level));

        // try to load the requested plugin before adding any special handling
        mi::Sint32 res = plugin_conf->load_plugin_library(path);
        if (res == 0) {
            logging_conf->set_log_level(old_level);
            return 0;
        }

#ifdef MI_PLATFORM_WINDOWS
        // fall back to libraries in a relative lib folder, relevant for install targets
        std::string fallback = std::string("../../../bin/") + path;
        res = plugin_conf->load_plugin_library(fallback.c_str());
        if (res == 0) {
            logging_conf->set_log_level(old_level);
            return 0;
        }
#endif

        // return the failure code
        logging_conf->set_log_level(old_level);
        fprintf(stderr, "Failed to load the plugin library '%s'\n", path);
        return res;
    }

#ifdef IRAY_SDK
    // The implementation is inline since it is used by load_and_get_ineuray().
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
