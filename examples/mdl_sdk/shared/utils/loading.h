/******************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

// examples/mdl_sdk/shared/utils/loading.h
//
// Methods to load/unload the SDK library and to load plugins.
//
// Be carefule with additional dependencies. This file is also used standalone without linking
// the entire shared project.

#ifndef EXAMPLE_SHARED_UTILS_LOADING_H
#define EXAMPLE_SHARED_UTILS_LOADING_H

#include <mi/mdl_sdk.h>

#ifdef MI_PLATFORM_WINDOWS
#include <direct.h>
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

namespace mi { namespace examples { namespace mdl {

/// The forwarding logger.
///
/// This instance is \em not initialized by #load_and_get_ineuray, but cleared by #unload().
extern mi::base::Handle<mi::base::ILogger> g_logger;

#ifdef MI_PLATFORM_WINDOWS
/// Pointer to the DSO handle. Cached here for unload().
extern HMODULE g_dso_handle;
#else
/// Pointer to the DSO handle. Cached here for unload().
extern void* g_dso_handle;
#endif

/// Loads the MDL SDK and calls the main factory function.
///
/// This convenience function loads the MDL SDK DSO, locates and calls the #mi_factory()
/// function. It returns an instance of the main #mi::neuraylib::INeuray interface.
/// The function may be called only once.
///
/// \param filename    The file name of the DSO. It is feasible to pass \c nullptr, which uses
///                    a built-in default value.
/// \return            A pointer to an instance of the main #mi::neuraylib::INeuray interface.
mi::neuraylib::INeuray* load_and_get_ineuray( const char* filename = nullptr);

/// Unloads the MDL SDK.
bool unload();

/// Loads an MDL SDK plugin.
///
/// This is a wrapper around #mi::neuraylib::IPlugin_configuration::load_plugin_library().
/// On Windows, it additionally considers plugins in a relative "bin" directory.
mi::Sint32 load_plugin( mi::neuraylib::INeuray* neuray, const char* path);


// --------------------------------------------------------------------------------------------
// Inline implementations such that this file can be used standalone if the two global variables
// are defined (or compile/link loading.cpp).

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
    if( !filename) {
       filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
    }

#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = LoadLibraryA( filename);
    if( !handle) {
        // Fall back to libraries in a relative "bin" directory, relevant for install targets
        // (examples).
        std::string fallback = std::string("../../../bin/") + filename;
        handle = LoadLibraryA( fallback.c_str());
    }
    if( !handle) {
        // Fall back to libraries in a relative "bin" directory, relevant for install targets
        // (Python binding, and examples as tools in vcpkg).
        std::string fallback = std::string( "../../bin/") + filename;
        handle = LoadLibraryA( fallback.c_str());
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
        return nullptr;
    }
    void* symbol = (void*) GetProcAddress( handle, "mi_factory");
    if( !symbol) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT( "unknown failure");
        DWORD error_code = GetLastError();
        if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        fprintf( stderr, "GetProcAddress error (%lu): " FMT_LPTSTR, error_code, message);
        if( buffer)
            LocalFree( buffer);
        return nullptr;
    }
#else // MI_PLATFORM_WINDOWS
    void* handle = dlopen( filename, RTLD_LAZY);
    if( !handle) {
        fprintf( stderr, "%s\n", dlerror());
        return nullptr;
    }
    void* symbol = dlsym( handle, "mi_factory");
    if( !symbol) {
        fprintf( stderr, "%s\n", dlerror());
        return nullptr;
    }
#endif // MI_PLATFORM_WINDOWS

    g_dso_handle = handle;

    mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>( symbol);
    if( !neuray) {
        mi::base::Handle<mi::neuraylib::IVersion> version(
            mi::neuraylib::mi_factory<mi::neuraylib::IVersion>( symbol));
        if( !version)
            fprintf( stderr, "Error: Incompatible library.\n");
        else
            fprintf( stderr, "Error: Library version %s does not match header version %s.\n",
                version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
        return nullptr;
    }


    return neuray;
}


inline bool unload()
{
    // Reset the global logger whose destructor might be defined in the library we are going to
    // unload now.
    g_logger.reset();

    // Handle failed loads and repeated unloads.
    if( !g_dso_handle)
        return false;

#ifdef MI_PLATFORM_WINDOWS
    BOOL result = FreeLibrary( g_dso_handle);
    if( !result) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT( "unknown failure");
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
    g_dso_handle = nullptr;
    return true;
#else
    int result = dlclose( g_dso_handle);
    if( result != 0) {
        printf( "%s\n", dlerror());
        return false;
    }
    g_dso_handle = nullptr;
    return true;
#endif
}


inline mi::Sint32 load_plugin( mi::neuraylib::INeuray* neuray, const char* path)
{
    mi::base::Handle plugin_configuration(
        neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

    // Temporarily disable warnings. This avoids a potentially confusing warning on Windows
    // where the first attempt with plain "path" might fail if PATH is not set correspondingly,
    // although the second attempt suceeds. If both fail, a suitable error message is generated
    // at the end.
    mi::base::Handle logging_configuration(
        neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    mi::base::Message_severity old_level = logging_configuration->get_log_level();
    logging_configuration->set_log_level( std::min( mi::base::MESSAGE_SEVERITY_ERROR, old_level));

    // Try to load the requested plugin before adding any special handling
    mi::Sint32 result = plugin_configuration->load_plugin_library( path);
    if( result == 0) {
        logging_configuration->set_log_level( old_level);
        return result;
    }

#ifdef MI_PLATFORM_WINDOWS
    // Fall back to libraries in a relative "bin" directory, relevant for install targets
    // (examples).
    std::string fallback = std::string( "../../../bin/") + path;
    result = plugin_configuration->load_plugin_library( fallback.c_str());
    if( result == 0) {
        logging_configuration->set_log_level( old_level);
        return result;
    }
    // Fall back to libraries in a relative "bin" directory, relevant for install targets
    // (examples as tools in vcpkg).
    fallback = std::string( "../../bin/") + path;
    result = plugin_configuration->load_plugin_library( fallback.c_str());
    if( result == 0) {
        logging_configuration->set_log_level( old_level);
        return result;
    }
#endif

    logging_configuration->set_log_level( old_level);
    fprintf( stderr, "Failed to load the plugin library '%s'\n", path);
    return result;
}



#ifdef MI_PLATFORM_WINDOWS
#undef FMT_LPTSTR
#endif // MI_PLATFORM_WINDOWS

}}}

#endif // EXAMPLE_SHARED_UTILS_LOADING_H

