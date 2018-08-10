/******************************************************************************
 * Copyright (c) 2012-2018, NVIDIA CORPORATION. All rights reserved.
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

// examples/example_shared.h
//
// Code shared by all examples

#ifndef EXAMPLE_SHARED_H
#define EXAMPLE_SHARED_H

#include <cstdio>
#include <cstdlib>

#include <mi/mdl_sdk.h>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#endif

#ifdef MI_PLATFORM_MACOSX
#include <mach-o/dyld.h>   // _NSGetExecutablePath
#endif

#ifndef MDL_SAMPLES_ROOT
#define MDL_SAMPLES_ROOT "."
#endif

// Pointer to the DSO handle. Cached here for unload().
void* g_dso_handle = 0;

// Returns the value of the given environment variable.
//
// \param env_var   environment variable name
// \return          the value of the environment variable or an empty string 
//                  if that variable does not exist or does not have a value.
std::string get_environment(const char* env_var)
{
    std::string value;
#ifdef MI_PLATFORM_WINDOWS
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, env_var) == 0 && buf != NULL) {
        value = buf;
        free(buf);
    }
#else
    const char* v = getenv(env_var);
    if (v)
        value = v;
#endif
    return value;
}

// Checks if the given directory exists.
//
// \param  directory path to check
// \return true, of the path points to a directory, false if not
bool dir_exists(const char* path)
{
#ifdef MI_PLATFORM_WINDOWS
    DWORD attrib = GetFileAttributesA(path);
    return (attrib != INVALID_FILE_ATTRIBUTES) && (attrib & FILE_ATTRIBUTE_DIRECTORY);
#else
    DIR* dir = opendir(path);
    if (dir == NULL)
        return false;

    closedir(dir);
    return true;
#endif
}

// Returns a string pointing to the directory relative to which the SDK examples
// expect their resources, e. g. materials or textures.
std::string get_samples_root()
{
    std::string samples_root = get_environment("MDL_SAMPLES_ROOT");
    if (samples_root.empty()) {
        samples_root = MDL_SAMPLES_ROOT;
    }
    if (dir_exists(samples_root.c_str()))
        return samples_root;

    return ".";
}

// Returns a string pointing to the MDL search root for the SDK examples 
std::string get_samples_mdl_root()
{
    return get_samples_root() + "/mdl";
}

// Ensures that the console with the log messages does not close immediately. On Windows, the user
// is asked to press enter. On other platforms, nothing is done as the examples are most likely
// started from the console anyway.
void keep_console_open() {
#ifdef MI_PLATFORM_WINDOWS
    if( IsDebuggerPresent()) {
        fprintf( stderr, "Press enter to continue . . . \n");
        fgetc( stdin);
    }
#endif // MI_PLATFORM_WINDOWS
}

// Helper macro. Checks whether the expression is true and if not prints a message and exits.
#define check_success( expr) \
    do { \
        if( !(expr)) { \
            fprintf( stderr, "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
            keep_console_open(); \
            exit( EXIT_FAILURE); \
        } \
    } while( false)

// Helper function similar to check_success(), but specifically for the result of
// #mi::neuraylib::INeuray::start().
void check_start_success( mi::Sint32 result)
{
    if( result == 0)
        return;
    fprintf( stderr, "mi::neuraylib::INeuray::start() failed with return code %d.\n", result);
    keep_console_open();
    exit( EXIT_FAILURE);
}

// Configures the MDL SDK by setting the default MDL search path and loading the 
// freeimage plugin.
//
// \param neuray    pointer to the main MDL SDK interface
void configure(mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
        neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

    // Set the module and texture search path.
    const std::string mdl_root = get_samples_mdl_root();
    check_success(mdl_compiler->add_module_path(mdl_root.c_str()) == 0);

    // Load the FreeImage plugin.
    check_success(mdl_compiler->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT) == 0);
}

// printf() format specifier for arguments of type LPTSTR (Windows only).
#ifdef MI_PLATFORM_WINDOWS
#ifdef UNICODE
#define FMT_LPTSTR "%ls"
#else // UNICODE
#define FMT_LPTSTR "%s"
#endif // UNICODE
#endif // MI_PLATFORM_WINDOWS

// Loads the MDL SDK and calls the main factory function.
//
// This convenience function loads the MDL SDK DSO, locates and calls the #mi_factory()
// function. It returns an instance of the main #mi::neuraylib::INeuray interface.
// The function may be called only once.
//
// \param filename    The file name of the DSO. It is feasible to pass \c NULL, which uses a
//                    built-in default value.
// \return            A pointer to an instance of the main #mi::neuraylib::INeuray interface
mi::neuraylib::INeuray* load_and_get_ineuray( const char* filename = 0)
{
    if( !filename)
        filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
#ifdef MI_PLATFORM_WINDOWS
    void* handle = LoadLibraryA((LPSTR) filename);
    if( !handle) {
        LPTSTR buffer = 0;
        LPTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        fprintf( stderr, "Failed to load library (%u): " FMT_LPTSTR, error_code, message);
        if( buffer)
            LocalFree( buffer);
        return 0;
    }
    void* symbol = GetProcAddress((HMODULE) handle, "mi_factory");
    if( !symbol) {
        LPTSTR buffer = 0;
        LPTSTR message = TEXT("unknown failure");
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
    return neuray;
}

// Unloads the MDL SDK.
bool unload()
{
#ifdef MI_PLATFORM_WINDOWS
    int result = FreeLibrary( (HMODULE)g_dso_handle);
    if( result == 0) {
        LPTSTR buffer = 0;
        LPTSTR message = TEXT("unknown failure");
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

// Sleep the indicated number of seconds.
void sleep_seconds( mi::Float32 seconds)
{
#ifdef MI_PLATFORM_WINDOWS
    Sleep( static_cast<DWORD>( seconds * 1000));
#else
    usleep( static_cast<useconds_t>( seconds * 1000000));
#endif
}

// Map snprintf to _snprintf on Windows.
#ifdef MI_PLATFORM_WINDOWS
#define snprintf _snprintf
#endif

// Returns the folder path of the current executable.
std::string get_executable_folder()
{
#ifdef MI_PLATFORM_WINDOWS
    char path[MAX_PATH];
    if (!GetModuleFileNameA(NULL, path, MAX_PATH))
        return "";

    const char sep = '\\';
#else  // MI_PLATFORM_WINDOWS
    char path[4096];

#ifdef MI_PLATFORM_MACOSX
    uint32_t buflen(sizeof(path));
    if (_NSGetExecutablePath(path, &buflen) != 0)
        return "";
#else  // MI_PLATFORM_MACOSX
    char proc_path[64];
#ifdef __FreeBSD__
    snprintf(proc_path, sizeof(proc_path), "/proc/%d/file", getpid());
#else
    snprintf(proc_path, sizeof(proc_path), "/proc/%d/exe", getpid());
#endif

    ssize_t written = readlink(proc_path, path, sizeof(path));
    if (written < 0 || size_t(written) >= sizeof(path))
        return "";
    path[written] = 0;  // add terminating null
#endif // MI_PLATFORM_MACOSX

    const char sep = '/';
#endif // MI_PLATFORM_WINDOWS

    char *last_sep = strrchr(path, sep);
    if (last_sep == NULL) return "";
    return std::string(path, last_sep + 1);
}

#endif // MI_EXAMPLE_SHARED_H
