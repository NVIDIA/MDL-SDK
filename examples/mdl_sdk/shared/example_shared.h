/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/shared/example_shared.h
//
// Code shared by all examples

#ifndef EXAMPLE_SHARED_H
#define EXAMPLE_SHARED_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mi/mdl_sdk.h>

#ifdef MI_PLATFORM_WINDOWS
#include <direct.h>
#include <Shlobj.h>
#include <Knownfolders.h>
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
static void* g_dso_handle = 0;

// Returns the value of the given environment variable.
//
// \param env_var   environment variable name
// \return          the value of the environment variable or an empty string 
//                  if that variable does not exist or does not have a value.
inline std::string get_environment(const char* env_var)
{
    std::string value;
#ifdef MI_PLATFORM_WINDOWS
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, env_var) == 0 && buf != nullptr) {
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

// Sets the value of the given environment variable.
//
// \param env_var   environment variable name
// \param value     the new value to set.
inline bool set_environment(const char* env_var, const char* value)
{
#ifdef MI_PLATFORM_WINDOWS
    return 0 == _putenv_s(env_var, value);
#else
    return 0 == setenv(env_var, value, 1);
#endif
}

// Checks if the given directory exists.
//
// \param  directory path to check
// \return true, of the path points to a directory, false if not
inline bool dir_exists(const char* path)
{
#ifdef MI_PLATFORM_WINDOWS
    DWORD attrib = GetFileAttributesA(path);
    return (attrib != INVALID_FILE_ATTRIBUTES) && (attrib & FILE_ATTRIBUTE_DIRECTORY);
#else
    DIR* dir = opendir(path);
    if (dir == nullptr)
        return false;

    closedir(dir);
    return true;
#endif
}

// Returns a string pointing to the directory relative to which the SDK examples
// expect their resources, e. g. materials or textures.
inline std::string get_samples_root()
{
    std::string samples_root = get_environment("MDL_SAMPLES_ROOT");
    if (samples_root.empty()) {
        samples_root = MDL_SAMPLES_ROOT;
    }
    if (dir_exists(samples_root.c_str()))
    {
        std::replace(samples_root.begin(), samples_root.end(), '\\', '/');
        return samples_root;
    }

    return ".";
}

// Returns a string pointing to the MDL search root for the SDK examples 
inline std::string get_samples_mdl_root()
{
    return get_samples_root() + "/mdl";
}

// Ensures that the console with the log messages does not close immediately. On Windows, the user
// is asked to press enter. On other platforms, nothing is done as the examples are most likely
// started from the console anyway.
inline void keep_console_open() {
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
inline void check_start_success( mi::Sint32 result)
{
    if( result == 0)
        return;
    fprintf( stderr, "mi::neuraylib::INeuray::start() failed with return code %d.\n", result);
    keep_console_open();
    exit( EXIT_FAILURE);
}

// Default logger for the MDL SDK examples.
//
// This logger is similar to the default logger of the MDL SDK. The only difference is that, on
// Windows, it does \em not convert the UTF8 log messages to UTF16 when stderr is connected to the
// console. This conversion step is wrong for the MDL SDK examples, since they explicitly switch
// the console to UTF8.
class Default_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity level,
        const char* /*module_category*/,
        const mi::base::Message_details& /*details*/,
        const char* message)
    {
        const char* severity = 0;
        switch( level) {
            case mi::base::MESSAGE_SEVERITY_FATAL:        severity = "fatal: "; break;
            case mi::base::MESSAGE_SEVERITY_ERROR:        severity = "error: "; break;
            case mi::base::MESSAGE_SEVERITY_WARNING:      severity = "warn:  "; break;
            case mi::base::MESSAGE_SEVERITY_INFO:         severity = "info:  "; break;
            case mi::base::MESSAGE_SEVERITY_VERBOSE:      return;
            case mi::base::MESSAGE_SEVERITY_DEBUG:        return;
            case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT: return;
        }

        fprintf( stderr, "%s", severity);
        fprintf( stderr, "%s", message);
        putc( '\n', stderr);

#ifdef MI_PLATFORM_WINDOWS
        fflush( stderr);
#endif
    }
};

// Configures the MDL SDK by installing our logger, setting the default MDL search path, and
// loading the freeimage plugin.
//
// \param neuray    pointer to the main MDL SDK interface
inline void configure(mi::neuraylib::INeuray* neuray)
{
    mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
        neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

    mi::base::Handle<mi::base::ILogger> logger(new Default_logger());
    mdl_compiler->set_logger(logger.get());

    // Set the module and texture search path.
    const std::string mdl_root = get_samples_mdl_root();
    check_success(mdl_compiler->add_module_path(mdl_root.c_str()) == 0);

    // Load the FreeImage plugin.
    check_success(mdl_compiler->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT) == 0);
}

// Returns a string-representation of the given message severity
inline const char* message_severity_to_string(mi::base::Message_severity severity)
{
    switch (severity) {

    case mi::base::MESSAGE_SEVERITY_ERROR:
        return "error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return "warning";
    case mi::base::MESSAGE_SEVERITY_INFO:
        return "info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return "verbose";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
        return "debug";
    default:
        break;
    }
    return "";
}


// Returns a string-representation of the given message category
inline const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
{
    switch (message_kind) {

    case mi::neuraylib::IMessage::MSG_INTEGRATION:
        return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
        return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
        return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
        return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
        return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
        return "Compiler DAG generator";
    default:
        break;
    }
    return "";
}


// Prints the messages of the given context.
// Returns true, if the context does not contain any error messages, false otherwise.
inline bool print_messages(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0; i < context->get_messages_count(); ++i) {

        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        fprintf(stderr, "%s %s: %s\n", 
            message_kind_to_string(message->get_kind()),
            message_severity_to_string(message->get_severity()),
            message->get_string());
    }
    return context->get_error_messages_count() == 0;
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
// \param filename    The file name of the DSO. It is feasible to pass \c nullptr, which uses a
//                    built-in default value.
// \return            A pointer to an instance of the main #mi::neuraylib::INeuray interface
inline mi::neuraylib::INeuray* load_and_get_ineuray( const char* filename = 0)
{
    if( !filename)
        filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
#ifdef MI_PLATFORM_WINDOWS
    void* handle = LoadLibraryA((LPSTR) filename);
    if( !handle) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
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
inline bool unload()
{
#ifdef MI_PLATFORM_WINDOWS
    int result = FreeLibrary( (HMODULE)g_dso_handle);
    if( result == 0) {
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

// Sleep the indicated number of seconds.
inline void sleep_seconds( mi::Float32 seconds)
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

// get the current working directory.
inline std::string get_working_directory()
{
    char current_path[FILENAME_MAX];
    #ifdef MI_PLATFORM_WINDOWS
        _getcwd(current_path, FILENAME_MAX);
    #else
        getcwd(current_path, FILENAME_MAX); // TODO
    #endif

    std::string res(current_path);
    std::replace(res.begin(), res.end(), '\\', '/');
    return res;
}

// Returns the folder path of the current executable.
inline std::string get_executable_folder()
{
#ifdef MI_PLATFORM_WINDOWS
    char path[MAX_PATH];
    if (!GetModuleFileNameA(nullptr, path, MAX_PATH))
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
    if (last_sep == nullptr) return "";

    std::string res = std::string(path, last_sep);
    std::replace(res.begin(), res.end(), '\\', '/');
    return res;
}

inline bool is_absolute_path(const std::string& path)
{
    std::string npath = path;
    std::replace(npath.begin(), npath.end(), '\\', '/');

    #ifdef MI_PLATFORM_WINDOWS
        return !(npath.size() < 2 || (npath[0] != '/' && npath[1] != ':'));
    #else
        return npath[0] == '/' || npath[0] != '~';
    #endif
}

namespace
{
    #ifdef MI_PLATFORM_WINDOWS
    //-----------------------------------------------------------------------------
    // helper function to create standard mdl path inside the known folder. WINDOWS only
    //
    std::string get_known_folder(const KNOWNFOLDERID& id, const std::string& postfix)
    {
        // Fetch the 'knownFolder' path.
        HRESULT hr = E_FAIL;
        wchar_t* knownFolderPath = nullptr;
        std::string result;
        #if(_WIN32_WINNT >= 0x0600)
        hr = SHGetKnownFolderPath(id, 0, nullptr, &knownFolderPath);
        #endif
        if (SUCCEEDED(hr))
        {
            // convert from wstring to string and append the postfix
            std::wstring s(knownFolderPath);
            int len;
            int slength = (int) s.length();
            len = WideCharToMultiByte(CP_ACP, 0, s.c_str(), slength, 0, 0, 0, 0);
            result = std::string(len, '\0');
            WideCharToMultiByte(CP_ACP, 0, s.c_str(), slength, &result[0], len, 0, 0);

            result.append(postfix);
            CoTaskMemFree(static_cast<void*>(knownFolderPath));
        }
        return result;
    }
    #endif // MI_PLATFORM_WINDOWS

    std::vector<std::string> string_split(const std::string& input, char sep)
    {
        std::vector<std::string> chunks;

        size_t offset(0);
        size_t pos(0);
        while (pos != std::string::npos)
        {
            pos = input.find(sep, offset);

            if (pos == std::string::npos)
            {
                chunks.push_back(input.substr(offset));
                break;
            }

            chunks.push_back(input.substr(offset, pos - offset));
            offset = pos + 1;
        }
        return chunks;
    }
}

inline std::vector<std::string> get_mdl_admin_space_search_paths()
{
    std::string paths = get_environment("MDL_SYSTEM_PATH");
    if (!paths.empty())
    {
        std::vector<std::string> result = string_split(paths, ';');
        return result;
    }

    // default paths on the different platforms
    std::vector<std::string> result;
    #if defined(MI_PLATFORM_WINDOWS)
        result.emplace_back(get_known_folder(FOLDERID_ProgramData, "/NVIDIA Corporation/mdl"));
    #elif defined(MI_PLATFORM_UNIX)
        result.emplace_back("/opt/nvidia/mdl");
    #elif defined(MI_PLATFORM_MACOSX)
        result.emplace_back("/Library/Application Support/NVIDIA Corporation/mdl");
    #endif
    return result;
}

inline std::vector<std::string> get_mdl_user_space_search_paths()
{
    std::string paths = get_environment("MDL_USER_PATH");
    if (!paths.empty())
    {
        std::vector<std::string> result = string_split(paths, ';');
        return result;
    }

    // default paths on the different platforms
    std::vector<std::string> result;
    #if defined(MI_PLATFORM_WINDOWS)
        result.emplace_back(get_known_folder(FOLDERID_Documents, "/mdl"));
    #else 
        const std::string home = getenv("HOME");
        result.emplace_back(home + "/Documents/mdl");
    #endif
    return result;
}

#ifdef MI_PLATFORM_WINDOWS

#define MAIN_UTF8 main_utf8

#define COMMANDLINE_TO_UTF8 \
int wmain(int argc, wchar_t* argv[]) { \
    char** argv_utf8 = new char*[argc]; \
    for (int i = 0; i < argc; i++) { \
        LPWSTR warg = argv[i]; \
        DWORD size = WideCharToMultiByte(CP_UTF8, 0, warg, -1, NULL, 0, NULL, NULL); \
        check_success(size > 0); \
        argv_utf8[i] = new char[size]; \
        DWORD result = WideCharToMultiByte(CP_UTF8, 0, warg, -1, argv_utf8[i], size, NULL, NULL); \
        check_success(result > 0); \
    } \
    SetConsoleOutputCP(CP_UTF8); \
    int result = main_utf8(argc, argv_utf8); \
    delete[] argv_utf8; \
    return result; \
}

#else

#define MAIN_UTF8 main
#define COMMANDLINE_TO_UTF8

#endif

// checks if a compiled material contains none-invalid hair BSDF
inline bool contains_hair_bsdf(const mi::neuraylib::ICompiled_material* compiled_material)
{
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body(
        compiled_material->get_body());

    mi::base::Handle<const mi::neuraylib::IExpression_list> body_args(body->get_arguments());
    for (mi::Size i = 0, n = body_args->get_size(); i < n; ++i)
    {
        const char* name = body_args->get_name(i);
        if (strcmp(name, "hair") == 0)
        {
            mi::base::Handle<const mi::neuraylib::IExpression> hair_exp(
                body_args->get_expression(i));

            if (hair_exp->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
                return true;

            mi::base::Handle<const mi::neuraylib::IExpression_constant> hair_exp_const(
                hair_exp->get_interface<const mi::neuraylib::IExpression_constant>());

            mi::base::Handle<const mi::neuraylib::IValue> hair_exp_const_value(
                hair_exp_const->get_value());

            return hair_exp_const_value->get_kind() != mi::neuraylib::IValue::VK_INVALID_DF;
        }
    }
    return true;
}

// Extracts and returns the module name from a qualified material name
inline std::string get_module_name(const std::string& material_name)
{
    std::size_t p = material_name.rfind("::");
    if (p == std::string::npos)
        return material_name;
    return material_name.substr(0, p);
}

// helper function to extract the material name from a fully-qualified material name
inline std::string get_material_name(const std::string& material_name)
{
    std::size_t p = material_name.rfind("::");
    if(p == std::string::npos)
        return material_name;
    return material_name.substr(p+2, material_name.size() - p);
}

#endif // MI_EXAMPLE_SHARED_H
