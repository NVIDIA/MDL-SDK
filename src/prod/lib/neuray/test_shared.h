/******************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

// Code shared by all unit tests

#ifndef PROD_LIB_NEURAY_TEST_SHARED_H
#define PROD_LIB_NEURAY_TEST_SHARED_H

#include <mi/base/config.h>
#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/ineuray.h>


#include <cstdio>
#include <fstream>

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#include <sys/time.h>
#include <unistd.h>
#endif

// Pointer to the DSO handle. Cached here for unload().
void* g_dso_handle = 0;

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

// Checks that there are no error messages (and dumps them otherwise).
#define MI_CHECK_CTX( context) \
    do { \
        if( context->get_error_messages_count() > 0) { \
            for( mi::Size i = 0, n = context->get_messages_count(); i < n; ++i) { \
                mi::base::Handle<const mi::neuraylib::IMessage> message( context->get_message( i));\
                std::cerr << message->get_string() << std::endl; \
            } \
            MI_CHECK_EQUAL( context->get_error_messages_count(), 0); \
        } \
    } while( false)

// Checks that there is at least one error message and that the first one matches the given code.
#define MI_CHECK_CTX_CODE( context, code) \
    MI_CHECK_GREATER( context->get_error_messages_count(), 0); \
    MI_CHECK_EQUAL( mi::base::make_handle( context->get_error_message( 0))->get_code(), code);

// Checks whether the expression is true and if not prints a message and exits.
// Used by the examples, not by the tests (which use MI_CHECK(), etc.).
#define check_success( expr) \
    do { \
        if( !(expr)) { \
            fprintf( stderr, "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
            keep_console_open(); \
            exit( EXIT_FAILURE); \
        } \
    } while( false)

// Checks that there are no error messages (and dumps them otherwise).
//  Used by the examples, not by the tests (which use MI_CHECK_CTX()).
#define check_ctx( context) \
    do { \
        if( context->get_error_messages_count() > 0) { \
            for( mi::Size i = 0, n = context->get_messages_count(); i < n; ++i) { \
                mi::base::Handle<const mi::neuraylib::IMessage> message( context->get_message( i));\
                std::cerr << message->get_string() << std::endl; \
            } \
            check_success( context->get_error_messages_count() == 0); \
        } \
    } while( false)

// printf() format specifier for arguments of type LPTSTR (Windows only).
#ifdef MI_PLATFORM_WINDOWS
#ifdef UNICODE
#define FMT_LPTSTR "%ls"
#else // UNICODE
#define FMT_LPTSTR "%s"
#endif // UNICODE
#endif // MI_PLATFORM_WINDOWS

// Loads the neuray library and calls the main factory function.
//
// This convenience function loads the neuray DSO, locates and calls the #mi_factory()
// function. It returns an instance of the main #mi::neuraylib::INeuray interface.
// The function may be called only once.
//
// \param filename    The file name of the DSO. It is feasible to pass \c nullptr, which uses a
//                    built-in default value.
// \return            A pointer to an instance of the main #mi::neuraylib::INeuray interface
mi::neuraylib::INeuray* load_and_get_ineuray( const char* filename = 0, bool authenticate = true)
{
#ifdef MI_PLATFORM_WINDOWS
    if( !filename)
        filename = "libmdl_sdk.dll";
    void* handle = LoadLibraryA((LPSTR) filename);
    if( !handle) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        printf( "Failed to load library (%lu): " FMT_LPTSTR, error_code, message);
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
        printf( "GetProcAddress error (%lu): " FMT_LPTSTR, error_code, message);
        if( buffer)
            LocalFree( buffer);
        return 0;
    }
#else // MI_PLATFORM_WINDOWS
    if( !filename)
        filename = "libmdl_sdk.so";
    void* handle = dlopen( filename, RTLD_LAZY);
    if( !handle) {
        printf( "%s\n", dlerror());
        return 0;
    }
    void* symbol = dlsym( handle, "mi_factory");
    if( !symbol) {
        printf( "%s\n", dlerror());
        return 0;
    }
#endif // MI_PLATFORM_WINDOWS
    g_dso_handle = handle;
    mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>( symbol);
    if( !neuray)
        return 0;


    return neuray;
}

// Unloads the neuray library.
bool unload()
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
        printf( "Failed to unload library (%lu): " FMT_LPTSTR, error_code, message);
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

// Type constants to avoid overload ambiguities with 0 and nullptr.
const char* zero_string = 0;
const mi::Size zero_size = 0;

// Constant to avoid writing down the static cast.
const mi::Size minus_one_size = static_cast<mi::Size>( -1);

#ifdef MI_PLATFORM_WINDOWS
const char dir_sep = '\\';
#else // MI_PLATFORM_WINDOWS
const char dir_sep = '/';
#endif // MI_PLATFORM_WINDOWS


const char* plugin_path_dds           = "dds"             MI_BASE_DLL_FILE_EXT;
const char* plugin_path_mdl_distiller = "mdl_distiller"   MI_BASE_DLL_FILE_EXT;
const char* plugin_path_openimageio   = "nv_openimageio"  MI_BASE_DLL_FILE_EXT;


#endif // PROD_LIB_NEURAY_TEST_SHARED_H

