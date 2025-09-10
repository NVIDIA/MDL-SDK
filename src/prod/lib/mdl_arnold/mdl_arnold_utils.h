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

#ifndef MDL_ARNOLD_UTILS_H
#define MDL_ARNOLD_UTILS_H

#include <string>
#include <cstdio>
#include <cstdlib>
#include <set>

#include <ai.h>
#include <ai_msg.h>
#include <mi/mdl_sdk.h>

#include <example_shared.h>


#ifdef MI_PLATFORM_WINDOWS
#include <direct.h>
#include <mi/base/miwindows.h>
#else // MI_PLATFORM_WINDOWS
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#endif

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
inline mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename, void** dso_handle)
{
    *dso_handle = nullptr;
#ifdef MI_PLATFORM_WINDOWS
    void* handle = LoadLibraryA((LPSTR) filename);
    if (!handle) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        AiMsgError("[mdl] Failed to load library (%lu): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return 0;
    }
    void* symbol = GetProcAddress((HMODULE) handle, "mi_factory");
    if (!symbol) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        AiMsgError("[mdl] GetProcAddress error (%lu): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return 0;
    }
#else // MI_PLATFORM_WINDOWS
    void* handle = dlopen(filename, RTLD_LAZY);
    if (!handle) {
        AiMsgError("[mdl] %s\n", dlerror());
        return 0;
    }
    void* symbol = dlsym(handle, "mi_factory");
    if (!symbol) {
        AiMsgError("[mdl] %s\n", dlerror());
        return 0;
    }
#endif // MI_PLATFORM_WINDOWS
    *dso_handle = handle;

    mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
    if (!neuray)
    {
        mi::base::Handle<mi::neuraylib::IVersion> version(
            mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
        if (!version)
            AiMsgError("[mdl] Incompatible library.\n");
        else if (strcmp(version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING) == 0)
            AiMsgError("[mdl] Previous instance of the library was not released properly.\n");
        else
            AiMsgError("[mdl] Library version %s does not match header version %s.\n",
            version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
        return 0;
    }
    return neuray;
}

// Unloads the MDL SDK.
inline bool unload(void* dso_handle)
{
    if(!dso_handle)
        return true; // sdk wasn't loaded in the first place

#ifdef MI_PLATFORM_WINDOWS
    int result = FreeLibrary((HMODULE) dso_handle);
    if (result == 0) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        AiMsgError("[mdl] Failed to unload library (%lu): " FMT_LPTSTR, error_code, message);
        if (buffer)
            LocalFree(buffer);
        return false;
    }
    return true;
#else // MI_PLATFORM_WINDOWS
    int result = dlclose(dso_handle);
    if (result != 0) {
        AiMsgError("[mdl] %s\n", dlerror());
        return false;
    }
    return true;
#endif // MI_PLATFORM_WINDOWS
}


// some function exported by the current shared library
AI_EXPORT_LIB bool NodeLoader(int i, AtNodeLib* node);

// get the path of the shared library this function is compiled in
inline std::string get_current_binary_directory()
{
    std::string current_shared_library_path = "";

    // get the path of the current dll/so
#ifdef MI_PLATFORM_WINDOWS
    #ifdef UNICODE
        wchar_t buffer[MAX_PATH];
    #else
        char buffer[MAX_PATH];
    #endif
    HMODULE hm = NULL;

    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCTSTR) &NodeLoader, &hm) == 0 ||
        GetModuleFileName(hm, buffer, sizeof(buffer)) == 0)
    {
        return current_shared_library_path;
    }
    #ifdef UNICODE
        current_shared_library_path = mi::examples::strings::wstr_to_str(std::wstring(buffer));
    #else
        current_shared_library_path = std::string(buffer);
    #endif

#else // MI_PLATFORM_WINDOWS
     Dl_info info;
     if (dladdr((void*)"NodeLoader", &info))
     {
        current_shared_library_path = std::string(info.dli_fname);
     }
#endif // MI_PLATFORM_WINDOWS

    // get the (normalized) parent path
     current_shared_library_path = mi::examples::io::normalize(current_shared_library_path);
    size_t last_sep = current_shared_library_path.find_last_of('/');
    if ((last_sep != std::string::npos) && (last_sep + 1 < current_shared_library_path.size()))
        current_shared_library_path = current_shared_library_path.substr(0, last_sep + 1);
    return current_shared_library_path;
}

#endif
