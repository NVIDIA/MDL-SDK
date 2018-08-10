/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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


#include "platform_helper.h"

// MDL SDK platform definitions
#include <mi/mdl_sdk.h>
#include <cstdio>
#include <iostream>
#include "string_helper.h"


#ifdef MI_PLATFORM_WINDOWS
    #include <direct.h>
    #include <windows.h>
    #include <strsafe.h>
    #include <io.h>
    #include <fcntl.h>
    #include <ctime>
    #include <sys/stat.h>
    #include <Shlobj.h>
    #include <Knownfolders.h>
#else
    #include <cstdlib>
    #include <dlfcn.h>
    #include <unistd.h>
    #include <libgen.h>
    #include <linux/limits.h>
#endif

std::string Platform_helper::get_working_directory()
{
    char current_path[FILENAME_MAX];
    #ifdef MI_PLATFORM_WINDOWS
        _getcwd(current_path, FILENAME_MAX);
    #else
        getcwd(current_path, FILENAME_MAX); // TODO
    #endif
    return current_path;
}

std::string Platform_helper::get_executable_directory()
{
    std::string path;

    #ifdef MI_PLATFORM_WINDOWS
        HMODULE hModule = GetModuleHandleW(NULL);
        WCHAR path_wchar[MAX_PATH];
        GetModuleFileNameW(hModule, path_wchar, MAX_PATH);
        std::wstring path_wstr(path_wchar);
        path = std::string(path_wstr.begin(), path_wstr.end());
        path = path.substr(0, path.find_last_of('\\'));
    # else
        char result[PATH_MAX];
        size_t count = readlink("/proc/self/exe", result, PATH_MAX);
        if (count != -1)
            path = dirname(result);
    #endif
    return path;
}

time_t Platform_helper::get_file_change_time(const std::string& path)
{
    time_t changed = 0;

    #ifdef MI_PLATFORM_WINDOWS

    struct _stat64i32 buffer;
    int fh;
    _sopen_s(&fh, path.c_str(), _O_RDONLY, _SH_DENYNO, _S_IREAD);
    if (fh > 0)
    {
        if (_fstat(fh, &buffer) == 0)
        {
            changed = buffer.st_mtime;
        }
    }

    #endif

    return changed;
}

double Platform_helper::get_time()
{

    #ifndef MI_PLATFORM_WINDOWS
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec)*1.0e-6;
    #else
        static LARGE_INTEGER freq = {0};
        if (freq.QuadPart == 0)
            QueryPerformanceFrequency(&freq);
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return static_cast<double>(counter.QuadPart) / static_cast<double>(freq.QuadPart);
    #endif

}


double Platform_helper::tic_toc(const std::function<void()>& action)
{
    const double start = get_time();
    action();
    const double finish = get_time();
    return finish - start;
}

double Platform_helper::tic_toc_log(const std::string& name, const std::function<void()>& action)
{
    const double time = tic_toc(action);
    std::cerr << "[Timing] " << name << ": " << time << "s\n";
    return time;
}

void Platform_helper::keep_console_open()
{
    #ifdef MI_PLATFORM_WINDOWS
        if (IsDebuggerPresent())
        {
            fprintf(stderr, "Press enter to continue . . . \n");
            fgetc(stdin);
        }
    #endif
}

std::string Platform_helper::get_environment_variable(const std::string& env_var)
{
    std::string value = "";
    #ifdef MI_PLATFORM_WINDOWS
        char* buf = nullptr;
        size_t sz = 0;
        if (_dupenv_s(&buf, &sz, env_var.c_str()) == 0 && buf != nullptr)
        {
            value = buf;
            free(buf);
        }
    #else
        const char* v = getenv(env_var.c_str());
        if (v)
            value = v;
    #endif
    return value;
}

namespace {

    #ifdef MI_PLATFORM_WINDOWS
    //-----------------------------------------------------------------------------
    // helper function to create standard mdl path inside the known folder. WINDOWS only
    //
    std::string get_known_folder(const KNOWNFOLDERID& id, const std::string& postfix)
    {
        // Fetch the 'knownFolder' path.
        HRESULT hr = -1;
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
}

std::vector<std::string> Platform_helper::get_mdl_admin_space_directories()
{
    std::string paths = get_environment_variable("MDL_SYSTEM_PATH");
    if (!paths.empty())
    {
        std::vector<std::string> result = String_helper::split(paths, ';');
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

std::vector<std::string> Platform_helper::get_mdl_user_space_directories()
{
    std::string paths = get_environment_variable("MDL_USER_PATH");
    if (!paths.empty())
    {
        std::vector<std::string> result = String_helper::split(paths, ';');
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
