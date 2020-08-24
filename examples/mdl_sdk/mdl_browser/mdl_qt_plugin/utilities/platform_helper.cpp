/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <sys/types.h>
#include <sys/stat.h>
#include <utils/io.h>

#if defined(MI_PLATFORM_WINDOWS)
    #include <direct.h>
    #include <windows.h>
    #include <strsafe.h>
    #include <io.h>
    #include <fcntl.h>
    #include <ctime>
    #include <sys/stat.h>
    #include <Shlobj.h>
    #include <Knownfolders.h>
    #include <codecvt>
#else
    #include <cstdlib>
    #include <dlfcn.h>
    #include <unistd.h>
    #include <libgen.h>
    #if defined(MI_PLATFORM_LINUX)
        #include <linux/limits.h>
    #elif defined(MI_PLATFORM_MACOSX)
        #include <limits.h>
        #include <mach-o/dyld.h>
    #endif
#endif

time_t Platform_helper::get_file_change_time(const std::string& path)
{
#ifdef MI_PLATFORM_WINDOWS
    struct _stat64i32 result {};

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    std::wstring wpath = conv.from_bytes(path);

    if(_wstat(wpath.c_str(), &result) != 0)
#else
    struct stat result {};
    if(stat(path.c_str(), &result) != 0)
#endif
    {
        std::cerr << "last modified date of \"" << path.c_str() << "\" could not be read.\n";
        return 0;
    }

    return result.st_mtime;
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
