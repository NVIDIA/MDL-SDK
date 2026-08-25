/******************************************************************************
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/shared/example_shared.cpp
//
// Code shared by all examples

#include "example_shared.h"

void* g_dso_handle = 0;

char sep()
{
#ifdef MI_PLATFORM_WINDOWS
    return '\\';
#else
    return '/';
#endif
}

/// Indicates whether that directory has a mdl/nvidia/sdk_examples subdirectory.
bool is_examples_root(const std::string& path)
{
    std::string subdirectory = path + sep() + "mdl" + sep() + "nvidia" + sep() + "sdk_examples";
    return dir_exists(subdirectory.c_str());
}

/// Intentionally not implemented inline which would require callers to define MDL_EXAMPLES_ROOT.
std::string get_examples_root()
{
    std::string path = get_environment("MDL_EXAMPLES_ROOT");
    if (!path.empty())
        return normalize_path(path);


    path = get_executable_folder();
    while (!path.empty()) {
        if (is_examples_root(path))
            return normalize_path(path);
        std::string candidate = path + sep() + "share/mdl-sdk/examples";
        if (is_examples_root(candidate))
            return normalize_path(candidate);
        path = dirname(path);
    }

#ifdef MDL_EXAMPLES_ROOT
    path = MDL_EXAMPLES_ROOT;
    if (is_examples_root(path))
        return normalize_path(path);
#endif

    return ".";
}

/// Indicates whether that directory has a /nvidia subdirectory.
bool is_src_shaders_mdl(const std::string& path)
{
    std::string subdirectory = path + sep() + "nvidia";
    return dir_exists(subdirectory.c_str());
}

/// Intentionally not implemented inline which would require callers to define MDL_SRC_SHADERS_MDL.
std::string get_src_shaders_mdl()
{
    std::string path = get_environment("MDL_SRC_SHADERS_MDL");
    if (!path.empty())
        return normalize_path(path);

#ifdef MDL_SRC_SHADERS_MDL
    path = MDL_SRC_SHADERS_MDL;
    if (is_src_shaders_mdl(path))
        return normalize_path(path);
#endif

    return ".";
}

std::string get_executable_folder()
{
#ifdef MI_PLATFORM_WINDOWS
    char path[MAX_PATH];
    if (!GetModuleFileNameA(nullptr, path, MAX_PATH))
        return "";
#else  // MI_PLATFORM_WINDOWS
    char path[4096];
#ifdef MI_PLATFORM_MACOSX
    uint32_t buflen = sizeof(path);
    if (_NSGetExecutablePath(path, &buflen) != 0)
       return "";
#else  // MI_PLATFORM_MACOSX
    char proc_path[64];
    snprintf(proc_path, sizeof(proc_path), "/proc/%d/exe", getpid());

    ssize_t written = readlink(proc_path, path, sizeof(path));
    if (written < 0 || size_t(written) >= sizeof(path))
        return "";
    path[written] = 0;  // add terminating null
#endif // MI_PLATFORM_MACOSX
#endif // MI_PLATFORM_WINDOWS

    char* last_sep = strrchr(path, sep());
    if (!last_sep)
        return "";

    return normalize_path(std::string(path, last_sep));
}

std::string read_text_file(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
        fprintf(stderr, "Cannot open file : \"%s\".\n", filename.c_str());
        return "";
    }

    std::stringstream string_stream;
    string_stream << file.rdbuf();
    return string_stream.str();
}

std::string find_resource_file(const char* relative_directory, const char* shader_filename)
{
    // Location for build/install targets (next to executable).
    std::string executable_dirname = get_executable_folder();
    std::string path = executable_dirname + "/" + shader_filename;
    if (file_exists(path.c_str()))
        return normalize_path(path.c_str());

    // Location for examples as tools in vcpkg.
    std::string examples_root = get_examples_root();
    if (examples_root == ".")
        return {};
    path = examples_root + "/" + relative_directory + "/" + shader_filename;
    if (file_exists(path.c_str()))
        return normalize_path(path.c_str());

    return {};
}

std::string read_resource_file(const char* relative_directory, const char* shader_filename)
{
    std::string filename = find_resource_file(relative_directory, shader_filename);
    if (filename.empty())
        return {};

    return read_text_file(filename);
}
