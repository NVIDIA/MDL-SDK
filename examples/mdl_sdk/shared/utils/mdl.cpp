/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/shared/utils/mdl.cpp
//
// Code shared by all examples

#include "mdl.h"

#include "io.h"

namespace mi { namespace examples { namespace mdl {

// Indicates whether that directory has a mdl/nvidia/sdk_examples subdirectory.
bool is_examples_root(const std::string& path)
{
    std::string subdirectory = path + io::sep() + "mdl/nvidia/sdk_examples";
    return io::directory_exists(subdirectory);
}

// Intentionally not implemented inline which would require callers to define MDL_SAMPLES_ROOT.
std::string get_examples_root()
{
    std::string path = mi::examples::os::get_environment("MDL_SAMPLES_ROOT");
    if (!path.empty())
        return io::normalize(path);


    path = io::get_executable_folder();
    while (!path.empty()) {
        if (is_examples_root(path))
            return io::normalize(path);
        path = io::dirname(path);
    }

#ifdef MDL_SAMPLES_ROOT
    path = MDL_SAMPLES_ROOT;
    if (is_examples_root(path))
        return io::normalize(path);
#endif

    return ".";
}

// Indicates whether that directory contains nvidia/core_definitions.mdl and nvidia/axf_to_mdl.mdl.
bool is_src_shaders_mdl(const std::string& path)
{
    std::string file1 = path + io::sep() + "nvidia" + io::sep() + "core_definitions.mdl";
    if (!io::file_exists(file1))
        return false;

    std::string file2 = path + io::sep() + "nvidia" + io::sep() + "axf_to_mdl.mdl";
    if (!io::file_exists(file2))
        return false;

    return true;
}

// Intentionally not implemented inline which would require callers to define MDL_SRC_SHADERS_MDL.
std::string get_src_shaders_mdl()
{
    std::string path = mi::examples::os::get_environment("MDL_SRC_SHADERS_MDL");
    if (!path.empty())
        return io::normalize(path);

#ifdef MDL_SRC_SHADERS_MDL
    path = MDL_SRC_SHADERS_MDL;
    if (is_src_shaders_mdl(path))
        return io::normalize(path);
#endif

    return ".";
}

}}}
