/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

namespace mi { namespace examples { namespace mdl {

// Intentionally not implemented inline which would require callers to define MDL_SAMPLES_ROOT.
std::string get_examples_root()
{
    std::string path = mi::examples::os::get_environment("MDL_SAMPLES_ROOT");
    if (path.empty())
    {
#ifdef MDL_SAMPLES_ROOT
        path = MDL_SAMPLES_ROOT;
#else
        path = ".";
#endif
    }
    if (!mi::examples::io::directory_exists(path))
        return ".";

    // normalize the paths
    return mi::examples::io::normalize(path);
}

// Intentionally not implemented inline which would require callers to define MDL_SRC_SHADERS_MDL.
std::string get_src_shaders_mdl()
{
    std::string path = mi::examples::os::get_environment("MDL_SRC_SHADERS_MDL");
    if (path.empty())
    {
#ifdef MDL_SRC_SHADERS_MDL
        path = MDL_SRC_SHADERS_MDL;
#else
        path = ".";
#endif
    }
    if (!mi::examples::io::directory_exists(path))
        return ".";

    // normalize the paths
    return mi::examples::io::normalize(path);
}

}}}
