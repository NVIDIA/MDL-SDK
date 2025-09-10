/***************************************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_python.h"

#include "utils/mdl.h"

// Avoid pulling in mdl.cpp which has more dependencies.
namespace mi { namespace examples { namespace mdl {

#ifdef MI_PLATFORM_WINDOWS
// Pointer to the DSO handle. Cached here for unload().
HMODULE g_dso_handle;
#else
// Pointer to the DSO handle. Cached here for unload().
void* g_dso_handle;
#endif

mi::base::Handle<mi::base::ILogger> g_logger;

} } } // mi::examples::mdl

mi::neuraylib::INeuray* g_neuray;

/// Indicates whether \p t is a suffix of \p s.
bool ends_with(const std::string& s, const std::string& t)
{
    if (t.size() > s.size())
        return false;
    return std::equal(t.rbegin(), t.rend(), s.rbegin());
}

mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename)
{
    if (!filename || filename[0] == '\0') {
        mi::neuraylib::INeuray* result = mi::examples::mdl::load_and_get_ineuray();
        g_neuray = result;
        return result;
    }

    std::string lib = filename;
    if (!ends_with(lib, MI_BASE_DLL_FILE_EXT))
        lib += MI_BASE_DLL_FILE_EXT;

    mi::neuraylib::INeuray* result = mi::examples::mdl::load_and_get_ineuray(lib.c_str());
    g_neuray = result;
    return result;
}

bool load_plugin(mi::neuraylib::INeuray* neuray, const char* filename)
{
    if (!neuray || !filename)
        return false;

    std::string lib = filename;
    if (!ends_with(lib, MI_BASE_DLL_FILE_EXT))
        lib += MI_BASE_DLL_FILE_EXT;

    return mi::examples::mdl::load_plugin(neuray, lib.c_str()) == 0;
}

bool unload()
{
    g_neuray = nullptr;
    return mi::examples::mdl::unload();
}
