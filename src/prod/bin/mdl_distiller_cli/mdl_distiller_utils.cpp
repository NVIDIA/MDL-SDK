/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \file mdl_distiller_utils.cpp
/// \brief Utilities for the MDL distiller.

#include "mdl_distiller_utils.h"

using mi::base::Handle;
using mi::IString;

using mi::neuraylib::INeuray;
using mi::neuraylib::IMdl_configuration;

std::string get_module_from_qualified_name(std::string qualified_name)
{
    std::string::size_type last_scope = qualified_name.rfind("::");
    if (last_scope == std::string::npos)
        return "";
    return std::string(qualified_name, 0, last_scope);
}

void replace_os_separator_by_slash(std::string &file_name)
{
#ifdef MI_PLATFORM_WINDOWS
    char sep = '\\';
#else
    char sep = '/';
#endif
    if (sep == '/')
        return;

    for (size_t i = 0, n = file_name.length(); i < n; ++i) {
        if (file_name[i] == sep)
            file_name[i] = '/';
    }
}

// Given a filename, remove a prefix that is equal to any MDL search path
// TODO: changes in the SDK will make this simpler in the future
std::string strip_path(INeuray* neuray, std::string file_name) {
    Handle<IMdl_configuration> mdl_config(neuray->get_api_component<IMdl_configuration>());
    mi::Size n = mdl_config->get_mdl_paths_length();
    for (mi::Size i = 0; i < n; ++i) {
        Handle<const IString> path(mdl_config->get_mdl_path(i));
        std::string::size_type prefix = file_name.find( path->get_c_str());
        if (prefix == 0) {
            file_name.erase(0, strlen(path->get_c_str()));
            break;
        }
    }
    replace_os_separator_by_slash(file_name);
    if (file_name.size() > 2 && file_name[0] == '/' && file_name[1] == '.') {
        file_name.erase( 0, 2);
    }
    return file_name;
}

// Remove function signature, i.e., everything after first parenthesis
std::string drop_signature(std::string function_name) {
    std::string::size_type sig = function_name.find("(");
    function_name.erase(sig);
    return function_name;
}
