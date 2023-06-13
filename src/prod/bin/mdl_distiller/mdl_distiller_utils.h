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
/// \file mdl_distiller_utils.h
/// \brief Utilities for the MDL distiller.

#pragma once

#include <mi/mdl_sdk.h>

#include <string>
#include <vector>

/// Type for baked paths: a pair of strings, <type to bake, path to
/// bake>
struct Bake_path {
    std::string type;
    std::string path;
    Bake_path() {}
    Bake_path( const std::string& t, const std::string& p) : type(t), path(p) {}
};

/// A vector of Bake_paths
typedef std::vector<Bake_path> Bake_paths;

/// An unary functor to find a path in a Bake_paths vector
struct Bake_path_cmp {
    const std::string& m_s;
    Bake_path_cmp( const std::string& s) : m_s(s) {}
    bool operator()( const Bake_path& path) const { return path.path == m_s; }
};

inline std::ostream& operator<<( std::ostream& out, const Bake_path& path) {
    out << "type: '" << path.type << "', path '" << path.path << "'";
    return out;
}


std::string get_module_from_qualified_name(std::string qualified_name);
void replace_os_separator_by_slash(std::string &file_name);
std::string strip_path(mi::neuraylib::INeuray* neuray, std::string file_name);
std::string drop_signature(std::string function_name);

/// Returns a filename incl. .png suffix suitable for baked texture
/// file names.
inline std::string baked_texture_file_name( const char* material_name,
                                            const std::string& path_prefix) {
    return std::string(material_name) + "." + path_prefix + ".png";
}

/// Returns true if the given bake path is known to contain a normal map
inline bool is_normal_map_path(const std::string& path_name) {
    return path_name == "geometry.normal";
}
