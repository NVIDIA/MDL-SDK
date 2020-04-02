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
// Generates a valid identifier from the input.

#include <string>
#include <iostream>
#include <set>
#include <iomanip>

#include "example_shared.h"

/// Checks, if the given character is a valid MDL letter.
bool is_mdl_letter(char c) 
{
    if ('A' <= c && c <= 'Z')
        return true;
    if ('a' <= c && c <= 'z')
        return true;
    return false;
}

/// Checks, if the given character is a valid MDL digit.
bool is_mdl_digit(char c)
{
    if ('0' <= c && c <= '9')
        return true;
    return false;
}

/// Checks, if the given identifier is an MDL keyword.
bool is_mdl_keyword(std::string& ident) {

    if (ident.empty())
        return false;

    static std::set <std::string> keywords = {
        // keywords
        "annotation", "bool", "bool2", "bool3", "bool4", "break", "bsdf", "bsdf_measurement",
        "case", "color", "const", "continue", "default", "do", "double", "double2",
        "double2x2", "double2x3", "double3", "double3x2", "double3x3", "double3x4", "double4",
        "double4x3", "double4x4", "double4x2", "double2x4", "edf", "else", "enum", "export",
        "false", "float", "float2", "float2x2", "float2x3", "float3", "float3x2", "float3x3",
        "float3x4", "float4", "float4x3", "float4x4", "float4x2", "float2x4", "for", "if",
        "import", "in", "int", "int2", "int3", "int4", "intensity_mode", "intensity_power",
        "intensity_radiant", "_exitance", "let", "light_profile", "material", "material_emission",
        "material_geometry", "material_surface", "material_volume", "mdl", "module", "package",
        "return", "string", "struct", "switch", "texture_2d", "texture_3d", "texture_cube",
        "texture_ptex", "true", "typedef", "uniform", "using", "varying", "vdf", "while",
        // reserved for future use
        "auto", "catch", "char", "class", "const_cast", "delete", "dynamic_cast", "explicit",
        "extern", "external", "foreach", "friend", "goto", "graph", "half", "half2", "half2x2",
        "half2x3", "half3", "half3x2", "half3x3", "half3x4", "half4", "half4x3", "half4x4",
        "half4x2", "half2x4", "inline", "inout", "lambda", "long", "mutable", "namespace",
        "native", "new", "operator", "out", "phenomenon", "private", "protected", "public",
        "reinterpret_cast", "sampler", "shader", "short", "signed", "sizeof", "static",
        "static_cast", "technique", "template", "this", "throw", "try", "typeid", "typename",
        "union", "unsigned", "virtual", "void", "volatile", "wchar_t" };

    return keywords.find(ident) !=  keywords.end();
}

/// Converts the given string into a valid mdl identifier.
std::string make_valid_mdl_identifier(
    const std::string& id)
{
    if (id.empty())
        return "m";

    std::string result;
    result.reserve(id.size());

    // first check general identifier rules:
    // IDENT = LETTER { LETTER | DIGIT | '_' } .

    size_t index = 0;
    result.push_back(is_mdl_letter(id[index]) ? id[index] : 'm');

    for (index = 1; index < id.size(); ++index) {
        const char c = id[index];
        if (is_mdl_digit(c) || is_mdl_letter(c) || c == '_')
            result.push_back(c);
        else {
            if (result[result.size()-1] != '_')
                result.push_back('_');
        }
    }

    // check, if identifier is mdl keyword
    if (is_mdl_keyword(result))
        return "m_" + result;
    else
        return result;
}


int MAIN_UTF8(int argc, char* argv[])
{
    if (argc < 2)
        std::cout << "Usage: " << argv[0] <<
        " <identifier_1> [<identifier_2> ...<identifier_n>]" << std::endl;

    for (int i = 1; i < argc; ++i) {

        std::cout << std::left << std::setw(25) << argv[i] << " --> " << make_valid_mdl_identifier(argv[i]) << std::endl;
    }
    std::cout << std::endl;

    return 0;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
