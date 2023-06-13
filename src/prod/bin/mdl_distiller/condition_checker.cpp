/******************************************************************************
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \file condition_checker.cpp

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <mi/mdl_sdk.h>

#include "condition_checker.h"
#include "options.h"
#include "mdl_distiller_utils.h"

bool check_distilled_material(const char *file, const char *distill_marker)
{

    // no distill target check needed
    if (std::string(distill_marker).find("any") == 0)
        return true;

    // trim spaces from marker
    std::string buffer(distill_marker);
    std::string::iterator newEnd = std::remove_if(buffer.begin(), buffer.end(), isspace);
    buffer = std::string(buffer.begin(), newEnd);
    distill_marker = buffer.c_str();

    FILE *fp = fopen(file, "r");
    if (fp == NULL)
        return false;

    const mi::Sint32 BUFSIZE = 1024;
    char  line[BUFSIZE];
    bool target_found = false;
    while ((fgets(line, BUFSIZE, fp) != 0) && !target_found) {
        std::string line_str(line);
        // find target material
        size_t found = line_str.find(distill_marker);
        if (found == std::string::npos)
            continue;
        target_found = true;
    }
    fclose(fp);
    return target_found;
}

std::string get_distill_marker(const char *path, const char *material_name)
{
    std::string module = get_module_from_qualified_name(material_name);
    std::stringstream full_name;

    // retrieve mdl file name
    std::string mdl_name(module);
    size_t found = mdl_name.find_first_of("::");
    if (found < mdl_name.length() - 2)
        mdl_name = mdl_name.substr(found + 2, mdl_name.length());
    full_name << path << SLASH << mdl_name << ".mdl";

    // extract material name
    std::string clear_name(material_name);
    std::size_t mat_begin = clear_name.find_last_of("::");
    clear_name = clear_name.substr(mat_begin + 1, clear_name.length());

    std::string distill_target("");
    FILE *fp = fopen(full_name.str().c_str(), "r");
    if (fp == NULL)
        return distill_target;

    const mi::Sint32 BUFSIZE = 1024;
    char  line[BUFSIZE];
    bool target_found = false;
    while ((fgets(line, BUFSIZE, fp) != 0) && !target_found) {
        std::string line_str(line);
        // find material
        size_t found = line_str.find(clear_name);

        if (found == std::string::npos)
            continue;

        // check if it is no prefix = a space or a bracket has to follow
        std::string p = line_str.substr(found + clear_name.length(), 1);
        if ((p.compare(" ") != 0) && (p.compare("(") != 0))
            continue;

        line_str = line_str.substr(found, line_str.length());

        // find distill target
        found = line_str.find("--->");
        if (found == std::string::npos)
            return distill_target;
        line_str = line_str.substr(found, line_str.length());

        std::size_t target_begin = line_str.find_first_of("::");
        if (target_begin == std::string::npos){
            std::size_t any_begin = line_str.find_first_of("---> any");
            if (any_begin == 0) {
                // No postcondition necessary
                return "any";
            }
            else{
                // Necessary postcondition not found -> return empty string
                return distill_target;
            }
        }
        distill_target = line_str.substr(target_begin, line_str.length() - 2);
        target_found = true;
    }

    fclose(fp);

    // remove line end
    size_t f = distill_target.find('\n');
    if (f != std::string::npos)
        distill_target = distill_target.substr(0, f);

    return distill_target;
}

void init_ruid_file(std::string path)
{
    std::stringstream file;
    file << path << SLASH <<RUID_FILE;
    // Empty old RUID file if exists
    std::ofstream ruid_file(file.str().c_str());
    ruid_file.flush();
    ruid_file.close();
}
