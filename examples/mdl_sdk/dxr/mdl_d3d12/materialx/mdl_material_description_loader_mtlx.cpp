/******************************************************************************
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_material_description_loader_mtlx.h"

#include <algorithm>

#include <MaterialXCore/Unit.h>

#include "example_materialx_shared.h"

#include "../base_application.h"
#include "../mdl_sdk.h"

namespace mi {namespace examples { namespace mdl_d3d12 { namespace materialx
{

Mdl_material_description_loader_mtlx::Mdl_material_description_loader_mtlx(
    const Base_options& options)
    : m_options(options)
    , m_paths(options.mtlx_paths)
    , m_add_mtlx_std_path(!options.no_mtlx_std_path)
    , m_libraries(options.mtlx_libraries)
    , m_mdl_version(options.mtlx_to_mdl)
    , m_materialxtest_mode(options.materialxtest_mode)
{
    std::string version = MaterialX::getVersionString();
    log_info("Enable MaterialX loader using SDK version: " + version);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description_loader_mtlx::match_gltf_name(const std::string& gltf_name) const
{
    return mi::examples::materialx::is_materialx_resource(gltf_name);
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description_loader_mtlx::generate_mdl_source_code(
    Mdl_sdk& mdl_sdk,
    const std::string& gltf_name,
    const std::string& scene_directory,
    std::string& out_generated_mdl_code,
    std::string& out_generated_mdl_name) const
{
    mi::examples::materialx::Mdl_generator mtlx2mdl(MDL_EXAMPLE_RELATIVE_DIRECTORY);
    mi::examples::materialx::Mdl_generator_result result;

    // set the material file to load
    std::string mtlx_material_file = mi::examples::io::is_absolute_path(gltf_name)
        ? gltf_name
        : scene_directory + "/" + gltf_name;

    // parse the file name and optional query
    std::string query = mi::examples::strings::get_url_query(mtlx_material_file);
    std::string selected_material_name = "";
    if (!query.empty())
    {
        // drop the query from the file name
        size_t pos = mtlx_material_file.find_first_of('?');
        mtlx_material_file = mtlx_material_file.substr(0, pos);

        // parse the query
        auto query_map = mi::examples::strings::parse_url_query(query);
        const auto& it = query_map.find("name");
        if (it != query_map.end())
            selected_material_name = it->second;
    }

    // allow to configure MaterialX search and library paths by the user
    mtlx2mdl.set_add_std_path(m_add_mtlx_std_path);
    for (auto& p : m_paths)
        mtlx2mdl.add_path(p);

    for (auto& l : m_libraries)
        mtlx2mdl.add_library(l);

    // set the MDL language version
    if (m_mdl_version == "1.6")
        mtlx2mdl.set_mdl_version(mi::neuraylib::MDL_VERSION_1_6);
    else if (m_mdl_version == "1.7")
        mtlx2mdl.set_mdl_version(mi::neuraylib::MDL_VERSION_1_7);
    else if (m_mdl_version == "1.8")
        mtlx2mdl.set_mdl_version(mi::neuraylib::MDL_VERSION_1_8);
    else if (m_mdl_version == "1.9")
        mtlx2mdl.set_mdl_version(mi::neuraylib::MDL_VERSION_1_9);
    else if (m_mdl_version == "1.10")
        mtlx2mdl.set_mdl_version(mi::neuraylib::MDL_VERSION_1_10);
    else if (m_mdl_version == "latest")
        mtlx2mdl.set_mdl_version(mi::neuraylib::MDL_VERSION_LATEST);
    else
        log_warning("Ignoring unexpected MDL version \"" + m_mdl_version + "\"", SRC);

    // set MaterialXTest mode
    mtlx2mdl.set_materialxtest_mode(m_materialxtest_mode);

    // set the materials main source file
    bool valid = true;
    valid &= mtlx2mdl.set_source(mtlx_material_file, selected_material_name);

    // generate the mdl code
    try
    {
        valid &= mtlx2mdl.generate(&mdl_sdk.get_config(), result);
    }
    catch (const std::exception & ex)
    {
        log_error("Generated MDL from materialX crashed: " + gltf_name, ex, SRC);
        return false;
    }

    if (!valid)
    {
        log_error("Generated MDL from materialX is not valid: " + gltf_name, SRC);
        return false;
    }

    // return the first generated code segment.
    out_generated_mdl_code = result.generated_mdl_code;
    out_generated_mdl_name = result.generated_mdl_name;
    return true;
}

// ------------------------------------------------------------------------------------------------

size_t Mdl_material_description_loader_mtlx::get_file_type_count() const
{
    return 1;
}

// ------------------------------------------------------------------------------------------------

std::string Mdl_material_description_loader_mtlx::get_file_type_extension(size_t index) const
{
    switch (index)
    {
        case 0: return "mtlx";
        default: return "";
    }
}

// ------------------------------------------------------------------------------------------------
std::string Mdl_material_description_loader_mtlx::get_file_type_description(size_t index) const
{
    switch (index)
    {
        case 0: return "MaterialX";
        default: return "";
    }
}

}}}}
