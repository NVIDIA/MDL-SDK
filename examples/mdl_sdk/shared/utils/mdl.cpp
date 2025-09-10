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

#include "utils/mdl.h"

#include "utils/io.h"
#include "utils/os.h"

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

// Indicates whether that directory contains nvidia/core_definitions.mdl and
// nvidia/axf_importer/axf_importer.mdl.
bool is_src_shaders_mdl(const std::string& path)
{
    std::string nvidia_dir = path + io::sep() + "nvidia" + io::sep();
    std::string file1 = nvidia_dir + "core_definitions.mdl";
    if (!io::file_exists(file1))
        return false;

    std::string file2 = nvidia_dir + "axf_importer" + io::sep() + "axf_importer.mdl";
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

// --------------------------------------------------------------------------------------------

class Default_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity /*level*/,
        const char* /*module_category*/,
        const mi::base::Message_details& /*details*/,
        const char* message) override
    {
        fprintf(stderr, "%s\n", message);
#ifdef MI_PLATFORM_WINDOWS
        fflush(stderr);
#endif
    }

    void message(
        mi::base::Message_severity level,
        const char* module_category,
        const char* message) override
    {
        this->message(level, module_category, mi::base::Message_details(), message);
    }

};

// --------------------------------------------------------------------------------------------

bool configure(
    mi::neuraylib::INeuray* neuray,
    Configure_options options)
{
    if (!neuray)
    {
        fprintf(stderr, "INeuray is invalid. Loading the SDK probably failed before.");
        return false;
    }

    mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_config(
        neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    // set user defined or default logger
    if (options.logger)
    {
        logging_config->set_receiving_logger(options.logger);
    }
    else
    {
        logging_config->set_receiving_logger(mi::base::make_handle(new Default_logger()).get());
    }
    g_logger = logging_config->get_forwarding_logger();

    // collect the search paths to add
    std::vector<std::string> mdl_paths(options.additional_mdl_paths);

    if (options.add_example_search_path)
    {
        const std::string example_search_path1 = mi::examples::mdl::get_examples_root() + "/mdl";
        if (example_search_path1 == "./mdl")
        {
            fprintf(stderr,
                "MDL Examples path was not found, "
                "consider setting the environment variable MDL_SAMPLES_ROOT.");
        }
        mdl_paths.push_back(example_search_path1);

        const std::string example_search_path2 = mi::examples::mdl::get_src_shaders_mdl();
        if (example_search_path2 != ".")
            mdl_paths.push_back(example_search_path2);
    }

    // add the search paths for MDL module and resource resolution outside of MDL modules
    for (size_t i = 0, n = mdl_paths.size(); i < n; ++i) {
        if (mdl_config->add_mdl_path(mdl_paths[i].c_str()) != 0 ||
                mdl_config->add_resource_path(mdl_paths[i].c_str()) != 0) {
            fprintf(stderr,
                "Warning: Failed to set MDL path \"%s\".\n",
                mdl_paths[i].c_str());
        }
    }

    // add user and system search paths with lowest priority
    if (options.add_user_space_search_paths)
    {
        mdl_config->add_mdl_user_paths();
    }
    if (options.add_admin_space_search_paths)
    {
        mdl_config->add_mdl_system_paths();
    }

    // load plugins if not skipped
    if (options.skip_loading_plugins)
        return true;

    if (load_plugin(neuray, "nv_openimageio" MI_BASE_DLL_FILE_EXT) != 0)
    {
        fprintf(stderr, "Fatal: Failed to load the nv_openimageio plugin.\n");
        return false;
    }

    if (load_plugin(neuray, "dds" MI_BASE_DLL_FILE_EXT) != 0)
    {
        fprintf(stderr, "Fatal: Failed to load the dds plugin.\n");
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------------------------

bool parse_cmd_argument_material_name(
    const std::string& argument,
    std::string& out_module_name,
    std::string& out_material_name,
    bool prepend_colons_if_missing)
{
    out_module_name = "";
    out_material_name = "";
    std::size_t p_left_paren = argument.rfind('(');
    if (p_left_paren == std::string::npos)
        p_left_paren = argument.size();
    std::size_t p_last = argument.rfind("::", p_left_paren-1);

    bool starts_with_colons = argument.length() > 2 && argument[0] == ':' && argument[1] == ':';

    // check for mdle
    if (!starts_with_colons)
    {
        std::string potential_path = argument;
        std::string potential_material_name = "main";

        // input already has ::main attached (optional)
        if (p_last != std::string::npos)
        {
            potential_path = argument.substr(0, p_last);
            potential_material_name = argument.substr(p_last + 2, argument.size() - p_last);
        }

        // is it an mdle?
        if (mi::examples::strings::ends_with(potential_path, ".mdle"))
        {
            if (potential_material_name != "main")
            {
                fprintf(stderr, "Error: Material and module name cannot be extracted from "
                    "'%s'.\nThe module was detected as MDLE but the selected material is "
                    "different from 'main'.\n", argument.c_str());
                return false;
            }
            out_module_name = potential_path;
            out_material_name = potential_material_name;
            return true;
        }
    }

    if (p_last == std::string::npos ||
        p_last == 0 ||
        p_last == argument.length() - 2 ||
        (!starts_with_colons && !prepend_colons_if_missing))
    {
        fprintf(stderr, "Error: Material and module name cannot be extracted from '%s'.\n"
            "An absolute fully-qualified material name of form "
            "'[::<package>]::<module>::<material>' is expected.\n", argument.c_str());
        return false;
    }

    if (!starts_with_colons && prepend_colons_if_missing)
    {
        fprintf(stderr, "Warning: The provided argument '%s' is not an absolute fully-qualified"
            " material name, a leading '::' has been added.\n", argument.c_str());
        out_module_name = "::";
    }

    out_module_name.append(argument.substr(0, p_last));
    out_material_name = argument.substr(p_last + 2, argument.size() - p_last);
    return true;
}

// --------------------------------------------------------------------------------------------

std::string add_missing_material_signature(
    const mi::neuraylib::IModule* module,
    const std::string& material_name)
{
    // Return input if it already contains a signature.
    if (material_name.back() == ')')
        return material_name;

    mi::base::Handle<const mi::IArray> result(
        module->get_function_overloads(material_name.c_str()));
    if (!result || result->get_length() != 1)
        return std::string();

    mi::base::Handle<const mi::IString> overloads(
        result->get_element<mi::IString>(static_cast<mi::Size>(0)));
    return overloads->get_c_str();
}


}}}
