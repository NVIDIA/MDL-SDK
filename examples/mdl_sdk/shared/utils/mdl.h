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

// examples/mdl_sdk/shared/utils/mdl.h
//
// Code shared by all examples

#ifndef EXAMPLE_SHARED_UTILS_MDL_H
#define EXAMPLE_SHARED_UTILS_MDL_H

#include <mi/mdl_sdk.h>

namespace mi { namespace examples { namespace mdl {

/// The forwarding logger.
extern mi::base::Handle<mi::base::ILogger> g_logger;

/// Returns the root directory of the examples.
///
/// The root directory of the examples is the one that contains the "mdl/nvidia/sdk_examples"
/// directory. The following steps are performed to find it:
/// - If the environment variable MDL_EXAMPLES_ROOT is set, it is returned (without checking for
///   the existence of the subdirectory mentioned above).
/// - Starting from the directory of the executable all parent directories are considered in
///   turn bottom-up, checked for the existence of the "mdl/nvidia/sdk_examples" and
///   "share/mdl-sdk/examples/mdl/nvidia/sdk_examples" (for tools in vcpkg) subdirectories, and the
///   first successful directory is returned.
/// - If that subdirectory of the source tree exists, it is returned.
/// - Finally, the current working directory is returned (as ".").
std::string get_examples_root();

/// Returns a directory that contains ::nvidia::core_definitions and ::nvida::axf_to_mdl.
///
/// Might also return "." if that directory is the "mdl" subdirectory of #get_examples_root()
/// and no extra handling is required.
///
/// The following steps are performed to find it:
/// - If the environment variable MDL_SRC_SHADERS_MDL is set, it is returned (without checking
//    for the existence of the MDL modules mentioned above).
/// - If that subdirectory of the source tree exists, it is returned.
/// - Finally, the current working directory is returned (as ".").
std::string get_src_shaders_mdl();

/// Input to the \c configure function. Allows to control the search path setup for the examples
/// as well as to control the loaded plugins.
struct Configure_options
{
    /// additional search paths that are added after admin/user and the example search paths
    std::vector<std::string> additional_mdl_paths;

    /// set to false to not add the admin space search paths. It's recommend to leave this true.
    bool add_admin_space_search_paths = true;

    /// set to false to not add the user space search paths. It's recommend to leave this true.
    bool add_user_space_search_paths = true;

    /// set to false to not add the example content mdl path
    bool add_example_search_path = true;

    /// set to true to disable (optional) plugin loading
    bool skip_loading_plugins = false;

    /// if true, render on one thread only
    bool single_threaded = false;

    /// The desired log level of the logger.
    ///
    /// Any messages below the given level will be discarded.
    mi::base::Message_severity log_level = mi::base::MESSAGE_SEVERITY_INFO;
};

/// Configures the MDL SDK by installing a default logger, setting the default MDL search path,
/// and loading the OpenImageIO and DDS image plugins. This done by many examples so it makes
/// sense to bundle this here in one place and focus on the actual example.
///
/// \param neuray                   pointer to the main MDL SDK interface
/// \param options                  see \Configure_options fields
bool configure(
    mi::neuraylib::INeuray* neuray,
    Configure_options options = Configure_options());

/// Many examples accept material names as command line arguments.
/// In general, the expected input is fully-qualified absolute MDL material name of the form:
/// [::<package>]::<module>::<material>
/// This function splits this input into a module and the material name.
/// Note, this is not working for function names.
///
/// \param argument                     input, a fully-qualified absolute MDL material name
/// \param[out] module_name             a fully-qualified absolute MDL module name
/// \param[out] out_material_name       the materials simple name
/// \param prepend_colons_if_missing    prepend "::" for non-empty module names, if missing
bool parse_cmd_argument_material_name(
    const std::string& argument,
    std::string& out_module_name,
    std::string& out_material_name,
    bool prepend_colons_if_missing = true);

/// Adds a missing signature to a material name.
///
/// Specifying material signatures on the command-line can be tedious. Hence, this convenience
/// method is used to add the missing signature. Since there are no overloads for materials, we
/// can simply search the module for the given material -- or simpler, let the overload
/// resolution handle that.
///
/// \param module                       the module containing the material
/// \param material_name                the DB name of the material without signature
/// \return                             the DB name of the material including signature, or the
///                                     empty string in case of errors.
std::string add_missing_material_signature(
    const mi::neuraylib::IModule* module,
    const std::string& material_name);

/// Finds the location of a resource directory.
///
/// Considers the directory of the executable first, then the one indicated by
/// \p relative_directory.
///
/// \param relative_directory   The location relative to the root directory of the examples.
/// \param resource_dirname     The name of the resource directory.
/// \return                     The found location, or the empty string in case of failure.
std::string find_resource_directory(
    const char* relative_directory, const char* resource_dirname);

/// Finds the location of a resource file.
///
/// Considers the directory of the executable first, then the one indicated by
/// \p relative_directory.
///
/// \param relative_directory   The location relative to the root directory of the examples.
/// \param resource_filename    The name of the resource file.
/// \return                     The found location, or the empty string in case of failure.
std::string find_resource_file(
    const char* relative_directory, const char* resource_filename);

/// Reads the contents of a resource file.
///
/// Considers the directory of the executable first, then the one indicated by
/// \p relative_directory.
///
/// \param relative_directory   The location relative to the root directory of the examples.
/// \param resource_filename    The name of the resource file.
/// \return                     The found location, or the empty string in case of failure.
std::string read_resource_file(const char* relative_directory, const char* resource_filename);

}}}

#endif

