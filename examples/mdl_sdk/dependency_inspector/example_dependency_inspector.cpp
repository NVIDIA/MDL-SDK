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

// examples/mdl_sdk/dependency_inspector/example_dependency_inspector.cpp
//
// Loads an MDL module and finds all paths of its import and resource dependencies.

#include <iostream>
#include <string>
#include <unordered_set>

#include "example_shared.h"

// Dependency options. Select the dependency types to be listed.
enum Filter_options
{
    DEPENDENCY_NOTHING      = 0x00,
    DEPENDENCY_IMPORTS      = 0x01,
    DEPENDENCY_RESOURCES    = 0x02,
    DEPENDENCY_ALL          = 0xFF
};

/// The options parsed from the command-line.
class Options
{
public:
    /// Prints the usage information to the stream \p s.
    static void print_usage(std::ostream& s);

    /// Parses the command-line given by \p argc and \p argv.
    ///
    /// \return   \c true in case of success and \c false otherwise.
    bool parse(int argc, char* argv[]);

    ///  Qualified module name.
    std::string m_qualified_module_name = "::nvidia::sdk_examples::tutorials";

    /// Listing options.
    int m_dependency_options = DEPENDENCY_ALL;
    bool m_recursive = false; // scan modules recursively.
    
    /// General options.
    bool m_nostdpath = false;
    std::vector<std::string> m_mdl_paths;
};

// This helper class contains some methods to scan and dump the dependencies of an MDL module
// and its imports (recursively). It keeps track of modules already visited or file paths
// already listed in order to avoid repeated outputs.
class Dependency_inspector
{
public:
    // Dependency_inspector constructor.
    Dependency_inspector(mi::neuraylib::INeuray* neuray, const Options& options);

    // The entry point for scanning a MDL module.
    void scan_module(const std::string& qualified_module_name);

private:
    // List the dependencies of MDL module and optionally from its imports (recursively).
    void list_dependencies( const mi::neuraylib::IModule* module);
    
    // Check if the filename contains embedded resources notation
    // from MDR or MDLE modules and cuts it away.
    const std::string get_resolved_filename(const char* filename);

    // Check if a filename was already listed.
    bool was_listed(const std::string& filename);

    mi::neuraylib::INeuray* m_neuray;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    
    Options m_options;

    // List of modules already visited.
    std::unordered_set<std::string> m_module_list;
    // List of filenames already listed.
    std::unordered_set<std::string> m_filename_list;
};

// Dependency_inspector constructor.
Dependency_inspector::Dependency_inspector(mi::neuraylib::INeuray* neuray, const Options& options)
: m_neuray(neuray)
, m_options(options)
{
}

// The entry point for scanning a MDL module.
void Dependency_inspector::scan_module(const std::string& qualified_module_name)
{
    // Access the database and scope.
    mi::base::Handle<mi::neuraylib::IDatabase>
        database(m_neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope>
        scope(database->get_global_scope());

    // Create execution context.
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory>
        mdl_factory(m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle <mi::neuraylib::IMdl_execution_context>
        context(mdl_factory->create_execution_context());

    // Create a transaction.
    m_transaction = scope->create_transaction();

    {
        // Load the module.
        check_success(mdl_impexp_api->load_module(
            m_transaction.get(), qualified_module_name.c_str(), context.get()) >= 0);
        print_messages(context.get());

        // Get module database name.
        mi::base::Handle<const mi::IString> module_db_name(
            mdl_factory->get_db_module_name(qualified_module_name.c_str()));

        mi::base::Handle<const mi::neuraylib::IModule> module(
            m_transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

        mi::base::Handle<const mi::IString>
            decoded_name(mdl_factory->decode_name(module->get_mdl_name()));
        std::cout << std::endl <<
            "Dependencies founds in module " << decoded_name->get_c_str() << std::endl;

        list_dependencies(module.get());
    }

    // All transactions need to get committed.
    m_transaction->commit();
}

// List dependencies of an MDL module and optionally from its imports (recursively).
void Dependency_inspector::list_dependencies( const mi::neuraylib::IModule* module)
{
    // List imports
    if (m_options.m_dependency_options & DEPENDENCY_IMPORTS)
    {
        const char* filename = module->get_filename();
        if(filename && !was_listed(filename))
            std::cout << filename << std::endl;
    }

    // List resources
    if (m_options.m_dependency_options & DEPENDENCY_RESOURCES)
    {
        // List resources referenced by this module.
        const mi::Size resources_count = module->get_resources_count();
        for (mi::Size r = 0; r < resources_count; ++r)
        {
            mi::base::Handle<const mi::neuraylib::IValue_resource>
                resource(module->get_resource(r));
            const char* db_name = resource->get_value();
            const mi::base::Handle<const mi::neuraylib::IType_resource>
                type(resource->get_type());

            switch (type->get_kind())
            {
            case mi::neuraylib::IType::TK_TEXTURE:
            {
                const mi::base::Handle<const mi::neuraylib::ITexture>
                    texture(m_transaction->access<mi::neuraylib::ITexture>(db_name));
                const mi::base::Handle<const mi::neuraylib::IImage>
                    image(m_transaction->access<mi::neuraylib::IImage>(texture->get_image()));

                for (mi::Size f = 0, fn = image->get_length(); f < fn; ++f)
                    for (mi::Size t = 0, tn = image->get_frame_length(f); t < tn; ++t)
                    {
                        const std::string resolved_filename = 
                            get_resolved_filename(image->get_filename(f, t));
                        if (!resolved_filename.empty() && !was_listed(resolved_filename))
                            std::cout << resolved_filename << std::endl;
                    }
                break;
            }

            case mi::neuraylib::IType::TK_LIGHT_PROFILE:
            {
                const mi::base::Handle<const mi::neuraylib::ILightprofile> light_profile(
                    m_transaction->access<mi::neuraylib::ILightprofile>(db_name));
                const std::string resolved_filename =
                    get_resolved_filename(light_profile->get_filename());
                if (!resolved_filename.empty() && !was_listed(resolved_filename))
                    std::cout << resolved_filename << std::endl;
                break;
            }

            case mi::neuraylib::IType::TK_BSDF_MEASUREMENT:
            {
                const mi::base::Handle<const mi::neuraylib::IBsdf_measurement> mbsdf(
                    m_transaction->access<mi::neuraylib::IBsdf_measurement>(db_name));
                const std::string resolved_filename =
                    get_resolved_filename(mbsdf->get_filename());
                if (!resolved_filename.empty() && !was_listed(resolved_filename))
                    std::cout << resolved_filename << std::endl;
                break;
            }

            default:
                exit_failure(std::string("Unexpected Resource type: " +
                    std::to_string(type->get_kind())).c_str());
                break;
            }
        }

        // List thumbnails referenced by function definitions (if any).
        const mi::Size function_count = module->get_function_count();
        for (mi::Size i = 0; i < function_count; i++)
        {
            const std::string function_name(module->get_function(i));
            mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
                m_transaction->access<mi::neuraylib::IFunction_definition>(function_name.c_str()));

            const std::string thumbnail =
                get_resolved_filename(function_definition->get_thumbnail());
            if (!thumbnail.empty() && !was_listed(thumbnail))
                std::cout << thumbnail << std::endl;
        }

        // List thumbnails referenced by material definitions (if any).
        const mi::Size material_count = module->get_material_count();
        for (mi::Size i = 0; i < material_count; i++)
        {
            const std::string material_name(module->get_material(i));
            mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
                m_transaction->access<mi::neuraylib::IFunction_definition>(material_name.c_str()));

            const std::string thumbnail =
                get_resolved_filename(material_definition->get_thumbnail());
            if (!thumbnail.empty() && !was_listed(thumbnail))
                std::cout << thumbnail << std::endl;
        }
    }

    // Inspect (optionally) imported modules recursively.
    if (m_options.m_recursive)
    {
        const mi::Size module_count = module->get_import_count();
        for (mi::Size i = 0; i < module_count; i++)
        {
            const std::string db_name(module->get_import(i));

            // Skip modules already visited.
            auto it = std::find(m_module_list.begin(), m_module_list.end(), db_name);
            if (it != m_module_list.end())
                continue;

            // Update module list.
            m_module_list.insert(db_name);

            // Access and list imported module.
            {
                mi::base::Handle<const mi::neuraylib::IModule> imported_module(
                    m_transaction->access<mi::neuraylib::IModule>(db_name.c_str()));

                if (imported_module->is_standard_module())
                    continue;

                list_dependencies(imported_module.get());
            }
        }
    }
}

// Check if the filename contains embedded resources notation
// from MDR or MDLE modules and cuts it away.
const std::string Dependency_inspector::get_resolved_filename(const char* filename)
{
    if (!filename)
        return std::string("");

    std::string resolved_filename(filename);

    // Find positions of ".mdr" and ".mdle" if any.
    size_t pos_mdr = resolved_filename.find(".mdr");
    size_t pos_mdle = resolved_filename.find(".mdle");

    // Erase everything from its position onwards
    if (pos_mdr != std::string::npos)
        resolved_filename = resolved_filename.substr(0, pos_mdr + 4); // Include ".mdr"
    else if (pos_mdle != std::string::npos)
        resolved_filename = resolved_filename.substr(0, pos_mdle + 5); // Include ".mdle"

    return resolved_filename;
}

// Check if a filename was already listed.
bool Dependency_inspector::was_listed(const std::string& filename)
{
    // Skip filenames already listed.
    auto it = std::find(m_filename_list.begin(), m_filename_list.end(), filename);
    if (it != m_filename_list.end())
        return true;

    // Update filenames
    m_filename_list.insert(filename);
    return false;
}

// Print command line usage.
void Options::print_usage(std::ostream& s)
{
    s << R"(
code_gen [options] <qualified_module_name>

options:

  -h|--help                 Print this usage message and exit.
  -p|--mdl_path <path>      Add the given path to the MDL search path.
  -n|--nostdpath            Prevent adding the MDL system and user search
                            path(s) to the MDL search path.
  -d|--dependencies         Select type of dependencies to be listed.
                              Format: -d "type1|type2|...|typeN".
                              Types: 'imports', 'resources', 'all'.
                              Default: 'all'.
  -r|--recursive            Scan imported modules recursively)";

    s << std::endl << std::endl;
}

// Parse command line options.
bool Options::parse(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg[0] == '-')
        {
            if (arg == "-h" || arg == "--help")
            {
                return false;
            }
            else if (arg == "-n" || arg == "--nostdpath")
            {
                m_nostdpath = true;
            }
            else if (arg == "-p" || arg == "--mdl_path")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -p|--mdl_path missing." << std::endl;
                    return false;
                }
                m_mdl_paths.push_back(argv[++i]);
            }
            else if (arg == "-d" || arg == "--dependencies")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -d|--dependencies missing." << std::endl;
                    return false;
                }

                std::istringstream ss(argv[++i]);
                std::string type;

                m_dependency_options = DEPENDENCY_NOTHING;

                while (std::getline(ss, type, '|'))
                {
                    if (type == "imports")
                    {
                        m_dependency_options |= DEPENDENCY_IMPORTS;
                    }
                    else if (type == "resources")
                    {
                        m_dependency_options |= DEPENDENCY_RESOURCES;
                    }
                    else if (type == "all")
                    {
                        m_dependency_options |= DEPENDENCY_ALL;
                    }
                    else
                    {
                        std::cerr << "error: Unknown filter type: " << type << "." << std::endl;
                        return false;
                    }
                }
            }
            else if (arg == "-r" || arg == "--recursive")
            {
                m_recursive = true;
            }
            else
            {
                std::cerr << "error: Unknown option \"" << arg << "\"." << std::endl;
                return false;
            }
        }
        else
        {
            if (i == argc - 1)
                m_qualified_module_name = arg;
        }
    }

    return true;
}

// Main function.
int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line options.
    Options options;
    if (!options.parse(argc, argv))
    {
        options.print_usage(std::cout);
        exit_failure("Failed to parse command line arguments.");
    }

    // Access the MDL SDK.
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK.
    mi::examples::mdl::Configure_options configure_options;

    // Apply the search path setup described on the command line.
    configure_options.additional_mdl_paths = options.m_mdl_paths;

    if (options.m_nostdpath)
    {
        configure_options.add_admin_space_search_paths = false;
        configure_options.add_user_space_search_paths = false;
        configure_options.add_example_search_path = false;
    }

    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK.
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    // Load an MDL module and list its dependencies.
    {
        Dependency_inspector dependency_inspector(neuray.get(), options);
        dependency_inspector.scan_module(options.m_qualified_module_name);
    }

    // Shut down the MDL SDK.
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK.
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows.
COMMANDLINE_TO_UTF8
