/******************************************************************************
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/modules/example_modules.cpp
 //
 // Loads an MDL module and scan for its imports, resource dependencies and more...

#include <iostream>
#include <string>
#include <unordered_set>

#include "example_shared.h"

// Dump options. The type of information that can be selected to be dump from a module
enum Dump_options
{
    DUMP_NOTHING = 0x00,
    DUMP_IMPORTS = 0x01,
    DUMP_TYPES = 0x02,
    DUMP_CONSTANTS = 0x04,
    DUMP_DEFINITIONS = 0x08,
    DUMP_RESOURCES = 0x10,
    DUMP_ALL = 0xFF
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

    ///  Qualified module name
    std::string m_qualified_module_name = "::nvidia::sdk_examples::tutorials";

    /// Dumping options
    bool m_recursive = false; // scan modules recursively.
    bool m_skip_standard = false; // skip standard modules.
    int m_dump_options = DUMP_IMPORTS | DUMP_RESOURCES;

    // Dump function/material definitions containing the m_def_filter substring keyword.
    std::string m_def_filter = "";

    /// General options.
    bool m_nostdpath = false;
    std::vector<std::string> m_mdl_paths;
};

// This helper class contains some methods to scan and dump selected information from
// an MDL module and its imports (recursively). It keeps track of modules already visited
// in order to avoid repeated outputs.
class MDL_module_scanner
{
public:
    // MDL_module_scanner constructor.
    MDL_module_scanner(mi::neuraylib::INeuray* neuray, const Options& options);

    // The entry point for scanning an MDL module.
    void scan_module(const std::string& qualified_module_name);

private:
    // Dump the parameters of a material or function definition.
    void dump_definition(
        const mi::neuraylib::IFunction_definition* definition,
        mi::Size depth,
        std::ostream& s);

    // Dumps selective information of an MDL module and optionally its imports (recursively).
    void dump_module(
        const mi::neuraylib::IModule* module,
        mi::Size level);

    mi::neuraylib::INeuray* m_neuray;
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdl_factory;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    Options m_options;

    // List of modules already visited.
    std::unordered_set<std::string> m_module_list;
};

// MDL_module_scanner constructor.
MDL_module_scanner::MDL_module_scanner(mi::neuraylib::INeuray* neuray, const Options& options)
    : m_neuray(neuray)
    , m_options(options)
{
    m_mdl_factory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();
}

// The entry point for scanning an MDL module.
void MDL_module_scanner::scan_module(const std::string& qualified_module_name)
{
    // Access the database and scope.
    mi::base::Handle<mi::neuraylib::IDatabase>
        database(m_neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope>
        scope(database->get_global_scope());

    // Create execution context.
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>
        mdl_impexp_api(m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    
    mi::base::Handle <mi::neuraylib::IMdl_execution_context>
        context(m_mdl_factory->create_execution_context());

    // Create a transaction.
    m_transaction = scope->create_transaction();

    {
        // Load the module.
        check_success(mdl_impexp_api->load_module(
            m_transaction.get(), qualified_module_name.c_str(), context.get()) >= 0);
        print_messages(context.get());

        // Get module database name.
        mi::base::Handle<const mi::IString> module_db_name(
            m_mdl_factory->get_db_module_name(qualified_module_name.c_str()));

        mi::base::Handle<const mi::neuraylib::IModule> module(
            m_transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

        dump_module(module.get(), 0);
    }

    // All transactions need to get committed.
    m_transaction->commit();
}

// Dump the parameters of a material or function definition.
void MDL_module_scanner::dump_definition(
    const mi::neuraylib::IFunction_definition* definition,
    mi::Size level,
    std::ostream& s)
{
    const std::string shift(level * 4 + 4, ' ');

    mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
        m_mdl_factory->create_type_factory(m_transaction.get()));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
        m_mdl_factory->create_expression_factory(m_transaction.get()));

    const mi::Size count = definition->get_parameter_count();
    mi::base::Handle<const mi::neuraylib::IType_list> types(definition->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IExpression_list> defaults(definition->get_defaults());

    for (mi::Size index = 0; index < count; index++)
    {

        mi::base::Handle<const mi::neuraylib::IType> type(types->get_type(index));
        mi::base::Handle<const mi::IString> type_text(type_factory->dump(type.get(), level + 1));
        const std::string name = definition->get_parameter_name(index);
        s << shift << "parameter " << type_text->get_c_str() << " " << name;

        mi::base::Handle<const mi::neuraylib::IExpression> default_(
            defaults->get_expression(name.c_str()));
        if (default_.is_valid_interface())
        {
            mi::base::Handle<const mi::IString> default_text(
                expression_factory->dump(default_.get(), 0, level + 1));
            s << ", default = " << default_text->get_c_str() << std::endl;
        }
        else
        {
            s << " (no default)" << std::endl;
        }
    }

    const mi::Size temporary_count = definition->get_temporary_count();
    for (mi::Size i = 0; i < temporary_count; ++i)
    {
        mi::base::Handle<const mi::neuraylib::IExpression> temporary(definition->get_temporary(i));
        std::stringstream name;
        name << i;
        mi::base::Handle<const mi::IString> result(
            expression_factory->dump(temporary.get(), name.str().c_str(), 1));
        s << shift << "temporary " << result->get_c_str() << std::endl;
    }

    mi::base::Handle<const mi::neuraylib::IExpression> body(definition->get_body());
    mi::base::Handle<const mi::IString> result(expression_factory->dump(body.get(), 0, 1));
    if (result)
        s << shift << "body " << result->get_c_str() << std::endl;
    else
        s << shift << "body not available for this function" << std::endl;

    s << std::endl;
}

// Dump selective information of an MDL module and optionally its imports (recursively).
void MDL_module_scanner::dump_module(
    const mi::neuraylib::IModule* module,
    mi::Size level)
{
    const std::string separator(120 - level * 4, '-');
    const std::string shift(level * 4, ' ');

    // Get module display name.
    // Note that IModule::get_mdl_name() only returns the module MDL name which could be encoded.
    // Then IMdl_factory::decode_name() has to be used for display purposes.
    mi::base::Handle<const mi::IString>
        decoded_name(m_mdl_factory->decode_name(module->get_mdl_name()));
    const std::string module_name = decoded_name->get_c_str();

    // Print the module name and the file name it was loaded from.
    std::cout << shift << separator << std::endl;
    if (module->is_standard_module())
    {
        std::cout << shift << "Found module: " << module_name << " (Standard Module)" << std::endl;
    }
    else
    {
        std::cout << shift << "Found module: " << module_name << std::endl;
        const char* filename = module->get_filename();
        if (filename)
            std::cout << shift << "Loaded file : " << filename << std::endl;
    }
    std::cout << std::endl;

    // Dump imported modules.
    const mi::Size module_count = module->get_import_count();
    if (module_count > 0 && m_options.m_dump_options & DUMP_IMPORTS)
    {
        std::cout << shift << "+Imports included in " << module_name << ":" << std::endl;
        for (mi::Size i = 0; i < module_count; i++)
        {
            const std::string import_name(module->get_import(i));

            mi::base::Handle<const mi::neuraylib::IModule> imported_module(
                m_transaction->access<mi::neuraylib::IModule>((import_name.c_str())));

                mi::base::Handle<const mi::IString>
                    decoded_name(m_mdl_factory->decode_name(imported_module->get_mdl_name()));
                const std::string imported_name = decoded_name->get_c_str();

            std::cout << shift << "    " << imported_name << std::endl;
        }
        std::cout << std::endl;
    }

    // Dump exported types.
    if (m_options.m_dump_options & DUMP_TYPES)
    {
        mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
            m_mdl_factory->create_type_factory(m_transaction.get()));
        mi::base::Handle<const mi::neuraylib::IType_list> types(module->get_types());

        const mi::Size num_types = types->get_size();
        if (num_types > 0)
        {
            std::cout << shift << "+Types defined in " << module_name << ":" << std::endl;
            for (mi::Size i = 0; i < num_types; ++i)
            {
                mi::base::Handle<const mi::neuraylib::IType> type(types->get_type(i));
                mi::base::Handle<const mi::IString> result(type_factory->dump(type.get(), level + 1));
                std::cout << shift << "    " << result->get_c_str() << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Dump exported constants.
    if (m_options.m_dump_options & DUMP_CONSTANTS)
    {
        mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
            m_mdl_factory->create_value_factory(m_transaction.get()));
        mi::base::Handle<const mi::neuraylib::IValue_list> constants(module->get_constants());

        const mi::Size num_constants = constants->get_size();
        if (num_constants > 0)
        {
            std::cout << shift << "+Constants defined in " << module_name << ":" << std::endl;
            for (mi::Size i = 0; i < num_constants; ++i)
            {
                const char* name = constants->get_name(i);
                mi::base::Handle<const mi::neuraylib::IValue> constant(constants->get_value(i));
                mi::base::Handle<const mi::IString> result(value_factory->dump(constant.get(), 0, level + 1));
                std::cout << shift << "    " << name << " = " << result->get_c_str() << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Dump function definitions of the module.
    const mi::Size function_count = module->get_function_count();
    if (function_count > 0 && m_options.m_dump_options & DUMP_DEFINITIONS)
    {
        std::cout << shift << "+Functions defined in " << module_name << ":" << std::endl;
        for (mi::Size i = 0; i < function_count; i++)
        {
            const std::string function_name(module->get_function(i));

            const bool apply_filter = !m_options.m_def_filter.empty() &&
                (function_name.find(m_options.m_def_filter) != std::string::npos);

            if (m_options.m_def_filter.empty() || apply_filter)
            {
                std::cout << shift << "    " << function_name << std::endl;
                mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
                    m_transaction->access<mi::neuraylib::IFunction_definition>(function_name.c_str()));

                if (apply_filter)
                    dump_definition(function_definition.get(), level + 1, std::cout);

                // Dump thumbnail filename is available.
                const char* thumbnail = function_definition->get_thumbnail();
                if (thumbnail)
                    std::cout << shift << "    Thumbnail: " << thumbnail << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Dump material definitions of the module.
    const mi::Size material_count = module->get_material_count();
    if (material_count > 0 && m_options.m_dump_options & DUMP_DEFINITIONS)
    {
        std::cout << shift << "+Materials defined in " << module_name << ":" << std::endl;
        for (mi::Size i = 0; i < material_count; i++)
        {
            const std::string material_name(module->get_material(i));

            const bool apply_filter = !m_options.m_def_filter.empty() &&
                (material_name.find(m_options.m_def_filter) != std::string::npos);

            if (m_options.m_def_filter.empty() || apply_filter)
            {
                std::cout << shift << "    " << material_name << std::endl;
                mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
                    m_transaction->access<mi::neuraylib::IFunction_definition>(material_name.c_str()));

                if (apply_filter)
                    dump_definition(material_definition.get(), level + 1, std::cout);

                // Dump thumbnail filename is available.
                const char* thumbnail = material_definition->get_thumbnail();
                if (thumbnail)
                    std::cout << shift << "    Thumbnail: " << thumbnail << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Dump the resources referenced by this module.
    const mi::Size resources_count = module->get_resources_count();
    if (resources_count > 0 && m_options.m_dump_options & DUMP_RESOURCES)
    {
        std::cout << shift << "+Resources used in " << module_name << ":" << std::endl;
        for (mi::Size r = 0; r < resources_count; ++r)
        {
            mi::base::Handle<const mi::neuraylib::IValue_resource> resource(
                module->get_resource(r));
            const char* db_name = resource->get_value();
            const char* mdl_file_path = resource->get_file_path();

            if (db_name == nullptr)
            {
                // resource is either not used and therefore has not been loaded or
                // could not be found.
                std::cout << shift << "    db_name:               none" << std::endl;
                std::cout << shift << "    mdl_file_path:         " << mdl_file_path << std::endl
                    << std::endl;
                continue;
            }
            std::cout << shift << "    db_name:               " << db_name << std::endl;
            std::cout << shift << "    mdl_file_path:         " << mdl_file_path << std::endl;

            const mi::base::Handle<const mi::neuraylib::IType_resource> type(
                resource->get_type());
            switch (type->get_kind())
            {
            case mi::neuraylib::IType::TK_TEXTURE:
            {
                const mi::base::Handle<const mi::neuraylib::ITexture> texture(
                    m_transaction->access<mi::neuraylib::ITexture>(db_name));
                const mi::base::Handle<const mi::neuraylib::IImage> image(
                    m_transaction->access<mi::neuraylib::IImage>(texture->get_image()));

                for (mi::Size f = 0, fn = image->get_length(); f < fn; ++f)
                    for (mi::Size t = 0, tn = image->get_frame_length(f); t < tn; ++t)
                    {
                        const char* resolved_file_path = image->get_filename(f, t);
                        if (resolved_file_path)
                        {
                            std::cout << shift << "    resolved_file_path[" << f << "," << t << "]: "
                                << resolved_file_path << std::endl;
                        }
                    }
                break;
            }

            case mi::neuraylib::IType::TK_LIGHT_PROFILE:
            {
                const mi::base::Handle<const mi::neuraylib::ILightprofile> light_profile(
                    m_transaction->access<mi::neuraylib::ILightprofile>(db_name));
                const char* resolved_file_path = light_profile->get_filename();
                if (resolved_file_path)
                    std::cout << shift << "    resolved_file_path:    " << resolved_file_path << std::endl;
                break;
            }

            case mi::neuraylib::IType::TK_BSDF_MEASUREMENT:
            {
                const mi::base::Handle<const mi::neuraylib::IBsdf_measurement> mbsdf(
                    m_transaction->access<mi::neuraylib::IBsdf_measurement>(db_name));
                const char* resolved_file_path = mbsdf->get_filename();
                if(resolved_file_path)
                    std::cout << shift << "    resolved_file_path:    " << resolved_file_path << std::endl;
                break;
            }

            default:
                exit_failure(std::string("Unexpected Resource type: " +
                    std::to_string(type->get_kind())).c_str());
                break;
            }
            std::cout << std::endl;
        }
    }

    // Dump imported modules recursively.
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

            // Access and dump imported module.
            {
                mi::base::Handle<const mi::neuraylib::IModule> imported_module(
                    m_transaction->access<mi::neuraylib::IModule>(db_name.c_str()));

                if (m_options.m_skip_standard && imported_module->is_standard_module())
                    continue;

                dump_module(imported_module.get(), level + 1);
            }
        }
    }

    std::cout << std::endl;
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
  -f|--filter               Select type of information to be dumped.
                              Format: -f "type1|type2|...|typeN".
                              Types: 'imports', 'types', 'constants', definitions', 'resources',
                                     'all'.
                              Default: "imports|resources".
  -r|--recursive            Scan imported modules recursively.
  -s|--skip_standard        Skip standard modules.
  -d|--filter_definitions   If 'definitions' have been included with the '-f' option, this extra
                            filter will select only function and material definitions with names
                            that contain the "substring", and for the previous, it will dump
                            extended information like parameters with its defaults, temporaries
                            and the body.
                              Format: -d "substring")";

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
            else if (arg == "-f" || arg == "--filter")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -f|--filter missing." << std::endl;
                    return false;
                }

                std::istringstream ss(argv[++i]);
                std::string type;

                m_dump_options = DUMP_NOTHING;

                while (std::getline(ss, type, '|'))
                {
                    if (type == "imports")
                    {
                        m_dump_options |= DUMP_IMPORTS;
                    }
                    else if (type == "types")
                    {
                        m_dump_options |= DUMP_TYPES;
                    }
                    else if (type == "constants")
                    {
                        m_dump_options |= DUMP_CONSTANTS;
                    }
                    else if (type == "definitions")
                    {
                        m_dump_options |= DUMP_DEFINITIONS;
                    }
                    else if (type == "resources")
                    {
                        m_dump_options |= DUMP_RESOURCES;
                    }
                    else if (type == "all")
                    {
                        m_dump_options |= DUMP_ALL;
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
            else if (arg == "-s" || arg == "--skip_standard")
            {
                m_skip_standard = true;
            }
            else if (arg == "-d" || arg == "--filter_definitions")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -d|--filter_definitions missing." << std::endl;
                    return false;
                }

                m_def_filter = argv[++i];
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

    // Load an MDL module to scan and dump selected information.
    {
        MDL_module_scanner module_scanner(neuray.get(), options);
        module_scanner.scan_module(options.m_qualified_module_name);
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
