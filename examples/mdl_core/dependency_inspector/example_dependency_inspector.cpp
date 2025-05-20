/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
// Loads a MDL module and finds all paths of its import and resource dependencies.

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
    
    /// MDL search paths
    std::vector<std::string> m_mdl_paths;
};

// This helper class contains some methods to scan and dump selected information from a MDL module
// and its imports (recursively). It also keeps track of modules already visited or files already
// listed in order to avoid repeated outputs.

class Dependency_inspector
{
public:
    // Dependency_inspector constructor.
    Dependency_inspector(mi::mdl::IMDL* mdl_compiler, const Options& options);

    // The entry point for scanning a MDL module.
    void scan_module(const std::string& qualified_module_name);

private:
    // List the dependencies of an MDL module and optionally from its imports (recursively).
    void list_dependencies( const mi::mdl::IModule* module);

    // Check if a filepath was already listed.
    bool was_listed(const char* filepath);

    mi::mdl::IMDL* m_mdl_compiler;
    mi::base::Handle<mi::mdl::IThread_context> m_ctx;
    mi::base::Handle<mi::mdl::IEntity_resolver> m_entity_resolver;
    
    Options m_options;

    // List of modules already visited.
    std::unordered_set<std::string> m_module_list;
    // List of file paths already listed.
    std::unordered_set<std::string> m_filepath_list;
};

// Dependency_inspector constructor.
Dependency_inspector::Dependency_inspector(mi::mdl::IMDL* mdl_compiler, const Options& options)
: m_mdl_compiler(mdl_compiler)
, m_options(options)
{
    m_ctx = mdl_compiler->create_thread_context();
    m_entity_resolver = mdl_compiler->get_entity_resolver(nullptr);
}

// The entry point for scanning a MDL module.
void Dependency_inspector::scan_module(const std::string& qualified_module_name)
{
    mi::base::Handle<mi::mdl::IModule const>
        module(m_mdl_compiler->load_module(m_ctx.get(), qualified_module_name.c_str(), nullptr));

    if (!module.is_valid_interface())
    {
        std::cout << std::endl <<
            "The module \"" << qualified_module_name << "\" could not be loaded." << std::endl;
        return;
    }

    std::cout << std::endl <<
        "Dependencies founds in module " << module->get_name() << std::endl;
    list_dependencies(module.get());
}

// List the dependencies of an MDL module and optionally from its imports (recursively).
void Dependency_inspector::list_dependencies( const mi::mdl::IModule* module)
{
    // List imports
    if (m_options.m_dependency_options & DEPENDENCY_IMPORTS)
    {
        const char* filename = module->get_filename();
        if (filename && !was_listed(filename))
            std::cout << filename << std::endl;
    }

    // List resources.
    if ((m_options.m_dependency_options & DEPENDENCY_RESOURCES) && !module->is_mdle())
    {
        // List resources referenced by this module.
        const mi::Size rn = module->get_referenced_resources_count();
        for (mi::Size r = 0; r < rn; ++r)
        {
            mi::base::Handle<mi::mdl::IMDL_resource_set> resource_set(
            m_entity_resolver->resolve_resource_file_name(
                module->get_referenced_resource_url(r), module->get_filename(), module->get_name(), nullptr, m_ctx.get()));

            // A get_count() > 1 means it's a tiled resource which consists of multiple files.
            for (mi::Size i = 0; i < resource_set->get_count(); ++i)
            {
                const mi::mdl::IMDL_resource_element *res_element = resource_set->get_element(i);
                for (mi::Size j = 0; j < res_element->get_count(); ++j)
                {
                    const char *filename = res_element->get_filename(j);
                        if (!was_listed(filename))
                            std::cout << filename << std::endl;
                }
            }
        }

        // List thumbnails referenced by function and material definitions of this module.
        const mi::Size definition_count = module->get_exported_definition_count();
        for (mi::Size d = 0; d < definition_count; d++)
        {
            // To get the filepath of thumbnail resources we need to look for the annotation
            // block of the function declarations.
            const mi::mdl::IDefinition* definition = module->get_exported_definition(d);
            if (definition->get_kind() == mi::mdl::IDefinition::DK_FUNCTION)
            {
                // Get declarations of this definition.
                const mi::mdl::IDefinition* orig_def = module->get_original_definition(definition);

                // It's either at the prototype declaration or at the declaration (in that order).
                const mi::mdl::IDeclaration_function *proto_decl =
                    static_cast<const mi::mdl::IDeclaration_function*>(orig_def->get_prototype_declaration());

                if (proto_decl == nullptr)
                    proto_decl = static_cast<const mi::mdl::IDeclaration_function*>(orig_def->get_declaration());

                // Retrieve the definition annotation block.
                const mi::mdl::IAnnotation_block* annotations = proto_decl->get_annotations();

                for (mi::Size i = 0; annotations && i < annotations->get_annotation_count(); ++i)
                {
                    const mi::mdl::IAnnotation *anno = annotations->get_annotation(i);
                    if (!anno)
                        continue;

                    // Search for names with thumbnail annotation semantics.
                    const mi::mdl::IQualified_name *qname = anno->get_name();
                    const mi::mdl::IDefinition::Semantics semantics = qname->get_definition()->get_semantics();

                    if (semantics == mi::mdl::IDefinition::DS_THUMBNAIL_ANNOTATION)
                    {
                        // For thumbnail annotations there is only one argument and it's an literal expression
                        // and its value is a string containing the thumbnail relative path.
                        const mi::mdl::IArgument* argument = anno->get_argument(0);
                        const mi::mdl::IExpression_literal* expr =
                            static_cast<const mi::mdl::IExpression_literal*>(argument->get_argument_expr());
                        const mi::mdl::IValue_string* val_str =
                            static_cast<const mi::mdl::IValue_string*>(expr->get_value());

                        // The entity resolver is used to obtain the full path.
                        mi::base::Handle<mi::mdl::IMDL_resource_set> resource_set(
                        m_entity_resolver->resolve_resource_file_name(
                            val_str->get_value(), module->get_filename(), module->get_name(), nullptr, m_ctx.get()));

                        const char* filename = resource_set->get_filename_mask();
                        if (!was_listed(filename))
                            std::cout << filename << std::endl;
                    }                  
                }                
            }
        }
    }

    // Inspect (optionally) imported modules recursively.
    if (m_options.m_recursive)
    {
        const mi::Size module_count = module->get_import_count();
        for (mi::Size i = 0; i < module_count; i++)
        {
            const mi::mdl::IModule* imported_module(module->get_import(i));

            // Skip built-in modules.
            if (imported_module->is_builtins() || imported_module->is_stdlib())
                continue;

            const std::string db_name(imported_module->get_name());

            // Skip modules already visited.
            auto it = std::find(m_module_list.begin(), m_module_list.end(), db_name);
            if (it != m_module_list.end())
                continue;

            // Update module list.
            m_module_list.insert(db_name);

            list_dependencies(imported_module);
        }
    }
}

// Check if a file path was already listed.
bool Dependency_inspector::was_listed(const char* filepath)
{
    // Skip file paths already listed.
    auto it = std::find(m_filepath_list.begin(), m_filepath_list.end(), filepath);
    if (it != m_filepath_list.end())
        return true;

    // Update filepath list
    m_filepath_list.insert(filepath);
    return false;
}

// Print command line usage.
void Options::print_usage(std::ostream& s)
{
    s << R"(
dependency_inspector [options] <qualified_module_name>

options:

  -h|--help                 Print this usage message and exit.
  -p|--mdl_path <path>      Add the given path to the MDL search path.
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
    // Get MDL search root for the MDL Core examples
    m_mdl_paths.push_back(get_samples_mdl_root());

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg[0] == '-')
        {
            if (arg == "-h" || arg == "--help")
            {
                return false;
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
        exit(EXIT_FAILURE);
    }

    // Access the MDL core compiler (also the entry point for the MDL Core API).
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    // Set MDL search paths.
    mi::base::Handle<MDL_search_path> search_path( new MDL_search_path());

    // Increment reference counter, as the MDL compiler will take ownership (and not increment it),
    // but we will keep a reference.
    search_path->retain();

    // Add user defined paths.
    for (auto path : options.m_mdl_paths)
        search_path->add_path(path.c_str());

    mdl_compiler->install_search_path(search_path.get());

    // Load an MDL module and list its dependencies.
    Dependency_inspector dependency_inspector(mdl_compiler.get(), options);
    dependency_inspector.scan_module(options.m_qualified_module_name);

    return EXIT_SUCCESS;
}
// Convert command line arguments to UTF8 on Windows.
COMMANDLINE_TO_UTF8
