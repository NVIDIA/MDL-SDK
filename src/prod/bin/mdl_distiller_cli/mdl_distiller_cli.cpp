/******************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief Loads an MDL material and exports a projected simplified MDL material

#include <fstream>
#include <iostream>

#include <mi/mdl_sdk.h>

// Helper classes and functions
#include "condition_checker.h"
#include "user_timer.h"
#include "neuray_factory.h"
#include "logger.h"

// MDL projection classes and functions
#include "options.h"
#include "framework.h"
#include "mdl_printer.h"
#include "mdl_distiller_utils.h"

using mi::base::Handle;
using mi::base::ILogger;

using mi::neuraylib::ICompiled_material;
using mi::neuraylib::ILogging_configuration;
using mi::neuraylib::IMdl_configuration;
using mi::neuraylib::IMdl_impexp_api;
using mi::neuraylib::INeuray;
using mi::neuraylib::IPlugin_configuration;
using mi::neuraylib::ITransaction;
using mi::neuraylib::IMdl_factory;
using mi::neuraylib::Neuray_factory;

/// Helper macro. Checks whether the expression is true and if not
/// prints a message and exits.
#define check_success( expr) \
    do { \
        if( !(expr)) { \
            fprintf( stderr, "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
            exit( EXIT_FAILURE); \
        } \
    } while( false)

/// Print usage message with command line documentation and exit the
/// process.  Return status code 0 to operating system if `ok` is
/// true, and EXIT_FAILURE otherwise.
void usage(bool ok = false) {
    std::cerr << "Usage: mdl_distiller_cli [<options>] <mdl_material>\n"
        "Options:\n"
        "    -v <level>             sets the log verbosity level for the MDL compiler\n"
        "    -trace [<lvl>]         lvl: 0 = none, 1 = report the matching rules (default)\n"
        "                                2 = report all checked path and the matching rules\n"
        "    -debug-print           enable MDLTL debug_print statements\n"
        "    -o <out-filename>      filename for distilled material, '-' = stdout (default)\n"
        "    -outline               DF node outline on stderr\n"
        "    -quiet                 no output on stderr\n"
        "    -p <path>              MDL search path, can be given multiple times\n"
        "    -m <mode>              distiller target mode. Ignored if '-test spec' is set\n"
        "    -lm                    list all distiller target modes.\n"
        "    -lrm                   list the required MDL module names "
                                    "for all loaded distiller targets.\n"
        "    -class                 uses class instead of instance compilation.\n"
        "                           Note: exports only a material instance though.\n"
        "    -export-spec <v>       MDL version used for export: auto, 1.3, 1.6, or 1.7\n"
        "    -dist-supp-file <name> alternate filename for distilling support module.\n"
        "    -no-layer-normal       do not copy layer normal to global normal map.\n"
        "    -merge-metal-and-base-color (true|false)\n"
        "                           merge metal and base color (default true)\n"
        "    -merge-transmission-and-base-color (true|false)\n"
        "                           merge transmission and base color (default false)\n"
        "    -emission              export emission, default is to not export emission.\n"
        "    -tmm                   enable target material mode.\n"
        "    -tm <path>             add module that defines a custom target material.\n"
        "    -bake <n>              enables texture baking and sets resolution to n times n.\n"
        "    -all-textures          bake all textures.\n"
        "    -texture-dir <path>    directory where to store textures, default '.'.\n"
        "    -plugin <filename>     add additional distiller plugin, can be used more than once.\n"
        "    -no-std-plugin         do not load standard 'mdl_{lod_}distiller.{so|dll} plugins.\n"
        "    -test <type>           run test mode. type is one of: normal, spec.\n"
        "    -test_log <path>       path where to store test results (default is '.').\n"
        "    -test-targets <list>   override targets for -spec test mode. "
                                    "list is a comma-separated\n"
        "                           list of target names.\n"
        "                           (default is 'diffuse,specular_glossy,transmissive_pbr,ue4').\n"
        "\n";
    exit( ok ? EXIT_SUCCESS : EXIT_FAILURE);
}


/// Configure the MDL SDK with module search paths and load requested
/// plugins.
void configuration(
    INeuray* neuray,
    ILogger* logger,
    Options* options,
    std::vector<const char*> plugins,
    bool no_std_plugin)
{
    // Empty old log file if exists and in test mode
    if (options->test_suite) {
        std::ofstream log_file((options->test_dir + SLASH + LOG_FILE).c_str());
        log_file.flush();
        log_file.close();
    }

    Handle<ILogging_configuration> logging_config(
        neuray->get_api_component<ILogging_configuration>());
    logging_config->set_receiving_logger(logger);

    Handle<IMdl_configuration> mdl_config(
        neuray->get_api_component<IMdl_configuration>());
    for ( std::size_t i = 0; i < options->paths.size(); ++i) {
        check_success(mdl_config->add_mdl_path(options->paths[i]) == 0);
    }
    // Load the OpenImageIO plugin.
    Handle<IPlugin_configuration> plugin_config(
        neuray->get_api_component<IPlugin_configuration>());
    check_success(plugin_config->load_plugin_library(
                      "nv_openimageio" MI_BASE_DLL_FILE_EXT) == 0);

    if ( ! no_std_plugin) {
        plugin_config->load_plugin_library( "mdl_distiller" MI_BASE_DLL_FILE_EXT);
    }
    for ( std::vector<const char*>::const_iterator it = plugins.begin();
          it != plugins.end();
          ++it) {
        check_success(plugin_config->load_plugin_library( *it) == 0);
    }
}

/// Query all required MDL module names and source code (as strings)
/// for the given target from the Distiller API and load them.
mi::Sint32 load_required_modules(INeuray *neuray,
                                 ITransaction *transaction,
                                 IMdl_impexp_api *mdl_impexp_api,
                                 Options *options,
                                 char const *target)
{
    Handle<IMdl_factory> mdl_factory(
        neuray->get_api_component<IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

    if (options->target_material_model_mode) {
        context->set_option("target_material_model_mode", true);
    }

    mi::Size req_modules = distiller_api->get_required_module_count(target);
    for (mi::Size i = 0; i < req_modules; i++) {
        char const *module_name = distiller_api->get_required_module_name(target, i);
        char const *module_code = distiller_api->get_required_module_code(target, i);

        // Load module.
        if (mdl_impexp_api->load_module_from_string( transaction, module_name, 
                    module_code, context.get()) != 0) {
            std::cerr << "message count" << context->get_messages_count() << "\n";
            for (mi::Size j = 0; j < context->get_messages_count(); j++) {
                Handle<mi::neuraylib::IMessage const> msg(context->get_message(j));
                std::cerr << "ERROR: " << msg->get_string() << "\n";
            }
            std::cerr << "ERROR: cannot load module '" << module_name << "'\n";
            return 1;
        }
    }
    return 0;
}

/// Load all MDL module that have been passed in with command line
/// option -tm.
mi::Sint32 load_additional_modules(INeuray *neuray,
                                   ITransaction *transaction,
                                   IMdl_impexp_api *mdl_impexp_api,
                                   Options *options)
{
    Handle<IMdl_factory> mdl_factory(
        neuray->get_api_component<IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    if (options->target_material_model_mode) {
        context->set_option("target_material_model_mode", true);
    }

    for (auto const &module_name : options->additional_modules) {
        // Load module.
        if (mdl_impexp_api->load_module( transaction, module_name, context.get()) != 0) {
            std::cerr << "ERROR: cannot load module '" << module_name << "'\n";
            return 1;
        }
    }
    return 0;
}

/// Load a material from the file system, deriving the module and
/// material names from the fully qualified name `material_name` from
/// the command line. Then create a material instance and store it in
/// the DB under name `instance_name`.
///
/// \return 0 on success, an error code on error.
mi::Sint32 load_and_instantiate_material(INeuray *neuray,
                                         ITransaction *transaction,
                                         IMdl_impexp_api *mdl_impexp_api,
                                         char const *material_name,
                                         char const *instance_name,
                                         Options *options)
{
    std::string module_mdl_name = get_module_from_qualified_name( material_name);

    // Load module and create a material instance
    std::string module_db_name = std::string("mdl") + module_mdl_name;
    std::string material_db_name = std::string("mdl") + material_name;

    Handle<IMdl_factory> mdl_factory(
        neuray->get_api_component<IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    if (options->target_material_model_mode) {
        context->set_option("target_material_model_mode", true);
    }

    // Load module.
    if (mdl_impexp_api->load_module( transaction, module_mdl_name.c_str(), context.get()) != 0) {
        std::cerr << "ERROR: cannot load module '" << module_mdl_name.c_str() << "'\n";
        return 1;
    }

    // Use overload resolution to find the exact signature.
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<const mi::neuraylib::IModule>(module_db_name.c_str()));
    if (!module) {
        std::cerr << "ERROR: failed to access module '" << module_mdl_name.c_str() << "'\n";
        return -1;
    }

    // Find material overload for the exact signature.
    mi::base::Handle<const mi::IArray> overloads(
        module->get_function_overloads(material_db_name.c_str()));
    if (!overloads || overloads->get_length() != 1) {
        std::cerr << "ERROR: failed to find signature for material '" << material_db_name
             << "'\n";
        return -1;
    }
    mi::base::Handle<const mi::IString> overload(overloads->get_element<mi::IString>(0));
    material_db_name = overload->get_c_str();

    // create instance with the default arguments.
    Handle<const mi::neuraylib::IFunction_definition> material_definition(
        transaction->access<mi::neuraylib::IFunction_definition>( material_db_name.c_str()));
    if (! material_definition.is_valid_interface()) {
        std::cerr << "ERROR: Material '" << material_name << "' does not exist.\n";
        return 1;
    }
    mi::Sint32 result;
    Handle<mi::neuraylib::IFunction_call> material_instance(
        material_definition->create_function_call( 0, &result));
    switch ( result) {
    case -3:
        std::cerr << "ERROR: Material '" << material_name << "' cannot be default initialized.\n";
        return 1;
    case -4:
        std::cerr << "ERROR: Material '" << material_name << "' is not exported.\n";
        return 1;
    }
    transaction->store( material_instance.get(), instance_name);

    return 0;
}

/// Compile the material instance `instance_name` and return it.
///
/// Depending on the given options, the material is compiled in
/// instance or class mode.
///
/// \return compiled material on success, NULL otherwise.
ICompiled_material const *compile_material(INeuray *neuray,
                                           ITransaction *transaction,
                                           Options *options,
                                           char const *instance_name)
{
    // Compile the material instance in instance compilation mode
    Handle<const mi::neuraylib::IFunction_call> material_instance(
        transaction->access<mi::neuraylib::IFunction_call>(instance_name));
    check_success(material_instance.is_valid_interface());

    mi::Uint32 flags =
        options->class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
    check_success(material_instance2.is_valid_interface());

    Handle<IMdl_factory> mdl_factory(
        neuray->get_api_component<IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    if (options->target_material_model_mode) {
        context->set_option("target_material_model_mode", true);
    }

    ICompiled_material const *compiled_material =
        material_instance2->create_compiled_material(flags, context.get());

    return compiled_material;
}

/// Run the test suite on the given compiled material.
///
/// The options that enable this function are intended to be used by
/// the neuray tests, see
/// http://qaweb.mi.nvidia.com:8090/results/testsuite/
/// overview-current-mdl-distiller-test-binary-trunk.html
/// for the current results.
///
/// Depending on the given options, this runs either:
///
/// - spec tests: distill the given material to all targets supported
///   by this program
///
/// - normal tests: distill the given material to the specified target
///   and check that the output contains a marker that is defined in
///   the input file as a special comment.
///
/// \return 0 success, error code otherwise.
mi::Sint32 run_test_suite(INeuray *neuray,
                          ITransaction *transaction,
                          IMdl_impexp_api *mdl_impexp_api,
                          ILogger *logger,
                          User_timer &total_time,
                          Options *options,
                          char const *target,
                          char const *material_name,
                          ICompiled_material const *compiled_material)
{
    mi::Sint32 test_result = 0;

    init_ruid_file(options->test_dir);

    std::string out_path( options->test_dir + SLASH + "dist_");

    std::string name(material_name);
    name = name.substr(name.find_last_of("::") + 1, name.length());

    if (options->spec_test) {
        // 'spec' test mode --> Distill to all targets in spec_test_targets
        out_path += name;
        mdl_spec old_export_spec = options->export_spec;
        if (options->test_targets.size() == 0) {
            for (mi::Size idx = 0; idx < dimension_of(spec_test_targets); ++idx) {
                const char* test_target = spec_test_targets[idx];
                options->test_targets.push_back(test_target);
            }
        }

        for ( mi::Size idx = 0; idx < options->test_targets.size(); ++idx) {
            const char* test_target = options->test_targets[idx].c_str();
            // Distill to mode
            options->export_spec = old_export_spec;  // mdl_distill overwrites this
            std::string out_dump( out_path + '_' + test_target + ".mdl");
            {
                std::ofstream out_file(out_dump.c_str());
                Handle<const ICompiled_material> res_mat(
                    mdl_distill(neuray, mdl_impexp_api, transaction,
                                compiled_material, material_name, test_target, options,
                                total_time.time() * 1000, &out_file));

                logger->message(mi::base::MESSAGE_SEVERITY_INFO, "",
                                (std::string("Wrote distilled material to: ")
                                 + out_dump).c_str());

                if (! res_mat.is_valid_interface())
                    test_result = ERR_DISTILLING;
            }
        }
    } else {
        // 'normal' test mode --> Distill to one target
        // Distill marker has to be available
        std::string d_marker = get_distill_marker( options->paths[0], material_name);
        if (d_marker.length() > 0) {
            logger->message(mi::base::MESSAGE_SEVERITY_INFO, "",
                            (std::string("Precondition: ") + material_name
                             + " must be distilled to " + d_marker).c_str());
        } else {
            logger->message(
                mi::base::MESSAGE_SEVERITY_ERROR, "",
                (std::string("Precondition failure: No marker defined in material ")
                 + material_name).c_str());

            test_result = ERR_PRECONDITION;
        }

        out_path += (name + ".mdl");
        {
            std::ofstream out_file(out_path.c_str());

            Handle<const ICompiled_material> result_material(
                mdl_distill(neuray, mdl_impexp_api, transaction,
                            compiled_material, material_name, target, options,
                            total_time.time() * 1000, &out_file));
            if ( ! result_material.is_valid_interface())
                test_result = ERR_DISTILLING;
            logger->message(mi::base::MESSAGE_SEVERITY_INFO, "",
                            (std::string("Wrote distilled material to: ") + out_path).c_str());
        }
        if (d_marker.length() > 0) {
            // Check distilled material for correctness
            if (check_distilled_material(out_path.c_str(), d_marker.c_str())) {
                logger->message(mi::base::MESSAGE_SEVERITY_INFO, "",
                                (std::string("Postcondition: ") + material_name
                                 + " matching " + d_marker).c_str());
            } else {
                logger->message(mi::base::MESSAGE_SEVERITY_ERROR, "",
                                (std::string("Postcondition failure: ") + material_name
                                 + " NOT matching " + d_marker).c_str());
                test_result = ERR_POSTCONDITION;
            }
        }
    }
    return test_result;
}

/// Main function to distill an MDL material.
/// It loads the MDL material from an MDL module, converts its
/// material to an MDL expression, applies selected rule sets, and
/// writes the distilled result back out.  It also checks whether the
/// generated module compiles.
///
/// Requires adaption to the use case.
///
/// \return 0 in case of success and 1 in case of failure.
///
mi::Sint32 mdl_distill_main( INeuray*    neuray,
                             const char* material_name,
                             const char* target,
                             Options*    options)
{
    mi::Sint32 test_result = 0;
    User_timer total_time;
    total_time.start();

    // Import/export API is required to load the module from the file
    // system.
    Handle<IMdl_impexp_api> mdl_impexp_api( neuray->get_api_component<IMdl_impexp_api>());

    // Create a database transaction to load, compile and distill a
    // material.
    Handle<mi::neuraylib::IDatabase> database(
        neuray->get_api_component<mi::neuraylib::IDatabase>());
    Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
    Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

    // Pre-load the distiller_support module and its dependent
    // modules.
    if (mdl_impexp_api->load_module(transaction.get(), "::nvidia::distilling_support") != 0) {
        std::cerr << "ERROR: cannot load module '::nvidia::distilling_support'\n";
        return 1;
    }

    // Load additional modules that have been requested on the command
    // line, e.g. modules defining custom target materials.
    if (load_additional_modules(neuray, transaction.get(), mdl_impexp_api.get(), options) != 0) {
        std::cerr << "ERROR: cannot load requested modules\n";
        return 1;
    }

    if (load_required_modules(neuray, transaction.get(), mdl_impexp_api.get(), options, 
                target) != 0) {
        std::cerr << "ERROR: cannot load modules required by target " << target << "\n";
        return 1;
    }

    const char* instance_name = "material_instance";

    // Load the material mentioned on the command line and create a
    // default-initialized material instance. This instance is stored
    // in the DB under the name `instance_name`.
    if (load_and_instantiate_material(neuray,
                                      transaction.get(),
                                      mdl_impexp_api.get(),
                                      material_name,
                                      instance_name,
                                      options) != 0)
        return 1;

    {
        // Compile the material instance.
        Handle<ICompiled_material const> compiled_material(
            compile_material(neuray, transaction.get(), options, instance_name));
        if (!compiled_material.is_valid_interface())
            return 1;

        total_time.stop();

        // Obtain a logger.
        Handle<ILogging_configuration> logging_config(
            neuray->get_api_component<ILogging_configuration>());
        Handle<ILogger> logger(logging_config->get_forwarding_logger());

        if (options->test_suite) {
            // If requested by the user, run the test suite.
            test_result = run_test_suite(neuray, transaction.get(), mdl_impexp_api.get(),
                                         logger.get(), total_time, options, target,
                                         material_name, compiled_material.get());
        } else {

            // If not testing, distill the material and write the
            // result to stdout or the given filename as MDL code.

            if ( options->out_filename == "-") {
                Handle<const ICompiled_material> result_material(
                    mdl_distill(neuray, mdl_impexp_api.get(), transaction.get(),
                        compiled_material.get(), material_name, target, options,
                                total_time.time() * 1000, &std::cout));

                if (!result_material.is_valid_interface())
                    test_result = ERR_DISTILLING;
            } else {
                std::ofstream out_file(options->out_filename.c_str());
                Handle<const ICompiled_material> result_material(
                    mdl_distill(neuray, mdl_impexp_api.get(), transaction.get(),
                        compiled_material.get(), material_name, target, options,
                        total_time.time() * 1000, &out_file));

                if (!result_material.is_valid_interface())
                    test_result = ERR_DISTILLING;

                logger->message(mi::base::MESSAGE_SEVERITY_INFO, "",
                                (std::string("Wrote distilled material to: ")
                                 + options->out_filename.c_str()).c_str());
            }
        }
    }
    transaction->commit();
    return test_result;
}

/// Returns true if distilling target is valid
bool check_target( INeuray* neuray, const char* target) {
    Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    mi::Size n = distiller_api->get_target_count();
    for ( mi::Size i = 0; i < n; ++i) {
        if ( 0 == strcmp( target, distiller_api->get_target_name(i)))
            return true;
    }
    return false;
}

/// Lists all available targets, works only after neuray is started and plugins are loaded
void list_all_targets( INeuray* neuray) {
    Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

    mi::Size n = distiller_api->get_target_count();

    std::cerr << "All distilling targets:";
    for ( mi::Size i = 0; i < n; ++i) {
        std::cerr << "\n    " << distiller_api->get_target_name(i);
    }
    std::cerr << "\nNicknames:";
    std::cerr << "\n    sg  -->  specular_glossy";
    std::cerr << "\n";
}

/// Lists all available targets, works only after neuray is started and plugins are loaded
void list_all_required_modules( INeuray* neuray) {
    Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

    mi::Size n = distiller_api->get_target_count();

    std::cerr << "Required modules:\n";
    for ( mi::Size i = 0; i < n; ++i) {
        std::cerr << "  " << distiller_api->get_target_name(i) << "\n";
        mi::Size m = distiller_api->get_required_module_count(distiller_api->get_target_name(i));
        for (mi::Size t = 0; t < m; t++) {
            std::cerr << "    " 
                << distiller_api->get_required_module_name(distiller_api->get_target_name(i), t) 
                << "\n";
        }
    }
}

/// main: collect command line, configure MDL compiler and run MDL distiller.
int main( int argc, char* argv[]) {
    Options options;
    std::vector<const char*> args;
    const char* target = "diffuse";
    bool list_modes = false;
    bool list_required_modules = false;
    std::vector<const char*> plugins;
    bool no_std_plugin = false;

    for ( int i = 1; i < argc; ++i) {
        if ((0 == strcmp( "-h", argv[i])) || (0 == strcmp( "--help", argv[i]))) {
            usage( true);
        } else if (0 == strcmp( "-v", argv[i])) {
            ++i;
            if ( i < argc) {
                options.verbosity = std::atoi( argv[i]);
            } else {
                std::cerr << "Error: command line argument -v misses <level> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-trace", argv[i])) {
            options.trace = 1;
            if ( i+1 < argc && std::isdigit( argv[i+1][0])) {
                ++i;
                int level = std::atoi(argv[i]);
                if ((level >= 0) && (level <= 2)) {
                    options.trace = level;
                } else {
                    std::cerr << "Error: command line argument -trace has out-of-range <level> "
                        "parameter.\n";
                    usage();
                }
            }
        } else if (0 == strcmp( "-outline", argv[i])) {
            options.outline = true;
        } else if (0 == strcmp( "-debug-print", argv[i])) {
            options.debug_print = true;
        } else if (0 == strcmp( "-quiet", argv[i])) {
            options.quiet = true;
            options.verbosity = 0;
        } else if (0 == strcmp( "-o", argv[i])) {
            ++i;
            if ( i < argc) {
                options.out_filename = argv[i];
            } else {
                std::cerr << "Error: command line argument -o misses <out-filename> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-p", argv[i])) {
            ++i;
            if ( i < argc) {
                options.paths.push_back( argv[i]);
            } else {
                std::cerr << "Error: command line argument -p misses <path> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-m", argv[i])) {
            ++i;
            if ( i < argc) {
                target = argv[i];
                if ( 0 == strcmp( target, "sg"))
                    target = "specular_glossy";
            } else {
                std::cerr << "Error: command line argument -m misses <mode> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-lm", argv[i])) {
            list_modes = true;
        } else if (0 == strcmp( "-lrm", argv[i])) {
            list_required_modules = true;
        } else if (0 == strcmp( "-no-layer-normal", argv[i])) {
            options.layer_normal = false;
        } else if (0 == strcmp( "-merge-metal-and-base-color", argv[i])) {
            ++i;
            if ( i < argc) {
                if ( 0 == strcmp( "true", argv[i]))
                    options.merge_metal_and_base_color = true;
                else if ( 0 == strcmp( "false", argv[i]))
                    options.merge_metal_and_base_color = false;
                else {
                    std::cerr << "Error: unknown argument for -merge-metal-and-base-color "
                        "parameter, needs to be 'true' or 'false'.\n";
                    usage();
                }
            } else {
                std::cerr << "Error: command line argument -merge-metal-and-base-color misses "
                    "value, needs to be 'true' or 'false'.\n";
                usage();
            }
        } else if (0 == strcmp( "-merge-transmission-and-base-color", argv[i])) {
            ++i;
            if ( i < argc) {
                if ( 0 == strcmp( "true", argv[i]))
                    options.merge_transmission_and_base_color = true;
                else if ( 0 == strcmp( "false", argv[i]))
                    options.merge_transmission_and_base_color = false;
                else {
                    std::cerr << "Error: unknown argument for -merge-transmission-and-base-color "
                        "parameter, needs to be 'true' or 'false'.\n";
                    usage();
                }
            } else {
                std::cerr << "Error: command line argument -merge-transmission-and-base-color "
                    "misses value, needs to be 'true' or 'false'.\n";
                usage();
            }
        } else if (0 == strcmp( "-emission", argv[i])) {
            options.emission = true;
        } else if (0 == strcmp( "-class", argv[i])) {
            options.class_compilation = true;
        } else if (0 == strcmp( "-export-spec", argv[i])) {
            ++i;
            if ( i < argc) {
                if ( 0 == strcmp( "auto", argv[i]))
                    options.export_spec = mdl_spec_auto;
                else if ( 0 == strcmp( "1.3", argv[i]))
                    options.export_spec = mdl_spec_1_3;
                else if ( 0 == strcmp( "1.6", argv[i]))
                    options.export_spec = mdl_spec_1_6;
                else if ( 0 == strcmp( "1.7", argv[i]))
                    options.export_spec = mdl_spec_1_7;
                else if ( 0 == strcmp( "1.8", argv[i]))
                    options.export_spec = mdl_spec_1_8;
                else if ( 0 == strcmp( "1.9", argv[i]))
                    options.export_spec = mdl_spec_1_9;
                else {
                    std::cerr << "Error: unknown argument for -export-spec parameter <v>.\n";
                    usage();
                }
            } else {
                std::cerr << "Error: command line argument -export-spec misses <v> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-dist-supp-file", argv[i])) {
            ++i;
            if ( i < argc) {
                options.dist_supp_file = argv[i];
            } else {
                std::cerr << "Error: command line argument -dist-supp-file misses <name> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-bake", argv[i])) {
            ++i;
            if ( i < argc) {
                options.bake = std::atoi( argv[i]);
            } else {
                std::cerr << "Error: command line argument -bake misses <n> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-all-textures", argv[i])) {
            options.all_textures = true;
        } else if (0 == strcmp( "-texture-dir", argv[i])) {
            ++i;
            if ( i < argc) {
                options.texture_dir = argv[i];
            } else {
                std::cerr << "Error: command line argument -texture-dir misses <path> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-plugin", argv[i])) {
            ++i;
            if ( i < argc) {
                plugins.push_back( argv[i]);
            } else {
                std::cerr << "Error: command line argument -plugin misses <filename> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-no-std-plugin", argv[i])) {
             no_std_plugin= true;
        }
        else if (0 == strcmp("-test", argv[i])) {
            options.test_suite = true;
            options.trace = 1;
            ++i;
            if (i < argc) {
                if (0 == strcmp("none", argv[i])) {
                    options.spec_test = false;
                    options.test_module = false;
                    options.test_suite = false;
                }
                else if (0 == strcmp("normal", argv[i])) {
                    options.spec_test = false;
                }
                else if (0 == strcmp("spec", argv[i])) {
                    options.spec_test = true;
                }
                else {
                    std::cerr << "Error: command line argument -test misses <type> value.\n";
                    usage();
                }
            }
        } else if (0 == strcmp("-test-targets", argv[i])) {
            ++i;
            if (i < argc) {
                std::string tmp(argv[i]);
                std::vector<std::string> test_targets;
                size_t from = 0;
                while (true) {
                    size_t pos = tmp.find(',', from);
                    if (pos == std::string::npos) {
                        if (from < tmp.size()) {
                            test_targets.push_back(tmp.substr(from, tmp.size() - from));
                        }
                        break;
                    }
                    else {
                        if (pos > from) {
                            test_targets.push_back(tmp.substr(from, pos - from));
                        }
                        from = pos + 1;
                    }
                }
                if (test_targets.size() == 0) {
                    std::cerr << "Error: command line argument -test-targets requires "
                        "at least one target.\n";
                    usage();
                }
                options.test_targets = test_targets;
            } else {
                std::cerr << "Error: command line argument -test_log misses <path> value.\n";
                usage();
            }
        } else if (0 == strcmp("-test_log", argv[i])) {
            ++i;
            if (i < argc) {
                options.test_dir = argv[i];
            }
            else {
                std::cerr << "Error: command line argument -test_log misses <path> value.\n";
                usage();
            }
        } else if (0 == strcmp( "-tmm", argv[i])) {
            options.target_material_model_mode = true;
        } else if (0 == strcmp( "-tm", argv[i])) {
            ++i;
            if ( i < argc) {
                options.additional_modules.push_back( argv[i]);
            } else {
                std::cerr << "Error: command line argument -tm misses <path> value.\n";
                usage();
            }
        } else if (argv[i][0] == '-') {
            std::cerr << "Error: unknown command line argument '" << argv[i] << "'.\n";
            usage();
        } else {
            // handle all other non-option arguments
            args.push_back( argv[i]);
        }
    }

    // Create an instance of our logger
    Handle<ILogger> logger(
        new Logger(options.verbosity, options.test_suite, options.debug_print,
                   options.test_dir));

    // Access the MDL SDK
    Neuray_factory neuray( logger.get());
    check_success( neuray.get_result_code() == Neuray_factory::RESULT_SUCCESS);

    // Configure the MDL SDK library
    configuration( neuray.get(), logger.get(), &options, plugins, no_std_plugin);

    // Start the MDL SDK
    check_success( neuray.get()->start() == 0);

    // List all target modes option
    if ( list_modes) {
        list_all_targets(neuray.get());
        return 0;
    }

    // List all target modes option
    if ( list_required_modules) {
        list_all_required_modules(neuray.get());
        return 0;
    }

    // Check for material argument
    if ( args.size() != 1) {
        std::cerr << "Error: wrong number of arguments.\n";
        usage();
    }
    const char* material_name = args[0];

    for (auto& test_target : options.test_targets) {
        // Check for valid target names for all test targets from the command line
        if (!check_target(neuray.get(), test_target.c_str())) {
            std::cerr << "Error: unknown target '" << test_target 
                << "' for command line argument -test-targets\n";
            usage();
        }

    }

    // Check for valid target name
    if ( ! check_target( neuray.get(), target)) {
        std::cerr << "Error: unknown target '" << target << "' for command line argument -m\n";
        usage();
    }

    // Load an MDL module and dump its contents
    int result = mdl_distill_main( neuray.get(), material_name, target, &options);

    // Shut down the MDL SDK
    check_success( neuray.get()->shutdown() == 0);

    return result;
}

