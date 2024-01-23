/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/distilling/example_distilling_target.cpp
//
// Introduces the distillation of mdl using a custom distilling target.

#include "example_shared.h"
#include "example_shared_dump.h"

// Custom timing output
#include <chrono>

struct Timing
{
    explicit Timing(std::string operation)
        : m_operation(operation)
    {
        m_start = std::chrono::steady_clock::now();
    }

    ~Timing()
    {
        auto stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = stop - m_start;
        printf("info:  %s time: %.2f ms.\n",
               m_operation.c_str(),
               elapsed_seconds.count() * 1000);
    }

private:
    std::string m_operation;
    std::chrono::steady_clock::time_point m_start;
};

// Creates an instance of the given material.
mi::neuraylib::IFunction_call* create_material_instance(
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_execution_context* context,
    const std::string& module_qualified_name,
    const std::string& material_simple_name)
{
    // Load the module.
    mdl_impexp_api->load_module(transaction, module_qualified_name.c_str(), context);
    if (!print_messages(context))
        exit_failure("Loading module '%s' failed.", module_qualified_name.c_str());

    // Get the database name for the module we loaded
    mi::base::Handle<const mi::IString> module_db_name(
        mdl_factory->get_db_module_name(module_qualified_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
        exit_failure("Failed to access the loaded module.");

    // Attach the material name
    std::string material_db_name
        = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
    material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), material_db_name);
    if (material_db_name.empty())
        exit_failure("Failed to find the material %s in the module %s.",
            material_simple_name.c_str(), module_qualified_name.c_str());

    // Get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Accessing definition '%s' failed.", material_db_name.c_str());

    // Create a material instance from the material definition with
    // the default arguments.
    mi::Sint32 result;
    mi::neuraylib::IFunction_call* material_instance =
        material_definition->create_function_call(0, &result);
    if (result != 0)
        exit_failure("Instantiating '%s' failed.", material_db_name.c_str());

    return material_instance;
}

// Compiles the given material instance in the given compilation modes
// and stores it in the DB.
mi::neuraylib::ICompiled_material* compile_material_instance(
    const mi::neuraylib::IFunction_call* material_instance,
    mi::neuraylib::IMdl_execution_context* context,
    bool class_compilation)
{
    Timing timing("Compiling");
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::neuraylib::ICompiled_material* compiled_material =
        material_instance2->create_compiled_material(flags, context);
    check_success(print_messages(context));

    return compiled_material;
}

// Distills the given compiled material to the requested target model,
// and returns it
const mi::neuraylib::ICompiled_material* create_distilled_material(
    mi::neuraylib::IMdl_distiller_api* distiller_api,
    const mi::neuraylib::ICompiled_material* compiled_material,
    const char* target_model)
{
    mi::Sint32 result = 0;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_material(
        distiller_api->distill_material(compiled_material, target_model, nullptr, &result));
    std::cerr << "result: " << result << "\n";
    check_success(result == 0);

    distilled_material->retain();
    return distilled_material.get();
}

// Prints program usage
static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1> ...]\n"
        << "-h                        print this text\n"
        << "--material_file <file>    file containing fully qualified names of materials to distill\n"
        << "--module <module_name>    distill all materials from the module, can occur multiple times\n"
        << "--mdl_path <path>         mdl search path, can occur multiple times.\n"
        << "--dump-original-material  dump structure of distilled material\n"
        << "--dump-distilled-material dump structure of distilled material\n";

    exit(EXIT_FAILURE);
}

void load_materials_from_file(const std::string & material_file, std::vector<std::string> & material_names)
{
    std::fstream file;
    file.open(material_file, std::fstream::in);
    if (!file)
    {
        std::cout << "Invalid file: " + material_file;
        return;
    }
    std::string fn;
    while (getline(file, fn))
    {
        material_names.emplace_back(fn);
    }
    file.close();
}

void load_materials_from_modules(
    mi::neuraylib::IMdl_factory * mdl_factory
    , mi::neuraylib::ITransaction * transaction
    , mi::neuraylib::IMdl_impexp_api * mdl_impexp_api
    , const std::vector<std::string> & module_names
    , std::vector<std::string> & material_names)
{
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    for (auto module_name : module_names)
    {
        // Sanity check
        if (module_name.find("::") != 0)
        {
            module_name = std::string("::") + module_name;
        }
        mi::Sint32 rtn(mdl_impexp_api->load_module(transaction, module_name.c_str(), context.get()));
        check_success(rtn == 0 || rtn == 1);
        mi::base::Handle<const mi::IString> db_module_name(
            mdl_factory->get_db_module_name(module_name.c_str()));
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>(db_module_name->get_c_str()));
        check_success(module.is_valid_interface());
        mi::Size material_count = module->get_material_count();
        for (mi::Size i = 0; i < material_count; i++)
        {
            std::string mname(module->get_material(i));
            mi::base::Handle<const mi::neuraylib::IFunction_definition> material(
                transaction->access<mi::neuraylib::IFunction_definition>(mname.c_str()));
            material_names.push_back(material->get_mdl_name());
        }
    }
}

int MAIN_UTF8(int argc, char* argv[])
{
    std::string target_model = "mini_glossy";
    std::vector<std::string> material_names;
    std::vector<std::string> module_names;
    std::string material_file;
    bool dump_distilled_material = false;
    bool dump_original_material = false;

    Timing timing("Elapsed");
    mi::examples::mdl::Configure_options configure_options;

    // Collect command line arguments, if any.
    for (int i = 1; i < argc; ++i) {
        const char *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--mdl_path") == 0) {
                if (i < argc - 1)
                    configure_options.additional_mdl_paths.push_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--material_file") == 0) {
                if (i < argc - 1)
                    material_file = argv[++i];
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--module") == 0) {
                if (i < argc - 1)
                    module_names.emplace_back(argv[++i]);
                else
                    usage(argv[0]);
            }
            else if (strcmp(opt, "--dump-distilled-material") == 0) {
                dump_distilled_material = true;
            }
            else if (strcmp(opt, "--dump-original-material") == 0) {
                dump_original_material = true;
            }
            else
                usage(argv[0]);
        }
        else
            material_names.push_back(opt);
    }
    if (!material_file.empty())
        load_materials_from_file(material_file, material_names);

    // Access the MDL SDK.
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK.
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Load the custom example distilling plugin.
    if (mi::examples::mdl::load_plugin(neuray.get(),
                                       "distilling_target_plugin" MI_BASE_DLL_FILE_EXT) != 0)
        exit_failure("Failed to load the 'distilling_target_plugin' plugin.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        // Get MDL factory.
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // Create a transaction.
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        if (!module_names.empty())
        {
            load_materials_from_modules(mdl_factory.get(), transaction.get(), mdl_impexp_api.get(), module_names, material_names);
        }
        if (material_names.empty())
        {
            material_names.push_back(
                "::nvidia::sdk_examples::tutorials_distilling::example_distilling3");
        }
        for (const auto& m : material_names)
        {
            // split module and material name
            std::string module_qualified_name, material_simple_name;
            if (!mi::examples::mdl::parse_cmd_argument_material_name(
                m, module_qualified_name, material_simple_name, true))
                exit_failure();

            // Create an execution context
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            // Load mdl module and create a material instance
            mi::base::Handle<mi::neuraylib::IFunction_call> instance(
                create_material_instance(
                    mdl_factory.get(),
                    transaction.get(),
                    mdl_impexp_api.get(),
                    context.get(),
                    module_qualified_name,
                    material_simple_name));

            // Compile the material instance
            mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
                compile_material_instance(instance.get(), context.get(), false));

            if (dump_original_material) {
                std::cout << "[[ Original material: " << m << " ]]\n";
                mi::examples::mdl::dump_compiled_material(transaction.get(),
                                                          mdl_factory.get(),
                                                          compiled_material.get(),
                                                          std::cout);
            }
            // Acquire distilling API used for material distilling and baking
            mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distilling_api(
                neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

            if (!distilling_api.is_valid_interface()) {
                exit_failure("Failed to obtain distiller component.");
            }
            // Distill compiled material to "mini_glossy" material model
            mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_material(
                create_distilled_material(
                    distilling_api.get(),
                    compiled_material.get(),
                    target_model.c_str()));

            if (dump_distilled_material) {
                std::cout << "[[ Distilled material: " << m << " (distilled to " <<
                    target_model << ") ]]\n";
                mi::examples::mdl::dump_compiled_material(transaction.get(),
                                                          mdl_factory.get(),
                                                          distilled_material.get(),
                                                          std::cout);
            }
        }
        transaction->commit();
    }

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
