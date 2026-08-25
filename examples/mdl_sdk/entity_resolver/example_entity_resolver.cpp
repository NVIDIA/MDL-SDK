/******************************************************************************
 * Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/compilation/example_entity_resolver.cpp
//
// Demonstrates a simplified implementation of a custom entity resolver.

#include <iostream>
#include <string>

#include "example_shared.h"
#include "entity_resolver.h"

// Command line options structure.
struct Options {
    // Flag that indicates to use a virtual file system as file system abstraction.
    bool virtual_fs = false;
    // Flag that indicates to trace resolve requests and results.
    bool trace = false;
    // Module to load.
    std::string module_name = "::nvidia::sdk_examples::entity_resolver::package1::main_mdl_1_6";
};

void usage( const char* prog_name)
{
    Options defaults;
    std::cout
        << "Usage: " << prog_name << " [options] [<module_name>]\n"
        << "Options:\n"
        << "  -p|--mdl_path <path>   mdl search path, can occur multiple times\n"
        << "  --trace                trace resolve requests and results\n"
        << "  --virtual_fs           use a virtual filesystem with the entity resolver\n"
        << "  <module_name>          qualified name of the module to load, defaults to\n"
        << "                         \"" << defaults.module_name << "\""
        << std::endl;
    exit_failure();
}

void enumerate_resources(
    mi::neuraylib::ITransaction* transaction, const mi::neuraylib::IModule* module)
{
    mi::Size n_resources = module->get_resources_count();
    for( mi::Size i = 0; i < n_resources; ++i) {

        mi::base::Handle<const mi::neuraylib::IValue_resource> resource(
            module->get_resource( i));

        mi::base::Handle<const mi::neuraylib::IType_resource> resource_type(
            resource->get_type());
        check_success( resource_type);

        const char* file_path = resource->get_file_path();
        std::cout << "resource " << i
                  << ": file path \"" << (file_path ? file_path : "(null)");

        const char* resource_name = resource->get_value();
        std::cout << "\", DB name \"" << (resource_name ? resource_name : "(null)");

        if( !resource_name) {
            std::cout << "\"" << std::endl;
            continue;
        }

        switch( resource_type->get_kind()) {

            case mi::neuraylib::IType::TK_TEXTURE: {
                std::cout << "\", kind \"texture";

                mi::base::Handle<const mi::neuraylib::ITexture> texture(
                    transaction->access<mi::neuraylib::ITexture>( resource_name));
                check_success( texture);

                const char* image_name = texture->get_image();
                std::cout << "\", image DB name \"" << (image_name ? image_name : "(null)");
                check_success( image_name);

                mi::base::Handle<const mi::neuraylib::IImage> image(
                    transaction->access<mi::neuraylib::IImage>( image_name));
                check_success( image);

                const char* filename = image->get_filename( 0, 0);
                std::cout << "\", filename \"" << (filename ? filename : "(null)")
                          << "\"" << std::endl;
                break;
            }

            case mi::neuraylib::IType::TK_LIGHT_PROFILE: {
                std::cout << "\", kind \"light profile";

                mi::base::Handle<const mi::neuraylib::ILightprofile> light_profile(
                    transaction->access<mi::neuraylib::ILightprofile>( resource_name));
                check_success( light_profile);

                const char* filename = light_profile->get_filename();
                std::cout << "\", filename \"" << (filename ? filename : "(null)")
                          << "\"" << std::endl;
                break;
            }

            case mi::neuraylib::IType::TK_BSDF_MEASUREMENT: {
                std::cout << "\", kind \"BSDF measurement";

                mi::base::Handle<const mi::neuraylib::IBsdf_measurement> bsdf_measurement(
                    transaction->access<mi::neuraylib::IBsdf_measurement>( resource_name));
                check_success( bsdf_measurement);

                const char* filename = bsdf_measurement->get_filename();
                std::cout << "\", filename \"" << (filename ? filename : "(null)")
                          << "\"" << std::endl;
                break;
            }

            default:
                std::cout << "\"" << std::endl;
                exit_failure(
                    "Unexpected resource type: %u",
                    static_cast<unsigned int>( resource_type->get_kind()));
        }
    }
}

int MAIN_UTF8( int argc, char* argv[])
{
    // Parse command line options
    Options options;
    mi::examples::mdl::Configure_options configure_options;

    for( int i = 1; i < argc; ++i) {
        std::string s =  argv[i];
        if( s[0] == '-') {
            if( (s == "-p" || s == "--mdl_path") && (i < argc - 1)) {
                configure_options.additional_mdl_paths.emplace_back( argv[++i]);
            } else if( s == "--trace") {
                options.trace = true;
            } else if( s == "--virtual_fs") {
                options.virtual_fs = true;
            } else {
                std::cout << "Unknown option: \"" << s << "\"" << std::endl;
                usage( argv[0]);
            }
        } else {
            options.module_name = s;
        }
    }

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray( mi::examples::mdl::load_and_get_ineuray());
    if( !neuray)
        exit_failure( "Failed to load the SDK.");

    // Configure the MDL SDK
    if( !mi::examples::mdl::configure( neuray.get(), configure_options))
        exit_failure( "Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if( ret != 0)
        exit_failure( "Failed to initialize the SDK. Result code: %d", ret);

    {
        // Create the file system abstraction.
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        mi::base::Handle<IFile_system> file_system;
        const std::string virtual_fs_prefix = "/virtual_fs";

        if( !options.virtual_fs) {

            file_system = new Os_file_system( mdl_impexp_api.get());

        } else {

            auto virtual_file_system = new Virtual_file_system( mdl_impexp_api.get());
            file_system = virtual_file_system;

            // For simplicity, populate the virtual file system with files from the actual file
            // system. In the virtual file system these files have \c virtual_fs_prefix as prefix,
            // which is also applied to the search paths passed to the entity resolver.
            std::string vfs_root = mi::examples::mdl::get_examples_root() + "/mdl";
            const char* files[] = {
                "/nvidia/sdk_examples/entity_resolver/package1/main_mdl_1_3.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/main_mdl_1_6.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/package2/down.png",
                "/nvidia/sdk_examples/entity_resolver/package1/package2/down1.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/package2/down2.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/package2/down3.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/package2/down4.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/same.png",
                "/nvidia/sdk_examples/entity_resolver/package1/same1.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/same2.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/same3.mdl",
                "/nvidia/sdk_examples/entity_resolver/package1/same4.mdl",
                "/nvidia/sdk_examples/entity_resolver/up.png",
                "/nvidia/sdk_examples/entity_resolver/up1.mdl",
                "/nvidia/sdk_examples/entity_resolver/up2.mdl",
                "/nvidia/sdk_examples/entity_resolver/up3.mdl",
            };

            for( const auto& file: files) {
                std::string s = vfs_root + file;
                virtual_file_system->add_file( s.c_str(), virtual_fs_prefix.c_str());
            }

        }

        // Create the custom entity resolver.
        mi::base::Handle<Mdl_entity_resolver> mdl_entity_resolver(
            new Mdl_entity_resolver( file_system.get(), options.trace));

        // Pass search paths to the entity resolver.
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_configuration(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
        mi::Size n = mdl_configuration->get_mdl_paths_length();
        for( mi::Size i = 0; i < n; ++i) {
            mi::base::Handle<const mi::IString> mdl_path( mdl_configuration->get_mdl_path( i));
            std::string modified_mdl_path
                = (options.virtual_fs ? virtual_fs_prefix : std::string()) + mdl_path->get_c_str();
            mdl_entity_resolver->add_search_path( modified_mdl_path.c_str());
            std::cout << "search path " << i << ": " << modified_mdl_path << std::endl;
        }

        // Install the entity resolver.
        mdl_configuration->set_entity_resolver( mdl_entity_resolver.get());

        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

        {
            // Create an execution context for options and error message handling
            mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
                neuray->get_api_component<mi::neuraylib::IMdl_factory>());
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            // Load the module
            mdl_impexp_api->load_module(
                transaction.get(), options.module_name.c_str(), context.get());
            if( !print_messages( context.get()))
                exit_failure( "Loading module '%s' failed.", options.module_name.c_str());

            // Access the module.
            mi::base::Handle<const mi::IString> module_db_name(
                mdl_factory->get_db_module_name( options.module_name.c_str()));
            mi::base::Handle<const mi::neuraylib::IModule> module(
                transaction->access<mi::neuraylib::IModule>( module_db_name->get_c_str()));
            std::cout << "module \"" << module->get_mdl_name()
                      << "\", file path \"" << module->get_filename() << "\"" << std::endl;

            // Enumerate the referenced resources.
            enumerate_resources( transaction.get(), module.get());

            module.reset();

            transaction->commit();
        }

        // Deinstall the entity resolver.
        mdl_configuration->set_entity_resolver( nullptr);
    }

    // Shut down the MDL SDK
    if( neuray->shutdown() != 0)
        exit_failure( "Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if( !mi::examples::mdl::unload())
        exit_failure( "Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
