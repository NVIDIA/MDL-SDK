/******************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/archives/example_archives.cpp
//
// Creates an MDL archive, extracts an MDL archive, or queries the manifest of an MDL archive.
//
// The example expects the following command line arguments:
//
//   example_archives create <directory> <archive> <mdl_path> [<key=value> ...]
//   example_archives extract <archive> <directory>
//   example_archives query_count <archive>
//   example_archives query_key <archive> <index>
//   example_archives query_value <archive> <index>
//   example_archives query_key_count <archive> <key>
//   example_archives query_key_value <archive> <key> <index>
//
// directory   Directory to create an archive from or to extract an existing archive into
// archive     File name of the archive to create, to extract, or to query its manifest
// key=value   Optional or user-defined fields to add to the manifest
// index       Index of a manifest field
// key         Key of a manifest field
//
// For example: example_archives create archives main.mdr . foo=bar

#include <iostream>
#include <string>

// Include code shared by all examples.
#include "example_shared.h"

// Creates an MDL archive from a directory. Allows to add optional or user-defined fields to the
// manifest.
void create_archive(
    mi::neuraylib::IMdl_archive_api* mdl_archive_api,
    int argc,
    char* argv[],
    mi::neuraylib::INeuray* neuray)
{
    if( argc < 5) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected at least 4, "
            "got %d).\n", argv[1], argc-1);
        return;
    }

    const char* directory   = argv[2];
    const char* archive     = argv[3];
    const char* module_path = argv[4];
    mi::Size fields_count   = static_cast<mi::Size>( argc-5);

    // set the search path for .mdl files
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_conf(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    check_success( mdl_conf->add_mdl_path( module_path) == 0);

    // convert argv[5], etc. to manifest fields
    mi::base::Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    mi::base::Handle<mi::IDynamic_array> manifest_fields(
        factory->create<mi::IDynamic_array>( "Manifest_field[]"));
    manifest_fields->set_length( fields_count);
    for( mi::Size i = 0; i < fields_count; ++i) {
        std::string arg = argv[5+i];
        size_t pos = arg.find( "=");
        if( pos == 0 || pos == std::string::npos) {
            fprintf( stderr, "Wrong format for field \"%s\".\n", arg.c_str());
            return;
        }
        mi::base::Handle<mi::IStructure> field( manifest_fields->get_value<mi::IStructure>( i));
        mi::base::Handle<mi::IString> key( field->get_value<mi::IString>( "key"));
        mi::base::Handle<mi::IString> value( field->get_value<mi::IString>( "value"));
        key->set_c_str( arg.substr( 0, pos).c_str());
        value->set_c_str( arg.substr( pos+1).c_str());
    }

    mi::Sint32 result = mdl_archive_api->create_archive( directory, archive, manifest_fields.get());
    if( result < 0)
        fprintf( stderr, "Archive creation failed with error code %d.\n", result);
}

// Extracts an MDL archive into a directory.
void extract_archive( mi::neuraylib::IMdl_archive_api* mdl_archive_api, int argc, char* argv[])
{
    if( argc != 4) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected 3, got %d).\n",
            argv[1], argc-1);
        return;
    }

    const char* archive   = argv[2];
    const char* directory = argv[3];

    mi::Sint32 result = mdl_archive_api->extract_archive( archive, directory);
    if( result < 0)
        fprintf( stderr, "Archive extraction failed with error code %d.\n", result);
}

// Outputs the number of fields in a manifest.
void query_count( const mi::neuraylib::IManifest* manifest, int argc, char* argv[])
{
    if( argc != 3) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected 2, got %d)\n",
            argv[1], argc-1);
        return;
    }

    mi::Size count = manifest->get_number_of_fields();
    fprintf( stderr, "The manifest contains %" MI_BASE_FMT_MI_SIZE " fields.\n", count);
}

// Outputs the key of a manifest field identified by its index.
void query_key( const mi::neuraylib::IManifest* manifest, int argc, char* argv[])
{
    if( argc != 4) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected 3, got %d)\n",
            argv[1], argc-1);
        return;
    }

    mi::Size index = static_cast<mi::Size>( atoi( argv[3]));
    const char* key = manifest->get_key( index);
    if( !key) {
        fprintf( stderr, "Index %" MI_BASE_FMT_MI_SIZE " is out of bounds.\n", index);
        return;
    }

    fprintf( stderr, "The key of field %" MI_BASE_FMT_MI_SIZE " is \"%s\".\n", index, key);
}

// Outputs the value of a manifest field identified by its index.
void query_value( const mi::neuraylib::IManifest* manifest, int argc, char* argv[])
{
    if( argc != 4) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected 3, got %d)\n",
            argv[1], argc-1);
        return;
    }

    mi::Size index = static_cast<mi::Size>( atoi( argv[3]));
    const char* value = manifest->get_value( index);
    if( !value) {
        fprintf( stderr, "Index %" MI_BASE_FMT_MI_SIZE " is out of bounds.\n", index);
        return;
    }

    fprintf( stderr, "The value of field %" MI_BASE_FMT_MI_SIZE " is \"%s\".\n", index, value);
}

// Outputs the number of fields in a manifest for a given key.
void query_key_count( const mi::neuraylib::IManifest* manifest, int argc, char* argv[])
{
    if( argc != 4) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected 3, got %d)\n",
            argv[1], argc-1);
        return;
    }

    const char* key = argv[3];
    mi::Size count = manifest->get_number_of_fields( key);
    fprintf( stderr, "The manifest contains %" MI_BASE_FMT_MI_SIZE " fields for key \"%s\".\n",
        count, key);
}

// Outputs the value of a manifest field identified by its key and index.
void query_key_value( const mi::neuraylib::IManifest* manifest, int argc, char* argv[])
{
    if( argc != 5) {
        fprintf( stderr, "Wrong number of arguments for mode \"%s\" (expected 4, got %d)\n",
            argv[1], argc-1);
        return;
    }

    const char* key = argv[3];
    mi::Size index = static_cast<mi::Size>( atoi( argv[4]));
    const char* value = manifest->get_value( key, index);
    if( !value) {
        fprintf( stderr, "Index %" MI_BASE_FMT_MI_SIZE " for key \"%s\" is out of bounds.\n",
            index, key);
        return;
    }

    fprintf( stderr, "The value of field %" MI_BASE_FMT_MI_SIZE " for key \"%s\" is \"%s\".\n",
        index, key, value);
}

// Outputs various information from the manifest of an MDL archive.
void query( mi::neuraylib::IMdl_archive_api* mdl_archive_api, int argc, char* argv[])
{
    const char* archive = argv[2];
    mi::base::Handle<const mi::neuraylib::IManifest> manifest(
        mdl_archive_api->get_manifest( archive));
    if( !manifest) {
        fprintf( stderr, "Failed to retrieve manifest from \"%s\".\n", archive);
        return;
    }

    std::string mode = argv[1];
    if( mode == "query_count")
        query_count( manifest.get(), argc, argv);
    else if( mode == "query_key")
        query_key( manifest.get(), argc, argv);
    else if( mode == "query_value")
        query_value( manifest.get(), argc, argv);
    else if( mode == "query_key_count")
        query_key_count( manifest.get(), argc, argv);
    else if( mode == "query_key_value")
        query_key_value( manifest.get(), argc, argv);
    else
        fprintf( stderr, "Invalid mode \"%s\".\n", mode.c_str());
}

int MAIN_UTF8( int argc, char* argv[])
{
    // Collect command line parameters
    if( argc == 1) {
        std::cerr << "Usage: example_archives create <directory> <archive> <mdl_path> "
            "[key=value ...]" << std::endl;
        std::cerr << "       example_archives extract <archive> <directory>" << std::endl;
        std::cerr << "       example_archives query_count <archive>" << std::endl;
        std::cerr << "       example_archives query_key <archive> <index>" << std::endl;
        std::cerr << "       example_archives query_value <archive> <index>" << std::endl;
        std::cerr << "       example_archives query_key_count <archive> <key>" << std::endl;
        std::cerr << "       example_archives query_key_value <archive> <key> <index>" << std::endl;
        exit_failure();
    }
    std::string mode = argv[1];

    // Access the neuray library
    mi::base::Handle<mi::neuraylib::INeuray> neuray( mi::examples::mdl::load_and_get_ineuray());
    check_success( neuray.is_valid_interface());

    // Install logger
    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>());
    mi::base::Handle<mi::base::ILogger> logger( new mi::examples::mdl::Default_logger());
    mdl_config->set_logger( logger.get());
    mdl_config = 0;

    // Start the neuray library
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        mi::base::Handle<mi::neuraylib::IMdl_archive_api> mdl_archive_api(
            neuray->get_api_component<mi::neuraylib::IMdl_archive_api>());

        if( mode == "create")
            create_archive( mdl_archive_api.get(), argc, argv, neuray.get());
        else if( mode == "extract")
            extract_archive( mdl_archive_api.get(), argc, argv);
        else
            query( mdl_archive_api.get(), argc, argv);
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
