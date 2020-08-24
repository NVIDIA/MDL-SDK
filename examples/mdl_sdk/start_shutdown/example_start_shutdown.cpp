/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/start_shutdown/example_start_shutdown.cpp
//
// Obtain an INeuray interface, start the MDL SDK and shut it down.

// Include code shared by all examples.
#include "example_shared.h"

// The main function initializes the MDL SDK, starts it, and shuts it down after waiting for
// user input.
int MAIN_UTF8( int /*argc*/, char* /*argv*/[])
{
    // Get the INeuray interface in a suitable smart pointer.
    mi::base::Handle<mi::neuraylib::INeuray> neuray( mi::examples::mdl::load_and_get_ineuray());
    if ( !neuray.is_valid_interface())
        exit_failure("Error: The MDL SDK library failed to load and to provide "
                     "the mi::neuraylib::INeuray interface.");

    // Print library version information.
    mi::base::Handle<const mi::neuraylib::IVersion> version(
        neuray->get_api_component<const mi::neuraylib::IVersion>());

    fprintf( stderr, "MDL SDK header version          = %s\n",
        MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    fprintf( stderr, "MDL SDK library product name    = %s\n", version->get_product_name());
    fprintf( stderr, "MDL SDK library product version = %s\n", version->get_product_version());
    fprintf( stderr, "MDL SDK library build number    = %s\n", version->get_build_number());
    fprintf( stderr, "MDL SDK library build date      = %s\n", version->get_build_date());
    fprintf( stderr, "MDL SDK library build platform  = %s\n", version->get_build_platform());
    fprintf( stderr, "MDL SDK library version string  = \"%s\"\n", version->get_string());

    mi::base::Uuid neuray_id_libraray = version->get_neuray_iid();
    mi::base::Uuid neuray_id_interface = mi::neuraylib::INeuray::IID();

    fprintf( stderr, "MDL SDK header interface ID     = <%2x, %2x, %2x, %2x>\n",
        neuray_id_interface.m_id1,
        neuray_id_interface.m_id2,
        neuray_id_interface.m_id3,
        neuray_id_interface.m_id4);
    fprintf( stderr, "MDL SDK library interface ID    = <%2x, %2x, %2x, %2x>\n\n",
        neuray_id_libraray.m_id1,
        neuray_id_libraray.m_id2,
        neuray_id_libraray.m_id3,
        neuray_id_libraray.m_id4);

    version = 0;

    // configuration settings go here, none in this example,
    // but for a standard initialization the other examples use this helper function:
    // if ( !mi::examples::mdl::configure(neuray.get()))
    //     exit_failure("Failed to initialize the SDK.");

    // After all configurations, the MDL SDK is started. A return code of 0 implies success. The
    // start can be blocking or non-blocking. Here the blocking mode is used so that you know that
    // the MDL SDK is up and running after the function call. You can use a non-blocking call to do
    // other tasks in parallel and check with
    //
    //      neuray->get_status() == mi::neuraylib::INeuray::STARTED
    //
    // if startup is completed.
    mi::Sint32 result = neuray->start( true);
    if ( result != 0)
        exit_failure( "Failed to initialize the SDK. Result code: %d", result);

    // scene graph manipulations and rendering calls go here, none in this example.
    // ...

    // Shutting the MDL SDK down in blocking mode. Again, a return code of 0 indicates success.
    if (neuray->shutdown( true) != 0)
        exit_failure( "Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr; // free the handles that holds the INeuray instance
    if ( !mi::examples::mdl::unload())
        exit_failure( "Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
