/******************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/generate_mdl_identifier/example_generate_mdl_identifier.cpp
//
// Generates a valid MDL identifier from the input.

#include <string>
#include <iostream>
#include <iomanip>

#include "example_shared.h"

/// Checks, if the given character is a valid MDL letter.
bool is_mdl_letter(char c)
{
    if ('A' <= c && c <= 'Z')
        return true;
    if ('a' <= c && c <= 'z')
        return true;
    return false;
}

/// Checks, if the given character is a valid MDL digit.
bool is_mdl_digit(char c)
{
    if ('0' <= c && c <= '9')
        return true;
    return false;
}

/// Demonstrates one way to convert the given string into a valid MDL identifier. Note that the
/// implemented mapping is not injective, e.g., "a_b" and "a%b" are both converted to "a_b".
std::string make_valid_mdl_identifier(
    mi::neuraylib::IMdl_factory* mdl_factory, const std::string& id)
{
    // Return "m" for empty input.
    if (id.empty())
        return "m";

    std::string result;
    result.reserve(id.size());

    // First, check general identifier rules:
    // IDENT = LETTER { LETTER | DIGIT | '_' } .

    // Replace leading underscore by 'm'.
    size_t index = 0;
    result.push_back(is_mdl_letter(id[index]) ? id[index] : 'm');

    // Replace sequences of invalid characters by a single underscore.
    for (index = 1; index < id.size(); ++index) {
        const char c = id[index];
        if (is_mdl_digit(c) || is_mdl_letter(c) || c == '_')
            result.push_back(c);
        else {
            if (result[result.size()-1] != '_')
                result.push_back('_');
        }
    }

    // Second, add prefix "m_" for MDL keywords.
    if (!mdl_factory->is_valid_mdl_identifier(result.c_str()))
        return "m_" + result;
    else
        return result;
}

void process(mi::neuraylib::INeuray* neuray, int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " <identifier_1> [<identifier_2> ...<identifier_n>]" << std::endl;
        return;
    }

    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    for (int i = 1; i < argc; ++i) {
        std::string result = make_valid_mdl_identifier(mdl_factory.get(), argv[i]);
        std::cout << std::left << std::setw(25) << argv[i] << " => " << result << std::endl;
    }
}

int MAIN_UTF8( int argc, char* argv[])
{
    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get()))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    // Process the command-line arguments
    process(neuray.get(), argc, argv);

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
