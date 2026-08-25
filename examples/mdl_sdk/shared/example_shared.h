/******************************************************************************
 * Copyright (c) 2012-2026, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/shared/example_shared.h
//
// Code shared by all examples

#ifndef EXAMPLE_SHARED_H
#define EXAMPLE_SHARED_H

#include <cstdio>
#include <cstdlib>
#include <string>

#include "utils/io.h"
#include "utils/loading.h"
#include "utils/main_utf8.h"
#include "utils/mdl.h"
#include "utils/os.h"
#include "utils/sdk_strings.h"


/// called to abort the execution of an example in case of failure.
/// \param  file        current file determined using the `__FILE__` macro
/// \param  line        current line in file determined using the `__LINE__` macro
/// \param  message     description of the error that caused the failure
inline void exit_failure_(
    const char* file, int line,
    std::string message)
{
    // print message
    if (message.empty())
        fprintf(stderr, "Fatal error in file: %s line: %d\n\nClosing the example.\n", file, line);
    else
        fprintf(stderr, "Fatal error in file: %s line: %d\n  %s\n\nClosing the example.\n",
            file, line, message.c_str());

    // keep console open
    #ifdef MI_PLATFORM_WINDOWS
        if (IsDebuggerPresent()) {
            fprintf(stderr, "Press enter to continue . . . \n");
            fgetc(stdin);
        }
    #endif

    // kill the application
    exit(EXIT_FAILURE);
}

/// called to abort the execution of an example in case of failure.
/// \param  file        current file determined using the `__FILE__` macro
/// \param  line        current line in file determined using the `__LINE__` macro
inline void exit_failure_(const char* file, int line)
{
    exit_failure_(file, line, "");
}
#define exit_failure(...) \
    exit_failure_(__FILE__, __LINE__, mi::examples::strings::format(__VA_ARGS__))


// ------------------------------------------------------------------------------------------------

/// called to end execution of an example in case of success.
/// use like this: 'return exit_success()'
inline void exit_success_()
{
    // keep console open
    #ifdef MI_PLATFORM_WINDOWS
        if (IsDebuggerPresent()) {
            fprintf(stderr, "\nPress enter to continue . . . \n");
            fgetc(stdin);
        }
    #endif
}

#define exit_success() exit_success_(); return EXIT_SUCCESS;

// ------------------------------------------------------------------------------------------------

// Helper macro. Checks whether the expression is true and if not prints a message and exits.
#define check_success( expr) \
    do { \
        if( !(expr)) \
            exit_failure( "%s", #expr); \
    } while( false)

/// Prints a message (std::string).
inline void print_message(
    mi::base::details::Message_severity severity,
    mi::neuraylib::IMessage::Kind kind,
    std::string msg,
    const std::string& file = {},
    int line = 0)
{
    std::string kind_str = mi::examples::strings::to_string(kind);

    if (!file.empty()) {
        msg.append("\n                file: ");
        msg.append(file);
        msg.append(", line: ");
        msg.append(std::to_string(line));
    }

    if (mi::examples::mdl::g_logger) {
        mi::examples::mdl::g_logger->message(severity, kind_str.c_str(), msg.c_str());
    } else {
        std::string severity_str = mi::examples::strings::to_string(severity);
        fprintf(stderr, "%s: %s %s\n", severity_str.c_str(), kind_str.c_str(), msg.c_str());
    }
}

/// Prints a message (std::string, without explicit kind parameter).
inline void print_message(
    mi::base::details::Message_severity severity,
    const std::string& msg,
    const std::string& file = {},
    int line = 0)
{
    print_message(severity, mi::neuraylib::IMessage::MSG_INTEGRATION, msg, file, line);
}

/// Prints the messages of the given context.
/// Returns true, if the context does not contain any error messages, false otherwise.
inline bool print_messages(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0, n = context->get_messages_count(); i < n; ++i) {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        print_message(message->get_severity(), message->get_kind(), message->get_string());
    }
    return context->get_error_messages_count() == 0;
}

#endif // MI_EXAMPLE_SHARED_H
