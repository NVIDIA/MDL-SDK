/******************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test public/mi/base components
///
/// See \ref mi_base_ilogger
///

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>

// suppress abort for fatal message severity
#define MI_BASE_DEFAULT_LOGGER_ABORT

#include <mi/base/types.h>
#include <mi/base/enums.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>
#include <cstdio>

#ifndef MI_BASE_DEFAULT_LOGGER_ABORT
#include <cstdlib>

/// A #mi::base::Message_severity of level #mi::base::MESSAGE_SEVERITY_FATAL
/// in the #mi::base::Default_logger::message implementation aborts the program
/// execution using this macro #MI_BASE_DEFAULT_LOGGER_ABORT.
/// This macro is by default set to std::abort() and you can override its
/// definition to customize abort behavior. Defining it to be empty will let
/// the program execution continue.
#define MI_BASE_DEFAULT_LOGGER_ABORT abort()
#endif // MI_BASE_DEFAULT_LOGGER_ABORT

namespace mi
{

namespace base
{

/** A default logger implementation based on fprintf to stderr.

    This implementation realizes the singleton pattern. An instance of the
    default logger can be obtained through the static inline method
    #mi::base::Default_logger::get_instance().

    A #mi::base::Message_severity of level #mi::base::MESSAGE_SEVERITY_FATAL
    aborts the program execution using the macro #MI_BASE_DEFAULT_LOGGER_ABORT.
    This macro is by default set to std::abort() and you can override its
    definition to customize abort behavior. Defining it to be empty will let
    the program execution continue.

       \par Include File:
       <tt> \#include <mi/base/default_logger.h></tt>

*/
class Default_logger : public Interface_implement_singleton<ILogger>
{
    Default_logger() = default;
    Default_logger( const Default_logger&) {}

public:
    void message(
        Message_severity level,
        const char* module_category,
        const mi::base::Message_details&,
        const char* message) final
    {
        const char* level_str = "";
        switch ( level) {
            case MESSAGE_SEVERITY_FATAL:
                level_str = "Fatal";
                break;
            case MESSAGE_SEVERITY_ERROR:
                level_str = "Error";
                break;
            case MESSAGE_SEVERITY_WARNING:
                level_str = "Warning";
                break;
            case MESSAGE_SEVERITY_INFO:
                level_str = "Info";
                break;
            case MESSAGE_SEVERITY_VERBOSE:
                level_str = "Verbose";
                break;
            case MESSAGE_SEVERITY_DEBUG:
                level_str = "Debug";
                break;
            default:
                break;
        }
        std::fprintf( stderr, "%s: [%s] %s\n", level_str, module_category, message);
        if( level == MESSAGE_SEVERITY_FATAL) {
            MI_BASE_DEFAULT_LOGGER_ABORT;
        }
    }

    /// Returns the single instance of the default logger.
    static ILogger* get_instance()
    {
        static Default_logger logger;
        return &logger;
    }
};

} // namespace base
} // namespace mi

MI_TEST_AUTO_FUNCTION( test_logger )
{
    mi::base::ILogger* log = mi::base::Default_logger::get_instance();
    log->message( mi::base::MESSAGE_SEVERITY_FATAL,   "TEST", "Test fatal severity");
    log->message( mi::base::MESSAGE_SEVERITY_ERROR,   "TEST", "Test error severity");
    log->message( mi::base::MESSAGE_SEVERITY_WARNING, "TEST", "Test warning severity");
    log->message( mi::base::MESSAGE_SEVERITY_INFO,    "TEST", "Test info severity");
    log->message( mi::base::MESSAGE_SEVERITY_VERBOSE, "TEST", "Test verbose severity");
}
