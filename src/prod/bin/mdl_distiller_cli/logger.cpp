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

#include <fstream>

#include "logger.h"
#include "options.h"

static const char* const severity_name[] = {
    "fatal", "error", "warn", "stat", "vstat", "progr", "info", "debug", "vdebg", "assrt" };


const char* Logger::get_log_level( mi::base::Message_severity level) {
    switch( level) {
        case mi::base::MESSAGE_SEVERITY_FATAL:
            return "fatal";
        case mi::base::MESSAGE_SEVERITY_ERROR:
            return "error";
        case mi::base::MESSAGE_SEVERITY_WARNING:
            return "warn ";
        case mi::base::MESSAGE_SEVERITY_INFO:
            return "info ";
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
            return "progr";
        case mi::base::MESSAGE_SEVERITY_DEBUG:
            return "debug";
        default:
            return "???";
    }
}

void Logger::message( 
    mi::base::Message_severity level,
    const char* module_category,
    const mi::base::Message_details&,
    const char* message)
{
    const char *message_text = message;
    if (0 == strncmp("info : ", message, 7)) {
        message_text = message + 7;
    }
    if (m_test_suite) {
        if (m_path.length() > 0) {
            if (0 == strncmp("Rule <", message_text, 6)) {
                // In test suite mode we trace rule matches in rule_matches.txt                
                std::ofstream rule_file((m_path + SLASH + RUID_FILE).c_str(),
                                        std::ios::out | std::ios::app);
                rule_file << "//RUID ";
                for (const char* s = message_text + 6; isdigit(*s); ++s)
                    rule_file << *s;
                rule_file << "\n";
                rule_file.close();
            } else {
                // In test suite mode we trace any logs in mdl_distiller.txt
                std::ofstream log_file((m_path + SLASH + LOG_FILE).c_str(), 
                                       std::ios::out | std::ios::app);
                log_file << "\nTESTSUITE " << message;
                log_file.close();
            }
        } else {
            fprintf(stderr, "Cannot access log path %s\n", m_path.c_str());
        }
    }

    // Detect exception: we show the distiller trace "Info" messages irresective
    // of the verbose level setting if trace is enabled.
    bool trace_message = false;
    if ( (0 == strncmp("Rule <", message_text, 6))
         || (0 == strncmp("Check rule set '", message_text, 16))
         || (m_debug_print && 0 == strncmp(">>> ", message_text, 4)))
        trace_message = true;

    if ( trace_message || (int(level) < m_level)) {
        const char* severity = get_log_level(level);// severity_name[ level];
        fprintf( stderr, "%s %s: %s\n", severity, module_category, message);
    }
}
