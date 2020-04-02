/***************************************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

#include "logger.h"
using namespace mdlm;

const char* Logger::get_log_level(mi::base::Message_severity level) 
{
    switch (level) {
    case mi::base::MESSAGE_SEVERITY_FATAL:
        return "fatal";
    case mi::base::MESSAGE_SEVERITY_ERROR:
        return "error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return "warn";
    case mi::base::MESSAGE_SEVERITY_INFO:
        return "info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return "verb";
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
    if (int(level) < m_level) 
    {
        const char* severity = get_log_level(level);
        // We want to ignore module_category
        // to avoid confusion between different modules (MDL, MDLM, ...)
        fprintf(stderr, "%s:\t%s\n", severity, message);
    }
}
