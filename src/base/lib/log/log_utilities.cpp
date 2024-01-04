/***************************************************************************************************
 * Copyright (c) 2010-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "i_log_utilities.h"
#include "i_log_logger.h"


namespace MI {
namespace LOG {

static const int NR_FATAL   = Mod_log::S_FATAL | Mod_log::S_ASSERT;
static const int NR_ERROR   = NR_FATAL   | Mod_log::S_ERROR;
static const int NR_WARNING = NR_ERROR   | Mod_log::S_WARNING;
static const int NR_INFO    = NR_WARNING | Mod_log::S_INFO | Mod_log::S_PROGRESS;
static const int NR_VERBOSE = NR_INFO    | Mod_log::S_VERBOSE;


mi::base::Message_severity convert_severity( int severity)
{
    // Note: this implementation handles single values like S_ERROR (needed by Receiving_logger)
    // as well as bitmasks like S_FATAL | S_ERROR (needed by Logging_configuration_impl).
    if( (severity & ~NR_FATAL  ) == 0) return mi::base::MESSAGE_SEVERITY_FATAL;
    if( (severity & ~NR_ERROR  ) == 0) return mi::base::MESSAGE_SEVERITY_ERROR;
    if( (severity & ~NR_WARNING) == 0) return mi::base::MESSAGE_SEVERITY_WARNING;
    if( (severity & ~NR_INFO   ) == 0) return mi::base::MESSAGE_SEVERITY_INFO;
    if( (severity & ~NR_VERBOSE) == 0) return mi::base::MESSAGE_SEVERITY_VERBOSE;
    return mi::base::MESSAGE_SEVERITY_DEBUG;
}


int convert_severity( mi::base::Message_severity severity)
{
    switch( severity) {
        case mi::base::MESSAGE_SEVERITY_FATAL:   return NR_FATAL;
        case mi::base::MESSAGE_SEVERITY_ERROR:   return NR_ERROR;
        case mi::base::MESSAGE_SEVERITY_WARNING: return NR_WARNING;
        case mi::base::MESSAGE_SEVERITY_INFO:    return NR_INFO;
        case mi::base::MESSAGE_SEVERITY_VERBOSE: return NR_VERBOSE;
        case mi::base::MESSAGE_SEVERITY_DEBUG:   return Mod_log::S_ALL;
        default:                                 return -1;
    }
}

}}
