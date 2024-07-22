/***************************************************************************************************
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
 **************************************************************************************************/

#include "pch.h"

#include "i_log_utilities.h"
#include "i_log_logger.h"

#include <cstring>

namespace MI {
namespace LOG {

namespace {
const int NR_FATAL   = ILogger::S_FATAL | ILogger::S_ASSERT;
const int NR_ERROR   = NR_FATAL   | ILogger::S_ERROR;
const int NR_WARNING = NR_ERROR   | ILogger::S_WARNING;
const int NR_INFO    = NR_WARNING | ILogger::S_INFO | ILogger::S_PROGRESS;
const int NR_VERBOSE = NR_INFO    | ILogger::S_VERBOSE;
} // namespace

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
        case mi::base::MESSAGE_SEVERITY_FATAL:        return NR_FATAL;
        case mi::base::MESSAGE_SEVERITY_ERROR:        return NR_ERROR;
        case mi::base::MESSAGE_SEVERITY_WARNING:      return NR_WARNING;
        case mi::base::MESSAGE_SEVERITY_INFO:         return NR_INFO;
        case mi::base::MESSAGE_SEVERITY_VERBOSE:      return NR_VERBOSE;
        case mi::base::MESSAGE_SEVERITY_DEBUG:        return ILogger::S_ALL;
        case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT:
            return -1;
    }

    return -1;
}

const char* convert_category_to_string( LOG::ILogger::Category category)
{
    mi_static_assert( 14 == LOG::ILogger::NUM_OF_CATEGORIES);

    switch( category) {
        case LOG::ILogger::C_MAIN:      return "MAIN";
        case LOG::ILogger::C_NETWORK:   return "NETWORK";
        case LOG::ILogger::C_MEMORY:    return "MEMORY";
        case LOG::ILogger::C_DATABASE:  return "DATABASE";
        case LOG::ILogger::C_DISK:      return "DISK";
        case LOG::ILogger::C_PLUGIN:    return "PLUGIN";
        case LOG::ILogger::C_RENDER:    return "RENDER";
        case LOG::ILogger::C_GEOMETRY:  return "GEOMETRY";
        case LOG::ILogger::C_IMAGE:     return "IMAGE";
        case LOG::ILogger::C_IO:        return "IO";
        case LOG::ILogger::C_ERRTRACE:  return "ERRTRACE";
        case LOG::ILogger::C_MISC:      return "MISC";
        case LOG::ILogger::C_DISKTRACE: return "DISKTRACE";
        case LOG::ILogger::C_COMPILER:  return "COMPILER";

        case LOG::ILogger::C_ALL:
        case LOG::ILogger::C_DEFAULT:
        case LOG::ILogger::NUM_OF_CATEGORIES:
            break;
    }

    MI_ASSERT( false);
    return nullptr;
}

bool convert_string_to_category( const char* in, LOG::ILogger::Category& out)
{
    mi_static_assert( 14 == LOG::ILogger::NUM_OF_CATEGORIES);

    if( !in)
        return false;

    if( strcmp( in, "MAIN"    ) == 0) { out = LOG::ILogger::C_MAIN;      return true; }
    if( strcmp( in, "NETWORK" ) == 0) { out = LOG::ILogger::C_NETWORK;   return true; }
    if( strcmp( in, "MEMORY"  ) == 0) { out = LOG::ILogger::C_MEMORY;    return true; }
    if( strcmp( in, "DATABASE") == 0) { out = LOG::ILogger::C_DATABASE;  return true; }
    if( strcmp( in, "DISK"    ) == 0) { out = LOG::ILogger::C_DISK;      return true; }
    if( strcmp( in, "PLUGIN"  ) == 0) { out = LOG::ILogger::C_PLUGIN;    return true; }
    if( strcmp( in, "RENDER"  ) == 0) { out = LOG::ILogger::C_RENDER;    return true; }
    if( strcmp( in, "GEOMETRY") == 0) { out = LOG::ILogger::C_GEOMETRY;  return true; }
    if( strcmp( in, "IMAGE"   ) == 0) { out = LOG::ILogger::C_IMAGE;     return true; }
    if( strcmp( in, "IO"      ) == 0) { out = LOG::ILogger::C_IO;        return true; }
    if( strcmp( in, "ERRTRACE") == 0) { out = LOG::ILogger::C_ERRTRACE;  return true; }
    if( strcmp( in, "MISC"    ) == 0) { out = LOG::ILogger::C_MISC;      return true; }
    if( strcmp( in, "DISTRACE") == 0) { out = LOG::ILogger::C_DISKTRACE; return true; }
    if( strcmp( in, "COMPILER") == 0) { out = LOG::ILogger::C_COMPILER;  return true; }
    return false;
}

void Forwarding_logger::message(
    mi::base::Message_severity level, const char* module_category,
    const mi::base::Message_details& details, const char* message)
{
    // Split module_category into module name and category enum
    std::string module = "?";
    std::string category;
    if( module_category) {
        const char* colon = strchr( module_category, ':');
        if( colon) {
            module = std::string( module_category).substr( 0, colon-module_category);
            category = colon+1;
        } else
            module = module_category;
    }

    LOG::Mod_log::Category category_enum = LOG::Mod_log::C_MAIN; // avoid warning
    if( !convert_string_to_category( category.c_str(), category_enum))
        category_enum = LOG::Mod_log::C_MAIN;

    switch( level) {
        case mi::base::MESSAGE_SEVERITY_FATAL:
            LOG::mod_log->fatal  ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_ERROR:
            LOG::mod_log->error  ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            LOG::mod_log->warning( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_INFO:
            LOG::mod_log->info   ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
            LOG::mod_log->vstat  ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_DEBUG:
            LOG::mod_log->debug  ( module.c_str(), category_enum, details, "%s", message);
            break;
        default: // treat invalid enum values as MESSAGE_SEVERITY_ERROR
            LOG::mod_log->error  ( module.c_str(), category_enum, details, "%s", message);
            break;
    }
}

}
}
