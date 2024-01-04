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

/** \file
 ** \brief Source for the Log_utilities, Forwarding_logger, and Receiving_logger implementation.
 **/

#include "pch.h"

#include "neuray_log_utilities.h"

#include <base/lib/log/i_log_utilities.h>

#include <cstring>
#include <string>

namespace MI {

namespace NEURAY {

const char* Log_utilities::convert_category_to_string( LOG::Mod_log::Category category)
{
    mi_static_assert( 14 == LOG::Mod_log::NUM_OF_CATEGORIES);

    switch( category) {
        case LOG::Mod_log::C_MAIN:      return "MAIN";
        case LOG::Mod_log::C_NETWORK:   return "NETWORK";
        case LOG::Mod_log::C_MEMORY:    return "MEMORY";
        case LOG::Mod_log::C_DATABASE:  return "DATABASE";
        case LOG::Mod_log::C_DISK:      return "DISK";
        case LOG::Mod_log::C_PLUGIN:    return "PLUGIN";
        case LOG::Mod_log::C_RENDER:    return "RENDER";
        case LOG::Mod_log::C_GEOMETRY:  return "GEOMETRY";
        case LOG::Mod_log::C_IMAGE:     return "IMAGE";
        case LOG::Mod_log::C_IO:        return "IO";
        case LOG::Mod_log::C_ERRTRACE:  return "ERRTRACE";
        case LOG::Mod_log::C_MISC:      return "MISC";
        case LOG::Mod_log::C_DISKTRACE: return "DISKTRACE";
        case LOG::Mod_log::C_COMPILER:  return "COMPILER";
        default:                        ASSERT( M_NEURAY_API, false); return nullptr;
    }
}

bool Log_utilities::convert_string_to_category( const char* in, LOG::Mod_log::Category& out)
{
    mi_static_assert( 14 == LOG::Mod_log::NUM_OF_CATEGORIES);

    if( !in)
        return false;

    if( strcmp( in, "MAIN"    ) == 0) { out = LOG::Mod_log::C_MAIN;      return true; }
    if( strcmp( in, "NETWORK" ) == 0) { out = LOG::Mod_log::C_NETWORK;   return true; }
    if( strcmp( in, "MEMORY"  ) == 0) { out = LOG::Mod_log::C_MEMORY;    return true; }
    if( strcmp( in, "DATABASE") == 0) { out = LOG::Mod_log::C_DATABASE;  return true; }
    if( strcmp( in, "DISK"    ) == 0) { out = LOG::Mod_log::C_DISK;      return true; }
    if( strcmp( in, "PLUGIN"  ) == 0) { out = LOG::Mod_log::C_PLUGIN;    return true; }
    if( strcmp( in, "RENDER"  ) == 0) { out = LOG::Mod_log::C_RENDER;    return true; }
    if( strcmp( in, "GEOMETRY") == 0) { out = LOG::Mod_log::C_GEOMETRY;  return true; }
    if( strcmp( in, "IMAGE"   ) == 0) { out = LOG::Mod_log::C_IMAGE;     return true; }
    if( strcmp( in, "IO"      ) == 0) { out = LOG::Mod_log::C_IO;        return true; }
    if( strcmp( in, "ERRTRACE") == 0) { out = LOG::Mod_log::C_ERRTRACE;  return true; }
    if( strcmp( in, "MISC"    ) == 0) { out = LOG::Mod_log::C_MISC;      return true; }
    if( strcmp( in, "DISTRACE") == 0) { out = LOG::Mod_log::C_DISKTRACE; return true; }
    if( strcmp( in, "COMPILER") == 0) { out = LOG::Mod_log::C_COMPILER;  return true; }
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
        if (colon) {
            module = std::string( module_category).substr( 0, colon-module_category);
            category = colon+1;
        } else
            module = module_category;
    }
    LOG::Mod_log::Category category_enum = LOG::Mod_log::C_MAIN; // avoid warning
    if( !Log_utilities::convert_string_to_category( category.c_str(), category_enum))
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

Receiving_logger::Receiving_logger( mi::base::ILogger* logger)
  : m_logger( logger)
{
    ASSERT( M_NEURAY_API, m_logger);
    m_logger->retain();
}

Receiving_logger::~Receiving_logger()
{
    m_logger->release();
}

bool Receiving_logger::message(
    const char* module,
    LOG::Mod_log::Category category,
    LOG::Mod_log::Severity severity,
    const mi::base::Message_details& det,
    const char* prefix,
    const char* message)
{
    // Convert severity from LOG::Severity to mi::base::Message_severity
    const mi::base::Message_severity severity_enum = LOG::convert_severity( severity);

    // Convert module and category into a string separated by ":"
    std::string module_category = module;
    module_category += ':';
    module_category += Log_utilities::convert_category_to_string( category); //-V769 PVS

    // Convert prefix and message into a string separated by " "
    std::string full_message = prefix;
    full_message += message;

    m_logger->message( severity_enum, module_category.c_str(), det, full_message.c_str());
    return true;
}

} // namespace NEURAY

} // namespace MI
