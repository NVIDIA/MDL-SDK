//*****************************************************************************
// Copyright 2018 NVIDIA Corporation. All rights reserved.
//*****************************************************************************

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
