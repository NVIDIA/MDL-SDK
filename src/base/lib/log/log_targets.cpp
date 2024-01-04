/******************************************************************************
 * Copyright (c) 2004-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief

#include "pch.h"

#include "log_targets.h"

#include <base/hal/hal/hal.h>
#ifdef WIN_NT
#include <mi/base/miwindows.h> // for ::OutputDebugString()
#endif

namespace MI {
namespace LOG {
namespace {

/// The colors used for the various severities.
HAL::Color color_by_severity(const ILogger::Severity sev)
{
    MI_ASSERT((sev & (sev - 1)) == 0);
    switch (sev) {
        case ILogger::S_FATAL:    return HAL::C_MAGENTA;
        case ILogger::S_ERROR:    return HAL::C_RED;
        case ILogger::S_WARNING:  return HAL::C_GREEN;
        case ILogger::S_STAT:     return HAL::C_YELLOW;
        case ILogger::S_VSTAT:    return HAL::C_YELLOW;
        case ILogger::S_PROGRESS: return HAL::C_DEFAULT;
        case ILogger::S_INFO:     return HAL::C_DEFAULT;
        case ILogger::S_DEBUG:    return HAL::C_CYAN;
        case ILogger::S_VDEBUG:   return HAL::C_BLUE;
        case ILogger::S_ASSERT:   return HAL::C_MAGENTA;
        default: break;
    }
    return HAL::C_DEFAULT;
}

}


Log_target_stderr::Log_target_stderr()
  : m_coloring( false)
{
}

void Log_target_stderr::configure_coloring( bool coloring)
{
    m_coloring = coloring;
}

bool Log_target_stderr::message(
    const char* mod, ILogger::Category, ILogger::Severity sev, const mi::base::Message_details& det,
    const char* pfx, const char* msg)
{
    if( m_coloring)
        HAL::set_console_color( color_by_severity(sev));

    HAL::fprintf_stderr_utf8( pfx);
    HAL::fprintf_stderr_utf8( msg);

    if( m_coloring)
        HAL::set_console_color( HAL::C_DEFAULT);

    putc( '\n', stderr);

#ifdef WIN_NT
    fflush( stderr);
#endif

    return true;
}

Log_target_debugger::Log_target_debugger()
{
#ifdef WIN_NT
    m_debugger_present = (::IsDebuggerPresent() != FALSE);
#else
    m_debugger_present = false;
#endif
}

bool  Log_target_debugger::message(
    const char* mod, ILogger::Category, ILogger::Severity, const mi::base::Message_details&,
    const char* pfx, const char* msg)
{
    if( m_debugger_present) {
#ifdef WIN_NT
        ::OutputDebugString( pfx);
        ::OutputDebugString( msg);
        ::OutputDebugString( "\n");
#endif
    }
    return m_debugger_present;
}

} // namespace LOG

} // namespace MI
