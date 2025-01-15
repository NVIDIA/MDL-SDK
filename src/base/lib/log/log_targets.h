/******************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_LIB_LOG_LOG_TARGETS_H
#define BASE_LIB_LOG_LOG_TARGETS_H

#include "i_log_target.h"

namespace MI {

namespace LOG {

/// Logs to stderr. Supports coloring.
class Log_target_stderr : public ILog_target
{
public:
    /// Constructor
    Log_target_stderr();

    // methods of ILog_target

    bool message(
        const char* mod, ILogger::Category, ILogger::Severity, const mi::base::Message_details& det,
        const char* pfx, const char* msg);

    // own methods

    /// Enables or disables coloring
    void configure_coloring( bool coloring);

private:
    bool m_coloring; ///< Is coloring enabled?
};

/// Logs to a debugger (Visual Studio only).
class Log_target_debugger : public ILog_target
{
public:
    /// Constructor
    Log_target_debugger();

    // methods of ILog_target

    bool message(
        const char* mod, ILogger::Category, ILogger::Severity, const mi::base::Message_details& det,
        const char* pfx, const char* msg);

private:
    bool m_debugger_present; ///< Is a debugger present?
};

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_LOG_TARGETS_H
