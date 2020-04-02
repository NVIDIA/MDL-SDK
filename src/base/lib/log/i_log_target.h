/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Declare the log target interface for the logging.
///
/// This declares the interface for implementors of log targets in neuray.

#ifndef BASE_LIB_LOG_I_LOG_TARGET_H
#define BASE_LIB_LOG_I_LOG_TARGET_H

#include "i_log_logger.h"

namespace MI {

namespace LOG {

/// Abstract interface for log targets.
///
/// Different log targets can be used to output log messages to stderr, or to forward them to remote
/// hosts.
class ILog_target
{
public:
    /// Destructor
    virtual ~ILog_target() { }

    /// Log a message to the log target.
    ///
    /// The decision that the log message must be forwarded to that target has already been taken by
    /// the log module. The return value indicates, if the target could forward the message or not.
    /// Note that in some cases this does not imply that the log message reached its final
    /// destination, though.
    ///
    /// \param mod       The module ID string.
    /// \param cat       The category of the log message.
    /// \param sev       The severity of the log messsage.
    /// \param pfx       The prefix of the log messsage including a final colon and space (if not
    ///                  empty).
    /// \param msg       The log message itself.
    /// \param handled   Indicates whether the message was already handled.
    /// \return          \c true iff the message was handled
    virtual bool message(
        const char* mod,
        ILogger::Category cat,
        ILogger::Severity sev,
        const mi::base::Message_details&,
        const char* pfx,
        const char* msg) = 0;
};

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_I_LOG_TARGET_H
