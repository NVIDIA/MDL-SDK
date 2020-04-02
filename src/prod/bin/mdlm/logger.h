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
/// \file logger.h
/// \brief Logger class implementation to re-direct log output.
///

#pragma once

#include <mi/mdl_sdk.h>

namespace mdlm
{
    /// Custom logger to re-direct log output.
    ///
    class Logger : public mi::base::Interface_implement<mi::base::ILogger>
    {
        int  m_level;           ///< Logging level up to which messages are reported

        /// Returns a string label corresponding to the log level severity.
        static const char* get_log_level(mi::base::Message_severity level);

    public:
        /// Logger where only messages of level lower than the \p level parameter
        /// are written to stderr, i.e., \p level = 0 will disable all logging, and
        /// \p level = 1 logs only fatal messages, \p level = 2 logs errors, 
        /// \p level = 3 logs warnings, \p level = 4 logs info, \p level = 5 logs debug.
        Logger(int level) :
            m_level(level) {}

        /// Callback function logging a message.
        void message(
            mi::base::Message_severity level,
            const char* module_category,
            const mi::base::Message_details&,
            const char* message);
    };

} // namespace mdlm
