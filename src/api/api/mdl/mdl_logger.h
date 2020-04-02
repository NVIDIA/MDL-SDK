/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief
 **/

#ifndef API_API_MDL_LOG_MODULE_STUB_H
#define API_API_MDL_LOG_MODULE_STUB_H

#include <string>
#include <vector>

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/lock.h>
#include <mi/base/interface_implement.h>

namespace mi { namespace base { class ILogger; } }

namespace MI {

namespace MDL {

/// This class forwards all messages to the wrapped logger.
///
/// If no logger is explicitly installed, a default logger is used that prints all messages of
/// severity #mi::base::MESSAGE_SEVERITY_INFO or higher to stderr. The .cpp file contains also
/// stubs for the LOG module to forward its methods to this  class.
class Logger
{
public:
    Logger();

    ~Logger();

    /// Install a new logger (replaces the old one).
    ///
    /// \c NULL can be used to re-install the default logger.
    void set_logger( mi::base::ILogger* logger);

    /// Returns the installed logger.
    mi::base::ILogger* get_logger();

    /// Forwards the message to the installed logger.
    void message( mi::base::Message_severity level, const char* category, const char* message);

    /// Sets the flag for delaying log messages.
    ///
    /// If enabled, log messages are queued up in an internal buffer instead of sending them to the
    /// log targets. If disabled, delayed log messages are emitted before the current message.
    void delay_log_messages( bool delay);

    /// Flushes any delayed log messages.
    ///
    /// Does not change the flag for delaying upcoming log messages.
    void emit_delayed_log_messages();

private:
    /// Represents a delayed log message.
    struct Message {

        /// Constructor
        Message( mi::base::Message_severity level, const char* category, const char* message)
          : m_level( level), m_category( category), m_message( message) { }

        /// Fields
        mi::base::Message_severity m_level;
        std::string m_category;
        std::string m_message;
    };

    /// The used logger.
    mi::base::Handle<mi::base::ILogger> m_logger;
    /// The default logger.
    mi::base::Handle<mi::base::ILogger> m_default_logger;
    /// Indicates whether log messages are delayed.
    bool m_delay_messages;
    /// Delayed log messages. Needs #m_delayed_messages_lock.
    std::vector<Message> m_delayed_messages;
    /// Lock for #m_delayed_messages.
    mi::base::Lock m_delayed_messages_lock;
};

} // namespace MDL

} // namespace MI

#endif // API_API_MDL_LOG_MODULE_STUB_H
