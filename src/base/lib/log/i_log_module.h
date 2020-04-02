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
/// \brief Log module
///
/// The log module interface is the interface which is used for configuring the logging. It has
/// nothing to do with the actual logger interface.

#ifndef BASE_LIB_LOG_I_LOG_MODULE_H
#define BASE_LIB_LOG_I_LOG_MODULE_H

#include <base/system/main/i_module.h>

#include "i_log_logger.h"

namespace MI {

namespace SYSTEM { class Module_registration_entry; }

namespace LOG {

class ILog_target;

// The global module class for the log module.
class Log_module : public SYSTEM::IModule
{
public:
    /// Configures the module. Should be called when the CONFIG module is ready.
    virtual void configure() = 0;

    /// \name Configuration options
    //@{

    /// Sets the global severity limit.
    virtual void set_severity_limit( unsigned int sev = ILogger::S_DEFAULT) = 0;

    /// Returns the global severity limit.
    virtual unsigned int get_severity_limit() const = 0;

    /// Sets a per-category severity limit
    virtual void set_severity_by_category(
        ILogger::Category cat = ILogger::C_ALL,
        unsigned int sev = ILogger::S_DEFAULT) = 0;

    /// Returns a per-category severity limit.
    virtual unsigned int get_severity_by_category( ILogger::Category cat) const = 0;

    /// Sets the prefix bitmask.
    virtual void set_prefix( unsigned int prefix) = 0;

    /// Returns the prefix bitmask
    virtual unsigned int get_prefix() const = 0;

    //@}
    /// \name Log targets
    //@{

    /// Adds a log target to the back of the list (highest priority).
    virtual void add_log_target( ILog_target* target) = 0;

    /// Adds a log target to the front of the list (lowest priority).
    virtual void add_log_target_front( ILog_target* target) = 0;

    /// Removes a log target.
    virtual void remove_log_target( ILog_target* target) = 0;

    //@}
    /// \name Delayed log messages
    //@{

    /// Sets the flag for delaying log messages.
    ///
    /// If enabled, log messages are queued up in an internal buffer instead of sending them to the
    /// log targets. If disabled, delayed log messages are emitted before the current message.
    virtual void delay_log_messages( bool delay) = 0;

    /// Flushes any delayed log messages.
    ///
    /// Does not change the flag for delaying upcoming log messages.
    virtual void emit_delayed_log_messages() = 0;

    //@}
    /// \name Interface for the DATA module
    //@{

    /// Sets the host ID.
    ///
    /// This method is supposed to be used by the DATA module once the host ID becomes known.
    /// Initially, host ID 0 is used until this method is called.
    virtual void set_host_id( unsigned int host_id) = 0;

    /// Emits a log message.
    ///
    /// This method is supposed only to be used by the DATA module for log message from remote
    /// hosts. Such messages will not be forwarded to remote targets, only to file or stderr.
    virtual void insert_message(
        const char* mod, ILogger::Category, ILogger::Severity, const mi::base::Message_details&,
        const char* pfx, const char* msg) = 0;

    //@{
    /// \name Module system support
    //@}

    /// Retrieves the name of module.
    static const char* get_name() { return "LOG"; }

    /// Allows link time detection.
    static SYSTEM::Module_registration_entry* get_instance();
};

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_I_LOG_MODULE_H
