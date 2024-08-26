/******************************************************************************
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
 *****************************************************************************/

/// \file
/// \brief The standard neuray log module
///
/// This is the standard neuray log module implementation. Other implementations can be done
/// overriding this one, if the complexity is not needed.

#ifndef BASE_LIB_LOG_LOG_MODULE_IMPL_H
#define BASE_LIB_LOG_LOG_MODULE_IMPL_H

#include "i_log_module.h"

#include <mi/base/lock.h>

#include <vector>

namespace MI {

namespace LOG {

class Log_target_stderr;

/// Implementation of the Log_module and ILogger interfaces.
class Log_module_impl : public Log_module, public ILogger
{
public:
    /// Constructor
    Log_module_impl();

    /// Destructor
    ~Log_module_impl();

    // method of SYSTEM::IModule

    bool init() { return true; }

    void exit() { }

    // methods of Log_module

    void configure() override;

    void set_severity_limit( unsigned int sev) override { m_sev_limit = sev; }

    unsigned int get_severity_limit() const override { return m_sev_limit; }

    void set_severity_by_category( Category cat, unsigned int sev) override;

    unsigned int get_severity_by_category( Category cat) const override;

    void set_prefix( unsigned int prefix) override { m_prefix = prefix; }

    unsigned int get_prefix() const override { return m_prefix; }

    void add_log_target( ILog_target* target) override;

    void add_log_target_front( ILog_target* target) override;

    void remove_log_target( ILog_target* target) override;

    void delay_log_messages( bool delay) override;

    void emit_delayed_log_messages() override;

    void set_host_id( unsigned int host_id) override { m_host_id = host_id; }

    void set_host_name( const char* host_name);

    void insert_message(
            const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
            const char* pfx, const char* msg) override;

    // methods of ILogger

    [[noreturn]]
    void fatal(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void error(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void warning(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void stat(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void vstat(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void progress(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void info(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void debug(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void vdebug(
        const char* mod, Category cat, const mi::base::Message_details&,
        const char* fmt, va_list args) override;

    void assertfailed(
        const char* mod, const char* fmt, const char* file, int line) override;

private:
    /// Inserts the arguments \p args into the format string \p fmt and calls #text_message().
    ///
    /// Used by #fatal() ... #vdebug().
    void message(
            const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
            const char* fmt, va_list args);

    /// Delays, emits delayed, or passes on log message(s).
    void handle_message(
            const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
            const char* text);

    /// Splits \p text into lines and calls #line_message().
    void text_message(
            const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
            const char* text);

    /// Generates the message prefix and calls #insert_message_internal().
    void line_message(
        const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
        const char* line_begin, const char* line_end);

    /// Invokes the registered log targets and/or calls #std_message() for the message.
    void insert_message_internal(
        const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
        const char* const pfx, const char* const msg);

    /// Implementation of #emit_delayed_log_messages() with #m_lock held.
    void emit_delayed_log_messages_internal();

    /// Logs message to the stderr and/or debugger logger.
    void std_message( const char* mod, Category cat, Severity sev, const mi::base::Message_details&,
                      const char* pfx, const char* msg);

    /// Maximum length of the message (excluding prefix).
    static constexpr size_t MAX_MSG_SIZE = 64 * 1024 - 1;

    /// Maximum length of a line of the message.
    static constexpr size_t MAX_LINE_SIZE = 8 * 1024 - 1;

    /// Maximum length of the message prefix
    static constexpr size_t MAX_PREFIX_SIZE = 127;

    /// Maximum length of the host name in the message prefix
    static constexpr size_t MAX_HOST_NAME_SIZE = 32;

    /// Represents a delayed message
    struct Message
    {
        /// Constructor
        Message(
                const std::string& mod,
                Category cat, Severity sev, const mi::base::Message_details& det,
                const std::string& msg)
          : m_mod( mod), m_cat( cat), m_sev( sev), m_details( det), m_msg( msg) { }

        /// Fields
        std::string m_mod;
        Category m_cat;
        Severity m_sev;
        mi::base::Message_details m_details;
        std::string m_msg;
    };

    char m_msg_buf[MAX_MSG_SIZE+1];    ///< Message buffer, access needs #m_lock.
    char m_line_buf[MAX_LINE_SIZE+1];  ///< Line buffer, access needs #m_lock.
    char m_pfx_buf[MAX_PREFIX_SIZE+1]; ///< Prefix buffer, access needs #m_lock.

    std::string m_host_name;        ///< Local host name

    unsigned int m_sev_limit;                     ///< Global severity limit
    unsigned int m_sev_by_cat[NUM_OF_CATEGORIES]; ///< Per-category severity limit
    unsigned int m_prefix;                        ///< Prefix bitmask
    bool m_assert_is_fatal;                       ///< Are failed assertions fatal?
    bool m_did_update_conf;                       ///< Was #configure() called?
    unsigned int m_host_id;                       ///< The host ID
    Log_target_stderr* m_log_target_stderr;       ///< Logs to stderr
    ILog_target* m_log_target_debugger;           ///< Logs to debugger
    typedef std::vector<ILog_target*> Log_targets;
    Log_targets m_targets;                        ///< Registered log targets, access needs #m_lock.
    bool m_delay_messages;                        ///< Are log messages delayed?
    std::vector<Message> m_delayed_messages;      ///< Delayed log messages

    /// Lock used for the various buffers and for #m_targets.
    mi::base::Lock m_lock;

    /// Maps categories to strings.
    static const char* const category_name[/*NUM_OF_CATEGORIES*/];
};

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_LOG_MODULE_IMPL_H
