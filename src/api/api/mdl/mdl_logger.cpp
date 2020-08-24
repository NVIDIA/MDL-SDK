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

#include "pch.h"

#include "mdl_logger.h"

#include <mi/base/ilogger.h>
#include <base/hal/hal/hal.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/log/i_log_module.h>

#include <cstdarg>

namespace MI {

namespace MDL {

Logger* g_logger;

class Default_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity level, const char* /*module_category*/, const char* message)
    {
        const char* severity = 0;
        switch( level) {
            case mi::base::MESSAGE_SEVERITY_FATAL:   severity = "fatal: "; break;
            case mi::base::MESSAGE_SEVERITY_ERROR:   severity = "error: "; break;
            case mi::base::MESSAGE_SEVERITY_WARNING: severity = "warn:  "; break;
            case mi::base::MESSAGE_SEVERITY_INFO:    severity = "info:  "; break;
            case mi::base::MESSAGE_SEVERITY_VERBOSE: return; //-V1037 PVS
            case mi::base::MESSAGE_SEVERITY_DEBUG:   return; //-V1037 PVS
            case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT:
                ASSERT( M_NEURAY_API, false);
                return;
        }

        HAL::fprintf_stderr_utf8( severity);
        HAL::fprintf_stderr_utf8( message);
        putc( '\n', stderr);

#ifdef MI_PLATFORM_WINDOWS
        fflush( stderr);
#endif
    }

    void message(
        mi::base::Message_severity level, const char* mod_cat, const mi::base::Message_details&,
        const char* message)
    {
        this->message(level, mod_cat, message);
    }
};

Logger::Logger()
{
    g_logger = this;
    m_default_logger = new Default_logger();
    m_logger = m_default_logger;
    m_delay_messages = false;
}

Logger::~Logger()
{
    g_logger = 0;
    m_logger = 0;
    m_default_logger = 0;
}

void Logger::set_logger( mi::base::ILogger* logger)
{
    m_logger = logger ? make_handle_dup( logger) : m_default_logger;
    emit_delayed_log_messages();
}

mi::base::ILogger* Logger::get_logger()
{
    m_logger->retain();
    return m_logger.get();
}

void Logger::message(
    mi::base::Message_severity level, const char* category, const char* message)
{
    if( m_delay_messages) {
        mi::base::Lock::Block block( &m_delayed_messages_lock);
        m_delayed_messages.push_back( Message( level, category, message));
        return;
    }

    if( !m_delayed_messages.empty())
        emit_delayed_log_messages();

    m_logger->message( level, category, message);
}

void Logger::delay_log_messages( bool delay)
{
    m_delay_messages = delay;
}

void Logger::emit_delayed_log_messages()
{
    mi::base::Lock::Block block( &m_delayed_messages_lock);

    for( mi::Size i = 0; i < m_delayed_messages.size(); ++i) {
        const Message& m = m_delayed_messages[i];
        m_logger->message( m.m_level, m.m_category.c_str(), m.m_message.c_str());
    }

    m_delayed_messages.clear();
}

} // namespace MDL

namespace LOG {

class Logger : public ILogger
{
    void fatal( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
                const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        // TODO support category
        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_FATAL, "MDL", buf);
    }

    void error( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
                const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_ERROR, "MDL", buf);
    }

    void warning( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
                  const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_WARNING, "MDL", buf);
    }

    void stat( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
               const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_VERBOSE, "MDL", buf);
    }

    void vstat( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
                const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_VERBOSE, "MDL", buf);
    }

    void progress( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
                   const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_VERBOSE, "MDL", buf);
    }

    void info( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
               const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_INFO, "MDL", buf);
    }

    void debug( const char* /*mod*/, Category /*cat*/, const mi::base::Message_details&,
                const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_DEBUG, "MDL", buf);
    }

    void vdebug( const char* /*mod*/, Category /*cat*/, //-V524 PVS
                 const mi::base::Message_details&, const char* fmt, va_list args)
    {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);

        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_DEBUG, "MDL", buf);
    }

    void assertfailed( const char* /*mod*/, const char* expr, const char* file, int line)
    {
        char buf[1024];
        snprintf( buf, sizeof( buf), "assertion failed in %s %d: \"%s\"", file, line, expr);
        if( MDL::g_logger)
            MDL::g_logger->message( mi::base::MESSAGE_SEVERITY_ERROR, "MDL", buf);
        abort();
    }
};

static Logger g_logger;
ILogger* mod_log = &g_logger;

SYSTEM::Module_registration_entry* Log_module::get_instance()
{
    ASSERT( M_NEURAY_API, false);
    return 0;
}

void report_assertion_failure(
    SYSTEM::Module_id mod, const char* exp, const char* file, unsigned int line)
{
    mod_log->assertfailed( mod, exp, file, line);
}

} // namespace LOG

} // namespace MI
