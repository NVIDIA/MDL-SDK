/******************************************************************************
 * Copyright (c) 2007-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "log_module_impl.h"
#include "log_targets.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <base/system/main/access_module.h>
#include <base/system/main/module_registration.h>
#include <base/util/registry/i_config_registry.h>
#include <base/hal/hal/hal.h>
#include <base/hal/thread/i_thread_attr.h>
#include <base/hal/time/i_time.h>
#include <base/lib/config/config.h>

MI::LOG::ILogger* MI::LOG::mod_log;

namespace MI {

namespace LOG {

const char* const Log_module_impl::category_name[] = {
    "main", "net", "mem", "db", "disk", "plug", "rend", "geo", "img",
    "io", "trac", "misc", "dskt", "comp" };

namespace {

const char* get_severity_name( const ILogger::Severity sev)
{
    switch( sev) {
        case ILogger::S_FATAL:   return "fatal";
        case ILogger::S_ERROR:   return "error";
        case ILogger::S_WARNING: return "warn";
        case ILogger::S_STAT:    return "stat";
        case ILogger::S_VSTAT:   return "vstat";
        case ILogger::S_PROGRESS:return "progr";
        case ILogger::S_INFO:    return "info";
        case ILogger::S_DEBUG:   return "debug";
        case ILogger::S_VDEBUG:  return "vdebg";
        case ILogger::S_ASSERT:  return "assrt";
        case ILogger::NUM_OF_SEVERITIES:
        case ILogger::S_TERSE:
        case ILogger::S_DEFAULT:
        case ILogger::S_VERBOSE:
        case ILogger::S_ALL:
            break;
    }
    return "";
}

} // namespace

Log_module_impl::Log_module_impl()
{
    // Do not access the CONFIG module here yet. Defer all such queries to the configure() method.

    m_msg_buf[0]  = '\0';
    m_line_buf[0] = '\0';
    m_pfx_buf[0]  = '\0';

    m_host_name = "localhost";

    m_sev_limit = S_DEFAULT;
    for( int c = 0; c < NUM_OF_CATEGORIES; c++)
        m_sev_by_cat[c] = ((1 << c) & C_DEFAULT) ? S_DEFAULT : S_TERSE;

    m_prefix = P_DEFAULT;
    m_assert_is_fatal = true;
    m_did_update_conf = false;
    m_host_id = 0;
    m_log_target_stderr = new Log_target_stderr;
    m_log_target_debugger = new Log_target_debugger;
    m_delay_messages = false;

    mod_log = this;
}

Log_module_impl::~Log_module_impl()
{
    mod_log = nullptr;
    delete m_log_target_debugger;
    delete m_log_target_stderr;
}

void Log_module_impl::configure()
{
    if( m_did_update_conf)
        return;

    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    if( config_module.get_status() == SYSTEM::MODULE_STATUS_INITIALIZED
        && config_module->is_initialization_complete()) {

        m_did_update_conf = true;
        const CONFIG::Config_registry& registry = config_module->get_configuration();

        bool coloring = false;
        CONFIG::update_value( registry, "coloring", coloring);
        m_log_target_stderr->configure_coloring( coloring);

        mi_static_assert( sizeof( m_assert_is_fatal) == sizeof(bool));
        CONFIG::update_value( registry, "assert_is_fatal", m_assert_is_fatal);

        mi_static_assert( sizeof( m_prefix) == sizeof(int));
        CONFIG::update_value( registry, "prefix", m_prefix);

        mi_static_assert( sizeof( m_sev_limit)  == sizeof(int));
        CONFIG::update_value( registry, "severity_limit", m_sev_limit);

        mi_static_assert( sizeof( m_sev_by_cat[0])  == sizeof(int));
        for( int c = 0; c < NUM_OF_CATEGORIES; c++) {
            std::string key_name = "severity_for_" + std::string( category_name[c]);
            CONFIG::update_value( registry, key_name, m_sev_by_cat[c]);
        }
    }
}

void Log_module_impl::set_severity_by_category( Category cat, unsigned int sev)
{
    if( cat == C_ALL)
        for( int c = 0; c < NUM_OF_CATEGORIES; c++)
            m_sev_by_cat[c] = sev;
    else
        m_sev_by_cat[cat] = sev;
}

unsigned int Log_module_impl::get_severity_by_category( Category cat) const
{
    ASSERT( M_LOG, cat >= 0 && cat < NUM_OF_CATEGORIES);
    return m_sev_by_cat[cat];
}

void Log_module_impl::add_log_target( ILog_target* target)
{
    ASSERT( M_LOG, target);
    mi::base::Lock::Block block( &m_lock);
    m_targets.push_back( target);
}

void Log_module_impl::add_log_target_front( ILog_target* target)
{
    ASSERT( M_LOG, target);
    mi::base::Lock::Block block( &m_lock);
    m_targets.insert( m_targets.begin(), target);
}

void Log_module_impl::remove_log_target( ILog_target* target)
{
    ASSERT( M_LOG, target);
    mi::base::Lock::Block block( &m_lock);
    const auto it = std::find( m_targets.begin(), m_targets.end(), target);
    if( it != m_targets.end())
        m_targets.erase( it);
}

void Log_module_impl::delay_log_messages( bool delay)
{
    mi::base::Lock::Block block( &m_lock);
    m_delay_messages = delay;
}

void Log_module_impl::emit_delayed_log_messages()
{
    mi::base::Lock::Block block( &m_lock);
    emit_delayed_log_messages_internal();
}

void Log_module_impl::set_host_name( const char* host_name)
{
    m_host_name = host_name;
    if( m_host_name.size() > MAX_HOST_NAME_SIZE) {
        m_host_name = m_host_name.substr( 0, MAX_HOST_NAME_SIZE - 3);
        m_host_name += "...";
    }
}

void Log_module_impl::insert_message(
    const char* mod, Category cat, Severity sev, const mi::base::Message_details& det,
    const char* pfx, const char* msg)
{
    mi::base::Lock::Block block( &m_lock);
    insert_message_internal( mod, cat, sev, det, pfx, msg);
}

void Log_module_impl::fatal( const char* mod, Category cat, const mi::base::Message_details& det,
                             const char* fmt, va_list args)
{
    message( mod, cat, S_FATAL, det, fmt, args);

    if( !HAL::has_stderr()) {
        char buf[32768];
        vsnprintf( buf, sizeof( buf), fmt, args);
        HAL::message_box( "Fatal Error", buf);
    } else {
        // Give a chance to read message on console, useful on temporary opened Windows consoles.
        TIME::sleep( 3);
    }

    // The assertion here causes an abort() in debug builds such that core dumps can be generated.
    // In release builds we call exit(1) (via HAL::fatal_exit() to flush stderr on Windows).
    ASSERT( M_LOG, false);

    HAL::fatal_exit( 1);
}

void Log_module_impl::error( const char* mod, Category cat, const mi::base::Message_details& det,
                             const char* fmt, va_list args)
{
    message( mod, cat, S_ERROR, det, fmt, args);
}

void Log_module_impl::warning( const char* mod, Category cat, const mi::base::Message_details& det,
                               const char* fmt, va_list args)
{
    message( mod, cat, S_WARNING, det, fmt, args);
}

void Log_module_impl::stat( const char* mod, Category cat, const mi::base::Message_details& det,
                            const char* fmt, va_list args)
{
    message( mod, cat, S_STAT, det, fmt, args);
}

void Log_module_impl::vstat( const char* mod, Category cat, const mi::base::Message_details& det,
                             const char* fmt, va_list args)
{
    message( mod, cat, S_VSTAT, det, fmt, args);
}

void Log_module_impl::progress( const char* mod, Category cat, const mi::base::Message_details& det,
                                const char* fmt, va_list args)
{
    message( mod, cat, S_PROGRESS, det, fmt, args);
}

void Log_module_impl::info( const char* mod, Category cat, const mi::base::Message_details& det,
                            const char* fmt, va_list args)
{
    message( mod, cat, S_INFO, det, fmt, args);
}

void Log_module_impl::debug( const char* mod, Category cat, const mi::base::Message_details& det,
                             const char* fmt, va_list args)
{
    message( mod, cat, S_DEBUG, det, fmt, args);
}

void Log_module_impl::vdebug( const char* mod, Category cat, const mi::base::Message_details& det,
                              const char* fmt, va_list args)
{
    message( mod, cat, S_VDEBUG, det, fmt, args);
}

void Log_module_impl::assertfailed( const char* mod, const char* expr, const char* file, int line)
{
    char buf[1024];
    snprintf( buf, sizeof( buf), "assertion failed in %s %d: \"%s\"", file, line, expr);

    // Avoid infinite recursion or deadlock for assertion from the LOG module itself.
    if( strcmp( mod , "LOG") == 0) {
        fprintf( stderr, "%s\n", buf);
#ifdef WIN_NT
        fflush( stderr);
#endif
    }
    else {
        mi::base::Lock::Block block( &m_lock);
        handle_message( mod, C_MAIN, S_ASSERT, {}, buf);
    }
    if( m_assert_is_fatal)
        abort();
}

void Log_module_impl::message(
    const char* mod, Category cat, Severity sev, const mi::base::Message_details& det,
    const char* fmt, va_list args)
{
    if( !(m_sev_limit & sev) || !(m_sev_by_cat[cat] & sev))
        return;

    mi::base::Lock::Block block( &m_lock);

    if( !fmt)
        fmt = "";

    m_msg_buf[MAX_MSG_SIZE] = '\0';
    const int count = vsnprintf( m_msg_buf, MAX_MSG_SIZE, fmt, args);
    ASSERT( M_LOG, count >= 0);

    if( count >= static_cast<int>( MAX_MSG_SIZE)) {
        m_msg_buf[MAX_MSG_SIZE-1] = '.';
        m_msg_buf[MAX_MSG_SIZE-2] = '.';
        m_msg_buf[MAX_MSG_SIZE-3] = '.';
    }
    handle_message( mod, cat, sev, det, m_msg_buf);
}

/// Finds the end of the line within [text,text_end) and returns it.
///
/// Recognize EOL delimiters "\n", "\r", "\n\r" and "\r\n".
///
/// \param text                   begin of text interval to search
/// \param text_end               one-past-the-end of text interval to search
/// \param[out] next_line_begin   beginning of next line (after delimiters), or text_end
/// \return                       end of the line (excluding delimiters)
static const char* find_line_end(
    const char* text,
    const char* text_end,
    const char*& next_line_begin)
{
    // we recognize "\n", "\r", "\n\r" and "\r\n"
    const char* line_end = text;
    while( line_end != text_end && *line_end != '\n' && *line_end != '\r')
        ++line_end;

    const char* next = line_end;
    if( line_end == text_end) {
        next = text_end;
    } else if( *line_end == '\n') {
        next = line_end + 1;
        if( next < text_end && *next == '\r')
            ++next;
    } else {
        ASSERT( M_LOG, *line_end == '\r');
        next = line_end + 1;
        if( next < text_end && *next == '\n')
            ++next;
    }
    next_line_begin = next;
    return line_end;
}

void Log_module_impl::handle_message(
        const char* mod, Category cat, Severity sev, const mi::base::Message_details& det,
        const char* text)
{
    ASSERT( M_LOG, text);

    if( m_delay_messages) {
        m_delayed_messages.emplace_back( mod, cat, sev, det, text);
        return;
    }

    if( !m_delayed_messages.empty())
        emit_delayed_log_messages_internal();

    text_message( mod, cat, sev, det, text);
}

void Log_module_impl::text_message(
    const char* mod,
    Category cat,
    Severity sev,
    const mi::base::Message_details& det,
    const char* text)
{
    const size_t len = strlen( text);
    const char* text_end = &text[len];

    const char* line_begin = text;
    const char* next_line_begin = text_end; // be safe
    const char* line_end = find_line_end( line_begin, text_end, next_line_begin);

    do {
        const size_t line_len = line_end - line_begin;
        if( line_len <= MAX_LINE_SIZE) {
            // line is short enough, send one line
            line_message( mod, cat, sev, det, line_begin, line_end);
        } else {
            // split line that is too long
            const char* sub_end = line_begin + MAX_LINE_SIZE;
            while( true) {
                // send one line of message
                line_message( mod, cat, sev, det, line_begin, sub_end);
                line_begin = sub_end;
                if( line_begin >= line_end) break;
                sub_end += MAX_LINE_SIZE;
                if( sub_end > line_end) sub_end = line_end;
            }
        }
        line_begin = next_line_begin;
        line_end = find_line_end( line_begin, text_end, next_line_begin);
    } while( line_begin != text_end);
}

/// snprintf()-like function that updates the pointer for the next message part automatically
///
/// \param buffer     The buffer to print into.
/// \param position   Start printing at buffer[position], updated accordingly.
/// \param max_size   Size of the buffer.
/// \param fmt        Format string.
/// \param ...        Format string arguments.
static void my_snprintf( char* buffer, int& position, size_t max_size, const char* fmt, ...)
{
    if( position >= static_cast<int>( max_size))
        return;

    va_list args;
    va_start( args, fmt);
    const int count = vsnprintf( buffer + position, max_size - position, fmt, args);
    va_end( args);

    position = count>=0 ? position+count : max_size;
}

void Log_module_impl::line_message(
    const char* mod, Category cat, Severity sev, const mi::base::Message_details& det,
    const char* line_begin, const char* line_end)
{
    char* pfx = m_pfx_buf;
    pfx[0] = '\0';

    mi_static_assert( sizeof( category_name) == NUM_OF_CATEGORIES * sizeof( const char*));
    ASSERT( M_LOG, (cat >= 0 && cat < NUM_OF_CATEGORIES));
    ASSERT( M_LOG, (sev & (sev-1)) == 0);

    int position = 0;

    if( m_prefix & P_TIME) {
        const TIME::Time now = TIME::get_wallclock_time();
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%-s ", now.to_string().c_str());
    }

    if( m_prefix & P_TIME_SECONDS) {
        const TIME::Time now = TIME::get_time();
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%3.3f ", now.get_seconds());
    }
    if( m_prefix & P_HOST_THREAD) {
        THREAD::Thread_attr thread_attr;
        my_snprintf( pfx, position, MAX_PREFIX_SIZE,"%3d.%-3u ", m_host_id, thread_attr.get_id());
    }
    if( m_prefix & P_CUDA_DEVICE) {
        THREAD::Thread_attr thread_attr;
        std::string device;
        switch (det.device_id) {
            case mi::base::Message_details::DEVICE_ID_CPU: break; // don't specify
            case mi::base::Message_details::DEVICE_ID_ALL_CUDA:
                device = "C*";
                break;
            case mi::base::Message_details::DEVICE_ID_UNKNOWN_CUDA:
                device = "C??";
                break;
            default:
                device = "C" + std::to_string(det.device_id);
                break;
        }
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%-3s ", device.c_str());
    }
    if( m_prefix & P_TAGS) {
        constexpr char flags[] = "CUIAVRMFS!";
        constexpr unsigned FLAG_COUNT = sizeof(flags)-1;
        char bits[FLAG_COUNT+1] = {};
        for (size_t i=0; i<FLAG_COUNT; ++i)
            bits[i] = det.is_tagged(1u<<i) ? flags[i] : '-';
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%s ", bits);
    }
    if( (m_prefix & P_HOST_NAME) && !m_host_name.empty())
        my_snprintf( pfx, position, MAX_PREFIX_SIZE,"%s ", m_host_name.c_str());

    if( m_prefix & P_MODULE) {
        char limited_mod[7];
        strncpy( limited_mod, mod ? mod : "?", 6);
        limited_mod[6] = '\0';
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%-6s ", limited_mod);
    }

    if( m_prefix & P_CATEGORY)
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%-4.4s ", category_name[cat]);

    if( m_prefix & P_SEVERITY)
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, "%-5.5s", get_severity_name(sev));

    if( m_prefix)
        my_snprintf( pfx, position, MAX_PREFIX_SIZE, ": ");

    pfx[MAX_PREFIX_SIZE] = '\0'; // not sure whether really needed

    ASSERT( M_LOG, line_begin <= line_end);
    const size_t len = line_end - line_begin;
    ASSERT( M_LOG, len <= MAX_LINE_SIZE);
    strncpy( m_line_buf, line_begin, len);
    m_line_buf[len] = '\0';

    m_line_buf[MAX_LINE_SIZE] = '\0'; // not sure whether really needed

    insert_message_internal( mod, cat, sev, det, pfx, m_line_buf);
}

void Log_module_impl::insert_message_internal(
    const char* mod, Category cat, Severity sev, const mi::base::Message_details& det,
    const char* pfx, const char* msg)
{
    // Add host info if not yet present
    if (det.host_id == Det::HOST_ID_LOCAL && m_host_id) {
        mi::base::Message_details updated{det};
        updated.host(m_host_id);
        return insert_message_internal(mod,cat,sev,updated,pfx,msg);
    }

    bool handled = false;

    // Log fatal messages to stderr/debugger first. Everything else might easily fail.
    if( sev == S_FATAL)
        std_message( mod, cat, sev, det, pfx, msg);

    // Invoke registered log targets in reverse order until one handles the message.
    auto rit = m_targets.rbegin();
    for( ; !handled && rit != m_targets.rend(); ++rit)
        if( (*rit)->message( mod, cat, sev, det, pfx, msg))
            handled = true;

    // If not handled yet (and not fatal), log to stderr/debugger.
    if( !handled && sev != S_FATAL)
        std_message( mod, cat, sev, det, pfx, msg);
}

void Log_module_impl::emit_delayed_log_messages_internal()
{
    for( const Message& m: m_delayed_messages) {
        // Re-apply the severity filter since delayed messages might have been generated before the
        // user had a chance to configure the filter levels.
        if( !(m_sev_limit & m.m_sev) || !(m_sev_by_cat[m.m_cat] & m.m_sev))
            continue;
        text_message(
            m.m_mod.c_str(), m.m_cat, m.m_sev, m.m_details, m.m_msg.c_str());
    }

    m_delayed_messages.clear();
}

void Log_module_impl::std_message(
    const char* mod, Category cat, Severity sev, const mi::base::Message_details& det,
    const char* pfx, const char* msg)
{
    if( m_log_target_stderr)
        m_log_target_stderr->message( mod, cat, sev, det, pfx, msg);
    m_log_target_debugger->message( mod, cat, sev, det, pfx, msg);
}

// Module registration.
static SYSTEM::Module_registration<Log_module_impl> s_module( M_LOG, "LOG");

SYSTEM::Module_registration_entry* Log_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

} // namespace LOG

} // namespace MI
