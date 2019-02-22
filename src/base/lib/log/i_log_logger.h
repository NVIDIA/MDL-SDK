/***************************************************************************************************
 * Copyright (c) 2010-2019, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Declare the logger interface.
/// 
/// This file declares the interface for logging. It allows to implement different implementations
/// for the logging functionality. It has nothing to do with log configuration. The access to the
/// logging functionality is done by the mod_log variable.

#ifndef BASE_LIB_LOG_I_LOG_LOGGER_H
#define BASE_LIB_LOG_I_LOG_LOGGER_H

#include <cstdarg>
#include <base/system/main/module.h>
#include <base/system/main/types.h>
#include "i_log_assert.h"

namespace MI {

namespace LOG {

class ILogger
{
public:
    /// Severities indicate the importance of a message.
    enum Severity {
        S_FATAL           = 0x0001, ///< fatal errors, almost never happen
        S_ERROR           = 0x0002, ///< recoverable errors causing trouble
        S_WARNING         = 0x0004, ///< problems that need user attention
        S_STAT            = 0x0008, ///< brief statistics
        S_VSTAT           = 0x0010, ///< verbose statistics
        S_PROGRESS        = 0x0020, ///< progress reports
        S_INFO            = 0x0100, ///< other informational messages
        S_DEBUG           = 0x0200, ///< debug message
        S_VDEBUG          = 0x0400, ///< verbose debug message
        S_ASSERT          = 0x0800, ///< assertions
        NUM_OF_SEVERITIES = 10,     ///< number of distinct severity levels

        /// These severities should never be filtered out. The per-category filters for the
        /// C_*TRACE categories use this by default.
        S_TERSE   = S_FATAL | S_ERROR | S_WARNING,

        /// These severities are enabled by default in the global filter and in the per-category
        /// filter for most categories.
        S_DEFAULT = S_TERSE | S_STAT  | S_PROGRESS | S_INFO | S_ASSERT,

        /// All messages useful to customers.
        S_VERBOSE = S_DEFAULT | S_VSTAT,

        /// All severities, including verbose and debug messages.
        S_ALL     = S_VERBOSE | S_DEBUG | S_VDEBUG
    };

    /// Categories are used to specify the functional area of a log message.
    ///
    /// This is in contrast to module names which specify the origin of the log message. Categories
    /// are fixed, whereas as module names are arbitrary strings. In addition to the overall
    /// severity limit, it is possible to set an additional severity limit per category (see
    /// C_DEFAULT).
    ///
    /// Do not forget to update Log_module_impl::category_name[] in case of changes, as well as
    /// the various stubs for the log module. The admin server page in
    /// DATA::Mod_data_impl::log_configuration_page_handler() might need to be adjusted, too.
    enum Category {
        C_MAIN,            ///< reserved for the application itself
        C_NETWORK,         ///< networking
        C_MEMORY,          ///< memory management
        C_DATABASE,        ///< database, scene data
        C_DISK,            ///< raw disk I/O, swapping
        C_PLUGIN,          ///< plugins (unless other categories fit better)
        C_RENDER,          ///< rendering
        C_GEOMETRY,        ///< geometry processing, e.g., tessellation
        C_IMAGE,           ///< texture processing, image and video plugins
        C_IO,              ///< scene data import and export
        C_ERRTRACE,        ///< ??? (base/hal/msg, render/render/softshader)
        C_MISC,            ///< other
        C_DISKTRACE,       ///< opened files and directories (base/hal/disk)
        C_COMPILER,        ///< MetaSL compiler
        NUM_OF_CATEGORIES, ///< number of distinct categories

        /// All categories set.
        C_ALL = (1 << NUM_OF_CATEGORIES) - 1,

        /// Categories for which the per-category severity limit is set to S_DEFAULT. For all
        /// other categories (C_*TRACE) it is set to S_TERSE.
        C_DEFAULT = (1 << C_MAIN)     | (1 << C_NETWORK)  | (1 << C_MEMORY)   |
                    (1 << C_DATABASE) | (1 << C_DISK)     | (1 << C_PLUGIN)   |
                    (1 << C_RENDER)   | (1 << C_GEOMETRY) | (1 << C_IMAGE)    |
                    (1 << C_IO)       | (1 << C_MISC)     | (1 << C_COMPILER)
    };

    /// The prefix is a short substring that is inserted at the beginning of each log message.
    ///
    /// It contains some useful meta-information about the log message. The prefix consists of
    /// various components which can be enabled or disabled individually. The order of the enum
    /// values is the same as the order of the corresponding components in the prefix of the log
    /// message. Note that the values here are different from the values of
    /// mi::neuraylib::Log_prefix.
    enum Prefix {
        P_TIME         = 0x0004, ///< human-readable timestamp
        P_TIME_SECONDS = 0x0080, ///< timestamp in seconds with milliseconds resolution
        P_HOST_THREAD  = 0x0001, ///< ID of the host and thread that generate the log message
        P_HOST_NAME    = 0x0100, ///< name of the host that generates the log message
        P_MODULE       = 0x0010, ///< module that generates the log message
        P_CATEGORY     = 0x0020, ///< category of the log message
        P_SEVERITY     = 0x0040, ///< severity of the log message
        P_DEFAULT      = 0x0071  ///< default prefix
    };

    /// \name Virtual functions to emit log messages of the corresponding severity.
    //@{

    virtual void fatal(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void error(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void warning(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void stat(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void vstat(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void progress(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void info(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void debug(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void vdebug(
        const char* mod, Category cat, const char* fmt, va_list args) = 0;

    virtual void assertfailed( const char* mod, const char* fmt, const char* file, int line) = 0;

    //@}
    /// \name Wrappers for the virtual functions above using an ellipsis instead of va_list.
    //@{

    void fatal( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        fatal( mod, cat, fmt, args);
        va_end( args);
    }

    void error( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        error( mod, cat, fmt, args);
        va_end( args);
    }

    void warning( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        warning( mod, cat, fmt, args);
        va_end( args);
    }

    void stat( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        stat( mod, cat, fmt, args);
        va_end( args);
    }

    void vstat( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        vstat( mod, cat, fmt, args);
        va_end( args);
    }

    void progress( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        progress( mod, cat, fmt, args);
        va_end( args);
    }

    void info( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        info( mod, cat, fmt, args);
        va_end( args);
    }

    void debug( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        debug( mod, cat, fmt, args);
        va_end( args);
    }

    void vdebug( const char* mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        vdebug( mod, cat, fmt, args);
        va_end( args);
    }

    //@}
    /// \name Wrappers for the virtual functions above using an ellipsis instead of va_list and
    ///       a module ID instead of a module name.
    //@{

    void fatal(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        fatal( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void error(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        error( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void warning(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        warning( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void stat(
         Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        stat( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void vstat(
         Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        vstat( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void progress(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        progress( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void info(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        info( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void debug(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        debug( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void vdebug(
        Module_id mod, Category cat, const char* fmt, ...) PRINTFLIKE4
    {
        va_list args;
        va_start( args, fmt);
        vdebug( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void assertfailed(
        Module_id mod, const char* fmt, const char* file, int line)
    {
        assertfailed( SYSTEM::Module::id_to_name( mod), fmt, file, line);
    }

    //@}
    /// \name Wrappers for the virtual functions above using an ellipsis instead of va_list,
    ///       a module ID instead of a module name, and an additional unused error number
    //@{

    void fatal(
        Module_id mod, Category cat, int /*code*/, const char* fmt, ...) PRINTFLIKE5
    {
        va_list args;
        va_start( args, fmt);
        fatal( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void error(
        Module_id mod, Category cat, int /*code*/, const char* fmt, ...) PRINTFLIKE5
    {
        va_list args;
        va_start( args, fmt);
        error( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void warning(
        Module_id mod, Category cat, int /*code*/, const char* fmt, ...) PRINTFLIKE5
    {
        va_list args;
        va_start( args, fmt);
        warning( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    void progress(
        Module_id mod, Category cat, int /*code*/, const char* fmt, ...) PRINTFLIKE5
    {
        va_list args;
        va_start( args, fmt);
        progress( SYSTEM::Module::id_to_name( mod), cat, fmt, args);
        va_end( args);
    }

    //@}

};

/// This typedef is for backwards compatibility with old code that used MI::LOG::Mod_log to access
/// the main ILogger instance.
typedef ILogger Mod_log;

/// This variable is used to access the main ILogger instance.
///
/// The name is misleading: mod_log is not the log module itself, but the main logger instance
/// (ILogger and Log_module used to be one interface in the past). Use Access_module<Log_module>
/// to access the log module itself.
extern ILogger* mod_log;

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_I_LOG_LOGGER_H
