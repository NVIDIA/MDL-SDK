/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Header for the Log_utilities, Forwarding_logger, and Receiving_logger implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_LOG_UTILITIES_H
#define API_API_NEURAY_NEURAY_LOG_UTILITIES_H

#include <mi/base/enums.h>
#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>

#include <boost/core/noncopyable.hpp>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/log/i_log_target.h>

namespace MI {

namespace NEURAY {

class Log_utilities : public boost::noncopyable
{
public:

    // public API methods

    // (none)

    // internal methods

    /// Converts a log category from enum into a string representation
    ///
    /// \return   A string representation of the log category, or \c NULL in case of errors (e.g.,
    ///           an invalid value casted to the Category enum).
    static const char* convert_category_to_string( LOG::Mod_log::Category category);

    /// Converts a log category from string to enum representation
    ///
    /// \param in         String representation of the log category.
    /// \param[out] out   Enum representation of the log category, or undefined in case of failure.
    /// \return           \c true in case of success, \c false in case of failure.
    static bool convert_string_to_category( const char* in, LOG::Mod_log::Category& out);

};

/// This logger forwards all messages to the LOG module.
class Forwarding_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(
        mi::base::Message_severity level,
        const char* category,
        const mi::base::Message_details&,
        const char* message);
};

/// This logger adapts mi::base::ILogger to LOG::ILog_target and is used for receiving logger.
class Receiving_logger : public LOG::ILog_target, public boost::noncopyable
{
public:
    /// Constructor. Adapts the passed logger.
    Receiving_logger( mi::base::ILogger* logger);

    /// Destructor .
    ~Receiving_logger();

    bool message(
        const char* module,
        LOG::Mod_log::Category category,
        LOG::Mod_log::Severity severity,
        const mi::base::Message_details& det,
        const char* prefix,
        const char* message);

private:

    /// The adapted logger.
    mi::base::ILogger* m_logger;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_LOG_UTILITIES_H
