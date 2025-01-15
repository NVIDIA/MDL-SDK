/***************************************************************************************************
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
 **************************************************************************************************/

#pragma once

#include "i_log_logger.h"

#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>

namespace MI {

namespace LOG {

/// Converts a LOG log severity to an API message severity.
///
/// \return   The hightest API message severity that includes the given LOG log severity.
mi::base::Message_severity convert_severity( int severity);

/// Converts an API message severity to a LOG log severity
///
/// \return   The LOG log severity, or -1 in case of failure.
int convert_severity( mi::base::Message_severity severity);

/// Converts a log category from enum into a string representation
///
/// \return   A string representation of the log category, or \c NULL in case of errors (e.g.,
///           an invalid value casted to the Category enum).
const char* convert_category_to_string( ILogger::Category category);

/// Converts a log category from string to enum representation
///
/// \param in         String representation of the log category.
/// \param[out] out   Enum representation of the log category, or undefined in case of failure.
/// \return           \c true in case of success, \c false in case of failure.
bool convert_string_to_category( const char* in, ILogger::Category& out);

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

}
}
