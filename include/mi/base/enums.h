/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/base/enums.h
/// \brief Basic enums.
///
/// See \ref mi_base_ilogger.

#ifndef MI_BASE_ENUMS_H
#define MI_BASE_ENUMS_H

#include <mi/base/assert.h>

namespace mi {

namespace base {

/// Namespace for details of the Base API.
/// \ingroup mi_base
namespace details {

/** \addtogroup mi_base_ilogger
@{
*/

/// Constants for possible message severities.
///
/// \see #mi::base::ILogger::message()
///      
enum Message_severity
{
    /// A fatal error has occurred.
    MESSAGE_SEVERITY_FATAL          = 0,
    /// An error has occurred.
    MESSAGE_SEVERITY_ERROR          = 1,
    /// A warning has occurred.
    MESSAGE_SEVERITY_WARNING        = 2,
    /// This is a normal operational message.
    MESSAGE_SEVERITY_INFO           = 3,
    /// This is a more verbose message.
    MESSAGE_SEVERITY_VERBOSE        = 4,
    /// This is debug message.
    MESSAGE_SEVERITY_DEBUG          = 5,
    //  Undocumented, for alignment only
    MESSAGE_SEVERITY_FORCE_32_BIT   = 0xffffffffU
};

mi_static_assert( sizeof( Message_severity) == 4);

/*@}*/ // end group mi_base_ilogger

}

using namespace details;

} // namespace base

} // namespace mi

#endif // MI_BASE_ENUMS_H
