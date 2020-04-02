/******************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief Abstract interface to debug logging.
 ///
 ///      MI_MSG_DEBUG(module, category, fmt, ...)
 ///      MI_MSG_VDEBUG(module, category, fmt, ...)
 ///          Log snprintf()-formatted debug messages.
 ///
 ///      MI_STREAM_DEBUG(module, category)
 ///      MI_STREAM_VDEBUG(module, category)
 ///          Log ostream-formatted debug messages.
 ///
 /// Description:
 ///
 ///      If the preprocessor symbol NDEBUG is defined at compile-time, all debug
 ///      logging is disabled.
 ///
 ///      Log messages are inherently line-oriented. Avoid end-of-line characters
 ///      such as CR ('\r') or LF ('\n') in log messages: not every logging
 ///      backend may deal with them the same way.
 ///
 ///      The snprintf()-like interface is used as follows:
 ///
 ///          MI_MSG_DEBUG(M_MAIN, ILogger::C_MAIN, "%s %s", "hello ", "world");
 ///
 ///          MI_MSG_VDEBUG
 ///          (
 ///              M_MAIN, ILogger::C_MAIN, "Usage: %s [files...]", argv[0]
 ///          );
 ///
 ///      The std::ostream-like interface is used as follows:
 ///
 ///          MI_STREAM_DEBUG(M_MAIN, ILogger::C_MAIN)
 ///              << "hello " << ' ' << "world";
 ///
 ///          MI_STREAM_VDEBUG(M_MAIN, ILogger::C_MAIN)
 ///              << "Usage: " << argv[0] << " [files...]";
 ///

#ifndef BASE_LIB_LOG_MACROS_H
#define BASE_LIB_LOG_MACROS_H

#include "i_log_logger.h"
#include "i_log_stream.h"

// ----------------------------------------------------------------------------
//                           Log snprintf()-style
// ----------------------------------------------------------------------------

#ifdef NDEBUG
namespace MI { namespace LOG { namespace INTERNAL {
    // The following class is used in non-debug mode to "fake" an access to all
    // arguments of MI_MSG_DEBUG() and MI_MSG_VDEBUG() in order to avoid
    // potential "unused variable" compiler warnings.
    struct Ignore_arguments
    {
        template <typename T>
        Ignore_arguments const & operator, (T const &) const
        {
            return *this;
        }
    };
}}}
#endif

#ifndef MI_MSG_DEBUG
#  ifndef NDEBUG
#    define MI_MSG_DEBUG(module, category, ...)                         \
           MI::LOG::mod_log->debug(module, category, __VA_ARGS__)
#  else
#    define MI_MSG_DEBUG(module, category, ...)                         \
       do                                                               \
       {                                                                \
           if (false)                                                   \
           {                                                            \
               MI::LOG::INTERNAL::Ignore_arguments a;                   \
               a, __VA_ARGS__;                                          \
           }                                                            \
       }                                                                \
       while (false)
#  endif
#endif

#ifndef MI_MSG_VDEBUG
#  ifndef NDEBUG
#    define MI_MSG_VDEBUG(module, category, ...)                        \
           MI::LOG::mod_log->debug(module, category, __VA_ARGS__)
#  else
#    define MI_MSG_VDEBUG(module, category, ...)                        \
       do                                                               \
       {                                                                \
           if (false)                                                   \
           {                                                            \
               MI::LOG::INTERNAL::Ignore_arguments a;                   \
               a, __VA_ARGS__;                                          \
           }                                                            \
       }                                                                \
       while (false)
#  endif
#endif

// ----------------------------------------------------------------------------
//                          Log std::ostream-style
// ----------------------------------------------------------------------------

namespace MI { namespace LOG { namespace MESSAGE
{
#ifdef NDEBUG
    struct Dummy
    {
        template <class T>
        Dummy & operator<< (T const &) { return *this; }
    };
#endif // NDEBUG
}}} // MI::LOG::MESSAGE

#ifndef MI_STREAM_DEBUG
#  ifndef NDEBUG
#    define MI_STREAM_DEBUG(mid, cid)  MI::LOG::MESSAGE::Debug(mid, cid)
#  else
#    define MI_STREAM_DEBUG(mid, cid)  if (false) MI::LOG::MESSAGE::Dummy()
#  endif
#endif

#ifndef MI_STREAM_VDEBUG
#  ifndef NDEBUG
#    define MI_STREAM_VDEBUG(mid, cid) MI::LOG::MESSAGE::Vdebug(mid, cid)
#  else
#    define MI_STREAM_VDEBUG(mid, cid) if (false) MI::LOG::MESSAGE::Dummy()
#  endif
#endif

#endif // BASE_LIB_LOG_MACROS_H
