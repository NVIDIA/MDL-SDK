/******************************************************************************
 * Copyright (c) 2007-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test infrastructure
///
/// - class Current_location          identify a source code context
/// - MI_CURRENT_LOCATION             generate a location identifier
/// - MI_CURRENT_FUNCTION             name of the current function scope

#ifndef BASE_SYSTEM_TEST_CURRENT_LOCATION_H
#define BASE_SYSTEM_TEST_CURRENT_LOCATION_H

#include <string>
#include <ostream>

namespace MI { namespace TEST {

struct Current_location
{
    Current_location( std::string const &     f
                    , unsigned int              l
                    , std::string const &     func
                    )
    : file(f), line(l), function(func)
    {
    }

    std::string       file;
    unsigned int        line;
    std::string       function;
};

inline std::ostream & operator<< (std::ostream & os, Current_location const & l)
{
    return os << l.file << ':' << l.line << ':' << l.function;
}

#if defined(__FUNCSIG__)
#  define MI_CURRENT_FUNCTION __FUNCSIG__
#elif defined(__GNUC__) || (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901))
#  define MI_CURRENT_FUNCTION __func__
#else
#  define MI_CURRENT_FUNCTION "unknown"
#endif

#if defined(__GNUC__)
#  define MI_CURRENT_FUNCTION_PRETTY __PRETTY_FUNCTION__
#else
#  define MI_CURRENT_FUNCTION_PRETTY MI_CURRENT_FUNCTION
#endif

#define MI_CURRENT_LOCATION MI::TEST::Current_location(__FILE__, __LINE__, MI_CURRENT_FUNCTION)

}} // MI::TEST

#endif // BASE_SYSTEM_TEST_CURRENT_LOCATION_H

