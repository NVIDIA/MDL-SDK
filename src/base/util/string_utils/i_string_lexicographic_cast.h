/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The lexicographic_cast_s definition.
///
/// This utility is intended to be used like
///
/// \code
///     int i_val = 7789;
///     Likely<string> s_val = lexicographic_cast_s<string>(i_val);
///     MI_REQUIRE(s_val.get_status());
///     MI_REQUIRE_EQUAL(static_cast<string>(s_val), "7789");
/// \endcode
///
/// If something goes wrong, the \c Likely<> will reflect it with a bad status. E.g.
///
/// \code
///    string wrong_value = "ali";
///    Likely<unsigned int> i_val = lexicographic_cast_s<unsigned int>(wrong_value);
///    MI_REQUIRE(!i_val.get_status());
/// \endcode

#ifndef BASE_UTIL_STRING_UTILS_I_STRING_LEXICOGRAPHIC_CAST_H
#define BASE_UTIL_STRING_UTILS_I_STRING_LEXICOGRAPHIC_CAST_H

#include <cctype>
#include <limits>
#include <sstream>
#include <base/system/stlext/i_stlext_likely.h>

namespace MI {
namespace STRING {

/// A completely generic lexicographic case.
///
/// Due to the lack of exceptions we are using a \c Likely<> wrapper here to allow for checking for
/// failures. The suffix _s stands for 'safe'. This is an extension of the following version in
/// that it allows for both the acceptance or rejection of partially read inputs, eg a
///
/// \code
///    string value = "100ali";
///    lexicographic_cast_s<int, string, true>(value);
/// \endcode
///
/// would still be accepted, while
///
/// \code
///    string value = "100ali";
///    lexicographic_cast_s<int, string, false>(value);
/// \endcode
///
/// would not. The following (and original) version always accepts partially read inputs, btw.
///
/// This method differs from boost::lexical_cast in two aspects:
/// (1) It accepts leading whitespace (probably not intentional, but an artefact of using stream
///     operators).
/// (2) It fails for attempts to convert strings with negative numbers to unsigned integral types
///     (see comment below). The feature is actively used in quite a few places.
///
/// \param value   the source input value
/// \return        the cast-to-target-type representation of the input
template<typename Target, typename Source, bool SupportPartialInput>
STLEXT::Likely<Target> lexicographic_cast_s(
    const Source& value)
{
    Target result = Target(); // to avoid warning about uninitialized variable
    std::stringstream s;
    s << value;
    if (s.fail())
        return STLEXT::Likely<Target>(result, false);
    // Note that std::stringstream::operator>>() used below does not set the fail bit for unsigned
    // integral types if s.str() represents a small negative number (see strtoul() for details).
    // Hence, we check whether the first non-whitespace character is a '-'.
    if (std::numeric_limits<Target>::is_specialized && !std::numeric_limits<Target>::is_signed){
        const std::string& str = s.str();
        size_t i = 0;
        while (isspace(str[i]))
            ++i;
        if (str[i] == '-')
            return STLEXT::Likely<Target>(result, false);
    }
    s >> result;
    if (s.fail())
        return STLEXT::Likely<Target>(result, false);
    return STLEXT::Likely<Target>(result, SupportPartialInput? true : s.eof());
}

/// A completely generic lexicographic case.
///
/// Due to the lack of exceptions we are using a \c Likely<> wrapper here to allow for checking for
/// failures. The suffix _s stands for 'safe'.
///
/// \param value   the source input value
/// \return        the cast-to-target-type representation of the input
template<typename Target, typename Source>
STLEXT::Likely<Target> lexicographic_cast_s(
    const Source& value)
{
    return lexicographic_cast_s<Target, Source, false>(value);
}

} // namespace STRING
} // namespace MI

#endif
