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
/// \brief regression test infrastructure
///
/// - show<T>()               format an object of type T and return a string
/// - pretty_function_name()  pretty-print a function name as readable text
/// - pad_to_length()         add padding to reach the specified length
/// - result_column()         pad progress indicator texts to this column
///
/// The functions provided in this module are mostly used by the test
/// driver, but they might also be useful to regression test writers --
/// particularly show().

#ifndef BASE_SYSTEM_TEST_PRETTY_PRINT_H
#define BASE_SYSTEM_TEST_PRETTY_PRINT_H

#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <cctype>
#include <functional>
#include <algorithm>

namespace MI { namespace TEST {

template <class T>
inline std::string show(T const & val)
{
    std::ostringstream os;
    os.setf( std::ios::fixed );
    os.unsetf( std::ios::showpoint );
    os.precision( 16 );
    os << val;
    return os.str();
}

template <>
inline std::string show(std::ptrdiff_t const & val)
{
    std::ostringstream os;
    os << static_cast<long long int>(val);
    return os.str();
}

template <>
inline std::string show(char const & val)
{
    std::ostringstream os;            // TODO: Use isprint(3) to determine whether the
    os << '\'' << val << '\'';          // value is printable. If it isn't, show it in hex.
    return os.str();
}

#if (__cplusplus >= 201402L) || (defined(_MSC_VER) && _MSC_VER >= 1900)
template <>
inline std::string show(const std::nullptr_t&)
{
    return "<null>";
}
#endif

inline std::string pretty_function_name(std::string name)
{
    std::replace(name.begin(), name.end(), '_', ' ');
    return name;
}

inline std::string pad_to_length(std::string name, size_t pad_to_length)
{
    size_t pad( pad_to_length - std::min<size_t>(pad_to_length, name.size()) );
    if (pad)
    {
        name.push_back(' ');
        if (--pad) name.append(pad, '.');
    }
    return name;
}

inline size_t result_column() { return 63u; }

}} // MI::TEST

namespace std {

template <class Lhs, class Rhs>
inline std::ostream & operator<< (std::ostream & os, std::pair<Lhs,Rhs> const & p)
{
    return os << "(" << p.first << "," << p.second << ")";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& str, const std::vector<T>& vec)
{
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(str, " "));
    return str;
}

} // std

#endif // BASE_SYSTEM_TEST_PRETTY_PRINT_H
