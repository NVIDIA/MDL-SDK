/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Collection of utilities for strings.

#ifndef BASE_UTIL_STRING_UTILS_I_STRING_UTILS_H
#define BASE_UTIL_STRING_UTILS_I_STRING_UTILS_H

#include <string>
#include <vector>

namespace MI {
namespace STRING {

/// \name Stripping
/// Stripping utilities.
//@{
/// Strip given leading and trailing characters. If no characters are given
/// then all valid whitespace characters will be taken instead.
/// \param str given input
/// \param sep characters to be stripped
/// \return stripped input
std::string strip(
    const std::string& str,
    const std::string& sep=std::string());

/// Strip given leading characters. If no characters are given
/// then all valid whitespace characters will be taken instead.
/// \param str given input
/// \param sep characters to be stripped
/// \return stripped input
std::string lstrip(
    const std::string& str,
    const std::string& sep=std::string());

/// Strip given trailing characters. If no characters are given
/// then all valid whitespace characters will be taken instead.
/// \param str given input
/// \param sep characters to be stripped
/// \return stripped input
std::string rstrip(
    const std::string& str,
    const std::string& sep=std::string());
//@}


/// \name Conversions
/// Conversion utilities.
//@{
/// Convert the given string \p input to a string where all characters are lowercase.
/// \param[in,out] input the input string
void to_lower(
    std::string& input);

/// Convert the given string \p input to a string where all characters are uppercase.
/// \param[in,out] input the input string
void to_upper(
    std::string& input);

/// Convert the given string \p input to a string where all characters are lowercase.
/// \param input the input string
/// \return converted string
std::string to_lower(
    const std::string& input);

/// Convert the given string \p input to a string where all characters are uppercase.
/// \param input the input string
/// \return converted string
std::string to_upper(
    const std::string& input);
//@}

/// Convert the given char input of UTF-8 format into a wchar.
///
/// Code points beyong U+FFFF are replaced by '?', even if the underlying implementation of wchar
/// (e.g. 4 byte on Linux) could handle that code point.
std::wstring utf8_to_wchar(
    const char* str);

#ifdef WIN_NT
/// Convert the given wchar string input into a multibyte char string output.
std::string wchar_to_mbs(
    const wchar_t* str);

/// Converts a wchar_t * string into an utf8 encoded string.
std::string wchar_to_utf8(
    const wchar_t* str);

#endif

/// Parse and get a token list.
/// A convenient function for getting a token list.
///
/// Example:
/// \code
///   std::vector<std::string> token_list;
///   Tokenizer::parse("A Non-Manifold,Mesh", " ,", token_list);
/// \endcode
/// gives
///   token_list[0] == "A";
///   token_list[1] == "Non-Manifold";
///   token_list[2] == "Mesh";
///
/// Notice: If the last entry is empty, it does not count.
///
/// For example, tokenize "hello," with "," gives only one token {
/// "hello" } instead of two tokens {"hello", ""}. But, tokenize
/// "hello,,world" with "," gives three tokens { "hello", "",
/// "world" }.
///
/// \param[in] source_str the source string
/// \param[in] separators the separators
/// \param[in,out] token_list the tokens will end up here
void split(
    const std::string& source_str,
    const std::string& separators,
    std::vector<std::string>& token_list);

/// Case insensitive string comparison which behaves like strcasecmp and strncasecmp
/// \param s1 string 1
/// \param s2 string 2
/// \return an integer less than, equal to, or greater than zero if s1 is found,
///         respectively, to be less than, to match, or be greater than s2.
int compare_case_insensitive(
    const char* s1,
    const char* s2);


/// Just like compare_case_insensitive() but only compares the n first characters
/// \param s1 string 1
/// \param s2 string 2
/// \param n  length to compare
/// \return an integer less than, equal to, or greater than zero if s1 (or the
///         first n bytes thereof) is found, respectively, to be less than,
///         to match, or be greater than s2.
int compare_case_insensitive(
    const char* s1,
    const char* s2,
    size_t n);
    

/// create a formated string.
/// \param  format  printf-like format string
/// \param  args    arguments to insert into the format string
/// \return the formated string
template <typename... Args>
std::string formatted_string(const char *format, Args ... args)
{
    // get string size + 1 for null terminator to allocate a string of correct size
    int size = 1 + snprintf(nullptr, 0, format, std::forward<Args>(args)...);

    std::string s;
    s.resize(size);
    snprintf(&s[0], size, format, std::forward<Args>(args)...);
    return s.substr(0, size - 1);
}

/// create a formated string.
/// \param  format  printf-like format string
/// \param  args    arguments to insert into the format string
/// \return the formated string
template <typename... Args>
std::string formatted_string(const std::string& format, Args ... args)
{
    return formatted_string(format.c_str(), std::forward<Args>(args)...);
}

}
}

#endif
