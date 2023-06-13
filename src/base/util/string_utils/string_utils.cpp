/***************************************************************************************************
 * Copyright (c) 2010-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Collection of utilities for stripping chars from a string.

#include "pch.h"
#include "i_string_utils.h"

#include <algorithm>
#include <cctype>

#include <boost/algorithm/string.hpp>

namespace MI {
namespace STRING {

using std::string;
using std::wstring;

//==================================================================================================

// Stripping functionality

namespace {

// Local functor for checking against whitespace.
struct Is_not_whitespace
{
    // Constructor. This sets the used locale to the default C locale.
    Is_not_whitespace() : m_locale(std::locale::classic()) {}
    // Return whether the given character \p one is a whitespace or not.
    // \return true when one is not a whitespace, false else
    bool operator()(char one) const { return !std::isspace(one, m_locale); }
  private:
    const std::locale& m_locale;	// the locale in use
};


// Local helper for stripping leading whitespace.
string lstrip_ws(
    const string& str)		// input
{
    string::const_iterator it =
        std::find_if(str.begin(), str.end(), Is_not_whitespace());
    string::size_type pos =
        (it != str.end())? it - str.begin() : string::npos;
    // the smart string::substr() will throw when pos is out of range,
    // hence due proper clamping here
    pos = std::min(pos, str.size());

    return str.substr(pos);
}

// Local helper for stripping trailing whitespace.
string rstrip_ws(
    const string& str)		// input
{
    string::const_reverse_iterator it =
        std::find_if(str.rbegin(), str.rend(), Is_not_whitespace());
    string::size_type pos =
        (it != str.rbegin())? it.base() - str.begin() : string::npos;
    return str.substr(0, pos);
}

}


// Strip given leading and trailing characters. If no characters are given
// then all valid whitespace characters will be taken instead.
string strip(
    const string& str,		// given input
    const string& sep)		// characters to be stripped
{
    string tmp = lstrip(str, sep);
    return rstrip(tmp, sep);
}


// Strip given leading characters. If no characters are given
// then all valid whitespace characters will be taken instead.
string lstrip(
    const string& str,		// given input
    const string& sep)		// characters to be stripped
{
    if (sep.empty()) {
        return lstrip_ws(str);
    }
    else {
        string::size_type pos = str.find_first_not_of(sep);
        // the smart string::substr() would throw when pos is out of range,
        // hence do proper clamping here
        pos = std::min(pos, str.size());

        return str.substr(pos, string::npos);
    }
}


// Strip given trailing characters. If no characters are given
// then all valid whitespace characters will be taken instead.
string rstrip(
    const string& str,		// given input
    const string& sep)		// characters to be stripped
{
    if (sep.empty()) {
        return rstrip_ws(str);
    }
    else {
        string::size_type pos = str.find_last_not_of(sep);
        if (pos != string::npos)
            return str.substr(0, pos+1);
        else
            return string();
    }
}


//==================================================================================================

// Conversion functionality

// Convert the given string \p input to a string where all characters are lowercase.
void to_lower(
    string& input)
{
    // using the global ::tolower() as the STLport implementation is breaking on MAC OSX.
    std::transform(input.begin(), input.end(), input.begin(), ::tolower);
}

// Convert the given string \p input to a string where all characters are uppercase.
void to_upper(
    string& input)
{
    // using the global ::toupper() as the STLport implementation is breaking on MAC OSX.
    std::transform(input.begin(), input.end(), input.begin(), ::toupper);
}

// Convert the given string \p input to a string where all characters are lowercase.
string to_lower(
    const string& input)
{
    std::string result(input);
    to_lower(result);
    return result;
}

// Convert the given string \p input to a string where all characters are uppercase.
string to_upper(
    const string& input)
{
    std::string result(input);
    to_upper(result);
    return result;
}

// Note further code copies in mdl/compiler/compilercore/compilercore_wchar_support.cpp and in
// prod/mdl_examples/mdl_sdk/shared/utils/strings.h
string wchar_to_utf8(const wchar_t *src)
{
    string res;

    for (wchar_t const *p = src; *p != L'\0'; ++p) {
        unsigned code = *p;

        if (code <= 0x7F) {
            // 0xxxxxxx
            res += char(code);
        } else if (code <= 0x7FF) {
            // 110xxxxx 10xxxxxx
            unsigned high = code >> 6;
            unsigned low  = code & 0x3F;
            res += char(0xC0 + high);
            res += char(0x80 + low);
        } else if (0xD800 <= code && code <= 0xDBFF && 0xDC00 <= p[1] && p[1] <= 0xDFFF) {
            // surrogate pair, 0x10000 to 0x10FFFF
            unsigned high = code & 0x3FF;
            unsigned low  = p[1] & 0x3FF;
            code = 0x10000 + ((high << 10) | low);

            if (code <= 0x10FFFF) {
                // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                unsigned high = (code >> 18) & 0x07;
                unsigned mh   = (code >> 12) & 0x3F;
                unsigned ml   = (code >> 6) & 0x3F;
                unsigned low  = code & 0x3F;
                res += char(0xF0 + high);
                res += char(0x80 + mh);
                res += char(0x80 + ml);
                res += char(0x80 + low);
            } else {
                // error, replace by (U+FFFD) (or EF BF BD in UTF-8)
                res += char(0xEF);
                res += char(0xBF);
                res += char(0xBD);
            }
            ++p;
        } else if (code <= 0xFFFF) {
            if (code < 0xD800 || code > 0xDFFF) {
                // 1110xxxx 10xxxxxx 10xxxxxx
                unsigned high   = code >> 12;
                unsigned middle = (code >> 6) & 0x3F;
                unsigned low    = code & 0x3F;
                res += char(0xE0 + high);
                res += char(0x80 + middle);
                res += char(0x80 + low);
            } else {
                // forbidden surrogate part, replace by (U+FFFD) (or EF BF BD in UTF-8)
                res += char(0xEF);
                res += char(0xBF);
                res += char(0xBD);
            }
        } else if (code <= 0x10FFFF) {
            // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            unsigned high = (code >> 18) & 0x07;
            unsigned mh   = (code >> 12) & 0x3F;
            unsigned ml   = (code >> 6) & 0x3F;
            unsigned low  = code & 0x3F;
            res += char(0xF0 + high);
            res += char(0x80 + mh);
            res += char(0x80 + ml);
            res += char(0x80 + low);
        } else {
            // error, replace by (U+FFFD) (or EF BF BD in UTF-8)
            res += char(0xEF);
            res += char(0xBF);
            res += char(0xBD);
        }
    }
    return res;
}

namespace {

// Converts one utf8 character to a utf32 encoded unicode character.
//
// Note further code copies in mdl/compiler/compilercore/compilercore_wchar_support.cpp and in
// prod/mdl_examples/mdl_sdk/shared/utils/strings.h
char const *utf8_to_unicode_char(char const *up, unsigned &res)
{
    bool error = false;
    unsigned char ch = up[0];

    // find start code: either 0xxxxxxx or 11xxxxxx
    while ((ch >= 0x80) && ((ch & 0xC0) != 0xC0)) {
        ++up;
        ch = up[0];
    }

    if (ch <= 0x7F) {
        // 0xxxxxxx
        res = ch;
        up += 1;
    } else if ((ch & 0xF8) == 0xF0) {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        unsigned c1 = ch & 0x07; ch = up[1]; error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F; ch = up[2]; error |= (ch & 0xC0) != 0x80;
        unsigned c3 = ch & 0x3F; ch = up[3]; error |= (ch & 0xC0) != 0x80;
        unsigned c4 = ch & 0x3F;
        res = (c1 << 18) | (c2 << 12) | (c3 << 6) | c4;

        // must be U+10000 .. U+10FFFF
        error |= (res < 0x1000) || (res > 0x10FFFF);

        // Because surrogate code points are not Unicode scalar values, any UTF-8 byte
        // sequence that would otherwise map to code points U+D800..U+DFFF is illformed
        error |= (0xD800 <= res) && (res <= 0xDFFF);

        if (!error) {
            up += 4;
        } else {
            res = 0xFFFD;  // replacement character
            up += 1;
        }
    } else if ((ch & 0xF0) == 0xE0) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        unsigned c1 = ch & 0x0F; ch = up[1]; error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F; ch = up[2]; error |= (ch & 0xC0) != 0x80;
        unsigned c3 = ch & 0x3F;
        res = (c1 << 12) | (c2 << 6) | c3;

        // must be U+0800 .. U+FFFF
        error |= res < 0x0800;

        // Because surrogate code points are not Unicode scalar values, any UTF-8 byte
        // sequence that would otherwise map to code points U+D800..U+DFFF is illformed
        error |= (0xD800 <= res) && (res <= 0xDFFF);

        if (!error) {
            up += 3;
        } else {
            res = 0xFFFD;  // replacement character
            up += 1;
        }
    } else if ((ch & 0xE0) == 0xC0) {
        // 110xxxxx 10xxxxxx
        unsigned c1 = ch & 0x1F; ch = up[1]; error |= (ch & 0xC0) != 0x80;
        unsigned c2 = ch & 0x3F;
        res = (c1 << 6) | c2;

        // must be U+0080 .. U+07FF
        error |= res < 0x80;

        if (!error) {
            up += 2;
        } else {
            res = 0xFFFD;  // replacement character
            up += 1;
        }
    } else {
        // error
        res = 0xFFFD;  // replacement character
        up += 1;
    }
    return up;
}

} // namespace

void utf16_append(wstring &s, unsigned c);

wstring utf8_to_wchar(char const *src)
{
    wstring res;

    // skip BOM
    if (    (static_cast<unsigned char>(src[0]) == 0xEFu)
        &&  (static_cast<unsigned char>(src[1]) == 0xBBu)
        &&  (static_cast<unsigned char>(src[2]) == 0xBFu))
        src += 3;

    while (*src != '\0') {
        unsigned unicode_char;

        src = utf8_to_unicode_char(src, unicode_char);
        utf16_append(res, unicode_char);
    }
    return res;
}

// Add an unicode utf32 character to an utf16 string.
void utf16_append(wstring &s, unsigned c)
{
    // assume only valid utf32 characters added
    if (c < 0x10000) {
        s += static_cast<wchar_t>(c);
    } else {
            // encode as surrogate pair
        c -= 0x10000;
        s += static_cast<wchar_t>((c >> 10) + 0xD800);
        s += static_cast<wchar_t>((c & 0x3FF) + 0xDC00);
    }
}

#ifdef MI_PLATFORM_WINDOWS

string wchar_to_mbs(
    const wchar_t* str)
{
    string result;
    if (!str)
        return result;

    wstring wstr(str);
    // to be on the safe side, simply use twice the size
    char* buffer = new char[wstr.size()*2 + 1];
    size_t count;
    errno_t err = wcstombs_s(&count, buffer, wstr.size()*2+1, str, wstr.size());
    if (!err)
        result = string(buffer, count-1);
    delete[] buffer;

    return result;
}

#endif // MI_PLATFORM_WINDOWS

// Parse and split a string to get a token list.
void split(
    const std::string& source_str,
    const std::string& separators,
    std::vector<std::string>& token_list)
{
    using namespace boost::algorithm;

    split(token_list, source_str, is_any_of(separators), token_compress_off);
    if (token_list.back().empty()) token_list.pop_back();
}


// Case insensitive string comparison which behaves like strcasecmp and strncasecmp
int compare_case_insensitive(
    const char* s1,
    const char* s2)
{
#ifdef WIN_NT
    return _stricmp(s1, s2);
#else
    return strcasecmp(s1, s2);
#endif
}


// Just like compare_case_insensitive() but only compares the n first characters
int compare_case_insensitive(
    const char* s1,
    const char* s2,
    size_t n)
{
#ifdef WIN_NT
    return _strnicmp(s1, s2, n);
#else
    return strncasecmp(s1, s2, n);
#endif
}

// create a formated string.
std::string formatted_string(const char* format)
{
    return format;
}

}
}
