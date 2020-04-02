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
/// \brief Collection of utilities for stripping chars from a string.

#include "pch.h"
#include "i_string_utils.h"

#include <algorithm>
#include <cctype>
#include <locale>
#include <cstring>

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
    to_lower(static_cast<string&>(result));
    return result;
}

// Convert the given string \p input to a string where all characters are uppercase.
string to_upper(
    const string& input)
{
    std::string result(input);
    to_upper(static_cast<string&>(result));
    return result;
}


// utility to convert from UTF8 to wide chars
#define BOM8A ((unsigned char)0xEF)
#define BOM8B ((unsigned char)0xBB)
#define BOM8C ((unsigned char)0xBF) 

// Convert the given char input of UTF-8 format into a wchar.
wstring utf8_to_wchar(
    const char* str)
{
    long b=0, c=0;
    if ((unsigned char)str[0]==BOM8A && (unsigned char)str[1]==BOM8B &&(unsigned char)str[2]==BOM8C)
        str+=3;
    for (const unsigned char *a=(unsigned char *)str;*a;a++)
        if (((unsigned char)*a)<128 || (*a&192)==192)
            c++;
    wchar_t *buf= new wchar_t[c+1];
    buf[c]=0;
    for (unsigned char *a=(unsigned char*)str;*a;a++){
        if (!(*a&128))
            //Byte represents an ASCII character. Direct copy will do.
            buf[b]=*a;
        else if ((*a&192)==128)
            //Byte is the middle of an encoded character. Ignore.
            continue;
        else if ((*a&224) == 192)
            //Byte represents the start of an encoded character in the range U+0080 to U+07FF
            buf[b]=((*a&31)<<6) | (a[1]&63);
        else if ((*a&240) == 224)
            //Byte represents the start of an encoded character in the range U+07FF to U+FFFF
            buf[b]=((*a&15)<<12) | ((a[1]&63)<<6) | (a[2]&63);
        else if ((*a&248) == 240){
            //Byte represents the start of an encoded character beyond U+FFFF limit of 16-bit ints
            buf[b]='?';
        }
        b++;
    }

    wstring wstr(buf, c);
    delete[] buf;

    return wstr;
}

#ifdef WIN_NT
// Convert the given wchar string input into a multibyte char string output.
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

// Converts a wchar_t * string into an utf8 encoded string.
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

#endif

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

}
}
