/******************************************************************************
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/shared/utils/strings.h
 //
 // Code shared by all examples

#ifndef EXAMPLE_SHARED_UTILS_STRINGS_H
#define EXAMPLE_SHARED_UTILS_STRINGS_H

#include <type_traits>
#include <stdint.h>
#include <locale>
#include <codecvt>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <map>

#include <mi/base/enums.h>
#include <mi/neuraylib/imdl_execution_context.h>

namespace mi { namespace examples { namespace strings
{
    /// convert std::string to std::wstring.
    inline std::wstring str_to_wstr(const std::string& s)
    {
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.from_bytes(s);
    }

    // --------------------------------------------------------------------------------------------

    /// convert std::wstring to std::string.
    inline std::string wstr_to_str(const std::wstring& s)
    {
        using convert_type = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.to_bytes(s);
    }

    // --------------------------------------------------------------------------------------------

 // utility to convert from UTF8 to wide chars
 #define BOM8A ((unsigned char)0xEF)
 #define BOM8B ((unsigned char)0xBB)
 #define BOM8C ((unsigned char)0xBF)

 // Convert the given char input of UTF-8 format into a wchar.
     inline std::wstring utf8_to_wchar(const char* str)
     {
         long b = 0, c = 0;
         if ((unsigned char)str[0] == BOM8A && (unsigned char)str[1] == BOM8B && (unsigned char)str[2] == BOM8C)
             str += 3;
         for (const unsigned char* a = (unsigned char*)str; *a; a++)
             if (((unsigned char)*a) < 128 || (*a & 192) == 192)
                 c++;
         wchar_t* buf = new wchar_t[c + 1];
         buf[c] = 0;
         for (unsigned char* a = (unsigned char*)str; *a; a++) {
             if (!(*a & 128))
                 //Byte represents an ASCII character. Direct copy will do.
                 buf[b] = *a;
             else if ((*a & 192) == 128)
                 //Byte is the middle of an encoded character. Ignore.
                 continue;
             else if ((*a & 224) == 192)
                 //Byte represents the start of an encoded character in the range U+0080 to U+07FF
                 buf[b] = ((*a & 31) << 6) | (a[1] & 63);
             else if ((*a & 240) == 224)
                 //Byte represents the start of an encoded character in the range U+07FF to U+FFFF
                 buf[b] = ((*a & 15) << 12) | ((a[1] & 63) << 6) | (a[2] & 63);
             else if ((*a & 248) == 240) {
                 //Byte represents the start of an encoded character beyond U+FFFF limit of 16-bit ints
                 buf[b] = '?';
             }
             b++;
         }

         std::wstring wstr(buf, c);
         delete[] buf;

         return wstr;
     }

    // --------------------------------------------------------------------------------------------

    /// Converts a wchar_t * string into an utf8 encoded string.
    inline std::string wchar_to_utf8(const wchar_t* src)
    {
        std::string res;

        for (wchar_t const *p = src; *p != L'\0'; ++p) 
        {
            unsigned code = *p;

            if (code <= 0x7F) 
            {
                // 0xxxxxxx
                res += char(code);
            } 
            else if (code <= 0x7FF) 
            {
                // 110xxxxx 10xxxxxx
                unsigned high = code >> 6;
                unsigned low  = code & 0x3F;
                res += char(0xC0 + high);
                res += char(0x80 + low);
            }
            else if (0xD800 <= code && code <= 0xDBFF && 0xDC00 <= p[1] && p[1] <= 0xDFFF)
            {
                // surrogate pair, 0x10000 to 0x10FFFF
                unsigned high = code & 0x3FF;
                unsigned low  = p[1] & 0x3FF;
                code = 0x10000 + ((high << 10) | low);

                if (code <= 0x10FFFF)
                {
                    // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                    unsigned high = (code >> 18) & 0x07;
                    unsigned mh   = (code >> 12) & 0x3F;
                    unsigned ml   = (code >> 6) & 0x3F;
                    unsigned low  = code & 0x3F;
                    res += char(0xF0 + high);
                    res += char(0x80 + mh);
                    res += char(0x80 + ml);
                    res += char(0x80 + low);
                }
                else 
                {
                    // error, replace by (U+FFFD) (or EF BF BD in UTF-8)
                    res += char(0xEF);
                    res += char(0xBF);
                    res += char(0xBD);
                }
            }
            else if (code <= 0xFFFF)
            {
                if (code < 0xD800 || code > 0xDFFF)
                {
                    // 1110xxxx 10xxxxxx 10xxxxxx
                    unsigned high   = code >> 12;
                    unsigned middle = (code >> 6) & 0x3F;
                    unsigned low    = code & 0x3F;
                    res += char(0xE0 + high);
                    res += char(0x80 + middle);
                    res += char(0x80 + low);
                }
                else
                {
                    // forbidden surrogate part, replace by (U+FFFD) (or EF BF BD in UTF-8)
                    res += char(0xEF);
                    res += char(0xBF);
                    res += char(0xBD);
                }
            }
            else if (code <= 0x10FFFF)
            {
                // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                unsigned high = (code >> 18) & 0x07;
                unsigned mh   = (code >> 12) & 0x3F;
                unsigned ml   = (code >> 6) & 0x3F;
                unsigned low  = code & 0x3F;
                res += char(0xF0 + high);
                res += char(0x80 + mh);
                res += char(0x80 + ml);
                res += char(0x80 + low);
            }
            else
            {
                // error, replace by (U+FFFD) (or EF BF BD in UTF-8)
                res += char(0xEF);
                res += char(0xBF);
                res += char(0xBD);
            }
        }
        return res;
    }

    // --------------------------------------------------------------------------------------------

    /// split a string into chunks at a separation character.
    inline std::vector<std::string> split(
        const std::string& input,
        char sep)
    {
        std::vector<std::string> chunks;

        size_t offset(0);
        size_t pos(0);
        std::string chunk;
        while (pos != std::string::npos)
        {
            pos = input.find(sep, offset);

            if (pos == std::string::npos)
                chunk = input.substr(offset);
            else
                chunk = input.substr(offset, pos - offset);

            if (!chunk.empty())
                chunks.push_back(chunk);
            offset = pos + 1;
        }
        return chunks;
    }

    // --------------------------------------------------------------------------------------------

    /// split a string into chunks at a separation substring.
    inline std::vector<std::string> split(
        const std::string& input,
        const std::string& sep)
    {
        std::vector<std::string> chunks;

        size_t offset(0);
        while (true)
        {
            size_t pos = input.find(sep, offset);

            if (pos == std::string::npos)
            {
                chunks.push_back(input.substr(offset));
                break;
            }

            chunks.push_back(input.substr(offset, pos - offset));
            offset = pos + sep.length();
        }
        return chunks;
    }

    // --------------------------------------------------------------------------------------------

    /// checks if a string starts with a given prefix.
    inline bool starts_with(const std::string& s, const std::string& potential_start)
    {
        size_t n = potential_start.size();

        if (s.size() < n)
            return false;

        for (size_t i = 0; i < n; ++i)
            if (s[i] != potential_start[i])
                return false;

        return true;
    }

    // --------------------------------------------------------------------------------------------

    /// checks if a string ends with a given suffix.
    inline bool ends_with(const std::string& s, const std::string& potential_end)
    {
        size_t n = potential_end.size();
        size_t sn = s.size();

        if (sn < n)
            return false;

        for (size_t i = 0; i < n; ++i)
            if (s[sn - i - 1] != potential_end[n - i - 1])
                return false;

        return true;
    }

    // --------------------------------------------------------------------------------------------

    /// replaces substrings within a given string.
    inline std::string replace(
        const std::string& input,
        const std::string& old,
        const std::string& with)
    {
        if (input.empty()) return input;

        std::string result(input);
        size_t offset(0);
        while (true)
        {
            size_t pos = result.find(old, offset);
            if (pos == std::string::npos)
                break;

            result.replace(pos, old.length(), with);
            offset = pos + with.length();
        }
        return result;
    }

    // --------------------------------------------------------------------------------------------

    /// replaces characters within a given string.
    inline std::string replace(
        const std::string& input,
        char old,
        char with)
    {
        // added this function for consistency with replace substring
        std::string output(input);
        std::replace(output.begin(), output.end(), old, with);
        return output;
    }

    // --------------------------------------------------------------------------------------------

    // get the query substring of an url if present otherwise returns an empty string.
    // also, drops the hash segment in case it is present.
    inline std::string get_url_query(const std::string& url)
    {
        std::string query = url;
        size_t pos = query.find_first_of('?');
        if (pos == std::string::npos)
            return "";
        else
            query = query.substr(pos + 1);

        // drop the hash if present at at the end of the query
        pos = query.find_first_of('#');
        if (pos != std::string::npos)
            query = query.substr(0, pos);
        return query;
    }

    // parses a query of an url and returns a map of key value pairs
    inline std::map<std::string, std::string> parse_url_query(const std::string& query)
    {
        std::map<std::string, std::string> result;
        std::vector<std::string> chunks = split(query, '&');
        for (auto& c : chunks)
        {
            size_t eqPos = c.find_first_of('=');
            if (eqPos == std::string::npos)
            {
                result.insert({ c, "" });
            }
            else
            {
                result.insert({ c.substr(0, eqPos), c.substr(eqPos + 1) });
            }
        }
        return result;
    }

    // --------------------------------------------------------------------------------------------

    /// create a formated string.
    /// \param  format  printf-like format string
    /// \param  args    arguments to insert into the format string
    /// \return the formated string
    template <typename... Args>
    inline std::string format(const char *format_string, Args ... args)
    {
        // get string size + 1 for null terminator to allocate a string of correct size
        int size = 1 + snprintf(nullptr, 0, format_string, std::forward<Args>(args)...);

        std::string s;
        s.resize(size);
        snprintf(&s[0], size, format_string, std::forward<Args>(args)...);
        return s.substr(0, size - 1);
    }

    /// create a formated string (variadic base function).
    /// \return the unchanged \format_string
    inline std::string format(const char *format_string)
    {
        return format_string;
    }

    /// create a formated string (variadic base function, needed for __VA_ARGS__ mappings).
    /// \return the empty string
    inline std::string format()
    {
        return "";
    }

    // --------------------------------------------------------------------------------------------

    /// removes leading and trailing quotes if there are some.
    /// returns true when it was a non-quoted string or valid quoted string before.
    /// returns false for single quotes and when a quote was only found at one end.
    inline bool remove_quotes(std::string& s)
    {
        size_t l = s.length();
        if (l == 0)
            return true;

        bool leading = s[0] == '\"';
        if (l == 1)
            return !leading; // one single quote

        bool trailing = s[l - 1] == '\"';
        if (leading != trailing) // quote one one side only
            return false;

        if (leading)
            s = s.substr(1, l - 2); // remove quotes on both sides
        return true;
    }

    // --------------------------------------------------------------------------------------------

    // Returns a string-representation of the given message severity
    inline std::string to_string(mi::base::Message_severity severity)
    {
        switch (severity)
        {
            case mi::base::MESSAGE_SEVERITY_FATAL:
                return "fatal";
            case mi::base::MESSAGE_SEVERITY_ERROR:
                return "error";
            case mi::base::MESSAGE_SEVERITY_WARNING:
                return "warning";
            case mi::base::MESSAGE_SEVERITY_INFO:
                return "info";
            case mi::base::MESSAGE_SEVERITY_VERBOSE:
                return "verbose";
            case mi::base::MESSAGE_SEVERITY_DEBUG:
                return "debug";
            default:
                break;
        }
        return "";
    }

    // --------------------------------------------------------------------------------------------

    // Returns a string-representation of the given message category
    inline std::string to_string(mi::neuraylib::IMessage::Kind message_kind)
    {
        switch (message_kind)
        {
            case mi::neuraylib::IMessage::MSG_INTEGRATION:
                return "MDL SDK";
            case mi::neuraylib::IMessage::MSG_IMP_EXP:
                return "Importer/Exporter";
            case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
                return "Compiler Backend";
            case mi::neuraylib::IMessage::MSG_COMILER_CORE:
                return "Compiler Core";
            case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
                return "Compiler Archive Tool";
            case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
                return "Compiler DAG generator";
            default:
                break;
        }
        return "";
    }

    // --------------------------------------------------------------------------------------------
    
    /// Convert the given value into its string representation.
    /// \return string resembling the given input parameter
    template <typename T>
    std::string lexicographic_cast(
        T value)
    {
        std::stringstream s;
        s << value;
        return s.str();
    }

    // --------------------------------------------------------------------------------------------

    /// Convert the given string into the given value.
    /// \return value resembling the given input string
    template <typename T>
    T lexicographic_cast(
        const std::string& str)
    {
        std::stringstream s;
        s << str;
        T result = T();
        s >> result;
        return result;
    }
    
}}}
#endif
