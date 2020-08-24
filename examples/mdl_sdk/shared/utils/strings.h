/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
}}}
#endif
