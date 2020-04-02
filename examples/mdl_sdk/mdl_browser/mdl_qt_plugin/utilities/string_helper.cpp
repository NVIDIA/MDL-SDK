/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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


#include "string_helper.h"
#include <algorithm>

std::string String_helper::replace(const std::string& input,
    const std::string& old, const std::string& with)
{
    if (input.empty()) return input;

    std::string sentance(input);
    size_t offset(0);
    size_t pos(0);
    while (pos != std::string::npos)
    {
        pos = sentance.find(old, offset);
        if (pos == std::string::npos)
            break;

        sentance.replace(pos, old.length(), with);
        offset = pos + with.length();
    }
    return sentance;
}

std::string String_helper::replace(const std::string& input,
    char old, char with)
{
    std::string output(input);
    std::replace(output.begin(), output.end(), old, with);
    return output;
}

std::vector<std::string> String_helper::split(const std::string& input, char sep)
{
    std::vector<std::string> chunks;

    size_t offset(0);
    size_t pos(0);
    while (pos != std::string::npos)
    {
        pos = input.find(sep, offset);

        if (pos == std::string::npos)
        {
            chunks.push_back(input.substr(offset));
            break;
        }

        chunks.push_back(input.substr(offset, pos - offset));
        offset = pos + 1;
    }
    return chunks;
}

std::vector<std::string> String_helper::split(const std::string& input, const std::string& sep)
{
    std::vector<std::string> chunks;

    size_t offset(0);
    size_t pos(0);
    while (pos != std::string::npos)
    {
        pos = input.find(sep, offset);

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
