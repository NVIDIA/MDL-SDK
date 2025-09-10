/***************************************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Simple file-related utilities.

#include "pch.h"

#include "disk_utils.h"

#include <mi/base/config.h>

#ifdef MI_PLATFORM_WINDOWS
#include <io.h>
#include <base/util/string_utils/i_string_utils.h>
#else
#include <unistd.h>
#endif

namespace MI {

namespace DISK {

bool access( const char* path, bool for_writing)
{
    if( !path)
        return false;

#ifdef MI_PLATFORM_WINDOWS
    std::wstring wpath( STRING::utf8_to_wchar( path));
    bool success = ::_waccess( wpath.c_str(), for_writing ? 0x02 : 0x00) == 0;
#else
    bool success = ::access( path, for_writing ? W_OK : R_OK) == 0;
#endif

    return success;
}

std::string to_string( const std::filesystem::path& path)
{
#ifdef MI_PLATFORM_WINDOWS
#if (__cplusplus >= 202002L)
    std::u8string tmp( path.u8string());
    return { reinterpret_cast<char*>( tmp.c_str()), tmp.size() };
#else
    return path.u8string();
#endif
#else
    return path.string();
#endif
}

std::string get_extension( const std::string& filename)
{
    if( filename.empty())
        return {};

    size_t i = filename.size() - 1;

    while( true) {
        char c = filename[i];
        if( (c == '.') && (i > 0))
            return filename.substr( i+1);
        if( c == '/' || c == '\\')
            break;
        if( i == 0)
            break;
        --i;
    }

    return {};
}

} // namespace DISK

} // namespace MI
