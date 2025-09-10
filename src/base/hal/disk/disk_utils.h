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

#ifndef BASE_HAL_DISK_UTILS_H
#define BASE_HAL_DISK_UTILS_H

#include <filesystem>
#include <string>

namespace MI {

namespace DISK {

/// Indicates whether the given path is accessible.
///
/// \param path          The path to be tested.
/// \param for_writing   \c true for write access, \c false for read access.
/// \return              \c true in case of success, \c false otherwise.
bool access( const char* path, bool for_writing = false);

/// Converts a std::filesystem::path into an UTF8 encoded std::string with native separators.
///
/// On Windows, calls path::u8string() to trigger the correct UTF8 conversion code (on C++20 and
/// later this requires unfortunately a detour of a temporary std::u8string).
///
/// On other platforms, there is no difference in the encoding between path::string() and
/// path::u8string() (only in the return type in C++20 and later), and we simply call
/// path::string() to avoid the temporary.
std::string to_string( const std::filesystem::path& path);

/// Returns the extension (without dot), or the empty string in case of failure.
///
/// Shortcut to avoid conversion to std::filesystem::path and back just to get the extension.
std::string get_extension( const std::string& filename);

} // namespace DISC

} // namespace MI

#endif // BASE_HAL_DISK_UTILS_H
