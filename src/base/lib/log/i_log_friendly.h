/******************************************************************************
 * Copyright (c) 2010-2018, NVIDIA CORPORATION. All rights reserved.
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
/// \brief

#ifndef BASE_LIB_LOG_I_LOG_FRIENDLY_H
#define BASE_LIB_LOG_I_LOG_FRIENDLY_H

#include <string>
#include <sstream>

namespace MI {
namespace LOG {

struct Bytes
{
    size_t bytes;

    explicit Bytes(size_t bytes=~size_t(0))
    : bytes(bytes) {}
};


/** \brief Prints a 'readable' presentation of the provided number of bytes to the given stream.

 This function converts \p bytes to kibi-, mebi-, ..., exbibytes as appropriate and prints the
 result.
 */
inline MISTD::ostream& operator<<(MISTD::ostream& str, const Bytes bytes)
{
    if (bytes.bytes == ~size_t(0)) {
        return str << "unknown";
    }

    int i = 0;
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    double size = (double)bytes.bytes;
    while (size > 1024.) {
        size /= 1024.;
        ++i;
    }
    return str << size << ' ' << units[i];
}


/// Converts an integer memory size into a human-readable form, e.g., 16777216 -> "16 MiB".
inline MISTD::string get_readable_memory_size(size_t mem_size)
{
    MISTD::stringstream str;
    str << Bytes(mem_size);
    return str.str();
}


} // namespace LOG

}  // namespace MI

#endif // BASE_LIB_LOG_I_LOG_FRIENDLY_H
