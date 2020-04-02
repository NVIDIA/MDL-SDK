/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The third-party compression library module.

#include "pch.h"
#include "i_zlib.h"
#ifdef DEBUG
#include "zutil.h"
#endif // DEBUG

namespace MI {
namespace ZLIB {

/// Compute/Update a CRC hash over a buffer.
///
/// \param crc  CRC to be taken into account when computing the new one.
/// \param buf  Buffer to hash.
/// \param len  Number of bytes in the buffer.
///
/// \return Computed crc.
Uint32 crc32(Uint32 crc, const void* buf, size_t len)
{
    return (Uint32)::crc32(crc, (const unsigned char *)buf, (unsigned)len);
}

/// Compute a CRC hash over a buffer. Used for ATTR ID codes.
///
/// It's the equivalent of crc32(0, buf, len).
///
/// \param buf  Buffer to hash.
/// \param len  Number of bytes in the buffer.
///
/// \return Computed crc
Uint32 crc32(const void* buf, size_t len)
{
    return (Uint32)::crc32(0, (const unsigned char *)buf, (unsigned)len);
}

Uint32 crc32_combine(Uint32 crc1, Uint32 crc2, size_t len2)
{
    return (Uint32)::crc32_combine(crc1, crc2, len2);
}

///
/// set the verbosity level of the zlib. Void in non DEBUG builds.
/// The global variable z_verbose is only defined for DEBUG builds
/// as a zlib internal variable. It is set to 0 by default which will
/// cause zlib to print occasionally a warning to stdout. Using this
/// function allows to overwrite this behavior and set the value to -1.
///

void set_z_verbose(int level)
{
#ifdef DEBUG
    z_verbose = level;
#endif // DEBUG
    level = level; // stop compiler warnings about unused parameter
}

///
/// get the verbosity level of the zlib. Returns 0 in non DEBUG builds.
/// The global variable z_verbose is only defined for DEBUG builds
/// as a zlib internal variable.
///

int get_z_verbose()
{
#ifdef DEBUG
    return z_verbose;
#else // DEBUG
    return 0;
#endif // DEBUG    
}

}}	// namespace MI::ZLIB
