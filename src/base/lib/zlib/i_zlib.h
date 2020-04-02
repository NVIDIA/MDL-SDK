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
///
/// Trivial front-end to make the zlib module known to the system. We didn't write zlib, and the
/// remainder of this module is called directly by other modules like EXR, TIFF, and PNG. That's
/// a violation of the coding standard, but we didn't write EXR, TIFF, and PNG either. This module
/// front-end is needed to make the module name known for error messages, and for accessing zlib
/// functionality from modules that we _did_ write.

#ifndef BASE_LIB_ZLIB_H
#define BASE_LIB_ZLIB_H

#include "zlib.h"			// the actual main ZLIB header file

#include <base/system/main/types.h>
#include <cstddef>

namespace MI {
namespace ZLIB {

/// Compute/Update a CRC hash over a buffer.
///
/// \param crc  CRC to be taken into account when computing the new one.
/// \param buf  Buffer to hash.
/// \param len  Number of bytes in the buffer.
///
/// \return Computed crc.
Uint32 crc32(Uint32 crc, const void* buf, size_t len);

/// Compute a CRC hash over a buffer. Used for ATTR ID codes.
///
/// It's the equivalent of crc32(0, buf, len).
///
/// \param buf  Buffer to hash.
/// \param len  Number of bytes in the buffer.
///
/// \return Computed crc
Uint32 crc32(const void* buf, size_t len);

/// Combine 2 CRC-32 check values into one
///
/// \param crc1 First value.
/// \param crc2 Second value.
/// \param len2 Length of buffer from which the second value was obtained.
///
/// \return Combined CRC of crc1 and crc2.
Uint32 crc32_combine(Uint32 crc1, Uint32 crc2, size_t len2);

/// Set the verbosity level of the zlib. Void in non DEBUG builds.
/// \param level    New level, default 0, higher levels report more.
void set_z_verbose(int level);

/// Get the verbosity level of the zlib.
/// \return verbosity level or 0 in non DEBUG builds
int get_z_verbose();

}
}

#endif
