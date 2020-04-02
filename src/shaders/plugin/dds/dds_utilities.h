/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef SHADERS_PLUGIN_DDS_DDS_UTILITIES_H
#define SHADERS_PLUGIN_DDS_DDS_UTILITIES_H

#include <mi/base.h>
#include <mi/neuraylib/itile.h>

namespace MI {

namespace DDS {

/// The logger used by #log() below. Do not use it directly, because it might be invalid if the
/// plugin API is not available. Use #log() instead.
extern mi::base::Handle<mi::base::ILogger> g_logger;

/// Logs a message.
///
/// The message is discarded if the plugin API and the logger is not available.
void log( mi::base::Message_severity severity, const char* message);

/// Returns the number of bytes per pixel for a neuray pixel type.
mi::Uint32 get_bytes_per_pixel( const char* pixel_type);

/// Returns the number of components per pixel for a neuray pixel type.
mi::Uint32 get_components_per_pixel( const char* pixel_type);

/// Copies a rectangular region of pixels from DDS data to a neuray API tile.
///
/// This method assumes that the DDS data and the neuray API tile use the same pixel type.
///
/// The size of the rectangle is given by the size of the tile.
///
/// \param data     The DDS data to read the pixels from.
/// \param x        The x coordinate of lower left corner of the rectangle in the DDS data.
/// \param y        The y coordinate of lower left corner of the rectangle in the DDS data.
/// \param w        The width of the rectangle in the DDS data.
/// \param h        The height of the rectangle in the DDS data.
/// \param tile     The tile to write the pixels to.
/// \return         \c true in case of success, \c false otherwise.
bool copy_from_dds_to_tile(
    const unsigned char* data, mi::Uint32 x, mi::Uint32 y, mi::Uint32 w, mi::Uint32 h,
    mi::neuraylib::ITile* tile);

/// Copies a rectangular region of pixels from a neuray API tile to DDS data.
///
/// This method assumes that the DDS data and the neuray API tile use the same pixel type.
///
/// The size of the rectangle is given by the size of the tile.
///
/// \param tile     The tile to read the pixels from.
/// \param data     The DDS data to write the pixels to.
/// \param x        The x coordinate of lower left corner of the rectangle in the DDS data.
/// \param y        The y coordinate of lower left corner of the rectangle in the DDS data.
/// \param w        The width of the rectangle in the DDS data.
/// \param h        The height of the rectangle in the DDS data.
/// \return         \c true in case of success, \c false otherwise.
bool copy_from_tile_to_dds(
    const mi::neuraylib::ITile* tile,
    unsigned char* data, mi::Uint32 x, mi::Uint32 y, mi::Uint32 w, mi::Uint32 h);

} // namespace DDS

} // namespace MI

#endif // SHADERS_PLUGIN_DDS_DDS_UTILITIES_H
