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

#include "pch.h"

#include "dds_utilities.h"
#include <cstring>

namespace MI {

namespace DDS {

mi::base::Handle<mi::base::ILogger> g_logger;

void log( mi::base::Message_severity severity, const char* message)
{
    if( !g_logger.is_valid_interface())
        return;

    g_logger->message( severity, "DDS:IMAGE", message);
}

mi::Uint32 get_bytes_per_pixel( const char* pixel_type)
{
    if( strcmp( pixel_type, "Sint8"   ) == 0) return  1;
    if( strcmp( pixel_type, "Float32" ) == 0) return  4;
    if( strcmp( pixel_type, "Rgb"     ) == 0) return  3;
    if( strcmp( pixel_type, "Rgba"    ) == 0) return  4;
    if( strcmp( pixel_type, "Rgba_16" ) == 0) return  8;
    if( strcmp( pixel_type, "Color"   ) == 0) return 16;
    return 0;
}

mi::Uint32 get_components_per_pixel( const char* pixel_type)
{
    if( strcmp( pixel_type, "Sint8"   ) == 0) return  1;
    if( strcmp( pixel_type, "Float32" ) == 0) return  1;
    if( strcmp( pixel_type, "Rgb"     ) == 0) return  3;
    if( strcmp( pixel_type, "Rgba"    ) == 0) return  4;
    if( strcmp( pixel_type, "Rgba_16" ) == 0) return  4;
    if( strcmp( pixel_type, "Color"   ) == 0) return  4;
    return 0;
}

bool copy_from_dds_to_tile(
    const unsigned char* data,
    mi::Uint32 x_start,
    mi::Uint32 y_start,
    mi::Uint32 data_width,
    mi::Uint32 data_height,
    mi::neuraylib::ITile* tile)
{
    // Compute the rectangular region that is to be copied
    mi::Uint32 tile_width  = tile->get_resolution_x();
    mi::Uint32 tile_height = tile->get_resolution_y();
    mi::Uint32 x_end = std::min( x_start + tile_width,  data_width);
    mi::Uint32 y_end = std::min( y_start + tile_height, data_height);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( tile->get_type());
    if( bytes_per_pixel == 0)
         return false;

    // Copy pixel data using memcpy() for each scanline.
    mi::Uint32 bytes_per_scanline = (x_end-x_start) * bytes_per_pixel;
    const unsigned char* src = data + (y_start * data_width + x_start) * bytes_per_pixel;
    char* dest = static_cast<char*>( tile->get_data());
    for( mi::Uint32 y = y_start; y < y_end; ++y) {
        memcpy( dest, src, bytes_per_scanline);
        src  += data_width * bytes_per_pixel;
        dest += tile_width * bytes_per_pixel;
    }
    return true;
}

bool copy_from_tile_to_dds(
    const mi::neuraylib::ITile* tile,
    unsigned char* data,
    mi::Uint32 x_start,
    mi::Uint32 y_start,
    mi::Uint32 data_width,
    mi::Uint32 data_height)
{
    // Compute the rectangular region that is to be copied
    mi::Uint32 tile_width  = tile->get_resolution_x();
    mi::Uint32 tile_height = tile->get_resolution_y();
    mi::Uint32 x_end = std::min( x_start + tile_width,  data_width);
    mi::Uint32 y_end = std::min( y_start + tile_height, data_height);

    mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( tile->get_type());
    if( bytes_per_pixel == 0)
         return false;

    // Copy pixel data using memcpy() for each scanline.
    mi::Uint32 bytes_per_scanline = (x_end-x_start) * bytes_per_pixel;
    const char* src = static_cast<const char*>( tile->get_data());
    unsigned char* dest = data + (y_start * data_width + x_start) * bytes_per_pixel;
    for( mi::Uint32 y = y_start; y < y_end; ++y) {
        memcpy( dest, src, bytes_per_scanline);
        src  += tile_width * bytes_per_pixel;
        dest += data_width * bytes_per_pixel;
    }
    return true;
}

} // namespace DDS

} // namespace MI
