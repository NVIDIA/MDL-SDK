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

#include "freeimage_utilities.h"

#if defined(MI_COMPILER_GCC)
#include <x86intrin.h>
#endif

#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iwriter.h>

#include <cassert>
#include <cstring>

namespace MI {

namespace FREEIMAGE {

mi::base::Handle<mi::base::ILogger> g_logger;

void log( mi::base::Message_severity severity, const char* message)
{
    if( !g_logger.is_valid_interface())
        return;

    g_logger->message( severity, "FREIMG:IMAGE", message);
}

unsigned DLL_CALLCONV read_handler( void* buffer, unsigned size, unsigned count, fi_handle handle)
{
    if( !handle)
        return 0;

    mi::neuraylib::IReader* reader = static_cast<mi::neuraylib::IReader*>( handle);
    return static_cast<unsigned>( reader->read( static_cast<char*>( buffer), size*count) / size);
}

unsigned DLL_CALLCONV write_handler( void* buffer, unsigned size, unsigned count, fi_handle handle)
{
    if( !handle)
        return 0;

    mi::neuraylib::IWriter* writer = static_cast<mi::neuraylib::IWriter*>( handle);
    return static_cast<unsigned>(
        writer->write( static_cast<const char*>( buffer), size*count) / size);
}

int DLL_CALLCONV seek_handler( fi_handle handle, long offset, int origin)
{
    if( !handle)
        return -1;

    mi::neuraylib::IReader_writer_base* reader_writer_base
        = static_cast<mi::neuraylib::IReader_writer_base*>( handle);
    assert( reader_writer_base->supports_absolute_access());

    mi::Sint64 position;
    if( origin == SEEK_SET)
        position = offset;
    else if( origin == SEEK_CUR)
        position = reader_writer_base->tell_absolute() + offset;
    else if( origin == SEEK_END) {
        reader_writer_base->seek_end();
        position = reader_writer_base->tell_absolute() + offset;
    } else
        return -1;

    return reader_writer_base->seek_absolute( position) ? 0 : -1;
}

long DLL_CALLCONV tell_handler( fi_handle handle)
{
    if( !handle)
        return 0;

    mi::neuraylib::IReader_writer_base* reader_writer_base
        = static_cast<mi::neuraylib::IReader_writer_base*>( handle);
    assert( reader_writer_base->supports_absolute_access());

    return static_cast<long>( reader_writer_base->tell_absolute());
}

FreeImageIO construct_io_for_reading()
{
    FreeImageIO result;
    result.read_proc = read_handler;
    result.write_proc = 0;
    result.seek_proc = seek_handler;
    result.tell_proc = tell_handler;
    return result;
}

FreeImageIO construct_io_for_writing()
{
    FreeImageIO result;
    result.read_proc = 0;
    result.write_proc = write_handler;
    result.seek_proc = seek_handler;
    result.tell_proc = tell_handler;
    return result;
}

const char* convert_freeimage_pixel_type_to_neuray_pixel_type( FREE_IMAGE_TYPE type)
{
    switch( type) {
        case FIT_RGB16:  return "Rgb_16";
        case FIT_RGBA16: return "Rgba_16";
        case FIT_RGBF:   return "Rgb_fp";
        case FIT_RGBAF:  return "Color";
        case FIT_FLOAT:  return "Float32";
        case FIT_INT32:  return "Sint32";
        default:         return 0;
    }
}

const char* convert_freeimage_pixel_type_to_neuray_pixel_type( FIBITMAP* bitmap, bool& convert)
{
    convert = false;

    FREE_IMAGE_TYPE type = FreeImage_GetImageType( bitmap);

    // Handle formats with more than 8 bits per pixel.
    if( type != FIT_BITMAP) {

        const char* neuray_pixel_type = convert_freeimage_pixel_type_to_neuray_pixel_type( type);
        if( neuray_pixel_type)
            return neuray_pixel_type;

        // Convert other formats with 8 bits per pixel to "Rgb_fp".
        convert = true;
        return "Rgb_fp";
    }

    // Handle directly supported formats with at most 8 bits per pixel.
    unsigned int bpp = FreeImage_GetBPP( bitmap);
    if( bpp == 32)
       return "Rgba";
    else if( bpp == 24)
       return "Rgb";

    assert( bpp == 1 || bpp == 4 || bpp == 8 || bpp == 16);

    // Convert other formats with at most 8 bits per pixel to "Rgba" or "Rgb".
    convert = true;
    return FreeImage_IsTransparent( bitmap) ? "Rgba" : "Rgb";
}

/// Returns the number of bytes per pixel for a neuray pixel type.
mi::Uint32 get_bytes_per_pixel( const char* pixel_type)
{
    if( strcmp( pixel_type, "Rgb"    ) == 0) return  3;
    if( strcmp( pixel_type, "Rgba"   ) == 0) return  4;
    if( strcmp( pixel_type, "Sint32" ) == 0) return  4;
    if( strcmp( pixel_type, "Float32") == 0) return  4;
    if( strcmp( pixel_type, "Rgb_16" ) == 0) return  6;
    if( strcmp( pixel_type, "Rgba_16") == 0) return  8;
    if( strcmp( pixel_type, "Rgb_fp" ) == 0) return 12;
    if( strcmp( pixel_type, "Color"  ) == 0) return 16;
    return 0;
}

static char* get_scanline( FIBITMAP* bitmap, mi::Uint32 y)
{
    return static_cast<char*>( static_cast<void*>( FreeImage_GetScanLine( bitmap, y)));
}

// Rotate p_val32 by p_nBits bits to the left
#ifndef ROL32
#if defined(_MSC_VER) || defined(MI_COMPILER_ICC) || (defined(MI_COMPILER_GCC) && !defined(MI_COMPILER_CLANG))
#define ROL32(p_val32,p_nBits) _rotl(p_val32,p_nBits)
#else
#define ROL32(p_val32,p_nBits) (((p_val32)<<(p_nBits))|((p_val32)>>(32-(p_nBits))))
#endif
#endif

bool copy_from_bitmap_to_tile(
    FIBITMAP* bitmap, mi::Uint32 x_start, mi::Uint32 y_start, mi::neuraylib::ITile* tile)
{
    // Compute the rectangular region that is to be copied
    const mi::Uint32 tile_width  = tile->get_resolution_x();
    const mi::Uint32 tile_height = tile->get_resolution_y();
    const mi::Uint32 bitmap_width  = FreeImage_GetWidth( bitmap);
    const mi::Uint32 bitmap_height = FreeImage_GetHeight( bitmap);
    const mi::Uint32 x_end = std::min( x_start + tile_width,  bitmap_width);
    const mi::Uint32 y_end = std::min( y_start + tile_height, bitmap_height);

    const char* const pixel_type = tile->get_type();

#if(FI_RGBA_RED == 2) && (FI_RGBA_GREEN == 1) && (FI_RGBA_BLUE == 0) && (FI_RGBA_ALPHA == 3)
    if( strcmp( pixel_type, "Rgb" ) == 0 || strcmp( pixel_type, "Rgba" ) == 0) {

#if 0//defined(_WIN32) && (defined(HAS_SSE) || defined(SSE_INTRINSICS)) //!! meh, otherwise needs special GCC compile flags //!! also, testing with VC2015, there was no real benefit using SSSE3, so most likely memory limited
        static int ssse3_supported = -1;
        if(ssse3_supported == -1)
        {
            int cpuInfo[4];
            __cpuid(cpuInfo,1);
            ssse3_supported = (cpuInfo[2] & (1 << 9));
        }
#endif
        // Copy pixel data pixel by pixel due to the different RGB component order.
        const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
        assert( bytes_per_pixel == 3 || bytes_per_pixel == 4);
        const char* __restrict src_start = static_cast<char*>(static_cast<void*>(FreeImage_GetBits(bitmap)));
        const size_t pitch = FreeImage_GetPitch(bitmap);

        if (bytes_per_pixel == 4)
        {
            for (mi::Uint32 y = y_start; y < y_end; ++y) {
                mi::Uint32* const __restrict dest = static_cast<mi::Uint32*>(tile->get_data()) + y*tile_width + x_start;
                const mi::Uint32* const __restrict src = reinterpret_cast<const mi::Uint32*>(src_start + y*pitch + x_start*4);
                mi::Uint32 x = 0;
                const mi::Uint32 xe = x_end-x_start;
#if 0//defined(_WIN32) && (defined(HAS_SSE) || defined(SSE_INTRINSICS)) // actually uses SSSE3 //!! meh, otherwise needs special GCC compile flags
                if(ssse3_supported)
                {
                    const __m128i mask = _mm_setr_epi8(2,1,0,3, 6,5,4,7, 10,9,8,11, 14,13,12,15);
                    for(; x < (xe & 0xFFFFFFFCu); x+=4)
                        _mm_storeu_si128((__m128i*)(dest+x), _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src+x)), mask)); //!! aligned loads/store?
                }
#endif
                for (; x < xe; ++x) {
                    const mi::Uint32 tmp0 = src[x];
                    const mi::Uint32 tmp00 = (tmp0 & 0xFF00FFu);
                    dest[x] = (tmp0 & 0xFF00FF00u) | ROL32(tmp00, 16); // swap R & B
                }
            }
        }
        else
        {
            char* __restrict dest = static_cast<char*>(tile->get_data());
            for (mi::Uint32 y = y_start; y < y_end; ++y) {
                const char* __restrict src = src_start + y*pitch + x_start * 3;
                mi::Uint32 x = 0;
                const mi::Uint32 xe = x_end - x_start;
#if 0//defined(_WIN32) && (defined(HAS_SSE) || defined(SSE_INTRINSICS)) // actually uses SSSE3 //!! meh, otherwise needs special GCC compile flags
                if(ssse3_supported)
                {
                    const __m128i mask0 = _mm_setr_epi8(2,1,0, 5,4,3, 8,7,6, 11,10,9, 14,13,12, 15); //!! last entry
                    const __m128i mask1 = _mm_setr_epi8(0,1, 4,3,2, 7,6,5, 10,9,8, 13,12,11, 14,15); //!! first & last entries
                    const __m128i mask2 = _mm_setr_epi8(0, 3,2,1, 6,5,4, 9,8,7, 12,11,10, 15,14,13); //!! first entry
                    for(; x < (xe & 0xFFFFFFF0u); x+=16,src+=48,dest+=48) {
                        const __m128i c[3] = {_mm_loadu_si128((__m128i*)src),_mm_loadu_si128((__m128i*)(src+16)),_mm_loadu_si128((__m128i*)(src+32))}; //!! aligned loads/store?
                        _mm_storeu_si128((__m128i*)(dest   ), _mm_shuffle_epi8(c[0], mask0));
                        _mm_storeu_si128((__m128i*)(dest+16), _mm_shuffle_epi8(c[1], mask1));
                        _mm_storeu_si128((__m128i*)(dest+32), _mm_shuffle_epi8(c[2], mask2));
                        dest[15] = src[17]; //!! fix up the 4 wrong locations, meh!
                        dest[17] = src[15];
                        dest[30] = src[32];
                        dest[32] = src[30];
                    }
                } else
#endif
                for (; x < (xe & 0xFFFFFFFCu); x+=4,src+=12,dest+=12) {
                    const mi::Uint32* __restrict const src4 = reinterpret_cast<const mi::Uint32*>(src);
                    const mi::Uint32 tmp0 = src4[0]; // BGRB
                    const mi::Uint32 tmp1 = src4[1]; // GRBG
                    const mi::Uint32 tmp2 = src4[2]; // RBGR
                    mi::Uint32* __restrict const dest4 = reinterpret_cast<mi::Uint32*>(dest);
                    const mi::Uint32 tmp00 = (tmp0 & 0xFF00FFu) | (tmp1 & 0xFF00u);
                    dest4[0] = (tmp0 & 0xFF00u) | ROL32(tmp00, 16);
                    const mi::Uint32 tmp11 = (tmp0 & 0xFF000000u) | (tmp2 & 0xFFu);
                    dest4[1] = (tmp1 & 0xFF0000FFu) | ROL32(tmp11, 16);
                    const mi::Uint32 tmp22 = (tmp2 & 0xFF00FF00u) | (tmp1 & 0xFF0000u);
                    dest4[2] = (tmp2 & 0xFF0000u) | ROL32(tmp22, 16);
                }
                // finish up the leftovers
                for (; x < xe; ++x,src+=3,dest+=3) {
                    dest[0] = src[FI_RGBA_RED];
                    dest[1] = src[FI_RGBA_GREEN];
                    dest[2] = src[FI_RGBA_BLUE];
                }
                dest += (x_start + tile_width - x_end) * 3;
            }
        }

        return true;

    } else if(
#else // little endian: just memcpy these types
    if(        strcmp( pixel_type, "Rgb"    ) == 0
        ||     strcmp( pixel_type, "Rgba"   ) == 0 ||
#endif
               strcmp( pixel_type, "Sint32" ) == 0
        ||     strcmp( pixel_type, "Float32") == 0
        ||     strcmp( pixel_type, "Rgb_16" ) == 0
        ||     strcmp( pixel_type, "Rgba_16") == 0
        ||     strcmp( pixel_type, "Rgb_fp" ) == 0
        ||     strcmp( pixel_type, "Color"  ) == 0) {

        // Copy pixel data using memcpy() for each scanline.
        const mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
        assert( bytes_per_pixel > 0);
        const mi::Uint32 bytes_per_scanline = (x_end-x_start) * bytes_per_pixel;
        char* __restrict dest = static_cast<char*>( tile->get_data());
        const char* __restrict src_start = static_cast<char*>(static_cast<void*>(FreeImage_GetBits(bitmap)));
        const size_t pitch = FreeImage_GetPitch(bitmap);
        for( mi::Uint32 y = y_start; y < y_end; ++y) {
            const char* __restrict src = src_start + y*pitch + x_start * bytes_per_pixel;
            memcpy( dest, src, bytes_per_scanline);
            dest += tile_width * bytes_per_pixel;
        }
        return true;

    } else
        return false;
}

bool copy_from_tile_to_bitmap(
    const mi::neuraylib::ITile* tile, FIBITMAP* bitmap, mi::Uint32 x_start, mi::Uint32 y_start)
{
    // Compute the rectangular region that is to be copied
    mi::Uint32 tile_width  = tile->get_resolution_x();
    mi::Uint32 tile_height = tile->get_resolution_y();
    mi::Uint32 bitmap_width  = FreeImage_GetWidth( bitmap);
    mi::Uint32 bitmap_height = FreeImage_GetHeight( bitmap);
    mi::Uint32 x_end = std::min( x_start + tile_width,  bitmap_width);
    mi::Uint32 y_end = std::min( y_start + tile_height, bitmap_height);

    const char* pixel_type = tile->get_type();

    if( strcmp( pixel_type, "Rgb" ) == 0 || strcmp( pixel_type, "Rgba" ) == 0) {

        // Copy pixel data pixel by pixel due to the different RGB component order.
        mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
        assert( bytes_per_pixel == 3 || bytes_per_pixel == 4);
        const char* src = static_cast<const char*>( tile->get_data());
        for( mi::Uint32 y = y_start; y < y_end; ++y) {
            char* dest = get_scanline( bitmap, y) + x_start * bytes_per_pixel;
            for( mi::Uint32 x = x_start; x < x_end; ++x) {
                dest[FI_RGBA_RED]   = src[0];
                dest[FI_RGBA_GREEN] = src[1];
                dest[FI_RGBA_BLUE]  = src[2];
                if( bytes_per_pixel == 4)
                    dest[FI_RGBA_ALPHA] = src[3];
                src  += bytes_per_pixel;
                dest += bytes_per_pixel;
            }
            src += (x_start + tile_width - x_end) * bytes_per_pixel;
        }
        return true;

    } else if( strcmp( pixel_type, "Sint32" ) == 0
        ||     strcmp( pixel_type, "Float32") == 0
        ||     strcmp( pixel_type, "Rgb_16" ) == 0
        ||     strcmp( pixel_type, "Rgba_16") == 0
        ||     strcmp( pixel_type, "Rgb_fp" ) == 0
        ||     strcmp( pixel_type, "Color"  ) == 0) {

        // Copy pixel data using memcpy() for each scanline.
        mi::Uint32 bytes_per_pixel = get_bytes_per_pixel( pixel_type);
        assert( bytes_per_pixel > 0);
        mi::Uint32 bytes_per_scanline = (x_end-x_start) * bytes_per_pixel;
        const char* src = static_cast<const char*>( tile->get_data());
        for( mi::Uint32 y = y_start; y < y_end; ++y) {
            char* dest = get_scanline( bitmap, y) + x_start * bytes_per_pixel;
            memcpy( dest, src, bytes_per_scanline);
            src += tile_width * bytes_per_pixel;
        }
        return true;

    } else
        return false;
}

} // namespace FREEIMAGE

} // namespace MI
