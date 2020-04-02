/***************************************************************************************************
 * Copyright (c) 2005-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dds_decompress.h"

#include <cstring>

namespace MI {

namespace DDS {

Dxt_decompressor::Dxt_decompressor()
  : m_decompress_block( 0),
    m_source_format( DXTC_none),
    m_target_component_count( 3),
    m_target_width( 0),
    m_blocks_x( 0),
    m_blocks_y( 0),
    m_mode( COLOR_ALPHA)
{
}

void Dxt_decompressor::set_source_format(
    Dds_compress_fmt format,
    mi::Uint32 width,
    mi::Uint32 height)
{
    m_source_format = format;
    m_blocks_x = width  / BLOCK_PIXEL_DIM;
    m_blocks_y = height / BLOCK_PIXEL_DIM;

    switch( m_source_format) {
        case DXTC1:
            m_decompress_block = &Dxt_decompressor::decompress_dxtc1; break;
        case DXTC3:
            m_decompress_block = &Dxt_decompressor::decompress_dxtc3; break;
        case DXTC5:
            m_decompress_block = &Dxt_decompressor::decompress_dxtc5; break;
        default:
            m_decompress_block = 0;
    }
}

void Dxt_decompressor::set_target_format(
    mi::Uint32 component_count,
    mi::Uint32 width)
{
    assert( component_count == 3 || component_count == 4);

    m_target_component_count = component_count;
    m_target_width = component_count * width;
    m_buffer.resize( BLOCK_PIXEL_DIM * m_target_width);
}

bool Dxt_decompressor::color_enabled() const
{
    return m_mode == COLOR_ALPHA || m_mode == COLOR_ONLY;
}

bool Dxt_decompressor::alpha_enabled() const
{
    return m_mode == COLOR_ALPHA || m_mode == ALPHA_ONLY || m_mode == ALPHA_AS_GREY;
}

void Dxt_decompressor::decompress_blockline( const mi::Uint8* blocks, mi::Uint32 block_y)
{
    const mi::Uint8* src = blocks + block_y * m_blocks_x * get_bytes_per_block();
    mi::Uint8* dest = &m_buffer[0];

    for( mi::Uint32 x = 0; x < m_blocks_x; ++x) {
        (this->*m_decompress_block)( src, dest);
        src += get_bytes_per_block();
        dest += BLOCK_PIXEL_DIM * m_target_component_count;
    }
}

/// Converts 16 bit BGR color (565) to 24 bit RGB color (888)
static void bgr565_to_rgb888( const mi::Uint8* c_in, mi::Uint8* c_out)
{
    // red
    c_out[0] =  (c_in[1] & 0xf8 /*11111000*/);
    c_out[0] |= c_out[0] >> 5;
    // green 
    c_out[1] = ((c_in[1] & 0x07 /*00000111*/) << 5) | ((c_in[0] >> 3) & 0x1c /*00011100*/);
    c_out[1] |= c_out[1] >> 6;
    // blue
    c_out[2] =  (c_in[0] & 0x1f /*00011111*/) << 3;
    c_out[2] |= c_out[2] >> 5;
}

/// Decodes color data for DXTC3 and DXTC5
///
/// The color sub-blocks of DXTC3 and DXTC5 are the same,
/// they are a simpler version of the DXT1 format.
void Dxt_decompressor::decode_colors( const mi::Uint8* color_block, mi::Uint8* pixels)
{
    // First 32bit of color_block represent the color table.
    mi::Uint8 color[4][3];
    bgr565_to_rgb888( color_block, color[0]);
    bgr565_to_rgb888( color_block + 2, color[1]);
    for( mi::Uint32 c = 0; c < 3; ++c) {
        color[2][c] = (2 * color[0][c] + color[1][c] + 1) / 3;
        color[3][c] = (color[0][c] + 2 * color[1][c] + 1) / 3;
    }

    // Decode 2-bit color table indices
    for( mi::Uint32 y = 0; y < BLOCK_PIXEL_DIM; ++y) {
        mi::Uint8 t = color_block[y + 4];
        for( mi::Uint32 x = 0; x < BLOCK_PIXEL_DIM; ++x) {
            mi::Uint32 index = (t >> (x * 2)) & 0x03;
            memcpy( pixels + x * m_target_component_count, color[index], 3);
        }
        pixels += m_target_width;
    }
}

/// Block decompressor method for DXTC1
///
/// This is an expanded version of decode_colors() since DXTC1 supports a 1 bit alpha additionally.
void Dxt_decompressor::decompress_dxtc1( const mi::Uint8* block, mi::Uint8* pixels)
{
    assert( block);
    assert( pixels);
    
    // First 32bit of block represent the color table.
    mi::Uint8 color[4][4];
    memset( color, 0xff, sizeof( color));
    
    // Decode colors only if requested.
    if( color_enabled()) {

        bgr565_to_rgb888( block, color[0]);
        bgr565_to_rgb888( block + 2, color[1]);
        // Interpret block colors as 16 bit integer for comparison
        mi::Uint16 c0 = block[0] + (block[1] << 8);
        mi::Uint16 c1 = block[2] + (block[3] << 8);
        if( c0 > c1) {
            for( mi::Uint32 c = 0; c < 3; ++c) {
                color[2][c] = (2 * color[0][c] + color[1][c] + 1) / 3;
                color[3][c] = (color[0][c] + 2 * color[1][c] + 1) / 3;
            }
        } else {
            for( mi::Uint32 c = 0; c < 3; ++c) {
                color[2][c] = (color[0][c] + color[1][c]) / 2;
                color[3][c] = 0;  // black
            }
            // Set alpha component of special color.
            color[3][3] = (m_mode == COLOR_ALPHA) ? 0 : 0xff;
        }
        
    } else if( alpha_enabled()) {

        // Interpret block colors as 16 bit integer for comparison
        mi::Uint16 c0 = block[0] + (block[1] << 8);
        mi::Uint16 c1 = block[2] + (block[3] << 8);
        // If only alpha is decoded, then all colors are white, except the special color.
        if( c0 <= c1) {
            color[3][0] = color[3][1] = color[3][2] = 0; // black
            color[3][3] = (m_mode == ALPHA_AS_GREY) ? 0xff : 0;
        }
    }
    
    // Decode 2-bit color table indices
    for( mi::Uint32 y = 0; y < BLOCK_PIXEL_DIM; ++y) {
        mi::Uint8 t = block[y + 4];
        for( mi::Uint32 x = 0; x < BLOCK_PIXEL_DIM; ++x) {
            mi::Uint32 index = (t >> (x * 2)) & 0x03;
            memcpy( pixels + x * m_target_component_count, color[index], m_target_component_count);
        }
        pixels += m_target_width;
    }
}

/// Block decompressor method for DXTC3
///
/// A DXTC3 block consists of an alpha sub-block and a color sub-block.
/// The alpha sub-block has direct 4-bit alpha data.
void Dxt_decompressor::decompress_dxtc3( const mi::Uint8* block, mi::Uint8* pixels)
{
    assert( block);
    assert( pixels);
    
    if( color_enabled())
        decode_colors( block + 8, pixels);

    if(( m_target_component_count == 4 && alpha_enabled()) || m_mode == ALPHA_AS_GREY) {
        for( mi::Uint32 y = 0; y < BLOCK_PIXEL_DIM; ++y) {
            for( mi::Uint32 x = 0; x < BLOCK_PIXEL_DIM; ++x) {

                mi::Uint32 index = y * BLOCK_PIXEL_DIM + x;
                
                mi::Uint8 alpha = block[index >> 1];
                alpha = (index % 2 == 0) ? (alpha & 0x0f) : (alpha >> 4);
                alpha = 17 * alpha; // spread from 0..15 to 0..255

                if( m_mode == ALPHA_AS_GREY)
                    memset( pixels + x * m_target_component_count, alpha, 3);
                else {
                    assert( m_target_component_count == 4);
                    pixels[x * m_target_component_count + 3] = alpha;
                }
            }
            pixels += m_target_width;
        }
    }
}

/// Block decompressor method for DXTC5
///
/// A DXTC5 block consists of an alpha sub-block and a color sub-block.
/// The alpha sub-block has indirect 3-bit alpha data and 2 reference alpha values.
void Dxt_decompressor::decompress_dxtc5( const mi::Uint8* block, mi::Uint8* pixels)
{
    assert( block);
    assert( pixels);
    
    if(( m_target_component_count == 4 && alpha_enabled()) || m_mode == ALPHA_AS_GREY) {

        // First 16 bit of block represent the alpha table.
        mi::Uint8 alpha[8];
        alpha[0] = block[0];
        alpha[1] = block[1];

        // Interpolate other alpha values in table
        if( alpha[0] > alpha[1]) {
            // 8-alpha block
            alpha[2] = (6 * alpha[0] + 1 * alpha[1] + 3) / 7;    // bit code 010
            alpha[3] = (5 * alpha[0] + 2 * alpha[1] + 3) / 7;    // bit code 011
            alpha[4] = (4 * alpha[0] + 3 * alpha[1] + 3) / 7;    // bit code 100
            alpha[5] = (3 * alpha[0] + 4 * alpha[1] + 3) / 7;    // bit code 101
            alpha[6] = (2 * alpha[0] + 5 * alpha[1] + 3) / 7;    // bit code 110
            alpha[7] = (1 * alpha[0] + 6 * alpha[1] + 3) / 7;    // bit code 111
        }
        else {
            // 6-alpha block
            alpha[2] = (4 * alpha[0] + 1 * alpha[1] + 2) / 5;    // Bit code 010
            alpha[3] = (3 * alpha[0] + 2 * alpha[1] + 2) / 5;    // Bit code 011
            alpha[4] = (2 * alpha[0] + 3 * alpha[1] + 2) / 5;    // Bit code 100
            alpha[5] = (1 * alpha[0] + 4 * alpha[1] + 2) / 5;    // Bit code 101
            alpha[6] = 0;                                        // Bit code 110
            alpha[7] = 255;                                      // Bit code 111
        }

        // Read next 48 bits (3 bits per pixel) of block into a mi::Uint64 for easier extraction.
        mi::Uint64 alpha_bits = 0;
        for( int i = 5; i >= 0; --i) {
            alpha_bits <<= 8;
            alpha_bits |= block[i+2];
        }

        mi::Uint8* pixels2 = pixels;
        for( mi::Uint32 y = 0; y < BLOCK_PIXEL_DIM; ++y) {
            for( mi::Uint32 x = 0; x < BLOCK_PIXEL_DIM; ++x) {

                mi::Uint32 index = y * BLOCK_PIXEL_DIM + x;
                mi::Uint8 index_alpha = static_cast<mi::Uint8>( alpha_bits >> (index * 3)) & 0x07;
                mi::Uint8 value = alpha[index_alpha];

                if( m_mode == ALPHA_AS_GREY)
                    memset( pixels2 + x * m_target_component_count, value, 3);
                else {
                    assert( m_target_component_count == 4);
                    pixels2[x * m_target_component_count + 3] = value;
                }
            }
            pixels2 += m_target_width;
        }
    }    

    if( color_enabled())
        decode_colors( block + 8, pixels);
}

} // namespace DDS

} // namespace MI
