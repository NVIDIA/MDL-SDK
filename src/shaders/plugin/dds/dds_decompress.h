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

#ifndef IO_IMAGE_DDS_DDS_DECOMPRESS_H
#define IO_IMAGE_DDS_DDS_DECOMPRESS_H

#include "dds_types.h"

#include <mi/base/types.h>

#include <cassert>
#include <vector>

namespace MI {

namespace DDS {

/// The DXT decompressor is a utility class to decode DXT-compressed images.
///
/// DXT-compression is block-oriented, each block encompasses BLOCK_PIXEL_DIM x BLOCK_PIXEL_DIM
/// pixels. Each decompress_blockline() call decompresses BLOCK_PIXEL_DIM many scanlines at once.
/// The decompressed pixel data is stored in an internal buffer and can be queried from there.
class Dxt_decompressor
{
public:

    /// Decompression mode
    enum Mode { 
        COLOR_ALPHA,  //< Decompress color and alpha.
        ALPHA_ONLY,   //< Decompress only alpha.
        COLOR_ONLY,   //< Decompress only color.
        ALPHA_AS_GREY //< Decompress alpha as grey value.
    };
    
    /// Constructor.
    Dxt_decompressor();

    /// Sets the format of the source data.
    ///
    /// \param format   The format of the compressed data.
    /// \param width    The width of the compressed data (in pixels).
    /// \param height   The height of the compressed data (in pixels).
    void  set_source_format(
        Dds_compress_fmt format,
        mi::Uint32 width,
        mi::Uint32 height);
     
    /// Sets format of the target data.
    ///
    /// This method might invalidate the address returned by #get_buffer().
    ///
    /// \param component_count   The number of components per pixel: 3 implies pixel type "Rgb",
    ///                          4 implies pixel type "
    /// \param width             The width of the uncompressed data (in pixels)
    ///
    void set_target_format(
        mi::Uint32 component_count,
        mi::Uint32 width);
    
    /// Sets the decompression mode, default is #COLOR_ALPHA.
    ///
    /// \param mode   The new decompression mode.
    void set_mode( Mode mode) { m_mode = mode; }
     
    /// Returns the decompression mode.
    Mode get_mode() const { return m_mode; }

    /// Decompresses one line of DXT blocks.
    /// \param blocks    The block data.
    /// \param block_y   The block line to decompress, range from 0 .. get_block_count_y().
    void decompress_blockline( const mi::Uint8* blocks, mi::Uint32 block_y);
     
    /// Returns the buffer of decompressed pixel data in target format.
    ///
    /// The buffer has get_block_dimension() scanlines. This value might be invalidated by
    /// later #set_target_format() calls.
    const mi::Uint8* get_buffer() const { return &m_buffer[0]; }
    
    /// Returns the buffer of decompressed pixel data in target format for one scanline.
    ///
    /// The buffer has get_block_dimension() scanlines.
    const mi::Uint8* get_scanline( mi::Uint32 scan_line) const
    {
        assert( scan_line < BLOCK_PIXEL_DIM);
        return get_buffer() + scan_line * m_target_width;
    }

    /// Returns the number of blocks in x direction.
    mi::Uint32 get_block_count_x() const { return m_blocks_x; }

    /// Returns the number of blocks in y direction.
    mi::Uint32 get_block_count_y() const { return m_blocks_y; }
    
    /// Returns the number of bytes per block for the current compression format.
    mi::Uint32 get_bytes_per_block() const
    {
        if( m_source_format == DXTC1) return 8; //-V525 PVS
        if( m_source_format == DXTC3) return 16;
        if( m_source_format == DXTC5) return 16;
        assert( false);
        return 0;
    }
    
    /// Returns the number of scanlines per block.
    mi::Uint32 get_block_dimension() const { return BLOCK_PIXEL_DIM; }
    
    /// Indicates whether color data will be decompressed (see #Mode).
    bool color_enabled() const;
    
    /// Indicates whether alpha data will be decompressed (see #Mode).
    bool alpha_enabled() const;
    
private:

    /// Blocks have a size of 4x4 pixels.
    static const mi::Uint32 BLOCK_PIXEL_DIM = 4;

    /// Block decompressor method for DXTC1
    ///
    /// \param block    The compressed DXT1 block (8 bytes), input.
    /// \param pixels   The decompressed pixel data, output.
    void decompress_dxtc1( const mi::Uint8* block, mi::Uint8* pixels);

    /// Block decompressor method for DXTC3
    ///
    /// \param block    The compressed DXT3 block (16 bytes), input.
    /// \param pixels   The decompressed pixel data, output.
    void decompress_dxtc3( const mi::Uint8* block, mi::Uint8* pixels);

    /// Block decompressor method for DXTC5
    ///
    /// \param block    The compressed DXT5 block (16 bytes), input.
    /// \param pixels   The decompressed pixel data, output.
    void decompress_dxtc5( const mi::Uint8* block, mi::Uint8* pixels);

    /// Decodes color data for DXTC3 and DXTC5
    ///
    /// \param block    The color data block, input.
    /// \param pixels   The decompressed pixel data, output.
    void decode_colors( const mi::Uint8* color_block, mi::Uint8* pixels);
    
    /// Type of the decompressor methods.
    typedef void (Dxt_decompressor::*FDecompress)(const mi::Uint8*, mi::Uint8*);

    /// Current decompressor method.
    FDecompress m_decompress_block;

    /// The compression format of the source data.
    Dds_compress_fmt m_source_format;

    /// The number of components in the target data.
    mi::Uint32 m_target_component_count;

    /// The width of the target data (in bytes).
    mi::Uint32 m_target_width;

    /// The number of blocks in x direction.
    mi::Uint32 m_blocks_x;

    /// The number of blocks in y direction.
    mi::Uint32 m_blocks_y;

    /// The decompression mode.
    Mode m_mode;
            
    /// The buffer for decompressed pixel data.
    std::vector<mi::Uint8> m_buffer;
};

} // namespace DDS

} // namespace MI

#endif // IO_IMAGE_DDS_DDS_DECOMPRESS_H
