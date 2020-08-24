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

#include "dds_image.h"
#include "dds_half_to_float.h"

#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iwriter.h>

#include <algorithm>
#include <cstring>
#include <io/image/image/i_image_utilities.h>

namespace MI {

namespace DDS {


Image::Image()
{
    clear();
}

void Image::create(
    IMAGE::Pixel_type pixel_type,
    const Texture& texture,
    bool is_cubemap,
    Dds_compress_fmt format)
{
    assert( pixel_type != IMAGE::PT_UNDEF);
    if( is_cubemap) {
        assert( texture.get_surface( 0).get_depth() == 6);
    }

    clear();

    m_texture_type = is_cubemap ? TEXTURE_CUBEMAP :
        texture.get_surface( 0).get_depth() == 1 ? TEXTURE_FLAT : TEXTURE_3D;
    m_pixel_type = pixel_type;
    m_texture = texture;
    m_compress_format = format;
    m_valid = true;
}

void Image::clear()
{
    m_compress_format = DXTC_none;
    m_pixel_type = IMAGE::PT_UNDEF;
    m_texture_type = TEXTURE_NONE;
    m_valid = false;
    m_texture.clear();
}

bool Image::load_header(
    mi::neuraylib::IReader* reader,
    Header& header,
    IMAGE::Pixel_type& pixel_type,
    Dds_compress_fmt& compress_format,
    bool for_hw)
{
    if( !reader)
        return false;

    // Check the magic string
    char buffer[4];
    mi::Sint64 bytes_read = reader->read( buffer, 4);
    if( bytes_read != 4 || strncmp( buffer, "DDS ", 4) != 0)
        return false;

    // Read the DDS header
    bytes_read = reader->read( reinterpret_cast<char*>( &header), sizeof( Header));
    if( bytes_read != sizeof( Header))
        return false;

    // Fix the DDS header

    // Apparently, there are some DDS files where this value is incorrectly set to 0 even though
    // the files contains at least one surface.
    if( header.m_mipmap_count == 0)
        header.m_mipmap_count = 1;

    // Apparently, there are some DDS files where this value is incorrectly set to 0 even though
    // the files contains at least one layer.
    if( header.m_depth == 0) {
        // Recognize cubemaps only if all six faces are present.
        mi::Uint32 cubemap_flags = DDSF_CUBEMAP | DDSF_CUBEMAP_ALL_FACES;
        header.m_depth = ((header.m_caps2 & cubemap_flags) == cubemap_flags) ? 6 : 1;
    }

    // For volume texture with more than one miplevel skip higher miplevels. The problem is that in
    // DDS the depth of volume texture is halved in each miplevel, but it is constant in neuray
    // (only width and height are halved).
    if( header.m_depth > 1 && (header.m_caps2 & DDSF_CUBEMAP) == 0 && header.m_mipmap_count > 1)
        header.m_mipmap_count = 1;

    compress_format = DXTC_none;

    // Check the pixel format
    if( header.m_ddspf.m_flags & DDSF_FOURCC) {

        // Compressed or floating point formats
        switch( header.m_ddspf.m_four_cc) {

            // Supported floating point formats
            case DDSF_R32F:
                pixel_type = IMAGE::PT_FLOAT32;
                return true;
            case DDSF_A32B32G32R32F:
                pixel_type = IMAGE::PT_COLOR; //-V1037 PVS
                return true;
            case DDSF_A16B16G16R16F:
                pixel_type = IMAGE::PT_COLOR; // Note: the half data needs to be converted
                return true;                  // to float first.

            // Unsupported floating point formats
            case DDSF_R16F:
            case DDSF_G16R16F:
            case DDSF_G32R32F:
                return false;

            // Supported compressed formats
            case FOURCC_DXT1:
                compress_format = DXTC1;
                pixel_type = IMAGE::PT_RGBA;
                return true;
            case FOURCC_DXT3:
                compress_format = DXTC3;
                pixel_type = IMAGE::PT_RGBA;
                return true;
            case FOURCC_DXT5:
                compress_format = DXTC5;
                pixel_type = IMAGE::PT_RGBA;
                return true;

            // Unsupported compressed formats
            default:
                return false;
         }
    }

    // For the test files these switch statements never match, hence the following if statements.
    // If you find a file that triggers the "true" cases in the switch statement, let me know.
    switch( header.m_ddspf.m_flags) {

        // Supported unsigned byte formats
        case DDSF_R8G8B8:
            assert( !"Support for this pixel format is experimental."); //-V547 PVS
            pixel_type = IMAGE::PT_RGB;
            return true;
        case DDSF_X8R8G8B8:
        case DDSF_A8R8G8B8:
        case DDSF_X8B8G8R8:
        case DDSF_A8B8G8R8:
            assert( !"Support for this pixel format is experimental."); //-V547 PVS
            pixel_type = IMAGE::PT_RGBA;
            return true;
        case DDSF_A8:
        case DDSF_L8:
            assert( !"Support for this pixel format is experimental."); //-V547 PVS
            pixel_type = IMAGE::PT_SINT8;
            return true;

        // Unsupported unsigned byte formats
        case DDSF_A16B16G16R16:
        case DDSF_L16:
        case DDSF_R5G6B5:
        case DDSF_X1R5G5B5:
        case DDSF_A1R5G5B5:
        case DDSF_A4R4G4B4:
        case DDSF_R3G3B2:
        case DDSF_A8R3G3B2:
        case DDSF_X4R4G4B4:
        case DDSF_A2B10G10R10:
        case DDSF_G16R16:
        case DDSF_A2R10G10B10:
        case DDSF_A8P8:
        case DDSF_P8:
        case DDSF_A8L8:
        case DDSF_A4L4:
            return false;
    }

    // Standard RGBA color formats
    if( header.m_ddspf.m_flags == DDSF_RGBA && header.m_ddspf.m_rgb_bit_count == 32) {
        pixel_type = IMAGE::PT_RGBA;
        return true;
    }

    if( header.m_ddspf.m_flags == DDSF_RGB  && header.m_ddspf.m_rgb_bit_count == 32) {
        pixel_type = IMAGE::PT_RGBA;
        return true;
    }

    if( header.m_ddspf.m_flags == DDSF_RGB  && header.m_ddspf.m_rgb_bit_count == 24) {
        pixel_type = IMAGE::PT_RGB;
        return true;
    }

    if( header.m_ddspf.m_rgb_bit_count == 8) {
        pixel_type = IMAGE::PT_SINT8;
        return true;
    }

    return false;
}

bool Image::load( mi::neuraylib::IReader* reader, bool for_hw)
{
    clear();

    Header header;
    if( !load_header( reader, header, m_pixel_type, m_compress_format, for_hw))
        return false;

    // Set the texture type
    m_texture_type = TEXTURE_FLAT;
    if( header.m_caps2 & (DDSF_CUBEMAP | DDSF_CUBEMAP_ALL_FACES))
        m_texture_type = TEXTURE_CUBEMAP;
    if( header.m_caps2 & DDSF_VOLUME)
        m_texture_type = TEXTURE_3D;

    bool halfs =    (header.m_ddspf.m_flags & DDSF_FOURCC)
                 && (header.m_ddspf.m_four_cc == DDSF_A16B16G16R16F);

    if( !is_cubemap()) {

        mi::Uint32 width  = header.m_width;
        mi::Uint32 height = header.m_height;
        mi::Uint32 depth  = header.m_depth;

        // Loop over all miplevels
        for( mi::Uint32 s = 0; s < header.m_mipmap_count; ++s) {

            mi::Uint32 size = get_layer_size( width, height) * depth;

            // Import miplevel
            if( halfs)
                size /= 2;

            std::vector<mi::Uint8> buffer( size);
            mi::Sint64 bytes_read = reader->read( reinterpret_cast<char*>( &buffer[0]), size);
            if( bytes_read != size) {
                clear();
                return false;
            }

            if( halfs)
                expand_half( buffer);

            // Create miplevel
            Surface surface( width, height, depth, buffer.size(), &buffer[0]);
            flip_surface( surface);

            m_texture.add_surface( surface);

            width  = std::max( width  >> 1, 1u);
            height = std::max( height >> 1, 1u);
            depth  = std::max( depth  >> 1, 1u);
        }

    } else {

        // Create all surfaces
        mi::Uint32 width  = header.m_width;
        mi::Uint32 height = header.m_height;
        mi::Uint32 depth  = header.m_depth;

        for( mi::Uint32 s = 0; s < header.m_mipmap_count; ++s) {

            mi::Uint32 size = get_layer_size( width, height) * depth;
            Surface surface( width, height, depth, size, 0);
            m_texture.add_surface( surface);

            width  = std::max( width  >> 1, 1u);
            height = std::max( height >> 1, 1u);
        }

        // Loop over all faces
        for( mi::Uint32 face = 0; face < 6; ++face) {

            width  = header.m_width;
            height = header.m_height;

            // Loop over all miplevels
            for( mi::Uint32 s = 0; s < header.m_mipmap_count; ++s) {

                mi::Uint32 size = get_layer_size( width, height);

                // Import miplevel of this face
                if( halfs)
                    size /= 2;

                std::vector<mi::Uint8> buffer( size);
                mi::Sint64 bytes_read = reader->read( reinterpret_cast<char*>( &buffer[0]), size);
                if( bytes_read != size) {
                    clear();
                    return false;
                }

                if( halfs) {
                    expand_half( buffer);
                    size *= 2;
                }

                // Copy miplevel of this face into corresponding miplevel layer
                Surface& surface = m_texture.get_surface( s);
                mi::Uint8* pixels = surface.get_pixels() + face * size;
                memcpy( pixels, &buffer[0], size);

                width  = std::max( width  >> 1, 1u);
                height = std::max( height >> 1, 1u);
            }

        }
    }

    reorder_rgb_or_rgba( header);

    m_valid = true;

    return true;
}

bool Image::save( mi::neuraylib::IWriter* writer)
{
    if( !m_valid)
        return false;

    assert( m_texture_type != TEXTURE_NONE);

    Header header;
    unsigned int header_size = sizeof( Header);
    memset( &header, 0, header_size);
    header.m_size = header_size;

    header.m_flags = DDSF_CAPS | DDSF_WIDTH | DDSF_HEIGHT | DDSF_PIXELFORMAT;
    header.m_height = m_texture.get_surface( 0).get_height();
    header.m_width  = m_texture.get_surface( 0).get_width();

    if( is_compressed()) {
        header.m_flags |= DDSF_LINEARSIZE;
        header.m_pitch = m_texture.get_surface( 0).get_size();
    } else {
        header.m_flags |= DDSF_PITCH;
        header.m_pitch = ((get_width() * get_components() * 8 + 31) & ~31) >> 3;
    }

    if( m_texture_type == TEXTURE_3D) {
        header.m_flags |= DDSF_DEPTH;
        header.m_depth = get_depth();
    }

    if( get_num_surfaces() > 1) {
        header.m_flags |= DDSF_MIPMAPCOUNT;
        header.m_mipmap_count = get_num_surfaces();
    }

    header.m_ddspf.m_size = sizeof( Pixel_format);

    if( is_compressed()) {
        switch( m_compress_format) {
            case DXTC1:
                header.m_ddspf.m_flags = DDSF_FOURCC;
                header.m_ddspf.m_four_cc = FOURCC_DXT1;
                break;
            case DXTC3:
                header.m_ddspf.m_flags = DDSF_FOURCC;
                header.m_ddspf.m_four_cc = FOURCC_DXT3;
                break;
            case DXTC5:
                header.m_ddspf.m_flags = DDSF_FOURCC;
                header.m_ddspf.m_four_cc = FOURCC_DXT5;
                break;
            default:
                assert( false);
        }
    } else {
        header.m_ddspf.m_flags = (get_components() == 4) ? DDSF_RGBA : DDSF_RGB;
        if( get_components() == 1)
            header.m_ddspf.m_flags = 0x20000;
        header.m_ddspf.m_rgb_bit_count = get_components() * 8;
        if( get_components() > 1) {
            header.m_ddspf.m_r_bit_mask = 0x000000ff;
            header.m_ddspf.m_g_bit_mask = 0x0000ff00;
            header.m_ddspf.m_b_bit_mask = 0x00ff0000;
        } else {
            header.m_ddspf.m_r_bit_mask = 0x000000ff;
        }
        if( get_components() == 4) {
            header.m_ddspf.m_flags |= DDSF_ALPHAPIXELS;
            header.m_ddspf.m_a_bit_mask = 0xff000000;
        }
    }

    header.m_caps1 = DDSF_TEXTURE;

    if( m_texture_type == TEXTURE_CUBEMAP) {
        header.m_caps1 |= DDSF_COMPLEX;
        header.m_caps2 = DDSF_CUBEMAP | DDSF_CUBEMAP_ALL_FACES;
    }

    if( m_texture_type == TEXTURE_3D) {
        header.m_caps1 |= DDSF_COMPLEX;
        header.m_caps2 = DDSF_VOLUME;
    }

    if( get_num_surfaces() > 1)
        header.m_caps2 |= DDSF_COMPLEX | DDSF_MIPMAP;

    // Write the magic string
    mi::Sint64 bytes_written = writer->write( "DDS ", 4);
    if( bytes_written != 4) {
        clear();
        return false;
    }

    // Write the DDS header
    bytes_written = writer->write( reinterpret_cast<const char*>( &header), sizeof( Header));
    if( bytes_written != sizeof( Header)) {
        clear();
        return false;
    }

    if( !is_cubemap()) {

        // Loop over the miplevels
        for( mi::Size s = 0; s < m_texture.get_num_surfaces(); ++s) {

            Surface& surface = m_texture.get_surface( s);

            // Prepare the miplevel for export
            flip_surface( surface);

            // Export the miplevel
            bytes_written = writer->write(
                reinterpret_cast<const char*>( surface.get_pixels()), surface.get_size());
            if( bytes_written != surface.get_size()) {
                clear();
                return false;
            }
        }

    } else {

        // Loop over the faces
        for( mi::Size face = 0; face < 6; ++face) {

            mi::Uint32 width  = header.m_width;
            mi::Uint32 height = header.m_height;

            // Loop over the miplevels
            for( mi::Size s = 0; s < m_texture.get_num_surfaces(); ++s) {

                Surface& surface = m_texture.get_surface( s);
                mi::Uint32 size = get_layer_size( width, height);
                mi::Uint8* pixels = surface.get_pixels() + face * size;

                // Export the miplevel of this face
                bytes_written = writer->write( reinterpret_cast<const char*>( pixels), size);
                if( bytes_written != size) {
                    clear();
                    return false;
                }

                width  = std::max( width  >> 1, 1u);
                height = std::max( height >> 1, 1u);
            }
        }

    }

    clear();

    return true;
}

mi::Uint32 Image::get_layer_size( mi::Uint32 width, mi::Uint32 height)
{
    return is_compressed()
        ? ((width+3)/4) * ((height+3)/4) * (m_compress_format == DXTC1 ? 8 : 16)
            : width * height * IMAGE::get_bytes_per_pixel( m_pixel_type);
}

void Image::flip_surface( Surface& surface)
{
    if( !is_compressed()) {

        mi::Uint32 layer_size = surface.get_size() / surface.get_depth();
        mi::Uint32 row_size   = layer_size/surface.get_height();

        for( mi::Uint32 z = 0; z < surface.get_depth(); ++z) {

            mi::Uint8* top    = surface.get_pixels() + z * layer_size;
            mi::Uint8* bottom = top + (layer_size-row_size);

            for( mi::Uint32 y = 0; y < surface.get_height() / 2; ++y) {
                swap( bottom, top, row_size);
                top    += row_size;
                bottom -= row_size;
            }
        }

    } else {

        mi::Uint32 block_size;
        void (*flip_blocks)(DXT_color_block*, mi::Uint32);

        switch( m_compress_format) {
            case DXTC1:
                block_size = 8; //-V525 PVS
                flip_blocks = &Image::flip_blocks_dxtc1;
                break;
            case DXTC3:
                block_size = 16;
                flip_blocks = &Image::flip_blocks_dxtc3;
                break;
            case DXTC5:
                block_size = 16;
                flip_blocks = &Image::flip_blocks_dxtc5;
                break;
            default:
                return;
        }

        mi::Uint32 blocks_x   = surface.get_width()  / 4;
        mi::Uint32 blocks_y   = surface.get_height() / 4;
        mi::Uint32 row_size   = blocks_x * block_size;
        mi::Uint32 layer_size = row_size * blocks_y;

        for( mi::Uint32 z = 0; z < surface.get_depth(); ++z) {

            for( mi::Uint32 y = 0; y < blocks_y / 2; ++y) {

                DXT_color_block* top = reinterpret_cast<DXT_color_block*>(
                    surface.get_pixels() + z * layer_size + y * row_size);
                DXT_color_block* bottom = reinterpret_cast<DXT_color_block*>(
                    surface.get_pixels() + z * layer_size + (blocks_y-y-1) * row_size);
                flip_blocks( top, blocks_x);
                flip_blocks( bottom, blocks_x);
                swap( bottom, top, row_size);
            }

            if( blocks_y % 2 == 1) {

                DXT_color_block* middle = reinterpret_cast<DXT_color_block*>(
                    surface.get_pixels() + z * layer_size + blocks_y/2 * row_size);
                flip_blocks( middle, blocks_x);

            }
        }
    }
}

void Image::flip_blocks_dxtc1( DXT_color_block* line, mi::Uint32 num_blocks)
{
    DXT_color_block* block = line;

    for( mi::Uint32 i = 0; i < num_blocks; ++i) {
        std::swap( block->m_row[0], block->m_row[3]);
        std::swap( block->m_row[1], block->m_row[2]);
        block++;
    }
}

void Image::flip_blocks_dxtc3( DXT_color_block* line, mi::Uint32 num_blocks)
{
    DXT_color_block* block = line;

    for( mi::Uint32 i = 0; i < num_blocks; ++i) {

        DXT3_alpha_block* alpha_block = reinterpret_cast<DXT3_alpha_block*>( block);
        std::swap( alpha_block->m_row[0], alpha_block->m_row[3]);
        std::swap( alpha_block->m_row[1], alpha_block->m_row[2]);
        block++;

        std::swap( block->m_row[0], block->m_row[3]);
        std::swap( block->m_row[1], block->m_row[2]);
        block++;
    }
}

void Image::flip_blocks_dxtc5( DXT_color_block* line, mi::Uint32 num_blocks)
{
    DXT_color_block* block = line;
    for( mi::Uint32 i = 0; i < num_blocks; ++i) {

        flip_dxt5_alpha( reinterpret_cast<DXT5_alpha_block*>( block));
        block++;

        std::swap( block->m_row[0], block->m_row[3]);
        std::swap( block->m_row[1], block->m_row[2]);
        block++;
    }
}

void Image::flip_dxt5_alpha( DXT5_alpha_block* block)
{
    // Read next 48 bits (3 bits per pixel) of block into a mi::Uint64 for easier manipulation.
    mi::Uint64 bits = 0;
    for( int i = 5; i >= 0; --i) {
        bits <<= 8;
        bits |= block->m_row[i];
    }

    // Swap the rows.
    mi::Uint64 flipped_bits = 0;
    flipped_bits |= (bits >> 36) & 0x0fffull;
    flipped_bits |= (bits >> 12) & 0x0fff000ull;
    flipped_bits |= (bits << 12) & 0x0fff000000ull;
    flipped_bits |= (bits << 36) & 0x0fff000000000ull;

    // Write flipped bits back.
    for( int i = 0; i < 6; ++i) {
        block->m_row[i] = static_cast<mi::Uint8>( flipped_bits & 0xff);
        flipped_bits >>= 8;
    }
}

void Image::reorder_rgb_or_rgba( Header& header)
{
    if( is_compressed())
        return;

    if( header.m_ddspf.m_rgb_bit_count == 0)
        return;

    // For floating point pixel types, no reordering is necessary.
    if( IMAGE::get_bytes_per_component( m_pixel_type) > 1)
        return;

    mi::Uint32 bytes_per_pixel      = IMAGE::get_bytes_per_pixel( m_pixel_type);
    mi::Uint32 components_per_pixel = IMAGE::get_components_per_pixel( m_pixel_type);

    for( mi::Uint32 s = 0; s < header.m_mipmap_count; ++s) {

        Surface& surface = m_texture.get_surface( s);
        mi::Uint32 layer_size  = surface.get_size() / surface.get_depth();
        mi::Uint32 row_size    = layer_size / surface.get_height();

        for( mi::Uint32 z = 0; z < surface.get_depth(); ++z) {

            mi::Uint32 offset = z * layer_size;
            mi::Uint8* top    = surface.get_pixels() + offset;

            for( mi::Uint32 y = 0; y < surface.get_height(); ++y) {
                for( mi::Uint32 x = 0; x < surface.get_width(); ++x) {

                    // Get color pixel
                    mi::Uint32 color = 0;
                    assert( bytes_per_pixel <= sizeof( mi::Uint32));
                    memcpy( &color, &top[x * bytes_per_pixel], bytes_per_pixel); //-V512 PVS

                    for( mi::Uint32 c = 0; c < components_per_pixel; ++c) {

                        // Get mask for current component
                        mi::Uint32 mask = 0;
                        switch( c) {
                            case 0: mask = header.m_ddspf.m_r_bit_mask; break;
                            case 1: mask = header.m_ddspf.m_g_bit_mask; break;
                            case 2: mask = header.m_ddspf.m_b_bit_mask; break;
                            case 3: mask = header.m_ddspf.m_a_bit_mask; break;
                        }

                        // Fake alpha for X8... formats
                        if( c == 3 && mask == 0) {
                            top[x * bytes_per_pixel + c] = 255;
                            continue;
                        }

                        // Get shift for current component
                        mi::Uint32 rshift = 0;
                        switch( mask) {
                            case 0xff000000: rshift = 24; break;
                            case 0x00ff0000: rshift = 16; break;
                            case 0x0000ff00: rshift =  8; break;
                            case 0x000000ff: rshift =  0; break;
                        }

                        // Extract component and write to buffer
                        top[x * bytes_per_pixel + c] = (color & mask) >> rshift;
                    }
                }

                // Next line
                top += row_size;
            }
        }
    }

    header.m_ddspf.m_r_bit_mask = 0x000000ff;
    header.m_ddspf.m_g_bit_mask = 0x0000ff00;
    header.m_ddspf.m_b_bit_mask = 0x00ff0000;
    if( get_components() == 4)
        header.m_ddspf.m_a_bit_mask = 0xff000000;
}

void Image::swap( void* addr1, void* addr2, mi::Uint32 size)
{
    mi::Uint8* tmp = new mi::Uint8[size];
    memcpy( tmp, addr1, size);
    memcpy( addr1, addr2, size);
    memcpy( addr2, tmp, size);
    delete [] tmp;
}

void Image::expand_half( std::vector<mi::Uint8>& buffer)
{
    mi::Size n = buffer.size() / 2;
    buffer.resize( buffer.size() * 2);
    const unsigned short* hp = reinterpret_cast<const unsigned short*>( &buffer[0]);
    float* fp = reinterpret_cast<float*>( &buffer[0]);
    for( mi::Size i = 0; i < n; ++i)
        fp[n-1-i] = half_to_float( hp[n-1-i] );
}

} // namespace DDS

} // namespace MI
