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

#ifndef IO_IMAGE_DDS_DDS_TYPES_H
#define IO_IMAGE_DDS_DDS_TYPES_H

#include <mi/base/types.h>

namespace MI {

namespace DDS {

// surface description flags
const mi::Uint32 DDSF_CAPS              = 0x00000001l;
const mi::Uint32 DDSF_HEIGHT            = 0x00000002l;
const mi::Uint32 DDSF_WIDTH             = 0x00000004l;
const mi::Uint32 DDSF_PITCH             = 0x00000008l;
const mi::Uint32 DDSF_PIXELFORMAT       = 0x00001000l;
const mi::Uint32 DDSF_MIPMAPCOUNT       = 0x00020000l;
const mi::Uint32 DDSF_LINEARSIZE        = 0x00080000l;
const mi::Uint32 DDSF_DEPTH             = 0x00800000l;

// pixel format flags
const mi::Uint32 DDSF_ALPHAPIXELS       = 0x00000001l;
const mi::Uint32 DDSF_FOURCC            = 0x00000004l;
const mi::Uint32 DDSF_RGB               = 0x00000040l;
const mi::Uint32 DDSF_RGBA              = 0x00000041l;

// compressed texture types
const mi::Uint32 FOURCC_DXT1            = 0x31545844l; // "DXT1" in reverse order
const mi::Uint32 FOURCC_DXT3            = 0x33545844l; // "DXT3" in reverse order
const mi::Uint32 FOURCC_DXT5            = 0x35545844l; // "DXT5" in reverse order

// floating point formats
const mi::Uint32 DDSF_R16F              = 111;
const mi::Uint32 DDSF_G16R16F           = 112;
const mi::Uint32 DDSF_A16B16G16R16F     = 113;
const mi::Uint32 DDSF_R32F              = 114;
const mi::Uint32 DDSF_G32R32F           = 115;
const mi::Uint32 DDSF_A32B32G32R32F     = 116;

// unsigned formats
const mi::Uint32 DDSF_R8G8B8            = 20;
const mi::Uint32 DDSF_A8R8G8B8          = 21;
const mi::Uint32 DDSF_X8R8G8B8          = 22;
const mi::Uint32 DDSF_R5G6B5            = 23;
const mi::Uint32 DDSF_X1R5G5B5          = 24;
const mi::Uint32 DDSF_A1R5G5B5          = 25;
const mi::Uint32 DDSF_A4R4G4B4          = 26;
const mi::Uint32 DDSF_R3G3B2            = 27;
const mi::Uint32 DDSF_A8                = 28;
const mi::Uint32 DDSF_A8R3G3B2          = 29;
const mi::Uint32 DDSF_X4R4G4B4          = 30;
const mi::Uint32 DDSF_A2B10G10R10       = 31;
const mi::Uint32 DDSF_A8B8G8R8          = 32;
const mi::Uint32 DDSF_X8B8G8R8          = 33;
const mi::Uint32 DDSF_G16R16            = 34;
const mi::Uint32 DDSF_A2R10G10B10       = 35;
const mi::Uint32 DDSF_A16B16G16R16      = 36;
const mi::Uint32 DDSF_A8P8              = 40;
const mi::Uint32 DDSF_P8                = 41;
const mi::Uint32 DDSF_L8                = 50;
const mi::Uint32 DDSF_L16               = 81;
const mi::Uint32 DDSF_A8L8              = 51;
const mi::Uint32 DDSF_A4L4              = 52;

// caps1 flags
const mi::Uint32 DDSF_COMPLEX           = 0x00000008l;
const mi::Uint32 DDSF_TEXTURE           = 0x00001000l;
const mi::Uint32 DDSF_MIPMAP            = 0x00400000l;

// caps2 flags
const mi::Uint32 DDSF_CUBEMAP           = 0x00000200l;
const mi::Uint32 DDSF_CUBEMAP_POSITIVEX = 0x00000400l;
const mi::Uint32 DDSF_CUBEMAP_NEGATIVEX = 0x00000800l;
const mi::Uint32 DDSF_CUBEMAP_POSITIVEY = 0x00001000l;
const mi::Uint32 DDSF_CUBEMAP_NEGATIVEY = 0x00002000l;
const mi::Uint32 DDSF_CUBEMAP_POSITIVEZ = 0x00004000l;
const mi::Uint32 DDSF_CUBEMAP_NEGATIVEZ = 0x00008000l;
const mi::Uint32 DDSF_CUBEMAP_ALL_FACES = 0x0000FC00l;
const mi::Uint32 DDSF_VOLUME            = 0x00200000l;

struct DXT_color_block
{
    mi::Uint16 m_col0;
    mi::Uint16 m_col1;
    mi::Uint8  m_row[4];
};

struct DXT3_alpha_block
{
    mi::Uint16 m_row[4];
};

struct DXT5_alpha_block
{
    mi::Uint8 m_alpha0;
    mi::Uint8 m_alpha1;
    mi::Uint8 m_row[6];
};

struct Pixel_format
{
    mi::Uint32 m_size;
    mi::Uint32 m_flags;
    mi::Uint32 m_four_cc;
    mi::Uint32 m_rgb_bit_count;
    mi::Uint32 m_r_bit_mask;
    mi::Uint32 m_g_bit_mask;
    mi::Uint32 m_b_bit_mask;
    mi::Uint32 m_a_bit_mask;
};

struct Header
{
    mi::Uint32 m_size;
    mi::Uint32 m_flags;
    mi::Uint32 m_height;
    mi::Uint32 m_width;
    mi::Uint32 m_pitch;
    mi::Uint32 m_depth;
    mi::Uint32 m_mipmap_count;
    mi::Uint32 m_reserved1[11];
    Pixel_format m_ddspf;
    mi::Uint32 m_caps1;
    mi::Uint32 m_caps2;
    mi::Uint32 m_reserved2[3];
};

enum Texture_type
{
    TEXTURE_NONE,
    TEXTURE_FLAT,    // 1D and 2D textures
    TEXTURE_3D,      // 3D textures
    TEXTURE_CUBEMAP  // cubemaps
};

enum Dds_compress_fmt {
    DXTC_none,
    DXTC1,
    DXTC3,
    DXTC5
};

} // namespace DDS

} // namespace MI

#endif // IO_IMAGE_DDS_DDS_TYPES_H
