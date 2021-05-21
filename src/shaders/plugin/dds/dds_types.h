/***************************************************************************************************
 * Copyright (c) 2005-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <string>

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

// four CC codes
const mi::Uint32 FOURCC_DXT1            = 0x31545844l; // "DXT1" in reverse order
const mi::Uint32 FOURCC_DXT3            = 0x33545844l; // "DXT3" in reverse order
const mi::Uint32 FOURCC_DXT5            = 0x35545844l; // "DXT5" in reverse order
const mi::Uint32 FOURCC_DX10            = 0x30315844l; // "DX10" in reverse order
const mi::Uint32 FOURCC_BC4U            = 0x55344342l; // "BC4U" in reverse order
const mi::Uint32 FOURCC_BC4S            = 0x53344342l; // "BC4S" in reverse order
const mi::Uint32 FOURCC_BC5U            = 0x55354342l; // "BC5U" in reverse order
const mi::Uint32 FOURCC_BC5S            = 0x53354342l; // "BC5S" in reverse order

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

enum Dxgi_format {
    DXGI_FORMAT_UNKNOWN                                 =   0,
    DXGI_FORMAT_R32G32B32A32_TYPELESS                   =   1,
    DXGI_FORMAT_R32G32B32A32_FLOAT                      =   2,
    DXGI_FORMAT_R32G32B32A32_UINT                       =   3,
    DXGI_FORMAT_R32G32B32A32_SINT                       =   4,
    DXGI_FORMAT_R32G32B32_TYPELESS                      =   5,
    DXGI_FORMAT_R32G32B32_FLOAT                         =   6,
    DXGI_FORMAT_R32G32B32_UINT                          =   7,
    DXGI_FORMAT_R32G32B32_SINT                          =   8,
    DXGI_FORMAT_R16G16B16A16_TYPELESS                   =   9,
    DXGI_FORMAT_R16G16B16A16_FLOAT                      =  10,
    DXGI_FORMAT_R16G16B16A16_UNORM                      =  11,
    DXGI_FORMAT_R16G16B16A16_UINT                       =  12,
    DXGI_FORMAT_R16G16B16A16_SNORM                      =  13,
    DXGI_FORMAT_R16G16B16A16_SINT                       =  14,
    DXGI_FORMAT_R32G32_TYPELESS                         =  15,
    DXGI_FORMAT_R32G32_FLOAT                            =  16,
    DXGI_FORMAT_R32G32_UINT                             =  17,
    DXGI_FORMAT_R32G32_SINT                             =  18,
    DXGI_FORMAT_R32G8X24_TYPELESS                       =  19,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT                    =  20,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS                =  21,
    DXGI_FORMAT_X32_TYPELESS_G8X24_UINT                 =  22,
    DXGI_FORMAT_R10G10B10A2_TYPELESS                    =  23,
    DXGI_FORMAT_R10G10B10A2_UNORM                       =  24,
    DXGI_FORMAT_R10G10B10A2_UINT                        =  25,
    DXGI_FORMAT_R11G11B10_FLOAT                         =  26,
    DXGI_FORMAT_R8G8B8A8_TYPELESS                       =  27,
    DXGI_FORMAT_R8G8B8A8_UNORM                          =  28,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB                     =  29,
    DXGI_FORMAT_R8G8B8A8_UINT                           =  30,
    DXGI_FORMAT_R8G8B8A8_SNORM                          =  31,
    DXGI_FORMAT_R8G8B8A8_SINT                           =  32,
    DXGI_FORMAT_R16G16_TYPELESS                         =  33,
    DXGI_FORMAT_R16G16_FLOAT                            =  34,
    DXGI_FORMAT_R16G16_UNORM                            =  35,
    DXGI_FORMAT_R16G16_UINT                             =  36,
    DXGI_FORMAT_R16G16_SNORM                            =  37,
    DXGI_FORMAT_R16G16_SINT                             =  38,
    DXGI_FORMAT_R32_TYPELESS                            =  39,
    DXGI_FORMAT_D32_FLOAT                               =  40,
    DXGI_FORMAT_R32_FLOAT                               =  41,
    DXGI_FORMAT_R32_UINT                                =  42,
    DXGI_FORMAT_R32_SINT                                =  43,
    DXGI_FORMAT_R24G8_TYPELESS                          =  44,
    DXGI_FORMAT_D24_UNORM_S8_UINT                       =  45,
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS                   =  46,
    DXGI_FORMAT_X24_TYPELESS_G8_UINT                    =  47,
    DXGI_FORMAT_R8G8_TYPELESS                           =  48,
    DXGI_FORMAT_R8G8_UNORM                              =  49,
    DXGI_FORMAT_R8G8_UINT                               =  50,
    DXGI_FORMAT_R8G8_SNORM                              =  51,
    DXGI_FORMAT_R8G8_SINT                               =  52,
    DXGI_FORMAT_R16_TYPELESS                            =  53,
    DXGI_FORMAT_R16_FLOAT                               =  54,
    DXGI_FORMAT_D16_UNORM                               =  55,
    DXGI_FORMAT_R16_UNORM                               =  56,
    DXGI_FORMAT_R16_UINT                                =  57,
    DXGI_FORMAT_R16_SNORM                               =  58,
    DXGI_FORMAT_R16_SINT                                =  59,
    DXGI_FORMAT_R8_TYPELESS                             =  60,
    DXGI_FORMAT_R8_UNORM                                =  61,
    DXGI_FORMAT_R8_UINT                                 =  62,
    DXGI_FORMAT_R8_SNORM                                =  63,
    DXGI_FORMAT_R8_SINT                                 =  64,
    DXGI_FORMAT_A8_UNORM                                =  65,
    DXGI_FORMAT_R1_UNORM                                =  66,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP                      =  67,
    DXGI_FORMAT_R8G8_B8G8_UNORM                         =  68,
    DXGI_FORMAT_G8R8_G8B8_UNORM                         =  69,
    DXGI_FORMAT_BC1_TYPELESS                            =  70,
    DXGI_FORMAT_BC1_UNORM                               =  71,
    DXGI_FORMAT_BC1_UNORM_SRGB                          =  72,
    DXGI_FORMAT_BC2_TYPELESS                            =  73,
    DXGI_FORMAT_BC2_UNORM                               =  74,
    DXGI_FORMAT_BC2_UNORM_SRGB                          =  75,
    DXGI_FORMAT_BC3_TYPELESS                            =  76,
    DXGI_FORMAT_BC3_UNORM                               =  77,
    DXGI_FORMAT_BC3_UNORM_SRGB                          =  78,
    DXGI_FORMAT_BC4_TYPELESS                            =  79,
    DXGI_FORMAT_BC4_UNORM                               =  80,
    DXGI_FORMAT_BC4_SNORM                               =  81,
    DXGI_FORMAT_BC5_TYPELESS                            =  82,
    DXGI_FORMAT_BC5_UNORM                               =  83,
    DXGI_FORMAT_BC5_SNORM                               =  84,
    DXGI_FORMAT_B5G6R5_UNORM                            =  85,
    DXGI_FORMAT_B5G5R5A1_UNORM                          =  86,
    DXGI_FORMAT_B8G8R8A8_UNORM                          =  87,
    DXGI_FORMAT_B8G8R8X8_UNORM                          =  88,
    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM              =  89,
    DXGI_FORMAT_B8G8R8A8_TYPELESS                       =  90,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB                     =  91,
    DXGI_FORMAT_B8G8R8X8_TYPELESS                       =  92,
    DXGI_FORMAT_B8G8R8X8_UNORM_SRGB                     =  93,
    DXGI_FORMAT_BC6H_TYPELESS                           =  94,
    DXGI_FORMAT_BC6H_UF16                               =  95,
    DXGI_FORMAT_BC6H_SF16                               =  96,
    DXGI_FORMAT_BC7_TYPELESS                            =  97,
    DXGI_FORMAT_BC7_UNORM                               =  98,
    DXGI_FORMAT_BC7_UNORM_SRGB                          =  99,
    DXGI_FORMAT_AYUV                                    = 100,
    DXGI_FORMAT_Y410                                    = 101,
    DXGI_FORMAT_Y416                                    = 102,
    DXGI_FORMAT_NV12                                    = 103,
    DXGI_FORMAT_P010                                    = 104,
    DXGI_FORMAT_P016                                    = 105,
    DXGI_FORMAT_420_OPAQUE                              = 106,
    DXGI_FORMAT_YUY2                                    = 107,
    DXGI_FORMAT_Y210                                    = 108,
    DXGI_FORMAT_Y216                                    = 109,
    DXGI_FORMAT_NV11                                    = 110,
    DXGI_FORMAT_AI44                                    = 111,
    DXGI_FORMAT_IA44                                    = 112,
    DXGI_FORMAT_P8                                      = 113,
    DXGI_FORMAT_A8P8                                    = 114,
    DXGI_FORMAT_B4G4R4A4_UNORM                          = 115,
    DXGI_FORMAT_P208                                    = 116,
    DXGI_FORMAT_V208                                    = 117,
    DXGI_FORMAT_V408                                    = 118,
    DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE         = 119,
    DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE = 120,
    // Unclear whether the value is correct. Should not occur in practice. Do no use for
    // comparisons.
    DXGI_FORMAT_FORCE_UINT                              = 0xffffffffU
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
    mi::Uint32 m_depth; // or linear size
    mi::Uint32 m_mipmap_count;
    mi::Uint32 m_reserved1[11];
    Pixel_format m_ddspf;
    mi::Uint32 m_caps1;
    mi::Uint32 m_caps2;
    mi::Uint32 m_caps3;
    mi::Uint32 m_caps4;
    mi::Uint32 m_reserved2;
};

struct Header_dx10
{
    Dxgi_format m_dxgi_format;
    // The enum for the DXGI format is documented to be 4 bytes, but the size of the resource
    // dimension enum does not seem to be documented.
    mi::Uint32 m_resource_dimension;
    mi::Uint32 m_misc_flag;
    mi::Uint32 m_array_size;
    mi::Uint32 m_misc_flags2;
};

std::string get_dxgi_format_string( Dxgi_format value);

} // namespace DDS

} // namespace MI

#endif // IO_IMAGE_DDS_DDS_TYPES_H
