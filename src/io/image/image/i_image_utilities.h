/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_IMAGE_IMAGE_I_IMAGE_UTILITIES_H
#define IO_IMAGE_IMAGE_I_IMAGE_UTILITIES_H

/// WARNING: This file is also used by external (plugin) code.
/// Be careful with the dependencies of this file.

#include <mi/base/types.h>
#include <cstring>

namespace MI {

namespace IMAGE {

/// Dummy type to select the overload/constructor for file-based images.
struct File_based_helper {};
typedef const File_based_helper* File_based;

/// Dummy type to select the overload/constructor for container-based images.
struct Container_based_helper {};
typedef const Container_based_helper* Container_based;

// Dummy type to select the overload/constructor for memory-based images.
struct Memory_based_helper {};
typedef const Memory_based_helper* Memory_based;

constexpr mi::Uint32 default_tile_width  = 64;
constexpr mi::Uint32 default_tile_height = 64;

/// The supported pixel types.
enum Pixel_type {
    PT_UNDEF,      /// undefined/invalid pixel type
    PT_SINT8,      /// pixel type "Sint8"
    PT_SINT32,     /// pixel type "Sint32"
    PT_FLOAT32,    /// pixel type "Float32"
    PT_FLOAT32_2,  /// pixel type "Float32<2>"
    PT_FLOAT32_3,  /// pixel type "Float32<3>"
    PT_FLOAT32_4,  /// pixel type "Float32<4>"
    PT_RGB,        /// pixel type "Rgb"
    PT_RGBA,       /// pixel type "Rgba"
    PT_RGBE,       /// pixel type "Rgbe"
    PT_RGBEA,      /// pixel type "Rgbea"
    PT_RGB_16,     /// pixel type "Rgb_16"
    PT_RGBA_16,    /// pixel type "Rgba_16"
    PT_RGB_FP,     /// pixel type "Rgb_fp"
    PT_COLOR       /// pixel type "Color"
};

/// Converts a pixel type from its string to enum representation.
inline Pixel_type convert_pixel_type_string_to_enum( const char* pixel_type)
{
    if( !pixel_type) return PT_UNDEF;
    if( strcmp( pixel_type, "Sint8")      == 0) return PT_SINT8;
    if( strcmp( pixel_type, "Sint32")     == 0) return PT_SINT32;
    if( strcmp( pixel_type, "Float32")    == 0) return PT_FLOAT32;
    if( strcmp( pixel_type, "Float32<2>") == 0) return PT_FLOAT32_2;
    if( strcmp( pixel_type, "Float32<3>") == 0) return PT_FLOAT32_3;
    if( strcmp( pixel_type, "Float32<4>") == 0) return PT_FLOAT32_4;
    if( strcmp( pixel_type, "Rgb")        == 0) return PT_RGB;
    if( strcmp( pixel_type, "Rgba")       == 0) return PT_RGBA;
    if( strcmp( pixel_type, "Rgbe")       == 0) return PT_RGBE;
    if( strcmp( pixel_type, "Rgbea")      == 0) return PT_RGBEA;
    if( strcmp( pixel_type, "Rgb_16")     == 0) return PT_RGB_16;
    if( strcmp( pixel_type, "Rgba_16")    == 0) return PT_RGBA_16;
    if( strcmp( pixel_type, "Rgb_fp")     == 0) return PT_RGB_FP;
    if( strcmp( pixel_type, "Color")      == 0) return PT_COLOR;
    return PT_UNDEF;
}

/// Converts a pixel type from its enum to string representation.
constexpr const char* convert_pixel_type_enum_to_string( Pixel_type pixel_type)
{
    switch( pixel_type) {
        case PT_UNDEF:     return nullptr;
        case PT_SINT8:     return "Sint8";
        case PT_SINT32:    return "Sint32";
        case PT_FLOAT32:   return "Float32";
        case PT_FLOAT32_2: return "Float32<2>";
        case PT_FLOAT32_3: return "Float32<3>";
        case PT_FLOAT32_4: return "Float32<4>";
        case PT_RGB:       return "Rgb";
        case PT_RGBA:      return "Rgba";
        case PT_RGBE:      return "Rgbe";
        case PT_RGBEA:     return "Rgbea";
        case PT_RGB_16:    return "Rgb_16";
        case PT_RGBA_16:   return "Rgba_16";
        case PT_RGB_FP:    return "Rgb_fp";
        case PT_COLOR:     return "Color";
        default:           return nullptr;
    }
}

/// Returns the number of components of a given pixel type.
///
/// For example, 3 for PT_RGB and 4 for PT_RGBA.
constexpr int get_components_per_pixel( Pixel_type pixel_type);

/// Returns the number of bytes used by a component of a given pixel type.
///
/// For example, 1 for PT_RGB, 2 for PT_RGB_16, and 4 for PT_RGB_FP.
constexpr int get_bytes_per_component( Pixel_type pixel_type);

/// Return the number of bytes used by a pixel of a given pixel type.
///
/// This is the product of #get_components_per_pixel() and #get_bytes_per_component().
constexpr mi::Uint32 get_bytes_per_pixel( Pixel_type pixel_type);

/// Indicates whether the pixel type has an alpha channel.
constexpr bool has_alpha( Pixel_type pixel_type);

/// Returns the default gamma value for a given pixel type.
///
/// The default gamma value is 1.0 for HDR pixel types and 2.2 for LDR pixel types.
constexpr mi::Float32 get_default_gamma( Pixel_type pixel_type);

/// Indicates whether \p selector is the alpha channel selector.
bool is_valid_alpha_channel( const char* selector);

/// Indicates whether \p selector is a valid RGBA channel selector.
bool is_valid_rgba_channel( const char* selector);

/// Returns the channel index for a valid RGBA channel selector (or -1 for invalid ones).
mi::Size get_channel_index( const char* selector);

/// Returns the pixel type of a channel.
///
/// Invalid pixel type/selector combinations are:
/// - \p pixel_type is not an RGB or RGBA pixel type
/// - \p selector is not an RGBA selector
///
/// \param pixel_type   The pixel type of the mipmap/canvas/tile.
/// \param selector     The RGBA channel selector.
/// \return             Returns PT_UNDEF for invalid pixel type/selector combinations.
///                     Otherwise, returns PT_SINT8 or PT_FLOAT32, depending on
///                     \p pixel_type.
Pixel_type get_pixel_type_for_channel( Pixel_type pixel_type, const char* selector);

template <Pixel_type>
struct Pixel_type_traits {
};

template <>
struct Pixel_type_traits<PT_SINT8>
{
    /// For most purposes, in particular for pixel type conversion, the data is actually treated as
    /// \em unsigned 8-bit integer. The pixel type name should be fixed to reflect this, but this is
    /// not trivial without breaking user code since the API boundary uses strings instead of enums
    /// to encode pixel type names.
    typedef mi::Uint8 Base_type;
    static constexpr int s_components_per_pixel = 1;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = false;
};

template <>
struct Pixel_type_traits<PT_SINT32>
{
    typedef mi::Sint32 Base_type;
    static constexpr int s_components_per_pixel = 1;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_FLOAT32>
{
    typedef mi::Float32 Base_type;
    static constexpr int s_components_per_pixel = 1;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_FLOAT32_2>
{
    typedef mi::Float32 Base_type;
    static constexpr int s_components_per_pixel = 2;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_FLOAT32_3>
{
    typedef mi::Float32 Base_type;
    static constexpr int s_components_per_pixel = 3;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_FLOAT32_4>
{
    typedef mi::Float32 Base_type;
    static constexpr int s_components_per_pixel = 4;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_RGB>
{
    typedef mi::Uint8 Base_type;
    static constexpr int s_components_per_pixel = 3;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = false;
};

template <>
struct Pixel_type_traits<PT_RGBA>
{
    typedef mi::Uint8 Base_type;
    static constexpr int s_components_per_pixel = 4;
    static constexpr bool s_has_alpha = true;
    static constexpr bool s_linear = false;
};

template <>
struct Pixel_type_traits<PT_RGBE>
{
    typedef mi::Uint8 Base_type;
    static constexpr int s_components_per_pixel = 4;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_RGBEA>
{
    typedef mi::Uint8 Base_type;
    static constexpr int s_components_per_pixel = 5;
    static constexpr bool s_has_alpha = true;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_RGB_16>
{
    typedef mi::Uint16 Base_type;
    static constexpr int s_components_per_pixel = 3;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = false;
};

template <>
struct Pixel_type_traits<PT_RGBA_16>
{
    typedef mi::Uint16 Base_type;
    static constexpr int s_components_per_pixel = 4;
    static constexpr bool s_has_alpha = true;
    static constexpr bool s_linear = false;
};

template <>
struct Pixel_type_traits<PT_RGB_FP>
{
    typedef mi::Float32 Base_type;
    static constexpr int s_components_per_pixel = 3;
    static constexpr bool s_has_alpha = false;
    static constexpr bool s_linear = true;
};

template <>
struct Pixel_type_traits<PT_COLOR>
{
    typedef mi::Float32 Base_type;
    static constexpr int s_components_per_pixel = 4;
    static constexpr bool s_has_alpha = true;
    static constexpr bool s_linear = true;
};

constexpr int get_components_per_pixel( Pixel_type pixel_type)
{
    switch( pixel_type) {
        case PT_UNDEF:     return 0;
        case PT_SINT8:     return Pixel_type_traits<PT_SINT8    >::s_components_per_pixel;
        case PT_SINT32:    return Pixel_type_traits<PT_SINT32   >::s_components_per_pixel;
        case PT_FLOAT32:   return Pixel_type_traits<PT_FLOAT32  >::s_components_per_pixel;
        case PT_FLOAT32_2: return Pixel_type_traits<PT_FLOAT32_2>::s_components_per_pixel;
        case PT_FLOAT32_3: return Pixel_type_traits<PT_FLOAT32_3>::s_components_per_pixel;
        case PT_FLOAT32_4: return Pixel_type_traits<PT_FLOAT32_4>::s_components_per_pixel;
        case PT_RGB:       return Pixel_type_traits<PT_RGB      >::s_components_per_pixel;
        case PT_RGBA:      return Pixel_type_traits<PT_RGBA     >::s_components_per_pixel;
        case PT_RGBE:      return Pixel_type_traits<PT_RGBE     >::s_components_per_pixel;
        case PT_RGBEA:     return Pixel_type_traits<PT_RGBEA    >::s_components_per_pixel;
        case PT_RGB_16:    return Pixel_type_traits<PT_RGB_16   >::s_components_per_pixel;
        case PT_RGBA_16:   return Pixel_type_traits<PT_RGBA_16  >::s_components_per_pixel;
        case PT_RGB_FP:    return Pixel_type_traits<PT_RGB_FP   >::s_components_per_pixel;
        case PT_COLOR:     return Pixel_type_traits<PT_COLOR    >::s_components_per_pixel;
        default:           return 0;
    }
}

constexpr int get_bytes_per_component( Pixel_type pixel_type)
{
    switch( pixel_type) {
        case PT_UNDEF:     return 0;
        case PT_SINT8:     return (int) sizeof( Pixel_type_traits<PT_SINT8    >::Base_type);
        case PT_SINT32:    return (int) sizeof( Pixel_type_traits<PT_SINT32   >::Base_type);
        case PT_FLOAT32:   return (int) sizeof( Pixel_type_traits<PT_FLOAT32  >::Base_type);
        case PT_FLOAT32_2: return (int) sizeof( Pixel_type_traits<PT_FLOAT32_2>::Base_type);
        case PT_FLOAT32_3: return (int) sizeof( Pixel_type_traits<PT_FLOAT32_3>::Base_type);
        case PT_FLOAT32_4: return (int) sizeof( Pixel_type_traits<PT_FLOAT32_4>::Base_type);
        case PT_RGB:       return (int) sizeof( Pixel_type_traits<PT_RGB      >::Base_type);
        case PT_RGBA:      return (int) sizeof( Pixel_type_traits<PT_RGBA     >::Base_type);
        case PT_RGBE:      return (int) sizeof( Pixel_type_traits<PT_RGBE     >::Base_type);
        case PT_RGBEA:     return (int) sizeof( Pixel_type_traits<PT_RGBEA    >::Base_type);
        case PT_RGB_16:    return (int) sizeof( Pixel_type_traits<PT_RGB_16   >::Base_type);
        case PT_RGBA_16:   return (int) sizeof( Pixel_type_traits<PT_RGBA_16  >::Base_type);
        case PT_RGB_FP:    return (int) sizeof( Pixel_type_traits<PT_RGB_FP   >::Base_type);
        case PT_COLOR:     return (int) sizeof( Pixel_type_traits<PT_COLOR    >::Base_type);
        default:           return 0;
    }
}

constexpr mi::Uint32 get_bytes_per_pixel( Pixel_type pixel_type)
{
#define MI_IMAGE_BYTES_PER_PIXEL(T) \
    Pixel_type_traits<T>::s_components_per_pixel * (int) sizeof( Pixel_type_traits<T>::Base_type)

    switch( pixel_type) {
        case PT_UNDEF:     return 0;
        case PT_SINT8:     return MI_IMAGE_BYTES_PER_PIXEL( PT_SINT8);
        case PT_SINT32:    return MI_IMAGE_BYTES_PER_PIXEL( PT_SINT32);
        case PT_FLOAT32:   return MI_IMAGE_BYTES_PER_PIXEL( PT_FLOAT32);
        case PT_FLOAT32_2: return MI_IMAGE_BYTES_PER_PIXEL( PT_FLOAT32_2);
        case PT_FLOAT32_3: return MI_IMAGE_BYTES_PER_PIXEL( PT_FLOAT32_3);
        case PT_FLOAT32_4: return MI_IMAGE_BYTES_PER_PIXEL( PT_FLOAT32_4);
        case PT_RGB:       return MI_IMAGE_BYTES_PER_PIXEL( PT_RGB);
        case PT_RGBA:      return MI_IMAGE_BYTES_PER_PIXEL( PT_RGBA);
        case PT_RGBE:      return MI_IMAGE_BYTES_PER_PIXEL( PT_RGBE);
        case PT_RGBEA:     return MI_IMAGE_BYTES_PER_PIXEL( PT_RGBEA);
        case PT_RGB_16:    return MI_IMAGE_BYTES_PER_PIXEL( PT_RGB_16);
        case PT_RGBA_16:   return MI_IMAGE_BYTES_PER_PIXEL( PT_RGBA_16);
        case PT_RGB_FP:    return MI_IMAGE_BYTES_PER_PIXEL( PT_RGB_FP);
        case PT_COLOR:     return MI_IMAGE_BYTES_PER_PIXEL( PT_COLOR);
        default:           return 0;
    }

#undef MI_IMAGE_BYTES_PER_PIXEL
}

constexpr bool has_alpha( Pixel_type pixel_type)
{
    switch( pixel_type) {
        case PT_UNDEF:     return false;
        case PT_SINT8:     return Pixel_type_traits<PT_SINT8    >::s_has_alpha;
        case PT_SINT32:    return Pixel_type_traits<PT_SINT32   >::s_has_alpha;
        case PT_FLOAT32:   return Pixel_type_traits<PT_FLOAT32  >::s_has_alpha;
        case PT_FLOAT32_2: return Pixel_type_traits<PT_FLOAT32_2>::s_has_alpha;
        case PT_FLOAT32_3: return Pixel_type_traits<PT_FLOAT32_3>::s_has_alpha;
        case PT_FLOAT32_4: return Pixel_type_traits<PT_FLOAT32_4>::s_has_alpha;
        case PT_RGB:       return Pixel_type_traits<PT_RGB      >::s_has_alpha;
        case PT_RGBA:      return Pixel_type_traits<PT_RGBA     >::s_has_alpha;
        case PT_RGBE:      return Pixel_type_traits<PT_RGBE     >::s_has_alpha;
        case PT_RGBEA:     return Pixel_type_traits<PT_RGBEA    >::s_has_alpha;
        case PT_RGB_16:    return Pixel_type_traits<PT_RGB_16   >::s_has_alpha;
        case PT_RGBA_16:   return Pixel_type_traits<PT_RGBA_16  >::s_has_alpha;
        case PT_RGB_FP:    return Pixel_type_traits<PT_RGB_FP   >::s_has_alpha;
        case PT_COLOR:     return Pixel_type_traits<PT_COLOR    >::s_has_alpha;
        default:           return false;
    }
}

constexpr mi::Float32 get_default_gamma( Pixel_type pixel_type)
{
    switch( pixel_type) {
        case PT_UNDEF:     return 1.0f;
        case PT_SINT8:     return Pixel_type_traits<PT_SINT8    >::s_linear ? 1.0f : 2.2f;
        case PT_SINT32:    return Pixel_type_traits<PT_SINT32   >::s_linear ? 1.0f : 2.2f;
        case PT_FLOAT32:   return Pixel_type_traits<PT_FLOAT32  >::s_linear ? 1.0f : 2.2f;
        case PT_FLOAT32_2: return Pixel_type_traits<PT_FLOAT32_2>::s_linear ? 1.0f : 2.2f;
        case PT_FLOAT32_3: return Pixel_type_traits<PT_FLOAT32_3>::s_linear ? 1.0f : 2.2f;
        case PT_FLOAT32_4: return Pixel_type_traits<PT_FLOAT32_4>::s_linear ? 1.0f : 2.2f;
        case PT_RGB:       return Pixel_type_traits<PT_RGB      >::s_linear ? 1.0f : 2.2f;
        case PT_RGBA:      return Pixel_type_traits<PT_RGBA     >::s_linear ? 1.0f : 2.2f;
        case PT_RGBE:      return Pixel_type_traits<PT_RGBE     >::s_linear ? 1.0f : 2.2f;
        case PT_RGBEA:     return Pixel_type_traits<PT_RGBEA    >::s_linear ? 1.0f : 2.2f;
        case PT_RGB_16:    return Pixel_type_traits<PT_RGB_16   >::s_linear ? 1.0f : 2.2f;
        case PT_RGBA_16:   return Pixel_type_traits<PT_RGBA_16  >::s_linear ? 1.0f : 2.2f;
        case PT_RGB_FP:    return Pixel_type_traits<PT_RGB_FP   >::s_linear ? 1.0f : 2.2f;
        case PT_COLOR:     return Pixel_type_traits<PT_COLOR    >::s_linear ? 1.0f : 2.2f;
        default:           return 1.0f;
    }
}

inline bool is_valid_alpha_channel( const char* selector)
{
    if( !selector)
        return false;
    if( selector[0] == 'A' && selector[1] == '\0')
        return true;
    return false;
}

inline bool is_valid_rgba_channel( const char* selector)
{
    if( !selector)
        return false;
    if( (    selector[0] == 'R'
          || selector[0] == 'G'
          || selector[0] == 'B'
          || selector[0] == 'A')
        && selector[1] == '\0')
        return true;
    return false;
}

inline mi::Size get_channel_index( const char* selector)
{
    if( !selector || !selector[0] || selector[1])
        return static_cast<mi::Size>( -1);

    if( selector[0] == 'R') return 0;
    if( selector[0] == 'G') return 1;
    if( selector[0] == 'B') return 2;
    if( selector[0] == 'A') return 3;

    return static_cast<mi::Size>( -1);
}

inline Pixel_type get_pixel_type_for_channel( Pixel_type pixel_type, const char* selector)
{
    if( !is_valid_rgba_channel( selector))
        return PT_UNDEF;

    switch( pixel_type) {

        case PT_SINT8:
        case PT_SINT32:
        case PT_FLOAT32:
        case PT_FLOAT32_2:
        case PT_FLOAT32_3:
        case PT_FLOAT32_4:
            return PT_UNDEF;

        case PT_RGB:
            return is_valid_alpha_channel( selector) ? PT_UNDEF : PT_SINT8;

        case PT_RGBA:
            return PT_SINT8;

        case PT_RGBE:
        case PT_RGB_16:
        case PT_RGB_FP:
            return is_valid_alpha_channel( selector) ? PT_UNDEF : PT_FLOAT32;

        case PT_RGBEA:
        case PT_RGBA_16:
        case PT_COLOR:
            return PT_FLOAT32;

        case PT_UNDEF:
            return PT_UNDEF; }

    return PT_UNDEF;
}

} // namespace IMAGE

} // namespace MI

#endif // IO_IMAGE_IMAGE_I_IMAGE_UTILITIES_H
