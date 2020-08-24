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

/// There are the following five groups of pixel conversion/copying functions:
///
/// (1) convert()
/// (2) convert<Source>()
/// (3) Pixel_converter<Source,Target>::convert()
/// (4) copy()
/// (5) Pixel_copier<Type>::copy()
///
/// In group (1) source and destination pixel type are parameters; in group (2) the source pixel
/// type is a template parameter; and in group (3) both pixel types are template parameters.
///
/// In group (4) the pixel type is a parameter; and in group (5) the pixel type is a template
/// parameter.
///
/// Typically, there are two or three functions in each group for single pixels, for a contiguous
/// region of pixels, and for rectangular regions with arbitrary row strides.

#ifndef IO_IMAGE_IMAGE_IMAGE_PIXEL_CONVERSION_H
#define IO_IMAGE_IMAGE_IMAGE_PIXEL_CONVERSION_H

#include "i_image_utilities.h"
#include "i_image_quantization.h"

#if defined(HAS_SSE) || defined(SSE_INTRINSICS)
 #include <xmmintrin.h>
#endif

#include <mi/math/function.h>
#include <base/lib/log/i_log_assert.h>

#include <algorithm>

namespace MI {

namespace IMAGE {


/// Performs a gamma correction for floating point pixel types.
///
/// For types with three \p components or less, every component is adjusted. For 4 components,
/// i.e. \c PT_COLOR, the alpha component is skipped.
///
/// \param data       The pixel data to be manipulated.
/// \param count      The number of pixels in \p data.
/// \param components The number of components of each pixel (3 for PT_RGB_FP, 4 for PT_COLOR).
/// \param exponent   The exponent that is applied to the red, green, and blue channel. For
///                   arbitrary gamma changes this exponent is the quotient of the old and new gamma
///                   value. For decoding gamma compressed data into linear data the exponent is the
///                   gamma value itself, and for encoding linear data into gamma compressed data
///                   the exponent is the inverse gamma value.
MI_HOST_DEVICE_INLINE void adjust_gamma( mi::Float32* data, mi::Size count, mi::Uint32 components, mi::Float32 exponent);

/// Indicates whether a particular pixel type conversion is implemented.
///
/// Currently, pixel type conversion is implemented between all valid pixel types (but not for
/// PT_UNDEF). Since this does not need to be the case in general, this method can be used to
/// query whether the free convert() methods can successfully convert from type \p Source to
/// type \p Dest.
///
/// \param Source   The pixel type to convert from.
/// \param Dest     The pixel type to convert to.
/// \return         \c true if the pixel type conversion from \p Source to \p Dest is implemented,
///                 \c false otherwise.
MI_HOST_DEVICE_INLINE bool exists_pixel_conversion( Pixel_type Source, Pixel_type Dest);

/// Converts a contiguous region of pixels from the source format to the destination format.
///
/// If the source pixel type a compile-time constant, you might want to use #convert<Source>()
/// instead. If both pixel types are a compile-time constant, you want to use
/// #Pixel_converter<Source,Dest>::convert() instead.
///
/// This method invokes #copy() if \p Source == \p Dest.
///
/// \param source   The first pixel to convert from pixel type \p Source.
/// \param dest     The first pixel to store the result in pixel type \p Dest.
/// \param Source   The pixel type of \p source.
/// \param Dest     The pixel type of \p dest.
/// \param count    The number of pixels to convert.
MI_HOST_DEVICE_INLINE bool convert(
    const void* source, void* dest, Pixel_type Source, Pixel_type Dest, mi::Size count = 1);

/// Converts a rectangular region of pixels with arbitrary row strides from the source format to
/// the destination format.
///
/// If the source pixel type a compile-time constant, you might want to use #convert<Source>()
/// instead. If both pixel types are a compile-time constant, you want to use
/// #Pixel_converter<Source,Dest>::convert() instead.
///
/// This method invokes #copy() if \p Source == \p Dest.
///
/// \param source          The first pixel of the first row to convert from pixel type
///                        \p Source.
/// \param dest            The first pixel of the first row to store the result in pixel type
///                        \p Dest.
/// \param Source          The pixel type of \p source.
/// \param Dest            The pixel type of \p dest.
/// \param width           The number of pixels in x direction to convert.
/// \param height          The number of pixels in y direction to convert.
/// \param source_stride   Offset between subsequent rows of \p source (in bytes). Can be
///                        negative in order to flip the row order, but then \p source must
///                        point to the first pixel of the last row.
/// \param dest_stride     Offset between subsequent rows of \p dest (in bytes). Can be
///                        negative in order to flip the row order, but then \p dest must
///                        point to the first pixel of the last row.
MI_HOST_DEVICE_INLINE bool convert(
    const void* source, void* dest,
    Pixel_type Source, Pixel_type Dest,
    mi::Size width, mi::Size height,
    mi::Difference source_stride, mi::Difference dest_stride);

/// Converts a contiguous region of pixels from the source format to the destination format.
///
/// See #convert(const void*,void*,Pixel_type,Pixel_type,mi::Size) for argument details. The first
/// pixel type argument there is a template parameter here.
template <Pixel_type Source>
MI_HOST_DEVICE_INLINE bool convert( const void* source, void* dest, Pixel_type Dest, mi::Size count = 1);

/// Converts rectangular region of pixels with arbitrary row strides from the source format to
/// the destination format.
///
/// See #convert(const void*,void*,Pixel_type,Pixel_type,mi::Size,mi::Size,mi::Difference,
/// mi::Difference) for argument details. The first pixel type argument there is a template
/// parameter here.
template <Pixel_type Source>
MI_HOST_DEVICE_INLINE bool convert(
    const void* source, void* dest,
    Pixel_type Dest,
    mi::Size width, mi::Size height,
    mi::Difference source_stride, mi::Difference dest_stride);

/// Copies a contiguous region of pixels.
///
/// \param source   The first pixel to copy from pixel type \p Type.
/// \param dest     The first pixel to store the result in pixel type \p Type.
/// \param Type     The pixel type of \p source and \p dest.
/// \param count    The number of pixels to copy.
MI_HOST_DEVICE_INLINE bool copy(
    const void* source, void* dest, Pixel_type Type, mi::Size count = 1);

/// Copies a rectangular region of pixels with arbitrary row strides.
///
/// \param source          The first pixel of the first row to copy from pixel type
///                        \p Type.
/// \param dest            The first pixel of the first row to store the result in pixel type
///                        \p Type.
/// \param Type            The pixel type of \p source and \p dest.
/// \param width           The number of pixels in x direction to copy.
/// \param height          The number of pixels in y direction to copy.
/// \param source_stride   Offset between subsequent rows of \p source (in bytes). Can be
///                        negative in order to flip the row order, but then \p source must
///                        point to the first pixel of the last row.
/// \param dest_stride     Offset between subsequent rows of \p dest (in bytes). Can be
///                        negative in order to flip the row order, but then \p dest must
///                        point to the first pixel of the last row.
MI_HOST_DEVICE_INLINE bool copy(
    const void* source, void* dest,
    Pixel_type Type,
    mi::Size width, mi::Size height,
    mi::Difference source_stride, mi::Difference dest_stride);

/// Converts a region of pixels from a fixed source pixel type to a fixed destination pixel type.
///
/// The region can be either a single pixel, a contiguous region (typically a scanline, or a part
/// thereof), or a rectangular (possibly non-contiguous) region (composed of rows, each row is a
/// contiguous region).
///
/// Note that not all pixel type combinations are implemented. Implemented are all pairs of
/// Pixel_type, except for the combinations involving PT_UNDEF, PT_SINT32, PT_FLOAT32_3, and
/// PT_FLOAT32_4. The latter three types are mapped to PT_RGBA, PT_RGB_FP, and PT_COLOR by the
/// free convert() methods, i.e., they are using the same pixel type conversion code as the
/// latter types.
///
/// WARNING: This class does not use memcpy() if source and destination pixel types are equal. If
/// they are equal, use Pixel_copier instead if the decision can be made at compile time. If the
/// decision can only be made at runtime, you will have to use the free convert() (or copy())
/// methods anyway, which will automatically use the Pixel_copier if source and destination pixel
/// types are equal.
template <Pixel_type Source, Pixel_type Dest>
struct Pixel_converter
{
    /// Typedef for the underlying base type of Source
    typedef typename Pixel_type_traits<Source>::Base_type Source_base_type;

    /// Typedef for the underlying base type of Dest
    typedef typename Pixel_type_traits<Dest>::Base_type   Dest_base_type;

    /// Converts a single pixel from the source format to the destination format.
    ///
    /// \param source   The pixel to convert from pixel type \p Source.
    /// \param dest     The pixel to store the result in pixel type \p Dest.
    static MI_HOST_DEVICE_INLINE void convert( const Source_base_type* source, Dest_base_type* dest);

    /// Converts a contiguous region of pixels from the source format to the destination format.
    ///
    /// \param source   The first pixel to convert from pixel type \p Source.
    /// \param dest     The first pixel to store the result in pixel type \p Dest.
    /// \param count    The number of pixels to convert.
    static MI_HOST_DEVICE_INLINE void convert( const Source_base_type* source, Dest_base_type* dest, mi::Size count);

    /// Converts a rectangular region of pixels with arbitrary row strides from the source format to
    /// the destination format.
    ///
    /// \param source          The first pixel of the first row to convert from pixel type
    ///                        \p Source.
    /// \param dest            The first pixel of the first row to store the result in pixel type
    ///                        \p Dest.
    /// \param width           The number of pixels in x direction to convert.
    /// \param height          The number of pixels in y direction to convert.
    /// \param source_stride   Offset between subsequent rows of \p source (in bytes). Can be
    ///                        negative in order to flip the row order, but then \p source must
    ///                        point to the first pixel of the last row.
    /// \param dest_stride     Offset between subsequent rows of \p dest (in bytes). Can be
    ///                        negative in order to flip the row order, but then \p dest must
    ///                        point to the first pixel of the last row.
    static MI_HOST_DEVICE_INLINE void convert(
        const Source_base_type* source, Dest_base_type* dest,
        mi::Size width, mi::Size height,
        mi::Difference source_stride, mi::Difference dest_stride);

    /// Converts a contiguous region of pixels from the source format to the destination format.
    ///
    /// This variant with "const void*" and "void*" arguments exists only to hide the casts in all
    /// the callers below.
    ///
    /// For documentation, see the corresponding method above.
    static MI_HOST_DEVICE_INLINE void convert( const void* source, void* dest, mi::Size count = 1);

    /// Converts a rectangular region of pixels with arbitrary row strides from the source format to
    /// the destination format.
    ///
    /// This variant with "const void*" and "void*" arguments exists only to hide the casts in all
    /// the callers below.
    ///
    /// For documentation, see the corresponding method above.
    static MI_HOST_DEVICE_INLINE void convert(
        const void* source, void* dest,
        mi::Size width, mi::Size height,
        mi::Difference source_stride, mi::Difference dest_stride);
};

/// Copies a region of pixels with a fixed pixel type.
///
/// The region can be either a single pixel, a contiguous region (typically a scanline, or a part
/// thereof), or a rectangular (possibly non-contiguous) region (composed of rows, each row is a
/// contiguous region).
///
/// The implementation uses memcpy() and assumes that the regions are not overlapping.
template <Pixel_type Type>
struct Pixel_copier
{
    /// Typedef for the underlying base type
    typedef typename Pixel_type_traits<Type>::Base_type Base_type;

    /// Copies a single pixel.
    ///
    /// Not very useful by itself, just for completeness.
    ///
    /// \param source   The pixel to copy from pixel type \p Type.
    /// \param dest     The pixel to store the result in pixel type \p Type.
    static MI_HOST_DEVICE_INLINE void copy( const Base_type* source, Base_type* dest);

    /// Copies a contiguous region of pixels.
    ///
    /// \param source   The first pixel to copy from pixel type \p Type.
    /// \param dest     The first pixel to store the result in pixel type \p Type.
    /// \param count    The number of pixels to copy.
    static MI_HOST_DEVICE_INLINE void copy( const Base_type* source, Base_type* dest, mi::Size count);

    /// Copies a rectangular region of pixels with arbitrary row strides.
    ///
    /// \param source          The first pixel of the first row to copy from pixel type
    ///                        \p Type.
    /// \param dest            The first pixel of the first row to store the result in pixel type
    ///                        \p Type.
    /// \param width           The number of pixels in x direction to copy.
    /// \param height          The number of pixels in y direction to copy.
    /// \param source_stride   Offset between subsequent rows of \p source (in bytes). Can be
    ///                        negative in order to flip the row order, but then \p source must
    ///                        point to the first pixel of the last row.
    /// \param dest_stride     Offset between subsequent rows of \p dest (in bytes). Can be
    ///                        negative in order to flip the row order, but then \p dest must
    ///                        point to the first pixel of the last row.
    static MI_HOST_DEVICE_INLINE void copy(
        const Base_type* source, Base_type* dest,
        mi::Size width, mi::Size height,
        mi::Difference source_stride, mi::Difference dest_stride);

    /// Copies a contiguous region of pixels.
    ///
    /// This variant with "const void*" and "void*" arguments exists only to hide the casts in all
    /// the callers below.
    ///
    /// For documentation, see the corresponding method above.
    static MI_HOST_DEVICE_INLINE void copy( const void* source, void* dest, mi::Size count = 1);

    /// Copies a rectangular region of pixels with arbitrary row strides.
    ///
    /// This variant with "const void*" and "void*" arguments exists only to hide the casts in all
    /// the callers below.
    ///
    /// For documentation, see the corresponding method above.
    static MI_HOST_DEVICE_INLINE void copy(
        const void* source, void* dest,
        mi::Size width, mi::Size height,
        mi::Difference source_stride, mi::Difference dest_stride);
};

// ---------- implementation -----------------------------------------------------------------------

template <typename Src, typename Dest>
MI_HOST_DEVICE_INLINE void quantize_u(Dest& dest, const Src src)
{
#ifndef __CUDACC__
    using mi::base::max;
#endif
    dest = quantize_unsigned<Dest>(max(src,0.0f)); // min(x,1.f) is done inside of quantize_unsigned
}

template <typename Src, typename Dest>
MI_HOST_DEVICE_INLINE void quantize_s(Dest& dest, const Src src)
{
    dest = quantize_signed<Dest>(src);
}

MI_HOST_DEVICE_INLINE void to_rgbe(mi::Float32 src, mi::Uint8 dest[3])
{
    const mi::Float32 tmp[3] = { src, src, src};
    mi::math::to_rgbe(tmp, dest);
}

MI_HOST_DEVICE_INLINE void adjust_gamma(
    mi::Float32* const data, const mi::Size count, const mi::Uint32 components, const mi::Float32 exponent)
{
    const mi::Uint32 active_components = std::min(components, 3u);
    for( mi::Size i = 0; i < count * components; i += components) {
#ifdef __CUDACC__
#pragma unroll
#endif
        for ( mi::Uint32 c = 0; c < active_components; ++c) {
#ifdef __CUDACC__
            data[i+c] = powf( data[i+c], exponent);
#else
            data[i+c] = mi::math::fast_pow( data[i+c], exponent); //!!
#endif
        }
    }
}

MI_HOST_DEVICE_INLINE bool exists_pixel_conversion( const Pixel_type Source, const Pixel_type Dest)
{
    return Source != PT_UNDEF && Dest != PT_UNDEF;
}

MI_HOST_DEVICE_INLINE bool convert(
    const void* const source, void* const dest, Pixel_type Source, Pixel_type Dest, const mi::Size count)
{
    if( Source == PT_SINT32)    Source = PT_RGBA;
    if( Source == PT_FLOAT32_3) Source = PT_RGB_FP;
    if( Source == PT_FLOAT32_4) Source = PT_COLOR;

    if( Dest == PT_SINT32)      Dest = PT_RGBA;
    if( Dest == PT_FLOAT32_3)   Dest = PT_RGB_FP;
    if( Dest == PT_FLOAT32_4)   Dest = PT_COLOR;

    if( Source == Dest)
        return copy( source, dest, Source, count);

    switch( Source) {
        case PT_SINT8:     return convert<PT_SINT8>    ( source, dest, Dest, count);
        case PT_FLOAT32:   return convert<PT_FLOAT32>  ( source, dest, Dest, count);
        case PT_FLOAT32_2: return convert<PT_FLOAT32_2>( source, dest, Dest, count);
        case PT_RGB:       return convert<PT_RGB>      ( source, dest, Dest, count);
        case PT_RGBA:      return convert<PT_RGBA>     ( source, dest, Dest, count);
        case PT_RGBE:      return convert<PT_RGBE>     ( source, dest, Dest, count);
        case PT_RGBEA:     return convert<PT_RGBEA>    ( source, dest, Dest, count);
        case PT_RGB_16:    return convert<PT_RGB_16>   ( source, dest, Dest, count);
        case PT_RGBA_16:   return convert<PT_RGBA_16>  ( source, dest, Dest, count);
        case PT_RGB_FP:    return convert<PT_RGB_FP>   ( source, dest, Dest, count);
        case PT_COLOR:     return convert<PT_COLOR>    ( source, dest, Dest, count);
        default:           ASSERT( M_IMAGE, false); return false;
    }
}

MI_HOST_DEVICE_INLINE bool convert(
    const void* const source, void* const dest,
    Pixel_type Source, Pixel_type Dest,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    if( Source == PT_SINT32)    Source = PT_RGBA;
    if( Source == PT_FLOAT32_3) Source = PT_RGB_FP;
    if( Source == PT_FLOAT32_4) Source = PT_COLOR;

    if( Dest == PT_SINT32)      Dest = PT_RGBA;
    if( Dest == PT_FLOAT32_3)   Dest = PT_RGB_FP;
    if( Dest == PT_FLOAT32_4)   Dest = PT_COLOR;

    if( Source == Dest)
        return copy( source, dest, Source, width, height, source_stride, dest_stride);

#define MI_IMAGE_ARGS source, dest, Dest, width, height, source_stride, dest_stride
    switch( Source) {
        case PT_SINT8:     return convert<PT_SINT8>    ( MI_IMAGE_ARGS);
        case PT_FLOAT32:   return convert<PT_FLOAT32>  ( MI_IMAGE_ARGS);
        case PT_FLOAT32_2: return convert<PT_FLOAT32_2>( MI_IMAGE_ARGS);
        case PT_RGB:       return convert<PT_RGB>      ( MI_IMAGE_ARGS);
        case PT_RGBA:      return convert<PT_RGBA>     ( MI_IMAGE_ARGS);
        case PT_RGBE:      return convert<PT_RGBE>     ( MI_IMAGE_ARGS);
        case PT_RGBEA:     return convert<PT_RGBEA>    ( MI_IMAGE_ARGS);
        case PT_RGB_16:    return convert<PT_RGB_16>   ( MI_IMAGE_ARGS);
        case PT_RGBA_16:   return convert<PT_RGBA_16>  ( MI_IMAGE_ARGS);
        case PT_RGB_FP:    return convert<PT_RGB_FP>   ( MI_IMAGE_ARGS);
        case PT_COLOR:     return convert<PT_COLOR>    ( MI_IMAGE_ARGS);
        default:           ASSERT( M_IMAGE, false); return false;
    }
#undef MI_IMAGE_ARGS
}

template <Pixel_type Source>
MI_HOST_DEVICE_INLINE bool convert( const void* const source, void* const dest, Pixel_type Dest, const mi::Size count)
{
    if( Dest == PT_SINT32)      Dest = PT_RGBA;
    if( Dest == PT_FLOAT32_3)   Dest = PT_RGB_FP;
    if( Dest == PT_FLOAT32_4)   Dest = PT_COLOR;

#define MI_IMAGE_ARGS source, dest, count
    switch( Dest) {
    case PT_SINT8:     Pixel_converter<Source, PT_SINT8>    ::convert( MI_IMAGE_ARGS); return true;
    case PT_FLOAT32:   Pixel_converter<Source, PT_FLOAT32>  ::convert( MI_IMAGE_ARGS); return true;
    case PT_FLOAT32_2: Pixel_converter<Source, PT_FLOAT32_2>::convert( MI_IMAGE_ARGS); return true;
    case PT_RGB:       Pixel_converter<Source, PT_RGB>      ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBA:      Pixel_converter<Source, PT_RGBA>     ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBE:      Pixel_converter<Source, PT_RGBE>     ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBEA:     Pixel_converter<Source, PT_RGBEA>    ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGB_16:    Pixel_converter<Source, PT_RGB_16>   ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBA_16:   Pixel_converter<Source, PT_RGBA_16>  ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGB_FP:    Pixel_converter<Source, PT_RGB_FP>   ::convert( MI_IMAGE_ARGS); return true;
    case PT_COLOR:     Pixel_converter<Source, PT_COLOR>    ::convert( MI_IMAGE_ARGS); return true;
    default:           ASSERT( M_IMAGE, false); return false;
    }
#undef MI_IMAGE_ARGS
}

template <Pixel_type Source>
MI_HOST_DEVICE_INLINE bool convert(
    const void* const source, void* const dest,
    Pixel_type Dest,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    if( Dest == PT_SINT32)      Dest = PT_RGBA;
    if( Dest == PT_FLOAT32_3)   Dest = PT_RGB_FP;
    if( Dest == PT_FLOAT32_4)   Dest = PT_COLOR;

#define MI_IMAGE_ARGS source, dest, width, height, source_stride, dest_stride
    switch( Dest) {
    case PT_SINT8:     Pixel_converter<Source, PT_SINT8>    ::convert( MI_IMAGE_ARGS); return true;
    case PT_FLOAT32:   Pixel_converter<Source, PT_FLOAT32>  ::convert( MI_IMAGE_ARGS); return true;
    case PT_FLOAT32_2: Pixel_converter<Source, PT_FLOAT32_2>::convert( MI_IMAGE_ARGS); return true;
    case PT_RGB:       Pixel_converter<Source, PT_RGB>      ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBA:      Pixel_converter<Source, PT_RGBA>     ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBE:      Pixel_converter<Source, PT_RGBE>     ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBEA:     Pixel_converter<Source, PT_RGBEA>    ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGB_16:    Pixel_converter<Source, PT_RGB_16>   ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGBA_16:   Pixel_converter<Source, PT_RGBA_16>  ::convert( MI_IMAGE_ARGS); return true;
    case PT_RGB_FP:    Pixel_converter<Source, PT_RGB_FP>   ::convert( MI_IMAGE_ARGS); return true;
    case PT_COLOR:     Pixel_converter<Source, PT_COLOR>    ::convert( MI_IMAGE_ARGS); return true;
    default:           ASSERT( M_IMAGE, false); return false;
    }
#undef MI_IMAGE_ARGS
}

MI_HOST_DEVICE_INLINE bool copy(
    const void* const source, void* const dest, Pixel_type Type, const mi::Size count)
{
    if( Type == PT_SINT32)      Type = PT_RGBA;
    if( Type == PT_FLOAT32_3)   Type = PT_RGB_FP;
    if( Type == PT_FLOAT32_4)   Type = PT_COLOR;

    switch( Type) {
        case PT_SINT8:     Pixel_copier<PT_SINT8>    ::copy( source, dest, count); return true;
        case PT_FLOAT32:   Pixel_copier<PT_FLOAT32>  ::copy( source, dest, count); return true;
        case PT_FLOAT32_2: Pixel_copier<PT_FLOAT32_2>::copy( source, dest, count); return true;
        case PT_RGB:       Pixel_copier<PT_RGB>      ::copy( source, dest, count); return true;
        case PT_RGBA:      Pixel_copier<PT_RGBA>     ::copy( source, dest, count); return true;
        case PT_RGBE:      Pixel_copier<PT_RGBE>     ::copy( source, dest, count); return true;
        case PT_RGBEA:     Pixel_copier<PT_RGBEA>    ::copy( source, dest, count); return true;
        case PT_RGB_16:    Pixel_copier<PT_RGB_16>   ::copy( source, dest, count); return true;
        case PT_RGBA_16:   Pixel_copier<PT_RGBA_16>  ::copy( source, dest, count); return true;
        case PT_RGB_FP:    Pixel_copier<PT_RGB_FP>   ::copy( source, dest, count); return true;
        case PT_COLOR:     Pixel_copier<PT_COLOR>    ::copy( source, dest, count); return true;
        default:           ASSERT( M_IMAGE, false); return false;
    }
}

MI_HOST_DEVICE_INLINE bool copy(
    const void* const source, void* const dest,
    Pixel_type Type,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    if( Type == PT_SINT32)      Type = PT_RGBA;
    if( Type == PT_FLOAT32_3)   Type = PT_RGB_FP;
    if( Type == PT_FLOAT32_4)   Type = PT_COLOR;

#define MI_IMAGE_ARGS source, dest, width, height, source_stride, dest_stride
    switch( Type) {
        case PT_SINT8:     Pixel_copier<PT_SINT8>    ::copy( MI_IMAGE_ARGS); return true;
        case PT_FLOAT32:   Pixel_copier<PT_FLOAT32>  ::copy( MI_IMAGE_ARGS); return true;
        case PT_FLOAT32_2: Pixel_copier<PT_FLOAT32_2>::copy( MI_IMAGE_ARGS); return true;
        case PT_RGB:       Pixel_copier<PT_RGB>      ::copy( MI_IMAGE_ARGS); return true;
        case PT_RGBA:      Pixel_copier<PT_RGBA>     ::copy( MI_IMAGE_ARGS); return true;
        case PT_RGBE:      Pixel_copier<PT_RGBE>     ::copy( MI_IMAGE_ARGS); return true;
        case PT_RGBEA:     Pixel_copier<PT_RGBEA>    ::copy( MI_IMAGE_ARGS); return true;
        case PT_RGB_16:    Pixel_copier<PT_RGB_16>   ::copy( MI_IMAGE_ARGS); return true;
        case PT_RGBA_16:   Pixel_copier<PT_RGBA_16>  ::copy( MI_IMAGE_ARGS); return true;
        case PT_RGB_FP:    Pixel_copier<PT_RGB_FP>   ::copy( MI_IMAGE_ARGS); return true;
        case PT_COLOR:     Pixel_copier<PT_COLOR>    ::copy( MI_IMAGE_ARGS); return true;
        default:           ASSERT( M_IMAGE, false); return false;
    }
#undef MI_IMAGE_ARGS
}

template <Pixel_type Source, Pixel_type Dest>
MI_HOST_DEVICE_INLINE void Pixel_converter<Source,Dest>::convert(
    const Source_base_type* const source, Dest_base_type* const dest)
{
#ifndef __CUDACC__
    // Use exists_pixel_conversion( Source, Dest) to find out whether the conversion from
    // Source to Dest is supported.
    ASSERT( M_IMAGE, !"pixel type conversion not implemented for this combination");
#endif
}

template <Pixel_type Source, Pixel_type Dest>
MI_HOST_DEVICE_INLINE void Pixel_converter<Source,Dest>::convert(
    const Source_base_type* source, Dest_base_type* dest, const mi::Size count)
{
    for( mi::Size i = 0; i < count; ++i) {
        convert( source, dest);
        source += Pixel_type_traits<Source>::s_components_per_pixel;
        dest   += Pixel_type_traits<Dest>::s_components_per_pixel;
    }
}

template <Pixel_type Source, Pixel_type Dest>
MI_HOST_DEVICE_INLINE void Pixel_converter<Source,Dest>::convert(
    const Source_base_type* const source, Dest_base_type* const dest,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    // use source2 and dest2 instead of source and dest for easier pointer arithmetic
    const char* source2 = reinterpret_cast<const char*>( source);
    char* dest2         = reinterpret_cast<char*>(       dest);

    for( mi::Size y = 0; y < height; ++y) {

        // use source3 and dest3 instead of source2 and dest2 inside a row
        const Source_base_type* const source3 = reinterpret_cast<const Source_base_type*>( source2);
        Dest_base_type*         const dest3   = reinterpret_cast<Dest_base_type*>(         dest2);

        // call 3-args variant instead of explicit loop to benefit from its SSE specializations
        convert( source3, dest3, width);

        source2 += source_stride;
        dest2   += dest_stride;
    }
}

template <Pixel_type Source, Pixel_type Dest>
MI_HOST_DEVICE_INLINE void Pixel_converter<Source,Dest>::convert( const void* const source, void* const dest, const mi::Size count)
{
    const Source_base_type* const source2 = static_cast<const Source_base_type*>( source);
    Dest_base_type* const dest2           = static_cast<Dest_base_type*>( dest);
    convert( source2, dest2, count);
}

template <Pixel_type Source, Pixel_type Dest>
MI_HOST_DEVICE_INLINE void Pixel_converter<Source,Dest>::convert(
    const void* const source, void* const dest,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    const Source_base_type* const source2 = static_cast<const Source_base_type*>( source);
    Dest_base_type* const dest2           = static_cast<Dest_base_type*>( dest);
    convert( source2, dest2, width, height, source_stride, dest_stride);
}

template <Pixel_type Type>
MI_HOST_DEVICE_INLINE void Pixel_copier<Type>::copy( const Base_type* const source, Base_type* const dest)
{
    const mi::Size bytes_per_pixel = Pixel_type_traits<Type>::s_components_per_pixel
                                   * sizeof( typename Pixel_type_traits<Type>::Base_type);
    memcpy( dest, source, bytes_per_pixel);
}

template <Pixel_type Type>
MI_HOST_DEVICE_INLINE void Pixel_copier<Type>::copy( const Base_type* const source, Base_type* const dest, const mi::Size count)
{
    const mi::Size bytes_per_pixel = Pixel_type_traits<Type>::s_components_per_pixel
                                   * sizeof( typename Pixel_type_traits<Type>::Base_type);
    memcpy( dest, source, count * bytes_per_pixel);
}

template <Pixel_type Type>
MI_HOST_DEVICE_INLINE void Pixel_copier<Type>::copy(
    const Base_type* const source, Base_type* const dest,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    // use source2 and dest2 instead of source and dest for easier pointer arithmetic
    const char* source2 = reinterpret_cast<const char*>( source);
    char* dest2         = reinterpret_cast<char*>( dest);

    const mi::Size bytes_per_row = width * Pixel_type_traits<Type>::s_components_per_pixel
                                         * sizeof( typename Pixel_type_traits<Type>::Base_type);

    // check if the rectangle is contiguous and allows to use a single memcpy() call
    if(    source_stride > 0 && static_cast<mi::Size>( source_stride) == bytes_per_row
        && dest_stride   > 0 && static_cast<mi::Size>( dest_stride  ) == bytes_per_row) {
        memcpy( dest2, source2, height * bytes_per_row);
        return;
    }
    if(    source_stride < 0 && static_cast<mi::Size>( -source_stride) == bytes_per_row
        && dest_stride   < 0 && static_cast<mi::Size>( -dest_stride  ) == bytes_per_row) {
        source2 -= (height - 1) * bytes_per_row;
        dest2   -= (height - 1) * bytes_per_row;
        memcpy( dest2, source2, height * bytes_per_row);
        return;
    }

    // rectangle is not contiguous, use one memcpy() per row
    for( mi::Size y = 0; y < height; ++y) {
        memcpy( dest2, source2, bytes_per_row);
        source2 += source_stride;
        dest2   += dest_stride;
    }
}

template <Pixel_type Type>
MI_HOST_DEVICE_INLINE void Pixel_copier<Type>::copy( const void* const source, void* dest, const mi::Size count)
{
    const Base_type* const source2 = static_cast<const Base_type*>( source);
    Base_type* const dest2         = static_cast<Base_type*>( dest);
    copy( source2, dest2, count);
}

template <Pixel_type Type>
MI_HOST_DEVICE_INLINE void Pixel_copier<Type>::copy(
    const void* const source, void* const dest,
    const mi::Size width, const mi::Size height,
    const mi::Difference source_stride, const mi::Difference dest_stride)
{
    const Base_type* const source2 = static_cast<const Base_type*>( source);
    Base_type* const dest2         = static_cast<Base_type*>( dest);
    copy( source2, dest2, width, height, source_stride, dest_stride);
}

// ---------- specializations of Pixel_converter<Source,Dest>::convert() ---------------------------

// ---------- source PT_SINT8 ----------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = src[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = src[0];
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    to_rgbe( src[0] * mi::Float32( 1.0/255.0), dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    to_rgbe( src[0] * mi::Float32( 1.0/255.0), dest);
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = mi::Uint16( mi::Float32( src[0]) * mi::Float32( 65535.0/255.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = mi::Uint16( mi::Float32( src[0]) * mi::Float32( 65535.0/255.0));
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_SINT8, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[3] = 1.0f;
}

// ---------- source PT_FLOAT32 --------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(*dest,*src); // actually SINT8 means UINT8 :/
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = src[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(*dest,*src);
    dest[1] = dest[2] = dest[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(*dest,*src);
    dest[1] = dest[2] = dest[0];
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = { src[0], src[0], src[0] };
    mi::math::to_rgbe( tmp, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = { src[0], src[0], src[0] };
    mi::math::to_rgbe( tmp, dest);
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(*dest,*src);
    dest[1] = dest[2] = dest[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(*dest,*src);
    dest[1] = dest[2] = dest[0];
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = src[0];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = dest[1] = dest[2] = src[0];
    dest[3] = 1.0f;
}

// ---------- source PT_FLOAT32_2 ------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(*dest,(src[0]+src[1])*0.5f); // actually SINT8 means UINT8 :/
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (src[0]+src[1]) * 0.5f;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    dest[2] = 0;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    dest[2] = 0;
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = { src[0], src[1], 0.0f };
    mi::math::to_rgbe( tmp, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = { src[0], src[1], 0.0f };
    mi::math::to_rgbe( tmp, dest);
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    dest[2] = 0;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    dest[2] = 0;
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = 0.0f;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_FLOAT32_2, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = 0.0f;
    dest[3] = 1.0f;
}

// ---------- source PT_RGB ------------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint8( 0.27f * mi::Float32( src[0])
                       + 0.67f * mi::Float32( src[1])
                       + 0.06f * mi::Float32( src[2]));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( 0.27/255.0) * mi::Float32( src[0])
            + mi::Float32( 0.67/255.0) * mi::Float32( src[1])
            + mi::Float32( 0.06/255.0) * mi::Float32( src[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/255.0)};
    mi::math::to_rgbe( tmp, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/255.0)};
    mi::math::to_rgbe( tmp, dest);
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint16( mi::Float32( src[0]) * mi::Float32( 65535.0/255.0));
    dest[1] = mi::Uint16( mi::Float32( src[1]) * mi::Float32( 65535.0/255.0));
    dest[2] = mi::Uint16( mi::Float32( src[2]) * mi::Float32( 65535.0/255.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint16( mi::Float32( src[0]) * mi::Float32( 65535.0/255.0));
    dest[1] = mi::Uint16( mi::Float32( src[1]) * mi::Float32( 65535.0/255.0));
    dest[2] = mi::Uint16( mi::Float32( src[2]) * mi::Float32( 65535.0/255.0));
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/255.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/255.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/255.0);
    dest[3] = 1.0f;
}

// ---------- source PT_RGBA -----------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint8( 0.27f * mi::Float32( src[0])
                       + 0.67f * mi::Float32( src[1])
                       + 0.06f * mi::Float32( src[2]));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( 0.27/255.0) * mi::Float32( src[0])
            + mi::Float32( 0.67/255.0) * mi::Float32( src[1])
            + mi::Float32( 0.06/255.0) * mi::Float32( src[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/255.0)};
    mi::math::to_rgbe( tmp, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/255.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/255.0)};
    mi::math::to_rgbe( tmp, dest);
    dest[4] = src[3];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint16( mi::Float32( src[0]) * mi::Float32( 65535.0/255.0));
    dest[1] = mi::Uint16( mi::Float32( src[1]) * mi::Float32( 65535.0/255.0));
    dest[2] = mi::Uint16( mi::Float32( src[2]) * mi::Float32( 65535.0/255.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint16( mi::Float32( src[0]) * mi::Float32( 65535.0/255.0));
    dest[1] = mi::Uint16( mi::Float32( src[1]) * mi::Float32( 65535.0/255.0));
    dest[2] = mi::Uint16( mi::Float32( src[2]) * mi::Float32( 65535.0/255.0));
    dest[3] = mi::Uint16( mi::Float32( src[3]) * mi::Float32( 65535.0/255.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/255.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/255.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/255.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/255.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/255.0);
    dest[3] = mi::Float32( src[3]) * mi::Float32( 1.0/255.0);
}

// ---------- source PT_RGBE -----------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    const mi::Float32 value = 0.27f * tmp[0] + 0.67f * tmp[1] + 0.06f * tmp[2];
    quantize_u(*dest,value); // actually SINT8 means UINT8 :/
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    dest[0] = 0.27f * tmp[0] + 0.67f * tmp[1] + 0.06f * tmp[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    dest[0] = tmp[0];
    dest[1] = tmp[1];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::from_rgbe( src, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBE, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::from_rgbe( src, dest);
    dest[3] = 1.0f;
}

// ---------- source PT_RGBEA ----------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    const mi::Float32 value = 0.27f * tmp[0] + 0.67f * tmp[1] + 0.06f * tmp[2];
    quantize_u(*dest,value); // actually SINT8 means UINT8 :/
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    dest[0] = 0.27f * tmp[0] + 0.67f * tmp[1] + 0.06f * tmp[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    dest[0] = tmp[0];
    dest[1] = tmp[1];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0], tmp[0]);
    quantize_u(dest[1], tmp[1]);
    quantize_u(dest[2], tmp[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
    dest[3] = src[4];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
    dest[4] = src[4];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::Float32 tmp[3];
    mi::math::from_rgbe( src, tmp);
    quantize_u(dest[0],tmp[0]);
    quantize_u(dest[1],tmp[1]);
    quantize_u(dest[2],tmp[2]);
    dest[3] = mi::Uint16( src[4] * mi::Float32( 65535.0/255.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::from_rgbe( src, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBEA, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::from_rgbe( src, dest);
    dest[3] = mi::Float32( src[4]) * mi::Float32( 1.0/255.0);
}

// ---------- source PT_RGB_16 ---------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint8( (0.27f * mi::Float32( src[0])
                        + 0.67f * mi::Float32( src[1])
                        + 0.06f * mi::Float32( src[2])) * mi::Float32( 255.0/65535.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (0.27f * mi::Float32( src[0])
             + 0.67f * mi::Float32( src[1])
             + 0.06f * mi::Float32( src[2])) * mi::Float32( 1.0/65535.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/65535.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/65535.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (src[0] >> 8);
    dest[1] = (src[1] >> 8);
    dest[2] = (src[2] >> 8);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (src[0] >> 8);
    dest[1] = (src[1] >> 8);
    dest[2] = (src[2] >> 8);
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/65535.0)};
    mi::math::to_rgbe( tmp, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/65535.0)};
    mi::math::to_rgbe( tmp, dest);
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/65535.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/65535.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/65535.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_16, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/65535.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/65535.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/65535.0);
    dest[3] = 1.0f;
}

// ---------- source PT_RGBA_16 --------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Uint8( (0.27f * mi::Float32( src[0])
                        + 0.67f * mi::Float32( src[1])
                        + 0.06f * mi::Float32( src[2])) * mi::Float32( 255.0/65535.0));
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (0.27f * mi::Float32( src[0])
             + 0.67f * mi::Float32( src[1])
             + 0.06f * mi::Float32( src[2])) * mi::Float32( 1.0/65535.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/65535.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/65535.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (src[0] >> 8);
    dest[1] = (src[1] >> 8);
    dest[2] = (src[2] >> 8);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = (src[0] >> 8);
    dest[1] = (src[1] >> 8);
    dest[2] = (src[2] >> 8);
    dest[3] = (src[3] >> 8);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/65535.0)};
    mi::math::to_rgbe( tmp, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 tmp[3] = {
        mi::Float32( src[0]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[1]) * mi::Float32( 1.0/65535.0),
        mi::Float32( src[2]) * mi::Float32( 1.0/65535.0)};
    mi::math::to_rgbe( tmp, dest);
    dest[4] = (src[3] >> 8);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/65535.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/65535.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/65535.0);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGBA_16, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = mi::Float32( src[0]) * mi::Float32( 1.0/65535.0);
    dest[1] = mi::Float32( src[1]) * mi::Float32( 1.0/65535.0);
    dest[2] = mi::Float32( src[2]) * mi::Float32( 1.0/65535.0);
    dest[3] = mi::Float32( src[3]) * mi::Float32( 1.0/65535.0);
}

// ---------- source PT_RGB_FP ---------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 value = 0.27f * src[0] + 0.67f * src[1] + 0.06f * src[2];
    quantize_u(*dest,value); // actually SINT8 means UINT8 :/
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = 0.27f * src[0] + 0.67f * src[1] + 0.06f * src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
    dest[3] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::to_rgbe( src, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::to_rgbe( src, dest);
    dest[4] = 255;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
    dest[3] = 65535;
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = 1.0f;
}

// ---------- source PT_COLOR ----------------------------------------------------------------------

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_SINT8>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    const mi::Float32 value = 0.27f * src[0] + 0.67f * src[1] + 0.06f * src[2];
    quantize_u(*dest,value); // actually SINT8 means UINT8 :/
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_FLOAT32>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = 0.27f * src[0] + 0.67f * src[1] + 0.06f * src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_FLOAT32_2>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGB>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGBA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
    quantize_u(dest[3],src[3]);

}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGBE>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::to_rgbe( src, dest);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGBEA>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    mi::math::to_rgbe( src, dest);
    quantize_u(dest[4],src[3]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGB_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGBA_16>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    quantize_u(dest[0],src[0]);
    quantize_u(dest[1],src[1]);
    quantize_u(dest[2],src[2]);
    quantize_u(dest[3],src[3]);
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_RGB_FP>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

template<> MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR, PT_COLOR>::convert(
    const Source_base_type* const src, Dest_base_type* const dest)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
}

#ifndef __CUDACC__
#if defined(HAS_SSE) || defined(SSE_INTRINSICS)
MI_FORCE_INLINE __m128i quantize_unsigned_sse(const float* const source)
{
    __m128 fp0 = _mm_loadu_ps(source);         // 4 floats (RGBA or RGBR, GBRG, BRGB)
    fp0 = _mm_mul_ps(_mm_min_ps(fp0, _mm_set1_ps(MI::STLEXT::binary_cast<float>(0x3f800000u-1))),_mm_set1_ps(256.0f)); // see quantize_unsigned(), need to mul by 256 and clamp instead of 255
    return _mm_cvttps_epi32(fp0);
}

MI_FORCE_INLINE void quantize_unsigned_sse(__m128i* const dest, const float* const source)
{
    __m128i i0 = quantize_unsigned_sse(source);
    __m128i i1 = quantize_unsigned_sse(source+4);

    i0 = _mm_packs_epi32( i0,  i1);         // 8 shorts

    const __m128i i2 = quantize_unsigned_sse(source+8);
    const __m128i i3 = quantize_unsigned_sse(source+12);

    i1 = _mm_packs_epi32( i2,  i3);         // 8 shorts

    _mm_storeu_si128( dest, _mm_packus_epi16( i0, i1)); // 16 uchars
}
#endif


// ---------- source PT_RGB_FP, target PT_RGB ------------------------------------------------------

template <>
MI_HOST_DEVICE_INLINE void Pixel_converter<PT_RGB_FP,PT_RGB>::convert(
    const Source_base_type* const source, Dest_base_type* const dest, const mi::Size count)
{
    // use source2 and dest2 instead of source and dest for easier pointer arithmetic
    const float* source2 = reinterpret_cast<const float*>( source);
    char* dest2          = reinterpret_cast<char*>(        dest);

    mi::Size i = 0;

#if defined(HAS_SSE) || defined(SSE_INTRINSICS)
    const mi::Size w16_3 = count/16*3;

    // convert 16 components in each iteration, i.e., 16/3 pixels
    // _mm_packs_epi32() and _mm_packus_epi16() saturate the result
    for( ; i < w16_3; ++i) {

        quantize_unsigned_sse((__m128i*)dest2,source2);

        source2 += 16;
        dest2   += 16;
    }

    i = count/16*16; // for falling through to the tail handling/non-SSE case below
#endif

    // use source3 and dest3 instead of source2 and dest2 inside a row
    const Source_base_type* source3 = reinterpret_cast<const Source_base_type*>( source2);
    Dest_base_type*         dest3   = reinterpret_cast<Dest_base_type*>(         dest2);

    for( ; i < count; ++i) {
        convert( source3, dest3);
        source3 += Pixel_type_traits<PT_RGB_FP>::s_components_per_pixel;
        dest3   += Pixel_type_traits<PT_RGB>::s_components_per_pixel;
    }
}

// ---------- source PT_RGBA_FP, target PT_COLOR --------------------------------------------------

template <>
MI_HOST_DEVICE_INLINE void Pixel_converter<PT_COLOR,PT_RGBA>::convert(
    const Source_base_type* const source, Dest_base_type* const dest, const mi::Size count)
{
    const float* source2 = reinterpret_cast<const float*>( source);
    char* dest2          = reinterpret_cast<char*>(        dest);

    mi::Size i = 0;

#if defined(HAS_SSE) || defined(SSE_INTRINSICS)
    const mi::Size w4 = count/4;

    // convert 16 components in each iteration, i.e., 16/4 = 4 pixels
    // _mm_packs_epi32() and _mm_packus_epi16() saturate the result
    for( ; i < w4; ++i) {

        quantize_unsigned_sse((__m128i*)dest2,source2);

        source2 += 16;
        dest2   += 16;
    }

    i = count/4*4; // for falling through to the tail handling/non-SSE case below
#endif

    const Source_base_type* source3 = reinterpret_cast<const Source_base_type*>( source2);
    Dest_base_type*         dest3   = reinterpret_cast<Dest_base_type*>(         dest2);

    for( ; i < count; ++i) {
        convert( source3, dest3);
        source3 += Pixel_type_traits<PT_COLOR>::s_components_per_pixel;
        dest3   += Pixel_type_traits<PT_RGBA>::s_components_per_pixel;
    }
}

#endif

} // namespace IMAGE

} // namespace MI

#endif // IO_IMAGE_IMAGE_IMAGE_PIXEL_CONVERSION_H
