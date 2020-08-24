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

#include "image_tile_impl.h"

#include <mi/math/color.h>
#include <mi/math/function.h>

#include <io/image/image/i_image_quantization.h>

#include <base/lib/log/i_log_logger.h>


namespace MI {

namespace IMAGE {

template <Pixel_type T>
Tile_impl<T>::Tile_impl( mi::Uint32 width, mi::Uint32 height)
{
    // check incorrect arguments
    ASSERT( M_IMAGE, width > 0 && height > 0);

    m_width = width;
    m_height = height;
    typedef typename Pixel_type_traits<T>::Base_type Base_type;
    m_data.resize(static_cast<mi::Size>( m_width) * m_height * s_components_per_pixel);
}

template <Pixel_type T>
const char* Tile_impl<T>::get_type() const
{
    return convert_pixel_type_enum_to_string( T);
}

template <Pixel_type T>
mi::Size Tile_impl<T>::get_size() const
{
    typedef typename Pixel_type_traits<T>::Base_type Base_type;
    return sizeof( *this)
        + static_cast<mi::Size>( m_width) * m_height * s_components_per_pixel * sizeof( Base_type);
}

// ---------- PT_SINT8 -----------------------------------------------------------------------------

template <>
void Tile_impl<PT_SINT8>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    mi::Float32 value = 0.27f * floats[0] + 0.67f * floats[1] + 0.06f * floats[2];
    position[0] = mi::Uint8( mi::math::clamp( value, 0.0f, 1.0f) * 255.0f);
}

template <>
void Tile_impl<PT_SINT8>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    mi::Float32 value = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[0] = floats[1] = floats[2] = value;
    floats[3] = 1.0f;
}

// ---------- PT_SINT32 ----------------------------------------------------------------------------

template <>
void Tile_impl<PT_SINT32>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    // treat PT_SINT32 as PT_RGBA
    mi::Uint8* const position = reinterpret_cast<mi::Uint8*>(m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel);
    position[0] = IMAGE::quantize_unsigned<mi::Uint8>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint8>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint8>(floats[2]);
    position[3] = IMAGE::quantize_unsigned<mi::Uint8>(floats[3]);
}

template <>
void Tile_impl<PT_SINT32>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    // treat PT_SINT32 as PT_RGBA
    const mi::Uint8* const position = reinterpret_cast<const mi::Uint8*>(m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel);
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/255.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/255.0);
    floats[3] = mi::Float32( position[3]) * mi::Float32( 1.0/255.0);
}

// ---------- PT_FLOAT32 ---------------------------------------------------------------------------

template <>
void Tile_impl<PT_FLOAT32>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = floats[0];
}

template <>
void Tile_impl<PT_FLOAT32>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = floats[1] = floats[2] = position[0];
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32_2 -------------------------------------------------------------------------

template <>
void Tile_impl<PT_FLOAT32_2>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = floats[0];
    position[1] = floats[1];
}

template <>
void Tile_impl<PT_FLOAT32_2>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = 0.0f;
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32_3 -------------------------------------------------------------------------

template <>
void Tile_impl<PT_FLOAT32_3>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
}

template <>
void Tile_impl<PT_FLOAT32_3>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32_4 -------------------------------------------------------------------------

template <>
void Tile_impl<PT_FLOAT32_4>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
    position[3] = floats[3];
}

template <>
void Tile_impl<PT_FLOAT32_4>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = position[3];
}

// ---------- PT_RGB -------------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGB>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = IMAGE::quantize_unsigned<mi::Uint8>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint8>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint8>(floats[2]);
}

template <>
void Tile_impl<PT_RGB>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/255.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/255.0);
    floats[3] = 1.0f;
}

// ---------- PT_RGBA ------------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGBA>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = IMAGE::quantize_unsigned<mi::Uint8>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint8>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint8>(floats[2]);
    position[3] = IMAGE::quantize_unsigned<mi::Uint8>(floats[3]);
}

template <>
void Tile_impl<PT_RGBA>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/255.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/255.0);
    floats[3] = mi::Float32( position[3]) * mi::Float32( 1.0/255.0);
}

// ---------- PT_RGBE ------------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGBE>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    mi::math::to_rgbe( floats, position);
}

template <>
void Tile_impl<PT_RGBE>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    mi::math::from_rgbe( position, floats);
    floats[3] = 1.0f;
}

// ---------- PT_RGBEA -----------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGBEA>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    mi::math::to_rgbe( floats, position);
    position[4] = IMAGE::quantize_unsigned<mi::Uint8>(floats[3]);
}

template <>
void Tile_impl<PT_RGBEA>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    mi::math::from_rgbe( position, floats);
    floats[3] = mi::Float32( position[4]) * mi::Float32( 1.0/255.0);
}

// ---------- PT_RGB_16 ----------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGB_16>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint16* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = IMAGE::quantize_unsigned<mi::Uint16>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint16>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint16>(floats[2]);
}

template <>
void Tile_impl<PT_RGB_16>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint16* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/65535.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/65535.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/65535.0);
    floats[3] = 1.0f;
}

// ---------- PT_RGBA_16 ---------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGBA_16>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint16* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = IMAGE::quantize_unsigned<mi::Uint16>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint16>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint16>(floats[2]);
    position[3] = IMAGE::quantize_unsigned<mi::Uint16>(floats[3]);
}

template <>
void Tile_impl<PT_RGBA_16>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint16* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/65535.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/65535.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/65535.0);
    floats[3] = mi::Float32( position[3]) * mi::Float32( 1.0/65535.0);
}

// ---------- PT_RGB_FP ----------------------------------------------------------------------------

template <>
void Tile_impl<PT_RGB_FP>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
}

template <>
void Tile_impl<PT_RGB_FP>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = 1.0f;
}

// ---------- PT_COLOR -----------------------------------------------------------------------------

template <>
void Tile_impl<PT_COLOR>::set_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
    position[3] = floats[3];
}

template <>
void Tile_impl<PT_COLOR>::get_pixel(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = m_data.data()
            + (x_offset + y_offset * static_cast<mi::Size>( m_width)) * s_components_per_pixel;
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = position[3];
}

// explicit template instantiation for Tile_impl<T>
template class Tile_impl<PT_SINT8>;
template class Tile_impl<PT_SINT32>;
template class Tile_impl<PT_FLOAT32>;
template class Tile_impl<PT_FLOAT32_2>;
template class Tile_impl<PT_FLOAT32_3>;
template class Tile_impl<PT_FLOAT32_4>;
template class Tile_impl<PT_RGB>;
template class Tile_impl<PT_RGBA>;
template class Tile_impl<PT_RGBE>;
template class Tile_impl<PT_RGBEA>;
template class Tile_impl<PT_RGB_16>;
template class Tile_impl<PT_RGBA_16>;
template class Tile_impl<PT_RGB_FP>;
template class Tile_impl<PT_COLOR>;

mi::neuraylib::ITile* create_tile( Pixel_type pixel_type, mi::Uint32 width, mi::Uint32 height)
{
    switch( pixel_type) {
        case PT_UNDEF:     ASSERT( M_IMAGE, false); return 0;
        case PT_SINT8:     return new Tile_impl<PT_SINT8    >( width, height);
        case PT_SINT32:    return new Tile_impl<PT_SINT32   >( width, height);
        case PT_FLOAT32:   return new Tile_impl<PT_FLOAT32  >( width, height);
        case PT_FLOAT32_2: return new Tile_impl<PT_FLOAT32_2>( width, height);
        case PT_FLOAT32_3: return new Tile_impl<PT_FLOAT32_3>( width, height);
        case PT_FLOAT32_4: return new Tile_impl<PT_FLOAT32_4>( width, height);
        case PT_RGB:       return new Tile_impl<PT_RGB      >( width, height);
        case PT_RGBA:      return new Tile_impl<PT_RGBA     >( width, height);
        case PT_RGBE:      return new Tile_impl<PT_RGBE     >( width, height);
        case PT_RGBEA:     return new Tile_impl<PT_RGBEA    >( width, height);
        case PT_RGB_16:    return new Tile_impl<PT_RGB_16   >( width, height);
        case PT_RGBA_16:   return new Tile_impl<PT_RGBA_16  >( width, height);
        case PT_RGB_FP:    return new Tile_impl<PT_RGB_FP   >( width, height);
        case PT_COLOR:     return new Tile_impl<PT_COLOR    >( width, height);
        default:           ASSERT( M_IMAGE, false); return 0;
    }
}

} // namespace IMAGE

} // namespace MI
