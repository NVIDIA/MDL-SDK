/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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


Tile_impl::Tile_impl( mi::Uint32 width, mi::Uint32 height, Pixel_type pixel_type)
  : m_width( width)
  , m_height( height)
  , m_type( pixel_type)
  , m_data( static_cast<mi::Size>( width) * height * get_bytes_per_pixel( pixel_type))
{
    // check incorrect arguments
    ASSERT( M_IMAGE, width > 0 && height > 0);
}

Tile_impl::Tile_impl( const mi::neuraylib::ITile* other)
  : m_width( other->get_resolution_x())
  , m_height( other->get_resolution_y())
  , m_type( IMAGE::convert_pixel_type_string_to_enum( other->get_type()))
  , m_data( reinterpret_cast<const char*>( other->get_data()),
            reinterpret_cast<const char*>( other->get_data())
                + static_cast<mi::Size>( other->get_resolution_x()) * other->get_resolution_y()
                    * get_bytes_per_pixel( m_type))
{
    // check incorrect arguments
    ASSERT( M_IMAGE, m_width > 0 && m_height > 0);
}

const char* Tile_impl::get_type() const
{
    return convert_pixel_type_enum_to_string( m_type);
}

mi::Size Tile_impl::get_size() const
{
    return sizeof( *this)
        + m_data.size() * sizeof( char);
}

void Tile_impl::reset( mi::Uint32 width, mi::Uint32 height, Pixel_type pixel_type)
{
    m_data.resize( static_cast<mi::Size>( width) * height * get_bytes_per_pixel( pixel_type));
    m_width = width;
    m_height = height;
    m_type = pixel_type;

    // check incorrect arguments
    ASSERT( M_IMAGE, m_width > 0 && m_height > 0);
}

template <Pixel_type PT>
inline auto* Tile_impl::get_position( mi::Uint32 x_offset, mi::Uint32 y_offset)
{
    using Ptt = Pixel_type_traits<PT>;
    return reinterpret_cast<typename Ptt::Base_type*>(m_data.data())
            + (x_offset + y_offset * std::size_t{m_width}) * Ptt::s_components_per_pixel;
}

template <Pixel_type PT>
inline auto* Tile_impl::get_position( mi::Uint32 x_offset, mi::Uint32 y_offset) const
{
    using Ptt = Pixel_type_traits<PT>;
    return reinterpret_cast<const typename Ptt::Base_type*>(m_data.data())
            + (x_offset + y_offset * std::size_t{m_width}) * Ptt::s_components_per_pixel;
}

// ---------- PT_SINT8 -----------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_SINT8>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = get_position<PT_SINT8>( x_offset, y_offset);
    const mi::Float32 value = 0.27f * floats[0] + 0.67f * floats[1] + 0.06f * floats[2];
    position[0] = mi::Uint8( mi::math::clamp( value, 0.0f, 1.0f) * 255.0f);
}

template <>
void Tile_impl::get_pixel<PT_SINT8>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = get_position<PT_SINT8>( x_offset, y_offset);
    const mi::Float32 value = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[0] = floats[1] = floats[2] = value;
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32 ---------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_FLOAT32>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = get_position<PT_FLOAT32>( x_offset, y_offset);
    position[0] = floats[0];
}

template <>
void Tile_impl::get_pixel<PT_FLOAT32>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = get_position<PT_FLOAT32>( x_offset, y_offset);
    floats[0] = floats[1] = floats[2] = position[0];
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32_2 -------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_FLOAT32_2>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = get_position<PT_FLOAT32_2>( x_offset, y_offset);
    position[0] = floats[0];
    position[1] = floats[1];
}

template <>
void Tile_impl::get_pixel<PT_FLOAT32_2>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = get_position<PT_FLOAT32_2>( x_offset, y_offset);
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = 0.0f;
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32_3 -------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_FLOAT32_3>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = get_position<PT_FLOAT32_3>( x_offset, y_offset);
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
}

template <>
void Tile_impl::get_pixel<PT_FLOAT32_3>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = get_position<PT_FLOAT32_3>( x_offset, y_offset);
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = 1.0f;
}

// ---------- PT_FLOAT32_4 -------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_FLOAT32_4>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = get_position<PT_FLOAT32_4>( x_offset, y_offset);
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
    position[3] = floats[3];
}

template <>
void Tile_impl::get_pixel<PT_FLOAT32_4>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = get_position<PT_FLOAT32_4>( x_offset, y_offset);
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = position[3];
}

// ---------- PT_RGB -------------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGB>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = get_position<PT_RGB>( x_offset, y_offset);
    position[0] = IMAGE::quantize_unsigned<mi::Uint8>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint8>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint8>(floats[2]);
}

template <>
void Tile_impl::get_pixel<PT_RGB>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = get_position<PT_RGB>( x_offset, y_offset);
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/255.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/255.0);
    floats[3] = 1.0f;
}

// ---------- PT_RGBA ------------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGBA>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = get_position<PT_RGBA>( x_offset, y_offset);
    position[0] = IMAGE::quantize_unsigned<mi::Uint8>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint8>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint8>(floats[2]);
    position[3] = IMAGE::quantize_unsigned<mi::Uint8>(floats[3]);
}

template <>
void Tile_impl::get_pixel<PT_RGBA>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = get_position<PT_RGBA>( x_offset, y_offset);
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/255.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/255.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/255.0);
    floats[3] = mi::Float32( position[3]) * mi::Float32( 1.0/255.0);
}

// ---------- PT_RGBE ------------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGBE>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = get_position<PT_RGBE>( x_offset, y_offset);
    mi::math::to_rgbe( floats, position);
}

template <>
void Tile_impl::get_pixel<PT_RGBE>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = get_position<PT_RGBE>( x_offset, y_offset);
    mi::math::from_rgbe( position, floats);
    floats[3] = 1.0f;
}

// ---------- PT_RGBEA -----------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGBEA>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint8* const position = get_position<PT_RGBEA>( x_offset, y_offset);
    mi::math::to_rgbe( floats, position);
    position[4] = IMAGE::quantize_unsigned<mi::Uint8>(floats[3]);
}

template <>
void Tile_impl::get_pixel<PT_RGBEA>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint8* const position = get_position<PT_RGBEA>( x_offset, y_offset);
    mi::math::from_rgbe( position, floats);
    floats[3] = mi::Float32( position[4]) * mi::Float32( 1.0/255.0);
}

// ---------- PT_RGB_16 ----------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGB_16>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint16* const position = get_position<PT_RGB_16>( x_offset, y_offset);
    position[0] = IMAGE::quantize_unsigned<mi::Uint16>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint16>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint16>(floats[2]);
}

template <>
void Tile_impl::get_pixel<PT_RGB_16>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint16* const position = get_position<PT_RGB_16>( x_offset, y_offset);
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/65535.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/65535.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/65535.0);
    floats[3] = 1.0f;
}

// ---------- PT_RGBA_16 ---------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGBA_16>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Uint16* const position = get_position<PT_RGBA_16>( x_offset, y_offset);
    position[0] = IMAGE::quantize_unsigned<mi::Uint16>(floats[0]);
    position[1] = IMAGE::quantize_unsigned<mi::Uint16>(floats[1]);
    position[2] = IMAGE::quantize_unsigned<mi::Uint16>(floats[2]);
    position[3] = IMAGE::quantize_unsigned<mi::Uint16>(floats[3]);
}

template <>
void Tile_impl::get_pixel<PT_RGBA_16>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Uint16* const position = get_position<PT_RGBA_16>( x_offset, y_offset);
    floats[0] = mi::Float32( position[0]) * mi::Float32( 1.0/65535.0);
    floats[1] = mi::Float32( position[1]) * mi::Float32( 1.0/65535.0);
    floats[2] = mi::Float32( position[2]) * mi::Float32( 1.0/65535.0);
    floats[3] = mi::Float32( position[3]) * mi::Float32( 1.0/65535.0);
}

// ---------- PT_RGB_FP ----------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_RGB_FP>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = get_position<PT_RGB_FP>( x_offset, y_offset);
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
}

template <>
void Tile_impl::get_pixel<PT_RGB_FP>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = get_position<PT_RGB_FP>( x_offset, y_offset);
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = 1.0f;
}

// ---------- PT_COLOR -----------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_COLOR>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    mi::Float32* const position = get_position<PT_COLOR>( x_offset, y_offset);
    position[0] = floats[0];
    position[1] = floats[1];
    position[2] = floats[2];
    position[3] = floats[3];
}

template <>
void Tile_impl::get_pixel<PT_COLOR>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    if( x_offset >= m_width || y_offset >= m_height)
        return;

    const mi::Float32* const position = get_position<PT_COLOR>( x_offset, y_offset);
    floats[0] = position[0];
    floats[1] = position[1];
    floats[2] = position[2];
    floats[3] = position[3];
}

// ---------- PT_SINT32 ----------------------------------------------------------------------------

template <>
void Tile_impl::set_pixel<PT_SINT32>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    // treat PT_SINT32 as PT_RGBA
    set_pixel<PT_RGBA>( x_offset, y_offset, floats);
}

template <>
void Tile_impl::get_pixel<PT_SINT32>(
    mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    // treat PT_SINT32 as PT_RGBA
    get_pixel<PT_RGBA>( x_offset, y_offset, floats);
}

// -------------------------------------------------------------------------------------------------


void Tile_impl::set_pixel( mi::Uint32 x_offset, mi::Uint32 y_offset, const mi::Float32* floats)
{
    switch ( m_type) {
        case PT_UNDEF:     ASSERT( M_IMAGE, false); return;
        case PT_SINT8:     return set_pixel<PT_SINT8    >( x_offset, y_offset, floats);
        case PT_SINT32:    return set_pixel<PT_SINT32   >( x_offset, y_offset, floats);
        case PT_FLOAT32:   return set_pixel<PT_FLOAT32  >( x_offset, y_offset, floats);
        case PT_FLOAT32_2: return set_pixel<PT_FLOAT32_2>( x_offset, y_offset, floats);
        case PT_FLOAT32_3: return set_pixel<PT_FLOAT32_3>( x_offset, y_offset, floats);
        case PT_FLOAT32_4: return set_pixel<PT_FLOAT32_4>( x_offset, y_offset, floats);
        case PT_RGB:       return set_pixel<PT_RGB      >( x_offset, y_offset, floats);
        case PT_RGBA:      return set_pixel<PT_RGBA     >( x_offset, y_offset, floats);
        case PT_RGBE:      return set_pixel<PT_RGBE     >( x_offset, y_offset, floats);
        case PT_RGBEA:     return set_pixel<PT_RGBEA    >( x_offset, y_offset, floats);
        case PT_RGB_16:    return set_pixel<PT_RGB_16   >( x_offset, y_offset, floats);
        case PT_RGBA_16:   return set_pixel<PT_RGBA_16  >( x_offset, y_offset, floats);
        case PT_RGB_FP:    return set_pixel<PT_RGB_FP   >( x_offset, y_offset, floats);
        case PT_COLOR:     return set_pixel<PT_COLOR    >( x_offset, y_offset, floats);
        default:           ASSERT( M_IMAGE, false); return;
    }
}

void Tile_impl::get_pixel( mi::Uint32 x_offset, mi::Uint32 y_offset, mi::Float32* floats) const
{
    switch ( m_type) {
        case PT_UNDEF:     ASSERT( M_IMAGE, false); return;
        case PT_SINT8:     return get_pixel<PT_SINT8    >( x_offset, y_offset, floats);
        case PT_SINT32:    return get_pixel<PT_SINT32   >( x_offset, y_offset, floats);
        case PT_FLOAT32:   return get_pixel<PT_FLOAT32  >( x_offset, y_offset, floats);
        case PT_FLOAT32_2: return get_pixel<PT_FLOAT32_2>( x_offset, y_offset, floats);
        case PT_FLOAT32_3: return get_pixel<PT_FLOAT32_3>( x_offset, y_offset, floats);
        case PT_FLOAT32_4: return get_pixel<PT_FLOAT32_4>( x_offset, y_offset, floats);
        case PT_RGB:       return get_pixel<PT_RGB      >( x_offset, y_offset, floats);
        case PT_RGBA:      return get_pixel<PT_RGBA     >( x_offset, y_offset, floats);
        case PT_RGBE:      return get_pixel<PT_RGBE     >( x_offset, y_offset, floats);
        case PT_RGBEA:     return get_pixel<PT_RGBEA    >( x_offset, y_offset, floats);
        case PT_RGB_16:    return get_pixel<PT_RGB_16   >( x_offset, y_offset, floats);
        case PT_RGBA_16:   return get_pixel<PT_RGBA_16  >( x_offset, y_offset, floats);
        case PT_RGB_FP:    return get_pixel<PT_RGB_FP   >( x_offset, y_offset, floats);
        case PT_COLOR:     return get_pixel<PT_COLOR    >( x_offset, y_offset, floats);
        default:           ASSERT( M_IMAGE, false); return;
    }
}


mi::neuraylib::ITile* create_tile( Pixel_type pixel_type, mi::Uint32 width, mi::Uint32 height)
{
    return new Tile_impl( width, height, pixel_type);
}

mi::neuraylib::ITile* copy_tile( const mi::neuraylib::ITile* other)
{
    return new Tile_impl( other);
}

} // namespace IMAGE

} // namespace MI
