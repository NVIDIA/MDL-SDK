/******************************************************************************
 * Copyright (c) 2013-2024, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/
/** \file
 ** \brief
 **/

#include "pch.h"

#include "i_mdlrt_texture.h"

#include <math.h>

#include <mi/math/color.h>
#include <mi/neuraylib/iimage.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/i_image_utilities.h>
#include <io/scene/texture/i_texture.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <base/data/db/i_db_access.h>

namespace MI {
namespace MDLRT {

//-------------------------------------------------------------------------------------------------

namespace {

float gamma_func(const float f, const float gamma_val)
{
    return f <= 0.0f ? 0.0f : powf(f, gamma_val);
}

void apply_gamma1(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f)
        c.r = gamma_func(c.r, gamma_val);
}

void apply_gamma2(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f) {
        c.r = gamma_func(c.r, gamma_val);
        c.g = gamma_func(c.g, gamma_val);
    }
}

void apply_gamma3(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f) {
        c.r = gamma_func(c.r, gamma_val);
        c.g = gamma_func(c.g, gamma_val);
        c.b = gamma_func(c.b, gamma_val);
    }
}

void apply_gamma4(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f) {
        c.r = gamma_func(c.r, gamma_val);
        c.g = gamma_func(c.g, gamma_val);
        c.b = gamma_func(c.b, gamma_val);
    }
}

float saturate(const float f)
{
    return std::max(0.0f, std::min(1.0f, f));
}

unsigned int float_as_uint(const float f) {
    union {
        float f;
        unsigned int i;
    } u;
    u.f = f;
    return u.i;
}

int          __float2int_rz( const float f)        { return (int)f; }
long long    __float2ll_rz(  const float f)        { return (long long)f; }
long long    __float2ll_rd(  const float f)        { return (long long)floorf(f); }
float        __uint2float_rn(const unsigned int i) { return (float)i; }
unsigned int __float2uint_rz(const float f)        { return (unsigned int)f; }

mi::Uint32_2 texremapll(
    const mi::mdl::stdlib::Tex_wrap_mode wrap_u,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_v,
    const mi::Uint32_2 &texres,
    const mi::Sint32_2 &crop_ofs,
    const mi::Float32_2 &tex)
{
    const long long texix = __float2ll_rz(tex.x);
    const long long texiy = __float2ll_rz(tex.y);

    mi::Sint32_2 texi;

    // early out if in range 0, texres.x-1 // extra _rd cast needed to catch -1..0 case
    if((unsigned long long)__float2ll_rd(tex.x) >= (unsigned long long)texres.x)
    {
        // Wrap or Clamp
        if (wrap_u == mi::mdl::stdlib::wrap_clamp || wrap_u == mi::mdl::stdlib::wrap_clip)
            texi.x = (int)std::min(std::max(texix, 0ll), (long long)(texres.x-1));
        else
        {
            const int s = signbit(tex.x); // extract sign to handle all < 0 magic below
            const long long d = texix/(long long)texres.x;
            texi.x = texix%(long long)texres.x;

            const int a =
                (int)(wrap_u == mi::mdl::stdlib::wrap_mirrored_repeat) & ((int)d^s) & 1;
            const bool altu = (a != 0);
            if(altu)   // if alternating, negative tex has to be flipped
                texi.x = -texi.x;
            if(s != a) // "otherwise" negative tex will be padded back to positive
                texi.x += (int)texres.x-1;
        }
    }
    else
        texi.x = (int)texix;

    // Crop
    texi.x += crop_ofs.x;

    // early out if in range 0, texres.y-1 // extra _rd cast needed to catch -1..0 case
    if((unsigned long long)__float2ll_rd(tex.y) >= (unsigned long long)texres.y)
    {
        // Wrap or Clamp
        if (wrap_v == mi::mdl::stdlib::wrap_clamp || wrap_v == mi::mdl::stdlib::wrap_clip)
            texi.y = (int)std::min(std::max(texiy, 0ll), (long long)(texres.y-1));
        else
        {
            const int s = signbit(tex.y); // extract sign to handle all < 0 magic below
            const long long d = texiy/(long long)texres.y;
            texi.y = texiy%(long long)texres.y;

            const int a =
                (int)(wrap_v == mi::mdl::stdlib::wrap_mirrored_repeat) & ((int)d^s) & 1;
            const bool altv = (a != 0);
            if(altv)   // if alternating, negative tex has to be flipped
                texi.y = -texi.y;
            if(s != a) // "otherwise" negative tex will be padded back to positive
                texi.y += (int)texres.y-1;
        }
    }
    else
        texi.y = (int)texiy;

    // Crop
    texi.y += crop_ofs.y;

    return mi::Uint32_2(texi.x, texi.y);
}

unsigned int texremapzll(
    const mi::mdl::stdlib::Tex_wrap_mode wrap_w,
    const unsigned int texresz,
    const int crop_ofsz,
    const float texz)
{
    const long long texiz = __float2ll_rz(texz);

    int texi;

    // early out if in range 0, texres.x-1 // extra _rd cast needed to catch -1..0 case
    if((unsigned long long)__float2ll_rd(texz) >= (unsigned long long)texresz)
    {
        // Wrap or Clamp
        if (wrap_w == mi::mdl::stdlib::wrap_clamp || wrap_w == mi::mdl::stdlib::wrap_clip)
            texi = (int)std::min(std::max(texiz, 0ll), (long long)(texresz-1));
        else
        {
            const int s = signbit(texz); // extract sign to handle all < 0 magic below
            const long long d = texiz/(long long)texresz;
            texi = texiz%(long long)texresz;

            const int a =
                (int)(wrap_w == mi::mdl::stdlib::wrap_mirrored_repeat) & ((int)d^s) & 1;
            const bool altu = (a != 0);
            if(altu)   // if alternating, negative tex has to be flipped
                texi = -texi;
            if(s != a) // "otherwise" negative tex will be padded back to positive
                texi += (int)texresz-1;
        }
    }
    else
        texi = (int)texiz;

    // Crop
    texi += crop_ofsz;

    return texi;
}

mi::Float32_4 interpolate_biquintic(
    const IMAGE::Access_canvas &canvas,
    const mi::Uint32_3 &texture_res,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_u,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_v,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_w,
    const mi::Float32_4 &crop_uv,
    const mi::Float32_2 &crop_w,
    const mi::Float32_3 &texo,
    const bool smootherstep,
    const float gamma_val,
    const unsigned int layer_offset = 0)
{
    if (texture_res.x == 0 || texture_res.y == 0)
        return {0.0f, 0.0f , 0.0f, 0.0f};

    if(((wrap_u == mi::mdl::stdlib::wrap_clip) && (texo.x < 0.0f || texo.x > 1.0f))
        ||
       ((wrap_v == mi::mdl::stdlib::wrap_clip) && (texo.y < 0.0f || texo.y > 1.0f)))

        return {0.0f, 0.0f, 0.0f, 0.0f};

    const mi::Uint32_2 full_texres(texture_res.x, texture_res.y);
    const mi::Sint32_2 crop_ofs(
        __float2int_rz(__uint2float_rn(full_texres.x-1) * crop_uv.x),
        __float2int_rz(__uint2float_rn(full_texres.y-1) * crop_uv.z));

    ASSERT(M_BACKENDS, crop_uv.x >= 0.0f && crop_uv.y >= 0.0f);
    const mi::Uint32_2 texres(
        std::max(__float2uint_rz(__uint2float_rn(texture_res.x) * crop_uv.y), 1u),
        std::max(__float2uint_rz(__uint2float_rn(texture_res.y) * crop_uv.w), 1u));

    //!! opt.? use floor'ed float values of texres instead of cast?
    const mi::Float32_2 tex(texo.x * __uint2float_rn(texres.x) - 0.5f,
                            texo.y * __uint2float_rn(texres.y) - 0.5f);

    // check for LLONG_MAX as texremapll overflows otherwise
    if((texres.x == 0) || (texres.y == 0) || (((float_as_uint(tex.x))&0x7FFFFFFF) >= 0x5f000000) ||
       (((float_as_uint(tex.y))&0x7FFFFFFF) >= 0x5f000000))
        return {0.0f, 0.0f, 0.0f, 0.0f};

    const mi::Uint32_2 texi0 = texremapll(wrap_u, wrap_v, texres, crop_ofs, tex);
    //!! +1 in float can screw-up bilerp
    const mi::Uint32_2 texi1 = texremapll(
        wrap_u, wrap_v, texres, crop_ofs, mi::Float32_2(tex.x+1.0f, tex.y+1.0f));
    const mi::Uint32_4 texi(texi0.x, texi0.y, texi1.x, texi1.y);

    ASSERT(M_BACKENDS, texi.x < full_texres.x && texi.y < full_texres.y);
    ASSERT(M_BACKENDS, texi.z < full_texres.x && texi.w < full_texres.y);

    mi::Float32_2 lerp(tex.x - floorf(tex.x), tex.y - floorf(tex.y));

    // 3D texture?
    unsigned int texi0_z = 0;
    unsigned int texi1_z = 0;
    float lerp_z = 0.f;
    if(texture_res.z > 1)
    {
        if((wrap_w == mi::mdl::stdlib::wrap_clip) && (texo.z < 0.0f || texo.z > 1.0f))
            return {0.0f, 0.0f, 0.0f, 0.0f};

        const int crop_ofs_z = __float2int_rz(__uint2float_rn(texture_res.z-1) * crop_w.x);

        ASSERT(M_BACKENDS, crop_w.x >= 0.0f && crop_w.y >= 0.0f);
        const unsigned int crop_texres_z = std::max(
            __float2uint_rz(__uint2float_rn(texture_res.z) * crop_w.y), 1u);

        //!! opt.? use floor'ed float values of texres instead of cast?
        const float tex_z = texo.z * __uint2float_rn(crop_texres_z) - 0.5f;

        // check for LLONG_MAX as texremapll overflows otherwise
        if((crop_texres_z == 0) || (((float_as_uint(tex_z))&0x7FFFFFFF) >= 0x5f000000))
            return {0.0f, 0.0f, 0.0f, 0.0f};

        texi0_z = texremapzll(wrap_w, crop_texres_z, crop_ofs_z, tex_z);
        //!! +1 in float can screw-up bilerp if precision maps it to same texel again
        texi1_z = texremapzll(wrap_w, crop_texres_z, crop_ofs_z, tex_z+1.0f);

        lerp_z = tex_z - floorf(tex_z);
    }

    if(smootherstep) {
        lerp.x *= lerp.x*lerp.x*(lerp.x*(lerp.x*6.0f-15.0f)+10.0f); // smootherstep
        lerp.y *= lerp.y*lerp.y*(lerp.y*(lerp.y*6.0f-15.0f)+10.0f);
    }

    const mi::Float32_4 st(
        (1.0f-lerp.x)*(1.0f-lerp.y), lerp.x*(1.0f-lerp.y), (1.0f-lerp.x)*lerp.y, lerp.x*lerp.y);


    mi::Float32_4 rgba(0.f, 0.f, 0.f, 1.f);
    mi::Float32_4 rgba2(0.f, 0.f, 0.f, 1.f);


    for (unsigned int i = 0; i < 2; ++i)
    {
        const unsigned int z_layer = ((i == 0) ? texi1_z : texi0_z) + layer_offset;

        mi::math::Color col(0.f, 0.f, 0.f, 1.f);
        mi::math::Color c0, c1, c2, c3;
        canvas.lookup(c0, texi.x, texi.y, z_layer);
        canvas.lookup(c1, texi.z, texi.y, z_layer);
        canvas.lookup(c2, texi.x, texi.w, z_layer);
        canvas.lookup(c3, texi.z, texi.w, z_layer);

        col = c0 * st.x + c1 * st.y + c2 * st.z + c3 * st.w;
        rgba = mi::Float32_4(col.r, col.g, col.b, col.a);

        // 3D textures loop twice
        if(lerp_z != 0.f)
            rgba2 = rgba;
        else
            break;
   }


    // 3D textures lerp between two layer results
    if(lerp_z != 0.f)
        rgba += (rgba2-rgba)*lerp_z;

    if(gamma_val != 1.0f) {
        rgba.x = gamma_func(rgba.x, gamma_val);
        rgba.y = gamma_func(rgba.y, gamma_val);
        rgba.z = gamma_func(rgba.z, gamma_val);
    }
    return rgba;
}

// Converts \p input to a floating-point pixel type with gamma 1.0.
//
// \param gamma   The gamma value of \p input (might be different from input->get_gamma() if
//                overridden on the texture).
mi::neuraylib::ICanvas* convert_to_fp_type_with_linear_gamma(
    IMAGE::Image_module* image_module, const mi::neuraylib::ICanvas* input, mi::Float32 gamma)
{
    // Choose floating-point pixel type.
    IMAGE::Pixel_type pixel_type = IMAGE::convert_pixel_type_string_to_enum(input->get_type());
    switch (pixel_type) {
        case IMAGE::PT_RGB:
        case IMAGE::PT_RGBE:
        case IMAGE::PT_RGB_16:
        case IMAGE::PT_FLOAT32_2:
        case IMAGE::PT_FLOAT32_3:
            pixel_type = IMAGE::PT_RGB_FP;
            break;
        case IMAGE::PT_RGBA:
        case IMAGE::PT_RGBEA:
        case IMAGE::PT_RGBA_16:
        case IMAGE::PT_FLOAT32_4:
            pixel_type = IMAGE::PT_COLOR;
            break;
        case IMAGE::PT_SINT8:
        case IMAGE::PT_SINT32:
            pixel_type = IMAGE::PT_FLOAT32;
            break;
        case IMAGE::PT_RGB_FP:
        case IMAGE::PT_COLOR:
        case IMAGE::PT_FLOAT32:
            // no change necessary
            break;
        case IMAGE::PT_UNDEF:
            ASSERT(M_BACKENDS, false);
            break;
    }

    mi::base::Handle<mi::neuraylib::ICanvas> result(
        image_module->convert_canvas(input, pixel_type));
    result->set_gamma(gamma);
    image_module->adjust_gamma(result.get(), 1.0f);
    result->retain();
    return result.get();
}

} // namespace

//-------------------------------------------------------------------------------------------------

mi::Size Texture::get_frame_id(mi::Float32 frame) const
{
    if (!m_is_animated)
        return 0;

    // just an optimization
    if (frame < 0.0f)
        return static_cast<mi::Size>(-1);

    mi::Size f = static_cast<mi::Size>(floorf(frame));
    auto it = m_frame_number_to_id.find(f);
    return it != m_frame_number_to_id.end() ? it->second : static_cast<mi::Size>(-1);
}

std::pair<mi::Size, mi::Size> Texture::get_frame_ids(mi::Float32 frame) const
{
    if (!m_is_animated)
        return std::make_pair<mi::Size, mi::Size>( 0, 0);

    // important to return invalid IDs for both values
    if (frame < m_frame_number_to_id.begin()->first || frame > m_frame_number_to_id.rbegin()->first)
        return std::make_pair<mi::Size, mi::Size>( -1,  -1);

    mi::Size f = static_cast<mi::Size>(floorf(frame));
    auto f_it = m_frame_number_to_id.find(f);
    mi::Size f_id = f_it != m_frame_number_to_id.end() ? f_it->second : static_cast<mi::Size>(-1);

    mi::Size c = static_cast<mi::Size>(ceilf(frame));
    auto c_it = m_frame_number_to_id.find(c);
    mi::Size c_id = c_it != m_frame_number_to_id.end() ? c_it->second : static_cast<mi::Size>(-1);

    return std::make_pair( f_id, c_id);
}

//-------------------------------------------------------------------------------------------------

Texture_2d::Texture_2d(
    const DB::Typed_tag<TEXTURE::Texture>& tag,
    bool use_derivatives,
    DB::Transaction* transaction)
  : m_use_derivatives(use_derivatives)
{
    SYSTEM::Access_module<IMAGE::Image_module> image_module(false);

    if (!tag)
        return;

    DB::Access<TEXTURE::Texture> texture(tag, transaction);
    DB::Tag image_tag = texture->get_image();
    if (!image_tag)
        return;

    DB::Access<DBIMAGE::Image> image(image_tag, transaction);
    m_is_valid = image->is_valid();
    if (!m_is_valid)
        return;

    // Just to accelerate the get_mipmap() calls below
    DB::Access<DBIMAGE::Image_impl> image_impl(image->get_impl_tag(), transaction);

    m_is_animated = image->is_animated();
    m_is_uvtile   = image->is_uvtile();

    mi::Size n_frames = image->get_length();
    m_first_last_frame = mi::Uint32_2(
        static_cast<mi::Uint32>(image->get_frame_number(0)),
        static_cast<mi::Uint32>(image->get_frame_number(n_frames-1)));
    m_frames.resize(n_frames);

    const auto& image_frames = image->get_frames_vector();

    for (mi::Size i = 0; i < n_frames; ++i) {

        Frame& frame = m_frames[i];

        frame.m_uv_to_id = image_frames[i].m_uv_to_id;

        mi::Size n_uvtiles = image->get_frame_length(i);
        frame.m_uvtiles.resize(n_uvtiles);

        for (mi::Size j = 0; j < n_uvtiles; ++j) {

            Uvtile& uvtile = frame.m_uvtiles[j];

            uvtile.m_gamma = texture->get_effective_gamma(transaction, i, j);
            if (uvtile.m_gamma <= 0.0f)
                uvtile.m_gamma = 0.0f;

            uvtile.m_canvas.resize(1);
            uvtile.m_resolution.resize(1);

            mi::base::Handle<const IMAGE::IMipmap> mipmap(image_impl->get_mipmap(i, j));
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(mipmap->get_level(/*level*/ 0));

            // Convert to linear gamma first if derivatives are enabled. TODO For non-derivative
            // mode, the gamma is still (incorrectly) applied after filtering.
            if (use_derivatives && uvtile.m_gamma != 1.0f) {
                canvas = convert_to_fp_type_with_linear_gamma(
                    image_module.get(), canvas.get(), uvtile.m_gamma);
                uvtile.m_gamma = 1.0f;
            }

            uvtile.m_canvas[0] = IMAGE::Access_canvas(canvas.get(), true);
            uvtile.m_resolution[0] = mi::Uint32_3(
                canvas->get_resolution_x(), canvas->get_resolution_y(), 0);

            if (!use_derivatives)
                continue;

            std::vector<mi::base::Handle<mi::neuraylib::ICanvas>> mipmaps;
            image_module->create_mipmap(mipmaps, canvas.get(), 1.0f);

            mi::Uint32 n_levels = 1 + mipmaps.size();
            uvtile.m_canvas.resize(n_levels);
            uvtile.m_resolution.resize(n_levels);

            for (mi::Uint32 k = 1; k < n_levels; ++k) {
                const auto& level = mipmaps[k-1];
                uvtile.m_canvas[k] = IMAGE::Access_canvas(level.get(), true);
                uvtile.m_resolution[k] = mi::Uint32_3(
                    level->get_resolution_x(),
                    level->get_resolution_y(),
                    0);
            }
        }

        mi::Size frame_number = image->get_frame_number(i);
        m_frame_number_to_id[frame_number] = i;
    }
}

mi::Uint32_2 Texture_2d::get_resolution(const mi::Sint32_2& uv_tile, mi::Float32 frame_param) const
{
    mi::Size frame_id = get_frame_id(frame_param);
    if (frame_id == static_cast<mi::Size>(-1))
        return {0, 0};

    const Frame& frame = m_frames[frame_id];
    mi::Uint32 uvtile_id = frame.m_uv_to_id.get(uv_tile.x, uv_tile.y);
    if (uvtile_id == ~0u)
        return {0, 0};

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];
    return {uvtile.m_resolution[0].x, uvtile.m_resolution[0].y};
}

float Texture_2d::lookup_float(
    const mi::Float32_2& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    mi::Float32 frame) const
{
    return lookup_float4(coord, wrap_u, wrap_v, crop_u, crop_v, frame).x;
}

mi::Float32_2 Texture_2d::lookup_float2(
    const mi::Float32_2& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(coord, wrap_u, wrap_v, crop_u, crop_v, frame);
    return {res.x, res.y};
}

mi::Float32_3 Texture_2d::lookup_float3(
    const mi::Float32_2& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(coord, wrap_u, wrap_v, crop_u, crop_v, frame);
    return {res.x, res.y, res.z};
}

mi::Float32_4 Texture_2d::lookup_float4(
    const mi::Float32_2& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    mi::Float32 frame) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    mi::Size floor_frame_id, ceil_frame_id;
    std::tie( floor_frame_id, ceil_frame_id) = get_frame_ids(frame);
    bool floor_frame_valid = floor_frame_id != static_cast<mi::Size>(-1);
    bool ceil_frame_valid  = ceil_frame_id  != static_cast<mi::Size>(-1);
    if (!floor_frame_valid && !ceil_frame_valid)
        return mi::Float32_4(0.0f);

    mi::Float32_3 coords(coord.x, coord.y, 0.0f);

    mi::Float32_4 crop_uv;
    mi::Sint32 u = 0;
    mi::Sint32 v = 0;
    if (m_is_uvtile) {
        u         = static_cast<mi::Sint32>(floorf(coords.x));
        v         = static_cast<mi::Sint32>(floorf(coords.y));
        coords.x -= floorf(coords.x);
        coords.y -= floorf(coords.y);
        crop_uv   = mi::Float32_4(0.0f, 1.0f, 0.0f, 1.0f);
        wrap_u    = mi::mdl::stdlib::wrap_clamp;
        wrap_v    = mi::mdl::stdlib::wrap_clamp;
    } else {
        crop_uv = mi::Float32_4(
          saturate(crop_u.x), saturate(crop_u.y - crop_u.x),
          saturate(crop_v.x), saturate(crop_v.y - crop_v.x));
    }

    const mi::Float32_2 crop_w(0.f, 1.f);

    mi::Float32_4 floor_value = lookup_float4_frame(
        coords,
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        floor_frame_id, u, v);

    // Use just the floor value if not animated or frame param is not fractional.
    if( floor_frame_id == ceil_frame_id)
        return floor_value;

    mi::Float32_4 ceil_value = lookup_float4_frame(
        coords,
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        ceil_frame_id, u, v);

    mi::Float32 ceil_weight = frame - floorf(frame);
    return (1.0f-ceil_weight) * floor_value + ceil_weight * ceil_value;
}

mi::Float32_4 Texture_2d::lookup_float4_frame(
    const mi::Float32_3& coords,
    mi::mdl::stdlib::Tex_wrap_mode wrap_u,
    mi::mdl::stdlib::Tex_wrap_mode wrap_v,
    mi::mdl::stdlib::Tex_wrap_mode wrap_w,
    const mi::Float32_4& crop_uv,
    const mi::Float32_2& crop_w,
    mi::Size frame_id,
    mi::Uint32 u,
    mi::Uint32 v) const
{
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_4(0.0f);

    const Frame& frame = m_frames[frame_id];

    mi::Uint32 uvtile_id = 0;
    if (m_is_uvtile) {
        uvtile_id = frame.m_uv_to_id.get(u, v);
        if (uvtile_id == ~0u)
            return mi::Float32_4(0.0f);
    }

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];

    return interpolate_biquintic(
        uvtile.m_canvas[0],
        uvtile.m_resolution[0],
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        coords, /*smootherstep*/ true, uvtile.m_gamma);
}

mi::Float32_4 Texture_2d::lookup_deriv_float4(
    const mi::Float32_2& coord_val,
    const mi::Float32_2& coord_dx,
    const mi::Float32_2& coord_dy,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    mi::Float32 frame) const
{
    ASSERT(M_BACKENDS, m_use_derivatives);
    if (!m_use_derivatives)
        return lookup_float4(
            coord_val, wrap_u, wrap_v, crop_u, crop_v, frame);

    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    mi::Size floor_frame_id, ceil_frame_id;
    std::tie( floor_frame_id, ceil_frame_id) = get_frame_ids(frame);
    bool floor_frame_valid = floor_frame_id != static_cast<mi::Size>(-1);
    bool ceil_frame_valid  = ceil_frame_id  != static_cast<mi::Size>(-1);
    if (!floor_frame_valid && !ceil_frame_valid)
        return mi::Float32_4(0.0f);

    mi::Float32_3 coords(coord_val.x, coord_val.y, 0.0f);

    mi::Float32_4 crop_uv;
    mi::Sint32 u = 0;
    mi::Sint32 v = 0;
    if (m_is_uvtile) {
        u         = static_cast<mi::Sint32>(floorf(coords.x));
        v         = static_cast<mi::Sint32>(floorf(coords.y));
        coords.x -= floorf(coords.x);
        coords.y -= floorf(coords.y);
        crop_uv   = mi::Float32_4(0.0f, 1.0f, 0.0f, 1.0f);
        wrap_u    = mi::mdl::stdlib::wrap_clamp;
        wrap_v    = mi::mdl::stdlib::wrap_clamp;
    } else {
        crop_uv = mi::Float32_4(
          saturate(crop_u.x), saturate(crop_u.y - crop_u.x),
          saturate(crop_v.x), saturate(crop_v.y - crop_v.x));
    }

    const mi::Float32_2 crop_w(0.f, 1.f);

    mi::Float32_4 floor_value = lookup_deriv_float4_frame(
        coords, coord_dx, coord_dy,
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        floor_frame_id, u, v);

    // Use just the floor value if not animated or frame param is not fractional.
    if( floor_frame_id == ceil_frame_id)
        return floor_value;

    mi::Float32_4 ceil_value = lookup_deriv_float4_frame(
        coords, coord_dx, coord_dy, wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        ceil_frame_id, u, v);

    mi::Float32 ceil_weight = frame - floorf(frame);
    return (1.0f-ceil_weight) * floor_value + ceil_weight * ceil_value;
}

mi::Float32_4 Texture_2d::lookup_deriv_float4_frame(
    const mi::Float32_3& coords,
    const mi::Float32_2& coord_dx,
    const mi::Float32_2& coord_dy,
    mi::mdl::stdlib::Tex_wrap_mode wrap_u,
    mi::mdl::stdlib::Tex_wrap_mode wrap_v,
    mi::mdl::stdlib::Tex_wrap_mode wrap_w,
    const mi::Float32_4& crop_uv,
    const mi::Float32_2& crop_w,
    mi::Size frame_id,
    mi::Uint32 u,
    mi::Uint32 v) const
{
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_4(0.0f);

    const Frame& frame = m_frames[frame_id];

    mi::Uint32 uvtile_id = 0;
    if (m_is_uvtile) {
        uvtile_id = frame.m_uv_to_id.get(u, v);
        if (uvtile_id == ~0u)
            return mi::Float32_4(0.0f);
    }

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];

    // isotropic filtering
    mi::Size n_levels = uvtile.m_canvas.size();
    float dx_len_sqr  = coord_dx.x * coord_dx.x + coord_dx.y * coord_dx.y;
    float dy_len_sqr  = coord_dy.x * coord_dy.x + coord_dy.y * coord_dy.y;
    float max_len_sqr = std::max(dx_len_sqr, dy_len_sqr);
    float level       = n_levels - 1 + 0.5f * std::log2f(std::max(max_len_sqr, 1e-8f));

    if (level < 0) {
        return interpolate_biquintic(
            uvtile.m_canvas[0],
            uvtile.m_resolution[0],
            wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
            crop_uv, crop_w,
            coords, /*smootherstep*/ true, 1.0f);
    }

    if (level >= n_levels - 1) {
        // just read the single pixel of the smallest mipmap
        mi::math::Color col;
        uvtile.m_canvas[n_levels-1].lookup(col, 0, 0);
        return {col.r, col.g, col.b, col.a};
    }

    // do trilinear filtering between the two mipmap levels
    unsigned int level_uint = static_cast<unsigned int>(floorf(level));
    float lerp = level - level_uint;

    mi::Float32_4 rgba_0 = interpolate_biquintic(
        uvtile.m_canvas[level_uint],
        uvtile.m_resolution[level_uint],
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        coords, /*smootherstep*/ true, 1.0f);

    mi::Float32_4 rgba_1 = interpolate_biquintic(
        uvtile.m_canvas[level_uint + 1],
        uvtile.m_resolution[level_uint + 1],
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        coords, /*smootherstep*/ true, 1.0f);

    return (1 - lerp) * rgba_0 + lerp * rgba_1;
}

mi::Spectrum Texture_2d::lookup_color(
    const mi::Float32_2& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(coord, wrap_u, wrap_v, crop_u, crop_v, frame);
    return {res.x, res.y, res.z};
}

float Texture_2d::texel_float(
    const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile, mi::Float32 frame_param) const
{
    if (!m_is_valid)
        return 0.0f;

    mi::Size frame_id = get_frame_id(frame_param);
    if (frame_id == static_cast<mi::Size>(-1))
        return 0.0f;

    const Frame& frame = m_frames[frame_id];
    mi::Uint32 uvtile_id = frame.m_uv_to_id.get(uv_tile.x, uv_tile.y);
    if (uvtile_id == ~0u)
        return 0.0f;

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];
    mi::math::Color res(0.0f);
    uvtile.m_canvas[0].lookup(res, coord.x, coord.y, 0);
    apply_gamma1(res, uvtile.m_gamma);
    return res.r;
}

mi::Float32_2 Texture_2d::texel_float2(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile,
    mi::Float32 frame_param) const
{
    if (!m_is_valid)
        return mi::Float32_2(0.0f);

    mi::Size frame_id = get_frame_id(frame_param);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_2(0.0f);

    const Frame& frame = m_frames[frame_id];
    mi::Uint32 uvtile_id = frame.m_uv_to_id.get(uv_tile.x, uv_tile.y);
    if (uvtile_id == ~0u)
        return mi::Float32_2(0.0f);

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];
    mi::math::Color res(0.0f);
    uvtile.m_canvas[0].lookup(res, coord.x, coord.y, 0);
    apply_gamma2(res, uvtile.m_gamma);
    return {res.r, res.g};
}

mi::Float32_3 Texture_2d::texel_float3(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile,
    mi::Float32 frame_param) const
{
    if (!m_is_valid)
        return mi::Float32_3(0.0f);

    mi::Size frame_id = get_frame_id(frame_param);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_3(0.0f);

    const Frame& frame = m_frames[frame_id];
    mi::Uint32 uvtile_id = frame.m_uv_to_id.get(uv_tile.x, uv_tile.y);
    if (uvtile_id == ~0u)
        return mi::Float32_3(0.0f);

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];
    mi::math::Color res(0.0f);
    uvtile.m_canvas[0].lookup(res, coord.x, coord.y, 0);
    apply_gamma3(res, uvtile.m_gamma);
    return {res.r, res.g, res.b};
}

mi::Float32_4 Texture_2d::texel_float4(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile,
    mi::Float32 frame_param) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    mi::Size frame_id = get_frame_id(frame_param);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_4(0.0f);

    const Frame& frame = m_frames[frame_id];
    mi::Uint32 uvtile_id = frame.m_uv_to_id.get(uv_tile.x, uv_tile.y);
    if (uvtile_id == ~0u)
        return mi::Float32_4(0.0f);

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];
    mi::math::Color res(0.0f);
    uvtile.m_canvas[0].lookup(res, coord.x, coord.y, 0);
    apply_gamma4(res, uvtile.m_gamma);
    return {res.r, res.g, res.b, res.a};
}

mi::Spectrum Texture_2d::texel_color(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile,
    mi::Float32 frame_param) const
{
    if (!m_is_valid)
        return mi::Spectrum(0.0f);

    mi::Size frame_id = get_frame_id(frame_param);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Spectrum(0.0f);

    const Frame& frame = m_frames[frame_id];
    mi::Uint32 uvtile_id = frame.m_uv_to_id.get(uv_tile.x, uv_tile.y);
    if (uvtile_id == ~0u)
        return mi::Spectrum(0.0f);

    const Uvtile& uvtile = frame.m_uvtiles[uvtile_id];
    mi::math::Color res(0.0f);
    uvtile.m_canvas[0].lookup(res, coord.x, coord.y, 0);
    apply_gamma3(res, uvtile.m_gamma);
    return {res.r, res.g, res.b};
}

//-------------------------------------------------------------------------------------------------

Texture_3d::Texture_3d(
    const DB::Typed_tag<TEXTURE::Texture>& tag,
    DB::Transaction* transaction)
{
    if (!tag)
        return;

    DB::Access<TEXTURE::Texture> texture(tag, transaction);
    DB::Tag image_tag = texture->get_image();
    if (!image_tag)
        return;

    DB::Access<DBIMAGE::Image> image(image_tag, transaction);
    m_is_valid = image->is_valid() && !image->is_uvtile();
    if (!m_is_valid)
        return;

    // Just to accelerate the get_mipmap() calls below
    DB::Access<DBIMAGE::Image_impl> image_impl(image->get_impl_tag(), transaction);

    m_is_animated = image->is_animated();

    mi::Size n_frames = image->get_length();
    m_first_last_frame = mi::Uint32_2(
        static_cast<mi::Uint32>(image->get_frame_number(0)),
        static_cast<mi::Uint32>(image->get_frame_number(n_frames-1)));
    m_frames.resize(n_frames);

    for (mi::Size i = 0; i < n_frames; ++i) {

        Frame& frame = m_frames[i];

        frame.m_gamma = texture->get_effective_gamma(transaction, i, /*uvtile_id*/ 0);
        if (frame.m_gamma <= 0.0f)
            frame.m_gamma = 0.0f;

        mi::base::Handle<const IMAGE::IMipmap> mipmap(
            image_impl->get_mipmap(i, /*uvtile_id*/ 0));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(mipmap->get_level(0));
        frame.m_canvas = IMAGE::Access_canvas(canvas.get(), true);

        frame.m_resolution = mi::Uint32_3(
            canvas->get_resolution_x(), canvas->get_resolution_y(), canvas->get_layers_size());

        mi::Size frame_number = image->get_frame_number(i);
        m_frame_number_to_id[frame_number] = i;
    }
}

mi::Uint32_3 Texture_3d::get_resolution(mi::Float32 frame) const
{
    mi::Size frame_id = get_frame_id(frame);
    if (frame_id == static_cast<mi::Size>(-1))
        return {0, 0, 0};

    return m_frames[frame_id].m_resolution;
}

float Texture_3d::lookup_float(
    const mi::Float32_3& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    Wrap_mode wrap_w,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    const mi::Float32_2& crop_w,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(
        coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame);
    return res.x;
}

mi::Float32_2 Texture_3d::lookup_float2(
    const mi::Float32_3& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    Wrap_mode wrap_w,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    const mi::Float32_2& crop_w,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(
        coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame);
    return {res.x, res.y};
}

mi::Float32_3 Texture_3d::lookup_float3(
    const mi::Float32_3& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    Wrap_mode wrap_w,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    const mi::Float32_2& crop_w,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(
        coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame);
    return {res.x, res.y, res.z};
}

mi::Float32_4 Texture_3d::lookup_float4(
    const mi::Float32_3& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    Wrap_mode wrap_w,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    const mi::Float32_2& crop_w,
    mi::Float32 frame) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    mi::Size floor_frame_id, ceil_frame_id;
    std::tie( floor_frame_id, ceil_frame_id) = get_frame_ids(frame);
    bool floor_frame_valid = floor_frame_id != static_cast<mi::Size>(-1);
    bool ceil_frame_valid  = ceil_frame_id  != static_cast<mi::Size>(-1);
    if (!floor_frame_valid && !ceil_frame_valid)
        return mi::Float32_4(0.0f);

    const mi::Float32_4 crop_uv(
        saturate(crop_u.x), saturate(crop_u.y - crop_u.x),
        saturate(crop_v.x), saturate(crop_v.y - crop_v.x));
    const mi::Float32_2 crop_w2(saturate(crop_w.x), saturate(crop_w.y - crop_w.x));

    mi::Float32_4 floor_value(0.0f);
    if( floor_frame_valid) {
        const Frame& floor_frame = m_frames[floor_frame_id];
        floor_value = interpolate_biquintic(
            floor_frame.m_canvas,
            floor_frame.m_resolution,
            wrap_u, wrap_v, wrap_w,
            crop_uv, crop_w2,
            coord,
            true,
            floor_frame.m_gamma);
    }

    // Use just the floor value if not animated or frame param is not fractional.
    if( floor_frame_id == ceil_frame_id)
        return floor_value;

    mi::Float32_4 ceil_value(0.0f);
    if( ceil_frame_valid) {
        const Frame& ceil_frame = m_frames[ceil_frame_id];
        ceil_value = interpolate_biquintic(
            ceil_frame.m_canvas,
            ceil_frame.m_resolution,
            wrap_u, wrap_v, wrap_w,
            crop_uv, crop_w2,
            coord,
            /*smootherstep*/ false,
            ceil_frame.m_gamma);
    }

    mi::Float32 ceil_weight = frame - floorf(frame);
    return (1.0f-ceil_weight) * floor_value + ceil_weight * ceil_value;
}

mi::Spectrum Texture_3d::lookup_color(
    const mi::Float32_3& coord,
    Wrap_mode wrap_u,
    Wrap_mode wrap_v,
    Wrap_mode wrap_w,
    const mi::Float32_2& crop_u,
    const mi::Float32_2& crop_v,
    const mi::Float32_2& crop_w,
    mi::Float32 frame) const
{
    const mi::Float32_4& res = lookup_float4(
        coord, wrap_u, wrap_v, wrap_w, crop_u, crop_v, crop_w, frame);
    return {res.x, res.y, res.z};
}

float Texture_3d::texel_float(const mi::Sint32_3& coord, mi::Float32 frame) const
{
    if (!m_is_valid)
        return 0.0f;

    mi::Size frame_id = get_frame_id(frame);
    if (frame_id == static_cast<mi::Size>(-1))
        return 0.0f;

    mi::math::Color c(0.0f);
    m_frames[frame_id].m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma1(c, m_frames[frame_id].m_gamma);

    return c.r;
}

mi::Float32_2 Texture_3d::texel_float2(const mi::Sint32_3& coord, mi::Float32 frame) const
{
    if (!m_is_valid)
        return mi::Float32_2(0.0f);

    mi::Size frame_id = get_frame_id(frame);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_2(0.0f);

    mi::math::Color c(0.0f);
    m_frames[frame_id].m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma2(c, m_frames[frame_id].m_gamma);
    return {c.r, c.g};
}

mi::Float32_3 Texture_3d::texel_float3(const mi::Sint32_3& coord, mi::Float32 frame) const
{
    if (!m_is_valid)
        return mi::Float32_3(0.0f);

    mi::Size frame_id = get_frame_id(frame);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_3(0.0f);

    mi::math::Color c(0.0f);
    m_frames[frame_id].m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma3(c, m_frames[frame_id].m_gamma);
    return {c.r, c.g, c.b};
}

mi::Float32_4 Texture_3d::texel_float4(const mi::Sint32_3& coord, mi::Float32 frame) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    mi::Size frame_id = get_frame_id(frame);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Float32_4(0.0f);

    mi::math::Color c(0.0f);
    m_frames[frame_id].m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma4(c, m_frames[frame_id].m_gamma);
    return {c.r, c.g, c.b, c.a};
}

mi::Spectrum Texture_3d::texel_color(const mi::Sint32_3& coord, mi::Float32 frame) const
{
    if (!m_is_valid)
        return mi::Spectrum(0.0f);

    mi::Size frame_id = get_frame_id(frame);
    if (frame_id == static_cast<mi::Size>(-1))
        return mi::Spectrum(0.0f);

    mi::math::Color c(0.0f);
    m_frames[frame_id].m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma3(c, m_frames[frame_id].m_gamma);
    return {c.r, c.g, c.b};
}

//-------------------------------------------------------------------------------------------------

Texture_cube::Texture_cube(
    const DB::Typed_tag<TEXTURE::Texture>& tag,
    DB::Transaction* transaction)
{
    if (!tag)
        return;

    DB::Access<TEXTURE::Texture> texture(tag, transaction);
    DB::Tag image_tag = texture->get_image();
    if (!image_tag)
        return;

    DB::Access<DBIMAGE::Image> image(image_tag, transaction);
    m_is_valid = image->is_valid() && !image->is_animated() && !image->is_uvtile();
    if (!m_is_valid)
        return;

    m_gamma = texture->get_effective_gamma(transaction, /*frame_id*/0, /*uvtile_id*/0);
    if (m_gamma <= 0.f)
        m_gamma = 1.f;

    mi::base::Handle<const IMAGE::IMipmap> mipmap(image->get_mipmap(
        transaction, /*frame_id*/0, /*uvtile_id*/0));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(mipmap->get_level(/*level*/0));
    m_canvas = IMAGE::Access_canvas(canvas.get(), true);

    m_resolution = mi::Uint32_3(canvas->get_resolution_x(), canvas->get_resolution_y(), 0);

    if (canvas->get_layers_size() != 6)
        m_is_valid = false;
}

float Texture_cube::lookup_float(const mi::Float32_3& direction) const
{
    return lookup_float4(direction).x;
}

mi::Float32_2 Texture_cube::lookup_float2(const mi::Float32_3& direction) const
{
    const mi::Float32_4& res = lookup_float4(direction);
    return {res.x, res.y};
}

mi::Float32_3 Texture_cube::lookup_float3(const mi::Float32_3& direction) const
{
    const mi::Float32_4& res = lookup_float4(direction);
    return {res.x, res.y, res.z};
}

mi::Float32_4 Texture_cube::lookup_float4(const mi::Float32_3& direction) const
{
    mi::Float32_4 res(0.0f);
    if (!m_is_valid)
        return res;

    // Figure out which face to look up by finding the component of the ray direction vector
    // with the largest magnitude.
    Uint max_comp = (Uint)(fabs(direction.y) > fabs(direction.x));
    if (fabs(direction.z) > fabs(direction[max_comp]))
        max_comp = 2;
    const Uint cube_face = max_comp*2 + (Uint)(direction[max_comp] < 0.0f);
    const Uint lut[6]    = {2, 1, 0, 2, 0, 1};

    // Texture lookup.
    const float inv_ldir = 0.5f/direction[max_comp];
    const mi::Float32_3 coords(
        direction[lut[max_comp*2  ]] *
        (((max_comp  == 0) || (cube_face == 3)) ? -inv_ldir : inv_ldir) + 0.5f,
        direction[lut[max_comp*2+1]] *
        (((cube_face == 0) || (cube_face == 4)) ? -inv_ldir : inv_ldir) + 0.5f,
        0.0f);

    const mi::Float32_4 crop_uv(0.0f, 1.0f, 0.0f, 1.0f);
    const mi::Float32_2 crop_w(0.0f, 1.0f);

    return interpolate_biquintic(
        m_canvas, m_resolution,
        mi::mdl::stdlib::wrap_clamp, mi::mdl::stdlib::wrap_clamp, mi::mdl::stdlib::wrap_repeat,
        crop_uv, crop_w,
        coords, /*smootherstep*/ true, m_gamma, cube_face);
}

mi::Spectrum Texture_cube::lookup_color(const mi::Float32_3& direction) const
{
    const mi::Float32_4& res = lookup_float4(direction);
    return {res.x, res.y, res.z};
}

//-------------------------------------------------------------------------------------------------

Texture_ptex::Texture_ptex(
    const DB::Typed_tag<TEXTURE::Texture>& tag,
    DB::Transaction* transaction)
{
}

float Texture_ptex::lookup_float(int channel) const
{
    // TODO not implemented
    ASSERT(M_BACKENDS, false);
    return 0.f;
}

mi::Float32_2 Texture_ptex::lookup_float2(int channel) const
{
    // TODO not implemented
    ASSERT(M_BACKENDS, false);
    return mi::Float32_2(0.f);
}

mi::Float32_3 Texture_ptex::lookup_float3(int channel) const
{
    // TODO not implemented
    ASSERT(M_BACKENDS, false);
    return mi::Float32_3(0.f);
}

mi::Float32_4 Texture_ptex::lookup_float4(int channel) const
{
    // TODO not implemented
    ASSERT(M_BACKENDS, false);
    return mi::Float32_4(0.f);
}

mi::Spectrum Texture_ptex::lookup_color(int channel) const
{
    // TODO not implemented
    ASSERT(M_BACKENDS, false);
    return mi::Spectrum(0.f);
}

}
}
