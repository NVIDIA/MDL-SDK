/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/iimage.h>
#include <mi/math/color.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/i_image_utilities.h>
#include <io/scene/texture/i_texture.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <base/data/db/i_db_access.h>


namespace MI {
namespace MDLRT {

//-------------------------------------------------------------------------------------------------

static float gamma_func(const float f, const float gamma_val) {
    return f <= 0.0f ? 0.0f : powf(f, gamma_val);
}
static void apply_gamma1(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f)
        c.r = gamma_func(c.r, gamma_val);
}
static void apply_gamma2(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f) {
        c.r = gamma_func(c.r, gamma_val);
        c.g = gamma_func(c.g, gamma_val);
    }
}
static void apply_gamma3(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f) {
        c.r = gamma_func(c.r, gamma_val);
        c.g = gamma_func(c.g, gamma_val);
        c.b = gamma_func(c.b, gamma_val);
    }
}
static void apply_gamma4(mi::math::Color &c, const float gamma_val)
{
    if (gamma_val != 1.0f) {
        c.r = gamma_func(c.r, gamma_val);
        c.g = gamma_func(c.g, gamma_val);
        c.b = gamma_func(c.b, gamma_val);
        c.a = gamma_func(c.a, gamma_val);
    }
}

static float saturate(const float f) {
    return std::max(0.0f, std::min(1.0f, f));
}
static unsigned int float_as_uint(const float f) {
    union {
        float f;
        unsigned int i;
    } u;
    u.f = f;
    return u.i;
}
static int          __float2int_rz( const float f)        { return (int)f; }
static long long    __float2ll_rz(  const float f)        { return (long long)f; }
static long long    __float2ll_rd(  const float f)        { return (long long)floorf(f); }
static float        __uint2float_rn(const unsigned int i) { return (float)i; }
static unsigned int __float2uint_rz(const float f)        { return (unsigned int)f; }

static mi::Uint32_2 texremapll(
    const mi::mdl::stdlib::Tex_wrap_mode wrap_u,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_v,
    const mi::Uint32_2 &texres,
    const mi::Sint32_2 &crop_ofs,
    const mi::Float32_2 &tex)
{
    const long long texix = __float2ll_rz(tex.x);
    const long long texiy = __float2ll_rz(tex.y);

    mi::Sint32_2 texi;

    // early out if in range 0,texres.x-1 // extra _rd cast needed to catch -1..0 case
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

    // early out if in range 0,texres.y-1 // extra _rd cast needed to catch -1..0 case
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

    return mi::Uint32_2(texi.x,texi.y);
}

static unsigned int texremapzll(
    const mi::mdl::stdlib::Tex_wrap_mode wrap_w,
    const unsigned int texresz,
    const int crop_ofsz,
    const float texz)
{
    const long long texiz = __float2ll_rz(texz);

    int texi;

    // early out if in range 0,texres.x-1 // extra _rd cast needed to catch -1..0 case
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


static mi::Float32_4 interpolate_biquintic(
    const MI::IMAGE::Access_canvas &canvas,
    const mi::Uint32_3 &texture_res,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_u,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_v,
    const mi::mdl::stdlib::Tex_wrap_mode wrap_w,
    const mi::Float32_4 &uv_crop,
    const mi::Float32_2 &w_crop,
    const mi::Float32_3 &texo,
    const bool linear,
    const float gamma_val,
    const unsigned int layer_offset = 0)
{
    if (texture_res.x == 0 || texture_res.y == 0)
        return mi::Float32_4(0.0f, 0.0f ,0.0f, 0.0f);

    if(((wrap_u == mi::mdl::stdlib::wrap_clip) && (texo.x < 0.0f || texo.x > 1.0f))
	||
       ((wrap_v == mi::mdl::stdlib::wrap_clip) && (texo.y < 0.0f || texo.y > 1.0f)))

	return mi::Float32_4(0.0f,0.0f,0.0f, 0.0f);

    const mi::Uint32_2 full_texres(texture_res.x, texture_res.y);
    const mi::Sint32_2 crop_ofs(
        __float2int_rz(__uint2float_rn(full_texres.x-1) * uv_crop.x),
        __float2int_rz(__uint2float_rn(full_texres.y-1) * uv_crop.z));

    ASSERT(M_BACKENDS, uv_crop.x >= 0.0f && uv_crop.y >= 0.0f);
    const mi::Uint32_2 texres(
        std::max(__float2uint_rz(__uint2float_rn(texture_res.x) * uv_crop.y),1u),
        std::max(__float2uint_rz(__uint2float_rn(texture_res.y) * uv_crop.w),1u));

    //!! opt.? use floor'ed float values of texres instead of cast?
    const mi::Float32_2 tex(texo.x * __uint2float_rn(texres.x) - 0.5f,
                            texo.y * __uint2float_rn(texres.y) - 0.5f);

    // check for LLONG_MAX as texremapll overflows otherwise
    if((texres.x == 0) || (texres.y == 0) || (((float_as_uint(tex.x))&0x7FFFFFFF) >= 0x5f000000) ||
       (((float_as_uint(tex.y))&0x7FFFFFFF) >= 0x5f000000)) 
	return mi::Float32_4(0.0f,0.0f,0.0f, 0.0f);

    const mi::Uint32_2 texi0 = texremapll(wrap_u, wrap_v, texres, crop_ofs, tex);
    //!! +1 in float can screw-up bilerp
    const mi::Uint32_2 texi1 = texremapll(
        wrap_u, wrap_v, texres, crop_ofs, mi::Float32_2(tex.x+1.0f,tex.y+1.0f));
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
	    return mi::Float32_4(0.0f,0.0f,0.0f, 0.0f);

	const int crop_ofs_z = __float2int_rz(__uint2float_rn(texture_res.z-1) * w_crop.x);

	ASSERT(M_BACKENDS, w_crop.x >= 0.0f && w_crop.y >= 0.0f);
	const unsigned int crop_texres_z = std::max(
            __float2uint_rz(__uint2float_rn(texture_res.z) * w_crop.y), 1u);

	//!! opt.? use floor'ed float values of texres instead of cast?
	const float tex_z = texo.z * __uint2float_rn(crop_texres_z) - 0.5f;

        // check for LLONG_MAX as texremapll overflows otherwise
	if((crop_texres_z == 0) || (((float_as_uint(tex_z))&0x7FFFFFFF) >= 0x5f000000))
	    return mi::Float32_4(0.0f,0.0f,0.0f, 0.0f);

	texi0_z = texremapzll(wrap_w, crop_texres_z, crop_ofs_z, tex_z);
        //!! +1 in float can screw-up bilerp if precision maps it to same texel again
	texi1_z = texremapzll(wrap_w, crop_texres_z, crop_ofs_z, tex_z+1.0f);

	lerp_z = tex_z - floorf(tex_z);
    }

    if(linear == false) {
	lerp.x *= lerp.x*lerp.x*(lerp.x*(lerp.x*6.0f-15.0f)+10.0f); // smootherstep
	lerp.y *= lerp.y*lerp.y*(lerp.y*(lerp.y*6.0f-15.0f)+10.0f);
    }

    const mi::Float32_4 st(
        (1.0f-lerp.x)*(1.0f-lerp.y), lerp.x*(1.0f-lerp.y), (1.0f-lerp.x)*lerp.y, lerp.x*lerp.y);


    mi::Float32_4 rgba(0.f,0.f,0.f,1.f);
    mi::Float32_4 rgba2(0.f,0.f,0.f,1.f);


    bool tex_layer_loop;
    for (unsigned int i = 0; i < 2; ++i)
    {
        const unsigned int z_layer = ((i == 0) ? texi1_z : texi0_z) + layer_offset;
        
        mi::math::Color col(0.f,0.f,0.f,1.f);
        mi::math::Color c0, c1, c2, c3;
        canvas.lookup(c0, texi.x, texi.y, z_layer);
        canvas.lookup(c1, texi.z, texi.y, z_layer);
        canvas.lookup(c2, texi.x, texi.w, z_layer);
        canvas.lookup(c3, texi.z, texi.w, z_layer);
        
        col = c0 * st.x + c1 * st.y + c2 * st.z + c3 * st.w;
        rgba = mi::Float32_4(col.r, col.g, col.b, col.a);
    
        tex_layer_loop = false;

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
        rgba.w = gamma_func(rgba.w, gamma_val);
    }
    return rgba;
}




Texture::Texture(Gamma_mode gamma_mode)
    : m_resolution(0u, 0u, 0u)
    , m_is_valid(false)
    , m_mdl_gamma_mode(gamma_mode)
{
}

//-------------------------------------------------------------------------------------------------


Texture_2d::Texture_2d()
    : Texture(mi::mdl::stdlib::gamma_default)
    , m_is_udim(false)
    , m_udim_num_u(1)
    , m_udim_num_v(1)
    , m_udim_offset_u(0)
    , m_udim_offset_v(0)
{
}


Texture_2d::Texture_2d(
    const DB::Typed_tag<TEXTURE::Texture>& tex_t,
    Gamma_mode gamma_mode,
    bool use_derivatives,
    DB::Transaction* trans)
    : Texture(gamma_mode)
    , m_is_udim(false)
    , m_udim_num_u(1)
    , m_udim_num_v(1)
    , m_udim_offset_u(0)
    , m_udim_offset_v(0)
{
    SYSTEM::Access_module<MI::IMAGE::Image_module> image_module(false);
    DB::Access<TEXTURE::Texture> texture(tex_t, trans);
    if (!texture)
        return;

    DB::Access<DBIMAGE::Image> image(texture->get_image(), trans);
    m_is_valid = image->is_valid();

    DB::Access<DBIMAGE::Image_impl> image_impl;
    if (m_is_valid) {
        image_impl.set(image->get_impl_tag(), trans);
        m_is_valid = image_impl.is_valid();
    }
    if (!m_is_valid)
        return;

    m_is_udim = image->is_uvtile();
    unsigned int num_tiles;
    if (m_is_udim)
    {
        const unsigned int *tm = image_impl->get_tile_mapping(
            m_udim_num_u, m_udim_num_v,
            m_udim_offset_u, m_udim_offset_v);
        const unsigned int s = m_udim_num_u * m_udim_num_v;
        m_udim_mapping.resize(s);
        for (unsigned int i = 0; i < s; ++i)
            m_udim_mapping[i] = tm[i];

        num_tiles = (unsigned int)(image->get_uvtile_length());
    }
    else
    {
        num_tiles = 1;
        m_udim_num_u = m_udim_num_v = 1;
        m_udim_offset_u = m_udim_offset_v = 0;
        m_udim_mapping.push_back(0);
    }

    m_canvases.resize(num_tiles);
    m_gamma.resize(num_tiles);
    m_tile_resolutions.resize(num_tiles);

    for (unsigned int i = 0; i < num_tiles; ++i) {
        mi::base::Handle<const IMAGE::IMipmap> mipmap(image_impl->get_mipmap(i));
        mi::Uint32 num_levels = use_derivatives ? mipmap->get_nlevels() : 1;

        m_canvases[i].resize(num_levels);
        m_tile_resolutions[i].resize(num_levels);

        mi::base::Handle<const mi::neuraylib::ICanvas> base_canvas(mipmap->get_level(0));

        if (i == 0) {
            m_resolution.x = base_canvas->get_resolution_x();
            m_resolution.y = base_canvas->get_resolution_y();
        }

        m_gamma[i] = texture->get_effective_gamma(trans, i);
        if (m_gamma[i] <= 0.f)
            m_gamma[i] = 1.f;

        // for derivative mode, convert to linear first, if necessary.
        // Note: for non-derivative mode, the gamma is still (incorrectly) applied after filtering
        if (use_derivatives && m_gamma[i] != 1.0f) {
            // Choose pixel format. For non-float formats, convert to float format
            // with same number of channels
            MI::IMAGE::Pixel_type pixel_type =
                MI::IMAGE::convert_pixel_type_string_to_enum(base_canvas->get_type());
            switch (pixel_type) {
            case MI::IMAGE::PT_RGB:
            case MI::IMAGE::PT_RGBE:
            case MI::IMAGE::PT_RGB_16:
                pixel_type = MI::IMAGE::PT_RGB_FP;
                break;
            case MI::IMAGE::PT_RGBA:
            case MI::IMAGE::PT_RGBEA:
            case MI::IMAGE::PT_RGBA_16:
                pixel_type = MI::IMAGE::PT_COLOR;
                break;
            case MI::IMAGE::PT_SINT8:
            case MI::IMAGE::PT_SINT32:
                pixel_type = MI::IMAGE::PT_FLOAT32;
                break;
            default:
                break;
            }

            // Copy canvas and adjust gamma from "effective gamma" to 1
            mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
                image_module->convert_canvas(
                    base_canvas.get(),
                    pixel_type));
            gamma_canvas->set_gamma(m_gamma[i]);
            image_module->adjust_gamma(gamma_canvas.get(), 1.0f);
            base_canvas = gamma_canvas;
            m_gamma[i] = 1.0f;
        }

        std::vector< mi::base::Handle<mi::neuraylib::ICanvas> > mipmaps;
        if (use_derivatives)
            image_module->create_mipmaps(mipmaps, base_canvas.get(), 1.0f);

        for (mi::Uint32 level = 0; level < num_levels; ++level) {
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas;
            if (level == 0) canvas = base_canvas;
            else canvas = mipmaps[level - 1];

            m_canvases[i][level] = IMAGE::Access_canvas(canvas.get(), true);

            m_tile_resolutions[i][level] = mi::Uint32_3(
                canvas->get_resolution_x(),
                canvas->get_resolution_y(), 0);
        }
    }
}

Texture_2d::~Texture_2d()
{
}

mi::Sint32_2 Texture_2d::get_resolution(const mi::Sint32_2& uv_tile) const
{
    const unsigned int tile_id = get_tile_id(uv_tile.x, uv_tile.y);
    if (tile_id == ~0u)
        return mi::Sint32_2(0);
    return mi::Sint32_2(m_tile_resolutions[tile_id][0].x, m_tile_resolutions[tile_id][0].y);
}

float Texture_2d::texel_float(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const
{
    const unsigned int tile_id = get_tile_id(uv_tile.x, uv_tile.y);
    if (tile_id == ~0u)
        return 0.0f;
    
    mi::math::Color res(0.0f);
    m_canvases[tile_id][0].lookup(res,coord.x,coord.y,0);
    apply_gamma1(res, m_gamma[tile_id]);
    return res.r;
}


mi::Float32_2 Texture_2d::texel_float2(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile) const
{
    const unsigned int tile_id = get_tile_id(uv_tile.x, uv_tile.y);
    if (tile_id == ~0u)
        return mi::Float32_2(0.0f);

    mi::math::Color res(0.0f);
    m_canvases[tile_id][0].lookup(res,coord.x,coord.y,0);
    apply_gamma2(res, m_gamma[tile_id]);
    return mi::Float32_2(res.r,res.g);
}


mi::Float32_3 Texture_2d::texel_float3(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile) const
{
    const unsigned int tile_id = get_tile_id(uv_tile.x, uv_tile.y);
    if (tile_id == ~0u)
        return mi::Float32_3(0.0f);

    mi::math::Color res(0.0f);
    m_canvases[tile_id][0].lookup(res,coord.x,coord.y,0);
    apply_gamma3(res, m_gamma[tile_id]);
    return mi::Float32_3(res.r,res.g,res.b);
}


mi::Float32_4 Texture_2d::texel_float4(
    const mi::Sint32_2& coord,
    const mi::Sint32_2& uv_tile) const
{
    const unsigned int tile_id = get_tile_id(uv_tile.x, uv_tile.y);
    if (tile_id == ~0u)
        return mi::Float32_4(0.0f);

    mi::math::Color res(0.0f);
    m_canvases[tile_id][0].lookup(res,coord.x,coord.y,0);
    apply_gamma4(res, m_gamma[tile_id]);
    return mi::Float32_4(res.r,res.g,res.b,res.a);
}


mi::Spectrum Texture_2d::texel_color(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const
{
    const unsigned int tile_id = get_tile_id(uv_tile.x, uv_tile.y);
    if (tile_id == ~0u)
        return mi::Spectrum(0.0f);

    mi::math::Color res(0.0f);
    m_canvases[tile_id][0].lookup(res,coord.x,coord.y,0);
    apply_gamma3(res, m_gamma[tile_id]);
    return mi::Spectrum(res.r,res.g,res.b);
}


float Texture_2d::lookup_float(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v
        ) const
{
    return lookup_float4(coord,wrap_u,wrap_v,crop_u,crop_v).x;
}


mi::Float32_2 Texture_2d::lookup_float2(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,crop_u,crop_v);
    return mi::Float32_2(res.x,res.y);
}


mi::Float32_3 Texture_2d::lookup_float3(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,crop_u,crop_v);
    return mi::Float32_3(res.x,res.y,res.z);
}


mi::Float32_4 Texture_2d::lookup_float4(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v
        ) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    const mi::Float32_4 uv_crop(
        saturate(crop_u.x), saturate(crop_u.y - crop_u.x),
        saturate(crop_v.x), saturate(crop_v.y - crop_v.x));
    const mi::Float32_2 w_crop(0.f, 1.f);

    mi::Float32_3 coords(coord.x, coord.y, 0.0f);
    unsigned int tile_id = 0;
    if (m_is_udim)
    {
        coords.x += (float)m_udim_offset_u;
        coords.y += (float)m_udim_offset_v;
        if (coords.x < 0.0f || coords.y < 0.0f)
            return mi::Float32_4(0.0f);

        const unsigned int tu = (unsigned int)(coords.x);
        const unsigned int tv = (unsigned int)(coords.y);
        
        if (tu >= m_udim_num_u || tv >= m_udim_num_v)
            return mi::Float32_4(0.0f);

        tile_id = m_udim_mapping[tv * m_udim_num_u + tu];
        if (tile_id == ~0u)
            return mi::Float32_4(0.0f);

        coords.x -= floorf(coords.x);
        coords.y -= floorf(coords.y);
    }

    return interpolate_biquintic(
        m_canvases[tile_id][0],
        m_tile_resolutions[tile_id][0],
        wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
        uv_crop, w_crop,
        coords, false, m_gamma[tile_id]);
}

mi::Float32_4 Texture_2d::lookup_deriv_float4(
        const mi::Float32_2& coord_val,
        const mi::Float32_2& coord_dx,
        const mi::Float32_2& coord_dy,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v
        ) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    const mi::Float32_4 uv_crop(
        saturate(crop_u.x), saturate(crop_u.y - crop_u.x),
        saturate(crop_v.x), saturate(crop_v.y - crop_v.x));
    const mi::Float32_2 w_crop(0.f, 1.f);

    mi::Float32_3 coords(coord_val.x, coord_val.y, 0.0f);
    unsigned int tile_id = 0;
    if (m_is_udim)
    {
        coords.x += (float)m_udim_offset_u;
        coords.y += (float)m_udim_offset_v;
        if (coords.x < 0.0f || coords.y < 0.0f)
            return mi::Float32_4(0.0f);

        const unsigned int tu = (unsigned int)(coords.x);
        const unsigned int tv = (unsigned int)(coords.y);

        if (tu >= m_udim_num_u || tv >= m_udim_num_v)
            return mi::Float32_4(0.0f);

        tile_id = m_udim_mapping[tv * m_udim_num_u + tu];
        if (tile_id == ~0u)
            return mi::Float32_4(0.0f);

        coords.x -= floorf(coords.x);
        coords.y -= floorf(coords.y);
    }

    unsigned int num_levels = (unsigned int)(m_canvases[tile_id].size());

    // isotropic filtering
    float dx_len_sqr = coord_dx.x * coord_dx.x + coord_dx.y * coord_dx.y;
    float dy_len_sqr = coord_dy.x * coord_dy.x + coord_dy.y * coord_dy.y;
    float max_len_sqr = std::max(dx_len_sqr, dy_len_sqr);
    float level = num_levels - 1 + 0.5f * std::log2f(std::max(max_len_sqr, 1e-8f));

    if (level < 0) {
        return interpolate_biquintic(
            m_canvases[tile_id][0],
            m_tile_resolutions[tile_id][0],
            wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
            uv_crop, w_crop,
            coords, false, 1.0f);
    } else if (level >= num_levels - 1) {
        // just read the single pixel of the smallest mipmap
        mi::math::Color col;
        m_canvases[tile_id][num_levels - 1].lookup(col, 0, 0);
        mi::Float32_4 rgba(col.r, col.g, col.b, col.a);
        return rgba;
    } else {
        // do trilinear filtering between the two mipmap levels
        unsigned int level_a = (unsigned int) floorf(level);
        float lerp = level - level_a;

        mi::Float32_4 rgba_0 = interpolate_biquintic(
            m_canvases[tile_id][level_a],
            m_tile_resolutions[tile_id][level_a],
            wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
            uv_crop, w_crop,
            coords, false, 1.0f);

        mi::Float32_4 rgba_1 = interpolate_biquintic(
            m_canvases[tile_id][level_a + 1],
            m_tile_resolutions[tile_id][level_a + 1],
            wrap_u, wrap_v, mi::mdl::stdlib::wrap_repeat,
            uv_crop, w_crop,
            coords, false, 1.0f);

        return (1 - lerp) * rgba_0 + lerp * rgba_1;
    }
}

mi::Spectrum Texture_2d::lookup_color(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,crop_u,crop_v);
    return mi::Spectrum(res.x,res.y,res.z);
}



//-------------------------------------------------------------------------------------------------



Texture_3d::Texture_3d()
    : Texture(mi::mdl::stdlib::gamma_default)
    , m_gamma(0.0f)
{
}

Texture_3d::~Texture_3d()
{
}


Texture_3d::Texture_3d(
    const DB::Typed_tag<TEXTURE::Texture>& tex_t,
    Gamma_mode gamma_mode,
    DB::Transaction* trans)
    : Texture(mi::mdl::stdlib::gamma_default)
{
    DB::Access<TEXTURE::Texture> texture(tex_t, trans);
    if (!texture)
        return;

    m_gamma = texture->get_effective_gamma(trans);
    if (m_gamma <= 0.f)
        m_gamma = 1.f;

    DB::Access<DBIMAGE::Image> image(texture->get_image(), trans);
    m_is_valid = image->is_valid();

    DB::Access<DBIMAGE::Image_impl> image_impl;
    if (m_is_valid)
        image_impl.set(image->get_impl_tag(), trans);

    mi::base::Handle<const IMAGE::IMipmap> mipmap(image->get_mipmap(trans));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0 ));
    m_canvas = IMAGE::Access_canvas(canvas.get(), true);
    m_resolution = mi::Uint32_3(
        canvas->get_resolution_x(),
        canvas->get_resolution_y(),
        canvas->get_layers_size());
}


float Texture_3d::lookup_float(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,wrap_w,crop_u,crop_v,crop_w);
    return res.x;
}


mi::Float32_2 Texture_3d::lookup_float2(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,wrap_w,crop_u,crop_v,crop_w);
    return mi::Float32_2(res.x, res.y);
}


mi::Float32_3 Texture_3d::lookup_float3(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,wrap_w,crop_u,crop_v,crop_w);
    return mi::Float32_3(res.x, res.y, res.z);
}


mi::Float32_4 Texture_3d::lookup_float4(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w
        ) const
{
    if (!m_is_valid)
        return mi::Float32_4(0.0f);

    const mi::Float32_4 uv_crop(
        saturate(crop_u.x), saturate(crop_u.y - crop_u.x),
        saturate(crop_v.x), saturate(crop_v.y - crop_v.x));
    const mi::Float32_2 w_crop(saturate(crop_w.x), saturate(crop_w.y - crop_w.x));

    return interpolate_biquintic(
        m_canvas,
        m_resolution,
        wrap_u, wrap_v, wrap_w,
        uv_crop, w_crop,
        coord, true, m_gamma);
}


mi::Spectrum Texture_3d::lookup_color(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w
        ) const
{
    const mi::Float32_4& res = lookup_float4(coord,wrap_u,wrap_v,wrap_w,crop_u,crop_v,crop_w);
    return mi::Spectrum(res.x, res.y, res.z);
}


float Texture_3d::texel_float(const mi::Sint32_3& coord) const
{
    if (!m_is_valid)
        return 0.0f;

    mi::math::Color c(0.0f);
    m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma1(c, m_gamma);

    return c.r;
}


mi::Float32_2 Texture_3d::texel_float2(const mi::Sint32_3& coord) const
{
    if (!m_is_valid)
        return mi::Float32_2(0.0f);

    mi::math::Color c(0.0f);
    m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma2(c, m_gamma);
    return mi::Float32_2(c.r, c.g);
}


mi::Float32_3 Texture_3d::texel_float3(const mi::Sint32_3& coord) const
{
    if (!m_is_valid)
        return mi::Float32_3(0.0f);

    mi::math::Color c(0.0f);
    m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma3(c, m_gamma);
    return mi::Float32_3(c.r, c.g, c.b);
}


mi::Float32_4 Texture_3d::texel_float4(const mi::Sint32_3& coord) const
{
    mi::math::Color c(0.0f);
    m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma4(c, m_gamma);
    return mi::Float32_4(c.r, c.g, c.b, c.a);
}


mi::Spectrum Texture_3d::texel_color(const mi::Sint32_3& coord) const
{
    if (!m_is_valid)
        return mi::Spectrum(0.0f);

    mi::math::Color c(0.0f);
    m_canvas.lookup(c, coord.x, coord.y, coord.z);
    apply_gamma3(c, m_gamma);
    return mi::Spectrum(c.r, c.g, c.b);
}


//-------------------------------------------------------------------------------------------------



Texture_cube::Texture_cube()
    : Texture(mi::mdl::stdlib::gamma_default)
    , m_gamma(0.0f)
{
}


Texture_cube::Texture_cube(
    const DB::Typed_tag<TEXTURE::Texture>& tex_t,
    Gamma_mode gamma_mode,
    DB::Transaction* trans)
    : Texture(mi::mdl::stdlib::gamma_default)
{
    DB::Access<TEXTURE::Texture> texture(tex_t, trans);
    if (!texture)
        return;

    m_gamma = texture->get_effective_gamma(trans);
    if (m_gamma <= 0.f)
        m_gamma = 1.f;

    DB::Access<DBIMAGE::Image> image(texture->get_image(), trans);
    m_is_valid = image->is_valid();

    mi::base::Handle<const IMAGE::IMipmap> mipmap( image->get_mipmap(trans) );
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0 ));
    m_canvas = IMAGE::Access_canvas(canvas.get(), true);
    m_resolution = mi::Uint32_3(
        canvas->get_resolution_x(),
        canvas->get_resolution_y(),
        0);

    if (canvas->get_layers_size() != 6) {
        m_is_valid = false;
    }
}


float Texture_cube::lookup_float(const mi::Float32_3& direction) const
{
    return lookup_float4(direction).x;
}


mi::Float32_2 Texture_cube::lookup_float2(const mi::Float32_3& direction) const
{
    const mi::Float32_4& res = lookup_float4(direction);
    return mi::Float32_2(res.x,res.y);
}


mi::Float32_3 Texture_cube::lookup_float3(const mi::Float32_3& direction) const
{
    const mi::Float32_4& res = lookup_float4(direction);
    return mi::Float32_3(res.x,res.y,res.z);
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
    const Uint lut[6]    = {2,1, 0,2, 0,1};

    // Texture lookup.
    const float inv_ldir = 0.5f/direction[max_comp];
    const mi::Float32_3 coords(
        direction[lut[max_comp*2  ]] *
        (((max_comp  == 0) || (cube_face == 3)) ? -inv_ldir : inv_ldir) + 0.5f,
        direction[lut[max_comp*2+1]] *
        (((cube_face == 0) || (cube_face == 4)) ? -inv_ldir : inv_ldir) + 0.5f,
        0.0f);

    return interpolate_biquintic(
        m_canvas, m_resolution,
        mi::mdl::stdlib::wrap_clamp, mi::mdl::stdlib::wrap_clamp, mi::mdl::stdlib::wrap_repeat,
        mi::Float32_4(0.0f, 1.0f, 0.0f, 1.0f), mi::Float32_2(0.0f, 1.0f),
        coords, false, m_gamma, cube_face);
}


mi::Spectrum Texture_cube::lookup_color(const mi::Float32_3& direction) const
{
    const mi::Float32_4& res = lookup_float4(direction);
    return mi::Spectrum(res.x,res.y,res.z);
}



//-------------------------------------------------------------------------------------------------



Texture_ptex::Texture_ptex() : Texture(mi::mdl::stdlib::gamma_default)
{
}


Texture_ptex::Texture_ptex(
    const DB::Typed_tag<TEXTURE::Texture>& tex_t,
    Gamma_mode gamma_mode,
    DB::Transaction* trans)
    : Texture(mi::mdl::stdlib::gamma_default)
{
}


float Texture_ptex::lookup_float(int channel) const
{
    // TODO
    return 0.f;
}


mi::Float32_2 Texture_ptex::lookup_float2(int channel) const
{
    // TODO
    return mi::Float32_2(0.f);
}


mi::Float32_3 Texture_ptex::lookup_float3(int channel) const
{
    // TODO
    return mi::Float32_3(0.f);
}


mi::Float32_4 Texture_ptex::lookup_float4(int channel) const
{
    // TODO
    return mi::Float32_4(0.f);
}


mi::Spectrum Texture_ptex::lookup_color(int channel) const
{
    // TODO
    return mi::Spectrum(0.f);
}



}}
