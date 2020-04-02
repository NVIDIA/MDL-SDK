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

#ifndef RENDER_MDL_RUNTIME_I_MDLRT_TEXTURE_H
#define RENDER_MDL_RUNTIME_I_MDLRT_TEXTURE_H

#include <mi/neuraylib/typedefs.h>
#include <mi/mdl/mdl_stdlib_types.h>

#include <io/scene/texture/i_texture.h>
#include <io/image/image/i_image_access_canvas.h>


namespace MI {
namespace MDLRT {

class Texture
{
public:
    typedef mi::mdl::stdlib::Tex_wrap_mode  Wrap_mode;
    typedef mi::mdl::stdlib::Tex_gamma_mode Gamma_mode;

    Texture(Gamma_mode);

    int get_width() const { return(int) m_resolution.x; }


    int get_height() const { return (int)m_resolution.y; }


    int get_depth() const { return (int)m_resolution.z; }

    bool is_valid() const { return m_is_valid; }

    Gamma_mode get_mdl_gamma() const { return m_mdl_gamma_mode; }

protected:
    mi::Uint32_3 m_resolution;
    bool         m_is_valid;
    Gamma_mode   m_mdl_gamma_mode;
};

    
class Texture_2d : public Texture
{
public:
    Texture_2d();
    ~Texture_2d();


    Texture_2d(const DB::Typed_tag<TEXTURE::Texture>&, Gamma_mode, bool, DB::Transaction*);

    mi::Sint32_2 get_resolution(const mi::Sint32_2& uv_tile) const;

    float lookup_float(
            const mi::Float32_2& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v
            ) const;


    mi::Float32_2 lookup_float2(
            const mi::Float32_2& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v
            ) const;


    mi::Float32_3 lookup_float3(
            const mi::Float32_2& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v
            ) const;


    mi::Float32_4 lookup_float4(
            const mi::Float32_2& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v
            ) const;


    mi::Float32_4 lookup_deriv_float4(
            const mi::Float32_2& coord_val,
            const mi::Float32_2& coord_dx,
            const mi::Float32_2& coord_dy,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v
            ) const;


    mi::Spectrum lookup_color(
            const mi::Float32_2& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v
            ) const;


    float texel_float(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const;


    mi::Float32_2 texel_float2(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const;


    mi::Float32_3 texel_float3(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const;


    mi::Float32_4 texel_float4(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const;


    mi::Spectrum texel_color(const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile) const;

private:
    unsigned int get_tile_id(int tile_u, int tile_v) const {
        if (!m_is_udim)
            return 0;
        tile_u += m_udim_offset_u;
        tile_v += m_udim_offset_v;
        if ((unsigned int)tile_u >= m_udim_num_u ||
            (unsigned int)tile_v >= m_udim_num_v)
            return ~0u;
        else
            return m_udim_mapping[tile_v * m_udim_num_u + tile_u];
    }

    std::vector< std::vector<mi::Uint32_3> >          m_tile_resolutions;
    std::vector< std::vector<IMAGE::Access_canvas> >  m_canvases;
    std::vector<float>                                  m_gamma;
    std::vector<unsigned int>                           m_udim_mapping;
    bool m_is_udim;
    unsigned int  m_udim_num_u;
    unsigned int  m_udim_num_v;
    int m_udim_offset_u;
    int m_udim_offset_v;                
};


class Texture_3d : public Texture
{
public:
    Texture_3d();
    ~Texture_3d();


    Texture_3d(const DB::Typed_tag<TEXTURE::Texture>&, Gamma_mode, DB::Transaction*);


    float lookup_float(
            const mi::Float32_3& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            Wrap_mode wrap_w,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v,
            const mi::Float32_2& crop_w
            ) const;


    mi::Float32_2 lookup_float2(
            const mi::Float32_3& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            Wrap_mode wrap_w,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v,
            const mi::Float32_2& crop_w
            ) const;


    mi::Float32_3 lookup_float3(
            const mi::Float32_3& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            Wrap_mode wrap_w,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v,
            const mi::Float32_2& crop_w
            ) const;


    mi::Float32_4 lookup_float4(
            const mi::Float32_3& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            Wrap_mode wrap_w,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v,
            const mi::Float32_2& crop_w
            ) const;


    mi::Spectrum lookup_color(
            const mi::Float32_3& coord,
            Wrap_mode wrap_u,
            Wrap_mode wrap_v,
            Wrap_mode wrap_w,
            const mi::Float32_2& crop_u,
            const mi::Float32_2& crop_v,
            const mi::Float32_2& crop_w
            ) const;

    float texel_float(const mi::Sint32_3& coord) const;


    mi::Float32_2 texel_float2(const mi::Sint32_3& coord) const;


    mi::Float32_3 texel_float3(const mi::Sint32_3& coord) const;


    mi::Float32_4 texel_float4(const mi::Sint32_3& coord) const;


    mi::Spectrum texel_color(const mi::Sint32_3& coord) const;

private:
    IMAGE::Access_canvas        m_canvas;
    float                       m_gamma;

};



class Texture_cube : public Texture
{
public:
    Texture_cube();


    Texture_cube(const DB::Typed_tag<TEXTURE::Texture>&, Gamma_mode, DB::Transaction*);


    float lookup_float(const mi::Float32_3& coord) const;


    mi::Float32_2 lookup_float2(const mi::Float32_3& coord) const;


    mi::Float32_3 lookup_float3(const mi::Float32_3& coord) const;


    mi::Float32_4 lookup_float4(const mi::Float32_3& coord) const;


    mi::Spectrum lookup_color(const mi::Float32_3& coord) const;

private:
    IMAGE::Access_canvas        m_canvas;
    float                       m_gamma;
};



class Texture_ptex : public Texture
{
public:
    Texture_ptex();


    Texture_ptex(const DB::Typed_tag<TEXTURE::Texture>&, Gamma_mode, DB::Transaction*);


    float lookup_float(int channel) const;


    mi::Float32_2 lookup_float2(int channel) const;


    mi::Float32_3 lookup_float3(int channel) const;


    mi::Float32_4 lookup_float4(int channel) const;


    mi::Spectrum lookup_color(int channel) const;
};


}}

#endif //RENDER_MDL_RUNTIME_I_MDLRT_TEXTURE_H
