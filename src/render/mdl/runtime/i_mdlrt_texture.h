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

#ifndef RENDER_MDL_RUNTIME_I_MDLRT_TEXTURE_H
#define RENDER_MDL_RUNTIME_I_MDLRT_TEXTURE_H

#include <mi/neuraylib/typedefs.h>
#include <mi/mdl/mdl_stdlib_types.h>

#include <io/scene/texture/i_texture.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/image/image/i_image_access_canvas.h>

#include <map>
#include <vector>

namespace MI {
namespace MDLRT {

class Texture
{
public:
    typedef mi::mdl::stdlib::Tex_wrap_mode  Wrap_mode;

    bool is_valid() const { return m_is_valid; }

    mi::Uint32_2 get_first_last_frame() const { return m_first_last_frame; }

protected:
    // Returns the (rounded-down) frame ID for the \p frame parameter.
    //
    // Checks whether there is a frame with frame number 'floor(frame)' and returns its index.
    // Otherwise returns -1.
    mi::Size get_frame_id(mi::Float32 frame) const;

    // Returns the enclosing frame IDs for the \p frame parameter.
    //
    // Checks whether there are frames with frame number 'floor(frame)' and 'ceil(frame)' and
    // returns their indices. Otherwise returns (-1, -1).
    std::pair<mi::Size, mi::Size> get_frame_ids(mi::Float32 frame) const;

    bool m_is_valid = false;
    bool m_is_animated = false;

    mi::Uint32_2 m_first_last_frame{0, 0};

    // Maps frame numbers to frame IDs.
    std::map<mi::Size, mi::Size> m_frame_number_to_id;
};

class Texture_2d : public Texture
{
public:
    Texture_2d(
        const DB::Typed_tag<TEXTURE::Texture>& tag,
        bool use_derivatives,
        DB::Transaction* transaction);

    mi::Uint32_2 get_resolution(const mi::Sint32_2& uv_tile, mi::Float32 frame) const;

    float lookup_float(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        mi::Float32 frame) const;

    mi::Float32_2 lookup_float2(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        mi::Float32 frame) const;

    mi::Float32_3 lookup_float3(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        mi::Float32 frame) const;

    mi::Float32_4 lookup_float4(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        mi::Float32 frame) const;

    mi::Float32_4 lookup_deriv_float4(
        const mi::Float32_2& coord_val,
        const mi::Float32_2& coord_dx,
        const mi::Float32_2& coord_dy,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        mi::Float32 frame) const;

    mi::Spectrum lookup_color(
        const mi::Float32_2& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        mi::Float32 frame) const;

    float texel_float(
        const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile, mi::Float32 frame) const;
    mi::Float32_2 texel_float2(
        const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile, mi::Float32 frame) const;
    mi::Float32_3 texel_float3(
        const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile, mi::Float32 frame) const;
    mi::Float32_4 texel_float4(
        const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile, mi::Float32 frame) const;
    mi::Spectrum texel_color(
        const mi::Sint32_2& coord, const mi::Sint32_2& uv_tile, mi::Float32 frame) const;

private:
    // Used to implement lookup_float4(). Frame/UV IDs as parameters.
    mi::Float32_4 lookup_float4_frame(
        const mi::Float32_3& coords,
        mi::mdl::stdlib::Tex_wrap_mode wrap_u,
        mi::mdl::stdlib::Tex_wrap_mode wrap_v,
        mi::mdl::stdlib::Tex_wrap_mode wrap_w,
        const mi::Float32_4& crop_uv,
        const mi::Float32_2& crop_w,
        mi::Size frame_id,
        mi::Uint32 u,
        mi::Uint32 v) const;

    // Used to implement lookup_deriv_float4(). Frame/UV IDs as parameters.
    mi::Float32_4 lookup_deriv_float4_frame(
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
        mi::Uint32 v) const;

    bool m_use_derivatives;
    bool m_is_uvtile;

    struct Uvtile {
        // Vector of mipmap levels. Only one element if \c m_use_derivatives is \c false.
        std::vector<IMAGE::Access_canvas> m_canvas;
        std::vector<mi::Uint32_3> m_resolution;
        float m_gamma;
    };

    struct Frame {
        std::vector<Uvtile> m_uvtiles;
        DBIMAGE::Uv_to_id m_uv_to_id;
    };

    std::vector<Frame> m_frames;
};

// Textures with uvtiles are treated as invalid textures.
class Texture_3d : public Texture
{
public:
    Texture_3d(
        const DB::Typed_tag<TEXTURE::Texture>& tag,
        DB::Transaction* transaction);

    mi::Uint32_3 get_resolution(mi::Float32 frame) const;

    float lookup_float(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w,
        mi::Float32 frame) const;

    mi::Float32_2 lookup_float2(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w,
        mi::Float32 frame) const;

    mi::Float32_3 lookup_float3(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w,
        mi::Float32 frame) const;

    mi::Float32_4 lookup_float4(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w,
        mi::Float32 frame) const;

    mi::Spectrum lookup_color(
        const mi::Float32_3& coord,
        Wrap_mode wrap_u,
        Wrap_mode wrap_v,
        Wrap_mode wrap_w,
        const mi::Float32_2& crop_u,
        const mi::Float32_2& crop_v,
        const mi::Float32_2& crop_w,
        mi::Float32 frame) const;

    float texel_float(const mi::Sint32_3& coord, mi::Float32 frame) const;
    mi::Float32_2 texel_float2(const mi::Sint32_3& coord, mi::Float32 frame) const;
    mi::Float32_3 texel_float3(const mi::Sint32_3& coord, mi::Float32 frame) const;
    mi::Float32_4 texel_float4(const mi::Sint32_3& coord, mi::Float32 frame) const;
    mi::Spectrum texel_color(const mi::Sint32_3& coord, mi::Float32 frame) const;

private:
    struct Frame {
        IMAGE::Access_canvas m_canvas;
        mi::Uint32_3 m_resolution;
        float m_gamma;
    };

    std::vector<Frame> m_frames;
};

// Animated textures, textures with uvtiles, and canvases with not exactly 6 layers are treated
// as invalid textures.
class Texture_cube : public Texture
{
public:
    Texture_cube(
        const DB::Typed_tag<TEXTURE::Texture>& tag,
        DB::Transaction* transaction);

    float lookup_float(const mi::Float32_3& coord) const;
    mi::Float32_2 lookup_float2(const mi::Float32_3& coord) const;
    mi::Float32_3 lookup_float3(const mi::Float32_3& coord) const;
    mi::Float32_4 lookup_float4(const mi::Float32_3& coord) const;
    mi::Spectrum lookup_color(const mi::Float32_3& coord) const;

private:
    IMAGE::Access_canvas m_canvas;
    mi::Uint32_3 m_resolution;
    float m_gamma;
};

// Not implemented, lookup functions return zero.
class Texture_ptex : public Texture
{
public:
    Texture_ptex(
        const DB::Typed_tag<TEXTURE::Texture>& tag,
        DB::Transaction* transaction);

    float lookup_float(int channel) const;
    mi::Float32_2 lookup_float2(int channel) const;
    mi::Float32_3 lookup_float3(int channel) const;
    mi::Float32_4 lookup_float4(int channel) const;
    mi::Spectrum lookup_color(int channel) const;
};

}
}

#endif //RENDER_MDL_RUNTIME_I_MDLRT_TEXTURE_H
