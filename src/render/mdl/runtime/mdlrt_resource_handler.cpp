/***************************************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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
///
/// \file
/// \brief Functionality to handle textures in jitted code
///

#include "pch.h"
#include "i_mdlrt_resource_handler.h"

#include <render/mdl/runtime/i_mdlrt_texture.h>
#include <render/mdl/runtime/i_mdlrt_light_profile.h>
#include <render/mdl/runtime/i_mdlrt_bsdf_measurement.h>

namespace MI {
namespace MDLRT {

size_t Resource_handler::get_data_size() const
{
    size_t size = sizeof(Texture_2d);
    if (size < sizeof(Texture_3d))
        size = sizeof(Texture_3d);
    if (size < sizeof(Texture_cube))
        size = sizeof(Texture_cube);
    if (size < sizeof(Texture_ptex))
        size = sizeof(Texture_ptex);
    if (size < sizeof(Light_profile))
        size = sizeof(Light_profile);
    if (size < sizeof(Bsdf_measurement))
        size = sizeof(Bsdf_measurement);
    return size;
}

void Resource_handler::tex_init(
    void                                *data,
    mi::mdl::IType_texture::Shape       shape,
    unsigned                            tag_v,
    mi::mdl::IValue_texture::gamma_mode gamma,
    void                               *ctx)
{
    DB::Tag                         tag(tag_v);
    DB::Typed_tag<TEXTURE::Texture> typed_tag(tag);

    switch (shape) {
    case mi::mdl::IType_texture::TS_2D:
        new (data) Texture_2d(typed_tag, m_use_derivatives, (DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_3D:
        new (data) Texture_3d(typed_tag, (DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_CUBE:
        new (data) Texture_cube(typed_tag, (DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_PTEX:
        new (data) Texture_ptex(typed_tag, (DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_BSDF_DATA:
        // handle like 3D texture
        new (data) Texture_3d(typed_tag, (DB::Transaction *)ctx);
        break;
    }
}

void Resource_handler::tex_term(
    void                          *data,
    mi::mdl::IType_texture::Shape shape)
{
    switch (shape) {
    case mi::mdl::IType_texture::TS_2D:
        {
            Texture_2d *o = reinterpret_cast<Texture_2d *>(data);
            o->~Texture_2d();
            break;
        }
    case mi::mdl::IType_texture::TS_3D:
        {
            Texture_3d *o = reinterpret_cast<Texture_3d *>(data);
            o->~Texture_3d();
            break;
        }
    case mi::mdl::IType_texture::TS_CUBE:
        {
            Texture_cube *o = reinterpret_cast<Texture_cube *>(data);
            o->~Texture_cube();
            break;
        }
    case mi::mdl::IType_texture::TS_PTEX:
        {
            Texture_ptex *o = reinterpret_cast<Texture_ptex *>(data);
            o->~Texture_ptex();
            break;
        }
    case mi::mdl::IType_texture::TS_BSDF_DATA:
        {
            // handle like 3D texture
            Texture_3d *o = reinterpret_cast<Texture_3d *>(data);
            o->~Texture_3d();
            break;
        }
    }
}

void Resource_handler::tex_resolution_2d(
    int           result[2],
    void const    *tex_data,
    int const     uv_tile[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);
    mi::Uint32_2 res = o->get_resolution(*reinterpret_cast<mi::Sint32_2 const *>(uv_tile), frame);
    result[0] = res.x;
    result[1] = res.y;
}

void Resource_handler::tex_resolution_3d(
    int           result[3],
    void const    *tex_data,
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);
    mi::Uint32_3 res = o->get_resolution(frame);
    result[0] = res.x;
    result[1] = res.y;
    result[2] = res.z;
}

float Resource_handler::tex_lookup_float_2d(
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    return o->lookup_float(
        *reinterpret_cast<mi::Float32_2 const *>(coord),
        Texture::Wrap_mode(wrap_u),
        Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
        frame);
}

float Resource_handler::tex_lookup_deriv_float_2d(
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2],
    float              frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    return o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        Texture::Wrap_mode(wrap_u),
        Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
        frame).x;
}

float Resource_handler::tex_lookup_float_3d(
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const   crop_u[2],
    float const   crop_v[2],
    float const   crop_w[2],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    return o->lookup_float(
        *reinterpret_cast<mi::Float32_3 const *>(coord),
        Texture::Wrap_mode(wrap_u),
        Texture::Wrap_mode(wrap_v),
        Texture::Wrap_mode(wrap_w),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_w),
        frame);
}

float Resource_handler::tex_lookup_float_cube(
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    Texture_cube const *o = reinterpret_cast<Texture_cube const *>(tex_data);

    return o->lookup_float(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

float Resource_handler::tex_lookup_float_ptex(
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    Texture_ptex const *o = reinterpret_cast<Texture_ptex const *>(tex_data);

    return o->lookup_float(channel);
}

void Resource_handler::tex_lookup_float2_2d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2_struct*>(result) =
        o->lookup_float2(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            Texture::Wrap_mode(wrap_u),
            Texture::Wrap_mode(wrap_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            frame);
}

void Resource_handler::tex_lookup_deriv_float2_2d(
    float              result[2],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2],
    float              frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    mi::Float32_4 res = o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        Texture::Wrap_mode(wrap_u),
        Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
        frame);

    result[0] = res.x;
    result[1] = res.y;
}

void Resource_handler::tex_lookup_float2_3d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const   crop_u[2],
    float const   crop_v[2],
    float const   crop_w[2],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2_struct*>(result) =
        o->lookup_float2(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            Texture::Wrap_mode(wrap_u),
            Texture::Wrap_mode(wrap_v),
            Texture::Wrap_mode(wrap_w),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w),
            frame);
}

void Resource_handler::tex_lookup_float2_cube(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    Texture_cube const *o = reinterpret_cast<Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_2_struct*>(result) =
        o->lookup_float2(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

void Resource_handler::tex_lookup_float2_ptex(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    Texture_ptex const *o = reinterpret_cast<Texture_ptex const *>(tex_data);

    *reinterpret_cast<mi::Float32_2_struct*>(result) = o->lookup_float2(channel);
}

void Resource_handler::tex_lookup_float3_2d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->lookup_float3(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            Texture::Wrap_mode(wrap_u),
            Texture::Wrap_mode(wrap_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            frame);
}

void Resource_handler::tex_lookup_deriv_float3_2d(
    float              result[3],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    mi::Float32_4 res = o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        Texture::Wrap_mode(wrap_u),
        Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
         frame);
    result[0] = res.x;
    result[1] = res.y;
    result[2] = res.z;
}

void Resource_handler::tex_lookup_float3_3d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const   crop_u[2],
    float const   crop_v[2],
    float const   crop_w[2],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->lookup_float3(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            Texture::Wrap_mode(wrap_u),
            Texture::Wrap_mode(wrap_v),
            Texture::Wrap_mode(wrap_w),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w),
            frame);
}

void Resource_handler::tex_lookup_float3_cube(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    Texture_cube const *o = reinterpret_cast<Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->lookup_float3(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

void Resource_handler::tex_lookup_float3_ptex(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    Texture_ptex const *o = reinterpret_cast<Texture_ptex const *>(tex_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) = o->lookup_float3(channel);
}


void Resource_handler::tex_lookup_float4_2d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->lookup_float4(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            wrap_u,
            wrap_v,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            frame);
}

void Resource_handler::tex_lookup_deriv_float4_2d(
    float              result[4],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->lookup_deriv_float4(
            *reinterpret_cast<mi::Float32_2_struct const *>(coord->val),
            *reinterpret_cast<mi::Float32_2_struct const *>(coord->dx),
            *reinterpret_cast<mi::Float32_2_struct const *>(coord->dy),
            wrap_u,
            wrap_v,
            *reinterpret_cast<mi::Float32_2_struct const *>(crop_u),
            *reinterpret_cast<mi::Float32_2_struct const *>(crop_v),
            frame);
}

void Resource_handler::tex_lookup_float4_3d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const   crop_u[2],
    float const   crop_v[2],
    float const   crop_w[2],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->lookup_float4(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            wrap_u,
            wrap_v,
            wrap_w,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w),
            frame);
}

void Resource_handler::tex_lookup_float4_cube(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    Texture_cube const *o = reinterpret_cast<Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->lookup_float4(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

void Resource_handler::tex_lookup_float4_ptex(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    Texture_ptex const *o = reinterpret_cast<Texture_ptex const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) = o->lookup_float4(channel);
}

void Resource_handler::tex_lookup_color_2d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->lookup_color(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            wrap_u,
            wrap_v,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            frame).to_vector3();
}

void Resource_handler::tex_lookup_deriv_color_2d(
    float              rgb[3],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2],
    float              frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    mi::Float32_4 res = o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        Texture::Wrap_mode(wrap_u),
        Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
        frame);
    rgb[0] = res.x;
    rgb[1] = res.y;
    rgb[2] = res.z;
}

void Resource_handler::tex_lookup_color_3d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const   crop_u[2],
    float const   crop_v[2],
    float const   crop_w[2],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->lookup_color(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            wrap_u,
            wrap_v,
            wrap_w,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w),
            frame).to_vector3();
}

void Resource_handler::tex_lookup_color_cube(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    Texture_cube const *o = reinterpret_cast<Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->lookup_color(*reinterpret_cast<mi::Float32_3 const *>(coord)).to_vector3();
}

void Resource_handler::tex_lookup_color_ptex(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    Texture_ptex const*o = reinterpret_cast<Texture_ptex const*>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) = o->lookup_color(channel).to_vector3();
}

float Resource_handler::tex_texel_float_2d(
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2],
    float         frame) const
{
    Texture_2d const*o = reinterpret_cast<Texture_2d const*>(tex_data);

    return o->texel_float(
        *reinterpret_cast<mi::Sint32_2 const *>(coord),
        *reinterpret_cast<mi::Sint32_2 const *>(uv_tile),
        frame);
}

void Resource_handler::tex_texel_float2_2d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2],
    float         frame) const
{
    Texture_2d const*o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2_struct*>(result) =
        o->texel_float2(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile),
            frame);
}

void Resource_handler::tex_texel_float3_2d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2],
    float         frame) const
{
    Texture_2d const*o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->texel_float3(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile),
            frame);
}

void Resource_handler::tex_texel_float4_2d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->texel_float4(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile),
            frame);
}

void Resource_handler::tex_texel_color_2d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2],
    float         frame) const
{
    Texture_2d const *o = reinterpret_cast<Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->texel_color(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile),
            frame).to_vector3();
}

float Resource_handler::tex_texel_float_3d(
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3],
    float         frame) const
{
    Texture_3d const*o = reinterpret_cast<Texture_3d const*>(tex_data);

    return o->texel_float(*reinterpret_cast<mi::Sint32_3 const *>(coord), frame);
}

void Resource_handler::tex_texel_float2_3d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3],
    float         frame) const
{
    Texture_3d const*o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2_struct*>(result) =
        o->texel_float2(*reinterpret_cast<mi::Sint32_3 const *>(coord), frame);
}

void Resource_handler::tex_texel_float3_3d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3],
    float         frame) const
{
    Texture_3d const*o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->texel_float3(*reinterpret_cast<mi::Sint32_3 const *>(coord), frame);
}

void Resource_handler::tex_texel_float4_3d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->texel_float4(*reinterpret_cast<mi::Sint32_3 const *>(coord), frame);
}

void Resource_handler::tex_texel_color_3d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3],
    float         frame) const
{
    Texture_3d const *o = reinterpret_cast<Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->texel_color(*reinterpret_cast<mi::Sint32_3 const *>(coord), frame).to_vector3();
}

bool Resource_handler::tex_isvalid(
    void const *tex_data) const
{
    Texture const *o = reinterpret_cast<Texture const *>(tex_data);
    return o->is_valid();
}

void Resource_handler::tex_frame(
    int        result[2],
    void const *tex_data) const
{
    Texture const *o = reinterpret_cast<Texture const *>(tex_data);
    const mi::Uint32_2& res = o->get_first_last_frame();
    result[0] = res.x;
    result[1] = res.y;
}

// Initializes a light profile data helper object from a given light profile tag.
void Resource_handler::lp_init(
    void     *data,
    unsigned tag_v,
    void     *ctx)
{
    DB::Tag                                   tag(tag_v);
    DB::Typed_tag<LIGHTPROFILE::Lightprofile> typed_tag(tag);

    new (data) Light_profile(typed_tag, (DB::Transaction *)ctx);
}

// Terminate a light profile data helper object.
void Resource_handler::lp_term(void *data)
{
    Light_profile const *o = reinterpret_cast<Light_profile const *>(data);
    o->~Light_profile();
}

// Get the light profile power value.
float Resource_handler::lp_power(
    void const *lp_data,
    void       *thread_data) const
{
    Light_profile const *o =
        reinterpret_cast<Light_profile const *>(lp_data);
    return o->get_power();
}

// Get the light profile maximum value.
float Resource_handler::lp_maximum(
    void const *lp_data,
    void       *thread_data) const
{
    Light_profile const *o =
        reinterpret_cast<Light_profile const *>(lp_data);
    return o->get_maximum();
}

bool Resource_handler::lp_isvalid(
    void const *lp_data) const
{
    Light_profile const *o =
        reinterpret_cast<Light_profile const *>(lp_data);
    return o->is_valid();
}


/// Handle df::light_profile_evaluate(...)
float Resource_handler::lp_evaluate(
    void const    *lp_data,
    void          *thread_data,
    const float   theta_phi[2]) const
{
    Light_profile const *o =
        reinterpret_cast<Light_profile const *>(lp_data);

    return o->evaluate(*reinterpret_cast<mi::Float32_2 const *>(theta_phi));
}

/// Handle df::light_profile_sample(...)
void Resource_handler::lp_sample(
    float         result[3],
    void const    *lp_data,
    void          *thread_data,
    const float   xi[3]) const
{
    Light_profile const *o =
        reinterpret_cast<Light_profile const *>(lp_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->sample(*reinterpret_cast<mi::Float32_3 const *>(xi));
}

/// Handle df::light_profile_pdf(...)
float Resource_handler::lp_pdf(
    void const    *lp_data,
    void          *thread_data,
    const float   theta_phi[2]) const
{
    Light_profile const *o =
        reinterpret_cast<Light_profile const *>(lp_data);

    return o->pdf(*reinterpret_cast<mi::Float32_2 const *>(theta_phi));
}


// Initializes a bsdf measurement data helper object from a given bsdf measurement tag.
void Resource_handler::bm_init(
    void     *data,
    unsigned tag_v,
    void     *ctx)
{
    DB::Tag                                tag(tag_v);
    DB::Typed_tag<BSDFM::Bsdf_measurement> typed_tag(tag);

    new (data) Bsdf_measurement(typed_tag, (DB::Transaction *)ctx);
}

// Terminate a bsdf measurement data helper object.
void Resource_handler::bm_term(void *data)
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(data);
    o->~Bsdf_measurement();
}

bool Resource_handler::bm_isvalid(
    void const *bm_data) const
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(bm_data);
    return o->is_valid();
}

void Resource_handler::bm_resolution(
    unsigned      result[3],
    void const    *bm_data,
    Mbsdf_part    part) const
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(bm_data);

    mi::Uint32_3 res = o->get_resolution(part);
    result[0] = res.x;
    result[1] = res.y;
    result[2] = res.z;
}

void Resource_handler::bm_evaluate(
    float         result[3],
    void const    *bm_data,
    void          *thread_data,
    const float   theta_phi_in[2],
    const float   theta_phi_out[2],
    Mbsdf_part    part) const
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(bm_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->evaluate(
            *reinterpret_cast<mi::Float32_2 const *>(theta_phi_in),
            *reinterpret_cast<mi::Float32_2 const *>(theta_phi_out),
            part
        );
}

/// Handle // Handle df::bsdf_measurement_sample(...)
void Resource_handler::bm_sample(
    float         result[3],
    void const    *bm_data,
    void          *thread_data,
    const float   theta_phi_out[2],
    const float   xi[3],
    Mbsdf_part    part) const
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(bm_data);

    *reinterpret_cast<mi::Float32_3_struct*>(result) =
        o->sample(
            *reinterpret_cast<mi::Float32_2 const *>(theta_phi_out),
            *reinterpret_cast<mi::Float32_3 const *>(xi),
            part
        );
}

float Resource_handler::bm_pdf(
    void const    *bm_data,
    void          *thread_data,
    const float   theta_phi_in[2],
    const float   theta_phi_out[2],
    Mbsdf_part    part) const
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(bm_data);

    return o->pdf(*reinterpret_cast<mi::Float32_2 const *>(theta_phi_in),
                  *reinterpret_cast<mi::Float32_2 const *>(theta_phi_out),
                  part);
}

/// Handle df::bsdf_measurement_albedos(...)
void Resource_handler::bm_albedos(
    float         result[4],
    void const    *bm_data,
    void          *thread_data,
    const float   theta_phi[2]) const
{
    Bsdf_measurement const *o =
        reinterpret_cast<Bsdf_measurement const *>(bm_data);

    *reinterpret_cast<mi::Float32_4_struct*>(result) =
        o->albedos(*reinterpret_cast<mi::Float32_2 const *>(theta_phi));
}

// Destructor.
Resource_handler::~Resource_handler()
{
}

}  // MDLRT
}  // MI
