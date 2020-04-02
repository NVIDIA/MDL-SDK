/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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

// Get the number of bytes that must be allocated for a resource object.
size_t Resource_handler::get_data_size() const
{
    size_t size = sizeof(MI::MDLRT::Texture_2d);
    if (size < sizeof(MI::MDLRT::Texture_3d))
        size = sizeof(MI::MDLRT::Texture_3d);
    if (size < sizeof(MI::MDLRT::Texture_cube))
        size = sizeof(MI::MDLRT::Texture_cube);
    if (size < sizeof(MI::MDLRT::Texture_ptex))
        size = sizeof(MI::MDLRT::Texture_ptex);
    if (size < sizeof(MI::MDLRT::Light_profile))
        size = sizeof(MI::MDLRT::Light_profile);
    if (size < sizeof(MI::MDLRT::Bsdf_measurement))
        size = sizeof(MI::MDLRT::Bsdf_measurement);
    return size;
}

// Initializes a texture data helper object.
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
        new (data) MI::MDLRT::Texture_2d(
            typed_tag,
            MI::MDLRT::Texture::Gamma_mode(gamma),
            m_use_derivatives,
            (MI::DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_3D:
        new (data) MI::MDLRT::Texture_3d(
            typed_tag, MI::MDLRT::Texture::Gamma_mode(gamma), (MI::DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_CUBE:
        new (data) MI::MDLRT::Texture_cube(
            typed_tag, MI::MDLRT::Texture::Gamma_mode(gamma), (MI::DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_PTEX:
        new (data) MI::MDLRT::Texture_ptex(
            typed_tag, MI::MDLRT::Texture::Gamma_mode(gamma), (MI::DB::Transaction *)ctx);
        break;
    case mi::mdl::IType_texture::TS_BSDF_DATA:
        // handle like 3D texture
        new (data) MI::MDLRT::Texture_3d(
            typed_tag, MI::MDLRT::Texture::Gamma_mode(gamma), (MI::DB::Transaction *)ctx);
        break;
    }
}

// Terminate a texture data helper object.
void Resource_handler::tex_term(
    void                          *data,
    mi::mdl::IType_texture::Shape shape)
{
    switch (shape) {
    case mi::mdl::IType_texture::TS_2D:
        {
            MI::MDLRT::Texture_2d *o = reinterpret_cast<MI::MDLRT::Texture_2d *>(data);
            o->~Texture_2d();
            break;
        }
    case mi::mdl::IType_texture::TS_3D:
        {
            MI::MDLRT::Texture_3d *o = reinterpret_cast<MI::MDLRT::Texture_3d *>(data);
            o->~Texture_3d();
            break;
        }
    case mi::mdl::IType_texture::TS_CUBE:
        {
            MI::MDLRT::Texture_cube *o = reinterpret_cast<MI::MDLRT::Texture_cube *>(data);
            o->~Texture_cube();
            break;
        }
    case mi::mdl::IType_texture::TS_PTEX:
        {
            MI::MDLRT::Texture_ptex *o = reinterpret_cast<MI::MDLRT::Texture_ptex *>(data);
            o->~Texture_ptex();
            break;
        }
    case mi::mdl::IType_texture::TS_BSDF_DATA:
        {
            // handle like 3D texture
            MI::MDLRT::Texture_3d *o = reinterpret_cast<MI::MDLRT::Texture_3d *>(data);
            o->~Texture_3d();
            break;
        }
    }
}

// Handle tex::width(texture_2d, int2) and tex::height(texture_2d, int2)
void Resource_handler::tex_resolution_2d(
    int           result[2],
    void const    *tex_data,
    int const     uv_tile[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);
    mi::Sint32_2 res = o->get_resolution(*reinterpret_cast<mi::Sint32_2 const *>(uv_tile));
    result[0] = res.x;
    result[1] = res.y;
}

// Handle tex::width(texture_*) (not for udim textures)
int Resource_handler::tex_width(
    void const    *tex_data) const
{
    MI::MDLRT::Texture const *o = reinterpret_cast<MI::MDLRT::Texture const *>(tex_data);
    return o->get_width();
}

// Handle tex::height(texture_*) (not for udim textures)
int Resource_handler::tex_height(
    void const    *tex_data) const
{
    MI::MDLRT::Texture const *o = reinterpret_cast<MI::MDLRT::Texture const *>(tex_data);
    return o->get_height();
}

// Handle tex::depth(texture_*)
int Resource_handler::tex_depth(
    void const    *tex_data) const
{
    MI::MDLRT::Texture const *o = reinterpret_cast<MI::MDLRT::Texture const *>(tex_data);
    return o->get_depth();
}

// Handle tex::lookup_float(texture_2d, ...)
float Resource_handler::tex_lookup_float_2d(
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    return o->lookup_float(
        *reinterpret_cast<mi::Float32_2 const *>(coord),
        MI::MDLRT::Texture::Wrap_mode(wrap_u),
        MI::MDLRT::Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v));
}

// Handle tex::lookup_float(texture_2d, ...) with derivatives
float Resource_handler::tex_lookup_deriv_float_2d(
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    return o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        MI::MDLRT::Texture::Wrap_mode(wrap_u),
        MI::MDLRT::Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v)).x;
}

// Handle tex::lookup_float(texture_3d, ...)
float Resource_handler::tex_lookup_float_3d(
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const   crop_u[2],
    float const   crop_v[2],
    float const   crop_w[2]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    return o->lookup_float(
        *reinterpret_cast<mi::Float32_3 const *>(coord),
        MI::MDLRT::Texture::Wrap_mode(wrap_u),
        MI::MDLRT::Texture::Wrap_mode(wrap_v),
        MI::MDLRT::Texture::Wrap_mode(wrap_w),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_w));
}

// Handle tex::lookup_float(texture_cube, ...)
float Resource_handler::tex_lookup_float_cube(
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    MI::MDLRT::Texture_cube const *o = reinterpret_cast<MI::MDLRT::Texture_cube const *>(tex_data);

    return o->lookup_float(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

// Handle tex::lookup_float(texture_ptex, ...)
float Resource_handler::tex_lookup_float_ptex(
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    MI::MDLRT::Texture_ptex const *o = reinterpret_cast<MI::MDLRT::Texture_ptex const *>(tex_data);

    return o->lookup_float(channel);
}

// Handle tex::lookup_float2(texture_2d, ...)
void Resource_handler::tex_lookup_float2_2d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2*>(result) =
        o->lookup_float2(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            MI::MDLRT::Texture::Wrap_mode(wrap_u),
            MI::MDLRT::Texture::Wrap_mode(wrap_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v));
}

// Handle tex::lookup_float2(texture_2d, ...) with derivatives
void Resource_handler::tex_lookup_deriv_float2_2d(
    float              result[2],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    mi::Float32_4 res = o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        MI::MDLRT::Texture::Wrap_mode(wrap_u),
        MI::MDLRT::Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v));

    result[0] = res.x;
    result[1] = res.y;
}

// Handle tex::lookup_float2(texture_3d, ...)
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
    float const   crop_w[2]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2*>(result) =
        o->lookup_float2(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            MI::MDLRT::Texture::Wrap_mode(wrap_u),
            MI::MDLRT::Texture::Wrap_mode(wrap_v),
            MI::MDLRT::Texture::Wrap_mode(wrap_w),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w));
}

// Handle tex::lookup_float2(texture_cube, ...)
void Resource_handler::tex_lookup_float2_cube(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    MI::MDLRT::Texture_cube const *o = reinterpret_cast<MI::MDLRT::Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_2*>(result) =
        o->lookup_float2(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

// Handle tex::lookup_float2(texture_ptex, ...)
void Resource_handler::tex_lookup_float2_ptex(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    MI::MDLRT::Texture_ptex const *o = reinterpret_cast<MI::MDLRT::Texture_ptex const *>(tex_data);

    *reinterpret_cast<mi::Float32_2*>(result) = o->lookup_float2(channel);
}

// Handle tex::lookup_float3(texture_2d, ...)
void Resource_handler::tex_lookup_float3_2d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->lookup_float3(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            MI::MDLRT::Texture::Wrap_mode(wrap_u),
            MI::MDLRT::Texture::Wrap_mode(wrap_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v));
}

// Handle tex::lookup_float3(texture_2d, ...) with derivatives
void Resource_handler::tex_lookup_deriv_float3_2d(
    float              result[3],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    mi::Float32_4 res = o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        MI::MDLRT::Texture::Wrap_mode(wrap_u),
        MI::MDLRT::Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v));
    result[0] = res.x;
    result[1] = res.y;
    result[2] = res.z;
}

// Handle tex::lookup_float3(texture_3d, ...)
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
    float const   crop_w[2]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->lookup_float3(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            MI::MDLRT::Texture::Wrap_mode(wrap_u),
            MI::MDLRT::Texture::Wrap_mode(wrap_v),
            MI::MDLRT::Texture::Wrap_mode(wrap_w),
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w));
}

// Handle tex::lookup_float3(texture_cube, ...)
void Resource_handler::tex_lookup_float3_cube(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    MI::MDLRT::Texture_cube const *o = reinterpret_cast<MI::MDLRT::Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->lookup_float3(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

// Handle tex::lookup_float3(texture_ptex, ...)
void Resource_handler::tex_lookup_float3_ptex(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    MI::MDLRT::Texture_ptex const *o = reinterpret_cast<MI::MDLRT::Texture_ptex const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(result) = o->lookup_float3(channel);
}


// Handle tex::lookup_float4(texture_2d, ...)
void Resource_handler::tex_lookup_float4_2d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->lookup_float4(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            wrap_u,
            wrap_v,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v));
}

// Handle tex::lookup_float4(texture_2d, ...) with derivatives
void Resource_handler::tex_lookup_deriv_float4_2d(
    float              result[4],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->lookup_deriv_float4(
            *reinterpret_cast<mi::Float32_2 const *>(coord->val),
            *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
            *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
            wrap_u,
            wrap_v,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v));
}

// Handle tex::lookup_float4(texture_3d, ...)
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
    float const   crop_w[2]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->lookup_float4(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            wrap_u,
            wrap_v,
            wrap_w,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w));
}

// Handle tex::lookup_float4(texture_cube, ...)
void Resource_handler::tex_lookup_float4_cube(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    MI::MDLRT::Texture_cube const *o = reinterpret_cast<MI::MDLRT::Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->lookup_float4(*reinterpret_cast<mi::Float32_3 const *>(coord));
}

// Handle tex::lookup_float4(texture_ptex, ...)
void Resource_handler::tex_lookup_float4_ptex(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    MI::MDLRT::Texture_ptex const *o = reinterpret_cast<MI::MDLRT::Texture_ptex const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) = o->lookup_float4(channel);
}

// Handle tex::lookup_color(texture_2d, ...)
void Resource_handler::tex_lookup_color_2d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const   crop_u[2],
    float const   crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->lookup_color(
            *reinterpret_cast<mi::Float32_2 const *>(coord),
            wrap_u,
            wrap_v,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v)).to_vector3();
}

// Handle tex::lookup_color(texture_2d, ...) with derivatives
void Resource_handler::tex_lookup_deriv_color_2d(
    float              rgb[3],
    void const         *tex_data,
    void               * /*thread_data*/,
    Deriv_float2 const *coord,
    Tex_wrap_mode      wrap_u,
    Tex_wrap_mode      wrap_v,
    float const        crop_u[2],
    float const        crop_v[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    mi::Float32_4 res = o->lookup_deriv_float4(
        *reinterpret_cast<mi::Float32_2 const *>(coord->val),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dx),
        *reinterpret_cast<mi::Float32_2 const *>(coord->dy),
        MI::MDLRT::Texture::Wrap_mode(wrap_u),
        MI::MDLRT::Texture::Wrap_mode(wrap_v),
        *reinterpret_cast<mi::Float32_2 const *>(crop_u),
        *reinterpret_cast<mi::Float32_2 const *>(crop_v));
    rgb[0] = res.x;
    rgb[1] = res.y;
    rgb[2] = res.z;
}

// Handle tex::lookup_color(texture_3d, ...)
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
    float const   crop_w[2]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->lookup_color(
            *reinterpret_cast<mi::Float32_3 const *>(coord),
            wrap_u,
            wrap_v,
            wrap_w,
            *reinterpret_cast<mi::Float32_2 const *>(crop_u),
            *reinterpret_cast<mi::Float32_2 const *>(crop_v),
            *reinterpret_cast<mi::Float32_2 const *>(crop_w)).to_vector3();
}

// Handle tex::lookup_color(texture_cube, ...)
void Resource_handler::tex_lookup_color_cube(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    float const   coord[3]) const
{
    MI::MDLRT::Texture_cube const *o = reinterpret_cast<MI::MDLRT::Texture_cube const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->lookup_color(*reinterpret_cast<mi::Float32_3 const *>(coord)).to_vector3();
}

// Handle tex::lookup_color(texture_ptex, ...)
void Resource_handler::tex_lookup_color_ptex(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int           channel) const
{
    MI::MDLRT::Texture_ptex const*o = reinterpret_cast<MI::MDLRT::Texture_ptex const*>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) = o->lookup_color(channel).to_vector3();
}

// Handle tex::texel_float(texture_2d, ...)
float Resource_handler::tex_texel_float_2d(
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2]) const
{
    MI::MDLRT::Texture_2d const*o = reinterpret_cast<MI::MDLRT::Texture_2d const*>(tex_data);

    return o->texel_float(
        *reinterpret_cast<mi::Sint32_2 const *>(coord),
        *reinterpret_cast<mi::Sint32_2 const *>(uv_tile));
}

// Handle tex::texel_float2(texture_2d, ...)
void Resource_handler::tex_texel_float2_2d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2]) const
{
    MI::MDLRT::Texture_2d const*o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2*>(result) =
        o->texel_float2(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile));
}

// Handle tex::texel_float3(texture_2d, ...)
void Resource_handler::tex_texel_float3_2d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2]) const
{
    MI::MDLRT::Texture_2d const*o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->texel_float3(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile));
}

// Handle tex::texel_float4(texture_2d, ...)
void Resource_handler::tex_texel_float4_2d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->texel_float4(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile));
}

// Handle tex::texel_color(texture_2d, ...)
void Resource_handler::tex_texel_color_2d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[2],
    int const     uv_tile[2]) const
{
    MI::MDLRT::Texture_2d const *o = reinterpret_cast<MI::MDLRT::Texture_2d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->texel_color(
            *reinterpret_cast<mi::Sint32_2 const *>(coord),
            *reinterpret_cast<mi::Sint32_2 const *>(uv_tile)).to_vector3();
}

// Handle tex::texel_float(texture_3d, ...)
float Resource_handler::tex_texel_float_3d(
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3]) const
{
    MI::MDLRT::Texture_3d const*o = reinterpret_cast<MI::MDLRT::Texture_3d const*>(tex_data);

    return o->texel_float(*reinterpret_cast<mi::Sint32_3 const *>(coord));
}

// Handle tex::texel_float2(texture_3d, ...)
void Resource_handler::tex_texel_float2_3d(
    float         result[2],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3]) const
{
    MI::MDLRT::Texture_3d const*o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_2*>(result) =
        o->texel_float2(*reinterpret_cast<mi::Sint32_3 const *>(coord));
}

// Handle tex::texel_float3(texture_3d, ...)
void Resource_handler::tex_texel_float3_3d(
    float         result[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3]) const
{
    MI::MDLRT::Texture_3d const*o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->texel_float3(*reinterpret_cast<mi::Sint32_3 const *>(coord));
}

// Handle tex::texel_float4(texture_3d, ...)
void Resource_handler::tex_texel_float4_3d(
    float         result[4],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->texel_float4(*reinterpret_cast<mi::Sint32_3 const *>(coord));
}

// Handle tex::texel_color(texture_3d, ...)
void Resource_handler::tex_texel_color_3d(
    float         rgb[3],
    void const    *tex_data,
    void          * /*thread_data*/,
    int const     coord[3]) const
{
    MI::MDLRT::Texture_3d const *o = reinterpret_cast<MI::MDLRT::Texture_3d const *>(tex_data);

    *reinterpret_cast<mi::Float32_3*>(rgb) =
        o->texel_color(*reinterpret_cast<mi::Sint32_3 const *>(coord)).to_vector3();
}

// Handle tex::texture_isvalid().
bool Resource_handler::tex_isvalid(
    void const *tex_data) const
{
    MI::MDLRT::Texture const *o = reinterpret_cast<MI::MDLRT::Texture const *>(tex_data);
    return o->is_valid();
}

// Initializes a light profile data helper object from a given light profile tag.
void Resource_handler::lp_init(
    void     *data,
    unsigned tag_v,
    void     *ctx)
{
    DB::Tag                                   tag(tag_v);
    DB::Typed_tag<LIGHTPROFILE::Lightprofile> typed_tag(tag);

    new (data) MI::MDLRT::Light_profile(typed_tag, (MI::DB::Transaction *)ctx);
}

// Terminate a light profile data helper object.
void Resource_handler::lp_term(void *data)
{
    MI::MDLRT::Light_profile const *o = reinterpret_cast<MI::MDLRT::Light_profile const *>(data);
    o->~Light_profile();
}

// Get the light profile power value.
float Resource_handler::lp_power(
    void const *lp_data,
    void       *thread_data) const
{
    MI::MDLRT::Light_profile const *o =
        reinterpret_cast<MI::MDLRT::Light_profile const *>(lp_data);
    return o->get_power();
}

// Get the light profile maximum value.
float Resource_handler::lp_maximum(
    void const *lp_data,
    void       *thread_data) const
{
    MI::MDLRT::Light_profile const *o =
        reinterpret_cast<MI::MDLRT::Light_profile const *>(lp_data);
    return o->get_maximum();
}

// Handle df::light_profile_isvalid().
bool Resource_handler::lp_isvalid(
    void const *lp_data) const
{
    MI::MDLRT::Light_profile const *o =
        reinterpret_cast<MI::MDLRT::Light_profile const *>(lp_data);
    return o->is_valid();
}


/// Handle df::light_profile_evaluate(...)
float Resource_handler::lp_evaluate(
    void const    *lp_data,
    void          *thread_data,
    const float   theta_phi[2]) const
{
    MI::MDLRT::Light_profile const *o =
        reinterpret_cast<MI::MDLRT::Light_profile const *>(lp_data);

    return o->evaluate(*reinterpret_cast<mi::Float32_2 const *>(theta_phi));
}

/// Handle df::light_profile_sample(...)
void Resource_handler::lp_sample(
    float         result[3],
    void const    *lp_data,
    void          *thread_data,
    const float   xi[3]) const
{
    MI::MDLRT::Light_profile const *o =
        reinterpret_cast<MI::MDLRT::Light_profile const *>(lp_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->sample(*reinterpret_cast<mi::Float32_3 const *>(xi));
}

/// Handle df::light_profile_pdf(...)
float Resource_handler::lp_pdf(
    void const    *lp_data,
    void          *thread_data,
    const float   theta_phi[2]) const
{
    MI::MDLRT::Light_profile const *o =
        reinterpret_cast<MI::MDLRT::Light_profile const *>(lp_data);

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

    new (data) MI::MDLRT::Bsdf_measurement(typed_tag, (MI::DB::Transaction *)ctx);
}

// Terminate a bsdf measurement data helper object.
void Resource_handler::bm_term(void *data)
{
    MI::MDLRT::Bsdf_measurement const *o =
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(data);
    o->~Bsdf_measurement();
}

// Handle df::bsdf_measurement_isvalid().
bool Resource_handler::bm_isvalid(
    void const *bm_data) const
{
    MI::MDLRT::Bsdf_measurement const *o =
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(bm_data);
    return o->is_valid();
}

// Handle df::bsdf_measurement_resolution(...)
void Resource_handler::bm_resolution(
    unsigned      result[3],
    void const    *bm_data,
    Mbsdf_part    part) const
{
    MI::MDLRT::Bsdf_measurement const *o = 
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(bm_data);

    mi::Uint32_3 res = o->get_resolution(part);
    result[0] = res.x;
    result[1] = res.y;
    result[2] = res.z;
}

// Handle df::bsdf_measurement_evaluate(...)
void Resource_handler::bm_evaluate(
    float         result[3],
    void const    *bm_data,
    void          *thread_data,
    const float   theta_phi_in[2],
    const float   theta_phi_out[2],
    Mbsdf_part    part) const
{
    MI::MDLRT::Bsdf_measurement const *o =
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(bm_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
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
    MI::MDLRT::Bsdf_measurement const *o =
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(bm_data);

    *reinterpret_cast<mi::Float32_3*>(result) =
        o->sample(
            *reinterpret_cast<mi::Float32_2 const *>(theta_phi_out),
            *reinterpret_cast<mi::Float32_3 const *>(xi),
            part
        );
}

// Handle df::bsdf_measurement_pdf(...)
float Resource_handler::bm_pdf(
    void const    *bm_data,
    void          *thread_data,
    const float   theta_phi_in[2],
    const float   theta_phi_out[2],
    Mbsdf_part    part) const
{
    MI::MDLRT::Bsdf_measurement const *o = 
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(bm_data);

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
    MI::MDLRT::Bsdf_measurement const *o =
        reinterpret_cast<MI::MDLRT::Bsdf_measurement const *>(bm_data);

    *reinterpret_cast<mi::Float32_4*>(result) =
        o->albedos(*reinterpret_cast<mi::Float32_2 const *>(theta_phi));
}

// Destructor.
Resource_handler::~Resource_handler()
{
}

}  // MDLRT
}  // MI
