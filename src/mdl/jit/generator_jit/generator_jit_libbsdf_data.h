/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_LIBBSDF_DATA_H
#define MDL_GENERATOR_JIT_LIBBSDF_DATA_H 1

#include <mi/mdl/mdl_values.h>

namespace mi {
namespace mdl {
namespace libbsdf_data {

    extern unsigned const libbsdf_multiscatter_res_theta_beckmann_smith;
    extern unsigned const libbsdf_multiscatter_res_roughness_beckmann_smith;
    extern unsigned const libbsdf_multiscatter_res_ior_beckmann_smith;
    extern unsigned char const libbsdf_multiscatter_data_beckmann_smith[];

    extern unsigned const libbsdf_multiscatter_res_theta_beckmann_vc;
    extern unsigned const libbsdf_multiscatter_res_roughness_beckmann_vc;
    extern unsigned const libbsdf_multiscatter_res_ior_beckmann_vc;
    extern unsigned char const libbsdf_multiscatter_data_beckmann_vc[];

    extern unsigned const libbsdf_multiscatter_res_theta_phong_vc;
    extern unsigned const libbsdf_multiscatter_res_roughness_phong_vc;
    extern unsigned const libbsdf_multiscatter_res_ior_phong_vc;
    extern unsigned char const libbsdf_multiscatter_data_phong_vc[];

    extern unsigned const libbsdf_multiscatter_res_theta_disk_bs;
    extern unsigned const libbsdf_multiscatter_res_roughness_disk_bs;
    extern unsigned const libbsdf_multiscatter_res_ior_disk_bs;
    extern unsigned char const libbsdf_multiscatter_data_disk_bs[];

    extern unsigned const libbsdf_multiscatter_res_theta_ggx_smith;
    extern unsigned const libbsdf_multiscatter_res_roughness_ggx_smith;
    extern unsigned const libbsdf_multiscatter_res_ior_ggx_smith;
    extern unsigned char const libbsdf_multiscatter_data_ggx_smith[];

    extern unsigned const libbsdf_multiscatter_res_theta_ggx_vc;
    extern unsigned const libbsdf_multiscatter_res_roughness_ggx_vc;
    extern unsigned const libbsdf_multiscatter_res_ior_ggx_vc;
    extern unsigned char const libbsdf_multiscatter_data_ggx_vc[];

    extern unsigned const libbsdf_multiscatter_res_theta_sink_vc;
    extern unsigned const libbsdf_multiscatter_res_roughness_sink_vc;
    extern unsigned const libbsdf_multiscatter_res_ior_sink_vc;
    extern unsigned char const libbsdf_multiscatter_data_sink_vc[];

    extern unsigned const libbsdf_multiscatter_res_theta_ward_gm;
    extern unsigned const libbsdf_multiscatter_res_roughness_ward_gm;
    extern unsigned const libbsdf_multiscatter_res_ior_ward_gm;
    extern unsigned char const libbsdf_multiscatter_data_ward_gm[];

    inline bool get_libbsdf_multiscatter_data_resolution(
        mi::mdl::IValue_texture::Bsdf_data_kind bsdf_data_kind,
        size_t &out_theta,
        size_t &out_roughness,
        size_t &out_ior)
    {
        // (currently and for the foreseeable future) constant for all BSDFs
        out_theta = libbsdf_multiscatter_res_theta_phong_vc;
        out_roughness = libbsdf_multiscatter_res_roughness_phong_vc;

        switch (bsdf_data_kind)
        {
            case IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER:
            case IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER:
            case IValue_texture::BDK_BECKMANN_VC_MULTISCATTER:
            case IValue_texture::BDK_GGX_SMITH_MULTISCATTER:
            case IValue_texture::BDK_GGX_VC_MULTISCATTER:
                out_ior = libbsdf_multiscatter_res_ior_phong_vc;
                break;

            case IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER:
            case IValue_texture::BDK_SHEEN_MULTISCATTER:
            case IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER:
                out_ior = libbsdf_multiscatter_res_ior_disk_bs;
                break;

            default:
                return false; // no data for other semantics
        }

        out_theta = out_theta + 1;
        out_ior = out_ior * 2 + 1;
        return true;
    }

    inline unsigned char const* get_libbsdf_multiscatter_data(
        mi::mdl::IValue_texture::Bsdf_data_kind bsdf_data_kind,
        size_t &size)
    {
        size_t theta, roughness, ior;
        if (!get_libbsdf_multiscatter_data_resolution(bsdf_data_kind, theta, roughness, ior))
            return NULL;

        size = theta * roughness * ior * sizeof(float);

        switch (bsdf_data_kind)
        {
            case IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER:
                return libbsdf_multiscatter_data_phong_vc;
            case IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER:
                return libbsdf_multiscatter_data_beckmann_smith;
            case IValue_texture::BDK_BECKMANN_VC_MULTISCATTER:
                return libbsdf_multiscatter_data_beckmann_vc;
            case IValue_texture::BDK_GGX_SMITH_MULTISCATTER:
                return libbsdf_multiscatter_data_ggx_smith;
            case IValue_texture::BDK_GGX_VC_MULTISCATTER:
                return libbsdf_multiscatter_data_ggx_vc;

            case IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER:
              return libbsdf_multiscatter_data_disk_bs;
             case IValue_texture::BDK_SHEEN_MULTISCATTER:
                 return libbsdf_multiscatter_data_sink_vc;
             case IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER:
                 return libbsdf_multiscatter_data_ward_gm;
            default:
                return NULL; // no data for other semantics
        }
    }

}  // libbsdf_data
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_LIBBSDF_DATA_H
