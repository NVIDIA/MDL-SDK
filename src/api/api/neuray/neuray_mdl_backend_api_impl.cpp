/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the IMdl_backend_api implementation.
 **/

#include "pch.h"

#include "neuray_mdl_backend_api_impl.h"
#include "neuray_mdl_backend_impl.h"

#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_mdl.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <render/mdl/backends/backends_target_code.h>

namespace MI {
namespace NEURAY {

Mdl_backend_api_impl::Mdl_backend_api_impl(mi::neuraylib::INeuray *neuray)
: m_neuray(neuray)
, m_mdlc_module(true)
{
}

Mdl_backend_api_impl::~Mdl_backend_api_impl()
{
    m_neuray = nullptr;
}

mi::neuraylib::IMdl_backend* Mdl_backend_api_impl::get_backend(Mdl_backend_kind kind)
{
    mi::base::Handle<mi::mdl::IMDL> compiler(m_mdlc_module->get_mdl());

    switch (kind) {
    case MB_LLVM_IR:
    case MB_CUDA_PTX:
    case MB_NATIVE:
    case MB_HLSL: {
        mi::base::Handle<mi::mdl::ICode_generator> generator(
            compiler->load_code_generator("jit"));
        if (!generator)
            return nullptr;
        mi::base::Handle<mi::mdl::ICode_generator_jit> jit(
            generator->get_interface<mi::mdl::ICode_generator_jit>());
        mi::base::Handle<mi::mdl::ICode_cache> code_cache(m_mdlc_module->get_code_cache());
        return new Mdl_llvm_backend(
            kind,
            compiler.get(),
            jit.get(),
            code_cache.get(),
            /*string_ids=*/true);
    }
    case MB_GLSL:
    case MB_FORCE_32_BIT:
        break;
    }

    return nullptr;
}

static mi::mdl::IValue_texture::Bsdf_data_kind convert_df_data_kind(
    mi::neuraylib::Df_data_kind kind)
{
    switch (kind)
    {
    case mi::neuraylib::DFK_NONE:
        return mi::mdl::IValue_texture::BDK_NONE;
    case mi::neuraylib::DFK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER;
    case mi::neuraylib::DFK_BECKMANN_SMITH_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_BECKMANN_SMITH_MULTISCATTER;
    case mi::neuraylib::DFK_BECKMANN_VC_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_BECKMANN_VC_MULTISCATTER;
    case  mi::neuraylib::DFK_GGX_SMITH_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_GGX_SMITH_MULTISCATTER;
    case mi::neuraylib::DFK_GGX_VC_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_GGX_VC_MULTISCATTER;
    case mi::neuraylib::DFK_SHEEN_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_SHEEN_MULTISCATTER;
    case mi::neuraylib::DFK_SIMPLE_GLOSSY_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_SIMPLE_GLOSSY_MULTISCATTER;
    case mi::neuraylib::DFK_WARD_GEISLER_MORODER_MULTISCATTER:
        return mi::mdl::IValue_texture::BDK_WARD_GEISLER_MORODER_MULTISCATTER;
    default:
        break;
    }
    return mi::mdl::IValue_texture::BDK_NONE;
}

const Float32* Mdl_backend_api_impl::get_df_data_texture(
        mi::neuraylib::Df_data_kind kind,
        mi::Size &rx,
        mi::Size &ry,
        mi::Size &rz) const
{
    mi::mdl::IValue_texture::Bsdf_data_kind data_kind = convert_df_data_kind(kind);
    if (data_kind == mi::mdl::IValue_texture::BDK_NONE)
        return nullptr;
    return BACKENDS::Target_code::get_df_data_texture(data_kind, rx, ry, rz);
}

mi::Sint32 Mdl_backend_api_impl::start()
{
    m_mdlc_module.set();
    return 0;
}

mi::Sint32 Mdl_backend_api_impl::shutdown()
{
    m_mdlc_module.reset();
    return 0;
}

} // namespace NEURAY
} // namespace MI

