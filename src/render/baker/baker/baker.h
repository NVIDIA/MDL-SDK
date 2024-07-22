/***************************************************************************************************
 * Copyright (c) 2017-2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief The definition of the implementation of the Baker_module

#ifndef RENDER_BAKER_BAKER_BAKER_H
#define RENDER_BAKER_BAKER_BAKER_H

#include <map>
#include <string>
#include <memory>

#include <mi/base/lock.h>
#include <mi/base/handle.h>
#include <mi/base/types.h>
#include <mi/base/interface_implement.h>

#include <base/system/main/access_module.h>


#include <io/scene/mdl_elements/i_mdl_elements_value.h>

#include "i_baker.h"

#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_module_id.h>
#define log_error(...) MI::LOG::mod_log->error(M_BAKER, MI::LOG::Mod_log::C_MISC, __VA_ARGS__)
#define log_warning(...) MI::LOG::mod_log->warning(M_BAKER, MI::LOG::Mod_log::C_MISC, __VA_ARGS__)
#define log_debug(...) MI::LOG::mod_log->debug(M_BAKER, MI::LOG::Mod_log::C_MISC, __VA_ARGS__)



namespace mi { namespace mdl { class IMDL; class ICode_generator_jit; } }

namespace MI {

namespace MDL { class Mdl_function_call; }

namespace MDLC { class Mdlc_module; }

namespace BAKER {

class Baker_code_impl : public mi::base::Interface_implement<IBaker_code>
{
public:
    /// Constructor.
    ///
    /// \param gpu_dev_id  the target device ID if should run on GPU
    /// \param gpu_code    if non-NULL, the GPU target code
    /// \param gpu_code    if non-NULL, the CPU target code
    /// \param is_environment true for environment functions
    Baker_code_impl(
        mi::Uint32 gpu_dev_id,
        const mi::neuraylib::ITarget_code *gpu_code,
        const mi::neuraylib::ITarget_code *cpu_code,
        bool is_environment);

    /// Get the target device ID if should run on GPU.
    mi::Uint32 get_used_gpu_device_id() const;

    /// Get the GPU target code.
    const mi::neuraylib::ITarget_code* get_gpu_target_code() const;

    /// Get the CPU target code.
    const mi::neuraylib::ITarget_code* get_cpu_target_code() const;

    /// Drop the GPU target code.
    void gpu_failed() const;

    bool is_environment() const {
        return m_is_environment;
    }

private:
    /// The GPU id if code should run on the GPU.
    mi::Uint32 m_gpu_dev_id;

    /// The generated GPU code if any.
    mutable mi::base::Handle<const mi::neuraylib::ITarget_code> m_gpu_code;

    /// The generated CPU code if any.
    mutable mi::base::Handle<const mi::neuraylib::ITarget_code> m_cpu_code;

    /// Is this an environment function?
    bool m_is_environment;
};

class Baker_module_impl : public Baker_module
{
public:
    Baker_module_impl();

    bool init();

    void exit();


    const IBaker_code* create_environment_baker_code(
        DB::Transaction* transaction,
        const MDL::Mdl_function_call* environment_function,
        mi::neuraylib::Baker_resource resource,
        mi::Uint32 gpu_device_id,
        bool &is_uniform) const;

    const IBaker_code* create_baker_code(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* compiled_material,
        const char* path,
        mi::neuraylib::Baker_resource resource,
        mi::Uint32 gpu_device_id,
        std::string& pixel_type,
        bool& is_uniform) const;

    mi::Sint32 bake_texture(
        DB::Transaction* transaction,
        const IBaker_code* baker_code,
        mi::neuraylib::ICanvas* texture,
        mi::Uint32 samples,
        mi::Uint32 state_flags) const;

    mi::Sint32 bake_texture(
        DB::Transaction* transaction,
        const IBaker_code* baker_code,
        mi::neuraylib::ICanvas* texture,
        mi::Float32 min_u,
        mi::Float32 max_u,
        mi::Float32 min_v,
        mi::Float32 max_v,
        mi::Uint32 samples,
        mi::Uint32 state_flags) const;

    mi::Sint32 bake_constant(
        DB::Transaction* transaction,
        const IBaker_code* baker_code,
        Constant_result& constant,
        mi::Uint32 samples,
        const char* pixel_type) const;

private:

    const IBaker_code* create_baker_code_internal(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* compiled_material,
        const MDL::Mdl_function_call* function_call,
        const char* path,
        mi::neuraylib::Baker_resource resource,
        mi::Uint32 gpu_device_id,
        std::string& pixel_type,
        bool& is_uniform,
        bool use_custom_cpu_tex_runtime) const;



private:
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;
    mi::base::Handle<mi::mdl::IMDL> m_compiler;
    mi::base::Handle<mi::mdl::ICode_generator_jit> m_code_generator_jit;

};

} // namespace BAKER
} // namespace MI

#endif  // RENDER_BAKER_BAKER_BAKER_H

