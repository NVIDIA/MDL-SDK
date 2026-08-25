/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

/// \file
/// \brief Host-side glue for the CUDA baker: links the embedded PTX with the
///        MDL-generated PTX, uploads textures, launches the bake kernel, and reads
///        the result back. Requires libcuda and libcudart to be linked at build time.

#ifndef RENDER_BAKER_BAKER_BAKER_CUDA_OSS_H
#define RENDER_BAKER_BAKER_BAKER_CUDA_OSS_H

#include <mi/base/handle.h>
#include <mi/base/types.h>

// CUDA forward declaration; matches <cuda.h>, which only baker_cuda_oss.cpp
// pulls in.
struct CUctx_st;
typedef struct CUctx_st *CUcontext;

namespace mi { namespace neuraylib {
    class ICanvas;
    class ITarget_code;
} }

namespace MI {

namespace DB { class Transaction; }

namespace BAKER {

class Baker_module_impl;

/// One-shot CUDA baker for the open-source MDL SDK release. Constructed,
/// invoked once via bake_texture(), then destroyed. All CUDA failures are
/// converted to a false return from bake_texture(); the caller then falls
/// back to the CPU path.
class Baker_cuda_oss {
public:
    Baker_cuda_oss(
        const Baker_module_impl           *baker_module,
        int                                dev_id,
        mi::neuraylib::ITarget_code const *target_code,
        mi::neuraylib::ICanvas            *texture,
        unsigned int                       samples,
        mi::Float32                        min_u,
        mi::Float32                        max_u,
        mi::Float32                        min_v,
        mi::Float32                        max_v,
        mi::Float32                        animation_time,
        float                             *constant_target,
        unsigned int                       constant_target_components,
        unsigned int                       state_flags,
        bool                               is_environment,
        bool                               want_constant_detection);

    ~Baker_cuda_oss();

    Baker_cuda_oss(const Baker_cuda_oss&) = delete;
    Baker_cuda_oss& operator=(const Baker_cuda_oss&) = delete;

    /// Returns true if the construction itself succeeded (CUDA context
    /// obtained and pushed). If false, do not call bake_texture(); construct
    /// the CPU path instead.
    bool is_ready() const { return m_ready; }

    /// Executes the bake on the GPU. Returns false on any CUDA failure; the
    /// canvas/constant_target is left unchanged in that case.
    bool bake_texture(DB::Transaction* transaction);

    /// Whether the last bake_texture() found a uniform result. Only valid
    /// if want_constant_detection was set at construction.
    bool are_all_pixels_equal() const { return m_all_pixels_equal; }

private:
    CUcontext                                            m_pushed_ctx = nullptr;
    mi::base::Handle<mi::neuraylib::ITarget_code const>  m_target_code;
    mi::base::Handle<mi::neuraylib::ICanvas>             m_texture;  // null when baking a constant
    unsigned int                                         m_samples;
    mi::Float32                                          m_min_u, m_max_u, m_min_v, m_max_v;
    mi::Float32                                          m_animation_time;
    float                                               *m_constant_target;
    unsigned int                                         m_constant_target_components;
    unsigned int                                         m_state_flags;
    bool                                                 m_is_environment;
    bool                                                 m_want_constant_detection;
    bool                                                 m_ready = false;
    bool                                                 m_all_pixels_equal = false;
};

} // namespace BAKER
} // namespace MI

#endif // RENDER_BAKER_BAKER_BAKER_CUDA_OSS_H
