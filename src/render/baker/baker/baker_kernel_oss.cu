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

/// \file baker_kernel_oss.cu
/// \brief Self-contained CUDA baker kernel for the open-source MDL SDK release.
///
/// Two entry points: bake_surface() and bake_environment(). They mirror the
/// per-pixel CPU loop in Baker_fragmented_job::execute_fragment(), but call
/// the MDL-generated lambda directly on the GPU. Pixel-type conversion (RGBE,
/// boolean, NaN clamp) stays on the host — this kernel always writes float4.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TARGET_CODE_USE_CUDA_TYPES
#include <mi/neuraylib/target_code_types.h>

#include "baker_shared.h"
#include "texture_support_cuda.h"
#include "baker_kernel_oss_params.h"

using namespace mi::neuraylib;

// Minimal float4 arithmetic — CUDA's <vector_types.h> declares the struct
// but no operators. Kept at file scope so NVCC operator lookup works
// regardless of where the call sites live.
__device__ __forceinline__ float4& operator+=(float4& a, const float4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

__device__ __forceinline__ float4& operator*=(float4& a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
    return a;
}

namespace {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ float radinv2(unsigned int i)
{
    return __uint2float_rn(__brev(i) >> 8) * 0x1p-24f;
}

__device__ __forceinline__ float fractf(float x)
{
    return x - floorf(x);
}

__device__ __forceinline__ float3 from_polar(float theta, float phi)
{
    const float cos_theta = -cosf(theta);
    const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    float sphi, cphi;
    sincosf(phi, &sphi, &cphi);
    return make_float3(-sin_theta * cphi, cos_theta, -sin_theta * sphi);
}

} // anonymous namespace

// The MDL backend emits the material/environment expression as an extern "C"
// function with the name supplied to translate_material_expression() — the
// baker uses "baker_lambda" (see baker.cpp create_baker_code_internal).
//
// Signature matches mi::neuraylib::Material_expr_function /
// Environment_function (target_code_types.h): the same pointer is used for
// both surface and environment baking — the state argument's runtime type is
// chosen by the caller.
extern "C" __device__ void baker_lambda(
    void                                *result,
    void const                          *state,
    mi::neuraylib::Resource_data const  *res_data,
    char const                          *arg_block_data);


// Stable C-linkage indirection so the host can find the device address of
// the texture-runtime vtable via cuModuleGetGlobal_v2. The unmangled name
// of `tex_vtable` itself depends on the C++ ABI; this pointer is reliable.
extern "C" __device__ const void* baker_oss_tex_vtable_addr = &tex_vtable;


// ----------------------------------------------------------------------------
// Surface baking
// ----------------------------------------------------------------------------

extern "C" __global__
void bake_surface(
    float4                                  *results,
    MI::BAKER::Bake_params_oss const         *params,
    char const                              *captured_args,
    mi::neuraylib::Texture_handler_base const *tex_handler)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params->width || y >= params->height)
        return;

    float3 uvw[BAKER_TEXTURE_SPACES];
    float3 t_u[BAKER_TEXTURE_SPACES];
    float3 t_v[BAKER_TEXTURE_SPACES];
    const bool position_is_direction =
        (params->state_flags & BAKER_STATE_POSITION_DIRECTION) != 0;
    for (uint32_t i = 0; i < BAKER_TEXTURE_SPACES; ++i) {
        uvw[i] = make_float3(0.0f, 0.0f, 0.0f);
        if (position_is_direction) {
            t_u[i] = make_float3(0.0f, 0.0f, 0.0f);
            t_v[i] = make_float3(0.0f, 0.0f, 0.0f);
        } else {
            t_u[i] = make_float3(1.0f, 0.0f, 0.0f);
            t_v[i] = make_float3(0.0f, 1.0f, 0.0f);
        }
    }

    const tct_float4 identity[4] = {
        {1.f, 0.f, 0.f, 0.f},
        {0.f, 1.f, 0.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
        {0.f, 0.f, 0.f, 1.f}
    };

    Shading_state_material state;
    if (position_is_direction) {
        state.normal      = make_float3(0.0f, 0.0f, 0.0f);
        state.geom_normal = make_float3(0.0f, 0.0f, 0.0f);
    } else {
        state.normal      = make_float3(0.0f, 0.0f, 1.0f);
        state.geom_normal = make_float3(0.0f, 0.0f, 1.0f);
    }
    state.animation_time        = params->animation_time;
    state.text_coords           = uvw;
    state.tangent_u             = t_u;
    state.tangent_v             = t_v;
    state.text_results          = nullptr;
    state.ro_data_segment       = nullptr;
    state.world_to_object       = identity;
    state.object_to_world       = identity;
    state.object_id             = 0;
    state.meters_per_scene_unit = 1.0f;

    Resource_data res_data = { nullptr, tex_handler };

    const float inv_spp = 1.0f / __uint2float_rn(params->num_samples);
    const float range_u = params->max_u - params->min_u;
    const float range_v = params->max_v - params->min_v;

    float4 result       = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 sample_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (unsigned int i = 0; i < params->num_samples; ++i) {
        const float u =
            (((__uint2float_rn(x) + fractf(__uint2float_rn(i) * inv_spp + 0.5f))
              * params->du) * range_u) + params->min_u;
        const float v =
            (((__uint2float_rn(y) + fractf(radinv2(i) + 0.5f))
              * params->dv) * range_v) + params->min_v;

        float3 pos;
        if (position_is_direction) {
            const float phi   = (float)(2.0 * M_PI) * u;
            const float theta = (float)(M_PI) * v;
            pos = from_polar(theta, phi);
            // uvw stays all-zero from its initialization above.
        } else {
            pos = make_float3(u, v, 0.0f);
            for (uint32_t k = 0; k < BAKER_TEXTURE_SPACES; ++k)
                uvw[k] = pos;
        }
        state.position = pos;

        baker_lambda(&sample_color, &state, &res_data, captured_args);

        sample_color.x = isnan(sample_color.x) ? 0.0f : sample_color.x;
        sample_color.y = isnan(sample_color.y) ? 0.0f : sample_color.y;
        sample_color.z = isnan(sample_color.z) ? 0.0f : sample_color.z;
        sample_color.w = isnan(sample_color.w) ? 0.0f : sample_color.w;

        // For bool (Sint8) targets the MDL lambda writes a raw byte (0x00/0x01)
        // into result[0].  That bit pattern is a denormal float; CUDA's DAZ mode
        // would flush it to 0.0f on the first += .  Convert to 0.0f/1.0f here so
        // normal float accumulation and the > 0.5 majority-vote threshold in the
        // host readback loop work correctly on both CPU and GPU.
        if (params->is_bool_target) {
            sample_color.x = (__float_as_uint(sample_color.x) != 0u) ? 1.0f : 0.0f;
            sample_color.y = 0.0f;
            sample_color.z = 0.0f;
            sample_color.w = 0.0f;
        }

        if (i == 0)
            result = sample_color;
        else
            result += sample_color;
    }

    if (params->num_samples > 1)
        result *= inv_spp;

    results[y * params->width + x] = result;
}


// ----------------------------------------------------------------------------
// Environment baking
// ----------------------------------------------------------------------------

extern "C" __global__
void bake_environment(
    float4                                  *results,
    MI::BAKER::Bake_params_oss const         *params,
    char const                              *captured_args,
    mi::neuraylib::Texture_handler_base const *tex_handler)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params->width || y >= params->height)
        return;

    Shading_state_environment state;
    state.ro_data_segment = nullptr;

    Resource_data res_data = { nullptr, tex_handler };

    const float inv_spp = 1.0f / __uint2float_rn(params->num_samples);
    const float range_u = params->max_u - params->min_u;
    const float range_v = params->max_v - params->min_v;

    float4 result       = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 sample_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (unsigned int i = 0; i < params->num_samples; ++i) {
        const float u =
            (((__uint2float_rn(x) + fractf(__uint2float_rn(i) * inv_spp + 0.5f))
              * params->du) * range_u) + params->min_u;
        const float v =
            (((__uint2float_rn(y) + fractf(radinv2(i) + 0.5f))
              * params->dv) * range_v) + params->min_v;

        const float phi   = (float)(2.0 * M_PI) * u;
        const float theta = (float)(M_PI) * v;
        state.direction = from_polar(theta, phi);

        baker_lambda(&sample_color, &state, &res_data, captured_args);

        sample_color.x = isnan(sample_color.x) ? 0.0f : sample_color.x;
        sample_color.y = isnan(sample_color.y) ? 0.0f : sample_color.y;
        sample_color.z = isnan(sample_color.z) ? 0.0f : sample_color.z;
        sample_color.w = isnan(sample_color.w) ? 0.0f : sample_color.w;

        if (i == 0)
            result = sample_color;
        else
            result += sample_color;
    }

    if (params->num_samples > 1)
        result *= inv_spp;

    results[y * params->width + x] = result;
}
