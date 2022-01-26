/******************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/execution_cuda/example_execution_cuda.cu
//
// This file contains the CUDA kernel used to evaluate the material sub-expressions.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "texture_support_cuda.h"

// To reuse this sample code for the MDL SDK and MDL Core the corresponding namespaces are used.

// when this CUDA code is used in the context of an SDK sample.
#if defined(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR)
    #define BSDF_USE_MATERIAL_IOR MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR
    using namespace mi::neuraylib;
// when this CUDA code is used in the context of an Core sample.
#elif defined(MDL_CORE_BSDF_USE_MATERIAL_IOR)
    #define BSDF_USE_MATERIAL_IOR MDL_CORE_BSDF_USE_MATERIAL_IOR
    using namespace mi::mdl;
#endif

#ifdef ENABLE_DERIVATIVES
typedef Material_expr_function_with_derivs Mat_expr_func;
typedef Shading_state_material_with_derivs Mdl_state;
typedef Texture_handler_deriv Tex_handler;
#define TEX_VTABLE tex_deriv_vtable
#else
typedef Material_expr_function Mat_expr_func;
typedef Shading_state_material Mdl_state;
typedef Texture_handler Tex_handler;
#define TEX_VTABLE tex_vtable
#endif


// Custom structure representing the resources used by the generated code of a target code object.
struct Target_code_data
{
    size_t       num_textures;      // number of elements in the textures field
    Texture     *textures;          // a list of Texture objects, if used
    char const  *ro_data_segment;   // the read-only data segment, if used
};


// The number of generated MDL sub-expression functions available.
extern __constant__ unsigned int     mdl_functions_count;

// The target argument block indices for the generated MDL sub-expression functions.
// Note: the original indices are incremented by one to allow us to use 0 as "not-used".
extern __constant__ unsigned int     mdl_arg_block_indices[];

// The function pointers of the generated MDL sub-expression functions.
// In this example it is assumed that only expressions are added to the link unit.
// For a more complex use case, see also example df_cuda.
extern __constant__ Mat_expr_func    *mdl_functions[];

// The target code indices for the generated MDL sub-expression functions.
// In contrast to the df_cuda sample, this example simply iterates over all generated expressions.
// Therefore, no target_code_indices and function_indices are passed from the host side.
// Instead, this additional array allows the mapping to target_code_index.
extern __constant__ unsigned int     mdl_target_code_indices[];

// Identity matrix.
// The last row is always implied to be (0, 0, 0, 1).
__constant__ const tct_float4 identity[3] = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f}
};


// Calculate radical inverse with base 2.
__device__ float radinv2(unsigned int bits)
{
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >> 8);
    bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >> 4);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >> 2);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >> 1);

    return float(bits) / float(0x100000000ULL);
}

// CUDA kernel evaluating the MDL sub-expression for one texel.
extern "C" __global__ void evaluate_mat_expr(
    float3 *out_buf,
    Target_code_data *tc_data_list,
    char const **arg_block_list,
    unsigned int width,
    unsigned int height,
    unsigned int num_samples)
{
    // Determine x and y coordinates of texel to be evaluated and check
    // whether it is out of bounds (due to block padding)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Calculate position and texture coordinates for a 2x2 quad around the center of the world
    float step_x = 1.f / width;
    float step_y = 1.f / height;
    float pos_x = 2.0f * x * step_x - 1.0f;  // [-1, 1)
    float pos_y = 2.0f * y * step_y - 1.0f;  // [-1, 1)
    float tex_x = float(x) * step_x;         // [0, 1)
    float tex_y = float(y) * step_y;         // [0, 1)

    // Assign materials in a checkerboard pattern
    unsigned int material_index =
        ((unsigned int)(tex_x * 4) ^ (unsigned int)(tex_y * 4)) % mdl_functions_count;
    unsigned int tc_idx = mdl_target_code_indices[material_index];
    char const *arg_block = arg_block_list[mdl_arg_block_indices[material_index]];

    // Setup MDL material state (with only one texture space)
#ifdef ENABLE_DERIVATIVES
    tct_deriv_float3 texture_coords[1] = {
        { { tex_x, tex_y, 0.0f }, { step_x, 0.0f, 0.0f }, { 0.0f, step_y, 0.0f } } };
#else
    tct_float3 texture_coords[1]    = { { tex_x, tex_y, 0.0f } };
#endif
    tct_float3 texture_tangent_u[1] = { { 1.0f, 0.0f, 0.0f } };
    tct_float3 texture_tangent_v[1] = { { 0.0f, 1.0f, 0.0f } };

    Mdl_state mdl_state = {
        /*normal=*/                { 0.0f, 0.0f, 1.0f },
        /*geom_normal=*/           { 0.0f, 0.0f, 1.0f },
#ifdef ENABLE_DERIVATIVES
        /*position=*/
        {
            { pos_x, pos_y, 0.0f },
            { 2 * step_x, 0.0f, 0.0f },
            { 0.0f, 2 * step_y, 0.0f }
        },
#else
        /*position=*/              { pos_x, pos_y, 0.0f },
#endif
        /*animation_time=*/        0.0f,
        /*texture_coords=*/        texture_coords,
        /*tangent_u=*/             texture_tangent_u,
        /*tangent_v=*/             texture_tangent_v,
        /*text_results=*/          NULL,
        /*ro_data_segment=*/       tc_data_list[tc_idx].ro_data_segment,
        /*world_to_object=*/       identity,
        /*object_to_world=*/       identity,
        /*object_id=*/             0,
        /*meters_per_scene_unit=*/ 1.0f
    };

    Tex_handler tex_handler;
    tex_handler.vtable       = &TEX_VTABLE;   // only required in 'vtable' mode, otherwise NULL
    tex_handler.num_textures = tc_data_list[tc_idx].num_textures;
    tex_handler.textures     = tc_data_list[tc_idx].textures;

    Resource_data res_data_pair = {
        NULL, reinterpret_cast<Texture_handler_base *>(&tex_handler) };

    // Super-sample the current texel with the given number of samples
    float3 res = make_float3(0, 0, 0);
    for (unsigned int i = 0; i < num_samples; ++i) {
        // Calculate the offset for the current sample
        float offs_x = float(i) / num_samples * step_x;
        float offs_y = radinv2(i) * step_y;

        // Update the position and the texture coordinate
#ifdef ENABLE_DERIVATIVES
        mdl_state.position.val.x = pos_x + 2 * offs_x;
        mdl_state.position.val.y = pos_y + 2 * offs_y;
        texture_coords[0].val.x = tex_x + offs_x;
        texture_coords[0].val.y = tex_y + offs_y;
#else
        mdl_state.position.x = pos_x + 2 * offs_x;
        mdl_state.position.y = pos_y + 2 * offs_y;
        texture_coords[0].x = tex_x + offs_x;
        texture_coords[0].y = tex_y + offs_y;
#endif

        // Add result for current sample
        float3 cur_res;
        mdl_functions[material_index](&cur_res, &mdl_state, &res_data_pair, NULL, arg_block);
        res.x += cur_res.x;
        res.y += cur_res.y;
        res.z += cur_res.z;
    }

    // Calculate average over all samples and apply gamma correction
    res.x = powf(res.x / num_samples, 1.f / 2.2f);
    res.y = powf(res.y / num_samples, 1.f / 2.2f);
    res.z = powf(res.z / num_samples, 1.f / 2.2f);

    // Write result to output buffer
    out_buf[y * width + x] = res;
}
