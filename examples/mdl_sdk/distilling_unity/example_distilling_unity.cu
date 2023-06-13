/******************************************************************************
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/distilling_unity/example_distilling_unity.cu
//
// This file contains the CUDA kernel used to evaluate the material sub-expressions.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "texture_support_cuda.h"
#include "example_distilling_unity.h"

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

// Lookup tables for baking oversampling
__constant__ float RADINV2[] = { 0, 0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f, 0.5625f, 0.3125f, 0.8125f, 0.1875f, 0.6875f, 0.4375f };
__constant__ float RADINV3[] = { 0, 0.333333f, 0.666667f, 0.111111f, 0.444444f, 0.777778f, 0.222222f, 0.555556f, 0.888889f, 0.037037f, 0.37037f, 0.703704f, 0.148148f, 0.481481f, 0.814815f };
__device__ float fractf(const float v) { return v - floorf(v); }

// CUDA kernel evaluating the MDL sub-expression for one texel.
extern "C" __global__ void evaluate_mat_expr(
    float *out_buf,
    Target_code_data *tc_data_list,
    char const **arg_block_list,
    unsigned int width,
    unsigned int height,
    unsigned int num_samples,
    size_t function_index,
    unsigned int num_channels,
    Expression_type expr_type,
    float metallic /* by convention > 0 if needs to be set*/
    )
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
    float tex_x = float(x) * step_x;         // [0, 1)
    float tex_y = float(y) * step_y;         // [0, 1)
    float pos_x = tex_x;
    float pos_y = tex_y;

    unsigned int tc_idx = mdl_target_code_indices[function_index];
    char const *arg_block = arg_block_list[mdl_arg_block_indices[function_index]];

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
        /*normal=*/           { 0.0f, 0.0f, 1.0f },
        /*geom_normal=*/      { 0.0f, 0.0f, 1.0f },
        /*position=*/         { pos_x, pos_y, 0.0f },
        /*animation_time=*/   0.0f,
        /*texture_coords=*/   texture_coords,
        /*tangent_u=*/        texture_tangent_u,
        /*tangent_v=*/        texture_tangent_v,
        /*text_results=*/     NULL,
        /*ro_data_segment=*/  0,
        /*world_to_object=*/  identity,
        /*object_to_world=*/  identity,
        /*object_id=*/        0
    };
    
    Tex_handler tex_handler;
    tex_handler.vtable       = &TEX_VTABLE;   // only required in 'vtable' mode, otherwise NULL
    tex_handler.num_textures = tc_data_list[tc_idx].num_textures;
    tex_handler.textures     = tc_data_list[tc_idx].textures;

    Resource_data res_data_pair = {
        NULL, reinterpret_cast<Texture_handler_base *>(&tex_handler) };

    // Super-sample the current texel with the given number of samples
    float4 res = make_float4(0, 0, 0, 0);
    for (unsigned int i = 0; i < num_samples; ++i) 
    {
        mdl_state.position.x = (x + fractf(RADINV2[i] + 0.5f)) * step_x;
        mdl_state.position.y = (y + fractf(RADINV3[i] + 0.5f)) * step_y;

#ifdef ENABLE_DERIVATIVES
        texture_coords[0].val.x = mdl_state.position.x;
        texture_coords[0].val.y = mdl_state.position.y;
#else
        texture_coords[0].x = mdl_state.position.x;
        texture_coords[0].y = mdl_state.position.y;
#endif

        // Add result for current sample
        float4 cur_res;
        mdl_functions[function_index](&cur_res, &mdl_state, &res_data_pair, NULL, arg_block);
        res.x += cur_res.x;
        res.y += cur_res.y;
        res.z += cur_res.z;
        res.w += cur_res.w;
    }

    res.x /= num_samples;
    res.y /= num_samples;
    res.z /= num_samples;
    res.w /= num_samples;

    const unsigned int result_idx = y * width + x;
    if (expr_type == EXT_BASE_COLOR)
    {
        // Gamma correction 
        const float gammainv(1.0f / 2.2f);
        res.x = powf(res.x, gammainv);
        res.y = powf(res.y, gammainv);
        res.z = powf(res.z, gammainv);
        res.w = 1.0f;
        ((float4*)out_buf)[result_idx] = res;
    }
    else if (expr_type == EXT_TRANSPARENCY)
    {
        // Transform transparency into opacity and multiply it in the w component
        ((float4*)out_buf)[result_idx].w *= 1.0f - res.x;
    }
    else if (expr_type == EXT_OPACITY)
    {
        // Multiply opactity with the w component
        ((float4*)out_buf)[result_idx].w *= res.x;
    }
    else if (expr_type == EXT_METALLIC)
    {
        // Red : Stores the metallic map.
        ((float4*)out_buf)[result_idx].x = res.x;
        // Green : Stores the ambient occlusion map.
        ((float4*)out_buf)[result_idx].y = 1.0f;
    }
    else if (expr_type == EXT_ROUGHNESS)
    {
        // Alpha: Stores the smoothness map.
        // smoothness = 1 - sqrt(roughness)
        ((float4*)out_buf)[result_idx].w = 1.0f - sqrtf(res.x);
        // Green : Stores the ambient occlusion map.
        ((float4*)out_buf)[result_idx].y = 1.0f;
        // If metallic has no bake_path, need to set its value here
        if (metallic > 0)
        {
            ((float4*)out_buf)[result_idx].x = metallic;
        }
    }
    else
    {
        if (num_channels == 1)
        {
            out_buf[result_idx] = res.x;
        }
        else if (num_channels == 3)
        {
            float3 r3 = { res.x, res.y, res.z };
            ((float3*)out_buf)[result_idx] = r3;
        }
        else
        {
            ((float4*)out_buf)[result_idx] = res;
        }
    }
}
