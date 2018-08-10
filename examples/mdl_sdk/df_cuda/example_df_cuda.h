/******************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef EXAMPLE_DF_CUDA_H
#define EXAMPLE_DF_CUDA_H

#include <vector_types.h>
#include <texture_types.h>

struct Target_code_data;

enum Mdl_test_type {
    MDL_TEST_EVAL    = 0, // only use BSDF evaluation
    MDL_TEST_SAMPLE  = 1, // only use BSDF sampling
    MDL_TEST_MIS     = 2, // multiple importance sampling
    MDL_TEST_MIS_PDF = 3, // multiple importance sampling, but use BSDF explicit pdf computation
    MDL_TEST_NO_ENV  = 4, // no environment sampling
    MDL_TEST_COUNT
};

struct Env_accel {
    unsigned int alias;
    float q;
    float pdf;
};

struct Kernel_params {
    // display
    uint2 resolution;
    float exposure_scale;
    unsigned int *display_buffer;
    float3 *accum_buffer;

    // parameters
    unsigned int iteration_start;
    unsigned int iteration_num;
    unsigned int mdl_test_type;

    // camera
    float3 cam_pos;
    float3 cam_dir;
    float3 cam_right;
    float3 cam_up;
    float  cam_focal;

    // environment
    uint2 env_size;
    cudaTextureObject_t env_tex;
    Env_accel *env_accel;

    // point light
    float3 light_pos;
    float3 light_intensity;

    // material data
    Target_code_data *tc_data;
    char const **arg_block_list;
    unsigned int current_material;
};

#endif // EXAMPLE_DF_CUDA_H

