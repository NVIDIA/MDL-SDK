/******************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <vector_types.h>
#include <texture_types.h>

struct Target_code_data;

enum Mdl_test_type {
    MDL_TEST_EVAL    = 0,  // only use BSDF evaluation
    MDL_TEST_SAMPLE  = 1,  // only use BSDF sampling
    MDL_TEST_MIS     = 2,  // multiple importance sampling
    MDL_TEST_MIS_PDF = 3,  // multiple importance sampling, but use BSDF explicit pdf computation
    MDL_TEST_NO_ENV  = 4,  // no environment sampling
    MDL_TEST_COUNT
};

const unsigned MAX_DF_HANDLES = 8;

struct Env_accel {
    unsigned int alias;
    float q;
    float pdf;
};

namespace
{
    #if defined(__CUDA_ARCH__)
    __host__ __device__
    #endif
    inline uint2 make_invalid()
    {
        uint2 index_pair;
        index_pair.x = ~0;
        index_pair.y = ~0;
        return index_pair;
    }
}

struct Df_cuda_material
{
    #if defined(__CUDA_ARCH__)
    __host__ __device__
    #endif
    Df_cuda_material()
        : compiled_material_index(0)
        , argument_block_index(~0)
        , init(make_invalid())
        , bsdf(make_invalid())
        , edf(make_invalid())
        , emission_intensity(make_invalid())
        , volume_absorption(make_invalid())
        , thin_walled(make_invalid())
        , contains_hair_bsdf(0)
    {
    }

    // used on host side only
    unsigned int compiled_material_index;

    // the argument block index of this material (~0 if not used)
    unsigned int argument_block_index;

    // pair of target_code_index and function_index to identify the init function
    uint2 init;

    // pair of target_code_index and function_index to identify the bsdf
    uint2 bsdf;

    // pair of target_code_index and function_index to identify the edf
    uint2 edf;

    // pair of target_code_index and function_index for intensity
    uint2 emission_intensity;

    // pair of target_code_index and function_index for volume absorption
    uint2 volume_absorption;

    // pair of target_code_index and function_index for thin_walled
    uint2 thin_walled;

    // maps 'material tags' to 'global tags' for the surface scattering distribution function
    unsigned int bsdf_mtag_to_gtag_map[MAX_DF_HANDLES];
    unsigned int bsdf_mtag_to_gtag_map_size;

    // maps 'material tags' to 'global tags' for the emission distribution function
    unsigned int edf_mtag_to_gtag_map[MAX_DF_HANDLES];
    unsigned int edf_mtag_to_gtag_map_size;

    unsigned int contains_hair_bsdf;
};

enum Geometry_type
{
    GT_SPHERE = 0,  // Intersect a sphere with unit radius located at the (0,0,0)
    GT_HAIR = 1,    // Intersect an infinite cylinder at (0,0,0) oriented in y-direction
};

struct Kernel_params {
    // display
    uint2         resolution;
    float         exposure_scale;
    unsigned int *display_buffer;
    float3       *accum_buffer;
    float3       *albedo_buffer;
    float3       *normal_buffer;
    bool          enable_auxiliary_output;
    unsigned      display_buffer_index;

    // parameters
    unsigned int iteration_start;
    unsigned int iteration_num;
    unsigned int mdl_test_type;
    unsigned int max_path_length;
    unsigned int use_derivatives;
    unsigned int disable_aa;

    // camera
    float3 cam_pos;
    float3 cam_dir;
    float3 cam_right;
    float3 cam_up;
    float  cam_focal;

    // geometry
    unsigned int geometry;

    // environment
    uint2                env_size;
    cudaTextureObject_t  env_tex;
    Env_accel           *env_accel;
    float                env_intensity;         // scaling factor
    uint32_t             env_gtag;              // global light group tag for handle 'env'
    float                env_rotation;          // rotation of the environment

    // point light
    float3 light_pos;
    float3 light_color;
    float light_intensity;
    uint32_t point_light_gtag;                  // global light group tag for handle 'point_light'

    // material data
    Target_code_data   *tc_data;
    char const        **arg_block_list;
    unsigned int        current_material;
    Df_cuda_material   *material_buffer;

    // LPE state machine
    uint32_t            lpe_num_states;         // number of states in the state machine
    uint32_t            lpe_num_transitions;    // number of possible transitions between 2 states
    uint32_t           *lpe_state_table;        // actual machine; size: #states x #transitions
    uint32_t           *lpe_final_mask;         // encodes final states; size: #states
    uint32_t            default_gtag;           // tag ID for the empty string
    uint32_t            lpe_ouput_expression;   // the LPE evaluated for output
                                                // only one here, but additional one analogously
};

enum Display_buffer_options
{
    DISPLAY_BUFFER_LPE = 0,
    DISPLAY_BUFFER_ALBEDO,
    DISPLAY_BUFFER_NORMAL,

    DISPLAY_BUFFER_COUNT
};

#endif // EXAMPLE_DF_CUDA_H
