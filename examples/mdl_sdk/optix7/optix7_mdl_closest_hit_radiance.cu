/******************************************************************************
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
 *****************************************************************************/

#include <optix.h>

#include "optix7_mdl.h"

#define TEX_SUPPORT_NO_VTABLES
#define TEX_SUPPORT_NO_DUMMY_SCENEDATA
#include "texture_support_cuda.h"  // texture runtime


// TODO: use matrix from OptiX
__device__ const float4 identity[3] = {
    { 1.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 1.0f, 0.0f }
};


#ifdef ENABLE_DERIVATIVES
typedef mi::neuraylib::Material_expr_function_with_derivs   Mat_expr_func;
typedef mi::neuraylib::Bsdf_init_function_with_derivs       Bsdf_init_func;
typedef mi::neuraylib::Bsdf_sample_function_with_derivs     Bsdf_sample_func;
typedef mi::neuraylib::Bsdf_evaluate_function_with_derivs   Bsdf_evaluate_func;
typedef mi::neuraylib::Shading_state_material_with_derivs   Mdl_state;
#else
typedef mi::neuraylib::Material_expr_function               Mat_expr_func;
typedef mi::neuraylib::Bsdf_init_function                   Bsdf_init_func;
typedef mi::neuraylib::Bsdf_sample_function                 Bsdf_sample_func;
typedef mi::neuraylib::Bsdf_evaluate_function               Bsdf_evaluate_func;
typedef mi::neuraylib::Shading_state_material               Mdl_state;
#endif

#ifdef NO_DIRECT_CALL

//
// Declarations of generated MDL functions
//

extern "C" __device__ Bsdf_init_func     mdlcode_init;
extern "C" __device__ Bsdf_sample_func   mdlcode_sample;
extern "C" __device__ Bsdf_evaluate_func mdlcode_evaluate;
extern "C" __device__ Mat_expr_func      mdlcode_thin_walled;


//
// Functions needed by texture runtime when this file is compiled with Clang
//

extern "C" __device__ __inline__ void __itex2D_float(
    float *retVal, cudaTextureObject_t texObject, float x, float y)
{
    float4 tmp;
    asm volatile ("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y));
    *retVal = (float)(tmp.x);
}

extern "C" __device__ __inline__ void __itex2D_float4(
    float4 *retVal, cudaTextureObject_t texObject, float x, float y)
{
    float4 tmp;
    asm volatile ("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex2DGrad_float4(
    float4 *retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
    float4 tmp;
    asm volatile ("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, {%9, %10};"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y), "f"(dPdx.x), "f"(dPdx.y), "f"(dPdy.x), "f"(dPdy.y));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex2DLod_float4(
    float4 *retVal, cudaTextureObject_t texObject, float x, float y, float level)
{
    float4 tmp;
    asm volatile ("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y), "f"(level));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex3D_float(
    float *retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile ("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = (float)(tmp.x);
}

extern "C" __device__ __inline__ void __itex3D_float4(
    float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile ("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itexCubemap_float4(
    float4 *retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile ("tex.cube.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

#else

#define mdlcode_init(state, res_data, exception_state, arg_block_data)                 \
    optixDirectCall<void>(rt_data->mdl_callable_base_index,                            \
        state, res_data, nullptr, arg_block_data)

#define mdlcode_sample(data, state, res_data, exception_state, arg_block_data)         \
    optixDirectCall<void>(rt_data->mdl_callable_base_index + 1,                        \
        data, state, res_data, nullptr, arg_block_data)

#define mdlcode_evaluate(data, state, res_data, exception_state, arg_block_data)       \
    optixDirectCall<void>(rt_data->mdl_callable_base_index + 2,                        \
        data, state, res_data, nullptr, arg_block_data)

#define mdlcode_thin_walled(result, state, res_data, exception_state, arg_block_data)  \
    optixDirectCall<void>(rt_data->mdl_callable_base_index + 4,                        \
        result, state, res_data, nullptr, arg_block_data)

#endif


static __forceinline__ __device__ float3 get_barycentrics()
{
    const float2 bary = optixGetTriangleBarycentrics();
    return make_float3(1.0f - bary.x - bary.y, bary.x, bary.y);
}


static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    uint32_t occluded = 0u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RAY_TYPE_OCCLUSION,      // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_OCCLUSION,      // missSBTIndex
        occluded);
    return occluded;
}


// Implementation of scene_data_isvalid().
extern "C" __device__ bool scene_data_isvalid(
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id)
{
    if (scene_data_id == 0)
        return false;

    HitGroupData const *rt_data = reinterpret_cast<HitGroupData const*>(optixGetSbtDataPointer());
    SceneDataInfo const *info = rt_data->scene_data_info + scene_data_id;
    return info->data_kind != SceneDataInfo::DK_NONE;
}

// Implementation of scene_data_lookup_float4().
extern "C" __device__ void scene_data_lookup_float4(
    float                                  result[4],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value[4],
    bool                                   uniform_lookup)
{
    if (scene_data_id == 0) {
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    HitGroupData const *rt_data = reinterpret_cast<HitGroupData const*>(optixGetSbtDataPointer());
    SceneDataInfo const *info = rt_data->scene_data_info + scene_data_id;
    if (info->data_kind == SceneDataInfo::DK_NONE || (uniform_lookup && !info->is_uniform)) {
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    const int    prim_idx = optixGetPrimitiveIndex();
    const float3 barycentrics = get_barycentrics();

    MeshVertex const &v0 = rt_data->vertices[rt_data->indices[prim_idx].x];
    MeshVertex const &v1 = rt_data->vertices[rt_data->indices[prim_idx].y];
    MeshVertex const &v2 = rt_data->vertices[rt_data->indices[prim_idx].z];

    float4 val0, val1, val2;

    switch (info->data_kind) {
    case SceneDataInfo::DK_VERTEX_COLOR:
        val0 = make_float4(v0.color.x, v0.color.y, v0.color.z, 0);
        val1 = make_float4(v1.color.x, v1.color.y, v1.color.z, 0);
        val2 = make_float4(v2.color.x, v2.color.y, v2.color.z, 0);
        break;

    case SceneDataInfo::DK_ROW_COLUMN:
        // converts the integers to floating point numbers
        val0 = make_float4(v0.row_column.x, v0.row_column.y, 0, 0);
        val1 = make_float4(v1.row_column.x, v1.row_column.y, 0, 0);
        val2 = make_float4(v2.row_column.x, v2.row_column.y, 0, 0);
        break;

    default:
        break;
    }

    float4 res_vector;
    switch (info->interpolation_mode) {
    case SceneDataInfo::IM_LINEAR:
        res_vector = val0 * barycentrics.x + val1 * barycentrics.y + val2 * barycentrics.z;
        break;

    case SceneDataInfo::IM_NEAREST:
        res_vector =
            barycentrics.x > barycentrics.y ?
            (barycentrics.x > barycentrics.z ? val0 : val2)
            :
            (barycentrics.y > barycentrics.z ? val1 : val2);
        break;

    default:
        // unsupported interpolation mode
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    result[0] = res_vector.x;
    result[1] = res_vector.y;
    result[2] = res_vector.z;
    result[3] = res_vector.w;
}


// Implementation of scene_data_lookup_float3().
extern "C" __device__ void scene_data_lookup_float3(
    float                                  result[3],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value[3],
    bool                                   uniform_lookup)
{
    float res4[4];
    float def_val4[4] = { default_value[0], default_value[1], default_value[2], 0.0f };
    scene_data_lookup_float4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    result[0] = res4[0];
    result[1] = res4[1];
    result[2] = res4[2];
}


// Implementation of scene_data_lookup_color().
extern "C" __device__ void scene_data_lookup_color(
    float                                  result[3],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value[3],
    bool                                   uniform_lookup)
{
    float res4[4];
    float def_val4[4] = { default_value[0], default_value[1], default_value[2], 0.0f };
    scene_data_lookup_float4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    result[0] = res4[0];
    result[1] = res4[1];
    result[2] = res4[2];
}


// Implementation of scene_data_lookup_float2().
extern "C" __device__ void scene_data_lookup_float2(
    float                                  result[2],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value[2],
    bool                                   uniform_lookup)
{
    float res4[4];
    float def_val4[4] = { default_value[0], default_value[1], 0.0f, 0.0f };
    scene_data_lookup_float4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    result[0] = res4[0];
    result[1] = res4[1];
}


// Implementation of scene_data_lookup_float().
extern "C" __device__ float scene_data_lookup_float(
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    float                                  default_value,
    bool                                   uniform_lookup)
{
    float res4[4];
    float def_val4[4] = { default_value, 0.0f, 0.0f, 0.0f };
    scene_data_lookup_float4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    return res4[0];
}


// Implementation of scene_data_lookup_int4().
extern "C" __device__ void scene_data_lookup_int4(
    int                                    result[4],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    int                                    default_value[4],
    bool                                   uniform_lookup)
{
    if (scene_data_id == 0) {
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    HitGroupData const *rt_data = reinterpret_cast<HitGroupData const*>(optixGetSbtDataPointer());
    SceneDataInfo const *info = rt_data->scene_data_info + scene_data_id;
    if (info->data_kind == SceneDataInfo::DK_NONE || (uniform_lookup && !info->is_uniform)) {
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    const int    prim_idx = optixGetPrimitiveIndex();
    const float3 barycentrics = get_barycentrics();

    MeshVertex const &v0 = rt_data->vertices[rt_data->indices[prim_idx].x];
    MeshVertex const &v1 = rt_data->vertices[rt_data->indices[prim_idx].y];
    MeshVertex const &v2 = rt_data->vertices[rt_data->indices[prim_idx].z];

    int4 val0, val1, val2;

    switch (info->data_kind) {
    case SceneDataInfo::DK_VERTEX_COLOR:
        // converts the floating point numbers to integers
        val0 = make_int4(v0.color.x, v0.color.y, v0.color.z, 0);
        val1 = make_int4(v1.color.x, v1.color.y, v1.color.z, 0);
        val2 = make_int4(v2.color.x, v2.color.y, v2.color.z, 0);
        break;

    case SceneDataInfo::DK_ROW_COLUMN:
        val0 = make_int4(v0.row_column.x, v0.row_column.y, 0, 0);
        val1 = make_int4(v1.row_column.x, v1.row_column.y, 0, 0);
        val2 = make_int4(v2.row_column.x, v2.row_column.y, 0, 0);
        break;

    default:
        // unsupported data kind
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    int4 res_vector;
    switch (info->interpolation_mode) {
    case SceneDataInfo::IM_LINEAR:
        res_vector = make_int4(
            make_float4(val0) * barycentrics.x +
            make_float4(val1) * barycentrics.y +
            make_float4(val2) * barycentrics.z);
        break;

    case SceneDataInfo::IM_NEAREST:
        res_vector =
            barycentrics.x > barycentrics.y ?
            (barycentrics.x > barycentrics.z ? val0 : val2)
            :
            (barycentrics.y > barycentrics.z ? val1 : val2);
        break;

    default:
        // unsupported interpolation mode
        result[0] = default_value[0];
        result[1] = default_value[1];
        result[2] = default_value[2];
        result[3] = default_value[3];
        return;
    }

    result[0] = res_vector.x;
    result[1] = res_vector.y;
    result[2] = res_vector.z;
    result[3] = res_vector.w;
}


// Implementation of scene_data_lookup_int3().
extern "C" __device__ void scene_data_lookup_int3(
    int                                    result[3],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    int                                    default_value[3],
    bool                                   uniform_lookup)
{
    int res4[4];
    int def_val4[4] = { default_value[0], default_value[1], default_value[2], 0 };
    scene_data_lookup_int4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    result[0] = res4[0];
    result[1] = res4[1];
    result[2] = res4[2];
}


// Implementation of scene_data_lookup_int2().
extern "C" __device__ void scene_data_lookup_int2(
    int                                    result[2],
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    int                                    default_value[2],
    bool                                   uniform_lookup)
{
    int res4[4];
    int def_val4[4] = { default_value[0], default_value[1], 0, 0 };
    scene_data_lookup_int4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    result[0] = res4[0];
    result[1] = res4[1];
}


// Implementation of scene_data_lookup_int().
extern "C" __device__ int scene_data_lookup_int(
    Texture_handler_base const            *self_base,
    mi::neuraylib::Shading_state_material *state,
    unsigned                               scene_data_id,
    int                                    default_value,
    bool                                   uniform_lookup)
{
    int res4[4];
    int def_val4[4] = { default_value, 0, 0, 0 };
    scene_data_lookup_int4(
        res4,
        self_base,
        state,
        scene_data_id,
        def_val4,
        uniform_lookup);
    return res4[0];
}


//------------------------------------------------------------------------------
//
// Closest-hit function of radiance ray
//
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    const int    prim_idx     = optixGetPrimitiveIndex();
    const float3 ray_dir      = optixGetWorldRayDirection();
    const float3 barycentrics = get_barycentrics();

    MeshVertex const &v0 = rt_data->vertices[rt_data->indices[prim_idx].x];
    MeshVertex const &v1 = rt_data->vertices[rt_data->indices[prim_idx].y];
    MeshVertex const &v2 = rt_data->vertices[rt_data->indices[prim_idx].z];

    // Calculate more precise intersection point based on barycentrics instead of the ray equation
    // (see Ray Tracing Gems, Ch. 6)
    const float3 P0 = v0.position;
    const float3 P1 = v1.position;
    const float3 P2 = v2.position;
    const float3 P  = optixTransformPointFromObjectToWorldSpace(
        P0 * barycentrics.x + P1 * barycentrics.y + P2 * barycentrics.z);

    const float3 geom_normal = optixTransformNormalFromObjectToWorldSpace(
        normalize(cross(P1 - P0, P2 - P0)));

    const float3 N0 = v0.normal;
    const float3 N1 = v1.normal;
    const float3 N2 = v2.normal;
    const float3 N  = optixTransformNormalFromObjectToWorldSpace(
        normalize(N0 * barycentrics.x + N1 * barycentrics.y + N2 * barycentrics.z));

    const float3 T0 = v0.tangent;
    const float3 T1 = v1.tangent;
    const float3 T2 = v2.tangent;
    const float3 T  = normalize(T0 * barycentrics.x + T1 * barycentrics.y + T2 * barycentrics.z);

    const float3 B0 = v0.binormal;
    const float3 B1 = v1.binormal;
    const float3 B2 = v2.binormal;
    const float3 B  = normalize(B0 * barycentrics.x + B1 * barycentrics.y + B2 * barycentrics.z);

    const float2 UV0 = v0.tex_coord;
    const float2 UV1 = v1.tex_coord;
    const float2 UV2 = v2.tex_coord;
    const float2 UV  = UV0 * barycentrics.x + UV1 * barycentrics.y + UV2 * barycentrics.z;

#ifdef ENABLE_DERIVATIVES
    // use fake derivatives just for testing
    const mi::neuraylib::tct_deriv_float3 text_coords = {
        make_float3(UV.x, UV.y, 0),
        make_float3(1, 0, 0),
        make_float3(0, 1, 0)
    };
#else
    const float3 text_coords = make_float3(UV.x, UV.y, 0);
#endif

    RadiancePRD* prd = get_radiance_prd();

    // setup state
    float4 texture_results[16];
    Mdl_state state;
    state.normal = N;
    state.geom_normal = geom_normal;
#ifdef ENABLE_DERIVATIVES
    // use fake derivatives just for testing
    state.position.val = P;
    state.position.dx = make_float3(1, 0, 0);
    state.position.dy = make_float3(0, 1, 0);
#else
    state.position = P;
#endif
    state.animation_time = 0.0f;
    state.text_coords = &text_coords;
    state.tangent_u = &T;
    state.tangent_v = &B;
    state.text_results = texture_results;
    state.ro_data_segment = nullptr;
    state.world_to_object = (float4 *)&identity;
    state.object_to_world = (float4 *)&identity;
    state.object_id = 0;
    state.meters_per_scene_unit = 1.0f;

    mi::neuraylib::Resource_data res_data = {
        nullptr,
        rt_data->texture_handler
    };

    mdlcode_init(&state, &res_data, nullptr, rt_data->arg_block);

    bool thin_walled;
    mdlcode_thin_walled(&thin_walled, &state, &res_data, nullptr, rt_data->arg_block);

    uint32_t seed = prd->seed;
    float3 cur_weight = prd->weight;
    bool is_inside = (get_radiance_payload_ray_flags() & RAY_FLAGS_INSIDE) != 0;

    // importance sample BSDF
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);
        const float z3 = rnd(seed);
        const float z4 = rnd(seed);

        mi::neuraylib::Bsdf_sample_data sample_data;
        if (is_inside && !thin_walled)
        {
            sample_data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
            sample_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
        }
        else
        {
            sample_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
            sample_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
        }
        sample_data.k1 = -ray_dir;
        sample_data.xi = make_float4(z1, z2, z3, z4);

        mdlcode_sample(&sample_data, &state, &res_data, nullptr, rt_data->arg_block);

        // stop on absorption
        if (sample_data.event_type == mi::neuraylib::BSDF_EVENT_ABSORB)
        {
            set_radiance_payload_depth(MAX_DEPTH);
        }
        else
        {
            prd->direction = sample_data.k2;
            prd->weight *= sample_data.bsdf_over_pdf;

            if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0)
                prd->last_pdf = -1.0f;
            else
                prd->last_pdf = sample_data.pdf;

            if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
            {
                set_radiance_payload_ray_flags(
                    RayFlags(int(get_radiance_payload_ray_flags()) ^ RAY_FLAGS_INSIDE));

                // continue on the opposite side
                prd->origin = offset_ray(P, -geom_normal);
            }
            else
            {
                // continue on the current side
                prd->origin = offset_ray(P, geom_normal);
            }
        }
    }

    float3 to_light;
    float light_dist;
    float pdf = 0.0f;
    const float3 radiance_over_pdf = sample_lights(P, to_light, light_dist, pdf, seed);

    prd->seed = seed;

    const float cos_theta = dot(to_light, N);
    {
        float3 light_contrib;
        if (cos_theta > 0.0f && pdf != 0.0f)
        {
            mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;
            if (is_inside && !thin_walled)
            {
                eval_data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
                eval_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
            }
            else
            {
                eval_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
                eval_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
            }
            eval_data.k1 = -ray_dir;
            eval_data.k2 = to_light;
            eval_data.bsdf_diffuse = make_float3(0.0f);
            eval_data.bsdf_glossy = make_float3(0.0f);

            mdlcode_evaluate(&eval_data, &state, &res_data, nullptr, rt_data->arg_block);

            const float mis_weight = pdf == DIRAC
                ? 1.0f
                : pdf / (pdf + eval_data.pdf);

            light_contrib = cur_weight * radiance_over_pdf * mis_weight
                * (eval_data.bsdf_diffuse + eval_data.bsdf_glossy);
        }
        else
        {
            light_contrib = make_float3(0);
        }

        const bool occluded = traceOcclusion(
            cos_theta > 0.0f && pdf != 0.0f ? params.handle : 0,
            prd->origin,
            to_light,
            0.01f,              // tmin
            light_dist - 0.01f  // tmax
        );

        if (!occluded)
        {
#ifdef CONTRIB_IN_PAYLOAD
            set_radiance_payload_contrib(get_radiance_payload_contrib() + light_contrib);
#else
            prd->contribution += light_contrib;
#endif
        }
    }
}
