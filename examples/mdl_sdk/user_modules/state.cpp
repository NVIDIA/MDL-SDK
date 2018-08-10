/***************************************************************************************************
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
 **************************************************************************************************/

//
// Compile with
// clang.exe -I<CUDA include path> -emit-llvm -c -O2 -ffast-math -target x86_64-pc-win32 state.cpp
//
// The target ensures that one fixed ABI will be used, so we know how to process the functions.
//

#include "mdl_user_modules.h"


//
// The state structures used inside the renderer.
//

/// The state of the MDL environment function.
struct User_state_environment {
    float3                direction;               ///< state::direction() result
};

/// The MDL material state inside the MDL SDK.
struct User_state_material
{
    float3                normal;                  ///< state::normal() result
    float3                geom_normal;             ///< state::geom_normal() result
    float3                position;                ///< state::position() result
    float                 animation_time;          ///< state::animation_time() result
    const float3         *text_coords;             ///< state::texture_coordinate() table
    const float3         *tangent_u;               ///< state::texture_tangent_u() table
    const float3         *tangent_v;               ///< state::texture_tangent_v() table
    const float4         *text_results;            ///< texture results lookup table
    const char           *ro_data_segment;         ///< read-only data segment

    // these fields are used only if the uniform state is included
    const float4         *world_to_object;         ///< world-to-object transform matrix
    const float4         *object_to_world;         ///< object-to-world transform matrix
    int                   object_id;               ///< state::object_id() result
};


// Convert the given matrix from row major to column major and set last row to (0, 0, 0, 1).
static vfloat4x4 make_matrix_rm2cm(float4 const *v)
{
    return vfloat4x4{
        v[0].x, v[1].x, v[2].x, 0,
        v[0].y, v[1].y, v[2].y, 0,
        v[0].z, v[1].z, v[2].z, 0,
        v[0].w, v[1].w, v[2].w, 1
    };
}


namespace state {

//
// Environment state functions
//


// float3 direction() varying
vfloat3 direction(State_environment const *state_context)
{
    User_state_environment const *state =
        reinterpret_cast<User_state_environment const *>(state_context);
    return make_vector(state->direction);
}



//
// Core state functions
//

// float3 position() varying
vfloat3 position(State_core const *state_context)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return make_vector(state->position);
}


// float3 normal() varying
vfloat3 normal(State_core const *state_context)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return make_vector(state->normal);
}

// Special function required by the init function for distribution functions for updating
// the normal field in the state.
void set_normal(State_core *state_context, vfloat3 new_normal)
{
    User_state_material *state = reinterpret_cast<User_state_material *>(state_context);
    state->normal.x = new_normal.x;
    state->normal.y = new_normal.y;
    state->normal.z = new_normal.z;
}


// float3 geometry_normal() varying
vfloat3 geometry_normal(State_core const *state_context)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return make_vector(state->geom_normal);
}


// float3 motion() varying
vfloat3 motion(State_core const *state_context)
{
    return vfloat3{ 0, 0, 0 };
}


// int texture_space_max()
// Use default implementation: return value of "num_texture_spaces" option


// float3 texture_coordinate(int index) varying
vfloat3 texture_coordinate(
    State_core const *state_context,
    Exception_state const *exc_state,
    int index)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return make_vector(state->text_coords[index]);
}


// float3 texture_tangent_u(int index) varying
vfloat3 texture_tangent_u(
    State_core const *state_context,
    Exception_state const *exc_state,
    int index)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return make_vector(state->tangent_u[index]);
}


// float3 texture_tangent_v(int index) varying
vfloat3 texture_tangent_v(
    State_core const *state_context,
    Exception_state const *exc_state,
    int index)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return make_vector(state->tangent_v[index]);
}


// float3x3 tangent_space(int index) varying
vfloat3x3 tangent_space(
    State_core const *state_context,
    Exception_state const *exc_state,
    int index)
{
    return vfloat3x3(0);
}


// float3 geometry_tangent_u(int index) varying
vfloat3 geometry_tangent_u(
    State_core const *state_context,
    Exception_state const *exc_state,
    int index)
{
    return vfloat3{ 0, 0, 0 };
}


// float3 geometry_tangent_v(int index) varying
vfloat3 geometry_tangent_v(
    State_core const *state_context,
    Exception_state const *exc_state,
    int index)
{
    return vfloat3{ 0, 0, 0 };
}


// float animation_time() varying
float animation_time(State_core const *state_context)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return state->animation_time;
}


// float[WAVELENGTH_BASE_MAX] wavelength_base() uniform
float *wavelength_base(State_core const *state_context)
{
    return nullptr;
}


// float4x4 transform(
//     coordinate_space from,
//     coordinate_space to) uniform
vfloat4x4 transform(State_core const *state_context, coordinate_space from, coordinate_space to)
{
    if (from == coordinate_internal) from = INTERNAL_SPACE;
    if (to == coordinate_internal) to = INTERNAL_SPACE;

    if (from == to) {
        return vfloat4x4{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
    }

    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    float4 const *transform_matrix =
        (from == coordinate_world) ? state->world_to_object : state->object_to_world;
    return make_matrix_rm2cm(transform_matrix);
}


// float3 transform_point(
//     coordinate_space from,
//     coordinate_space to,
//     float3 point) uniform
vfloat3 transform_point(
    State_core const *state_context,
    coordinate_space from,
    coordinate_space to,
    vfloat3 point)
{
    if (from == coordinate_internal) from = INTERNAL_SPACE;
    if (to == coordinate_internal) to = INTERNAL_SPACE;

    if (from == to) return point;

    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    float4 const *mat =
        (from == coordinate_world) ? state->world_to_object : state->object_to_world;

    return vfloat3{
        point.x * mat[0].x + point.y * mat[0].y + point.z * mat[0].z + mat[0].w,
        point.x * mat[1].x + point.y * mat[1].y + point.z * mat[1].z + mat[1].w,
        point.x * mat[2].x + point.y * mat[2].y + point.z * mat[2].z + mat[2].w
    };
}


// float3 transform_vector(
//     coordinate_space from,
//     coordinate_space to,
//     float3 vector) uniform
vfloat3 transform_vector(
    State_core const *state_context,
    coordinate_space from,
    coordinate_space to,
    vfloat3 vector)
{
    if (from == coordinate_internal) from = INTERNAL_SPACE;
    if (to == coordinate_internal) to = INTERNAL_SPACE;

    if (from == to) return vector;

    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    float4 const *mat =
        (from == coordinate_world) ? state->world_to_object : state->object_to_world;

    return vfloat3{
        vector.x * mat[0].x + vector.y * mat[0].y + vector.z * mat[0].z,
        vector.x * mat[1].x + vector.y * mat[1].y + vector.z * mat[1].z,
        vector.x * mat[2].x + vector.y * mat[2].y + vector.z * mat[2].z
    };
}


// float3 transform_normal(
//     coordinate_space from,
//     coordinate_space to,
//     float3 normal) uniform
vfloat3 transform_normal(
    State_core const *state_context,
    coordinate_space from,
    coordinate_space to,
    vfloat3 normal)
{
    if (from == coordinate_internal) from = INTERNAL_SPACE;
    if (to == coordinate_internal) to = INTERNAL_SPACE;

    if (from == to) return normal;

    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);

    // the inverse matrix of world_to_object is object_to_world and vice versa
    float4 const *inv_mat =
        (from == coordinate_world) ? state->object_to_world : state->world_to_object;

    // multiply with transpose of inversed matrix
    return vfloat3{
        normal.x * inv_mat[0].x + normal.y * inv_mat[1].x + normal.z * inv_mat[2].x,
        normal.x * inv_mat[0].y + normal.y * inv_mat[1].y + normal.z * inv_mat[2].y,
        normal.x * inv_mat[0].z + normal.y * inv_mat[1].z + normal.z * inv_mat[2].z
    };
}


// float transform_scale(
//     coordinate_space from,
//     coordinate_space to,
//     float scale) uniform
float transform_scale(
    State_core const *state_context,
    coordinate_space from,
    coordinate_space to,
    float scale)
{
    if (from == coordinate_internal) from = INTERNAL_SPACE;
    if (to == coordinate_internal) to = INTERNAL_SPACE;

    if (from == to) return scale;

    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    float4 const *mat =
        (from == coordinate_world) ? state->world_to_object : state->object_to_world;

    // t_0 = || transform_vector(float3(1,0,0), from, to) || = | mat[0] |
    // t_1 = || transform_vector(float3(0,1,0), from, to) || = | mat[1] |
    // t_2 = || transform_vector(float3(0,0,1), from, to) || = | mat[2] |
    // res = scale * (t_0 + t_1 + t_2) / 3

    float t_0 = math::length(make_vector(*reinterpret_cast<const float3 *>(&mat[0])));
    float t_1 = math::length(make_vector(*reinterpret_cast<const float3 *>(&mat[1])));
    float t_2 = math::length(make_vector(*reinterpret_cast<const float3 *>(&mat[2])));
    return scale * ((t_0 + t_1 + t_2) / 3.0f);
}


// float3 rounded_corner_normal(
//     uniform float radius = 0.0,
//     uniform bool  across_materials = false,
//     uniform float roundness = 1.0
//     ) varying
// Use default implementation -> return state::normal()


// float meters_per_scene_unit() uniform
// Use default implementation -> return value provided to backend


// float scene_units_per_meter() uniform
// Use default implementation -> return inverse meters_per_scene_unit value provided to backend


// int object_id() uniform
int object_id(State_core const *state_context)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return state->object_id;
}


// float wavelength_min() uniform
// Use default implementation -> return value provided to backend


// float wavelength_max() uniform
// Use default implementation -> return value provided to backend


// Get the texture results table, required by the distribution functions.
// Init will write the texture results, sample, evaluate and PDF will only read it.
float4 *get_texture_results(State_core *state_context)
{
    User_state_material *state = reinterpret_cast<User_state_material *>(state_context);
    return const_cast<float4 *>(state->text_results);
}

// Get the read-only data segment.
const char *get_ro_data_segment(State_core const *state_context)
{
    User_state_material const *state = reinterpret_cast<User_state_material const *>(state_context);
    return state->ro_data_segment;
}

}  // namespace state
