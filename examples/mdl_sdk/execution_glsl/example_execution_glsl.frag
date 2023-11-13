/******************************************************************************
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/execution_glsl/example_execution_glsl.frag
//
// This file contains the implementations of the texture access functions
// and the fragment shader used to evaluate the material sub-expressions.

// Material pattern as chosen by the user.
uniform int material_pattern;

// Current time in seconds since the start of the render loop.
uniform float animation_time;

// Mapping from material index to start index in material_texture_samplers_2d.
uniform int material_texture_starts[MAX_MATERIALS];

// Array containing all 2D texture samplers of all used materials.
uniform sampler2D material_texture_samplers_2d[MAX_TEXTURES];


// Start offset of the current material inside material_texture_samplers_2d, set in main.
int tex_start;


// The input variables coming from the vertex shader.
in vec3[1] texture_coordinate;    // used for state::texture_coordinate(tex_space) in "arg" mode
in vec3    vPosition;


// The color output variable of this fragment shader.
out vec4 FragColor;


// The MDL material state structure as configured via the GLSL backend options.
// Note: Must be in sync with the state struct in generate_glsl_switch_func and the code generated
//       by the MDL SDK (see dumped code when enabling DUMP_GLSL in example_execution_glsl.cpp).
struct State {
    vec3 normal;
    vec3 geom_normal;
    vec3 position;
    float animation_time;
    vec3 text_coords[1];
    vec3 tangent_u[1];
    vec3 tangent_v[1];
    int ro_data_segment_offset;
    mat4 world_to_object;
    mat4 object_to_world;
    int object_id;
    float meters_per_scene_unit;
    int arg_block_offset;
};

//
// The prototypes of the functions generated in our generate_glsl_switch_func() function.
//

// Return the number of available MDL material subexpressions.
int get_mdl_num_mat_subexprs();

// Return the result of the MDL material subexpression given by the id.
vec3 mdl_mat_subexpr(int id, State state);


#if __VERSION__ < 400
int bitCount(uint x)
{
    x = x - ((x >> 1u) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
    x = (x + (x >> 4u)) & 0x0F0F0F0Fu;
    x = x + (x >> 8u);
    x = x + (x >> 16u);
    return int(x & 0x0000003Fu);
}
#endif

// Implementation of tex::lookup_*() for a texture_2d texture.
vec4 tex_lookup_float4_2d(
    int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)
{
    if (tex == 0) return vec4(0);
    return texture(material_texture_samplers_2d[tex_start + tex - 1], coord);
}


// Implementation of tex::texel_*() for a texture_2d texture.
vec4 tex_texel_2d(int tex, ivec2 coord, ivec2 uv_tile)
{
    if (tex == 0) return vec4(0);
    return texelFetch(material_texture_samplers_2d[tex_start + tex - 1], coord, 0);
}


// The fragment shader main function evaluating the MDL sub-expression.
void main() {
    // Set number of materials to use according to selected pattern
    uint num_materials = uint(bitCount(uint(material_pattern)));

    // Assign materials in a checkerboard pattern
    int material_index =
        int(
            (
                uint(texture_coordinate[0].x * 4) ^
                uint(texture_coordinate[0].y * 4)
            ) % num_materials);

    // Change material index according to selected pattern
    switch (material_pattern)
    {
        case 2: material_index = 1; break;
        case 4: material_index = 2; break;
        case 5: if (material_index == 1) material_index = 2; break;
        case 6: material_index += 1; break;
    }
    if (material_index > get_mdl_num_mat_subexprs())
        material_index = get_mdl_num_mat_subexprs();

    // Set up texture access for the chosen material
    tex_start = material_texture_starts[material_index];
    
    // Set MDL material state for state functions in "field" mode
    State state = State(
        /*normal=*/                 vec3(0.0, 0.0, 1.0),
        /*geometry_normal=*/        vec3(0.0, 0.0, 1.0),
        /*position=*/               vPosition,
        /*animation_time=*/         animation_time,
        /*text_coords=*/            texture_coordinate,
        /*texture_tangent_u=*/      vec3[1](vec3(1.0, 0.0, 0.0)),
        /*texture_tangent_v=*/      vec3[1](vec3(0.0, 1.0, 0.0)),
        /*ro_data_segment_offset=*/ 0,
        /*world_to_object=*/        mat4(1.0),
        /*object_to_world=*/        mat4(1.0),
        /*object_id=*/              0,
        /*meters_per_scene_unit=*/  1.0,
        /*arg_block_offset=*/       0
    );

    // Evaluate material sub-expression
    vec3 res = mdl_mat_subexpr(material_index, state);

    // Apply gamma correction and write to output variable
    FragColor = pow(vec4(res, 1.0), vec4(1.0 / 2.2));
}
