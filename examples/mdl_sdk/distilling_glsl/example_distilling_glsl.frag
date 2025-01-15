/******************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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

// This shader is based on the Unreal 4 PBR shading model as described in
//
// "Real Shading in Unreal Engine 4" by Brian Karis
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

const float ONE_OVER_PI = 0.3183099;

float rought_to_gloss(float roughness)
{
    return 1.0 - roughness;
}

vec3 fresnel_schlick_roughness(vec3 F0, float cos_theta, float roughness)
{
    return F0 + (max(vec3(rought_to_gloss(roughness)), F0) - F0) * pow(1.0 - cos_theta, 5.0);
}

//Varying inputs
in vec3 texture_coordinate;
in vec3 world_position;
in vec3 world_normal;
in vec3 world_tangent;
in vec3 world_binormal;

// Uniform inputs
uniform samplerCube  irradiance_map;
uniform samplerCube  refl_map;
uniform sampler2D    brdf_lut;

uniform vec3 cam_position = vec3(5.0, 0.0, 0.0);

uniform float exposure_scale = 0.0;

// The color output variable of this fragment shader.
out vec4 FragColor;

const float MAX_REFLECTION_LOD = 4.0;

// A simple Reinhard tonemapper.
vec3 display(vec3 val, float tonemap_scale)
{
    val *= tonemap_scale;

    val = max(val, vec3(0.0f));
    
    float burn_out = 0.1;
    val.x *= (1.0 + val.x * burn_out) / (1.0 + val.x);
    val.y *= (1.0 + val.y * burn_out) / (1.0 + val.y);
    val.z *= (1.0 + val.z * burn_out) / (1.0 + val.z);

    float gamma = 1.0/2.2;
    float r = min(pow(val.x, gamma), 1.0);
    float g = min(pow(val.y, gamma), 1.0);
    float b = min(pow(val.z, gamma), 1.0);
    
    return vec3(r, g, b);
}

void main() {
    
    // Setup material state required for MDL expressions.
    State state = State(
        /* normal=*/                 world_normal,
        /* geom_normal=*/            world_normal,
        /* position=*/               world_position,
        /* animation_time=*/         0.0,
        /* text_coords[1]=*/         vec3[1](texture_coordinate),
        /* tangent_u[1]=*/           vec3[1](world_tangent),
        /* tangent_v[1]=*/           vec3[1](world_binormal),
        /*ro_data_segment_offset=*/  0,
        /* world_to_object=*/        mat4(1.0),
        /* object_to_world=*/        mat4(1.0),
        /*object_id=*/               0,
        /*meters_per_scene_unit=*/   1.0,
        /*arg_block_offset=*/        0
    );

    // get values from mdl material expressions
    vec3 clearcoat_color = clamp(get_clearcoat_color(state), vec3(0.0f), vec3(1.0f));
    float clearcoat_roughness = sqrt(clamp(get_clearcoat_roughness(state), 0.0f, 1.0f));
    float clearcoat_weight = clamp(get_clearcoat_weight(state), 0.0f, 1.0f);

    vec3 base_color = clamp(get_base_color(state), vec3(0.0f), vec3(1.0f));
    float roughness = sqrt(clamp(get_roughness(state), 0.0f, 1.0f));
    float metallic = clamp(get_metallic(state), 0.0f, 1.0f);
    float spec_weight = clamp(get_specular(state), 0.0f, 1.0f);

    vec3 V = normalize(cam_position - world_position);
    vec3 N = get_normal(state);
    vec3 R = reflect(-V, N);
    float cos_theta = max(dot(N, V), 0.0);
    
    vec3 F0 = vec3(0.04);
    vec3 clearcoat = vec3(0, 0, 0);
    if(clearcoat_weight > 0.001)
    {
        vec3 CN = get_clearcoat_normal(state);
        float cos_theta0 = max(dot(CN, V), 0.0);
        
        vec3 F_c = fresnel_schlick_roughness(F0, cos_theta0, clearcoat_roughness);
        vec3 refl_color_c = textureLod(refl_map, reflect(-V, CN), clearcoat_roughness * MAX_REFLECTION_LOD).rgb;
        vec2 brdf = texture(brdf_lut, vec2(cos_theta0, clearcoat_roughness)).rg;
        clearcoat = clearcoat_weight * refl_color_c * clearcoat_color * (F_c * brdf.x + brdf.y);
        
        clearcoat_weight *= F_c.x;
    }
    
    F0 = mix(F0, base_color, metallic);
    spec_weight = mix(spec_weight, 1.0, metallic);

    vec3 F = fresnel_schlick_roughness(F0, cos_theta, roughness);
    vec3 kS = F * spec_weight;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - vec3(metallic);
    kD = clamp(kD, vec3(0.0f), vec3(1.0f));

    vec3 diffuse = vec3(0.0);
    vec3 black = vec3(0.0);
    if(kD != black)
    {
        vec3 irradiance = texture(irradiance_map, N).rgb;
        diffuse = kD * irradiance * base_color * ONE_OVER_PI;
    }

    vec3 refl_color = textureLod(refl_map, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdf_lut, vec2(cos_theta, roughness)).rg;
    vec3 specular = spec_weight * refl_color * (F * brdf.x + brdf.y);

    FragColor = vec4(display(clearcoat + (1.0 - clearcoat_weight) * (diffuse + specular), exposure_scale), 1);
}
