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
#version 330 core

// varying inputs
in vec3 world_position;

// uniforms
uniform sampler2D env_tex;
uniform float exposure_scale = 1.0f;

// outputs
out vec4 FragColor;

const float ONE_OVER_PI = 0.3183099;

vec2 get_spherical_uv(vec3 v)
{
    float gamma = asin(v.y);
    float theta = atan(v.z, v.x);

    return vec2(theta * ONE_OVER_PI * 0.5, gamma * ONE_OVER_PI) + 0.5;
}

// A simple Reinhard tonemapper.
vec3 display(vec3 val, float tonemap_scale)
{
    val *= tonemap_scale;
    float burn_out = 0.1;
    val.x *= (1.0 + val.x * burn_out) / (1.0 + val.x);
    val.y *= (1.0 + val.y * burn_out) / (1.0 + val.y);
    val.z *= (1.0 + val.z * burn_out) / (1.0 + val.z);
    
    float gamma = 1.0/2.2;
    float r = min(pow(max(val.x, 0.0), gamma), 1.0);
    float g = min(pow(max(val.y, 0.0), gamma), 1.0);
    float b = min(pow(max(val.z, 0.0), gamma), 1.0);
    
    return vec3(r, g, b);
}


void main() {

    vec2 uv = get_spherical_uv(normalize(world_position));
    vec3 color = texture(env_tex, uv).rgb;
    FragColor = vec4(display(color, exposure_scale), 1.0);
}
