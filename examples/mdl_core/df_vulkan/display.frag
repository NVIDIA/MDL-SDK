/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/df_vulkan/display.frag

#version 450

layout(location = 0) out vec4 FragColor;

layout(rgba32f, set = 0, binding = 0) uniform readonly restrict image2D uBeautyBuffer;
layout(rgba32f, set = 0, binding = 1) uniform readonly restrict image2D uAuxAlbedoDiffuseBuffer;
layout(rgba32f, set = 0, binding = 2) uniform readonly restrict image2D uAuxAlbedoGlossyBuffer;
layout(rgba32f, set = 0, binding = 3) uniform readonly restrict image2D uAuxNormalBuffer;
layout(rgba32f, set = 0, binding = 4) uniform readonly restrict image2D uAuxRoughnessBuffer;

layout(push_constant) uniform UserData
{
    uint uBufferIndex;
};

void main()
{
    ivec2 uv = ivec2(gl_FragCoord.xy);

    // Flip image because Vulkan uses the bottom-left corner as the origin,
    // but the rendering code assumed the origin to be the top-left corner.
    uv.y = imageSize(uBeautyBuffer).y - uv.y - 1;

    vec3 color;
    switch (uBufferIndex)
    {
    case 1:
        color = imageLoad(uAuxAlbedoDiffuseBuffer, uv).xyz + imageLoad(uAuxAlbedoGlossyBuffer, uv).xyz;
        break;

    case 2:
        color = imageLoad(uAuxAlbedoDiffuseBuffer, uv).xyz;
        break;

    case 3:
        color = imageLoad(uAuxAlbedoGlossyBuffer, uv).xyz;
        break;

    case 4:
        color = imageLoad(uAuxNormalBuffer, uv).xyz;
        if (dot(color, color) > 0.01)
            color = normalize(color) * 0.5 + 0.5;
        break;

    case 5:
        color = imageLoad(uAuxRoughnessBuffer, uv).xyz;
        break;

    default:
        color = imageLoad(uBeautyBuffer, uv).xyz;
        break;
    }

    // Apply reinhard tone mapping
    const float burn_out = 0.1;
    color *= (vec3(1.0) + color * burn_out) / (vec3(1.0) + color);

    // Apply gamma correction
    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
